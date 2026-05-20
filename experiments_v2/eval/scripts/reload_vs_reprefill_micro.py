"""Microbenchmark: V3 reload time vs full reprefill, across context lengths.

Phase 1 (reprefill cost): single vllm_fcfs engine, send L-token prompt
with max_tokens=1; the TTFT = full prefill time (the cost a no-checkpoint
system pays on every preempt-and-resume).

Phase 2 (V3 reload cost): dual-engine ours_no_picker setup with router.
Send single long-prompt request through the router, wait until ~100
output tokens have been generated (host-RAM checkpoint built), then
SIGKILL the engine hosting it. Router detects death, reroutes to peer
engine, peer engine runs the V3 reload state machine and emits its
first post-reroute token. reload_time = first_post_reroute_token_ts
- kill_ts.

Context lengths: 8K / 16K / 24K / 32K. 3 trials per length.

Output: experiments_v2/eval/results/a6000/reload_vs_reprefill_micro.json
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
MODEL = "/home/yzhong76/model/Qwen2.5-14B-Instruct"
EVAL_RESULTS_DIR = REPO_ROOT / "experiments_v2/eval/results/a6000"

CONTEXT_LENGTHS = [8192, 16384, 24576]  # 32K w/ KV may not fit dual setup
TRIALS_PER_LENGTH = 3
MAX_OUTPUT = 200
TOKENS_BEFORE_KILL = 100

# Phase 1: vllm_fcfs single engine
SINGLE_PORT = 8500

# Phase 2: dual engine + router
ENGINE_0_PORT = 8501
ENGINE_1_PORT = 8502
ROUTER_PORT = 8500  # router exposes same client-facing port


def cleanup_state() -> None:
    for p in ("vllm_ft_preempt_queue", "vllm_ft_engine_status",
              "vllm_ft_req_map", "vllm_ft_checkpoints"):
        os.system(f"rm -rf /dev/shm/{p} 2>/dev/null")
    os.system("pkill -KILL -f 'vllm.entrypoints.openai.api_server' "
              "2>/dev/null")
    os.system("pkill -KILL -f 'experiments_v2.router.router' 2>/dev/null")
    os.system("pkill -KILL -f 'EngineCore' 2>/dev/null")
    time.sleep(4)


def make_prompt_tokens(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, 100000) for _ in range(length)]


def start_single_engine(log_path: Path, max_model_len: int):
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["FT_ROUTER_SHM_BUS"] = "0"
    env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
    env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
    env["FT_DELTA_CHECKPOINT"] = "0"
    env["SLO_PRIORITY_PREEMPT"] = "0"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--port", str(SINGLE_PORT),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.9",
        "--dtype", "float16",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def start_dual_engine(engine_id: int, port: int, log_path: Path,
                      max_model_len: int):
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = str(engine_id)
    env["FT_ROUTER_SHM_BUS"] = "1"
    # ours_no_picker: checkpoint + reload on, picker off.
    env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
    env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
    env["FT_DELTA_CHECKPOINT"] = "1"
    env["SLO_PRIORITY_PREEMPT"] = "0"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.85",
        "--dtype", "float16",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def start_router(log_path: Path):
    env = os.environ.copy()
    env["FT_ROUTER_POLICY"] = "round_robin"  # deterministic placement
    cmd = [
        sys.executable, "-m", "experiments_v2.router.router",
        "--port", str(ROUTER_PORT),
        "--engines",
        f"0=http://127.0.0.1:{ENGINE_0_PORT}",
        f"1=http://127.0.0.1:{ENGINE_1_PORT}",
    ]
    log_f = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)


def shutdown(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def wait_url_ready(port: int, timeout_s: int = 180,
                   path: str = "/v1/models") -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}{path}"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def measure_prefill(prompt_tokens, port: int) -> float:
    """Send one request with max_tokens=1, return TTFT (ms)."""
    payload = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{port}/v1/completions"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw_line in r:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
                if obj.get("choices", [{}])[0].get("text", ""):
                    ttft = time.perf_counter() - t0
                    break
            except Exception:
                pass
    return ttft * 1000 if ttft else None


def phase1_reprefill(log_dir: Path) -> dict:
    print("[phase1] reprefill timing on vllm_fcfs single engine")
    cleanup_state()
    log_path = log_dir / "phase1_engine.log"
    proc = start_single_engine(log_path, 32768)
    if not wait_url_ready(SINGLE_PORT, 240):
        shutdown(proc)
        cleanup_state()
        raise RuntimeError("phase1 engine did not start")
    print("[phase1] engine ready")
    results = {}
    try:
        for L in CONTEXT_LENGTHS:
            trials = []
            for trial in range(TRIALS_PER_LENGTH):
                seed = trial * 9931 + L
                prompt = make_prompt_tokens(seed, L)
                print(f"[phase1] L={L} trial={trial}, prefilling...",
                      flush=True)
                t = measure_prefill(prompt, SINGLE_PORT)
                trials.append(t)
                print(f"  prefill={t:.0f}ms", flush=True)
            results[L] = trials
    finally:
        shutdown(proc)
        cleanup_state()
    return results


def find_engine_pid(log_path: Path) -> int | None:
    """Read engine log; extract PID of EngineCore subprocess."""
    try:
        with open(log_path) as f:
            for line in f:
                m = re.search(r"EngineCore_DP0 pid=(\d+)", line)
                if m:
                    return int(m.group(1))
    except FileNotFoundError:
        return None
    return None


def stream_until_n_tokens(prompt_tokens, n: int, port: int) -> tuple:
    """Stream a request; return (request_id_extracted, generated_count,
    last_t)."""
    payload = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": MAX_OUTPUT,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    data = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{port}/v1/completions"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})

    return req


def measure_reload(prompt_tokens, e0_log: Path, e1_log: Path,
                   expected_prefill_ms: float):
    """Send non-streaming request via background thread, wait roughly
    until ~100 tokens generated (prefill + 100 * 50ms decode), then
    SIGKILL engine 0. Parse engine 1 log for first_post_reroute_token."""
    payload = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": MAX_OUTPUT,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }
    data = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{ROUTER_PORT}/v1/completions"

    import threading
    send_done = [False]

    def sender():
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
        except Exception as e:
            print(f"  sender error: {e}", flush=True)
        finally:
            send_done[0] = True

    t = threading.Thread(target=sender, daemon=True)
    t.start()

    # Wait = prefill + 3 s of decode (~50-60 tokens, enough to publish a
    # few KV blocks). Must stay well below total request time so SIGKILL
    # fires while the request is still in flight.
    wait_s = (expected_prefill_ms / 1000.0) + 3.0
    print(f"  waiting {wait_s:.1f}s for prefill + ~50 decode tokens...",
          flush=True)
    time.sleep(wait_s)

    if send_done[0]:
        print("  WARN: request already completed before kill", flush=True)
        return None

    # SIGKILL engine 0
    pid = find_engine_pid(e0_log)
    if pid is None:
        print("  no engine 0 pid in log", flush=True)
        return None
    kill_ts = time.time()
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    print(f"  SIGKILL e0 pid={pid} at kill_ts={kill_ts:.3f}", flush=True)

    # Wait for reroute + reload + first new token on engine 1
    time.sleep(15)
    t.join(timeout=30)

    # Parse engine 1 log for first_post_reroute_token timestamp
    first_post_reroute_ts = None
    try:
        with open(e1_log) as f:
            for line in f:
                m = re.search(
                    r"first_post_reroute_token\s+req=\S+\s+ts=([\d.]+)",
                    line)
                if m:
                    first_post_reroute_ts = float(m.group(1))
                    break
    except FileNotFoundError:
        pass

    if first_post_reroute_ts is None:
        print("  WARN: no first_post_reroute_token in e1 log", flush=True)
        return None

    reload_time_ms = (first_post_reroute_ts - kill_ts) * 1000.0
    print(f"  reload_time={reload_time_ms:.0f}ms", flush=True)
    return reload_time_ms


def phase2_reload(log_dir: Path, prefill_data: dict) -> dict:
    print("[phase2] reload timing on dual engine ours_no_picker setup")
    results = {}
    for L in CONTEXT_LENGTHS:
        # Median expected prefill so we know how long to wait before kill
        import numpy as np
        rp = [t for t in prefill_data.get(L, []) if t is not None]
        expected_prefill_ms = float(np.median(rp)) if rp else 5000.0
        trials = []
        for trial in range(TRIALS_PER_LENGTH):
            print(f"[phase2] L={L} trial={trial}, fresh setup...", flush=True)
            cleanup_state()
            e0_log = log_dir / f"phase2_L{L}_t{trial}_e0.log"
            e1_log = log_dir / f"phase2_L{L}_t{trial}_e1.log"
            router_log = log_dir / f"phase2_L{L}_t{trial}_router.log"
            e0 = start_dual_engine(0, ENGINE_0_PORT, e0_log, 32768)
            e1 = start_dual_engine(1, ENGINE_1_PORT, e1_log, 32768)
            ok0 = wait_url_ready(ENGINE_0_PORT, 240)
            ok1 = wait_url_ready(ENGINE_1_PORT, 240)
            if not (ok0 and ok1):
                shutdown(e0); shutdown(e1)
                cleanup_state()
                print("  engines did not start", flush=True)
                continue
            router = start_router(router_log)
            if not wait_url_ready(ROUTER_PORT, 30, path="/health"):
                shutdown(e0); shutdown(e1); shutdown(router)
                cleanup_state()
                print("  router did not start", flush=True)
                continue
            seed = trial * 9931 + L + 7
            prompt = make_prompt_tokens(seed, L)
            try:
                rt = measure_reload(prompt, e0_log, e1_log,
                                    expected_prefill_ms)
                trials.append(rt)
            finally:
                shutdown(e0); shutdown(e1); shutdown(router)
                cleanup_state()
        results[L] = trials
    return results


def main():
    log_dir = EVAL_RESULTS_DIR / f"logs/reload_vs_reprefill_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] log_dir={log_dir}")

    # Reuse phase1 data if available
    p1_cache_file = EVAL_RESULTS_DIR / "reload_vs_reprefill_micro.json"
    p1 = None
    if p1_cache_file.exists():
        try:
            cached = json.load(open(p1_cache_file))
            cached_p1 = {
                int(L): cached["summary"][str(L)]["reprefill_ms_trials"]
                for L in cached.get("summary", {})
            }
            if all(len(cached_p1.get(L, [])) >= TRIALS_PER_LENGTH
                   for L in CONTEXT_LENGTHS):
                p1 = cached_p1
                print(f"[phase1] reusing cached prefill data: {p1}")
        except Exception:
            pass
    if p1 is None:
        p1 = phase1_reprefill(log_dir)
        print(f"\n[phase1 results] {p1}\n")

    p2 = phase2_reload(log_dir, p1)
    print(f"\n[phase2 results] {p2}\n")

    import numpy as np
    print("\n[summary] median across trials:")
    print(f"{'Context':<10} {'Reprefill (ms)':<18} {'V3 Reload (ms)':<18} {'Ratio':<10}")
    summary = {}
    for L in CONTEXT_LENGTHS:
        rp = [t for t in p1.get(L, []) if t is not None]
        rl = [t for t in p2.get(L, []) if t is not None]
        rp_med = float(np.median(rp)) if rp else None
        rl_med = float(np.median(rl)) if rl else None
        ratio = (rp_med / rl_med) if (rp_med and rl_med) else None
        print(f"{L:<10} {str(rp_med):<18} {str(rl_med):<18} {str(ratio):<10}")
        summary[L] = {
            "reprefill_ms_trials": rp, "reload_ms_trials": rl,
            "reprefill_ms_median": rp_med, "reload_ms_median": rl_med,
            "ratio": ratio,
        }

    out_file = EVAL_RESULTS_DIR / "reload_vs_reprefill_micro.json"
    with open(out_file, "w") as f:
        json.dump({"model": MODEL, "context_lengths": CONTEXT_LENGTHS,
                   "trials_per_length": TRIALS_PER_LENGTH,
                   "summary": summary}, f, indent=2)
    print(f"\n[bench] saved → {out_file}")


if __name__ == "__main__":
    main()
