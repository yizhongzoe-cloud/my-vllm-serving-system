"""Closed-loop concurrent microbenchmark for continuous-checkpoint
mechanism overhead at sustained high load.

N worker threads each send K requests serially; each worker waits for
its current request to complete before issuing the next. With N
workers running in parallel, there are always N requests in flight
(except at the very end), giving a sustained high-concurrency load
that does not depend on stochastic Poisson timing.

The KV pool is never overfilled (N is chosen below the engine's
max_num_seqs), so vLLM does not trigger capacity preemption. The only
runtime difference between vllm_fcfs and ckpt_only baselines is the
host-RAM checkpoint write path.

Output: experiments_v2/eval/results/a6000/ckpt_overhead_concurrent.json
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

REPO_ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
MODEL = "/home/yzhong76/model/Qwen2.5-14B-Instruct"
ENGINE_PORT = 8000
EVAL_RESULTS_DIR = REPO_ROOT / "experiments_v2/eval/results/a6000"


def start_engine(baseline: str, log_path: Path,
                 max_model_len: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["FT_ROUTER_SHM_BUS"] = "0"
    if baseline == "vllm_fcfs":
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    elif baseline == "ckpt_only":
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    else:
        raise ValueError(baseline)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(ENGINE_PORT),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.9",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[bench] launching ({baseline}) → {log_path.name}", flush=True)
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def shutdown(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def wait_ready(timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{ENGINE_PORT}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def make_prompt_tokens(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, 100000) for _ in range(length)]


def send_request(prompt_tokens, max_tokens, trial_idx, worker_id):
    payload = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    data = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{ENGINE_PORT}/v1/completions"
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    last_t = None
    num_tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw_line in r:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                obj = json.loads(line)
                txt = obj.get("choices", [{}])[0].get("text", "")
                if txt:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    num_tokens += 1
                    last_t = now
            except Exception:
                pass
    e2e = (last_t - t0) if last_t else None
    if ttft is not None and last_t is not None and num_tokens > 1:
        decode_time = (last_t - t0) - ttft
        tpot = decode_time / (num_tokens - 1)
    else:
        tpot = None
    return {
        "worker": worker_id,
        "trial": trial_idx,
        "ttft_ms": ttft * 1000 if ttft else None,
        "tpot_ms": tpot * 1000 if tpot else None,
        "e2e_ms": e2e * 1000 if e2e else None,
        "num_tokens": num_tokens,
        "start_t": t0,
    }


def cleanup_state() -> None:
    for p in ("vllm_ft_preempt_queue", "vllm_ft_engine_status",
              "vllm_ft_req_map", "vllm_ft_checkpoints"):
        os.system(f"rm -rf /dev/shm/{p} 2>/dev/null")
    os.system("pkill -KILL -f 'vllm.entrypoints.openai.api_server' "
              "2>/dev/null")
    os.system("pkill -KILL -f 'EngineCore' 2>/dev/null")
    time.sleep(4)


def worker_loop(worker_id, num_requests, prompt_length, max_tokens,
                baseline, results, results_lock):
    for trial in range(num_requests):
        seed = (worker_id * 100000 + trial * 9931 +
                (1 if baseline == "ckpt_only" else 0))
        prompt = make_prompt_tokens(seed=seed, length=prompt_length)
        print(f"[bench] {baseline} worker={worker_id} trial={trial}, sending...",
              flush=True)
        r = send_request(prompt, max_tokens, trial, worker_id)
        with results_lock:
            results.append(r)
            ttft = r["ttft_ms"]
            tpot = r["tpot_ms"]
            e2e = r["e2e_ms"]
            print(f"  w{worker_id} t{trial}: ttft={ttft:.0f}ms "
                  f"tpot={tpot:.2f}ms e2e={e2e:.0f}ms "
                  f"tokens={r['num_tokens']}", flush=True)


def run_baseline(baseline, prompt_length, max_tokens, num_workers,
                 requests_per_worker, max_model_len, log_dir):
    cleanup_state()
    log_path = log_dir / f"engine_{baseline}.log"
    proc = start_engine(baseline, log_path, max_model_len)
    if not wait_ready():
        shutdown(proc)
        cleanup_state()
        raise RuntimeError(f"{baseline} engine did not start")
    print(f"[bench] {baseline} engine ready, launching {num_workers} workers",
          flush=True)

    results = []
    results_lock = Lock()
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [
                ex.submit(worker_loop, w, requests_per_worker,
                          prompt_length, max_tokens, baseline,
                          results, results_lock)
                for w in range(num_workers)
            ]
            for f in as_completed(futs):
                f.result()
    finally:
        shutdown(proc)
        cleanup_state()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--requests-per-worker", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2,
                        help="Discard first N requests per worker as warmup")
    args = parser.parse_args()

    max_model_len = args.prompt_length + args.max_tokens + 256
    if max_model_len > 32768:
        print(f"[bench] ERROR: prompt+max_tokens > 32K window "
              f"({max_model_len})", file=sys.stderr)
        sys.exit(1)

    log_dir = EVAL_RESULTS_DIR / f"logs/ckpt_overhead_concurrent_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)

    total_per_baseline = args.num_workers * args.requests_per_worker
    print(f"[bench] prompt={args.prompt_length} max_tokens={args.max_tokens}")
    print(f"[bench] concurrency={args.num_workers} requests_per_worker={args.requests_per_worker}")
    print(f"[bench] total={total_per_baseline} req/baseline, "
          f"warmup={args.warmup} req/worker")
    print(f"[bench] log_dir={log_dir}")

    all_results = {}
    for baseline in ("vllm_fcfs", "ckpt_only"):
        all_results[baseline] = run_baseline(
            baseline, args.prompt_length, args.max_tokens,
            args.num_workers, args.requests_per_worker,
            max_model_len, log_dir,
        )

    # Summary: drop warmup trials per worker
    import numpy as np
    print(f"\n[summary] median over trials {args.warmup}.."
          f"{args.requests_per_worker - 1} per worker:")
    summary = {}
    for b in ("vllm_fcfs", "ckpt_only"):
        keep = [r for r in all_results[b] if r["trial"] >= args.warmup]
        ttft = float(np.median([r["ttft_ms"] for r in keep]))
        tpot = float(np.median([r["tpot_ms"] for r in keep]))
        e2e = float(np.median([r["e2e_ms"] for r in keep]))
        summary[b] = {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": e2e,
                      "n_samples": len(keep)}
        print(f"  {b}: TTFT={ttft:.0f}ms TPOT={tpot:.2f}ms E2E={e2e:.0f}ms "
              f"(n={len(keep)})")

    delta_ttft = summary["ckpt_only"]["ttft_ms"] - summary["vllm_fcfs"]["ttft_ms"]
    delta_tpot = summary["ckpt_only"]["tpot_ms"] - summary["vllm_fcfs"]["tpot_ms"]
    delta_e2e = summary["ckpt_only"]["e2e_ms"] - summary["vllm_fcfs"]["e2e_ms"]
    print(f"\n[mechanism overhead @ concurrency={args.num_workers}] "
          f"ckpt_only minus vllm_fcfs:")
    print(f"  TTFT delta = {delta_ttft:+.0f} ms")
    print(f"  TPOT delta = {delta_tpot:+.2f} ms")
    print(f"  E2E  delta = {delta_e2e:+.0f} ms")

    out_file = EVAL_RESULTS_DIR / "ckpt_overhead_concurrent.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": MODEL,
            "prompt_length": args.prompt_length,
            "max_tokens": args.max_tokens,
            "num_workers": args.num_workers,
            "requests_per_worker": args.requests_per_worker,
            "warmup": args.warmup,
            "per_request": all_results,
            "median_summary": summary,
            "delta_ckpt_minus_fcfs": {
                "ttft_ms": delta_ttft,
                "tpot_ms": delta_tpot,
                "e2e_ms": delta_e2e,
            },
        }, f, indent=2)
    print(f"[bench] saved → {out_file}")


if __name__ == "__main__":
    main()
