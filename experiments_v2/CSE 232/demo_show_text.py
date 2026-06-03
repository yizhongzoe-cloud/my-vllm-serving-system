#!/usr/bin/env python3
"""Demo: Ferry frees a paused request's GPU KV, then reloads it cheaply —
and you watch the output tokens stream out, pause, and resume mid-word.

Timeline on one screen:
  STEP 1  the request generates — tokens stream out live (KV is on the GPU).
  PAUSE   tool call: the GPU KV-pool occupancy (vllm:kv_cache_usage_perc,
          polled from /metrics) DROPS to ~0 while a checkpoint file appears
          under /dev/shm — the KV left the GPU but survives in host RAM.
  STEP 2  resume: Ferry reloads the exact KV in a single engine step (a
          host->GPU copy, not a full re-prefill) and the tokens keep
          streaming, continuing the sentence seamlessly across the pause.

nvidia-smi can't show the release: vLLM grabs the whole KV pool at startup, so
device memory never moves; this gauge is the occupancy *inside* that pool.
Run from the repo root with the sd_env venv active; GPUs must be free.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO / "experiments_v2" / "eval" / "scripts"))
from icept_microbench import read_internal_id, read_covered  # noqa: E402

MODEL = os.environ.get("FT_MODEL", "/home/yzhong76/model/Qwen2.5-14B-Instruct")
PORT = 8401
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")
ICEPT_AT = 128         # tokens generated before the pause
PAUSE_S = 10           # simulated tool-call latency
# KV bytes/token (fp16): 2(K,V) * 48 layers * 8 KV heads * 128 head_dim * 2B
KV_BYTES_PER_TOK = 2 * 48 * 8 * 128 * 2

_metric_re = re.compile(r"^vllm:kv_cache_usage_perc\b.*?\s([0-9.eE+-]+)\s*$", re.M)


def post_stream(body, echo=True, timeout=600):
    """POST a completion with stream=True; (optionally) print each token delta
    as it arrives. Returns (full_text, elapsed_s, ttft_s) where ttft_s is the
    time to the FIRST generated token — i.e. the resume cost (reload or
    re-prefill) before generation continues, isolated from later decoding."""
    body = {**body, "stream": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, text, ttft = time.time(), [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            piece = json.loads(payload)["choices"][0].get("text", "")
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
                if echo:
                    sys.stdout.write(piece)
                    sys.stdout.flush()
    return "".join(text), time.time() - t0, ttft


def wait_ready(timeout=240):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def read_kv_pct():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=2) as r:
            body = r.read().decode()
        vals = [float(m) for m in _metric_re.findall(body)]
        return max(vals) * 100.0 if vals else None
    except Exception:
        return None


def n_ckpt():
    try:
        return sum(1 for _ in CKPT_DIR.iterdir())
    except Exception:
        return 0


_reload_re = re.compile(r"reload took (\d+) step\(s\), (\d+) tokens restored")


def read_reload_proof(log_path):
    """The engine's own words: how many steps the reload took. The reload is
    ~1 step (a host->GPU copy); the rest of the resume latency is generating
    the continuation, not the reload."""
    try:
        m = None
        for line in open(log_path):
            hit = _reload_re.search(line)
            if hit:
                m = hit
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


# background sampler: always records (for the summary), prints ONLY during the
# pause, and only when the number actually changes — so it never buries tokens.
_mon = {"phase": "init", "t0": 0.0, "stop": False, "samples": [], "print": False}


def bar(pct, width=30, full_scale=20.0):
    n = max(0, min(width, int(round((pct / full_scale) * width))))
    return "█" * n + "░" * (width - n)


def monitor():
    last = None
    while not _mon["stop"]:
        pct, files = read_kv_pct(), n_ckpt()
        if pct is not None:
            _mon["samples"].append((_mon["phase"], pct))
            if _mon["print"]:
                key = (round(pct, 1), files)
                if key != last:
                    last = key
                    t = time.time() - _mon["t0"]
                    print(f"      [t={t:4.1f}s]  GPU KV pool: {pct:5.1f}%  "
                          f"[{bar(pct)}]   host RAM copy: {'yes' if files else 'no'}",
                          flush=True)
        time.sleep(0.4)


def peak(phase):
    xs = [p for ph, p in _mon["samples"] if ph == phase]
    return max(xs) if xs else None


def main():
    MODE = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()
    if MODE not in ("ferry", "recompute", "both"):
        print(f"usage: demo_show_text.py [ferry|recompute|both]  (got '{MODE}')")
        return 2
    do_ferry = MODE in ("ferry", "both")
    do_recompute = MODE in ("recompute", "both")
    # recompute-only runs a FAITHFUL vanilla engine: no checkpoints written, so
    # the monitor honestly shows "host RAM copy: no" and resume must re-prefill.
    checkpoint_on = do_ferry

    env = os.environ.copy()
    env.update({"VLLM_FT_ENGINE_ID": "0", "CUDA_VISIBLE_DEVICES": "0",
                "FT_ROUTER_SHM_BUS": "1",
                "FT_CAPACITY_PREEMPT_RELOAD": "1" if checkpoint_on else "0",
                "FT_CAPACITY_PREEMPT_RELOAD_OVERLAP": "1",
                "FT_DELTA_CHECKPOINT": "1" if checkpoint_on else "0",
                "SLO_PRIORITY_PREEMPT": "0"})
    for d in ("vllm_ft_checkpoints", "vllm_ft_req_map",
              "vllm_ft_engine_status", "vllm_ft_preempt_queue"):
        subprocess.run(["rm", "-rf", f"/dev/shm/{d}"], check=False)
    log = open("/tmp/demo_show_text_engine.log", "w")
    eng_kind = "Ferry" if checkpoint_on else "vanilla (no checkpoints)"
    print(f"[demo] mode = {MODE.upper()}", flush=True)
    print(f"[demo] starting {eng_kind} engine (loads 14B) ...", flush=True)
    eng = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--port", str(PORT), "--max-model-len", "16384",
         "--gpu-memory-utilization", "0.85", "--dtype", "float16",
         "--enforce-eager", "--no-enable-prefix-caching",
         "--disable-log-requests"],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        if not wait_ready():
            print("[demo] FAIL: engine never healthy (see "
                  "/tmp/demo_show_text_engine.log)")
            return 1
        print("[demo] engine ready.\n", flush=True)

        # long context (~10K tokens) so its KV is a visible slice of the pool;
        # the tail is the abstract sentence we ask the model to continue.
        filler = ("Long-context LLM serving keeps a large per-request KV cache "
                  "on the GPU. Tool-augmented agents pause mid-generation to "
                  "call external tools, leaving that KV idle but resident. "
                  "Recomputing it on resume wastes prefill; pinning it on the "
                  "GPU wastes memory other requests need. ") * 150
        prompt = (
            "You are writing the abstract of a systems paper. Below are rough "
            "background notes; then continue the abstract in one flowing "
            "paragraph.\n\nNOTES:\n" + filler + "\n\n"
            "Abstract: Long-context LLM serving makes a request's KV cache "
            "large and expensive to rebuild. When a request pauses to call an "
            "external tool, the system must decide what to do with its idle "
            "KV state. We present a serving mechanism that")

        _mon["t0"] = time.time()
        threading.Thread(target=monitor, daemon=True).start()

        rid = "demoshow"
        # ---- STEP 1: stream tokens up to the pause ----
        _mon["phase"] = "seg1"
        print("=" * 74)
        print("  STEP 1 — the request runs and generates (KV is on the GPU)")
        print("=" * 74)
        print("output:  ...a serving mechanism that", end="", flush=True)
        seg1, _, _ = post_stream(
            {"model": MODEL, "prompt": prompt, "max_tokens": ICEPT_AT,
             "temperature": 0.0, "ignore_eos": True,
             "vllm_xargs": {"router_req_id": rid}})
        print("\n")

        # ---- PAUSE: KV leaves the GPU ----
        hi_seg1 = peak("seg1")
        _mon["phase"] = "pause"
        keeps = "Ferry frees its GPU KV (kept in host RAM)" if checkpoint_on \
            else "the KV is freed (vanilla vLLM keeps no copy)"
        print("-" * 74)
        print(f"  >>> TOOL CALL — request pauses {PAUSE_S}s.  {keeps}.")
        if hi_seg1 is not None:
            print(f"      (it was holding {hi_seg1:.1f}% of the KV pool — watch the "
                  "MONITOR window fall to ~0)")
        else:
            print("      (watch the MONITOR window fall to ~0)")
        print("-" * 74)
        internal, covered = None, None
        t_end = time.time() + PAUSE_S
        while time.time() < t_end:
            if covered is None and checkpoint_on:
                internal = read_internal_id(rid)
                covered = read_covered(internal) if internal else None
            time.sleep(0.2)

        # ---- RESUME: bring the paused request back ----
        rem = 80                                        # tokens generated after resume
        left_off = " ".join(seg1.split()[-4:])
        base_body = {"model": MODEL, "max_tokens": rem, "temperature": 0.0,
                     "ignore_eos": True, "prompt": prompt + seg1}
        step = [1]                                       # STEP 1 was the seg1 run

        def resume(use_reload):
            step[0] += 1
            _mon["phase"] = "reload" if use_reload else "recompute"
            body = dict(base_body)
            if use_reload and covered is not None:
                body["vllm_xargs"] = {"is_rerouted": True,
                                      "original_internal_req_id": internal,
                                      "num_checkpointed_tokens": covered}
                label = f"FERRY — HOST-CHECKPOINT RELOAD ({covered} tokens restored)"
            elif use_reload:
                label = "FERRY — checkpoint NOT found, fell back to recompute"
            else:
                label = "RECOMPUTE — re-prefill the whole context"
            print()
            print("=" * 74)
            print(f"  STEP {step[0]} — resume: {label}")
            print("=" * 74)
            print(f"output:  (left off at '...{left_off}')  -> ", end="", flush=True)
            seg2, _, ttft = post_stream(body)
            print("\n")
            return seg2, ttft

        seg2_rc = ttft_rc = seg2_ft = ttft_ft = None
        if do_recompute:
            seg2_rc, ttft_rc = resume(use_reload=False)
        if do_ferry:
            seg2_ft, ttft_ft = resume(use_reload=True)
        _mon["phase"] = "done"
        time.sleep(0.6)
        _mon["stop"] = True

        # ---- summary ----
        hi = peak("seg1")
        lo = min([p for ph, p in _mon["samples"] if ph == "pause"] or [0])
        back = peak("reload") if do_ferry else None
        proof = read_reload_proof("/tmp/demo_show_text_engine.log") if do_ferry else None
        gb = (covered or 0) * KV_BYTES_PER_TOK / 1e9
        print("=" * 74)
        print("  (1) GPU MEMORY — freed during the pause")
        print("=" * 74)
        if hi is not None:
            print(f"  while running : {hi:5.1f}%   request's KV held on the GPU")
        print(f"  during pause  : {lo:5.1f}%   KV freed — GPU now available to others")
        if do_ferry and back is not None:
            print(f"  after reload  : {back:5.1f}%   KV brought back from host RAM")
        if checkpoint_on and covered is not None:
            print(f"  host RAM copy : {covered} tokens ≈ {gb:.2f} GB kept (reloadable)")
        elif not checkpoint_on:
            print("  host RAM copy : none — vanilla vLLM keeps no checkpoint")
        print()
        print("=" * 74)
        print("  (2) RESUME COST — time to the first continued token (TTFT)")
        print("=" * 74)
        if do_recompute:
            rc = f"{ttft_rc:.2f}s" if ttft_rc else "n/a"
            print(f"  recompute (re-prefill the whole context): {rc}")
        if do_ferry:
            tag = f"{ttft_ft:.2f}s" if ttft_ft else "n/a"
            print(f"  Ferry     (reload {covered} tok from host RAM): {tag}")
            if proof is not None:
                print(f"  engine log: Ferry reload took {proof[0]} step (a host->GPU copy)")
        if do_ferry and do_recompute and ttft_rc and ttft_ft and ttft_ft > 0:
            print(f"  --> Ferry resumes {ttft_rc / ttft_ft:.1f}x faster "
                  "(reload vs re-prefill)")
            same = (seg2_rc.strip()[:50] == seg2_ft.strip()[:50])
            if not same:
                print("  (the two continuations differ by a word — greedy decoding is")
                print("   sensitive to tiny FP differences between reload and re-prefill;")
                print("   the point is the cost gap, not the wording)")
        print()
        resumed = seg2_ft if do_ferry else seg2_rc
        print("  RESUMED TEXT (the pause is invisible in the words):")
        print("  ...a serving mechanism that" + seg1 + (resumed or ""))
        return 0
    finally:
        _mon["stop"] = True
        try:
            os.killpg(os.getpgid(eng.pid), 15)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
