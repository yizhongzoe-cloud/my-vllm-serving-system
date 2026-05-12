#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone regression smoke for V3 capacity-preempt + shm publish/restore.

Goal: verify that the cross-engine shm publish/restore framework transplanted
in round 3 (2026-05-11) did not break the round 1/2 baseline behavior of the
V3 capacity-preempt + same-engine reload path. Single GPU, single engine.

What it does:
  1. Starts a vLLM OpenAI API server with FT_CAPACITY_PREEMPT_RELOAD=1.
  2. Fires 30 concurrent 16K-token prompts. The KV pool cannot hold all 30
     simultaneously, so capacity preempt should fire and the V3 reload state
     machine should drive each preempted req back to running.
  3. Greps the server log for "FT overlap V3:" markers (queued / done / failed).
  4. Verifies /dev/shm/vllm_ft_checkpoints/ contains manifest + chunk files
     (proves the publish step actually wrote /dev/shm, not just host pool).
  5. Greps for crash markers (device-side assert, EngineDeadError, etc).

Exit 0 = PASS, 1 = FAIL.

NOT a paper experiment. No dataset loader, no workload driver, no metric
collector. Self-contained. Update freely as code changes; do not let this
grow into an evaluation framework.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
NUM_CONCURRENT = 30
PROMPT_TOKENS = 8192   # ~0.45 GB/req KV; with 15+ in-flight will overflow the ~6 GB pool → V3 fires.
MAX_OUTPUT_TOKENS = 128  # Longer decode so KV keeps growing under sustained concurrency.
PORT = 8491
SERVER_READY_TIMEOUT_S = 240
REQUEST_TIMEOUT_S = 600
SHM_DIR = Path("/dev/shm/vllm_ft_checkpoints")
LOG_PATH = Path(__file__).parent / "smoke_server.log"


def cleanup_shm() -> None:
    shutil.rmtree(SHM_DIR, ignore_errors=True)


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
    env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
    env["FT_DELTA_CHECKPOINT"] = "1"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--max-model-len", str(PROMPT_TOKENS + 2048),
        # Force capacity preempt on A6000 48 GB / Qwen-7B:
        #  * gpu_memory_utilization 0.45  → ~5 GB KV pool after model+activation
        #  * max-num-seqs 32              → vllm dares to admit ≥15 in-flight
        #  * short prompts (4K) → vllm doesn't cap in-flight on prefill budget
        # 15 in-flight × 4K tok × 56 KB/tok ≈ 3.4 GB; close to pool → preempt.
        "--gpu-memory-utilization", "0.45",
        "--max-num-seqs", "32",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(LOG_PATH, "w")
    print(f"[smoke] launching server, log → {LOG_PATH}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def wait_ready() -> bool:
    deadline = time.time() + SERVER_READY_TIMEOUT_S
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def build_prompt(req_idx: int) -> str:
    """Build a prompt of approximately PROMPT_TOKENS tokens.

    Uses Llama tokenizer for accuracy. Prefix uniqueness avoids any accidental
    prefix-cache hit even with --no-enable-prefix-caching (defense in depth).
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The quick brown fox jumps over the lazy dog. Wikipedia says " * 16
    ids = tok.encode(base)
    repeats = PROMPT_TOKENS // max(1, len(ids)) + 2
    text = (f"Request {req_idx}. " * 4) + (base * repeats)
    trimmed_ids = tok.encode(text)[:PROMPT_TOKENS]
    return tok.decode(trimmed_ids)


def send_request(prompt: str, req_idx: int) -> dict:
    payload = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
            return {"idx": req_idx, "ok": "choices" in data,
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0)}
    except Exception as e:
        return {"idx": req_idx, "ok": False, "error": str(e)}


def fire_concurrent_load() -> list[dict]:
    print(f"[smoke] tokenizing {NUM_CONCURRENT} prompts (~{PROMPT_TOKENS} tok each)")
    prompts = [build_prompt(i) for i in range(NUM_CONCURRENT)]
    print(f"[smoke] firing {NUM_CONCURRENT} concurrent completions")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as ex:
        futs = {ex.submit(send_request, p, i): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            results.append(fut.result())
    print(f"[smoke] all requests returned in {time.time() - t0:.1f}s")
    return results


def grep_log() -> dict:
    txt = LOG_PATH.read_text(errors="replace")
    return {
        "v3_queued": txt.count("queued, waiting for"),
        "v3_done": txt.count(" done — alloc waited"),
        "v3_skipped": txt.count("reload skipped (no ckpt)"),
        "v3_alloc_failed": txt.count("get_block_ids failed")
                          + txt.count("allocate_slots crashed"),
        "v3_restore_failed": txt.count("restore RPC failed"),
        "fatal": (txt.count("device-side assert")
                  + txt.count("EngineDeadError")
                  + txt.count("CUDA error")
                  + txt.count("Fatal Python error")),
    }


def grep_shm() -> dict:
    if not SHM_DIR.exists():
        return {"chunks": 0, "manifests": 0, "latest": 0}
    return {
        "chunks": len(list(SHM_DIR.rglob("chunk_*.pt"))),
        "manifests": len(list(SHM_DIR.rglob("manifest_*.json"))),
        "latest": len(list(SHM_DIR.rglob("latest_*"))),
    }


def shutdown(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    cleanup_shm()
    proc = start_server()
    try:
        if not wait_ready():
            print("[smoke] FAIL: server never became healthy")
            return 1
        results = fire_concurrent_load()
        n_ok = sum(1 for r in results if r["ok"])
        log_stats = grep_log()
        shm_stats = grep_shm()
        print(f"  completed:        {n_ok}/{NUM_CONCURRENT}")
        print(f"  V3 queued:        {log_stats['v3_queued']}")
        print(f"  V3 done:          {log_stats['v3_done']}")
        print(f"  V3 skipped:       {log_stats['v3_skipped']}")
        print(f"  V3 alloc fail:    {log_stats['v3_alloc_failed']}")
        print(f"  V3 restore fail:  {log_stats['v3_restore_failed']}")
        print(f"  shm chunks:       {shm_stats['chunks']}")
        print(f"  shm manifests:    {shm_stats['manifests']}")
        print(f"  shm latest:       {shm_stats['latest']}")
        print(f"  fatal markers:    {log_stats['fatal']}")
        passed = (
            n_ok == NUM_CONCURRENT
            and log_stats["fatal"] == 0
            and log_stats["v3_queued"] > 0
            and shm_stats["chunks"] > 0
            and shm_stats["manifests"] > 0
            and shm_stats["latest"] > 0
        )
        if passed:
            print("[smoke] PASS")
            return 0
        print("[smoke] FAIL")
        return 1
    finally:
        shutdown(proc)
        # Leave LOG_PATH on disk for debugging; clean shm.
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
