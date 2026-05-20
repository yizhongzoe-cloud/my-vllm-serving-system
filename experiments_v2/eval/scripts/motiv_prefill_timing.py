"""Prefill timing benchmark for §2 Motivation Figure 1.

Measures wall-clock prefill latency on Qwen2.5-14B-Instruct, single A6000,
across prompt lengths 1K..64K. Each length is repeated NUM_TRIALS times
and we report median + min/max.

Output: experiments_v2/eval/results/a6000/motiv_prefill_14b.json

Usage:
    python -u experiments_v2/eval/scripts/motiv_prefill_timing.py
"""
import json
import os
import sys
import time
from pathlib import Path

# Force single-process inference
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/home/yzhong76/model/Qwen2.5-14B-Instruct"
MAX_MODEL_LEN = 65536
# Model config.json was patched (rope_scaling: yarn, factor 2, original 32K)
# to extend the position embedding to 64K. Remember to restore from
# config.json.bak after the benchmark.
PROMPT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 49152, 65536 - 256]
NUM_TRIALS = 3
OUTPUT_FILE = Path("experiments_v2/eval/results/a6000/motiv_prefill_14b.json")


def build_prompt_tokens(target_len: int) -> list[int]:
    """Build a synthetic prompt of `target_len` tokens. Use a fixed pattern
    so prefill compute is determined entirely by length, not content."""
    # Mix of common tokens to defeat any unlikely fast-path
    pattern = [128, 256, 384, 512, 640, 768, 896, 1024]
    return (pattern * ((target_len // len(pattern)) + 1))[:target_len]


def main():
    print(f"[motiv] loading {MODEL} with max_model_len={MAX_MODEL_LEN}")
    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        # Push GPU memory usage high enough to fit 64K KV cache;
        # enforce_eager frees the CUDA-graph workspace.
        gpu_memory_utilization=0.97,
        dtype="float16",
        enforce_eager=True,
        # Disable prefix caching so every trial pays full prefill cost.
        enable_prefix_caching=False,
        # Run prefill in one batch (no chunking) so wall-clock reflects
        # the O(N^2) GPU work that a reprefill would actually pay.
        enable_chunked_prefill=False,
        max_num_batched_tokens=MAX_MODEL_LEN,
    )
    print("[motiv] model loaded")

    # Sampling params: 1 output token only — we want prefill, not decode
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    results = []
    for plen in PROMPT_LENGTHS:
        trials = []
        for trial in range(NUM_TRIALS):
            tokens = build_prompt_tokens(plen)
            print(f"[motiv] prompt={plen} trial={trial}, generating...",
                  flush=True)
            t0 = time.perf_counter()
            llm.generate(
                prompts=[{"prompt_token_ids": tokens}],
                sampling_params=sp,
                use_tqdm=False,
            )
            dt = time.perf_counter() - t0
            trials.append(dt)
            print(f"        wall={dt:.3f}s", flush=True)
        results.append({
            "prompt_tokens": plen,
            "trial_seconds": trials,
            "median_s": sorted(trials)[len(trials) // 2],
            "min_s": min(trials),
            "max_s": max(trials),
        })
        # Save incremental in case of OOM at longer lengths
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"model": MODEL, "results": results}, f, indent=2)
        print(f"[motiv] wrote partial → {OUTPUT_FILE}", flush=True)

    print("[motiv] done")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
