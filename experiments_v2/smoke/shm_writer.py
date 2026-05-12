#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-GPU shm test — writer side.

Launches a vLLM engine on the current visible GPU, runs N short
generations, and waits for the publish background to flush /dev/shm.
On success, dumps a one-line JSON to stdout:

    {"req_ids": [...], "shm_dir": "...", "n_chunks": N, "n_manifests": M}

Intended to be invoked by test_cross_gpu_shm.py with
CUDA_VISIBLE_DEVICES=0 set.
"""
import json
import os
import sys
import time
from pathlib import Path

# Must be set before vllm imports so the worker sees them.
os.environ.setdefault("FT_CAPACITY_PREEMPT_RELOAD", "1")
os.environ.setdefault("FT_CAPACITY_PREEMPT_RELOAD_OVERLAP", "1")
os.environ.setdefault("FT_DELTA_CHECKPOINT", "1")

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
NUM_REQS = 3
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32
SHM_DIR = Path("/dev/shm/vllm_ft_checkpoints")


def build_prompt(req_idx: int) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The quick brown fox jumps over the lazy dog. " * 16
    ids = tok.encode(base)
    repeats = PROMPT_TOKENS // max(1, len(ids)) + 2
    text = (f"Request {req_idx}. " * 4) + (base * repeats)
    return tok.decode(tok.encode(text)[:PROMPT_TOKENS])


def main() -> int:
    from vllm import LLM, SamplingParams

    print(f"[writer] launching LLM on GPU (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})",
          file=sys.stderr, flush=True)

    llm = LLM(
        model=MODEL,
        dtype="float16",
        gpu_memory_utilization=0.50,
        max_model_len=PROMPT_TOKENS + 1024,
        max_num_seqs=8,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    prompts = [build_prompt(i) for i in range(NUM_REQS)]
    params = SamplingParams(
        max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0,
    )
    print(f"[writer] generating {NUM_REQS} requests", file=sys.stderr, flush=True)
    llm.generate(prompts, sampling_params=params)

    # Force any pending FT_BG_PUBLISH future to finish before we exit
    # (best-effort; if FT_BG_PUBLISH is OFF this is mainly a guard against
    # vllm's worker subprocess + CUDA context cleanup race with the reader
    # starting up too soon).
    time.sleep(3.0)

    if not SHM_DIR.exists():
        print(f"[writer] FAIL: {SHM_DIR} does not exist after generation",
              file=sys.stderr)
        return 1

    # The shm dir names are vllm's INTERNAL request_ids (with UUID suffix).
    # RequestOutput.request_id only exposes the short user-facing ID like
    # "0", "1", "2", which does NOT match the shm directory layout.
    # Orchestrator clears the dir before launching writer, so every dir
    # present now was written by this writer.
    req_ids = sorted(d.name for d in SHM_DIR.iterdir() if d.is_dir())

    n_chunks = len(list(SHM_DIR.rglob("chunk_*.pt")))
    n_manifests = len(list(SHM_DIR.rglob("manifest_*.json")))
    if n_chunks == 0 or n_manifests == 0 or not req_ids:
        print(f"[writer] FAIL: shm empty after generation "
              f"(chunks={n_chunks}, manifests={n_manifests}, "
              f"req_dirs={len(req_ids)})", file=sys.stderr)
        return 1

    # Single JSON line on stdout for the orchestrator.
    print(json.dumps({
        "req_ids": req_ids,
        "shm_dir": str(SHM_DIR),
        "n_chunks": n_chunks,
        "n_manifests": n_manifests,
    }), flush=True)
    print(f"[writer] wrote {n_chunks} chunks, {n_manifests} manifests for "
          f"{len(req_ids)} reqs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
