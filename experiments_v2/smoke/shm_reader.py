#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-GPU shm test — reader side.

Launches a vLLM engine on the current visible GPU (a *different* GPU
from the writer), then invokes the worker `restore_kv_blocks` RPC with
each writer-provided req_id. Because this engine never ran the original
request, its local host pool is empty — restore_kv_blocks must fall
back to /dev/shm and read what the writer's GPU 0 engine published.

Pass condition: every restore returns tokens > 0.

Invoked by test_cross_gpu_shm.py with CUDA_VISIBLE_DEVICES=1 and the
writer's req_ids as positional args.
"""
import json
import os
import sys

# Match writer env so RPC dispatchers / scheduler hooks behave the same.
os.environ.setdefault("FT_CAPACITY_PREEMPT_RELOAD", "1")
os.environ.setdefault("FT_CAPACITY_PREEMPT_RELOAD_OVERLAP", "1")

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
PROMPT_TOKENS = 4096
# Fresh target blocks on this engine's KV pool. Reader engine just started
# and has not served any request, so blocks [0..N-1] are unused.
TARGET_BLOCK_IDS = list(range(64))


def main() -> int:
    req_ids = sys.argv[1:]
    if not req_ids:
        print("[reader] FAIL: no req_ids provided", file=sys.stderr)
        return 1

    from vllm import LLM

    print(f"[reader] launching LLM on GPU (CUDA_VISIBLE_DEVICES="
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

    n_pass = 0
    n_fail = 0
    per_req: list[dict] = []
    for req_id in req_ids:
        # Each RPC returns a list (one entry per worker). With TP=1/PP=1 it
        # is a 1-element list; the value is the int tokens-restored count.
        try:
            results = llm.collective_rpc(
                "restore_kv_blocks",
                args=(req_id, TARGET_BLOCK_IDS, True),
            )
        except Exception as e:
            per_req.append({"req_id": req_id, "ok": False, "err": str(e)})
            n_fail += 1
            continue

        tokens = results[0] if results else 0
        ok = isinstance(tokens, int) and tokens > 0
        per_req.append({"req_id": req_id, "ok": ok, "tokens": tokens})
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print(json.dumps({"per_req": per_req,
                      "pass": n_pass, "fail": n_fail}), flush=True)
    print(f"[reader] {n_pass}/{len(req_ids)} reqs restored from shm",
          file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
