#!/usr/bin/env python3
"""Download arxiv-summarization (Cohan et al. 2018), format prompts, cache as jsonl.

Output:
  experiments_v2/datasets/cached/arxivsumm.jsonl

Schema (matches loader.py):
  id, dataset: "arxivsumm", prompt, prompt_tokens, expected_output_tokens

Filter: keep prompt_tokens <= 30000 (Qwen2.5-7B native max 32K, leave 2K
output budget). Andes-style: same workload + similar truncation strategy.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

OUT_PATH = Path(
    "/home/yzhong76/code/my-vllm-serving-system/experiments_v2/datasets/cached/arxivsumm.jsonl"
)
TOKENIZER_PATH = "/home/yzhong76/model/Qwen2.5-7B-Instruct"

MAX_PROMPT_TOKENS = 30000   # 32K native ctx - 2K output budget
EXPECTED_OUTPUT_TOKENS = 400   # cap to bound experiment runtime

PROMPT_TEMPLATE = (
    "You are given a scientific paper. Write a concise abstract (one"
    " paragraph, around 200 words) that captures the paper's contributions,"
    " methodology, and key findings.\n\nPaper:\n{article}\n\nAbstract:"
)


def main() -> None:
    print(f"[arxivsumm] loading tokenizer from {TOKENIZER_PATH}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("[arxivsumm] downloading test split")
    ds = load_dataset("ccdv/arxiv-summarization", split="test")
    print(f"[arxivsumm] raw test split: {len(ds)} records")

    out_records: list[dict] = []
    dropped_long = 0
    dropped_short = 0
    for i, row in enumerate(ds):
        article = row.get("article", "").strip()
        if not article:
            continue
        prompt = PROMPT_TEMPLATE.format(article=article)
        n_tokens = len(tok.encode(prompt))
        if n_tokens > MAX_PROMPT_TOKENS:
            dropped_long += 1
            continue
        # Drop trivially short ones (<1K) so the workload stays
        # long-context-like.
        if n_tokens < 1024:
            dropped_short += 1
            continue
        out_records.append({
            "id": f"arxivsumm-{i:05d}",
            "dataset": "arxivsumm",
            "prompt": prompt,
            "prompt_tokens": n_tokens,
            "expected_output_tokens": EXPECTED_OUTPUT_TOKENS,
        })

    rng = random.Random(42)
    rng.shuffle(out_records)

    print(f"[arxivsumm] kept: {len(out_records)} "
          f"(dropped {dropped_long} >30K, {dropped_short} <1K)")

    lens = sorted(r["prompt_tokens"] for r in out_records)
    n = len(lens)
    def pct(p: float) -> int:
        return lens[min(n - 1, int(n * p))]
    print(f"[arxivsumm] prompt_tokens: min={lens[0]} P25={pct(0.25)} "
          f"P50={pct(0.50)} mean={sum(lens)//n} P75={pct(0.75)} "
          f"P95={pct(0.95)} max={lens[-1]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")
    print(f"[arxivsumm] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
