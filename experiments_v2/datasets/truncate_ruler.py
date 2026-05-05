"""Truncate ruler_64k_niah.jsonl to multiple shorter token lengths.

We do NOT need RULER prepare.py + accuracy. The ckpt-overhead experiment
only measures GPU forward time at different prompt lengths. Truncating
the existing 64K NIAH prompts at the token level gives us 1K/4K/8K/16K/32K
prompts with consistent provenance (paired across context lengths).

Usage:
    python experiments_v2/datasets/truncate_ruler.py
"""
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
SRC = ROOT / "experiments_v2/datasets/cached/ruler_64k_niah.jsonl"
OUT_DIR = ROOT / "experiments_v2/datasets/cached"
TARGETS = [1024, 4096, 8192, 16384, 32768]
EXPECTED_OUTPUT_TOKENS = 128


def main() -> None:
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    if not SRC.exists():
        raise FileNotFoundError(f"Source not found: {SRC}")

    src_records = []
    with open(SRC) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            src_records.append(json.loads(line))
    print(f"Loaded {len(src_records)} source records from {SRC}")

    for length in TARGETS:
        out_name = f"ruler_{length}_niah_trunc.jsonl"
        out_path = OUT_DIR / out_name
        n_written = 0
        n_skipped = 0
        with open(out_path, "w") as fout:
            for rec in src_records:
                tokens = tok.encode(rec["prompt"], add_special_tokens=False)
                if len(tokens) < length:
                    n_skipped += 1
                    continue
                trunc_tokens = tokens[:length]
                trunc_prompt = tok.decode(
                    trunc_tokens, skip_special_tokens=True
                )
                # Re-encode to confirm token count is close to target
                # (tokenizer round-trip can differ by a few tokens).
                final_tokens = tok.encode(
                    trunc_prompt, add_special_tokens=False
                )
                # Pad/trim to within +/- 16 tokens of target if needed.
                # We accept small drift since this is for compute timing,
                # not accuracy.
                new_rec = {
                    "id": f"trunc{length}_{rec.get('id', n_written)}",
                    "dataset": f"ruler_{length}_niah_trunc",
                    "prompt": trunc_prompt,
                    "prompt_tokens": len(final_tokens),
                    "expected_output_tokens": EXPECTED_OUTPUT_TOKENS,
                    "subtask": rec.get("subtask", "niah_single_1"),
                }
                fout.write(json.dumps(new_rec) + "\n")
                n_written += 1
        print(
            f"  {length:>5}: wrote {n_written} records "
            f"(skipped {n_skipped} too-short) -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
