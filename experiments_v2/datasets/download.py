#!/usr/bin/env python3
"""Download and preprocess datasets for FT serving experiments.

Outputs unified JSONL files to experiments_v2/datasets/cached/.

Each record:
    {"id": "sharegpt-00001", "dataset": "sharegpt",
     "prompt": "...", "prompt_tokens": 235, "expected_output_tokens": 180}

Usage:
    python experiments_v2/datasets/download.py [--tokenizer MODEL] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Tokenizer (loaded once, shared across all datasets)
# ---------------------------------------------------------------------------

_TOKENIZER = None
_DEFAULT_TOKENIZER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _get_tokenizer(model_name: str | None = None):
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    model_name = model_name or _DEFAULT_TOKENIZER_MODEL
    logger.info("Loading tokenizer: %s", model_name)
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def _count_tokens(text: str, tokenizer=None) -> int:
    tokenizer = tokenizer or _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

def download_sharegpt(
    output_path: str,
    max_samples: int = 5000,
    max_total_tokens: int = 4096,
    min_turns: int = 2,
    seed: int = 42,
) -> None:
    """Download and preprocess ShareGPT conversations.

    Multi-turn handling: concatenate full conversation history up to the last
    user turn as prompt; the last assistant turn's token count becomes
    expected_output_tokens.
    """
    logger.info("=== ShareGPT ===")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        sys.exit(1)

    logger.info("Downloading ShareGPT (this may take a while)...")
    # Try the common ShareGPT source
    try:
        ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
        )
    except Exception:
        # Fallback: ShareGPT with different repo name
        logger.info("Primary source failed, trying alternative...")
        try:
            ds = load_dataset(
                "theblackcat102/sharegpt-english",
                split="train",
            )
        except Exception:
            logger.error(
                "Could not download ShareGPT. Please download manually and "
                "place a JSONL file at the output path."
            )
            _save_jsonl([], output_path)
            return

    tokenizer = _get_tokenizer()
    rng = np.random.RandomState(seed)

    records: list[dict] = []
    indices = rng.permutation(len(ds))

    for idx in indices:
        if len(records) >= max_samples:
            break

        row = ds[int(idx)]
        conversations = row.get("conversations") or row.get("conversation") or []
        if not conversations:
            continue

        # Need at least min_turns pairs (user + assistant)
        # Filter to user/assistant roles
        turns = []
        for turn in conversations:
            role = turn.get("from") or turn.get("role") or turn.get("user", "")
            value = turn.get("value") or turn.get("content") or turn.get("text", "")
            if role in ("human", "user"):
                turns.append(("user", value))
            elif role in ("gpt", "assistant"):
                turns.append(("assistant", value))

        if len(turns) < min_turns * 2:
            continue

        # Find last user-assistant pair
        last_assistant_idx = None
        for i in range(len(turns) - 1, -1, -1):
            if turns[i][0] == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is None or last_assistant_idx == 0:
            continue

        # Prompt = everything up to (not including) last assistant turn
        prompt_parts = []
        for role, text in turns[:last_assistant_idx]:
            prompt_parts.append(f"{role}: {text}")
        prompt = "\n\n".join(prompt_parts)

        # Output = last assistant turn
        output_text = turns[last_assistant_idx][1]

        prompt_tokens = _count_tokens(prompt, tokenizer)
        output_tokens = _count_tokens(output_text, tokenizer)

        if prompt_tokens + output_tokens > max_total_tokens:
            continue
        if prompt_tokens < 10 or output_tokens < 5:
            continue

        records.append({
            "id": f"sharegpt-{len(records):05d}",
            "dataset": "sharegpt",
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "expected_output_tokens": output_tokens,
        })

    _save_jsonl(records, output_path)
    _print_stats("ShareGPT", records)


# ---------------------------------------------------------------------------
# CNN/DailyMail
# ---------------------------------------------------------------------------

def download_cnndm(
    output_path: str,
    max_samples: int = 3000,
    max_total_tokens: int = 4096,
    seed: int = 42,
) -> None:
    """Download and preprocess CNN/DailyMail."""
    logger.info("=== CNN/DailyMail ===")

    from datasets import load_dataset

    logger.info("Downloading CNN/DailyMail...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

    tokenizer = _get_tokenizer()
    rng = np.random.RandomState(seed)

    records: list[dict] = []
    indices = rng.permutation(len(ds))

    for idx in indices:
        if len(records) >= max_samples:
            break

        row = ds[int(idx)]
        article = row["article"]
        highlights = row["highlights"]

        prompt = article + "\n\nSummarize the above article in one paragraph."
        prompt_tokens = _count_tokens(prompt, tokenizer)
        output_tokens = _count_tokens(highlights, tokenizer)

        if prompt_tokens + output_tokens > max_total_tokens:
            continue
        if prompt_tokens < 10 or output_tokens < 5:
            continue

        records.append({
            "id": f"cnndm-{len(records):05d}",
            "dataset": "cnndm",
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "expected_output_tokens": output_tokens,
        })

    _save_jsonl(records, output_path)
    _print_stats("CNN/DailyMail", records)


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

def download_alpaca(
    output_path: str,
    seed: int = 42,
) -> None:
    """Download and preprocess Alpaca."""
    logger.info("=== Alpaca ===")

    from datasets import load_dataset

    logger.info("Downloading Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    tokenizer = _get_tokenizer()

    records: list[dict] = []
    for i, row in enumerate(ds):
        instruction = row["instruction"]
        inp = row.get("input", "")
        output_text = row["output"]

        if inp:
            prompt = f"{instruction}\n\n{inp}"
        else:
            prompt = instruction

        prompt_tokens = _count_tokens(prompt, tokenizer)
        output_tokens = _count_tokens(output_text, tokenizer)

        if prompt_tokens < 3 or output_tokens < 3:
            continue

        records.append({
            "id": f"alpaca-{len(records):05d}",
            "dataset": "alpaca",
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "expected_output_tokens": output_tokens,
        })

    _save_jsonl(records, output_path)
    _print_stats("Alpaca", records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def _print_stats(name: str, records: list[dict]) -> None:
    if not records:
        logger.warning("%s: 0 records", name)
        return

    prompt_lens = [r["prompt_tokens"] for r in records]
    output_lens = [r["expected_output_tokens"] for r in records]

    def _stats(arr):
        a = np.array(arr)
        return {
            "mean": f"{a.mean():.0f}",
            "std": f"{a.std():.0f}",
            "P50": f"{np.percentile(a, 50):.0f}",
            "P95": f"{np.percentile(a, 95):.0f}",
            "P99": f"{np.percentile(a, 99):.0f}",
            "min": f"{a.min():.0f}",
            "max": f"{a.max():.0f}",
        }

    ps = _stats(prompt_lens)
    os_ = _stats(output_lens)

    logger.info(
        "%s (%d records):\n"
        "  Prompt  tokens: mean=%s  std=%s  P50=%s  P95=%s  P99=%s  min=%s  max=%s\n"
        "  Output  tokens: mean=%s  std=%s  P50=%s  P95=%s  P99=%s  min=%s  max=%s",
        name, len(records),
        ps["mean"], ps["std"], ps["P50"], ps["P95"], ps["P99"], ps["min"], ps["max"],
        os_["mean"], os_["std"], os_["P50"], os_["P95"], os_["P99"], os_["min"], os_["max"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & preprocess datasets")
    parser.add_argument(
        "--tokenizer", default=_DEFAULT_TOKENIZER_MODEL,
        help="HuggingFace tokenizer model name",
    )
    parser.add_argument(
        "--output-dir", default="experiments_v2/datasets/cached",
        help="Output directory for JSONL files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only", default=None,
        help="Only download specific dataset: sharegpt, cnndm, alpaca",
    )
    args = parser.parse_args()

    # Pre-load tokenizer
    _get_tokenizer(args.tokenizer)

    out = args.output_dir

    if args.only is None or args.only == "sharegpt":
        download_sharegpt(
            os.path.join(out, "sharegpt_5000.jsonl"),
            max_samples=5000, seed=args.seed,
        )

    if args.only is None or args.only == "cnndm":
        download_cnndm(
            os.path.join(out, "cnndm_3000.jsonl"),
            max_samples=3000, seed=args.seed,
        )

    if args.only is None or args.only == "alpaca":
        download_alpaca(
            os.path.join(out, "alpaca_full.jsonl"),
            seed=args.seed,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
