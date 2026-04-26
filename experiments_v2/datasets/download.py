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
# ArXiv summarization (long-context: ~6-7k input, used in DistServe/Sarathi-Serve)
# ---------------------------------------------------------------------------

def download_arxiv(
    output_path: str,
    max_samples: int = 2000,
    min_prompt_tokens: int = 4000,
    max_total_tokens: int = 7500,
    seed: int = 42,
) -> None:
    """Download and preprocess ArXiv summarization (ccdv/arxiv-summarization).

    Filters to prompts in [min_prompt_tokens, max_total_tokens - expected_output_tokens]
    so they fit within max_model_len=8192 budget with generation headroom.
    """
    logger.info("=== ArXiv (long-context) ===")

    from datasets import load_dataset

    logger.info("Downloading ccdv/arxiv-summarization...")
    ds = load_dataset("ccdv/arxiv-summarization", "document", split="test", trust_remote_code=True)

    tokenizer = _get_tokenizer()
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds))

    records: list[dict] = []
    examined = 0
    for idx in indices:
        if len(records) >= max_samples:
            break
        examined += 1
        if examined > 10000:
            break

        row = ds[int(idx)]
        article = row["article"]
        abstract = row["abstract"]

        prompt = article + "\n\nWrite the abstract for the above paper."
        prompt_tokens = _count_tokens(prompt, tokenizer)
        output_tokens = _count_tokens(abstract, tokenizer)

        if prompt_tokens < min_prompt_tokens:
            continue
        if prompt_tokens + output_tokens > max_total_tokens:
            continue
        if output_tokens < 50:
            continue

        records.append({
            "id": f"arxiv-{len(records):05d}",
            "dataset": "arxiv",
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "expected_output_tokens": output_tokens,
        })

    _save_jsonl(records, output_path)
    _print_stats("ArXiv", records)


# ---------------------------------------------------------------------------
# LongBench (THUDM/LongBench) — long-context multi-task benchmark
# ---------------------------------------------------------------------------
#
# LongBench is distributed as a single data.zip on HuggingFace; we download it,
# extract per-task JSONL files, and convert to the unified format. Each subtask
# has 200 records by design.
#
# We support two output modes here:
#   - single-task file (e.g., longbench_narrativeqa.jsonl)
#   - combined file mixing multiple subtasks (e.g., longbench_qmsum_musique.jsonl)

# Official LongBench prompt templates (from their repo's task2prompt.py).
_LONGBENCH_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one "
        "or more sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
}


def _extract_longbench_zip() -> str:
    """Download LongBench data.zip from HF and extract once. Returns dir path."""
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        "THUDM/LongBench", "data.zip", repo_type="dataset"
    )
    extract_dir = os.path.join(os.path.dirname(zip_path), "extracted")
    data_dir = os.path.join(extract_dir, "data")
    if not os.path.isdir(data_dir):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return data_dir


def _build_longbench_records(
    subtask: str,
    output_dataset_label: str,
    id_prefix: str,
    min_prompt_tokens: int,
    max_prompt_tokens: int,
    min_output_tokens: int,
    max_output_tokens: int,
    start_idx: int = 0,
) -> list[dict]:
    """Tokenize one LongBench subtask, filter by length, return unified records."""
    data_dir = _extract_longbench_zip()
    src = os.path.join(data_dir, f"{subtask}.jsonl")
    if not os.path.exists(src):
        raise FileNotFoundError(f"LongBench subtask file missing: {src}")

    template = _LONGBENCH_PROMPTS[subtask]
    tokenizer = _get_tokenizer()

    rows: list[dict] = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    records: list[dict] = []
    for r in rows:
        context = r.get("context", "")
        question = r.get("input", "")
        answers = r.get("answers") or []
        if not context or not question or not answers:
            continue

        prompt = template.format(context=context, input=question)
        prompt_tokens = _count_tokens(prompt, tokenizer)
        if prompt_tokens < min_prompt_tokens or prompt_tokens > max_prompt_tokens:
            continue

        # Use first reference answer; clamp output token count to a reasonable range
        answer = answers[0]
        ans_tokens = _count_tokens(answer, tokenizer)
        output_tokens = max(min_output_tokens, min(max_output_tokens, ans_tokens))

        records.append({
            "id": f"{id_prefix}-{start_idx + len(records):05d}",
            "dataset": output_dataset_label,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "expected_output_tokens": output_tokens,
            "subtask": subtask,
        })

    return records


def download_longbench(
    output_dir: str,
    subtasks_single: list[str] | None = None,
    mix_subtasks: list[str] | None = None,
    mix_filename: str = "longbench_mix.jsonl",
    mix_label: str = "longbench_mix",
    min_prompt_tokens: int = 2000,
    max_prompt_tokens: int = 28000,
    min_output_tokens: int = 16,
    max_output_tokens: int = 256,
) -> None:
    """Download LongBench and emit unified JSONL files.

    Args:
        output_dir: Where to write the output JSONLs.
        subtasks_single: Subtasks to emit as their own files
            (one file per subtask, label="longbench_<subtask>").
        mix_subtasks: Subtasks to merge into a single combined file.
        mix_filename: Filename for the combined file.
        mix_label: `dataset` label written to records in the combined file.
        min/max_prompt_tokens: Filter prompts outside this Llama-3 token range.
        min/max_output_tokens: Clamp the per-record `expected_output_tokens`.
    """
    logger.info("=== LongBench ===")
    subtasks_single = subtasks_single or []
    mix_subtasks = mix_subtasks or []

    for sub in subtasks_single:
        records = _build_longbench_records(
            sub,
            output_dataset_label=f"longbench_{sub}",
            id_prefix=f"longbench-{sub}",
            min_prompt_tokens=min_prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
            min_output_tokens=min_output_tokens,
            max_output_tokens=max_output_tokens,
        )
        out = os.path.join(output_dir, f"longbench_{sub}.jsonl")
        _save_jsonl(records, out)
        _print_stats(f"LongBench[{sub}]", records)

    if mix_subtasks:
        combined: list[dict] = []
        for sub in mix_subtasks:
            part = _build_longbench_records(
                sub,
                output_dataset_label=mix_label,
                id_prefix=f"longbench-{sub}",
                min_prompt_tokens=min_prompt_tokens,
                max_prompt_tokens=max_prompt_tokens,
                min_output_tokens=min_output_tokens,
                max_output_tokens=max_output_tokens,
                start_idx=0,
            )
            combined.extend(part)
        out = os.path.join(output_dir, mix_filename)
        _save_jsonl(combined, out)
        _print_stats(f"LongBench[mix:{'+'.join(mix_subtasks)}]", combined)


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

    if args.only == "arxiv":
        download_arxiv(
            os.path.join(out, "arxiv_2000.jsonl"),
            max_samples=2000, seed=args.seed,
        )

    if args.only == "longbench":
        download_longbench(
            output_dir=out,
            subtasks_single=["narrativeqa"],
            mix_subtasks=["qmsum", "musique"],
            mix_filename="longbench_qmsum_musique.jsonl",
            mix_label="longbench_qmsum_musique",
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
