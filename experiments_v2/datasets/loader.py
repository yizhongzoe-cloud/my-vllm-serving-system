"""Unified dataset loader for FT serving experiments.

Loads preprocessed JSONL files produced by download.py.
Supports reproducible sampling and wrap-around for large request counts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_dataset(
    dataset_name: str,
    dataset_path: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load a preprocessed dataset from JSONL.

    Args:
        dataset_name: "sharegpt" | "cnndm" | "alpaca" (used for validation).
        dataset_path: Path to the JSONL file.
        max_samples: If set, randomly sample at most this many records.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys:
            id, dataset, prompt, prompt_tokens, expected_output_tokens
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            f"Run: python experiments_v2/datasets/download.py"
        )

    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)

    if not records:
        raise ValueError(f"Empty dataset: {dataset_path}")

    # Validate dataset field matches expected
    sample_ds = records[0].get("dataset", "")
    if sample_ds and sample_ds != dataset_name:
        raise ValueError(
            f"Dataset mismatch: expected '{dataset_name}', "
            f"got '{sample_ds}' in {dataset_path}"
        )

    # Subsample if requested
    if max_samples is not None and max_samples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)

    return records


def sample_from_dataset(
    dataset: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n records from dataset with wrap-around.

    If n > len(dataset), records are reused (sampling with replacement).

    Args:
        dataset: List of dataset records.
        n: Number of samples needed.
        rng: Random number generator for reproducibility.

    Returns:
        List of n records (may contain duplicates if n > len(dataset)).
    """
    if not dataset:
        raise ValueError("Cannot sample from empty dataset")
    size = len(dataset)
    return [dataset[rng.randint(0, size - 1)] for _ in range(n)]
