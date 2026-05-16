# SPDX-License-Identifier: Apache-2.0
"""Workload builder: dataset prompts + Poisson arrival schedule.

Standard Poisson process: inter-arrival times drawn from Exp(rate), giving
mean throughput of `rate` requests/sec. Same convention as vLLM's official
benchmark_serving_multi_turn.py (np.random.exponential(1/rate)).

Usage:
    schedule = build_schedule(
        dataset_path="experiments_v2/datasets/cached/ruler_64k_niah.jsonl",
        dataset_name="ruler",
        num_requests=100,
        arrival_rate_qps=0.3,
        seed=0,
    )
    for arrival_offset_s, prompt, max_tokens in schedule:
        # block until t = experiment_start + arrival_offset_s
        # POST request
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments_v2.datasets.loader import load_dataset

DEFAULT_MAX_OUTPUT_CAP = 200


# Dataset short name → (cached JSONL filename, loader's expected
# `dataset` field for validation).
_DATASET_INFO: dict[str, tuple[str, str]] = {
    "ruler_64k": ("ruler_64k_niah.jsonl", "ruler_64k_niah"),
    "ruler_16k": ("ruler_16384_niah_trunc.jsonl", "ruler_16384_niah_trunc"),
    "ruler_8k": ("ruler_8192_niah_trunc.jsonl", "ruler_8192_niah_trunc"),
    "ruler_4k": ("ruler_4096_niah_trunc.jsonl", "ruler_4096_niah_trunc"),
    "ruler_2k": ("ruler_2048_niah_trunc.jsonl", "ruler_2048_niah_trunc"),
    "ruler_1k": ("ruler_1024_niah_trunc.jsonl", "ruler_1024_niah_trunc"),
    "ruler_mixed": (
        "ruler_mixed_niah_trunc.jsonl", "ruler_mixed_niah_trunc",
    ),
    "sharegpt": ("sharegpt_5000.jsonl", "sharegpt"),
}


def resolve_dataset_info(short_name: str) -> tuple[str, str]:
    """Map a short dataset name to (cached JSONL path, loader tag)."""
    if short_name not in _DATASET_INFO:
        raise ValueError(
            f"unknown dataset '{short_name}'. "
            f"Known: {sorted(_DATASET_INFO)}"
        )
    filename, loader_tag = _DATASET_INFO[short_name]
    base = (Path(__file__).resolve().parents[2]
            / "datasets" / "cached")
    p = base / filename
    if not p.exists():
        raise FileNotFoundError(
            f"dataset file missing: {p}. "
            f"Checkout from zoe/slo-scheduling or re-download."
        )
    return str(p), loader_tag


def build_schedule(
    dataset_name: str,
    num_requests: int,
    arrival_rate_qps: float,
    seed: int = 0,
    max_tokens_cap: int = DEFAULT_MAX_OUTPUT_CAP,
    force_max_tokens: int | None = None,
) -> list[tuple[float, str, int]]:
    """Build a Poisson arrival schedule of (offset_s, prompt, max_tokens).

    Args:
        dataset_name: short name like 'ruler_64k', 'sharegpt'.
        num_requests: how many requests to schedule.
        arrival_rate_qps: target QPS (Poisson rate λ).
        seed: reproducibility — controls both the dataset sample and the
              inter-arrival sequence.
        max_tokens_cap: cap on per-request max_tokens, so a long expected
                        output in the dataset doesn't blow up our runs.
        force_max_tokens: if set, OVERRIDE each request's max_tokens to
                          this value, ignoring the dataset's
                          expected_output_tokens. Used for long-output
                          experiments where we want every request to
                          decode N tokens regardless of the natural
                          answer length (must be paired with
                          ignore_eos=True on the client side, otherwise
                          the model emits EOS early and never reaches
                          this cap).

    Returns:
        List of (arrival_offset_s, prompt_text, max_tokens), sorted by
        arrival_offset. First request has offset_s = 0.0.
    """
    if arrival_rate_qps <= 0:
        raise ValueError(f"arrival_rate_qps must be > 0, got {arrival_rate_qps}")
    if num_requests <= 0:
        raise ValueError(f"num_requests must be > 0, got {num_requests}")

    # Step 1: load prompts.
    path, loader_tag = resolve_dataset_info(dataset_name)
    records = load_dataset(
        dataset_name=loader_tag,
        dataset_path=path,
        max_samples=num_requests,
        seed=seed,
    )
    if len(records) < num_requests:
        raise RuntimeError(
            f"dataset {dataset_name} only has {len(records)} records but "
            f"{num_requests} requested"
        )

    # Step 2: Poisson inter-arrival sequence.
    rng = np.random.default_rng(seed)
    if num_requests == 1:
        offsets = np.array([0.0])
    else:
        inter = rng.exponential(
            scale=1.0 / arrival_rate_qps, size=num_requests - 1,
        )
        offsets = np.concatenate([[0.0], np.cumsum(inter)])

    # Step 3: assemble.
    schedule: list[tuple[float, str, int]] = []
    for i, rec in enumerate(records):
        if force_max_tokens is not None:
            # Caller wants a deterministic long-output workload — use
            # the override directly, ignore dataset's natural answer
            # length. Caller MUST set ignore_eos=True on the client
            # for this to actually generate `force_max_tokens` tokens.
            max_tokens = max(1, int(force_max_tokens))
        else:
            out_tokens = int(rec.get("expected_output_tokens", max_tokens_cap))
            max_tokens = max(1, min(max_tokens_cap, out_tokens))
        schedule.append((float(offsets[i]), rec["prompt"], max_tokens))
    return schedule


def summarize_schedule(
    schedule: list[tuple[float, str, int]],
) -> dict:
    """Diagnostic summary of a built schedule."""
    if not schedule:
        return {"n": 0}
    offsets = np.array([s[0] for s in schedule])
    max_toks = np.array([s[2] for s in schedule])
    # Empirical rate over the [first, last] arrival window.
    if len(schedule) > 1:
        window = offsets[-1] - offsets[0]
        empirical_qps = (len(schedule) - 1) / window if window > 0 else 0.0
    else:
        empirical_qps = 0.0
    return {
        "n": len(schedule),
        "window_s": float(offsets[-1]),
        "empirical_qps": float(empirical_qps),
        "max_tokens_p50": int(np.percentile(max_toks, 50)),
        "max_tokens_p95": int(np.percentile(max_toks, 95)),
    }
