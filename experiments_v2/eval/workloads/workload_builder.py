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
    "arxivsumm": ("arxivsumm.jsonl", "arxivsumm"),
    "burstgpt_mixed": ("burstgpt_mixed.jsonl", "burstgpt_mixed"),
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


def build_tiered_schedule(
    dataset_name: str,
    num_requests: int,
    arrival_rate_qps: float,
    tight_ratio: float,
    seed: int = 0,
    burst_size: int = 1,
    burst_spread_s: float = 0.0,
    max_tokens_cap: int = DEFAULT_MAX_OUTPUT_CAP,
) -> list[tuple[float, str, int, int, str]]:
    """Build a Poisson schedule of ONE long-context dataset, but split
    into two SLO tiers ("tight" / "loose").

    Unlike build_mixed_schedule (which mixes two *datasets* — short
    interactive vs long document), this keeps every request on the SAME
    long-context dataset and differs only in the SLO tier assigned. This
    is the heterogeneous-SLO setting: identical workload, some requests
    are latency-critical (tight) and some are best-effort (loose). The
    tier label lets the caller give loose requests a looser deadline
    (e.g. 2x the tight deadline) so the picker has a genuine winner when
    it delays a slack-rich loose request to save a tight one.

    Args:
        dataset_name: long-context dataset (e.g. 'arxivsumm').
        num_requests: total request count.
        arrival_rate_qps: Poisson rate λ.
        tight_ratio: fraction assigned the "tight" tier. Rest are "loose".
        seed: reproducibility — controls dataset sample, arrivals, and
              the (shuffled) tier assignment.
        max_tokens_cap: per-request max_tokens cap.

    Returns:
        List of (arrival_offset_s, prompt, max_tokens, prompt_tokens,
        tier), sorted by arrival_offset. `prompt_tokens` is the dataset's
        recorded prompt length (so the caller can derive a prompt-aware
        startup budget without re-tokenizing); `tier` is "tight"/"loose".
    """
    if not 0.0 <= tight_ratio <= 1.0:
        raise ValueError(f"tight_ratio must be in [0,1], got {tight_ratio}")
    if arrival_rate_qps <= 0:
        raise ValueError(
            f"arrival_rate_qps must be > 0, got {arrival_rate_qps}")
    if num_requests <= 0:
        raise ValueError(f"num_requests must be > 0, got {num_requests}")

    path, loader_tag = resolve_dataset_info(dataset_name)
    records = load_dataset(
        dataset_name=loader_tag, dataset_path=path,
        max_samples=num_requests, seed=seed,
    )
    if len(records) < num_requests:
        raise RuntimeError(
            f"dataset {dataset_name} only has {len(records)} records but "
            f"{num_requests} requested"
        )

    # Poisson inter-arrival sequence (identical convention to
    # build_schedule so tiered vs uniform runs share arrival statistics).
    rng = np.random.default_rng(seed)
    if num_requests == 1:
        offsets = np.array([0.0])
    elif burst_size > 1:
        # Compound-Poisson (batched) arrivals. Burst ONSETS are Poisson at
        # rate lambda/burst_size, so the MEAN request rate stays
        # arrival_rate_qps, but each onset injects `burst_size` requests
        # spread over burst_spread_s. This gives transient overload + drain
        # (CV >> 1) — the sub-saturation-with-spikes regime where slack-aware
        # preempt-resume can help. burst_size=1 reduces to plain Poisson.
        n_bursts = (num_requests + burst_size - 1) // burst_size
        onset_rate = arrival_rate_qps / burst_size
        if n_bursts == 1:
            onsets = np.array([0.0])
        else:
            onset_inter = rng.exponential(
                scale=1.0 / onset_rate, size=n_bursts - 1,
            )
            onsets = np.concatenate([[0.0], np.cumsum(onset_inter)])
        times: list[float] = []
        for o in onsets:
            for _ in range(burst_size):
                jitter = (rng.uniform(0.0, burst_spread_s)
                          if burst_spread_s > 0 else 0.0)
                times.append(float(o) + jitter)
        arr = np.sort(np.array(times))[:num_requests]
        offsets = arr - arr[0]
    else:
        inter = rng.exponential(
            scale=1.0 / arrival_rate_qps, size=num_requests - 1,
        )
        offsets = np.concatenate([[0.0], np.cumsum(inter)])

    # Tier assignment: deterministically mark n_tight request *positions*
    # as tight via a seeded permutation, so tight/loose are interleaved
    # in arrival order rather than blocked. n_tight uses floor; with
    # tight_ratio=0.3 and 60 reqs that is 18 tight / 42 loose.
    n_tight = int(num_requests * tight_ratio)
    tier_rng = np.random.default_rng(seed + 12345)
    perm = tier_rng.permutation(num_requests)
    tight_positions = set(int(p) for p in perm[:n_tight])

    schedule: list[tuple[float, str, int, int, str]] = []
    for i, rec in enumerate(records):
        out_tokens = int(rec.get("expected_output_tokens", max_tokens_cap))
        max_tokens = max(1, min(max_tokens_cap, out_tokens))
        prompt_tokens = int(rec.get("prompt_tokens", 0))
        tier = "tight" if i in tight_positions else "loose"
        schedule.append(
            (float(offsets[i]), rec["prompt"], max_tokens,
             prompt_tokens, tier)
        )
    return schedule


def build_mixed_schedule(
    short_dataset: str,
    long_dataset: str,
    short_ratio: float,
    num_requests: int,
    arrival_rate_qps: float,
    seed: int = 0,
    max_tokens_cap: int = DEFAULT_MAX_OUTPUT_CAP,
) -> list[tuple[float, str, int, str]]:
    """Build a Poisson schedule mixing two datasets.

    For modeling realistic deployments where short interactive queries
    (e.g. ShareGPT chatbot) coexist with long document analysis
    (e.g. ArXiv-Summarization) on the same engine pool. Each request
    is tagged with its origin class ("short" or "long") so the caller
    can apply per-class SLO thresholds (JITServe-style).

    Args:
        short_dataset: workload_builder dataset name for the short class
                       (e.g. 'sharegpt').
        long_dataset: workload_builder dataset name for the long class
                      (e.g. 'arxivsumm').
        short_ratio: fraction of requests drawn from short_dataset.
                     Long fraction = 1 - short_ratio.
        num_requests: total request count across both classes.
        arrival_rate_qps: Poisson rate λ over the combined stream.
        seed: reproducibility.
        max_tokens_cap: per-request max_tokens cap.

    Returns:
        List of (arrival_offset_s, prompt, max_tokens, class), sorted
        by arrival_offset. `class` is "short" or "long".
    """
    if not 0.0 < short_ratio < 1.0:
        raise ValueError(f"short_ratio must be in (0,1), got {short_ratio}")
    n_short = int(round(num_requests * short_ratio))
    n_long = num_requests - n_short
    if n_short < 1 or n_long < 1:
        raise ValueError(
            f"num_requests={num_requests} with short_ratio={short_ratio} "
            f"gives n_short={n_short}, n_long={n_long}; need both >= 1"
        )

    # Load each dataset independently; reuse seed for reproducibility.
    short_path, short_tag = resolve_dataset_info(short_dataset)
    long_path, long_tag = resolve_dataset_info(long_dataset)
    short_recs = load_dataset(
        dataset_name=short_tag, dataset_path=short_path,
        max_samples=n_short, seed=seed,
    )
    long_recs = load_dataset(
        dataset_name=long_tag, dataset_path=long_path,
        max_samples=n_long, seed=seed + 1,  # different seed offset
    )
    if len(short_recs) < n_short:
        raise RuntimeError(
            f"short dataset {short_dataset} only has {len(short_recs)} "
            f"records, need {n_short}"
        )
    if len(long_recs) < n_long:
        raise RuntimeError(
            f"long dataset {long_dataset} only has {len(long_recs)} "
            f"records, need {n_long}"
        )

    # Tag each record with its class.
    tagged: list[tuple[dict, str]] = (
        [(r, "short") for r in short_recs] +
        [(r, "long") for r in long_recs]
    )

    # Shuffle so short/long are interleaved in arrival order (not
    # blocked into "all shorts first then all longs").
    rng = np.random.default_rng(seed)
    rng.shuffle(tagged)

    # Poisson inter-arrival sequence over the combined stream.
    if num_requests == 1:
        offsets = np.array([0.0])
    else:
        inter = rng.exponential(
            scale=1.0 / arrival_rate_qps, size=num_requests - 1,
        )
        offsets = np.concatenate([[0.0], np.cumsum(inter)])

    schedule: list[tuple[float, str, int, str]] = []
    for i, (rec, cls) in enumerate(tagged):
        out_tokens = int(rec.get("expected_output_tokens", max_tokens_cap))
        max_tokens = max(1, min(max_tokens_cap, out_tokens))
        schedule.append((float(offsets[i]), rec["prompt"], max_tokens, cls))
    return schedule


def build_trace_schedule(
    dataset_name: str,
    num_requests: int,
    seed: int = 0,
    max_tokens_cap: int = DEFAULT_MAX_OUTPUT_CAP,
) -> list[tuple[float, str, int, str]]:
    """Load a workload whose arrival pattern is embedded in the dataset.

    Used by trace-driven workloads like burstgpt_mixed, where each record
    has an `arrival_offset_s` field pre-computed from a real production
    trace (e.g. BurstGPT's per-request timestamps).

    Args:
        dataset_name: registered name (e.g. 'burstgpt_mixed').
        num_requests: how many records to use (truncates the schedule).
        seed: reserved; trace-driven schedules are deterministic given
              the source jsonl, so seed only affects which subset is
              sampled when num_requests < len(file).
        max_tokens_cap: per-request max_tokens cap.

    Returns:
        List of (arrival_offset_s, prompt, max_tokens, class), sorted
        by arrival_offset. The records' embedded class tag carries
        through; arrival pattern is whatever the trace dictates.
    """
    path, loader_tag = resolve_dataset_info(dataset_name)
    records = load_dataset(
        dataset_name=loader_tag, dataset_path=path,
        max_samples=num_requests, seed=seed,
    )
    if len(records) < num_requests:
        raise RuntimeError(
            f"trace dataset {dataset_name} only has {len(records)} "
            f"records, need {num_requests}"
        )
    # Re-sort by arrival_offset_s in case load_dataset's sampling
    # scrambles it (sample is by random index, not time).
    records.sort(key=lambda r: r.get("arrival_offset_s", 0.0))
    # Re-base so the earliest offset is 0.
    base = records[0].get("arrival_offset_s", 0.0)
    schedule: list[tuple[float, str, int, str]] = []
    for rec in records:
        off = float(rec.get("arrival_offset_s", 0.0)) - base
        out_tokens = int(rec.get("expected_output_tokens", max_tokens_cap))
        max_tokens = max(1, min(max_tokens_cap, out_tokens))
        cls = rec.get("class", "short")
        schedule.append((off, rec["prompt"], max_tokens, cls))
    return schedule
