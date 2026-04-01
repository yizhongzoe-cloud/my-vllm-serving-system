"""Workload trace generation for FT serving experiments (v2).

Generates request traces from real datasets (ShareGPT, CNN/DailyMail, Alpaca)
with controlled arrival patterns and per-request SLO targets.

Key differences from experiments/workloads.py:
  - Real dataset prompts instead of synthetic template text
  - Real output length distributions instead of uniform random
  - Per-request TPOT SLO (different workloads can have different SLOs)
  - Mixed workload support (W4_Mixed)
  - Wrap-around sampling when request count > dataset size
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from experiments_v2.datasets.loader import load_dataset, sample_from_dataset


@dataclass
class RequestSpec:
    """One request in a workload trace."""

    request_id: str
    arrival_time: float  # seconds from experiment start
    prompt_text: str
    prompt_len: int  # actual prompt token count (from dataset)
    expected_output_len: int  # actual output token count (from dataset)
    ttft_slo_ms: float
    tpot_slo_ms: float
    failure_gap_slo_ms: float
    dataset: str = ""  # source dataset name
    original_id: str = ""  # original ID in the dataset


def generate_trace(
    workload_config: dict,
    rps: float,
    duration_sec: float,
    seed: int,
    slo_config: dict,
    warmup_sec: float = 0.0,
    all_workload_configs: dict | None = None,
    max_model_len: int = 4096,
) -> list[RequestSpec]:
    """Generate a workload trace from a real dataset.

    Args:
        workload_config: Workload definition from config YAML. Must have either
            {dataset, dataset_path, tpot_slo_ms} for single-dataset workloads,
            or {mix: [{workload, weight}, ...]} for mixed workloads.
        rps: Target requests per second.
        duration_sec: Total trace duration in seconds.
        seed: Random seed for reproducibility.
        slo_config: Global SLO dict with ttft_ms, failure_gap_ms.
        warmup_sec: Shift all arrival times forward by this amount.
        all_workload_configs: All workload definitions (needed for mixed).
        max_model_len: Maximum context length. Requests exceeding this are
            clamped (output shortened) or skipped (prompt alone exceeds).

    Returns:
        List of RequestSpec sorted by arrival_time.
    """
    if "mix" in workload_config:
        return _generate_mixed_trace(
            workload_config, all_workload_configs or {},
            rps, duration_sec, seed, slo_config, warmup_sec, max_model_len,
        )
    return _generate_single_trace(
        workload_config, rps, duration_sec, seed, slo_config, warmup_sec,
        max_model_len,
    )


def _generate_single_trace(
    workload_config: dict,
    rps: float,
    duration_sec: float,
    seed: int,
    slo_config: dict,
    warmup_sec: float,
    max_model_len: int = 4096,
) -> list[RequestSpec]:
    """Generate trace from a single dataset."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    arrival_type = workload_config.get("arrival", "poisson")
    ttft_slo = slo_config.get("ttft_ms", 0.0) or 0.0
    gap_slo = slo_config.get("failure_gap_ms", 0.0) or 0.0
    tpot_slo = workload_config.get("tpot_slo_ms", 0.0) or 0.0

    # Load dataset
    dataset_name = workload_config["dataset"]
    dataset_path = workload_config["dataset_path"]
    dataset = load_dataset(dataset_name, dataset_path, seed=seed)

    # Generate arrival times
    arrival_times = _generate_arrivals(
        arrival_type, rps, duration_sec, np_rng, workload_config,
    )

    # Sample from dataset (with wrap-around)
    samples = sample_from_dataset(dataset, len(arrival_times), rng)

    trace = []
    for i, (t, sample) in enumerate(zip(arrival_times, samples)):
        # Clamp to max_model_len: reduce output if prompt+output exceeds limit
        prompt_len = sample["prompt_tokens"]
        output_len = sample["expected_output_tokens"]
        if prompt_len + output_len > max_model_len:
            output_len = max(1, max_model_len - prompt_len)
        if prompt_len >= max_model_len:
            # Skip samples that can't even fit the prompt
            continue
        trace.append(RequestSpec(
            request_id=f"req-{i:05d}",
            arrival_time=t + warmup_sec,
            prompt_text=sample["prompt"],
            prompt_len=prompt_len,
            expected_output_len=output_len,
            ttft_slo_ms=ttft_slo,
            tpot_slo_ms=tpot_slo,
            failure_gap_slo_ms=gap_slo,
            dataset=sample.get("dataset", dataset_name),
            original_id=sample.get("id", ""),
        ))

    return trace


def _generate_mixed_trace(
    workload_config: dict,
    all_workload_configs: dict,
    rps: float,
    duration_sec: float,
    seed: int,
    slo_config: dict,
    warmup_sec: float,
    max_model_len: int = 4096,
) -> list[RequestSpec]:
    """Generate trace from a mix of workloads with per-request SLO."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    mix = workload_config["mix"]
    arrival_type = workload_config.get("arrival", "poisson")

    ttft_slo = slo_config.get("ttft_ms", 0.0) or 0.0
    gap_slo = slo_config.get("failure_gap_ms", 0.0) or 0.0

    # Pre-load all component datasets
    component_datasets: dict[str, list[dict]] = {}
    component_tpot: dict[str, float] = {}
    weights: list[float] = []
    workload_names: list[str] = []

    for entry in mix:
        wname = entry["workload"]
        weight = entry["weight"]
        wconfig = all_workload_configs[wname]

        workload_names.append(wname)
        weights.append(weight)
        component_tpot[wname] = wconfig.get("tpot_slo_ms", 0.0) or 0.0

        if wname not in component_datasets:
            component_datasets[wname] = load_dataset(
                wconfig["dataset"], wconfig["dataset_path"], seed=seed,
            )

    # Normalize weights
    total_w = sum(weights)
    cum_weights = []
    running = 0.0
    for w in weights:
        running += w / total_w
        cum_weights.append(running)

    # Generate arrival times
    arrival_times = _generate_arrivals(
        arrival_type, rps, duration_sec, np_rng, workload_config,
    )

    trace = []
    for i, t in enumerate(arrival_times):
        # Sample which workload this request belongs to
        r = rng.random()
        chosen_idx = 0
        for j, cw in enumerate(cum_weights):
            if r <= cw:
                chosen_idx = j
                break

        wname = workload_names[chosen_idx]
        ds = component_datasets[wname]
        sample = ds[rng.randint(0, len(ds) - 1)]

        prompt_len = sample["prompt_tokens"]
        output_len = sample["expected_output_tokens"]
        if prompt_len + output_len > max_model_len:
            output_len = max(1, max_model_len - prompt_len)
        if prompt_len >= max_model_len:
            continue

        trace.append(RequestSpec(
            request_id=f"req-{i:05d}",
            arrival_time=t + warmup_sec,
            prompt_text=sample["prompt"],
            prompt_len=prompt_len,
            expected_output_len=output_len,
            ttft_slo_ms=ttft_slo,
            tpot_slo_ms=component_tpot[wname],
            failure_gap_slo_ms=gap_slo,
            dataset=sample.get("dataset", ""),
            original_id=sample.get("id", ""),
        ))

    return trace


# ===================================================================
# Arrival processes (unchanged from v1)
# ===================================================================

def _generate_arrivals(
    arrival_type: str,
    rps: float,
    duration_sec: float,
    np_rng: np.random.RandomState,
    config: dict,
) -> list[float]:
    """Generate arrival times based on the arrival process type."""
    if rps <= 0:
        return []
    if arrival_type == "poisson":
        return _poisson_arrivals(rps, duration_sec, np_rng)
    elif arrival_type == "bursty":
        return _bursty_arrivals(rps, duration_sec, np_rng, config)
    else:
        raise ValueError(f"Unknown arrival type: {arrival_type}")


def _poisson_arrivals(
    rps: float, duration_sec: float, np_rng: np.random.RandomState,
) -> list[float]:
    """Generate Poisson process arrival times."""
    times = []
    t = 0.0
    while t < duration_sec:
        gap = np_rng.exponential(1.0 / rps)
        t += gap
        if t < duration_sec:
            times.append(t)
    return times


def _bursty_arrivals(
    base_rps: float,
    duration_sec: float,
    np_rng: np.random.RandomState,
    config: dict,
) -> list[float]:
    """Generate bursty arrivals: alternating normal and burst periods."""
    burst_mult = config.get("burst_high_multiplier", 3.0)
    burst_dur = config.get("burst_duration_sec", 5.0)
    burst_interval = config.get("burst_interval_sec", 15.0)

    times = []
    t = 0.0
    while t < duration_sec:
        cycle_pos = t % (burst_interval + burst_dur)
        if cycle_pos < burst_interval:
            current_rps = base_rps
        else:
            current_rps = base_rps * burst_mult
        gap = np_rng.exponential(1.0 / current_rps) if current_rps > 0 else 1.0
        t += gap
        if t < duration_sec:
            times.append(t)
    return times
