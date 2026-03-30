"""Workload trace generation for FT serving experiments.

Generates request traces with specified arrival patterns, prompt/output
length distributions, and SLO targets. Each trace is a list of
RequestSpec objects that the experiment runner sends to the server.
"""

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class RequestSpec:
    """One request in a workload trace."""

    request_id: str
    arrival_time: float  # seconds from experiment start
    prompt_text: str
    prompt_len: int  # target prompt token count
    expected_output_len: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    failure_gap_slo_ms: float


# Corpus of diverse prompts that can be repeated/truncated to target lengths.
_PROMPT_TEMPLATES = [
    "Write a detailed essay about the history of {topic}. Cover all major "
    "developments, key figures, and turning points from the earliest origins "
    "to the present day. Include specific dates and examples.",
    "Explain in depth how {topic} works, starting from first principles. "
    "Describe the underlying mechanisms, current state of the art, and "
    "practical applications in modern systems.",
    "Provide a comprehensive analysis of {topic}, including its advantages, "
    "disadvantages, common misconceptions, and future directions. Support "
    "your analysis with concrete examples.",
    "Describe the complete process of {topic} step by step, including all "
    "intermediate stages, potential failure modes, and best practices for "
    "achieving optimal results.",
    "Compare and contrast different approaches to {topic}. Evaluate each "
    "approach on criteria such as performance, scalability, reliability, "
    "and ease of implementation.",
]

_TOPICS = [
    "artificial intelligence", "distributed systems", "quantum computing",
    "renewable energy", "space exploration", "molecular biology",
    "cryptography", "neural networks", "operating systems",
    "database design", "compiler optimization", "computer graphics",
    "network protocols", "machine learning", "software engineering",
    "parallel computing", "information theory", "robotics",
    "natural language processing", "computer vision",
]


def _make_prompt(target_tokens: int, rng: random.Random) -> str:
    """Generate a prompt string targeting approximately target_tokens tokens.

    Uses ~1.3 chars per token as a rough approximation.
    Repeats template text to reach desired length.
    """
    chars_needed = int(target_tokens * 4.0)  # ~4 chars per token for English
    template = rng.choice(_PROMPT_TEMPLATES)
    topic = rng.choice(_TOPICS)
    base = template.format(topic=topic)

    # Repeat to reach target length.
    if len(base) >= chars_needed:
        return base[:chars_needed]

    repeats = (chars_needed // len(base)) + 1
    extended = " ".join([base] * repeats)
    return extended[:chars_needed]


def generate_trace(
    workload_config: dict,
    rps: float,
    duration_sec: float,
    seed: int,
    slo_config: dict,
    warmup_sec: float = 0.0,
) -> list[RequestSpec]:
    """Generate a workload trace.

    Args:
        workload_config: Dict with prompt_len_min/max, output_len_min/max,
            arrival type, etc.
        rps: Target requests per second.
        duration_sec: Total trace duration in seconds.
        seed: Random seed for reproducibility.
        slo_config: Dict with ttft_ms, tpot_ms, failure_gap_ms.
        warmup_sec: Shift all arrival times forward by this amount.

    Returns:
        List of RequestSpec sorted by arrival_time.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    prompt_min = workload_config["prompt_len_min"]
    prompt_max = workload_config["prompt_len_max"]
    output_min = workload_config["output_len_min"]
    output_max = workload_config["output_len_max"]
    arrival_type = workload_config.get("arrival", "poisson")

    ttft_slo = slo_config.get("ttft_ms", 0.0)
    tpot_slo = slo_config.get("tpot_ms", 0.0)
    gap_slo = slo_config.get("failure_gap_ms", 0.0)

    # Generate arrival times.
    arrival_times = _generate_arrivals(
        arrival_type, rps, duration_sec, np_rng, workload_config
    )

    trace = []
    for i, t in enumerate(arrival_times):
        prompt_len = rng.randint(prompt_min, prompt_max)
        output_len = rng.randint(output_min, output_max)
        prompt_text = _make_prompt(prompt_len, rng)

        trace.append(RequestSpec(
            request_id=f"req-{i:05d}",
            arrival_time=t + warmup_sec,
            prompt_text=prompt_text,
            prompt_len=prompt_len,
            expected_output_len=output_len,
            ttft_slo_ms=ttft_slo,
            tpot_slo_ms=tpot_slo,
            failure_gap_slo_ms=gap_slo,
        ))

    return trace


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
    rps: float, duration_sec: float, np_rng: np.random.RandomState
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
        # Determine current period: burst or normal.
        cycle_pos = t % (burst_interval + burst_dur)
        if cycle_pos < burst_interval:
            # Normal period.
            current_rps = base_rps
        else:
            # Burst period.
            current_rps = base_rps * burst_mult

        gap = np_rng.exponential(1.0 / current_rps) if current_rps > 0 else 1.0
        t += gap
        if t < duration_sec:
            times.append(t)

    return times
