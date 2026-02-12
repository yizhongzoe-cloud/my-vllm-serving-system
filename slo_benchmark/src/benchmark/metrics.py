"""Metrics computation for benchmark results."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .runner import BenchmarkResult, RequestResult


@dataclass
class LatencyMetrics:
    """Latency statistics."""

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    @classmethod
    def from_values(cls, values: list[float]) -> "LatencyMetrics":
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)

        arr = np.array(values)
        return cls(
            mean_ms=float(np.mean(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
        )


@dataclass
class SLOMetrics:
    """SLO compliance metrics."""

    # TTFT SLO
    ttft_total: int  # Requests with TTFT SLO
    ttft_met: int  # Requests that met TTFT SLO
    ttft_violated: int  # Requests that violated TTFT SLO
    ttft_compliance_rate: float  # Percentage of SLO met

    # E2E SLO
    e2e_total: int
    e2e_met: int
    e2e_violated: int
    e2e_compliance_rate: float

    # Combined
    any_slo_total: int  # Requests with any SLO
    all_slo_met: int  # Requests that met all their SLOs
    overall_compliance_rate: float


@dataclass
class BenchmarkMetrics:
    """Complete benchmark metrics."""

    # Basic stats
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float

    # Throughput
    total_time_s: float
    throughput_rps: float  # Requests per second
    tokens_per_second: float

    # Latency
    ttft: LatencyMetrics
    e2e_latency: LatencyMetrics

    # SLO compliance
    slo: SLOMetrics

    # Config
    config: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "total_time_s": self.total_time_s,
            "throughput_rps": self.throughput_rps,
            "tokens_per_second": self.tokens_per_second,
            "ttft": asdict(self.ttft),
            "e2e_latency": asdict(self.e2e_latency),
            "slo": asdict(self.slo),
            "config": self.config,
        }

    def save(self, path: str | Path) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            "",
            "## Overview",
            f"  Total requests:     {self.total_requests}",
            f"  Successful:         {self.successful_requests}",
            f"  Failed:             {self.failed_requests}",
            f"  Success rate:       {self.success_rate:.1%}",
            "",
            "## Throughput",
            f"  Total time:         {self.total_time_s:.2f}s",
            f"  Throughput:         {self.throughput_rps:.2f} req/s",
            f"  Token throughput:   {self.tokens_per_second:.2f} tok/s",
            "",
            "## Latency (TTFT)",
            f"  Mean:   {self.ttft.mean_ms:.1f}ms",
            f"  P50:    {self.ttft.p50_ms:.1f}ms",
            f"  P90:    {self.ttft.p90_ms:.1f}ms",
            f"  P99:    {self.ttft.p99_ms:.1f}ms",
            "",
            "## Latency (E2E)",
            f"  Mean:   {self.e2e_latency.mean_ms:.1f}ms",
            f"  P50:    {self.e2e_latency.p50_ms:.1f}ms",
            f"  P90:    {self.e2e_latency.p90_ms:.1f}ms",
            f"  P99:    {self.e2e_latency.p99_ms:.1f}ms",
            "",
            "## SLO Compliance",
            f"  TTFT SLO:    {self.slo.ttft_met}/{self.slo.ttft_total} met ({self.slo.ttft_compliance_rate:.1%})",
            f"  E2E SLO:     {self.slo.e2e_met}/{self.slo.e2e_total} met ({self.slo.e2e_compliance_rate:.1%})",
            f"  Overall:     {self.slo.all_slo_met}/{self.slo.any_slo_total} met ({self.slo.overall_compliance_rate:.1%})",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_metrics(result: BenchmarkResult) -> BenchmarkMetrics:
    """Compute all metrics from benchmark result."""
    successful = [r for r in result.results if r.success]
    failed = [r for r in result.results if not r.success]

    # Latency metrics
    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    e2e_values = [r.e2e_latency_ms for r in successful if r.e2e_latency_ms is not None]

    ttft_metrics = LatencyMetrics.from_values(ttft_values)
    e2e_metrics = LatencyMetrics.from_values(e2e_values)

    # Token throughput
    total_tokens = sum(r.output_tokens for r in successful)
    tokens_per_second = total_tokens / result.total_time_s if result.total_time_s > 0 else 0

    # SLO metrics
    ttft_slo_requests = [r for r in successful if r.request.ttft_slo_ms is not None]
    ttft_met = sum(1 for r in ttft_slo_requests if r.ttft_slo_met)

    e2e_slo_requests = [r for r in successful if r.request.e2e_latency_slo_ms is not None]
    e2e_met = sum(1 for r in e2e_slo_requests if r.e2e_slo_met)

    any_slo_requests = [
        r for r in successful
        if r.request.ttft_slo_ms is not None or r.request.e2e_latency_slo_ms is not None
    ]

    # All SLOs met means: TTFT met (if exists) AND E2E met (if exists)
    all_slo_met = sum(
        1 for r in any_slo_requests
        if (r.ttft_slo_met is None or r.ttft_slo_met)
        and (r.e2e_slo_met is None or r.e2e_slo_met)
    )

    slo_metrics = SLOMetrics(
        ttft_total=len(ttft_slo_requests),
        ttft_met=ttft_met,
        ttft_violated=len(ttft_slo_requests) - ttft_met,
        ttft_compliance_rate=ttft_met / len(ttft_slo_requests) if ttft_slo_requests else 1.0,
        e2e_total=len(e2e_slo_requests),
        e2e_met=e2e_met,
        e2e_violated=len(e2e_slo_requests) - e2e_met,
        e2e_compliance_rate=e2e_met / len(e2e_slo_requests) if e2e_slo_requests else 1.0,
        any_slo_total=len(any_slo_requests),
        all_slo_met=all_slo_met,
        overall_compliance_rate=all_slo_met / len(any_slo_requests) if any_slo_requests else 1.0,
    )

    return BenchmarkMetrics(
        total_requests=len(result.results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        success_rate=len(successful) / len(result.results) if result.results else 0,
        total_time_s=result.total_time_s,
        throughput_rps=result.throughput,
        tokens_per_second=tokens_per_second,
        ttft=ttft_metrics,
        e2e_latency=e2e_metrics,
        slo=slo_metrics,
        config=result.config,
    )


def compare_results(
    baseline: BenchmarkMetrics,
    experiment: BenchmarkMetrics,
) -> dict[str, Any]:
    """Compare two benchmark results."""
    return {
        "throughput_change": (experiment.throughput_rps - baseline.throughput_rps) / baseline.throughput_rps if baseline.throughput_rps > 0 else 0,
        "ttft_p50_change": (experiment.ttft.p50_ms - baseline.ttft.p50_ms) / baseline.ttft.p50_ms if baseline.ttft.p50_ms > 0 else 0,
        "ttft_p99_change": (experiment.ttft.p99_ms - baseline.ttft.p99_ms) / baseline.ttft.p99_ms if baseline.ttft.p99_ms > 0 else 0,
        "e2e_p50_change": (experiment.e2e_latency.p50_ms - baseline.e2e_latency.p50_ms) / baseline.e2e_latency.p50_ms if baseline.e2e_latency.p50_ms > 0 else 0,
        "e2e_p99_change": (experiment.e2e_latency.p99_ms - baseline.e2e_latency.p99_ms) / baseline.e2e_latency.p99_ms if baseline.e2e_latency.p99_ms > 0 else 0,
        "slo_compliance_change": experiment.slo.overall_compliance_rate - baseline.slo.overall_compliance_rate,
    }
