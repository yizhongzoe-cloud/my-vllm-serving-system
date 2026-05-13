#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot E_M1 main figure: SLO attainment vs QPS, one line per baseline.

Reads experiments_v2/eval/results/e_m1_*_metrics.json. Groups by
(baseline, dataset), aggregates over seeds (mean ± std). Plots x=QPS,
y=SLO_met%, with error bars for std.

Output: experiments_v2/figures/e_m1_slo_vs_qps_<dataset>.{png,pdf}

Usage:
  python experiments_v2/eval/analysis/plot_em1.py --dataset ruler_64k
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_ORDER = ["vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours"]
BASELINE_STYLE = {
    "vllm_fcfs":       {"color": "#999999", "marker": "s", "label": "vLLM-FCFS"},
    "reroute_no_ckpt": {"color": "#1f77b4", "marker": "o", "label": "Reroute-no-ckpt"},
    "ours_no_picker":  {"color": "#ff7f0e", "marker": "^", "label": "Ours (no picker)"},
    "ours":            {"color": "#2ca02c", "marker": "D", "label": "Ours"},
}


def load_em1(dataset: str) -> list[dict]:
    out: list[dict] = []
    for fp in sorted(RESULTS_DIR.glob("e_m1_*_metrics.json")):
        try:
            data = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("dataset") != dataset:
            continue
        out.append(data)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ruler_64k", "ruler_16k", "sharegpt"])
    parser.add_argument("--metric", default="slo_met_pct",
                        choices=["slo_met_pct", "ttft_p95", "tpot_p95",
                                 "throughput_tok_per_s"])
    args = parser.parse_args()

    runs = load_em1(args.dataset)
    if not runs:
        print(f"no e_m1 runs for dataset={args.dataset}")
        return 1

    # Group by (baseline, qps) → list of metric values.
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in runs:
        baseline = r.get("baseline")
        qps = r.get("arrival_rate_qps")
        if args.metric == "slo_met_pct":
            val = r.get("slo_met_pct")
        elif args.metric == "ttft_p95":
            val = r.get("ttft_ms", {}).get("p95")
        elif args.metric == "tpot_p95":
            val = r.get("tpot_ms", {}).get("p95")
        elif args.metric == "throughput_tok_per_s":
            val = r.get("throughput_tok_per_s")
        if val is not None:
            groups[(baseline, qps)].append(val)

    if not groups:
        print(f"no values to plot for metric={args.metric}")
        return 1

    fig, ax = plt.subplots(figsize=(6, 4))
    for baseline in BASELINE_ORDER:
        # Collect (qps, mean, std) for this baseline.
        qps_list = sorted(q for (b, q) in groups.keys() if b == baseline)
        if not qps_list:
            continue
        means: list[float] = []
        stds: list[float] = []
        for q in qps_list:
            vals = groups[(baseline, q)]
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        style = BASELINE_STYLE[baseline]
        ax.errorbar(qps_list, means, yerr=stds,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], capsize=3, linewidth=1.5,
                    markersize=6)

    ax.set_xlabel("Request rate (QPS)")
    metric_labels = {
        "slo_met_pct": "SLO attainment (%)",
        "ttft_p95": "TTFT P95 (ms)",
        "tpot_p95": "TPOT P95 (ms)",
        "throughput_tok_per_s": "Throughput (tok/s)",
    }
    ax.set_ylabel(metric_labels[args.metric])
    ax.set_title(f"E_M1: {metric_labels[args.metric]} vs QPS "
                 f"({args.dataset})")
    if args.metric == "slo_met_pct":
        ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = FIGURES_DIR / f"e_m1_{args.metric}_{args.dataset}.{ext}"
        fig.savefig(out, dpi=150)
        print(f"saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
