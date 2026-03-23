#!/usr/bin/env python3
"""Fault-Tolerant Benchmark Visualization.

Reads data collected by collect_ft_data.py and generates publication-quality
plots for analyzing FT serving performance.

Generated plots:
  1. Throughput comparison (baseline vs failure scenarios)
  2. Latency CDF (TTFT and E2E)
  3. Per-request timeline (latency over time, with failure event marked)
  4. Failover gap bar chart
  5. Success rate & rerouting summary
  6. Per-replica request distribution

Usage:
    # Plot from a specific results JSON
    python slo_benchmark/scripts/plot_ft_results.py -i slo_benchmark/data/results/full_results_XXXX.json

    # Plot from the latest results in default directory
    python slo_benchmark/scripts/plot_ft_results.py

    # Specify output directory for plots
    python slo_benchmark/scripts/plot_ft_results.py -o plots/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# Style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "baseline": "#2196F3",
    "failure": "#F44336",
    "rerouted": "#FF9800",
    "replica0": "#4CAF50",
    "replica1": "#9C27B0",
    "success": "#4CAF50",
    "fail": "#F44336",
}


def load_data(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load full_results JSON → (summaries_df, requests_df, metadata)."""
    with open(input_path) as f:
        data = json.load(f)

    summaries_df = pd.DataFrame(data["summaries"])
    requests_df = pd.DataFrame(data["per_request"])
    metadata = data["metadata"]
    return summaries_df, requests_df, metadata


# ---------------------------------------------------------------------------
# Plot 1: Throughput comparison
# ---------------------------------------------------------------------------

def plot_throughput(summaries: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by experiment type
    baseline = summaries[summaries["experiment_type"] == "baseline"]
    failure = summaries[summaries["experiment_type"] == "failure"]

    # (a) Goodput (tok/s)
    ax = axes[0]
    x_bl = range(len(baseline))
    x_fl = range(len(baseline), len(baseline) + len(failure))

    ax.bar(x_bl, baseline["goodput_tok_per_sec"],
           color=COLORS["baseline"], label="Baseline (no failure)", alpha=0.85)
    ax.bar(x_fl, failure["goodput_tok_per_sec"],
           color=COLORS["failure"], label="With GPU failure", alpha=0.85)

    labels = []
    for _, row in baseline.iterrows():
        labels.append(f"n={int(row['num_requests'])}")
    for _, row in failure.iterrows():
        labels.append(f"n={int(row['num_requests'])}\nkill@{row['kill_after_sec']:.0f}s")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Goodput (tok/s)")
    ax.set_title("(a) Goodput Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) Request throughput (req/s)
    ax = axes[1]
    ax.bar(x_bl, baseline["request_throughput_rps"],
           color=COLORS["baseline"], label="Baseline", alpha=0.85)
    ax.bar(x_fl, failure["request_throughput_rps"],
           color=COLORS["failure"], label="With GPU failure", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Request Throughput (req/s)")
    ax.set_title("(b) Request Throughput Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Throughput: Baseline vs. Failure Scenarios", fontsize=15, y=1.02)
    plt.tight_layout()
    path = output_dir / "01_throughput_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [1/6] {path}")


# ---------------------------------------------------------------------------
# Plot 2: Latency CDF
# ---------------------------------------------------------------------------

def plot_latency_cdf(requests: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ok = requests[requests["success"] == True]

    for ax, col, title in [
        (axes[0], "ttft_ms", "(a) TTFT CDF"),
        (axes[1], "e2e_latency_ms", "(b) E2E Latency CDF"),
    ]:
        for exp_type, color, label in [
            ("baseline", COLORS["baseline"], "Baseline"),
            ("failure", COLORS["failure"], "With Failure"),
        ]:
            vals = ok[ok["experiment_type"] == exp_type][col].dropna().sort_values()
            if len(vals) == 0:
                continue
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=color, label=label, linewidth=2)

        # Also plot rerouted requests specifically
        rerouted_vals = ok[(ok["was_rerouted"] == True)][col].dropna().sort_values()
        if len(rerouted_vals) > 0:
            cdf = np.arange(1, len(rerouted_vals) + 1) / len(rerouted_vals)
            ax.plot(rerouted_vals, cdf, color=COLORS["rerouted"],
                    label="Rerouted", linewidth=2, linestyle="--")

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Latency Distribution", fontsize=15, y=1.02)
    plt.tight_layout()
    path = output_dir / "02_latency_cdf.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [2/6] {path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-request timeline
# ---------------------------------------------------------------------------

def plot_request_timeline(requests: pd.DataFrame, summaries: pd.DataFrame,
                          output_dir: Path):
    # Pick the failure experiment with the most requests
    failure_exps = summaries[summaries["experiment_type"] == "failure"]
    if failure_exps.empty:
        print("  [3/6] Skipped (no failure experiments)")
        return

    best_exp = failure_exps.loc[failure_exps["num_requests"].idxmax()]
    exp_id = best_exp["experiment_id"]
    kill_time = best_exp["kill_after_sec"]

    exp_requests = requests[requests["experiment_id"] == exp_id].copy()
    # Extract request index from request_id
    exp_requests["req_idx"] = exp_requests["request_id"].str.extract(r"req-(\d+)").astype(int)
    exp_requests = exp_requests.sort_values("req_idx")

    fig, ax = plt.subplots(figsize=(12, 5))

    ok = exp_requests[exp_requests["success"] == True]
    fail = exp_requests[exp_requests["success"] == False]
    rerouted = ok[ok["was_rerouted"] == True]
    normal = ok[ok["was_rerouted"] == False]

    # Scatter: x = request index, y = e2e latency
    if not normal.empty:
        colors_arr = normal["replica_id"].map({0: COLORS["replica0"], 1: COLORS["replica1"]})
        ax.scatter(normal["req_idx"], normal["e2e_latency_ms"],
                   c=colors_arr, s=50, alpha=0.7, edgecolors="white", linewidth=0.5,
                   zorder=3)
    if not rerouted.empty:
        ax.scatter(rerouted["req_idx"], rerouted["e2e_latency_ms"],
                   c=COLORS["rerouted"], s=80, marker="D", alpha=0.9,
                   edgecolors="black", linewidth=0.8, label="Rerouted", zorder=4)
    if not fail.empty:
        ax.scatter(fail["req_idx"], [0] * len(fail),
                   c=COLORS["fail"], s=40, marker="x", alpha=0.8,
                   label="Failed", zorder=3)

    # Mark kill time
    if kill_time is not None:
        kill_req_idx = kill_time / 0.5  # Requests sent 0.5s apart
        ax.axvline(x=kill_req_idx, color="red", linestyle="--", alpha=0.7, linewidth=2)
        ax.text(kill_req_idx + 0.3, ax.get_ylim()[1] * 0.9,
                f"GPU 1 killed\n(t={kill_time:.0f}s)",
                color="red", fontsize=10, va="top")

    # Legend
    handles = [
        mpatches.Patch(color=COLORS["replica0"], label="Replica 0 (GPU 0)"),
        mpatches.Patch(color=COLORS["replica1"], label="Replica 1 (GPU 1)"),
    ]
    if not rerouted.empty:
        handles.append(plt.Line2D([0], [0], marker="D", color=COLORS["rerouted"],
                                  label="Rerouted", markersize=8, linestyle=""))
    if not fail.empty:
        handles.append(plt.Line2D([0], [0], marker="x", color=COLORS["fail"],
                                  label="Failed", markersize=8, linestyle=""))
    ax.legend(handles=handles, loc="upper left")

    ax.set_xlabel("Request Index")
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title(f"Per-Request Timeline — {exp_id}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / "03_request_timeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [3/6] {path}")


# ---------------------------------------------------------------------------
# Plot 4: Failover gap
# ---------------------------------------------------------------------------

def plot_failover_gap(summaries: pd.DataFrame, output_dir: Path):
    failure = summaries[
        (summaries["experiment_type"] == "failure") &
        (summaries["failover_gap_ms"].notna())
    ].copy()

    if failure.empty:
        print("  [4/6] Skipped (no failover gap data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    failure = failure.sort_values("kill_after_sec")

    labels = [f"n={int(r['num_requests'])}\nkill@{r['kill_after_sec']:.0f}s"
              for _, r in failure.iterrows()]
    bars = ax.bar(range(len(failure)), failure["failover_gap_ms"],
                  color=COLORS["rerouted"], alpha=0.85, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, failure["failover_gap_ms"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Failover Gap (ms)")
    ax.set_title("Failover Gap: Time from Failure Detection to First Rerouted Response")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "04_failover_gap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [4/6] {path}")


# ---------------------------------------------------------------------------
# Plot 5: Success rate & rerouting
# ---------------------------------------------------------------------------

def plot_success_rate(summaries: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Success rate
    ax = axes[0]
    for exp_type, color in [("baseline", COLORS["baseline"]),
                             ("failure", COLORS["failure"])]:
        subset = summaries[summaries["experiment_type"] == exp_type]
        labels = []
        for _, r in subset.iterrows():
            lbl = f"n={int(r['num_requests'])}"
            if r["kill_after_sec"] is not None and not pd.isna(r["kill_after_sec"]):
                lbl += f"\nkill@{r['kill_after_sec']:.0f}s"
            labels.append(lbl)
        x = range(len(subset))
        ax.bar([i + (0.2 if exp_type == "failure" else -0.2) for i in
                range(len(subset))],
               subset["success_rate"] * 100,
               width=0.35, color=color, alpha=0.85,
               label="Baseline" if exp_type == "baseline" else "With Failure")

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(a) Request Success Rate")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) Rerouted / Failed / Completed stacked bar for failure experiments
    ax = axes[1]
    failure = summaries[summaries["experiment_type"] == "failure"].reset_index(drop=True)
    if not failure.empty:
        labels = [f"n={int(r['num_requests'])}\nkill@{r['kill_after_sec']:.0f}s"
                  for _, r in failure.iterrows()]
        x = range(len(failure))
        ax.bar(x, failure["completed"] - failure["rerouted"],
               color=COLORS["success"], label="Completed (direct)", alpha=0.85)
        ax.bar(x, failure["rerouted"], bottom=failure["completed"] - failure["rerouted"],
               color=COLORS["rerouted"], label="Completed (rerouted)", alpha=0.85)
        ax.bar(x, failure["failed"], bottom=failure["completed"],
               color=COLORS["fail"], label="Failed", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Requests")
    ax.set_title("(b) Request Outcome Breakdown (Failure Experiments)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Success Rate & Request Outcomes", fontsize=15, y=1.02)
    plt.tight_layout()
    path = output_dir / "05_success_rate.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [5/6] {path}")


# ---------------------------------------------------------------------------
# Plot 6: Per-replica distribution
# ---------------------------------------------------------------------------

def plot_replica_distribution(requests: pd.DataFrame, summaries: pd.DataFrame,
                              output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Request count per replica across experiments
    ax = axes[0]
    exp_ids = summaries["experiment_id"].tolist()
    r0_counts = []
    r1_counts = []
    for eid in exp_ids:
        exp_req = requests[requests["experiment_id"] == eid]
        r0_counts.append(len(exp_req[exp_req["replica_id"] == 0]))
        r1_counts.append(len(exp_req[exp_req["replica_id"] == 1]))

    x = np.arange(len(exp_ids))
    width = 0.35
    ax.bar(x - width/2, r0_counts, width, color=COLORS["replica0"],
           label="Replica 0 (GPU 0)", alpha=0.85)
    ax.bar(x + width/2, r1_counts, width, color=COLORS["replica1"],
           label="Replica 1 (GPU 1)", alpha=0.85)

    short_labels = []
    for _, r in summaries.iterrows():
        lbl = f"{'B' if r['experiment_type'] == 'baseline' else 'F'}"
        lbl += f"-n{int(r['num_requests'])}"
        if r['kill_after_sec'] is not None and not pd.isna(r['kill_after_sec']):
            lbl += f"-k{r['kill_after_sec']:.0f}"
        short_labels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Requests Handled")
    ax.set_title("(a) Requests per Replica")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) Latency per replica (box plot for largest failure experiment)
    ax = axes[1]
    failure_exps = summaries[summaries["experiment_type"] == "failure"]
    if not failure_exps.empty:
        best_exp = failure_exps.loc[failure_exps["num_requests"].idxmax()]
        exp_id = best_exp["experiment_id"]
        exp_req = requests[(requests["experiment_id"] == exp_id) &
                           (requests["success"] == True)]

        data_to_plot = []
        labels_bp = []
        for rid, color in [(0, COLORS["replica0"]), (1, COLORS["replica1"])]:
            vals = exp_req[exp_req["replica_id"] == rid]["e2e_latency_ms"].dropna()
            if len(vals) > 0:
                data_to_plot.append(vals.values)
                labels_bp.append(f"Replica {rid}")

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, tick_labels=labels_bp, patch_artist=True)
            box_colors = [COLORS["replica0"], COLORS["replica1"]][:len(data_to_plot)]
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("(b) Latency by Replica (largest failure exp)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Replica Analysis", fontsize=15, y=1.02)
    plt.tight_layout()
    path = output_dir / "06_replica_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [6/6] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_results(results_dir: Path) -> Path | None:
    """Find the most recent full_results_*.json file."""
    files = sorted(results_dir.glob("full_results_*.json"))
    return files[-1] if files else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot FT benchmark results"
    )
    parser.add_argument("-i", "--input", type=str,
                        help="Path to full_results_*.json file")
    parser.add_argument("-o", "--output", type=str,
                        default="slo_benchmark/data/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    # Find input file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_results(Path("slo_benchmark/data/results"))
        if input_path is None:
            print("No results found. Run collect_ft_data.py first.")
            sys.exit(1)

    print(f"Loading: {input_path}")
    summaries, requests, metadata = load_data(input_path)
    print(f"  {len(summaries)} experiments, {len(requests)} total requests")
    print(f"  Model: {metadata['model']}")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots → {output_dir}/")
    plot_throughput(summaries, output_dir)
    plot_latency_cdf(requests, output_dir)
    plot_request_timeline(requests, summaries, output_dir)
    plot_failover_gap(summaries, output_dir)
    plot_success_rate(summaries, output_dir)
    plot_replica_distribution(requests, summaries, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
