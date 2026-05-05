"""Analyze the ckpt overhead sweep results.

Reads forward_times CSVs from each cell directory under
experiments_v2/results_ckpt_overhead/, computes paired (No-FT vs CkptOnly) overhead
per (workload, seed), and produces:

    summary.md
    paired_diff.csv
    figures/scaling_overhead.png
    figures/forward_time_distribution.png

The analysis intentionally:
  - Drops the first WARMUP_SEC of forward times (CUDA JIT, allocator
    cold-start, ckpt buffer warming).
  - Pairs strictly by (workload, seed): No-FT and CkptOnly with the
    same seed see the same Poisson trace with the same prompts.
  - Uses median + p99 of forward_ms (robust to outliers).
  - Reports paired ΔForward% with mean ± std across seeds.

Run from repo root:
    python experiments_v2/analysis/ckpt_overhead.py [results_dir]
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
DEFAULT_RESULTS_DIR = ROOT / "experiments_v2" / "results_ckpt_overhead"

WORKLOAD_ORDER = [
    "W_Ruler1K",
    "W_Ruler4K",
    "W_Ruler8K",
    "W_Ruler16K",
    "W_Ruler32K",
]
WORKLOAD_TO_TOKENS = {
    "W_Ruler1K": 1024,
    "W_Ruler4K": 4096,
    "W_Ruler8K": 8192,
    "W_Ruler16K": 16384,
    "W_Ruler32K": 32768,
}

# Drop forward times during these many seconds at the start of each
# cell — these include CUDA JIT, allocator cold-start, and (for
# CkptOnly) initial ckpt buffer settling.
WARMUP_DROP_SEC = 60.0


def load_forward_times(cell_dir: Path) -> list[tuple[float, float, int]]:
    """Return list of (timestamp, forward_ms, num_input_tokens)."""
    rows = []
    for csv_path in cell_dir.glob("forward_times_pid*.csv"):
        with open(csv_path) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                try:
                    rows.append(
                        (
                            float(row["timestamp"]),
                            float(row["forward_ms"]),
                            int(row["num_input_tokens"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
    return rows


def load_ckpt_stats(cell_dir: Path) -> dict:
    """Aggregate ckpt fire stats."""
    fires = 0
    delta_fires = 0
    full_fires = 0
    bytes_total = 0
    bytes_per_fire = []
    for csv_path in cell_dir.glob("ckpt_stats_pid*.csv"):
        with open(csv_path) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                try:
                    fires += 1
                    if row["mode"] == "delta":
                        delta_fires += 1
                    else:
                        full_fires += 1
                    b = int(row["bytes_written"])
                    bytes_total += b
                    bytes_per_fire.append(b)
                except (KeyError, ValueError):
                    continue
    return {
        "fires": fires,
        "delta_fires": delta_fires,
        "full_fires": full_fires,
        "bytes_total": bytes_total,
        "bytes_p50": (
            statistics.median(bytes_per_fire) if bytes_per_fire else 0
        ),
        "bytes_max": max(bytes_per_fire) if bytes_per_fire else 0,
    }


def filter_steady_state(rows, warmup_drop_sec=WARMUP_DROP_SEC):
    """Drop rows within first `warmup_drop_sec` of cell start."""
    if not rows:
        return rows
    t_start = min(r[0] for r in rows)
    cutoff = t_start + warmup_drop_sec
    return [r for r in rows if r[0] >= cutoff]


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def main(results_dir: Path = DEFAULT_RESULTS_DIR) -> None:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover cells
    cell_dirs = []
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        # Expected: <Workload>_<Baseline>_seed<N>
        parts = child.name.split("_")
        if len(parts) < 4:
            continue
        # Workload starts with W_Ruler
        if not child.name.startswith("W_Ruler"):
            continue
        cell_dirs.append(child)

    print(f"Found {len(cell_dirs)} cell dirs in {results_dir}")

    # Group: (workload, baseline, seed) -> aggregated stats
    cells = {}
    for cd in cell_dirs:
        # Parse name: e.g. "W_Ruler16K_CkptOnly_seed42"
        # Workload is "W_RulerXK", followed by baseline, then "seedN"
        name = cd.name
        # find last "_seed" token
        if "_seed" not in name:
            continue
        head, seed_part = name.rsplit("_seed", 1)
        try:
            seed = int(seed_part)
        except ValueError:
            continue
        # head is "W_RulerXK_<Baseline>"
        for wl in WORKLOAD_ORDER:
            if head.startswith(wl + "_"):
                workload = wl
                baseline = head[len(wl) + 1:]
                break
        else:
            continue

        rows = load_forward_times(cd)
        rows_steady = filter_steady_state(rows)
        forward_ms = [r[1] for r in rows_steady]

        ckpt = load_ckpt_stats(cd)

        cells[(workload, baseline, seed)] = {
            "n_total": len(rows),
            "n_steady": len(rows_steady),
            "p50_ms": percentile(forward_ms, 50),
            "p95_ms": percentile(forward_ms, 95),
            "p99_ms": percentile(forward_ms, 99),
            "mean_ms": statistics.mean(forward_ms) if forward_ms else float("nan"),
            "ckpt": ckpt,
        }

    # Paired diff per (workload, seed)
    paired = defaultdict(dict)
    for (workload, baseline, seed), stats in cells.items():
        paired[(workload, seed)][baseline] = stats

    # Compute ΔForward% across seeds, per workload
    out_rows = []
    summary_rows = []  # for markdown
    for workload in WORKLOAD_ORDER:
        deltas_p50 = []
        deltas_p95 = []
        deltas_p99 = []
        ckpt_fires_total = []
        for (wl, seed), bdict in sorted(paired.items()):
            if wl != workload:
                continue
            if "No-FT" not in bdict or "CkptOnly" not in bdict:
                continue
            a = bdict["No-FT"]
            b = bdict["CkptOnly"]
            if a["p50_ms"] > 0:
                d50 = (b["p50_ms"] - a["p50_ms"]) / a["p50_ms"] * 100
                deltas_p50.append(d50)
            if a["p95_ms"] > 0:
                d95 = (b["p95_ms"] - a["p95_ms"]) / a["p95_ms"] * 100
                deltas_p95.append(d95)
            if a["p99_ms"] > 0:
                d99 = (b["p99_ms"] - a["p99_ms"]) / a["p99_ms"] * 100
                deltas_p99.append(d99)
            ckpt_fires_total.append(b["ckpt"]["fires"])
            out_rows.append(
                {
                    "workload": workload,
                    "context_tokens": WORKLOAD_TO_TOKENS[workload],
                    "seed": seed,
                    "noft_p50_ms": f"{a['p50_ms']:.3f}",
                    "ckpt_p50_ms": f"{b['p50_ms']:.3f}",
                    "delta_p50_pct": (
                        f"{d50:.3f}" if a["p50_ms"] > 0 else "NaN"
                    ),
                    "noft_p99_ms": f"{a['p99_ms']:.3f}",
                    "ckpt_p99_ms": f"{b['p99_ms']:.3f}",
                    "delta_p99_pct": (
                        f"{d99:.3f}" if a["p99_ms"] > 0 else "NaN"
                    ),
                    "ckpt_fires": b["ckpt"]["fires"],
                    "ckpt_delta_fires": b["ckpt"]["delta_fires"],
                    "ckpt_full_fires": b["ckpt"]["full_fires"],
                    "ckpt_bytes_p50_mb": (
                        f"{b['ckpt']['bytes_p50']/1048576:.3f}"
                    ),
                }
            )

        if deltas_p50:
            mean_d50 = statistics.mean(deltas_p50)
            std_d50 = (
                statistics.stdev(deltas_p50) if len(deltas_p50) > 1 else 0.0
            )
            mean_d99 = statistics.mean(deltas_p99) if deltas_p99 else 0.0
            std_d99 = (
                statistics.stdev(deltas_p99) if len(deltas_p99) > 1 else 0.0
            )
            summary_rows.append(
                {
                    "workload": workload,
                    "tokens": WORKLOAD_TO_TOKENS[workload],
                    "n_seeds": len(deltas_p50),
                    "mean_d50": mean_d50,
                    "std_d50": std_d50,
                    "mean_d99": mean_d99,
                    "std_d99": std_d99,
                    "ckpt_fires_mean": (
                        statistics.mean(ckpt_fires_total)
                        if ckpt_fires_total else 0
                    ),
                }
            )

    # Write paired_diff.csv
    out_csv = results_dir / "paired_diff.csv"
    if out_rows:
        with open(out_csv, "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print(f"Wrote {out_csv}")

    # Write summary.md
    md = ["# Ckpt Overhead Sweep — Summary\n"]
    md.append(f"Cells analyzed: {len(out_rows)} paired (No-FT, CkptOnly) ")
    md.append(f"observations across {len(summary_rows)} contexts.\n\n")
    md.append("## Per-Context Overhead (mean ± std across seeds)\n\n")
    md.append("| Context | Tokens | Seeds | ΔForward p50 % | ΔForward p99 % | Ckpt fires (mean) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for r in summary_rows:
        md.append(
            f"| {r['workload']} | {r['tokens']} | {r['n_seeds']} | "
            f"{r['mean_d50']:+.2f} ± {r['std_d50']:.2f} | "
            f"{r['mean_d99']:+.2f} ± {r['std_d99']:.2f} | "
            f"{r['ckpt_fires_mean']:.0f} |\n"
        )
    md.append("\n## Interpretation\n\n")
    md.append("- **ΔForward p50 / p99 < 5%**: TokenFlow's claim that async KV ")
    md.append("transfer overhead is negligible **holds for our setup** in ")
    md.append("this context-length regime. Mechanism is sound.\n")
    md.append("- **5-15%**: edge case — likely PCIe back-pressure exerting ")
    md.append("partial pressure on SMs. Worth engineering optimization ")
    md.append("(pinned memory, async copy stream tuning).\n")
    md.append("- **>15%**: PCIe contention is meaningfully squeezing GPU ")
    md.append("compute. Investigate before claiming the mechanism.\n\n")
    md.append("## Paired ΔForward by Seed (raw)\n\n")
    md.append("See `paired_diff.csv` for full detail.\n")

    summary_path = results_dir / "summary.md"
    with open(summary_path, "w") as fout:
        fout.writelines(md)
    print(f"Wrote {summary_path}")

    # Try to plot if matplotlib is available
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import EngFormatter

        fig_dir = results_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        if summary_rows:
            xs = [r["tokens"] for r in summary_rows]
            d50_means = [r["mean_d50"] for r in summary_rows]
            d50_stds = [r["std_d50"] for r in summary_rows]
            d99_means = [r["mean_d99"] for r in summary_rows]
            d99_stds = [r["std_d99"] for r in summary_rows]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.errorbar(
                xs, d50_means, yerr=d50_stds, marker="o",
                label="ΔForward p50%", capsize=3,
            )
            ax.errorbar(
                xs, d99_means, yerr=d99_stds, marker="s",
                label="ΔForward p99%", capsize=3,
            )
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.axhline(5, color="green", linewidth=0.5, linestyle="--",
                       alpha=0.5, label="5% threshold")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Context length (tokens)")
            ax.set_ylabel("Δ Forward kernel time (%)")
            ax.set_title(
                "Async KV checkpoint overhead vs context length\n"
                "(No-FT vs CkptOnly, paired)"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(EngFormatter())
            fig.tight_layout()
            fig.savefig(fig_dir / "scaling_overhead.png", dpi=150)
            plt.close(fig)
            print(f"Wrote {fig_dir / 'scaling_overhead.png'}")
    except ImportError:
        print("matplotlib not available; skipping plots.")
        pass


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS_DIR
    main(Path(rd))
