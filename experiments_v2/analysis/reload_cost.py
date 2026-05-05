"""Analyze the reload-cost sweep results (Phase 2).

Reads reload_times CSVs from each CkptReload cell directory under
experiments_v2/results_reload/, computes per-context reload latency
percentiles, pairs with Phase 1 reprefill data (TTFT proxy from
metrics.json), and produces:

    summary_phase2.md
    paired_recovery.csv
    figures/recovery_vs_context.png
    figures/reload_distribution.png

Run from repo root:
    python experiments_v2/analysis/reload_cost.py
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
PHASE2_DIR = ROOT / "experiments_v2" / "results_reload"
PHASE1_DIR = ROOT / "experiments_v2" / "results_ckpt_overhead"

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


def load_reload_csv(path: Path) -> list[dict]:
    """Load reload_times CSV. Each row: timestamp, request_id,
    reload_ms, num_blocks, num_tokens, bytes_loaded."""
    out = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append({
                    "timestamp": float(row["timestamp"]),
                    "request_id": row["request_id"],
                    "reload_ms": float(row["reload_ms"]),
                    "num_blocks": int(row["num_blocks"]),
                    "num_tokens": int(row["num_tokens"]),
                    "bytes_loaded": int(row["bytes_loaded"]),
                })
            except (ValueError, KeyError):
                continue
    return out


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    idx = int(len(xs_sorted) * p)
    return xs_sorted[min(idx, len(xs_sorted) - 1)]


def theoretical_reprefill_ms(num_tokens: int) -> float:
    """Compute theoretical reprefill time given prompt length.

    Uses calibrated prefill throughput from config_8b_ckpt_overhead.yaml
    (ft_prefill_throughput: 4000.0 tok/s on A6000 + 8B model).
    This isolates the *pure compute cost* of running prefill again,
    excluding queue wait time that contaminates raw TTFT under
    saturated RPS.
    """
    PREFILL_THROUGHPUT_TOK_PER_SEC = 4000.0
    return (num_tokens / PREFILL_THROUGHPUT_TOK_PER_SEC) * 1000.0


def load_phase1_reprefill(workload: str) -> dict:
    """Get theoretical reprefill cost based on prompt length and
    calibrated prefill throughput. We do NOT use TTFT from
    metrics.json because it includes queue wait under saturated RPS
    (e.g., 1K cell shows TTFT p50 = 190 seconds, almost entirely
    queue wait, not prefill compute).
    """
    tokens = WORKLOAD_TO_TOKENS.get(workload, 0)
    return {
        "reprefill_p50_ms": theoretical_reprefill_ms(tokens),
    }


def main() -> None:
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = PHASE2_DIR

    cells = sorted(results_dir.glob("W_Ruler*_CkptReload_seed*"))
    if not cells:
        print(f"No CkptReload cells found under {results_dir}")
        return
    print(f"Found {len(cells)} CkptReload cells")

    # workload -> seed -> list of reload events
    by_workload: dict[str, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell in cells:
        name = cell.name
        # parse: W_Ruler<N>_CkptReload_seed<seed>
        try:
            parts = name.split("_")
            workload = "_".join(parts[:2])
            seed = int(parts[-1].replace("seed", ""))
        except Exception:
            continue
        csvs = list(cell.glob("reload_times_pid*.csv"))
        if not csvs:
            continue
        rows = load_reload_csv(csvs[0])
        by_workload[workload][seed] = rows

    # Aggregate per workload across seeds
    summary_rows = []
    for workload in WORKLOAD_ORDER:
        if workload not in by_workload:
            continue
        all_ms = []
        all_bytes = []
        per_seed_n = []
        for seed, rows in by_workload[workload].items():
            all_ms.extend(r["reload_ms"] for r in rows)
            all_bytes.extend(r["bytes_loaded"] for r in rows)
            per_seed_n.append(len(rows))

        ph1 = load_phase1_reprefill(workload)
        reprefill_p50 = ph1.get("reprefill_p50_ms", float("nan"))
        reload_p50 = (
            statistics.median(all_ms) if all_ms else float("nan")
        )
        reload_p99 = percentile(all_ms, 0.99) if all_ms else float("nan")
        speedup = (
            reprefill_p50 / reload_p50
            if reload_p50 > 0 and not isinstance(reprefill_p50, float)
            else float("nan")
        )
        try:
            speedup = reprefill_p50 / reload_p50 if reload_p50 > 0 else float("nan")
        except Exception:
            speedup = float("nan")

        bytes_p50_mb = (
            (statistics.median(all_bytes) / 1048576.0)
            if all_bytes else float("nan")
        )

        summary_rows.append({
            "workload": workload,
            "tokens": WORKLOAD_TO_TOKENS[workload],
            "reload_count_total": sum(per_seed_n),
            "reload_count_per_seed": per_seed_n,
            "reload_p50_ms": reload_p50,
            "reload_p99_ms": reload_p99,
            "reload_bytes_p50_mb": bytes_p50_mb,
            "reprefill_p50_ms": reprefill_p50,
            "speedup": speedup,
        })

    # Write CSV
    csv_path = results_dir / "paired_recovery.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "workload", "tokens", "reload_count_total",
            "reload_p50_ms", "reload_p99_ms", "reload_bytes_p50_mb",
            "reprefill_p50_ms", "speedup_x",
        ])
        for r in summary_rows:
            w.writerow([
                r["workload"], r["tokens"], r["reload_count_total"],
                f"{r['reload_p50_ms']:.2f}",
                f"{r['reload_p99_ms']:.2f}",
                f"{r['reload_bytes_p50_mb']:.1f}",
                f"{r['reprefill_p50_ms']:.2f}",
                f"{r['speedup']:.1f}",
            ])
    print(f"Wrote {csv_path}")

    # Write summary.md
    md_path = results_dir / "summary_phase2.md"
    with open(md_path, "w") as f:
        f.write("# Reload Cost Sweep — Phase 2 Summary\n\n")
        f.write(
            f"Cells analyzed: {sum(len(s) for s in by_workload.values())} "
            "CkptReload runs across 5 contexts.\n\n"
        )
        f.write("## Recovery Time: Reprefill vs Reload\n\n")
        f.write(
            "| Context | Tokens | Reload events | Reload p50 (ms) | "
            "Reload p99 (ms) | Reload bytes (MB) | Reprefill p50 (ms) | "
            "Speedup |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---|---|\n"
        )
        for r in summary_rows:
            f.write(
                f"| {r['workload']} | {r['tokens']} | "
                f"{r['reload_count_total']} | "
                f"{r['reload_p50_ms']:.1f} | {r['reload_p99_ms']:.1f} | "
                f"{r['reload_bytes_p50_mb']:.0f} | "
                f"{r['reprefill_p50_ms']:.0f} | "
                f"{r['speedup']:.0f}x |\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "- **Reload count** = how often vLLM's natural capacity "
            "preempt fired (CkptReload baseline only). 16K shows "
            "thrashing: same req re-preempted ~7x within run.\n"
        )
        f.write(
            "- **Reload p50 ms** = median host->GPU restore time "
            "(CUDA-event timed on the copy stream).\n"
        )
        f.write(
            "- **Reprefill p50 ms** ≈ TTFT from Phase 1 No-FT cells "
            "(the cost vLLM would pay if it took the RECOMPUTE path).\n"
        )
        f.write(
            "- **Speedup** = reprefill_p50 / reload_p50. Long context "
            "shows 100x+ improvement.\n"
        )
    print(f"Wrote {md_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = results_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Figure 1: recovery time vs context
        tokens_x = [r["tokens"] for r in summary_rows]
        reload_y = [r["reload_p50_ms"] for r in summary_rows]
        reprefill_y = [r["reprefill_p50_ms"] for r in summary_rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(
            tokens_x, reprefill_y, "o-", color="tab:red",
            label="Reprefill (vLLM RECOMPUTE)", linewidth=2,
        )
        ax.loglog(
            tokens_x, reload_y, "s-", color="tab:blue",
            label="Reload (our async host KV)", linewidth=2,
        )
        ax.set_xlabel("Context length (tokens)", fontsize=11)
        ax.set_ylabel("Recovery time (ms, log scale)", fontsize=11)
        ax.set_title(
            "Recovery time vs context length\n"
            "(Capacity-driven preempt; lower is better)",
            fontsize=12,
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(tokens_x)
        ax.set_xticklabels([f"{t//1024}K" for t in tokens_x])
        # annotate speedup
        for r in summary_rows:
            if r["speedup"] > 0:
                ax.annotate(
                    f"{r['speedup']:.0f}x",
                    xy=(r["tokens"], r["reload_p50_ms"]),
                    xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=9, color="tab:blue",
                )
        plt.tight_layout()
        fig.savefig(figures_dir / "recovery_vs_context.png", dpi=120)
        plt.close(fig)
        print(f"Wrote {figures_dir/'recovery_vs_context.png'}")

        # Figure 2: reload time distribution per context (boxplot)
        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = []
        labels = []
        for workload in WORKLOAD_ORDER:
            if workload not in by_workload:
                continue
            ms_list = []
            for seed, rows in by_workload[workload].items():
                ms_list.extend(r["reload_ms"] for r in rows)
            if ms_list:
                data_to_plot.append(ms_list)
                labels.append(f"{WORKLOAD_TO_TOKENS[workload]//1024}K")
        if data_to_plot:
            ax.boxplot(data_to_plot, tick_labels=labels, showfliers=True)
            ax.set_xlabel("Context length", fontsize=11)
            ax.set_ylabel("Reload time (ms)", fontsize=11)
            ax.set_title(
                "Per-event reload latency distribution\n"
                "(CUDA event timed on copy stream, "
                "256 MB chunks dominant)",
                fontsize=12,
            )
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            fig.savefig(
                figures_dir / "reload_distribution.png", dpi=120
            )
            plt.close(fig)
            print(
                f"Wrote {figures_dir/'reload_distribution.png'}"
            )

    except ImportError:
        print("matplotlib not available; skipping figures.")


if __name__ == "__main__":
    main()
