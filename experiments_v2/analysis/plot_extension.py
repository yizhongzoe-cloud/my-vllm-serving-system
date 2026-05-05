"""Plot extension sweep results + cross-sweep comparison.

Three figures into experiments_v2/results_extension/figures/:
  1. Forward kernel time per step type, vanilla vs V3, across all contexts
  2. Step composition (decode vs prefill-chunk count) per cell
  3. Cell-level throughput (forward step counts) vanilla vs V3
"""
import csv
import glob
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/yzhong76/code/my-vllm-serving-system/experiments_v2")
EXT_DIR = ROOT / "results_extension"
MAIN_DIR = ROOT / "results_reload_v2"
FIG_DIR = EXT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_forward_times(cell_dir: Path) -> list[tuple[float, int]]:
    """Return list of (forward_ms, num_input_tokens)."""
    files = list(cell_dir.glob("forward_times_pid*.csv"))
    if not files:
        return []
    rows = []
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            try:
                rows.append((
                    float(row["forward_ms"]),
                    int(row["num_input_tokens"]),
                ))
            except (ValueError, KeyError):
                continue
    return rows


def aggregate(cells: list[Path]) -> dict:
    """Aggregate forward times across multiple cells (e.g. 3 seeds)."""
    decode_ms = []   # batch <= 100 tokens (decode-only)
    prefill_ms = []  # batch > 100 tokens (contains prefill chunk)
    for d in cells:
        rows = load_forward_times(d)
        for ms, nt in rows:
            if nt <= 100:
                decode_ms.append(ms)
            else:
                prefill_ms.append(ms)
    return {
        "decode_p50": statistics.median(decode_ms) if decode_ms else None,
        "decode_count": len(decode_ms),
        "prefill_p50": statistics.median(prefill_ms) if prefill_ms else None,
        "prefill_count": len(prefill_ms),
        "total_count": len(decode_ms) + len(prefill_ms),
    }


def find_cells(parent: Path, ctx_name: str, baseline: str) -> list[Path]:
    cells = []
    for seed in [42, 123, 456]:
        d = parent / f"{ctx_name}_{baseline}_seed{seed}"
        if d.exists():
            cells.append(d)
    return cells


# Combine main sweep (1K-32K high-RPS) + extension (1K low-RPS, 4K low, 64K)
# Strategy:
#  - "1K" point uses extension low-RPS data (more directly comparable to other low-RPS experiments)
#  - 4K, 8K, 16K, 32K from main sweep
#  - 64K from extension
data = [
    ("1K (low RPS)", EXT_DIR, "W_Ruler1K_Low"),
    ("4K (low RPS)", EXT_DIR, "W_Ruler4K_Low"),
    ("8K", MAIN_DIR, "W_Ruler8K"),
    ("16K", MAIN_DIR, "W_Ruler16K"),
    ("32K", MAIN_DIR, "W_Ruler32K"),
    ("64K", EXT_DIR, "W_Ruler64K"),
]


def fig_step_times():
    """Figure: forward kernel time per step type, vanilla vs V3."""
    labels = []
    vanilla_decode = []
    v3_decode = []
    vanilla_prefill = []
    v3_prefill = []
    for label, parent, ctx_name in data:
        labels.append(label)
        v_cells = find_cells(parent, ctx_name, "NoFT-Reprefill")
        c_cells = find_cells(parent, ctx_name, "CkptReload")
        v_agg = aggregate(v_cells)
        c_agg = aggregate(c_cells)
        vanilla_decode.append(v_agg["decode_p50"] or 0)
        v3_decode.append(c_agg["decode_p50"] or 0)
        vanilla_prefill.append(v_agg["prefill_p50"] or 0)
        v3_prefill.append(c_agg["prefill_p50"] or 0)

    x = np.arange(len(labels))
    width = 0.2

    fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(14, 5))

    # Decode-only steps
    ax_d.bar(x - width / 2, vanilla_decode, width,
             label="vanilla vLLM", color="tab:gray")
    ax_d.bar(x + width / 2, v3_decode, width,
             label="V3 CkptReload", color="tab:blue")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=15)
    ax_d.set_ylabel("p50 forward kernel time (ms)")
    ax_d.set_title("Decode-only step time per context")
    ax_d.legend()
    ax_d.grid(True, axis="y", alpha=0.3)
    # Annotate ratios
    for i, (v, c) in enumerate(zip(vanilla_decode, v3_decode)):
        if v > 0 and c > 0:
            ratio = c / v
            color = "tab:red" if ratio > 1.5 else "tab:green"
            ax_d.annotate(
                f"{ratio:.1f}x",
                xy=(x[i] + width / 2, c),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", fontsize=9, color=color,
            )

    # Prefill-chunk steps
    ax_p.bar(x - width / 2, vanilla_prefill, width,
             label="vanilla vLLM", color="tab:gray")
    ax_p.bar(x + width / 2, v3_prefill, width,
             label="V3 CkptReload", color="tab:blue")
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(labels, rotation=15)
    ax_p.set_ylabel("p50 forward kernel time (ms)")
    ax_p.set_title("Step containing prefill chunk per context")
    ax_p.legend()
    ax_p.grid(True, axis="y", alpha=0.3)
    for i, (v, c) in enumerate(zip(vanilla_prefill, v3_prefill)):
        if v > 0 and c > 0:
            ratio = c / v
            color = "tab:red" if ratio > 1.5 else "tab:green"
            ax_p.annotate(
                f"{ratio:.1f}x",
                xy=(x[i] + width / 2, c),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", fontsize=9, color=color,
            )

    fig.suptitle(
        "GPU forward kernel time per step type — vanilla vs V3\n"
        "ratios in red/green: V3 / vanilla. >1.5x means V3 step is significantly slower."
    )
    plt.tight_layout()
    out = FIG_DIR / "step_kernel_time.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_step_composition():
    """Figure: how many decode-only vs prefill-chunk steps per cell."""
    labels = []
    vanilla_decode_n = []
    vanilla_prefill_n = []
    v3_decode_n = []
    v3_prefill_n = []
    for label, parent, ctx_name in data:
        labels.append(label)
        v_cells = find_cells(parent, ctx_name, "NoFT-Reprefill")
        c_cells = find_cells(parent, ctx_name, "CkptReload")
        v_agg = aggregate(v_cells)
        c_agg = aggregate(c_cells)
        # Average over 3 seeds
        n_seeds_v = max(len(v_cells), 1)
        n_seeds_c = max(len(c_cells), 1)
        vanilla_decode_n.append(v_agg["decode_count"] / n_seeds_v)
        vanilla_prefill_n.append(v_agg["prefill_count"] / n_seeds_v)
        v3_decode_n.append(c_agg["decode_count"] / n_seeds_c)
        v3_prefill_n.append(c_agg["prefill_count"] / n_seeds_c)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    # Stacked bars: decode (bottom) + prefill (top)
    ax.bar(x - width / 2, vanilla_decode_n, width,
           label="vanilla decode-only steps",
           color="tab:gray", alpha=0.6)
    ax.bar(x - width / 2, vanilla_prefill_n, width,
           bottom=vanilla_decode_n,
           label="vanilla prefill-chunk steps",
           color="tab:gray", alpha=0.95)
    ax.bar(x + width / 2, v3_decode_n, width,
           label="V3 decode-only steps",
           color="tab:blue", alpha=0.6)
    ax.bar(x + width / 2, v3_prefill_n, width,
           bottom=v3_decode_n,
           label="V3 prefill-chunk steps",
           color="tab:blue", alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Avg step count per seed")
    ax.set_title(
        "Step composition: decode-only vs prefill-chunk steps per cell\n"
        "(grey = vanilla vLLM, blue = V3 CkptReload)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "step_composition.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_throughput_combined():
    """Figure: cell-level throughput across all 6 contexts."""
    labels = []
    vanilla_means = []
    vanilla_stds = []
    v3_means = []
    v3_stds = []
    for label, parent, ctx_name in data:
        labels.append(label)
        v_counts, c_counts = [], []
        for seed in [42, 123, 456]:
            v_files = glob.glob(
                str(parent / f"{ctx_name}_NoFT-Reprefill_seed{seed}"
                            / "forward_times_pid*.csv")
            )
            c_files = glob.glob(
                str(parent / f"{ctx_name}_CkptReload_seed{seed}"
                            / "forward_times_pid*.csv")
            )
            if v_files:
                with open(v_files[0]) as f:
                    v_counts.append(sum(1 for _ in f) - 1)
            if c_files:
                with open(c_files[0]) as f:
                    c_counts.append(sum(1 for _ in f) - 1)
        vanilla_means.append(statistics.mean(v_counts) if v_counts else 0)
        vanilla_stds.append(
            statistics.stdev(v_counts) if len(v_counts) > 1 else 0
        )
        v3_means.append(statistics.mean(c_counts) if c_counts else 0)
        v3_stds.append(
            statistics.stdev(c_counts) if len(c_counts) > 1 else 0
        )

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, vanilla_means, width,
           yerr=vanilla_stds, label="vanilla vLLM",
           color="tab:gray", capsize=4)
    ax.bar(x + width / 2, v3_means, width,
           yerr=v3_stds, label="V3 CkptReload",
           color="tab:blue", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Forward step count (cell-level throughput)")
    ax.set_title(
        "Cell-level throughput: vanilla vs V3 across all contexts\n"
        "(combines main sweep + extension sweep)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, (v, c) in enumerate(zip(vanilla_means, v3_means)):
        if v > 0:
            pct = (c - v) / v * 100
            color = "tab:red" if pct < -20 else "tab:green"
            ax.annotate(
                f"{pct:+.0f}%",
                xy=(x[i] + width / 2, c),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", fontsize=9, color=color,
            )

    plt.tight_layout()
    out = FIG_DIR / "throughput_all_contexts.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    fig_step_times()
    fig_step_composition()
    fig_throughput_combined()


if __name__ == "__main__":
    main()
