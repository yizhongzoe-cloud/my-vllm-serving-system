"""Plot V3 sweep results.

Three figures:
  1. Timeline overlap: forward windows and reload windows on the same
     time axis, showing they run concurrently (true overlap evidence).
  2. GPU utilization timeline: gpu_util.csv plotted over time, with
     reload events marked.
  3. Cell-level throughput: NoFT vs CkptReload forward step counts
     vs context length (paired bar chart).

Run from repo root:
    python experiments_v2/analysis/plot_v3.py
"""
import csv
import glob
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
RESULTS_DIR = ROOT / "experiments_v2" / "results_reload_v2"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def find_cell_csv(cell_name: str, prefix: str) -> Path | None:
    matches = list((RESULTS_DIR / cell_name).glob(f"{prefix}_pid*.csv"))
    return matches[0] if matches else None


def fig_timeline_overlap():
    """Figure 1: forward and reload time windows overlap on the time axis.

    We pick the 32K seed42 cell (most reload events) and zoom into a
    30-second window with dense reload activity.
    """
    cell = "W_Ruler32K_CkptReload_seed42"
    forward_csv = find_cell_csv(cell, "forward_times")
    reload_csv = find_cell_csv(cell, "reload_times")
    if forward_csv is None or reload_csv is None:
        print(f"Missing CSVs for {cell}")
        return

    forward = load_csv(forward_csv)
    reload_evt = load_csv(reload_csv)

    if not reload_evt:
        print(f"No reload events in {cell}")
        return

    # Show full cell. Use first event timestamp as t=0.
    all_ts = [float(f["timestamp"]) for f in forward] + [
        float(r["timestamp"]) for r in reload_evt
    ]
    win_start = min(all_ts)
    win_end = max(all_ts) + 1.0

    forward_in = forward
    reload_in = reload_evt

    fig, ax = plt.subplots(figsize=(16, 4))

    # Forward bars on the upper row.
    y_forward = 1.0
    for f in forward_in:
        t0 = float(f["timestamp"]) - win_start
        dur_s = float(f["forward_ms"]) / 1000.0
        ax.add_patch(
            Rectangle(
                (t0, y_forward), dur_s, 0.4,
                facecolor="tab:blue", alpha=0.8, edgecolor="none",
            )
        )

    # Reload bars on the lower row. Add a small marker (vertical line)
    # for each event in addition to the duration bar — duration bars
    # are sub-second so invisible at full-cell scale.
    y_reload = 0.3
    for r in reload_in:
        t0 = float(r["timestamp"]) - win_start
        dur_s = float(r["reload_ms"]) / 1000.0
        ax.add_patch(
            Rectangle(
                (t0, y_reload), max(dur_s, 0.1), 0.4,
                facecolor="tab:orange", alpha=0.8, edgecolor="none",
            )
        )
        # Vertical marker through both rows so reload events are visible
        # at any zoom level.
        ax.axvline(t0, color="tab:orange", alpha=0.35, linewidth=0.8,
                   ymin=0.0, ymax=0.95, zorder=0)

    ax.set_xlim(0, win_end - win_start)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([y_forward + 0.2, y_reload + 0.2])
    ax.set_yticklabels(["Default stream\n(forward)", "Copy stream\n(reload)"])
    ax.set_xlabel(f"Time (seconds, full cell)")
    ax.set_title(
        f"V3: Default stream and copy stream concurrent activity\n"
        f"({cell}, {len(forward_in)} forward steps, "
        f"{len(reload_in)} reload events total)"
    )
    ax.grid(True, axis="x", alpha=0.3)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="tab:blue", alpha=0.8,
                  label="Forward (default stream)"),
        Rectangle((0, 0), 1, 1, facecolor="tab:orange", alpha=0.8,
                  label="Reload (copy stream)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "timeline_overlap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_gpu_util_timeline():
    """Figure 2: GPU utilization over time for a representative cell,
    with reload events marked as vertical lines."""
    cell = "W_Ruler32K_CkptReload_seed42"
    gpu_path = RESULTS_DIR / cell / "gpu_util.csv"
    reload_csv = find_cell_csv(cell, "reload_times")
    if not gpu_path.exists() or reload_csv is None:
        print(f"Missing data for {cell} GPU util")
        return

    # gpu_util.csv format: "timestamp, index, utilization.gpu, ..."
    times = []
    util_pct = []
    mem_used_mb = []
    with open(gpu_path) as f:
        # Skip header (nvidia-smi csv has header line)
        reader = csv.reader(f)
        next(reader, None)
        from datetime import datetime
        for row in reader:
            if len(row) < 5:
                continue
            try:
                ts_str = row[0].strip()
                idx = int(row[1].strip())
                if idx != 0:
                    continue
                util = int(row[2].strip())
                mem = int(row[4].strip())
                # nvidia-smi timestamp format: 2026/05/04 02:18:09.123
                dt = datetime.strptime(
                    ts_str, "%Y/%m/%d %H:%M:%S.%f"
                )
                times.append(dt.timestamp())
                util_pct.append(util)
                mem_used_mb.append(mem)
            except (ValueError, IndexError):
                continue

    if not times:
        print(f"No GPU samples parsed for {cell}")
        return

    t0 = times[0]
    times_rel = [t - t0 for t in times]

    reload_evt = load_csv(reload_csv)
    reload_x = [float(r["timestamp"]) - t0 for r in reload_evt]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(times_rel, util_pct, color="tab:blue", linewidth=1)
    ax1.set_ylabel("GPU compute %")
    ax1.set_title(
        f"V3 GPU activity over time ({cell})\n"
        f"orange ticks = reload events ({len(reload_x)} total)"
    )
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    for x in reload_x:
        ax1.axvline(x, color="tab:orange", alpha=0.4, linewidth=0.6)

    ax2.plot(
        times_rel, [m / 1024.0 for m in mem_used_mb],
        color="tab:green", linewidth=1,
    )
    ax2.set_ylabel("GPU memory used (GB)")
    ax2.set_xlabel("Time (s) from cell start")
    ax2.grid(True, alpha=0.3)
    for x in reload_x:
        ax2.axvline(x, color="tab:orange", alpha=0.4, linewidth=0.6)

    plt.tight_layout()
    out = FIG_DIR / "gpu_util_timeline.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_throughput_comparison():
    """Figure 3: paired throughput bar chart, NoFT vs CkptReload, per
    context. Uses forward step counts as a proxy for throughput."""
    contexts = [
        ("W_Ruler1K", 1024),
        ("W_Ruler4K", 4096),
        ("W_Ruler8K", 8192),
        ("W_Ruler16K", 16384),
        ("W_Ruler32K", 32768),
    ]
    seeds = [42, 123, 456]
    noft_means = []
    noft_stds = []
    ckpt_means = []
    ckpt_stds = []
    labels = []
    for ctx_name, ctx_tokens in contexts:
        labels.append(f"{ctx_tokens // 1024}K")
        for baseline, mean_list, std_list in [
            ("NoFT-Reprefill", noft_means, noft_stds),
            ("CkptReload", ckpt_means, ckpt_stds),
        ]:
            counts = []
            for seed in seeds:
                files = list(
                    (RESULTS_DIR / f"{ctx_name}_{baseline}_seed{seed}")
                    .glob("forward_times_pid*.csv")
                )
                if not files:
                    continue
                with open(files[0]) as f:
                    counts.append(sum(1 for _ in f) - 1)
            if counts:
                mean_list.append(statistics.mean(counts))
                std_list.append(
                    statistics.stdev(counts) if len(counts) > 1 else 0.0
                )
            else:
                mean_list.append(0)
                std_list.append(0)

    import numpy as np
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2, noft_means, width,
        yerr=noft_stds, label="vanilla vLLM (NoFT-Reprefill)",
        color="tab:gray", capsize=4,
    )
    ax.bar(
        x + width / 2, ckpt_means, width,
        yerr=ckpt_stds, label="V3 CkptReload (release + true overlap)",
        color="tab:blue", capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Forward step count (cell-level throughput proxy)")
    ax.set_title(
        "V3 vs vanilla vLLM throughput (paired by seed, "
        "mean ± std across 3 seeds)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate percent change
    for i, (n, c) in enumerate(zip(noft_means, ckpt_means)):
        if n > 0:
            pct = (c - n) / n * 100
            ax.annotate(
                f"{pct:+.0f}%",
                xy=(x[i] + width / 2, c),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", fontsize=9,
                color="tab:red" if pct < -10 else "tab:green",
            )

    plt.tight_layout()
    out = FIG_DIR / "throughput_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    fig_timeline_overlap()
    fig_gpu_util_timeline()
    fig_throughput_comparison()


if __name__ == "__main__":
    main()
