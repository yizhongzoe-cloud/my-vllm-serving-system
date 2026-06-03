#!/usr/bin/env python3
"""Disruption recovery figure: failover gap after one of two engines is
killed mid-decode, split into detection latency (common to both systems)
and resume cost (the differentiator). Ferry reloads the host checkpoint;
Reroute-reprefill re-prefills the long prompt on the surviving engine.
Clean low-contention setting (concurrency 2, long prompts), mean of
seeds 0 and 1. Renders fig_failover_bar.{pdf,png}.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(os.environ.get(
    "EVAL_RESULTS_DIR",
    Path(__file__).resolve().parents[1] / "eval" / "results" / "a6000"))
SEEDS = [0, 1, 2]
DET_COLOR = "#d9d9d9"
CONFIGS = [
    ("reroute_no_ckpt", "Reroute + re-prefill", "Reroute + re-prefill", "#e07a5f"),
    ("ours", "Ferry", "Ferry", "#2e86ab"),
]


def gap_detection(baseline):
    """Return (mean_detection_s, mean_resume_s, totals) over seeds; resume =
    raw failover gap minus detection latency, totals = per-seed raw gaps
    (for the error bar)."""
    dets, resumes, totals = [], [], []
    for s in SEEDS:
        f = R / f"e_d1cl_{baseline}_n2_long_seed{s}_metrics.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        det = (d.get("detection_latency_s") or 0.0)
        raw = d["failover_gap_stats_ms"]["mean"] / 1000.0
        dets.append(det)
        resumes.append(max(0.0, raw - det))
        totals.append(raw)
    n = len(dets) or 1
    return sum(dets) / n, sum(resumes) / n, totals


from matplotlib.patches import Patch  # noqa: E402

fig, ax = plt.subplots(figsize=(4.8, 3.3))
labels, shorts, det_vals, res_vals, colors, totals_all = [], [], [], [], [], []
for b, lab, short, c in CONFIGS:
    det, res, totals = gap_detection(b)
    labels.append(lab); shorts.append(short)
    det_vals.append(det); res_vals.append(res)
    colors.append(c); totals_all.append(totals)
    print(f"{b}: detection={det:.2f}s resume={res:.2f}s "
          f"total={det+res:.2f}s n={len(totals)} range={min(totals):.1f}-{max(totals):.1f}")

x = range(len(labels))
ax.bar(x, det_vals, color=DET_COLOR, edgecolor="black", linewidth=0.5)
# error bar on the total bar = min/max range of per-seed total failover gap
yerr_lo = [(d + r) - min(t) for d, r, t in zip(det_vals, res_vals, totals_all)]
yerr_hi = [max(t) - (d + r) for d, r, t in zip(det_vals, res_vals, totals_all)]
ax.bar(x, res_vals, bottom=det_vals, color=colors, edgecolor="black",
       linewidth=0.5,
       yerr=[yerr_lo, yerr_hi], capsize=4, ecolor="black",
       error_kw={"linewidth": 1.0})
for i, (d, r) in enumerate(zip(det_vals, res_vals)):
    ax.text(i, max(totals_all[i]) + 0.35, f"{d+r:.1f}s", ha="center",
            fontsize=12, fontweight="bold")
    ax.text(i, d + r / 2, f"resume\n{r:.1f}s", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(shorts, fontsize=11)
ax.set_ylabel("Failover gap (s)", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylim(0, max(max(t) for t in totals_all) * 1.32)
handles = [Patch(facecolor=DET_COLOR, edgecolor="black",
                 label="Failure detection (common)")]
ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
for out in (Path(__file__).resolve().parent / "fig_failover_bar.pdf",
            Path(__file__).resolve().parent / "fig_failover_bar.png"):
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print("saved:", out)
