"""Single GPU QPS sweep — fcfs vs ours, mean of seed 0,1.

Shows the saturation-edge sweet spot pattern: ours wins decisively at
QPS=0.25 (the saturation edge for single 14B engine), narrower wins or
ties at other operating points."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval/results/a6000"
OUT = Path(__file__).resolve().parent


def load(baseline, qps, seed):
    f = R / (f"single_{baseline}_arxivsumm_qps{qps}_n60_seed{seed}"
             f"_v1_metrics.json")
    return json.load(open(f)) if f.exists() else None


def gp(d):
    return d["slo_met_pct"]/100 * d["schedule_summary"]["empirical_qps"]


SEEDS = [0, 1]
QPS_LEVELS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
BASELINES = [
    ("vllm_fcfs", "vllm_fcfs", "o-", "#888888"),
    ("ours",      "ours",      "s-", "#1f77b4"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# Left: SLO_met
ax = axes[0]
for b, label, style, color in BASELINES:
    means, mins, maxs = [], [], []
    for q in QPS_LEVELS:
        vals = []
        for s in SEEDS:
            d = load(b, q, s)
            if d: vals.append(d["slo_met_pct"])
        if vals:
            means.append(np.mean(vals))
            mins.append(min(vals))
            maxs.append(max(vals))
        else:
            means.append(None); mins.append(None); maxs.append(None)
    xs = [q for q, m in zip(QPS_LEVELS, means) if m is not None]
    ys = [m for m in means if m is not None]
    los = [mn for mn in mins if mn is not None]
    his = [mx for mx in maxs if mx is not None]
    ax.plot(xs, ys, style, label=label, color=color, linewidth=2, markersize=8)
    ax.fill_between(xs, los, his, alpha=0.2, color=color)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("SLO-meeting requests (%)")
ax.set_title("SLO satisfaction rate (single 14B, mean ± seed range)")
ax.set_ylim(0, 105)
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.25, color="red", linestyle="--", alpha=0.4)
ax.text(0.255, 60, "saturation\nedge", color="red", alpha=0.7, fontsize=9)

# Right: goodput
ax = axes[1]
for b, label, style, color in BASELINES:
    means, mins, maxs = [], [], []
    for q in QPS_LEVELS:
        vals = []
        for s in SEEDS:
            d = load(b, q, s)
            if d: vals.append(gp(d))
        if vals:
            means.append(np.mean(vals))
            mins.append(min(vals))
            maxs.append(max(vals))
        else:
            means.append(None); mins.append(None); maxs.append(None)
    xs = [q for q, m in zip(QPS_LEVELS, means) if m is not None]
    ys = [m for m in means if m is not None]
    los = [mn for mn in mins if mn is not None]
    his = [mx for mx in maxs if mx is not None]
    ax.plot(xs, ys, style, label=label, color=color, linewidth=2, markersize=8)
    ax.fill_between(xs, los, his, alpha=0.2, color=color)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("Goodput (SLO-meeting req/s)")
ax.set_title("Goodput (single 14B, mean ± seed range)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.25, color="red", linestyle="--", alpha=0.4)

fig.suptitle("Single GPU arxivsumm QPS sweep — ours wins decisively at the "
             "saturation edge (QPS=0.25)",
             fontsize=11, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig_single_gpu_sweep.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "fig_single_gpu_sweep.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_single_gpu_sweep.png")
