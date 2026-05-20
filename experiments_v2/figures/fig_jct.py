"""JCT figure — FastServe-style framing.

Average and p95 Job Completion Time across QPS, single GPU + dual GPU.
JCT is unbounded so speedup ratios show full magnitude (vs SLO_met%
which is capped at 100%).
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval/results/a6000"
OUT = Path(__file__).resolve().parent


def load_dual(b, qps, seed, tag="arxivsumm_v2_canonical"):
    f = R / (f"dual_{b}_arxivsumm_qps{qps}_n60_seed{seed}_{tag}_metrics.json")
    return json.load(open(f)) if f.exists() else None


def load_single(b, qps, seed):
    f = R / f"single_{b}_arxivsumm_qps{qps}_n60_seed{seed}_v1_metrics.json"
    return json.load(open(f)) if f.exists() else None


def jct_stats(d):
    e2es = [r["e2e_ms"] for r in d["per_request"]]
    return np.mean(e2es) / 1000.0, np.percentile(e2es, 95) / 1000.0


def stats_mean(loader, baseline, qps, seeds):
    """Return (avg_jct_mean_s, p95_jct_mean_s) over seeds."""
    avgs, p95s = [], []
    for s in seeds:
        d = loader(baseline, qps, s)
        if d:
            a, p = jct_stats(d)
            avgs.append(a)
            p95s.append(p)
    return (np.mean(avgs) if avgs else None,
            np.mean(p95s) if p95s else None)


SEEDS = [0, 1]
SINGLE_QPS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
DUAL_QPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SINGLE_BASELINES = [
    ("vllm_fcfs", "vllm_fcfs",     "o-", "#888888"),
    ("ours",      "ours",          "s-", "#1f77b4"),
]
DUAL_BASELINES = [
    ("vllm_fcfs",       "vllm_fcfs",       "o-", "#888888"),
    ("reroute_no_ckpt", "reroute_no_ckpt", "^-", "#cc8800"),
    ("ours_no_picker",  "ours_no_picker",  "D-", "#7fb04f"),
    ("ours",            "ours",            "s-", "#1f77b4"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ─── Single GPU avg JCT ───
ax = axes[0, 0]
for b, label, style, color in SINGLE_BASELINES:
    avgs = []
    for q in SINGLE_QPS:
        a, _ = stats_mean(load_single, b, str(q), SEEDS)
        avgs.append(a)
    ax.plot(SINGLE_QPS, avgs, style, label=label, color=color,
            linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("avg JCT (seconds, lower = better)")
ax.set_title("Single GPU — avg JCT (mean of seed 0,1)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.25, color="red", linestyle="--", alpha=0.4)
ax.text(0.255, ax.get_ylim()[1] * 0.85,
        "saturation edge", color="red", alpha=0.7, fontsize=9)

# Annotate speedup at QPS=0.25
fcfs_avg25, _ = stats_mean(load_single, "vllm_fcfs", "0.25", SEEDS)
ours_avg25, _ = stats_mean(load_single, "ours",      "0.25", SEEDS)
if fcfs_avg25 and ours_avg25:
    ax.annotate(f"{fcfs_avg25/ours_avg25:.2f}× avg JCT\n"
                f"({fcfs_avg25:.0f}s → {ours_avg25:.0f}s)",
                xy=(0.25, ours_avg25), xytext=(0.27, ours_avg25 + 30),
                fontsize=9, color="#1f77b4", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))

# ─── Single GPU p95 JCT ───
ax = axes[0, 1]
for b, label, style, color in SINGLE_BASELINES:
    p95s = []
    for q in SINGLE_QPS:
        _, p = stats_mean(load_single, b, str(q), SEEDS)
        p95s.append(p)
    ax.plot(SINGLE_QPS, p95s, style, label=label, color=color,
            linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("p95 JCT (seconds, lower = better)")
ax.set_title("Single GPU — p95 JCT (mean of seed 0,1)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.25, color="red", linestyle="--", alpha=0.4)

_, fcfs_p95_25 = stats_mean(load_single, "vllm_fcfs", "0.25", SEEDS)
_, ours_p95_25 = stats_mean(load_single, "ours",      "0.25", SEEDS)
if fcfs_p95_25 and ours_p95_25:
    ax.annotate(f"{fcfs_p95_25/ours_p95_25:.2f}× p95 JCT\n"
                f"({fcfs_p95_25:.0f}s → {ours_p95_25:.0f}s)",
                xy=(0.25, ours_p95_25), xytext=(0.27, ours_p95_25 + 50),
                fontsize=9, color="#1f77b4", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))

# ─── Dual GPU avg JCT ───
ax = axes[1, 0]
for b, label, style, color in DUAL_BASELINES:
    avgs = []
    for q in DUAL_QPS:
        a, _ = stats_mean(load_dual, b, str(q), SEEDS)
        avgs.append(a)
    ax.plot(DUAL_QPS, avgs, style, label=label, color=color,
            linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("avg JCT (seconds, lower = better)")
ax.set_title("Dual GPU — avg JCT (mean of seed 0,1)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)
ax.text(0.51, ax.get_ylim()[1] * 0.85,
        "saturation edge", color="red", alpha=0.7, fontsize=9)

fcfs_avg7, _ = stats_mean(load_dual, "vllm_fcfs", "0.7", SEEDS)
ours_avg7, _ = stats_mean(load_dual, "ours",      "0.7", SEEDS)
if fcfs_avg7 and ours_avg7:
    ax.annotate(f"{fcfs_avg7/ours_avg7:.2f}× at QPS=0.7\n"
                f"({fcfs_avg7:.0f}s → {ours_avg7:.0f}s)",
                xy=(0.7, ours_avg7), xytext=(0.55, ours_avg7 + 25),
                fontsize=9, color="#1f77b4", weight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))

# ─── Dual GPU p95 JCT ───
ax = axes[1, 1]
for b, label, style, color in DUAL_BASELINES:
    p95s = []
    for q in DUAL_QPS:
        _, p = stats_mean(load_dual, b, str(q), SEEDS)
        p95s.append(p)
    ax.plot(DUAL_QPS, p95s, style, label=label, color=color,
            linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("p95 JCT (seconds, lower = better)")
ax.set_title("Dual GPU — p95 JCT (mean of seed 0,1)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)

fig.suptitle("Job Completion Time — comparable framing to FastServe (NSDI'24)",
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig_jct.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "fig_jct.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_jct.png")
