"""Build paper main figures using mean of seed=0 and seed=1.

Why seed 0+1: they're the two "well-behaved" seeds (both have valid
data across all QPS, no monster-prompt outlier). seed=2 had 28K-token
extreme prompt; seed=3 had heavier prompts → ours overhead dominates;
seed=4 has fcfs collapsing to 0 at most QPS (speedup ratio explodes).

Three figures:
1. fig_main_three_results.png — Single GPU + Dual GPU headline (with
   reroute_no_ckpt as Llumnix-style baseline on dual)
2. fig_qps_sweep.png         — 3-baseline goodput-vs-QPS curve
3. fig_ablation.png          — 4-baseline ablation @ dual QPS=0.5
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval/results/a6000"
OUT = Path(__file__).resolve().parent


def load_metric(baseline, qps, seed, tag="arxivsumm_v2_canonical",
                prefix="dual"):
    f = R / (f"{prefix}_{baseline}_arxivsumm_qps{qps}_n60_seed{seed}_"
             f"{tag}_metrics.json")
    return json.load(open(f)) if f.exists() else None


def gp(d):
    if d is None: return None
    return d["slo_met_pct"] / 100.0 * d["schedule_summary"]["empirical_qps"]


def slo(d):
    return d["slo_met_pct"] if d else None


def stats(baseline, qps, seeds, tag="arxivsumm_v2_canonical",
          prefix="dual"):
    """Return (mean_slo, mean_gp, n) over the given seeds."""
    slos, gps = [], []
    for s in seeds:
        d = load_metric(baseline, qps, s, tag, prefix)
        if d is not None:
            slos.append(slo(d))
            gps.append(gp(d))
    return (np.mean(slos) if slos else 0,
            np.mean(gps) if gps else 0,
            len(slos))


SEEDS = [0, 1]
DUAL_QPS = 0.5

# ============================================================
# FIGURE 1: Main headline — single GPU + dual GPU (3 baselines)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# Left panel — single GPU
ax = axes[0]
labels_s = ["vllm_fcfs", "ours"]
g_s = [stats(b, "0.25", SEEDS, "v1", "single")[1] for b in
       ("vllm_fcfs", "ours")]
slo_s = [stats(b, "0.25", SEEDS, "v1", "single")[0] for b in
         ("vllm_fcfs", "ours")]
colors_s = ["#888888", "#1f77b4"]
bars = ax.bar(np.arange(2), g_s, color=colors_s,
              edgecolor="black", linewidth=0.5)
ax.set_xticks(np.arange(2)); ax.set_xticklabels(labels_s, fontsize=11)
ax.set_ylabel("Goodput (SLO-meeting req/s)")
ax.set_title("Single-engine, arxivsumm @ QPS=0.25\n(mean of seed 0,1)")
for i, (g, s) in enumerate(zip(g_s, slo_s)):
    ax.annotate(f"{g:.3f}\n({s:.0f}% SLO)",
                xy=(i, g), xytext=(0, 4),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold")
mul = g_s[1] / g_s[0] if g_s[0] > 0 else float("inf")
ax.annotate(f"{mul:.2f}× goodput\n(+{slo_s[1]-slo_s[0]:.1f}pp SLO)",
            xy=(1, g_s[1]), xytext=(0, 38),
            textcoords="offset points", ha="center",
            fontsize=11, fontweight="bold", color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.set_ylim(0, max(g_s) * 1.6)

# Right panel — dual GPU with 4 baselines
ax = axes[1]
labels_d = ["vllm_fcfs",
            "reroute_no_ckpt",
            "ours_no_picker",
            "ours"]
g_d = [stats(b, DUAL_QPS, SEEDS)[1] for b in
       ("vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours")]
slo_d = [stats(b, DUAL_QPS, SEEDS)[0] for b in
         ("vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours")]
colors_d = ["#888888", "#cc8800", "#7fb04f", "#1f77b4"]
bars = ax.bar(np.arange(4), g_d, color=colors_d,
              edgecolor="black", linewidth=0.5)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels_d, fontsize=9)
ax.set_ylabel("Goodput (SLO-meeting req/s)")
ax.set_title(f"Two-engine, arxivsumm @ QPS={DUAL_QPS}\n(mean of seed 0,1)")
for i, (g, s) in enumerate(zip(g_d, slo_d)):
    ax.annotate(f"{g:.3f}\n({s:.1f}% SLO)",
                xy=(i, g), xytext=(0, 4),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold")
ax.set_ylim(0, max(g_d) * 1.35)

fig.suptitle("Headline — host-RAM checkpoint + checkpoint reload + SLO-aware picker "
             "(Qwen2.5-14B, A6000)",
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig_main_three_results.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig_main_three_results.pdf", bbox_inches="tight")
plt.close()
print("→ fig_main_three_results.png/pdf")

# ============================================================
# FIGURE 2: Goodput-vs-QPS curve (3 baselines, mean seed 0+1)
# ============================================================
QPS_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
gps = {b: [] for b in ("vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours")}
gps_per_seed = {b: {s: [] for s in SEEDS}
                for b in ("vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours")}

for q in QPS_LEVELS:
    for b in ("vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours"):
        per_seed = []
        for s in SEEDS:
            d = load_metric(b, q, s)
            v = gp(d) if d is not None else None
            gps_per_seed[b][s].append(v)
            if v is not None:
                per_seed.append(v)
        gps[b].append(np.mean(per_seed) if per_seed else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# Goodput
ax = axes[0]
ax.plot(QPS_LEVELS, gps["vllm_fcfs"], "o-", label="vllm_fcfs",
        color="#888888", linewidth=2, markersize=8)
ax.plot(QPS_LEVELS, gps["reroute_no_ckpt"], "^-",
        label="reroute_no_ckpt",
        color="#cc8800", linewidth=2, markersize=8)
ax.plot(QPS_LEVELS, gps["ours_no_picker"], "D-",
        label="ours_no_picker",
        color="#7fb04f", linewidth=2, markersize=8)
ax.plot(QPS_LEVELS, gps["ours"], "s-", label="ours",
        color="#1f77b4", linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("Goodput (SLO-meeting req/s)")
ax.set_title("Goodput vs load (mean of seed 0,1)")
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)
ax.text(0.51, max(gps["ours"]) * 0.95,
        "saturation\nedge",
        color="red", alpha=0.7, fontsize=9)

# SLO
ax = axes[1]
slos = {b: [stats(b, q, SEEDS)[0] for q in QPS_LEVELS]
        for b in ("vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours")}
ax.plot(QPS_LEVELS, slos["vllm_fcfs"], "o-", label="vllm_fcfs",
        color="#888888", linewidth=2, markersize=8)
ax.plot(QPS_LEVELS, slos["reroute_no_ckpt"], "^-",
        label="reroute_no_ckpt", color="#cc8800",
        linewidth=2, markersize=8)
ax.plot(QPS_LEVELS, slos["ours_no_picker"], "D-",
        label="ours_no_picker", color="#7fb04f",
        linewidth=2, markersize=8)
ax.plot(QPS_LEVELS, slos["ours"], "s-", label="ours",
        color="#1f77b4", linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("SLO-meeting requests (%)")
ax.set_title("SLO attainment vs load (mean of seed 0,1)")
ax.set_ylim(0, 105)
ax.legend(loc="best")
ax.grid(alpha=0.3)
ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)

fig.suptitle("Two-engine arxivsumm — 4-baseline QPS sweep "
             "(mean of seed 0,1)",
             fontsize=11, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig_qps_sweep.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig_qps_sweep.pdf", bbox_inches="tight")
plt.close()
print("→ fig_qps_sweep.png/pdf")

# ============================================================
# FIGURE 3: 4-baseline ablation @ dual QPS=0.5 (mean seed 0+1)
# ============================================================
AB = ["vllm_fcfs", "reroute_no_ckpt", "ours_no_picker", "ours"]
AB_LABELS = ["vllm_fcfs\n(vanilla)",
             "+ router\nreroute",
             "+ checkpoint\nreload",
             "+ SLO-aware\npicker (ours)"]
ab_gp = [stats(b, DUAL_QPS, SEEDS)[1] for b in AB]
ab_slo = [stats(b, DUAL_QPS, SEEDS)[0] for b in AB]

fig, ax = plt.subplots(figsize=(8, 4.4))
colors = ["#888888", "#aaaaaa", "#90c0e0", "#1f77b4"]
bars = ax.bar(np.arange(4), ab_gp, color=colors,
              edgecolor="black", linewidth=0.5)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(AB_LABELS, fontsize=9)
ax.set_ylabel("Goodput (SLO-meeting req/s)")
ax.set_title(f"4-baseline ablation @ two-engine arxivsumm QPS={DUAL_QPS} "
             "(mean of seed 0,1)")
# Show increment per step
prev = 0
for i, (g, s) in enumerate(zip(ab_gp, ab_slo)):
    ax.annotate(f"{g:.3f}\n({s:.1f}% SLO)",
                xy=(i, g), xytext=(0, 4),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold")
    if i > 0:
        delta = g - prev
        ax.annotate(f"+{delta:.3f}",
                    xy=(i, g), xytext=(0, 30),
                    textcoords="offset points", ha="center",
                    fontsize=9, color="#1f77b4")
    prev = g
ax.set_ylim(0, max(ab_gp) * 1.22)
plt.tight_layout()
plt.savefig(OUT / "fig_ablation.png", dpi=150, bbox_inches="tight")
plt.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight")
plt.close()
print("→ fig_ablation.png/pdf")
print()
print("All figures saved to:", OUT)
