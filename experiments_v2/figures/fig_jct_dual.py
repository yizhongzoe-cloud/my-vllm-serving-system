"""JCT figure for the paper (§5.3 sibling): two-engine avg + p95 JCT vs QPS.

Where SLO attainment saturates to near-zero under overload, Job Completion
Time still separates the baselines and reveals the full magnitude of the
mechanism's benefit: cheap reload-backed preemption keeps the engines
flowing, while reroute-no-ckpt cannot afford preemption, so head-of-line
blocking inflates JCT.

Source: experiments_v2/eval/results/a6000/
        dual_{baseline}_arxivsumm_qps{q}_n60_seed{0,1}_arxivsumm_v2_canonical_metrics.json
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval/results/a6000"
OUT = Path(__file__).resolve().parent
SEEDS = [0, 1]
QPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BASELINES = [
    ("vllm_fcfs",       "vllm_fcfs",       "o-", "#888888"),
    ("reroute_no_ckpt", "reroute_no_ckpt", "^-", "#cc8800"),
    ("ours_no_picker",  "ours_no_picker",  "D-", "#7fb04f"),
    ("ours",            "ours",            "s-", "#1f77b4"),
]


def load(b, q, s):
    f = R / (f"dual_{b}_arxivsumm_qps{q}_n60_seed{s}_"
             f"arxivsumm_v2_canonical_metrics.json")
    return json.load(open(f)) if f.exists() else None


def jct_mean_p95(b, q):
    """Mean over seeds of (avg JCT, p95 JCT) in seconds."""
    avgs, p95s = [], []
    for s in SEEDS:
        d = load(b, q, s)
        if d is None:
            continue
        e2e = [r["e2e_ms"] / 1000.0 for r in d["per_request"]]
        avgs.append(np.mean(e2e))
        p95s.append(np.percentile(e2e, 95))
    return (np.mean(avgs) if avgs else None,
            np.mean(p95s) if p95s else None)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for panel, (idx, title) in enumerate([(0, "avg JCT"), (1, "p95 JCT")]):
    ax = axes[panel]
    for b, label, style, color in BASELINES:
        ys = []
        for q in QPS:
            a, p = jct_mean_p95(b, str(q))
            ys.append((a if idx == 0 else p))
        xs = [q for q, y in zip(QPS, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        ax.plot(xs, yy, style, label=label, color=color,
                linewidth=2, markersize=8)
    ax.set_xlabel("Arrival rate (QPS, Poisson)")
    ax.set_ylabel(f"{title} (seconds, lower = better)")
    ax.set_title(f"Two-engine — {title}")
    ax.grid(alpha=0.3)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)

# Annotate the reroute-vs-ours gap at QPS=0.8 on the avg panel.
a_ours, _ = jct_mean_p95("ours", "0.8")
a_rr, _ = jct_mean_p95("reroute_no_ckpt", "0.8")
if a_ours and a_rr:
    axes[0].annotate(f"{a_rr/a_ours:.1f}$\\times$ at QPS=0.8\n"
                     f"({a_rr:.0f}s vs {a_ours:.0f}s)",
                     xy=(0.8, a_ours), xytext=(0.57, 88),
                     fontsize=9, color="#1f77b4", weight="bold",
                     ha="center",
                     arrowprops=dict(arrowstyle="->", color="#1f77b4"))

fig.suptitle("Two-engine arxivsumm — Job Completion Time (mean of seed 0,1)",
             fontsize=11, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig_jct_dual.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "fig_jct_dual.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_jct_dual.pdf")
print(f"{'QPS':>5} {'vllm_fcfs':>12} {'reroute_nc':>12} "
      f"{'ours_no_pick':>13} {'ours':>8}  (avg JCT s)")
for q in QPS:
    row = [f"{q:>5}"]
    for b, *_ in BASELINES:
        a, _ = jct_mean_p95(b, str(q))
        row.append(f"{a:>12.1f}" if a else f"{'n/a':>12}")
    print("  ".join(row))
