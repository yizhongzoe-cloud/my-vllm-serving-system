"""Main paper figure: ours vs vllm_fcfs across 3 settings.

Settings:
  1. Single GPU, arxivsumm (long-only), QPS=0.25
  2. Dual GPU, arxivsumm (long-only), QPS=0.5 (canonical picker config)
  3. Dual GPU, mixed_short_long, QPS=1.5 (measured at realistic SLO 3000/300)

Numbers are seed=0 only — multi-seed in progress.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ────────── data ──────────
settings = [
    ("Single GPU\narxivsumm\nQPS=0.25",          0.065, 0.177, 28.3, 76.7),
    ("Dual GPU\narxivsumm\nQPS=0.5",             0.262, 0.369, 56.7, 80.0),
    ("Dual GPU\nmixed_short_long\nQPS=1.5",      0.897, 0.958, 73.3, 78.3),
]
labels = [s[0] for s in settings]
fcfs_gp = [s[1] for s in settings]
ours_gp = [s[2] for s in settings]
fcfs_slo = [s[3] for s in settings]
ours_slo = [s[4] for s in settings]

# improvement factor for annotation
gp_x = [o / f for f, o in zip(fcfs_gp, ours_gp)]

# ────────── figure ──────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(labels))
w = 0.36

# Panel 1: goodput
ax = axes[0]
b1 = ax.bar(x - w/2, fcfs_gp, w, label="vllm_fcfs", color="#888888")
b2 = ax.bar(x + w/2, ours_gp, w, label="ours",      color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Goodput (SLO-meeting req/s)")
ax.set_title("Goodput")
ax.legend(loc="upper left")
# annotate improvement factors
for i, (f, o, mul) in enumerate(zip(fcfs_gp, ours_gp, gp_x)):
    ax.annotate(f"{mul:.2f}×",
                xy=(i + w/2, o), xytext=(0, 4),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold", color="#1f77b4")

# Panel 2: SLO_met%
ax = axes[1]
b1 = ax.bar(x - w/2, fcfs_slo, w, label="vllm_fcfs", color="#888888")
b2 = ax.bar(x + w/2, ours_slo, w, label="ours",      color="#1f77b4")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("SLO-meeting requests (%)")
ax.set_ylim(0, 100)
ax.set_title("SLO satisfaction rate")
ax.legend(loc="upper left")
for i, (f, o) in enumerate(zip(fcfs_slo, ours_slo)):
    delta = o - f
    ax.annotate(f"+{delta:.1f}pp",
                xy=(i + w/2, o), xytext=(0, 4),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold", color="#1f77b4")

fig.suptitle("Host-RAM KV checkpoint: goodput + SLO across 3 settings (14B Qwen2.5, A6000)",
             fontsize=11, y=1.00)
plt.tight_layout()

out_path = Path(__file__).parent / "fig_main_three_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved: {out_path}")

# also PDF for paper
plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
print(f"saved: {out_path.with_suffix('.pdf')}")
