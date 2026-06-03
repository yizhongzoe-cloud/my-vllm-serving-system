#!/usr/bin/env python3
"""vLLM request pipeline with Ferry's hook points highlighted — for the
CSE 232 implementation slide. Modified components (coral), new subsystem
(blue), untouched vLLM (gray). Renders fig_vllm_hooks.{pdf,png}."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BLUE = "#2e86ab"; CORAL = "#e07a5f"; GRAY = "#9aa0a6"; INK = "#1a1a1a"
CORAL_F = "#fbeae4"; BLUE_F = "#e3eff5"; GRAY_F = "#f0f1f2"

fig, ax = plt.subplots(figsize=(11.2, 5.0))
ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.0); ax.axis("off")

# stage boxes: (x, label, kind)  kind: 'mod' / 'new-host' / 'plain'
stages = [
    ("API server", "plain"),
    ("EngineCore", "mod"),
    ("Scheduler", "mod"),
    ("KV block pool", "plain"),
    ("ModelRunner", "mod"),
    ("Model + kernels", "plain"),
]
n = len(stages); bw, bh, gap = 1.55, 0.78, 0.18
x0 = 0.25; y = 3.55
centers = []
for i, (lab, kind) in enumerate(stages):
    x = x0 + i * (bw + gap)
    cx = x + bw / 2; centers.append(cx)
    if kind == "mod":
        ec, fc, lw = CORAL, CORAL_F, 2.2
    else:
        ec, fc, lw = GRAY, GRAY_F, 1.4
    box = FancyBboxPatch((x, y), bw, bh,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(cx, y + bh / 2, lab, ha="center", va="center", fontsize=11.5,
            color=INK, fontweight="bold")
    if i < n - 1:
        ax.annotate("", xy=(x + bw + gap, y + bh / 2),
                    xytext=(x + bw, y + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))

# annotations under modified boxes
ann = {1: "status + preempt-\nqueue bus  (~190)",
       2: "capacity-preempt\n→ reload  (~170)",
       4: "per-step publish\n+ reload  (~830)"}
for i, txt in ann.items():
    ax.annotate("", xy=(centers[i], y - 0.05), xytext=(centers[i], y - 0.55),
                arrowprops=dict(arrowstyle="-", color=CORAL, lw=1.4))
    ax.text(centers[i], y - 0.95, txt, ha="center", va="center",
            fontsize=9.2, color=CORAL, fontweight="bold")

# NEW host store box
hw, hh = 4.4, 0.95; hx = centers[3] - hw / 2; hy = 0.55
hb = FancyBboxPatch((hx, hy), hw, hh,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=2.4, edgecolor=BLUE, facecolor=BLUE_F)
ax.add_patch(hb)
ax.text(hx + hw / 2, hy + hh / 2 + 0.16, "Host Checkpoint Store   (NEW)",
        ha="center", va="center", fontsize=12, color=BLUE, fontweight="bold")
ax.text(hx + hw / 2, hy + hh / 2 - 0.22,
        "kv_checkpoint_pool.py  ~760 LOC", ha="center", va="center",
        fontsize=9.5, color=INK)
# arrow ModelRunner <-> store (publish down, reload up)
ax.add_patch(FancyArrowPatch((centers[4], y - 1.35), (hx + hw, hy + hh * 0.6),
             arrowstyle="-|>", color=BLUE, lw=1.8,
             connectionstyle="arc3,rad=-0.2", mutation_scale=14))
ax.text(centers[4] + 0.15, y - 1.65, "publish / reload", ha="center",
        fontsize=9, color=BLUE, fontweight="bold")

# legend
lx = 8.6; ly = 0.95
for j, (c, cf, t) in enumerate([(CORAL, CORAL_F, "modified hook"),
                                (BLUE, BLUE_F, "new subsystem"),
                                (GRAY, GRAY_F, "untouched vLLM")]):
    yy = ly - j * 0.42
    ax.add_patch(FancyBboxPatch((lx, yy), 0.34, 0.24,
                 boxstyle="round,pad=0.01,rounding_size=0.04",
                 linewidth=1.8, edgecolor=c, facecolor=cf))
    ax.text(lx + 0.46, yy + 0.12, t, ha="left", va="center", fontsize=9.5,
            color=INK)

ax.text(0.25, 4.78, "vLLM request pipeline — where Ferry plugs in",
        ha="left", va="center", fontsize=13.5, color=INK, fontweight="bold")

fig.tight_layout()
for ext in ("png", "pdf"):
    out = Path(__file__).resolve().parent / f"fig_vllm_hooks.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    print("saved:", out)
