"""Paper system design figure.

Three boxes + dataflow arrows, no implementation detail (pinned memory
vs /dev/shm collapsed into 'Host RAM KV pool')."""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ────────── canvas ──────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.set_aspect("equal"); ax.axis("off")

# colors
C_BG = "#fafafa"
C_CLIENT = "#dddddd"
C_ROUTER = "#ffd580"
C_ENGINE = "#a8d5ff"
C_PICKER = "#7eb8ff"
C_RAM    = "#c8e6c9"
C_ARROW  = "#444444"
C_PREEMPT = "#c0392b"
C_RELOAD  = "#1e8449"

def box(ax, x, y, w, h, label, color, fontsize=10, fontweight="normal",
        edgecolor="black", linewidth=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=linewidth)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)

def arrow(ax, x0, y0, x1, y1, label=None, color="#444", style="-",
          rad=0, labeloffset=(0, 0), fontsize=8, linewidth=1.3,
          mutation=18):
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          arrowstyle="-|>",
                          mutation_scale=mutation, linewidth=linewidth,
                          color=color,
                          connectionstyle=f"arc3,rad={rad}",
                          linestyle=style)
    ax.add_patch(arr)
    if label is not None:
        mx = (x0 + x1) / 2 + labeloffset[0]
        my = (y0 + y1) / 2 + labeloffset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=fontsize, color=color,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.85, pad=1.5))

# ────────── boxes ──────────
# client
box(ax, 0.3, 5.0, 1.4, 0.7, "Client(s)", C_CLIENT, fontsize=11)

# router
box(ax, 4.0, 5.0, 2.0, 0.7,
    "Router\n(KV-aware least-load)", C_ROUTER, fontsize=10)

# engines
box(ax, 0.6, 2.7, 2.3, 1.4,
    "Engine 0 (GPU 0)\nvLLM scheduler\n+ SLO-aware picker",
    C_ENGINE, fontsize=10)
box(ax, 7.1, 2.7, 2.3, 1.4,
    "Engine 1 (GPU 1)\nvLLM scheduler\n+ SLO-aware picker",
    C_ENGINE, fontsize=10)

# host ram
box(ax, 2.4, 0.4, 5.2, 1.0,
    "Host RAM KV pool\n(continuous delta checkpoint, accessible by both engines)",
    C_RAM, fontsize=10, fontweight="bold")

# ────────── arrows ──────────
# client → router (request)
arrow(ax, 1.7, 5.35, 4.0, 5.35,
      label="HTTP /v1/completions", color=C_ARROW, fontsize=9,
      labeloffset=(0, 0.18))

# router → engine 0 (initial dispatch)
arrow(ax, 4.7, 4.95, 1.8, 4.15,
      label="dispatch", color=C_ARROW, rad=-0.15,
      labeloffset=(-0.35, 0.18))
# router → engine 1
arrow(ax, 5.3, 4.95, 8.2, 4.15,
      label="dispatch", color=C_ARROW, rad=0.15,
      labeloffset=(0.35, 0.18))

# engine 0 → host RAM (checkpoint write)
arrow(ax, 1.8, 2.65, 3.5, 1.45,
      label="continuous\ncheckpoint", color=C_RELOAD, rad=-0.18,
      labeloffset=(-0.6, 0.2))
# engine 1 → host RAM
arrow(ax, 8.2, 2.65, 6.5, 1.45,
      label="continuous\ncheckpoint", color=C_RELOAD, rad=0.18,
      labeloffset=(0.6, 0.2))

# host RAM → engine 0 (V3 reload on resume)
arrow(ax, 3.3, 1.45, 1.8, 2.65,
      label="V3 reload\n(cheap resume)", color=C_RELOAD,
      rad=0.18, style="--", labeloffset=(0.85, -0.05))
# host RAM → engine 1
arrow(ax, 6.7, 1.45, 8.2, 2.65,
      label="V3 reload\n(cheap resume)", color=C_RELOAD,
      rad=-0.18, style="--", labeloffset=(-0.85, -0.05))

# picker preempt (engine A → router → engine B)
# engine 0 → router (picker fired victim)
arrow(ax, 2.5, 4.05, 4.7, 5.05,
      label="picker preempts\nvictim → preempt queue",
      color=C_PREEMPT, rad=0.15, style=":", fontsize=8,
      labeloffset=(-0.2, 0.45))
# router → engine 1 (reroute)
arrow(ax, 5.5, 5.0, 7.5, 4.1,
      label="reroute victim\nto peer engine",
      color=C_PREEMPT, rad=-0.15, style=":", fontsize=8,
      labeloffset=(0.4, 0.4))

# ────────── legend ──────────
legend_y = 0.06
ax.plot([0.3, 0.6], [legend_y, legend_y], color=C_ARROW, linewidth=1.3)
ax.text(0.7, legend_y, "request dataflow", fontsize=8, va="center")
ax.plot([2.8, 3.1], [legend_y, legend_y], color=C_RELOAD, linewidth=1.3)
ax.text(3.2, legend_y, "checkpoint write / reload", fontsize=8, va="center")
ax.plot([5.7, 6.0], [legend_y, legend_y], color=C_PREEMPT,
        linewidth=1.3, linestyle=":")
ax.text(6.1, legend_y, "picker preempt + reroute (cross-engine)",
        fontsize=8, va="center")

# ────────── title ──────────
ax.text(5, 5.85,
        "System architecture — host-RAM continuous checkpoint enables "
        "cheap cross-engine preempt-resume",
        ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "fig_system_design.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "fig_system_design.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_system_design.png")
