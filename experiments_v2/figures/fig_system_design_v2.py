"""System architecture figure — top-conf paper style.

Layout:
  Top:    Client → Router (KV-aware least-load)
  Middle: Engine 1 (GPU 1) | Engine 2 (GPU 2), symmetric structure
            Scheduler (Waiting/Running queue + SLO-aware Picker)
              ↓
            Model Executor
              ↓
            GPU Memory (Model Weights + KV Manager)
  Bottom: Host RAM KV Pool (spans full width)

Two mechanisms shown as arrow flows:
  M1 (green): Engine.KV Manager → Host RAM (continuous delta checkpoint write)
  M2 (green dashed): Host RAM → Engine.KV Manager (V3 cheap reload read)
  Cross-engine reroute (orange): victim flows Engine1.Picker → Router → Engine2
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Color palette (conference-paper restrained)
C_BG       = "white"
C_OUTER    = "#cccccc"           # GPU outer frame
C_GPU_FILL = "#f6f6f6"           # GPU box background
C_SCHED    = "#cfe2f3"           # Scheduler (light blue)
C_EXEC     = "#d9d2e9"           # Model executor (light purple)
C_KVMGR    = "#cfe2f3"           # KV Manager (light blue)
C_WEIGHTS  = "#fff2cc"           # Model weights (light yellow)
C_HOST     = "#d9ead3"           # Host RAM pool (light green)
C_ROUTER   = "#fce5cd"           # Router (light orange)
C_PICKER   = "#f4cccc"           # Picker badge (light red)
C_VICTIM   = "#f9cb9c"           # Victim block (orange)

C_REQUEST  = "#333333"           # Request flow (dark gray)
C_CHECKPT  = "#38761d"           # Checkpoint write/read (green)
C_REROUTE  = "#cc4125"           # Reroute (red-orange)
C_EDGE     = "#222222"

# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.set_aspect("equal"); ax.axis("off")

def rounded(x, y, w, h, color, edge=C_EDGE, lw=1.0, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                       facecolor=color, edgecolor=edge, linewidth=lw, alpha=alpha)
    ax.add_patch(p)

def text(x, y, s, fontsize=10, weight="normal", color="black", ha="center",
         va="center", style="normal"):
    ax.text(x, y, s, fontsize=fontsize, fontweight=weight, color=color,
            ha=ha, va=va, fontstyle=style)

def arrow(x0, y0, x1, y1, color=C_REQUEST, ls="-", lw=1.6, rad=0,
          mut=18, label=None, lx=None, ly=None, lfs=8, lcolor=None, lbg=True):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=mut, linewidth=lw, color=color,
                        linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx = lx if lx is not None else (x0 + x1) / 2
        my = ly if ly is not None else (y0 + y1) / 2
        bbox = dict(facecolor="white", edgecolor="none", alpha=0.92,
                    pad=2.0) if lbg else None
        text(mx, my, label, fontsize=lfs, color=lcolor or color, weight="bold")
        if bbox:
            ax.text(mx, my, label, fontsize=lfs, color=lcolor or color,
                    ha="center", va="center", weight="bold", bbox=bbox)

# ────────── Client + Router (top row) ──────────
text(1.0, 7.55, "Client", fontsize=11, weight="bold")

# Router
rounded(5.0, 7.05, 3.0, 0.70, C_ROUTER, lw=1.2)
text(6.5, 7.40, "Router  (KV-aware least-load)", fontsize=11, weight="bold")

# Client → Router
arrow(1.6, 7.40, 5.0, 7.40, color=C_REQUEST,
      label="HTTP /v1/completions", lfs=9, ly=7.62)

def draw_engine(x0, y0, engine_id, show_picker_active=False):
    """Draw one engine box at (x0, y0). Width=5, Height=5."""
    W, H = 5.0, 5.4
    # outer GPU frame
    rounded(x0, y0, W, H, C_GPU_FILL, edge=C_OUTER, lw=1.8)
    text(x0 + W - 0.55, y0 + H - 0.22, f"GPU {engine_id}", fontsize=10,
         weight="bold", color="#666666")

    # Scheduler box (top of engine)
    sx, sy, sw, sh = x0 + 0.30, y0 + H - 1.65, W - 0.60, 1.30
    rounded(sx, sy, sw, sh, C_SCHED, lw=0.9)
    text(sx + sw/2, sy + sh - 0.18, "Scheduler", fontsize=10, weight="bold")
    # waiting queue (left)
    rounded(sx + 0.15, sy + 0.18, 1.50, 0.30, "white", lw=0.6)
    text(sx + 0.20, sy + 0.62, "Waiting Queue", fontsize=8.5, ha="left")
    # queue cells (waiting)
    for i in range(4):
        rounded(sx + 0.20 + i*0.30, sy + 0.22, 0.26, 0.22,
                "#9fc5e8", lw=0.5)
    # running queue (right)
    rounded(sx + sw - 1.65, sy + 0.18, 1.50, 0.30, "white", lw=0.6)
    text(sx + sw - 1.60, sy + 0.62, "Running Queue", fontsize=8.5, ha="left")
    for i in range(4):
        rounded(sx + sw - 1.60 + i*0.30, sy + 0.22, 0.26, 0.22,
                "#9fc5e8", lw=0.5)

    # Picker badge (on top of scheduler box)
    pxc, pyc = sx + sw/2, sy - 0.05
    rounded(pxc - 0.95, pyc - 0.18, 1.90, 0.30, C_PICKER, lw=0.8)
    text(pxc, pyc - 0.03, "SLO-aware Picker", fontsize=8.5, weight="bold")

    # Model Executor
    ex, ey, ew, eh = x0 + 0.55, y0 + 1.95, W - 1.10, 0.70
    rounded(ex, ey, ew, eh, C_EXEC, lw=0.9)
    text(ex + ew/2, ey + eh/2, "Model Executor", fontsize=10, weight="bold")

    # GPU Memory frame
    mx, my, mw, mh = x0 + 0.30, y0 + 0.30, W - 0.60, 1.50
    rounded(mx, my, mw, mh, "white", edge="#aaaaaa", lw=0.9)
    text(mx + mw/2, my + mh - 0.18, "GPU Memory", fontsize=10, weight="bold")
    # Model weights
    rounded(mx + 0.20, my + 0.18, 1.3, 0.50, C_WEIGHTS, lw=0.7)
    text(mx + 0.85, my + 0.43, "Model\nWeights", fontsize=8.5)
    # KV Manager
    kvx, kvy, kvw, kvh = mx + 1.70, my + 0.18, mw - 1.90, 0.85
    rounded(kvx, kvy, kvw, kvh, C_KVMGR, lw=0.7)
    text(kvx + kvw/2, kvy + kvh - 0.15, "KV Manager", fontsize=9, weight="bold")
    # KV cells
    for i in range(6):
        rounded(kvx + 0.10 + i*0.22, kvy + 0.18, 0.18, 0.30, "#9fc5e8",
                lw=0.4)
    text(kvx + kvw - 0.20, kvy + 0.33, "…", fontsize=10)

    # vertical arrow inside engine: Scheduler → Model Executor
    arrow(x0 + W/2, sy, x0 + W/2, ey + eh, color=C_EDGE, lw=1.0, mut=14)
    # Model Executor → KV Manager
    arrow(ex + ew*0.7, ey, kvx + kvw*0.5, my + mh - 0.05,
          color=C_EDGE, lw=1.0, mut=14)
    # return arrows: KV Manager → Model Executor (decode read)
    arrow(kvx + kvw*0.3, my + mh - 0.05, ex + ew*0.3, ey,
          color="#888888", ls="--", lw=0.9, mut=12)

    return dict(
        W=W, H=H,
        sched=(sx, sy, sw, sh),
        picker=(pxc, pyc),
        exec=(ex, ey, ew, eh),
        kvmgr=(kvx, kvy, kvw, kvh),
    )

# ────────── Two engines ──────────
e1 = draw_engine(0.5, 1.0, engine_id=1)
e2 = draw_engine(7.5, 1.0, engine_id=2)

# Router dispatch to engines
arrow(5.7, 7.05, e1["exec"][0] + e1["exec"][2]/2, e1["sched"][1] + e1["sched"][3] + 0.10,
      color=C_REQUEST, rad=-0.18,
      label="dispatch", lfs=9, lx=3.5, ly=6.6)
arrow(7.3, 7.05, e2["exec"][0] + e2["exec"][2]/2, e2["sched"][1] + e2["sched"][3] + 0.10,
      color=C_REQUEST, rad=0.18,
      label="dispatch", lfs=9, lx=9.5, ly=6.6)

# ────────── Host RAM KV pool (bottom) ──────────
hx, hy, hw, hh = 0.5, 0.05, 12.0, 0.78
rounded(hx, hy, hw, hh, C_HOST, lw=1.2)
text(hx + hw/2, hy + hh - 0.18, "Host RAM KV Pool  (shared, accessible by all engines)",
     fontsize=11, weight="bold")
# Some KV cells in host RAM
for i in range(20):
    rounded(hx + 0.4 + i*0.30, hy + 0.10, 0.24, 0.30, "#a8d5a8", lw=0.4)
text(hx + hw - 0.4, hy + 0.25, "…", fontsize=12)

# Mechanism 1: continuous delta checkpoint (write) — both engines write
arrow(e1["kvmgr"][0] + e1["kvmgr"][2]*0.5, e1["kvmgr"][1] - 0.05,
      e1["kvmgr"][0] + e1["kvmgr"][2]*0.5, hy + hh + 0.02,
      color=C_CHECKPT, rad=-0.12, lw=1.8,
      label="M1: delta\ncheckpoint\n(write)", lfs=8.5,
      lx=e1["kvmgr"][0] + e1["kvmgr"][2]*0.5 - 0.9, ly=0.95)
arrow(e2["kvmgr"][0] + e2["kvmgr"][2]*0.5, e2["kvmgr"][1] - 0.05,
      e2["kvmgr"][0] + e2["kvmgr"][2]*0.5, hy + hh + 0.02,
      color=C_CHECKPT, rad=0.12, lw=1.8,
      label="M1: delta\ncheckpoint\n(write)", lfs=8.5,
      lx=e2["kvmgr"][0] + e2["kvmgr"][2]*0.5 + 0.9, ly=0.95)

# Mechanism 2: V3 cheap reload (read) — Host RAM → both engines' KV Manager
arrow(hx + hw*0.30, hy + hh + 0.02,
      e1["kvmgr"][0] + e1["kvmgr"][2]*0.35, e1["kvmgr"][1] - 0.05,
      color=C_CHECKPT, ls="--", rad=0.20, lw=1.5,
      label="M2: V3 cheap\nreload (read)", lfs=8.5,
      lx=e1["kvmgr"][0] + e1["kvmgr"][2]*0.35 + 1.0, ly=0.6, lbg=True)
arrow(hx + hw*0.70, hy + hh + 0.02,
      e2["kvmgr"][0] + e2["kvmgr"][2]*0.35, e2["kvmgr"][1] - 0.05,
      color=C_CHECKPT, ls="--", rad=-0.20, lw=1.5,
      label="M2: V3 cheap\nreload (read)", lfs=8.5,
      lx=e2["kvmgr"][0] + e2["kvmgr"][2]*0.35 - 1.0, ly=0.6, lbg=True)

# Cross-engine reroute (picker preempt) — Engine 1 picker → Router → Engine 2
# 1) E1 Picker → Router (push victim to /dev/shm queue, router pulls it)
arrow(e1["picker"][0] + 0.5, e1["picker"][1] + 0.10,
      5.5, 7.05, color=C_REROUTE, ls=":", lw=1.8, rad=-0.10,
      label="(1) preempt victim", lfs=8.5,
      lx=3.0, ly=6.95, lbg=True)
# 2) Router → Engine 2 (reroute)
arrow(7.5, 7.05, e2["sched"][0] + e2["sched"][2]*0.40,
      e2["sched"][1] + e2["sched"][3] + 0.10,
      color=C_REROUTE, ls=":", lw=1.8, rad=0.10,
      label="(2) reroute to peer", lfs=8.5,
      lx=10.0, ly=6.95, lbg=True)

# Note on engine 2 about partial-token replay
text(e2["exec"][0] + e2["exec"][2]/2, e2["exec"][1] - 0.18,
     "(replays last 1 token after reload)",
     fontsize=8, style="italic", color="#666666")

# ────────── Legend ──────────
ly0 = 6.15
lx0 = 11.0
# Box legend background
rounded(lx0 - 0.05, ly0 - 0.45, 1.95, 1.10, "white", edge=C_OUTER, lw=0.7)
arrow(lx0, ly0 + 0.40, lx0 + 0.30, ly0 + 0.40, color=C_REQUEST, lw=1.4)
text(lx0 + 0.40, ly0 + 0.40, "request flow", fontsize=8, ha="left")
arrow(lx0, ly0 + 0.10, lx0 + 0.30, ly0 + 0.10, color=C_CHECKPT, lw=1.4)
text(lx0 + 0.40, ly0 + 0.10, "ckpt write/read", fontsize=8, ha="left")
arrow(lx0, ly0 - 0.20, lx0 + 0.30, ly0 - 0.20, color=C_REROUTE, ls=":", lw=1.4)
text(lx0 + 0.40, ly0 - 0.20, "preempt + reroute", fontsize=8, ha="left")

# Title
text(6.5, 7.92,
     "System Architecture — host-RAM checkpoint enables cheap cross-engine preempt-resume",
     fontsize=11.5, weight="bold")

plt.tight_layout()
plt.savefig(OUT / "fig_system_design.png", dpi=180, bbox_inches="tight")
plt.savefig(OUT / "fig_system_design.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_system_design.png")
print("       ", OUT / "fig_system_design.pdf")
