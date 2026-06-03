#!/usr/bin/env python3
"""Overlay Ferry's code changes onto vLLM's official LLMEngine diagram.
Highlights Scheduling + Model Execution (our hooks) and adds the NEW host
checkpoint store. Source: docs.vllm.ai LLMEngine arch overview (cite it)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from pathlib import Path

CORAL = "#e07a5f"; BLUE = "#2e86ab"; INK = "#1a1a1a"
HERE = Path(__file__).resolve().parent
img = Image.open(HERE / "vllm_llm_engine_official.png")
W, H = img.size            # 1874 x 1178
MB = 430                   # bottom margin for our annotations

fig, ax = plt.subplots(figsize=(W / 150, (H + MB) / 150))
ax.imshow(img, extent=[0, W, H, 0])
ax.set_xlim(-20, W + 20); ax.set_ylim(H + MB, -40)
ax.axis("off")

def hl(x, y, w, h):  # coral highlight outline on an existing box
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=2,rounding_size=24", linewidth=4.5,
        edgecolor=CORAL, facecolor="none"))

# bottom-row boxes (pixel coords measured from the official png)
hl(452, 940, 372, 232)     # Scheduling
hl(892, 940, 384, 232)     # Model Execution

# callout: Scheduling
ax.add_patch(FancyArrowPatch((420, 1300), (590, 1175), arrowstyle="-|>",
             color=CORAL, lw=2.4, mutation_scale=18,
             connectionstyle="arc3,rad=0.2"))
ax.text(60, 1330, "OURS  ·  capacity-preempt → RELOAD\n(not recompute)   sched/scheduler.py ~170",
        fontsize=12.5, color=CORAL, fontweight="bold", va="top")

# callout: Model Execution + NEW host store
store = FancyBboxPatch((1230, 1245), 600, 150,
        boxstyle="round,pad=3,rounding_size=18", linewidth=3,
        edgecolor=BLUE, facecolor="#e3eff5")
ax.add_patch(store)
ax.text(1530, 1300, "Host Checkpoint Store  (NEW)", ha="center", fontsize=13,
        color=BLUE, fontweight="bold")
ax.text(1530, 1352, "kv_checkpoint_pool.py  ~760 LOC", ha="center",
        fontsize=11, color=INK)
ax.add_patch(FancyArrowPatch((1120, 1175), (1380, 1245), arrowstyle="-|>",
             color=BLUE, lw=2.6, mutation_scale=18,
             connectionstyle="arc3,rad=-0.25"))
ax.text(1110, 1190, "publish / reload\n(ModelRunner ~830)", fontsize=12,
        color=BLUE, fontweight="bold", va="bottom", ha="left")

# footer note
ax.text(W / 2, H + MB - 25,
        "Unchanged: LLM class · API server · Input/Output Processing · model + attention kernels"
        "      |   base diagram: vLLM docs (arch overview)",
        ha="center", fontsize=10.5, color="#666666")

fig.tight_layout()
for ext in ("png", "pdf"):
    out = HERE / f"fig_vllm_official_annotated.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print("saved:", out)
