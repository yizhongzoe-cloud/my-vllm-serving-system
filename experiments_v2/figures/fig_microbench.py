"""Render fig_microbench.pdf: reload vs reprefill across context lengths.

Single-panel log-log line plot showing host-RAM reload time vs full
reprefill time from 1K to 64K context (Qwen2.5-7B, A6000, batch=4), with
per-point speedup annotations.

Source: /home/yzhong76/code/vllm/experiments_zoe/results/
        figure2_prefill_a6000.csv  (vLLM single-engine prefill)
        figure2_reload_a6000.csv   (pinned host -> GPU DMA, no engine)
"""
import csv
import matplotlib.pyplot as plt
from pathlib import Path

SRC = Path("/home/yzhong76/code/vllm/experiments_zoe/results")
OUT = Path(__file__).resolve().parent


def load(csv_path, col):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            v = r[col]
            if v:
                rows[int(r["context_length"])] = float(v)
    return rows


prefill = load(SRC / "figure2_prefill_a6000.csv", "prefill_min_ms")
reload = load(SRC / "figure2_reload_a6000.csv", "reload_min_ms")

contexts = sorted(set(prefill) & set(reload))
reprefills = [prefill[c] for c in contexts]
reloads = [reload[c] for c in contexts]


def label(c):
    return f"{c // 1024}K" if c >= 1024 else str(c)


fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(contexts, reprefills, "o-", label="full reprefill",
        color="#cc4444", linewidth=2, markersize=8)
ax.plot(contexts, reloads, "s-", label="Checkpoint reload",
        color="#1f77b4", linewidth=2, markersize=8)
ax.set_xlabel("Prompt context (tokens)")
ax.set_ylabel("Time (ms, log scale)")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks(contexts)
ax.set_xticklabels([label(c) for c in contexts])
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, which="both", alpha=0.3)

for c, rl, rp in zip(contexts, reloads, reprefills):
    speedup = rp / rl
    ax.annotate(f"{speedup:.0f}$\\times$",
                xy=(c, rl), xytext=(6, -16), textcoords="offset points",
                fontsize=10, color="#1f77b4", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "fig_microbench.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "fig_microbench.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_microbench.pdf")
for c, rl, rp in zip(contexts, reloads, reprefills):
    print(f"  {label(c):>4}: reprefill={rp:8.1f}ms  reload={rl:7.2f}ms  "
          f"speedup={rp/rl:5.1f}x")
