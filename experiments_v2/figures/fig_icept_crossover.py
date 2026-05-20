#!/usr/bin/env python3
"""Interception crossover figure: segment-2 resume latency vs tool-call
pause duration, for ours (host-RAM reload) / vLLM recompute / vLLM GPU
prefix-caching (APC). Mean of seed 0 and 1.

Story: APC is cheap at short pauses (cache survives) but degrades toward
recompute as the pause lengthens (LRU eviction); Ferry's host checkpoint
is flat-low regardless of pause. Renders fig_icept_crossover.{pdf,png}.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(os.environ.get(
    "EVAL_RESULTS_DIR",
    Path(__file__).resolve().parents[1] / "eval" / "results" / "a6000"))
PAUSES = [2, 30, 60]
SEEDS = [0, 1]


def seg2_p50(baseline_tag_fn):
    """Return list of mean-over-seeds seg2 p50 (s) per pause."""
    ys = []
    for p in PAUSES:
        vals = []
        for s in SEEDS:
            f = R / baseline_tag_fn(p, s)
            if f.exists():
                vals.append(json.load(open(f))["seg2_latency_s"]["p50"])
        ys.append(sum(vals) / len(vals) if vals else None)
    return ys


ours = seg2_p50(lambda p, s: f"icept_ours_qps0.3_n16_seed{s}_cross_p{p}_metrics.json")
recomp = seg2_p50(lambda p, s: f"icept_vllm_fcfs_qps0.3_n16_seed{s}_cross_p{p}_metrics.json")
apc = seg2_p50(lambda p, s: f"icept_vllm_fcfs_qps0.3_n16_seed{s}_cross_apc_p{p}_metrics.json")
print("pause:", PAUSES)
print("ours :", [None if v is None else round(v, 2) for v in ours])
print("recmp:", [None if v is None else round(v, 2) for v in recomp])
print("apc  :", [None if v is None else round(v, 2) for v in apc])

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.plot(PAUSES, recomp, "s--", color="#999999", label="vLLM (recompute)")
ax.plot(PAUSES, apc, "^-", color="#d1495b", label="vLLM + prefix cache (APC)")
ax.plot(PAUSES, ours, "o-", color="#2e86ab", linewidth=2, label="Ferry (host reload)")
ax.set_xlabel("Tool-call pause duration (s)")
ax.set_ylabel("Resume latency (s)")
ax.set_xticks(PAUSES)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
for out in (Path(__file__).resolve().parent / "fig_icept_crossover.pdf",
            Path(__file__).resolve().parent / "fig_icept_crossover.png"):
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print("saved:", out)
