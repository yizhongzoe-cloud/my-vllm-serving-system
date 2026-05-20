"""Render e_d1_failover.pdf: recovery-gap CDF after one engine is killed.

Closed-loop micro-benchmark (§5.5): 8 concurrent 6.5-8K-token arxivsumm
requests on two Qwen2.5-14B engines (4 per engine). One engine is
SIGKILLed mid-decode, rerouting its 4 requests; we measure the recovery
gap (kill -> first resumed token on the surviving peer) for each affected
request, pooled across seeds.

Source: experiments_v2/eval/results/
        e_d1cl_{ours,reroute_no_ckpt}_n8_seed{0..3}_metrics.json
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval/results"
OUT = Path(__file__).resolve().parent
SEEDS = [0, 1, 2, 3]

SERIES = [
    ("ours", "Checkpoint reload (Ferry)", "#1f77b4", "-"),
    ("reroute_no_ckpt", "Reroute + reprefill", "#cc4444", "--"),
]


def pooled_gaps_s(baseline):
    vals = []
    for s in SEEDS:
        f = R / f"e_d1cl_{baseline}_n8_seed{s}_metrics.json"
        if not f.exists():
            continue
        m = json.load(open(f))
        for g in m.get("failover_gaps_ms", []):
            vals.append(g["gap_ms"] / 1000.0)
    return sorted(vals)


fig, ax = plt.subplots(figsize=(6.0, 3.6))
for baseline, label, color, ls in SERIES:
    gaps = pooled_gaps_s(baseline)
    if not gaps:
        print(f"WARN: no data for {baseline}")
        continue
    ys = np.arange(1, len(gaps) + 1) / len(gaps)
    # step CDF starting at 0
    xs = [0.0] + gaps
    ys = [0.0] + list(ys)
    ax.step(xs, ys, where="post", label=label, color=color,
            linewidth=2.2, linestyle=ls)
    mean_v = np.mean(gaps)
    ax.axvline(mean_v, color=color, alpha=0.35, linewidth=1, linestyle=":")
    print(f"{baseline:18s} n={len(gaps)} mean={mean_v:.1f}s "
          f"p50={np.percentile(gaps,50):.1f}s max={max(gaps):.1f}s")

ax.set_xlabel("Recovery gap (s)")
ax.set_ylabel("Fraction of affected requests")
ax.set_ylim(0, 1.02)
ax.set_xlim(left=0)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "e_d1_failover.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "e_d1_failover.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "e_d1_failover.pdf")
