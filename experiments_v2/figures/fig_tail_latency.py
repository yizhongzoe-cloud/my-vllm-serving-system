"""Tail latency figure — TTFT_p95 and E2E_p95 vs QPS, 3 baselines.

Why this figure: SLO_met% can miss the mechanism's actual contribution
when the SLO threshold lies near the median. Our mechanism's value
shows up most strongly in tail latency: continuous checkpoint adds
~1s to median TTFT but cheap-resume slashes p95 by ~40-50%.

Lower bars / lower curves = better."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval/results/a6000"
OUT = Path(__file__).resolve().parent


def load_metric(baseline, qps, seed):
    f = R / (f"dual_{baseline}_arxivsumm_qps{qps}_n60_seed{seed}_"
             f"arxivsumm_v2_canonical_metrics.json")
    return json.load(open(f)) if f.exists() else None


def seed_stat(baseline, qps, seeds, field):
    """Mean of d[field][p95] across given seeds (field='ttft_ms' or 'e2e_ms')."""
    vals = []
    for s in seeds:
        d = load_metric(baseline, qps, s)
        if d is not None and d.get(field, {}).get("p95") is not None:
            vals.append(d[field]["p95"])
    return np.mean(vals) if vals else None


SEEDS = [0, 1]
QPS_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BASELINES = [
    ("vllm_fcfs",       "vllm_fcfs",       "o-", "#888888"),
    ("reroute_no_ckpt", "reroute_no_ckpt", "^-", "#cc8800"),
    ("ours_no_picker",  "ours_no_picker",  "D-", "#7fb04f"),
    ("ours",            "ours",            "s-", "#1f77b4"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

# Left: TTFT_p95
ax = axes[0]
for b, label, style, color in BASELINES:
    vals = [seed_stat(b, q, SEEDS, "ttft_ms") for q in QPS_LEVELS]
    # convert to seconds for readability
    vals_s = [v / 1000.0 if v is not None else None for v in vals]
    xs = [q for q, v in zip(QPS_LEVELS, vals_s) if v is not None]
    ys = [v for v in vals_s if v is not None]
    ax.plot(xs, ys, style, label=label, color=color,
            linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("TTFT p95 (seconds, lower = better)")
ax.set_title("TTFT p95 — tail latency")
ax.set_yscale("log")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3, which="both")
ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)
ax.text(0.51, ax.get_ylim()[1] * 0.5,
        "saturation\nedge",
        color="red", alpha=0.7, fontsize=9)

# Right: E2E_p95
ax = axes[1]
for b, label, style, color in BASELINES:
    vals = [seed_stat(b, q, SEEDS, "e2e_ms") for q in QPS_LEVELS]
    vals_s = [v / 1000.0 if v is not None else None for v in vals]
    xs = [q for q, v in zip(QPS_LEVELS, vals_s) if v is not None]
    ys = [v for v in vals_s if v is not None]
    ax.plot(xs, ys, style, label=label, color=color,
            linewidth=2, markersize=8)
ax.set_xlabel("Arrival rate (QPS, Poisson)")
ax.set_ylabel("E2E p95 (seconds, lower = better)")
ax.set_title("End-to-end p95 — tail latency")
ax.set_yscale("log")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3, which="both")
ax.axvline(0.5, color="red", linestyle="--", alpha=0.4)

fig.suptitle("Two-engine arxivsumm tail latency (mean of seed 0,1)",
             fontsize=11, y=1.00)
plt.tight_layout()
plt.savefig(OUT / "fig_tail_latency.png", dpi=160, bbox_inches="tight")
plt.savefig(OUT / "fig_tail_latency.pdf", bbox_inches="tight")
plt.close()
print("saved:", OUT / "fig_tail_latency.png")

# Also print the underlying numbers for sanity check
print()
print("Underlying numbers (mean of seed 0,1):")
print(f'{"QPS":<5}', end='')
for _, label, _, _ in BASELINES:
    print(f'  {label.split()[0]:<22}', end='')
print()
print("-"*80)
for q in QPS_LEVELS:
    print(f"{q:<5}", end='')
    for b, label, _, _ in BASELINES:
        ttft = seed_stat(b, q, SEEDS, "ttft_ms")
        e2e = seed_stat(b, q, SEEDS, "e2e_ms")
        if ttft is None or e2e is None:
            cell = "n/a"
        else:
            cell = f"T{ttft/1000:.1f}s/E{e2e/1000:.1f}s"
        print(f"  {cell:<22}", end='')
    print()
