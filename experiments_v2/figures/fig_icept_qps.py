#!/usr/bin/env python3
"""Interception QPS sweep: completed throughput and P95 work latency vs
arrival rate, Ferry (host reload) vs vLLM recompute, on the interception
workload (tool-call pause-resume). seed 0.

Story: as load rises, the baseline's reprefill-on-resume clogs the prefill
path, so its throughput plateaus / latency blows up; Ferry's cheap reload
keeps throughput climbing and latency lower. Renders fig_icept_qps.{pdf,png}.
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
QPS = ["0.2", "0.3", "0.4", "0.5", "0.6"]
X = [float(q) for q in QPS]


def series(baseline, key):
    ys = []
    for q in QPS:
        f = R / f"icept_{baseline}_qps{q}_n40_seed0_v1_metrics.json"
        if not f.exists():
            ys.append(None); continue
        d = json.load(open(f))
        ys.append(d["throughput_req_s"] if key == "thru"
                  else d["work_latency_s"]["p95"])
    return ys

o_thru, r_thru = series("ours", "thru"), series("vllm_fcfs", "thru")
o_wp95, r_wp95 = series("ours", "wp95"), series("vllm_fcfs", "wp95")
print("qps:", QPS)
print("thru ours:", [round(v, 3) if v else None for v in o_thru])
print("thru recmp:", [round(v, 3) if v else None for v in r_thru])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.3))
ax1.plot(X, o_thru, "o-", color="#2e86ab", linewidth=2, label="Ferry")
ax1.plot(X, r_thru, "s--", color="#999999", label="vLLM (recompute)")
ax1.set_xlabel("Arrival rate (req/s)"); ax1.set_ylabel("Completed throughput (req/s)")
ax1.legend(frameon=False, fontsize=9); ax1.grid(True, alpha=0.3)

ax2.plot(X, o_wp95, "o-", color="#2e86ab", linewidth=2, label="Ferry")
ax2.plot(X, r_wp95, "s--", color="#999999", label="vLLM (recompute)")
ax2.set_xlabel("Arrival rate (req/s)"); ax2.set_ylabel("P95 work latency (s)")
ax2.legend(frameon=False, fontsize=9); ax2.grid(True, alpha=0.3)
fig.tight_layout()
for out in (Path(__file__).resolve().parent / "fig_icept_qps.pdf",
            Path(__file__).resolve().parent / "fig_icept_qps.png"):
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print("saved:", out)
