#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E_M4: Picker ablation (driver around e_m1_slo_sweep.py).

WHAT THE EXPERIMENT DOES
========================
Isolates the contribution of the slack-based priority preempt picker
by running ours with picker ON vs OFF, otherwise identical (same
checkpoint pool, same V3 reload path, same cross-engine reroute,
same router). Sweeps QPS so we see whether picker's benefit is QPS-
dependent (we expect bigger gap at higher contention).

Baselines:
  ours           = SLO_PRIORITY_PREEMPT=1  (full system)
  ours_no_picker = SLO_PRIORITY_PREEMPT=0  (everything else same)

For each (baseline, qps, seed): delegates to e_m1_slo_sweep.py.

HOW TO RUN
==========
PYTHONPATH=. python -m experiments_v2.eval.scripts.e_m4_picker_ablation \\
    --dataset sharegpt --num-requests 60 --seed 0 \\
    --qps-sweep 1.0 2.0 4.0 6.0 8.0 \\
    --ttft-slo-tight-ms 684 --ttft-slo-normal-ms 1368 \\
    --ttft-slo-loose-ms 2736 \\
    --tpot-slo-tight-ms 33 --tpot-slo-normal-ms 66 \\
    --tpot-slo-loose-ms 132

Output: per-run metrics via the underlying e_m1 script. Also writes
e_m4_summary_<dataset>_seed<s>.json indexing (baseline, qps) for
easy pivot in plots.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "experiments_v2" / "eval" / "results"
E_M1_MODULE = "experiments_v2.eval.scripts.e_m1_slo_sweep"


def run_one(
    baseline: str,
    dataset: str,
    qps: float,
    n: int,
    seed: int,
    ttft: dict,
    tpot: dict,
) -> int:
    cmd = [
        sys.executable, "-m", E_M1_MODULE,
        "--baseline", baseline,
        "--dataset", dataset,
        "--arrival-rate-qps", str(qps),
        "--num-requests", str(n),
        "--seed", str(seed),
        "--slo-mode", "tiered",
        "--ttft-slo-tight-ms", f"{ttft['tight']}",
        "--ttft-slo-normal-ms", f"{ttft['normal']}",
        "--ttft-slo-loose-ms", f"{ttft['loose']}",
        "--tpot-slo-tight-ms", f"{tpot['tight']}",
        "--tpot-slo-normal-ms", f"{tpot['normal']}",
        "--tpot-slo-loose-ms", f"{tpot['loose']}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    print(f"[E_M4] >>> {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sharegpt", "ruler_16k", "ruler_64k"],
        required=True,
    )
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--qps-sweep", nargs="+", type=float,
        default=[1.0, 2.0, 4.0, 6.0, 8.0],
    )
    parser.add_argument(
        "--baselines", nargs="+",
        default=["ours_no_picker", "ours"],
        choices=["ours_no_picker", "ours"],
        help="Picker ablation usually only needs these two — other "
             "baselines (fcfs, reroute_no_ckpt) are covered by E_M1.",
    )
    parser.add_argument("--ttft-slo-tight-ms", type=float, required=True)
    parser.add_argument("--ttft-slo-normal-ms", type=float, required=True)
    parser.add_argument("--ttft-slo-loose-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-tight-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-normal-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-loose-ms", type=float, required=True)
    args = parser.parse_args()

    ttft = {
        "tight": args.ttft_slo_tight_ms,
        "normal": args.ttft_slo_normal_ms,
        "loose": args.ttft_slo_loose_ms,
    }
    tpot = {
        "tight": args.tpot_slo_tight_ms,
        "normal": args.tpot_slo_normal_ms,
        "loose": args.tpot_slo_loose_ms,
    }

    summary = {
        "dataset": args.dataset,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "qps_sweep": args.qps_sweep,
        "ttft_tiers": ttft,
        "tpot_tiers": tpot,
        "runs": [],
    }

    for qps in args.qps_sweep:
        for baseline in args.baselines:
            print(f"\n[E_M4] === qps={qps} baseline={baseline} ===")
            ret = run_one(
                baseline=baseline,
                dataset=args.dataset,
                qps=qps,
                n=args.num_requests,
                seed=args.seed,
                ttft=ttft,
                tpot=tpot,
            )
            # e_m1_slo_sweep.py writes outputs under an e_m1_* tag.
            # If E_M4 also leaves them there, the next E_M1 sweep
            # would overwrite them (and vice versa). Rename to an
            # e_m4_* tag so the ablation data stays separate.
            src_tag = (
                f"e_m1_{baseline}_{args.dataset}_qps{qps}_"
                f"n{args.num_requests}_seed{args.seed}"
            )
            dst_tag = (
                f"e_m4_{baseline}_{args.dataset}_qps{qps}_"
                f"n{args.num_requests}_seed{args.seed}"
            )
            renamed_metrics = None
            for sfx in (
                "metrics.json", "engine0.log",
                "engine1.log", "router.log",
            ):
                src = RESULTS_DIR / f"{src_tag}_{sfx}"
                dst = RESULTS_DIR / f"{dst_tag}_{sfx}"
                if src.exists():
                    src.rename(dst)
                    if sfx == "metrics.json":
                        renamed_metrics = dst.name
            summary["runs"].append({
                "qps": qps,
                "baseline": baseline,
                "metrics_file": renamed_metrics,
                "exit_code": ret,
            })

    summary_path = (
        RESULTS_DIR
        / f"e_m4_summary_{args.dataset}_seed{args.seed}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[E_M4] summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
