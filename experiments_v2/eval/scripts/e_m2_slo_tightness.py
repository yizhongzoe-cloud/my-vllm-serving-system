#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E_M2: SLO tightness sweep (driver around e_m1_slo_sweep.py).

WHAT THE EXPERIMENT DOES
========================
Same fixed QPS (chosen in the contended-but-not-saturated regime
where the picker has room to help), three SLO-tightness levels, three
systems. Goal: show that ours' advantage grows as SLO tightens — the
3-tier (tight/normal/loose) attainment chart shifts in favor of ours
under tighter SLO.

For each (baseline, tightness, seed):
  1) Compute the per-tier SLO thresholds from the calibrated baseline
     P95 × tightness_factor × {tight, normal, loose} multipliers.
  2) Delegate to e_m1_slo_sweep.py as a subprocess.
  3) Tag output with the tightness label so figures can group by it.

SLO TIGHTNESS LEVELS
====================
The baseline P95 (from slo_calibration.py) is multiplied by:

  tight tier  = base_p95 × 1.5 × tightness_factor
  normal tier = base_p95 × 3.0 × tightness_factor
  loose tier  = base_p95 × 6.0 × tightness_factor

tightness_factor:
  "tight"   = 0.7   → all tiers 30% tighter than E_M1's default
  "default" = 1.0   → identical to E_M1's default tier definition
  "loose"   = 1.5   → all tiers 50% looser

This holds the per-tier ratio (1.5/3/6) constant — what sweeps is the
absolute SLO bar. Tight means even the loose tier becomes pressured;
loose means even tight tier comfortably hits SLO.

HOW TO RUN
==========
PYTHONPATH=. python -m experiments_v2.eval.scripts.e_m2_slo_tightness \\
    --dataset sharegpt --arrival-rate-qps 4.0 --num-requests 60 \\
    --seed 0 --baselines vllm_fcfs reroute_no_ckpt ours
    # Optional: --tightness-factors 0.7 1.0 1.5
    # Optional: --base-ttft-p95-ms 456 --base-tpot-p95-ms 22

Reads calibrated baseline P95 from the SLO calibration metrics if
--base-ttft-p95-ms / --base-tpot-p95-ms are not given.

Output: per-run metrics go to e_m1_<baseline>_<dataset>_qps<x>_n<n>_seed<s>_metrics.json
(via the underlying e_m1 script). The driver also writes
e_m2_summary_<dataset>_qps<x>_seed<s>.json indexing all runs by
(baseline, tightness) so analysis can pivot easily.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "experiments_v2" / "eval" / "results"

# Per-dataset SLO calibration metrics file. Each dataset has its own
# baseline P95 because prompt-length distributions differ.
_CALIB_FILES = {
    "sharegpt": "slo_calib_sharegpt_n30_qps0.1_seed0_metrics.json",
    "ruler_16k": "slo_calib_ruler_16k_n30_qps0.02_seed0_metrics.json",
    "ruler_64k": "slo_calib_ruler_64k_n30_qps0.04_seed0_metrics.json",
}

E_M1_MODULE = "experiments_v2.eval.scripts.e_m1_slo_sweep"


def read_calibration(dataset: str) -> tuple[float, float]:
    """Read baseline P95 TTFT and P95 TPOT from the calibration run
    for the given dataset. Returns (ttft_p95_ms, tpot_p95_ms).
    """
    calib_name = _CALIB_FILES.get(dataset)
    if calib_name is None:
        raise ValueError(
            f"No calibration mapping for dataset {dataset!r}; pass "
            "--base-ttft-p95-ms / --base-tpot-p95-ms instead."
        )
    calib_path = RESULTS_DIR / calib_name
    if not calib_path.exists():
        raise FileNotFoundError(
            f"SLO calibration metrics not found at {calib_path}. "
            "Run slo_calibration.py first or pass "
            "--base-ttft-p95-ms / --base-tpot-p95-ms."
        )
    data = json.loads(calib_path.read_text())
    ttft_p95 = data["ttft_ms"]["p95"]
    tpot_p95 = data["tpot_ms"]["p95"]
    return float(ttft_p95), float(tpot_p95)


def tier_thresholds(
    base_ttft_p95: float, base_tpot_p95: float, factor: float,
) -> tuple[dict, dict]:
    """Return (ttft_tiers, tpot_tiers) dicts keyed by tier name."""
    ttft = {
        "tight": base_ttft_p95 * 1.5 * factor,
        "normal": base_ttft_p95 * 3.0 * factor,
        "loose": base_ttft_p95 * 6.0 * factor,
    }
    tpot = {
        "tight": base_tpot_p95 * 1.5 * factor,
        "normal": base_tpot_p95 * 3.0 * factor,
        "loose": base_tpot_p95 * 6.0 * factor,
    }
    return ttft, tpot


def run_one(
    baseline: str,
    dataset: str,
    qps: float,
    n: int,
    seed: int,
    ttft: dict,
    tpot: dict,
) -> int:
    """Invoke e_m1_slo_sweep.py once. Return exit code."""
    cmd = [
        sys.executable, "-m", E_M1_MODULE,
        "--baseline", baseline,
        "--dataset", dataset,
        "--arrival-rate-qps", str(qps),
        "--num-requests", str(n),
        "--seed", str(seed),
        "--slo-mode", "tiered",
        "--ttft-slo-tight-ms", f"{ttft['tight']:.2f}",
        "--ttft-slo-normal-ms", f"{ttft['normal']:.2f}",
        "--ttft-slo-loose-ms", f"{ttft['loose']:.2f}",
        "--tpot-slo-tight-ms", f"{tpot['tight']:.2f}",
        "--tpot-slo-normal-ms", f"{tpot['normal']:.2f}",
        "--tpot-slo-loose-ms", f"{tpot['loose']:.2f}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    print(f"[E_M2] >>> {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["sharegpt", "ruler_16k", "ruler_64k"],
        required=True,
    )
    parser.add_argument(
        "--arrival-rate-qps", type=float, required=True,
        help="Fixed QPS — pick a value in the contended-but-not-"
             "saturated regime so SLO tightness sweep is meaningful.",
    )
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baselines", nargs="+",
        default=["vllm_fcfs", "reroute_no_ckpt", "ours"],
        choices=[
            "vllm_fcfs", "reroute_no_ckpt", "ours", "ours_no_picker",
        ],
    )
    parser.add_argument(
        "--tightness-factors", nargs="+", type=float,
        default=[0.7, 1.0, 1.5],
        help="One factor per SLO tightness level. Default 0.7/1.0/1.5.",
    )
    parser.add_argument(
        "--tightness-labels", nargs="+", type=str,
        default=["tight", "default", "loose"],
        help="Labels parallel to --tightness-factors, used in tags.",
    )
    parser.add_argument(
        "--base-ttft-p95-ms", type=float, default=None,
        help="Override calibrated baseline TTFT p95 (ms). If unset, "
             "read from slo_calib metrics.",
    )
    parser.add_argument(
        "--base-tpot-p95-ms", type=float, default=None,
        help="Override calibrated baseline TPOT p95 (ms).",
    )
    args = parser.parse_args()

    if len(args.tightness_factors) != len(args.tightness_labels):
        parser.error(
            "--tightness-factors and --tightness-labels must have "
            "the same length"
        )

    if args.base_ttft_p95_ms is None or args.base_tpot_p95_ms is None:
        base_ttft, base_tpot = read_calibration(args.dataset)
        if args.base_ttft_p95_ms is None:
            args.base_ttft_p95_ms = base_ttft
        if args.base_tpot_p95_ms is None:
            args.base_tpot_p95_ms = base_tpot

    print(
        f"[E_M2] baseline P95 from calibration: "
        f"TTFT={args.base_ttft_p95_ms:.1f}ms TPOT={args.base_tpot_p95_ms:.1f}ms"
    )

    summary = {
        "dataset": args.dataset,
        "qps": args.arrival_rate_qps,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "base_ttft_p95_ms": args.base_ttft_p95_ms,
        "base_tpot_p95_ms": args.base_tpot_p95_ms,
        "runs": [],
    }

    for label, factor in zip(
        args.tightness_labels, args.tightness_factors,
    ):
        ttft, tpot = tier_thresholds(
            args.base_ttft_p95_ms,
            args.base_tpot_p95_ms,
            factor,
        )
        for baseline in args.baselines:
            print(f"\n[E_M2] === tightness={label} ({factor}x) "
                  f"baseline={baseline} ===")
            ret = run_one(
                baseline=baseline,
                dataset=args.dataset,
                qps=args.arrival_rate_qps,
                n=args.num_requests,
                seed=args.seed,
                ttft=ttft,
                tpot=tpot,
            )
            # e_m1_slo_sweep.py writes its outputs under a tag that
            # doesn't include tightness — so different tightness runs
            # for the same (baseline, dataset, qps, seed) overwrite
            # each other. Rename to a tightness-tagged path right
            # after the run so all three tightness levels survive.
            src_tag = (
                f"e_m1_{baseline}_{args.dataset}_"
                f"qps{args.arrival_rate_qps}_n{args.num_requests}_"
                f"seed{args.seed}"
            )
            dst_tag = (
                f"e_m2_{baseline}_{args.dataset}_"
                f"qps{args.arrival_rate_qps}_n{args.num_requests}_"
                f"seed{args.seed}_tight-{label}"
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
                "tightness": label,
                "tightness_factor": factor,
                "baseline": baseline,
                "metrics_file": renamed_metrics,
                "exit_code": ret,
                "ttft_tiers": ttft,
                "tpot_tiers": tpot,
            })

    summary_path = (
        RESULTS_DIR
        / f"e_m2_summary_{args.dataset}_"
          f"qps{args.arrival_rate_qps}_seed{args.seed}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[E_M2] summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
