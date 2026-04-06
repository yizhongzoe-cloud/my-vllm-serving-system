#!/usr/bin/env python3
"""Experiment suite orchestrator (v2).

Reads config YAML and runs all combinations for the specified experiment(s).
Runs sequentially (one server at a time) to avoid GPU contention.

Key v2 changes from experiments/suite.py:
  - model_tag in output directory path
  - slo_scales dimension for E6
  - load_levels as {pct, rps} dicts (null rps check)
  - --calibrate mode to run calibrate.py first

Usage:
    python experiments_v2/suite.py --config experiments_v2/config_8b_calibrated.yaml \
        --experiment E1a_Main --resume
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_matrix(config: dict, experiment_name: str) -> list[dict]:
    """Build list of run configurations for an experiment.

    Handles slo_scales dimension if present (E6).
    Output directory: results_dir / model_tag / exp_name / baseline / workload / load / fault / [slo_scale /] seed
    """
    exp = config["experiments"][experiment_name]
    seeds = config.get("seeds", [42])
    model_tag = config.get("model_tag", "unknown")
    results_dir = config.get("results_dir", "results_v2")

    slo_scales = exp.get("slo_scales")
    if slo_scales:
        scale_names = list(slo_scales.keys())
    else:
        scale_names = [None]  # no slo_scale dimension

    runs = []
    for baseline, workload, load, fault, scale, seed in itertools.product(
        exp["baselines"],
        exp["workloads"],
        exp["load_levels"],
        exp["faults"],
        scale_names,
        seeds,
    ):
        # Build output directory
        parts = [results_dir, model_tag, experiment_name, baseline, workload, load, fault]
        if scale is not None:
            parts.append(scale)
        parts.append(str(seed))
        output_dir = os.path.join(*parts)

        run = {
            "baseline": baseline,
            "workload": workload,
            "load": load,
            "fault": fault,
            "seed": seed,
            "output_dir": output_dir,
        }
        if scale is not None:
            run["slo_scale"] = scale
        runs.append(run)

    return runs


def is_run_complete(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "metrics.json"))


def run_single(config_path: str, run: dict, port: int = 8300) -> bool:
    """Execute a single experiment run as a subprocess."""
    cmd = [
        sys.executable, str(_THIS_DIR / "run.py"),
        "--config", config_path,
        "--baseline", run["baseline"],
        "--workload", run["workload"],
        "--load", run["load"],
        "--fault", run["fault"],
        "--seed", str(run["seed"]),
        "--output-dir", run["output_dir"],
        "--port", str(port),
    ]
    if "slo_scale" in run:
        cmd.extend(["--slo-scale", run["slo_scale"]])

    scale_str = f" / slo={run['slo_scale']}" if "slo_scale" in run else ""
    print(f"\n{'#'*70}")
    print(f"# Running: {run['baseline']} / {run['workload']} / "
          f"{run['load']} / {run['fault']}{scale_str} / seed={run['seed']}")
    print(f"# Output:  {run['output_dir']}")
    print(f"{'#'*70}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "OK" if success else "FAILED"
    print(f"\n  [{status}] Completed in {elapsed:.0f}s\n")
    return success


def _check_calibration(config: dict) -> bool:
    """Check that all load levels have RPS values (not null)."""
    for name, spec in config.get("load_levels", {}).items():
        if name.startswith("_"):
            continue
        if isinstance(spec, dict) and spec.get("rps") is None:
            return False
    slo = config.get("slo", {})
    if slo.get("ttft_ms") is None:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="FT Experiment Suite (v2)")
    parser.add_argument("--config", required=True,
                        help="Config YAML (should be calibrated)")
    parser.add_argument("--experiment", default="all",
                        help="Experiment name or 'all'")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed runs")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results (re-run all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run matrix without executing")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibrate.py first, then run experiments")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    config = load_config(args.config)

    # Optionally run calibration first.
    if args.calibrate:
        calibrated_path = args.config.replace(".yaml", "_calibrated.yaml")
        print(f"Running calibration → {calibrated_path}")
        cal_cmd = [
            sys.executable, str(_THIS_DIR / "calibrate.py"),
            "--config", args.config,
            "--output", calibrated_path,
            "--port", str(args.port),
        ]
        result = subprocess.run(cal_cmd, cwd=str(_REPO_ROOT))
        if result.returncode != 0:
            print("ERROR: Calibration failed")
            sys.exit(1)
        args.config = calibrated_path
        config = load_config(args.config)

    # Check calibration.
    if not _check_calibration(config):
        print("ERROR: Config is not calibrated (load_levels have rps=null or SLO missing).")
        print("Run: python experiments_v2/calibrate.py --config ... --output ...")
        sys.exit(1)

    # Determine which experiments to run.
    if args.experiment == "all":
        experiment_names = list(config["experiments"].keys())
    else:
        experiment_names = [args.experiment]

    # Build full run matrix.
    all_runs = []
    for exp_name in experiment_names:
        if exp_name not in config["experiments"]:
            print(f"ERROR: Unknown experiment '{exp_name}'")
            print(f"Available: {list(config['experiments'].keys())}")
            sys.exit(1)
        runs = build_run_matrix(config, exp_name)
        all_runs.extend(runs)

    # Dedup by output_dir (unique per experiment + combo).
    seen = set()
    unique_runs = []
    for run in all_runs:
        key = run["output_dir"]
        if key not in seen:
            seen.add(key)
            unique_runs.append(run)

    # Filter if resuming.
    if args.resume:
        remaining = [r for r in unique_runs if not is_run_complete(r["output_dir"])]
        skipped = len(unique_runs) - len(remaining)
        print(f"Resume: skipping {skipped} completed runs, {len(remaining)} remaining")
        unique_runs = remaining

    model_tag = config.get("model_tag", "?")
    print(f"\nModel: {config.get('model', '?')} ({model_tag})")
    print(f"Total runs: {len(unique_runs)}")
    print(f"Experiments: {experiment_names}")

    if args.dry_run:
        print("\nDry run — would execute:")
        for i, run in enumerate(unique_runs):
            scale = f" / {run['slo_scale']}" if "slo_scale" in run else ""
            print(f"  [{i+1:3d}] {run['baseline']:20s} / {run['workload']:15s} / "
                  f"{run['load']:10s} / {run['fault']:10s}{scale} / seed={run['seed']}")
        return

    # Execute.
    suite_start = time.time()
    results = {"ok": 0, "fail": 0}

    for i, run in enumerate(unique_runs):
        print(f"\n[{i+1}/{len(unique_runs)}]")
        ok = run_single(args.config, run, port=args.port)
        results["ok" if ok else "fail"] += 1

    elapsed = time.time() - suite_start
    print(f"\n{'='*70}")
    print(f"SUITE COMPLETE ({model_tag}): {results['ok']} ok, {results['fail']} failed, "
          f"{elapsed/60:.1f} minutes total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
