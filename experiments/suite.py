#!/usr/bin/env python3
"""Experiment suite orchestrator.

Reads config.yaml and runs all combinations for the specified experiment(s).
Runs sequentially (one server at a time) to avoid GPU contention.

Usage:
    # Run all E1 experiments
    python experiments/suite.py --experiment E1_Main

    # Run all experiments
    python experiments/suite.py --experiment all

    # Resume (skip completed runs)
    python experiments/suite.py --experiment E1_Main --resume

    # Prescan: find saturation point for load levels
    python experiments/suite.py --prescan

    # Dry run: print what would be run
    python experiments/suite.py --experiment E1_Main --dry-run
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_matrix(config: dict, experiment_name: str) -> list[dict]:
    """Build list of run configurations for an experiment."""
    exp = config["experiments"][experiment_name]
    seeds = config.get("seeds", [42])

    runs = []
    for baseline, workload, load, fault, seed in itertools.product(
        exp["baselines"],
        exp["workloads"],
        exp["load_levels"],
        exp["faults"],
        seeds,
    ):
        results_dir = config.get("results_dir", "results")
        output_dir = os.path.join(
            results_dir, experiment_name, baseline, workload, load, fault, str(seed)
        )
        runs.append({
            "baseline": baseline,
            "workload": workload,
            "load": load,
            "fault": fault,
            "seed": seed,
            "output_dir": output_dir,
        })

    return runs


def is_run_complete(output_dir: str) -> bool:
    """Check if a run has already completed (metrics.json exists)."""
    return os.path.exists(os.path.join(output_dir, "metrics.json"))


def run_single(config_path: str, run: dict, port: int = 8300) -> bool:
    """Execute a single experiment run as a subprocess."""
    cmd = [
        sys.executable, "experiments/run.py",
        "--config", config_path,
        "--baseline", run["baseline"],
        "--workload", run["workload"],
        "--load", run["load"],
        "--fault", run["fault"],
        "--seed", str(run["seed"]),
        "--output-dir", run["output_dir"],
        "--port", str(port),
    ]

    print(f"\n{'#'*70}")
    print(f"# Running: {run['baseline']} / {run['workload']} / "
          f"{run['load']} / {run['fault']} / seed={run['seed']}")
    print(f"# Output:  {run['output_dir']}")
    print(f"{'#'*70}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - start

    success = result.returncode == 0
    status = "OK" if success else "FAILED"
    print(f"\n  [{status}] Completed in {elapsed:.0f}s\n")
    return success


def run_prescan(config_path: str, port: int = 8300) -> None:
    """Run No-FT baseline at increasing RPS to find saturation point.

    Tests each workload at RPS = 0.5, 1, 2, 3, 5, 8, 10 and reports
    where performance degrades (TTFT p95 > 2x baseline).
    """
    config = load_config(config_path)
    rps_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    seed = config.get("seeds", [42])[0]

    print("="*70)
    print("PRESCAN: Finding load levels for each workload")
    print("="*70)

    for workload in config["workloads"]:
        print(f"\n--- Workload: {workload} ---")
        for rps in rps_levels:
            # Temporarily override load level.
            config["load_levels"]["_prescan"] = rps
            output_dir = os.path.join(
                config.get("results_dir", "results"),
                "prescan", workload, f"rps_{rps}",
            )

            run = {
                "baseline": "No-FT",
                "workload": workload,
                "load": "_prescan",
                "fault": "none",
                "seed": seed,
                "output_dir": output_dir,
            }

            # Save temp config.
            import tempfile
            tmp = tempfile.mktemp(suffix=".yaml")
            with open(tmp, "w") as f:
                yaml.dump(config, f)

            success = run_single(tmp, run, port=port)
            os.unlink(tmp)

            if not success:
                print(f"  RPS={rps}: FAILED")
                break

            # Read metrics.
            metrics_file = os.path.join(output_dir, "metrics.json")
            if os.path.exists(metrics_file):
                import json
                with open(metrics_file) as f:
                    m = json.load(f)
                print(f"  RPS={rps:.1f}: goodput={m['goodput']:.1f} tok/s, "
                      f"TTFT_p95={m['ttft_p95_ms']:.0f}ms, "
                      f"TPOT_p95={m['tpot_p95_ms']:.0f}ms, "
                      f"completion={m['completion_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="FT Experiment Suite")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--experiment", default="all",
                        help="Experiment name (E1_Main, E2_Recovery, ...) or 'all'")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run matrix without executing")
    parser.add_argument("--prescan", action="store_true",
                        help="Run load prescan")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.prescan:
        run_prescan(args.config, port=args.port)
        return

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

    # Dedup (same run might appear in multiple experiments).
    seen = set()
    unique_runs = []
    for run in all_runs:
        key = (run["baseline"], run["workload"], run["load"],
               run["fault"], run["seed"])
        if key not in seen:
            seen.add(key)
            unique_runs.append(run)

    # Filter if resuming.
    if args.resume:
        remaining = [r for r in unique_runs if not is_run_complete(r["output_dir"])]
        skipped = len(unique_runs) - len(remaining)
        print(f"Resume: skipping {skipped} completed runs, {len(remaining)} remaining")
        unique_runs = remaining

    print(f"\nTotal runs: {len(unique_runs)}")
    print(f"Experiments: {experiment_names}")

    if args.dry_run:
        print("\nDry run — would execute:")
        for i, run in enumerate(unique_runs):
            print(f"  [{i+1:3d}] {run['baseline']:25s} / {run['workload']:25s} / "
                  f"{run['load']:8s} / {run['fault']:10s} / seed={run['seed']}")
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
    print(f"SUITE COMPLETE: {results['ok']} ok, {results['fail']} failed, "
          f"{elapsed/60:.1f} minutes total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
