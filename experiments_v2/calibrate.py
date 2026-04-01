#!/usr/bin/env python3
"""SLO calibration and load saturation point detection.

Execution order (internal dependencies, must be sequential):
  1. find_saturation_rps()       → RPS_sat per workload
  2. compute_load_levels()       → Light/Moderate/Heavy RPS
  3. measure_baseline_latency()  → TTFT_base per workload
  4. measure_recovery_gap_base() → Gap_base per workload (uses Moderate RPS)
  5. write_calibrated_config()   → fill nulls in config YAML

Usage:
    python experiments_v2/calibrate.py \
        --config experiments_v2/config_1b.yaml \
        --port 8300 \
        --output experiments_v2/config_1b_calibrated.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent


# ===================================================================
# Helpers: run a short experiment and read metrics
# ===================================================================

def _run_experiment(
    config_path: str,
    baseline: str,
    workload: str,
    rps: float,
    fault: str,
    seed: int,
    port: int,
    duration_sec: float = 60.0,
    output_dir: str | None = None,
) -> dict | None:
    """Run a single short experiment and return metrics dict."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="calibrate_")

    os.makedirs(output_dir, exist_ok=True)

    # Write a temporary config with the desired RPS as a load level
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create temp config with a "Calibrate" load level at exact RPS
    tmp_config = copy.deepcopy(config)
    tmp_config["load_levels"]["_Calibrate"] = {"pct": 0, "rps": rps}
    tmp_config["run_duration_sec"] = duration_sec
    tmp_config["warmup_sec"] = min(10.0, duration_sec / 4)
    tmp_config["seeds"] = [seed]

    tmp_config_path = os.path.join(output_dir, "_calibrate_config.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(tmp_config, f, default_flow_style=False)

    cmd = [
        sys.executable, str(_THIS_DIR / "run.py"),
        "--config", tmp_config_path,
        "--baseline", baseline,
        "--workload", workload,
        "--load", "_Calibrate",
        "--fault", fault,
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--port", str(port),
    ]

    logger.info("  Running: %s @ %.1f rps, fault=%s ...", workload, rps, fault)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode != 0:
            logger.warning("  Run failed (exit %d): %s", result.returncode,
                           result.stderr[-500:] if result.stderr else "")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("  Run timed out")
        return None

    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        logger.warning("  No metrics.json produced")
        return None

    with open(metrics_path) as f:
        return json.load(f)


# ===================================================================
# T3.1: Find saturation RPS
# ===================================================================

def find_saturation_rps(
    config_path: str,
    workload: str,
    port: int,
    rps_candidates: list[float] | None = None,
    seed: int = 42,
) -> float:
    """Find the RPS at which SLO violation exceeds 10% (No-FT, no fault).

    Returns the highest RPS that stays below the threshold.
    """
    if rps_candidates is None:
        rps_candidates = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]

    logger.info("Finding saturation RPS for %s ...", workload)

    last_good_rps = 0.0  # 0 means "nothing confirmed safe yet"
    base_tpot: float | None = None

    for rps in rps_candidates:
        tmpdir = tempfile.mkdtemp(prefix=f"cal_{workload}_{rps}_")
        metrics = _run_experiment(
            config_path, "No-FT", workload, rps, "none", seed, port,
            duration_sec=60.0, output_dir=tmpdir,
        )
        if metrics is None:
            logger.info("  RPS=%.1f: run failed, treating as saturated", rps)
            break

        tpot_p95 = metrics.get("tpot_p95_ms", 0)
        violation_rate = metrics.get("slo_violation_rate", 0)

        if base_tpot is None and tpot_p95 > 0:
            base_tpot = tpot_p95

        saturated = False
        if violation_rate > 0.10:
            saturated = True
        elif base_tpot and base_tpot > 0 and tpot_p95 > 2.0 * base_tpot:
            saturated = True

        logger.info(
            "  RPS=%5.1f: tpot_p95=%6.1fms  violation=%.1f%%  %s",
            rps, tpot_p95, violation_rate * 100,
            "SATURATED" if saturated else "ok",
        )

        if saturated:
            break
        last_good_rps = rps

    if last_good_rps <= 0:
        logger.warning(
            "  WARNING: No RPS level passed saturation check for %s. "
            "The system may be overloaded or misconfigured. Using 0.5 as fallback.",
            workload,
        )
        last_good_rps = rps_candidates[0]

    logger.info("  → RPS_sat(%s) = %.1f", workload, last_good_rps)
    return last_good_rps


def compute_load_levels(
    rps_sat: float,
    pct_config: dict[str, dict],
) -> dict[str, float]:
    """Compute absolute RPS for each load level from saturation point.

    Args:
        rps_sat: Saturation RPS (full system, not per-GPU).
        pct_config: {level_name: {pct: float, rps: ...}}

    Returns:
        {level_name: rps_value}
    """
    result = {}
    for name, spec in pct_config.items():
        pct = spec["pct"]
        result[name] = round(rps_sat * pct, 2)
    return result


# ===================================================================
# T3.2: Measure baseline latency and recovery gap
# ===================================================================

def measure_baseline_latency(
    config_path: str,
    workload: str,
    port: int,
    seed: int = 42,
) -> dict[str, float]:
    """Measure TTFT and TPOT at near-zero load (single request stream).

    Returns {ttft_base_ms, tpot_base_ms}.
    """
    logger.info("Measuring baseline latency for %s ...", workload)

    tmpdir = tempfile.mkdtemp(prefix=f"cal_base_{workload}_")
    metrics = _run_experiment(
        config_path, "No-FT", workload, 0.1, "none", seed, port,
        duration_sec=30.0, output_dir=tmpdir,
    )

    if metrics is None:
        logger.warning("  Baseline measurement failed, using defaults")
        return {"ttft_base_ms": 500.0, "tpot_base_ms": 50.0}

    ttft = metrics.get("ttft_p50_ms", 500.0)
    tpot = metrics.get("tpot_p50_ms", 50.0)

    logger.info("  → TTFT_base(%s) = %.1fms, TPOT_base = %.1fms", workload, ttft, tpot)
    return {"ttft_base_ms": ttft, "tpot_base_ms": tpot}


def measure_recovery_gap_base(
    config_path: str,
    workload: str,
    moderate_rps: float,
    port: int,
    seed: int = 42,
) -> float:
    """Measure P50 failover gap under Periodic-High + F2_Mid fault.

    Args:
        config_path: Config YAML path.
        workload: Workload name.
        moderate_rps: RPS for Moderate load (from T3.1).
        port: Server port.

    Returns:
        Gap_base_ms (P50 of failover gaps).
    """
    logger.info("Measuring recovery gap base for %s @ %.1f rps ...", workload, moderate_rps)

    tmpdir = tempfile.mkdtemp(prefix=f"cal_gap_{workload}_")
    metrics = _run_experiment(
        config_path, "Periodic-High", workload, moderate_rps, "F2_Mid",
        seed, port, duration_sec=300.0, output_dir=tmpdir,
    )

    if metrics is None:
        logger.warning("  Recovery measurement failed, using default gap=1000ms")
        return 1000.0

    gap_p50 = metrics.get("failover_gap_p50_ms", 0)
    gap_p95 = metrics.get("failover_gap_p95_ms", 0)

    # Use P50 as base; fall back to P95/2 if P50 is 0
    gap_base = gap_p50 if gap_p50 > 0 else (gap_p95 / 2 if gap_p95 > 0 else 1000.0)

    logger.info(
        "  → Gap_base(%s) = %.1fms (p50=%.1f, p95=%.1f)",
        workload, gap_base, gap_p50, gap_p95,
    )
    return gap_base


# ===================================================================
# T3.3: Write calibrated config
# ===================================================================

def calibrate_config(
    config_path: str,
    output_path: str,
    port: int,
    seed: int = 42,
    skip_recovery: bool = False,
) -> None:
    """Full calibration pipeline: prescan + baseline + recovery → write config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    workloads = config["workloads"]
    load_levels_cfg = config["load_levels"]

    # Identify non-mixed workloads (have dataset field)
    single_workloads = {
        name: wl for name, wl in workloads.items()
        if "dataset" in wl
    }

    # ---- Step 1: Find saturation RPS per workload ----
    rps_sat_map: dict[str, float] = {}
    for wname in single_workloads:
        rps_sat_map[wname] = find_saturation_rps(config_path, wname, port, seed=seed)

    # Mixed workload saturation: weighted harmonic mean of components
    for wname, wl in workloads.items():
        if "mix" in wl:
            weighted_sum = 0.0
            for entry in wl["mix"]:
                comp = entry["workload"]
                weight = entry["weight"]
                if comp in rps_sat_map and rps_sat_map[comp] > 0:
                    weighted_sum += weight / rps_sat_map[comp]
            rps_sat_map[wname] = (1.0 / weighted_sum) if weighted_sum > 0 else 5.0

    # Use minimum RPS_sat across all workloads for global load levels
    # (all workloads share the same Light/Moderate/Heavy)
    global_rps_sat = min(rps_sat_map.values()) if rps_sat_map else 5.0
    logger.info("Global RPS_sat = %.1f (min across workloads)", global_rps_sat)

    # ---- Step 2: Compute load levels ----
    rps_by_level = compute_load_levels(global_rps_sat, load_levels_cfg)
    for name, rps in rps_by_level.items():
        config["load_levels"][name]["rps"] = rps
    logger.info("Load levels: %s", rps_by_level)

    moderate_rps = rps_by_level.get("Moderate", 2.0)

    # ---- Step 3: Measure baseline latency ----
    ttft_bases: list[float] = []
    tpot_bases: list[float] = []

    for wname in single_workloads:
        latency = measure_baseline_latency(config_path, wname, port, seed=seed)
        ttft_bases.append(latency["ttft_base_ms"])
        tpot_bases.append(latency["tpot_base_ms"])

    ttft_base = max(ttft_bases) if ttft_bases else 500.0
    tpot_base = max(tpot_bases) if tpot_bases else 50.0

    # ---- Step 4: Measure recovery gap base ----
    gap_base = 1000.0  # default
    if not skip_recovery:
        gap_bases: list[float] = []
        for wname in single_workloads:
            gb = measure_recovery_gap_base(
                config_path, wname, moderate_rps, port, seed=seed,
            )
            gap_bases.append(gb)
        gap_base = max(gap_bases) if gap_bases else 1000.0
    else:
        logger.info("Skipping recovery prescan (--skip-recovery)")

    # ---- Step 5: Compute SLO values ----
    ttft_slo = round(5.0 * ttft_base, 1)
    gap_slo = round(3.0 * gap_base, 1)

    config["slo"]["ttft_ms"] = ttft_slo
    config["slo"]["failure_gap_ms"] = gap_slo
    config["slo"]["ttft_base_ms"] = round(ttft_base, 1)
    config["slo"]["gap_base_ms"] = round(gap_base, 1)

    # ---- Write output ----
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # ---- Print calibration report ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("CALIBRATION REPORT")
    logger.info("=" * 60)
    logger.info("Model: %s", config.get("model", "?"))
    logger.info("")
    logger.info("RPS saturation per workload:")
    for wname, rps in rps_sat_map.items():
        logger.info("  %-20s  RPS_sat = %6.1f", wname, rps)
    logger.info("  Global RPS_sat = %.1f", global_rps_sat)
    logger.info("")
    logger.info("Load levels:")
    for name, rps in rps_by_level.items():
        logger.info("  %-10s  %.1f rps", name, rps)
    logger.info("")
    logger.info("Baseline latency:")
    logger.info("  TTFT_base = %.1fms (max across workloads)", ttft_base)
    logger.info("  TPOT_base = %.1fms (max across workloads)", tpot_base)
    logger.info("")
    logger.info("Recovery gap:")
    logger.info("  Gap_base  = %.1fms (max across workloads)", gap_base)
    logger.info("")
    logger.info("SLO values:")
    logger.info("  TTFT SLO        = %.1fms (5x base)", ttft_slo)
    logger.info("  Failover Gap SLO = %.1fms (3x base)", gap_slo)
    logger.info("  TPOT SLO        = per-workload (50/100/200ms)")
    logger.info("")
    logger.info("Calibrated config written to: %s", output_path)
    logger.info("=" * 60)


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibrate SLO and load levels")
    parser.add_argument("--config", required=True, help="Input config YAML")
    parser.add_argument("--output", required=True, help="Output calibrated config YAML")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-recovery", action="store_true",
        help="Skip recovery gap measurement (faster, uses default gap=1000ms)",
    )
    args = parser.parse_args()

    calibrate_config(
        args.config, args.output, args.port,
        seed=args.seed, skip_recovery=args.skip_recovery,
    )


if __name__ == "__main__":
    main()
