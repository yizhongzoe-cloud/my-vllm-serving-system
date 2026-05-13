#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-run metrics JSON files into a summary table.

Reads all *_metrics.json in experiments_v2/eval/results/ and groups by
(experiment, baseline, dataset, qps), computing mean ± std over seeds
for SLO_met%, TTFT/TPOT percentiles, and throughput.

Usage:
  python experiments_v2/eval/analysis/aggregate.py
  python experiments_v2/eval/analysis/aggregate.py --experiment e_m3
  python experiments_v2/eval/analysis/aggregate.py --filter "ruler_64k"
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _agg(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    if len(vals) == 1:
        return {"n": 1, "mean": vals[0], "std": 0.0}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals),
    }


def load_all(filter_str: str | None = None) -> list[dict]:
    out: list[dict] = []
    for fp in sorted(RESULTS_DIR.glob("*_metrics.json")):
        if filter_str and filter_str not in fp.name:
            continue
        try:
            data = json.loads(fp.read_text())
            data["__filename"] = fp.name
            # Infer experiment id from filename prefix.
            data["__experiment"] = fp.name.split("_")[0]
            if fp.name.startswith("e_m"):
                data["__experiment"] = fp.name[:4]  # e_m1, e_m3
            elif fp.name.startswith("e_d"):
                data["__experiment"] = fp.name[:4]  # e_d1
            elif fp.name.startswith("slo_calib"):
                data["__experiment"] = "slo_calib"
            out.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def group_key(rec: dict) -> tuple:
    """Group runs by experiment / baseline / dataset / qps."""
    return (
        rec.get("__experiment", "?"),
        rec.get("baseline", rec.get("dataset", "?")),
        rec.get("dataset", "-"),
        rec.get("arrival_rate_qps", rec.get("inter_arrival_s", "-")),
    )


def fmt_pair(agg: dict, fmt: str = ".1f") -> str:
    if agg["n"] == 0:
        return "n/a"
    if agg["n"] == 1:
        return f"{agg['mean']:{fmt}}"
    return f"{agg['mean']:{fmt}} ± {agg['std']:{fmt}}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        help="Filter by experiment id (e_m1, e_m3, e_d1, slo_calib)")
    parser.add_argument("--filter", default=None,
                        help="Substring filter on filename")
    args = parser.parse_args()

    records = load_all(filter_str=args.filter)
    if args.experiment:
        records = [r for r in records
                   if r.get("__experiment") == args.experiment]
    if not records:
        print("no records matched.")
        return 0

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)

    # Print table grouped by experiment.
    by_exp: dict[str, list[tuple[tuple, list[dict]]]] = defaultdict(list)
    for k, v in sorted(groups.items()):
        by_exp[k[0]].append((k, v))

    for exp, entries in sorted(by_exp.items()):
        print(f"\n=== {exp} ===")
        # Header
        print(f"{'baseline':<22} {'dataset':<10} {'qps':<8} {'n':<3} "
              f"{'TTFT_p50':<18} {'TTFT_p95':<18} {'TPOT_p50':<14} "
              f"{'SLO_met%':<14} {'thrpt tok/s':<14}")
        print("-" * 132)
        for (_exp, baseline, dataset, qps), runs in sorted(entries):
            n = len(runs)
            ttft_p50 = _agg([r.get("ttft_ms", {}).get("p50") for r in runs])
            ttft_p95 = _agg([r.get("ttft_ms", {}).get("p95") for r in runs])
            tpot_p50 = _agg([r.get("tpot_ms", {}).get("p50") for r in runs])
            slo_met = _agg([r.get("slo_met_pct") for r in runs])
            thrpt = _agg([r.get("throughput_tok_per_s") for r in runs])
            print(f"{baseline:<22} {str(dataset):<10} {str(qps):<8} {n:<3} "
                  f"{fmt_pair(ttft_p50):<18} {fmt_pair(ttft_p95):<18} "
                  f"{fmt_pair(tpot_p50):<14} {fmt_pair(slo_met):<14} "
                  f"{fmt_pair(thrpt, '.1f'):<14}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
