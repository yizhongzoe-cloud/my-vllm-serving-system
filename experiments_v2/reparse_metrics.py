#!/usr/bin/env python3
"""Reparse metrics.json from existing requests.csv with corrected SLO definition.

P0-impl-3a-followup (2026-04-08):
The original metric definition silently excluded admitted-but-failed requests
from slo_violation_rate (denominator was `completed`, not `admitted`). This
made No-FT look perfect under faults despite dropping 1-4% of requests.

The fix:
  - Definition: a request "violates SLO" if it was admitted by the server
    AND either failed to complete OR completed with a TTFT/TPOT/gap violation
  - Denominator: admitted requests (not total — admission rejections aren't
    the server's fault)

This script:
  1. Walks results_v2/8B/E1a_Quick_v2_HalfA + HalfB
  2. For each metrics.json, reads the sibling requests.csv
  3. Recomputes slo_violation_rate, completion_rate, etc. with the new defn
  4. Writes back metrics.json (preserving non-recomputed fields)

Usage:
  python experiments_v2/reparse_metrics.py [--dry-run] [PATH...]

If no PATH given, defaults to results_v2/8B/E1a_Quick_v2_*
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _to_bool(v: str) -> bool:
    return v == "True"


def _to_float_or_none(v: str) -> float | None:
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_int_or_zero(v: str) -> int:
    try:
        return int(v) if v else 0
    except ValueError:
        return 0


def reparse_one(metrics_path: Path, dry_run: bool = False) -> dict[str, Any]:
    """Reparse one cell. Returns dict with old/new comparison."""
    requests_path = metrics_path.parent / "requests.csv"
    if not requests_path.exists():
        return {"path": str(metrics_path), "error": "no requests.csv"}

    with open(metrics_path) as f:
        old_metrics = json.load(f)

    rows = []
    with open(requests_path) as f:
        for r in csv.DictReader(f):
            success = _to_bool(r["success"])
            error = r["error"] or ""
            output_tokens = _to_int_or_zero(r["output_tokens"])
            # P0-impl-3a-followup: recompute `admitted` with corrected definition.
            # The original csv was written with the buggy definition that excluded
            # timeout/ClientError requests even if streaming had started.
            # New definition: admitted iff (no error) OR (output_tokens > 0)
            # OR (error is HTTP 5xx — that's server-side rejection so NOT admitted)
            if not error:
                admitted = True
            elif error.startswith("HTTP"):
                admitted = False
            elif output_tokens > 0:
                admitted = True
            else:
                admitted = False
            rows.append({
                "success": success,
                "admitted": admitted,
                "ttft_violated": _to_bool(r["ttft_violated"]),
                "tpot_violated": _to_bool(r["tpot_violated"]),
                "gap_violated": _to_bool(r["gap_violated"]),
                "ttft_ms": _to_float_or_none(r["ttft_ms"]),
                "tpot_ms": _to_float_or_none(r["tpot_ms"]),
                "max_gap_ms": _to_float_or_none(r["max_gap_ms"]) or 0.0,
                "output_tokens": output_tokens,
                "affected_by_failure": _to_bool(r["affected_by_failure"]),
            })

    total = len(rows)
    if total == 0:
        return {"path": str(metrics_path), "error": "empty requests.csv"}

    admitted = [r for r in rows if r["admitted"]]
    n_admitted = len(admitted)
    successful = [r for r in rows if r["success"]]
    completed = len(successful)
    failed = total - completed

    completion_rate = completed / total

    # ── NEW: SLO violation includes admitted-but-failed requests ──
    violated_admitted_failed = sum(1 for r in admitted if not r["success"])
    violated_admitted_slo = sum(
        1 for r in admitted
        if r["success"]
        and (r["ttft_violated"] or r["tpot_violated"] or r["gap_violated"])
    )
    violated = violated_admitted_failed + violated_admitted_slo
    new_slo_violation_rate = violated / n_admitted if n_admitted > 0 else 0.0

    # ── Build new metrics dict (preserve unchanged fields) ──
    new_metrics = dict(old_metrics)  # copy everything first
    new_metrics.update({
        "total_requests": total,
        "completed": completed,
        "failed": failed,
        "admitted": n_admitted,
        "admission_rate": n_admitted / total if total > 0 else 0.0,
        "completion_rate": completion_rate,
        "slo_violation_rate": new_slo_violation_rate,
        "slo_violations_admitted_failed": violated_admitted_failed,
        "slo_violations_admitted_slo": violated_admitted_slo,
        "slo_violation_rate_failed_only": (
            violated_admitted_failed / n_admitted if n_admitted > 0 else 0.0
        ),
        "slo_violation_rate_slo_only": (
            violated_admitted_slo / n_admitted if n_admitted > 0 else 0.0
        ),
    })

    if not dry_run:
        with open(metrics_path, "w") as f:
            json.dump(new_metrics, f, indent=2)

    return {
        "path": str(metrics_path.relative_to(metrics_path.parent.parent.parent.parent.parent.parent)),
        "old_slo_v": old_metrics.get("slo_violation_rate", 0.0),
        "new_slo_v": new_slo_violation_rate,
        "completion_rate": completion_rate,
        "n_admitted": n_admitted,
        "n_failed": failed,
        "violated_failed": violated_admitted_failed,
        "violated_slo": violated_admitted_slo,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reparse metrics.json with corrected SLO definition"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes but don't write")
    parser.add_argument("paths", nargs="*",
                        help="Paths to scan (default: results_v2/8B/E1a_Quick_v2_*)")
    args = parser.parse_args()

    if args.paths:
        roots = [Path(p) for p in args.paths]
    else:
        roots = [
            Path("results_v2/8B/E1a_Quick_v2_HalfA"),
            Path("results_v2/8B/E1a_Quick_v2_HalfB"),
        ]

    metrics_files = []
    for root in roots:
        if root.exists():
            metrics_files.extend(sorted(root.rglob("metrics.json")))

    print(f"Found {len(metrics_files)} metrics.json files")
    print()
    print(f"{'Cell':<70} {'old slo%':>10} {'new slo%':>10} {'compl':>7} {'failed':>7}")
    print("-" * 110)

    results = []
    n_changed = 0
    for mf in metrics_files:
        result = reparse_one(mf, dry_run=args.dry_run)
        results.append(result)
        if "error" in result:
            print(f"  ERROR: {mf}: {result['error']}")
            continue

        old_v = result["old_slo_v"] * 100
        new_v = result["new_slo_v"] * 100
        compl = result["completion_rate"]
        failed = result["n_failed"]

        cell = str(mf.relative_to(mf.parent.parent.parent.parent.parent.parent))
        # Strip the "/42/metrics.json" suffix and the "results_v2/8B/" prefix
        cell = cell.replace("results_v2/8B/", "").replace("/42/metrics.json", "")
        marker = " ⚠" if abs(new_v - old_v) > 0.5 else ""
        print(f"{cell[:69]:<70} {old_v:>9.1f}% {new_v:>9.1f}% {compl:>7.3f} {failed:>7d}{marker}")
        if abs(new_v - old_v) > 0.001:
            n_changed += 1

    print()
    print(f"Reparsed: {len(results)} cells, {n_changed} cells with changed slo_violation_rate")
    if args.dry_run:
        print("(dry-run: no files written)")
    else:
        print("Files written.")


if __name__ == "__main__":
    main()
