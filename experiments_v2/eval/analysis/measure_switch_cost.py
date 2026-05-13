"""Compute empirical switch_cost from engine logs.

Joins two log lines per picker fire:

  Engine A (the engine the picker fired on):
    PICKER_DIAG: victim=<vid> ... fire_ts=<wall>

  Engine B (the receiving engine that did the V3 reload):
    FT overlap V3: <newid> done — ... resume_ts=<wall>
      original_internal_req_id=<vid>

By joining `victim` (engine A) == `original_internal_req_id` (engine B)
we recover the full reroute pipeline cost (resume_ts − fire_ts).

Usage:
    python measure_switch_cost.py <dir>

  Scans every *_engine*.log under <dir>. Prints per-run and aggregate
  P50 / P95 / P99 of switch_cost in ms.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

FIRE_RE = re.compile(
    r"PICKER_DIAG: victim=(\S+).*?fire_ts=([\d.]+)"
)
RESUME_RE = re.compile(
    r"FT overlap V3: \S+ done.*?resume_ts=([\d.]+) "
    r"original_internal_req_id=(\S+)"
)


def parse_log(path: Path):
    fires: dict[str, float] = {}
    resumes: dict[str, float] = {}
    with path.open() as fp:
        for line in fp:
            m = FIRE_RE.search(line)
            if m:
                fires[m.group(1)] = float(m.group(2))
                continue
            m = RESUME_RE.search(line)
            if m:
                # original_internal_req_id keyed; that's the engine A victim id
                resumes[m.group(2)] = float(m.group(1))
    return fires, resumes


def pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(len(s) * p)))
    return s[k]


def summarize(name: str, deltas: list[float]) -> None:
    if not deltas:
        print(f"  {name}: (no matched fires)")
        return
    print(
        f"  {name}: n={len(deltas):3d} "
        f"P50={pctl(deltas, 0.50):7.0f}ms "
        f"P95={pctl(deltas, 0.95):7.0f}ms "
        f"P99={pctl(deltas, 0.99):7.0f}ms "
        f"max={max(deltas):7.0f}ms"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    args = p.parse_args()

    # Group logs by run-tag prefix: e_m1_ours_ruler_16k_qps2.0_n60_seed0
    runs: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.dir.glob("*_engine*.log")):
        tag = path.name.rsplit("_engine", 1)[0]
        runs[tag].append(path)

    if not runs:
        print(f"no engine logs found under {args.dir}", file=sys.stderr)
        return 1

    all_deltas: list[float] = []
    print(f"=== switch_cost from {args.dir} ===\n")
    for tag, paths in runs.items():
        fires_all: dict[str, float] = {}
        resumes_all: dict[str, float] = {}
        for path in paths:
            f, r = parse_log(path)
            fires_all.update(f)
            resumes_all.update(r)
        deltas = [
            (resumes_all[vid] - fires_all[vid]) * 1000.0
            for vid in fires_all
            if vid in resumes_all
        ]
        n_fire = len(fires_all)
        n_match = len(deltas)
        print(f"{tag}  (fires={n_fire}, matched={n_match})")
        summarize("run", deltas)
        all_deltas.extend(deltas)
        print()

    print(f"=== aggregate across all runs ===")
    summarize("ALL", all_deltas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
