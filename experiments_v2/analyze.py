#!/usr/bin/env python3
"""Analyze experiment results and generate figures/tables.

Reads results directories produced by run.py/suite.py, computes
summary statistics, and generates the figures required by the
experiment plan.

Usage:
    python experiments/analyze.py results/E1_Main --output figures/

    # Generate all figures from all experiments
    python experiments/analyze.py results/ --output figures/ --all
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, skipping plots")


# ============================================================
# Data loading
# ============================================================

def load_all_runs(results_dir: str) -> list[dict]:
    """Recursively find and load all completed runs under results_dir."""
    runs = []
    for root, dirs, files in os.walk(results_dir):
        if "metrics.json" in files:
            run = {}
            with open(os.path.join(root, "metrics.json")) as f:
                run["metrics"] = json.load(f)
            if "run_meta.json" in files:
                with open(os.path.join(root, "run_meta.json")) as f:
                    run["meta"] = json.load(f)
            if "requests.csv" in files:
                run["requests"] = _load_csv(os.path.join(root, "requests.csv"))
            if "epochs.csv" in files:
                run["epochs"] = _load_csv(os.path.join(root, "epochs.csv"))
            if "recoveries.json" in files:
                with open(os.path.join(root, "recoveries.json")) as f:
                    run["recoveries"] = json.load(f)
            run["dir"] = root
            runs.append(run)
    return runs


def _load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_recoveries(run: dict) -> list[dict]:
    """Load recoveries.json for a run if it is not already cached."""
    recoveries = run.get("recoveries", [])
    if not recoveries and run.get("dir"):
        recoveries_file = os.path.join(run["dir"], "recoveries.json")
        if os.path.exists(recoveries_file):
            with open(recoveries_file) as f:
                recoveries = json.load(f)
            run["recoveries"] = recoveries
    return recoveries


def _has_subsecond_precision(ts: float) -> bool:
    """Heuristic: true wall-clock event times include subsecond precision."""
    return abs(ts - round(ts)) > 1e-6


def _parse_bool_field(value: object) -> bool:
    """Parse a CSV/JSON bool-ish field into a Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _get_direct_hit_requests(run: dict) -> list[dict]:
    """Return requests with known ownership that directly hit the failed GPU."""
    if run.get("meta", {}).get("fault", "none") == "none":
        return []

    requests = run.get("requests", [])
    if not requests:
        return []

    required_fields = {"ownership_known", "affected_by_failure", "success",
                       "max_gap_ms"}
    missing = [
        field for field in required_fields
        if field not in requests[0]
    ]
    if missing:
        raise ValueError(
            f"{run.get('dir', '<run>')}: requests.csv is missing required "
            f"columns for direct-hit failover-gap analysis: {missing}. "
            "Re-run experiments with the updated runner before analyzing "
            "failover gaps."
        )

    return [
        req for req in requests
        if _parse_bool_field(req.get("ownership_known"))
        and _parse_bool_field(req.get("affected_by_failure"))
    ]


def _get_direct_hit_success_counts(run: dict) -> tuple[int, int]:
    """Return (successful_direct_hit_requests, total_direct_hit_requests)."""
    direct_hit_requests = _get_direct_hit_requests(run)
    total = len(direct_hit_requests)
    success = sum(
        1 for req in direct_hit_requests
        if _parse_bool_field(req.get("success"))
    )
    return success, total


def group_runs(runs: list[dict], key_fields: list[str]) -> dict[tuple, list[dict]]:
    """Group runs by specified metadata fields."""
    groups = defaultdict(list)
    for run in runs:
        meta = run.get("meta", {})
        key = tuple(meta.get(k, "") for k in key_fields)
        groups[key].append(run)
    return dict(groups)


# ============================================================
# Summary tables
# ============================================================

def make_summary_table(runs: list[dict], output_path: str) -> None:
    """Generate summary CSV table across baselines × workloads × loads."""
    rows = []
    for run in runs:
        meta = run.get("meta", {})
        m = run["metrics"]
        timing = _get_recovery_detection_metrics(run)
        rows.append({
            "baseline": meta.get("baseline", ""),
            "workload": meta.get("workload", ""),
            "load": meta.get("load_level", ""),
            "fault": meta.get("fault", ""),
            "seed": meta.get("seed", ""),
            "goodput": m.get("goodput", 0),
            "ttft_p50": m.get("ttft_p50_ms", 0),
            "ttft_p95": m.get("ttft_p95_ms", 0),
            "ttft_p99": m.get("ttft_p99_ms", 0),
            "tpot_p50": m.get("tpot_p50_ms", 0),
            "tpot_p95": m.get("tpot_p95_ms", 0),
            "tpot_p99": m.get("tpot_p99_ms", 0),
            "slo_violation_rate": m.get("slo_violation_rate", 0),
            "completion_rate": m.get("completion_rate", 0),
            "failover_gap_p95": _get_recomputed_failover_gap_p95(run),
            "declare_failed_ms": (
                timing["declare_failed_ms"]
                if timing["declare_failed_ms"] is not None else ""
            ),
            "monitor_detect_ms": (
                timing["monitor_detect_ms"]
                if timing["monitor_detect_ms"] is not None else ""
            ),
            "failover_start_ms": (
                timing["failover_start_ms"]
                if timing["failover_start_ms"] is not None else ""
            ),
            "time_to_stable": m.get("time_to_stable_sec", ""),
            "total_requests": m.get("total_requests", 0),
            "completed": m.get("completed", 0),
        })

    if not rows:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary table: {output_path}")


def make_mean_table(runs: list[dict], output_path: str) -> None:
    """Generate mean-across-seeds summary table."""
    grouped = group_runs(runs, ["baseline", "workload", "load_level", "fault"])
    rows = []

    for key, group in sorted(grouped.items()):
        baseline, workload, load, fault = key
        metrics_list = [r["metrics"] for r in group]

        row = {
            "baseline": baseline,
            "workload": workload,
            "load": load,
            "fault": fault,
            "n_seeds": len(group),
        }

        for field in ["goodput", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms",
                       "tpot_p95_ms", "slo_violation_rate", "completion_rate"]:
            vals = [m.get(field, 0) for m in metrics_list]
            row[f"{field}_mean"] = np.mean(vals) if vals else 0
            row[f"{field}_std"] = np.std(vals) if vals else 0

        # Use recomputed direct-hit failover gap from successful direct-hit
        # requests only. Failed direct-hit requests are reported separately
        # via succ x/y annotations in the figures.
        gap_vals = [_get_recomputed_failover_gap_p95(r) for r in group]
        row["failover_gap_p95_ms_mean"] = np.mean(gap_vals) if gap_vals else 0
        row["failover_gap_p95_ms_std"] = np.std(gap_vals) if gap_vals else 0

        for field in [
            "declare_failed_ms",
            "monitor_detect_ms",
            "failover_start_ms",
        ]:
            vals = [
                timing[field]
                for timing in (_get_recovery_detection_metrics(r) for r in group)
                if timing[field] is not None
            ]
            row[f"{field}_mean"] = np.mean(vals) if vals else ""
            row[f"{field}_std"] = np.std(vals) if vals else ""

        rows.append(row)

    if not rows:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Mean table: {output_path}")


# ============================================================
# Plotting
# ============================================================

# Consistent colors and markers for baselines.
BASELINE_STYLE = {
    # v2 names
    "No-FT":              {"color": "#888888", "marker": "x", "ls": "--"},
    "Periodic-Low":       {"color": "#E69F00", "marker": "s", "ls": "-"},
    "Periodic-High":      {"color": "#D55E00", "marker": "^", "ls": "-"},
    "Benders-Only":       {"color": "#009E73", "marker": "D", "ls": "-"},
    "Adaptive-Only":      {"color": "#CC79A7", "marker": "p", "ls": "-"},
    "Our-System":         {"color": "#0072B2", "marker": "o", "ls": "-"},
    # v1 names (backward compat)
    "Fixed-Low-CKPT":     {"color": "#E69F00", "marker": "s", "ls": "-"},
    "Fixed-High-CKPT":    {"color": "#D55E00", "marker": "^", "ls": "-"},
    "Robust-Routing-Only":{"color": "#009E73", "marker": "D", "ls": "-"},
    "Checkpoint-Only":    {"color": "#CC79A7", "marker": "p", "ls": "-"},
}


def _get_style(baseline: str) -> dict:
    return BASELINE_STYLE.get(baseline, {"color": "black", "marker": ".", "ls": "-"})


def plot_goodput_by_load(runs: list[dict], output_dir: str) -> None:
    """Figure: Goodput vs load level for each baseline (one subplot per workload)."""
    if not HAS_MPL:
        return

    grouped = group_runs(runs, ["baseline", "workload", "load_level", "fault"])
    workloads = sorted(set(r.get("meta", {}).get("workload", "") for r in runs))
    # Only show load levels that actually have data
    all_loads = set(r.get("meta", {}).get("load_level", "") for r in runs)
    load_order_pref = ["Light", "Moderate", "Heavy", "Low", "Medium", "High"]
    load_order = [l for l in load_order_pref if l in all_loads]

    for fault_filter in ["none", "F2_Mid"]:
        fig, axes = plt.subplots(1, len(workloads), figsize=(5*len(workloads), 4),
                                 sharey=True, squeeze=False)

        for wi, workload in enumerate(workloads):
            ax = axes[0][wi]
            ax.set_title(workload.replace("_", " "), fontsize=10)
            ax.set_xlabel("Load Level")
            if wi == 0:
                ax.set_ylabel("Goodput (tok/s)")

            for baseline in BASELINE_STYLE:
                xs, ys, yerrs = [], [], []
                for li, load in enumerate(load_order):
                    key = (baseline, workload, load, fault_filter)
                    group = grouped.get(key, [])
                    if group:
                        vals = [r["metrics"]["goodput"] for r in group]
                        xs.append(li)
                        ys.append(np.mean(vals))
                        yerrs.append(np.std(vals))

                if xs:
                    style = _get_style(baseline)
                    ax.errorbar(xs, ys, yerr=yerrs, label=baseline,
                                color=style["color"], marker=style["marker"],
                                ls=style["ls"], capsize=3, markersize=6)

            ax.set_xticks(range(len(load_order)))
            ax.set_xticklabels(load_order)

        axes[0][-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fname = f"goodput_by_load_{fault_filter}.pdf"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fname}")


def plot_slo_violation(runs: list[dict], output_dir: str) -> None:
    """Figure: SLO violation rate bar chart per baseline."""
    if not HAS_MPL:
        return

    grouped = group_runs(runs, ["baseline", "fault"])
    baselines = list(BASELINE_STYLE.keys())

    for fault_filter in ["none", "F2_Mid"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        x_pos = []
        labels = []
        vals = []
        errs = []
        colors = []

        for i, bl in enumerate(baselines):
            key = (bl, fault_filter)
            group = [r for k, rs in grouped.items()
                     for r in rs if k[0] == bl and k[1] == fault_filter]
            if group:
                v = [r["metrics"]["slo_violation_rate"] for r in group]
                x_pos.append(i)
                labels.append(bl)
                vals.append(np.mean(v) * 100)
                errs.append(np.std(v) * 100)
                colors.append(_get_style(bl)["color"])

        if x_pos:
            ax.bar(x_pos, vals, yerr=errs, color=colors, capsize=3, alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("SLO Violation Rate (%)")
            ax.set_title(f"SLO Violations (fault={fault_filter})")
            fig.tight_layout()
            fname = f"slo_violation_{fault_filter}.pdf"
            fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
            print(f"  Plot: {fname}")
        plt.close(fig)


def _get_actual_fault_time(run: dict) -> float | None:
    """Return the earliest known fault-start timestamp for a run.

    We prefer the first fault-related event we know about:
    - client-side ``fault_injection_time`` for planned SIGKILL runs
    - server-side ``monitor_observed`` / ``failure_declared`` /
      ``failover_start`` for crashes that happen before the planned
      injection fires

    Using the earliest known timestamp keeps failover-gap accounting
    aligned with the actual failure onset rather than the later
    control-plane declaration.
    """
    meta = run.get("meta", {})
    candidates: list[float] = []

    ft = meta.get("fault_injection_time")
    if ft is not None:
        candidates.append(float(ft))

    recoveries = _load_recoveries(run)
    for ev in recoveries:
        if ev.get("type") not in {
            "monitor_observed",
            "failure_declared",
            "failover_start",
        }:
            continue
        ts = ev.get("wall_time")
        if ts is not None:
            candidates.append(float(ts))
            continue

        ts = ev.get("timestamp")
        if ts is not None:
            ts = float(ts)
            if _has_subsecond_precision(ts):
                candidates.append(ts)

    return min(candidates) if candidates else None


def _recompute_failover_gaps(run: dict) -> list[float]:
    """Recompute per-request failover gaps from requests.csv.

    Only direct-hit requests are included: those with
    ``ownership_known=True`` and ``affected_by_failure=True`` in requests.csv.
    The gap metric is computed on successful direct-hit requests only; failed
    direct-hit requests are tracked separately via succ x/y annotations in the
    figures instead of being folded into the gap metric with a penalty.
    """
    direct_hit_requests = _get_direct_hit_requests(run)

    gaps = []
    for req in direct_hit_requests:
        if not _parse_bool_field(req.get("success")):
            continue
        max_gap = float(req.get("max_gap_ms", 0) or 0)
        gaps.append(max_gap)

    return gaps


def _get_recomputed_failover_gap_p95(run: dict) -> float:
    """Return the recomputed p95 failover gap for a run (cached on run dict).

    Uses _recompute_failover_gaps() which includes successful direct-hit
    requests only.
    """
    cached = run.get("_recomputed_gap_p95")
    if cached is not None:
        return cached
    gaps = _recompute_failover_gaps(run)
    if gaps:
        val = float(np.percentile(gaps, 95))
    else:
        val = 0.0
    run["_recomputed_gap_p95"] = val
    return val


def _get_recovery_detection_metrics(run: dict) -> dict[str, float | None]:
    """Return precise recovery timing markers for a run.

    The main detection metric is declare_failed_ms. The other two are
    debugging aids that help explain whether time is spent in monitor
    observation or in control-plane startup before failover begins.
    """
    cached = run.get("_recovery_detection_metrics")
    if cached is not None:
        return cached

    # Use the earliest known fault-start timestamp, not just the planned
    # injection time. This keeps W2-style pre-injection crashes measurable.
    fault_time = _get_actual_fault_time(run)
    metrics = {
        "declare_failed_ms": None,
        "monitor_detect_ms": None,
        "failover_start_ms": None,
    }
    if fault_time is None:
        run["_recovery_detection_metrics"] = metrics
        return metrics

    recoveries = _load_recoveries(run)
    monitor_wt = None
    declared_wt = None
    failover_start_wt = None

    for ev in recoveries:
        wt = ev.get("wall_time")
        if wt is None:
            continue
        wt = float(wt)
        etype = ev.get("type", "")
        if etype == "monitor_observed":
            monitor_wt = wt if monitor_wt is None else min(monitor_wt, wt)
        elif etype == "failure_declared":
            declared_wt = wt if declared_wt is None else min(declared_wt, wt)
        elif etype == "failover_start":
            failover_start_wt = (
                wt if failover_start_wt is None else min(failover_start_wt, wt)
            )

    metrics = {
        "declare_failed_ms": (
            max(0.0, (declared_wt - fault_time) * 1000)
            if declared_wt is not None else None
        ),
        "monitor_detect_ms": (
            max(0.0, (monitor_wt - fault_time) * 1000)
            if monitor_wt is not None else None
        ),
        "failover_start_ms": (
            max(0.0, (failover_start_wt - fault_time) * 1000)
            if failover_start_wt is not None else None
        ),
    }
    run["_recovery_detection_metrics"] = metrics
    return metrics


def plot_failover_gap(runs: list[dict], output_dir: str) -> None:
    """Figure: p95 direct-hit failover gap bar chart, split by workload.

    Recomputes gaps from per-request data (not metrics.json) using only
    successful direct-hit requests. Each bar is annotated with succ x/y,
    where y is the total direct-hit count and x is the successful subset.
    """
    if not HAS_MPL:
        return

    baselines = list(BASELINE_STYLE.keys())

    # Discover workloads present in fault runs.
    workloads = sorted({
        r.get("meta", {}).get("workload", "")
        for r in runs
        if r.get("meta", {}).get("fault", "none") != "none"
    })
    workloads = [w for w in workloads if w]

    if not workloads:
        return

    n_wl = len(workloads)
    fig, axes = plt.subplots(1, n_wl, figsize=(6 * n_wl, 4), squeeze=False)

    for wl_idx, wl in enumerate(workloads):
        ax = axes[0, wl_idx]
        x_pos = []
        labels = []
        vals = []
        errs = []
        colors = []
        succ_labels = []

        for i, bl in enumerate(baselines):
            group = [
                r for r in runs
                if r.get("meta", {}).get("baseline") == bl
                and r.get("meta", {}).get("fault", "none") != "none"
                and r.get("meta", {}).get("workload") == wl
            ]
            if not group:
                continue

            # Per-seed p95 gaps.
            seed_p95s = []
            succ_count = 0
            total_direct_hit = 0
            for r in group:
                gaps = _recompute_failover_gaps(r)
                if gaps:
                    seed_p95s.append(float(np.percentile(gaps, 95)))
                # No gaps → skip (NaN), don't record 0.0
                succ, total = _get_direct_hit_success_counts(r)
                succ_count += succ
                total_direct_hit += total

            x_pos.append(i)
            labels.append(bl)
            if seed_p95s:
                vals.append(np.mean(seed_p95s))
                errs.append(np.std(seed_p95s))
            else:
                vals.append(0.0)
                errs.append(0.0)
            colors.append(_get_style(bl)["color"])
            succ_labels.append(f"succ {succ_count}/{total_direct_hit}")

        if not x_pos:
            continue

        bars = ax.bar(
            x_pos, vals, yerr=errs, color=colors, capsize=3, alpha=0.8
        )

        # Annotate direct-hit success counts on each bar.
        for bar, label in zip(bars, succ_labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                label,
                ha="center", va="bottom", fontsize=7, color="#444444",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Direct-Hit Failover Gap p95 (ms)")
        ax.set_title(f"Direct-Hit Failover Gap — {wl}")

    fig.tight_layout()
    fname = "failover_gap_p95.pdf"
    fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {fname}")


def plot_controller_overhead(runs: list[dict], output_dir: str) -> None:
    """Figure: Solver epoch latency vs active request count (E5)."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    all_request_count = []
    all_solve_times = []

    for run in runs:
        if run.get("meta", {}).get("baseline") != "Our-System":
            continue
        epochs = run.get("epochs", [])
        for e in epochs:
            try:
                pending = int(e.get("num_pending", 0))
                active = int(e.get("num_active", 0))
                solve_t = float(e.get("total_solve_time_sec", 0))
                if solve_t > 0:
                    all_request_count.append(pending + active)
                    all_solve_times.append(solve_t * 1000)  # ms
            except (ValueError, TypeError):
                continue

    if all_request_count:
        ax.scatter(all_request_count, all_solve_times, alpha=0.3, s=10, color="#0072B2")
        ax.set_xlabel("Requests in Epoch (pending + active)")
        ax.set_ylabel("Solver Epoch Latency (ms)")
        ax.set_title("Controller Overhead per Epoch")
        fig.tight_layout()
        fname = "controller_overhead.pdf"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        print(f"  Plot: {fname}")
    plt.close(fig)


def plot_recovery_breakdown(runs: list[dict], output_dir: str) -> None:
    """Figure: Stacked bar of recovery time breakdown per baseline, split by
    workload.

    Breakdown phases: detection, KV restore, replay.

    Data source:
    - Per-request CSV fields written by run.py from direct system-side phase
      timestamps. `affected_by_failure` uses strict GPU-hit semantics, and
      `ownership_known` indicates whether route/reroute data was sufficient
      to determine GPU ownership.
    """
    if not HAS_MPL:
        return

    baselines = list(BASELINE_STYLE.keys())
    ft_baselines = [bl for bl in baselines if bl != "No-FT"]

    def _request_lookup_keys(request_id: str | None) -> set[str]:
        if not request_id:
            return set()

        keys = {request_id}
        if request_id.startswith("chatcmpl-") and request_id.count("-") >= 2:
            keys.add(request_id.rsplit("-", 1)[0])

        for candidate in tuple(keys):
            for prefix in ("chatcmpl-", "cmpl-", "resp-"):
                if candidate.startswith(prefix):
                    keys.add(candidate[len(prefix):])
                else:
                    keys.add(f"{prefix}{candidate}")

        return {k for k in keys if k}

    # Discover workloads present in fault runs.
    workloads = sorted({
        r.get("meta", {}).get("workload", "")
        for r in runs
        if r.get("meta", {}).get("fault", "none") != "none"
    })
    workloads = [w for w in workloads if w]

    if not workloads:
        return

    n_wl = len(workloads)
    fig, axes = plt.subplots(1, n_wl, figsize=(8 * n_wl, 4), squeeze=False)

    for wl_idx, wl in enumerate(workloads):
        ax = axes[0, wl_idx]
        detection_means = []
        restore_means = []
        replay_means = []
        labels = []

        for bl in ft_baselines:
            group = [r for r in runs
                     if r.get("meta", {}).get("baseline") == bl
                     and r.get("meta", {}).get("fault", "none") != "none"
                     and r.get("meta", {}).get("workload") == wl]
            if not group:
                continue

            detect_vals: list[float] = []
            restore_vals: list[float] = []
            replay_vals: list[float] = []

            for run in group:
                # Per-request data from CSV — directly use phase times
                # computed by run.py from system-side timestamps.
                requests = run.get("requests", [])
                for req in requests:
                    if req.get("affected_by_failure") not in (True, "True", "true"):
                        continue

                    det_raw = req.get("detection_ms")
                    replay_raw = req.get("replay_time_ms")
                    # Skip requests missing direct phase measurements.
                    if det_raw in (None, "", "None") or replay_raw in (None, "", "None"):
                        continue
                    det = float(det_raw)
                    restore_t = float(req.get("restore_time_ms") or 0)
                    replay_t = float(replay_raw)
                    if det + restore_t + replay_t <= 0:
                        continue

                    detect_vals.append(det)
                    restore_vals.append(restore_t)
                    replay_vals.append(replay_t)

            if detect_vals or restore_vals or replay_vals:
                labels.append(bl)
                detection_means.append(np.mean(detect_vals) if detect_vals else 0.0)
                restore_means.append(np.mean(restore_vals) if restore_vals else 0.0)
                replay_means.append(np.mean(replay_vals) if replay_vals else 0.0)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.5

        ax.bar(x, detection_means, width, label="Detection", color="#E69F00")
        ax.bar(x, restore_means, width, bottom=detection_means,
               label="KV Restore", color="#56B4E9")
        bottoms_2 = [d + r for d, r in zip(detection_means, restore_means)]
        ax.bar(x, replay_means, width, bottom=bottoms_2,
               label="Replay", color="#009E73")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Recovery Time (ms)")
        ax.set_title(f"Recovery Breakdown — {wl}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fname = "recovery_breakdown.pdf"
    fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {fname}")


def plot_ablation(runs: list[dict], output_dir: str) -> None:
    """Figure: Ablation — Our-System vs routing-only vs checkpoint-only.

    Splits by workload (rows) × load level (columns) to avoid mixing
    short and long request results into the same bar.
    """
    if not HAS_MPL:
        return

    ablation_baselines = [
        "Our-System", "Benders-Only", "Adaptive-Only",
        "Periodic-Low", "Periodic-High",
        # v1 compat
        "Robust-Routing-Only", "Checkpoint-Only",
        "Fixed-Low-CKPT", "Fixed-High-CKPT",
    ]

    # Discover workloads present in ablation runs.
    workloads = sorted({
        r.get("meta", {}).get("workload", "")
        for r in runs
        if r.get("meta", {}).get("baseline") in ablation_baselines
    } - {""})
    if not workloads:
        workloads = ["W1_Chat", "W4_Mixed"]

    for fault_filter in ["none", "F2_Mid"]:
        load_levels = ["Moderate", "Heavy", "Medium", "High"]
        n_rows = len(workloads)
        n_cols = len(load_levels)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 squeeze=False)

        for wi, wl in enumerate(workloads):
            for li, load in enumerate(load_levels):
                ax = axes[wi][li]
                ax.set_title(f"{wl} / Load={load} / fault={fault_filter}",
                             fontsize=9)
                x = np.arange(len(ablation_baselines))

                goodputs = []
                errs = []
                for bl in ablation_baselines:
                    group = [r for r in runs
                             if r.get("meta", {}).get("baseline") == bl
                             and r.get("meta", {}).get("workload") == wl
                             and r.get("meta", {}).get("load_level") == load
                             and r.get("meta", {}).get("fault") == fault_filter]
                    if group:
                        v = [r["metrics"]["goodput"] for r in group]
                        goodputs.append(np.mean(v))
                        errs.append(np.std(v))
                    else:
                        goodputs.append(0)
                        errs.append(0)

                colors = [_get_style(bl)["color"] for bl in ablation_baselines]
                ax.bar(x, goodputs, yerr=errs, color=colors, capsize=3,
                       alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [bl.replace("-", "\n") for bl in ablation_baselines],
                    fontsize=7, rotation=30, ha="right")
                ax.set_ylabel("Goodput (tok/s)")

        fig.tight_layout()
        fname = f"ablation_{fault_filter}.pdf"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fname}")


def plot_checkpoint_tradeoff(runs: list[dict], output_dir: str) -> None:
    """Figure: Checkpoint overhead vs recovery benefit (E4)."""
    if not HAS_MPL:
        return

    ckpt_baselines = ["No-FT", "Periodic-Low", "Periodic-High", "Our-System",
                      "Fixed-Low-CKPT", "Fixed-High-CKPT"]  # v1 compat

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Normal case: goodput
    ax = axes[0]
    ax.set_title("Normal Case (no fault)")
    group_data = {}
    for bl in ckpt_baselines:
        group = [r for r in runs
                 if r.get("meta", {}).get("baseline") == bl
                 and r.get("meta", {}).get("fault") == "none"]
        if group:
            v = [r["metrics"]["goodput"] for r in group]
            group_data[bl] = (np.mean(v), np.std(v))

    if group_data:
        x = np.arange(len(group_data))
        labels = list(group_data.keys())
        vals = [group_data[bl][0] for bl in labels]
        errs = [group_data[bl][1] for bl in labels]
        colors = [_get_style(bl)["color"] for bl in labels]
        ax.bar(x, vals, yerr=errs, color=colors, capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Goodput (tok/s)")

    # Failure case: successful direct-hit failover gap
    ax = axes[1]
    ax.set_title("Failure Case (successful direct-hit gap p95)")
    group_data = {}
    succ_labels = {}
    for bl in ckpt_baselines:
        group = [r for r in runs
                 if r.get("meta", {}).get("baseline") == bl
                 and r.get("meta", {}).get("fault", "none") != "none"]
        if group:
            v = [_get_recomputed_failover_gap_p95(r) for r in group]
            group_data[bl] = (np.mean(v), np.std(v))
            succ = 0
            total = 0
            for r in group:
                succ_r, total_r = _get_direct_hit_success_counts(r)
                succ += succ_r
                total += total_r
            succ_labels[bl] = f"succ {succ}/{total}"

    if group_data:
        x = np.arange(len(group_data))
        labels = list(group_data.keys())
        vals = [group_data[bl][0] for bl in labels]
        errs = [group_data[bl][1] for bl in labels]
        colors = [_get_style(bl)["color"] for bl in labels]
        bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        for bar, bl in zip(bars, labels):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                succ_labels.get(bl, "succ 0/0"),
                ha="center", va="bottom", fontsize=7, color="#444444",
            )
        ax.set_ylabel("Failover Gap p95 (ms)")

    fig.tight_layout()
    fname = "checkpoint_tradeoff.pdf"
    fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {fname}")


# ============================================================
# New v2 plots
# ============================================================

def plot_recovery_gap_cdf(runs: list[dict], output_dir: str) -> None:
    """Figure 6: CDF of per-request recovery gap (direct-hit requests)."""
    if not HAS_MPL:
        return

    fault_runs = [r for r in runs if r.get("meta", {}).get("fault", "none") != "none"]
    if not fault_runs:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    baselines_present = sorted(set(r.get("meta", {}).get("baseline", "") for r in fault_runs))

    for bl in baselines_present:
        bl_runs = [r for r in fault_runs if r.get("meta", {}).get("baseline") == bl]
        gaps = []
        for r in bl_runs:
            dh = _get_direct_hit_requests(r)
            for req in dh:
                g = float(req.get("max_gap_ms", 0))
                if g > 0:
                    gaps.append(g)
        if not gaps:
            continue
        gaps_sorted = np.sort(gaps)
        cdf = np.arange(1, len(gaps_sorted) + 1) / len(gaps_sorted)
        style = _get_style(bl)
        ax.plot(gaps_sorted, cdf, label=bl, color=style["color"], ls=style["ls"], lw=2)

    ax.set_xscale("log")
    ax.set_xlabel("Failover Gap (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Recovery Gap Distribution (direct-hit requests)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "recovery_gap_cdf.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Plot: recovery_gap_cdf.pdf")


def plot_ablation_heatmap(runs: list[dict], output_dir: str) -> None:
    """Figure 8: Heatmap of SLO violation across (baseline × condition) combos."""
    if not HAS_MPL:
        return

    grouped = group_runs(runs, ["baseline", "workload", "load_level", "fault"])

    ablation_baselines = ["Periodic-Low", "Periodic-High", "Benders-Only",
                          "Adaptive-Only", "Our-System",
                          "Fixed-Low-CKPT", "Fixed-High-CKPT",
                          "Robust-Routing-Only", "Checkpoint-Only"]
    baselines = [bl for bl in ablation_baselines
                 if any(r.get("meta", {}).get("baseline") == bl for r in runs)]
    if not baselines:
        return

    conditions = sorted(set(
        (r.get("meta", {}).get("workload", ""), r.get("meta", {}).get("load_level", ""),
         r.get("meta", {}).get("fault", ""))
        for r in runs
    ))

    matrix = np.full((len(baselines), len(conditions)), np.nan)
    for i, bl in enumerate(baselines):
        for j, (wl, load, fault) in enumerate(conditions):
            key = (bl, wl, load, fault)
            group = grouped.get(key, [])
            if group:
                vals = [r["metrics"]["slo_violation_rate"] * 100 for r in group]
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 0.8), max(3, len(baselines) * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([f"{w}\n{l}\n{f}" for w, l, f in conditions], fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baselines, fontsize=8)

    for i in range(len(baselines)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=6,
                        color="white" if val > 50 else "black")

    fig.colorbar(im, ax=ax, label="SLO Violation Rate (%)")
    ax.set_title("SLO Violation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Plot: ablation_heatmap.pdf")


def plot_checkpoint_tradeoff_scatter(runs: list[dict], output_dir: str) -> None:
    """Figure 9: Scatter plot — checkpoint overhead (x) vs recovery benefit (y)."""
    if not HAS_MPL:
        return

    grouped = group_runs(runs, ["baseline", "workload", "load_level", "fault"])

    # Find No-FT goodput as anchor
    noft_goodput = {}  # (workload, load, fault) -> mean goodput
    for key, group in grouped.items():
        bl, wl, load, fault = key
        if bl == "No-FT" and group:
            vals = [r["metrics"]["goodput"] for r in group]
            noft_goodput[(wl, load, fault)] = np.mean(vals)

    if not noft_goodput:
        return

    # For each baseline, compute (overhead_pct, recovery_benefit_pct)
    fig, ax = plt.subplots(figsize=(7, 5))

    baselines_seen = set()
    for key, group in grouped.items():
        bl, wl, load, fault = key
        if bl == "No-FT" or not group:
            continue
        baselines_seen.add(bl)

    for bl in baselines_seen:
        overheads = []
        benefits = []
        for key, group in grouped.items():
            b, wl, load, fault = key
            if b != bl or not group:
                continue
            noft_ref = noft_goodput.get((wl, load, fault))
            if noft_ref is None or noft_ref <= 0:
                continue
            mean_gp = np.mean([r["metrics"]["goodput"] for r in group])
            if fault == "none":
                overhead = (1 - mean_gp / noft_ref) * 100
                overheads.append(overhead)
            else:
                noft_fault = noft_goodput.get((wl, load, fault), noft_ref)
                benefit = ((mean_gp - noft_fault) / max(noft_fault, 1)) * 100
                benefits.append(benefit)

        if overheads and benefits:
            style = _get_style(bl)
            ax.scatter(np.mean(overheads), np.mean(benefits),
                       color=style["color"], marker=style["marker"], s=120, zorder=5)
            ax.annotate(bl, (np.mean(overheads), np.mean(benefits)),
                        fontsize=7, ha="left", va="bottom")

    # Anchor: No-FT at (0, 0)
    ax.scatter(0, 0, color="#888888", marker="x", s=120, zorder=5)
    ax.annotate("No-FT", (0, 0), fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Normal-Operation Goodput Degradation vs No-FT (%)")
    ax.set_ylabel("Fault Goodput Improvement vs No-FT (%)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Checkpoint Tradeoff: Overhead vs Recovery Benefit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "checkpoint_tradeoff_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Plot: checkpoint_tradeoff_scatter.pdf")


def plot_goodput_timeline(runs: list[dict], output_dir: str) -> None:
    """Figure 10: Goodput over time around fault injection."""
    if not HAS_MPL:
        return

    fault_runs = [r for r in runs if r.get("meta", {}).get("fault", "none") != "none"]
    if not fault_runs:
        return

    # Pick first seed, first workload/load combo
    target_baselines = ["No-FT", "Periodic-High", "Our-System"]

    fig, ax = plt.subplots(figsize=(10, 4))
    window_sec = 5.0

    for bl in target_baselines:
        bl_runs = [r for r in fault_runs if r.get("meta", {}).get("baseline") == bl]
        if not bl_runs:
            continue
        run = bl_runs[0]  # first matching run

        reqs = run.get("requests", [])
        if not reqs:
            continue

        # Compute rolling goodput from per-request data
        completed = [
            r for r in reqs
            if _parse_bool_field(r.get("success", False)) and r.get("end_time")
        ]
        if not completed:
            continue

        end_times = [float(r["end_time"]) for r in completed]
        tokens = [int(r.get("output_tokens", 0)) for r in completed]
        min_t = min(end_times)
        max_t = max(end_times)

        bins = int((max_t - min_t) / window_sec) + 1
        goodput_bins = [0.0] * bins
        for et, tok in zip(end_times, tokens):
            idx = int((et - min_t) / window_sec)
            if 0 <= idx < bins:
                goodput_bins[idx] += tok / window_sec

        times = [i * window_sec for i in range(bins)]
        style = _get_style(bl)
        ax.plot(times, goodput_bins, label=bl, color=style["color"], ls=style["ls"], lw=1.5)

    # Mark fault time
    fault_time = fault_runs[0].get("meta", {}).get("fault_time_sec")
    if fault_time:
        ax.axvline(float(fault_time), color="red", ls="--", alpha=0.7, label="Fault injected")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Goodput (tok/s, {window_sec:.0f}s window)")
    ax.set_title("Goodput Timeline Around Fault")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "goodput_timeline.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Plot: goodput_timeline.pdf")


def plot_slo_sensitivity(runs: list[dict], output_dir: str) -> None:
    """Figure 13: SLO sensitivity — goodput and violation at different SLO levels."""
    if not HAS_MPL:
        return

    # Group by baseline and slo_scale (from output_dir path or metadata)
    # For E6 runs, the output_dir contains the scale level
    grouped: dict[tuple, list[dict]] = {}
    for r in runs:
        bl = r.get("meta", {}).get("baseline", "")
        # Try to extract slo_scale from directory path
        d = r.get("dir", "")
        for scale in ["Tight", "Moderate", "Loose"]:
            if f"/{scale}/" in d:
                key = (bl, scale)
                grouped.setdefault(key, []).append(r)
                break

    if not grouped:
        return

    scale_order = ["Tight", "Moderate", "Loose"]
    baselines = sorted(set(k[0] for k in grouped.keys()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(scale_order))
    width = 0.8 / max(len(baselines), 1)

    for i, bl in enumerate(baselines):
        gp_vals = []
        viol_vals = []
        for scale in scale_order:
            group = grouped.get((bl, scale), [])
            if group:
                gp_vals.append(np.mean([r["metrics"]["goodput"] for r in group]))
                viol_vals.append(np.mean([r["metrics"]["slo_violation_rate"] * 100 for r in group]))
            else:
                gp_vals.append(0)
                viol_vals.append(0)
        style = _get_style(bl)
        ax1.bar(x + i * width, gp_vals, width, label=bl, color=style["color"], alpha=0.8)
        ax2.bar(x + i * width, viol_vals, width, label=bl, color=style["color"], alpha=0.8)

    ax1.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax1.set_xticklabels(scale_order)
    ax1.set_ylabel("Goodput (tok/s)")
    ax1.set_title("Goodput by SLO Tightness")
    ax1.legend(fontsize=7)

    ax2.set_xticks(x + width * (len(baselines) - 1) / 2)
    ax2.set_xticklabels(scale_order)
    ax2.set_ylabel("SLO Violation Rate (%)")
    ax2.set_title("Violation by SLO Tightness")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "slo_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Plot: slo_sensitivity.pdf")


def plot_gap_slo_sensitivity(runs: list[dict], output_dir: str) -> None:
    """Figure 14: Gap SLO sensitivity — fraction meeting gap SLO at varying thresholds."""
    if not HAS_MPL:
        return

    fault_runs = [r for r in runs if r.get("meta", {}).get("fault", "none") != "none"]
    if not fault_runs:
        return

    multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    fig, ax = plt.subplots(figsize=(7, 4))

    baselines_present = sorted(set(r.get("meta", {}).get("baseline", "") for r in fault_runs))

    for bl in baselines_present:
        bl_runs = [r for r in fault_runs if r.get("meta", {}).get("baseline") == bl]
        gaps = []
        for r in bl_runs:
            dh = _get_direct_hit_requests(r)
            for req in dh:
                g = float(req.get("max_gap_ms", 0))
                if g > 0:
                    gaps.append(g)
        if not gaps:
            continue

        # Use median as "base"
        gap_base = np.median(gaps) if gaps else 1000.0

        fractions = []
        for mult in multipliers:
            threshold = gap_base * mult
            fraction = sum(1 for g in gaps if g <= threshold) / len(gaps)
            fractions.append(fraction)

        style = _get_style(bl)
        ax.plot(multipliers, fractions, label=bl, color=style["color"],
                marker=style["marker"], ls=style["ls"], lw=2, markersize=6)

    ax.set_xlabel("Gap SLO Multiplier (× median gap)")
    ax.set_ylabel("Fraction Meeting Gap SLO")
    ax.set_title("Gap SLO Sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gap_slo_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  Plot: gap_slo_sensitivity.pdf")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze FT experiment results (v2)")
    parser.add_argument("results_dir", nargs="+",
                        help="Path(s) to results directories (supports multiple for cross-model)")
    parser.add_argument("--output", default="figures_v2/",
                        help="Output directory for figures and tables")
    parser.add_argument("--all", action="store_true",
                        help="Generate all figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    runs = []
    for rd in args.results_dir:
        print(f"Loading results from: {rd}")
        loaded = load_all_runs(rd)
        print(f"  Found {len(loaded)} completed runs")
        runs.extend(loaded)
    print(f"  Total: {len(runs)} runs")

    if not runs:
        print("No results found!")
        return

    # Tables.
    print("\n--- Generating tables ---")
    make_summary_table(runs, os.path.join(args.output, "summary_all.csv"))
    make_mean_table(runs, os.path.join(args.output, "summary_mean.csv"))

    # Figures — only generate plots relevant to the experiment.
    EXPERIMENT_PLOTS: dict[str, list[str]] = {
        "E0_Smoke":               ["goodput_by_load", "slo_violation"],
        "E1_Smoke":               ["goodput_by_load", "slo_violation", "failover_gap"],
        "E1a_Main":               ["goodput_by_load", "slo_violation", "failover_gap"],
        "E1b_Main":               ["goodput_by_load"],
        "E1_Main":                ["goodput_by_load", "slo_violation", "failover_gap"],
        "E2_Recovery":            ["recovery_breakdown", "recovery_gap_cdf"],
        "E3_Ablation":            ["ablation", "ablation_heatmap"],
        "E4_Checkpoint_Tradeoff": ["checkpoint_tradeoff", "checkpoint_tradeoff_scatter",
                                   "goodput_timeline"],
        "E5_Controller":          ["controller_overhead"],
        "E6_SLO_Sensitivity":     ["slo_sensitivity", "gap_slo_sensitivity"],
    }

    plot_dispatch = {
        "goodput_by_load":              plot_goodput_by_load,
        "slo_violation":                plot_slo_violation,
        "failover_gap":                 plot_failover_gap,
        "recovery_breakdown":           plot_recovery_breakdown,
        "controller_overhead":          plot_controller_overhead,
        "ablation":                     plot_ablation,
        "checkpoint_tradeoff":          plot_checkpoint_tradeoff,
        "recovery_gap_cdf":             plot_recovery_gap_cdf,
        "ablation_heatmap":             plot_ablation_heatmap,
        "checkpoint_tradeoff_scatter":  plot_checkpoint_tradeoff_scatter,
        "goodput_timeline":             plot_goodput_timeline,
        "slo_sensitivity":              plot_slo_sensitivity,
        "gap_slo_sensitivity":          plot_gap_slo_sensitivity,
    }

    exp_name = os.path.basename(os.path.normpath(args.results_dir[0]))

    if HAS_MPL:
        print("\n--- Generating figures ---")
        if args.all:
            selected = list(plot_dispatch.keys())
            print("  --all: generating all plots")
        elif exp_name in EXPERIMENT_PLOTS:
            selected = EXPERIMENT_PLOTS[exp_name]
            print(f"  Experiment {exp_name} → {selected}")
        else:
            selected = list(plot_dispatch.keys())
            print(f"  Unknown experiment '{exp_name}', generating all plots")

        for name in selected:
            plot_dispatch[name](runs, args.output)

    print(f"\nDone. Output in: {args.output}")


if __name__ == "__main__":
    main()
