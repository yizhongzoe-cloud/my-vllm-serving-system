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
    "No-FT":              {"color": "#888888", "marker": "x", "ls": "--"},
    "Fixed-Low-CKPT":     {"color": "#E69F00", "marker": "s", "ls": "-"},
    "Fixed-High-CKPT":    {"color": "#D55E00", "marker": "^", "ls": "-"},
    "Robust-Routing-Only":{"color": "#009E73", "marker": "D", "ls": "-"},
    "Checkpoint-Only":    {"color": "#CC79A7", "marker": "p", "ls": "-"},
    "Our-System":         {"color": "#0072B2", "marker": "o", "ls": "-"},
}


def _get_style(baseline: str) -> dict:
    return BASELINE_STYLE.get(baseline, {"color": "black", "marker": ".", "ls": "-"})


def plot_goodput_by_load(runs: list[dict], output_dir: str) -> None:
    """Figure: Goodput vs load level for each baseline (one subplot per workload)."""
    if not HAS_MPL:
        return

    grouped = group_runs(runs, ["baseline", "workload", "load_level", "fault"])
    workloads = sorted(set(r.get("meta", {}).get("workload", "") for r in runs))
    load_order = ["Low", "Medium", "High"]

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
                else:
                    seed_p95s.append(0.0)
                succ, total = _get_direct_hit_success_counts(r)
                succ_count += succ
                total_direct_hit += total

            x_pos.append(i)
            labels.append(bl)
            vals.append(np.mean(seed_p95s))
            errs.append(np.std(seed_p95s))
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
    """Figure: Ablation — Our-System vs routing-only vs checkpoint-only."""
    if not HAS_MPL:
        return

    ablation_baselines = [
        "Our-System", "Robust-Routing-Only", "Checkpoint-Only",
        "Fixed-Low-CKPT", "Fixed-High-CKPT",
    ]

    for fault_filter in ["none", "F2_Mid"]:
        load_levels = ["Medium", "High"]
        fig, axes = plt.subplots(1, len(load_levels), figsize=(5*len(load_levels), 4))

        for li, load in enumerate(load_levels):
            ax = axes[li]
            ax.set_title(f"Load={load}, fault={fault_filter}", fontsize=10)
            x = np.arange(len(ablation_baselines))

            goodputs = []
            errs = []
            for bl in ablation_baselines:
                group = [r for r in runs
                         if r.get("meta", {}).get("baseline") == bl
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
            ax.bar(x, goodputs, yerr=errs, color=colors, capsize=3, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([bl.replace("-", "\n") for bl in ablation_baselines],
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

    ckpt_baselines = ["No-FT", "Fixed-Low-CKPT", "Fixed-High-CKPT", "Our-System"]

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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze FT experiment results")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--output", default="figures/",
                        help="Output directory for figures and tables")
    parser.add_argument("--all", action="store_true",
                        help="Generate all figures")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    runs = load_all_runs(args.results_dir)
    print(f"  Found {len(runs)} completed runs")

    if not runs:
        print("No results found!")
        return

    # Tables.
    print("\n--- Generating tables ---")
    make_summary_table(runs, os.path.join(args.output, "summary_all.csv"))
    make_mean_table(runs, os.path.join(args.output, "summary_mean.csv"))

    # Figures — only generate plots relevant to the experiment.
    EXPERIMENT_PLOTS: dict[str, list[str]] = {
        "E1_Smoke":               ["goodput_by_load", "slo_violation", "failover_gap"],
        "E1_Main":                ["goodput_by_load", "slo_violation", "failover_gap"],
        "E2_Recovery":            ["recovery_breakdown"],
        "E3_Ablation":            ["ablation"],
        "E4_Checkpoint_Tradeoff": ["checkpoint_tradeoff"],
        "E5_Controller":          ["controller_overhead"],
    }

    plot_dispatch = {
        "goodput_by_load":     plot_goodput_by_load,
        "slo_violation":       plot_slo_violation,
        "failover_gap":        plot_failover_gap,
        "recovery_breakdown":  plot_recovery_breakdown,
        "controller_overhead": plot_controller_overhead,
        "ablation":            plot_ablation,
        "checkpoint_tradeoff": plot_checkpoint_tradeoff,
    }

    exp_name = os.path.basename(os.path.normpath(args.results_dir))

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
