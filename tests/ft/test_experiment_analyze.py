#!/usr/bin/env python3

import csv

import pytest

from experiments.analyze import (
    _get_recovery_detection_metrics,
    _get_recomputed_failover_gap_p95,
    _recompute_failover_gaps,
    make_summary_table,
)


def test_recompute_failover_gap_ignores_failed_direct_hit_request():
    run = {
        "dir": "results/fake_run",
        "meta": {
            "fault": "F2_Mid",
            "fault_injection_time": 10.0,
            "failure_gap_slo_ms": 3000.0,
        },
        "requests": [
            {
                "request_id": "req-1",
                "send_time": "8.0",
                "end_time": "0.0",
                "success": "False",
                "max_gap_ms": "5.0",
                "ownership_known": "True",
                "affected_by_failure": "True",
            },
        ],
    }

    assert _recompute_failover_gaps(run) == []
    assert _get_recomputed_failover_gap_p95(run) == 0.0


def test_recompute_failover_gap_excludes_request_completed_before_fault():
    run = {
        "dir": "results/fake_run",
        "meta": {
            "fault": "F2_Mid",
            "fault_injection_time": 10.0,
            "failure_gap_slo_ms": 3000.0,
        },
        "requests": [
            {
                "request_id": "req-pre",
                "send_time": "8.0",
                "end_time": "9.5",
                "success": "True",
                "max_gap_ms": "250.0",
                "ownership_known": "True",
                "affected_by_failure": "False",
            },
            {
                "request_id": "req-hit",
                "send_time": "9.0",
                "end_time": "11.0",
                "success": "True",
                "max_gap_ms": "180.0",
                "ownership_known": "True",
                "affected_by_failure": "True",
            },
        ],
    }

    assert _recompute_failover_gaps(run) == [180.0]


def test_make_summary_table_uses_recomputed_failover_gap(tmp_path):
    run = {
        "dir": "results/fake_run",
        "meta": {
            "baseline": "No-FT",
            "workload": "W1",
            "load_level": "Medium",
            "fault": "F2_Mid",
            "seed": 42,
            "fault_injection_time": 10.0,
            "failure_gap_slo_ms": 3000.0,
        },
        "metrics": {
            "goodput": 1.0,
            "ttft_p50_ms": 2.0,
            "ttft_p95_ms": 3.0,
            "ttft_p99_ms": 4.0,
            "tpot_p50_ms": 5.0,
            "tpot_p95_ms": 6.0,
            "tpot_p99_ms": 7.0,
            "slo_violation_rate": 0.1,
            "completion_rate": 0.2,
            "time_to_stable_sec": 1.0,
            "total_requests": 10,
            "completed": 2,
        },
        "requests": [
            {
                "request_id": "req-1",
                "send_time": "8.0",
                "end_time": "11.0",
                "success": "True",
                "max_gap_ms": "180.0",
                "ownership_known": "True",
                "affected_by_failure": "True",
            },
        ],
    }

    output_path = tmp_path / "summary_all.csv"
    make_summary_table([run], str(output_path))

    rows = list(csv.DictReader(output_path.open()))
    assert len(rows) == 1
    assert float(rows[0]["failover_gap_p95"]) == 180.0


def test_recompute_failover_gap_requires_new_results_schema():
    run = {
        "dir": "results/old_run",
        "meta": {
            "fault": "F2_Mid",
            "fault_injection_time": 10.0,
        },
        "requests": [
            {
                "request_id": "req-1",
                "send_time": "8.0",
                "success": "False",
                "max_gap_ms": "0.0",
            },
        ],
    }

    with pytest.raises(ValueError, match="required columns"):
        _recompute_failover_gaps(run)


def test_recovery_detection_metrics_use_precise_structured_events():
    run = {
        "dir": "results/fake_run",
        "meta": {
            "fault_injection_time": 10.0,
        },
        "recoveries": [
            {"type": "monitor_observed", "wall_time": 10.012},
            {"type": "failure_declared", "wall_time": 10.045},
            {"type": "failover_start", "wall_time": 10.081},
        ],
    }

    metrics = _get_recovery_detection_metrics(run)
    assert metrics["monitor_detect_ms"] == pytest.approx(12.0)
    assert metrics["declare_failed_ms"] == pytest.approx(45.0)
    assert metrics["failover_start_ms"] == pytest.approx(81.0)


def test_recovery_detection_metrics_use_earliest_fault_event_for_preinjection_crash():
    run = {
        "dir": "results/fake_run",
        "meta": {
            "fault_injection_time": 20.0,
        },
        "recoveries": [
            {"type": "monitor_observed", "wall_time": 10.012},
            {"type": "failure_declared", "wall_time": 10.045},
            {"type": "failover_start", "wall_time": 10.081},
        ],
    }

    metrics = _get_recovery_detection_metrics(run)
    assert metrics["monitor_detect_ms"] == pytest.approx(0.0)
    assert metrics["declare_failed_ms"] == pytest.approx(33.0)
    assert metrics["failover_start_ms"] == pytest.approx(69.0)


def test_make_summary_table_includes_detection_columns(tmp_path):
    run = {
        "dir": "results/fake_run",
        "meta": {
            "baseline": "Our-System",
            "workload": "W1",
            "load_level": "Medium",
            "fault": "F2_Mid",
            "seed": 42,
            "fault_injection_time": 10.0,
            "failure_gap_slo_ms": 3000.0,
        },
        "metrics": {
            "goodput": 1.0,
            "ttft_p50_ms": 2.0,
            "ttft_p95_ms": 3.0,
            "ttft_p99_ms": 4.0,
            "tpot_p50_ms": 5.0,
            "tpot_p95_ms": 6.0,
            "tpot_p99_ms": 7.0,
            "slo_violation_rate": 0.1,
            "completion_rate": 0.2,
            "time_to_stable_sec": 1.0,
            "total_requests": 10,
            "completed": 2,
        },
        "requests": [
            {
                "request_id": "req-1",
                "send_time": "8.0",
                "end_time": "11.0",
                "success": "True",
                "max_gap_ms": "180.0",
                "ownership_known": "True",
                "affected_by_failure": "True",
            },
        ],
        "recoveries": [
            {"type": "monitor_observed", "wall_time": 10.012},
            {"type": "failure_declared", "wall_time": 10.045},
            {"type": "failover_start", "wall_time": 10.081},
        ],
    }

    output_path = tmp_path / "summary_all.csv"
    make_summary_table([run], str(output_path))

    rows = list(csv.DictReader(output_path.open()))
    assert len(rows) == 1
    assert float(rows[0]["monitor_detect_ms"]) == pytest.approx(12.0)
    assert float(rows[0]["declare_failed_ms"]) == pytest.approx(45.0)
    assert float(rows[0]["failover_start_ms"]) == pytest.approx(81.0)
