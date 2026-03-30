#!/usr/bin/env python3

from experiments.run import LogParser, RequestResult, _enrich_results


def test_log_parser_parses_ft_client_failover_events(tmp_path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "\n".join([
            "03-25 15:27:29 WARNING x FT Client: engine 0 declared FAILED. "
            "Starting failover for displaced requests.",
            "03-25 15:27:29 INFO x Request chatcmpl-abc123-sfx: re-routed 0→1, "
            "restored=16 tokens, replay=4 tokens, est_gap=0.0ms, slo_met=True",
            "03-25 15:27:29 INFO x FT Client: failover complete. "
            "Re-routed 1/1 requests.",
        ])
    )

    epochs, recoveries, ckpt_classes = LogParser.parse(str(log_file))

    assert epochs == []
    assert ckpt_classes == {}
    assert [r["type"] for r in recoveries] == [
        "failover_start",
        "reroute",
        "failover_complete",
    ]
    assert recoveries[0]["replica_id"] == 0
    assert recoveries[1]["initial_gpu"] == 0
    assert recoveries[1]["resumed_gpu"] == 1
    assert recoveries[1]["tokens_restored"] == 16
    assert recoveries[1]["replay_tokens"] == 4
    assert recoveries[2]["recovered"] == 1
    assert recoveries[2]["total"] == 1


def test_log_parser_parses_fault_event_monitor_and_declared(tmp_path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "\n".join([
            "03-25 15:27:29 WARNING x FAULT_EVENT monitor_observed "
            "engine=0 wall_time=1711234567.100000 source=engine_core_dead",
            "03-25 15:27:29 WARNING x FAULT_EVENT failure_declared "
            "replica=0 wall_time=1711234567.150000",
            "03-25 15:27:29 WARNING x FAULT_EVENT failover_start "
            "engine=0 wall_time=1711234567.200000",
        ])
    )

    epochs, recoveries, ckpt_classes = LogParser.parse(str(log_file))

    assert epochs == []
    assert ckpt_classes == {}
    assert [r["type"] for r in recoveries] == [
        "monitor_observed",
        "failure_declared",
        "failover_start",
    ]
    assert recoveries[0]["replica_id"] == 0
    assert recoveries[0]["source"] == "engine_core_dead"
    assert recoveries[0]["wall_time"] == 1711234567.1
    assert recoveries[1]["wall_time"] == 1711234567.15
    assert recoveries[2]["wall_time"] == 1711234567.2


def test_enrich_results_marks_ft_client_reroutes():
    result = RequestResult(
        request_id="req-00001",
        arrival_time=0.0,
        prompt_len=32,
        expected_output_len=64,
        server_request_id="chatcmpl-abc123",
        max_gap_ms=220.0,
    )

    recoveries = [
        {
            "type": "failover_start",
            "replica_id": 0,
            "timestamp": 10.05,
            "source": "ft_client",
        },
        {
            "type": "reroute",
            "request_id": "chatcmpl-abc123-sfx",
            "initial_gpu": 0,
            "resumed_gpu": 1,
            "tokens_restored": 16,
            "replay_tokens": 4,
            "est_gap_ms": 0.0,
            "timestamp": 10.06,
        },
        {
            "type": "failover_complete",
            "replica_id": -1,
            "recovered": 1,
            "total": 1,
            "dropped": 0,
            "timestamp": 10.20,
            "source": "ft_client",
        },
    ]

    _enrich_results(
        [result],
        recoveries,
        fault_injection_time=10.0,
        ckpt_classes=None,
    )

    assert result.affected_by_failure is True
    assert result.was_rerouted is True
    assert result.initial_gpu == 0
    assert result.resumed_gpu == 1
    assert result.replay_tokens == 4
    assert result.restore_time_ms is not None and result.restore_time_ms > 0
    assert result.replay_time_ms is not None and result.replay_time_ms >= 0


def test_enrich_results_ignores_coarse_failover_timestamps():
    result = RequestResult(
        request_id="req-00002",
        arrival_time=0.0,
        prompt_len=32,
        expected_output_len=64,
        server_request_id="chatcmpl-coarse",
        max_gap_ms=220.0,
    )

    recoveries = [
        {
            "type": "failover_start",
            "replica_id": 0,
            "timestamp": 11.0,  # whole-second log timestamp
            "source": "ft_client",
        },
        {
            "type": "reroute",
            "request_id": "chatcmpl-coarse-sfx",
            "initial_gpu": 0,
            "resumed_gpu": 1,
            "tokens_restored": 0,
            "replay_tokens": 4,
            "est_gap_ms": 0.0,
            "timestamp": 11.0,
        },
        {
            "type": "failover_complete",
            "replica_id": -1,
            "recovered": 1,
            "total": 1,
            "dropped": 0,
            "timestamp": 11.0,
            "source": "ft_client",
        },
    ]

    _enrich_results(
        [result],
        recoveries,
        fault_injection_time=10.92353,
        ckpt_classes=None,
    )

    assert result.affected_by_failure is True
    assert result.restore_time_ms == 0.0
    assert result.replay_time_ms == 220.0


def test_enrich_results_splits_coarse_gap_using_restore_and_replay_tokens():
    result = RequestResult(
        request_id="req-00003",
        arrival_time=0.0,
        prompt_len=32,
        expected_output_len=64,
        server_request_id="chatcmpl-coarse-split",
        max_gap_ms=200.0,
    )

    recoveries = [
        {
            "type": "failover_start",
            "replica_id": 0,
            "timestamp": 11.0,
            "source": "ft_client",
        },
        {
            "type": "reroute",
            "request_id": "chatcmpl-coarse-split-sfx",
            "initial_gpu": 0,
            "resumed_gpu": 1,
            "tokens_restored": 150,
            "replay_tokens": 50,
            "est_gap_ms": 0.0,
            "timestamp": 11.0,
        },
        {
            "type": "failover_complete",
            "replica_id": -1,
            "recovered": 1,
            "total": 1,
            "dropped": 0,
            "timestamp": 11.0,
            "source": "ft_client",
        },
    ]

    _enrich_results(
        [result],
        recoveries,
        fault_injection_time=10.92353,
        ckpt_classes=None,
    )

    assert result.affected_by_failure is True
    assert result.restore_time_ms == 150.0
    assert result.replay_time_ms == 50.0


def test_enrich_results_does_not_zero_replay_when_detection_exceeds_gap():
    result = RequestResult(
        request_id="req-00003b",
        arrival_time=0.0,
        prompt_len=32,
        expected_output_len=64,
        server_request_id="chatcmpl-short-gap",
        max_gap_ms=140.0,
    )

    recoveries = [
        {
            "type": "failure_declared",
            "replica_id": 0,
            "timestamp": 10.150,
            "wall_time": 10.150,
        },
        {
            "type": "failover_start",
            "replica_id": 0,
            "timestamp": 10.151,
            "wall_time": 10.151,
            "source": "fault_event",
        },
        {
            "type": "reroute",
            "request_id": "chatcmpl-short-gap-sfx",
            "initial_gpu": 0,
            "resumed_gpu": 1,
            "tokens_restored": 0,
            "replay_tokens": 25,
            "est_gap_ms": 0.0,
            "timestamp": 10.152,
        },
    ]

    _enrich_results(
        [result],
        recoveries,
        fault_injection_time=10.0,
        ckpt_classes=None,
    )

    assert result.affected_by_failure is True
    assert result.restore_time_ms == 0.0
    assert result.replay_time_ms == 140.0


def test_enrich_results_ignores_zero_delta_coarse_kv_restore_timestamps():
    result = RequestResult(
        request_id="req-00004",
        arrival_time=0.0,
        prompt_len=32,
        expected_output_len=64,
        server_request_id="chatcmpl-coarse-kv",
        max_gap_ms=240.0,
    )

    recoveries = [
        {
            "type": "failover_start",
            "replica_id": 0,
            "timestamp": 11.0,
            "source": "ft_client",
        },
        {
            "type": "reroute",
            "request_id": "chatcmpl-coarse-kv-sfx",
            "initial_gpu": 0,
            "resumed_gpu": 1,
            "tokens_restored": 180,
            "replay_tokens": 60,
            "est_gap_ms": 0.0,
            "timestamp": 11.0,
        },
        {
            "type": "kv_restore",
            "request_id": "chatcmpl-coarse-kv-sfx",
            "tokens_restored": 180,
            "blocks_restored": 12,
            "timestamp": 11.0,
        },
        {
            "type": "failover_complete",
            "replica_id": -1,
            "recovered": 1,
            "total": 1,
            "dropped": 0,
            "timestamp": 11.0,
            "source": "ft_client",
        },
    ]

    _enrich_results(
        [result],
        recoveries,
        fault_injection_time=10.92353,
        ckpt_classes=None,
    )

    assert result.restore_time_ms == 180.0
    assert result.replay_time_ms == 60.0


def test_enrich_results_matches_client_request_id_without_stream_id():
    result = RequestResult(
        request_id="req-00030",
        arrival_time=0.0,
        prompt_len=32,
        expected_output_len=64,
        max_gap_ms=180.0,
    )

    recoveries = [
        {
            "type": "failover_start",
            "replica_id": 0,
            "timestamp": 12.0,
            "source": "ft_client",
        },
        {
            "type": "reroute",
            "request_id": "chatcmpl-req-00030-abc123",
            "initial_gpu": 0,
            "resumed_gpu": 1,
            "tokens_restored": 64,
            "replay_tokens": 0,
            "est_gap_ms": 0.0,
            "timestamp": 12.0,
        },
        {
            "type": "failover_complete",
            "replica_id": -1,
            "recovered": 1,
            "total": 1,
            "dropped": 0,
            "timestamp": 12.0,
            "source": "ft_client",
        },
    ]

    _enrich_results(
        [result],
        recoveries,
        fault_injection_time=11.5,
        ckpt_classes={"chatcmpl-req-00030": 2},
    )

    assert result.was_rerouted is True
    assert result.affected_by_failure is True
    assert result.initial_gpu == 0
    assert result.resumed_gpu == 1
    assert result.replay_tokens == 0
    assert result.checkpoint_class == 2
