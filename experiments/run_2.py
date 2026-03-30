#!/usr/bin/env python3
"""Experiment runner for FT multi-GPU serving experiments.

This module restores the missing experiment runner used by the local
experiment suite. It can be executed directly as a script and also exposes
the log parsing / result enrichment helpers relied on by tests.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import aiohttp
import numpy as np
import yaml

from experiments.workloads import RequestSpec, generate_trace

SERVER_LOG_NAME = "server.log"

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
LEADING_TIMESTAMP_RE = re.compile(
    r"(?<!\d)(?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?!\d)"
)
FAULT_EVENT_RE = re.compile(r"FAULT_EVENT (?P<event>\w+) (?P<body>.*)")
KV_PAIR_RE = re.compile(r"(\w+)=([^\s]+)")
FAILOVER_START_RE = re.compile(
    r"(?:Centralized )?FT Client: engine (?P<replica>\d+) declared FAILED\. "
    r"Starting failover for displaced requests\."
)
REROUTE_RE = re.compile(
    r"Request (?P<request_id>\S+): re-routed (?P<initial>\d+)→(?P<resumed>\d+), "
    r"restored=(?P<restored>\d+) tokens, replay=(?P<replay>\d+) tokens, "
    r"est_gap=(?P<gap>[0-9.]+)ms, slo_met=(?P<slo_met>True|False)"
    r"(?:, wall_time=(?P<wall_time>[0-9.]+))?"
)
FAILOVER_COMPLETE_RE = re.compile(
    r"(?:Centralized )?FT Client: failover complete\. Re-routed "
    r"(?P<recovered>\d+)/(?P<total>\d+) requests\."
)
BENDERS_CONVERGED_RE = re.compile(
    r"Benders converged in (?P<num_iterations>\d+) iterations "
    r"\((?P<solve_time>[0-9.]+)s\): new_admitted=(?P<num_admitted>\d+)/"
    r"(?P<num_pending>\d+) pending, active=(?P<num_active>\d+), "
    r"goodput=(?P<goodput>[0-9.]+)"
)
BENDERS_FALLBACK_RE = re.compile(
    r"Benders did not converge in (?P<num_iterations>\d+) iterations "
    r"\((?P<solve_time>[0-9.]+)s\); falling back to greedy"
)
BENDERS_CKPT_CLASSES_RE = re.compile(r"Benders ckpt_classes:\s*(?P<body>.*)")
ENGINE_PID_RE = re.compile(r"EngineCore_DP(?P<replica>\d+) pid=(?P<pid>\d+)")

REQUEST_CSV_FIELDS = [
    "request_id",
    "server_request_id",
    "arrival_time",
    "prompt_len",
    "expected_output_len",
    "send_time",
    "end_time",
    "ttft_ms",
    "e2e_ms",
    "output_tokens",
    "tpot_ms",
    "max_gap_ms",
    "success",
    "error",
    "was_rerouted",
    "affected_by_failure",
    "ownership_known",
    "detection_ms",
    "restore_time_ms",
    "replay_tokens",
    "replay_time_ms",
    "initial_gpu",
    "resumed_gpu",
    "checkpoint_class",
    "ttft_slo_ms",
    "tpot_slo_ms",
    "failure_gap_slo_ms",
    "ttft_violated",
    "tpot_violated",
    "gap_violated",
]

EPOCHS_CSV_FIELDS = [
    "epoch_id",
    "timestamp",
    "num_iterations",
    "total_solve_time_sec",
    "master_time_sec",
    "recovery_time_sec",
    "num_cuts",
    "num_admitted",
    "num_pending",
    "num_active",
    "goodput",
    "fallback",
]


def _strip_ansi(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", line).strip()


def _parse_coarse_timestamp(line: str) -> float | None:
    match = LEADING_TIMESTAMP_RE.search(line)
    if match is None:
        return None
    now = datetime.now()
    dt = datetime(
        year=now.year,
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
        second=int(match.group("second")),
    )
    return float(int(dt.timestamp()))


def _has_subsecond_precision(value: float | None) -> bool:
    return value is not None and abs(value - round(value)) > 1e-6


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


def _serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_str(value)
    return value


def _parse_kv_pairs(body: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in KV_PAIR_RE.finditer(body)}


def _request_lookup_keys(request_id: str | None) -> set[str]:
    if not request_id:
        return set()

    keys = {request_id}
    if request_id.startswith("chatcmpl-") and request_id.count("-") >= 2:
        keys.add(request_id.rsplit("-", 1)[0])

    match = re.match(r"^(chatcmpl-req-\d+)(?:-.+)?$", request_id)
    if match:
        keys.add(match.group(1))
        keys.add(match.group(1).removeprefix("chatcmpl-"))

    for candidate in tuple(keys):
        for prefix in ("chatcmpl-", "cmpl-", "resp-"):
            if candidate.startswith(prefix):
                keys.add(candidate[len(prefix):])
            else:
                keys.add(f"{prefix}{candidate}")

    return {key for key in keys if key}


def _find_best_event_for_result(
    events: list[dict[str, Any]],
    result: "RequestResult",
) -> dict[str, Any] | None:
    if not events:
        return None

    result_keys = _request_lookup_keys(result.request_id)
    result_keys.update(_request_lookup_keys(result.server_request_id))

    for event in events:
        event_keys = _request_lookup_keys(event.get("request_id"))
        if result_keys & event_keys:
            return event
    return None


def _parse_json_safely(data: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _map_fixed_checkpoint_level(fixed_blocks: int) -> int:
    mapping = {
        0: -1,
        10: 1,
        1: 2,
    }
    return mapping.get(fixed_blocks, fixed_blocks)


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            cwd=str(REPO_ROOT),
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _read_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class RequestResult:
    request_id: str
    arrival_time: float
    prompt_len: int
    expected_output_len: int
    server_request_id: str | None = None
    send_time: float | None = None
    end_time: float | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    tpot_ms: float | None = None
    max_gap_ms: float = 0.0
    success: bool = False
    error: str | None = None
    was_rerouted: bool = False
    affected_by_failure: bool = False
    ownership_known: bool = False
    detection_ms: float | None = None
    restore_time_ms: float | None = None
    replay_tokens: int | None = None
    replay_time_ms: float | None = None
    initial_gpu: int | None = None
    resumed_gpu: int | None = None
    checkpoint_class: int | None = None
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    failure_gap_slo_ms: float | None = None
    ttft_violated: bool = False
    tpot_violated: bool = False
    gap_violated: bool = False
    admitted: bool = True

    def finalize_metrics(self) -> None:
        if self.output_tokens > 1 and self.ttft_ms is not None and self.e2e_ms is not None:
            self.tpot_ms = (self.e2e_ms - self.ttft_ms) / (self.output_tokens - 1)

        self.ttft_violated = (
            self.ttft_ms is not None
            and self.ttft_slo_ms is not None
            and self.ttft_ms > self.ttft_slo_ms
        )
        self.tpot_violated = (
            self.tpot_ms is not None
            and self.tpot_slo_ms is not None
            and self.tpot_ms > self.tpot_slo_ms
        )
        self.gap_violated = (
            self.affected_by_failure
            and self.failure_gap_slo_ms is not None
            and self.max_gap_ms > self.failure_gap_slo_ms
        )

    def to_csv_row(self) -> dict[str, Any]:
        row = {
            "request_id": self.request_id,
            "server_request_id": self.server_request_id,
            "arrival_time": self.arrival_time,
            "prompt_len": self.prompt_len,
            "expected_output_len": self.expected_output_len,
            "send_time": self.send_time,
            "end_time": self.end_time,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "output_tokens": self.output_tokens,
            "tpot_ms": self.tpot_ms,
            "max_gap_ms": self.max_gap_ms,
            "success": self.success,
            "error": self.error,
            "was_rerouted": self.was_rerouted,
            "affected_by_failure": self.affected_by_failure,
            "ownership_known": self.ownership_known,
            "detection_ms": self.detection_ms,
            "restore_time_ms": self.restore_time_ms,
            "replay_tokens": self.replay_tokens,
            "replay_time_ms": self.replay_time_ms,
            "initial_gpu": self.initial_gpu,
            "resumed_gpu": self.resumed_gpu,
            "checkpoint_class": self.checkpoint_class,
            "ttft_slo_ms": self.ttft_slo_ms,
            "tpot_slo_ms": self.tpot_slo_ms,
            "failure_gap_slo_ms": self.failure_gap_slo_ms,
            "ttft_violated": self.ttft_violated,
            "tpot_violated": self.tpot_violated,
            "gap_violated": self.gap_violated,
        }
        return {key: _serialize_csv_value(row[key]) for key in REQUEST_CSV_FIELDS}


class LogParser:
    """Parse server.log into epochs, recovery events, and ckpt classes."""

    @staticmethod
    def _parse_fault_event(
        event_type: str,
        body: str,
        coarse_timestamp: float | None,
    ) -> dict[str, Any] | None:
        data = _parse_kv_pairs(body)
        wall_time = _safe_float(data.get("wall_time"))

        if event_type == "monitor_observed":
            replica_id = _safe_int(data.get("engine") or data.get("replica"))
            return {
                "type": "monitor_observed",
                "replica_id": replica_id,
                "wall_time": wall_time,
                "source": data.get("source"),
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
            }

        if event_type == "failure_declared":
            replica_id = _safe_int(data.get("replica") or data.get("engine"))
            return {
                "type": "failure_declared",
                "replica_id": replica_id,
                "wall_time": wall_time,
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
            }

        if event_type == "failover_start":
            replica_id = _safe_int(data.get("engine") or data.get("replica"))
            return {
                "type": "failover_start",
                "replica_id": replica_id,
                "wall_time": wall_time,
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
                "source": "fault_event",
            }

        if event_type == "failover_complete":
            return {
                "type": "failover_complete",
                "replica_id": _safe_int(data.get("engine") or data.get("replica")) or -1,
                "wall_time": wall_time,
                "recovered": _safe_int(data.get("rerouted")) or _safe_int(data.get("recovered")) or 0,
                "total": _safe_int(data.get("total")) or 0,
                "dropped": _safe_int(data.get("dropped")) or 0,
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
                "source": "fault_event",
            }

        if event_type == "request_route":
            request_id = data.get("request")
            if request_id is None:
                return None
            return {
                "type": "request_route",
                "request_id": request_id,
                "gpu": _safe_int(data.get("gpu")),
                "wall_time": wall_time,
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
            }

        if event_type == "kv_restore_done":
            request_id = data.get("request")
            if request_id is None:
                return None
            return {
                "type": "kv_restore_done",
                "request_id": request_id,
                "wall_time": wall_time,
                "tokens": _safe_int(data.get("tokens")) or 0,
                "blocks": _safe_int(data.get("blocks")) or 0,
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
            }

        if event_type == "first_token_after_recovery":
            request_id = data.get("request")
            if request_id is None:
                return None
            return {
                "type": "first_token_after_recovery",
                "request_id": request_id,
                "wall_time": wall_time,
                "timestamp": wall_time if wall_time is not None else coarse_timestamp,
            }

        return None

    @classmethod
    def parse(
        cls, log_path: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        epochs: list[dict[str, Any]] = []
        recoveries: list[dict[str, Any]] = []
        ckpt_classes: dict[str, int] = {}

        with open(log_path, errors="replace") as f:
            for raw_line in f:
                line = _strip_ansi(raw_line)
                if not line:
                    continue

                coarse_timestamp = _parse_coarse_timestamp(line)

                match = BENDERS_CONVERGED_RE.search(line)
                if match:
                    epochs.append({
                        "epoch_id": len(epochs) + 1,
                        "timestamp": coarse_timestamp,
                        "num_iterations": int(match.group("num_iterations")),
                        "total_solve_time_sec": float(match.group("solve_time")),
                        "master_time_sec": 0.0,
                        "recovery_time_sec": 0.0,
                        "num_cuts": 0,
                        "num_admitted": int(match.group("num_admitted")),
                        "num_pending": int(match.group("num_pending")),
                        "num_active": int(match.group("num_active")),
                        "goodput": float(match.group("goodput")),
                        "fallback": False,
                    })
                    continue

                match = BENDERS_FALLBACK_RE.search(line)
                if match:
                    epochs.append({
                        "epoch_id": len(epochs) + 1,
                        "timestamp": coarse_timestamp,
                        "num_iterations": int(match.group("num_iterations")),
                        "total_solve_time_sec": float(match.group("solve_time")),
                        "master_time_sec": 0.0,
                        "recovery_time_sec": 0.0,
                        "num_cuts": 0,
                        "num_admitted": 0,
                        "num_pending": 0,
                        "num_active": 0,
                        "goodput": 0.0,
                        "fallback": True,
                    })
                    continue

                match = BENDERS_CKPT_CLASSES_RE.search(line)
                if match:
                    body = match.group("body").strip()
                    if body:
                        for pair in body.split():
                            if ":" not in pair:
                                continue
                            request_id, class_str = pair.rsplit(":", 1)
                            ckpt_level = _safe_int(class_str)
                            if ckpt_level is not None:
                                ckpt_classes[request_id] = ckpt_level
                    continue

                match = FAULT_EVENT_RE.search(line)
                if match:
                    event = cls._parse_fault_event(
                        match.group("event"),
                        match.group("body"),
                        coarse_timestamp,
                    )
                    if event is not None:
                        recoveries.append(event)
                    continue

                match = FAILOVER_START_RE.search(line)
                if match:
                    recoveries.append({
                        "type": "failover_start",
                        "replica_id": int(match.group("replica")),
                        "timestamp": coarse_timestamp,
                        "source": "ft_client",
                    })
                    continue

                match = REROUTE_RE.search(line)
                if match:
                    wall_time = _safe_float(match.group("wall_time"))
                    recoveries.append({
                        "type": "reroute",
                        "request_id": match.group("request_id"),
                        "initial_gpu": int(match.group("initial")),
                        "resumed_gpu": int(match.group("resumed")),
                        "tokens_restored": int(match.group("restored")),
                        "replay_tokens": int(match.group("replay")),
                        "est_gap_ms": float(match.group("gap")),
                        "slo_met": match.group("slo_met") == "True",
                        "timestamp": wall_time if wall_time is not None else coarse_timestamp,
                        "wall_time": wall_time,
                    })
                    continue

                match = FAILOVER_COMPLETE_RE.search(line)
                if match:
                    recoveries.append({
                        "type": "failover_complete",
                        "replica_id": -1,
                        "recovered": int(match.group("recovered")),
                        "total": int(match.group("total")),
                        "dropped": 0,
                        "timestamp": coarse_timestamp,
                        "source": "ft_client",
                    })

        return epochs, recoveries, ckpt_classes


def _find_failed_replica(recoveries: list[dict[str, Any]]) -> int | None:
    for event in recoveries:
        if event.get("type") == "failure_declared":
            return _safe_int(event.get("replica_id"))
    for event in recoveries:
        if event.get("type") == "failover_start":
            return _safe_int(event.get("replica_id"))
    return None


def _coarse_split_gap(
    max_gap_ms: float,
    tokens_restored: int,
    replay_tokens: int,
) -> tuple[float, float]:
    if max_gap_ms <= 0:
        return 0.0, 0.0

    total_tokens = max(tokens_restored, 0) + max(replay_tokens, 0)
    if total_tokens <= 0:
        return 0.0, max_gap_ms

    restore_ms = max_gap_ms * (max(tokens_restored, 0) / total_tokens)
    replay_ms = max_gap_ms - restore_ms
    if tokens_restored <= 0:
        restore_ms = 0.0
        replay_ms = max_gap_ms
    if replay_tokens <= 0:
        restore_ms = max_gap_ms
        replay_ms = 0.0
    return restore_ms, replay_ms


def _enrich_results(
    results: list[RequestResult],
    recoveries: list[dict[str, Any]],
    fault_injection_time: float | None,
    ckpt_classes: dict[str, int] | None,
) -> None:
    routes = [event for event in recoveries if event.get("type") == "request_route"]
    reroutes = [event for event in recoveries if event.get("type") == "reroute"]
    kv_restores = [
        event for event in recoveries
        if event.get("type") in {"kv_restore", "kv_restore_done"}
    ]
    first_tokens = [
        event for event in recoveries
        if event.get("type") == "first_token_after_recovery"
    ]
    failed_replica = _find_failed_replica(recoveries)

    for result in results:
        route_event = _find_best_event_for_result(routes, result)
        reroute_event = _find_best_event_for_result(reroutes, result)
        kv_restore_event = _find_best_event_for_result(kv_restores, result)
        first_token_event = _find_best_event_for_result(first_tokens, result)

        if ckpt_classes:
            result_keys = _request_lookup_keys(result.request_id)
            result_keys.update(_request_lookup_keys(result.server_request_id))
            for ckpt_request_id, ckpt_level in ckpt_classes.items():
                if result_keys & _request_lookup_keys(ckpt_request_id):
                    result.checkpoint_class = ckpt_level
                    break

        if route_event is not None:
            result.initial_gpu = _safe_int(route_event.get("gpu"))
            result.ownership_known = result.initial_gpu is not None

        if reroute_event is not None:
            result.was_rerouted = True
            result.affected_by_failure = True
            result.ownership_known = True
            result.initial_gpu = _safe_int(reroute_event.get("initial_gpu"))
            result.resumed_gpu = _safe_int(reroute_event.get("resumed_gpu"))
            result.replay_tokens = _safe_int(reroute_event.get("replay_tokens")) or 0

            reroute_wall_time = _safe_float(reroute_event.get("wall_time"))
            kv_restore_wall_time = _safe_float(
                kv_restore_event.get("wall_time") if kv_restore_event else None
            )
            first_token_wall_time = _safe_float(
                first_token_event.get("wall_time") if first_token_event else None
            )

            precise_restore = False
            precise_replay = False

            if _has_subsecond_precision(reroute_wall_time) and fault_injection_time is not None:
                result.detection_ms = max(
                    0.0,
                    (reroute_wall_time - fault_injection_time) * 1000,
                )

            if (
                _has_subsecond_precision(reroute_wall_time)
                and _has_subsecond_precision(kv_restore_wall_time)
                and kv_restore_wall_time is not None
                and reroute_wall_time is not None
                and kv_restore_wall_time > reroute_wall_time
            ):
                result.restore_time_ms = max(
                    0.0,
                    (kv_restore_wall_time - reroute_wall_time) * 1000,
                )
                precise_restore = True

            replay_base = kv_restore_wall_time if precise_restore else reroute_wall_time
            if (
                _has_subsecond_precision(replay_base)
                and _has_subsecond_precision(first_token_wall_time)
                and replay_base is not None
                and first_token_wall_time is not None
                and first_token_wall_time > replay_base
            ):
                result.replay_time_ms = max(
                    0.0,
                    (first_token_wall_time - replay_base) * 1000,
                )
                precise_replay = True

            if not precise_restore or not precise_replay:
                restore_ms, replay_ms = _coarse_split_gap(
                    result.max_gap_ms,
                    _safe_int(reroute_event.get("tokens_restored")) or 0,
                    result.replay_tokens or 0,
                )
                if not precise_restore:
                    result.restore_time_ms = restore_ms
                if not precise_replay:
                    result.replay_time_ms = replay_ms

        if (
            failed_replica is not None
            and result.ownership_known
            and result.initial_gpu == failed_replica
            and fault_injection_time is not None
            and result.send_time is not None
            and result.send_time <= fault_injection_time
            and (
                result.end_time is None
                or result.end_time == 0
                or result.end_time >= fault_injection_time
            )
        ):
            result.affected_by_failure = True

        result.finalize_metrics()


async def _wait_for_server(port: int, timeout_sec: float) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout_sec
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)
    return False


def _find_engine_pids_from_log(log_path: Path) -> dict[int, int]:
    pids: dict[int, int] = {}
    if not log_path.exists():
        return pids

    with log_path.open(errors="replace") as f:
        for raw_line in f:
            line = _strip_ansi(raw_line)
            match = ENGINE_PID_RE.search(line)
            if match:
                pids[int(match.group("replica"))] = int(match.group("pid"))
    return pids


async def _wait_for_engine_pids(
    log_path: Path,
    expected_count: int,
    timeout_sec: float,
) -> dict[int, int]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        pids = _find_engine_pids_from_log(log_path)
        if len(pids) >= expected_count:
            return pids
        await asyncio.sleep(0.5)
    return _find_engine_pids_from_log(log_path)


def _fallback_child_pids(parent_pid: int) -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []
    return [int(pid) for pid in result.stdout.strip().splitlines() if pid.strip()]


def _build_server_flags(
    config: dict[str, Any],
    baseline: dict[str, Any],
    port: int,
) -> list[str]:
    flags = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config["model"]),
        "--port",
        str(port),
        "--max-model-len",
        str(config["max_model_len"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--dtype",
        str(config["dtype"]),
        "--data-parallel-size",
        str(config["dp_size"]),
        "--scheduling-policy",
        str(baseline["scheduling_policy"]),
    ]

    if config.get("enforce_eager"):
        flags.append("--enforce-eager")
    if baseline.get("enable_checkpointing"):
        flags.append("--enable-checkpointing")

    flags.extend([
        "--max-gpu-failures",
        str(baseline.get("max_gpu_failures", 0)),
        "--fixed-checkpoint-level",
        str(_map_fixed_checkpoint_level(baseline.get("fixed_checkpoint_blocks", 0))),
        "--failure-timeout-sec",
        str(config["failure_timeout_sec"]),
        "--failure-detection-time-ms",
        str(config["failure_detection_time_ms"]),
        "--checkpoint-pool-bytes",
        str(config["checkpoint_pool_bytes"]),
        "--default-ttft-slo-ms",
        str(config["slo"]["ttft_ms"]),
        "--default-tpot-slo-ms",
        str(config["slo"]["tpot_ms"]),
        "--default-failure-gap-slo-ms",
        str(config["slo"]["failure_gap_ms"]),
    ])

    if baseline["scheduling_policy"] in {"fault_tolerant", "ft_benders_centralized"}:
        flags.extend([
            "--ft-prefill-throughput",
            str(config["ft_prefill_throughput"]),
            "--ft-decode-throughput",
            str(config["ft_decode_throughput"]),
            "--ft-load-bandwidth",
            str(config["ft_load_bandwidth"]),
            "--ft-planning-horizon",
            str(config["ft_planning_horizon"]),
        ])

    return flags


async def _send_streaming_request(
    session: aiohttp.ClientSession,
    port: int,
    model: str,
    spec: RequestSpec,
    seed: int,
    request_timeout_sec: float,
) -> RequestResult:
    url = f"http://localhost:{port}/v1/chat/completions"
    result = RequestResult(
        request_id=spec.request_id,
        server_request_id=f"chatcmpl-{spec.request_id}",
        arrival_time=spec.arrival_time,
        prompt_len=spec.prompt_len,
        expected_output_len=spec.expected_output_len,
        ttft_slo_ms=spec.ttft_slo_ms,
        tpot_slo_ms=spec.tpot_slo_ms,
        failure_gap_slo_ms=spec.failure_gap_slo_ms,
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt_text}],
        "max_tokens": spec.expected_output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "request_id": spec.request_id,
        "ttft_slo_ms": spec.ttft_slo_ms,
        "tpot_slo_ms": spec.tpot_slo_ms,
        "failure_gap_slo_ms": spec.failure_gap_slo_ms,
        "expected_output_len": spec.expected_output_len,
        "seed": seed,
        "temperature": 0.0,
    }

    start_time = time.time()
    result.send_time = start_time
    ttft_ms: float | None = None
    last_token_time: float | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    max_gap_ms = 0.0
    streamed_tokens = 0

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request_timeout_sec),
        ) as resp:
            if resp.status != 200:
                result.success = False
                result.admitted = False
                result.end_time = time.time()
                result.e2e_ms = (result.end_time - start_time) * 1000
                result.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                result.finalize_metrics()
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break

                chunk = _parse_json_safely(payload_str)
                if chunk is None:
                    continue

                if chunk.get("id"):
                    result.server_request_id = chunk["id"]

                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])

                for choice in chunk.get("choices", []):
                    token_ids = choice.get("token_ids") or []
                    if token_ids:
                        now = time.time()
                        if ttft_ms is None:
                            ttft_ms = (now - start_time) * 1000
                        if last_token_time is not None:
                            max_gap_ms = max(max_gap_ms, (now - last_token_time) * 1000)
                        last_token_time = now
                        streamed_tokens += len(token_ids)

                    finish_reason = choice.get("finish_reason") or finish_reason

        result.end_time = time.time()
        result.e2e_ms = (result.end_time - start_time) * 1000
        result.ttft_ms = ttft_ms
        result.output_tokens = max(streamed_tokens, completion_tokens or 0)
        result.max_gap_ms = max_gap_ms

        if finish_reason in {"abort", "error"}:
            result.success = False
            result.admitted = finish_reason != "abort"
            result.error = finish_reason
        else:
            result.success = True

        result.finalize_metrics()
        return result
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.end_time = time.time()
        result.e2e_ms = (result.end_time - start_time) * 1000
        result.success = False
        result.admitted = False
        result.error = str(exc)
        result.finalize_metrics()
        return result


def _build_metrics(results: list[RequestResult], actual_runtime_sec: float) -> dict[str, Any]:
    total_requests = len(results)
    completed = sum(1 for result in results if result.success)
    failed = total_requests - completed
    admitted = sum(1 for result in results if result.admitted)
    total_output_tokens = sum(result.output_tokens for result in results if result.success)

    ttfts = [result.ttft_ms for result in results if result.success and result.ttft_ms is not None]
    tpots = [result.tpot_ms for result in results if result.success and result.tpot_ms is not None]
    direct_hit_gaps = [
        result.max_gap_ms
        for result in results
        if result.success and result.ownership_known and result.affected_by_failure
    ]
    affected = [result for result in results if result.affected_by_failure]

    return {
        "total_requests": total_requests,
        "completed": completed,
        "failed": failed,
        "admission_rate": admitted / total_requests if total_requests else 0.0,
        "completion_rate": completed / total_requests if total_requests else 0.0,
        "goodput": total_output_tokens / actual_runtime_sec if actual_runtime_sec > 0 else 0.0,
        "ttft_p50_ms": _percentile([float(v) for v in ttfts], 50),
        "ttft_p95_ms": _percentile([float(v) for v in ttfts], 95),
        "ttft_p99_ms": _percentile([float(v) for v in ttfts], 99),
        "tpot_p50_ms": _percentile([float(v) for v in tpots], 50),
        "tpot_p95_ms": _percentile([float(v) for v in tpots], 95),
        "tpot_p99_ms": _percentile([float(v) for v in tpots], 99),
        "slo_violation_rate": (
            sum(
                1 for result in results
                if result.ttft_violated or result.tpot_violated or result.gap_violated
            ) / total_requests if total_requests else 0.0
        ),
        "failover_gap_p50_ms": _percentile([float(v) for v in direct_hit_gaps], 50),
        "failover_gap_p95_ms": _percentile([float(v) for v in direct_hit_gaps], 95),
        "failover_gap_p99_ms": _percentile([float(v) for v in direct_hit_gaps], 99),
        "time_to_stable_sec": None,
        "recovery_success_rate": (
            sum(1 for result in affected if result.success) / len(affected)
            if affected else 1.0
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_requests_csv(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUEST_CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


def _write_epochs_csv(path: Path, epochs: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCHS_CSV_FIELDS)
        writer.writeheader()
        for epoch in epochs:
            writer.writerow({
                key: _serialize_csv_value(epoch.get(key))
                for key in EPOCHS_CSV_FIELDS
            })


async def _run_single_experiment(args: argparse.Namespace) -> None:
    config = _read_config(args.config)
    baseline = config["baselines"][args.baseline]
    workload = config["workloads"][args.workload]
    load_rps = float(config["load_levels"][args.load])
    fault_time_sec = config["fault_timing"][args.fault]
    run_duration_sec = float(config["run_duration_sec"])
    warmup_sec = float(config.get("warmup_sec", 0.0))
    trace_duration_sec = max(run_duration_sec - warmup_sec, 0.0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = generate_trace(
        workload_config=workload,
        rps=load_rps,
        duration_sec=trace_duration_sec,
        seed=args.seed,
        slo_config=config["slo"],
        warmup_sec=warmup_sec,
    )

    server_flags = _build_server_flags(config, baseline, args.port)
    log_path = output_dir / SERVER_LOG_NAME
    log_handle = log_path.open("w", buffering=1)
    server_proc = subprocess.Popen(
        server_flags,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    experiment_start: float | None = None
    fault_injection_time: float | None = None
    failed_gpu_id: int | None = None
    active_requests_at_fault: int | None = None

    try:
        ready = await _wait_for_server(
            args.port,
            float(config.get("server_startup_timeout", 300.0)),
        )
        if not ready:
            raise RuntimeError("Server failed to become healthy before timeout")

        engine_pids = await _wait_for_engine_pids(
            log_path,
            int(config["dp_size"]),
            20.0,
        )
        experiment_start = time.time()
        in_flight: set[str] = set()
        results_by_id: dict[str, RequestResult] = {}

        async def send_one(spec: RequestSpec) -> RequestResult:
            sleep_sec = max(0.0, experiment_start + spec.arrival_time - time.time())
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
            in_flight.add(spec.request_id)
            try:
                async with aiohttp.ClientSession() as session:
                    result = await _send_streaming_request(
                        session=session,
                        port=args.port,
                        model=config["model"],
                        spec=spec,
                        seed=args.seed,
                        request_timeout_sec=float(config.get("request_timeout_sec", 60.0)),
                    )
                return result
            finally:
                in_flight.discard(spec.request_id)

        async def inject_fault() -> None:
            nonlocal fault_injection_time, failed_gpu_id, active_requests_at_fault
            if fault_time_sec is None:
                return
            assert experiment_start is not None
            delay = max(0.0, experiment_start + float(fault_time_sec) - time.time())
            if delay > 0:
                await asyncio.sleep(delay)

            target_replica = 0 if 0 in engine_pids else (min(engine_pids) if engine_pids else None)
            target_pid = engine_pids.get(target_replica) if target_replica is not None else None
            if target_pid is None:
                child_pids = _fallback_child_pids(server_proc.pid)
                if child_pids:
                    target_pid = child_pids[-1]
            if target_pid is None:
                return

            active_requests_at_fault = len(in_flight)
            os.kill(target_pid, signal.SIGKILL)
            fault_injection_time = time.time()
            failed_gpu_id = target_replica

        request_tasks = [asyncio.create_task(send_one(spec)) for spec in trace]
        fault_task = asyncio.create_task(inject_fault())

        task_results = await asyncio.gather(*request_tasks, return_exceptions=True)
        await fault_task

        for spec, task_result in zip(trace, task_results, strict=True):
            if isinstance(task_result, RequestResult):
                results_by_id[spec.request_id] = task_result
            else:
                result = RequestResult(
                    request_id=spec.request_id,
                    server_request_id=f"chatcmpl-{spec.request_id}",
                    arrival_time=spec.arrival_time,
                    prompt_len=spec.prompt_len,
                    expected_output_len=spec.expected_output_len,
                    ttft_slo_ms=spec.ttft_slo_ms,
                    tpot_slo_ms=spec.tpot_slo_ms,
                    failure_gap_slo_ms=spec.failure_gap_slo_ms,
                    success=False,
                    admitted=False,
                    error=str(task_result),
                )
                result.finalize_metrics()
                results_by_id[spec.request_id] = result

        await asyncio.sleep(1.0)
        actual_runtime_sec = time.time() - experiment_start

        epochs, recoveries, ckpt_classes = LogParser.parse(str(log_path))
        ordered_results = [results_by_id[spec.request_id] for spec in trace]
        _enrich_results(
            ordered_results,
            recoveries,
            fault_injection_time=fault_injection_time,
            ckpt_classes=ckpt_classes,
        )
        for result in ordered_results:
            result.finalize_metrics()

        metrics = _build_metrics(ordered_results, actual_runtime_sec)
        run_meta = {
            "git_commit": _get_git_commit(),
            "model": config["model"],
            "gpu_type": "",
            "gpu_count": 0,
            "baseline": args.baseline,
            "workload": args.workload,
            "load_level": args.load,
            "load_rps": load_rps,
            "fault": args.fault,
            "fault_time_sec": fault_time_sec,
            "seed": args.seed,
            "run_duration_sec": run_duration_sec,
            "actual_runtime_sec": actual_runtime_sec,
            "server_flags": server_flags,
            "policy": baseline["scheduling_policy"],
            "failure_gap_slo_ms": config["slo"]["failure_gap_ms"],
            "fault_injection_time": fault_injection_time,
            "failed_gpu_id": failed_gpu_id,
            "active_requests_at_fault": active_requests_at_fault,
        }

        _write_json(output_dir / "run_meta.json", run_meta)
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "recoveries.json", recoveries)
        _write_requests_csv(output_dir / "requests.csv", ordered_results)
        _write_epochs_csv(output_dir / "epochs.csv", epochs)
    finally:
        if server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="FT experiment runner")
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    asyncio.run(_run_single_experiment(args))


if __name__ == "__main__":
    main()
