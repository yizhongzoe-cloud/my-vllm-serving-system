
"""Single experiment run for FT serving evaluation.

Invoked by suite.py as a subprocess.  Launches a vLLM server, sends a
workload trace, optionally injects a GPU fault, collects per-request
metrics, parses server logs for recovery events, and saves results.

Usage:
    python experiments/run.py \
        --config experiments/config.yaml \
        --baseline Our-System \
        --workload W1_Short_Interactive \
        --load Medium \
        --fault F2_Mid \
        --seed 42 \
        --output-dir results/E1_Main/Our-System/W1_Short_Interactive/Medium/F2_Mid/42 \
        --port 8300
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Allow ``python experiments/run.py`` to resolve the repo root package.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments_v2.workloads import RequestSpec, generate_trace  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class RequestResult:
    """Per-request metrics collected by the client."""

    request_id: str
    arrival_time: float
    prompt_len: int
    expected_output_len: int
    server_request_id: str | None = None

    # Timing (seconds, absolute epoch).
    send_time: float = 0.0
    first_token_time: float | None = None
    last_token_time: float | None = None
    end_time: float | None = None

    # Derived (milliseconds).
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    tpot_ms: float | None = None  # mean time per output token

    # Streaming analysis.
    max_gap_ms: float = 0.0  # largest gap between consecutive tokens
    token_timestamps: list[float] = field(default_factory=list)

    # Status.
    success: bool = False
    error: str | None = None
    was_rerouted: bool = False
    affected_by_failure: bool = False
    ownership_known: bool = False

    # Dataset source (v2).
    dataset: str = ""

    # Recovery (populated post-hoc from log data).
    detection_ms: float | None = None
    restore_time_ms: float | None = None
    tokens_restored: int | None = None
    replay_tokens: int | None = None
    replay_time_ms: float | None = None
    initial_gpu: int | None = None
    resumed_gpu: int | None = None
    checkpoint_class: int | None = None

    # SLO.
    ttft_slo_ms: float = 0.0
    tpot_slo_ms: float = 0.0
    failure_gap_slo_ms: float = 0.0
    ttft_violated: bool = False
    tpot_violated: bool = False
    gap_violated: bool = False
    admitted: bool = True

    def finalize_metrics(self) -> None:
        """Compute derived metrics (tpot) and SLO violations in one place."""
        if self.output_tokens > 1 and self.ttft_ms is not None and self.e2e_ms is not None:
            self.tpot_ms = (self.e2e_ms - self.ttft_ms) / (self.output_tokens - 1)

        self.ttft_violated = (
            self.ttft_ms is not None
            and self.ttft_slo_ms > 0
            and self.ttft_ms > self.ttft_slo_ms
        )
        self.tpot_violated = (
            self.tpot_ms is not None
            and self.tpot_slo_ms > 0
            and self.tpot_ms > self.tpot_slo_ms
        )
        self.gap_violated = (
            self.affected_by_failure
            and self.failure_gap_slo_ms > 0
            and self.max_gap_ms > self.failure_gap_slo_ms
        )

    def to_csv_row(self) -> dict[str, Any]:
        """Serialize to a dict suitable for csv.DictWriter."""
        def _csv_val(v: Any) -> Any:
            if v is None:
                return ""
            if isinstance(v, bool):
                return "True" if v else "False"
            return v

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
            "admitted": self.admitted,
            "was_rerouted": self.was_rerouted,
            "affected_by_failure": self.affected_by_failure,
            "ownership_known": self.ownership_known,
            "dataset": self.dataset,
            "detection_ms": self.detection_ms,
            "restore_time_ms": self.restore_time_ms,
            "tokens_restored": self.tokens_restored,
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
        return {k: _csv_val(row[k]) for k in _CSV_FIELDS}


@dataclass
class ExperimentMetadata:
    """Run-level metadata persisted as run_meta.json."""

    git_commit: str = ""
    model: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    baseline: str = ""
    workload: str = ""
    load_level: str = ""
    load_rps: float = 0.0
    fault: str = ""
    fault_time_sec: float | None = None
    seed: int = 42
    run_duration_sec: float = 0.0
    actual_runtime_sec: float = 0.0
    server_flags: list[str] = field(default_factory=list)
    policy: str = ""
    failure_gap_slo_ms: float = 0.0
    fault_injection_time: float | None = None
    failed_gpu_id: int | None = None
    active_requests_at_fault: int | None = None
    in_flight_at_fault: list[str] = field(default_factory=list)


# ===================================================================
# Log parser
# ===================================================================

class LogParser:
    """Parse vLLM server logs for recovery events, epoch data, and
    checkpoint class assignments."""

    # -- Reroute (ft_client human-readable line) --
    _REROUTE_RE = re.compile(
        r"Request (\S+): re-routed (\d+)→(\d+), "
        r"restored=(\d+) tokens, replay=(\d+) tokens, "
        r"est_gap=([\d.]+)ms, slo_met=(\w+)"
        r"(?:, wall_time=(\d+\.\d+))?"
    )

    # -- FAULT_EVENT structured instrumentation --
    _FAULT_EVENT_MONITOR_RE = re.compile(
        r"FAULT_EVENT monitor_observed engine=(\d+) wall_time=(\d+\.\d+)"
        r"(?: source=(\S+))?"
    )
    _FAULT_EVENT_DECLARED_RE = re.compile(
        r"FAULT_EVENT failure_declared replica=(\d+) wall_time=(\d+\.\d+)"
    )
    _FAULT_EVENT_START_RE = re.compile(
        r"FAULT_EVENT failover_start engine=(\d+) wall_time=(\d+\.\d+)"
    )
    _FAULT_EVENT_COMPLETE_RE = re.compile(
        r"FAULT_EVENT failover_complete engine=(\d+) wall_time=(\d+\.\d+)"
        r" rerouted=(\d+) total=(\d+)"
    )
    _FAULT_EVENT_KV_RESTORE_DONE_RE = re.compile(
        r"FAULT_EVENT kv_restore_done request=(\S+) wall_time=(\d+\.\d+)"
        r" tokens=(\d+) blocks=(\d+)"
    )
    _FAULT_EVENT_FIRST_TOKEN_RE = re.compile(
        r"FAULT_EVENT first_token_after_recovery request=(\S+) "
        r"wall_time=(\d+\.\d+)"
    )
    _FAULT_EVENT_ROUTE_RE = re.compile(
        r"FAULT_EVENT request_route request=(\S+) gpu=(\d+) "
        r"wall_time=(\d+\.\d+)"
    )

    # -- Legacy ft_client patterns (no FAULT_EVENT prefix) --
    _LEGACY_FAILOVER_START_RE = re.compile(
        r"(?:Centralized )?FT Client: engine (\d+) declared FAILED\. "
        r"Starting failover for displaced requests\."
    )
    _LEGACY_FAILOVER_COMPLETE_RE = re.compile(
        r"(?:Centralized )?FT Client: failover complete\. Re-routed "
        r"(\d+)/(\d+) requests\."
    )
    _LEGACY_KV_RESTORE_RE = re.compile(
        r"KV restore for request (\S+): restored (\d+) tokens "
        r"\((\d+) blocks\)"
    )

    # -- Checkpoint class assignments --
    _CKPT_CLASSES_RE = re.compile(r"ckpt_classes:\s*(.*)")

    # -- Epoch / solver output --
    _EPOCH_CONVERGED_RE = re.compile(
        r"Benders converged in (\d+) iterations? \(([\d.]+)s\):\s*"
        r"new_admitted=(\d+)/(\d+) pending,\s*active=(\d+),\s*goodput=(\d+)"
    )
    _EPOCH_FALLBACK_RE = re.compile(
        r"Greedy fallback: dispatched (\d+) requests"
    )

    # -- Timestamp patterns in vLLM logs --
    _TS_RE_FULL = re.compile(
        r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )
    _TS_RE_SHORT = re.compile(
        r"(?<!\d)(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})(?!\d)"
    )

    @classmethod
    def _parse_timestamp(cls, line: str) -> float | None:
        """Extract a coarse log timestamp from a vLLM log line."""
        m = cls._TS_RE_FULL.search(line)
        if m:
            try:
                dt = datetime(
                    int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)), int(m.group(6)),
                )
                return dt.timestamp() + int(m.group(7)) / 1000.0
            except (ValueError, OSError):
                return None

        m = cls._TS_RE_SHORT.search(line)
        if m:
            try:
                now = datetime.now()
                dt = datetime(
                    now.year, int(m.group(1)), int(m.group(2)),
                    int(m.group(3)), int(m.group(4)), int(m.group(5)),
                )
                return dt.timestamp()
            except (ValueError, OSError):
                return None
        return None

    @classmethod
    def parse(cls, log_path: str) -> tuple[list[dict], list[dict], dict[str, int]]:
        """Parse a vLLM server log file.

        Returns:
            (epochs, recoveries, ckpt_classes)
        """
        epochs: list[dict] = []
        recoveries: list[dict] = []
        ckpt_classes: dict[str, int] = {}
        epoch_counter = 0

        if not os.path.exists(log_path):
            return epochs, recoveries, ckpt_classes

        with open(log_path) as f:
            for line in f:
                ts = cls._parse_timestamp(line)

                # ---- FAULT_EVENT structured events ----

                m = cls._FAULT_EVENT_MONITOR_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "monitor_observed",
                        "replica_id": int(m.group(1)),
                        "wall_time": float(m.group(2)),
                        "source": m.group(3) or "",
                        "timestamp": float(m.group(2)),
                    })
                    continue

                m = cls._FAULT_EVENT_DECLARED_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "failure_declared",
                        "replica_id": int(m.group(1)),
                        "wall_time": float(m.group(2)),
                        "timestamp": float(m.group(2)),
                    })
                    continue

                m = cls._FAULT_EVENT_START_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "failover_start",
                        "replica_id": int(m.group(1)),
                        "wall_time": float(m.group(2)),
                        "timestamp": float(m.group(2)),
                        "source": "fault_event",
                    })
                    continue

                m = cls._FAULT_EVENT_COMPLETE_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "failover_complete",
                        "replica_id": int(m.group(1)),
                        "wall_time": float(m.group(2)),
                        "recovered": int(m.group(3)),
                        "total": int(m.group(4)),
                        "dropped": int(m.group(4)) - int(m.group(3)),
                        "timestamp": float(m.group(2)),
                        "source": "fault_event",
                    })
                    continue

                m = cls._FAULT_EVENT_KV_RESTORE_DONE_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "kv_restore_done",
                        "request_id": m.group(1),
                        "wall_time": float(m.group(2)),
                        "tokens": int(m.group(3)),
                        "blocks": int(m.group(4)),
                        "timestamp": float(m.group(2)),
                    })
                    continue

                m = cls._FAULT_EVENT_FIRST_TOKEN_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "first_token_after_recovery",
                        "request_id": m.group(1),
                        "wall_time": float(m.group(2)),
                        "timestamp": float(m.group(2)),
                    })
                    continue

                m = cls._FAULT_EVENT_ROUTE_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "request_route",
                        "request_id": m.group(1),
                        "gpu": int(m.group(2)),
                        "wall_time": float(m.group(3)),
                        "timestamp": float(m.group(3)),
                    })
                    continue

                # ---- Legacy ft_client patterns ----

                # Reroute (human-readable).
                m = cls._REROUTE_RE.search(line)
                if m:
                    ev: dict[str, Any] = {
                        "type": "reroute",
                        "request_id": m.group(1),
                        "initial_gpu": int(m.group(2)),
                        "resumed_gpu": int(m.group(3)),
                        "tokens_restored": int(m.group(4)),
                        "replay_tokens": int(m.group(5)),
                        "est_gap_ms": float(m.group(6)),
                        "slo_met": m.group(7).lower() == "true",
                        "timestamp": ts if ts else 0.0,
                    }
                    if m.group(8) is not None:
                        ev["wall_time"] = float(m.group(8))
                    recoveries.append(ev)
                    continue

                m = cls._LEGACY_FAILOVER_START_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "failover_start",
                        "replica_id": int(m.group(1)),
                        "timestamp": ts if ts else 0.0,
                        "source": "ft_client",
                    })
                    continue

                m = cls._LEGACY_FAILOVER_COMPLETE_RE.search(line)
                if m:
                    recovered = int(m.group(1))
                    total = int(m.group(2))
                    recoveries.append({
                        "type": "failover_complete",
                        "replica_id": -1,
                        "recovered": recovered,
                        "total": total,
                        "dropped": total - recovered,
                        "timestamp": ts if ts else 0.0,
                        "source": "ft_client",
                    })
                    continue

                m = cls._LEGACY_KV_RESTORE_RE.search(line)
                if m:
                    recoveries.append({
                        "type": "kv_restore",
                        "request_id": m.group(1),
                        "tokens_restored": int(m.group(2)),
                        "blocks_restored": int(m.group(3)),
                        "timestamp": ts if ts else 0.0,
                    })
                    continue

                # ---- Checkpoint class assignments ----
                m = cls._CKPT_CLASSES_RE.search(line)
                if m:
                    for pair in m.group(1).split():
                        parts = pair.split(":", 1)
                        if len(parts) == 2:
                            ckpt_classes[parts[0]] = int(parts[1])
                    continue

                # ---- Epoch / solver output ----
                m = cls._EPOCH_CONVERGED_RE.search(line)
                if m:
                    epoch_counter += 1
                    epochs.append({
                        "epoch_id": epoch_counter,
                        "timestamp": ts if ts else 0.0,
                        "num_iterations": int(m.group(1)),
                        "total_solve_time_sec": float(m.group(2)),
                        "master_time_sec": 0.0,
                        "recovery_time_sec": 0.0,
                        "num_cuts": 0,
                        "num_admitted": int(m.group(3)),
                        "num_pending": int(m.group(4)),
                        "num_active": int(m.group(5)),
                        "goodput": float(m.group(6)),
                        "fallback": False,
                    })
                    continue

                m = cls._EPOCH_FALLBACK_RE.search(line)
                if m:
                    epoch_counter += 1
                    epochs.append({
                        "epoch_id": epoch_counter,
                        "timestamp": ts if ts else 0.0,
                        "num_iterations": 0,
                        "total_solve_time_sec": 0.0,
                        "master_time_sec": 0.0,
                        "recovery_time_sec": 0.0,
                        "num_cuts": 0,
                        "num_admitted": int(m.group(1)),
                        "num_pending": int(m.group(1)),
                        "num_active": 0,
                        "goodput": 0.0,
                        "fallback": True,
                    })
                    continue

        return epochs, recoveries, ckpt_classes


# ===================================================================
# Result enrichment
# ===================================================================


def _find_failed_replica(recoveries: list[dict]) -> int | None:
    """Discover which replica failed from recovery events."""
    for event in recoveries:
        if event.get("type") == "failure_declared":
            rid = event.get("replica_id")
            if rid is not None:
                return int(rid)
    for event in recoveries:
        if event.get("type") == "failover_start":
            rid = event.get("replica_id")
            if rid is not None:
                return int(rid)
    return None


def _coarse_split_gap(
    max_gap_ms: float,
    tokens_restored: int,
    replay_tokens: int,
) -> tuple[float, float]:
    """Split max_gap_ms proportionally by token counts when no precise
    wall_time timestamps are available."""
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
    recoveries: list[dict],
    fault_injection_time: float | None,
    ckpt_classes: dict[str, int] | None = None,
    failed_gpu_id: int | None = None,
) -> None:
    """Enrich request results with recovery data and failure flags (in-place).

    Uses structured recovery events from LogParser:
    - 'reroute' events: initial/resumed GPU, restored/replay tokens, est_gap
    - 'kv_restore_done' events: wall_time for KV restore completion
    - 'first_token_after_recovery': wall_time for first token after failover
    - 'request_route': initial GPU assignment
    - ckpt_classes: solver-assigned checkpoint class per request
    """

    # Discover failed replica from recovery events if not passed explicitly.
    if failed_gpu_id is None:
        failed_gpu_id = _find_failed_replica(recoveries)

    def _request_lookup_keys(request_id: str | None) -> set[str]:
        """Generate all ID variants for fuzzy matching between client-side
        ``req-NNNNN`` and server-side ``chatcmpl-req-NNNNN-suffix``."""
        if not request_id:
            return set()

        keys = {request_id}
        # Strip last segment for chatcmpl- IDs with a stream/session suffix.
        if request_id.startswith("chatcmpl-") and request_id.count("-") >= 2:
            keys.add(request_id.rsplit("-", 1)[0])

        # Match chatcmpl-req-NNNNN-xxx patterns.
        m = re.match(r"^(chatcmpl-req-\d+)(?:-.+)?$", request_id)
        if m:
            keys.add(m.group(1))
            keys.add(m.group(1).removeprefix("chatcmpl-"))

        # Expand with/without common prefixes.
        for candidate in tuple(keys):
            for prefix in ("chatcmpl-", "cmpl-", "resp-"):
                if candidate.startswith(prefix):
                    keys.add(candidate[len(prefix):])
                else:
                    keys.add(f"{prefix}{candidate}")

        return {key for key in keys if key}

    def _has_subsecond_precision(ts: float | None) -> bool:
        if ts is None:
            return False
        return abs(ts - round(ts)) > 1e-6

    # Always fill checkpoint_class if available (not gated on fault).
    if ckpt_classes:
        for r in results:
            lookup_keys = (
                _request_lookup_keys(r.server_request_id)
                | _request_lookup_keys(r.request_id)
            )
            for key in lookup_keys:
                if key in ckpt_classes:
                    r.checkpoint_class = ckpt_classes[key]
                    break

    if fault_injection_time is None:
        return

    # Build indexed lookups from structured recovery events.
    reroute_map: dict[str, dict] = {}       # request_id -> reroute event
    kv_done_map: dict[str, float] = {}      # request_id -> kv_restore_done wall_time
    first_tok_map: dict[str, float] = {}    # request_id -> first_token_after_recovery wall_time
    initial_route_map: dict[str, int] = {}  # request_id -> initial gpu index

    for rec in recoveries:
        rtype = rec.get("type", "")
        rid = rec.get("request_id", "")
        if rtype == "reroute" and rid:
            for key in _request_lookup_keys(rid):
                reroute_map[key] = rec
        elif rtype == "kv_restore_done" and rid:
            wt = rec.get("wall_time")
            if wt is not None:
                for key in _request_lookup_keys(rid):
                    kv_done_map[key] = wt
        elif rtype == "first_token_after_recovery" and rid:
            wt = rec.get("wall_time")
            if wt is not None:
                for key in _request_lookup_keys(rid):
                    first_tok_map[key] = wt
        elif rtype == "request_route" and rid:
            gpu = rec.get("gpu")
            if gpu is not None:
                for key in _request_lookup_keys(rid):
                    initial_route_map[key] = gpu

    # Compute fault_start: min of fault_injection_time and precise event
    # wall_times for monitor_observed / failure_declared / failover_start.
    fault_start_candidates: list[float] = []
    if fault_injection_time is not None:
        fault_start_candidates.append(fault_injection_time)
    for rec in recoveries:
        if rec.get("type") in ("monitor_observed", "failure_declared", "failover_start"):
            wt = rec.get("wall_time")
            if wt is not None:
                fault_start_candidates.append(float(wt))
    fault_start = min(fault_start_candidates) if fault_start_candidates else None

    for r in results:
        lookup_keys = (
            _request_lookup_keys(r.server_request_id)
            | _request_lookup_keys(r.request_id)
        )
        # 1. If this request has a reroute event, fill from structured data.
        rec = next((reroute_map[key] for key in lookup_keys if key in reroute_map), None)
        if rec is not None:
            r.affected_by_failure = True
            r.was_rerouted = True
            r.ownership_known = True
            r.initial_gpu = rec["initial_gpu"]
            r.resumed_gpu = rec["resumed_gpu"]
            r.tokens_restored = rec.get("tokens_restored")
            r.replay_tokens = rec["replay_tokens"]

            # Direct phase computation from wall_time timestamps.
            reroute_wall = rec.get("wall_time")
            kv_done_wall = next(
                (kv_done_map[key] for key in lookup_keys if key in kv_done_map),
                None,
            )
            first_tok_wall = next(
                (first_tok_map[key] for key in lookup_keys if key in first_tok_map),
                None,
            )

            if reroute_wall and first_tok_wall and fault_start:
                r.detection_ms = max(0.0, (reroute_wall - fault_start) * 1000)
                if kv_done_wall:
                    r.restore_time_ms = max(0.0, (kv_done_wall - reroute_wall) * 1000)
                    r.replay_time_ms = max(0.0, (first_tok_wall - kv_done_wall) * 1000)
                else:
                    r.restore_time_ms = 0.0
                    r.replay_time_ms = max(0.0, (first_tok_wall - reroute_wall) * 1000)
            else:
                # Coarse timestamps: split max_gap_ms proportionally.
                total_gap = r.max_gap_ms
                tokens_restored = rec.get("tokens_restored", 0)
                replay_tok = rec.get("replay_tokens", 0)
                total_tokens = tokens_restored + replay_tok
                if total_tokens > 0 and tokens_restored > 0:
                    r.restore_time_ms = tokens_restored / total_tokens * total_gap
                    r.replay_time_ms = replay_tok / total_tokens * total_gap
                else:
                    r.restore_time_ms = 0.0
                    r.replay_time_ms = total_gap

            continue

        # 2. Strict GPU-hit: initial_gpu == failed_gpu_id and in-flight at fault.
        initial_gpu = next(
            (initial_route_map[key] for key in lookup_keys if key in initial_route_map),
            None,
        )
        if initial_gpu is not None:
            r.initial_gpu = initial_gpu
            r.ownership_known = True

        if (
            failed_gpu_id is not None
            and initial_gpu is not None
            and initial_gpu == failed_gpu_id
            and fault_start is not None
            and r.send_time is not None
            and r.send_time < fault_start
            and r.end_time is not None
            and r.end_time > fault_start
        ):
            r.affected_by_failure = True

    # Set gap_violated only for failure-affected requests.
    for r in results:
        if (
            r.affected_by_failure
            and r.failure_gap_slo_ms > 0
            and r.max_gap_ms > r.failure_gap_slo_ms
        ):
            r.gap_violated = True


# ===================================================================
# Metrics computation
# ===================================================================

def _compute_time_to_stable(
    results: list[RequestResult],
    fault_time: float,
    window_sec: float = 1.0,
    threshold_frac: float = 0.9,
    stable_duration_sec: float = 10.0,
) -> float | None:
    """Compute time from fault until throughput recovers to ≥90% of
    pre-fault level and stays there for ``stable_duration_sec``."""
    # Build per-second token counts.
    completed = [r for r in results if r.success and r.end_time is not None]
    if not completed:
        return None

    min_t = min(r.send_time for r in completed)
    max_t = max(r.end_time for r in completed)  # type: ignore[arg-type]
    if max_t <= min_t:
        return None

    # Bin output tokens by completion second.
    nbins = int((max_t - min_t) / window_sec) + 1
    bins = [0.0] * nbins
    for r in completed:
        idx = int((r.end_time - min_t) / window_sec)  # type: ignore[operator]
        if 0 <= idx < nbins:
            bins[idx] += r.output_tokens

    fault_bin = int((fault_time - min_t) / window_sec)
    if fault_bin <= 0:
        return None

    # Pre-fault average throughput.
    pre_fault_bins = bins[:fault_bin]
    if not pre_fault_bins:
        return None
    pre_avg = sum(pre_fault_bins) / len(pre_fault_bins)
    if pre_avg <= 0:
        return None

    target = pre_avg * threshold_frac
    stable_bins_needed = int(stable_duration_sec / window_sec)
    consecutive = 0

    for i in range(fault_bin, nbins):
        if bins[i] >= target:
            consecutive += 1
            if consecutive >= stable_bins_needed:
                stable_start = i - stable_bins_needed + 1
                return (stable_start * window_sec + min_t) - fault_time
        else:
            consecutive = 0

    return None


def compute_metrics(
    results: list[RequestResult],
    actual_runtime: float,
    fault_time: float | None = None,
    recoveries: list[dict] | None = None,
    epochs: list[dict] | None = None,
) -> dict:
    """Compute aggregate metrics from per-request results.

    P0-impl-3a-followup (2026-04-08): SLO violation rate now includes
    admitted-but-failed requests (e.g. dropped due to GPU failure with
    no fault tolerance). Previously these were silently excluded
    because the metric only iterated `successful` requests, making
    No-FT look perfect under faults despite dropping requests.

    Definition: a request "violates SLO" if it was admitted by the
    server (HTTP 200 + no transport error) and either:
      (a) failed to complete (incomplete_stream / empty_stream / abort), OR
      (b) completed but violated TTFT, TPOT, or failover-gap SLO

    The denominator is `admitted` requests, not `total` (we don't
    blame the server for client-side admission rejections, e.g.
    HTTP 429 / 5xx, which are counted separately).
    """
    total = len(results)
    admitted = [r for r in results if r.admitted]
    n_admitted = len(admitted)
    successful = [r for r in results if r.success]
    completed = len(successful)
    failed = total - completed

    completion_rate = completed / total if total > 0 else 0.0

    # Goodput: output tokens from admitted requests satisfying SLO / actual runtime.
    # Note: an admitted-but-failed request contributes 0 tokens to goodput
    # (it didn't satisfy SLO). This is the same as before for tokens, but
    # the denominator (slo_violation_rate) now includes failed requests.
    slo_satisfied = [
        r for r in successful
        if not (r.ttft_violated or r.tpot_violated or r.gap_violated)
    ]
    total_output_tokens = sum(r.output_tokens for r in slo_satisfied)
    goodput = total_output_tokens / actual_runtime if actual_runtime > 0 else 0.0

    # Percentile helpers.
    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    ttft_vals = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    tpot_vals = [r.tpot_ms for r in successful if r.tpot_ms is not None]

    # Failover gap: max_gap_ms for affected successful requests.
    gap_vals = [
        r.max_gap_ms for r in successful
        if r.affected_by_failure
    ]

    # SLO violation: an admitted request violates SLO if it either:
    #   - failed to complete (admitted but no clean finish), OR
    #   - completed but violated TTFT / TPOT / failover-gap.
    # Denominator: admitted requests (not total — admission rejections
    # aren't the server's fault).
    violated_admitted_failed = sum(
        1 for r in admitted if not r.success
    )
    violated_admitted_slo = sum(
        1 for r in admitted
        if r.success and (r.ttft_violated or r.tpot_violated or r.gap_violated)
    )
    violated = violated_admitted_failed + violated_admitted_slo
    slo_violation_rate = (
        violated / n_admitted if n_admitted > 0 else 0.0
    )

    # Recovery success rate: fraction of failure-affected requests that succeeded.
    affected = [r for r in results if r.affected_by_failure]
    affected_ok = [r for r in affected if r.success]
    recovery_success_rate = (
        len(affected_ok) / len(affected) if affected else 1.0
    )

    # Time to stable.
    time_to_stable = None
    if fault_time is not None:
        time_to_stable = _compute_time_to_stable(results, fault_time)

    return {
        "total_requests": total,
        "completed": completed,
        "failed": failed,
        "admitted": n_admitted,
        "admission_rate": n_admitted / total if total > 0 else 0.0,
        "completion_rate": completion_rate,
        "goodput": goodput,
        "ttft_p50_ms": pct(ttft_vals, 50),
        "ttft_p95_ms": pct(ttft_vals, 95),
        "ttft_p99_ms": pct(ttft_vals, 99),
        "tpot_p50_ms": pct(tpot_vals, 50),
        "tpot_p95_ms": pct(tpot_vals, 95),
        "tpot_p99_ms": pct(tpot_vals, 99),
        "slo_violation_rate": slo_violation_rate,
        # P0-impl-3a-followup: breakdown of slo_violation_rate
        # so we can see how much comes from failed admitted requests
        # vs SLO-violated successful requests
        "slo_violations_admitted_failed": violated_admitted_failed,
        "slo_violations_admitted_slo": violated_admitted_slo,
        "slo_violation_rate_failed_only": (
            violated_admitted_failed / n_admitted if n_admitted > 0 else 0.0
        ),
        "slo_violation_rate_slo_only": (
            violated_admitted_slo / n_admitted if n_admitted > 0 else 0.0
        ),
        "failover_gap_p50_ms": pct(gap_vals, 50),
        "failover_gap_p95_ms": pct(gap_vals, 95),
        "failover_gap_p99_ms": pct(gap_vals, 99),
        "time_to_stable_sec": time_to_stable,
        "recovery_success_rate": recovery_success_rate,
    }


# ===================================================================
# Server management
# ===================================================================

def _build_server_cmd(config: dict, baseline_name: str, port: int) -> list[str]:
    """Build the vLLM server launch command from config and baseline."""
    baseline = config["baselines"][baseline_name]
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model"],
        "--port", str(port),
        "--max-model-len", str(config.get("max_model_len", 2048)),
        "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.9)),
        "--dtype", config.get("dtype", "auto"),
        "--data-parallel-size", str(config.get("dp_size", 1)),
    ]

    if config.get("enforce_eager"):
        cmd.append("--enforce-eager")

    policy = baseline.get("scheduling_policy", "fcfs")
    cmd.extend(["--scheduling-policy", policy])

    if baseline.get("enable_checkpointing"):
        cmd.append("--enable-checkpointing")

    max_failures = baseline.get("max_gpu_failures")
    if max_failures is not None:
        cmd.extend(["--max-gpu-failures", str(max_failures)])

    ckpt_level = baseline.get("fixed_checkpoint_level")
    if ckpt_level is not None and ckpt_level >= 0:
        cmd.extend(["--fixed-checkpoint-level", str(ckpt_level)])

    ckpt_blocks = baseline.get("fixed_checkpoint_blocks")
    if ckpt_blocks is not None and ckpt_blocks > 0:
        cmd.extend(["--fixed-checkpoint-blocks", str(ckpt_blocks)])

    # FT parameters from top-level config.
    for cfg_key, flag in [
        ("failure_timeout_sec", "--failure-timeout-sec"),
        ("failure_detection_time_ms", "--failure-detection-time-ms"),
        ("checkpoint_pool_bytes", "--checkpoint-pool-bytes"),
        ("ft_prefill_throughput", "--ft-prefill-throughput"),
        ("ft_decode_throughput", "--ft-decode-throughput"),
        ("ft_load_bandwidth", "--ft-load-bandwidth"),
        ("ft_planning_horizon", "--ft-planning-horizon"),
        ("ft_checkpoint_cost_profile", "--ft-checkpoint-cost-profile"),
        ("ft_decode_capacity_profile", "--ft-decode-capacity-profile"),
    ]:
        val = config.get(cfg_key)
        if val is not None and str(val).strip():
            cmd.extend([flag, str(val)])

    # SLO defaults.
    slo = config.get("slo", {})
    for slo_key, flag in [
        ("ttft_ms", "--default-ttft-slo-ms"),
        ("failure_gap_ms", "--default-failure-gap-slo-ms"),
    ]:
        val = slo.get(slo_key)
        if val is not None:
            cmd.extend([flag, str(val)])

    # TPOT SLO: use the most strict (min) across all workloads as server default.
    # Per-request TPOT SLO is passed in the request payload.
    workloads = config.get("workloads", {})
    tpot_values = [
        wl.get("tpot_slo_ms") for wl in workloads.values()
        if isinstance(wl.get("tpot_slo_ms"), (int, float))
    ]
    if tpot_values:
        cmd.extend(["--default-tpot-slo-ms", str(min(tpot_values))])

    return cmd


def _start_server(cmd: list[str], output_dir: str) -> subprocess.Popen:
    """Start the vLLM server, capturing output to server.log."""
    log_path = os.path.join(output_dir, "server.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    # Keep log_file handle on proc so we can close it later.
    proc._log_file = log_file  # type: ignore[attr-defined]
    logger.info("Server started (pid=%d), logging to %s", proc.pid, log_path)
    return proc


def _wait_for_health(port: int, timeout: float = 300.0) -> bool:
    """Poll the server /health endpoint until it responds 200."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                logger.info("Server healthy on port %d", port)
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    logger.error("Server health check timed out after %.0fs", timeout)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group.

    P0-impl-3a-followup fix (2026-04-08): vLLM dp=2 + ft_scheduler servers
    can take >15s to fully shut down (dp coordinator + 2 worker processes
    + ft cleanup). Previously the second `proc.wait(timeout=5)` would raise
    an uncaught TimeoutExpired, killing run.py and losing metrics.json
    even though benchmark had already completed. Now we catch all timeouts
    and trust the OS to reap the SIGKILL'd process group.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=15)  # was 10
    except subprocess.TimeoutExpired:
        logger.warning(
            "Server did not exit within 15s of SIGTERM; sending SIGKILL"
        )
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=10)  # was 5
        except subprocess.TimeoutExpired:
            logger.warning(
                "Server did not exit within 10s of SIGKILL; abandoning "
                "process (OS will reap). Continuing to next cell."
            )
    log_file = getattr(proc, "_log_file", None)
    if log_file:
        log_file.close()
    logger.info("Server stopped")


# ===================================================================
# Fault injection
# ===================================================================

def _find_engine_pids(server_pid: int) -> list[tuple[int, int]]:
    """Find (engine_index, pid) pairs for EngineCore worker processes.

    Uses psutil if available, falls back to /proc traversal.
    """
    try:
        import psutil
        parent = psutil.Process(server_pid)
        children = parent.children(recursive=True)
        engines = []
        idx = 0
        for child in children:
            try:
                cmdline = " ".join(child.cmdline())
                if "EngineCore" in cmdline or "engine_core" in cmdline:
                    engines.append((idx, child.pid))
                    idx += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return engines
    except ImportError:
        pass

    # Fallback: enumerate /proc children.
    engines = []
    idx = 0
    try:
        proc_dir = Path(f"/proc/{server_pid}/task/{server_pid}/children")
        if proc_dir.exists():
            child_pids = proc_dir.read_text().split()
            for cpid_str in child_pids:
                cpid = int(cpid_str)
                try:
                    cmdline = Path(f"/proc/{cpid}/cmdline").read_text()
                    if "EngineCore" in cmdline or "engine_core" in cmdline:
                        engines.append((idx, cpid))
                        idx += 1
                except (FileNotFoundError, PermissionError):
                    continue
    except (FileNotFoundError, PermissionError):
        pass
    return engines


def _inject_fault_at(
    server_proc: subprocess.Popen,
    metadata: ExperimentMetadata,
    request_results: list[RequestResult],
) -> None:
    """Kill one engine worker process to simulate a GPU failure."""
    engines = _find_engine_pids(server_proc.pid)
    if not engines:
        logger.error("No engine processes found for fault injection")
        return

    # Kill the first engine (engine index 0).
    killed_engine_idx, killed_pid = engines[0]
    logger.info("Injecting fault: killing engine %d (pid %d)", killed_engine_idx, killed_pid)

    try:
        os.kill(killed_pid, signal.SIGKILL)
    except (ProcessLookupError, OSError) as e:
        logger.error("Failed to kill engine %d: %s", killed_engine_idx, e)
        return

    metadata.fault_injection_time = time.time()
    metadata.failed_gpu_id = (
        killed_engine_idx if killed_pid else None
    )

    # Count active requests at fault time.
    active = sum(
        1 for r in request_results
        if r.send_time and r.send_time < metadata.fault_injection_time
        and (r.end_time is None or r.end_time > metadata.fault_injection_time)
    )
    metadata.active_requests_at_fault = active
    logger.info(
        "Fault injected: engine=%d, pid=%d, active_requests=%d",
        killed_engine_idx, killed_pid, active,
    )


def _classify_result(result: RequestResult, finish_reason: str | None) -> None:
    """Classify request outcome based on finish_reason, error, and output_tokens.

    Sets result.admitted, result.success, and result.error.

    P0-impl-3a-followup (2026-04-08): "admitted" now means "the server
    accepted the request and either returned data or started streaming",
    not just "no transport error". Previously, requests that started
    streaming but later hit a client-side timeout (e.g. due to backpressure)
    were marked admitted=False, which made them invisible to admission_rate
    and slo_violation_rate metrics. They are now correctly marked admitted=True
    so they count as SLO violations under the corrected metric definition.
    """
    # Mark admission. A request was admitted by the server if any of:
    #   1. No error at all (clean completion)
    #   2. HTTP 200 was returned (server accepted, even if stream incomplete)
    #   3. We received at least one token (server started streaming)
    # Only client-side rejections (HTTP 4xx/5xx, connection refused) are NOT admitted.
    if result.error is None:
        result.admitted = True
    elif result.error.startswith("HTTP"):
        # HTTP 4xx/5xx = server rejected admission
        result.admitted = False
    elif result.output_tokens > 0:
        # Streaming started → server admitted, regardless of how it ended
        # (timeout, ClientError, partial stream, etc.)
        result.admitted = True
    else:
        # Transport error before any data → not admitted
        result.admitted = False

    # Mark success based on finish_reason.
    if finish_reason in {"stop", "length"}:
        result.success = True
    elif finish_reason in {"abort", "error"}:
        result.success = False
        result.error = finish_reason
    else:
        # No finish_reason = not a clean completion.
        result.success = False
        if result.error is None:
            if result.output_tokens > 0:
                result.error = "incomplete_stream"
            else:
                result.error = "empty_stream"


# ===================================================================
# Request sending (async)
# ===================================================================

async def _send_single_request(
    session: aiohttp.ClientSession,
    spec: RequestSpec,
    port: int,
    timeout_sec: float,
    experiment_start: float,
    model_name: str,
    force_output_len: bool = False,
) -> RequestResult:
    """Send one request and collect streaming metrics."""
    result = RequestResult(
        request_id=spec.request_id,
        arrival_time=spec.arrival_time,
        prompt_len=spec.prompt_len,
        expected_output_len=spec.expected_output_len,
        ttft_slo_ms=spec.ttft_slo_ms,
        tpot_slo_ms=spec.tpot_slo_ms,
        failure_gap_slo_ms=spec.failure_gap_slo_ms,
        dataset=getattr(spec, "dataset", ""),
    )

    # Wait until arrival time.
    now = time.time()
    target = experiment_start + spec.arrival_time
    if target > now:
        await asyncio.sleep(target - now)

    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
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
    }
    # Force output length: prevent model from stopping early at EOS
    if force_output_len and spec.expected_output_len > 0:
        payload["min_tokens"] = spec.expected_output_len

    result.send_time = time.time()
    token_times: list[float] = []
    streamed_tokens = 0
    completion_tokens: int | None = None
    finish_reason: str | None = None

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}"
                result.end_time = time.time()
                result.admitted = False
                return result

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract server request ID.
                if result.server_request_id is None:
                    result.server_request_id = data.get("id")

                # Extract completion token count from usage.
                usage = data.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                # token_ids is at choice level, not in delta.
                token_ids = choices[0].get("token_ids") or []
                if token_ids:
                    now_tok = time.time()
                    token_times.append(now_tok)
                    streamed_tokens += len(token_ids)

                # finish_reason is at choice level, not in delta.
                finish_reason = choices[0].get("finish_reason") or finish_reason

    except asyncio.TimeoutError:
        result.error = "timeout"
    except aiohttp.ClientError as e:
        result.error = str(e)
    except Exception as e:
        result.error = str(e)

    result.end_time = time.time()
    # Use completion_tokens from usage if available; fall back to streamed token count.
    result.output_tokens = max(streamed_tokens, completion_tokens or 0)
    result.token_timestamps = token_times

    if token_times:
        result.first_token_time = token_times[0]
        result.last_token_time = token_times[-1]
        result.ttft_ms = (token_times[0] - result.send_time) * 1000
        result.e2e_ms = (result.end_time - result.send_time) * 1000
        if result.output_tokens > 1:
            result.tpot_ms = (
                (token_times[-1] - token_times[0]) * 1000
                / (result.output_tokens - 1)
            )

        # Max gap between token arrivals.
        max_gap = 0.0
        for i in range(1, len(token_times)):
            gap = (token_times[i] - token_times[i - 1]) * 1000
            if gap > max_gap:
                max_gap = gap
        result.max_gap_ms = max_gap

    _classify_result(result, finish_reason)

    # SLO violations.
    if result.ttft_ms is not None and spec.ttft_slo_ms > 0:
        result.ttft_violated = result.ttft_ms > spec.ttft_slo_ms
    if result.tpot_ms is not None and spec.tpot_slo_ms > 0:
        result.tpot_violated = result.tpot_ms > spec.tpot_slo_ms
    # gap_violated is deferred: set by _enrich_results after affected_by_failure is known.
    # Only failure-affected requests should have gap_violated=True.

    return result


async def _send_requests(
    trace: list[RequestSpec],
    port: int,
    timeout_sec: float,
    experiment_start: float,
    model_name: str,
    force_output_len: bool = False,
) -> list[RequestResult]:
    """Send all requests concurrently, respecting arrival times."""
    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _send_single_request(
                session, spec, port, timeout_sec, experiment_start,
                model_name, force_output_len=force_output_len,
            )
            for spec in trace
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to failed results.
    final: list[RequestResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            final.append(RequestResult(
                request_id=trace[i].request_id,
                arrival_time=trace[i].arrival_time,
                prompt_len=trace[i].prompt_len,
                expected_output_len=trace[i].expected_output_len,
                error=str(r),
                end_time=time.time(),
            ))
        else:
            final.append(r)
    return final


# ===================================================================
# Output saving
# ===================================================================

_CSV_FIELDS = [
    "request_id", "server_request_id", "arrival_time", "prompt_len",
    "expected_output_len", "send_time", "end_time", "ttft_ms", "e2e_ms",
    "output_tokens", "tpot_ms", "max_gap_ms", "success", "error", "admitted",
    "was_rerouted", "affected_by_failure", "ownership_known",
    "dataset",
    "detection_ms", "restore_time_ms", "tokens_restored", "replay_tokens", "replay_time_ms",
    "initial_gpu", "resumed_gpu", "checkpoint_class",
    "ttft_slo_ms", "tpot_slo_ms", "failure_gap_slo_ms",
    "ttft_violated", "tpot_violated", "gap_violated",
]

_EPOCH_FIELDS = [
    "epoch_id", "timestamp", "num_iterations", "total_solve_time_sec",
    "master_time_sec", "recovery_time_sec", "num_cuts", "num_admitted",
    "num_pending", "num_active", "goodput", "fallback",
]


def _save_results(
    results: list[RequestResult],
    epochs: list[dict],
    recoveries: list[dict],
    metadata: ExperimentMetadata,
    metrics: dict,
    output_dir: str,
) -> None:
    """Save all experiment outputs to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # metrics.json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # run_meta.json
    meta_dict = asdict(metadata)
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)

    # requests.csv
    with open(os.path.join(output_dir, "requests.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_csv_row())

    # epochs.csv
    with open(os.path.join(output_dir, "epochs.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_EPOCH_FIELDS)
        writer.writeheader()
        for ep in epochs:
            writer.writerow({k: ep.get(k, "") for k in _EPOCH_FIELDS})

    # recoveries.json
    with open(os.path.join(output_dir, "recoveries.json"), "w") as f:
        json.dump(recoveries, f, indent=2)

    logger.info("Results saved to %s", output_dir)


# ===================================================================
# Main
# ===================================================================

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _get_gpu_info() -> tuple[str, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,count", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        lines = out.strip().split("\n")
        gpu_type = lines[0].split(",")[0].strip() if lines else "unknown"
        gpu_count = len(lines)
        return gpu_type, gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", 0


def main():
    parser = argparse.ArgumentParser(description="Single FT experiment run (v2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--slo-scale", default=None,
                        help="SLO scale name (e.g. Tight/Moderate/Loose) for E6")
    args = parser.parse_args()

    # Load config.
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve load level (v2: load_levels are {pct, rps} dicts).
    load_spec = config["load_levels"][args.load]
    if isinstance(load_spec, dict):
        load_rps = load_spec.get("rps")
        if load_rps is None:
            logger.error("Load level '%s' has rps=null. Run calibrate.py first.", args.load)
            sys.exit(1)
    else:
        load_rps = float(load_spec)

    fault_time_sec = config["fault_timing"].get(args.fault)
    workload_config = config["workloads"][args.workload]
    slo_config = dict(config.get("slo", {}))  # copy so we can mutate
    run_duration = config.get("run_duration_sec", 300.0)
    warmup_sec = config.get("warmup_sec", 30.0)
    timeout_sec = config.get("request_timeout_sec", 120.0)
    startup_timeout = config.get("server_startup_timeout", 300.0)
    baseline_config = config["baselines"][args.baseline]
    max_model_len = config.get("max_model_len", 4096)

    # Apply SLO scale if specified (E6).
    if args.slo_scale:
        # Find slo_scales in any experiment definition that has them
        for exp_def in config.get("experiments", {}).values():
            scales = exp_def.get("slo_scales", {})
            if args.slo_scale in scales:
                scale = scales[args.slo_scale]
                base_ttft = slo_config.get("ttft_base_ms") or (slo_config.get("ttft_ms", 2000) / 5.0)
                base_gap = slo_config.get("gap_base_ms") or (slo_config.get("failure_gap_ms", 3000) / 3.0)
                slo_config["ttft_ms"] = round(scale["ttft_mult"] * base_ttft, 1)
                slo_config["failure_gap_ms"] = round(scale["gap_mult"] * base_gap, 1)
                logger.info("SLO scale '%s': TTFT=%.1fms, Gap=%.1fms",
                            args.slo_scale, slo_config["ttft_ms"], slo_config["failure_gap_ms"])
                break

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize metadata.
    gpu_type, gpu_count = _get_gpu_info()
    metadata = ExperimentMetadata(
        git_commit=_get_git_commit(),
        model=config["model"],
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        baseline=args.baseline,
        workload=args.workload,
        load_level=args.load,
        load_rps=load_rps,
        fault=args.fault,
        fault_time_sec=fault_time_sec,
        seed=args.seed,
        run_duration_sec=run_duration,
        policy=baseline_config.get("scheduling_policy", "fcfs"),
        failure_gap_slo_ms=slo_config.get("failure_gap_ms", 0.0) or 0.0,
    )

    # Generate workload trace (v2: real datasets, mixed support).
    logger.info("Generating workload trace: %s @ %.1f rps for %.0fs",
                args.workload, load_rps, run_duration)
    trace = generate_trace(
        workload_config=workload_config,
        rps=load_rps,
        duration_sec=run_duration,
        seed=args.seed,
        slo_config=slo_config,
        warmup_sec=warmup_sec,
        all_workload_configs=config.get("workloads"),
        max_model_len=max_model_len,
    )
    logger.info("Generated %d requests", len(trace))

    # Build and start server.
    cmd = _build_server_cmd(config, args.baseline, args.port)
    metadata.server_flags = cmd[2:]  # skip python -m
    logger.info("Starting server: %s", " ".join(cmd))
    server_proc = _start_server(cmd, args.output_dir)

    try:
        if not _wait_for_health(args.port, timeout=startup_timeout):
            logger.error("Server failed to start")
            _stop_server(server_proc)
            sys.exit(1)

        # Send requests with optional fault injection.
        experiment_start = time.time()

        if fault_time_sec is not None and args.fault != "none":
            # Run with fault injection: need to track in-flight requests.
            async def _run_with_fault():
                in_flight: set[str] = set()

                async def send_one(session: aiohttp.ClientSession, spec: RequestSpec) -> tuple[str, RequestResult]:
                    # Schedule at arrival time.
                    sleep_sec = max(0.0, experiment_start + spec.arrival_time - time.time())
                    if sleep_sec > 0:
                        await asyncio.sleep(sleep_sec)

                    in_flight.add(spec.request_id)
                    try:
                        result = await _send_single_request(
                            session, spec, args.port, timeout_sec, experiment_start,
                            config["model"],
                            force_output_len=config.get("force_output_len", False),
                        )
                        return spec.request_id, result
                    finally:
                        in_flight.discard(spec.request_id)

                async def inject_fault_delayed():
                    nonlocal metadata
                    await asyncio.sleep(fault_time_sec)
                    metadata.in_flight_at_fault = list(in_flight)
                    metadata.fault_injection_time = time.time()

                    # Kill first engine process.
                    engines = _find_engine_pids(server_proc.pid)
                    if engines:
                        killed_idx, killed_pid = engines[0]
                        try:
                            os.kill(killed_pid, signal.SIGKILL)
                            metadata.failed_gpu_id = killed_idx
                            logger.info("Fault injected: engine=%d, pid=%d, in_flight=%d",
                                      killed_idx, killed_pid, len(metadata.in_flight_at_fault))
                        except OSError:
                            pass

                # Use a single session for all requests (same as no-fault path).
                connector = aiohttp.TCPConnector(limit=200)
                async with aiohttp.ClientSession(connector=connector) as session:
                    # Run requests and fault injection concurrently.
                    send_tasks = [
                        asyncio.create_task(send_one(session, spec)) for spec in trace
                    ]
                    fault_task = asyncio.create_task(inject_fault_delayed())

                    results_by_id = {}
                    for spec, task in zip(trace, send_tasks, strict=True):
                        try:
                            rid, result = await task
                            results_by_id[spec.request_id] = result
                        except Exception as e:
                            results_by_id[spec.request_id] = RequestResult(
                                request_id=spec.request_id,
                                arrival_time=spec.arrival_time,
                                prompt_len=spec.prompt_len,
                                expected_output_len=spec.expected_output_len,
                                error=str(e),
                                end_time=time.time(),
                            )

                    await fault_task
                    return [results_by_id[spec.request_id] for spec in trace]

            request_results = asyncio.run(_run_with_fault())
        else:
            # No fault: just send requests.
            async def _send_no_fault():
                connector = aiohttp.TCPConnector(limit=200)
                async with aiohttp.ClientSession(connector=connector) as session:
                    _force = config.get("force_output_len", False)
                    send_tasks = [
                        _send_single_request(
                            session, spec, args.port, timeout_sec,
                            experiment_start, config["model"],
                            force_output_len=_force,
                        )
                        for spec in trace
                    ]
                    results = await asyncio.gather(*send_tasks, return_exceptions=True)

                final = []
                for spec, result in zip(trace, results, strict=True):
                    if isinstance(result, RequestResult):
                        final.append(result)
                    else:
                        final.append(RequestResult(
                            request_id=spec.request_id,
                            arrival_time=spec.arrival_time,
                            prompt_len=spec.prompt_len,
                            expected_output_len=spec.expected_output_len,
                            error=str(result),
                            end_time=time.time(),
                        ))
                return final

            request_results = asyncio.run(_send_no_fault())

        # Calculate actual runtime excluding warmup: from first to last request completion.
        if request_results:
            first_send = min(r.send_time for r in request_results if r.send_time > 0)
            last_end = max(r.end_time for r in request_results if r.end_time is not None)
            actual_runtime = last_end - first_send
        else:
            actual_runtime = time.time() - experiment_start
        metadata.actual_runtime_sec = actual_runtime

    finally:
        _stop_server(server_proc)

    # Parse server log.
    log_path = os.path.join(args.output_dir, "server.log")
    logger.info("Parsing server log: %s", log_path)
    epochs, recoveries, ckpt_classes = LogParser.parse(log_path)
    logger.info("  Parsed %d epochs, %d recovery events, %d checkpoint classes",
                len(epochs), len(recoveries), len(ckpt_classes))

    # Enrich results.
    print(f"  Fault injection time: {metadata.fault_injection_time}")
    print(f"  Failed GPU: {metadata.failed_gpu_id}")
    if metadata.active_requests_at_fault is not None:
        print(f"  Active requests at fault time: {metadata.active_requests_at_fault}")

    _enrich_results(
        request_results, recoveries, metadata.fault_injection_time,
        ckpt_classes=ckpt_classes,
        failed_gpu_id=metadata.failed_gpu_id,
    )

    # Calculate active requests on failed GPU at fault time.
    if metadata.failed_gpu_id is not None and metadata.in_flight_at_fault:
        active_on_failed_gpu = sum(
            1 for r in request_results
            if r.request_id in metadata.in_flight_at_fault
            and r.initial_gpu == metadata.failed_gpu_id
        )
        metadata.active_requests_at_fault = active_on_failed_gpu
        logger.info("Active requests on failed GPU %d at fault time: %d",
                    metadata.failed_gpu_id, active_on_failed_gpu)

    # Compute metrics.
    metrics = compute_metrics(
        request_results, actual_runtime, metadata.fault_injection_time,
        recoveries=recoveries, epochs=epochs,
    )

    # Print summary.
    print(f"\n--- Results ---")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Completed: {metrics['completed']}")
    print(f"  Completion rate: {metrics['completion_rate']:.1%}")
    print(f"  Goodput: {metrics['goodput']:.1f} tok/s")
    print(f"  TTFT p95: {metrics['ttft_p95_ms']:.1f}ms")
    print(f"  TPOT p95: {metrics['tpot_p95_ms']:.1f}ms")
    print(f"  SLO violation rate: {metrics['slo_violation_rate']:.1%}")

    # Save.
    _save_results(request_results, epochs, recoveries, metadata, metrics,
                  args.output_dir)

    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
