# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Failure Detector for Fault-Tolerant Multi-GPU Serving.

Monitors the health of GPU replicas and emits failure events when a
replica becomes unavailable. Builds on vLLM's existing worker monitoring
(multiproc_executor's sentinel-based detection) and adds replica-level
health tracking with configurable detection parameters.
"""

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


class ReplicaStatus(enum.Enum):
    """Health status of a GPU replica."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Intermittent issues detected.
    FAILED = "failed"  # Confirmed failure.
    RECOVERING = "recovering"  # Failover in progress.


@dataclass
class ReplicaHealth:
    """Health state of a single GPU replica."""

    replica_id: int
    status: ReplicaStatus = ReplicaStatus.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    failure_time: float | None = None
    consecutive_failures: int = 0
    total_failures: int = 0


# Type for failure event callbacks.
FailureEventCallback = Callable[[int], None]  # (replica_id) -> None


class FailureDetector:
    """Monitors GPU replica health and emits failure events.

    This component sits in the Control Plane and:
    1. Receives heartbeats from each replica.
    2. Detects when a replica has stopped responding (timeout-based).
    3. Notifies registered callbacks (Recovery Manager, Scheduler) of failures.

    The detection time T^{det} is a key parameter in the failover-gap
    constraint from the paper.
    """

    def __init__(
        self,
        heartbeat_interval_sec: float = 1.0,
        failure_timeout_sec: float = 5.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        """
        Args:
            heartbeat_interval_sec: Expected interval between heartbeats.
            failure_timeout_sec: Time without heartbeat before declaring failure.
                This corresponds to T^{det} in the paper.
            max_consecutive_failures: Number of missed heartbeats before
                transitioning from DEGRADED to FAILED.
        """
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.failure_timeout_sec = failure_timeout_sec
        self.max_consecutive_failures = max_consecutive_failures

        self._replicas: dict[int, ReplicaHealth] = {}
        self._callbacks: list[FailureEventCallback] = []
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def detection_time_sec(self) -> float:
        """T^{det}: worst-case failure detection time."""
        return self.failure_timeout_sec

    def register_replica(self, replica_id: int) -> None:
        """Register a new replica to monitor."""
        with self._lock:
            self._replicas[replica_id] = ReplicaHealth(replica_id=replica_id)
        logger.info("Failure detector: registered replica %d", replica_id)

    def unregister_replica(self, replica_id: int) -> None:
        """Stop monitoring a replica."""
        with self._lock:
            self._replicas.pop(replica_id, None)

    def register_callback(self, callback: FailureEventCallback) -> None:
        """Register a callback to be called on replica failure."""
        self._callbacks.append(callback)

    def record_heartbeat(self, replica_id: int) -> None:
        """Record a heartbeat from a replica."""
        with self._lock:
            health = self._replicas.get(replica_id)
            if health is None:
                return
            health.last_heartbeat = time.time()
            health.consecutive_failures = 0
            if health.status == ReplicaStatus.DEGRADED:
                health.status = ReplicaStatus.HEALTHY
                logger.info(
                    "Replica %d recovered from degraded state", replica_id
                )

    def report_failure(self, replica_id: int) -> None:
        """Immediately report a replica as failed (e.g., from executor).

        This is the fast path — called when the executor detects a worker
        process death via its sentinel monitoring, bypassing timeout-based
        detection.
        """
        with self._lock:
            health = self._replicas.get(replica_id)
            if health is None:
                return
            if health.status == ReplicaStatus.FAILED:
                return  # Already reported.
            health.status = ReplicaStatus.FAILED
            health.failure_time = time.time()
            health.total_failures += 1

        logger.error(
            "Replica %d reported as FAILED (direct report)", replica_id
        )
        self._notify_failure(replica_id)

    def mark_remote_failed(self, replica_id: int) -> None:
        """Mark a remote replica as failed WITHOUT triggering callbacks.

        Used when the coordinator notifies this engine that a remote replica
        died. We want to update the health status (so the solver excludes it)
        but NOT trigger local RecoveryManager — recovery for remote requests
        is handled by the failed replica's own engine or the coordinator.
        """
        with self._lock:
            health = self._replicas.get(replica_id)
            if health is None:
                return
            if health.status == ReplicaStatus.FAILED:
                return
            health.status = ReplicaStatus.FAILED
            health.failure_time = time.time()
            health.total_failures += 1

        logger.info(
            "Replica %d marked as FAILED (remote notification, "
            "no local recovery triggered)",
            replica_id,
        )

    def mark_recovering(self, replica_id: int) -> None:
        """Mark a replica as currently undergoing failover recovery."""
        with self._lock:
            health = self._replicas.get(replica_id)
            if health is not None:
                health.status = ReplicaStatus.RECOVERING

    def mark_healthy(self, replica_id: int) -> None:
        """Mark a replica as healthy again (e.g., after replacement)."""
        with self._lock:
            health = self._replicas.get(replica_id)
            if health is not None:
                health.status = ReplicaStatus.HEALTHY
                health.last_heartbeat = time.time()
                health.consecutive_failures = 0

    def get_status(self, replica_id: int) -> ReplicaStatus | None:
        """Get the current status of a replica."""
        with self._lock:
            health = self._replicas.get(replica_id)
            return health.status if health is not None else None

    def get_healthy_replicas(self) -> list[int]:
        """Return IDs of all healthy replicas."""
        with self._lock:
            return [
                rid
                for rid, h in self._replicas.items()
                if h.status == ReplicaStatus.HEALTHY
            ]

    def get_failed_replicas(self) -> list[int]:
        """Return IDs of all failed replicas."""
        with self._lock:
            return [
                rid
                for rid, h in self._replicas.items()
                if h.status == ReplicaStatus.FAILED
            ]

    def get_all_statuses(self) -> dict[int, ReplicaStatus]:
        """Return status of all replicas."""
        with self._lock:
            return {rid: h.status for rid, h in self._replicas.items()}

    def start_monitoring(self) -> None:
        """Start the background heartbeat monitoring thread."""
        if self._monitor_thread is not None:
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="failure-detector",
        )
        self._monitor_thread.start()
        logger.info(
            "Failure detector started (timeout=%.1fs, interval=%.1fs)",
            self.failure_timeout_sec,
            self.heartbeat_interval_sec,
        )

    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        """Background thread that checks for missed heartbeats."""
        while not self._stop_event.is_set():
            self._check_heartbeats()
            self._stop_event.wait(timeout=self.heartbeat_interval_sec)

    def _check_heartbeats(self) -> None:
        """Check all replicas for missed heartbeats.

        Uses elapsed time since last heartbeat to determine status:
        - elapsed > failure_timeout_sec → FAILED (immediate)
        - elapsed > heartbeat_interval_sec * 2 → DEGRADED (early warning)
        - otherwise → remains at current status

        The previous implementation incorrectly incremented a counter on
        every check after timeout, causing the actual failure declaration
        time to depend on the check interval rather than the configured
        timeout.  Now we use pure timeout-based detection: once
        failure_timeout_sec elapses without a heartbeat, the replica is
        immediately declared FAILED.
        """
        now = time.time()
        newly_failed: list[int] = []

        with self._lock:
            for replica_id, health in self._replicas.items():
                if health.status in (
                    ReplicaStatus.FAILED,
                    ReplicaStatus.RECOVERING,
                ):
                    continue

                elapsed = now - health.last_heartbeat

                if elapsed > self.failure_timeout_sec:
                    # Timeout exceeded — declare FAILED immediately.
                    health.status = ReplicaStatus.FAILED
                    health.failure_time = now
                    health.total_failures += 1
                    health.consecutive_failures = int(
                        elapsed / self.heartbeat_interval_sec
                    )
                    newly_failed.append(replica_id)
                    logger.error(
                        "Replica %d declared FAILED "
                        "(no heartbeat for %.1fs, timeout=%.1fs)",
                        replica_id,
                        elapsed,
                        self.failure_timeout_sec,
                    )
                elif elapsed > self.heartbeat_interval_sec * 2:
                    # Missed multiple heartbeats — early warning.
                    health.consecutive_failures = int(
                        elapsed / self.heartbeat_interval_sec
                    )
                    if health.status != ReplicaStatus.DEGRADED:
                        health.status = ReplicaStatus.DEGRADED
                        logger.warning(
                            "Replica %d DEGRADED "
                            "(no heartbeat for %.1fs, missed ~%d intervals)",
                            replica_id,
                            elapsed,
                            health.consecutive_failures,
                        )

        for replica_id in newly_failed:
            self._notify_failure(replica_id)

    def _notify_failure(self, replica_id: int) -> None:
        """Notify all registered callbacks about a replica failure."""
        for callback in self._callbacks:
            try:
                callback(replica_id)
            except Exception:
                logger.exception(
                    "Error in failure callback for replica %d", replica_id
                )
