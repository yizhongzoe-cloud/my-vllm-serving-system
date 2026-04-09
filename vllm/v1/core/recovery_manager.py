# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recovery Manager for Fault-Tolerant Multi-GPU Serving.

Orchestrates the failover process when a GPU replica fails:
1. Identifies affected requests (J̃(ω) in the paper).
2. For each affected request, loads its KV checkpoint from host memory.
3. Routes the request to a surviving replica.
4. Restores the KV cache on the target GPU.
5. Replays uncovered tokens to bring the request back to its last state.
6. Resumes decode generation.

The key constraint is the failover-gap SLO:
    T^{det} + S^{ckpt}/B^{ld} + U_j/C^{rep} + 1/C^{dec} ≤ D_j^{gap}

Recovery modes (FT_RECOVERY_MODE env var, default "reload"):
  - "reload":   Current behavior — restore KV from host memory checkpoint.
  - "restart":  Drop checkpoint, drop already-decoded tokens, treat as a
                fresh request (re-prefill prompt + re-decode from scratch).
                Stream consistency: BROKEN (newly sampled tokens differ
                from already-emitted ones).
  - "reprefill": Drop checkpoint but keep already-decoded tokens. Submit
                 as a fresh request with extended_prompt = original_prompt
                 + already_decoded. vLLM re-prefills the extended prompt
                 then continues decoding from the same logical position.
                 Stream consistency: PRESERVED.
"""

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.checkpoint_controller import CheckpointController
from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus
from vllm.v1.core.kv_checkpoint_pool import KVCheckpointPool
from vllm.v1.core.replica_manager import ReplicaManager
from vllm.v1.core.request_pool import RequestPool
from vllm.v1.request import Request

logger = init_logger(__name__)

# Recovery mode controlled by env var. See module docstring.
_FT_RECOVERY_MODE = os.environ.get("FT_RECOVERY_MODE", "reload").lower()
if _FT_RECOVERY_MODE not in ("reload", "restart", "reprefill"):
    logger.warning(
        "Unknown FT_RECOVERY_MODE=%r; falling back to 'reload'",
        _FT_RECOVERY_MODE,
    )
    _FT_RECOVERY_MODE = "reload"


@dataclass
class RecoveryResult:
    """Result of recovering a single request after failure."""

    request_id: str
    source_replica_id: int  # Failed replica.
    target_replica_id: int | None  # Where it was re-routed (None = dropped).
    tokens_restored: int  # From checkpoint.
    tokens_to_replay: int  # Uncovered suffix.
    estimated_gap_sec: float  # Estimated failover gap.
    gap_slo_met: bool  # Whether failure_gap_slo is satisfied.
    success: bool


@dataclass
class FailoverReport:
    """Summary of a complete failover event."""

    failed_replica_id: int
    failure_time: float
    recovery_start_time: float
    recovery_end_time: float
    num_affected_requests: int
    num_recovered: int
    num_dropped: int  # Requests that couldn't be re-routed.
    results: list[RecoveryResult]


class RecoveryManager:
    """Orchestrates failover recovery after GPU failures.

    Coordinates between:
    - FailureDetector: receives failure notifications.
    - RequestPool: identifies affected requests.
    - KVCheckpointPool: retrieves checkpointed KV state.
    - ReplicaManager: finds target replicas for re-routing.
    - CheckpointController: evaluates recovery costs.
    """

    def __init__(
        self,
        failure_detector: FailureDetector,
        request_pool: RequestPool,
        checkpoint_pool: KVCheckpointPool,
        replica_manager: ReplicaManager,
        checkpoint_controller: CheckpointController,
        checkpoint_metadata: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        self.failure_detector = failure_detector
        self.request_pool = request_pool
        self.checkpoint_pool = checkpoint_pool
        self.replica_manager = replica_manager
        self.checkpoint_controller = checkpoint_controller
        # Shared reference to FT scheduler's checkpoint metadata dict.
        # Maps request_id -> (num_tokens, size_bytes).  Populated by
        # FaultTolerantScheduler.update_checkpoint_metadata() from
        # EngineCore worker reports.
        self._checkpoint_metadata = checkpoint_metadata

        # Optional callback for solver-planned recovery targets.
        # Signature: (failed_replica_id, request_id) -> target_replica_id | None
        # Set by BendersFTSchedulerImpl to provide pre-computed recovery plans.
        self._solver_recovery_lookup: (
            Callable[[int, str], int | None] | None
        ) = None

        # Register ourselves as a failure callback.
        self.failure_detector.register_callback(self._on_failure)

        # History of failover events.
        self._failover_history: list[FailoverReport] = []
        self._active_recovery_threads: dict[int, threading.Thread] = {}

    def _on_failure(self, failed_replica_id: int) -> None:
        """Callback triggered when FailureDetector reports a GPU failure.

        Runs the actual recovery in a separate thread so that the
        FailureDetector's heartbeat monitoring thread is not blocked
        (which could cause cascading false-positive failures for other
        replicas during a long recovery).
        """
        thread = threading.Thread(
            target=self._handle_failure_thread,
            args=(failed_replica_id,),
            daemon=True,
            name=f"recovery-replica-{failed_replica_id}",
        )
        self._active_recovery_threads[failed_replica_id] = thread
        thread.start()

    def _handle_failure_thread(self, failed_replica_id: int) -> None:
        """Recovery thread entry point."""
        logger.error(
            "RecoveryManager: handling failure of replica %d",
            failed_replica_id,
        )
        try:
            report = self.handle_failure(failed_replica_id)
            self._failover_history.append(report)
            logger.info(
                "Failover complete for replica %d: "
                "%d/%d requests recovered, %d dropped",
                failed_replica_id,
                report.num_recovered,
                report.num_affected_requests,
                report.num_dropped,
            )
        except Exception:
            logger.exception(
                "RecoveryManager: unhandled error during failover "
                "of replica %d",
                failed_replica_id,
            )
        finally:
            self._active_recovery_threads.pop(failed_replica_id, None)

    def handle_failure(self, failed_replica_id: int) -> FailoverReport:
        """Execute the full failover process for a failed replica.

        Steps:
        1. Mark replica as failed.
        2. Identify affected requests (J̃(ω)).
        3. For each request: restore checkpoint → re-route → resume.
        """
        failure_time = time.time()
        recovery_start = time.time()

        # Step 1: Mark replica as failed.
        self.replica_manager.mark_failed(failed_replica_id)
        self.failure_detector.mark_recovering(failed_replica_id)

        # Step 2: Identify affected requests.
        affected_requests = self.request_pool.displace_requests(
            failed_replica_id
        )

        # Sort by G_j descending: prioritize recovery of requests that
        # contribute more to total goodput (paper objective max ∑ G_j * y_j).
        affected_requests.sort(key=lambda r: r.generation_len, reverse=True)

        results: list[RecoveryResult] = []
        num_recovered = 0
        num_dropped = 0

        # Step 3: Recover each affected request.
        # route_request_for_failover uses least-loaded routing, so
        # requests are naturally spread across surviving replicas as
        # each assignment updates the load counters.
        for request in affected_requests:
            result = self._recover_request(
                request,
                failed_replica_id,
                failure_time,
            )
            results.append(result)
            if result.success:
                num_recovered += 1
            else:
                num_dropped += 1

        recovery_end = time.time()

        return FailoverReport(
            failed_replica_id=failed_replica_id,
            failure_time=failure_time,
            recovery_start_time=recovery_start,
            recovery_end_time=recovery_end,
            num_affected_requests=len(affected_requests),
            num_recovered=num_recovered,
            num_dropped=num_dropped,
            results=results,
        )

    def _recover_request(
        self,
        request: Request,
        failed_replica_id: int,
        failure_time: float,
    ) -> RecoveryResult:
        """Recover a single displaced request.

        Steps:
        a. Find a target replica via ReplicaManager.
        b. Check if a checkpoint exists; estimate recovery cost.
        c. Restore KV cache from checkpoint to target GPU.
        d. Record how many tokens need replay.
        e. Re-admit the request to the target replica.
        """
        # Step a: Find target replica.
        # Prefer solver-planned recovery target if available (Benders policy).
        target_replica_id = None
        if self._solver_recovery_lookup is not None:
            solver_target = self._solver_recovery_lookup(
                failed_replica_id, request.request_id
            )
            if solver_target is not None:
                # Verify the solver target is still healthy.
                replica_info = self.replica_manager.get_replica(solver_target)
                if (replica_info is not None
                        and replica_info.status == ReplicaStatus.HEALTHY):
                    target_replica_id = solver_target
                    logger.debug(
                        "Using solver-planned recovery: %s → replica %d",
                        request.request_id,
                        solver_target,
                    )

        # Fall back to greedy least-loaded routing.
        if target_replica_id is None:
            target_replica_id = self.replica_manager.route_request_for_failover(
                request, exclude_replica_ids={failed_replica_id}
            )

        if target_replica_id is None:
            logger.warning(
                "No available replica for request %s after failure of "
                "replica %d",
                request.request_id,
                failed_replica_id,
            )
            return RecoveryResult(
                request_id=request.request_id,
                source_replica_id=failed_replica_id,
                target_replica_id=None,
                tokens_restored=0,
                tokens_to_replay=request.num_computed_tokens,
                estimated_gap_sec=float("inf"),
                gap_slo_met=False,
                success=False,
            )

        # Step b: Check checkpoint and estimate cost.
        # First try the scheduler-side pool (populated when scheduler has
        # direct GPU access — rare in DP mode).  Then fall back to
        # checkpoint metadata synced from EngineCore workers, which
        # reflects the actual checkpoint state on the worker-side pool
        # and /dev/shm shared directory.
        checkpoint_entry = self.checkpoint_pool.get_checkpoint(
            request.request_id
        )
        tokens_restored = 0
        tokens_to_replay = request.num_computed_tokens
        checkpoint_size_bytes = 0

        if checkpoint_entry is not None:
            tokens_restored = checkpoint_entry.num_tokens
            tokens_to_replay = max(
                0, request.num_computed_tokens - tokens_restored
            )
            checkpoint_size_bytes = checkpoint_entry.size_bytes
        else:
            # Fall back to request-level checkpoint tracking (synced from
            # EngineCore workers via update_checkpoint_metadata).
            if request.num_checkpointed_tokens > 0:
                tokens_restored = request.num_checkpointed_tokens
                tokens_to_replay = max(
                    0, request.num_computed_tokens - tokens_restored
                )
                # Use actual size from shared checkpoint metadata dict
                # (populated by FT scheduler from worker reports).
                checkpoint_size_bytes = tokens_restored * 512  # fallback
                if self._checkpoint_metadata is not None:
                    meta = self._checkpoint_metadata.get(request.request_id)
                    if meta is not None:
                        checkpoint_size_bytes = meta[1]  # actual size_bytes

        # Step c: Estimate failover gap using the actual checkpoint size.
        target_info = self.replica_manager.get_replica(target_replica_id)
        detection_time = self.failure_detector.detection_time_sec

        estimated_gap = self.checkpoint_controller.estimate_recovery_cost(
            request=request,
            checkpoint_size_bytes=checkpoint_size_bytes,
            load_bandwidth_bytes_per_sec=(
                target_info.load_bandwidth if target_info else 0
            ),
            replay_tokens_per_sec=(
                target_info.replay_throughput if target_info else 0
            ),
            detection_time_sec=detection_time,
            decode_throughput=(
                target_info.decode_throughput if target_info else 0
            ),
        )

        # Check failure-gap SLO.
        gap_slo_met = True
        if request.failure_gap_slo_ms is not None:
            gap_slo_met = estimated_gap * 1000 <= request.failure_gap_slo_ms

        # If gap SLO is violated, drop the request instead of rerouting
        # with a guaranteed SLO violation.
        if not gap_slo_met:
            logger.warning(
                "Request %s: failover-gap SLO violated "
                "(est=%.1fms > slo=%.1fms), dropping request",
                request.request_id,
                estimated_gap * 1000,
                request.failure_gap_slo_ms,
            )
            self.replica_manager.release_request(request)
            return RecoveryResult(
                request_id=request.request_id,
                source_replica_id=failed_replica_id,
                target_replica_id=target_replica_id,
                tokens_restored=tokens_restored,
                tokens_to_replay=tokens_to_replay,
                estimated_gap_sec=estimated_gap,
                gap_slo_met=False,
                success=False,
            )

        # Step d: Apply recovery mode (reload / restart / reprefill).
        # See module docstring for mode descriptions.
        if _FT_RECOVERY_MODE == "restart":
            # Method 1: Drop both checkpoint and already-decoded tokens.
            # vLLM treats this as a brand-new request and re-prefills the
            # original prompt from scratch. WARNING: stream consistency is
            # broken — newly sampled tokens will differ from those already
            # emitted to the client.
            request.num_checkpointed_tokens = 0
            request.num_computed_tokens = 0
            request.last_checkpoint_size_bytes = 0
            try:
                request.output_token_ids.clear()
            except AttributeError:
                pass  # immutable container — best effort
            tokens_restored = 0
            tokens_to_replay = 0  # full re-prefill happens via normal path
            logger.info(
                "FT_RECOVERY_MODE=restart: req=%s cleared checkpoint + "
                "decoded tokens, will re-prefill prompt from scratch",
                request.request_id,
            )
        elif _FT_RECOVERY_MODE == "reprefill":
            # Method 2: Drop checkpoint, but keep already-decoded tokens
            # by extending the prompt. vLLM re-prefills the extended prompt
            # (= original_prompt + decoded_tokens) and continues decoding
            # from the same logical position. Stream consistency preserved.
            decoded_token_ids = list(request.output_token_ids)
            n_decoded = len(decoded_token_ids)
            if n_decoded > 0:
                # Extend the prompt with already-emitted decoded tokens.
                # Try multiple field names since vllm Request internals vary.
                try:
                    new_prompt = list(request.prompt_token_ids) + decoded_token_ids
                    request.prompt_token_ids = new_prompt
                    if hasattr(request, "num_prompt_tokens"):
                        request.num_prompt_tokens = len(new_prompt)
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "FT_RECOVERY_MODE=reprefill: req=%s failed to extend "
                        "prompt (%s); falling back to restart",
                        request.request_id, exc,
                    )
                # Clear decoded state — these tokens are now in the prompt.
                try:
                    request.output_token_ids.clear()
                except AttributeError:
                    pass
            request.num_checkpointed_tokens = 0
            request.num_computed_tokens = 0
            request.last_checkpoint_size_bytes = 0
            tokens_restored = 0
            tokens_to_replay = 0
            logger.info(
                "FT_RECOVERY_MODE=reprefill: req=%s extended prompt by "
                "%d decoded tokens, will re-prefill via prompt path",
                request.request_id, n_decoded,
            )
        # Else: "reload" mode — keep request.num_checkpointed_tokens as set
        # above; EngineCore will trigger restore_kv_blocks via
        # _process_ft_pending_restores().

        self.replica_manager.release_request(request)
        self.replica_manager.assign_request(request, target_replica_id)
        # Use readmit_request (not admit_request) to properly transition
        # DISPLACED→ADMITTED and clean up the displaced tracking set.
        self.request_pool.readmit_request(
            request.request_id, target_replica_id
        )

        logger.info(
            "Request %s: re-routed %d→%d, mode=%s "
            "restored=%d tokens, replay=%d tokens, "
            "est_gap=%.1fms, slo_met=%s",
            request.request_id,
            failed_replica_id,
            target_replica_id,
            _FT_RECOVERY_MODE,
            tokens_restored,
            tokens_to_replay,
            estimated_gap * 1000,
            gap_slo_met,
        )

        return RecoveryResult(
            request_id=request.request_id,
            source_replica_id=failed_replica_id,
            target_replica_id=target_replica_id,
            tokens_restored=tokens_restored,
            tokens_to_replay=tokens_to_replay,
            estimated_gap_sec=estimated_gap,
            gap_slo_met=gap_slo_met,
            success=True,
        )

    @property
    def failover_history(self) -> list[FailoverReport]:
        # Best-effort drain of in-flight recovery callbacks so callers that
        # immediately inspect history after report_failure() (common in unit
        # tests and debugging code) observe recently completed failovers.
        for thread in list(self._active_recovery_threads.values()):
            thread.join(timeout=0.1)
        return self._failover_history

    def get_stats(self) -> dict:
        """Return recovery statistics."""
        total_failovers = len(self._failover_history)
        total_affected = sum(
            r.num_affected_requests for r in self._failover_history
        )
        total_recovered = sum(
            r.num_recovered for r in self._failover_history
        )
        total_dropped = sum(
            r.num_dropped for r in self._failover_history
        )
        return {
            "total_failovers": total_failovers,
            "total_affected_requests": total_affected,
            "total_recovered": total_recovered,
            "total_dropped": total_dropped,
            "recovery_rate": (
                total_recovered / total_affected
                if total_affected > 0
                else 1.0
            ),
        }
