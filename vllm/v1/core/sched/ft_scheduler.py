# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fault-tolerant request scheduler.

The scheduler coordinates admission, routing, online checkpointing, and
failover recovery. Checkpoint publication is now a pure runtime policy:
each engine evaluates whether publishing the current unpublished stable
prefix is worthwhile, while the scheduler handles admission/routing and
recovery orchestration.
"""

import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.checkpoint_controller import (
    CheckpointConfig,
    CheckpointController,
)
from vllm.v1.core.failure_detector import FailureDetector
from vllm.v1.core.kv_checkpoint_pool import KVCheckpointPool
from vllm.v1.core.recovery_manager import RecoveryManager
from vllm.v1.core.replica_manager import ReplicaManager
from vllm.v1.core.request_pool import RequestPool
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class FTSchedulerConfig:
    """Configuration for the fault-tolerant scheduler."""

    # Maximum number of simultaneous GPU failures to tolerate (k).
    max_gpu_failures: int = 1
    # Host memory budget for KV checkpoint pool (bytes).
    checkpoint_pool_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB
    # Failure detection time T^{det} (seconds).
    detection_time_sec: float = 0.1
    # Heartbeat monitoring interval (seconds).
    heartbeat_interval_sec: float = 1.0
    # Heartbeat timeout before declaring failure (seconds).
    failure_timeout_sec: float = 5.0
    # Checkpoint policy config.
    checkpoint_config: CheckpointConfig | None = None
    # Whether to enable checkpointing at all.
    enable_checkpointing: bool = True
    # Deprecated legacy fixed checkpoint level (0/1/2).
    fixed_checkpoint_level: int = -1
    # Fixed checkpoint cadence in stable full blocks. 0 = online policy.
    fixed_checkpoint_blocks: int = 0
    # Stable KV block size in tokens.
    block_size: int = 1
    # Replay throughput estimate (tokens/sec) for the online checkpoint rule.
    replay_throughput_tokens_per_sec: float = 0.0
    # Host→GPU load bandwidth (bytes/sec) for restore cost.
    load_bandwidth_bytes_per_sec: float = 0.0
    # GPU→Host checkpoint bandwidth (bytes/sec) for publish cost.
    checkpoint_bandwidth_bytes_per_sec: float = 0.0
    # λ weight on steady-state checkpoint copy overhead.
    checkpoint_lambda: float = 1.0
    # Estimated KV bytes per token used by the controller/cost model.
    kv_bytes_per_token: int = 8192
    # Path to checkpoint_cost_profile.json for profile-driven decisions.
    checkpoint_cost_profile: str = ""


class FaultTolerantScheduler:
    """Fault-tolerant scheduler that coordinates all FT components.

    This is the top-level orchestrator in the Control Plane. It owns
    and coordinates:
    - RequestPool: centralized request tracking.
    - ReplicaManager: multi-GPU replica management.
    - FailureDetector: health monitoring.
    - CheckpointController: adaptive checkpoint decisions.
    - KVCheckpointPool: host-memory checkpoint storage.
    - RecoveryManager: failover orchestration.
    """

    def __init__(
        self,
        config: FTSchedulerConfig | None = None,
        dp_size: int = 1,
    ) -> None:
        self.config = config or FTSchedulerConfig()

        # Initialize all FT components.
        self.request_pool = RequestPool()

        self.replica_manager = ReplicaManager(dp_size=dp_size)

        self.failure_detector = FailureDetector(
            heartbeat_interval_sec=self.config.heartbeat_interval_sec,
            failure_timeout_sec=self.config.failure_timeout_sec,
        )

        self.checkpoint_controller = CheckpointController(
            config=self.config.checkpoint_config,
            fixed_level=self.config.fixed_checkpoint_level,
            fixed_blocks=self.config.fixed_checkpoint_blocks,
            block_size=self.config.block_size,
            replay_throughput_tokens_per_sec=(
                self.config.replay_throughput_tokens_per_sec
            ),
            load_bandwidth_bytes_per_sec=(
                self.config.load_bandwidth_bytes_per_sec
            ),
            checkpoint_bandwidth_bytes_per_sec=(
                self.config.checkpoint_bandwidth_bytes_per_sec
            ),
            checkpoint_lambda=self.config.checkpoint_lambda,
            default_kv_bytes_per_token=self.config.kv_bytes_per_token,
            cost_profile_path=self.config.checkpoint_cost_profile,
        )

        self.checkpoint_pool = KVCheckpointPool(
            max_memory_bytes=self.config.checkpoint_pool_bytes,
        )

        # Lightweight checkpoint metadata synced from EngineCore workers.
        # Maps request_id -> (num_tokens, size_bytes).
        # The actual KV data lives on the worker side (in its own
        # KVCheckpointPool and /dev/shm); this metadata lets the
        # scheduler-side RecoveryManager estimate recovery costs
        # without needing the full KV tensors.
        # Created before RecoveryManager so we can pass a shared reference.
        self._checkpoint_metadata: dict[str, tuple[int, int]] = {}

        self.recovery_manager = RecoveryManager(
            failure_detector=self.failure_detector,
            request_pool=self.request_pool,
            checkpoint_pool=self.checkpoint_pool,
            replica_manager=self.replica_manager,
            checkpoint_controller=self.checkpoint_controller,
            checkpoint_metadata=self._checkpoint_metadata,
        )

        # Cache checkpoint decisions per step to avoid double evaluation.
        self._pending_checkpoint_requests: list[Request] | None = None

    def start(self) -> None:
        """Start the fault-tolerant scheduler and monitoring."""
        self.failure_detector.start_monitoring()
        logger.info(
            "FaultTolerantScheduler started "
            "(max_failures=%d, checkpointing=%s)",
            self.config.max_gpu_failures,
            self.config.enable_checkpointing,
        )

    def stop(self) -> None:
        """Stop the scheduler and clean up."""
        self.failure_detector.stop_monitoring()
        self.checkpoint_pool.clear()

    # ---- Admission control (y_j decision) ----

    def admit_request(self, request: Request) -> bool:
        """Decide whether to admit a new request.

        Implements the admission decision y_j from the paper.
        Objective: maximize ∑ G_j * y_j (greedy heuristic: prefer
        requests with larger expected output for higher goodput).

        Checks:
        1. There is a healthy replica with capacity.
        2. Under worst-case k failures, the request can still be served.
        3. TTFT/TPOT SLOs can be met, including under failover scenarios.
        4. Failover-gap SLO can be met.

        Args:
            request: The incoming request.

        Returns:
            True if admitted, False if rejected.
        """
        # Find a replica to route to.
        replica_id = self.replica_manager.route_request(request)
        if replica_id is None:
            logger.debug(
                "Rejected request %s: no available replica",
                request.request_id,
            )
            return False

        # Check robust feasibility under failures.
        current_requests = self.request_pool.get_admitted_requests() \
            + self.request_pool.get_displaced_requests()
        all_requests = current_requests + [request]

        if not self.replica_manager.check_capacity_under_failures(
            all_requests, self.config.max_gpu_failures
        ):
            logger.debug(
                "Rejected request %s: insufficient capacity under "
                "%d-failure scenario",
                request.request_id,
                self.config.max_gpu_failures,
            )
            return False

        # Check TTFT/TPOT SLO feasibility on the target replica,
        # accounting for queuing delay from co-located requests.
        target = self.replica_manager.get_replica(replica_id)
        if (
            request.ttft_slo_ms is not None
            and target is not None
            and target.prefill_throughput > 0
        ):
            # Account for queuing: existing prefill tokens on this replica
            # must be processed before this request's prefill starts.
            queued_prefill = target.active_prefill_tokens
            estimated_ttft_ms = (
                (queued_prefill + request.prompt_len)
                / target.prefill_throughput
            ) * 1000
            if estimated_ttft_ms > request.ttft_slo_ms:
                logger.debug(
                    "Rejected request %s: TTFT SLO infeasible "
                    "(est=%.1fms > slo=%.1fms, queued=%d tokens)",
                    request.request_id,
                    estimated_ttft_ms,
                    request.ttft_slo_ms,
                    queued_prefill,
                )
                return False

        # Check TPOT SLO feasibility on the target replica.
        if (
            request.tpot_slo_ms is not None
            and target is not None
            and target.decode_throughput > 0
        ):
            estimated_tpot_ms = (1.0 / target.decode_throughput) * 1000
            if estimated_tpot_ms > request.tpot_slo_ms:
                logger.debug(
                    "Rejected request %s: TPOT SLO infeasible "
                    "(est=%.1fms > slo=%.1fms)",
                    request.request_id,
                    estimated_tpot_ms,
                    request.tpot_slo_ms,
                )
                return False

        # Check TTFT/TPOT SLOs under all failure scenarios (§8.5).
        # Ensures SLOs hold even on the worst surviving replica.
        if not self.replica_manager.check_slo_under_failures(
            request, self.config.max_gpu_failures
        ):
            logger.debug(
                "Rejected request %s: TTFT/TPOT SLO infeasible under "
                "%d-failure scenario",
                request.request_id,
                self.config.max_gpu_failures,
            )
            return False

        # Check failover-gap SLO feasibility.
        # At admission, no checkpoint exists yet, so the worst case is
        # replaying the full prompt (P_j tokens). This gives a
        # conservative upper bound on recovery cost.
        if (
            request.failure_gap_slo_ms is not None
            and target is not None
            and (target.replay_throughput > 0 or target.decode_throughput > 0)
        ):
            estimated_gap = self.checkpoint_controller.estimate_recovery_cost(
                request=request,
                checkpoint_size_bytes=0,  # No checkpoint yet at admission.
                load_bandwidth_bytes_per_sec=target.load_bandwidth,
                replay_tokens_per_sec=target.replay_throughput,
                detection_time_sec=self.config.detection_time_sec,
                decode_throughput=target.decode_throughput,
                # At admission, num_computed_tokens=0 so
                # get_uncovered_tokens() returns 0.  The worst case is
                # replaying the full prompt (no checkpoint exists yet).
                replay_tokens_override=request.prompt_len,
            )
            if estimated_gap * 1000 > request.failure_gap_slo_ms:
                logger.debug(
                    "Rejected request %s: failover-gap SLO infeasible "
                    "(est=%.1fms > slo=%.1fms)",
                    request.request_id,
                    estimated_gap * 1000,
                    request.failure_gap_slo_ms,
                )
                return False

        # All checks passed — admit and assign.
        self.request_pool.add_request(request)
        self.request_pool.admit_request(request.request_id, replica_id)
        self.replica_manager.assign_request(request, replica_id)

        # Assign initial coarse checkpoint stage for telemetry/costing.
        level = self.checkpoint_controller.get_checkpoint_level(request)
        request.checkpoint_level = level

        logger.debug(
            "Admitted request %s → replica %d (ckpt_level=%d, G_j=%d)",
            request.request_id,
            replica_id,
            level,
            request.generation_len,
        )
        return True

    def register_admitted_request(self, request: Request) -> None:
        """Register a request that was already admitted by the centralized
        solver. Skips all local capacity/SLO checks — trusts the solver."""
        replica_id = self.replica_manager.route_request(request)
        if replica_id is None:
            # Fallback: pick any healthy replica.
            for r in self.replica_manager.get_healthy_replicas():
                replica_id = r
                break
        if replica_id is None:
            logger.warning(
                "register_admitted_request: no replica for %s, "
                "falling back to admit_request",
                request.request_id,
            )
            self.admit_request(request)
            return

        self.request_pool.add_request(request)
        self.request_pool.admit_request(request.request_id, replica_id)
        self.replica_manager.assign_request(request, replica_id)

        level = self.checkpoint_controller.get_checkpoint_level(request)
        request.checkpoint_level = level

        logger.debug(
            "Registered (centralized) request %s → replica %d "
            "(ckpt_level=%d, G_j=%d)",
            request.request_id,
            replica_id,
            level,
            request.generation_len,
        )

    # ---- Per-step checkpoint decisions ----

    def run_checkpoint_step(
        self,
        running_requests: list[Request],
        gpu_kv_caches: list["torch.Tensor"] | None = None,
        kv_cache_manager: "KVCacheManager | None" = None,
    ) -> list[str]:
        """Called each scheduling step to handle adaptive checkpointing.

        For each running request:
        1. Update its coarse checkpoint stage for telemetry/costing.
        2. Use the runtime controller to decide whether to publish a new
           incremental checkpoint now.

        Caches checkpoint decisions so that get_checkpoint_requests()
        can retrieve them without re-evaluating.

        Args:
            running_requests: Currently running requests.
            gpu_kv_caches: Per-layer GPU KV cache tensors (for actual copy).
            kv_cache_manager: KV cache manager (to get block IDs).

        Returns:
            List of request_ids that were checkpointed this step.
        """
        if not self.config.enable_checkpointing:
            self._pending_checkpoint_requests = None
            return []

        to_checkpoint = self.checkpoint_controller.get_requests_to_checkpoint(
            running_requests
        )

        # Cache for get_checkpoint_requests() to avoid double evaluation.
        self._pending_checkpoint_requests = to_checkpoint

        checkpointed_ids = []
        for request in to_checkpoint:
            # Always update coarse checkpoint stage regardless of
            # whether GPU tensors are available.
            level = self.checkpoint_controller.get_checkpoint_level(request)
            request.checkpoint_level = level

            # Actual GPU→CPU copy requires both the KV cache tensors
            # (on the GPU workers) and the kv_cache_manager (for block IDs).
            if gpu_kv_caches is not None and kv_cache_manager is not None:
                block_ids, covered_tokens = self._get_request_stable_checkpoint_view(
                    request,
                    kv_cache_manager,
                )
                if block_ids:
                    entry = self.checkpoint_pool.save_checkpoint(
                        request_id=request.request_id,
                        gpu_kv_caches=gpu_kv_caches,
                        block_ids=block_ids,
                        num_tokens=covered_tokens,
                        async_copy=True,
                    )
                    if entry is not None:
                        self.checkpoint_controller.record_checkpoint(request)
                        request.num_checkpointed_tokens = covered_tokens
                        request.last_checkpoint_size_bytes = entry.size_bytes
                        self._checkpoint_metadata[request.request_id] = (
                            covered_tokens,
                            entry.size_bytes,
                        )
                        checkpointed_ids.append(request.request_id)

        return checkpointed_ids

    def get_and_clear_cached_checkpoint_requests(self) -> list[Request] | None:
        """Return the cached checkpoint decisions from the last step.

        Returns None if step_checkpoints() hasn't been called yet this step.
        After retrieval, the cache is cleared.
        """
        result = self._pending_checkpoint_requests
        self._pending_checkpoint_requests = None
        return result

    def _get_request_kv_block_ids(
        self,
        request: Request,
        kv_cache_manager: "KVCacheManager",
    ) -> list[int]:
        """Extract block IDs for a request from the KV cache manager."""
        try:
            all_ids = kv_cache_manager.get_block_ids(request.request_id)
            if all_ids:
                return list(all_ids[0])  # First KV cache group.
        except (AttributeError, KeyError, TypeError):
            pass

        # Older/specialized KV managers may only expose raw req_to_blocks.
        try:
            blocks = kv_cache_manager.req_to_blocks.get(request.request_id)
            if blocks:
                return [blk.block_id for blk in blocks]
        except (AttributeError, KeyError, TypeError):
            pass
        return []

    def _get_request_stable_checkpoint_view(
        self,
        request: Request,
        kv_cache_manager: "KVCacheManager",
    ) -> tuple[list[int], int]:
        """Return the stable full-block prefix eligible for fallback save.

        The scheduler-local fallback path must mirror the worker/shared
        incremental checkpoint semantics: only the stable full-block prefix is
        published, and frontier/partial blocks are left for replay.
        """
        stable_full_tokens = max(
            0,
            (request.num_computed_tokens // self.config.block_size)
            * self.config.block_size,
        )
        if stable_full_tokens <= 0:
            return [], 0

        stable_full_blocks = stable_full_tokens // self.config.block_size
        block_ids = self._get_request_kv_block_ids(request, kv_cache_manager)
        if not block_ids:
            return [], 0

        if len(block_ids) < stable_full_blocks:
            logger.warning(
                "Request %s: expected %d stable KV blocks but found only %d; "
                "skipping scheduler-local fallback checkpoint this step",
                request.request_id,
                stable_full_blocks,
                len(block_ids),
            )
            return [], 0

        return block_ids[:stable_full_blocks], stable_full_tokens

    # ---- Checkpoint metadata (synced from EngineCore workers) ----

    def update_checkpoint_metadata(
        self, updates: dict[str, tuple[int, int]]
    ) -> None:
        """Record checkpoint progress reported by EngineCore workers.

        Called by EngineCore._maybe_ft_checkpoint() after successful
        GPU→CPU copies.  This keeps the scheduler-side metadata in sync
        with the actual checkpoint state on the workers, so that
        RecoveryManager can estimate recovery costs accurately.

        Args:
            updates: Maps request_id -> (num_tokens, size_bytes).
                size_bytes is the actual checkpoint size reported by the
                worker's KVCheckpointPool, not an estimate.
        """
        for req_id, (num_tokens, size_bytes) in updates.items():
            self._checkpoint_metadata[req_id] = (num_tokens, size_bytes)
            # Also update the request object if available.
            request = self.request_pool.get_request(req_id)
            if request is not None:
                request.num_checkpointed_tokens = num_tokens
                request.last_checkpoint_size_bytes = size_bytes

    def get_checkpoint_metadata(
        self, request_id: str
    ) -> tuple[int, int] | None:
        """Get checkpoint metadata for a request.

        Returns:
            (num_tokens, estimated_size_bytes) or None if no checkpoint.
        """
        return self._checkpoint_metadata.get(request_id)

    # ---- Request completion ----

    def complete_request(self, request: Request) -> None:
        """Handle request completion: clean up all FT state."""
        self.replica_manager.release_request(request)
        self.request_pool.complete_request(request.request_id)
        self.checkpoint_pool.delete_checkpoint(request.request_id)
        self.checkpoint_controller.remove_request(request.request_id)
        self._checkpoint_metadata.pop(request.request_id, None)

    def abort_request(self, request_id: str) -> None:
        """Handle request abort: clean up all FT state."""
        request = self.request_pool.get_request(request_id)
        if request is not None:
            self.replica_manager.release_request(request)
        self.request_pool.remove_request(request_id)
        self.checkpoint_pool.delete_checkpoint(request_id)
        self.checkpoint_controller.remove_request(request_id)
        self._checkpoint_metadata.pop(request_id, None)

    # ---- Replica management ----

    def register_replica(
        self,
        replica_id: int,
        gpu_id: int,
        prefill_throughput: float = 0.0,
        decode_throughput: float = 0.0,
        load_bandwidth: float = 0.0,
        max_num_seqs: int = 128,
        max_num_batched_tokens: int = 2048,
    ) -> None:
        """Register a new GPU replica with the FT scheduler."""
        self.replica_manager.add_replica(
            replica_id=replica_id,
            gpu_id=gpu_id,
            prefill_throughput=prefill_throughput,
            decode_throughput=decode_throughput,
            load_bandwidth=load_bandwidth,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.failure_detector.register_replica(replica_id)

    # ---- Stats and monitoring ----

    def get_stats(self) -> dict:
        """Return comprehensive FT scheduler statistics."""
        return {
            "request_pool": {
                "pending": self.request_pool.num_pending,
                "admitted": self.request_pool.num_admitted,
                "displaced": self.request_pool.num_displaced,
            },
            "replicas": {
                rid: {
                    "status": info.status.value,
                    "active_requests": info.num_active_requests,
                    "load": self.replica_manager.get_replica_load_fraction(rid),
                }
                for rid, info in self.replica_manager._replicas.items()
            },
            "checkpoint_pool": self.checkpoint_pool.get_stats(),
            "recovery": self.recovery_manager.get_stats(),
        }
