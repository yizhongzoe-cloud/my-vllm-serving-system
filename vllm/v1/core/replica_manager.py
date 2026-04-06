# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Replica Manager for Fault-Tolerant Multi-GPU Serving.

Manages multiple GPU replicas that each load the same model. Handles:
- Replica registration and lifecycle.
- Load tracking per replica (prefill and decode capacity).
- Initial routing decisions (x_{j,r} in the paper).
- Capacity accounting under failure scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.failure_detector import ReplicaStatus
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReplicaInfo:
    """Metadata and runtime state for a single GPU replica."""

    replica_id: int
    gpu_id: int  # Physical GPU device index.

    # Throughput parameters (tokens/sec) — profiled or configured.
    prefill_throughput: float = 0.0  # C_r^{pre}
    decode_throughput: float = 0.0  # C_r^{dec}
    load_bandwidth: float = 0.0  # B_r^{ld} (host→GPU bytes/sec)
    replay_throughput: float = 0.0  # C_r^{rep} (replay tokens/sec)

    # Capacity limits.
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048

    # Runtime load tracking.
    num_active_requests: int = 0
    active_prefill_tokens: int = 0
    active_decode_tokens: int = 0
    assigned_request_ids: set[str] = field(default_factory=set)

    # Status (mirrors FailureDetector but kept here for convenience).
    status: ReplicaStatus = ReplicaStatus.HEALTHY


class ReplicaManager:
    """Manages the set of GPU replicas and routing decisions.

    This component sits in the Control Plane and coordinates with the
    FailureDetector and RequestPool.
    """

    def __init__(
        self,
        planning_horizon: float = 10.0,
        dp_size: int = 1,
    ) -> None:
        """
        Args:
            planning_horizon: H in the paper. Time horizon (seconds) for
                capacity accounting.
            dp_size: Total number of replicas across all EngineCore
                instances.  Each EngineCore only registers its own
                replica locally, but capacity-under-failure checks
                need the global count.
        """
        self.planning_horizon = planning_horizon
        self.dp_size = dp_size
        self._replicas: dict[int, ReplicaInfo] = {}

    def add_replica(
        self,
        replica_id: int,
        gpu_id: int,
        prefill_throughput: float = 0.0,
        decode_throughput: float = 0.0,
        load_bandwidth: float = 0.0,
        max_num_seqs: int = 128,
        max_num_batched_tokens: int = 2048,
    ) -> ReplicaInfo:
        """Register a new GPU replica."""
        info = ReplicaInfo(
            replica_id=replica_id,
            gpu_id=gpu_id,
            prefill_throughput=prefill_throughput,
            decode_throughput=decode_throughput,
            load_bandwidth=load_bandwidth,
            replay_throughput=prefill_throughput,  # Replay ≈ prefill.
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self._replicas[replica_id] = info
        # Keep the effective replica count at least as large as the number of
        # locally registered replicas. This preserves the A-route semantics
        # (where dp_size is passed in explicitly) while keeping unit tests and
        # single-process setups that register multiple replicas locally
        # working without requiring an explicit dp_size override.
        self.dp_size = max(self.dp_size, len(self._replicas))
        logger.info(
            "Registered replica %d on GPU %d "
            "(prefill=%.0f tok/s, decode=%.0f tok/s)",
            replica_id,
            gpu_id,
            prefill_throughput,
            decode_throughput,
        )
        return info

    def remove_replica(self, replica_id: int) -> None:
        """Unregister a replica."""
        self._replicas.pop(replica_id, None)

    def get_replica(self, replica_id: int) -> ReplicaInfo | None:
        return self._replicas.get(replica_id)

    def get_all_replicas(self) -> list[ReplicaInfo]:
        return list(self._replicas.values())

    def get_healthy_replicas(self) -> list[ReplicaInfo]:
        return [
            r for r in self._replicas.values()
            if r.status == ReplicaStatus.HEALTHY
        ]

    def mark_failed(self, replica_id: int) -> None:
        """Mark a replica as failed."""
        info = self._replicas.get(replica_id)
        if info is not None:
            info.status = ReplicaStatus.FAILED
            logger.warning("Replica %d marked as FAILED", replica_id)

    def notify_remote_replica_failed(self) -> None:
        """Decrement dp_size when a remote replica fails.

        Called by EngineCore when it receives REPLICA_FAILED from the
        coordinator.  This keeps capacity-under-failure checks aligned
        with the actual number of surviving replicas.
        """
        if self.dp_size > 1:
            self.dp_size -= 1
            logger.warning(
                "ReplicaManager: dp_size decremented to %d after "
                "remote replica failure",
                self.dp_size,
            )

    def mark_healthy(self, replica_id: int) -> None:
        """Mark a replica as healthy."""
        info = self._replicas.get(replica_id)
        if info is not None:
            info.status = ReplicaStatus.HEALTHY

    # ---- Routing decisions ----

    def route_request(self, request: Request) -> int | None:
        """Choose the best replica for a new request (initial routing).

        Implements the x_{j,r} decision. Uses least-loaded routing by default.

        Args:
            request: The request to route.

        Returns:
            replica_id of the chosen replica, or None if no capacity.
        """
        healthy = self.get_healthy_replicas()
        if not healthy:
            return None

        # Least-loaded by active request count.
        best = min(healthy, key=lambda r: r.num_active_requests)

        # Check capacity.
        if best.num_active_requests >= best.max_num_seqs:
            return None

        return best.replica_id

    def route_request_for_failover(
        self,
        request: Request,
        exclude_replica_ids: set[int] | None = None,
    ) -> int | None:
        """Choose a replica for a displaced request (failover re-routing).

        Implements the x̃_{j,r}(ω) decision. Excludes the failed replica(s).

        Args:
            request: The displaced request.
            exclude_replica_ids: Replicas to exclude (failed ones).

        Returns:
            replica_id of the chosen target, or None if no capacity.
        """
        exclude = exclude_replica_ids or set()
        candidates = [
            r
            for r in self._replicas.values()
            if r.status == ReplicaStatus.HEALTHY and r.replica_id not in exclude
        ]
        if not candidates:
            return None

        # Prefer the replica with most remaining capacity.
        best = min(candidates, key=lambda r: r.num_active_requests)
        if best.num_active_requests >= best.max_num_seqs:
            return None

        return best.replica_id

    def assign_request(self, request: Request, replica_id: int) -> None:
        """Record that a request has been assigned to a replica."""
        info = self._replicas.get(replica_id)
        if info is None:
            return
        info.num_active_requests += 1
        info.active_prefill_tokens += request.prompt_len
        info.active_decode_tokens += request.generation_len
        info.assigned_request_ids.add(request.request_id)
        request.assigned_replica_id = replica_id

    def release_request(self, request: Request) -> None:
        """Record that a request has finished on its assigned replica."""
        replica_id = request.assigned_replica_id
        if replica_id is None:
            return
        info = self._replicas.get(replica_id)
        if info is None:
            return
        info.num_active_requests = max(0, info.num_active_requests - 1)
        info.active_prefill_tokens = max(
            0, info.active_prefill_tokens - request.prompt_len
        )
        info.active_decode_tokens = max(
            0, info.active_decode_tokens - request.generation_len
        )
        info.assigned_request_ids.discard(request.request_id)

    # ---- Capacity accounting ----

    def check_capacity_under_failures(
        self,
        requests: list[Request],
        max_failures: int,
        decode_capacity_model: "DecodeCapacityModel | None" = None,
    ) -> bool:
        """Check if the system can handle all requests even with up to
        max_failures GPU failures (robust feasibility check).

        This is the Ω_k constraint from the paper: for any failure
        scenario with |ω| ≤ k, the surviving replicas must have enough
        capacity for all admitted requests.

        Args:
            requests: Currently admitted requests (including the candidate).
            max_failures: k — maximum number of simultaneous GPU failures.
            decode_capacity_model: If provided, use decode-first capacity
                model (slot-based) instead of serial-time model.

        Returns:
            True if feasible under worst-case failure.
        """
        all_replicas = self.get_healthy_replicas()
        num_replicas = self.dp_size

        if max_failures >= num_replicas:
            return False

        surviving = num_replicas - max_failures
        total_requests = len(requests)

        # 1. Sequence count constraint.
        max_seqs_per_replica = min(
            r.max_num_seqs for r in all_replicas
        ) if all_replicas else 0
        if total_requests > surviving * max_seqs_per_replica:
            return False

        # Decode-first capacity model: use decode slots + residual prefill.
        if (decode_capacity_model is not None
                and not decode_capacity_model.is_fallback):
            # Decode slot check: total requests ≤ surviving * Cap_dec
            cap_dec = decode_capacity_model.decode_capacity()
            if total_requests > surviving * cap_dec:
                return False

            # Residual prefill check: pending prefill tokens fit?
            # Worst case: all requests land on one survivor after failures.
            # Only count requests that still need prefill (not yet started).
            total_prefill = sum(
                r.prompt_len for r in requests
                if r.num_computed_tokens == 0
            )
            # Use conservative estimate: max load per survivor
            per_survivor_requests = (total_requests + surviving - 1) // surviving
            rem_prefill_cap = decode_capacity_model.residual_prefill_capacity(
                per_survivor_requests
            )
            if total_prefill > surviving * rem_prefill_cap:
                return False

            return True

        # Legacy: serial-time capacity model.
        # 2. Prefill token capacity: ∑P_j ≤ surviving * C_r^{pre} * H
        total_prefill_tokens = sum(r.prompt_len for r in requests)
        min_prefill_throughput = min(
            (r.prefill_throughput for r in all_replicas),
            default=0.0,
        )
        if min_prefill_throughput > 0:
            prefill_capacity = surviving * min_prefill_throughput * self.planning_horizon
            if total_prefill_tokens > prefill_capacity:
                return False
        else:
            max_batched = min(
                r.max_num_batched_tokens for r in all_replicas
            ) if all_replicas else 0
            if total_prefill_tokens > surviving * max_batched * self.planning_horizon:
                return False

        # 3. Decode token capacity: ∑G_j ≤ surviving * C_r^{dec} * H
        total_decode_tokens = sum(r.generation_len for r in requests)
        min_decode_throughput = min(
            (r.decode_throughput for r in all_replicas),
            default=0.0,
        )
        if min_decode_throughput > 0:
            decode_capacity = surviving * min_decode_throughput * self.planning_horizon
            if total_decode_tokens > decode_capacity:
                return False
        else:
            max_batched = min(
                r.max_num_batched_tokens for r in all_replicas
            ) if all_replicas else 0
            if total_decode_tokens > surviving * max_batched * self.planning_horizon:
                return False

        return True

    def check_slo_under_failures(
        self,
        request: Request,
        max_failures: int,
    ) -> bool:
        """Check if a request's TTFT/TPOT SLOs can be met even after
        failover to a surviving replica (worst-case scenario).

        Paper §8.5: ∑_r x̃_{j,r}(ω) * P_j / C_r^{pre} ≤ D_j^{ttft}
        for all ω ∈ Ω_k.

        In the worst case, the request lands on the slowest surviving
        replica (minimum throughput).
        """
        all_replicas = self.get_healthy_replicas()
        num_replicas = len(all_replicas)

        if max_failures >= num_replicas:
            return False

        # Worst-case: the k fastest replicas fail, leaving the slowest.
        # For identical replicas this is the same as any surviving set.
        # For heterogeneous replicas, take the minimum throughput among
        # the (num_replicas - max_failures) slowest replicas.

        # Check TTFT SLO under worst-case failover.
        if request.ttft_slo_ms is not None:
            prefill_throughputs = sorted(
                r.prefill_throughput for r in all_replicas
            )
            # After losing the k best, the worst surviving replica has
            # throughput = prefill_throughputs[0] (sorted ascending).
            worst_prefill = prefill_throughputs[0] if prefill_throughputs else 0
            if worst_prefill > 0:
                # Also account for queuing delay: in the worst case, all
                # requests are concentrated on surviving replicas.
                worst_ttft_ms = (request.prompt_len / worst_prefill) * 1000
                if worst_ttft_ms > request.ttft_slo_ms:
                    return False

        # Check TPOT SLO under worst-case failover.
        if request.tpot_slo_ms is not None:
            decode_throughputs = sorted(
                r.decode_throughput for r in all_replicas
            )
            worst_decode = decode_throughputs[0] if decode_throughputs else 0
            if worst_decode > 0:
                worst_tpot_ms = (1.0 / worst_decode) * 1000
                if worst_tpot_ms > request.tpot_slo_ms:
                    return False

        return True

    def get_replica_load_fraction(self, replica_id: int) -> float:
        """Get how loaded a replica is (0.0 = empty, 1.0 = full)."""
        info = self._replicas.get(replica_id)
        if info is None or info.max_num_seqs == 0:
            return 1.0
        return info.num_active_requests / info.max_num_seqs
