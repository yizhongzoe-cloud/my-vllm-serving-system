# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cost table builder for the Benders-style robust FT solver.

The solver no longer reasons about checkpoint stages/classes. It consumes
only the real published checkpoint state visible in the current snapshot:

    p_j                     — normal-case prefill time demand (seconds)
    d_j                     — normal-case decode time demand (seconds)
    checkpoint_overhead_sec — one-shot online publication cost if the
                              current unpublished stable suffix would publish
    restore_time_sec        — restore time from the published checkpoint state
    replay_time_sec         — replay time for the uncovered failure suffix
    gap_time_sec            — T_det + restore + replay + resume
    run_mem_bytes           — normal-case GPU memory footprint
    recovery_mem_bytes      — post-failure GPU memory footprint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.checkpoint_controller import (
    CheckpointConfig,
    estimate_online_checkpoint_publication,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.checkpoint_cost_model import CheckpointCostModel
    from vllm.v1.engine import EngineCoreRequest, RequestSnapshot


@dataclass
class RequestCosts:
    """Precomputed solver-side costs for a single request."""

    request_id: str
    G_j: int  # expected output length (goodput weight)

    p_j: float  # prefill time (seconds)
    d_j: float  # decode time (seconds)

    checkpoint_overhead_sec: float = 0.0
    restore_time_sec: float = 0.0
    replay_time_sec: float = 0.0
    gap_time_sec: float = 0.0
    run_mem_bytes: float = 0.0
    recovery_mem_bytes: float = 0.0

    # Whether the request is already active (placement fixed in master).
    is_active: bool = False
    # If active, the current replica placement is fixed in the master.
    active_replica_id: int | None = None

    # Prompt length (for recovery work estimation).
    prompt_len: int = 0
    # Number of tokens already computed (for active requests).
    num_computed_tokens: int = 0
    # Number of tokens already checkpointed/published.
    num_checkpointed_tokens: int = 0
    # Actual bytes in the latest published checkpoint, if known.
    checkpoint_size_bytes: int = 0

    # SLO bounds (seconds). None means no constraint.
    ttft_slo_sec: float | None = None  # D_j^{ttft}
    tpot_slo_sec: float | None = None  # D_j^{tpot}
    gap_slo_sec: float | None = None  # D_j^{gap}


class CostTableBuilder:
    """Builds RequestCosts for all requests in the current snapshot."""

    DEFAULT_KV_BYTES_PER_TOKEN: int = 8192

    def __init__(
        self,
        planning_horizon: float,
        prefill_throughput: float,
        decode_throughput: float,
        load_bandwidth: float,
        checkpoint_bandwidth: float | None = None,
        replay_throughput: float = 0.0,
        detection_time_sec: float = 0.0,
        memory_capacity_bytes: int = 0,
        checkpoint_config: CheckpointConfig | None = None,
        kv_bytes_per_token: int = 0,
        block_size: int = 1,
        checkpoint_lambda: float = 1.0,
        cost_model: "CheckpointCostModel | None" = None,
    ) -> None:
        self.planning_horizon = planning_horizon
        self.prefill_throughput = prefill_throughput
        self.decode_throughput = decode_throughput
        self.load_bandwidth = load_bandwidth
        self.checkpoint_bandwidth = checkpoint_bandwidth or load_bandwidth
        self.replay_throughput = replay_throughput
        self.detection_time_sec = detection_time_sec
        self.memory_capacity_bytes = memory_capacity_bytes
        self.ckpt_cfg = checkpoint_config or CheckpointConfig()
        self.kv_bytes_per_token = (
            kv_bytes_per_token
            if kv_bytes_per_token > 0
            else self.DEFAULT_KV_BYTES_PER_TOKEN
        )
        self.block_size = max(1, block_size)
        self.checkpoint_lambda = checkpoint_lambda
        self._cost_model = cost_model

    def compute_request_costs(
        self,
        request: Request,
        is_active: bool = False,
    ) -> RequestCosts:
        return self._compute_costs_raw(
            request_id=request.request_id,
            prompt_len=request.prompt_len,
            generation_len=request.generation_len,
            num_computed_tokens=request.num_computed_tokens,
            num_output_tokens=request.num_output_tokens,
            num_checkpointed_tokens=request.num_checkpointed_tokens,
            checkpoint_size_bytes=request.last_checkpoint_size_bytes,
            is_active=is_active,
            assigned_replica_id=(
                request.assigned_replica_id if is_active else None
            ),
            ttft_slo_ms=request.ttft_slo_ms,
            tpot_slo_ms=request.tpot_slo_ms,
            failure_gap_slo_ms=request.failure_gap_slo_ms,
        )

    def compute_costs_from_snapshot(
        self,
        snapshot: "RequestSnapshot",
        is_active: bool = False,
    ) -> RequestCosts:
        return self._compute_costs_raw(
            request_id=snapshot.request_id,
            prompt_len=snapshot.prompt_len,
            generation_len=snapshot.generation_len,
            num_computed_tokens=snapshot.num_computed_tokens,
            num_output_tokens=snapshot.num_output_tokens,
            num_checkpointed_tokens=snapshot.num_checkpointed_tokens,
            checkpoint_size_bytes=snapshot.checkpoint_size_bytes,
            is_active=is_active,
            assigned_replica_id=(
                snapshot.assigned_replica_id if is_active else None
            ),
            ttft_slo_ms=snapshot.ttft_slo_ms,
            tpot_slo_ms=snapshot.tpot_slo_ms,
            failure_gap_slo_ms=snapshot.failure_gap_slo_ms,
        )

    def compute_costs_from_engine_request(
        self,
        request: "EngineCoreRequest",
    ) -> RequestCosts:
        prompt_len = (
            len(request.prompt_token_ids) if request.prompt_token_ids else 0
        )
        generation_len = request.expected_output_len or 0
        return self._compute_costs_raw(
            request_id=request.request_id,
            prompt_len=prompt_len,
            generation_len=generation_len,
            num_computed_tokens=0,
            num_output_tokens=0,
            num_checkpointed_tokens=0,
            checkpoint_size_bytes=0,
            is_active=False,
            assigned_replica_id=None,
            ttft_slo_ms=request.ttft_slo_ms,
            tpot_slo_ms=request.tpot_slo_ms,
            failure_gap_slo_ms=request.failure_gap_slo_ms,
        )

    def _compute_costs_raw(
        self,
        *,
        request_id: str,
        prompt_len: int,
        generation_len: int,
        num_computed_tokens: int,
        num_output_tokens: int,
        num_checkpointed_tokens: int,
        checkpoint_size_bytes: int,
        is_active: bool,
        assigned_replica_id: int | None,
        ttft_slo_ms: float | None,
        tpot_slo_ms: float | None,
        failure_gap_slo_ms: float | None,
    ) -> RequestCosts:
        gen_len = generation_len

        if is_active:
            remaining_output = max(0, gen_len - num_output_tokens)
            remaining_prefill = max(0, prompt_len - num_computed_tokens)
            recovery_tokens = max(0, num_computed_tokens)
            run_tokens = max(0, num_computed_tokens)
        else:
            remaining_output = gen_len
            remaining_prefill = prompt_len
            # Pending/new requests have no published state yet. On failure
            # they only need to redo prompt prefill, not speculative decode.
            recovery_tokens = max(0, prompt_len)
            # Keep the normal-case memory approximation conservative.
            run_tokens = max(0, prompt_len + gen_len)

        p_j = (
            remaining_prefill / self.prefill_throughput
            if self.prefill_throughput > 0
            else 0.0
        )
        d_j = (
            remaining_output / self.decode_throughput
            if self.decode_throughput > 0
            else 0.0
        )

        published_tokens = (
            min(max(0, num_checkpointed_tokens), recovery_tokens)
            if is_active else 0
        )
        kv_bytes_per_token = self._estimate_kv_bytes_per_token(
            published_tokens=published_tokens,
            checkpoint_size_bytes=checkpoint_size_bytes,
        )
        restore_bytes = (
            checkpoint_size_bytes
            if published_tokens > 0 and checkpoint_size_bytes > 0
            else published_tokens * kv_bytes_per_token
        )
        replay_tokens = max(0, recovery_tokens - published_tokens)

        if self._cost_model is not None:
            # Profile-driven: piecewise linear interpolation (ms → sec)
            restore_time_sec = (
                self._cost_model.t_load(restore_bytes) / 1000.0
                if restore_bytes > 0 else 0.0
            )
            replay_time_sec = (
                (self._cost_model.t_prefill(recovery_tokens)
                 - self._cost_model.t_prefill(published_tokens)) / 1000.0
                if replay_tokens > 0 else 0.0
            )
        else:
            # Linear fallback
            restore_time_sec = (
                restore_bytes / self.load_bandwidth
                if restore_bytes > 0 and self.load_bandwidth > 0
                else 0.0
            )
            replay_time_sec = (
                replay_tokens / self.replay_throughput
                if replay_tokens > 0 and self.replay_throughput > 0
                else 0.0
            )

        resume_time_sec = (
            1.0 / self.decode_throughput
            if self.decode_throughput > 0
            else 0.0
        )
        gap_time_sec = (
            self.detection_time_sec
            + restore_time_sec
            + replay_time_sec
            + resume_time_sec
        )

        if self._cost_model is not None and is_active:
            # Profile-driven checkpoint overhead.
            # Total cost = c0 (fixed overhead) + t_ckpt(ΔS) (size-dependent).
            # checkpoint_ms_by_bytes in the profile has c0 already subtracted,
            # so we must add it back for the actual overhead estimate.
            stable_full_tokens = (num_computed_tokens // self.block_size) * self.block_size
            L = published_tokens
            u = max(0, stable_full_tokens - L)
            S = checkpoint_size_bytes
            delta_S = int(u * kv_bytes_per_token)
            if u > 0 and self._cost_model.should_publish(L, u, S, delta_S, self.checkpoint_lambda):
                checkpoint_overhead_sec = (self._cost_model._c0 + self._cost_model.t_ckpt(delta_S)) / 1000.0
            else:
                checkpoint_overhead_sec = 0.0
        else:
            # Linear fallback
            publish_estimate = estimate_online_checkpoint_publication(
                num_computed_tokens=num_computed_tokens if is_active else 0,
                num_checkpointed_tokens=num_checkpointed_tokens if is_active else 0,
                checkpoint_size_bytes=checkpoint_size_bytes if is_active else 0,
                block_size=self.block_size,
                replay_throughput_tokens_per_sec=self.replay_throughput,
                load_bandwidth_bytes_per_sec=self.load_bandwidth,
                checkpoint_bandwidth_bytes_per_sec=self.checkpoint_bandwidth,
                checkpoint_lambda=self.checkpoint_lambda,
                default_kv_bytes_per_token=self.kv_bytes_per_token,
            )
            checkpoint_overhead_sec = (
                publish_estimate.checkpoint_cost_sec
                if publish_estimate.should_publish
                else 0.0
            )

        ttft_slo = ttft_slo_ms / 1000.0 if ttft_slo_ms is not None else None
        tpot_slo = tpot_slo_ms / 1000.0 if tpot_slo_ms is not None else None
        gap_slo = (
            failure_gap_slo_ms / 1000.0
            if failure_gap_slo_ms is not None else None
        )

        costs = RequestCosts(
            request_id=request_id,
            G_j=gen_len,
            p_j=p_j,
            d_j=d_j,
            checkpoint_overhead_sec=checkpoint_overhead_sec,
            restore_time_sec=restore_time_sec,
            replay_time_sec=replay_time_sec,
            gap_time_sec=gap_time_sec,
            run_mem_bytes=float(run_tokens * kv_bytes_per_token),
            recovery_mem_bytes=float(run_tokens * kv_bytes_per_token),
            is_active=is_active,
            active_replica_id=assigned_replica_id if is_active else None,
            prompt_len=prompt_len,
            num_computed_tokens=num_computed_tokens,
            num_checkpointed_tokens=published_tokens,
            checkpoint_size_bytes=restore_bytes,
            ttft_slo_sec=ttft_slo,
            tpot_slo_sec=tpot_slo,
            gap_slo_sec=gap_slo,
        )
        return costs

    def build_snapshot_costs(
        self,
        active_requests: list[Request],
        pending_requests: list[Request],
    ) -> dict[str, RequestCosts]:
        costs: dict[str, RequestCosts] = {}
        for req in active_requests:
            costs[req.request_id] = self.compute_request_costs(
                req, is_active=True
            )
        for req in pending_requests:
            costs[req.request_id] = self.compute_request_costs(
                req, is_active=False
            )
        return costs

    def build_global_costs(
        self,
        active_snapshots: list["RequestSnapshot"],
        pending_requests: list["EngineCoreRequest"],
    ) -> dict[str, RequestCosts]:
        costs: dict[str, RequestCosts] = {}
        for snap in active_snapshots:
            costs[snap.request_id] = self.compute_costs_from_snapshot(
                snap, is_active=True
            )
        for req in pending_requests:
            costs[req.request_id] = self.compute_costs_from_engine_request(
                req
            )
        return costs

    @property
    def H_pre(self) -> float:
        return self.planning_horizon

    @property
    def H_dec(self) -> float:
        return self.planning_horizon

    def _estimate_kv_bytes_per_token(
        self,
        *,
        published_tokens: int,
        checkpoint_size_bytes: int,
    ) -> int:
        if published_tokens > 0 and checkpoint_size_bytes > 0:
            return max(1, checkpoint_size_bytes // published_tokens)
        return self.kv_bytes_per_token
