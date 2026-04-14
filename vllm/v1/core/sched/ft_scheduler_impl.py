# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fault-Tolerant Scheduler Implementation.

A wrapper scheduler that implements SchedulerInterface by delegating to the
base Scheduler while injecting FT-specific logic at key points:

1. **add_request**: Run FT admission control before passing to base scheduler.
2. **schedule**: After base scheduling, trigger adaptive checkpoint decisions.
3. **update_from_output**: After base processing, handle FT cleanup for
   finished requests.
4. **finish_requests**: Clean up FT state when requests are aborted.

In DP (data-parallel) mode, each EngineCore runs its own instance of this
scheduler.  The replica_id is derived from the DP rank, and the FT
coordinator (running in a separate process) handles cross-replica health
monitoring and failover signaling.

In single-replica mode (DP=1), FT admission still applies SLO checks, but
failure-tolerance for GPU crashes is not possible (no surviving replica).
"""

import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.ft_scheduler import FaultTolerantScheduler, FTSchedulerConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

# ─────────────────────────────────────────────────────────────────────────────
# P0-impl-3a: cProfile instrumentation for diagnosing the ~32 ms wrapper
# overhead. Disabled by default (zero cost). Set FT_PROFILE_MAX_CALLS=N
# to capture the first N schedule() calls and dump to FT_PROFILE_OUTPUT.
#
# Usage:
#   FT_PROFILE_MAX_CALLS=500 \
#   FT_PROFILE_OUTPUT=/tmp/ft_schedule_profile.txt \
#   <run server normally>
#
# After N calls, a pstats dump (sorted by cumulative + tottime) is written
# to the output file. Subsequent schedule() calls are NOT profiled.
# ─────────────────────────────────────────────────────────────────────────────
_ft_profile_state: dict = {
    "profile": None,
    "calls": 0,
    "max_calls": int(os.environ.get("FT_PROFILE_MAX_CALLS", "0") or "0"),
    "output_path": os.environ.get(
        "FT_PROFILE_OUTPUT", "/tmp/ft_schedule_profile.txt"
    ),
    "wall_time_samples": [],  # list of (process_pending, base_schedule, ckpt_step) tuples
    "dumped": False,
}

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


def _auto_kv_bytes_per_token(vllm_config: "VllmConfig") -> int:
    """Compute KV bytes per token from model config.

    Formula: 2(KV) × num_kv_heads × head_dim × 2(fp16) × num_layers
    Falls back to the user-configured value if model config is unavailable.
    """
    configured = vllm_config.scheduler_config.ft_kv_bytes_per_token
    try:
        hf_config = vllm_config.model_config.hf_config
        num_kv_heads = getattr(hf_config, "num_key_value_heads",
                               hf_config.num_attention_heads)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        num_layers = hf_config.num_hidden_layers
        # 2 for K+V, 2 for fp16 bytes
        auto_val = 2 * num_kv_heads * head_dim * 2 * num_layers
        if auto_val != configured:
            logger.info(
                "Auto-computed ft_kv_bytes_per_token=%d from model config "
                "(configured default=%d)", auto_val, configured)
        return auto_val
    except Exception:
        return configured


def _normalize_base_scheduler_config(vllm_config: "VllmConfig") -> "VllmConfig":
    """Map FT wrapper policies to a base Scheduler policy the engine understands."""
    sched_cfg = vllm_config.scheduler_config
    if sched_cfg.policy == "fault_tolerant":
        return vllm_config
    base_sched_kwargs = asdict(sched_cfg)
    base_sched_kwargs["policy"] = "fault_tolerant"
    base_sched_kwargs["max_model_len"] = vllm_config.model_config.max_model_len
    base_sched_kwargs["is_encoder_decoder"] = (
        vllm_config.model_config.is_encoder_decoder
    )
    base_sched_cfg = sched_cfg.default_factory(**base_sched_kwargs)
    return vllm_config.replace(scheduler_config=base_sched_cfg)


class FaultTolerantSchedulerImpl(SchedulerInterface):
    """SchedulerInterface implementation that wraps the base Scheduler
    with fault-tolerant admission, checkpointing, and recovery logic.

    Composition pattern: this class owns a base Scheduler (for all the
    standard vLLM scheduling logic) and a FaultTolerantScheduler (for
    FT-specific components like RequestPool, ReplicaManager, etc.).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # Build the base scheduler (handles all standard vLLM logic).
        base_vllm_config = _normalize_base_scheduler_config(vllm_config)
        # P0-impl-3a fix (2026-04-08): when async_scheduling is enabled,
        # base must be AsyncScheduler (which manages num_output_placeholders).
        # Otherwise the batch_queue=2 pipeline causes 50% under-sampling
        # (num_new_tokens==0 every other step), doubling tpot. See
        # experiments_v2/docs/e1a_quick_diagnosis.md "Final Root Cause".
        if base_vllm_config.scheduler_config.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            _BaseSchedulerCls = AsyncScheduler
            logger.info(
                "FaultTolerantSchedulerImpl: async_scheduling=True, "
                "using AsyncScheduler as base (P0-impl-3a fix)"
            )
        else:
            _BaseSchedulerCls = Scheduler
        self._base = _BaseSchedulerCls(
            vllm_config=base_vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        self.connector = self._base.connector
        self._original_policy = vllm_config.scheduler_config.policy

        # Build the FT scheduler with config from SchedulerConfig.
        sched_cfg = vllm_config.scheduler_config
        parallel_cfg = vllm_config.parallel_config

        # Determine replica identity.
        # In DP mode: each EngineCore is a separate replica.
        # The DP rank is the replica_id.
        #
        # NOTE: In vLLM's DP implementation, each EngineCore subprocess
        # sees data_parallel_size=1 (local view). But FT needs the global
        # replica count for capacity-under-failures checks. We use
        # max_gpu_failures + 1 as the minimum (you can't tolerate k failures
        # with fewer than k+1 replicas), and take the max with the local
        # value to handle both single-process and multi-process cases.
        local_dp_size = parallel_cfg.data_parallel_size
        dp_size = max(local_dp_size, sched_cfg.max_gpu_failures + 1)
        dp_rank = parallel_cfg.data_parallel_rank or 0

        ft_config = FTSchedulerConfig(
            max_gpu_failures=sched_cfg.max_gpu_failures,
            checkpoint_pool_bytes=sched_cfg.checkpoint_pool_bytes,
            detection_time_sec=sched_cfg.failure_detection_time_ms / 1000.0,
            heartbeat_interval_sec=sched_cfg.heartbeat_interval_sec,
            failure_timeout_sec=sched_cfg.failure_timeout_sec,
            enable_checkpointing=sched_cfg.enable_checkpointing,
            fixed_checkpoint_level=sched_cfg.fixed_checkpoint_level,
            fixed_checkpoint_blocks=sched_cfg.fixed_checkpoint_blocks,
            block_size=block_size,
            replay_throughput_tokens_per_sec=(
                sched_cfg.ft_prefill_throughput or 0.0
            ),
            load_bandwidth_bytes_per_sec=(
                sched_cfg.ft_load_bandwidth or 0.0
            ),
            checkpoint_bandwidth_bytes_per_sec=(
                sched_cfg.ft_checkpoint_bandwidth
                or sched_cfg.ft_load_bandwidth
                or 0.0
            ),
            checkpoint_lambda=sched_cfg.ft_checkpoint_lambda,
            kv_bytes_per_token=_auto_kv_bytes_per_token(vllm_config),
            checkpoint_cost_profile=sched_cfg.ft_checkpoint_cost_profile,
        )
        self._ft = FaultTolerantScheduler(config=ft_config, dp_size=dp_size)

        # Load decode-first capacity model for greedy admission.
        from vllm.v1.core.sched.benders.decode_capacity_model import (
            DecodeCapacityModel,
        )
        self._ft._decode_cap_model = DecodeCapacityModel(
            sched_cfg.ft_decode_capacity_profile or None
        )

        # Get throughput estimates from config (user-configured or defaults).
        # These can also be profiled at runtime later.
        prefill_tput = sched_cfg.ft_prefill_throughput or 0.0
        decode_tput = sched_cfg.ft_decode_throughput or 0.0
        load_bw = sched_cfg.ft_load_bandwidth or 0.0

        # Register this engine as a replica.
        self._replica_id = dp_rank
        self._ft.register_replica(
            replica_id=dp_rank,
            gpu_id=dp_rank,
            prefill_throughput=prefill_tput,
            decode_throughput=decode_tput,
            load_bandwidth=load_bw,
            max_num_seqs=sched_cfg.max_num_seqs,
            max_num_batched_tokens=sched_cfg.max_num_batched_tokens,
        )

        # In single-replica mode, skip failure monitoring entirely to
        # avoid false-alarming the only replica (no heartbeat source).
        # In multi-replica mode, failure monitoring is handled by the
        # FTCoordinator process, not the in-engine failure detector.
        # So we do NOT start the failure detector's background thread here.
        # Pending admission queue: requests are added here first, then
        # batch-admitted in schedule() sorted by G_j (descending) to
        # maximize goodput (paper objective: max ∑ G_j * y_j).
        self._pending_ft_admission: list["Request"] = []

        logger.info(
            "FaultTolerantSchedulerImpl initialized "
            "(replica_id=%d, dp_size=%d, max_gpu_failures=%d, "
            "checkpointing=%s, prefill_tput=%.1f, decode_tput=%.1f)",
            dp_rank,
            dp_size,
            self._ft.config.max_gpu_failures,
            ft_config.enable_checkpointing,
            prefill_tput,
            decode_tput,
        )

    # ---- SchedulerInterface delegation with FT hooks ----

    def add_request(self, request: "Request") -> None:
        """FT hook: queue request for batch admission, add to base scheduler.

        Requests are NOT admitted immediately. Instead they are queued
        and batch-admitted in schedule() sorted by G_j (descending) to
        maximize the goodput objective: max ∑ G_j * y_j.
        """
        self._pending_ft_admission.append(request)
        self._base.add_request(request)

    # KV pressure threshold: defer admission when free fraction is below
    # this value (i.e. usage > 1 - threshold). 0.10 means defer when KV
    # cache is more than 90% full.
    _KV_FREE_RATIO_FLOOR: float = 0.10

    def _kv_pressure_too_high(self) -> bool:
        """Check whether GPU KV cache is too full to safely admit more.

        Returns True iff the fraction of free KV blocks is below
        _KV_FREE_RATIO_FLOOR. The FT wrapper bypasses vLLM base
        scheduler's natural KV-pressure throttling, so we re-implement
        an equivalent check here at admission time.
        """
        try:
            block_pool = self._base.kv_cache_manager.block_pool
            total_blocks = block_pool.num_gpu_blocks - 1  # exclude null
            if total_blocks <= 0:
                logger.info(
                    "FT_KV_THROTTLE: skip check (total_blocks=%d)", total_blocks
                )
                return False
            free_blocks = block_pool.get_num_free_blocks()
            free_ratio = free_blocks / total_blocks
            triggered = free_ratio < self._KV_FREE_RATIO_FLOOR
            # P0-impl-3a follow-up: was INFO, demoted to DEBUG.  This
            # check fires every admission step (~9 calls/sec) and the
            # log spam consumed measurable API-server CPU.
            logger.debug(
                "FT_KV_THROTTLE: free=%d/%d (%.1f%% used) "
                "threshold=%.1f%% triggered=%s pending=%d",
                free_blocks, total_blocks, (1 - free_ratio) * 100,
                (1 - self._KV_FREE_RATIO_FLOOR) * 100,
                triggered, len(self._pending_ft_admission),
            )
            return triggered
        except Exception as exc:
            logger.warning("FT_KV_THROTTLE: check failed: %s", exc)
            return False

    def _process_pending_admissions(self) -> None:
        """Batch-admit pending requests sorted by G_j (descending).

        Implements the greedy heuristic for the paper's goodput
        objective: admit requests with larger expected output first,
        as they contribute more to total goodput (∑ G_j * y_j).
        Requests that fail admission are aborted.
        """
        if not self._pending_ft_admission:
            return

        # KV pressure throttle: defer admission when GPU KV cache is too full.
        # vLLM base scheduler handles this naturally for fcfs by checking
        # free blocks in alloc_slots(); the FT wrapper bypasses base
        # admission so we re-implement the same throttle here. Without
        # this, the FT wrapper over-admits in decode-heavy heavy-load
        # workloads (W1_Chat/Heavy) and Running reqs grow until KV cache
        # saturates at 99-100%, causing per-request latency to balloon.
        if self._kv_pressure_too_high():
            return  # leave pending requests in queue, retry next epoch

        # Sort by G_j descending — prefer requests that contribute
        # more to total goodput.
        pending = sorted(
            self._pending_ft_admission,
            key=lambda r: r.generation_len,
            reverse=True,
        )
        self._pending_ft_admission.clear()

        # In centralized Benders mode, the client-side solver has already
        # made the admission decision. Trust it and skip local checks.
        is_centralized = self._original_policy == "ft_benders_centralized"

        for request in pending:
            if is_centralized:
                # Solver already approved — just register with FT scheduler.
                self._ft.register_admitted_request(request)
                admitted = True
            else:
                admitted = self._ft.admit_request(request)
            if not admitted:
                logger.warning(
                    "FT admission rejected request %s (G_j=%d); "
                    "finishing with FINISHED_ABORTED",
                    request.request_id,
                    request.generation_len,
                )
                self._base.finish_requests(
                    request.request_id, RequestStatus.FINISHED_ABORTED
                )

    def schedule(self) -> "SchedulerOutput":
        """FT hook: batch-admit pending requests, then run base scheduling
        and checkpoint decisions."""
        # ── P0-impl-3a: optional cProfile + per-section wall timing ──
        # Zero overhead when FT_PROFILE_MAX_CALLS is unset/0.
        _state = _ft_profile_state
        _profile_active = (
            _state["max_calls"] > 0
            and _state["calls"] < _state["max_calls"]
        )
        _pr = None
        if _profile_active:
            if _state["profile"] is None:
                import cProfile
                _state["profile"] = cProfile.Profile()
            _pr = _state["profile"]
            _pr.enable()

        try:
            _t0 = time.perf_counter() if _profile_active else 0.0
            # Batch-admit pending requests sorted by G_j (goodput-max).
            self._process_pending_admissions()

            _t1 = time.perf_counter() if _profile_active else 0.0
            output = self._base.schedule()

            _t2 = time.perf_counter() if _profile_active else 0.0
            # Trigger adaptive checkpointing for running requests.
            # Block IDs are resolved via kv_cache_manager; actual GPU→CPU
            # copies are triggered separately via collective_rpc in
            # EngineCore.step() (see core.py checkpoint hook).
            #
            # P0-impl-3a follow-up: optionally throttle the per-step
            # checkpoint policy iteration. The full Python loop over
            # running requests is the dominant scheduler overhead vs
            # upstream fcfs; with K>1 we evaluate only every K steps.
            if self._ft.config.enable_checkpointing:
                self._ckpt_step_counter = (
                    getattr(self, "_ckpt_step_counter", 0) + 1
                )
                _interval = max(
                    1,
                    int(os.environ.get("FT_CHECKPOINT_STEP_INTERVAL", "1")),
                )
                if self._ckpt_step_counter % _interval == 0:
                    running = self._base.running
                    self._ft.run_checkpoint_step(
                        running_requests=running,
                        gpu_kv_caches=None,
                        kv_cache_manager=self._base.kv_cache_manager,
                    )
            _t3 = time.perf_counter() if _profile_active else 0.0

            if _profile_active:
                _state["wall_time_samples"].append(
                    (_t1 - _t0, _t2 - _t1, _t3 - _t2)
                )
            return output
        finally:
            if _profile_active:
                _pr.disable()
                _state["calls"] += 1
                if _state["calls"] >= _state["max_calls"] and not _state["dumped"]:
                    _state["dumped"] = True
                    self._ft_dump_profile()

    def _ft_dump_profile(self) -> None:
        """P0-impl-3a: dump cProfile + per-section wall timing summary."""
        _state = _ft_profile_state
        try:
            import pstats
            samples = _state["wall_time_samples"]
            n = len(samples)
            with open(_state["output_path"], "w") as f:
                # ── Header & per-section wall timing ──
                f.write(
                    f"FT_PROFILE: captured {n} schedule() calls\n"
                    f"Output path: {_state['output_path']}\n"
                    f"\n"
                    f"=== Per-section wall timing (ms) ===\n"
                )
                if n > 0:
                    sums = [sum(s[i] for s in samples) for i in range(3)]
                    avgs = [s / n * 1000 for s in sums]
                    p50s = [
                        sorted(s[i] for s in samples)[n // 2] * 1000
                        for i in range(3)
                    ]
                    p95s = [
                        sorted(s[i] for s in samples)[int(n * 0.95)] * 1000
                        for i in range(3)
                    ]
                    section_names = [
                        "_process_pending_admissions",
                        "self._base.schedule()",
                        "self._ft.run_checkpoint_step (if enabled)",
                    ]
                    f.write(
                        f"{'Section':<45} {'avg':>10} {'p50':>10} {'p95':>10}\n"
                    )
                    for name, a, p50, p95 in zip(section_names, avgs, p50s, p95s):
                        f.write(f"{name:<45} {a:>9.3f}ms {p50:>9.3f}ms {p95:>9.3f}ms\n")
                    f.write(
                        f"{'TOTAL':<45} {sum(avgs):>9.3f}ms "
                        f"{sum(p50s):>9.3f}ms {sum(p95s):>9.3f}ms\n"
                    )
                f.write("\n")

                # ── cProfile cumulative time top 50 ──
                f.write("=== cProfile: top 50 by cumulative time ===\n")
                stats = pstats.Stats(_state["profile"], stream=f)
                stats.strip_dirs()
                stats.sort_stats("cumulative")
                stats.print_stats(50)

                # ── cProfile total time top 50 ──
                f.write("\n=== cProfile: top 50 by total time (excluding subcalls) ===\n")
                stats.sort_stats("tottime")
                stats.print_stats(50)

            logger.info(
                "FT_PROFILE: dumped %d schedule() samples to %s",
                n, _state["output_path"],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("FT_PROFILE dump failed: %s", exc)

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """FT hook: after base processing, clean up FT state for finished
        requests."""
        result = self._base.update_from_output(
            scheduler_output, model_runner_output
        )

        # Check for requests that finished in this step and clean up FT state.
        for req_id in self._base.finished_req_ids:
            request = self._ft.request_pool.get_request(req_id)
            if request is not None:
                self._ft.complete_request(request)

        return result

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "RequestStatus",
    ) -> None:
        """FT hook: clean up FT state when requests are aborted."""
        if isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = list(request_ids)

        for req_id in ids:
            self._ft.abort_request(req_id)

        self._base.finish_requests(request_ids, finished_status)

    # ---- Checkpoint support for EngineCore integration ----

    def get_checkpoint_requests(
        self,
    ) -> list[tuple[str, list[int], int]]:
        """Get requests that need checkpointing this step.

        Returns list of (request_id, block_ids, num_tokens) tuples.
        Called by EngineCore.step() to know which requests to checkpoint
        on the GPU side via collective_rpc.

        Uses cached decisions from step_checkpoints() (called in
        schedule()) to avoid double-evaluating should_checkpoint(),
        which has time/step guards that would give inconsistent results
        on the second call.
        """
        if not self._ft.config.enable_checkpointing:
            return []

        # Use cached decisions from step_checkpoints(); fall back to
        # fresh evaluation only if cache is empty (shouldn't happen
        # in normal flow).
        cached = self._ft.get_and_clear_cached_checkpoint_requests()
        if cached is None:
            cached = self._ft.checkpoint_controller.get_requests_to_checkpoint(
                self._base.running
            )

        result = []
        for request in cached:
            block_ids = self._ft._get_request_kv_block_ids(
                request, self._base.kv_cache_manager
            )
            if block_ids:
                result.append((
                    request.request_id,
                    block_ids,
                    request.num_computed_tokens,
                ))
        return result

    @property
    def ft_scheduler(self) -> FaultTolerantScheduler:
        """Expose FT scheduler for EngineCore checkpoint integration."""
        return self._ft

    # ---- Pure delegation (no FT logic needed) ----

    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        return self._base.get_grammar_bitmask(scheduler_output)

    def update_draft_token_ids(
        self, draft_token_ids: "DraftTokenIds"
    ) -> None:
        self._base.update_draft_token_ids(draft_token_ids)

    def update_draft_token_ids_in_output(
        self,
        draft_token_ids: "DraftTokenIds",
        scheduler_output: "SchedulerOutput",
    ) -> None:
        self._base.update_draft_token_ids_in_output(
            draft_token_ids, scheduler_output
        )

    def get_num_unfinished_requests(self) -> int:
        return self._base.get_num_unfinished_requests()

    def has_finished_requests(self) -> bool:
        return self._base.has_finished_requests()

    def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
        reset_connector: bool = False,
    ) -> bool:
        return self._base.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def get_request_counts(self) -> tuple[int, int]:
        return self._base.get_request_counts()

    def make_stats(self) -> Optional["SchedulerStats"]:
        return self._base.make_stats()

    def shutdown(self) -> None:
        self._ft.stop()
        self._base.shutdown()

    def get_kv_connector(self) -> Optional["KVConnectorBase_V1"]:
        return self._base.get_kv_connector()
