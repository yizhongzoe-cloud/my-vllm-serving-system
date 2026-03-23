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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.ft_scheduler import FaultTolerantScheduler, FTSchedulerConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

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
        self._base = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        # Build the FT scheduler with config from SchedulerConfig.
        sched_cfg = vllm_config.scheduler_config
        parallel_cfg = vllm_config.parallel_config
        ft_config = FTSchedulerConfig(
            max_gpu_failures=sched_cfg.max_gpu_failures,
            checkpoint_pool_bytes=sched_cfg.checkpoint_pool_bytes,
            detection_time_sec=sched_cfg.failure_detection_time_ms / 1000.0,
            heartbeat_interval_sec=sched_cfg.heartbeat_interval_sec,
            failure_timeout_sec=sched_cfg.failure_timeout_sec,
            enable_checkpointing=sched_cfg.enable_checkpointing,
        )
        self._ft = FaultTolerantScheduler(config=ft_config)

        # Determine replica identity.
        # In DP mode: each EngineCore is a separate replica.
        # The DP rank is the replica_id.
        dp_size = parallel_cfg.data_parallel_size
        dp_rank = parallel_cfg.data_parallel_rank or 0

        # Get throughput estimates from config (user-configured or defaults).
        # These can also be profiled at runtime later.
        prefill_tput = getattr(sched_cfg, "ft_prefill_throughput", 0.0) or 0.0
        decode_tput = getattr(sched_cfg, "ft_decode_throughput", 0.0) or 0.0
        load_bw = getattr(sched_cfg, "ft_load_bandwidth", 0.0) or 0.0

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

        # In single-replica mode (DP=1), fault tolerance for GPU failures
        # is impossible (no surviving replica), so clamp max_gpu_failures
        # to 0.  SLO admission checks still apply.
        num_replicas = dp_size
        if ft_config.max_gpu_failures >= num_replicas:
            clamped = max(0, num_replicas - 1)
            logger.warning(
                "max_gpu_failures=%d >= num_replicas=%d; "
                "clamping to %d (FT needs at least k+1 replicas)",
                ft_config.max_gpu_failures,
                num_replicas,
                clamped,
            )
            self._ft.config.max_gpu_failures = clamped

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

    def _process_pending_admissions(self) -> None:
        """Batch-admit pending requests sorted by G_j (descending).

        Implements the greedy heuristic for the paper's goodput
        objective: admit requests with larger expected output first,
        as they contribute more to total goodput (∑ G_j * y_j).
        Requests that fail admission are aborted.
        """
        if not self._pending_ft_admission:
            return

        # Sort by G_j descending — prefer requests that contribute
        # more to total goodput.
        pending = sorted(
            self._pending_ft_admission,
            key=lambda r: r.generation_len,
            reverse=True,
        )
        self._pending_ft_admission.clear()

        for request in pending:
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
        # Batch-admit pending requests sorted by G_j (goodput-max).
        self._process_pending_admissions()

        output = self._base.schedule()

        # Trigger adaptive checkpointing for running requests.
        # Block IDs are resolved via kv_cache_manager; actual GPU→CPU
        # copies are triggered separately via collective_rpc in
        # EngineCore.step() (see core.py checkpoint hook).
        if self._ft.config.enable_checkpointing:
            running = self._base.running
            self._ft.run_checkpoint_step(
                running_requests=running,
                gpu_kv_caches=None,
                kv_cache_manager=self._base.kv_cache_manager,
            )

        return output

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
