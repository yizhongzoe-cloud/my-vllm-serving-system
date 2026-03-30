# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benders-style Fault-Tolerant Scheduler Implementation.

Same composition pattern as FaultTolerantSchedulerImpl (base Scheduler +
FaultTolerantScheduler), but replaces the greedy batch admission with the
Benders decomposition solver from benders/solve_loop.py.

Architecture (A-route — local admission with global recovery screening):
    Each EngineCore runs its own solver instance. The solver decides:
    - **Admission**: which pending requests to accept on THIS replica.
    - **Recovery feasibility**: whether all requests on this replica can be
      recovered onto surviving replicas if this replica fails.

    The solver does NOT do cross-replica routing (each EngineCore only
    executes on its own GPU). Routing is done client-side (round-robin
    or load-balanced). Recovery plans are propagated to the client so
    failover can prefer solver-planned targets.

    Checkpoint publication itself is not a solver variable. The runtime uses
    the shared online checkpoint controller, while the solver consumes only
    the real published checkpoint state carried in request snapshots.

    Remote replicas' load is unknown to each EngineCore. The recovery
    checker conservatively estimates remote load using the local replica's
    load fraction (assumes balanced routing).

When the solver fails to converge or times out, falls back to the baseline
greedy admission to ensure liveness.

Activated by setting scheduling_policy="ft_benders" in the engine config.
"""

from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.checkpoint_controller import CheckpointConfig
from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder
from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop
from vllm.v1.core.sched.ft_scheduler import FaultTolerantScheduler, FTSchedulerConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.failure_detector import ReplicaStatus
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


class BendersFTSchedulerImpl(SchedulerInterface):
    """Per-EngineCore scheduler that uses Benders decomposition for
    admission control and recovery-aware routing on this replica, with
    recovery feasibility screening across all replicas.

    Does NOT do cross-replica routing (vllm DP architecture constraint).
    Falls back to greedy admission when the solver cannot converge.
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
        # Build the base scheduler (all standard vLLM logic).
        base_vllm_config = _normalize_base_scheduler_config(vllm_config)
        self._base = Scheduler(
            vllm_config=base_vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        self.connector = self._base.connector

        # Build the FT scheduler (shared components).
        sched_cfg = vllm_config.scheduler_config
        parallel_cfg = vllm_config.parallel_config

        # Replica identity.
        dp_size = parallel_cfg.data_parallel_size
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
            kv_bytes_per_token=sched_cfg.ft_kv_bytes_per_token,
            checkpoint_cost_profile=sched_cfg.ft_checkpoint_cost_profile,
        )
        self._ft = FaultTolerantScheduler(config=ft_config, dp_size=dp_size)

        prefill_tput = sched_cfg.ft_prefill_throughput or 0.0
        decode_tput = sched_cfg.ft_decode_throughput or 0.0
        load_bw = sched_cfg.ft_load_bandwidth or 0.0

        self._replica_id = dp_rank
        self._dp_size = dp_size

        # Register this engine as a replica in the FT scheduler.
        # The FT scheduler only tracks this EngineCore's own replica.
        self._ft.register_replica(
            replica_id=dp_rank,
            gpu_id=dp_rank,
            prefill_throughput=prefill_tput,
            decode_throughput=decode_tput,
            load_bandwidth=load_bw,
            max_num_seqs=sched_cfg.max_num_seqs,
            max_num_batched_tokens=sched_cfg.max_num_batched_tokens,
        )

        # For the Benders solver, we need to know about ALL replicas
        # so recovery subproblems can verify that if this replica fails,
        # the remaining (dp_size - 1) replicas have enough capacity.
        # We store the full replica set for the solver only.
        self._all_replica_ids = list(range(dp_size))

        # Clamp max_gpu_failures if not enough replicas.
        num_replicas = dp_size
        if ft_config.max_gpu_failures >= num_replicas:
            clamped = max(0, num_replicas - 1)
            logger.warning(
                "max_gpu_failures=%d >= num_replicas=%d; clamping to %d",
                ft_config.max_gpu_failures,
                num_replicas,
                clamped,
            )
            self._ft.config.max_gpu_failures = clamped

        # Benders solver components.
        planning_horizon = sched_cfg.ft_planning_horizon or self._ft.replica_manager.planning_horizon
        ckpt_cfg = self._ft.checkpoint_controller.config

        mem_cap = sched_cfg.ft_memory_capacity_bytes or 0

        self._cost_builder = CostTableBuilder(
            planning_horizon=planning_horizon,
            prefill_throughput=prefill_tput,
            decode_throughput=decode_tput,
            load_bandwidth=load_bw,
            checkpoint_bandwidth=(
                sched_cfg.ft_checkpoint_bandwidth
                or sched_cfg.ft_load_bandwidth
                or 0.0
            ),
            replay_throughput=prefill_tput,  # replay ≈ prefill
            detection_time_sec=ft_config.detection_time_sec,
            memory_capacity_bytes=mem_cap,
            checkpoint_config=ckpt_cfg,
            kv_bytes_per_token=sched_cfg.ft_kv_bytes_per_token,
            block_size=block_size,
            checkpoint_lambda=sched_cfg.ft_checkpoint_lambda,
        )

        max_iter = sched_cfg.benders_max_iterations or 20
        master_tl = sched_cfg.benders_master_time_limit or 1.0
        recovery_tl = sched_cfg.benders_recovery_time_limit or 0.5

        self._solve_loop = BendersSolveLoop(
            cost_builder=self._cost_builder,
            max_iterations=max_iter,
            master_time_limit_sec=master_tl,
            recovery_time_limit_sec=recovery_tl,
            max_gpu_failures=self._ft.config.max_gpu_failures,
        )

        # Pending admission queue.
        self._pending_ft_admission: list["Request"] = []

        # Latest recovery plans from the solver, keyed by failure scenario.
        # Used by RecoveryManager to prefer solver-planned assignments.
        self._recovery_plans: dict[frozenset[int], dict[str, int]] = {}

        # Wire solver recovery lookup into RecoveryManager.
        self._ft.recovery_manager._solver_recovery_lookup = (
            self.get_solver_recovery_target
        )

        logger.info(
            "BendersFTSchedulerImpl initialized "
            "(replica_id=%d, dp_size=%d, max_gpu_failures=%d, "
            "checkpointing=%s, benders_max_iter=%d)",
            dp_rank,
            dp_size,
            self._ft.config.max_gpu_failures,
            ft_config.enable_checkpointing,
            max_iter,
        )

    # ---- SchedulerInterface with Benders admission ----

    def add_request(self, request: "Request") -> None:
        """Queue request for Benders-based batch admission."""
        self._pending_ft_admission.append(request)
        self._base.add_request(request)

    def _process_pending_admissions(self) -> None:
        """Replace greedy admission with Benders solver."""
        if not self._pending_ft_admission:
            return

        pending = list(self._pending_ft_admission)
        self._pending_ft_admission.clear()

        # Build snapshot: only this replica's active requests.
        # In DP mode, each EngineCore only sees its own requests.
        active = self._ft.request_pool.get_admitted_requests()

        # The solver uses only this replica for admission/routing
        # (since this EngineCore can only execute on its own GPU),
        # but all healthy replicas for recovery subproblems (to verify
        # that if this replica fails, survivors can absorb the requests).
        solve_replica_ids = [self._replica_id]

        # Filter all_replica_ids to only healthy ones. In coordinated FT
        # mode, the failure detector is notified of remote failures.
        # Replicas not registered in this engine's detector (i.e. remote
        # replicas that haven't been reported as failed) are assumed healthy.
        healthy_all = []
        for r in self._all_replica_ids:
            status = self._ft.failure_detector.get_status(r)
            if status is None or status == ReplicaStatus.HEALTHY:
                healthy_all.append(r)

        # Run Benders solve.
        result = self._solve_loop.solve_epoch(
            active_requests=active,
            pending_requests=pending,
            replica_ids=solve_replica_ids,
            all_replica_ids=healthy_all,
        )

        if result is not None:
            # Commit solver decisions for pending requests.
            for req in pending:
                if req.request_id in result.master_solution.admitted:
                    assignment = result.master_solution.assignments.get(
                        req.request_id
                    )
                    if assignment is not None:
                        r_id = assignment
                        self._ft.request_pool.add_request(req)
                        self._ft.request_pool.admit_request(
                            req.request_id, r_id
                        )
                        self._ft.replica_manager.assign_request(req, r_id)
                        logger.debug(
                            "Benders admitted %s → replica %d",
                            req.request_id,
                            r_id,
                        )
                    else:
                        # Admitted but no assignment (shouldn't happen).
                        self._base.finish_requests(
                            req.request_id, RequestStatus.FINISHED_ABORTED
                        )
                else:
                    # Rejected by solver.
                    logger.debug(
                        "Benders rejected %s (G_j=%d)",
                        req.request_id,
                        req.generation_len,
                    )
                    self._base.finish_requests(
                        req.request_id, RequestStatus.FINISHED_ABORTED
                    )

            # Store recovery plans for use during failover.
            self._recovery_plans = {
                omega: plan.assignments
                for omega, plan in result.recovery_plans.items()
            }
        else:
            # Solver failed — fall back to greedy.
            logger.warning(
                "Benders solver returned None; falling back to greedy "
                "for %d pending requests",
                len(pending),
            )
            self._greedy_fallback(pending)

    def _greedy_fallback(self, pending: list["Request"]) -> None:
        """Fall back to baseline greedy admission."""
        # Clear stale recovery plans from the previous solver epoch.
        # The admission set is about to change, so old plans are invalid.
        self._recovery_plans.clear()
        pending.sort(key=lambda r: r.generation_len, reverse=True)
        for request in pending:
            admitted = self._ft.admit_request(request)
            if not admitted:
                self._base.finish_requests(
                    request.request_id, RequestStatus.FINISHED_ABORTED
                )

    def schedule(self) -> "SchedulerOutput":
        """Batch-admit pending requests via Benders, then run base scheduling."""
        self._process_pending_admissions()

        output = self._base.schedule()

        # Adaptive checkpointing (same as baseline).
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
        result = self._base.update_from_output(
            scheduler_output, model_runner_output
        )

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
        if isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = list(request_ids)

        for req_id in ids:
            self._ft.abort_request(req_id)

        self._base.finish_requests(request_ids, finished_status)

    # ---- Checkpoint support for EngineCore ----

    def get_checkpoint_requests(
        self,
    ) -> list[tuple[str, list[int], int]]:
        if not self._ft.config.enable_checkpointing:
            return []

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
        return self._ft

    def get_solver_recovery_target(
        self, failed_replica_id: int, request_id: str
    ) -> int | None:
        """Look up the solver-planned recovery target for a request.

        Checks plans in order of specificity:
        1. Exact match for the single-failure scenario {failed_replica_id}.
        2. Any scenario that is a superset (multi-failure plan covering
           this failure).
        Returns the target replica_id or None if no plan exists.
        """
        if not self._recovery_plans:
            return None

        # Exact single-failure match (most common case).
        omega = frozenset({failed_replica_id})
        plan = self._recovery_plans.get(omega)
        if plan is not None:
            return plan.get(request_id)

        # Superset fallback: find a multi-failure plan that includes
        # the failed replica.
        for scenario, plan in self._recovery_plans.items():
            if failed_replica_id in scenario:
                target = plan.get(request_id)
                if target is not None:
                    return target

        return None

    # ---- Pure delegation ----

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
