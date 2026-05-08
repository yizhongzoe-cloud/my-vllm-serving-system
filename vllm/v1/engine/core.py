# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    ReplicaSnapshot,
    RequestSnapshot,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc

# ── Solution-4 lazy KV reload (FT_LAZY_RELOAD env var, default off) ──────────
# When enabled, rerouted requests arriving from a failed engine are NOT
# immediately admitted to the scheduler.  Instead they sit in an in-engine
# holding deque (`_ft_lazy_pending`) and are drained into the scheduler in
# small batches as previously-admitted rerouted requests finish.  This caps
# how much KV cache the surviving engine devotes to migrated state at any
# moment, freeing slots for new arrivals and preventing the post-fault
# queue blowup observed in W1_Chat/Heavy/F2_Mid (see
# experiments_v2/docs/e1a_quick_diagnosis.md Follow-up finding #5).
#
# Trade-off: drains the migrated set serially, so the per-request
# failover_gap for the *last* drained request grows. The reload mode's
# failover_gap is already well above the W1_Chat 3 s gap_slo, so this
# trade is favourable for total goodput.
_FT_LAZY_RELOAD_ENABLED = os.environ.get("FT_LAZY_RELOAD", "0") == "1"
try:
    _FT_LAZY_MAX_CONCURRENT = int(os.environ.get("FT_LAZY_MAX_CONCURRENT", "6"))
except ValueError:
    _FT_LAZY_MAX_CONCURRENT = 6
if _FT_LAZY_MAX_CONCURRENT < 1:
    _FT_LAZY_MAX_CONCURRENT = 1
if _FT_LAZY_RELOAD_ENABLED:
    logger.info(
        "FT_LAZY_RELOAD enabled: rerouted requests will be drained "
        "with max %d concurrent in scheduler",
        _FT_LAZY_MAX_CONCURRENT,
    )

# ─────────────────────────────────────────────────────────────────────────────
# P0-impl-3a: per-section wall timing for EngineCore.step().
# Disabled by default. Enable with FT_STEP_TIMING_MAX_CALLS=N (e.g. 500).
# Output: FT_STEP_TIMING_OUTPUT (default /tmp/ft_step_timing.txt)
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_ft_step_timing_state: dict = {
    "calls": 0,
    "max_calls": int(_os.environ.get("FT_STEP_TIMING_MAX_CALLS", "0") or "0"),
    "output_path": _os.environ.get(
        "FT_STEP_TIMING_OUTPUT", "/tmp/ft_step_timing.txt"
    ),
    # samples is a list of 8-tuples: (sched, restore, exec_launch, grammar,
    # forward_wait, aborts, update, post)
    "samples": [],
    "dumped": False,
}


def _ft_dump_step_timing() -> None:
    """P0-impl-3a: dump per-section wall timing summary.

    Sample tuple schema for step_with_batch_queue:
        (total, prep, pop_and_wait, post, flag, _, _, _)
    where flag = -1.0 (enqueue path) or 1.0 (dequeue path)
    """
    _state = _ft_step_timing_state
    samples = _state["samples"]
    n = len(samples)
    # Add per-process suffix so dp workers don't overwrite each other
    pid = _os.getpid()
    output_path = f"{_state['output_path']}.pid{pid}"

    enqueue_samples = [s for s in samples if s[4] < 0]
    dequeue_samples = [s for s in samples if s[4] > 0]

    def _stats(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return (0.0, 0.0, 0.0)
        sorted_v = sorted(values)
        avg = sum(values) / len(values) * 1000
        p50 = sorted_v[len(values) // 2] * 1000
        p95_idx = min(len(values) - 1, int(len(values) * 0.95))
        p95 = sorted_v[p95_idx] * 1000
        return (avg, p50, p95)

    try:
        with open(output_path, "w") as f:
            f.write(
                f"FT_STEP_TIMING: captured {n} step_with_batch_queue() calls "
                f"(pid={pid})\n"
                f"  enqueue path (no GPU wait): {len(enqueue_samples)}\n"
                f"  dequeue path (GPU wait):    {len(dequeue_samples)}\n"
                f"\n"
            )
            f.write("=== Enqueue path: total call time ===\n")
            if enqueue_samples:
                a, p50, p95 = _stats([s[0] for s in enqueue_samples])
                f.write(f"  avg={a:.3f}ms  p50={p50:.3f}ms  p95={p95:.3f}ms\n")
            else:
                f.write("  (none)\n")
            f.write("\n=== Dequeue path: per-section breakdown ===\n")
            if dequeue_samples:
                section_names = [
                    "1. total call",
                    "2. enqueue-side prep (sched + exec launch)",
                    "3. pop + future.result() — GPU forward wait",
                    "4. update_from_output + deferred handling",
                ]
                f.write(
                    f"{'Section':<48} {'avg':>10} {'p50':>10} {'p95':>10}\n"
                )
                for i, name in enumerate(section_names):
                    a, p50, p95 = _stats([s[i] for s in dequeue_samples])
                    f.write(
                        f"{name:<48} {a:>9.3f}ms {p50:>9.3f}ms {p95:>9.3f}ms\n"
                    )
                # Verify section 1 = sum of 2..4 (sanity)
                a1, _, _ = _stats([s[0] for s in dequeue_samples])
                a234 = sum(_stats([s[i] for s in dequeue_samples])[0] for i in (1, 2, 3))
                f.write(f"\n  sanity: section 1 ({a1:.3f}) ≈ sum 2..4 ({a234:.3f})\n")
            else:
                f.write("  (none)\n")
        logger.info(
            "FT_STEP_TIMING: dumped %d samples to %s "
            "(enqueue=%d, dequeue=%d)",
            n, output_path, len(enqueue_samples), len(dequeue_samples),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("FT_STEP_TIMING dump failed: %s", exc)


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()
        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        # ── Solution 4: lazy KV reload gate ───────────────────────────
        # If lazy reload is enabled and this is a rerouted request from
        # a failed engine, defer admission. The drain hook in step()
        # will promote it later when the surviving engine has capacity.
        if (
            _FT_LAZY_RELOAD_ENABLED
            and getattr(request, "is_rerouted", False)
        ):
            if not hasattr(self, "_ft_lazy_pending"):
                self._ft_lazy_pending: deque[Request] = deque()
            self._ft_lazy_pending.append(request)
            logger.info(
                "FT_LAZY_RELOAD: deferred rerouted req=%s "
                "(lazy queue depth=%d, cap=%d)",
                request.request_id,
                len(self._ft_lazy_pending),
                _FT_LAZY_MAX_CONCURRENT,
            )
            return

        self._admit_to_scheduler(request)

    def _admit_to_scheduler(self, request: Request) -> None:
        """Actually push the request into the scheduler queue + register
        it for FT KV restore. Extracted from add_request() so the
        Solution-4 lazy drain path can call it directly without going
        back through the gate.
        """
        self.scheduler.add_request(request)

        # FT: if the request carries checkpoint info from a failed engine,
        # queue it for KV restore after blocks are allocated.
        #
        # We do NOT set num_computed_tokens here.  The first schedule()
        # treats this as a fresh request: allocates blocks for all tokens
        # and schedules full computation.  After schedule() allocates
        # blocks, _process_ft_pending_restores() restores checkpoint KV
        # into those blocks AND patches scheduler_output to reduce the
        # scheduled tokens, so the model runner skips the restored prefix
        # and only computes the remaining tokens.  If restore fails, the
        # scheduler_output is left unchanged and the request does a full
        # recompute — always correct.
        if getattr(request, "num_checkpointed_tokens", 0) > 0:
            if not hasattr(self, "_ft_pending_restores"):
                self._ft_pending_restores: list[
                    tuple[str, int]
                ] = []
            self._ft_pending_restores.append(
                (request.request_id, request.num_checkpointed_tokens)
            )

        # Solution 4: prepend rerouted requests to the front of the
        # base waiting queue so they are scheduled before any new
        # arrivals that queued up after the fault. For FCFS this is a
        # real prepend; for priority queues prepend_request() falls
        # back to add (no-op since priority field on the request will
        # already place it ahead of new arrivals).
        if getattr(request, "is_rerouted", False):
            base = getattr(self.scheduler, "_base", self.scheduler)
            waiting = getattr(base, "waiting", None)
            if waiting is not None and hasattr(
                waiting, "prepend_request"
            ):
                try:
                    waiting.remove_request(request)
                    waiting.prepend_request(request)
                except (ValueError, KeyError, IndexError):
                    # remove may fail if the scheduler chose to admit
                    # the request directly to running (rare). In that
                    # case there is nothing to prepend.
                    pass

    def _drain_slo_preempted_restores(self) -> None:
        """M3 drain hook: migrate scheduler-side SLO-preempted restore
        entries into the engine's _ft_pending_restores queue.

        Scheduler.preempt_for_slo() appends (req_id, num_checkpointed_tokens)
        tuples to scheduler.slo_preempted_pending_restore. This method drains
        them into the engine's _ft_pending_restores list, which is processed
        by _process_ft_pending_restores after schedule() — same path used for
        cross-engine fault recovery. The result is checkpoint-based restore
        when the preempted req is later re-admitted.

        No-op when FT_SLO_PREEMPT is off (list will be empty).
        """
        base = getattr(self.scheduler, "_base", self.scheduler)
        pending = getattr(base, "slo_preempted_pending_restore", None)
        if not pending:
            return
        if not hasattr(self, "_ft_pending_restores"):
            self._ft_pending_restores: list[tuple[str, int]] = []
        self._ft_pending_restores.extend(pending)
        pending.clear()

    def _drain_ft_lazy_pending(self) -> None:
        """Solution 4 drain hook. Promotes rerouted requests from the
        lazy holding deque into the scheduler when the count of
        currently-running rerouted requests is below the cap. Called
        from step() / step_with_batch_queue() before schedule().
        """
        pending = getattr(self, "_ft_lazy_pending", None)
        if not pending:
            return
        base = getattr(self.scheduler, "_base", self.scheduler)
        running_iter = getattr(base, "running", None)
        if running_iter is None:
            return
        rerouted_running = sum(
            1 for r in running_iter
            if getattr(r, "is_rerouted", False)
        )
        drained = 0
        while pending and rerouted_running < _FT_LAZY_MAX_CONCURRENT:
            request = pending.popleft()
            self._admit_to_scheduler(request)
            rerouted_running += 1
            drained += 1
        if drained > 0:
            logger.info(
                "FT_LAZY_RELOAD: drained %d reqs "
                "(rerouted_running=%d/%d, lazy_queued=%d)",
                drained,
                rerouted_running,
                _FT_LAZY_MAX_CONCURRENT,
                len(pending),
            )

    def _ft_prebudget_pending_restores(self) -> None:
        """Pre-set num_computed_tokens for pending-restore reqs BEFORE
        scheduler.schedule() runs. This tricks the scheduler into thinking
        these reqs are mostly-computed (just need 1 more decode token),
        so it admits ALL of them in a single step instead of 1-2 per step
        (prefill budget limit). Recovery finishes in one big step instead
        of fragmenting into many 370ms steps.

        Enabled via FT_RECOVERY_PREBUDGET=1. Partial/failed restores are
        repaired by _process_ft_pending_restores (it bumps num_scheduled_tokens
        back up to cover any replay gap).
        """
        if os.environ.get("FT_RECOVERY_PREBUDGET") != "1":
            return
        if not hasattr(self, "_ft_pending_restores") or not self._ft_pending_restores:
            return
        if not hasattr(self.scheduler, "ft_scheduler"):
            return
        for req_id, num_ckpt_tokens in self._ft_pending_restores:
            if num_ckpt_tokens <= 0:
                continue
            try:
                request = self.scheduler._base.requests.get(req_id)
            except (AttributeError, KeyError):
                continue
            if request is None:
                continue
            # Lie to scheduler — mark as mostly computed.
            request.num_computed_tokens = num_ckpt_tokens

    def _ft_will_checkpoint_fire_next_step(self) -> bool:
        """Predict whether the next step will trigger a checkpoint save.

        Used by FT_CKPT_FIRE_BUDGET_RATIO to pre-shrink scheduler budget
        before the step that will be heavier.

        Implementation: step-counter based ONLY. Requires
        FT_CKPT_STEP_INTERVAL > 1 (the counter mechanism throttles fire
        to every K-th step, so prediction is exact).

        When FT_CKPT_STEP_INTERVAL is unset or ≤ 1, every step may fire
        based on per-req block-boundary state — we cannot predict that
        cheaply without side-effects (calling get_requests_to_checkpoint
        mutates _last_evaluated_stable_tokens / _step_time_ema). In that
        regime the feature is disabled (returns False → no budget
        reduction → no harm).
        """
        if os.environ.get("FT_CKPT_FIRE_BUDGET_RATIO") is None:
            return False
        try:
            interval = int(os.environ.get("FT_CKPT_STEP_INTERVAL", "0"))
        except ValueError:
            interval = 0
        if interval <= 1:
            # Cannot cheaply predict without side-effects. Disabled.
            return False
        # Reuse the real fire counter (incremented in _ft_post_process).
        # Next step count = current + 1; fires when divisible by interval.
        step_ct = getattr(self, "_ft_ckpt_step_counter", 0) + 1
        return (step_ct % interval) == 0

    def _ft_apply_ckpt_aware_budget(self) -> int | None:
        """Temporarily shrink scheduler token budget if checkpoint will
        fire this step. Returns the original budget for later restore,
        or None if no override was applied.
        """
        ratio_str = os.environ.get("FT_CKPT_FIRE_BUDGET_RATIO")
        if not ratio_str:
            return None
        try:
            ratio = float(ratio_str)
        except ValueError:
            return None
        if not (0.0 < ratio < 1.0):
            return None
        if not self._ft_will_checkpoint_fire_next_step():
            return None
        sched = getattr(self.scheduler, "_base", self.scheduler)
        saved = getattr(sched, "max_num_scheduled_tokens", None)
        if saved is None:
            return None
        sched.max_num_scheduled_tokens = max(1, int(saved * ratio))
        return saved

    def _ft_restore_ckpt_aware_budget(self, saved: int | None) -> None:
        """Restore scheduler budget after schedule() runs. No-op if no
        override was applied.
        """
        if saved is None:
            return
        sched = getattr(self.scheduler, "_base", self.scheduler)
        sched.max_num_scheduled_tokens = saved

    def _process_ft_pending_restores(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Restore checkpoint KV into allocated blocks and patch scheduler_output.

        Called in step() after schedule() but BEFORE execute_model().
        For each pending restore:
          1. Copy checkpoint KV from host memory into the GPU blocks
             that schedule() just allocated.
          2. Update request.num_computed_tokens and
             request.num_checkpointed_tokens to reflect the restore.
          3. Patch scheduler_output so the model runner skips the
             restored prefix: reduce num_scheduled_tokens and update
             NewRequestData.num_computed_tokens.

        If restore fails, the scheduler_output is left unchanged and the
        request does a full recompute — always correct.

        Requests whose blocks are not yet allocated (still in waiting
        queue) are kept for the next step.

        Phase 2 C-mode (FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1):
          - Newly-enqueued restores use sync=False; this step's forward
            for the recovering req is skipped (num_scheduled_tokens=0).
          - Per-step we query whether prior in-flight restores have
            completed; only when complete do we admit the req's forward
            and patch num_computed_tokens.
          - This achieves true overlap: copy stream runs reload while
            default stream concurrently runs forward of OTHER reqs.
        """
        # When overlap is enabled, V3 capacity-preempted reqs are
        # parked in the waiting queue with WAITING_FOR_RELOAD status,
        # and _process_overlap_reload_queue() drives their reload state
        # machine from step() (called BEFORE schedule()). They never
        # enter _ft_pending_restores, so this function is a no-op for
        # the V3 path — return early.
        use_overlap = (
            os.environ.get("FT_CAPACITY_PREEMPT_RELOAD_OVERLAP") == "1"
        )
        if use_overlap:
            return

        if not hasattr(self, "_ft_pending_restores") or not self._ft_pending_restores:
            return

        if not hasattr(self.scheduler, "ft_scheduler"):
            self._ft_pending_restores.clear()
            return

        kv_cache_mgr = self.scheduler._base.kv_cache_manager
        still_pending: list[tuple[str, int]] = []

        # FT_ASYNC_RESTORE: when set, all restore_kv_blocks RPCs in this
        # step enqueue onto a shared worker-side CUDA stream without
        # per-call sync. We flush the stream once at the end of the
        # loop, before execute_model reads the restored KV. This turns
        # N sequential (~5 ms each) syncs into 1, and lets the layer
        # copies overlap across requests.
        use_async_restore = os.environ.get("FT_ASYNC_RESTORE") == "1"
        async_restore_triggered = False
        # FT_RECOVERY_PREBUDGET: whether num_computed_tokens was pre-set
        # (lied to scheduler) so this method must handle partial/failed
        # restores by bumping num_scheduled_tokens back up.
        prebudget_active = os.environ.get("FT_RECOVERY_PREBUDGET") == "1"

        # Build a lookup for NewRequestData so we can patch it.
        new_req_data_by_id = {
            nrd.req_id: nrd
            for nrd in scheduler_output.scheduled_new_reqs
        }

        # FT_RESTORE_BATCH_RPC=1: combine N per-request restore_kv_blocks
        # RPCs into a single batched RPC. Saves N-1 collective_rpc round
        # trips (~50-100us each) and reduces Python orchestration overhead
        # in the worker. All restores share the async stream when
        # FT_ASYNC_RESTORE=1, so the single flush at the end synchronizes
        # everything (same semantics as per-req loop, fewer cross-process
        # round-trips).
        batched_rpc = (
            os.environ.get("FT_RESTORE_BATCH_RPC") == "1"
            and use_async_restore
        )

        # First pass: collect specs for reqs that have target blocks
        # allocated; defer those still in waiting queue.
        restore_specs: list[tuple[str, list[int], int]] = []
        for req_id, num_ckpt_tokens in self._ft_pending_restores:
            try:
                target_block_ids = self._get_ft_target_block_ids(
                    req_id, kv_cache_mgr
                )
            except Exception:
                target_block_ids = []
            if not target_block_ids:
                still_pending.append((req_id, num_ckpt_tokens))
                continue
            restore_specs.append((req_id, target_block_ids, num_ckpt_tokens))

        # Fire restores. Batched path = one RPC for all reqs.
        per_req_results: dict[str, int] = {}
        if batched_rpc and restore_specs:
            specs_only = [(rid, tbids) for rid, tbids, _ in restore_specs]
            try:
                rpc_results = self.collective_rpc(
                    "restore_kv_blocks_batch",
                    args=(specs_only, False),
                )
                async_restore_triggered = True
                if rpc_results and rpc_results[0]:
                    counts = rpc_results[0]
                    for (rid, _, _), tokens in zip(restore_specs, counts):
                        per_req_results[rid] = tokens
            except Exception:
                logger.exception("Batched restore RPC failed; falling back to per-request")
                batched_rpc = False  # fall through to per-req path

        for req_id, target_block_ids, num_ckpt_tokens in restore_specs:
            try:
                # Get the restore result (from batch or per-req call)
                if batched_rpc:
                    tokens_restored = per_req_results.get(req_id, 0)
                    results = [tokens_restored]
                elif use_async_restore:
                    results = self.collective_rpc(
                        "restore_kv_blocks",
                        args=(req_id, target_block_ids, False),
                    )
                    async_restore_triggered = True
                else:
                    results = self.collective_rpc(
                        "restore_kv_blocks",
                        args=(req_id, target_block_ids),
                    )

                if results and results[0] and results[0] > 0:
                    tokens_restored = results[0]

                    # Update request metadata.
                    request = self.scheduler._base.requests.get(req_id)
                    if request is not None:
                        request.num_computed_tokens = tokens_restored
                        request.num_checkpointed_tokens = tokens_restored

                    # Patch scheduler_output so the model runner skips
                    # the restored prefix instead of recomputing it.
                    old_scheduled = scheduler_output.num_scheduled_tokens.get(
                        req_id
                    )
                    if old_scheduled is not None and old_scheduled > tokens_restored:
                        # Normal path (no pre-budget, or pre-budget with
                        # full restore): scheduler allocated full prefill,
                        # we now subtract the restored prefix.
                        new_scheduled = old_scheduled - tokens_restored
                        scheduler_output.num_scheduled_tokens[req_id] = (
                            new_scheduled
                        )
                        scheduler_output.total_num_scheduled_tokens -= (
                            tokens_restored
                        )

                        # Also patch NewRequestData.num_computed_tokens
                        # so the model runner positions start from the
                        # restored offset.
                        nrd = new_req_data_by_id.get(req_id)
                        if nrd is not None:
                            nrd.num_computed_tokens = tokens_restored
                    elif (
                        prebudget_active
                        and old_scheduled is not None
                        and tokens_restored < num_ckpt_tokens
                    ):
                        # Pre-budget + partial restore: restore returned
                        # FEWER tokens than promised. Set num_computed to
                        # the actual restored value so subsequent steps
                        # see the correct state. DO NOT bump
                        # num_scheduled_tokens for this step — that would
                        # exceed max_num_batched_tokens (scheduler had
                        # already filled the budget assuming this req
                        # only needed 1 token). Crashed with
                        # ValueError: shapes (X,) (X,) (max_budget,) in
                        # gpu_model_runner._prepare_inputs (2026-04-14).
                        # The req decodes 1 token from the restored
                        # prefix this step; subsequent steps will pick
                        # up the gap normally via standard scheduling.
                        nrd = new_req_data_by_id.get(req_id)
                        if nrd is not None:
                            nrd.num_computed_tokens = tokens_restored

                    logger.info(
                        "FT restore: request %s restored %d tokens "
                        "from checkpoint, skipping prefill for "
                        "restored prefix (%d → %d scheduled tokens)",
                        req_id,
                        tokens_restored,
                        old_scheduled or 0,
                        scheduler_output.num_scheduled_tokens.get(
                            req_id, 0
                        ),
                    )
                else:
                    # Restore failed (returned 0 or None). If pre-budget
                    # was active, num_computed_tokens was set to num_ckpt
                    # but no KV was actually restored → must NOT let the
                    # model decode at the lying position with garbage KV.
                    #
                    # Safer rollback: reset num_computed_tokens=0 so the
                    # next scheduler step does fresh prefill from scratch.
                    # DO NOT bump scheduled_tokens this step — that
                    # exceeds max_num_batched_tokens and crashes
                    # _prepare_inputs (see partial-restore comment above).
                    # Set num_scheduled_tokens=0 to skip the req this
                    # step entirely.
                    if prebudget_active:
                        request = self.scheduler._base.requests.get(req_id)
                        if request is not None:
                            request.num_computed_tokens = 0
                        nrd = new_req_data_by_id.get(req_id)
                        old_scheduled = (
                            scheduler_output.num_scheduled_tokens.get(
                                req_id, 0
                            )
                        )
                        if nrd is not None:
                            nrd.num_computed_tokens = 0
                        # Skip this step; total -= old_scheduled (give
                        # the slot back to the budget).
                        scheduler_output.num_scheduled_tokens[req_id] = 0
                        scheduler_output.total_num_scheduled_tokens -= (
                            old_scheduled
                        )
                    logger.info(
                        "FT restore: request %s checkpoint not found "
                        "or empty, will recompute fully",
                        req_id,
                    )
                # Don't retry regardless of outcome.
            except Exception:
                logger.debug(
                    "FT restore failed for request %s (will recompute)",
                    req_id,
                )

        # FT_ASYNC_RESTORE: flush the shared restore stream exactly once
        # before execute_model consumes the restored KV. Skipped if no
        # async restore was triggered (e.g. still_pending on every req).
        if use_async_restore and async_restore_triggered:
            try:
                self.collective_rpc("flush_pending_restore")
            except Exception:
                logger.exception(
                    "FT async restore flush failed — falling back to "
                    "per-call sync on next step"
                )

        self._ft_pending_restores = still_pending

    def _process_slo_retained_queue(self) -> None:
        """Phase 2 SLO preempt retain path: tick down per-request
        retain_steps and drain expired entries back into vLLM waiting
        queue.

        Called every step BEFORE scheduler.schedule(). For each
        retained request:
          - Decrement retain_steps_remaining
          - When it reaches 0, prepend the request to vLLM waiting
            queue. Standard admit picks it up next step.
          - The request's blocks were never freed and KV is intact, so
            vLLM admit sees existing blocks + non-zero
            num_computed_tokens, and forward continues from where it
            stopped (zero reload cost).

        This implementation is the simple version: fixed-step retain
        without GPU memory pressure detection. Adding LRU eviction +
        memory-pressure guards is future work.
        """
        if os.environ.get("SLO_PRIORITY_PREEMPT") != "1":
            return
        base = getattr(self.scheduler, "_base", self.scheduler)
        retained = getattr(base, "slo_preempted_retained", None)
        if not retained:
            return
        # Tick down each entry's retain_steps and drain those that
        # reached 0.
        # Defensive: between preempt and resume, the request may have
        # transitioned to a FINISHED_* status (e.g. client disconnect,
        # request_timeout, abort). Re-adding such a request to the
        # waiting queue causes vLLM's schedule() to raise
        # "Invalid request status: FINISHED_*". Skip those.
        from vllm.v1.request import RequestStatus
        new_retained: list[tuple] = []
        for request, steps_remaining in retained:
            steps_remaining -= 1
            if request.status != RequestStatus.PREEMPTED:
                # Stale entry — request was finished or aborted while
                # in the retained queue. Drop it (no prepend).
                logger.info(
                    "FT SLO retain: %s dropped from retain queue "
                    "(status=%s, no resume)",
                    request.request_id, request.status,
                )
                continue
            if steps_remaining <= 0:
                base.waiting.prepend_request(request)
                logger.info(
                    "FT SLO retain: %s resumed (KV preserved on "
                    "GPU, num_computed_tokens=%d)",
                    request.request_id,
                    getattr(request, "num_computed_tokens", 0),
                )
            else:
                new_retained.append((request, steps_remaining))
        base.slo_preempted_retained = new_retained

    def _process_overlap_reload_queue(self) -> None:
        """V3 capacity-preempt reload driver.

        The req sits in vLLM's waiting queue with status=
        WAITING_FOR_RELOAD throughout this state machine; we don't
        manage queue membership here. We only drive:
          waiting_for_blocks → reloading → done
        and once done we patch num_computed_tokens + flip status to
        PREEMPTED. vLLM's standard waiting-loop admit (resumed_req_ids
        path) takes over from there.

        Called every step BEFORE scheduler.schedule().
        """
        use_overlap = (
            os.environ.get("FT_CAPACITY_PREEMPT_RELOAD_OVERLAP") == "1"
        )
        if not use_overlap:
            return

        base = getattr(self.scheduler, "_base", self.scheduler)

        if not hasattr(self, "_overlap_reload_inflight"):
            self._overlap_reload_inflight: dict[str, dict] = {}

        # Drain new entries from scheduler-side handoff list.
        pending_list = getattr(
            base, "slo_preempted_for_overlap_reload", None
        )
        if pending_list:
            for request, ckpt_tokens in list(pending_list):
                req_id = request.request_id
                self._overlap_reload_inflight[req_id] = {
                    "request": request,
                    "ckpt_tokens": ckpt_tokens,
                    "state": "waiting_for_blocks",
                    "steps_waited": 0,
                    "enqueue_time": time.time(),
                }
                logger.info(
                    "FT overlap V3: %s queued, waiting for "
                    "%d-token block allocation",
                    req_id, ckpt_tokens,
                )
            pending_list.clear()

        if not self._overlap_reload_inflight:
            return

        kv_cache_mgr = base.kv_cache_manager

        # Stage A: try alloc for waiting_for_blocks entries.
        for req_id, state in list(
            self._overlap_reload_inflight.items()
        ):
            if state["state"] != "waiting_for_blocks":
                continue
            request = state["request"]
            ckpt_tokens = state["ckpt_tokens"]
            try:
                kv_blocks = kv_cache_mgr.allocate_slots(
                    request,
                    num_new_tokens=ckpt_tokens,
                )
            except Exception:
                logger.exception(
                    "FT overlap V3: allocate_slots crashed for %s",
                    req_id,
                )
                kv_blocks = None

            if kv_blocks is None:
                state["steps_waited"] += 1
                continue

            try:
                block_ids_tuple = kv_cache_mgr.get_block_ids(req_id)
            except Exception:
                logger.exception(
                    "FT overlap V3: get_block_ids failed for %s "
                    "after allocate_slots succeeded",
                    req_id,
                )
                state["steps_waited"] += 1
                continue
            target_block_ids: list[int] = []
            for grp in block_ids_tuple:
                target_block_ids.extend(int(b) for b in grp)
            if not target_block_ids:
                state["steps_waited"] += 1
                continue

            # NOTE: V3 debug logging removed for race-reproduction
            # testing. Even a single light-weight log line here adds
            # enough latency to potentially mask the race we are trying
            # to reproduce. Restore log if needed for diagnosis.

            try:
                results = self.collective_rpc(
                    "restore_kv_blocks",
                    args=(req_id, target_block_ids, False),  # sync=False
                )
                tokens_started = (
                    results[0] if results and results[0] else 0
                )
            except Exception:
                logger.exception(
                    "FT overlap V3: restore RPC failed for %s",
                    req_id,
                )
                tokens_started = 0

            if tokens_started <= 0:
                # No checkpoint or RPC failed. Drop the alloc'd blocks
                # and fall back to full reprefill: vLLM admit will see
                # PREEMPTED with num_computed_tokens=0 and treat it
                # like a vanilla preempt resume.
                try:
                    kv_cache_mgr.free(request)
                except Exception:
                    pass
                request.num_computed_tokens = 0
                request.status = RequestStatus.PREEMPTED
                logger.info(
                    "FT overlap V3: %s reload skipped (no ckpt); "
                    "fell back to full reprefill",
                    req_id,
                )
                del self._overlap_reload_inflight[req_id]
                continue

            state["state"] = "reloading"
            state["tokens_started"] = tokens_started
            state["reload_started_step"] = state.get(
                "steps_waited", 0
            )
            logger.info(
                "FT overlap V3: %s alloc'd %d blocks (after "
                "%d step(s) wait), reload started",
                req_id, len(target_block_ids),
                state.get("steps_waited", 0),
            )

        # Stage B: query reloading entries; flip status when done.
        for req_id, state in list(
            self._overlap_reload_inflight.items()
        ):
            if state["state"] != "reloading":
                continue
            steps_in_reload = state.get("steps_in_reload", 0) + 1
            state["steps_in_reload"] = steps_in_reload
            try:
                results = self.collective_rpc(
                    "query_restore_done",
                    args=(req_id, steps_in_reload),
                )
                done = bool(results[0]) if results else True
            except Exception:
                logger.exception(
                    "FT overlap V3: query failed for %s "
                    "(treating as not-done)",
                    req_id,
                )
                done = False

            if not done:
                continue

            # Reload done — patch request state and flip status.
            # vLLM waiting loop will admit it via the resumed path
            # next iteration.
            request = state["request"]
            tokens_started = state["tokens_started"]
            request.num_computed_tokens = tokens_started
            request.num_checkpointed_tokens = tokens_started
            request.status = RequestStatus.PREEMPTED
            logger.info(
                "FT overlap V3: %s done — alloc waited %d step(s), "
                "reload took %d step(s), %d tokens restored",
                req_id,
                state.get("steps_waited", 0),
                steps_in_reload,
                tokens_started,
            )
            del self._overlap_reload_inflight[req_id]

    def _process_ft_pending_restores_overlap(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Phase 2 C-mode: async restore with same-step forward of OTHER reqs.

        Per-step flow:
          Stage 1: Check `_ft_inflight_restores` (started in prior steps).
            For each: query worker via collective_rpc("query_restore_done").
            - Done: patch scheduler_output to admit this req's forward
              this step (set num_computed_tokens, subtract restored prefix
              from num_scheduled_tokens). Remove from inflight dict.
            - Not done: set num_scheduled_tokens=0 for this req (skip its
              forward this step). Increment steps_waited. Keep in inflight.
          Stage 2: Process newly-arrived `_ft_pending_restores`.
            Enqueue async via restore_kv_blocks(sync=False). Add to
            inflight dict. Set num_scheduled_tokens=0 for this req
            (forward this step skips it; copy stream runs reload while
            default stream forwards the OTHER reqs).

        The CSV row for each reload is written by the worker side when
        query_async_restore reports done (with steps_to_complete +
        wait_ms_total).
        """
        if not hasattr(self, "_ft_inflight_restores"):
            self._ft_inflight_restores: dict[str, dict] = {}

        # Build NewRequestData lookup once for both stages.
        new_req_data_by_id = {
            nrd.req_id: nrd
            for nrd in scheduler_output.scheduled_new_reqs
        }

        # ── Stage 1: Check existing in-flight restores ─────────────────
        new_inflight: dict[str, dict] = {}
        for req_id, state in list(self._ft_inflight_restores.items()):
            steps_waited = state.get("steps_waited", 1) + 1
            try:
                results = self.collective_rpc(
                    "query_restore_done",
                    args=(req_id, steps_waited),
                )
                done = bool(results[0]) if results else True
            except Exception:
                logger.exception(
                    "FT overlap: query_restore_done failed for %s "
                    "(treating as not-done)",
                    req_id,
                )
                done = False

            if done:
                # Reload finished. Patch scheduler_output to admit it.
                tokens_restored = state["tokens_restored"]
                request = self.scheduler._base.requests.get(req_id)
                if request is not None:
                    request.num_computed_tokens = tokens_restored
                    request.num_checkpointed_tokens = tokens_restored
                old_scheduled = (
                    scheduler_output.num_scheduled_tokens.get(req_id)
                )
                if (
                    old_scheduled is not None
                    and old_scheduled > tokens_restored
                ):
                    new_scheduled = old_scheduled - tokens_restored
                    scheduler_output.num_scheduled_tokens[req_id] = (
                        new_scheduled
                    )
                    scheduler_output.total_num_scheduled_tokens -= (
                        tokens_restored
                    )
                    nrd = new_req_data_by_id.get(req_id)
                    if nrd is not None:
                        nrd.num_computed_tokens = tokens_restored
                logger.info(
                    "FT overlap restore done: %s after %d step(s), "
                    "%d tokens restored",
                    req_id, steps_waited, tokens_restored,
                )
            else:
                # Reload not done yet. Remove X completely from this
                # step's batch (not just num_sched=0). Otherwise model
                # runner still sees X in scheduled_new_reqs /
                # scheduled_running_reqs and the batch token count
                # won't match its expected shape.
                self._ft_overlap_remove_from_batch(
                    scheduler_output, req_id
                )
                state["steps_waited"] = steps_waited
                new_inflight[req_id] = state
        self._ft_inflight_restores = new_inflight

        # ── Stage 2: Process newly-arrived pending restores ────────────
        if (
            not hasattr(self, "_ft_pending_restores")
            or not self._ft_pending_restores
        ):
            return
        if not hasattr(self.scheduler, "ft_scheduler"):
            self._ft_pending_restores.clear()
            return

        kv_cache_mgr = self.scheduler._base.kv_cache_manager
        still_pending: list[tuple[str, int]] = []

        for req_id, num_ckpt_tokens in self._ft_pending_restores:
            try:
                target_block_ids = self._get_ft_target_block_ids(
                    req_id, kv_cache_mgr
                )
            except Exception:
                target_block_ids = []
            if not target_block_ids:
                still_pending.append((req_id, num_ckpt_tokens))
                continue

            # Enqueue async restore (sync=False). Worker enqueues the
            # copy on copy stream and returns immediately.
            try:
                results = self.collective_rpc(
                    "restore_kv_blocks",
                    args=(req_id, target_block_ids, False),
                )
                tokens_restored = (
                    results[0] if results and results[0] else 0
                )
            except Exception:
                logger.exception(
                    "FT overlap: restore_kv_blocks(sync=False) failed "
                    "for %s; falling back to recompute next step",
                    req_id,
                )
                tokens_restored = 0

            if tokens_restored <= 0:
                # Restore failed (no checkpoint or RPC error). Don't
                # add to inflight. Scheduler_output stays as-is, model
                # runner does full prefill normally.
                logger.info(
                    "FT overlap: %s restore returned 0; "
                    "will recompute fully",
                    req_id,
                )
                continue

            # Remove X from this step's batch entirely (not just
            # num_sched=0; see comment in Stage 1 above).
            self._ft_overlap_remove_from_batch(
                scheduler_output, req_id
            )

            # Track for next step's query.
            self._ft_inflight_restores[req_id] = {
                "tokens_restored": tokens_restored,
                "num_ckpt_tokens": num_ckpt_tokens,
                "steps_waited": 1,
                "enqueue_time": time.time(),
            }
            logger.info(
                "FT overlap restore enqueued: %s, %d tokens, "
                "deferring forward to next step",
                req_id, tokens_restored,
            )

        self._ft_pending_restores = still_pending

    def _ft_overlap_remove_from_batch(
        self,
        scheduler_output: "SchedulerOutput",
        req_id: str,
    ) -> None:
        """Remove a request entirely from this step's forward batch.

        Used by overlap mode when reload hasn't completed yet — we
        mustn't let model_runner try to forward this req with stale or
        partial KV. Setting num_scheduled_tokens=0 isn't enough because
        the req is still in scheduled_new_reqs / scheduled_running_reqs
        lists, causing batch shape inconsistencies.

        After removal, scheduler will re-add the req in a future step
        (it's still in the running/waiting queue with status PREEMPTED).
        """
        old = scheduler_output.num_scheduled_tokens.pop(req_id, 0)
        if old > 0:
            scheduler_output.total_num_scheduled_tokens -= old

        # Remove from scheduled_new_reqs (list of NewRequestData).
        if hasattr(scheduler_output, "scheduled_new_reqs"):
            scheduler_output.scheduled_new_reqs = [
                nrd for nrd in scheduler_output.scheduled_new_reqs
                if getattr(nrd, "req_id", None) != req_id
            ]

        # Remove from scheduled_running_reqs if present.
        if hasattr(scheduler_output, "scheduled_running_reqs"):
            scheduler_output.scheduled_running_reqs = [
                rrd for rrd in scheduler_output.scheduled_running_reqs
                if getattr(rrd, "req_id", None) != req_id
            ]

        # Some vLLM versions also have scheduled_resumed_reqs.
        if hasattr(scheduler_output, "scheduled_resumed_reqs"):
            scheduler_output.scheduled_resumed_reqs = [
                r for r in scheduler_output.scheduled_resumed_reqs
                if getattr(r, "req_id", None) != req_id
            ]

    def _get_ft_target_block_ids(
        self,
        request_id: str,
        kv_cache_mgr: "KVCacheManager",
    ) -> list[int]:
        """Resolve the allocated KV block IDs for an FT restore target."""
        try:
            all_ids = kv_cache_mgr.get_block_ids(request_id)
            if all_ids:
                return list(all_ids[0])  # First KV cache group.
        except (AttributeError, KeyError, TypeError):
            pass

        # Fall back to older/specialized managers that expose raw blocks.
        try:
            blocks = kv_cache_mgr.req_to_blocks.get(request_id)
            if blocks:
                return [blk.block_id for blk in blocks]
        except (AttributeError, KeyError, TypeError):
            pass
        return []

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput):
        if not self.vllm_config.observability_config.enable_logging_iteration_details:
            yield
            return
        self._iteration_index = getattr(self, "_iteration_index", 0)
        iteration_details = compute_iteration_details(scheduler_output)
        before = time.monotonic()
        yield
        logger.info(
            "".join(
                [
                    "Iteration(",
                    str(self._iteration_index),
                    "): ",
                    str(iteration_details.num_ctx_requests),
                    " context requests, ",
                    str(iteration_details.num_ctx_tokens),
                    " context tokens, ",
                    str(iteration_details.num_generation_requests),
                    " generation requests, ",
                    str(iteration_details.num_generation_tokens),
                    " generation tokens, iteration elapsed time: ",
                    format((time.monotonic() - before) * 1000, ".2f"),
                    " ms",
                ]
            )
        )
        self._iteration_index += 1

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Solution 4: drain pending rerouted requests into the scheduler
        # if there is capacity. No-op when FT_LAZY_RELOAD is off.
        if _FT_LAZY_RELOAD_ENABLED:
            self._drain_ft_lazy_pending()

        # M3 SLO-aware preemption: drain scheduler-side pending restores
        # (entries from _preempt_for_slo) into the engine's _ft_pending_restores
        # queue so they are processed after schedule() like cross-engine
        # rerouted reqs. No-op when FT_SLO_PREEMPT is off.
        self._drain_slo_preempted_restores()

        # Phase 2 C-mode (queue-based): manage overlap reload side queue.
        # Starts new async reloads on copy stream + checks completion of
        # in-flight ones; admits completed ones into vLLM waiting queue.
        # No-op when FT_CAPACITY_PREEMPT_RELOAD_OVERLAP is off.
        self._process_overlap_reload_queue()

        # Phase 2 SLO retain path: tick down retained requests and
        # resume those whose retain window expired. No-op when
        # FT_SLO_PREEMPT_RETAIN is off.
        self._process_slo_retained_queue()

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False

        # ── P0-impl-3a: per-section wall timing ──
        _ts_state = _ft_step_timing_state
        _ts_active = (
            _ts_state["max_calls"] > 0
            and _ts_state["calls"] < _ts_state["max_calls"]
        )
        if _ts_active:
            import time as _time
            _t0 = _time.perf_counter()

        # FT_RECOVERY_PREBUDGET: lie to scheduler about pending-restore
        # reqs' num_computed_tokens BEFORE schedule() runs. Normally the
        # scheduler treats these as fresh (full prefill) reqs and only
        # admits 1-2 per step (prefill budget limit), fragmenting
        # recovery into multiple 370ms steps. By pre-setting
        # num_computed_tokens = num_ckpt_tokens, scheduler sees them as
        # "mostly-computed, just need 1 decode", admits all in one step.
        # _process_ft_pending_restores will patch num_computed_tokens +
        # num_scheduled_tokens back if the actual restore was partial.
        self._ft_prebudget_pending_restores()

        # FT_CKPT_FIRE_BUDGET_RATIO: when a checkpoint fire is predicted
        # this step, temporarily shrink scheduler's token budget so it
        # under-admits (leaving headroom for the ~5-10ms extra step cost
        # of gather+publish+copy). Restored after schedule() so next
        # step's budget is unaffected.
        _saved_budget = self._ft_apply_ckpt_aware_budget()

        scheduler_output = self.scheduler.schedule()
        if _ts_active:
            _t1 = _time.perf_counter()

        # Restore original budget (idempotent — no-op if not overridden).
        self._ft_restore_ckpt_aware_budget(_saved_budget)

        # FT restore: schedule() has allocated KV blocks.  Restore
        # checkpoint KV into those blocks and patch scheduler_output
        # to skip the restored prefix (avoids redundant prefill).
        self._process_ft_pending_restores(scheduler_output)
        if _ts_active:
            _t2 = _time.perf_counter()

        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        if _ts_active:
            _t3 = _time.perf_counter()

        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        if _ts_active:
            _t4 = _time.perf_counter()

        # FT_CKPT_GPU_OVERLAP=1: do checkpoint publish work here, while
        # GPU is busy with forward pass. The key insight:
        #
        #   future.result() below waits ~11ms for GPU forward to finish.
        #   During that wait, CPU is IDLE.
        #   Checkpoint publish needs ~2-3ms of CPU work.
        #   By doing publish BEFORE future.result(), we overlap CPU work
        #   with GPU compute → net 0 extra latency, no background thread
        #   needed, 0 GIL contention.
        #
        # This collects the PREVIOUS step's checkpoint RPC result
        # (GPU gather already completed by now) and does the publish.
        # The current step's checkpoint will be fired in _ft_post_process
        # after update_from_output, and collected next step here.
        if os.environ.get("FT_CKPT_GPU_OVERLAP") == "1":
            self._ft_collect_and_publish_checkpoint()

        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
        if _ts_active:
            _t5 = _time.perf_counter()

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        if _ts_active:
            _t6 = _time.perf_counter()

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        if _ts_active:
            _t7 = _time.perf_counter()

        self._ft_post_process(engine_core_outputs)
        if _ts_active:
            _t8 = _time.perf_counter()
            _ts_state["samples"].append((
                _t1 - _t0,  # 1. scheduler.schedule()
                _t2 - _t1,  # 2. _process_ft_pending_restores
                _t3 - _t2,  # 3. execute_model (launch, non-block)
                _t4 - _t3,  # 4. get_grammar_bitmask
                _t5 - _t4,  # 5. future.result() — GPU forward wait
                _t6 - _t5,  # 6. _process_aborts_queue
                _t7 - _t6,  # 7. update_from_output
                _t8 - _t7,  # 8. _ft_post_process
            ))
            _ts_state["calls"] += 1
            if _ts_state["calls"] >= _ts_state["max_calls"] and not _ts_state["dumped"]:
                _ts_state["dumped"] = True
                _ft_dump_step_timing()

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def _capture_ft_checkpoint_plan(
        self,
    ) -> list[tuple[str, list[int]]] | None:
        """Capture per-batch checkpoint work before async scheduling advances.

        In async scheduling, schedule() may run multiple times before the
        output for an earlier batch is processed.  We therefore snapshot the
        requests chosen for checkpointing together with the KV block IDs that
        were valid for that scheduled batch, and defer only the token-count
        lookup until post-step.
        """
        if not hasattr(self.scheduler, "ft_scheduler"):
            return None

        checkpoint_requests = self.scheduler.get_checkpoint_requests()
        if not checkpoint_requests:
            return None

        return [
            (req_id, list(block_ids))
            for req_id, block_ids, _num_tokens in checkpoint_requests
        ]

    def _maybe_ft_checkpoint(
        self,
        checkpoint_plan: list[tuple[str, list[int]]] | None = None,
    ) -> dict[str, int] | None:
        """Trigger GPU→CPU KV checkpoint if using FT scheduler.

        Async pipeline:
        1. Collect results from PREVIOUS step's checkpoint RPC (if any).
        2. Fire a NEW checkpoint RPC for this step (don't wait).

        Returns:
            Dict mapping request_id -> num_checkpointed_tokens for
            requests whose checkpoint completed (from previous step).
        """
        if not hasattr(self.scheduler, "ft_scheduler"):
            return None

        ft = self.scheduler.ft_scheduler
        ckpt_updates: dict[str, int] = {}

        # FT_CKPT_GPU_OVERLAP=1: collection is done in
        # _ft_collect_and_publish_checkpoint (called during GPU wait
        # slot, before future.result()). Skip collection here — only
        # fire new RPC.
        if os.environ.get("FT_CKPT_GPU_OVERLAP") == "1":
            # Still need back-pressure: don't fire if previous not done
            if hasattr(self, "_ft_ckpt_future") and self._ft_ckpt_future is not None:
                if not self._ft_ckpt_future.done():
                    return ckpt_updates if ckpt_updates else None
                # Done but not collected yet → will be collected in GPU wait slot
                # For now just clear it so we can fire a new one
                # (collect already happened or will happen next step)
                self._ft_ckpt_future = None
            # Skip to Step 2 (fire new RPC)
        else:
            pass  # Fall through to original Step 1 below

        # Step 1: Collect results from previous async RPC.
        #
        # FT_CKPT_NONBLOCK env var: when set to "1", use a NON-BLOCKING
        # collection (only process the previous future if it's already
        # done). Otherwise the API server's step T blocks waiting on
        # step T-1's checkpoint RPC, which under W1_Chat/Heavy fault
        # load takes ~70 ms (vs ~30 ms step time), causing a 60 %
        # goodput drop. See overnight_2026-04-09.md follow-up
        # "stream blocking root cause".
        #
        # Trade-off when enabled: if the RPC is slower than step time,
        # we skip firing a new checkpoint RPC for that step (the
        # `_ft_ckpt_future is None` check at submit time below). The
        # checkpoint controller's policy already handles missed cycles
        # gracefully — the next save just covers a larger delta.
        #
        # Default OFF (no behavior change). Validated in phase 7 of
        # the 2026-04-09 overnight investigation.
        ckpt_nonblock = os.environ.get("FT_CKPT_NONBLOCK") == "1"
        if hasattr(self, "_ft_ckpt_future") and self._ft_ckpt_future is not None:
            future_done = (
                self._ft_ckpt_future.done() if ckpt_nonblock else True
            )
            if future_done:
                try:
                    results = self._ft_ckpt_future.result()
                    if results and results[0]:
                        metadata_updates: dict[str, tuple[int, int]] = {}
                        for result in results[0]:
                            if len(result) == 3:
                                req_id, size_bytes, covered_tokens = result
                            else:
                                req_id, size_bytes = result
                                request = ft.request_pool.get_request(req_id)
                                covered_tokens = (
                                    request.num_computed_tokens
                                    if request is not None else 0
                                )
                            request = ft.request_pool.get_request(req_id)
                            if request is not None:
                                # Only update size_bytes from RPC result.
                                # num_checkpointed_tokens was already set
                                # eagerly when the RPC was fired.
                                request.last_checkpoint_size_bytes = size_bytes
                                metadata_updates[req_id] = (
                                    covered_tokens, size_bytes,
                                )
                        if metadata_updates:
                            ft.update_checkpoint_metadata(metadata_updates)
                except Exception as e:
                    logger.warning("FT checkpoint RPC failed: %s", e)
                self._ft_ckpt_future = None

        # Step 2: Fire new checkpoint RPC (async, don't wait).
        #
        # In ckpt_nonblock mode, don't fire a new RPC if the previous
        # one hasn't returned yet — that's the back-pressure that keeps
        # the in-flight queue bounded at depth 1.
        if ckpt_nonblock and getattr(self, "_ft_ckpt_future", None) is not None:
            return ckpt_updates if ckpt_updates else None

        # FT_CKPT_STEP_INTERVAL: fire checkpoint RPC every K steps
        # instead of every step. Reduces GIL acquire/release frequency
        # by K× without changing checkpoint controller logic.
        # Default 1 = every step (original behavior).
        _ckpt_interval = int(os.environ.get("FT_CKPT_STEP_INTERVAL", "1") or "1")
        if _ckpt_interval > 1:
            self._ft_ckpt_step_counter = getattr(
                self, "_ft_ckpt_step_counter", 0
            ) + 1
            if self._ft_ckpt_step_counter % _ckpt_interval != 0:
                return ckpt_updates if ckpt_updates else None

        request_block_map: list[tuple[str, list[int], int]] = []
        if checkpoint_plan is None:
            checkpoint_requests = self.scheduler.get_checkpoint_requests()
            if checkpoint_requests:
                request_block_map = checkpoint_requests
        else:
            for req_id, block_ids in checkpoint_plan:
                request = ft.request_pool.get_request(req_id)
                if request is None or not block_ids:
                    continue
                request_block_map.append((
                    req_id, block_ids, request.num_computed_tokens,
                ))

        if request_block_map:
            # FT_CKPT_INSTRUMENT: lightweight counter for fire count + block
            # count, broken down by prefill/decode stage. Only int adds, no
            # IO. Periodic INFO dump every 500 fires (~30s at our cadence).
            if not hasattr(self, "_ft_ckpt_inst"):
                self._ft_ckpt_inst = {
                    "fires_total": 0, "blocks_total": 0,
                    "fires_prefill": 0, "blocks_prefill": 0,
                    "fires_decode": 0, "blocks_decode": 0,
                    "per_req_fires": {}, "per_req_blocks": {},
                }
            _inst = self._ft_ckpt_inst
            _fire_blocks = 0
            for req_id, block_ids, _num_tokens in request_block_map:
                _fire_blocks += len(block_ids)
                _r = ft.request_pool.get_request(req_id)
                _is_decode = (
                    _r is not None
                    and getattr(_r, "num_output_tokens", 0) > 0
                )
                if _is_decode:
                    _inst["fires_decode"] += 1
                    _inst["blocks_decode"] += len(block_ids)
                else:
                    _inst["fires_prefill"] += 1
                    _inst["blocks_prefill"] += len(block_ids)
                _inst["per_req_fires"][req_id] = (
                    _inst["per_req_fires"].get(req_id, 0) + 1
                )
                _inst["per_req_blocks"][req_id] = (
                    _inst["per_req_blocks"].get(req_id, 0) + len(block_ids)
                )
            _inst["fires_total"] += 1  # one RPC fire per request_block_map
            _inst["blocks_total"] += _fire_blocks
            if _inst["fires_total"] % 500 == 0:
                logger.info(
                    "FT_CKPT_INST fires=%d blocks=%d "
                    "(prefill: %d fires/%d blocks; decode: %d fires/%d blocks)",
                    _inst["fires_total"], _inst["blocks_total"],
                    _inst["fires_prefill"], _inst["blocks_prefill"],
                    _inst["fires_decode"], _inst["blocks_decode"],
                )

            # Eagerly update checkpoint metadata BEFORE firing RPC.
            # num_checkpointed_tokens is known now; only size_bytes
            # needs the RPC result (updated when RPC completes next step).
            for req_id, block_ids, num_tokens in request_block_map:
                request = ft.request_pool.get_request(req_id)
                if request is not None:
                    ft.checkpoint_controller.record_checkpoint(request)
                    # record_checkpoint sets num_checkpointed_tokens to the
                    # correct block-aligned value. Don't override.
                    ckpt_updates[req_id] = request.num_checkpointed_tokens

            # FT_CKPT_NO_GIL=1: run checkpoint RPC synchronously on
            # main thread but use a dedicated CUDA stream for the GPU
            # gather. The key insight: the GIL contention that causes
            # the +8.4ms per-step overhead comes from the background
            # ThreadPoolExecutor thread holding GIL while doing Python
            # work (CUDA launches, publish file IO). By running on the
            # main thread instead, we eliminate all GIL contention —
            # the main thread does checkpoint work at a predictable
            # point (between update_from_output and next schedule()),
            # instead of racing with schedule() unpredictably.
            #
            # Trade-off: checkpoint work blocks the main thread for
            # ~2-3ms (publish IO), but this is LESS than the 8.4ms
            # average GIL delay from the background thread approach.
            #
            # FT_BATCH_GATHER + FT_GATHER_STREAM should also be set
            # to ensure GPU gather is on a non-default stream and
            # doesn't serialize with decode.
            if os.environ.get("FT_CKPT_NO_GIL") == "1":
                # Synchronous on main thread — no background thread
                try:
                    results = self.collective_rpc(
                        "checkpoint_kv_blocks",
                        None,
                        (request_block_map,),
                    )
                    if results and results[0]:
                        for result in results[0]:
                            if len(result) == 3:
                                req_id, size_bytes, covered_tokens = result
                            else:
                                req_id, size_bytes = result
                                covered_tokens = 0
                            request = ft.request_pool.get_request(req_id)
                            if request is not None:
                                request.last_checkpoint_size_bytes = size_bytes
                except Exception as e:
                    logger.warning("FT checkpoint RPC failed: %s", e)
            else:
                # Original: fire in background thread (GIL contention)
                if not hasattr(self, "_ft_ckpt_executor"):
                    from concurrent.futures import ThreadPoolExecutor
                    self._ft_ckpt_executor = ThreadPoolExecutor(max_workers=1)

                self._ft_ckpt_future = self._ft_ckpt_executor.submit(
                    self.collective_rpc,
                    "checkpoint_kv_blocks",
                    None,  # timeout
                    (request_block_map,),
                )

        return ckpt_updates if ckpt_updates else None

    def _get_solver_recovery_targets(self) -> dict[str, int] | None:
        """Extract solver-planned recovery targets from Benders scheduler.

        Returns:
            None if not using Benders scheduler (don't touch buffer).
            {} if Benders but no plans (clear buffer).
            dict if plans exist (replace buffer).
        """
        if not hasattr(self.scheduler, "get_solver_recovery_target"):
            return None  # Not a Benders scheduler.

        plans = getattr(self.scheduler, "_recovery_plans", None)
        if not plans:
            return {}  # Benders but no plans (cleared after fallback).

        replica_id = getattr(self.scheduler, "_replica_id", None)
        if replica_id is None:
            return {}

        # Find best matching scenario containing the local replica.
        omega = frozenset({replica_id})
        plan = plans.get(omega)
        if plan:
            return dict(plan)

        # For multi-failure scenarios, try any scenario containing local.
        for scenario, sp in plans.items():
            if replica_id in scenario and sp:
                return dict(sp)

        return {}

    def _build_active_snapshots(self) -> list[RequestSnapshot] | None:
        """Build RequestSnapshots for the centralized solver.

        Only active when policy is ft_benders_centralized and the
        scheduler has an ft_scheduler with a request pool.

        Solution-? snapshot bypass: when FT_DISABLE_SNAPSHOTS=1 is set,
        skip snapshot construction entirely. Snapshots are only used by
        the centralized Benders solver, which on W1_Chat/Heavy is in
        greedy fallback ~98% of the time (the cost model is for A6000
        dp=1; see e1a_quick_diagnosis.md follow-up #6). When that's
        true, the snapshots are wasted CPU + msgpack + ZMQ traffic.
        Default is OFF (no behavior change). Set the env var to test.
        """
        if not hasattr(self.scheduler, "ft_scheduler"):
            return None

        if os.environ.get("FT_DISABLE_SNAPSHOTS") == "1":
            return None

        ft = self.scheduler.ft_scheduler
        snapshots: list[RequestSnapshot] = []
        for req in ft.request_pool.get_admitted_requests():
            snapshots.append(RequestSnapshot(
                request_id=req.request_id,
                prompt_len=req.prompt_len,
                generation_len=req.generation_len,
                num_computed_tokens=req.num_computed_tokens,
                num_output_tokens=req.num_output_tokens,
                num_checkpointed_tokens=req.num_checkpointed_tokens,
                checkpoint_size_bytes=req.last_checkpoint_size_bytes,
                checkpoint_level=req.checkpoint_level,
                assigned_replica_id=req.assigned_replica_id,
                ttft_slo_ms=req.ttft_slo_ms,
                tpot_slo_ms=req.tpot_slo_ms,
                failure_gap_slo_ms=req.failure_gap_slo_ms,
            ))
        return snapshots

    def _build_replica_snapshot(self) -> ReplicaSnapshot | None:
        """Build ReplicaSnapshot for the centralized solver.

        Same FT_DISABLE_SNAPSHOTS gate as _build_active_snapshots.
        """
        if not hasattr(self.scheduler, "ft_scheduler"):
            return None

        if os.environ.get("FT_DISABLE_SNAPSHOTS") == "1":
            return None

        waiting, running = self.scheduler.get_request_counts()

        # Access KV cache stats. The base scheduler (accessed via _base
        # for FT schedulers) has the kv_cache_manager.
        kv_usage = 0.0
        free_blocks = 0
        base_sched = getattr(self.scheduler, "_base", self.scheduler)
        kv_mgr = getattr(base_sched, "kv_cache_manager", None)
        if kv_mgr is not None:
            kv_usage = kv_mgr.usage
            free_blocks = kv_mgr.block_pool.get_num_free_blocks()

        replica_id = getattr(self, "engine_index",
                             self.vllm_config.parallel_config.data_parallel_rank or 0)
        return ReplicaSnapshot(
            replica_id=replica_id,
            is_healthy=True,
            num_waiting_reqs=waiting,
            num_running_reqs=running,
            gpu_kv_cache_usage=kv_usage,
            available_kv_blocks=free_blocks,
        )

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Solution 4: drain pending rerouted requests into the scheduler
        # if there is capacity. No-op when FT_LAZY_RELOAD is off.
        if _FT_LAZY_RELOAD_ENABLED:
            self._drain_ft_lazy_pending()

        # M3 SLO-aware preemption drain: see step() for rationale.
        self._drain_slo_preempted_restores()

        # Phase 2 C-mode (queue-based): manage overlap reload side queue.
        # See step() for rationale.
        self._process_overlap_reload_queue()

        # Phase 2 SLO retain path: tick down retained requests and
        # resume those whose retain window expired. See step() for
        # rationale.
        self._process_slo_retained_queue()

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        # ── P0-impl-3a: per-section wall timing for step_with_batch_queue ──
        _ts_state = _ft_step_timing_state
        _ts_active = (
            _ts_state["max_calls"] > 0
            and _ts_state["calls"] < _ts_state["max_calls"]
        )
        if _ts_active:
            import time as _time
            _t_call_start = _time.perf_counter()

        model_executed = False
        deferred_scheduler_output = None
        deferred_checkpoint_plan: list[tuple[str, list[int]]] | None = None
        if self.scheduler.has_requests():
            # FT_RECOVERY_PREBUDGET: see step() for rationale. Also
            # applied on the batch-queue path.
            self._ft_prebudget_pending_restores()
            _saved_budget = self._ft_apply_ckpt_aware_budget()

            scheduler_output = self.scheduler.schedule()

            self._ft_restore_ckpt_aware_budget(_saved_budget)

            # FT restore: schedule() has allocated KV blocks. Restore
            # checkpoint KV into those blocks before execute_model().
            self._process_ft_pending_restores(scheduler_output)
            checkpoint_plan = self._capture_ft_checkpoint_plan()

            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output
                    deferred_checkpoint_plan = checkpoint_plan

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((
                    future,
                    scheduler_output,
                    exec_future,
                    checkpoint_plan,
                ))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    if _ts_active:
                        _t_call_end = _time.perf_counter()
                        _ts_state["samples"].append((
                            _t_call_end - _t_call_start,  # 1. total call
                            0.0,  # 2. future.result wait (not in this path)
                            0.0,  # 3. update_from_output
                            0.0,  # 4. _ft_post_process
                            -1.0,  # 5. flag: enqueue path (no wait)
                            0.0, 0.0, 0.0,
                        ))
                        _ts_state["calls"] += 1
                        if (_ts_state["calls"] >= _ts_state["max_calls"]
                                and not _ts_state["dumped"]):
                            _ts_state["dumped"] = True
                            _ft_dump_step_timing()
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        if _ts_active:
            _t_pre_pop = _time.perf_counter()
        future, scheduler_output, exec_model_fut, checkpoint_plan = batch_queue.pop()
        if _ts_active:
            _t_pre_wait = _time.perf_counter()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
        if _ts_active:
            _t_post_wait = _time.perf_counter()
        # Bug fix 2026-04-13: this None check was previously nested inside
        # `if _ts_active:` so it never ran in production (_ts_active is
        # False by default). A None model_output then propagated into
        # scheduler.update_from_output() → AttributeError: 'NoneType' has
        # no attribute 'sampled_token_ids' → engine crash. Must run
        # unconditionally so we re-surface the original exec_model failure.
        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((
                future,
                deferred_scheduler_output,
                exec_future,
                deferred_checkpoint_plan,
            ))

        self._ft_post_process(
            engine_core_outputs, checkpoint_plan=checkpoint_plan
        )

        if _ts_active:
            _t_call_end = _time.perf_counter()
            # 5 segments captured for the dequeue path:
            # 1. total call time
            # 2. enqueue-side prep (schedule + execute_model launch + sample) before pop
            # 3. pop + future.result() wait — GPU forward wait time
            # 4. update_from_output + deferred handling
            # 5. _ft_post_process
            _ts_state["samples"].append((
                _t_call_end - _t_call_start,                  # 1. total
                _t_pre_pop - _t_call_start,                   # 2. enqueue-side prep (this call)
                _t_post_wait - _t_pre_pop,                    # 3. pop + future.result wait
                _t_call_end - _t_post_wait,                   # 4. post-wait + ft_post_process
                1.0,                                           # 5. flag: dequeue path
                0.0, 0.0, 0.0,
            ))
            _ts_state["calls"] += 1
            if (_ts_state["calls"] >= _ts_state["max_calls"]
                    and not _ts_state["dumped"]):
                _ts_state["dumped"] = True
                _ft_dump_step_timing()

        return engine_core_outputs, model_executed

    def _ft_collect_and_publish_checkpoint(self) -> None:
        """FT_CKPT_GPU_OVERLAP: collect previous step's checkpoint result
        and do publish, overlapping with GPU forward compute.

        Called BEFORE future.result() so CPU work overlaps with GPU.
        No background thread needed → 0 GIL contention.

        This replaces the "Step 1: collect" part of _ft_maybe_checkpoint.
        When FT_CKPT_GPU_OVERLAP=1, _ft_maybe_checkpoint only fires
        the RPC (no collect), and this method does the collect+publish.
        """
        if not hasattr(self, "_ft_ckpt_future") or self._ft_ckpt_future is None:
            return

        # Non-blocking check: is the RPC done?
        if not self._ft_ckpt_future.done():
            return  # GPU gather still running, skip — will retry next step

        # Collect result (RPC done → no blocking)
        ft = getattr(self.scheduler, "ft_scheduler", None)
        if ft is None:
            self._ft_ckpt_future = None
            return

        try:
            results = self._ft_ckpt_future.result()
            if results and results[0]:
                for result in results[0]:
                    if len(result) == 3:
                        req_id, size_bytes, covered_tokens = result
                    else:
                        req_id, size_bytes = result
                        covered_tokens = 0
                    request = ft.request_pool.get_request(req_id)
                    if request is not None:
                        request.last_checkpoint_size_bytes = size_bytes
                        metadata = {req_id: (covered_tokens, size_bytes)}
                        ft.update_checkpoint_metadata(metadata)
        except Exception as e:
            logger.warning("FT checkpoint collect failed: %s", e)

        self._ft_ckpt_future = None

    def _ft_post_process(
        self,
        engine_core_outputs: dict | None,
        checkpoint_plan: list[tuple[str, list[int]]] | None = None,
    ) -> None:
        """FT post-processing shared by step() and step_with_batch_queue().

        Handles checkpoint updates, solver recovery targets/ckpt classes,
        and centralized solver snapshots.
        """
        # FT checkpoint hook: trigger GPU→CPU KV checkpoint.
        ckpt_updates = self._maybe_ft_checkpoint(checkpoint_plan)

        # Merge any buffered checkpoint updates from previous steps.
        if not hasattr(self, "_ft_buffered_ckpt_updates"):
            self._ft_buffered_ckpt_updates: dict[str, int] = {}
        if ckpt_updates:
            self._ft_buffered_ckpt_updates.update(ckpt_updates)

        # Inject checkpoint updates into ALL output entries so every
        # client receives the update (DP may have multiple clients).
        if self._ft_buffered_ckpt_updates and engine_core_outputs:
            for outputs in engine_core_outputs.values():
                outputs.checkpoint_updates = dict(
                    self._ft_buffered_ckpt_updates
                )
            self._ft_buffered_ckpt_updates.clear()

        # FT Benders: buffer solver recovery targets.
        if not hasattr(self, "_ft_buffered_recovery_targets"):
            self._ft_buffered_recovery_targets: dict[str, int] | None = None

        recovery_targets = self._get_solver_recovery_targets()
        if recovery_targets is not None:
            self._ft_buffered_recovery_targets = dict(recovery_targets)

        # Flush buffers to outputs.
        if self._ft_buffered_recovery_targets is not None and engine_core_outputs:
            for outputs in engine_core_outputs.values():
                outputs.recovery_targets = dict(
                    self._ft_buffered_recovery_targets
                )
            self._ft_buffered_recovery_targets = None

        # FT Benders centralized: emit request + replica snapshots for
        # the client-side global solver. Throttled to once per 100ms.
        if (self.vllm_config.scheduler_config.policy
                == "ft_benders_centralized" and engine_core_outputs):
            if not hasattr(self, "_ft_last_snapshot_time"):
                self._ft_last_snapshot_time: float = 0.0
            now = time.monotonic()
            if now - self._ft_last_snapshot_time >= 0.1:
                req_snaps = self._build_active_snapshots()
                rep_snap = self._build_replica_snapshot()
                if req_snaps is not None:
                    for outputs in engine_core_outputs.values():
                        outputs.active_request_snapshots = req_snaps
                        outputs.replica_snapshot = rep_snap
                self._ft_last_snapshot_time = now

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        # Dump final FT_CKPT_INST stats if any.
        _inst = getattr(self, "_ft_ckpt_inst", None)
        if _inst:
            logger.info(
                "FT_CKPT_INST FINAL fires=%d blocks=%d "
                "(prefill: %d fires/%d blocks; decode: %d fires/%d blocks) "
                "per_req_count=%d",
                _inst["fires_total"], _inst["blocks_total"],
                _inst["fires_prefill"], _inst["blocks_prefill"],
                _inst["fires_decode"], _inst["blocks_decode"],
                len(_inst["per_req_fires"]),
            )

        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[
            tuple[int, EngineCoreOutputs] | tuple[bytes, int]
        ]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing

            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # FT_GIL_INTERVAL: tune Python GIL switch interval for this
        # EngineCore process. Default 5ms → shorter values reduce the
        # compound GIL delay from checkpoint background thread.
        # Must be set in the EngineCore process (not parent) because
        # setswitchinterval is per-process.
        import sys as _sys
        _gil_interval = os.environ.get("FT_GIL_INTERVAL")
        if _gil_interval:
            try:
                _sys.setswitchinterval(float(_gil_interval))
                logger.info(
                    "FT_GIL_INTERVAL: set GIL switch interval to %.4fs",
                    float(_gil_interval),
                )
            except (ValueError, TypeError):
                pass

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                set_process_title("EngineCore", f"DP{dp_rank}")
            else:
                set_process_title("EngineCore")
            decorate_logs()

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    @property
    def _ft_idle_heartbeat_sec(self) -> float:
        """Heartbeat interval for idle engines.

        Derived from the coordinator's failure_timeout_sec to avoid the
        configuration trap where heartbeat > timeout causes false deaths.
        Uses timeout / 3 so that at least 2 heartbeats arrive within each
        timeout window.
        """
        timeout = getattr(
            self.vllm_config.scheduler_config,
            "failure_timeout_sec",
            5.0,
        )
        return max(0.5, timeout / 3.0)

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True

            # Use a timeout so that idle engines periodically send a
            # heartbeat to the coordinator, preventing false failure
            # detection.  Without this, an engine with no requests
            # would block forever and be declared dead by the
            # coordinator's output-based liveness check.
            try:
                req = self.input_queue.get(
                    timeout=self._ft_idle_heartbeat_sec
                )
            except queue.Empty:
                # No input received — send a heartbeat (empty stats)
                # so the coordinator knows we're alive.
                self._send_idle_heartbeat()
                continue
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _send_idle_heartbeat(self) -> None:
        """Send a heartbeat to the coordinator while idle.

        Publishes a minimal EngineCoreOutputs with scheduler stats so
        that the coordinator's output-based liveness check doesn't
        falsely declare this engine as dead.
        """
        if not hasattr(self, "output_queue"):
            return
        try:
            from vllm.v1.core.sched.output import SchedulerStats
            counts = self.scheduler.get_request_counts()
            stats = SchedulerStats(
                *counts,
                step_counter=getattr(self, "step_counter", 0),
                current_wave=getattr(self, "current_wave", 0),
            )
            outputs = EngineCoreOutputs(scheduler_stats=stats)
            self._attach_centralized_snapshot_metadata(outputs)
            self.output_queue.put_nowait((-1, outputs))
        except Exception:
            pass  # Best-effort heartbeat.

    def _attach_centralized_snapshot_metadata(
        self,
        outputs: EngineCoreOutputs,
    ) -> None:
        """Attach centralized-solver snapshot metadata to a single output.

        Stats-only heartbeats are often the only messages emitted by an idle
        replica. Without snapshots on those heartbeats, the centralized client
        can fail to observe healthy but idle engines and ends up solving over
        an incomplete replica set.
        """
        if self.vllm_config.scheduler_config.policy != "ft_benders_centralized":
            return

        outputs.active_request_snapshots = self._build_active_snapshots()
        outputs.replica_snapshot = self._build_replica_snapshot()

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there are waiting requests
        # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow
        # background threads (like NIXL handshake) to make progress.
        # Without this, the tight polling loop can starve background threads.
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.REPLICA_FAILED:
            self._handle_replica_failed(request)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    def _handle_replica_failed(self, failed_replica_id: int) -> None:
        """Handle REPLICA_FAILED from coordinator for FT schedulers."""
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "ft_scheduler"):
            return

        ft = scheduler.ft_scheduler
        fd = ft.failure_detector

        from vllm.v1.core.failure_detector import ReplicaStatus

        status = fd.get_status(failed_replica_id)
        if status is None:
            fd.register_replica(failed_replica_id)
        if status != ReplicaStatus.FAILED:
            fd.mark_remote_failed(failed_replica_id)
            ft.replica_manager.notify_remote_replica_failed()
            logger.info(
                "Engine %d: marked remote replica %d as FAILED "
                "(notified by coordinator)",
                self.engine_index,
                failed_replica_id,
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put (ENGINE_CORE_DEAD, engine_index) in the queue so the output
        # thread can send a 2-frame multipart that identifies *which*
        # engine died.  This lets FT clients trigger immediate failover
        # without guessing.
        self.output_queue.put_nowait(
            (EngineCoreProc.ENGINE_CORE_DEAD, self.engine_index)
        )

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if (
                    isinstance(output, tuple)
                    and len(output) == 2
                    and output[0] == EngineCoreProc.ENGINE_CORE_DEAD
                ):
                    # Send 2-frame multipart: [ENGINE_CORE_DEAD, engine_index].
                    # Frame 0 is the sentinel for backward compat with base
                    # client; frame 1 carries the engine identity so FT
                    # clients can trigger immediate, targeted failover.
                    _, dead_engine_index = output
                    idx_bytes = msgspec.msgpack.encode(dead_engine_index)
                    for socket in sockets:
                        socket.send_multipart(
                            [EngineCoreProc.ENGINE_CORE_DEAD, idx_bytes]
                        )
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                        )
                    ],
                ),
            )
        )


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        elif request_type == EngineCoreRequestType.REPLICA_FAILED:
            self._handle_replica_failed(request)
        else:
            super()._handle_client_request(request_type, request)

    def _handle_replica_failed(self, failed_replica_id: int) -> None:
        """Handle REPLICA_FAILED from coordinator."""
        super()._handle_replica_failed(failed_replica_id)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            outputs = EngineCoreOutputs(scheduler_stats=stats)
            self._attach_centralized_snapshot_metadata(outputs)
            self.output_queue.put_nowait((-1, outputs))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
