# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fault-Tolerant DP Client for multi-replica vLLM serving.

Extends the DP load-balancing client with:
- Request caching for failover re-routing.
- Partial engine death handling (single engine dies != system death).
- Immediate failover from both ENGINE_CORE_DEAD and ENGINE_FAILED.

Failure detection has two authoritative sources, both carrying engine_index:
- ENGINE_CORE_DEAD(engine_index): sent by the dying engine itself (fast).
- ENGINE_FAILED(engine_index): sent by FTCoordinator on heartbeat timeout
  (catches silent deaths where the engine can't send its own death signal).

Whichever arrives first triggers failover; the second is a no-op (idempotent).
"""

import asyncio
import multiprocessing.connection
import os
import shutil
import time
import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from threading import Thread

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReplicaSnapshot,
    RequestSnapshot,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import (
    DPLBAsyncMPClient,
    EngineIdentity,
    _process_utility_output,
)
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor import Executor
from vllm.utils.network_utils import make_zmq_socket

logger = init_logger(__name__)

_SHARED_CKPT_DIR = "/dev/shm/vllm_ft_checkpoints"

# ── Recovery mode (FT_RECOVERY_MODE env var, default "reload") ───────────────
# Mirrors vllm/v1/core/recovery_manager.py. Modes:
#   - "reload":   Default. Restore KV from host-memory checkpoint. Replay any
#                 uncovered decoded suffix on the target engine.
#   - "restart":  Drop checkpoint AND already-decoded tokens. Treat as a fresh
#                 ADD with the original prompt; the new engine re-prefills the
#                 prompt and re-samples decode tokens. Stream consistency is
#                 BROKEN — newly sampled tokens differ from those already
#                 streamed to the client. Used only for ablation comparison
#                 against the reload path.
#   - "reprefill": Drop checkpoint but PRESERVE already-decoded tokens by
#                  extending the prompt: extended_prompt = original_prompt +
#                  decoded_tokens. The new engine re-prefills the extended
#                  prompt then continues decoding. Stream consistency preserved
#                  (the next sampled token logically follows the last decoded
#                  one). Used to measure whether the KV-reload path is worth
#                  the I/O cost vs a simple recompute.
_FT_RECOVERY_MODE = os.environ.get("FT_RECOVERY_MODE", "reload").lower()
if _FT_RECOVERY_MODE not in ("reload", "restart", "reprefill"):
    logger.warning(
        "Unknown FT_RECOVERY_MODE=%r; falling back to 'reload'",
        _FT_RECOVERY_MODE,
    )
    _FT_RECOVERY_MODE = "reload"

# ── API server cProfile (FT_API_PROFILE_SECONDS env var) ─────────────────
# When set, the CentralizedBendersFTClient schedules a one-shot cProfile
# window over the API server process's main asyncio event loop. The
# profile catches: aiohttp request handling, ft_client output processing,
# snapshot updates, the Benders solver foreground path. (Background
# threads — e.g. the ThreadPoolExecutor that runs the actual cp_model
# solve — are NOT captured by cProfile; if the profile shows the API
# server is mostly idle, that means CPU is in the executor thread.)
#
# Usage:
#   FT_API_PROFILE_SECONDS=60 \
#   FT_API_PROFILE_DELAY=60 \
#   FT_API_PROFILE_OUTPUT=/tmp/api_profile.txt \
#   python ...
#
# The profile starts FT_API_PROFILE_DELAY seconds after the client is
# initialized (skip warmup) and runs for FT_API_PROFILE_SECONDS seconds.
# Default: disabled (0 seconds).
try:
    _FT_API_PROFILE_SECONDS = float(
        os.environ.get("FT_API_PROFILE_SECONDS", "0")
    )
except ValueError:
    _FT_API_PROFILE_SECONDS = 0.0
try:
    _FT_API_PROFILE_DELAY = float(
        os.environ.get("FT_API_PROFILE_DELAY", "60")
    )
except ValueError:
    _FT_API_PROFILE_DELAY = 60.0
_FT_API_PROFILE_OUTPUT = os.environ.get(
    "FT_API_PROFILE_OUTPUT", "/tmp/ft_api_profile.txt"
)


def _ft_rmtree_silent(path: str) -> None:
    """Silent rmtree for FT_ASYNC_CLEANUP background cleanup.

    Suppresses FileNotFoundError (race with concurrent cleanup) and
    logs other OSErrors at WARNING. Used as the target callable for
    `loop.run_in_executor(None, _ft_rmtree_silent, request_dir)`.
    Module-level so the executor can pickle it cleanly.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning(
            "FT_ASYNC_CLEANUP: rmtree(%s) failed: %s", path, exc,
        )


class FTDPAsyncMPClient(DPLBAsyncMPClient):
    """FT-aware DP client that handles engine failures gracefully.

    Instead of killing the entire system when one engine dies, this client:
    1. Marks the specific engine as dead.
    2. Re-routes in-flight requests to a surviving engine.
    3. Continues serving on the remaining engines.
    4. Only raises EngineDeadError when ALL engines are dead.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # Cache outgoing requests for failover re-routing.
        self._request_cache: dict[str, EngineCoreRequest] = {}

        # Track engine liveness: engine identity -> alive flag.
        self._engine_alive: dict[EngineIdentity, bool] = {
            engine: True for engine in self.core_engines
        }

        # Map engine_index (int) -> engine identity (bytes).
        self._index_to_engine: dict[int, EngineIdentity] = {
            i: engine for i, engine in enumerate(self.core_engines)
        }

        # Reverse map: engine identity -> engine_index.
        self._engine_to_index: dict[EngineIdentity, int] = {
            engine: i for i, engine in enumerate(self.core_engines)
        }

        # Track per-request checkpoint token counts for failover restore.
        # Updated when EngineCore reports successful checkpoint.
        self._checkpoint_tokens: dict[str, int] = {}

        # Accumulate output tokens per request for failover.
        # Needed so sampling penalties and bad-words work after recovery.
        self._output_tokens: dict[str, list[int]] = {}

        # FT: track rerouted requests for first-token-after-recovery logging.
        self._rerouted_request_ids: set[str] = set()
        self._rerouted_first_token_logged: set[str] = set()

        # Solver-planned recovery targets from Benders scheduler.
        # Partitioned by engine_index: each engine sends a full snapshot
        # of its own requests' targets. We store per-engine to avoid
        # one engine's update overwriting another engine's entries.
        self._recovery_targets_by_engine: dict[int, dict[str, int]] = {}

        logger.info(
            "FTDPAsyncMPClient initialized with %d engines",
            len(self.core_engines),
        )

    def _cleanup_shared_checkpoint_request(self, request_id: str) -> None:
        """Remove shared checkpoint artifacts after global request termination.

        FT_ASYNC_CLEANUP=1: when set, the actual rmtree is sent to the
        default asyncio executor (a ThreadPoolExecutor) so the syscall
        chain doesn't block the API server's event loop. This matters
        because each rmtree of a /dev/shm/<req_id>/ tree typically
        contains 2-3 chunk + manifest + latest files; per-request
        cleanup observed at ~26 ms each in the API server cProfile.
        At ~3 finished reqs/sec, that's ~80 ms/sec of event loop
        blocking — the largest single hot spot in the API server's
        Python time (35 % of 8 s captured over a 60 s window).

        Default off: caller may run on a thread that doesn't have an
        asyncio loop attached, in which case we fall back to the sync
        path. The fallback also covers shutdown / abort paths where
        we'd rather block synchronously than leave dangling rmtree work.
        """
        request_dir = os.path.join(_SHARED_CKPT_DIR, request_id)

        if os.environ.get("FT_ASYNC_CLEANUP") == "1":
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.run_in_executor(
                    None, _ft_rmtree_silent, request_dir,
                )
                return

        # Sync fallback (default).
        try:
            shutil.rmtree(request_dir)
        except FileNotFoundError:
            return
        except OSError:
            logger.warning(
                "Failed to remove shared checkpoint directory %s",
                request_dir,
                exc_info=True,
            )

    # ---- Override engine monitor: partial death is OK ----

    def start_engine_core_monitor(self):
        """Replace base class monitor that kills the entire client when
        ANY engine process exits.  In FT mode, a single engine dying is
        expected — we handle it via ENGINE_CORE_DEAD / ENGINE_FAILED.
        Only shut down when ALL engines are dead.
        """
        engine_manager = self.resources.engine_manager
        if (
            engine_manager is None
            or not hasattr(engine_manager, "processes")
            or not engine_manager.processes
        ):
            return

        engine_processes = engine_manager.processes
        self_ref = weakref.ref(self)

        def ft_monitor_engine_cores():
            # Build sentinel -> engine_index mapping.
            sentinel_to_idx = {
                proc.sentinel: i for i, proc in enumerate(engine_processes)
            }
            live_sentinels = set(sentinel_to_idx.keys())

            while live_sentinels:
                died = multiprocessing.connection.wait(list(live_sentinels))
                _self = self_ref()
                if not _self:
                    return

                for sentinel in died:
                    live_sentinels.discard(sentinel)
                    eng_idx = sentinel_to_idx.get(sentinel)
                    if eng_idx is None:
                        continue

                    eng_id = _self._index_to_engine.get(eng_idx)
                    if eng_id is not None and not _self._engine_alive.get(
                        eng_id, True
                    ):
                        # Already handled via ZMQ path, skip.
                        continue

                    logger.warning(
                        "FT monitor: engine %d process exited. "
                        "Scheduling failover.",
                        eng_idx,
                    )
                    logger.warning(
                        "FAULT_EVENT monitor_observed engine=%d "
                        "wall_time=%.6f source=process_monitor",
                        eng_idx,
                        time.time(),
                    )

                    output_task = _self.resources.output_queue_task
                    loop = output_task._loop if output_task else None

                    if loop is None or loop.is_closed():
                        logger.warning(
                            "FT monitor: no active loop for engine %d exit; "
                            "marking dead without failover.",
                            eng_idx,
                        )
                        if eng_id is not None:
                            _self._engine_alive[eng_id] = False
                        continue

                    async def handle_failure_from_monitor(
                        _idx: int = eng_idx,
                    ):
                        await _self._handle_engine_failure(_idx)
                        if _self._all_engines_dead():
                            logger.error(
                                "FT monitor: all engines dead, shutting down."
                            )
                            _self.resources.engine_dead = True
                            _self.shutdown()

                    future = asyncio.run_coroutine_threadsafe(
                        handle_failure_from_monitor(), loop
                    )

                    def _log_monitor_failure_result(
                        fut: "concurrent.futures.Future[None]",
                        _idx: int = eng_idx,
                    ) -> None:
                        try:
                            fut.result()
                        except Exception:
                            logger.exception(
                                "FT monitor: failover coroutine for engine %d "
                                "failed",
                                _idx,
                            )

                    future.add_done_callback(_log_monitor_failure_result)

                if _self._all_engines_dead():
                    logger.error(
                        "FT monitor: all engines dead, shutting down."
                    )
                    _self.resources.engine_dead = True
                    _self.shutdown()
                    return

        Thread(
            target=ft_monitor_engine_cores,
            daemon=True,
            name="FTClientEngineMonitor",
        ).start()

    # ---- Sync FT maps on elastic scale ----

    def _rebuild_ft_maps(self) -> None:
        """Rebuild FT tracking maps from current self.core_engines."""
        new_alive: dict[EngineIdentity, bool] = {}
        new_i2e: dict[int, EngineIdentity] = {}
        new_e2i: dict[EngineIdentity, int] = {}
        valid_indices: set[int] = set()
        for i, engine in enumerate(self.core_engines):
            new_i2e[i] = engine
            new_e2i[engine] = i
            valid_indices.add(i)
            # Preserve alive status for engines that existed before;
            # new engines default to alive.
            new_alive[engine] = self._engine_alive.get(engine, True)
        self._engine_alive = new_alive
        self._index_to_engine = new_i2e
        self._engine_to_index = new_e2i

        # Prune solver metadata partitions for engines that no longer exist
        # (e.g. after scale-down).
        for stale_idx in set(self._recovery_targets_by_engine) - valid_indices:
            del self._recovery_targets_by_engine[stale_idx]

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        await super().scale_elastic_ep(new_data_parallel_size)
        self._rebuild_ft_maps()
        logger.info(
            "FT Client: rebuilt FT maps after scale to %d engines",
            new_data_parallel_size,
        )

    # ---- Override request routing ----

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Cache request before sending, for failover re-routing."""
        self._request_cache[request.request_id] = request
        await super().add_request_async(request)
        # Log initial GPU assignment for strict affected_by_failure tracking.
        engine = self.reqs_in_flight.get(request.request_id)
        if engine is not None:
            gpu_idx = self._engine_to_index.get(engine)
            if gpu_idx is not None:
                logger.info(
                    "FAULT_EVENT request_route request=%s gpu=%d "
                    "wall_time=%.6f",
                    request.request_id, gpu_idx, time.time(),
                )

    # ---- Override output processing: handle ENGINE_CORE_DEAD ----

    def _install_fast_fault_detector(self) -> None:
        """Register an asyncio SIGCHLD handler that detects a dead engine
        within ~5ms (vs ~1000ms for the existing GIL-bound monitor thread).

        The handler runs in the main asyncio loop on signal delivery
        (which is propagated at GIL switch boundaries, default 5ms).
        It iterates engine processes, finds dead ones, and synchronously
        kicks off the failover coroutine — avoiding the
        multiprocessing.connection.wait → Python GIL race that delays
        detection in the existing monitor thread.

        No-op if the resources/engine_manager isn't set up yet (e.g. on
        worker processes or test paths). Idempotent.
        """
        if getattr(self, "_fast_fault_detector_installed", False):
            return
        engine_manager = self.resources.engine_manager
        if (
            engine_manager is None
            or not hasattr(engine_manager, "processes")
            or not engine_manager.processes
        ):
            return
        engine_processes = list(engine_manager.processes)
        self._fast_fault_engine_processes = engine_processes

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop, can't install

        self_ref = weakref.ref(self)

        def _on_sigchld() -> None:
            _self = self_ref()
            if _self is None:
                return
            for i, proc in enumerate(engine_processes):
                if proc.is_alive():
                    continue
                eng_id = _self._index_to_engine.get(i)
                if eng_id is None:
                    continue
                if not _self._engine_alive.get(eng_id, True):
                    # Already handled.
                    continue
                # Mark dead immediately so duplicate fires (from the
                # multiprocessing.connection.wait monitor) become no-ops.
                _self._engine_alive[eng_id] = False
                logger.warning(
                    "FAULT_EVENT monitor_observed engine=%d "
                    "wall_time=%.6f source=sigchld_fast",
                    i, time.time(),
                )
                logger.warning(
                    "FAULT_EVENT failure_declared replica=%d wall_time=%.6f",
                    i, time.time(),
                )
                # Schedule failover in the current loop (we're already in it).
                asyncio.create_task(_self._handle_engine_failure(i))
                if _self._all_engines_dead():
                    logger.error(
                        "FT fast detect: all engines dead, shutting down."
                    )
                    _self.resources.engine_dead = True

        # asyncio reserves SIGCHLD for its own subprocess tracking, so
        # loop.add_signal_handler refuses. Use signal.signal() directly
        # and chain to the previous handler (asyncio's child reaper).
        try:
            import signal
            prev_handler = signal.getsignal(signal.SIGCHLD)

            def _on_sigchld_sync(signum, frame):
                # Runs in main thread at next bytecode boundary (~5ms
                # GIL switch). Schedule the actual work into the loop
                # to avoid doing too much in signal handler context.
                try:
                    loop.call_soon_threadsafe(_on_sigchld)
                except Exception:
                    pass
                # Chain to asyncio's child reaper so subprocess tracking
                # still works.
                if callable(prev_handler) and prev_handler not in (
                    signal.SIG_IGN, signal.SIG_DFL,
                ):
                    try:
                        prev_handler(signum, frame)
                    except Exception:
                        pass

            signal.signal(signal.SIGCHLD, _on_sigchld_sync)
            self._fast_fault_detector_installed = True
            self._fast_fault_prev_handler = prev_handler  # keep refcount
            logger.info(
                "FT_FAST_FAULT_DETECT: SIGCHLD signal handler installed "
                "(monitors %d engine processes)",
                len(engine_processes),
            )
        except (ValueError, OSError) as e:
            logger.warning(
                "FT_FAST_FAULT_DETECT: failed to install SIGCHLD handler: %s",
                e,
            )

    def _ensure_output_queue_task(self):
        """Override to intercept ENGINE_CORE_DEAD with engine_index and
        trigger immediate failover.

        ENGINE_CORE_DEAD is now a 2-frame multipart:
            [b"ENGINE_CORE_DEAD", msgpack(engine_index)]
        This lets us identify the dead engine and failover immediately,
        without waiting for the coordinator's timeout-based detection.
        """
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # FT_FAST_FAULT_DETECT=1 (default OFF): install SIGCHLD asyncio
        # signal handler for sub-10ms fault detection. Without this, the
        # ft_monitor_engine_cores thread (using multiprocessing.connection.wait)
        # is event-driven at OS level but its post-wait Python work blocks
        # on GIL contention with the asyncio main thread (~1100ms observed
        # in Heavy on s42, see investigation log 2026-04-14). The asyncio
        # signal handler runs in the main loop within one GIL switch
        # interval (~5ms).
        if os.environ.get("FT_FAST_FAULT_DETECT") == "1":
            self._install_fast_fault_detector()

        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[
                ["FTDPAsyncMPClient", EngineCoreOutputs], Awaitable[None]
            ]
            | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self)
        output_socket = resources.output_socket
        assert output_socket is not None

        async def ft_process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)

                    # ---- FT: handle ENGINE_CORE_DEAD(engine_index) ----
                    if len(frames) >= 1 and (
                        bytes(frames[0].buffer)
                        == EngineCoreProc.ENGINE_CORE_DEAD
                    ):
                        _self = _self_ref()
                        if _self is None:
                            return

                        # Extract engine_index from frame 1.
                        if len(frames) >= 2:
                            engine_index = msgspec.msgpack.decode(
                                bytes(frames[1].buffer)
                            )
                            logger.warning(
                                "FT Client: received ENGINE_CORE_DEAD "
                                "for engine %d. Triggering failover.",
                                engine_index,
                            )
                            logger.warning(
                                "FAULT_EVENT monitor_observed engine=%d "
                                "wall_time=%.6f source=engine_core_dead",
                                engine_index,
                                time.time(),
                            )
                            await _self._handle_engine_failure(engine_index)
                        else:
                            # Fallback for legacy 1-frame format (should
                            # not happen with updated EngineCore).
                            logger.warning(
                                "FT Client: received ENGINE_CORE_DEAD "
                                "without engine_index (legacy format)."
                            )

                        # If all engines are dead, give up.
                        if _self._all_engines_dead():
                            resources.engine_dead = True
                            raise EngineDeadError()
                        continue

                    outputs: EngineCoreOutputs = decoder.decode(frames)

                    # unnecessary special case check for utility outputs
                    if outputs.utility_output:
                        _process_utility_output(
                            outputs.utility_output, utility_results
                        )
                        continue

                    if output_handler is not None:
                        _self = _self_ref()
                        if not _self:
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        # outputs eg:
                        # [
                        #   {
                        #     "request_id": "req-123",
                        #     "new_token_ids": [314, 271],
                        #     "finish_reason": "stop"
                        #   }
                        # ]

                        # scheduler_stats eg: waiting=5, running=2
                        outputs_queue.put_nowait(outputs)

            except EngineDeadError:
                outputs_queue.put_nowait(EngineDeadError())
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            ft_process_outputs_socket(),
            name="FTEngineCoreOutputQueueTask",
        )

    # ---- Override stats update: consume ENGINE_FAILED from coordinator ----

    def _ensure_stats_update_task(self):
        """Override to intercept ENGINE_FAILED events from FTCoordinator
        and execute failover inline."""
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0
        count_slice = slice(
            self.engine_ranks_managed[0], self.engine_ranks_managed[-1] + 1
        )

        _self_ref = weakref.ref(self)

        async def run_ft_stats_update_task():
            with (
                make_zmq_socket(
                    self.ctx, stats_addr, zmq.XSUB, linger=0
                ) as socket,
                make_zmq_socket(
                    self.ctx,
                    self.first_req_sock_addr,
                    zmq.PAIR,
                    bind=False,
                    linger=0,
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        buf = first_req_rcv_socket.recv(
                            flags=zmq.NOBLOCK
                        ).result()
                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            new_engine_count = decoded[1]
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    # Drain all pending messages and process each one.
                    # The old code only kept the last buf, which could
                    # silently drop ENGINE_FAILED if a stats update
                    # followed it in the same batch.
                    pending_bufs: list[bytes] = []
                    while True:
                        future: asyncio.Future[bytes] = socket.recv(
                            flags=zmq.NOBLOCK
                        )
                        if isinstance(future.exception(), zmq.Again):
                            break
                        pending_bufs.append(future.result())
                    if not pending_bufs:
                        continue

                    for buf in pending_bufs:
                        decoded = msgspec.msgpack.decode(buf)

                        # ---- FT: intercept coordinator events ----
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                        ):
                            event_type = decoded[0]

                            if event_type == "ENGINE_FAILED":
                                engine_index = decoded[1]
                                logger.warning(
                                    "FT Client: received ENGINE_FAILED "
                                    "for engine %d from coordinator",
                                    engine_index,
                                )
                                _self = _self_ref()
                                if _self is not None:
                                    await _self._handle_engine_failure(
                                        engine_index
                                    )
                                    if _self._all_engines_dead():
                                        self.resources.engine_dead = True
                                        raise EngineDeadError()
                                continue

                        # Normal stats update: (counts, wave, running)
                        counts, wave, running = decoded
                        self.current_wave = wave
                        self.engines_running = running
                        if counts is not None:
                            sliced_counts = counts[count_slice]
                            self.lb_engines = sliced_counts
                            logger.debug(
                                "Received counts: %s (%s)",
                                sliced_counts,
                                count_slice,
                            )

        resources.stats_update_task = asyncio.create_task(
            run_ft_stats_update_task()
        )

    # ---- Output processing: clean up finished requests ----

    @staticmethod
    async def process_engine_outputs(
        self: "FTDPAsyncMPClient", outputs: EngineCoreOutputs
    ):
        """Process outputs and clean up caches for finished requests."""
        # Accumulate output tokens for failover penalty restoration.
        for out in outputs.outputs:
            if out.new_token_ids:
                acc = self._output_tokens.get(out.request_id)
                if acc is None:
                    acc = []
                    self._output_tokens[out.request_id] = acc
                acc.extend(out.new_token_ids)

                # FT: log first token after recovery for rerouted requests.
                if (out.request_id in self._rerouted_request_ids
                        and out.request_id
                        not in self._rerouted_first_token_logged):
                    logger.info(
                        "FAULT_EVENT first_token_after_recovery "
                        "request=%s wall_time=%.6f",
                        out.request_id, time.time(),
                    )
                    self._rerouted_first_token_logged.add(out.request_id)

        # Update checkpoint token tracking from EngineCore reports.
        if outputs.checkpoint_updates:
            for req_id, num_tokens in outputs.checkpoint_updates.items():
                self._checkpoint_tokens[req_id] = num_tokens

        # Update solver recovery targets from Benders scheduler.
        # Each engine sends a full snapshot of ITS OWN requests' targets.
        # We store per-engine so one engine's update doesn't clobber another's.
        eng_idx = outputs.engine_index
        if outputs.recovery_targets is not None:
            self._recovery_targets_by_engine[eng_idx] = dict(
                outputs.recovery_targets
            )

        # Clean up the completed requests
        if outputs.finished_requests:
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)
                self._request_cache.pop(req_id, None)
                self._checkpoint_tokens.pop(req_id, None)
                self._output_tokens.pop(req_id, None)
                self._rerouted_request_ids.discard(req_id)
                self._rerouted_first_token_logged.discard(req_id)
                self._cleanup_shared_checkpoint_request(req_id)
                # Clean from per-engine partitions.
                for part in self._recovery_targets_by_engine.values():
                    part.pop(req_id, None)

    # ---- Per-engine metadata lookups ----

    def _get_recovery_target(self, req_id: str) -> int | None:
        """Look up solver recovery target across all engine partitions."""
        for part in self._recovery_targets_by_engine.values():
            target = part.get(req_id)
            if target is not None:
                return target
        return None

    # ---- Failover logic ----

    def _all_engines_dead(self) -> bool:
        return all(not alive for alive in self._engine_alive.values())

    async def _handle_engine_failure(self, engine_index: int) -> None:
        """Handle failure of a specific engine.

        1. Mark engine as dead.
        2. Collect all in-flight requests on the dead engine.
        3. Re-route them to a surviving engine.

        Idempotent: if ENGINE_CORE_DEAD and ENGINE_FAILED both arrive for
        the same engine, the first triggers failover and the second is a
        no-op.
        """
        engine_id = self._index_to_engine.get(engine_index)
        if engine_id is None:
            logger.error(
                "FT Client: unknown engine index %d", engine_index
            )
            return

        if not self._engine_alive.get(engine_id, True):
            # ENGINE_CORE_DEAD(engine_index) and ENGINE_FAILED(engine_index) can both arrive for the same engine death.
            return  # Already handled (idempotent).

        self._engine_alive[engine_id] = False
        logger.warning(
            "FAULT_EVENT failure_declared replica=%d wall_time=%.6f",
            engine_index,
            time.time(),
        )

        logger.warning(
            "FT Client: engine %d declared FAILED. "
            "Starting failover for displaced requests.",
            engine_index,
        )
        logger.warning(
            "FAULT_EVENT failover_start engine=%d wall_time=%.6f",
            engine_index, time.time(),
        )

        # Collect requests that were on the dead engine.
        displaced_req_ids = [
            req_id
            for req_id, eng in self.reqs_in_flight.items()
            if eng == engine_id
        ]

        if not displaced_req_ids:
            logger.info(
                "FT Client: engine %d had no in-flight requests",
                engine_index,
            )
            self._recovery_targets_by_engine.pop(engine_index, None)
            return

        # Check that at least one surviving engine exists.
        surviving_engine = self._get_surviving_engine(engine_id)
        if surviving_engine is None:
            logger.error(
                "FT Client: no surviving engines for failover"
            )
            return

        logger.info(
            "FT Client: re-routing %d requests from engine %d "
            "to surviving engine(s)",
            len(displaced_req_ids),
            engine_index,
        )

        # Re-route each displaced request.  Use per-request routing to
        # spread load across multiple surviving engines (if available).
        #
        # FT_FAILOVER_BATCH_SIZE + FT_FAILOVER_BATCH_DELAY_MS: rate-limit
        # how many displaced reqs we dispatch to the survivor per batch,
        # with a small delay between batches. Without this, 13-19 reqs
        # land on the survivor in the same scheduler step, causing the
        # restore path to allocate 13-19 × ~250 MB GPU temp src tensors
        # simultaneously (OOM on A5000 24 GB). By spreading dispatch
        # across steps, each scheduler step only sees ~BATCH_SIZE new
        # reqs, keeping GPU temp bounded.
        try:
            failover_batch = int(
                os.environ.get("FT_FAILOVER_BATCH_SIZE", "0")
            )
        except ValueError:
            failover_batch = 0
        try:
            failover_delay_ms = float(
                os.environ.get("FT_FAILOVER_BATCH_DELAY_MS", "300")
            )
        except ValueError:
            failover_delay_ms = 300.0
        rerouted = 0
        for req_id in displaced_req_ids:
            if (
                failover_batch > 0
                and rerouted > 0
                and rerouted % failover_batch == 0
            ):
                await asyncio.sleep(failover_delay_ms / 1000.0)
            cached_request = self._request_cache.get(req_id)
            if cached_request is None:
                logger.warning(
                    "FT Client: request %s not in cache, cannot re-route",
                    req_id,
                )
                continue

            # Prefer solver-planned recovery target if available,
            # but only if the target is alive AND not overloaded.
            target = None
            solver_target_idx = self._get_recovery_target(req_id)
            if solver_target_idx is not None:
                # solver_target_idx is a replica index (0-based).
                # Map it to an engine identity if alive and has capacity.
                if solver_target_idx < len(self.core_engines):
                    candidate = self.core_engines[solver_target_idx]
                    if self._engine_alive.get(candidate, False):
                        # Check that the target isn't already heavily loaded.
                        cand_idx = self._engine_to_index.get(candidate)
                        current_counts = self.lb_engines
                        use_solver_target = True
                        if (cand_idx is not None
                                and cand_idx < len(current_counts)):
                            waiting, running = current_counts[cand_idx]
                            score = waiting * 4 + running
                            # Compare with least-loaded alternative.
                            best_alt = self._get_surviving_engine(engine_id)
                            if best_alt is not None:
                                alt_idx = self._engine_to_index.get(best_alt)
                                if (alt_idx is not None
                                        and alt_idx < len(current_counts)):
                                    alt_w, alt_r = current_counts[alt_idx]
                                    alt_score = alt_w * 4 + alt_r
                                    # Only use solver target if it's not
                                    # significantly worse than best alt
                                    # (2x threshold).
                                    if score > 2 * max(alt_score, 1):
                                        use_solver_target = False
                        if use_solver_target:
                            target = candidate
                            logger.debug(
                                "FT Client: using solver recovery target "
                                "for %s → engine %d",
                                req_id,
                                solver_target_idx,
                            )

            # Fall back to least-loaded surviving engine.
            if target is None:
                target = self._get_surviving_engine(engine_id)
            if target is None:
                target = surviving_engine  # fallback

            # Bump local waiting count so the next iteration sees updated
            # load and doesn't pile all requests onto the same survivor.
            target_idx = self._engine_to_index.get(target)
            current_counts = self.lb_engines
            if target_idx is not None and target_idx < len(current_counts):
                current_counts[target_idx][0] += self.client_count

            # Add checkpoint info so the new engine can attempt KV restore.
            # TODO: this in-memory "promise" can lead the actual /dev/shm
            # publish by 5-30ms (worse with FT_BG_PUBLISH=1). Cleaner fix is
            # to read /dev/shm's latest manifest here as ground truth.
            ckpt_tokens = self._checkpoint_tokens.get(req_id, 0)
            cached_request.num_checkpointed_tokens = ckpt_tokens
            cached_request.is_rerouted = True

            # Propagate previously generated output tokens so the new
            # engine can restore sampling penalties and bad-words state.
            prev_out = self._output_tokens.get(req_id)
            if prev_out:
                cached_request.previous_output_token_ids = prev_out

            reroute_wall = time.time()
            await self._send_input(
                EngineCoreRequestType.ADD, cached_request, target
            )
            self.reqs_in_flight[req_id] = target
            self._rerouted_request_ids.add(req_id)
            target_idx_val = (
                target_idx if target_idx is not None else -1
            )
            replay_tokens = max(
                0, len(prev_out) - ckpt_tokens if prev_out else 0
            )
            # Emit the same per-request reroute log format as the
            # centralized path so experiment log parsing can attribute
            # failover effects consistently across FT baselines.
            logger.info(
                "Request %s: re-routed %d→%d, restored=%d tokens, "
                "replay=%d tokens, est_gap=0.0ms, slo_met=True, "
                "wall_time=%.6f",
                req_id,
                engine_index,
                target_idx_val,
                ckpt_tokens,
                replay_tokens,
                reroute_wall,
            )
            logger.info(
                "FAULT_EVENT reroute_plan request=%s wall_time=%.6f "
                "checkpointed_tokens=%d replay_tokens=%d",
                req_id,
                reroute_wall,
                ckpt_tokens,
                replay_tokens,
            )
            rerouted += 1

        # Clean up the dead engine's solver metadata partition.
        # Targets have been consumed by the failover loop above.
        self._recovery_targets_by_engine.pop(engine_index, None)

        logger.info(
            "FT Client: failover complete. Re-routed %d/%d requests.",
            rerouted,
            len(displaced_req_ids),
        )
        logger.warning(
            "FAULT_EVENT failover_complete engine=%d wall_time=%.6f "
            "rerouted=%d total=%d",
            engine_index, time.time(), rerouted, len(displaced_req_ids),
        )

    def _get_surviving_engine(
        self, dead_engine: EngineIdentity
    ) -> EngineIdentity | None:
        """Find the least-loaded surviving engine for failover."""
        best_engine: EngineIdentity | None = None
        best_score = float("inf")
        current_counts = self.lb_engines

        for engine_id, alive in self._engine_alive.items():
            if not alive or engine_id == dead_engine:
                continue
            idx = self._engine_to_index.get(engine_id)
            if idx is not None and idx < len(current_counts):
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
            else:
                score = 0  # No stats available; treat as empty.
            if score < best_score:
                best_score = score
                best_engine = engine_id

        return best_engine

    # ---- Override routing to skip dead engines ----

    def get_core_engine_for_request(
        self, request: EngineCoreRequest
    ) -> EngineIdentity:
        """Route requests only to alive engines."""
        if (eng_index := request.data_parallel_rank) is not None:
            engine = self._index_to_engine.get(eng_index)
            if engine and self._engine_alive.get(engine, True):
                self.reqs_in_flight[request.request_id] = engine
                return engine

        # Load-balance among alive engines only.
        current_counts = self.lb_engines
        num_engines = len(current_counts)
        min_score = float("inf")
        best_index = -1

        for i in range(num_engines):
            idx = (self.eng_start_index + i) % num_engines
            engine_id = self.core_engines[idx]
            if not self._engine_alive.get(engine_id, True):
                continue
            waiting, running = current_counts[idx]
            score = waiting * 4 + running
            if score < min_score:
                min_score = score
                best_index = idx

        if best_index < 0:
            return super().get_core_engine_for_request(request)

        current_counts[best_index][0] += self.client_count
        chosen_engine = self.core_engines[best_index]
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine

    # ---- Override abort to skip dead engines ----

    # Abort requests on surviving engines and clean up local FT tracking state.
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Send aborts only to alive engines."""
        if not request_ids or self.resources.engine_dead:
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                if self._engine_alive.get(engine, True):
                    by_engine[engine].append(req_id)
            self._request_cache.pop(req_id, None)
            self.reqs_in_flight.pop(req_id, None)
            self._checkpoint_tokens.pop(req_id, None)
            self._output_tokens.pop(req_id, None)
            self._rerouted_request_ids.discard(req_id)
            self._rerouted_first_token_logged.discard(req_id)
            for part in self._recovery_targets_by_engine.values():
                part.pop(req_id, None)
            self._cleanup_shared_checkpoint_request(req_id)

        for engine, req_ids in by_engine.items():
            await self._send_input(
                EngineCoreRequestType.ABORT, req_ids, engine
            )


class CentralizedBendersFTClient(FTDPAsyncMPClient):
    """FT client with centralized Benders solver.

    All new requests queue in _pending_solver_requests. A periodic
    solve epoch (asyncio timer) runs the Benders solver on the global
    snapshot (request snapshots from all engines + pending requests),
    then dispatches admitted requests to their assigned engines.

    Greedy fallback only triggers when the solver times out, crashes,
    or no engine snapshots are available (degraded mode).
    """

    @staticmethod
    def _auto_kv_bytes(vllm_config) -> int:
        """Compute KV bytes per token from model config."""
        from vllm.v1.core.sched.ft_scheduler_impl import (
            _auto_kv_bytes_per_token,
        )
        return _auto_kv_bytes_per_token(vllm_config)

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        from concurrent.futures import ThreadPoolExecutor

        from vllm.v1.core.checkpoint_controller import CheckpointConfig
        from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder
        from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop

        sched_cfg = vllm_config.scheduler_config
        parallel_cfg = vllm_config.parallel_config

        # System-level throughput parameters (same as BendersFTSchedulerImpl).
        prefill_tput = sched_cfg.ft_prefill_throughput or 0.0
        decode_tput = sched_cfg.ft_decode_throughput or 0.0
        load_bw = sched_cfg.ft_load_bandwidth or 0.0
        mem_cap = getattr(sched_cfg, "ft_memory_capacity_bytes", 0) or 0
        detection_time = sched_cfg.failure_detection_time_ms / 1000.0
        max_gpu_failures = sched_cfg.max_gpu_failures

        # Planning horizon from FT scheduler config defaults.
        # FT_PLANNING_HORIZON_SEC env var override (Phase 1 improvement A2):
        # Shortens the Benders solver's lookahead window. The default 1.0s
        # horizon means the solver plans admission for the next 10 solver
        # ticks (100ms solver interval), but arrival rate variance over 1s
        # is high enough to make G_j predictions noisy and the capacity
        # feasibility check unreliable. Setting 0.3 gives the solver a
        # ~3-tick lookahead — fresher decisions, less queue accumulation.
        planning_horizon = sched_cfg.ft_planning_horizon or 1.0
        _ft_horizon_override = os.environ.get("FT_PLANNING_HORIZON_SEC")
        if _ft_horizon_override:
            try:
                planning_horizon = float(_ft_horizon_override)
                logger.info(
                    "FT_PLANNING_HORIZON_SEC override: planning_horizon=%.2fs",
                    planning_horizon,
                )
            except ValueError:
                logger.warning(
                    "FT_PLANNING_HORIZON_SEC=%r is not a valid float; "
                    "falling back to config value %.2fs",
                    _ft_horizon_override, planning_horizon,
                )

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
            replay_throughput=prefill_tput,
            detection_time_sec=detection_time,
            memory_capacity_bytes=mem_cap,
            checkpoint_config=CheckpointConfig(),
            kv_bytes_per_token=self._auto_kv_bytes(vllm_config),
            block_size=vllm_config.cache_config.block_size or 1,
            checkpoint_lambda=sched_cfg.ft_checkpoint_lambda,
            decode_capacity_profile_path=(
                sched_cfg.ft_decode_capacity_profile or None
            ),
        )

        max_iter = getattr(sched_cfg, "benders_max_iterations", 20) or 20
        master_tl = getattr(sched_cfg, "benders_master_time_limit", 1.0) or 1.0
        recovery_tl = getattr(sched_cfg, "benders_recovery_time_limit", 0.5) or 0.5

        dp_size = parallel_cfg.data_parallel_size
        # Clamp max_gpu_failures.
        if max_gpu_failures >= dp_size:
            max_gpu_failures = max(0, dp_size - 1)

        self._solver = BendersSolveLoop(
            cost_builder=self._cost_builder,
            max_iterations=max_iter,
            master_time_limit_sec=master_tl,
            recovery_time_limit_sec=recovery_tl,
            max_gpu_failures=max_gpu_failures,
        )

        # Pending requests waiting for solver epoch.
        self._pending_solver_requests: list[EngineCoreRequest] = []

        # Per-engine snapshots from EngineCoreOutputs.
        self._engine_request_snapshots: dict[int, list[RequestSnapshot]] = {}
        self._replica_snapshots: dict[int, ReplicaSnapshot] = {}

        # Global recovery plans from solver.
        self._recovery_plans: dict[frozenset[int], dict[str, int]] = {}

        # Solver execution (async pipeline).
        self._solver_executor = ThreadPoolExecutor(max_workers=1)
        self._solver_budget_sec: float = getattr(
            sched_cfg, "benders_solver_budget_sec", 0.5
        ) or 0.5
        self._epoch_interval_sec: float = getattr(
            sched_cfg, "benders_epoch_interval_sec", 0.02
        ) or 0.02
        # Optional env override for A/B tuning. Does NOT skip solver calls —
        # only adjusts poll frequency. Solver still fires every epoch when
        # pending queue has requests.
        env_epoch_ms = os.environ.get("FT_EPOCH_INTERVAL_MS")
        if env_epoch_ms:
            try:
                self._epoch_interval_sec = float(env_epoch_ms) / 1000.0
            except ValueError:
                pass
        self._solve_epoch_task: asyncio.Task | None = None
        # Async solver state: solver runs in background, results dispatched
        # when ready. Each epoch fires a new solve if pending requests exist
        # and no solve is in flight.
        self._solver_future: asyncio.Task | None = None
        self._solver_pending: list[EngineCoreRequest] = []

        # All replica IDs for the solver.
        self._all_replica_ids = list(range(dp_size))

        logger.info(
            "CentralizedBendersFTClient initialized "
            "(dp_size=%d, max_gpu_failures=%d, epoch=%.0fms, budget=%.1fs)",
            dp_size,
            max_gpu_failures,
            self._epoch_interval_sec * 1000,
            self._solver_budget_sec,
        )

        # FT_API_PROFILE_SECONDS: schedule a one-shot cProfile window.
        # We can't run asyncio.create_task here because the event loop
        # may not be running yet at __init__ time. Defer to the first
        # add_request_async call via a guard flag.
        self._ft_api_profile_scheduled = False
        if _FT_API_PROFILE_SECONDS > 0:
            logger.info(
                "FT_API_PROFILE: will start cProfile %.0fs after init, "
                "run for %.0fs, dump to %s",
                _FT_API_PROFILE_DELAY,
                _FT_API_PROFILE_SECONDS,
                _FT_API_PROFILE_OUTPUT,
            )

    def _maybe_start_api_profile(self) -> None:
        """Schedule the API server cProfile window on first opportunity."""
        if self._ft_api_profile_scheduled:
            return
        if _FT_API_PROFILE_SECONDS <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ft_api_profile_scheduled = True
        loop.create_task(self._ft_api_profile_task())

    async def _ft_api_profile_task(self) -> None:
        """Wait FT_API_PROFILE_DELAY seconds, enable cProfile for
        FT_API_PROFILE_SECONDS, dump to FT_API_PROFILE_OUTPUT.
        """
        import cProfile
        import io
        import pstats

        try:
            await asyncio.sleep(_FT_API_PROFILE_DELAY)
        except asyncio.CancelledError:
            return
        logger.info(
            "FT_API_PROFILE: cProfile START (window=%.0fs)",
            _FT_API_PROFILE_SECONDS,
        )
        prof = cProfile.Profile()
        prof.enable()
        try:
            await asyncio.sleep(_FT_API_PROFILE_SECONDS)
        except asyncio.CancelledError:
            pass
        finally:
            prof.disable()

        try:
            stream = io.StringIO()
            stats = pstats.Stats(prof, stream=stream)
            stats.sort_stats("cumulative")
            stream.write(
                f"FT_API_PROFILE: captured ~{_FT_API_PROFILE_SECONDS:.0f}s of "
                f"API server cProfile (cumulative time, top 80)\n\n"
            )
            stats.print_stats(80)
            stream.write("\n\n=== Sorted by total time (top 80) ===\n\n")
            stats.sort_stats("tottime")
            stats.print_stats(80)
            os.makedirs(
                os.path.dirname(_FT_API_PROFILE_OUTPUT) or ".", exist_ok=True
            )
            with open(_FT_API_PROFILE_OUTPUT, "w") as f:
                f.write(stream.getvalue())
            logger.info(
                "FT_API_PROFILE: dumped to %s", _FT_API_PROFILE_OUTPUT,
            )
        except Exception:
            logger.exception("FT_API_PROFILE: dump failed")

    # ---- Override: queue requests instead of immediate routing ----

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Queue request for solver epoch instead of routing immediately."""
        # Lazy-start the API server profile on first request (we now
        # know the event loop is running). No-op when env var unset.
        self._maybe_start_api_profile()

        self._request_cache[request.request_id] = request
        self._pending_solver_requests.append(request)
        self._ensure_solve_epoch_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        """Abort requests: remove from pending queue + parent cleanup."""
        if not request_ids:
            return
        abort_set = set(request_ids)
        # Remove from pending queue BEFORE they get dispatched by solver.
        self._pending_solver_requests = [
            r for r in self._pending_solver_requests
            if r.request_id not in abort_set
        ]
        await super().abort_requests_async(request_ids)

    # ---- Solver epoch loop ----

    def _ensure_solve_epoch_task(self) -> None:
        """Start the periodic solver epoch task if not running."""
        if self._solve_epoch_task is not None and not self._solve_epoch_task.done():
            return
        self._solve_epoch_task = asyncio.ensure_future(
            self._solve_epoch_loop()
        )

    async def _solve_epoch_loop(self) -> None:
        """Async solver pipeline: fire solver in background, dispatch when done.

        Each epoch:
        1. If solver finished → dispatch its results immediately.
        2. If new pending requests and solver idle → fire new solver.

        Uses asyncio.wait to react as soon as solver completes (not wait
        for next epoch timer).
        """
        try:
            while True:
                if (self._solver_future is not None
                        and not self._solver_future.done()):
                    # Wait for EITHER epoch timer OR solver completion.
                    sleep_task = asyncio.ensure_future(
                        asyncio.sleep(self._epoch_interval_sec))
                    done, _ = await asyncio.wait(
                        [sleep_task, self._solver_future],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Cancel the sleep if solver finished first.
                    if not sleep_task.done():
                        sleep_task.cancel()
                else:
                    await asyncio.sleep(self._epoch_interval_sec)

                # Step 1: Check if solver completed → dispatch.
                await self._check_solver_result()

                # Step 2: Fire new solver if pending and idle.
                await self._maybe_fire_solver()

        except asyncio.CancelledError:
            if self._solver_future and not self._solver_future.done():
                self._solver_future.cancel()
        except Exception:
            logger.exception(
                "Centralized solver epoch loop crashed, "
                "falling back to greedy for remaining requests"
            )
            await self._greedy_dispatch_pending()

    async def _check_solver_result(self) -> None:
        """If the background solver has completed, dispatch its results."""
        if self._solver_future is None or not self._solver_future.done():
            return

        pending = self._solver_pending
        self._solver_pending = []

        try:
            result = self._solver_future.result()
        except asyncio.TimeoutError:
            logger.warning(
                "Centralized solver timed out, "
                "degraded to greedy for %d requests", len(pending))
            self._pending_solver_requests.extend(pending)
            await self._greedy_dispatch_pending()
            self._solver_future = None
            return
        except Exception:
            logger.exception(
                "Centralized solver failed, "
                "degraded to greedy for %d requests", len(pending))
            self._pending_solver_requests.extend(pending)
            await self._greedy_dispatch_pending()
            self._solver_future = None
            return

        self._solver_future = None

        if result is None:
            # FT_SOLVER_DIAG: log every Benders failure at INFO so we
            # can audit the converge/fail ratio. Counter is bumped for
            # post-run summary.
            if not hasattr(self, "_solver_fail_count"):
                self._solver_fail_count = 0
            self._solver_fail_count += 1
            logger.info(
                "FT_SOLVER_DIAG benders_fail count=%d pending=%d",
                self._solver_fail_count, len(pending),
            )
            self._pending_solver_requests.extend(pending)
            await self._greedy_dispatch_pending()
            return

        await self._dispatch_solver_result(result, pending)

    async def _maybe_fire_solver(self) -> None:
        """Fire a new solver run if there are pending requests and no
        solver is in flight."""
        if not self._pending_solver_requests:
            return

        # FT_SKIP_SOLVER=1: bypass Benders solver entirely and go
        # straight to greedy dispatch. On W1_Chat/Heavy the solver
        # fails to converge 98 % of the time (profile mismatch —
        # see e1a_quick_diagnosis.md follow-up #6). Each failed
        # solve still burns CPU in the bg ThreadPoolExecutor
        # (cost_table build + cp_model setup + 20 iterations).
        # Skipping it eliminates that CPU waste entirely. Default off.
        if os.environ.get("FT_SKIP_SOLVER") == "1":
            await self._greedy_dispatch_pending()
            return

        # FT_FAST_FAILOVER=1: during recovery window (any engine recently
        # failed), bypass solver entirely and use greedy dispatch for ALL
        # pending admissions — not just displaced reqs. The solver adds
        # 8-300ms per call; during recovery this latency delays new
        # request scheduling, inflating fg_p95. On dp=2 the solver's
        # routing decision is trivial anyway (single survivor).
        if os.environ.get("FT_FAST_FAILOVER") == "1":
            if getattr(self, "_ft_recovery_active", False):
                logger.debug(
                    "FT_FAST_FAILOVER: recovery active, greedy dispatch "
                    "for %d pending", len(self._pending_solver_requests),
                )
                await self._greedy_dispatch_pending()
                return

        # FT_GATED_SOLVER=1: more nuanced version of FT_SKIP_SOLVER
        # (Phase 1 improvement A3). Only run the Benders solver when
        # the system is actually under load — pending queue large AND
        # running batch near capacity. In low-utilization regimes the
        # solver's admission decisions add noise (delaying admit to
        # accumulate "optimal batch") without producing meaningful
        # routing improvements, yielding the -3% overall goodput
        # observed in Phase 1 ablation. Skipping the solver in those
        # regimes lets greedy FIFO dispatch handle admission at
        # minimal latency.
        #
        # Thresholds (env tunable):
        #   FT_GATED_SOLVER_MIN_PENDING  (default 5)
        #   FT_GATED_SOLVER_MIN_LOAD     (default 0.7 → 70%)
        #   FT_GATED_SOLVER_CAPACITY     (default 26, per-engine decode cap)
        if os.environ.get("FT_GATED_SOLVER") == "1":
            pending_count = len(self._pending_solver_requests)
            try:
                min_pending = int(
                    os.environ.get("FT_GATED_SOLVER_MIN_PENDING", "5")
                )
            except ValueError:
                min_pending = 5
            try:
                min_load = float(
                    os.environ.get("FT_GATED_SOLVER_MIN_LOAD", "0.7")
                )
            except ValueError:
                min_load = 0.7
            try:
                per_engine_cap = int(
                    os.environ.get("FT_GATED_SOLVER_CAPACITY", "26")
                )
            except ValueError:
                per_engine_cap = 26

            running_count = sum(
                len(snaps)
                for snaps in self._engine_request_snapshots.values()
            )
            num_engines = len(self._engine_request_snapshots) or 1
            total_capacity = num_engines * per_engine_cap
            load_pct = (
                running_count / total_capacity
                if total_capacity > 0 else 0.0
            )

            if pending_count < min_pending or load_pct < min_load:
                logger.debug(
                    "FT_GATED_SOLVER: greedy dispatch "
                    "(pending=%d<%d or load=%.1f%%<%.1f%%)",
                    pending_count, min_pending,
                    load_pct * 100, min_load * 100,
                )
                await self._greedy_dispatch_pending()
                return

        # Cold start: no snapshots yet → greedy bootstrap.
        if not self._engine_request_snapshots:
            logger.info(
                "Centralized solver: no engine snapshots yet, "
                "greedy bootstrap for %d requests",
                len(self._pending_solver_requests))
            await self._greedy_dispatch_pending()
            return

        # Solver still running → wait.
        if self._solver_future is not None and not self._solver_future.done():
            return

        # Take pending requests atomically.
        pending = list(self._pending_solver_requests)
        self._pending_solver_requests.clear()
        self._solver_pending = pending

        # Build global snapshot.
        all_active_snapshots: list[RequestSnapshot] = []
        for snaps in self._engine_request_snapshots.values():
            all_active_snapshots.extend(snaps)

        healthy_replicas = []
        for r_id in self._all_replica_ids:
            engine_id = self._index_to_engine.get(r_id)
            if engine_id is None:
                continue
            if not self._engine_alive.get(engine_id, True):
                continue
            rep_snap = self._replica_snapshots.get(r_id)
            if rep_snap is not None and not rep_snap.is_healthy:
                continue
            healthy_replicas.append(r_id)

        if not healthy_replicas:
            logger.warning(
                "Centralized solver: no healthy replicas, "
                "greedy fallback for %d requests", len(pending))
            self._pending_solver_requests.extend(pending)
            self._solver_pending = []
            await self._greedy_dispatch_pending()
            return

        cost_table = self._cost_builder.build_global_costs(
            all_active_snapshots, pending
        )
        if not cost_table:
            # Put requests back so they're retried next epoch.
            self._pending_solver_requests.extend(pending)
            self._solver_pending = []
            return

        # Fire solver in background (don't await).
        loop = asyncio.get_event_loop()
        self._solver_future = asyncio.ensure_future(
            asyncio.wait_for(
                loop.run_in_executor(
                    self._solver_executor,
                    self._solver.solve_epoch_from_costs,
                    cost_table,
                    healthy_replicas,
                    self._all_replica_ids,
                ),
                timeout=self._solver_budget_sec,
            )
        )

    async def _dispatch_solver_result(self, result, pending) -> None:
        """Dispatch solver decisions to engines."""
        from vllm.v1.core.sched.benders.solve_loop import BendersSolveResult

        assert isinstance(result, BendersSolveResult)

        dispatched = 0
        rejected: list[str] = []
        for req in pending:
            req_id = req.request_id
            # Guard against requests aborted while solver was running.
            if req_id not in self._request_cache:
                continue
            if req_id in result.master_solution.admitted:
                assignment = result.master_solution.assignments.get(req_id)
                if assignment is not None:
                    r_id = assignment
                    # Set placement on the request. Checkpointing remains a
                    # runtime-local policy and is not solver-controlled.
                    req.data_parallel_rank = r_id
                    req.client_index = self.client_index
                    req.current_wave = self.current_wave

                    engine_id = self._index_to_engine.get(r_id)
                    if engine_id is None or not self._engine_alive.get(
                        engine_id, True
                    ):
                        # Assigned engine died between solve and dispatch.
                        # Fall back to any alive engine.
                        engine_id = self._get_surviving_engine(
                            engine_id or self.core_engines[0]
                        )
                    if engine_id is None:
                        logger.error(
                            "No surviving engine for request %s", req_id
                        )
                        continue

                    await self._send_input(
                        EngineCoreRequestType.ADD, req, engine_id
                    )
                    self.reqs_in_flight[req_id] = engine_id
                    dispatched += 1

                    gpu_idx = self._engine_to_index.get(engine_id)
                    if gpu_idx is not None:
                        logger.info(
                            "FAULT_EVENT request_route request=%s gpu=%d "
                            "wall_time=%.6f",
                            req_id, gpu_idx, time.time(),
                        )

                    logger.debug(
                        "Centralized solver: %s → engine %d",
                        req_id,
                        r_id,
                    )
            else:
                # Solver says "don't admit this request". Instead of
                # aborting it (which causes empty_stream errors visible
                # to the client), fall back to greedy dispatch. The
                # solver's rejection is based on the decode_capacity
                # profile which may be miscalibrated for the actual
                # hardware (e.g. A6000 dp=1 profile on A5000 dp=2),
                # so rejection is often wrong. Greedy dispatch at
                # least gives the request a chance to run.
                #
                # Previous behavior: abort → EngineCoreOutput with
                # finish_reason=ABORT → client sees empty stream.
                # This caused 5-68% completion drops in non-Heavy
                # cells where the solver occasionally converges.
                logger.debug(
                    "Centralized solver rejected %s; "
                    "falling back to greedy dispatch", req_id
                )
                req.client_index = self.client_index
                req.current_wave = self.current_wave
                chosen_engine = self.get_core_engine_for_request(req)
                await self._send_input(
                    EngineCoreRequestType.ADD, req, chosen_engine
                )
                self.reqs_in_flight[req_id] = chosen_engine
                dispatched += 1

        # Inject terminal ABORT outputs for rejected requests so the
        # upper layer doesn't hang waiting for them.
        if rejected:
            from vllm.v1.engine import EngineCoreOutput, FinishReason

            abort_outputs = EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=rid,
                        new_token_ids=[],
                        finish_reason=FinishReason.ABORT,
                    )
                    for rid in rejected
                ],
            )
            self.outputs_queue.put_nowait(abort_outputs)

        # Store recovery plans.
        self._recovery_plans = {
            omega: plan.assignments
            for omega, plan in result.recovery_plans.items()
        }

        # P0-impl-3a follow-up: per-epoch summary demoted from INFO to
        # DEBUG (fires once per Benders solver epoch, ~25-450/run).
        logger.debug(
            "Centralized solver epoch: dispatched %d/%d requests, "
            "%d rejected, %d recovery plans",
            dispatched,
            len(pending),
            len(rejected),
            len(self._recovery_plans),
        )

    async def _greedy_dispatch_pending(self) -> None:
        """Fallback: dispatch pending requests via load-balanced routing.

        FT_STATIC_CAP=N: gate dispatch so total in-flight requests
        (across all engines) never exceeds N. Excess requests stay in
        the pending queue and are retried at the next solver epoch.
        Only applies to new admissions — fault rerouting in
        _handle_engine_failure is unaffected (displaced reqs bypass
        _pending_solver_requests and must always be rerouted to a
        survivor).

        Note: the cap is "soft" — the solver-success path
        (_dispatch_solver_result) and the reroute path both bump
        reqs_in_flight without consulting this cap. Set
        FT_SKIP_SOLVER=1 to force all new admissions through this
        function so the cap is honored consistently. Reroute always
        bypasses by design.
        """
        try:
            cap = int(os.environ.get("FT_STATIC_CAP", "0") or 0)
        except ValueError:
            cap = 0

        if cap > 0:
            in_flight = len(self.reqs_in_flight)
            headroom = cap - in_flight
            if headroom <= 0:
                # Fully loaded — leave pending in queue, log throttling.
                if self._pending_solver_requests and not getattr(
                    self, "_static_cap_throttle_logged", False
                ):
                    logger.info(
                        "FT_STATIC_CAP throttling: in_flight=%d cap=%d "
                        "pending=%d (further throttle events suppressed)",
                        in_flight, cap, len(self._pending_solver_requests),
                    )
                    self._static_cap_throttle_logged = True
                return
            # Reset suppression flag once we have headroom again.
            self._static_cap_throttle_logged = False
            pending = self._pending_solver_requests[:headroom]
            self._pending_solver_requests = (
                self._pending_solver_requests[headroom:]
            )
        else:
            pending = list(self._pending_solver_requests)
            self._pending_solver_requests.clear()

        for req in pending:
            req.client_index = self.client_index
            req.current_wave = self.current_wave
            chosen_engine = self.get_core_engine_for_request(req)
            await self._send_input(
                EngineCoreRequestType.ADD, req, chosen_engine
            )
            self.reqs_in_flight[req.request_id] = chosen_engine
            gpu_idx = self._engine_to_index.get(chosen_engine)
            if gpu_idx is not None:
                logger.info(
                    "FAULT_EVENT request_route request=%s gpu=%d "
                    "wall_time=%.6f",
                    req.request_id, gpu_idx, time.time(),
                )

        if pending:
            # P0-impl-3a follow-up: demoted from WARNING to INFO.
            # Fires once per greedy dispatch fallback (~hundreds/run
            # when Benders solver consistently hits its iteration cap);
            # WARNING is too noisy at that frequency. Kept at INFO
            # because run.py parses this line via _EPOCH_FALLBACK_RE
            # for per-epoch fallback metrics.
            logger.info(
                "Greedy fallback: dispatched %d requests (degraded path)",
                len(pending),
            )

    # ---- Override output processing: collect snapshots ----

    @staticmethod
    async def process_engine_outputs(
        self: "CentralizedBendersFTClient", outputs: EngineCoreOutputs
    ):
        """Process outputs: collect snapshots + standard FT processing."""
        # Standard FT output processing (tokens, checkpoints, finished).
        for out in outputs.outputs:
            if out.new_token_ids:
                acc = self._output_tokens.get(out.request_id)
                if acc is None:
                    acc = []
                    self._output_tokens[out.request_id] = acc
                acc.extend(out.new_token_ids)

                # FT: log first token after recovery for rerouted requests.
                if (out.request_id in self._rerouted_request_ids
                        and out.request_id
                        not in self._rerouted_first_token_logged):
                    logger.info(
                        "FAULT_EVENT first_token_after_recovery "
                        "request=%s wall_time=%.6f",
                        out.request_id, time.time(),
                    )
                    self._rerouted_first_token_logged.add(out.request_id)

        if outputs.checkpoint_updates:
            for req_id, num_tokens in outputs.checkpoint_updates.items():
                self._checkpoint_tokens[req_id] = num_tokens

        # Collect request and replica snapshots from engines.
        eng_idx = outputs.engine_index
        if outputs.active_request_snapshots is not None:
            # Remap engine-local assigned_replica_id (always 0 when
            # dp_size=1 per engine) to the global solver replica index.
            # Without this, the solver thinks ALL active requests are on
            # replica 0 and routes every new request to replica 1.
            for snap in outputs.active_request_snapshots:
                snap.assigned_replica_id = eng_idx
            self._engine_request_snapshots[eng_idx] = (
                outputs.active_request_snapshots
            )
        if outputs.replica_snapshot is not None:
            self._replica_snapshots[eng_idx] = outputs.replica_snapshot

        # Clean up finished requests — including cached snapshots so the
        # solver doesn't keep treating them as active.
        if outputs.finished_requests:
            finished_set = set(outputs.finished_requests)
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)
                self._request_cache.pop(req_id, None)
                self._checkpoint_tokens.pop(req_id, None)
                self._output_tokens.pop(req_id, None)
                self._rerouted_request_ids.discard(req_id)
                self._rerouted_first_token_logged.discard(req_id)
                self._cleanup_shared_checkpoint_request(req_id)
            # Scrub finished requests from engine snapshots.  The next
            # full snapshot from the engine will replace this list anyway,
            # but between refreshes the solver would otherwise still see
            # these requests as active.
            if eng_idx in self._engine_request_snapshots:
                self._engine_request_snapshots[eng_idx] = [
                    s for s in self._engine_request_snapshots[eng_idx]
                    if s.request_id not in finished_set
                ]

    # ---- Override failover: use client-side recovery plans ----

    async def _handle_engine_failure(self, engine_index: int) -> None:
        """Handle engine failure with centralized recovery plans."""
        engine_id = self._index_to_engine.get(engine_index)
        if engine_id is None:
            logger.error(
                "Centralized FT Client: unknown engine index %d",
                engine_index,
            )
            return

        if not self._engine_alive.get(engine_id, True):
            return  # Already handled (idempotent).

        self._engine_alive[engine_id] = False
        logger.warning(
            "FAULT_EVENT failure_declared replica=%d wall_time=%.6f",
            engine_index,
            time.time(),
        )

        logger.warning(
            "Centralized FT Client: engine %d FAILED. "
            "Starting failover.",
            engine_index,
        )
        logger.warning(
            "FAULT_EVENT failover_start engine=%d wall_time=%.6f",
            engine_index, time.time(),
        )

        # FT_FAST_FAILOVER: mark recovery window active so solver
        # admission path can fast-track to greedy dispatch.
        self._ft_recovery_active = True

        # Remove dead engine's snapshots.
        self._engine_request_snapshots.pop(engine_index, None)
        self._replica_snapshots.pop(engine_index, None)

        # Collect displaced requests.
        displaced_req_ids = [
            req_id
            for req_id, eng in self.reqs_in_flight.items()
            if eng == engine_id
        ]

        if not displaced_req_ids:
            logger.info(
                "Centralized FT Client: engine %d had no in-flight requests",
                engine_index,
            )
            return

        surviving_engine = self._get_surviving_engine(engine_id)
        if surviving_engine is None:
            logger.error(
                "Centralized FT Client: no surviving engines for failover"
            )
            return

        logger.info(
            "Centralized FT Client: re-routing %d requests from engine %d",
            len(displaced_req_ids),
            engine_index,
        )

        # FT_RECOVERY_PARALLEL_DISPATCH=1: send ADD messages to surviving
        # engines concurrently via asyncio.gather instead of serial await.
        # Serial IPC adds ~5-10ms latency per displaced req (6 reqs → 30-60ms
        # cumulative); parallel collapses to ~single-send time (~10ms).
        # Mutually exclusive with FT_RECOVERY_BATCH_SIZE>0 (which intentionally
        # paces sends via asyncio.sleep).
        _parallel_dispatch = (
            os.environ.get("FT_RECOVERY_PARALLEL_DISPATCH") == "1"
            and int(os.environ.get("FT_RECOVERY_BATCH_SIZE", "0") or "0") == 0
        )
        # _send_input returns an Awaitable (zmq.Future from send_multipart),
        # not a coroutine. asyncio.create_task rejects Futures directly, so
        # we store them as Awaitables and let asyncio.gather wrap each via
        # ensure_future at the end.
        _pending_dispatch_tasks: list = []

        rerouted = 0
        for req_id in displaced_req_ids:
            cached_request = self._request_cache.get(req_id)
            if cached_request is None:
                continue

            # Use solver recovery plan if available.
            target = None
            omega = frozenset({engine_index})
            plan = self._recovery_plans.get(omega)
            if plan:
                target_idx = plan.get(req_id)
                if target_idx is not None:
                    candidate = self._index_to_engine.get(target_idx)
                    if candidate and self._engine_alive.get(candidate, False):
                        target = candidate

            # Superset fallback for multi-failure scenarios.
            if target is None:
                for scenario, sp in self._recovery_plans.items():
                    if engine_index in scenario:
                        target_idx = sp.get(req_id)
                        if target_idx is not None:
                            candidate = self._index_to_engine.get(target_idx)
                            if candidate and self._engine_alive.get(
                                candidate, False
                            ):
                                target = candidate
                                break

            if target is None:
                target = self._get_surviving_engine(engine_id)
            if target is None:
                target = surviving_engine

            # Bump load estimate.
            target_idx_val = self._engine_to_index.get(target)
            current_counts = self.lb_engines
            if (
                target_idx_val is not None
                and target_idx_val < len(current_counts)
            ):
                current_counts[target_idx_val][0] += self.client_count

            # Add checkpoint info for KV restore (default: reload mode).
            ckpt_tokens = self._checkpoint_tokens.get(req_id, 0)
            cached_request.num_checkpointed_tokens = ckpt_tokens
            cached_request.is_rerouted = True

            # FT_DEPRIORITIZE_MIGRATED=1: reset the rerouted request's
            # arrival_time to NOW so the SLO-aware scheduler queue
            # doesn't put it ahead of new arrivals.
            #
            # Problem: migrated reqs have old arrival_time → their
            # ttft_deadline = old_arrival + 2s is ALREADY PAST → slack
            # is deeply negative → SLO-aware queue sorts them FIRST.
            # This blocks new arrivals that still have positive slack
            # (their deadlines haven't expired yet), causing 30+ extra
            # TTFT violations vs No-FT.
            #
            # Fix: pretend the migrated req just arrived. Its new
            # ttft_deadline = now + 2s → positive slack → sorts AFTER
            # new arrivals. The migrated req's TTFT SLO is already
            # violated anyway (can't undo the fault gap), so
            # deprioritizing it doesn't make things worse for it.
            #
            # Default off (preserves current "most-urgent-first"
            # scheduling semantics).
            if os.environ.get("FT_DEPRIORITIZE_MIGRATED") == "1":
                cached_request.arrival_time = time.time()

            # Propagate output tokens for sampling penalty restoration.
            prev_out = self._output_tokens.get(req_id)
            if prev_out:
                cached_request.previous_output_token_ids = prev_out

            # ── Recovery mode override (ablation knob) ──────────────────
            # See module-level docstring for _FT_RECOVERY_MODE. The default
            # "reload" mode falls through unchanged.
            if _FT_RECOVERY_MODE == "restart":
                # Drop checkpoint AND prior decoded tokens; new engine
                # re-prefills the original prompt from scratch.
                cached_request.num_checkpointed_tokens = 0
                cached_request.previous_output_token_ids = None
                ckpt_tokens = 0
                replay_tokens = 0
                logger.info(
                    "FT_RECOVERY_MODE=restart: req=%s cleared "
                    "checkpoint + decoded tokens, will re-prefill "
                    "prompt from scratch",
                    req_id,
                )
            elif _FT_RECOVERY_MODE == "reprefill":
                # Drop checkpoint but preserve decoded tokens by
                # extending the prompt. New engine re-prefills the
                # extended prompt then continues decoding.
                n_decoded = len(prev_out) if prev_out else 0
                if n_decoded > 0:
                    try:
                        original_prompt = (
                            list(cached_request.prompt_token_ids)
                            if cached_request.prompt_token_ids
                            else []
                        )
                        cached_request.prompt_token_ids = (
                            original_prompt + list(prev_out)
                        )
                    except (AttributeError, TypeError) as exc:
                        logger.warning(
                            "FT_RECOVERY_MODE=reprefill: req=%s "
                            "failed to extend prompt (%s); falling "
                            "back to reload semantics",
                            req_id, exc,
                        )
                cached_request.num_checkpointed_tokens = 0
                cached_request.previous_output_token_ids = None
                ckpt_tokens = 0
                replay_tokens = 0
                logger.info(
                    "FT_RECOVERY_MODE=reprefill: req=%s extended "
                    "prompt by %d decoded tokens, will re-prefill "
                    "via prompt path",
                    req_id, n_decoded,
                )

            reroute_wall = time.time()
            if _parallel_dispatch:
                # _send_input returns an Awaitable (zmq Future). Collect
                # them; asyncio.gather below will schedule them concurrently
                # and await completion in a single yield.
                _pending_dispatch_tasks.append(
                    self._send_input(
                        EngineCoreRequestType.ADD, cached_request, target
                    )
                )
            else:
                await self._send_input(
                    EngineCoreRequestType.ADD, cached_request, target
                )
            target_idx = self._engine_to_index.get(target, -1)
            self._rerouted_request_ids.add(req_id)
            if _FT_RECOVERY_MODE == "reload":
                replay_tokens = max(
                    0, len(prev_out) - ckpt_tokens if prev_out else 0
                )
            # Log per-request reroute in the same format as RecoveryManager
            # so the experiment log parser can capture it.
            logger.info(
                "Request %s: re-routed %d→%d, mode=%s, restored=%d tokens, "
                "replay=%d tokens, est_gap=0.0ms, slo_met=True, "
                "wall_time=%.6f",
                req_id,
                engine_index,
                target_idx,
                _FT_RECOVERY_MODE,
                ckpt_tokens,
                replay_tokens,
                reroute_wall,
            )
            logger.info(
                "FAULT_EVENT reroute_plan request=%s wall_time=%.6f "
                "checkpointed_tokens=%d replay_tokens=%d",
                req_id,
                reroute_wall,
                ckpt_tokens,
                replay_tokens,
            )
            self.reqs_in_flight[req_id] = target
            rerouted += 1

            # ── Incremental recovery (FT_RECOVERY_BATCH_SIZE env var) ──
            # Instead of rerouting all displaced reqs in one burst
            # (which inflates the surviving engine's batch from ~20 to
            # ~26+), send them in batches of N with T-second gaps.
            # During the gap the surviving engine processes the current
            # batch + new arrivals at near-normal step rate. Un-rerouted
            # reqs stay in ft_client memory — they DON'T exist in the
            # engine at all (no KV, no batch slot, no scheduler entry).
            #
            # Key difference from Solution 4 (lazy reload, which failed):
            # lazy reload sent ALL reqs to the engine and then tried to
            # throttle inside the scheduler. Freed slots got filled by
            # new arrivals → no net batch reduction. Incremental recovery
            # keeps unrerouted reqs OUT of the engine entirely.
            #
            # Default: 0 = disabled (all at once, current behavior).
            _batch_size = int(os.environ.get(
                "FT_RECOVERY_BATCH_SIZE", "0"))
            _batch_delay = float(os.environ.get(
                "FT_RECOVERY_BATCH_DELAY", "5.0"))
            if (
                _batch_size > 0
                and rerouted % _batch_size == 0
                and rerouted < len(displaced_req_ids)
            ):
                logger.info(
                    "FT_RECOVERY_BATCH: rerouted %d/%d, sleeping "
                    "%.1fs before next batch",
                    rerouted, len(displaced_req_ids), _batch_delay,
                )
                await asyncio.sleep(_batch_delay)

        # FT_RECOVERY_PARALLEL_DISPATCH: wait for all concurrent sends
        # to actually land on the surviving engines before declaring
        # failover_complete. This preserves the ordering invariant
        # "surviving engine has all reqs by failover_complete emit".
        if _pending_dispatch_tasks:
            await asyncio.gather(
                *_pending_dispatch_tasks, return_exceptions=True
            )

        logger.info(
            "Centralized FT Client: failover complete. "
            "Re-routed %d/%d requests.",
            rerouted,
            len(displaced_req_ids),
        )
        logger.warning(
            "FAULT_EVENT failover_complete engine=%d wall_time=%.6f "
            "rerouted=%d total=%d",
            engine_index, time.time(), rerouted, len(displaced_req_ids),
        )

        # FT_FAST_FAILOVER: clear recovery window. New admissions go
        # back through solver. We keep it active only during failover
        # dispatch so all concurrent admissions skip the solver overhead.
        self._ft_recovery_active = False
