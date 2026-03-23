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

        logger.info(
            "FTDPAsyncMPClient initialized with %d engines",
            len(self.core_engines),
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
                        "Marking dead (failover via ZMQ path).",
                        eng_idx,
                    )
                    if eng_id is not None:
                        _self._engine_alive[eng_id] = False

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
        for i, engine in enumerate(self.core_engines):
            new_i2e[i] = engine
            new_e2i[engine] = i
            # Preserve alive status for engines that existed before;
            # new engines default to alive.
            new_alive[engine] = self._engine_alive.get(engine, True)
        self._engine_alive = new_alive
        self._index_to_engine = new_i2e
        self._engine_to_index = new_e2i

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

    # ---- Override output processing: handle ENGINE_CORE_DEAD ----

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
        # Update checkpoint token tracking from EngineCore reports.
        if outputs.checkpoint_updates:
            for req_id, num_tokens in outputs.checkpoint_updates.items():
                self._checkpoint_tokens[req_id] = num_tokens

        # Clean up the completed requests   
        if outputs.finished_requests:
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)
                self._request_cache.pop(req_id, None)
                self._checkpoint_tokens.pop(req_id, None)

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
            "FT Client: engine %d declared FAILED. "
            "Starting failover for displaced requests.",
            engine_index,
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
        rerouted = 0
        for req_id in displaced_req_ids:
            cached_request = self._request_cache.get(req_id)
            if cached_request is None:
                logger.warning(
                    "FT Client: request %s not in cache, cannot re-route",
                    req_id,
                )
                continue

            # Pick best surviving engine per-request (load-aware).
            target = self._get_surviving_engine(engine_id)
            if target is None:
                target = surviving_engine  # fallback

            # Bump local waiting count so the next iteration sees updated
            # load and doesn't pile all requests onto the same survivor.
            target_idx = self._engine_to_index.get(target)
            current_counts = self.lb_engines
            if target_idx is not None and target_idx < len(current_counts):
                current_counts[target_idx][0] += self.client_count

            # add checkpoint info
            # engine can attempt KV restore instead of full recomputation.
            ckpt_tokens = self._checkpoint_tokens.get(req_id, 0)
            cached_request.num_checkpointed_tokens = ckpt_tokens

            await self._send_input(
                EngineCoreRequestType.ADD, cached_request, target
            )
            self.reqs_in_flight[req_id] = target
            rerouted += 1

        logger.info(
            "FT Client: failover complete. Re-routed %d/%d requests.",
            rerouted,
            len(displaced_req_ids),
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

        for engine, req_ids in by_engine.items():
            await self._send_input(
                EngineCoreRequestType.ABORT, req_ids, engine
            )
