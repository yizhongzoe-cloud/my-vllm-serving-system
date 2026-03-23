# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fault-Tolerant Coordinator for multi-replica vLLM serving.

Extends the DP coordinator with per-engine health monitoring.
When an engine stops sending messages (heartbeat timeout), the coordinator
publishes an ENGINE_FAILED event so that front-end clients can re-route
displaced requests to surviving engines.
"""

import copy
import multiprocessing
import time
import weakref

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import get_mp_context, set_process_title
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

logger = init_logger(__name__)


class FTCoordinator:
    """FT-aware coordinator that monitors engine health and signals failover.

    Drop-in replacement for DPCoordinator.  Uses FTCoordinatorProc which
    extends the coordinator loop with per-engine liveness tracking and
    publishes ENGINE_FAILED events on the front-end PUB socket.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        enable_wave_coordination: bool = True,
        failure_timeout_sec: float = 10.0,
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "FTCoordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip

        local_only = not parallel_config.local_engines_only
        front_publish_address = get_engine_client_zmq_addr(
            local_only=local_only, host=host
        )

        local_only_eng = dp_size == parallel_config.data_parallel_size_local
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host)
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host)

        context = get_mp_context()
        self.proc: multiprocessing.Process = context.Process(
            target=FTCoordinatorProc.run_coordinator,
            name="VLLM_FT_Coordinator",
            kwargs={
                "engine_count": dp_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "enable_wave_coordination": enable_wave_coordination,
                "failure_timeout_sec": failure_timeout_sec,
            },
            daemon=True,
        )
        self.proc.start()

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])

    def get_stats_publish_address(self) -> str:
        return self.stats_publish_address

    def get_engine_socket_addresses(self) -> tuple[str, str]:
        return self.coord_in_address, self.coord_out_address

    def close(self):
        self._finalizer()


class FTEngineState:
    """Per-engine tracking state."""

    def __init__(self, engine_index: int):
        self.engine_index = engine_index
        self.request_counts = [0, 0]  # [waiting, running]
        self.last_message_time: float = time.time()
        self.is_alive: bool = True


class FTCoordinatorProc:
    """FT coordinator process with health monitoring.

    Extends DPCoordinatorProc logic with:
    - Per-engine last-message-time tracking.
    - Timeout-based failure detection.
    - Publishes ("ENGINE_FAILED", engine_index) to front-end on failure.
    """

    def __init__(
        self,
        engine_count: int,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
        failure_timeout_sec: float = 10.0,
    ):
        set_process_title("FTCoordinator")
        self.ctx = zmq.Context()

        self.engines = [FTEngineState(i) for i in range(engine_count)]
        self.stats_update_interval_ms = min_stats_update_interval_ms
        self.enable_wave_coordination = enable_wave_coordination
        self.failure_timeout_sec = failure_timeout_sec

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
        failure_timeout_sec: float = 10.0,
    ):
        coordinator = FTCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
            failure_timeout_sec=failure_timeout_sec,
        )
        try:
            coordinator.process_input_socket(
                front_publish_address,
                back_output_address,
                back_publish_address,
            )
        except KeyboardInterrupt:
            logger.info("FT Coordinator process exiting")

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        current_wave = 0
        engines_running = False

        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "FT Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            publish_back.send(b"READY")

            logger.info(
                "All engine subscriptions received by FT coordinator "
                "(failure_timeout=%.1fs)",
                self.failure_timeout_sec,
            )

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0

            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                wait_for = (
                    self.stats_update_interval_ms if stats_changed else 5000
                )
                min_timeout = 50 if last_step_counts is None else 0

                events = poller.poll(
                    timeout=max(min_timeout, wait_for - elapsed)
                )

                # ---- Health check: detect engine timeouts ----
                self._check_engine_health(publish_front)

                if not events:
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._snapshot_engine_loads()
                        stats_changed = False

                    to_publish = (
                        engine_req_counts_list,
                        current_wave,
                        engines_running,
                    )
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for i in range(new_engine_count - current_count):
                                self.engines.append(
                                    FTEngineState(current_count + i)
                                )
                            engines_running = False
                            logger.info(
                                "FTCoordinator scaled up from %s to %s",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "FTCoordinator scaled down from %s to %s",
                                current_count,
                                new_engine_count,
                            )
                        continue

                    if self.enable_wave_coordination:
                        engine_to_exclude, wave = decoded
                        if not engines_running:
                            if wave < current_wave:
                                engine_to_exclude = None
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                if output_back in events:
                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs
                    assert outputs.utility_output is None

                    eng_index = outputs.engine_index

                    # ---- FT: record heartbeat ----
                    if 0 <= eng_index < len(self.engines):
                        self.engines[eng_index].last_message_time = time.time()

                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._snapshot_engine_loads(
                                    do_copy=True
                                )
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d "
                                "(expected > (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, wave, eng_index
                            )

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))

    def _check_engine_health(self, publish_front: zmq.Socket) -> None:
        """Check for engine timeouts and publish ENGINE_FAILED events."""
        now = time.time()
        for engine in self.engines:
            if not engine.is_alive:
                continue
            if now - engine.last_message_time > self.failure_timeout_sec:
                engine.is_alive = False
                logger.warning(
                    "FT Coordinator: engine %d timed out (%.1fs since "
                    "last message). Declaring ENGINE_FAILED.",
                    engine.engine_index,
                    now - engine.last_message_time,
                )
                # Publish failure event to front-end clients.
                failure_msg = (
                    "ENGINE_FAILED",
                    engine.engine_index,
                )
                publish_front.send(msgspec.msgpack.encode(failure_msg))

    @staticmethod
    def _send_start_wave(
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None
    ):
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))
        socket.send_multipart(
            (EngineCoreRequestType.START_DP_WAVE.value, wave_encoded)
        )

    def _snapshot_engine_loads(self, do_copy=False) -> list[list[int]]:
        if do_copy:
            return [copy.copy(e.request_counts) for e in self.engines]
        return [e.request_counts for e in self.engines]
