#!/usr/bin/env python3

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

from vllm.v1.core.sched.ft_scheduler import FaultTolerantScheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.worker.gpu_worker import Worker


class _DummyRequestPool:
    def __init__(self, requests):
        self._requests = requests

    def get_request(self, req_id):
        return self._requests.get(req_id)


class _DummyCheckpointController:
    def __init__(self):
        self.recorded = []

    def record_checkpoint(self, request):
        self.recorded.append(request.request_id)


def test_capture_checkpoint_plan_preserves_block_ids_and_uses_latest_tokens():
    request = SimpleNamespace(
        request_id="r1",
        num_computed_tokens=9,
        num_checkpointed_tokens=0,
    )
    controller = _DummyCheckpointController()
    metadata_updates = []

    ft_scheduler = SimpleNamespace(
        request_pool=_DummyRequestPool({"r1": request}),
        checkpoint_controller=controller,
        update_checkpoint_metadata=lambda updates: metadata_updates.append(updates),
    )

    scheduler = SimpleNamespace(
        ft_scheduler=ft_scheduler,
        get_checkpoint_requests=lambda: [("r1", [11, 12], 5)],
    )

    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler

    captured_rpc = []

    def _collective_rpc(method, args=(), **_kwargs):
        captured_rpc.append((method, args))
        return [[("r1", 2048)]]

    core.collective_rpc = _collective_rpc

    plan = core._capture_ft_checkpoint_plan()
    assert plan == [("r1", [11, 12])]

    # Simulate update_from_output having advanced the request before the
    # checkpoint copy actually happens.
    request.num_computed_tokens = 17

    ckpt_updates = core._maybe_ft_checkpoint(plan)

    assert ckpt_updates == {"r1": 17}
    assert captured_rpc == [
        ("checkpoint_kv_blocks", ([("r1", [11, 12], 17)],)),
    ]
    assert controller.recorded == ["r1"]
    assert metadata_updates == [{"r1": (17, 2048)}]


def test_step_with_batch_queue_restores_before_execute_and_queues_plan():
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
    )

    exec_future = Future()
    sample_future = Future()

    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: scheduler_output,
        get_grammar_bitmask=lambda _out: None,
    )

    model_executor = SimpleNamespace(
        execute_model=lambda _out, non_block=True: exec_future,
        sample_tokens=lambda _grammar, non_block=True: sample_future,
    )

    restore_calls = []

    core = EngineCore.__new__(EngineCore)
    core.batch_queue = deque()
    core.batch_queue_size = 2
    core.scheduler = scheduler
    core.model_executor = model_executor
    core.is_ec_producer = False
    core.is_pooling_model = False
    core.use_spec_decode = False
    core._process_ft_pending_restores = (
        lambda out: restore_calls.append(out)
    )
    core._capture_ft_checkpoint_plan = lambda: [("r1", [5, 6])]

    output, model_executed = core.step_with_batch_queue()

    assert output is None
    assert model_executed is True
    assert restore_calls == [scheduler_output]
    assert len(core.batch_queue) == 1
    queued = core.batch_queue[0]
    assert queued[1] is scheduler_output
    assert queued[3] == [("r1", [5, 6])]


def test_step_with_batch_queue_forwards_checkpoint_plan_to_post_process():
    future = Future()
    future.set_result(object())
    exec_future = Future()

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
    )

    forwarded_plans = []

    scheduler = SimpleNamespace(
        has_requests=lambda: False,
        update_from_output=lambda _sched, _model: {},
    )

    core = EngineCore.__new__(EngineCore)
    core.batch_queue = deque([
        (future, scheduler_output, exec_future, [("r2", [9])]),
    ])
    core.batch_queue_size = 2
    core.scheduler = scheduler
    core._process_aborts_queue = lambda: None
    core.log_error_detail = lambda _out: nullcontext()
    core.log_iteration_details = lambda _out: nullcontext()
    core.use_spec_decode = False
    core._ft_post_process = (
        lambda outputs, checkpoint_plan=None:
        forwarded_plans.append(checkpoint_plan)
    )

    outputs, _ = core.step_with_batch_queue()

    assert outputs == {}
    assert forwarded_plans == [[("r2", [9])]]


def test_ft_scheduler_uses_public_kv_block_id_api():
    request = SimpleNamespace(request_id="r1")
    kv_cache_manager = SimpleNamespace(
        req_to_blocks={"r1": [SimpleNamespace(block_id=3), SimpleNamespace(block_id=4)]},
        get_block_ids=lambda req_id: ([11, 12],) if req_id == "r1" else ([],),
    )

    block_ids = FaultTolerantScheduler._get_request_kv_block_ids(
        SimpleNamespace(),
        request,
        kv_cache_manager,
    )

    assert block_ids == [11, 12]


def test_process_ft_pending_restores_uses_public_kv_block_id_api():
    request = SimpleNamespace(
        request_id="r1",
        num_computed_tokens=0,
        num_checkpointed_tokens=6,
    )
    nrd = SimpleNamespace(req_id="r1", num_computed_tokens=0)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[nrd],
        num_scheduled_tokens={"r1": 10},
        total_num_scheduled_tokens=10,
    )

    kv_cache_manager = SimpleNamespace(
        req_to_blocks={"r1": [SimpleNamespace(block_id=21), SimpleNamespace(block_id=22)]},
        get_block_ids=lambda req_id: ([21, 22],) if req_id == "r1" else ([],),
    )

    base = SimpleNamespace(
        kv_cache_manager=kv_cache_manager,
        requests={"r1": request},
    )
    core = EngineCore.__new__(EngineCore)
    core.scheduler = SimpleNamespace(
        ft_scheduler=SimpleNamespace(),
        _base=base,
    )
    core._ft_pending_restores = [("r1", 6)]
    restore_calls = []

    def _collective_rpc(method, args=(), **_kwargs):
        restore_calls.append((method, args))
        return [6]

    core.collective_rpc = _collective_rpc

    core._process_ft_pending_restores(scheduler_output)

    assert restore_calls == [("restore_kv_blocks", ("r1", [21, 22]))]
    assert request.num_computed_tokens == 6
    assert request.num_checkpointed_tokens == 6
    assert scheduler_output.num_scheduled_tokens["r1"] == 4
    assert scheduler_output.total_num_scheduled_tokens == 4
    assert nrd.num_computed_tokens == 6
    assert core._ft_pending_restores == []


def test_gpu_worker_delegates_ft_checkpoint_rpc_to_model_runner():
    worker = Worker.__new__(Worker)
    calls = []
    worker.model_runner = SimpleNamespace(
        checkpoint_kv_blocks=lambda request_block_map: (
            calls.append(("checkpoint", request_block_map)) or [("r1", 4096)]
        ),
        restore_kv_blocks=lambda request_id, target_block_ids: (
            calls.append(("restore", request_id, target_block_ids)) or 12
        ),
    )

    ckpt_result = worker.checkpoint_kv_blocks([("r1", [1, 2], 8)])
    restore_result = worker.restore_kv_blocks("r1", [7, 8])

    assert ckpt_result == [("r1", 4096)]
    assert restore_result == 12
    assert calls == [
        ("checkpoint", [("r1", [1, 2], 8)]),
        ("restore", "r1", [7, 8]),
    ]
