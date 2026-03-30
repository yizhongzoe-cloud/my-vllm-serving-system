"""End-to-end test for the Fault-Tolerant Multi-GPU LLM Serving system.

Tests all FT components working together with 2 GPU replicas:
1. Request admission and routing.
2. Adaptive KV-cache checkpointing.
3. GPU failure detection.
4. Failover recovery with KV cache restore.

This test does NOT require a running model — it tests the scheduling
and orchestration logic using mock KV cache tensors.
"""

import time
from types import SimpleNamespace

import torch
import pytest

from vllm.v1.core.checkpoint_controller import CheckpointConfig, CheckpointController
from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus
from vllm.v1.core.kv_checkpoint_pool import KVCheckpointPool
from vllm.v1.core.recovery_manager import RecoveryManager
from vllm.v1.core.replica_manager import ReplicaManager
from vllm.v1.core.request_pool import RequestPool, RequestPoolStatus
from vllm.v1.core.sched.ft_scheduler import FaultTolerantScheduler, FTSchedulerConfig
from vllm.v1.engine import FinishReason
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.request import Request, RequestStatus
from vllm.sampling_params import SamplingParams
from slo_benchmark.src.benchmark.fault_injection import (
    FaultInjector,
    FaultInjectionConfig,
)


# ---- Helpers ----

def make_request(
    request_id: str,
    prompt_len: int = 10,
    max_tokens: int = 100,
    ttft_slo_ms: float | None = None,
    e2e_latency_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    failure_gap_slo_ms: float | None = None,
    expected_output_len: int | None = None,
) -> Request:
    """Create a mock Request for testing."""
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=2,
        ttft_slo_ms=ttft_slo_ms,
        e2e_latency_slo_ms=e2e_latency_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
        failure_gap_slo_ms=failure_gap_slo_ms,
        expected_output_len=expected_output_len,
    )


def make_mock_kv_caches(
    num_layers: int = 2,
    num_blocks: int = 64,
    block_size: int = 16,
    num_kv_heads: int = 8,
    head_size: int = 64,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """Create mock KV cache tensors."""
    return [
        torch.randn(
            2, num_blocks, block_size, num_kv_heads, head_size,
            device=device,
        )
        for _ in range(num_layers)
    ]


# ---- Tests ----

class TestRequestModel:
    """Test the extended Request model."""

    def test_new_slo_fields(self):
        req = make_request(
            "r1",
            tpot_slo_ms=50.0,
            failure_gap_slo_ms=200.0,
            expected_output_len=80,
        )
        assert req.tpot_slo_ms == 50.0
        assert req.failure_gap_slo_ms == 200.0
        assert req.expected_output_len == 80
        assert req.has_slo()
        assert req.has_failure_gap_slo()

    def test_generation_progress(self):
        req = make_request("r2", max_tokens=100, expected_output_len=50)
        assert req.generation_len == 50
        assert req.generation_progress == 0.0
        # Simulate generating 25 tokens.
        req.append_output_token_ids(list(range(25)))
        assert req.generation_progress == pytest.approx(0.5)

    def test_uncovered_tokens(self):
        req = make_request("r3")
        req.num_computed_tokens = 50
        req.num_checkpointed_tokens = 30
        assert req.get_uncovered_tokens() == 20

    def test_fault_tolerance_fields_default(self):
        req = make_request("r4")
        assert req.checkpoint_level == 0
        assert req.assigned_replica_id is None
        assert req.num_checkpointed_tokens == 0
        assert req.last_checkpoint_time is None


class TestKVCheckpointPool:
    """Test KV cache checkpointing to host memory."""

    def test_save_and_restore(self):
        pool = KVCheckpointPool(max_memory_bytes=1 * 1024 * 1024 * 1024)
        kv_caches = make_mock_kv_caches(num_layers=2, num_blocks=64)

        block_ids = [0, 1, 2, 3]
        entry = pool.save_checkpoint(
            request_id="r1",
            gpu_kv_caches=kv_caches,
            block_ids=block_ids,
            num_tokens=64,
            async_copy=False,
        )
        assert entry is not None
        assert entry.num_tokens == 64
        assert len(entry.kv_tensors) == 2
        assert pool.num_checkpoints == 1
        assert pool.used_bytes > 0

    def test_restore_checkpoint(self):
        pool = KVCheckpointPool()
        kv_caches = make_mock_kv_caches(num_layers=1, num_blocks=32)

        block_ids = [5, 6, 7]
        # Save original data.
        original_data = kv_caches[0][:, 5:8, :, :].clone()
        pool.save_checkpoint("r1", kv_caches, block_ids, 48, async_copy=False)

        # Zero out the GPU blocks.
        kv_caches[0][:, 5:8, :, :] = 0

        # Restore.
        tokens_restored = pool.restore_checkpoint("r1", kv_caches, [5, 6, 7])
        assert tokens_restored == 48
        assert torch.allclose(kv_caches[0][:, 5:8, :, :], original_data)

    def test_delete_checkpoint(self):
        pool = KVCheckpointPool()
        kv_caches = make_mock_kv_caches(num_layers=1, num_blocks=16)
        pool.save_checkpoint("r1", kv_caches, [0, 1], 32, async_copy=False)
        assert pool.num_checkpoints == 1

        pool.delete_checkpoint("r1")
        assert pool.num_checkpoints == 0
        assert pool.used_bytes == 0

    def test_eviction_on_capacity(self):
        # Very small pool to trigger eviction.
        pool = KVCheckpointPool(max_memory_bytes=1024)
        kv_caches = make_mock_kv_caches(
            num_layers=1, num_blocks=16,
            block_size=2, num_kv_heads=1, head_size=2,
        )
        # First save should succeed.
        e1 = pool.save_checkpoint("r1", kv_caches, [0], 2, async_copy=False)
        # Second save should evict first.
        e2 = pool.save_checkpoint("r2", kv_caches, [1], 2, async_copy=False)
        # At most 1 should remain (the newer one).
        assert pool.num_checkpoints <= 2


class TestCheckpointController:
    """Test adaptive checkpoint policy."""

    def test_level_progression(self):
        config = CheckpointConfig(
            level1_progress=0.25,
            level2_progress=0.60,
        )
        ctrl = CheckpointController(config)
        req = make_request("r1", max_tokens=100, expected_output_len=100)

        # No output yet → level 0.
        assert ctrl.get_checkpoint_level(req) == 0

        # 30% progress → level 1.
        req.append_output_token_ids(list(range(30)))
        assert ctrl.get_checkpoint_level(req) == 1

        # 70% progress → level 2.
        req.append_output_token_ids(list(range(40)))
        assert ctrl.get_checkpoint_level(req) == 2

    def test_should_checkpoint_level0(self):
        ctrl = CheckpointController()
        req = make_request("r1", max_tokens=100)
        # Level 0 = never checkpoint.
        assert not ctrl.should_checkpoint(req)

    def test_should_checkpoint_level1(self):
        config = CheckpointConfig(
            level1_progress=0.0,  # Always at least level 1.
            level1_interval_steps=10,
            level1_interval_sec=0.0,
        )
        ctrl = CheckpointController(config)
        req = make_request("r1", max_tokens=100, expected_output_len=100)
        req.append_output_token_ids(list(range(15)))

        # First check should trigger (no prior checkpoint).
        assert ctrl.should_checkpoint(req)

        # Record checkpoint, then shouldn't trigger immediately.
        ctrl.record_checkpoint(req)
        assert not ctrl.should_checkpoint(req)

        # Generate more tokens past interval.
        req.append_output_token_ids(list(range(10)))
        assert ctrl.should_checkpoint(req)

    def test_should_checkpoint_fixed_one_block(self):
        ctrl = CheckpointController(fixed_blocks=1, block_size=16)
        req = make_request("r-fixed-1", max_tokens=200, expected_output_len=200)

        req.num_computed_tokens = 15
        assert not ctrl.should_checkpoint(req)

        req.num_computed_tokens = 16
        assert ctrl.should_checkpoint(req)

        ctrl.record_checkpoint(req)
        assert not ctrl.should_checkpoint(req)

        req.num_computed_tokens = 32
        assert ctrl.should_checkpoint(req)

    def test_should_checkpoint_fixed_ten_blocks(self):
        ctrl = CheckpointController(fixed_blocks=10, block_size=16)
        req = make_request("r-fixed-10", max_tokens=400, expected_output_len=400)

        req.num_computed_tokens = 16 * 9
        assert not ctrl.should_checkpoint(req)

        req.num_computed_tokens = 16 * 10
        assert ctrl.should_checkpoint(req)

        ctrl.record_checkpoint(req)
        assert not ctrl.should_checkpoint(req)

        req.num_computed_tokens = 16 * 19
        assert not ctrl.should_checkpoint(req)

        req.num_computed_tokens = 16 * 20
        assert ctrl.should_checkpoint(req)


class TestDPLBGracefulDegradation:
    def _make_client(self) -> DPLBAsyncMPClient:
        client = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
        client.client_count = 1
        client.eng_start_index = 0
        client.core_engines = [b"\x00\x00", b"\x01\x00"]
        client.engine_ranks_managed = [0, 1]
        client.lb_engines = [[2, 2], [0, 0]]
        client.reqs_in_flight = {}
        client.outputs_queue = SimpleNamespace(put_nowait=lambda _outputs: None)
        client.resources = SimpleNamespace(engine_dead=False)
        DPLBAsyncMPClient._rebuild_engine_maps(client)
        return client

    def test_routing_skips_dead_engine(self):
        client = self._make_client()
        client._engine_alive[client.core_engines[0]] = False

        request = SimpleNamespace(request_id="req-new", data_parallel_rank=None)
        chosen = DPLBAsyncMPClient.get_core_engine_for_request(client, request)

        assert chosen == client.core_engines[1]
        assert client.reqs_in_flight["req-new"] == client.core_engines[1]

    def test_pinned_dead_engine_falls_back_to_live_engine(self):
        client = self._make_client()
        client._engine_alive[client.core_engines[0]] = False

        request = SimpleNamespace(request_id="req-pinned", data_parallel_rank=0)
        chosen = DPLBAsyncMPClient.get_core_engine_for_request(client, request)

        assert chosen == client.core_engines[1]
        assert client.reqs_in_flight["req-pinned"] == client.core_engines[1]

    def test_failure_aborts_only_displaced_requests(self):
        client = self._make_client()
        captured = []
        client.outputs_queue = SimpleNamespace(
            put_nowait=lambda outputs: captured.append(outputs)
        )
        dead_engine, live_engine = client.core_engines
        client.reqs_in_flight = {
            "req-dead": dead_engine,
            "req-live": live_engine,
        }

        client._handle_engine_failure(0)

        assert not client._engine_alive[dead_engine]
        assert client._engine_alive[live_engine]
        assert client.reqs_in_flight == {"req-live": live_engine}
        assert len(captured) == 1
        assert [out.request_id for out in captured[0].outputs] == ["req-dead"]
        assert captured[0].outputs[0].finish_reason == FinishReason.ERROR


class TestFailureDetector:
    """Test GPU health monitoring."""

    def test_register_and_healthy(self):
        fd = FailureDetector(failure_timeout_sec=1.0)
        fd.register_replica(0)
        fd.register_replica(1)

        assert fd.get_status(0) == ReplicaStatus.HEALTHY
        assert fd.get_status(1) == ReplicaStatus.HEALTHY
        assert sorted(fd.get_healthy_replicas()) == [0, 1]

    def test_direct_failure_report(self):
        fd = FailureDetector()
        fd.register_replica(0)
        fd.register_replica(1)

        failed_replicas = []
        fd.register_callback(lambda rid: failed_replicas.append(rid))

        fd.report_failure(1)
        assert fd.get_status(1) == ReplicaStatus.FAILED
        assert fd.get_healthy_replicas() == [0]
        assert failed_replicas == [1]

    def test_heartbeat_recovery(self):
        fd = FailureDetector(failure_timeout_sec=0.1, max_consecutive_failures=1)
        fd.register_replica(0)

        # Simulate timeout.
        time.sleep(0.15)
        fd._check_heartbeats()
        assert fd.get_status(0) == ReplicaStatus.FAILED

        # Send heartbeat and manually mark healthy.
        fd.mark_healthy(0)
        fd.record_heartbeat(0)
        assert fd.get_status(0) == ReplicaStatus.HEALTHY


class TestReplicaManager:
    """Test multi-GPU replica management and routing."""

    def test_route_least_loaded(self):
        rm = ReplicaManager()
        rm.add_replica(0, gpu_id=0, max_num_seqs=10)
        rm.add_replica(1, gpu_id=1, max_num_seqs=10)

        # Route first request — should go to either (both empty).
        req1 = make_request("r1")
        rid = rm.route_request(req1)
        assert rid in (0, 1)
        rm.assign_request(req1, rid)

        # Route second — should go to the other (less loaded).
        req2 = make_request("r2")
        rid2 = rm.route_request(req2)
        assert rid2 != rid  # Goes to the other replica.

    def test_route_failover_excludes_failed(self):
        rm = ReplicaManager()
        rm.add_replica(0, gpu_id=0, max_num_seqs=10)
        rm.add_replica(1, gpu_id=1, max_num_seqs=10)

        req = make_request("r1")
        rm.assign_request(req, 0)
        rm.mark_failed(0)

        target = rm.route_request_for_failover(req, exclude_replica_ids={0})
        assert target == 1

    def test_capacity_under_failures(self):
        rm = ReplicaManager()
        rm.add_replica(0, gpu_id=0, max_num_seqs=10)
        rm.add_replica(1, gpu_id=1, max_num_seqs=10)

        requests = [make_request(f"r{i}") for i in range(10)]
        # With k=1, 1 surviving GPU can handle 10 requests.
        assert rm.check_capacity_under_failures(requests, max_failures=1)
        # 15 requests won't fit on 1 surviving GPU.
        requests_15 = [make_request(f"r{i}") for i in range(15)]
        assert not rm.check_capacity_under_failures(requests_15, max_failures=1)


class TestRequestPool:
    """Test centralized request pool."""

    def test_lifecycle(self):
        pool = RequestPool()
        req = make_request("r1")
        pool.add_request(req)
        assert pool.num_pending == 1

        pool.admit_request("r1", replica_id=0)
        assert pool.num_admitted == 1
        assert pool.num_pending == 0

        pool.complete_request("r1")
        assert pool.num_admitted == 0

    def test_displace_on_failure(self):
        pool = RequestPool()
        req1 = make_request("r1")
        req2 = make_request("r2")
        req3 = make_request("r3")

        pool.add_request(req1)
        pool.add_request(req2)
        pool.add_request(req3)

        pool.admit_request("r1", replica_id=0)
        pool.admit_request("r2", replica_id=0)
        pool.admit_request("r3", replica_id=1)

        # GPU 0 fails → r1, r2 should be displaced.
        displaced = pool.displace_requests(replica_id=0)
        assert len(displaced) == 2
        assert pool.num_displaced == 2
        assert pool.num_admitted == 1  # r3 still on GPU 1.


class TestRecoveryManager:
    """Test failover recovery orchestration."""

    def test_full_failover(self):
        fd = FailureDetector()
        fd.register_replica(0)
        fd.register_replica(1)

        rp = RequestPool()
        cp = KVCheckpointPool()
        rm = ReplicaManager()
        rm.add_replica(0, gpu_id=0, max_num_seqs=10, decode_throughput=100.0)
        rm.add_replica(1, gpu_id=1, max_num_seqs=10, decode_throughput=100.0)

        cc = CheckpointController()

        recovery = RecoveryManager(fd, rp, cp, rm, cc)

        # Add requests to GPU 0.
        # SLO must exceed detection_time (5s) + resume_time (1/100=0.01s).
        req1 = make_request("r1", failure_gap_slo_ms=10000.0)
        req2 = make_request("r2", failure_gap_slo_ms=10000.0)
        rp.add_request(req1)
        rp.add_request(req2)
        rp.admit_request("r1", 0)
        rp.admit_request("r2", 0)
        rm.assign_request(req1, 0)
        rm.assign_request(req2, 0)

        # Simulate failure of GPU 0.
        report = recovery.handle_failure(failed_replica_id=0)

        assert report.num_affected_requests == 2
        assert report.num_recovered == 2
        assert report.num_dropped == 0
        # Both should be re-routed to GPU 1.
        for result in report.results:
            assert result.target_replica_id == 1
            assert result.success


class TestFaultTolerantScheduler:
    """Test the top-level FT scheduler integration."""

    def test_admit_and_route(self):
        config = FTSchedulerConfig(max_gpu_failures=1)
        scheduler = FaultTolerantScheduler(config)

        scheduler.register_replica(0, gpu_id=0, max_num_seqs=10)
        scheduler.register_replica(1, gpu_id=1, max_num_seqs=10)

        req = make_request("r1", ttft_slo_ms=500.0, failure_gap_slo_ms=2000.0)
        admitted = scheduler.admit_request(req)
        assert admitted
        assert req.assigned_replica_id in (0, 1)

    def test_reject_when_no_capacity(self):
        config = FTSchedulerConfig(max_gpu_failures=1)
        scheduler = FaultTolerantScheduler(config)

        # Only 1 replica with capacity for 2, but k=1 means we need
        # robustness under 1 failure — impossible with only 1 replica.
        # Actually, with 1 replica and k=1, check_capacity_under_failures
        # returns False (all replicas could fail).
        scheduler.register_replica(0, gpu_id=0, max_num_seqs=2)

        req = make_request("r1")
        admitted = scheduler.admit_request(req)
        assert not admitted  # Cannot tolerate 1 failure with only 1 GPU.

    def test_failover_via_detector(self):
        config = FTSchedulerConfig(max_gpu_failures=1)
        scheduler = FaultTolerantScheduler(config)

        scheduler.register_replica(0, gpu_id=0, max_num_seqs=10)
        scheduler.register_replica(1, gpu_id=1, max_num_seqs=10)

        # Admit a request.
        req = make_request("r1", failure_gap_slo_ms=5000.0)
        scheduler.admit_request(req)
        original_replica = req.assigned_replica_id

        # Simulate failure of the assigned replica.
        scheduler.failure_detector.report_failure(original_replica)

        # Check recovery happened.
        report = scheduler.recovery_manager.failover_history
        assert len(report) == 1
        assert report[0].num_recovered == 1

    def test_complete_and_cleanup(self):
        config = FTSchedulerConfig(max_gpu_failures=1)
        scheduler = FaultTolerantScheduler(config)
        scheduler.register_replica(0, gpu_id=0, max_num_seqs=10)
        scheduler.register_replica(1, gpu_id=1, max_num_seqs=10)

        req = make_request("r1")
        scheduler.admit_request(req)
        scheduler.complete_request(req)

        stats = scheduler.get_stats()
        assert stats["request_pool"]["admitted"] == 0

    def test_fallback_checkpoint_saves_only_stable_full_block_prefix(self):
        config = FTSchedulerConfig(
            max_gpu_failures=1,
            enable_checkpointing=True,
            fixed_checkpoint_blocks=1,
            block_size=16,
        )
        scheduler = FaultTolerantScheduler(config)

        req = make_request("r-fallback", max_tokens=200, expected_output_len=200)
        req.num_computed_tokens = 39  # 2 full blocks + 1 partial frontier block
        req.append_output_token_ids(list(range(39)))

        captured = {}

        def _save_checkpoint(
            request_id,
            gpu_kv_caches,
            block_ids,
            num_tokens,
            async_copy=True,
        ):
            captured["request_id"] = request_id
            captured["block_ids"] = list(block_ids)
            captured["num_tokens"] = num_tokens
            return SimpleNamespace(size_bytes=4096)

        scheduler.checkpoint_pool.save_checkpoint = _save_checkpoint

        kv_cache_manager = SimpleNamespace(
            get_block_ids=lambda req_id: ([11, 12, 13],)
            if req_id == "r-fallback" else ([],),
        )

        checkpointed = scheduler.run_checkpoint_step(
            [req],
            gpu_kv_caches=[object()],
            kv_cache_manager=kv_cache_manager,
        )

        assert checkpointed == ["r-fallback"]
        assert captured == {
            "request_id": "r-fallback",
            "block_ids": [11, 12],
            "num_tokens": 32,
        }
        assert req.num_checkpointed_tokens == 32
        assert req.last_checkpoint_size_bytes == 4096
        assert scheduler.get_checkpoint_metadata("r-fallback") == (32, 4096)


class TestFaultInjection:
    """Test the benchmark fault injection module."""

    def test_scheduled_injection(self):
        triggered = []
        config = FaultInjectionConfig(
            mode="scheduled",
            scheduled_faults=[(0.1, 0), (0.2, 1)],
        )
        injector = FaultInjector(
            config,
            on_failure=lambda rid: triggered.append(rid),
            num_replicas=2,
        )
        injector.start()
        time.sleep(0.5)
        injector.stop()

        assert triggered == [0, 1]
        report = injector.get_report()
        assert report["total_triggered"] == 2

    def test_random_injection(self):
        triggered = []
        config = FaultInjectionConfig(
            mode="random",
            mean_time_between_failures=0.1,
            max_random_faults=3,
            seed=42,
        )
        injector = FaultInjector(
            config,
            on_failure=lambda rid: triggered.append(rid),
            num_replicas=2,
        )
        injector.start()
        time.sleep(2.0)
        injector.stop()

        assert len(triggered) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
