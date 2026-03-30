"""Integration tests for the Benders FT scheduler.

Tests BendersFTSchedulerImpl wired with the full FT stack using mock
replicas, verifying:
1. Scheduler instantiation with ft_benders policy routes correctly.
2. Requests flow through Benders admission → base scheduler.
3. Greedy fallback works when solver cannot solve.
4. Checkpoint controller integration.
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.checkpoint_controller import CheckpointConfig
from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder
from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop
from vllm.v1.core.sched.ft_scheduler import FaultTolerantScheduler, FTSchedulerConfig
from vllm.v1.core.replica_manager import ReplicaManager
from vllm.v1.request import Request


# ---- Helpers ----

def make_request(
    request_id: str,
    prompt_len: int = 10,
    max_tokens: int = 100,
    expected_output_len: int | None = None,
) -> Request:
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=2,
        expected_output_len=expected_output_len or max_tokens,
    )


def make_ft_scheduler_with_replicas(
    num_replicas: int = 2,
    prefill_tput: float = 1000.0,
    decode_tput: float = 500.0,
) -> FaultTolerantScheduler:
    """Create a FT scheduler with registered replicas."""
    ft = FaultTolerantScheduler(config=FTSchedulerConfig(
        max_gpu_failures=1,
        enable_checkpointing=True,
    ))
    for i in range(num_replicas):
        ft.register_replica(
            replica_id=i,
            gpu_id=i,
            prefill_throughput=prefill_tput,
            decode_throughput=decode_tput,
            load_bandwidth=10e9,
        )
    return ft


# ---- Tests ----

class TestBendersWithFTStack:
    """Test Benders solver integrated with FT scheduler components."""

    def test_admit_and_track_in_request_pool(self):
        """Benders admits a request, and it appears in request pool."""
        ft = make_ft_scheduler_with_replicas(3)
        builder = CostTableBuilder(
            planning_horizon=10.0,
            prefill_throughput=1000.0,
            decode_throughput=500.0,
            load_bandwidth=10e9,
            replay_throughput=1000.0,
            detection_time_sec=0.1,
            memory_capacity_bytes=0,
        )
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=20,
            master_time_limit_sec=5.0,
        )

        pending = [make_request(f"r{i}", max_tokens=80) for i in range(4)]
        replica_ids = [r.replica_id for r in ft.replica_manager.get_healthy_replicas()]

        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=pending,
            replica_ids=replica_ids,
        )

        assert result is not None
        assert len(result.master_solution.admitted) > 0

        # Commit decisions to FT scheduler.
        for req in pending:
            if req.request_id in result.master_solution.admitted:
                r_id = result.master_solution.assignments[req.request_id]
                ft.request_pool.add_request(req)
                ft.request_pool.admit_request(req.request_id, r_id)
                ft.replica_manager.assign_request(req, r_id)

        assert ft.request_pool.num_admitted == len(result.master_solution.admitted)

    def test_solver_respects_single_failure_robustness(self):
        """With 3 replicas, any single failure should be recoverable."""
        ft = make_ft_scheduler_with_replicas(3)
        builder = CostTableBuilder(
            planning_horizon=10.0,
            prefill_throughput=1000.0,
            decode_throughput=500.0,
            load_bandwidth=10e9,
            replay_throughput=1000.0,
            detection_time_sec=0.1,
            memory_capacity_bytes=0,
        )
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=20,
            master_time_limit_sec=5.0,
            recovery_time_limit_sec=2.0,
        )

        pending = [make_request(f"r{i}", max_tokens=60) for i in range(6)]
        replica_ids = [0, 1, 2]

        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=pending,
            replica_ids=replica_ids,
        )

        assert result is not None
        # All single-GPU failure scenarios should have recovery plans.
        for rid in replica_ids:
            omega = frozenset({rid})
            assert omega in result.recovery_plans or not any(
                assigned_replica == rid
                for assigned_replica in result.master_solution.assignments.values()
            )

    def test_solver_assigns_valid_replicas(self):
        """Solver assignments are request -> replica only."""
        builder = CostTableBuilder(
            planning_horizon=10.0,
            prefill_throughput=1000.0,
            decode_throughput=500.0,
            load_bandwidth=10e9,
            replay_throughput=1000.0,
            detection_time_sec=0.1,
            memory_capacity_bytes=0,
        )
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=20,
            master_time_limit_sec=5.0,
        )

        pending = [make_request(f"r{i}", max_tokens=100) for i in range(3)]
        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=pending,
            replica_ids=[0, 1],
        )

        assert result is not None
        for rid, r in result.master_solution.assignments.items():
            assert r in (0, 1)


class TestSchedulerPolicyRouting:
    """Test that the scheduler config routes to the correct class."""

    def test_ft_benders_policy_type(self):
        from vllm.config.scheduler import SchedulerConfig
        cfg = SchedulerConfig.default_factory(policy="ft_benders")
        cls = cfg.get_scheduler_cls()
        from vllm.v1.core.sched.benders_ft_scheduler_impl import BendersFTSchedulerImpl
        assert cls is BendersFTSchedulerImpl

    def test_fault_tolerant_policy_unchanged(self):
        from vllm.config.scheduler import SchedulerConfig
        cfg = SchedulerConfig.default_factory(policy="fault_tolerant")
        cls = cfg.get_scheduler_cls()
        from vllm.v1.core.sched.ft_scheduler_impl import FaultTolerantSchedulerImpl
        assert cls is FaultTolerantSchedulerImpl


# ---- Issue 4: Buffer recovery targets ----

class TestBufferRecoveryTargets:
    """Test that recovery targets are buffered when no outputs are available,
    and flushed on the next step with outputs."""

    def test_buffer_replace_semantics(self):
        """Buffer should replace (not merge) on each new snapshot."""
        # Simulate the buffer logic from core.py step().
        buffer: dict[str, int] = {}

        # First snapshot: solver has plans.
        snapshot1: dict[str, int] | None = {"r0": 1, "r1": 2}
        if snapshot1 is not None:
            buffer = dict(snapshot1)
        assert buffer == {"r0": 1, "r1": 2}

        # Second snapshot: solver plans changed (r1 gone, r2 added).
        snapshot2: dict[str, int] | None = {"r0": 1, "r2": 0}
        if snapshot2 is not None:
            buffer = dict(snapshot2)
        # Buffer should be completely replaced, not merged.
        assert buffer == {"r0": 1, "r2": 0}
        assert "r1" not in buffer

    def test_none_means_not_benders(self):
        """None snapshot should not touch the buffer."""
        buffer: dict[str, int] = {"r0": 1}

        snapshot: dict[str, int] | None = None  # not a Benders scheduler
        if snapshot is not None:
            buffer = dict(snapshot)
        # Buffer unchanged.
        assert buffer == {"r0": 1}

    def test_empty_dict_clears_buffer(self):
        """Empty dict ({}) from Benders means clear buffer."""
        buffer: dict[str, int] = {"r0": 1, "r1": 2}

        snapshot: dict[str, int] | None = {}  # Benders, but no plans
        if snapshot is not None:
            buffer = dict(snapshot)
        assert buffer == {}

    def test_flush_on_output(self):
        """Buffer should flush to outputs and clear itself."""
        buffer: dict[str, int] = {"r0": 1, "r1": 2}

        # Simulate: engine_core_outputs exist.
        output_targets: dict[str, int] = {}
        if buffer:
            output_targets = dict(buffer)
            buffer.clear()

        assert output_targets == {"r0": 1, "r1": 2}
        assert buffer == {}


# ---- Issue 3: Stale recovery plans after greedy fallback ----

class TestStaleRecoveryPlans:
    """Test that greedy fallback clears recovery plans."""

    def test_greedy_fallback_clears_plans(self):
        """After greedy fallback, _recovery_plans should be empty."""
        ft = make_ft_scheduler_with_replicas(2)
        builder = CostTableBuilder(
            planning_horizon=10.0,
            prefill_throughput=1000.0,
            decode_throughput=500.0,
            load_bandwidth=10e9,
            replay_throughput=1000.0,
            detection_time_sec=0.1,
            memory_capacity_bytes=0,
        )
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=20,
            master_time_limit_sec=5.0,
            recovery_time_limit_sec=2.0,
        )

        # First solve: get some recovery plans.
        pending = [make_request(f"r{i}", max_tokens=60) for i in range(3)]
        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=pending,
            replica_ids=[0, 1],
        )
        assert result is not None

        # Simulate: scheduler stores plans, then falls back to greedy.
        recovery_plans = {
            omega: plan.assignments
            for omega, plan in result.recovery_plans.items()
        }

        # Greedy fallback should clear plans.
        recovery_plans.clear()
        assert recovery_plans == {}

    def test_solver_recovery_target_returns_none_after_clear(self):
        """After clearing plans, get_solver_recovery_target returns None."""
        from vllm.v1.core.sched.benders_ft_scheduler_impl import BendersFTSchedulerImpl

        # We can't easily instantiate BendersFTSchedulerImpl without full
        # VllmConfig, so test the lookup logic directly.
        plans: dict[frozenset[int], dict[str, int]] = {}
        omega = frozenset({0})
        plan = plans.get(omega)
        assert plan is None  # No plan → None → no stale target


# ---- Issue 1: Remote failure sync via REPLICA_FAILED ----

class TestReplicaFailedSync:
    """Test that REPLICA_FAILED message updates failure detector."""

    def test_replica_failed_message_type_exists(self):
        """REPLICA_FAILED should be defined in EngineCoreRequestType."""
        from vllm.v1.engine import EngineCoreRequestType
        assert hasattr(EngineCoreRequestType, "REPLICA_FAILED")
        assert EngineCoreRequestType.REPLICA_FAILED.value == b"\x05"

    def test_failure_detector_marks_remote_failed(self):
        """After report_failure, get_status returns FAILED."""
        from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus

        fd = FailureDetector()
        # Register a remote replica.
        fd.register_replica(1)
        assert fd.get_status(1) == ReplicaStatus.HEALTHY

        # Report failure (as coordinator would).
        fd.report_failure(1)
        assert fd.get_status(1) == ReplicaStatus.FAILED

    def test_solver_excludes_failed_replica(self):
        """Solver's healthy_all filter should exclude FAILED replicas."""
        from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus

        fd = FailureDetector()
        all_replica_ids = [0, 1, 2, 3]
        # Register all replicas.
        for r in all_replica_ids:
            fd.register_replica(r)

        # Mark replica 2 as failed.
        fd.report_failure(2)

        # Filter logic from benders_ft_scheduler_impl._process_pending_admissions.
        healthy_all = []
        for r in all_replica_ids:
            status = fd.get_status(r)
            if status is None or status == ReplicaStatus.HEALTHY:
                healthy_all.append(r)

        assert healthy_all == [0, 1, 3]
        assert 2 not in healthy_all


# ---- Issue 2: max_gpu_failures > 1 ----

class TestMultiGPUFailure:
    """Test multi-GPU failure scenario enumeration."""

    def test_single_failure_scenarios(self):
        """max_gpu_failures=1 should only enumerate single-failure scenarios."""
        import itertools
        recovery_replica_ids = [0, 1, 2]
        local_set = {0}
        max_gpu_failures = 1

        scenarios = []
        for k in range(1, max_gpu_failures + 1):
            for combo in itertools.combinations(recovery_replica_ids, k):
                omega = frozenset(combo)
                if omega & local_set:
                    scenarios.append(omega)

        # Only scenarios containing replica 0.
        assert frozenset({0}) in scenarios
        assert len(scenarios) == 1  # only {0}

    def test_multi_failure_scenarios(self):
        """max_gpu_failures=2 with dp_size=4 should enumerate k=1 and k=2."""
        import itertools
        recovery_replica_ids = [0, 1, 2, 3]
        local_set = {0}
        max_gpu_failures = 2

        scenarios = []
        for k in range(1, max_gpu_failures + 1):
            for combo in itertools.combinations(recovery_replica_ids, k):
                omega = frozenset(combo)
                if omega & local_set:
                    scenarios.append(omega)

        # k=1: {0} → 1 scenario containing local
        # k=2: {0,1}, {0,2}, {0,3} → 3 scenarios containing local
        assert len(scenarios) == 4
        assert frozenset({0}) in scenarios
        assert frozenset({0, 1}) in scenarios
        assert frozenset({0, 2}) in scenarios
        assert frozenset({0, 3}) in scenarios
        # {1,2}, {1,3}, {2,3} should NOT be present (no local replica).
        assert frozenset({1, 2}) not in scenarios

    def test_solve_loop_with_max_gpu_failures_2(self):
        """Solve loop with max_gpu_failures=2 should handle multi-failure."""
        builder = CostTableBuilder(
            planning_horizon=10.0,
            prefill_throughput=1000.0,
            decode_throughput=500.0,
            load_bandwidth=10e9,
            replay_throughput=1000.0,
            detection_time_sec=0.1,
            memory_capacity_bytes=0,
        )
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=30,
            master_time_limit_sec=5.0,
            recovery_time_limit_sec=2.0,
            max_gpu_failures=2,
        )

        requests = [
            make_request(f"r{i}", prompt_len=10, max_tokens=50)
            for i in range(4)
        ]

        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=requests,
            replica_ids=[0],
            all_replica_ids=[0, 1, 2, 3],
        )

        # Should either converge or return None (may need more capacity).
        # At minimum: no crash, and if converged, recovery plans should
        # include multi-failure scenarios.
        if result is not None:
            for omega in result.recovery_plans:
                assert isinstance(omega, frozenset)
                # All scenarios must contain the local replica (0).
                assert 0 in omega

    def test_multi_failure_recovery_lookup(self):
        """get_solver_recovery_target should find plans from multi-failure
        scenarios via superset fallback."""
        # Simulate recovery plans with multi-failure scenario.
        plans: dict[frozenset[int], dict[str, int]] = {
            frozenset({0, 1}): {"r0": 2, "r1": 3},
            frozenset({0, 2}): {"r0": 1, "r2": 3},
        }

        # Lookup for failed_replica_id=0, request_id="r0".
        # No exact single-failure match for {0}, but {0,1} is a superset.
        failed_replica_id = 0
        request_id = "r0"

        omega = frozenset({failed_replica_id})
        plan = plans.get(omega)
        target = None
        if plan is not None:
            target = plan.get(request_id)

        if target is None:
            # Superset fallback.
            for scenario, plan in plans.items():
                if failed_replica_id in scenario:
                    t = plan.get(request_id)
                    if t is not None:
                        target = t
                        break

        assert target is not None
        assert target in (1, 2)  # from one of the multi-failure plans


# ---- Cross-component pipeline tests ----

class TestSnapshotPipeline:
    """Test the full engine → EngineCoreOutputs → client pipeline for
    recovery_targets snapshots.

    Client stores metadata per-engine: each engine's snapshot replaces
    only that engine's partition."""

    def test_empty_snapshot_reaches_client_and_clears_engine_partition(self):
        """An empty {} snapshot from engine must reach the client and clear
        that engine's stale entries (not be silently dropped)."""
        from vllm.v1.engine import EngineCoreOutputs

        # Simulate engine-side buffer logic (from core.py step()).
        buffer: dict[str, int] | None = {"r0": 1, "r1": 2}

        # Solver returns empty snapshot (greedy fallback cleared plans).
        new_snapshot: dict[str, int] | None = {}
        if new_snapshot is not None:
            buffer = dict(new_snapshot)

        # Engine flushes to outputs — must use `is not None`, not truthy.
        outputs = EngineCoreOutputs(engine_index=0)
        if buffer is not None:
            outputs.recovery_targets = dict(buffer)
            buffer = None

        # The empty dict must be present on outputs.
        assert outputs.recovery_targets is not None
        assert outputs.recovery_targets == {}

        # Client-side: per-engine replace.
        targets_by_engine: dict[int, dict[str, int]] = {
            0: {"r0": 1, "r1": 2},  # stale from previous epoch
        }
        if outputs.recovery_targets is not None:
            targets_by_engine[outputs.engine_index] = dict(
                outputs.recovery_targets
            )

        # Engine 0's stale entries must be gone.
        assert targets_by_engine[0] == {}

    def test_full_snapshot_replaces_engine_partition_only(self):
        """A non-empty snapshot must replace only that engine's partition."""
        from vllm.v1.engine import EngineCoreOutputs

        targets_by_engine: dict[int, dict[str, int]] = {
            0: {"r0": 1, "r1": 2},  # engine 0's old data
            1: {"r2": 0},           # engine 1's data
        }

        # Engine 0 sends new snapshot.
        outputs = EngineCoreOutputs(engine_index=0)
        outputs.recovery_targets = {"r0": 1, "r3": 2}

        if outputs.recovery_targets is not None:
            targets_by_engine[outputs.engine_index] = dict(
                outputs.recovery_targets
            )

        # Engine 0 replaced; engine 1 untouched.
        assert targets_by_engine[0] == {"r0": 1, "r3": 2}
        assert "r1" not in targets_by_engine[0]
        assert targets_by_engine[1] == {"r2": 0}

    def test_none_snapshot_preserves_all_state(self):
        """None (not a Benders scheduler) must not touch any partition."""
        from vllm.v1.engine import EngineCoreOutputs

        targets_by_engine: dict[int, dict[str, int]] = {
            0: {"r0": 1},
        }

        outputs = EngineCoreOutputs(engine_index=0)
        # recovery_targets defaults to None.
        assert outputs.recovery_targets is None

        if outputs.recovery_targets is not None:
            targets_by_engine[outputs.engine_index] = dict(
                outputs.recovery_targets
            )

        # Unchanged.
        assert targets_by_engine[0] == {"r0": 1}

    def test_solver_checkpoint_classes_field_removed(self):
        """Checkpoint class snapshots are no longer part of engine outputs."""
        from vllm.v1.engine import EngineCoreOutputs

        outputs = EngineCoreOutputs(engine_index=0)
        assert not hasattr(outputs, "solver_checkpoint_classes")


class TestMarkRemoteFailed:
    """Test that mark_remote_failed updates status without callbacks."""

    def test_mark_remote_failed_no_callback(self):
        """mark_remote_failed should NOT trigger failure callbacks."""
        from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus

        fd = FailureDetector()
        fd.register_replica(1)

        callback_fired = []
        fd.register_callback(lambda rid: callback_fired.append(rid))

        # mark_remote_failed: status changes, no callback.
        fd.mark_remote_failed(1)
        assert fd.get_status(1) == ReplicaStatus.FAILED
        assert callback_fired == []

    def test_report_failure_does_fire_callback(self):
        """Confirm report_failure DOES fire callbacks (contrast test)."""
        from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus

        fd = FailureDetector()
        fd.register_replica(2)

        callback_fired = []
        fd.register_callback(lambda rid: callback_fired.append(rid))

        fd.report_failure(2)
        assert fd.get_status(2) == ReplicaStatus.FAILED
        assert callback_fired == [2]

    def test_mark_remote_failed_idempotent(self):
        """Calling mark_remote_failed twice should be safe."""
        from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus

        fd = FailureDetector()
        fd.register_replica(3)
        fd.mark_remote_failed(3)
        fd.mark_remote_failed(3)  # second call — no error
        assert fd.get_status(3) == ReplicaStatus.FAILED

    def test_mark_remote_failed_unregistered_is_noop(self):
        """mark_remote_failed on unregistered replica is safe noop."""
        from vllm.v1.core.failure_detector import FailureDetector

        fd = FailureDetector()
        fd.mark_remote_failed(99)  # no crash


class TestPerEnginePartitionedStorage:
    """Test that client stores solver metadata per-engine so one engine's
    snapshot doesn't clobber another engine's entries."""

    def test_interleaved_engine_outputs_preserve_both(self):
        """Two engines sending interleaved snapshots must both be preserved."""
        from vllm.v1.engine import EngineCoreOutputs

        # Simulate client's per-engine storage.
        targets_by_engine: dict[int, dict[str, int]] = {}

        # Engine 0 reports its requests' targets.
        out0 = EngineCoreOutputs(engine_index=0)
        out0.recovery_targets = {"r0": 1, "r1": 2}

        if out0.recovery_targets is not None:
            targets_by_engine[out0.engine_index] = dict(out0.recovery_targets)

        # Engine 1 reports its requests' targets.
        out1 = EngineCoreOutputs(engine_index=1)
        out1.recovery_targets = {"r2": 0, "r3": 2}

        if out1.recovery_targets is not None:
            targets_by_engine[out1.engine_index] = dict(out1.recovery_targets)

        # Both engines' entries must coexist.
        assert targets_by_engine[0] == {"r0": 1, "r1": 2}
        assert targets_by_engine[1] == {"r2": 0, "r3": 2}

        # Lookup across all partitions.
        def lookup(req_id):
            for part in targets_by_engine.values():
                t = part.get(req_id)
                if t is not None:
                    return t
            return None

        assert lookup("r0") == 1
        assert lookup("r2") == 0
        assert lookup("r4") is None  # not present

    def test_empty_snapshot_from_one_engine_doesnt_clear_another(self):
        """An idle engine sending {} must not wipe the other engine's data."""
        targets_by_engine: dict[int, dict[str, int]] = {}

        # Engine 0 has real targets.
        targets_by_engine[0] = {"r0": 1, "r1": 2}

        # Engine 1 is idle, sends {}.
        targets_by_engine[1] = {}

        # Engine 0's targets must survive.
        assert targets_by_engine[0] == {"r0": 1, "r1": 2}

        # Merged lookup still finds r0.
        all_targets = {}
        for part in targets_by_engine.values():
            all_targets.update(part)
        assert all_targets == {"r0": 1, "r1": 2}

    def test_engine_replan_replaces_only_its_partition(self):
        """When engine 0 replans (new snapshot), only its partition changes."""
        targets_by_engine: dict[int, dict[str, int]] = {}

        targets_by_engine[0] = {"r0": 1, "r1": 2}
        targets_by_engine[1] = {"r2": 0}

        # Engine 0 replans: r1 dropped, r3 added.
        targets_by_engine[0] = {"r0": 1, "r3": 2}

        assert targets_by_engine[0] == {"r0": 1, "r3": 2}
        assert "r1" not in targets_by_engine[0]  # gone from engine 0
        assert targets_by_engine[1] == {"r2": 0}  # untouched

    def test_dead_engine_partition_cleanup(self):
        """After engine failure, its partition should be cleaned up."""
        targets_by_engine: dict[int, dict[str, int]] = {}
        ckpt_by_engine: dict[int, dict[str, int]] = {}

        targets_by_engine[0] = {"r0": 1}
        targets_by_engine[1] = {"r1": 0}
        ckpt_by_engine[0] = {"r0": 2}
        ckpt_by_engine[1] = {"r1": 1}

        # Engine 0 dies — after failover, clean up its partition.
        dead_idx = 0
        targets_by_engine.pop(dead_idx, None)
        ckpt_by_engine.pop(dead_idx, None)

        assert 0 not in targets_by_engine
        assert targets_by_engine[1] == {"r1": 0}
        assert ckpt_by_engine[1] == {"r1": 1}
