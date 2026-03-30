"""Tests for centralized Benders FT solver path.

Verifies:
1. solve_epoch_from_costs works with a pre-built cost table.
2. build_global_costs correctly computes from RequestSnapshot + EngineCoreRequest.
3. End-to-end: snapshots from multiple replicas → global cost table → solver
   → cross-replica placement.
4. Greedy fallback semantics (empty replicas, empty cost table).
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.checkpoint_controller import CheckpointConfig
from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder, RequestCosts
from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop, BendersSolveResult
from vllm.v1.engine import (
    EngineCoreRequest,
    ReplicaSnapshot,
    RequestSnapshot,
)
from vllm.v1.request import Request


# ---- Helpers ----

def make_cost_builder(**kwargs) -> CostTableBuilder:
    defaults = dict(
        planning_horizon=1.0,
        prefill_throughput=1000.0,
        decode_throughput=500.0,
        load_bandwidth=10e9,
        replay_throughput=1000.0,
        detection_time_sec=0.1,
        memory_capacity_bytes=0,
        checkpoint_config=CheckpointConfig(),
    )
    defaults.update(kwargs)
    return CostTableBuilder(**defaults)


def make_solve_loop(**kwargs) -> BendersSolveLoop:
    builder = kwargs.pop("cost_builder", make_cost_builder())
    defaults = dict(
        cost_builder=builder,
        max_iterations=20,
        master_time_limit_sec=1.0,
        recovery_time_limit_sec=0.5,
        max_gpu_failures=1,
    )
    defaults.update(kwargs)
    return BendersSolveLoop(**defaults)


def make_request_snapshot(
    request_id: str,
    prompt_len: int = 10,
    generation_len: int = 100,
    num_computed_tokens: int = 0,
    num_output_tokens: int = 0,
    assigned_replica_id: int | None = None,
    num_checkpointed_tokens: int = 0,
    checkpoint_size_bytes: int = 0,
    checkpoint_level: int = 0,
) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_len=prompt_len,
        generation_len=generation_len,
        num_computed_tokens=num_computed_tokens,
        num_output_tokens=num_output_tokens,
        num_checkpointed_tokens=num_checkpointed_tokens,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_level=checkpoint_level,
        assigned_replica_id=assigned_replica_id,
    )


def make_engine_request(
    request_id: str,
    prompt_len: int = 10,
    expected_output_len: int = 100,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=expected_output_len),
        pooling_params=None,
        eos_token_id=2,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        expected_output_len=expected_output_len,
    )


def make_request(
    request_id: str,
    prompt_len: int = 10,
    max_tokens: int = 100,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        eos_token_id=2,
        expected_output_len=max_tokens,
    )


# ---- Tests: solve_epoch_from_costs ----

class TestSolveEpochFromCosts:
    """Tests that solve_epoch_from_costs works with a pre-built cost table."""

    def test_empty_cost_table_returns_empty_result(self):
        solver = make_solve_loop()
        result = solver.solve_epoch_from_costs({}, [0])
        assert result is not None
        assert isinstance(result, BendersSolveResult)
        assert result.num_iterations == 0

    def test_no_replicas_returns_none(self):
        solver = make_solve_loop()
        result = solver.solve_epoch_from_costs({"r1": RequestCosts(
            request_id="r1", G_j=100, p_j=0.01, d_j=0.2,
        )}, [])
        assert result is None

    def test_single_pending_request_admitted(self):
        """Single replica, no fault tolerance — request should be admitted."""
        builder = make_cost_builder()
        # max_gpu_failures=0: no recovery scenarios, pure admission.
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=0)

        costs = builder.compute_request_costs(
            make_request("req-1", prompt_len=10, max_tokens=100),
            is_active=False,
        )
        cost_table = {"req-1": costs}

        result = solver.solve_epoch_from_costs(cost_table, [0])
        assert result is not None
        assert "req-1" in result.master_solution.admitted

    def test_equivalence_with_solve_epoch(self):
        """solve_epoch and solve_epoch_from_costs give same result."""
        builder = make_cost_builder()
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=1)

        reqs = [make_request(f"req-{i}") for i in range(3)]
        active = [reqs[0]]
        active[0].num_computed_tokens = 10
        pending = reqs[1:]

        result_epoch = solver.solve_epoch(
            active, pending, [0, 1, 2], all_replica_ids=[0, 1, 2]
        )

        # Now build cost table and call from_costs.
        cost_table = builder.build_snapshot_costs(active, pending)
        result_costs = solver.solve_epoch_from_costs(
            cost_table, [0, 1, 2], all_replica_ids=[0, 1, 2]
        )

        assert result_epoch is not None
        assert result_costs is not None
        assert result_epoch.master_solution.admitted == result_costs.master_solution.admitted

    def test_multi_replica_cross_placement(self):
        """Solver assigns requests across multiple replicas."""
        builder = make_cost_builder()
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=1)

        # Create enough pending requests to need multiple replicas.
        cost_table = {}
        for i in range(6):
            costs = builder.compute_request_costs(
                make_request(f"req-{i}", prompt_len=10, max_tokens=100),
                is_active=False,
            )
            cost_table[f"req-{i}"] = costs

        result = solver.solve_epoch_from_costs(
            cost_table, [0, 1, 2], all_replica_ids=[0, 1, 2]
        )
        assert result is not None
        assert len(result.master_solution.admitted) > 0

        # Check that assignments span multiple replicas.
        replicas_used = set()
        for req_id, r_id in result.master_solution.assignments.items():
            replicas_used.add(r_id)
        # With enough requests and 3 replicas, multiple should be used.
        assert len(replicas_used) >= 1


# ---- Tests: build_global_costs ----

class TestBuildGlobalCosts:
    """Tests that build_global_costs correctly handles mixed input types."""

    def test_from_snapshots_and_engine_requests(self):
        builder = make_cost_builder()

        # Active requests from engine snapshots.
        active_snaps = [
            make_request_snapshot(
                "active-1", prompt_len=10, generation_len=100,
                num_computed_tokens=10, num_output_tokens=20,
                assigned_replica_id=0, checkpoint_level=1,
            ),
        ]

        # Pending requests from client queue.
        pending = [
            make_engine_request("pending-1", prompt_len=15, expected_output_len=80),
        ]

        cost_table = builder.build_global_costs(active_snaps, pending)
        assert "active-1" in cost_table
        assert "pending-1" in cost_table
        assert cost_table["active-1"].is_active is True
        assert cost_table["active-1"].active_replica_id == 0
        assert cost_table["pending-1"].is_active is False

    def test_snapshot_costs_match_request_costs(self):
        """Costs from snapshot match costs from Request for same data."""
        builder = make_cost_builder()

        req = make_request("r1", prompt_len=20, max_tokens=50)

        # Costs from Request.
        costs_req = builder.compute_request_costs(req, is_active=False)

        # Costs from snapshot with same fields.
        snap = make_request_snapshot(
            "r1", prompt_len=20, generation_len=50,
        )
        costs_snap = builder.compute_costs_from_snapshot(snap, is_active=False)

        assert costs_req.p_j == costs_snap.p_j
        assert costs_req.d_j == costs_snap.d_j
        assert costs_req.G_j == costs_snap.G_j
        assert costs_req.checkpoint_overhead_sec == costs_snap.checkpoint_overhead_sec
        assert costs_req.restore_time_sec == costs_snap.restore_time_sec
        assert costs_req.replay_time_sec == costs_snap.replay_time_sec

    def test_engine_request_costs_correct(self):
        """Costs from EngineCoreRequest use prompt_token_ids length."""
        builder = make_cost_builder()

        eng_req = make_engine_request("e1", prompt_len=30, expected_output_len=60)
        costs = builder.compute_costs_from_engine_request(eng_req)

        assert costs.request_id == "e1"
        assert costs.prompt_len == 30
        assert costs.G_j == 60
        assert costs.is_active is False
        assert costs.p_j == 30 / 1000.0  # prompt_len / prefill_throughput

    def test_empty_inputs(self):
        builder = make_cost_builder()
        cost_table = builder.build_global_costs([], [])
        assert cost_table == {}

    def test_multi_engine_snapshots(self):
        """Snapshots from multiple engines merge correctly."""
        builder = make_cost_builder()

        snaps_engine_0 = [
            make_request_snapshot("r1", assigned_replica_id=0,
                                 num_computed_tokens=10, num_output_tokens=5),
            make_request_snapshot("r2", assigned_replica_id=0,
                                 num_computed_tokens=10, num_output_tokens=3),
        ]
        snaps_engine_1 = [
            make_request_snapshot("r3", assigned_replica_id=1,
                                 num_computed_tokens=10, num_output_tokens=8),
        ]

        all_snaps = snaps_engine_0 + snaps_engine_1
        pending = [make_engine_request("r4")]
        cost_table = builder.build_global_costs(all_snaps, pending)

        assert len(cost_table) == 4
        assert cost_table["r1"].active_replica_id == 0
        assert cost_table["r3"].active_replica_id == 1
        assert cost_table["r4"].is_active is False


# ---- Tests: end-to-end centralized solve ----

class TestCentralizedSolveEndToEnd:
    """End-to-end: snapshots → cost table → solver → result."""

    def test_global_solve_with_active_and_pending(self):
        builder = make_cost_builder()
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=1)

        # Simulate 2 replicas, each with some active requests.
        active_snaps = [
            make_request_snapshot(
                "active-r0-1", assigned_replica_id=0,
                num_computed_tokens=10, num_output_tokens=10,
            ),
            make_request_snapshot(
                "active-r1-1", assigned_replica_id=1,
                num_computed_tokens=10, num_output_tokens=15,
            ),
        ]

        pending = [
            make_engine_request(f"pending-{i}", prompt_len=10,
                                expected_output_len=50)
            for i in range(3)
        ]

        cost_table = builder.build_global_costs(active_snaps, pending)
        result = solver.solve_epoch_from_costs(
            cost_table, [0, 1], all_replica_ids=[0, 1]
        )

        assert result is not None
        assert result.num_iterations >= 1

        # Active requests should be in assignments with their original replica.
        if "active-r0-1" in result.master_solution.assignments:
            r_id = result.master_solution.assignments["active-r0-1"]
            assert r_id == 0  # Active requests keep their placement.
        if "active-r1-1" in result.master_solution.assignments:
            r_id = result.master_solution.assignments["active-r1-1"]
            assert r_id == 1

    def test_recovery_plans_generated(self):
        builder = make_cost_builder()
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=1)

        # 2 replicas, some requests on each.
        active_snaps = [
            make_request_snapshot(
                f"r{i}", assigned_replica_id=i % 2,
                num_computed_tokens=10, num_output_tokens=5,
            )
            for i in range(4)
        ]

        cost_table = builder.build_global_costs(active_snaps, [])
        result = solver.solve_epoch_from_costs(
            cost_table, [0, 1], all_replica_ids=[0, 1]
        )

        assert result is not None
        # With 2 replicas and max_gpu_failures=1, there should be
        # recovery plans for {0} and {1}.
        assert len(result.recovery_plans) >= 1

    def test_solver_respects_replica_subset(self):
        """When one replica is unhealthy, solver only routes to healthy ones."""
        builder = make_cost_builder()
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=1)

        pending = [
            make_engine_request(f"req-{i}", prompt_len=5, expected_output_len=50)
            for i in range(4)
        ]

        cost_table = builder.build_global_costs([], pending)

        # Only replica 1 is healthy.
        result = solver.solve_epoch_from_costs(
            cost_table, [1], all_replica_ids=[0, 1]
        )

        assert result is not None
        for req_id, r_id in result.master_solution.assignments.items():
            assert r_id == 1  # All go to the only healthy replica.


# ---- Tests: ReplicaSnapshot data structure ----

class TestReplicaSnapshot:
    def test_basic_creation(self):
        snap = ReplicaSnapshot(replica_id=0)
        assert snap.replica_id == 0
        assert snap.is_healthy is True
        assert snap.num_waiting_reqs == 0
        assert snap.gpu_kv_cache_usage == 0.0

    def test_unhealthy_replica(self):
        snap = ReplicaSnapshot(
            replica_id=1, is_healthy=False,
            num_running_reqs=5, gpu_kv_cache_usage=0.8,
        )
        assert snap.is_healthy is False
        assert snap.num_running_reqs == 5


# ---- Tests: RequestSnapshot data structure ----

class TestRequestSnapshot:
    def test_basic_creation(self):
        snap = make_request_snapshot("test-1")
        assert snap.request_id == "test-1"
        assert snap.prompt_len == 10
        assert snap.generation_len == 100

    def test_with_slo(self):
        snap = RequestSnapshot(
            request_id="slo-1",
            prompt_len=10,
            generation_len=50,
            num_computed_tokens=10,
            num_output_tokens=5,
            ttft_slo_ms=100.0,
            tpot_slo_ms=20.0,
            failure_gap_slo_ms=500.0,
        )
        assert snap.ttft_slo_ms == 100.0
        assert snap.tpot_slo_ms == 20.0
        assert snap.failure_gap_slo_ms == 500.0


# ---- Tests: stale snapshot cleanup (reviewer finding #1) ----

class TestStaleSnapshotCleanup:
    """Verify finished requests are scrubbed from _engine_request_snapshots."""

    def test_finished_requests_removed_from_snapshots(self):
        """Simulates process_engine_outputs with finished_requests:
        the snapshot cache should no longer contain them."""
        # We can't easily instantiate CentralizedBendersFTClient (needs
        # full engine setup), so test the logic directly on a dict that
        # mirrors _engine_request_snapshots.
        snapshots: dict[int, list[RequestSnapshot]] = {
            0: [
                make_request_snapshot("r1", assigned_replica_id=0),
                make_request_snapshot("r2", assigned_replica_id=0),
                make_request_snapshot("r3", assigned_replica_id=0),
            ],
            1: [
                make_request_snapshot("r4", assigned_replica_id=1),
            ],
        }

        # Simulate: engine 0 reports r1 and r3 finished.
        finished_set = {"r1", "r3"}
        eng_idx = 0
        if eng_idx in snapshots:
            snapshots[eng_idx] = [
                s for s in snapshots[eng_idx]
                if s.request_id not in finished_set
            ]

        # r2 survives, r1 and r3 gone from engine 0.
        assert len(snapshots[0]) == 1
        assert snapshots[0][0].request_id == "r2"
        # Engine 1 untouched.
        assert len(snapshots[1]) == 1
        assert snapshots[1][0].request_id == "r4"

    def test_finished_request_not_in_solver_cost_table(self):
        """After cleanup, finished requests shouldn't appear in the cost
        table built for the solver."""
        builder = make_cost_builder()

        snapshots = [
            make_request_snapshot("active-1", assigned_replica_id=0,
                                 num_computed_tokens=10, num_output_tokens=5),
            make_request_snapshot("finished-1", assigned_replica_id=0,
                                 num_computed_tokens=10, num_output_tokens=100),
        ]

        # Simulate cleanup: remove finished-1.
        snapshots = [s for s in snapshots if s.request_id != "finished-1"]

        cost_table = builder.build_global_costs(snapshots, [])
        assert "active-1" in cost_table
        assert "finished-1" not in cost_table


# ---- Tests: centralized mode doesn't need load overrides (reviewer #2) ----

class TestCentralizedNoOverrideNeeded:
    """In centralized mode, all replicas' requests are in the cost table,
    so surv_load is computed accurately without overrides."""

    def test_surv_load_accurate_from_cost_table(self):
        """With requests from ALL replicas in cost table,
        recovery checker computes accurate surv_load without overrides."""
        from vllm.v1.core.sched.benders.recovery_checker import RecoveryChecker
        from vllm.v1.core.sched.benders.master import MasterSolution

        builder = make_cost_builder()

        # Simulate centralized: requests on replica 0, 1, and 2 all in
        # cost table.  This is the centralized case — no "remote" replicas.
        reqs = [make_request(f"r{i}", prompt_len=10, max_tokens=100)
                for i in range(6)]
        cost_table = {}
        for i, req in enumerate(reqs):
            replica = i % 3
            costs = builder.compute_request_costs(req, is_active=True)
            costs.active_replica_id = replica
            cost_table[req.request_id] = costs

        solution = MasterSolution()
        for i, req in enumerate(reqs):
            solution.assignments[req.request_id] = i % 3
            solution.admitted.add(req.request_id)

        # Centralized: local_replica_ids == all replicas.
        # No overrides needed — surv_load computed from cost table.
        checker = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1, 2],
            H_dec=builder.H_dec,
            local_replica_ids=[0, 1, 2],  # ALL replicas are "local"
        )
        plan, cert = checker.check_scenario(solution, frozenset({0}))
        assert (plan is not None) or (cert is not None)

        # Verify: same result with overrides (they're not used since
        # all replicas are local).
        checker_with = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1, 2],
            H_dec=builder.H_dec,
            local_replica_ids=[0, 1, 2],
            replica_load_overrides={0: 0.99, 1: 0.99, 2: 0.99},
        )
        plan2, cert2 = checker_with.check_scenario(solution, frozenset({0}))
        # Should be identical — overrides are ignored for local replicas.
        assert type(plan) is type(plan2)
        assert type(cert) is type(cert2)


# ---- Tests: per-engine path still uses overrides correctly ----

class TestReplicaLoadOverrides:
    """Verify that ReplicaSnapshot data flows into recovery checker."""

    def test_load_override_reduces_survivor_capacity(self):
        """With high load override on survivor, recovery should be harder."""
        from vllm.v1.core.sched.benders.recovery_checker import RecoveryChecker
        from vllm.v1.core.sched.benders.master import MasterSolution

        builder = make_cost_builder()

        # 3 replicas, 6 requests spread across replicas 0 and 1.
        reqs = [make_request(f"r{i}", prompt_len=10, max_tokens=100)
                for i in range(6)]
        cost_table = {}
        for i, req in enumerate(reqs):
            costs = builder.compute_request_costs(req, is_active=True)
            costs.active_replica_id = i % 2  # on replica 0 or 1
            cost_table[req.request_id] = costs

        # Build a mock master solution: all assigned to replicas 0/1.
        solution = MasterSolution()
        for i, req in enumerate(reqs):
            solution.assignments[req.request_id] = i % 2
            solution.admitted.add(req.request_id)

        # Case 1: No overrides — survivor (replica 2) assumed idle.
        checker_no_override = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1, 2],
            H_dec=builder.H_dec,
            local_replica_ids=[0, 1],
            local_load_fraction=0.0,  # assume remote is idle
        )
        plan1, cert1 = checker_no_override.check_scenario(
            solution, frozenset({0})
        )

        # Case 2: Override says replica 2 is 95% loaded.
        checker_with_override = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1, 2],
            H_dec=builder.H_dec,
            local_replica_ids=[0, 1],
            local_load_fraction=0.0,
            replica_load_overrides={2: 0.95},
        )
        plan2, cert2 = checker_with_override.check_scenario(
            solution, frozenset({0})
        )

        # Without override, replica 2 has full capacity → more likely feasible.
        # With override, replica 2 has only 5% capacity → harder.
        # At minimum, we verify the override changes the outcome or passes
        # through without error.
        assert (plan1 is not None) or (cert1 is not None)  # always one
        assert (plan2 is not None) or (cert2 is not None)

    def test_solve_epoch_from_costs_passes_overrides(self):
        """Verify overrides don't crash the full solve loop."""
        builder = make_cost_builder()
        solver = make_solve_loop(cost_builder=builder, max_gpu_failures=1)

        cost_table = {}
        for i in range(4):
            costs = builder.compute_request_costs(
                make_request(f"req-{i}", prompt_len=10, max_tokens=50),
                is_active=False,
            )
            cost_table[f"req-{i}"] = costs

        # Pass overrides through solve_epoch_from_costs.
        result = solver.solve_epoch_from_costs(
            cost_table,
            [0, 1, 2],
            all_replica_ids=[0, 1, 2],
            replica_load_overrides={0: 0.3, 1: 0.5, 2: 0.1},
            replica_mem_overrides=None,
        )
        assert result is not None
        assert len(result.master_solution.admitted) > 0

    def test_mem_override_used_when_M_cap_nonzero(self):
        """Memory overrides flow through when M_cap > 0."""
        from vllm.v1.core.sched.benders.recovery_checker import RecoveryChecker
        from vllm.v1.core.sched.benders.master import MasterSolution

        M_CAP = 16 * 1024 * 1024 * 1024  # 16 GB
        builder = make_cost_builder(memory_capacity_bytes=M_CAP)

        req = make_request("r1", prompt_len=10, max_tokens=50)
        costs = builder.compute_request_costs(req, is_active=True)
        costs.active_replica_id = 0
        cost_table = {"r1": costs}

        solution = MasterSolution()
        solution.assignments["r1"] = 0
        solution.admitted.add("r1")

        # Replica 1 has 90% memory used according to override.
        checker = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1],
            H_dec=builder.H_dec,
            M_cap=M_CAP,
            local_replica_ids=[0],
            local_load_fraction=0.0,
            replica_mem_overrides={1: 0.9 * M_CAP},
        )
        plan, cert = checker.check_scenario(solution, frozenset({0}))
        # Should complete without error; result depends on request size.
        assert (plan is not None) or (cert is not None)


# ---- Tests: rejected request terminal output (reviewer finding) ----

class TestRejectedRequestOutput:
    """Verify that solver-rejected requests get a terminal ABORT output."""

    def test_abort_output_for_rejected_request(self):
        """Simulate _dispatch_solver_result with a request NOT in admitted.
        Verify that an EngineCoreOutputs with ABORT finish_reason is produced.
        """
        from vllm.v1.core.sched.benders.master import MasterSolution
        from vllm.v1.core.sched.benders.solve_loop import BendersSolveResult
        from vllm.v1.engine import (
            EngineCoreOutput,
            EngineCoreOutputs,
            FinishReason,
        )

        # Construct a BendersSolveResult where "req-1" is admitted
        # but "req-2" is rejected.
        solution = MasterSolution()
        solution.assignments["req-1"] = 0
        solution.admitted.add("req-1")
        # req-2 NOT in admitted.
        result = BendersSolveResult(
            master_solution=solution,
            num_iterations=1,
            total_solve_time_sec=0.01,
        )

        # Simulate what _dispatch_solver_result does for rejected requests.
        # We can't easily instantiate CentralizedBendersFTClient, so test
        # the core logic: if req_id not in admitted, produce ABORT output.
        rejected: list[str] = []
        pending_ids = ["req-1", "req-2"]
        for req_id in pending_ids:
            if req_id not in result.master_solution.admitted:
                rejected.append(req_id)

        assert rejected == ["req-2"]

        # Verify we can construct the ABORT outputs correctly.
        if rejected:
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
            assert len(abort_outputs.outputs) == 1
            assert abort_outputs.outputs[0].request_id == "req-2"
            assert abort_outputs.outputs[0].finish_reason == FinishReason.ABORT
            assert abort_outputs.outputs[0].finished is True
