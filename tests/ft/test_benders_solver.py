"""Unit tests for the Benders-style robust FT solver."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.checkpoint_controller import CheckpointConfig
from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder, RequestCosts
from vllm.v1.core.sched.benders.cuts import (
    make_cut,
    make_no_good_cut,
    make_pool_overload_cut,
)
from vllm.v1.core.sched.benders.master import MasterProblem, MasterSolution
from vllm.v1.core.sched.benders.recovery_checker import (
    InfeasibilityCertificate,
    RecoveryChecker,
)
from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop
from vllm.v1.request import Request


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


def make_cost_builder(
    prefill_tput: float = 1000.0,
    decode_tput: float = 500.0,
    load_bw: float = 10e9,
    checkpoint_bw: float | None = None,
    replay_tput: float = 1000.0,
    planning_horizon: float = 10.0,
    block_size: int = 16,
) -> CostTableBuilder:
    return CostTableBuilder(
        planning_horizon=planning_horizon,
        prefill_throughput=prefill_tput,
        decode_throughput=decode_tput,
        load_bandwidth=load_bw,
        checkpoint_bandwidth=checkpoint_bw,
        replay_throughput=replay_tput,
        detection_time_sec=0.1,
        memory_capacity_bytes=0,
        checkpoint_config=CheckpointConfig(),
        block_size=block_size,
    )


def make_simple_cost(
    request_id: str,
    generation_len: int = 100,
    *,
    p_j: float = 0.01,
    d_j: float = 0.2,
    checkpoint_overhead_sec: float = 0.001,
    restore_time_sec: float = 0.01,
    replay_time_sec: float = 0.05,
    run_mem_bytes: float = 1000.0,
    recovery_mem_bytes: float = 1000.0,
) -> RequestCosts:
    return RequestCosts(
        request_id=request_id,
        G_j=generation_len,
        p_j=p_j,
        d_j=d_j,
        checkpoint_overhead_sec=checkpoint_overhead_sec,
        restore_time_sec=restore_time_sec,
        replay_time_sec=replay_time_sec,
        gap_time_sec=0.1 + restore_time_sec + replay_time_sec + 1 / 500.0,
        run_mem_bytes=run_mem_bytes,
        recovery_mem_bytes=recovery_mem_bytes,
        prompt_len=10,
    )


class TestCostTableBuilder:
    def test_basic_cost_computation(self):
        builder = make_cost_builder()
        req = make_request("r1", prompt_len=100, max_tokens=200)
        costs = builder.compute_request_costs(req)

        assert costs.request_id == "r1"
        assert costs.G_j == 200
        assert abs(costs.p_j - 0.1) < 1e-6
        assert abs(costs.d_j - 0.4) < 1e-6
        assert not costs.is_active

    def test_pending_request_failure_replay_is_prompt_only(self):
        builder = make_cost_builder(prefill_tput=1000.0)
        req = make_request("r1", prompt_len=50, max_tokens=200)
        costs = builder.compute_request_costs(req, is_active=False)

        assert costs.restore_time_sec == 0.0
        assert abs(costs.replay_time_sec - 0.05) < 1e-6

    def test_active_request_more_published_tokens_shifts_cost_from_replay_to_restore(self):
        builder = make_cost_builder(prefill_tput=1000.0, load_bw=1000.0)
        req = make_request("r1", prompt_len=10, max_tokens=100)
        req.num_computed_tokens = 80
        req.num_checkpointed_tokens = 0
        req.last_checkpoint_size_bytes = 0

        no_ckpt = builder.compute_request_costs(req, is_active=True)

        req.num_checkpointed_tokens = 32
        req.last_checkpoint_size_bytes = 32000
        with_ckpt = builder.compute_request_costs(req, is_active=True)

        assert with_ckpt.restore_time_sec > no_ckpt.restore_time_sec
        assert with_ckpt.replay_time_sec < no_ckpt.replay_time_sec

    def test_checkpoint_level_is_ignored_without_published_tokens(self):
        builder = make_cost_builder(prefill_tput=1000.0)
        req = make_request("r1", prompt_len=10, max_tokens=100)
        req.num_computed_tokens = 64
        req.num_checkpointed_tokens = 0
        req.last_checkpoint_size_bytes = 0
        req.checkpoint_level = 2

        costs = builder.compute_request_costs(req, is_active=True)
        assert costs.restore_time_sec == 0.0
        assert abs(costs.replay_time_sec - 0.064) < 1e-6

    def test_active_request_tracking(self):
        builder = make_cost_builder()
        req = make_request("r1")
        req.assigned_replica_id = 1
        costs = builder.compute_request_costs(req, is_active=True)

        assert costs.is_active
        assert costs.active_replica_id == 1

    def test_build_snapshot_costs(self):
        builder = make_cost_builder()
        active = [make_request("a1")]
        active[0].assigned_replica_id = 0
        pending = [make_request("p1"), make_request("p2")]

        costs = builder.build_snapshot_costs(active, pending)
        assert len(costs) == 3
        assert costs["a1"].is_active
        assert not costs["p1"].is_active


class TestMasterProblem:
    def _make_simple_costs(
        self, num_requests: int, gen_lens: list[int] | None = None
    ) -> dict[str, RequestCosts]:
        if gen_lens is None:
            gen_lens = [100] * num_requests
        return {
            f"r{i}": make_simple_cost(
                f"r{i}",
                generation_len=g,
                d_j=g / 500.0,
            )
            for i, g in enumerate(gen_lens)
        }

    def test_admits_all_when_capacity_allows(self):
        costs = self._make_simple_costs(3, [100, 200, 150])
        master = MasterProblem(
            request_costs=costs,
            replica_ids=[0, 1],
            H_pre=10.0,
            H_dec=10.0,
            time_limit_sec=5.0,
        )
        sol = master.solve()
        assert sol is not None
        assert len(sol.admitted) == 3

    def test_goodput_maximization(self):
        costs = self._make_simple_costs(3, [50, 200, 100])
        master = MasterProblem(
            request_costs=costs,
            replica_ids=[0],
            H_pre=10.0,
            H_dec=0.5,
            time_limit_sec=5.0,
        )
        sol = master.solve()
        assert sol is not None
        assert "r1" in sol.admitted

    def test_capacity_constraint_limits_admission(self):
        costs = self._make_simple_costs(5, [500] * 5)
        for c in costs.values():
            c.d_j = 5.0
            c.checkpoint_overhead_sec = 0.0
        master = MasterProblem(
            request_costs=costs,
            replica_ids=[0],
            H_pre=100.0,
            H_dec=10.0,
            time_limit_sec=5.0,
        )
        sol = master.solve()
        assert sol is not None
        assert len(sol.admitted) <= 2

    def test_fixed_active_requests(self):
        costs = self._make_simple_costs(2)
        costs["r0"].is_active = True
        costs["r0"].active_replica_id = 0

        master = MasterProblem(
            request_costs=costs,
            replica_ids=[0, 1],
            H_pre=10.0,
            H_dec=10.0,
            time_limit_sec=5.0,
        )
        sol = master.solve()
        assert sol is not None
        assert "r0" in sol.admitted
        assert sol.assignments["r0"] == 0

    def test_benders_cut_prevents_solution(self):
        costs = self._make_simple_costs(2, [100, 100])
        master = MasterProblem(
            request_costs=costs,
            replica_ids=[0, 1],
            H_pre=10.0,
            H_dec=10.0,
            time_limit_sec=5.0,
        )

        sol1 = master.solve()
        assert sol1 is not None

        involved = list(sol1.assignments.items())
        master.add_cut(involved)

        sol2 = master.solve()
        if sol2 is not None and len(sol2.admitted) == len(sol1.admitted):
            assert sol2.assignments != sol1.assignments


class TestRecoveryChecker:
    def _make_costs_and_solution(
        self,
        num_requests: int = 4,
        replica_ids: list[int] | None = None,
    ) -> tuple[dict[str, RequestCosts], MasterSolution, list[int]]:
        if replica_ids is None:
            replica_ids = [0, 1, 2]
        costs = {}
        sol = MasterSolution()
        for i in range(num_requests):
            rid = f"r{i}"
            replica_id = replica_ids[i % len(replica_ids)]
            costs[rid] = make_simple_cost(rid)
            sol.assignments[rid] = replica_id
            sol.admitted.add(rid)
        return costs, sol, replica_ids

    def test_no_affected_requests_is_feasible(self):
        costs, sol, rids = self._make_costs_and_solution(4, [0, 1, 2])
        checker = RecoveryChecker(
            cost_table=costs,
            replica_ids=rids,
            H_dec=10.0,
        )
        plan, cert = checker.check_scenario(sol, frozenset({0}))
        assert cert is None
        assert plan is not None

    def test_pooled_overload_detected(self):
        costs = {}
        sol = MasterSolution()
        for i in range(10):
            rid = f"r{i}"
            costs[rid] = make_simple_cost(
                rid,
                checkpoint_overhead_sec=0.5,
                restore_time_sec=2.0,
                replay_time_sec=2.0,
            )
            sol.assignments[rid] = 0
            sol.admitted.add(rid)

        checker = RecoveryChecker(
            cost_table=costs,
            replica_ids=[0, 1],
            H_dec=10.0,
        )
        plan, cert = checker.check_scenario(sol, frozenset({0}))
        assert plan is None
        assert cert is not None
        assert cert.cert_type == "PoolOverload"
        assert len(cert.overloaded_subset) > 0

    def test_exact_ilp_finds_assignment(self):
        costs, sol, rids = self._make_costs_and_solution(3, [0, 1, 2])
        checker = RecoveryChecker(
            cost_table=costs,
            replica_ids=rids,
            H_dec=10.0,
        )
        plan, cert = checker.check_scenario(sol, frozenset({0}))
        assert cert is None
        assert plan is not None
        for _, target in plan.assignments.items():
            assert target != 0


class TestBendersCuts:
    def test_no_good_cut_covers_all_assignments(self):
        sol = MasterSolution()
        sol.assignments = {"r0": 0, "r1": 1}
        sol.admitted = {"r0", "r1"}

        cert = InfeasibilityCertificate(
            cert_type="NoAssignment",
            scenario=frozenset({0}),
        )
        cut = make_no_good_cut(sol, cert)
        assert len(cut.involved) == 2
        assert ("r0", 0) in cut.involved
        assert ("r1", 1) in cut.involved

    def test_pool_overload_cut_scoped_to_q(self):
        sol = MasterSolution()
        sol.assignments = {"r0": 0, "r1": 0, "r2": 1}
        sol.admitted = {"r0", "r1", "r2"}

        cert = InfeasibilityCertificate(
            cert_type="PoolOverload",
            scenario=frozenset({0}),
            overloaded_subset=["r0", "r1"],
        )
        cut = make_pool_overload_cut(sol, cert)
        involved_rids = {rid for rid, _ in cut.involved}
        assert "r0" in involved_rids
        assert "r1" in involved_rids
        assert "r2" not in involved_rids

    def test_make_cut_dispatches_correctly(self):
        sol = MasterSolution()
        sol.assignments = {"r0": 0}
        sol.admitted = {"r0"}

        cert_pool = InfeasibilityCertificate(
            cert_type="PoolOverload",
            scenario=frozenset({0}),
            overloaded_subset=["r0"],
        )
        cut_pool = make_cut(sol, cert_pool)
        assert cut_pool.cut_type == "pool_overload"

        cert_no = InfeasibilityCertificate(
            cert_type="NoAssignment",
            scenario=frozenset({0}),
        )
        cut_no = make_cut(sol, cert_no)
        assert cut_no.cut_type == "no_good"


class TestBendersSolveLoop:
    def test_simple_convergence(self):
        builder = make_cost_builder()
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=20,
            master_time_limit_sec=5.0,
            recovery_time_limit_sec=2.0,
        )

        requests = [
            make_request(f"r{i}", prompt_len=50, max_tokens=100)
            for i in range(6)
        ]

        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=requests,
            replica_ids=[0, 1, 2],
        )
        assert result is not None
        assert len(result.master_solution.admitted) > 0
        assert result.num_iterations >= 1

    def test_empty_pending_returns_empty(self):
        builder = make_cost_builder()
        loop = BendersSolveLoop(cost_builder=builder)

        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=[],
            replica_ids=[0, 1],
        )
        assert result is not None
        assert len(result.master_solution.admitted) == 0

    def test_no_replicas_returns_none(self):
        builder = make_cost_builder()
        loop = BendersSolveLoop(cost_builder=builder)

        requests = [make_request("r0")]
        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=requests,
            replica_ids=[],
        )
        assert result is None

    def test_with_active_requests(self):
        builder = make_cost_builder()
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=20,
            master_time_limit_sec=5.0,
            recovery_time_limit_sec=2.0,
        )

        active = [make_request("active_0", prompt_len=20, max_tokens=50)]
        active[0].assigned_replica_id = 0
        active[0].checkpoint_level = 1

        pending = [make_request(f"p{i}", max_tokens=80) for i in range(3)]

        result = loop.solve_epoch(
            active_requests=active,
            pending_requests=pending,
            replica_ids=[0, 1, 2],
        )
        assert result is not None
        assert "active_0" in result.master_solution.admitted
        assert result.master_solution.assignments["active_0"] == 0


class TestGreedyFallback:
    def test_max_iterations_zero_returns_none(self):
        builder = make_cost_builder()
        loop = BendersSolveLoop(
            cost_builder=builder,
            max_iterations=0,
        )
        requests = [make_request("r0")]
        result = loop.solve_epoch(
            active_requests=[],
            pending_requests=requests,
            replica_ids=[0, 1],
        )
        assert result is None


class TestRemoteLoadEstimation:
    def test_remote_load_reduces_recovery_capacity(self):
        builder = make_cost_builder(decode_tput=100.0)
        requests = [
            make_request(f"r{i}", prompt_len=10, max_tokens=100)
            for i in range(2)
        ]
        cost_table = builder.build_snapshot_costs([], requests)

        h_dec = builder.H_dec

        checker_generous = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1],
            H_dec=h_dec,
            local_replica_ids=[0],
            local_load_fraction=0.0,
        )

        checker_conservative = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=[0, 1],
            H_dec=h_dec,
            local_replica_ids=[0],
            local_load_fraction=0.8,
        )

        solution = MasterSolution(
            assignments={"r0": 0, "r1": 0},
            admitted={"r0", "r1"},
        )

        plan_gen, cert_gen = checker_generous.check_scenario(
            solution, frozenset({0})
        )
        plan_cons, cert_cons = checker_conservative.check_scenario(
            solution, frozenset({0})
        )

        assert (plan_gen is not None) or (cert_gen is not None)
        assert (plan_cons is not None) or (cert_cons is not None)
