# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benders decomposition solve loop for the robust FT solver.

Corresponds to §10.7.12 Algorithm 1 of idea_online_periodic.md.

At each decision epoch:
1. Build the current snapshot J_t from active + pending requests.
2. Enumerate failure scenarios: Ω_k = {ω ⊆ R : |ω| ≤ k} for
   k = max_gpu_failures.  Only scenarios containing at least one
   local replica are relevant (remote-only failures don't affect
   local admission).
3. Start the epoch master with no cuts.
4. Solve the master on the current snapshot.
5. For each failure scenario, run the recovery checker.
6. If all scenarios feasible → commit decisions.
7. Otherwise add one cut and repeat.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder, RequestCosts
from vllm.v1.core.sched.benders.cuts import make_cut
from vllm.v1.core.sched.benders.master import MasterProblem, MasterSolution
from vllm.v1.core.sched.benders.recovery_checker import (
    RecoveryChecker,
    RecoveryPlan,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BendersSolveResult:
    """Result of one decision-epoch solve."""

    master_solution: MasterSolution
    recovery_plans: dict[frozenset[int], RecoveryPlan] = field(
        default_factory=dict
    )
    num_iterations: int = 0
    total_solve_time_sec: float = 0.0


class BendersSolveLoop:
    """Orchestrates the Benders decomposition for one decision epoch."""

    def __init__(
        self,
        cost_builder: CostTableBuilder,
        max_iterations: int = 20,
        master_time_limit_sec: float = 1.0,
        recovery_time_limit_sec: float = 0.5,
        max_gpu_failures: int = 1,
    ) -> None:
        self._cost_builder = cost_builder
        self._max_iterations = max_iterations
        self._master_time_limit = master_time_limit_sec
        self._recovery_time_limit = recovery_time_limit_sec
        self._max_gpu_failures = max_gpu_failures

    def solve_epoch(
        self,
        active_requests: list[Request],
        pending_requests: list[Request],
        replica_ids: list[int],
        all_replica_ids: list[int] | None = None,
    ) -> BendersSolveResult | None:
        """Run the full Benders decomposition for one decision epoch.

        Args:
            active_requests: Requests already running (placement fixed).
            pending_requests: Newly arrived requests (free variables).
            replica_ids: IDs of replicas for admission/routing in master MIP.
                In DP mode this is just [self_replica_id].
            all_replica_ids: IDs of ALL replicas in the cluster, used for
                recovery subproblems. If None, defaults to replica_ids.
                In DP mode this is all dp_size replicas, so the recovery
                checker can verify that survivors can absorb requests
                if this replica fails.

        Returns:
            BendersSolveResult if a feasible solution is found.
            None if max iterations exceeded or solver fails (caller
            should fall back to greedy).
        """
        cost_table = self._cost_builder.build_snapshot_costs(
            active_requests, pending_requests
        )
        return self.solve_epoch_from_costs(
            cost_table, replica_ids, all_replica_ids
        )

    def solve_epoch_from_costs(
        self,
        cost_table: dict[str, RequestCosts],
        replica_ids: list[int],
        all_replica_ids: list[int] | None = None,
        replica_load_overrides: dict[int, float] | None = None,
        replica_mem_overrides: dict[int, float] | None = None,
    ) -> BendersSolveResult | None:
        """Run Benders decomposition from a pre-built cost table.

        This is the core solve routine. ``solve_epoch()`` builds the cost
        table from ``Request`` objects and delegates here.  The centralized
        client calls this directly with a cost table built from
        ``RequestSnapshot`` + ``EngineCoreRequest`` via
        ``CostTableBuilder.build_global_costs()``.

        Args:
            cost_table: Pre-built {request_id: RequestCosts} dict.
            replica_ids: IDs of replicas for admission/routing in master MIP.
            all_replica_ids: IDs of ALL replicas (for recovery subproblems).
            replica_load_overrides: Per-replica load fraction [0, 1] from
                ReplicaSnapshots. Replaces heuristic estimation in the
                recovery checker for replicas that have real data.
            replica_mem_overrides: Per-replica used memory (bytes) from
                ReplicaSnapshots. Same purpose as load overrides.

        Returns:
            BendersSolveResult or None (caller should fall back to greedy).
        """
        start_time = time.monotonic()

        if not replica_ids:
            return None

        recovery_replica_ids = all_replica_ids or replica_ids

        if not cost_table:
            return BendersSolveResult(
                master_solution=MasterSolution(),
                num_iterations=0,
                total_solve_time_sec=0.0,
            )

        # Compute local load fraction for conservative remote estimation.
        # This is the fraction of H_dec used by requests on local replicas.
        local_load = 0.0
        for req_id, costs in cost_table.items():
            if costs.is_active:
                local_load += costs.d_j + costs.checkpoint_overhead_sec
        H_dec_val = self._cost_builder.H_dec
        local_load_fraction = (
            min(1.0, local_load / H_dec_val) if H_dec_val > 0 else 0.0
        )

        # Decode-first capacity: compute per-replica decode load and
        # residual prefill capacity.
        use_df = self._cost_builder.use_decode_first_model
        decode_cap = 0
        residual_prefill: dict[int, int] = {}
        if use_df:
            decode_cap = self._cost_builder.get_decode_capacity()
            # Count active decode requests per replica
            L_current: dict[int, int] = {r: 0 for r in recovery_replica_ids}
            for req_id, costs in cost_table.items():
                if costs.is_active and costs.active_replica_id is not None:
                    r = costs.active_replica_id
                    if r in L_current:
                        L_current[r] += int(costs.w_dec)
            # Conservative: add pending requests evenly distributed
            pending_count = sum(
                1 for c in cost_table.values() if not c.is_active
            )
            per_replica_pending = (
                pending_count // max(len(replica_ids), 1)
            )
            residual_prefill = {
                r: self._cost_builder.get_residual_prefill_capacity(
                    L_current.get(r, 0) + per_replica_pending
                )
                for r in recovery_replica_ids
            }

        # Step 2: Enumerate failure scenarios up to max_gpu_failures.
        # Ω_k = {ω ⊆ recovery_replica_ids : 1 ≤ |ω| ≤ k}.
        # Only keep scenarios containing at least one local replica —
        # remote-only failures don't affect local admission decisions.
        local_set = set(replica_ids)
        scenarios: list[frozenset[int]] = []
        for k in range(1, self._max_gpu_failures + 1):
            for combo in itertools.combinations(recovery_replica_ids, k):
                omega = frozenset(combo)
                if omega & local_set:
                    scenarios.append(omega)

        if not scenarios:
            # No relevant failure scenarios (e.g. single replica).
            # Return master solution without recovery checking.
            solution = MasterProblem(
                request_costs=cost_table,
                replica_ids=replica_ids,
                H_pre=self._cost_builder.H_pre,
                H_dec=H_dec_val,
                M_cap=self._cost_builder.memory_capacity_bytes,
                time_limit_sec=self._master_time_limit,
                decode_capacity=decode_cap,
                residual_prefill_capacity=residual_prefill,
                use_decode_first=use_df,
            ).solve()
            if solution is None:
                return None
            return BendersSolveResult(
                master_solution=solution,
                num_iterations=1,
                total_solve_time_sec=time.monotonic() - start_time,
            )

        num_scenarios = len(scenarios)
        if num_scenarios > 50:
            logger.warning(
                "Benders: %d failure scenarios (dp_size=%d, "
                "max_gpu_failures=%d). Solver may be slow.",
                num_scenarios,
                len(recovery_replica_ids),
                self._max_gpu_failures,
            )

        # Step 3: Initialize master (no cuts).
        # Master only sees the replicas this EngineCore can route to.
        master = MasterProblem(
            request_costs=cost_table,
            replica_ids=replica_ids,
            H_pre=self._cost_builder.H_pre,
            H_dec=H_dec_val,
            M_cap=self._cost_builder.memory_capacity_bytes,
            time_limit_sec=self._master_time_limit,
            decode_capacity=decode_cap,
            residual_prefill_capacity=residual_prefill,
            use_decode_first=use_df,
        )

        # Recovery checker sees ALL replicas so it can verify that
        # survivors have enough capacity to absorb affected requests.
        # In per-engine mode, remote replicas' load is conservatively
        # estimated using local_load_fraction. In centralized mode,
        # replica_load_overrides provides real per-replica load from
        # ReplicaSnapshots, replacing the heuristic.
        recovery_checker = RecoveryChecker(
            cost_table=cost_table,
            replica_ids=recovery_replica_ids,
            H_dec=H_dec_val,
            M_cap=self._cost_builder.memory_capacity_bytes,
            detection_time_sec=self._cost_builder.detection_time_sec,
            decode_throughput=self._cost_builder.decode_throughput,
            time_limit_sec=self._recovery_time_limit,
            local_replica_ids=replica_ids,
            local_load_fraction=local_load_fraction,
            replica_load_overrides=replica_load_overrides,
            replica_mem_overrides=replica_mem_overrides,
            # Decode-first capacity model
            decode_capacity=decode_cap,
            decode_cap_model=(
                self._cost_builder._decode_cap_model if use_df else None
            ),
            use_decode_first=use_df,
        )

        # Step 4–7: Iterate.
        best_result: BendersSolveResult | None = None

        for iteration in range(self._max_iterations):
            # Step 4: Solve master.
            solution = master.solve()
            if solution is None:
                logger.debug(
                    "Benders iteration %d: master infeasible", iteration
                )
                break

            # Step 5: Check all failure scenarios.
            all_feasible = True
            recovery_plans: dict[frozenset[int], RecoveryPlan] = {}

            for omega in scenarios:
                plan, cert = recovery_checker.check_scenario(solution, omega)

                if cert is not None:
                    # Infeasible — generate cut and break.
                    cut = make_cut(solution, cert)
                    master.add_cut(cut.involved)
                    all_feasible = False
                    logger.debug(
                        "Benders iteration %d: scenario %s infeasible "
                        "(%s), added cut #%d",
                        iteration,
                        omega,
                        cert.cert_type,
                        master.num_cuts,
                    )
                    break
                else:
                    if plan is not None:
                        recovery_plans[omega] = plan

            # Step 6: If all feasible, return.
            if all_feasible:
                elapsed = time.monotonic() - start_time
                result = BendersSolveResult(
                    master_solution=solution,
                    recovery_plans=recovery_plans,
                    num_iterations=iteration + 1,
                    total_solve_time_sec=elapsed,
                )
                num_active = sum(
                    1 for c in cost_table.values() if c.is_active
                )
                num_pending = len(cost_table) - num_active
                # new_admitted = pending requests that got admitted
                # (active requests are always in solution.admitted)
                new_admitted = len(solution.admitted) - num_active
                logger.info(
                    "Benders converged in %d iterations (%.3fs): "
                    "new_admitted=%d/%d pending, active=%d, "
                    "goodput=%.0f",
                    result.num_iterations,
                    result.total_solve_time_sec,
                    new_admitted,
                    num_pending,
                    num_active,
                    solution.objective_value,
                )
                return result

        # Step 7: Max iterations exceeded.
        elapsed = time.monotonic() - start_time
        logger.warning(
            "Benders did not converge in %d iterations (%.3fs); "
            "falling back to greedy",
            self._max_iterations,
            elapsed,
        )
        return None
