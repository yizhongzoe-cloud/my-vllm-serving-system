# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Master MIP for the Benders-style robust FT solver.

The solver now reasons only about request admission and replica routing:

    x_{j,r} ∈ {0,1}

Checkpointing is a runtime-local online policy, so the master consumes
only scalar per-request costs derived from the real published state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.sched.benders.cost_tables import RequestCosts

logger = init_logger(__name__)

SCALE: int = 1_000_000


def _to_int(value: float) -> int:
    return int(round(value * SCALE))


@dataclass
class MasterSolution:
    """Output of the master MIP."""

    assignments: dict[str, int] = field(default_factory=dict)
    admitted: set[str] = field(default_factory=set)
    objective_value: float = 0.0


class MasterProblem:
    """Builds and solves the master MIP at each Benders iteration."""

    def __init__(
        self,
        request_costs: dict[str, RequestCosts],
        replica_ids: list[int],
        H_pre: float,
        H_dec: float,
        M_cap: int = 0,
        time_limit_sec: float = 1.0,
    ) -> None:
        self._costs = request_costs
        self._replica_ids = list(replica_ids)
        self._H_pre = _to_int(H_pre)
        self._H_dec = _to_int(H_dec)
        self._M_cap = M_cap
        self._time_limit_sec = time_limit_sec
        self._cuts: list[list[tuple[str, int]]] = []

    def add_cut(self, involved: list[tuple[str, int]]) -> None:
        self._cuts.append(list(involved))

    @property
    def num_cuts(self) -> int:
        return len(self._cuts)

    def solve(self) -> MasterSolution | None:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            logger.error(
                "ortools is required for ft_benders policy. "
                "Install with: pip install ortools"
            )
            return None

        model = cp_model.CpModel()

        x: dict[tuple[str, int], cp_model.IntVar] = {}
        y: dict[str, cp_model.IntVar] = {}

        for req_id, costs in self._costs.items():
            x_vars_for_req: list[cp_model.IntVar] = []
            for r in self._replica_ids:
                var = model.new_bool_var(f"x_{req_id}_{r}")
                x[(req_id, r)] = var
                x_vars_for_req.append(var)

            y_var = model.new_bool_var(f"y_{req_id}")
            y[req_id] = y_var
            model.add(sum(x_vars_for_req) == y_var)

            if costs.is_active and costs.active_replica_id is not None:
                r0 = costs.active_replica_id
                if (req_id, r0) in x:
                    model.add(x[(req_id, r0)] == 1)
                for r in self._replica_ids:
                    if r != r0 and (req_id, r) in x:
                        model.add(x[(req_id, r)] == 0)

        for req_id, costs in self._costs.items():
            if costs.is_active:
                continue

            infeasible = False

            if costs.ttft_slo_sec is not None and costs.p_j > costs.ttft_slo_sec:
                infeasible = True

            if costs.tpot_slo_sec is not None and costs.G_j > 0:
                tpot_est = costs.d_j / costs.G_j
                if tpot_est > costs.tpot_slo_sec:
                    infeasible = True

            if costs.gap_slo_sec is not None and costs.gap_time_sec > costs.gap_slo_sec:
                infeasible = True

            if infeasible:
                for r in self._replica_ids:
                    if (req_id, r) in x:
                        model.add(x[(req_id, r)] == 0)

        for r in self._replica_ids:
            prefill_terms = []
            for req_id, costs in self._costs.items():
                if (req_id, r) in x:
                    prefill_terms.append(_to_int(costs.p_j) * x[(req_id, r)])
            if prefill_terms:
                model.add(sum(prefill_terms) <= self._H_pre)

        for r in self._replica_ids:
            decode_terms = []
            for req_id, costs in self._costs.items():
                if (req_id, r) in x:
                    coeff = _to_int(costs.d_j + costs.checkpoint_overhead_sec)
                    decode_terms.append(coeff * x[(req_id, r)])
            if decode_terms:
                model.add(sum(decode_terms) <= self._H_dec)

        if self._M_cap > 0:
            for r in self._replica_ids:
                mem_terms = []
                for req_id, costs in self._costs.items():
                    mem = int(costs.run_mem_bytes)
                    if mem > 0 and (req_id, r) in x:
                        mem_terms.append(mem * x[(req_id, r)])
                if mem_terms:
                    model.add(sum(mem_terms) <= self._M_cap)

        for cut_involved in self._cuts:
            cut_vars = []
            for (req_id, r) in cut_involved:
                if (req_id, r) in x:
                    cut_vars.append(x[(req_id, r)])
            if cut_vars:
                model.add(sum(cut_vars) <= len(cut_vars) - 1)

        obj_terms = []
        for req_id, costs in self._costs.items():
            obj_terms.append(costs.G_j * y[req_id])

        if len(self._replica_ids) > 1:
            load_vars: dict[int, cp_model.LinearExpr] = {}
            for r in self._replica_ids:
                terms = [
                    x[(req_id, r)]
                    for req_id in self._costs
                    if (req_id, r) in x
                ]
                load_vars[r] = sum(terms) if terms else 0

            max_load = model.new_int_var(
                0, len(self._costs), "max_replica_load"
            )
            for r in self._replica_ids:
                model.add(max_load >= load_vars[r])
            avg_gj = (
                sum(c.G_j for c in self._costs.values()) / len(self._costs)
                if self._costs else 1
            )
            penalty_weight = max(1, int(avg_gj // 2))
            obj_terms.append(-penalty_weight * max_load)

        model.maximize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._time_limit_sec

        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.debug("Master MIP: no feasible solution (status=%s)", status)
            return None

        solution = MasterSolution()
        solution.objective_value = solver.objective_value

        for req_id in self._costs:
            if solver.value(y[req_id]) == 1:
                solution.admitted.add(req_id)
                for r in self._replica_ids:
                    if (req_id, r) in x and solver.value(x[(req_id, r)]) == 1:
                        solution.assignments[req_id] = r
                        break

        return solution
