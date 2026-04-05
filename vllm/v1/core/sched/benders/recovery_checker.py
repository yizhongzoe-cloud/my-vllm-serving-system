# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recovery checker (subproblem) for the Benders-style robust FT solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.sched.benders.cost_tables import RequestCosts
from vllm.v1.core.sched.benders.master import MasterSolution

logger = init_logger(__name__)

SCALE: int = 1_000_000


def _to_int(value: float) -> int:
    return int(round(value * SCALE))


@dataclass
class InfeasibilityCertificate:
    cert_type: str
    scenario: frozenset[int]
    overloaded_subset: list[str] = field(default_factory=list)
    overload_margin: float = 0.0


@dataclass
class RecoveryPlan:
    scenario: frozenset[int]
    assignments: dict[str, int] = field(default_factory=dict)


class RecoveryChecker:
    """Checks recovery feasibility for each failure scenario."""

    def __init__(
        self,
        cost_table: dict[str, RequestCosts],
        replica_ids: list[int],
        H_dec: float,
        M_cap: int = 0,
        detection_time_sec: float = 0.1,
        decode_throughput: float = 0.0,
        time_limit_sec: float = 0.5,
        local_replica_ids: list[int] | None = None,
        local_load_fraction: float = 0.0,
        replica_load_overrides: dict[int, float] | None = None,
        replica_mem_overrides: dict[int, float] | None = None,
        # Decode-first capacity model
        decode_capacity: int = 0,
        decode_cap_model: "DecodeCapacityModel | None" = None,
        use_decode_first: bool = False,
    ) -> None:
        self._costs = cost_table
        self._replica_ids = list(replica_ids)
        self._H_dec = H_dec
        self._M_cap = M_cap
        self._detection_time_sec = detection_time_sec
        self._decode_throughput = decode_throughput
        self._time_limit_sec = time_limit_sec
        self._local_replica_ids = set(local_replica_ids or replica_ids)
        self._local_load_fraction = local_load_fraction
        self._replica_load_overrides = replica_load_overrides
        self._replica_mem_overrides = replica_mem_overrides
        # Decode-first
        self._use_decode_first = use_decode_first
        self._decode_capacity = decode_capacity
        self._decode_cap_model = decode_cap_model

    def check_scenario(
        self,
        solution: MasterSolution,
        omega: frozenset[int],
    ) -> tuple[RecoveryPlan | None, InfeasibilityCertificate | None]:
        omega_set = set(omega)
        surviving = [r for r in self._replica_ids if r not in omega_set]

        if not surviving:
            return None, InfeasibilityCertificate(
                cert_type="NoAssignment",
                scenario=omega,
            )

        affected: list[str] = []
        for req_id, replica_id in solution.assignments.items():
            if replica_id in omega_set:
                affected.append(req_id)

        if not affected:
            return RecoveryPlan(scenario=omega), None

        if self._use_decode_first:
            return self._check_scenario_decode_first(
                solution, omega, omega_set, surviving, affected)
        return self._check_scenario_legacy(
            solution, omega, omega_set, surviving, affected)

    def _check_scenario_decode_first(
        self,
        solution: MasterSolution,
        omega: frozenset[int],
        omega_set: set[int],
        surviving: list[int],
        affected: list[str],
    ) -> tuple[RecoveryPlan | None, InfeasibilityCertificate | None]:
        """Recovery check using decode-first capacity model (dual-dimension)."""

        # Per-request recovery: each affected request occupies 1 decode slot
        # and needs replay_tokens of prefill work.
        w_decode: dict[str, float] = {}
        w_replay: dict[str, int] = {}
        for req_id in affected:
            costs = self._costs[req_id]
            w_decode[req_id] = costs.w_dec
            w_replay[req_id] = costs.replay_tokens

        # Surviving replica budgets
        surv_decode_count: dict[int, int] = {r: 0 for r in surviving}
        for req_id, replica_id in solution.assignments.items():
            if replica_id not in omega_set and req_id not in affected:
                costs = self._costs[req_id]
                if costs.is_active:
                    surv_decode_count[replica_id] += int(costs.w_dec)

        # Decode slot budget
        u_decode: dict[int, int] = {}
        for r in surviving:
            u_decode[r] = max(0, self._decode_capacity - surv_decode_count[r])

        # Prefill budget (worst case: all affected recover to this replica)
        u_prefill: dict[int, int] = {}
        for r in surviving:
            if self._decode_cap_model is not None:
                u_prefill[r] = self._decode_cap_model.residual_prefill_capacity(
                    surv_decode_count[r] + len(affected)
                )
            else:
                u_prefill[r] = 999999

        # Pool overload check — decode dimension
        total_slots_needed = len(affected)
        total_slots_available = sum(u_decode.values())
        if total_slots_needed > total_slots_available:
            return None, InfeasibilityCertificate(
                cert_type="PoolOverload",
                scenario=omega,
                overloaded_subset=affected[:total_slots_needed - total_slots_available],
                overload_margin=float(total_slots_needed - total_slots_available),
            )

        # Pool overload check — prefill dimension
        total_replay = sum(w_replay.values())
        total_prefill_available = sum(u_prefill.values())
        if total_replay > total_prefill_available:
            sorted_by_replay = sorted(
                affected, key=lambda r: w_replay[r], reverse=True)
            overloaded: list[str] = []
            partial = 0
            for req_id in sorted_by_replay:
                overloaded.append(req_id)
                partial += w_replay[req_id]
                if partial > total_prefill_available:
                    break
            return None, InfeasibilityCertificate(
                cert_type="PoolOverload",
                scenario=omega,
                overloaded_subset=overloaded,
                overload_margin=float(total_replay - total_prefill_available),
            )

        # Memory check
        free_mem = self._compute_free_mem(solution, omega_set, surviving)

        # Feasible edges (dual-dimension + TTFT/TPOT + gap + memory)
        feasible_edge: dict[tuple[str, int], bool] = {}
        for req_id in affected:
            costs = self._costs[req_id]
            for replica_id in surviving:
                feasible = True
                # Decode slot check
                if costs.w_dec > u_decode.get(replica_id, 0):
                    feasible = False
                # Prefill/replay check
                if costs.replay_tokens > u_prefill.get(replica_id, 0):
                    feasible = False
                # Memory check
                if (self._M_cap > 0
                        and costs.recovery_mem_bytes > free_mem.get(replica_id, 0.0)):
                    feasible = False
                # Gap SLO check
                if (costs.gap_slo_sec is not None
                        and costs.gap_time_sec > costs.gap_slo_sec):
                    feasible = False
                # TTFT SLO check
                if (costs.ttft_slo_sec is not None
                        and costs.p_j > costs.ttft_slo_sec):
                    feasible = False
                # TPOT SLO check
                if (costs.tpot_slo_sec is not None
                        and self._decode_throughput > 0
                        and (1.0 / self._decode_throughput) > costs.tpot_slo_sec):
                    feasible = False
                feasible_edge[(req_id, replica_id)] = feasible

        return self._solve_recovery_ilp(
            affected, surviving, w_decode, u_decode,
            free_mem, feasible_edge, omega,
            prefill_work=w_replay, prefill_budget=u_prefill,
        )

    def _check_scenario_legacy(
        self,
        solution: MasterSolution,
        omega: frozenset[int],
        omega_set: set[int],
        surviving: list[int],
        affected: list[str],
    ) -> tuple[RecoveryPlan | None, InfeasibilityCertificate | None]:
        """Legacy recovery check using serial-time model."""

        w: dict[str, float] = {}
        w_total: dict[str, float] = {}
        for req_id in affected:
            costs = self._costs[req_id]
            w[req_id] = costs.restore_time_sec + costs.replay_time_sec
            w_total[req_id] = (
                w[req_id] + costs.d_j + costs.checkpoint_overhead_sec
            )

        u_r: dict[int, float] = {}
        surv_load: dict[int, float] = {r: 0.0 for r in surviving}
        for req_id, replica_id in solution.assignments.items():
            if replica_id not in omega_set and req_id not in affected:
                costs = self._costs[req_id]
                surv_load[replica_id] += (
                    costs.d_j + costs.checkpoint_overhead_sec
                )

        for r in surviving:
            if r not in self._local_replica_ids:
                if (self._replica_load_overrides is not None
                        and r in self._replica_load_overrides):
                    real_load = self._replica_load_overrides[r] * self._H_dec
                    surv_load[r] = max(surv_load[r], real_load)
                else:
                    estimated_load = self._local_load_fraction * self._H_dec
                    surv_load[r] = max(surv_load[r], estimated_load)

        for r in surviving:
            u_r[r] = max(0.0, self._H_dec - surv_load[r])

        total_recovery_work = sum(w_total.values())
        pooled_headroom = sum(u_r.values())
        if total_recovery_work > pooled_headroom + 1e-9:
            sorted_affected = sorted(
                affected, key=lambda req_id: w_total[req_id], reverse=True
            )
            overloaded_subset: list[str] = []
            partial_sum = 0.0
            for req_id in sorted_affected:
                overloaded_subset.append(req_id)
                partial_sum += w_total[req_id]
                if partial_sum > pooled_headroom:
                    break
            return None, InfeasibilityCertificate(
                cert_type="PoolOverload",
                scenario=omega,
                overloaded_subset=overloaded_subset,
                overload_margin=total_recovery_work - pooled_headroom,
            )

        free_mem = self._compute_free_mem(solution, omega_set, surviving)

        feasible_edge: dict[tuple[str, int], bool] = {}
        for req_id in affected:
            costs = self._costs[req_id]
            for replica_id in surviving:
                feasible = True
                if self._M_cap > 0 and costs.recovery_mem_bytes > free_mem.get(
                    replica_id, 0.0
                ):
                    feasible = False
                if (costs.gap_slo_sec is not None
                        and costs.gap_time_sec > costs.gap_slo_sec):
                    feasible = False
                feasible_edge[(req_id, replica_id)] = feasible

        return self._solve_recovery_ilp(
            affected, surviving, w_total, u_r,
            free_mem, feasible_edge, omega,
        )

    def _compute_free_mem(
        self,
        solution: MasterSolution,
        omega_set: set[int],
        surviving: list[int],
    ) -> dict[int, float]:
        """Compute free memory on surviving replicas."""
        free_mem: dict[int, float] = {}
        if self._M_cap > 0:
            survivor_mem: dict[int, float] = {r: 0.0 for r in surviving}
            affected_set = {
                req_id for req_id, rid in solution.assignments.items()
                if rid in omega_set
            }
            for req_id, replica_id in solution.assignments.items():
                if replica_id not in omega_set and req_id not in affected_set:
                    survivor_mem[replica_id] += self._costs[
                        req_id].run_mem_bytes
            for r in surviving:
                if r not in self._local_replica_ids:
                    if (self._replica_mem_overrides is not None
                            and r in self._replica_mem_overrides):
                        survivor_mem[r] = max(
                            survivor_mem[r], self._replica_mem_overrides[r])
                    else:
                        survivor_mem[r] = max(
                            survivor_mem[r],
                            self._local_load_fraction * self._M_cap,
                        )
            for r in surviving:
                free_mem[r] = max(0.0, self._M_cap - survivor_mem[r])
        return free_mem

    def _solve_recovery_ilp(
        self,
        affected: list[str],
        surviving: list[int],
        work: dict[str, float],
        residual_budget: dict[int, float],
        free_mem: dict[int, float],
        feasible_edge: dict[tuple[str, int], bool],
        omega: frozenset[int],
        # Decode-first: optional prefill dimension
        prefill_work: dict[str, int] | None = None,
        prefill_budget: dict[int, int] | None = None,
    ) -> tuple[RecoveryPlan | None, InfeasibilityCertificate | None]:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            logger.error("ortools required for recovery checker")
            return None, InfeasibilityCertificate(
                cert_type="NoAssignment", scenario=omega
            )

        model = cp_model.CpModel()
        assign: dict[tuple[str, int], cp_model.IntVar] = {}
        for req_id in affected:
            for replica_id in surviving:
                if feasible_edge.get((req_id, replica_id), False):
                    assign[(req_id, replica_id)] = model.new_bool_var(
                        f"a_{req_id}_{replica_id}")

        for req_id in affected:
            vars_for_req = [
                assign[(req_id, replica_id)]
                for replica_id in surviving
                if (req_id, replica_id) in assign
            ]
            if not vars_for_req:
                return None, InfeasibilityCertificate(
                    cert_type="NoAssignment",
                    scenario=omega,
                    overloaded_subset=[req_id],
                )
            model.add(sum(vars_for_req) == 1)

        for replica_id in surviving:
            work_terms = []
            for req_id in affected:
                if (req_id, replica_id) in assign:
                    work_terms.append(
                        _to_int(work[req_id]) * assign[(req_id, replica_id)])
            if work_terms:
                model.add(sum(work_terms) <= _to_int(
                    residual_budget.get(replica_id, 0.0)))

        # Optional prefill constraint (decode-first model)
        if prefill_work is not None and prefill_budget is not None:
            for replica_id in surviving:
                replay_terms = []
                for req_id in affected:
                    if (req_id, replica_id) in assign:
                        rp = prefill_work.get(req_id, 0)
                        if rp > 0:
                            replay_terms.append(
                                rp * assign[(req_id, replica_id)])
                if replay_terms:
                    model.add(sum(replay_terms) <= prefill_budget.get(
                        replica_id, 0))

        if self._M_cap > 0:
            for replica_id in surviving:
                mem_terms = []
                for req_id in affected:
                    if (req_id, replica_id) in assign:
                        mem = int(self._costs[req_id].recovery_mem_bytes)
                        if mem > 0:
                            mem_terms.append(mem * assign[(req_id, replica_id)])
                if mem_terms:
                    model.add(sum(mem_terms) <= int(
                        free_mem.get(replica_id, 0.0)))

        model.minimize(0)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._time_limit_sec

        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, InfeasibilityCertificate(
                cert_type="NoAssignment",
                scenario=omega,
            )

        plan = RecoveryPlan(scenario=omega)
        for req_id in affected:
            for replica_id in surviving:
                if ((req_id, replica_id) in assign
                        and solver.value(assign[(req_id, replica_id)]) == 1):
                    plan.assignments[req_id] = replica_id
                    break
        return plan, None
