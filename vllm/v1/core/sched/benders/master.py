# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Master MIP for the Benders-style robust FT solver.

The solver now reasons only about request admission and replica routing:

    x_{j,r} ∈ {0,1}

Checkpointing is a runtime-local online policy, so the master consumes
only scalar per-request costs derived from the real published state.
"""

from __future__ import annotations

import os
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
        # Decode-first capacity model parameters
        decode_capacity: int = 0,
        residual_prefill_capacity: dict[int, int] | None = None,
        use_decode_first: bool = False,
    ) -> None:
        self._costs = request_costs
        self._replica_ids = list(replica_ids)
        self._H_pre = _to_int(H_pre)
        self._H_dec = _to_int(H_dec)
        self._M_cap = M_cap
        self._time_limit_sec = time_limit_sec
        self._cuts: list[list[tuple[str, int]]] = []
        # Decode-first
        self._use_decode_first = use_decode_first
        self._decode_capacity = decode_capacity
        self._residual_prefill_capacity = residual_prefill_capacity or {}

    def add_cut(self, involved: list[tuple[str, int]]) -> None:
        self._cuts.append(list(involved))

    @property
    def num_cuts(self) -> int:
        return len(self._cuts)

    def _try_trivial_greedy(self) -> MasterSolution | None:
        """FIX #3: trivial-case skip.

        When the master problem is small and trivially solvable, return a
        greedy FCFS solution without invoking CP-SAT. Controlled by env
        FT_SOLVER_TRIVIAL_SKIP=1 (default on).

        Bail-out conditions (defer to MIP):
          - non-active requests > 8 (non-trivial size)
          - any Benders cuts present (recovery-infeasible scenario must
            be re-solved with cuts applied — greedy can't honor cuts)
          - use_decode_first is True (decode_capacity / residual_prefill
            constraints are non-trivial; correctness > speed here)

        Otherwise, admits requests whose TTFT/TPOT/gap SLO is feasible,
        respecting per-replica running_cap if set. This matches the MIP's
        feasibility filter but skips the CP-SAT solve.
        """
        # Default OFF: the paper's Benders ILP is the algorithmic
        # contribution. Bypassing it via greedy trivial-skip defeats
        # the paper's value proposition. Legitimate MIP-preserving
        # speedups (warm-start, time-cap, model cache) are preferred.
        # Enable only for diagnostic A/B ablation.
        if os.environ.get("FT_SOLVER_TRIVIAL_SKIP", "0") != "1":
            return None

        # Bail if cuts are present: solve_loop adds cuts across iterations
        # for recovery-infeasible scenarios. Greedy can't honor cuts, so
        # second and later iterations must use the MIP.
        if self._cuts:
            return None

        # Bail if decode-first capacity model is active — greedy would
        # need to check Cap_dec / residual_prefill per replica, which
        # basically reconstructs the LP. MIP is cleaner.
        if self._use_decode_first:
            return None

        non_active = [
            (req_id, costs)
            for req_id, costs in self._costs.items()
            if not costs.is_active
        ]
        # Threshold tunable via FT_SOLVER_TRIVIAL_MAX (default 8).
        # Long-prompt workloads with queue build-up benefit from higher values
        # so greedy admits more under load rather than handing off to MIP.
        try:
            max_trivial = int(os.environ.get("FT_SOLVER_TRIVIAL_MAX", "8"))
        except ValueError:
            max_trivial = 8
        if len(non_active) > max_trivial:
            return None  # non-trivial size, defer to MIP

        try:
            running_cap = int(os.environ.get("FT_SOLVER_RUNNING_CAP", "0"))
        except ValueError:
            running_cap = 0

        solution = MasterSolution()

        # Keep active requests pinned to their current replica.
        per_replica_count: dict[int, int] = {r: 0 for r in self._replica_ids}
        for req_id, costs in self._costs.items():
            if costs.is_active and costs.active_replica_id is not None:
                r0 = costs.active_replica_id
                if r0 in per_replica_count:
                    solution.assignments[req_id] = r0
                    solution.admitted.add(req_id)
                    per_replica_count[r0] += 1

        # SLO feasibility filter matches MIP infeasibility logic around
        # lines 195-215 (ttft/tpot/gap SLO).
        feasible: list[tuple[str, "RequestCosts"]] = []
        for req_id, costs in non_active:
            if costs.ttft_slo_sec is not None and costs.p_j > costs.ttft_slo_sec:
                continue
            if costs.tpot_slo_sec is not None and costs.G_j > 0:
                tpot_est = costs.d_j / costs.G_j
                if tpot_est > costs.tpot_slo_sec:
                    continue
            if costs.gap_slo_sec is not None and costs.gap_time_sec > costs.gap_slo_sec:
                continue
            feasible.append((req_id, costs))

        replicas = self._replica_ids
        if not replicas:
            return None

        # Greedy FCFS with per-replica running_cap constraint + load-balance.
        for req_id, costs in feasible:
            best_r = min(replicas, key=lambda r: per_replica_count[r])
            if running_cap > 0 and per_replica_count[best_r] >= running_cap:
                continue  # cap reached on all replicas; drop this req
            solution.assignments[req_id] = best_r
            solution.admitted.add(req_id)
            per_replica_count[best_r] += 1

        # Objective approximation for logging (same scale as MIP).
        SLO_WEIGHT_SCALE = 1000
        obj = 0
        for req_id in solution.admitted:
            costs = self._costs[req_id]
            obj += int(round(costs.G_j * costs.slo_weight * SLO_WEIGHT_SCALE))
        solution.objective_value = obj

        logger.debug(
            "MIP trivial-skip: %d/%d admitted via greedy FCFS",
            len(solution.admitted), len(self._costs),
        )
        return solution

    def solve(
        self, prev_solution: MasterSolution | None = None
    ) -> MasterSolution | None:
        # FIX #3: trivial-case skip.
        trivial = self._try_trivial_greedy()
        if trivial is not None:
            return trivial

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

        # FIX #4 (A3 — presolve pruning): determine SLO-infeasibility BEFORE
        # creating variables. Standard MIP preprocessing: every variable known
        # to be 0 at optimal is removed from the model upfront, which shrinks
        # the CP-SAT search space. Not a bypass — the MIP still runs on the
        # reduced feasible set and returns an optimal solution w.r.t. the
        # original problem.
        pruned_req_ids: set[str] = set()
        for req_id, costs in self._costs.items():
            if costs.is_active:
                continue  # active requests always in model (placement fixed)
            if costs.ttft_slo_sec is not None and costs.p_j > costs.ttft_slo_sec:
                pruned_req_ids.add(req_id); continue
            if costs.tpot_slo_sec is not None and costs.G_j > 0:
                tpot_est = costs.d_j / costs.G_j
                if tpot_est > costs.tpot_slo_sec:
                    pruned_req_ids.add(req_id); continue
            if costs.gap_slo_sec is not None and costs.gap_time_sec > costs.gap_slo_sec:
                pruned_req_ids.add(req_id); continue

        if pruned_req_ids:
            logger.debug(
                "MIP presolve: pruned %d infeasible requests (SLO-blown)",
                len(pruned_req_ids),
            )

        for req_id, costs in self._costs.items():
            if req_id in pruned_req_ids:
                continue  # infeasible — skip variable creation entirely
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

        if self._use_decode_first:
            logger.debug(
                "Decode-first: Cap_dec=%d, replica_ids=%s, "
                "residual_prefill_keys=%s, residual_prefill=%s",
                self._decode_capacity,
                self._replica_ids,
                list(self._residual_prefill_capacity.keys()),
                self._residual_prefill_capacity,
            )
            # Decode-first capacity model:
            #   Σ w_j * x[j,r] ≤ Cap_r^dec       (decode slot constraint)
            #   Σ prefill_tokens_j * x[j,r] ≤ RemPreCap_r  (residual prefill)
            for r in self._replica_ids:
                decode_terms = []
                for req_id, costs in self._costs.items():
                    if (req_id, r) in x:
                        decode_terms.append(
                            _to_int(costs.w_dec) * x[(req_id, r)])
                if decode_terms:
                    model.add(
                        sum(decode_terms) <= _to_int(self._decode_capacity))

            for r in self._replica_ids:
                rem_cap = self._residual_prefill_capacity.get(r, 0)
                prefill_terms = []
                for req_id, costs in self._costs.items():
                    if (req_id, r) in x and costs.prefill_tokens > 0:
                        prefill_terms.append(
                            costs.prefill_tokens * x[(req_id, r)])
                if prefill_terms:
                    model.add(sum(prefill_terms) <= rem_cap)
        else:
            # Legacy: serial-time capacity model (fallback)
            for r in self._replica_ids:
                prefill_terms = []
                for req_id, costs in self._costs.items():
                    if (req_id, r) in x:
                        prefill_terms.append(
                            _to_int(costs.p_j) * x[(req_id, r)])
                if prefill_terms:
                    model.add(sum(prefill_terms) <= self._H_pre)

            for r in self._replica_ids:
                decode_terms = []
                for req_id, costs in self._costs.items():
                    if (req_id, r) in x:
                        coeff = _to_int(
                            costs.d_j + costs.checkpoint_overhead_sec)
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

        # Phase 2 A1: SLO-aware objective. Each request's contribution to
        # the MIP goodput objective is scaled by its slo_weight (∈ (0, 1]).
        # slo_weight < 1 means the request has been queued long enough that
        # admitting it now is unlikely to meet its TTFT SLO — the solver
        # discounts its value accordingly. When FT_SLO_AWARE_OBJECTIVE is
        # not set, slo_weight stays at 1.0 and this is identical to the
        # previous "maximize ∑ G_j · y_j" objective.
        #
        # CP-SAT needs integer coefficients, so we multiply by SLO_WEIGHT_SCALE
        # (default 1000) and floor. The ratio between req j and k is
        # preserved up to 1/SLO_WEIGHT_SCALE precision.
        SLO_WEIGHT_SCALE = 1000
        obj_terms = []
        for req_id, costs in self._costs.items():
            if req_id not in y:
                continue  # pruned by presolve
            weighted = int(round(costs.G_j * costs.slo_weight * SLO_WEIGHT_SCALE))
            if weighted <= 0:
                weighted = 1  # keep the variable in the objective
            obj_terms.append(weighted * y[req_id])

        # Variant A: FT_SOLVER_RECOVERY_PENALTY=α (default 0, back-compat).
        # Adds −α · replay_tokens · y_j to the objective. Discourages
        # admitting reqs that would cost a lot to replay on fault.
        # α is on the same unit scale as G_j·SLO_WEIGHT_SCALE, so typical
        # useful α is small (0.001 – 0.1). Larger α → more conservative.
        try:
            recovery_alpha = float(
                os.environ.get("FT_SOLVER_RECOVERY_PENALTY", "0")
            )
        except ValueError:
            recovery_alpha = 0.0
        if recovery_alpha > 0:
            penalty_terms = []
            for req_id, costs in self._costs.items():
                if req_id not in y:
                    continue  # pruned by presolve
                # Use replay_tokens as recovery cost proxy: it is the
                # number of prompt/output tokens that must be re-prefilled
                # if the replica fails. Active reqs typically have replay
                # = num_computed_tokens − num_checkpointed_tokens.
                r_j = max(costs.replay_tokens, costs.prefill_tokens)
                if r_j > 0:
                    weighted_r = int(round(recovery_alpha * r_j * SLO_WEIGHT_SCALE))
                    if weighted_r > 0:
                        penalty_terms.append(weighted_r * y[req_id])
            if penalty_terms:
                obj_terms.append(-sum(penalty_terms))
                logger.debug(
                    "FT_SOLVER_RECOVERY_PENALTY: α=%.4f applied to %d reqs",
                    recovery_alpha, len(penalty_terms),
                )

        model.maximize(sum(obj_terms))

        # Variant C: FT_SOLVER_RUNNING_CAP=N (default 0, back-compat).
        # Hard cap on per-replica admitted req count. Bounds worst-case
        # recovery work on failure to N × per-req restore cost.
        try:
            running_cap = int(os.environ.get("FT_SOLVER_RUNNING_CAP", "0"))
        except ValueError:
            running_cap = 0
        if running_cap > 0:
            for r in self._replica_ids:
                cap_terms = []
                for req_id in self._costs:
                    if (req_id, r) in x:
                        cap_terms.append(x[(req_id, r)])
                if cap_terms:
                    model.add(sum(cap_terms) <= running_cap)
            logger.debug(
                "FT_SOLVER_RUNNING_CAP: per-replica admit cap=%d", running_cap,
            )

        # FIX #2: warm-start from previous solution (solution hints).
        # For each request admitted in the previous solve that is still in
        # the current problem (not pruned by presolve), hint y=1 and
        # x=(chosen_r -> 1, others -> 0). Skip reqs pruned by presolve
        # (their variables don't exist in the current model).
        if prev_solution is not None:
            hint_count = 0
            for req_id in self._costs:
                if req_id not in prev_solution.admitted:
                    continue
                if req_id not in y:
                    continue  # pruned by presolve — no hint possible
                model.add_hint(y[req_id], 1)
                hint_count += 1
                chosen_r = prev_solution.assignments.get(req_id)
                for r in self._replica_ids:
                    if (req_id, r) in x:
                        model.add_hint(
                            x[(req_id, r)], 1 if r == chosen_r else 0
                        )
                        hint_count += 1
            if hint_count:
                logger.debug(
                    "MIP warm-start: seeded %d hints from prev solution",
                    hint_count,
                )

        solver = cp_model.CpSolver()
        # FIX #1: tighter default time cap overridable via env
        # FT_SOLVER_TIME_CAP_MS (default: keep constructor value).
        env_cap_ms = os.environ.get("FT_SOLVER_TIME_CAP_MS")
        if env_cap_ms:
            try:
                solver.parameters.max_time_in_seconds = float(env_cap_ms) / 1000.0
            except ValueError:
                solver.parameters.max_time_in_seconds = self._time_limit_sec
        else:
            solver.parameters.max_time_in_seconds = self._time_limit_sec

        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.debug("Master MIP: no feasible solution (status=%s)", status)
            return None

        solution = MasterSolution()
        solution.objective_value = solver.objective_value

        for req_id in self._costs:
            if req_id not in y:
                continue  # pruned by presolve — guaranteed not admitted
            if solver.value(y[req_id]) == 1:
                solution.admitted.add(req_id)
                for r in self._replica_ids:
                    if (req_id, r) in x and solver.value(x[(req_id, r)]) == 1:
                        solution.assignments[req_id] = r
                        break

        return solution
