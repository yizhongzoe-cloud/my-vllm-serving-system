# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy seed for Benders master warm-start (R7 optimization).

Produces a feasible-if-possible admission + routing solution via FCFS load-
balanced greedy, intended to be fed as CP-SAT `model.add_hint()` into the
master MIP. This is a *warm-start seed* — the MIP still runs and may produce
a different optimal solution. Never bypasses the solver.

Env gate: FT_SOLVER_GREEDY_SEED=1 (default OFF).

Rationale:
    Warm-start from prev epoch's MasterSolution only helps if the request set
    is similar. On high-churn workloads (e.g., W5_LongDoc with long-running
    requests completing one at a time), hit rate is ~30%. A fresh greedy seed
    built from current cost_table + replica state gives a better initial
    feasible solution on every epoch, at O(N × R) Python cost.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.benders.cost_tables import RequestCosts
    from vllm.v1.core.sched.benders.master import MasterSolution

logger = init_logger(__name__)


def compute_greedy_seed(
    cost_table: "dict[str, RequestCosts]",
    replica_ids: list[int],
    running_cap: int = 0,
    slo_weight_scale: int = 1000,
) -> "MasterSolution | None":
    """Build a feasibility-respecting greedy admission + routing solution.

    Args:
        cost_table: RequestCosts per request (same format as master MIP input).
        replica_ids: Allowed replicas for admission.
        running_cap: Per-replica admission cap (0 = unlimited).
        slo_weight_scale: Scale used to compute objective value (matches MIP).

    Returns:
        MasterSolution with greedy admission + routing, or None on empty input.
    """
    # Local import to avoid circularity at module-load time.
    from vllm.v1.core.sched.benders.master import MasterSolution

    if not cost_table or not replica_ids:
        return None

    solution = MasterSolution()

    # Step 1: pin active requests to their current replica (placement fixed).
    per_replica_count: dict[int, int] = {r: 0 for r in replica_ids}
    for req_id, costs in cost_table.items():
        if costs.is_active and costs.active_replica_id is not None:
            r0 = costs.active_replica_id
            if r0 in per_replica_count:
                solution.assignments[req_id] = r0
                solution.admitted.add(req_id)
                per_replica_count[r0] += 1

    # Step 2: collect non-active candidates that pass SLO feasibility.
    # Same filter as master MIP infeasibility constraints (ttft/tpot/gap).
    feasible: list[tuple[str, "RequestCosts"]] = []
    for req_id, costs in cost_table.items():
        if costs.is_active:
            continue
        if costs.ttft_slo_sec is not None and costs.p_j > costs.ttft_slo_sec:
            continue
        if costs.tpot_slo_sec is not None and costs.G_j > 0:
            tpot_est = costs.d_j / costs.G_j
            if tpot_est > costs.tpot_slo_sec:
                continue
        if costs.gap_slo_sec is not None and costs.gap_time_sec > costs.gap_slo_sec:
            continue
        feasible.append((req_id, costs))

    # Step 3: greedy FCFS with load-balanced replica selection + running_cap.
    # Sort by value/cost ratio (goodput per token) to prioritize high-value.
    feasible.sort(
        key=lambda it: it[1].G_j * it[1].slo_weight,
        reverse=True,
    )

    for req_id, costs in feasible:
        # Pick replica with smallest current count (load-balance).
        best_r = min(replica_ids, key=lambda r: per_replica_count[r])
        if running_cap > 0 and per_replica_count[best_r] >= running_cap:
            # All replicas full — drop this request from the seed.
            continue
        solution.assignments[req_id] = best_r
        solution.admitted.add(req_id)
        per_replica_count[best_r] += 1

    # Step 4: compute approximate objective (used by CP-SAT to gauge hint quality).
    obj = 0
    for req_id in solution.admitted:
        costs = cost_table[req_id]
        obj += int(round(costs.G_j * costs.slo_weight * slo_weight_scale))
    solution.objective_value = obj

    return solution


def is_enabled() -> bool:
    """Check FT_SOLVER_GREEDY_SEED env flag. Default OFF."""
    return os.environ.get("FT_SOLVER_GREEDY_SEED", "0") == "1"
