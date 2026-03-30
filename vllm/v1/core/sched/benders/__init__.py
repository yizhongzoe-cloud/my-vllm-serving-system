# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benders-style robust solver for fault-tolerant LLM serving.

This package implements the online periodic Benders decomposition
described in idea_online_periodic.md (§10.7). It replaces the greedy
admission heuristic with a joint optimization of admission and routing,
validated against all single-GPU failure scenarios. Checkpointing is
modeled as a fixed runtime-side input, not a solver decision.
"""

from vllm.v1.core.sched.benders.cost_tables import CostTableBuilder, RequestCosts
from vllm.v1.core.sched.benders.cuts import BendersCut, make_cut
from vllm.v1.core.sched.benders.master import MasterProblem, MasterSolution
from vllm.v1.core.sched.benders.recovery_checker import (
    InfeasibilityCertificate,
    RecoveryChecker,
    RecoveryPlan,
)
from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop, BendersSolveResult

__all__ = [
    "BendersCut",
    "BendersSolveLoop",
    "BendersSolveResult",
    "CostTableBuilder",
    "InfeasibilityCertificate",
    "MasterProblem",
    "MasterSolution",
    "RecoveryChecker",
    "RecoveryPlan",
    "RequestCosts",
    "make_cut",
]
