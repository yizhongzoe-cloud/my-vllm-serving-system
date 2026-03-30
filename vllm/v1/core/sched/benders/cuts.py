# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benders cut generation for the robust FT solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.core.sched.benders.master import MasterSolution
from vllm.v1.core.sched.benders.recovery_checker import InfeasibilityCertificate


@dataclass
class BendersCut:
    """A logic-based Benders cut on request→replica assignments."""

    involved: list[tuple[str, int]] = field(default_factory=list)
    cut_type: str = "no_good"


def make_no_good_cut(
    solution: MasterSolution,
    certificate: InfeasibilityCertificate,
) -> BendersCut:
    involved = [
        (req_id, replica_id)
        for req_id, replica_id in solution.assignments.items()
    ]
    return BendersCut(involved=involved, cut_type="no_good")


def make_pool_overload_cut(
    solution: MasterSolution,
    certificate: InfeasibilityCertificate,
) -> BendersCut:
    q_ids = set(certificate.overloaded_subset)
    involved = [
        (req_id, replica_id)
        for req_id, replica_id in solution.assignments.items()
        if req_id in q_ids
    ]
    if not involved:
        return make_no_good_cut(solution, certificate)
    return BendersCut(involved=involved, cut_type="pool_overload")


def make_cut(
    solution: MasterSolution,
    certificate: InfeasibilityCertificate,
) -> BendersCut:
    if certificate.cert_type == "PoolOverload" and certificate.overloaded_subset:
        return make_pool_overload_cut(solution, certificate)
    return make_no_good_cut(solution, certificate)
