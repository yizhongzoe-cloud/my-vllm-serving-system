# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math
import os

from vllm.v1.request import Request, RequestStatus

# Fixed per-fire switch cost in ms — the wall-clock spent on
# preempt state machine + cross-engine HTTP forward + V3 reload
# of the host checkpoint into the receiving engine + admit on
# that engine. Paid in full every time the picker fires,
# independent of how many tokens need replay. The picker rolls
# this into replay_cost so the fire rule (slack_gap > δ +
# replay_cost) naturally reflects the full cost of kicking a
# victim, not just the per-token re-decode piece. Override via
# FT_PICKER_SWITCH_COST_MS for sweeps.
_SWITCH_COST_MS = float(os.environ.get("FT_PICKER_SWITCH_COST_MS", "1000"))


def compute_slo_budgets(
    request: Request, now_sec: float
) -> dict[str, float]:
    """Compute the request's deadline *laxity* (ms) at time `now_sec`.

    Ferry models the SLO as a single per-request completion deadline,
    parameterized by two per-request knobs supplied by the client:

        a = ttft_slo_ms   — one-time startup allowance (queue + prefill +
                            first token). The CLIENT computes this value;
                            it may be prompt-aware and/or SLO-tier-aware
                            (e.g. tight tier = 0.73*prompt_tokens, loose
                            tier = 2x that). The engine treats ttft_slo_ms
                            as the literal `a` budget in ms — it does NOT
                            re-derive it from prompt length. This keeps the
                            engine policy-agnostic: all tier/prompt logic
                            lives in the workload/client.
        b = tpot_slo_ms   — per-output-token allowance

    A request that produces N output tokens must complete by

        deadline = arrival + a + N * b.

    Laxity is the running form of that deadline check:

        laxity = a + produced_tokens * b - elapsed_since_arrival

    Interpretation:
      laxity >= 0  → on or ahead of the deadline pace.
      laxity  < 0  → behind; will miss unless it speeds up.
    A preempt-and-reload pauses the request for P ms, which reduces its
    laxity by exactly P. So a running victim can absorb a preempt iff
    laxity > pause_cost (see compute_replay_cost). This puts waiting and
    decoding requests on ONE comparable axis (ms): a waiting request has
    produced=0, so laxity = a - elapsed (its TTFT slack); a decoding
    request accrues b of budget per token it has produced. It also avoids
    the earlier bug of folding TTFT/queue time into a per-token average.

    Returns dict keys:
      laxity_ms          — the unified slack (smaller = more urgent).
      deadline_budget_ms — a + produced*b (the budget accrued so far).
      stage              — "waiting" | "prefill" | "decode".
      min_ms             — back-compat alias for laxity_ms (picker).
    No-SLO requests get laxity = +inf (never urgent, freely pausable).
    """
    # SLO present iff the client attached either budget. `a` is the
    # client-supplied startup allowance (ttft_slo_ms, taken literally as
    # the ms budget), `b` the per-output-token allowance (tpot_slo_ms).
    has_slo = (request.ttft_slo_ms is not None
               or request.tpot_slo_ms is not None)
    if not has_slo:
        laxity_ms = math.inf
        budget_ms = math.inf
    else:
        a = request.ttft_slo_ms if request.ttft_slo_ms is not None else 0.0
        b = request.tpot_slo_ms if request.tpot_slo_ms is not None else 0.0
        elapsed_ms = max(0.0, (now_sec - request.arrival_time) * 1000.0)
        budget_ms = a + request.num_output_tokens * b
        laxity_ms = budget_ms - elapsed_ms

    # Determine stage (diagnostic only).
    if request.num_output_tokens > 0:
        stage = "decode"
    elif request.num_computed_tokens > 0:
        stage = "prefill"
    else:
        stage = "waiting"

    return {
        "laxity_ms": laxity_ms,
        "deadline_budget_ms": budget_ms,
        "stage": stage,
        "min_ms": laxity_ms,  # back-compat alias
    }


def compute_replay_cost(request: Request, now_sec: float) -> float:
    """Estimate the total cost (ms) of preempting `request` now.

    Two components:

    1. **switch_cost** (fixed, per-fire): wall-clock spent on the
       preempt state machine + cross-engine HTTP forward + V3 reload
       of the host checkpoint into the receiving engine + admit on
       that engine. Paid in full every fire regardless of victim
       progress. Module-level constant `_SWITCH_COST_MS`, overridable
       via `FT_PICKER_SWITCH_COST_MS` env var (default 1000ms).

    2. **variable replay** (per-token): host KV checkpoints lag
       behind GPU progress (saved per full block, asynchronously).
       After preempt + reload, tokens between
       num_checkpointed_tokens and num_computed_tokens must be re-run.

           replay_tokens = num_computed_tokens − num_checkpointed_tokens
           per_token_ms = elapsed_since_arrival_ms / num_computed_tokens
           variable_ms = replay_tokens × per_token_ms

       Using num_computed_tokens (prompt + decode) keeps the formula
       meaningful for both prefill-stage and decode-stage victims.

    Returns switch_cost + variable_ms. The picker rule
    (slack_gap > δ + replay_cost) then reflects the FULL cost of
    kicking, not just the per-token recompute piece — which alone
    underestimates fires by ~1s on long-context workloads and made
    the picker trade net-negative on RULER.
    """
    elapsed_ms = max(0.0, (now_sec - request.arrival_time) * 1000.0)
    replay_tokens = max(
        0, request.num_computed_tokens - request.num_checkpointed_tokens
    )
    if replay_tokens == 0 or request.num_computed_tokens == 0:
        variable_ms = 0.0
    else:
        per_token_ms = elapsed_ms / max(1, request.num_computed_tokens)
        variable_ms = replay_tokens * per_token_ms
    return _SWITCH_COST_MS + variable_ms


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False
