# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math

from vllm.v1.request import Request, RequestStatus


def compute_slo_budgets(
    request: Request, now_sec: float
) -> dict[str, float]:
    """Compute remaining SLO budgets (ms) for a request at time `now_sec`.

    Returns dict with keys: ttft_ms, tpot_ms, min_ms, stage.
    Negative budget means already in violation. Inapplicable budgets are inf.

    Stage:
      "waiting"  — not yet started (no GPU work done; ttft budget applies)
      "prefill"  — has GPU work but no first output token yet
      "decode"   — already produced first output token (tpot applies)

    Used by SLO priority preempt picker — preempt running req with
    largest min_ms (most slack) when waiting head's min_ms is most
    negative (most-violating).
    """
    # ttft budget: from arrival until first output token expected
    if request.num_output_tokens == 0:
        elapsed_ms = max(0.0, (now_sec - request.arrival_time) * 1000.0)
        if request.ttft_slo_ms is not None:
            ttft_ms = request.ttft_slo_ms - elapsed_ms
        else:
            ttft_ms = math.inf
    else:
        ttft_ms = math.inf  # already passed first token

    # tpot budget: average TPOT across decoded tokens vs SLO budget
    if request.num_output_tokens > 0 and request.tpot_slo_ms is not None:
        decode_elapsed_ms = max(0.0, (now_sec - request.arrival_time) * 1000.0)
        avg_tpot_ms = decode_elapsed_ms / max(1, request.num_output_tokens)
        # Budget = how much slack before next-token must come
        tpot_ms = request.tpot_slo_ms - avg_tpot_ms
    else:
        tpot_ms = math.inf

    # Determine stage
    if request.num_output_tokens > 0:
        stage = "decode"
    elif request.num_computed_tokens > 0:
        stage = "prefill"
    else:
        stage = "waiting"

    min_ms = min(ttft_ms, tpot_ms)

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "min_ms": min_ms,
        "stage": stage,
    }


def compute_replay_cost(request: Request, now_sec: float) -> float:
    """Estimate the replay cost (ms) of preempting `request` now.

    Host KV checkpoints lag behind GPU progress (saved per full block,
    asynchronously). After preempt + reload, tokens between
    num_checkpointed_tokens and num_computed_tokens must be re-run.

        replay_tokens = num_computed_tokens − num_checkpointed_tokens
        per_token_ms = elapsed_since_arrival_ms / num_computed_tokens
        replay_cost_ms = replay_tokens × per_token_ms

    Using num_computed_tokens (prompt + decode) keeps the formula
    meaningful for both prefill-stage and decode-stage victims; both
    operands above are in the same units. A victim with checkpoint
    fully caught up returns 0.

    The picker uses this to inflate the hysteresis gap so reqs that
    have done more uncheckpointed work are harder to preempt than
    fresh reqs with the same slack.
    """
    replay_tokens = max(
        0, request.num_computed_tokens - request.num_checkpointed_tokens
    )
    if replay_tokens == 0:
        return 0.0
    elapsed_ms = max(0.0, (now_sec - request.arrival_time) * 1000.0)
    per_token_ms = elapsed_ms / max(1, request.num_computed_tokens)
    return replay_tokens * per_token_ms


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
