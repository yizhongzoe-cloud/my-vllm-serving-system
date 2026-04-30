# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import math

from vllm.v1.request import Request, RequestStatus


def compute_slo_budgets(
    request: Request, now_sec: float
) -> dict[str, float]:
    """Compute remaining SLO budgets (ms) for a request at time `now_sec`.

    Returns dict with keys: ttft_ms, tpot_ms, gap_ms, min_ms, stage.
    Negative budget means already in violation. Inapplicable budgets are inf.

    Stage:
      "waiting"  — not yet started (no GPU work done; ttft budget applies)
      "prefill"  — has GPU work but no first output token yet
      "decode"   — already produced first output token (tpot applies)
      "recovery" — fault-affected, post-fault, awaiting first new token (gap applies)

    Used by SLO-aware scheduler. M1 logs only; M2 sorts waiting queue by min_ms;
    M3 preempts running reqs whose min_ms is much looser than top waiting's.
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
    # NOTE: M1 uses a coarse approximation (avg TPOT since arrival); M2/M3 may
    # add per-step last_token_time tracking for tighter bounds.
    if request.num_output_tokens > 0 and request.tpot_slo_ms is not None:
        decode_elapsed_ms = max(0.0, (now_sec - request.arrival_time) * 1000.0)
        avg_tpot_ms = decode_elapsed_ms / max(1, request.num_output_tokens)
        # Budget = how much slack before next-token must come
        tpot_ms = request.tpot_slo_ms - avg_tpot_ms
    else:
        tpot_ms = math.inf

    # gap budget: only for fault-rerouted reqs (recovery path)
    if (
        getattr(request, "is_rerouted", False)
        and request.failure_gap_slo_ms is not None
        and request.num_output_tokens == 0  # haven't yet first-new-token after fault
    ):
        # NOTE: M1 doesn't track fault_time per req. Approximate as
        # (now - max(arrival_time, last_admitted_time)). Refine in M2 with
        # explicit fault_time field set by ft_client when rerouting.
        elapsed_since_reroute_ms = max(
            0.0, (now_sec - request.arrival_time) * 1000.0
        )
        gap_ms = request.failure_gap_slo_ms - elapsed_since_reroute_ms
    else:
        gap_ms = math.inf

    # Determine stage
    if request.num_output_tokens > 0:
        stage = "decode"
    elif getattr(request, "is_rerouted", False):
        stage = "recovery"
    elif request.num_computed_tokens > 0:
        stage = "prefill"
    else:
        stage = "waiting"

    min_ms = min(ttft_ms, tpot_ms, gap_ms)

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "gap_ms": gap_ms,
        "min_ms": min_ms,
        "stage": stage,
    }


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
