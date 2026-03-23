# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        resumable: bool = False,
        ttft_slo_ms: float | None = None,
        e2e_latency_slo_ms: float | None = None,
        tpot_slo_ms: float | None = None,
        failure_gap_slo_ms: float | None = None,
        expected_output_len: int | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        # SLO-related fields
        self.ttft_slo_ms = ttft_slo_ms
        self.e2e_latency_slo_ms = e2e_latency_slo_ms
        self.tpot_slo_ms = tpot_slo_ms
        self.failure_gap_slo_ms = failure_gap_slo_ms
        self.expected_output_len = expected_output_len

        # Calculate deadlines from SLO requirements
        self.ttft_deadline: float | None = None
        if ttft_slo_ms is not None:
            self.ttft_deadline = self.arrival_time + (ttft_slo_ms / 1000.0)

        self.e2e_deadline: float | None = None
        if e2e_latency_slo_ms is not None:
            self.e2e_deadline = self.arrival_time + (e2e_latency_slo_ms / 1000.0)

        # TPOT deadline: estimated completion time based on TPOT bound.
        # D_j^{tpot} * G_j gives total decode budget; add to arrival + prefill.
        self.tpot_deadline: float | None = None
        if tpot_slo_ms is not None and expected_output_len is not None:
            total_decode_budget_sec = (tpot_slo_ms / 1000.0) * expected_output_len
            self.tpot_deadline = self.arrival_time + total_decode_budget_sec

        # Cached sort key for SLO-aware scheduling (avoids time.time()
        # calls during heap comparisons; updated by update_sort_key()).
        self._cached_slack: float = float('inf')

        # Fault-tolerance fields
        self.checkpoint_level: int = 0  # 0=none, 1=low freq, 2=high freq
        self.assigned_replica_id: int | None = None
        self.num_checkpointed_tokens: int = 0
        self.last_checkpoint_time: float | None = None

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        req = cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            resumable=request.resumable,
            ttft_slo_ms=request.ttft_slo_ms,
            e2e_latency_slo_ms=request.e2e_latency_slo_ms,
            tpot_slo_ms=request.tpot_slo_ms,
            failure_gap_slo_ms=request.failure_gap_slo_ms,
            expected_output_len=request.expected_output_len,
        )
        # FT: propagate checkpoint info for failover restore.
        req.num_checkpointed_tokens = getattr(
            request, "num_checkpointed_tokens", 0
        )
        return req

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def has_slo(self) -> bool:
        """Check if the request has any SLO requirement."""
        return (self.ttft_slo_ms is not None
                or self.e2e_latency_slo_ms is not None
                or self.tpot_slo_ms is not None)

    def has_failure_gap_slo(self) -> bool:
        """Check if the request has a failure-gap SLO."""
        return self.failure_gap_slo_ms is not None

    @property
    def prompt_len(self) -> int:
        """P_j: prompt length in tokens."""
        return self.num_prompt_tokens

    @property
    def generation_len(self) -> int:
        """G_j: expected output length.

        Uses the user-provided expected_output_len if available.
        Otherwise estimates as half of max_tokens (a heuristic that
        avoids the extreme overestimate of using the full max_tokens
        upper bound, which would distort capacity accounting).
        """
        if self.expected_output_len is not None:
            return self.expected_output_len
        return max(1, self.max_tokens // 2)

    @property
    def generation_progress(self) -> float:
        """Fraction of expected output tokens generated so far (0.0 to 1.0)."""
        gen_len = self.generation_len
        if gen_len <= 0:
            return 0.0
        return min(self.num_output_tokens / gen_len, 1.0)

    def get_uncovered_tokens(self) -> int:
        """Tokens generated since last checkpoint (would need replay)."""
        return max(0, self.num_computed_tokens - self.num_checkpointed_tokens)

    def get_ttft_slack(self, current_time: float) -> float:
        """
        Calculate TTFT slack time.

        Args:
            current_time: Current time (time.time())

        Returns:
            Slack time in seconds. Returns float('inf') if no TTFT SLO is set.
            Negative value means SLO is already violated.
        """
        if self.ttft_deadline is None:
            return float('inf')
        return self.ttft_deadline - current_time

    def get_e2e_slack(self, current_time: float) -> float:
        """
        Calculate E2E slack time.

        Args:
            current_time: Current time (time.time())

        Returns:
            Slack time in seconds. Returns float('inf') if no E2E SLO is set.
            Negative value means SLO is already violated.
        """
        if self.e2e_deadline is None:
            return float('inf')
        return self.e2e_deadline - current_time

    def get_tpot_slack(self, current_time: float) -> float:
        """
        Calculate TPOT slack time.

        Returns:
            Slack time in seconds. Returns float('inf') if no TPOT SLO is set.
        """
        if self.tpot_deadline is None:
            return float('inf')
        return self.tpot_deadline - current_time

    def get_effective_slack(self, current_time: float) -> float:
        """
        Get the effective slack time for scheduling.

        WAITING requests use TTFT slack, RUNNING requests use
        min(E2E slack, TPOT slack) to respect both constraints.
        """
        if self.status == RequestStatus.RUNNING:
            return min(self.get_e2e_slack(current_time),
                       self.get_tpot_slack(current_time))
        else:
            return self.get_ttft_slack(current_time)

    def update_sort_key(self, current_time: float) -> None:
        """Pre-compute the sort key for SLO-aware scheduling.

        Called once per reheapify cycle (not per comparison) to avoid
        repeated time.time() syscalls during heap operations.
        """
        if self.has_slo():
            self._cached_slack = self.get_effective_slack(current_time)
        else:
            self._cached_slack = float('inf')

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on SLO slack, priority, arrival time,
        and request ID. Used in priority/SLO-aware scheduling.

        For SLO-aware scheduling:
        - Requests with SLO are prioritized over those without
        - Among requests with SLO, smaller slack time (more urgent) comes first
        - WAITING requests use TTFT slack, RUNNING requests use E2E slack

        Uses pre-computed _cached_slack (set by update_sort_key or
        SLOAwareRequestQueue._maybe_reheapify) to avoid calling
        time.time() on every comparison during heap operations.
        """
        self_has_slo = self.has_slo()
        other_has_slo = other.has_slo()

        if self_has_slo and other_has_slo:
            self_slack = getattr(self, '_cached_slack', float('inf'))
            other_slack = getattr(other, '_cached_slack', float('inf'))
            if self_slack != other_slack:
                return self_slack < other_slack  # Smaller slack = more urgent
        elif self_has_slo and not other_has_slo:
            return True
        elif not self_has_slo and other_has_slo:
            return False

        # Fall back to original priority-based comparison
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
}
