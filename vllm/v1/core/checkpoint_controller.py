# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online checkpoint controller for fault-tolerant serving.

The controller now separates two concerns:

1. Fixed baselines can use a fixed block cadence (`fixed_checkpoint_blocks > 0`)
   or the deprecated legacy fixed-level cadence (`fixed_checkpoint_level >= 0`).
2. The default adaptive policy is fully online and local:
   - only re-evaluate when a new full KV block becomes stable;
   - evaluate the *entire unpublished stable prefix*;
   - publish it when
       Δreplay_saved > Δload + λ * Δcheckpoint_overhead.

This keeps checkpointing runtime-local while aligning the trigger with
incremental block-granular checkpoint publication.
"""

import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for runtime checkpoint policy.

    Attributes:
        level1_progress: Coarse progress threshold for reporting stage 1.
        level2_progress: Coarse progress threshold for reporting stage 2.
        level1_interval_steps: Decode steps between checkpoints at level 1.
        level2_interval_steps: Decode steps between checkpoints at level 2.
        level1_interval_sec: Minimum seconds between checkpoints at level 1.
        level2_interval_sec: Minimum seconds between checkpoints at level 2.
    """

    level1_progress: float = 0.25
    level2_progress: float = 0.60
    level1_interval_steps: int = 64
    level2_interval_steps: int = 16
    level1_interval_sec: float = 2.0
    level2_interval_sec: float = 0.5


@dataclass(frozen=True)
class OnlineCheckpointEstimate:
    """Pure estimate of the current online publication decision."""

    stable_full_tokens: int
    published_tokens: int
    unpublished_tokens: int
    kv_bytes_per_token: int
    unpublished_bytes: int
    replay_saved_sec: float
    load_cost_sec: float
    checkpoint_cost_sec: float
    should_publish: bool


def estimate_online_checkpoint_publication(
    *,
    num_computed_tokens: int,
    num_checkpointed_tokens: int,
    checkpoint_size_bytes: int,
    block_size: int,
    replay_throughput_tokens_per_sec: float,
    load_bandwidth_bytes_per_sec: float,
    checkpoint_bandwidth_bytes_per_sec: float,
    checkpoint_lambda: float,
    default_kv_bytes_per_token: int,
) -> OnlineCheckpointEstimate:
    """Estimate whether the current unpublished stable suffix should publish.

    This helper is intentionally pure and side-effect free so both the
    runtime controller and the solver cost model can share the exact same
    economic trigger logic.
    """
    stable_full_tokens = max(0, (num_computed_tokens // max(1, block_size))
                             * max(1, block_size))
    published_tokens = min(max(0, num_checkpointed_tokens), stable_full_tokens)

    if published_tokens > 0 and checkpoint_size_bytes > 0:
        kv_bytes_per_token = max(1, checkpoint_size_bytes // published_tokens)
    else:
        kv_bytes_per_token = max(1, default_kv_bytes_per_token)

    unpublished_tokens = max(0, stable_full_tokens - published_tokens)
    unpublished_bytes = unpublished_tokens * kv_bytes_per_token

    replay_saved_sec = (
        unpublished_tokens / replay_throughput_tokens_per_sec
        if replay_throughput_tokens_per_sec > 0
        else 0.0
    )
    load_cost_sec = (
        unpublished_bytes / load_bandwidth_bytes_per_sec
        if load_bandwidth_bytes_per_sec > 0
        else 0.0
    )
    checkpoint_cost_sec = (
        unpublished_bytes / checkpoint_bandwidth_bytes_per_sec
        if checkpoint_bandwidth_bytes_per_sec > 0
        else 0.0
    )

    should_publish = (
        unpublished_tokens > 0
        and replay_throughput_tokens_per_sec > 0
        and load_bandwidth_bytes_per_sec > 0
        and checkpoint_bandwidth_bytes_per_sec > 0
        and replay_saved_sec > (
            load_cost_sec + checkpoint_lambda * checkpoint_cost_sec)
    )

    return OnlineCheckpointEstimate(
        stable_full_tokens=stable_full_tokens,
        published_tokens=published_tokens,
        unpublished_tokens=unpublished_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        unpublished_bytes=unpublished_bytes,
        replay_saved_sec=replay_saved_sec,
        load_cost_sec=load_cost_sec,
        checkpoint_cost_sec=checkpoint_cost_sec,
        should_publish=should_publish,
    )


class CheckpointController:
    """Controls runtime checkpoint decisions for active requests.

    The controller does NOT perform checkpoint I/O itself. It only decides
    when a request should publish a new incremental checkpoint.
    """

    DEFAULT_KV_BYTES_PER_TOKEN = 8192

    def __init__(
        self,
        config: CheckpointConfig | None = None,
        fixed_level: int = -1,
        fixed_blocks: int = 0,
        block_size: int = 1,
        replay_throughput_tokens_per_sec: float = 0.0,
        load_bandwidth_bytes_per_sec: float = 0.0,
        checkpoint_bandwidth_bytes_per_sec: float = 0.0,
        checkpoint_lambda: float = 1.0,
        default_kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
    ) -> None:
        self.config = config or CheckpointConfig()
        self._fixed_level = fixed_level
        self._fixed_blocks = max(0, fixed_blocks)
        self._block_size = max(1, block_size)
        self._replay_throughput = replay_throughput_tokens_per_sec
        self._load_bandwidth = load_bandwidth_bytes_per_sec
        self._checkpoint_bandwidth = checkpoint_bandwidth_bytes_per_sec
        self._checkpoint_lambda = checkpoint_lambda
        self._default_kv_bytes_per_token = max(1, default_kv_bytes_per_token)
        self._economic_policy_available = (
            self._replay_throughput > 0
            and self._load_bandwidth > 0
            and self._checkpoint_bandwidth > 0
        )
        self._warned_missing_economic_inputs = False

        self._last_checkpoint_step: dict[str, int] = {}
        self._last_evaluated_stable_tokens: dict[str, int] = {}

    def _get_reporting_level(self, request: Request) -> int:
        """Return a coarse runtime checkpoint stage for telemetry/costs."""
        progress = request.generation_progress
        if progress >= self.config.level2_progress:
            return 2
        if progress >= self.config.level1_progress:
            return 1
        return 0

    def _get_stable_full_tokens(self, request: Request) -> int:
        return max(0, (request.num_computed_tokens // self._block_size) * self._block_size)

    def _estimate_online_publication(
        self,
        request: Request,
    ) -> OnlineCheckpointEstimate:
        return estimate_online_checkpoint_publication(
            num_computed_tokens=request.num_computed_tokens,
            num_checkpointed_tokens=request.num_checkpointed_tokens,
            checkpoint_size_bytes=request.last_checkpoint_size_bytes,
            block_size=self._block_size,
            replay_throughput_tokens_per_sec=self._replay_throughput,
            load_bandwidth_bytes_per_sec=self._load_bandwidth,
            checkpoint_bandwidth_bytes_per_sec=self._checkpoint_bandwidth,
            checkpoint_lambda=self._checkpoint_lambda,
            default_kv_bytes_per_token=self._default_kv_bytes_per_token,
        )

    def _should_checkpoint_by_level(
        self,
        request: Request,
        level: int,
    ) -> bool:
        if level == 0:
            return False

        current_step = request.num_output_tokens
        last_step = self._last_checkpoint_step.get(request.request_id, 0)
        interval = (
            self.config.level1_interval_steps
            if level == 1
            else self.config.level2_interval_steps
        )
        if current_step - last_step < interval:
            return False

        now = time.time()
        min_interval = (
            self.config.level1_interval_sec
            if level == 1
            else self.config.level2_interval_sec
        )
        if (
            request.last_checkpoint_time is not None
            and now - request.last_checkpoint_time < min_interval
        ):
            return False

        return True

    def _should_checkpoint_by_fixed_blocks(self, request: Request) -> bool:
        stable_full_tokens = self._get_stable_full_tokens(request)
        published_tokens = min(request.num_checkpointed_tokens, stable_full_tokens)
        if stable_full_tokens <= published_tokens:
            self._last_evaluated_stable_tokens[request.request_id] = stable_full_tokens
            return False

        last_evaluated_tokens = self._last_evaluated_stable_tokens.get(
            request.request_id,
            published_tokens,
        )
        if stable_full_tokens <= last_evaluated_tokens:
            return False

        self._last_evaluated_stable_tokens[request.request_id] = stable_full_tokens
        required_tokens = self._fixed_blocks * self._block_size
        return (stable_full_tokens - published_tokens) >= required_tokens

    def _should_checkpoint_by_economic_policy(self, request: Request) -> bool:
        estimate = self._estimate_online_publication(request)
        stable_full_tokens = estimate.stable_full_tokens
        published_tokens = request.num_checkpointed_tokens
        if stable_full_tokens <= published_tokens:
            self._last_evaluated_stable_tokens[request.request_id] = stable_full_tokens
            return False

        last_evaluated_tokens = self._last_evaluated_stable_tokens.get(
            request.request_id,
            published_tokens,
        )
        if stable_full_tokens <= last_evaluated_tokens:
            return False
        self._last_evaluated_stable_tokens[request.request_id] = stable_full_tokens
        return estimate.should_publish

    def get_checkpoint_level(self, request: Request) -> int:
        if self._fixed_blocks > 0:
            return 2 if self._fixed_blocks == 1 else 1
        if self._fixed_level >= 0:
            return self._fixed_level
        return self._get_reporting_level(request)

    def should_checkpoint(self, request: Request) -> bool:
        """Decide whether this request should publish a checkpoint now."""
        level = self.get_checkpoint_level(request)
        request.checkpoint_level = level

        if self._fixed_blocks > 0:
            return self._should_checkpoint_by_fixed_blocks(request)

        if self._fixed_level >= 0:
            return self._should_checkpoint_by_level(request, level)

        if self._economic_policy_available:
            return self._should_checkpoint_by_economic_policy(request)

        if not self._warned_missing_economic_inputs:
            logger.warning(
                "Online checkpoint policy missing throughput/bandwidth inputs; "
                "falling back to legacy level-based adaptive policy."
            )
            self._warned_missing_economic_inputs = True
        return self._should_checkpoint_by_level(request, level)

    def record_checkpoint(self, request: Request) -> None:
        """Record that a checkpoint was just successfully published."""
        now = time.time()
        stable_full_tokens = self._get_stable_full_tokens(request)
        request.last_checkpoint_time = now
        request.num_checkpointed_tokens = max(
            request.num_checkpointed_tokens,
            stable_full_tokens,
        )
        self._last_checkpoint_step[request.request_id] = (
            request.num_output_tokens
        )
        self._last_evaluated_stable_tokens[request.request_id] = stable_full_tokens

    def get_requests_to_checkpoint(
        self, requests: list[Request]
    ) -> list[Request]:
        """From a list of running requests, return those that need checkpointing.

        This is the main entry point called by the scheduler each step.

        Args:
            requests: List of currently running requests.

        Returns:
            Subset of requests that should be checkpointed this step.
        """
        to_checkpoint = []
        for req in requests:
            if self.should_checkpoint(req):
                to_checkpoint.append(req)
        return to_checkpoint

    def remove_request(self, request_id: str) -> None:
        """Clean up tracking state when a request finishes or is aborted."""
        self._last_checkpoint_step.pop(request_id, None)
        self._last_evaluated_stable_tokens.pop(request_id, None)

    def estimate_checkpoint_overhead(
        self,
        request: Request,
        checkpoint_size_bytes: int = 0,
        gpu_to_host_bandwidth: float = 0.0,
    ) -> float:
        """Estimate the one-shot publish cost for the current stable suffix."""
        if checkpoint_size_bytes > 0 and gpu_to_host_bandwidth > 0:
            return checkpoint_size_bytes / gpu_to_host_bandwidth

        estimate = self._estimate_online_publication(request)
        if not estimate.should_publish:
            return 0.0
        return estimate.checkpoint_cost_sec

    def estimate_recovery_cost(
        self,
        request: Request,
        checkpoint_size_bytes: int = 0,
        load_bandwidth_bytes_per_sec: float = 0.0,
        replay_tokens_per_sec: float = 0.0,
        detection_time_sec: float = 0.0,
        decode_throughput: float = 0.0,
        replay_tokens_override: int | None = None,
    ) -> float:
        """Estimate total failover time for a request given its current state.

        Implements the failover-gap formula from the paper:
            T^{det} + S^{ckpt}/B^{ld} + U_j/C^{rep} + 1/C^{dec}

        Args:
            request: The request to estimate recovery cost for.
            checkpoint_size_bytes: S^{ckpt} — actual checkpoint size in bytes.
            load_bandwidth_bytes_per_sec: B^{ld} — Host→GPU bandwidth.
            replay_tokens_per_sec: C^{rep} — Prefill throughput for replay.
            detection_time_sec: T^{det} — failure detection overhead.
            decode_throughput: C^{dec} — decode tokens per second.
            replay_tokens_override: If provided, use this instead of
                request.get_uncovered_tokens(). Useful at admission time
                when num_computed_tokens is 0 but prompt replay is needed.

        Returns:
            Estimated failover time in seconds.
        """
        # Checkpoint restore time: S^{ckpt} / B^{ld}
        checkpoint_restore_time = 0.0
        if checkpoint_size_bytes > 0 and load_bandwidth_bytes_per_sec > 0:
            checkpoint_restore_time = (
                checkpoint_size_bytes / load_bandwidth_bytes_per_sec
            )

        # Replay time: U_j / C^{rep}
        uncovered = (
            replay_tokens_override
            if replay_tokens_override is not None
            else request.get_uncovered_tokens()
        )
        replay_time = (
            uncovered / replay_tokens_per_sec
            if replay_tokens_per_sec > 0
            else 0.0  # Not profiled yet; skip this term.
        )

        # Resume time: 1 / C^{dec}
        resume_time = (
            1.0 / decode_throughput
            if decode_throughput > 0
            else 0.0  # Not profiled yet; skip this term.
        )

        total = detection_time_sec + checkpoint_restore_time + replay_time + resume_time
        return total
