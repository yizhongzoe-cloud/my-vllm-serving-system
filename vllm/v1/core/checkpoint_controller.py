# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adaptive Checkpoint Controller for Fault-Tolerant Serving.

Implements the core insight from the paper: checkpoint frequency should
increase as a request generates more tokens. Early in generation there is
little accumulated state to lose, but as generation progresses the cost
of losing KV cache grows, warranting stronger checkpointing.

Checkpoint levels:
    0 — No checkpointing. Used when generation has just started.
    1 — Low-frequency checkpointing. Used during mid-generation.
    2 — High-frequency checkpointing. Used for long-running requests
        that have accumulated significant KV cache state.
"""

import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for adaptive checkpoint policy.

    Attributes:
        level1_progress: Generation progress threshold to enter level 1.
        level2_progress: Generation progress threshold to enter level 2.
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


class CheckpointController:
    """Controls adaptive checkpoint decisions for all active requests.

    For each request, determines:
    1. The current checkpoint level (0, 1, or 2) based on generation progress.
    2. Whether a checkpoint should be triggered at the current step.

    The controller does NOT perform the actual checkpointing — it only
    makes the decision. The actual GPU→CPU copy is done by KVCheckpointPool.
    """

    def __init__(self, config: CheckpointConfig | None = None) -> None:
        self.config = config or CheckpointConfig()
        # Track per-request state: last checkpointed step count.
        self._last_checkpoint_step: dict[str, int] = {}

    def get_checkpoint_level(self, request: Request) -> int:
        """Determine the checkpoint level for a request based on progress.

        The level monotonically increases as the request generates more tokens:
            progress < level1_progress → level 0 (no checkpointing)
            level1_progress ≤ progress < level2_progress → level 1
            progress ≥ level2_progress → level 2

        Args:
            request: The request to evaluate.

        Returns:
            Checkpoint level: 0, 1, or 2.
        """
        progress = request.generation_progress
        if progress >= self.config.level2_progress:
            return 2
        elif progress >= self.config.level1_progress:
            return 1
        else:
            return 0

    def should_checkpoint(self, request: Request) -> bool:
        """Decide whether to checkpoint this request right now.

        Considers:
        - The current checkpoint level (0 = never checkpoint).
        - Steps since last checkpoint vs the level's interval.
        - Time since last checkpoint vs the level's minimum interval.

        Args:
            request: The request to evaluate.

        Returns:
            True if the request should be checkpointed now.
        """
        level = self.get_checkpoint_level(request)
        request.checkpoint_level = level

        if level == 0:
            return False

        # Check step-based interval.
        current_step = request.num_output_tokens
        last_step = self._last_checkpoint_step.get(request.request_id, 0)
        interval = (
            self.config.level1_interval_steps
            if level == 1
            else self.config.level2_interval_steps
        )
        if current_step - last_step < interval:
            return False

        # Check time-based interval.
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

    def record_checkpoint(self, request: Request) -> None:
        """Record that a checkpoint was just performed for this request.

        Updates both the controller's tracking state and the request's
        checkpoint metadata.

        Args:
            request: The request that was just checkpointed.
        """
        now = time.time()
        request.last_checkpoint_time = now
        request.num_checkpointed_tokens = request.num_computed_tokens
        self._last_checkpoint_step[request.request_id] = (
            request.num_output_tokens
        )

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

    def estimate_checkpoint_overhead(
        self,
        request: Request,
        checkpoint_size_bytes: int = 0,
        gpu_to_host_bandwidth: float = 0.0,
    ) -> float:
        """Estimate the steady-state overhead of checkpointing per step.

        Stronger checkpointing (higher level) incurs more frequent
        GPU→CPU copies, which competes with inference for memory bandwidth.

        Args:
            request: The request to evaluate.
            checkpoint_size_bytes: Size of one checkpoint in bytes.
            gpu_to_host_bandwidth: GPU→Host copy bandwidth (bytes/sec).

        Returns:
            Estimated overhead per decode step in seconds.
        """
        level = self.get_checkpoint_level(request)
        if level == 0 or checkpoint_size_bytes == 0 or gpu_to_host_bandwidth <= 0:
            return 0.0

        copy_time = checkpoint_size_bytes / gpu_to_host_bandwidth
        # Amortize over the checkpoint interval (steps between copies).
        interval = (
            self.config.level1_interval_steps
            if level == 1
            else self.config.level2_interval_steps
        )
        return copy_time / interval if interval > 0 else copy_time

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
