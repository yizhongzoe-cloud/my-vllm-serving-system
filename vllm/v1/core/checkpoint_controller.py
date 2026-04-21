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

import os
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
        cost_profile_path: str = "",
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

        # Load profile-driven cost model if provided
        self._cost_model = None
        if cost_profile_path:
            from vllm.v1.core.checkpoint_cost_model import CheckpointCostModel
            self._cost_model = CheckpointCostModel(cost_profile_path)

        # Economic policy is available if we have either:
        # - A profile-driven cost model (preferred), OR
        # - Linear model inputs (fallback)
        self._economic_policy_available = (
            self._cost_model is not None
            or (
                self._replay_throughput > 0
                and self._load_bandwidth > 0
                and self._checkpoint_bandwidth > 0
            )
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
        stable_full_tokens = (request.num_computed_tokens // self._block_size) * self._block_size
        published_tokens = request.num_checkpointed_tokens

        # Guard: only re-evaluate if a new stable block has been produced
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

        # Use profile-driven model if available
        if self._cost_model is not None:
            L = published_tokens
            u = stable_full_tokens - L
            S = request.last_checkpoint_size_bytes

            # Estimate KV bytes per token from last checkpoint, fall back to default
            if request.num_checkpointed_tokens > 0 and S > 0:
                kv_bytes_per_token = S / request.num_checkpointed_tokens
            else:
                kv_bytes_per_token = self._default_kv_bytes_per_token

            delta_S = int(u * kv_bytes_per_token)
            return self._cost_model.should_publish(L, u, S, delta_S, self._checkpoint_lambda)

        # Fall back to linear economic policy
        estimate = self._estimate_online_publication(request)
        return estimate.should_publish

    def get_checkpoint_level(self, request: Request) -> int:
        if self._fixed_blocks > 0:
            return 2 if self._fixed_blocks == 1 else 1
        if self._fixed_level >= 0:
            return self._fixed_level
        return self._get_reporting_level(request)

    def should_checkpoint(self, request: Request) -> bool:
        """Decide whether this request should publish a checkpoint now.

        Three optional runtime guards (A/B/C) can suppress checkpointing
        when the system is too busy, preventing the economic policy from
        degenerating into per-block save (every 16 tokens). Each guard
        is gated by its own env var (default OFF) and checked in order
        before the existing policy. If ANY guard fires, we skip this
        checkpoint opportunity — the request will be re-evaluated next
        step when a new stable block is produced.

        Guard C (FT_CKPT_MIN_INTERVAL_BLOCKS): Minimum block interval
          between consecutive saves for the same request. Fixed
          frequency cap. E.g. =4 means save at most once per 64 tokens.

        Guard B (FT_CKPT_LOAD_GUARD): N_running / max_batch threshold.
          Skip checkpoint when running batch is above a fraction of
          decode capacity. E.g. =0.7 means skip if batch > 70% full.

        Guard A (FT_CKPT_SLO_GUARD): Step-time headroom vs TPOT SLO.
          Skip checkpoint when the observed step time leaves less than
          X% headroom before violating the request's TPOT SLO. This is
          the most principled guard — it implicitly captures N_running,
          GPU contention, and decode interference from checkpoint copies
          (all of which inflate step_time). E.g. =0.2 means skip when
          step_time > 80% of tpot_slo.
        """
        # ── Guard E: tail-skip (skip save for req close to EOS) ─────
        # Mirror of warmup (Guard D): skip saves for the LAST N% of a
        # request's expected output. If the request is almost finished,
        # checkpointing buys little (fault before EOS is increasingly
        # unlikely as we approach EOS; remaining replay cost is small
        # anyway). Typical threshold 0.1 = skip final 10% of tokens.
        # Combined with warmup, can reduce total saves by another
        # 10-15%.
        _tail_frac_str = os.environ.get("FT_CKPT_TAIL_SKIP_FRAC")
        if _tail_frac_str:
            try:
                tail_frac = float(_tail_frac_str)
            except ValueError:
                tail_frac = 0.0
            if 0.0 < tail_frac < 1.0:
                sp = getattr(request, "sampling_params", None)
                mt = getattr(sp, "max_tokens", None) if sp else None
                num_output = getattr(request, "num_output_tokens", 0)
                if mt and mt > 0 and (num_output / mt) > (1 - tail_frac):
                    return False

        # ── Guard D: warm-up tokens (skip short reqs before they prove ──
        # they're worth saving). Short chat requests (< WARMUP decoded
        # tokens) are likely to complete before the next fault, so any
        # checkpoint for them is wasted work. Delay the first save until
        # the request has proven itself "long enough". If the request
        # completes before this threshold, it never gets saved (win);
        # if it decodes past the threshold, normal save cadence kicks
        # in (at most WARMUP tokens of lost progress on fault, which is
        # well within failure_gap_slo).
        _warmup_str = os.environ.get("FT_CKPT_WARMUP_TOKENS")
        if _warmup_str:
            try:
                warmup_tokens = int(_warmup_str)
            except ValueError:
                warmup_tokens = 0
            if warmup_tokens > 0:
                num_output = getattr(request, "num_output_tokens", 0)
                if num_output < warmup_tokens:
                    return False

        # ── Guard C: minimum block interval ──────────────────────────
        _min_blocks_str = os.environ.get("FT_CKPT_MIN_INTERVAL_BLOCKS")
        if _min_blocks_str:
            try:
                min_blocks = int(_min_blocks_str)
            except ValueError:
                min_blocks = 0
            if min_blocks > 0:
                stable = (
                    (request.num_computed_tokens // self._block_size)
                    * self._block_size
                )
                published = request.num_checkpointed_tokens
                if (stable - published) < min_blocks * self._block_size:
                    return False

        # ── Guard B: running batch load threshold ────────────────────
        _load_guard_str = os.environ.get("FT_CKPT_LOAD_GUARD")
        if _load_guard_str:
            try:
                load_threshold = float(_load_guard_str)
            except ValueError:
                load_threshold = 0.0
            if load_threshold > 0:
                n_running = getattr(self, "_n_running", 0)
                try:
                    max_batch = int(
                        os.environ.get("FT_CKPT_LOAD_GUARD_CAPACITY", "26")
                    )
                except ValueError:
                    max_batch = 26
                if max_batch > 0 and n_running > load_threshold * max_batch:
                    return False

        # ── Guard A: step-time SLO headroom ──────────────────────────
        _headroom_str = os.environ.get("FT_CKPT_SLO_GUARD")
        if _headroom_str:
            try:
                min_headroom = float(_headroom_str)
            except ValueError:
                min_headroom = 0.0
            if min_headroom > 0:
                step_ema = getattr(self, "_step_time_ema", 0.0)
                tpot_slo = getattr(request, "tpot_slo_ms", 0.0)
                if step_ema > 0 and tpot_slo > 0:
                    headroom = 1.0 - (step_ema / tpot_slo)
                    if headroom < min_headroom:
                        return False

        # ── Existing policy (unchanged) ──────────────────────────────
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

        Phase 2 improvement B2 (FT_BATCH_CKPT_EVAL=1, default OFF):
        Most requests in a decode step only produce 1 token, so the
        expensive should_checkpoint() cost-model evaluation almost
        always hits the "no new full block" early return. We skip the
        full should_checkpoint() call for requests that provably
        cannot checkpoint this step (no new full block since last
        eval), using a cheap arithmetic predicate instead. This
        bypasses the level lookup + attribute access overhead for the
        ~85-90% of (req, step) pairs that would early-return anyway.

        For the remaining candidates (those with a new full block),
        we still call should_checkpoint() unchanged — preserving the
        exact economic policy semantics.

        Args:
            requests: List of currently running requests.

        Returns:
            Subset of requests that should be checkpointed this step.
        """
        if not requests:
            return []

        # ── Runtime context tracking (for guards B + A) ──────────────
        # Track N_running for the load guard (方案 B).
        self._n_running = len(requests)

        # Track step time EMA for the SLO headroom guard (方案 A).
        # Approximate step_time as wall-clock interval between
        # consecutive get_requests_to_checkpoint() calls. This avoids
        # changing caller signatures — the interval naturally measures
        # schedule() + execute_model() + checkpoint cycle time.
        now = time.time()
        prev = getattr(self, "_last_ckpt_eval_time", 0.0)
        if prev > 0:
            step_ms = (now - prev) * 1000.0
            if 1.0 < step_ms < 500.0:  # filter warmup / outliers
                ema = getattr(self, "_step_time_ema", step_ms)
                self._step_time_ema = 0.1 * step_ms + 0.9 * ema
        self._last_ckpt_eval_time = now

        # Phase 2 B2: fast predicate pre-filter. Default OFF — set
        # FT_BATCH_CKPT_EVAL=1 to enable.
        if os.environ.get("FT_BATCH_CKPT_EVAL") == "1":
            block_size = self._block_size
            last_eval = self._last_evaluated_stable_tokens
            # Pre-compute warmup gate once per batch (Guard D fast path).
            try:
                warmup_tokens = int(os.environ.get(
                    "FT_CKPT_WARMUP_TOKENS", "0"
                ))
            except ValueError:
                warmup_tokens = 0
            candidates: list[Request] = []
            for req in requests:
                # Guard D fast skip: short req below warmup threshold.
                if (
                    warmup_tokens > 0
                    and getattr(req, "num_output_tokens", 0) < warmup_tokens
                ):
                    continue
                stable_full_tokens = (
                    (req.num_computed_tokens // block_size) * block_size
                )
                published_tokens = req.num_checkpointed_tokens
                if stable_full_tokens <= published_tokens:
                    # No new full block since last publish — update
                    # tracker to match should_checkpoint's invariant
                    # and skip.
                    last_eval[req.request_id] = stable_full_tokens
                    continue
                # Also honor the "same stable tokens already evaluated
                # this step" guard the inner _should_checkpoint_by_*
                # methods use.
                last_evaluated = last_eval.get(
                    req.request_id, published_tokens
                )
                if stable_full_tokens <= last_evaluated:
                    continue
                candidates.append(req)

            to_checkpoint = []
            for req in candidates:
                if self.should_checkpoint(req):
                    to_checkpoint.append(req)
            return to_checkpoint

        # Default path: per-request should_checkpoint() call (unchanged).
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
