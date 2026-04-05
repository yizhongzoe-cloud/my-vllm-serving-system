# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decode-first capacity model for admission control.

Loads a profile JSON with two tables:
  - decode_capacity: max concurrent decode requests per replica (under SLO)
  - residual_prefill_capacity: prefill tokens achievable under a given decode load

See experiments_v2/decode_first_capacity_plan.md for design details.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right

from vllm.logger import init_logger

logger = init_logger(__name__)


class DecodeCapacityModel:
    """Profile-based decode-first capacity model.

    Provides:
      - decode_capacity(ctx_bucket) -> int: max decode requests per replica
      - residual_prefill_capacity(decode_load) -> int: remaining prefill tokens
    """

    def __init__(self, profile_path: str | None = None):
        self._fallback = False
        self._default_cap: int = 100
        self._ctx_buckets: dict[int, int] = {}
        self._prefill_loads: list[int] = []
        self._prefill_caps: list[int] = []
        self._use_fitted = False
        self._fit_a: float = 0.0
        self._fit_b: float = 0.0

        if profile_path is None or not os.path.exists(profile_path):
            logger.warning(
                "Decode capacity profile not found: %s. "
                "Using legacy fallback (no capacity-based admission).",
                profile_path,
            )
            self._fallback = True
            return

        with open(profile_path) as f:
            data = json.load(f)

        # Decode capacity
        dec_cap = data.get("decode_capacity", {})
        self._default_cap = dec_cap.get("default", 100)
        by_bucket = dec_cap.get("by_avg_ctx_bucket", {})
        self._ctx_buckets = {int(k): int(v) for k, v in by_bucket.items()}

        # Residual prefill capacity
        rem_pre = data.get("residual_prefill_capacity", {})
        pairs = sorted((int(k), int(v)) for k, v in rem_pre.items())
        if pairs:
            self._prefill_loads = [p[0] for p in pairs]
            self._prefill_caps = [p[1] for p in pairs]

        logger.info(
            "Loaded decode capacity profile: Cap_dec=%d (default), "
            "%d ctx buckets, %d prefill-load points",
            self._default_cap, len(self._ctx_buckets), len(self._prefill_loads),
        )

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    def decode_capacity(self, avg_ctx_bucket: int = 0) -> int:
        """Return Cap_r^dec for the given context-length bucket.

        Args:
            avg_ctx_bucket: average context length of active requests.
                If 0 or not in profile, returns the default (conservative) value.
        """
        if self._fallback:
            return self._default_cap

        if avg_ctx_bucket <= 0 or not self._ctx_buckets:
            return self._default_cap

        # Find nearest bucket (round down)
        buckets = sorted(self._ctx_buckets.keys())
        best = buckets[0]
        for b in buckets:
            if b <= avg_ctx_bucket:
                best = b
            else:
                break
        return self._ctx_buckets[best]

    def residual_prefill_capacity(self, decode_load: int) -> int:
        """Return RemPreCap_r(L) via linear interpolation.

        Args:
            decode_load: current number of active decode requests on this
                replica (using w_j^dec weights).

        Returns:
            Maximum prefill tokens achievable in planning_horizon under
            this decode load. Returns 0 if decode_load exceeds profiled range.
        """
        if self._fallback:
            # No limit in fallback mode
            return 999999

        if self._use_fitted:
            return max(0, int(self._fit_a - self._fit_b * decode_load))

        return self._interpolate(decode_load)

    def _interpolate(self, L: int) -> int:
        """Piecewise linear interpolation of residual prefill capacity."""
        if not self._prefill_loads:
            return 999999

        loads = self._prefill_loads
        caps = self._prefill_caps

        # Below minimum profiled load → return max capacity
        if L <= loads[0]:
            return caps[0]

        # Above maximum profiled load → return 0
        if L >= loads[-1]:
            return max(0, caps[-1])

        # Find segment for interpolation
        idx = bisect_right(loads, L) - 1
        L0, L1 = loads[idx], loads[idx + 1]
        C0, C1 = caps[idx], caps[idx + 1]

        # Linear interpolation
        frac = (L - L0) / (L1 - L0) if L1 != L0 else 0.0
        result = C0 + (C1 - C0) * frac
        return max(0, int(result))

    def fit_linear(self) -> None:
        """Fit a linear model RemPreCap = a - b*L from profile data.

        After calling this, residual_prefill_capacity() uses the fitted
        formula instead of piecewise interpolation.
        """
        if len(self._prefill_loads) < 2:
            logger.warning("Not enough data points to fit linear model")
            return

        import numpy as np
        loads = np.array(self._prefill_loads, dtype=float)
        caps = np.array(self._prefill_caps, dtype=float)

        # Least squares: C = a - b*L  →  C = a + (-b)*L
        A = np.vstack([np.ones_like(loads), loads]).T
        result = np.linalg.lstsq(A, caps, rcond=None)
        self._fit_a = float(result[0][0])
        self._fit_b = float(-result[0][1])
        self._use_fitted = True

        logger.info(
            "Fitted linear model: RemPreCap = %.1f - %.3f * L",
            self._fit_a, self._fit_b,
        )
