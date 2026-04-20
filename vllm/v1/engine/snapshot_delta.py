# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Delta-encoded RequestSnapshot protocol (R4 optimization, env-gated).

Reduces per-step Python overhead in process_engine_outputs when centralized
solver is enabled. The default protocol transmits a full snapshot of every
active request each cycle — this module emits only deltas (added / removed /
changed fields), and the client maintains a persistent state.

Env gate: FT_SNAPSHOT_DELTA=1 (default OFF).

**Status**: EXPERIMENTAL — framework is ready but NOT integrated into
the hot path until validated. Check experiments_v2/docs for validation
status before enabling in production.

Protocol:
    Version N — monotonically incremented per engine. Applier requires
    contiguous versions; gap triggers full-snapshot resync.

Fields transmitted in `updated`:
    - request_id
    - num_computed_tokens (increments every step)
    - num_output_tokens   (increments on token emission)
    - num_checkpointed_tokens (increments on checkpoint events)
    - checkpoint_level    (rare transitions)

Static fields (prompt_len, generation_len, SLOs, replica_id) are sent
once in `added`, cached locally afterward.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.engine import RequestSnapshot

logger = init_logger(__name__)


@dataclass
class SnapshotDelta:
    """Delta message transmitted in EngineCoreOutputs when FT_SNAPSHOT_DELTA=1."""

    version: int
    added: list[RequestSnapshot] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    # Compact tuple form to minimize msgpack size on the wire.
    # (request_id, num_computed_tokens, num_output_tokens,
    #  num_checkpointed_tokens, checkpoint_level)
    updated: list[tuple[str, int, int, int, int]] = field(default_factory=list)


class SnapshotDeltaBuilder:
    """Engine-side: converts full snapshot list into a delta.

    Remembers the last emitted snapshot per request_id. On each call:
      - Reqs in current but not last → `added`
      - Reqs in last but not current → `removed`
      - Reqs in both with dynamic-field changes → `updated`
    """

    def __init__(self) -> None:
        self._last_full: dict[str, RequestSnapshot] = {}
        self._version = 0

    def build_delta(
        self, current: list[RequestSnapshot] | None
    ) -> SnapshotDelta:
        self._version += 1
        if current is None:
            current = []

        current_map = {s.request_id: s for s in current}
        last_ids = set(self._last_full.keys())
        current_ids = set(current_map.keys())

        removed = sorted(last_ids - current_ids)
        added = [
            current_map[rid] for rid in sorted(current_ids - last_ids)
        ]

        updated: list[tuple[str, int, int, int, int]] = []
        for rid in current_ids & last_ids:
            prev = self._last_full[rid]
            cur = current_map[rid]
            if (
                prev.num_computed_tokens != cur.num_computed_tokens
                or prev.num_output_tokens != cur.num_output_tokens
                or prev.num_checkpointed_tokens != cur.num_checkpointed_tokens
                or prev.checkpoint_level != cur.checkpoint_level
            ):
                updated.append((
                    rid,
                    cur.num_computed_tokens,
                    cur.num_output_tokens,
                    cur.num_checkpointed_tokens,
                    cur.checkpoint_level,
                ))

        # Update cache for next delta
        self._last_full = current_map
        return SnapshotDelta(
            version=self._version,
            added=added,
            removed=removed,
            updated=updated,
        )

    def reset(self) -> None:
        """Reset state. Called on engine restart after failure."""
        self._last_full.clear()
        self._version = 0


class SnapshotDeltaApplier:
    """Client-side: maintains persistent snapshot state, applies deltas.

    Returns updated snapshot list. On version gap, returns None to signal
    that a full-snapshot resync is required.
    """

    def __init__(self) -> None:
        self._state: dict[str, RequestSnapshot] = {}
        self._last_version = 0  # 0 = uninitialized; first delta must be 1

    def apply(
        self, delta: SnapshotDelta
    ) -> list[RequestSnapshot] | None:
        # Version gap detection
        if self._last_version != 0 and delta.version != self._last_version + 1:
            logger.warning(
                "Snapshot delta version gap: expected %d, got %d — resync",
                self._last_version + 1, delta.version,
            )
            return None

        self._last_version = delta.version

        # Apply removals
        for rid in delta.removed:
            self._state.pop(rid, None)

        # Apply additions
        for snap in delta.added:
            self._state[snap.request_id] = snap

        # Apply updates — only dynamic fields change
        for rid, num_computed, num_output, num_ckpt, ckpt_level in delta.updated:
            prev = self._state.get(rid)
            if prev is None:
                logger.warning(
                    "Snapshot delta update for unknown request %s — resync", rid
                )
                return None
            prev.num_computed_tokens = num_computed
            prev.num_output_tokens = num_output
            prev.num_checkpointed_tokens = num_ckpt
            prev.checkpoint_level = ckpt_level

        return list(self._state.values())

    def full_refresh(self, snapshots: list[RequestSnapshot], version: int) -> None:
        """Replace state with a full snapshot list (on resync)."""
        self._state = {s.request_id: s for s in snapshots}
        self._last_version = version

    def reset(self) -> None:
        self._state.clear()
        self._last_version = 0


def is_enabled() -> bool:
    """Check FT_SNAPSHOT_DELTA env flag. Default OFF."""
    return os.environ.get("FT_SNAPSHOT_DELTA", "0") == "1"
