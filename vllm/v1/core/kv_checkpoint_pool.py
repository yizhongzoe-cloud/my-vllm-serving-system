# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV Cache Checkpoint Pool for Fault-Tolerant Serving.

Manages checkpointed KV cache state in host (CPU) pinned memory.
Each checkpoint stores the KV cache blocks for a request at a point in time,
enabling fast recovery after GPU failure by restoring from host memory
instead of full recomputation.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CheckpointEntry:
    """A single KV cache checkpoint for one request.

    Stores the KV cache tensors in host pinned memory along with
    metadata about what portion of the request's computation is covered.
    """

    request_id: str
    # Per-layer KV cache tensors on CPU pinned memory.
    # Key: layer index (int), Value: tensor of shape
    #   (2, num_blocks, block_size, num_kv_heads, head_size)
    # where 2 = K and V.
    kv_tensors: dict[int, torch.Tensor] = field(default_factory=dict)
    # Block IDs that were checkpointed (for restore mapping).
    block_ids: list[int] = field(default_factory=list)
    # Number of tokens covered by this checkpoint.
    num_tokens: int = 0
    # Timestamp when checkpoint was created.
    timestamp: float = 0.0
    # Total size in bytes of the checkpoint.
    size_bytes: int = 0
    # Output (sampled) token IDs at the moment of checkpoint, in order.
    # Used by cross-engine reroute so the receiving engine can resume
    # mid-decode without re-sampling. Length may exceed num_tokens-prompt
    # tokens (KV is block-aligned, output_token_ids is per-token); the
    # receiving engine clamps to the KV-covered portion.
    output_token_ids: list[int] = field(default_factory=list)

    def compute_size(self) -> int:
        """Compute total size in bytes of stored tensors."""
        total = 0
        for t in self.kv_tensors.values():
            total += t.nelement() * t.element_size()
        self.size_bytes = total
        return total


class KVCheckpointPool:
    """Host-memory pool for KV cache checkpoints.

    Manages the lifecycle of KV cache checkpoints stored in CPU pinned memory.
    Supports async GPU→CPU copy using CUDA streams to minimize overhead on
    the inference path.

    The pool tracks memory usage and enforces a capacity limit.
    When the limit is reached, the oldest/least-valuable checkpoints
    can be evicted.
    """

    def __init__(
        self,
        max_memory_bytes: int = 8 * 1024 * 1024 * 1024,  # 8 GB default
    ) -> None:
        self.max_memory_bytes = max_memory_bytes
        self._used_bytes: int = 0
        # Track reserved bytes for in-flight copies (not yet in _store).
        self._reserved_bytes: int = 0
        self._store: dict[str, CheckpointEntry] = {}
        self._lock = threading.Lock()
        # Dedicated CUDA stream for async checkpoint copies.
        self._copy_stream: Optional[torch.cuda.Stream] = None

    def _get_copy_stream(self) -> torch.cuda.Stream:
        """Lazily create a CUDA stream for async copies."""
        if self._copy_stream is None and torch.cuda.is_available():
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def available_bytes(self) -> int:
        return self.max_memory_bytes - self._used_bytes - self._reserved_bytes

    @property
    def num_checkpoints(self) -> int:
        return len(self._store)

    def has_checkpoint(self, request_id: str) -> bool:
        return request_id in self._store

    def get_checkpoint(self, request_id: str) -> CheckpointEntry | None:
        return self._store.get(request_id)

    def save_checkpoint(
        self,
        request_id: str,
        gpu_kv_caches: list[torch.Tensor],
        block_ids: list[int],
        num_tokens: int,
        async_copy: bool = True,
    ) -> CheckpointEntry | None:
        """Save a request's KV cache blocks from GPU to host pinned memory.

        FT_DELTA_CHECKPOINT=1 (default OFF): only copy blocks that are NEW
        since the last checkpoint for this request, then append them to the
        existing entry's host tensors (torch.cat on dim 1). This reduces
        the per-save GPU→host copy from O(all_blocks) to O(new_blocks),
        which is typically 1-4 blocks (~128-512 KB) instead of 20-60
        blocks (2.5-7.5 MB) — a 10-15× reduction in copy volume.

        The entry's kv_tensors remain cumulative (blocks 0..N), so
        _publish_shared_checkpoint's delta slice [delta_start:delta_end]
        continues to work unchanged.

        When FT_DELTA_CHECKPOINT is off, behavior is identical to before:
        evict old entry, copy all blocks fresh.

        Args:
            request_id: The request identifier.
            gpu_kv_caches: Per-layer GPU KV cache tensors.
            block_ids: List of block IDs allocated to this request
                (ALL stable blocks, not just new ones).
            num_tokens: Number of tokens covered by these blocks.
            async_copy: If True, use a separate CUDA stream for the copy.

        Returns:
            The created/updated CheckpointEntry, or None if insufficient memory.
        """
        if not block_ids:
            return None

        # Filter out block IDs that exceed the KV cache capacity.
        num_kv_blocks = gpu_kv_caches[0].shape[1]
        valid_block_ids = [
            bid for bid in block_ids if 0 <= bid < num_kv_blocks
        ]
        if len(valid_block_ids) < len(block_ids):
            logger.warning(
                "Checkpoint %s: %d/%d block IDs out of range "
                "(num_kv_blocks=%d), skipping invalid blocks",
                request_id,
                len(block_ids) - len(valid_block_ids),
                len(block_ids),
                num_kv_blocks,
            )
            block_ids = valid_block_ids
            if not block_ids:
                return None

        # ── Delta checkpoint: only copy NEW blocks ──────────────────
        use_delta = os.environ.get("FT_DELTA_CHECKPOINT") == "1"
        existing_entry: CheckpointEntry | None = None
        delta_block_ids = block_ids  # default: copy all

        if use_delta:
            with self._lock:
                existing_entry = self._store.get(request_id)

            if existing_entry is not None:
                prev_n_blocks = len(existing_entry.block_ids)
                if len(block_ids) > prev_n_blocks:
                    # Only copy the new blocks (index prev_n_blocks onwards)
                    delta_block_ids = block_ids[prev_n_blocks:]
                elif len(block_ids) == prev_n_blocks:
                    # Nothing new to copy — update metadata and return
                    existing_entry.num_tokens = num_tokens
                    existing_entry.block_ids = list(block_ids)
                    existing_entry.timestamp = time.time()
                    return existing_entry
                else:
                    # Regression (blocks shrunk) — fall back to full copy
                    existing_entry = None
                    delta_block_ids = block_ids

        block_indices = torch.tensor(delta_block_ids, dtype=torch.int64)

        # Estimate size of the NEW data to copy.
        sample = gpu_kv_caches[0]
        per_block_bytes = (
            2  # K and V
            * sample.shape[2]  # block_size
            * sample.shape[3]  # num_kv_heads
            * sample.shape[4]  # head_size
            * sample.element_size()
        )
        estimated_bytes = per_block_bytes * len(delta_block_ids) * len(gpu_kv_caches)

        with self._lock:
            if not use_delta or existing_entry is None:
                # Full save: evict old entry first
                self._evict_entry(request_id)

            if estimated_bytes > self.available_bytes:
                if not self._evict_to_free(estimated_bytes):
                    logger.warning(
                        "KV Checkpoint Pool: insufficient memory for "
                        "request %s (need %d bytes, available %d bytes)",
                        request_id,
                        estimated_bytes,
                        self.available_bytes,
                    )
                    return None

            self._reserved_bytes += estimated_bytes

        # ── GPU→CPU copy (only delta blocks) ────────────────────────
        device = gpu_kv_caches[0].device

        if async_copy and torch.cuda.is_available():
            stream = self._get_copy_stream()

            n_sel = len(delta_block_ids)
            out_shape = (sample.shape[0], n_sel) + sample.shape[2:]
            delta_pinned: dict[int, torch.Tensor] = {}
            for layer_idx in range(len(gpu_kv_caches)):
                delta_pinned[layer_idx] = torch.empty(
                    out_shape, dtype=sample.dtype, device="cpu",
                ).pin_memory()

            # Run the GPU-side gather + clone + PCIe copy entirely on
            # the copy stream so they don't block decode kernels
            # enqueued on the default stream.
            #
            # Mark "default stream has finished writing the KV blocks
            # we are about to read". Recording on the current (default)
            # stream does NOT block it — the next decode step continues
            # to enqueue right behind this marker.
            event = torch.cuda.current_stream(device).record_event()

            with torch.cuda.stream(stream):
                # Copy stream waits for decode to reach the marker.
                # Decode never waits on copy stream, so decode N+1 can
                # run concurrently with the gather/clone/PCIe-copy below
                # whenever GPU SMs are free.
                stream.wait_event(event)

                # Move the small block-index tensor onto the copy
                # stream too, otherwise the H2D transfer would be
                # implicit on the default stream.
                block_indices_gpu = block_indices.to(
                    device, non_blocking=True
                )

                gpu_buffers: dict[int, torch.Tensor] = {}
                for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                    gpu_buffers[layer_idx] = gpu_tensor[
                        :, block_indices_gpu, :, :, :
                    ].clone()

                for layer_idx in range(len(gpu_kv_caches)):
                    delta_pinned[layer_idx].copy_(
                        gpu_buffers[layer_idx], non_blocking=True
                    )
        else:
            block_indices_gpu = block_indices.to(device)
            delta_pinned = {}
            for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                subset = gpu_tensor[
                    :, block_indices_gpu, :, :, :
                ].clone()
                delta_pinned[layer_idx] = subset.cpu()

        # ── Build / update entry ────────────────────────────────────
        if use_delta and existing_entry is not None:
            # Append delta to existing entry's host tensors (cat on dim 1)
            for layer_idx in range(len(gpu_kv_caches)):
                existing_entry.kv_tensors[layer_idx] = torch.cat(
                    [existing_entry.kv_tensors[layer_idx],
                     delta_pinned[layer_idx]],
                    dim=1,
                )
            existing_entry.block_ids = list(block_ids)
            existing_entry.num_tokens = num_tokens
            existing_entry.timestamp = time.time()
            entry = existing_entry

            actual_bytes = entry.compute_size()
            with self._lock:
                self._reserved_bytes -= estimated_bytes
                # Update used_bytes: old size was already counted, add delta
                old_size = sum(
                    t.nelement() * t.element_size()
                    for t in delta_pinned.values()
                )
                self._used_bytes += old_size

            logger.debug(
                "Delta checkpoint %s: appended %d new blocks "
                "(total %d blocks, %d tokens, %.2f MB)",
                request_id,
                len(delta_block_ids),
                len(block_ids),
                num_tokens,
                actual_bytes / (1024 * 1024),
            )
        else:
            # Full save (first checkpoint or delta disabled)
            entry = CheckpointEntry(
                request_id=request_id,
                block_ids=list(block_ids),
                num_tokens=num_tokens,
                timestamp=time.time(),
            )
            entry.kv_tensors = delta_pinned

            actual_bytes = entry.compute_size()
            with self._lock:
                self._reserved_bytes -= estimated_bytes
                self._store[request_id] = entry
                self._used_bytes += actual_bytes

            logger.debug(
                "Checkpointed request %s: %d tokens, %d blocks, %.2f MB",
                request_id,
                num_tokens,
                len(block_ids),
                actual_bytes / (1024 * 1024),
            )

        # ── Per-fire ckpt stats logging (FT_CKPT_STATS_LOG=1) ──────────
        # Records each save_checkpoint invocation to a CSV. Used in
        # sanity verification to confirm: (1) ckpt is actually firing,
        # (2) fire frequency matches fixed_checkpoint_blocks=1 cadence,
        # (3) per-fire bytes are incremental (≈1 block) when
        # FT_DELTA_CHECKPOINT=1, not full (entire KV cumulative).
        if os.environ.get("FT_CKPT_STATS_LOG") == "1":
            if not hasattr(self, "_ckpt_stats_csv_file"):
                from collections import defaultdict as _defaultdict
                _out_dir = os.environ.get(
                    "FT_CKPT_STATS_OUTPUT_DIR", "/tmp"
                )
                os.makedirs(_out_dir, exist_ok=True)
                _csv_path = os.path.join(
                    _out_dir, f"ckpt_stats_pid{os.getpid()}.csv"
                )
                self._ckpt_stats_csv_file = open(
                    _csv_path, "w", buffering=1
                )
                self._ckpt_stats_csv_file.write(
                    "timestamp,request_id,fire_count,mode,"
                    "num_blocks_written,bytes_written,"
                    "num_tokens_total,num_layers\n"
                )
                self._ckpt_stats_fire_count = _defaultdict(int)
            self._ckpt_stats_fire_count[request_id] += 1
            _mode = (
                "delta" if (use_delta and existing_entry is not None)
                else "full"
            )
            _n_blocks = len(delta_block_ids)
            _bytes_written = (
                per_block_bytes * _n_blocks * len(gpu_kv_caches)
            )
            self._ckpt_stats_csv_file.write(
                f"{time.time():.6f},{request_id},"
                f"{self._ckpt_stats_fire_count[request_id]},"
                f"{_mode},{_n_blocks},{_bytes_written},"
                f"{num_tokens},{len(gpu_kv_caches)}\n"
            )

        return entry

    def restore_checkpoint(
        self,
        request_id: str,
        gpu_kv_caches: list[torch.Tensor],
        target_block_ids: list[int],
        sync: bool = True,
    ) -> int:
        """Restore a checkpointed KV cache from host memory to GPU.

        Args:
            request_id: The request identifier.
            gpu_kv_caches: Per-layer GPU KV cache tensors to write into.
                Same layout assumption as save_checkpoint(): this helper
                indexes blocks on dim 1 of a logical
                (2, num_blocks, block_size, num_kv_heads, head_size) view.
                The current FT experiments only validate this path with
                FLASH_ATTN. Other backends need a normalized view or a
                backend-aware restore path.
            target_block_ids: Block IDs on the target GPU to write the
                restored data into (may differ from original block_ids).
            sync: If True (default), block until copy completes before
                returning (legacy behavior).
                If False, enqueue copy on copy stream and return
                immediately. Caller must use query_async_restore() to
                check completion before reading restored KV. Used by
                Phase 2 C-mode to overlap reload with concurrent forward
                of other requests.

        Returns:
            Number of tokens restored, or 0 if no checkpoint found.
        """
        entry = self._store.get(request_id)
        if entry is None:
            logger.warning(
                "No checkpoint found for request %s", request_id
            )
            return 0

        num_checkpoint_blocks = len(entry.block_ids)
        if len(target_block_ids) < num_checkpoint_blocks:
            logger.warning(
                "Target has fewer blocks (%d) than checkpoint (%d) "
                "for request %s",
                len(target_block_ids),
                num_checkpoint_blocks,
                request_id,
            )
            num_checkpoint_blocks = len(target_block_ids)

        device = gpu_kv_caches[0].device
        num_kv_blocks = gpu_kv_caches[0].shape[1]

        # Filter out block IDs that exceed the KV cache capacity to
        # prevent CUDA index-out-of-bounds errors during failover.
        valid_ids = [
            bid for bid in target_block_ids[:num_checkpoint_blocks]
            if 0 <= bid < num_kv_blocks
        ]
        if len(valid_ids) < num_checkpoint_blocks:
            logger.warning(
                "Restore %s: %d/%d target blocks out of range "
                "(num_kv_blocks=%d), truncating",
                request_id,
                num_checkpoint_blocks - len(valid_ids),
                num_checkpoint_blocks,
                num_kv_blocks,
            )
            num_checkpoint_blocks = len(valid_ids)
            if num_checkpoint_blocks == 0:
                return 0

        target_indices = torch.tensor(
            valid_ids,
            dtype=torch.int64,
            device=device,
        )

        # ── CUDA-event-timed reload profiling (FT_CUDA_EVENT_PROFILE=1) ──
        # Times the host->GPU restore copy on the copy stream. CSV row is
        # written when the copy completes (lazily for sync=True or via
        # query_async_restore for sync=False).
        _profile = (
            os.environ.get("FT_CUDA_EVENT_PROFILE") == "1"
            and torch.cuda.is_available()
        )
        if _profile:
            self._init_reload_csv_if_needed()

        # Compute bytes_loaded upfront (we need it whether or not we sync).
        sample = gpu_kv_caches[0]
        per_block_bytes = (
            2  # K and V
            * sample.shape[2]  # block_size
            * sample.shape[3]  # num_kv_heads
            * sample.shape[4]  # head_size
            * sample.element_size()
        )
        bytes_loaded = (
            per_block_bytes * num_checkpoint_blocks
            * len(gpu_kv_caches)
        )

        _profile_start = None
        _profile_end = None
        _profile_ts = time.time()
        if _profile:
            _profile_start = torch.cuda.Event(enable_timing=True)
            _profile_end = torch.cuda.Event(enable_timing=True)

        if torch.cuda.is_available():
            stream = self._get_copy_stream()
            # Ensure any in-flight async save completes before we read.
            stream.synchronize()
            # Record profile events ON the copy stream so elapsed_time
            # reflects actual host->GPU transfer cost.
            if _profile and _profile_start is not None:
                _profile_start.record(stream)
            with torch.cuda.stream(stream):
                for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                    host_tensor = entry.kv_tensors.get(layer_idx)
                    if host_tensor is None:
                        continue
                    src = host_tensor[:, :num_checkpoint_blocks].to(
                        device, non_blocking=True
                    )
                    gpu_tensor[:, target_indices, :, :, :] = src
            if _profile and _profile_end is not None:
                _profile_end.record(stream)
            # Always need a "done" marker for query_async_restore even
            # if profile is off (but profile is on by default in our
            # FT runs, so _profile_end is the same).
            done_event = _profile_end
            if not sync and done_event is None:
                done_event = torch.cuda.Event()
                done_event.record(stream)

            if sync:
                # Legacy synchronous path: host blocks until copy done.
                stream.synchronize()
                if _profile:
                    self._write_reload_csv_row(
                        request_id=request_id,
                        start_event=_profile_start,
                        end_event=_profile_end,
                        enqueue_ts=_profile_ts,
                        num_blocks=num_checkpoint_blocks,
                        num_tokens=entry.num_tokens,
                        bytes_loaded=bytes_loaded,
                        steps_to_complete=1,
                        sync_mode="sync",
                    )
            else:
                # Phase 2 C-mode async path: do NOT block host. Caller
                # must call query_async_restore() on subsequent steps to
                # check completion. CSV row is deferred until completion.
                if not hasattr(self, "_async_restore_state"):
                    self._async_restore_state: dict[str, dict] = {}
                self._async_restore_state[request_id] = {
                    "start_event": _profile_start,
                    "done_event": done_event,
                    "enqueue_time": _profile_ts,
                    "tokens": entry.num_tokens,
                    "num_blocks": num_checkpoint_blocks,
                    "bytes_loaded": bytes_loaded,
                }
        else:
            for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                host_tensor = entry.kv_tensors.get(layer_idx)
                if host_tensor is None:
                    continue
                src = host_tensor[:, :num_checkpoint_blocks].to(device)
                gpu_tensor[:, target_indices, :, :, :] = src

        import time as _time
        logger.info(
            "FAULT_EVENT kv_restore_done request=%s wall_time=%.6f "
            "tokens=%d blocks=%d",
            request_id, _time.time(),
            entry.num_tokens, num_checkpoint_blocks,
        )
        return entry.num_tokens

    # ── Reload CSV / async restore helpers (Phase 2) ──────────────────

    def _init_reload_csv_if_needed(self) -> None:
        """Lazy-init reload_times CSV. Idempotent."""
        if hasattr(self, "_reload_csv_file"):
            return
        out_dir = os.environ.get("FT_CUDA_EVENT_OUTPUT_DIR", "/tmp")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(
            out_dir, f"reload_times_pid{os.getpid()}.csv"
        )
        self._reload_csv_file = open(csv_path, "w", buffering=1)
        self._reload_csv_file.write(
            "timestamp,request_id,reload_ms,num_blocks,num_tokens,"
            "bytes_loaded,steps_to_complete,wait_ms_total,sync_mode\n"
        )
        # Register atexit flush for any pending async restores.
        import atexit as _atexit
        _atexit.register(self._flush_pending_async_restores)

    def _write_reload_csv_row(
        self,
        request_id: str,
        start_event: "torch.cuda.Event | None",
        end_event: "torch.cuda.Event | None",
        enqueue_ts: float,
        num_blocks: int,
        num_tokens: int,
        bytes_loaded: int,
        steps_to_complete: int,
        sync_mode: str,
    ) -> None:
        """Write one reload row to CSV. Caller must ensure end_event
        is ready (queryable). Failures are logged and skipped."""
        if not hasattr(self, "_reload_csv_file"):
            return
        try:
            if start_event is not None and end_event is not None:
                reload_ms = start_event.elapsed_time(end_event)
            else:
                reload_ms = 0.0
        except Exception:
            reload_ms = 0.0
        wait_ms_total = (time.time() - enqueue_ts) * 1000.0
        try:
            self._reload_csv_file.write(
                f"{enqueue_ts:.6f},{request_id},{reload_ms:.4f},"
                f"{num_blocks},{num_tokens},{bytes_loaded},"
                f"{steps_to_complete},{wait_ms_total:.4f},{sync_mode}\n"
            )
        except Exception:
            pass

    def query_async_restore(
        self,
        request_id: str,
        steps_waited: int = 1,
    ) -> bool:
        """Check if an async restore for request_id is complete.

        On completion, writes the CSV row (with steps_waited and
        wait_ms_total) and clears the per-request state.

        Args:
            request_id: The request whose async restore to check.
            steps_waited: How many engine steps have elapsed since
                enqueue (engine tracks this and passes in).

        Returns:
            True if complete (or never registered, defensive), False if
            the restore is still in flight.
        """
        if not hasattr(self, "_async_restore_state"):
            return True
        state = self._async_restore_state.get(request_id)
        if state is None:
            return True
        end_event = state["done_event"]
        if end_event is None:
            # No event recorded — treat as complete (defensive).
            del self._async_restore_state[request_id]
            return True
        try:
            ready = bool(end_event.query())
        except Exception:
            # If query fails, conservatively report not ready; atexit
            # flush will write the row.
            return False
        if not ready:
            return False
        # Write completion row.
        self._write_reload_csv_row(
            request_id=request_id,
            start_event=state.get("start_event"),
            end_event=end_event,
            enqueue_ts=state["enqueue_time"],
            num_blocks=state["num_blocks"],
            num_tokens=state["tokens"],
            bytes_loaded=state.get("bytes_loaded", 0),
            steps_to_complete=steps_waited,
            sync_mode="async",
        )
        del self._async_restore_state[request_id]
        return True

    def _flush_pending_async_restores(self) -> None:
        """Called via atexit. Flush any unfinished async restore CSV
        rows so we don't lose data if the process dies mid-flight."""
        if not hasattr(self, "_async_restore_state"):
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for req_id, state in list(
            self._async_restore_state.items()
        ):
            try:
                self._write_reload_csv_row(
                    request_id=req_id,
                    start_event=state.get("start_event"),
                    end_event=state.get("done_event"),
                    enqueue_ts=state["enqueue_time"],
                    num_blocks=state["num_blocks"],
                    num_tokens=state["tokens"],
                    bytes_loaded=state.get("bytes_loaded", 0),
                    steps_to_complete=-1,  # marker: never completed
                    sync_mode="async_atexit",
                )
            except Exception:
                continue
        self._async_restore_state.clear()
        try:
            self._reload_csv_file.close()
        except Exception:
            pass

    def delete_checkpoint(self, request_id: str) -> None:
        """Delete a checkpoint and free its memory."""
        with self._lock:
            self._evict_entry(request_id)

    def _evict_entry(self, request_id: str) -> None:
        """Remove an entry from the store (must hold _lock)."""
        entry = self._store.pop(request_id, None)
        if entry is not None:
            self._used_bytes -= entry.size_bytes

    def _evict_to_free(self, needed_bytes: int) -> bool:
        """Evict oldest checkpoints until enough memory is free.

        Must hold _lock. Returns True if enough space was freed.
        """
        if self.available_bytes >= needed_bytes:
            return True

        # Sort by timestamp (oldest first) for eviction.
        sorted_entries = sorted(
            self._store.values(), key=lambda e: e.timestamp
        )
        for entry in sorted_entries:
            if self.available_bytes >= needed_bytes:
                return True
            self._store.pop(entry.request_id, None)
            self._used_bytes -= entry.size_bytes
            logger.debug(
                "Evicted checkpoint for request %s to free %.2f MB",
                entry.request_id,
                entry.size_bytes / (1024 * 1024),
            )
        return self.available_bytes >= needed_bytes

    def clear(self) -> None:
        """Delete all checkpoints."""
        with self._lock:
            self._store.clear()
            self._used_bytes = 0
            self._reserved_bytes = 0

    def get_stats(self) -> dict:
        """Return checkpoint pool statistics."""
        return {
            "num_checkpoints": self.num_checkpoints,
            "used_bytes": self._used_bytes,
            "reserved_bytes": self._reserved_bytes,
            "used_mb": self._used_bytes / (1024 * 1024),
            "max_bytes": self.max_memory_bytes,
            "max_mb": self.max_memory_bytes / (1024 * 1024),
            "utilization": (
                (self._used_bytes + self._reserved_bytes)
                / self.max_memory_bytes
                if self.max_memory_bytes > 0
                else 0.0
            ),
        }
