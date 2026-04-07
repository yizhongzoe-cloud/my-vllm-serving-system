# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV Cache Checkpoint Pool for Fault-Tolerant Serving.

Manages checkpointed KV cache state in host (CPU) pinned memory.
Each checkpoint stores the KV cache blocks for a request at a point in time,
enabling fast recovery after GPU failure by restoring from host memory
instead of full recomputation.
"""

import logging
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

        Args:
            request_id: The request identifier.
            gpu_kv_caches: Per-layer GPU KV cache tensors. Each tensor has
                shape (2, num_blocks, block_size, num_kv_heads, head_size)
                where dimension 1 is indexed by block_id.
                NOTE: this FT checkpoint helper currently assumes that
                EngineCore/model-runner exposes KV tensors in that logical
                layout. That assumption holds for the FLASH_ATTN path we
                validate in experiments. Backends whose logical KV view is
                different (for example TRITON_ATTN with blocks on dim 0)
                must normalize the view before calling into this helper, or
                move block-level copy into a backend-aware implementation.
            block_ids: List of block IDs allocated to this request.
            num_tokens: Number of tokens covered by these blocks.
            async_copy: If True, use a separate CUDA stream for the copy.

        Returns:
            The created CheckpointEntry, or None if insufficient memory.
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

        block_indices = torch.tensor(block_ids, dtype=torch.int64)

        # Estimate size before copying.
        sample = gpu_kv_caches[0]
        per_block_bytes = (
            2  # K and V
            * sample.shape[2]  # block_size
            * sample.shape[3]  # num_kv_heads
            * sample.shape[4]  # head_size
            * sample.element_size()
        )
        estimated_bytes = per_block_bytes * len(block_ids) * len(gpu_kv_caches)

        with self._lock:
            # Evict old checkpoint for this request if exists.
            self._evict_entry(request_id)

            if estimated_bytes > self.available_bytes:
                # Try to free space by evicting oldest checkpoints.
                if not self._evict_to_free(estimated_bytes):
                    logger.warning(
                        "KV Checkpoint Pool: insufficient memory for "
                        "request %s (need %d bytes, available %d bytes)",
                        request_id,
                        estimated_bytes,
                        self.available_bytes,
                    )
                    return None

            # Reserve space before releasing lock for GPU→CPU copy.
            self._reserved_bytes += estimated_bytes

        # Perform the GPU→CPU copy (outside lock to avoid blocking others).
        entry = CheckpointEntry(
            request_id=request_id,
            block_ids=list(block_ids),
            num_tokens=num_tokens,
            timestamp=time.time(),
        )

        device = gpu_kv_caches[0].device
        # Transfer block indices to GPU (on default stream).
        block_indices_gpu = block_indices.to(device)

        if async_copy and torch.cuda.is_available():
            stream = self._get_copy_stream()

            # Pre-allocate pinned memory.
            pinned_tensors: dict[int, torch.Tensor] = {}
            n_sel = len(block_ids)
            sample = gpu_kv_caches[0]
            out_shape = (sample.shape[0], n_sel) + sample.shape[2:]
            for layer_idx in range(len(gpu_kv_caches)):
                pinned_tensors[layer_idx] = torch.empty(
                    out_shape,
                    dtype=sample.dtype,
                    device="cpu",
                ).pin_memory()

            # Stage 1 (default stream): gather the KV blocks into contiguous
            # GPU buffers. This runs on the same stream as decode, so it
            # naturally waits for the latest decode step to finish writing.
            gpu_buffers: dict[int, torch.Tensor] = {}
            for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                gpu_buffers[layer_idx] = gpu_tensor[:, block_indices_gpu, :, :, :].clone()

            # Stage 2 (copy stream): async copy from GPU buffers to pinned
            # host memory. The event ensures copy stream waits for the
            # gather above (on default stream) to complete. CPU does NOT
            # block — decode continues on the default stream.
            event = torch.cuda.current_stream(device).record_event()
            with torch.cuda.stream(stream):
                stream.wait_event(event)
                for layer_idx in range(len(gpu_kv_caches)):
                    pinned_tensors[layer_idx].copy_(
                        gpu_buffers[layer_idx], non_blocking=True)

            entry.kv_tensors = pinned_tensors
        else:
            for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                subset = gpu_tensor[:, block_indices_gpu, :, :, :].clone()
                entry.kv_tensors[layer_idx] = subset.cpu()

        actual_bytes = entry.compute_size()
        with self._lock:
            # Release reservation and account actual usage.
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
        return entry

    def restore_checkpoint(
        self,
        request_id: str,
        gpu_kv_caches: list[torch.Tensor],
        target_block_ids: list[int],
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

        if torch.cuda.is_available():
            stream = self._get_copy_stream()
            # Ensure any in-flight async save completes before we read.
            stream.synchronize()
            with torch.cuda.stream(stream):
                for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
                    host_tensor = entry.kv_tensors.get(layer_idx)
                    if host_tensor is None:
                        continue
                    src = host_tensor[:, :num_checkpoint_blocks].to(
                        device, non_blocking=True
                    )
                    gpu_tensor[:, target_indices, :, :, :] = src
            stream.synchronize()
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
