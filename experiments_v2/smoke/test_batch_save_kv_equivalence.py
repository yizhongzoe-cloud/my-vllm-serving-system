#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit test: batch-save KV correctness.

Goal: verify that `_batch_save_checkpoints` (Stage 2) writes the same
KV bytes to host pinned memory that you'd get by reading the
corresponding GPU blocks directly. Any slice-offset miscalculation,
torch.cat misalignment, or bf16-view bug shows up as a tensor mismatch.

Three sub-tests:
  T1 — All-first-save: 3 reqs none of which have an existing entry.
  T2 — All-delta-append: 3 reqs that already have entries, second save
       grows each by a few blocks.
  T3 — Mixed batch: 1 first-save + 1 delta-append + 1 metadata-only
       (no new blocks).

Each sub-test asserts:
  for every (req, layer) in batch: entry.kv_tensors[layer] equals
  gpu_kv_caches[layer][:, full_block_ids, :, :, :] (byte-exact).

For delta-append we further check the *prefix* of the entry's tensor
(the blocks already saved before this batch) is unchanged by the cat.

Exit 0 = PASS, 1 = FAIL.
"""
import os
import sys
import traceback
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.expanduser(
    "/home/yzhong76/code/my-vllm-serving-system"
))

# Force delta mode on — this is the path Stage 2 is most likely to break.
os.environ["FT_DELTA_CHECKPOINT"] = "1"

from vllm.v1.core.kv_checkpoint_pool import KVCheckpointPool  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402


def make_runner(num_layers: int = 4) -> tuple:
    """Build a minimal model-runner-shaped object that
    `_batch_save_checkpoints` can run against. We only need:
      self.kv_caches, self._ft_checkpoint_pool
    Returns (runner, gpu_kv_caches).
    """
    device = torch.device("cuda:0")
    num_blocks = 64
    block_size = 16
    num_kv_heads = 8
    head_size = 64
    dtype = torch.float16

    torch.manual_seed(2026)
    gpu_kv_caches = [
        torch.randn(
            2, num_blocks, block_size, num_kv_heads, head_size,
            dtype=dtype, device=device,
        )
        for _ in range(num_layers)
    ]

    pool = KVCheckpointPool(max_memory_bytes=512 * 1024 * 1024)

    runner = SimpleNamespace()
    runner.kv_caches = gpu_kv_caches
    runner._ft_checkpoint_pool = pool

    # Bind the unbound method to our SimpleNamespace.
    runner._batch_save_checkpoints = (
        GPUModelRunner._batch_save_checkpoints.__get__(
            runner, SimpleNamespace
        )
    )

    return runner, gpu_kv_caches, pool


def assert_entry_matches_gpu(
    label: str,
    entry,
    full_block_ids: list,
    gpu_kv_caches: list,
) -> None:
    """Assert entry.kv_tensors[layer] == gpu[layer][:, full_block_ids]
    for every layer."""
    if entry is None:
        raise AssertionError(f"{label}: entry is None")
    if len(entry.block_ids) != len(full_block_ids):
        raise AssertionError(
            f"{label}: entry.block_ids has {len(entry.block_ids)} "
            f"blocks, expected {len(full_block_ids)}"
        )
    if entry.block_ids != list(full_block_ids):
        raise AssertionError(
            f"{label}: entry.block_ids={entry.block_ids} "
            f"!= expected {list(full_block_ids)}"
        )
    for layer_idx, gpu_tensor in enumerate(gpu_kv_caches):
        gt = gpu_tensor[:, full_block_ids, :, :, :].cpu()
        host = entry.kv_tensors.get(layer_idx)
        if host is None:
            raise AssertionError(
                f"{label}: entry missing layer {layer_idx}"
            )
        if host.shape != gt.shape:
            raise AssertionError(
                f"{label}: layer {layer_idx} shape mismatch "
                f"host={host.shape} gt={gt.shape}"
            )
        if not torch.equal(host, gt):
            # Diagnostic: report first mismatched block index.
            diff = (host != gt).any(dim=0).any(dim=-1).any(dim=-1).any(dim=-1)
            bad = diff.nonzero(as_tuple=False).flatten().tolist()
            raise AssertionError(
                f"{label}: layer {layer_idx} BYTE MISMATCH; "
                f"bad block-slot indices in entry: {bad[:8]}"
            )


def test_T1_all_first_save() -> None:
    runner, gpu_kv_caches, pool = make_runner()
    reqs = [
        ("req_A", [0, 1, 2], 48),
        ("req_B", [10, 11, 12, 13], 64),
        ("req_C", [20, 21], 32),
    ]
    entries = runner._batch_save_checkpoints(reqs)
    if torch.cuda.is_available() and pool._copy_stream is not None:
        pool._copy_stream.synchronize()
    if len(entries) != 3:
        raise AssertionError(
            f"T1: expected 3 entries, got {len(entries)}"
        )
    entry_by_id = {req_id: ent for req_id, ent, _ in entries}
    for req_id, block_ids, _ in reqs:
        assert_entry_matches_gpu(
            f"T1.{req_id}", entry_by_id[req_id],
            block_ids, gpu_kv_caches,
        )
    print("T1 (all first-save) OK")


def test_T2_all_delta_append() -> None:
    runner, gpu_kv_caches, pool = make_runner()
    # Round 1: initial save with smaller block list.
    initial_reqs = [
        ("req_A", [0, 1], 32),
        ("req_B", [10, 11, 12], 48),
        ("req_C", [20], 16),
    ]
    runner._batch_save_checkpoints(initial_reqs)
    if torch.cuda.is_available() and pool._copy_stream is not None:
        pool._copy_stream.synchronize()

    # Round 2: extend each with a few more blocks (delta append path).
    extended_reqs = [
        ("req_A", [0, 1, 2, 3], 64),
        ("req_B", [10, 11, 12, 13, 14], 80),
        ("req_C", [20, 21], 32),
    ]
    entries = runner._batch_save_checkpoints(extended_reqs)
    if torch.cuda.is_available() and pool._copy_stream is not None:
        pool._copy_stream.synchronize()

    entry_by_id = {req_id: ent for req_id, ent, _ in entries}
    for req_id, block_ids, _ in extended_reqs:
        assert_entry_matches_gpu(
            f"T2.{req_id}", entry_by_id[req_id],
            block_ids, gpu_kv_caches,
        )
    print("T2 (all delta-append) OK")


def test_T3_mixed_batch() -> None:
    runner, gpu_kv_caches, pool = make_runner()

    # Pre-seed: A and C have existing entries; B is new.
    initial_reqs = [
        ("req_A", [0, 1], 32),
        ("req_C", [20, 21], 32),
    ]
    runner._batch_save_checkpoints(initial_reqs)
    if torch.cuda.is_available() and pool._copy_stream is not None:
        pool._copy_stream.synchronize()

    # Mixed: A grows (delta append), B is new (first save),
    # C unchanged (metadata-only path).
    mixed_reqs = [
        ("req_A", [0, 1, 2], 48),       # delta-append: +1 block
        ("req_B", [30, 31, 32], 48),    # first-save
        ("req_C", [20, 21], 32),        # metadata-only (no new blocks)
    ]
    entries = runner._batch_save_checkpoints(mixed_reqs)
    if torch.cuda.is_available() and pool._copy_stream is not None:
        pool._copy_stream.synchronize()

    entry_by_id = {req_id: ent for req_id, ent, _ in entries}
    for req_id, block_ids, _ in mixed_reqs:
        assert_entry_matches_gpu(
            f"T3.{req_id}", entry_by_id[req_id],
            block_ids, gpu_kv_caches,
        )
    print("T3 (mixed: delta + first + metadata-only) OK")


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available; this test requires a GPU.")
        return 1

    for name, fn in [
        ("T1", test_T1_all_first_save),
        ("T2", test_T2_all_delta_append),
        ("T3", test_T3_mixed_batch),
    ]:
        try:
            fn()
        except Exception:
            print(f"FAIL {name}:")
            traceback.print_exc()
            return 1

    print("ALL TESTS PASS — batch_save KV bytes match GPU ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
