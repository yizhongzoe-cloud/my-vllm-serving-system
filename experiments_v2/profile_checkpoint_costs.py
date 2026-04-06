#!/usr/bin/env python3
"""
Profile checkpoint costs (T_prefill, T_load, T_ckpt) for adaptive checkpoint model.

This script generates a checkpoint_cost_profile.json by benchmarking:
1. T_prefill(n): prefill latency for n tokens (using max_tokens=1 microbench)
2. T_load(S): KV cache restore latency for S bytes
3. T_ckpt(S): KV cache checkpoint latency for S bytes
4. c0: publication fixed overhead

All measurements are median over multiple runs with warmup.
"""

import argparse
import json
import logging

logger = logging.getLogger(__name__)
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm.v1.core.kv_checkpoint_pool import KVCheckpointPool


# ============================================================================
# Prefill Microbench
# ============================================================================

class PrefillBenchmark:
    """Benchmark prefill latency using OpenAI-compatible API with max_tokens=1.

    WARNING: This measures end-to-end HTTP latency including network, JSON serialization,
    and first-token generation, NOT pure prefill. Use the resulting profile with caution;
    it includes system overheads beyond pure KV cache computation.
    """

    def __init__(self, model: str, port: int = 8300, timeout_sec: float = 60.0):
        self.model = model
        self.port = port
        self.timeout_sec = timeout_sec
        self.base_url = f"http://localhost:{port}/v1"
        self._tokenizer = None
        self.tokenizer_failed = False  # Track if tokenizer load failed

    def _get_tokenizer(self):
        """Lazy load tokenizer to get true token counts."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            except Exception as e:
                print(f"  Warning: Failed to load tokenizer: {e}")
                print(f"  Falling back to character-based estimation")
                self._tokenizer = False  # Mark as failed
                self.tokenizer_failed = True  # Track for is_real_measurement
        return self._tokenizer if self._tokenizer else None

    def _generate_text_with_n_tokens(self, target_tokens: int) -> str:
        """Generate text with approximately target_tokens tokens.

        Uses tokenizer if available, otherwise estimates (4 chars ≈ 1 token).
        """
        tokenizer = self._get_tokenizer()
        if not tokenizer:
            # Fallback: estimate 4 chars per token
            return "a " * target_tokens

        # Start with an estimate and refine
        text = "a " * target_tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Binary search to get close to target
        for _ in range(5):
            if len(tokens) == target_tokens:
                return text
            elif len(tokens) < target_tokens:
                # Add more text
                deficit = target_tokens - len(tokens)
                text += "a " * deficit
            else:
                # Remove text
                excess = len(tokens) - target_tokens
                text = text[:len(text) - excess * 2]

            tokens = tokenizer.encode(text, add_special_tokens=False)

        return text

    def run(
        self,
        token_lengths: list[int],
        num_warmup: int = 2,
        num_trials: int = 5,
    ) -> dict[int, float]:
        """
        Benchmark prefill latency for each token length.

        Returns:
            {token_length: median_latency_ms}
        """
        results = {}

        for n_tokens in token_lengths:
            print(f"  Benchmarking T_prefill({n_tokens})...", end=" ", flush=True)

            latencies = []

            # Warmup
            for _ in range(num_warmup):
                self._measure_prefill(n_tokens)

            # Actual measurements
            for _ in range(num_trials):
                latency_ms = self._measure_prefill(n_tokens)
                latencies.append(latency_ms)

            median_latency = float(np.median(latencies))
            results[n_tokens] = median_latency
            print(f"median={median_latency:.2f}ms (min={min(latencies):.2f}, max={max(latencies):.2f})")

        return results

    def _measure_prefill(self, n_tokens: int, max_retries: int = 3) -> float:
        """
        Measure latency of a single prefill request with max_tokens=1.

        NOTE: This measures end-to-end HTTP latency including network roundtrip,
        JSON encoding/decoding, and first-token decode, NOT pure prefill computation.
        The resulting values are suitable for amortizing communication costs in
        checkpoint decisions but should not be interpreted as prefill throughput.

        Returns:
            latency in milliseconds
        """
        import aiohttp
        import asyncio

        for attempt in range(max_retries):
            try:
                async def _request():
                    timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        # Generate text with true token count using tokenizer
                        content = self._generate_text_with_n_tokens(n_tokens)

                        payload = {
                            "model": self.model,
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": 1,
                            "stream": False,
                        }

                        url = f"{self.base_url}/chat/completions"
                        start = time.perf_counter()
                        async with session.post(url, json=payload) as resp:
                            if resp.status != 200:
                                raise RuntimeError(f"HTTP {resp.status}")
                            await resp.json()
                        end = time.perf_counter()

                        return (end - start) * 1000.0  # Convert to ms

                return asyncio.run(_request())

            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Prefill benchmark failed after {max_retries} retries "
                        f"for {n_tokens} tokens: {e}. "
                        f"Check that server is running and model is correct."
                    ) from e
                if attempt < max_retries - 1:
                    print(f"    (attempt {attempt + 1}/{max_retries} failed: {e}, retrying...)", flush=True)
                time.sleep(0.1)


# ============================================================================
# KV Checkpoint/Restore Benchmark
# ============================================================================

class KVCheckpointBenchmark:
    """Benchmark KV checkpoint save/restore latency using actual KVCheckpointPool API.

    Uses realistic per-layer KV cache tensors matching the target model's geometry.
    Automatically detects model geometry from HuggingFace config.
    """

    def __init__(
        self,
        num_kv_heads: int = 8,
        head_size: int = 64,
        num_layers: int = 32,
        block_size: int = 16,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.block_size = block_size

    @classmethod
    def from_model_name(cls, model_name: str, block_size: int = 16) -> "KVCheckpointBenchmark":
        """Auto-detect KV geometry from HuggingFace model config."""
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_size = cfg.hidden_size // cfg.num_attention_heads
        num_layers = cfg.num_hidden_layers
        logger.info(
            "KV geometry for %s: num_kv_heads=%d, head_size=%d, num_layers=%d",
            model_name, num_kv_heads, head_size, num_layers,
        )
        return cls(
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            num_layers=num_layers,
            block_size=block_size,
        )

    def benchmark_save_restore(
        self,
        block_counts: list[int],
        num_trials: int = 5,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """
        Benchmark save and restore latency for different block counts using
        the actual KVCheckpointPool implementation.

        Uses realistic KV cache structure with per-layer tensors to match
        the actual checkpoint/restore paths.

        Returns:
            (save_latencies, restore_latencies)
            {bytes: median_latency_ms}
        """
        import torch

        save_results = {}
        restore_results = {}

        pool = KVCheckpointPool(max_memory_bytes=4 * 1024 * 1024 * 1024)  # 4GB pool

        for num_blocks in block_counts:
            # Create realistic per-layer KV cache tensors (mimic actual GPU KV cache)
            # Shape per layer: (2, num_blocks, block_size, num_kv_heads, head_size)
            # where 2 = K and V dimensions
            num_layers = self.num_layers

            gpu_kv_caches = []
            total_bytes = 0
            for _ in range(num_layers):
                kv_tensor = torch.randn(
                    2, num_blocks, self.block_size, self.num_kv_heads, self.head_size,
                    dtype=torch.float16, device="cuda"
                )
                gpu_kv_caches.append(kv_tensor)
                total_bytes += kv_tensor.numel() * kv_tensor.element_size()

            block_ids = list(range(num_blocks))
            num_tokens = num_blocks * self.block_size

            print(f"  Benchmarking KV save/restore ({num_blocks} blocks, {total_bytes} bytes)...", end=" ", flush=True)

            # Benchmark save using actual KVCheckpointPool API
            save_latencies = []
            for trial in range(num_trials):
                # Clear previous checkpoint
                pool._store.clear()
                pool._used_bytes = 0

                start = time.perf_counter()
                entry = pool.save_checkpoint(
                    request_id=f"test_req_{trial}",
                    gpu_kv_caches=gpu_kv_caches,
                    block_ids=block_ids,
                    num_tokens=num_tokens,
                    async_copy=False,  # Synchronous for accurate measurement
                )
                if entry is not None:
                    # Ensure copy is complete (synchronize CUDA)
                    torch.cuda.synchronize()
                end = time.perf_counter()

                if entry is not None:
                    save_latencies.append((end - start) * 1000.0)

            # Benchmark restore using actual KVCheckpointPool API
            restore_latencies = []
            # Save one checkpoint to restore
            entry = pool.save_checkpoint(
                request_id="restore_test",
                gpu_kv_caches=gpu_kv_caches,
                block_ids=block_ids,
                num_tokens=num_tokens,
                async_copy=False,
            )

            if entry is not None:
                for _ in range(num_trials):
                    # Create fresh GPU tensors for restore target
                    target_kv_caches = []
                    for _ in range(num_layers):
                        t = torch.zeros(
                            2, num_blocks, self.block_size, self.num_kv_heads, self.head_size,
                            dtype=torch.float16, device="cuda"
                        )
                        target_kv_caches.append(t)

                    start = time.perf_counter()
                    tokens_restored = pool.restore_checkpoint(
                        request_id="restore_test",
                        gpu_kv_caches=target_kv_caches,
                        target_block_ids=block_ids,
                    )
                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    if tokens_restored > 0:
                        restore_latencies.append((end - start) * 1000.0)

            if save_latencies:
                save_results[total_bytes] = float(np.median(save_latencies))
            if restore_latencies:
                restore_results[total_bytes] = float(np.median(restore_latencies))

            median_save = save_results.get(total_bytes, 0)
            median_restore = restore_results.get(total_bytes, 0)
            print(f"save={median_save:.2f}ms, restore={median_restore:.2f}ms")

            # Cleanup
            for t in gpu_kv_caches:
                del t
            torch.cuda.empty_cache()

        return save_results, restore_results


# ============================================================================
# Estimate c0
# ============================================================================

def estimate_c0_from_checkpoint_data(
    checkpoint_latencies: dict[int, float],
) -> tuple[float, float]:
    """
    Estimate c0 and size-dependent slope from checkpoint latency data.

    Assumes: T_ckpt(S) = c0 + a*S

    Uses linear regression on smallest 3-5 points to be robust to measurement noise.

    Returns:
        (c0, a)
    """
    if len(checkpoint_latencies) < 2:
        # Fallback: just use min as c0
        return min(checkpoint_latencies.values()), 0.0

    items = sorted(checkpoint_latencies.items())

    # Use first few points for regression (smallest sizes are typically most accurate)
    num_points_for_fit = min(5, len(items))
    fit_items = items[:num_points_for_fit]

    sizes = np.array([S for S, _ in fit_items])
    latencies = np.array([T for _, T in fit_items])

    # Linear regression: T = c0 + a*S
    # polyfit returns [a, c0] for degree 1
    try:
        coeffs = np.polyfit(sizes, latencies, deg=1)
        a = float(coeffs[0])
        c0_est = float(coeffs[1])
    except Exception:
        # Fallback to two-point method
        S1, T1 = fit_items[0]
        S2, T2 = fit_items[1]
        a = (T2 - T1) / (S2 - S1) if S2 != S1 else 0.0
        c0_est = T1 - a * S1

    c0 = max(0.0, c0_est)

    return c0, a


# ============================================================================
# Main
# ============================================================================

def _start_server(model: str, port: int, max_model_len: int = 4096) -> subprocess.Popen:
    """Start vLLM server for prefill profiling."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.45",
        "--dtype", "float16",
        "--data-parallel-size", "1",
        "--enforce-eager",
        "--scheduling-policy", "fcfs",
    ]
    log_file = open("/tmp/ckpt_profile_server.log", "w")
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env,
    )
    proc._log_file = log_file  # type: ignore
    print(f"  Server started (pid={proc.pid})")
    return proc


def _wait_health(port: int, timeout: float = 300.0) -> bool:
    import urllib.request
    import urllib.error
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait(timeout=5)
    log_file = getattr(proc, "_log_file", None)
    if log_file:
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Profile checkpoint costs")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output", default="experiments_v2/checkpoint_cost_profile.json")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--skip-prefill", action="store_true", help="Skip prefill benchmark")
    parser.add_argument("--skip-kv", action="store_true", help="Skip KV benchmark")
    parser.add_argument("--skip-server", action="store_true",
                        help="Assume server is already running on --port")

    args = parser.parse_args()

    print("=" * 70)
    print("Checkpoint Cost Profiler")
    print("=" * 70)

    # Auto-start server for prefill benchmark (unless skipped or already running).
    server_proc = None
    if not args.skip_prefill and not args.skip_server:
        print("\n  Starting server for prefill benchmark...")
        server_proc = _start_server(args.model, args.port)
        if not _wait_health(args.port):
            print("  ERROR: Server failed to start")
            _stop_server(server_proc)
            sys.exit(1)
        print("  Server healthy")

    # Prefill benchmark
    print("\n[1/2] T_prefill(n) benchmark...")
    tokenizer_failed = False

    if args.skip_prefill:
        print("  Skipped (use --no-skip-prefill to enable)")
        prefill_data = {
            16: 1.5, 32: 2.0, 64: 4.5, 96: 7.2, 128: 10.0, 192: 16.5, 256: 24.0,
            384: 50.0, 512: 78.0, 768: 165.0, 1024: 280.0, 1536: 620.0, 2048: 1100.0,
        }
        print("  Using placeholder data (update by running without --skip-prefill)")
    else:
        profiler = PrefillBenchmark(args.model, port=args.port)
        prefill_data = profiler.run(
            token_lengths=[16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
            num_warmup=2,
            num_trials=5,
        )
        tokenizer_failed = profiler.tokenizer_failed

    # Stop server after prefill benchmark (KV benchmark doesn't need it).
    if server_proc is not None:
        print("  Stopping server...")
        _stop_server(server_proc)
        print("  Server stopped")

    # KV checkpoint benchmark
    print("\n[2/2] T_load(S) / T_ckpt(S) benchmark...")
    kv_measurement_failed = False  # Track if KV benchmark failed

    if args.skip_kv:
        print("  Skipped")
        load_data = {131072: 0.5, 262144: 1.0, 524288: 2.0, 1048576: 4.0, 2097152: 8.0, 4194304: 16.0, 8388608: 32.0}
        checkpoint_data = {131072: 0.8, 262144: 1.5, 524288: 3.0, 1048576: 6.0, 2097152: 12.0, 4194304: 24.0, 8388608: 48.0}
        print("  Using placeholder data")
    else:
        try:
            kv_bench = KVCheckpointBenchmark.from_model_name(args.model)
            block_counts = [1, 2, 4, 8, 16, 32, 64]
            save_data, restore_data = kv_bench.benchmark_save_restore(block_counts)
            # save = checkpoint (GPU→Host), restore = load (Host→GPU)
            checkpoint_data = save_data
            load_data = restore_data
        except Exception as e:
            print(f"  KV benchmark failed: {e}")
            print("  Using placeholder data")
            load_data = {131072: 0.5, 262144: 1.0, 524288: 2.0, 1048576: 4.0}
            checkpoint_data = {131072: 0.8, 262144: 1.5, 524288: 3.0, 1048576: 6.0}
            kv_measurement_failed = True  # Mark as failed

    # Estimate c0 using multiple points for robustness
    c0, a = estimate_c0_from_checkpoint_data(checkpoint_data)
    print(f"\nEstimated c0 = {c0:.2f}ms, a = {a:.6f} ms/byte")

    # Extract size-dependent part
    checkpoint_data_sizedep = {
        S: max(0.0, T - c0) for S, T in checkpoint_data.items()
    }

    # Determine if measurements are real or placeholder
    is_real_prefill = not args.skip_prefill and not tokenizer_failed  # Prefill must not be skipped AND tokenizer must load
    is_real_kv = not args.skip_kv and not kv_measurement_failed  # KV must not be skipped AND not failed
    is_real_measurement = is_real_prefill and is_real_kv

    # Build profile JSON
    profile = {
        "meta": {
            "model": args.model,
            "dtype": "float16",
            "device": "cuda",
            "block_size_tokens": 16,
            "layout": "standard_kv_cache",
            "generated_at": datetime.now().isoformat(),
            "is_real_measurement": is_real_measurement,
            "notes": (
                "REAL measurements" if is_real_measurement
                else f"PLACEHOLDER: skip_prefill={args.skip_prefill}, tokenizer_failed={tokenizer_failed}, skip_kv={args.skip_kv}, kv_measurement_failed={kv_measurement_failed}"
            ),
        },
        "prefill_ms_by_tokens": {str(n): lat for n, lat in prefill_data.items()},
        "load_ms_by_bytes": {str(S): lat for S, lat in load_data.items()},
        "checkpoint_ms_by_bytes": {str(S): lat for S, lat in checkpoint_data_sizedep.items()},
        "publication_overhead_ms": c0,
    }

    # Save to JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n✅ Profile saved to: {args.output}")
    print("\nProfile summary:")
    print(f"  - Prefill: {len(prefill_data)} points, {min(prefill_data.values()):.1f}-{max(prefill_data.values()):.1f} ms")
    print(f"  - Load: {len(load_data)} points, {min(load_data.values()):.2f}-{max(load_data.values()):.2f} ms")
    print(f"  - Checkpoint: {len(checkpoint_data)} points, {min(checkpoint_data.values()):.2f}-{max(checkpoint_data.values()):.2f} ms (total)")
    print(f"  - Publication overhead (c0): {c0:.2f} ms")
    print(f"\nNext: python experiments/inspect_checkpoint_profile.py {args.output}")


if __name__ == "__main__":
    main()
