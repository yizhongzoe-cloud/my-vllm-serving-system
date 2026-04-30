#!/usr/bin/env python3
"""Profile decode capacity and residual prefill capacity for a model.

Produces a JSON file consumed by DecodeCapacityModel.

Usage:
    python experiments_v2/profile_decode_capacity.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --port 8300 \
        --tpot-slo-ms 50 \
        --output experiments_v2/decode_capacity_profile_1b.json

IMPORTANT: Run with dp_size=1 (single replica) to measure per-replica capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import signal
import sys
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _start_server(model: str, port: int, max_model_len: int = 2048) -> subprocess.Popen:
    """Start vLLM server with dp=1, No-FT."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.55",
        "--dtype", "float16",
        "--data-parallel-size", "1",  # Must be 1 for per-replica profiling
        "--enforce-eager",
        "--scheduling-policy", "fcfs",
    ]
    log_file = open("/tmp/decode_capacity_profile_server.log", "w")
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, env=env,
    )
    proc._log_file = log_file  # type: ignore
    logger.info("Server started (pid=%d)", proc.pid)
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


async def _send_decode_requests(
    port: int,
    model: str,
    n: int,
    prompt_len: int,
    output_len: int,
    duration_sec: float,
) -> dict:
    """Send n concurrent decode requests and measure TPOT P95 at steady state."""
    prompt = "Hello " * (prompt_len // 2)

    async def send_one(session: aiohttp.ClientSession, req_id: int) -> list[float]:
        url = f"http://localhost:{port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "min_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        token_times: list[float] = []
        try:
            timeout = aiohttp.ClientTimeout(total=duration_sec + 30)
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices", [])
                    if choices:
                        # Detect token arrival via token_ids or delta content
                        token_ids = choices[0].get("token_ids") or []
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if token_ids or content:
                            token_times.append(time.time())
        except Exception:
            pass
        return token_times

    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_one(session, i) for i in range(n)]
        results = await asyncio.gather(*tasks)

    # Compute TPOT from all token inter-arrival times (skip first token = TTFT)
    all_gaps_ms: list[float] = []
    for token_times in results:
        if len(token_times) < 2:
            continue
        # Skip first token (TTFT), use rest as decode gaps
        for i in range(1, len(token_times)):
            gap = (token_times[i] - token_times[i - 1]) * 1000
            if gap < 5000:  # filter out obvious outliers (>5s)
                all_gaps_ms.append(gap)

    if not all_gaps_ms:
        return {"n": n, "tpot_p95_ms": 999.0, "tpot_p50_ms": 999.0, "samples": 0}

    import numpy as np
    arr = np.array(all_gaps_ms)
    return {
        "n": n,
        "tpot_p95_ms": float(np.percentile(arr, 95)),
        "tpot_p50_ms": float(np.percentile(arr, 50)),
        "samples": len(all_gaps_ms),
    }


def measure_decode_capacity(
    model: str,
    port: int,
    tpot_slo_ms: float,
    ctx_buckets: list[int] | None = None,
    n_candidates: list[int] | None = None,
    max_model_len: int = 2048,
) -> dict:
    """Measure decode capacity: max concurrent decode requests under SLO."""
    if ctx_buckets is None:
        ctx_buckets = [256, 1024]
    if n_candidates is None:
        n_candidates = [5, 10, 15, 20, 25, 30, 40, 50]

    results: dict = {"default": 0, "by_avg_ctx_bucket": {}}

    for ctx in ctx_buckets:
        logger.info("  Measuring decode capacity for ctx_bucket=%d ...", ctx)
        last_good = 0

        for n in n_candidates:
            logger.info("    N=%d ...", n)
            metrics = asyncio.run(_send_decode_requests(
                port, model, n, prompt_len=ctx,
                output_len=min(1500, max_model_len - ctx - 10),
                duration_sec=60.0,
            ))
            tpot = metrics["tpot_p95_ms"]
            logger.info("    N=%d: TPOT P95=%.1fms (SLO=%.1fms) %s",
                        n, tpot, tpot_slo_ms,
                        "OK" if tpot < tpot_slo_ms else "EXCEEDED")
            if tpot < tpot_slo_ms:
                last_good = n
            else:
                break

        results["by_avg_ctx_bucket"][str(ctx)] = last_good

    # Default = min across buckets (conservative)
    bucket_vals = list(results["by_avg_ctx_bucket"].values())
    results["default"] = min(bucket_vals) if bucket_vals else 10
    return results


def measure_residual_prefill_capacity(
    model: str,
    port: int,
    decode_counts: list[int],
    horizon_sec: float,
) -> dict[str, int]:
    """Measure residual prefill capacity under different decode loads."""
    results: dict[str, int] = {}

    for dc in decode_counts:
        logger.info("  Measuring residual prefill capacity with %d decode requests ...", dc)

        # Step 1: Send dc long requests to create sustained decode load
        async def run():
            connector = aiohttp.TCPConnector(limit=200)
            async with aiohttp.ClientSession(connector=connector) as session:
                # Start decode requests (background, long output)
                decode_tasks = []
                for i in range(dc):
                    decode_tasks.append(asyncio.create_task(
                        _send_one_request(session, port, model,
                                          prompt_len=256, output_len=1000)
                    ))

                # Wait for them to start decoding
                await asyncio.sleep(10.0)

                # Step 2: Measure prefill throughput
                prefill_start = time.time()
                prefill_count = 0
                prefill_tokens = 0

                while time.time() - prefill_start < 30.0:
                    # Send short prefill requests (max_tokens=1)
                    batch = []
                    for _ in range(5):
                        batch.append(asyncio.create_task(
                            _send_one_request(session, port, model,
                                              prompt_len=512, output_len=1)
                        ))
                    done = await asyncio.gather(*batch, return_exceptions=True)
                    for r in done:
                        if isinstance(r, dict) and r.get("success"):
                            prefill_count += 1
                            prefill_tokens += r.get("prompt_len", 512)

                elapsed = time.time() - prefill_start
                prefill_tput = prefill_tokens / elapsed if elapsed > 0 else 0

                # Cancel decode tasks
                for t in decode_tasks:
                    t.cancel()

                return prefill_tput

        tput = asyncio.run(run())
        rem_cap = int(tput * horizon_sec)
        logger.info("    decode_count=%d: prefill_tput=%.0f tok/s → RemPreCap=%.0f tokens",
                     dc, tput, rem_cap)
        results[str(dc)] = rem_cap

    return results


async def _send_one_request(
    session: aiohttp.ClientSession,
    port: int,
    model: str,
    prompt_len: int,
    output_len: int,
) -> dict:
    """Send a single request and return basic info."""
    prompt = "Hello " * (prompt_len // 2)
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "min_tokens": output_len,
        "stream": False,
    }
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status == 200:
                return {"success": True, "prompt_len": prompt_len}
            return {"success": False}
    except Exception:
        return {"success": False}


def main():
    parser = argparse.ArgumentParser(
        description="Profile decode capacity and residual prefill capacity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--tpot-slo-ms", type=float, default=50.0)
    parser.add_argument("--horizon", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-server", action="store_true",
                        help="Assume server is already running on --port")
    args = parser.parse_args()

    proc = None
    if not args.skip_server:
        logger.info("Starting server (dp=1) ...")
        proc = _start_server(args.model, args.port, args.max_model_len)
        if not _wait_health(args.port):
            logger.error("Server failed to start")
            if proc:
                _stop_server(proc)
            sys.exit(1)
        logger.info("Server healthy")

    try:
        # Measure decode capacity
        logger.info("=== Measuring decode capacity ===")
        decode_cap = measure_decode_capacity(
            args.model, args.port, args.tpot_slo_ms,
            max_model_len=args.max_model_len)

        # Measure residual prefill capacity
        logger.info("=== Measuring residual prefill capacity ===")
        max_dc = decode_cap["default"]
        decode_counts = [0]
        step = max(1, max_dc // 5)
        for i in range(step, max_dc + 1, step):
            decode_counts.append(i)
        if max_dc not in decode_counts:
            decode_counts.append(max_dc)

        residual_prefill = measure_residual_prefill_capacity(
            args.model, args.port, decode_counts, args.horizon)

        # Build profile
        import subprocess as sp
        try:
            gpu_name = sp.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=sp.DEVNULL,
            ).decode().strip().split("\n")[0]
        except Exception:
            gpu_name = "unknown"

        profile = {
            "meta": {
                "model": args.model,
                "gpu": gpu_name,
                "dp_size": 1,
                "max_model_len": args.max_model_len,
                "tpot_slo_ms": args.tpot_slo_ms,
                "planning_horizon_sec": args.horizon,
            },
            "decode_capacity": decode_cap,
            "residual_prefill_capacity": residual_prefill,
        }

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info("Profile saved to %s", args.output)
        logger.info("  Decode capacity (default): %d", decode_cap["default"])
        logger.info("  Residual prefill points: %d", len(residual_prefill))

    finally:
        if proc:
            _stop_server(proc)


if __name__ == "__main__":
    main()
