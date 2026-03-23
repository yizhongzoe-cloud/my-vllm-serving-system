#!/usr/bin/env python3
"""Fault-Tolerant Benchmark Data Collector.

Runs multiple FT E2E experiments with varying parameters and collects
structured results into JSON + CSV for downstream analysis/plotting.

Two modes:
  --mode client-reroute  (default, legacy)
      Launches two independent vLLM servers (one per GPU). On failure,
      the *client* detects the HTTP error and retries on the surviving
      server. This does NOT exercise the internal FT recovery path.

  --mode internal-recovery
      Launches a single vLLM server with --data-parallel-size 2 and
      FT enabled (--max-gpu-failures 1 --enable-checkpointing). The
      FTCoordinator / FTDPAsyncMPClient handle failure detection and
      checkpoint-based recovery internally. The client sends all
      requests to a single endpoint and observes the recovery
      transparently (failover gap shows up as increased latency, not
      client-side retry).

Experiment dimensions:
  - Baseline (no failure) vs. failure scenarios
  - Different kill timings
  - Different request counts / load levels

Usage:
    # Internal recovery (recommended for FT benchmarking)
    python slo_benchmark/scripts/collect_ft_data.py --model meta-llama/Llama-3.2-1B-Instruct --mode internal-recovery --quick

    # Client-side reroute (legacy)
    python slo_benchmark/scripts/collect_ft_data.py --model meta-llama/Llama-3.2-1B-Instruct --mode client-reroute --quick

    # Full sweep
    python slo_benchmark/scripts/collect_ft_data.py --model meta-llama/Llama-3.2-1B-Instruct --mode internal-recovery

    # Custom output directory
    python slo_benchmark/scripts/collect_ft_data.py --model meta-llama/Llama-3.2-1B-Instruct -o data/my_exp
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerRequestRecord:
    """One row per request."""
    experiment_id: str
    experiment_type: str  # "baseline" | "failure"
    num_requests: int
    kill_after_sec: float | None
    request_id: str
    replica_id: int
    success: bool
    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    total_tokens: int = 0
    was_rerouted: bool = False
    error: str | None = None


@dataclass
class ExperimentSummary:
    """One row per experiment."""
    experiment_id: str
    experiment_type: str
    model: str
    num_requests: int
    kill_after_sec: float | None
    max_tokens: int
    max_model_len: int

    # Results
    completed: int = 0
    failed: int = 0
    rerouted: int = 0
    total_tokens: int = 0
    test_duration_sec: float = 0.0

    # Throughput
    goodput_tok_per_sec: float = 0.0
    request_throughput_rps: float = 0.0

    # Latency (successful requests)
    ttft_mean_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    e2e_mean_ms: float = 0.0
    e2e_p50_ms: float = 0.0
    e2e_p90_ms: float = 0.0
    e2e_p99_ms: float = 0.0

    # Failure-specific
    failover_gap_ms: float | None = None
    success_rate: float = 0.0


# ---------------------------------------------------------------------------
# Server helpers  (reused from test_ft_e2e_serving.py logic)
# ---------------------------------------------------------------------------

def start_vllm_server(model: str, gpu_id: int, port: int,
                      max_model_len: int = 512) -> subprocess.Popen:
    """Start a single-GPU vLLM server (used in client-reroute mode)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.45",
        "--dtype", "float16",
        "--enforce-eager",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)


def start_dp2_ft_server(
    model: str,
    port: int,
    max_model_len: int = 512,
    failure_timeout_sec: float = 5.0,
) -> subprocess.Popen:
    """Start a DP=2 vLLM server with FT enabled (internal-recovery mode).

    Uses --data-parallel-size 2 so two engine replicas run in-process.
    The FTCoordinator handles failure detection and recovery internally.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.45",
        "--dtype", "float16",
        "--enforce-eager",
        "--data-parallel-size", "2",
        "--max-gpu-failures", "1",
        "--enable-checkpointing",
        "--failure-timeout-sec", str(failure_timeout_sec),
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)


async def wait_for_server(port: int, timeout: float = 180.0) -> bool:
    url = f"http://localhost:{port}/health"
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)
    return False


async def send_chat_request(port: int, model: str, prompt: str,
                            max_tokens: int = 80) -> dict:
    """Send one chat completion request. Returns raw timing dict."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"success": False, "error": f"HTTP {resp.status}: {text[:200]}",
                            "ttft_ms": None, "e2e_latency_ms": None, "total_tokens": 0}
                ttft = (time.time() - start) * 1000
                data = await resp.json()
                e2e = (time.time() - start) * 1000
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                return {"success": True, "ttft_ms": ttft, "e2e_latency_ms": e2e,
                        "total_tokens": tokens, "error": None}
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {"success": False, "error": str(e),
                "ttft_ms": None, "e2e_latency_ms": None, "total_tokens": 0}


def kill_dp_engine_worker(server_proc: subprocess.Popen, target_rank: int = 1) -> bool:
    """Kill a specific DP engine worker child process.

    In DP mode, the main server process spawns child processes for each
    engine. We find the child that owns GPU `target_rank` and SIGKILL it.
    The FTCoordinator should detect the failure and trigger recovery.

    Returns True if a child was killed.
    """
    try:
        import psutil
    except ImportError:
        # Fallback: can't find children without psutil, skip kill
        print("  WARNING: psutil not installed, cannot kill DP worker. "
              "Install with: pip install psutil")
        return False

    parent = psutil.Process(server_proc.pid)
    children = parent.children(recursive=True)
    if not children:
        print(f"  WARNING: No child processes found for server PID {server_proc.pid}")
        return False

    # In DP mode, each EngineCore runs in a child process.
    # We kill the child that corresponds to the target rank.
    # Heuristic: children are spawned in order of DP rank,
    # so the (target_rank+1)-th child process is the engine worker.
    # A more robust approach checks /proc/<pid>/environ for RANK, but
    # the spawn-order heuristic works for the common case.
    engine_children = [c for c in children if c.is_running()]
    if target_rank < len(engine_children):
        target = engine_children[target_rank]
        print(f"  >>> Killing DP engine worker PID {target.pid} "
              f"(rank {target_rank}) <<<")
        target.kill()
        target.wait(timeout=5)
        return True
    else:
        print(f"  WARNING: target_rank={target_rank} but only "
              f"{len(engine_children)} children found. "
              f"Killing last child as fallback.")
        engine_children[-1].kill()
        engine_children[-1].wait(timeout=5)
        return True


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

PROMPTS = [
    "Write a short poem about the ocean.",
    "Explain how photosynthesis works in simple terms.",
    "What are three interesting facts about Mars?",
    "Describe a day in the life of a medieval blacksmith.",
    "How does a computer CPU work?",
    "Tell me about the history of the internet in a few sentences.",
    "What is quantum computing and why does it matter?",
    "Summarize the plot of Romeo and Juliet.",
]


async def run_single_experiment(
    experiment_id: str,
    experiment_type: str,  # "baseline" | "failure"
    model: str,
    num_requests: int,
    kill_after_sec: float | None,
    max_tokens: int,
    ports: list[int],
    server_procs: list[subprocess.Popen] | None = None,
) -> tuple[ExperimentSummary, list[PerRequestRecord], bool]:
    """Run one experiment and return (summary, per_request_records, was_killed).

    Args:
        server_procs: List of server Popen objects. If provided, the server
            for the killed replica will be SIGKILL'd during the experiment
            (not just removed from client-side routing).
    """

    records: list[PerRequestRecord] = []
    test_start = time.time()
    killed = False
    kill_replica = 1
    failure_time = None
    first_reroute_time = None

    # Simple round-robin among available ports
    available_ports = list(ports)  # [port0, port1]
    rr_idx = 0

    async def get_port() -> tuple[int, int]:
        nonlocal rr_idx
        idx = rr_idx % len(available_ports)
        rr_idx += 1
        port = available_ports[idx]
        replica_id = ports.index(port)
        return replica_id, port

    tasks = []

    async def do_request(idx: int):
        nonlocal killed, first_reroute_time
        prompt = PROMPTS[idx % len(PROMPTS)]
        rid, port = await get_port()
        res = await send_chat_request(port, model, prompt, max_tokens)
        was_rerouted = False

        # Reroute if request hit the killed replica
        if not res["success"] and killed and rid == kill_replica:
            was_rerouted = True
            # Only route to surviving replica
            survivor_port = ports[0]
            res = await send_chat_request(survivor_port, model, prompt, max_tokens)
            rid = 0
            if first_reroute_time is None:
                first_reroute_time = time.time()

        records.append(PerRequestRecord(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            num_requests=num_requests,
            kill_after_sec=kill_after_sec,
            request_id=f"req-{idx}",
            replica_id=rid,
            success=res["success"],
            ttft_ms=res["ttft_ms"],
            e2e_latency_ms=res["e2e_latency_ms"],
            total_tokens=res["total_tokens"],
            was_rerouted=was_rerouted,
            error=res["error"],
        ))

    # Send requests with 0.5s stagger
    for i in range(num_requests):
        tasks.append(asyncio.create_task(do_request(i)))
        elapsed = time.time() - test_start
        if experiment_type == "failure" and not killed and kill_after_sec is not None and elapsed >= kill_after_sec:
            # Actually kill the GPU 1 server process NOW, not after the loop.
            killed = True
            failure_time = time.time()
            if server_procs is not None and server_procs[kill_replica].poll() is None:
                server_procs[kill_replica].send_signal(signal.SIGKILL)
                server_procs[kill_replica].wait()
                print(f"    >>> KILLED server on GPU {kill_replica} at "
                      f"t={elapsed:.1f}s <<<")
            # Remove killed port from round-robin
            if ports[kill_replica] in available_ports:
                available_ports.remove(ports[kill_replica])
        await asyncio.sleep(0.5)

    await asyncio.gather(*tasks, return_exceptions=True)
    test_duration = time.time() - test_start

    # Compute summary stats
    ok_records = [r for r in records if r.success]
    ttfts = [r.ttft_ms for r in ok_records if r.ttft_ms is not None]
    e2es = [r.e2e_latency_ms for r in ok_records if r.e2e_latency_ms is not None]

    import numpy as np

    def pctl(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    total_tokens = sum(r.total_tokens for r in ok_records)
    failover_gap = None
    if failure_time and first_reroute_time:
        failover_gap = (first_reroute_time - failure_time) * 1000

    summary = ExperimentSummary(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        model=model,
        num_requests=num_requests,
        kill_after_sec=kill_after_sec,
        max_tokens=max_tokens,
        max_model_len=512,
        completed=len(ok_records),
        failed=len(records) - len(ok_records),
        rerouted=sum(1 for r in records if r.was_rerouted),
        total_tokens=total_tokens,
        test_duration_sec=test_duration,
        goodput_tok_per_sec=total_tokens / test_duration if test_duration > 0 else 0,
        request_throughput_rps=len(ok_records) / test_duration if test_duration > 0 else 0,
        ttft_mean_ms=float(np.mean(ttfts)) if ttfts else 0,
        ttft_p50_ms=pctl(ttfts, 50),
        ttft_p90_ms=pctl(ttfts, 90),
        ttft_p99_ms=pctl(ttfts, 99),
        e2e_mean_ms=float(np.mean(e2es)) if e2es else 0,
        e2e_p50_ms=pctl(e2es, 50),
        e2e_p90_ms=pctl(e2es, 90),
        e2e_p99_ms=pctl(e2es, 99),
        failover_gap_ms=failover_gap,
        success_rate=len(ok_records) / len(records) if records else 0,
    )
    return summary, records, killed


async def run_internal_recovery_experiment(
    experiment_id: str,
    experiment_type: str,  # "baseline" | "failure"
    model: str,
    num_requests: int,
    kill_after_sec: float | None,
    max_tokens: int,
    port: int,
    server_proc: subprocess.Popen | None = None,
) -> tuple[ExperimentSummary, list[PerRequestRecord], bool]:
    """Run one experiment against a DP=2 FT server (internal recovery mode).

    All requests go to a single endpoint. On failure, the FTCoordinator
    handles detection and recovery internally — the client does NOT retry.
    In-flight requests on the failed engine are recovered via checkpoint
    restore and show up as increased latency, not HTTP errors.
    """
    import numpy as np

    records: list[PerRequestRecord] = []
    test_start = time.time()
    killed = False
    failure_time = None

    tasks = []

    async def do_request(idx: int):
        prompt = PROMPTS[idx % len(PROMPTS)]
        res = await send_chat_request(port, model, prompt, max_tokens)
        # In internal-recovery mode, requests are never client-rerouted.
        # The FT system handles recovery transparently.
        records.append(PerRequestRecord(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            num_requests=num_requests,
            kill_after_sec=kill_after_sec,
            request_id=f"req-{idx}",
            replica_id=-1,  # DP routing is internal; we don't know the replica.
            success=res["success"],
            ttft_ms=res["ttft_ms"],
            e2e_latency_ms=res["e2e_latency_ms"],
            total_tokens=res["total_tokens"],
            was_rerouted=False,  # Recovery is internal, not client-side.
            error=res["error"],
        ))

    # Send requests with 0.5s stagger
    for i in range(num_requests):
        tasks.append(asyncio.create_task(do_request(i)))
        elapsed = time.time() - test_start

        # Kill a DP engine worker at the scheduled time.
        if (experiment_type == "failure" and not killed
                and kill_after_sec is not None
                and elapsed >= kill_after_sec
                and server_proc is not None):
            killed = True
            failure_time = time.time()
            kill_dp_engine_worker(server_proc, target_rank=1)
            print(f"    >>> Killed DP engine worker at t={elapsed:.1f}s <<<")

        await asyncio.sleep(0.5)

    await asyncio.gather(*tasks, return_exceptions=True)
    test_duration = time.time() - test_start

    # Compute summary stats
    ok_records = [r for r in records if r.success]
    fail_records = [r for r in records if not r.success]
    ttfts = [r.ttft_ms for r in ok_records if r.ttft_ms is not None]
    e2es = [r.e2e_latency_ms for r in ok_records if r.e2e_latency_ms is not None]

    def pctl(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    total_tokens = sum(r.total_tokens for r in ok_records)

    # Estimate failover gap from latency spike.
    # In internal-recovery mode, there's no explicit "reroute" event.
    # Instead, the failover gap manifests as a latency increase for
    # requests that were in-flight during the failure. We estimate it
    # as the difference between the max e2e latency after failure and
    # the median e2e latency before failure.
    failover_gap = None
    if failure_time and e2es and len(e2es) > 1:
        # Requests started before kill: their e2e includes recovery delay
        pre_fail_e2es = []
        post_fail_e2es = []
        for r in ok_records:
            if r.e2e_latency_ms is not None:
                # Heuristic: requests started before kill_after_sec
                req_idx = int(r.request_id.split("-")[1])
                req_start_approx = req_idx * 0.5  # 0.5s stagger
                if kill_after_sec and req_start_approx < kill_after_sec:
                    pre_fail_e2es.append(r.e2e_latency_ms)
                else:
                    post_fail_e2es.append(r.e2e_latency_ms)
        if pre_fail_e2es and post_fail_e2es:
            # Gap ≈ max post-failure latency - median pre-failure latency
            median_pre = float(np.median(pre_fail_e2es))
            max_post = max(post_fail_e2es)
            if max_post > median_pre:
                failover_gap = max_post - median_pre

    summary = ExperimentSummary(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        model=model,
        num_requests=num_requests,
        kill_after_sec=kill_after_sec,
        max_tokens=max_tokens,
        max_model_len=512,
        completed=len(ok_records),
        failed=len(fail_records),
        rerouted=0,  # Internal recovery, not client-side reroute.
        total_tokens=total_tokens,
        test_duration_sec=test_duration,
        goodput_tok_per_sec=total_tokens / test_duration if test_duration > 0 else 0,
        request_throughput_rps=len(ok_records) / test_duration if test_duration > 0 else 0,
        ttft_mean_ms=float(np.mean(ttfts)) if ttfts else 0,
        ttft_p50_ms=pctl(ttfts, 50),
        ttft_p90_ms=pctl(ttfts, 90),
        ttft_p99_ms=pctl(ttfts, 99),
        e2e_mean_ms=float(np.mean(e2es)) if e2es else 0,
        e2e_p50_ms=pctl(e2es, 50),
        e2e_p90_ms=pctl(e2es, 90),
        e2e_p99_ms=pctl(e2es, 99),
        failover_gap_ms=failover_gap,
        success_rate=len(ok_records) / len(records) if records else 0,
    )
    return summary, records, killed


# ---------------------------------------------------------------------------
# Main orchestrators
# ---------------------------------------------------------------------------

async def run_all_experiments_internal_recovery(
    model: str,
    output_dir: Path,
    max_tokens: int,
    max_model_len: int,
    quick: bool = False,
) -> None:
    """Run experiment sweep using DP=2 FT server (internal recovery)."""

    if quick:
        experiments = [
            {"type": "baseline", "num_requests": 10, "kill_after": None},
            {"type": "failure", "num_requests": 15, "kill_after": 3.0},
            {"type": "failure", "num_requests": 20, "kill_after": 5.0},
        ]
    else:
        experiments = [
            {"type": "baseline", "num_requests": 10, "kill_after": None},
            {"type": "baseline", "num_requests": 20, "kill_after": None},
            {"type": "baseline", "num_requests": 30, "kill_after": None},
            {"type": "failure", "num_requests": 20, "kill_after": 3.0},
            {"type": "failure", "num_requests": 30, "kill_after": 3.0},
            {"type": "failure", "num_requests": 20, "kill_after": 5.0},
            {"type": "failure", "num_requests": 30, "kill_after": 5.0},
            {"type": "failure", "num_requests": 30, "kill_after": 8.0},
            {"type": "failure", "num_requests": 30, "kill_after": 10.0},
        ]

    port = 8100
    all_summaries: list[ExperimentSummary] = []
    all_records: list[PerRequestRecord] = []

    print("=" * 60)
    print(f"Starting DP=2 FT vLLM server ({model})...")
    print("  Mode: internal-recovery (FTCoordinator handles failover)")
    print("=" * 60)
    proc = start_dp2_ft_server(model, port, max_model_len)
    print(f"  DP=2 server → port {port}")

    print("  Waiting for server...")
    if not await wait_for_server(port, timeout=300.0):
        print("  FAILED to start DP=2 FT server. Aborting.")
        proc.kill()
        return
    print("  Server READY.\n")

    for exp_idx, exp in enumerate(experiments):
        exp_id = f"exp_{exp_idx:03d}_{exp['type']}_n{exp['num_requests']}"
        if exp["kill_after"] is not None:
            exp_id += f"_kill{exp['kill_after']:.0f}s"

        print(f"[{exp_idx+1}/{len(experiments)}] {exp_id}")
        print(f"  type={exp['type']}, requests={exp['num_requests']}, "
              f"kill_after={exp['kill_after']}")

        summary, records, was_killed = await run_internal_recovery_experiment(
            experiment_id=exp_id,
            experiment_type=exp["type"],
            model=model,
            num_requests=exp["num_requests"],
            kill_after_sec=exp["kill_after"],
            max_tokens=max_tokens,
            port=port,
            server_proc=proc,
        )

        all_summaries.append(summary)
        all_records.extend(records)

        print(f"  → completed={summary.completed}, failed={summary.failed}, "
              f"goodput={summary.goodput_tok_per_sec:.1f} tok/s")
        if summary.failover_gap_ms is not None:
            print(f"  → failover_gap≈{summary.failover_gap_ms:.1f}ms (estimated)")

        # After killing a DP worker, the server may need to be restarted
        # since the dead engine cannot be revived in the current architecture.
        if was_killed:
            print("  Restarting DP=2 FT server (engine worker was killed)...")
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            proc = start_dp2_ft_server(model, port, max_model_len)
            if await wait_for_server(port, timeout=300.0):
                print("  DP=2 FT server restarted.")
            else:
                print("  WARNING: DP=2 FT server failed to restart. "
                      "Remaining experiments may be affected.")
        print()

    # Save results
    _save_results(output_dir, model, max_tokens, max_model_len,
                  all_summaries, all_records)

    # Cleanup
    print("\nCleaning up server...")
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("Done.")


async def run_all_experiments_client_reroute(
    model: str,
    output_dir: Path,
    max_tokens: int,
    max_model_len: int,
    quick: bool = False,
) -> None:
    """Run experiment sweep using two independent servers (client reroute).

    This is the legacy mode where the client detects failures and retries.
    """

    # Define experiment plan
    if quick:
        experiments = [
            {"type": "baseline", "num_requests": 10, "kill_after": None},
            {"type": "failure", "num_requests": 15, "kill_after": 3.0},
            {"type": "failure", "num_requests": 20, "kill_after": 5.0},
        ]
    else:
        experiments = [
            # Baselines (no failure)
            {"type": "baseline", "num_requests": 10, "kill_after": None},
            {"type": "baseline", "num_requests": 20, "kill_after": None},
            {"type": "baseline", "num_requests": 30, "kill_after": None},
            # Failure: early kill
            {"type": "failure", "num_requests": 20, "kill_after": 3.0},
            {"type": "failure", "num_requests": 30, "kill_after": 3.0},
            # Failure: mid kill
            {"type": "failure", "num_requests": 20, "kill_after": 5.0},
            {"type": "failure", "num_requests": 30, "kill_after": 5.0},
            # Failure: late kill
            {"type": "failure", "num_requests": 30, "kill_after": 8.0},
            {"type": "failure", "num_requests": 30, "kill_after": 10.0},
        ]

    ports = [8100, 8101]
    all_summaries: list[ExperimentSummary] = []
    all_records: list[PerRequestRecord] = []

    # --- Start servers ---
    print("=" * 60)
    print(f"Starting 2 vLLM servers ({model})...")
    print("=" * 60)
    procs = []
    for gpu_id, port in enumerate(ports):
        proc = start_vllm_server(model, gpu_id, port, max_model_len)
        procs.append(proc)
        print(f"  GPU {gpu_id} → port {port}")

    print("  Waiting for servers...")
    for port in ports:
        if not await wait_for_server(port, timeout=180.0):
            print(f"  FAILED to start server on port {port}. Aborting.")
            for p in procs:
                p.kill()
            return
    print("  Both servers READY.\n")

    # --- Run experiments ---
    for exp_idx, exp in enumerate(experiments):
        exp_id = f"exp_{exp_idx:03d}_{exp['type']}_n{exp['num_requests']}"
        if exp["kill_after"] is not None:
            exp_id += f"_kill{exp['kill_after']:.0f}s"

        print(f"[{exp_idx+1}/{len(experiments)}] {exp_id}")
        print(f"  type={exp['type']}, requests={exp['num_requests']}, "
              f"kill_after={exp['kill_after']}")

        # For failure experiments, pass server_procs so the server is
        # actually killed during the experiment (not just client-side).
        need_restart = False
        if exp["type"] == "failure":
            summary, records, was_killed = await run_single_experiment(
                experiment_id=exp_id,
                experiment_type=exp["type"],
                model=model,
                num_requests=exp["num_requests"],
                kill_after_sec=exp["kill_after"],
                max_tokens=max_tokens,
                ports=ports,
                server_procs=procs,
            )
            if was_killed:
                need_restart = True
        else:
            summary, records, _ = await run_single_experiment(
                experiment_id=exp_id,
                experiment_type=exp["type"],
                model=model,
                num_requests=exp["num_requests"],
                kill_after_sec=None,
                max_tokens=max_tokens,
                ports=ports,
            )

        all_summaries.append(summary)
        all_records.extend(records)

        print(f"  → completed={summary.completed}, failed={summary.failed}, "
              f"rerouted={summary.rerouted}, "
              f"goodput={summary.goodput_tok_per_sec:.1f} tok/s")
        if summary.failover_gap_ms is not None:
            print(f"  → failover_gap={summary.failover_gap_ms:.1f}ms")

        # Restart GPU 1 if killed
        if need_restart:
            print("  Restarting GPU 1 server...")
            procs[1] = start_vllm_server(model, 1, ports[1], max_model_len)
            if await wait_for_server(ports[1], timeout=180.0):
                print("  GPU 1 server restarted.")
            else:
                print("  WARNING: GPU 1 failed to restart. "
                      "Remaining failure experiments may be affected.")
        print()

    # Save results
    _save_results(output_dir, model, max_tokens, max_model_len,
                  all_summaries, all_records)

    # Cleanup
    print("\nCleaning up servers...")
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    print("Done.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_results(
    output_dir: Path,
    model: str,
    max_tokens: int,
    max_model_len: int,
    all_summaries: list[ExperimentSummary],
    all_records: list[PerRequestRecord],
) -> None:
    """Save experiment results to CSV and JSON."""
    if not all_summaries or not all_records:
        print("No results to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Summaries CSV
    summary_csv = output_dir / f"experiment_summaries_{timestamp}.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_summaries[0]).keys()))
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(asdict(s))
    print(f"Summaries CSV → {summary_csv}")

    # 2. Per-request CSV
    records_csv = output_dir / f"per_request_{timestamp}.csv"
    with open(records_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))
    print(f"Per-request CSV → {records_csv}")

    # 3. Full JSON (for programmatic use)
    full_json = output_dir / f"full_results_{timestamp}.json"
    with open(full_json, "w") as f:
        json.dump({
            "metadata": {
                "model": model,
                "max_tokens": max_tokens,
                "max_model_len": max_model_len,
                "timestamp": timestamp,
                "num_experiments": len(all_summaries),
            },
            "summaries": [asdict(s) for s in all_summaries],
            "per_request": [asdict(r) for r in all_records],
        }, f, indent=2)
    print(f"Full JSON    → {full_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect FT benchmark data across multiple experiments"
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("-o", "--output", default="slo_benchmark/data/results",
                        help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--quick", action="store_true",
                        help="Run fewer experiments for quick validation")
    parser.add_argument("--mode", choices=["internal-recovery", "client-reroute"],
                        default="internal-recovery",
                        help="Recovery mode: 'internal-recovery' uses DP=2 FT "
                             "server with in-engine failover (recommended); "
                             "'client-reroute' uses two independent servers "
                             "with client-side retry (legacy)")
    args = parser.parse_args()

    print("=" * 60)
    print("FT Benchmark Data Collector")
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Output:       {args.output}")
    print(f"  Mode:         {args.mode}")
    print(f"  Quick mode:   {args.quick}")
    print()

    if args.mode == "internal-recovery":
        asyncio.run(run_all_experiments_internal_recovery(
            model=args.model,
            output_dir=Path(args.output),
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            quick=args.quick,
        ))
    else:
        asyncio.run(run_all_experiments_client_reroute(
            model=args.model,
            output_dir=Path(args.output),
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            quick=args.quick,
        ))


if __name__ == "__main__":
    main()
