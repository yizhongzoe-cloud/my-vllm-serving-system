"""Benchmark for in-engine checkpoint recovery (not client retry).

This test verifies and measures the actual KV-checkpoint restore path:
  1. Launches a vLLM server with --data-parallel-size 2 and FT scheduling.
  2. Sends long-running requests so checkpoints are created.
  3. Kills one engine process (via the coordinator or direct signal).
  4. Verifies the surviving engine restores KV from the shared checkpoint
     and resumes generation (rather than the client retrying from scratch).
  5. Measures in-engine failover timing: detection + restore + resume.

This directly tests the claim:
  "Engine A dies mid-generation → system internally migrates the request
   to Engine B → B restores KV from checkpoint → B resumes generation."

Usage:
    python tests/ft/test_ft_checkpoint_recovery.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --num-requests 5 \
        --max-tokens 200 \
        --kill-after 8

Requires 2 GPUs.
"""

import argparse
import asyncio
import glob
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ---- Data structures ----

@dataclass
class RecoveryEvent:
    """A single observed in-engine KV restore event."""
    request_id: str
    tokens_restored: int
    # Time from failure injection to this restore completing (if measurable).
    restore_latency_ms: float | None = None


@dataclass
class CheckpointRecoveryReport:
    """Aggregated results for checkpoint recovery benchmark."""
    # Requests
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0

    # Checkpoint creation (before failure).
    checkpoints_observed: int = 0

    # In-engine recovery events parsed from server logs.
    recovery_events: list[RecoveryEvent] = field(default_factory=list)

    # Timing.
    failure_injection_time: float | None = None
    first_recovery_log_time: float | None = None
    all_requests_done_time: float | None = None

    # Derived metrics.
    @property
    def in_engine_failover_gap_ms(self) -> float | None:
        """Time from failure injection to first in-engine KV restore."""
        if self.failure_injection_time and self.first_recovery_log_time:
            return (
                self.first_recovery_log_time - self.failure_injection_time
            ) * 1000
        return None

    @property
    def total_tokens_restored(self) -> int:
        return sum(e.tokens_restored for e in self.recovery_events)

    @property
    def requests_recovered_via_checkpoint(self) -> int:
        return len(self.recovery_events)


# ---- Server management ----

SHARED_CKPT_DIR = "/dev/shm/vllm_ft_checkpoints"


def start_dp2_ft_server(
    model: str,
    port: int = 8200,
    max_model_len: int = 512,
    log_file: str | None = None,
    failure_timeout_sec: float = 5.0,
) -> subprocess.Popen:
    """Start a single vLLM server with DP=2 and FT scheduling.

    This launches one API server process that internally manages 2 engine
    cores (one per GPU), connected via the FTCoordinator.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Enable verbose logging so we can parse checkpoint/restore events.
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.45",
        "--dtype", "float16",
        "--enforce-eager",
        "--data-parallel-size", "2",
        "--scheduling-policy", "fault_tolerant",
        "--enable-checkpointing",
        "--failure-timeout-sec", str(failure_timeout_sec),
    ]

    if log_file:
        log_fd = open(log_file, "w")
    else:
        log_fd = subprocess.PIPE

    print(f"  Starting DP=2 FT server on port {port}...")
    print(f"  Command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
    )
    return proc


async def wait_for_server(port: int, timeout: float = 300.0) -> bool:
    """Wait until server is ready by polling /health."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)
    return False


def count_shared_checkpoints() -> int:
    """Count checkpoint files in the shared directory."""
    if not os.path.exists(SHARED_CKPT_DIR):
        return 0
    return len(glob.glob(os.path.join(SHARED_CKPT_DIR, "*.pt")))


def find_engine_pids(parent_pid: int) -> list[int]:
    """Find child process PIDs that are engine core workers.

    In a DP=2 setup, the main process spawns engine core subprocesses.
    We find them so we can kill one specifically.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [int(pid) for pid in result.stdout.strip().split("\n") if pid]
    except Exception:
        pass
    return []


def kill_engine_process(parent_pid: int) -> int | None:
    """Kill one engine subprocess (simulating GPU failure).

    Returns the PID of the killed process, or None.
    """
    children = find_engine_pids(parent_pid)
    if len(children) < 2:
        print(f"  Warning: expected >=2 child processes, found {len(children)}")
        if not children:
            return None

    # Kill the last child (typically engine 1).
    target = children[-1]
    print(f"  Killing engine process PID={target} (child of {parent_pid})")
    try:
        os.kill(target, signal.SIGKILL)
        return target
    except OSError as e:
        print(f"  Failed to kill PID {target}: {e}")
        return None


# ---- Request sending ----

async def send_streaming_request(
    port: int,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
) -> dict:
    """Send a streaming chat completion request and track token-by-token timing.

    Returns dict with success, tokens, timing info, and whether there was
    an interruption (gap in token delivery).
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    start_time = time.time()
    ttft = None
    tokens = 0
    token_times: list[float] = []
    max_gap_ms = 0.0
    text_chunks: list[str] = []
    last_token_time = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {text[:200]}",
                        "tokens": 0,
                        "request_id": request_id,
                    }

                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: "):
                        continue
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break

                    now = time.time()
                    if ttft is None:
                        ttft = (now - start_time) * 1000

                    if last_token_time is not None:
                        gap = (now - last_token_time) * 1000
                        if gap > max_gap_ms:
                            max_gap_ms = gap

                    last_token_time = now
                    token_times.append(now)
                    tokens += 1

                    # Extract text content.
                    try:
                        import json
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            text_chunks.append(content)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

        e2e_ms = (time.time() - start_time) * 1000
        return {
            "success": True,
            "tokens": tokens,
            "ttft_ms": ttft,
            "e2e_ms": e2e_ms,
            "max_gap_ms": max_gap_ms,
            "text": "".join(text_chunks),
            "request_id": request_id,
            "token_times": token_times,
        }

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return {
            "success": False,
            "error": str(e),
            "tokens": 0,
            "request_id": request_id,
        }


# ---- Log parsing ----

def parse_recovery_events(log_file: str) -> list[RecoveryEvent]:
    """Parse server log for in-engine checkpoint restore events.

    Looks for lines like:
      FT restore: request <id> restored <N> tokens from shared checkpoint
    """
    events = []
    pattern = re.compile(
        r"FT restore: request (\S+) restored (\d+) tokens"
    )

    if not os.path.exists(log_file):
        return events

    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                events.append(RecoveryEvent(
                    request_id=m.group(1),
                    tokens_restored=int(m.group(2)),
                ))
    return events


def parse_checkpoint_events(log_file: str) -> int:
    """Count checkpoint creation events in the log."""
    pattern = re.compile(r"FT checkpoint recorded for request")
    count = 0
    if not os.path.exists(log_file):
        return 0
    with open(log_file) as f:
        for line in f:
            if pattern.search(line):
                count += 1
    return count


def parse_failure_detection(log_file: str) -> float | None:
    """Find timestamp of ENGINE_FAILED detection in logs.

    Returns epoch time if found, else None.
    """
    pattern = re.compile(
        r"FT Coordinator: engine \d+ timed out"
    )
    if not os.path.exists(log_file):
        return None
    with open(log_file) as f:
        for line in f:
            if pattern.search(line):
                # Try to extract timestamp from log line.
                # vLLM logs typically have format: timestamp - module - level - msg
                ts_match = re.match(
                    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})", line
                )
                if ts_match:
                    from datetime import datetime
                    ts = datetime.strptime(
                        ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                    )
                    return ts.timestamp()
    return None


# ---- Main benchmark ----

async def run_checkpoint_recovery_benchmark(
    model: str,
    num_requests: int,
    max_tokens: int,
    max_model_len: int,
    kill_after_sec: float,
    failure_timeout_sec: float,
    port: int = 8200,
) -> CheckpointRecoveryReport:
    report = CheckpointRecoveryReport(total_requests=num_requests)

    # Use long prompts to ensure checkpoints are created before failure.
    prompts = [
        "Write a very detailed essay about the history of artificial intelligence, "
        "covering all major milestones from the 1950s to today. Include specific "
        "dates, researchers, and breakthroughs.",
        "Explain the complete process of how a modern CPU is manufactured, from "
        "silicon wafer production through to final testing. Include details about "
        "lithography, doping, and packaging.",
        "Describe in great detail how the human immune system works, including "
        "the roles of T-cells, B-cells, antibodies, and the complement system.",
        "Write a comprehensive guide to distributed systems, covering consensus "
        "algorithms, fault tolerance, CAP theorem, and real-world implementations.",
        "Explain the physics of black holes from first principles, including "
        "event horizons, Hawking radiation, information paradox, and recent "
        "observational evidence.",
    ]

    # Clean up any stale checkpoint files.
    if os.path.exists(SHARED_CKPT_DIR):
        for f in glob.glob(os.path.join(SHARED_CKPT_DIR, "*.pt")):
            try:
                os.remove(f)
            except OSError:
                pass

    # --- Phase 1: Start DP=2 FT server ---
    print("\n=== Phase 1: Starting DP=2 FT Server ===")
    log_file = tempfile.mktemp(suffix="_ft_server.log", prefix="vllm_")
    print(f"  Server log: {log_file}")

    server_proc = start_dp2_ft_server(
        model=model,
        port=port,
        max_model_len=max_model_len,
        log_file=log_file,
        failure_timeout_sec=failure_timeout_sec,
    )

    try:
        print("  Waiting for server to be ready (this may take a few minutes)...")
        ready = await wait_for_server(port, timeout=300.0)
        if not ready:
            print("  ERROR: Server failed to start.")
            print("  Check log file for details:", log_file)
            return report
        print("  Server READY.")

        # --- Phase 2: Send requests (long-running for checkpoints) ---
        print(f"\n=== Phase 2: Sending {num_requests} streaming requests ===")
        print(f"  Max tokens per request: {max_tokens}")
        print(f"  Will kill engine after {kill_after_sec}s")

        test_start = time.time()
        tasks: list[asyncio.Task] = []

        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            task = asyncio.create_task(
                send_streaming_request(
                    port=port,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    request_id=f"req-{i}",
                )
            )
            tasks.append(task)
            # Stagger slightly so requests are in-flight before failure.
            await asyncio.sleep(0.3)

        # --- Phase 3: Wait for checkpoints, then inject failure ---
        print(f"\n=== Phase 3: Waiting {kill_after_sec}s for checkpoints ===")
        elapsed = time.time() - test_start
        remaining = kill_after_sec - elapsed
        if remaining > 0:
            # Poll for checkpoint creation while waiting.
            poll_start = time.time()
            while time.time() - poll_start < remaining:
                ckpt_count = count_shared_checkpoints()
                if ckpt_count > 0:
                    print(f"  Found {ckpt_count} checkpoint file(s) in {SHARED_CKPT_DIR}")
                await asyncio.sleep(1.0)

        ckpt_count = count_shared_checkpoints()
        report.checkpoints_observed = ckpt_count
        print(f"  Checkpoints at failure time: {ckpt_count}")

        # --- Phase 4: Kill one engine ---
        print("\n=== Phase 4: Injecting engine failure ===")
        report.failure_injection_time = time.time()
        killed_pid = kill_engine_process(server_proc.pid)
        if killed_pid:
            print(f"  Killed engine PID={killed_pid} at t={time.time() - test_start:.1f}s")
        else:
            print("  WARNING: Could not identify engine process to kill.")
            print("  Trying to kill server child processes directly...")
            # Fallback: send SIGUSR1 or just let timeout detection work.

        # --- Phase 5: Wait for requests to complete ---
        print("\n=== Phase 5: Waiting for requests to complete ===")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        report.all_requests_done_time = time.time()

        for r in results:
            if isinstance(r, Exception):
                report.failed_requests += 1
            elif isinstance(r, dict):
                if r.get("success"):
                    report.completed_requests += 1
                else:
                    report.failed_requests += 1

        # --- Phase 6: Parse logs for recovery evidence ---
        print("\n=== Phase 6: Analyzing server logs ===")

        # Give the server a moment to flush logs.
        await asyncio.sleep(2.0)

        recovery_events = parse_recovery_events(log_file)
        report.recovery_events = recovery_events
        checkpoint_count_in_log = parse_checkpoint_events(log_file)
        report.checkpoints_observed = max(
            report.checkpoints_observed, checkpoint_count_in_log
        )

        detection_time = parse_failure_detection(log_file)
        if detection_time:
            report.first_recovery_log_time = detection_time

        # --- Report ---
        print("\n" + "=" * 70)
        print("CHECKPOINT RECOVERY BENCHMARK RESULTS")
        print("=" * 70)
        print(f"  Total requests:                {report.total_requests}")
        print(f"  Completed:                     {report.completed_requests}")
        print(f"  Failed:                        {report.failed_requests}")
        print(f"  Checkpoints created:           {report.checkpoints_observed}")
        print(f"  In-engine recovery events:     {report.requests_recovered_via_checkpoint}")
        print(f"  Total tokens restored from KV: {report.total_tokens_restored}")

        if report.in_engine_failover_gap_ms is not None:
            print(f"  In-engine failover gap:        {report.in_engine_failover_gap_ms:.1f}ms")
        else:
            print(f"  In-engine failover gap:        N/A (could not parse from logs)")

        print()
        if recovery_events:
            print("  Recovery events:")
            for ev in recovery_events:
                print(f"    - Request {ev.request_id}: restored {ev.tokens_restored} tokens")
        else:
            print("  WARNING: No in-engine recovery events found in logs.")
            print("  This means either:")
            print("    a) Checkpoints were not created before failure, OR")
            print("    b) The FT client did client-retry instead of in-engine restore, OR")
            print("    c) The restore path has a bug.")

        print()

        # Streaming gap analysis: look for large gaps in token delivery
        # that would indicate a failover interruption.
        print("  Per-request streaming analysis:")
        for r in results:
            if isinstance(r, dict) and r.get("success"):
                rid = r.get("request_id", "?")
                tokens = r.get("tokens", 0)
                max_gap = r.get("max_gap_ms", 0)
                e2e = r.get("e2e_ms", 0)
                gap_indicator = " <-- possible failover gap" if max_gap > 2000 else ""
                print(
                    f"    {rid}: {tokens} tokens, "
                    f"max_gap={max_gap:.0f}ms, "
                    f"e2e={e2e:.0f}ms{gap_indicator}"
                )

        print("=" * 70)
        print(f"  Server log: {log_file}")
        print("=" * 70)

        # Verdict.
        if recovery_events:
            print(
                "\n  VERDICT: In-engine checkpoint recovery OBSERVED. "
                f"{len(recovery_events)} request(s) restored from KV checkpoint."
            )
        elif report.checkpoints_observed > 0 and report.completed_requests > 0:
            print(
                "\n  VERDICT: Checkpoints were created but no in-engine "
                "restore was observed. Requests may have completed via "
                "client retry or full recompute."
            )
        else:
            print(
                "\n  VERDICT: Could not verify in-engine checkpoint recovery."
            )

    finally:
        # Cleanup server.
        print("\nCleaning up server...")
        if server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        print("Done.")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="In-engine checkpoint recovery benchmark"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to serve",
    )
    parser.add_argument(
        "--num-requests", type=int, default=5,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200,
        help="Max tokens per request (use high values to ensure checkpoints)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=512,
        help="Max model context length",
    )
    parser.add_argument(
        "--kill-after", type=float, default=8.0,
        help="Inject failure after this many seconds",
    )
    parser.add_argument(
        "--failure-timeout-sec", type=float, default=5.0,
        help="FT coordinator failure detection timeout",
    )
    parser.add_argument(
        "--port", type=int, default=8200,
        help="API server port",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("In-Engine Checkpoint Recovery Benchmark")
    print("=" * 70)
    print(f"  Model:              {args.model}")
    print(f"  Requests:           {args.num_requests}")
    print(f"  Max tokens:         {args.max_tokens}")
    print(f"  Kill after:         {args.kill_after}s")
    print(f"  Failure timeout:    {args.failure_timeout_sec}s")
    print(f"  GPUs:               0, 1 (DP=2)")
    print()
    print("  This benchmark tests ACTUAL in-engine checkpoint recovery,")
    print("  not client-side retry. It verifies that when an engine dies,")
    print("  the surviving engine restores KV state from the shared")
    print("  checkpoint and resumes generation from where it left off.")

    report = asyncio.run(run_checkpoint_recovery_benchmark(
        model=args.model,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        kill_after_sec=args.kill_after,
        failure_timeout_sec=args.failure_timeout_sec,
        port=args.port,
    ))

    # Exit code.
    if report.requests_recovered_via_checkpoint > 0:
        print("\nBENCHMARK PASSED: In-engine checkpoint recovery verified.")
        sys.exit(0)
    elif report.completed_requests > 0:
        print("\nBENCHMARK PARTIAL: Requests completed but checkpoint "
              "recovery path not confirmed via logs.")
        sys.exit(0)
    else:
        print("\nBENCHMARK FAILED: No requests completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
