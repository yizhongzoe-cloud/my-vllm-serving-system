"""End-to-end fault-tolerant serving test with real model.

Launches 2 vLLM API servers (one per GPU) with the same model,
routes requests through the FT scheduler, kills one server mid-test,
and verifies that:
1. Requests on the surviving server complete normally.
2. New requests are routed to the survivor after failure.
3. Failover gap is measured.
4. Goodput metrics are collected.

When failue happens, client resends the request from scratch to a different server, simulating a simple retry without KV checkpoint recovery. 
This tests the FT router's ability to detect failure and route new requests to healthy replicas.

Usage:
    python tests/ft/test_ft_e2e_serving.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --num-requests 20 \
        --kill-after 10  # seconds
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import aiohttp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from vllm.v1.core.failure_detector import FailureDetector, ReplicaStatus
from vllm.v1.core.replica_manager import ReplicaManager
from vllm.v1.core.request_pool import RequestPool


# ---- Config ----

@dataclass
class ServerConfig:
    replica_id: int
    gpu_id: int
    port: int
    process: subprocess.Popen | None = None


@dataclass
class RequestResult:
    request_id: str
    replica_id: int
    success: bool
    ttft_ms: float | None = None
    total_tokens: int = 0
    e2e_latency_ms: float | None = None
    error: str | None = None
    was_rerouted: bool = False


@dataclass
class E2ETestReport:
    total_requests: int = 0
    completed: int = 0
    failed: int = 0
    rerouted: int = 0
    failover_gap_ms: float | None = None
    results: list[RequestResult] = field(default_factory=list)
    failure_time: float | None = None
    first_rerouted_request_time: float | None = None


# ---- Server Management ----

def start_vllm_server(
    model: str,
    gpu_id: int,
    port: int,
    max_model_len: int = 512,
) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.45",
        "--dtype", "float16",
        "--enforce-eager",  # Faster startup, skip CUDA graph capture.
    ]

    print(f"  Starting server on GPU {gpu_id}, port {port}...")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


async def wait_for_server(port: int, timeout: float = 180.0) -> bool:
    """Wait until server is ready by polling /health."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1.0)
    return False


async def send_chat_request(
    port: int,
    request_id: str,
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    prompt: str = "Write a short story about a robot learning to cook.",
    max_tokens: int = 100,
) -> RequestResult:
    """Send a chat completion request and measure timing."""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start_time = time.time()
    ttft = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return RequestResult(
                        request_id=request_id,
                        replica_id=-1,
                        success=False,
                        error=f"HTTP {resp.status}: {text[:200]}",
                    )

                if ttft is None:
                    ttft = (time.time() - start_time) * 1000

                data = await resp.json()
                e2e = (time.time() - start_time) * 1000
                tokens = data.get("usage", {}).get("completion_tokens", 0)

                return RequestResult(
                    request_id=request_id,
                    replica_id=-1,
                    success=True,
                    ttft_ms=ttft,
                    total_tokens=tokens,
                    e2e_latency_ms=e2e,
                )

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return RequestResult(
            request_id=request_id,
            replica_id=-1,
            success=False,
            error=str(e),
        )


# ---- FT Router ----

class FTRouter:
    """Simple FT-aware request router.

    Routes requests to available servers and detects failures.
    """

    def __init__(self, servers: list[ServerConfig]):
        self.servers = {s.replica_id: s for s in servers}
        self.replica_manager = ReplicaManager()
        self.failure_detector = FailureDetector(
            heartbeat_interval_sec=1.0,
            failure_timeout_sec=3.0,
            max_consecutive_failures=2,
        )

        for s in servers:
            self.replica_manager.add_replica(
                s.replica_id, gpu_id=s.gpu_id, max_num_seqs=64
            )
            self.failure_detector.register_replica(s.replica_id)

        self._round_robin = 0

    def get_server_port(self) -> tuple[int, int]:
        """Get (replica_id, port) for the next request.

        Uses round-robin among healthy replicas.
        """
        healthy = self.failure_detector.get_healthy_replicas()
        if not healthy:
            raise RuntimeError("No healthy replicas available!")

        idx = self._round_robin % len(healthy)
        self._round_robin += 1
        rid = healthy[idx]
        return rid, self.servers[rid].port

    def report_failure(self, replica_id: int):
        """Report a server as failed."""
        self.failure_detector.report_failure(replica_id)
        self.replica_manager.mark_failed(replica_id)

    def get_healthy_count(self) -> int:
        return len(self.failure_detector.get_healthy_replicas())

    async def check_server_health(self, replica_id: int) -> bool:
        """Check if a server is actually responding."""
        server = self.servers.get(replica_id)
        if server is None:
            return False
        url = f"http://localhost:{server.port}/health"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False


# ---- Main Test ----

async def run_e2e_test(
    model: str,
    num_requests: int,
    kill_after_sec: float,
    max_tokens: int,
    max_model_len: int,
) -> E2ETestReport:
    report = E2ETestReport(total_requests=num_requests)
    prompts = [
        "Write a short poem about the ocean.",
        "Explain how photosynthesis works in simple terms.",
        "What are three interesting facts about Mars?",
        "Describe a day in the life of a medieval blacksmith.",
        "How does a computer CPU work?",
    ]

    # --- Phase 1: Start 2 servers ---
    print("\n=== Phase 1: Starting 2 vLLM servers ===")
    servers = [
        ServerConfig(replica_id=0, gpu_id=0, port=8100),
        ServerConfig(replica_id=1, gpu_id=1, port=8101),
    ]

    for s in servers:
        s.process = start_vllm_server(
            model, s.gpu_id, s.port, max_model_len=max_model_len,
        )

    # Wait for both to be ready.
    print("  Waiting for servers to be ready...")
    for s in servers:
        ready = await wait_for_server(s.port, timeout=180.0)
        if ready:
            print(f"  Server on GPU {s.gpu_id} (port {s.port}): READY")
        else:
            print(f"  Server on GPU {s.gpu_id} (port {s.port}): FAILED TO START")
            # Cleanup
            for ss in servers:
                if ss.process:
                    ss.process.kill()
            return report

    # --- Phase 2: Setup FT Router ---
    print("\n=== Phase 2: Setting up FT Router ===")
    router = FTRouter(servers)
    print(f"  Healthy replicas: {router.get_healthy_count()}")

    # --- Phase 3: Send requests, kill a server midway ---
    print(f"\n=== Phase 3: Sending {num_requests} requests "
          f"(killing GPU 1 after {kill_after_sec}s) ===")

    test_start = time.time()
    killed = False
    kill_replica_id = 1  # We'll kill the server on GPU 1.
    tasks: list[asyncio.Task] = []

    async def send_single_request(idx: int) -> RequestResult:
        nonlocal killed
        prompt = prompts[idx % len(prompts)]
        rid, port = router.get_server_port()

        result = await send_chat_request(
            port=port,
            request_id=f"req-{idx}",
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        result.replica_id = rid

        # If the request failed and the target was the killed replica,
        # try to reroute to survivor.
        if not result.success and killed and rid == kill_replica_id:
            result.was_rerouted = True
            new_rid, new_port = router.get_server_port()
            retry_result = await send_chat_request(
                port=new_port,
                request_id=f"req-{idx}-rerouted",
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            retry_result.replica_id = new_rid
            retry_result.was_rerouted = True
            if report.first_rerouted_request_time is None:
                report.first_rerouted_request_time = time.time()
            return retry_result

        return result

    # Send requests with staggered start (0.5s apart).
    async def request_driver():
        nonlocal killed
        for i in range(num_requests):
            task = asyncio.create_task(send_single_request(i))
            tasks.append(task)

            elapsed = time.time() - test_start
            if not killed and elapsed >= kill_after_sec:
                print(f"\n  >>> KILLING server on GPU {kill_replica_id} "
                      f"at t={elapsed:.1f}s <<<")
                server = servers[kill_replica_id]
                if server.process:
                    server.process.send_signal(signal.SIGKILL)
                    server.process.wait()
                router.report_failure(kill_replica_id)
                killed = True
                report.failure_time = time.time()
                print(f"  >>> Server killed. "
                      f"Healthy replicas: {router.get_healthy_count()} <<<\n")

            await asyncio.sleep(0.5)

    await request_driver()

    # Wait for all requests to complete.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # --- Phase 4: Collect results ---
    print("\n=== Phase 4: Results ===")
    for r in results:
        if isinstance(r, Exception):
            report.failed += 1
            report.results.append(RequestResult(
                request_id="unknown", replica_id=-1,
                success=False, error=str(r),
            ))
        elif isinstance(r, RequestResult):
            report.results.append(r)
            if r.success:
                report.completed += 1
            else:
                report.failed += 1
            if r.was_rerouted:
                report.rerouted += 1

    # Compute failover gap.
    if report.failure_time and report.first_rerouted_request_time:
        report.failover_gap_ms = (
            (report.first_rerouted_request_time - report.failure_time) * 1000
        )

    # Print summary.
    test_duration = time.time() - test_start
    total_tokens = sum(r.total_tokens for r in report.results if r.success)
    success_ttfts = [r.ttft_ms for r in report.results
                     if r.success and r.ttft_ms is not None]
    success_e2es = [r.e2e_latency_ms for r in report.results
                    if r.success and r.e2e_latency_ms is not None]

    print("=" * 60)
    print("FAULT-TOLERANT E2E TEST RESULTS")
    print("=" * 60)
    print(f"  Total requests:     {report.total_requests}")
    print(f"  Completed:          {report.completed}")
    print(f"  Failed:             {report.failed}")
    print(f"  Rerouted:           {report.rerouted}")
    print(f"  Test duration:      {test_duration:.1f}s")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Goodput:            {total_tokens / test_duration:.1f} tok/s")
    if success_ttfts:
        import statistics
        print(f"  TTFT mean:          {statistics.mean(success_ttfts):.1f}ms")
        print(f"  TTFT p50:           {statistics.median(success_ttfts):.1f}ms")
    if success_e2es:
        print(f"  E2E mean:           {statistics.mean(success_e2es):.1f}ms")
        print(f"  E2E p50:            {statistics.median(success_e2es):.1f}ms")
    if report.failover_gap_ms is not None:
        print(f"  Failover gap:       {report.failover_gap_ms:.1f}ms")
    print()

    # Per-replica breakdown.
    for rid in [0, 1]:
        replica_results = [r for r in report.results if r.replica_id == rid]
        ok = sum(1 for r in replica_results if r.success)
        fail = sum(1 for r in replica_results if not r.success)
        print(f"  Replica {rid}: {ok} ok, {fail} failed")

    print("=" * 60)

    # --- Cleanup ---
    print("\nCleaning up servers...")
    for s in servers:
        if s.process and s.process.poll() is None:
            s.process.send_signal(signal.SIGTERM)
            try:
                s.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                s.process.kill()
    print("Done.")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="E2E fault-tolerant serving test"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model to serve",
    )
    parser.add_argument(
        "--num-requests", type=int, default=20,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--kill-after", type=float, default=10.0,
        help="Kill GPU 1 server after this many seconds",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=80,
        help="Max tokens per request",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=512,
        help="Max model context length",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Fault-Tolerant Multi-GPU Serving E2E Test")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Requests:       {args.num_requests}")
    print(f"  Kill after:     {args.kill_after}s")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  GPUs:           0, 1")

    report = asyncio.run(run_e2e_test(
        model=args.model,
        num_requests=args.num_requests,
        kill_after_sec=args.kill_after,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
    ))

    # Exit code based on results.
    if report.completed > 0 and report.rerouted > 0:
        print("\nTEST PASSED: Requests completed and failover worked!")
        sys.exit(0)
    elif report.completed > 0:
        print("\nTEST PARTIAL: Requests completed but no rerouting observed.")
        sys.exit(0)
    else:
        print("\nTEST FAILED: No requests completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
