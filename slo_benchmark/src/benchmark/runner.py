"""Benchmark runner for sending requests and collecting results."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .dataset import Dataset, Request


@dataclass
class RequestResult:
    """Result of a single request."""

    request: Request
    success: bool
    start_time: float
    end_time: float
    ttft_ms: float | None = None  # Time to first token
    e2e_latency_ms: float | None = None  # End-to-end latency
    output_tokens: int = 0
    output_text: str = ""
    error: str | None = None

    # SLO compliance
    ttft_slo_met: bool | None = None
    e2e_slo_met: bool | None = None

    @property
    def latency_ms(self) -> float:
        """Total request latency in ms."""
        return (self.end_time - self.start_time) * 1000

    def compute_slo_compliance(self) -> None:
        """Compute whether SLOs were met."""
        if self.request.ttft_slo_ms is not None and self.ttft_ms is not None:
            self.ttft_slo_met = self.ttft_ms <= self.request.ttft_slo_ms
        if self.request.e2e_latency_slo_ms is not None and self.e2e_latency_ms is not None:
            self.e2e_slo_met = self.e2e_latency_ms <= self.request.e2e_latency_slo_ms


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    results: list[RequestResult]
    total_time_s: float
    config: dict = field(default_factory=dict)

    @property
    def num_requests(self) -> int:
        return len(self.results)

    @property
    def num_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def throughput(self) -> float:
        """Requests per second."""
        if self.total_time_s == 0:
            return 0
        return self.num_success / self.total_time_s


class BenchmarkRunner:
    """Run benchmarks against vLLM server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        model: str = "facebook/opt-125m",
        timeout: float = 120.0,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        request: Request,
        stream: bool = False,
    ) -> RequestResult:
        """Send a single request and measure timing."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = request.to_api_payload(self.model)
        payload["stream"] = stream

        start_time = time.time()
        ttft = None

        try:
            async with session.post(url, json=payload) as response:
                if stream:
                    # For streaming, measure TTFT as time to first chunk
                    first_chunk = True
                    content = ""
                    async for chunk in response.content:
                        if first_chunk:
                            ttft = (time.time() - start_time) * 1000
                            first_chunk = False
                        content += chunk.decode("utf-8", errors="ignore")
                    end_time = time.time()
                    # Parse SSE response (simplified)
                    output_text = content
                    output_tokens = len(content.split())  # Rough estimate
                else:
                    result = await response.json()
                    end_time = time.time()

                    if "choices" in result:
                        output_text = result["choices"][0]["message"].get("content", "")
                        usage = result.get("usage", {})
                        output_tokens = usage.get("completion_tokens", 0)
                    else:
                        raise ValueError(f"API error: {result}")

                e2e_latency = (end_time - start_time) * 1000

                req_result = RequestResult(
                    request=request,
                    success=True,
                    start_time=start_time,
                    end_time=end_time,
                    ttft_ms=ttft if stream else e2e_latency,  # Non-stream: TTFT ≈ E2E
                    e2e_latency_ms=e2e_latency,
                    output_tokens=output_tokens,
                    output_text=output_text[:200],  # Truncate for storage
                )
                req_result.compute_slo_compliance()
                return req_result

        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request=request,
                success=False,
                start_time=start_time,
                end_time=end_time,
                error=str(e),
            )

    async def run_sequential(
        self,
        dataset: Dataset,
        stream: bool = False,
        progress_callback=None,
    ) -> BenchmarkResult:
        """Run requests sequentially (one at a time)."""
        results = []
        start_time = time.time()

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, request in enumerate(dataset):
                result = await self._send_request(session, request, stream)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(dataset), result)

        total_time = time.time() - start_time
        return BenchmarkResult(
            results=results,
            total_time_s=total_time,
            config={"mode": "sequential", "stream": stream},
        )

    async def run_concurrent(
        self,
        dataset: Dataset,
        concurrency: int = 10,
        stream: bool = False,
        progress_callback=None,
    ) -> BenchmarkResult:
        """Run requests concurrently with limited parallelism."""
        results = []
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        completed = 0

        async def run_with_semaphore(request: Request) -> RequestResult:
            nonlocal completed
            async with semaphore:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    result = await self._send_request(session, request, stream)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(dataset), result)
                    return result

        tasks = [run_with_semaphore(req) for req in dataset]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        return BenchmarkResult(
            results=list(results),
            total_time_s=total_time,
            config={"mode": "concurrent", "concurrency": concurrency, "stream": stream},
        )

    async def run_poisson(
        self,
        dataset: Dataset,
        rate: float = 10.0,
        stream: bool = False,
        progress_callback=None,
    ) -> BenchmarkResult:
        """
        Run requests with Poisson arrival pattern.

        Args:
            dataset: Dataset to run
            rate: Average requests per second
            stream: Whether to use streaming
            progress_callback: Callback for progress updates
        """
        import random

        results = []
        start_time = time.time()
        completed = 0
        tasks = []

        async def run_request(request: Request, delay: float) -> RequestResult:
            nonlocal completed
            await asyncio.sleep(delay)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                result = await self._send_request(session, request, stream)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(dataset), result)
                return result

        # Schedule requests with exponential inter-arrival times
        cumulative_delay = 0.0
        for request in dataset:
            inter_arrival = random.expovariate(rate)
            cumulative_delay += inter_arrival
            tasks.append(run_request(request, cumulative_delay))

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        return BenchmarkResult(
            results=list(results),
            total_time_s=total_time,
            config={"mode": "poisson", "rate": rate, "stream": stream},
        )

    def run(
        self,
        dataset: Dataset,
        mode: str = "concurrent",
        **kwargs,
    ) -> BenchmarkResult:
        """
        Synchronous wrapper to run benchmark.

        Args:
            dataset: Dataset to benchmark
            mode: "sequential", "concurrent", or "poisson"
            **kwargs: Additional arguments for the specific mode
        """
        if mode == "sequential":
            coro = self.run_sequential(dataset, **kwargs)
        elif mode == "concurrent":
            coro = self.run_concurrent(dataset, **kwargs)
        elif mode == "poisson":
            coro = self.run_poisson(dataset, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return asyncio.run(coro)
