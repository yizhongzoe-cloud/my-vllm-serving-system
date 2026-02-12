#!/usr/bin/env python3
"""
Main entry point for running SLO scheduling benchmarks.

Usage (run from repo root):
    python slo_benchmark/scripts/run_benchmark.py
    python slo_benchmark/scripts/run_benchmark.py --dataset slo_benchmark/data/datasets/sample.jsonl
    python slo_benchmark/scripts/run_benchmark.py --synthetic 100
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# slo_benchmark/ is the benchmark project root
BENCHMARK_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(BENCHMARK_ROOT))

import yaml

from src.benchmark.dataset import Dataset, DatasetLoader
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.metrics import compute_metrics
from src.utils.server import VLLMServer, check_server


def load_config(config_path: str | None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = BENCHMARK_ROOT / "configs" / "default.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(config: dict, args: argparse.Namespace) -> Dataset:
    """Load dataset based on config or args."""
    # Command line args override config
    if args.dataset:
        path = Path(args.dataset)
        if path.suffix == ".jsonl":
            return DatasetLoader.from_jsonl(path)
        else:
            return DatasetLoader.from_json(path)

    if args.synthetic:
        return DatasetLoader.generate_synthetic(
            n=args.synthetic,
            seed=args.seed,
        )

    # Use config
    ds_config = config.get("dataset", {})
    ds_type = ds_config.get("type", "synthetic")

    if ds_type == "synthetic":
        return DatasetLoader.generate_synthetic(
            n=ds_config.get("num_requests", 100),
            slo_distribution=ds_config.get("slo_distribution"),
            seed=args.seed,
        )
    else:
        path = ds_config.get("path")
        if not path:
            raise ValueError("Dataset path not specified in config")
        return DatasetLoader.from_jsonl(path)


def progress_callback(completed: int, total: int, result):
    """Print progress during benchmark."""
    status = "✓" if result.success else "✗"
    slo_info = ""
    if result.request.ttft_slo_ms:
        met = "✓" if result.ttft_slo_met else "✗"
        slo_info += f" TTFT:{met}"
    if result.request.e2e_latency_slo_ms:
        met = "✓" if result.e2e_slo_met else "✗"
        slo_info += f" E2E:{met}"

    print(f"  [{completed}/{total}] {status} {result.latency_ms:.0f}ms{slo_info}")


def main():
    parser = argparse.ArgumentParser(description="Run SLO scheduling benchmark")

    # Config
    parser.add_argument("--config", type=str, help="Path to config YAML file")

    # Dataset options
    parser.add_argument("--dataset", type=str, help="Path to dataset file (jsonl/json)")
    parser.add_argument("--synthetic", type=int, help="Generate N synthetic requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Server options
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--start-server", action="store_true", help="Start server automatically")

    # Benchmark options
    parser.add_argument("--mode", choices=["sequential", "concurrent", "poisson"], help="Benchmark mode")
    parser.add_argument("--concurrency", type=int, help="Concurrency level")
    parser.add_argument("--rate", type=float, help="Request rate for poisson mode")

    # Output options
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    server_config = config.get("server", {})
    if args.model:
        server_config["model"] = args.model
    if args.host:
        server_config["host"] = args.host
    if args.port:
        server_config["port"] = args.port

    benchmark_config = config.get("benchmark", {})
    if args.mode:
        benchmark_config["mode"] = args.mode
    if args.concurrency:
        benchmark_config["concurrency"] = args.concurrency
    if args.rate:
        benchmark_config["rate"] = args.rate

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config, args)
    print(f"  Loaded {len(dataset)} requests")

    # Check/start server
    host = server_config.get("host", "localhost")
    port = server_config.get("port", 8000)
    model = server_config.get("model", "facebook/opt-125m")

    server = None
    if args.start_server:
        server = VLLMServer(model=model, host=host, port=port)
        server.start()
    elif not check_server(host, port):
        print(f"Error: Server not running at {host}:{port}")
        print("Start with --start-server or run server manually")
        sys.exit(1)

    try:
        # Run benchmark
        print(f"\nRunning benchmark...")
        print(f"  Mode: {benchmark_config.get('mode', 'concurrent')}")

        runner = BenchmarkRunner(
            host=host,
            port=port,
            model=model,
            timeout=benchmark_config.get("timeout", 120.0),
        )

        mode = benchmark_config.get("mode", "concurrent")
        callback = None if args.quiet else progress_callback

        if mode == "sequential":
            result = runner.run(dataset, mode="sequential", progress_callback=callback)
        elif mode == "concurrent":
            result = runner.run(
                dataset,
                mode="concurrent",
                concurrency=benchmark_config.get("concurrency", 10),
                progress_callback=callback,
            )
        else:  # poisson
            result = runner.run(
                dataset,
                mode="poisson",
                rate=benchmark_config.get("rate", 10.0),
                progress_callback=callback,
            )

        # Compute and display metrics
        metrics = compute_metrics(result)
        print(metrics.summary())

        # Save results
        output_dir = Path(args.output or config.get("output", {}).get("results_dir", "data/results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = output_dir / f"metrics_{timestamp}.json"
        metrics.save(metrics_path)
        print(f"\nResults saved to: {metrics_path}")

    finally:
        if server:
            server.stop()


if __name__ == "__main__":
    main()
