#!/usr/bin/env python3
"""Generate mock FT benchmark data for testing the plotting script."""

import json
import random
from pathlib import Path

random.seed(42)

def mock_request(exp_id, exp_type, num_requests, kill_after, idx, replica_id,
                 was_killed=False):
    success = True
    was_rerouted = False
    if was_killed and replica_id == 1:
        # Simulate failure: some requests fail, some get rerouted
        if random.random() < 0.1:
            success = False
        else:
            was_rerouted = True
            replica_id = 0

    ttft = random.gauss(800, 200) if success else None
    e2e = ttft * random.uniform(1.0, 1.05) if ttft else None
    tokens = random.randint(50, 80) if success else 0

    # Rerouted requests have higher latency
    if was_rerouted and ttft:
        ttft += random.gauss(300, 100)
        e2e = ttft * random.uniform(1.0, 1.05)

    return {
        "experiment_id": exp_id,
        "experiment_type": exp_type,
        "num_requests": num_requests,
        "kill_after_sec": kill_after,
        "request_id": f"req-{idx}",
        "replica_id": replica_id,
        "success": success,
        "ttft_ms": round(ttft, 2) if ttft else None,
        "e2e_latency_ms": round(e2e, 2) if e2e else None,
        "total_tokens": tokens,
        "was_rerouted": was_rerouted,
        "error": "Connection refused" if not success else None,
    }


def generate():
    experiments = [
        {"type": "baseline", "n": 10, "kill": None},
        {"type": "baseline", "n": 20, "kill": None},
        {"type": "baseline", "n": 30, "kill": None},
        {"type": "failure", "n": 20, "kill": 3.0},
        {"type": "failure", "n": 30, "kill": 3.0},
        {"type": "failure", "n": 20, "kill": 5.0},
        {"type": "failure", "n": 30, "kill": 5.0},
        {"type": "failure", "n": 30, "kill": 8.0},
        {"type": "failure", "n": 30, "kill": 10.0},
    ]

    summaries = []
    per_request = []

    for exp_idx, exp in enumerate(experiments):
        exp_id = f"exp_{exp_idx:03d}_{exp['type']}_n{exp['n']}"
        if exp["kill"]:
            exp_id += f"_kill{exp['kill']:.0f}s"

        kill_req_idx = int(exp["kill"] / 0.5) if exp["kill"] else exp["n"] + 1
        records = []
        for i in range(exp["n"]):
            rid = i % 2
            was_killed = (exp["type"] == "failure" and i >= kill_req_idx and rid == 1)
            r = mock_request(exp_id, exp["type"], exp["n"], exp["kill"], i, rid,
                             was_killed)
            records.append(r)
            per_request.append(r)

        ok = [r for r in records if r["success"]]
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"]]
        e2es = [r["e2e_latency_ms"] for r in ok if r["e2e_latency_ms"]]
        total_tok = sum(r["total_tokens"] for r in ok)
        duration = exp["n"] * 0.5 + 1.0

        import numpy as np

        failover_gap = None
        rerouted = [r for r in records if r["was_rerouted"]]
        if rerouted and exp["kill"]:
            failover_gap = random.gauss(900, 200)

        summaries.append({
            "experiment_id": exp_id,
            "experiment_type": exp["type"],
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "num_requests": exp["n"],
            "kill_after_sec": exp["kill"],
            "max_tokens": 80,
            "max_model_len": 512,
            "completed": len(ok),
            "failed": len(records) - len(ok),
            "rerouted": len(rerouted),
            "total_tokens": total_tok,
            "test_duration_sec": round(duration, 2),
            "goodput_tok_per_sec": round(total_tok / duration, 2),
            "request_throughput_rps": round(len(ok) / duration, 2),
            "ttft_mean_ms": round(float(np.mean(ttfts)), 2) if ttfts else 0,
            "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0,
            "ttft_p90_ms": round(float(np.percentile(ttfts, 90)), 2) if ttfts else 0,
            "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 2) if ttfts else 0,
            "e2e_mean_ms": round(float(np.mean(e2es)), 2) if e2es else 0,
            "e2e_p50_ms": round(float(np.percentile(e2es, 50)), 2) if e2es else 0,
            "e2e_p90_ms": round(float(np.percentile(e2es, 90)), 2) if e2es else 0,
            "e2e_p99_ms": round(float(np.percentile(e2es, 99)), 2) if e2es else 0,
            "failover_gap_ms": round(failover_gap, 2) if failover_gap else None,
            "success_rate": round(len(ok) / len(records), 4),
        })

    output_dir = Path("slo_benchmark/data/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "full_results_mock.json"
    with open(output, "w") as f:
        json.dump({
            "metadata": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "max_tokens": 80,
                "max_model_len": 512,
                "timestamp": "mock",
                "num_experiments": len(summaries),
            },
            "summaries": summaries,
            "per_request": per_request,
        }, f, indent=2)

    print(f"Mock data written to {output}")
    print(f"  {len(summaries)} experiments, {len(per_request)} requests")


if __name__ == "__main__":
    generate()
