#!/usr/bin/env python3
"""Single-engine cheap-resume vs recompute microbench.

Goal: isolate the value of host-RAM KV checkpoint by removing the
router and the peer engine. Under high enough load, vLLM's native
capacity preemption fires; this script measures the cost difference
between:
  - vllm_fcfs : preempt → discard KV → reprefill on resume
  - ours      : preempt → V3 reload from host RAM → resume in 1s

No router, no cross-engine reroute, no slack picker. Just engine
internals + client.

Reuses workload_builder / Client / parse_request_done_log from
e_m1_slo_sweep so behavior matches the multi-engine path exactly
in everything except (a) only 1 engine, (b) no router.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Make experiments_v2.* importable when launched as script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments_v2.eval.workloads.workload_builder import (  # noqa: E402
    build_schedule, summarize_schedule,
)
from experiments_v2.eval.scripts.e_m1_slo_sweep import (  # noqa: E402
    Client, parse_request_done_log, cleanup_shm, percentile,
    MODEL, REQUEST_TIMEOUT_S,
)

ENGINE_PORT = 8401
ENGINE_READY_TIMEOUT_S = 300


def start_engine(baseline: str, log_path: Path,
                 max_model_len: int) -> subprocess.Popen:
    """Launch a single vLLM engine with FT flags set per baseline."""
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # No router and no peer in this experiment.
    env["FT_ROUTER_SHM_BUS"] = "0"
    if baseline == "ours":
        # Continuous checkpoint + V3 reload on capacity preempt.
        # Without a peer, V3 reload happens locally after KV pool frees up.
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["SLO_PRIORITY_PREEMPT"] = "0"   # picker irrelevant single-engine
    elif baseline == "vllm_fcfs":
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    else:
        raise ValueError(f"unsupported baseline: {baseline}")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(ENGINE_PORT),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization",
        os.environ.get("FT_GPU_MEMORY_UTILIZATION", "0.9"),
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[single] launching engine ({baseline}) on GPU 0 → "
          f"{log_path.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def wait_url_ready(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def shutdown(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["vllm_fcfs", "ours"],
                        required=True)
    parser.add_argument("--dataset", default="arxivsumm",
                        choices=["arxivsumm", "sharegpt",
                                 "mixed_short_long"])
    parser.add_argument("--mixed-short-ratio", type=float, default=0.7)
    parser.add_argument("--arrival-rate-qps", type=float, required=True)
    parser.add_argument("--num-requests", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float, required=True,
                        help="Uniform SLO TTFT threshold (ms).")
    parser.add_argument("--tpot-slo-ms", type=float, required=True,
                        help="Uniform SLO TPOT threshold (ms).")
    parser.add_argument("--out-tag", type=str, default="single",
                        help="Suffix on output filenames.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    # Workload.
    if args.dataset == "mixed_short_long":
        from experiments_v2.eval.workloads.workload_builder import (
            build_mixed_schedule,
        )
        schedule_full = build_mixed_schedule(
            short_dataset="sharegpt", long_dataset="arxivsumm",
            short_ratio=args.mixed_short_ratio,
            num_requests=args.num_requests,
            arrival_rate_qps=args.arrival_rate_qps,
            seed=args.seed,
        )
        schedule = [(s[0], s[1], s[2]) for s in schedule_full]
        classes = [s[3] for s in schedule_full]
    else:
        schedule = build_schedule(
            dataset_name=args.dataset,
            num_requests=args.num_requests,
            arrival_rate_qps=args.arrival_rate_qps,
            seed=args.seed,
        )
        classes = ["uniform"] * len(schedule)

    summary = summarize_schedule(schedule)
    print(f"[single] schedule: {summary}")
    print(f"[single] SLO uniform: TTFT≤{args.ttft_slo_ms:.0f}ms, "
          f"TPOT≤{args.tpot_slo_ms:.1f}ms")

    # Output paths.
    results_dir = Path(
        os.environ.get("EVAL_RESULTS_DIR",
                       Path(__file__).resolve().parent.parent / "results")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"single_{args.baseline}_{args.dataset}_"
           f"qps{args.arrival_rate_qps}_n{args.num_requests}_"
           f"seed{args.seed}_{args.out_tag}")
    engine_log = results_dir / f"{tag}_engine0.log"
    metrics_out = results_dir / f"{tag}_metrics.json"

    cleanup_shm()
    engine = start_engine(args.baseline, engine_log, args.max_model_len)
    try:
        if not wait_url_ready(
            f"http://127.0.0.1:{ENGINE_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ):
            print("[single] FAIL: engine never became healthy")
            return 1
        print("[single] engine ready")

        engine_url = f"http://127.0.0.1:{ENGINE_PORT}"

        # Always pass SLO via vllm_xargs (so engine logs include it for
        # potential picker logic — picker is disabled but harmless).
        clients: list[Client] = []
        t0 = time.time()
        for i, (offset_s, prompt, max_tokens) in enumerate(schedule):
            target_t = t0 + offset_s
            now = time.time()
            if target_t > now:
                time.sleep(target_t - now)
            c = Client(i, engine_url, prompt, max_tokens,
                       args.ttft_slo_ms, args.tpot_slo_ms,
                       ignore_eos=False)
            c.start()
            clients.append(c)
        print(f"[single] fired {len(clients)} requests; waiting "
              "completion")

        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)

        n_ok = sum(1 for c in clients
                   if c.error is None and c.status_code == 200)
        n_err = sum(1 for c in clients if c.error is not None)
        print(f"[single] outcomes: {n_ok}/{len(clients)} 200 OK, "
              f"{n_err} errored")

        # Parse engine log for per-request timings.
        events = parse_request_done_log(engine_log)
        events_sorted = sorted(events, key=lambda e: e["arrival_ts"])
        for k, e in enumerate(events_sorted):
            e["class"] = (classes[k] if k < len(classes) else "uniform")
            e["slo_ttft_ms"] = args.ttft_slo_ms
            e["slo_tpot_ms"] = args.tpot_slo_ms
            e["slo_met"] = (
                e["ttft_ms"] <= args.ttft_slo_ms
                and e["tpot_ms"] <= args.tpot_slo_ms
            )
        slo_met_flags = [e["slo_met"] for e in events_sorted]
        slo_met_pct = (100.0 * sum(slo_met_flags) / len(slo_met_flags)
                       if slo_met_flags else 0.0)

        ttft_vals = [e["ttft_ms"] for e in events_sorted]
        tpot_vals = [e["tpot_ms"] for e in events_sorted if e["tpot_ms"] > 0]
        e2e_vals = [e["e2e_ms"] for e in events_sorted]

        print(f"[single] SLO_met: {slo_met_pct:.1f}% "
              f"({sum(slo_met_flags)}/{len(slo_met_flags)})")
        print(f"[single] TTFT_ms P50={percentile(ttft_vals, 50):.1f} "
              f"P95={percentile(ttft_vals, 95):.1f}")
        print(f"[single] TPOT_ms P50={percentile(tpot_vals, 50):.1f} "
              f"P95={percentile(tpot_vals, 95):.1f}")
        print(f"[single] E2E_ms  P50={percentile(e2e_vals, 50):.1f} "
              f"P95={percentile(e2e_vals, 95):.1f}")

        with open(metrics_out, "w") as f:
            json.dump({
                "baseline": args.baseline,
                "dataset": args.dataset,
                "num_requests": args.num_requests,
                "arrival_rate_qps": args.arrival_rate_qps,
                "seed": args.seed,
                "slo_mode": "uniform_single_engine",
                "slo": {"ttft_ms": args.ttft_slo_ms,
                        "tpot_ms": args.tpot_slo_ms},
                "schedule_summary": summary,
                "n_events": len(events_sorted),
                "slo_met_pct": slo_met_pct,
                "slo_met_count": sum(slo_met_flags),
                "ttft_ms": {
                    "p50": percentile(ttft_vals, 50),
                    "p95": percentile(ttft_vals, 95),
                },
                "tpot_ms": {
                    "p50": percentile(tpot_vals, 50),
                    "p95": percentile(tpot_vals, 95),
                },
                "e2e_ms": {
                    "p50": percentile(e2e_vals, 50),
                    "p95": percentile(e2e_vals, 95),
                },
                "client_outcomes": {"200_ok": n_ok, "errored": n_err},
                "per_request": events_sorted,
            }, f, indent=2)
        print(f"[single] metrics → {metrics_out}")
    finally:
        shutdown(engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
