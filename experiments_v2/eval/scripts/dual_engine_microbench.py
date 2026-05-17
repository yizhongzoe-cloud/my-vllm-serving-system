#!/usr/bin/env python3
"""Dual-engine microbench: vllm_fcfs vs ours on 2 GPUs.

Goal: see if ours wins on multi-GPU long-context (14B) at the
saturation edge, where the previous mixed_short_long sweep (7B,
KV utilization 5-20%) had no preempts and couldn't reveal the
mechanism.

Mirrors single_engine_microbench in spirit: uniform SLO, no
class-tier complications, identical flag set as e_m1 for ours
vs vllm_fcfs. Difference vs single: 2 engines on GPU 0 and 1,
ours runs through router + cross-engine reroute picker.

Reuses Client, parse_request_done_log, start_router, wait_*,
MODEL, port constants from e_m1_slo_sweep so the router path
is byte-identical to the well-tested setup.
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments_v2.eval.workloads.workload_builder import (  # noqa: E402
    build_schedule, build_mixed_schedule, summarize_schedule,
)
from experiments_v2.eval.scripts.e_m1_slo_sweep import (  # noqa: E402
    Client, parse_request_done_log, cleanup_shm, percentile,
    MODEL, REQUEST_TIMEOUT_S, start_router,
    ENGINE_0_PORT, ENGINE_1_PORT, ROUTER_PORT,
    ENGINE_READY_TIMEOUT_S, ROUTER_READY_TIMEOUT_S,
    wait_url_ready, wait_both_engines_alive_in_router,
)


def start_engine(engine_id: int, gpu_id: int, baseline: str,
                 log_path: Path, max_model_len: int) -> subprocess.Popen:
    """Launch one vLLM engine. Flag set matches e_m1's ours/vllm_fcfs."""
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if baseline == "ours":
        # full method: substrate + V3 reload + picker
        env["FT_ROUTER_SHM_BUS"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["SLO_PRIORITY_PREEMPT"] = "1"
        env.setdefault("FT_PICKER_IN_DANGER_MS", "300")
        env.setdefault("FT_PICKER_HEAD_TOO_LATE_MS", "9999999")
    elif baseline == "ours_no_picker":
        # substrate + V3 reload, picker OFF.
        # Isolates picker contribution vs substrate-only.
        env["FT_ROUTER_SHM_BUS"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    elif baseline == "reroute_no_ckpt":
        # router + reroute path, but no checkpoint substrate.
        # Llumnix-class baseline. Picker forced off since it requires
        # the manifest gate to be satisfied (no manifest without ckpt).
        env["FT_ROUTER_SHM_BUS"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    elif baseline == "vllm_fcfs":
        # vanilla vLLM, no router, no FT.
        env["FT_ROUTER_SHM_BUS"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    else:
        raise ValueError(f"unsupported baseline: {baseline}")
    port = ENGINE_0_PORT if engine_id == 0 else ENGINE_1_PORT
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization",
        os.environ.get("FT_GPU_MEMORY_UTILIZATION", "0.9"),
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[dual] launching engine {engine_id} on GPU {gpu_id} "
          f"({baseline}) → {log_path.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def shutdown(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _fmt(v: float | None, spec: str) -> str:
    """Format a percentile that may be None (empty input list)."""
    if v is None:
        return "n/a"
    return format(v, spec)


def main() -> int:
    # Canonical router policy default: round_robin lets workload variance
    # create natural imbalance so picker has fire space. least_load (the
    # router default) actively synchronizes both engines to saturation,
    # which blocks picker via peer-load gate (v4 vs v1-v3 finding).
    # Set in parent env so the router subprocess inherits it.
    os.environ.setdefault("FT_ROUTER_POLICY", "round_robin")

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",
                        choices=["vllm_fcfs", "reroute_no_ckpt",
                                 "ours_no_picker", "ours"],
                        required=True)
    parser.add_argument("--burst", action="store_true",
                        help="Fire all requests at t=0 (no Poisson "
                        "inter-arrival delay). Models a flash crowd / "
                        "burst-arrival workload.")
    parser.add_argument("--dataset", default="arxivsumm",
                        choices=["arxivsumm", "sharegpt",
                                 "mixed_short_long"])
    parser.add_argument("--mixed-short-ratio", type=float, default=0.7,
                        help="Fraction of short requests in mixed mode.")
    parser.add_argument("--arrival-rate-qps", type=float, required=True)
    parser.add_argument("--num-requests", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    # Uniform-SLO args (used when --dataset is arxivsumm or sharegpt).
    parser.add_argument("--ttft-slo-ms", type=float, default=None,
                        help="Uniform SLO TTFT (ms). Required for "
                        "arxivsumm/sharegpt datasets.")
    parser.add_argument("--tpot-slo-ms", type=float, default=None,
                        help="Uniform SLO TPOT (ms). Required for "
                        "arxivsumm/sharegpt datasets.")
    # Per-class SLO args (used when --dataset is mixed_short_long).
    parser.add_argument("--ttft-slo-short-ms", type=float, default=None,
                        help="TTFT SLO for short class (ms). Required "
                        "for mixed_short_long.")
    parser.add_argument("--tpot-slo-short-ms", type=float, default=None)
    parser.add_argument("--ttft-slo-long-ms", type=float, default=None,
                        help="TTFT SLO for long class (ms). Required "
                        "for mixed_short_long.")
    parser.add_argument("--tpot-slo-long-ms", type=float, default=None)
    parser.add_argument("--out-tag", type=str, default="dual",
                        help="Suffix on output filenames.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    # Validate SLO args + assemble per-class SLO map.
    if args.dataset == "mixed_short_long":
        for name in ("ttft_slo_short_ms", "tpot_slo_short_ms",
                     "ttft_slo_long_ms", "tpot_slo_long_ms"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_','-')} is required "
                             f"for --dataset mixed_short_long")
        slo_by_class = {
            "short": (args.ttft_slo_short_ms, args.tpot_slo_short_ms),
            "long": (args.ttft_slo_long_ms, args.tpot_slo_long_ms),
        }
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
        if args.ttft_slo_ms is None or args.tpot_slo_ms is None:
            parser.error("--ttft-slo-ms and --tpot-slo-ms are required "
                         f"for --dataset {args.dataset}")
        slo_by_class = {
            "uniform": (args.ttft_slo_ms, args.tpot_slo_ms),
        }
        schedule = build_schedule(
            dataset_name=args.dataset,
            num_requests=args.num_requests,
            arrival_rate_qps=args.arrival_rate_qps,
            seed=args.seed,
        )
        classes = ["uniform"] * len(schedule)
    summary = summarize_schedule(schedule)
    print(f"[dual] model: {MODEL}")
    print(f"[dual] schedule: {summary}")
    if args.dataset == "mixed_short_long":
        print(f"[dual] class counts: short="
              f"{classes.count('short')} long={classes.count('long')}")
        for cls, (ttft, tpot) in slo_by_class.items():
            print(f"[dual] SLO {cls}: TTFT≤{ttft:.0f}ms TPOT≤{tpot:.1f}ms")
    else:
        ttft, tpot = slo_by_class["uniform"]
        print(f"[dual] SLO uniform: TTFT≤{ttft:.0f}ms TPOT≤{tpot:.1f}ms")

    results_dir = Path(
        os.environ.get("EVAL_RESULTS_DIR",
                       Path(__file__).resolve().parent.parent / "results")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"dual_{args.baseline}_{args.dataset}_"
           f"qps{args.arrival_rate_qps}_n{args.num_requests}_"
           f"seed{args.seed}_{args.out_tag}")
    engine0_log = results_dir / f"{tag}_engine0.log"
    engine1_log = results_dir / f"{tag}_engine1.log"
    router_log = results_dir / f"{tag}_router.log"
    metrics_out = results_dir / f"{tag}_metrics.json"

    cleanup_shm()
    engine0 = start_engine(0, 0, args.baseline, engine0_log,
                           args.max_model_len)
    engine1 = start_engine(1, 1, args.baseline, engine1_log,
                           args.max_model_len)
    router = None
    try:
        ok0 = wait_url_ready(
            f"http://127.0.0.1:{ENGINE_0_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        )
        ok1 = wait_url_ready(
            f"http://127.0.0.1:{ENGINE_1_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        )
        if not (ok0 and ok1):
            print("[dual] FAIL: engine never became healthy")
            return 1
        print("[dual] both engines ready")

        if args.baseline == "vllm_fcfs":
            target_urls = [
                f"http://127.0.0.1:{ENGINE_0_PORT}",
                f"http://127.0.0.1:{ENGINE_1_PORT}",
            ]
            print("[dual] vllm_fcfs: bypassing router, client round-robin")
        else:
            router = start_router(router_log)
            if not wait_url_ready(
                f"http://127.0.0.1:{ROUTER_PORT}/health",
                ROUTER_READY_TIMEOUT_S,
            ):
                print("[dual] FAIL: router not healthy")
                return 1
            if not wait_both_engines_alive_in_router():
                print("[dual] FAIL: router didn't see both engines alive")
                return 1
            target_urls = [f"http://127.0.0.1:{ROUTER_PORT}"]
            print(f"[dual] {args.baseline}: using router")

        # Only the picker (in 'ours') reads SLO from request xargs.
        # Other baselines don't use SLO at scheduling time.
        pass_slo = (args.baseline == "ours")

        if args.burst:
            print(f"[dual] BURST mode: firing all {len(schedule)} "
                  "requests at t=0 (Poisson offsets ignored)")
        clients: list[Client] = []
        t0 = time.time()
        for i, (offset_s, prompt, max_tokens) in enumerate(schedule):
            if not args.burst:
                target_t = t0 + offset_s
                now = time.time()
                if target_t > now:
                    time.sleep(target_t - now)
            engine_url = target_urls[i % len(target_urls)]
            cls = classes[i]
            class_ttft, class_tpot = slo_by_class[cls]
            ttft = class_ttft if pass_slo else None
            tpot = class_tpot if pass_slo else None
            c = Client(i, engine_url, prompt, max_tokens, ttft, tpot,
                       ignore_eos=False)
            c.start()
            clients.append(c)
        print(f"[dual] fired {len(clients)} requests; waiting completion")

        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)

        n_ok = sum(1 for c in clients
                   if c.error is None and c.status_code == 200)
        n_err = sum(1 for c in clients if c.error is not None)
        print(f"[dual] outcomes: {n_ok}/{len(clients)} 200 OK, "
              f"{n_err} errored")

        events = (parse_request_done_log(engine0_log)
                  + parse_request_done_log(engine1_log))
        # Sort events by arrival_ts; aligns with the order we fired
        # clients so events_sorted[k].class == classes[k] if all
        # requests completed.
        events_sorted = sorted(events, key=lambda e: e["arrival_ts"])
        for k, e in enumerate(events_sorted):
            cls = classes[k] if k < len(classes) else "uniform"
            class_ttft, class_tpot = slo_by_class[cls]
            e["class"] = cls
            e["slo_ttft_ms"] = class_ttft
            e["slo_tpot_ms"] = class_tpot
            e["slo_met"] = (
                e["ttft_ms"] <= class_ttft
                and e["tpot_ms"] <= class_tpot
            )
        flags = [e["slo_met"] for e in events_sorted]
        slo_met_pct = (100.0 * sum(flags) / len(flags)
                       if flags else 0.0)
        # Per-class breakdown for mixed mode.
        per_class_breakdown = {}
        for cls in slo_by_class:
            cls_events = [e for e in events_sorted if e["class"] == cls]
            if cls_events:
                cls_met = sum(1 for e in cls_events if e["slo_met"])
                per_class_breakdown[cls] = {
                    "n": len(cls_events),
                    "met": cls_met,
                    "pct": 100.0 * cls_met / len(cls_events),
                }

        ttft_vals = [e["ttft_ms"] for e in events_sorted]
        tpot_vals = [e["tpot_ms"] for e in events_sorted
                     if e["tpot_ms"] > 0]
        e2e_vals = [e["e2e_ms"] for e in events_sorted]

        print(f"[dual] SLO_met: {slo_met_pct:.1f}% "
              f"({sum(flags)}/{len(flags)})")
        for cls, b in per_class_breakdown.items():
            print(f"[dual]   {cls}: {b['pct']:.1f}% "
                  f"({b['met']}/{b['n']})")
        print(f"[dual] TTFT_ms P50={_fmt(percentile(ttft_vals,50),'.1f')} "
              f"P95={_fmt(percentile(ttft_vals,95),'.1f')}")
        print(f"[dual] TPOT_ms P50={_fmt(percentile(tpot_vals,50),'.1f')} "
              f"P95={_fmt(percentile(tpot_vals,95),'.1f')}")
        print(f"[dual] E2E_ms  P50={_fmt(percentile(e2e_vals,50),'.1f')} "
              f"P95={_fmt(percentile(e2e_vals,95),'.1f')}")

        with open(metrics_out, "w") as f:
            json.dump({
                "baseline": args.baseline,
                "model": MODEL,
                "dataset": args.dataset,
                "num_requests": args.num_requests,
                "arrival_rate_qps": args.arrival_rate_qps,
                "seed": args.seed,
                "slo_mode": ("class_tier_dual_engine"
                             if args.dataset == "mixed_short_long"
                             else "uniform_dual_engine"),
                "slo_by_class": {
                    cls: {"ttft_ms": t, "tpot_ms": p}
                    for cls, (t, p) in slo_by_class.items()
                },
                "schedule_summary": summary,
                "n_events": len(events_sorted),
                "slo_met_pct": slo_met_pct,
                "slo_met_count": sum(flags),
                "per_class_breakdown": per_class_breakdown,
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
        print(f"[dual] metrics → {metrics_out}")
    finally:
        if router is not None:
            shutdown(router)
        shutdown(engine0)
        shutdown(engine1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
