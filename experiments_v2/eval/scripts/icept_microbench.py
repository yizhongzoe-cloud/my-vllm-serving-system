#!/usr/bin/env python3
"""Interception (augmented-LLM) microbench, single engine.

Models the INFERCEPT setting: some requests get *intercepted* mid-
generation (they pause to call an external tool / RAG / agent step),
then resume. We compare how the two systems resume:

  vllm_fcfs : the continuation is a fresh request carrying the whole
              accumulated context (prompt + tokens generated so far),
              so it REPREFILLS everything — INFERCEPT's "37% of time
              spent recomputing KV".
  ours      : during segment 1 the KV is continuously checkpointed to
              host RAM; the continuation RELOADS that KV (+ bounded
              replay of the uncovered suffix) instead of reprefilling.

Both FREE the KV during the pause (the segment-1 request finishes), so
the difference is purely the resume cost. Reload << reprefill, so under
concurrency ours keeps throughput/latency up while the baseline's
reprefills clog the prefill path.

Per logical request with an interception at output token K, pause D:
  seg1: POST prompt P, max_tokens=K, vllm_xargs.router_req_id=<rid>
  pause D seconds (the tool call)
  seg2: ours     -> POST P, reload xargs (original_internal_req_id +
                    num_checkpointed_tokens read from /dev/shm)
        baseline -> POST P + seg1_text, max_tokens = T-K  (reprefill)
Non-intercepted requests run as one segment (max_tokens=T), identical
for both systems.

Metrics: completed throughput (req/s over the firing window), per-resume
segment-2 latency (reload vs reprefill), and per-request total latency
(excluding the injected pause).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/yzhong76/code/my-vllm-serving-system")
MODEL = os.path.expanduser("~/model/Qwen2.5-14B-Instruct")
PORT = 8401
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")
REQUEST_TIMEOUT_S = 1200.0


def post(body: dict, timeout: float = REQUEST_TIMEOUT_S) -> tuple[dict, float]:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def read_internal_id(rid: str) -> str | None:
    try:
        return json.loads((REQ_MAP_DIR / rid).read_text()).get(
            "internal_req_id")
    except (OSError, json.JSONDecodeError):
        return None


def read_covered(internal: str) -> int | None:
    d = CKPT_DIR / internal
    try:
        name = (d / "latest_rank0").read_text().strip()
        m = json.loads((d / name).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ct = m.get("covered_tokens")
    return int(ct) if ct is not None else None


def wait_health(timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


class IceptClient(threading.Thread):
    """One logical request, possibly with one interception."""

    def __init__(self, idx: int, prompt: str, total_out: int,
                 icept_at: int | None, pause_s: float, baseline: str):
        super().__init__()
        self.idx = idx
        self.prompt = prompt
        self.total_out = total_out
        self.icept_at = icept_at        # None -> no interception
        self.pause_s = pause_s
        self.baseline = baseline
        # results
        self.ok = False
        self.err: str | None = None
        # segment-2 latency = resume mechanism (reload vs reprefill) + the
        # SAME rem-token decode for both arms, so ours-vs-baseline isolates
        # reload_time - reprefill_time. This is the resume-cost metric.
        self.seg2_e2e_s: float | None = None
        self.total_s: float = 0.0                 # work latency (excl pause)
        self.covered: int | None = None
        self.reloaded: bool | None = None         # ours: True=reload, False=fallback reprefill

    def run(self) -> None:
        work = 0.0
        try:
            if self.icept_at is None:
                _, e = post({"model": MODEL, "prompt": self.prompt,
                             "max_tokens": self.total_out, "temperature": 0.0,
                             "ignore_eos": True})
                work += e
            else:
                rid = f"icept{self.idx}"
                # segment 1 (generates the pre-tool-call query)
                a, e1 = post({"model": MODEL, "prompt": self.prompt,
                              "max_tokens": self.icept_at, "temperature": 0.0,
                              "ignore_eos": True,
                              "vllm_xargs": {"router_req_id": rid}})
                work += e1
                seg1_text = a["choices"][0]["text"]
                # tool call (pause) — NOT counted as work latency
                time.sleep(self.pause_s)
                rem = max(1, self.total_out - self.icept_at)

                # Build segment-2 body: ours reloads from the host
                # checkpoint; baseline (and ours-fallback) reprefills the
                # accumulated context.
                seg2_body = {"model": MODEL, "max_tokens": rem,
                             "temperature": 0.0, "ignore_eos": True}
                if self.baseline == "ours":
                    internal = None
                    for _ in range(20):           # let async ckpt settle (<=2s)
                        internal = read_internal_id(rid)
                        cov = read_covered(internal) if internal else None
                        if cov is not None:
                            self.covered = cov
                            break
                        time.sleep(0.1)
                    if self.covered is not None:
                        self.reloaded = True
                        seg2_body["prompt"] = self.prompt
                        seg2_body["vllm_xargs"] = {
                            "is_rerouted": True,
                            "original_internal_req_id": internal,
                            "num_checkpointed_tokens": self.covered}
                    else:
                        self.reloaded = False     # checkpoint missing -> reprefill
                        seg2_body["prompt"] = self.prompt + seg1_text
                else:
                    seg2_body["prompt"] = self.prompt + seg1_text

                _, e2 = post(seg2_body)
                self.seg2_e2e_s = e2
                work += e2
            self.ok = True
        except Exception as ex:
            self.err = f"{type(ex).__name__}: {ex}"
        finally:
            self.total_s = work


def start_engine(baseline: str, log_path: Path, max_model_len: int,
                 prefix_caching: bool = False):
    env = os.environ.copy()
    env.update({"VLLM_FT_ENGINE_ID": "0", "CUDA_VISIBLE_DEVICES": "0"})
    if baseline == "ours":
        env.update({"FT_ROUTER_SHM_BUS": "1", "FT_CAPACITY_PREEMPT_RELOAD": "1",
                    "FT_CAPACITY_PREEMPT_RELOAD_OVERLAP": "1",
                    "FT_DELTA_CHECKPOINT": "1", "SLO_PRIORITY_PREEMPT": "0"})
    elif baseline == "vllm_fcfs":
        env.update({"FT_ROUTER_SHM_BUS": "0", "FT_CAPACITY_PREEMPT_RELOAD": "0",
                    "FT_CAPACITY_PREEMPT_RELOAD_OVERLAP": "0",
                    "FT_DELTA_CHECKPOINT": "0", "SLO_PRIORITY_PREEMPT": "0"})
    else:
        raise ValueError(baseline)
    # GPU automatic prefix caching: when on, vLLM keeps finished requests'
    # KV blocks resident in HBM and a continuation reuses them (zero-copy)
    # UNLESS LRU-evicted under memory pressure, in which case it recomputes.
    apc_flag = ("--enable-prefix-caching" if prefix_caching
                else "--no-enable-prefix-caching")
    log = open(log_path, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--port", str(PORT),
         "--max-model-len", str(max_model_len), "--gpu-memory-utilization",
         "0.9", "--dtype", "float16", "--enforce-eager",
         apc_flag, "--disable-log-requests"],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (k - f) * (s[f + 1] - s[f])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["ours", "vllm_fcfs"], required=True)
    ap.add_argument("--arrival-rate-qps", type=float, required=True)
    ap.add_argument("--num-requests", type=int, default=40)
    ap.add_argument("--icept-ratio", type=float, default=0.5)
    ap.add_argument("--icept-at", type=int, default=30,
                    help="output token index where the interception happens")
    ap.add_argument("--total-output", type=int, default=120)
    ap.add_argument("--pause-min", type=float, default=2.0)
    ap.add_argument("--pause-max", type=float, default=7.0)
    ap.add_argument("--prompt-min-tokens", type=int, default=2000)
    ap.add_argument("--prompt-max-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--prefix-caching", action="store_true",
                    help="enable vLLM GPU automatic prefix caching (APC). "
                    "Use with --baseline vllm_fcfs for the APC control.")
    ap.add_argument("--out-tag", type=str, default="icept")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in
            open(REPO / "experiments_v2/datasets/cached/arxivsumm.jsonl")]
    pool = [r for r in recs if args.prompt_min_tokens <= r.get(
        "prompt_tokens", 0) <= args.prompt_max_tokens]
    if len(pool) < args.num_requests:
        print(f"[icept] FAIL: only {len(pool)} arxivsumm prompts in "
              f"[{args.prompt_min_tokens}, {args.prompt_max_tokens}] tokens, "
              f"need {args.num_requests} DISTINCT (no repeats, else APC "
              f"would share KV across requests and contaminate the baseline)")
        return 1
    rng.shuffle(pool)
    prompts = pool[:args.num_requests]   # distinct prompts, one per request
    # Poisson arrivals
    offs = [0.0]
    for _ in range(args.num_requests - 1):
        offs.append(offs[-1] + rng.expovariate(args.arrival_rate_qps))
    # interception assignment
    n_icept = int(args.num_requests * args.icept_ratio)
    icept_idx = set(rng.sample(range(args.num_requests), n_icept))

    results_dir = Path(os.environ.get(
        "EVAL_RESULTS_DIR", REPO / "experiments_v2/eval/results/a6000"))
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"icept_{args.baseline}_qps{args.arrival_rate_qps}_"
           f"n{args.num_requests}_seed{args.seed}_{args.out_tag}")
    eng_log = results_dir / f"{tag}_engine.log"
    out = results_dir / f"{tag}_metrics.json"

    for d in ("vllm_ft_checkpoints", "vllm_ft_req_map",
              "vllm_ft_engine_status", "vllm_ft_preempt_queue"):
        subprocess.run(["rm", "-rf", f"/dev/shm/{d}"], check=False)
    print(f"[icept] {args.baseline} qps={args.arrival_rate_qps} "
          f"n={args.num_requests} icept={n_icept}/{args.num_requests} "
          f"pause={args.pause_min}-{args.pause_max}s")
    proc = start_engine(args.baseline, eng_log, args.max_model_len,
                        prefix_caching=args.prefix_caching)
    try:
        if not wait_health(600):
            print("[icept] FAIL: engine not healthy")
            return 1
        print("[icept] engine ready, firing")
        clients: list[IceptClient] = []
        t0 = time.time()
        for i in range(args.num_requests):
            tgt = t0 + offs[i]
            now = time.time()
            if tgt > now:
                time.sleep(tgt - now)
            c = IceptClient(
                idx=i, prompt=prompts[i]["prompt"],
                total_out=args.total_output,
                icept_at=(args.icept_at if i in icept_idx else None),
                pause_s=rng.uniform(args.pause_min, args.pause_max),
                baseline=args.baseline)
            c.start()
            clients.append(c)
        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)
        wall = time.time() - t0

        ok = [c for c in clients if c.ok]
        n_ok = len(ok)
        # firing-window throughput: completed reqs / wall (pauses overlap
        # other requests' work, so this is the realistic system view).
        thru = n_ok / wall if wall > 0 else 0.0
        # resume cost = segment-2 e2e for intercepted requests (decode of
        # rem tokens is identical for both arms, so the ours-vs-baseline
        # difference is exactly reload_time - reprefill_time).
        seg2_lat = [c.seg2_e2e_s for c in ok
                    if c.icept_at is not None and c.seg2_e2e_s is not None]
        work_lat = [c.total_s for c in ok]
        # validity: for 'ours', how many intercepted requests actually
        # reloaded vs fell back to reprefill (checkpoint not ready).
        icept_ok = [c for c in ok if c.icept_at is not None]
        n_reload = sum(1 for c in icept_ok if c.reloaded is True)
        n_fallback = sum(1 for c in icept_ok if c.reloaded is False)
        print(f"[icept] done: {n_ok}/{args.num_requests} ok, wall={wall:.1f}s")
        print(f"[icept] throughput = {thru:.3f} req/s")
        print(f"[icept] segment-2 latency (resume+decode) p50="
              f"{percentile(seg2_lat,50)}  p95={percentile(seg2_lat,95)}"
              f"  n={len(seg2_lat)}")
        def _fmt(x: float | None) -> str:
            return f"{x:.2f}" if x is not None else "n/a"
        _wl50, _wl95 = percentile(work_lat, 50), percentile(work_lat, 95)
        print(f"[icept] work latency p50={_fmt(_wl50)}s p95={_fmt(_wl95)}s")
        if args.baseline == "ours":
            print(f"[icept] reload validity: {n_reload} reloaded / "
                  f"{n_fallback} fell back to reprefill")
        with open(out, "w") as f:
            json.dump({
                "baseline": args.baseline, "prefix_caching": args.prefix_caching,
                "qps": args.arrival_rate_qps,
                "num_requests": args.num_requests, "n_icept": n_icept,
                "n_ok": n_ok, "wall_s": wall, "throughput_req_s": thru,
                "icept_at": args.icept_at, "total_output": args.total_output,
                "pause_range": [args.pause_min, args.pause_max],
                "reload_validity": {"reloaded": n_reload,
                                    "fell_back_reprefill": n_fallback},
                "seg2_latency_s": {
                    "p50": percentile(seg2_lat, 50),
                    "p95": percentile(seg2_lat, 95),
                    "mean": sum(seg2_lat) / len(seg2_lat)
                    if seg2_lat else None,
                    "n": len(seg2_lat)},
                "work_latency_s": {"p50": percentile(work_lat, 50),
                                   "p95": percentile(work_lat, 95)},
                "errors": [c.err for c in clients if c.err][:10],
            }, f, indent=2)
        print(f"[icept] metrics -> {out}")
        return 0
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
