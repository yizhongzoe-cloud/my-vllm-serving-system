#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E_D1: Disruption recovery demo — measure failover_gap distribution.

Non-streaming. failover_gap measured from engine-side log timestamps:
  - kill_ts: time.time() recorded immediately before SIGKILL engine 0.
  - first_post_reroute_token_ts: extracted from engine 1's log line
        `FT first_post_reroute_token req=<id> ts=<ts>`
    which is emitted (scheduler.py) on the first token append of any
    request with is_rerouted=True.
  - failover_gap(req) = first_post_reroute_token_ts(req) − kill_ts

Baselines (selected via --baseline):
  ours              : V3 reload + checkpoint mirror enabled. KV restored from
                      /dev/shm; engine 1 resumes decode quickly.
  reroute_no_ckpt   : V3 reload disabled. Engine 0 publishes no checkpoint.
                      Router still reroutes; engine 1 does vanilla prefill
                      from prompt. Stands in for Llumnix-class behavior under
                      disruption (live migration impossible when source is
                      dead, so we capture the no-checkpoint fallback).

Run:
  python experiments_v2/eval/scripts/e_d1_disruption_demo.py --baseline ours
"""
import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Make experiments_v2.* importable when launched as a script
# (otherwise build_prompts → workload_builder import fails).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL = os.path.expanduser("~/model/Qwen2.5-14B-Instruct")
ENGINE_0_PORT = 8401
ENGINE_1_PORT = 8402
ROUTER_PORT = 8400
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(
    os.environ.get(
        "EVAL_RESULTS_DIR", SCRIPT_DIR.parent / "results"
    )
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENGINE_READY_TIMEOUT_S = 240
ROUTER_READY_TIMEOUT_S = 30
MAX_MODEL_LEN = 32768               # arxivsumm prompts up to ~30K + output
MAX_OUTPUT_TOKENS = 200
NUM_REQUESTS = 40                   # 40 reqs at QPS=0.25 → ~160s schedule
DATASET_NAME = "arxivsumm"          # matches main paper experiments
ARRIVAL_RATE_QPS = 0.25             # Poisson rate. Just at single-engine
                                    # sustained edge (~0.2 for 14B
                                    # arxivsumm). Post-kill survivor at
                                    # 0.25 is slight overload — acceptable
                                    # for failover demo. Higher density
                                    # ensures engine 0 has in-flight
                                    # requests at kill time (QPS=0.15
                                    # was too sparse).
KILL_AT_TIME_S = 25                 # kill engine 0 at t=25s — early enough
                                    # that ~3-4 requests are mid-decode
                                    # on engine 0 (each takes ~13s e2e).
CHUNK_WAIT_TIMEOUT_S = 90           # for ours: at least one ckpt chunk lands
REROUTE_DETECT_TIMEOUT_S = 30
REQUEST_TIMEOUT_S = 300             # generous; reroute_no_ckpt reprefills up to 30K tokens

FIRST_TOKEN_LOG_RE = re.compile(
    r"FT first_post_reroute_token req=(\S+) ts=(\d+\.\d+)"
)


def cleanup_shm() -> None:
    for d in (STATUS_DIR, REQ_MAP_DIR, CKPT_DIR):
        shutil.rmtree(d, ignore_errors=True)


def start_engine(
    engine_id: int, port: int, gpu: str, log_path: Path, baseline: str
) -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_ROUTER_SHM_BUS"] = "1"
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if baseline == "ours":
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
        env["FT_DELTA_CHECKPOINT"] = "1"
    elif baseline == "reroute_no_ckpt":
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
    else:
        raise ValueError(f"unsupported baseline: {baseline}")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.9",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[E_D1] launching engine {engine_id} on GPU {gpu} → {log_path.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def start_router(log_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "experiments_v2.router.router",
        "--port", str(ROUTER_PORT),
        "--engines",
        f"0=http://127.0.0.1:{ENGINE_0_PORT}",
        f"1=http://127.0.0.1:{ENGINE_1_PORT}",
        "--log-level", "info",
    ]
    log_f = open(log_path, "w")
    print(f"[E_D1] launching router → {log_path.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        start_new_session=True,
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
        time.sleep(2)
    return False


def fetch_json(url: str, timeout: int = 5) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def wait_both_engines_alive_in_router(timeout_s: int = 10) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        h = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/health")
        if h is not None:
            engines = h.get("engines", {})
            if (engines.get("0", {}).get("alive")
                    and engines.get("1", {}).get("alive")):
                return True
        time.sleep(1)
    return False


def build_schedule_for_e_d1(num_requests: int, seed: int) -> list[tuple[float, str, int]]:
    """Return a (offset_s, prompt, max_tokens) Poisson schedule on
    arxivsumm — same workload distribution as the main microbench, so
    failover_gap is measured against realistic steady-state load."""
    from experiments_v2.eval.workloads.workload_builder import build_schedule
    return build_schedule(
        dataset_name=DATASET_NAME,
        num_requests=num_requests,
        arrival_rate_qps=ARRIVAL_RATE_QPS,
        seed=seed,
    )


class Client(threading.Thread):
    """Single non-streaming completion request. Records timing + outcome.
    Schedule has its own max_tokens per request (from dataset)."""

    def __init__(self, idx: int, prompt_body: str, max_tokens: int) -> None:
        super().__init__(daemon=True)
        self.idx = idx
        self.prompt_body = prompt_body
        self.max_tokens = max_tokens
        self.start_ts: float | None = None
        self.end_ts: float | None = None
        self.status_code: int | None = None
        self.error: str | None = None
        self.completion_len: int = 0

    def run(self) -> None:
        # Per-request unique suffix so prefix-cache (already disabled in the
        # engine launch args) can't help even if accidentally re-enabled.
        prompt = f"{self.prompt_body} Request {self.idx}."
        payload = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{ROUTER_PORT}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self.start_ts = time.time()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
                self.status_code = r.status
                data = json.loads(r.read())
                if "choices" in data:
                    self.completion_len = len(
                        data["choices"][0].get("text", "")
                    )
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.end_ts = time.time()


def wait_for_n_req_maps(n: int, timeout_s: int) -> int:
    """Block until /dev/shm/vllm_ft_req_map/ has >= n entries (or timeout).
    Returns the count seen.
    """
    deadline = time.time() + timeout_s
    last = 0
    while time.time() < deadline:
        if REQ_MAP_DIR.exists():
            last = len(list(REQ_MAP_DIR.iterdir()))
            if last >= n:
                return last
        time.sleep(0.3)
    return last


def wait_for_any_chunk(timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if CKPT_DIR.exists():
            if any(CKPT_DIR.rglob("chunk_*.pt")):
                return True
        time.sleep(0.5)
    return False


def scrape_engine_assignments() -> dict[str, int]:
    """{router_req_id → engine_id} from /dev/shm/vllm_ft_req_map/."""
    out: dict[str, int] = {}
    if not REQ_MAP_DIR.exists():
        return out
    for fp in REQ_MAP_DIR.iterdir():
        try:
            data = json.loads(fp.read_text())
            out[fp.name] = int(data.get("engine_id"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def wait_for_reroute(timeout_s: int) -> tuple[bool, dict]:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        m = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/metrics")
        if m is not None:
            last = m
            if (m.get("dead_event_count", 0) >= 1
                    and m.get("reroute_count", 0) >= 1):
                return True, m
        time.sleep(0.3)
    return False, last


def shutdown(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_first_token_log(log_path: Path) -> list[tuple[str, float]]:
    """Extract (req_id, log_ts) pairs from engine 1's log."""
    out: list[tuple[str, float]] = []
    try:
        txt = log_path.read_text(errors="replace")
    except OSError:
        return out
    for m in FIRST_TOKEN_LOG_RE.finditer(txt):
        try:
            out.append((m.group(1), float(m.group(2))))
        except ValueError:
            continue
    return out


def main() -> int:
    # Use router default (least_load) — verified 2026-05-17 to be the
    # correct choice for main experiments. e_d1 also uses Poisson
    # arrival now, so the load distribution is similar to the main
    # microbench.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", choices=["ours", "reroute_no_ckpt"], default="ours",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    args = parser.parse_args()

    n_req = args.num_requests
    tag = f"e_d1_{args.baseline}_n{n_req}_seed{args.seed}"
    engine_0_log = RESULTS_DIR / f"{tag}_engine0.log"
    engine_1_log = RESULTS_DIR / f"{tag}_engine1.log"
    router_log = RESULTS_DIR / f"{tag}_router.log"
    metrics_out = RESULTS_DIR / f"{tag}_metrics.json"

    cleanup_shm()
    e0 = start_engine(0, ENGINE_0_PORT, "0", engine_0_log, args.baseline)
    e1 = start_engine(1, ENGINE_1_PORT, "1", engine_1_log, args.baseline)
    router = None
    try:
        if not (wait_url_ready(f"http://127.0.0.1:{ENGINE_0_PORT}/health",
                               ENGINE_READY_TIMEOUT_S)
                and wait_url_ready(f"http://127.0.0.1:{ENGINE_1_PORT}/health",
                                   ENGINE_READY_TIMEOUT_S)):
            print("[E_D1] FAIL: an engine never became healthy")
            return 1
        print("[E_D1] both engines ready")

        router = start_router(router_log)
        if not wait_url_ready(f"http://127.0.0.1:{ROUTER_PORT}/health",
                              ROUTER_READY_TIMEOUT_S):
            print("[E_D1] FAIL: router not healthy")
            return 1
        if not wait_both_engines_alive_in_router():
            print("[E_D1] FAIL: router didn't see both engines alive")
            return 1
        print("[E_D1] router ready")

        schedule = build_schedule_for_e_d1(n_req, args.seed)
        print(f"[E_D1] Poisson schedule: n={n_req} QPS={ARRIVAL_RATE_QPS} "
              f"window={schedule[-1][0]:.1f}s  kill_at=t+{KILL_AT_TIME_S}s")

        # Fire clients one-by-one according to Poisson offsets. Spawn a
        # background kill thread that triggers SIGKILL at fire_ts + KILL_AT_TIME_S.
        clients: list[Client] = []
        fire_ts = time.time()
        kill_event = threading.Event()
        kill_ts_holder: dict = {"ts": None}

        def _kill_at_time():
            target = fire_ts + KILL_AT_TIME_S
            now = time.time()
            if target > now:
                time.sleep(target - now)
            # For ours: confirm at least one ckpt chunk exists before kill.
            if args.baseline == "ours":
                has_chunks = (CKPT_DIR.exists()
                              and any(CKPT_DIR.rglob("chunk_*.pt")))
                if not has_chunks:
                    print("[E_D1] WARN: kill triggered but no ckpt chunks "
                          "visible yet — reroute may fall back to reprefill")
            pre_kill = scrape_engine_assignments()
            kill_ts_holder["eng0_count"] = sum(
                1 for v in pre_kill.values() if v == 0
            )
            kill_ts_holder["eng1_count"] = sum(
                1 for v in pre_kill.values() if v == 1
            )
            print(f"[E_D1] pre-kill: engine0={kill_ts_holder['eng0_count']}, "
                  f"engine1={kill_ts_holder['eng1_count']}, "
                  f"total_dispatched={len(pre_kill)}")
            kill_ts_holder["ts"] = time.time()
            shutdown(e0, sig=signal.SIGKILL)
            print(f"[E_D1] SIGKILL engine 0 at t={kill_ts_holder['ts']:.3f} "
                  f"(t={kill_ts_holder['ts']-fire_ts:.1f}s into schedule)")
            kill_event.set()

        kill_thread = threading.Thread(target=_kill_at_time, daemon=True)
        kill_thread.start()
        print(f"[E_D1] starting Poisson dispatch at t={fire_ts:.3f}")

        for i, (offset_s, prompt, max_tokens) in enumerate(schedule):
            target_t = fire_ts + offset_s
            now = time.time()
            if target_t > now:
                time.sleep(target_t - now)
            c = Client(i, prompt, max_tokens)
            c.start()
            clients.append(c)

        # Wait for kill thread (it should already have fired by now since
        # KILL_AT_TIME_S < schedule window for most configs).
        kill_thread.join(timeout=10)
        if not kill_event.is_set():
            print("[E_D1] FAIL: kill thread didn't trigger")
            return 1
        kill_ts = kill_ts_holder["ts"]

        ok, m = wait_for_reroute(REROUTE_DETECT_TIMEOUT_S)
        print(f"[E_D1] router reroute observed: {'OK' if ok else 'FAIL'} — {m}")
        if not ok:
            return 1

        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)

        # Parse engine 1 log for first-token-after-reroute timestamps.
        first_token_events = parse_first_token_log(engine_1_log)
        gaps_ms = [
            (req_id, (ts - kill_ts) * 1000.0)
            for req_id, ts in first_token_events
            if ts > kill_ts  # filter out anything weird (shouldn't happen)
        ]

        n_completed = sum(1 for c in clients
                          if c.error is None and c.status_code == 200)
        n_errored = sum(1 for c in clients if c.error is not None)

        if gaps_ms:
            gap_vals = [g for _, g in gaps_ms]
            p50 = statistics.median(gap_vals)
            mean_v = statistics.mean(gap_vals)
            max_v = max(gap_vals)
            min_v = min(gap_vals)
            print(f"[E_D1] failover_gap_ms over N={len(gap_vals)} rerouted: "
                  f"min={min_v:.0f}, P50={p50:.0f}, mean={mean_v:.0f}, "
                  f"max={max_v:.0f}")
        else:
            p50 = mean_v = max_v = min_v = None
            print("[E_D1] no first_post_reroute_token log entries found")

        print(f"[E_D1] client outcomes: {n_completed}/{n_req} 200 OK, "
              f"{n_errored} errored")

        metrics_out.write_text(json.dumps({
            "baseline": args.baseline,
            "seed": args.seed,
            "num_requests": n_req,
            "kill_ts": kill_ts,
            "pre_kill_engine_counts": {
                "0": kill_ts_holder.get("eng0_count", 0),
                "1": kill_ts_holder.get("eng1_count", 0),
            },
            "kill_at_time_s": KILL_AT_TIME_S,
            "arrival_rate_qps": ARRIVAL_RATE_QPS,
            "num_first_token_events": len(gaps_ms),
            "failover_gaps_ms": [
                {"req_id": rid, "gap_ms": g} for rid, g in gaps_ms
            ],
            "failover_gap_stats_ms": {
                "n": len(gaps_ms),
                "min": min_v, "p50": p50, "mean": mean_v, "max": max_v,
            },
            "client_outcomes": {
                "200_ok": n_completed,
                "errored": n_errored,
                "per_client": [
                    {
                        "idx": c.idx,
                        "status_code": c.status_code,
                        "error": c.error,
                        "completion_len": c.completion_len,
                        "total_s": (c.end_ts - c.start_ts)
                        if (c.start_ts and c.end_ts) else None,
                    }
                    for c in clients
                ],
            },
            "router_metrics_at_end": fetch_json(
                f"http://127.0.0.1:{ROUTER_PORT}/metrics"
            ),
        }, indent=2))
        print(f"[E_D1] metrics → {metrics_out}")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
