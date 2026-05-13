#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E_M3: System overhead under healthy, low-load workload.

Goal: show that our system (router + V3 reload + slack picker + ckpt
publishing) does not add measurable overhead vs simpler baselines, in
the absence of disruption.

Three baselines:
  vllm_fcfs       : plain vLLM, no router. Client round-robins requests
                    directly against 2 engines' OpenAI endpoints.
  reroute_no_ckpt : our router + 2 engines, but FT features off
                    (FT_CAPACITY_PREEMPT_RELOAD=0, *_OVERLAP=0,
                    FT_DELTA_CHECKPOINT=0). Router still serves requests
                    through its dispatch path, but no ckpt publish, no
                    V3 reload state machine.
  ours            : full system (FT features on).

Workload: ShareGPT (sampled), 40 requests at fixed 5s inter-arrival.
No engine kill. No SLO passed in extra_args (so slack picker stays
dormant — we're testing the *unconditional* overhead).

Metrics (from `FT request_done` log line emitted by engine on each
finish):
  - TTFT_ms P50 / P95
  - TPOT_ms P50 / P95
  - throughput_tok_per_s = total_output_tokens / (t_last_done − t_first_start)

3 seeds per baseline.

Run:
  python experiments_v2/eval/scripts/e_m3_overhead.py --baseline ours --seed 0
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

# Make `experiments_v2.*` importable when launched as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
ENGINE_0_PORT = 8401
ENGINE_1_PORT = 8402
ROUTER_PORT = 8400
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SHAREGPT_PATH = (Path(__file__).resolve().parents[2]
                 / "datasets" / "cached" / "sharegpt_5000.jsonl")

ENGINE_READY_TIMEOUT_S = 240
ROUTER_READY_TIMEOUT_S = 30
NUM_REQUESTS = 40
INTER_ARRIVAL_S = 5.0
MAX_MODEL_LEN = 4096          # ShareGPT prompts max ~4K
DEFAULT_MAX_OUTPUT = 400      # cap so a few outliers don't bloat e2e
REQUEST_TIMEOUT_S = 120

REQUEST_DONE_RE = re.compile(
    r"FT request_done req=(\S+) arrival_ts=(\d+\.\d+) "
    r"ttft_ms=(\d+\.\d+) tpot_ms=(\d+\.\d+) e2e_ms=(\d+\.\d+) "
    r"num_output=(\d+) prompt_len=(\d+) finish=(\S+)"
)


def cleanup_shm() -> None:
    for d in (STATUS_DIR, REQ_MAP_DIR, CKPT_DIR):
        shutil.rmtree(d, ignore_errors=True)


def start_engine(
    engine_id: int, port: int, gpu: str, log_path: Path, baseline: str
) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # FT_ROUTER_SHM_BUS:
    #   - ours, reroute_no_ckpt: router needs status writes to detect engines
    #   - vllm_fcfs: bypasses router entirely, so no shm bus needed
    if baseline in ("ours", "reroute_no_ckpt"):
        env["FT_ROUTER_SHM_BUS"] = "1"
    else:
        env["FT_ROUTER_SHM_BUS"] = "0"
    # FT feature ablation matrix (3 independent env vars):
    #   FT_DELTA_CHECKPOINT       — ckpt publish to /dev/shm
    #   FT_CAPACITY_PREEMPT_RELOAD          — slack-based preempt picker
    #   FT_CAPACITY_PREEMPT_RELOAD_OVERLAP  — V3 reload state machine
    # Two ablation variants split the FT cost: ckpt vs runtime (picker + V3).
    if baseline == "ours":
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
    elif baseline == "ours_ckpt_only":
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
    elif baseline == "ours_runtime_only":
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
    else:  # reroute_no_ckpt or vllm_fcfs — everything off
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN + DEFAULT_MAX_OUTPUT + 256),
        "--gpu-memory-utilization", "0.5",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[E_M3] launching engine {engine_id} on GPU {gpu} → {log_path.name}")
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
    print(f"[E_M3] launching router → {log_path.name}")
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


def load_workload(num_requests: int, seed: int) -> list[dict]:
    from experiments_v2.datasets.loader import load_dataset
    records = load_dataset(
        "sharegpt", str(SHAREGPT_PATH),
        max_samples=num_requests, seed=seed,
    )
    if len(records) < num_requests:
        raise RuntimeError(
            f"only {len(records)} ShareGPT records loaded "
            f"(asked {num_requests})"
        )
    return records


class Client(threading.Thread):
    """One non-streaming completion request."""

    def __init__(
        self, idx: int, target_url: str, prompt: str, max_tokens: int,
    ) -> None:
        super().__init__(daemon=True)
        self.idx = idx
        self.target_url = target_url
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.start_ts: float | None = None
        self.end_ts: float | None = None
        self.status_code: int | None = None
        self.error: str | None = None
        self.completion_text: str = ""

    def run(self) -> None:
        payload = json.dumps({
            "model": MODEL,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.target_url + "/v1/completions",
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
                    self.completion_text = data["choices"][0].get("text", "")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.end_ts = time.time()


def shutdown(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_request_done_log(log_path: Path) -> list[dict]:
    """Extract per-request timing dicts from engine log."""
    out: list[dict] = []
    try:
        txt = log_path.read_text(errors="replace")
    except OSError:
        return out
    for m in REQUEST_DONE_RE.finditer(txt):
        out.append({
            "req_id": m.group(1),
            "arrival_ts": float(m.group(2)),
            "ttft_ms": float(m.group(3)),
            "tpot_ms": float(m.group(4)),
            "e2e_ms": float(m.group(5)),
            "num_output": int(m.group(6)),
            "prompt_len": int(m.group(7)),
            "finish": m.group(8),
        })
    return out


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        choices=[
            "vllm_fcfs", "reroute_no_ckpt",
            "ours_ckpt_only", "ours_runtime_only", "ours",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    parser.add_argument(
        "--inter-arrival-s", type=float, default=INTER_ARRIVAL_S,
    )
    args = parser.parse_args()

    tag = f"e_m3_{args.baseline}_n{args.num_requests}_seed{args.seed}"
    engine_0_log = RESULTS_DIR / f"{tag}_engine0.log"
    engine_1_log = RESULTS_DIR / f"{tag}_engine1.log"
    router_log = RESULTS_DIR / f"{tag}_router.log"
    metrics_out = RESULTS_DIR / f"{tag}_metrics.json"

    records = load_workload(args.num_requests, args.seed)
    print(f"[E_M3] loaded {len(records)} ShareGPT records "
          f"(prompt P50≈{statistics.median([r['prompt_tokens'] for r in records]):.0f} "
          f"tokens, output target P50≈{statistics.median([r['expected_output_tokens'] for r in records]):.0f})")

    cleanup_shm()
    e0 = start_engine(0, ENGINE_0_PORT, "0", engine_0_log, args.baseline)
    e1 = start_engine(1, ENGINE_1_PORT, "1", engine_1_log, args.baseline)
    router = None
    try:
        if not (wait_url_ready(f"http://127.0.0.1:{ENGINE_0_PORT}/health",
                               ENGINE_READY_TIMEOUT_S)
                and wait_url_ready(f"http://127.0.0.1:{ENGINE_1_PORT}/health",
                                   ENGINE_READY_TIMEOUT_S)):
            print("[E_M3] FAIL: an engine never became healthy")
            return 1
        print("[E_M3] both engines ready")

        if args.baseline == "vllm_fcfs":
            target_urls = [
                f"http://127.0.0.1:{ENGINE_0_PORT}",
                f"http://127.0.0.1:{ENGINE_1_PORT}",
            ]
            print("[E_M3] vllm_fcfs: bypassing router, round-robin "
                  "to engine endpoints")
        else:
            router = start_router(router_log)
            if not wait_url_ready(
                f"http://127.0.0.1:{ROUTER_PORT}/health",
                ROUTER_READY_TIMEOUT_S,
            ):
                print("[E_M3] FAIL: router not healthy")
                return 1
            target_urls = [f"http://127.0.0.1:{ROUTER_PORT}"]
            print(f"[E_M3] {args.baseline}: using router")

        clients: list[Client] = []
        first_start = time.time()
        for i, rec in enumerate(records):
            target = target_urls[i % len(target_urls)]
            max_tokens = min(
                DEFAULT_MAX_OUTPUT,
                max(1, rec["expected_output_tokens"]),
            )
            c = Client(i, target, rec["prompt"], max_tokens)
            c.start()
            clients.append(c)
            if i < len(records) - 1:
                time.sleep(args.inter_arrival_s)
        print(f"[E_M3] all {len(clients)} requests fired over "
              f"{(len(clients) - 1) * args.inter_arrival_s:.0f}s window")

        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)

        n_ok = sum(1 for c in clients
                   if c.error is None and c.status_code == 200)
        n_err = sum(1 for c in clients if c.error is not None)
        print(f"[E_M3] outcomes: {n_ok}/{len(clients)} 200 OK, "
              f"{n_err} errored")

        events = (parse_request_done_log(engine_0_log)
                  + parse_request_done_log(engine_1_log))
        print(f"[E_M3] parsed {len(events)} request_done log entries")

        ttft_vals = [e["ttft_ms"] for e in events]
        tpot_vals = [e["tpot_ms"] for e in events if e["tpot_ms"] > 0]
        e2e_vals = [e["e2e_ms"] for e in events]
        total_out_tokens = sum(e["num_output"] for e in events)

        first_arrival_ts = min((e["arrival_ts"] for e in events), default=None)
        last_done_ts = max(
            (c.end_ts for c in clients if c.end_ts), default=None,
        )
        if first_arrival_ts is not None and last_done_ts is not None:
            window_s = last_done_ts - first_arrival_ts
            throughput_tok_per_s = total_out_tokens / max(1e-6, window_s)
        else:
            window_s = None
            throughput_tok_per_s = None

        def fmt(v: float | None) -> str:
            return f"{v:.1f}" if v is not None else "n/a"

        print(f"[E_M3] TTFT_ms P50={fmt(percentile(ttft_vals, 50))} "
              f"P95={fmt(percentile(ttft_vals, 95))}")
        print(f"[E_M3] TPOT_ms P50={fmt(percentile(tpot_vals, 50))} "
              f"P95={fmt(percentile(tpot_vals, 95))}")
        print(f"[E_M3] E2E_ms  P50={fmt(percentile(e2e_vals, 50))} "
              f"P95={fmt(percentile(e2e_vals, 95))}")
        print(f"[E_M3] throughput: {fmt(throughput_tok_per_s)} tok/s "
              f"({total_out_tokens} tokens / {window_s:.1f}s)" if window_s
              else "[E_M3] throughput: n/a")

        metrics_out.write_text(json.dumps({
            "baseline": args.baseline,
            "seed": args.seed,
            "num_requests": args.num_requests,
            "inter_arrival_s": args.inter_arrival_s,
            "first_arrival_ts": first_arrival_ts,
            "last_done_ts": last_done_ts,
            "window_s": window_s,
            "total_output_tokens": total_out_tokens,
            "throughput_tok_per_s": throughput_tok_per_s,
            "ttft_ms": {
                "n": len(ttft_vals),
                "p50": percentile(ttft_vals, 50),
                "p95": percentile(ttft_vals, 95),
                "mean": statistics.mean(ttft_vals) if ttft_vals else None,
            },
            "tpot_ms": {
                "n": len(tpot_vals),
                "p50": percentile(tpot_vals, 50),
                "p95": percentile(tpot_vals, 95),
                "mean": statistics.mean(tpot_vals) if tpot_vals else None,
            },
            "e2e_ms": {
                "n": len(e2e_vals),
                "p50": percentile(e2e_vals, 50),
                "p95": percentile(e2e_vals, 95),
                "mean": statistics.mean(e2e_vals) if e2e_vals else None,
            },
            "client_outcomes": {
                "200_ok": n_ok,
                "errored": n_err,
            },
            "per_request": events,
        }, indent=2))
        print(f"[E_M3] metrics → {metrics_out}")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
