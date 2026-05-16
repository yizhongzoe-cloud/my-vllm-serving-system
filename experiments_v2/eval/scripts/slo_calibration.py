#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SLO calibration: measure baseline P95 TTFT and P95 TPOT on RULER 64K
using vllm_fcfs at low QPS. Used to set the SLO thresholds for E_M1+:
  S_TTFT = 2 × baseline_P95_TTFT
  S_TPOT = 2 × baseline_P95_TPOT
(JITServe-style calibration.)

Standalone: doesn't share infrastructure with e_m3 since the arrival
pattern differs (Poisson here vs fixed 5s in e_m3) and workload differs
(RULER 64K vs ShareGPT).

Run:
  python experiments_v2/eval/scripts/slo_calibration.py \
      --num-requests 30 --arrival-rate-qps 0.05 --seed 0
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

from experiments_v2.eval.workloads.workload_builder import (  # noqa: E402
    build_schedule, summarize_schedule,
)

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
ENGINE_0_PORT = 8401
ENGINE_1_PORT = 8402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(
    os.environ.get(
        "EVAL_RESULTS_DIR", SCRIPT_DIR.parent / "results"
    )
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENGINE_READY_TIMEOUT_S = 300         # 64K context: longer warmup
REQUEST_TIMEOUT_S = 600              # 64K prefill can take 30s+
MAX_MODEL_LEN = 32768                # Qwen2.5-7B-Instruct native: 32768.
                                     # 16K prompt + room for output fits.

REQUEST_DONE_RE = re.compile(
    r"FT request_done req=(\S+) arrival_ts=(\d+\.\d+) "
    r"ttft_ms=(\d+\.\d+) tpot_ms=(\d+\.\d+) e2e_ms=(\d+\.\d+) "
    r"num_output=(\d+) prompt_len=(\d+) finish=(\S+)"
)


def cleanup_shm() -> None:
    for d in ("/dev/shm/vllm_ft_engine_status",
              "/dev/shm/vllm_ft_req_map",
              "/dev/shm/vllm_ft_checkpoints"):
        shutil.rmtree(d, ignore_errors=True)


def start_engine(engine_id: int, port: int, gpu: str,
                 log_path: Path) -> subprocess.Popen:
    """Launch a plain vLLM engine (vllm_fcfs baseline) — all FT off."""
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["FT_ROUTER_SHM_BUS"] = "0"
    env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
    env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
    env["FT_DELTA_CHECKPOINT"] = "0"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.9",   # 64K KV needs more headroom
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[calib] launching engine {engine_id} on GPU {gpu} → {log_path.name}")
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


class Client(threading.Thread):
    """One non-streaming RULER 64K completion against a specific engine."""

    def __init__(self, idx: int, engine_url: str, prompt: str,
                 max_tokens: int, ignore_eos: bool = False) -> None:
        super().__init__(daemon=True)
        self.idx = idx
        self.engine_url = engine_url
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.fire_ts: float | None = None
        self.end_ts: float | None = None
        self.status_code: int | None = None
        self.error: str | None = None
        self.completion_len: int = 0

    def run(self) -> None:
        body: dict = {
            "model": MODEL,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if self.ignore_eos:
            body["ignore_eos"] = True
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.engine_url + "/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self.fire_ts = time.time()
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
    parser.add_argument("--num-requests", type=int, default=30,
                        help="Total request count.")
    parser.add_argument("--arrival-rate-qps", type=float, default=0.05,
                        help="Poisson rate λ (req/s). Default 0.05 = "
                             "1 req per 20s, well below 64K-prompt "
                             "service time so no queueing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="ruler_64k",
                        choices=["ruler_64k", "ruler_16k", "ruler_8k",
                                 "ruler_4k", "ruler_2k", "ruler_1k",
                                 "ruler_mixed", "sharegpt"])
    parser.add_argument("--force-max-output-tokens", type=int, default=None,
                        help="Override per-request max_tokens (paired with "
                             "--ignore-eos) for long-output calibration.")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Set ignore_eos=True so model decodes full "
                             "max_tokens regardless of EOS.")
    args = parser.parse_args()

    # Tag pieces: dataset, optionally a "_out<N>" suffix for long-output
    # variants. Default (no override) keeps the original tag so existing
    # calibration files stay valid.
    dataset_tag = args.dataset
    if args.force_max_output_tokens is not None:
        dataset_tag = f"{args.dataset}_out{int(args.force_max_output_tokens)}"
    tag = (f"slo_calib_{dataset_tag}_n{args.num_requests}_"
           f"qps{args.arrival_rate_qps}_seed{args.seed}")
    engine_0_log = RESULTS_DIR / f"{tag}_engine0.log"
    engine_1_log = RESULTS_DIR / f"{tag}_engine1.log"
    metrics_out = RESULTS_DIR / f"{tag}_metrics.json"

    # Build workload.
    schedule = build_schedule(
        dataset_name=args.dataset,
        num_requests=args.num_requests,
        arrival_rate_qps=args.arrival_rate_qps,
        seed=args.seed,
        force_max_tokens=args.force_max_output_tokens,
    )
    sched_summary = summarize_schedule(schedule)
    print(f"[calib] schedule built: {sched_summary}")
    print(f"[calib] expected duration: ~{sched_summary['window_s']:.0f}s "
          f"+ tail decode")

    cleanup_shm()
    e0 = start_engine(0, ENGINE_0_PORT, "0", engine_0_log)
    e1 = start_engine(1, ENGINE_1_PORT, "1", engine_1_log)
    try:
        ok = (wait_url_ready(
            f"http://127.0.0.1:{ENGINE_0_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ) and wait_url_ready(
            f"http://127.0.0.1:{ENGINE_1_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ))
        if not ok:
            print("[calib] FAIL: an engine never became healthy")
            return 1
        print("[calib] both engines ready")

        engine_urls = [
            f"http://127.0.0.1:{ENGINE_0_PORT}",
            f"http://127.0.0.1:{ENGINE_1_PORT}",
        ]
        clients: list[Client] = []
        t0 = time.time()
        for i, (offset_s, prompt, max_tokens) in enumerate(schedule):
            target_t = t0 + offset_s
            now = time.time()
            if target_t > now:
                time.sleep(target_t - now)
            engine_url = engine_urls[i % len(engine_urls)]
            c = Client(i, engine_url, prompt, max_tokens,
                       ignore_eos=args.ignore_eos)
            c.start()
            clients.append(c)
        print(f"[calib] fired {len(clients)} requests; waiting on completion")

        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)
        n_ok = sum(1 for c in clients
                   if c.error is None and c.status_code == 200)
        n_err = sum(1 for c in clients if c.error is not None)
        print(f"[calib] outcomes: {n_ok}/{len(clients)} 200 OK, "
              f"{n_err} errored")

        events = (parse_request_done_log(engine_0_log)
                  + parse_request_done_log(engine_1_log))
        print(f"[calib] parsed {len(events)} request_done log entries")

        ttft_vals = [e["ttft_ms"] for e in events]
        tpot_vals = [e["tpot_ms"] for e in events if e["tpot_ms"] > 0]
        e2e_vals = [e["e2e_ms"] for e in events]

        p50_ttft = percentile(ttft_vals, 50)
        p95_ttft = percentile(ttft_vals, 95)
        p50_tpot = percentile(tpot_vals, 50)
        p95_tpot = percentile(tpot_vals, 95)

        def fmt(v: float | None) -> str:
            return f"{v:.1f}" if v is not None else "n/a"

        print(f"[calib] TTFT_ms P50={fmt(p50_ttft)} P95={fmt(p95_ttft)}")
        print(f"[calib] TPOT_ms P50={fmt(p50_tpot)} P95={fmt(p95_tpot)}")
        print(f"[calib] E2E_ms  P50={fmt(percentile(e2e_vals, 50))} "
              f"P95={fmt(percentile(e2e_vals, 95))}")
        print(f"[calib] suggested SLO (2x P95):")
        print(f"  S_TTFT = {2 * p95_ttft:.1f} ms" if p95_ttft else "  S_TTFT n/a")
        print(f"  S_TPOT = {2 * p95_tpot:.1f} ms" if p95_tpot else "  S_TPOT n/a")

        metrics_out.write_text(json.dumps({
            "dataset": args.dataset,
            "num_requests": args.num_requests,
            "arrival_rate_qps": args.arrival_rate_qps,
            "seed": args.seed,
            "schedule_summary": sched_summary,
            "n_events": len(events),
            "ttft_ms": {"p50": p50_ttft, "p95": p95_ttft,
                        "mean": statistics.mean(ttft_vals) if ttft_vals else None},
            "tpot_ms": {"p50": p50_tpot, "p95": p95_tpot,
                        "mean": statistics.mean(tpot_vals) if tpot_vals else None},
            "e2e_ms": {"p50": percentile(e2e_vals, 50),
                       "p95": percentile(e2e_vals, 95)},
            "suggested_slo": {
                "S_TTFT_ms": 2 * p95_ttft if p95_ttft else None,
                "S_TPOT_ms": 2 * p95_tpot if p95_tpot else None,
            },
            "client_outcomes": {"200_ok": n_ok, "errored": n_err},
            "per_request": events,
        }, indent=2))
        print(f"[calib] metrics → {metrics_out}")
        return 0
    finally:
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
