#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E_M1: SLO attainment vs load (main figure of the paper).

WHAT THE EXPERIMENT DOES
========================
For each (system, qps, seed) tuple: spin up 2 vllm engines + (optionally)
the router, then fire N requests Poisson-distributed at the given QPS.
Per request we record TTFT (time-to-first-token) and TPOT (time-per-
output-token) from engine logs. A request meets SLO iff TTFT ≤ S_TTFT
AND TPOT ≤ S_TPOT. The headline metric is "SLO attainment %" — the
fraction of requests that meet SLO.

The point of the sweep is to plot this attainment as a function of load
(QPS) for the 3 systems, on the same figure. Higher load → more queue
pressure → more SLO violations; the systems are differentiated by how
well they protect tight-SLO requests when the engines are contended.

3 SYSTEMS COMPARED
==================
  vllm_fcfs       : 2 plain vLLM engines, client round-robins. No router,
                    no FT plumbing. This is the "industry floor".
  reroute_no_ckpt : router + 2 engines but FT_CAPACITY_PREEMPT_RELOAD=0,
                    SLO_PRIORITY_PREEMPT=0. Stand-in for "live migration
                    serving" papers (Llumnix-style); reroute happens but
                    target engine has to fully reprefill.
  ours            : router + 2 engines, full FT (host KV checkpoint +
                    slack picker + V3 reload). Picker is gated by
                    SLO_PRIORITY_PREEMPT=1 (engine env). Client must
                    also pass ttft_slo_ms / tpot_slo_ms in vllm_xargs so
                    the picker can compare slack across requests.

For non-ours baselines the picker env is set to 0, and even if a client
passes SLO values the engine ignores them (picker never runs).

SLO MODE: uniform vs tiered
===========================
--slo-mode uniform: all N requests get the same (ttft_slo_ms,
  tpot_slo_ms). Useful for "system-vs-system at a single SLO point".

--slo-mode tiered (Niyama-style): N requests are split into 3 QoS
  classes (tight / normal / loose). A balanced shuffled assignment
  seeded by --seed (different stream from prompt sampling) gives each
  class ⌊N/3⌋ or ⌈N/3⌉ members. Each class has its own (ttft,tpot) SLO.
  Required args (ms): --ttft-slo-{tight,normal,loose}-ms and
  --tpot-slo-{tight,normal,loose}-ms. The metrics JSON breaks down
  attainment per class.

The paper plot uses tiered mode — it stresses the slack picker, which
is fundamentally a tight-vs-loose triage mechanism.

WHERE THE SLO NUMBERS COME FROM (don't hand-tune!)
==================================================
SLO calibration runs once per (dataset, hardware) in:
    experiments_v2/eval/scripts/slo_calibration.py

The calibrator measures baseline P95 TTFT and P95 TPOT at LOW LOAD
(qps≈0.1 for ShareGPT, qps≈0.04 for RULER 64K — basically uncontended)
on plain vLLM (no FT plumbing). The 3 SLO tiers are then defined as:

    tight   = baseline_p95 × 1.5
    normal  = baseline_p95 × 3.0
    loose   = baseline_p95 × 6.0

For ShareGPT on the A6000 the calibration produced
    baseline TPOT P95 ≈ 22ms → tiers 33/66/132 ms
    baseline TTFT P95 ≈ 456ms → tiers 684/1368/2736 ms
(see results/slo_calib_sharegpt_n30_qps0.1_seed0_metrics.json)

IMPORTANT: the tight SLO is intentionally tight enough that the system
under high load will fail it unless scheduling is SLO-aware. But it's
calibrated against LOW LOAD baseline, so at very high QPS even ours
will fail tight SLO because TPOT itself grows with batch size — the
picker only affects admission (TTFT), not decode speed (TPOT). When
interpreting results, separate "TTFT-bound failures" (picker can save)
from "TPOT-bound failures" (no scheduling can save without re-calib).

HOW TO RUN
==========
Single point (always set PYTHONPATH to repo root so the experiments_v2
package import works from -m):

  cd /path/to/my-vllm-serving-system
  PYTHONPATH=. python -m experiments_v2.eval.scripts.e_m1_slo_sweep \\
    --baseline ours --dataset sharegpt \\
    --arrival-rate-qps 4.0 --num-requests 60 --seed 0 \\
    --slo-mode tiered \\
    --ttft-slo-tight-ms 684 --ttft-slo-normal-ms 1368 \\
    --ttft-slo-loose-ms 2736 \\
    --tpot-slo-tight-ms 33 --tpot-slo-normal-ms 66 \\
    --tpot-slo-loose-ms 132

Full sweep (5 QPS × 3 baselines × 3 seeds): drive this script from a
shell for-loop. Engine startup is ~3-4 min per point (loading the 7B
model), so plan ~6-8 min per run. Between runs clear /dev/shm:

  rm -rf /dev/shm/vllm_ft_{preempt_queue,engine_status,req_map,checkpoints}

Output: results/e_m1_<baseline>_<dataset>_qps<x>_n<n>_seed<s>_metrics.json
with per-request rows + per-class aggregates (when tiered mode is on)
+ overall {p50, p95, slo_met%}.
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
    build_schedule, build_mixed_schedule, build_trace_schedule,
    summarize_schedule,
)

MODEL = os.path.expanduser("~/model/Qwen2.5-14B-Instruct")
ENGINE_0_PORT = 8401
ENGINE_1_PORT = 8402
ROUTER_PORT = 8400

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(
    os.environ.get(
        "EVAL_RESULTS_DIR", SCRIPT_DIR.parent / "results"
    )
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENGINE_READY_TIMEOUT_S = 300       # 64K context warmup
ROUTER_READY_TIMEOUT_S = 30
REQUEST_TIMEOUT_S = 600

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
                 log_path: Path, baseline: str,
                 max_model_len: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # FT flags by baseline.
    if baseline == "ours":
        env["FT_ROUTER_SHM_BUS"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["SLO_PRIORITY_PREEMPT"] = "1"
    elif baseline == "ours_no_picker":
        # E_M4 ablation: full FT machinery (ckpt + V3 reload) but slack
        # picker disabled. Tells us how much the picker policy itself
        # contributes vs the mechanism.
        env["FT_ROUTER_SHM_BUS"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
        env["FT_DELTA_CHECKPOINT"] = "1"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    elif baseline == "reroute_no_ckpt":
        env["FT_ROUTER_SHM_BUS"] = "1"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    elif baseline == "vllm_fcfs":
        env["FT_ROUTER_SHM_BUS"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD"] = "0"
        env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "0"
        env["FT_DELTA_CHECKPOINT"] = "0"
        env["SLO_PRIORITY_PREEMPT"] = "0"
    else:
        raise ValueError(f"unsupported baseline: {baseline}")
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
    print(f"[E_M1] launching engine {engine_id} on GPU {gpu} ({baseline}) "
          f"→ {log_path.name}")
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
    print(f"[E_M1] launching router → {log_path.name}")
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
        time.sleep(3)
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


class Client(threading.Thread):
    """Single non-streaming completion. Optionally includes SLO in extra_args
    so the engine's slack picker can use it.
    """

    def __init__(self, idx: int, target_url: str, prompt: str,
                 max_tokens: int, ttft_slo_ms: float | None,
                 tpot_slo_ms: float | None,
                 ignore_eos: bool = False) -> None:
        super().__init__(daemon=True)
        self.idx = idx
        self.target_url = target_url
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.ttft_slo_ms = ttft_slo_ms
        self.tpot_slo_ms = tpot_slo_ms
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
            # vLLM's OpenAI server accepts ignore_eos as a top-level
            # field; with it set the model decodes max_tokens tokens
            # regardless of any EOS the sampler would emit. Required
            # to make a "force long output" stress workload.
            body["ignore_eos"] = True
        if self.ttft_slo_ms is not None or self.tpot_slo_ms is not None:
            xargs = {}
            if self.ttft_slo_ms is not None:
                xargs["ttft_slo_ms"] = self.ttft_slo_ms
            if self.tpot_slo_ms is not None:
                xargs["tpot_slo_ms"] = self.tpot_slo_ms
            body["vllm_xargs"] = xargs
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.target_url + "/v1/completions",
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
    parser.add_argument("--baseline",
                        choices=["vllm_fcfs", "reroute_no_ckpt",
                                 "ours_no_picker", "ours"],
                        required=True)
    parser.add_argument("--dataset",
                        choices=["ruler_64k", "ruler_16k", "ruler_8k",
                                 "ruler_4k", "ruler_2k", "ruler_1k",
                                 "ruler_mixed", "sharegpt", "arxivsumm",
                                 "mixed_short_long", "burstgpt_mixed"],
                        required=True)
    parser.add_argument("--mixed-short-ratio", type=float, default=0.7,
                        help="(dataset=mixed_short_long) fraction of "
                             "requests from short dataset (sharegpt). "
                             "Long fraction = 1 - this. Default 0.7.")
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--arrival-rate-qps", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ttft-slo-ms", type=float, default=None,
                        help="(uniform mode) Single SLO threshold for TTFT, "
                             "used for SLO_met%% computation AND passed to "
                             "engine via vllm_xargs (so slack picker sees it).")
    parser.add_argument("--tpot-slo-ms", type=float, default=None,
                        help="(uniform mode) Single SLO threshold for TPOT.")
    # ---- Tiered mode (Niyama-style 3 QoS classes) ----
    parser.add_argument("--slo-mode",
                        choices=["uniform", "tiered", "mixed"],
                        default="uniform",
                        help="uniform: every request has same SLO. "
                             "tiered: 3 random classes (Niyama-style). "
                             "mixed: 2 classes by origin dataset "
                             "(short=sharegpt, long=arxivsumm), each "
                             "with its own SLO. Requires "
                             "--dataset mixed_short_long.")
    parser.add_argument("--ttft-slo-tight-ms", type=float, default=None)
    parser.add_argument("--ttft-slo-normal-ms", type=float, default=None)
    parser.add_argument("--ttft-slo-loose-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-tight-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-normal-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-loose-ms", type=float, default=None)
    # ---- Mixed mode (2 classes by origin dataset) ----
    parser.add_argument("--short-ttft-slo-ms", type=float, default=None,
                        help="(mixed mode) TTFT SLO for short-class "
                             "requests (origin=sharegpt).")
    parser.add_argument("--short-tpot-slo-ms", type=float, default=None)
    parser.add_argument("--long-ttft-slo-ms", type=float, default=None,
                        help="(mixed mode) TTFT SLO for long-class "
                             "requests (origin=arxivsumm).")
    parser.add_argument("--long-tpot-slo-ms", type=float, default=None)
    parser.add_argument("--force-max-output-tokens", type=int, default=None,
                        help="If set, override every request's max_tokens "
                             "to this value (ignore the dataset's "
                             "expected_output_tokens). Pair with "
                             "--ignore-eos to actually generate that many "
                             "tokens — otherwise the sampler may emit EOS "
                             "early. Used for long-output stress runs.")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Set ignore_eos=True on every request so the "
                             "model decodes the full max_tokens regardless "
                             "of EOS. Required to realize a long-output "
                             "workload built with --force-max-output-tokens.")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Engine --max-model-len. Auto-set by dataset.")
    parser.add_argument("--out-tag", type=str, default="",
                        help="Optional suffix appended to output filenames "
                             "(before _metrics.json / _engine0.log etc). "
                             "Used by E_M2 sweep to distinguish multiple "
                             "SLO tightness runs that share (baseline, "
                             "qps, seed). Empty string = no suffix.")
    args = parser.parse_args()

    # Validate SLO arguments based on mode.
    _TIER_CLASSES = ("tight", "normal", "loose")
    if args.slo_mode == "uniform":
        if args.ttft_slo_ms is None or args.tpot_slo_ms is None:
            parser.error("--slo-mode uniform requires --ttft-slo-ms and "
                         "--tpot-slo-ms")
        # Treat uniform as a degenerate 3-tier: same SLO across tiers.
        tier_ttft = {c: args.ttft_slo_ms for c in _TIER_CLASSES}
        tier_tpot = {c: args.tpot_slo_ms for c in _TIER_CLASSES}
    elif args.slo_mode == "tiered":
        for k in ("ttft_slo_tight_ms", "ttft_slo_normal_ms",
                  "ttft_slo_loose_ms", "tpot_slo_tight_ms",
                  "tpot_slo_normal_ms", "tpot_slo_loose_ms"):
            if getattr(args, k) is None:
                parser.error(f"--slo-mode tiered requires --{k.replace('_', '-')}")
        tier_ttft = {
            "tight": args.ttft_slo_tight_ms,
            "normal": args.ttft_slo_normal_ms,
            "loose": args.ttft_slo_loose_ms,
        }
        tier_tpot = {
            "tight": args.tpot_slo_tight_ms,
            "normal": args.tpot_slo_normal_ms,
            "loose": args.tpot_slo_loose_ms,
        }
    else:  # mixed
        if args.dataset not in ("mixed_short_long", "burstgpt_mixed"):
            parser.error("--slo-mode mixed requires --dataset "
                         "mixed_short_long or burstgpt_mixed")
        for k in ("short_ttft_slo_ms", "short_tpot_slo_ms",
                  "long_ttft_slo_ms", "long_tpot_slo_ms"):
            if getattr(args, k) is None:
                parser.error(f"--slo-mode mixed requires --{k.replace('_', '-')}")
        tier_ttft = {
            "short": args.short_ttft_slo_ms,
            "long": args.long_ttft_slo_ms,
        }
        tier_tpot = {
            "short": args.short_tpot_slo_ms,
            "long": args.long_tpot_slo_ms,
        }

    # Class assignment for tiered mode: balanced random shuffle. For
    # mixed mode: class comes from the schedule tuple itself (origin
    # dataset tag). For uniform mode: random tiered assignment is
    # used internally but every class maps to the same SLO so it
    # doesn't matter.
    import random as _stdrand
    # Per-request class list, populated after schedule is built when
    # dataset=mixed_short_long. Unused in uniform/tiered modes.
    _per_request_class: list[str] = []
    if args.slo_mode == "mixed":
        def class_of(idx: int) -> str:
            if 0 <= idx < len(_per_request_class):
                return _per_request_class[idx]
            return "long"  # safe fallback
    else:
        _n = args.num_requests
        _per_class = _n // 3
        _remainder = _n - 3 * _per_class
        _labels = (["tight"] * _per_class
                   + ["normal"] * _per_class
                   + ["loose"] * _per_class
                   + list(_TIER_CLASSES[:_remainder]))
        _class_rng = _stdrand.Random(args.seed * 7919 + 31)
        _class_rng.shuffle(_labels)
        def class_of(idx: int) -> str:
            if idx < 0 or idx >= len(_labels):
                return _TIER_CLASSES[idx % 3]
            return _labels[idx]

    # Default max_model_len by dataset. The +N margin must cover the
    # longest output we plan to generate. For long-output sweeps
    # (--force-max-output-tokens), bump the margin so input+output fits.
    if args.force_max_output_tokens is not None:
        output_margin = int(args.force_max_output_tokens) + 256
    else:
        output_margin = 512  # default short-output (back-compat with prior runs)
    if args.max_model_len is None:
        if args.dataset == "ruler_64k":
            max_model_len = 65536 + (256 if args.force_max_output_tokens is None
                                     else output_margin)
        elif args.dataset in ("ruler_16k", "ruler_mixed"):
            # ruler_mixed has prompts up to 16K; engine needs the 16K
            # cap so the longest record fits.
            max_model_len = 16384 + output_margin
        elif args.dataset == "ruler_8k":
            max_model_len = 8192 + output_margin
        elif args.dataset == "ruler_4k":
            max_model_len = 4096 + output_margin
        elif args.dataset == "ruler_2k":
            max_model_len = 2048 + output_margin
        elif args.dataset == "ruler_1k":
            max_model_len = 1024 + output_margin
        elif args.dataset == "arxivsumm":
            # arxivsumm prompts capped at 30K; need 32K headroom for output.
            max_model_len = 32768
        elif args.dataset == "mixed_short_long":
            # Mixed contains arxivsumm prompts up to 30K + sharegpt up
            # to ~4K. Set to 32K so engine fits both.
            max_model_len = 32768
        elif args.dataset == "burstgpt_mixed":
            # burstgpt_mixed embeds arrival pattern + can include any
            # length from short sharegpt to long arxivsumm. Use 32K cap.
            max_model_len = 32768
        else:  # sharegpt
            max_model_len = 4096 + output_margin
    else:
        max_model_len = args.max_model_len

    # Tag pieces: dataset, optionally an output-length tag for
    # long-output sweeps (e.g. "ruler_16k_out1024"). Keeping the
    # default (no override) producing the un-suffixed name avoids
    # invalidating existing results on disk.
    dataset_tag = args.dataset
    if args.force_max_output_tokens is not None:
        dataset_tag = f"{args.dataset}_out{int(args.force_max_output_tokens)}"
    tag = (f"e_m1_{args.baseline}_{dataset_tag}_"
           f"qps{args.arrival_rate_qps}_n{args.num_requests}_seed{args.seed}")
    if args.out_tag:
        tag = f"{tag}_{args.out_tag}"
    engine_0_log = RESULTS_DIR / f"{tag}_engine0.log"
    engine_1_log = RESULTS_DIR / f"{tag}_engine1.log"
    router_log = RESULTS_DIR / f"{tag}_router.log"
    metrics_out = RESULTS_DIR / f"{tag}_metrics.json"

    if args.dataset == "mixed_short_long":
        schedule_full = build_mixed_schedule(
            short_dataset="sharegpt",
            long_dataset="arxivsumm",
            short_ratio=args.mixed_short_ratio,
            num_requests=args.num_requests,
            arrival_rate_qps=args.arrival_rate_qps,
            seed=args.seed,
        )
        # Strip class tag to keep downstream code 3-tuple-compatible;
        # remember per-request class for class_of().
        schedule: list[tuple[float, str, int]] = [
            (s[0], s[1], s[2]) for s in schedule_full
        ]
        _per_request_class[:] = [s[3] for s in schedule_full]
    elif args.dataset == "burstgpt_mixed":
        # Trace-driven: arrival pattern comes from BurstGPT jsonl, not
        # Poisson. --arrival-rate-qps is ignored (the trace fixes it).
        schedule_full = build_trace_schedule(
            dataset_name="burstgpt_mixed",
            num_requests=args.num_requests,
            seed=args.seed,
        )
        schedule = [(s[0], s[1], s[2]) for s in schedule_full]
        _per_request_class[:] = [s[3] for s in schedule_full]
    else:
        schedule = build_schedule(
            dataset_name=args.dataset,
            num_requests=args.num_requests,
            arrival_rate_qps=args.arrival_rate_qps,
            seed=args.seed,
            force_max_tokens=args.force_max_output_tokens,
        )
    sched_summary = summarize_schedule(schedule)
    print(f"[E_M1] schedule: {sched_summary}")
    print(f"[E_M1] expected fire window: ~{sched_summary['window_s']:.0f}s")
    print(f"[E_M1] SLO mode: {args.slo_mode}")
    _CLASS_LIST = (("short", "long") if args.slo_mode == "mixed"
                   else _TIER_CLASSES)
    for c in _CLASS_LIST:
        print(f"[E_M1]   {c}: TTFT≤{tier_ttft[c]:.0f}ms, "
              f"TPOT≤{tier_tpot[c]:.1f}ms")
    if args.dataset in ("mixed_short_long", "burstgpt_mixed"):
        n_short = sum(1 for c in _per_request_class if c == "short")
        n_long = sum(1 for c in _per_request_class if c == "long")
        print(f"[E_M1]   composition: short={n_short} long={n_long}")

    cleanup_shm()
    e0 = start_engine(0, ENGINE_0_PORT, "0", engine_0_log,
                      args.baseline, max_model_len)
    e1 = start_engine(1, ENGINE_1_PORT, "1", engine_1_log,
                      args.baseline, max_model_len)
    router = None
    try:
        ok = (wait_url_ready(
            f"http://127.0.0.1:{ENGINE_0_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ) and wait_url_ready(
            f"http://127.0.0.1:{ENGINE_1_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ))
        if not ok:
            print("[E_M1] FAIL: an engine never became healthy")
            return 1
        print("[E_M1] both engines ready")

        if args.baseline == "vllm_fcfs":
            target_urls = [
                f"http://127.0.0.1:{ENGINE_0_PORT}",
                f"http://127.0.0.1:{ENGINE_1_PORT}",
            ]
            print("[E_M1] vllm_fcfs: bypassing router")
        else:
            router = start_router(router_log)
            if not wait_url_ready(
                f"http://127.0.0.1:{ROUTER_PORT}/health",
                ROUTER_READY_TIMEOUT_S,
            ):
                print("[E_M1] FAIL: router not healthy")
                return 1
            if not wait_both_engines_alive_in_router():
                print("[E_M1] FAIL: router didn't see both engines alive")
                return 1
            target_urls = [f"http://127.0.0.1:{ROUTER_PORT}"]
            print(f"[E_M1] {args.baseline}: using router")

        # SLO is passed in extra_args only when the picker is active
        # (so it has SLO info to compare). 'ours_no_picker' has the
        # picker disabled even though it has FT_CAPACITY_PREEMPT_RELOAD=1,
        # so SLO in extra_args wouldn't fire anything.
        pass_slo_to_engine = (args.baseline == "ours")

        clients: list[Client] = []
        t0 = time.time()
        for i, (offset_s, prompt, max_tokens) in enumerate(schedule):
            target_t = t0 + offset_s
            now = time.time()
            if target_t > now:
                time.sleep(target_t - now)
            engine_url = target_urls[i % len(target_urls)]
            cls = class_of(i)
            if pass_slo_to_engine:
                client_ttft_slo = tier_ttft[cls]
                client_tpot_slo = tier_tpot[cls]
            else:
                client_ttft_slo = None
                client_tpot_slo = None
            c = Client(i, engine_url, prompt, max_tokens,
                       client_ttft_slo, client_tpot_slo,
                       ignore_eos=args.ignore_eos)
            c.start()
            clients.append(c)
        print(f"[E_M1] fired {len(clients)} requests; waiting completion")

        for c in clients:
            c.join(timeout=REQUEST_TIMEOUT_S)

        n_ok = sum(1 for c in clients
                   if c.error is None and c.status_code == 200)
        n_err = sum(1 for c in clients if c.error is not None)
        print(f"[E_M1] outcomes: {n_ok}/{len(clients)} 200 OK, "
              f"{n_err} errored")

        events = (parse_request_done_log(engine_0_log)
                  + parse_request_done_log(engine_1_log))

        # Sort events globally by arrival_ts. Since clients fire sequentially
        # in idx order with Poisson delays, the k-th event by arrival_ts
        # corresponds to client idx k → class_of(k). This lets us apply the
        # right per-class SLO threshold post-hoc.
        events_sorted = sorted(events, key=lambda e: e["arrival_ts"])
        for k, e in enumerate(events_sorted):
            cls = class_of(k)
            e["class"] = cls
            e["slo_ttft_ms"] = tier_ttft[cls]
            e["slo_tpot_ms"] = tier_tpot[cls]
            e["slo_met"] = (e["ttft_ms"] <= tier_ttft[cls]
                            and e["tpot_ms"] <= tier_tpot[cls])

        slo_met_flags = [e["slo_met"] for e in events_sorted]
        slo_met_pct = (100.0 * sum(slo_met_flags) / len(slo_met_flags)
                       if slo_met_flags else 0.0)

        # Per-class breakdown. Class set depends on slo_mode.
        _CLASS_LIST_AGG = (("short", "long") if args.slo_mode == "mixed"
                           else _TIER_CLASSES)
        per_class: dict[str, dict] = {}
        for cls in _CLASS_LIST_AGG:
            cls_events = [e for e in events_sorted if e["class"] == cls]
            if not cls_events:
                per_class[cls] = {"n": 0, "slo_met_pct": None,
                                  "ttft_p95": None, "tpot_p95": None}
                continue
            cls_met = sum(1 for e in cls_events if e["slo_met"])
            per_class[cls] = {
                "n": len(cls_events),
                "slo_met_count": cls_met,
                "slo_met_pct": 100.0 * cls_met / len(cls_events),
                "ttft_p50": percentile([e["ttft_ms"] for e in cls_events], 50),
                "ttft_p95": percentile([e["ttft_ms"] for e in cls_events], 95),
                "tpot_p95": percentile(
                    [e["tpot_ms"] for e in cls_events if e["tpot_ms"] > 0],
                    95,
                ),
            }
        # Re-bind events to keep the rest of the code working.
        events = events_sorted

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

        print(f"[E_M1] SLO_met overall: {slo_met_pct:.1f}% "
              f"({sum(slo_met_flags)}/{len(slo_met_flags)})")
        if args.slo_mode in ("tiered", "mixed"):
            for cls in _CLASS_LIST_AGG:
                p = per_class[cls]
                pct = (f"{p['slo_met_pct']:.1f}%"
                       if p["slo_met_pct"] is not None else "n/a")
                print(f"[E_M1]   {cls}: SLO_met={pct} (n={p['n']})")
        print(f"[E_M1] TTFT_ms P50={fmt(percentile(ttft_vals, 50))} "
              f"P95={fmt(percentile(ttft_vals, 95))}")
        print(f"[E_M1] TPOT_ms P50={fmt(percentile(tpot_vals, 50))} "
              f"P95={fmt(percentile(tpot_vals, 95))}")
        print(f"[E_M1] throughput: {fmt(throughput_tok_per_s)} tok/s")

        metrics_out.write_text(json.dumps({
            "baseline": args.baseline,
            "dataset": args.dataset,
            "num_requests": args.num_requests,
            "arrival_rate_qps": args.arrival_rate_qps,
            "seed": args.seed,
            "slo_mode": args.slo_mode,
            "slo": {
                "tier_ttft_ms": tier_ttft,
                "tier_tpot_ms": tier_tpot,
            },
            "schedule_summary": sched_summary,
            "n_events": len(events),
            "slo_met_pct": slo_met_pct,
            "slo_met_count": sum(slo_met_flags),
            "per_class": per_class,
            "ttft_ms": {
                "p50": percentile(ttft_vals, 50),
                "p95": percentile(ttft_vals, 95),
                "mean": statistics.mean(ttft_vals) if ttft_vals else None,
            },
            "tpot_ms": {
                "p50": percentile(tpot_vals, 50),
                "p95": percentile(tpot_vals, 95),
                "mean": statistics.mean(tpot_vals) if tpot_vals else None,
            },
            "e2e_ms": {
                "p50": percentile(e2e_vals, 50),
                "p95": percentile(e2e_vals, 95),
            },
            "window_s": window_s,
            "total_output_tokens": total_out_tokens,
            "throughput_tok_per_s": throughput_tok_per_s,
            "client_outcomes": {"200_ok": n_ok, "errored": n_err},
            "per_request": events,
        }, indent=2))
        print(f"[E_M1] metrics → {metrics_out}")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
