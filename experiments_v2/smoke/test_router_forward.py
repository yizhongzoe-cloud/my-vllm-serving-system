#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test 2: router forwards OpenAI request to one engine + extra_args wire-up.

Starts one vLLM engine on GPU 0 with FT_ROUTER_SHM_BUS=1, then starts the
router pointing at that engine, sends one completion request through the
router with `vllm_xargs = {"ttft_slo_ms": 1234.0, "router_req_id": "..."}`,
and verifies:
  1. Router /health returns ok and engine 0 is marked alive.
  2. The completion request returns 200 with valid choices.
  3. Engine server log contains "ttft_slo_ms=1234.0" — proves
     EngineCore.add_request lifted the SLO field from sampling_params.extra_args
     into Request.ttft_slo_ms.
  4. /dev/shm/vllm_ft_req_map/<router_req_id> exists with internal_req_id.
  5. Router /metrics shows dispatch_count >= 1, reroute_count == 0.

Exit 0 = PASS, 1 = FAIL.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
ENGINE_PORT = 8401
ROUTER_PORT = 8400
ENGINE_ID = 0
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_LOG = SCRIPT_DIR / "test2_engine.log"
ROUTER_LOG = SCRIPT_DIR / "test2_router.log"
ENGINE_READY_TIMEOUT_S = 240
ROUTER_READY_TIMEOUT_S = 30
TEST_TTFT_SLO_MS = 1234.0


def cleanup_shm() -> None:
    shutil.rmtree(STATUS_DIR, ignore_errors=True)
    shutil.rmtree(REQ_MAP_DIR, ignore_errors=True)


def start_engine() -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_ROUTER_SHM_BUS"] = "1"
    env["VLLM_FT_ENGINE_ID"] = str(ENGINE_ID)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Need DEBUG to grep the "control fields lifted" line from
    # EngineCore.add_request; in paper experiments default INFO is fine.
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(ENGINE_PORT),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.5",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(ENGINE_LOG, "w")
    print(f"[test2] launching engine, log → {ENGINE_LOG.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def start_router() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "experiments_v2.router.router",
        "--port", str(ROUTER_PORT),
        "--engines", f"0=http://127.0.0.1:{ENGINE_PORT}",
        "--log-level", "info",
    ]
    log_f = open(ROUTER_LOG, "w")
    print(f"[test2] launching router, log → {ROUTER_LOG.name}")
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


def check_router_engine_alive() -> tuple[bool, str]:
    """Router needs ~1-2 cycles of status polling before marking engine alive."""
    deadline = time.time() + 10
    while time.time() < deadline:
        health = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/health")
        if health is None:
            time.sleep(1)
            continue
        engines = health.get("engines", {})
        # JSON keys come back as strings.
        e0 = engines.get("0") or engines.get(str(ENGINE_ID))
        if e0 and e0.get("alive"):
            return True, f"engine 0 alive (running={e0.get('running')})"
        time.sleep(1)
    return False, f"router never saw engine 0 alive: {health}"


def send_via_router(router_req_id: str) -> tuple[bool, str]:
    payload = json.dumps({
        "model": MODEL,
        "prompt": "Hello, world.",
        "max_tokens": 8,
        "temperature": 0.0,
        "vllm_xargs": {
            "router_req_id": router_req_id,
            "ttft_slo_ms": TEST_TTFT_SLO_MS,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{ROUTER_PORT}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        return False, f"completion via router failed: {e}"
    if "choices" not in data:
        return False, f"completion response missing choices: {data!r}"
    return True, "completion ok"


def check_engine_log_lifted_ttft_slo() -> tuple[bool, str]:
    """Grep engine log for the FT_ROUTER_SHM_BUS lift message and confirm
    ttft_slo_ms was set to TEST_TTFT_SLO_MS."""
    try:
        log_text = ENGINE_LOG.read_text(errors="replace")
    except OSError as e:
        return False, f"engine log unreadable: {e}"
    needle = f"ttft_slo_ms={TEST_TTFT_SLO_MS}"
    if needle not in log_text:
        # Surface the most relevant lines for debug.
        lifts = [
            ln for ln in log_text.splitlines()
            if "control fields lifted" in ln
        ]
        return False, (
            f"engine log missing ttft_slo_ms={TEST_TTFT_SLO_MS}. "
            f"Found lift lines: {lifts[-3:] if lifts else 'none'}"
        )
    return True, f"engine log contains {needle!r}"


def check_req_map(router_req_id: str) -> tuple[bool, str]:
    fp = REQ_MAP_DIR / router_req_id
    if not fp.exists():
        return False, f"req_map file {fp} missing"
    data = json.loads(fp.read_text())
    if not data.get("internal_req_id"):
        return False, f"req_map missing internal_req_id: {data}"
    return True, f"req_map ok (internal_req_id={data['internal_req_id']!r})"


def check_router_metrics() -> tuple[bool, str]:
    m = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/metrics")
    if m is None:
        return False, "router /metrics unreachable"
    if m.get("dispatch_count", 0) < 1:
        return False, f"dispatch_count < 1: {m}"
    if m.get("reroute_count", 0) != 0:
        return False, f"reroute_count != 0 (unexpected): {m}"
    return True, f"metrics ok (dispatch={m['dispatch_count']}, reroute={m['reroute_count']})"


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
    cleanup_shm()
    engine = start_engine()
    router = None
    try:
        engine_health_url = f"http://127.0.0.1:{ENGINE_PORT}/health"
        if not wait_url_ready(engine_health_url, ENGINE_READY_TIMEOUT_S):
            print("[test2] FAIL: engine never became healthy")
            return 1
        print("[test2] engine ready")

        router = start_router()
        router_health_url = f"http://127.0.0.1:{ROUTER_PORT}/health"
        if not wait_url_ready(router_health_url, ROUTER_READY_TIMEOUT_S):
            print("[test2] FAIL: router never became healthy")
            return 1
        print("[test2] router ready")

        ok, msg = check_router_engine_alive()
        print(f"[test2] router sees engine alive: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        router_req_id = uuid.uuid4().hex
        ok, msg = send_via_router(router_req_id)
        print(f"[test2] completion via router: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        ok, msg = check_engine_log_lifted_ttft_slo()
        print(f"[test2] ttft_slo_ms lifted in engine: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        ok, msg = check_req_map(router_req_id)
        print(f"[test2] req_map written: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        ok, msg = check_router_metrics()
        print(f"[test2] router metrics: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        print("[test2] PASS")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(engine)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
