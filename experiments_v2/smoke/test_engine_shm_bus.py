#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test 1: verify FT_ROUTER_SHM_BUS=1 produces engine_status + req_map.

Starts a single vLLM OpenAI server with FT_ROUTER_SHM_BUS=1, then:
  1. Waits for /health = 200.
  2. Polls /dev/shm/vllm_ft_engine_status/engine_0.json for ~5s and
     verifies its `ts` field is being refreshed by the engine.
  3. Sends one OpenAI /v1/completions request with a router-style
     X-Request-Id header and verifies the engine wrote a matching
     /dev/shm/vllm_ft_req_map/<user_req_id> file with a valid
     internal_req_id field.
  4. SIGTERMs the engine and verifies the status `ts` stops updating
     within 3s (i.e. router would be able to detect engine death).

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
PORT = 8401
ENGINE_ID = 0
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
STATUS_PATH = STATUS_DIR / f"engine_{ENGINE_ID}.json"
SERVER_LOG = Path(__file__).parent / "test1_server.log"
START_TIMEOUT_S = 240
TS_REFRESH_WAIT_S = 5
TS_AFTER_KILL_WAIT_S = 3


def cleanup_shm() -> None:
    shutil.rmtree(STATUS_DIR, ignore_errors=True)
    shutil.rmtree(REQ_MAP_DIR, ignore_errors=True)


def start_engine() -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_ROUTER_SHM_BUS"] = "1"
    env["VLLM_FT_ENGINE_ID"] = str(ENGINE_ID)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("FT_STATUS_WRITE_EVERY_N_STEPS", "5")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.5",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(SERVER_LOG, "w")
    print(f"[test1] launching engine, log → {SERVER_LOG.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def wait_ready() -> bool:
    deadline = time.time() + START_TIMEOUT_S
    url = f"http://127.0.0.1:{PORT}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def read_status() -> dict | None:
    try:
        return json.loads(STATUS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def check_status_refreshing() -> tuple[bool, str]:
    """Wait TS_REFRESH_WAIT_S seconds and verify status `ts` changed."""
    s1 = read_status()
    if s1 is None:
        return False, f"status file {STATUS_PATH} missing or unreadable"
    expected = {"engine_id", "alive", "running", "waiting", "kv_usage", "ts"}
    missing = expected - s1.keys()
    if missing:
        return False, f"status missing fields: {missing}"
    if s1["engine_id"] != ENGINE_ID:
        return False, f"engine_id mismatch: {s1['engine_id']} != {ENGINE_ID}"
    ts1 = s1["ts"]
    time.sleep(TS_REFRESH_WAIT_S)
    s2 = read_status()
    if s2 is None:
        return False, "status file vanished mid-test"
    ts2 = s2["ts"]
    if ts2 <= ts1:
        return False, (
            f"status ts not advancing: ts1={ts1:.3f}, ts2={ts2:.3f} "
            f"after {TS_REFRESH_WAIT_S}s wait"
        )
    return True, f"status refreshed (ts1={ts1:.3f} → ts2={ts2:.3f})"


def send_completion(user_req_id: str) -> tuple[bool, str]:
    payload = json.dumps({
        "model": MODEL,
        "prompt": "Hello, world.",
        "max_tokens": 8,
        "temperature": 0.0,
        # Router-style: pass router_req_id via vllm_xargs so engine writes
        # /dev/shm/vllm_ft_req_map/<router_req_id>. We can't use the
        # X-Request-Id header because vllm's OpenAI handler mangles it
        # with "cmpl-...-0" before storing it as external_req_id.
        "vllm_xargs": {"router_req_id": user_req_id},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except Exception as e:
        return False, f"completion request failed: {e}"
    if "choices" not in data:
        return False, f"completion response missing choices: {data!r}"
    return True, "completion ok"


def check_req_map(user_req_id: str) -> tuple[bool, str]:
    fp = REQ_MAP_DIR / user_req_id
    if not fp.exists():
        return False, f"req_map file {fp} not written"
    try:
        data = json.loads(fp.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"req_map unparseable: {e}"
    expected = {"internal_req_id", "engine_id", "start_ts"}
    missing = expected - data.keys()
    if missing:
        return False, f"req_map missing fields: {missing}"
    if not data["internal_req_id"]:
        return False, f"req_map internal_req_id empty: {data}"
    if data["engine_id"] != ENGINE_ID:
        return False, f"req_map engine_id mismatch: {data}"
    return True, (
        f"req_map ok (internal_req_id={data['internal_req_id']!r}, "
        f"engine_id={data['engine_id']})"
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


def check_status_freezes_after_kill() -> tuple[bool, str]:
    s1 = read_status()
    if s1 is None:
        return True, "status file vanished (also acceptable: dead engine)"
    ts1 = s1["ts"]
    time.sleep(TS_AFTER_KILL_WAIT_S)
    s2 = read_status()
    if s2 is None:
        return True, "status file vanished after kill"
    ts2 = s2["ts"]
    if ts2 > ts1:
        return False, (
            f"status ts still advancing after engine kill: "
            f"ts1={ts1:.3f} → ts2={ts2:.3f}"
        )
    return True, f"status frozen after kill (ts={ts2:.3f})"


def main() -> int:
    cleanup_shm()
    proc = start_engine()
    try:
        if not wait_ready():
            print("[test1] FAIL: engine never became healthy")
            return 1
        print("[test1] engine ready")

        ok, msg = check_status_refreshing()
        print(f"[test1] status refresh: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        user_req_id = uuid.uuid4().hex
        ok, msg = send_completion(user_req_id)
        print(f"[test1] completion request: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        ok, msg = check_req_map(user_req_id)
        print(f"[test1] req_map written: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        print("[test1] killing engine, verifying status freezes…")
        shutdown(proc)
        proc = None  # don't try to kill again in finally
        ok, msg = check_status_freezes_after_kill()
        print(f"[test1] status freeze: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        print("[test1] PASS")
        return 0
    finally:
        if proc is not None:
            shutdown(proc)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
