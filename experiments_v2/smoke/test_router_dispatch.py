#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test 3: router dispatches across two live engines (no failure injection).

Starts engine 0 on GPU 0 and engine 1 on GPU 1, both with FT_ROUTER_SHM_BUS=1,
then starts the router pointing at both. Sends N short completion requests
through the router and verifies:
  1. Router /health sees both engines alive.
  2. All N requests return 200.
  3. Both engines received at least one request each (router /metrics
     dispatch_count == N and engines' shm req_map dirs together contain
     ≥ N files keyed by router_req_id).
  4. reroute_count == 0 (no engine ever died → no reroute).

This isolates the dispatch path from the reroute / failure path (Test 4).

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
ENGINE_0_PORT = 8401
ENGINE_1_PORT = 8402
ROUTER_PORT = 8400
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_0_LOG = SCRIPT_DIR / "test3_engine0.log"
ENGINE_1_LOG = SCRIPT_DIR / "test3_engine1.log"
ROUTER_LOG = SCRIPT_DIR / "test3_router.log"
ENGINE_READY_TIMEOUT_S = 240
ROUTER_READY_TIMEOUT_S = 30
NUM_REQUESTS = 10


def cleanup_shm() -> None:
    shutil.rmtree(STATUS_DIR, ignore_errors=True)
    shutil.rmtree(REQ_MAP_DIR, ignore_errors=True)


def start_engine(engine_id: int, port: int, gpu: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_ROUTER_SHM_BUS"] = "1"
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.5",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[test3] launching engine {engine_id} on GPU {gpu} → {log_path.name}")
    return subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )


def start_router() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "experiments_v2.router.router",
        "--port", str(ROUTER_PORT),
        "--engines",
        f"0=http://127.0.0.1:{ENGINE_0_PORT}",
        f"1=http://127.0.0.1:{ENGINE_1_PORT}",
        "--log-level", "info",
    ]
    log_f = open(ROUTER_LOG, "w")
    print(f"[test3] launching router → {ROUTER_LOG.name}")
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


def check_both_engines_alive() -> tuple[bool, str]:
    deadline = time.time() + 10
    while time.time() < deadline:
        h = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/health")
        if h is not None:
            engines = h.get("engines", {})
            e0 = engines.get("0")
            e1 = engines.get("1")
            if e0 and e1 and e0.get("alive") and e1.get("alive"):
                return True, "both engines alive"
        time.sleep(1)
    return False, f"router never saw both alive: {h}"


def send_one(idx: int) -> dict:
    router_req_id = uuid.uuid4().hex
    payload = json.dumps({
        "model": MODEL,
        "prompt": f"Request {idx}. Tell me a fact.",
        "max_tokens": 8,
        "temperature": 0.0,
        "vllm_xargs": {"router_req_id": router_req_id},
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
            ok = "choices" in data
    except Exception as e:
        ok = False
        data = {"error": str(e)}
    return {"idx": idx, "ok": ok, "router_req_id": router_req_id, "data": data}


def send_concurrent(n: int) -> list[dict]:
    print(f"[test3] firing {n} concurrent requests")
    results = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(send_one, i) for i in range(n)]
        for f in as_completed(futs):
            results.append(f.result())
    return results


def check_dispatch_balance(results: list[dict]) -> tuple[bool, str]:
    """Each successful request wrote a req_map file containing engine_id.
    Verify both engines received at least one request.
    """
    engine_counts: dict[int, int] = {0: 0, 1: 0}
    for r in results:
        if not r["ok"]:
            continue
        fp = REQ_MAP_DIR / r["router_req_id"]
        if not fp.exists():
            continue
        try:
            data = json.loads(fp.read_text())
            eid = data.get("engine_id")
            if eid in engine_counts:
                engine_counts[eid] += 1
        except (OSError, json.JSONDecodeError):
            continue
    if engine_counts[0] == 0 or engine_counts[1] == 0:
        return False, (
            f"dispatch not balanced: engine0={engine_counts[0]}, "
            f"engine1={engine_counts[1]}"
        )
    return True, f"engine0={engine_counts[0]}, engine1={engine_counts[1]}"


def check_metrics(expected_dispatch: int) -> tuple[bool, str]:
    m = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/metrics")
    if m is None:
        return False, "router /metrics unreachable"
    if m.get("dispatch_count") != expected_dispatch:
        return False, (
            f"dispatch_count={m.get('dispatch_count')} != "
            f"expected {expected_dispatch} (full metrics: {m})"
        )
    if m.get("reroute_count") != 0:
        return False, f"reroute_count != 0 (unexpected): {m}"
    if m.get("dead_event_count") != 0:
        return False, f"dead_event_count != 0 (unexpected): {m}"
    return True, (
        f"metrics ok (dispatch={m['dispatch_count']}, "
        f"reroute={m['reroute_count']}, dead={m['dead_event_count']})"
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


def main() -> int:
    cleanup_shm()
    e0 = start_engine(0, ENGINE_0_PORT, "0", ENGINE_0_LOG)
    e1 = start_engine(1, ENGINE_1_PORT, "1", ENGINE_1_LOG)
    router = None
    try:
        if not wait_url_ready(
            f"http://127.0.0.1:{ENGINE_0_PORT}/health", ENGINE_READY_TIMEOUT_S
        ):
            print("[test3] FAIL: engine 0 never became healthy")
            return 1
        if not wait_url_ready(
            f"http://127.0.0.1:{ENGINE_1_PORT}/health", ENGINE_READY_TIMEOUT_S
        ):
            print("[test3] FAIL: engine 1 never became healthy")
            return 1
        print("[test3] both engines ready")

        router = start_router()
        if not wait_url_ready(
            f"http://127.0.0.1:{ROUTER_PORT}/health", ROUTER_READY_TIMEOUT_S
        ):
            print("[test3] FAIL: router never became healthy")
            return 1
        print("[test3] router ready")

        ok, msg = check_both_engines_alive()
        print(f"[test3] both engines alive: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        results = send_concurrent(NUM_REQUESTS)
        n_ok = sum(1 for r in results if r["ok"])
        print(f"[test3] {n_ok}/{NUM_REQUESTS} requests returned 200")
        if n_ok != NUM_REQUESTS:
            failures = [r for r in results if not r["ok"]][:3]
            print(f"[test3] FAIL: failures (first 3): {failures}")
            return 1

        ok, msg = check_dispatch_balance(results)
        print(f"[test3] dispatch balance: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        ok, msg = check_metrics(NUM_REQUESTS)
        print(f"[test3] router metrics: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        print("[test3] PASS")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
