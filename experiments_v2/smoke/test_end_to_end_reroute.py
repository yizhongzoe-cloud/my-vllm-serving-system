#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test 4: end-to-end cross-engine reroute (Path A).

  1. Start engine 0 (GPU 0) and engine 1 (GPU 1), both with:
       FT_ROUTER_SHM_BUS=1, FT_CAPACITY_PREEMPT_RELOAD=1,
       FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1, FT_DELTA_CHECKPOINT=1.
  2. Start router pointing at both.
  3. Send ONE long-running completion request through the router. It
     lands on engine 0 (only alive engine when picked, or lowest load).
  4. Poll /dev/shm/vllm_ft_checkpoints/<dir> in the background until at
     least one chunk file shows up — proves engine 0 has published KV.
  5. SIGKILL engine 0.
  6. Wait until router /metrics shows dead_event_count >= 1 (router saw
     status ts go stale within ~2s) and reroute_count >= 1.
  7. Join the client thread. The request should return a 200 response
     with a non-empty completion (engine 1 restored from shm + finished
     decode).
  8. Grep engine 1 log for "Cross-engine reroute" + "FT overlap V3:".

Exit 0 = PASS, 1 = FAIL.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODEL = os.path.expanduser("~/model/Qwen2.5-7B-Instruct")
ENGINE_0_PORT = 8401
ENGINE_1_PORT = 8402
ROUTER_PORT = 8400
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")
SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_0_LOG = SCRIPT_DIR / "test4_engine0.log"
ENGINE_1_LOG = SCRIPT_DIR / "test4_engine1.log"
ROUTER_LOG = SCRIPT_DIR / "test4_router.log"
ENGINE_READY_TIMEOUT_S = 240
ROUTER_READY_TIMEOUT_S = 30
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 200
PUBLISH_WAIT_TIMEOUT_S = 90
REROUTE_DETECT_TIMEOUT_S = 15
REQUEST_TIMEOUT_S = 600


def cleanup_shm() -> None:
    shutil.rmtree(STATUS_DIR, ignore_errors=True)
    shutil.rmtree(REQ_MAP_DIR, ignore_errors=True)
    shutil.rmtree(CKPT_DIR, ignore_errors=True)


def start_engine(engine_id: int, port: int, gpu: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_ROUTER_SHM_BUS"] = "1"
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # V3 reload + capacity preempt machinery: required so engine A publishes
    # KV (so engine B can restore) and so engine B's reload state machine
    # runs when it receives a rerouted request.
    env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
    env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
    env["FT_DELTA_CHECKPOINT"] = "1"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", str(PROMPT_TOKENS + MAX_OUTPUT_TOKENS + 256),
        "--gpu-memory-utilization", "0.5",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(f"[test4] launching engine {engine_id} on GPU {gpu} → {log_path.name}")
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
    print(f"[test4] launching router → {ROUTER_LOG.name}")
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
            e0 = h.get("engines", {}).get("0")
            e1 = h.get("engines", {}).get("1")
            if e0 and e1 and e0.get("alive") and e1.get("alive"):
                return True
        time.sleep(1)
    return False


def build_long_prompt() -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The quick brown fox jumps over the lazy dog. " * 16
    ids = tok.encode(base)
    repeats = PROMPT_TOKENS // max(1, len(ids)) + 2
    text = base * repeats
    return tok.decode(tok.encode(text)[:PROMPT_TOKENS])


class ClientWorker(threading.Thread):
    """Sends one long completion request via the router; stores result.

    Doesn't supply router_req_id — that's the router's internal control id,
    not a client concern. Router generates one and writes it into
    sampling_params.extra_args.router_req_id before forwarding.
    """
    def __init__(self, prompt: str) -> None:
        super().__init__(daemon=True)
        self.prompt = prompt
        self.result: dict | None = None
        self.error: str | None = None
        self.status_code: int | None = None
        self.response_headers: dict | None = None

    def run(self) -> None:
        payload = json.dumps({
            "model": MODEL,
            "prompt": self.prompt,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{ROUTER_PORT}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
                self.status_code = r.status
                self.response_headers = dict(r.headers.items())
                self.result = json.loads(r.read())
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"


def wait_for_published_chunk(timeout_s: int) -> tuple[bool, str]:
    """Poll until at least one chunk file shows up under any req dir."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if CKPT_DIR.exists():
            chunks = list(CKPT_DIR.rglob("chunk_*.pt"))
            if chunks:
                return True, f"saw {len(chunks)} chunk file(s); first={chunks[0].name}"
        time.sleep(0.5)
    return False, f"no chunk files in {CKPT_DIR} within {timeout_s}s"


def wait_for_reroute(timeout_s: int) -> tuple[bool, dict]:
    """Poll router metrics until dead + reroute counters increment."""
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        m = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/metrics")
        if m is not None:
            last = m
            if m.get("dead_event_count", 0) >= 1 and m.get("reroute_count", 0) >= 1:
                return True, m
        time.sleep(0.5)
    return False, last


def check_surviving_engine_log_for_reroute(log_path: Path) -> tuple[bool, str]:
    try:
        txt = log_path.read_text(errors="replace")
    except OSError as e:
        return False, f"engine log {log_path.name} unreadable: {e}"
    has_diversion = "Cross-engine reroute" in txt
    has_reload_done = "FT overlap V3: " in txt and "done" in txt
    if not has_diversion:
        return False, f"{log_path.name} missing 'Cross-engine reroute' (reroute didn't divert into V3 state machine)"
    if not has_reload_done:
        return False, f"{log_path.name} missing 'FT overlap V3: ... done' (restore from shm didn't complete)"
    return True, f"{log_path.name} shows reroute diversion + V3 reload done"


def shutdown(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
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
        ) or not wait_url_ready(
            f"http://127.0.0.1:{ENGINE_1_PORT}/health", ENGINE_READY_TIMEOUT_S
        ):
            print("[test4] FAIL: an engine never became healthy")
            return 1
        print("[test4] both engines ready")

        router = start_router()
        if not wait_url_ready(
            f"http://127.0.0.1:{ROUTER_PORT}/health", ROUTER_READY_TIMEOUT_S
        ):
            print("[test4] FAIL: router never became healthy")
            return 1
        print("[test4] router ready")

        if not wait_both_engines_alive_in_router():
            print("[test4] FAIL: router didn't see both engines alive")
            return 1

        prompt = build_long_prompt()
        worker = ClientWorker(prompt)
        print(f"[test4] firing long request")
        worker.start()

        # Engine 0 must publish at least one checkpoint before we kill it,
        # otherwise reroute will have no shm data to restore from.
        ok, msg = wait_for_published_chunk(PUBLISH_WAIT_TIMEOUT_S)
        print(f"[test4] published chunk seen: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        # Figure out which engine the request landed on. The router owns
        # the router_req_id (it's a uuid we can't predict), so we list the
        # one req_map file currently in the dir to find the engine id.
        target_eid = None
        if REQ_MAP_DIR.exists():
            for fp in REQ_MAP_DIR.iterdir():
                try:
                    target_eid = json.loads(fp.read_text()).get("engine_id")
                    break
                except (OSError, json.JSONDecodeError):
                    continue
        if target_eid is None:
            print(f"[test4] FAIL: couldn't read any req_map file to learn target engine")
            return 1
        if target_eid not in (0, 1):
            print(f"[test4] FAIL: unexpected engine_id={target_eid}")
            return 1
        victim_proc = e0 if target_eid == 0 else e1
        surviving_eid = 1 - target_eid
        print(f"[test4] request landed on engine {target_eid}; killing it")

        # SIGKILL to simulate sudden death (no graceful shutdown).
        shutdown(victim_proc, sig=signal.SIGKILL)

        ok, metrics = wait_for_reroute(REROUTE_DETECT_TIMEOUT_S)
        print(f"[test4] router observed reroute: {'OK' if ok else 'FAIL'} — {metrics}")
        if not ok:
            return 1

        worker.join(timeout=REQUEST_TIMEOUT_S)
        if worker.is_alive():
            print("[test4] FAIL: client thread didn't return in time")
            return 1
        if worker.error:
            print(f"[test4] FAIL: client request errored — {worker.error}")
            return 1
        if not worker.result or "choices" not in worker.result:
            print(f"[test4] FAIL: client got bad response — {worker.result!r}")
            return 1
        completion_text = worker.result["choices"][0].get("text", "")
        print(
            f"[test4] client got response: status={worker.status_code}, "
            f"completion_len={len(completion_text)}, "
            f"rerouted_from={worker.response_headers.get('x-rerouted-from') if worker.response_headers else None}"
        )

        # Grep the surviving engine's log for proof of the diversion.
        log_path = ENGINE_1_LOG if surviving_eid == 1 else ENGINE_0_LOG
        ok, msg = check_surviving_engine_log_for_reroute(log_path)
        print(f"[test4] engine {surviving_eid} log evidence: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        print("[test4] PASS")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
