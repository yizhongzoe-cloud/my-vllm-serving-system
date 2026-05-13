#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test 5: picker-triggered cross-engine redispatch (Option B).

End-to-end flow being exercised:
  1. Two engines (0, 1) + router, with SLO priority preempt enabled.
     max-num-seqs=1 on each engine so a single in-flight request fills
     the engine and any second request lands in waiting.
  2. Client sends a loose-SLO long-context completion → router lands it
     on engine X.
  3. Wait for engine X to publish at least one KV checkpoint chunk.
  4. Client sends a tight-SLO short completion. Engine X has only one
     seq slot, so the tight request goes to waiting. Picker fires:
     loose victim has much more slack than tight head, so engine X
     calls _preempt_for_slo_redispatch — frees loose's KV, publishes
     /dev/shm/vllm_ft_preempt_queue/<id>.json.
  5. Router's preempt-queue poller sees the file, cancels the loose
     request's HTTP task on engine X, redispatches to engine Y with
     is_rerouted=True. Engine Y's V3 reload state machine restores KV
     from /dev/shm/vllm_ft_checkpoints.
  6. Both clients receive valid responses.

Pass conditions:
  - Loose client got 200 + non-empty completion (with X-Rerouted-From
    header set to engine X's id).
  - Tight client got 200 + non-empty completion.
  - Router metrics: reroute_count >= 1.
  - Engine X log shows "FT SLO redispatch: ... preempted for cross-engine".
  - Engine Y log shows "Cross-engine reroute" + "FT overlap V3: ... done".

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
ENGINE_0_PORT = 8501
ENGINE_1_PORT = 8502
ROUTER_PORT = 8500
STATUS_DIR = Path("/dev/shm/vllm_ft_engine_status")
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")
PREEMPT_QUEUE_DIR = Path("/dev/shm/vllm_ft_preempt_queue")
SCRIPT_DIR = Path(__file__).resolve().parent
ENGINE_0_LOG = SCRIPT_DIR / "test5_engine0.log"
ENGINE_1_LOG = SCRIPT_DIR / "test5_engine1.log"
ROUTER_LOG = SCRIPT_DIR / "test5_router.log"
ENGINE_READY_TIMEOUT_S = 240
ROUTER_READY_TIMEOUT_S = 30
LOOSE_PROMPT_TOKENS = 4096
LOOSE_MAX_OUTPUT_TOKENS = 200
TIGHT_PROMPT_TOKENS = 128
TIGHT_MAX_OUTPUT_TOKENS = 50
PUBLISH_WAIT_TIMEOUT_S = 90
REROUTE_DETECT_TIMEOUT_S = 30
REQUEST_TIMEOUT_S = 600


def cleanup_shm() -> None:
    shutil.rmtree(STATUS_DIR, ignore_errors=True)
    shutil.rmtree(REQ_MAP_DIR, ignore_errors=True)
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    shutil.rmtree(PREEMPT_QUEUE_DIR, ignore_errors=True)


def start_engine(
    engine_id: int, port: int, gpu: str, log_path: Path
) -> subprocess.Popen:
    env = os.environ.copy()
    env["FT_ROUTER_SHM_BUS"] = "1"
    env["VLLM_FT_ENGINE_ID"] = str(engine_id)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # V3 reload + ckpt machinery: engine X publishes KV, engine Y
    # restores it on the redispatched request.
    env["FT_CAPACITY_PREEMPT_RELOAD"] = "1"
    env["FT_CAPACITY_PREEMPT_RELOAD_OVERLAP"] = "1"
    env["FT_DELTA_CHECKPOINT"] = "1"
    # Picker: enabled with a low min_gap_ms so the slack difference
    # between our loose (60s SLO) and tight (500ms SLO) requests
    # easily clears the threshold.
    env["SLO_PRIORITY_PREEMPT"] = "1"
    env["SLO_PRIORITY_PREEMPT_MIN_GAP_MS"] = "500"
    env["SLO_PRIORITY_PREEMPT_MIN_INTERVAL_MS"] = "0"
    env["SLO_PRIORITY_PREEMPT_PER_REQ_COOLDOWN_MS"] = "0"
    # Short fallback timer so a test failure mode (router not picking
    # up) doesn't hang the loose request for the default 5s.
    env["FT_REDISPATCH_TIMEOUT_S"] = "3.0"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len",
        str(LOOSE_PROMPT_TOKENS + LOOSE_MAX_OUTPUT_TOKENS + 256),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.5",
        "--dtype", "float16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--disable-log-requests",
    ]
    log_f = open(log_path, "w")
    print(
        f"[test5] launching engine {engine_id} on GPU {gpu} "
        f"→ {log_path.name}"
    )
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
    print(f"[test5] launching router → {ROUTER_LOG.name}")
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


def build_prompt(num_tokens: int) -> str:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = "The quick brown fox jumps over the lazy dog. " * 16
    ids = tok.encode(base)
    repeats = num_tokens // max(1, len(ids)) + 2
    text = base * repeats
    return tok.decode(tok.encode(text)[:num_tokens])


class ClientWorker(threading.Thread):
    """Sends one completion request; stores result.

    SLO is passed via vllm_xargs.ttft_slo_ms / tpot_slo_ms so the
    engine's picker sees it via Request.ttft_slo_ms after add_request
    lifts the value out of extra_args.

    `target_url` decides where the request is POSTed:
      - router URL (e.g. http://127.0.0.1:8500): normal flow, router
        load-balances.
      - engine URL directly: bypasses router. Used in this test to
        force both reqs onto engine 0 so its picker fires.
    """
    def __init__(
        self,
        prompt: str,
        max_tokens: int,
        ttft_slo_ms: float,
        tpot_slo_ms: float,
        label: str,
        target_url: str,
        router_req_id: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.ttft_slo_ms = ttft_slo_ms
        self.tpot_slo_ms = tpot_slo_ms
        self.label = label
        self.target_url = target_url
        self.router_req_id = router_req_id
        self.result: dict | None = None
        self.error: str | None = None
        self.status_code: int | None = None
        self.response_headers: dict | None = None

    def run(self) -> None:
        vllm_xargs = {
            "ttft_slo_ms": self.ttft_slo_ms,
            "tpot_slo_ms": self.tpot_slo_ms,
        }
        # When bypassing router, supply our own router_req_id so the
        # engine still writes the req_map file. Otherwise the engine
        # has no router_req_id and won't publish to preempt_queue
        # (publish path is gated on router_req_id being non-empty).
        if self.router_req_id is not None:
            vllm_xargs["router_req_id"] = self.router_req_id
        body = {
            "model": MODEL,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "vllm_xargs": vllm_xargs,
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.target_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=REQUEST_TIMEOUT_S
            ) as r:
                self.status_code = r.status
                self.response_headers = dict(r.headers.items())
                self.result = json.loads(r.read())
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"


def wait_for_published_chunk(timeout_s: int) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if CKPT_DIR.exists():
            chunks = list(CKPT_DIR.rglob("chunk_*.pt"))
            if chunks:
                return True, (
                    f"saw {len(chunks)} chunk file(s); "
                    f"first={chunks[0].name}"
                )
        time.sleep(0.5)
    return False, f"no chunk files in {CKPT_DIR} within {timeout_s}s"


def wait_for_reroute(timeout_s: int) -> tuple[bool, dict]:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        m = fetch_json(f"http://127.0.0.1:{ROUTER_PORT}/metrics")
        if m is not None:
            last = m
            if m.get("reroute_count", 0) >= 1:
                return True, m
        time.sleep(0.2)
    return False, last


def get_target_engine_id() -> int | None:
    if not REQ_MAP_DIR.exists():
        return None
    for fp in REQ_MAP_DIR.iterdir():
        try:
            return int(json.loads(fp.read_text()).get("engine_id"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return None


def check_origin_log(log_path: Path) -> tuple[bool, str]:
    try:
        txt = log_path.read_text(errors="replace")
    except OSError as e:
        return False, f"{log_path.name} unreadable: {e}"
    needle = "FT SLO redispatch:"
    if needle not in txt:
        return False, (
            f"{log_path.name} missing '{needle}' "
            "(picker preempt didn't fire on origin engine)"
        )
    return True, f"{log_path.name} shows picker preempt"


def check_recipient_log(log_path: Path) -> tuple[bool, str]:
    try:
        txt = log_path.read_text(errors="replace")
    except OSError as e:
        return False, f"{log_path.name} unreadable: {e}"
    has_diversion = "Cross-engine reroute" in txt
    has_reload_done = "FT overlap V3: " in txt and "done" in txt
    if not has_diversion:
        return False, (
            f"{log_path.name} missing 'Cross-engine reroute' "
            "(redispatch didn't divert into V3 state machine)"
        )
    if not has_reload_done:
        return False, (
            f"{log_path.name} missing 'FT overlap V3: ... done' "
            "(restore from shm didn't complete)"
        )
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
            f"http://127.0.0.1:{ENGINE_0_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ) or not wait_url_ready(
            f"http://127.0.0.1:{ENGINE_1_PORT}/health",
            ENGINE_READY_TIMEOUT_S,
        ):
            print("[test5] FAIL: an engine never became healthy")
            return 1
        print("[test5] both engines ready")

        router = start_router()
        if not wait_url_ready(
            f"http://127.0.0.1:{ROUTER_PORT}/health",
            ROUTER_READY_TIMEOUT_S,
        ):
            print("[test5] FAIL: router never became healthy")
            return 1
        print("[test5] router ready")

        if not wait_both_engines_alive_in_router():
            print("[test5] FAIL: router didn't see both engines alive")
            return 1

        loose_prompt = build_prompt(LOOSE_PROMPT_TOKENS)
        tight_prompt = build_prompt(TIGHT_PROMPT_TOKENS)

        # Loose: through router (normal flow). Router will dispatch to
        # one of the two engines; we'll figure out which one by reading
        # req_map and then send the tight req directly to that engine.
        loose = ClientWorker(
            loose_prompt, LOOSE_MAX_OUTPUT_TOKENS,
            ttft_slo_ms=60_000.0, tpot_slo_ms=2000.0, label="loose",
            target_url=(
                f"http://127.0.0.1:{ROUTER_PORT}/v1/completions"
            ),
        )
        print("[test5] firing loose request (60s SLO)")
        loose.start()

        ok, msg = wait_for_published_chunk(PUBLISH_WAIT_TIMEOUT_S)
        print(
            f"[test5] published chunk seen: "
            f"{'OK' if ok else 'FAIL'} — {msg}"
        )
        if not ok:
            return 1

        origin_eid = get_target_engine_id()
        if origin_eid not in (0, 1):
            print(
                f"[test5] FAIL: couldn't determine origin engine "
                f"(got {origin_eid})"
            )
            return 1
        recipient_eid = 1 - origin_eid
        origin_engine_port = (
            ENGINE_0_PORT if origin_eid == 0 else ENGINE_1_PORT
        )
        print(
            f"[test5] loose request running on engine {origin_eid}; "
            f"firing tight request directly to engine {origin_eid} "
            f"(bypass router so both reqs land on origin and picker "
            f"actually fires)"
        )

        # Tight: bypass router and POST directly to origin engine so
        # both reqs sit on the same engine. Router-based dispatch would
        # load-balance the tight req to the other engine (it has 0
        # in_flight, looks idler), which would leave the origin's
        # waiting queue empty and picker would never fire.
        tight = ClientWorker(
            tight_prompt, TIGHT_MAX_OUTPUT_TOKENS,
            ttft_slo_ms=500.0, tpot_slo_ms=100.0, label="tight",
            target_url=(
                f"http://127.0.0.1:{origin_engine_port}/v1/completions"
            ),
        )
        tight.start()

        ok, metrics = wait_for_reroute(REROUTE_DETECT_TIMEOUT_S)
        print(
            f"[test5] router observed redispatch: "
            f"{'OK' if ok else 'FAIL'} — {metrics}"
        )
        if not ok:
            return 1

        loose.join(timeout=REQUEST_TIMEOUT_S)
        tight.join(timeout=REQUEST_TIMEOUT_S)
        if loose.is_alive() or tight.is_alive():
            print("[test5] FAIL: a client thread didn't return in time")
            return 1
        for w in (loose, tight):
            if w.error:
                print(
                    f"[test5] FAIL: {w.label} request errored — "
                    f"{w.error}"
                )
                return 1
            if not w.result or "choices" not in w.result:
                print(
                    f"[test5] FAIL: {w.label} got bad response — "
                    f"{w.result!r}"
                )
                return 1
        loose_text = loose.result["choices"][0].get("text", "")
        tight_text = tight.result["choices"][0].get("text", "")
        print(
            f"[test5] loose: status={loose.status_code} "
            f"completion_len={len(loose_text)} "
            f"rerouted_from={loose.response_headers.get('x-rerouted-from') if loose.response_headers else None}"
        )
        print(
            f"[test5] tight: status={tight.status_code} "
            f"completion_len={len(tight_text)}"
        )
        if not loose.response_headers or not loose.response_headers.get(
            "x-rerouted-from"
        ):
            print(
                "[test5] FAIL: loose response missing X-Rerouted-From "
                "header (router didn't reroute via the cancellation path)"
            )
            return 1

        origin_log = ENGINE_0_LOG if origin_eid == 0 else ENGINE_1_LOG
        recipient_log = (
            ENGINE_0_LOG if recipient_eid == 0 else ENGINE_1_LOG
        )
        ok, msg = check_origin_log(origin_log)
        print(f"[test5] origin log: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1
        ok, msg = check_recipient_log(recipient_log)
        print(f"[test5] recipient log: {'OK' if ok else 'FAIL'} — {msg}")
        if not ok:
            return 1

        print("[test5] PASS")
        return 0
    finally:
        if router is not None:
            shutdown(router)
        shutdown(e0)
        shutdown(e1)
        cleanup_shm()


if __name__ == "__main__":
    sys.exit(main())
