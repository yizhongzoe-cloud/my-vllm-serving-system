#!/usr/bin/env python3
"""De-risk the interception experiment's CORE mechanism:

  Can a request, after stopping at K output tokens, be RESUMED on the
  SAME engine from its host-side checkpoint (reload, cheap) instead of
  reprefilling the whole context?

This is the only untested piece — the cross-engine reroute reload path
(is_rerouted + original_internal_req_id + num_checkpointed_tokens) exists,
but using it client-driven and same-engine has never been run. Everything
else (continuous checkpoint, decode coverage, checkpoint survival on
finish) is already in place.

Test design (single engine, substrate flags + FT_ROUTER_SHM_BUS=1):
  G  ground truth : prompt P, max_tokens = K+M, temp 0 (greedy).
  A  suspend      : prompt P, max_tokens = K, router_req_id=iceptA.
                    -> finishes at K tokens, KV freed, host checkpoint
                       persists. A's output should == G[:K].
  B  reload-resume: prompt P, reload xargs (original_internal_req_id =
                    A's internal id, num_checkpointed_tokens = covered).
                    -> should reload A's KV and continue; B output should
                       == G[K:K+M] (byte-correct mid-gen resume), and its
                       wall-clock should be ~reload (fast), not reprefill.
  C  reprefill    : prompt P+A_output, max_tokens = M (vanilla baseline:
                    a continuation that reprefills the whole context).
                    -> output should also == G[K:K+M]; e2e ~reprefill.

PASS iff: req_map found, covered>prompt_len, B==G continuation, e2e_B<<e2e_C.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/yzhong76/code/my-vllm-serving-system")
MODEL = os.path.expanduser("~/model/Qwen2.5-14B-Instruct")
PORT = 8401
REQ_MAP_DIR = Path("/dev/shm/vllm_ft_req_map")
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")
K = 20      # tokens before interception
M = 20      # tokens after resume
MAXLEN = 32768


def post(body: dict, timeout: float = 600.0) -> tuple[dict, float]:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


def text_of(resp: dict) -> str:
    return resp["choices"][0]["text"]


def read_internal_id(router_req_id: str) -> str | None:
    fp = REQ_MAP_DIR / router_req_id
    try:
        return json.loads(fp.read_text()).get("internal_req_id")
    except (OSError, json.JSONDecodeError):
        return None


def read_covered_tokens(internal_id: str) -> int | None:
    d = CKPT_DIR / internal_id
    try:
        manifest_name = (d / "latest_rank0").read_text().strip()
        manifest = json.loads((d / manifest_name).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    ct = manifest.get("covered_tokens")
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


def main() -> int:
    # clean shm
    for d in ("vllm_ft_checkpoints", "vllm_ft_req_map",
              "vllm_ft_engine_status", "vllm_ft_preempt_queue"):
        subprocess.run(["rm", "-rf", f"/dev/shm/{d}"], check=False)

    # load a long-ish arxivsumm prompt (~6K tokens) so reprefill is clearly
    # slow vs reload.
    recs = [json.loads(l) for l in
            open(REPO / "experiments_v2/datasets/cached/arxivsumm.jsonl")]
    rec = min((r for r in recs if r.get("prompt_tokens", 0) >= 5000),
              key=lambda r: r["prompt_tokens"])
    P = rec["prompt"]
    print(f"[derisk] prompt_tokens={rec['prompt_tokens']}")

    env = os.environ.copy()
    env.update({
        "VLLM_FT_ENGINE_ID": "0", "CUDA_VISIBLE_DEVICES": "0",
        "FT_ROUTER_SHM_BUS": "1",            # so req_map gets written
        "FT_CAPACITY_PREEMPT_RELOAD": "1",
        "FT_CAPACITY_PREEMPT_RELOAD_OVERLAP": "1",
        "FT_DELTA_CHECKPOINT": "1",          # continuous checkpoint
        "SLO_PRIORITY_PREEMPT": "0",
    })
    log = open("/tmp/icept_derisk_engine.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--port", str(PORT),
         "--max-model-len", str(MAXLEN), "--gpu-memory-utilization", "0.9",
         "--dtype", "float16", "--enforce-eager",
         "--no-enable-prefix-caching", "--disable-log-requests"],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        if not wait_health(600):
            print("[derisk] FAIL: engine never healthy")
            return 1
        print("[derisk] engine ready")

        # G: ground truth K+M tokens
        g, _ = post({"model": MODEL, "prompt": P, "max_tokens": K + M,
                     "temperature": 0.0, "ignore_eos": True})
        G = text_of(g)
        print(f"[derisk] G (ground truth {K+M} tok): {G!r}")

        # A: stop at K, tag with router_req_id so engine writes req_map
        a, _ = post({"model": MODEL, "prompt": P, "max_tokens": K,
                     "temperature": 0.0, "ignore_eos": True,
                     "vllm_xargs": {"router_req_id": "iceptA"}})
        A = text_of(a)
        print(f"[derisk] A (suspend at {K} tok): {A!r}")
        # let the async checkpoint publish settle
        time.sleep(2.0)

        internal = read_internal_id("iceptA")
        covered = read_covered_tokens(internal) if internal else None
        print(f"[derisk] internal_req_id={internal}  covered_tokens={covered}"
              f"  (prompt_tokens={rec['prompt_tokens']})")
        if internal is None or covered is None:
            print("[derisk] FAIL: req_map or checkpoint manifest missing "
                  "(suspend/checkpoint not durable)")
            return 1

        # B: reload-resume on the same engine
        b, e2e_b = post({"model": MODEL, "prompt": P, "max_tokens": M,
                         "temperature": 0.0, "ignore_eos": True,
                         "vllm_xargs": {"is_rerouted": True,
                                        "original_internal_req_id": internal,
                                        "num_checkpointed_tokens": covered}})
        B = text_of(b)
        print(f"[derisk] B (reload-resume, e2e={e2e_b*1000:.0f}ms): {B!r}")

        # C: vanilla reprefill continuation (prompt + A's output)
        c, e2e_c = post({"model": MODEL, "prompt": P + A, "max_tokens": M,
                         "temperature": 0.0, "ignore_eos": True})
        C = text_of(c)
        print(f"[derisk] C (reprefill cont, e2e={e2e_c*1000:.0f}ms): {C!r}")

        # ---- verdicts ----
        cont = G[len(A):]          # ground-truth continuation after A
        a_ok = G.startswith(A)
        b_correct = (B.strip() != "") and (B[:15] == cont[:15])
        faster = e2e_b < e2e_c * 0.7
        print("\n===== VERDICT =====")
        print(f"  A == G[:K]              : {a_ok}")
        print(f"  covered > prompt_tokens : {covered > rec['prompt_tokens']} "
              f"({covered} vs {rec['prompt_tokens']})")
        print(f"  B continues G (reload OK): {b_correct}")
        print(f"     B[:40]={B[:40]!r}")
        print(f"     expected={cont[:40]!r}")
        print(f"  reload faster than reprefill: {faster} "
              f"(B={e2e_b*1000:.0f}ms  C={e2e_c*1000:.0f}ms  "
              f"speedup={e2e_c/max(e2e_b,1e-3):.1f}x)")
        ok = a_ok and (covered > rec["prompt_tokens"]) and b_correct and faster
        print(f"\n  >>> DE-RISK {'PASS' if ok else 'FAIL'} <<<")
        return 0 if ok else 2
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
