#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-GPU shm round-trip test.

Orchestrator: writer subprocess on GPU 0 writes /dev/shm; reader
subprocess on GPU 1 reads it back via worker `restore_kv_blocks` RPC.

Two separate Python processes are required because CUDA_VISIBLE_DEVICES
cannot be re-set after CUDA initializes in the same process.

PASS: writer flushes shm, reader restores tokens > 0 for every req.

Usage:
    python experiments_v2/smoke/test_cross_gpu_shm.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WRITER = SCRIPT_DIR / "shm_writer.py"
READER = SCRIPT_DIR / "shm_reader.py"
SHM_DIR = Path("/dev/shm/vllm_ft_checkpoints")
WRITER_GPU = "0"
READER_GPU = "1"
# Full stdout + stderr dumps. vllm logger writes to stdout by default, so the
# bulk of worker log lives in *_stdout.log, not *_stderr.log.
WRITER_STDOUT_LOG = SCRIPT_DIR / "writer_stdout.log"
WRITER_STDERR_LOG = SCRIPT_DIR / "writer_stderr.log"
READER_STDOUT_LOG = SCRIPT_DIR / "reader_stdout.log"
READER_STDERR_LOG = SCRIPT_DIR / "reader_stderr.log"


def run_subproc(label: str, cmd: list[str], env_overrides: dict[str, str],
                stdout_log: Path, stderr_log: Path) -> tuple[int, str]:
    """Run a child process, capturing stdout to a PIPE (returned as str) AND
    mirroring it to a log file, while stderr goes to a separate file.
    """
    env = os.environ.copy()
    env.update(env_overrides)
    print(f"[orch] launching {label}: CUDA_VISIBLE_DEVICES="
          f"{env_overrides.get('CUDA_VISIBLE_DEVICES')}, "
          f"stdout → {stdout_log.name}, stderr → {stderr_log.name}")
    stderr_f = open(stderr_log, "w")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_f,
            env=env, text=True,
        )
        stdout, _ = proc.communicate()
        # Persist full stdout for post-mortem (vllm INFO log lives here).
        stdout_log.write_text(stdout)
        return proc.returncode, stdout
    finally:
        stderr_f.close()


def main() -> int:
    if not WRITER.exists() or not READER.exists():
        print(f"[orch] FAIL: writer/reader missing in {SCRIPT_DIR}")
        return 1

    # Fresh /dev/shm to avoid contaminating with leftover state.
    shutil.rmtree(SHM_DIR, ignore_errors=True)

    # ── Phase 1: writer on GPU 0 ──
    rc, stdout = run_subproc(
        "writer", [sys.executable, str(WRITER)],
        {"CUDA_VISIBLE_DEVICES": WRITER_GPU},
        WRITER_STDOUT_LOG, WRITER_STDERR_LOG,
    )
    if rc != 0:
        print(f"[orch] FAIL: writer exited rc={rc}")
        print(f"--- writer stdout (full at {WRITER_STDOUT_LOG}) ---")
        print(WRITER_STDOUT_LOG.read_text()[-2000:])
        print(f"--- writer stderr (full at {WRITER_STDERR_LOG}) ---")
        print(WRITER_STDERR_LOG.read_text()[-2000:])
        return 1

    # Writer prints exactly one JSON line on stdout summarizing the
    # publish; anything else is informational (and goes to stderr).
    json_line = next(
        (ln for ln in reversed(stdout.splitlines()) if ln.startswith("{")),
        None,
    )
    if json_line is None:
        print("[orch] FAIL: no JSON on writer stdout")
        print("--- writer stdout ---")
        print(stdout)
        return 1
    writer_out = json.loads(json_line)
    req_ids = writer_out["req_ids"]
    print(f"[orch] writer published {writer_out['n_chunks']} chunks, "
          f"{writer_out['n_manifests']} manifests for "
          f"{len(req_ids)} req_ids")

    # Give vllm worker subprocess + CUDA context full time to tear down
    # before launching the reader on the other GPU. Cheap insurance against
    # transient inter-process races we saw in the first orchestrator run.
    time.sleep(2.0)

    # ── Phase 2: reader on GPU 1 ──
    rc, stdout = run_subproc(
        "reader",
        [sys.executable, str(READER), *req_ids],
        {"CUDA_VISIBLE_DEVICES": READER_GPU},
        READER_STDOUT_LOG, READER_STDERR_LOG,
    )
    json_line = next(
        (ln for ln in reversed(stdout.splitlines()) if ln.startswith("{")),
        None,
    )
    if json_line is None:
        print(f"[orch] FAIL: no JSON on reader stdout (rc={rc})")
        print(f"--- reader stdout (full at {READER_STDOUT_LOG}) ---")
        print(READER_STDOUT_LOG.read_text()[-2000:])
        print(f"--- reader stderr (full at {READER_STDERR_LOG}) ---")
        print(READER_STDERR_LOG.read_text()[-2000:])
        return 1
    reader_out = json.loads(json_line)
    n_pass = reader_out["pass"]
    n_fail = reader_out["fail"]
    print(f"[orch] reader restored {n_pass}/{len(req_ids)} reqs from shm")
    for entry in reader_out["per_req"]:
        if not entry["ok"]:
            err = entry.get("err") or f"tokens={entry.get('tokens', 0)}"
            print(f"  {entry['req_id']}: FAIL ({err})")

    # Leave /dev/shm in place for post-mortem; only clean if PASS.
    if rc == 0 and n_fail == 0:
        shutil.rmtree(SHM_DIR, ignore_errors=True)
        print("[cross-gpu-shm] PASS")
        return 0
    # On any reader failure, dump tails of BOTH stdout (vllm log lives here
    # by default) and stderr, so the root cause isn't hidden.
    print(f"--- reader stdout (full at {READER_STDOUT_LOG}) ---")
    print(READER_STDOUT_LOG.read_text()[-4000:])
    print(f"--- reader stderr (full at {READER_STDERR_LOG}) ---")
    print(READER_STDERR_LOG.read_text()[-2000:])
    print("[cross-gpu-shm] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
