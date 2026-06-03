#!/usr/bin/env python3
"""Standalone GPU KV-pool monitor — run in a SECOND terminal, next to the demo.

Polls the Ferry engine's own gauge `vllm:kv_cache_usage_perc` from /metrics
(~2x/s) and prints a live, scrolling timeline of how full the KV-block pool is,
plus whether a host-RAM checkpoint exists. This is the signal nvidia-smi can't
show: vLLM grabs the whole pool at startup so device memory never moves; this is
the occupancy *inside* that pool, which falls when a paused request's KV is freed
and rises again when it is reloaded from host RAM.

Usage (run this BEFORE/alongside demo_show_text.py, which serves on port 8401):
    python3 "experiments_v2/CSE 232/watch_kv_pool.py"
    python3 "experiments_v2/CSE 232/watch_kv_pool.py" 8401     # explicit port
Ctrl-C to stop. It waits for the engine and keeps running if it restarts.
"""
import re
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8401
CKPT_DIR = Path("/dev/shm/vllm_ft_checkpoints")
INTERVAL = 0.5
_metric_re = re.compile(r"^vllm:kv_cache_usage_perc\b.*?\s([0-9.eE+-]+)\s*$", re.M)


def read_kv_pct():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=2) as r:
            body = r.read().decode()
        vals = [float(m) for m in _metric_re.findall(body)]
        return max(vals) * 100.0 if vals else None
    except Exception:
        return None


def n_ckpt():
    try:
        return sum(1 for _ in CKPT_DIR.iterdir())
    except Exception:
        return 0


def bar(pct, width=34, full_scale=20.0):
    n = max(0, min(width, int(round((pct / full_scale) * width))))
    return "█" * n + "░" * (width - n)


def main():
    print(f"=== GPU KV-pool monitor — engine on port {PORT} "
          f"(Ctrl-C to stop) ===", flush=True)
    print("    (run the demo in the other window; watch this fall to 0 on the "
          "pause and climb back on resume)\n", flush=True)
    t0 = time.time()
    last_wait_print = -1e9
    while True:
        pct, files = read_kv_pct(), n_ckpt()
        t = time.time() - t0
        if pct is None:
            # keep polling fast so we catch the engine the instant it's up,
            # but only announce "waiting" every 5s so it doesn't spam
            if t - last_wait_print >= 5.0:
                last_wait_print = t
                print(f"  [t={t:5.1f}s]  waiting for engine on :{PORT} ...", flush=True)
            time.sleep(INTERVAL)
            continue
        flag = "host RAM copy: yes" if files else "host RAM copy: no "
        print(f"  [t={t:5.1f}s]  GPU KV pool: {pct:5.1f}%  [{bar(pct)}]   {flag}",
              flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[monitor stopped]")
