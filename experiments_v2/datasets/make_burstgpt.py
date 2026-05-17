#!/usr/bin/env python3
"""Build a bursty workload using BurstGPT arrival trace + matched prompts.

Real BurstGPT trace (lzzmm/BurstGPT on HuggingFace) provides:
  - arrival Timestamp (seconds, integer)
  - Request tokens (input length)
  - Response tokens (output length)

It does NOT provide prompt text. Following community practice (JITServe,
TokenFlow), we use the trace's arrival pattern + token counts, and
substitute prompts from existing local datasets matched by length:

  - Request tokens < 4000  →  pick from sharegpt
  - Request tokens >= 4000 →  pick from arxivsumm

Output: experiments_v2/datasets/cached/burstgpt_mixed.jsonl
  fields: id, dataset, prompt, prompt_tokens, expected_output_tokens,
          arrival_offset_s, class

The workload's bursty arrival pattern is the value-add — single dataset
prompts can be paired with our existing Poisson scheduler too, but
BurstGPT bursts let us exercise the picker / router under realistic
production-style load spikes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path("/home/yzhong76/code/my-vllm-serving-system")
CACHED_DIR = REPO_ROOT / "experiments_v2/datasets/cached"
OUT_PATH = CACHED_DIR / "burstgpt_mixed.jsonl"
TOKENIZER_PATH = "/home/yzhong76/model/Qwen2.5-7B-Instruct"

NUM_REQUESTS = 60       # match other workloads
SHORT_RATIO = 0.7        # 70% from sharegpt, 30% from arxivsumm
MAX_PROMPT_TOKENS = 30000   # Qwen2.5-7B native ctx limit minus output budget

# Pick a window that gives roughly QPS ~ 2 (target the contended regime).
# BurstGPT trace covers 2 days at production scale. We pick a window
# where ~60 requests arrive within 30s window (avg QPS=2) with natural
# bursty variation.
#
# NOTE: BurstGPT's token counts are skewed short (mostly 1-3K) so we
# ignore them; the value-add is the arrival timing pattern (bursty
# vs Poisson). We assign class labels manually to get a 70:30 mix
# matching the mixed_short_long workload's distribution, so the two
# workloads only differ in arrival pattern.
TARGET_AVG_QPS = 2.0
WINDOW_S = int(NUM_REQUESTS / TARGET_AVG_QPS)   # = 30s


def load_local_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def find_burst_window(trace_rows: list[dict], window_s: int,
                      n_requests: int, rng: random.Random) -> list[dict]:
    """Find a contiguous window of `n_requests` arrivals spanning ~window_s.

    BurstGPT timestamps are integer seconds. We slide a candidate start
    forward until we find n_requests within window_s, OR fall back to
    'closest match' if no exact window fits.
    """
    # Filter to ChatGPT model only for consistency.
    chatgpt_rows = [r for r in trace_rows if r.get("Model") == "ChatGPT"]
    if len(chatgpt_rows) < n_requests:
        raise RuntimeError(
            f"not enough ChatGPT rows: {len(chatgpt_rows)} < {n_requests}"
        )

    # Sort by timestamp.
    chatgpt_rows.sort(key=lambda r: r["Timestamp"])

    # Sliding window scan. Find windows where exactly n_requests fall
    # within [t, t + window_s).
    best_window: list[dict] = []
    best_score = float("inf")  # prefer windows close to target n
    candidate_starts = list(range(0, len(chatgpt_rows) - n_requests))
    rng.shuffle(candidate_starts)
    for start_idx in candidate_starts[:5000]:
        t0 = chatgpt_rows[start_idx]["Timestamp"]
        end_idx = start_idx + n_requests - 1
        if end_idx >= len(chatgpt_rows):
            continue
        t_end = chatgpt_rows[end_idx]["Timestamp"]
        actual_span = t_end - t0
        # Want actual_span close to window_s.
        score = abs(actual_span - window_s)
        if score < best_score:
            best_score = score
            best_window = chatgpt_rows[start_idx:end_idx + 1]
            if score <= 2:  # close enough
                break

    if not best_window:
        raise RuntimeError("could not find a suitable BurstGPT window")
    return best_window


def main() -> None:
    print(f"[burstgpt] loading tokenizer from {TOKENIZER_PATH}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("[burstgpt] loading local sharegpt + arxivsumm jsonls")
    sharegpt = load_local_jsonl(CACHED_DIR / "sharegpt_5000.jsonl")
    arxivsumm = load_local_jsonl(CACHED_DIR / "arxivsumm.jsonl")
    # Pre-sort each by prompt_tokens for length matching.
    sharegpt_by_len = sorted(sharegpt, key=lambda r: r["prompt_tokens"])
    arxivsumm_by_len = sorted(arxivsumm, key=lambda r: r["prompt_tokens"])
    print(f"  sharegpt:  {len(sharegpt)} records "
          f"(token range {sharegpt_by_len[0]['prompt_tokens']} – "
          f"{sharegpt_by_len[-1]['prompt_tokens']})")
    print(f"  arxivsumm: {len(arxivsumm)} records "
          f"(token range {arxivsumm_by_len[0]['prompt_tokens']} – "
          f"{arxivsumm_by_len[-1]['prompt_tokens']})")

    print("[burstgpt] loading BurstGPT trace (lzzmm/BurstGPT)")
    ds = load_dataset("lzzmm/BurstGPT", split="train")
    # Convert to plain list for slicing — easier than dataset slicing.
    trace_rows = [
        {"Timestamp": r["Timestamp"],
         "Model": r["Model"],
         "Request tokens": r["Request tokens"],
         "Response tokens": r["Response tokens"]}
        for r in ds
    ]
    print(f"[burstgpt] trace loaded: {len(trace_rows)} rows")

    rng = random.Random(42)
    window = find_burst_window(
        trace_rows, window_s=WINDOW_S, n_requests=NUM_REQUESTS, rng=rng,
    )
    t0 = window[0]["Timestamp"]
    arrivals = [(r["Timestamp"] - t0, r["Request tokens"], r["Response tokens"])
                for r in window]
    print(f"[burstgpt] selected window: spans {arrivals[-1][0]}s")
    print(f"           request token stats: min={min(a[1] for a in arrivals)} "
          f"max={max(a[1] for a in arrivals)} "
          f"avg={sum(a[1] for a in arrivals)//len(arrivals)}")

    # Insert sub-second jitter so requests aren't all clumped at integer
    # seconds (BurstGPT timestamps are integer). Use deterministic
    # jitter from the RNG so reruns are identical.
    jitter_rng = random.Random(43)
    schedule_offsets = []
    for off, _, _ in arrivals:
        schedule_offsets.append(off + jitter_rng.random())   # off + [0,1)s
    schedule_offsets.sort()

    # Match each BurstGPT entry to a real prompt by length.
    def match_prompt(target_tokens: int) -> dict:
        target_tokens = min(target_tokens, MAX_PROMPT_TOKENS)
        if target_tokens < SHORT_LONG_BOUNDARY_TOKENS:
            pool = sharegpt_by_len
        else:
            pool = arxivsumm_by_len
        # Binary-search-ish: pick the record closest in length.
        best_rec, best_diff = None, float("inf")
        # Linear scan is fine — pool is at most a few K entries.
        for r in pool:
            diff = abs(r["prompt_tokens"] - target_tokens)
            if diff < best_diff:
                best_diff = diff
                best_rec = r
                if diff == 0:
                    break
        return best_rec

    out_records = []
    short_count = long_count = 0
    used_ids = set()  # avoid duplicate prompts in one workload
    for i, (off, req_t, resp_t) in enumerate(arrivals):
        rec = match_prompt(req_t)
        # Avoid reusing exact same record. If duplicate, pick next-best.
        attempts = 0
        while rec["id"] in used_ids and attempts < 50:
            # Random walk one step in length to find a different rec.
            pool = (sharegpt_by_len if req_t < SHORT_LONG_BOUNDARY_TOKENS
                    else arxivsumm_by_len)
            jitter_idx = rng.randrange(max(1, len(pool)))
            rec = pool[jitter_idx]
            attempts += 1
        used_ids.add(rec["id"])

        is_short = req_t < SHORT_LONG_BOUNDARY_TOKENS
        if is_short:
            short_count += 1
        else:
            long_count += 1
        out_records.append({
            "id": f"burstgpt-{i:03d}",
            "dataset": "burstgpt_mixed",
            "prompt": rec["prompt"],
            "prompt_tokens": rec["prompt_tokens"],
            "expected_output_tokens": min(int(resp_t), 400),  # cap at 400
            "arrival_offset_s": schedule_offsets[i],
            "class": "short" if is_short else "long",
        })

    print(f"[burstgpt] built workload: short={short_count} long={long_count}")
    print(f"           arrival window: 0.00s – "
          f"{schedule_offsets[-1]:.2f}s = "
          f"{NUM_REQUESTS / schedule_offsets[-1]:.2f} avg QPS")
    # Burstiness check: how many requests in busiest 5s window?
    busy_max = 0
    for i in range(len(schedule_offsets)):
        t_i = schedule_offsets[i]
        count_in_window = sum(1 for t in schedule_offsets
                              if t_i <= t < t_i + 5.0)
        busy_max = max(busy_max, count_in_window)
    print(f"           busiest 5s window: {busy_max} requests "
          f"({busy_max/5:.2f} qps peak)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"[burstgpt] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
