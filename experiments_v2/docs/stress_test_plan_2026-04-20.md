# Stress Test Plan — 2026-04-20

## Motivation

Overnight 2026-04-19 results show OS (V2 = `FT_CHECKPOINT_STEP_INTERVAL=2`) ties NR on goodput/completion across all tested workloads (W1/W2/W4) but never strictly wins on fg_p95. Component ablation (P5) shows ~907ms of V2's 1264ms fg_p95 on W2 is OS-specific orchestration overhead (solver 384ms + snapshot 228ms + misc 295ms).

**Hypothesis**: Our test regime is too forgiving for NR. OS's design assumptions (Benders admission, proactive checkpoint) require *pressure* to pay off — long context, saturated RPS, multi-fault, tight SLO. We haven't hit any of these.

**Goal**: Construct stress workloads that activate at least 2 of OS's "should-win" conditions, then fairly compare V2 vs NR. Three possible outcomes:
1. **OS wins** → paper story confirmed, plan scale-up validation
2. **OS ties on stress** → OS has value (no regression under pressure), reframe paper as mechanism contribution
3. **OS loses on stress** → algorithm limits in 8B regime, pivot framing or scale up

## OS "should-win" conditions checklist (none currently satisfied)

| Condition | Why it helps OS | Current state |
|---|---|---|
| Saturated RPS (queue pressure) | NR's reprefill chokes prefill slots → goodput drop; OS admission protects flow | RPS=1.5, slack |
| Long context (8k+) | Prefill cost ∝ prompt length; NR's "redo prefill" becomes expensive; OS's KV reload doesn't scale with context | W2 p95≈2k |
| Fault burst (≥2 faults) | Solver reroute decisions compound value; NR reprefills fresh each time | Single F2 only |
| Many replicas (dp≥4) | Solver reroute space grows; MIP overhead amortized | dp=2 (tight) |
| Tight SLO | Small delay = SLO violation; admission value ↑ | fg SLO = 3s (loose) |

Target: hit 2-3 of these in new workloads.

---

## Phase 0 — Data reconnaissance (30 min, blocking)

- [x] **P0-1** Read `experiments_v2/config_8b.yaml`
- [x] **P0-2** Read `experiments_v2/run.py` fault logic
- [x] **P0-3** Quantify existing prompt length distributions
- [ ] **P0-4** RPS saturation sweep — DEFERRED to Phase 3 smoke (conservative estimate used for initial run)

### P0 findings

**Config constraints** (from config_8b.yaml):
- `max_model_len = 8192` — HARD ceiling. Any W5 prompt + generation must fit ≤ 8192 tokens.
  - Raising to 16384 is feasible (Llama-3.1 supports 128k natively) but requires GPU memory headroom recheck.
- `run_duration_sec = 300.0`, `Heavy RPS = 1.5` → 300 × 1.5 = 450 requests per run.
- Global SLO: `ttft_ms = 2000`, `failure_gap_ms = 3000` (loose).
- `W3_Instruct` already has `tpot_slo_ms = 50.0` (tight) — good base for W7.
- `max_gpu_failures = 1` per baseline — means only ONE GPU kill per run supported. Multi-fault (W6) needs server-side change.

**Prompt length distributions** (actual data):

| dataset | n | p50 | p90 | p95 | p99 | max | ≥4k tokens |
|---|---|---|---|---|---|---|---|
| sharegpt (W1) | 5000 | 982 | 2746 | 3206 | 3698 | **3977** | 0% |
| cnndm (W2) | 3000 | 780 | 1531 | 1796 | 2152 | **3104** | 0% |
| alpaca (W3) | 50154 | 14 | 29 | 38 | 70 | 518 | 0% |

**Critical constraint**: **no existing dataset has any prompt ≥ 4k tokens**. Longest prompt in entire corpus is 3977 tokens (sharegpt). To build W5_LongDoc we must synthesize.

**run.py fault injection** (line 1642-1665):
- Single `--fault` arg → single `asyncio.sleep(fault_time_sec)` then single `os.kill()`
- No loop. Multi-fault requires refactor + server config change (`max_gpu_failures > 1`).
- **W6_Burst is NOT feasible** without non-trivial code changes. **Dropped from Phase 1 scope.**

**GPU availability**: GPU 0-3 free (24GB each), GPU 4-7 occupied by other user.

---

### P0-5 ArXiv data characterization (done)

Downloaded `ccdv/arxiv-summarization` test split, filtered to `prompt_tokens ∈ [4000, 7500−output]`, kept 2000 records:

| metric | value |
|---|---|
| prompt p50 / p95 / max | 5500 / 7106 / **7383** |
| output p50 / p95 / max | 186 / 321 / 538 |
| prompt ≥ 4k (all of them) | 100% |

Every ArXiv prompt is longer than every cnndm prompt. Pressure ×4 on prefill confirmed.

### P0-6 W5 smoke FAILED — preemption bug triggered

First smoke (NR W5, Heavy, fault=none, GPU 0-1) crashed at request 324 with:
```
AssertionError at vllm/v1/worker/gpu_model_runner.py:1106
(assert new_block_ids is not None)
```

**Root cause**:
1. ArXiv avg 5.5k prompts × ~7 concurrent requests = ~40k KV tokens / 44k budget → preemption triggers
2. Preempted request resumed without new block allocation (scheduler.py:1205 `allow_none=True`)
3. Model runner asserts non-None → engine dies
4. FT client sees `ENGINE_CORE_DEAD` → triggers false failover of 78 requests

**Not a bug in our FT code** — upstream vLLM preemption + chunked_prefill + long-context interaction. Our FT scheduler wraps around this correctly, but the underlying vLLM path is broken.

**Mitigations (cheapest first)**:
- **M1**: filter ArXiv to `prompt_tokens ≤ 5500` (keep ~50% of data, still 3× W2 p95) — avoids KV pressure entirely
- **M2**: lower RPS 1.5 → 1.0 for W5 (fewer concurrent requests)
- **M3**: cap `max_num_seqs=8` (hard concurrency limit, requires server arg change)
- **M4**: fix upstream vLLM scheduler bug (2-3h work, risky)

**Decision**: apply M1 + M2 → build `arxiv_filtered_1000.jsonl` with prompts ≤ 5500 tokens, run W5 at RPS=1.0. This preserves the long-context signal (avg 5k prompts is still 6× cnndm p50 780) while sidestepping the preemption path. If smoke passes, then try bumping RPS back up.

---

## Phase 0 decisions

1. **W5_LongDoc — synthesize from cnndm** (no external dataset download):
   - Concatenate 4-6 cnndm articles with multi-doc summarization prefix: *"Summarize the following articles into a single cohesive summary:\n\n## Article 1\n...\n\n## Article 2\n..."*
   - Target: prompt_tokens ∈ [5500, 7000] (fits within 8192 − 1024 gen budget, no `max_model_len` change)
   - If initial 7k results look promising → raise `max_model_len` to 12288 in follow-up for stronger signal
   - max_new_tokens = 512 (shorter summaries for multi-doc)

2. **W7_Saturated — reuse W1 (sharegpt) with RPS push**:
   - Prompts unchanged (sharegpt short-to-medium)
   - RPS starts at 3.0 (2× Heavy). Iterate up until saturation signal appears.
   - Tighten SLO: `ttft_ms = 1000`, `failure_gap_ms = 1500`
   - If saturation doesn't hit at RPS=3, escalate to 4-5 in smoke

3. **W6_Burst — DROPPED** (server-side change risk outweighs value at this stage)

4. **P0-4 saturation sweep DEFERRED** — conservatively start W7 at RPS=3.0 (~2× current Heavy). If goodput doesn't drop, bump up in a second smoke run. Saves ~30min of sweep experiments.

---

---

## Phase 1 — Candidate workload design (3 options, run 2)

### ⭐ W5_LongDoc — Long-context pressure (HIGH PRIORITY)

- **Source**: CNN/DailyMail top-20% longest articles (reuse W2 pipeline, filter by token count) + GovReport / LongBench if available locally
- **Target prompt length**: p50 = 8k, p95 = 14k (×7 current W2 p95)
- **max_new_tokens**: 512 (summarization)
- **RPS**: 1.5 (match W2 for comparability)
- **Fault**: F2_Mid
- **Activates**: long context → NR's reprefill cost scales linearly with prompt length; OS's KV reload stays O(1) in prompt → cost ratio flips

### W7_Saturated — Queue saturation pressure (HIGH PRIORITY)

- **Source**: W1 prompts (short, identical to existing W1_Chat)
- **RPS**: saturation knee × 0.9 (determined in P0-4, expected 3.5–4.5)
- **SLO**: tightened — TTFT = 800ms, fg = 1500ms (vs current 2000/3000)
- **Fault**: F2_Mid
- **Activates**: queue pressure → NR reprefill occupies prefill slots → queue backs up → goodput drop; OS's admission gate protects flow

### W6_Burst — Multi-fault cascade (OPTIONAL, requires run.py change)

- **Mechanism**: modify `run.py` to accept `--fault "F1_Early,F2_Mid,F3_Late"` (3 faults in one run)
- **Source**: W2 (long-prompt amplifies reroute decision space)
- **Risk**: run.py change introduces bugs; defer unless W5+W7 are conclusive
- **Activates**: solver reuses snapshot state across faults; NR pays full reprefill each time

**Recommendation**: run W5 + W7 first. W6 is stretch goal contingent on those results.

---

## Phase 2 — Dataset construction

- [x] **P2-1** Dataset source: `ccdv/arxiv-summarization` (HuggingFace, widely used in DistServe/Sarathi-Serve)
- [x] **P2-2** Download + filter → `arxiv_2000.jsonl` (prompt ∈ [4001, 7383])
- [x] **P2-3** Added W5_LongDoc to `config_8b.yaml`
- [ ] **P2-3b** **NEW**: Build `arxiv_filtered_1000.jsonl` with prompts ≤ 5500 tokens (P0-6 preemption mitigation)
- [ ] **P2-4** Add W7_Saturated workload block (reuses W1 data, adjusts RPS/SLO)

---

## Phase 3 — Smoke verification (1.5 h)

Per Code-Change Protocol (single seed, monitor server.log for errors):

- [ ] **P3-1a** W5 NR baseline smoke on **filtered** dataset at RPS=1.0 (s42) — first attempt after P0-6 mitigation
- [ ] **P3-1b** If P3-1a passes, bump RPS to 1.5 and retry
- [ ] **P3-2** W5 V2 smoke
- [ ] **P3-3** W7 NR + V2 smoke
- [ ] **Gate**: repeated failure → further reduce RPS or cap `max_num_seqs=8`

---

## Phase 4 — Full 3-seed comparison (~3-4 h)

**W5_LongDoc × {NR, V2, V2+skip_solver} × 3 seeds × F2_Mid** = 9 runs
- Rationale for `skip_solver` variant: long context may shift solver cost/value ratio

**W7_Saturated × {NR, V2} × 3 seeds × F2_Mid** = 6 runs

Total: 15 runs, ~4 h at 10–15 min/run depending on context length.

---

## Phase 5 — Verdict (1 h)

- [ ] **P5-1** Write `experiments_v2/docs/stress_test_2026-04-20.md` with tables
- [ ] **P5-2** Generate 3 comparison plots: fg_p95 / goodput / completion vs workload
- [ ] **P5-3** Decision tree:

| Outcome | Interpretation | Action |
|---|---|---|
| OS beats NR on any axis (fg_p95 / goodput / comp) | Algorithm has regime where it wins | Paper story validated; plan 70B scale-up |
| OS ≈ NR under stress | Algorithm at least doesn't regress under pressure | Reframe paper as mechanism contribution (design space, admission control tradeoffs) |
| OS < NR on all axes even under stress | Algorithm regime fundamentally limited in 8B dp=2 | Scale-up (70B) OR pivot to different framing (fault-tolerance mechanism study) |

---

## Time budget

| Phase | Duration |
|---|---|
| P0 reconnaissance | 0.5 h |
| P1 design | 0.5 h |
| P2 dataset build | 1.0 h |
| P3 smoke | 1.5 h |
| P4 full runs | 3.5 h |
| P5 analysis | 1.0 h |
| **Total** | **~8 h** |

Fits a single daytime session.

---

## Execution protocol

- Only use GPU 0-3 (CUDA_VISIBLE_DEVICES=0,1 or 2,3)
- Code-change protocol before any code goes live: syntax check → import test → smoke → grep server.log for errors
- Background runs via `nohup`; monitor via `ScheduleWakeup`
- Results root: `results_v2/8B/stress_2026-04-20/`

## Files

- Plan: `experiments_v2/docs/stress_test_plan_2026-04-20.md` (this doc)
- Reference: `experiments_v2/docs/overnight_2026-04-19_summary.md` (P5 ablation results)
- **Unified pipeline**: `experiments_v2/stress_pipeline_2026-04-20.sh` ← auto-gated execution
- Dataset (filtered): `experiments_v2/datasets/cached/arxiv_filtered.jsonl` (1002 records, prompt ≤ 5500)
- Pipeline log: `/tmp/stress_pipeline.log`, W7 side log: `/tmp/stress_w7.log`

## Unified execution plan (launched 2026-04-20 ~12:45)

The pipeline `stress_pipeline_2026-04-20.sh` auto-runs the full matrix with debug/recovery logic:

### Phase A — Smoke gate (RPS fallback ladder)
1. Wait up to 15min for in-progress W5 NR @ Moderate (RPS=1.0) smoke
2. If it succeeds (completion ≥ 80%, goodput > 0) → MAIN_LOAD = Moderate
3. Else retry at Light (RPS=0.5) → MAIN_LOAD = Light if that passes
4. Else skip W5 entirely, run W7 only

### Phase B — W7 on GPU 2-3 (parallel, independent)
- W7_Saturated (sharegpt @ Heavy RPS=1.5) × {NR, V2} × {42, 123, 456} × F2_Mid = 6 runs
- Independent of W5, runs in background

### Phase C — W5 on GPU 0-1 (serial, gated on smoke)
- W5_LongDoc (arxiv filtered) × {NR, V2, V2+skip_solver} × {42, 123, 456} × F2_Mid = 9 runs
- Uses MAIN_LOAD determined in Phase A

### Phase D — Auto summary table
- Prints 3-seed means for each variant, filters comp ≥ 90% for gp/fg_p95

### Failure handling
- Each cell failure is logged but does NOT stop the pipeline (`|| true`)
- `metrics.json` acts as checkpoint — existing cells skipped on re-run
- Failed cells annotated with cause (preemption/OOM) if detectable

## Open design questions

1. **Long-doc source**: is CNN/DM long tail enough, or do we need GovReport/LongBench? (resolved in P0)
2. **W6 fault-burst feasibility**: does `run.py` fault engine selection cleanly support multiple sequential faults? (resolved in P0-2)
3. **Saturation knee**: likely RPS ~4-5 for 8B dp=2 on A5000, but needs measurement (P0-4)
4. **Ablation scope**: should we also run V2+no_prebudget on W5? defer until W5 primary results are in.
