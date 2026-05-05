# Current Status & Pending Work (2026-04-29)

This doc preserves the working context across compressed Claude sessions.

## What's Running Right Now

### A6000 (local workstation) — overnight matrix
- task id: bsfgdarui (tight_slo_5s, finished) + bjwm2m2gm (load_slo_matrix, in progress)
- script: `/tmp/run_overnight_load_slo_matrix.sh`
- main log: `/tmp/v2full_load_slo_matrix_main.log`
- Three groups, each 9 cells (3 baselines × 3 seeds), W8 LongMix, F2_Mid:
  - Group A: Moderate(0.4 RPS) + SLO 5s/5s   → `/tmp/config_mod_5s.yaml`
  - Group B: Moderate(0.4 RPS) + SLO 2s/2s   → `/tmp/config_mod_2s.yaml`
  - Group C: Light(0.2 RPS)    + SLO 5s/5s   → `/tmp/config_light_5s.yaml`
- Output: `/tmp/v2full_load_slo_matrix/{A,B,C}_*/`
- Expected to finish: ~09:00 morning

### L40S (planned, not started)
- User has 4 L40S machines (each 2 GPUs), 10 Gbps network
- One machine has been used before, has venv probably
- TODO: ssh in, run `profile_decode_capacity.py` (needs `decode_capacity_profile_8b_l40s.json`, missing)
- Already exists: `checkpoint_cost_profile_l40s.json` (4/26)
- Already exists: `config_8b_step5_l40s.yaml` (config variant)

## Key Findings (DO NOT FORGET)

### 1. Async checkpoint stream separation is the ONE big win (commit `e6b108a0a`, 4/26)
- Figure: `experiments_v2/figures/ckpt_async_fix/stream_timeline_en.png`
- TPOT p95 1219ms → 189ms (-85%)
- SLO violation 63% → 3.9%
- Goodput 2.5 → 4.58 (with prefix cache; ~3.10 with prefix cache disabled)

### 2. Failover gap p95 is the ONLY robust signal we have
- Our-System 10.6s vs NoFT-Reprefill 16.6s on tight SLO (5s/5s, seed=42)
- **-36%** advantage on FT recovery time
- Goodput differences are within noise (std ~0.3-0.5 on mean ~3.0)

### 3. NO_GIL=1 is a TRAP — never enable it
- Tested 4/28 ckpt_flags_sweep
- Sec1 step time +33% (231ms → 308ms)
- Recovery rate dropped from 89% to 54%
- Reason: under UniProcExecutor (TP=1), stage 2 GPU sync directly blocks main thread

### 4. Profile re-cal didn't fix Benders
- New A6000 profile: decode_capacity 26 → 50
- Benders fallback rate: 99.5% → 98% (no real change)
- Root cause: PoolOverload constraint in `recovery_checker.py:144-153` triggers
  regardless of profile values; constraint logic, not parameter calibration
- At dp=2, Benders mostly degenerates to greedy fallback

### 5. Section 2 timing measurement was misleading
- "13ms checkpoint overhead" earlier finding was a Section 2 quirk
- Total step time (Section 1) actually unchanged across BG_PUBLISH / GPU_OVERLAP / combo
- Time just MOVES between sections, doesn't get saved

### 6. BG_PUBLISH and GPU_OVERLAP both within noise
- BG_PUBLISH=1: goodput same (2.89 → 2.89 mean across 3 seeds)
- GPU_OVERLAP=1: goodput in noise
- The /dev/shm tmpfs file IO is already fast (memory write), nothing to async-ify

### 7. Tight SLO + Heavy load = bad for everyone
- 4/29 experiment: TTFT p50 11-17s for all methods at 5s SLO
- System is OVERLOADED at Heavy 0.6 RPS, no method can meet SLO
- TTFT violations dominate, FT mechanism contribution invisible
- Need Moderate or Light load for clean signal (overnight matrix testing this)

## Uncommitted Changes (not in git)

- `vllm/v1/engine/ft_client.py`: 3 lines added (BG_PUBLISH lag TODO comment around line 999-1004)
- `experiments_v2/docs/course_proposal.tex`: untracked
- `experiments_v2/docs/advisor_summary_2026_04_29.md`: untracked
- `experiments_v2/docs/CURRENT_STATUS_2026_04_29.md`: this file, untracked
- `experiments_v2/figures/ckpt_async_fix/stream_timeline_en.{png,pdf}`: untracked
- `experiments_v2/decode_capacity_profile_8b_a6000.json`: untracked
- `experiments_v2/checkpoint_cost_profile_8b_a6000.json`: untracked

Last committed state: `347a31dcc profile and close prefix reuse` (4/28 17:59)

## Open Questions / Pending Decisions

### High priority
1. **Does Moderate load show clean Our-System advantage?** (overnight matrix, ~09:00)
2. **dp>2 testing**: code untested. Need local L40S smoke before renting cloud machines.
3. **Workload swap**: if all (load, SLO) combos fail to show signal, try RULER 32K-128K (longer prompts).
4. **Metric reframe**: should paper lead with failover_gap_p95 instead of goodput? Defensible
   but needs careful framing.

### Medium priority
5. **L40S decode_capacity profile**: missing, should run on idle L40S machine
6. **Cross-host NCCL setup**: test 4 L40S machines can communicate (ping + small distributed test)
7. **Deployment scripts**: rsync repo, install vllm venv, copy datasets to N machines

### Low priority
8. **Promise vs delivered checkpoint lag**: known issue at `ft_client.py:999-1004` (TODO comment).
   `_checkpoint_tokens` updates before /dev/shm publish completes. Cleaner fix is to read
   /dev/shm latest manifest as ground truth.
9. **Greedy fallback dominates Benders 99%**: known via FT_SOLVER_DIAG. Root cause is
   PoolOverload constraint in `recovery_checker.py`. May not be worth fixing if dp=2 is
   inherently trivial for Benders.

## Risk Assessment

1. **Effect size below noise floor**: std ~15%, baseline diffs 5-15%. Need either bigger
   effects (workload change) or many more seeds.
2. **dp scaling is hypothesis only**: not measured. If dp=4 still shows our system within
   noise, this avenue closes.
3. **Workload may be wrong dimension**: 16K prompt + loose 12s SLO is too easy for reprefill.
4. **Goodput aggregates obscure FT signal**: TTFT/TPOT under overload dominate.

## Metric Framing (added evening 2026-04-29)

- **TTFT is orthogonal to our work**. Don't lead with it. TTFT = admission/scheduling territory.
- **TPOT is the async fix story, not FT mechanism**. Fix brings Our-System DOWN to vanilla (vanilla never had the spike). Mention once as prerequisite.
- **Failover_gap is the only direct FT signal**. Lead with this.
- **Failover_gap leaks into TPOT mean** for fault-affected requests (single fault adds ~10-15ms to that request's TPOT mean). But aggregate TPOT doesn't show it because fault-affected reqs are only 7%. Action: split TPOT by fault-affected vs unaffected, parse from requests.csv.
- **Set SLOs asymmetric**: TTFT loose (12s), gap tight (2s), TPOT moderate. Don't tighten TTFT and gap together — they measure different things.

## Paper framing pivot: RAG-style long-context FT serving (decided 2026-04-29 afternoon)

Switched away from generic LongBench framing toward RAG long-context serving.

Why this is a better story:
- RAG explains WHY prompts are long (retrieved docs + query) without scope creep into actual retrieval
- Adds an "upstream sunk cost" angle: a fault doesn't just lose prefill compute, it forfeits the retrieval pipeline's output (vector search + rerank + assembly). Reprefill alone is wasteful; reprefill + re-retrieve is worse.
- Cleaner than agent framing: agent has tool calls / multi-step state that we don't model. RAG = "after retrieval, KV cache IS the entire serving state" — perfectly aligned with our work.

Paper pitch (one paragraph):
> "Long-context RAG serving has an asymmetric recovery cost: a failure not only forfeits the prefill compute but also invalidates the entire retrieval pipeline's output. Re-prefill alone takes 20-30 s on 64K prompts; combined with retrieval, recovery time can dominate end-to-end latency. KV-checkpoint-based recovery sidesteps both costs by restoring the assembled prompt's KV state directly."

Implication for evaluation:
- RULER 64K (and eventually 128K on L40S) is the right stand-in workload
- Goodput de-emphasized; failover_gap_p95 is THE metric
- Don't need to defend "why 64K" — RAG explains it
- Don't need to implement retrieval — out of scope

## RULER 64K integration (done 2026-04-29 afternoon)

Status: integrated, boot test passed, ready for real experiments.

What was done:
- Cloned RULER to `/tmp/RULER`. Generated 200 niah_single_1 samples at max_seq_length=65536 using `prepare.py` with hf tokenizer (Llama-3.1-8B-Instruct). pip installed nltk + wonderwords + html2text.
- Wrote `experiments_v2/datasets/convert_ruler.py` to convert RULER's `{index, input, outputs}` into our loader's `{id, dataset, prompt, prompt_tokens, expected_output_tokens, subtask}` format.
- Final dataset: `experiments_v2/datasets/cached/ruler_64k_niah.jsonl` (200 records, 48MB, prompt_tokens P50=65389, P95=65393).
- expected_output_tokens=64 (RULER niah answer is one number ~7 tokens; 128 was too aggressive and pushed prompt+output > max_model_len).
- Added `W9_Ruler64K` to `experiments_v2/config_8b_step5_overnight.yaml`.
- Created `experiments_v2/config_8b_ruler64k.yaml`: max_model_len=65536, gpu_memory_utilization=0.80, checkpoint_pool_bytes=64GB, RPS lowered (Light/Mod/Heavy = 0.05/0.10/0.15), TTFT SLO 30s, failover_gap SLO 5s.

Boot test on A6000 (No-FT, W9_Ruler64K, Light, no fault, 240s):
- 11/11 requests completed
- Clean prefill TTFT ≈ 22s (when no concurrent prefill). This IS the reprefill cost — 10x larger than LongBench 16K's 1.6s.
- Queued TTFT 40-64s (when 2 reqs prefilling concurrently)
- Clean TPOT ≈ 34ms; congested TPOT ≈ 686ms (chunked prefill + decode mixed in same step)
- TPOT 200ms SLO is unrealistic at 64K — even No-FT vanilla can't meet it. Loosen to 1000-1500ms or de-prioritize TPOT.

Bug fixes encountered:
- `aiohttp.ClientSession` default `read_bufsize=64KB` chokes on long-prompt SSE chunks (vLLM streams prompt token_ids in early frames when `return_token_ids=True`). Bumped to 4MB at three call sites in `experiments_v2/run.py`.
- RULER's `--max_seq_length` budget includes generation tokens; reducing `expected_output_tokens` from 128 to 64 keeps prompt+output under max_model_len.

Open items for next experiment:
- Loosen TPOT SLO to ~1000ms in W9 config
- Run No-FT vs Our-System vs NoFT-Reprefill × F2_Mid × 3 seeds at W9_Ruler64K. Hypothesis: failover_gap p95 ≈ 22s (reprefill) vs ≈ 2s (KV reload) → 10x advantage that holds across seeds.

## CRITICAL FINDINGS (2026-04-29 evening) — checkpoint mechanism is NOT firing at 64K

### Symptom
First 9-cell run at W9_Ruler64K produced:
- Cell 5 (Our-System seed=123) crashed mid-run with `OSError: No space left on device` on /dev/shm
- Completed cells (seed=42 + No-FT seed=123) showed: Our-System failover_gap_p95 = 41925ms, NoFT-Reprefill = 38637ms — Our-System NOT winning, possibly slightly worse (3s).

### Investigation chain

**1. /dev/shm cleanup gap (engineering bug, not paper-relevant)**
- `_cleanup_shared_checkpoint_request` in `vllm/v1/engine/ft_client.py:202` only fires when client receives `finished_requests` event
- When server is SIGTERM'd at end of cell, in-flight reqs leave their /dev/shm files behind
- 4 cells worth of leakage filled tmpfs (498GB)
- Not a real bug in production (long-running server, per-req cleanup fires) but breaks experimental cell-cycling
- Fix: either shell-script `rm -rf /dev/shm/vllm_ft_checkpoints/` between cells, or paranoid wipe in ft_client startup

**2. Instrumentation revealed 1 fire per request (real problem)**
- Added `_ft_ckpt_inst` counter dict to `_maybe_ft_checkpoint` in `vllm/v1/engine/core.py` (perf-neutral: int adds + dict updates only). Logs `FT_CKPT_INST FINAL fires=N blocks=M` at engine shutdown.
- Test cell (Our-System / no fault / 600s / 34 reqs): **DP0 fired 29 times across 29 reqs = 1.0 fire/req. Avg blocks/fire = 45 = 720 tokens.**
- 64K prompt = 4096 blocks. Coverage = 720/65000 = **1.7% of KV checkpointed per request.**
- This means: on fault, Our-System has ~1.7% of state to restore, has to reprefill 98% anyway. That's why failover_gap looks identical to NoFT-Reprefill.
- **Paper claims about 10x advantage have no empirical basis until this is fixed.**

**3. Root cause: profile calibration**
- `experiments_v2/checkpoint_cost_profile_8b_a6000.json` only measures up to **2048 tokens prefill** and **128 MB load/ckpt bytes**
- We run **65536 tokens (32x larger) and 8 GB KV (64x larger)**
- The cost model's `should_publish` formula uses linear interpolation/extrapolation of `t_prefill`. Linear extrapolation from short-context profile dramatically underestimates 64K prefill cost (real ~22s; extrapolated ~1.3s).
- → economic policy says "replay would save almost nothing, don't bother checkpointing" → fires only once per request.

**4. Profile re-measurement attempt also broken**
- Modified `experiments_v2/profile_checkpoint_costs.py` to extend `token_lengths` to [4096, 8192, 16384, 32768, 65536] and `block_counts` to cover 8 GB.
- Bumped `_start_server` defaults: `max_model_len 4096 → 65536`, `gpu_memory_utilization 0.45 → 0.80`.
- Re-ran. Got measurements but **values are 100x off from production**:
  - Profile: 32K prefill = 106ms
  - Boot test (production conditions, dp=2 chunked prefill + concurrent decode): 64K prefill = 22000ms
  - Implied throughput: profile shows ~309K tok/s — physically impossible for 8B model on A6000 (peak ~3-9K tok/s)
- Suspected causes (NOT YET CONFIRMED, debug tomorrow):
  - **Prefix caching not disabled in profile server** — likely culprit; same prompt content reused across trials would hit cache
  - Profile single-request isolated load doesn't trigger chunked-prefill scheduling that production uses
  - Profile's `_measure_prefill` uses `max_tokens=1`; if vLLM treats this as instant decode-skip somehow, prefill timing is masked
- The current `checkpoint_cost_profile_8b_a6000.json` ON DISK is now a partial overwrite (failed mid-run after 2048 token point). **Old short-coverage profile is backed up at `checkpoint_cost_profile_8b_a6000.json.bak.short`.**

### Action plan for next session

**Priority 1: Fix profile measurement methodology**
1. In `_start_server`: add `--no-enable-prefix-caching` to match production config
2. Verify `max_num_batched_tokens` matches production (2048 in our experiments)
3. Sanity check by measuring 64K prefill, expect ~22s ± 30%, NOT 100ms
4. If still wrong, dump server.log during one measurement to see what vLLM is actually doing
5. Consider: profile might need to run with concurrent decode background traffic to match real conditions. If so, that's a more invasive rewrite.

**Priority 2: Restore correct profile data**
- After fix, re-run profile generation
- Verify on disk values: e.g., `prefill_ms_by_tokens["65536"]` should be ~20000-25000 (not 106 not 1300)
- If profile script can't be made to match production, fall back to:
  - Manually patch JSON with 5-6 measurements taken from a real production-config dp=2 run
  - Or: derive t_prefill curve analytically from peak throughput × number of chunks × per-chunk overhead

**Priority 3: Verify mechanism actually fires after profile fix**
- Run ONE Our-System / no-fault sanity cell with corrected profile
- Grep `FT_CKPT_INST FINAL` from server.log
- Expected: ~30+ fires per request (close to 32 prefill chunks + 4 decode block boundaries)
- If still 1 fire/req: profile fix didn't work, need deeper investigation

**Priority 4: Once mechanism confirmed firing**
- Fix /dev/shm cleanup (option A: shell script `rm -rf` per cell)
- Run the 9 cell main experiment (3 baselines × 3 seeds, F2_Mid only)
- Expected: Our-System failover_gap_p95 << NoFT-Reprefill failover_gap_p95 (real advantage, vs the 1-seed seed=42 result that was within noise)

### Cross-machine / cross-session continuity (when renting cloud)
This same calibration bug will appear on rented machines. The fix is per-hardware:
- Each physical setup (A6000, L40S, H100, etc.) needs its own profile JSON
- Profile must be calibrated under the same conditions as actual experiments (chunked prefill, no prefix cache, expected concurrency)
- Always verify after profiling: pick ONE long-context value (e.g., 64K), run a single production-config request, compare measurement to profile. Should match within ~30%.

### Files affected by today's investigation
- `experiments_v2/profile_checkpoint_costs.py` — modified `_start_server` defaults (max_model_len, gpu_memory_utilization), modified hardcoded token_lengths and block_counts. **Still has the methodology bug** (no --no-enable-prefix-caching).
- `experiments_v2/checkpoint_cost_profile_8b_a6000.json` — overwritten with bad short-coverage data partway through. Old version saved at `.bak.short`.
- `vllm/v1/engine/core.py` — added `_ft_ckpt_inst` instrumentation in `_maybe_ft_checkpoint` and `shutdown()`. Perf-neutral. Keep this in for ongoing visibility.
- `experiments_v2/run.py` — added `read_bufsize=4MB` to 3 ClientSession sites (long-prompt SSE buffer fix). Production-safe; keep.
- `experiments_v2/config_8b_ruler64k.yaml` — TPOT SLO 200→400, TTFT 30s→40s, F2_Mid 350→500, fixed_blocks 0 (Our-System uses economic policy → broken at 64K until profile fixed).

## Sanity comparison after profile fix (2026-04-29 evening, 600s no-fault cells)

After re-profiling (prefill correct to 22s @ 64K, c0 = 1.23ms measured):

| Cell | Baseline | c0 | Fires/req | Completion | TTFT p95 | TPOT p95 |
|---|---|---|---|---|---|---|
| A | Our-System | 1.23 (default) | 8.7 | 73.5% | 245s | 712ms |
| B | Our-System | 400 (manual policy threshold) | 8.6 | 73.5% | 245s | 713ms |
| C | No-FT (vanilla, no checkpointing at all) | n/a | 0 | 76.5% | 242s | 689ms |

**Three observations:**

1. **Profile fix worked**: fires/req went from 1 → 8.7 (8x improvement). Mechanism is now actually firing during 64K prefill, no longer silently inactive.

2. **c0 tuning had no effect**: A vs B identical despite c0 jumping 300x. The actual rate-limiter is `FT_CKPT_NONBLOCK=1` back-pressure (only one async DMA in flight at a time). Each fire's GPU→host copy of one chunk's KV (256MB) takes ~50-100ms; decode steps are 30ms; back-pressure naturally caps fires at ~1 per 4 chunks. economic policy could say "fire every chunk" but back-pressure ignores it. **c0 only matters when fire rate is below the back-pressure ceiling**, which we never hit at 64K.

3. **Vanilla (no checkpointing) hits the same wall**: TTFT p95 = 242s with No-FT. That's 240+ seconds of queue wait without any FT machinery. **The bottleneck is the workload itself, not Our-System overhead.** Our-System adds ~3% completion-rate cost vs vanilla (76.5 → 73.5%); rest of the gap is hardware capacity.

### Why A6000 dp=2 cannot sustain 0.05 RPS at 64K

- Single 64K request prefill on dp=2: 22s clean (boot test).
- Capacity: 1/22 × 2 = 0.09 RPS theoretical.
- Operating at 0.05 / 0.09 = 56% utilization SHOULD be stable.
- But Poisson arrivals + chunked-prefill + decode interleaving create instability bursts that grow the queue past timeout.
- With Our-System overhead, effective capacity drops to ~0.04 → RPS 0.05 exceeds it → queue blows up.
- Even No-FT vanilla shows same pattern → A6000 dp=2 is the hardware limit, not the FT mechanism.

### Decision: stop running 64K experiments on A6000

- Lower RPS (0.02) would stabilize but cuts in-flight reqs to ~1, making fault sample size statistically meaningless.
- L40S × 4 dp=4 is the right hardware: dp=4 doubles capacity (capacity ≈ 0.18 RPS), allows RPS 0.05-0.10 with comfortable headroom.

### Tonight's deliverable for L40S handoff

Cloud rental runbook at `experiments_v2/docs/CLOUD_RENTAL_RUNBOOK.md` covers:
- Required code patches (run.py read_bufsize, core.py instrumentation, profile script defaults)
- Required dataset transfer
- Per-hardware profile re-measurement procedure (with sanity check that prevents tonight's bug)
- Config calibration steps (boot test → SLOs → load levels → fault timing)
- Smoke test that catches under-firing before paying for full experiment
- 11-line pre-experiment checklist

Cross-machine state to transfer (in priority order):
1. Code at branch `zoe/slo-scheduling` with all today's patches
2. RULER 64K dataset (`experiments_v2/datasets/cached/ruler_64k_niah.jsonl`)
3. The corrected `experiments_v2/checkpoint_cost_profile_8b_a6000.json` (as reference for what GOOD numbers look like; you'll generate a new one for L40S)
4. Backup of original profile at `*.json.bak.short` (for diff/comparison)

### Net status after today

- ❌ A6000 64K full experiment: not viable, abandon
- ✅ Profile bug found, root-caused, partially fixed (need L40S re-profile per Phase 2 of runbook)
- ✅ Instrumentation in place for cross-session diagnostic continuity (`FT_CKPT_INST FINAL`)
- ✅ Comprehensive runbook for cloud rental
- ✅ vanilla baseline data confirms hardware limit, not FT-specific issue
- ⏸ Main 9-cell experiment: pending L40S setup

## NEW PAPER DIRECTION: SLO-aware scheduling on top of KV checkpoint primitive (2026-04-29 night)

User raised: after fault, current scheduler treats recovery requests as ordinary new arrivals at the FCFS queue tail. Surviving engine continues its existing batch, recovery reqs wait. At high post-fault utilization, recovery's mechanical advantage (1-2s KV restore vs. 22s reprefill) is buried under queue wait time.

The full re-framing: **after fault, scheduler should re-evaluate ALL pending requests (existing + recovered) by remaining SLO budget (EDF-style), preempting in-flight requests as needed.** KV checkpoints already maintained for fault tolerance enable cheap preemption — the existing in-flight req's state is already on host memory.

This generalizes beyond fault scenarios: any over-budget request can preempt under-budget ones. The checkpoint mechanism becomes a dual-purpose primitive: fault recovery (its original purpose) AND SLO-aware preemptive scheduling.

### Why this is a real contribution (not just engineering)

- KV checkpointing alone has weak demo at moderate utilization (mechanism vs reprefill ~10x but only when surviving engine has spare capacity).
- SLO-aware scheduling alone CAN'T preempt long-prefill in-flight reqs in vanilla systems — the state would be lost on preemption (no checkpoint to resume from).
- **Together**: FT machinery makes preemption cheap; SLO scheduling makes preemption useful. Each piece alone is weak, the synergy is the contribution.

### Mechanism, concretely

Per scheduling step:
```
1. Collect: in-flight reqs on this engine + pending reqs (incl. recovery) + queued
2. Compute per-req remaining SLO budget:
   budget_TTFT = ttft_slo - (now - arrival_time) for unstarted reqs
   budget_TPOT = tpot_slo - last_decode_step_time for decoding reqs
   budget_gap  = failure_gap_slo - (now - fault_time) for recovery reqs
3. Sort by min budget (most urgent first)
4. Preempt currently-executing req if its budget is much looser than top pending:
   - Wait for current chunk/decode step to complete (no mid-step kill)
   - Save its state (already checkpointed; no extra work in steady state)
   - Schedule the urgent req in next step
5. Resume preempted reqs when their budget tightens
```

### Implementation effort

- ~50 lines: replace FCFS sort in scheduler with budget-aware sort
- ~30 lines: integrate "force checkpoint at preempt time" if checkpoint isn't fresh
- ~20 lines: hook fault event to trigger immediate reschedule
- **~100 lines total + ~1-2 days debug. Doable.**

### Evaluation needed

- Compare 4 conditions on the same fault scenario:
  1. No-FT (drops failed reqs)
  2. NoFT-Reprefill (FCFS, no checkpoint)
  3. Our-System with FCFS (current)
  4. Our-System with SLO-aware scheduling (new)
- Expected: condition 4 cuts failover_gap_p95 from "queue + restore" to "preempt + restore", ~5-10x improvement over condition 3 at moderate utilization.

### Risk inventory (real, must mitigate)

1. **Budget calculation accuracy**: estimating remaining work needs request-state knowledge. Tractable but must be precise.
2. **Preemption cost vs benefit**: each preempt costs 0-3s (loss since last checkpoint + restore on resume). Gating: don't preempt unless gain >> cost (e.g., only when over-budget by ≥3s).
3. **Starvation**: persistent low-budget reqs could starve high-budget regulars. Mitigation: budgets tighten over time, low-budget reqs become high-budget eventually.
4. **Oscillation**: similar-budget reqs preempting each other. Mitigation: hysteresis (X% margin to switch).
5. **Per-step overhead**: ~0.5-1ms (sort + budget calc) → ~1-3% TPOT impact.
6. **Adversarial workloads**: if all reqs have similar deadlines, FCFS is optimal; SLO-aware adds overhead with no benefit. Need to characterize when it pays off.

### Paper framing (the angle)

Don't lead with "we added an SLO scheduler". Lead with the synergy:

> "Long-context fault-tolerant serving has an under-explored optimization: the host-memory KV state maintained for fault recovery is also a preemption primitive. Vanilla SLO-aware schedulers cannot preempt mid-prefill long-context requests because their state is too expensive to lose. We show that the same KV checkpointing that bounds failover_gap also enables preemption-based SLO-aware scheduling — making FT machinery a dual-purpose primitive at no extra steady-state cost."

This positions our work as discovering a NEW use for an existing primitive, which is harder for reviewers to dismiss as "just engineering".

**Full paper narrative (Sections 1-5):** see `~/.claude/projects/.../memory/project_paper_framing_rag.md`. Includes section-by-section structure, ablation table for evaluation, abstract-ready takeaway paragraph.

## M3 implementation done (2026-04-30 night)

SLO-aware preemption code is committed to the same branch (`zoe/slo-scheduling`) as uncommitted changes on top of `083bb3269 long context profile,ruler dataset of 64k`.

### Files modified (post-commit, uncommitted)

| File | Change | Purpose |
|---|---|---|
| `vllm/v1/core/sched/utils.py` | Added `compute_slo_budgets(req, now) -> dict` | Per-req SLO budget math (TTFT/TPOT/gap), used by M1 telemetry and M3 trigger |
| `vllm/v1/core/sched/scheduler.py` | M1 telemetry block (FT_SLO_BUDGET_LOG=1) + M3 trigger (FT_SLO_PREEMPT=1, calls `_pick_slo_preempt_victim` + `_preempt_for_slo`) | Per-step budget log + SLO-aware preempt decision and execution |
| `vllm/v1/engine/core.py` | `_drain_slo_preempted_restores` migrates scheduler-side pending into `_ft_pending_restores` | Wires same-engine SLO preempt resume through existing FT recovery restore path |

### M3 design summary

`_pick_slo_preempt_victim(now)` returns the running req to preempt, or None. Four guards:
1. Global rate limit (FT_SLO_PREEMPT_MIN_INTERVAL_MS, default 2000ms)
2. Waiting head must be is_rerouted=True (failure_gap_slo recovery — only legitimate priority signal at saturation)
3. Per-req cooldown (FT_SLO_PREEMPT_PER_REQ_COOLDOWN_MS, default 30000ms) — same req can't be preempted twice within cooldown
4. Budget gap > FT_SLO_PREEMPT_MIN_GAP_MS (3000ms) AND relative > FT_SLO_PREEMPT_HYSTERESIS (0.30)

`_preempt_for_slo(request, timestamp)`:
- Frees GPU KV (Option B per memory; Option C "warm preempt" deferred to follow-up)
- Sets is_rerouted=True, num_computed_tokens=0
- Queues (req_id, num_checkpointed_tokens) onto `slo_preempted_pending_restore`
- Engine drains queue at step() entry → fed into existing _ft_pending_restores → restore_kv_blocks runs before forward pass

### Verification on A6000 (mechanically only — A6000 too saturated for performance demo)

| Test | Preempts | Restores | Asserts | Completion |
|---|---|---|---|---|
| Smoke 1 (no fault, no FT_SLO_PREEMPT) | 0 | 0 | 0 | 100% — baseline |
| Smoke 2 (no fault, FT_SLO_PREEMPT=1) | **0** | 0 | 0 | **100%** ← short-circuit on no-recovery works |
| Fault test 1 (no per-req cooldown) | 78 (cycle bug) | 80 | 0 | 32% |
| Fault test 2 (per-req cooldown 30s) | **13 (different victims)** | 12 | 0 | 38% |

Mechanical correctness confirmed. Performance demo requires L40S — see ablation plan in memory.

### Next: L40S phase 1-5 per MACHINE_SETUP_RUNBOOK.md

5-condition ablation for paper:

| Cell | Mechanism | Scheduler | Notes |
|---|---|---|---|
| 1 | No-FT | FCFS | drops fault-affected reqs |
| 2 | NoFT-Reprefill | FCFS | full reprefill on recovery |
| 3 | Our-System | FCFS (FT_SLO_PREEMPT unset) | mechanism present, queue-blocked |
| 4 | Our-System | M1+M2 only (FT_SLO_BUDGET_LOG=1, FT_SLO_PREEMPT=0) | sort prioritizes recovery in waiting, no proactive preempt |
| 5 | Our-System | M1+M2+M3 (FT_SLO_PREEMPT=1) | full proposed mechanism |

Same `zoe/slo-scheduling` branch. Toggle via env vars.

## Files To Know

### Configs
- `experiments_v2/config_8b_step5_overnight.yaml` — main config (currently used by overnight matrix)
- `experiments_v2/config_8b_step5_l40s.yaml` — L40S variant
- `/tmp/config_mod_5s.yaml`, `/tmp/config_mod_2s.yaml`, `/tmp/config_light_5s.yaml` — overnight matrix configs

### Profiles
- `decode_capacity_profile_8b_a6000.json` — new A6000 profile (used by current config)
- `decode_capacity_profile_8b.json` — old A5000 profile (DON'T use)
- `checkpoint_cost_profile_8b_a6000.json` — new
- `checkpoint_cost_profile_l40s.json` — exists, 4/26

### Key code paths
- `vllm/v1/engine/ft_client.py` — FT client, has FT_STATIC_CAP, FT_SOLVER_DIAG
- `vllm/v1/engine/core.py` — engine core, has FT_STEP_TIMING, _ft_collect_and_publish_checkpoint
- `vllm/v1/core/sched/benders/solve_loop.py` — Benders solver, has FT_SOLVER_DIAG
- `vllm/v1/core/sched/benders/recovery_checker.py` — PoolOverload constraint (root cause of 99% fallback)
- `vllm/v1/core/kv_checkpoint_pool.py` — KV checkpoint pool, has async stream logic
- `vllm/v1/worker/gpu_model_runner.py` — restore_kv_blocks, _publish_shared_checkpoint

### Useful env vars
- `FT_CKPT_TRUE_ASYNC=1` (default ON, controls async copy stream)
- `FT_CKPT_NONBLOCK=1` (default ON)
- `FT_WORKLOAD_EXHAUST=1` (shuffle-without-replacement sampling)
- `FT_BG_PUBLISH=1` (no-op in our setup, don't bother)
- `FT_CKPT_GPU_OVERLAP=1` (no-op in our setup, don't bother)
- `FT_CKPT_NO_GIL=1` (BAD, don't enable, +33% step time, recovery -38%)
- `FT_STATIC_CAP=N` (admission cap experiment, noise-sensitive)
- `FT_SKIP_SOLVER=1` (force greedy, used for static cap experiment)
- `FT_STEP_TIMING_MAX_CALLS=2000` (instrumentation, must be ≤2000 to dump within cell duration)
- `FT_ENABLE_PREFIX_CACHE=1` (override prefix cache disable, default OFF)

## Recent Experiment Output Locations

- `/tmp/overnight_run/` — 4/26-27 overnight (post-async-fix, prefix cache ON, 5 baselines × 2 workloads × 3 seeds)
- `/tmp/v2full_smoke_5way/` — 4/27 evening (5 baselines × seed=42)
- `/tmp/v2full_3seed_b/` — 4/27 night (5 baselines × seed=123, 456 — completes 3-seed sweep)
- `/tmp/v2full_overnight_2026_04_28/` — 4/27-28 overnight (4 phases including profiles)
- `/tmp/v2full_phase2_rerun/` — 4/28 (W8 timing diagnosis with FT_STEP_TIMING)
- `/tmp/v2full_static_cap/` — 4/28 (FT_STATIC_CAP experiment)
- `/tmp/v2full_bgpublish/` — 4/28 (BG_PUBLISH no-op)
- `/tmp/v2full_ckpt_flags_sweep/` — 4/28 (gpu_overlap / bg_publish / no_gil / combo)
- `/tmp/v2full_vanilla_vs_oursystem/` — 4/29 morning (vanilla vs Our-System with new profile)
- `/tmp/v2full_tight_slo/` — 4/29 (TTFT/gap 5s, 3 baselines × 3 seeds)
- `/tmp/v2full_load_slo_matrix/` — running (overnight matrix)
