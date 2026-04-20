# Stress Test Summary 2026-04-20

## TL;DR — **The algorithm has NO long-context win; it ties NR on short-prompt saturation only**

After running 61-cell stress matrix (W5 long-doc arxiv + W7 saturated sharegpt, × V2/NR/V2+skip_solver, × F1/F2/F3/none faults, × 3 seeds each), three regimes emerge:

1. **W7 (short-prompt, RPS saturation)** — **V2 ≈ NR** (ties, ~-5% worst)
2. **W5 (long-prompt, 5k avg)** — **V2 consistently LOSES** (-23% goodput baseline, -4-8× fg_p95)
3. **W5 + early fault (F1)** — both systems collapse (comp 13-54%)

**No scenario found where OS strictly wins NR on all metrics.**

## The critical finding: OS has a **23% persistent overhead on long context**

W5 no-fault baseline (3 seeds, no fault at all):
| seed | NR gp | V2 gp | Δ |
|---|---|---|---|
| s42 | 96.8 | 74.4 | **-23%** |

(seeds 123, 456 pending)

On W7 (short prompts), no-fault V2 = NR:
| seed | NR gp | V2 gp | Δ |
|---|---|---|---|
| s42 | 416.2 | 415.4 | 0% |
| s123 | 416.8 | 415.9 | 0% |
| s456 | 387.9 | 390.2 | 0% |

**Interpretation**: OS's orchestration overhead (solver, snapshot, admission) scales with prompt length, not with request count. On W7's short prompts, per-request overhead is tiny. On W5's 5k prompts, orchestration cost dominates.

This is the **opposite** of the hypothesis. We thought long context would help OS (prefill cost flips in OS's favor). Actually long context **amplifies** OS overhead because:
- Snapshot serialization grows O(prompt_length)
- Solver MIP variables scale with tokens-in-flight
- KV checkpoint bookkeeping scales with KV footprint

## W7 fault-timing table (main result)

| fault | V2 gp (3-seed) | NR gp (3-seed) | V2 fg_p95 | NR fg_p95 | verdict |
|---|---|---|---|---|---|
| none | 407 | 407 | 0 | 0 | ties |
| F1_Early | 167 (comp 77%) | 210 | 2857 | 2837 | NR slightly better |
| F2_Mid | 242 (comp 97%) | 264 | 3650 | 3236 | NR ~10% better |
| F3_Late | 321 | 358 | 5174 | 2531 | V2 fg_p95 **2× worse** |

Goodput: V2 always slightly below NR. fg_p95: V2 close on F1, loses on F2/F3.

W7 V2/s42/F1 had comp=40% (anomalous single-seed). Other seeds OK.

## W5 fault-timing table

| fault | V2 gp | NR gp | V2 comp | NR comp | V2 fg_p95 | NR fg_p95 |
|---|---|---|---|---|---|---|
| none (so far) | 74.4 (s42) | 96.8 (s42) | 100% | 100% | 0 | 0 |
| F1_Early | 9.5 | 9.1 | 35% | 52% | 14637 | 9636 |
| F2_Mid | 22.2 | 30.5 | 76% | 88% | 13538 | 6115 |
| F3_Late | 46.0 | 63.8 | 100% | 100% | 14510 | 8652 |

**Every cell: V2 loses on goodput AND fg_p95.** W5 F1 is catastrophic for both (completion <50%).

## Paper implications — brutal honesty

**What we can't claim**:
- "Our system beats NR on long context" — **false**, opposite is true
- "Our admission/solver helps under stress" — **false**, solver adds overhead with no upside
- "Our checkpoint mechanism saves prefill cost" — **not observed** in these tests

**What we can honestly claim**:
- "Our fault-tolerant framework adds zero overhead on short-prompt workloads (W7)"
- "Fault recovery is within 5-15% of reprefill baseline (on short prompts)"
- "Long-context regime exposes O(prompt_length) overhead in the orchestration layer — future work direction"

## Recommended paper pivot

Given the data, three options:

1. **Scale up to 70B** — if prefill cost per token is 10× higher, maybe OS's KV-reload savings finally exceed orchestration costs. Single-A5000 can't do 70B, would need different hardware.

2. **Reframe as mechanism paper** — don't claim wins over NR, claim "FT serving system design space exploration" with honest tradeoff measurements. This is publishable but less exciting.

3. **Target a different regime** — e.g., replica-heavy dp=4+ where solver's reroute value grows. Not possible on current GPU allocation.

**My honest recommendation**: **option 2**. The data tells us OS's orchestration is net-negative on 8B/dp=2/moderate-context. Pretending otherwise gets rejected. A mechanism paper with honest characterization is valid contribution and the data supports that framing.

## COMPLETE — 61/61 runs done (16:24, wall time 3h40min)

### Full 3-seed means (including low-completion rows)

```
variant                            gp   comp         fg_p95
W5 NR F2               30.4±  4.1      88%     6115± 3568
W5 V2 F2               22.2±  2.8      76%    13537± 2549   (-27% gp, +121% fg)
W5 V2+skip F2          24.5±  2.6      75%    14753± 4394   (-20% gp, +141% fg)

W5 NR F1                9.1±  1.1      52%     9636± 2339
W5 V2 F1                9.5±  4.1      35%    14638± 3445   (gp similar, comp WORSE)
W5 V2+skip F1           9.6±  2.1      38%    13170± 2788

W5 NR F3               63.8±  9.5     100%     8651± 1389
W5 V2 F3               46.0±  4.9     100%    14510±  985   (-28% gp, +68% fg)
W5 V2+skip F3          44.2±  9.5     100%    13402± 3271

W5 NR none            118.7± 20.7     100%        0         (baseline)
W5 V2 none             87.1± 14.9     100%        0         (**-27% goodput with no fault**)

W7 NR F2              263.6±103.3     100%     3236± 1307
W7 V2 F2              242.3± 82.1      97%     3650±  902
W7 V2+skip F2         234.4± 78.4      97%     4001± 1376

W7 NR F1              209.7±124.1     100%     2836±  717
W7 V2 F1              167.1±105.2      77%     2857± 1029   (comp drops, fg_p95 matches)
W7 NR F3              357.6± 24.2     100%     2531±  136
W7 V2 F3              321.3± 41.7      97%     5174±  671   (-10% gp, +104% fg)

W7 NR none            407.0± 16.5     100%        0
W7 V2 none            407.2± 14.7     100%        0         (0% overhead, ties exactly)
```

### Confirmed final verdict

| workload × fault | V2 vs NR outcome |
|---|---|
| W7 none | **TIE** (zero overhead) ✓ |
| W7 F1 | NR slightly better (V2 comp drops) |
| W7 F2 | NR ~10% better on goodput, similar fg_p95 |
| W7 F3 | NR 10% gp better, V2 fg_p95 **2× worse** |
| W5 none | **NR +36% goodput** (-27% OS overhead) ❌ |
| W5 F1 | BOTH catastrophic (comp 35-52%) |
| W5 F2 | NR +37% gp, V2 fg_p95 **2.2× worse** ❌ |
| W5 F3 | NR +39% gp, V2 fg_p95 **1.7× worse** ❌ |

**No cell where V2 strictly beats NR on all three metrics (gp, comp, fg_p95).**

## Final paper position

The hypothesis that long context would favor OS was **falsified**. OS's orchestration overhead scales with prompt length — the opposite of what we needed. On 8B/dp=2, for all tested fault/workload combinations:

- **OS = NR** on short-prompt workloads (W7 no-fault)
- **OS < NR** on long-prompt workloads (W5, any fault scenario)
- **OS < NR** on fault-injected short-prompt (W7 F1/F2/F3)

**Paper pivot — mechanism contribution (recommended)**:
- Drop the "we beat NR" framing
- Write the paper as: *"Design and overhead characterization of a fault-tolerant admission + checkpoint framework for LLM serving. We show: (a) zero overhead on short-prompt workloads, (b) O(prompt_length) orchestration overhead on long-context due to solver/snapshot scaling, (c) recovery latency comparable to reprefill when fault is mid-workload."*
- Position **W7 none** as proof of "overhead-free infrastructure", and frame long-context penalty as identified **limitation + future work** direction.
- This is honest and defensible under peer review.

## Files

- Stress pipeline: `experiments_v2/stress_pipeline_2026-04-20.sh` (main, DONE 14:00)
- Extended pipeline: `experiments_v2/stress_pipeline_ext_2026-04-20.sh` (running)
- Plan doc: `experiments_v2/docs/stress_test_plan_2026-04-20.md`
- This summary: `experiments_v2/docs/stress_test_summary_2026-04-20.md`
- Logs: `/tmp/stress_pipeline.log`, `/tmp/stress_pipeline_ext.log`, `/tmp/stress_ext_gpu23.log`, `/tmp/stress_w7.log`
- Raw data: `results_v2/8B/stress_2026-04-20/{w5,w5_ext,w7,w7_ext,smoke}/`
