# Paper Claim v3 — Final (2026-04-21, 19:00)

> **Final update (19:00)**: `gpu47_expansion.sh` completed. After
> 4 workloads × 6-12 paired seeds each, **NO V2 configuration
> strict-beats NR**. Paper goes negative-result (Option A). See
> 4-workload summary table below.

## TL;DR

After 2 days + ~120 paired runs across 4 workloads, **V2 (in any
configuration — reload, reprefill, or NoCkpt) does not reliably
beat the NoFT-Reprefill (NR) baseline**. The earlier signals (W7
Heavy 4/6, W5 Moderate +9.9%) were selection bias — both collapsed
under seed expansion. The paper is a negative result with two
useful sub-findings:

1. **KV checkpointing is net-negative** at 8B/A5000 (orchestration
   overhead exceeds re-prefill savings).
2. **Under saturation (RPS=2.5)**, admission control provides a small
   goodput advantage (+5.9% on the 2 stable seeds, n=2 too small to
   generalize).

## Final 4-workload summary (all paired with NR-reprefill)

| Workload       | Paired n | V2-NoCkpt gp vs NR | fg_p95 vs NR | comp Δ | Strict wins | Verdict |
|----------------|----------|-------------------|--------------|--------|-------------|---------|
| W7 Heavy       | 9        | -9.3%             | +6.6%        | —      | 4/9         | Lose (non-sig) |
| W5 Moderate    | 11       | -2.1%             | +4.6%        | +0.2%  | —           | NULL (tie) |
| W7 Saturated   | 2 stable | +5.9%             | +10.8%       | -2.2% (all 9) | 1/2   | Mixed |
| W4_Mixed Heavy | 6        | +0.5%             | -7.3%        | 0      | 3/6         | Narrow tie |

No workload gives V2 a clean strict-win majority. Across all 4:
- gp: V2 ahead on 2/4, behind on 2/4, mean Δ ≈ -1%
- fg_p95: V2 ahead on 1/4, behind on 3/4, mean Δ ≈ +4%
- completion-rate: V2 tied or slightly worse

The one positive is **W7 Saturated's +5.9% gp on stable seeds** — but
only 2 of 9 seeds were comp≥95%, so the sample is too small for a
load-bearing paper claim.

## What we tested

| Recovery | Checkpoint | Solver | Label           | W7 Heavy strict wins (n=9) |
|----------|------------|--------|-----------------|----------------------------|
| reload   | on         | on     | V2 (original)   | 0/9 (losing ~-11% gp) |
| reprefill| on         | on     | V2-reprefill    | 0/8 (losing ~-14% gp) |
| reprefill| off        | on     | **V2-NoCkpt**   | **4/9** (agg -9.3% gp, +6.6% fg) |
| reprefill| off        | off    | NR (baseline)   | reference |

Nothing we tried produces a statistically-meaningful win over NR on
W7 Heavy (paired t-test p≈0.11 for gp, n=9).

## 9-seed W7 Heavy V2-NoCkpt vs NR (per seed)

| seed   | V2-NoCkpt gp/fg | NR gp/fg     | gp | fg | strict |
|--------|-----------------|--------------|----|----|--------|
| s42    | 274.3 / 2796   | 278.1 / 3089 | ✗ -1.4% | ✓  | partial |
| s123   | 162.3 / 3251   | 154.1 / 4555 | ✓ +5.3% | ✓  | 🎯      |
| s456   | 364.1 / 1557   | 357.9 / 2012 | ✓ +1.7% | ✓  | 🎯      |
| s789   | 284.9 / 3540   | 308.4 / 1704 | ✗       | ✗  | no      |
| s1234  | 325.7 / 1053   | 323.6 / 1739 | ✓ +0.6% | ✓  | 🎯      |
| s22222 | 302.0 / 498    | 301.8 / 755  | ✓ +0.1% | ✓  | 🎯      |
| s5678  | 244.7 / 2544   | 305.7 / 2134 | ✗ -20%  | ✗  | no      |
| s9999  | 232.9 / 3522   | 363.0 / 1182 | ✗ -36%  | ✗  | no      |
| s11111 | 176.8 / 941    | 216.8 / 1307 | ✗ -18%  | ✓  | partial |
| agg    | **263.1 / 2189** | **289.9 / 2053** | **-9.3%** | **+6.6%** | **4/9** |

Observation: when V2-NoCkpt wins it wins by small margins (0.1-5.3% gp),
but when it loses it loses big (-18% to -36% gp). Not a reliable
improvement.

## W5 Moderate protection — collapsed under seed expansion (REVISED)

The 6-seed +9.9% protection did NOT hold at 12 seeds. Full final data:

| seed    | V2-NoCkpt comp | NR comp | Δ   |
|---------|----------------|---------|-----|
| s42     | 78%            | 90%     | -12 |
| s123    | 89%            | 90%     | -1  |
| s456    | **85%**        | **24%** | **+61** |
| s789    | 84%            | 84%     | 0   |
| s1234   | 96%            | 96%     | 0   |
| s22222  | 100%           | 88%     | +12 |
| s5678   | 77%            | 82%     | -5  |
| s9999   | 69%            | 70%     | -1  |
| s11111  | 91%            | 92%     | -1  |
| s98765  | 94%            | 95%     | -1  |
| s54321  | 74%            | 89%     | -15 |
| **mean (n=11)** | **85.2%**  | **85.0%** | **+0.2%** (null) |

s456 (V2 +61) is the only real protection signal. It's an outlier —
averaged across 11 seeds, the protection is **statistically null**.

## W7 Saturated (RPS=2.5) — small edge when stable

Only 2 of 9 seeds had comp>=95% on both sides (s11111, s5678).

| seed    | V2-NoCkpt gp/fg/comp | NR gp/fg/comp | gp | fg |
|---------|----------------------|---------------|----|----|
| s5678   | 207.7 / 5770 / 98%   | 189.4 / 6389 / 98% | ✓ | ✓ 🎯 |
| s11111  | 193.8 / 5665 / 96%   | 189.8 / 3931 / 99% | ✓ | ✗ |

- V2 +5.9% gp on the 2 stable seeds
- Both sides unstable on 7/9 other seeds (comp 88-94%)
- Sample too small (n=2) for paper-grade claim
- Matches intuition: admission control should help more when load
  exceeds capacity (NR's FCFS overadmits, V2 throttles)

## W4_Mixed Heavy — paired production mix

Production workload mix (50% chat 100ms SLO / 30% summary 200ms /
20% instruction 50ms) at Heavy load. All 6 seeds reach 100%
completion on both V2-NoCkpt and NR.

| seed    | V2-NoCkpt gp/fg | NR gp/fg | gp | fg | strict |
|---------|-----------------|----------|----|----|--------|
| s42     | 251.1 / 1324    | 249.2 / 1630 | ✓ | ✓ | 🎯 |
| s123    | 276.2 / 1457    | 272.8 / 1076 | ✓ | ✗ | no |
| s456    | 242.6 / 778     | 242.2 / 720  | ✓ | ✗ | no |
| s789    | 256.5 / **384** | 256.3 / 724  | ✓ | ✓ | 🎯 |
| s1234   | 239.3 / **0**   | 239.3 / 589  | ✓ | ✓ | 🎯 |
| s22222  | 220.2 / 953     | 218.9 / 541  | ✓ | ✗ | no |
| **mean**| **247.7 / 816** | **246.4 / 880** | **+0.5%** | **-7.3%** | **3/6** |

- V2 wins 6/6 on goodput (but by tiny margins, +0.2%-+1.2%)
- V2 wins 3/6 on fg_p95 (with some dramatic wins: s789 -47%, s1234 -100%)
- 3/6 strict wins — same ratio as W7 Heavy (4/9 ≈ 44%)
- Too tied to call a reliable improvement

## What we CANNOT claim

- ❌ V2 (original, with checkpointing) beats NR. It loses by ~11% gp.
- ❌ V2-reprefill beats NR. Same failure mode.
- ❌ V2-NoCkpt reliably beats NR. 4/9 strict wins, aggregate slightly
   loses both metrics, not statistically significant.
- ❌ Checkpointing reduces fg_p95. Ablation shows the opposite.
- ❌ Warm-start (R7 greedy seed) improves anything measurable.

## What we CAN claim

- ✅ **V2 with checkpointing is consistently worse than NR-reprefill on
   W7 Heavy** (the primary benchmark). Strong negative result.
- ✅ **Checkpointing adds orchestration cost that exceeds its recovery
   benefit at 8B/A5000.** Removing it recovers parity with NR but does
   not exceed it. Ablation tree (V2-reload / V2-reprefill /
   V2-NoCkpt) isolates checkpoint as the cost center.
- ✅ **Under saturation (RPS=2.5, W7_Saturated)**, V2-NoCkpt's solver
   admission control gives +5.9% goodput on stable seeds (n=2 only).
   Not strong enough for a main claim, but consistent with the
   admission-control theory.
- ✅ **Re-prefill cost on 8B is cheap enough that KV reload's
   break-even never materializes** at this model size / hardware.

## What we CANNOT claim (REVISED after full seed expansion)

- ❌ V2 (original, with checkpointing) beats NR. Loses by ~11% gp.
- ❌ V2-reprefill beats NR. Same failure mode.
- ❌ V2-NoCkpt reliably beats NR. 4/9 (W7 Heavy) + 3/6 (W4_Mixed)
   strict wins, aggregate slightly loses both metrics, not
   statistically significant.
- ❌ V2 provides completion-rate protection. **s456 was an outlier**;
   mean Δ at 11 seeds is +0.2%, null.
- ❌ W4_Mixed's per-request SLO differentiation is solver's natural
   advantage. Data shows 3/6 wins and essentially tied means.
- ❌ Checkpointing reduces fg_p95. Ablation shows the opposite.
- ❌ Warm-start (R7 greedy seed) improves anything measurable.

## Final paper framing: Option A (negative-result)

Option B (overload-protection pivot) was tested and falsified — the
W5 Moderate protection claim collapsed under seed expansion (6 → 12)
and W7 Saturated's signal (n=2 stable) is too narrow.

**Verdict**: ship as a **negative-result paper** with strong
methodology and a clear message to the community.

### Proposed abstract sketch

"We implement a Benders-decomposition admission controller with KV
checkpointing for fault-tolerant LLM serving (V2) and evaluate it
against a FCFS + reprefill baseline (NR) on Llama-3.1-8B / 2×A5000.
Across 4 workloads (W7-chat, W5-arxiv, W4-mixed, and a saturated
variant) and 6-12 paired seeds per cell, **V2 does not reliably beat
NR** in any configuration — with or without the checkpoint path, with
or without warm-start, across early/mid/late fault timings. Ablation
isolates KV checkpointing as a net-negative at this scale:
orchestration overhead (pinned-pool management, staged gather/scatter,
per-request version tracking) competes with the decode critical path,
while re-prefill cost on 8B is cheap enough that KV reuse never
pays off. We suggest the admission-control + checkpoint design
requires 70B+ model sizes or NVLink-rich interconnects to invert the
trade-off, and publish our full evaluation framework for the
community to retest."

### Strengths of this framing

1. Rigorous: 2 days, ~120 paired runs, 4 workloads, 5 configurations.
2. Honest: flags the 1 outlier (W5 s456) and explicitly notes the
   null result at 12 seeds.
3. Actionable: tells future work where to look (70B+, NVLink).
4. Code + data released, fully reproducible.

### Concrete next steps

1. **No more V2-vs-NR seeding**. Data is conclusive at 4 workloads ×
   6-12 seeds. More seeds won't flip the verdict.
2. **Profile V2-Full vs V2-NoCkpt** on one seed with `nvtx` to
   decompose the orchestration cost into specific phases (pool,
   gather, scatter, version-bump). Concrete per-phase numbers
   strengthen the negative-result argument.
3. **Discuss with advisor** whether to ship as negative-result or
   pivot to a different contribution (e.g., the admission controller
   itself as a technical novelty independent of fault tolerance).
4. **Consider framing as "Bamboo at wrong scale"** — show that
   reload-based recovery is only valuable when KV-reuse savings
   exceed orchestration cost, quantify the crossover point, and
   argue 8B is below it.
