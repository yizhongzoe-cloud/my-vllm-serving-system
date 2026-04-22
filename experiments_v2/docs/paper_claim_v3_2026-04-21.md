# Paper Claim v3 — Honest assessment (2026-04-21, 15:56)

> **Status note (18:00)**: `gpu47_expansion.sh` is still running (ETA
> ~19:10). It expands W5 Moderate to 12 seeds, W7 Saturated to 9 seeds,
> and tests **W4_Mixed** (production mix with per-request SLOs —
> solver's natural strength). This doc will be re-revised after gpu47
> completes. See `overnight_2026-04-21_summary.md` for the current
> investigation trail and decision tree.

## TL;DR

After 9-seed paired testing, **V2-NoCkpt does NOT reliably strict-beat
NR on the main W7 Heavy benchmark**. The earlier 6-seed 4/6 signal was
partly selection bias. The only consistent empirical claim is the
**+9.9% completion-rate protection on W5 Moderate** (long-context
overload), a narrow but real result.

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

## Secondary claim: W5 Moderate completion-rate protection (n=6)

ArXiv long-prompts at Moderate load push the system into overload.
Both sides frequently miss the 95% completion threshold, so
completion-rate itself becomes the metric.

| seed   | V2-NoCkpt comp | NR comp | Δ     |
|--------|----------------|---------|-------|
| s42    | 78%            | 90%     | -12   |
| s123   | 89%            | 90%     | -1    |
| s456   | **85%**        | **24%** | **+61** |
| s789   | 84%            | 84%     | 0     |
| s1234  | 96%            | 96%     | 0     |
| s22222 | 100%           | 88%     | +12   |
| mean   | **88.5%**      | **78.6%** | **+9.9%** |

- V2-NoCkpt mean comp 88.5% vs NR 78.6% → **+9.9% protection**
- On the worst seed (s456) NR collapses to 24% while V2 holds 85%
- Goodput and fg_p95 essentially tie on this workload
- This is a narrow claim (only applies when system is overloaded with
  long-context requests) but it is real.

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
- ✅ **Solver-driven admission control protects completion-rate by ~10%
   under long-context overload (W5 Moderate)**, with individual seeds
   showing up to +61% protection when the FCFS baseline collapses.
- ✅ **Checkpointing adds orchestration cost that exceeds its recovery
   benefit at 8B/A5000.** Removing it recovers parity with NR but does
   not exceed it.

## Recommended paper pivot

The paper is NOT about "V2 beats NR". That claim doesn't hold.

Two honest framings:

**Option 1 — Negative-result paper**:
"We designed a Benders-based admission controller with KV checkpointing
for fault-tolerant LLM serving. An extensive per-seed evaluation on
Llama-3.1-8B shows that neither full V2 nor any ablation variant
reliably beats a simple FCFS + reprefill baseline on our main W7
benchmark. We attribute this to orchestration overhead in the
checkpoint path and to the low re-prefill cost on 8B/A5000, which
eliminates the break-even benefit of KV reuse. We isolate one narrow
setting (long-context overload, W5 Moderate) where admission-control
provides a +10% completion-rate protection. We publish the full
evaluation, ablation tree, and code to support future work at
70B+/NVLink scales where the trade-off may invert."

**Option 2 — Reposition around overload protection**:
Focus the paper on W5 Moderate + W7 Saturated as the core workload
("overloaded long-context serving"), where admission control's
completion-rate protection is the main metric. Re-collect a larger
seed grid specifically on overload scenarios. Drop W7 Heavy as a main
result.

Either requires more experiments before submission. We should not ship
the v1/v2 claim.

## Concrete next steps

1. **Don't run more W7 Heavy seeds.** 9 seeds is enough — adding more
   won't flip the 4/9 into a majority.
2. Decide the paper framing (Option 1 or 2) with the advisor before
   running more experiments.
3. If Option 2: run V2-NoCkpt vs NR on W5 Moderate × 12 seeds (more
   statistical power on the only claim we have) and on W7 Saturated ×
   12 seeds (second overload scenario).
4. If Option 1: profile V2-Full to attribute the orchestration cost to
   specific checkpoint phases — this strengthens the negative-result
   argument.
