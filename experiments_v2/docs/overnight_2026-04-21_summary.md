# Overnight 2026-04-21 Summary

## TL;DR — Two-day investigation concludes: **V2 with checkpointing does NOT beat NR on W7 Heavy**. The only surviving claim is **+9.9% completion-rate protection on W5 Moderate (long-context overload)**.

- Day 1 (2026-04-20 overnight → 2026-04-21 morning): 5-phase overnight
  (63 runs) tried V2-reprefill, multi-timing faults, Saturated load,
  ablations — all 0/6 or worse strict wins vs NR.
- Day 2 (2026-04-21 afternoon): discovered **V2-NoCkpt** (solver +
  reprefill, checkpoint OFF) hit 4/6 strict wins on W7 Heavy. Extended
  to 9 seeds → regressed to **4/9**, non-significant.
- Currently running (PID 1552076, ETA 2026-04-21 ~19:10): parallel
  GPU 4-7 expansion — W5 Moderate 12-seed (firm up protection) + W7
  Saturated 9-seed + W4_Mixed 6-seed (new SLO-differentiation angle).

## What we proved

1. **V2 with KV checkpointing + reload recovery consistently loses NR**
   on W7 Heavy: 9-seed paired Δ gp=-11%, Δ fg_p95=+42%. Strong
   negative result.
2. **V2-reprefill** (keep solver + ckpt, recovery→reprefill) does not
   save it: same ~-14% gp shortfall on W7.
3. **Removing checkpoint entirely** (`Our-System-NoCkpt`) recovers
   near-parity (Δ gp=-9%, Δ fg=+7%, 4/9 strict wins) but not a
   statistically significant win.
4. **Checkpoint orchestration is net negative** at 8B/A5000:
   ablation tree clearly shows every checkpoint-enabled variant
   losing, NoCkpt tying/winning narrowly.
5. **W5 Moderate +9.9% completion-rate protection** (n=6) is the only
   V2 metric that reliably beats NR; individual seed s456 shows +61%
   (NR 24%, V2-NoCkpt 85%) when FCFS queue collapses.

## What failed

| Attempted angle                     | Outcome                   |
|-------------------------------------|---------------------------|
| Reload mode tuning (workers=32/interval=1/cap=20/combo) | 0/3 strict wins × 4 configs |
| W5 Light low-load (stabilize cuda_assert) | 0/3, V2 gp 30% below NR |
| V2-reprefill (keep solver+ckpt, swap recovery) | 0/9 W7, 1/6 partial W5 |
| F1_Early / F3_Late fault timing     | 0/3 + 0/3                 |
| RPS=2.5 Saturated (W1 + W7)         | Both unstable, inconclusive |
| no-warmstart ablation               | no measurable effect       |

## Commits pushed today

| SHA        | Summary                                          |
|------------|--------------------------------------------------|
| `d6111e859`| overnight 0421 config + scripts + paper claim v3 |
| `e2067e358`| pre-overnight vllm optimization stack            |
| `b623f11cd`| older diag scripts + docs backlog                |
| `3c6181cd1`| experiment result archive (all except gpu47)     |

## Still running (as of 18:00)

`experiments_v2/gpu47_expansion.sh` — parallel tracks on GPU 4-5 and
GPU 6-7. Writes to `results_v2/8B/gpu47_2026-04-21/`.

- Track A: W5 Moderate NoCkpt × 6 new seeds + NR × 6 (12 runs)
- Track B: W7 Saturated NoCkpt × 6 + NR × 6 + W4_Mixed NoCkpt × 6 + NR × 6 (24 runs)

**W4_Mixed is the most promising remaining angle**: production mix
with per-request SLOs (chat 100ms, summary 200ms, instruction 50ms).
Solver-driven admission control should naturally differentiate these
under contention, whereas FCFS cannot. If W4_Mixed shows strict wins
where single-SLO workloads did not, the paper pivots to
"SLO-aware fault-tolerant admission control."

## Decision tree after gpu47 finishes

| gpu47 outcome                                        | Paper pivot |
|------------------------------------------------------|-------------|
| W4_Mixed ≥5/6 strict wins                            | **New main claim: SLO-differentiating admission control**. Re-run full seed grid on W4. |
| W5 12-seed protection holds at +8% or more           | Overload-protection paper, narrow scope. Need more long-context workloads. |
| W4 and W5 both flat, W7 Sat still inconclusive       | Accept **negative-result paper** per `paper_claim_v3_2026-04-21.md`. |

## Reproducing

```bash
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

# 5-phase overnight (2026-04-21 baseline, ~6h)
bash experiments_v2/overnight_0421.sh

# NoCkpt 6-seed validation (~75 min)
bash experiments_v2/nockpt_validation.sh

# NoCkpt W7 Heavy 9-seed extension (~20 min)
bash experiments_v2/nockpt_9seed.sh

# GPU 4-7 parallel expansion (~3h)
bash experiments_v2/gpu47_expansion.sh
```

See also: `paper_claim_v3_2026-04-21.md` for the full evidence table
and paper-writing recommendation.
