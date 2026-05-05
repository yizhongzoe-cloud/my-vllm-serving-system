# Ckpt Overhead Sweep — Summary
Cells analyzed: 15 paired (No-FT, CkptOnly) observations across 5 contexts.

## Per-Context Overhead (mean ± std across seeds)

| Context | Tokens | Seeds | ΔForward p50 % | ΔForward p99 % | Ckpt fires (mean) |
|---|---|---|---|---|---|
| W_Ruler1K | 1024 | 3 | +0.06 ± 0.04 | +0.02 ± 0.03 | 1707 |
| W_Ruler4K | 4096 | 3 | +0.14 ± 0.04 | +0.04 ± 0.03 | 1287 |
| W_Ruler8K | 8192 | 3 | +0.54 ± 0.30 | -0.02 ± 0.07 | 861 |
| W_Ruler16K | 16384 | 3 | -1.94 ± 0.31 | -1.34 ± 0.21 | 2242 |
| W_Ruler32K | 32768 | 3 | -3.02 ± 0.93 | -0.08 ± 0.05 | 1547 |

## Interpretation

- **ΔForward p50 / p99 < 5%**: TokenFlow's claim that async KV transfer overhead is negligible **holds for our setup** in this context-length regime. Mechanism is sound.
- **5-15%**: edge case — likely PCIe back-pressure exerting partial pressure on SMs. Worth engineering optimization (pinned memory, async copy stream tuning).
- **>15%**: PCIe contention is meaningfully squeezing GPU compute. Investigate before claiming the mechanism.

## Paired ΔForward by Seed (raw)

See `paired_diff.csv` for full detail.
