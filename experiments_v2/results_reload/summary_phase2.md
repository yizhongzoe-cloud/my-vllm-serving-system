# Reload Cost Sweep — Phase 2 Summary

Cells analyzed: 15 CkptReload runs across 5 contexts.

## Recovery Time: Reprefill vs Reload

| Context | Tokens | Reload events | Reload p50 (ms) | Reload p99 (ms) | Reload bytes (MB) | Reprefill p50 (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| W_Ruler1K | 1024 | 8 | 11.3 | 27.3 | 134 | 256 | 23x |
| W_Ruler4K | 4096 | 114 | 69.3 | 132.9 | 252 | 1024 | 15x |
| W_Ruler8K | 8192 | 145 | 43.4 | 642.6 | 254 | 2048 | 47x |
| W_Ruler16K | 16384 | 624 | 12.9 | 68.3 | 256 | 4096 | 319x |
| W_Ruler32K | 32768 | 144 | 69.7 | 105.7 | 256 | 8192 | 118x |

## Interpretation

- **Reload count** = how often vLLM's natural capacity preempt fired (CkptReload baseline only). 16K shows thrashing: same req re-preempted ~7x within run.
- **Reload p50 ms** = median host->GPU restore time (CUDA-event timed on the copy stream).
- **Reprefill p50 ms** ≈ TTFT from Phase 1 No-FT cells (the cost vLLM would pay if it took the RECOMPUTE path).
- **Speedup** = reprefill_p50 / reload_p50. Long context shows 100x+ improvement.
