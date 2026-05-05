# Reload Cost Sweep — Phase 2 Summary

Cells analyzed: 9 CkptReload runs across 5 contexts.

## Recovery Time: Reprefill vs Reload

| Context | Tokens | Reload events | Reload p50 (ms) | Reload p99 (ms) | Reload bytes (MB) | Reprefill p50 (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| W_Ruler1K | 1024 | 10 | 30.0 | 45.4 | 132 | 256 | 9x |
| W_Ruler16K | 16384 | 6 | 185.6 | 399.4 | 1048 | 4096 | 22x |
| W_Ruler32K | 32768 | 44 | 125.0 | 134.2 | 2026 | 8192 | 66x |

## Interpretation

- **Reload count** = how often vLLM's natural capacity preempt fired (CkptReload baseline only). 16K shows thrashing: same req re-preempted ~7x within run.
- **Reload p50 ms** = median host->GPU restore time (CUDA-event timed on the copy stream).
- **Reprefill p50 ms** ≈ TTFT from Phase 1 No-FT cells (the cost vLLM would pay if it took the RECOMPUTE path).
- **Speedup** = reprefill_p50 / reload_p50. Long context shows 100x+ improvement.
