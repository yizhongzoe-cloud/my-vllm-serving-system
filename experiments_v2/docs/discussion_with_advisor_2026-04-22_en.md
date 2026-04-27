# Project Status & Direction Discussion

**Date**: 2026-04-22
**Purpose**: Present current experimental results, diagnose why we are not beating the baseline, and propose a path forward.

---

## 1. What Our System Does

Our system provides **fault-tolerant LLM serving** with two core mechanisms:

1. **Benders-based solver** for admission control: every 20 ms, jointly decides which incoming requests to accept and which replica to route them to. The solver **proactively reasons about all possible GPU failure scenarios**, ensuring that if any GPU fails, the surviving GPUs can still recover in-flight requests within the SLO.

2. **Adaptive checkpoint**: periodically saves KV cache from GPU to CPU host memory. When a failure occurs, in-flight requests can resume from the most recent checkpoint instead of re-running prefill.

**Baselines we compare against**:
- **No-FT (NR)**: vanilla vLLM, no fault tolerance, no checkpoint
- **Periodic-Low**: fixed checkpoint every 10 blocks
- **Periodic-High**: fixed checkpoint every 1 block
- **Our-System (V2)**: Benders routing + adaptive checkpoint

---

## 2. Experimental Results: We Are Losing to the Baseline

### 2.1 Goodput under Fault (F2_Mid) — W1_Chat

![Goodput under fault](figs_advisor_2026-04-22/goodput_by_load_F2_Mid.png)

**Observation**: No-FT has the highest goodput across all load levels; Our-System trails, especially at Heavy load.

### 2.2 SLO Violation Rate under Fault

![SLO violation](figs_advisor_2026-04-22/slo_violation_F2_Mid.png)

**Observation**: No-FT looks to have the lowest SLO violation, but this is partly because No-FT **silently drops** failed requests (doesn't count them). Periodic-High is the worst — showing that **more checkpointing is NOT better**.

### 2.3 Failover Gap (p95 Recovery Time)

![Failover gap p95](figs_advisor_2026-04-22/failover_gap_p95.png)

**Observation**: Our-System achieves competitive recovery gap, but the normal-case overhead erases this benefit in overall goodput.

### 2.4 Our Strongest 9-seed Paired Comparison (W7_Saturated Heavy)

| Recovery | Checkpoint | Solver | Config | Strict wins vs NR (n=9) |
|---|---|---|---|---|
| reload | ✓ | ✓ | **V2-full** | **0 / 9** (−11% gp) |
| reprefill | ✓ | ✓ | V2-reprefill | 0 / 8 (−14% gp) |
| reprefill | ✗ | ✓ | **V2-NoCkpt** | 4 / 9 (−9.3% gp, not significant) |
| reprefill | ✗ | ✗ | NR (baseline) | reference |

**Critical finding**:
- **Every version with checkpoint enabled LOSES to NR**
- **Removing the checkpoint mechanism brings performance closest to NR**
- Paired t-test p ≈ 0.11, not statistically significant even for V2-NoCkpt

### 2.5 The Only Positive Result: W5 LongDoc Overload

On ArXiv long-document workload at Moderate load, V2-NoCkpt **protects completion rate**:

| seed | V2-NoCkpt completion | NR completion | Δ |
|---|---|---|---|
| 42 | 78% | 90% | −12 |
| 123 | 89% | 90% | −1 |
| 456 | **85%** | **24%** | **+61** |
| 789 | 84% | 84% | 0 |
| 1234 | 96% | 96% | 0 |
| 22222 | 100% | 88% | +12 |
| **mean** | **88.5%** | **78.6%** | **+9.9%** |

On the worst seed (s456), NR's FCFS queue collapses (24% completion); V2's admission control holds the line at 85%. This is our **only reliable positive claim**.

### 2.6 Ablation: Which Component Matters?

![Ablation](figs_advisor_2026-04-22/ablation_F2_Mid.png)

**Observation**: Adaptive checkpoint (alone) reaches parity with full Our-System; Benders routing alone underperforms significantly. **Routing is not contributing in our current dp=2 setup.**

### 2.7 Controller (Benders Solver) Overhead

![Controller overhead](figs_advisor_2026-04-22/controller_overhead.png)

**Observation**: Solver latency ~10-25 ms per epoch. Epoch fires every 20 ms. **The solver is running almost continuously on the API server main thread**, directly contending with decode scheduling.

### 2.8 Recovery Breakdown (E2)

![Recovery breakdown](figs_advisor_2026-04-22/e2_recovery_breakdown.png)

**Observation**: Recovery latency is dominated by detection + re-prefill at 8B/A6000. Checkpoint reload savings are small relative to total recovery time.

---

## 3. Current problems

We checked five possible causes. Summary:

| # | Possible cause | Does this affect us? | Fix |
|---|---|---|---|
| 1 | Solver algorithm too slow | No — MIP itself runs in 5-15 ms | Not needed |
| 2 | Python overhead around solver (snapshot, dispatch) | Yes — adds ~10-25 ms / epoch | Lower solver frequency (20 ms → 1 s), move hot code to C |
| 3 | Solver pipeline on main thread, blocks decode | Yes — snapshot/dispatch run on main thread | Move snapshot/dispatch to sidecar process, or make event-driven |
| 4 | Systems-level contention (GIL, async, RPC) | Yes — main thread only 13% CPU, stuck waiting | Batch RPCs, C extensions, fewer sync points |
| 5 | **Checkpoint brings no measurable benefit in this setting** | **Yes — and this is the biggest issue** | Cannot be fixed by engineering. Need different setting. |

**Much of Mode 2-4 has already been addressed**:

Even after all these engineering wins, **V2-full still loses to NR by ~11% on the primary 9-seed benchmark**. This is the strongest evidence that engineering fixes cannot close the gap — we are bottlenecked by Mode 5.

**Mode 5 is the critical one**: even if we remove all solver overhead, re-prefill on 8B/A6000/short-prompt is already only ~300 ms — close to the cost of checkpoint reload itself. There is no headroom for checkpoint to win.

### But Mode 5 is setting-dependent

The current setting (8B + A6000 + short prompts) makes re-prefill too cheap for checkpoint to win. We need to change other settings:

- **Long context (32K+)**: re-prefill grows O(N²), reaching seconds-to-minutes
- **Larger model (70B+)**: per-token FLOPs much higher
- **Multi-GPU TP on PCIe**: re-prefill pays communication overhead(need adjust algorithm)

Our one positive result (W5 LongDoc, +9.9% completion) is exactly in a longer-context setting — suggesting the value proposition could be real when re-prefill is genuinely expensive.

### Planned adjustment

- **Hardware**: L40S 48 GB (larger memory)
- **Model**: Llama-70B (higher per-token cost)
- **Workload**: 32K+ long context (ArXiv full, LongBench, InfiniteBench)

Machines are still pending. If results on the new setting are still not good, we will need to rethink the algorithm or the system design.


