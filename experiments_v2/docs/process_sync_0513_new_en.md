# 2026-05-13

## Idea

- **Old idea**: host-side KV checkpoint for fault tolerance — when an gpu dies, a peer gpu resumes from the checkpoint. Problem: failure probability is too low in practice, so the benefit is too small.
- **New idea**: leverage the KV checkpoint mechanism to make preemption cheap, so the scheduler can better meet SLO requirements and improve overall goodput. **We reuse the resume-without-reprefill path from the old idea** — the rest of the design (slack-based preempt policy) is new.

## Design

### KV checkpoint mechanism

| | |
|---|---|
| What | per-block delta (only newly-stable blocks) |
| When | every 16 tokens decoded (one vLLM KV block) |
| Where | L1 host pinned (same-engine) + L2 shared host store (cross-engine readable) |

Overhead under normal load is small: TTFT P50 goes up by about 30 ms, throughput stays within 0.5%.

![checkpoint cadence](../arch_drafts/checkpoint_cadence.svg)

### Slack-based preempt picker

- **slack** = wall-clock remaining until the request misses its SLO
- Each scheduler step: pick max-slack from running queue (victim), min-slack from waiting queue (head)
- Preempt iff (three AND): `gap > δ + replay_cost` · `victim has checkpoint` · `cooldown elapsed`

## Method

### Setup

| | |
|---|---|
| Hardware | 2× A6000 (+ L40S portability) |
| Model | Qwen2.5-7B-Instruct fp16, `max_model_len 32K` |
| Workloads | RULER 16K (target) · ShareGPT (don't break) |
| Arrivals | Poisson · 3-tier QoS (tight / normal / loose) |
| SLO tiers | baseline P95 × {2×, 3×, 6×} |
| Baselines | `vllm_fcfs` · `reroute_no_ckpt` · `ours_no_picker` · `ours` |
| Seeds | 3 per config |

### Experiments

| | Measures | Status |
|---|---|---|
| **E_M1** | SLO attainment vs QPS | A6000 done · L40S in progress |
| **E_M4** | picker ablation (+ `ours_no_picker`) | scheduled this week |
| **E_D1** | failover gap (FT use case, bonus) | RULER 16K demo working |

## Submission

- **Target**: APSys 2026 Workshop
- **Deadline**: May 20 (Wed)
