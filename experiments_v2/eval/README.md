# Paper evaluation scripts

Experiments that produce paper numbers/figures live here.

**Not smoke tests.** smoke tests (`../smoke/`) only verify mechanism correctness
(no perf claim). Don't conflate the two; don't modify smoke tests for eval work.
Copy if needed.

## Layout

```
eval/
  scripts/      # one .py per experiment in the matrix
  configs/      # per-experiment config (SLO tiers, QPS, kill schedule, …)
  workloads/    # workload generators (RULER prompts, Azure trace replayer)
  analysis/     # log → metrics parsers, plotting
  results/      # output dir; .gitignore'd later
  README.md
```

## Experiment matrix

Defined in `experiments_v2/docs/notes.md` § "实验计划（最终版）". Summary:

| ID | What | Status |
|---|---|---|
| **E_M1** | SLO attainment vs load (Azure trace, 3 system × QPS sweep) | TODO |
| **E_M2** | SLO tightness sweep (3 SLO tier × 3 system) | TODO |
| **E_M3** | System overhead, healthy (3 system, no disruption) | TODO |
| **E_M4** | Picker ablation (ours w/ vs w/o slack picker) | TODO |
| **E_D1** | Disruption recovery demo (3 system × single engine SIGKILL) | scaffolded (`scripts/e_d1_disruption_demo.py`) |

Each experiment: **3 seeds, mean ± std** (rigor bar: above 7/8 surveyed papers).

## Baselines (3, all self-implemented)

| Name | Description |
|---|---|
| `vllm_fcfs` | Upstream vLLM. Engine dies → client gets 5xx. |
| `reroute_no_ckpt` | Our router + cross-engine reroute, but V3 reload disabled (env `FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=0`). Target engine reprefills from scratch. **Llumnix stand-in** — Llumnix's live migration requires source engine alive, so under disruption it degenerates to this. Disclaim in paper text. |
| `ours` | Full system: router + V3 reload + cross-engine restore via `/dev/shm`. |

## SLO calibration

JITServe-style: run `vllm_fcfs` at 1 req/s on RULER 64K, measure baseline
P95 TTFT / P95 TPOT, then set `S_TTFT = 2 × p95_TTFT`, `S_TPOT = 2 × p95_TPOT`.

Three SLO tiers (Niyama-style): `S × {1.5, 3, 6}` for tight / medium / loose.

## Hardware

- Primary: 2x A6000 (Ampere, 48 GB × 2, PCIe).
- Portability check: 2x L40S (Ada, 48 GB × 2, PCIe). Re-run E_M1 only.
- No rentals needed for APSys-scope experiments.

## Running E_D1

```bash
python experiments_v2/eval/scripts/e_d1_disruption_demo.py --baseline ours
python experiments_v2/eval/scripts/e_d1_disruption_demo.py --baseline reroute_no_ckpt
# vllm_fcfs path: TODO
```

Outputs per-request `failover_gap` (seconds from engine SIGKILL to first new
token on the surviving engine) for all rerouted requests, plus P50/P95.
