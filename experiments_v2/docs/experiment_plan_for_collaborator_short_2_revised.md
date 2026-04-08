# Experiment Checklist for FT Multi-GPU LLM Serving

## Goal
Run a clean evaluation for our fault-tolerant multi-GPU serving system and answer four questions:
1. What is the steady-state overhead in normal operation?
2. How much does the system improve recovery and SLOs under a single-GPU failure?
3. Why do we need joint admission + routing + adaptive checkpointing?
4. Is the epoch-level controller fast enough online?

Default assumptions:
- Single node, multiple identical GPUs
- Online serving with periodic decision epochs
- Single GPU failure only (`k = 1`)
- Policies available: `none / fixed-low / fixed-high / adaptive`

## Ground Rules
- Use the same model, hardware, and epoch length within each comparison group.
- Run at least 3 seeds per setting. Report means for main results and add error bars when useful.
- For every baseline, save the exact runtime mode, flags, and command or config used. Do not rely only on the baseline nickname.
- If a baseline is approximated rather than exactly implemented, state the approximation explicitly.
- Do not silently replace an unsupported baseline with a different one.

## Systems to Compare
Please use these names consistently in scripts, plots, and tables.

- `No-FT`: no checkpoint; affected requests fail or restart
- `Fixed-Low-CKPT`: fixed low checkpointing
- `Fixed-High-CKPT`: fixed high checkpointing
- `Reactive-Reroute`: no failure-aware reservation during normal operation; reroute only after failure
- `Robust-Routing-Only`: failure-aware admission/routing, but fixed checkpoint policy
- `Checkpoint-Only`: adaptive checkpointing, but simple dispatch/admission
- `Our-System`: joint admission + epoch dispatch + failover rerouting + adaptive checkpointing

Minimum required set if time is tight:
- `No-FT`
- `Fixed-Low-CKPT`
- `Fixed-High-CKPT`
- `Reactive-Reroute`
- `Robust-Routing-Only`
- `Our-System`

## Workloads
Use three representative workloads:

- `W1 Short-Interactive`: short prompt, short or medium output, stable arrivals
- `W2 Long-Generation`: medium or long prompt, long output, stable arrivals
- `W3 Bursty-Mixed`: mixed prompt and output lengths, bursty arrivals

Before the main sweep, instantiate each workload with concrete token ranges and arrival-process parameters in `config.yaml`. Do not leave `short / medium / long` only as words.

For each workload, run three load levels:
- `Low`: clearly underloaded
- `Medium`: near the target operating region
- `High`: close to saturation

Choose the load levels from a no-failure prescan and save the prescan result:
- find where performance starts to degrade clearly
- pick one point well below it as `Low`
- pick one near it as `Medium`
- pick one slightly above or at the edge as `High`

## Fault Injection
Default fault setting:
- single GPU fail-stop or worker kill
- one failure per run
- at least three timing points: `F1 Early`, `F2 Mid`, `F3 Late`

Important:
- define `F1 / F2 / F3` as explicit injection rules in the run config
- use either absolute times or explicit progress-based rules
- do not leave them as qualitative labels only
- use the same fault schedule across systems for matched comparisons

Record in every run:
- failed GPU id
- absolute fault timestamp
- fault seed
- active request count on the failed GPU right before failure
- checkpoint count or checkpointed state count on the failed GPU right before failure

## Metrics to Report
### Main metrics
- Goodput
- TTFT p50 / p95 / p99
- TPOT p50 / p95 / p99
- Admission rate
- Completion rate
- SLO violation rate

Metric definitions (use these consistently across all systems):
- `Goodput` = completed output tokens per second from requests that satisfy the target SLO definition for that experiment.
- `SLO violation rate` = fraction of requests that violate any required bound for that experiment. In normal-case runs, use TTFT and TPOT bounds. In failure-case runs, use TTFT, TPOT, and failover-gap bounds.
- `Time to stable throughput after failure` = time from fault injection until post-failure throughput reaches at least 90% of the pre-failure moving-average throughput and stays there for at least 10 seconds.

### Recovery metrics
- Failover gap p50 / p95 / p99
- Restore time
- Replay tokens / replay time
- Recovery success rate
- Time to stable throughput after failure

### Controller metrics
- Epoch optimization latency
- Master solve time
- Recovery checking time
- Number of cuts / iterations
- Solver timeout or fallback rate if it happens

### Resource metrics
- Checkpoint traffic
- Host memory footprint

## Experiments
### E1. Main end-to-end results
For each workload and load level, compare the required baselines under:
- no failure
- one injected GPU failure at `F1 / F2 / F3`

Report:
- goodput
- TTFT / TPOT tails
- admission and completion rate
- SLO violation rate
- failover gap

Question answered:
- Does the full system help in both normal and failure cases?

### E2. Recovery breakdown
Focus only on failure runs.

Break recovery into:
- detection
- restore
- replay
- resume

Also report:
- replay tokens
- recovery success rate
- throughput recovery curve after failure

If a baseline does not support true recovery, mark it explicitly as failed or restart-from-scratch.

Question answered:
- Where does the recovery improvement come from?

### E3. Ablation / joint optimization
Compare:
- `Our-System`
- `Robust-Routing-Only`
- `Checkpoint-Only`
- fixed checkpoint baselines

Run at least `Medium` and `High` load.

Question answered:
- Do we really need the joint design?

### E4. Checkpoint tradeoff
Compare:
- `No-FT`
- `Fixed-Low-CKPT`
- `Fixed-High-CKPT`
- `Our-System`

Report:
- steady-state goodput / TTFT / TPOT
- failover gap
- replay work
- checkpoint traffic
- host memory footprint

Question answered:
- Is adaptive checkpointing worth its overhead?

### E5. Controller overhead
Measure controller latency per epoch and how it scales with:
- active requests
- queued requests
- number of GPUs if hardware allows
- number of scenarios checked

Also report controller time as a fraction of epoch length.

Question answered:
- Is the controller practical online?

## Required Logs
### Per run
- git commit
- model
- GPU type and GPU count
- epoch length
- workload id
- workload parameter file or config
- load level
- baseline name
- exact policy name and key flags
- fault seed, fault time, and failed GPU
- total runtime

### Per request
- request id
- arrival time
- admitted or rejected
- initial GPU
- checkpoint policy or class
- prompt length
- output length
- TTFT
- TPOT
- completion status
- affected by failure or not
- failover gap
- restore time
- replay tokens / replay time
- resumed GPU id
- SLO violations

### Per epoch
- epoch id
- start and end time
- controller latency
- master solve time
- pooled screen time
- exact recovery time
- number of scenarios checked
- number of cuts
- number of iterations
- queued, active, and admitted request counts
- timeout or fallback event if it happens

## Figures Needed
Please produce these figures or tables first:

1. No-failure steady-state summary table
2. Failure-case summary table
3. Main performance plots: goodput, SLO violation, p95 failover gap
4. Recovery breakdown plot
5. Ablation plot
6. Checkpoint tradeoff plot
7. Controller overhead plot

## Execution Order
Please run in this order:
1. Validate logging and one failure-recovery pilot run
2. Run `Our-System` and `No-FT` on all workloads and load levels
3. Add fixed checkpoint baselines
4. Add routing-only and checkpoint-only ablations
5. Run controller overhead study
6. Generate plots, CSV summaries, and a short written summary

## Deliverables
Please return:
- raw logs
- cleaned CSV summaries
- plotting scripts
- final figures in PDF or PNG
- a short note with anomalies, failed runs, and any baseline limitations

If any baseline is impossible to run with the current code, flag it explicitly instead of silently replacing it.
