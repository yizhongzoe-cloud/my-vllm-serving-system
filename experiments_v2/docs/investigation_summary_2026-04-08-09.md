# Investigation Summary — 2026-04-08 ~ 04-09

> 从 e1a_quick_diagnosis.md 的 1600+ 行调查记录中提炼的结构化总结。

## 1. 起点

Our-System（ft_benders_centralized + adaptive checkpoint + KV reload recovery）在 W1_Chat/Heavy/F2_Mid 上的 goodput 只有 **~120 tok/s**，而 No-FT (fcfs) 达到 **~320 tok/s**。差距 ~200 tok/s，约 62%。

## 2. 找到的 Bug（已修复）

### Bug 1: AsyncScheduler 缺失（commit `da5c8f25e`）
- **现象**: TPOT ~60 ms（No-FT 只有 ~30 ms）
- **根因**: ft_scheduler_impl.py + benders_ft_scheduler_impl.py hardcode `Scheduler(...)` 而不是 `AsyncScheduler`
- **后果**: batch_queue=2 pipeline 下 `num_output_placeholders` 没递增，50% running reqs 每步被 skip
- **修复**: 检测 `async_scheduling=True` 时用 `AsyncScheduler`
- **效果**: TPOT 从 ~60 ms → ~30 ms

### Bug 2: SLO metric 定义错误（commit `522c2a9dd`）
- **现象**: admitted-but-failed requests 不计入 SLO violation
- **修复**: 把 admitted + failed 也算 SLO violation
- **效果**: metric 更准确

### Bug 3: Checkpoint pipeline 同步阻塞（commit `2044810d0`）
- **现象**: goodput 120 tok/s（vs No-FT 320）
- **根因**: `_maybe_ft_checkpoint()` line 740 用 `_ft_ckpt_future.result()` **阻塞**等上一步 checkpoint RPC。Worker 端 RPC ~70 ms（torch.save + fsync × 14 reqs），step time ~30 ms → API server step rate 从 30/s 跌到 14/s
- **修复**: `FT_CKPT_NONBLOCK=1` — 用 `future.done()` peek 而不是 `.result()`
- **效果**: goodput 120 → **~250-300 tok/s**（+111%）

### Bug 4: torch.save pickle 开销（commit `1aa319f92`）
- **现象**: NONBLOCK fix 导致 5% completion drop（跳过 checkpoint cycles）
- **根因**: Worker 端 `torch.save` 用 pickle 序列化 KV tensors，~14 ms/chunk
- **修复**: `FT_FAST_CHUNK_FORMAT=1` — raw bytes + struct header 替代 pickle，~8.6 ms/chunk
- **效果**: completion 95% → **100%**（worker RPC 快到 NONBLOCK 不需要 skip cycles）

## 3. 发现但不是 Bug 的 Issues

### Benders solver 收敛失败率 98%
- **根因**: decode_capacity_profile_8b.json 是 A6000 dp=1 配的（default=10），实际硬件 A5000 dp=2（running batch ~25）。Master MIP `25 > 10` 直接 infeasible。
- **修了 profile 反而更差**: Benders 开始工作但 over-reject（goodput 124 → 43, completion 31%）
- **结论**: Benders 在 greedy fallback 模式反而更好。**不修。**

### KV pressure throttle（无效）
- 加了 `_kv_pressure_too_high()` 在 admission 入口，threshold 90%
- 实测无效——因为 admission 已经被 base scheduler 自然处理
- **留着但不用。**

## 4. 尝试过但失败的优化（12 个 phase）

### Recovery 路径优化（Finding #5）

| Mode | Goodput | Completion | 为什么失败 |
|---|---|---|---|
| **reload** (默认 KV restore) | **124.2** | 100% | baseline |
| restart (re-prefill prompt) | 47.9 | 98.9% | Compute-bound, 跟 new reqs 抢 prefill capacity |
| reprefill (extended prompt) | 39.5 | 99.6% | 同上但 prompt 更大，更慢 |
| drop (丢弃 displaced) | 117.0 | 94.8% | 证明 recovery 本身不是 overhead 来源 |

→ **KV reload 是最优 recovery 策略**

### Framework overhead 优化（Phase 6-17）

**攻击目标 = Bug 3 (checkpoint pipeline 同步阻塞)** — goodput 从 120 → 320 的 200 tok/s gap，drop mode 证明不来自 recovery 路径，Our-System-NoCkpt 证明来自 checkpoint controller。cProfile 定位到 `_maybe_ft_checkpoint()` line 740 的 `.result()` 阻塞 + worker 端 `torch.save` + `fsync`。

| Phase | 优化 | 攻击的 bottleneck | 结果 | 为什么 |
|---|---|---|---|---|
| 6 | FT_FAST_TMPFS_WRITE (skip fsync) | Worker: fsync on tmpfs (无意义 I/O) | +5 tok/s | fsync 只占 RPC 时间的 ~5%，大头是 torch.save |
| **7** | **FT_CKPT_NONBLOCK** | **API server: `.result()` 阻塞等 RPC** | **+130 tok/s** ⭐ | 消除了 Bug 3 的阻塞。step rate 从 14/s → 30/s |
| **8** | **FT_FAST_CHUNK_FORMAT** | **Worker: torch.save pickle 开销 (14ms/chunk)** | **+17 tok/s + 100% completion** ⭐ | raw bytes (8.6ms/chunk) 让 worker RPC 够快，NONBLOCK 不再 skip cycles |
| 9 | Pinned buffer pool | Worker: pin_memory() syscall (~ms/call) | ❌ reverted | Data race: async GPU→host copy 未同步，buffer 被复用前旧 copy 未完成 |
| 10 | Background publish (ThreadPool) | Worker: file write 阻塞 RPC return | -28 tok/s | GIL contention: bg thread 的 Python 工作跟 main thread 抢 GIL |
| 11 | Inline manifest (v2 chunk format) | Worker: 3 file ops → 1 file op per save | within variance | Phase 8 后 file ops 不在 critical path（NONBLOCK 不等 RPC） |
| 12 | API server cProfile | 🔍 **profiling** | 🔍 发现 cleanup hot spot | `_cleanup_shared_checkpoint` = 35% API server CPU (shutil.rmtree 9ms/unlink) |
| 13 | Async cleanup (run_in_executor) | API server: rmtree 阻塞 event loop | net zero | API server main thread 只 13% CPU 利用率 — 不是 bottleneck |
| 14 | Step timing (FT_STEP_TIMING) | 🔍 **profiling** | 🔍 发现 enqueue prep 占 62% | section 2 (schedule + exec launch) = 18.5ms, GPU wait 只 37% |
| 15 | Dynamic post-fault batch cap | Scheduler: post-fault batch 膨胀 (20→26) | -49 tok/s | Throughput math 走不通: step_rate × batch = net 更低 tokens/sec |
| 16 | Skip Benders solver | API server: solver bg thread CPU | within variance | Solver 已 98% greedy fallback，CPU 占用可忽略 |
| 17 | Deprioritize migrated reqs | Scheduler: migrated 排在 new reqs 前面 | -19 tok/s, -6% comp | 推到队尾 → migrated 等太久 → timeout 丢失 28 reqs |

**Phase 7+8 是唯一有效的两步**，对应 Bug 3 的两个组成部分（API server 阻塞 + worker serialize 慢）。Phase 9-17 攻击的都是 **Bug 3 修完后不再是 bottleneck 的位置** — API server idle (13% CPU), scheduler 只 0.6ms/step, file ops 已经 off critical path。

### Scheduling 策略优化（Phase 18-19）

| Phase | 优化 | 结果 | 为什么 |
|---|---|---|---|
| 18a | NoFT-Restart baseline | 290.1 | 最轻量 FT, 比 Our-System 差因为 re-prefill 占 compute |
| 18b | NoFT-Reprefill baseline | 291.3 | Extended prompt ≈ original prompt (差距小) |
| 19a | Incremental recovery (batch=5, delay=10s) | 296.0 | Net neutral: delay 不够长让前一批 drain |
| 19b | Incremental recovery (batch=3, delay=25s) | 278.5, 98% comp | 更差: reqs timeout 在长 delay 里 |

## 5. 最终对比表

### 5.1 原始 single-seed 对比表(2026-04-08 ~ 09,历史记录)

> ⚠️ **READ THIS — 这张原始表被发现是"数据拼盘",不是 apples-to-apples 对比**
>
> 2026-04-10 做 3-seed validation 时,trace 了每一行数据的 git_commit / 目录,发现 6 行数据**跨越 4 个不同 commit 和 3 个不同实验目录**,不是同一次 run 的结果。详见 [5.2 节 — Data provenance audit](#52-data-provenance-audit-2026-04-10)。
>
> 特别是 **Our-System-Restart=47.9** 和 **Our-System-Reprefill=39.5** 这两行实际上来自 **phase 7+8 fix 之前** (`commit 522c2a9dd`,04-08 20:39) 的一次 ad-hoc ablation,当时 [checkpoint pipeline 还是阻塞的 (Bug 3)](#bug-3-checkpoint-pipeline-同步阻塞commit-2044810d0) — 实际上测的是"有 bug 的 Our-System 的两种 recovery mode",跟同表其他 phase 8 后的数据不可比。`actual_runtime=401s`(其他行 319s)和 `active@fault=34`(其他行 7-8)也印证了当时系统已 backlogged。
>
> 下面这张表**保留原样作为历史记录**。正确的 3-seed 对比表见 [5.3 节](#53-phase-8-条件-3-seed-对比表-2026-04-10)。

| Strategy | Recovery | Ckpt save | Goodput | TPOT p50 | TPOT p95 | TTFT>2s | Comp% | Raw tok/s |
|---|---|---|---|---|---|---|---|---|
| **No-FT** | 丢弃 displaced | 无 | **319.5** | 43.0 | 63.3 | 18.8% | 98.5 | 330.9 |
| NoFT-Restart | Re-prefill prompt | 无 | 290.1 | 45.1 | 64.9 | 29.4% | 100 | 337.3 |
| NoFT-Reprefill | Extended prompt | 无 | 291.3 | 45.0 | 64.6 | 29.0% | 100 | 337.3 |
| Our-System-Restart ❌ | Re-prefill prompt | **(Bug 3 stale data)** | 47.9 | 85.5 | 140.4 | 87.4% | 98.9 | — |
| Our-System-Reprefill ❌ | Extended prompt | **(Bug 3 stale data)** | 39.5 | 86.5 | 180.4 | 88.5% | 99.6 | — |
| **Our-System** ⚠️ | **KV reload** | **Non-blocking** ⭐ | **299.1** | **47.5** | **65.2** | **25.3%** | **100** | **337.3** |

- ❌ = **stale data**,来自 bug-阻塞版 commit 522c2a9dd (04-08 20:39),不是 phase 8 条件
- ⚠️ = Benders solver 在这一行实测**不在工作**(A6000 dp=1 profile default=10 → 98% infeasible → greedy fallback)。profile 校准后 Benders 真工作时同 seed 测出 187.4,见 [section 6 "Benders solver 工作时的额外 gap"](#benders-solver-工作时的额外-gap-新增2026-04-10-e1a_3seed)

### 5.2 Data provenance audit (2026-04-10)

| Doc 行 | Goodput | 数据源目录 | commit | 提交时间 | actual_rt | active@fault |
|---|---|---|---|---|---|---|
| No-FT | 319.5 | `8B_overnight_2026-04-09/02_noft_baseline/` | `8a7a98c59` | 04-09 04:19 | 319.6s | 7 |
| NoFT-Restart | 290.1 | `8B_overnight_2026-04-09/phase18_noft_restart/` | `3446fa203` | 04-09 16:47 | 319.9s | 8 |
| NoFT-Reprefill | 291.3 | `8B_overnight_2026-04-09/phase18_noft_reprefill/` | `3446fa203` | 04-09 16:47 | 319.9s | 7 |
| **Our-System-Restart** | **47.9** | `8B_recovery_modes/restart/` | **`522c2a9dd`** | **04-08 20:39** ❌ | **401.8s** | **34** |
| **Our-System-Reprefill** | **39.5** | `8B_recovery_modes/reprefill/` | **`522c2a9dd`** | **04-08 20:39** ❌ | **395.3s** | **27** |
| Our-System | 299.1 | `8B_overnight_2026-04-09/phase8_fast_chunk/reload_s42/` | `563a3827f` | 04-09 13:55 | 319.0s | 8 |

`522c2a9dd` 是"fix(metrics): admitted-but-failed requests count as SLO violations",比 `2044810d0` (NONBLOCK + FAST_TMPFS) 早约 20 小时 — 也就是说 Restart/Reprefill 那两行当时 checkpoint pipeline 还在阻塞。

### 5.3 Phase 8 条件 3-seed 对比表 (2026-04-10)

> **Re-run conditions**: 用 `experiments_v2/phase8_repro_3seed.sh` 在 **相同 profile** (`decode_capacity_profile_8b_phase8.json` = A6000 dp=1 default=10) **相同 env vars** (`FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1`) 下跑了所有 6 个 baseline × 3 seeds (42/123/456) 在 W1_Chat/Heavy/F2_Mid。
>
> **当前 commit**: `044463ae6`+ (已包含 `b09d82bd4` 的 greedy fallback fix)。数据目录 `results_v2/8B/E1a_phase8_repro/`。

**3-seed mean ± stdev**:

| Strategy | Recovery | Ckpt save | **Goodput mean±std** | Comp% | TTFT p50 | TTFT p95 | SLO viol |
|---|---|---|---|---|---|---|---|
| **No-FT** | 丢弃 displaced | 无 | **347.6 ± 23.5** | 98.4±0.4 | 423±65 | 5052±4709 | 13.2%±8.0 |
| NoFT-Restart | Re-prefill prompt | 无 | 279.8 ± 113.8 | 100.0 | 1445±1788 | 7581±5199 | 28.8%±28.3 |
| NoFT-Reprefill | Extended prompt | 无 | 277.2 ± 105.5 | 100.0 | 1332±1558 | 7403±5064 | 29.8%±26.9 |
| Our-System-Restart | Re-prefill prompt | Non-blocking | 253.8 ± 97.1 | 100.0 | 4545±6939 | 10612±7455 | 37.2%±23.5 |
| Our-System-Reprefill | Extended prompt | Non-blocking | 230.9 ± 85.0 | 100.0 | 5832±9031 | 12491±9085 | 41.4%±21.0 |
| **Our-System** ⚠️ | **KV reload** | **Non-blocking** ⭐ | **258.3 ± 76.2 (n=10)** | 100.0 | — | — | — |

⚠️ **Our-System s42 deterministic CUDA bug**: seeds 42 and 1337 deterministically crash in this profile with `torch.AcceleratorError: CUDA error: device-side assert triggered` at [gpu_model_runner.py:291 `async_copy_ready_event.synchronize()`](../../vllm/v1/worker/gpu_model_runner.py#L291), triggered ~3 seconds after fault injection by the KV restore → next decode step async race. `CUDA_LAUNCH_BLOCKING=1` sidesteps the crash (classic kernel race signature). **2/12 seeds crash = 17% crash rate.** See [overnight_2026-04-11.md §6](overnight_2026-04-11.md#6-phase-4--phase-8-profile-our-system-14-seeds-3-crashed).

**Per-seed raw** (all cells: W1_Chat/Heavy/F2_Mid, phase 8 profile):

```
baseline                seed  goodput   comp%   TTFT p50   SLO%
No-FT                    42    322.6    98.5       406     19.9
No-FT                   123    351.1    98.7       495     15.4
No-FT                   456    369.1    98.0       368      4.3

NoFT-Restart             42    311.4   100.0       452     23.6
NoFT-Restart            123    153.5   100.0      3509     59.3
NoFT-Restart            456    374.4   100.0       374      3.4

NoFT-Reprefill           42    292.2   100.0       485     28.8
NoFT-Reprefill          123    165.1   100.0      3130     57.1
NoFT-Reprefill          456    374.4   100.0       381      3.4

Our-System-Restart       42    271.6   100.0       594     37.0
Our-System-Restart      123    149.0   100.0     12557     60.8
Our-System-Restart      456    340.8   100.0       484     13.8

Our-System-Reprefill     42    236.0   100.0       696     43.3
Our-System-Reprefill    123    143.4   100.0     16259     61.5
Our-System-Reprefill    456    313.2   100.0       541     19.5

Our-System               42    CRASH (CUDA assert, 3 attempts: 164.2 / 164.9 / 89.4[blocking])
Our-System              123    133.9   100.0     13986     62.1
Our-System              456    307.5   100.0       492     19.5
Our-System              789    258.8   100.0        —       —
Our-System              888    361.4   100.0        —       —
Our-System              999    251.2   100.0        —       —
Our-System              1337   CRASH (CUDA assert, 37.6% completion, 276 failed)
Our-System              2024   195.6   100.0        —       —
Our-System              3141   308.1   100.0        —       —
Our-System              5678   249.1   100.0        —       —
Our-System              6666   172.6   100.0        —       —
Our-System              9999   344.2   100.0        —       —
```

**Our-System 10-seed clean mean**: 258.3 tok/s, std 76.2, stderr ±24.1 (excluding 42/1337 crashes).
This falls **just below** the doc's original single-seed s42 = 299.1 by ~0.5 stderr — 299.1 was plausible but not representative. The "Our-System is only 6.4% away from No-FT" narrative from the original phase 8 report is wrong: real distance is **(347.6 − 258.3) / 347.6 = 25.7% gap** at 10-seed / 3-seed confidence.

### 5.4 关键发现 (修订版)

1. **Raw throughput 在 NoFT-Restart/Reprefill + Our-System-Restart/Reprefill 之间基本持平**(都在 330 tok/s 左右,seed 间方差很大)。差异几乎全部来自 TTFT。
2. **No-FT 在 3-seed mean 下仍然领先约 26%** (347.6 vs Our-System 10-seed 258.3)。原始 table 的 "6.4% gap" 只在 s42 单点成立。
3. **Our-System-Restart/Reprefill 的真实数字是 253.8 / 230.9** — **不是** doc 原始表里的 47.9 / 39.5。后者是 Bug 3 未修前的数据,跟后面的 phase 8 数据拼在一起属于误导。
4. **Seed 之间方差巨大**: NoFT-Restart 三个 seed 跨越 153 / 311 / 374,NoFT-Reprefill 跨越 165 / 292 / 374。W1_Chat/Heavy/F2_Mid 这个 cell 对 "active_requests_at_fault" 极度敏感,任何结论都不该基于单 seed。Section 5.6 用 8-seed 重新 validate 了这个 cell。
5. **Our-System ⚠️ Benders solver "不工作"的前提已不再成立**: phase8_repro 重跑时,每个 seed 的 Benders solver 都 100% converged(25/25, 25/25, 12/12 次),0 次 infeasible。原因是 `b09d82bd4` 修掉了 solver rejection → 就算 MIP infeasible solver 也不再 reject req,但 solver 在 feasible 时会做 admission/routing 决策。所以 phase 8 conclusion("Our-System 的 solver 在 greedy fallback")在当前代码上**不可复现**。
6. **Our-System (KV reload) 的真实 10-seed mean 是 258.3 ± 76.2** (excluding 2 CUDA crashes)。s42 和 s1337 在 phase 8 profile 下 deterministic CUDA device-side assert,17% crash rate,真 bug 但非 paper-blocking(见 [overnight_2026-04-11.md §6](overnight_2026-04-11.md#6-phase-4--phase-8-profile-our-system-14-seeds-3-crashed))。

### 5.5 Benders solver net value (Phase 1, 2026-04-11)

**36-run pair comparison**: Our-System with `FT_SKIP_SOLVER=1` (solver OFF,走 greedy 派发) vs `E1a_3seed` Our-System (solver ON),同一 commit + A5000 profile + 同 seeds (42/123/456) + 全部 12 cells。完整数据见 [overnight_2026-04-11.md §3](overnight_2026-04-11.md#3-phase-1--ft_skip_solver-ablation-benders-net-value)。

| Aggregation | Solver ON | Solver OFF | Δ |
|---|---|---|---|
| Overall mean (12 cells) | 197.2 | 203.2 | **+3.0%** (OFF wins) |
| Cells where OFF ≥ ON | — | — | **10 / 12** |
| Significantly different (\|Δ\| > 1σ combined) | — | — | 2 cells, both OFF wins |

**结论**: Benders solver 在 8B + dp=2 + chat workload 上是 **marginally net negative** — 关掉反而快 3%,且 10/12 cells 领先。没有一个 cell solver ON 有统计显著优势。这 validate 了 [section 3 — Benders solver 收敛失败率 98%](#benders-solver-收敛失败率-98) 当时的 prediction:"Benders 在 greedy fallback 模式反而更好"。

### 5.6 Hard-cell 8-seed aggregate (Phase 2, 2026-04-11)

**把 W1_Chat/Heavy/F2_Mid 和 W4_Mixed/Heavy/F2_Mid 的 3-seed (E1a_3seed) 扩到 8-seed** (原 3 + 新 5 = 111/222/333/555/777),A5000 profile。完整数据见 [overnight_2026-04-11.md §4](overnight_2026-04-11.md#4-phase-2--hard-cell-8-seed-aggregate)。

**W1_Chat/Heavy/F2_Mid (n=8)**:

| Baseline | mean ± std | stderr | vs Our-System |
|---|---|---|---|
| **NoFT-Reprefill** 🏆 | **268.3 ± 73.8** | ±26.1 | **+103.3** (+62%) |
| Our-System | 165.0 ± 49.4 | ±17.5 | baseline |
| Periodic-High | 156.8 ± 30.2 | ±10.7 | −8.2 |

Gap NoFT-Reprefill vs Our-System = 103 tok/s / 31.4 combined stderr = **~3.3σ → 强显著**。

**W4_Mixed/Heavy/F2_Mid (n=8)**:

| Baseline | mean ± std | stderr | vs Our-System |
|---|---|---|---|
| **NoFT-Reprefill** 🏆 | **252.0 ± 16.0** | ±5.7 | +34.6 (+16%) |
| Periodic-High | 243.3 ± 15.8 | ±5.6 | +25.9 |
| Our-System | 217.4 ± 11.1 | ±3.9 | baseline |

Gap = 35 tok/s / 6.9 combined stderr = **~5σ → 非常强显著**,而且 W4 variance 小得多,说明这是 robust trend 而非 outlier。

**结论**: **NoFT-Reprefill 在两个 hard cell 都显著优于 Our-System**,尤其是 W1_Chat/Heavy/F2_Mid (+62%)。这不是 lucky seed — 8 个 seed 加起来的 stderr 已经把 noise 压到 <30 tok/s,gap 远超 noise floor。简单的 "fault_tolerant scheduler + no ckpt + reprefill recovery" 在这个 cell 上比 ft_benders_centralized + adaptive ckpt + KV reload 快得多。

## 6. 剩余 gap 分析

> ⚠️ **原始分析 (2026-04-09) 基于 single-seed s42 Our-System=299.1 做的 "6.4% gap" 拆解已被 overnight_2026-04-11 的 143-run 实验 **强烈反驳**。原始 "6.4% gap" 不存在。2026-04-11 的 5-seed 全 12-cell 对比确认真实 gap 是 **~10-26%**,而且 NoFT-Reprefill 比 Our-System 好,不是反过来。**

### 6.1 原始分析 (phase 8 当晚,s42 单点) — 历史记录

Our-System (299, **当时 Benders 在 greedy fallback + 单 seed**) vs No-FT (319) = **6.4% gap**

| 来源 | 原估占比 | 现证据 |
|---|---|---|
| Post-fault batch 膨胀 (20→26 reqs) → step 慢 7ms | ~3-4% | 仍然真实,inherent cost |
| FT scheduler Python wrapper overhead (+0.5 ms/step) | ~1.5% | 仍然真实,framework overhead |
| **Stochastic variance** (active@fault 不同) | ~1-2% | ❌ **严重低估** — 实际 std 50-80 tok/s on Heavy cells |

**原来的 "Phase 8 是 practical limit" 结论不成立**。Phase 8 的 299.1 是一个 ~75 std 的分布里的 1 个 lucky draw,不是 practical limit。

### 6.2 2026-04-11 definitive gap analysis

**Phase 3 — 5-seed main comparison (12 cells × 3 baselines, A5000 profile)** 见 [overnight_2026-04-11.md §5](overnight_2026-04-11.md#5-phase-3--5-seed-main-comparison-12-cells--3-baselines):

| Baseline | Overall mean goodput (12 cells, 5 seeds) |
|---|---|
| **NoFT-Reprefill** 🏆 | **214.2** |
| Periodic-High | 199.5 |
| Our-System | 195.3 |

- **NoFT-Reprefill 领先 Our-System 约 10%** (跨所有 12 cells average)
- **NoFT-Reprefill 赢每一个 cell** — 不是只在 hard cell 赢
- **Periodic-High 介于两者之间**

**Phase 8 profile 下的 Our-System 10-seed mean = 258.3 ± 76.2**,对比 phase 8 原始 single-seed 299.1 低 ~14%,而且 76 的 std 说明 "299.1 ± 76 = [223, 375]" 都是可能值 — 299.1 不是"真实性能"。

**关键是 gap 的来源不再是"inherent cost"**:
1. **Benders solver 本身** 是 net negative -3% (Phase 1 ablation, [section 5.5](#55-benders-solver-net-value-phase-1-2026-04-11))
2. **KV reload recovery** 本身比简单 reprefill 差 —— 不是理论差,是实测差,hard cell 上 +62% 差距,全 sweep 上 +10% 差距
3. **Checkpoint controller** 开销 — phase 7+8 修复过后变成 near-zero,不是 gap 的来源
4. 剩下的 noise 源 (active@fault 等) 已经被 5-seed + 8-seed aggregation 平均掉了

### 6.3 Paper pivot 建议

当前 "Our-System = ft_benders_centralized + adaptive ckpt + KV reload" 的 contribution 在 8B + dp=2 + chat workload 上**全部组件都是 net negative**:

| 组件 | 实测净值 | 证据 |
|---|---|---|
| `ft_benders_centralized` scheduler (Benders MIP) | **−3%** | [Section 5.5](#55-benders-solver-net-value-phase-1-2026-04-11) |
| Adaptive checkpoint + KV reload recovery | **−10% ~ −62%** vs NoFT-Reprefill | [Section 5.6](#56-hard-cell-8-seed-aggregate-phase-2-2026-04-11) + [overnight_2026-04-11.md §5](overnight_2026-04-11.md#5-phase-3--5-seed-main-comparison-12-cells--3-baselines) |
| Checkpoint pipeline optimization (NONBLOCK + FAST_TMPFS + FAST_CHUNK) | **+正面** (phase 7+8 fix 消除 framework overhead) | [overnight_2026-04-09.md](overnight_2026-04-09.md) |

也就是说:**只有 framework overhead 的优化是真实的 contribution**,但这只是让 Our-System 不至于比 No-FT 慢很多,它没有让 Our-System 比 simpler baselines 更快。Paper 必须 pivot:

**选项 A — Scale up**: 跑 70B + dp=4/8 + W2_Summary (long-context prefill-heavy workload)。理论上 KV reload 的 I/O 优势在 checkpoint << re-prefill 时才显 — 当前 8B chat 不满足这条件。推荐优先跑这个 setup 看是否 sell。

**选项 B — Reframe recovery**: 不 claim "faster than reprefill",而 claim "preserves token stream determinism" —— NoFT-Reprefill 在 fault 时 re-prefill 生成的 suffix tokens 跟 fault 前 streamed 的 tokens 可能不一样(因为 RNG state 丢失了),Our-System 通过 KV reload 保证 suffix 跟 stream 前缀严格 consistent。把 contribution 从 "performance" 改成 "correctness guarantee"。

**选项 C — Gate the solver**: Benders 只在 `dp ≥ 4` 或 `load > 0.7` 时激活,默认 greedy。保留分析框架,不 tank 小规模性能。

**选项 D — Null result + 分析框架**: 老实 report "在 8B + dp=2 上 KV reload 没 pay off",把 contribution 定位成 "fault-tolerant serving 的系统测试框架 + ablation 方法论" 而不是 "最快的方案"。

详细讨论见 [overnight_2026-04-11.md §8](overnight_2026-04-11.md#8-implications-for-paper)。

## 7. 三大 component 深度分析 + 改进方案

Our-System 的三个核心 component (`ft_benders_centralized` scheduler、adaptive checkpoint controller、KV reload recovery) 在实验数据下都是 net negative,但每个的"坏法"不同。以下是逐个的 root cause + 改进方案 ranking。

### A. Benders Solver (ft_benders_centralized scheduler)

#### A.1 当前实现

- **位置**: [vllm/v1/core/sched/benders/solve_loop.py:52-346](../../vllm/v1/core/sched/benders/solve_loop.py#L52-L346) + [master.py](../../vllm/v1/core/sched/benders/master.py)
- **MIP 结构** (centralized,EngineCore 每 ~100ms 解一次):
    ```
    maximize  ∑_j  G_j · y_j                         # y_j = 1 if admit req j
    subject to:
      ∑_j d_j + checkpoint_overhead_sec ≤ H_dec     # decode capacity per replica
      ∑_j p_j                          ≤ H_pre     # prefill capacity
      ∑_j run_mem_bytes_j              ≤ M_cap     # GPU memory
      ∑_r x_{j,r} = y_j                             # 1 req → 1 replica
      (SLO feasibility checks: TTFT, TPOT, gap)
      Benders cuts from recovery infeasibility scenarios
    ```
- **Greedy fallback** ([benders_ft_scheduler_impl.py:451-470](../../vllm/v1/core/sched/benders_ft_scheduler_impl.py#L451-L470)): solver 返回 None 时按 `generation_len` 降序 batch-admit,等价于"无 solver"
- **设计意图**: 在 planning horizon `H=1.0s` 内,选最能产 tokens 的 subset,并把它们路由到 replicas 让 fault 时 recovery 可行

#### A.2 实测 net −3% 的根因

1. **目标函数跟 SLO metric 不对齐**:
    - Solver 最大化 admitted goodput `∑ G_j·y_j`,但 paper metric 是 SLO-satisfied goodput
    - Solver 为了 theoretical optimal admission 会积累 queue → 被延后的 reqs TTFT 崩
    - 证据: Phase 1 显著 cells 集中在 `W1/Moderate/F2_Mid` 和 `W4/Heavy/F2_Mid` (SLO 敏感区域)

2. **Planning horizon H=1.0s 太长**:
    - Solver 每 100ms 一次决策,但 H=1.0 意味着 planning 未来 1 秒的 admission
    - 1 秒内 arrival rate 方差巨大,`G_j` 预测基本是噪声

3. **Cost model 跟实际不匹配**:
    - [cost_tables.py:62-70](../../vllm/v1/core/sched/benders/cost_tables.py#L62-L70) 用线性模型估 prefill/decode cost
    - 实际是 chunked prefill + async scheduling,真实 step time 跟 solver `d_j/p_j` 估计差异巨大
    - 结果: capacity feasibility 约束变成噪声约束

4. **Solver 决策跟 recovery 路径解耦**:
    - Recovery 走 `recovery_manager` 独立逻辑 (选最少 loaded engine),跟 solver 的 `x_{j,r}` 路由决策不一致
    - solver "分散 reqs 让 recovery 可行" 的意图没被 enforced

#### A.3 Benders 改进方案 (ROI 排序)

| # | 改进 | 预期 | 难度 | 修复的根因 |
|---|---|---|---|---|
| **A1** | 把目标函数换成 SLO-aware: `G_j → G_j · exp(-queue_wait/ttft_slo)` | 对齐 metric → 恢复 solver value | ⭐⭐ | 根因 1 |
| **A2** | Shorten `ft_planning_horizon` 1.0 → 0.3 (一行 config) | 减少噪声决策 | ⭐ | 根因 2 |
| **A3** | Gate solver: 只在 `load > 0.7 and pending > K` 激活 | 低负载走 greedy | ⭐ | 根因 3 |
| **A4** | Solver 只做 routing,admission 走 FIFO greedy | 解耦 admit / route | ⭐⭐ | 根因 1 |
| **A5** | 丢弃 Benders,只保留 replica_manager routing | Accept 现实 | ⭐ | All |

---

### B. Adaptive Checkpoint (checkpoint_controller)

#### B.1 当前实现

- **位置**: [vllm/v1/core/checkpoint_controller.py:265-300](../../vllm/v1/core/checkpoint_controller.py#L265-L300)
- **已有的 economic policy** (比想象中聪明):
    ```python
    # 每个 decode step,对每个 req 评估:
    stable_full_tokens = (num_computed_tokens // block_size) * block_size
    published_tokens = num_checkpointed_tokens

    if stable_full_tokens <= published_tokens:
        return False    # 没有新 full block,skip (L270-279)

    # Cost model decision
    should_publish = replay_saved_sec > load_cost + λ · checkpoint_cost
    ```
- 有 "new block guard" 和 "economic cost-benefit" 决策,**不是无脑 per-step save**
- **设计意图**: 只在 "expected replay cost saved" > "checkpoint cost" 时 save,理论上 Pareto 最优

#### B.2 实测 net negative 的根因

1. **Cost model 参数未经 calibration**:
    - `replay_saved_sec` 基于 `prefill_throughput=4000 tok/s` ([config_8b.yaml:17](../config_8b.yaml#L17)),实测 A5000 dp=2 Heavy 下 5000-7000 tok/s
    - `load_cost` 基于 `ft_load_bandwidth=10 GB/s` ([L19](../config_8b.yaml#L19)),实际 pinned memory → GPU 15-20 GB/s
    - `checkpoint_cost` profile 是 A6000 dp=1 标定,没在 A5000 dp=2 重标
    - 结果: `should_publish` breakeven point 全偏

2. **Per-step CPU overhead 分摊不均**:
    - 每 step 对每个 req evaluate cost model → O(N_running_reqs) Python overhead
    - 14 reqs × 30 steps/sec × ~50µs/eval = 21ms/sec 纯 eval 开销
    - **这是 no-fault cell 1-3% 劣势的主要来源**

3. **Uncovered suffix 仍然存在**:
    - Economic policy 在 decode 初期 (短 output) 避免 save,late save 追不上
    - Recovery 时 `uncovered_tokens` 还有 50-200 tokens 要 replay
    - Re-prefill 占 prefill capacity → 拖慢整个 post-fault batch

4. **λ (`checkpoint_lambda`) 默认值 1.0 不对**:
    - 默认 `risk-neutral`,但 fault 稀有 → 过度倾向"不存"
    - 应该 `λ < 1` (risk-averse,更频繁 save)

#### B.3 Adaptive checkpoint 改进方案 (ROI 排序)

| # | 改进 | 预期 | 难度 | 修复的根因 |
|---|---|---|---|---|
| **B1** | 重标 cost model 参数到 A5000 dp=2 (profile run + config update) | Fix no-fault 1-3% | ⭐ | 根因 1 |
| **B2** | Batch evaluate cost model (1 call per step 而非 N) | −21ms/sec overhead | ⭐⭐ | 根因 2 |
| **B3** | Decouple trigger: 每 10 decoded tokens 才评估一次 | 降低 eval 频率 10× | ⭐ | 根因 2 |
| **B4** | `checkpoint_lambda: 1.0 → 0.3` (risk averse) | 更频繁 save,uncovered suffix 更小 | ⭐ | 根因 4 |
| **B5** | Delta checkpoint format (增量 save) | 结构性解决 per-step save cost | ⭐⭐⭐ | 根因 2+3 |
| **B6** | Speculative save: 在 idle decode steps overlap save | 跟 decode 并行 | ⭐⭐⭐ | 根因 2 |

---

### C. KV Reload Recovery

#### C.1 当前实现

- **路径概览**:
    1. Fault → `recovery_manager.handle_failure()` (daemon thread)
    2. 对每个 displaced req 算 `uncovered_tokens = num_computed - num_checkpointed`
    3. `ReplicaManager.route_request_for_failover()` 选 target engine
    4. `readmit_request()` 设 `num_checkpointed_tokens = tokens_restored`
    5. Scheduler alloc blocks → 进入 `_ft_pending_restores` 队列
    6. `core.py _process_ft_pending_restores()` 在 execute_model 前发 `restore_kv_blocks` RPC
    7. Worker `_restore_shared_checkpoint()` 从 /dev/shm load → GPU
    8. Patch `scheduler_output.num_scheduled_tokens` → 跳过已 restored 的 prefix
    9. **Uncovered suffix 走 re-prefill path**,跟 new reqs 抢 prefill capacity
- **设计意图**: Host memory KV 比 re-compute 便宜 (I/O vs compute),保留 decoded tokens → preserve stream continuity

#### C.2 实测输给 NoFT-Reprefill 的根因

1. **Uncovered suffix 跟 new reqs 抢 prefill capacity**:
    - 即使 80% KV 被 reload,剩 20% uncovered 还是要 re-prefill
    - **NoFT-Reprefill 反而更规则**: 100% 的 token 一起走 chunked prefill,vLLM 能高效 batch 处理
    - KV reload 打破 regularity: 一半 reqs 做 CPU/IO restore,一半做 compute replay → mixed mode 更低效

2. **Restore 是 sequential 的,阻塞 decode**:
    - [core.py:586-589](../../vllm/v1/engine/core.py#L586-L589) per-req 发 RPC
    - [gpu_model_runner.py:6977](../../vllm/v1/worker/gpu_model_runner.py#L6977) per-call stream sync
    - 14 reqs × ~5ms sync = 70ms blocking per step
    - ✅ **A/B 已验证修法**: `FT_ASYNC_RESTORE=1` 在 s42 测出 **+25% goodput, −41% TTFT p50**

3. **`num_checkpointed_tokens` 的 granularity 是 block (16 tokens)**:
    - Uncovered suffix 向下 round 到 block boundary → 多 0-15 tokens replay cost
    - Short output 场景下,可能 50% 的存的 KV 永远不会被用

4. **Host→GPU copy 不是瓶颈,stream sync 是**:
    - 14 × 100MB 在 PCIe 4 x16 (20 GB/s effective) 是 5ms
    - 但 14 次 `stream.synchronize()` × ~0.5-1ms round-trip = 7-14ms 被 sync 吃
    - 这就是 `FT_ASYNC_RESTORE` 优化的直接收益

5. **Block table 重建 cost**:
    - Restore 时 allocate `N_blocks * len(target_block_ids)` entries 更新 block_table tensor
    - Python overhead 吃 event loop time

#### C.3 KV reload 改进方案 (ROI 排序)

| # | 改进 | 预期 | 难度 | 修复的根因 | 状态 |
|---|---|---|---|---|---|
| **C1** | **Async pipelined restore (`FT_ASYNC_RESTORE=1`)** | **+25% goodput** (s42 实测) | ⭐⭐ | 根因 2+4 | ✅ **已实施 + A/B 验证中** |
| **C2** | Selective recovery: drop near-done reqs (`remaining < 50`) | −batch 膨胀,+5-10% | ⭐ | post-fault batch 膨胀 | 未做 |
| **C3** | Fast-path for small uncovered (`uncovered < 32 → 直接 reprefill`) | 减少 mixed-mode overhead | ⭐⭐ | 根因 1 | 未做 |
| **C4** | Delta + incremental checkpoint,目标 `uncovered ≈ 0` | 消除 replay path | ⭐⭐⭐ | 根因 1+3 | 未做 |
| **C5** | Pinned buffer pool (fix phase 9 race) | save+restore 省 pin_memory cost | ⭐⭐ | 根因 5 | 未做 |
| **C6** | Batch multi-req restore RPC (N reqs 1 RPC) | 消除 IPC overhead | ⭐⭐ | 根因 2 | 未做 |
| **C7** | Prefix cache integration (用 vLLM prefix cache 代替 host checkpoint) | 消除 no-fault save overhead | ⭐⭐⭐⭐ | All | 未做 |

---

### 统一 root cause: paper 的三个前提都不成立

Our-System 真正的问题不是 3 个 component 各自 broken,而是它们**集体基于错误的假设**:

| 假设 | Our-System 预设 | 8B + dp=2 + chat 实际情况 |
|---|---|---|
| **Admission 值得优化** | Benders: 选最优 subset 能提 goodput | 系统几乎不被 admission bound;admit 决策 value 很低 |
| **Checkpoint 成本值得省** | Adaptive: 只在 net positive 时 save | Fault 概率低,但 per-step eval 是 certain overhead |
| **KV reload 比 reprefill 便宜** | 避开 compute,走 I/O 快 | 1k prompt 的 prefill 只有 30ms,跟 100MB PCIe copy 差不多 |

**三个前提都需要更 demanding 的 workload (大模型 / 长 prompt / 高 fault rate) 才成立**。当前 8B + dp=2 + 1k prompt 是"太简单"的 regime。

---

### 综合改进方案 (按紧迫程度分阶段)

#### Phase 1 — 短期补救 (1-2 天,保住 paper)

| # | 动作 | 预期 goodput 收益 | 难度 |
|---|---|---|---|
| 1 | ✅ **C1 — FT_ASYNC_RESTORE** (A/B 进行中) | +25% (s42 实测) | ⭐⭐ |
| 2 | **C2 — Selective recovery** (10 行改动) | +5-10% | ⭐ |
| 3 | **A2 — Shorten planning horizon 1.0 → 0.3** (一行 config) | +1-3% | ⭐ |
| 4 | **A3 — Gated solver** (`load > 0.7` 才激活) | 消除 solver −3% 拖累 | ⭐ |

**Phase 1 目标**: Our-System W1/Heavy/F2 goodput 从 165 → 200+ (match Periodic-High)

#### Phase 2 — 中期改造 (1 周,救回叙事)

| # | 动作 | 目的 |
|---|---|---|
| 5 | **B1 — 重标 cost model 到 A5000 dp=2** | 消除 no-fault 1-3% 劣势 |
| 6 | **B2 — Batch cost eval** | 消除 per-step eval overhead |
| 7 | **A1 — SLO-aware objective** | 让 solver 真的 help |
| 8 | **C3 — Fast-path for small uncovered** | 消除 mixed-mode 低效 |

**Phase 2 目标**: Our-System overall 195 → 215 (match NoFT-Reprefill mean)

#### Phase 3 — 结构改造 (2-3 周)

| # | 动作 | 目的 |
|---|---|---|
| 9 | **B5 + C4 — Delta checkpoint 组合** | 根治 save cost + uncovered suffix |
| 10 | **C7 — Prefix cache integration** | 最终形态,no-fault save cost = 0 |

**Phase 3 目标**: Our-System 超越 NoFT-Reprefill,paper story 成立

#### Phase 4 — 如果 Phase 1-3 都没救回来

- **Scale up to 70B + W2_Summary**: 去"前提成立"的 regime 测
- **Reframe as correctness**: 测 token stream determinism,改 claim 从 performance → correctness

---

### Phase 1 实施 checklist

- [x] C1 FT_ASYNC_RESTORE worker + core 实现 — commit pending
- [x] A/B test on W1_Chat/Heavy/F2_Mid × 3 seeds — running (s42 confirmed +25%)
- [ ] C2 selective recovery 代码改动 (10 行 on [ft_client.py recovery loop](../../vllm/v1/engine/ft_client.py))
- [ ] A2 `ft_planning_horizon: 1.0 → 0.3` in [config_8b.yaml:20](../config_8b.yaml#L20)
- [ ] A3 gated solver (复用 FT_SKIP_SOLVER 机制)
- [ ] Re-run Phase 2 hard-cell 3-baseline × 5-seed 验证 Phase 1 累积收益

## 8. Committed code changes

| Commit | 类型 | 内容 |
|---|---|---|
| `da5c8f25e` | 🐛 fix | AsyncScheduler wrap |
| `522c2a9dd` | 🐛 fix | SLO metric: admitted-but-failed |
| `ec533cd72` | 🧹 chore | Demote per-step logs (INFO→DEBUG) |
| `f22104557` | ✨ feat | FT_RECOVERY_MODE env var (reload/restart/reprefill/drop) |
| `8a7a98c59` | ✨ feat | FT_DISABLE_SNAPSHOTS env var |
| `2044810d0` | ⭐ perf | **FT_CKPT_NONBLOCK + FT_FAST_TMPFS_WRITE** |
| `1aa319f92` | ⭐ perf | **FT_FAST_CHUNK_FORMAT** (raw bytes vs torch.save) |
| `5f45a1bbf` | 🧪 test | FT_BG_PUBLISH (tested negative) |
| `90e2db8eb` | 🧪 test | FT_INLINE_MANIFEST (tested negative) |
| `675bcf31c` | 🔧 tool | FT_API_PROFILE_SECONDS + FT_ASYNC_CLEANUP |
| `c936a44bc` | ✨ feat | NoFT-Reprefill baseline + FT_SKIP_SOLVER + FT_DEPRIORITIZE_MIGRATED + FT_POST_FAULT_MAX_SEQS |

### 推荐的 env var 组合

```bash
FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1
```

## 9. TODO List（按优先级）

### 🔴 论文 blocking

- [ ] **3-seed validation of phase 8** — 验证 299 不是 lucky seed (~21 min)
- [ ] **3-seed validation of NoFT-Reprefill** — 论文对比表需要 variance bounds (~21 min)
- [ ] **Full ablation matrix** — W1/W2 × Light/Moderate/Heavy × none/F2_Mid × 3 seeds (几小时)
- [ ] **Update paper thesis** — "in-flight preservation vs TTFT latency trade-off"
- [ ] **Decide FT_CKPT_NONBLOCK default** — 3-seed 数据稳定后改成默认 ON

### 🟡 算法优化（future work）

- [ ] **Selective recovery** — 只恢复 remaining_output > threshold 的 displaced reqs，放弃快完成的（减少 batch 膨胀）
- [ ] **Partial KV restore** — 只 restore 后半段 KV blocks，前面 re-prefill（减少 KV footprint per migrated req）
- [ ] **Adaptive checkpoint frequency** — KV > 80% 时降低 save 频率（减少 post-fault checkpoint pressure）
- [ ] **Prefix cache recovery** — 利用 vLLM prefix cache 代替 host memory checkpoint（零 per-step save overhead）
- [ ] **Multi-engine recovery routing** — dp>2 时把 migrated reqs 分散到多个 surviving engines

### 🟢 代码清理

- [ ] Revert KV pressure throttle patches（确认无效）
- [ ] Add requirements/ft.txt 到 install instructions
- [ ] Calibrate decode_capacity_profile for A5000 dp=2（让 Benders 在正确 profile 下跑一次）
- [ ] 清理 working tree 中的失败实验代码（Solution 4 lazy reload 等）
