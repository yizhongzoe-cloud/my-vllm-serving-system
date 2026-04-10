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

### 单 seed (s42), W1_Chat / Heavy / F2_Mid

| Strategy | Recovery | Ckpt save | Goodput | TTFT>2s | Comp% | Raw tok/s |
|---|---|---|---|---|---|---|
| **No-FT** | 丢弃 displaced | 无 | **319.5** | 18.8% | 98.5 | 330.9 |
| NoFT-Restart | Re-prefill prompt | 无 | 290.1 | 29.4% | 100 | 337.3 |
| NoFT-Reprefill | Extended prompt | 无 | 291.3 | 29.0% | 100 | 337.3 |
| Our-System-Restart | Re-prefill prompt | Non-blocking | 47.9 | 87.4% | 98.9 | — |
| Our-System-Reprefill | Extended prompt | Non-blocking | 39.5 | 88.5% | 99.6 | — |
| **Our-System** | **KV reload** | **Non-blocking** ⭐ | **299.1** | **25.3%** | **100** | **337.3** |

### 关键发现

1. **Raw throughput 完全一样 (337 tok/s)**：所有"保住 displaced reqs"的方案产出相同 tokens。差异只在 TTFT latency → SLO violation
2. **No-FT 的 raw throughput 反而更低 (331)**：因为丢了 7 reqs 的 ~2500 tokens
3. **Our-System 比 NoFT-Restart/Reprefill 少 17 个 TTFT violations**：因为 KV reload 走 I/O 不走 compute
4. **SLO violation 几乎 100% 来自 TTFT > 2000 ms**：TPOT 和 gap 几乎不违反

## 6. 剩余 gap 分析

Our-System (299) vs No-FT (319) = **6.4% gap**

| 来源 | 占比 | 可优化？ |
|---|---|---|
| Post-fault batch 膨胀 (20→26 reqs) → step 慢 7ms | ~3-4% | ❌ 保住 displaced reqs 的 inherent cost |
| FT scheduler Python wrapper overhead (+0.5 ms/step) | ~1.5% | ❌ 需要 C++ 重写 |
| Stochastic variance (active@fault 不同) | ~1-2% | ❌ 不可控 |

**Phase 8 (NONBLOCK + FAST_TMPFS + FAST_CHUNK) 是 practical limit。**

## 7. Committed code changes

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

## 8. TODO List（按优先级）

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
