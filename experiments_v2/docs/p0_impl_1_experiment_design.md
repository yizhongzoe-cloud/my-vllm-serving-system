# P0-impl-1 实验设计：定位 tpot 60ms 的真凶

> **状态**：设计草稿，**未执行**。等 vLLM build (`/tmp/vllm_build.log`) 完成后跑。
> **目的**：用最少的对照实验（每实验 ~2 min），唯一定位"开了 checkpointing 后 decode tpot 从 26 → 60 ms"这 ~35 ms 固定开销的来源。
> **环境**：8 × A5000，build 后用 `CUDA_VISIBLE_DEVICES=4,5,6,7` 避开占用的 GPU 0-3。

---

## 1. 已经从现有数据得到的事实

W1_Chat / Moderate / none cell 上的 4 个 baseline（[results_v2/8B/E1a_Quick/](../../results_v2/8B/E1a_Quick/)）：

| Baseline | 调度器 | Checkpoint | tpot_p50 (ms) | tpot_p95 (ms) | Δ vs No-FT |
|---|---|---|---|---|---|
| **No-FT** | fcfs | ❌ off | **26.2** | 30.0 | baseline |
| **Periodic-Low** | fault_tolerant | 固定每 10 blocks | **62.0** | 80.4 | **+35.8** |
| **Periodic-High** | fault_tolerant | 固定每 1 block | **66.7** | 86.4 | **+40.5** |
| **Our-System** | ft_benders_centralized | adaptive | **60.7** | 76.2 | **+34.5** |

**两个关键观察**：

1. **三个 ckpt-on baseline 全部 ~60 ms**，跟用什么调度器（fault_tolerant vs ft_benders_centralized）和 checkpoint 策略（固定 vs 自适应）都几乎无关。
2. **频率从 10× 增加（Low → High），tpot 只多 4.7 ms（+7.5%）**。

→ "ckpt-on" 这件事本身引入了 **~35 ms 的固定开销**，与频率几乎无关。频率只贡献边际增加。

## 2. 已被现有数据排除的 hypothesis

| 原嫌疑 | 排除依据 |
|---|---|
| Benders solver 自身开销（master MIP / cost table / Python overhead） | Periodic 不用 Benders 也是 62 ms |
| TPOT 约束代数退化导致 over-admission | Periodic 没有 Benders 也是 62 ms |
| decode_capacity cap=10 hardware mismatch | Periodic 不走 decode_capacity_model 路径也是 62 ms |
| Benders solver 在 Python 主线程跑导致 GIL 抢占 | Periodic 没有 Benders 也是 62 ms |
| `solve_loop.py:161` 漏传 `avg_ctx_bucket` | Periodic 不走 solve_loop 也是 62 ms |
| Planning horizon mismatch | 同上 |

## 3. 剩余的 hypothesis（聚焦 checkpoint 路径）

| ID | Hypothesis | 机理 | 频率敏感性 |
|---|---|---|---|
| **H1** | 每个 decode step 都跑了 `_should_checkpoint_*` 决策逻辑，引入固定开销 | controller 在 [checkpoint_controller.py:215-302](../../vllm/v1/core/checkpoint_controller.py#L215-L302) 的策略检查每 step 调用 | **不敏感**（每 step 都跑，与频率无关）|
| **H2** | KV checkpoint pool 的某个 lock / shared state 引入争用 | [kv_checkpoint_pool.py](../../vllm/v1/core/kv_checkpoint_pool.py) 的 `self._lock` 在每次 save/restore/evict 都拿 | 略敏感（频率高时争用更多）|
| **H3** | Pinned host memory allocation 影响 page table / OS scheduler | 第一次启用 ckpt 时分配大量 pinned memory，触发 mlock 或 page faulting | **不敏感**（一次性 alloc）|
| **H4** | Stage 1 GPU clone 在 default stream 上的累积阻塞 | [kv_checkpoint_pool.py:215-220](../../vllm/v1/core/kv_checkpoint_pool.py#L215-L220) | **敏感**（应该随频率线性增长）|
| **H5** | 启用 ckpt 后某个 background thread 持续做 polling / IO，干扰 GPU 计算 | 比如 checkpoint pool eviction、metrics collector、watcher thread | **不敏感** |
| **H6** | `enable_checkpointing: true` 让 vLLM 在 model_executor 层启用了某个 hook（比如每 step 复制 KV 元数据），不在 checkpoint controller 路径上 | 待 read code 验证 | **不敏感** |

**频率敏感性是关键**：现有数据显示频率 10× → tpot +7.5%，所以**主因必须是频率不敏感的 hypothesis**（H1/H3/H5/H6），而 H4 只能解释边际增加部分。

## 4. 实验集 A：判罪 — checkpoint 路径整体

**目的**：confirm "开 ckpt → +35 ms" 这个事实，把开销定位到 `enable_checkpointing` 这个开关上。

| 实验 ID | Baseline 名 | 配置差异 | 预期 tpot_p50 (ms) | 含义 |
|---|---|---|---|---|
| **A0** (existing) | No-FT | scheduling=fcfs, ckpt=off | 26.2 | 已有数据 |
| **A1** (existing) | Our-System | scheduling=ft_benders_centralized, ckpt=adaptive | 60.7 | 已有数据 |
| **A2** | **Our-System-NoCkpt** | 同 A1 但 `enable_checkpointing: false` | **?** | **核心对照** |

**A2 的判定**：

| A2 tpot_p50 | 推断 | 下一步 |
|---|---|---|
| 26-32 ms | 整个 ckpt 路径是元凶（包括 controller decision + pool save + 任何相关 hook）| 进入实验集 B 细分 |
| 38-50 ms | ckpt 路径解释一半，还有 ft_benders_centralized scheduler 自身的少量开销 | 进入实验集 B + C |
| 55-65 ms | ckpt 路径不是元凶；元凶在 ft_benders_centralized scheduler 框架本身 | 进入实验集 C |

> **A2 是整个判罪流程的根节点**，必须先跑。

## 5. 实验集 B：定位 ckpt 路径内的子层（仅当 A2 指向 ckpt 路径时跑）

**目的**：区分 H1/H3/H5/H6（"启用 ckpt" 的固定开销）vs H4（实际 save_checkpoint 调用的边际开销）。

| 实验 ID | Baseline 名 | 配置差异 | 设计意图 |
|---|---|---|---|
| **A1** (existing) | Our-System | adaptive ckpt | 已有：60.7 ms |
| **A2** (above) | Our-System-NoCkpt | `enable_checkpointing: false` | 已规划 |
| **B1** | **Our-System-VeryLowFreq** | `enable_checkpointing: true`，但 `fixed_checkpoint_blocks: 99999`（实际永不触发）| 隔离 "enable=true 的固定开销" vs "实际 save 的边际开销" |
| **B2** | **Our-System-LowFreq** | `fixed_checkpoint_blocks: 100` | 频率扫描点 |
| **B3** | **Our-System-MidFreq** | `fixed_checkpoint_blocks: 20` | 频率扫描点 |
| Periodic-Low (existing) | — | `fixed_checkpoint_blocks: 10` | 已有：62.0 ms |
| Periodic-High (existing) | — | `fixed_checkpoint_blocks: 1` | 已有：66.7 ms |

> 注意 B1/B2/B3 仍然用 `ft_benders_centralized` 调度器（而非 fault_tolerant），保持调度器变量不变。

**B1 的判定（最关键）**：

| B1 tpot_p50 | 推断 | 真凶定位 |
|---|---|---|
| **26-32 ms**（接近 No-FT）| "enable=true 自身没固定开销"。元凶是**实际 save_checkpoint 调用**的累积，但单次开销小。这与现有 P-Low/P-High 相差小的现象矛盾 → 不太可能 | H4 |
| **55-65 ms**（接近 Our-System）| **"enable=true 自身就有 ~35 ms 固定开销，与是否真的 save 无关"**。这是最可能的 outcome，与现有 P-Low/P-High 数据一致 | H1/H3/H5/H6 之一 |
| 38-50 ms | 部分固定 + 部分边际 | 混合 |

**频率扫描的判定**（B2/B3 + 已有数据）：画一条 tpot vs `1/fixed_checkpoint_blocks` 的曲线：

```
tpot_p50 (ms)
   80 ┤              .. P-High (66.7, freq=1.0/blk)
   70 ┤        .. B3 (?, freq=0.05)
      │   .. B2 (?, freq=0.01)
   60 ┤   ╱── P-Low (62.0, freq=0.10)  ← B1 (?, freq=~0)
      │
   50 ┤
      │
   30 ┤── No-FT (26.2)
```

- **如果曲线在 freq=0 处的截距 ~60 ms** → 几乎所有开销是固定的（H1/H3/H5/H6），频率几乎无影响
- **如果曲线在 freq=0 处的截距 ~26 ms** → 所有开销随频率线性，元凶是 H4
- **如果截距 ~45 ms** → 一半固定一半边际

## 6. 实验集 C：scheduler 框架对比（仅当 A2 显示 scheduler 也有贡献时跑）

**目的**：分离"`fault_tolerant` scheduler 框架开销" vs "`ft_benders_centralized` scheduler 框架开销"，确认 [ft_scheduler_impl.py](../../vllm/v1/core/sched/ft_scheduler_impl.py) 包装层是否引入额外开销。

| 实验 ID | Baseline 名 | 配置差异 | 设计意图 |
|---|---|---|---|
| **C1** | **Adaptive-Only** | `scheduling_policy: fault_tolerant`, `enable_checkpointing: true`, `fixed_checkpoint_blocks: 0` (adaptive) | 与 Our-System 比，差异只在 scheduler |
| **C2** | **Adaptive-Only-NoCkpt** | 同 C1 但 `enable_checkpointing: false` | 与 No-FT 比，差异只在 scheduler 框架（无 ckpt 干扰）|

**判定**：

| 比较 | tpot 差异 | 推断 |
|---|---|---|
| C1 vs Our-System | < 5 ms | 调度器框架不重要（已被现有数据预期）|
| C1 vs Our-System | > 10 ms | Benders 调度器有额外开销（与现有 Periodic 数据矛盾，需要追溯）|
| C2 vs No-FT | < 3 ms | `fault_tolerant` 框架本身无开销 |
| C2 vs No-FT | > 5 ms | `fault_tolerant` 框架引入开销（独立于 checkpoint）|

## 7. 实验集 D：Profile-level ground truth（仅 A/B/C 都不能解释时做）

**目的**：用 nsys 在 forward kernel 时间轴上直接看出 stream blocking 模式。

| 实验 ID | 操作 |
|---|---|
| **D1** | `nsys profile -t cuda,nvtx -o /tmp/our-system.nsys-rep` 包住 30s 的 Our-System / W1_Chat / Moderate / none |
| **D2** | 同 D1 但跑 No-FT 作为对照 |
| **D3** | `nsys-ui /tmp/our-system.nsys-rep` 看时间轴 |

**关注点**：
- default stream (stream 7) 上有没有 `cudaMemcpyAsync` / `cudaMemcpy` / `aten::clone` 出现在 forward kernel 之间
- forward step 之间的 gap 时长
- copy stream 的 utilization

## 8. 推荐执行顺序

```
等 vLLM build 完成
        │
        ▼
[Step 1] 跑 A2 (Our-System-NoCkpt)
        │
        ├──→ tpot ≈ 26-32 ms ────→ ckpt 路径是元凶 → [Step 2]
        │
        ├──→ tpot ≈ 55-65 ms ────→ ckpt 路径不是元凶 → [Step 3]
        │
        └──→ 中间值          ────→ 两边都做 → [Step 2 + Step 3]

[Step 2] 跑 B1, B2, B3 三个频率扫描点
        │
        ▼
        画 tpot vs ckpt frequency 曲线
        │
        ├──→ 截距 ~60 ms（频率不敏感）→ 元凶是 "enable=true 的固定开销"
        │                                 → [Step 4: read code 找 enable=true 路径]
        │
        ├──→ 截距 ~26 ms（频率敏感）→ 元凶是 stage 1 clone 累积
        │                              → 写 stage 1 fix patch
        │
        └──→ 截距 ~45 ms（混合）   → 两个都修

[Step 3] 跑 C1, C2 scheduler 框架对比
        │
        ▼
        [Step 4: 读 vllm/v1/core/sched/ft_scheduler_impl.py]

[Step 4] 根据 [Step 2/3] 结果，要么 read code 定位固定开销，要么写 fix patch

[Step 5] (可选) D1/D2 nsys profile 做 ground truth confirm
```

## 9. 总成本

| Step | Cells | 时间/cell (含 server startup) | 总时间 |
|---|---|---|---|
| A2 | 1 | ~2.5 min | ~2.5 min |
| B1, B2, B3 | 3 | ~2.5 min | ~7.5 min |
| C1, C2 | 2 | ~2.5 min | ~5 min |
| D1, D2 | 2 (with profiler overhead) | ~5 min | ~10 min |
| **总计** | **8 cells (+ 4 existing)** | — | **~25 min wall clock** |

如果只跑最关键路径（Step 1 → Step 2）：~10 min。

## 10. config_8b_diag.yaml 需要的扩展

需要在 [config_8b_diag.yaml](../config_8b_diag.yaml) 里新增以下 baselines：

```yaml
baselines:
  Our-System: { existing }
  Our-System-NoCkpt: { existing, A2 }

  # ★ 实验集 B：频率扫描
  Our-System-VeryLowFreq:    # B1
    scheduling_policy: "ft_benders_centralized"
    enable_checkpointing: true     # ← 关键：保持 true
    max_gpu_failures: 1
    fixed_checkpoint_blocks: 99999  # ← 实际永不触发

  Our-System-LowFreq:        # B2
    scheduling_policy: "ft_benders_centralized"
    enable_checkpointing: true
    max_gpu_failures: 1
    fixed_checkpoint_blocks: 100

  Our-System-MidFreq:        # B3
    scheduling_policy: "ft_benders_centralized"
    enable_checkpointing: true
    max_gpu_failures: 1
    fixed_checkpoint_blocks: 20

  # ★ 实验集 C：scheduler 框架对比
  Adaptive-Only:             # C1
    scheduling_policy: "fault_tolerant"
    enable_checkpointing: true
    max_gpu_failures: 1
    fixed_checkpoint_blocks: 0  # adaptive

  Adaptive-Only-NoCkpt:      # C2
    scheduling_policy: "fault_tolerant"
    enable_checkpointing: false
    max_gpu_failures: 1
    fixed_checkpoint_blocks: 0

experiments:
  E_P0_Diag_A:               # 1 cell, ~2.5 min
    description: "P0-impl-1 Step 1: confirm checkpoint path is the culprit"
    baselines: ["Our-System", "Our-System-NoCkpt"]
    workloads: ["W1_Chat"]
    load_levels: ["Moderate"]
    faults: ["none"]

  E_P0_Diag_B:               # 3 cells, ~7.5 min
    description: "P0-impl-1 Step 2: ckpt frequency scan"
    baselines: ["Our-System-VeryLowFreq", "Our-System-LowFreq", "Our-System-MidFreq"]
    workloads: ["W1_Chat"]
    load_levels: ["Moderate"]
    faults: ["none"]

  E_P0_Diag_C:               # 2 cells, ~5 min
    description: "P0-impl-1 Step 3: scheduler framework comparison"
    baselines: ["Adaptive-Only", "Adaptive-Only-NoCkpt"]
    workloads: ["W1_Chat"]
    load_levels: ["Moderate"]
    faults: ["none"]
```

## 11. 数据收集 + 分析脚本

**输出位置**：`results_v2/8B_diag/E_P0_Diag_{A,B,C}/{baseline}/W1_Chat/Moderate/none/42/metrics.json`

**对比命令**（实验跑完后）：
```bash
for exp in A B C; do
  echo "=== E_P0_Diag_$exp ==="
  for baseline_dir in results_v2/8B_diag/E_P0_Diag_$exp/*/; do
    baseline=$(basename "$baseline_dir")
    f="$baseline_dir/W1_Chat/Moderate/none/42/metrics.json"
    [ -f "$f" ] && python3 -c "
import json
m = json.load(open('$f'))
print(f'  {\"$baseline\":<25}: tpot_p50={m[\"tpot_p50_ms\"]:6.1f} ms  tpot_p95={m[\"tpot_p95_ms\"]:6.1f} ms  goodput={m[\"goodput\"]:6.1f}')
"
  done
done
```

**期望输出形态**（A2 是核心数据点）：

```
=== E_P0_Diag_A ===
  Our-System              : tpot_p50=  60.7 ms  tpot_p95=  76.2 ms  goodput= 205.7
  Our-System-NoCkpt       : tpot_p50=  ??.? ms  tpot_p95=  ??.? ms  goodput= ???.?    ← 看这个
=== E_P0_Diag_B ===
  Our-System-VeryLowFreq  : tpot_p50=  ??.? ms  ...
  Our-System-LowFreq      : tpot_p50=  ??.? ms  ...
  Our-System-MidFreq      : tpot_p50=  ??.? ms  ...
=== E_P0_Diag_C ===
  Adaptive-Only           : tpot_p50=  ??.? ms  ...
  Adaptive-Only-NoCkpt    : tpot_p50=  ??.? ms  ...
```

## 12. 风险与限制

| 风险 | 缓解 |
|---|---|
| 单 seed (42)，方差未知 | A2 是 binary 判断，差距 35 ms 远大于 seed 噪音 (~3 ms)，足够 |
| 90s run 比正式 300s 短，warmup 后稳态时间只有 75s | W1_Chat Moderate rps=1.0 → 75 个请求，统计够用 |
| `fixed_checkpoint_blocks: 99999` 在代码里是不是真的不触发 | 需要 grep 验证，最差情况是触发但很少 |
| Build 完成后第一次跑 server 可能慢（kernel cache 冷启动） | 给前两个 cell 留 buffer 时间，或先跑一次 throwaway 预热 |
| GPU 4-7 上启动 vLLM 用 `CUDA_VISIBLE_DEVICES=4,5` 是否被 [run.py](../run.py) / [suite.py](../suite.py) 正确传递 | 需要 verify suite.py 把 env 透传给子进程；如果不传需要在 shell 层面 export |

## 13. 不在本设计范围内

以下问题**不**用这套实验来回答（避免 scope creep）：

- ❌ A6000 vs A5000 hardware mismatch 的影响（属于 P1-env-1）
- ❌ failover gap 的 contention 折扣（属于 P2）
- ❌ Benders solver 的迭代收敛性（已被现有数据排除）
- ❌ W4_Mixed completion 异常（属于另一条诊断线，与 tpot 无直接关系）

这些都先放一放，集中精力解决 "tpot 60ms" 这一个核心症状。
