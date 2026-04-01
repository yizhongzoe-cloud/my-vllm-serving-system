# Repo Overview: my-vllm-serving-system

基于 vLLM 的 fork，加了**单机多GPU容错LLM serving**。

---

## 一句话汇总

| 分类 | 文件夹 | 来源 |
|---|---|---|
| vLLM 原有 | `benchmarks/`, `cmake/`, `csrc/`, `docker/`, `examples/`, `requirements/`, `tools/`, `vllm.egg-info/` | vLLM |
| vLLM 原有（我们改了一部分） | `vllm/`, `tests/`, `docs/` | vLLM + 我们 |
| 我们全新写的 | `experiments/`, `figures/`, `results/`, `slo_benchmark/`, `slides/`, `papers/`, `data/` | 我们 |

---

## 代码量统计（我们写的部分）

| 模块 | 行数 | 说明 |
|---|---|---|
| `vllm/` 新增FT代码 | **~7,000** | 18个新文件（调度器、Benders求解器、checkpoint、恢复、故障检测等） |
| `vllm/` 修改的文件 | 7个 | 在vLLM已有文件上加FT逻辑（config、engine、worker等） |
| `experiments/` | **~5,300** | 实验框架（运行、分析、画图） |
| `tests/ft/` | **~5,400** | 11个测试文件 |
| `slo_benchmark/` | **~2,800** | 早期benchmark框架（已废弃） |
| `docs/design/` | **~2,100** | 4篇设计文档 |
| **合计** | **~22,600** | |

---

## 按文件夹逐个说明

### `benchmarks/` — vLLM 自带

vLLM的性能benchmark脚本（kernel micro-bench、throughput、latency等），和我们无关。

---

### `cmake/` — vLLM 自带

CMake构建配置，管理C++/CUDA外部依赖编译。

---

### `csrc/` — vLLM 自带

C++/CUDA kernel实现（attention、quantization、MoE、sampling等）。

---

### `data/` — 我们的（已废弃）

早期 slo_benchmark 输出的几个 `metrics_*.json`，现在不用了。

---

### `docker/` — vLLM 自带

Docker镜像构建文件。

---

### `docs/` — vLLM 自带 + 我们的设计文档

大部分是vLLM的文档（usage、features、serving、API等）。

**我们加的**（在 `docs/design/` 下）：

| 文件 | 行数 | 说明 |
|---|---|---|
| `fault_tolerant_llm_serving_idea_summary.md` | 393 | 核心idea：问题形式化、优化目标、约束 |
| `idea_online_periodic.md` | 1013 | 在线版：epoch决策 + 自适应checkpoint + Benders算法 |
| `CHECKPOINT_PROFILING.md` | 454 | Checkpoint profiling设计 |
| `experiment_plan_for_collaborator_short_2_revised.md` | 262 | 实验计划 |

---

### `examples/` — vLLM 自带

vLLM的使用示例（offline inference、online serving、pooling等）。

---

### `experiments/` — 我们全新写的（~5,300行）

实验全流程。

| 文件 | 行数 | 说明 |
|---|---|---|
| `config.yaml` | 171 | 总配置：模型、3种workload、6种baseline、5个实验(E1-E5) |
| `checkpoint_cost_profile.json` | 45 | 实测checkpoint开销（prefill/load/save延迟） |
| `suite.py` | 242 | 实验编排：读config，自动跑所有组合 |
| `run.py` | 1620 | 单次run：启server → 发请求 → 注入故障 → 解析日志 → 存结果 |
| `run_2.py` | 1212 | run.py备份版 |
| `workloads.py` | 199 | trace生成：Poisson/bursty到达 + prompt/output分布 |
| `analyze.py` | 978 | 结果分析 + 画图（goodput、SLO、failover gap等） |
| `profile_checkpoint_costs.py` | 493 | 实测 T_prefill/T_load/T_ckpt |
| `inspect_checkpoint_profile.py` | 296 | 验证cost profile（单调/凸性检查） |
| `run_selected_experiments.sh` | 72 | 选择性跑E1-E5的bash wrapper |

---

### `figures/` — 我们的（analyze.py 生成）

| 子目录 | 内容 |
|---|---|
| `E1_Main/` | goodput曲线、SLO violation、failover gap |
| `E2_Recovery/` | recovery时间拆解 |
| `E3_Ablation/` | 消融实验 |
| `E4_Checkpoint_Tradeoff/` | checkpoint开销 vs 恢复收益 |
| `E5_Controller/` | solver延迟 |
| `E1_Smoke/`, `smoke_run_*/` | 早期调试 |

---

### `papers/` — 我们的

论文草稿/参考材料。

---

### `requirements/` — vLLM 自带

pip依赖（common、cuda、rocm、test等）。

---

### `results/` — 我们的（run.py 生成）

按 `实验/baseline/workload/load/fault/seed/` 层级存储。每个run含：
- `metrics.json` — goodput、TTFT/TPOT百分位、SLO violation率
- `requests.csv` — 每条请求详情
- `epochs.csv` — 每个scheduler epoch
- `recoveries.json` — recovery事件
- `run_meta.json` — 运行元信息

正式实验：`E1_Main/` ~ `E5_Controller/`。还有大量 `smoke_*/` 调试run。

---

### `slides/` — 我们的

组会slides和repo overview（就是这个文件）。

---

### `slo_benchmark/` — 我们的（已被 experiments/ 替代）

早期独立写的benchmark框架，~2,800行。包含：
- `src/benchmark/` — dataset、fault injection、metrics、runner
- `scripts/` — 数据收集、画图、mock数据
- `src/utils/` — server管理

---

### `tests/` — vLLM 自带 + 我们的 FT 测试

vLLM自带很多测试（engine、kernels、models、quantization等）。

**我们加的**（`tests/ft/`，~5,400行）：

| 文件 | 行数 | 说明 |
|---|---|---|
| `test_ft_e2e_serving.py` | 499 | 端到端：启server → 注入故障 → 验证恢复 |
| `test_ft_checkpoint_recovery.py` | 674 | checkpoint恢复 + failover gap验证 |
| `test_ft_system.py` | 644 | 系统级（scheduler + controller + recovery） |
| `test_ft_integration.py` | 312 | FT组件集成 |
| `test_async_ft_checkpoint_flow.py` | 244 | 异步checkpoint流程 |
| `test_checkpoint_cost_model.py` | 413 | cost model单元测试 |
| `test_benders_solver.py` | 502 | Benders求解器单测 |
| `test_benders_integration.py` | 708 | Benders集成测试 |
| `test_centralized_benders.py` | 711 | centralized Benders测试 |
| `test_experiment_analyze.py` | 230 | 实验分析代码测试 |
| `test_experiment_log_parser.py` | 443 | 日志解析测试 |

---

### `tools/` — vLLM 自带

构建/profiling/pre-commit工具。

---

### `vllm/` — vLLM 主代码 + 我们的 FT 扩展

绝大部分是vLLM原代码。我们的改动集中在 `vllm/v1/core/`、`vllm/v1/engine/`、和少量config/worker文件。

#### 我们新增的文件（18个，~7,000行）

**调度器** — `vllm/v1/core/sched/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `ft_scheduler.py` | 547 | FT调度器核心：config + admission/routing/checkpoint/failover |
| `ft_scheduler_impl.py` | 383 | 在base Scheduler上包装注入FT逻辑 |
| `benders_ft_scheduler_impl.py` | 513 | Benders版，用MIP替换贪心admission |

**Benders求解器** — `vllm/v1/core/sched/benders/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `solve_loop.py` | 301 | 主循环：master → 检查故障场景 → cut → 迭代 |
| `master.py` | 204 | Master MIP：admission + routing |
| `recovery_checker.py` | 264 | Subproblem：per-scenario recovery可行性 |
| `cost_tables.py` | 384 | 开销表（prefill/decode/checkpoint/restore/replay） |
| `cuts.py` | 54 | Cut生成（no_good_cut + pool_overload_cut） |
| `__init__.py` | 35 | 包导出 |

**Checkpoint管理** — `vllm/v1/core/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `checkpoint_controller.py` | 440 | 在线自适应：`Δreplay > Δload + λ·Δckpt` → publish |
| `kv_checkpoint_pool.py` | 437 | Host pinned memory pool，存KV checkpoint |
| `checkpoint_cost_model.py` | 137 | 基于profile的开销查询 |

**故障恢复** — `vllm/v1/core/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `recovery_manager.py` | 410 | Failover编排：加载checkpoint → 路由存活GPU → replay |
| `failure_detector.py` | 309 | 心跳监控，HEALTHY→DEGRADED→FAILED→RECOVERING |
| `replica_manager.py` | 384 | 多GPU replica管理 |
| `request_pool.py` | 224 | 中心化请求池，PENDING→ADMITTED→DISPLACED→COMPLETED |

**Engine集成** — `vllm/v1/engine/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `ft_client.py` | 1531 | FT版DP client，处理故障信号做failover |
| `ft_coordinator.py` | 433 | 监控engine健康，心跳超时发布故障事件 |

#### 我们修改的vLLM已有文件（7个）

| 文件 | 改了什么 |
|---|---|
| `vllm/config/scheduler.py` | 加了 `"fault_tolerant"`, `"ft_benders"`, `"ft_benders_centralized"` 策略 |
| `vllm/engine/arg_utils.py` | 加了 `--enable-checkpointing` 等FT CLI参数 |
| `vllm/v1/engine/core.py` | 加了FT scheduler选择和checkpoint controller初始化 |
| `vllm/v1/engine/core_client.py` | 加了failover请求缓存支持 |
| `vllm/v1/engine/utils.py` | FT通信工具 |
| `vllm/v1/worker/gpu_model_runner.py` | checkpoint发布和KV恢复能力 |
| `vllm/v1/worker/gpu_worker.py` | worker侧FT支持 |

---

### `vllm.egg-info/` — 自动生成

Python包元信息，不用管。

---

## 改动分析：改了什么，哪里改动最多

### 总量

vLLM代码改动共 **~9,700行**：
- 新增 18 个文件：+6,990 行
- 修改 13 个已有文件：+2,740 行 / -853 行

### 改动最大的文件（按新增行数排）

| 文件 | +行/-行 | 干了什么 |
|---|---|---|
| `v1/worker/gpu_model_runner.py` | +735/-283 | **改动最多的已有文件**。加了checkpoint发布（KV写到host RAM）和KV恢复（从host RAM load回GPU），是FT系统和GPU交互的底层接口 |
| `v1/engine/core.py` | +581/-78 | Engine初始化加了FT scheduler选择、checkpoint controller创建、故障处理流程 |
| `v1/engine/ft_client.py` | +1531 (新) | **最大的新文件**。FT版DP client：多engine请求分发、故障检测后re-routing、请求缓存用于failover |
| `v1/engine/input_processor.py` | +367/-171 | 输入处理加了FT相关的请求元信息传递 |
| `v1/engine/core_client.py` | +289/-62 | DP client底层加了failover请求缓存 |
| `v1/core/sched/ft_scheduler.py` | +547 (新) | FT调度器核心逻辑 |
| `v1/core/sched/benders_ft_scheduler_impl.py` | +513 (新) | Benders版调度器 |
| `v1/core/checkpoint_controller.py` | +440 (新) | 自适应checkpoint控制器 |
| `v1/core/kv_checkpoint_pool.py` | +437 (新) | Host RAM checkpoint存储池 |
| `v1/engine/ft_coordinator.py` | +433 (新) | Engine健康监控 |
| `v1/core/recovery_manager.py` | +410 (新) | Failover恢复编排 |
| `v1/core/replica_manager.py` | +384 (新) | GPU replica管理 |
| `v1/core/sched/benders/cost_tables.py` | +384 (新) | Benders求解器开销表 |
| `v1/core/sched/ft_scheduler_impl.py` | +383 (新) | FT scheduler包装层 |
| `v1/core/failure_detector.py` | +309 (新) | 心跳故障检测 |
| `v1/core/sched/benders/solve_loop.py` | +301 (新) | Benders主循环 |
| `v1/core/sched/benders/recovery_checker.py` | +264 (新) | Recovery可行性检查 |
| `v1/core/request_pool.py` | +224 (新) | 中心化请求池 |
| `v1/request.py` | +201/-24 | Request对象加了FT字段（checkpoint状态、recovery信息） |
| `v1/core/sched/benders/master.py` | +204 (新) | Master MIP |
| `v1/core/checkpoint_cost_model.py` | +137 (新) | Checkpoint开销模型 |
| `config/scheduler.py` | +123/-2 | 加了FT调度策略枚举 |
| `engine/arg_utils.py` | +114/-69 | 加了FT CLI参数 |
| `v1/core/sched/request_queue.py` | +108/-0 | 请求队列加了FT支持 |
| `v1/engine/__init__.py` | +89/-12 | Engine包加了FT导出 |
| `entrypoints/openai/chat_completion/protocol.py` | +66/-44 | OpenAI API加了FT相关字段 |
| `v1/core/sched/benders/cuts.py` | +54 (新) | Benders cut生成 |
| `v1/worker/gpu_worker.py` | +32/-102 | Worker侧FT支持 |
| `v1/engine/utils.py` | +29/-6 | FT通信工具 |
| `v1/core/sched/output.py` | +6/-0 | 调度输出加FT字段 |

### 主要改了三层

**1. Worker层（底层）** — `gpu_model_runner.py` 改动最大
- 让GPU能把KV cache存到host RAM（checkpoint发布）
- 让GPU能从host RAM恢复KV cache（checkpoint恢复）
- 改动最大因为要深入嵌入vLLM的KV block管理逻辑

**2. Engine层（中间层）** — `ft_client.py`, `core.py`, `core_client.py`, `ft_coordinator.py`, `input_processor.py`
- 多GPU协调、故障信号传递、请求re-routing
- `ft_client.py` 最大（1531行）因为要处理各种failover边界情况（engine挂了、请求在飞、部分完成等）

**3. Scheduler层（上层）** — `ft_scheduler.py`, `benders_ft_scheduler_impl.py`, `benders/*`
- 决策谁进谁出、放哪张GPU、什么时候checkpoint
- 这是paper的核心算法（Benders分解、自适应checkpoint策略）

**一句话：底层改GPU让它会存/恢复KV，中间层改Engine让它会检测故障和re-route请求，上层加Scheduler让它会做聪明的决策。**

---

## 架构图

```
请求进来
  │
  ▼
ft_coordinator ──── 心跳监控每个engine
  │
  ▼
ft_client ──── 负载均衡 + failover re-routing
  │
  ▼
benders_ft_scheduler ──── Benders求解: admission + routing
  ├── master (MIP)
  └── recovery_checker (subproblem per failure scenario)
  │
  ▼
checkpoint_controller ──── 自适应checkpoint
  │                        Δreplay > Δload + λ·Δckpt → publish
  ▼
kv_checkpoint_pool ──── host RAM存KV
  │
  ▼
[GPU故障]
  │
  ▼
failure_detector ──── 心跳超时
  │
  ▼
recovery_manager ──── 加载checkpoint → 路由存活GPU → replay
  │
  ▼
请求继续生成
```
