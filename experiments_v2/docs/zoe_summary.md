# 文档总览

整理所有和 FT 实验相关的 md 文档，标注状态和建议。

---

## experiments_v2/ 下的文档

| 文件 | 内容 | 状态 | 建议 |
|---|---|---|---|
| [full_experiment_spec.md](../../experiments_v2/full_experiment_spec.md) | 完整实验说明：7 个实验的矩阵、图表、预期趋势、已知问题、Tiny 诊断、换模型标准步骤 | **最新** | **保留**，主文档 |
| [decode_first_capacity_plan.md](../../experiments_v2/decode_first_capacity_plan.md) | decode-first 容量模型的设计方案：问题分析、新约束、profile 方法、论文描述、方案评估 | **最新** | **保留**，设计参考 |
| [decode_first_capacity_todo.md](../../experiments_v2/decode_first_capacity_todo.md) | decode-first 的实现 TODO 清单（8 个 Phase，~40 个任务） | **过时** | **可删**，代码已全部实现完毕 |
| [tiny_experiment_spec.md](../../experiments_v2/tiny_experiment_spec.md) | Tiny 实验（12 runs）的设置说明 | **过时** | **可删**，已被 full_experiment_spec.md 的 Tiny 诊断章节覆盖 |
| [progress_slides.md](../../experiments_v2/progress_slides.md) | 组会 slides（15 页），含 Tiny 实验图表和根因分析 | **过时** | **需更新**，基于调参前的旧数据（Benders 33% 拒绝），图表需要用最新结果重新生成 |
| [figures/E1a_Tonight/slides.md](../../experiments_v2/figures/E1a_Tonight/slides.md) | tonight 54 runs 实验结果分析 | **过时** | **可删**，调参前跑的，结论（Benders 仍然过度拒绝）已被推翻 |

## papers/ 下的文档

| 文件 | 内容 | 状态 | 建议 |
|---|---|---|---|
| [experiment_design.md](../../papers/experiment_design.md) | 面向论文的实验设计：硬件、模型、数据集、SLO 校准、14 个图表设计、执行顺序 | **最新** | **保留**，论文实验章节的蓝本 |
| [experiment_analysis.md](../../papers/experiment_analysis.md) | 5 篇相关论文（AdaServe/TurboSpec/AdaSpec/SOLA/SLOs-Serve）的实验设置对比分析 | **最新** | **保留**，独立参考 |
| [experiment_implementation_todo.md](../../papers/experiment_implementation_todo.md) | v2 实验框架的实现 TODO（9 个 Phase，~40 个任务） | **过时** | **可删**，代码已全部实现完毕 |

## docs/design/ 下的文档

| 文件 | 内容 | 状态 | 建议 |
|---|---|---|---|
| [fault_tolerant_llm_serving_idea_summary.md](fault_tolerant_llm_serving_idea_summary.md) | 核心 idea：问题形式化、优化目标、约束 | **最新** | **保留** |
| [idea_online_periodic.md](idea_online_periodic.md) | 在线版：epoch 决策 + 自适应 checkpoint + Benders 算法 | **最新** | **保留** |
| [CHECKPOINT_PROFILING.md](CHECKPOINT_PROFILING.md) | Checkpoint profiling 设计 | **最新** | **保留** |
| [experiment_plan_for_collaborator_short_2_revised.md](experiment_plan_for_collaborator_short_2_revised.md) | 实验计划 | **最新** | **保留** |

---

## 汇总

| 操作 | 文件 |
|---|---|
| **保留（8 个）** | [full_experiment_spec.md](../../experiments_v2/full_experiment_spec.md)、[decode_first_capacity_plan.md](../../experiments_v2/decode_first_capacity_plan.md)、[experiment_design.md](../../papers/experiment_design.md)、[experiment_analysis.md](../../papers/experiment_analysis.md)、及 docs/design/ 下 4 个设计文档 |
| **可删（4 个）** | [decode_first_capacity_todo.md](../../experiments_v2/decode_first_capacity_todo.md)、[tiny_experiment_spec.md](../../experiments_v2/tiny_experiment_spec.md)、[figures/E1a_Tonight/slides.md](../../experiments_v2/figures/E1a_Tonight/slides.md)、[experiment_implementation_todo.md](../../papers/experiment_implementation_todo.md) |
| **需更新（1 个）** | [progress_slides.md](../../experiments_v2/progress_slides.md)——等 8B 实验跑完后用最新数据重新生成 |

---

## 改进计划汇总

散落在各文档中的改进方向，集中整理在这里。

### 系统 + 实验改进（按优先级排序）

| 优先级 | 方向 | 说明 | 出处 |
|---|---|---|---|
| ~~done~~ | ~~**Solver 异步化 + 缩短 epoch**~~ | ~~solver 移到后台线程 + epoch 从 100ms 缩到 20ms。~~ **已实现。** TPOT 没变，但 Goodput +7%，TTFT -11%。 | ft_client.py |
| ~~done~~ | ~~**Checkpoint 拷贝异步化**~~ | ~~`save_checkpoint` 里 3 处 CPU 阻塞 + `collective_rpc` 同步等待。~~ **已修：(1) GPU 端改用 `record_event`+`wait_event` 跨 stream 同步，CPU 零阻塞。(2) checkpoint RPC 改成 fire-and-forget（ThreadPoolExecutor 后台执行），metadata 在发 RPC 前乐观更新。** 最终结果：Goodput 174.8→215.6（+23%，接近 No-FT 的 223.1），SLO violation 18.9%→4.9%，完成率恢复到 98.8%。 | `kv_checkpoint_pool.py`, `core.py` |
| **P4** | **扩展到 dp=4** | 8B E3 结果表明 dp=2 下 routing 没有收益（Benders-Only < Adaptive-Only）。dp=4 下 routing 选择空间大，Benders 优势才能体现。 | 8B E3 实验结果 |
| **P5** | **更强的 Benders cut** | 当前只用 no-good cut。可加 pool-overload cut、lifted cut，加速 solver 收敛。 | [slides/progress_slides.md](../../slides/progress_slides.md) Slide 13 |
| **P5** | **SLO 改为应用驱动的固定值** | calibrate.py 目前用 5×TTFT_base 算 SLO。应改为固定值。不紧急——当前 SLO 问题是 TPOT 本身太高，不是阈值设定问题。 | 对话中讨论 |
| ~~done~~ | ~~**诊断 FT 运行时开销来源**~~ | ~~根因是 `max_gpu_failures` 被钳制到 0 + EngineCore admission 过度拒绝。~~ **已修。** | ft_scheduler_impl.py, benders_ft_scheduler_impl.py |
| ~~done~~ | ~~**Decode-first 容量模型**~~ | ~~用 decode slot + residual prefill 替代旧的串行时间约束。~~ **已实现。** 后续可升级：`w_j = α + β·ctx_j`、线性拟合替代查表。 | [decode_first_capacity_plan.md](../../experiments_v2/decode_first_capacity_plan.md) |

### 实验层面

| 方向 | 说明 | 出处 |
|---|---|---|
| **扩展到 4-8 GPU** | dp=4 时故障丢 25% 容量（vs dp=2 的 50%），恢复场景更现实，routing 的收益更明显。 | [slides/progress_slides.md](../../slides/progress_slides.md) Slide 13 |
| ~~**真实数据集**~~ | ~~已从合成模板文本换成 ShareGPT/CNN-DailyMail/Alpaca。~~ **已完成。** 后续可加 Azure LLM traces 做到达模式验证。 | [experiment_design.md §1.4](../../papers/experiment_design.md) |
| **多 seeds** | 当前 1 seed 看趋势，需要补充 3-5 seeds 报告 mean ± std。 | [slides/progress_slides.md](../../slides/progress_slides.md) Slide 13 |
| ~~**SLO 校准**~~ | ~~用 `calibrate.py` 自动找饱和 RPS + SLO baseline，不再手动拍常数。~~ **已实现。** | [full_experiment_spec.md](../../experiments_v2/full_experiment_spec.md) 换模型标准步骤 |

### 论文/Future Work

| 方向 | 说明 | 出处 |
|---|---|---|
| **多节点容错** | 当前系统是单机多 GPU | [experiment_design.md §6](../../papers/experiment_design.md) |
| **多 GPU 同时故障** | 系统目前支持 `max_gpu_failures=1` | [experiment_design.md §6](../../papers/experiment_design.md) |
| **投机解码集成** | 正交技术，可以和 FT 系统结合 | [experiment_design.md §6](../../papers/experiment_design.md) |
| **Checkpoint 压缩** | 模型级 KV cache 压缩，降低 checkpoint 大小和 restore 时间 | [experiment_design.md §6](../../papers/experiment_design.md) |
| **Prefill/Decode 分离部署** | 正交架构（参考 DistServe），可以和 FT 结合 | [experiment_design.md §6](../../papers/experiment_design.md) |
| **真实 GPU 硬件故障** | 目前用 SIGKILL 模拟，真实故障的检测特征可能不同 | [experiment_design.md §6](../../papers/experiment_design.md) |
