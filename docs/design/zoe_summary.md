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
| ~~done~~ | ~~**诊断 FT 运行时开销来源**~~ | ~~TPOT 飙升的根因是 `max_gpu_failures` 被钳制到 0（EngineCore 的 dp_size=1 触发了保护逻辑），导致 checkpoint 从不 publish、solver 大量 infeasible。~~ **已修：删掉钳制逻辑，`max_gpu_failures` 保持原值。** | ft_scheduler_impl.py, benders_ft_scheduler_impl.py |
| **P2** | **Checkpoint 异步拷贝** | KV checkpoint 从 GPU→Host 的拷贝改成非阻塞（`async_copy=True`），和 decode 计算重叠执行。代码已支持 `async_copy` 参数，但默认关闭。**如果 P1 确认 checkpoint 是瓶颈，这个直接解决。** | `kv_checkpoint_pool.py` 的 `save_checkpoint()` |
| **P2** | **Solver 异步化** | 把 Benders solver 移到后台线程，当前 decode step 用上一轮的 admission 决策。**如果 P1 确认 solver 阻塞是瓶颈，这个直接解决。** | [idea_online_periodic.md §13](idea_online_periodic.md) |
| **P3** | **SLO 改为应用驱动的固定值** | calibrate.py 目前用 5×TTFT_base 算 SLO，等于系统给自己出考卷。应改为固定值：TTFT=2000ms, Gap=3000ms。5 分钟改完。 | 对话中讨论 |
| **P3** | **负载百分比降低** | 当前 25/40/55% of No-FT 饱和点，8B 上 FT baselines 在 Heavy 已过载。改为 15/25/35%。5 分钟改完。 | 8B E3 实验结果 |
| **P4** | **LP 松弛替代 MIP** | Master problem 的 0/1 变量 LP 松弛，求解速度提升 10-100 倍。但不是当前 TPOT 高的主因。 | 对话中讨论 |
| **P4** | **更强的 Benders cut** | 当前只用 no-good cut。可加 pool-overload cut、lifted cut，加速 solver 收敛。 | [slides/progress_slides.md](../../slides/progress_slides.md) Slide 13 |
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
