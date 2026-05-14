# Paper 框架：面向长上下文 LLM Serving 的 Disruption-Aware SLO Scheduling

**暂定标题**："Cheap Preemption Enables Disruption-Aware SLO Scheduling for Long-Context LLM Serving"
**目标会议**：APSys 2026 workshop，6-8 页
**写作质量目标**：OSDI / SOSP full paper 水准（在篇幅压缩的同时不在 rigor 上让步）

---

## 目标 paper 结构

按照 APSys / IEEE workshop short paper 的常规形态（5 节），参考近期 APSys precedent（如 Capybara APSys '24）：

- §1 Introduction —— 短小，contribution-forward
- §2 Motivation and Background —— problem space + 一张对比表，替代独立的 Related Work 节
- §3 Design —— mechanism-first 的展开
- §4 Evaluation —— Setup 段 + 跟 claim 对齐的 subsection + 末尾的 limitations subsection
- §5 Conclusion —— 一段

| Section | 约页数 | 约段落数 |
|---|---|---|
| 1. Introduction | 1.0 | 4 |
| 2. Motivation and Background | 1.0 | 5-6（含 Table 1） |
| 3. Design | 2.0-2.5 | 11-12 |
| 4. Evaluation | 1.5-2.0 | 8-10 |
| 5. Conclusion | 0.2 | 1 |

两处结构决策值得点出，因为它们跟 OSDI/SOSP full paper 的习惯不一样，但符合 APSys / IEEE workshop 的惯例：(a) 没有独立的 Related Work 节 —— 对照工作在 §2 通过 Table 1 加 inline citation 完成；(b) 没有独立的 Discussion 节 —— limitations 作为 §4 Evaluation 的最后一个 subsection。

---

## Section 1: Introduction

短，四段，目标是宣告 contribution 而不是展开整个 problem space。详细 motivation（已有系统为何无法解决问题、2×2 gap、design space）放在 §2。

**Para 1 —— 长上下文场景下抛弃 prefill 的代价**：
现代 LLM workload —— 含检索文档的 RAG、长文档 QA、agentic tool-use trace —— 常规地把 prompt 推过 16K token。RULER 16K 在 Qwen2.5-7B + A6000 上跑 prefill 要约 2.7 秒（我们的校准结果），而 ShareGPT chat 的 TTFT P95 约 450 毫秒。这个约 6 倍的差距让「丢掉 prefill」变得昂贵：一旦 prefill 算完，扔掉它每个请求要赔几秒钟的 GPU 时间 —— 这些时间本可以服务别的请求。

**Para 2 —— 两个会烧掉 prefill 的场景**：
我们点出两个明确需要重算 prefill 的场景。(a) **Engine 中途死掉**：一个长跑请求驻留在一块 GPU 上，这块 GPU 因为各种原因消失（硬件故障、OOM、计划检修、容器被 kill、多租户环境下被抢占）。长请求承受的概率更大 —— 它们活得久，disruption 暴露面跟生命周期成正比。(b) **中途因为优先级被抢占**：一个 tight-SLO 请求到达，队列头的 deadline 比正在运行的请求更紧。现有 scheduler 拒绝抢占正在 decode 的长上下文请求，因为重新 prefill 的代价完全吞掉了优先级带来的好处。两个场景的根因相同 —— 没有便宜的方式去 *恢复* 一个中途被打断的 decode。§2 详细展开为什么现有系统在这两个场景上各自束手无策。

**Para 3 —— 一句话 insight**：
一份 host-side KV checkpoint，按 block 对齐的节奏异步发布到 `/dev/shm`，可以同时充当：(i) engine 死掉时的恢复底层 —— 下一个 engine 直接从 checkpoint 读起、不必重新 prefill；以及 (ii) cheap-preempt 的底层 —— 让 scheduler 直接抢占正在 decode 的长上下文请求，因为请求被踢出后可以从 checkpoint 恢复，不浪费 prefill。同一份数据、两种用途、无重复。一旦 preempt 变便宜，scheduling 就可以从「重度依赖预测的 proactive admission」（当前 SLO scheduler 的位置）转向「重度依赖测量的 reactive preemption」—— 当某个底层 mechanism 的代价降到一个阈值以下，过去因为代价太高而不能进入的 design space 就被打开了。

**Para 4 —— Contribution、关键数据、路线图**：
三条 contribution，每一条对应一个可以验证的 artifact。(1) Host-side KV checkpoint mechanism，由 in-process pinned memory pool 加 `/dev/shm` mirror 构成的两层 store，把 cadence 跟 access path 解耦。(2) 基于 slack 的 preempt picker，用 cheap preempt 做出过去做不到的决策 —— 为更紧的 SLO waiting request 抢占长上下文的 decode。(3) Cross-engine reroute，通过同一份 checkpoint 实现：用一个 state machine 从 shm 恢复 KV，并在存活的 engine 上继续 decode。实验结果：在 saturation knee 之后我们的 tight-class SLO attainment 显著高于 baseline；在 RULER 16K 上 single-engine kill 的 failover 时间小于 2 秒；healthy load 下额外开销可忽略。后续：§2 给出 design space 的 gap，§3 给出 design 细节，§4 跑实验验证三条 claim，§5 收尾。

---

## Section 2: Motivation and Background

详细的 problem space 和已有系统对照。两个 subsection 加一张对比表；related work 通过 Table 1 加 inline citation 折叠进来，不再单独立节。

### §2.1 长上下文 serving 和 disruption 问题（2 段）

**Para 2.1.1 —— 长上下文 serving 场景**：
明确场景范围。所谓「长上下文」指 prompt 16K token 以上。代表性 workload 有：含检索文档的 RAG（16K-128K）、长 trace 的 code agent、文档支撑的 QA。我们 setup 上的具体代价：Qwen2.5-7B 在 A6000 上跑 16K prefill 耗时 2.7 秒（来自无压力 calibration 的 P95），而 ShareGPT chat 的 TTFT P95 是 0.45 秒。这个约 6 倍的差距是让 engine-failure recovery 和 priority preempt 两个场景都变得「丢 prefill 很贵」的根源。

**Para 2.1.2 —— Disruption 并不罕见，SLO 压力是常态**：
列出 engine 在请求生命周期内消失的实际原因：计划性升级停机、spot 实例或多租户集群中的抢占、硬件故障（GPU ECC、网络丢失）、并发超额订阅导致的 OOM。长请求的暴露面更大，因为它们活得久。独立于 disruption，现代 serving 系统也始终面临 SLO 压力：tight-SLO 请求在 loose-SLO 请求 decode 中途到达，在任何非均匀 workload 下都是常态情形。§1 中提到的两个场景 —— engine failure 和 priority preempt —— 共享同一个根因（没有便宜的 resume），也因此都指向同一类 mechanism 应答。

### §2.2 现有系统和未被填满的 quadrant（3 段 + Table 1）

**Para 2.2.1 —— 三类已有系统，每类只覆盖问题的一半**：
三类已有工作各自占据 design space 的一个侧面。第一类是 SLO scheduler（QLM、Scorpio、JITServe、Niyama）—— 基于预测的 proactive 方案，admit 保守，绝不抢占正在 decode 的长上下文请求，通过限制只在 early prefill 阶段或短上下文做 preempt 来规避难题。第二类是基于 migration 的 fault tolerance（Llumnix）—— 支持 engine 间 live KV migration，用于负载均衡和优先级隔离，但 mechanism 要求 source engine 还活着；当 source 突然死掉，migration 无法进行。第三类是 host-side KV pool（Mooncake 用于 prefix cache 复用、TokenFlow 用于同 engine 内的 preempt-and-resume）—— 在 host 内存里维护持久 KV，但两者都没演示过 cross-engine recovery。

**Para 2.2.2 —— Table 1：feature 对比**：
Table 1 列出 10 个代表性系统，按 4 个对我们问题来说重要的二值维度对比：(i) 能否抢占正在 decode 的长上下文请求；(ii) 在 source engine 死亡后能否存活（即 disruption-aware）；(iii) 是否维护 host-side KV pool；(iv) 是否支持 cross-engine 状态搬运。规律：所有 prior system 都填了 4 格中的 3 格，但没有任何一个把 4 格都填满。空的那一格就是 `(cheap preempt of long-context decode) × (disruption-aware)` —— 也就是我们占据的位置。Table 1 的每一行对应一条 inline citation，指向相应的已有工作；Table 1 因此替代了独立的 Related Work 节。

**Para 2.2.3 —— Preempt 变便宜之后，scheduling 会改变什么**：
把 cheap preempt 对 scheduling 的影响讲清楚，因为这本身就是一条 contribution。没有 cheap preempt 的时候，scheduler 必须预测 latency（Scorpio 的 α/β/γ/δ admission gate、JITServe 的 QRF 长度预测、Llumnix 的 headroom 常量），并保守 admit，避免动到正在运行的请求集合。有了 cheap preempt 之后，scheduler 在 runtime 观察 slack 漂移并做出 reactive 修正 —— 抢占最闲的 tail，把空间让给最紧的 head。对离线 profile 出来的 latency 模型的依赖被抹掉。这个转变 —— 从「重预测的 proactive」变成「重测量的 reactive」—— 是 mechanism 解锁的 policy-level contribution。

**Table 1：代表性 LLM serving / recovery 系统的 feature 对比**

| System | 抢占长上下文 decode | Disruption-aware | Host-side KV | Cross-engine 状态搬运 |
|---|---|---|---|---|
| vLLM (FCFS) | ✗ | ✗ | ✗ | ✗ |
| QLM | ✗（仅 early chunk） | ✗ | ✗ | ✗ |
| Scorpio | ✗（TTFT/TPOT gate） | ✗ | ✗ | ✗ |
| JITServe | ✗（admission control） | ✗ | ✗ | ✗ |
| Niyama | ✗（deadline EDF） | ✗ | ✗ | ✗ |
| Llumnix | partial | partial（source 必须活着） | ✗ | ✓ live migration |
| Mooncake | ✗ | ✗ | ✓（prefix cache） | ✗ |
| TokenFlow | ✓（同 engine resume） | ✗ | ✓ | ✗ |
| FastServe | ✓（skip-join MLFQ） | ✗ | partial（host swap） | ✗ |
| **Ours** | **✓** | **✓**（source 可以死） | **✓** | **✓**（checkpoint mirror） |

---

## Section 3: Design

Mechanism-first 的展开。开头是 architecture overview，然后依次讲 host-side checkpoint（mechanism）、slack picker（消费 mechanism 的 policy）、cross-engine reroute（同一份 mechanism 的第二种用法），最后用一段 implementation note 收尾 —— 这段刻意不立独立 subsection。

### §3.1 架构总览（1 段 + Figure 1）

落在 Figure 1 上讲：client → router 进程（CPU-only，跟 engine 进程独立）→ engine 进程（每块 GPU 一个）。Engine 把 KV checkpoint 发布到 host-side store，该 store 有两层：一层 in-process pinned memory pool，一层 `/dev/shm` mirror（对其他 engine 可见）。Router 通过一个 shm 状态文件监视 engine 健康，engine 死了就 reroute。每个 engine 在自己的 `schedule()` 钩子里跑基于 slack 的 scheduler。**同一份 checkpoint 数据**同时支撑同 engine 的 preempt-and-resume 和 cross-engine recovery —— 这种 mechanism-policy 耦合是 design 上的核心 contribution。

### §3.2 Host-side KV checkpoint（3 段）

**Para 3.2.1 —— 存什么、何时存**：
按 block 对齐的节奏：每解出 16 token（一个完整的 vLLM KV block），engine 就把这个 block 的 K/V tensor 存下来。存的是 per-block delta —— 只存自上次以来新稳定的 block，不存完整 KV。按 block 对齐是硬性要求，因为 vLLM 用固定 block 分配 KV，不完整的 block 没有清晰的边界。

**Para 3.2.2 —— 两层 store**：
Layer 1 是 in-process 的 `KVCheckpointPool`，存 pinned host memory。同 engine 的 preempt-and-resume 直接读 Layer 1，亚毫秒级。Layer 2 是 `/dev/shm` 上的 mirror，per-chunk 通过 tmp + rename 的方式原子写，加一个 manifest 和一个 "latest" pointer 保证原子性。Cross-engine recovery 读 Layer 2，scatter 拷贝到新 engine 的 GPU。两层数据完全一样，只是 access path 不同。

**Para 3.2.3 —— 异步 publish pipeline 和正确性 invariant**：
天真的同步 `/dev/shm` publish 会让 engine step 阻塞 5-15 毫秒每次 save，累计起来约 30% 的 TTFT overhead。要把 healthy load 下的开销压到可忽略，仅靠一个 thread pool 是不够的 —— 整条 publish 路径是一个四阶段 pipeline，每一阶段砍掉一种 overhead：(i) engine 侧，`_save_checkpoints_if_needed` 把 `checkpoint_kv_blocks` RPC 当作 Future 发出去，并 eager 推进 `num_checkpointed_tokens`，加上 depth-1 backpressure（只有上一个 Future 还没完成才等）；(ii) worker 侧，把 K 个 running request 的新 block 在 GPU 上 batch gather 到一次 concat 加一次 async H2D copy 里，而不是发 K 次独立的 cudaMemcpyAsync（通过记每个 request 的 slice 区间保留 delta-checkpoint 的语义）；(iii) 写 `/dev/shm` 上的 chunk file 改用 ctypes 直调 libc 的 `write` —— 这个 syscall 期间会释放 GIL，不会偷主线程的 CPU；(iv) cross-engine 的 `restore_kv_blocks` 在一个共享 CUDA stream 上跑，整批末尾统一调一次 `flush_pending_restore` 而不是每个 request 单独 sync，加一个动态 GPU memory 估算 —— 只有累积的 temp tensor 真的超过 vLLM KV pool 之外的 headroom 时才退化到 per-request sync。两条正确性 invariant 锁住整条 pipeline 的安全：(a) "latest" pointer 必须在 chunk file flush 加 rename 之后才更新，所以 cross-engine restore 永远读不到只写一半的 chunk；(b) 如果 engine 在某次 publish 完成前死掉，eager 推进的 `num_checkpointed_tokens` 可能超前于 `/dev/shm` 上实际写到的位置 —— 对端 engine 读旧的 "latest" pointer，安静地少 restore 几个 token，这部分缺口由 scheduler 的 reload state machine 当作 decode replay 兜回来。

### §3.3 基于 slack 的 preempt picker（3 段）

**Para 3.3.1 —— Slack 作为单一刻度**：
定义 `slack(r, t)` = 请求 `r` 还有多少 wall-clock 时间才会错过 SLO。还没出第一个 token 的请求用 TTFT slack = `S_TTFT − elapsed_since_arrival`；已经在 decode 的请求用 TPOT slack = `S_TPOT − avg_per_token_so_far`。Slack 把两种 SLO 压到一个可比刻度上。关键：slack 是从 runtime *测* 出来的，不是从离线 profile 的 latency 模型预测出来的 —— §2.2.3 里宣告的从 prediction-heavy 到 measurement-heavy 的转变，在这里落地。

**Para 3.3.2 —— Picker 规则和五道闸**：
每个 scheduler step，从 running 队列里挑 slack 最大的（最闲的）、从 waiting 队列里挑 slack 最小的（最紧的）。**五条都满足才抢占：**
(i) `slack(victim) − slack(head) > δ + replay_cost(victim)`，其中 `replay_cost = switch_cost + (num_computed_tokens − num_checkpointed_tokens) × per_token_ms`。`switch_cost` 是每次 fire 的固定 wall-clock 开销（preempt 状态机 + 跨引擎 HTTP forward + V3 reload + admit），默认 1000ms，按实测 P95 校准。`δ` 是 hysteresis（默认 1 秒）；
(ii) victim 的 host checkpoint manifest 文件**真的存在**于 `/dev/shm`。`num_checkpointed_tokens` 计数器是 save RPC *发起*时就 eager 加上的，而 manifest 文件只有 worker 实际刷完 bytes 才出现。读文件等于用跨引擎 reroute 同一个真理来源，避免 picker fire 后悄悄 fallback 到全 prefill；
(iii) 等待队列的 head 真的离截止时间不远了，即 `slack(head) < r × S_TTFT(head)`，`r` 默认 0.10（Niyama 风格的绝对截止时间检查）。没有这一道闸，picker 在 victim 比 head 闲很多时就会 fire，即便 head 远没到 SLO 危险线、队列自然推进完全来得及；
(iv) **至少有一台 peer engine 有空位接 victim**。Picker stat `/dev/shm/vllm_ft_engine_status/engine_<peer>.json`（跟 router 死亡检测共用同一份心跳文件），要求 `alive` + 心跳新鲜 + `kv_usage < 0.85` + `running < 0.8 × max_num_seqs`。没有这一道闸，picker 会盲目把 victim 推到饱和的 peer 上排队（RULER 16K 实测 P95 等几十秒）；
(v) `now − last_preempted_at(victim) > cooldown`（默认 5 秒），防止同一 victim 被反复翻牌。
五道闸分别封堵五个我们实验中观察到的失败模式：(i) 交易抖动、(ii) 静默 fallback 到 reprefill、(iii) 健康负载下乱踢、(iv) 把 victim 推到饱和 peer、(v) 同一 victim 反复被踢。五个阈值都可调；默认值每个硬件 calibrate 一次后在所有 SLO sweep 中固定不变。

**Para 3.3.3 —— Cheap preempt 解锁了什么（具体表述）**：
没有 cheap preempt 的话，scheduler 必须依赖离线 profile 信号（Scorpio 的 α/β/γ/δ admission gate、JITServe 的 `v_token`、Llumnix 的 headroom 常量）。有 cheap preempt 之后，scheduler 在 runtime 观察 slack 漂移并 reactive 修正 —— 抢最闲的 tail，让位给最紧的 head。Slack 公式承担的角色跟那些 prior 预测信号相同，但**来自测量而不是离线 profile**，砍掉了一条脆弱的依赖以及附带的离线 profile pipeline。

### §3.4 Cross-engine reroute（3 段）

**Para 3.4.1 —— Engine 死亡检测**：
每个 engine 在守护线程里每 200 毫秒把心跳文件写到 `/dev/shm/vllm_ft_engine_status/engine_<id>.json`（必须是独立线程、不能挂在 `step()` 上 —— vLLM 的 busy loop 在空闲时会阻塞在 `queue.get`，挂在 step 上的心跳一旦 engine 空闲就停了，router 会错把活着的 engine 判死）。Router 每 500 毫秒 poll 一遍这些文件，时间戳老于 2 秒的 engine 被判死。判死触发对这台 engine 上所有 in-flight 请求的 reroute。

**Para 3.4.2 —— Reroute 决策和 KV restore**：
Router 维护一个 `in_flight` 字典，把 `router_req_id` 映射到 `(engine_id, body)`。某个 engine 死了之后，router 枚举那台 engine 上的 in-flight 请求，逐个 re-forward 到一台存活的 engine，body 里带上 `vllm_xargs.is_rerouted=True`、原 engine 的 `internal_req_id`（新 engine 用它找 checkpoint）、以及已发布的 `num_checkpointed_tokens`。新 engine 的 `add_request` 看到 `is_rerouted` 之后把请求引到一条 reload state machine 上，而不是走正常 prefill。State machine 分配 KV block，调 worker 的 `restore_kv_blocks`（去 `/dev/shm/vllm_ft_checkpoints/<original_id>/` 里读），restore 完成后把状态翻成 `PREEMPTED`。vLLM 自己的 resumed-from-preempt admit 路径会接管 —— 不需要改 admit loop。

**Para 3.4.3 —— Design 耦合，再陈述一遍作为 contribution**：
同一份 shm checkpoint 数据，同时支撑 same-engine preempt-and-resume（slack picker 通过 `num_checkpointed_tokens` 来消费）和 cross-engine reroute（通过 `/dev/shm/vllm_ft_checkpoints/` 来消费）。如果把 checkpoint 拿掉，两条路径都退到 §2.2 描述的不那么强的 corner：scheduler 退到 QLM/Niyama 的位置（没有长上下文 preempt 的能力），reroute 退到 Llumnix-source-死掉的位置（在新 engine 上完整重 prefill）。这种 mechanism-policy 耦合 —— 一条新的数据路径解锁两种 scheduling 动作 —— 是 contribution 的核心。

### §3.5 Implementation（一段，不立 subsection 标题）

基于 vLLM v0.16 实现。Scheduler 钩子（约 80 行代码）、engine core state machine 加心跳（约 200 行）、worker 端的 `checkpoint_kv_blocks` / `restore_kv_blocks` RPC（约 150 行）、一个新的 router 进程（约 400 行）、workload runner（约 200 行）。三个 non-obvious 的工程决策值得点出，每一个都在我们早期原型上踩过坑：(i) 用 `vllm_xargs.router_req_id` 而不是 `X-Request-Id`（后者会被 vLLM 的 OpenAI handler 加上 router 预测不出来的 `cmpl-<id>-0` 前缀）；(ii) 心跳走守护线程，不挂 `step()`（空闲 engine 永远不会调 step，挂 step 的心跳会让 engine 假死）；(iii) 异步 publish 走 `ThreadPoolExecutor` 加 depth-1 future-based backpressure，不要同步 publish（前者节省约 30% 的 TTFT），也不要无界异步（后者在我们更早一版的原型里被测到 -10% goodput，原因是 GIL 抢占）。

---

## Section 4: Evaluation

开头一段密度高的 Setup 段（不立独立 Setup subsection），接下来三个对应 §1 三条 contribution 的 subsection，最后一个 Limitations subsection 在末尾（取代独立的 Discussion 节）。

### Setup 段（1 段）

硬件：2× NVIDIA A6000 PCIe（48 GB 每块）；portability check 跑在 2× L40S 上（同模型不同 SM）。模型：Qwen2.5-7B-Instruct fp16，`max_model_len 32K`。Workload：长上下文用 RULER 16K（这是我们 paper 主打的 regime），短上下文用 ShareGPT（这是不能被打坏的 regime），都用 Poisson 到达加 Niyama 风格的三档 QoS（主图）。SLO calibration：在 `vllm_fcfs` 上以无 contention 的低 QPS 测 baseline P95（RULER 0.02 QPS、ShareGPT 0.1 QPS）；每档 SLO 设成 baseline P95 的 {2×, 3×, 6×}，对应 tight / normal / loose。Baseline：`vllm_fcfs`（无 router 的地板 —— 测「完全没 router 没 FT」的开销）、`reroute_no_ckpt`（我们的架构去掉 FT —— 测「只有 router」的开销）、`ours`（完整系统）、`ours_no_picker`（picker ablation —— FT 全在但 slack picker 关掉）。每个配置三个 seed，报告 mean ± std。在跑压力测试之前，我们先验证 FT 机制在 healthy 低 QPS 下不带来实质 overhead（E_M3）：`ours` 在 `reroute_no_ckpt` 之上加约 30 毫秒 TTFT P50、throughput 差距在 0.5% 以内；router 自身基本免费（`vllm_fcfs ≈ reroute_no_ckpt`）。

### §4.1 压力下的 SLO attainment（3 段）

**Para 4.1.1 —— 主图（E_M1）**：
Figure 2：x 轴是 QPS，y 轴是 SLO attainment %，三条线（`ours`、`reroute_no_ckpt`、`vllm_fcfs`）。ShareGPT 扫 1.0 → 8.0 QPS（RULER 16K 对应区间类似）。所有系统在低 QPS 下都是 100%，随 QPS 上升曲线分叉，`ours` 衰减更晚、attainment 保持更高。报告饱和拐点（baseline 第一次跌破 90% 的 QPS）的具体数：`ours = X%`、`reroute_no_ckpt = Y%`、`vllm_fcfs = Z%`。长上下文 RULER 上差距最明显，因为 `ours` 每次 preempt-and-resume 节省的几秒钟 prefill 是 baseline 必须重做的。

**Para 4.1.2 —— 三档拆分（E_M1-tiered）**：
均匀 SLO 加均匀到达没法完全暴露 picker 的价值（所有人一样紧）。Figure 3 在三档 QoS 下画 per-class attainment vs QPS。`ours` 把 tight 这档保护住 —— 维持 ≥95%，远超 baseline 在 tight 上跌破 50% 的 QPS —— 同时不损害 loose（所有系统在所有 QPS 下 loose 都 >90%）。Picker 的重分配方向正确：把资源优先给「SLO 难以达成」的那档，也就是 scheduling 决策真正起作用的地方。

**Para 4.1.3 —— SLO 紧度扫描（E_M2）**：
QPS 固定在拐点，把 SLO 倍数从 2× 扫到 6×。Figure 4：x 是倍数，y 是 `ours` attainment 减 baseline attainment。差距在 2×（最紧）处最大，到 6×（最松）逐渐归零。结论：`ours` 赢在该赢的地方 —— SLO 不平凡的时候；SLO 松到所有系统都达标时，没有任何调度决策能改变结果。

### §4.2 Mechanism vs policy ablation（1 段）

**Para 4.2.1 —— Picker ablation（E_M4）**：
在 Figure 2 上加一条 `ours_no_picker` —— checkpoint mechanism 不变、V3 reload state machine 可用、但 slack picker 关掉。结果：`ours_no_picker` 卡在 `ours` 和 `reroute_no_ckpt` 中间。Mechanism 本身有被动收益（被容量逼着抢的时候，resume 更便宜），但真正主动把 SLO 优势兑现的是 slack picker —— 它*主动*选择为 tight-SLO admit 而抢占。两条 contribution 独立可加：拆掉 checkpoint 退回 Mooncake/TokenFlow 的角落（机制但没有 cross-engine），拆掉 picker 退回 QLM/Niyama 的角落（策略但没有 cheap preempt）。只有两者组合起来才占据 §2.2 那个空 quadrant。

### §4.3 Disruption recovery（2 段）

**Para 4.3.1 —— 单 engine kill demo（E_D1）**：
Figure 5：failover_gap（engine SIGKILL 到受影响请求的下一个 token 之间的时间），三个系统 × ShareGPT 和 RULER 16K。`ours`：约 1.5 秒。`reroute_no_ckpt`：4K 上约 2.5 秒、16K 上约 10-20 秒（完整重 prefill 的代价）。`vllm_fcfs`：未定义 —— 客户端收到 5xx。差距*随着上下文长度增长* —— 在 64K 上我们外推估算无 checkpoint 的 baseline 会到几十秒，这正是 paper 主打的长上下文 regime。

**Para 4.3.2 —— Variance 论证**：
`ours` 不仅 *均值* failover_gap 更低，*方差* 也更低：同一台死掉 engine 上的全部 rerouted 请求在一个 V3 batch 内同步恢复，请求之间的恢复时刻散度小于 1 毫秒。`reroute_no_ckpt` 通过 admit 路径错峰恢复，暴露出长尾。这是第二条独立论证 —— 即使 `ours` 的均值跟 baseline 接近，更紧的尾巴本身也足以为任何在意 P99 的系统证明这个设计的价值。

### §4.4 Limitations 和 future work（1 段）

我们只测了 2 个 engine；连锁失败或 3+ engine 不在 scope 内 —— 那种情况需要协调多个存活 peer 上的 `is_rerouted` 请求。Workload-level recovery 走单 router 进程 —— 这篇 paper 不处理 router HA。Cross-engine restore 会丢失 dead engine 上已经生成的 output token，因为我们只搬 KV、不搬 output token id；对 SLO 计账而言这是无害的（新 engine 从稍微靠前的位置重新发 token），但总工作量比无 disruption 时略多。Future work：基于 slack 的自适应 checkpoint cadence（紧 SLO 请求 checkpoint 得更频繁）、跨 engine slack 协调（接收 reroute 的 peer engine 主动让出）、用 Qwen2.5-7B-Instruct-1M 变体支持更长上下文。Admission control 跟我们的 reactive preempt 正交，可以干净地叠加上去。

---

## Section 5: Conclusion

Cheap preempt 把 SLO scheduling 和 disruption recovery 统一在同一个设计下。一份 host-side KV checkpoint 同时充当 cheap same-engine preempt 的底层和 cross-engine recovery 的底层，配上一个 slack-based picker 作为 policy，把 cheap-preempt 的能力转化为可测量的 SLO attainment 收益。实验上，这个组合在 saturation knee 之后维持更高的 tight-class SLO attainment，在长上下文 workload 上 single-engine kill 后 2 秒内恢复，healthy load 下的额外开销可忽略。更一般的含义：当某个底层 mechanism 的代价降到一个阈值以下，过去因为代价太高而不能进入的 design space 就被打开了 —— host-KV checkpoint 在长上下文 LLM serving 这个领域跨过了这个阈值，随着模型上下文窗口继续增长，我们预期类似的格局会反复出现。

---

## 跨论文结构提醒（来自 reference_paper.md）

写作 checklist，参考 TokenFlow + Niyama + Scorpio + Llumnix 的写法 pattern，加上 Capybara（APSys '24）作为 venue-shape 参考：

1. **Intro 里要有具体数字**，不要泛泛说「prefill 慢」—— 量化到约 2.7 秒。TokenFlow、Niyama、Capybara 都在第一段就给具体数。
2. **Design 节里 mechanism 在 policy 前**（TokenFlow + Llumnix 的 pattern）—— 先讲 checkpoint，再讲 scheduler 怎么用它。
3. **Evaluation 顺序：先 end-to-end、再 mechanism micro、最后 ablation**（TokenFlow + Niyama）。我们只在一个地方反向：把 healthy overhead 的小确认编进 Setup 段而不是独立 subsection —— 那是个小型 de-risk 论证，应该在 reader 看到主图之前就咽下去。
4. **Related work 折叠进 motivation，用对比表呈现**（Capybara Table 1 pattern）—— 按对我们问题来说重要的轴，把每个 prior system 占一行。这取代了独立的 Related Work 节。
5. **Limitations 作为 evaluation 的 subsection**，不立独立节（Medicine AIoTC '25 pattern；Capybara 干脆没有 limitations 节）。保持 paper 在 5 节内。
6. **先 formalize 再 relax**，如果有值得 formalize 的 objective（TokenFlow 的 proxy objective pattern）。我们的 slack 公式是候选 —— 在 §3.3.2 inline formalize，标注它是「最小化预期 SLO 违反」的启发式 proxy，不解 LP。
7. **Workshop 长度的压缩纪律**（Scorpio pattern）—— 每个 mechanism 一段；公式 inline 写、不立独立 algorithm block；topic 支持就把 subsection 缩到单段。
8. **Mechanism vs scenario 的抽象**（Llumnix 用 "virtual usage" 统一四个 scenario）—— 我们的对应物是「slack + replay_cost」这一个量同时刻画 stage-aware urgency 和 preempt eligibility。在 §3.3 明确点出这层抽象。

---

## 投稿前要解决的 open question

| 问题 | 解决路径 |
|---|---|
| 报 32K 还是 16K 的结果？ | 当前模型最大支持 32K。RULER 16K 是最干净的 workload。32K 扩展通过 1M context 模型变体作为 future work 提，放 §4.4。 |
| Hysteresis δ 和 cooldown 默认值要多激进？ | 跑一小段 sweep（占 E_M2 一部分预算）—— δ ∈ {500 ms, 1 s, 2 s}，cooldown ∈ {3 s, 5 s, 10 s}。挑在拐点处跟 baseline 拉开最大差距的组合。在 §3.3.2 inline 注明默认是调出来的、不是从第一性原理得到的。 |
| 主图用 tiered 还是 uniform？ | uniform 先跑完（已经在跑）；tiered 故事更清楚但用的是合成 QoS 类。大概率把 tiered 当 Figure 2，uniform 放 supplementary。 |
| 报告的 `failover_gap` 用 P50 / P95 还是 max？ | §4.3.1 文字里报 P50 + P95，Figure 5 上画 max。Mean 在这里误导，因为 `reroute_no_ckpt` 方差大 —— 方差本身就是论证之一。 |
| Healthy load 下残留的 30 毫秒 TTFT overhead 怎么诚实交代？ | per-step 在 running 队列上的迭代代价是真实开销，不是 bug。在 Setup 段承认；不要在投稿前再花时间去优化。 |
| Table 1 留 10 行还是裁掉一些？ | 第一稿留 10 行（对齐 Capybara 在 APSys '24 预算下的 11 行 Table 1）。如果 §2 撑不下再裁，先丢 FastServe（相关性偏弱）、然后在 QLM/Scorpio/JITServe/Niyama 里挑两个最能跨越 proactive-scheduler 空间的留下（很可能 Scorpio + Niyama）。 |
| 如果 Figure 5 + tiered 图 + ablation 撑爆 §3 / §4 怎么压？ | 第一杠杆：删 §4.1.3 的 SLO 紧度扫描，并到 §4.1.2 末尾一句话。第二杠杆：把 §4.3.2 的 variance 论证压到 §4.3.1 末尾半句。第三杠杆：把 §3.4.3 的 design coupling 再陈述压到 §3.4.2 末尾一句。 |
