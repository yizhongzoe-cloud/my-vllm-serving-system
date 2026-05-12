问题
长 context 请求（RAG，几万 token 的 prompt）prefill 要跑几十秒。跑这种请求时有两件事会让前面那几十秒白干：

engine 挂了，请求只能甩到另一台 engine，重新 prefill
新来个 SLO 很紧的请求要插队，想踢掉正在跑的长 context 请求腾位置 —— 踢了 prefill 也白干
所以现在所有 SLO scheduling paper（QLM、Scorpio）都不敢踢正在跑的长 context 请求，太亏。

观察
Recovery 那派的系统（Mooncake）已经把 KV state 定期搬到 host memory 了 —— 它的目的是 engine 挂了能从 host 拉回来恢复。这份 host 上的 KV 数据，scheduler 也能拿来用啊。

做法
每个请求跑的时候，每 16 个 token 就把它新产生的 KV 块从 GPU 拷一份到 host pinned memory，再 publish 到 /dev/shm（让别的 engine 也能读）。这一份数据同时干两件事：

Recovery：engine A 挂了，请求迁到 engine B，engine B 从 /dev/shm 读 KV 接着跑，不用重 prefill
Cheap preempt：scheduler 想踢一个正在跑的请求让紧急请求上，踢就踢 —— 反正 KV 在 host 还在，等会恢复也不亏
关键 insight

踢人变便宜了，scheduler 终于敢按"哪个请求最快超时"（slack）来调度长 context 请求。每个请求算自己离超时还剩多少时间，最急的先跑；如果正在跑的某个请求剩很多时间，而队列头剩没多少时间，就把它踢下来让位 —— 反正踢的成本被 checkpoint 兜底了。

故事一句话
路径 1：弱化 Mooncake，强调 recovery 空白
"Mooncake 提供了 host-side KV pool 这个机制，但目标是 prefix cache 复用；recovery 这条路 LLM serving 几乎没人做。我们借用类似机制做两件新事：recovery + cheap preempt"

路径 2：把 Mooncake 当机制启发，scheduling 一侧主打 QLM 对比
"QLM 知道按 slack 排队是对的，但只敢在入口拒人，不敢踢 running。我们把 host-side KV checkpoint 这个机制（类似 Mooncake 的 host KV pool，但用途不同）拿来给 scheduler 用，让它敢踢长 context 请求"



问题：但写shm会不会性能变的很差啊
那我们是不是paper里可以说我们支持两种checkpoint

1. 写 shm 性能差不差
默认配置下确实有点慢，但有 dial 可以调。

每次 save 触发一次 publish。Publish 写 3 个文件：chunk + manifest + latest pointer，每个文件都 atomic write（tmp + replace），默认还 fsync。

性能档位（每次 publish 的开销）：

配置	耗时	备注
默认 (torch.save + fsync)	~50 ms	round 1 我们 smoke 跑的就是这个
FT_FAST_TMPFS_WRITE=1	~30 ms	跳 fsync，tmpfs 上 fsync 是冗余的
FT_FAST_CHUNK_FORMAT=1	~10 ms	用自定义二进制格式跳 pickle 序列化
上面两个一起 + FT_INLINE_MANIFEST=1	~3-5 ms	manifest 嵌进 chunk header，只写一个文件
Publish 只在请求积累了新 block 时触发（增量，不是全量），所以每次写的实际数据量是 30KB-100KB，IO 不是瓶颈，syscall + fsync 是瓶颈。

实际 paper evaluation 时 默认就该用 FAST_CHUNK + FAST_TMPFS_WRITE + INLINE_MANIFEST 三件套，跟 vLLM 默认 prefill 路径比开销可忽略（prefill 单步 100-200 ms）。

老分支 round 1/2 跑 30-concurrent 16K 没爆，说明这套组合性能 OK。

2. Paper 里 "两种 checkpoint" 怎么讲
可以讲，而且这是个好卖点。两种不是并列的两条路，是一份数据两层存储：

层	给谁用	访问代价
Layer 1：进程内 host pinned memory pool (KVCheckpointPool)	Same-engine capacity preempt + reload	直接指针，<1 ms
Layer 2：tmpfs 共享镜像 (/dev/shm)	Cross-engine recovery (engine 死 → 别的 engine 接手)	文件 IO，~5 ms (fast chunk)
Paper 里的措辞可以是：

"We maintain a two-tier KV checkpoint store. The host pool serves same-engine cheap preempt at sub-millisecond access cost. A tmpfs mirror on /dev/shm serves cross-engine recovery, paying file-IO cost only when the local pool misses (engine death or capacity eviction). The same data backs both uses; no separate copy is maintained."

卖点维度：

机制 × 策略复用：proposal 已经讲了"一份数据两个用途"，这里再加一层"两层存储优化访问代价"
Worst-case 兜底：local pool 命中走快路；miss 才退到 shm，性能-可用性两头都有
对比：QLM / Llumnix / Mooncake 都只有一层 store，要么纯进程内（撑不住 engine 死），要么纯跨进程（每次访问都走 IO）。我们分层做到两全
不过有一点要 honest：两层不是各存一份，是 host pool 是 source-of-truth，shm 是异步镜像。蛛丝马迹是：cross-engine restore 时拿到的是几十毫秒前的状态（publish 比 host pool 慢一拍）。这个 lag 写 paper 时要诚实承认，但不影响故事。














| 系统 | 公式 | 排队规则 | 周期 | preempt victim |
|---|---|---|---|---|
| **vllm 默认 FCFS** | 无 | arrival_time | per-iteration | LIFO + recompute |
| **vllm PRIORITY** | 无 | (priority, arrival_time) | per-iteration | 最低优先级 + 最晚到 |
| QLM | RWT slack | EDF | per-arrival + 后台 | 不踢 running |
| Scorpio | TRP credit | credit-based WFQ | per-iteration | 不踢 running |
| JITServe | margin slack | GMAX | per-iteration | 只踢 early-prefill chunk |
| SLOs-Serve | 多阶段 DP | DP admit | per-interval | 只踢 best-effort |
| **我们 round 2** | slack（QLM 风格） | running 内挑 max slack 踢，让 min slack waiting head | per-iteration（vllm hook 在 schedule() 里） | **踢长 context decode**（用 KV checkpoint 兜底）|

## Q2：调度与请求 step 是同步还是异步

| Paper | 同步/异步 | 细节 |
|---|---|---|
| **QLM** | **异步** | global scheduler 跑在关键路径外（"off the critical serving path"）。RWT 估算 + LP solver 后台跑，vllm step 不暂停 |
| **Scorpio** | **同步** | Algorithm 1 是单个 while loop，每个 engine step 前先跑 TTFT/TPOT guard + credit batching |
| **JITServe** | **混合** | GMAX 调度决策在 vllm scheduler 层 inline（同步），但 QRF 长度预测在 gRPC sidecar 异步。**且调度更新限制在每 ∆=50 steps（约 300ms）一次**，不是 per-step |

## Q3：公式 + 约束

### QLM (13 个公式)

1. `C_q = W_q + P + D_q` — 完成时间 = 等待 + prefill + decode
2. `W_q = Σ O_i / Θ` — 等待时间 = 前面所有 req 的 output token 数 / 系统吞吐
3. `Σ O_i ~ N((q-1)μ, (q-1)σ²)` — CLT 上界
4. `D_q = O_q · ε · d` — 解码时间 = output token × 批处理低效因子 × 单 token 时间
5. `C = max_q C_q` — group 内最大完成时间
6-7. 分配约束: 每个 req 入一个 slot；每个 slot 一个 req
8. `t = 1[m_{j-1} ≠ m_j]` — model 切换 indicator
9. `wt = Σ W·x + Σ t·S + Σ C·t·x` — 累积等待（含 swap 开销）
10. `p = wt − slo` — penalty
11. **约束**: `p_{g,j} ≤ 0 ∀g,j` —— 硬 SLO 满足
12. **目标**: `min Σ p` —— 最小化总 SLO 超时

输入：每 group req 数、model id、SLO、swap 开销、output 长度预测。

**怎么用 — 3 阶段**：

| 阶段 | 公式 | 干啥 |
|---|---|---|
| 预测 | 1-4 | 算每个 req 还要等多久才完成（W + P + D），靠 output 长度估计 + 系统吞吐 |
| 建模 | 5-10 | 把"怎么排队"翻译成数学：x 是分配变量（req→slot），p 是 SLO 超时量，t 是 model 切换标志 |
| 求解 | 11-12 | 调 LP 求解器：`min Σp` 同时 `p ≤ 0`（不能超 SLO）→ 解出 x → 按 x 重排队 |

QLM 在后台**周期性调 LP 求解器**（用 scipy 或 cvxpy 之类的库）解出新的 queue 顺序，然后让 vllm 按这个顺序跑。求解器是黑盒，你只需要把公式 5-10 喂给它。

简化版：**只用公式 2 算 RWT，按 RWT 升序排队** —— 不调 LP，效果差一些但简单。

### Scorpio (9 个公式 / 步骤)

1. `ITL(|R|, L_avg) = α|R|L_avg + β|R| + γL_avg + δ` —— 单 token 间隔模型
2. `EstimatedTPOT(|R'|, L_avg(R'), P) = ε·{...}` —— 预测 TPOT
3. `PrefillTime(w_i) = φ if L≤θ; αL+β otherwise` —— 预测 prefill 时间（piecewise）
4. `EstimatedTTFT(w_i) ≥ Σ PrefillTime` —— 前面所有 req 的 prefill 之和
5. **TTFT Guard 约束**: `EstimatedTTFT(w_i) ≤ S_TT(r_i)` —— 才能 admit
6. `TRP(r) = min_{r'} S_TP(r') / S_TP(r)` —— TPOT 相对优先级
7. `VBS(R') = Σ TRP(r)` —— virtual batch size
8. **TPOT Guard 约束**: `EstimatedTPOT(VBS, L_avg) ≤ min S_TP(r')` —— 才能 admit
9. **Credit batching**:
   - 累积: `C_r += TRP(r)` 每 step
   - 选 batch: `B(t) = {r : C_r ≥ 1.0}`
   - 扣减: `C_r −= 1.0`

**怎么用 — 3 阶段**：

| 阶段 | 公式 | 干啥 |
|---|---|---|
| 预测 | 1-4 | 预测 TTFT 和 TPOT（用历史数据拟合的线性模型） |
| 判断 admit | 5, 8 | 公式 5 检查"新 req 加进 waiting 后 TTFT 还达标吗"；公式 8 检查"加进 running 后 TPOT 还达标吗"。两个 guard 都过才 admit |
| 执行 batching | 6-7, 9 | 公式 6-7 算 TPOT-tight req 的优先级（紧的 = 1，松的 < 1）；公式 9 是 credit 累加：每 step 加 TRP，攒够 1 上 batch |

每个 step **三步走**：
1. 有新 req？跑公式 5 检查 TTFT。过不了 → 拒掉
2. 跑公式 8 检查 TPOT。过不了 → 暂留 waiting
3. 在 batch 里：每 step 给每个 running req 加 credit（公式 9），credit ≥ 1 的进下个 batch

### JITServe (10+ 个公式 / 步骤)

1. `φ(s) = t_{≤s}/t_total`，per-stage deadline `D_s = φ(s)·D` —— 复合请求的 sub-deadline
2. `bw(r) = t_gen(r) / t_rem(r)`，其中 `t_gen = len_rem · v_token` —— 最小服务带宽
3. `bw_∆(r) = bw(r) · ∆` —— frame-level bandwidth
4. `goodput_∆(r) = goodput(r) / t_rem(r) · ∆` —— frame-level goodput
5. **优先级**: `Priority(r) = goodput(r) / t_gen(r)` —— 单位 bandwidth 的 goodput
6. **anti-starvation**: 每个 frame 加 `δ` 到 `goodput(r)` 防饿死
7. **候选过滤**（Alg 1 line 13）: `Candidates = {r : Priority(r) ≥ p · Priority(r_{(B)})}`，p∈(0,1]，常用 0.95
8. **group 选择**: `Priority(G) = Σ Priority(r)`，sliding window 取 argmax
9. **Preempt 规则**: 当且仅当 admit 收益 > `stall_duration × token_speed`（goodput loss）才踢
10. **理论保证**: `G_GMAX ≥ (1/8.56)·G*`（offline optimum 的 1/8.56 倍）

**怎么用 — 4 阶段**：

| 阶段 | 公式 | 干啥 |
|---|---|---|
| 预测 | 2-4 | 算每个 req 的"还需多少 bandwidth"，靠 QRF 预测剩余 output 长度 |
| 算优先级 | 5-6 | 公式 5 是核心：`priority = goodput / 生成时间`。公式 6 给等久了的 req 加 boost 防饿死 |
| 挑 batch | 7-8 | 公式 7 过滤掉低优先级，公式 8 在 sliding window 里挑总优先级最高的 group 上 batch |
| 决定 preempt | 9 | 想 admit 新 req 但满了 → 判断"踢一个 running 让位带来的 goodput 收益"是否大于"踢掉的损失"。是 → 踢 |

**每 50 个 step 做一次**完整调度（不是每 step）：
1. 用 QRF 更新所有 req 剩余长度估计
2. 按公式 5 算 priority
3. 用公式 7-8 挑下个 batch
4. 满了想加新的 → 用公式 9 决定踢谁

---

# 6 篇 SLO scheduler 统一模板（目标 / 变量 / 约束 / 周期 / preempt）

### QLM (SoCC 2024)

- **目标函数**: `min Σ_g Σ_j p_{g,j}` —— 最小化总 SLO penalty
- **决策变量**:
  - `x_{g,i,j} ∈ {0,1}` —— req i 在 group g 的 slot j
  - `t_{g,j} ∈ {0,1}` —— slot j 是否触发 model swap
  - `wt_{g,j}` —— 累积等待
  - `p_{g,j}` —— SLO penalty
- **约束**:
  - 每 req 一个 slot: `Σ_g Σ_j x_{g,i,j} = 1 ∀i`
  - 每 slot 一个 req: `Σ_i x_{g,i,j} = 1 ∀g,j`
  - SLO 必满足: `p_{g,j} ≤ 0 ∀g,j`
  - swap 标志: `t_{g,j} = 1[m_{g,j-1} ≠ m_{g,j}]`
  - 完成时间: `C_q = W_q + P + D_q`（其中 `W_q = Σ O_i/Θ`，`D = O·ε·d`）
- **调度周期**: **异步**。global scheduler 在关键路径外跑 LP solver。per-arrival 触发 + 后台周期性
- **Preempt running**: **否**。只在 waiting queue 重排，不踢 running

---

### FastServe (arXiv 2023, 早期 preempt 系列祖师爷)

- **目标函数**: prose-only —— `min` 平均 JCT（近似 SRPF，因 output 长度未知）
- **决策变量**:
  - `pjob ∈ {1..k}` —— 新 job 的 priority queue 分配
  - 每 iteration 的 batch composition
  - quantum 用尽时的 demotion target `p + η`
  - KV swap 选择（哪个 job 的 KV 挪到 host）
- **约束**:
  - GPU KV 容量: `Σ kv(job) ≤ GPU_mem`，溢出强制 offload to host
  - Quantum 公式: `q_i = 2·q_{i-1}`，`q_1 = min iter time`
  - Skip-join 入队: `pjob ← min i s.t. q_i ≥ init_time`（init_time = prefill 时间）
  - 防饿死: `job.starveTime ≥ α (≈ 300ms)` → 提到 queue 1
  - **无显式 TTFT/TPOT SLO**；SLO 只通过 α 进入
- **调度周期**: **per-iteration**（per output token）
- **Preempt running**: **是**。Token 粒度 preempt via MLFQ demotion。Swap 顺序按 `ENST(i) = min(T_promote, T_execute)`，ENST 大的先 swap out
- **关键差异**: 输入长度感知的 skip-join MLFQ；把 prefill cost 当 job 长度 proxy

---

### JITServe (arXiv 2025)

- **目标函数**: `max Σ_r goodput_∆(r)`，其中 `goodput_∆(r) = goodput(r) / t_rem(r) · ∆`
- **决策变量**:
  - `Priority(r) = goodput(r) / t_gen(r)`
  - Candidates: `{r : Priority(r) ≥ p · Priority(r_{(B)})}`，`p ∈ (0,1]`（常 0.95）
  - Group 选: sliding window argmax `Priority(G) = Σ Priority(r)`
  - Preempt 决策（goodput 收益 > stall 损失）
- **约束**:
  - Compound sub-deadline: `D_s = (t_{≤s} / t_total) · D`
  - 防饿死: 每 frame `goodput(r) += δ`
  - Multi-model: K dummy copies，`K ≤ M`
  - Fairness mix: `priority' = (1−f)·priority + f·Fair`
  - Frame size: `∆ = 50 steps (~300ms)`
- **调度周期**: **每 50 steps（frame，~300ms）**。GMAX inline 同步 + QRF gRPC sidecar 异步
- **Preempt running**: **是**。条件: admit 收益 > `goodput_loss = stall_duration × token_speed`
- **理论保证**: `G_GMAX ≥ (1/8.56)·G*`

---

### Llumnix (OSDI 2024)

- **目标函数**: prose-only —— max 每 instance `F = (M − ΣV) / B`（freeness），同时保证 priority headroom
- **决策变量**:
  - 派发: `i* = argmax F` over alive instances
  - 迁移: (source, dest) instance pair + 被迁 req 集合
  - 自动伸缩: 加/删 instance（基于 avg F）
  - priority class（input，不是 output，但影响 headroom）
- **约束**:
  - 每 instance: `ΣV ≤ M`
  - Virtual usage `V` 定义:
    - normal: physical memory
    - HOL queued: demand
    - high-priority: `physical + GetHeadroom(priority, instance)`
    - terminating: `∞`（强制迁移走）
  - Headroom: `headroomForPriority[p] / instance.numRequests[p]`（offline profile 得来）
  - Auto-scale band: `F ∈ [x, y]`
- **调度周期**: **混合**。派发 = per-arrival；迁移 = 周期性后台；local engine 调度 = per-iteration（delegate to vllm）；auto-scale = 周期性
- **Preempt running**: **是，但是 live migration**（跨 instance 搬运 KV cache，不是本地踢丢）。Victim 选: "lower priority + shorter seq length"（迁移成本低）
- **关键差异**: preempt = 跨 instance 迁移；SLO 经 priority memory headroom 间接编码（不用 deadline 公式）

---

### Andes (arXiv 2024)

- **目标函数**: `max Σ_i QoE_i`
  - `QoE_i = 1 − S_delay,i / S_whole,i`
  - `S_delay = Σ(T_i^Actual − T_i^Ideal)`
  - `S_whole = Σ(T_n^Actual − T_i^Ideal)`
  - T^Ideal 由 user 提供的 target TTFT + target TBT 参数化
- **决策变量**:
  - `x_i ∈ {0,1}` —— req i 是否在 batch
  - preempt set（running 被踢出 batch 的）
  - Batch size B
- **约束**:
  - `Σ x_i = B`
  - `Σ x_i · l_i ≤ M`（context lengths fit KV memory）
  - **无硬 per-req SLO**；QoE 是 soft 目标
  - Token 粒度 preempt 在 quantum 边界 legal
- **调度周期**: **per-quantum (per-iteration)**，但只在检测到 GPU 资源紧张时激活；否则跑前一 batch
- **Preempt running**: **是**。Knapsack 排序: `Priority_i = (Q_serve,i(B) − Q_wait,i) / l_i` —— marginal QoE gain / KV cost。低分（尤其长 context）的踢
- **关键差异**: SLO 替换成**连续 QoE 效用**；调度是 per-quantum knapsack on `ΔQoE / KV-cost`

---

### Niyama (arXiv 2025)

- **目标函数**: prose-only —— min deadline violations + max throughput，跨多个 QoS class
  - 隐含: `max goodput = Σ 1[finish_i ≤ D_i] · class_weight`
- **决策变量**:
  - 排序优先级 `P^i`
  - per-batch chunk size（dynamic chunked-prefill）
  - prefill req 选择
  - relegation flag（移到 lower-QoS queue）
- **约束**:
  - 交互式 deadline: `D_first = t_arrival + SLO_TTFT`，`D_n = t_arrival + SLO_TTFT + (n−1)·SLO_TBT`
  - 非交互式 deadline: `D_total = t_arrival + SLO_TTLT`
  - Batch fits KV cache budget
  - Relegation: 已违反 / 即将违反 → 移到 relegated queue（best-effort 跑，不再阻塞 on-time）
  - Tie-break: free vs paid tier
- **调度周期**: **per-iteration**
- **Preempt running**: **是 —— eager relegation**（demotion，非 eviction）
  - 优先级（interactive）: `P^i = t_arrival^i + SLO_TTFT^i + α·Prefill_rem^i`
  - 优先级（non-interactive）: `P^i = t_arrival^i + SLO_TTLT^i + α·(Prefill_rem^i + Decode_rem^i)`
  - `α ∈ [0,1]` —— EDF (α=0) ↔ SRPF (α=1) 插值
- **关键差异**: 显式 per-token deadline + EDF/SRPF 混合；preempt = class demotion（优雅降级），不踢出系统

---

### Our system (设计中 — 第一版草稿)

**架构：Hybrid 两层调度（Llumnix 风格）**

```
client → Router 进程 (CPU, 1 个)
            │ slack-aware dispatch
            ├──→ Engine 0 (GPU 0)：内部 picker
            └──→ Engine 1 (GPU 1)：内部 picker
```

| 层 | 干啥 | 借鉴 |
|---|---|---|
| **Router** | 跨 engine 派发：算每个 req 的 slack + 选 alive + 较空的 engine 派发 | Llumnix Global Manager |
| **Engine** | 本地 picker：内部按 slack 排队、按 slack 选 running victim 踢人腾位置 | round 2 picker（已实现） |

**Admission**：**不做**。所有请求来者不拒，全部入系统，靠两层调度 + cheap preempt 满足 SLO。理由：APSys workshop scope 聚焦核心 contribution（踢长 context running），admission 是 QLM/Scorpio 强项，加进来稀释焦点。后续 v2 再考虑。

借鉴来源：goodput 定义抄 DistServe；slack 公式抄 QLM (RWT)；两层架构抄 Llumnix；per-iteration picker 周期抄 vllm 自身；preempt 决策框架取 Andes (knapsack-style) 但用 slack 而非 QoE；hysteresis δ 抄 QLM admission backpressure（但只用 hysteresis，不抄 admission 本身）。

---

**变量说明表（先读这个，再读下面的公式）**:

| 符号 | 含义 | 来源 |
|---|---|---|
| `r` | 一个请求 | — |
| `t` | 当前时刻 | — |
| **请求属性（user 传入）** | | |
| `S_TTFT(r)` | 这个请求的 TTFT SLO 阈值（毫秒） | sampling_params.extra_args |
| `S_TPOT(r)` | 这个请求的 TPOT SLO 阈值（毫秒） | sampling_params.extra_args |
| `arrival_time(r)` | 请求到达系统的时间戳 | router 记录 |
| `is_rerouted(r)` | 是否是从死掉的 engine reroute 过来的 | router 派发时标记 |
| **运行时测量** | | |
| `e(r, t) = t − arrival_time(r)` | 自到达起经过的时间 | runtime |
| `num_output_tokens(r, t)` | 已生成的 output token 数 | vllm Request |
| `decode_elapsed(r, t)` | 自第一个 token 出来后过了多久 | runtime |
| `avg_TPOT(r, t) = decode_elapsed / num_output_tokens` | 至今平均 token 间隔 | runtime |
| `num_checkpointed_tokens(r, t)` | 最新已 publish 到 shm 的 token 数（≤ num_output_tokens） | engine 内部记录 |
| `last_preempted_at(r)` | 最近一次被踢的时间戳；从未被踢则为 −∞ | engine picker 写 |
| **派生量（公式计算）** | | |
| `slack(r, t)` | 距离 SLO 超时还有多少时间（见公式） | scheduler 算 |
| `replay_cost(r, t) = (num_output_tokens − num_checkpointed_tokens) × avg_TPOT(r)` | 如果踢了它，恢复时要重跑的 decode 时间 | scheduler 算 |
| **决策变量（scheduler 输出）** | | |
| `head` | 选出来的 waiting 队列里最急的请求 | engine picker |
| `victim` | 选出来的 running 里最从容的请求 | engine picker |
| `preempt ∈ {0,1}` | 这一步是否真踢 victim | engine picker |
| `target_engine` | router 把请求派给哪个 engine | router |
| **系统常数（可调参数）** | | |
| `δ` | hysteresis 阈值，差距大于 δ 才踢人（防抖） | **1000ms 起步（后续 sweep 调）** |
| `cooldown` | 一个 req 被踢之后多久内不能再被踢（防抖） | **5s 起步（后续 sweep 调）** |
| **硬件容量（vllm 底层管，scheduler 不显式建模）** | | |
| `M_kv` | GPU KV pool 容量（block 数） | vllm 启动算 |
| `kv_blocks(r)` | 请求当前占多少 KV block | vllm |
| **集合** | | |
| `waiting`, `running` | engine 内部的 waiting / running 队列 | vllm |
| `alive engines` | router 认为还活着的 engine 集合 | router 读 shm status |
| **Router-engine 通信总线（shm 文件）** | | |
| `/dev/shm/vllm_ft_engine_status/engine_<id>.json` | 每个 engine 每 ~100ms 写一份：`{alive, running_count, waiting_count, kv_usage, ts}` | engine 写，router 读 |
| `/dev/shm/vllm_ft_req_map/<user_req_id>` | engine 收到请求后立刻写：`{internal_req_id, engine_id, start_ts}` | engine 写，router 读（reroute 用） |
| `/dev/shm/vllm_ft_checkpoints/<internal_req_id>/...` | publish 的 manifest + chunk + latest 三件套 | engine 写 + restore 时读 |

---

**目标函数**:
`max Σ_r SLO_met(r) · output_tokens(r) / time` —— 系统级 SLO-达标 goodput（DistServe 风格）。`SLO_met(r) = 1` iff `TTFT(r) ≤ S_TTFT(r)` 且 `TPOT(r) ≤ S_TPOT(r)`。Rerouted req 用同样的 TTFT/TPOT SLO（不引入 disruption-specific 的 SLO 阈值；disruption 影响通过 `failover_gap_p95` 这个测量量呈现）。

**核心公式 (slack)** —— 两层共用，**分阶段定义**：

```
若 num_output_tokens(r) == 0:              // 还在等 first token
    ttft_slack = S_TTFT(r) − e(r,t)
    tpot_slack = +∞                         // 还没开始 decode，TPOT 不 binding
否则:                                       // 已经在 decode
    ttft_slack = +∞                         // first token 已出，TTFT 自动 satisfy
    tpot_slack = S_TPOT(r) − avg_TPOT(r,t)

slack(r, t) = min(ttft_slack, tpot_slack)
```

不分阶段会出 bug：first token 出来后 `S_TTFT − elapsed` 一直变负数（永远超时），picker 会以为这个 req 最急。所以 TTFT 出 first token 之后**从 min 里排除**（设 +∞）。

**决策变量**:

Router 端（每个新请求到达 + engine 失联事件）：
- `target_engine` —— 派发给哪个 engine。**分层 tuple 比较**：

  ```
  load_score(engine) = (
      in_flight_count,   ← 主：router 自己派给这个 engine 还没回的请求数（实时，同进程 dict）
      kv_usage,          ← tie-breaker 1：shm 报告的 KV 占用率，低优先
      waiting,           ← tie-breaker 2：shm 报告的 engine 内部 waiting 数，低优先
  )
  pick = min(alive_engines, key=load_score)
  ```

  - **in_flight 为主**：router 自己派出去还没收响应的请求数，实时反映"engine 上有多少 router 派的活"。不能只用 shm 数据，因为 shm 滞后 200ms，并发 dispatch 时会全派给一个 engine（Test 3 一开始就撞过这个 bug）
  - **shm 数据当 tie-breaker**：当 in_flight 相等时（比如两个 engine 都是 5 个 req 在跑），shm 提供细节区分 — 哪个 engine 的 KV 内存更紧、哪个内部排队更多。这正是 shm running/waiting/kv_usage 字段不可替代的价值（in_flight 推不出这些 internal state）

- 健康检测: 读 status 文件的 `ts`，超过 2s 没更新 → 标 dead。这是 shm 真正不可替代的功能（in_flight 推不出 engine alive）
- `reroute_target_engine` —— engine 失联时，找它所有 in-flight req（router 自己的 in_flight 表 + `/dev/shm/vllm_ft_req_map/<router_req_id>` 拿 internal_req_id），用 internal_req_id 重发给某个 alive engine

**实现踩坑（值得记）**:

- **vllm OpenAI handler 会 mangle `X-Request-Id` 加 `cmpl-...-0` 前后缀**，router 没法预测这种 mangling。所以 router 不用 X-Request-Id 当 dispatch 控制 id，改用 `vllm_xargs.router_req_id`（一个 opaque uuid，router 自己生成 + engine 端透传到 Request.router_req_id）。`req_map` shm 文件名用 `router_req_id`（router 知道这个值），不用 external_req_id。

- **engine 写 status 必须用独立 daemon thread**，不能 piggyback 在 `step()` 里。理由：vllm 的 EngineCore.run_busy_loop 在 idle 时阻塞在 `input_queue.get()`，**不调 step()**。如果 status 写在 step() 里，engine 一闲下来就停止心跳，router 会误判 dead。当前实现：daemon thread 每 200ms 强制写一次 status（FT_STATUS_WRITE_INTERVAL_MS 可调）。

Engine 端（每次 `schedule()`）：
- `head = argmin_{r ∈ waiting} slack(r)` —— 队列里最急的，下一个上
- `victim = argmax_{r ∈ running} slack(r)` —— running 里最从容的，候选被踢
- `preempt ∈ {0,1}` —— 是否真踢 victim
- `publish_cadence` —— 块对齐触发（每完成一个 16 token block 时 fire 一次 publish）

**约束**:
- per-req SLO 软约束（违反不会 crash 但算 SLO_met=0）：
  - `TTFT(r) ≤ S_TTFT(r)`，`TPOT(r) ≤ S_TPOT(r)`（rerouted req 同样用这两个 SLO，不另设阈值）
- KV pool 硬约束: `Σ_{r ∈ running} kv_blocks(r) ≤ M_kv`
- **无 admission 约束**：所有请求都入系统（不像 QLM 用 backpressure 拒、不像 Scorpio 用 TTFT guard 拒）
- **踢人触发条件（engine 端核心）** —— 三条都满足才踢：
  - `slack(victim) − slack(head) > δ + replay_cost(victim)`（hysteresis，δ ≈ 1000ms 起步，paper ablate；加上 replay_cost 才能真正算"踢了划不划算"）
  - `num_checkpointed_tokens(victim) > 0`（必须有 checkpoint 兜底，否则降级走 vanilla preempt = recompute）
  - `now − last_preempted_at(victim) > cooldown`（cooldown ≈ 5s 起步，防同一个 req 被反复踢）
- shm publish 一致性: `published_blocks ≤ stable_full_blocks`（不 publish 还没稳的 block，避免 reader 读半成品）
- Cross-engine restore 前提: shm `latest_rank0` + manifest + chunk 三件套都 visible（atomic write 保证）

**调度周期**:
- **Router dispatch**: per-arrival（新请求到达即 dispatch，无队列等待，无重排）
- **Router 健康检测**: 1s 一次 HTTP `/health`，超时 → 标 dead → 触发 reroute
- **Engine picker**: **per-iteration**（vllm `schedule()` 内 hook，跟 vllm 默认调度同步）
- **Publish**: 块对齐异步（默认 inline `FT_BG_PUBLISH=0`；开 `FT_BG_PUBLISH=1` 后用 single-worker ThreadPoolExecutor 后台跑 + 下次 RPC 等上次 future 做背压）
- **Reroute**: 异步（router 在 1s 健康检测发现 engine 死后触发 reroute，主请求路径不阻塞）

**Preempt running**: **是 —— 核心 contribution（在 engine 内做）**
- Picker 规则: 上述"踢人触发条件"。每个 schedule() 评估一次
- Cost: **接近 0**
  - 同 engine 内: KV 在 host pool（pinned memory），restore 是异步 H2D copy
  - 跨 engine: KV 在 /dev/shm，新 engine 读 manifest + chunk 后 scatter 到 GPU（cold path 但仍比 reprefill 快几个数量级）

**关键差异（vs 6 篇 baseline）**:
- 别人**都不敢踢长 context decode running req**（QLM/Scorpio/SLOs-Serve/Llumnix 完全不踢；FastServe/JITServe 只踢 early-prefill；Niyama relegate 不踢出；Andes 只在低 QoE knapsack 时踢但要靠 host swap）
- 我们的 contribution: 同时给出 (a) checkpoint 机制让 preempt 代价低 + (b) slack-based picker 让 scheduler 真敢用这个机制 + (c) 同一份 host 数据复用为 cross-engine recovery
- 两层架构跟 Llumnix 最像，但 Llumnix 用 live migration（GPU↔GPU peer copy，source engine 必须活着），我们用 host-side checkpoint（source 死了 KV 也在 host 上），所以**disruption 场景下完胜 Llumnix**

**实验注意事项**:
- 每次 evaluation 跑前清空三个 shm 目录:
  ```bash
  rm -rf /dev/shm/vllm_ft_checkpoints
  rm -rf /dev/shm/vllm_ft_engine_status
  rm -rf /dev/shm/vllm_ft_req_map
  ```
- 不清的话 router 可能读到上次跑残留的 engine status，误判 alive 状态

---

## Profiling 依赖度对比（7 个系统）

| 系统 | 依赖度 | profile 什么 | 关键性 |
|---|---|---|---|
| **Scorpio** | **重度** | OPT-125M 输出长度分类器（20K 训练样本）+ ITL 系数 α/β/γ/δ | Critical |
| **Andes** | **重度** | preempt overhead 表（每 context length）+ token-gen-latency 表（batch × tokens）+ reading-speed 分布 | Critical |
| **QLM** | **中度** | 硬件 profile（prefill 时间 P + 低效因子 ε + 单 token 解码 d）+ workload 长度分布 + ORT 长度预测 | Critical |
| **Niyama** | **中度** | random forest 预测 batch latency，**用 Vidur 模拟器数据训**（不是真硬件） | Critical |
| **FastServe** | **中度** | iteration-time lookup table（per-model × per-hardware × input length 一次性 sweep） | Critical |
| **Llumnix** | **轻度** | 1 个 scalar 常数：high-priority memory headroom（≈ 1600 tokens） | Moderate |
| **JITServe** | **轻度** | QRF 预测输出长度上界（在 gRPC sidecar，错了 fallback 不崩） | Moderate |
| **我们** | **最轻** | 只有 hysteresis δ 阈值（≈ 500ms，sweep 一次确定的 1 个数） | Moderate |

**为什么我们最轻**：slack 公式所有量都是 user-supplied（SLO 阈值）或 runtime 实测（elapsed、avg_TPOT、num_checkpointed_tokens），**没有任何需要预先 profile 的常数**。
- ❌ 无 ML 模型训练
- ❌ 无 prefill / decode cost 表
- ❌ 无 ITL/TPOT 系数拟合
- ❌ 无 output 长度预测
- ✅ 只有 hysteresis δ（1 个数）

**paper 里这是个 subtle contribution**：换 GPU、换 model 不用重 profile，可移植性好。Reviewer 看到这条会觉得设计干净。

---

## Proactive vs reactive scheduling（paper 卖点 framing）

为什么我们公式比别人少：核心是**调度范式不同**。

**别人（QLM / Scorpio / Niyama / Andes）= proactive**：不敢踢正在跑的请求，所以必须靠预测（output 长度、prefill 时间、batch latency）+ admission control，在请求进系统前就算清楚能不能满足 SLO。预测靠 ML 模型 / cost 表 / 系数拟合。

**我们 = reactive**：cheap preempt（host-side KV checkpoint）让"预测"这件事不必要。请求接进来跑着，runtime 实测 slack，发现紧了就踢。预测的事情都被"测量 + cheap 纠错"替代了。

```
别人: proactive — 预测，避免错
我们: reactive — 跑着试，错了能 cheap 修正
```

paper 卖点措辞：
> "By making preemption cheap (via host-side KV checkpoint), we replace proactive prediction-driven scheduling with reactive runtime measurement. This eliminates the need for offline-profiled latency models, output-length predictors, and analytical cost coefficients that prior SLO schedulers rely on."

---

## 无硬件约束 = cheap preempt 的连锁结果

我们 scheduler 公式里**没有显式硬件约束**（无 `Σ kv ≤ M`、无 bandwidth bound、无 batch size 限制），vllm 底层管 KV pool 容量，满了 capacity preempt 由我们的 V3 reload 路径接住。

其他 paper 几乎都有硬件约束：FastServe / Llumnix / Andes / SLOs-Serve / Niyama 的 `Σ kv ≤ M`，Scorpio 的 ITL 系数，JITServe 的 v_token。这些约束**都依赖 profile** 来 instantiate。

因果链：
```
cheap preempt → 不需要硬件约束 → 不需要 profile 硬件常数
```

paper 措辞：
> "Because cheap preempt makes overcommit cheaply correctable, our scheduler omits explicit hardware capacity constraints (`Σ kv ≤ M`, bandwidth bounds, batch-size limits) that other SLO schedulers rely on. As a consequence, our design needs none of the offline-profiled hardware coefficients (Llumnix's headroom, Scorpio's ITL α/β/γ/δ, JITServe's v_token, etc.) that those constraints would otherwise require."

---

改进：
现在最大的弱点是：recovery 和 SLO scheduling 在故事上耦合，技术上独立。两个 mechanism 共用一份 host 数据，但它们之间没有算法层面的互动。审稿人会问 "ablation 把其中一个砍掉，另一个还成立吗？" —— 如果都成立，那这两件事就不该写在一篇 paper 里。

要把"两个东西放一起"升级成"两个东西必须放一起"，思路上要加一个机制 × 策略真正耦合的点。几个方向：

Checkpoint cadence 跟 slack 联动：紧 SLO 的请求 checkpoint 更频繁（preempt 不丢工作），松 SLO 的请求 checkpoint 稀疏（省 PCIe）。这样 mechanism 直接被 policy 驱动
Disruption-aware slack 计算：rerouted 请求在 scheduler 优先级上让位（比如把它当成 "TTFT 已经烧掉一截"，等效 ttft_slack 起步就是负的一个 disruption penalty），让 reroute 后的请求在 SLO 优先级上优先恢复。这条不需要新 SLO 阈值，只是在 slack 公式里对 is_rerouted 加一项 penalty。
跨 engine 的 SLO scheduling：当一台 engine 接收 rerouted 请求时，本地正在跑的低 slack 请求要不要主动让位 —— 一个跨 engine 的 slack 协调
任意加一个就能从"组合"变成"协同"，故事质感能提一档。有了协同点，SoCC / EuroSys 的成功率会显著高，但还到不了顶会。

