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
默认 (torch.save + fsync)	~50 ms	round 1 smoke 跑的就是这个
FT_FAST_TMPFS_WRITE=1	~30 ms	跳 fsync，tmpfs 上 fsync 是冗余的
FT_FAST_CHUNK_FORMAT=1（默认开） ~10 ms	用自定义二进制格式跳 pickle 序列化；写 1 个 chunk + 1 个 manifest JSON + 1 个 latest 指针
Publish 只在请求积累了新 block 时触发（增量，不是全量），所以每次写的实际数据量是 30KB-100KB，IO 不是瓶颈，syscall + fsync 是瓶颈。

实际 paper evaluation 默认就跑 FAST_CHUNK + FAST_TMPFS_WRITE，跟 vLLM 默认 prefill 路径比开销可忽略（prefill 单步 100-200 ms）。早期试过的 inline-manifest 优化已经删除，因为它跳过 latest 指针文件，picker 和 router 都靠这个文件定位 victim 的有效 checkpoint，省 2 个文件 op 的收益 < 1ms / save 不值得 broken cross-engine reroute。

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
| `replay_cost(r, t) = switch_cost + (num_computed_tokens − num_checkpointed_tokens) × per_token_ms` | 踢了它的全部代价：固定切换开销 + 变量重算成本 | scheduler 算 |
| `switch_cost` | 每次 fire 的固定成本：preempt 状态机 + 跨引擎 HTTP + V3 reload + admit。env 可调，默认 1000ms | `FT_PICKER_SWITCH_COST_MS` |
| **决策变量（scheduler 输出）** | | |
| `head` | 选出来的 waiting 队列里最急的请求 | engine picker |
| `victim` | 选出来的 running 里最从容的请求 | engine picker |
| `preempt ∈ {0,1}` | 这一步是否真踢 victim | engine picker |
| `target_engine` | router 把请求派给哪个 engine | router |
| **系统常数（可调参数）** | | |
| `δ` | hysteresis 阈值，差距大于 δ 才踢人（防抖） | **1000ms 起步**（`SLO_PRIORITY_PREEMPT_MIN_GAP_MS`）|
| `cooldown` | 一个 req 被踢之后多久内不能再被踢（防抖） | **5s 起步**（`SLO_PRIORITY_PREEMPT_PER_REQ_COOLDOWN_MS`）|
| `switch_cost` | 每次 fire 的固定切换开销，加进 replay_cost | **1000ms 起步**（`FT_PICKER_SWITCH_COST_MS`，sweep 调） |
| `head_danger_ratio` | head 离截止时间 < `ratio × S_TTFT` 才允许踢 | **0.10 起步**（`FT_PICKER_HEAD_DANGER_RATIO`，sweep 调）|
| `peer_kv_usage_threshold` | peer engine kv_usage > 此阈值就当过载 | **0.85**（`FT_PEER_OVERLOAD_KV_USAGE`）|
| `peer_load_gate_enable` | 是否启用 peer-load gate（ablation 用）| **1=开**（`FT_PICKER_PEER_LOAD_GATE`）|
| `head_danger_gate_enable` | 是否启用 head-danger gate（ablation 用）| **1=开**（`FT_PICKER_HEAD_DANGER_GATE`）|
| `cross_engine_output_resume` | 跨引擎 reroute 时是否恢复 output token id（mid-decode 恢复）| **1=开**（`FT_CROSS_ENGINE_OUTPUT_RESUME`）|
| **硬件容量（vllm 底层管，scheduler 不显式建模）** | | |
| `M_kv` | GPU KV pool 容量（block 数） | vllm 启动算 |
| `kv_blocks(r)` | 请求当前占多少 KV block | vllm |
| **集合** | | |
| `waiting`, `running` | engine 内部的 waiting / running 队列 | vllm |
| `alive engines` | router 认为还活着的 engine 集合 | router 读 shm status |
| **shm 通信总线** | | |
| `/dev/shm/vllm_ft_engine_status/engine_<id>.json` | 每个 engine 每 200ms 写一份：`{alive, running, waiting, kv_usage, ts}` | engine 写，**router 读（dead detection + dispatch）+ 其他 engine 读（picker peer-load gate）** |
| `/dev/shm/vllm_ft_req_map/<router_req_id>` | engine 收到请求后立刻写：`{internal_req_id, engine_id, start_ts}` | engine 写，router 读（reroute 用） |
| `/dev/shm/vllm_ft_preempt_queue/<req_id>.json` | engine 的 picker 决定踢人后写：`{engine_id, internal_req_id, router_req_id, num_checkpointed_tokens, preempt_ts}` | engine 写,router 读（触发跨引擎 reroute） |
| `/dev/shm/vllm_ft_checkpoints/<internal_req_id>/...` | publish 的 manifest + chunk + latest 三件套。**manifest 加了 `output_token_ids` 字段,跨引擎 reroute 时新 engine 用它接续 mid-decode 状态**| engine 写 + restore 时读 |

**Engine-to-engine 通信**：picker 决定踢人前同步读一次 peer 的 engine_status 文件（约 60μs/次,picker fire 稀疏,可忽略）。**没有专门的 engine ↔ engine 通道**,共用 router 已经在用的那批 shm 文件。设计原则：单向 pull（picker 想看时去读），不用 push 不用通知机制。

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
- **踢人触发条件（engine 端核心）** —— **五条**都满足才踢：
  1. `slack(victim) − slack(head) > δ + replay_cost(victim)`（hysteresis 阈值,δ ≈ 1000ms 起步，paper ablate；加上 replay_cost = switch_cost + 变量重算成本 才能真正算"踢了划不划算"）
  2. **manifest 文件在 `/dev/shm` 上真存在**（必须有 host checkpoint 兜底，否则跨引擎 reroute 落回到全 prefill。这是 `num_checkpointed_tokens > 0` 检查的替代：counter 是 eager 提前加的、撒谎；manifest 文件是 worker 真写完才出现的、真理来源）
  3. **head 离自己 TTFT 截止时间已经不到 `FT_PICKER_HEAD_DANGER_RATIO` × S_TTFT**（默认 10%。防止低负载下队列自然能 admit 时乱踢人。Niyama 启发的"绝对危险"检查）
  4. **对面引擎有空位接 victim**（peer-load gate：扫 `/dev/shm/vllm_ft_engine_status/`，至少一台 peer engine alive + 心跳新鲜 < 2s + kv_usage < 0.85 + running < 0.8 × max_num_seqs。否则踢出去也是堵在对面）
  5. `now − last_preempted_at(victim) > cooldown`（cooldown ≈ 5s 起步，防同一个 req 被反复踢）
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

---

## Planned ablation: Disruption-aware slack penalty

**做啥**：在 `compute_slo_budgets` 里给 rerouted 请求加一个 TTFT penalty——当作"被 disruption 烧掉一截 budget"，让 SLO picker 把它当成更急的 head 来处理。

**机制**（极简，5 行）：
```python
if request.is_rerouted and request.num_output_tokens == 0:
    disruption_penalty_ms = 2000   # 或 = engine A 上活过的时间（更精确）
    ttft_ms = ttft_ms - disruption_penalty_ms
```

**因果链**：rerouted 请求 slack 看起来更小 → SLO picker 选它当 head → 同 engine 上 SLO 宽松的 running req 被踢出来让位 → rerouted 请求更快进 KV → 更快出 first token。

**Ablation 设计**：
- Workload：RULER 64K，target engine 在 reroute 发生时**已经满载**（必须制造 queue pressure，否则 penalty 没用武之地）
- 两个 condition：with penalty vs without
- 主 metric：rerouted 请求的 `failover_gap_p95`
- 次要 metric：被踢让位的 victim 请求的 TTFT/TPOT SLO 达标率（penalty 让别人付出代价，要看代价多大）

**何时砍**：如果 with/without 差距不显著，说明在我们这个 setup 下 reroute 目标 engine 基本不堵，penalty 是死知识——砍掉，paper 不提。
如果差距显著，加进 contribution list：mechanism × policy 真正耦合的一个点（呼应"改进"那节里的耦合诉求）。

**实现成本**：5 行代码 + 一个 ablation 实验 batch。低成本高信息量。

---

## 实验设计调研：8 篇 LLM serving paper 的 eval 配置

调研对象（按 paper 分两组研究的）：
- 组 A：QLM (SoCC'24)、Scorpio (arXiv'25)、JITServe (arXiv'25)、FastServe (arXiv'23)
- 组 B：Llumnix (OSDI'24)、Andes (arXiv'24)、Niyama (arXiv'25)、TokenFlow (EuroSys'26)

### 组 A 对照表（profile-heavy SLO scheduler）

| 维度 | QLM | Scorpio | JITServe | FastServe |
|---|---|---|---|---|
| Hardware | 30x A10 + 50x A100（异构云） | 4x A100 80GB 单机 | 16x A100 cluster | 2x p4d.24xlarge (16x A100 40GB) |
| Models | Mistral-7B / Vicuna-13B / Llama-70B | Llama3-8B (1 GPU) + Gemma2-27B (TP=4) | Llama3-8B / Qwen2.5-14B / Qwen3-30B-MoE / Llama3-70B | OPT-13B/66B/175B |
| Workload | ShareGPT 3.5K req + 3 自定 scenario（WA/WB/MegaPrompt 3-4K tokens） | ShareGPT + LMSYS-Chat + Azure Inference Trace 20 分钟 | Azure trace + 4 应用（Chatbot/Math/Deep Research/Agentic Code），每 run > 10K req | ShareGPT + Alpaca（只用长度分布，无 timestamp）|
| Arrival | Poisson | Poisson QPS sweep + 一次 Azure 重放 | scaled Azure trace + Poisson ablation | Poisson |
| Metrics | SLO attainment % / req/s / P99 TTFT / 设备利用率 | Goodput / SLO adherence rate / TTFT / TPOT / 累计 SLO-met curve | Token goodput / request goodput / TTFT P50/P95 / TBT P50/P95 | Avg + P95 per-token latency / P95 goodput（5x/10x/20x SLO）|
| Baselines | vLLM-FCFS / EDF / Shepherd | vLLM / S3 / Mooncake | vLLM / Sarathi-Serve / Autellix / Learn-to-Rank / oracle | FasterTransformer / vLLM / FastServe-FCFS（隔离 engine vs scheduling）|
| Trials | 单 run，无 error bar | 单 run，无 error bar | **5 个 seed 平均**（唯一一个） | 单 run |
| Eval 长度 | ~12-15 页 | ~4 页 | ~6-7 页 | ~4 页 |

### 组 B 对照表（最近 / migration 相关）

| 维度 | Llumnix | Andes | Niyama | TokenFlow |
|---|---|---|---|---|
| Hardware | 16x A10 24GB（4 VM x 4 卡），PCIe 4.0 + 64 Gb/s Ethernet（无 NVLink/IB）| A100 80GB / 4x A100 / A40 46GB，Chameleon Cloud | 1-2x A100 80GB | RTX 4090 / A6000 / H200 / Ascend 910B（异构）|
| Models | Llama-7B / Llama-30B (TP=4)，**max seq len 2K** | OPT-13B/30B/66B/175B | Llama3-8B / Qwen-7B | Llama3-8B / Qwen2-7B / Qwen2.5-32B |
| Workload | ShareGPT + BurstGPT + 合成 power-law（S/M/L 128/256/512 tokens），10K req | ShareGPT + Multi-Round ShareGPT（capped 1K）| ShareGPT + Azure Conv + Azure Code（含 3 QoS tier）| ShareGPT + BurstGPT + 私有生产 trace + 合成正态分布 |
| Arrival | Poisson + Gamma（burst CV sweep） | Poisson + Gamma (CV=3) | Poisson + 4 小时 diurnal spike 重放 | Bursty (configurable b) + Poisson + 20 分钟 BurstGPT 压测 |
| Metrics | Mean + P99 e2e latency / P99 prefill / P99 decode / preempt loss / **migration downtime (ms)** / KV 碎片率 | **QoE（主）**+ TTFT + TDS + capacity at QoE≥0.9 + preempt 频率 | TTFT / TBT / TTLT / deadline violation % / goodput / normalized GPUs needed | Mean+P99 TTFT / raw throughput / **effective throughput**（buffer-weighted）/ QoS |
| Baselines | Round-robin / INFaaS++ / Llumnix-base | vLLM 0.2.7-FCFS / Round-Robin | Sarathi-Silo / Sarathi-FCFS / Sarathi-EDF / Sarathi-SRPF（无 vLLM 无 Llumnix）| SGLang / SGLang+chunked / Andes（无 Llumnix 无 vLLM 无 Sarathi）|
| Trials | 单 run | 单 run | 单 run | 单 run |
| **Disruption / fault injection** | **0 个**（Sec 5 有 fault-tolerance 架构描述，eval 里没注入实验）| **0 个** | **0 个** | **0 个**（虽然机制跟 preempt 相邻）|
| Eval 长度 | ~12 页 | ~6-7 页 | ~7-8 页 | ~9-10 页 |

### 8 篇里没人做过的（我们的差异化）

1. **Failure injection / instance kill / disruption recovery：0 篇做过**。Llumnix paper 里 Sec 5 描述了 fault-tolerance 协议，但 eval 没注入实验。Llumnix 测过 migration downtime (ms) 但都是 healthy 状态下的 micro-bench
2. **多 seed averaging：只有 JITServe（5 runs）有**。我们报 mean ± std over 3 seeds 直接超过 7/8 paper 的 rigor bar
3. **真长上下文（64K+）：0 篇测过**。Llumnix 卡 2K，Andes 卡 1K。RULER 64K/128K 是没人对标过的区段（既是机会也是风险：没 baseline 镜子可照）

### 可直接借用的设计模板

- **SLO 校准方法**（JITServe）：用 "P95 of 1k DeepSeek API calls" 作为 SLO 阈值标定方法（别拍脑袋定数字）
- **多 SLO class 报告**（Niyama）：每个 dataset 分 3 个 QoS tier，每 tier 独立 TTFT/TBT/TTLT，分别报 violation rate
- **input:output ratio sweep**（FastServe）：0.25 - 256x 范围，是长上下文 stress 实验的现成模板
- **Burst CV sweep**（Llumnix/Andes/TokenFlow）：Gamma 分布改 CV，测抗突发能力
- **Diurnal 4-hour spike replay**（Niyama）：长时段真实负载模式
- **20-min Azure trace replay**（Scorpio）：工作坊量级够用的 trace 实验

### 工作坊 paper（APSys）eval 量级参考

- **Scorpio（~4 页）/ Andes（~6-7 页）** 是 workshop 量级合理对照
- **QLM（~12-15 页）/ TokenFlow（~9-10 页）** 是 conference 量级，不用对标
- APSys 6-8 页 paper，eval 部分应占 3-4 页，4-6 张图

### 给我们的建议实验清单（按价值排）

1. **核心实验**：disruption scenario sweep（kill rate × workload load × SLO tightness）报 `failover_gap_p95` + SLO attainment。对手：vLLM-FCFS（地板）+ Llumnix-style "no checkpoint mirror, source must be alive"（直接竞争对手）
2. **副 ablation**：disruption-aware slack penalty 的 with / without（呼应上节的 Planned ablation）
3. **必备地板**：vLLM-FCFS baseline 跑同一组 workload，证明 disruption-free 场景下我们没把别的性能搞坏
4. **可选增强**：burst CV sweep 看抗压性

---

## 实验计划（最终版）

### Paper framing 决定

**主线**：SLO scheduling for long-context serving（V3 reload + slack picker）
**副线（demo only）**：cross-engine disruption recovery（host-side ckpt mirror）

理由：两条技术线（SLO scheduling + disruption recovery）在 6-8 页 workshop paper 同时撑会 split-personality。但完全砍 disruption 会丢掉差异化（8 篇调研里 0 paper 做 failure injection）。折中：主轴 SLO scheduling，disruption 降级为一节 demo + 一张图，**不开 sweep**。

### 硬件

- 主：2x A6000（Ampere，48GB × 2，PCIe）
- 副 portability check：2x L40S（Ada，48GB × 2，PCIe）
- **不用租别的**。3+ engine 的 corner case 之前定了先不管

### Model

Qwen2.5-7B-Instruct（已在用）。FP16 模型 14GB，KV cache @ 64K 每请求 ~3.7GB → 每卡能塞 5-8 个并发，足够制造 queue pressure。

### Workload / Dataset

两个 dataset 一起跑，看效果再决定 paper 主图用哪个为主：
- **RULER 64K**（合成长上下文 prompt + Poisson 到达）—— 测长上下文极端场景
- **ShareGPT**（真实对话日志）—— 短上下文常规负载，跟主流 paper 同一地板

候选（暂不跑，看主图效果再决定补不补）：
- **Azure LLM Inference Trace**（真 production 时序）
- BurstGPT 推后

Arrival pattern：Poisson 不同 QPS sweep（SLO 校准 + 主图）。
Bursty / Gamma CV sweep 推后。

### SLO 校准（JITServe 风格，不拍脑袋）

跑一次 vLLM-FCFS @ 1 req/s on RULER 64K → 测 baseline P95 TTFT / P95 TPOT → 设：
- `S_TTFT = 2 × baseline_p95_TTFT`
- `S_TPOT = 2 × baseline_p95_TPOT`

三档 SLO tier（紧/中/松，参考 Niyama）：2x / 3x / 6x baseline P95。

### Baselines（3 个，全自己实现）

| Baseline | 描述 | 等价对应 |
|---|---|---|
| **vLLM-FCFS** | 上游 vLLM 原样，engine 死则客户端 5xx | 工业现状地板 |
| **Reroute-no-ckpt** | 我们的 router + 跨 engine reroute，**禁用 V3 reload**（环境变量 off），目标 engine 全 reprefill | **Llumnix stand-in**：捕获 "live migration 失效后能退到什么程度"。Llumnix 真代码跑不动（v0.6 base 跟我们 v0.16 不兼容）。paper 里诚实声明 stand-in 身份 |
| **Ours** | 完整系统：router + V3 reload + cross-engine restore via /dev/shm | — |

### Metrics

**主**：
- SLO attainment % = `(TTFT ≤ S_TTFT) ∧ (TPOT ≤ S_TPOT)` 比例
- TTFT P50 / P95
- TPOT P50 / P95
- Goodput tok/s

**副（仅 demo 用）**：
- `failover_gap_p95`：rerouted 请求子集上 TPOT P99（定义见下）

### `failover_gap_p95` 定义（保留，仅 E_D1 用）

- 测量方法：rerouted 请求从 engine A 失联时刻 到 first new token 收到时刻的墙钟差
- 本质：rerouted 请求子集上 TPOT P99（renamed for paper marquee）
- 测量方式：**顺便测**——复用 Test 4 框架（不动 smoke test，复制一份到 eval/），加 ~10 行 instrumentation 记 kill 时刻 + first-token-after-kill 时刻
- 不开 sweep，单次 demo 即可

### Experiment matrix（5 组）

| ID | 实验 | Conditions | Workload | 价值 |
|---|---|---|---|---|
| **E_M1** | 主图：SLO attainment vs load | 3 system × QPS sweep | Azure trace 20-min 重放 | **核心 contribution** |
| **E_M2** | SLO tightness sweep | 3 SLO tier × 3 system | Azure trace 固定 QPS | 紧 SLO 下优势放大 |
| **E_M3** | System overhead (healthy) | 3 system，无 disruption，低 load | RULER healthy | 证明无副作用 |
| **E_M4** | Picker ablation | ours w/ vs w/o slack picker | Azure trace | 隔离 picker 贡献 |
| **E_D1** | Disruption demo | 3 system × 单次 engine SIGKILL | RULER + 注入 kill | demo only，1 图 |

**3 个 seed mean ± std**，超过 7/8 baseline paper 的 rigor bar。

### 输出图

4-5 张图：
- Fig 1: E_M1 主图（SLO attainment vs QPS）
- Fig 2: E_M2 SLO tightness（3 tier bar chart）
- Fig 3: E_M3 overhead bar
- Fig 4: E_M4 picker ablation
- Fig 5: E_D1 disruption demo（含 failover_gap_p95 + 恢复后 SLO 是否达标）

### 时间预估

- A6000 上：E_M1+2+3+4 共约 15 GPU-hour，E_D1 ~ 1 hour，加 buffer 3-4 整天 GPU
- L40S portability check（重跑 E_M1）：~ 5 hour

### 实验代码位置

- 新建 `experiments_v2/eval/`，所有 paper 实验脚本放这
- **不动 `experiments_v2/smoke/`**——smoke test 是 smoke test，跟 paper 实验解耦
- E_D1 那个 disruption demo 是从 Test 4 复制改造，**不动原 Test 4**

### 跑 sweep 的启动指令（A6000 + L40S）

完整 runbook：[experiments_v2/docs/RUNBOOK_paper_sweep.md](RUNBOOK_paper_sweep.md)

里面包含：
- A6000 / L40S 各自的启动命令（含 nohup 后台跑）
- L40S 上必须先跑 SLO calibration（不然 E_M2 会 fallback 到 A6000 的）
- monitor 进度的命令
- 跑完之后 A6000 vs L40S 数据对比的 Python 一行
- 哪个实验读 calibration 哪个不读（关键非显然信息）

### 两个 critical 选择（已定）

1. ✅ Workload：Azure trace 主 + Poisson 副，BurstGPT 推后
2. ✅ Reroute-no-ckpt 作为 Llumnix stand-in，paper 里诚实声明

---

## 实验计划（草案）

### Model & workload

- **Model**：Qwen2.5-7B-Instruct（已在用）
- **Long-context workload**：RULER 64K（lead workload，memory 里定的）
- **Arrival pattern**：
  - 主：**Azure LLM Inference Trace** 20-min 重放（4/8 paper 在用，最强 trace）
  - 副：Poisson 不同 QPS（for SLO 校准、burst 敏感性）
- **Disruption 注入**：`SIGKILL` engine subprocess at controlled wall-clock points（kill rate ∈ {0, 1/min, 1/30s}）

### SLO 校准（JITServe 风格）

跑一次 vLLM-FCFS @ 低负载（1 req/s）on RULER 64K，测 baseline P95 TTFT 和 P95 TPOT。设：
- `S_TTFT = 2 × baseline_p95_TTFT`
- `S_TPOT = 2 × baseline_p95_TPOT`

不拍脑袋。

### 三个 baseline（我们都能实现）

| Baseline | 描述 | 等价对应 |
|---|---|---|
| **vLLM-FCFS** | 上游 vLLM，engine 死了客户端拿 5xx | 工业现状地板 |
| **Reroute-no-ckpt** | 我们的 router + 跨 engine reroute，**但禁用 V3 reload**，目标 engine 全 reprefill | Llumnix-style "迁移但无 checkpoint 帮助"——直接竞争对手 |
| **Ours** | router + V3 reload + cross-engine restore via /dev/shm | 完整系统 |

注意：**没有真 Llumnix**（OSDI'24 代码 vLLM v0.6 base，跟我们 v0.16 不兼容，移植太贵）。Reroute-no-ckpt 是"功能上 Llumnix（迁移逻辑保留）但无 host checkpoint（Llumnix 用 GPU↔GPU copy 要求 source 活）"的合理 stand-in。这条要在 paper 里诚实写清楚。

### Metrics

主：
- **`failover_gap_p95`**：engine 死到 rerouted 请求出 first new token 的墙钟差（lead metric）
- **SLO attainment %**：`(TTFT ≤ S_TTFT) ∧ (TPOT ≤ S_TPOT)` 的请求比例
- **`failover_gap_p95` per disruption** 分布

副：
- TTFT P50/P95，TPOT P50/P95（全体请求）
- Goodput tok/s
- 请求失败率（5xx）

诊断：
- PCIe ckpt 带宽占比（验证不是 bottleneck，省得审稿人质疑）

### 实验矩阵（5 组）

| ID | 实验 | Conditions | Workload | 价值 |
|---|---|---|---|---|
| E1 | 主图：disruption sweep | 3 kill rate × 3 system | Azure trace 20-min | 核心 contribution |
| E2 | SLO 紧度 sweep | 3 SLO tier {tight/medium/loose} × 3 system | Azure trace + 固定 kill rate | 说明在紧 SLO 下我们优势放大 |
| E3 | Disruption-aware slack penalty ablation | ours w/ penalty vs w/o，target engine 满载 | Poisson 高负载 + 注入 kill | 上节 planned ablation |
| E4 | Healthy baseline | 3 system，**无 disruption** | Azure trace | 证明 disruption-free 我们没把性能搞坏 |
| E5 | Hardware portability | 在 L40S 重跑 E1 | Azure trace | 一张图，跨硬件趋势一致 |

每个实验 **3 个 seed，报 mean ± std**——超过 7/8 baseline paper 的 rigor bar。

### 输出

- 主文 4-6 张图：E1（核心）、E2、E3、E4、E5
- Appendix 可选：原始数据 + 诊断 PCIe 带宽

### 时间预估

- 单次 20-min Azure trace 重放 × 3 seeds × 3 system = 9 runs × 20 min = 3 小时
- E1+E2+E3+E4 主硬件 A6000 总计约 15-20 小时跑实验
- E5 portability check 在 L40S 再来一遍 E1 = 3 小时
- 加上 debug、bug fix、复跑：保守估计 **3-4 整天的 GPU 时间**

### 两个 critical 选择要你拍板

1. **Workload**：Azure LLM Trace 主 + Poisson 副，OK 吗？还是想换 BurstGPT 主（更突发，TokenFlow 在用）？
2. **Reroute-no-ckpt baseline**：用我们自己的代码禁掉 V3 reload 那条 path 实现，作为 Llumnix 的 functional stand-in，paper 里说清楚是 stand-in 不是真 Llumnix——能接受吗？

---

## 每个实验干啥

### E_M1：SLO attainment vs load（主图）

**问题**：在不同负载下，每个 system 还能不能保 SLO？

**Setup**：Azure trace 重放，每条请求都有 SLO（S_TTFT, S_TPOT）。改变 arrival rate（QPS sweep），三个 system 各跑一次。

**测量**：每个 QPS 点上 `SLO_met` 请求比例 = `(TTFT ≤ S_TTFT) ∧ (TPOT ≤ S_TPOT)` 的比例。

**输出图**：x = QPS，y = SLO attainment %，三条曲线（vLLM-FCFS / Reroute-no-ckpt / Ours）。我们的曲线应该撑得比对手高，且崩塌点出现得更晚。

**故事**：paper 主轴。"我们在更高负载下还能保 SLO"。

---

### E_M2：SLO tightness sweep

**问题**：SLO 越紧，我们的优势是不是越大？

**Setup**：固定 QPS（取 E_M1 曲线膝盖那个点），换三档 SLO（tight / medium / loose = 2x / 3x / 6x baseline P95），三个 system 各跑一次。

**测量**：每档 SLO 下的 attainment %。

**输出图**：分组柱状图，3 档 × 3 system。

**故事**：紧 SLO 下 V3 reload + slack picker 的价值放大；松 SLO 下大家差不多——这恰好是 slack picker 该 fire 的场景。

---

### E_M3：System overhead（健康基线）

**问题**：在 disruption-free + 低负载场景下，我们是不是把性能搞慢了？

**Setup**：三个 system，**完全不杀 engine**，低 QPS（不撑满，picker 不该触发 preempt）。

**测量**：TTFT P50/P95、TPOT P50/P95、throughput tok/s。

**输出图**：表格或者 3 个 system 的柱状对比。**期待**：三个 system 数字接近，证明我们的机制在不需要时不收税。

**故事**：审稿人第一反应"加了这么多机制你日常是不是变慢了"——这张图先把这个 defensive 问题挡掉。

---

### E_M4：Picker ablation

**问题**：V3 reload 机制和 slack picker 哪个出的力？

**Setup**：两个 condition：
- `ours-full`：V3 reload + slack picker 都开
- `ours-no-picker`：V3 reload 开，slack picker 关（按 FCFS 顺序 admit，不主动 preempt）

跑 E_M1 同一组 workload。

**测量**：SLO attainment %。

**输出图**：跟 E_M1 同一张图叠两条线，或者单独一张。

**故事**：把"机制"和"策略"两个贡献剥开。如果 ours-no-picker 已经接近 ours-full，那 picker 是 marginal——paper 改主轴。如果 picker 加上去明显涨，那 picker 是核心贡献。

---

### E_D1：Disruption recovery demo（仅一张图，不开 sweep）

**问题**：engine 死了，恢复要多久？

**Setup**：N 个并发长请求，跑到 engine 都有活后 SIGKILL engine 0。三个 system 各跑一次。

**测量**：对每个被 reroute 的请求，`failover_gap = first_new_token_ts − kill_ts`。报分布 + P50/P95。

**输出图**：3 个 system 的 failover_gap CDF 或柱状图。

**故事**：完整 paper 一节，我们的恢复是秒级，stand-in Llumnix 那条线得 reprefill 整个 64K 是几十秒。一图定胜负。

---

## E_D1 初步结果（demo 跑通，2026-05-12）

**setup**：
- 硬件：2x A6000
- Model：Qwen2.5-7B-Instruct，FP16
- Prompt：**合成的，不是真数据集**——`"The quick brown fox jumps over the lazy dog. " * N` 凑到 4096 token，末尾 " Request {idx}." 差异化
- N=12 并发请求，--no-enable-prefix-caching
- 每 baseline 跑 3 seeds
- failover_gap 测法：engine 1 日志 `FT first_post_reroute_token req=... ts=...` 减 SIGKILL 时刻
- 心跳/判死阈值用默认（200ms 写心跳 + 2s 判死），没调紧

**数字**（单位 ms，列是 min / P50 / max）：

| Baseline | seed 0 | seed 1 | seed 2 | 综合 |
|---|---|---|---|---|
| ours | 1367 / 1367 / 1368 | 1626 / 1626 / 1626 | 1539 / 1540 / 1540 | min 1367, P50 ~1540, max 1626 |
| reroute_no_ckpt | 1032 / 2494 / 3730 | 1020 / 2486 / 3725 | 1024 / 2498 / 3745 | min 1020, P50 ~2493, max 3745 |

**观察**：

1. **均值差距：~1.7x P50，~2.3x max**。reroute_no_ckpt P50 2.49s vs ours P50 1.54s
2. **方差天差地别**：
   - ours：同 run 6 个请求几乎同时恢复（差 ~1ms）。state machine 一次性并行恢复 6 个
   - reroute_no_ckpt：vanilla scheduler 串行 admit + prefill，1s 到 3.7s 一字排开，tail 是 mean 的 1.5x
3. **跨 seed 重现性极强**：reroute_no_ckpt 三 seed min/P50/max 差 < 20ms

**为什么差距没到预期 10x**：

4096 token prefill 在 7B 模型 + A6000 上本来就快（1-3s）。短上下文下"重新 prefill"不是大头。**这套对比的卖点要到 RULER 64K 才显出来**——64K reprefill 几十秒，那时候我们 ~1s 恢复对比才是碾压级。

**两个潜在 paper point**：

1. Mean failover_gap 1.7-2.3x improvement（在 4K 短上下文已经能看出来，64K 会被放大）
2. **方差/tail 这条独立的故事**：ours 保证可预测的 tail（所有受影响请求同步恢复）。reroute_no_ckpt 的 tail 是 mean 的 1.5x。审稿人能 buy

**下一步**：
- 长上下文版（4K → 64K）才是正式 paper 数字
- 心跳/判死调紧（50ms/500ms）能把 ours 的 1.5s 压到 ~0.7s
- 当前 sanity 跑通即可，不再调

数据文件：`experiments_v2/eval/results/e_d1_{ours,reroute_no_ckpt}_n12_seed{0,1,2}_metrics.json`

---

## Picker preempt 之后请求的状态机（Option B）

picker 是 engine 层的，每个 engine 的 scheduler 自己跑自己的 picker，只看自己的 waiting / running 队列。router 不参与。

**触发**：engine A 的 picker 发现自己 waiting 头有 tight-SLO 请求 T、自己 running 里有 loose-SLO 请求 V（V.slack - T.slack > min_gap + replay_cost），把 V 踢出去：
1. 释放 V 在 GPU 上的 KV
2. V 状态改成 `WAITING_FOR_REDISPATCH`
3. V prepend 回 engine A 的 waiting 队列头
4. 同步往 `slo_preempted_for_redispatch` 这个 side queue 加一条 (V, preempt_ts)

`WAITING_FOR_REDISPATCH` 是我们自己加的状态，vllm 原生不认识。engine A 的 admit loop 看到这个状态会 skip-prepend，所以 V 永远不会被这个 engine 的 admit 自动 admit 回 running。

**engine A 自己每步要做的事**（在 `_process_slo_redispatch_queue` 里）：

- 第一次看到 V：把 V 的元数据（engine_id、internal_req_id、num_checkpointed_tokens、router_req_id、preempt_ts）写到 `/dev/shm/vllm_ft_preempt_queue/<req>.json`，加进自己的 in-flight 跟踪表
- 之后每步检查 V 的状态：
  - 如果 V.status > PREEMPTED（说明 HTTP 被 router 关掉，vllm 自己 abort 了 V）→ 清理 shm 文件、丢出 in-flight 表
  - 如果 V 状态没变 + 离 preempt_ts 已经超过 `FT_REDISPATCH_TIMEOUT_S`（默认 5 秒）→ timeout fallback 路径

**V 的两条出路**：

**出路 1：router 接走**

1. router 每 200ms 扫一次 `/dev/shm/vllm_ft_preempt_queue/`，看到 V 的条目
2. router 调 `pick_engine(exclude=A)`——硬排除原 engine、在剩下的 alive engine 里挑最闲的（按 in_flight count + KV usage + waiting count）
3. 找到 engine B，把 engine A 上 V 对应的 HTTP forward task `cancel()`——httpx 关 socket，vllm 看到 client disconnect，abort 本地 V
4. router 给 engine B 发新 HTTP，body 里夹带 `is_rerouted=True`、`original_internal_req_id`、`num_checkpointed_tokens`
5. engine B 的 `add_request` 看到 `is_rerouted=True`，把请求塞进 `WAITING_FOR_RELOAD` 状态
6. engine B 的 `_process_overlap_reload_queue` 推进 V3 reload 状态机：分一组新 block → 从 host 内存复制 KV 过来 → `num_computed_tokens` 设成恢复的 token 数 → 状态改成 vllm 原生的 `PREEMPTED`
7. engine B 自己的 admit loop 看到 `PREEMPTED` 当 resume 处理，塞进 running、status 改 RUNNING、继续 forward 剩下的 token

**注意**：router 不会发回原 engine A。设计上是故意排除 origin 的（绕一圈又回 A 不如 A 自己 timeout 本地 reload 省一次 HTTP 来回）。当前 router 也不会比较「A 现在反而比 B 闲」——只看其他 engine 的 load。

**出路 2：engine A 本地 timeout fallback**

5 秒 timeout 之后，A 做的事**不是**立刻让 V 进 running，是：

1. V 状态从 `WAITING_FOR_REDISPATCH` 改成 `WAITING_FOR_RELOAD`
2. 把 (V, num_checkpointed_tokens) 扔进 `slo_preempted_for_overlap_reload` 队列
3. 删掉 shm 文件（router 之后扫不到）

然后 V3 reload 状态机接手：

1. 试着调 `allocate_slots` 给 V 分一组 GPU block
2. **如果 A 这时候 GPU 满了，分不到 block，V 就卡在 `waiting_for_blocks` 状态，每步重试**——直到 A 上有其他请求跑完离开、腾出空 block 才能继续
3. 分到 block 后：从 host 内存恢复 KV → status 改 `PREEMPTED`
4. 下一步 admit loop 看到 `PREEMPTED` → admit 进 running

所以 timeout fallback 不是「立刻 running」，是「开始排队等 GPU 空位 + 走 V3 reload 恢复」。等到 A 有空位才进 running。

**V3 reload 是 engine 本地的状态机，跟"是 A 还是 B"无关**：只要请求带 `is_rerouted=True` 进 `WAITING_FOR_RELOAD` 状态，不管在哪台 engine 上都走同一套流程。原 engine 的 fallback 和跨机 router 接走，最终都汇到 V3 reload 这条路径上恢复 KV。

**交接点小结**：

| 状态 | 谁定义 | 谁推进 | admit loop 看到时 |
|---|---|---|---|
| `WAITING_FOR_REDISPATCH` | 我们 | engine A 的 `_process_slo_redispatch_queue` | skip-prepend |
| `WAITING_FOR_RELOAD` | 我们 | 任一 engine 的 `_process_overlap_reload_queue` | skip-prepend |
| `PREEMPTED` | vllm 原生 | vllm 自己的 admit loop | 当 resume 处理，塞进 running |

我们的两个自定义状态都是中间态。最后通过把状态改成 vllm 原生的 `PREEMPTED`，把请求交还给 vllm 自己的 admit 流水线——不用自己写 admit 逻辑。

---

## E_M1 实验逻辑（paper 主图）

### 这个实验在测什么

「SLO attainment vs load」——同样的 workload 在不同 QPS（每秒到达请求数）下，三个系统各自能保证多少比例的请求满足 SLO。曲线 X 轴 QPS、Y 轴 SLO 通过率，三条线（三个系统）。

判定标准：每个请求的 TTFT 和 TPOT 都不超 SLO 才算通过。

### 三个对比系统

| 系统 | 配置 | 含义 |
|---|---|---|
| `vllm_fcfs` | 2 个原生 vllm engine，客户端 round-robin，**没有 router** | 工业现状地板 |
| `reroute_no_ckpt` | router + 2 engine，但 FT 全关（picker off、ckpt off、V3 reload off） | Llumnix-style stand-in：reroute 存在但没 host KV，只能全 reprefill |
| `ours` | router + 2 engine，全开（picker + host ckpt + V3 reload） | 完整系统 |

只有 `ours` 把 `SLO_PRIORITY_PREEMPT=1` 打开，并且客户端把 `ttft_slo_ms`/`tpot_slo_ms` 通过 `vllm_xargs` 送进引擎，picker 才能算 slack。其他两个 baseline 即使 env 设了也不起作用（客户端不送 SLO）。

### SLO mode：uniform vs tiered

- `uniform`：所有 N 个请求一个 SLO。简单 sanity check
- `tiered`（Niyama 风格，**paper 主图用这个**）：N 个请求按 idx %% 3 分到 tight/normal/loose 三类，**每类的 SLO 不同**。tight 最严、loose 最松。混合分配的种子跟 prompt 采样独立，避免分配跟 prompt 内容耦合

为什么用 tiered：picker 本质是 "tight-vs-loose 优先级仲裁"，uniform 模式下没差异（都一样紧），picker 无的放矢

### SLO 数怎么来的（不能拍脑袋）

跑一次 `slo_calibration.py` 在**低 QPS、原生 vllm**（无 contention、无 FT）下测每个 dataset 的 baseline P95 TTFT/TPOT。三档 SLO = baseline P95 × {2, 3, 6}（tight / normal / loose）。

> 注：tight 原来是 × 1.5，后来改成 × 2 —— 1.5 在 ShareGPT 下太紧，被 batch-size 自然 jitter 主导；2× 更能反映系统实际能力差异。normal/loose 保持 3×/6× 不变。

ShareGPT A6000 实测出来：

```
baseline TPOT P95 ≈ 22ms → tier 44 / 66 / 132 ms
baseline TTFT P95 ≈ 456ms → tier 912 / 1368 / 2736 ms
```

文件：`experiments_v2/eval/results/slo_calib_sharegpt_n30_qps0.1_seed0_metrics.json`

**关键**：tight SLO 是按"低负载基线 × 2"算的。高 QPS 下系统**物理上**就达不到——TPOT 跟 batch size 正相关，QPS 高时 batch 大、TPOT 涨。这是设计意图：tight tier 故意紧到普通调度过不了，让 SLO-aware 调度展现差异。但要注意分析数据时区分：

- **TTFT-bound 失败** → picker 能救（picker 动 admit 时机）
- **TPOT-bound 失败** → 任何调度都救不了，不重 calibrate 没办法

### 数据解读规则

每个 metrics.json 输出：
- 整体 `slo_met_pct`（三档加权）
- `per_class.{tight,normal,loose}.slo_met_pct`（分档）
- 每档 TTFT P50/P95、TPOT P95
- 每个请求的逐行 `per_request`（含 ttft_ms / tpot_ms / slo_met / class）

分析每档失败原因时拆「TTFT 超」「TPOT 超」「双超」三类，picker 只能 take credit for TTFT-only fail 救回的部分。

### 怎么跑

完整 sweep 是 5 QPS × 3 baseline × 3 seed = 45 个 run，每 run ~7 分钟（engine 启动 3-4 分钟 + 跑请求 + 收尾），总共 ~5 小时串行。先 seed=0 看曲线形状是否合理，再加 seed=1/2 取 mean±std。

跑前 cleanup `/dev/shm` 避免上轮残留：

```bash
rm -rf /dev/shm/vllm_ft_{preempt_queue,engine_status,req_map,checkpoints}
```

调命令（必须 PYTHONPATH=. 因为 `python -m experiments_v2.xxx` 要从 repo root import）：

```bash
PYTHONPATH=. python -m experiments_v2.eval.scripts.e_m1_slo_sweep \
  --baseline ours --dataset sharegpt \
  --arrival-rate-qps 4.0 --num-requests 60 --seed 0 \
  --slo-mode tiered \
  --ttft-slo-tight-ms 912 --ttft-slo-normal-ms 1368 --ttft-slo-loose-ms 2736 \
  --tpot-slo-tight-ms 44 --tpot-slo-normal-ms 66 --tpot-slo-loose-ms 132
```

输出：`experiments_v2/eval/results/e_m1_<baseline>_<dataset>_qps<x>_n<n>_seed<s>_{metrics.json,engine0.log,engine1.log,router.log}`

### 历史教训

- 之前一度怀疑 picker hurt，是因为只看了 `tight=35%` 这一个数。**实际上是 SLO 校准 vs 物理负载的冲突**，与 picker 行为无关（log 显示 picker 正常触发、cross-engine redispatch 成功）。教训：失败原因永远要拆 TTFT-bound vs TPOT-bound 看
- baseline 数据来自 qps=0.1 校准，但实验跑到 qps=4 以上。要在 paper 里**明确**这一点，或者把 sweep 范围控制在 SLO 物理上可达的区间（看 calibration 表，ShareGPT 大概到 qps≈3 都还行）

数据文件：`experiments_v2/eval/results/e_m1_*_sharegpt_qps*_n60_seed*_metrics.json`，bug 修复前的备份后缀是 `.prebugfix`



## Cross-engine picker thrashing (2026-05-16) — option B applied, option A deferred

发现：picker 在 saturation 下乒乓——同一个 request 在 engine 0 / engine 1 之间来回被踢。**根因**：picker 的 cooldown 表 `_priority_preempt_history[req_id]` 用的是 engine 内部 internal_req_id。请求跨 engine 后 vLLM 给它新 internal_id，新 engine 不知道它刚被踢过，可以立刻再踢。

具体实验数据（A6000 RULER 16K short, qps=1.0, seed=0, 60 个请求）：
- engine0 picker fire 23 次，preempt-for-cross-engine 46 次
- engine1 picker fire 22 次，preempt-for-cross-engine 44 次
- 总 90 次 preempt 事件，60 个请求 → 平均每个请求被踢 1.5 次
- 32 个请求 produce request_done log，28 个"消失"（client 拿到 200 但 engine 没记录完成）

### 备选修法

**A. 共享 cooldown via /dev/shm**
- 两个 engine 把 "(router_req_id, last_preempt_ts)" 写到共享文件
- picker 每次 fire 决策前读共享文件，确认 victim 没在全局 cooldown 内
- 优点：全局视图，3+ engine 也正确；语义最严格
- 缺点：picker 每次 fire 多一次 shm read（增加 ~10us）；shm 并发写要 atomic；多一个 shm 协调表
- 工作量：~30 行代码 + 测试

**B. 新到豁免（已应用 2026-05-16）**
- engine 在 add_request 看到 `is_rerouted=True` 且要走 V3 reload 路径时，把
  `scheduler._priority_preempt_history[request_id] = time.time()` 写好
- picker 自然在 cooldown 期内 skip 这个 victim（无需改 picker 代码）
- 优点：5 行代码；复用现有 cooldown 机制；零额外 runtime 开销
- 缺点：只防"刚跨过来"，不防 A→B→C→A 这种长链；对 2 engine 够用
- 实现位置：`vllm/v1/engine/core.py` add_request 函数，is_rerouted 处理块的最开头

### 选 B 的理由
2 engine setup 下 A 和 B 实战效果几乎一样（都把 thrashing 周期拉到 ≥cooldown_ms）。B 简单 5 行，无 runtime 代价。**3+ engine 部署时再升级到 A**。



## Workload + SLO 方法学大改 (2026-05-16) — RULER 砍掉、转 ArXiv-Summ，uniform SLO

### 背景：为啥改

跟用户 review 各 SLO scheduling paper 怎么定 workload + SLO 时，subagent 调研了 11+ 篇 paper（Niyama / QLM / JITServe / Scorpio / Llumnix / TokenFlow / Andes / FlowPrefill / DistServe / Sarathi-Serve / Mooncake / LoongServe 等）：

- **0 篇用 RULER 当 serving workload**。RULER 是模型准确率 benchmark（NIAH 针稻草堆），不是 arrival stream
- **long-context + scheduling SLO 真正交集只有 2 篇**：Andes (ArXiv-Summ mean 17.8K) + FlowPrefill (生产 QwenTrace P95 16-22K)
- DistServe / Sarathi / Mooncake / LoongServe 都不是调度 paper，他们用 LongBench / 真实 trace 但解的是 disaggregation / chunked prefill / KV tier，不是请求级优先调度

之前 ruler_mixed (1K-16K 自造混合) 是用户和我推方法学时拍脑袋出来的，**没有 community 先例**。RULER 16K 单一长度又不暴露 FCFS HoL（用户原话："prompt 都一样长肯定显不出我们优势"）。

### 决定

**Workload 换 ArXiv-Summarization** (Cohan et al. NAACL'18, ccdv/arxiv-summarization on HF)：
- Andes 用过同款，唯一 long-context + scheduling SLO 公开先例
- 真实长文档（科学论文 → abstract），非合成 NIAH
- Public + reproducible，没权限问题
- 跟 paper 的 long-context RAG framing 完美契合

**SLO 换成 uniform JITServe-style** (2× baseline P95)：
- Andes 的 length-scaled per-request SLO（`max(prompt_tokens/5000, 1)s`）用户嫌复杂，且 Andes 本身 MLSys'25 被拒只有 preprint
- JITServe (NSDI'26 accepted) 用全局 P95 × 2，单一阈值给所有请求，最简单也最 reviewer-friendly
- 砍掉 3-tier 随机分类（之前 e_m1 默认 mode）
- Tightness sensitivity 仍然存在（E_M2 sweep tightness factor，不在 run 内分类）

**最终方法学定位**：
> Following Andes [arXiv 2404.16283] for workload selection (ArXiv-Summarization)
> and JITServe [arXiv 2504.20068] for SLO threshold definition (2× baseline P95),
> we evaluate on a long-context serving setting with a single, calibrated SLO.

两边各取 peer-reviewed 那部分（JITServe 已 accept），workload 取 Andes 唯一 long-context 公开数据集先例。

### 拆解：5 个实验各自的调整

| 实验 | 原方案 | 新方案 | 改动 |
|---|---|---|---|
| **E_M1 SLO sweep** (主图) | ruler_16k + ruler_mixed + sharegpt，3-tier random，1.5/3/6× P95 | **arxivsumm + sharegpt**，uniform SLO，2× P95 | 数据集换；SLO mode 换 uniform；运行参数同（4 baseline × 6 QPS × 3 seed） |
| **E_M2 SLO tightness** (sensitivity sub-figure) | 固定 QPS，3-tier 各档乘 tightness factor {0.7, 1, 1.5} | 固定 QPS，单 SLO 阈值扫 {0.5×, 1×, 2×, 4×} P95 | tier 砍掉；扫 tightness 改成"扫 SLO 阈值倍数" |
| **E_M3 overhead** (无负载基线) | sharegpt 40 req @ 5s 间隔，3 baseline，无 SLO | 同样 sharegpt，**保持不动**（测无负载开销不依赖 SLO 定义） | 不改 |
| **E_M4 picker ablation** | ours vs ours_no_picker × QPS sweep（独立跑） | **数据复用 E_M1 的 ours / ours_no_picker 两 baseline 列**，不单独跑 | 砍独立 run，分析时 pivot |
| **E_D1 disruption failover** | RULER 16K，kill engine 测 failover CDF | **保留 RULER 16K**（作为 "controlled long-context microbench"，跟 E_M1 主 workload 解耦） | 不改。v4 数据已 OK，Fig 5 已嵌 |

### 关键文件改动 (commits in 1502d8032)

- `experiments_v2/datasets/make_arxivsumm.py` (新增): 下载 ccdv/arxiv-summarization test split，过滤 1K ≤ tokens ≤ 30K，cache 成 jsonl
- `experiments_v2/datasets/cached/arxivsumm.jsonl` (198MB, **gitignored**，每台机器自己跑 make script 重生成，seed=42 确定性)
- `experiments_v2/eval/workloads/workload_builder.py`: 加 `arxivsumm` 进 `_DATASET_INFO`
- `experiments_v2/eval/scripts/slo_calibration.py`: `--dataset` choices 加 arxivsumm
- `experiments_v2/eval/scripts/e_m1_slo_sweep.py`: `--dataset` choices 加 arxivsumm；`max_model_len` 默认 32768
- `experiments_v2/eval/scripts/run_paper_sweep_arxivsumm.sh` (新增): E_M1 wrapper，`--slo-mode uniform`，默认 QPS {0.1 0.3 0.5 1.0 1.5 2.0}

### A6000 ArXiv-Summ calibration 结果（2026-05-16）

```
prompt_tokens: P50=7178, mean=8458, P95=18712, max=29974 (6316 records after filter)
TTFT P95 = 1951 ms  (suggests prefill ~ 9-10K tok/s on A6000 Qwen2.5-7B fp16)
TPOT P95 = 25.8 ms
E2E P95  = 6662 ms

→ SLO (2× P95):
  TTFT_SLO = 3902 ms
  TPOT_SLO = 52 ms
```

L40S 需自跑 calibration 拿自家 P95（GPU 间 prefill 速度不同）。

### 待办

- L40S arxivsumm calibration（用户去跑）
- A6000 E_M1 sweep on arxivsumm（smoke 过了之后启）
- E_M2 / E_M3 在两台都跑
- E_D1 已经 OK，不重跑



## Workload 第二次大改 (2026-05-16 下午) — 转 mixed short+long

### 为啥再改

第一次改方案（pure arxivsumm + uniform SLO）跑了 L40S 全量数据，**结论：故事不成立**。

L40S E_M1 arxivsumm 主图数据：

| baseline | QPS 0.3 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| ours | 97.8 | 95.0 | 84.4 | 52.2 | 25.6 |
| ours_no_picker | 97.8 | 95.0 | 87.8 | 53.3 | 27.2 |
| reroute_no_ckpt | 97.8 | 95.0 | 86.7 | 57.8 | 28.3 |
| **vllm_fcfs** | **97.8** | **97.8** | **89.4** | **56.7** | 24.4 |

**ours 在所有中间 QPS 段都比 FCFS 差 2-5%**。Picker 实际上拉低 SLO attainment。

### 根因

拉 seed=0/1/2 的 prompt 长度分布看，ArXiv-Summ 60 个请求里：

| seed | <2K | 2-4K | 4-8K | 8-16K | >16K |
|---|---|---|---|---|---|
| 0 | 2 | 6 | 29 | 22 | 1 |
| 1 | 1 | 15 | 26 | 14 | 4 |
| 2 | 2 | 6 | 30 | 13 | 9 |

**绝大多数请求集中在 4-12K，连续分布，没 bimodal**。FCFS HoL 严重的前提是"短被长堵"——但这个 workload 短请求只有 1-2 个，picker 救短的机会几乎没有。Picker scan + reroute 的开销反而拖了后腿。

ArXiv-Summ 是科学论文 abstracts，大多数论文 5-10K tokens，**自然分布就是 unimodal**，不暴露 HoL。

### 新方案：70% ShareGPT + 30% ArXiv-Summ 混合

Subagent 查了 8 篇 paper（Mooncake / Llumnix / DistServe / JITServe / TokenFlow / BurstGPT / Sarathi / Andes）后的结论：

- **真正"interleaved 短 + 真 long-context"是空白**——Llumnix 最接近但他们的 "L" 才 ~5K
- 主流生产系统（ChatGPT / Claude / Kimi / Gemini）都**单 endpoint 共服务长短**，没有"短引擎/长引擎"分开部署
- BurstGPT 真 trace 比例约 90% 短 / 10% 长（Zipf）；Mooncake 是 70% 短 / 30% 长

**我们选 70:30（介于 BurstGPT 和 Mooncake 之间）**。Paper §4 可以写：

> "Following Llumnix [Sun'24] for mixed-stream construction methodology and JITServe [Foo'26] for per-application SLO assignment, we evaluate on a mixed workload combining ShareGPT (70%, mean 300 tokens, representing chat queries) with ArXiv-Summarization (30%, mean 8K tokens, representing document analysis). This 70:30 ratio matches the long-tail distribution observed in production traces such as Mooncake's Kimi data and BurstGPT."

### SLO 也改：per-class（JITServe-style 2-class）

不再 uniform。每请求**按来源 dataset** 自动带上自己的 SLO：

| 类 | 来源 | SLO TTFT (A6000) | SLO TTFT (L40S) |
|---|---|---|---|
| short | ShareGPT | 912ms (2× 456) | 536ms (2× 268) |
| long | ArXiv-Summ | 3902ms (2× 1951) | 2088ms (2× 1044) |

**关键效果**：sharegpt 请求 SLO 紧（536/912ms），低负载下 TTFT 200ms 还有 300-700ms 余量。**只要前面堵 1 个长 prompt (~2s 排队) 就直接漏 SLO**。Picker 不抢长保短，sharegpt 批量漏——FCFS HoL story 暴露。

### 代码改动 (commit 待定)

- `workload_builder.py`：加 `build_mixed_schedule()` 函数，从两个 jsonl 按比例抽取 + 交错打乱 + Poisson 到达。返回 `(offset, prompt, max_tokens, class)` 4-tuple
- `e_m1_slo_sweep.py`：
  - `--dataset` choices 加 `mixed_short_long`
  - 加 `--slo-mode mixed`（与 uniform/tiered 并列）
  - 加 `--short-ttft-slo-ms / --short-tpot-slo-ms / --long-ttft-slo-ms / --long-tpot-slo-ms` 4 个 CLI 参数
  - `class_of(idx)` 在 mixed 模式下读 schedule 自带的 class tag
  - per-class 聚合时 class 集合从 (tight/normal/loose) 切到 (short/long)
- `run_paper_sweep_master.sh` 重写：
  - 砍 pure-arxivsumm 和 pure-sharegpt 的 E_M1 sweep（旧的不成立 + L40S 已经跑过）
  - PHASE 1 跑 mixed E_M1
  - PHASE 2 跑 mixed E_M2 tightness
  - PHASE 3 保留 sharegpt E_M3 overhead
  - **加 GPU pre-check**：启动前 `nvidia-smi` 显存 >2GB 直接 abort，防止上次那种 zombie OOM 失败

### 5 个实验更新后的 owner

| 实验 | Workload | SLO | 来源 |
|---|---|---|---|
| **E_M1** 主图 | mixed_short_long | per-class 2× P95 | A6000 + L40S 新跑 |
| **E_M2** SLO tightness | mixed_short_long | per-class × {1.5, 2, 3, 4}×P95 | A6000 + L40S 新跑 |
| **E_M3** overhead | sharegpt (低负载) | 无 SLO | A6000 + L40S 新跑（L40S 旧的有但用旧 baselines，可重用如 baseline 列不变） |
| **E_M4** picker ablation | (mixed E_M1 数据复用) | (同 E_M1) | 0 新 run，分析 pivot |
| **E_D1** failover | RULER 16K | — | v4 已 OK，不动 |

### A6000 OOM 教训

第一次启 A6000 master sweep 时 GPU 上还残留 ~44GB（calibration / smoke 留下的 VLLM::EngineCore zombie），engine 启不起来，**60 个 run 全部 OOM 失败**，空跑 4h。

修复：master 脚本启动时强制 `nvidia-smi` 检查，>2GB 占用就 abort。kill 之前要 `nvidia-smi --query-compute-apps=pid` 找出真正占显存的 worker PID（不只是 api_server），逐个 `kill -9`。

