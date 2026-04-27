# Serverless + PCIe-only Pivot — Idea Summary

## 背景

当前系统（Benders solver + adaptive checkpoint）在 8B/A5000/dp=2 设置下的数据：

- V2 完整系统（checkpoint + reload recovery）：-11% goodput vs NR
- V2-reprefill（checkpoint + reprefill recovery）：-14% goodput vs NR
- V2-NoCkpt（solver-only，无 checkpoint）：-9.3% goodput vs NR（4/9 strict wins，不显著）
- 唯一正向结果：W5 Moderate 长 context 过载下 +9.9% completion rate 保护

结论：**在当前 setting 下，checkpoint 机制是净负收益**。Re-prefill 在 8B 上只需要 ~300ms，checkpoint 整套链路的 overhead 超过了它的恢复收益。原 idea 的核心假设（"故障后重算昂贵"）不成立。

---

## 核心洞察

原 idea 的前提（re-prefill 昂贵）在两个场景下**真正成立**：

1. **PCIe-only 多卡部署**（无 NVLink）
2. **Serverless / ephemeral replica 部署**（lifecycle event 频繁）

这两个场景在现实部署中**高度重叠**——serverless LLM serving 几乎必然跑在 PCIe-only 的 commodity hardware 上（L40S、A10G、L4 等）。

### Why PCIe-only

- NVLink 是 A100/H100 SXM 版本的奢侈品，PCIe 版本没有
- 学术机构、创业公司、企业私有部署主要用 L40S / A6000 / RTX 6000 Ada
- TP（张量并行）下，PCIe 带宽 ~25 GB/s vs NVLink 600 GB/s，**慢 24 倍**
- PCIe-only + TP 下 prefill 要付巨额"通信税"
- **Re-prefill 成本被放大 5-10 倍**，而 checkpoint reload 绕过 TP 通信（每个 GPU 从本地 host memory 读自己那份 KV），**checkpoint 的优势被显著放大**

### Why Serverless

在 serverless / auto-scaling / spot instance 部署下，"故障"事件的频率从**一年几次**变成**一小时几次**：

| 事件 | 频率 | 影响 |
|---|---|---|
| Auto scale-down | 分钟级 | 丢 in-flight 请求 |
| Spot preemption | 小时级 | 丢 in-flight 请求 |
| 多租户迁移 | 分钟-小时级 | 丢 in-flight 请求 |
| Cold start / 热池切换 | 视策略 | 丢 in-flight 请求 |
| 硬件故障 | 月-年级 | 丢 in-flight 请求 |

每次 lifecycle event 的代价被放大：
- **Cold start**（模型加载）：70B 从磁盘加载 30-60 秒
- **Warm start**（有热池）：5-15 秒
- **重新 prefill in-flight 请求**：每个几秒到几十秒

总恢复时间从**秒级**变成**分钟级**。

---

## 新 framing

> Serverless LLM serving 运行在 PCIe-only commodity GPU 上，频繁的 lifecycle event 让 re-prefill 成为主要 overhead（cold start + prompt 重算 + TP 通信税）。现有系统（ServerlessLLM 处理模型 cold start，vLLM 处理单租户 serving）都没解决 **in-flight 请求状态在 lifecycle event 下的保存问题**。我们用 Benders-based admission + tiered KV checkpoint 解决这个 gap。

### 三个维度自然锁死在一起

1. **硬件**：L40S / A10G / L4 是 serverless 的主流硬件（不需要额外辩护）
2. **Workload**：serverless 的 lifecycle event 制造真实频繁的"故障"
3. **算法**：你的 checkpoint + admission 在这个场景是**必需的组件**，不是 "可能有用"

---

## 与现有工作的定位

**ServerlessLLM (OSDI 2024)** 是最相关的 prior work：

- 他们解决：模型权重的 cold start（locality-aware 调度 + warm pool）
- **我们解决**：请求状态的 cold start（in-flight KV cache 保留 + cross-replica handoff）

**互补而不是竞争**。合在一起才是完整的 serverless LLM serving。

其他相关工作：
- **Splitwise (ISCA 2024)**：prefill/decode 分离，关注硬件异质性
- **AlpaServe**：model parallelism for serving
- **SLOs-Serve**：SLO-aware 调度

---

## 算法层面的新 contribution

为了适配新 setting，算法需要以下扩展：

### 1. Communication-aware cost model

原 cost model：
```
Δreplay = U / replay_throughput
```

新 cost model：
```
Δreplay = compute_cost + tp_comm_cost
tp_comm_cost = num_layers × 2 × all_reduce_time(context_len, bandwidth)
```

在 L40S 上 `bandwidth` 按 PCIe 实测（~25 GB/s），NVLink 集群按 600 GB/s。算法根据硬件自适应。

### 2. TP-aware decode capacity

TP 下 decode 的瓶颈从 compute 转到 all-reduce 带宽：
```
decode_capacity_TP = min(compute_capacity, allreduce_capacity)
```

### 3. Tiered checkpoint storage

不再只存 host memory，引入多层存储：
```
L1: GPU HBM              (KV cache 原生位置)
L2: Host CPU memory      (当前实现)
L3: 本地 NVMe SSD        (replica 死了还在)
L4: 共享分布式存储        (跨 replica 可访问)
```

新策略：根据访问频率、故障概率、成本权衡决定存到哪一层。

### 4. Cross-replica checkpoint handoff

新 replica 启动时，从 registry 发现存活的 checkpoint，直接接管 in-flight 请求状态，避免 cold start + 重新 prefill。

### 5. Lifecycle-event-aware admission

Solver 的 admission 决策考虑 replica lifecycle：
- 临近 scale-down 时不接新请求
- Cold start 后优先恢复 checkpoint 请求
- 成本驱动：新请求 vs 状态保留的 tradeoff

### 6. Hardware-adaptive checkpoint policy

**同一套算法**，在不同硬件上自动调整：
- NVLink 集群：re-prefill 便宜，少存
- PCIe 集群：re-prefill 贵，多存
- Serverless：lifecycle event 频繁，主动存

这是论文的主卖点——"**一套方法覆盖多个场景**"。

---

## 期望的实验结果

### Re-prefill vs Checkpoint reload 交叉点

| 场景 | Re-prefill 成本 | Checkpoint reload 成本 | Ratio |
|---|---|---|---|
| 8B A5000 dp=2（当前）| ~300 ms | ~200 ms | 1.5× |
| 30B FP16 TP=2 L40S + 32K context | ~6-8 秒 | ~0.4 秒 | **15-20×** |
| 70B FP8 TP=2 L40S + 32K context | ~8-11 秒 | ~0.3 秒 | **25-30×** |
| Serverless cold start + 8K prefill | 30-60 秒 | ~1 秒 | **30-60×** |

在新 setting 下 checkpoint 优势**放大 10-50 倍**。

### 主要 metrics

- **P99 end-to-end request latency** 跨 lifecycle event
- **In-flight preservation rate**（event 发生时请求存活率）
- **Cost ($/req)**：GPU-hour + storage
- **Recovery time after lifecycle event**

---

## 实验设计概览

### Phase 0：Motivation 验证（1-2 天）

在 L40S 上跑 prefill benchmark，验证 re-prefill 成本：
- Cold start 时间（70B / 30B / 8B）
- Prefill 成本（8K / 32K / 64K context，TP=1 vs TP=2）
- NCCL all-reduce bandwidth 实测

**判断标准**：64K prefill > 5 秒 → motivation 成立，继续。

### Phase 1：Lifecycle event 代价分解

量化 lifecycle event 下状态保留的价值：
- Cold start + re-prefill vs warm pool vs 你的系统
- 不同 event rate 下的对比

### Phase 2：End-to-end 主实验

Workload：
- W_Steady（稳定 + 定期 scale-down）
- W_Bursty（突发请求 + 频繁 auto-scale）
- W_Preempt（spot preemption 场景）
- W_LongCtx（长 context + lifecycle）

Baseline：
- NR-Cold（每次 cold start + re-prefill）
- NR-Warm（warm pool 模拟 ServerlessLLM 策略）
- Full-Replicate（全量 checkpoint 上界）
- Our-System

### Phase 3：Ablation

- 去掉 tiered storage
- 去掉 cross-replica handoff
- 去掉 Benders admission
- 去掉 adaptive checkpoint
- Only L2+L3（无共享存储）

### Phase 4：敏感度分析

- Event rate：1/hour → 60/hour
- Context length：4K → 128K
- 模型大小：8B / 30B / 70B FP8
- Storage 成本：NVMe / 共享存储
- DP size：2 / 4 / 8 replicas

找到 checkpoint 开始赢的 **decision boundary**。

---

## 硬件选择

组里可用：4× workstation，每台 2× L40S 48GB，1.48TB RAM

### 可行的 serving 配置

| 配置 | 模型 | 部署方式 | 说明 |
|---|---|---|---|
| 单 workstation | 8B FP16 | dp=2 | 基础场景 |
| 单 workstation | 13B FP16 | dp=2 | 单卡装得下 |
| 单 workstation | 30B FP16 | TP=2 (dp=1) | 无 replica |
| 单 workstation | 70B FP8 | TP=2 (dp=1) | 无 replica |
| **2× workstation** | **30B FP16** | **TP=2 × DP=2** | **有 replica，FT 可做** |
| **2× workstation** | **70B FP8** | **TP=2 × DP=2** | **有 replica，FT 可做** |
| 4× workstation | 70B FP8 | TP=2 × DP=4 | 完整 FT 实验 |

### 关键限制

- **L40S 无 NVLink**，跨 GPU 通信走 PCIe
- **跨节点无 InfiniBand**（需确认具体网络带宽）
- TP 必须限制在**单节点内**，DP 可跨节点

---

## 模拟 serverless 的做法

**不需要真的部署到 AWS Lambda / Knative / Kubernetes**。研究里的 serverless 是模拟的：

- Replica 启停 = `subprocess.Popen` / `os.kill`
- Scale event = 当前 fault injection 框架的扩展
- Cold start 计时 = 新 engine 从 spawn 到健康的时间
- Lifecycle event 序列 = 合成规则 or Azure Functions trace 驱动

现有 `experiments_v2/run.py` 框架基础足够，扩展 ~几百行 Python 就能模拟 lifecycle。

---

## 代码改动估算

| 模块 | 新增 LOC | 修改 LOC |
|---|---|---|
| Tiered checkpoint storage | 500-800 | 200-300 |
| Cross-replica handoff | 800-1200 | 100-200 |
| Lifecycle event handling | 400-600 | 200 |
| Communication-aware cost model | 300-500 | 200 |
| 实验框架扩展 | 600-900 | 300 |
| 测试 | 400-600 | - |
| **合计** | **~3000-4600** | **~1000-1200** |

参考：合作者这一轮 push 的代码量约 `+4800 / -280`，属于同一量级。

---

## 风险与开放问题

### 技术风险

- 70B FP8 on L40S TP=2 的 vLLM 支持度
- 跨节点 DP 的网络瓶颈（取决于 elves 间网络速度）
- 分布式 handoff 协议的一致性和 race condition
- Tiered storage 策略调参

### 研究风险

- ServerlessLLM 及后续工作可能已经覆盖了部分 claim
- Serverless LLM 是 hot topic，竞争激烈
- 需要搞清楚"什么是别人没做的"

### 待验证

- **最关键**：Phase 0 benchmark 结果，验证 re-prefill 在 L40S 上真的贵
- TP=2 on L40S 的 all-reduce 实际带宽
- Cold start 的真实时间（受磁盘 I/O、shared memory 等影响）

---

## 需要跟导师讨论的点

1. **是否值得做这个 pivot**（vs 接受负面结果 / 等 70B 硬件 / 其他方向）
2. **Scope**：做完整 serverless 故事 vs 只做 PCIe-only 故事
3. **硬件资源**：能否把 2-4 台 elves 长期占用
4. **投稿目标**：OSDI/SOSP 级别的野心 vs ATC/SoCC 的稳妥路线
5. **时间预算**：接受 3-5 个月的完整开发，还是压缩 scope 赶 deadline

---

# 替代 Pivot: Cross-Node Overflow Routing

> 比 serverless 改动小，但同样把故事从"单节点 FT 恢复"升级到"集群级 admission + routing"。
> 记录于 2026-04-26（Step 5 修复后 W8/Heavy 单 cell 跑出 V2 admission rate 41% 触发的想法）。

## 触发问题

修复（`FT_CKPT_TRUE_ASYNC=1` + `FT_CKPT_NONBLOCK=1`）后 V2 在 W8/Heavy/F2_Mid 数据：

- in-flight at fault: 7（终于有积压可恢复）
- goodput 4.76 / TPOT p95 463 ms / completion 100% (admitted)
- **admission rate 仅 41%**（193 个请求里 Benders 拒了 114 个）

**问题**：「admission 拒 60% 请求」单看是缺点。读者会质疑「拒了那么多还说自己好？」

## 想法

现在的 V2：
```
请求 → Benders：能保证 SLO 吗？
       ├─ 能 → 接收
       └─ 不能 → 拒绝（用户体验：连接失败）
```

改成：
```
请求 → 节点 A Benders：能保证吗？
       ├─ 能 → 接收
       └─ 不能 → 转发到节点 B（FT-aware overflow）
                  ├─ 能 → B 接收
                  └─ 不能 → ... → 全集群满 → 拒绝
```

这就是 **overflow routing**——云上 LB（Envoy / K8s HPA + 多副本）的标准做法。

## 故事重定位

**当前**："单节点 FT 恢复"——读者听 "V2 拒了 60%" 觉得难看。

**加 overflow routing 后**："FT-感知的集群级 admission + routing"
- 拒绝 ≠ 丢失，是**主动负载均衡**
- 在 N 节点集群下，V2 用 FT-aware 决策比朴素 round-robin 多服务 M% 请求 + 少 X% SLO 违规
- 跟生产部署直接对接，更实用

## 三档实施成本

| 档 | 思路 | 工作量 |
|---|---|---|
| **A. 文字重定位** | 不动代码，把"admission 拒了 60%"重述为"V2 主动决策 100% 已接受能完成"，留 "future work: route to peers" | **0 天** |
| **B. 简单 RR proxy** | FastAPI 50 行 proxy 挂多个 vLLM 实例，拒绝时 round-robin 重试。需要 4+ GPU 跑 2 节点 | **1-2 天 + 4 GPU 实验** |
| **C. FT-aware router** | 节点共享容量、in-flight、故障预算状态。Router 用集群级 Benders 决策 | **2-4 周** |

## 与原 idea / serverless pivot 的关系

| 维度 | 单节点 FT (原 idea) | Cross-node overflow | Serverless pivot |
|---|---|---|---|
| 节点数 | 1 节点 / 多 GPU 副本 | N 节点静态集群 | 弹性节点池 |
| 故障类型 | GPU 故障 | GPU + 节点级容量 | + lifecycle event |
| 状态保留 | KV checkpoint 到 host | 同左 + cross-node KV migration | 同左 + tiered storage |
| 故事强度 | 弱（被 Step 5 否定） | 中（"过载下保护用户"）| 强（hot topic） |
| 工程量 | 已有 | +1-4 周 | +3-5 个月 |

**Overflow routing 的位置**：比单节点 FT 强，比 serverless 容易。如果论文需要"比单节点强一点的故事但不想全栈做 serverless"，这是中间路径。

## 决策点

如果需要写 paper claim，**至少要做 A**（文字重定位 + future work）。Step 5 修复后的数据自带这个故事——不做 A 就解释不清"V2 admission 41% 为啥不算缺点"。

是否做 B/C，看：
- 论文需不需要更强的故事（B 给一个具体 baseline 击败 RR）
- 时间预算（B 一周内出数据，C 一两个月）
- 跟 serverless pivot 二选一还是叠加（overflow + serverless 可以叠加，但太大）

## 代码改动初步设想（B 档）

**vLLM 服务端**：
- 拒绝时返回 `429 Too Many Requests` + 自定义 header `X-Reject-Reason: capacity-exhausted`（不是 5xx，让 router 知道可重试）
- Benders 在拒绝时**可选**返回 hint：哪些故障场景下确定不能服务（让 router 决策）

**Router**：
```python
# FastAPI proxy
async def route(req):
    nodes = get_nodes_by_load_order()  # 简单 RR 或按 in-flight 排序
    for node in nodes:
        resp = await forward(req, node)
        if resp.status == 429:
            continue  # 试下一个
        return resp
    return 503  # 全满了
```

**实验设置**：
- 4 GPU = 2 节点（每节点 dp=2）
- W8/Heavy 同样 193 个请求
- 对比：① 单节点 V2（41% admit）② 双节点 V2 + overflow（接近 100% admit）③ 双节点 NR + RR（无 admission，看完成率）

