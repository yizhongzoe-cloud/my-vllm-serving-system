# 优化问题求解方式分析

代码如何求解优化问题的详细总结。

## 核心观察

**代码并没有求解论文中写的那个 MIP（混合整数规划），而是用了一个贪心在线启发式算法，把联合优化分解成了逐个决策 + 约束检查的管道。**

---

## 1. 准入决策 y_j — 贪心排序 + 逐项可行性检查

论文目标是 `max ∑ G_j · y_j`，这是一个带多种约束的 0-1 背包问题。代码的做法是：

**实现位置**: `vllm/v1/core/sched/ft_scheduler_impl.py:164-195`

```python
# 按 G_j 降序排列 — 贡献大的优先
pending = sorted(
    self._pending_ft_admission,
    key=lambda r: r.generation_len,
    reverse=True,
)
# 逐个尝试准入
for request in pending:
    admitted = self._ft.admit_request(request)
    if not admitted:
        self._base.finish_requests(request.request_id, FINISHED_ABORTED)
```

这是一个**贪心背包启发式**：把物品（请求）按价值（G_j）降序排列，逐个尝试放入背包。每次放入时检查所有约束是否仍然满足，不满足就拒绝。

这不是最优解（贪心背包没有保证），但对于在线场景（请求实时到达，不能等凑齐一批再求解）来说是合理的。

---

## 2. 初始路由 x_{j,r} — Least-Loaded 启发式

论文是一个多维分配问题。代码直接用了最简单的负载均衡：

**实现位置**: `vllm/v1/core/replica_manager.py:128-150`

```python
def route_request(self, request):
    healthy = self.get_healthy_replicas()
    # 选当前活跃请求数最少的 replica
    best = min(healthy, key=lambda r: r.num_active_requests)
    if best.num_active_requests >= best.max_num_seqs:
        return None
    return best.replica_id
```

没有做全局优化（比如考虑"这个请求放在 replica A 后，后续请求的路由会不会更差"），就是**单步 least-loaded**。

---

## 3. 鲁棒可行性 Ω_k — 最坏情况分析（不枚举场景）

论文要求约束对所有 `|ω| ≤ k` 的故障场景都成立，这意味着 `C(n,k)` 种场景。代码**没有枚举所有场景**，而是利用了一个关键观察：

> 对于同构 GPU（所有 replica 参数相同），最坏情况就是失去 k 个 replica 后剩 `n-k` 个。

**实现位置**: `vllm/v1/core/replica_manager.py:214-290`

```python
def check_capacity_under_failures(self, requests, max_failures):
    surviving = num_replicas - max_failures  # 直接算最坏情况

    # 约束1: 序列数 ≤ surviving × max_num_seqs
    if total_requests > surviving * max_seqs_per_replica:
        return False

    # 约束2: ∑P_j ≤ surviving × C_r^{pre} × H
    if total_prefill_tokens > surviving * min_prefill_throughput * H:
        return False

    # 约束3: ∑G_j ≤ surviving × C_r^{dec} × H
    if total_decode_tokens > surviving * min_decode_throughput * H:
        return False
```

注意这里用了 `min` throughput（最差的 replica），这保证了无论哪 k 个挂了，剩下的都能承载。**这是一个保守但精确的松弛**——对同构情况是精确的，对异构情况是保守的。

类似地，SLO 检查也用最坏情况：

**实现位置**: `vllm/v1/core/replica_manager.py:292-343`

```python
def check_slo_under_failures(self, request, max_failures):
    # 最坏情况：最快的 k 个挂了，请求落在最慢的 surviving replica 上
    prefill_throughputs = sorted(r.prefill_throughput for r in all_replicas)
    worst_prefill = prefill_throughputs[0]  # 最慢的那个
    worst_ttft_ms = (request.prompt_len / worst_prefill) * 1000
    if worst_ttft_ms > request.ttft_slo_ms:
        return False
```

---

## 4. Checkpoint Level ℓ_j — 基于进度的确定性规则

论文写 `ℓ_j ∈ {0,1,2}` 是一个决策变量。代码**完全不做优化**，而是用了一个确定性的阈值规则：

**实现位置**: `vllm/v1/core/checkpoint_controller.py:64-84`

```python
def get_checkpoint_level(self, request):
    progress = request.generation_progress  # 已生成/预期总数
    if progress >= 0.60:
        return 2   # 高频 checkpoint
    elif progress >= 0.25:
        return 1   # 低频 checkpoint
    else:
        return 0   # 不 checkpoint
```

这不是优化出来的，是人工设计的启发式。0.25 和 0.60 这两个阈值是固定配置。

Checkpoint 频率也是确定性的：

| Level | 步数间隔 | 时间间隔 |
|-------|---------|---------|
| 0 | ∞ (不做) | ∞ |
| 1 | 64 steps | 2.0 sec |
| 2 | 16 steps | 0.5 sec |

---

## 5. Failover-Gap 约束 — 公式直接计算

这是最接近论文的部分。代码直接实现了公式：

**实现位置**: `vllm/v1/core/checkpoint_controller.py:203-259`

```python
def estimate_recovery_cost(self, request, ...):
    # T^{det} + S^{ckpt}/B^{ld} + U_j/C^{rep} + 1/C^{dec}
    total = (detection_time_sec           # T^{det}
           + checkpoint_size / load_bw     # S^{ckpt}/B^{ld}
           + uncovered_tokens / replay_tps # U_j/C^{rep}
           + 1.0 / decode_tps)            # 1/C^{dec}
    return total
```

这个在两个地方被用作**约束检查**（不是优化）：
- **准入时** (`vllm/v1/core/sched/ft_scheduler.py:239-264`)：估算最坏情况恢复代价，如果超过 `failure_gap_slo_ms` 就拒绝
- **故障恢复时** (`vllm/v1/core/recovery_manager.py:272-311`)：估算实际恢复代价，如果超过 SLO 就 drop 该请求

---

## 6. 故障后重路由 x̃_{j,r}(ω) — 又是 Least-Loaded

**实现位置**: `vllm/v1/core/replica_manager.py:152-182`

```python
def route_request_for_failover(self, request, exclude_replica_ids):
    candidates = [r for r in replicas
                  if r.status == HEALTHY and r.replica_id not in exclude]
    best = min(candidates, key=lambda r: r.num_active_requests)
    return best.replica_id
```

加上 Recovery Manager 中的一个小优化：按 G_j 降序恢复，确保高价值请求优先得到处理：

**实现位置**: `vllm/v1/core/recovery_manager.py:159-160`

```python
affected_requests.sort(key=lambda r: r.generation_len, reverse=True)
```

---

## 总结：论文 vs 代码实现

| 论文中的问题 | 代码实现 |
|------------|--------|
| 联合优化 `max ∑ G_j·y_j` | **分解成独立子决策**，不联合 |
| y_j: 0-1 背包 | **贪心**：按 G_j 降序逐个尝试 |
| x_{j,r}: 多维分配 | **Least-loaded**：选最空闲的 |
| Ω_k: 枚举 C(n,k) 种场景 | **最坏情况分析**：用 `n-k` 台最差 replica 的容量 |
| ℓ_j ∈ {0,1,2}: 优化选择 | **确定性阈值规则**：根据 progress 直接映射 |
| x̃_{j,r}(ω): 重路由分配 | **Least-loaded** + 按 G_j 排序 |
| Failover-gap 约束 | **直接计算公式**，用于准入拒绝和故障恢复丢弃 |

---

## 设计哲学

本质上，代码把论文中的整数规划问题转化成了一个**在线决策管道**：

```
新请求到达
    ↓
检查 Ω_k 鲁棒可行性？
    ├→ No → 拒绝
    └→ Yes ↓
检查 SLO 约束？
    ├→ No → 拒绝
    └→ Yes ↓
估算 recovery cost？
    ├→ > threshold → 拒绝
    └→ OK ↓
准入 (Admit)
    ↓
选择 checkpoint level
    ↓
Least-loaded 初始路由
```

### 优势
- **O(1) 延迟**：不需要求解 LP/MIP，所有决策都是确定式/启发式
- **简单实现**：容易理解和调试
- **适合在线场景**：请求实时到达

### 缺点
- **不最优**：贪心背包不保证最优准入集合（可能拒绝一个大 G_j 的请求，但后来无法用剩余容量）
- **启发式路由**：Least-loaded 对某些流量模式不够好
- **保守约束**：Ω_k 最坏情况分析对异构 replica 可能过于保守
