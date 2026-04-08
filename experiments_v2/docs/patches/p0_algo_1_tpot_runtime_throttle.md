# P0-algo-1：TPOT runtime throttle（**重定位版**）

> Patch 草稿，**未应用**。等 P0-impl-1 对照实验跑完再决定要不要 apply。
> **用户偏好**：只加不删。

## ⚠️ 重要说明：原 P0-algo-1 的设计假设已被部分推翻

原 [TODO List](../e1a_quick_diagnosis.md#todo-list) 里写的 P0-algo-1 是"加 throttle 弥补 TPOT 退化的 over-admission"。但准备 patch 时新读到 [master.py:100-106](../../../vllm/v1/core/sched/benders/master.py#L100-L106)，发现：

- master 的 `cost_table` 包含 active + pending 两类请求
- active 请求的 `x` 变量被强制 pin 为 1
- 因此 master 的 decode capacity 约束 `Σ x ≤ Cap_dec` **实际包含 active 请求**，已经在限制 GPU 上的总并发数

详见 [诊断更新章节](../e1a_quick_diagnosis.md#诊断更新2026-04-08新发现master-实际上有-decode-capacity-约束过度-admit论断需要修正)。

**所以 P0-algo-1 的真实定位需要从"防止过度 admit"重写为以下两件事之一**：

| 重新定位 | 描述 |
|---|---|
| **(a) profile-free 兜底** | 让 admission cap 不依赖 [decode_capacity_profile_8b.json](../../decode_capacity_profile_8b.json) 的硬编码值，用实测 latency 反馈做动态收紧 |
| **(b) 抗 hardware drift** | 换硬件（A6000 → A5000）时不需要重 profile，throttle 自动收敛 |

**这两件事的优先级低于 P0-impl-1**（对照实验）。先确认 tpot 60ms 的真凶，再决定要不要做这个 patch。

---

## 触发条件（**先 check 这个再决定动手**）

P0-algo-1 patch **只在以下情况下值得做**：

| P0-impl-1 对照实验结果 | 是否做 P0-algo-1 |
|---|---|
| 关闭 checkpointing 后 `tpot_p50` 回到 ~30 ms | **不做** —— 真凶是 checkpoint controller (P0-impl-3)，throttle 帮不上 |
| 关闭后 `tpot_p50` 仍 ~60 ms 且 `decode_cap` 已经是 hardware-correct（P1-env-1 完成） | **不做** —— 真凶是别的，throttle 也没用 |
| 关闭后 `tpot_p50` 仍 ~60 ms 且 P1-env-1 未做 | **做** —— throttle 是 P1-env-1 的廉价替代，能在不重 profile 的情况下让 cap 自适应 |
| W4_Mixed `completion < 0.9` 在 P0-verify-1 fix 后**仍然存在** | **做** —— 说明静态 cap 即使按正确 bucket 取值也不准，需要动态反馈 |

---

## 设计概要（重定位版）

### 数据流

```
forward step finishes
        │
        ├─ vLLM scheduler reports per-step latency
        │
        ▼
┌──────────────────────────┐
│ TpotMonitor (sliding win)│ ← N=10 step EMA
│   .observe(latency_sec,  │
│            batch_size)   │
└──────────┬───────────────┘
           │
           ▼
   .estimated_tpot_at(N)   ← projected per-token latency at N concurrent
           │
           ▼
┌──────────────────────────┐
│ BendersFTSchedulerImpl   │
│   _process_pending       │
│   _admissions            │
│                          │
│   for req in admitted:   │
│     N = current_running  │
│     est = tpot.at(N + 1) │
│     if est > slo[req]:   │
│       reject(req)        │
│     else:                │
│       admit(req)         │
└──────────────────────────┘
```

### 关键设计选择

1. **TpotMonitor 数据来源**：vLLM scheduler 的 `update_from_output` 路径已经有每个 step 的 timing。在 [benders_ft_scheduler_impl.py](../../../vllm/v1/core/sched/benders_ft_scheduler_impl.py) 的 `update_from_output` 包装函数里采集 (latency, batch_size) tuple。
2. **Sliding window 大小**：N=10 step 起步。step 间隔 ~30 ms（A5000 8B），10 step ≈ 300 ms 的滑动平均。
3. **Throttle 时机**：在 [_process_pending_admissions:289-307](../../../vllm/v1/core/sched/benders_ft_scheduler_impl.py#L289-L307) 里，**solver 决定 admit 后、实际 admit 前**加 runtime check。这意味着 throttle 是 **post-solver veto**，不动 solver 本身。
4. **不动的部分**：
   - master.py / cost_tables.py / recovery_checker.py 完全不动
   - solve_loop.py 不动
   - Cap_dec 仍然来自 profile（throttle 是更紧的额外约束，不替代）

### 为什么不直接改 master 的 decode_capacity 约束

考虑过另一个方案：把 `decode_capacity` 从 profile 静态值改成 runtime 估计值（动态 cap）。被否决的原因：
- master 是 MIP solver，每次 solve 都 rebuild model，cap 在 solve 之间稳定才合理
- runtime 值跳变会让 solver 决策不稳定，可能引发 admission oscillation
- post-solver veto 更稳：solver 在 stable cap 上做 batch 决策，throttle 在动态层做 fine-grained 拒绝

---

## Patch 草稿

### Step 1：新建 `TpotMonitor`

**新文件**：`vllm/v1/core/sched/benders/tpot_monitor.py`

```python
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window TPOT monitor for runtime admission throttling.

P0-algo-1 (重定位版): provides a profile-free, hardware-drift-resilient
estimate of "if I add one more concurrent decode request, what will
per-token latency become". Used by BendersFTSchedulerImpl to veto
solver admission decisions when projected tpot would exceed SLO.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass


@dataclass
class _Sample:
    latency_sec: float
    batch_size: int


class TpotMonitor:
    """Per-replica sliding-window decode latency monitor."""

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        self._samples: collections.deque[_Sample] = collections.deque(
            maxlen=window_size
        )

    def observe(self, latency_sec: float, batch_size: int) -> None:
        """Record one decode step's measured per-token latency + batch size."""
        if latency_sec <= 0 or batch_size <= 0:
            return
        self._samples.append(_Sample(latency_sec, batch_size))

    def estimated_tpot_at(self, target_batch_size: int) -> float | None:
        """Estimate per-token latency at a hypothetical batch size.

        Uses a simple linear regression in (batch_size, latency) over the
        sliding window. Falls back to None if not enough samples.

        Returns:
            Estimated per-token latency in seconds, or None if no estimate.
        """
        if len(self._samples) < 3:
            return None
        # Simple linear fit: latency ≈ a + b * batch_size
        n = len(self._samples)
        sx = sum(s.batch_size for s in self._samples)
        sy = sum(s.latency_sec for s in self._samples)
        sxx = sum(s.batch_size * s.batch_size for s in self._samples)
        sxy = sum(s.batch_size * s.latency_sec for s in self._samples)
        denom = n * sxx - sx * sx
        if denom == 0:
            # All samples at same batch size; return their mean
            return sy / n
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        return max(0.0, a + b * target_batch_size)
```

### Step 2：在 `BendersFTSchedulerImpl` 集成

**修改**：[vllm/v1/core/sched/benders_ft_scheduler_impl.py](../../../vllm/v1/core/sched/benders_ft_scheduler_impl.py)

```diff
@@ benders_ft_scheduler_impl.py: imports @@
 from vllm.v1.core.sched.benders.solve_loop import BendersSolveLoop
+from vllm.v1.core.sched.benders.tpot_monitor import TpotMonitor

@@ class BendersFTSchedulerImpl: __init__ ~line 89-200 @@
         # ... existing init code unchanged ...
+
+        # P0-algo-1: per-replica TPOT monitor for runtime admission throttle
+        self._tpot_monitors: dict[int, TpotMonitor] = {
+            r: TpotMonitor(window_size=10) for r in self._all_replica_ids
+        }
+        self._tpot_throttle_enabled: bool = bool(
+            getattr(sched_cfg, "enable_tpot_throttle", True)
+        )

@@ def _process_pending_admissions ~line 289-307 @@
         if result is not None:
             # Commit solver decisions for pending requests.
             for req in pending:
                 if req.request_id in result.master_solution.admitted:
                     assignment = result.master_solution.assignments.get(
                         req.request_id
                     )
                     if assignment is not None:
                         r_id = assignment
+
+                        # P0-algo-1: post-solver TPOT throttle veto
+                        if self._tpot_throttle_enabled:
+                            current_running = (
+                                self._ft.request_pool
+                                .num_admitted_on_replica(r_id)
+                            )
+                            est_tpot_sec = (
+                                self._tpot_monitors[r_id]
+                                .estimated_tpot_at(current_running + 1)
+                            )
+                            req_tpot_slo_sec = (
+                                req.tpot_slo_ms / 1000.0
+                                if req.tpot_slo_ms is not None
+                                else None
+                            )
+                            if (est_tpot_sec is not None
+                                    and req_tpot_slo_sec is not None
+                                    and est_tpot_sec > req_tpot_slo_sec):
+                                logger.info(
+                                    "P0-algo-1: throttle vetoed admit of "
+                                    "%s on r%d: est_tpot=%.1fms > slo=%.1fms "
+                                    "(running=%d)",
+                                    req.request_id, r_id,
+                                    est_tpot_sec * 1000,
+                                    req_tpot_slo_sec * 1000,
+                                    current_running,
+                                )
+                                self._base.finish_requests(
+                                    req.request_id,
+                                    RequestStatus.FINISHED_ABORTED,
+                                )
+                                continue
+
                         self._ft.request_pool.add_request(req)
                         self._ft.request_pool.admit_request(
                             req.request_id, r_id
                         )
                         self._ft.replica_manager.assign_request(req, r_id)
```

### Step 3：在 `update_from_output` 里采集 latency 数据

**修改**：[vllm/v1/core/sched/benders_ft_scheduler_impl.py:368](../../../vllm/v1/core/sched/benders_ft_scheduler_impl.py#L368)

```diff
@@ def update_from_output ~line 368 @@
     def update_from_output(
         self,
         scheduler_output: "SchedulerOutput",
         model_runner_output: "ModelRunnerOutput",
     ) -> ...:
+        # P0-algo-1: capture per-step decode latency for tpot monitor
+        if self._tpot_throttle_enabled:
+            try:
+                step_latency_sec = getattr(
+                    model_runner_output, "step_latency_sec", None
+                )
+                if step_latency_sec is not None:
+                    batch_size = len(self._base.running)
+                    # The local replica id; in DP each engine sees only its own
+                    self._tpot_monitors[self._replica_id].observe(
+                        step_latency_sec, batch_size
+                    )
+            except Exception as exc:
+                logger.debug("tpot monitor observe failed: %s", exc)
+
         return self._base.update_from_output(
             scheduler_output, model_runner_output
         )
```

**注意**：`model_runner_output.step_latency_sec` 这个字段**不存在**——这是设计假设。我需要在 vLLM model runner output 里添加这个字段。**这会扩散改动**到 model runner 路径。

**fallback**：如果不想改 model runner output，可以在 `update_from_output` 内部用 wall clock 计时（`time.monotonic()` 在调用前后），代价是不那么精确（包含调度 overhead）：

```diff
+        if self._tpot_throttle_enabled:
+            t0 = time.monotonic()
+            ret = self._base.update_from_output(...)
+            t1 = time.monotonic()
+            ...
```

但 wall clock 时机偏差对 throttle 来说可接受，因为 throttle 是统计阈值不是精确时刻。

### Step 4：在 `request_pool` 加 `num_admitted_on_replica()`（如果不存在）

**修改**：[vllm/v1/core/request_pool.py](../../../vllm/v1/core/request_pool.py)（待 verify 是否已经有）

```diff
+    def num_admitted_on_replica(self, replica_id: int) -> int:
+        """P0-algo-1: count active requests assigned to a replica."""
+        return sum(
+            1 for r in self._admitted.values()
+            if r.assigned_replica_id == replica_id
+        )
```

---

## 验收方法

1. apply patch + 跑 [config_8b_diag.yaml](../../config_8b_diag.yaml) 的 `Our-System` cell（带 throttle）
2. 检查 server log：
   ```
   P0-algo-1: throttle vetoed admit of req-XYZ on r0: est_tpot=120.0ms > slo=100.0ms (running=12)
   ```
3. 比较 metrics.json：
   - `admission_rate`：可能略 < 1.0（被 throttle 拒了一些）
   - `tpot_p50`：应该 ≤ No-FT 的 1.2 × （28 → 35 ms 以内）
   - `goodput`：应该 ≥ Periodic-Low

## 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| 线性回归在前 N 个 step 不稳定，throttle 误拒 | 中 | window_size=10，前 3 个 sample 不做估计（已在代码里） |
| 跨 workload 的 latency 模型差异大（W3 vs W1）混在一个 monitor 里 | 中 | 后续可以拆 per-workload monitor，但起步先合并 |
| 跟 P0-verify-1 fix #1 (avg_ctx_bucket) 互相影响 | 低 | 两个 fix 独立，按顺序 apply 即可 |
| `model_runner_output.step_latency_sec` 字段需要新加 → 改动扩散 | **高** | 用 wall clock fallback，避免改 model runner |

## 代码改动规模

| 文件 | 行数变化 | 性质 |
|---|---|---|
| `vllm/v1/core/sched/benders/tpot_monitor.py` | +60 (新文件) | additive |
| `vllm/v1/core/sched/benders_ft_scheduler_impl.py` | +50 | additive |
| `vllm/v1/core/request_pool.py` | +5 (如果方法不存在) | additive |
| 总计 | ~115 行新代码，**0 行删除** | 100% additive |

---

## 这个 patch 能改善什么

| 实测症状 | 是否能由此 patch 改善 |
|---|---|
| `Our-System / W1_Chat / Moderate / none` tpot_p50 = 60ms | **取决于 P0-impl-1 结果** —— 如果 60ms 来自 checkpoint controller 而不是 batch contention，throttle 帮不上 |
| W4_Mixed `completion < 0.9` | **可能能** —— 如果 admit 后立刻被 batch contention 拖死，throttle 提前拒掉能让 completion 提高 |
| 18 个 cell goodput 输给 Periodic | **可能能** —— 如果 Periodic 隐式有 max_num_seqs 限流，throttle 给 Our-System 一个对等的限流机制 |
| Hardware drift（A6000 profile → A5000 实跑）造成的 cap 不准 | **能** —— 这是 throttle 的核心价值，profile-free |

---

## 最终建议

**先跑 P0-impl-1 对照实验**，根据结果再决定：
- 如果对照实验说凶手是 checkpoint controller → 优先做 P0-impl-3（instrumentation + 修 stream 阻塞），不做这个 patch
- 如果对照实验说凶手是 batch contention 或 hardware mismatch → 才做这个 patch
