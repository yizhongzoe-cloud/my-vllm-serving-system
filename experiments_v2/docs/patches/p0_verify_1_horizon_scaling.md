# P0-verify-1 Fix #2：profile 与 runtime 的 planning_horizon 不一致

> Patch 草稿，**未应用**。等 vLLM build 完成 + 用户确认后再 apply。
> **用户偏好**：只加不删。

## Bug 来源

P0-verify-1 [发现 B](../e1a_quick_diagnosis.md#p0-verify-1-验证结果2026-04-08)：

| 来源 | 字段 | 值 |
|---|---|---|
| [config_8b.yaml:20](../../config_8b.yaml#L20) | `ft_planning_horizon` | **1.0 s** |
| [decode_capacity_profile_8b.json](../../decode_capacity_profile_8b.json) | `meta.planning_horizon_sec` | **0.5 s** |

[profile_decode_capacity.py:266-279](../../profile_decode_capacity.py#L266-L279) 把 prefill capacity 写入 JSON 时，单位是 "tokens / `--horizon` 秒"（默认 0.5）：

```python
prefill_tput = prefill_tokens / elapsed   # tokens/s
rem_cap = int(tput * horizon_sec)          # tokens / horizon_sec
```

但 [decode_capacity_model.py:60-65](../../../vllm/v1/core/sched/benders/decode_capacity_model.py#L60-L65) 加载时**不读** `meta.planning_horizon_sec`，直接把字典原样存到 `self._prefill_caps`：

```python
rem_pre = data.get("residual_prefill_capacity", {})
pairs = sorted((int(k), int(v)) for k, v in rem_pre.items())
if pairs:
    self._prefill_loads = [p[0] for p in pairs]
    self._prefill_caps = [p[1] for p in pairs]
```

[master.py:152-160](../../../vllm/v1/core/sched/benders/master.py#L152-L160) 用这个数字时也不缩放：

```python
rem_cap = self._residual_prefill_capacity.get(r, 0)
...
model.add(sum(prefill_terms) <= rem_cap)
```

**结果**：master 的 prefill capacity 约束的隐式 horizon 是 0.5 s（profile 的），但开发者**意图**的 horizon 是 `ft_planning_horizon = 1.0` s。两者错配 2×。

**方向**：profile horizon (0.5) < config horizon (1.0)，所以 master 看到的 cap 数值偏小，实际是把 prefill 容量**低估** 2×。

## 修复策略

两条互补的做法，**建议都做**：

### A：runtime 加 horizon scaling（让 profile 数据自适应 runtime horizon）

在 `DecodeCapacityModel.__init__` 加载 profile 时读 `meta.planning_horizon_sec`，并暴露一个 `set_runtime_horizon(h)` 方法做缩放。

### B：startup 加 assertion（让两边强制一致，未来有人改一边时立刻报错）

在 `CostBuilder` 或 server startup 处加一个 sanity check：要求 `config.ft_planning_horizon == profile.meta.planning_horizon_sec`，否则 fail-fast 并打印明确错误。

## Patch 草稿 — Part A（horizon scaling）

**文件**：[vllm/v1/core/sched/benders/decode_capacity_model.py](../../../vllm/v1/core/sched/benders/decode_capacity_model.py)

```diff
@@ decode_capacity_model.py: __init__ ~line 32-71 @@
     def __init__(self, profile_path: str | None = None):
         self._fallback = False
         self._default_cap: int = 100
         self._ctx_buckets: dict[int, int] = {}
         self._prefill_loads: list[int] = []
         self._prefill_caps: list[int] = []
+        # P0-verify-1 fix: track profile-time horizon, allow runtime rescaling
+        self._profile_horizon_sec: float = 0.0
+        self._runtime_horizon_sec: float = 0.0
         self._use_fitted = False
         self._fit_a: float = 0.0
         self._fit_b: float = 0.0

         if profile_path is None or not os.path.exists(profile_path):
             logger.warning(...)
             self._fallback = True
             return

         with open(profile_path) as f:
             data = json.load(f)

+        # P0-verify-1 fix: read profile horizon for later rescaling
+        meta = data.get("meta", {})
+        self._profile_horizon_sec = float(
+            meta.get("planning_horizon_sec", 0.0)
+        )

         # Decode capacity
         dec_cap = data.get("decode_capacity", {})
         self._default_cap = dec_cap.get("default", 100)
         by_bucket = dec_cap.get("by_avg_ctx_bucket", {})
         self._ctx_buckets = {int(k): int(v) for k, v in by_bucket.items()}

         # Residual prefill capacity
         rem_pre = data.get("residual_prefill_capacity", {})
         pairs = sorted((int(k), int(v)) for k, v in rem_pre.items())
         if pairs:
             self._prefill_loads = [p[0] for p in pairs]
             self._prefill_caps = [p[1] for p in pairs]
```

**新增方法**（在 class 末尾追加，不删任何已有方法）：

```diff
+    def set_runtime_horizon(self, runtime_horizon_sec: float) -> None:
+        """Configure the runtime planning horizon (typically ft_planning_horizon).
+
+        residual_prefill_capacity values in the profile JSON are measured
+        for `meta.planning_horizon_sec` (typically 0.5 s). The master MIP
+        uses these values directly as a token budget over `runtime_horizon_sec`.
+        If the two horizons differ, we must rescale.
+
+        Call this once after construction, before any
+        `residual_prefill_capacity()` query.
+        """
+        self._runtime_horizon_sec = runtime_horizon_sec
+        if self._fallback:
+            return
+        if self._profile_horizon_sec <= 0:
+            logger.warning(
+                "Profile JSON missing meta.planning_horizon_sec; "
+                "cannot rescale residual_prefill_capacity. "
+                "Skipping horizon adjustment."
+            )
+            return
+        if abs(runtime_horizon_sec - self._profile_horizon_sec) < 1e-6:
+            return  # already consistent
+        scale = runtime_horizon_sec / self._profile_horizon_sec
+        old_caps = list(self._prefill_caps)
+        self._prefill_caps = [int(c * scale) for c in self._prefill_caps]
+        logger.info(
+            "P0-verify-1 fix: rescaled residual_prefill_capacity by %.3f "
+            "(profile_horizon=%.3fs → runtime_horizon=%.3fs). "
+            "Old caps=%s, new caps=%s",
+            scale, self._profile_horizon_sec, runtime_horizon_sec,
+            old_caps, self._prefill_caps,
+        )
```

**调用点**：在 `CostBuilder.__init__`（[cost_tables.py:122](../../../vllm/v1/core/sched/benders/cost_tables.py#L122)）创建 `DecodeCapacityModel` 后立即调用：

```diff
@@ cost_tables.py @@
         self._decode_cap_model = DecodeCapacityModel(decode_capacity_profile_path)
+        # P0-verify-1 fix: rescale profile-horizon to match runtime horizon
+        self._decode_cap_model.set_runtime_horizon(planning_horizon)
```

## Patch 草稿 — Part B（startup assertion）

**文件**：[experiments_v2/run.py](../../run.py)

在 server 启动前加一个 sanity check（**新增函数 + 在 main 流程里调一次**，不动现有逻辑）：

```diff
+def _verify_horizon_consistency(config: dict) -> None:
+    """P0-verify-1 fix: ensure profile horizon matches runtime horizon.
+
+    The decode capacity profile JSON measures residual_prefill_capacity
+    using `meta.planning_horizon_sec`. The runtime master MIP uses these
+    numbers as token budgets over `ft_planning_horizon`. If the two
+    differ, fail loudly so the operator can decide whether to rescale
+    or re-profile.
+    """
+    profile_path = config.get("ft_decode_capacity_profile")
+    if not profile_path or not os.path.exists(profile_path):
+        return
+    runtime_h = float(config.get("ft_planning_horizon", 0.0))
+    if runtime_h <= 0:
+        return
+    with open(profile_path) as f:
+        profile = json.load(f)
+    profile_h = float(
+        profile.get("meta", {}).get("planning_horizon_sec", 0.0)
+    )
+    if profile_h <= 0:
+        logger.warning(
+            "P0-verify-1: profile %s has no meta.planning_horizon_sec; "
+            "cannot verify consistency. Re-profile recommended.",
+            profile_path,
+        )
+        return
+    if abs(runtime_h - profile_h) > 1e-6:
+        logger.warning(
+            "P0-verify-1: HORIZON MISMATCH detected. "
+            "ft_planning_horizon=%.3fs in config, but profile %s "
+            "was measured at %.3fs. "
+            "Part-A horizon scaling in DecodeCapacityModel will rescale, "
+            "but consider re-profiling at the runtime horizon for accuracy.",
+            runtime_h, profile_path, profile_h,
+        )
```

调一次的位置：在 `run.py` 的 `main()` 加载 config 后立即调（不阻塞，只 warn）。

## 验收方法

1. apply Part A 后，server 启动 log 应当看到：
   ```
   P0-verify-1 fix: rescaled residual_prefill_capacity by 2.000 (profile_horizon=0.500s → runtime_horizon=1.000s). Old caps=[17234, 14321, ...], new caps=[34468, 28642, ...]
   ```
2. apply Part B 后（如果有人**故意**让 horizon 不一致），server 启动 log 应该看到 warning，但不会 abort。
3. 跑一次 prefill-heavy workload (W2_Summary / Heavy)，对比修前修后的 `admission_rate`：
   - 修前：可能 < 1.0（被错配的低估 cap 限制）
   - 修后：应该 ≈ 1.0

## 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| profile 在不同 horizon 上的 capacity 不严格线性（startup overhead 等） | 中 | 这是合理简化。要更精确就只能在 runtime horizon 重新 profile（对应 P1-env-1）|
| 旧的 profile JSON 没有 `meta.planning_horizon_sec` 字段 | 低 | Part A 已 handle（落到 warning + skip）|
| 已有的 `residual_prefill_capacity` 在某些 cell 是 binding 约束，rescale 后导致 admission 行为变化 | 中 | 这是 fix 的**预期效果**，不是 bug |

## 这个 fix 能解释/改善什么

| 实测症状 | 是否能由此 fix 解释/改善 |
|---|---|
| `Our-System / W1_Chat / Moderate / none` tpot_p50 = 60ms | **不能** —— W1_Chat decode-heavy，prefill capacity 不是 binding |
| 5 个 W4_Mixed `completion < 0.9` cell | **不能** —— admission_rate 已 ~1.0，prefill cap 没在卡 |
| W2_Summary / Heavy 上潜在的 underutilization | **能**（潜在），但当前 8B 实测 cell 不包含这个工况 |

**结论**：真 bug 应当修，但**对当前实测的 47 个 cell 几乎不产生改善效果**。修它的主要价值是**防止后续切换工况时撞到这个坑**。
