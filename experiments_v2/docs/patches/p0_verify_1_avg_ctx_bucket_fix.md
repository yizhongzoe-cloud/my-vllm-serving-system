# P0-verify-1 Fix #1：solve_loop.py 漏传 `avg_ctx_bucket`

> Patch 草稿，**未应用**。等 vLLM build 完成 + 用户确认后再 apply。
> **用户偏好**：只加不删，必要时保留 backup。

## Bug 来源

P0-verify-1 [bonus 发现](../e1a_quick_diagnosis.md#p0-verify-1-验证结果2026-04-08)：

[solve_loop.py:161](../../../vllm/v1/core/sched/benders/solve_loop.py#L161) 调用 `get_decode_capacity()` 时**没有传 `avg_ctx_bucket` 参数**：

```python
decode_cap = self._cost_builder.get_decode_capacity()
#                                    ^^^ 没有 avg_ctx_bucket
```

[decode_capacity_model.py:77-98](../../../vllm/v1/core/sched/benders/decode_capacity_model.py#L77-L98) 默认 `avg_ctx_bucket=0`，`<= 0` 时直接返回 `default = min(buckets)`。

在 8B 配置下：
- profile JSON `decode_capacity = { "default": 10, "by_avg_ctx_bucket": {"256": 50, "1024": 10} }`
- `default = min(50, 10) = 10`
- → solver 永远看到 `Cap_dec = 10`，与请求实际 prompt 长度无关

**影响**：
- W1_Chat (avg prompt 1259) → 应该走 1024 bucket → cap 10 ✅ 正确
- W3_Instruct (avg prompt 17) → 应该走 256 bucket → cap 应为 **50**，实际仍用 **10**
- W4_Mixed → 取决于 mix 比例的动态平均，但实际仍用 10

W3/W4 上 admission 上限被人为压低 5×。

## 修复策略

最小改动：在 `get_decode_capacity()` 调用前算一个 batch 平均 ctx，传进去。

`RequestCosts` 已经有 `prompt_len` 字段（[cost_tables.py:59](../../../vllm/v1/core/sched/benders/cost_tables.py#L59)），可以直接用 `cost_table.values()` 算。

**注意**：
- 对 active 请求，理想情况应该用"prompt_len + tokens_decoded_so_far"。但 cost_table 没暴露 tokens_decoded，先用 prompt_len 起步（保守，会让 cap 略偏小但不会偏大）
- 如果 cost_table 只有 active 请求（没有 pending），不进 admission 路径，可以跳过

## Patch 草稿

**文件**：[vllm/v1/core/sched/benders/solve_loop.py](../../../vllm/v1/core/sched/benders/solve_loop.py)

**修改位置**：line 160-161 附近，在 `decode_cap = self._cost_builder.get_decode_capacity()` 之前插入。

**Diff（unified format）**：

```diff
@@ solve_loop.py: solve_epoch_from_costs ~line 155-165 @@
         # Decode-first capacity: compute per-replica decode load and
         # residual prefill capacity.
         use_df = self._cost_builder.use_decode_first_model
         decode_cap = 0
         residual_prefill: dict[int, int] = {}
         if use_df:
-            decode_cap = self._cost_builder.get_decode_capacity()
+            # P0-verify-1 fix: compute avg ctx bucket from cost_table prompt_len
+            # rather than always using the conservative `default` cap.
+            # Without this fix, short-prompt workloads (W3/W4) get
+            # cap=min(buckets) instead of the correct per-bucket value.
+            _prompt_lens = [
+                c.prompt_len for c in cost_table.values() if c.prompt_len > 0
+            ]
+            if _prompt_lens:
+                _avg_ctx = int(sum(_prompt_lens) / len(_prompt_lens))
+            else:
+                _avg_ctx = 0
+            decode_cap = self._cost_builder.get_decode_capacity(
+                avg_ctx_bucket=_avg_ctx
+            )
+            logger.debug(
+                "P0-verify-1 fix: avg_ctx=%d → decode_cap=%d "
+                "(was always default=%d before fix)",
+                _avg_ctx, decode_cap,
+                self._cost_builder.get_decode_capacity(),
+            )
             # Count active decode requests per replica
             L_current: dict[int, int] = {r: 0 for r in recovery_replica_ids}
             for req_id, costs in cost_table.items():
```

**改动说明**：
- 新增 8 行（包括 docstring 和 logger.debug），**没删任何行**
- 复用 `cost_table` 已有数据，没有新增 IO
- `logger.debug` 顺手在 fix 后输出 "before vs after"，方便确认 fix 是否生效

## 验收方法

1. apply patch 后单独跑一次 `Our-System / W3_Instruct / Moderate / none`（W3 是最受益的 cell，alpaca avg prompt 17）
2. 检查 server log，应该能看到：
   ```
   P0-verify-1 fix: avg_ctx=17 → decode_cap=50 (was always default=10 before fix)
   ```
3. 对比 metrics.json：
   - 修前：`admission_rate ≈ 0.X`（被 cap=10 限制）
   - 修后：`admission_rate ≈ 1.0`（cap 放宽到 50）
   - tpot_p50 不应显著变化（因为 W3 本来 prompt 短，并发能力本来就高）

## 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| W4_Mixed 上 mix 比例变化时，平均 ctx 计算抖动导致 cap 反复跳变 | 中 | 加 EMA 平滑或者每 N step 更新一次 |
| 把 active 请求的 prompt_len 也算进去会让平均偏向已经在跑的请求 | 低 | 等真出问题时改成"只算 pending" |
| `bucket_lookup()` 用 round-down，跨 bucket 边界时仍然偏保守 | 低 | acceptable trade-off |

## 这个 fix 能解释/改善什么

| 实测症状 | 是否能由此 fix 解释/改善 |
|---|---|
| `Our-System / W4_Mixed / *` 5 个 cell `completion < 0.9` | **部分能** —— W4_Mixed 包含 W3_Instruct (alpaca)，cap 过严会让短请求被拒 |
| `Our-System / W1_Chat / Moderate / none` tpot_p50 = 60ms | **不能** —— W1 的 prompt 平均 1259，本来就走 1024 bucket，cap 没变化 |
| 18 个 cell goodput 输给 Periodic | **部分** —— 仅 W4_Mixed 那些 cell 受影响 |

**结论**：这是个真 bug，应当修，但**不解决核心 tpot 问题**。
