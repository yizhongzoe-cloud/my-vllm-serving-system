# Patches 草稿目录

这里放的是 [e1a_quick_diagnosis.md](../e1a_quick_diagnosis.md) 中识别出的 bug 的修复 **patch 草稿**。

> ⚠️ **所有 patch 都是草稿，未应用到代码**。等 vLLM build 完成 + 用户审阅后再决定 apply 顺序。
>
> **用户偏好**：只加不删（additive only），必要时保留 backup。所有 patch 都遵守这个原则。

## Patch 列表

| 文件 | 对应 bug | 优先级 | apply 触发条件 | 预计代码改动 |
|---|---|---|---|---|
| [p0_verify_1_avg_ctx_bucket_fix.md](./p0_verify_1_avg_ctx_bucket_fix.md) | `solve_loop.py:161` 漏传 `avg_ctx_bucket`，强制使用最保守 default cap=10 | **P0** | 立即可 apply（独立、低风险） | ~10 行 additive |
| [p0_verify_1_horizon_scaling.md](./p0_verify_1_horizon_scaling.md) | profile 0.5s vs config 1.0s 错配，prefill capacity 被低估 2× | P0 | 立即可 apply（独立、低风险） | ~50 行 additive（含新方法 + assertion）|
| [p0_algo_1_tpot_runtime_throttle.md](./p0_algo_1_tpot_runtime_throttle.md) | TPOT 约束代数退化（**重定位版** —— 详见文档开头说明） | **P0-impl-1 之后** | 仅当 P0-impl-1 对照实验指向 batch contention 时才做 | ~115 行 additive，含新文件 |

## 推荐 apply 顺序

```
等 vLLM build 完成
        │
        ▼
1. 跑 P0-impl-1 对照实验
   (使用 config_8b_diag.yaml + Our-System vs Our-System-NoCkpt)
        │
        ├──→ tpot 回到 ~30ms ────→ apply P0-impl-3 (修 checkpoint_controller)
        │                            (不 apply p0_algo_1，throttle 帮不上)
        │
        └──→ tpot 仍 ~60ms
                │
                ▼
        2. apply p0_verify_1_avg_ctx_bucket_fix.md (廉价、独立)
        3. apply p0_verify_1_horizon_scaling.md (廉价、独立)
                │
                ▼
        4. 重跑 P0-impl-1 对照实验
                │
                ├──→ W4_Mixed completion 回到 ≥0.95 ────→ stop, run E1a_Quick
                │
                └──→ 仍未恢复
                        │
                        ▼
                5. apply p0_algo_1_tpot_runtime_throttle.md
                6. 重跑 P0-impl-1
```

## 共同前置条件

所有 patch apply 前必须：
1. ✅ vLLM venv build 完成
2. ✅ `git status` clean（patch 应用前 commit 当前所有修改）
3. ✅ 单独的 git branch（建议：`zoe/p0-fixes`，与 `zoe/slo-scheduling` 隔离）
4. ✅ patch 应用顺序记录到 commit message
5. ✅ 每个 patch 应用后单独跑一次 sanity check（不要批量 apply）

## 与原 TODO List 的关系

| 原 TODO | Patch 文件 | 状态 |
|---|---|---|
| P0-impl-1 (config + 对照实验) | [config_8b_diag.yaml](../../config_8b_diag.yaml) | ✅ 草稿就绪 |
| P0-impl-2 (判定) | （手工动作） | 等 P0-impl-1 跑完 |
| P0-impl-3 (instrumentation) | （未起草） | 仅 P0-impl-2 指向 controller 时做 |
| P0-algo-1 (TPOT throttle) | [p0_algo_1_tpot_runtime_throttle.md](./p0_algo_1_tpot_runtime_throttle.md) | ✅ 草稿就绪（重定位版） |
| P0-verify-1 (horizon mismatch) | [p0_verify_1_horizon_scaling.md](./p0_verify_1_horizon_scaling.md) | ✅ 草稿就绪 |
| P0-verify-1 (avg_ctx_bucket) | [p0_verify_1_avg_ctx_bucket_fix.md](./p0_verify_1_avg_ctx_bucket_fix.md) | ✅ 草稿就绪 |
| P0-verify-2 (failover gap dump 日志) | （未起草，待后续） | — |
| P0-verify-3 (Periodic batch size 对比日志) | （未起草，待后续） | — |
