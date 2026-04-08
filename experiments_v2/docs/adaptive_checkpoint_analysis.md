# Adaptive Checkpoint Economic Model Analysis

## Decision Formula

```
Δreplay > Δload + λ · Δckpt
```

If true → checkpoint. Otherwise → skip.

Only evaluated when a new full KV block (16 tokens) is produced.

---

## Term Definitions

### Δreplay（省下的重算时间）

```
Δreplay = U / replay_throughput
```

- `U` = unpublished_tokens（没存的 token 数）= stable_full_tokens − num_checkpointed_tokens
- `replay_throughput` ≈ `GPU_FLOPS / (2 × num_params + 2 × num_layers × context_len × hidden_size)`
  - 第一项 `2 × num_params`：线性层（QKV 投影、output 投影、FFN），跟 position 无关
  - 第二项 `2 × num_layers × context_len × hidden_size`：attention score，跟 position 有关（越靠后 context 越长）
  - 在 max_model_len=8192 内，attention 项只占 ~13%，近似可忽略：`replay_throughput ≈ F / 2P`

### Δload（恢复时多花的加载时间）

```
Δload = U × B / load_bandwidth
```

- `B` = kv_bytes_per_token = `num_layers × 2(K+V) × num_kv_heads × head_dim × bytes_per_element`
  - 8B model: 32 × 2 × 8 × 128 × 2 = 131,072 bytes (128 KB)
  - `2(K+V)`: 每个 token 每层存一个 Key vector 和一个 Value vector
  - `num_kv_heads = 8`（GQA，4 个 query head 共享 1 个 KV head）
- `load_bandwidth` = CPU→GPU 搬运速度（config 写的 10 GB/s）

### Δckpt（存盘的拷贝开销）

```
Δckpt = U × B / checkpoint_bandwidth
```

- `checkpoint_bandwidth` = GPU→CPU 拷贝速度（profile 实测的，约 10 GB/s）

---

## 展开不等式

```
U / replay_throughput > U × B / load_bw + λ × U × B / ckpt_bw
```

两边约掉 U：

```
1 / replay_throughput > B × (1/load_bw + λ/ckpt_bw)
```

**左右都是常数**（给定模型和硬件就定了）。不等式要么恒成立（每个 block 都存），要么恒不成立（永远不存）。

---

## 代入 8B + A6000

左边（重算 1 个 token 的时间）：

```
2P / F = 2 × 8×10⁹ / 38.7×10¹² = 4.13 × 10⁻⁴ s/token
```

右边（搬运 1 个 token KV cache 的时间）：

```
B × (1/BW_load + λ/BW_ckpt) = 131072 × (1/10¹⁰ + 1.0/10¹⁰) = 2.62 × 10⁻⁵ s/token
```

```
4.13 × 10⁻⁴  >>  2.62 × 10⁻⁵    （16 倍）
```

**恒成立** → 每产生一个新 block 就触发存盘 → adaptive 退化成 Periodic-High。

即使把 λ 调到 16，左边才刚好等于右边。说明在当前硬件下，重算（GPU 计算）比搬运（内存拷贝）慢一个数量级，经济模型永远认为"存比不存划算"。

---

## 退化成因

当前 Δckpt 只算了 **IO 时间**（数据搬运本身要多久），没算 **checkpoint 拷贝对 decode 的干扰**——拷贝会抢 GPU 显存带宽，导致同时在跑的 decode 请求 TPOT 变高。

当前模型：

```
Δckpt = U × B / BW_ckpt    （只有 IO 时间）
```

应该改成：

```
Δckpt = U × B / BW_ckpt + α × N_running × TPOT_degradation
```

- `N_running` = 当前在跑的 decode 请求数
- `TPOT_degradation` = checkpoint 拷贝导致的 per-request TPOT 增量
- `α` = 权重

这样高负载时 Δckpt 大 → 不容易触发 → 减少 checkpoint 频率。低负载时 Δckpt 小 → 该存就存。

---

## 代码对应

代码在 `vllm/v1/core/checkpoint_controller.py`。

三个值的计算（L95-109）：

```python
replay_saved_sec = unpublished_tokens / replay_throughput_tokens_per_sec
load_cost_sec = unpublished_bytes / load_bandwidth_bytes_per_sec
checkpoint_cost_sec = unpublished_bytes / checkpoint_bandwidth_bytes_per_sec
```

unpublished_tokens 的计算（L83-93）：

```python
stable_full_tokens = (num_computed_tokens // block_size) * block_size
published_tokens = num_checkpointed_tokens
unpublished_tokens = stable_full_tokens - published_tokens
unpublished_bytes = unpublished_tokens * kv_bytes_per_token
```

kv_bytes_per_token 的计算（L87-90）：

```python
if published_tokens > 0 and checkpoint_size_bytes > 0:
    kv_bytes_per_token = checkpoint_size_bytes / published_tokens  # 从上次 checkpoint 反推
else:
    kv_bytes_per_token = default_kv_bytes_per_token  # 默认 8192（不准，实际 131072）
```

判定（L111-118）：

```python
should_publish = (
    unpublished_tokens > 0
    and replay_saved_sec > (load_cost_sec + checkpoint_lambda * checkpoint_cost_sec)
)
```

当前参数：

| 参数 | 当前值 | 实际值 | 问题 |
|---|---|---|---|
| ft_prefill_throughput | 4000 tok/s | ~2400 tok/s | 改了也不影响结论（还是恒成立） |
| default_kv_bytes_per_token | 8192 | 131,072 | 16× 偏小，第一次 checkpoint 后自动修正 |
| checkpoint_lambda | 1.0 | — | 即使 λ=16 也刚好打平 |
| **decode 干扰** | **未建模** | **显著** | **退化的根因** |
