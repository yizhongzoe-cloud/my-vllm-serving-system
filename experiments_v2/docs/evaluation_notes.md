# Evaluation / Implementation Notes

写 paper 时需要回忆的实现细节、踩坑、ablation 路径、半成品。一边搬代码一边记。

---

## 2026-05-11 — /dev/shm publish + restore 框架移植

### 这次干了什么

从 `zoe/slo-scheduling` 把 cross-engine KV checkpoint 框架原样搬到 `zoe/disruption`：
- `vllm/v1/worker/gpu_model_runner.py` 新增约 1100 行（publish + restore + fast chunk + manifest 管理）
- 新增 `vllm/v1/worker/checkpoint_write_ext.py`（136 行，FT_NOGIL_WRITE 的 ctypes 写文件优化）
- 在 `GPUModelRunner.__init__` / `_update_states` 加 instance attr 和 finished cleanup
- 顶层 dataclass 加了 `SharedCheckpointManifest` / `SharedCheckpointState`
- wire 进 `checkpoint_kv_blocks`（save 完 inline publish）和 `restore_kv_blocks`（local pool fail → shm fallback）

**没搬**的：FT scheduler、FT solver、recovery_manager、ft_client（router / 失联检测）、batched restore 优化、FT_BG_PUBLISH 异步 publish executor、FT_BATCH_GATHER 单 GPU gather。要搬这些得专门动 scheduler 层和 router 层。

### 决策点：A 完整搬 vs B 砍 fast chunk

选 A。fast chunk format 是 ablation（FT_FAST_CHUNK_FORMAT=1 才用，默认 OFF），不搬就要改 `_publish_shared_checkpoint` 的 if/else 分支，那是"创作"不是"搬"。死代码躺着 480 行没成本，搬了对未来 ablation 有用。

### 写 paper 时要解释清楚的实现细节

#### 1. 增量 publish（generation 计数）
每次 `_publish_shared_checkpoint` 只写**新增**的 full block，不重写全量。manifest 记录累计的 `block_map`（每个逻辑 block 在哪个 chunk 文件的哪个 slot）。restore 时读 manifest，按 block_map 把不同 chunk 的对应 slot 拼回来。

为什么这样：每次都全量写 5-10MB 一个 chunk，PCIe + tmpfs write 顶不住。增量写每次只动 1 个 block（16 token × ~7MB / layer × 32 layers ≈ 几十 KB 一层），快很多。

#### 2. rank-aware 目录结构
路径长这样：`/dev/shm/vllm_ft_checkpoints/<req_id>/{latest_rank0, manifest_rank0_<gen>.json, chunk_rank0_<gen>.pt}`

每个 TP/PP rank 写自己的文件，互不干扰。TP rank 拼接公式：`pp_rank * tp_size + tp_rank`。restore 时也按自己的 rank tag 读对应文件。

#### 3. atomic write
写文件用 `tmp_path = "{final}.tmp.{pid}.{time_ns}"` + `os.replace`。能保证 reader 要么看到完整旧版本要么看到完整新版本，不会读到一半。

#### 4. sweep 防 race
`_sweep_shared_ckpt_dir` 启动时清孤儿 tmp 文件，但**只能删 dead PID 的**。曾经踩过 warmup-scan bug（2026-04-14）：sweep 误删 own-PID 的 in-flight tmp，导致 OS replace 失败 → 后续 OOM。**这块代码别动**。

#### 5. restore 路径的 OOB 防御
`_restore_shared_checkpoint` 里有两处 bounds check（slot_indices 和 tgt_block_list），防 `vectorized_gather_kernel` device-side assert。一旦 cuda assert，整个 engine 死。fallback 是 `return 0`，触发 reprefill 而不是 crash。曾经在 `W5_reload_w16/42` 和 `phase8` seed 上中过。

#### 6. CUDA lock for parallel restore
restore 在 batched 路径下从多线程被调用，CUDA stream context + scatter writes 不 thread-safe。`_ft_restore_cuda_lock` 序列化 CUDA enqueue 部分。CPU 那段（文件读 + load）仍然并发。

但本次没搬 `restore_kv_blocks_batch`，所以单 engine restore 走单线程，lock 不会用上。cross-engine 场景如果要用 batched restore 得再把那块搬过来。

### Env flag 清单（搬过来都保留，默认状态）

| Env flag | 默认 | 干嘛 |
|---|---|---|
| `FT_CKPT_SKIP_PUBLISH` | OFF | 跳过 /dev/shm 写，只算 size。Ablation：隔离 GPU gather+copy vs 文件 IO 开销 |
| `FT_FAST_CHUNK_FORMAT` | OFF | 用自定义二进制格式（5-10× 快于 torch.save） |
| `FT_INLINE_MANIFEST` | OFF | 配合 FAST_CHUNK，manifest 嵌入 chunk header，省 2/3 文件 op |
| `FT_FAST_TMPFS_WRITE` | OFF | 跳过 flush+fsync（tmpfs 上 fsync 没意义） |
| `FT_NOGIL_WRITE` | OFF | ctypes-based libc write，跳过 GIL（依赖 checkpoint_write_ext.py） |
| `FT_MERGE_LAYER_WRITE` | OFF | 32 layer 一次性 cat + tobytes + write（减 GIL acquire/release） |
| `FT_ASYNC_RESTORE` | OFF | 共用 restore stream，跳 per-call sync（要求 caller flush_pending_restore） |
| `FT_RESTORE_PER_REQ_SYNC` | ON | parallel batch restore 时每个 req sync 一次，防 temp src GPU 内存爆 |
| `FT_BG_PUBLISH` | OFF | 把 publish 文件写从 inline (阻塞 RPC) 改为 background ThreadPoolExecutor 异步执行，下次 RPC 进来 wait 上次 future 做背压。开了之后 RPC 不再被 publish 阻塞 (~50ms inline / ~0ms bg) |

**默认全部 OFF（除了 PER_REQ_SYNC）** → publish 走 torch.save 同步写，restore 走 sync stream，跟 proposal 里写的 baseline 行为一致。

**实验 / 性能跑分建议组合**：
- `FT_BG_PUBLISH=1` —— publish 不阻塞主线程（推荐 paper evaluation 默认开）
- `FT_FAST_CHUNK_FORMAT=1` + `FT_FAST_TMPFS_WRITE=1` + `FT_INLINE_MANIFEST=1` —— 三件套把 publish 单次成本从 ~50ms 降到 ~3-5ms（推荐 paper evaluation 默认开）

### Proposal mapping

Paper proposal `System Design` 那段：

> "incrementally copy each request's newly produced KV blocks from GPU to host pinned memory in the background at a block-aligned cadence (every 16 tokens, matching the PagedAttention block size). The host copy is then published to a shared memory file system (/dev/shm) via atomic write (temp file plus rename), so the checkpoint survives the original engine's death and can be read by any peer engine on the same machine."

对应代码：
- "GPU→host pinned memory": `kv_checkpoint_pool.save_checkpoint` (调用 `_copy_stream.synchronize()`)
- "block-aligned cadence (every 16 tokens)": `engine/core.py: _save_checkpoints_if_needed` 触发逻辑（**注：cadence 触发还在 engine/core.py，不在这次搬运范围**）
- "/dev/shm via atomic write": `_atomic_write_bytes` / `_atomic_torch_save` 的 `tmp + os.replace`
- "survives engine's death": 这是 tmpfs 的内核管理特性，不在我们代码里，但 proposal 这句话指的是 `/dev/shm` 本身的属性

> "the new engine restores KV blocks from the shared file system into freshly allocated GPU blocks"

对应代码：
- `_restore_shared_checkpoint`（读 latest pointer → 读 manifest → 按 block_map 收 chunk → scatter 到 target_block_ids 上）
- `restore_kv_blocks` RPC 的 shm fallback 路径

### 还没做的事（影响 paper 完整性）

1. **Router 层失联检测 + 请求 reroute**：proposal 的 "Cross-engine restore" 这一 bullet 现在**只有 worker 侧能力**，没有客户端调度。要等把简化版 reroute client 搬过来才完整。
2. **SLO 调度入口**：sampling_params.extra_args → EngineCoreRequest 的字段透传（字段已加，但没人填）
3. **Waiting queue 改 slack 排序**：现在还是 FCFS
4. **Hysteresis 阈值 δ**：picker 写了但没 gate
5. **Disruption injector + workload driver + metric collector**：实验工具链
6. **4 个 baseline 开关**：vanilla / reprefill / ckpt+FCFS / ckpt+SLO 切换

### 风险提示（写 paper 时如果 reviewer 问）

- **Fast chunk format 的 PyTorch 跨版本兼容性**：自定义二进制格式只在我们测过的 PyTorch 版本上保证 round-trip。如果 reviewer 问"为什么不直接用 torch.save"——答 ablation 用，默认 OFF
- **`/dev/shm` 的容量限制**：tmpfs 默认 50% 物理内存。Llama-3.1-8B 一个 64K context 请求大约 8GB KV，连续跑 ~6 个就把 /dev/shm 占满。**还没有 eviction policy**，跑长 workload 会需要加
- **多 PID 同 rank tag 冲突**：sweep 用 `os.kill(pid, 0)` 探活，前提是同一台机器。跨节点（多机 dp）的话 rank tag 会冲突，**这块设计假设是 single-machine multi-engine**

### Smoke test 结果

待跑：30-concurrent 16K stress test，确认 round 2 baseline 不变。

---

<!-- 后续按日期追加新条目 -->
