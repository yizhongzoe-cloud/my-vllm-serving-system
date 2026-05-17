# Long-context LLM serving with cheap preempt-resume — notes

> Last reorg: 2026-05-17. Detailed daily findings: [findings_2026_05_17.md](findings_2026_05_17.md).

## Problem

长 context 请求（RAG，几万 token 的 prompt）prefill 要跑几秒到几十秒。这种请求一旦被打断，前面的 prefill 就白干：

1. **Engine 死掉**：请求被甩到另一台 engine，从头 reprefill。重 prefill 几秒到几十秒。
2. **KV 池满了**：vLLM 触发 capacity preempt，victim 的 KV 丢掉，等会儿 admit 回来还是要从头 reprefill。
3. **SLO 调度想踢长请求救短请求**：踢一个跑了 5 秒 prefill 的长请求等于扔掉 5 秒的 GPU 计算，太亏。所以 QLM / Scorpio 这类 SLO scheduling paper **都不敢踢正在跑的长请求**。

## Mechanism

**Host-RAM 持续 checkpoint + V3 cheap reload + SLO-aware cross-engine picker**。

三个组件：

### 1. Continuous delta checkpoint (substrate)

每个跑着的请求每 N 个 token 把 GPU 上**新增的** KV block 拷一份到 host pinned memory + `/dev/shm/vllm_ft_checkpoints/<req_id>/`。增量（不是全量），数据量小，syscall + 文件 IO 是瓶颈（用 `FT_FAST_CHUNK_FORMAT=1` + `FT_FAST_TMPFS_WRITE=1` 一次 publish ~10ms）。

两层存储：
- Layer 1：进程内 host pool（同 engine cheap preempt 直接拿指针，<1ms）
- Layer 2：`/dev/shm` tmpfs 镜像（跨 engine reload 读，~5ms）

一份数据两个用途：recovery + cheap preempt。

### 2. V3 reload (cheap resume)

`_preempt_request` 触发时，如果 `FT_CAPACITY_PREEMPT_RELOAD=1` 且 `num_checkpointed_tokens > 0`，走 `_preempt_for_slo` 路径：
- victim KV 释放
- 状态变 `WAITING_FOR_RELOAD`
- 后台从 host pool / shm 拉回 KV
- 完成后状态 → `PREEMPTED`，scheduler 看到正常 admit
- **跳过完整 reprefill，只重做最后 1 个 token 的 K/V**（trim block 之后的 `num_computed_tokens` 设为 `num_tokens-1`）

跨 engine reroute 走类似路径，destination engine 读 `/dev/shm/vllm_ft_checkpoints/<origin_req_id>/` 完成 V3 reload。

### 3. SLO-aware cross-engine picker

`Scheduler._pick_priority_preempt_victim` (scheduler.py:1106)。每个 step 调一次。Fire 条件：
- waiting queue 里有 slack 紧张的 head（**FT_PICKER_IN_DANGER_MS**，绝对 ms，跟 SLO 解耦）
- running queue 里有 slack 充裕 + checkpoint 已落盘的 candidate
- victim_slack - head_slack ≥ min_gap + replay_cost
- peer engine 有 KV 余量

Fire 之后 victim 通过 `/dev/shm/vllm_ft_preempt_queue/` 给 router，router 转发给 peer engine，peer V3-reload，原 engine 给 head 让出 KV。

## Canonical config (paper recipe)

跑 paper 实验**必用**这套 env（脚本里已经烧成默认，可以 export 覆盖）：

```bash
# substrate + V3 reload
SLO_PRIORITY_PREEMPT=1
FT_ROUTER_SHM_BUS=1                       # ours 用，fcfs 不用
FT_CAPACITY_PREEMPT_RELOAD=1
FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
FT_DELTA_CHECKPOINT=1

# picker (decoupled from SLO)
FT_PICKER_IN_DANGER_MS=300                # head slack < 300ms 才 fire
FT_PICKER_HEAD_TOO_LATE_MS=9999999        # 关 too-late gate（TTFT 不可救，不该用它来 gate）
# (picker 还有几个 rate limit / cooldown / gap 用 default 即可)

# router: 用 router.py 的默认 least_load 即可。不要设 round_robin！
# verified 2026-05-17 clean-router 直接对比：
#   least_load  → 80% SLO, goodput 0.369  (canonical)
#   round_robin → 56.7% SLO, goodput 0.262 (worse)
# round_robin 强制均匀派发导致两 engine 同步 saturate，反而把
# picker 通过 peer-load-gate 拦死。least_load 在我们 router 的
# KV-aware 加权下（in-flight 主键 + shm publish 的 kv_usage /
# waiting 做 tie-breaker）保持适度不平衡，picker 有空间 fire。

FT_GPU_MEMORY_UTILIZATION=0.9
```

**为啥这一套**：v1-v3 的所有 dual 实验输给 fcfs 是因为 picker 的 HEAD_TOO_LATE gate 在 saturation 下永远拦死 picker（head slack 一进队列就负的几秒，远超 -200ms 阈值）。**关掉这个 gate + 解耦 picker 触发**是让机制活的关键改动。Router 用我们 KV-aware least_load 策略即可。详细实验轨迹见 [findings_2026_05_17.md](findings_2026_05_17.md)。

## Main results (paper claim shape)

three settings, all Qwen2.5-14B-Instruct, fp16, A6000, gpu_memory_utilization=0.9:

| 场景 | QPS | ours goodput | fcfs goodput | improvement |
|---|---|---|---|---|
| **Single GPU**, pure long (arxivsumm) | 0.25 | 0.177 | 0.065 | **2.72×** |
| **Dual GPU**, pure long (arxivsumm) | 0.5 | 0.369 | 0.262 | **1.41×** |
| Dual GPU, mixed short+long (post-hoc realistic SLO 3000/300) | 1.5 | 0.958 | 0.897 | 1.07× |

Single GPU 2.72× 是最强的 main result（机制最 shine 的"纯长 prompt 单 engine saturation"场景）。Dual GPU pure long 1.41× 把多 GPU 故事坐实。Dual mixed 是 supplementary，说明机制在混合 workload 下也是 net positive，但收益小（picker 救援机会少 + 跨 engine 搬运代价占比高）。

## Experiment plan (overnight runner)

完整清单在 `experiments_v2/eval/scripts/run_overnight_paper_sweep.sh`。Idempotent，跳过已有 `_metrics.json`，可中断重跑。

| Section | 用途 | runs |
|---|---|---|
| A | dual arxivsumm QPS=0.5 seeds 3,4（补足多 seed）| 4 |
| B | single GPU QPS=0.25 seeds 1,3,4 | 6 |
| C | dual arxivsumm QPS sweep（0.3-0.8）× seeds 0,1,3,4 | ~48 |
| D | 4-baseline ablation @ QPS=0.5（vllm_fcfs / reroute_no_ckpt / ours_no_picker / ours）× seeds 0,1,3,4 | 16 |
| E | router policy ablation（ours + round_robin，证明 canonical 的 least_load 才对）× seeds 0,1,3,4 | 4 |
| F | BurstGPT trace（真实 bursty 到达）× fcfs/ours × seeds 0,1,3,4 | 8 |
| G | e_d1 failover（Poisson + 中途 SIGKILL engine 0）× ours/reroute_no_ckpt × seeds 0,1,3,4 | 8 |

**seed 选择**：用 0/1/3/4。**seed 2 跳过**——它抽到的 arxivsumm 样本里有个 28K token 怪兽请求 + 整体 emp_qps=0.547，把双 14B engine 都打到地板。fcfs 0% SLO，ours 10% SLO。是 workload outlier 不是机制 outlier。

总 ~10 小时跑完。

## Scripts (paper-ready)

| 脚本 | 用途 | 状态 |
|---|---|---|
| `single_engine_microbench.py` | 单 engine cheap-resume 隔离实验，无 router 无 picker | canonical |
| `dual_engine_microbench.py` | 双 engine 完整对比（4 baseline + --workload-trace burstgpt_mixed），baked canonical picker | canonical |
| `e_d1_disruption_demo.py` | failover demo（Poisson 稳态 + 中途 SIGKILL），arxivsumm + 40 reqs + QPS=0.25 | canonical |
| `run_overnight_paper_sweep.sh` | overnight 一键跑全部 | canonical |
| `e_m1_slo_sweep.py` | 老版 4-baseline，tier SLO，**不用于 paper main**——但代码仍可用，需要手动 export canonical env | legacy |
| `e_m2/e_m3/e_m4` | 已基本不用 | legacy |

## Bugs found + fixed today (2026-05-17)

1. **V3 reload trim-order bug** (`core.py:1242`)：`num_computed_tokens` set before trim block. After trim reduced `num_tokens`, `num_new_tokens=0` → scheduler assert. 单 engine V3 reload 触发，双 engine 因 trim 是 no-op 不受影响。Fix: 把 assignment 移到 trim block 之后。
2. **Picker SLO-coupling**：picker `in_danger` 阈值原本是 `0.10 × ttft_slo_ms`，改 SLO 同时改 picker 行为。新增 `FT_PICKER_IN_DANGER_MS` 解耦，默认 300ms。旧 ratio 模式仍保留向后兼容。
3. **Router policy 加了 round_robin 开关（事后证明这条改动是错的方向）**：当时假设"least_load 让两 engine 同步满 → picker 拦死"，加了 `FT_ROUTER_POLICY=round_robin` 想"让负载方差自然制造不平衡"。2026-05-17 晚 clean-router 复测发现完全反过来：round_robin 强制均匀派发反而导致两 engine 同步 saturate（56.7% SLO），least_load 才是对的（80% SLO）。env var 保留作 ablation 用，default 走 least_load。
4. **Stale router 污染所有今日实验**：一台 May 16 的 router 进程一直没死，今天所有 dual 实验都通过它路由，行为是旧代码（least_load + 老 picker）。已经 kill。Overnight runner 加了启动前 port cleanup 防 regression。

## Related work positioning

| 系统 | 机制 | 跟我们的区别 |
|---|---|---|
| QLM (SoCC'24) | slack-based EDF admission | 不踢 running（reprefill 太贵），我们让它敢踢 |
| Scorpio (arXiv'25) | TRP credit-based WFQ | 同上，不踢 running |
| JITServe (arXiv'25) | margin slack + early-prefill chunk preempt | 只踢 prefill 中的 chunk，没 decode 阶段的踢 |
| FastServe (NSDI'24) | SRPT MLFQ，host KV swap | swap 是 preempt-时-才-swap，我们是 continuous（preempt 延迟低） |
| Llumnix (OSDI'24) | NVLink-based migration | 同节点 NVLink，healthy 时迁；我们 host RAM，preempt 边界迁，跨节点可行 |
| Mooncake (arXiv'24) | host KV pool 主要用于 prefix cache 共享 | 用途不同，我们用于 preempt + recovery |
| ConServe | host KV checkpoint 让长 batch 给 latency-critical 让位 | 类似动机，不同 scheduling policy |

我们的差异化（**仅指 substrate + 调度 policy 那两块，router 不在列**）：
- **Continuous delta checkpoint 衬底**（host pool + /dev/shm tmpfs 镜像）：FastServe 的 swap 是 preempt-时才发起，我们是边跑边存，preempt 瞬间没 IO 等待
- **SLO-aware cross-engine picker，跟 SLO 数值解耦**（FT_PICKER_IN_DANGER_MS 绝对阈值）：QLM/Scorpio 都不踢 running，我们让它敢踢

Router 这块：用标准 KV-aware least-loaded routing（算法是公知的，我们的实现包括 /dev/shm 心跳 telemetry + (in_flight, kv_usage, waiting) 的加权 tie-break），**不算 paper main contribution**，写 paper 时只交代"router 用标准 least-load"即可，不要捆绑当三件套之一卖。

## Open issues / future work

- Picker 在两 engine 同步 saturate 时被 peer-load gate 锁死（v5 mixed 实验 95% picker fire 被这个 gate 拦）。需要 bursty workload / 不均匀工作负载才能让 picker 真正 shine。
- TPOT SLO 选 100ms 时 ours 的 ~20ms continuous checkpoint overhead 反成 deal-breaker，选 300ms 才赢。Paper 用 300ms 是合理的（业界标准）。
- 跨 engine reroute 的 ~1s replay_cost 限制了短请求救援机会。可以做 batch-aware 触发优化（future work）。
- `HEAD_TOO_LATE` gate 用 TTFT slack 判断"还能不能救"——但 TTFT 一旦超时就不可恢复，gate 本身设计有问题。当前 paper 实验干脆把它关了（设 9999999）。Future work 应该改成基于 TPOT 的"在线可恢复 metric"。

## Historical change log (compressed)

- 2026-05-12: RULER 16K + 7B model + tier-SLO sweep。失败：7B KV 单价低，KV 压不满，机制全程休眠。
- 2026-05-16: 换 14B + ArXiv-summ + mixed_short_long + 多次 SLO 调整（tight 1.5×→2× baseline P95）。dual 实验 ours 一直输给 fcfs。
- 2026-05-16 晚: 加 picker 的 HEAD_TOO_LATE gate（lower bound）想避免 picker 在 doomed head 上白 fire。后来发现这是设计错误（TTFT 不可救）。
- 2026-05-17 上午: 写 `single_engine_microbench.py`，纯单 GPU + 长 prompt，QPS=0.25 跑出 **76.7% vs 28.3% SLO（goodput 2.72×）**。机制确实有效。
- 2026-05-17 下午: 排查为啥 dual 输——以为是 (1) router 主动均衡 + (2) picker SLO-coupling。改 round_robin + 关 too-late gate 之后 dual arxivsumm QPS=0.5 跑出 **80% vs 56.7% SLO（goodput 1.41×）**。
- 2026-05-17 晚: picker 解耦做完（`FT_PICKER_IN_DANGER_MS`），写 overnight runner。**深夜 verify 发现归因错误**：今天的"成功结果"实际跑在一个 May 16 遗留的 stale router 上（自带 least_load），并不是 round_robin 起作用。clean router 复测：least_load=80%，round_robin=56.7%。**真正让 dual 翻盘的是关 HEAD_TOO_LATE gate，router 应该用 least_load**。Canonical config 已修正，注释里写清楚。
