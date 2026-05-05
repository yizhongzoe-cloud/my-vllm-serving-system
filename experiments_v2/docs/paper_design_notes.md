# Paper Design Decisions — Key Insights

Living document. Captures design / writing / experimental decisions
that emerged from measurement and discussion. Most of these are
**measurement-driven** — they recharacterize prior assumptions in
light of Phase 1 + Phase 2 data.

---

## -1. Preempt 按触发原因分流：release vs no-release

**核心 insight (2026-05-03)**: Preempt 不是单一动作，是**两类动作**，
release 策略应跟触发原因匹配。

### 两种 preempt 类型 × 释放策略

| 触发原因 | 真正目标 | block 释放策略 | reload 路线 |
|---|---|---|---|
| **Capacity-driven** (vLLM 自然 KV 内存压力) | 让出 KV 容量 | **释放 block** | 后续 admit 走 chunked prefill，reload 跟着分批 (v1 路线) |
| **SLO/priority-driven** (M3 之类，紧急 req 插队) | 让出 forward batch slot | **不释放 block** | block 留着，几步后 resume；如果 KV 还没被覆盖甚至不需要 reload (pause-resume 路线) |

### 为什么不能一刀切

Phase 2 v2 当前实现把 capacity preempt 也走「不释放 + 整体 reload」
路径。Sanity 测出来 forward step 数下降 60% — 因为 X 占着 1027 个
block 不还，其他 req `kv_cache_manager.allocate()` 看到 free pool 不
够，admit 受阻。

**Capacity preempt 的根本目的是要内存。不释放等于没解决根本问题，
还反而拖累集群并发**。所以 capacity-driven 场景必须 release。

反之 SLO/priority preempt 的目的是要 forward batch slot — 紧急 req
进 batch 跑几步就行，被踢的 X 不需要让内存。如果几步后 X 又被 admit
回来，block 没释放反而省一次 reload（甚至 KV 物理还在 GPU 上没被
覆盖 → pause-resume，零 PCIe 流量）。

### Adaptive policy 第二条规则: short-context 不开 ckpt (实测验证)

V3 sweep + extension sweep 实测发现:

- 4K-64K 上 V3 forward kernel time 跟 vanilla 完全持平 (1.0×)
- 1K 上 V3 完全崩溃: -82% throughput, decode-only step 数量从 vanilla 的 ~3500 降到 V3 的 ~6
- 原因不是 GPU compute 被挤, 是系统级动态循环: 短 prompt 下 reload + 重新 chunked admit + 又被 preempt 形成紧耦合循环

所以 "(loose-SLO ∧ long-context)" 那条规则的**长 context 下限实测 ≥ 4K**, 1K 必须 disable ckpt:

```
context < 4K (or 8K to be safe):
  disable ckpt entirely; rely on vLLM RECOMPUTE path
context >= 4K + (loose SLO):
  enable ckpt + reload (V3)
```

详细数据 + 图见 ckpt_overhead_findings.md "Phase 2 Extension Sweep" section.

### Paper framing 升级

不是「一种 reload 机制」，是「**两种 preempt 类型 × 两种 reload 策略**」。

```
Trigger=capacity:
  release block → 走 chunked admit + chunked reload
Trigger=SLO/priority:
  retain block → 直接 resume，可能跳过 reload
```

混合系统识别 trigger 原因再分流，不一刀切。

### 当前 Phase 2 实验局限

**目前实验是为了证明 reload 机制的成本和效果（reload 比 reprefill
快 100x+），不证明分流 design 必要。**

要支撑分流 design 必要性的实验，需要重新设计：

1. **Capacity-driven preempt 场景**: 跑 release (v1 风格) vs no-release
   (v2 风格)。看 throughput 差距 — 应观察到 no-release 集群并发下降。
2. **SLO-driven preempt 场景**: 跑 release vs no-release。看紧急 req
   的 SLO 满足率 + 被 preempted req 的 reload 触发率 — 应观察到
   no-release 省了多次不必要的 reload。
3. **混合 workload**: 验证分流 design 在两种 trigger 都存在的场景下
   优于一刀切。

这些实验设计还需要进一步思考。Phase 2 数据可作为「reload 机制成本基准」
但不直接 prove 分流 design。

---

## 0. Reprefill 是噪声源：拖慢同 batch 的 decode 请求

**关键观察**: vLLM 默认 chunked prefill 把 prefill chunks 跟 decode 混在
**同一个 forward batch step**。一个 prefill chunk (~2048 tokens) dominates
batch forward time，**同 batch 其他在 decode 的 req 也跟着慢**：

- 纯 decode batch（10 个 req）→ forward step ~30 ms
- 同 batch + 1 个 prefill chunk → forward step ~100-200 ms
- 那 9 个 decode req 这一步**3-7× 慢**，不是它们自己的问题

**对 reprefill 路径 (NoFT-Reprefill baseline) 的影响**:
- 每次 capacity preempt 后，被踢的 req 重新 chunked prefill 整个 prompt
- 16K context = 8 个 2K chunk = 8 个连续 step 都被污染
- Thrashing 状态下，**几乎每个 forward step 都混着 prefill chunk**
- 系统的所有 decode req latency 集体被拖累

**对 CkptReload 路径的影响**:
- 被 reload 的 req 这步 `num_scheduled_tokens=0`（C 方案）
- 完全不进 forward batch
- 其他 decode req 该多快多快，**零 collateral damage**

**Paper 写法**:
> "Reprefill is not just self-slow; under chunked prefill it pollutes
> concurrent decode latency. Each reprefill chunk shares a forward step
> with active decode requests, dilating per-step latency by 3-7x. Async
> reload sidesteps this by removing the recovering request from the
> current step's batch, achieving zero-collateral recovery."

**已有数据能验证 (无需重跑)**:
- `forward_times.csv` 的 `num_input_tokens` 字段反映 batch 总 token 数
- num_input_tokens ≈ batch_size → 纯 decode step
- num_input_tokens 远大 → 混 prefill chunk
- NoFT cell 的纯 decode steps 应该比 CkptReload cell 慢（因为 thrashing 期 prefill chunks 多）

**Paper 主图 +1 候选**：纯 decode step p99 latency vs context（NoFT vs CkptReload paired）。

---

## 1. Thrashing as motivation framing

**Observation**: At 16K context + RPS 1.2, vLLM's natural capacity-driven
preempt fires at sub-second frequency — sanity logged the same
request being preempted+reloaded **4 times in 1 second**.

**Implication for motivation**:
- Default RECOMPUTE: each preempt costs 5-10s reprefill. 4 preempts/sec
  × 5-10s = 20-40s of work per second of wall time → **system collapses**.
- Our reload: 16ms per restore. 4 preempts/sec × 16ms = 64ms per
  second → **system continues**.
- This is not "faster recovery", it's **enabling a regime that was
  impossible**.

**How to write it**:
- Don't call it "thrashing" in motivation (negative framing). Use
  "high-frequency capacity preempt under long context oversubscription".
- Pair it with: "in production, oversubscription occurs from traffic
  spikes, mixed-priority traffic, lagging elasticity, multi-tenant
  sharing — these are real, not contrived".

**Honest caveat**: thrashing itself is a bad state. Our mechanism
doesn't *eliminate* it (admission control / cooldown does). It makes
the bad state survivable while upper-layer scheduling brings the
system back to a healthy state.

**Headline figure candidate**: throughput-vs-time curve under
oversubscription — RECOMPUTE collapses to ~0, reload sustains.

---

## 2. Adaptive policy: only (loose-SLO ∧ long-context) needs ckpt

**Two independent questions decide whether to checkpoint a request**:
1. Will this request be a preempt victim? — Tight-SLO won't, loose-SLO will.
2. Is recompute cost on preempt non-trivial? — Short-context cheap, long-context expensive.

**Only when BOTH yes** is checkpoint worth it.

| | Short context | Long context |
|---|---|---|
| **Tight SLO** | no ckpt (won't be preempted) | no ckpt (won't be preempted) |
| **Loose SLO** | no ckpt (cheap recompute) | **CKPT** (expensive recompute + will be preempted) |

The (loose-SLO × long-context) cell is the only payoff regime.

**Why this beats simple "context-only" threshold**:
- Long-context tight-SLO requests are protected by scheduler from
  preemption → checkpointing them wastes host memory.
- The two-dim policy aligns ckpt decision with actual eviction risk.

**System-level coherence**: M3's `_pick_slo_preempt_victim` already
filters running reqs by `num_checkpointed_tokens > 0`. So
*not-checkpointing* tight-SLO reqs *automatically protects them from
M3 preemption*. The policy and the scheduler reinforce each other.

**Paper writing**:
> We propose a two-dimensional adaptive policy: checkpoint only when
> the request is both (a) a likely preempt victim (loose SLO) and
> (b) expensive to recompute (long context). The product space is
> the only regime where checkpoint pays off; the other three cells
> incur host memory cost without proportional benefit.

---

## 3. Economic policy is obsolete under async+delta

**Prior assumption** (in `checkpoint_controller.py::_should_checkpoint_by_economic_policy`):
- Checkpoint cost is non-trivial (forward time impact, PCIe pressure)
- Decide per-step: `replay_saved > load_cost + λ × ckpt_cost`?
- Only checkpoint when reprefill savings exceed write+load overhead.

**Phase 1 measurement collapses this**:
- Async + delta + pinned memory + dedicated copy stream → forward
  overhead < 1% across 1K-32K context.
- Plug ckpt_cost ≈ 0 into formula → degenerates to `replay_saved >
  load_cost`, which is **always true** for any non-trivial context.
- Policy degenerates to "always checkpoint" → its complexity is
  unjustified.

**Why economic policy was once reasonable**:
- vLLM's legacy SWAP was synchronous — each save did stall forward
  by hundreds of ms.
- Under sync semantics, economic policy correctly identified savings.
- TokenFlow (EuroSys '26) introduced async + delta; we measure that
  this collapses the cost side of the equation.

**Paper angle (methodology contribution)**:
> Prior FT serving systems use cost-based economic policies to
> balance checkpoint write cost against expected reprefill savings.
> Our isolated measurement shows that under async incremental
> checkpoint, write overhead is below the noise floor of paired
> CUDA-event measurement (<1%). The economic decision degenerates,
> and we propose replacing it with a simpler context+SLO threshold
> policy.

This is a **measurement-driven recharacterization of prior assumption**
— a clean systems-paper move.

---

## 4. Host memory is not a binding constraint

**Measurement** (Phase 1 sweep):
- Pool sized at 128 GB.
- Peak active host KV in steady state: ~30 GB.
- Eviction never triggered.

**Why**:
- Per-GPU active KV is bounded by GPU KV pool capacity, not by
  request rate.
- Single A6000 + 32K context → ~12 in-flight cap → ~48 GB host KV peak.
- Typical server RAM: 256 GB - 1 TB. ~5-20× headroom.

**Paper writing** (don't say "we ignore eviction"):
> We assume host memory is sufficient relative to per-GPU peak KV
> demand. In our target deployment (single GPU, context up to 32K),
> peak host KV occupancy is bounded by GPU pool capacity at ~50 GB,
> well within typical server RAM. Eviction logic exists as a safety
> net but is never triggered in steady-state evaluation. For
> deployments where host memory is constrained (very-long context,
> dense multi-GPU, edge), simple LRU eviction provides correctness
> fallback.

**Important**: combined with the previous point, the "host memory
limit" justification for economic policy also collapses. Both rationales
for economic policy are gone:
1. ✗ ckpt cost not negligible → measured <1%
2. ✗ host RAM scarce → measured well within budget

---

## 5. SSD as memory hierarchy extension (future work)

**Insight**: When host RAM saturates (extreme long context, dense
sharing, multi-model), the same async+delta abstraction extends to
NVMe SSD as a third tier.

**Numbers**:
- Host → GPU: 25 GB/s (PCIe 4.0)
- NVMe SSD → host: 5-7 GB/s
- Reprefill: equivalent to ~tens of GB/s of "compute throughput" but
  superlinear in context length

**SSD reload time for 16K KV (256 MB)**: ~50 ms — still 100-200×
faster than reprefill.

**Existing precedent**: Mooncake (FAST '25) already builds this
multi-tier KV pool.

**Paper writing**:
- Place in **limitations / extensions** section.
- Don't try to implement this for the current paper (scope creep).
- Can be presented as: "the abstraction extends naturally to a third
  NVMe tier; we leave full-stack hierarchy implementation as future
  work, citing Mooncake as a related design".

---

## 6. Why vLLM doesn't use ckpt+reload by default

(Useful for related-work / motivation framing.)

vLLM's history with this mechanism:
1. **Tried it as SWAP mode** (sync) — was slower than RECOMPUTE for
   short contexts, deprecated.
2. **Default to RECOMPUTE in v1**.
3. Trade-off was rationally drawn at the time given (a) sync impl, (b)
   short-context dominance, (c) over-provisioning best practices.

**What changed** (motivating our paper):
1. Long context (32K-128K) entered production en masse — reprefill
   superlinear, RECOMPUTE no longer comfortable.
2. Async + delta + dedicated copy stream — TokenFlow's contribution
   that we extend to long context.
3. Elastic / multi-tenant deployments where over-provisioning isn't
   feasible.

These three shifts flip the trade-off. The paper's job is to make
this flip rigorous via measurement.

---

## 7. Paper writing principles distilled

From this discussion + earlier work:

1. **Use measurement to recharacterize prior assumptions** — don't
   propose a new mechanism, propose new evidence that the old
   trade-off is gone.
2. **Never assert without scope** — say "in our setting X" and back
   it with data, never "X is generally true".
3. **Acknowledge limitations explicitly** — eviction, SSD, multi-GPU
   without NVLink — pair each with a brief mitigation or extension
   path.
4. **Distinguish mechanism contribution vs methodology contribution**
   — we may have both: async+delta+threshold policy (mechanism),
   isolated CUDA-event measurement of forward kernel (methodology).
5. **Headline figures should be system-level, not just mechanism-level**
   — recovery time vs context length is mechanism; throughput-under-
   oversubscription is system. The latter sells the paper better.

---

## 8. SLO priority preempt path (independent design)

Capacity-driven preempt (V3) and SLO-priority-driven preempt (this
path) are two separate mechanisms with different release strategies
and different triggers. They must not share code, state, or env vars.

**SLO priority preempt path uses**:
- Trigger: SLO slack of waiting head vs running queue (no fault
  recovery markers, no ckpt requirement on victim).
- Victim picker: loosest-SLO running request.
- Preempt action: retain GPU blocks (`_preempt_for_slo_retain`).
- Resume: engine drains retained queue after N steps
  (`_process_slo_retained_queue`).
- Env vars: all `SLO_PRIORITY_PREEMPT_*` prefix, never `FT_*`.

**Disjoint from fault path**: do not reuse `_pick_slo_preempt_victim`,
`is_rerouted`, `slo_preempted_pending_restore`, or any `FT_SLO_*` env
var. Fault path stays intact for fault-recovery experiments.

**Open issues for production but out of scope for first paper data**:
- HBM pressure during retain window: retained reqs hold blocks; new
  admissions can saturate the KV pool. Need LRU eviction of oldest
  retained req with fallback to release path.
- Quantitative "tight SLO" definition: first version uses coarse SLO
  budget difference; production may need a more principled metric.

**Simplifications in current implementation (revisit before paper exp)**:

1. **`retain_steps` fixed at 5**. Resume after exactly 5 scheduler
   steps regardless of whether the urgent request that triggered the
   preempt has finished. Better: track which urgent request triggered
   the preempt, resume the victim only after that request's progress
   passes a threshold (or completes).

2. **"Tight SLO" derived from SLO budget slack** (`compute_slo_budgets
   ["min_ms"]`). All requests in the current sweep config share one
   SLO setting, so slack differences come from queue-wait time, not
   from explicit priority labels. Paper's framing assumes mixed SLO
   tiers (tight / normal / loose) — this requires per-request SLO
   labels in the workload generator, not a uniform SLO across the
   cell.

3. **Polling trigger every scheduler step**. The picker runs every
   step (gated by `MIN_INTERVAL_MS=2000`). An event-driven trigger
   that fires only when an urgent request enters the waiting queue
   would be cleaner and lower overhead.

4. **FINISHED-status retained requests are silently dropped**. If a
   retained request transitions to FINISHED_* between preempt and
   resume (e.g. client disconnect, request_timeout), the drain skips
   it. Fine for sanity but a real system should surface a metric.

5. **Retain queue size unbounded**. Multiple urgent requests in
   quick succession can fill the queue with retained victims; no cap.
   May cause large bursts of resumed requests when the retain window
   expires.

These are simplifications that let the mechanism run end-to-end. None
break correctness for the sanity test, but #1 and #2 affect the paper
narrative directly and should be revisited when designing the
per-trigger routing experiment (§9a above).

---

## 9. Remaining experiments before paper submission

Four experiments are needed to bring the paper from "measurement-rigor
study" to "complete contribution". Listed by priority for paper impact:

### 9a. Per-trigger routing 4-cell comparison

Demonstrates that capacity preempt and SLO preempt should run
different release strategies. Without this experiment the per-trigger
routing argument is just a design proposal.

Setup:
- Construct mixed workload with both capacity pressure and SLO
  priority pressure (e.g. tight-SLO requests injected at intervals
  into a steady high-load batch).
- Run 4 baselines:
  1. Capacity preempt × release blocks (V3 current)
  2. Capacity preempt × retain blocks
  3. SLO priority preempt × release blocks
  4. SLO priority preempt × retain blocks
- Metrics: SLO satisfaction rate per priority tier, end-to-end
  goodput, retain-pool peak occupancy.

Expected: each strategy wins in its own trigger scenario;
single-strategy baselines lose in the off-scenario.

### 9b. Multi-GPU PCIe contention

Independent niche TokenFlow does not cover. Quantifies how reload
performance degrades when TP collective communication shares PCIe
with reload copies on machines without NVLink (L40S, A6000 PCIe,
RTX series).

Setup:
- L40S × 2 or A6000 × 2 (no NVLink), TP=2.
- Reload while TP communication is active vs while TP is idle.
- Sweep model size (8B, 13B, 70B) where possible.

Expected: reload time is meaningfully longer when TP traffic shares
PCIe; gives a deployment-relevance argument for the design.

### 9c. Production-shape workload

Required for the motivation chapter to ground the "batch API"
framing in real workload characteristics.

Setup:
- Mixed prompt length distribution (4K / 16K / 64K with realistic
  proportions, e.g. modeled on Anthropic / OpenAI batch traffic).
- Mixed SLO tiers (tight / normal / loose).
- Time-varying arrival rate including bursts.
- Multi-hour cell duration (not 6-minute).

Expected: V3 maintains tight-tier SLO satisfaction under bursts that
collapse vanilla vLLM's RECOMPUTE path.

### 9d. 128K scaling extension

Optional but extends the recovery-time-vs-context main figure. 128K
cannot fit on a single A6000 for an 8B model, so this requires
multi-GPU.

Setup:
- 2-GPU TP, 8B model, RULER 128K.
- Same NoFT-Reprefill vs V3 paired comparison as 1K-64K.

Expected: continues the speedup curve; reprefill cost approaches
minutes while reload stays sub-second.

---

## Cross-references

- `experiments_v2/docs/ckpt_overhead_findings.md` — Phase 1+2 findings
  (measurement + figures)
- `~/.claude/projects/.../memory/project_paper_framing_rag.md` —
  earlier paper framing decisions (long-context non-interactive batch
  framing)
- `~/.claude/projects/.../memory/project_preemption_literature.md` —
  related work map (TokenFlow, QoServe, etc.)
