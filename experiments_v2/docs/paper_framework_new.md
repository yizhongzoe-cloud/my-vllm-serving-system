# Paper framework: Disruption-Aware SLO Scheduling for Long-Context LLM Serving

**Working title**: "Cheap Preemption Enables Disruption-Aware SLO Scheduling for Long-Context LLM Serving"
**Target venue**: APSys 2026 workshop, 6-8 pages
**Writing quality target**: OSDI / SOSP full-paper level (compressed, but no compromise on rigor)

---

## Target paper structure

Following Niyama's compact 6-section structure (Niyama is closest in length to workshop scope), but with TokenFlow's mechanism-first design ordering and Llumnix's mechanism-microbench-first evaluation discipline. Section page estimates assume single-column APSys format.

| Section | ~pages | ~paragraphs |
|---|---|---|
| 1. Introduction | 1.0 | 6 |
| 2. Background & motivation | 1.0 | 6 |
| 3. Design | 2.0 | 12-14 |
| 4. Implementation | 0.5 | 3-4 |
| 5. Evaluation | 2.0-2.5 | 12-15 |
| 6. Discussion / Limitations | 0.3 | 2-3 |
| 7. Related work | 0.5 | 3-4 |
| 8. Conclusion | 0.2 | 1 |

---

## Section 1: Introduction

**Paragraph 1 — open with the prefill cost of long context**:
Establish that modern LLM workloads (RAG with retrieved document context, long-document QA, agentic tool-use traces) routinely push prompts past 16K tokens. Quote a concrete number — RULER 16K prefill on Qwen2.5-7B + A6000 takes ~2.7 seconds (our calibration). Frame this prefill cost as "burned compute" — once it's done, throwing it away is expensive.

**Paragraph 2 — two scenarios that burn the prefill**:
Name two distinct cases where the system has to discard a request's prefill state and start over. (a) **Engine failure mid-decode**: the request lives on one GPU; that GPU dies (hardware fault, OOM, scheduled drain, container kill). To finish the request you reprefill on another GPU. (b) **Priority preemption mid-decode**: a tight-SLO request arrives, queue head's deadline is closer than running request's. Existing schedulers (QLM, Scorpio, Niyama) cannot preempt the running long-context request because reprefill cost dominates any priority benefit. Both scenarios share the same root cause — the system has no cheap way to *resume* a partial decode.

**Paragraph 3 — why existing solutions are siloed**:
Llumnix's live-migration recovers from preempt-like events but requires the source engine alive (KV is transferred GPU↔GPU). It collapses under disruption — exactly the case we care about. SLO schedulers (QLM, Scorpio, Niyama, JITServe) sidestep the problem by limiting preempt to early-prefill chunks or to short-context requests; they leave long-context decode untouchable. Mooncake and TokenFlow maintain host-side KV pools, but the published systems target prefix-cache reuse or pure preempt-and-resume — *not* recovery + cross-engine. No published system co-designs the host KV mechanism with a scheduling policy that exploits it for both.

**Paragraph 4 — our insight in one sentence**:
A single host-side KV checkpoint, published asynchronously to /dev/shm at block-aligned cadence, can simultaneously serve as (i) the recovery substrate when an engine dies — the next engine reads checkpoint and resumes decode without reprefilling — and (ii) the cheap-preempt substrate that lets a *scheduler* preempt running long-context decode without wasting prefill, because the request resumes from checkpoint after eviction. The same data, two uses, no duplication.

**Paragraph 5 — contributions, three claims**:
List three contributions corresponding to three artifacts the reader can verify. (1) A host-side KV checkpoint mechanism with a two-tier store (in-process pool + /dev/shm mirror) that decouples cadence from access path. (2) A slack-based preempt picker that uses cheap preempt to make decisions impossible without it — preempts long-context decode when a tighter-SLO request needs the slot. (3) Cross-engine reroute via the same checkpoint, with a state machine that restores KV from shm and resumes decode on a surviving engine. Empirical claims previewed: minimal overhead under healthy load, SLO attainment improvement under pressure, sub-2-second failover gap on long-context recovery.

**Paragraph 6 — preview implementation + roadmap**:
Note that the system is built on vLLM v0.16 with `O(small)` lines added in the scheduler hook and worker RPC, runs on commodity 2-GPU hardware (A6000 / L40S), and the rest of the paper proceeds: §2 motivation, §3 design, §4 implementation, §5 evaluation, §6-7 discussion and related work.

---

## Section 2: Background & motivation

**Paragraph 1 — long-context serving regime, concrete cost**:
Pin down the regime. Long-context = 16K+ tokens in prompt. Workloads that drive this: RAG with retrieved context (16K-128K), code agents (long traces), document-grounded QA. Give a measured number from our setup: 16K prefill on Qwen2.5-7B on A6000 = 2.7s (95% percentile from a no-pressure calibration run). For comparison, ShareGPT chat workload TTFT P95 = 0.45s. The gap (~6x) is what makes "burn the prefill" expensive.

**Paragraph 2 — disruption is not rare**:
Cite operational reasons engines disappear during request lifetime: (a) scheduled drains for upgrades, (b) preemption in spot/multi-tenant clusters, (c) hardware faults (GPU ECC, network loss), (d) OOM from concurrent oversubscription. Cloud LLM serving operators report incidents at non-trivial rate; long requests are disproportionately hit (they live longer, exposure proportional to lifetime). Note that LLM-specific traffic adds a new disruption vector — model swap-out / cold restart.

**Paragraph 3 — what current SLO schedulers do under load**:
Walk through QLM/Scorpio/Niyama/JITServe in one paragraph. Their common shape: predict prefill+decode cost from history, schedule under deadline constraints, *avoid* preempting running requests because preempt = recompute = SLO violation. The result is admission-control-heavy designs: keep new requests waiting, never disturb the running set. This works for short contexts (low prefill) but degrades for long contexts where waiting tail dominates.

**Paragraph 4 — what current recovery systems do**:
Llumnix migrates requests between engines via GPU↔GPU peer-copy of KV. It improves load balance and defragmentation. But its mechanism requires both endpoints alive — when source dies suddenly, migration is impossible. Mooncake/TokenFlow host-side KV pools target prefix cache or local preempt; neither has been demonstrated for cross-engine recovery.

**Paragraph 5 — the gap nobody has filled**:
State the gap as a 2x2: (preempt vs disruption) × (cost-cheap vs cost-expensive). All four quadrants have well-known systems except one: "cheap preempt + disruption-aware" is empty. Existing host-KV mechanisms could fill it, but they aren't co-designed with a scheduler that exploits the cheap preempt for SLO purposes. This co-design is our contribution.

**Paragraph 6 — what changes if you have cheap preempt**:
Spell out the consequence of cheap preempt for scheduling. Without cheap preempt, you must predict latencies and admit conservatively (proactive). With cheap preempt, you can admit liberally and *correct reactively* when you see a request slipping — preempt the most-relaxed running request to make room for the tightest waiting one. This shifts scheduling from "prediction-heavy proactive" to "measurement-heavy reactive". The shift removes the dependency on offline-profiled latency models (Scorpio's α/β/γ/δ, Llumnix's headroom constant, JITServe's v_token). Tease the slack formula by name; full definition in §3.

---

## Section 3: Design

**Subsection 3.1: Overview** (1 paragraph + figure)
Describe the architecture in one paragraph anchored to Figure 1. Client → router process (CPU, separate) → engine processes (one per GPU). Engines publish KV checkpoints to a host-side store with two tiers: an in-process pinned-memory pool, and a /dev/shm mirror visible to other engines. The router watches engine health via a shm status file and reroutes on death. Each engine runs a slack-based scheduler hook in its `schedule()` call. Note explicitly: the same checkpoint data backs three uses (same-engine preempt-and-resume, cross-engine recovery, future prefix-cache reuse — we discuss only the first two).

**Subsection 3.2: Host-side KV checkpoint** (3-4 paragraphs)

**Para 3.2.1 — what gets saved, when**:
Block-aligned cadence: every 16 tokens (one full vLLM KV block) decoded, the engine saves that block's K/V tensors. The save is per-block delta, not full-KV — only newly-stable blocks since last save. Block alignment is mandatory because vLLM allocates KV in fixed blocks and a partial block has no well-defined boundary.

**Para 3.2.2 — two-tier store**:
Layer 1 is an in-process `KVCheckpointPool` of pinned host memory. Same-engine preempt-and-resume reads from Layer 1 directly, sub-millisecond. Layer 2 is a /dev/shm mirror written atomically via tmp + rename per chunk, plus a manifest and "latest" pointer for atomicity. Cross-engine recovery reads Layer 2 and scatter-copies to the new engine's GPU. The data is identical in both tiers; only the access path differs.

**Para 3.2.3 — async publish**:
The Layer 2 mirror would naively block the engine step on file I/O (a few ms per save × multiple saves per decode = real overhead). We move the shm write to a background thread; the engine step returns immediately after the GPU→host copy completes (~40 µs for 1 MB on PCIe Gen4). A single ThreadPoolExecutor with one worker plus a future-based backpressure rule (next RPC waits for previous future before submitting) bounds in-flight publish to one batch.

**Para 3.2.4 — correctness invariants**:
State three: (i) "latest" pointer only updates after the full chunk file is flushed and renamed, so cross-engine restore never sees a partial chunk; (ii) num_checkpointed_tokens is updated only after the publish future completes, so the scheduler never assumes data is durable before it is; (iii) on engine death, in-flight publish thread exits cleanly because the engine subprocess dies — the worst case is the last partial save is not visible to readers, which is captured by "latest" pointer not advancing.

**Subsection 3.3: Slack-based preempt picker** (4 paragraphs)

**Para 3.3.1 — slack as the single quantity**:
Define slack(r, t) = how much wall-clock time remains before request r misses its SLO. Split by stage — pre-first-token requests use TTFT slack = S_TTFT − elapsed_since_arrival; post-first-token requests use TPOT slack = S_TPOT − avg_per_token_so_far. Slack collapses both SLOs into a comparable signal. Recall from §2 that scheduler is now reactive, so slack is *measured* from runtime, not predicted from a model.

**Para 3.3.2 — the picker rule**:
Every scheduler step, pick the running request with maximum slack (the most relaxed) and the waiting request with minimum slack (the most urgent). If their slack difference exceeds a hysteresis threshold δ *plus the replay cost*, preempt the running request. Replay cost is the extra TPOT × tokens that the new engine would have to recompute on resume — it represents the wasted work of preempting. Formalize: preempt iff `slack(victim) − slack(head) > δ + replay_cost(victim) AND num_checkpointed_tokens(victim) > 0 AND time-since-last-preempt(victim) > cooldown`.

**Para 3.3.3 — three guards**:
The hysteresis δ prevents thrashing. The replay-cost term ensures we only preempt when the trade is favorable (preempting a victim with no checkpoint is a no-op because resume would need full reprefill — caught by `num_checkpointed_tokens > 0`). The cooldown prevents reprocessing the same victim every step. All three are tunable but defaults (δ=1s, cooldown=5s) work across our experiments.

**Para 3.3.4 — comparison to prior schedulers in one sentence**:
QLM/Scorpio/Niyama are proactive — they predict and admit. We are reactive — we measure and preempt. The shift is enabled by cheap preempt: predicting becomes optional when correcting is cheap.

**Subsection 3.4: Cross-engine reroute** (3 paragraphs)

**Para 3.4.1 — engine death detection**:
Each engine writes a heartbeat file to `/dev/shm/vllm_ft_engine_status/engine_<id>.json` every 200 ms via a daemon thread (must be a thread, not piggybacked on `step()` — vLLM's busy loop blocks in queue.get when idle). The router polls these files every 500 ms; an engine whose timestamp is older than 2 s is marked dead. Death triggers reroute of all that engine's in-flight requests.

**Para 3.4.2 — reroute decision**:
The router maintains an `in_flight` dict mapping `router_req_id → (engine_id, body)`. On engine death, the router enumerates its in-flight requests and re-forwards each to a surviving engine. The reroute body carries `vllm_xargs.is_rerouted=True`, the original engine's `internal_req_id` (so the new engine can find the checkpoint), and the published `num_checkpointed_tokens`.

**Para 3.4.3 — KV restore + decode continuation**:
The new engine sees `is_rerouted` in `EngineCore.add_request` and diverts the request into a V3 reload state machine instead of normal prefill. The state machine allocates KV blocks for `num_checkpointed_tokens`, calls `restore_kv_blocks` RPC on the worker (which reads from `/dev/shm/vllm_ft_checkpoints/<original_id>/`), and flips the request to `PREEMPTED` once restore completes. vLLM's standard resumed-from-preempt admission path picks it up next step. We clamp restored tokens to a block boundary at most equal to the prompt length, then re-decode from there — this discards output tokens but is necessary because output token IDs aren't shipped cross-engine (only KV is).

**Subsection 3.5: One mechanism, three uses** (1 paragraph)
Tie the design together. The host-side checkpoint is the only mechanism that creates new data; the scheduler and reroute both consume it. The slack picker (§3.3) reads `num_checkpointed_tokens` to gate preempt safety. The reroute (§3.4) reads `/dev/shm/vllm_ft_checkpoints/<id>/` to find the KV. The same data path. If we removed the checkpoint, both the scheduler and reroute would degrade — the scheduler would have to fall back to no-preempt-of-long-context (the QLM/Scorpio position), and reroute would have to fall back to reprefill (the Llumnix-when-source-dies position). The mechanism-policy coupling is the contribution.

---

## Section 4: Implementation

**Paragraph 1 — base + scope**:
Built on vLLM v0.16. Modified files: scheduler.py (slack picker, ~80 lines), engine core (state machine + status writer, ~200 lines), worker (`checkpoint_kv_blocks` and `restore_kv_blocks` RPCs, ~150 lines), one new router process (~400 lines), one new workload runner (~200 lines). Total ~1000 lines of new code on top of vLLM.

**Paragraph 2 — engineering pitfalls worth noting**:
Three concrete pitfalls solved during development, each illustrating a non-obvious design choice. (i) vLLM's OpenAI API server mangles `X-Request-Id` headers with `cmpl-<id>-0` prefixes, so the router can't use the user-supplied id for tracking — we introduce `vllm_xargs.router_req_id` that flows through cleanly. (ii) Engine status writes piggybacked on `step()` break when the engine is idle (busy loop blocks in queue.get with no step calls); fix is a daemon thread that writes unconditionally every 200 ms. (iii) Default inline /dev/shm write blocks the engine step ~5-15 ms per save, accumulating ~30% TTFT overhead — moving to background-thread publish drops this to ~10 ms TTFT overhead.

**Paragraph 3 — workload + measurement infrastructure**:
Built a parameterized Poisson workload generator (Niyama-style 3-tier QoS) and an engine-log post-hoc analyzer (per-request TTFT/TPOT extracted from a structured `FT request_done` log line added in `output_processor`). Measurement is server-side (no streaming HTTP needed) — TTFT/TPOT come from vLLM's internal `RequestStateStats` computed in the correct clock domain. This is the same convention as Llumnix and other prior LLM serving papers.

**Paragraph 4 — open-source intent (optional)**:
State availability — code, scripts, dataset prep at github.com/[redacted]. If we open-source, mention here.

---

## Section 5: Evaluation

**Subsection 5.1: Setup** (1 paragraph as a block + a small table)

Single dense paragraph documenting hardware (2x NVIDIA A6000 PCIe, 48GB each + 2x L40S for portability), model (Qwen2.5-7B-Instruct fp16, max_model_len 32K), workloads (RULER 16K for long-context, ShareGPT for short-context, both with Poisson arrivals), SLO calibration methodology (1× baseline P95 from `vllm_fcfs` at 0.02 QPS RULER 16K / 0.1 QPS ShareGPT, then 2× multiplier per JITServe convention for the main figure), baselines (vllm_fcfs as router-less floor, reroute_no_ckpt as our-architecture-without-FT control, ours as full system, ours_no_picker as picker-ablation variant). Three seeds per configuration, mean ± std reported.

**Subsection 5.2: Healthy-load overhead** (1-2 paragraphs)

**Para 5.2.1 — overhead numbers (E_M3)**:
At low QPS with no disruption, our system adds ~30 ms TTFT P50 overhead vs reroute_no_ckpt (which has the router but no FT features). Throughput is identical to within 0.5%. TPOT is identical. Conclude: FT mechanisms do not impose meaningful tax in the absence of pressure. The router adds essentially zero overhead (vllm_fcfs ≈ reroute_no_ckpt). This is Figure 2.

**Para 5.2.2 — what the residual overhead is from**:
Honest accounting. The residual is primarily from per-step `_save_checkpoints_if_needed` iteration over running queue plus the always-on V3 state machine queue check. Both are O(running queue) and trigger every scheduler step but only do work when there's something to save. The async publish (§3.2.3) already removes the dominant source. We argue this is acceptable because the metric that matters (throughput, TPOT) is unaffected.

**Subsection 5.3: SLO attainment under load** (3-4 paragraphs)

**Para 5.3.1 — main figure (E_M1) describing the sweep**:
Figure 3: x = QPS, y = SLO attainment %, three lines (ours, reroute_no_ckpt, vllm_fcfs). Sweep 0.1 — 4.0 QPS for ShareGPT (and analogous for RULER 16K). Describe the shape of the curves: all systems start at 100% attainment at low QPS; the curves diverge as QPS increases; ours holds attainment higher and degrades later. Quote a specific "saturation knee" point — e.g., at 1.0 QPS, ours = X%, reroute_no_ckpt = Y%, vllm_fcfs = Z%.

**Para 5.3.2 — why the picker wins under load**:
Connect numbers to mechanism. At pressure, the picker activates: tight-SLO waiting heads get served by preempting loose-SLO running tails. Baselines have no such option (no cheap preempt) so the head waits behind the tail. Long-context decode (in RULER 16K) shows the win most clearly because the preempt-and-resume saves seconds of prefill per preempt.

**Para 5.3.3 — tiered workload (E_M1-tiered)**:
Argue that uniform-SLO uniform-arrival doesn't fully expose the scheduling advantage (everyone is equally urgent). Switch to Niyama-style 3-tier QoS classes (tight=1×, normal=3×, loose=6× baseline P95). Figure 4: per-class attainment vs QPS. Tight class: ours maintains ≥95% well past where baselines drop below 50%. Loose class: all systems maintain >90% (loose SLO is easy). Normal class: intermediate. Story: ours protects tight class without breaking loose class — the "right" reallocation.

**Para 5.3.4 — SLO tightness sweep (E_M2)**:
Fix QPS at the knee. Sweep SLO multiplier from 1.5× to 6×. Figure 5: x = multiplier, y = ours' attainment minus baseline's attainment. The gap is largest at tight SLO (1.5×) and shrinks toward zero at loose (6×). Conclude that ours wins exactly where the win matters — when SLOs are non-trivial to meet.

**Subsection 5.4: Mechanism vs policy ablation** (1-2 paragraphs)

**Para 5.4.1 — picker ablation (E_M4)**:
Add `ours_no_picker` to Figure 3 — same FT machinery (checkpoint enabled, V3 reload available) but slack picker disabled. Shows the contribution of the *policy* vs the *mechanism*. Expected result: ours_no_picker is between ours and reroute_no_ckpt — the checkpoint mechanism gives some passive benefit (cheaper resumes when forced by capacity), but the slack picker is what actively uses cheap preempt to maintain SLO.

**Para 5.4.2 — interpretation**:
Tie back to §3.5: the mechanism alone is half the story; the policy alone is the QLM/Niyama position; only the co-design fully realizes the advantage.

**Subsection 5.5: Disruption recovery** (2 paragraphs)

**Para 5.5.1 — single-disruption demo (E_D1)**:
Figure 6: bar chart of `failover_gap` (time from engine SIGKILL to next token of affected requests) for three systems × ShareGPT and RULER 16K. Ours: ~1.5s. reroute_no_ckpt: depends on context length — ~2.5s at 4K, ~10-20s at 16K (full reprefill cost). vllm_fcfs: undefined (clients receive 5xx errors). The gap *grows with context length* — at 64K it would be tens of seconds, which is exactly the long-context regime we target.

**Para 5.5.2 — tail-latency claim (extending failover_gap interpretation)**:
Note that our system not only has lower *mean* failover_gap but also lower *variance* — all rerouted requests recover in one V3 batch synchronously (sub-1ms spread). reroute_no_ckpt staggers recoveries through the scheduler's admission path, exposing tail. This is a second, independent argument for our design.

**Subsection 5.6: Sensitivity** (optional, 1 paragraph if space)

If space allows: vary one of (publish cadence, hysteresis δ, cooldown). Show monotone or non-monotone trend. If no space, drop this in favor of more disruption-recovery detail.

---

## Section 6: Discussion / Limitations

**Paragraph 1 — explicit scope limits**:
Three honest limits. (i) We tested 2 engines; chained failure or 3+ engines are out of scope. (ii) Workload-level recovery (router re-binds request → new engine) requires the router to be a single point — we don't address router HA. (iii) Cross-engine restore loses output tokens already produced on the dead engine; this is harmless for our SLO accounting (the new engine re-emits from a slightly earlier point) but means total work done is slightly larger than no-disruption.

**Paragraph 2 — extensions, marked as future work**:
Adaptive checkpoint cadence tied to slack (tight requests checkpoint more frequently); cross-engine slack coordination (when an engine reroutes in, peer engines could yield); larger model context via Qwen2.5-7B-Instruct-1M variant.

**Paragraph 3 (optional) — why we didn't address admission control**:
QLM, Scorpio rely on admission. We don't — by choice, to keep the policy minimal. Admission is orthogonal and can be added.

---

## Section 7: Related work

**Paragraph 1 — LLM serving SLO schedulers**:
QLM (SoCC'24, slack-based queueing with LP solver), Scorpio (TTFT/TPOT guards, ITL-based admission), JITServe (GMAX with QRF length predictor), Niyama (3-tier QoS with deadline EDF + relegation). Common pattern: proactive prediction-based, no preempt of long-context running requests. We differ on the last point — we preempt long-context decode by making preempt cheap.

**Paragraph 2 — host-side KV pools**:
Mooncake (host KV for prefix cache reuse), TokenFlow (host-side write-through for preempt-and-resume within an engine). Both stop short of cross-engine recovery. Our checkpoint mirror reuses these mechanisms' philosophy but adds the /dev/shm tier for cross-engine access and co-designs with the scheduler.

**Paragraph 3 — migration-based fault tolerance**:
Llumnix is the canonical example — live KV migration, used for load balancing + priority isolation. Their fault-tolerance discussion (Sec 5) is architectural-only; the eval has no instance-loss experiments. The fundamental constraint is mechanism: live migration requires source alive. Our checkpoint-mirror approach is the complementary side of this design space — survives source death.

**Paragraph 4 — broader preempt-capable systems**:
FastServe (skip-join MLFQ + host swap), Andes (knapsack on QoE / KV cost), SLOs-Serve (multi-stage DP). All preempt-capable but at different granularities and for different purposes. Position our work as the disruption-aware corner of this space.

---

## Section 8: Conclusion

**Paragraph 1 — restate + look ahead**:
Restate that cheap preempt unifies SLO scheduling and disruption recovery into one design. Quote the headline number — SLO attainment improvement at saturation + failover_gap improvement at disruption. Close with one line on the broader implication: when an underlying mechanism's cost drops below a threshold, design space that was previously off-limits becomes available — host-KV checkpoint crossed that threshold for long-context LLM serving.

---

## Cross-paper structural reminders (from reference_paper.md)

Drafting checklist informed by TokenFlow + Niyama + Scorpio + Llumnix patterns:

1. **Numerical pain point in intro**, not vague — quote ~2.7s prefill, not "long prefill cost". TokenFlow and Niyama both lead with concrete numbers in para 1.
2. **Mechanism before policy** in design section (TokenFlow + Llumnix pattern) — describe checkpoint first, then how scheduler uses it.
3. **Eval order: end-to-end first, then mechanism micro, then ablation** (TokenFlow + Niyama). We invert only for §5.2 healthy overhead since that's the smallest claim and de-risks the reader before showing the wins.
4. **Compact contrastive related-work** (Niyama bucket-by-theme style) — group by what each system *can't* do, mapping to our contribution.
5. **Explicit scope-limits section** (TokenFlow §8 discussion pattern) — write down what we don't handle, in a separate paragraph in §6.
6. **Formalize then relax** if there's an objective worth formalizing (TokenFlow proxy-objective pattern). Our slack formula is the candidate — formalize in §3.3.1, note that it's a heuristic proxy for "minimize expected SLO violations" without solving the LP.
7. **Workshop-length compression discipline** (Scorpio pattern) — 1 paragraph per mechanism; preserve math inline, not in separate algorithm blocks; single-paragraph related work that groups citations.
8. **Mechanism vs scenario abstraction** (Llumnix's "virtual usage" unifies four scenarios) — our equivalent is "slack + replay cost" as a single quantity that captures both stage-aware urgency and preempt eligibility. Surface this framing in §3.3.

---

## Open questions to resolve before submission

| Question | Resolution path |
|---|---|
| Should we report 32K or 16K results? | Current model caps at 32K. RULER 16K is the cleanest workload. Mention 32K extension via 1M-context model variant as future work. |
| How aggressive should hysteresis δ and cooldown defaults be? | Run a small sweep (E_M2 sensitivity slot) — δ ∈ {500ms, 1s, 2s}, cooldown ∈ {3s, 5s, 10s}. Pick the pair that maximizes the gap-with-baseline at the knee. |
| Do we report tiered or uniform main figure? | Uniform runs first (already in progress); tiered shows the clearer story but uses synthetic QoS classes. Likely report tiered as Figure 3, uniform as supplementary. |
| What's our cited "failover_gap" — P50, P95, max? | Report P50 + P95 in text, max in figure. Mean misleads here because reroute_no_ckpt has high variance. |
| How to honestly explain the residual 30ms TTFT overhead? | Per-step iteration cost is honest, not a real bug. Mention in §5.2.2. Don't try to optimize it away last-minute. |
