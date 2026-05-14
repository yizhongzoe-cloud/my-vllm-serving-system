# Paper framework: Disruption-Aware SLO Scheduling for Long-Context LLM Serving

**Working title**: "Cheap Preemption Enables Disruption-Aware SLO Scheduling for Long-Context LLM Serving"
**Target venue**: APSys 2026 workshop, 6-8 pages
**Writing quality target**: OSDI / SOSP full-paper level (compressed, but no compromise on rigor)

---

## Target paper structure

Aligned with the standard APSys / IEEE workshop short-paper shape (5 sections), modeled after recent APSys precedents (e.g. Capybara APSys '24):

- §1 Introduction — short and contribution-forward
- §2 Motivation and Background — problem space + a comparison table that replaces an independent Related Work section
- §3 Design — mechanism-first walk
- §4 Evaluation — Setup paragraph + claim-aligned subsections + a limitations subsection at the end
- §5 Conclusion — one paragraph

| Section | ~pages | ~paragraphs |
|---|---|---|
| 1. Introduction | 1.0 | 4 |
| 2. Motivation and Background | 1.0 | 5-6 (with Table 1) |
| 3. Design | 2.0-2.5 | 11-12 |
| 4. Evaluation | 1.5-2.0 | 8-10 |
| 5. Conclusion | 0.2 | 1 |

Two structural choices worth flagging because they differ from OSDI/SOSP full-paper habits but match APSys / IEEE workshop conventions: (a) no independent Related Work section — comparison happens in §2 via Table 1 plus inline citations; (b) no independent Discussion section — limitations live as the last subsection of §4 Evaluation.

---

## Section 1: Introduction

Short — four paragraphs that announce the contribution rather than develop the full problem space. The detailed motivation (existing systems' inability to address the problem, the 2×2 gap, the design space) lives in §2.

**Paragraph 1 — long-context regime and the cost of throwing prefill away**:
Modern LLM workloads — RAG with retrieved document context, long-document QA, agentic tool-use traces — routinely push prompts past 16K tokens. RULER 16K prefill on Qwen2.5-7B + A6000 takes ~2.7 s (our calibration), versus ~450 ms TTFT P95 for ShareGPT chat. The ~6× gap makes "burn the prefill" expensive: once it's computed, throwing it away costs seconds per request and seconds of GPU time that could have served someone else.

**Paragraph 2 — two scenarios that burn the prefill**:
We identify two distinct cases where the system has to discard prefill state and recompute. (a) **Engine failure mid-decode**: a long-running request lives on one GPU that disappears (hardware fault, OOM, scheduled drain, container kill, multi-tenant preemption). Long requests are disproportionately hit — they live longer, so disruption exposure scales with lifetime. (b) **Priority preemption mid-decode**: a tight-SLO request arrives, and the queue head's deadline is closer than the running request's. Existing schedulers refuse to preempt running long-context decode because reprefill cost dominates any priority benefit. Both scenarios share the same root cause — no cheap way to *resume* a partial decode after eviction. §2 expands on why current systems are stuck on each scenario.

**Paragraph 3 — insight in one sentence**:
A single host-side KV checkpoint, published asynchronously to `/dev/shm` at block-aligned cadence, can simultaneously serve as (i) the recovery substrate when an engine dies — the next engine reads the checkpoint and resumes decode without reprefilling — and (ii) the cheap-preempt substrate that lets the scheduler preempt running long-context decode without wasting prefill, because the request resumes from checkpoint after eviction. The same data, two uses, no duplication. Once preempt becomes cheap, scheduling can shift from prediction-heavy proactive admission (the current SLO-scheduler position) to measurement-heavy reactive preemption — when an underlying mechanism's cost drops below a threshold, design space that was previously off-limits opens up.

**Paragraph 4 — contributions, headline numbers, and roadmap**:
Three contributions, each tied to a verifiable artifact. (1) A host-side KV checkpoint mechanism with a two-tier store (in-process pinned-memory pool + `/dev/shm` mirror) that decouples cadence from access path. (2) A slack-based preempt picker that uses cheap preempt to make decisions impossible without it — preempting long-context decode for tighter-SLO waiting requests. (3) Cross-engine reroute via the same checkpoint, with a state machine that restores KV from shm and resumes decode on a surviving engine. Empirically, ours sustains higher tight-class SLO attainment past the saturation knee, recovers from single-engine kill in under 2 s on RULER 16K, and adds negligible overhead under healthy load. §2 details the gap in the existing design space; §3 develops the design; §4 evaluates the three claims; §5 concludes.

---

## Section 2: Motivation and Background

Detailed problem space and existing-systems comparison. Two subsections plus a comparison table; related work is folded in here via Table 1 + inline citations rather than living in its own section.

### §2.1 Long-context serving and the disruption problem (2 paragraphs)

**Para 2.1.1 — long-context serving regime**:
Pin down the regime. Long-context = 16K+ tokens in prompt. The workloads driving this are RAG with retrieved context (16K-128K), code agents with long traces, and document-grounded QA. Concrete cost on our setup: 16K prefill on Qwen2.5-7B on A6000 takes 2.7 s (P95 from a no-pressure calibration), versus ShareGPT chat TTFT P95 of 0.45 s. The ~6× gap is what makes "burn the prefill" expensive both for engine-failure recovery and for priority preempt.

**Para 2.1.2 — disruption is not rare, and SLO pressure is constant**:
Cite operational reasons engines disappear during a request's lifetime: scheduled drains for upgrades, preemption in spot or multi-tenant clusters, hardware faults (GPU ECC, network loss), and OOM from concurrent oversubscription. Long requests are disproportionately exposed because their lifetime is longer. Independently of disruption, modern serving systems also face constant SLO pressure: tight-SLO requests arriving while loose-SLO requests are mid-decode is the routine case under any non-uniform workload. The two scenarios from §1 — engine failure and priority preempt — share the same root cause (no cheap resume) and motivate the same mechanism response.

### §2.2 Existing systems and the empty quadrant (3 paragraphs + Table 1)

**Para 2.2.1 — three families of existing systems, each addressing one half of the problem**:
Three families of prior work each take a partial position in the design space. SLO schedulers (QLM, Scorpio, JITServe, Niyama) are proactive prediction-based — they admit conservatively, never preempt running long-context decode, and avoid the problem by limiting preempt to early-prefill chunks or short context. Migration-based fault tolerance (Llumnix) supports live KV migration between engines for load balancing and priority isolation, but the mechanism requires the source engine alive — when source dies suddenly, migration is impossible. Host-side KV pools (Mooncake for prefix cache reuse, TokenFlow for local preempt-and-resume) maintain durable KV on host memory, but neither has been demonstrated for cross-engine recovery.

**Para 2.2.2 — Table 1: feature comparison**:
Table 1 enumerates ten representative systems against four binary axes that matter for our problem: (i) can preempt long-context decode; (ii) survives source-engine death (disruption-aware); (iii) maintains host-side KV pool; (iv) cross-engine state transfer. The pattern: every prior system fills three of the four cells, but no system fills all four. The empty quadrant is `(cheap preempt of long-context decode) × (disruption-aware)` — the position we occupy. Each row of the table corresponds to one inline citation pointing the reader to the relevant prior work; Table 1 thus replaces an independent Related Work section.

**Para 2.2.3 — what changes when preempt is cheap**:
Spell out the consequence of cheap preempt for scheduling, because the consequence is itself a contribution. Without cheap preempt, schedulers must predict latencies (Scorpio's α/β/γ/δ admission gates, JITServe's QRF length predictor, Llumnix's headroom constant) and admit conservatively to avoid disturbing the running set. With cheap preempt, the scheduler observes slack drift at runtime and corrects reactively — preempt the laziest tail to admit the most urgent head. The dependency on offline-profiled latency models drops out. This shift — from prediction-heavy proactive to measurement-heavy reactive — is the policy-level contribution that our mechanism enables.

**Table 1: feature comparison of representative LLM serving and recovery systems**

| System | Pre-empt long-context decode | Disruption-aware | Host-side KV | Cross-engine state transfer |
|---|---|---|---|---|
| vLLM (FCFS) | ✗ | ✗ | ✗ | ✗ |
| QLM | ✗ (early chunks only) | ✗ | ✗ | ✗ |
| Scorpio | ✗ (TTFT/TPOT gates) | ✗ | ✗ | ✗ |
| JITServe | ✗ (admission control) | ✗ | ✗ | ✗ |
| Niyama | ✗ (deadline EDF) | ✗ | ✗ | ✗ |
| Llumnix | partial | partial (source must be alive) | ✗ | ✓ live migration |
| Mooncake | ✗ | ✗ | ✓ (prefix cache) | ✗ |
| TokenFlow | ✓ (same-engine resume) | ✗ | ✓ | ✗ |
| FastServe | ✓ (skip-join MLFQ) | ✗ | partial (host swap) | ✗ |
| **Ours** | **✓** | **✓** (source can be dead) | **✓** | **✓** (checkpoint mirror) |

---

## Section 3: Design

Mechanism-first walk. The section opens with an architecture overview, then walks through the host-side checkpoint (the mechanism), the slack picker (the policy that consumes it), and the cross-engine reroute (the second use of the same mechanism). It ends with an implementation note paragraph that is intentionally not its own §X header.

### §3.1 Architecture overview (1 paragraph + Figure 1)

Anchor on Figure 1: client → router process (CPU-only, separate from engine processes) → engine processes (one per GPU). Engines publish KV checkpoints to a host-side store with two tiers: an in-process pinned-memory pool, and a `/dev/shm` mirror visible to other engines. The router watches engine health via a shm status file and reroutes on death. Each engine runs a slack-based scheduler hook in its `schedule()` call. The same checkpoint data backs both same-engine preempt-and-resume and cross-engine recovery — this mechanism-policy coupling is the central design contribution.

### §3.2 Host-side KV checkpoint (3 paragraphs)

**Para 3.2.1 — what gets saved, when**:
Block-aligned cadence: every 16 tokens decoded (one full vLLM KV block), the engine saves that block's K/V tensors. The save is per-block delta — only newly-stable blocks since last save, not the full KV. Block alignment is mandatory because vLLM allocates KV in fixed blocks and a partial block has no well-defined boundary.

**Para 3.2.2 — two-tier store**:
Layer 1 is an in-process `KVCheckpointPool` of pinned host memory. Same-engine preempt-and-resume reads from Layer 1 directly, sub-millisecond. Layer 2 is a `/dev/shm` mirror written atomically via tmp + rename per chunk, plus a manifest and a "latest" pointer for atomicity. Cross-engine recovery reads Layer 2 and scatter-copies to the new engine's GPU. The data is identical in both tiers; only the access path differs.

**Para 3.2.3 — async publish pipeline and correctness invariants**:
A naive synchronous `/dev/shm` publish blocks the engine step ~5-15 ms per save and accumulates ~30% TTFT overhead. Reaching negligible overhead under healthy load requires more than a thread pool — the publish path is a four-stage pipeline that strips a different source of overhead at each stage: (i) on the engine side, `_save_checkpoints_if_needed` fires a `checkpoint_kv_blocks` RPC as a Future and advances `num_checkpointed_tokens` eagerly, with depth-1 backpressure (the next save waits only if the prior Future is still in flight); (ii) on the worker side, the K running requests' new blocks are batch-gathered into a single GPU-side concat and one async H2D copy, instead of K separate cudaMemcpyAsync (preserves the per-request delta-checkpoint semantics by remembering each request's slice); (iii) the chunk file write to `/dev/shm` uses a ctypes call into libc `write` so the syscall releases the GIL and does not steal the main thread; (iv) cross-engine `restore_kv_blocks` runs on a shared CUDA stream with a single batch-end `flush_pending_restore` instead of per-request synchronize, with dynamic GPU-memory estimation that falls back to per-request sync only when accumulated temp tensors would exceed the headroom outside the vLLM KV pool. Two correctness invariants pin down safety across all four stages: (a) the "latest" pointer only updates after the chunk file is flushed and renamed, so cross-engine restore never reads a partial chunk; (b) if the engine dies before a publish completes, the eagerly-advanced `num_checkpointed_tokens` may exceed what is actually in `/dev/shm` — the peer engine reads the older "latest" pointer and silently restores fewer tokens, and the scheduler's reload state machine accounts for the gap as decode-replay.

### §3.3 Slack-based preempt picker (3 paragraphs)

**Para 3.3.1 — slack as the single quantity**:
Define `slack(r, t)` = how much wall-clock time remains before request `r` misses its SLO. Pre-first-token requests use TTFT slack = `S_TTFT − elapsed_since_arrival`; post-first-token requests use TPOT slack = `S_TPOT − avg_per_token_so_far`. Slack collapses both SLOs into a comparable signal. Crucially, slack is *measured* from runtime, not predicted from an offline-profiled latency model — the shift from prediction-heavy proactive scheduling to measurement-heavy reactive (announced in §2.2.3) materializes here.

**Para 3.3.2 — picker rule and five guards**:
Every scheduler step, pick the running request with maximum slack (most relaxed) and the waiting request with minimum slack (most urgent). Preempt iff all five of these hold:
(i) `slack(victim) − slack(head) > δ + replay_cost(victim)`, where `replay_cost = switch_cost + (num_computed_tokens − num_checkpointed_tokens) × per_token_ms`. `switch_cost` is the fixed wall-clock of preempt + cross-engine forward + V3 reload + admit (default 1000 ms, calibrated to measured P95). `δ` is hysteresis (default 1 s);
(ii) the victim's host checkpoint manifest file exists on `/dev/shm`. The eager `num_checkpointed_tokens` counter is set when the save RPC is *fired*, but the manifest file appears only after the worker actually flushes bytes to host. Reading the file gives us the same ground truth the cross-engine reroute uses, eliminating fires that would silently fall back to full reprefill;
(iii) the waiting head is genuinely close to deadline, i.e. `slack(head) < r × S_TTFT(head)`, with `r` defaulting to 0.10 (Niyama-style absolute-deadline gate). Without this guard the picker fires whenever a victim happens to be much more relaxed than the head, even when the head is far from its SLO and the natural admit loop would have let it through in time;
(iv) at least one peer engine has spare capacity to receive the rerouted victim. The picker stats `/dev/shm/vllm_ft_engine_status/engine_<peer>.json` (the same heartbeat files the router consumes for dead-engine detection) and requires `alive` + fresh timestamp + `kv_usage < 0.85` + `running < 0.8 × max_num_seqs`. Without this guard, the picker fires blindly into an overloaded peer where the rerouted victim queues for tens of seconds (measured P95 on RULER 16K);
(v) `now − last_preempted_at(victim) > cooldown` (default 5 s), preventing the same victim from being preempted every step.
Each guard shuts down a distinct failure mode that we observed empirically before adding it: (i) trade thrash, (ii) silent fallback to reprefill, (iii) waste under healthy load, (iv) overshoot into a saturated peer, (v) repeat-victim thrash. All five thresholds are tunable; defaults are calibrated once per hardware and held fixed across our SLO sweeps.

**Para 3.3.3 — what cheap preempt enables, made concrete**:
Without cheap preempt, schedulers must rely on offline-profiled signals (Scorpio's α/β/γ/δ admission gates, JITServe's `v_token`, Llumnix's headroom constant). With cheap preempt, the scheduler observes slack drift at runtime and corrects reactively — preempting the laziest tail to make room for the most urgent head. The slack formula plays the same role those prior predicted signals played, but is computed from measurements rather than offline profile, removing a brittle dependency and the offline-profile pipeline that came with it.

### §3.4 Cross-engine reroute (3 paragraphs)

**Para 3.4.1 — engine death detection**:
Each engine writes a heartbeat file to `/dev/shm/vllm_ft_engine_status/engine_<id>.json` every 200 ms via a daemon thread (must be a thread, not piggybacked on `step()` — vLLM's busy loop blocks in `queue.get` when idle, so step-driven writes stop the moment the engine is idle, and the router would falsely declare a healthy engine dead). The router polls these files every 500 ms; an engine whose timestamp is older than 2 s is marked dead. Death triggers reroute of all in-flight requests on that engine.

**Para 3.4.2 — reroute decision and KV restore**:
The router maintains an `in_flight` dict mapping `router_req_id → (engine_id, body)`. On engine death, it enumerates that engine's in-flight requests and re-forwards each to a surviving engine, with `vllm_xargs.is_rerouted=True`, the original engine's `internal_req_id` (so the new engine finds the checkpoint), and the published `num_checkpointed_tokens`. The new engine's `add_request` sees `is_rerouted` and diverts the request into a reload state machine instead of normal prefill. The state machine allocates KV blocks, calls `restore_kv_blocks` on the worker (which reads from `/dev/shm/vllm_ft_checkpoints/<original_id>/`), and flips status to `PREEMPTED` once restore completes. vLLM's standard resumed-from-preempt admission path takes over from there — no admit-loop modification needed.

**Para 3.4.3 — design coupling, restated as the contribution**:
The same shm checkpoint data backs same-engine preempt-and-resume (consumed by the slack picker through `num_checkpointed_tokens`) and cross-engine reroute (consumed through `/dev/shm/vllm_ft_checkpoints/`). If we removed the checkpoint, both paths fall back to less-capable corners of the design space described in §2.2: the scheduler reverts to the QLM/Niyama position (no long-context preempt available), and reroute reverts to the Llumnix-source-dead position (full reprefill on the new engine). The mechanism-policy coupling — one new data path, two scheduling moves it unlocks — is the contribution.

### §3.5 Implementation (1 paragraph, no subsection header)

Built on vLLM v0.16. Scheduler hook (~80 LOC), engine core state machine + heartbeat (~200 LOC), worker RPCs for `checkpoint_kv_blocks` / `restore_kv_blocks` (~150 LOC), one new router process (~400 LOC), workload runner (~200 LOC). Three non-obvious engineering choices worth noting because each one tripped an early prototype: (i) `vllm_xargs.router_req_id` instead of `X-Request-Id` (vLLM's OpenAI handler mangles the latter with `cmpl-<id>-0` prefixes the router can't predict); (ii) heartbeat from a daemon thread, not piggybacked on `step()` (an idle engine never calls step, and step-driven heartbeats would falsely report dead); (iii) async publish via `ThreadPoolExecutor` with depth-1 future-based backpressure, not synchronous publish (saves ~30% TTFT under load) and not unbounded async (avoids the GIL-contention regression we measured in an earlier version that lost 10% goodput).

---

## Section 4: Evaluation

Opens with a single dense Setup paragraph (no separate Setup subsection), then three claim-aligned subsections corresponding to §1's three contributions, and finally a Limitations subsection at the end (in place of an independent Discussion section).

### Setup paragraph (1 paragraph)

Hardware: 2× NVIDIA A6000 PCIe (48 GB each); portability check on 2× L40S (same model, different SM). Model: Qwen2.5-7B-Instruct fp16, `max_model_len 32K`. Workloads: RULER 16K for long-context (the regime we target), ShareGPT for short-context (the regime we shouldn't break), both with Poisson arrivals and Niyama-style 3-tier QoS classes in the main figure. SLO calibration: baseline P95 measured on `vllm_fcfs` at uncontested low QPS (0.02 QPS for RULER, 0.1 QPS for ShareGPT); per-class SLO tiers set at {2×, 3×, 6×} baseline P95 for tight/normal/loose. Baselines: `vllm_fcfs` (router-less floor — captures the cost of having no router or FT at all), `reroute_no_ckpt` (our architecture without FT features — captures the cost of just having the router), `ours` (full system), `ours_no_picker` (picker ablation, same FT machinery but slack picker off). Three seeds per configuration, mean ± std reported. Before the pressure tests, we verify that the FT machinery does not impose meaningful tax under healthy low-QPS load (E_M3): ours adds ~30 ms TTFT P50 over `reroute_no_ckpt` and matches throughput within 0.5%; the router itself is essentially free (`vllm_fcfs ≈ reroute_no_ckpt`).

### §4.1 SLO attainment under load (3 paragraphs)

**Para 4.1.1 — main figure (E_M1)**:
Figure 2: x = QPS, y = SLO attainment %, three lines (`ours`, `reroute_no_ckpt`, `vllm_fcfs`). Sweep 1.0 → 8.0 QPS for ShareGPT (analogous range for RULER 16K). All systems start at 100% at low QPS; curves diverge as QPS increases; `ours` holds attainment higher and degrades later. Quote saturation-knee numbers at the QPS where baselines first drop below 90%: `ours = X%`, `reroute_no_ckpt = Y%`, `vllm_fcfs = Z%`. Long-context RULER shows the gap most clearly because each preempt-and-resume by `ours` saves seconds of prefill that the baselines must redo.

**Para 4.1.2 — tiered breakdown (E_M1-tiered)**:
Uniform SLO over uniform arrivals doesn't fully expose the picker (everyone is equally urgent). Figure 3 plots per-class attainment vs QPS under 3-tier QoS. `ours` protects the tight class — maintains ≥95% well past where baselines drop below 50% on tight — without breaking loose class (>90% across all systems and QPS). The picker's reallocation moves in the right direction: it prioritizes requests where the SLO is non-trivial to meet, exactly where the scheduling decision matters.

**Para 4.1.3 — SLO tightness sweep (E_M2)**:
Fix QPS at the knee, sweep the SLO multiplier from 2× to 6×. Figure 4: x = multiplier, y = `ours` attainment minus baseline attainment. Gap is largest at 2× (tight) and shrinks toward zero at 6× (loose). Conclusion: `ours` wins exactly where winning matters — when SLOs are non-trivial; under loose SLOs no scheduling decision changes the outcome and no system needs to be clever.

### §4.2 Mechanism vs policy ablation (1 paragraph)

**Para 4.2.1 — picker ablation (E_M4)**:
Add `ours_no_picker` to Figure 2 — same checkpoint mechanism, V3 reload state machine available, but slack picker disabled. Result: `ours_no_picker` lies between `ours` and `reroute_no_ckpt`. The mechanism alone gives passive benefit (cheaper resumes when capacity-driven preempts happen anyway), but the slack picker is what actively realizes the SLO advantage by *choosing* to preempt for tight-SLO admission. The two contributions are independent and additive: removing the checkpoint reverts to the Mooncake/TokenFlow corner (mechanism without cross-engine), and removing the picker reverts to the QLM/Niyama corner (policy without cheap preempt). Only the combination occupies the empty quadrant from §2.2.

### §4.3 Disruption recovery (2 paragraphs)

**Para 4.3.1 — single-engine kill demo (E_D1)**:
Figure 5: failover_gap (time from engine SIGKILL to next token of affected requests) for three systems × ShareGPT and RULER 16K. `ours`: ~1.5 s. `reroute_no_ckpt`: ~2.5 s at 4K, ~10-20 s at 16K (the full reprefill cost). `vllm_fcfs`: undefined — clients receive 5xx. The gap *grows with context length* — at 64K we extrapolate tens of seconds for the no-checkpoint baseline, exactly the long-context regime that motivates the paper.

**Para 4.3.2 — variance argument**:
`ours` has lower *mean* failover_gap and lower *variance*: all rerouted requests on a single dead engine recover in one V3 batch synchronously (sub-1 ms spread across requests). `reroute_no_ckpt` staggers recoveries through the admission path, exposing high tail. This is a second independent argument — even if `ours` mean were comparable to the baseline, the tighter tail alone would justify the design for any system that cares about P99.

### §4.4 Limitations and future work (1 paragraph)

We test 2 engines; chained failure or 3+ engines are out of scope and would require coordinating multiple `is_rerouted` requests across surviving peers. Workload-level recovery routes through a single router process — router HA is not addressed in this work. Cross-engine restore loses output tokens already produced on the dead engine because output token IDs are not shipped cross-engine (only KV is); for SLO accounting this is harmless (the new engine re-emits from a slightly earlier point) but the total work done is marginally larger than no-disruption. Future work: adaptive checkpoint cadence tied to slack (tight requests checkpoint more frequently), cross-engine slack coordination (peer engines yield to an incoming rerouted request), longer context via the Qwen2.5-7B-Instruct-1M variant. Admission control is orthogonal and could layer on top of our reactive preempt cleanly.

---

## Section 5: Conclusion

Cheap preempt unifies SLO scheduling and disruption recovery into one design. A single host-side KV checkpoint serves as both the substrate for cheap same-engine preempt and the substrate for cross-engine recovery, with a slack-based picker as the policy that converts the cheap-preempt capability into measurable SLO attainment gains. Empirically, the combination sustains higher tight-class SLO attainment past the saturation knee, recovers from single-engine kill in under 2 s on long-context workloads, and adds negligible overhead under healthy load. The broader implication: when an underlying mechanism's cost drops below a threshold, design space that was previously off-limits opens up — host-KV checkpoint crosses that threshold for long-context LLM serving, and we expect the same pattern to apply as model context windows continue to grow.

---

## Cross-paper structural reminders (from reference_paper.md)

Drafting checklist informed by TokenFlow + Niyama + Scorpio + Llumnix patterns, plus Capybara (APSys '24) as the venue-shape reference:

1. **Numerical pain point in intro**, not vague — quote ~2.7 s prefill, not "long prefill cost". TokenFlow, Niyama, Capybara all lead with concrete numbers in their first paragraph.
2. **Mechanism before policy** in design section (TokenFlow + Llumnix pattern) — describe checkpoint first, then how scheduler uses it.
3. **Eval order: end-to-end first, then mechanism micro, then ablation** (TokenFlow + Niyama). We invert only by tucking the healthy-overhead check into the Setup paragraph rather than its own subsection — it's a small de-risking claim that the reader should swallow before the headline results.
4. **Related-work folded into motivation via a comparison table** (Capybara Table 1 pattern) — group by what each prior system can or can't do across the axes that matter for our problem, with one row per cited system. This replaces an independent §Related Work section.
5. **Limitations as a subsection of evaluation**, not its own section (Medicine AIoTC '25 pattern, Capybara doesn't even have one). Keeps the structure to 5 sections.
6. **Formalize then relax** if there's an objective worth formalizing (TokenFlow proxy-objective pattern). Our slack formula is the candidate — formalize inline in §3.3.2, note that it's a heuristic proxy for "minimize expected SLO violations" without solving the LP.
7. **Workshop-length compression discipline** (Scorpio pattern) — 1 paragraph per mechanism; preserve math inline, not in separate algorithm blocks; single-paragraph subsections where the topic supports it.
8. **Mechanism vs scenario abstraction** (Llumnix's "virtual usage" unifies four scenarios) — our equivalent is "slack + replay_cost" as a single quantity that captures both stage-aware urgency and preempt eligibility. Surface this framing explicitly in §3.3.

---

## Open questions to resolve before submission

| Question | Resolution path |
|---|---|
| Should we report 32K or 16K results? | Current model caps at 32K. RULER 16K is the cleanest workload. Mention 32K extension via 1M-context model variant as future work in §4.4. |
| How aggressive should hysteresis δ and cooldown defaults be? | Run a small sweep (small slice of E_M2 budget) — δ ∈ {500 ms, 1 s, 2 s}, cooldown ∈ {3 s, 5 s, 10 s}. Pick the pair that maximizes the gap-with-baseline at the knee. Note inline in §3.3.2 that defaults are tuned, not first-principles. |
| Do we report tiered or uniform as the main figure? | Uniform runs first (already in progress); tiered shows the clearer story but uses synthetic QoS classes. Likely report tiered as Figure 2, uniform as supplementary. |
| What's our cited "failover_gap" — P50, P95, max? | Report P50 + P95 in §4.3.1 text, max in Figure 5. Mean misleads here because `reroute_no_ckpt` has high variance — the variance is itself one of the arguments. |
| How to honestly explain the residual 30 ms TTFT overhead under healthy load? | Per-step iteration over the running queue is honest, not a bug. Acknowledge in the Setup paragraph; do not try to optimize it away last-minute. |
| Table 1 row count — keep all 10 or trim? | Keep all 10 for first draft (matches Capybara's 11-row Table 1 in APSys '24 budget). Trim if §2 overflows; first to drop are FastServe (less directly comparable) and the duplicate among QLM/Scorpio/JITServe/Niyama (keep the two that most clearly span the proactive-scheduler space — likely Scorpio + Niyama). |
| Compress §3 or §4 if Figure 5 + tiered fig + ablation overflow? | First lever: drop §4.1.3 SLO-tightness sweep, fold into one sentence in §4.1.2. Second lever: collapse §4.3.2 variance argument into a half-sentence in §4.3.1. Third lever: collapse §3.4.3 design-coupling restatement into one sentence at end of §3.4.2. |
