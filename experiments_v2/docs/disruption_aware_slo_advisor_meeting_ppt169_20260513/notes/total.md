# 01_cover

This is the weekly update slot — same paper draft we've been talking about. Working title is *Cheap Preemption Enables Disruption-Aware SLO Scheduling for Long-Context LLM Serving*. Goal today: walk through the framing, the design, and the experiment plan, and get your read on two open questions at the end.

Key points: ① title + one-liner framing ② target = APSys 2026 workshop ③ today is structure + open questions, not a final-draft review
Duration: 1 minute

---

# 02_pain_point

[Transition] Let me start with why we keep coming back to the prefill-throwaway problem.
On our setup, RULER 16K prefill takes about 2.7 seconds, while ShareGPT chat TTFT P95 is 0.45 seconds — roughly 6× gap. So in the long-context regime, every time we discard a prefill and recompute, we burn seconds of GPU time per request — time that could have served someone else. Two distinct events keep forcing us to discard prefill, and they share the same root cause; that's the next slide.

Key points: ① concrete cost (2.7 s vs 0.45 s) ② ~6× gap drives all downstream design ③ throwaway is the wedge, not just "prefill is slow"
Duration: 2 minutes

---

# 03_two_scenarios

[Transition] Two scenarios — engine failure and priority preemption — share one root cause: there's no cheap way to resume a partial decode.
Engine failure: a long request lives on one GPU, that GPU disappears (hardware fault, OOM, drain, kill, multi-tenant preempt). Long requests are disproportionately exposed because their lifetime scales with disruption probability. Today's recovery is full reprefill on a surviving engine.
Priority preempt: a tight-SLO request arrives, deadline tighter than the running request's. Schedulers today refuse to preempt long-context decode because reprefill cost dominates priority benefit, so the tight request waits or misses. Same root cause: no cheap resume.

Key points: ① two scenarios, one root cause ② long requests overexposed in both ③ "no cheap resume" is the unifying mechanism gap
Duration: 2.5 minutes

---

# 04_empty_quadrant

[Transition] Mapped against four axes that matter for this problem, the existing-systems landscape leaves one quadrant empty.
Axes: can preempt long-context decode, disruption-aware, host-side KV pool, cross-engine state transfer. Ten representative systems each fill three of the four cells. None fills all four. The empty quadrant — cheap preempt of long-context decode crossed with disruption-aware — is the position we occupy. Each row in this table will become an inline citation, which is also how we replace an independent related-work section.

Key points: ① 4-axis comparison, 10 systems ② empty quadrant = our position ③ table replaces related-work section per APSys convention
Duration: 3 minutes

---

# 05_key_insight

[Transition] The whole design collapses to one sentence.
A single host-side KV checkpoint, async-published to /dev/shm at block-aligned cadence, simultaneously serves as the recovery substrate when an engine dies, and the cheap-preempt substrate that lets the scheduler preempt running long-context decode without wasting prefill. Same data, two uses, no duplication. The broader point — when a mechanism's cost drops below a threshold, design space that was previously off-limits opens up — is what we'd like to argue as the paper's framing contribution.

Key points: ① one checkpoint, two uses ② mechanism-policy coupling is the core design move ③ "cost threshold → new design space" is the framing story
Duration: 2 minutes

---

# 06_architecture

[Transition] Architecturally: client, router as a separate CPU-only process, and one engine process per GPU.
Engines publish KV checkpoints to a host-side store with two tiers — an in-process pinned-memory pool that backs sub-millisecond same-engine resume, and a /dev/shm mirror that other engines can read for cross-engine restore. Router watches engine health via a shm status file and reroutes in-flight requests when an engine goes silent. The key design fact: the same checkpoint data backs both same-engine preempt-and-resume and cross-engine recovery.

Key points: ① router separate from engines ② two-tier checkpoint store ③ one data path → two scheduling moves
Duration: 2 minutes

---

# 07_mechanism

[Transition] Mechanism details first, since policy depends on what the mechanism cheap-enables.
We save per-block deltas — every 16 tokens decoded is one vLLM KV block, and we checkpoint the new K/V tensors of that block. Block alignment is mandatory because vLLM allocates KV in fixed blocks. Naive synchronous /dev/shm publish costs about 30% TTFT under load; we drive that down to negligible with a four-stage async pipeline — Future + depth-1 backpressure on the engine side, GPU-side batch gather on the worker, libc-write via ctypes for GIL release on the shm side, and a shared CUDA stream with batched flush on the restore side. Two correctness invariants keep this safe: the "latest" pointer updates only after rename so readers never see a partial chunk, and the eager-counter gap is absorbed by the reload state machine as decode replay.

Key points: ① 4-stage pipeline → ~30% TTFT overhead reduced to ~30 ms ② invariants make crash-during-publish safe ③ same data backs both tiers
Duration: 3 minutes

---

# 08_slack_picker

[Transition] Policy on top of the mechanism: slack-based preempt picker.
Slack is defined as wall-clock remaining before a request misses its SLO — TTFT slack pre-first-token, TPOT slack post-first-token. One signal collapses both SLO regimes. Picker rule: every scheduler step, pick the running request with maximum slack as candidate victim, the waiting request with minimum slack as candidate head, and preempt iff slack(victim) − slack(head) > δ + replay_cost AND the victim has checkpointed tokens AND cooldown elapsed. Three guards each kill one failure mode — hysteresis δ kills thrashing, replay_cost kills bad trades, cooldown kills repeat-victim. The point relative to prior work: where Scorpio, JITServe, Llumnix rely on offline-profiled latency models, we observe slack drift at runtime and correct reactively — that's the prediction-to-measurement shift the paper argues for.

Key points: ① slack collapses TTFT + TPOT to one signal ② three guards, each kills one failure mode ③ measurement replaces offline profiling
Duration: 3 minutes

---

# 09_reroute

[Transition] Cross-engine reroute uses the same checkpoint data.
Detection: each engine writes a heartbeat file every 200 ms from a daemon thread — not piggybacked on step() because idle engines never call step. Router polls every 500 ms; timestamp older than 2 seconds means dead. Reroute: router enumerates in-flight requests on the dead engine and re-forwards each to a surviving engine, body carries is_rerouted=True plus the original internal_req_id plus the published num_checkpointed_tokens. Restore: new engine's add_request diverts the request into a reload state machine that allocates KV blocks, calls restore_kv_blocks against /dev/shm, and flips status to PREEMPTED. From there vLLM's standard resumed-from-preempt admit path takes over, so we don't touch the admit loop.

Key points: ① heartbeat from daemon thread (lesson from earlier dead-engine false positive) ② is_rerouted flag + original req_id wire it together ③ reuse vLLM's resumed-from-preempt path, no admit-loop changes
Duration: 2.5 minutes

---

# 10_eval_setup

[Transition] Setup, then claims, then placeholder figures.
Hardware is 2× A6000 PCIe with an L40S portability check. Model is Qwen2.5-7B-Instruct fp16, max_model_len 32K. Workloads are RULER 16K for the long-context regime we target and ShareGPT for the short-context regime we must not break, both with Poisson arrivals and Niyama-style 3-tier QoS in the main figure. SLOs are calibrated as multiples of baseline P95 on vllm_fcfs at uncontested low QPS. Four baselines isolate router cost, FT cost, and the slack picker contribution: vllm_fcfs as the no-router floor, reroute_no_ckpt as router-on but FT-off, ours_no_picker as FT-on with picker disabled, and ours as the full system. Three seeds per config.

Key points: ① 4 baselines isolate router / FT / picker contributions ② RULER 16K is the target regime, ShareGPT is the don't-break check ③ 3 seeds, mean ± std
Duration: 2 minutes

---

# 11_experiments

[Transition] Three experiments, one per contribution. Plots are placeholders for now.
E_M1 — main figure: SLO attainment versus QPS, three systems. Claim: ours holds attainment past the saturation knee where baselines drop below 90%. Long-context gap is widest because each preempt saves seconds of prefill the baselines redo. We have 1.0–8.0 QPS, 3 seeds on A6000; L40S replicate in progress.
E_M4 — picker ablation: same axes plus ours_no_picker. Claim: ours_no_picker lies between ours and reroute_no_ckpt — mechanism alone gives passive benefit, picker actively realizes the SLO advantage. Two contributions independent and additive. ours_no_picker runs scheduled this week.
E_D1 — failover gap: SIGKILL to next-token latency versus context length. Claim: ours about 1.5 s on RULER 16K, no_ckpt about 10–20 s at 16K, gap grows with context length, tens of seconds extrapolated at 64K. Demo works on RULER 16K; scaling to longer context next.

Key points: ① three experiments, one per contribution ② E_M1 mostly in, E_M4 + E_D1 in progress ③ failover gap grows with context length — wins exactly where we target
Duration: 3.5 minutes

---

# 12_submission_plan

[Transition] Plan and two open questions.
Target is APSys 2026 Workshop, 6–8 pages, deadline May 20 — next Wednesday. Status: paper framework and section outline drafted; E_M1 partially in; E_M4 ablation runs scheduled this week; E_D1 demo works on RULER 16K. Two open questions I want your read on: first, whether the headline figure should be the tiered QoS view or the uniform view — tiered tells a clearer story but uses synthetic classes; second, whether to fold a small δ / cooldown sweep into the E_M2 budget to justify defaults.

Key points: ① APSys deadline May 20 ② experiments are partial, ablation + failover still running ③ two open questions: headline figure choice, δ/cooldown sweep
Duration: 2 minutes
