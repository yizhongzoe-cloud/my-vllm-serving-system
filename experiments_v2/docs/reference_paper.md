# Reference papers — section-by-section paragraph summaries

Two papers used as writing templates for our APSys workshop submission on disruption-aware long-context LLM serving (host-side KV checkpoint + slack-based scheduling + cross-engine reroute):

- **TokenFlow (EuroSys 2026)** — closest prior art mechanically. Use it as the template for how to describe a co-designed scheduler + KV memory mechanism. Their flow: motivate buffer/QoS mismatch, formulate a tractable proxy objective, describe scheduler, describe memory layer, evaluate end-to-end then ablate.
- **Niyama (MSR, arXiv 2025)** — template for SLO-driven scheduling experimental structure. Their flow: define QoS classes and deadlines, build hybrid policy, evaluate goodput + SLO violations at uniform vs transient overload, ablate each component.

Reference these when drafting our intro, design, and evaluation sections.

---

## TokenFlow (EuroSys 2026)

### Section 1: Introduction
**Paragraph 1**: Anchors the paper in the rise of real-time LLM applications (chat, voice, agents) using GPT/LLaMA/Qwen as the named systems.
**Paragraph 2**: Frames the problem domain: token streaming has a producer/consumer rate gap because user reading speed is finite, which opens optimization room.
**Paragraph 3**: Names the core tension — TTFT vs sustained generation under burst — and accuses standard systems of being inflexible here.
**Paragraph 4**: Drops the one-sentence insight: exploit the buffer gap between generation and consumption to do just-in-time preemptive context switching.
**Paragraph 5**: Introduces TokenFlow as a co-design of (a) buffer-aware scheduler and (b) hierarchical KV cache manager.
**Paragraph 6**: States headline numbers — 82.5% effective throughput gain, 80.2% P99 TTFT reduction — across multiple GPUs and models.
**Paragraph 7**: Lists four contributions: QoS metric, scheduler, memory manager, end-to-end gains.

### Section 2: Background and Motivation

#### 2.1 LLM Inference and KV Cache
**Paragraph 1**: Recaps prefill/decode and the role of KV cache as standard background.

#### 2.2 Text-Streaming in LLM Serving
**Paragraph 1**: Defines text streaming and notes consumption rates are heterogeneous across users/modalities.
**Paragraph 2**: Cites empirical user-study numbers (≤1.3 s startup tolerance, ≤30% rate variation for fluency) to make QoS requirements concrete.
**Paragraph 3**: Names the fundamental tension between startup latency and steady-state rate under heterogeneous load.

#### 2.3 Resource Scheduling for LLM Text Streaming
**Paragraph 1**: Calls out FCFS as wrong default for interactive workloads.
**Paragraph 2**: Drops a micro-benchmark — TTFT >20 s during burst while active requests run at unnecessarily high 30 tok/s — to make the waste visible.
**Paragraph 3**: Diagnoses the cause: preemption is only reactive (for memory) not proactive (for QoS), and is not buffer-aware.
**Paragraph 4**: Notes that conventional metrics (TTFT, ITL, throughput) don't capture user-perceived experience.

#### 2.4 Hierarchical Memory Management for LLM
**Paragraph 1**: Attacks reactive memory management — only triggers at thresholds, causes I/O stalls, doesn't coordinate with scheduling.

### Section 3: Overview and Formulation

#### 3.1 TokenFlow Overview
**Paragraph 1**: Names the five components: Request Tracker, Buffer-aware Scheduler, Request Offload Manager, LLM Executor, Hierarchical KV Cache Manager.
**Paragraph 2**: Describes their interaction loop (waiting/active state machine, preemption-resumption controlled by scheduler, transfers by memory manager).

#### 3.2 Quality of Text-Streaming Service Metric
**Paragraph 1**: Argues for a richer QoS metric than throughput, combining token utility, startup, and rebuffering.
**Paragraph 2**: Establishes formal notation (TTFT, ITL, reading speeds, buffer-occupancy-dependent weights).
**Paragraph 3**: Writes the closed-form QoS expression: weighted tokens minus TTFT and rebuffering penalties.
**Paragraph 4**: Re-explains the formula verbally as three integrated user-experience factors.

#### 3.3 Scheduling Problem Formulation
**Paragraph 1**: Concedes QoS is intractable online and switches to a proxy: maximize expected effective tokens minus buffer-underflow penalty.
**Paragraph 2**: Defines per-request utility balancing token value against buffer depletion with a regularizer.
**Paragraph 3**: Identifies the two binding resources: GPU KV memory and batch-size-dependent compute trade-offs.
**Paragraph 4**: Elaborates the batch-size trade-off (I/O overhead, decode speed, preemption flexibility).
**Paragraph 5**: Writes the combinatorial optimization: pick request subset to maximize utility under batch and memory constraints.
**Paragraph 6**: Notes two corrections to the formulation — context-switch overhead and predicted buffer state under delays.

### Section 4: Buffer-Aware Request Scheduling

#### 4.1 A Motivating Example
**Paragraph 1**: Walks through a three-request toy scenario showing how buffer-dependent preemption avoids stalls.
**Paragraph 2**: Extracts the operational rule of thumb: keep buffers in an optimal range while accounting for recompute and I/O cost.

#### 4.2 Two-Step Scheduler Design
**Paragraph 1**: Names the two phases: (1) determine the working set, (2) balance buffers inside it.

##### 4.2.1 Determine the working set
**Paragraph 1**: Defines working set as the dynamic upper bound for overcommitment scheduling.
**Paragraph 2**: Gives the static formula (memory / per-request footprint) and dynamic scaling rule.
**Paragraph 3**: Describes the time-sliced trigger (fixed interval, only fires under stress or buffer-critical conditions).
**Paragraph 4**: States admission criteria — capacity available and enough remaining buffer relative to latency and margin.

##### 4.2.2 Buffer balancing inside the working set
**Paragraph 1**: Justifies overcommitment with transparent CPU offload, requiring preemptive scheduling on the GPU-CPU hierarchy.
**Paragraph 2**: Defines per-request priority using utility with exponential buffer decay and token-generation weighting.
**Paragraph 3**: Specifies the greedy + local-search algorithm: pick by priority under GPU memory, then evaluate adjacent swaps.

##### 4.2.3 Balance recompute vs load from CPU memory
**Paragraph 1**: Describes the runtime decision rule: compare I/O cost (queue + transfer) against recompute cost via sliding-window prefill latency.
**Paragraph 2**: Notes that batching recompute with prefill is tricky and proposes a dynamic partition policy.

#### 4.3 Schedulability Analysis
**Paragraph 1**: Imposes a capacity constraint that aggregate working-set generation rate ≤ system throughput, estimated online.
**Paragraph 2**: Describes the fallback: graceful degradation to FCFS plus memory-aware admission when the capacity bound is violated.

### Section 5: Hierarchical Memory Management
**Intro paragraph**: Positions hierarchical KV cache as required for concurrency beyond GPU memory, contrasting proactive vs reactive transfers.

#### 5.1 Write-Through Policy
**Paragraph 1**: Justifies write-through over write-back because preemption patterns are unpredictable, requiring continuous GPU-CPU sync.
**Paragraph 2**: Lists three advantages: no need to predict preemption, full PCIe utilization, incremental updates.

#### 5.2 Synchronous Chunked Writing
**Paragraph 1**: Argues synchronous writes (finishing within a compute iteration) beat async because they avoid scheduler stalls.
**Paragraph 2**: Walks through the mechanism: buffer generated KV, predict next-iteration time, pull a sized chunk, launch a tight write.
**Paragraph 3**: Lists benefits: no I/O wait, PCIe-optimized transfer size, priority-based write ordering.

#### 5.3 Load-Evict Overlap
**Paragraph 1**: Introduces concurrent preempt/resume: write-through lets preemption reclaim already-synced segments while loading runs.
**Paragraph 2**: Walks an example with overlapped eviction + chunked loading and dynamic buffer repartitioning.

### Section 6: Implementation
**Paragraph 1**: States the codebase footprint: ~4000 Python LOC on top of SGLang.
**Paragraph 2**: Describes the scheduler replacement and the request-tracking modules.
**Paragraph 3**: Details the KV manager using parallel CUDA streams plus Python multithreading (compute, load, evict streams).

### Section 7: Evaluation

#### 7.1 Experimental Setup
- **Hardware**: RTX 4090, A6000, H200; also Huawei Ascend 910B for portability test.
- **Models**: Llama3-8B, Qwen2-7B, Qwen2.5-32B.
- **Workloads**: ShareGPT (general), BurstGPT (burst-specific), production traces.
- **Baselines**: SGLang vanilla, SGLang + chunked prefill, Andes (QoE-aware).
- **Metrics**: TTFT, throughput, and *effective throughput* = timeliness-weighted tokens.

#### 7.2 End-to-end Evaluation in Real-World Traces
**Paragraph 1**: Reports headline numbers on A6000 — 52.6% mean TTFT reduction, 45.1% effective throughput gain.
**Paragraph 2**: Drops in a 20-min Qwen2.5-32B stress test showing lower queue depth and higher concurrency than baselines.

#### 7.3 Controlled Request Distribution Test
**Paragraph 1**: Describes the synthetic workload methodology (ShareGPT shape + controlled burst + Poisson) on RTX 4090 and H200.
**Paragraph 2**: Reports burst-scenario numbers: 80.2% lower P99 TTFT, 48.4% lower mean TTFT, 52.9% higher effective throughput.
**Paragraph 3**: Reports Poisson-scenario numbers: 82.5% effective throughput gain, 53.7% TTFT reduction on H200.

#### 7.4 Micro Experiments
**Paragraph 1**: Plots token-generation timelines comparing SGLang and TokenFlow to show earlier service start matched to consumption rate.
**Paragraph 2**: Visualizes preemption mechanism showing buffer thresholds trigger reallocation without latency disruption.
**Paragraph 3**: Tests a mixed-rate workload (40% @ 15 tok/s, 60% @ 20 tok/s) showing automatic rate-aware prioritization.
**Paragraph 4**: Sweeps generation speed 20–30 tok/s to confirm gains hold.
**Paragraph 5**: Ports the system to Ascend 910B as portability evidence.

#### 7.5 Hyperparameter Sensitivity
**Paragraph 1**: Sweeps reschedule interval; shorter is slightly better despite overhead.
**Paragraph 2**: Sweeps buffer conservativeness, showing the stability vs agility trade-off.

#### 7.6 Overhead Quantification and Ablation
**Paragraph 1**: Quantifies scheduling overhead growth: ~0.07 ms (SGLang) → ~0.4 ms (TokenFlow); tracker is negligible.
**Paragraph 2**: Runs an ablation isolating write-through and hierarchical offload as the dominant contributors.

### Section 8: Discussion
**Paragraph 1**: Contrasts TokenFlow's QoS with Andes' narrower QoE and emphasizes bidirectional scheduler-memory coordination.
**Paragraph 2**: Sketches multi-node extension via distributed cache + RDMA on top of local PCIe optimization.
**Paragraph 3**: Discusses how to handle non-human clients (need explicit output-rate specification or reference rate as priority).

### Section 9: Related Work
**Paragraph 1**: Surveys LLM serving systems (Orca, vLLM, SGLang, Sarathi-Serve) as throughput-focused, missing concurrency.
**Paragraph 2**: Surveys scheduling work from FCFS to preemptive, calling out Andes as closest but lacking buffer-awareness and memory coordination.
**Paragraph 3**: Surveys memory work (PagedAttention, hierarchical KV systems, compression) and slots TokenFlow's proactive write-through against them.

### Section 10: Conclusion
**Paragraph 1**: Restates TokenFlow as a co-designed system and re-cites the two headline numbers (82.5% / 80.2%).

---

## Niyama (MSR arXiv 2025)

### Section 1: Introduction
**Paragraph 1**: Establishes the diversity of LLM applications and their varying latency needs at scale.
**Paragraph 2**: Calls out today's siloed infrastructure (separate clusters per workload class) as the source of inefficiency.
**Paragraph 3**: Introduces Niyama as a co-scheduling system over fine-grained QoS classes.
**Paragraph 4**: Names the first mechanism — dynamic chunking that exploits deadline slack for throughput.
**Paragraph 5**: Names the second + third mechanisms — hybrid prioritization and eager relegation for overload control.
**Paragraph 6**: Lists three contributions.
**Paragraph 7**: Gives the paper's structural roadmap.

### Section 2: Background and Motivation

#### 2.1 LLM Inference
**Paragraph 1**: Recaps prefill/decode as the basis for understanding workload characteristics.
**Paragraph 2**: Explains chunked prefill scheduling as the current production baseline.
**Paragraph 3**: Defines TTFT and its role for interactive apps.
**Paragraph 4**: Defines TBT and its role for response fluidity.
**Paragraph 5**: Defines TTLT and its role for batch apps.
**Paragraph 6**: Notes which metric matters depends on application type.

#### 2.2 Production Deployment Landscape
**Paragraph 1**: Describes the siloed deployment model (separate clusters for interactive vs batch).
**Paragraph 2**: Discusses rate limiting as a simple but blunt overload tool.
**Paragraph 3**: Discusses short-request prioritization as another tool with fairness issues.

#### 2.3 Deployment Challenges
**Paragraph 1**: Argues that silos waste resources under fluctuating workloads.
**Paragraph 2**: Argues operational complexity grows with each additional QoS class.
**Paragraph 3**: Argues existing mechanisms lack graceful degradation.

#### 2.4 Analysis of Multi-SLA Scheduling Policies for LLM Inference
**Paragraph 1**: Sets up an evaluation of four classical policies (FCFS, SJF, SRPF, EDF).
**Paragraph 2**: Reports that none of the classical policies handle LLM workloads well across load regimes.
**Paragraph 3**: Pitches Niyama as exploiting LLM-specific structure to fix this.

### Section 3: Niyama Design and Implementation

#### 3.1 Overview
**Paragraph 1**: Names the three-queue architecture: prefill, decode, relegated.
**Paragraph 2**: Describes iterative batch construction with hybrid prioritization and deadline-violation checks.
**Paragraph 3**: Walks through the prediction → batch construction → execution loop.

#### 3.2 QoS Classes and Deadlines
**Paragraph 1**: Defines two QoS classes (interactive and non-interactive) with their target metrics.
**Paragraph 2**: Notes per-class flexibility — custom SLO targets allowed.
**Paragraph 3**: Writes the deadline formula for interactive first-token.
**Paragraph 4**: Writes the deadline formula for interactive subsequent tokens.
**Paragraph 5**: Writes the deadline formula for non-interactive completion.
**Paragraph 6**: States the objective: minimize violations while maximizing throughput.

#### 3.3 Dynamic Chunking
**Paragraph 1**: Frames the throughput-latency trade-off inherent to chunk size.
**Paragraph 2**: Argues smallest-chunk-for-all is suboptimal — leaves throughput on the table.
**Paragraph 3**: Introduces dynamic chunking that opportunistically grows chunks when there's deadline slack.

#### 3.4 Niyama Scheduling
**Paragraph 1**: Motivates a hybrid policy by attacking pure EDF and pure SRPF.
**Paragraph 2**: Writes the priority formula for interactive requests (deadline + work).
**Paragraph 3**: Writes the priority formula for non-interactive requests.
**Paragraph 4**: Addresses unknown decode length using historical mean + std-dev approximation.
**Paragraph 5**: Introduces eager relegation: proactively deprioritize requests predicted to miss their deadlines.
**Paragraph 6**: Adds application hints for preferential relegation in multi-tenant cases.
**Paragraph 7**: Adds selective preemption constraints to bound KV-cache thrash.

#### 3.5 An Illustrative Example
**Paragraph 1**: Walks a five-request example to show how dynamic chunking + hybrid prioritization improve throughput.

#### 3.6 Implementation
**Paragraph 1**: Implementation built on Sarathi scheduler atop vLLM.
**Paragraph 2**: Extends API for QoS class and priority specification.
**Paragraph 3**: Trains a random-forest predictor to drive dynamic chunk sizing.
**Paragraph 4**: Implements hybrid prioritization via a priority queue.
**Paragraph 5**: Tracks history for decode-length estimation and supports multi-tenant priority.

### Section 4: Evaluation

#### Setup (extracted across the evaluation section)
- **Hardware**: A100 single GPU (Llama3-8B), 2× A100 with TP (Qwen-7B).
- **Models**: Llama3-8B and Qwen-7B.
- **Workloads**: ShareGPT, Azure Conversation traces, Azure Code traces. Three QoS tiers — interactive (6 s TTFT, 50 ms TBT) and two non-interactive (600 s, 1800 s TTLT) — split with equal traffic.
- **Baselines**: Sarathi-Silo (siloed SOTA), Sarathi-FCFS, Sarathi-EDF, Sarathi-SRPF.
- **Metrics**: serving capacity (GPUs needed for fixed QPS), goodput (requests/sec meeting p99 SLO), SLO violation rate, p99 latencies.

#### 4.1 Capacity Evaluation at Uniform Load

##### 4.1.1 Cost Efficiency from Co-located Scheduling
**Paragraph 1**: Sets up the methodology — count GPUs needed to serve 50 QPS across three QoS classes.
**Paragraph 2**: Reports that Niyama needs 13–32% fewer GPUs than siloed baselines.
**Paragraph 3**: Attributes the savings to dynamic chunking + deadline-slack exploitation.

##### 4.1.2 Goodput
**Paragraph 1**: Defines goodput as requests/sec meeting p99 latency targets.
**Paragraph 2**: Reports 1.5–2.4× higher goodput than Sarathi-FCFS and 20–40% over Sarathi-EDF.
**Paragraph 3**: Attributes gains to the three-mechanism combination (chunking + hybrid prio + eager relegation).

#### 4.2 Latency and SLO Violations under Overload
**Paragraph 1**: Lists the evaluation axes (three latency parameters + SLO violation metrics).
**Paragraph 2**: Shows FCFS breaks down under load because it's deadline-unaware.
**Paragraph 3**: Shows EDF holds up better than FCFS but degrades at high load.
**Paragraph 4**: Shows SRPF keeps median latency low at the cost of long-request fairness.
**Paragraph 5**: Reports Niyama handles up to 40% higher load while keeping tail-latency SLOs.
**Paragraph 6**: Reports overall SLO violation rates: Niyama maintains zero violations at 30% higher load than the best baseline.
**Paragraph 7**: Slices violations by request length: FCFS and EDF treat short and long the same.
**Paragraph 8**: Shows SRPF unfairly punishes long requests even at low load.
**Paragraph 9**: Frames Niyama's behavior as fair at normal load, gracefully degrading at overload.
**Paragraph 10**: Slices violations by QoS tier: FCFS hits strict tiers first.
**Paragraph 11**: Shows EDF spreads violations evenly across tiers while SRPF ignores long jobs.
**Paragraph 12**: Reports Niyama has fewest total violations while staying deadline-aware.

#### 4.3 Latency and SLO Violations under Transient Overload
**Paragraph 1**: Describes the diurnal load pattern (2.0–6 QPS oscillation) for transient evaluation.
**Paragraph 2**: Sets up the multi-priority experiment with 20% low-priority within each tier.
**Paragraph 3**: Reports Niyama misses zero deadlines for important tasks under the diurnal load.
**Paragraph 4**: Attributes this to application hints driving selective relegation.
**Paragraph 5**: Shows rolling p99 latency plots: Niyama absorbs spikes while baselines collapse.
**Paragraph 6**: Reports Niyama meets p99 SLO for 100% of important requests and 92% of all requests.
**Paragraph 7**: Frames Niyama's behavior as graceful degradation preventing cascade.

#### 4.4 Ablation Studies
**Paragraph 1**: Sets up the three-knob ablation (dynamic chunking, hybrid prio, eager relegation).
**Paragraph 2**: Reports dynamic chunking gives 20% throughput, eager relegation adds 9%.
**Paragraph 3**: Notes hybrid prioritization looks marginal at optimal load but matters under overload.
**Paragraph 4**: Sweeps the α parameter to show the median-latency vs fairness trade-off.

### Section 5: Related Work

#### LLM Inference Optimization
**Paragraph 1**: Cites Orca, vLLM, Sarathi as foundational.
**Paragraph 2**: Critiques them for throughput-only focus, no QoS heterogeneity.
**Paragraph 3**: Positions Niyama as building on top of these.

#### Multi-Tenant Serving Systems
**Paragraph 1**: Cites Clockwork and INFaaS.
**Paragraph 2**: Critiques them for not co-scheduling across QoS classes.

#### QoS-Aware Scheduling
**Paragraph 1**: Cites EDF and rate-monotonic scheduling as theoretical foundation.
**Paragraph 2**: Cites Delimitrou & Kozyrakis on cloud QoS-aware resource management.
**Paragraph 3**: Positions Niyama as applying these to LLM serving as a new domain.

#### Graceful Service Degradation
**Paragraph 1**: Cites load balancing, quotas, throttling as classical overload tools.
**Paragraph 2**: Cites Bouncer on admission control for low-latency systems.
**Paragraph 3**: Claims Niyama is first to do graceful degradation specifically for LLM inference.

### Section 6: Conclusion
**Paragraph 1**: Restates the three mechanisms (chunking, hybrid prio, eager relegation).
**Paragraph 2**: Restates the headline evaluation numbers.
**Paragraph 3**: Forward-looks: QoS-aware scheduling becomes necessary as LLM deployments scale.

---

## Scorpio (arXiv 2505.23022, 2025, workshop-style preprint)

### Section 1: Introduction
**Paragraph 1**: Anchors the work in production LLM applications (Copilot, deep search, agents) and lists vLLM/SGLang plus continuous batching, PagedAttention, and chunked prefill as the prevailing toolchain.
**Paragraph 2**: Pivots from throughput to SLO by stating that existing schedulers admit greedily without TTFT/TPOT awareness, and that requirements differ across applications (code vs chatbot).
**Paragraph 3**: Declares scope — design an SLO-oriented serving system for heterogeneous SLOs — and states the key insight that SLO heterogeneity itself is an exploitable scheduling signal.
**Paragraph 4**: Pre-names the mechanisms (TPOT Guard with TRP/VBS/credit-batching, TTFT Guard with LDF + reject, predictor with seq-length model + two analytical models) without yet describing how they work.
**Paragraph 5 (contribution bullets)**: Lists five contributions in a numbered list: gap identification, TPOT Guard, TTFT Guard, predictive module, and integrated Scorpio system with up to 14.4× goodput and 46.5% SLO-adherence gains.

### Section 2: Background and Problem Formulation
**Paragraph 1**: Recaps prefill/decode and names TTFT and TPOT as the two canonical SLO metrics.
**Paragraph 2**: Borrows SLO precedent from cloud and edge computing to motivate why MaaS providers face heterogeneous-SLO scheduling.
**Paragraph 3**: Formalizes the problem: a request stream R under policy π, with per-request TTFT/TPOT thresholds defining R_good, and goodput / adherence-rate as the two objectives.
**Paragraph 4**: Writes the formal optimization — maximize expected goodput and expected adherence as T→∞.

### Section 3: Method
**Paragraph 1 (3.1 System Overview)**: Walks the request lifecycle through the three components — predictor estimates length/TPOT/TTFT, TTFT Guard reorders and rejects, TPOT Guard does VBS admission then credit-based batching, execution engine runs the chosen batch.

#### 3.2 Predictor
**Paragraph 1**: Describes the sequence-length predictor as a fine-tuned OPT-125M text classifier over discretized length bins.
**Paragraph 2**: Critiques prior bucketing (10 bins, max/10 width) for class imbalance and resolution loss, and argues for ~100 bins as a sweet spot.
**Paragraph 3 (TPOT Estimator)**: Derives an analytical ITL model linear in batch size × avg sequence length with four fitted coefficients, calibrated to R² > 0.9.
**Paragraph 4**: Extends the model to predict batch-level TPOT over the next P(r) steps under a conservative continuation assumption plus an inefficiency coefficient ε.
**Paragraph 5 (TTFT Estimator)**: Fits prefill time as a piecewise function of prompt length (constant below θ, affine above) and bounds the TTFT of a queued request by the cumulative prefill of its predecessors.

#### 3.3 Heterogeneous TTFT Guard
**Paragraph 1**: Justifies Least-Deadline-First reordering as a simple but effective TTFT scheduler.
**Paragraph 2**: Adds an unattainable-SLO rejection rule and explicitly defers smarter alternatives (lower-priority lane, cross-node migration) to future work — a self-imposed scope limit.

#### 3.4 Heterogeneous TPOT Guard
**Paragraph 1 (Key Insight)**: Diagnoses two failure modes of indiscriminate batching (cross-class interference and overload cascade) and pre-announces the two corresponding fixes (credit batching, VBS admission).
**Paragraph 2 (Credit-based Batching)**: Defines TRP as the ratio of the strictest TPOT in the batch to a request's own TPOT, then describes credit earning / batch selection / credit debit each iteration.
**Paragraph 3 (VBS-based Admission Control)**: Replaces raw batch count with the sum of TRPs as the effective load proxy and admits a new request only if EstimatedTPOT(VBS) does not exceed the strictest in-batch TPOT.
**Paragraph 4 (Algorithm 1)**: Presents the full TPOT-guarantee pseudocode binding the admission-control loop with the credit loop.

### Section 4: Experiments

#### 4.1 Experimental Setup
- **Hardware**: 4× NVIDIA A100 80GB on one server, NVLink pairs (GPU0-1, GPU2-3), PCIe across pairs.
- **Models**: Llama-3.1 8B (single GPU) and Gemma-2 27B (4-GPU TP), FP16/BF16.
- **Workloads**: ShareGPT and LMSYS-Chat-1M; six SLO categories spanning {tight, loose} × {TTFT, TPOT} mappings to code/tool-call/chatbot/summarization personas.
- **Baselines**: vLLM (throughput-oriented), S³ (SJF on predicted length, re-implemented inside vLLM), Mooncake (early-rejection admission, integrated into vLLM).
- **Metrics**: system goodput (SLO-met requests per second) and SLO adherence rate.

#### 4.2 QPS-Scaling SLO Attainment
**Paragraph 1**: Reports headline numbers — at QPS 15, Scorpio yields 8.8–14.4× goodput and 40.7–46.5% higher SLO adherence than baselines, and attributes baseline weaknesses (vLLM greedy, S³ length-ranked, Mooncake too-strict reject).
**Paragraph 2**: Concedes a low-QPS regression (vLLM beats Scorpio by 1.08× on Gemma2-27B/ShareGPT at QPS 5) and blames predictor-server resource contention, suggesting separate-GPU deployment or low-load fallback as fixes.

#### 4.3 Real-World Trace Serving
**Paragraph 1**: Switches to 20-minute Azure traces with bursty/light interleaving and reports cumulative SLO-met counts where Scorpio leads Mooncake/vLLM/S³ by 1.25/2.01/2.11× respectively, with prose attribution of where each baseline breaks.

#### 4.4 Effectiveness Analysis
**Paragraph 1 (Ablation Study)**: Adds TPOT Guard and TTFT Guard incrementally at QPS 14 and shows that each alone solves only its own SLO class while creating violations on the other — framed as evidence of mechanism interdependence.
**Paragraph 2 (Overhead)**: Reports Scorpio's scheduling overhead at under 0.2% of overall serving time across all four (model, dataset) settings, with predictor cost folded in via prior-work measurements.

### Section 5: Related Work
**Paragraph 1**: Compact one-paragraph survey grouping prior work into (a) throughput-focused engines (Orca, vLLM, Sarathi-Serve, SplitWise, DistServe), (b) fairness scheduling, (c) length-prediction schedulers (SSJF, SRTF), and (d) SLO-adherence systems (Mooncake, QM, AdaServe, SLOs-Serve) — then claims Scorpio's distinctive position as heterogeneous-SLO fine-grained scheduling.

### Section 6: Conclusion
**Paragraph 1**: Re-states Scorpio as TPOT Guard + TTFT Guard + predictor co-design and re-cites the 14.4× / 46.5% headline numbers.

---

## Llumnix (OSDI 2024)

### Section 1: Introduction
**Paragraph 1**: Frames LLMs (GPT family) as the next inflection point in generative AI.
**Paragraph 2**: Defines the serving topology — scheduler dispatches requests to model instances on a GPU cluster, with continuous batching inside each instance.
**Paragraph 3**: Names the first unique LLM characteristic — workload heterogeneity (input lengths, output lengths, latency expectations) driven by application diversity.
**Paragraph 4**: Names the second characteristic — execution unpredictability (output length unknown, KV memory grows dynamically).
**Paragraph 5**: Bridges the two characteristics into the framing "LLMs are like multi-tenant OSes, not stateless DNNs," and accuses prior serving systems of treating cross-instance scheduling with legacy DNN-era policies.
**Paragraph 6 (Isolation)**: Calls out memory contention causing inter-request interference and preemptions as a concrete pain point.
**Paragraph 7 (Fragmentation)**: Calls out external memory fragmentation across instances as a queuing-delay cause for long-input requests.
**Paragraph 8 (Priorities)**: Calls out the inability of existing systems to differentiate request priorities despite commercial tiers (ChatGPT Plus).
**Paragraph 9**: Introduces Llumnix as runtime rescheduling across instances, drawing the explicit analogy to CPU context switching across cores.
**Paragraph 10**: Enumerates the four rescheduling scenarios — load balancing, de-fragmentation, prioritization, auto-scaling — pre-mapped to Figure 1.
**Paragraph 11**: Names the technical centerpiece: a live-migration mechanism with near-zero, sequence-length-independent downtime by pipelining KV copy with computation.
**Paragraph 12**: Names the policy centerpiece: a distributed (global + per-instance llumlet) architecture plus the "virtual usage" abstraction that collapses all four scenarios into one load-balancing policy.
**Paragraph 13**: States implementation (vLLM backend, 16-GPU cluster) and headline numbers — up to 15× P99 first-token latency and 2× P99 per-token over INFaaS, 1.5× speedup for high-priority requests, 36% cost savings at parity tail latency.
**Paragraph 14 (contribution bullets)**: Four contributions — characterize LLM scheduling challenges, propose rescheduling + migration, design distributed scheduler + virtual usage, implement and evaluate.

### Section 2: Background
**Paragraph 1**: Recaps LLM application diversity and the implication of longer context windows.
**Paragraph 2**: Recaps autoregressive prefill/decode and why both phases are user-perceivable.
**Paragraph 3**: Recaps KV cache mechanics as the source of GPU-memory growth.
**Paragraph 4**: Recaps continuous batching plus PagedAttention dynamic memory allocation as the production baseline, with a worked example of queuing and preemption.

### Section 3: Motivation
**Paragraph 1 (Unpredictable memory demands and preemptions)**: Drops a measurement — at 62% average memory load on a single A10 LLaMA-7B, 8% of requests get preempted; P99 per-token latency is 3.8× P50 and preemption accounts for 70% of P99 latency.
**Paragraph 2 (Performance interference among requests)**: Shows decode-step latency vs batch size × sequence length, up to 2.6× spread at fixed sequence length — used to argue spread-out scheduling.
**Paragraph 3 (Memory fragmentation)**: Reports a 4-instance experiment where total free memory could satisfy queued requests but per-instance fragmentation forces queuing — motivates de-fragmentation as a distinct goal.
**Paragraph 4 (Priorities)**: Calls out applications with differentiated latency sensitivity (ChatGPT Plus) and indicts existing systems for treating all requests equally.
**Paragraph 5 (Opportunity)**: Pre-announces that cross-instance request rescheduling is the missing degree of freedom.

### Section 4: Llumnix Design

#### 4.1 Overview
**Paragraph 1**: Restates that Llumnix builds on continuous batching and PagedAttention but adds rescheduling.
**Paragraph 2**: Walks the four rescheduling goals (load balancing, de-fragmentation, prioritization, auto-scaling) and previews their tradeoffs.
**Paragraph 3**: Names the challenge — KV cache copy/recompute can cost 50× of a decode step — and the answer: live migration with pipelined KV copy.
**Paragraph 4**: Names the architecture answer — distributed global + per-instance scheduling — and the policy answer — virtual usage abstraction.

#### 4.2 Live Migration of LLM Requests
**Paragraph 1**: Exploits the append-only nature of the KV cache as the key property that makes pipelined copy safe.
**Paragraph 2**: Walks the multi-stage migration mechanism — stage k copies the KV blocks generated by stage k-1 while decoding continues, with a tiny final stop-the-world stage.
**Paragraph 3**: Cites VM live migration as the conceptual ancestor but flags KV-specific challenges — running out of memory mid-migration, request completion mid-migration.
**Paragraph 4**: Describes the source/destination handshake (pre-allocate / ACK / abort / commit) that guarantees correctness under continuous batching.

#### 4.3 Distributed Scheduling Architecture
**Paragraph 1**: Argues that the scheduling pressure of continuous rescheduling exceeds what a centralized scheduler can sustain.
**Paragraph 2**: Names the global-scheduler + llumlets split and the narrow interface — global only sees instance-level loads, not individual request state.
**Paragraph 3**: Describes the global scheduler's responsibilities — dispatch, pair source/destination for migration, control auto-scaling.
**Paragraph 4**: Describes the llumlet — local scheduler reports memory load (sum of virtual usages), picks which requests to migrate, and a migration coordinator drives the copy.

#### 4.4 Dynamic Scheduling Policy
**Paragraph 1 (Goals)**: Lists three policy goals — latency improvement (load balancing + de-frag), load-adaptivity (cost-aware auto-scaling), and request priorities (new for LLMs).
**Paragraph 2 (Virtual Usage)**: Introduces the unifying abstraction — assign a synthetic load to a request so that one load-balancing policy generates all four behaviors.
**Paragraph 3 (Queuing requests)**: For head-of-line queued requests, set virtual usage to demand so the instance looks overloaded and load balancing triggers de-fragmentation.
**Paragraph 4 (Execution priorities)**: For high-priority requests, add a headroom to physical usage so other normal requests get migrated away to create isolation.
**Paragraph 5 (Auto-scaling)**: For draining instances, inject a fake request with infinite virtual usage; for new instances, the normal load balancer saturates them automatically.
**Paragraph 6 (Policies — Dispatching)**: New requests go to the freest instance, where freeness F = (M − ΣV)/B captures both space and consumption rate; allows negative freeness as an automatic overload signal.
**Paragraph 7 (Policies — Migration)**: Trigger periodically; pair instances above/below freeness thresholds; prefer migrating short-sequence low-priority requests.
**Paragraph 8 (Policies — Auto-scaling)**: Maintain average freeness in [x, y], add when below x, terminate the instance with fewest requests when above y.

### Section 5: Implementation
**Paragraph 1**: 3,300 lines of Python; standalone library on top of vLLM; designed for backend portability.
**Paragraph 2 (Multi-instance serving)**: Uses Ray actors for all components plus an OpenAI-style API frontend that survives migration.
**Paragraph 3 (KV cache transfer)**: Uses Gloo Send/Recv (not NCCL because concurrent NCCL is unsafe) plus a separate CUDA stream to overlap with inference.
**Paragraph 4 (Block fusion)**: Coalesces many small PagedAttention blocks into one contiguous CPU buffer before sending, to amortize transport overhead.
**Paragraph 5 (Fault tolerance)**: Documents fallback to direct dispatch when the global scheduler dies, and Ray-based restart for failed instances and llumlets.

### Section 6: Evaluation

#### 6.1 Experimental Setup
- **Hardware**: 16-GPU cluster, 4× ecs.gn7i-c32g1.32xlarge Alibaba Cloud VMs each with 4× NVIDIA A10 (24 GB) over PCIe 4.0, 64 Gb/s network.
- **Models**: LLaMA-7B (single GPU) and LLaMA-30B (4-GPU TP), FP16.
- **Workloads**: synthesized Poisson and Gamma arrival traces; real ShareGPT-GPT4 and BurstGPT length distributions; plus generated power-law length distributions in five combinations (S-S, M-M, L-L, S-L, L-S).
- **Baselines**: round-robin, INFaaS++ (their improved INFaaS with GPU-memory-load tracking and load-aware auto-scaling), and Llumnix-base (priority-agnostic Llumnix).
- **Metrics**: end-to-end, prefill, and decode request latencies (mean and P99), preemption loss, cost (avg instances), scheduling stall.

#### 6.2 Migration Efficiency
**Paragraph 1**: Reports that migration downtime is nearly constant in sequence length (~20-30 ms) and up to 111× shorter than recompute/blocking-copy baselines.
**Paragraph 2**: Reports overhead on co-running batches at under 1% and notes that migration is only active for ~10% of the time anyway.

#### 6.3 Serving Performance
**Paragraph 1 (Real datasets)**: On ShareGPT/BurstGPT, Llumnix beats round-robin by up to 26.6× mean prefill and INFaaS++ by up to 5.5× P99 prefill and 1.3× P99 decode, with prose attribution to load balancing and migration.
**Paragraph 2 (Generated distributions)**: Sweeps the five length combinations and reports up to 7.7×/14.8× mean/P99 prefill gains and 2× P99 decode gain over INFaaS++, with 70.4% average preemption-loss reduction.

#### 6.4 Support for Priorities
**Paragraph 1**: Marks 10% of requests as high-priority and reports 1.2–1.5× mean request latency speedup and 3.6–10× P99 prefill speedup over Llumnix-base, with normal-request latency degradation bounded under 13%.

#### 6.5 Auto-scaling
**Paragraph 1**: Sweeps Poisson request rates and Gamma CVs to show Llumnix matches INFaaS++ tail latency at 16–18% lower instance count.
**Paragraph 2**: Sweeps the scaling threshold and shows Llumnix hits the same 5 s P99 prefill at 36% fewer instances than INFaaS++.

#### 6.6 Scheduling Scalability
**Paragraph 1**: Runs a 64-instance stress test with a sleep-based execution stub; centralized scheduling causes 40 ms / iteration stalls (1.7× slowdown), Llumnix's distributed scheduler stays near zero.

### Section 7: Related Work
**Paragraph 1 (LLM inference)**: Groups single-instance engines (FasterTransformer, vLLM, Orca, SpotServe, FastServe, AlpaServe) as complementary, then positions Llumnix as the missing multi-instance layer.
**Paragraph 2 (Request scheduling)**: Surveys DNN-era schedulers (Clipper, Nexus, DVABatch, Clockwork, Reef, Shepherd, AlpaServe, DeepSpeed-MII, INFaaS) and accuses them of either targeting one-shot stateless DNN inference or, in DeepSpeed-MII's case, falling back to round-robin.
**Paragraph 3 (Isolation vs fragmentation)**: Cites the classical packing-vs-spreading tradeoff in datacenter scheduling (Amaral, Gandiva) and re-frames Llumnix as the LLM-serving instantiation enabled by migration.
**Paragraph 4 (Migration)**: Contrasts Gandiva's checkpoint-resume DL-training migration with Llumnix's pipelined live migration, and cites VM live migration as the more direct inspiration.

### Section 8: Conclusion
**Paragraph 1**: Closes on the "LLM as Unix" framing — universality + multi-tenancy + dynamism — and positions Llumnix's rescheduling as the LLM analogue of OS context switching.

---

## Cross-paper observations: structural patterns to borrow

- **Both motivate with concrete numerical pain points before any design**. TokenFlow shows a micro-benchmark with TTFT >20 s while running requests hit 30 tok/s; Niyama shows the four classical policies failing across load regimes. For our paper: before introducing checkpoint+slack+reroute, we need one figure that makes the disruption pain quantitatively obvious (e.g., failover_gap p95 under preemption thrash on RULER 64K).
- **Both formalize an objective and then admit it's intractable, then introduce a tractable proxy**. TokenFlow defines QoS (with token utility + TTFT + rebuffering) and replaces it with utility-minus-buffer-penalty for online scheduling. Niyama defines per-class deadlines and then defines priority formulas that approximate the deadline objective. For us: state the slack-aware objective formally, then describe the heuristic priority rule we actually run.
- **Both organize design around a numbered list of named mechanisms**. TokenFlow: working set + buffer balancing + load/recompute trade-off + write-through + chunked writing + load/evict overlap. Niyama: dynamic chunking + hybrid prio + eager relegation. The paper structure mirrors the mechanism list one-to-one. For us: name three mechanisms (host-side KV checkpoint, slack-aware scheduling, cross-engine reroute) and give each its own subsection.
- **Both put end-to-end results before microbenchmarks before ablation**. The order is: setup → real-world / uniform-load end-to-end → controlled / overload scenarios → micro experiments → sensitivity → ablation. The ablation isolates which mechanism contributes how much. For us: lead with failover_gap p95 on RULER, then controlled-burst experiments, then per-mechanism ablation (checkpoint-only / slack-only / reroute-only).
- **Both keep related work compact and contrastive, not encyclopedic**. They group prior work into 3–4 named buckets and use each bucket to position one of their contributions. TokenFlow's three buckets map directly to its three contributions; Niyama's four buckets map to its four claims. For us: bucket prior work into (a) preemption-capable serving like TokenFlow/Llumnix/Andes, (b) KV offload like LMCache/AttentionStore, (c) SLO/deadline scheduling like Niyama, and use each bucket to contrast one of our design choices.
- **Both include a "discussion" or scope-limiting section that names what they don't do**. TokenFlow's §8 explicitly notes multi-node extension and non-human clients as out of scope but tractable. Niyama's related-work effectively does this. For us: an APSys workshop submission should include a short "limitations and future work" paragraph naming what we punt on (e.g., SSD tier, multi-node host-tier sharing) as known gaps rather than ignored ones.
- **Workshop-length papers (Scorpio is the closest in form to our APSys target) compress hard at the front but keep mechanism math intact**. Scorpio has only ~5 intro paragraphs, only ~4 background paragraphs, and only ~1 paragraph per mechanism in §3, but it still writes Equations 4–8 (ITL model, EstimatedTPOT, prefill piecewise, VBS sum) inline. The math survives the cut, not the prose. For our APSys version: keep the slack/priority math; cut narrative paragraphs about why preemption is bad; do not cut the formula that defines slack.
- **One paper does mechanism-first, the other does scenario-first — Llumnix shows that a single mechanism can unify multiple goals via an abstraction layer**. Llumnix's §4 leads with the live-migration mechanism, then introduces the "virtual usage" abstraction that makes load-balancing, de-fragmentation, prioritization, and auto-scaling all reduce to one policy. This is a stronger paper structure than enumerating policies separately. For us: consider whether checkpoint + slack + reroute can be unified under a single "virtual urgency" or "virtual slack" abstraction, so the scheduler is one rule (pick by virtual slack) and the three mechanisms are just how that rule is realized — that would tighten the design section and the related-work positioning.
- **Llumnix migration mechanism is the right contrast for our cross-engine reroute**. Llumnix migrates a live request between instances by pipelining KV copy with continuing decode; we copy KV blocks during preemption (not during live serving) and resume on a different engine. The mechanical difference — proactive disruption-driven checkpoint vs reactive migration triggered by per-instance load imbalance — is exactly the kind of comparison Llumnix's related-work would write about us. For us: in our related-work, contrast Llumnix as "live migration on healthy instances for load-balancing" vs ours as "checkpointed reroute on disrupted instances for failover-gap reduction," and call out the append-only KV trick as a shared low-level mechanism.
