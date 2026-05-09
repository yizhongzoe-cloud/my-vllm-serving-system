# Paper Framing — Non-interactive Long-Context Serving

Decided 2026-05-01 morning. Replaces earlier "Long-context RAG FT" framing.

---

## Core insight

Interactive (chat-style) LLM serving caps usable context at ~16K because TTFT
budget < prefill time. Production long-context (64K-128K) is dominantly
**non-interactive**:

- Anthropic / OpenAI batch APIs (24h SLO, 50% discount)
- Document analysis pipelines (legal eDiscovery, scientific paper review)
- Agentic workflows (Devin-style autonomous coding, AutoGen, research agents)
- Synthetic data generation with retrieval grounding

These tolerate minutes-hours TTLT, do NOT tolerate work loss on disruption.

## Why this framing is stronger than "long-context RAG"

1. **Less competition**: most preempt papers (QoServe, FlowPrefill, DBMS-inspired)
   target interactive SLO. Non-interactive batch + disruption recovery is
   under-explored.
2. **Cleaner mechanism fit**: TBT constraints don't apply, so preempt mid-decode
   is OK. Our 1-2s restore is negligible against minutes-long TTLT (vs being
   20-40% of an interactive TTFT budget).
3. **Context goes further**: 128K natively (Llama-3.1-8B trained limit), 256K
   with care. Recovery-time-vs-context-length figure becomes much sharper
   because reprefill grows superlinearly with context.
4. **Production motivation is real**: concrete use cases with named industry
   examples.

## Recovery advantage scales with context

| Context | Reprefill cost | Our restore | Advantage |
|---|---|---|---|
| 16K  | ~5s   | ~1.5s | 3x  |
| 32K  | ~10s  | ~1.5s | 7x  |
| 64K  | ~22s  | ~1.5s | 15x |
| 128K | ~60s  | ~2s   | **30x** |
| 256K | ~250s | ~3s   | **80x** |

Main paper figure: `recovery-time-vs-context-length`. Flat for us, superlinear
for reprefill. The divergence enables disruption-tolerant batch serving at long
contexts; reprefill cannot keep pace as context scales.

## RAG fits inside this framing

RAG spans interactive (chat + retrieval) and non-interactive (batch document
analysis with retrieval). At long context (64K+), RAG is dominantly
non-interactive in production. Our framing covers all long-context
non-interactive workloads, of which batch RAG is one example.

## Use cases under new framing

- **Batch RAG**: process 1000 documents with retrieval-augmented analysis
- **Agentic workflows**: autonomous coding agents (Devin), research agents (AutoGen)
- **Document pipelines**: legal eDiscovery, scientific paper review, content moderation
- **Synthetic data generation**: long-context retrieval-grounded synthesis

## Implementation impact (vs current code)

- Mechanism: **unchanged** (KV checkpoint, M3 preempt, restore wiring reusable)
- Workload: extend to 128K (RULER 128K dataset; reuse prepare.py and
  convert_ruler.py with `max_seq_length=131072`)
- Experiments: add multi-context sweep (32K / 64K / 128K), measure
  TTLT-relative disruption overhead
- Paper writing: rewrite intro/abstract; main figure becomes context-length
  sweep; metric shifts from TTFT/SLO violation to TTLT-relative overhead

## Title candidates

- "Disruption-Tolerant Long-Context Batch LLM Serving"
- "Bounded-Cost Recovery for Non-Interactive Long-Context LLM Tasks"
- "Checkpoint-Based Disruption Recovery for Long-Running LLM Workloads"

## Detailed mechanism comparison (人话版)

How each preempt-capable system actually handles preemption:

| System | When triggered | Who is preempted | How KV is handled |
|---|---|---|---|
| **vLLM default (RECOMPUTE)** | GPU memory full, can't admit new req | Most-recently-admitted or oldest (FCFS) | **Discarded**, victim re-prefills from prompt |
| **vLLM SWAP mode (legacy)** | Same as above | Same | Synchronous DMA GPU→host RAM, swap back on resume. V1 default disabled (recompute usually faster for short prompts) |
| **QoServe / Niyama** (Microsoft, ASPLOS '26) | High-priority req arrives needing SLO | "**Only requests with a few prefill chunks processed**" — early prefill | Not explicitly stated; uses underlying vLLM mechanism (likely RECOMPUTE since cheap for early prefill) |
| **FlowPrefill** | High-priority arrival + head-of-line blocking | Any prefill-stage req, but **only at operator boundaries** (qkv_proj, attn, etc.) | KV stays in GPU (paused between operators, never leaves GPU) |
| **DBMS-Inspired Preemption** (arXiv 2411.07447) | Memory pressure + cost model decision | "**Short** requests" (small recompute overhead) | Goes through vLLM RECOMPUTE path, drops KV |
| **QLLM** (EuroMLSys '25, MoE) | High-priority arrival | Best-effort batch jobs (entire batch evicted) | MoE-specific, not directly comparable to dense models |
| **Hummingbird** | Microsecond-scale (hardware-level) | Whole-request | Hardware-level GPU preempt, doesn't address req-level state |
| **ConServe** | GPU harvest signal | Best-effort jobs | Cluster-level granularity, not within a single inference engine |
| **Llumnix** (OSDI '24) | Load imbalance across instances | Any req on overloaded instance | KV peer-to-peer transfer (GPU→GPU across instances), not host offload |
| **TokenFlow** ⚠️ (EuroSys '26) | Request burst, low buffer, low token consumption | Dynamic priority by token-buffer occupancy + consumption rate | **GPU↔CPU memory KV transfer in background, overlap I/O with compute** — sounds VERY close to our approach |
| **Aegaeon** (SOSP '25) | Multi-model auto-scaling at token granularity | Across multiple co-resident models | Component reuse + GPU-CPU KV synchronization (multi-model focus, not single-model long-context) |
| **Ascendra** | At-risk requests near SLO deadline | Low-priority instance offloads to high-priority | Cross-instance offload (similar to Llumnix style); 2-priority pool |
| **PecSched** | Long-input req head-of-line blocks short-inputs | Long-input requests (deferred) | Cluster-level, doesn't address single-engine preempt |
| **Semantic Scheduling** | Urgency by predicted output / semantic | Lower urgency reqs | Standard preempt, KV-cache optimizations |
| **Andes** | QoE-driven (TTFT + smooth TBT under burst) | Reqs that hurt aggregate QoE | Streaming-focused, not long-context preempt |
| **Our system** | **SLO budget tight** OR **fault detected** OR **agent pause** OR **load spike** | **Any req, any prefill stage, any context length** (no scope limit) | **KV offloaded to host via incremental checkpoint** (already on host before preempt event); resume = host→GPU DMA ~1.5s |

### ⚠️ Closest competitor: TokenFlow (EuroSys '26)

TokenFlow's description from arXiv (2510.02758):

> "Preemptive request scheduling and proactive key-value (KV) cache management.
> Dynamically prioritizes requests based on real-time token buffer occupancy and
> token consumption rate, while actively transferring KV cache between GPU and
> CPU memory in the background and overlapping I/O with computation to minimize
> request preemption overhead."

**This sounds extremely close to our approach** (incremental host KV offload to
make preempt cheap). Need to read this paper carefully before claiming novelty.

Possible differentiation points to verify by reading TokenFlow:
- Their target: "responsive text streaming under burst" → likely interactive
  (TTFT/TBT focus, short context). Ours is non-interactive long-context.
- Their trigger: "token buffer occupancy" — likely an output buffer for streaming.
  Different from our SLO/disruption-driven trigger.
- Their workload: 8B–32B at standard context. Ours: 64K–128K long context.
- Their preempt scope: still constrained by streaming SLOs (TBT).

If TokenFlow is interactive-focused, we still differentiate by long-context
non-interactive regime. If TokenFlow already covers our use cases, we'd need
to find sharper differentiation or re-frame.

**Action item**: read TokenFlow paper carefully before finalizing paper framing.

### Three dimensions cleanly contrasted

**Trigger source** (not just capacity-driven):
- Capacity-driven: vLLM, DBMS-Inspired, ConServe
- Priority/SLO arrival: QoServe, QLLM
- Head-of-line blocking: FlowPrefill
- Load imbalance: Llumnix
- **Multi-source (capacity + SLO + fault + agent + spike)**: ours

**Scope of preemptable requests**:
- Any: vLLM, FlowPrefill, ours
- Only early prefill: QoServe (state-loss must be small)
- Only short prompts: DBMS-Inspired (recompute must be cheap)
- Only best-effort tier: QLLM, ConServe (priority-based)

**KV state handling** (the cost-determining choice):
- Discard + reprefill: vLLM default, QoServe (likely), DBMS-Inspired
- Keep on GPU: FlowPrefill (operator-boundary pause, KV never leaves GPU)
- Offload to host (synchronous): vLLM SWAP legacy
- **Offload to host (asynchronous incremental)**: ours
- GPU-to-GPU peer transfer: Llumnix
- Hardware-level: Hummingbird

### Where we sit uniquely

The intersection (multi-source trigger × any-stage scope × incremental host offload) is empty in the literature. Each existing system optimizes a subset; combining all three is the contribution.

## How this differentiates from existing literature

| Paper class | Their scope | Our differentiator |
|---|---|---|
| Interactive SLO preempt (QoServe / Niyama, FlowPrefill, JITServe) | Short or early-prefill, tight TBT, drop-and-reprefill | Long context any stage, non-interactive (no TBT), checkpoint-based |
| Cluster batch scheduling (AlphaServe, K8s-based) | Process-level granularity | Request-level fine-grained inside a single vLLM process |
| Fault tolerance for LLM serving | Few papers, mostly node-level | Per-request checkpoint, sub-second recovery |
| Cross-instance migration (Llumnix) | Load balance across instances | Local offload within instance |
| KV offload for capacity (Mooncake, vAttention) | Capacity scaling, not recovery | Same primitive applied to recovery / preempt |

## Risks / honest caveats

- **vLLM SWAP mode already exists** (deprecated): mechanism-wise we're close to
  reviving it. Our novelty lives in (a) async incremental save, (b)
  SLO/disruption-driven trigger (not just memory pressure), (c) multi-disruption
  use case framing, and (d) re-validation in long-context regime where SWAP is
  the right choice (vs deployed RECOMPUTE).
- **Comparison baseline matters**: vs vLLM RECOMPUTE (deployed): 15-30x
  improvement. vs hypothetical maintained vLLM SWAP: ~2x improvement. Paper
  should compare against deployed RECOMPUTE (the actual production choice).
- **Non-interactive motivation depends on production examples**: cite Anthropic
  batch API, OpenAI batch API, agentic system papers explicitly to ground the
  motivation.

## Apply when writing/talking

Lead with "long-context non-interactive serving" — RAG/agentic/batch as
instances. Don't lead with RAG alone (too narrow) or interactive SLO (wrong
lane). Recovery-time-vs-context-length is the headline figure.

## Status (as of 2026-05-01)

- Framing decision recorded
- Experiments not yet adapted to multi-context sweep
- L40S 5-cell ablation (current data) is single-context (64K), needs extension
- Implementation effort: ~1-2 days for workload + experiments
- Code mechanism is reusable as-is

## Cross-references

- `experiments_v2/docs/CURRENT_STATUS_2026_04_29.md` — full session findings
- `experiments_v2/docs/MACHINE_SETUP_RUNBOOK.md` — environment setup procedure
- `~/.claude/projects/.../memory/project_paper_framing_rag.md` — same content
  in cross-session memory; same source of truth
