# disruption_aware_slo_advisor_meeting - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | disruption_aware_slo_advisor_meeting |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 12 |
| **Design Style** | B) General Consulting — academic / minimalist |
| **Target Audience** | PhD advisor (1-on-1 weekly meeting) |
| **Use Case** | Internal pitch of paper idea + experiment plan; preparation for APSys 2026 workshop submission |
| **Created Date** | 2026-05-13 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top/bottom 50px |
| **Content Area** | 1160×620 |

---

## III. Visual Theme

### Theme Style

- **Style**: Academic / minimalist — black and grey on white. No colored backgrounds. No accent colors. Emphasis carried by weight (bold), size, and italic — not hue.
- **Theme**: Light theme
- **Tone**: Restrained, technical, OSDI / SOSP slide aesthetic

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | Page background, always white |
| **Primary text** | `#111827` | Titles, body text, primary content |
| **Secondary text** | `#4B5563` | Captions, secondary points, table secondary cells |
| **Tertiary text** | `#9CA3AF` | Footnotes, page numbers, muted annotation |
| **Border / divider** | `#D1D5DB` | Table borders, dividers, placeholder figure dashed frames |
| **Subtle fill** | `#F3F4F6` | Table header band, "Ours" row highlight, callout block fill |

> No accent / brand color. Anything that would normally use color (✓ vs ✗ marks, "Ours" row highlight, baseline line vs ours line in placeholder figures) is rendered with **weight, fill grey, italic, or label** instead.

### Gradient Scheme

None used. All fills are flat solid.

---

## IV. Typography System

**Typography direction**: academic serif title + sans body, Latin-only (deck is fully English)

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | — | `Georgia`, `Times New Roman` | `serif` |
| **Body** | — | `Arial`, `Helvetica Neue` | `sans-serif` |
| **Emphasis** | — | `Georgia` (italic) | `serif` |
| **Code** | — | `Consolas`, `Courier New` | `monospace` |

**Per-role font stacks**:

- Title: `Georgia, "Times New Roman", serif`
- Body: `Arial, "Helvetica Neue", sans-serif`
- Emphasis: same as Title (used italic for inline term emphasis)
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = `18px` (dense — technical content density on a 12-page deck)

| Purpose | Ratio to body | Px @ body=18 | Weight |
| ------- | ------------- | ------------ | ------ |
| Cover title | 3-4x | 56-72px | Bold |
| Page title | 1.6-1.8x | 30-32px | Bold |
| Subtitle / section header | 1.2-1.3x | 22-24px | SemiBold |
| **Body** | **1x** | **18px** | Regular |
| Code inline | 0.9-1x | 16-18px | Regular monospace |
| Annotation / caption | 0.75-0.85x | 14-15px | Regular |
| Page number / footnote | 0.6-0.7x | 11-12px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area** (~80px): page title + thin divider rule (`#D1D5DB`, 1px)
- **Content area** (~570px): main body
- **Footer area** (~30px): page number right, deck title left, both `#9CA3AF` 11px

### Layout Pattern Library

| Pattern | Used for |
| ------- | -------- |
| Single column centered | Cover (P01), key insight (P05), submission plan (P12) |
| Symmetric split (5:5) | Two-scenario comparison (P03) |
| Top-bottom split | Numerical pain point (P02), architecture overview (P06) |
| Asymmetric split | Mechanism + invariants (P07), policy formula + guards (P08), reroute path (P09) |
| Full-width table | Empty quadrant (P04), eval setup (P10) |
| Three-column placeholder figures | Experiments (P11) |

### Spacing Specification

**Universal**:

| Element | Value |
| ------- | ----- |
| Safe margin from canvas edge | 60px L/R, 50px T/B |
| Content block gap | 28-36px |
| Icon-text gap | 10px |

**Non-card (default — this deck uses dividers and whitespace, not card grids)**:

- Vertical rhythm via whitespace + 1px `#D1D5DB` rules
- Line-height: 1.5× body
- No rounded card containers; everything reads as flat naked blocks separated by dividers

---

## VI. Icon Usage Specification

### Source

- **Library**: `tabler-outline` (line art, academic aesthetic)
- **Stroke width**: `2`
- **Color**: stroke = `#111827` (primary text); never colored

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Long prefill cost / time | `tabler-outline/clock` | P02 |
| Engine death / fault | `tabler-outline/skull` | P03 |
| Priority / urgent | `tabler-outline/bolt` | P03 |
| Existing systems quadrant gap | `tabler-outline/target` | P04 |
| Key insight / one mechanism | `tabler-outline/cube` | P05 |
| Architecture: router | `tabler-outline/router` | P06 |
| Architecture: GPU engine | `tabler-outline/cpu` | P06 |
| Architecture: shm checkpoint store | `tabler-outline/database` | P06, P07 |
| Async pipeline / refresh | `tabler-outline/refresh` | P07 |
| Time / SLO clock | `tabler-outline/hourglass` | P08 |
| Network reroute | `tabler-outline/network` | P09 |
| Heartbeat / history | `tabler-outline/history` | P09 |
| Chart / experiment | `tabler-outline/chart-bar` | P10, P11 |
| Submission / export | `tabler-outline/file-export` | P12 |

---

## VII. Visualization Reference List

This deck contains no real data charts (figures on P11 are placeholder dashed frames with caption only). The single semi-structured visual is:

- **P04** — feature comparison table (10-row × 5-col), drawn as native SVG `<rect>` + `<text>` grid; no template needed.
- **P06** — architecture diagram (router + 2 engines + shm checkpoint store + arrows), drawn as native SVG `<rect>` + `<line>` + `<text>`.
- **P07** — async pipeline 4-stage horizontal flow (engine → worker → shm → restore), drawn as native SVG boxes + arrows.
- **P09** — reroute state machine sequence diagram, drawn as native SVG.
- **P11** — 3 placeholder line / bar plot frames (dashed border, axis stub, "Figure X: caption" label).

No chart template lookup needed; all visuals are simple geometric SVG.

---

## VIII. Image Resource List

No raster images. All visuals are native SVG geometry. (No `images/` directory entries.)

---

## IX. Content Outline

### Part 1: Motivation

#### Slide 01 - Cover
- **Layout**: Single column centered
- **Title**: Cheap Preemption Enables Disruption-Aware SLO Scheduling for Long-Context LLM Serving
- **Subtitle**: Weekly research update — APSys 2026 workshop draft
- **Info**: Yi Zhong · 2026-05-13

#### Slide 02 - The cost of throwing prefill away
- **Layout**: Top-bottom split — pain statement on top, two-bar comparison stub below
- **Title**: Long-context prefill is expensive — and we keep throwing it away
- **Content**:
  - Modern LLM workloads (RAG, long-doc QA, agent traces) routinely push prompts past 16K tokens
  - **RULER 16K prefill on Qwen2.5-7B + A6000 ≈ 2.7 s** (P95, calibrated)
  - **ShareGPT chat TTFT P95 ≈ 0.45 s** — ~6× gap
  - Once prefill is computed, discarding it costs seconds of GPU time per request

#### Slide 03 - Two scenarios that burn the prefill
- **Layout**: Symmetric split (5:5)
- **Title**: Two scenarios share the same root cause: no cheap resume
- **Left — Engine failure mid-decode**:
  - Long request lives on one GPU, GPU disappears (HW fault, OOM, drain, kill, multi-tenant preempt)
  - Long requests disproportionately exposed (lifetime ∝ disruption probability)
  - Recovery today: full reprefill on a surviving engine
- **Right — Priority preempt mid-decode**:
  - Tight-SLO request arrives, head-of-queue deadline tighter than running request's
  - Schedulers refuse to preempt running long-context decode — reprefill cost > priority benefit
  - Tight-SLO request waits or misses

#### Slide 04 - Existing systems leave one quadrant empty
- **Layout**: Full-width table
- **Title**: Every prior system fills 3 of 4 cells. None fills all four.
- **Content**: 10-row × 5-col feature comparison table (vLLM / QLM / Scorpio / JITServe / Niyama / Llumnix / Mooncake / TokenFlow / FastServe / **Ours**) across 4 axes: preempt long-context decode · disruption-aware · host-side KV · cross-engine state transfer
- The empty quadrant: `(cheap preempt of long-context decode) × (disruption-aware)` — the position we occupy

### Part 2: Idea & Design

#### Slide 05 - Key insight (one sentence)
- **Layout**: Single column centered (breathing page)
- **Title**: One mechanism, two uses
- **Content**:
  - A single host-side KV checkpoint, async-published to `/dev/shm` at block-aligned cadence, simultaneously serves as
    - (i) **recovery substrate** when an engine dies — peer reads checkpoint, resumes decode without reprefill
    - (ii) **cheap-preempt substrate** so the scheduler can preempt running long-context decode without wasting prefill
  - Same data. Two uses. No duplication.
  - When a mechanism's cost drops below a threshold, design space that was off-limits opens up.

#### Slide 06 - Architecture overview
- **Layout**: Top-bottom split — diagram top, one-line takeaway bottom
- **Title**: Router + per-GPU engines + shared shm checkpoint store
- **Content**:
  - Diagram: client → router (CPU-only, separate process) → 2× engine processes (one per GPU)
  - Each engine publishes KV checkpoints to host-side store with two tiers:
    - Layer 1: in-process pinned-memory pool
    - Layer 2: `/dev/shm` mirror, visible to peer engines
  - Router watches engine health via shm status file; reroutes on death
  - Same checkpoint backs both same-engine preempt-and-resume and cross-engine recovery

#### Slide 07 - Mechanism: host-side KV checkpoint
- **Layout**: Asymmetric split — diagram left (~5/12), text right (~7/12)
- **Title**: Block-aligned, two-tier, async pipeline
- **Content**:
  - **What/when**: per-block delta — every 16 tokens decoded (one vLLM KV block), save K/V tensors of newly-stable blocks
  - **Two tiers**: pinned host pool (sub-ms, same-engine) + `/dev/shm` mirror (cross-engine, atomic via tmp+rename)
  - **Async pipeline (4 stages)** — naive sync publish costs ~30% TTFT; pipeline drives that to negligible:
    1. engine: `checkpoint_kv_blocks` as Future + depth-1 backpressure, eager `num_checkpointed_tokens`
    2. worker: GPU-side batch gather → one async H2D for all running requests
    3. shm write: ctypes → libc `write` (releases GIL)
    4. restore: shared CUDA stream + batch-end `flush_pending_restore`
  - **Invariants**: "latest" pointer updates only after rename → readers never see partial chunks; engine-death gap absorbed by reload state machine as decode-replay

#### Slide 08 - Policy: slack-based preempt picker
- **Layout**: Asymmetric split — formula box left, three guards right
- **Title**: Pick laziest tail, admit most urgent head — measurement, not prediction
- **Content**:
  - **Slack**: `slack(r,t)` = wall-clock remaining before SLO miss
    - pre-first-token: `S_TTFT − elapsed_since_arrival`
    - post-first-token: `S_TPOT − avg_per_token_so_far`
  - **Picker rule**:
    - `slack(victim) − slack(head) > δ + replay_cost(victim)`
    - `AND num_checkpointed_tokens(victim) > 0`
    - `AND time_since_last_preempt(victim) > cooldown`
  - **Three guards**: hysteresis `δ` blocks thrash · `replay_cost` blocks bad trades · `cooldown` blocks repeat-victim
  - Defaults: `δ = 1 s`, `cooldown = 5 s` — measured at runtime, no offline profile

#### Slide 09 - Cross-engine reroute
- **Layout**: Asymmetric split — sequence diagram left, mechanism notes right
- **Title**: Heartbeat → reroute → reload state machine on surviving engine
- **Content**:
  - **Detect**: engine writes heartbeat file every 200 ms via daemon thread (NOT piggybacked on `step()` — idle engines never call step). Router polls every 500 ms; >2 s stale ⇒ dead.
  - **Reroute**: router enumerates dead engine's `in_flight` requests, re-forwards each to surviving engine with `vllm_xargs.is_rerouted=True` + original `internal_req_id` + published `num_checkpointed_tokens`
  - **Restore**: new engine's `add_request` sees `is_rerouted` → reload state machine: alloc KV blocks → `restore_kv_blocks` from `/dev/shm` → flip status to `PREEMPTED`. vLLM's resumed-from-preempt admit path takes over.

### Part 3: Evaluation

#### Slide 10 - Evaluation setup
- **Layout**: Full-width table (4-column setup matrix)
- **Title**: Setup at a glance
- **Content** (table):
  - Hardware: 2× A6000 PCIe (48 GB ea); portability check on 2× L40S
  - Model: Qwen2.5-7B-Instruct fp16, `max_model_len 32K`
  - Workloads: RULER 16K (long-context regime we target) · ShareGPT (short-context — must not break)
  - Arrivals: Poisson · Niyama-style 3-tier QoS (tight / normal / loose)
  - SLO calibration: baseline P95 on `vllm_fcfs` at uncontested low QPS; tiers = {2×, 3×, 6×} baseline
  - **Baselines**: `vllm_fcfs` (no router floor) · `reroute_no_ckpt` (architecture, no FT) · `ours` (full) · `ours_no_picker` (mechanism only, picker off)
  - 3 seeds per config, mean ± std reported

#### Slide 11 - Experiments and claims
- **Layout**: Three-column placeholder figures with caption blocks
- **Title**: Three experiments aligned to three contributions
- **Content**: Three placeholder figure frames with captions
  - **Figure 2 (E_M1)**: SLO attainment vs QPS — `ours` vs `reroute_no_ckpt` vs `vllm_fcfs`. Claim: ours holds attainment past saturation knee where baselines drop below 90%.
  - **Figure 4 (E_M4)**: ablation — add `ours_no_picker` to Figure 2. Claim: mechanism alone gives passive benefit; picker actively realizes the SLO advantage. Two contributions independent and additive.
  - **Figure 5 (E_D1)**: failover_gap (SIGKILL → next token) vs context length. Claim: ours ~1.5 s on RULER 16K; `reroute_no_ckpt` ~10–20 s at 16K, extrapolated tens of s at 64K. Gap grows with context length.

### Part 4: Plan

#### Slide 12 - Submission plan
- **Layout**: Single column centered
- **Title**: Submission plan
- **Content**:
  - **Target venue**: APSys 2026 Workshop · 6–8 pages
  - **Deadline**: May 20 (Wednesday)
  - **Status**:
    - Paper framework & section outline: drafted
    - E_M1 (a6000): partial results in (1.0–8.0 QPS sweep, 3 seeds)
    - E_M4 ablation: partial — `ours_no_picker` runs scheduled this week
    - E_D1 failover demo: working on RULER 16K, scaling to longer context
  - **Open questions** (from paper framework):
    - Tiered vs uniform as the headline figure
    - δ / cooldown sweep — small budget from E_M2

---

## X. Speaker Notes Requirements

- One file per page in `notes/`
- Filename matches SVG: `01_cover.md` … `12_submission_plan.md`
- Style: terse bullet-style script — this is a 1-on-1 advisor meeting, not a public talk
- Total presentation duration: ~25 min + Q&A
- Purpose: inform + get feedback on idea framing and experiment plan

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` (always white `#FFFFFF`)
3. Text wrapping uses `<tspan>`; `<foreignObject>` FORBIDDEN
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`, `textPath`, `animate*`, `script`
6. Raw Unicode for `—`, `→`, `≥`, `≤`, `×`, `µ`, `δ`, `α`, `β`, `γ`, `≈`; XML reserved chars `& < > " '` escaped
7. `marker-end` triangles only, in `<defs>`, `orient="auto"` (used for arrows in architecture / reroute / pipeline diagrams)
8. No `clipPath` (no images in this deck)

### PPT Compatibility Rules:

- `<g opacity="...">` FORBIDDEN; set on each child element individually
- Inline styles only; no external CSS, no `@font-face`
- All fonts in stacks end with cross-platform pre-installed family
