# proposal - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | proposal |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 3 |
| **Design Style** | Academic clean / technical |
| **Target Audience** | Academic readers familiar with LLM serving |
| **Use Case** | Research proposal — KV-checkpoint dual-purpose primitive for FT + SLO scheduling |
| **Created Date** | 2026-04-30 |

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

- **Style**: Academic clean, technical, high information density
- **Theme**: Light theme (white background)
- **Tone**: Serious, technical, no decorative noise

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | Page background |
| **Title text** | `#000000` | All page titles and section headings (per user requirement) |
| **Body text** | `#374151` | Main body text (dark gray) |
| **Secondary text** | `#6B7280` | Captions, annotations, footnotes |
| **Accent** | `#DC2626` | Key insight emphasis, critical numbers, "key observation" callouts |
| **Border/divider** | `#D1D5DB` | Light gray dividers and box borders |
| **Subtle bg** | `#F3F4F6` | Light gray background for code/quote blocks |

---

## IV. Typography System

### Font Plan

**Typography direction**: Cross-platform PPT-safe sans-serif, academic clean

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Body** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Emphasis** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Code** | — | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**:

- Title: `Arial, "Microsoft YaHei", sans-serif`
- Body: `Arial, "Microsoft YaHei", sans-serif`
- Emphasis: same as Body
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 18px (dense academic content)

| Purpose | px | Weight |
| ------- | -- | ------ |
| Page title | 32 | Bold |
| Section heading | 22 | Bold |
| Body content | 18 | Regular |
| Emphasis (accent color) | 18 | Bold |
| Annotation | 14 | Regular |
| Code inline | 16 | Regular (monospace) |

---

## V. Layout Principles

### Page Structure

- **Header**: ~60px — page title + thin divider line below
- **Content**: ~600px — main content
- **Footer**: ~30px — page number "p.N / 3" right-aligned, secondary text color

### Layout Patterns Used

- **P01** Problem + RQ: Top-bottom split — top half motivation/problem, bottom half boxed Research Question with accent left border
- **P02** System Design: Top-bottom split — top three columns showing 3 components, bottom a single-line key insight callout with accent
- **P03** Evaluation: 2×2 + bottom row — top-left baselines, top-right workloads, bottom-left metrics, bottom-right hypothesis

### Spacing

- Safe margin: 60px L/R, 50px T/B
- Block gap: 28px
- Line-height: 1.5× body

---

## VI. Icon Usage

No icons. Pure typography + thin dividers + one schematic diagram on P02.

---

## VII. Visualization Reference

P02 includes a custom inline SVG diagram (3-component data flow): Engine A KV blocks → host pinned mem → /dev/shm shared FS → Engine B restore. Drawn as labeled boxes with arrows. Custom inline SVG, no template.

---

## VIII. Image Resource List

None.

---

## IX. Content Outline

### Slide 01 — Problem & Research Question

- **Layout**: Top-bottom split (Motivation top ~50%, RQ box bottom ~50%)
- **Title**: "Long-context RAG serving has a recovery problem"
- **Top half content**:
  - Long-context RAG (32K–128K tokens) is now production-common
  - Mid-run engine failure forfeits 20–90s of prefill compute per request
  - Naive re-prefill on a surviving replica cannot meet TTFT SLOs
  - Existing FT systems (Mooncake) and SLO-aware schedulers (QLM, Scorpio, JITServe, SLOs-Serve) work in isolation
  - **Hidden opportunity**: the host-memory KV state maintained for FT recovery is exactly what an SLO-aware scheduler needs to preempt cheaply
- **Bottom box (RQ)** with red accent left border:
  - **Research Question**: Can the host-memory KV state maintained for fault recovery serve as a *dual-purpose primitive* — enabling both fast restore after failure and cheap preemption of in-flight long-context requests during normal operation?

### Slide 02 — System Design

- **Layout**: Top-bottom split (3-column components ~70%, key insight callout ~30%)
- **Title**: "System Design: KV checkpoint as dual-purpose primitive"
- **Three columns** (each a labeled card with section heading + 2-3 short bullets):
  - **① KV Cache Checkpointing** — Periodic async GPU→host pinned mem copy, then publish to `/dev/shm` via atomic write. Frequency governed by online cost model.
  - **② Cross-engine Restore** — On engine failure, displaced requests rerouted to surviving engine; restore KV blocks from shared FS into freshly allocated GPU blocks. Resume decoding without re-prefill.
  - **③ Preemption-aware SLO Scheduler** — Per-step compute remaining SLO budget (TTFT, TPOT). Sort tightest-budget-first. Preempt executing request when budget exceeds top pending by hysteresis margin. Post-fault: immediately reschedule all in-flight + recovered requests.
- **Key insight callout** (full-width, accent-bordered, slightly elevated):
  - Vanilla SLO schedulers can't preempt long-context requests cheaply (mid-prefill state too costly to abandon). Vanilla FT systems already maintain that state. Sharing the checkpoint between subsystems makes preemption a side-effect of work already paid for.
- Built on vLLM v0.16

### Slide 03 — Evaluation Plan

- **Layout**: 2×2 grid (Baselines, Workloads, Metrics, Hypothesis) + thin bottom strip for Resources
- **Title**: "Evaluation: isolating FT vs. scheduling contributions"
- **Quadrant 1 — Baselines** (4 configurations on same vLLM substrate):
  - No-FT + FCFS (vanilla; in-flight lost on failure)
  - Reprefill + FCFS (reroute + re-prefill)
  - Checkpoint + FCFS (KV restore, vanilla scheduling)
  - **Checkpoint + SLO-aware (ours)** — accent-highlighted
  - 3 vs. 4 isolates scheduling contribution; 4 vs. 2 isolates FT contribution
- **Quadrant 2 — Workloads & Faults**:
  - ShareGPT (short-context chat)
  - LongBench: NarrativeQA, QMSum, MuSiQue (long-context)
  - Azure LLM trace (production arrivals)
  - Poisson arrivals, 3 random seeds
  - Fault injection: single engine failure at 350s into 700s cell
- **Quadrant 3 — Metrics**:
  - **Primary**: SLO-met output tokens/sec (goodput)
  - Secondary: recovery success rate, TTFT/TPOT distributions, GPU utilization
  - SLO swept along TTFT and failover-gap axes
- **Quadrant 4 — Hypothesis** (accent border):
  - Tight TTFT (<2s for 15K-token prompts): Checkpoint+FCFS beats Reprefill+FCFS but advantage hidden by surviving-engine queueing. Checkpoint+SLO-aware preserves advantage by letting recovery requests preempt non-urgent work. Loose SLO: all FT configurations converge.
- **Bottom strip — Resources**: 2+ GPUs (DP), Llama-3.1-8B-Instruct, ~50 GPU-hours main + 20 sensitivity

---

## X. Speaker Notes

One file per page in `notes/`. Match SVG name (`01_problem_rq.md`, `02_system_design.md`, `03_evaluation.md`).

---

## XI. Technical Constraints

- viewBox `0 0 1280 720`
- White `<rect>` background
- `<tspan>` for text wrapping (no `<foreignObject>`)
- Use `fill-opacity` / `stroke-opacity`, no `rgba()`
- No `<style>`, `class`, `mask`, `textPath`, `animate*`, `script`
- Per-element opacity, not `<g opacity>`
- Inline styles only
