#!/usr/bin/env python3
"""Generate progress_slides_2.pdf using fpdf2 + PNG images."""

import os
from fpdf import FPDF

DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(DIR, "img")
OUT = os.path.join(DIR, "progress_slides_2.pdf")

# 16:9 landscape in mm
W, H = 338, 190
MARGIN = 12
CW = W - 2 * MARGIN  # content width

# Colors
TITLE_BG = (30, 58, 95)
WHITE = (255, 255, 255)
BLACK = (34, 34, 34)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
RED = (220, 38, 38)
GRAY_BG = (240, 244, 248)


class Slides(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format=(H, W))
        self.set_auto_page_break(auto=False)

    # --- helpers ---
    def _title_page(self, title, subtitle=""):
        self.add_page()
        self.set_fill_color(*TITLE_BG)
        self.rect(0, 0, W, H, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 26)
        self.set_y(60)
        for line in title.split("\n"):
            self.cell(W, 12, line, align="C", new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self.set_y(self.get_y() + 10)
            self.set_font("Helvetica", "", 16)
            self.set_text_color(160, 200, 230)
            self.cell(W, 10, subtitle, align="C")

    def _slide(self, title):
        self.add_page()
        self.set_fill_color(*TITLE_BG)
        self.rect(0, 0, W, 20, "F")
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 15)
        self.set_xy(MARGIN, 3)
        self.cell(CW, 14, title)
        self.set_text_color(*BLACK)
        self.set_y(24)

    def _h3(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLUE)
        self.set_x(MARGIN)
        self.cell(CW, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*BLACK)

    def _bullet(self, text, indent=0, size=9):
        x = MARGIN + 4 + indent
        self.set_font("Helvetica", "", size)
        self.set_x(x)
        # Split bold markers
        parts = []
        rest = text
        while "**" in rest:
            i = rest.index("**")
            j = rest.index("**", i + 2)
            parts.append(("", rest[:i]))
            parts.append(("B", rest[i+2:j]))
            rest = rest[j+2:]
        parts.append(("", rest))

        self.cell(3, 5, "-")
        self.set_x(x + 4)
        for style, seg in parts:
            self.set_font("Helvetica", style, size)
            self.write(5, seg)
        self.ln(6)

    def _sub_bullet(self, text, size=9):
        self._bullet(text, indent=6, size=size)

    def _img(self, name, w=None, center=True):
        path = os.path.join(IMG, name)
        if not os.path.exists(path):
            self.set_font("Helvetica", "I", 9)
            self.set_x(MARGIN)
            self.cell(CW, 6, f"[Image: {name}]", new_x="LMARGIN", new_y="NEXT")
            return
        if w is None:
            w = min(CW, 280)
        x = (W - w) / 2 if center else MARGIN
        self.image(path, x=x, y=self.get_y(), w=w)
        # estimate height from aspect ratio
        from PIL import Image
        im = Image.open(path)
        iw, ih = im.size
        drawn_h = w * ih / iw
        self.set_y(self.get_y() + drawn_h + 2)

    def _table(self, headers, rows, col_widths=None):
        n = len(headers)
        if col_widths is None:
            col_widths = [CW / n] * n
        self.set_x(MARGIN)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*TITLE_BG)
        self.set_text_color(*WHITE)
        for h, cw in zip(headers, col_widths):
            self.cell(cw, 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(*BLACK)
        for ri, row in enumerate(rows):
            self.set_x(MARGIN)
            self.set_font("Helvetica", "", 8)
            bg = GRAY_BG if ri % 2 == 0 else WHITE
            self.set_fill_color(*bg)
            for c, cw in zip(row, col_widths):
                self.cell(cw, 5.5, c, border=1, fill=True)
            self.ln()
        self.set_fill_color(*WHITE)
        self.ln(1)


def build():
    pdf = Slides()

    # ---- Title ----
    pdf._title_page(
        "Fault-Tolerant Multi-GPU LLM Serving\nwith Adaptive KV-Cache Checkpointing",
        "Progress Update",
    )

    # ---- Slide 1: Scope ----
    pdf._slide("Scope")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_x(MARGIN + 10)
    pdf.write(8, "All results are from a ")
    pdf.set_font("Helvetica", "B", 14)
    pdf.write(8, "proof-of-concept setup")
    pdf.set_font("Helvetica", "", 14)
    pdf.write(8, ":")
    pdf.ln(14)
    pdf._bullet("1B model, 2 GPUs, synthetic workloads", size=12)
    pdf._bullet("Goal: validate the end-to-end pipeline", size=12)
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_x(MARGIN + 10)
    pdf.write(7, "Production-scale evaluation (8B model, 4+ GPUs, real datasets) is the next step.")

    # ---- Slide 2: Problem ----
    pdf._slide("Problem")
    pdf._bullet("Single-node multi-GPU LLM serving: replicas can fail (GPU reset, OOM, driver crash...)")
    pdf._bullet("Failure -> capacity loss -> interrupted streams, SLO violations, goodput drop")
    pdf._bullet("Naive checkpoint: too frequent = high overhead; too rare = long recovery")
    pdf._bullet("**Core question: how to jointly optimize routing, admission, and adaptive checkpointing to maximize goodput while meeting SLOs under failures?**")

    # ---- Slide 3: Our System ----
    pdf._slide("Our System")
    # Left side
    pdf._h3("(1) Benders-based Robust Scheduler")
    pdf._bullet("Periodic decision epochs on system snapshot")
    pdf._bullet("Master: admission + routing decisions")
    pdf._bullet("Subproblem: verify recovery per failure scenario")
    pdf._bullet("Pooled-flow screening + logic-based cuts")
    pdf.ln(1)
    pdf._h3("(2) Adaptive Checkpoint Policy")
    pdf._bullet("Per-request, triggered on new stable KV blocks")
    pdf._bullet("Publish when: Dreplay > Dload + lambda * Dckpt")
    pdf._bullet("Early: skip (cheap to replay) / Late: checkpoint (expensive to lose)")
    pdf.ln(2)
    pdf._h3("Ablation Matrix")
    pdf._table(
        ["", "Fixed Checkpoint", "Adaptive Checkpoint"],
        [
            ["Greedy Routing", "Fixed-Low / Fixed-High", "Checkpoint-Only"],
            ["Benders Routing", "Robust-Routing-Only", "Our-System"],
        ],
        col_widths=[45, 60, 60],
    )
    pdf._bullet("Plus **No-FT** (no fault tolerance) as overhead-free reference")

    # ---- Slide 4: Setup ----
    pdf._slide("Experiment Setup")
    pdf._h3("Hardware & Model")
    pdf._bullet("Llama-3.2-1B-Instruct, 2 GPUs (DP replicas), 8GB checkpoint pool (host RAM)")
    pdf.ln(1)
    pdf._h3("Workloads")
    pdf._table(
        ["Workload", "Prompt", "Output", "Arrival"],
        [
            ["W1 Short Interactive", "50-200", "20-100", "Poisson"],
            ["W2 Long Generation", "200-500", "200-500", "Poisson"],
            ["W3 Bursty Mixed", "50-500", "20-500", "Bursty"],
        ],
        col_widths=[55, 30, 30, 30],
    )
    pdf._bullet("**Poisson**: steady rate, random intervals (stable load)")
    pdf._bullet("**Bursty**: 15s normal -> 5s at 3x rate -> repeat (traffic spikes)")
    pdf.ln(1)
    pdf._h3("Load & Faults")
    pdf._bullet("Load: Low (1), Medium (3), High (5 rps) / Faults: none / F1 (15s) / F2 (30s) / F3 (50s)")
    pdf._bullet("Duration: 70s / SLO: TTFT 2s, TPOT 100ms, Failover gap 3s")
    pdf.ln(1)
    pdf._h3("5 Experiments")
    pdf._table(
        ["Experiment", "Goal", "Key Metrics"],
        [
            ["E1: Main", "End-to-end comparison", "Goodput, SLO violation, failover gap"],
            ["E2: Recovery", "Recovery time breakdown", "Detection / KV Restore / Replay"],
            ["E3: Ablation", "Routing vs Checkpoint contrib.", "Goodput by component"],
            ["E4: Tradeoff", "Ckpt overhead vs recovery", "Goodput + failover gap"],
            ["E5: Controller", "Solver overhead", "Solver latency per epoch"],
        ],
        col_widths=[40, 60, 70],
    )

    # ---- Slide 5: E1 No Fault ----
    pdf._slide("E1: Goodput -- No Fault")
    pdf._img("goodput_by_load_none.png", w=260)
    pdf._bullet("All methods ~ No-FT -> **near-zero normal-case overhead**")
    pdf._bullet("Exception: **Fixed-High-CKPT** drops ~29% on W2/High (1216 vs 1715 tok/s)")

    # ---- Slide 6: E1 With Fault ----
    pdf._slide("E1: Goodput -- With Fault (F2_Mid)")
    pdf._img("goodput_by_load_F2_Mid.png", w=260)
    pdf._bullet("No-FT **silently drops fault-hit requests** (completion 97-99%, 0% failover)")
    pdf._bullet("Our-System matches Fixed-Low on W2/Med (~1040) with 100% completion")
    pdf._bullet("Fixed-High-CKPT worst: overhead + slow recovery + low completion")

    # ---- Slide 7: SLO ----
    pdf._slide("E1: SLO Violations (F2_Mid)")
    pdf._img("slo_violation_F2_Mid.png", w=220)
    pdf._bullet("With fault: **Fixed-High-CKPT** avg ~7% (worst 20%+); **Our-System ~ 0%**")
    pdf._bullet("Without fault: all methods 0% violation (omitted -- no differentiation)")

    # ---- Slide 8: Failover Gap ----
    pdf._slide("E1: Failover Gap (p95)")
    pdf._img("failover_gap_p95.png", w=280)
    pdf._bullet("No-FT: 0% recovery on W2/W3 (succ 0/34, 0/27)")
    pdf._bullet("Fixed-High: highest gap (~700ms) + lowest success (77/117 = 66%)")
    pdf._bullet("**Our-System: moderate gap + highest success rate** (W3: 90/97 = 93%)")

    # ---- Slide 9: E2 Recovery ----
    pdf._slide("E2: Recovery Time Breakdown")
    pdf._img("recovery_breakdown.png", w=260)
    pdf._bullet("W1 (short): detection dominates (~200ms), all similar")
    pdf._bullet("W2 (long): Fixed-High ~830ms; Fixed-Low ~350ms; **Our-System ~330ms**")
    pdf._bullet("Adaptive ckpt finds **sweet spot**: not too much to load, not too much to replay")

    # ---- Slide 10: E3 Ablation ----
    pdf._slide("E3: Ablation -- With Fault (F2_Mid)")
    pdf._img("ablation_F2_Mid.png", w=260)
    pdf._bullet("W1 (short): all similar -- short tasks easy to recover")
    pdf._bullet("W2/Med: Our-System (~1042) ~ Ckpt-Only (~1048) >> Robust-Only (~666)")
    pdf._bullet("**Adaptive checkpoint is the primary contributor**; routing matters more at scale")
    pdf._bullet("No-fault ablation omitted -- same story as Slide 5 (Fixed-High -40%, others equal)", size=8)

    # ---- Slide 11: E4 Tradeoff ----
    pdf._slide("E4: Checkpoint Overhead vs Recovery Benefit")
    pdf._img("checkpoint_tradeoff.png", w=260)
    pdf._bullet("Left (normal): Fixed-High loses ~29%; others ~ No-FT")
    pdf._bullet("Right (fault): No-FT 0/10; Fixed-High 22/39 (56%), gap ~883ms")
    pdf._bullet("**Our-System: 27/27 (100%), gap ~491ms -- best on both axes**")
    pdf._bullet("**More checkpointing != better**")

    # ---- Slide 12: E5 Controller ----
    pdf._slide("E5: Controller Overhead")
    pdf._img("controller_overhead.png", w=180)
    pdf._bullet("Median solver latency: **~15ms/epoch**, flat across 1-12 requests")
    pdf._bullet("Outliers 160-180ms: cold start (first 1-2 calls), fixable via warmup")
    pdf._bullet("Acceptable for online serving (decode step is ms-scale)")

    # ---- Slide 13: Summary ----
    pdf._slide("Summary")
    pdf._h3("Key Findings")
    pdf._bullet("1. **Near-zero normal-case overhead**: goodput matches No-FT")
    pdf._bullet("2. **Best under failures**: highest recovery rate + lowest SLO violations")
    pdf._bullet("3. **More checkpointing != better**: Fixed-High hurts both goodput and recovery")
    pdf._bullet("4. **Adaptive checkpoint is the primary contributor**; routing matters more at scale")
    pdf._bullet("5. **Solver overhead is small**: ~15ms per epoch")
    pdf.ln(3)
    pdf._h3("Next Steps")
    pdf._bullet("Real datasets: ShareGPT / CNN-DailyMail / Alpaca (real length distributions)")
    pdf._bullet("Lightweight output-length predictor for better admission decisions")
    pdf._bullet("Scale to 8B model on 4 GPUs (dp=4, losing 1 GPU = 25% capacity loss)")
    pdf._bullet("Profile solver parameters from actual hardware measurements")
    pdf._bullet("Per-request SLO differentiation (e.g. chat 100ms, summarization 200ms)")
    pdf._bullet("3 seeds, 300s runs for statistical significance")

    pdf.output(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    build()
