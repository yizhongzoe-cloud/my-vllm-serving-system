"""Generate Step5_Main result figures.

Reads <results-dir>/8B/Step5_Main/<baseline>/<workload>/<load>/<fault>/<seed>/metrics.json
and writes 5 PNGs to <out-dir>/.

Usage:
    # A6000 (default)
    python experiments_v2/plot_step5.py

    # L40S
    python experiments_v2/plot_step5.py \\
        --results-dir results_v2_l40s \\
        --out-dir experiments_v2/figures/8B/Step5_Main_l40s
"""
import argparse, json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results_v2",
                    help="Top-level results directory (default: results_v2)")
parser.add_argument("--out-dir", default="experiments_v2/figures/8B/Step5_Main",
                    help="Output directory for PNGs")
args = parser.parse_args()

RESULTS = args.results_dir
OUT = args.out_dir
os.makedirs(OUT, exist_ok=True)

# Load all 48 cells
rows = []
glob_pattern = f"{RESULTS}/8B/Step5_Main/*/*/*/*/*/metrics.json"
for p in glob.glob(glob_pattern):
    parts = p.split("/")
    # Strip the results-dir prefix (which may contain '/') so indexing into
    # parts[3..7] for baseline/workload/load/fault/seed still works.
    n_prefix = len(RESULTS.rstrip("/").split("/"))
    baseline = parts[n_prefix + 2]
    workload = parts[n_prefix + 3]
    fault    = parts[n_prefix + 5]
    seed     = int(parts[n_prefix + 6])
    m = json.load(open(p))
    rows.append({
        "baseline": baseline, "workload": workload, "fault": fault, "seed": seed,
        "goodput": m["goodput"],
        "completion": m["completion_rate"],
        "slo_viol": m["slo_violation_rate"],
        "ttft_p95": m["ttft_p95_ms"],
        "tpot_p95": m["tpot_p95_ms"],
    })

ORDER = ["NoFT-Reprefill", "Periodic-Low", "Our-System-NoCkpt", "Our-System"]
LABEL = {"NoFT-Reprefill":"NR (baseline)", "Periodic-Low":"Periodic-Low",
         "Our-System-NoCkpt":"V2-NoCkpt", "Our-System":"V2-full"}
COLOR = {"NoFT-Reprefill":"#666666", "Periodic-Low":"#d8a838",
         "Our-System-NoCkpt":"#3578e5", "Our-System":"#c0392b"}

def agg_mean(metric, baseline, workload, fault):
    vals = [r[metric] for r in rows
            if r["baseline"]==baseline and r["workload"]==workload and r["fault"]==fault]
    return np.mean(vals) if vals else 0

def agg_std(metric, baseline, workload, fault):
    vals = [r[metric] for r in rows
            if r["baseline"]==baseline and r["workload"]==workload and r["fault"]==fault]
    return np.std(vals) if vals else 0


# --- Figure 1: Goodput bar chart, 2x2 grid (workload x fault) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
for i, wl in enumerate(["W6_NarrativeQA", "W8_LongMix"]):
    for j, fault in enumerate(["none", "F2_Mid"]):
        ax = axes[i][j]
        means = [agg_mean("goodput", b, wl, fault) for b in ORDER]
        stds  = [agg_std("goodput",  b, wl, fault) for b in ORDER]
        colors = [COLOR[b] for b in ORDER]
        x = np.arange(len(ORDER))
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[b] for b in ORDER], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Goodput (tok/s)", fontsize=9)
        title = f"{wl}  /  fault={fault}"
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        # Annotate values
        for xi, m in zip(x, means):
            ax.text(xi, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
plt.suptitle("Goodput across baselines (mean ± std over 3 seeds)", fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(f"{OUT}/goodput_2x2.png", dpi=130, bbox_inches="tight")
plt.close()


# --- Figure 2: Completion rate ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for j, wl in enumerate(["W6_NarrativeQA", "W8_LongMix"]):
    ax = axes[j]
    width = 0.35
    x = np.arange(len(ORDER))
    none_means = [agg_mean("completion", b, wl, "none")*100 for b in ORDER]
    f2_means   = [agg_mean("completion", b, wl, "F2_Mid")*100 for b in ORDER]
    ax.bar(x-width/2, none_means, width, label="no fault", color="#888888", edgecolor="black", lw=0.5)
    ax.bar(x+width/2, f2_means,   width, label="F2_Mid",   color="#c0392b", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[b] for b in ORDER], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Completion rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(wl)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for xi, m in zip(x-width/2, none_means):
        ax.text(xi, m+1, f"{m:.0f}", ha="center", fontsize=8)
    for xi, m in zip(x+width/2, f2_means):
        ax.text(xi, m+1, f"{m:.0f}", ha="center", fontsize=8)
plt.suptitle("Completion rate (% of admitted requests that finished)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/completion_rate.png", dpi=130, bbox_inches="tight")
plt.close()


# --- Figure 3: TPOT p95 (the smoking gun) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for j, wl in enumerate(["W6_NarrativeQA", "W8_LongMix"]):
    ax = axes[j]
    width = 0.35
    x = np.arange(len(ORDER))
    none_means = [agg_mean("tpot_p95", b, wl, "none") for b in ORDER]
    f2_means   = [agg_mean("tpot_p95", b, wl, "F2_Mid") for b in ORDER]
    ax.bar(x-width/2, none_means, width, label="no fault", color="#888888", edgecolor="black", lw=0.5)
    ax.bar(x+width/2, f2_means,   width, label="F2_Mid",   color="#c0392b", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[b] for b in ORDER], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("TPOT p95 (ms)")
    ax.set_title(wl)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.axhline(y=200, color="green", linestyle="--", alpha=0.6, label="SLO=200ms")
    for xi, m in zip(x-width/2, none_means):
        ax.text(xi, m*1.1, f"{m:.0f}", ha="center", fontsize=8)
    for xi, m in zip(x+width/2, f2_means):
        ax.text(xi, m*1.1, f"{m:.0f}", ha="center", fontsize=8)
plt.suptitle("TPOT p95 (log scale) — V2-full's smoking gun", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/tpot_p95.png", dpi=130, bbox_inches="tight")
plt.close()


# --- Figure 4: Per-seed dotplot V2 vs NR ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
markers = {"none":"o", "F2_Mid":"X"}
for j, wl in enumerate(["W6_NarrativeQA", "W8_LongMix"]):
    ax = axes[j]
    for fault in ["none", "F2_Mid"]:
        for seed in [42, 123, 456]:
            nr = next((r["goodput"] for r in rows if r["baseline"]=="NoFT-Reprefill"
                      and r["workload"]==wl and r["fault"]==fault and r["seed"]==seed), None)
            v2 = next((r["goodput"] for r in rows if r["baseline"]=="Our-System"
                      and r["workload"]==wl and r["fault"]==fault and r["seed"]==seed), None)
            if nr is None or v2 is None: continue
            color = "#666666" if fault=="none" else "#c0392b"
            ax.plot([0, 1], [nr, v2], color=color, alpha=0.6, marker=markers[fault], markersize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NR", "V2-full"], fontsize=10)
    ax.set_ylabel("Goodput (tok/s)")
    ax.set_title(wl)
    ax.grid(alpha=0.3)
    # Custom legend
    ax.plot([], [], "o", color="#666666", label="no fault")
    ax.plot([], [], "X", color="#c0392b", label="F2_Mid")
    ax.legend(fontsize=9)
plt.suptitle("Per-seed paired comparison: NR vs V2-full (every seed: V2 loses)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/paired_seeds.png", dpi=130, bbox_inches="tight")
plt.close()


# --- Figure 5: SLO violation rate ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for j, wl in enumerate(["W6_NarrativeQA", "W8_LongMix"]):
    ax = axes[j]
    width = 0.35
    x = np.arange(len(ORDER))
    none_means = [agg_mean("slo_viol", b, wl, "none")*100 for b in ORDER]
    f2_means   = [agg_mean("slo_viol", b, wl, "F2_Mid")*100 for b in ORDER]
    ax.bar(x-width/2, none_means, width, label="no fault", color="#888888", edgecolor="black", lw=0.5)
    ax.bar(x+width/2, f2_means,   width, label="F2_Mid",   color="#c0392b", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[b] for b in ORDER], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("SLO violation rate (%)")
    ax.set_title(wl)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("SLO violation rate", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/slo_violation.png", dpi=130, bbox_inches="tight")
plt.close()

print("Figures saved to", OUT)
for f in os.listdir(OUT):
    print(" ", f)
