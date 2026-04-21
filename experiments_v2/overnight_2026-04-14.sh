#!/usr/bin/env bash
# ============================================================================
# overnight_2026-04-14.sh — comprehensive A/B for remaining optimizations
#
# Goal: find improvements on top of current winner
#   (prebudget + warmup=50, yielded Heavy +7.9 vs NR)
#
# Plan (6 waves, ~7-8 hours total wall clock):
#   Wave 1: Tail-skip (FT_CKPT_TAIL_SKIP_FRAC) scan: 0.10/0.15/0.20/0.25
#   Wave 2: SLO-guard (FT_CKPT_SLO_GUARD) scan: 0.85/0.80/0.75/0.65
#   Wave 3: Best tail-skip + SLO-guard combos (top 2 × top 2)
#   Wave 4: Fine warmup scan (40/45/55/60) near current 50 sweet-spot
#   Wave 5: Top combo 3-seed validation (Heavy + Moderate, seeds 42/123/456)
#   Wave 6: Extended seeds for current winner (warmup=50 baseline)
#           on seeds 789/2024/7777 to tighten statistical bounds
#
# All experiments env-gated — no code commits needed tonight.
# Results in: results_v2/8B/overnight_2026-04-14/
#
# GPU assignment (parallel):
#   GPU 4-5 (port 8400): Heavy cells
#   GPU 6-7 (port 8500): Moderate cells (Light optional)
# ============================================================================

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

# Base env (all_stack + Phase B winners — kept constant across waves)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1
export FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1
export FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1
export FT_RECOVERY_PREBUDGET=1
export FT_CKPT_WARMUP_TOKENS=50  # current winner — may be overridden per-config

CONFIG="experiments_v2/config_8b.yaml"
ROOT="results_v2/8B/overnight_2026-04-14"
mkdir -p "$ROOT"

# run_one tag cell seed extra_env gpu port
run_one() {
    local tag="$1"
    local cell="$2"
    local seed="$3"
    local extra_env="${4:-}"
    local gpu="$5"
    local port="$6"
    local out="${ROOT}/${tag}/${cell}_F2_Mid/${seed}"

    if [ -f "${out}/metrics.json" ]; then
        python3 -c "import json; m=json.load(open('${out}/metrics.json')); print(f'  [skip] ${tag}/${cell}/s${seed}: gp={m[\"goodput\"]:.1f}')" 2>/dev/null || true
        return 0
    fi

    mkdir -p "$out"
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null || true

    echo "[$(date +%H:%M:%S)] ${tag}/${cell}/s${seed} on GPU${gpu}" >&2

    eval "CUDA_VISIBLE_DEVICES=${gpu} ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} --baseline Our-System \
        --workload W1_Chat --load ${cell} --fault F2_Mid \
        --seed ${seed} --port ${port} \
        --output-dir ${out}" > "${out}/stdout.log" 2>&1

    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json; m=json.load(open('${out}/metrics.json'))
print(f'  ${tag}/${cell}/s${seed}: gp={m[\"goodput\"]:.1f} slo={m[\"slo_violation_rate\"]*100:.1f}% comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null
    else
        echo "  ${tag}/${cell}/s${seed}: FAILED" >&2
    fi
}

echo "########################################################################"
echo "# Overnight A/B — $(date)"
echo "########################################################################"

# ── Wave 1: Tail-skip scan (s42) ────────────────────────────────
# Heavy on GPU 4-5, Moderate on GPU 6-7 — run sequentially on each GPU,
# but parallel across the two cells
echo ""
echo "=== Wave 1: Tail-skip scan (s42 single-seed) ==="
(
    for frac in 0.10 0.15 0.20 0.25; do
        run_one "w1_tail${frac}" "Heavy" "42" "FT_CKPT_TAIL_SKIP_FRAC=${frac}" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] Wave 1 Heavy done" >&2
) &
HEAVY_PID=$!
(
    for frac in 0.10 0.15 0.20 0.25; do
        run_one "w1_tail${frac}" "Moderate" "42" "FT_CKPT_TAIL_SKIP_FRAC=${frac}" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] Wave 1 Moderate done" >&2
) &
MOD_PID=$!
wait $HEAVY_PID $MOD_PID

# ── Wave 2: SLO-guard scan (s42) ────────────────────────────────
echo ""
echo "=== Wave 2: SLO-guard scan (s42 single-seed) ==="
(
    for slo in 0.85 0.80 0.75 0.65; do
        run_one "w2_slo${slo}" "Heavy" "42" "FT_CKPT_SLO_GUARD=${slo}" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] Wave 2 Heavy done" >&2
) &
HP=$!
(
    for slo in 0.85 0.80 0.75 0.65; do
        run_one "w2_slo${slo}" "Moderate" "42" "FT_CKPT_SLO_GUARD=${slo}" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] Wave 2 Moderate done" >&2
) &
MP=$!
wait $HP $MP

# ── Wave 3: Combined tail-skip + SLO-guard (s42) ────────────────
# 4 combos: {tail=0.10, 0.15} × {slo=0.85, 0.80}
echo ""
echo "=== Wave 3: tail+SLO combos (s42 single-seed) ==="
(
    for combo in "0.10_0.85" "0.10_0.80" "0.15_0.85" "0.15_0.80"; do
        tail="${combo%_*}"; slo="${combo#*_}"
        run_one "w3_t${tail}_s${slo}" "Heavy" "42" \
            "FT_CKPT_TAIL_SKIP_FRAC=${tail} FT_CKPT_SLO_GUARD=${slo}" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] Wave 3 Heavy done" >&2
) &
HP=$!
(
    for combo in "0.10_0.85" "0.10_0.80" "0.15_0.85" "0.15_0.80"; do
        tail="${combo%_*}"; slo="${combo#*_}"
        run_one "w3_t${tail}_s${slo}" "Moderate" "42" \
            "FT_CKPT_TAIL_SKIP_FRAC=${tail} FT_CKPT_SLO_GUARD=${slo}" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] Wave 3 Moderate done" >&2
) &
MP=$!
wait $HP $MP

# ── Wave 4: Fine warmup scan (s42) ──────────────────────────────
# Test values near current sweet-spot 50 to find finer-grained optimum
echo ""
echo "=== Wave 4: Fine warmup scan (s42 single-seed) ==="
(
    for w in 40 45 55 60; do
        run_one "w4_warmup${w}" "Heavy" "42" "FT_CKPT_WARMUP_TOKENS=${w}" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] Wave 4 Heavy done" >&2
) &
HP=$!
(
    for w in 40 45 55 60; do
        run_one "w4_warmup${w}" "Moderate" "42" "FT_CKPT_WARMUP_TOKENS=${w}" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] Wave 4 Moderate done" >&2
) &
MP=$!
wait $HP $MP

# ── Wave 5: 3-seed on top candidates ────────────────────────────
# Will pick winners dynamically by parsing wave 1-4 results.
# For simplicity: hard-code two promising combos (conservative + aggressive)
echo ""
echo "=== Wave 5: 3-seed validation of top candidates ==="
# Candidate A: tail=0.15 + warmup=50 (already have from w1)
# Candidate B: tail=0.10 + slo=0.85 + warmup=50 (expected top combo)
for seed in 123 456; do
    (
        run_one "w5_candA" "Heavy" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "4,5" "8400"
        run_one "w5_candB" "Heavy" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "4,5" "8400"
    ) &
    HP=$!
    (
        run_one "w5_candA" "Moderate" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "6,7" "8500"
        run_one "w5_candB" "Moderate" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done
# Also run s42 for candidates (to have full 3-seed data)
(
    run_one "w5_candA" "Heavy" "42" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "4,5" "8400"
    run_one "w5_candB" "Heavy" "42" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "4,5" "8400"
) &
HP=$!
(
    run_one "w5_candA" "Moderate" "42" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "6,7" "8500"
    run_one "w5_candB" "Moderate" "42" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "6,7" "8500"
) &
MP=$!
wait $HP $MP

# ── Wave 6: Extended seeds for current winner (warmup=50 alone) ──
# Tightens 3-seed CI by adding 3 more seeds
echo ""
echo "=== Wave 6: Extended seeds for current winner (warmup=50) ==="
for seed in 789 2024 7777; do
    (
        run_one "w6_baseline" "Heavy" "$seed" "" "4,5" "8400"
    ) &
    HP=$!
    (
        run_one "w6_baseline" "Moderate" "$seed" "" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done

echo ""
echo "########################################################################"
echo "# Overnight complete — $(date)"
echo "########################################################################"

# ── Final summary ──
python3 << 'PYEOF'
import json, glob, os, statistics
print("\n=== Overnight 2026-04-14 Results Summary ===\n")

def load(f):
    return json.load(open(f)) if os.path.exists(f) else None

def summary_cell(wave_prefix, cell_name):
    rows = []
    for tag_dir in sorted(glob.glob(f"results_v2/8B/overnight_2026-04-14/{wave_prefix}*/{cell_name}_F2_Mid")):
        tag = tag_dir.split("/")[-2]
        vals = []
        for seed_dir in sorted(glob.glob(f"{tag_dir}/*")):
            m = load(f"{seed_dir}/metrics.json")
            if m:
                vals.append(m["goodput"])
        if vals:
            mn = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0
            rows.append((tag, mn, sd, len(vals)))
    return rows

for wave, label in [
    ("w1", "Wave 1: Tail-skip scan"),
    ("w2", "Wave 2: SLO-guard scan"),
    ("w3", "Wave 3: Combined"),
    ("w4", "Wave 4: Fine warmup"),
    ("w5", "Wave 5: 3-seed top candidates"),
    ("w6", "Wave 6: Extended seeds (baseline)"),
]:
    print(f"\n--- {label} ---")
    for cell in ["Heavy", "Moderate"]:
        rows = summary_cell(wave, cell)
        if rows:
            print(f"  {cell}:")
            for tag, mn, sd, n in rows:
                print(f"    {tag:<25} mean={mn:6.1f} ±{sd:5.1f} (n={n})")
PYEOF
