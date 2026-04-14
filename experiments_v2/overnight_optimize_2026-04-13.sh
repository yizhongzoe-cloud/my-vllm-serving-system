#!/usr/bin/env bash
# ============================================================================
# overnight_optimize_2026-04-13.sh — Autonomous checkpoint optimization loop
#
# Runs until ~10 AM. Sequential on GPU 6-7 to avoid /dev/shm race.
#
# Phase 1: Wait for gil_reduce to finish, find best variant
# Phase 2: Try more step intervals (2, 4, 5) with best config
# Phase 3: Run best config on 3 seeds × 2 cells (Heavy/F2 + Moderate/F2)
# Phase 4: Run 12-cell sweep with best config for paper table
# ============================================================================

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1
export FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1

CONFIG="experiments_v2/config_8b.yaml"
GPU="6,7"
PORT="8500"

run_one() {
    local tag="$1"
    local workload="$2"
    local load="$3"
    local fault="$4"
    local seed="$5"
    local extra_env="${6:-}"
    local out="results_v2/8B/overnight_opt/${tag}/${workload}/${load}/${fault}/${seed}"

    if [ -f "${out}/metrics.json" ]; then
        python3 -c "import json; m=json.load(open('${out}/metrics.json')); print(f'  [skip] ${tag}/${workload}/${load}/${fault}/s${seed}: {m[\"goodput\"]:.1f}')"
        return 0
    fi

    mkdir -p "$out"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%H:%M:%S)] ${tag}/${workload}/${load}/${fault}/s${seed}" >&2

    eval "CUDA_VISIBLE_DEVICES=${GPU} ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} --baseline Our-System \
        --workload ${workload} --load ${load} --fault ${fault} \
        --seed ${seed} --port ${PORT} \
        --output-dir ${out}" > "${out}/stdout.log" 2>&1

    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json; m=json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} slo={m[\"slo_violation_rate\"]*100:.1f}% comp={m[\"completion_rate\"]*100:.0f}%')
" 2>/dev/null
    else
        echo "  ${tag}/s${seed}: FAILED" >&2
    fi
}

echo "########################################################################"
echo "# Overnight optimization — $(date)"
echo "########################################################################"

# ── Phase 1: Wait for gil_reduce to finish ───────────────────────
echo ""
echo "=== Phase 1: Waiting for gil_reduce to finish ==="
while pgrep -f "gil_reduce_ab" > /dev/null 2>&1; do
    sleep 30
done
sleep 5
echo "gil_reduce done."

# ── Phase 2: Try more step intervals ────────────────────────────
echo ""
echo "=== Phase 2: Step interval sweep (Heavy/F2_Mid s42) ==="
for interval in 2 4 5; do
    run_one "interval${interval}" "W1_Chat" "Heavy" "F2_Mid" "42" "FT_CKPT_STEP_INTERVAL=${interval}"
done
# Also try combined (merge + interval) with different intervals
for interval in 2 4 5; do
    run_one "combined_i${interval}" "W1_Chat" "Heavy" "F2_Mid" "42" "FT_CKPT_STEP_INTERVAL=${interval} FT_MERGE_LAYER_WRITE=1"
done

# ── Phase 2b: Find best variant so far ──────────────────────────
echo ""
echo "=== Phase 2b: Finding best variant ==="
python3 << 'PYEOF'
import json, glob
best_tag = "all_stack"
best_gp = 0
for f in glob.glob("results_v2/8B/overnight_opt/*/W1_Chat/Heavy/F2_Mid/42/metrics.json"):
    try:
        m = json.load(open(f))
        tag = f.split("/")[3]
        gp = m.get("goodput", 0)
        print(f"  {tag}: {gp:.1f}")
        if gp > best_gp:
            best_gp = gp
            best_tag = tag
    except: pass
# Also check gil_reduce results
for f in glob.glob("results_v2/8B/gil_reduce_ab/*/42/metrics.json"):
    try:
        m = json.load(open(f))
        tag = f.split("/")[3]
        gp = m.get("goodput", 0)
        print(f"  gil_reduce/{tag}: {gp:.1f}")
        if gp > best_gp:
            best_gp = gp
            best_tag = f"gil_reduce/{tag}"
    except: pass
print(f"\n  BEST: {best_tag} = {best_gp:.1f}")
with open("/tmp/best_ckpt_tag.txt", "w") as f:
    f.write(best_tag)
PYEOF

# ── Phase 3: 3-seed validation on best variant ──────────────────
echo ""
echo "=== Phase 3: 3-seed validation (Heavy/F2_Mid + Moderate/F2_Mid) ==="

# Determine best env vars
BEST_TAG=$(cat /tmp/best_ckpt_tag.txt 2>/dev/null || echo "base")
BEST_ENV=""
case "$BEST_TAG" in
    *interval2*) BEST_ENV="FT_CKPT_STEP_INTERVAL=2" ;;
    *interval3*) BEST_ENV="FT_CKPT_STEP_INTERVAL=3" ;;
    *interval4*) BEST_ENV="FT_CKPT_STEP_INTERVAL=4" ;;
    *interval5*) BEST_ENV="FT_CKPT_STEP_INTERVAL=5" ;;
    *combined_i2*) BEST_ENV="FT_CKPT_STEP_INTERVAL=2 FT_MERGE_LAYER_WRITE=1" ;;
    *combined_i3*) BEST_ENV="FT_CKPT_STEP_INTERVAL=3 FT_MERGE_LAYER_WRITE=1" ;;
    *combined_i4*) BEST_ENV="FT_CKPT_STEP_INTERVAL=4 FT_MERGE_LAYER_WRITE=1" ;;
    *combined_i5*) BEST_ENV="FT_CKPT_STEP_INTERVAL=5 FT_MERGE_LAYER_WRITE=1" ;;
    *merge*) BEST_ENV="FT_MERGE_LAYER_WRITE=1" ;;
    *) BEST_ENV="" ;;
esac
echo "Best variant: ${BEST_TAG}, env: ${BEST_ENV:-none}"

for seed in 42 123 456; do
    run_one "best" "W1_Chat" "Heavy" "F2_Mid" "$seed" "$BEST_ENV"
    run_one "best" "W1_Chat" "Moderate" "F2_Mid" "$seed" "$BEST_ENV"
done

# Also run NoFT-Reprefill for pair comparison on Moderate
for seed in 42 123 456; do
    out="results_v2/8B/overnight_opt/noft_reprefill/W1_Chat/Moderate/F2_Mid/${seed}"
    if [ ! -f "${out}/metrics.json" ]; then
        mkdir -p "$out"
        rm -rf /dev/shm/vllm_ft_checkpoints
        echo "[$(date +%H:%M:%S)] NR/Moderate/F2/s${seed}"
        CUDA_VISIBLE_DEVICES=${GPU} FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
            --config ${CONFIG} --baseline NoFT-Reprefill \
            --workload W1_Chat --load Moderate --fault F2_Mid \
            --seed ${seed} --port ${PORT} \
            --output-dir "$out" > "${out}/stdout.log" 2>&1
        python3 -c "import json; m=json.load(open('${out}/metrics.json')); print(f'  NR/Mod/F2/s${seed}: {m[\"goodput\"]:.1f}')" 2>/dev/null
    fi
done

# ── Phase 4: 12-cell sweep with best config ─────────────────────
echo ""
echo "=== Phase 4: 12-cell sweep (best config) ==="
for seed in 42 123 456; do
    for load in Light Moderate Heavy; do
        for fault in none F2_Mid; do
            run_one "best_sweep" "W1_Chat" "$load" "$fault" "$seed" "$BEST_ENV"
        done
    done
done

# ── Phase 5: Summary ────────────────────────────────────────────
echo ""
echo "########################################################################"
echo "# Overnight optimization finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics

print("\n=== Results Summary ===\n")

# Phase 2: interval sweep
print("--- Phase 2: Step interval sweep (s42 Heavy/F2_Mid) ---")
for tag in sorted(glob.glob("results_v2/8B/overnight_opt/*/W1_Chat/Heavy/F2_Mid/42/metrics.json")):
    try:
        m = json.load(open(tag))
        name = tag.split("/")[3]
        print(f"  {name:<20} gp={m['goodput']:.1f}  slo={m['slo_violation_rate']*100:.1f}%")
    except: pass

# Phase 3: 3-seed validation
print("\n--- Phase 3: 3-seed best config ---")
for cell in ["Heavy/F2_Mid", "Moderate/F2_Mid"]:
    load, fault = cell.split("/")
    for tag_name in ["best", "noft_reprefill"]:
        vals = []
        for seed in [42, 123, 456]:
            f = f"results_v2/8B/overnight_opt/{tag_name}/W1_Chat/{load}/{fault}/{seed}/metrics.json"
            try:
                m = json.load(open(f))
                vals.append(m["goodput"])
            except: pass
        if vals:
            m_ = statistics.mean(vals)
            s_ = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {tag_name:<20} {cell}: mean={m_:.1f}±{s_:.0f} (n={len(vals)})")

# Phase 4: 12-cell sweep
print("\n--- Phase 4: 12-cell sweep (best config) ---")
cells = {}
for f in sorted(glob.glob("results_v2/8B/overnight_opt/best_sweep/W1_Chat/*/*/*/metrics.json")):
    parts = f.split("/")
    cell = f"{parts[-4]}/{parts[-3]}"
    try:
        m = json.load(open(f))
        cells.setdefault(cell, []).append(m["goodput"])
    except: pass
overall = []
for cell in sorted(cells):
    vals = cells[cell]
    m_ = statistics.mean(vals)
    overall.append(m_)
    print(f"  {cell:<25} mean={m_:.1f} (n={len(vals)})")
if overall:
    print(f"  {'OVERALL':<25} mean={statistics.mean(overall):.1f}")
PYEOF
