#!/usr/bin/env bash
# ============================================================================
# overnight_2026-04-14_extra.sh — extended waves (runs AFTER main script)
#
# Waits for overnight_2026-04-14.sh to finish, then runs additional experiments
# to fill the remaining 3-4 hours of overnight time.
#
# Waves 7-11: statistical power + coverage + more tuning
#
# Auto-skips any run that already has metrics.json (safe to rerun).
# ============================================================================

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

# Base env (same as main script)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1
export FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1
export FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1
export FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1
export FT_RECOVERY_PREBUDGET=1
export FT_CKPT_WARMUP_TOKENS=50

CONFIG="experiments_v2/config_8b.yaml"
ROOT="results_v2/8B/overnight_2026-04-14"

run_one() {
    local tag="$1" cell="$2" seed="$3" extra_env="${4:-}" gpu="$5" port="$6"
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
    python3 -c "
import json; m=json.load(open('${out}/metrics.json'))
print(f'  ${tag}/${cell}/s${seed}: gp={m[\"goodput\"]:.1f} slo={m[\"slo_violation_rate\"]*100:.1f}% comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  ${tag}/${cell}/s${seed}: FAILED" >&2
}

# NR run (different baseline)
run_nr() {
    local cell="$1" seed="$2" gpu="$3" port="$4"
    local out="${ROOT}/w10_nr_baseline/${cell}_F2_Mid/${seed}"
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "import json; m=json.load(open('${out}/metrics.json')); print(f'  [skip] NR/${cell}/s${seed}: gp={m[\"goodput\"]:.1f}')" 2>/dev/null || true
        return 0
    fi
    mkdir -p "$out"
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] NR/${cell}/s${seed} on GPU${gpu}" >&2
    CUDA_VISIBLE_DEVICES=${gpu} FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
        --config ${CONFIG} --baseline NoFT-Reprefill \
        --workload W1_Chat --load ${cell} --fault F2_Mid \
        --seed ${seed} --port ${port} \
        --output-dir "${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json; m=json.load(open('${out}/metrics.json'))
print(f'  NR/${cell}/s${seed}: gp={m[\"goodput\"]:.1f}')
" 2>/dev/null || echo "  NR/${cell}/s${seed}: FAILED" >&2
}

echo "########################################################################"
echo "# Overnight Extra — waiting for main script at $(date)"
echo "########################################################################"

# Wait for primary script to finish (check every 60 seconds).
# Use anchored regex "\.sh$" to match only overnight_2026-04-14.sh
# and NOT overnight_2026-04-14_extra.sh (which is this script, would
# otherwise cause infinite self-match deadlock).
while pgrep -f "overnight_2026-04-14\.sh$" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] Main script done — starting extra waves"

# Small safety delay to let GPUs fully release
sleep 5

# ── Wave 7: Extended 5-seed for top combo + baseline ────────────
# Adds seeds 789/2024/7777 to wave 5 candidates for 6-seed total
echo ""
echo "=== Wave 7: Extended seeds for top candidates ==="
for seed in 789 2024 7777; do
    (
        run_one "w7_candA" "Heavy" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "4,5" "8400"
        run_one "w7_candB" "Heavy" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "4,5" "8400"
    ) &
    HP=$!
    (
        run_one "w7_candA" "Moderate" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "6,7" "8500"
        run_one "w7_candB" "Moderate" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done

# ── Wave 8: Light cell regression test ────────────────────────────
# Make sure new opts don't regress on light load
echo ""
echo "=== Wave 8: Light cell with winner configs ==="
for seed in 42 123 456; do
    (
        run_one "w8_baseline" "Light" "$seed" "" "4,5" "8400"
        run_one "w8_candA" "Light" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.15" "4,5" "8400"
    ) &
    HP=$!
    (
        run_one "w8_candB" "Light" "$seed" "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done

# ── Wave 9: LOAD_GUARD tuning (s42) ──────────────────────────────
# LOAD_GUARD controls when to skip save at high batch utilization.
# Current 0.7 — try 0.5/0.6/0.8 to see if there's a better threshold.
echo ""
echo "=== Wave 9: LOAD_GUARD scan (s42) ==="
for lg in 0.50 0.60 0.80; do
    (
        run_one "w9_lg${lg}" "Heavy" "42" "FT_CKPT_LOAD_GUARD=${lg}" "4,5" "8400"
    ) &
    HP=$!
    (
        run_one "w9_lg${lg}" "Moderate" "42" "FT_CKPT_LOAD_GUARD=${lg}" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done

# ── Wave 10: NR baseline extended seeds ───────────────────────────
# Our-System had 6 seeds now (42/123/456 + 789/2024/7777); NR only has 3.
# Make them comparable for paper stats by adding 3 more NR seeds.
echo ""
echo "=== Wave 10: NR baseline extended seeds ==="
for seed in 789 2024 7777; do
    (
        run_nr "Heavy" "$seed" "4,5" "8400"
    ) &
    HP=$!
    (
        run_nr "Moderate" "$seed" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done

# ── Wave 11: "Best of best" combo validation ────────────────────
# Combines all Phase B winners + best of {tail, slo, load_guard} from this night
# on 3 seeds. Uses CONSERVATIVE defaults (tail=0.10, slo=0.85, load=0.7).
# If Wave 2/3/9 show a better value, manual follow-up can swap it in tomorrow.
echo ""
echo "=== Wave 11: All-stack super-combo 3-seed ==="
for seed in 42 123 456; do
    (
        run_one "w11_all_stack" "Heavy" "$seed" \
            "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "4,5" "8400"
    ) &
    HP=$!
    (
        run_one "w11_all_stack" "Moderate" "$seed" \
            "FT_CKPT_TAIL_SKIP_FRAC=0.10 FT_CKPT_SLO_GUARD=0.85" "6,7" "8500"
    ) &
    MP=$!
    wait $HP $MP
done

echo ""
echo "########################################################################"
echo "# Extra waves complete — $(date)"
echo "########################################################################"

# Final summary (all waves: 1-11)
python3 << 'PYEOF'
import json, glob, os, statistics
print("\n=== FULL Overnight 2026-04-14 Results (all waves) ===\n")

def load(f):
    return json.load(open(f)) if os.path.exists(f) else None

for wave_prefix, label in [
    ("w1_", "Wave 1: Tail-skip scan"),
    ("w2_", "Wave 2: SLO-guard scan"),
    ("w3_", "Wave 3: tail+SLO combos"),
    ("w4_", "Wave 4: Fine warmup scan"),
    ("w5_", "Wave 5: Top candidates 3-seed"),
    ("w6_", "Wave 6: Baseline extra seeds"),
    ("w7_", "Wave 7: Top candidates extra seeds"),
    ("w8_", "Wave 8: Light cell"),
    ("w9_", "Wave 9: LOAD_GUARD scan"),
    ("w10_", "Wave 10: NR extra seeds"),
    ("w11_", "Wave 11: All-stack super-combo 3-seed"),
]:
    print(f"\n--- {label} ---")
    rows = {}
    for f in sorted(glob.glob(f"results_v2/8B/overnight_2026-04-14/{wave_prefix}*/*/*/metrics.json")):
        parts = f.split("/")
        tag = parts[-4]
        cell = parts[-3]
        seed = parts[-2]
        m = load(f)
        if m:
            key = (tag, cell)
            rows.setdefault(key, []).append(m["goodput"])
    for (tag, cell), vals in sorted(rows.items()):
        mn = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {cell:<10} {tag:<28} mean={mn:6.1f} ±{sd:5.1f} (n={len(vals)})")
PYEOF
