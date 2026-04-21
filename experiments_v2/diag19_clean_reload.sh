#!/usr/bin/env bash
# Diag19: CRITICAL re-run of reload mode with PROPER cleanup.
# Previous scripts cleaned /dev/shm/vllm_ft_checkpoints_${port} but the
# actual save path is /dev/shm/vllm_ft_checkpoints/ (no port suffix).
# This caused /dev/shm to fill to 252GB (100%), all checkpoint saves to
# fail with ENOSPC, and V2 reload to silently degrade to reprefill.
#
# This script cleans the CORRECT path before every run.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/stress_2026-04-20/diag19"
mkdir -p "$ROOT"

V2_BASE='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100'

run_cell() {
    local tag="$1" gpus="$2" port="$3" baseline="$4" wl="$5" load="$6" seed="$7"
    local out="${ROOT}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    # CRITICAL FIX: clean correct path (no port suffix)
    rm -rf /dev/shm/vllm_ft_checkpoints/ 2>/dev/null
    rm -rf "/dev/shm/vllm_ft_checkpoints_${port}" 2>/dev/null
    sleep 1
    local env_base=""
    if [ "$baseline" = "NoFT-Reprefill" ]; then
        env_base="FT_RECOVERY_MODE=reprefill"
    else
        env_base="$V2_BASE"
    fi
    # Verify /dev/shm has space before launch
    local shm_free_mb=$(df -m /dev/shm | awk 'NR==2 {print $4}')
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed} (shm_free=${shm_free_mb}MB)" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpus} ${env_base} python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        # Check for ENOSPC errors in this run
        local enospc=$(grep -c "No space left" "${out}/server.log" 2>/dev/null || echo 0)
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f} (ENOSPC errors: ${enospc})')
" 2>&1
    fi
}

echo "######## Diag19: CLEAN reload retest $(date) ########"
echo "Initial /dev/shm status:"
df -h /dev/shm | tail -1

# GPU 0-1: W7 V2 reload + NR (6 seeds each for solid paired comparison)
(
    for s in 42 123 456 789 1234 5678; do
        run_cell "W7_V2_reload_clean" "0,1" "8500" "Our-System"     "W7_Saturated" "Heavy" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 0-1 done" >&2
) > /tmp/diag19_gpu01.log 2>&1 &
P1=$!

# GPU 2-3: W5 + W1 clean reload
(
    for s in 42 123 456 789 1234 5678; do
        run_cell "W5_V2_reload_clean" "2,3" "8501" "Our-System"     "W5_LongDoc" "Moderate" "$s" || true
    done
    echo "[$(date +%H:%M:%S)] GPU 2-3 done" >&2
) > /tmp/diag19_gpu23.log 2>&1 &
P2=$!

wait $P1 $P2

echo ""
echo "######## Diag19 FINAL $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/stress_2026-04-20"

# W7 paired per-seed with CLEAN reload
print("=== W7 paired per-seed (CLEAN reload vs NR) ===")
v2_base = f"{ROOT}/diag19/W7_V2_reload_clean"
nr_bases = [
    (f"{ROOT}/w7/W7_NR_Heavy", [42,123,456]),
    (f"{ROOT}/diag18/W7_NR_extra", [789,1234,5678]),
]
def get_seed(base, s):
    p = f"{base}/{s}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        if m['completion_rate'] >= 0.95:
            return (m['goodput'], m.get('failover_gap_p95_ms',0))
    return None
nr = {}
for base, seeds in nr_bases:
    for s in seeds:
        r = get_seed(base, s)
        if r: nr[s] = r

wins_both, wins_gp, wins_fg = 0, 0, 0
paired_gp = []; paired_fg = []
for s in [42,123,456,789,1234,5678]:
    v = get_seed(v2_base, s)
    if v and s in nr:
        v_gp, v_fg = v; n_gp, n_fg = nr[s]
        gp_w = "✓" if v_gp >= n_gp else "✗"
        fg_w = "✓" if v_fg <= n_fg else "✗"
        strict = " 🎯" if (v_gp >= n_gp and v_fg <= n_fg) else ""
        print(f"  s{s}: V2 {v_gp:.1f}/{v_fg:.0f} vs NR {n_gp:.1f}/{n_fg:.0f} [{gp_w}][{fg_w}]{strict}")
        paired_gp.append(v_gp - n_gp); paired_fg.append(v_fg - n_fg)
        if gp_w=='✓' and fg_w=='✓': wins_both += 1
        if gp_w=='✓': wins_gp += 1
        if fg_w=='✓': wins_fg += 1

if paired_gp:
    print(f"\n  Summary: {wins_both}/{len(paired_gp)} strict wins, {wins_gp}/{len(paired_gp)} gp wins, {wins_fg}/{len(paired_gp)} fg_p95 wins")
    print(f"  Mean diff: gp={st.mean(paired_gp):+.1f} fg_p95={st.mean(paired_fg):+.0f}")

# W5 analysis
print("\n=== W5 V2 reload clean ===")
gps, c, f95 = [], [], []
for s in [42,123,456,789,1234,5678]:
    p = f"{ROOT}/diag19/W5_V2_reload_clean/{s}/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))
        print(f"  s{s}: gp={m['goodput']:.1f} comp={m['completion_rate']*100:.0f}% fg_p95={m.get('failover_gap_p95_ms',0):.0f}")
        if m['completion_rate'] >= 0.80:
            gps.append(m['goodput']); c.append(m['completion_rate']*100); f95.append(m.get('failover_gap_p95_ms',0))
if gps:
    print(f"  comp>=80% (n={len(gps)}): gp={st.mean(gps):.1f} fg_p95={st.mean(f95):.0f}")
PYEOF
