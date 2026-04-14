#!/usr/bin/env bash
# W3_Instruct quick scan — does Our-System (KV reload) win in tight-SLO
# decode-heavy workloads?
#
# W3_Instruct: alpaca dataset, prompt mean ~16 tokens (ultrashort),
#              tpot_slo_ms=50 (tight), decode-heavy workload.
#
# Theory: short prompts → re-prefill is nearly free → KV reload has NO
# advantage. BUT tight tpot_slo → the decode gap during recovery might
# be more SLO-sensitive → KV reload's faster first-token-after-recovery
# could help?
#
# 3 baselines × 2 cells × 3 seeds = 18 runs, split 2 phases:
#   Phase A (parallel GPU 4-5 + 6-7): Our-System + NoFT-Reprefill = 12 runs
#     GPU 4-5: Our-System     (writes /dev/shm checkpoints)
#     GPU 6-7: NoFT-Reprefill (no checkpoint, safe to parallel)
#   Phase B (sequential GPU 4-5): Periodic-High = 6 runs
#     (writes checkpoints, must not parallel with phase A Our-System)
#
# Our-System uses best-known config: FT_ASYNC_RESTORE=1 + FT_GATED_SOLVER=1
#
# Total: ~42 min (phase A ~21 min parallel + phase B ~21 min sequential)

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1

OUT_BASE="results_v2/8B/w3_instruct_scan"
CONFIG="experiments_v2/config_8b.yaml"
SEEDS=(42 123 456)
WORKLOAD="W3_Instruct"
CELLS=("Heavy/F2_Mid" "Moderate/F2_Mid")

run_cell() {
    local baseline="$1"
    local load="$2"
    local fault="$3"
    local seed="$4"
    local gpu="$5"
    local port="$6"
    local extra_env="${7:-}"

    local out_dir="${OUT_BASE}/${baseline}/${WORKLOAD}/${load}/${fault}/${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        return 0
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%H:%M:%S)] START ${baseline}/${WORKLOAD}/${load}/${fault}/s${seed} GPU=${gpu}" >&2

    eval "CUDA_VISIBLE_DEVICES=${gpu} ${extra_env} python experiments_v2/run.py \
        --config ${CONFIG} \
        --baseline ${baseline} \
        --workload ${WORKLOAD} \
        --load ${load} \
        --fault ${fault} \
        --seed ${seed} \
        --port ${port} \
        --output-dir ${out_dir}" \
        > "${out_dir}/stdout.log" 2>&1

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  ${baseline}/${load}/${fault}/s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  tpot_p50={m.get(\"tpot_p50_ms\",-1):.1f}  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%'
)
" 2>/dev/null
    else
        echo "  ${baseline}/${load}/${fault}/s${seed}: FAILED" >&2
    fi
}

echo "########################################################################"
echo "# W3_Instruct quick scan — $(date)"
echo "# W3: alpaca, prompt~16 tok, tpot_slo=50ms, decode-heavy"
echo "# Cells: Heavy/F2_Mid + Moderate/F2_Mid"
echo "# Baselines: Our-System (C1+A3) | NoFT-Reprefill | Periodic-High"
echo "########################################################################"

# ── Phase A: Our-System (GPU 4-5) + NoFT-Reprefill (GPU 6-7) parallel ──
phase_a_our_system() {
    for cell in "${CELLS[@]}"; do
        local load="${cell%/*}"
        local fault="${cell#*/}"
        for seed in "${SEEDS[@]}"; do
            run_cell "Our-System" "$load" "$fault" "$seed" "4,5" "8400" \
                "FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1"
        done
    done
}

phase_a_noft_reprefill() {
    for cell in "${CELLS[@]}"; do
        local load="${cell%/*}"
        local fault="${cell#*/}"
        for seed in "${SEEDS[@]}"; do
            run_cell "NoFT-Reprefill" "$load" "$fault" "$seed" "6,7" "8500" \
                "FT_RECOVERY_MODE=reprefill"
        done
    done
}

echo ""
echo "--- Phase A: Our-System (GPU 4-5) + NoFT-Reprefill (GPU 6-7) parallel ---"
phase_a_our_system &
PID_OS=$!
phase_a_noft_reprefill &
PID_NR=$!
echo "Our-System pid=$PID_OS, NoFT-Reprefill pid=$PID_NR"
wait $PID_OS; echo "[$(date +%H:%M:%S)] Our-System done (rc=$?)"
wait $PID_NR; echo "[$(date +%H:%M:%S)] NoFT-Reprefill done (rc=$?)"

# ── Phase B: Periodic-High (GPU 4-5 sequential) ──
echo ""
echo "--- Phase B: Periodic-High (GPU 4-5 sequential) ---"
for cell in "${CELLS[@]}"; do
    local_load="${cell%/*}"
    local_fault="${cell#*/}"
    for seed in "${SEEDS[@]}"; do
        run_cell "Periodic-High" "$local_load" "$local_fault" "$seed" "4,5" "8400"
    done
done

echo ""
echo "########################################################################"
echo "# Finished at $(date)"
echo "########################################################################"

python3 << 'PYEOF'
import json, glob, statistics

base = "results_v2/8B/w3_instruct_scan"
baselines = ["Our-System", "NoFT-Reprefill", "Periodic-High"]
cells = [("Heavy", "F2_Mid"), ("Moderate", "F2_Mid")]

print("\n=== W3_Instruct scan summary ===\n")
for load, fault in cells:
    print(f"--- W3_Instruct/{load}/{fault} (3 seeds) ---")
    print(f"{'baseline':<22} {'mean±std':>16} {'tpot_p50':>10} {'slo%':>8}")
    print("-" * 60)
    for bl in baselines:
        vals = []
        tpots = []
        slos = []
        for f in sorted(glob.glob(f"{base}/{bl}/W3_Instruct/{load}/{fault}/*/metrics.json")):
            try:
                m = json.load(open(f))
                vals.append(m.get("goodput", -1))
                tpots.append(m.get("tpot_p50_ms", -1))
                slos.append(m.get("slo_violation_rate", -1) * 100)
            except: pass
        if len(vals) >= 2:
            m_ = statistics.mean(vals); s_ = statistics.stdev(vals)
            t_ = statistics.mean(tpots); sl_ = statistics.mean(slos)
            print(f"{bl:<22} {m_:>7.1f}±{s_:>4.0f}    {t_:>8.1f}   {sl_:>6.1f}%")
        elif vals:
            print(f"{bl:<22} {vals[0]:>7.1f}         {tpots[0]:>8.1f}   {slos[0]:>6.1f}%")
        else:
            print(f"{bl:<22} (no data)")
    print()
PYEOF
