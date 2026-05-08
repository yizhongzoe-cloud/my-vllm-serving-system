#!/usr/bin/env bash
# Per-trigger routing 4-cell main experiment (paper §9a).
#
# Cells:
#   A_vanilla            both off, NoFT baseline. Capacity preempt =>
#                        vLLM RECOMPUTE. SLO trigger never preempts.
#   B_capacity_release   FT_CAPACITY_PREEMPT_RELOAD=1, CkptReload.
#                        Capacity => release+reload. SLO trigger off.
#   C_slo_retain         SLO_PRIORITY_PREEMPT=1, NoFT. Capacity =>
#                        vLLM RECOMPUTE. SLO trigger => retain.
#   D_routed             both on, CkptReload. Capacity => release+
#                        reload. SLO trigger => retain.
#
# Workload: W_Ruler16K. Run at 2 RPS levels:
#   0.6  — light load, baseline shouldn't backlog. V3 may not fire much.
#   1.0  — medium-heavy, capacity preempts expected. Was 1.2 in earlier
#          runs but that overloaded all cells (TTFT p95 400+s).
#
# Run duration 600s + warmup 60s for ~10 min of post-warmup measurement
# per cell. Total: 4 cells × 3 seeds × 2 RPS = 24 cells × ~13 min ≈ 5.2h.

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
cd "$ROOT"

CONFIG="${ROOT}/experiments_v2/config_8b_reload.yaml"
OUT_ROOT="${ROOT}/experiments_v2/results_main_4cell"
SHM_DIR="/dev/shm/vllm_ft_checkpoints"
PORT=8492

mkdir -p "${OUT_ROOT}"

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*" | tee -a "${OUT_ROOT}/run.log"; }

cleanup_shm() {
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    mkdir -p "${SHM_DIR}"
}

reset_env() {
    unset FT_CAPACITY_PREEMPT_RELOAD
    unset FT_CAPACITY_PREEMPT_RELOAD_OVERLAP
    unset FT_SLO_PREEMPT
    unset SLO_PRIORITY_PREEMPT
    unset SLO_PRIORITY_PREEMPT_MIN_INTERVAL_MS
    unset SLO_PRIORITY_PREEMPT_PER_REQ_COOLDOWN_MS
    unset SLO_PRIORITY_PREEMPT_MIN_GAP_MS
    unset SLO_PRIORITY_PREEMPT_RETAIN_STEPS
}

# build_config <rps> <out_path>
# Produces a temp YAML config with overridden run_duration, warmup,
# and W_Ruler16K's RPS.
build_config() {
    local rps="$1"
    local outpath="$2"
    python3 - <<EOF
import yaml
cfg = yaml.safe_load(open('${CONFIG}'))
cfg['run_duration_sec'] = 600.0
cfg['warmup_sec'] = 60.0
cfg['load_levels']['Pressure_16K']['rps'] = float(${rps})
yaml.safe_dump(cfg, open('${outpath}', 'w'), sort_keys=False)
EOF
}

# run_cell <cell_name> <baseline_name> <seed> <rps_tag> <config_path>
run_cell() {
    local cell="$1"
    local baseline="$2"
    local seed="$3"
    local rps_tag="$4"
    local cfg="$5"
    local workload="W_Ruler16K"
    local load_level="Pressure_16K"

    local cell_out="${OUT_ROOT}/${rps_tag}/${cell}/${baseline}_${workload}_seed${seed}"
    mkdir -p "${cell_out}"

    log "===== rps=${rps_tag} | Cell ${cell} | baseline=${baseline} | seed=${seed} ====="

    cleanup_shm
    reset_env

    export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
    export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"
    export FT_CUDA_EVENT_PROFILE=1
    export FT_CKPT_STATS_LOG=1
    export FT_DELTA_CHECKPOINT=1
    export FT_USE_FCFS_BASE_QUEUE=1
    export FT_SKIP_SOLVER=1
    export FT_SLO_AWARE_OBJECTIVE=0
    export FT_SLO_PREEMPT=0
    export CUDA_VISIBLE_DEVICES=0

    case "${cell}" in
        A_vanilla)
            export FT_CAPACITY_PREEMPT_RELOAD=0
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=0
            export SLO_PRIORITY_PREEMPT=0
            ;;
        B_capacity_release)
            export FT_CAPACITY_PREEMPT_RELOAD=1
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
            export SLO_PRIORITY_PREEMPT=0
            ;;
        C_slo_retain)
            export FT_CAPACITY_PREEMPT_RELOAD=0
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=0
            export SLO_PRIORITY_PREEMPT=1
            export SLO_PRIORITY_PREEMPT_MIN_GAP_MS=3000
            export SLO_PRIORITY_PREEMPT_RETAIN_STEPS=5
            ;;
        D_routed)
            export FT_CAPACITY_PREEMPT_RELOAD=1
            export FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1
            export SLO_PRIORITY_PREEMPT=1
            export SLO_PRIORITY_PREEMPT_MIN_GAP_MS=3000
            export SLO_PRIORITY_PREEMPT_RETAIN_STEPS=5
            ;;
        *)
            log "  unknown cell ${cell}, skipping"
            return 1
            ;;
    esac

    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free \
        --format=csv,nounits \
        -lms 1000 \
        > "${cell_out}/gpu_util.csv" 2>/dev/null &
    local nvsmi_pid=$!

    python "${ROOT}/experiments_v2/run.py" \
        --config "${cfg}" \
        --baseline "${baseline}" \
        --workload "${workload}" \
        --load "${load_level}" \
        --fault none \
        --seed "${seed}" \
        --port "${PORT}" \
        --output-dir "${cell_out}" \
        > "${cell_out}/run.log" 2>&1
    local rc=$?
    log "  rc=${rc}"

    kill "${nvsmi_pid}" 2>/dev/null
    wait "${nvsmi_pid}" 2>/dev/null

    cleanup_shm
}

# ===== Top-level: run 2 RPS levels × 4 cells × 3 seeds =====
SEEDS=(42 123 456)
RPS_LEVELS=("0.3" "0.5")

for rps in "${RPS_LEVELS[@]}"; do
    rps_tag="rps${rps/./}"   # 0.6 -> rps06, 1.0 -> rps10
    cfg_path="${OUT_ROOT}/${rps_tag}/config_${rps_tag}.yaml"
    mkdir -p "${OUT_ROOT}/${rps_tag}"
    build_config "${rps}" "${cfg_path}"
    log "===== Built config for rps=${rps} at ${cfg_path} ====="

    for seed in "${SEEDS[@]}"; do
        run_cell A_vanilla          NoFT-Reprefill "${seed}" "${rps_tag}" "${cfg_path}"
        run_cell B_capacity_release CkptReload     "${seed}" "${rps_tag}" "${cfg_path}"
        run_cell C_slo_retain       NoFT-Reprefill "${seed}" "${rps_tag}" "${cfg_path}"
        run_cell D_routed           CkptReload     "${seed}" "${rps_tag}" "${cfg_path}"
    done
done

# ===== Validation summary =====
log "===== Validation summary ====="

for rps in "${RPS_LEVELS[@]}"; do
    rps_tag="rps${rps/./}"
    log "######## RPS=${rps} ########"
    for cell in A_vanilla B_capacity_release C_slo_retain D_routed; do
        log "--- ${cell} ---"
        for seed in "${SEEDS[@]}"; do
            case "${cell}" in
                A_vanilla|C_slo_retain) baseline="NoFT-Reprefill" ;;
                *)                      baseline="CkptReload" ;;
            esac
            cell_out="${OUT_ROOT}/${rps_tag}/${cell}/${baseline}_W_Ruler16K_seed${seed}"
            srv="${cell_out}/server.log"
            [ -f "${srv}" ] || { log "  seed=${seed}: NO server.log"; continue; }

            fcsv=$(ls "${cell_out}"/forward_times_pid*.csv 2>/dev/null | head -1)
            if [ -n "${fcsv}" ]; then
                n_steps=$(($(wc -l < "${fcsv}") - 1))
            else
                n_steps=0
            fi

            v3_queued=$(grep -c "FT overlap V3:.*queued" "${srv}" 2>/dev/null | head -1)
            v3_done=$(grep -c "FT overlap V3:.*done" "${srv}" 2>/dev/null | head -1)
            slo_fires=$(grep -c "SLO_PRIORITY_PREEMPT #" "${srv}" 2>/dev/null | head -1)
            slo_resumes=$(grep -c "FT SLO retain.*resumed" "${srv}" 2>/dev/null | head -1)
            slo_dropped=$(grep -c "FT SLO retain.*dropped" "${srv}" 2>/dev/null | head -1)
            fatal=$(grep -cE "fatal error|EngineDeadError|RuntimeError|AssertionError" "${srv}" 2>/dev/null | head -1)

            mjson="${cell_out}/metrics.json"
            rcsv="${cell_out}/requests.csv"
            agg=$(python3 - <<PYEOF 2>/dev/null
import json, csv, numpy as np
m = json.load(open("${mjson}"))
gaps = []
try:
    for r in csv.DictReader(open("${rcsv}")):
        try:
            g = float(r['max_gap_ms']) if r['max_gap_ms'] else 0
            if g > 0: gaps.append(g)
        except (ValueError, KeyError):
            pass
except FileNotFoundError:
    pass
gap_p50 = float(np.percentile(gaps, 50)) if gaps else 0
gap_p95 = float(np.percentile(gaps, 95)) if gaps else 0
gap_max = float(max(gaps)) if gaps else 0
print(f"total={m.get('total_requests',0)} done={m.get('completed',0)} "
      f"goodput={m.get('goodput',0):.0f}tok/s "
      f"ttft_p95={m.get('ttft_p95_ms',0):.0f}ms "
      f"tpot_p95={m.get('tpot_p95_ms',0):.0f}ms "
      f"slo_viol={m.get('slo_violation_rate',0):.1%} "
      f"gap_p50={gap_p50:.0f}ms gap_p95={gap_p95:.0f}ms gap_max={gap_max:.0f}ms")
PYEOF
)
            [ -z "$agg" ] && agg="NO metrics.json"

            log "  seed=${seed}: steps=${n_steps} v3_queued=${v3_queued} v3_done=${v3_done} slo_fire=${slo_fires} slo_resume=${slo_resumes} slo_drop=${slo_dropped} fatal=${fatal}"
            log "           ${agg}"
        done
    done
done

log "===== 4-cell run complete. Results in ${OUT_ROOT} ====="
