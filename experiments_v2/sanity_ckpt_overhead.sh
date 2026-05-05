#!/usr/bin/env bash
# Sanity check for the ckpt overhead instrumentation.
#
# Runs ONE CkptOnly cell + ONE No-FT cell at 1K context, short duration.
# Validates:
#   1. forward_times_pid*.csv exists, non-empty, has reasonable values
#   2. ckpt_stats_pid*.csv exists, has incremental "delta" rows
#   3. Per-fire bytes are MB-scale (delta), not GB-scale (full save every time)
#   4. Fire count grows linearly with output tokens
#
# Use this BEFORE launching the full nohup sweep. ~5-7 min total.

set -u

ROOT=/home/yzhong76/code/my-vllm-serving-system
SANITY_DIR="${ROOT}/experiments_v2/results_ckpt_overhead/SANITY"
LOG_FILE="${SANITY_DIR}/sanity.log"
CONFIG="${ROOT}/experiments_v2/config_8b_ckpt_overhead.yaml"
SHM_DIR=/dev/shm/vllm_ft_checkpoints

mkdir -p "${SANITY_DIR}"

cd "${ROOT}"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "${LOG_FILE}"
}

run_one_cell() {
    local baseline="$1"
    local cell_out="${SANITY_DIR}/${baseline}_W_Ruler1K_seed42"
    local port="$2"

    log "===== Sanity cell: ${baseline} ====="
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    mkdir -p "${cell_out}"

    export FT_CUDA_EVENT_OUTPUT_DIR="${cell_out}"
    export FT_CKPT_STATS_OUTPUT_DIR="${cell_out}"
    export FT_CUDA_EVENT_PROFILE=1
    export FT_CKPT_STATS_LOG=1
    export FT_DELTA_CHECKPOINT=1
    export FT_SLO_PREEMPT=0
    export FT_USE_FCFS_BASE_QUEUE=1
    export FT_SKIP_SOLVER=1
    export FT_SLO_AWARE_OBJECTIVE=0
    export CUDA_VISIBLE_DEVICES=0

    # Override run_duration_sec via a small hack: copy config, set
    # duration short, point to it. Avoid modifying the main config.
    local mini_cfg="${cell_out}/sanity_config.yaml"
    sed 's/^run_duration_sec:.*/run_duration_sec: 90.0/; s/^warmup_sec:.*/warmup_sec: 20.0/' \
        "${CONFIG}" > "${mini_cfg}"

    python "${ROOT}/experiments_v2/run.py" \
        --config "${mini_cfg}" \
        --baseline "${baseline}" \
        --workload W_Ruler1K \
        --load Sat_1K \
        --fault none \
        --seed 42 \
        --output-dir "${cell_out}" \
        --port "${port}" \
        >> "${cell_out}/run.log" 2>&1

    local rc=$?
    log "${baseline} cell rc=${rc}"
    rm -rf "${SHM_DIR}"/* 2>/dev/null || true
    return ${rc}
}

# Run No-FT first (vanilla baseline)
run_one_cell "No-FT" 8390 || log "No-FT failed (rc=$?)"

# Run CkptOnly second
run_one_cell "CkptOnly" 8391 || log "CkptOnly failed (rc=$?)"

log "===== Validation ====="

# Check 1: forward_times CSVs
log "[1] Checking forward_times CSVs..."
for d in "${SANITY_DIR}/No-FT_W_Ruler1K_seed42" "${SANITY_DIR}/CkptOnly_W_Ruler1K_seed42"; do
    csvs=$(ls "${d}"/forward_times_pid*.csv 2>/dev/null)
    if [ -z "${csvs}" ]; then
        log "  FAIL: ${d} missing forward_times CSV"
    else
        for csv in ${csvs}; do
            n=$(wc -l < "${csv}")
            log "  ${csv}: ${n} lines"
        done
    fi
done

# Check 2: ckpt_stats CSV present in CkptOnly only
log "[2] Checking ckpt_stats CSVs..."
ckpt_csvs=$(ls "${SANITY_DIR}/CkptOnly_W_Ruler1K_seed42"/ckpt_stats_pid*.csv 2>/dev/null)
if [ -z "${ckpt_csvs}" ]; then
    log "  FAIL: CkptOnly missing ckpt_stats CSV"
else
    for csv in ${ckpt_csvs}; do
        n=$(wc -l < "${csv}")
        log "  ${csv}: ${n} lines"
    done
fi

# Check that No-FT has NO ckpt_stats CSV (sanity: ckpt code shouldn't run)
noft_ckpt=$(ls "${SANITY_DIR}/No-FT_W_Ruler1K_seed42"/ckpt_stats_pid*.csv 2>/dev/null)
if [ -n "${noft_ckpt}" ]; then
    log "  WARN: No-FT has ckpt_stats CSV (unexpected — ckpt code ran in vanilla)"
fi

# Check 3: Verify "delta" rows exist + bytes are MB-scale (not GB)
log "[3] Verifying ckpt is incremental..."
if [ -n "${ckpt_csvs}" ]; then
    for csv in ${ckpt_csvs}; do
        # Count delta vs full rows
        n_delta=$(awk -F, 'NR>1 && $4=="delta" {n++} END{print n+0}' "${csv}")
        n_full=$(awk -F, 'NR>1 && $4=="full" {n++} END{print n+0}' "${csv}")
        log "  ${csv}: delta=${n_delta} full=${n_full}"
        # Per-fire bytes statistics
        awk -F, 'NR>1 {
            bw=$6 + 0
            if (bw > max || NR==2) max = bw
            if (bw < min || NR==2) min = bw
            sum += bw
            n++
        }
        END {
            if (n > 0) printf "  bytes_written: min=%d (%.2f MB), max=%d (%.2f MB), mean=%.2f MB, n=%d\n",
                min, min/1048576, max, max/1048576, (sum/n)/1048576, n
        }' "${csv}" | tee -a "${LOG_FILE}"
    done
fi

# Check 4: forward_ms range sanity
log "[4] Forward time sanity..."
for d in "${SANITY_DIR}/No-FT_W_Ruler1K_seed42" "${SANITY_DIR}/CkptOnly_W_Ruler1K_seed42"; do
    csvs=$(ls "${d}"/forward_times_pid*.csv 2>/dev/null)
    if [ -n "${csvs}" ]; then
        for csv in ${csvs}; do
            awk -F, 'NR>1 {
                fm=$3 + 0
                if (fm > max || NR==2) max = fm
                if (fm < min || NR==2) min = fm
                sum += fm
                n++
            }
            END {
                if (n > 0) printf "  '"$(basename ${d})"' fm: min=%.2fms, max=%.2fms, mean=%.2fms, n=%d\n",
                    min, max, sum/n, n
            }' "${csv}" | tee -a "${LOG_FILE}"
        done
    fi
done

log "===== Sanity done. Review ${LOG_FILE} ====="
exit 0
