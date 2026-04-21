#!/usr/bin/env bash
# Validate parallel-restore fix: Heavy 3-seed with FT_RESTORE_BATCH_RPC=1
# + FT_RESTORE_PARALLEL_LOAD=1. Target: fg_p95 drops from 4.7s to <3.2s,
# goodput beats NR (mean 284) on Heavy.

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

while pgrep -f "python experiments_v2/run.py" > /dev/null 2>&1; do sleep 15; done
sleep 5

CONFIG="experiments_v2/config_8b.yaml"
ROOT="results_v2/8B/parallel_restore_2026-04-14"
mkdir -p "$ROOT"

COMMON_ENV='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_ASYNC_RESTORE=1 FT_GATED_SOLVER=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_GPU_OVERLAP=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1'

wait_for_gpu() {
    local gpus="$1"
    while true; do
        local min_free=1000000
        for g in ${gpus//,/ }; do
            local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
            [ -n "$free" ] && [ "$free" -lt "$min_free" ] && min_free=$free
        done
        [ "$min_free" -ge 20000 ] && return 0
        sleep 30
    done
}

run_one() {
    local tag="$1" seed="$2" extra="$3" gpu="$4" port="$5"
    local out="${ROOT}/${tag}/Heavy_F2_Mid/${seed}"
    [ -f "${out}/metrics.json" ] && {
        python3 -c "import json; m=json.load(open('${out}/metrics.json')); print(f'  [skip] ${tag}/s${seed}: gp={m[\"goodput\"]:.1f}')" 2>/dev/null
        return 0
    }
    wait_for_gpu "${gpu}"
    mkdir -p "$out" && rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] ${tag}/s${seed}" >&2
    eval "CUDA_VISIBLE_DEVICES=${gpu} ${COMMON_ENV} ${extra} python experiments_v2/run.py \
        --config ${CONFIG} --baseline Our-System \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port ${port} --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  ${tag}/s${seed}: FAILED" >&2
}

echo "######## Parallel-restore validation $(date) ########"
# Run os_parallel (prebudget + batch_rpc + parallel_load) 3-seed on GPU 4-5
# AND os_parallel_nopb (same without prebudget) 3-seed on GPU 6-7 as control
(
    for seed in 42 123 456; do
        run_one "os_parallel_pb" "$seed" "FT_RECOVERY_PREBUDGET=1" "4,5" "8400"
    done
    echo "[$(date +%H:%M:%S)] 4-5 done" >&2
) &
HP=$!
(
    for seed in 42 123 456; do
        run_one "os_parallel" "$seed" "" "6,7" "8500"
    done
    echo "[$(date +%H:%M:%S)] 6-7 done" >&2
) &
MP=$!
wait $HP $MP

echo "######## Done $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
def load(p):
    try: return json.load(open(p))
    except: return None

nr_h = [load(f"results_v2/8B/pb_validate/heavy/nr/{s}/metrics.json") for s in [42,123,456]]
nr_gps = [m['goodput'] for m in nr_h if m]
nr_fgs = [m['failover_gap_p95_ms'] for m in nr_h if m]
print(f"\nNR Heavy: gp={st.mean(nr_gps):.1f}±{st.stdev(nr_gps):.0f}  fg={st.mean(nr_fgs):.0f}±{st.stdev(nr_fgs):.0f}")

for tag in ["os_parallel", "os_parallel_pb"]:
    rows = []
    for s in [42,123,456]:
        m = load(f"results_v2/8B/parallel_restore_2026-04-14/{tag}/Heavy_F2_Mid/{s}/metrics.json")
        if m and m['completion_rate']>=0.95:
            rows.append((m['goodput'], m['failover_gap_p95_ms']))
    if len(rows)==3:
        gp = st.mean([r[0] for r in rows])
        fg = st.mean([r[1] for r in rows])
        gp_s = st.stdev([r[0] for r in rows])
        d = gp - st.mean(nr_gps)
        print(f"  {tag}: gp={gp:.1f}±{gp_s:.0f}  fg={fg:.0f}  Δvs.NR={d:+.1f}  {'WIN' if d>0 else 'LOSE'}")
    else:
        print(f"  {tag}: only {len(rows)}/3 comp>=95%")
PYEOF
