#!/usr/bin/env bash
# NR baseline at gpu_util=0.80 — fair apples-to-apples with v7 OS results.
# Runs on GPU 6-7 in parallel with v7 (GPU 4-5).

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

ROOT="results_v2/8B/v7_nr_gpu080"
mkdir -p "$ROOT"

run_one() {
    local seed="$1"
    local out="${ROOT}/${seed}"
    [ -f "${out}/metrics.json" ] && return 0
    mkdir -p "$out"
    echo "[$(date +%H:%M:%S)] NR/s${seed} starting (GPU 6-7)" >&2
    eval "CUDA_VISIBLE_DEVICES=6,7 FT_RECOVERY_MODE=reprefill python experiments_v2/run.py \
        --config experiments_v2/config_8b_gpu080.yaml --baseline NoFT-Reprefill \
        --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed ${seed} --port 8500 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  NR/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}%% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}ms')
" 2>/dev/null || echo "  NR/s${seed}: FAILED" >&2
}

echo "######## NR gpu_util=0.80 $(date) ########"
for s in 42 123 456; do
    run_one "$s"
done
echo "######## done $(date) ########"
