#!/usr/bin/env bash
# Sequential rerun of Our-System on 3 new seeds (789, 1337, 2024) in phase 8
# conditions. Avoids s42's deterministic CUDA device-side assert bug on the
# phase8 A6000 dp=1 profile.
#
# Runs sequentially (NOT parallel) on GPU 6-7 to avoid /dev/shm race.
#
# Usage:
#   nohup bash experiments_v2/phase8_our_system_new_seeds.sh \
#       > /tmp/phase8_our_system_new_seeds.log 2>&1 &
#   disown

set -u
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1
export FT_FAST_CHUNK_FORMAT=1
export FT_RECOVERY_MODE=reload

OUT_BASE="results_v2/8B/E1a_phase8_repro"
CONFIG="experiments_v2/config_8b_phase8repro.yaml"

for seed in 789 1337 2024; do
    out_dir="${OUT_BASE}/Our-System/W1_Chat/Heavy/F2_Mid/${seed}"

    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP Our-System/s${seed} (already done)" >&2
        continue
    fi

    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%H:%M:%S)] START Our-System/s${seed} (GPU 6,7, port 8500)" >&2

    CUDA_VISIBLE_DEVICES=6,7 python experiments_v2/run.py \
        --config "$CONFIG" \
        --baseline Our-System \
        --workload W1_Chat \
        --load Heavy \
        --fault F2_Mid \
        --seed "$seed" \
        --port 8500 \
        --output-dir "$out_dir" \
        > "${out_dir}/stdout.log" 2>&1
    rc=$?

    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  s${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%  ttft_p50={m.get(\"ttft_p50_ms\",-1):.0f}  slo={m.get(\"slo_violation_rate\",-1)*100:.1f}%'
)
" 2>/dev/null
    else
        echo "  s${seed}: FAILED (rc=$rc)" >&2
    fi
done

echo ""
echo "=== Summary ==="
for seed in 789 1337 2024; do
    f="${OUT_BASE}/Our-System/W1_Chat/Heavy/F2_Mid/${seed}/metrics.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
m = json.load(open('$f'))
print(f's${seed}: goodput={m.get(\"goodput\",-1):.1f}  comp={m.get(\"completion_rate\",-1)*100:.1f}%')
"
    else
        echo "s${seed}: NO DATA"
    fi
done
