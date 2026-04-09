#!/usr/bin/env bash
# ============================================================================
# phase6_fast_tmpfs.sh — Validate FT_FAST_TMPFS_WRITE=1 with 3 seeds.
# Compare against phase2_variance/reload_s* (same 3 seeds without env var).
# ============================================================================

set -u
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_FAST_TMPFS_WRITE=1

OUT_BASE="results_v2/8B_overnight_2026-04-09/phase6_fast_tmpfs"

for seed in 42 123 456; do
    out_dir="${OUT_BASE}/reload_s${seed}/Our-System/W1_Chat/Heavy/F2_Mid/${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%T)] SKIP seed=${seed} (already exists)"
        continue
    fi
    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%T)] START phase6 seed=${seed}"
    python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml \
        --baseline Our-System --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed "$seed" --port 8400 \
        --output-dir "$out_dir" > "${out_dir}/stdout.log" 2>&1
    rc=$?
    echo "[$(date +%T)] END   phase6 seed=${seed} rc=$rc"
    if [ -f "${out_dir}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out_dir}/metrics.json'))
print(f'  goodput={m[\"goodput\"]:.1f}  '
      f'ttft_p50={m[\"ttft_p50_ms\"]:.0f}ms  '
      f'tpot_p50={m[\"tpot_p50_ms\"]:.1f}ms  '
      f'slo_viol={m[\"slo_violation_rate\"]*100:.1f}%  '
      f'completion={m[\"completion_rate\"]*100:.1f}%')
"
    fi
done

# Summary comparison
python3 << 'PYEOF'
import json
import statistics
from pathlib import Path

base = Path("results_v2/8B_overnight_2026-04-09")

def load_seeds(path_glob):
    rows = []
    for mf in sorted(base.glob(path_glob)):
        try:
            m = json.load(open(mf))
            rows.append({
                "path": str(mf.relative_to(base)),
                "goodput": m["goodput"],
                "ttft_p50": m["ttft_p50_ms"],
                "ttft_p95": m["ttft_p95_ms"],
                "tpot_p50": m["tpot_p50_ms"],
                "slo_viol": m["slo_violation_rate"] * 100,
                "completion": m["completion_rate"] * 100,
            })
        except Exception:
            pass
    return rows

baseline = load_seeds("phase2_variance/reload_s*/**/metrics.json")
phase6 = load_seeds("phase6_fast_tmpfs/reload_s*/**/metrics.json")

print("\n=== Phase 6 vs Phase 2 baseline ===\n")
print(f"{'cell':<40} {'goodput':>10} {'ttft_p50':>10} {'tpot_p50':>10} {'slo%':>7} {'comp%':>7}")
print("-" * 86)

print("--- baseline (no env var) ---")
for r in baseline:
    print(f"{r['path']:<40} {r['goodput']:>10.1f} {r['ttft_p50']:>10.0f} "
          f"{r['tpot_p50']:>10.1f} {r['slo_viol']:>7.1f} {r['completion']:>7.1f}")
if baseline:
    g = [r['goodput'] for r in baseline]
    print(f"baseline mean ± stdev: {statistics.mean(g):.1f} ± {statistics.stdev(g):.1f}")

print("\n--- FT_FAST_TMPFS_WRITE=1 ---")
for r in phase6:
    print(f"{r['path']:<40} {r['goodput']:>10.1f} {r['ttft_p50']:>10.0f} "
          f"{r['tpot_p50']:>10.1f} {r['slo_viol']:>7.1f} {r['completion']:>7.1f}")
if phase6:
    g = [r['goodput'] for r in phase6]
    print(f"phase6 mean ± stdev:   {statistics.mean(g):.1f} ± {statistics.stdev(g):.1f}")

if baseline and phase6:
    bg = statistics.mean([r['goodput'] for r in baseline])
    pg = statistics.mean([r['goodput'] for r in phase6])
    print(f"\ndelta: {pg - bg:+.1f} tok/s ({(pg-bg)/bg*100:+.1f}%)")
PYEOF
