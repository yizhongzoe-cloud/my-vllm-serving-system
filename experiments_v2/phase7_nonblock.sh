#!/usr/bin/env bash
# ============================================================================
# phase7_nonblock.sh — 3-seed validation of FT_CKPT_NONBLOCK=1 +
# FT_FAST_TMPFS_WRITE=1 against phase2_variance/reload_s* baseline.
# ============================================================================

set -u
set -o pipefail

cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FT_CKPT_NONBLOCK=1
export FT_FAST_TMPFS_WRITE=1

OUT_BASE="results_v2/8B_overnight_2026-04-09/phase7_nonblock"

for seed in 42 123 456; do
    out_dir="${OUT_BASE}/reload_s${seed}/Our-System/W1_Chat/Heavy/F2_Mid/${seed}"
    if [ -f "${out_dir}/metrics.json" ]; then
        echo "[$(date +%T)] SKIP seed=${seed} (already exists)"
        continue
    fi
    mkdir -p "$out_dir"
    rm -rf /dev/shm/vllm_ft_checkpoints

    echo "[$(date +%T)] START phase7 seed=${seed}"
    python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml \
        --baseline Our-System --workload W1_Chat --load Heavy --fault F2_Mid \
        --seed "$seed" --port 8400 \
        --output-dir "$out_dir" > "${out_dir}/stdout.log" 2>&1
    rc=$?
    echo "[$(date +%T)] END   phase7 seed=${seed} rc=$rc"
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

# Comparison
python3 << 'PYEOF'
import json, statistics
from pathlib import Path

base = Path("results_v2/8B_overnight_2026-04-09")

def load(glob):
    rows = []
    for mf in sorted(base.glob(glob)):
        try:
            m = json.load(open(mf))
            rows.append({
                "path": str(mf.relative_to(base)),
                "goodput": m["goodput"],
                "ttft_p50": m["ttft_p50_ms"],
                "ttft_p95": m["ttft_p95_ms"],
                "tpot_p50": m["tpot_p50_ms"],
                "tpot_p95": m["tpot_p95_ms"],
                "slo_viol": m["slo_violation_rate"] * 100,
                "completion": m["completion_rate"] * 100,
            })
        except Exception:
            pass
    return rows

baseline = load("phase2_variance/reload_s*/**/metrics.json")
phase6 = load("phase6_fast_tmpfs/reload_s*/**/metrics.json")
phase7 = load("phase7_nonblock/reload_s*/**/metrics.json")
nockpt = load("phase2_variance/nockpt_s*/**/metrics.json")
noft = load("02_noft_baseline/**/metrics.json")

print("\n=== Phase 7 vs others — W1_Chat/Heavy/F2_Mid 3 seeds ===\n")
for label, rows in [
    ("baseline (default)", baseline),
    ("FT_FAST_TMPFS_WRITE only", phase6),
    ("FT_CKPT_NONBLOCK + FT_FAST_TMPFS_WRITE", phase7),
    ("Our-System-NoCkpt (control)", nockpt),
    ("No-FT (single seed, target)", noft),
]:
    if not rows:
        continue
    g = [r['goodput'] for r in rows]
    t50 = [r['ttft_p50'] for r in rows]
    p50 = [r['tpot_p50'] for r in rows]
    s = [r['slo_viol'] for r in rows]
    c = [r['completion'] for r in rows]
    n = len(g)
    if n == 1:
        print(f"{label:<42}: goodput={g[0]:.1f}  ttft_p50={t50[0]:.0f}  tpot_p50={p50[0]:.1f}  slo%={s[0]:.1f}  comp%={c[0]:.1f}")
    else:
        print(f"{label:<42}: goodput={statistics.mean(g):.1f}±{statistics.stdev(g):.1f}  ttft_p50={statistics.mean(t50):.0f}±{statistics.stdev(t50):.0f}  tpot_p50={statistics.mean(p50):.1f}±{statistics.stdev(p50):.1f}  slo%={statistics.mean(s):.1f}±{statistics.stdev(s):.1f}  comp%={statistics.mean(c):.1f}±{statistics.stdev(c):.1f}")

if baseline and phase7:
    bg = statistics.mean([r['goodput'] for r in baseline])
    pg = statistics.mean([r['goodput'] for r in phase7])
    print(f"\ndelta phase7 vs baseline: {pg - bg:+.1f} tok/s ({(pg-bg)/bg*100:+.1f}%)")
PYEOF
