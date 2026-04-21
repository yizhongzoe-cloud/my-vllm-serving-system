#!/usr/bin/env bash
# Overnight 2026-04-21: 5-phase plan to find a config that strict-beats NR.
# Phase 1: V2-reprefill ablation (keep solver, swap reload->reprefill)
# Phase 2: Fault timing robustness (F1_Early, F3_Late)
# Phase 3: Saturation load (RPS=2.5)
# Phase 4: Ablation (no-warmstart, no-ckpt)
# Phase 5: Extra seeds on best config from Phase 1-3

set -u
cd /home/jlpang/my-vllm-serving-system
source venv/bin/activate
ROOT="results_v2/8B/overnight_2026-04-21"
mkdir -p "$ROOT"

# Wait for diag23 to fully release GPUs + /dev/shm.
while pgrep -f "diag23_w5_low_load\|diag22_tuning\|auto_rerun" > /dev/null 2>&1; do sleep 20; done
sleep 10
pkill -9 -f "api_server|EngineCore" 2>/dev/null
sleep 5
rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null

# ---- Env bases ----
V2_SOLVER='PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FT_CKPT_NONBLOCK=1 FT_FAST_TMPFS_WRITE=1 FT_FAST_CHUNK_FORMAT=1 FT_BATCH_GATHER=1 FT_PINNED_POOL=1 FT_CKPT_LOAD_GUARD=0.7 FT_BATCH_CKPT_EVAL=1 FT_SOLVER_RUNNING_CAP=3 FT_RECOVERY_PREBUDGET=1 FT_CHECKPOINT_STEP_INTERVAL=2 FT_SOLVER_TIME_CAP_MS=100 FT_SOLVER_GREEDY_SEED=1'
V2_RELOAD="$V2_SOLVER FT_ASYNC_RESTORE=1 FT_CKPT_GPU_OVERLAP=1 FT_RESTORE_BATCH_RPC=1 FT_RESTORE_PARALLEL_LOAD=1 FT_RESTORE_PER_REQ_SYNC=1 FT_RECOVERY_MODE=reload"
V2_REPREFILL="$V2_SOLVER FT_RECOVERY_MODE=reprefill"
NR_ENV="FT_RECOVERY_MODE=reprefill"

# ---- Run helper ----
run_cell() {
    local phase="$1" tag="$2" baseline="$3" wl="$4" load="$5" fault="$6" seed="$7" env_str="$8"
    local out="${ROOT}/${phase}/${tag}/${seed}"
    mkdir -p "$out"
    [ -f "${out}/metrics.json" ] && return 0
    # Zombie cleanup
    pkill -9 -f "api_server|EngineCore" 2>/dev/null
    rm -rf /dev/shm/vllm_ft_checkpoints 2>/dev/null
    sleep 3
    local shm_free=$(df -m /dev/shm | awk 'NR==2 {print $4}')
    echo "[$(date +%H:%M:%S)] ${phase}/${tag}/s${seed} ${baseline} ${wl} ${load} ${fault} (shm=${shm_free}MB)"
    eval "CUDA_VISIBLE_DEVICES=0,1 ${env_str} timeout 600 python experiments_v2/run.py \
        --config experiments_v2/config_8b.yaml --baseline ${baseline} \
        --workload ${wl} --load ${load} --fault ${fault} \
        --seed ${seed} --port 8500 --output-dir ${out}" > "${out}/stdout.log" 2>&1
    if [ -f "${out}/metrics.json" ]; then
        python3 -c "
import json
m = json.load(open('${out}/metrics.json'))
print(f'  ${phase}/${tag}/s${seed}: gp={m[\"goodput\"]:.1f} comp={m[\"completion_rate\"]*100:.0f}% fg_p95={m.get(\"failover_gap_p95_ms\",0):.0f}')
"
    else
        echo "  ${phase}/${tag}/s${seed}: NO METRICS (crash or timeout)"
    fi
}

# Summary helper
summarize_phase() {
    local phase="$1"
    python3 << PYEOF
import json, os, statistics as st
ROOT = "${ROOT}/${phase}"
if not os.path.isdir(ROOT): print("no data"); exit()
print(f"\n========= Phase summary: ${phase} =========")
for tag in sorted(os.listdir(ROOT)):
    tdir = os.path.join(ROOT, tag)
    if not os.path.isdir(tdir): continue
    results = {}
    for seed in sorted(os.listdir(tdir)):
        p = os.path.join(tdir, seed, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            results[seed] = (m['goodput'], m.get('failover_gap_p95_ms',0), m['completion_rate']*100)
    if not results: continue
    stable = {s:v for s,v in results.items() if v[2] >= 95}
    print(f"\n-- {tag} (n={len(results)}, stable={len(stable)}) --")
    for s, (g,f,c) in results.items():
        mark = ' CRASH' if c < 95 else ''
        print(f"  s{s}: gp={g:.1f} fg_p95={f:.0f} comp={c:.0f}%{mark}")
    if stable:
        gps = [v[0] for v in stable.values()]
        fgs = [v[1] for v in stable.values()]
        print(f"  stable agg: gp={st.mean(gps):.1f}±{st.stdev(gps) if len(gps)>1 else 0:.0f}  fg_p95={st.mean(fgs):.0f}±{st.stdev(fgs) if len(fgs)>1 else 0:.0f}")
PYEOF
}

##################################################
# Phase 1: V2-reprefill main test                #
##################################################
echo "######## Phase 1: V2-reprefill ablation $(date) ########"
SEEDS_P1="42 123 456 789 1234 22222"

# W7 Heavy V2-reprefill
for s in $SEEDS_P1; do
    run_cell "P1" "W7_V2reprefill" "Our-System" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$V2_REPREFILL" || true
done

# W7 Heavy NR (paired, fresh)
for s in $SEEDS_P1; do
    run_cell "P1" "W7_NR" "NoFT-Reprefill" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$NR_ENV" || true
done

# W5 Moderate V2-reprefill (skip reload-mode cuda_assert by using reprefill)
for s in $SEEDS_P1; do
    run_cell "P1" "W5_V2reprefill" "Our-System" "W5_LongDoc" "Moderate" "F2_Mid" "$s" "$V2_REPREFILL" || true
done

# W5 Moderate NR
for s in $SEEDS_P1; do
    run_cell "P1" "W5_NR" "NoFT-Reprefill" "W5_LongDoc" "Moderate" "F2_Mid" "$s" "$NR_ENV" || true
done

summarize_phase "P1"

##################################################
# Phase 2: Fault timing robustness               #
##################################################
echo ""
echo "######## Phase 2: Fault timing robustness $(date) ########"
SEEDS_P2="42 123 456"

for fault in F1_Early F3_Late; do
    for s in $SEEDS_P2; do
        run_cell "P2" "W7_V2rp_${fault}" "Our-System" "W7_Saturated" "Heavy" "$fault" "$s" "$V2_REPREFILL" || true
    done
    for s in $SEEDS_P2; do
        run_cell "P2" "W7_NR_${fault}" "NoFT-Reprefill" "W7_Saturated" "Heavy" "$fault" "$s" "$NR_ENV" || true
    done
done

summarize_phase "P2"

##################################################
# Phase 3: Saturation load                       #
##################################################
echo ""
echo "######## Phase 3: Saturation load $(date) ########"
SEEDS_P3="42 123 456"

# W1 Chat Saturated (admission control main stage)
for s in $SEEDS_P3; do
    run_cell "P3" "W1_V2rp_Sat" "Our-System" "W1_Chat" "Saturated" "F2_Mid" "$s" "$V2_REPREFILL" || true
done
for s in $SEEDS_P3; do
    run_cell "P3" "W1_NR_Sat" "NoFT-Reprefill" "W1_Chat" "Saturated" "F2_Mid" "$s" "$NR_ENV" || true
done

# W7 Saturated
for s in $SEEDS_P3; do
    run_cell "P3" "W7_V2rp_Sat" "Our-System" "W7_Saturated" "Saturated" "F2_Mid" "$s" "$V2_REPREFILL" || true
done
for s in $SEEDS_P3; do
    run_cell "P3" "W7_NR_Sat" "NoFT-Reprefill" "W7_Saturated" "Saturated" "F2_Mid" "$s" "$NR_ENV" || true
done

summarize_phase "P3"

##################################################
# Phase 4: Component ablation                    #
##################################################
echo ""
echo "######## Phase 4: Component ablation $(date) ########"
SEEDS_P4="42 123 456"

# V2-reprefill NO warmstart: disable greedy_seed
V2_REPREFILL_NOWS="$V2_SOLVER FT_RECOVERY_MODE=reprefill FT_SOLVER_GREEDY_SEED=0"
for s in $SEEDS_P4; do
    run_cell "P4" "V2rp_nowarmstart" "Our-System" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$V2_REPREFILL_NOWS" || true
done

# V2-reload NO warmstart: to isolate warmstart contribution in reload mode
V2_RELOAD_NOWS="$V2_RELOAD FT_SOLVER_GREEDY_SEED=0"
for s in $SEEDS_P4; do
    run_cell "P4" "V2rl_nowarmstart" "Our-System" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$V2_RELOAD_NOWS" || true
done

# V2-NoCkpt: solver admission only, no checkpointing
for s in $SEEDS_P4; do
    run_cell "P4" "V2_NoCkpt" "Our-System-NoCkpt" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$V2_REPREFILL" || true
done

summarize_phase "P4"

##################################################
# Phase 5: Pick best config, expand to 9 seeds   #
##################################################
echo ""
echo "######## Phase 5: Best-config expansion $(date) ########"
# Add 3 extra seeds to W7_V2reprefill if it showed promise
# (Phase 1 already ran 6; add 3 more for 9-seed total)
SEEDS_P5="5678 9999 11111"
for s in $SEEDS_P5; do
    run_cell "P5" "W7_V2rp_extra" "Our-System" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$V2_REPREFILL" || true
done
for s in $SEEDS_P5; do
    run_cell "P5" "W7_NR_extra" "NoFT-Reprefill" "W7_Saturated" "Heavy" "F2_Mid" "$s" "$NR_ENV" || true
done

summarize_phase "P5"

##################################################
# Final paired analysis                          #
##################################################
echo ""
echo "######## FINAL PAIRED ANALYSIS $(date) ########"
python3 << 'PYEOF'
import json, os, statistics as st
ROOT = "results_v2/8B/overnight_2026-04-21"

def read_tag(path):
    out = {}
    if not os.path.isdir(path): return out
    for seed in sorted(os.listdir(path)):
        p = os.path.join(path, seed, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            if m['completion_rate'] >= 0.95:
                out[seed] = (m['goodput'], m.get('failover_gap_p95_ms',0))
    return out

def paired(v2_tag, nr_tag, label):
    v2 = read_tag(v2_tag)
    nr = read_tag(nr_tag)
    common = sorted(set(v2.keys()) & set(nr.keys()))
    if not common:
        print(f"\n=== {label} === no paired data")
        return
    wins = 0
    print(f"\n=== {label} (n={len(common)}) ===")
    v2_gps, v2_fgs, nr_gps, nr_fgs = [], [], [], []
    for s in common:
        vg, vf = v2[s]; ng, nf = nr[s]
        v2_gps.append(vg); v2_fgs.append(vf); nr_gps.append(ng); nr_fgs.append(nf)
        gp_w = '✓' if vg >= ng else '✗'
        fg_w = '✓' if vf <= nf else '✗'
        strict = ' 🎯' if (vg >= ng and vf <= nf) else ''
        print(f"  s{s}: V2 {vg:.1f}/{vf:.0f} vs NR {ng:.1f}/{nf:.0f} [{gp_w}][{fg_w}]{strict}")
        if gp_w == '✓' and fg_w == '✓': wins += 1
    print(f"  strict wins: {wins}/{len(common)}")
    print(f"  V2 mean: gp={st.mean(v2_gps):.1f}  fg_p95={st.mean(v2_fgs):.0f}")
    print(f"  NR mean: gp={st.mean(nr_gps):.1f}  fg_p95={st.mean(nr_fgs):.0f}")
    v2_gp_delta = st.mean(v2_gps) - st.mean(nr_gps)
    v2_fg_delta = st.mean(v2_fgs) - st.mean(nr_fgs)
    print(f"  Δ V2-NR: gp={v2_gp_delta:+.1f}  fg_p95={v2_fg_delta:+.0f}")

# Phase 1
paired(f"{ROOT}/P1/W7_V2reprefill", f"{ROOT}/P1/W7_NR", "P1 W7 Heavy V2-reprefill")
paired(f"{ROOT}/P1/W5_V2reprefill", f"{ROOT}/P1/W5_NR", "P1 W5 Moderate V2-reprefill")
# Phase 2
paired(f"{ROOT}/P2/W7_V2rp_F1_Early", f"{ROOT}/P2/W7_NR_F1_Early", "P2 W7 F1_Early V2-reprefill")
paired(f"{ROOT}/P2/W7_V2rp_F3_Late", f"{ROOT}/P2/W7_NR_F3_Late", "P2 W7 F3_Late V2-reprefill")
# Phase 3
paired(f"{ROOT}/P3/W1_V2rp_Sat", f"{ROOT}/P3/W1_NR_Sat", "P3 W1 Saturated")
paired(f"{ROOT}/P3/W7_V2rp_Sat", f"{ROOT}/P3/W7_NR_Sat", "P3 W7 Saturated")
# Phase 5 = P1 W7 V2rp + extra
# Merge manually
def merge_dirs(*dirs):
    out = {}
    for d in dirs:
        for k,v in read_tag(d).items():
            out[k] = v
    return out

v2_all = merge_dirs(f"{ROOT}/P1/W7_V2reprefill", f"{ROOT}/P5/W7_V2rp_extra")
nr_all = merge_dirs(f"{ROOT}/P1/W7_NR", f"{ROOT}/P5/W7_NR_extra")
common = sorted(set(v2_all.keys()) & set(nr_all.keys()))
if common:
    print(f"\n=== FINAL W7 Heavy paired (merged P1+P5, n={len(common)}) ===")
    wins = 0
    for s in common:
        vg, vf = v2_all[s]; ng, nf = nr_all[s]
        strict = ' 🎯' if (vg >= ng and vf <= nf) else ''
        print(f"  s{s}: V2 {vg:.1f}/{vf:.0f} vs NR {ng:.1f}/{nf:.0f}{strict}")
        if vg >= ng and vf <= nf: wins += 1
    v2_gps = [v[0] for s,v in v2_all.items() if s in common]
    v2_fgs = [v[1] for s,v in v2_all.items() if s in common]
    nr_gps = [v[0] for s,v in nr_all.items() if s in common]
    nr_fgs = [v[1] for s,v in nr_all.items() if s in common]
    print(f"  strict wins: {wins}/{len(common)}")
    print(f"  V2: gp={st.mean(v2_gps):.1f}  fg_p95={st.mean(v2_fgs):.0f}")
    print(f"  NR: gp={st.mean(nr_gps):.1f}  fg_p95={st.mean(nr_fgs):.0f}")
PYEOF

echo ""
echo "######## Overnight 0421 DONE $(date) ########"
