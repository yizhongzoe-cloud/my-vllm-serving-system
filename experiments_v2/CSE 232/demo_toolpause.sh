#!/usr/bin/env bash
# ============================================================
# DEMO A — Ferry tool-pause resume  vs  recompute baseline.
# Record this terminal. Small/fast config so it finishes quickly.
#   Ferry : pauses release GPU KV, resume = host-checkpoint reload (~3s)
#   recompute (vanilla vLLM): resume = full re-prefill of the long prompt
# ============================================================
set -u
REPO="/home/yzhong76/code/my-vllm-serving-system"; cd "$REPO"
export EVAL_RESULTS_DIR="$REPO/experiments_v2/eval/results/a6000"
LOG="$EVAL_RESULTS_DIR/logs/demo_$(date +%H%M%S)"; mkdir -p "$LOG"
N=8; PAUSE=15; OUT=40; PMIN=6000; PMAX=12000

cleanup(){ pkill -KILL -f api_server 2>/dev/null; pkill -KILL -f EngineCore 2>/dev/null
  rm -rf /dev/shm/vllm_ft_checkpoints /dev/shm/vllm_ft_req_map \
         /dev/shm/vllm_ft_engine_status /dev/shm/vllm_ft_preempt_queue 2>/dev/null; sleep 3; }

run(){ # $1 baseline
  cleanup
  echo; echo "=================================================================="
  echo "  RUN: $1     pause=${PAUSE}s · $N requests · 6-12K-token prompts"
  echo "=================================================================="
  python -u experiments_v2/eval/scripts/icept_microbench.py \
    --baseline "$1" --arrival-rate-qps 0.3 --num-requests "$N" \
    --icept-ratio 0.5 --icept-at 20 --total-output "$OUT" \
    --pause-min "$PAUSE" --pause-max "$PAUSE" --seed 0 \
    --prompt-min-tokens "$PMIN" --prompt-max-tokens "$PMAX" \
    --out-tag demo 2>&1 | tee "$LOG/$1.log" | grep -E "\[icept\]"
  if [ "$1" = "ours" ]; then
    echo; echo "  >>> engine reload events (proof: host-checkpoint reload, not recompute):"
    grep -hE "reload took|tokens restored" \
      "$EVAL_RESULTS_DIR"/icept_ours_*n${N}_seed0_demo_engine.log 2>/dev/null \
      | sed 's/.*FT overlap V3: //; s/.*\] //' | tail -4 | sed 's/^/      /'
  fi
}

echo "######################  DEMO A: tool-pause resume  ######################"
run ours
run vllm_fcfs
echo; echo "########################  SIDE-BY-SIDE  ########################"
python3 - <<PY
import json, glob
for b, name in [("ours","Ferry (host reload)"), ("vllm_fcfs","recompute    ")]:
    fs = glob.glob("$EVAL_RESULTS_DIR/icept_%s_*n${N}_seed0_demo_metrics.json" % b)
    if fs:
        d = json.load(open(fs[0]))
        print("  %-22s resume p50 = %5.2f s    reload_validity = %s"
              % (name, d["seg2_latency_s"]["p50"], d.get("reload_validity", "-")))
print("  ----------------------------------------------------------------")
print("  Ferry resumes in seconds via host-checkpoint reload;")
print("  recompute re-prefills the whole long prompt every time.")
PY
echo "################################################################"
