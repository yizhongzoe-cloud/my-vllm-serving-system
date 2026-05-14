# FT / SLO environment variables reference

Single source of truth for every `FT_*`, `SLO_*`, and `VLLM_FT_*` knob
the experiment system reads. Add a row here every time a new env var
ships — see the bottom of the file for the audit grep used to detect
drift.

Naming conventions:

- `FT_*` — fault tolerance & disruption mechanism (host KV ckpt,
  reroute, status bus, picker tunables)
- `SLO_*` — slack-based priority preempt picker policy
- `VLLM_FT_*` — vLLM-side identity (e.g. engine id)

---

## Engine identity

| var | default | read at | purpose |
|---|---|---|---|
| `VLLM_FT_ENGINE_ID` | `0` | engine core init, scheduler init | which engine I am (0 or 1). Drives the path of my own status file (`engine_<id>.json`). |

## Router ↔ engine shm bus

| var | default | read at | purpose |
|---|---|---|---|
| `FT_ROUTER_SHM_BUS` | unset (=0) | engine core init | master switch for `/dev/shm` bus between router and engines. When 0, engines don't write status / preempt-queue / req-map files. |
| `FT_STATUS_WRITE_INTERVAL_MS` | `200` | engine status writer thread | how often the engine pings its status file in ms. Lower = router detects death faster, higher = less CPU. |

## Host KV checkpoint + reroute

| var | default | read at | purpose |
|---|---|---|---|
| `FT_CAPACITY_PREEMPT_RELOAD` | unset (=0) | engine core init | master switch for the V3 reload path. When 0, capacity preempt falls back to vanilla vLLM recompute. |
| `FT_CAPACITY_PREEMPT_RELOAD_OVERLAP` | unset (=0) | engine core init | enables the overlap reload state machine (alloc → reload → admit pipelined across scheduler steps). Without this, V3 reload is synchronous and ties up a scheduler step. |
| `FT_DELTA_CHECKPOINT` | unset (=0) | engine core init | enables incremental (per-new-block) ckpt save instead of full snapshot. |
| `FT_INLINE_MANIFEST` | unset | worker save path | optional optimization: inline manifest bytes into chunk file instead of separate manifest write. |
| `FT_FAST_CHUNK_FORMAT` | unset | worker save path | use the raw fast chunk format (ctypes libc.write) instead of pickle. |
| `FT_CKPT_FIXED_BLOCKS` | `1` | engine `_save_checkpoints_if_needed` | how many new full blocks must accumulate before a save RPC is fired. 1 = save every block. |
| `FT_CKPT_SKIP_PUBLISH` | unset | worker save path | debug flag: skip the /dev/shm publish step (compute the save but don't publish). |
| `FT_CKPT_STATS_LOG` | unset | engine | turn on per-step ckpt save stats logging (for analysis). |
| `FT_REDISPATCH_TIMEOUT_S` | `5.0` | engine core init | if router doesn't pick up a redispatch entry within this many seconds, the engine resumes the victim locally via V3 reload (host KV → GPU). Failsafe for router death / partition. |
| `FT_CROSS_ENGINE_OUTPUT_RESUME` | `1` (on) | engine core `add_request` | when on, cross-engine reroute reads the origin engine's manifest output_token_ids and resumes mid-decode (full KV coverage). When 0, falls back to clamping num_checkpointed_tokens to the prompt boundary and re-decoding from prompt end. Off matches pre-feature behavior; useful as an ablation to isolate the contribution of mid-decode resume. |

## Picker policy

| var | default | read at | purpose |
|---|---|---|---|
| `SLO_PRIORITY_PREEMPT` | unset (=0) | scheduler init | master switch for the slack-based priority preempt picker. Only `ours` baseline enables this. |
| `SLO_PRIORITY_PREEMPT_USE_RETAIN` | unset (=0) | scheduler dispatch | ablation: when 1, preempted victims retain GPU blocks and resume locally (`_preempt_for_slo_retain`) instead of cross-engine reroute (`_preempt_for_slo_redispatch`). |
| `SLO_PRIORITY_PREEMPT_PER_REQ_COOLDOWN_MS` | `5000` | scheduler init | per-victim cooldown — a victim that was just preempted can't be preempted again within this window. Prevents thrashing on the same long-lived request. |
| `SLO_PRIORITY_PREEMPT_MIN_INTERVAL_MS` | `2000` | scheduler init | global picker cooldown — picker won't fire again across the whole scheduler if it just fired within this window. Prevents back-to-back preempts thrashing. |
| `SLO_PRIORITY_PREEMPT_MIN_GAP_MS` | `1000` | scheduler init | hysteresis δ in the picker rule `slack_gap > δ + replay_cost`. Prevents thrashing when victim and head have similar slack. |
| `SLO_PRIORITY_PREEMPT_RETAIN_STEPS` | (set via env) | retain-path scheduler | how many scheduler steps to keep a retained victim out before re-admitting. |
| `FT_PICKER_SWITCH_COST_MS` | `1000` | `compute_replay_cost` (utils.py) | fixed per-fire cost (wall ms) of preempt + cross-engine forward + V3 reload + admit. Added to `replay_cost` so the picker rule reflects the *full* cost of firing, not just the per-token replay piece. Tune via empirical measurement (see `measure_switch_cost.py`). |
| `FT_PICKER_PEER_LOAD_GATE` | `1` (on) | scheduler init | when on, picker reads peer engine status files and refuses to fire if all peer engines are overloaded or unreachable. Disable for ablation that isolates the gate's contribution. |
| `FT_PEER_OVERLOAD_KV_USAGE` | `0.85` | scheduler init | a peer is "overloaded" if its kv_usage exceeds this fraction (0-1). KV cache is the real bottleneck for long-context workloads. |
| `FT_PICKER_HEAD_DANGER_GATE` | `1` (on) | scheduler init | when on, picker only fires if the waiting head is within the last `FT_PICKER_HEAD_DANGER_RATIO` fraction of its TTFT SLO budget. Prevents firing on heads that the natural admit loop would have let through in time. Disable for ablation. |
| `FT_PICKER_HEAD_DANGER_RATIO` | `0.10` | scheduler init | head must be within this fraction of its TTFT SLO before picker fires. Empirical starting value; TODO sweep 0.05 / 0.10 / 0.20 per-hardware to pick best. Lower = stricter (picker fires less), higher = looser (picker fires more). |

## Non-tunable constants (hardcoded, but document them here so they don't get lost)

These live inside `_peer_overloaded` in `scheduler.py`. They were
briefly env vars but pulled out because the values aren't worth a
sweep — they're either matched to another constant elsewhere or the
workload doesn't depend on them.

| name | value | where | why hardcoded |
|---|---|---|---|
| running cap fraction | `0.8` | `_peer_overloaded` `_PEER_RUNNING_FRAC` | Niyama-style 20% headroom. Mostly relevant for short-prompt workloads (ShareGPT) where seq count fills before KV — our long-context paper rarely hits this branch. |
| stale heartbeat threshold | `2.0` seconds | `_peer_overloaded` `_PEER_STALE_S` | Matches the router's death detector (also 2s, in `experiments_v2/router/router.py`). If you change one, change both — picker and router must agree on "alive". |

## Diagnostic / profiling

| var | default | read at | purpose |
|---|---|---|---|
| `FT_CUDA_EVENT_PROFILE` | unset | worker | dump per-event CUDA timing for save/restore ops. |
| `FT_CUDA_EVENT_OUTPUT_DIR` | unset | worker | where the CUDA event traces go. |
| `FT_CKPT_STATS_OUTPUT_DIR` | `/tmp` | kv_checkpoint_pool | output dir for per-save ckpt stats when `FT_CKPT_STATS_LOG` is on. |

---

## Audit grep — keep this doc honest

Every time you add a new `FT_*` / `SLO_*` / `VLLM_FT_*` env var, run:

```bash
grep -rhoE '"(FT_|SLO_|VLLM_FT_)[A-Z_]+"' vllm/ experiments_v2/ \
  | sed 's/"//g' | sort -u
```

(A naive `os\.environ\.get\("...` regex matches the single-line form
but misses the multi-line continuation form vLLM uses heavily, where
the var name is on the line *after* `os.environ.get(`.)

The output should match the table above. If it doesn't, either the
code has an undocumented knob or this doc has a stale entry.

---

## Recipe cards

### "Just run our system the way the paper describes"

```bash
FT_ROUTER_SHM_BUS=1 \
FT_CAPACITY_PREEMPT_RELOAD=1 \
FT_CAPACITY_PREEMPT_RELOAD_OVERLAP=1 \
FT_DELTA_CHECKPOINT=1 \
SLO_PRIORITY_PREEMPT=1 \
VLLM_FT_ENGINE_ID=<0 or 1> \
... start engine ...
```

Picker gate, switch_cost, etc. use sane defaults from this doc.

### "Ablate the picker peer-load gate"

```bash
FT_PICKER_PEER_LOAD_GATE=0 ... start engine ...
```

Everything else identical to the recipe above.

### "Sweep switch_cost to find the empirical knee"

```bash
for x in 500 1000 1600 3000 5000; do
  FT_PICKER_SWITCH_COST_MS=$x HARDWARE_TAG=a6000 \
    bash experiments_v2/eval/scripts/run_paper_sweep_ruler16k.sh
done
```
