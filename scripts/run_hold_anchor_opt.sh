#!/usr/bin/env bash
# Mixer closed-form hold vs CEM-mean hold, wam-only, on the two underwater
# weak protocols (constant-goal blackout and fault_mid). Requires exclusive sim.
set -uo pipefail
CKPT="${CKPT:-/hy-tmp/models/uwam/best_ou.pt}"
LOGS=/hy-tmp/logs/uwam
WS=/hy-tmp/underwater_wam
mkdir -p "$LOGS"

run_stage() {
  local name="$1"; shift
  echo "=== hold-opt ${name} start $(date +%H:%M:%S) ==="
  bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/closed_loop.py" --ckpt "${CKPT}" "$@"
  local rc=$?
  for p in $(ps -eo pid,cmd | awk '/\/parsed_simulator |roslaunch stonefish|rosmaster --core/ && !/awk/{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  echo "=== hold-opt ${name} done rc=${rc} $(date +%H:%M:%S) ==="
  return $rc
}

for anc in mean mixer median; do
  run_stage "blind_${anc}" --seed 0 --seconds 12 --modes wam --drop-dvl-at 6.0 \
    --hold-anchor "$anc" --out "${LOGS}/closed_loop_blind_hold_${anc}.json" || true
  run_stage "fault_mid_${anc}" --seed 0 --seconds 12 --modes wam --drop-dvl-at 6.0 \
    --regime-set fault_mid --hold-anchor "$anc" \
    --out "${LOGS}/closed_loop_fault_mid_hold_${anc}.json" || true
done
echo "HOLD_ANCHOR_OPT_DONE $(date +%H:%M:%S)"
python3 - <<'PY'
import json
from pathlib import Path
logs = Path("/hy-tmp/logs/uwam")
print(f"{'file':<48} {'err_blind':>10}")
for p in sorted(logs.glob("closed_loop_*_hold_*.json")):
    d = json.loads(p.read_text())
    rows = d.get("results") or d
    # flatten trial means if present
    if isinstance(rows, dict) and "trials" not in rows:
        for k, v in rows.items():
            if isinstance(v, dict) and "err_blind" in v:
                print(f"{p.name+'/'+k:<48} {v['err_blind']:.4f}")
    elif "trials" in d:
        errs = [t.get("err_blind") for t in d["trials"] if t.get("err_blind") is not None]
        if errs:
            print(f"{p.name:<48} {sum(errs)/len(errs):.4f}")
PY
