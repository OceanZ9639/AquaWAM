#!/usr/bin/env bash
# The two discriminating experiments: extended fault-during-blackout, then goal-step-while-blind.
set -uo pipefail
CKPT="${CKPT:-/hy-tmp/models/uwam/best_ou.pt}"
LOGS=/hy-tmp/logs/uwam
WS=/hy-tmp/underwater_wam

run_stage() {
  local name="$1"; shift
  echo "=== stage ${name} start $(date +%H:%M:%S) ==="
  bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/closed_loop.py" --ckpt "${CKPT}" "$@"
  local rc=$?
  for p in $(ps -eo pid,cmd | awk '/\/parsed_simulator |roslaunch stonefish|rosmaster --core/ && !/awk/{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  echo "=== stage ${name} done rc=${rc} $(date +%H:%M:%S) ==="
  return $rc
}

run_stage fault_mid_ext --seconds 12 --modes mixer,sysid,wam --drop-dvl-at 6.0 --regime-set fault_mid \
  --out "${LOGS}/closed_loop_fault_mid.json" || exit 1
run_stage goal_step --seconds 12 --modes mixer,sysid,wam --drop-dvl-at 6.0 --goal-step-at 8.0 \
  --out "${LOGS}/closed_loop_goalstep.json" || exit 1
echo "ALL_STAGES_DONE"
