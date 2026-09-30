#!/usr/bin/env bash
# Multi-seed repeats of the two discriminating experiments (fault-during-blackout, goal-step).
set -uo pipefail
CKPT="${CKPT:-/hy-tmp/models/uwam/best_ou.pt}"
SEEDS="${SEEDS:-0 1 2}"
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

for s in ${SEEDS}; do
  run_stage "fault_mid_s${s}" --seed "${s}" --seconds 12 --modes mixer,sysid,wam \
    --drop-dvl-at 6.0 --regime-set fault_mid \
    --out "${LOGS}/closed_loop_fault_mid_s${s}.json" || exit 1
  run_stage "goal_step_s${s}" --seed "${s}" --seconds 12 --modes mixer,sysid,wam \
    --drop-dvl-at 6.0 --goal-step-at 8.0 \
    --out "${LOGS}/closed_loop_goalstep_s${s}.json" || exit 1
done
echo "ALL_SEEDS_DONE"
