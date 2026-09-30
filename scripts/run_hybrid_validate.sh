#!/usr/bin/env bash
# Hybrid blind-policy validation: constant-goal dropout (hold should now tie) and
# fault_mid seed 0 (mild faults should tie, severe still win).
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

run_stage blind_hybrid --seed 0 --seconds 12 --modes mixer,wam --drop-dvl-at 6.0 \
  --out "${LOGS}/closed_loop_blind_hybrid.json" || exit 1
run_stage fault_mid_hybrid --seed 0 --seconds 12 --modes mixer,wam --drop-dvl-at 6.0 \
  --regime-set fault_mid --out "${LOGS}/closed_loop_fault_mid_hybrid.json" || exit 1
run_stage goal_step_hybrid --seed 0 --seconds 12 --modes mixer,wam --drop-dvl-at 6.0 \
  --goal-step-at 8.0 --out "${LOGS}/closed_loop_goalstep_hybrid.json" || exit 1
echo "HYBRID_VALIDATE_DONE"
