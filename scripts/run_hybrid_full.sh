#!/usr/bin/env bash
# Final reruns with the uncertainty-gated hybrid blind policy:
# blind main -> fault_mid x3 seeds -> goal_step x3 seeds -> duration sweep -> severity sweep.
# The full-sensing gate is untouched by the hybrid (no dropout) and keeps its P3 results.
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

run_stage blind_main --seed 0 --seconds 12 --modes mixer,sysid,rls,wam --drop-dvl-at 6.0 \
  --out "${LOGS}/closed_loop_blind.json" || exit 1

for s in 0 1 2; do
  run_stage "fault_mid_s${s}" --seed "${s}" --seconds 12 --modes mixer,sysid,rls,wam \
    --drop-dvl-at 6.0 --regime-set fault_mid \
    --out "${LOGS}/closed_loop_fault_mid_s${s}.json" || exit 1
  run_stage "goal_step_s${s}" --seed "${s}" --seconds 12 --modes mixer,sysid,rls,wam \
    --drop-dvl-at 6.0 --goal-step-at 8.0 \
    --out "${LOGS}/closed_loop_goalstep_s${s}.json" || exit 1
done

for drop in 2 4 6 8 10; do
  run_stage "bsweep_d${drop}" --seed 0 --seconds 12 --modes mixer,sysid,wam \
    --drop-dvl-at "${drop}" --goal-step-at 8.0 \
    --regimes nominal,thruster_degrade,current_and_fail \
    --save-traces "${LOGS}/traces_bsweep_d${drop}" \
    --out "${LOGS}/closed_loop_bsweep_d${drop}.json" || exit 1
done

run_stage fault_sweep --seed 0 --seconds 12 --modes mixer,sysid,wam \
  --drop-dvl-at 6.0 --regime-set fault_sweep \
  --out "${LOGS}/closed_loop_fault_sweep.json" || exit 1

echo "HYBRID_FULL_DONE"
