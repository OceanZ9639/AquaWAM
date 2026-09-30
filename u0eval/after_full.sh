#!/usr/bin/env bash
# Runs after the paper-scale eval finishes (FULL_all_DONE in full_all.log):
#   1. final Table IV for both conditions
#   2. underwater benchmark re-run, all arms, mixer hold-anchor caliber
#      (blind main + fault_mid x3 seeds + goal_step x3 seeds) -> *_mixerhold.json
set -uo pipefail
# The IDE shell that launches these orchestrators carries HTTP_PROXY=127.0.0.1:17890
# with 0.0.0.0 absent from NO_PROXY; the bridge posts to http://0.0.0.0:<port>/act
# and got 502 Bad Gateway from the proxy. Policy traffic is local: never proxy it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
LOGS=/hy-tmp/logs/uwam
FULL_LOG="$LOG/full_all.log"
CKPT=/hy-tmp/models/uwam/best_scenes.pt   # the paper's single core

echo "=== waiting for FULL_grasp_DONE $(date +%H:%M:%S)"
while ! grep -q 'FULL_grasp_DONE' "$FULL_LOG" 2>/dev/null; do
  if ! pgrep -f 'run_full.sh' >/dev/null; then
    echo "run_full.sh no longer running; proceeding $(date +%H:%M:%S)"
    break
  fi
  sleep 120
done

echo "=== final tables $(date +%H:%M:%S)"
mkdir -p /hy-tmp/results
/usr/local/bin/python3 "$WS/u0eval/write_u0_table.py" --auto \
  --condition "Full sensing" --out /hy-tmp/results/u0_table_full.md || true
/usr/local/bin/python3 "$WS/u0eval/write_u0_table.py" --auto \
  --condition "DVL dropout" --no-paper-rows \
  --out /hy-tmp/results/u0_table_drop.md || true

echo "=== stop eval harness + servers for exclusive sim $(date +%H:%M:%S)"
for pat in 'run_full.sh' 'run_eval_task.sh' 'batch_run_ext' 'parsed_simulator' 'roslaunch' 'rosmaster' \
           'wam_policy_server' 'fallback_policy_server' 'inference_service_u0'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
done
sleep 5

run_stage() {
  local name="$1"; shift
  echo "=== mixerhold ${name} start $(date +%H:%M:%S) ==="
  bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/closed_loop.py" --ckpt "${CKPT}" \
    --hold-anchor mixer "$@"
  local rc=$?
  for p in $(ps -eo pid,cmd | awk '/\/parsed_simulator |roslaunch stonefish|rosmaster --core/ && !/awk/{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  echo "=== mixerhold ${name} done rc=${rc} $(date +%H:%M:%S) ==="
  return $rc
}

run_stage blind_main --seed 0 --seconds 12 --modes mixer,sysid,rls,wam --drop-dvl-at 6.0 \
  --out "${LOGS}/closed_loop_blind_mixerhold.json" || true
for s in 0 1 2; do
  run_stage "fault_mid_s${s}" --seed "$s" --seconds 12 --modes mixer,sysid,rls,wam \
    --drop-dvl-at 6.0 --regime-set fault_mid \
    --out "${LOGS}/closed_loop_fault_mid_mixerhold_s${s}.json" || true
  run_stage "goal_step_s${s}" --seed "$s" --seconds 12 --modes mixer,sysid,rls,wam \
    --drop-dvl-at 6.0 --goal-step-at 8.0 \
    --out "${LOGS}/closed_loop_goalstep_mixerhold_s${s}.json" || true
done
echo "AFTER_FULL_ALL_DONE $(date +%H:%M:%S)"
