#!/usr/bin/env bash
# P0: re-run only the losing fault_mid configs (surge goal, surge-side faults) with per-tick traces.
set -uo pipefail
CKPT="${CKPT:-/hy-tmp/models/uwam/best_ou.pt}"
LOGS=/hy-tmp/logs/uwam
WS=/hy-tmp/underwater_wam

for s in 0 1 2; do
  echo "=== diag seed ${s} start $(date +%H:%M:%S) ==="
  bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/closed_loop.py" --ckpt "${CKPT}" \
    --seed "${s}" --seconds 12 --modes mixer,wam --drop-dvl-at 6.0 \
    --regime-set fault_mid --regimes fail_mid_t0,fail_mid_t1,fail_mid_t03 \
    --goals "0.25,0,0" --save-traces "${LOGS}/traces_p0" \
    --out "${LOGS}/closed_loop_p0diag_s${s}.json" || exit 1
  for p in $(ps -eo pid,cmd | awk '/\/parsed_simulator |roslaunch stonefish|rosmaster --core/ && !/awk/{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  echo "=== diag seed ${s} done $(date +%H:%M:%S) ==="
done
echo "P0_DIAG_DONE"
