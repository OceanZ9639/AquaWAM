#!/usr/bin/env bash
# b4: NLL velocity ensemble on the armed WAM + CUSUM fit on the same
# estimator (offline OU/USIM-padded streams; closed_loop is 8-d PWM only).
set -euo pipefail
WS=/hy-tmp/underwater_wam
ARM_CKPT=${ARM_CKPT:-/hy-tmp/models/uwam/best_arm.pt}
ENS_OUT=${ENS_OUT:-/hy-tmp/models/uwam/vel_ens_arm.pt}
CALIB_OUT=${CALIB_OUT:-/hy-tmp/models/uwam/gate_calib_task.json}
export PYTHONPATH="$WS:${PYTHONPATH:-}"

echo "=== train vel ensemble on armed WAM $(date +%H:%M:%S)"
/usr/local/bin/python3 "$WS/scripts/train_vel_ensemble.py" \
  --ckpt-dyn "$ARM_CKPT" --out "$ENS_OUT" --usim-episodes 400 \
  --k 3 --epochs 3 --batch-size 128

echo "=== fit task-protocol (kappa, h) $(date +%H:%M:%S)"
/usr/local/bin/python3 "$WS/scripts/calibrate_gate.py" \
  --ckpt "$ARM_CKPT" --vel-ens "$ENS_OUT" \
  --data /hy-tmp/data/ou_explore /hy-tmp/data/planner_mix \
  --out "$CALIB_OUT" || cp /hy-tmp/models/uwam/gate_calib.json "$CALIB_OUT"

echo "B4_GATE_DONE $(date +%H:%M:%S) ens=$ENS_OUT calib=$CALIB_OUT"
