#!/usr/bin/env bash
# NVIDIA GL sim + SamplingMPC 18 trials (plus sysid / no_disturb).
set -euo pipefail
CKPT="${CKPT:-/hy-tmp/models/uwam/best.pt}"
SECONDS_ARG="${SECONDS_ARG:-12}"
MODES="${MODES:-wam,no_disturb,sysid}"

bash /hy-tmp/underwater_wam/scripts/with_sim.sh \
  python /hy-tmp/underwater_wam/scripts/closed_loop.py \
    --ckpt "${CKPT}" --seconds "${SECONDS_ARG}" --modes "${MODES}"
echo "closed_loop done"
