#!/usr/bin/env bash
# b3: mix MPC-in-the-loop dumps into a short USIM fine-tune (prevents §7.6
# cos_to_ideal < 0). Dumps come from wam_policy_server --dump-dir.
set -euo pipefail
WS=/hy-tmp/underwater_wam
DUMP=${1:-/hy-tmp/data/planner_task}
CKPT_IN=${CKPT_IN:-/hy-tmp/models/uwam/best_ou.pt}
CKPT_NAME=${CKPT_NAME:-best_ou_b3}
N=$(ls -1 "$DUMP"/*.npz 2>/dev/null | wc -l)
echo "planner-task dumps: $N in $DUMP"
if [ "$N" -lt 2 ]; then
  echo "need >=2 npz dumps; aborting fine-tune (keep $CKPT_IN)"
  exit 0
fi
export PYTHONPATH="$WS:${PYTHONPATH:-}"
/usr/local/bin/python3 "$WS/scripts/train_stage1.py" \
  --resume "$CKPT_IN" --mix-ou --extra-ou "$DUMP" --extra-reps 24 \
  --epochs 2 --batch-size 64 --ckpt-name "$CKPT_NAME" --workers 4
echo "B3_FINETUNE_DONE $CKPT_NAME $(date +%H:%M:%S)"
