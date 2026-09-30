#!/usr/bin/env bash
# Surface the first WAM pick-task result as soon as it lands (grasp-stage sanity).
set -uo pipefail
OUT=/hy-tmp/logs/u0eval/first_wam_pick.log
while true; do
  f=$(ls -t /hy-tmp/u0env/dataset/eval_runs/wam_full/pick_*/results.csv 2>/dev/null | tail -1)
  if [ -n "${f:-}" ]; then
    echo "FIRST_WAM_PICK_RESULT $f $(date +%H:%M:%S)" | tee -a "$OUT"
    cat "$f" | tee -a "$OUT"
    exit 0
  fi
  sleep 300
done
