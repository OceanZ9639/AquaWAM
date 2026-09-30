#!/usr/bin/env bash
# Re-run the scan blocks for wam/fallback with the yaw-clamp fix (the pilot's
# scan numbers for those arms predate it). 8 blocks x N episodes.
set -uo pipefail
NEP=${1:-5}
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
mkdir -p "$LOG"
for task in scan_ship_ancient scan_ship_modern; do
  for arm in wam fallback; do
    PORT=$([ "$arm" = wam ] && echo 8000 || echo 8002)
    echo "########## RESCAN $task / $arm / full  $(date +%H:%M:%S)"
    bash "$WS/u0eval/run_eval_task.sh" "$task" "$arm" "$NEP" "$PORT" -1.0 zero \
      >"$LOG/rescan_${task}_${arm}_full.log" 2>&1 || echo "  (nonzero exit)"
    echo "########## RESCAN $task / $arm / drop40s  $(date +%H:%M:%S)"
    bash "$WS/u0eval/run_eval_task.sh" "$task" "$arm" "$NEP" "$PORT" 40 zero \
      >"$LOG/rescan_${task}_${arm}_drop.log" 2>&1 || echo "  (nonzero exit)"
  done
done
echo "RESCAN_ALL_DONE $(date +%H:%M:%S)"
