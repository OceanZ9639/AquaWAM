#!/usr/bin/env bash
# Pause the paper-scale eval at the current block boundary, run the percept-arm
# grasp pilot (the no-asterisk arm's first closed-loop contact), then resume the
# full eval with the resumable runner. Assumes the old run_full parents are
# already killed and the in-flight block finishes as an orphan.
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
mkdir -p "$LOG"

echo "=== waiting for in-flight block to finish $(date +%H:%M:%S)"
for i in $(seq 1 480); do
  if ! pgrep -f 'batch_run_ext|run_eval_task' >/dev/null; then break; fi
  sleep 15
done
sleep 10
for pat in 'parsed_simulator' 'roslaunch' 'rosmaster'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
done
sleep 3

echo "=== install resumable runner $(date +%H:%M:%S)"
cp "$WS/u0eval/run_full_v2.sh" "$WS/u0eval/run_full.sh"
chmod +x "$WS/u0eval/run_full.sh"

echo "=== start wam_percept server (8003) $(date +%H:%M:%S)"
for p in $(pgrep -f 'goal-source percept' || true); do kill "$p" 2>/dev/null || true; done
sleep 2
nohup bash "$WS/u0eval/start_server.sh" wam_percept 8003 >/dev/null 2>&1 &
for i in $(seq 1 24); do
  sleep 5
  if curl -s -m 5 http://127.0.0.1:8003/health | grep -q healthy; then break; fi
done
curl -s -m 5 http://127.0.0.1:8003/health || { echo "percept server failed"; exit 1; }

echo "=== percept grasp pilot $(date +%H:%M:%S)"
for task in pick_pipe0_shallow pick_red_shallow; do
  bash "$WS/u0eval/run_eval_task.sh" "$task" wam_percept 5 8003 -1.0 zero \
    >"$LOG/pilot_${task}_wam_percept_full.log" 2>&1 || echo "  (nonzero exit)"
  bash "$WS/u0eval/run_eval_task.sh" "$task" wam_percept 5 8003 30 zero \
    >"$LOG/pilot_${task}_wam_percept_drop.log" 2>&1 || echo "  (nonzero exit)"
done

echo "=== percept pilot results ==="
for d in /hy-tmp/u0env/dataset/eval_runs/wam_percept_*/pick_*/results.csv; do
  echo "--- $d"; cat "$d" 2>/dev/null
done
echo "PERCEPT_PILOT_DONE $(date +%H:%M:%S)"

echo "=== resume full eval $(date +%H:%M:%S)"
nohup bash "$WS/u0eval/run_full.sh" all >>"$LOG/full_all.log" 2>&1 &
echo "resumed run_full pid $!"
sleep 5
nohup bash "$WS/u0eval/after_full.sh" >"$LOG/after_full.log" 2>&1 &
echo "re-armed after_full watcher pid $!"
echo "INSERT_PERCEPT_PILOT_DONE $(date +%H:%M:%S)"
