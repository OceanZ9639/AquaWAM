#!/usr/bin/env bash
# Retest the percept arm (search-mode server) on grasp tasks, then resume the
# full eval and re-arm the after-full watcher. Assumes orchestrators are dead
# and at most one in-flight block remains.
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
TASKS=${TASKS:-"pick_pipe0_shallow pick_red_shallow"}
NEP=${NEP:-5}

echo "=== waiting for in-flight block to finish $(date +%H:%M:%S)"
for i in $(seq 1 480); do
  if ! pgrep -f 'batch_run_ext|run_eval_task' >/dev/null; then break; fi
  sleep 15
done
sleep 5
for pat in 'parsed_simulator' 'roslaunch' 'rosmaster'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
done
sleep 3

echo "=== restart wam_percept server with search mode $(date +%H:%M:%S)"
for p in $(pgrep -f 'goal-source percept' || true); do kill "$p" 2>/dev/null || true; done
sleep 3
nohup bash "$WS/u0eval/start_server.sh" wam_percept 8003 >/dev/null 2>&1 &
for i in $(seq 1 24); do
  sleep 5
  if curl -s -m 5 http://127.0.0.1:8003/health | grep -q healthy; then break; fi
done
curl -s -m 5 http://127.0.0.1:8003/health || { echo "percept server failed"; exit 1; }

echo "=== percept retest (search mode) $(date +%H:%M:%S)"
for task in $TASKS; do
  bash "$WS/u0eval/run_eval_task.sh" "$task" wam_percept "$NEP" 8003 -1.0 zero \
    >"$LOG/retest_${task}_wam_percept_full.log" 2>&1 || echo "  (nonzero exit)"
done

echo "=== retest results ==="
for task in $TASKS; do
  f="/hy-tmp/u0env/dataset/eval_runs/wam_percept_full/$task/results.csv"
  echo "--- $f"; cat "$f" 2>/dev/null
done
echo "PERCEPT_RETEST_DONE $(date +%H:%M:%S)"

echo "=== resume full eval $(date +%H:%M:%S)"
nohup bash "$WS/u0eval/run_full.sh" all >>"$LOG/full_all.log" 2>&1 &
echo "resumed run_full pid $!"
sleep 5
nohup bash "$WS/u0eval/after_full.sh" >"$LOG/after_full.log" 2>&1 &
echo "re-armed after_full watcher pid $!"
echo "RETEST_ORCH_DONE $(date +%H:%M:%S)"
