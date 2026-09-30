#!/usr/bin/env bash
# After the locomotion pilot finishes: tables, hold-anchor opt, b3/b4, then
# paper-scale eval. Safe to start while run_pilot.sh is still running.
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
PILOT_LOG=${PILOT_LOG:-$LOG/pilot_main.log}
mkdir -p "$LOG" /hy-tmp/results

echo "=== waiting for PILOT_ALL_DONE in $PILOT_LOG $(date +%H:%M:%S)"
for i in $(seq 1 720); do
  if grep -q PILOT_ALL_DONE "$PILOT_LOG" 2>/dev/null; then
    echo "pilot finished $(date +%H:%M:%S)"
    break
  fi
  # also succeed if follow_boat drop log exists (script may have been started without tee)
  if [ -f "$LOG/pilot_follow_boat_fallback_drop.log" ] && ! pgrep -f 'run_pilot.sh' >/dev/null; then
    echo "pilot processes gone; treating as done $(date +%H:%M:%S)"
    break
  fi
  sleep 30
done

echo "=== write pilot Table IV $(date +%H:%M:%S)"
/usr/local/bin/python3 "$WS/u0eval/write_u0_table.py" --auto \
  --condition "Full sensing (pilot)" \
  --out /hy-tmp/results/u0_table_full_pilot.md || true
/usr/local/bin/python3 "$WS/u0eval/write_u0_table.py" --auto \
  --condition "DVL dropout (pilot)" --no-paper-rows \
  --out /hy-tmp/results/u0_table_drop_pilot.md || true

echo "=== install extended bridge into devel $(date +%H:%M:%S)"
cp /hy-tmp/u0env/ros_ws/src/bluerov2_control/scripts/ros_gr00t_bridge_ext.py \
   /hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py
chmod +x /hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py

# stop u0env sim leftover from the last pilot episode
for pat in 'run_pilot.sh' 'run_eval_task.sh' 'batch_run_ext' 'parsed_simulator' 'roslaunch' 'rosmaster'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
done
sleep 5

echo "=== opt-underwater hold-anchor (wam-only) $(date +%H:%M:%S)"
bash "$WS/scripts/run_hold_anchor_opt.sh" >"$LOG/hold_anchor_opt.log" 2>&1 || true

echo "=== restart policy servers with mixer hold + dump $(date +%H:%M:%S)"
for p in $(pgrep -f 'wam_policy_server|fallback_policy_server' || true); do kill "$p" 2>/dev/null || true; done
sleep 2
nohup bash "$WS/u0eval/start_server.sh" wam 8000 >/dev/null 2>&1 &
nohup bash "$WS/u0eval/start_server.sh" fallback 8002 >/dev/null 2>&1 &
sleep 8
curl -s -m 5 http://127.0.0.1:8000/health || echo "wam health failed"
curl -s -m 5 http://127.0.0.1:8001/health || echo "u0 health failed"
curl -s -m 5 http://127.0.0.1:8002/health || echo "fallback health failed"

echo "=== b3 collect: 3 goto episodes to seed planner_task dumps $(date +%H:%M:%S)"
bash "$WS/u0eval/run_eval_task.sh" goto_charge_station wam 3 8000 -1.0 zero \
  >"$LOG/b3_collect_goto.log" 2>&1 || true

echo "=== b3 fine-tune (stop U0 briefly to free VRAM) $(date +%H:%M:%S)"
U0PID=$(pgrep -f 'inference_service_u0.py' | head -1 || true)
if [ -n "${U0PID:-}" ]; then kill "$U0PID" || true; sleep 5; fi
bash "$WS/scripts/b3_iterate.sh" /hy-tmp/data/planner_task >"$LOG/b3_iterate.log" 2>&1 || true
# restore U0
nohup bash "$WS/u0eval/start_server.sh" u0 8001 >/dev/null 2>&1 &
sleep 20
# if b3 wrote a ckpt, point the WAM server at it
if [ -f /hy-tmp/models/uwam/best_ou_b3.pt ]; then
  for p in $(pgrep -f 'wam_policy_server|fallback_policy_server' || true); do kill "$p" 2>/dev/null || true; done
  sleep 2
  # keep default ckpt name: copy over best_ou only if the file exists and is newer
  cp -n /hy-tmp/models/uwam/best_ou.pt /hy-tmp/models/uwam/best_ou_pre_b3.pt || true
  cp /hy-tmp/models/uwam/best_ou_b3.pt /hy-tmp/models/uwam/best_ou.pt
  nohup bash "$WS/u0eval/start_server.sh" wam 8000 >/dev/null 2>&1 &
  nohup bash "$WS/u0eval/start_server.sh" fallback 8002 >/dev/null 2>&1 &
  sleep 8
fi

echo "=== b4 arm ensemble + task gate $(date +%H:%M:%S)"
# training needs VRAM; stop the live servers for this block
U0PID=$(pgrep -f 'inference_service_u0.py' | head -1 || true)
if [ -n "${U0PID:-}" ]; then kill "$U0PID" || true; sleep 5; fi
for p in $(pgrep -f 'wam_policy_server|fallback_policy_server' || true); do kill "$p" 2>/dev/null || true; done
bash "$WS/scripts/b4_arm_gate.sh" >"$LOG/b4_arm_gate.log" 2>&1 || true
nohup bash "$WS/u0eval/start_server.sh" u0 8001 >/dev/null 2>&1 &
nohup bash "$WS/u0eval/start_server.sh" wam 8000 >/dev/null 2>&1 &
nohup bash "$WS/u0eval/start_server.sh" fallback 8002 >/dev/null 2>&1 &
sleep 25

echo "=== d1-full paper-scale eval $(date +%H:%M:%S)"
nohup bash "$WS/u0eval/run_full.sh" all >"$LOG/full_all.log" 2>&1 &
echo "FULL_LAUNCHED pid=$! $(date +%H:%M:%S)"
echo "CONTINUE_PLAN_LAUNCHED_FULL $(date +%H:%M:%S)"
