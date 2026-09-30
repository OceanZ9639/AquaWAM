#!/usr/bin/env bash
# Deploy the blind-path fixes (AHRS yaw from IMU attitude + pressure depth) and
# rerun every WAM / fallback dropout block completed so far, single caliber:
#   pause at block boundary -> install bridge -> restart wam+fallback servers ->
#   archive old dropout results -> relaunch the resumable runner (it redoes the
#   archived blocks in task order, then continues with follow_boat + grasp).
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
RUNS=/hy-tmp/u0env/dataset/eval_runs
ARCH=$RUNS/_archive_gyrodrift
mkdir -p "$LOG" "$ARCH"

echo "=== pause resumers $(date +%H:%M:%S)"
for p in $(pgrep -f 'after_full.sh' || true); do kill "$p" 2>/dev/null || true; done
for p in $(pgrep -f 'run_full.sh' || true); do kill "$p" 2>/dev/null || true; done

echo "=== waiting for in-flight block $(date +%H:%M:%S)"
for i in $(seq 1 600); do
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

echo "=== install bridge (imu_rpy key) $(date +%H:%M:%S)"
cp /hy-tmp/u0env/ros_ws/src/bluerov2_control/scripts/ros_gr00t_bridge_ext.py \
   /hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py
chmod +x /hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py

echo "=== restart wam (8000) + fallback (8002) $(date +%H:%M:%S)"
for p in $(pgrep -f 'wam_policy_server.py --port 8000|fallback_policy_server' || true); do
  kill "$p" 2>/dev/null || true
done
sleep 3
nohup bash "$WS/u0eval/start_server.sh" wam 8000 >/dev/null 2>&1 &
nohup bash "$WS/u0eval/start_server.sh" fallback 8002 >/dev/null 2>&1 &
for i in $(seq 1 30); do
  sleep 5
  ok=1
  curl -s -m 5 http://127.0.0.1:8000/health | grep -q healthy || ok=0
  curl -s -m 5 http://127.0.0.1:8002/health | grep -q healthy || ok=0
  [ "$ok" = 1 ] && break
done
curl -s -m 5 http://127.0.0.1:8000/health; curl -s -m 5 http://127.0.0.1:8002/health; echo

echo "=== archive old dropout results (wam + fallback, locomotion) $(date +%H:%M:%S)"
for d in "$RUNS"/wam_drop*/ "$RUNS"/fallback_drop*/; do
  [ -d "$d" ] || continue
  cond=$(basename "$d")
  for t in "$d"/goto_* "$d"/scan_* "$d"/inspect_* "$d"/follow_*; do
    [ -d "$t" ] || continue
    task=$(basename "$t")
    mv "$t" "$ARCH/${cond}__${task}" && echo "  archived $cond/$task"
  done
done

echo "=== resume full eval (resumable runner redoes archived blocks) $(date +%H:%M:%S)"
nohup bash "$WS/u0eval/run_full.sh" all >>"$LOG/full_all.log" 2>&1 &
echo "run_full pid $!"
sleep 5
nohup bash "$WS/u0eval/after_full.sh" >"$LOG/after_full.log" 2>&1 &
echo "after_full pid $!"
echo "FIX_BLIND_ROUND_LAUNCHED $(date +%H:%M:%S)"
