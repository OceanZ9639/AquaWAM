#!/usr/bin/env bash
# Before follow_boat / grasp blocks: install the bridge that publishes
# state.ee_pose and restart BOTH policy servers (wam 8000, fallback 8002) with
# the reviewed code (live-target follow, gripper-servo grasp, no stale
# waypoints in the blind path). Waits for a u0 block (8000/8002 idle). One-shot.
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
SRC=/hy-tmp/u0env/ros_ws/src/bluerov2_control/scripts/ros_gr00t_bridge_ext.py
DST=/hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py
while true; do
  last=$(grep -E '########## FULL' "$LOG/full_all.log" 2>/dev/null | tail -1)
  if echo "$last" | grep -q '/ u0 /'; then
    echo "u0 block active ($last) $(date +%H:%M:%S)"
    sleep 30
    cp "$SRC" "$DST" && chmod +x "$DST" && echo "bridge installed (ee_pose refs: $(grep -c ee_pose "$DST"))"
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
    if [ "$ok" = 1 ]; then echo "SERVERS_RESTARTED_OK $(date +%H:%M:%S)"; exit 0; fi
    echo "health failed; retry in 60s"; sleep 60
  else
    sleep 60
  fi
done
