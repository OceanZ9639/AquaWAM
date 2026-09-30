#!/usr/bin/env bash
# Deploy the blind-path estimator fixes (deployment-trained canonicalized
# ensemble as v_est + takeover coast) and the reviewed task logic (live-target
# follow, gripper-servo grasp, ee_pose bridge), at a block boundary:
#   1. stop the resumers, let the in-flight block finish
#   2. install bridge, restart wam (8000) + fallback (8002)
#   3. archive the blind blocks run under the previous blind caliber
#      (AHRS-yaw code, 21:03 servers): wam_drop* / fallback_drop* with
#      results newer than that restart; sighted blocks are untouched
#   4. relaunch the resumable runner + after_full watcher
set -uo pipefail
# The IDE shell that launches these orchestrators carries HTTP_PROXY=127.0.0.1:17890
# with 0.0.0.0 absent from NO_PROXY; the bridge posts to http://0.0.0.0:<port>/act
# and got 502 Bad Gateway from the proxy. Policy traffic is local: never proxy it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
RUNS=/hy-tmp/u0env/dataset/eval_runs
ARCH=$RUNS/_archive_estimator
SRC=/hy-tmp/u0env/ros_ws/src/bluerov2_control/scripts/ros_gr00t_bridge_ext.py
DST=/hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py
mkdir -p "$LOG" "$ARCH"

echo "=== stop resumers + installer watcher $(date +%H:%M:%S)"
for pat in 'after_full.sh' 'run_full.sh' 'install_grasp_code_on_u0_block'; do
  for p in $(pgrep -f "$pat" || true); do kill "$p" 2>/dev/null || true; done
done

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

echo "=== install bridge (imu_rpy + ee_pose) $(date +%H:%M:%S)"
cp "$SRC" "$DST" && chmod +x "$DST" && echo "  ee_pose refs: $(grep -c ee_pose "$DST")"

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
grep -m1 'vel ensemble' "$LOG/wam_server.log"

echo "=== archive blocks whose controller changed $(date +%H:%M:%S)"
# (a) every WAM full-sensing block: the attitude loop was open in the eval port
#     (vehicle at 7-8 deg tilt), now closed -> the sighted controller changed
# (b) blind blocks run under the previous blind caliber (after the 21:03 restart)
#     -> new estimator + takeover coast + leveling
# U0 rows and fallback full-sensing rows (U0 passthrough) are untouched.
tally() { awk -F, 'NR>1{n++; if($2=="success") s++} END{print s+0"/"n}' "$1"; }
for t in "$RUNS"/wam_full/*/; do
  [ -f "$t/results.csv" ] || continue
  task=$(basename "$t")
  mv "$t" "$ARCH/wam_full__${task}" && echo "  archived wam_full/$task ($(tally "$ARCH/wam_full__${task}/results.csv"))"
done
for d in "$RUNS"/wam_drop*/ "$RUNS"/fallback_drop*/; do
  [ -d "$d" ] || continue
  cond=$(basename "$d")
  for t in "$d"/*/; do
    [ -f "$t/results.csv" ] || continue
    task=$(basename "$t")
    if [ "$(stat -c %Y "$t/results.csv")" -gt "$(date -d '2026-09-03 21:00' +%s)" ]; then
      mv "$t" "$ARCH/${cond}__${task}" && echo "  archived $cond/$task ($(tally "$ARCH/${cond}__${task}/results.csv"))"
    fi
  done
done

echo "=== resume full eval $(date +%H:%M:%S)"
nohup bash "$WS/u0eval/run_full.sh" all >>"$LOG/full_all.log" 2>&1 &
echo "run_full pid $!"
sleep 5
nohup bash "$WS/u0eval/after_full.sh" >"$LOG/after_full.log" 2>&1 &
echo "after_full pid $!"
echo "DEPLOY_ESTIMATOR_ROUND_LAUNCHED $(date +%H:%M:%S)"
