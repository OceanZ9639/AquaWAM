#!/usr/bin/env bash
# Stop this machine's pulse_queue.sh runs (and their sim/bridge trees), point every instance's
# devel-space bridge at the src file (catkin left a stale COPY there, so src edits were not live),
# wipe the working dirs of the given tasks, and relaunch the dropout queues.
#   usage: relaunch_pulse_drop.sh "<inst> <cuda> <tasks...>" ["<inst> <cuda> <tasks...>" ...]
set -uo pipefail
WS=/hy-tmp/underwater_wam; LOG=/hy-tmp/logs/u0eval; cd $WS
# 1. stop queues + blocks + their ROS trees
for p in $(pgrep -f "u0eval/pulse_queue.sh" || true); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 1
for p in $(pgrep -f "u0eval/run_eval_task.sh" || true); do kill "$p" 2>/dev/null; done
sleep 2
for p in $(ps -eo pid,cmd | awk '/roslaunch|rosmaster|parsed_simulator|gr00t_bridge|data_collector|eval_grasping|eval_tracking|alpha5_|move_group|batch_run_ext/ && !/awk/{print $1}'); do
  kill -9 "$p" 2>/dev/null
done
sleep 3
# 2. devel-space bridge -> symlink to src (every instance root present on this machine)
for root in /hy-tmp/u0env /hy-tmp/u0env_b /hy-tmp/u0env_c /hy-tmp/u0env_d; do
  src=$root/ros_ws/src/bluerov2_control/scripts/ros_gr00t_bridge_ext.py
  dev=$root/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py
  [ -f "$src" ] && [ -e "$dev" ] && { rm -f "$dev"; ln -s "$src" "$dev"; echo "$root: devel bridge -> src ($(grep -c box_rel "$src") box_rel refs)"; }
done
# 3. wipe working dirs and relaunch
for spec in "$@"; do
  set -- $spec; inst=$1; cuda=$2; shift 2
  if [ "$inst" = local ]; then root=/hy-tmp/u0env; else root=$(bash -c "source u0eval/instance_env.sh $inst >/dev/null; echo \$U0ENV"); fi
  for t in "$@"; do rm -rf "$root/dataset/eval/$t"; done
  DROP_T=30 setsid nohup bash u0eval/pulse_queue.sh "$inst" "$cuda" "$@" > "$LOG/pulse_drop_${inst}.log" 2>&1 < /dev/null &
  echo "relaunched $inst: $*"
done
