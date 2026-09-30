#!/usr/bin/env bash
# Reset the rendering stack of one simulator instance: kill that instance's ROS tree (by ROS master port
# range, nothing else is touched) and restart its headless NVIDIA X server. Cure for the "Stonefish dies
# at startup (exit -5) / cameras never publish, policy never queried, every episode times out" disease
# that an X server develops after a long GPU job ran next to it (seen on box 3 both GPUs, box 2 both GPUs).
#   usage: reset_instance_x.sh <A|B|C|D>
INST=${1:?A|B|C|D}
case "$INST" in
  A) ROS0=11311; DISP=1; GPU=0 ;;
  B) ROS0=12311; DISP=2; GPU=1 ;;
  C) ROS0=13311; DISP=3; GPU=0 ;;
  D) ROS0=14311; DISP=4; GPU=1 ;;
  *) echo "instance must be A, B, C or D"; exit 1 ;;
esac
# a single-GPU box carries both displays on GPU 0
[ "$(nvidia-smi -L | wc -l)" -lt 2 ] && GPU=0
n=0
for p in $(ps -eo pid,cmd | awk '/roslaunch|rosmaster|parsed_simulator|gr00t_bridge|data_collector|eval_tracking|eval_grasping|eval_transporting|eval_follow|batch_run_ext|rosout|robot_state_publisher|move_group/ && !/awk/{print $1}'); do
  uri=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^ROS_MASTER_URI=http:\/\/[^:]*:\([0-9]*\).*/\1/p')
  if [ -n "$uri" ] && [ "$uri" -ge $ROS0 ] && [ "$uri" -lt $((ROS0 + 100)) ]; then kill -9 "$p" 2>/dev/null; n=$((n + 1)); fi
done
pkill -f "Xorg :$DISP " ; sleep 3; rm -f /tmp/.X11-unix/X$DISP /tmp/.X$DISP-lock
XDISP=$DISP XGPU=$GPU bash /hy-tmp/underwater_wam/scripts/start_nvidia_x.sh >/dev/null 2>&1
sleep 3
if ps -eo args | grep -q "[X]org :$DISP "; then
  echo "[reset_instance_x] $INST: killed $n ROS processes, Xorg :$DISP restarted on GPU $GPU $(date +%H:%M:%S)"
else
  echo "[reset_instance_x] $INST: killed $n ROS processes, Xorg :$DISP FAILED to start $(date +%H:%M:%S)"; exit 1
fi
# roslaunch chokes on an oversized log dir; keep only the last day
find /root/.ros/log -mindepth 1 -maxdepth 1 -type d -mtime +1 -exec rm -rf {} + 2>/dev/null
exit 0
