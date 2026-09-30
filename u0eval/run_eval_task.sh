#!/usr/bin/env bash
# Run one task x one arm x one condition through the official u0env harness.
#
# Usage:
#   bash run_eval_task.sh <task_code> <arm_tag> <n_episodes> <vla_port> [drop_dvl_at] [dvl_drop_mode]
#
#   task_code     e.g. goto_charge_station (see u0env/exp_setting.csv)
#   arm_tag       u0 | wam | fallback  (naming only; the server on vla_port decides)
#   n_episodes    trials for this task (pilot: 5-10; paper scale: 40/20)
#   vla_port      port of the policy server to evaluate
#   drop_dvl_at   seconds after first action; -1 = full sensing (default)
#   dvl_drop_mode zero | freeze (default zero)
#
# Results are archived to u0env/dataset/eval_runs/<arm>_<cond>/<task_code>/.
set -uo pipefail
# The IDE shell that launches these orchestrators carries HTTP_PROXY=127.0.0.1:17890
# with 0.0.0.0 absent from NO_PROXY; the bridge posts to http://0.0.0.0:<port>/act
# and got 502 Bad Gateway from the proxy. Policy traffic is local: never proxy it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy


TASK=${1:?task_code}
ARM=${2:?arm_tag}
NEP=${3:?n_episodes}
PORT=${4:?vla_port}
DROP=${5:--1.0}
DMODE=${6:-zero}

# Multi-instance knobs (defaults = the single-instance layout used so far):
#   U0ENV            harness root (a second instance uses its own copy, e.g. /hy-tmp/u0env_b)
#   DISPLAY_NUM      X display of this instance's GPU (:1 GPU0, :2 GPU1); X_GPU its nvidia-smi index
#   INSTANCE_ROSPORT base ROS master port of this instance; when set, the stale-process sweep below
#                    only kills processes talking to THIS instance's masters (ports base..base+99)
U0ENV=${U0ENV:-/hy-tmp/u0env}
DISPLAY_NUM=${DISPLAY_NUM:-:1}
COND=full
if [[ "$DROP" != "-1.0" && "$DROP" != "-1" ]]; then COND="drop${DROP}s_${DMODE}"; fi
# USIM-Hard protocol: COND_TAG names the perturbation (e.g. hard_heading); HARD_MODE /
# THRUSTER_ETA / POS_TOL / YAW_TOL / SEQUENTIAL are inherited by batch_run_ext.sh; CONFIG
# selects the task table (exp_setting_hard.csv holds the zero-shot composed tasks).
if [ -n "${COND_TAG:-}" ]; then COND="${COND_TAG}"; fi
CONFIG=${CONFIG:-exp_setting.csv}
DEST="$U0ENV/dataset/eval_runs/${ARM}_${COND}/$TASK"

# --- simulation environment (mirrors with_sim.sh) ---
XDISP=${DISPLAY_NUM#:} XGPU=${X_GPU:-0} bash /hy-tmp/underwater_wam/scripts/start_nvidia_x.sh || true
export DISPLAY=$DISPLAY_NUM
set +u
export PATH=/hy-tmp/envs/ros_env/bin:$PATH
export CONDA_PREFIX=/hy-tmp/envs/ros_env
export LD_LIBRARY_PATH=$U0ENV/build/stonefish_install/lib:/hy-tmp/envs/ros_env/lib:${LD_LIBRARY_PATH:-}
source /hy-tmp/envs/ros_env/setup.bash
source $U0ENV/ros_ws/devel/setup.bash
set -u

# stale harness nodes from a previous run steal the ROS port or, worse, the node
# NAME: a surviving ros_gr00t_bridge makes the master kill the new bridge with
# "new node registered with same name" (observed). Kill every harness process.
mine() {  # is pid $1 part of THIS instance? (an instance without INSTANCE_ROSPORT owns the default 11311 range)
  local base=${INSTANCE_ROSPORT:-11311}
  local uri; uri=$(tr '\0' '\n' < /proc/$1/environ 2>/dev/null | sed -n 's/^ROS_MASTER_URI=http:\/\/[^:]*:\([0-9]*\).*/\1/p')
  [ -n "$uri" ] && [ "$uri" -ge "$base" ] && [ "$uri" -lt $((base + 100)) ]
}
for pat in 'ros_gr00t_bridge' 'vla_data_collector' 'eval_tracking' 'eval_grasping' \
           'eval_follow' 'eval_transporting' 'mapper_setup' 'alpha5_' 'move_group' \
           'rostopic' '/parsed_simulator ' 'roslaunch' 'rosmaster' 'rosout'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    mine "$p" && kill -9 "$p" 2>/dev/null || true
  done
done
sleep 3

cd "$U0ENV"
rm -rf "dataset/eval/$TASK"   # fresh trial dir; prior runs are archived under eval_runs/
echo "=== [$ARM/$COND] $TASK x $NEP episodes  (port $PORT, drop=$DROP/$DMODE)  $(date +%H:%M:%S)"
bash batch_run_ext.sh -m eval -t "$TASK" --vlaport "$PORT" -c "$CONFIG" \
  --eval-num "$NEP" --drop-dvl-at "$DROP" --dvl-drop-mode "$DMODE" \
  ${INSTANCE_ROSPORT:+--rosport "$INSTANCE_ROSPORT"}
rc=$?

mkdir -p "$DEST"
if [ -d "dataset/eval/$TASK" ]; then
  # Archive what the metrics need (results.csv + logs/) plus ONE episode's raw
  # recording (episode0: 4 cameras x 10 Hz jpg+pkl, ~300 MB) for diagnostics.
  # Full recordings of every trial were ~290 MB each and filled 575 GB in two
  # days; tools/evalmetrics reads only results.csv and logs/episode_*_data.csv.
  rm -rf "$DEST/logs" "$DEST/episode0" "$DEST/results.csv"
  cp -r "dataset/eval/$TASK/logs" "$DEST/" 2>/dev/null || true
  cp "dataset/eval/$TASK/results.csv" "$DEST/" 2>/dev/null || true
  # a flag file lets a running queue switch to keep-all for its NEXT blocks without a restart
  # (wrist-camera DAgger: the camera-driven episodes are the next training round's data)
  [ -f "$U0ENV/KEEP_ALL_EPISODES" ] && KEEP_ALL_EPISODES=1
  if [ "${KEEP_ALL_EPISODES:-0}" = 1 ]; then
    # data-collection runs: every trial's recording is the product
    for e in "dataset/eval/$TASK"/episode*/; do [ -d "$e" ] && cp -r "$e" "$DEST/"; done
  else
    [ -d "dataset/eval/$TASK/episode0" ] && cp -r "dataset/eval/$TASK/episode0" "$DEST/"
    # free the working dir's other recordings right away (logs stay: the WAM
    # server reads the trial boundary from them)
    find "dataset/eval/$TASK" -mindepth 1 -maxdepth 1 -type d -name 'episode[1-9]*' -exec rm -rf {} +
  fi
  echo "=== archived to $DEST (results + logs + episode0)"
  echo "--- results.csv ---"
  cat "$DEST/results.csv" 2>/dev/null || echo "(no results.csv)"
fi
exit $rc
