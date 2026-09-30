#!/usr/bin/env bash
# Source RoboStack + u0env and optionally launch a headless USIM scene.
set -euo pipefail

U0="${U0:-/hy-tmp/u0env}"
CONDA_PREFIX="${CONDA_PREFIX:-/hy-tmp/envs/ros_env}"
export CONDA_PREFIX
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"
export LD_LIBRARY_PATH="${U0}/build/stonefish_install/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

# RoboStack/catkin setup scripts use empty-var tests that break under `set -u`.
set +u
# shellcheck disable=SC1091
source "${CONDA_PREFIX}/setup.bash"
# shellcheck disable=SC1091
source "${U0}/ros_ws/devel/setup.bash"
set -u
export CMAKE_PREFIX_PATH="${U0}/build/stonefish_install:${CMAKE_PREFIX_PATH:-}"

if [[ "${1:-}" == "env" ]]; then
  echo "ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}"
  command -v roslaunch
  exit 0
fi

# Headless display: prefer an existing NVIDIA X, else Xvfb + Mesa.
if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=:99
fi
if [[ ! -S /tmp/.X11-unix/X${DISPLAY#:} ]]; then
  mkdir -p /hy-tmp/logs
  Xvfb "${DISPLAY}" -screen 0 1280x720x24 +extension GLX +render -noreset >/hy-tmp/logs/xvfb.log 2>&1 &
  sleep 1
fi
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
if [[ "${UWAM_NVIDIA_GL:-0}" == "1" ]]; then
  unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
else
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
fi

DATA="${U0}/ros_ws/src/stonefish_bluerov2/data"
SCN="${SCN:-${U0}/ros_ws/src/stonefish_bluerov2/scenarios/bluerov2_test.scn}"
MODE="${1:-gpu}"
if [[ "${#}" -ge 2 ]]; then
  SCN="$2"
fi

if [[ "${MODE}" == "nogpu" ]]; then
  exec stdbuf -oL -eL roslaunch stonefish_ros simulator_nogpu.launch \
    simulation_data:="${DATA}" \
    scenario_description:="${SCN}" \
    simulation_rate:="100"
fi

exec stdbuf -oL -eL roslaunch stonefish_ros simulator.launch \
  simulation_data:="${DATA}" \
  scenario_description:="${SCN}" \
  simulation_rate:="100" \
  graphics_resolution:="800 600" \
  graphics_quality:="low"
