#!/usr/bin/env bash
# Prepare 10 Hz DVL runtime scene, launch official BlueROV2, collect 6 REGIMES.
set -euo pipefail
ROOT=/hy-tmp
U0="${U0:-/hy-tmp/u0env}"
OUT="${OUT:-/hy-tmp/data/ou_explore}"
LOG=/hy-tmp/logs/u0env_ou.log
FRAMES="${FRAMES:-2700}"
export DISPLAY="${DISPLAY:-:99}"

export PYTHONPATH=/hy-tmp/underwater_wam:${PYTHONPATH:-}
/usr/local/bin/python3 - <<'PY'
from pathlib import Path
from uwam.sim import prepare_ou_scene
p = prepare_ou_scene(Path("/hy-tmp/u0env"), Path("/hy-tmp/data/ou_explore/scn"),
                     dvl_rate=10.0, camera_rate=10.0, fls_rate=10.0)
print("SCN", p)
PY

SCN=/hy-tmp/data/ou_explore/scn/bluerov2_test_runtime.scn
# stop previous simulator by exact binary pid only
for p in $(ps -eo pid,cmd | awk '/\/parsed_simulator / && !/awk/{print $1}'); do
  kill "$p" 2>/dev/null || true
done
sleep 2
if [[ ! -S /tmp/.X11-unix/X${DISPLAY#:} ]]; then
  Xvfb "${DISPLAY}" -screen 0 1280x720x24 +extension GLX +render -noreset >/hy-tmp/logs/xvfb.log 2>&1 &
  sleep 1
fi
# Prefer NVIDIA GL when dummy X :1 is up
if [[ -S /tmp/.X11-unix/X1 ]]; then
  export DISPLAY=:1
  export UWAM_NVIDIA_GL=1
fi
nohup bash /hy-tmp/underwater_wam/scripts/run_u0env.sh gpu "${SCN}" >"${LOG}" 2>&1 &
echo "sim pid $! log ${LOG}"
# wait for DVL
set +u
export PATH=/hy-tmp/envs/ros_env/bin:$PATH
export CONDA_PREFIX=/hy-tmp/envs/ros_env
export LD_LIBRARY_PATH=/hy-tmp/u0env/build/stonefish_install/lib:/hy-tmp/envs/ros_env/lib:${LD_LIBRARY_PATH:-}
source /hy-tmp/envs/ros_env/setup.bash
source /hy-tmp/u0env/ros_ws/devel/setup.bash
set -u
export ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:11311
for i in $(seq 1 180); do
  if timeout 3 rostopic echo -n 1 /bluerov2/dvl_sim >/dev/null 2>&1; then
    echo "DVL live after ${i} tries"
    break
  fi
  sleep 5
  if [[ "$i" -eq 180 ]]; then
    echo "DVL never appeared"; tail -40 "${LOG}"; exit 1
  fi
done
python /hy-tmp/underwater_wam/scripts/collect_ou.py --all --frames "${FRAMES}" --out "${OUT}" --rgb
echo "OU campaign done"
