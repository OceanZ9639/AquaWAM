#!/usr/bin/env bash
# Bring up the NVIDIA-GL Stonefish scene, wait for the DVL, then run "$@" under RoboStack python.
# Usage: bash scripts/with_sim.sh python /path/script.py --flag
set -euo pipefail
SCN=/hy-tmp/data/ou_explore/scn/bluerov2_test_runtime.scn
LOG=/hy-tmp/logs/u0env_nvidia.log

bash /hy-tmp/underwater_wam/scripts/start_nvidia_x.sh || true
export DISPLAY=:1
export UWAM_NVIDIA_GL=1

export PYTHONPATH=/hy-tmp/underwater_wam:${PYTHONPATH:-}
/usr/local/bin/python3 - <<'PY'
from pathlib import Path
from uwam.sim import prepare_ou_scene
p = prepare_ou_scene(Path("/hy-tmp/u0env"), Path("/hy-tmp/data/ou_explore/scn"),
                     dvl_rate=10.0, camera_rate=10.0, fls_rate=10.0)
print("SCN", p)
PY

for pat in '/parsed_simulator ' 'roslaunch stonefish' 'rosmaster --core'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
done
for _ in $(seq 1 15); do
  if ! ps -eo cmd | grep -q '[p]arsed_simulator '; then break; fi
  sleep 1
done
sleep 2
nohup bash /hy-tmp/underwater_wam/scripts/run_u0env.sh gpu "${SCN}" >"${LOG}" 2>&1 &
echo "sim launcher $!"

set +u
export PATH=/hy-tmp/envs/ros_env/bin:$PATH
export CONDA_PREFIX=/hy-tmp/envs/ros_env
export LD_LIBRARY_PATH=/hy-tmp/u0env/build/stonefish_install/lib:/hy-tmp/envs/ros_env/lib:${LD_LIBRARY_PATH:-}
source /hy-tmp/envs/ros_env/setup.bash
source /hy-tmp/u0env/ros_ws/devel/setup.bash
set -u
export ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:11311
export PYTHONPATH=/hy-tmp/underwater_wam:/usr/local/lib/python3.11/dist-packages:${PYTHONPATH:-}

# Require DVL *and* odometry, twice in a row, so a dying previous instance cannot look ready.
streak=0
for i in $(seq 1 90); do
  if timeout 3 rostopic echo -n 1 /bluerov2/dvl_sim >/dev/null 2>&1 \
     && timeout 3 rostopic echo -n 1 /bluerov2/odometry >/dev/null 2>&1; then
    streak=$((streak + 1))
  else
    streak=0
  fi
  if [[ "${streak}" -ge 2 ]]; then
    echo "sim live (DVL+odometry) after ${i} tries"
    break
  fi
  sleep 4
  if [[ "$i" -eq 90 ]]; then
    echo "sim never became ready"; tail -40 "${LOG}"; exit 1
  fi
done

# Currents are intentionally NOT enabled: this Stonefish build is a no-op for
# /bluerov2/ocean_current while currents are off, and calling /stonefish_simulator/enable_currents
# drives the vehicle dynamics to NaN (see logs/uwam/current_probe*.json). Set UWAM_ENABLE_CURRENTS=1
# to reproduce that failure.
if [[ "${UWAM_ENABLE_CURRENTS:-0}" == "1" ]]; then
  timeout 15 rosservice call /stonefish_simulator/enable_currents || echo "WARNING: enable_currents failed"
fi

"$@"
