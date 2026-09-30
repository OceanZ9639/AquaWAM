#!/usr/bin/env bash
# Build the USIM paper simulator: u0env = Stonefish + stonefish_ros + BlueROV2.
set -euo pipefail

ROOT="${1:-/hy-tmp/u0env}"
CONDA_PREFIX="${CONDA_PREFIX:-/hy-tmp/envs/ros_env}"
GH_PROXIES=(
  "https://gitclone.com/github.com"
  "https://kkgithub.com"
  "https://ghproxy.net/https://github.com"
  "https://github.com"
)

log() { printf '[setup_u0env] %s\n' "$*"; }

clone_try() {
  local dest="$1" spec="$2" branch="$3"
  if [[ -d "$dest/.git" ]]; then
    log "already cloned $dest"
    return 0
  fi
  rm -rf "$dest"
  local owner_repo="${spec}"
  for base in "${GH_PROXIES[@]}"; do
    local url="${base}/${owner_repo}.git"
    log "clone $url -> $dest (branch $branch)"
    if GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$branch" "$url" "$dest"; then
      return 0
    fi
    rm -rf "$dest"
  done
  log "FAILED to clone $spec"
  return 1
}

log "u0env root = $ROOT"
mkdir -p "$ROOT"

clone_try "$ROOT/Stonefish" "patrykcieslak/stonefish" "v1.5"
clone_try "$ROOT/ros_ws/src/stonefish_ros" "VincentGu2000/stonefish_ros" "v1.4-fixed"

# Alpha5 description (public ROS2 driver repo; meshes may be license-limited)
if [[ ! -d "$ROOT/ros_ws/src/description_alpha/origin/alpha_description" ]]; then
  mkdir -p "$ROOT/ros_ws/src/description_alpha/origin"
  if [[ -d /hy-tmp/src/alpha/alpha_description ]]; then
    cp -a /hy-tmp/src/alpha/alpha_description "$ROOT/ros_ws/src/description_alpha/origin/alpha_description"
    log "copied public alpha_description"
  fi
fi

if [[ -d "$ROOT/ros_ws/src/description_alpha/origin/alpha_description/meshes" ]]; then
  if python3 "$ROOT/tools/dataprocess/pkg_install.py" --project-root "$ROOT"; then
    log "generated alpha description files"
  else
    log "pkg_install.py failed (often missing proprietary STL meshes). BlueROV2-only scenes still work."
  fi
fi

if [[ ! -x "$CONDA_PREFIX/bin/python" ]]; then
  log "ROS conda env missing at $CONDA_PREFIX"
  log "Create it with: micromamba create -p $CONDA_PREFIX -c conda-forge -c robostack-noetic python=3.11 ros-noetic-desktop"
  exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_PREFIX/etc/profile.d/conda.sh" 2>/dev/null || true
export PATH="$CONDA_PREFIX/bin:$PATH"

log "installing glm/sdl2/freetype + extra ROS pkgs if missing"
mamba install -y -p "$CONDA_PREFIX" -c conda-forge glm sdl2 freetype compilers cmake make pkg-config \
  || conda install -y -p "$CONDA_PREFIX" -c conda-forge glm sdl2 freetype cmake make

# optional extras from u0env README
mamba install -y -p "$CONDA_PREFIX" -c robostack-noetic -c conda-forge \
  ros-noetic-ros-control ros-noetic-perception ros-noetic-moveit || true
"$CONDA_PREFIX/bin/python" -m pip install trimesh fast-simplification pandas matplotlib || true

log "building Stonefish"
mkdir -p "$ROOT/build/stonefish_build"
cmake -S "$ROOT/Stonefish" -B "$ROOT/build/stonefish_build" -DCMAKE_INSTALL_PREFIX="$ROOT/build/stonefish_install"
cmake --build "$ROOT/build/stonefish_build" -j"$(nproc)"
cmake --install "$ROOT/build/stonefish_build"

log "building cpp_env FishSim"
mkdir -p "$ROOT/build/mysim_build"
cmake -S "$ROOT/cpp_env" -B "$ROOT/build/mysim_build" -DCMAKE_PREFIX_PATH="$ROOT/build/stonefish_install"
cmake --build "$ROOT/build/mysim_build" -j"$(nproc)"

log "building ROS workspace"
# Upstream src/CMakeLists.txt is a machine-specific catkin symlink; rewrite it.
cat > "$ROOT/ros_ws/src/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.0.2)
include(${CONDA_PREFIX}/share/catkin/cmake/toplevel.cmake)
EOF
touch "$ROOT/ros_ws/src/description_alpha/origin/CATKIN_IGNORE" 2>/dev/null || true
# ROS1 genmsg needs classic empy 3.x (em.RAW_OPT). conda-forge often ships 4.x.
"$CONDA_PREFIX/bin/python" -m pip install -q 'empy==3.3.4' || true
export CMAKE_PREFIX_PATH="$ROOT/build/stonefish_install:${CMAKE_PREFIX_PATH:-}"
cd "$ROOT/ros_ws"
rm -rf build devel
# catkin_make lives in the ROS env
"$CONDA_PREFIX/bin/catkin_make" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 || catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5

log "done. Headless test:"
log "  export DISPLAY=:99; Xvfb :99 -screen 0 1280x720x24 &"
log "  source $CONDA_PREFIX/etc/profile.d/conda.sh && conda activate $CONDA_PREFIX"
log "  source $ROOT/ros_ws/devel/setup.bash"
log "  # BlueROV2 without proprietary arm meshes:"
log "  roslaunch stonefish_bluerov2 cpp_env.launch"
log "  # or FishSim: $ROOT/build/mysim_build/FishSim"
