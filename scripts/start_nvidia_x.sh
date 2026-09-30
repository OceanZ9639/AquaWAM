#!/usr/bin/env bash
# Headless NVIDIA X on :1 so Stonefish can render cameras/FLS (Xvfb is Mesa-only).
set -euo pipefail
# XDISP: display number (default 1); XGPU: nvidia-smi index of the GPU to render on (default 0).
# A second simulator instance on a 2-GPU box uses XDISP=2 XGPU=1.
XDISP=${XDISP:-1}
XGPU=${XGPU:-0}
CONF=/hy-tmp/logs/xorg-nvidia-${XDISP}.conf
mkdir -p /hy-tmp/logs /etc/X11
BUS=$(nvidia-smi -i "${XGPU}" --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d ' ')
# 00000000:01:00.0 -> PCI:1:0:0
PCI=$(python3 - <<PY
b="${BUS}"
p=b.split(":")
print("PCI:%d:%d:%d" % (int(p[1],16), int(p[2].split(".")[0],16), int(p[2].split(".")[1],16)))
PY
)
cat > "${CONF}" <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection
Section "Device"
    Identifier "NVIDIA"
    Driver "nvidia"
    BusID "${PCI}"
    Option "AllowEmptyInitialConfiguration" "True"
    Option "UseDisplayDevice" "None"
    Option "ConnectedMonitor" "DFP-0"
    Option "CustomEDID" "DFP-0:/hy-tmp/logs/edid.bin"
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "NVIDIA"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1280 720
    EndSubSection
EndSection
EOF
# skip CustomEDID if no file
sed -i '/CustomEDID/d' "${CONF}"
if ! command -v Xorg >/dev/null 2>&1; then
  echo "Xorg missing; install xserver-xorg-core"
  exit 1
fi
if [[ -S /tmp/.X11-unix/X${XDISP} ]]; then
  # a socket alone is not proof of a server: after a container stop/start the sockets and locks
  # survive while no Xorg runs, the simulator then starts without a GL context and publishes no
  # camera/sensor topics (policy never called, every episode a silent timeout). Check the process.
  if pgrep -f "Xorg :${XDISP} " >/dev/null; then
    echo "DISPLAY :${XDISP} already up"
    exit 0
  fi
  echo "stale socket for :${XDISP} (no Xorg process) -- removing and restarting"
  rm -f /tmp/.X11-unix/X${XDISP} /tmp/.X${XDISP}-lock
fi
Xorg :${XDISP} -config "${CONF}" -noreset +extension GLX +extension RANDR +extension RENDER >/hy-tmp/logs/xorg-nvidia-${XDISP}.log 2>&1 &
sleep 2
if [[ -S /tmp/.X11-unix/X${XDISP} ]]; then
  echo "NVIDIA Xorg on :${XDISP} (GPU ${XGPU}) pid $!"
else
  echo "Xorg :${XDISP} failed"; tail -30 /hy-tmp/logs/xorg-nvidia-${XDISP}.log; exit 1
fi
