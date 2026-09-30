#!/usr/bin/env bash
# Continue a LeRobot v2 eval after an in-flight rec-queue exits.
#   WAIT_PAT  pgrep pattern that must disappear before we start (empty = start now)
#   ARM / RUN_TAG / *_CKPT / ES / INST / GPU / KIND / NEP  as for lerobot_rec_queue.sh
#   remaining args = tasks
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
mkdir -p "$LOG"
cd "$WS"
ts() { date +%m-%d\ %H:%M:%S; }
WAIT_PAT=${WAIT_PAT:-}
if [ -n "$WAIT_PAT" ]; then
  echo "[$(ts)] waiting for '$WAIT_PAT' to exit"
  while pgrep -f "$WAIT_PAT" >/dev/null; do sleep 60; done
  echo "[$(ts)] predecessor gone"
fi
echo "[$(ts)] ARM=$ARM INST=${1:-?} GPU=${2:-?} KIND=${3:-?} ES=${4:-?} tasks=${*:5}"
exec bash u0eval/lerobot_rec_queue.sh "$@"
