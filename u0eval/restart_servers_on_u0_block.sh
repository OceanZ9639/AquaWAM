#!/usr/bin/env bash
# Restart wam + fallback policy servers to pick up code fixes, but only while a
# "/ u0 /" block is running (ports 8000/8002 idle then). One-shot.
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
FULL_LOG="$LOG/full_all.log"

while true; do
  last=$(grep '##########' "$FULL_LOG" 2>/dev/null | tail -1)
  if echo "$last" | grep -q '/ u0 /'; then
    echo "u0 block active ($last); restarting wam+fallback $(date +%H:%M:%S)"
    for p in $(pgrep -f 'wam_policy_server|fallback_policy_server' || true); do
      kill "$p" 2>/dev/null || true
    done
    sleep 3
    nohup bash "$WS/u0eval/start_server.sh" wam 8000 >/dev/null 2>&1 &
    nohup bash "$WS/u0eval/start_server.sh" fallback 8002 >/dev/null 2>&1 &
    sleep 15
    ok=1
    curl -s -m 5 http://127.0.0.1:8000/health | grep -q healthy || ok=0
    curl -s -m 5 http://127.0.0.1:8002/health | grep -q healthy || ok=0
    if [ "$ok" = 1 ]; then
      echo "SERVERS_RESTARTED_OK $(date +%H:%M:%S)"
      exit 0
    fi
    echo "health check failed; retrying in 30s"
    sleep 30
  else
    sleep 20
  fi
done
