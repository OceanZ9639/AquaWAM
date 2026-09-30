#!/usr/bin/env bash
# Put OUR method first: deploy the scene-diverse core (best_scenes + vel_ens_scenes,
# attitude loop, estimator, coast, live-target follow, gripper-servo grasp) and redo
# every WAM / fallback block. U0 grasp blocks resume afterwards (resumable runner
# skips what is already done).
#
#   1. pause at the current block boundary (U0 grasp block finishes as an orphan)
#   2. restart wam + fallback on the new core at V_MAX x1.8 (nav 0.45 m/s)
#   3. PILOT goto_water_tower / wam / full, 5 eps: >= 3/5 keeps the speed,
#      otherwise restart at x1.0 (new core only)
#   4. archive every WAM full block and every WAM/fallback blind block of the
#      previous calibers (U0 rows and fallback full rows stay)
#   5. relaunch the resumable runner for all arms: locomotion redo, then grasp
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
RUNS=/hy-tmp/u0env/dataset/eval_runs
ARCH=$RUNS/_archive_precore
mkdir -p "$LOG" "$ARCH"
tally() { awk -F, 'NR>1{n++; if($2=="success") s++} END{print s+0"/"n}' "$1"; }
ok_count() { awk -F, 'NR>1 && $2=="success"{s++} END{print s+0}' "$1"; }

restart_servers() {  # $1 = vmax scale
  for p in $(pgrep -f '[w]am_policy_server.py --port 8000|[f]allback_policy_server' || true); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
  WAM_VMAX_SCALE=$1 nohup bash "$WS/u0eval/start_server.sh" wam 8000 >/dev/null 2>&1 &
  WAM_VMAX_SCALE=$1 nohup bash "$WS/u0eval/start_server.sh" fallback 8002 >/dev/null 2>&1 &
  for i in $(seq 1 36); do
    sleep 5
    ok=1
    curl -s -m 5 http://127.0.0.1:8000/health | grep -q healthy || ok=0
    curl -s -m 5 http://127.0.0.1:8002/health | grep -q healthy || ok=0
    [ "$ok" = 1 ] && break
  done
  echo "servers (vmax x$1): $(curl -s -m 5 http://127.0.0.1:8000/health) $(curl -s -m 5 http://127.0.0.1:8002/health)"
  grep -m1 'V_MAX scale\|action library' "$LOG/wam_server.log"
}

echo "=== pause resumers $(date +%H:%M:%S)"
for p in $(pgrep -f '[r]un_full.sh|[a]fter_full.sh' || true); do kill "$p" 2>/dev/null || true; done
echo "=== waiting for in-flight block $(date +%H:%M:%S)"
for i in $(seq 1 600); do
  if ! pgrep -f '[b]atch_run_ext|[r]un_eval_task' >/dev/null; then break; fi
  sleep 15
done
sleep 5
for p in $(pgrep -f '[p]arsed_simulator|[r]oslaunch|[r]osmaster' || true); do kill -9 "$p" 2>/dev/null || true; done
sleep 3

echo "=== new core servers at V_MAX x1.8 $(date +%H:%M:%S)"
restart_servers 1.8

echo "=== speed pilot: goto_water_tower / wam / full x5 $(date +%H:%M:%S)"
mkdir -p "$RUNS/_pilot_speed"
bash "$WS/u0eval/run_eval_task.sh" goto_water_tower wam_speedpilot 5 8000 -1.0 zero \
  >"$LOG/pilot_speed_water_tower.log" 2>&1 || echo "  (nonzero exit)"
PF="$RUNS/wam_speedpilot_full/goto_water_tower/results.csv"
echo "pilot: $(tally "$PF")  durations: $(awk -F, 'NR>1{printf "%s ", $3}' "$PF")"
if [ "$(ok_count "$PF")" -ge 3 ]; then
  echo "SPEED_OK keep x1.8"
  SCALE=1.8
else
  echo "SPEED_FAIL -> x1.0"
  SCALE=1.0
  restart_servers 1.0
fi
echo "FINAL_VMAX_SCALE=$SCALE"

echo "=== archive WAM full + all WAM/fallback blind blocks $(date +%H:%M:%S)"
for t in "$RUNS"/wam_full/*/; do
  [ -f "$t/results.csv" ] || continue
  task=$(basename "$t"); mv "$t" "$ARCH/wam_full__${task}" && echo "  archived wam_full/$task ($(tally "$ARCH/wam_full__${task}/results.csv"))"
done
for d in "$RUNS"/wam_drop*/ "$RUNS"/fallback_drop*/; do
  [ -d "$d" ] || continue
  cond=$(basename "$d")
  for t in "$d"/*/; do
    [ -f "$t/results.csv" ] || continue
    task=$(basename "$t")
    case "$task" in goto_*|scan_*|inspect_*|follow_*) ;; *) continue ;; esac
    mv "$t" "$ARCH/${cond}__${task}" && echo "  archived $cond/$task ($(tally "$ARCH/${cond}__${task}/results.csv"))"
  done
done

echo "=== relaunch runner: all arms, locomotion then grasp $(date +%H:%M:%S)"
nohup bash "$WS/u0eval/run_full.sh" all >>"$LOG/full_all.log" 2>&1 &
echo "run_full pid $!"
sleep 5
nohup bash "$WS/u0eval/after_full.sh" >"$LOG/after_full.log" 2>&1 &
echo "after_full pid $!"
echo "DEPLOY_NEW_CORE_LAUNCHED $(date +%H:%M:%S)"
