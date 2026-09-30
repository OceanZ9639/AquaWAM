#!/usr/bin/env bash
# USIM-Hard S7: closed-loop data-efficiency / held-out cores on goto (full sensing, 10 eps/task).
# For each core: restart THIS instance's WAM server with WAM_CKPT=<core>, run goto x2, restore.
# Must run with instance_env sourced (U0ENV, DISPLAY_NUM, INSTANCE_ROSPORT, PORT_WAM, X_GPU).
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam
MODELS=/hy-tmp/models/uwam
NEP=${NEP:-10}
CORES=${CORES:-"de_f05 de_f10 de_f25 de_f50 de_f100 ho_nav ho_manip ho_wt ho_scan"}
GPU=${X_GPU:-0}
EVAL_ROOT="$U0ENV/dataset/eval"
restart_wam() {  # $1 = ckpt path
  for p in $(pgrep -f "python3 .*wam_policy_server.py --port $PORT_WAM"); do kill "$p" 2>/dev/null; done
  sleep 3
  CUDA_VISIBLE_DEVICES=$GPU EVAL_ROOT=$EVAL_ROOT WAM_CKPT=$1 WAM_VMAX_SCALE=${WAM_VMAX_SCALE:-1.8} \
    nohup bash "$WS/u0eval/start_server.sh" wam "$PORT_WAM" >/dev/null 2>&1 &
  for i in $(seq 1 40); do sleep 5; curl -s -m 5 "http://127.0.0.1:$PORT_WAM/health" | grep -q healthy && return 0; done
  echo "WARNING: wam server on $PORT_WAM not healthy with $1"; return 1
}
for core in $CORES; do
  ck="$MODELS/$core.pt"
  [ -f "$ck" ] || { echo "skip $core (no checkpoint yet)"; continue; }
  tag="dataeff_${core}"
  have=$(tail -n +2 "$U0ENV/dataset/eval_runs/wam_${tag}/goto_water_tower/results.csv" 2>/dev/null | wc -l)
  if [ "$have" -ge "$NEP" ]; then echo "skip $core (done)"; continue; fi
  echo "=== core $core $(date +%H:%M:%S)"
  restart_wam "$ck" || continue
  for task in goto_charge_station goto_water_tower; do
    COND_TAG="$tag" bash "$WS/u0eval/run_eval_task.sh" "$task" wam "$NEP" "$PORT_WAM" -1.0 zero \
      >"${LOG_DIR:-/hy-tmp/logs/u0eval}/dataeff_${core}_${task}.log" 2>&1 || true
    tail -n +2 "$U0ENV/dataset/eval_runs/wam_${tag}/$task/results.csv" | awk -F, -v c=$core -v t=$task '{n++; if($2=="success")s++} END{print c, t, s"/"n}'
  done
done
echo "=== restore deployed core $(date +%H:%M:%S)"
restart_wam "$MODELS/best_scenes.pt"
echo "DATAEFF_CLOSEDLOOP_DONE $(date +%H:%M:%S)"
