#!/usr/bin/env bash
# WAM manipulation with the world-model pulse planner (hull fine positioning chosen by imagination,
# primitive stage machine, arm at the armed pose), 40 episodes per task, archived under wam_gp_wam/.
#   usage: [DROP_T=30] pulse_queue.sh <instance: local|A|B> <cuda devices ("" = CPU)> <task> [task ...]
# DROP_T (default -1 = full sensing) injects a DVL dropout at t=DROP_T s (zero mode) and archives under
# wam_gp_wam_drop${DROP_T}s_zero/ -- the manipulation counterpart of the main table's dropout column.
# The server for this instance is (re)started with --grasp-planner wam and 0.5 s replanning (EXEC_STEPS=5).
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam; LOG=/hy-tmp/logs/u0eval; export PYTHONPATH=$WS; cd $WS; mkdir -p $LOG
INST=$1; CUDA=$2; shift 2
DROP_T=${DROP_T:--1.0}
NEP=${NEP:-40}
if [ "$DROP_T" = "-1.0" ] || [ "$DROP_T" = "-1" ]; then TAG=gp_wam; else TAG="gp_wam_drop${DROP_T}s_zero"; fi
# TAG_OVERRIDE names the archive (e.g. wrist / wrist_drop30s_zero); WAM_WRIST_POSE / WAM_OBJ_SOURCE are
# inherited by the server: wrist-camera head checkpoint and whether it DRIVES (wrist) or shadows (privileged)
TAG=${TAG_OVERRIDE:-$TAG}
if [ "$INST" = local ]; then
  export U0ENV=/hy-tmp/u0env PORT_WAM=8000
else
  source u0eval/instance_env.sh "$INST" >/dev/null
fi
healthy() { curl -s -m 5 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q healthy; }
for p in $(pgrep -f "wam_policy_server.py --port $PORT_WAM" || true); do kill "$p" 2>/dev/null; done; sleep 2
CUDA_VISIBLE_DEVICES="$CUDA" OMP_NUM_THREADS=16 EVAL_ROOT=$U0ENV/dataset/eval WAM_VMAX_SCALE=1.8 \
  WAM_VEL_ENS=/hy-tmp/models/uwam/vel_ens_v2.pt WAM_DUMP_DIR="" WAM_GRASP_PLANNER=wam \
  nohup bash u0eval/start_server.sh wam "$PORT_WAM" >"$LOG/wam_server_pulse_$PORT_WAM.out" 2>&1 &
for i in $(seq 1 60); do healthy "$PORT_WAM" && break; sleep 5; done
healthy "$PORT_WAM" || { echo "server $PORT_WAM did not come up"; exit 1; }
echo "wam pulse server $PORT_WAM up ($INST, cuda='$CUDA', wrist='${WAM_WRIST_POSE:-}', obj='${WAM_OBJ_SOURCE:-privileged}') $(date +%H:%M:%S)"
for t in "$@"; do
  if [ -f "$U0ENV/dataset/eval_runs/wam_$TAG/$t/results.csv" ] && [ "$(tail -n +2 "$U0ENV/dataset/eval_runs/wam_$TAG/$t/results.csv" | wc -l)" -ge "$NEP" ]; then
    echo "--- [$INST] $TAG/$t already complete"; continue
  fi
  echo "--- [$INST] $TAG/$t x$NEP $(date +%H:%M:%S)"
  rm -rf "$U0ENV/dataset/eval/$t"    # the official runner appends to a surviving results.csv
  EXEC_STEPS=5 COND_TAG=$TAG bash u0eval/run_eval_task.sh "$t" wam "$NEP" "$PORT_WAM" "$DROP_T" zero >"$LOG/${TAG}_$t.log" 2>&1 || true
  echo "    [$INST] $TAG/$t: $(tail -n +2 "$U0ENV/dataset/eval_runs/wam_$TAG/$t/results.csv" 2>/dev/null | cut -d, -f2 | sort | uniq -c | tr '\n' ' ')  $(date +%H:%M:%S)"
done
echo "PULSE_QUEUE_DONE $INST $(date +%H:%M:%S)"
