#!/usr/bin/env bash
# Paper-scale evaluation, batched + RESUMABLE: blocks whose eval_runs results.csv
# already holds >= the expected episode count are skipped, so the run can be
# paused at block boundaries and relaunched.
#
# Usage: bash run_full.sh [stage]   # locomotion | grasp | all
set -uo pipefail
# The IDE shell that launches these orchestrators carries HTTP_PROXY=127.0.0.1:17890
# with 0.0.0.0 absent from NO_PROXY; the bridge posts to http://0.0.0.0:<port>/act
# and got 502 Bad Gateway from the proxy. Policy traffic is local: never proxy it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

STAGE=${1:-locomotion}
WS=/hy-tmp/underwater_wam
LOG=${LOG_DIR:-/hy-tmp/logs/u0eval}
U0ENV=${U0ENV:-/hy-tmp/u0env}      # exported through to run_eval_task.sh (multi-instance)
export U0ENV
mkdir -p "$LOG"

eps_for() {
  if [ -n "${EPS_OVERRIDE:-}" ]; then echo "$EPS_OVERRIDE"; return; fi
  case "$1" in
    goto_*|pick_*|transfer_*) echo 40 ;;
    scan_*|inspect_*|follow_*) echo 20 ;;
    *) echo 20 ;;
  esac
}
drop_for() {
  case "$1" in
    goto_*) echo 8 ;;
    scan_*|inspect_*) echo 40 ;;
    follow_*) echo 20 ;;
    pick_*|transfer_*) echo 30 ;;
    *) echo 20 ;;
  esac
}
# policy-server ports; a second instance on the same box overrides them (PORT_WAM=8010 ...)
port_for() { case "$1" in u0) echo "${PORT_U0:-8001}";; wam) echo "${PORT_WAM:-8000}";; fallback) echo "${PORT_FALLBACK:-8002}";; wam_percept) echo "${PORT_WAM_PERCEPT:-8003}";; esac; }
ARMS=${ARMS:-"u0 wam fallback"}   # e.g. ARMS=u0 runs only the model-independent U0 blocks
CONDS=${CONDS:-"full drop"}        # e.g. CONDS=drop: U0 full-sensing rows are cited from the paper

done_rows() { [ -f "$1" ] && tail -n +2 "$1" 2>/dev/null | wc -l || echo 0; }

run_block() {  # task arm cond
  local task=$1 arm=$2 cond=$3
  case " $CONDS " in *" $cond "*) ;; *) return 0;; esac
  local port; port=$(port_for "$arm")
  local nep; nep=$(eps_for "$task")
  local drop=-1.0 cname=full
  if [ "$cond" = drop ]; then drop=$(drop_for "$task"); cname="drop${drop}s_zero"; fi
  cname="${COND_PREFIX:-}${cname}"     # e.g. COND_PREFIX=r1_ -> wam_percept_r1_full (DAgger round 1)
  local have; have=$(done_rows "$U0ENV/dataset/eval_runs/${arm}_${cname}/$task/results.csv")
  if [ "$have" -ge "$nep" ]; then
    echo "########## SKIP $task / $arm / $cond (already $have/$nep) $(date +%H:%M:%S)"
    return 0
  fi
  echo "########## FULL $task / $arm / $cond (n=$nep drop=$drop) $(date +%H:%M:%S)"
  COND_TAG="$cname" bash "$WS/u0eval/run_eval_task.sh" "$task" "$arm" "$nep" "$port" "$drop" zero \
    >"$LOG/full_${task}_${arm}_${cond}.log" 2>&1 || echo "  (nonzero exit)"
}

if [ "$STAGE" = all ]; then
  bash "$0" locomotion
  bash "$0" grasp
  echo "FULL_all_DONE $(date +%H:%M:%S)"
  exit 0
fi

if [ "$STAGE" = locomotion ]; then
  TASKS=(goto_charge_station goto_water_tower scan_ship_ancient scan_ship_modern
         inspect_pipeline_pool inspect_pipeline_sea follow_boat)
  for task in "${TASKS[@]}"; do
    for arm in $ARMS; do
      run_block "$task" "$arm" full
      run_block "$task" "$arm" drop
    done
  done
elif [ "$STAGE" = grasp ]; then
  TASKS=(pick_pipe0_shallow pick_pipe1_shallow pick_pipe0_factory pick_pipe1_factory
         pick_red_shallow pick_redx_shallow pick_red_factory pick_redx_factory
         pick_blue_shallow pick_bluex_shallow pick_blue_factory pick_bluex_factory
         transfer_red_shallow)
  # TASK_ORDER=reverse lets a second box work the same list from the other end; results are
  # pulled across so done_rows() skips whatever the other box finished first
  if [ "${TASK_ORDER:-}" = reverse ]; then
    rev=(); for ((i=${#TASKS[@]}-1; i>=0; i--)); do rev+=("${TASKS[$i]}"); done; TASKS=("${rev[@]}")
  fi
  for task in "${TASKS[@]}"; do
    for arm in $ARMS; do
      run_block "$task" "$arm" full
      run_block "$task" "$arm" drop
    done
  done
fi
echo "FULL_${STAGE}_DONE $(date +%H:%M:%S)"
