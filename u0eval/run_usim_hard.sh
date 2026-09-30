#!/usr/bin/env bash
# USIM-Hard protocol blocks (resumable). Same simulator, same scenes, same USIM data -- only the
# evaluation protocol changes, and every arm meets the identical seeded perturbation per episode.
#
#   H1 heading   start yaw rotated by 90-180 deg (USIM demos always start facing the goal)
#   H2 turbidity Jerlov 0.50 water (official 0.15): vision-only perturbation
#   H3 thruster  one horizontal thruster at 50 % efficiency, unobserved by the policy
#   H4 tight     goto endpoint ball 1.0 m -> 0.5 m
#   H5 zero-shot composed tasks never in USIM (round trip / full loop / new depth), sequential judge
#
# Usage: bash run_usim_hard.sh [h1|h2|h3|h4|h5|all]
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam
LOG=${LOG_DIR:-/hy-tmp/logs/usim_hard}
U0ENV=${U0ENV:-/hy-tmp/u0env}
export U0ENV
mkdir -p "$LOG"
NEP=${NEP:-20}
STAGE=${1:-all}
# ARMS: which arms this instance runs (default all). The U0-dependent arms need the machine that
# hosts the U0 server; WAM-only arms can run anywhere.
ARMS=${ARMS:-"u0 wam fallback wam_percept"}
port_for() { case "$1" in u0) echo "${PORT_U0:-8001}";; wam) echo "${PORT_WAM:-8000}";; fallback) echo "${PORT_FALLBACK:-8002}";; wam_percept) echo "${PORT_WAM_PERCEPT:-8003}";; esac; }
want() { case " $ARMS " in *" $1 "*) return 0;; *) return 1;; esac; }
done_rows() { [ -f "$1" ] && tail -n +2 "$1" 2>/dev/null | wc -l || echo 0; }

# block <cond_tag> <task> <arm> [drop_s]   (perturbation env vars are inherited from the caller)
block() {
  local tag=$1 task=$2 arm=$3 drop=${4:--1.0}
  want "$arm" || return 0
  local port; port=$(port_for "$arm")
  local have; have=$(done_rows "$U0ENV/dataset/eval_runs/${arm}_${tag}/$task/results.csv")
  if [ "$have" -ge "$NEP" ]; then
    echo "########## SKIP $tag / $task / $arm (already $have/$NEP) $(date +%H:%M:%S)"; return 0
  fi
  echo "########## HARD $tag / $task / $arm (n=$NEP drop=$drop) $(date +%H:%M:%S)"
  COND_TAG="$tag" bash "$WS/u0eval/run_eval_task.sh" "$task" "$arm" "$NEP" "$port" "$drop" zero \
    >"$LOG/hard_${tag}_${task}_${arm}.log" 2>&1 || echo "  (nonzero exit)"
}

h1() {  # heading
  for task in goto_charge_station goto_water_tower scan_ship_modern inspect_pipeline_pool; do
    for arm in u0 wam wam_percept; do HARD_MODE=heading block hard_heading "$task" "$arm"; done
  done
}
h2() {  # turbidity (vision): state-goal WAM is the immune control, wam_percept is the vision-WAM
  for arm in u0 wam wam_percept; do HARD_MODE=jerlov=0.5 block hard_jerlov05 goto_water_tower "$arm"; done
  for arm in u0 wam_percept; do HARD_MODE=jerlov=0.5 block hard_jerlov05 pick_pipe0_shallow "$arm"; done
}
h3() {  # actuator fault: thruster 3 (index 2, seen in OU regimes) then thruster 1 (index 0, unseen)
  for task in goto_charge_station goto_water_tower; do
    for arm in u0 wam wam_percept; do THRUSTER_ETA="1,1,0.5,1,1,1,1,1" block hard_eta2_05 "$task" "$arm"; done
  done
  for task in goto_charge_station goto_water_tower; do
    for arm in u0 wam wam_percept; do THRUSTER_ETA="0.5,1,1,1,1,1,1,1" block hard_eta0_05 "$task" "$arm"; done
  done
}
h4() {  # tight endpoint
  for task in goto_charge_station goto_water_tower; do
    for arm in u0 wam wam_percept; do POS_TOL=0.5 block hard_tight05 "$task" "$arm"; done
  done
}
h5() {  # zero-shot composed tasks (task table exp_setting_hard.csv)
  for arm in u0 wam; do
    CONFIG=exp_setting_hard.csv HARD_MODE=roundtrip SEQUENTIAL=true block hard_zeroshot goto_water_tower_roundtrip "$arm"
    CONFIG=exp_setting_hard.csv HARD_MODE=depth=-1.2 block hard_zeroshot goto_water_tower_depth "$arm"
    CONFIG=exp_setting_hard.csv HARD_MODE=loop SEQUENTIAL=true block hard_zeroshot scan_ship_loop "$arm"
  done
}

case "$STAGE" in
  h1) h1 ;; h2) h2 ;; h3) h3 ;; h4) h4 ;; h5) h5 ;;
  all) h5; h1; h4; h3; h2 ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "USIM_HARD_${STAGE}_DONE $(date +%H:%M:%S)"
