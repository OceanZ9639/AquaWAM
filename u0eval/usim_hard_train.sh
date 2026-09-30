#!/usr/bin/env bash
# USIM-Hard P2/P3: data-efficiency and held-out-task cores, trained *sequentially* (one GPU job at
# a time, CPU niced) so the running U0/WAM evaluation servers are never starved. Pure USIM only:
# no OU / planner / self-collected data enters any of these cores.
#
#   P2 data efficiency : seeded stratified 5/10/25/50/100 % of USIM train episodes
#                        epochs scaled so small fractions are not step-starved (best ckpt by test metric)
#   P3 held-out tasks  : ho_nav   = locomotion tasks only (goto/scan/inspect/follow) -> read grasp tasks
#                        ho_manip = manipulation tasks only                          -> read locomotion
#                        ho_wt    = everything except "Go to the water tower"        -> read water tower
#                        ho_scan  = everything except "Scan the ship"                -> read scan
# Then eval_per_task.py on every core + the deployed best_scenes reference.
set -u
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/usim_hard
OUT=/hy-tmp/results/usim_hard
MODELS=/hy-tmp/models/uwam
mkdir -p "$LOG" "$OUT"
cd "$WS"
PY=/usr/local/bin/python3
WORKERS=${WORKERS:-4}

train_core() {  # name epochs extra-args...
  local name=$1 epochs=$2; shift 2
  if [ -f "$MODELS/$name.pt" ]; then echo "skip $name (exists)"; return; fi
  echo "=== train $name epochs=$epochs $* $(date +%m-%d\ %H:%M)"
  nice -n 10 $PY scripts/train_stage1.py --epochs "$epochs" --batch-size 128 --workers "$WORKERS" \
    --ckpt-name "$name" "$@" >"$LOG/train_$name.log" 2>&1 || echo "TRAIN FAILED $name"
  tail -1 "$LOG/train_$name.log" | cut -c1-200
}

# ---- P2 data efficiency (pure USIM; epochs ~ 8/frac capped at 40) ----
train_core de_f05  40 --usim-frac 0.05
train_core de_f10  40 --usim-frac 0.10
train_core de_f25  32 --usim-frac 0.25
train_core de_f50  16 --usim-frac 0.50
train_core de_f100  8 --usim-frac 1.00

# ---- P3 held-out tasks (pure USIM) ----
train_core ho_nav    8 --usim-include 0,3,4,5,8
train_core ho_manip  8 --usim-include 1,2,6,7
train_core ho_wt     8 --usim-exclude 8
train_core ho_scan   8 --usim-exclude 3

echo "=== per-task offline metrics $(date +%m-%d\ %H:%M)"
CK=$MODELS/best_scenes.pt
for n in de_f05 de_f10 de_f25 de_f50 de_f100 ho_nav ho_manip ho_wt ho_scan; do
  [ -f "$MODELS/$n.pt" ] && CK="$CK,$MODELS/$n.pt"
done
nice -n 10 $PY scripts/eval_per_task.py --ckpts "$CK" --out "$OUT/per_task.json" >"$LOG/per_task.log" 2>&1 \
  || echo "PER_TASK FAILED"
grep -v Warning "$LOG/per_task.log" | grep -E '^(de_|ho_|best_)' 
echo "USIM_HARD_TRAIN_DONE $(date +%m-%d\ %H:%M)"
