#!/usr/bin/env bash
# P4 chain, started after P3: waits for P3_ALL_DONE, then
#   1. USIM-ablation training (GPU)
#   2. planner re-collect with images (sim)     -> /hy-tmp/data/planner_mix_img
#   3. 0.2 s grid planner collection (sim)      -> /hy-tmp/data/planner_dt2
#   4. multimodal vel head training (GPU)
#   5. dt-conditioned model training (GPU)
#   6. closed-loop: usim-ablation gate, 5 Hz ZOH gate, dt-model 5 Hz gate, wam_mm blind
set -uo pipefail
LOGS=/hy-tmp/logs/uwam
WS=/hy-tmp/underwater_wam
PY=/usr/local/bin/python3
export PYTHONPATH=/hy-tmp/underwater_wam:${PYTHONPATH:-}

echo "waiting for P3 ..."
for i in $(seq 1 2000); do
  if grep -q "P3_ALL_DONE" "${LOGS}/run_p3.log" 2>/dev/null; then break; fi
  sleep 60
done
grep -q "P3_ALL_DONE" "${LOGS}/run_p3.log" || { echo "P3 never finished"; exit 1; }
echo "P3 done at $(date +%H:%M:%S)"

kill_sim() {
  for p in $(ps -eo pid,cmd | awk '/\/parsed_simulator |roslaunch stonefish|rosmaster --core/ && !/awk/{print $1}'); do
    kill "$p" 2>/dev/null || true
  done
  sleep 3
}
kill_sim

echo "=== p4 aggregate P3 $(date +%H:%M:%S) ==="
${PY} "${WS}/scripts/aggregate_multiseed.py" > "${LOGS}/p4_agg_multiseed.log" 2>&1
${PY} "${WS}/scripts/aggregate_sweeps.py" > "${LOGS}/p4_agg_sweeps.log" 2>&1

echo "=== p4 usim-ablation train $(date +%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=0 ${PY} "${WS}/scripts/train_stage1.py" --no-usim --mix-ou \
  --extra-ou /hy-tmp/data/planner_mix --extra-reps 48 --epochs 6 --batch-size 128 \
  --workers 8 --lr 3e-4 --ckpt-name ou_only > "${LOGS}/stage1_ou_only.log" 2>&1 || exit 1

echo "=== p4 collect planner images $(date +%H:%M:%S) ==="
bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/collect_planner.py" \
  --out /hy-tmp/data/planner_mix_img --frames 900 --images \
  > "${LOGS}/collect_planner_img.log" 2>&1 || exit 1
kill_sim

echo "=== p4 collect 0.2s grid $(date +%H:%M:%S) ==="
bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/collect_planner.py" \
  --out /hy-tmp/data/planner_dt2 --frames 900 --act-every 2 \
  > "${LOGS}/collect_planner_dt2.log" 2>&1 || exit 1
kill_sim

echo "=== p4 train vel_mm $(date +%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=0 ${PY} "${WS}/scripts/train_vel_mm.py" \
  --ckpt-dyn /hy-tmp/models/uwam/best_ou.pt --extra /hy-tmp/data/planner_mix_img \
  --epochs 4 > "${LOGS}/train_vel_mm.log" 2>&1 || exit 1

echo "=== p4 train dt model $(date +%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=0 ${PY} "${WS}/scripts/train_stage1.py" --use-dt --mix-ou \
  --extra-ou /hy-tmp/data/planner_mix,/hy-tmp/data/planner_dt2 --extra-reps 48 \
  --resume /hy-tmp/models/uwam/best_ou.pt --epochs 4 --batch-size 128 --workers 8 \
  --lr 1e-4 --ckpt-name dt_model > "${LOGS}/stage1_dt_model.log" 2>&1 || exit 1

run_cl() {
  local name="$1"; shift
  echo "=== p4 closed-loop ${name} $(date +%H:%M:%S) ==="
  bash "${WS}/scripts/with_sim.sh" python "${WS}/scripts/closed_loop.py" "$@"
  local rc=$?
  kill_sim
  return $rc
}

run_cl ablation_gate --ckpt /hy-tmp/models/uwam/ou_only.pt --seed 0 --seconds 12 \
  --modes wam --out "${LOGS}/closed_loop_usim_ablation.json" || exit 1
run_cl hz5 --ckpt /hy-tmp/models/uwam/best_ou.pt --seed 0 --seconds 12 \
  --modes mixer,wam --control-every 2 --out "${LOGS}/closed_loop_hz5.json" || exit 1
run_cl dt5 --ckpt /hy-tmp/models/uwam/dt_model.pt --seed 0 --seconds 12 \
  --modes wam --control-every 2 --out "${LOGS}/closed_loop_dt5.json" || exit 1
run_cl wam_mm --ckpt /hy-tmp/models/uwam/best_ou.pt --seed 0 --seconds 12 \
  --modes wam,wam_mm --vel-mm /hy-tmp/models/uwam/vel_mm.pt \
  --drop-dvl-at 6.0 --regime-set fault_mid \
  --out "${LOGS}/closed_loop_wam_mm.json" || exit 1

echo "P4_ALL_DONE"
