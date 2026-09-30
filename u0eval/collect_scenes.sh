#!/usr/bin/env bash
# Re-collect the WAM training set in the task scenes, then keep the simulator
# busy with model-independent U0 grasp blocks while the new core trains.
#
#   1. wait for the in-flight eval block; install the bridge
#   2. exploration server (8005) drives every locomotion scene + 2 armed scenes
#      through the harness; recordings kept for all episodes (KEEP_ALL_EPISODES)
#   3. convert recordings -> OU-schema npz (/hy-tmp/data/collect_scenes)
#   4. launch ARMS=u0 grasp-stage eval (1040 U0 episodes, model-independent)
#   5. train the scene-diverse core (best_scenes.pt) + its estimator ensemble,
#      then write the offline validation report
set -uo pipefail
# The IDE shell that launches these orchestrators carries HTTP_PROXY=127.0.0.1:17890
# with 0.0.0.0 absent from NO_PROXY; the bridge posts to http://0.0.0.0:<port>/act
# and got 502 Bad Gateway from the proxy. Policy traffic is local: never proxy it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
RUNS=/hy-tmp/u0env/dataset/eval_runs
SRC=/hy-tmp/u0env/ros_ws/src/bluerov2_control/scripts/ros_gr00t_bridge_ext.py
DST=/hy-tmp/u0env/ros_ws/devel/lib/bluerov2_control/ros_gr00t_bridge_ext.py
mkdir -p "$LOG"
export PYTHONPATH=$WS

echo "=== waiting for in-flight block $(date +%H:%M:%S)"
for i in $(seq 1 600); do
  if ! pgrep -f 'batch_run_ext|run_eval_task' >/dev/null; then break; fi
  sleep 15
done
sleep 5
for pat in 'parsed_simulator' 'roslaunch' 'rosmaster'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    kill -9 "$p" 2>/dev/null || true
  done
done
sleep 3
cp "$SRC" "$DST" && chmod +x "$DST" && echo "bridge installed"

echo "=== exploration server (8005) $(date +%H:%M:%S)"
for p in $(pgrep -f 'explore_server' || true); do kill "$p" 2>/dev/null || true; done
nohup /usr/local/bin/python3 "$WS/u0eval/explore_server.py" --port 8005 --seed 0 \
  >"$LOG/explore_server.log" 2>&1 &
sleep 4
curl -s -m 5 http://127.0.0.1:8005/health || { echo "explore server failed"; exit 1; }
echo

echo "=== collection $(date +%H:%M:%S)"
# task:episodes -- each episode runs to the task's eval timeout (no success)
for spec in goto_charge_station:5 goto_water_tower:5 scan_ship_ancient:3 scan_ship_modern:3 \
            inspect_pipeline_pool:2 inspect_pipeline_sea:3 follow_boat:3 \
            pick_pipe0_shallow:4 pick_red_factory:4; do
  task=${spec%%:*}; n=${spec##*:}
  echo "--- collect $task x$n $(date +%H:%M:%S)"
  KEEP_ALL_EPISODES=1 bash "$WS/u0eval/run_eval_task.sh" "$task" collect "$n" 8005 -1.0 zero \
    >"$LOG/collect_${task}.log" 2>&1 || echo "  (nonzero exit)"
  echo "    episodes on disk: $(ls -d "$RUNS"/collect_full/"$task"/episode* 2>/dev/null | wc -l)"
done
for p in $(pgrep -f 'explore_server' || true); do kill "$p" 2>/dev/null || true; done
echo "COLLECT_DONE $(date +%H:%M:%S)"

echo "=== convert recordings $(date +%H:%M:%S)"
cd "$WS"
/usr/local/bin/python3 scripts/recordings_to_ou.py --include collect_ --holdout "" \
  --out /hy-tmp/data/collect_scenes >"$LOG/convert_collect.log" 2>&1 || echo "convert failed"
tail -2 "$LOG/convert_collect.log"

echo "=== launch U0 grasp-stage blocks (model-independent) $(date +%H:%M:%S)"
ARMS=u0 nohup bash "$WS/u0eval/run_full.sh" grasp >>"$LOG/full_all.log" 2>&1 &
echo "U0_GRASP_LAUNCHED pid=$! $(date +%H:%M:%S)"

echo "=== train scene-diverse core $(date +%H:%M:%S)"
/usr/local/bin/python3 scripts/train_stage1.py --mix-ou \
  --extra-ou /hy-tmp/data/collect_scenes,/hy-tmp/data/deploy_rec --extra-reps 6 \
  --epochs 8 --batch-size 128 --workers 6 --ckpt-name best_scenes \
  >"$LOG/train_best_scenes.log" 2>&1 || echo "core training failed"
tail -3 "$LOG/train_best_scenes.log"
if [ -f /hy-tmp/models/uwam/best_scenes.pt ]; then
  echo "=== estimator ensemble on the new core $(date +%H:%M:%S)"
  /usr/local/bin/python3 scripts/train_vel_ensemble.py --ckpt-dyn /hy-tmp/models/uwam/best_scenes.pt \
    --extra /hy-tmp/data/planner_mix,/hy-tmp/data/collect_scenes,/hy-tmp/data/deploy_rec --extra-reps 1,2,3 \
    --canon-pressure 27916 --canon-alt 2.4 --k 3 --epochs 6 --batch-size 256 \
    --out /hy-tmp/models/uwam/vel_ens_scenes.pt >"$LOG/train_vel_ens_scenes.log" 2>&1 || echo "ensemble failed"
  tail -2 "$LOG/train_vel_ens_scenes.log"
fi
echo "TRAIN_DONE $(date +%H:%M:%S)"
echo "COLLECT_SCENES_ALL_DONE $(date +%H:%M:%S)"
