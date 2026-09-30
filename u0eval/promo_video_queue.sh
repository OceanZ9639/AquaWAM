#!/usr/bin/env bash
# Promo-page recordings: the Table 1 models (same weights and protocol as their Table 1 blocks) re-run for a
# few full-sensing episodes per task with EVERY episode's cameras kept, archived under <arm>_promo/<task>/ so
# the Table 1 blocks are untouched. The left camera is dropped (the page uses the ego and wrist views).
#   usage: ARM=<start_server.sh arm> [ES=<exec steps>] [NEP=2] promo_video_queue.sh <inst A|B|C|D> <gpu> <task ...>
# Weights / protocol per model are passed as env: GR00T_CKPT, XVLA_CKPT, SMOLVLA_CKPT, OPENVLA_CKPT,
# OPENVLA_UNNORM_KEY, PI05_OPENPI_CKPT, XLA_PYTHON_CLIENT_MEM_FRACTION; ES unset = the bridge default (16).
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam; LOG=/hy-tmp/logs/u0eval; export PYTHONPATH=$WS; cd $WS; mkdir -p $LOG
ARM=${ARM:?arm}
INST=$1; GPU=$2; shift 2
NEP=${NEP:-2}
source u0eval/instance_env.sh "$INST" >/dev/null
exec 9>"/tmp/promo_queue_$INST.lock"
flock -n 9 || { echo "PROMO_QUEUE_ALREADY_RUNNING on $INST"; exit 0; }
PORT=$((PORT_WAM + 6))
RUNS=$U0ENV/dataset/eval_runs
ts() { date '+%m-%d %H:%M:%S'; }
healthy() { curl -s -m 5 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q healthy; }
valid_eps() {  # unique episodes whose judge log has >= 3 data rows
  [ -f "$1" ] || { echo 0; return; }
  local d; d=$(dirname "$1"); local n=0
  for e in $(tail -n +2 "$1" | cut -d, -f1 | sort -u); do
    [ -f "$d/logs/episode_${e}_data.csv" ] && [ "$(wc -l < "$d/logs/episode_${e}_data.csv")" -ge 4 ] && n=$((n + 1))
  done
  echo $n
}
for p in $(pgrep -f "inference_service_u0.py .*--port $PORT" || true) $(pgrep -f "lerobot_policy_server.py --port $PORT" || true) \
         $(pgrep -f "openvla_policy_server.py --port $PORT" || true) $(pgrep -f "inference_service_openpi.py .*--port $PORT" || true); do
  kill "$p" 2>/dev/null
done
sleep 2
CUDA_VISIBLE_DEVICES=$GPU nohup bash u0eval/start_server.sh "$ARM" "$PORT" >"$LOG/${ARM}_server_promo_$PORT.out" 2>&1 &
for i in $(seq 1 120); do healthy "$PORT" && break; sleep 10; done
healthy "$PORT" || { echo "PROMO_SERVER_FAILED $ARM $INST $PORT $(ts)"; exit 1; }
echo "$ARM server up on $PORT (instance $INST, gpu $GPU, exec_steps ${ES:-default}) $(ts)"
for t in "$@"; do
  f="$RUNS/${ARM}_promo/$t/results.csv"
  for attempt in 1 2 3; do
    [ "$(valid_eps "$f")" -ge "$NEP" ] && break
    [ "$attempt" != 1 ] && { echo "    [$INST] $t invalid/incomplete ($(valid_eps "$f")/$NEP) -> reset X, retry"; bash u0eval/reset_instance_x.sh "$INST" 2>&1 | tail -n 1; sleep 5; }
    echo "--- [$INST] ${ARM}_promo/$t x$NEP (attempt $attempt) $(ts)"
    rm -rf "$U0ENV/dataset/eval/$t"
    ( [ -n "${ES:-}" ] && export EXEC_STEPS=$ES
      KEEP_ALL_EPISODES=1 COND_TAG=promo bash u0eval/run_eval_task.sh "$t" "$ARM" "$NEP" "$PORT" -1.0 zero ) >"$LOG/${ARM}_promo_$t.log" 2>&1 || true
    rm -rf "$U0ENV/dataset/eval/$t" "$RUNS/${ARM}_promo/$t"/episode*/images/left
  done
  echo "    [$INST] ${ARM}_promo/$t: $(tail -n +2 "$f" 2>/dev/null | sort -u -t, -k1,1 | cut -d, -f2,3 | tr '\n' ' ') [$(valid_eps "$f") valid / $NEP]  $(ts)"
done
for p in $(pgrep -f "port $PORT" || true); do kill "$p" 2>/dev/null; done
echo "PROMO_QUEUE_DONE $ARM $INST $(ts)"
