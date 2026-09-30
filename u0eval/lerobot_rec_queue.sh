#!/usr/bin/env bash
# LeRobot-family baselines (pi05 / xvla / smolvla / fastwam) evaluated with RECURRENT inference: the
# bridge executes EXEC_STEPS steps (0.1 s each) of every predicted chunk and re-queries the policy,
# instead of the 16-step (1.6 s) open-loop chunk that GR00T / U0 were trained for. USIM's protocol for
# pi0.5 is exactly this (10 Hz recurrent inference, first step of the chunk executed).
# Archives under <arm>_<full|drop<T>s_zero>_rec<EXEC_STEPS>/<task>/, USIM's trial allocation (40 per
# goto / grasp / transport task, 20 per scan / inspect / follow) unless NEP is given.
#   usage: ARM=xvla [NEP=20] lerobot_rec_queue.sh <inst: local|A|B> <gpu> <full|drop> <exec_steps> <task ...>
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam; LOG=/hy-tmp/logs/u0eval; export PYTHONPATH=$WS; cd $WS; mkdir -p $LOG
ARM=${ARM:?arm tag: pi05|xvla|smolvla|fastwam}
INST=$1; GPU=$2; KIND=$3; ES=$4; shift 4
if [ "$INST" = local ]; then export U0ENV=/hy-tmp/u0env; PORT=8006; else source u0eval/instance_env.sh "$INST" >/dev/null; PORT=$((PORT_WAM + 6)); fi
RUNS=$U0ENV/dataset/eval_runs
drop_for() { case "$1" in goto_*) echo 8 ;; scan_*|inspect_*) echo 40 ;; follow_*) echo 20 ;; pick_*|transfer_*) echo 30 ;; *) echo 20 ;; esac; }
nep_for() { case "$1" in goto_*|pick_*|transfer_*) echo 40 ;; *) echo 20 ;; esac; }
healthy() { curl -s -m 5 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q healthy; }
valid_eps() {  # <results.csv>: unique episodes whose judge log has >= 3 data rows
  [ -f "$1" ] || { echo 0; return; }
  local d; d=$(dirname "$1"); local n=0
  for e in $(tail -n +2 "$1" | cut -d, -f1 | sort -u); do
    [ -f "$d/logs/episode_${e}_data.csv" ] && [ "$(wc -l < "$d/logs/episode_${e}_data.csv")" -ge 4 ] && n=$((n + 1))
  done
  echo $n
}
for p in $(pgrep -f "inference_service_u0.py .*--port $PORT" || true) $(pgrep -f "lerobot_policy_server.py --port $PORT" || true); do kill "$p" 2>/dev/null; done
sleep 2
CUDA_VISIBLE_DEVICES=$GPU nohup bash u0eval/start_server.sh "$ARM" "$PORT" >"$LOG/${ARM}_server_$PORT.out" 2>&1 &
for i in $(seq 1 90); do healthy "$PORT" && break; sleep 10; done
healthy "$PORT" || { echo "REC_SERVER_FAILED $ARM $INST $PORT $(date +%m-%d\ %H:%M:%S)"; exit 1; }
echo "$ARM server up on $PORT (instance $INST, gpu $GPU, exec_steps $ES) $(date +%m-%d\ %H:%M:%S)"
for t in "$@"; do
  # RUN_TAG (e.g. v2_) distinguishes checkpoint rounds in the archive name: <arm>_<RUN_TAG><cond>_rec<ES>
  if [ "$KIND" = full ]; then D=-1.0; COND="${RUN_TAG:-}full_rec${ES}"; else D=$(drop_for "$t"); COND="${RUN_TAG:-}drop${D}s_zero_rec${ES}"; fi
  n=${NEP:-$(nep_for "$t")}
  f="$RUNS/${ARM}_${COND}/$t/results.csv"
  for attempt in 1 2 3; do
    ok=$(valid_eps "$f")
    [ "$ok" -ge "$n" ] && break
    [ "$attempt" != 1 ] && { echo "    [$INST] ${ARM}_${COND}/$t invalid/incomplete ($ok/$n) -> reset X, retry"; bash u0eval/reset_instance_x.sh "$([ "$INST" = local ] && echo A || echo "$INST")" 2>&1 | tail -n 1; sleep 5; }
    echo "--- [$INST] ${ARM}_${COND}/$t x$n (attempt $attempt) $(date +%m-%d\ %H:%M:%S)"
    rm -rf "$U0ENV/dataset/eval/$t"
    EXEC_STEPS=$ES COND_TAG=$COND bash u0eval/run_eval_task.sh "$t" "$ARM" "$n" "$PORT" "$D" zero >"$LOG/${ARM}_${COND}_$t.log" 2>&1 || true
  done
  echo "    [$INST] ${ARM}_${COND}/$t: $(tail -n +2 "$f" 2>/dev/null | sort -u -t, -k1,1 | cut -d, -f2 | sort | uniq -c | tr '\n' ' ') [$(valid_eps "$f") valid / $n]  $(date +%m-%d\ %H:%M:%S)"
done
echo "REC_QUEUE_DONE $ARM $KIND rec$ES $INST $(date +%m-%d\ %H:%M:%S)"
