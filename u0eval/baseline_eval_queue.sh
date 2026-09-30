#!/usr/bin/env bash
# Online evaluation of a baseline VLA we fine-tuned on USIM, through the official u0env harness.
#   usage: ARM=gr00t|pi05|xvla|smolvla [DROP_T=30] [NEP=20] baseline_eval_queue.sh <instance A|B|local> <gpu> <task> ...
# The dropout time is per task family (the protocol of the main tables); DROP_T=-1 = full sensing.
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam; LOG=/hy-tmp/logs/u0eval; export PYTHONPATH=$WS; cd $WS; mkdir -p $LOG
ARM=${ARM:?arm tag: gr00t|gr00t2|gr00t_zs|pi05|pi05_zs|xvla|xvla_zs|smolvla|smolvla_zs|fastwam|u0rep}
INST=$1; GPU=$2; shift 2
NEP=${NEP:-20}
if [ "$INST" = local ]; then export U0ENV=/hy-tmp/u0env; PORT=8006; else source u0eval/instance_env.sh "$INST" >/dev/null; PORT=$((PORT_WAM + 6)); fi
drop_for() { case "$1" in goto_*) echo 8 ;; scan_*|inspect_*) echo 40 ;; follow_*) echo 20 ;; pick_*|transfer_*) echo 30 ;; *) echo 20 ;; esac; }
healthy() { curl -s -m 5 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q healthy; }
for p in $(pgrep -f "inference_service_u0.py .*--port $PORT" || true) $(pgrep -f "lerobot_policy_server.py --port $PORT" || true); do kill "$p" 2>/dev/null; done
sleep 2
CUDA_VISIBLE_DEVICES=$GPU nohup bash u0eval/start_server.sh "$ARM" "$PORT" >"$LOG/${ARM}_server_$PORT.out" 2>&1 &
for i in $(seq 1 90); do healthy "$PORT" && break; sleep 10; done
healthy "$PORT" || { echo "$ARM server on $PORT did not come up"; exit 1; }
echo "$ARM server up on $PORT (instance $INST, gpu $GPU) $(date +%H:%M:%S)"
for t in "$@"; do
  # run_eval_task derives the condition dir itself: <arm>_full or <arm>_drop<T>s_zero
  if [ "${DROP_T:-auto}" = "-1" ]; then D=-1.0; TAG="${ARM}_full"; else D=$(drop_for "$t"); TAG="${ARM}_drop${D}s_zero"; fi
  f="$U0ENV/dataset/eval_runs/$TAG/$t/results.csv"
  # completeness is counted in UNIQUE, VALID episode ids: the harness occasionally writes a row twice
  # for the same episode; a block can be cut short (batch_run_ext SIGKILLed mid-block); and when the
  # instance's X server has gone bad the simulator dies at startup or never renders, the policy is
  # never queried and the episode is recorded as a plain "timeout" with an EMPTY judge log (0-1 rows;
  # a genuine timeout has one row per second). Such episodes are not evidence about the policy. A block
  # with any invalid episode gets the instance's X server reset and is re-run (up to 3 attempts).
  uniq_eps() { [ -f "$1" ] && tail -n +2 "$1" | cut -d, -f1 | sort -u | wc -l || echo 0; }
  valid_eps() {  # <results.csv>: unique episodes whose judge log has >= 3 data rows
    [ -f "$1" ] || { echo 0; return; }
    local d; d=$(dirname "$1"); local n=0
    for e in $(tail -n +2 "$1" | cut -d, -f1 | sort -u); do
      [ -f "$d/logs/episode_${e}_data.csv" ] && [ "$(wc -l < "$d/logs/episode_${e}_data.csv")" -ge 4 ] && n=$((n + 1))
    done
    echo $n
  }
  for attempt in 1 2 3; do
    have=$(uniq_eps "$f"); ok=$(valid_eps "$f")
    if [ "$have" -ge "$NEP" ] && [ "$ok" -ge "$NEP" ]; then break; fi
    if [ "$attempt" != 1 ] || [ "$have" -gt 0 ]; then
      echo "    [$INST] $TAG/$t incomplete or invalid ($have/$NEP unique, $ok valid) -> reset X, retry"
      bash u0eval/reset_instance_x.sh "$([ "$INST" = local ] && echo A || echo "$INST")" 2>&1 | tail -n 1
      sleep 5
    fi
    echo "--- [$INST] $TAG/$t x$NEP (attempt $attempt) $(date +%H:%M:%S)"
    rm -rf "$U0ENV/dataset/eval/$t"
    bash u0eval/run_eval_task.sh "$t" "$ARM" "$NEP" "$PORT" "$D" zero >"$LOG/${TAG}_$t.log" 2>&1 || true
  done
  echo "    [$INST] $TAG/$t: $(tail -n +2 "$f" 2>/dev/null | sort -u -t, -k1,1 | cut -d, -f2 | sort | uniq -c | tr '\n' ' ') [$(uniq_eps "$f")/$NEP unique, $(valid_eps "$f") valid]  $(date +%H:%M:%S)"
done
echo "BASELINE_QUEUE_DONE $ARM $INST $(date +%H:%M:%S)"
