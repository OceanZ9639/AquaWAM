#!/usr/bin/env bash
# Top a fine-tuned baseline up to USIM's trial allocation (40 per goto / grasp / transport task; the
# scan / inspect / follow tasks already have their 20): for each task, run the missing episodes into
# <arm>_<cond>_topup/ and merge them into the protocol block <arm>_<cond>/ with renumbering
# (u0eval/topup_merge.py). Same server, harness, judges and drop times as the original blocks.
#   usage: ARM=gr00t2|pi05|xvla|smolvla [TARGET=40] baseline_topup_queue.sh <inst: local|A|B> <gpu> <full|drop> <task ...>
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam; LOG=/hy-tmp/logs/u0eval; export PYTHONPATH=$WS; cd $WS; mkdir -p $LOG
ARM=${ARM:?arm tag: gr00t2|pi05|xvla|smolvla}
INST=$1; GPU=$2; KIND=$3; shift 3
TARGET=${TARGET:-40}
if [ "$INST" = local ]; then export U0ENV=/hy-tmp/u0env; PORT=8006; else source u0eval/instance_env.sh "$INST" >/dev/null; PORT=$((PORT_WAM + 6)); fi
RUNS=$U0ENV/dataset/eval_runs
drop_for() { case "$1" in goto_*) echo 8 ;; scan_*|inspect_*) echo 40 ;; follow_*) echo 20 ;; pick_*|transfer_*) echo 30 ;; *) echo 20 ;; esac; }
healthy() { curl -s -m 5 "http://127.0.0.1:$1/health" 2>/dev/null | grep -q healthy; }
uniq_eps() { [ -f "$1" ] && tail -n +2 "$1" | cut -d, -f1 | sort -u | wc -l || echo 0; }
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
healthy "$PORT" || { echo "TOPUP_SERVER_FAILED $ARM $INST $PORT $(date +%m-%d\ %H:%M:%S)"; exit 1; }
echo "$ARM server up on $PORT (instance $INST, gpu $GPU) $(date +%m-%d\ %H:%M:%S)"
for t in "$@"; do
  if [ "$KIND" = full ]; then D=-1.0; COND=full; else D=$(drop_for "$t"); COND="drop${D}s_zero"; fi
  main="$RUNS/${ARM}_${COND}/$t/results.csv"
  for attempt in 1 2 3; do
    have=$(valid_eps "$main")
    need=$((TARGET - have))
    [ "$need" -le 0 ] && break
    echo "--- [$INST] ${ARM}_${COND}/$t: $have valid, running $need more (attempt $attempt) $(date +%m-%d\ %H:%M:%S)"
    [ "$attempt" != 1 ] && { bash u0eval/reset_instance_x.sh "$([ "$INST" = local ] && echo A || echo "$INST")" 2>&1 | tail -n 1; sleep 5; }
    rm -rf "$U0ENV/dataset/eval/$t" "$RUNS/${ARM}_${COND}_topup/$t"
    COND_TAG="${COND}_topup" bash u0eval/run_eval_task.sh "$t" "$ARM" "$need" "$PORT" "$D" zero >"$LOG/${ARM}_${COND}_topup_$t.log" 2>&1 || true
    # drop the invalid trials of the top-up block before merging (empty judge log = simulator death)
    tu="$RUNS/${ARM}_${COND}_topup/$t"
    if [ -f "$tu/results.csv" ]; then
      python3 - "$tu" <<'EOF'
import csv, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = [r for r in csv.reader(open(d / "results.csv")) if r and r[0] != "episode"]
keep = [r for r in rows if (d / "logs" / f"episode_{r[0]}_data.csv").exists()
        and sum(1 for _ in open(d / "logs" / f"episode_{r[0]}_data.csv")) >= 4]
with open(d / "results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["episode", "result"]); [w.writerow(r) for r in keep]
print(f"    top-up block: {len(keep)}/{len(rows)} valid trials kept")
EOF
      python3 u0eval/topup_merge.py "$RUNS" "${ARM}_${COND}" "$t"
    fi
  done
  echo "    [$INST] ${ARM}_${COND}/$t: $(tail -n +2 "$main" 2>/dev/null | sort -u -t, -k1,1 | cut -d, -f2 | sort | uniq -c | tr '\n' ' ') [$(uniq_eps "$main") unique, $(valid_eps "$main") valid / $TARGET]  $(date +%m-%d\ %H:%M:%S)"
done
echo "TOPUP_QUEUE_DONE $ARM $KIND $INST $(date +%m-%d\ %H:%M:%S)"
