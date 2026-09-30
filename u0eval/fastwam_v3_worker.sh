#!/usr/bin/env bash
# One FastWAM v3 worker. A server is ~40 GB, so one worker per GPU.
# Args: <inst A|B|C|D> <gpu> <full:task|drop:task> ...
# Full-sensing tasks run first, then dropout, so the 40 GB server loads twice at most.
# Archives: fastwam_v3_full_rec8/ and fastwam_v3_drop<T>s_zero_rec8/
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
mkdir -p "$LOG"
cd "$WS"
INST=${1:?inst}; GPU=${2:?gpu}; shift 2
[ "$#" -gt 0 ] || { echo "no tasks"; exit 2; }
export FASTWAM_CKPT=/hy-tmp/baselines/ft/fastwam_usim_v3/checkpoints/last/pretrained_model
export RUN_TAG=v3_ ARM=fastwam
[ -f "$FASTWAM_CKPT/model.safetensors" ] || { echo "MISSING $FASTWAM_CKPT"; exit 2; }
ts() { date +%m-%d\ %H:%M:%S; }
full=() drop=()
for spec in "$@"; do
  case "$spec" in
    full:*) full+=("${spec#full:}") ;;
    drop:*) drop+=("${spec#drop:}") ;;
    *) echo "bad spec $spec"; exit 2 ;;
  esac
done
echo "=== worker inst=$INST gpu=$GPU full=${#full[@]} drop=${#drop[@]} $(ts)"
bash u0eval/reset_instance_x.sh "$INST" || true
run_kind() {  # <full|drop> tasks...
  local kind=$1; shift
  [ "$#" -gt 0 ] || return 0
  echo "=== $INST $kind ($#) $(ts)"
  ARM=fastwam bash u0eval/lerobot_rec_queue.sh "$INST" "$GPU" "$kind" 8 "$@"
}
run_kind full "${full[@]}"
run_kind drop "${drop[@]}"
echo "FASTWAM_V3_WORKER_DONE $INST $(ts)"
