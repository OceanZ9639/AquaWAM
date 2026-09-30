#!/usr/bin/env bash
# Full cross-domain protocol for one gym env:
#   deployment-condition gate calibration -> headline (4 regimes x 5 arms x 5 seeds)
#   -> severity sweep -> duration sweep.
# Usage: run_gym_suite.sh <env> <goal> <goal2> [dt]
set -uo pipefail
ENV="$1"; GOAL="$2"; GOAL2="$3"; DT="${4:-0.05}"
WS=/hy-tmp/underwater_wam
RES=/hy-tmp/results
CAL=/hy-tmp/logs/gym_${ENV}_calib
PY=python3

cd "$WS"
rm -rf "$CAL"   # stale dumps from a previous hold-semantics would poison the fit

echo "=== [$ENV] gate calibration runs $(date +%H:%M:%S)"
$PY gym_port/closed_loop.py --env "$ENV" --arms calib --regimes nominal,fault --goal "$GOAL" \
  --drop-at 100 --T 300 --eta 0.5 --seeds 4 --seed-base 100 --calib-dump "$CAL" \
  --out "$RES/tmp_${ENV}_calib1.json" || exit 1
$PY gym_port/closed_loop.py --env "$ENV" --arms calib --regimes midfault --goal "$GOAL" \
  --drop-at 100 --change-at 160 --T 300 --eta 0.45 --seeds 5 --seed-base 100 --calib-dump "$CAL" \
  --out "$RES/tmp_${ENV}_calib2.json" || exit 1
$PY gym_port/closed_loop.py --env "$ENV" --arms calib --regimes midfault --goal "$GOAL" \
  --drop-at 100 --change-at 190 --T 300 --eta 0.3 --seeds 5 --seed-base 105 --calib-dump "$CAL" \
  --out "$RES/tmp_${ENV}_calib3.json" || exit 1
# stationary replan reference on the calibration seeds: together with the hold cost in
# calib1 it decides the gate's alpha_base (which endpoint is stationary-optimal)
$PY gym_port/closed_loop.py --env "$ENV" --arms replan --regimes nominal --goal "$GOAL" \
  --drop-at 100 --T 300 --seeds 4 --seed-base 100 \
  --out "$RES/tmp_${ENV}_calibreplan.json" || exit 1
PYTHONPATH="$WS" $PY scripts/gate_calib_from_dumps.py --dumps "$CAL" --dt "$DT" \
  --hold-ref "$RES/tmp_${ENV}_calib1.json" --replan-ref "$RES/tmp_${ENV}_calibreplan.json" \
  --out "/hy-tmp/models/gym_${ENV}/gate_calib.json" || exit 1

echo "=== [$ENV] headline $(date +%H:%M:%S)"
$PY gym_port/closed_loop.py --env "$ENV" --goal "$GOAL" --goal2="$GOAL2" \
  --drop-at 100 --change-at 175 --T 300 --eta 0.55 --seeds 5 \
  --out "$RES/gym_${ENV}_closed_loop.json" || exit 1

echo "=== [$ENV] severity sweep $(date +%H:%M:%S)"
for eta in 0.8 0.65 0.5 0.35 0.2; do
  $PY gym_port/closed_loop.py --env "$ENV" --regimes midfault --arms oracle,open,replan,gated \
    --goal "$GOAL" --goal2="$GOAL2" --drop-at 100 --change-at 175 --T 300 --eta "$eta" --seeds 3 \
    --out "$RES/gym_${ENV}_sev_e${eta}.json" || exit 1
done

echo "=== [$ENV] duration sweep $(date +%H:%M:%S)"
for drop in 50 150 220; do
  $PY gym_port/closed_loop.py --env "$ENV" --regimes midfault,goalstep --arms open,replan,gated \
    --goal "$GOAL" --goal2="$GOAL2" --drop-at "$drop" --change-at 175 --T 300 --eta 0.55 --seeds 3 \
    --out "$RES/gym_${ENV}_dur_d${drop}.json" || exit 1
done
echo "GYM_SUITE_DONE $ENV $(date +%H:%M:%S)"
