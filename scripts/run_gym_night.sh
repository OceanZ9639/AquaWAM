#!/usr/bin/env bash
# Overnight gym batch: rename the sigma_p=0.02 artifacts to *_lo, build the 0.06 (main)
# and 0.12 (hi) pointmass variants, then run the full suite on all pointmass variants
# and reacher. Runs alongside the underwater chain (GPU is shared, both are small).
set -uo pipefail
WS=/hy-tmp/underwater_wam
cd "$WS"

# wait for the in-flight pointmass_lo headline run to finish
while ps -eo cmd | grep -q "[c]losed_loop.py --env pointmass"; do sleep 20; done

if [ -d /hy-tmp/data/gym_pointmass ] && [ ! -d /hy-tmp/data/gym_pointmass_lo ]; then
  mv /hy-tmp/data/gym_pointmass /hy-tmp/data/gym_pointmass_lo
  mv /hy-tmp/models/gym_pointmass /hy-tmp/models/gym_pointmass_lo
  mv /hy-tmp/results/gym_pointmass_closed_loop.json /hy-tmp/results/gym_pointmass_lo_closed_loop.json 2>/dev/null || true
  mv /hy-tmp/logs/gym_pointmass_calib /hy-tmp/logs/gym_pointmass_lo_calib 2>/dev/null || true
fi

for env in pointmass pointmass_hi; do
  echo "=== [$env] collect+train $(date +%H:%M:%S)"
  python3 gym_port/collect.py --env "$env" --episodes 300 --ep-len 400 || exit 1
  CUDA_VISIBLE_DEVICES=0 python3 gym_port/train.py --env "$env" --epochs 6 || exit 1
done

bash scripts/run_gym_suite.sh pointmass    "1.0,0.0" "-1.0,0.0" 0.05 || exit 1
bash scripts/run_gym_suite.sh pointmass_lo "1.0,0.0" "-1.0,0.0" 0.05 || exit 1
bash scripts/run_gym_suite.sh pointmass_hi "1.0,0.0" "-1.0,0.0" 0.05 || exit 1
bash scripts/run_gym_suite.sh reacher      "8,0"     "-8,0"     0.02 || exit 1
echo "GYM_NIGHT_DONE $(date +%H:%M:%S)"
