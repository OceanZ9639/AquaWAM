#!/usr/bin/env bash
# Phase-1 closeout: run after BOTH chains (meanhold_resume + gym_resume) finish.
#  1. splice the wam-only sweep reruns into the full-arm files (baselines were
#     measured unaffected by the mean-hold fix: +-1% vs wam -13%)
#  2. regenerate every aggregate the paper reads
set -uo pipefail
WS=/hy-tmp/underwater_wam
LOGS=/hy-tmp/logs/uwam
cd "$WS"

echo "=== merge wam-only sweeps $(date +%H:%M:%S)"
for d in 2 4 6 8 10; do
  python3 scripts/merge_wam_rows.py --base "$LOGS/closed_loop_bsweep_d$d.json" \
                                    --wam  "$LOGS/wamonly_bsweep_d$d.json"
done
python3 scripts/merge_wam_rows.py --base "$LOGS/closed_loop_fault_sweep.json" \
                                  --wam  "$LOGS/wamonly_fault_sweep.json"

echo "=== aggregates $(date +%H:%M:%S)"
python3 scripts/aggregate_multiseed.py --logs "$LOGS" --out "$LOGS/multiseed_summary.json"
python3 scripts/aggregate_sweeps.py    --logs "$LOGS" --out "$LOGS/sweeps_summary.json"
python3 scripts/aggregate_gym.py       --out "$LOGS/gym_summary.json"
python3 scripts/write_paper_table.py

echo "=== acceptance checks $(date +%H:%M:%S)"
python3 - <<'PY'
import json
ms = json.load(open('/hy-tmp/logs/uwam/multiseed_summary.json'))
for proto, blk in ms.items():
    if not isinstance(blk, dict) or 'blind_track' not in blk:
        continue
    wam = blk['blind_track']['wam']
    print(f"{proto}: wam per_seed={wam['per_seed']} mean={wam['mean']}±{wam['std']}")
try:
    gym = json.load(open('/hy-tmp/logs/uwam/gym_summary.json'))
    print('gym_summary keys:', list(gym)[:8])
except Exception as e:
    print('gym summary:', e)
import json as j
r = j.load(open('/hy-tmp/results/gym_reacher_closed_loop.json'))['results']
print('reacher nominal/gated =', r['nominal/gated']['err_blind'], '(expect ~1.09, was 2.0591)')
lo = j.load(open('/hy-tmp/results/gym_pointmass_lo_closed_loop.json'))['results']
print('pointmass_lo nominal/gated =', lo['nominal/gated']['err_blind'], '(expect ~0.0293, was 0.0421)')
PY
echo "PHASE1_FINISH_DONE"
