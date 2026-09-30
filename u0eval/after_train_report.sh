#!/usr/bin/env bash
# When collect_scenes.sh finishes training, write the offline old-vs-new report.
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
LOG=/hy-tmp/logs/u0eval
while ! grep -q 'TRAIN_DONE' "$LOG/collect_scenes.log" 2>/dev/null; do
  if ! pgrep -f '[c]ollect_scenes.sh' >/dev/null; then echo "collect_scenes died"; tail -5 "$LOG/collect_scenes.log"; exit 1; fi
  sleep 120
done
cd /hy-tmp/underwater_wam
echo "=== offline core comparison $(date +%H:%M:%S)"
/usr/local/bin/python3 u0eval/diag_core_compare.py 2>&1 | grep -v Warning
echo "=== new-core estimator by speed (holdout) ==="
/usr/local/bin/python3 - << 'PY' 2>&1 | grep -v Warning
import sys, glob, numpy as np, torch
sys.path.insert(0,'/hy-tmp/underwater_wam/u0eval'); sys.path.insert(0,'/hy-tmp/underwater_wam')
from pathlib import Path
from uwam.control import SamplingMPC
from wam_policy_server import _load_model
from diag_estimator_holdout import windows
for name, ck, ens in (("old","/hy-tmp/models/uwam/best_ou.pt","/hy-tmp/models/uwam/vel_ens_deploy.pt"),("new","/hy-tmp/models/uwam/best_scenes.pt","/hy-tmp/models/uwam/vel_ens_scenes.pt")):
    if not Path(ck).exists(): continue
    model, dn, pn, cfg = _load_model(Path(ck), "cuda"); m = SamplingMPC(model, dn, pn, cfg.control, device="cuda"); m.load_vel_ensemble(ens)
    for arm in ("u0_","wam_"):
        E=[];S=[]
        for f in sorted(glob.glob('/hy-tmp/data/deploy_rec_holdout/*.npz')):
            if not Path(f).name.startswith(arm): continue
            for fr, ha, v in windows(f):
                E.append(np.linalg.norm(m.estimate_velocity_ens(fr, ha)[0][:2]-v[:2])); S.append(np.linalg.norm(v[:2]))
        E=np.array(E); S=np.array(S)
        bins=[(0,0.15),(0.15,0.3),(0.3,0.45),(0.45,0.6),(0.6,1.5)]
        row=" ".join(f"{E[(S>=a)&(S<b)].mean():.3f}" if ((S>=a)&(S<b)).sum()>10 else "  -  " for a,b in bins)
        print(f"{name:<4} {arm[:-1]:<4} by-speed [{row}]  all {E.mean():.3f}")
PY
echo "REPORT_DONE $(date +%H:%M:%S)"
