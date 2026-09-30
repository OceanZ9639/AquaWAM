#!/usr/bin/env python3
"""Old core (best_ou) vs new scene-diverse core (best_scenes): offline, on
held-out task-scene recordings (deploy_rec_holdout: inspect_pipeline_sea,
scan_ship_modern -- never in either training set).

  dyn K-step : DVL prediction MAE of the dynamics rollout (planning quality),
               K = 5 steps (0.5 s) from the true state with the executed pwm
  est        : dead-reckoning ensemble error (blind-path quality)
Both per driving arm, in m/s.
"""
import glob
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/hy-tmp/underwater_wam/u0eval")
sys.path.insert(0, "/hy-tmp/underwater_wam")
from uwam.control import SamplingMPC  # noqa: E402
from wam_policy_server import _load_model  # noqa: E402

L, K = 16, 5


def load(npz):
    z = np.load(npz)
    pwm, dvl, av, la = z["pwm"], z["dvl"], z["imu_av"], z["imu_la"]
    pr, alt = z["pressure"].reshape(-1, 1), z["dvl_h"].reshape(-1, 1)
    prev = np.vstack([np.zeros((1, 8), np.float32), pwm[:-1]])
    dyn = np.concatenate([dvl, av, la, pr, alt, prev], axis=1).astype(np.float32)
    return dyn, pwm.astype(np.float32)


@torch.no_grad()
def kstep_err(mpc, dyn, pwm, t):
    """||dvl_hat - dvl_true|| averaged over the K predicted steps."""
    hs = torch.from_numpy(mpc.dyn_norm(dyn[t - L:t])).unsqueeze(0).to(mpc.device)
    ha = torch.from_numpy(mpc.pwm_norm(pwm[t - L:t])).unsqueeze(0).to(mpc.device)
    st = torch.from_numpy(mpc.dyn_norm(dyn[t])).unsqueeze(0).to(mpc.device)
    af = torch.from_numpy(mpc.pwm_norm(pwm[t:t + K])).unsqueeze(0).to(mpc.device)
    d = mpc.model.disturbance(hs, ha)
    s_hat, _, _ = mpc.model.rollout(st, af, d)
    pred = mpc.dyn_norm.invert(s_hat[0].cpu().numpy())[:, 0:3]
    return float(np.linalg.norm(pred[:, :2] - dyn[t + 1:t + K + 1, 0:2], axis=1).mean())


def main():
    cores = {
        "old best_ou": ("/hy-tmp/models/uwam/best_ou.pt", "/hy-tmp/models/uwam/vel_ens_deploy.pt"),
        "new best_scenes": ("/hy-tmp/models/uwam/best_scenes.pt", "/hy-tmp/models/uwam/vel_ens_scenes.pt"),
    }
    mpcs = {}
    for name, (ck, ens) in cores.items():
        if not Path(ck).exists():
            print(f"{name}: checkpoint missing, skipped")
            continue
        model, dn, pn, cfg = _load_model(Path(ck), "cuda" if torch.cuda.is_available() else "cpu")
        m = SamplingMPC(model, dn, pn, cfg.control, device="cuda" if torch.cuda.is_available() else "cpu")
        m.load_vel_ensemble(ens)
        mpcs[name] = m
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/data/deploy_rec_holdout/*.npz"))
    for arm in ("u0_", "wam_"):
        fs = [f for f in files if Path(f).name.startswith(arm)]
        if not fs:
            continue
        print(f"\n=== {arm[:-1]}-driven holdout ({len(fs)} eps) ===")
        print(f"{'core':<18} {'dyn 5-step MAE':>15} {'est |v err|':>12}  (m/s)")
        for name, m in mpcs.items():
            dyn_e, est_e = [], []
            for f in fs:
                dyn, pwm = load(f)
                for t in range(L, len(dyn) - K - 1, 4):
                    dyn_e.append(kstep_err(m, dyn, pwm, t))
                    ens = m.estimate_velocity_ens(dyn[t - L + 1:t + 1], pwm[t - L + 1:t + 1])
                    if ens is not None:
                        est_e.append(np.linalg.norm(ens[0][:2] - dyn[t, 0:2]))
            print(f"{name:<18} {np.mean(dyn_e):15.3f} {np.mean(est_e) if est_e else float('nan'):12.3f}")


if __name__ == "__main__":
    main()
