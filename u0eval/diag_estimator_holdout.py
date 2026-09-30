#!/usr/bin/env python3
"""Held-out estimator comparison on deployment recordings the ensemble never saw.

Sources: model vel head (current blind path), old ensemble mean (vel_ens.pt),
new deployment-trained ensemble mean (vel_ens_deploy.pt). Errors in m/s on the
horizontal DVL velocity, binned by true speed, per driving arm.
"""
import glob
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, "/hy-tmp/underwater_wam/u0eval")
sys.path.insert(0, "/hy-tmp/underwater_wam")
from uwam.control import SamplingMPC  # noqa: E402
from wam_policy_server import WamPolicy  # noqa: E402

L = 16


def windows(npz):
    z = np.load(npz)
    pwm, dvl, av, la = z["pwm"], z["dvl"], z["imu_av"], z["imu_la"]
    pr, alt = z["pressure"].reshape(-1, 1), z["dvl_h"].reshape(-1, 1)
    prev = np.vstack([np.zeros((1, 8), np.float32), pwm[:-1]])
    dyn = np.concatenate([dvl, av, la, pr, alt, prev], axis=1).astype(np.float32)
    for t in range(L, len(dyn), 3):
        yield dyn[t - L + 1:t + 1], pwm[t - L + 1:t + 1], dvl[t]


def main():
    args = types.SimpleNamespace(
        ckpt="/hy-tmp/models/uwam/best_ou.pt", vel_ens="/hy-tmp/models/uwam/vel_ens.pt",
        gate_calib="/hy-tmp/models/uwam/gate_calib.json", ou="", eval_root="/tmp/smoke_eval_root",
        hold_anchor="mixer", dump_dir="", goal_source="privileged")
    pol = WamPolicy(args)
    mpc_new = SamplingMPC(pol.model, pol.dyn_norm, pol.pwm_norm, pol.cfg.control, device=pol.device)
    assert mpc_new.load_vel_ensemble("/hy-tmp/models/uwam/vel_ens_deploy.pt")
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/data/deploy_rec_holdout/*.npz"))
    bins = [0.0, 0.15, 0.3, 0.45, 0.6, 1.5]
    for arm in ("u0_", "wam_", "fallback_"):
        fs = [f for f in files if Path(f).name.startswith(arm)]
        if not fs:
            continue
        E = {"head": [], "ens_old": [], "ens_new": []}
        S = []
        for f in fs:
            for fr, ha, v_true in windows(f):
                E["head"].append(np.linalg.norm(pol.mpc.estimate_velocity(fr, ha)[:2] - v_true[:2]))
                E["ens_old"].append(np.linalg.norm(pol.mpc.estimate_velocity_ens(fr, ha)[0][:2] - v_true[:2]))
                E["ens_new"].append(np.linalg.norm(mpc_new.estimate_velocity_ens(fr, ha)[0][:2] - v_true[:2]))
                S.append(np.linalg.norm(v_true[:2]))
        S = np.asarray(S)
        E = {k: np.asarray(v) for k, v in E.items()}
        print(f"\n=== {arm[:-1]}-driven holdout ({len(fs)} eps, {len(S)} windows) ===")
        print(f"{'speed':<12} {'n':>5} {'head':>8} {'ens_old':>8} {'ens_new':>8}")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (S >= lo) & (S < hi)
            if m.sum() < 10:
                continue
            print(f"{lo:.2f}-{hi:.2f}    {m.sum():5d} {E['head'][m].mean():8.3f} "
                  f"{E['ens_old'][m].mean():8.3f} {E['ens_new'][m].mean():8.3f}")
        print(f"{'all':<12} {len(S):5d} {E['head'].mean():8.3f} {E['ens_old'].mean():8.3f} {E['ens_new'].mean():8.3f}")


if __name__ == "__main__":
    main()
