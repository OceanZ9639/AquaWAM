#!/usr/bin/env python3
"""Model-based dead reckoning vs windowed regression, offline on recordings.

At a simulated dropout t0 the last true DVL is known. Two ways to carry the
velocity through the blackout:
  A) windowed regressor (current blind path): v_est from a DVL-masked 1.6 s
     window each tick, no memory of the initial condition;
  B) world-model rollout: start from the true state at t0 and integrate the
     learned dynamics forward with the EXECUTED pwm, refreshing the disturbance
     code from the (self-filled) history. The initial condition is used.
Reports velocity error and integrated horizontal displacement error after
5/10/20/40 s, per driving arm, on held-out recordings.
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
from wam_policy_server import WamPolicy  # noqa: E402

L, DT = 16, 0.1


def load(npz):
    z = np.load(npz)
    pwm, dvl, av, la = z["pwm"], z["dvl"], z["imu_av"], z["imu_la"]
    pr, alt = z["pressure"].reshape(-1, 1), z["dvl_h"].reshape(-1, 1)
    prev = np.vstack([np.zeros((1, 8), np.float32), pwm[:-1]])
    dyn = np.concatenate([dvl, av, la, pr, alt, prev], axis=1).astype(np.float32)
    return dyn, pwm.astype(np.float32)


class ModelDR:
    """Roll the dynamics core one tick at a time from a known state."""

    def __init__(self, mpc: SamplingMPC):
        self.mpc = mpc
        self.model, self.dn, self.pn, self.dev = mpc.model, mpc.dyn_norm, mpc.pwm_norm, mpc.device
        self.K = mpc.cfg.horizon

    @torch.no_grad()
    def step(self, hist_s, hist_a, s_t, a_next):
        """One tick: s_t (raw, with velocity filled) + executed pwm a_next -> s_{t+1}."""
        hs = torch.from_numpy(self.dn(hist_s)).unsqueeze(0).to(self.dev)
        ha = torch.from_numpy(self.pn(hist_a)).unsqueeze(0).to(self.dev)
        d = self.model.disturbance(hs, ha)
        st = torch.from_numpy(self.dn(s_t)).unsqueeze(0).to(self.dev)
        af = torch.from_numpy(self.pn(np.tile(a_next[None], (self.K, 1)))).unsqueeze(0).to(self.dev)
        s_hat, _, _ = self.model.rollout(st, af, d)
        return self.dn.invert(s_hat[0, 0].cpu().numpy())


def run(files, mpc_reg, mpc_ens, t0=100, horizon=400):
    dr = ModelDR(mpc_reg)
    marks = [50, 100, 200, 400]
    acc = {m: {"A_v": [], "B_v": [], "C_v": [], "A_p": [], "B_p": [], "C_p": []} for m in marks}
    n = 0
    for f in files:
        dyn, pwm = load(f)
        if len(dyn) < t0 + 60:
            continue
        T = min(len(dyn) - 1, t0 + horizon)
        # A/B/C state
        hist_A = dyn[t0 - L + 1:t0 + 1].copy()
        hist_B = hist_A.copy()
        hist_C = hist_A.copy()
        s_B = dyn[t0].copy()
        s_C = dyn[t0].copy()
        pA = pB = pC = np.zeros(2)
        p_true = np.zeros(2)
        for t in range(t0, T):
            a_exec = pwm[t]                     # command issued at tick t
            ha = pwm[t - L + 1:t + 1]
            # --- A: regressor on a DVL-masked window (DVL cols irrelevant) ---
            v_A = mpc_reg.estimate_velocity(hist_A, ha)[:3]
            # --- B: model rollout from the last state ---
            s_B = dr.step(hist_B, ha, s_B, a_exec)
            s_B[3:] = dyn[t + 1][3:]           # IMU/pressure/alt/pwm are still measured
            v_B = s_B[0:3].copy()
            # --- C: fusion -- model prior corrected toward the ensemble (gain 0.3) ---
            ens = mpc_ens.estimate_velocity_ens(hist_C, ha)
            s_C = dr.step(hist_C, ha, s_C, a_exec)
            s_C[3:] = dyn[t + 1][3:]
            if ens is not None:
                s_C[0:3] = 0.7 * s_C[0:3] + 0.3 * ens[0]
            v_C = s_C[0:3].copy()
            # advance histories with self-filled velocity
            nxt = dyn[t + 1].copy()
            rowA = nxt.copy(); rowA[0:3] = v_A
            rowB = nxt.copy(); rowB[0:3] = v_B
            rowC = nxt.copy(); rowC[0:3] = v_C
            hist_A = np.vstack([hist_A[1:], rowA])
            hist_B = np.vstack([hist_B[1:], rowB])
            hist_C = np.vstack([hist_C[1:], rowC])
            v_true = dyn[t + 1][0:3]
            pA = pA + v_A[:2] * DT; pB = pB + v_B[:2] * DT; pC = pC + v_C[:2] * DT
            p_true = p_true + v_true[:2] * DT
            k = t - t0 + 1
            if k in marks:
                acc[k]["A_v"].append(np.linalg.norm(v_A[:2] - v_true[:2]))
                acc[k]["B_v"].append(np.linalg.norm(v_B[:2] - v_true[:2]))
                acc[k]["C_v"].append(np.linalg.norm(v_C[:2] - v_true[:2]))
                acc[k]["A_p"].append(np.linalg.norm(pA - p_true))
                acc[k]["B_p"].append(np.linalg.norm(pB - p_true))
                acc[k]["C_p"].append(np.linalg.norm(pC - p_true))
        n += 1
    print(f"  {n} episodes")
    print(f"  {'after':<8} {'|v err| A':>10} {'B':>7} {'C':>7} | {'|pos err| A':>12} {'B':>7} {'C':>7}")
    for m in marks:
        a = acc[m]
        if not a["A_v"]:
            continue
        print(f"  {m/10:5.0f} s  {np.mean(a['A_v']):10.3f} {np.mean(a['B_v']):7.3f} {np.mean(a['C_v']):7.3f} | "
              f"{np.mean(a['A_p']):12.2f} {np.mean(a['B_p']):7.2f} {np.mean(a['C_p']):7.2f}")


def main():
    args = types.SimpleNamespace(
        ckpt="/hy-tmp/models/uwam/best_ou.pt", vel_ens="/hy-tmp/models/uwam/vel_ens.pt",
        gate_calib="/hy-tmp/models/uwam/gate_calib.json", ou="", eval_root="/tmp/smoke_eval_root",
        hold_anchor="mixer", dump_dir="", goal_source="privileged")
    pol = WamPolicy(args)
    mpc_ens = SamplingMPC(pol.model, pol.dyn_norm, pol.pwm_norm, pol.cfg.control, device=pol.device)
    assert mpc_ens.load_vel_ensemble("/hy-tmp/models/uwam/vel_ens_deploy.pt")
    files = sorted(glob.glob("/hy-tmp/data/deploy_rec_holdout/*.npz"))
    for arm in ("wam_", "u0_", "fallback_"):
        fs = [f for f in files if Path(f).name.startswith(arm)]
        if fs:
            print(f"\n=== {arm[:-1]}-driven holdout ===")
            run(fs, pol.mpc, mpc_ens)


if __name__ == "__main__":
    main()
