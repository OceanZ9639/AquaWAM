#!/usr/bin/env python3
"""Calibrate the CUSUM trust gate from held-out data instead of hand-tuning.

Builds z streams that mirror the deployment computation exactly (self-filled blind
history, n-tick smoothing, sigma from the NLL ensemble, anchor = the estimator's own
pre-dropout output):
  null streams  = windows with NO change during the simulated blackout -> false alarms
  shift streams = randomized-fault windows with a real mid-blackout eta change -> delay
then grid-searches (kappa, h) for the smallest detection delay subject to a
false-alarm budget. Output: gate_calib.json consumed by closed_loop.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg, enable_arm  # noqa: E402
from uwam.control import SamplingMPC  # noqa: E402
from uwam.data import RunningNorm, load_ou_split  # noqa: E402
from uwam.gate import calibrate  # noqa: E402
from uwam.models import DynamicsWAM  # noqa: E402


def _load_mpc(ckpt: Path, device: str) -> SamplingMPC:
    import torch

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = Cfg()
    cfg.model.use_language = False
    if ck.get("cfg", {}).get("use_arm"):
        enable_arm(cfg)
    model = DynamicsWAM(cfg)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    dyn_norm, pwm_norm = RunningNorm(), RunningNorm()
    dyn_norm.load_state_dict(ck["dyn_norm"])
    pwm_norm.load_state_dict(ck["pwm_norm"])
    return SamplingMPC(model, dyn_norm, pwm_norm, cfg.control, device=device)


def build_stream(mpc, dyn, pwm, t0, T_blind, L, n_smooth, floor):
    """Replay the deployment estimator over one window; return the z stream.

    dyn rows carry the recorded state; during the blind part the DVL columns are
    self-filled with the estimator's own output, exactly as in closed_loop.py.
    """
    vel_std = mpc.dyn_norm.std[0:3].astype(np.float64)
    hist = dyn[t0 - L - n_smooth:t0 + T_blind].copy()  # local buffer we can overwrite
    off = L + n_smooth                                  # index of t0 inside `hist`
    est_sighted, est_hist, zs = [], [], []
    anchor = None
    for k in range(-n_smooth, T_blind):
        i = off + k
        hs = hist[i - L:i]
        ha = pwm[t0 + k - L:t0 + k]
        v_est = mpc.estimate_velocity(hs, ha)
        ens = mpc.estimate_velocity_ens(hs, ha)
        if v_est is None or ens is None:
            return None
        sd_alea, sd_epi = ens[1].astype(np.float64), ens[2].astype(np.float64)
        if k < 0:
            est_sighted.append(v_est)
            continue                                    # sighted tick: recorded DVL stays
        if anchor is None:
            anchor = np.mean(np.stack(est_sighted, 0), axis=0) / vel_std
        hist[i, 0:3] = v_est                            # self-fill this blind tick
        est_hist.append(v_est.copy())
        est_hist = est_hist[-n_smooth:]
        v_sm = np.mean(np.stack(est_hist, 0), axis=0) / vel_std
        sig_eff = np.sqrt(sd_alea ** 2 / max(1, len(est_hist)) + sd_epi ** 2)
        z = float(np.linalg.norm(v_sm - anchor)) / max(float(np.linalg.norm(sig_eff + floor)), 1e-9)
        zs.append(z)
    return np.asarray(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_ou.pt")
    ap.add_argument("--vel-ens", default="/hy-tmp/models/uwam/vel_ens.pt")
    ap.add_argument("--data", nargs="+", default=["/hy-tmp/data/ou_explore", "/hy-tmp/data/planner_mix"])
    ap.add_argument("--t-blind", type=float, default=6.0, help="simulated blackout length (s)")
    ap.add_argument("--far", type=float, default=0.05, help="false-alarm budget per blackout window")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-null", type=int, default=150)
    ap.add_argument("--stat-tol", type=float, default=0.08,
                    help="stationarity filter (m/s): the gate's null hypothesis is a HELD "
                         "command, so calibration windows must be velocity-stationary; "
                         "wandering-action windows are a different (harder) null")
    ap.add_argument("--out", default="/hy-tmp/models/uwam/gate_calib.json")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mpc = _load_mpc(Path(args.ckpt), device)
    if not mpc.load_vel_ensemble(args.vel_ens):
        raise SystemExit(f"sigma ensemble not found at {args.vel_ens}")

    cfg = Cfg()
    L = cfg.model.history_len
    n_smooth = int(cfg.control.blind_innov_smooth)
    floor = 0.05
    dt = 0.1
    T_blind = int(round(args.t_blind / dt))

    use_arm = int(np.asarray(mpc.dyn_norm.mean).reshape(-1).shape[0]) > 19
    eps = []
    for d in args.data:
        eps += load_ou_split(Path(d), use_arm=use_arm)
    print(f"{len(eps)} episodes from {args.data} use_arm={use_arm}", flush=True)

    vel_std = mpc.dyn_norm.std[0:3].astype(np.float64)
    z_null, z_shift, shift_sizes = [], [], []
    for ep in eps:
        eta = ep.eta_arr if ep.eta_arr is not None else np.ones((len(ep.pwm), 8), np.float32)
        dvl = ep.dyn[:, 0:3]
        T = len(ep.pwm)
        lo = L + n_smooth
        for t0 in range(lo, T - T_blind, args.stride):
            win = eta[t0 - lo:t0 + T_blind]
            changes = np.nonzero(np.any(np.abs(np.diff(eta[t0:t0 + T_blind], axis=0)) > 1e-3, axis=1))[0]
            v = dvl[t0:t0 + T_blind]
            if len(changes) == 0 and np.all(np.abs(win - win[0]) < 1e-3):
                if len(z_null) < args.max_null and np.abs(v - v[0]).max() < args.stat_tol:
                    z = build_stream(mpc, ep.dyn, ep.pwm, t0, T_blind, L, n_smooth, floor)
                    if z is not None:
                        z_null.append(z)
            elif len(changes) == 1 and 8 <= changes[0] <= T_blind - 15:
                tc = int(changes[0]) + 1
                pre = v[:tc]
                if np.abs(pre - pre[0]).max() < args.stat_tol:
                    z = build_stream(mpc, ep.dyn, ep.pwm, t0, T_blind, L, n_smooth, floor)
                    if z is not None:
                        # effect size: how far the change actually moved the (smoothed,
                        # normalized) velocity from its pre-change level
                        vs = np.stack([v[max(0, i - n_smooth + 1):i + 1].mean(0)
                                       for i in range(len(v))], 0) / vel_std
                        pre_m = vs[:tc].mean(0)
                        z_shift.append((z, tc))
                        shift_sizes.append(float(np.linalg.norm(vs[tc:] - pre_m, axis=1).max()))
    print(f"streams: null={len(z_null)} shift={len(z_shift)}", flush=True)
    if not z_null or not z_shift:
        raise SystemExit("not enough streams for calibration")

    report = calibrate(z_null, z_shift, dt=dt, far_target=args.far, shift_sizes=shift_sizes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
