#!/usr/bin/env python3
"""
Blind velocity estimator, v1 (exploration + early deployment data) vs v2 (v1 data + 88 h of closed-loop
sensor logs = DAgger for the estimator), replayed offline on HELD-OUT closed-loop episodes (trial
index % 10 == 0, never converted for training; plus any extra dirs given).

Per set: surge bias, xy MAE, bias at cruise (|v| > 0.45 m/s), MAE near hover (|v| < 0.2), and the
30 s dead-reckoning position error obtained by integrating the estimate from the drop tick with the
true heading (the quantity that decides whether a blind goto arrives). Writes u0eval/estimator_eval.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import SamplingMPC  # noqa: E402
from uwam.direct import load_core  # noqa: E402

RUNS = "/hy-tmp/u0env/dataset/eval_runs"
ESTS = {"v1": "/hy-tmp/models/uwam/vel_ens_scenes.pt", "v2": "/hy-tmp/models/uwam/vel_ens_v2.pt"}
SETS = {
    "goto_water_tower blind (WAM r4 vision, drop 8 s)": f"{RUNS}/wam_percept_r4_drop8s_zero/goto_water_tower",
    "goto_charge_station blind (WAM r4 vision, drop 8 s)": f"{RUNS}/wam_percept_r4_drop8s_zero/goto_charge_station",
    "scan_ship_ancient blind (fallback, drop 40 s)": f"{RUNS}/fallback_drop40s_zero/scan_ship_ancient",
    "follow_boat blind (WAM, drop 20 s)": f"{RUNS}/wam_drop20s_zero/follow_boat",
    "pick_red_shallow (U0, drop 30 s, hover)": f"{RUNS}/u0_drop30s_zero/pick_red_shallow",
}


def load(p):
    z = np.load(p, allow_pickle=True)
    rows, cols, lay = z["rows"], list(z["cols"]), list(z["layout"])
    off, out = 0, {}
    for c, n in zip(cols, lay):
        out[str(c)] = rows[:, off:off + int(n)]
        off += int(n)
    return out


def replay(d, mp, L=16):
    n = len(d["t"])
    est = np.full((n, 3), np.nan, np.float32)
    for t in range(L, n):
        sl = slice(t - L, t)
        frames = np.concatenate([d["dvl_eff"][sl], d["imu_av"][sl], d["imu_la"][sl], d["pressure"][sl] * 1e4,
                                 d["alt_eff"][sl], d["pwm_cmd"][sl]], -1).astype(np.float32)
        hist_a = np.vstack([d["pwm_cmd"][t - L + 1:t], d["pwm_cmd"][t - 1:t]]).astype(np.float32)
        est[t] = mp.estimate_velocity_ens(frames, hist_a)[0]
    return est


def metrics(d, est):
    ok = np.isfinite(est[:, 0]) & (d["pressure"][:, 0] > 0)
    vt, ve = d["dvl_true"][ok], est[ok]
    sp = np.linalg.norm(vt[:, :2], axis=1)
    fast, slow = sp > 0.45, sp < 0.2
    dropped = d["dvl_valid"][:, 0] < 0.5
    k0 = max(16, int(np.argmax(dropped)) if dropped.any() else 16)
    k1 = min(len(d["t"]) - 1, k0 + 300)
    pos, yaw = d["odom_pos"], d["odom_rpy"][:, 2]
    pp = pos[k0, :2].copy()
    for k in range(k0, k1):
        if not np.isfinite(est[k, 0]):
            continue
        cy, sy = np.cos(yaw[k]), np.sin(yaw[k])
        vx, vy = est[k, 0], -est[k, 1]
        pp[0] += (cy * vx - sy * vy) * 0.1
        pp[1] += (sy * vx + cy * vy) * 0.1
    return dict(bias=float(np.mean(ve[:, 0] - vt[:, 0])), mae=float(np.mean(np.abs(ve[:, :2] - vt[:, :2]))),
                cruise_bias=float(np.mean(ve[fast, 0] - vt[fast, 0])) if fast.any() else np.nan,
                cruise_frac=float(fast.mean()),
                hover_mae=float(np.mean(np.abs(ve[slow, :2] - vt[slow, :2]))) if slow.any() else np.nan,
                dr30=float(np.linalg.norm(pp - pos[k1, :2])))


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, dn, pn, cfg = load_core("/hy-tmp/models/uwam/best_scenes.pt", dev)
    ens = {}
    for name, path in ESTS.items():
        if not Path(path).exists():
            continue
        mp = SamplingMPC(m, dn, pn, cfg.control, device=dev)
        mp.load_vel_ensemble(path)
        ens[name] = mp
    out = {"estimators": {k: v for k, v in ESTS.items() if k in ens}, "sets": {}}
    for label, d in SETS.items():
        paths = [p for p in sorted(glob.glob(f"{d}/logs/episode_*_sensors.npz"))
                 if int(os.path.basename(p).split("_")[1]) % 10 == 0][:6]
        if not paths:
            continue
        res = {name: [] for name in ens}
        for p in paths:
            data = load(p)
            for name, mp in ens.items():
                res[name].append(metrics(data, replay(data, mp)))
        agg = {name: {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]} for name, rs in res.items()}
        agg["n"] = len(paths)
        out["sets"][label] = agg
        for name in ens:
            a = agg[name]
            print(f"{label:52s} {name}: n={len(paths)} bias {a['bias']:+.3f} MAE {a['mae']:.3f} cruise bias "
                  f"{a['cruise_bias']:+.3f} ({100 * a['cruise_frac']:.0f}% ticks) hover MAE {a['hover_mae']:.3f} "
                  f"30 s DR {a['dr30']:.2f} m")
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
