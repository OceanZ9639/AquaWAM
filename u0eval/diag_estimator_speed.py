#!/usr/bin/env python3
"""Dead-reckoning estimator error vs vehicle speed, on recorded eval episodes.

Replays the 10 Hz recordings (episode0 of each block) through the same
estimate_velocity() the blind path uses (DVL columns masked), and bins the
error by true speed. Separates U0-driven windows (saturated PWM, up to 0.7 m/s)
from WAM-driven ones (<= 0.3 m/s) to test the takeover-at-speed hypothesis.
"""
import glob
import pickle
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, "/hy-tmp/underwater_wam/u0eval")
sys.path.insert(0, "/hy-tmp/underwater_wam")
from wam_policy_server import WamPolicy  # noqa: E402

L = 16


def load_ep(ep_dir: Path):
    pk = sorted(ep_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    rows = []
    for p in pk:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        st = d["observation"]["state"]
        if not (st.get("dvl") and st.get("imu") and st.get("pressure")):
            continue
        pwm = d["action"].get("pwm")
        pwm = np.zeros(8, np.float32) if pwm is None else np.asarray(pwm, np.float32)[:8]
        v = st["dvl"]["velocity"]
        av = st["imu"]["angular_velocity"]
        la = st["imu"]["linear_acceleration"]
        alt = st["dvl"].get("altitude", 0.0)
        rows.append((np.array([v["x"], v["y"], v["z"]], np.float32),
                     np.array([av["x"], av["y"], av["z"]], np.float32),
                     np.array([la["x"], la["y"], la["z"]], np.float32),
                     np.float32(st["pressure"]["fluid_pressure"]), np.float32(alt), pwm))
    return rows


def frames_from(rows, t):
    """[L,19] window ending at t and the matching [L,8] action history."""
    dyn, act = [], []
    for i in range(t - L + 1, t + 1):
        v, av, la, pr, alt, pwm = rows[i]
        prev_pwm = rows[i - 1][5] if i > 0 else np.zeros(8, np.float32)
        dyn.append(np.concatenate([v, av, la, [pr], [alt], prev_pwm]))
        act.append(pwm)
    return np.asarray(dyn, np.float32), np.asarray(act, np.float32)


def main():
    args = types.SimpleNamespace(
        ckpt="/hy-tmp/models/uwam/best_ou.pt", vel_ens="/hy-tmp/models/uwam/vel_ens.pt",
        gate_calib="/hy-tmp/models/uwam/gate_calib.json", ou="", eval_root="/tmp/smoke_eval_root",
        hold_anchor="mixer", dump_dir="", goal_source="privileged")
    pol = WamPolicy(args)
    root = Path("/hy-tmp/u0env/dataset/eval_runs")
    groups = {
        "U0-driven": sorted(glob.glob(str(root / "u0_full" / "*" / "episode0"))),
        "WAM-driven": sorted(glob.glob(str(root / "wam_full" / "*" / "episode0"))),
    }
    bins = [0.0, 0.15, 0.3, 0.45, 0.6, 1.5]
    for name, eps in groups.items():
        errs, speeds, sat = [], [], []
        for ep in eps:
            rows = load_ep(Path(ep))
            for t in range(L, len(rows), 3):
                fr, ha = frames_from(rows, t)
                v_est = pol.mpc.estimate_velocity(fr, ha)
                if v_est is None:
                    continue
                v_true = rows[t][0]
                errs.append(np.linalg.norm(v_est[:2] - v_true[:2]))
                speeds.append(np.linalg.norm(v_true[:2]))
                sat.append(np.mean(np.abs(ha[:, :4]) > 0.9))
        errs, speeds, sat = map(np.asarray, (errs, speeds, sat))
        print(f"\n=== {name}: {len(eps)} episodes, {len(errs)} windows; "
              f"saturated-thruster fraction {sat.mean():.2f} ===")
        print(f"{'speed bin (m/s)':<18} {'n':>6} {'|v err| mean':>13} {'p90':>7}  -> pos err after 10 s")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (speeds >= lo) & (speeds < hi)
            if m.sum() < 10:
                continue
            e = errs[m]
            print(f"{lo:4.2f}-{hi:4.2f}          {m.sum():6d} {e.mean():13.3f} {np.percentile(e, 90):7.3f}"
                  f"     {10 * e.mean():5.2f} m")


if __name__ == "__main__":
    main()
