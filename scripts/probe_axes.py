#!/usr/bin/env python3
"""Calibrate the mixer: per-axis sign and reachable DVL speed, with leveling + homing on.

Run this before closed_loop.py; it is the ground truth for AXIS_SIGN and for the goal set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.control import AXIS_SIGN, sixdof_thrust_to_pwm, vel_track_pwm
from uwam.rosbridge import SimBridge

TARGET_DT = 0.1


def run_segment(bridge: SimBridge, u_fn, seconds: float, tail: float = 2.0) -> dict:
    """Drive u_fn(state) for `seconds` of sim time; average DVL over the last `tail` seconds."""
    t0 = None
    rows = []
    while True:
        st = bridge.tick()
        if st is None:
            break
        if t0 is None:
            t0 = st["stamp"]
        t = st["stamp"] - t0
        bridge.publish_pwm(u_fn(st))
        rows.append((t, st["dvl"].copy(), st["rpy"].copy(), float(st["pos"][2])))
        bad, why = bridge.is_unstable(st)
        if bad:
            return {"aborted": why, "t": t}
        if t >= seconds:
            break
    if not rows:
        return {"aborted": "no_data"}
    keep = [r for r in rows if r[0] >= max(0.0, rows[-1][0] - tail)]
    dvl = np.stack([r[1] for r in keep])
    rpy = np.stack([r[2] for r in keep])
    return {
        "dvl_mean": [round(float(x), 4) for x in dvl.mean(0)],
        "dvl_max_abs": [round(float(x), 4) for x in np.abs(dvl).max(0)],
        "roll_deg": round(float(np.rad2deg(np.abs(rpy[:, 0]).mean())), 2),
        "pitch_deg": round(float(np.rad2deg(np.abs(rpy[:, 1]).mean())), 2),
        "z": round(float(keep[-1][3]), 2),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="/hy-tmp/logs/uwam/axis_probe.json")
    args = ap.parse_args()

    bridge = SimBridge(node="uwam_probe_axes")
    if not bridge.wait_ready():
        raise SystemExit("no DVL/odometry from the simulator")

    report = {"axis_sign_used": [float(x) for x in AXIS_SIGN], "segments": {}}

    report["home_initial"] = bridge.home()
    print("home", json.dumps(report["home_initial"]), flush=True)

    # 1) open-loop unit thrust per axis: does +thrust move the DVL axis positively?
    for axis, name in enumerate("xyz"):
        for sgn in (1.0, -1.0):
            bridge.home()
            mag = 0.5 if axis < 2 else 0.4
            t = np.zeros(3, np.float32)
            t[axis] = sgn * mag
            u = sixdof_thrust_to_pwm(t)
            key = f"open_{name}{'+' if sgn > 0 else '-'}"
            report["segments"][key] = run_segment(bridge, lambda st, u=u: u, args.seconds)
            print(key, json.dumps(report["segments"][key]), flush=True)

    # 2) closed-loop velocity tracking through the mixer at the goals we actually score
    for goal in ([0.25, 0, 0], [0, 0.20, 0], [0, 0, 0.15], [0.12, 0, 0], [0, 0.12, 0]):
        g = np.array(goal, np.float32)
        bridge.home()
        key = "track_" + "_".join(f"{x:g}" for x in goal)
        report["segments"][key] = run_segment(
            bridge,
            lambda st, g=g: vel_track_pwm(st["dvl"], g, st["imu_av"], st["rpy"]),
            args.seconds,
        )
        seg = report["segments"][key]
        if "dvl_mean" in seg:
            seg["goal"] = goal
            seg["track_err"] = round(float(np.linalg.norm(np.array(seg["dvl_mean"]) - g)), 4)
        print(key, json.dumps(seg), flush=True)

    bridge.publish_pwm(np.zeros(8, np.float32))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
