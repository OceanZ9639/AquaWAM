#!/usr/bin/env python3
"""Randomized thruster-fault collection on the 10 Hz DVL clock.

The fixed regimes only ever show four eta patterns, so an eta-regression head trained on them
memorises patterns instead of estimating per-thruster health, and the fault-severity sweep would
test severities the model has never seen. Here every segment draws a fresh fault (which thrusters,
how severe), the npz stores the COMMANDED pwm plus the full eta schedule as supervision, and
apply_efficiency stays purely on the plant side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.control import sixdof_thrust_to_pwm, vel_track_pwm
from uwam.rosbridge import SimBridge
from uwam.sim import apply_efficiency

TARGET_DT = 0.1
DEPTH_KEEP = (1.5, 5.0)


def sample_motion(rng: np.random.Generator) -> dict:
    if rng.random() < 0.2:
        thrust = np.zeros(3, np.float32)
        axis = int(rng.integers(0, 3))
        lim = 0.85 if axis < 2 else 0.5
        thrust[axis] = float(rng.uniform(-lim, lim))
        return {"kind": "primitive", "thrust": thrust}
    goal = np.zeros(3, np.float32)
    for ax, lim in enumerate((0.30, 0.30, 0.25)):
        if rng.random() > 0.3:
            goal[ax] = float(rng.uniform(-lim, lim))
    return {"kind": "mixer", "goal": goal,
            "kp": float(rng.uniform(1.0, 3.5)), "kff": float(rng.uniform(1.4, 2.6))}


def sample_eta(rng: np.random.Generator) -> np.ndarray:
    eta = np.ones(8, np.float32)
    if rng.random() < 0.3:
        return eta
    n_bad = 2 if rng.random() < 0.25 else 1
    for _ in range(n_bad):
        idx = int(rng.integers(0, 4)) if rng.random() < 0.7 else int(rng.integers(4, 8))
        eta[idx] = float(rng.uniform(0.2, 0.9))
    return eta


def collect_episode(bridge: SimBridge, n_frames: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pwm = np.zeros((n_frames, 8), np.float32)      # commanded
    eta_log = np.ones((n_frames, 8), np.float32)
    dvl = np.zeros((n_frames, 3), np.float32)
    imu_av = np.zeros((n_frames, 3), np.float32)
    imu_la = np.zeros((n_frames, 3), np.float32)
    pressure = np.zeros((n_frames, 1), np.float32)
    alt = np.zeros((n_frames, 1), np.float32)
    stamps = np.zeros((n_frames,), np.float64)
    dvl_seq = np.zeros((n_frames,), np.int32)

    bridge.home()
    motion = sample_motion(rng)
    eta = sample_eta(rng)
    seg_t0 = None
    hold = float(rng.uniform(5.0, 9.0))
    n_home = 0
    i = 0
    t0 = None
    while i < n_frames:
        st = bridge.tick()
        if st is None:
            break
        if t0 is None:
            t0 = st["stamp"]
        z = float(st["pos"][2])
        if not (DEPTH_KEEP[0] <= z <= DEPTH_KEEP[1]):
            bridge.home()
            n_home += 1
            motion, eta = sample_motion(rng), sample_eta(rng)
            seg_t0 = None
            hold = float(rng.uniform(5.0, 9.0))
            continue
        if seg_t0 is None:
            seg_t0 = st["stamp"]
        if st["stamp"] - seg_t0 >= hold:
            motion, eta = sample_motion(rng), sample_eta(rng)
            seg_t0 = st["stamp"]
            hold = float(rng.uniform(5.0, 9.0))
        if motion["kind"] == "primitive":
            u_cmd = sixdof_thrust_to_pwm(motion["thrust"])
        else:
            u_cmd = vel_track_pwm(st["dvl"], motion["goal"], st["imu_av"], st["rpy"],
                                  kp=motion["kp"], kff=motion["kff"])
        bridge.publish_pwm(apply_efficiency(u_cmd, tuple(eta)))
        pwm[i] = u_cmd
        eta_log[i] = eta
        dvl[i] = st["dvl"]
        imu_av[i] = st["imu_av"]
        imu_la[i] = st["imu_la"]
        pressure[i, 0] = st["pressure"]
        alt[i, 0] = st["alt"] if st["alt_valid"] else 1.0
        stamps[i] = st["stamp"]
        dvl_seq[i] = st["seq"]
        i += 1
    bridge.publish_pwm(np.zeros(8, np.float32))
    return {
        "n": i, "pwm": pwm[:i], "eta": eta_log[:i], "dvl": dvl[:i], "imu_av": imu_av[:i],
        "imu_la": imu_la[:i], "pressure": pressure[:i], "dvl_h": alt[:i],
        "stamps": stamps[:i], "dvl_seq": dvl_seq[:i], "n_home": n_home,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/hy-tmp/data/ou_explore",
                    help="written as ou_randfault_XX.npz next to the OU set so load_ou_split finds them")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bridge = SimBridge(node="uwam_collect_randfault")
    if not bridge.wait_ready():
        raise SystemExit("no DVL/odometry from the simulator")

    for k in range(args.episodes):
        t0 = time.time()
        print(f"collecting randfault episode {k} frames={args.frames}", flush=True)
        r = collect_episode(bridge, args.frames, args.seed + k)
        if r["n"] < 8:
            raise SystemExit(f"only {r['n']} frames in episode {k}")
        path = out / f"ou_randfault_{k:02d}.npz"
        np.savez_compressed(
            path,
            pwm=r["pwm"], eta=r["eta"], pwm_is_commanded=np.bool_(True),
            dvl=r["dvl"], imu_av=r["imu_av"], imu_la=r["imu_la"],
            pressure=r["pressure"], dvl_h=r["dvl_h"],
            timestamp=(r["stamps"] - r["stamps"][0]).astype(np.float64), dvl_seq=r["dvl_seq"],
        )
        frac_fault = float((r["eta"] < 0.99).any(axis=1).mean())
        meta = {
            "episode": k, "frames": int(r["n"]),
            "policy": "randomized thruster faults over mixer/primitive motion segments",
            "frac_frames_with_fault": round(frac_fault, 3),
            "n_rehome": int(r["n_home"]),
            "wall_sec": round(time.time() - t0, 1),
        }
        (out / f"ou_randfault_{k:02d}.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)
    print("saved randfault", out, flush=True)


if __name__ == "__main__":
    main()
