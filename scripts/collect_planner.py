#!/usr/bin/env python3
"""Collect the planner's action distribution, on the same 10 Hz DVL clock as the OU set.

Recovery doc section 8: a WAM trained only on expert PWM (USIM) and OU noise answers queries it
has never seen, so its surge response is nearly action-independent. This collects exactly the
commands SamplingMPC proposes -- the mixer P+FF family over a gain grid, plus saturated per-axis
primitives -- across the same 6 regimes, written in the OU npz schema so load_ou_split reads it.
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
from uwam.sim import REGIMES, apply_efficiency, regime_at

TARGET_DT = 0.1
DEPTH_KEEP = (1.5, 5.0)


def sample_segment(rng: np.random.Generator) -> dict:
    """One held command: either a mixer velocity goal or a saturated axis primitive."""
    if rng.random() < 0.2:
        thrust = np.zeros(3, np.float32)
        axis = int(rng.integers(0, 3))
        lim = 0.85 if axis < 2 else 0.5
        thrust[axis] = float(rng.uniform(-lim, lim))
        return {"kind": "primitive", "thrust": thrust, "hold": float(rng.uniform(2.0, 4.0))}
    goal = np.zeros(3, np.float32)
    for ax, lim in enumerate((0.30, 0.30, 0.25)):
        if rng.random() > 0.3:
            goal[ax] = float(rng.uniform(-lim, lim))
    return {
        "kind": "mixer",
        "goal": goal,
        "kp": float(rng.uniform(1.0, 3.5)),
        "kff": float(rng.uniform(1.4, 2.6)),
        "hold": float(rng.uniform(2.0, 4.0)),
    }


def collect_regime(bridge: SimBridge, regime, n_frames: int, seed: int,
                   act_every: int = 1, images: bool = False) -> dict:
    """One regime. act_every > 1 puts the whole record on a coarser grid (0.1 * act_every s):
    the command is recomputed and the state recorded only on acting ticks, with the command
    held in between (that IS the coarse-rate plant)."""
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
    rgb_frames, fls_frames = [], []

    bridge.home()
    seg = sample_segment(rng)
    seg_t0 = None
    n_home = 0
    i = 0
    t0 = None
    tick = -1
    u_cmd = np.zeros(8, np.float32)
    while i < n_frames:
        st = bridge.tick()
        if st is None:
            break
        if t0 is None:
            t0 = st["stamp"]
        t_rel = st["stamp"] - t0
        tick += 1
        z = float(st["pos"][2])
        if not (DEPTH_KEEP[0] <= z <= DEPTH_KEEP[1]):
            bridge.home()
            n_home += 1
            seg, seg_t0 = sample_segment(rng), None
            continue
        cur, eta = regime_at(regime, t_rel)
        bridge.publish_current(cur)
        if tick % act_every != 0:
            # hold the previous command between acting ticks (ZOH plant at the coarse rate)
            bridge.publish_pwm(apply_efficiency(u_cmd, eta))
            continue
        if seg_t0 is None:
            seg_t0 = st["stamp"]
        if st["stamp"] - seg_t0 >= seg["hold"]:
            seg, seg_t0 = sample_segment(rng), st["stamp"]
        if seg["kind"] == "primitive":
            u_cmd = sixdof_thrust_to_pwm(seg["thrust"])
        else:
            u_cmd = vel_track_pwm(st["dvl"], seg["goal"], st["imu_av"], st["rpy"],
                                  kp=seg["kp"], kff=seg["kff"])
        bridge.publish_pwm(apply_efficiency(u_cmd, eta))
        # OU convention: pwm[i] is the COMMAND issued at record tick i, applied until the next one
        pwm[i] = u_cmd
        eta_log[i] = np.asarray(eta, np.float32)
        dvl[i] = st["dvl"]
        imu_av[i] = st["imu_av"]
        imu_la[i] = st["imu_la"]
        pressure[i, 0] = st["pressure"]
        alt[i, 0] = st["alt"] if st["alt_valid"] else 1.0
        stamps[i] = st["stamp"]
        dvl_seq[i] = st["seq"]
        if images:
            rgb_frames.append(None if st["rgb"] is None else st["rgb"].copy())
            fls_frames.append(None if st["fls"] is None else st["fls"].copy())
        i += 1
    bridge.publish_pwm(np.zeros(8, np.float32))
    out = {
        "n": i,
        "pwm": pwm[:i], "eta": eta_log[:i], "dvl": dvl[:i], "imu_av": imu_av[:i],
        "imu_la": imu_la[:i], "pressure": pressure[:i], "dvl_h": alt[:i],
        "stamps": stamps[:i], "dvl_seq": dvl_seq[:i], "n_home": n_home,
    }
    if images:
        def _pack(frames):
            good = next((f for f in frames if f is not None), None)
            if good is None:
                return None
            return np.stack([f if f is not None else good for f in frames], axis=0)
        out["rgb"] = _pack(rgb_frames)
        out["fls"] = _pack(fls_frames)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/hy-tmp/data/planner_mix")
    ap.add_argument("--frames", type=int, default=900, help="DVL ticks per regime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-every", type=int, default=1,
                    help="record/act on every N-th DVL tick (2 = 0.2 s grid dataset)")
    ap.add_argument("--images", action="store_true", help="record RGB + FLS frames per tick")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bridge = SimBridge(node="uwam_collect_planner", images=args.images)
    if not bridge.wait_ready():
        raise SystemExit("no DVL/odometry from the simulator")
    dt_grid = TARGET_DT * args.act_every

    for idx, reg in enumerate(REGIMES):
        t0 = time.time()
        print(f"collecting planner-distribution {reg.name} frames={args.frames} dt={dt_grid}", flush=True)
        r = collect_regime(bridge, reg, args.frames, args.seed + idx,
                           act_every=args.act_every, images=args.images)
        if r["n"] < 8:
            raise SystemExit(f"only {r['n']} frames for {reg.name}")
        path = out / f"ou_{reg.name}.npz"
        payload = dict(
            pwm=r["pwm"], eta=r["eta"], pwm_is_commanded=np.bool_(True),
            dvl=r["dvl"], imu_av=r["imu_av"], imu_la=r["imu_la"],
            pressure=r["pressure"], dvl_h=r["dvl_h"],
            timestamp=(r["stamps"] - r["stamps"][0]).astype(np.float64), dvl_seq=r["dvl_seq"],
            dt=np.float64(dt_grid),
        )
        if args.images:
            if r.get("rgb") is not None:
                payload["rgb"] = r["rgb"]
            if r.get("fls") is not None:
                payload["fls"] = r["fls"]
        np.savez_compressed(path, **payload)
        dts = np.diff(r["stamps"]) if r["n"] > 1 else np.array([dt_grid])
        meta = {
            "regime": reg.name,
            "frames": int(r["n"]),
            "policy": "mixer P+FF gain grid + saturated axis primitives (planner query distribution)",
            "dt_median": float(np.median(dts)),
            "dt_target": dt_grid,
            "act_every": args.act_every,
            "images": bool(args.images),
            "n_rehome": int(r["n_home"]),
            "speed_p90": [round(float(x), 4) for x in np.percentile(np.abs(r["dvl"]), 90, axis=0)],
            "wall_sec": round(time.time() - t0, 1),
        }
        (out / f"ou_{reg.name}.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
