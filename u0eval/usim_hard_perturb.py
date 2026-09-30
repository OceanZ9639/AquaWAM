#!/usr/bin/env python3
"""USIM-Hard: perturb the official mapper's output for one evaluation episode.

Runs between `mapper_setup.py` (which writes the scene .scn and the reference path
episode_<i>_traj.npy) and `roslaunch` (which loads them). The perturbation is seeded by
(task, episode), so U0 / WAM / fallback meet the identical start heading, water and path in
episode i -- a paired comparison. Nothing here touches the official mapper.

--mode is a comma list of:
  heading          start yaw += sign * U(90 deg, 180 deg): the vehicle no longer starts facing the goal
                   (every USIM demo starts facing it; the eval start pose is otherwise the official one)
  jerlov=<v>       water turbidity (Jerlov coefficient; official 0.15). Vision-only perturbation.
  roundtrip        composed task: reference path there AND back to the start (endpoint = start)
  loop             composed task (scan): full loop around the ship = official side + mirrored side back
  depth=<dz>       composed task: whole reference path shifted by dz metres in z (positive = deeper)

A side-car episode_<i>_traj.json is always written next to the path with the judge settings
(pos/yaw tolerance, sequential) so the WAM server can match its waypoint-advance radius to the
judge instead of assuming the official tolerances.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from pathlib import Path

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def perturb_scn(text: str, ops: dict, rng: np.random.Generator, info: dict) -> str:
    if "heading" in ops:
        m = re.search(r'(<arg name="orientation" value=")([^"]*)(")', text)
        if not m:
            raise RuntimeError("robot <arg name=\"orientation\"> not found in scn")
        r, p, y = (float(v) for v in m.group(2).split())
        delta = float(rng.uniform(math.radians(90), math.radians(180))) * (1 if rng.random() < 0.5 else -1)
        y2 = wrap(y + delta)
        text = text[:m.start(2)] + f"{r:.3f} {p:.3f} {y2:.3f}" + text[m.end(2):]
        info["heading"] = {"yaw_official": y, "yaw_hard": y2, "delta_deg": math.degrees(delta)}
    if "jerlov" in ops:
        v = float(ops["jerlov"])
        text, n = re.subn(r'jerlov="[^"]*"', f'jerlov="{v:.3f}"', text, count=1)
        if n != 1:
            raise RuntimeError("water jerlov attribute not found in scn")
        info["jerlov"] = v
    return text


def perturb_traj(wps: np.ndarray, ops: dict, info: dict) -> np.ndarray:
    if "roundtrip" in ops:
        back = wps[-2::-1].copy()
        # return leg: face the direction of travel (nav judges ignore yaw; WAM nav mode steers by travel)
        for k in range(len(back)):
            nxt = back[k + 1] if k + 1 < len(back) else back[k]
            prev = wps[-1] if k == 0 else back[k - 1]
            back[k, 3] = math.atan2(nxt[1] - prev[1], nxt[0] - prev[0])
        wps = np.vstack([wps, back])
        info["roundtrip"] = {"n_points": int(len(wps))}
    if "loop" in ops:
        # scan paths look at the ship centre (yaw = atan2(-y, -x)); the mirrored side is y -> -y,
        # which maps that look-at yaw to -yaw. Walk it back from the far end to the start.
        mir = wps[-2::-1].copy()
        mir[:, 1] = -mir[:, 1]
        mir[:, 3] = -mir[:, 3]
        wps = np.vstack([wps, mir])
        info["loop"] = {"n_points": int(len(wps))}
    if "depth" in ops:
        dz = float(ops["depth"])
        wps = wps.copy()
        wps[:, 2] += dz
        info["depth"] = {"dz": dz, "z_hard": float(wps[-1, 2])}
    return wps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scn", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--mode", default="")
    ap.add_argument("--pos-tol", type=float, default=-1.0)
    ap.add_argument("--yaw-tol", type=float, default=-1.0)
    ap.add_argument("--sequential", default="false")
    args = ap.parse_args()

    ops: dict = {}
    for tok in [t.strip() for t in args.mode.split(",") if t.strip()]:
        k, _, v = tok.partition("=")
        ops[k] = v if v else True
    sequential = str(args.sequential).lower() in ("1", "true", "yes")
    if ("roundtrip" in ops or "loop" in ops) and not sequential:
        print("[usim_hard] roundtrip/loop endpoint coincides with the start: forcing sequential judging")
        sequential = True

    seed = zlib.crc32(f"{args.task}:{args.episode}".encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    info: dict = {"task": args.task, "episode": args.episode, "seed": int(seed), "mode": args.mode,
                  "pos_tol": args.pos_tol, "yaw_tol": args.yaw_tol, "sequential": sequential}

    scn = Path(args.scn)
    if any(k in ops for k in ("heading", "jerlov")):
        text = scn.read_text()
        text = perturb_scn(text, ops, rng, info)
        scn.write_text(text)

    traj = Path(args.traj)
    if any(k in ops for k in ("roundtrip", "loop", "depth")):
        if not traj.exists():
            raise FileNotFoundError(traj)
        wps = np.asarray(np.load(traj, allow_pickle=True), dtype=np.float64)
        wps = perturb_traj(wps, ops, info)
        np.save(traj, wps)

    side = traj.with_suffix(".json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(info, indent=1))
    print(f"[usim_hard] {args.task} ep{args.episode}: {json.dumps(info)}")


if __name__ == "__main__":
    main()
