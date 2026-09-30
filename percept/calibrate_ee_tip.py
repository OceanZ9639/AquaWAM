#!/usr/bin/env python3
"""
Calibrate the constant offset between the recorded end-effector pose (/alpha_ee_pose, what the
bridge and our labels use) and the gripper point the JUDGE scores (MoveIt get_current_pose of the
grasp group). Both are logged with wall-clock time: the judge writes dx,dy,dz (gripper - object,
world axes) at 1 Hz into logs/episode_i_data.csv, the recorder stamps every 10 Hz frame.

Model:  off_judge_world = off_ours_world + R_ee_world @ delta_ee      (delta_ee constant, 3 unknowns)
Solved by least squares over all aligned rows of all episodes found under --roots.  The result is
written to percept/ee_tip_offset.json and consumed by pack_grasp_recordings.py (and the planner).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_grasp_recordings import R_EE2ROV, T_EE2ROV, pose_q, pose_xyz, quat_to_R  # noqa: E402


def wall(ts: str) -> float:
    return datetime.strptime(ts, "%Y%m%d_%H%M%S_%f").timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="/hy-tmp/u0env/dataset/eval_runs,/hy-tmp/u0env_b/dataset/eval_runs")
    ap.add_argument("--glob", default="*/pick_*/episode*,*/transfer_*/episode*")
    ap.add_argument("--max-episodes", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ee_tip_offset.json"))
    args = ap.parse_args()
    A, B = [], []   # rows of R_ee_world (3x3) and residuals (3,)
    n_ep = 0
    for root in args.roots.split(","):
        root = Path(root)
        for g in args.glob.split(","):
            for ep in sorted(root.glob(g.strip())):
                if n_ep >= args.max_episodes or not re.match(r"episode\d+$", ep.name):
                    continue
                n = int(ep.name[len("episode"):])
                jcsv = ep.parent / "logs" / f"episode_{n}_data.csv"
                if not jcsv.exists():
                    continue
                judge = []
                for r in csv.reader(open(jcsv)):
                    if r and r[0] != "time" and len(r) >= 5:
                        try:
                            judge.append((datetime.fromisoformat(r[0]).timestamp(), float(r[1]), float(r[2]), float(r[3])))
                        except ValueError:
                            pass
                if not judge:
                    continue
                pkls = sorted((q for q in ep.glob("*.pkl") if q.stem.isdigit()), key=lambda q: int(q.stem))
                frames = []
                for p in pkls:
                    try:
                        d = pickle.load(open(p, "rb"))
                    except Exception:
                        continue
                    st = d["observation"]["state"]
                    if st.get("odom") is None or st.get("object_odom") is None or st.get("ee_pose") is None:
                        continue
                    try:
                        t = wall(str(d["timestamp"]))
                    except Exception:
                        continue
                    rov_p = pose_xyz(st["odom"]["pose"]["pose"]["position"]); rov_q = pose_q(st["odom"]["pose"]["pose"]["orientation"])
                    obj_p = pose_xyz(st["object_odom"]["pose"]["pose"]["position"])
                    ee_p = pose_xyz(st["ee_pose"]["pose"]["position"]); ee_q = pose_q(st["ee_pose"]["pose"]["orientation"])
                    R_wb = quat_to_R(rov_q)
                    ee_body = R_EE2ROV @ ee_p + T_EE2ROV
                    R_ee_world = R_wb @ R_EE2ROV @ quat_to_R(ee_q)
                    off_ours = R_wb @ ee_body + rov_p - obj_p
                    frames.append((t, off_ours, R_ee_world))
                if not frames:
                    continue
                ft = np.array([f[0] for f in frames])
                used = 0
                for tj, dx, dy, dz in judge:
                    i = int(np.argmin(np.abs(ft - tj)))
                    if abs(ft[i] - tj) > 0.15:
                        continue
                    A.append(frames[i][2]); B.append(np.array([dx, dy, dz]) - frames[i][1]); used += 1
                n_ep += 1
                print(f"  {ep.parent.parent.name}/{ep.parent.name}/{ep.name}: {used} aligned rows", flush=True)
    if not A:
        print("no aligned data"); return
    A = np.concatenate(A, 0); B = np.concatenate(B, 0)             # (3N,3), (3N,)
    delta, *_ = np.linalg.lstsq(A, B, rcond=None)
    resid = B - A @ delta
    resid_before = B
    out = {"delta_ee_m": delta.round(4).tolist(), "n_rows": int(len(B) // 3), "n_episodes": n_ep,
           "rms_before_cm": float(np.sqrt((resid_before ** 2).mean()) * 100),
           "rms_after_cm": float(np.sqrt((resid ** 2).mean()) * 100),
           "note": "off_judge = off_ours + R_ee_world @ delta_ee; apply as ee_body += R_ee_body @ delta_ee"}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
