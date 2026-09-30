#!/usr/bin/env python3
"""
Pack grasp/transfer recordings (harness pkl frames + JPEGs) into the training set of the
wrist-camera relative-pose head and the contact (grasp-success) model.

Every recorded frame -- successful episode or not -- is a labelled sample: the simulator's
privileged object odometry gives the object pose, the arm's end-effector pose and the vehicle
odometry give the gripper pose, so the label "object relative to the gripper" is exact.

Per frame (stride s):
  wrist_jpg   224x224 JPEG bytes of the hand camera        ego_jpg  224x224 of the right camera
  obj_ee      object position in the END-EFFECTOR frame [m] (what the wrist camera sees)
  obj_body    object position in the vehicle body frame [m]
  ee_body     end-effector position in the body frame [m] (from /alpha_ee_pose + the judge's
              fixed arm-base -> body transform)     ee_R  its rotation matrix in the body frame (9)
  off_world   gripper - object offset in WORLD axes [dx, dy, dz] (the judge's quantity)
  rel_yaw     object yaw - vehicle yaw [rad]              dist    |gripper - object| [m]
  joints(5)   joint positions          effort(5)          joint efforts (axis_a = gripper)
  grip_cmd    commanded gripper joint (action)             t       seconds since episode start
  task        task code                episode / success  outcome of the whole episode
Output: <out>.npz (object arrays for JPEG bytes).  Usage: see --help.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# arm base -> vehicle body (planner/eval_grasping.py calc_ee_pose_world): arm x -> body z,
# arm y -> -body y, arm z -> body x; gripper base offset from the vehicle centre
R_EE2ROV = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
T_EE2ROV = np.array([0.196, -0.084, 0.145 + 0.05])


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_yaw(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def jpeg224(path: Path):
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None


def pose_xyz(p):
    return np.array([p["x"], p["y"], p["z"]], np.float64)


def pose_q(o):
    return np.array([o["x"], o["y"], o["z"], o["w"]], np.float64)


def pack_episode(args):
    ep_dir, stride, task, success, with_images = args
    ep_dir = Path(ep_dir)
    pkls = sorted((q for q in ep_dir.glob("*.pkl") if q.stem.isdigit()), key=lambda q: int(q.stem))
    rows = []
    for p in pkls[::stride]:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        st = d["observation"]["state"]
        im = d["observation"]["image"]
        odom, obj, ee, js = st.get("odom"), st.get("object_odom"), st.get("ee_pose"), st.get("joint_states")
        if odom is None or obj is None or ee is None or js is None:
            continue
        rov_p = pose_xyz(odom["pose"]["pose"]["position"]); rov_q = pose_q(odom["pose"]["pose"]["orientation"])
        obj_p = pose_xyz(obj["pose"]["pose"]["position"]); obj_q = pose_q(obj["pose"]["pose"]["orientation"])
        ee_p = pose_xyz(ee["pose"]["position"]); ee_q = pose_q(ee["pose"]["orientation"])
        R_wb = quat_to_R(rov_q)
        ee_body = R_EE2ROV @ ee_p + T_EE2ROV
        R_ee_body = R_EE2ROV @ quat_to_R(ee_q)           # end-effector orientation in the body frame
        obj_body = R_wb.T @ (obj_p - rov_p)
        obj_ee = R_ee_body.T @ (obj_body - ee_body)
        ee_world = R_wb @ ee_body + rov_p
        off_world = ee_world - obj_p
        wrist = im.get("hand_image"); ego = im.get("right_image")
        if with_images:
            wj = jpeg224(ep_dir / "images" / wrist) if wrist else None
            ej = jpeg224(ep_dir / "images" / ego) if ego else None
            if wj is None:
                continue
        else:
            wj, ej = b"", b""
        act = d.get("action") or {}
        grip_cmd = np.nan
        dj = act.get("desired_joint_state")
        if isinstance(dj, dict) and dj.get("position"):
            names = dj.get("name", [])
            for n_, v in zip(names, dj["position"]):
                if str(n_).endswith("axis_a"):
                    grip_cmd = float(v)
        rows.append({
            "wrist_jpg": wj, "ego_jpg": ej if ej is not None else b"",
            "obj_ee": obj_ee.astype(np.float32), "obj_body": obj_body.astype(np.float32),
            "ee_body": ee_body.astype(np.float32), "ee_R": R_ee_body.astype(np.float32).reshape(9),
            "off_world": off_world.astype(np.float32),
            "rel_yaw": np.float32(np.arctan2(np.sin(quat_to_yaw(obj_q) - quat_to_yaw(rov_q)),
                                             np.cos(quat_to_yaw(obj_q) - quat_to_yaw(rov_q)))),
            "dist": np.float32(np.linalg.norm(off_world)),
            "joints": np.asarray(js.get("position", [0] * 5)[:5], np.float32),
            "effort": np.asarray(js.get("effort", [0] * 5)[:5], np.float32),
            "grip_cmd": np.float32(grip_cmd),
            "t": np.float32(int(p.stem) / 10.0),   # frames are recorded at 10 Hz; N.pkl = tick N
            "task": task, "episode": str(ep_dir), "success": np.int8(success),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="/hy-tmp/u0env/dataset/eval_runs,/hy-tmp/u0env_b/dataset/eval_runs")
    ap.add_argument("--glob", default="*/pick_*/episode*,*/transfer_*/episode*",
                    help="comma-separated globs under each root (arm_cond/task/episodeN)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="/hy-tmp/data/grasp/wrist_pose.npz")
    ap.add_argument("--no-images", action="store_true", help="labels only (kinematics / dynamics packs)")
    args = ap.parse_args()

    jobs = []
    for root in args.roots.split(","):
        root = Path(root)
        if not root.exists():
            continue
        for g in args.glob.split(","):
            for ep in sorted(root.glob(g.strip())):
                if not ep.is_dir() or not re.match(r"episode\d+$", ep.name):
                    continue
                task = ep.parent.name
                res = ep.parent / "results.csv"
                success = 0
                if res.exists():
                    n = int(ep.name[len("episode"):])
                    for r in csv.reader(open(res)):
                        if r and r[0].isdigit() and int(r[0]) == n:
                            success = int(r[1].strip() == "success")
                jobs.append((str(ep), args.stride, task, success, not args.no_images))
    print(f"{len(jobs)} episodes", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, out in enumerate(ex.map(pack_episode, jobs, chunksize=1)):
            rows += out
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes, {len(rows)} frames", flush=True)
    if not rows:
        print("no frames"); return
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        if k in ("wrist_jpg", "ego_jpg", "task", "episode"):
            out[k] = np.array(vals, dtype=object)
        else:
            out[k] = np.stack(vals) if np.ndim(vals[0]) else np.asarray(vals)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    d = out["dist"]; near = d < 0.4
    print(f"saved {len(rows)} frames -> {args.out}; {near.sum()} within 0.4 m; success frames {out['success'].mean():.2f}; "
          f"|obj_ee| median {np.median(np.linalg.norm(out['obj_ee'], axis=1)):.3f} m", flush=True)


if __name__ == "__main__":
    main()
