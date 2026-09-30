#!/usr/bin/env python3
"""Pack transfer recordings into a training set for the DESTINATION (container) head: forward (right)
camera frame -> container position in the yaw-rotated vehicle body frame. Emits the same npz layout as
pack_grasp_recordings.py so percept/train_wrist_pose.py trains it unchanged (--no-ego; joints are zeros):
  wrist_jpg = right-camera 224x224 JPEG, obj_ee = container position in the body frame [m],
  rel_yaw = container yaw - vehicle yaw, joints = zeros(5), dist = |container - vehicle|, episode id.
The container pose comes from the harness's episode_<i>_desti.npy (training label only)."""
import argparse, glob, pickle
from pathlib import Path

import cv2
import numpy as np


def quat_to_yaw(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def jpeg224(path):
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="/hy-tmp/u0env/dataset/eval_runs,/hy-tmp/u0env_b/dataset/eval_runs")
    ap.add_argument("--glob", default="*/transfer_*/episode*")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="/hy-tmp/data/grasp/box_pose.npz")
    args = ap.parse_args()
    out = {k: [] for k in ("wrist_jpg", "ego_jpg", "obj_ee", "rel_yaw", "joints", "dist", "episode", "task")}
    n_ep = 0
    for root in args.roots.split(","):
        for ep_dir in sorted(glob.glob(f"{root}/{args.glob}")):
            ep_dir = Path(ep_dir)
            m = ep_dir.name.replace("episode", "")
            desti = ep_dir.parent / "logs" / f"episode_{m}_desti.npy"
            if not desti.exists():
                continue
            dest = np.load(desti, allow_pickle=True).astype(np.float64).reshape(-1)[:4]
            pkls = sorted(ep_dir.glob("*.pkl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
            if len(pkls) < 20:
                continue
            n_ep += 1
            for p in pkls[::args.stride]:
                try:
                    d = pickle.load(open(p, "rb"))
                except Exception:
                    continue
                st = d["observation"]["state"]; im = d["observation"].get("image", {})
                odom = st.get("odom")
                if odom is None:
                    continue
                rp = odom["pose"]["pose"]["position"]; rq = odom["pose"]["pose"]["orientation"]
                rov_p = np.array([rp["x"], rp["y"], rp["z"]]); yaw = quat_to_yaw([rq["x"], rq["y"], rq["z"], rq["w"]])
                dvec = dest[:3] - rov_p
                cy, sy = np.cos(yaw), np.sin(yaw)
                rel = np.array([cy * dvec[0] + sy * dvec[1], -sy * dvec[0] + cy * dvec[1], dvec[2]], np.float32)
                right = im.get("right_image")
                rj = jpeg224(ep_dir / "images" / right) if right else None
                if rj is None:
                    continue
                dyaw = float(np.arctan2(np.sin(dest[3] - yaw), np.cos(dest[3] - yaw)))
                out["wrist_jpg"].append(rj); out["ego_jpg"].append(b"")
                out["obj_ee"].append(rel); out["rel_yaw"].append(np.float32(dyaw)); out["joints"].append(np.zeros(5, np.float32))
                out["dist"].append(np.float32(np.linalg.norm(dvec))); out["episode"].append(n_ep); out["task"].append(str(ep_dir.parent.name))
    n = len(out["obj_ee"])
    print(f"{n_ep} episodes, {n} frames; |container-vehicle| median {np.median(out['dist']):.2f} m, within 1.5 m {np.mean(np.array(out['dist']) < 1.5):.2f}")
    arrs = {}
    for k, v in out.items():
        if k in ("wrist_jpg", "ego_jpg", "task"):
            arrs[k] = np.array(v, dtype=object)
        else:
            arrs[k] = np.asarray(v)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrs)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
