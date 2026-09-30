#!/usr/bin/env python3
"""Pack recorded locomotion episodes (box 2 recordings) into a compact training set for the
end-to-end perception model: 224x224 JPEG pairs (ego = right camera, wrist = hand or left camera)
plus labels per frame:
  disp3   expert-intent displacement over the next 3 s in the body frame = direction to the node the
          follower is tracking, times min(distance, 3 s x cruise speed)  (follow: boat standoff)
  wp6     the tracked node in the body frame [dx dy dz 0 0 yaw_rel]     (same as synth_nav_labels)
  state7  [joints(5)=0, pressure/1e4, altitude], task index
Output: <out>/dagger_e2e.npz (jpeg bytes as object arrays + label arrays). Runs on CPU only.
"""
from __future__ import annotations

import argparse
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

TASK_IDX = {"charge station": 0, "scan the ship": 3, "inspect the pipeline": 4,
            "follow the boat": 5, "water tower": 8}
PASS_R = 0.9
CRUISE = 0.45


def yaw_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def task_index(instr):
    low = instr.lower()
    for k, v in TASK_IDX.items():
        if k in low:
            return v
    return None


def jpeg(path):
    im = cv2.imread(str(path))
    if im is None:
        return None
    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None


def process_episode(args):
    ep, stride = args
    ep = Path(ep)
    m = re.search(r"episode(\d+)", ep.name)
    tf = ep.parent / "logs" / f"episode_{m.group(1)}_traj.npy"
    traj = np.asarray(np.load(tf, allow_pickle=True), np.float64) if tf.exists() else None
    pkls = sorted((q for q in ep.glob("*.pkl") if q.stem.isdigit()), key=lambda q: int(q.stem))[::stride]
    out = []
    k = 0
    for p in pkls:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        ti = task_index(d.get("instruction") or "")
        if ti is None:
            return []
        st = d["observation"]["state"]
        odom = st.get("odom")
        if odom is None:
            continue
        pp = odom["pose"]["pose"]["position"]
        qq = odom["pose"]["pose"]["orientation"]
        pos = np.array([pp["x"], pp["y"], pp["z"]])
        yaw = quat_to_yaw([qq["x"], qq["y"], qq["z"], qq["w"]])
        if ti == 5:
            boat = st.get("object_odom")
            if boat is None:
                continue
            bp = boat["pose"]["pose"]["position"]
            bq = boat["pose"]["pose"]["orientation"]
            byaw = quat_to_yaw([bq["x"], bq["y"], bq["z"], bq["w"]])
            tgt = np.array([bp["x"] - 3.0 * np.cos(byaw), bp["y"] - 3.0 * np.sin(byaw), bp["z"] + 0.5])
            tyaw, lookat = byaw, False
        else:
            if traj is None or len(traj) == 0:
                return []
            while k < len(traj) - 1:
                node = traj[k, :3]
                if np.linalg.norm(node - pos) <= PASS_R:
                    k += 1
                    continue
                seg = traj[k + 1, :3] - node
                if np.dot(pos - node, seg) > 0 and np.linalg.norm(traj[k + 1, :3] - pos) < np.linalg.norm(node - pos):
                    k += 1
                    continue
                break
            tgt, tyaw, lookat = traj[k, :3], float(traj[k, 3]), ti in (3, 4)
        dw = tgt - pos
        cy, sy = np.cos(yaw), np.sin(yaw)
        b = np.array([cy * dw[0] + sy * dw[1], -sy * dw[0] + cy * dw[1], dw[2]])
        rel_yaw = yaw_wrap(tyaw - yaw) if lookat else float(np.arctan2(b[1], b[0]))
        wp6 = np.array([b[0], b[1], b[2], 0, 0, rel_yaw], np.float32)
        nb = float(np.linalg.norm(b))
        disp3 = (b / max(nb, 1e-6) * min(nb, 3.0 * CRUISE)).astype(np.float32)
        img_r = ep / "images" / "right" / f"{p.stem}.jpg"
        img_h = ep / "images" / "hand" / f"{p.stem}.jpg"
        if not img_h.exists():
            img_h = ep / "images" / "left" / f"{p.stem}.jpg"
        if not img_r.exists() or not img_h.exists():
            continue
        je, jw = jpeg(img_r), jpeg(img_h)
        if je is None or jw is None:
            continue
        pr = (st.get("pressure") or {}).get("fluid_pressure", 0.0) / 1e4
        al = (st.get("dvl") or {}).get("altitude", 0.0)
        out.append((je, jw, disp3, wp6, np.array([0, 0, 0, 0, 0, pr, al], np.float32), ti))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--glob", default="wam_percept_*/*/episode*")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="/hy-tmp/data/percept_e2e/dagger_e2e.npz")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    eps = [e for e in sorted(Path(args.episodes_root).glob(args.glob))
           if e.is_dir() and re.search(r"episode\d+$", e.name) and (e / "images").exists()]
    print(f"{len(eps)} episodes", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process_episode, [(str(e), args.stride) for e in eps])):
            rows += r
            if i % 40 == 0:
                print(f"  {i}/{len(eps)} episodes, {len(rows)} frames", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             ego=np.array([r[0] for r in rows], dtype=object), wrist=np.array([r[1] for r in rows], dtype=object),
             disp3=np.stack([r[2] for r in rows]), wp6=np.stack([r[3] for r in rows]),
             state=np.stack([r[4] for r in rows]), task=np.array([r[5] for r in rows], np.int64))
    print(f"saved {len(rows)} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
