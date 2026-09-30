#!/usr/bin/env python3
"""Pack USIM locomotion episodes for end-to-end perception training.

Per frame: 224x224 JPEG pair (ego, wrist) decoded from the AV1 videos, and labels
  disp3   the expert's displacement over the next HORIZON seconds in the CURRENT body frame,
          integrated from the recorded DVL body velocity with the gyro-z heading change (cm-level over
          3 s) -- a smooth, always-defined "intent" target that also encodes obstacle avoidance
  wp6     USIM's own target_pos (expert's current node in the body frame), auxiliary
  state7  joints(5), pressure, altitude ; task index ; valid flag (target published)
Output: <out>/<split>_e2e.npz (object arrays of JPEG bytes + label arrays).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

NAV_TASKS = {0, 3, 4, 5, 8}
HORIZON_S = 3.0
DT = 0.1
DVL, IMU_AV, PRESSURE, DVL_H = slice(18, 21), slice(21, 24), 27, 28


def decode(path: Path, stride: int):
    import av
    c = av.open(str(path))
    out = []
    for i, f in enumerate(c.decode(video=0)):
        if i % stride == 0:
            im = f.to_ndarray(format="bgr24")
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            out.append(buf.tobytes() if ok else None)
    c.close()
    return out


def future_disp(st: np.ndarray) -> np.ndarray:
    """Body-frame displacement over the next HORIZON_S for every frame (zero-padded at the end)."""
    T = len(st)
    H = int(round(HORIZON_S / DT))
    v = st[:, DVL].astype(np.float64)                      # body velocity
    wz = st[:, IMU_AV][:, 2].astype(np.float64)            # yaw rate
    psi = np.concatenate([[0.0], np.cumsum(wz[:-1] * DT)])  # heading relative to frame 0
    # world-ish increments (relative heading frame), then re-express in the frame at t
    cy, sy = np.cos(psi), np.sin(psi)
    dx_w = (cy * v[:, 0] - sy * v[:, 1]) * DT
    dy_w = (sy * v[:, 0] + cy * v[:, 1]) * DT
    dz = v[:, 2] * DT
    cx, cyy, cz = np.cumsum(dx_w), np.cumsum(dy_w), np.cumsum(dz)
    out = np.zeros((T, 3), np.float32)
    for t in range(T):
        t2 = min(T - 1, t + H)
        Dx, Dy, Dz = cx[t2] - cx[t], cyy[t2] - cyy[t], cz[t2] - cz[t]
        c, s = np.cos(psi[t]), np.sin(psi[t])
        out[t] = [c * Dx + s * Dy, -s * Dx + c * Dy, Dz]
    return out


def process(args):
    parquet, root, split, stride = args
    import pyarrow.parquet as pq
    p = Path(parquet)
    t = pq.read_table(p)
    ti = int(t.column("task_index")[0].as_py())
    if ti not in NAV_TASKS:
        return []
    st = np.array([np.asarray(x, np.float32) for x in t.column("observation.state").to_pylist()])
    tp = np.array([np.asarray(x, np.float32) for x in t.column("target_pos").to_pylist()])
    ep = int(p.stem.split("_")[1])
    chunk = p.parent.name
    vids = {}
    for cam in ("ego", "wrist"):
        vp = Path(root) / split / "videos" / chunk / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
        if not vp.exists():
            return []
        vids[cam] = decode(vp, stride)
    disp = future_disp(st)
    k = min(len(vids["ego"]), len(vids["wrist"]), len(st[::stride]))
    idx = np.arange(0, len(st), stride)[:k]
    rows = []
    for j, i in enumerate(idx):
        je, jw = vids["ego"][j], vids["wrist"][j]
        if je is None or jw is None:
            continue
        valid = bool(np.any(tp[i] != 0))
        s7 = np.concatenate([st[i, 0:5], [st[i, PRESSURE]], [st[i, DVL_H]]]).astype(np.float32)
        rows.append((je, jw, disp[i], tp[i].astype(np.float32), s7, ti, valid, ep))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="/hy-tmp/data/percept_e2e")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    root = Path(args.usim)
    parquets = sorted((root / args.split / "data").glob("chunk-*/episode_*.parquet"))
    print(f"{args.split}: {len(parquets)} episodes (locomotion only kept)", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process, [(str(p), str(root), args.split, args.stride) for p in parquets])):
            rows += r
            if i % 200 == 0:
                print(f"  {i}/{len(parquets)} episodes, {len(rows)} frames", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = Path(args.out) / f"{args.split}_e2e.npz"
    np.savez(out,
             ego=np.array([r[0] for r in rows], dtype=object), wrist=np.array([r[1] for r in rows], dtype=object),
             disp3=np.stack([r[2] for r in rows]), wp6=np.stack([r[3] for r in rows]),
             state=np.stack([r[4] for r in rows]), task=np.array([r[5] for r in rows], np.int64),
             valid=np.array([r[6] for r in rows], bool), episode=np.array([r[7] for r in rows], np.int64))
    d = np.stack([r[2] for r in rows])
    print(f"saved {len(rows)} frames -> {out}; |disp3| median {np.median(np.linalg.norm(d[:, :2], axis=1)):.2f} m", flush=True)


if __name__ == "__main__":
    main()
