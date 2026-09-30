#!/usr/bin/env python3
"""Shadow replay of the navigation perception head over a recorded eval episode.

Feeds the recorded right (ego) + hand (wrist) frames through the exact deployment path
(PerceptGoal.predict_nav) and compares the predicted body-frame goal with the waypoint the
vehicle was actually tracking (reference path + recorded odometry, 0.9 m advance radius, the
WAM follower's rule). Reports bearing / range error by range bucket -- on the deployed arm's own
trajectory distribution, i.e. including the covariate shift away from the USIM expert.

  python3 diag_percept_nav_replay.py --episode <eval_runs/<arm>_<cond>/<task>/episode0> [--stride 3]
"""
from __future__ import annotations

import argparse
import glob
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/hy-tmp/underwater_wam")
from percept.goal_head import PerceptGoal  # noqa: E402


def quat_yaw(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def load_img(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB").resize((320, 240), Image.BILINEAR), np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--advance", type=float, default=0.9)
    args = ap.parse_args()
    ep = Path(args.episode)
    task_dir = ep.parent
    m = re.search(r"episode(\d+)", ep.name)
    ei = int(m.group(1)) if m else 0
    traj = np.load(task_dir / "logs" / f"episode_{ei}_traj.npy", allow_pickle=True).astype(float)
    pkls = sorted(glob.glob(str(ep / "*.pkl")), key=lambda p: int(Path(p).stem))
    pg = PerceptGoal()
    assert pg.nav_head is not None, "nav head missing"
    wp_idx = 0
    rows = []
    poses = []
    for k, pk in enumerate(pkls):
        if k % args.stride:
            continue
        d = pickle.load(open(pk, "rb"))
        st = d["observation"]["state"]
        od = st["odom"]["pose"]["pose"]
        pos = np.array([od["position"]["x"], od["position"]["y"], od["position"]["z"]])
        yaw = quat_yaw([od["orientation"][c] for c in "xyzw"])
        instr = d.get("instruction", "")
        # waypoint the follower would be tracking
        while wp_idx < len(traj) - 1 and np.linalg.norm(traj[wp_idx, :3] - pos) <= args.advance:
            wp_idx += 1
        dw = traj[wp_idx, :3] - pos
        cy, sy = np.cos(yaw), np.sin(yaw)
        gt = np.array([cy * dw[0] + sy * dw[1], -sy * dw[0] + cy * dw[1], dw[2]])
        ego = load_img(ep / "images" / "right" / f"{Path(pk).stem}.jpg")
        hand_p = ep / "images" / "hand" / f"{Path(pk).stem}.jpg"
        wrist = load_img(hand_p if hand_p.exists() else ep / "images" / "left" / f"{Path(pk).stem}.jpg")
        pr = float(st["pressure"]["fluid_pressure"]) / 1e4 if st.get("pressure") else 0.0
        alt = float(st["dvl"]["altitude"]) if st.get("dvl") else 0.0
        g, raw = pg.predict_nav(ego, wrist, np.zeros(5, np.float32), pr, alt, instr)
        rows.append((np.linalg.norm(gt[:2]), gt, g[:3], raw[:3]))
        poses.append((pos.copy(), yaw, traj[wp_idx, :3].copy()))
    rng = np.array([r[0] for r in rows])
    gt = np.array([r[1] for r in rows])
    pr_ = np.array([r[2] for r in rows])
    b = np.degrees(np.abs(np.arctan2(pr_[:, 1], pr_[:, 0]) - np.arctan2(gt[:, 1], gt[:, 0])))
    b = np.minimum(b, 360 - b)
    rerr = np.abs(np.linalg.norm(pr_[:, :2], axis=1) - rng)
    print(f"{ep}  frames={len(rows)}  waypoints={len(traj)}")
    for lo, hi in ((0, 0.5), (0.5, 2), (2, 4), (4, 99)):
        s = (rng >= lo) & (rng < hi)
        if s.sum() < 3:
            continue
        print(f"  range[{lo},{hi}) n={s.sum():4d}  bearing med {np.median(b[s]):5.1f} deg  p75 {np.percentile(b[s], 75):5.1f}"
              f"  | range err med {np.median(rerr[s]):.2f} m  | dz err med {np.median(np.abs(pr_[s, 2] - gt[s, 2])):.2f}")
    print(f"  ALL bearing med {np.median(b):.1f} deg, frac < 20 deg: {(b < 20).mean():.2f}, "
          f"frac < 45 deg: {(b < 45).mean():.2f}")

    # --- visual-inertial goal filter: fuse the RAW per-frame predictions over the recorded poses
    # and compare the fused world-frame goal with the tracked node ---
    from uwam.goal_filter import GoalFilter  # noqa: PLC0415
    gf = GoalFilter()
    fused_err, single_err, fused_b = [], [], []
    for (rng_i, gt_i, _, raw_i), (pos_i, yaw_i, node_i) in zip(rows, poses):
        x = gf.update(pos_i, yaw_i, raw_i)
        fused_err.append(np.linalg.norm((x - node_i)[:2]))
        cy, sy = np.cos(yaw_i), np.sin(yaw_i)
        o_w = pos_i + np.array([cy * raw_i[0] - sy * raw_i[1], sy * raw_i[0] + cy * raw_i[1], raw_i[2]])
        single_err.append(np.linalg.norm((o_w - node_i)[:2]))
        dwf = x - pos_i
        bb = np.degrees(abs(np.arctan2(dwf[1], dwf[0]) - np.arctan2((node_i - pos_i)[1], (node_i - pos_i)[0])))
        fused_b.append(min(bb, 360 - bb))
    fe, se, fb = np.array(fused_err), np.array(single_err), np.array(fused_b)
    for lo, hi in ((0, 2), (2, 4), (4, 99)):
        s = (rng >= lo) & (rng < hi)
        if s.sum() < 3:
            continue
        print(f"  FUSED range[{lo},{hi}) goal pos err med single {np.median(se[s]):.2f} m -> fused {np.median(fe[s]):.2f} m"
              f"  | bearing med single {np.median(b[s]):.1f} -> fused {np.median(fb[s]):.1f} deg")


if __name__ == "__main__":
    main()
