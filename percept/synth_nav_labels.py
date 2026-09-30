#!/usr/bin/env python3
"""DAgger set for the NAVIGATION head: label recorded locomotion episodes with the waypoint the
expert would be tracking and featurize them into the percept training schema.

Per recorded frame (vla_data_collector pkl + images/right + images/hand|left): read the vehicle
odometry, pick the reference-path node the expert planner would currently target (nodes are
visited in order; a node counts as passed once the vehicle is within PASS_R of it or has moved
beyond it along the path), express that node in the vehicle body frame and pair it with the
DINOv2-base features of the recorded ego / wrist images. Follow-the-boat frames use the expert's
standoff (3 m behind the boat along its heading, 0.5 m deeper) from the recorded boat odometry.

Label = [dx dy dz roll pitch yaw_rel] like USIM's target_pos: yaw_rel is the waypoint's look-at yaw
for scan / inspect and the bearing to the node for goto / follow.

Output: <out>/dagger_nav_feats_base.npz (train_head.py --extra dagger_nav).
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_feats import ENCODERS, featurize  # noqa: E402

TASK_IDX = {"charge station": 0, "scan the ship": 3, "inspect the pipeline": 4,
            "follow the boat": 5, "water tower": 8}
PASS_R = 0.9   # follower's advance radius


def yaw_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_yaw(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def task_index(instr: str) -> int | None:
    low = instr.lower()
    for k, v in TASK_IDX.items():
        if k in low:
            return v
    return None


def body_label(pos, yaw, tgt_xyz, tgt_yaw, use_lookat):
    d = np.asarray(tgt_xyz, np.float64) - pos
    cy, sy = np.cos(yaw), np.sin(yaw)
    b = np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1], d[2]])
    rel_yaw = yaw_wrap(tgt_yaw - yaw) if use_lookat else float(np.arctan2(b[1], b[0]))
    return np.array([b[0], b[1], b[2], 0.0, 0.0, rel_yaw], np.float32)


def main():
    import torch
    from transformers import AutoModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--glob", default="wam_percept_*/*/episode*")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="/hy-tmp/data/percept_probe")
    ap.add_argument("--name", default="dagger_nav")
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(ENCODERS["base"]).to(device).half().eval()

    eps = sorted(Path(args.episodes_root).glob(args.glob))
    eps = [e for e in eps if e.is_dir() and re.search(r"episode\d+$", e.name) and (e / "images").exists()]
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    print(f"{len(eps)} recorded locomotion episodes", flush=True)
    ego_f, wrist_f, targets, states, tasks = [], [], [], [], []
    n_frames = 0
    for ep in eps:
        task_dir = ep.parent
        m = re.search(r"episode(\d+)", ep.name)
        tf = task_dir / "logs" / f"episode_{m.group(1)}_traj.npy"
        traj = np.asarray(np.load(tf, allow_pickle=True), np.float64) if tf.exists() else None
        pkls = sorted((q for q in ep.glob("*.pkl") if q.stem.isdigit()), key=lambda q: int(q.stem))[:: args.stride]
        egos, wrists, labs, st_l, ti = [], [], [], [], None
        k = 0
        for p in pkls:
            try:
                d = pickle.load(open(p, "rb"))
            except Exception:
                continue
            instr = d.get("instruction") or ""
            ti = task_index(instr)
            if ti is None:
                break
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
                lab = body_label(pos, yaw, tgt, byaw, use_lookat=False)
            else:
                if traj is None or len(traj) == 0:
                    break
                # advance the tracked node: passed if within PASS_R, or moved beyond it along the path
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
                lab = body_label(pos, yaw, traj[k, :3], float(traj[k, 3]), use_lookat=(ti in (3, 4)))
            img_r = ep / "images" / "right" / f"{p.stem}.jpg"
            img_h = ep / "images" / "hand" / f"{p.stem}.jpg"
            if not img_h.exists():
                img_h = ep / "images" / "left" / f"{p.stem}.jpg"
            if not img_r.exists() or not img_h.exists():
                continue
            e = cv2.cvtColor(cv2.imread(str(img_r)), cv2.COLOR_BGR2RGB)
            h = cv2.cvtColor(cv2.imread(str(img_h)), cv2.COLOR_BGR2RGB)
            e = cv2.resize(e, (320, 240), interpolation=cv2.INTER_LINEAR)
            h = cv2.resize(h, (320, 240), interpolation=cv2.INTER_LINEAR)
            pr = (st.get("pressure") or {}).get("fluid_pressure", 0.0) / 1e4
            al = (st.get("dvl") or {}).get("altitude", 0.0)
            egos.append(e); wrists.append(h); labs.append(lab)
            st_l.append(np.concatenate([np.zeros(5, np.float32), [np.float32(pr)], [np.float32(al)]]))
        if not egos or ti is None:
            continue
        ego_f.append(featurize(model, np.stack(egos), device))
        wrist_f.append(featurize(model, np.stack(wrists), device))
        targets.append(np.stack(labs)); states.append(np.stack(st_l).astype(np.float32))
        tasks.append(np.full(len(labs), ti, np.int64))
        n_frames += len(labs)
        print(f"  {task_dir.parent.name}/{task_dir.name}/{ep.name}: {len(labs)} frames task={ti}", flush=True)

    if n_frames == 0:
        print("no frames; nothing written")
        return
    out = Path(args.out)
    np.savez_compressed(
        out / f"{args.name}_feats_base.npz",
        ego=np.concatenate(ego_f), wrist=np.concatenate(wrist_f),
        target=np.concatenate(targets), state=np.concatenate(states),
        task=np.concatenate(tasks),
        episode=np.zeros(n_frames, np.int64),
        valid=np.ones(n_frames, bool),
    )
    print(f"saved {args.name} set: {n_frames} frames from {len(targets)} episodes", flush=True)


if __name__ == "__main__":
    main()
