#!/usr/bin/env python3
"""Synthesize expert-staged target labels for wander (DAgger) episodes and
featurize them into the percept training schema.

Per recorded frame (vla_data_collector pkl): read privileged object + vehicle
odometry, compute the staged target the expert WOULD publish from that pose
(search / approach / grasp standoffs, the same constants the privileged arm
uses), express it in the vehicle body frame, and pair it with the DINOv2-base
features of the recorded images. Gripper label = 1 inside the close window.

Output: appends dagger_feats_base.npz + dagger_grip_base.npy under the probe
data dir; train_head consumes them via --extra dagger.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_feats import ENCODERS, featurize  # noqa: E402

GRASP_STANDOFF = {
    "search": np.array([0.45, -0.08, 0.35]),
    "approach": np.array([0.32, -0.08, 0.27]),
    "grasp": np.array([0.30, -0.08, 0.20]),
}
TASK_IDX = {"Pick up the pipe": 1, "Pick up the blue cylinder": 2,
            "Pick up the red cylinder": 6,
            "Pick up the red cylinder and transfer it to the box": 7}


def yaw_wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def quat_to_yaw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def standoff_world(box, box_yaw, key):
    off = GRASP_STANDOFF[key]
    cy, sy = np.cos(box_yaw), np.sin(box_yaw)
    world = np.array([cy * off[0] - sy * off[1], sy * off[0] + cy * off[1], off[2]])
    return box - world


def staged_label(pos, yaw, box, box_yaw):
    """Body-frame [dx dy dz r p yaw] the expert would command from this pose."""
    cands = [box_yaw, yaw_wrap(box_yaw + np.pi)]
    byaw = min(cands, key=lambda a: abs(yaw_wrap(a - yaw)))
    d_grasp = standoff_world(box, byaw, "grasp") - pos
    d_appr = standoff_world(box, byaw, "approach") - pos
    if np.linalg.norm(d_grasp) < 0.35:
        d_world, stage = d_grasp, "grasp"
    elif np.linalg.norm(d_appr) < 0.6:
        d_world, stage = d_appr, "approach"
    else:
        d_world, stage = standoff_world(box, byaw, "search") - pos, "search"
    cy, sy = np.cos(yaw), np.sin(yaw)
    body = np.array([cy * d_world[0] + sy * d_world[1],
                     -sy * d_world[0] + cy * d_world[1],
                     d_world[2]])
    grip = 1.0 if (stage == "grasp" and np.linalg.norm(d_grasp) < 0.06) else 0.0
    return np.array([body[0], body[1], body[2], 0.0, 0.0, yaw_wrap(byaw - yaw)],
                    np.float32), grip


def main():
    import torch
    from transformers import AutoModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--glob", default="dagger_full/pick_*/episode*")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="/hy-tmp/data/percept_probe")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(ENCODERS["base"]).to(device).half().eval()

    eps = sorted(Path(args.episodes_root).glob(args.glob))
    print(f"{len(eps)} wander episodes", flush=True)
    ego_f, wrist_f, targets, grips, states, tasks = [], [], [], [], [], []
    n_frames = 0
    for ep in eps:
        pkls = sorted((q for q in ep.glob("*.pkl") if q.stem.isdigit()), key=lambda q: int(q.stem))[:: args.stride]
        egos, wrists, labs, grip_l, st_l, ti = [], [], [], [], [], 1
        for p in pkls:
            try:
                d = pickle.load(open(p, "rb"))
            except Exception:
                continue
            st = d["observation"]["state"]
            odom = st.get("odom")
            obj = st.get("object_odom")
            js = st.get("joint_states")
            if odom is None or obj is None:
                continue
            pp = odom["pose"]["pose"]["position"]
            qq = odom["pose"]["pose"]["orientation"]
            pos = np.array([pp["x"], pp["y"], pp["z"]])
            yaw = quat_to_yaw([qq["x"], qq["y"], qq["z"], qq["w"]])
            op = obj["pose"]["pose"]["position"]
            oq = obj["pose"]["pose"]["orientation"]
            box = np.array([op["x"], op["y"], op["z"]])
            box_yaw = quat_to_yaw([oq["x"], oq["y"], oq["z"], oq["w"]])
            img_r = ep / "images" / "right" / f"{p.stem}.jpg"
            img_h = ep / "images" / "hand" / f"{p.stem}.jpg"
            if not img_r.exists() or not img_h.exists():
                continue
            e = cv2.cvtColor(cv2.imread(str(img_r)), cv2.COLOR_BGR2RGB)
            h = cv2.cvtColor(cv2.imread(str(img_h)), cv2.COLOR_BGR2RGB)
            e = cv2.resize(e, (320, 240), interpolation=cv2.INTER_LINEAR)
            h = cv2.resize(h, (320, 240), interpolation=cv2.INTER_LINEAR)
            lab, grip = staged_label(pos, yaw, box, box_yaw)
            jp = np.asarray(js["position"][:5], np.float32) if js else np.zeros(5, np.float32)
            pr = (st.get("pressure") or {}).get("fluid_pressure", 0.0) / 1e4
            al = (st.get("dvl") or {}).get("altitude", 0.0)
            instr = d.get("instruction") or "Pick up the pipe"
            ti = TASK_IDX.get(instr, 1)
            egos.append(e); wrists.append(h); labs.append(lab); grip_l.append(grip)
            st_l.append(np.concatenate([jp, [np.float32(pr)], [np.float32(al)]]))
        if not egos:
            continue
        ego_f.append(featurize(model, np.stack(egos), device))
        wrist_f.append(featurize(model, np.stack(wrists), device))
        targets.append(np.stack(labs)); grips.append(np.asarray(grip_l, np.float32))
        states.append(np.stack(st_l).astype(np.float32))
        tasks.append(np.full(len(labs), ti, np.int64))
        n_frames += len(labs)
        print(f"  {ep.parent.name}/{ep.name}: {len(labs)} frames "
              f"(grip-rate {np.mean(grip_l):.3f})", flush=True)

    out = Path(args.out)
    np.savez_compressed(
        out / "dagger_feats_base.npz",
        ego=np.concatenate(ego_f), wrist=np.concatenate(wrist_f),
        target=np.concatenate(targets), state=np.concatenate(states),
        task=np.concatenate(tasks),
        episode=np.zeros(n_frames, np.int64),
        valid=np.ones(n_frames, bool),
    )
    np.save(out / "dagger_grip_base.npy", np.concatenate(grips))
    print(f"saved dagger set: {n_frames} frames", flush=True)


if __name__ == "__main__":
    main()
