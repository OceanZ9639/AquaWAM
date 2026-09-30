#!/usr/bin/env python3
"""Offline smoke of the percept-goal server path: replay a real USIM test pick
episode's frames through WamPolicy.act() (goal-source=percept) and check that
goals track and the gripper trigger fires. No simulator, no privileged reads.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq  # noqa: E402

from percept.goal_head import _decode  # noqa: E402
from wam_policy_server import WamPolicy  # noqa: E402

args = types.SimpleNamespace(
    ckpt="/hy-tmp/models/uwam/best_scenes.pt",
    vel_ens="/hy-tmp/models/uwam/vel_ens_scenes.pt",
    gate_calib="/hy-tmp/models/uwam/gate_calib.json",
    ou="", eval_root="/tmp/smoke_eval_root",
    hold_anchor="mixer", dump_dir="", goal_source="percept",
)
Path(args.eval_root).mkdir(parents=True, exist_ok=True)
policy = WamPolicy(args)

root = Path("/hy-tmp/data/usim/test")
EP = 34  # "Pick up the pipe", shadow close-MAE 3.4 cm
chunk = f"chunk-{EP // 1000:03d}"
t = pq.read_table(root / "data" / chunk / f"episode_{EP:06d}.parquet")
st = np.array([np.asarray(x, np.float32) for x in t.column("observation.state").to_pylist()])
lab = np.array([np.asarray(x, np.float32) for x in t.column("target_pos").to_pylist()])
ego = _decode(root / "videos" / chunk / "observation.images.ego" / f"episode_{EP:06d}.mp4", 3)
wrist = _decode(root / "videos" / chunk / "observation.images.wrist" / f"episode_{EP:06d}.mp4", 3)
k = min(len(ego), len(wrist), len(st[::3]))
sts = st[::3][:k]
labs = lab[::3][:k]

L = 16
closed_at = None
for i in range(k):
    obs = {
        "video.ego": ego[i][None],
        "video.wrist": wrist[i][None],
        "state.joint_pos": sts[i, 0:5].reshape(1, -1),
        "state.pressure": np.array([[sts[i, 27]]], np.float32),
        "state.dvl_h": np.array([[sts[i, 28]]], np.float32),
        "state.odom_pos": np.zeros((1, 3), np.float32),
        "state.odom_rpy": np.zeros((1, 3), np.float32),
        "state.hist_dvl": np.full((L, 3), 0.01, np.float32),
        "state.hist_imu_av": np.zeros((L, 3), np.float32),
        "state.hist_imu_la": np.zeros((L, 3), np.float32),
        "state.hist_pressure": np.full((L, 1), sts[i, 27], np.float32),
        "state.hist_dvl_h": np.full((L, 1), sts[i, 28], np.float32),
        "state.hist_pwm": np.zeros((L, 8), np.float32),
        "meta.dvl_valid": np.array([[1.0]], np.float32),
        "annotation.human.action.task_description": ["Pick up the pipe"],
    }
    out = policy.act(obs)
    a = out[0]
    assert a["action.pwm"].shape == (16, 8)
    assert a["action.joint_pos"].shape == (16, 5)
    if policy.pg_grasped and closed_at is None:
        closed_at = i
        print(f"gripper closed at replay tick {i}/{k} "
              f"(label |target|={np.linalg.norm(labs[i,:3]):.3f} m)")

assert policy.mode == "grasp", policy.mode
print(json.dumps({
    "episode": EP, "ticks": k,
    "grasped": bool(policy.pg_grasped),
    "closed_at_tick": closed_at,
    "final_joint_cmd": [round(float(x), 4) for x in policy.joint_cmd],
}))
assert policy.pg_grasped, "gripper never closed on a successful expert pick episode"

# --- navigation head: replay a "Go to the water tower" episode; the vision goal must steer
# toward the labelled expert waypoint (bearing) and the sighted + blind paths must both run ---
import pyarrow as pa  # noqa: E402

tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(root / "meta" / "tasks.jsonl")}
nav_ep = None
for p in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
    if int(pq.read_table(p, columns=["task_index"]).column("task_index")[0].as_py()) == 8:
        nav_ep = int(p.stem.split("_")[1])
        break
assert nav_ep is not None
chunk = f"chunk-{nav_ep // 1000:03d}"
t = pq.read_table(root / "data" / chunk / f"episode_{nav_ep:06d}.parquet")
st = np.array([np.asarray(x, np.float32) for x in t.column("observation.state").to_pylist()])
lab = np.array([np.asarray(x, np.float32) for x in t.column("target_pos").to_pylist()])
ego = _decode(root / "videos" / chunk / "observation.images.ego" / f"episode_{nav_ep:06d}.mp4", 3)
wrist = _decode(root / "videos" / chunk / "observation.images.wrist" / f"episode_{nav_ep:06d}.mp4", 3)
k = min(len(ego), len(wrist), len(st[::3]))
sts, labs = st[::3][:k], lab[::3][:k]
policy._reset_episode(None)
bearing_err, n_far = [], 0
for i in range(k):
    blind = i >= k // 2  # second half: DVL dropout, goal must still come from vision
    obs = {
        "video.ego": ego[i][None], "video.wrist": wrist[i][None],
        "state.joint_pos": np.zeros((1, 5), np.float32),
        "state.pressure": np.array([[sts[i, 27]]], np.float32),
        "state.dvl_h": np.array([[sts[i, 28]]], np.float32),
        "state.odom_pos": np.zeros((1, 3), np.float32),
        "state.odom_rpy": np.zeros((1, 3), np.float32),
        "state.imu_rpy": np.zeros((1, 3), np.float32),
        "state.hist_dvl": np.zeros((L, 3), np.float32) if blind else np.full((L, 3), 0.2, np.float32),
        "state.hist_imu_av": np.zeros((L, 3), np.float32),
        "state.hist_imu_la": np.zeros((L, 3), np.float32),
        "state.hist_pressure": np.full((L, 1), sts[i, 27], np.float32),
        "state.hist_dvl_h": np.full((L, 1), sts[i, 28], np.float32),
        "state.hist_pwm": np.zeros((L, 8), np.float32),
        "meta.dvl_valid": np.array([[0.0 if blind else 1.0]], np.float32),
        "annotation.human.action.task_description": ["Go to the water tower"],
    }
    out = policy.act(obs)
    assert out[0]["action.pwm"].shape == (16, 8)
    g = policy.percept.nav_smoothed
    if g is not None and np.linalg.norm(labs[i, :2]) > 1.0 and np.any(labs[i] != 0):
        b = np.degrees(abs(np.arctan2(g[1], g[0]) - np.arctan2(labs[i, 1], labs[i, 0])))
        bearing_err.append(min(b, 360 - b))
        n_far += 1
assert policy.mode == "nav", policy.mode
med = float(np.median(bearing_err)) if bearing_err else float("nan")
print(json.dumps({"nav_episode": nav_ep, "ticks": k, "far_ticks": n_far,
                  "bearing_med_deg": round(med, 1), "blind_alpha": round(float(policy.alpha), 2)}))
assert n_far > 10 and med < 25.0, f"vision nav bearing too poor: {med:.1f} deg"
print("SMOKE_PERCEPT_OK")
