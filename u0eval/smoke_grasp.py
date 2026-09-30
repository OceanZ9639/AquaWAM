#!/usr/bin/env python3
"""Offline smoke test of the WAM server's grasp/transfer path (no sim, no HTTP).

Feeds synthetic observations through WamPolicy.act() and walks the vehicle to each
stage's standoff pose, asserting the stage machine advances and the gripper closes.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wam_policy_server import ARMED_JOINTS, GRASP_STANDOFF, GRIPPER_CLOSE, WamPolicy  # noqa: E402

args = types.SimpleNamespace(
    ckpt="/hy-tmp/models/uwam/best_scenes.pt",
    vel_ens="/hy-tmp/models/uwam/vel_ens_scenes.pt",
    gate_calib="/hy-tmp/models/uwam/gate_calib.json",
    ou="",  # skip OU library for speed
    eval_root="/tmp/smoke_eval_root",  # empty -> no waypoints; grasp path ignores them
    hold_anchor="mixer",
    dump_dir="",
)
Path(args.eval_root).mkdir(parents=True, exist_ok=True)
policy = WamPolicy(args)

BOX = np.array([10.0, 5.0, 3.0])
BOX_YAW = 0.3


# armed-pose end effector in the arm frame, and the hull offset it implies
EE = np.array([0.05, 0.0, 0.30], np.float32)


def gripper_offset(yaw):
    b = np.array([EE[2] + 0.196, -EE[1] - 0.084, EE[0] + 0.195])
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([cy * b[0] - sy * b[1], sy * b[0] + cy * b[1], b[2]])


def obs_at(pos, yaw, task="Pick up the red cylinder", dvl_valid=1.0, ee=True):
    L = 16
    extra = {"state.ee_pose": EE.reshape(1, 3)} if ee else {}
    return {**extra,
        "state.hist_dvl": np.full((L, 3), 0.01, np.float32),
        "state.hist_imu_av": np.zeros((L, 3), np.float32),
        "state.hist_imu_la": np.zeros((L, 3), np.float32),
        "state.hist_pressure": np.full((L, 1), 3.0, np.float32),
        "state.hist_dvl_h": np.full((L, 1), 2.0, np.float32),
        "state.hist_pwm": np.zeros((L, 8), np.float32),
        "state.odom_pos": np.asarray(pos, np.float32).reshape(1, 3),
        "state.odom_rpy": np.array([[0.0, 0.0, yaw]], np.float32),
        "state.box_pos": BOX.reshape(1, 3).astype(np.float32),
        "state.box_rpy": np.array([[0.0, 0.0, BOX_YAW]], np.float32),
        "meta.dvl_valid": np.array([[dvl_valid]], np.float32),
        "annotation.human.action.task_description": [task],
    }


def standoff(key):
    off = GRASP_STANDOFF[key]
    cy, sy = np.cos(BOX_YAW), np.sin(BOX_YAW)
    world = np.array([cy * off[0] - sy * off[1], sy * off[0] + cy * off[1], off[2]])
    return BOX - world


def act(pos, yaw=BOX_YAW, **kw):
    out = policy.act(obs_at(pos, yaw, **kw))
    a = out[0]
    assert a["action.pwm"].shape == (16, 8), a["action.pwm"].shape
    assert a["action.joint_pos"].shape == (16, 5), a["action.joint_pos"].shape
    return a


# 1. mode routing
a = act(standoff("search") + np.array([2.0, 0, 0]))
assert policy.mode == "grasp", policy.mode
assert np.allclose(a["action.joint_pos"][0], ARMED_JOINTS, atol=0.2), a["action.joint_pos"][0]
print(f"1. grasp mode, armed joints OK; stage={policy.grasp_stage}")

# 2. stage transitions. search -> approach is gated on the HULL standoff;
#    approach -> grasp is gated on the GRIPPER sitting at the hover point
#    (6 cm above the object), since the approach stage servos the gripper.
act(standoff("search"))
assert policy.grasp_stage == "approach", policy.grasp_stage
hover_pos = (BOX - np.array([0.0, 0.0, 0.06])) - gripper_offset(BOX_YAW)
act(hover_pos)
assert policy.grasp_stage == "grasp", policy.grasp_stage
print(f"2. search->approach->grasp OK; stage={policy.grasp_stage}")

# 3. jaw is owned by the alignment gate: standing at the hull standoff is NOT
#    enough to close (this is what the pilot got wrong)
for _ in range(4):
    a = act(standoff("grasp"))
assert policy.joint_cmd[0] < 0.01, ("closed without alignment", policy.joint_cmd)
print(f"3. hull standoff alone does NOT close the jaw (jaw={policy.joint_cmd[0]:.3f}) OK")

# 4. gripper actually on the object -> close, stage -> lift
pos_aligned = BOX - gripper_offset(BOX_YAW)
policy.grasp_stage, policy.grasp_hold_ticks = "grasp", 0
for _ in range(3):
    a = act(pos_aligned)
    if policy.grasp_stage == "lift":
        break
assert policy.grasp_stage == "lift", policy.grasp_stage
assert abs(policy.joint_cmd[0] - GRIPPER_CLOSE) < 1e-3, policy.joint_cmd
assert abs(a["action.joint_pos"][0][0] - GRIPPER_CLOSE) < 0.02, a["action.joint_pos"][0]
print(f"4. gripper on object -> closed ({policy.joint_cmd[0]:.3f}), stage=lift OK")

# 5. transfer task: grasp -> lift -> carry -> release at destination
policy._reset_episode(None)
policy.task_str = ""
dest = np.array([12.0, 8.0, 2.5, 0.0])
logs = Path(args.eval_root) / "transfer_red_shallow" / "logs"
logs.mkdir(parents=True, exist_ok=True)
np.save(logs / "episode_0_desti.npy", dest)
task = "Pick up the red cylinder and transfer it to the box"
act(standoff("search") + np.array([2.0, 0, 0]), task=task)
assert policy.mode == "transfer", policy.mode
policy.grasp_stage, policy.grasp_hold_ticks = "grasp", 0
for _ in range(3):
    act(pos_aligned, task=task)
    if policy.grasp_stage == "lift":
        break
assert policy.grasp_stage == "lift", policy.grasp_stage
act(standoff("lift"), task=task)
assert policy.grasp_stage == "carry", policy.grasp_stage
# carry target: gripper 0.5 m above the destination point (expert TRANSPORTING)
carry_pos = (dest[:3] - np.array([0.0, 0.0, 0.5])) - gripper_offset(dest[3])
for _ in range(4):
    a = act(carry_pos, yaw=dest[3], task=task)
assert policy.joint_cmd[0] < 0.01, policy.joint_cmd  # released
assert policy.waypoints is None, "grasp task must not adopt a stale traj file"
print("5. transfer: grasp->lift->carry->release at desti OK (no stale waypoints)")

# 6. dropout during grasp: blind path keeps the privileged primitive alive from
#    the dead-reckoned pose (alpha latched: live target), no crash
a = act(standoff("grasp"), task=task, dvl_valid=0.0)
assert policy.alpha > 0.99, policy.alpha
print(f"6. blind grasp chunk OK |u|={np.abs(a['action.pwm']).mean():.3f} alpha={policy.alpha:.2f}")

# 7. follow mode: track the live boat pose, heading = bearing to the boat
policy._reset_episode(None)
policy.task_str = ""
boat = np.array([20.0, 0.0, 0.3])
o = obs_at(np.array([10.0, 0.0, 0.8]), 0.0, task="Follow the boat")
o["state.box_pos"] = boat.reshape(1, 3).astype(np.float32)
o["state.box_rpy"] = np.array([[0.0, 0.0, 0.0]], np.float32)
out = policy.act(o)
assert policy.mode == "follow" and policy.waypoints is None, (policy.mode, policy.waypoints)
vg, ye = policy._follow_goal(np.array([10.0, 0.0, 0.8]), 0.0, o)
assert vg[0] > 0.2, vg  # boat 10 m ahead, standoff 3 m behind it -> surge forward
assert abs(ye) < 1e-6, ye  # already pointing at the boat
vg2, ye2 = policy._follow_goal(np.array([17.0, 0.0, 0.8]), 0.0, o)
assert abs(vg2[0]) < 0.05, vg2  # at the standoff -> no surge
print(f"7. follow: surge {vg[0]:.2f} at 10 m, {vg2[0]:.2f} at standoff, yaw_err {ye:.2f} OK")

# 8. follow under dropout: blind path plans toward the boat (alpha latched)
o_blind = dict(o)
o_blind["meta.dvl_valid"] = np.array([[0.0]], np.float32)
policy.act(o_blind)
assert policy.alpha > 0.99, policy.alpha
print(f"8. blind follow OK alpha={policy.alpha:.2f}")

print("SMOKE_GRASP_ALL_OK")
