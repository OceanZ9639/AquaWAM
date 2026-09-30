#!/usr/bin/env python3
"""Offline smoke test of the WAM grasp planner inside the policy server (no sim, no HTTP).

Puts the vehicle at the approach standoff of a synthetic object with the arm at the armed pose,
forces the grasp stage, and calls act() a few times: checks the action shapes, that the planner is
used (log lines), that predictions are finite / physically plausible, and reports the imagined
gripper-object error and the per-act latency."""
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_policy_server import ARMED_JOINTS, WamPolicy  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/models/uwam/grasp_core.pt"
args = types.SimpleNamespace(
    ckpt="/hy-tmp/models/uwam/best_scenes.pt", vel_ens="/hy-tmp/models/uwam/vel_ens_v2.pt",
    gate_calib="/hy-tmp/models/uwam/gate_calib.json", ou="", eval_root="/tmp/smoke_gp_root", hold_anchor="mixer",
    dump_dir="", grasp_planner="wam", grasp_ckpt=ckpt, arm_kin="/hy-tmp/models/uwam/arm_kin.pt")
Path(args.eval_root).mkdir(parents=True, exist_ok=True)
pol = WamPolicy(args)

BOX = np.array([10.0, 5.0, 3.0]); BOX_YAW = 0.0


def obs_at(pos, joints, vel=(0, 0, 0)):
    L = 16
    return {
        "state.hist_dvl": np.tile(np.asarray(vel, np.float32), (L, 1)), "state.hist_imu_av": np.zeros((L, 3), np.float32),
        "state.hist_imu_la": np.tile(np.array([0, 0, -9.8], np.float32), (L, 1)),
        "state.hist_pressure": np.full((L, 1), 3.0, np.float32), "state.hist_dvl_h": np.full((L, 1), 2.0, np.float32),
        "state.hist_pwm": np.zeros((L, 8), np.float32),
        "state.hist_joint_pos": np.tile(np.asarray(joints, np.float32), (L, 1)), "state.hist_joint_v": np.zeros((L, 5), np.float32),
        "state.joint_pos": np.asarray(joints, np.float32).reshape(1, 5), "state.joint_v": np.zeros((1, 5), np.float32),
        "state.joint_effort": np.zeros((1, 5), np.float32),
        "state.odom_pos": np.asarray(pos, np.float32).reshape(1, 3), "state.odom_rpy": np.zeros((1, 3), np.float32),
        "state.imu_rpy": np.zeros((1, 3), np.float32), "state.dvl_v": np.zeros((1, 3), np.float32),
        "state.box_pos": BOX.reshape(1, 3).astype(np.float32), "state.box_rpy": np.array([[0, 0, BOX_YAW]], np.float32),
        "state.ee_pose": np.array([[0.005, 0.0, 0.171]], np.float32),
        "meta.dvl_valid": np.array([[1.0]], np.float32),
        "annotation.human.action.task_description": ["Pick up the red cylinder"],
    }


# hull placed so that the gripper (FK at armed: body (0.367, -0.084, 0.2005)) sits 7 cm above the object
ee_body = pol.grasp_planner.ee_body(ARMED_JOINTS)
pos = BOX - np.array([0.0, 0.0, 0.07]) - ee_body   # body z points down: 7 cm above = z smaller
o = obs_at(pos, ARMED_JOINTS)
pol.act(o)                      # new episode -> mode grasp, stage search
pol.grasp_stage = "grasp"       # force the planner stage
t0 = time.perf_counter()
n = 6
for i in range(n):
    out = pol.act(o)[0]
    assert out["action.pwm"].shape == (16, 8) and out["action.joint_pos"].shape == (16, 5)
    assert np.isfinite(out["action.pwm"]).all() and np.isfinite(out["action.joint_pos"]).all()
dt = (time.perf_counter() - t0) / n
q = out["action.joint_pos"][0]
print(f"planner acts ok: {1e3 * dt:.0f} ms/act, joint target {np.round(q, 3).tolist()}, |pwm| {np.abs(out['action.pwm']).mean():.3f}, "
      f"stage now {pol.grasp_stage}, insane candidates last act: {getattr(pol.grasp_planner, 'n_insane', '?')}")

# --- DVL dropout mid-grasp: the bridge freezes odometry and zeroes the DVL, but keeps sending the
# relative object measurement; the policy must keep running the staged grasp on its dead-reckoned pose.
sighted_box = pol._box_in_frame(o, pos, 0.0)["state.box_pos"].ravel()
assert np.allclose(sighted_box, BOX, atol=1e-4), sighted_box      # re-anchoring is exact when sighted
ob = dict(o)
true_pos = pos + np.array([0.02, -0.01, 0.0])                     # the hull drifted 2 cm since the freeze
ob["state.odom_pos"] = np.asarray(pos, np.float32).reshape(1, 3)  # frozen (stale) odometry
ob["state.box_rel"] = (BOX - true_pos).astype(np.float32).reshape(1, 3)
ob["state.box_rel_yaw"] = np.array([[0.0]], np.float32)
ob["state.hist_dvl"] = np.zeros((16, 3), np.float32)
ob["state.dvl_v"] = np.zeros((1, 3), np.float32)
ob["meta.dvl_valid"] = np.array([[0.0]], np.float32)
for i in range(4):
    out = pol.act(ob)[0]
    assert out["action.pwm"].shape == (16, 8) and np.isfinite(out["action.pwm"]).all()
assert pol.dr_pos is not None and pol.grasp_stage in ("approach", "grasp", "lift", "search"), pol.grasp_stage
rel_seen = pol._box_in_frame(ob, pol.dr_pos, pol.dr_yaw or 0.0)["state.box_pos"].ravel() - pol.dr_pos
assert np.allclose(rel_seen, BOX - true_pos, atol=1e-4), (rel_seen, BOX - true_pos)  # relative geometry exact in DR frame
print(f"blind grasp ok: dr_pos {np.round(pol.dr_pos, 3).tolist()} stage {pol.grasp_stage} |pwm| {np.abs(out['action.pwm']).mean():.3f}")
print("SMOKE_GRASP_PLANNER_OK")
