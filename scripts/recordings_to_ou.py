#!/usr/bin/env python3
"""Convert eval-harness episode recordings (10 Hz pkl per frame) into OU-schema
npz files so the estimator can be trained on the DEPLOYMENT distribution:
real task scenes (0.5-17 m depth), U0-driven windows with saturated thrusters
and 0.4-0.7 m/s speeds, the bridge's smoothed PWM. The raw DVL in the pkl is
the simulator truth even in dropout episodes (the recorder bypasses the bridge).

Units follow the OU npz convention consumed by uwam.data.load_ou_split:
pressure raw Pa, dvl_h metres, pwm the published command (pwm_is_commanded).
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np

SKIP = ("_archive", "percept", "dagger", "pilot")


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _yaw(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def load_ep(ep_dir: Path):
    """10 Hz arrays from a harness recording. Besides the vehicle channels, emits the manipulator
    extension used by the arm/object world model: joint_pos / joint_v (5, bridge order a..e),
    joint_cmd (5: recorded arm targets b..e; the gripper target is not logged -> next actual
    gripper angle) and obj_body (6: object position in the body frame, cos/sin of the object yaw
    relative to the vehicle, valid flag; zeros when the scene has no object)."""
    pk = sorted((q for q in ep_dir.glob("*.pkl") if q.stem.isdigit()), key=lambda p: int(p.stem))
    out = {k: [] for k in ("pwm", "dvl", "imu_av", "imu_la", "pressure", "dvl_h", "joint_pos", "joint_v", "joint_cmd", "obj_body")}
    for p in pk:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        st = d["observation"]["state"]
        if not (st.get("dvl") and st.get("imu") and st.get("pressure")):
            continue
        pwm = d["action"].get("pwm")
        pwm = np.zeros(8, np.float32) if pwm is None else np.asarray(pwm, np.float32)[:8]
        v, av, la = st["dvl"]["velocity"], st["imu"]["angular_velocity"], st["imu"]["linear_acceleration"]
        out["pwm"].append(pwm)
        out["dvl"].append([v["x"], v["y"], v["z"]])
        out["imu_av"].append([av["x"], av["y"], av["z"]])
        out["imu_la"].append([la["x"], la["y"], la["z"]])
        out["pressure"].append([st["pressure"]["fluid_pressure"]])
        out["dvl_h"].append([st["dvl"].get("altitude", 0.0)])
        js = st.get("joint_states") if isinstance(st.get("joint_states"), dict) else None
        jp = list(js.get("position", []))[:5] if js else []
        jv = list(js.get("velocity", []))[:5] if js else []
        out["joint_pos"].append((jp + [0.0] * 5)[:5])
        out["joint_v"].append((jv + [0.0] * 5)[:5])
        cmd = [np.nan] * 5
        dj = d["action"].get("desired_joint_state") if isinstance(d.get("action"), dict) else None
        if isinstance(dj, dict) and dj.get("position"):
            for n_, val in zip(dj.get("name", []), dj["position"]):
                n_ = str(n_).split("/")[-1]
                if n_ in ("axis_b", "axis_c", "axis_d", "axis_e"):
                    cmd["abcde".index(n_[-1])] = float(val)
        out["joint_cmd"].append(cmd)
        odom, obj = st.get("odom"), st.get("object_odom")
        if odom and obj:
            rp = odom["pose"]["pose"]["position"]; rq = odom["pose"]["pose"]["orientation"]
            op = obj["pose"]["pose"]["position"]; oq = obj["pose"]["pose"]["orientation"]
            R_wb = _quat_to_R([rq["x"], rq["y"], rq["z"], rq["w"]])
            rel = R_wb.T @ np.array([op["x"] - rp["x"], op["y"] - rp["y"], op["z"] - rp["z"]])
            dpsi = _yaw([oq["x"], oq["y"], oq["z"], oq["w"]]) - _yaw([rq["x"], rq["y"], rq["z"], rq["w"]])
            out["obj_body"].append([rel[0], rel[1], rel[2], np.cos(dpsi), np.sin(dpsi), 1.0])
        else:
            out["obj_body"].append([0.0] * 6)
    res = {k: np.asarray(v, np.float32) for k, v in out.items()}
    T = len(res["pwm"])
    if T:
        jc = res["joint_cmd"]
        # gripper target = next actual gripper angle; arm targets hold the last known value when absent
        jc[:, 0] = np.concatenate([res["joint_pos"][1:, 0], res["joint_pos"][-1:, 0]])
        for j in range(1, 5):
            col = jc[:, j]
            if np.isnan(col).all():
                col[:] = res["joint_pos"][:, j]
            else:
                last = res["joint_pos"][0, j]
                for t in range(T):
                    if np.isnan(col[t]):
                        col[t] = last
                    last = col[t]
        res["joint_cmd"] = jc
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--out", default="/hy-tmp/data/deploy_rec")
    ap.add_argument("--holdout", default="inspect_pipeline_sea,scan_ship_modern",
                    help="tasks written to <out>_holdout instead (validation)")
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--include", default="",
                    help="only <arm_cond> dirs starting with this prefix (e.g. collect_)")
    ap.add_argument("--exclude", default="collect_",
                    help="skip <arm_cond> dirs starting with this prefix")
    ap.add_argument("--skip", default=",".join(SKIP),
                    help="comma-separated substrings of episode paths to skip (default keeps pilots / percept "
                         "arms out of the locomotion sets; pass '' to convert everything, e.g. grasp recordings)")
    ap.add_argument("--eta", default="",
                    help="comma-separated per-thruster efficiency the recordings were made under "
                         "(e.g. 0,1,1,1,1,1,1,1 for a dead thruster 1); default = nominal ones")
    args = ap.parse_args()
    skip = tuple(x for x in args.skip.split(",") if x)
    eta_row = np.ones(8, np.float32)
    if args.eta:
        eta_row = np.asarray([float(x) for x in args.eta.split(",")], np.float32).reshape(8)
    holdout = set(args.holdout.split(",")) if args.holdout else set()
    out, out_h = Path(args.out), Path(args.out + "_holdout")
    out.mkdir(parents=True, exist_ok=True)
    out_h.mkdir(parents=True, exist_ok=True)
    n_tr = n_ho = 0
    for ep in sorted(glob.glob(f"{args.runs}/*/*/episode*")):
        if any(s in ep for s in skip):
            continue
        parts = Path(ep).parts
        arm_cond, task, epn = parts[-3], parts[-2], parts[-1]
        if args.include and not arm_cond.startswith(args.include):
            continue
        if not args.include and args.exclude and arm_cond.startswith(args.exclude):
            continue
        rec = load_ep(Path(ep))
        T = len(rec["pwm"])
        if T < args.min_frames:
            continue
        # simulator blow-ups (vehicle teleported to ~3e5 m) leave |accel| in the
        # thousands; they must not enter the estimator's training set
        ob = rec.get("obj_body")
        obj_blow = ob is not None and len(ob) and (not np.isfinite(ob[:, :3]).all() or np.abs(ob[:, :3]).max() > 5.0)
        if np.linalg.norm(rec["imu_la"], axis=1).max() > 30.0 or np.abs(rec["dvl"]).max() > 5.0 or obj_blow:
            print(f"SKIP blow-up {arm_cond}/{task}/{epn}" + (" (object > 5 m)" if obj_blow else ""))
            continue
        dest = out_h if task in holdout else out
        name = f"{arm_cond}__{task}__{epn}.npz"
        np.savez_compressed(
            dest / name, eta=np.tile(eta_row, (T, 1)), pwm_is_commanded=np.bool_(True),
            timestamp=np.arange(T, dtype=np.float64) * 0.1, dt=np.float64(0.1), **rec)
        if dest is out_h:
            n_ho += T
        else:
            n_tr += T
        print(f"{name:<64} {T:5d} frames  speed p90 {np.percentile(np.linalg.norm(rec['dvl'][:, :2], axis=1), 90):.2f}")
    print(f"train frames {n_tr}  holdout frames {n_ho}")


if __name__ == "__main__":
    main()
