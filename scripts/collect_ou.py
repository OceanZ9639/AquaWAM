#!/usr/bin/env python3
"""OU collection on official u0env, locked to the WAM 10 Hz (DVL) clock.

Steps on each new /bluerov2/dvl_sim message so Δt is simulation time, not wall
clock. Run with RoboStack python:

  source /hy-tmp/envs/ros_env/setup.bash
  source /hy-tmp/u0env/ros_ws/devel/setup.bash
  python /hy-tmp/underwater_wam/scripts/collect_ou.py --all --frames 2700
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg, ensure_dirs
from uwam.control import OUProcess
from uwam.sim import REGIMES, apply_efficiency, collect_ou_offline_placeholder, regime_at

FLS_TOPIC = "/bluerov2/fls/image"
RGB_TOPIC = "/bluerov2/left/image_color"
CURRENT_TOPIC = "/bluerov2/ocean_current"
TARGET_DT = 0.1


import cv2


def _img_to_u8(msg) -> np.ndarray:
    h, w = msg.height, msg.width
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = raw.reshape(h, w, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
        return img
    if msg.encoding in ("mono8", "8UC1"):
        return raw.reshape(h, w)
    if msg.step // max(1, w) == 2:
        return raw.view(np.uint16).reshape(h, w).astype(np.uint8)
    return raw.reshape(h, -1)[:, :w]


def _stamp_sec(msg) -> float:
    return float(msg.header.stamp.secs) + 1e-9 * float(msg.header.stamp.nsecs)


def collect_ros(out_dir: Path, n_frames: int, regime_name: str, save_fls: bool, save_rgb: bool) -> Path:
    import rospy
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import Imu, Range, FluidPressure, Image
    from geometry_msgs.msg import Vector3
    from stonefish_ros.msg import DVL

    if not getattr(rospy, "core", None) or not rospy.core.is_initialized():
        rospy.init_node("uwam_ou_collect", anonymous=True)
    pub = rospy.Publisher("/bluerov2/setpoint/pwm", Float64MultiArray, queue_size=1)
    cur_pub = rospy.Publisher(CURRENT_TOPIC, Vector3, queue_size=1)
    lock = threading.Lock()
    dvl_event = threading.Event()
    buf = {
        "dvl": np.zeros(3, np.float32),
        "dvl_stamp": 0.0,
        "dvl_seq": -1,
        "imu_av": np.zeros(3, np.float32),
        "imu_la": np.array([0, 0, -9.81], np.float32),
        "pressure": np.float32(101300.0),
        "alt": np.float32(1.0),
        "fls": None,
        "rgb": None,
        "n_imu": 0,
        "n_dvl": 0,
        "n_fls": 0,
        "n_rgb": 0,
    }

    def on_imu(msg):
        with lock:
            buf["imu_av"] = np.array(
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], np.float32
            )
            buf["imu_la"] = np.array(
                [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], np.float32
            )
            buf["n_imu"] += 1

    def on_dvl(msg):
        with lock:
            buf["dvl"] = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z], np.float32)
            if msg.altitude > 0:
                buf["alt"] = np.float32(msg.altitude)
            buf["dvl_stamp"] = _stamp_sec(msg)
            buf["dvl_seq"] = int(msg.header.seq)
            buf["n_dvl"] += 1
        dvl_event.set()

    def on_alt(msg):
        with lock:
            buf["alt"] = np.float32(msg.range)

    def on_pressure(msg):
        with lock:
            buf["pressure"] = np.float32(msg.fluid_pressure)

    def on_fls(msg):
        try:
            img = _img_to_u8(msg)
        except Exception:
            return
        with lock:
            buf["fls"] = img
            buf["n_fls"] += 1

    def on_rgb(msg):
        try:
            img = _img_to_u8(msg)
        except Exception:
            return
        with lock:
            buf["rgb"] = img
            buf["n_rgb"] += 1

    rospy.Subscriber("/bluerov2/imu", Imu, on_imu, queue_size=20)
    rospy.Subscriber("/bluerov2/dvl_sim", DVL, on_dvl, queue_size=20)
    rospy.Subscriber("/bluerov2/altitude", Range, on_alt, queue_size=5)
    rospy.Subscriber("/bluerov2/pressure", FluidPressure, on_pressure, queue_size=10)
    if save_fls:
        rospy.Subscriber(FLS_TOPIC, Image, on_fls, queue_size=1)
    if save_rgb:
        rospy.Subscriber(RGB_TOPIC, Image, on_rgb, queue_size=1)

    rospy.loginfo("waiting for /bluerov2/imu and /bluerov2/dvl_sim (DVL clock) ...")
    t_wait = time.time()
    while not rospy.is_shutdown() and (buf["n_imu"] == 0 or buf["n_dvl"] == 0):
        if time.time() - t_wait > 180:
            raise TimeoutError("no IMU/DVL after 180s; is parsed_simulator in the main loop?")
        rospy.sleep(0.05)

    ou = OUProcess()
    pwm = np.zeros((n_frames, 8), np.float32)
    dvl = np.zeros((n_frames, 3), np.float32)
    imu_av = np.zeros((n_frames, 3), np.float32)
    imu_la = np.zeros((n_frames, 3), np.float32)
    pressure = np.zeros((n_frames, 1), np.float32)
    alt = np.zeros((n_frames, 1), np.float32)
    stamps = np.zeros((n_frames,), np.float64)
    dvl_seq = np.zeros((n_frames,), np.int32)
    fls_frames = []
    rgb_frames = []
    reg = next(x for x in REGIMES if x.name == regime_name)
    last_seq = -1
    t0_wall = time.time()
    i = 0
    while i < n_frames and not rospy.is_shutdown():
        dvl_event.clear()
        if not dvl_event.wait(timeout=2.0):
            rospy.logwarn("DVL timeout 2s at frame %d", i)
            continue
        with lock:
            seq = int(buf["dvl_seq"])
            if seq == last_seq:
                continue
            last_seq = seq
            t_sim = float(buf["dvl_stamp"])
            snapshot = {
                "dvl": buf["dvl"].copy(),
                "imu_av": buf["imu_av"].copy(),
                "imu_la": buf["imu_la"].copy(),
                "pressure": float(buf["pressure"]),
                "alt": float(buf["alt"]),
                "fls": None if buf["fls"] is None else buf["fls"].copy(),
                "rgb": None if buf["rgb"] is None else buf["rgb"].copy(),
                "seq": seq,
                "stamp": t_sim,
            }
        t_rel = t_sim - stamps[0] if i > 0 else 0.0
        if i == 0:
            t_rel = 0.0
        else:
            t_rel = snapshot["stamp"] - stamps[0]
        cur, eta = regime_at(reg, t_rel)
        cur_pub.publish(Vector3(x=float(cur[0]), y=float(cur[1]), z=float(cur[2])))
        u = apply_efficiency(ou.sample(), eta)
        pub.publish(Float64MultiArray(data=u.tolist()))
        pwm[i] = u
        dvl[i] = snapshot["dvl"]
        imu_av[i] = snapshot["imu_av"]
        imu_la[i] = snapshot["imu_la"]
        pressure[i, 0] = snapshot["pressure"]
        alt[i, 0] = snapshot["alt"]
        stamps[i] = snapshot["stamp"]
        dvl_seq[i] = snapshot["seq"]
        if save_fls:
            fls_frames.append(snapshot["fls"])
        if save_rgb:
            rgb_frames.append(snapshot["rgb"])
        i += 1

    n = i
    pwm, dvl, imu_av, imu_la = pwm[:n], dvl[:n], imu_av[:n], imu_la[:n]
    pressure, alt, stamps, dvl_seq = pressure[:n], alt[:n], stamps[:n], dvl_seq[:n]
    dts = np.diff(stamps) if n > 1 else np.array([TARGET_DT])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ou_{regime_name}.npz"
    payload = dict(
        pwm=pwm, dvl=dvl, imu_av=imu_av, imu_la=imu_la, pressure=pressure, dvl_h=alt,
        timestamp=(stamps - stamps[0]).astype(np.float64), dvl_seq=dvl_seq,
    )
    n_fls_real = int(buf["n_fls"])
    if fls_frames and n_fls_real > 0:
        last = next((f for f in fls_frames if f is not None), None)
        if last is not None:
            packed = [f if f is not None else last for f in fls_frames]
            payload["fls"] = np.stack(packed, axis=0)
    if rgb_frames and buf["n_rgb"] > 0:
        last = next((f for f in rgb_frames if f is not None), None)
        if last is not None:
            packed = [f if f is not None else last for f in rgb_frames]
            payload["rgb"] = np.stack(packed, axis=0)
    np.savez_compressed(path, **payload)
    meta = {
        "regime": regime_name,
        "frames": int(n),
        "dt_median": float(np.median(dts)) if n > 1 else None,
        "dt_target": TARGET_DT,
        "n_imu": int(buf["n_imu"]),
        "n_dvl": int(buf["n_dvl"]),
        "n_fls": int(buf["n_fls"]),
        "n_rgb": int(buf["n_rgb"]),
        "n_fls_real": n_fls_real,
        "unique_dvl_seq": int(len(np.unique(dvl_seq))),
        "wall_sec": time.time() - t0_wall,
        "placeholder": False,
        "fls_topic": FLS_TOPIC,
        "rgb_topic": RGB_TOPIC,
    }
    (out_dir / f"ou_{regime_name}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/hy-tmp/data/ou_explore")
    ap.add_argument("--seconds", type=float, default=270.0)
    ap.add_argument("--frames", type=int, default=0, help="if >0, ignore --seconds; count of DVL ticks")
    ap.add_argument("--rate", type=float, default=10.0, help="target Hz (must stay 10)")
    ap.add_argument("--regime", default="nominal", choices=[r.name for r in REGIMES])
    ap.add_argument("--all", action="store_true", help="collect every REGIME in one ROS node")
    ap.add_argument("--placeholder", action="store_true", help="offline surrogate only (not official)")
    ap.add_argument("--no-fls", action="store_true")
    ap.add_argument("--rgb", action="store_true")
    args = ap.parse_args()
    if abs(args.rate - 10.0) > 1e-6:
        raise SystemExit("WAM clock is 10 Hz; refusing --rate != 10")
    out = Path(args.out)
    ensure_dirs(Cfg())
    n_frames = args.frames if args.frames > 0 else int(round(args.seconds * args.rate))
    if args.placeholder:
        path = collect_ou_offline_placeholder(out, n_frames=n_frames)
        print("placeholder (not official)", path)
        return
    regimes = [r.name for r in REGIMES] if args.all else [args.regime]
    for name in regimes:
        print("collecting", name, "frames", n_frames, flush=True)
        collect_ros(out, n_frames, name, save_fls=not args.no_fls, save_rgb=args.rgb)


if __name__ == "__main__":
    main()
