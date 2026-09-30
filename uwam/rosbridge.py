"""Single-subscription ROS bridge to the Stonefish BlueROV2 plus PD re-homing.

`/stonefish_simulator/respawn_robot` aborts the simulator process (exit -4), so trials
are re-homed with a pose PD controller instead of a hard reset.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

from .control import pose_home_pwm

HOME_XYZ = (0.0, 0.0, 4.0)      # matches <arg name="position"> in the runtime scenario
# World z is depth: 0 is the surface, the seabed in this scene sits near 6 m.
DEPTH_BAND = (1.0, 5.5)
TILT_LIMIT = np.deg2rad(60.0)
FLS_TOPIC = "/bluerov2/fls/image"
RGB_TOPIC = "/bluerov2/left/image_color"


def img_to_u8(msg) -> np.ndarray:
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


def quat_to_rpy(x: float, y: float, z: float, w: float) -> np.ndarray:
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw], np.float32)


class SimBridge:
    """Subscribes once, steps on the DVL 10 Hz tick, and owns the PWM / current publishers."""

    def __init__(self, node: str = "uwam_bridge", robot: str = "bluerov2",
                 current_topic: str = "/bluerov2/ocean_current", images: bool = False):
        import rospy
        from std_msgs.msg import Float64MultiArray
        from sensor_msgs.msg import Image, Imu, Range, FluidPressure
        from geometry_msgs.msg import Vector3
        from nav_msgs.msg import Odometry
        from stonefish_ros.msg import DVL

        self._rospy = rospy
        self._Float64MultiArray = Float64MultiArray
        self._Vector3 = Vector3
        if not rospy.core.is_initialized():
            rospy.init_node(node, anonymous=True)

        self._lock = threading.Lock()
        self._ev = threading.Event()
        self._s = {
            "dvl": np.zeros(3, np.float32),
            "imu_av": np.zeros(3, np.float32),
            "imu_la": np.array([0.0, 0.0, -9.81], np.float32),
            "pressure": 101300.0,
            "alt": float("nan"),
            "alt_valid": False,
            "pos": np.zeros(3, np.float32),
            "rpy": np.zeros(3, np.float32),
            "have_odom": False,
            "n": 0,
            "seq": -1,
            "stamp": 0.0,
            "rgb": None,
            "fls": None,
            "n_img": 0,
        }

        def on_dvl(msg):
            with self._lock:
                self._s["dvl"] = np.array(
                    [msg.velocity.x, msg.velocity.y, msg.velocity.z], np.float32
                )
                self._s["n"] += 1
                self._s["seq"] = int(msg.header.seq)
                self._s["stamp"] = float(msg.header.stamp.secs) + 1e-9 * float(msg.header.stamp.nsecs)
            self._ev.set()

        def on_imu(msg):
            with self._lock:
                self._s["imu_av"] = np.array(
                    [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], np.float32
                )
                self._s["imu_la"] = np.array(
                    [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                    np.float32,
                )

        def on_alt(msg):
            # Stonefish reports out-of-range beams as min_range / max_range; those are not altitudes.
            lo, hi = float(msg.min_range), float(msg.max_range)
            r = float(msg.range)
            with self._lock:
                self._s["alt"] = r
                self._s["alt_valid"] = bool(lo * 1.05 < r < hi * 0.95)

        def on_press(msg):
            with self._lock:
                self._s["pressure"] = float(msg.fluid_pressure)

        def on_odom(msg):
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            with self._lock:
                self._s["pos"] = np.array([p.x, p.y, p.z], np.float32)
                self._s["rpy"] = quat_to_rpy(q.x, q.y, q.z, q.w)
                self._s["have_odom"] = True

        self._subs = [
            rospy.Subscriber(f"/{robot}/dvl_sim", DVL, on_dvl, queue_size=20),
            rospy.Subscriber(f"/{robot}/imu", Imu, on_imu, queue_size=20),
            rospy.Subscriber(f"/{robot}/altitude", Range, on_alt, queue_size=20),
            rospy.Subscriber(f"/{robot}/pressure", FluidPressure, on_press, queue_size=20),
            rospy.Subscriber(f"/{robot}/odometry", Odometry, on_odom, queue_size=20),
        ]
        if images:
            def on_rgb(msg):
                try:
                    img = img_to_u8(msg)
                except Exception:
                    return
                with self._lock:
                    self._s["rgb"] = img
                    self._s["n_img"] += 1

            def on_fls(msg):
                try:
                    img = img_to_u8(msg)
                except Exception:
                    return
                with self._lock:
                    self._s["fls"] = img

            self._subs.append(rospy.Subscriber(RGB_TOPIC, Image, on_rgb, queue_size=1))
            self._subs.append(rospy.Subscriber(FLS_TOPIC, Image, on_fls, queue_size=1))
        self._pwm_pub = rospy.Publisher(f"/{robot}/setpoint/pwm", Float64MultiArray, queue_size=1)
        self._cur_pub = rospy.Publisher(current_topic, Vector3, queue_size=1)
        self._last_seq = -1

    def wait_ready(self, timeout: float = 60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._s["n"] > 0 and self._s["have_odom"]:
                    return True
            self._rospy.sleep(0.05)
        with self._lock:
            return self._s["n"] > 0

    def state(self) -> dict:
        with self._lock:
            return {
                "dvl": self._s["dvl"].copy(),
                "imu_av": self._s["imu_av"].copy(),
                "imu_la": self._s["imu_la"].copy(),
                "pressure": float(self._s["pressure"]),
                "alt": float(self._s["alt"]),
                "alt_valid": bool(self._s["alt_valid"]),
                "pos": self._s["pos"].copy(),
                "rpy": self._s["rpy"].copy(),
                "seq": int(self._s["seq"]),
                "stamp": float(self._s["stamp"]),
                "rgb": None if self._s["rgb"] is None else self._s["rgb"].copy(),
                "fls": None if self._s["fls"] is None else self._s["fls"].copy(),
            }

    def tick(self, timeout: float = 2.0) -> Optional[dict]:
        """Block until a fresh DVL sample arrives; returns None on timeout/shutdown."""
        while True:
            if self._rospy.is_shutdown():
                return None
            self._ev.clear()
            if not self._ev.wait(timeout=timeout):
                return None
            st = self.state()
            if st["seq"] == self._last_seq:
                continue
            self._last_seq = st["seq"]
            return st

    def publish_pwm(self, u) -> None:
        self._pwm_pub.publish(self._Float64MultiArray(data=[float(x) for x in np.asarray(u).reshape(-1)]))

    def publish_current(self, cur) -> None:
        c = np.asarray(cur, dtype=np.float64).reshape(3)
        self._cur_pub.publish(self._Vector3(x=float(c[0]), y=float(c[1]), z=float(c[2])))

    def is_unstable(self, st: dict) -> Tuple[bool, str]:
        z = float(st["pos"][2])
        if not (DEPTH_BAND[0] <= z <= DEPTH_BAND[1]):
            return True, "depth_out_of_band"
        rpy = st["rpy"]
        if abs(float(rpy[0])) > TILT_LIMIT or abs(float(rpy[1])) > TILT_LIMIT:
            return True, "tilted"
        return False, ""

    def home(self, target_z: float = HOME_XYZ[2], timeout: float = 30.0, hold: float = 1.0) -> dict:
        """Reset the dynamical state: nominal depth, level, at rest.

        x/y are deliberately left free. The metric is body-frame DVL velocity, and each trial
        translates the vehicle a few metres, so chasing an absolute waypoint would ask the PD
        to traverse hundreds of metres at 0.25 m/s and never converge.
        """
        t0 = None
        ok_since = None
        settled = False
        last = self.state()
        while True:
            st = self.tick()
            if st is None:
                break
            if t0 is None:
                t0 = st["stamp"]
            last = st
            err = np.array([0.0, 0.0, target_z - float(st["pos"][2])], np.float32)
            u = pose_home_pwm(err, float(st["rpy"][2]), st["dvl"], st["rpy"], st["imu_av"])
            self.publish_pwm(u)
            settled = (
                abs(float(err[2])) < 0.25
                and float(np.linalg.norm(st["dvl"])) < 0.05
                and abs(float(st["rpy"][0])) < np.deg2rad(8)
                and abs(float(st["rpy"][1])) < np.deg2rad(8)
            )
            if settled:
                ok_since = st["stamp"] if ok_since is None else ok_since
                if st["stamp"] - ok_since >= hold:
                    break
            else:
                ok_since = None
            if st["stamp"] - t0 > timeout:
                break
        self.publish_pwm(np.zeros(8, np.float32))
        return {
            "ok": bool(settled),
            "z_err": round(float(target_z - last["pos"][2]), 4),
            "speed": round(float(np.linalg.norm(last["dvl"])), 4),
            "roll_deg": round(float(np.rad2deg(last["rpy"][0])), 2),
            "pitch_deg": round(float(np.rad2deg(last["rpy"][1])), 2),
            "z": round(float(last["pos"][2]), 3),
        }
