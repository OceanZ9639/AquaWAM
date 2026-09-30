#!/usr/bin/env python3
"""Scene-diverse exploration policy for re-collecting the WAM training set.

Runs behind the eval harness like any policy server, so the harness's recorder
captures the DEPLOYMENT sensor stack in every task scene (10 Hz DVL/IMU/
pressure/altitude, the bridge-smoothed PWM, and all four cameras for the later
multimodal stage). Command regimes are chosen to cover what the first training
set lacked (measured on eval recordings):
  * speeds up to 0.65 m/s and saturated horizontal thrusters (U0's regime),
  * level flight via the attitude loop, with occasional deliberate mild tilts,
  * passive coasting (decay data), yaw spins, OU noise, mixer velocity goals.
Bounded around the spawn point (privileged odometry is fine for collection).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import AXIS_SIGN, attitude_pwm, sixdof_thrust_to_pwm, vel_track_pwm  # noqa: E402

CHUNK = 16
DT = 0.1
ARMED = np.array([0.0, 0.4965, 0.4965, 0.5056, 0.0], np.float64)


def _yaw_wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class Explore:
    def __init__(self, seed: int, radius: float, depth_up: float, depth_down: float, min_alt: float):
        self.rng = np.random.default_rng(seed)
        self.radius, self.depth_up, self.depth_down, self.min_alt = radius, depth_up, depth_down, min_alt
        self.spawn = None
        self.spawn_alt = None
        self.seg = None
        self.seg_left = 0
        self.ou = np.zeros(8, np.float32)
        self.n = 0
        self.last_task = None

    def _new_segment(self):
        r = self.rng.random()
        hold = int(self.rng.integers(1, 4))  # chunks (1.6-4.8 s)
        if r < 0.30:
            speed = float(self.rng.uniform(0.05, 0.65))
            ang = float(self.rng.normal(0.0, 0.6))
            goal = np.array([speed * np.cos(ang), speed * np.sin(ang), float(self.rng.uniform(-0.12, 0.12))], np.float32)
            seg = {"kind": "mixer", "goal": goal, "kp": float(self.rng.uniform(1.0, 3.5)),
                   "kff": float(self.rng.uniform(1.4, 2.6))}
        elif r < 0.55:
            thr = np.zeros(3, np.float32)
            axis = int(self.rng.integers(0, 2))
            thr[axis] = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.6, 1.0))
            if self.rng.random() < 0.4:
                thr[1 - axis] = float(self.rng.uniform(-0.5, 0.5))
            thr[2] = float(self.rng.uniform(-0.3, 0.3))
            seg = {"kind": "primitive", "thrust": thr, "yaw": float(self.rng.uniform(-0.3, 0.3))}
        elif r < 0.80:
            seg = {"kind": "ou", "sigma": float(self.rng.uniform(0.3, 0.6)), "theta": 0.15}
        elif r < 0.90:
            seg = {"kind": "spin", "yaw": float(self.rng.choice([-1, 1]) * self.rng.uniform(0.25, 0.5)),
                   "surge": float(self.rng.uniform(0.0, 0.5))}
        else:
            seg = {"kind": "coast"}
        # deliberate mild tilt in a minority of segments (estimator robustness)
        seg["tilt"] = (np.array([self.rng.uniform(-0.2, 0.2), self.rng.uniform(-0.2, 0.2)], np.float32)
                       if self.rng.random() < 0.10 else None)
        self.seg, self.seg_left = seg, hold

    def _bounds_violation(self, pos, alt):
        """None, 'xy' (too far), 'deep' (below band / too close to the seabed) or
        'shallow' (above band). Scenes differ in spawn altitude (lake: 0.5 m), so
        the seabed clearance is relative to the altitude at spawn."""
        d = pos[:2] - self.spawn[:2]
        if np.linalg.norm(d) > self.radius:
            return "xy"
        if pos[2] > self.spawn[2] + self.depth_down:
            return "deep"
        if pos[2] < self.spawn[2] - self.depth_up:
            return "shallow"
        if alt is not None and 0.05 < alt and self.spawn_alt is not None:
            if alt < min(self.min_alt, 0.6 * self.spawn_alt):
                return "deep"
        return None

    def act(self, obs):
        task = obs.get("annotation.human.action.task_description") or [""]
        task = task[0] if isinstance(task, (list, tuple)) else str(task)
        pos = obs.get("state.odom_pos")
        rpy = obs.get("state.imu_rpy", obs.get("state.odom_rpy"))
        dvl = obs.get("state.dvl_v")
        av = obs.get("state.imu_av")
        alt = obs.get("state.dvl_h")
        pos = None if pos is None else np.ravel(np.asarray(pos, np.float64))[:3]
        rpy = None if rpy is None else np.ravel(np.asarray(rpy, np.float64))[:3]
        v = np.zeros(3) if dvl is None else np.ravel(np.asarray(dvl, np.float64))[:3]
        om = np.zeros(3) if av is None else np.ravel(np.asarray(av, np.float64))[:3]
        alt = None if alt is None else float(np.ravel(alt)[0])
        # new episode: spawn moved > 3 m or task string changed
        if pos is not None and (self.spawn is None or task != self.last_task
                                or np.linalg.norm(pos - self.spawn) > 60.0):
            self.spawn = pos.copy()
            self.spawn_alt = alt if (alt is not None and alt > 0.05) else None
            self.last_task = task
            self.seg = None
            print(f"[explore] new episode task={task!r} spawn={np.round(pos, 1)} alt={self.spawn_alt}",
                  flush=True)
        chunk = np.zeros((CHUNK, 8), np.float32)
        homing = self._bounds_violation(pos, alt) if pos is not None else None
        if homing:
            yaw = float(rpy[2]) if rpy is not None else 0.0
            if homing == "xy":
                # steer back toward the spawn point at moderate speed
                d = self.spawn - pos
                d[2] = np.clip(d[2], -0.3, 0.3)
                n = np.linalg.norm(d[:2])
                vw = 0.35 * d / max(n, 1e-6) if n > 1e-6 else d
            else:
                # depth-only violation: pure vertical correction toward spawn depth
                vw = np.array([0.0, 0.0, np.clip(self.spawn[2] - pos[2], -0.25, 0.25)])
                if homing == "deep" and vw[2] > -0.1:
                    vw[2] = -0.2
            cy, sy = np.cos(yaw), np.sin(yaw)
            vb = np.array([cy * vw[0] + sy * vw[1], -sy * vw[0] + cy * vw[1], vw[2]])
            goal = np.array([vb[0], AXIS_SIGN[1] * vb[1], AXIS_SIGN[2] * vb[2]], np.float32)
            yaw_err = _yaw_wrap(np.arctan2(d[1], d[0]) - yaw) if homing == "xy" else 0.0
            u = vel_track_pwm(v, goal, omega=om, rpy=rpy, yaw_err=float(np.clip(yaw_err, -0.6, 0.6)))
            chunk[:] = u
            self.seg = None
        else:
            if self.seg is None or self.seg_left <= 0:
                self._new_segment()
            self.seg_left -= 1
            s = self.seg
            tilt_rpy = None
            if rpy is not None:
                tilt_rpy = rpy.copy()
                if s["tilt"] is not None:
                    tilt_rpy[0] -= s["tilt"][0]  # bias the leveling setpoint
                    tilt_rpy[1] -= s["tilt"][1]
            lvl = attitude_pwm(tilt_rpy, om)
            if s["kind"] == "mixer":
                u = vel_track_pwm(v, s["goal"], omega=om, rpy=tilt_rpy, kp=s["kp"], kff=s["kff"])
                chunk[:] = u
            elif s["kind"] == "primitive":
                u = sixdof_thrust_to_pwm(s["thrust"], np.array([0, 0, s["yaw"]], np.float32))
                chunk[:] = np.clip(u + lvl, -1, 1)
            elif s["kind"] == "ou":
                for i in range(CHUNK):
                    self.ou += -s["theta"] * self.ou + s["sigma"] * np.sqrt(DT) * self.rng.standard_normal(8).astype(np.float32)
                    self.ou = np.clip(self.ou, -1, 1)
                    chunk[i] = np.clip(self.ou + lvl, -1, 1)
            elif s["kind"] == "spin":
                u = sixdof_thrust_to_pwm(np.array([s["surge"], 0, 0], np.float32), np.array([0, 0, s["yaw"]], np.float32))
                chunk[:] = np.clip(u + lvl, -1, 1)
            else:  # coast
                chunk[:] = lvl
        self.n += 1
        if self.n % 10 == 1:
            kind = "HOMING" if homing else (self.seg["kind"] if self.seg else "?")
            print(f"[explore] #{self.n} {kind} speed={np.linalg.norm(v[:2]):.2f} "
                  f"pos={None if pos is None else np.round(pos, 1)} |u|={np.abs(chunk).mean():.2f}", flush=True)
        return [{"action.pwm": chunk.astype(np.float64),
                 "action.joint_pos": np.tile(ARMED, (CHUNK, 1))}, {}]


def make_handler(policy):
    import json_numpy

    json_numpy.patch()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"status": "healthy", "model": "EXPLORE"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n).decode())
                if "encoded" in payload:
                    payload = json.loads(payload["encoded"])
                out = policy.act(payload["observation"])
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                body = json.dumps({"detail": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=18.0, help="horizontal bound around spawn (m)")
    ap.add_argument("--depth-up", type=float, default=1.5, help="allowed rise above spawn depth (m)")
    ap.add_argument("--depth-down", type=float, default=4.0, help="allowed descent below spawn depth (m)")
    ap.add_argument("--min-alt", type=float, default=0.8, help="seabed clearance (m)")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port),
                              make_handler(Explore(a.seed, a.radius, a.depth_up, a.depth_down, a.min_alt)))
    print(f"[explore] listening on {a.port}", flush=True)
    srv.serve_forever()
