#!/usr/bin/env python3
"""Off-corridor DAgger collector policy: wander random vantage points around the
grasp object so the harness's recorder captures close-range views from entry
angles the expert corridor never visits. Labels are synthesized OFFLINE from the
recorded privileged poses (percept/synth_labels.py); the deployed head stays
vision-only.

Privileged reads (box + odom) are fine here: this server only GENERATES training
data, exactly like the paper's expert collector. No torch, no learned model --
a P velocity tracker is all the wandering needs.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import AXIS_SIGN, attitude_pwm, vel_track_pwm  # noqa: E402

CHUNK = 16
ARMED = np.array([0.0, 0.4965, 0.4965, 0.5056, 0.0], np.float64)


def _yaw_wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class Wander:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.vantage = None
        self.hold = 0
        self.last_box = None

    def _sample_vantage(self, box, box_yaw):
        r = self.rng.uniform(0.12, 0.5) if self.rng.random() < 0.6 else self.rng.uniform(0.5, 1.1)
        th = self.rng.uniform(0, 2 * np.pi)
        dz = self.rng.uniform(0.08, 0.45)  # above the object (world z down => shallower)
        tgt = box + np.array([r * np.cos(th), r * np.sin(th), -dz])
        # mostly look at the object, sometimes随机朝向
        if self.rng.random() < 0.8:
            yaw = np.arctan2(box[1] - tgt[1], box[0] - tgt[0]) + self.rng.uniform(-0.5, 0.5)
        else:
            yaw = self.rng.uniform(-np.pi, np.pi)
        return tgt, _yaw_wrap(yaw)

    def act(self, obs):
        box = obs.get("state.box_pos")
        rpyb = obs.get("state.box_rpy")
        pos = obs.get("state.odom_pos")
        rpy = obs.get("state.odom_rpy")
        dvl = obs.get("state.dvl_v")
        av = obs.get("state.imu_av")
        if box is None or pos is None:
            u = attitude_pwm(None, None)
            return [{"action.pwm": np.tile(u, (CHUNK, 1)).astype(np.float64),
                     "action.joint_pos": np.tile(ARMED, (CHUNK, 1))}, {}]
        box = np.ravel(np.asarray(box, np.float64))[:3]
        box_yaw = float(np.ravel(rpyb)[2]) if rpyb is not None else 0.0
        pos = np.ravel(np.asarray(pos, np.float64))[:3]
        rpy = np.ravel(np.asarray(rpy, np.float64))[:3] if rpy is not None else np.zeros(3)
        v = np.ravel(np.asarray(dvl, np.float64))[:3] if dvl is not None else np.zeros(3)
        om = np.ravel(np.asarray(av, np.float64))[:3] if av is not None else np.zeros(3)
        # new episode detection: box moved a lot -> resample immediately
        if self.last_box is None or np.linalg.norm(box - self.last_box) > 1.0:
            self.vantage = None
        self.last_box = box.copy()
        if self.vantage is None or self.hold <= 0:
            self.vantage = self._sample_vantage(box, box_yaw)
            self.hold = int(self.rng.integers(2, 4))
        self.hold -= 1
        tgt, yaw_ref = self.vantage
        d_world = tgt - pos
        v_world = 0.8 * d_world
        n = float(np.linalg.norm(v_world))
        if n > 0.2:
            v_world *= 0.2 / n
        yaw = float(rpy[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        vx = cy * v_world[0] + sy * v_world[1]
        vy = -sy * v_world[0] + cy * v_world[1]
        v_goal = np.array([vx, AXIS_SIGN[1] * vy, AXIS_SIGN[2] * v_world[2]], np.float32)
        yaw_err = float(np.clip(_yaw_wrap(yaw_ref - yaw), -0.4, 0.4))
        u = vel_track_pwm(v, v_goal, omega=om, rpy=rpy, yaw_err=yaw_err)
        return [{"action.pwm": np.tile(u, (CHUNK, 1)).astype(np.float64),
                 "action.joint_pos": np.tile(ARMED, (CHUNK, 1))}, {}]


def make_handler(policy):
    import json_numpy

    json_numpy.patch()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"status": "healthy", "model": "WANDER"}).encode()
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
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(Wander(a.seed)))
    print(f"[wander] listening on {a.port}", flush=True)
    srv.serve_forever()
