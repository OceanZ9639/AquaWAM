#!/usr/bin/env python3
"""
Arm probe policy (data collection, no task): holds the vehicle roughly still with a gentle
level/hover law and drives the manipulator through random joint targets around the armed pose.
Every recorded frame then pairs joint angles with the simulator's end-effector pose, which is
exactly the forward-kinematics data the grasp planner needs (uwam/arm_kin.py fits it), plus the
joint-tracking response (how fast the MoveIt-executed targets are reached) for the structured
world model's arm prior.

  python3 u0eval/arm_probe_server.py --port 8006
  KEEP_ALL_EPISODES=1 COND_TAG=armprobe bash u0eval/run_eval_task.sh pick_pipe0_shallow collect 4 8006 -1.0 zero
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import attitude_pwm  # noqa: E402

CHUNK = 16
ARMED = np.array([0.0, 0.4965, 0.4965, 0.5056, 0.0], np.float32)   # gripper, b, c, d, e (bridge order)
# safe box around the armed pose (rad); MoveIt still clips to the URDF limits and rejects collisions
SPAN = np.array([0.0, 0.35, 0.35, 0.35, 0.6], np.float32)
HOLD_ACTS = 3   # keep each target for 3 acts (4.8 s at the 1.6 s cadence) so the arm settles


class ArmProbe:
    def __init__(self, seed: int = 0, span_scale: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.span = (SPAN * span_scale).astype(np.float32)
        self.target = ARMED.copy()
        self.n = 0
        self.grip_toggle = 0

    def act(self, obs):
        if self.n % HOLD_ACTS == 0:
            self.target = ARMED + self.rng.uniform(-1, 1, size=5).astype(np.float32) * self.span
            # every third target also exercises the gripper (0 open .. 0.015 closed)
            self.grip_toggle = (self.grip_toggle + 1) % 3
            self.target[0] = 0.015 if self.grip_toggle == 0 else 0.0
        self.n += 1
        rpy = obs.get("state.imu_rpy", obs.get("state.odom_rpy"))
        av = obs.get("state.imu_av")
        rpy = None if rpy is None else np.ravel(np.asarray(rpy, np.float64))[:3]
        av = None if av is None else np.ravel(np.asarray(av, np.float64))[:3]
        # level the hull only; no translation command (the vehicle may drift, FK is body-relative)
        lvl = attitude_pwm(rpy, av, clip=0.3)
        chunk = np.tile(lvl.reshape(1, 8), (CHUNK, 1)).astype(np.float64)
        joints = np.tile(self.target.reshape(1, 5), (CHUNK, 1)).astype(np.float64)
        return [{"action.pwm": chunk, "action.joint_pos": joints}, {}]


def make_handler(policy):
    import json_numpy

    json_numpy.patch()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"status": "healthy", "model": "ARMPROBE"}).encode()
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
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--span-scale", type=float, default=1.0, help="widen the joint box around the armed pose")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(ArmProbe(args.seed, args.span_scale)))
    print(f"[armprobe] listening on {args.port}", flush=True)
    srv.serve_forever()
