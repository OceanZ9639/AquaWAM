#!/usr/bin/env python3
"""U0 + WAM fallback-layer server.

Same /act HTTP contract as every other policy server. Arbitration:

  - DVL valid (full sensing): pass the U0 action chunk through untouched, while
    the WAM side only OBSERVES (updates its history window, sighted estimator
    stream and hold anchor). By construction the fallback arm is identical to
    U0 whenever nothing is wrong.
  - DVL dropped: the outage is exogenous, directly observed evidence (a real
    DVL reports loss of bottom-lock), so the takeover latches -- same logic as
    the goal-change latch in the underwater stack. Inside the blind phase the
    calibrated CUSUM gate keeps arbitrating hold vs replanning exactly as in
    the standalone WAM arm. Arm joints hold their current positions (safe mode).

Run under the SYSTEM python. The U0 service runs separately in the GR00T venv.

Usage:
  python3 u0eval/fallback_policy_server.py --port 8002 --u0-url http://127.0.0.1:8001/act
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wam_policy_server import CHUNK, N_JOINTS, WamPolicy  # noqa: E402


class FallbackPolicy:
    def __init__(self, args):
        import concurrent.futures
        import requests

        self.requests = requests
        self.u0_url = args.u0_url
        self.wam = WamPolicy(args)
        self.was_blind = False
        self.blind_joints = args.blind_joints
        # Full sensing must cost exactly what U0 costs, otherwise the arm is not
        # "U0 when the gate is closed": running the WAM observation inline added
        # its forward pass to every tick and broke the 10 Hz chunk cadence
        # (measured: fallback 65/80 vs U0 71/80 on locomotion). The observation
        # produces no action, so it runs on a background worker instead; torch
        # CUDA ops and the U0 HTTP wait both release the GIL, so they overlap.
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._obs_future = None
        self.n_obs_done = 0
        self.n_obs_skipped = 0

    def _observe_async(self, obs):
        """Queue the WAM sighted-phase observation off the response path.

        A tick is dropped when the worker is still busy: the estimator keeps a
        5-sample stream and the hold anchor is rebuilt from the observation's own
        pwm history, so an occasional gap costs nothing, while a blocked response
        would cost a control chunk.
        """
        if self._obs_future is not None and not self._obs_future.done():
            self.n_obs_skipped += 1
            return
        self.n_obs_done += 1
        if self.n_obs_done % 50 == 1:
            print(f"[fallback] observations done={self.n_obs_done} "
                  f"skipped={self.n_obs_skipped}", flush=True)
        self._obs_future = self._pool.submit(self._observe_body, obs)

    def _observe_body(self, obs):
        try:
            self.wam.observe_sighted(obs)
        except Exception as e:  # noqa: BLE001
            print(f"[fallback] background observe failed: {e}", flush=True)

    def _wait_observe(self):
        """Before a takeover, let any in-flight observation finish so the gate
        anchor and waypoint index reflect the last sighted tick."""
        f = self._obs_future
        if f is not None and not f.done():
            try:
                f.result(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass

    def _u0_act(self, obs):
        import json_numpy  # noqa: F401  (patched at import in server main)

        r = self.requests.post(self.u0_url, json={"observation": obs}, timeout=30)
        r.raise_for_status()
        result = r.json()
        return result[0] if isinstance(result, (list, tuple)) and len(result) >= 1 else result

    def act(self, obs):
        dvl_valid = obs.get("meta.dvl_valid")
        blind = dvl_valid is not None and float(np.ravel(np.asarray(dvl_valid))[0]) < 0.5
        if not blind:
            if self.was_blind:
                print("[fallback] DVL restored -> control back to U0", flush=True)
                self.was_blind = False
            self._observe_async(obs)
            try:
                return [self._u0_act(obs), {"arm": "u0"}]
            except Exception as e:  # noqa: BLE001
                # VLA service down is itself a takeover trigger (fail-operational)
                print(f"[fallback] U0 unreachable ({e}) -> WAM takes over", flush=True)
                return self.wam.act(obs)
        # Mode-aware takeover: the VLA's manipulation is wrist-camera visual servoing and does not
        # consume the DVL, while WAM's grasp primitive is the weaker controller. Measured on the
        # paper-scale blocks: naive takeover on grasp tasks under dropout 11/160 (7 %), U0 left in
        # charge 30-57 %. So the takeover is taken only when the task's competence depends on the
        # lost sensor (locomotion); for grasp / transfer the VLA keeps every channel and WAM stays
        # a silent monitor (observing, ready for the fail-operational trigger above).
        if getattr(self.wam, "mode", "nav") in ("grasp", "transfer"):
            if not self.was_blind:
                print(f"[fallback] DVL DROPOUT observed in {self.wam.mode}: manipulation does not "
                      "depend on the DVL -> U0 keeps control (mode-aware fallback)", flush=True)
                self.was_blind = True
            self._observe_async(obs)
            try:
                return [self._u0_act(obs), {"arm": "u0", "mode_aware": True}]
            except Exception as e:  # noqa: BLE001
                print(f"[fallback] U0 unreachable ({e}) -> WAM takes over", flush=True)
                return self.wam.act(obs)
        if not self.was_blind:
            self._wait_observe()
            print(f"[fallback] DVL DROPOUT observed -> WAM takeover latched "
                  f"(joints={self.blind_joints}; observations done="
                  f"{self.n_obs_done} skipped={self.n_obs_skipped})", flush=True)
            self.was_blind = True
        action = self.wam.blind_act(obs)
        joints_done = False
        if self.blind_joints == "u0":
            # split-channel fallback: WAM drives the thrusters (the channel that
            # depends on the dead sensor); the VLA keeps commanding the arm, whose
            # wrist-camera manipulation does not consume DVL
            try:
                u0_action = self._u0_act(obs)
                action[0]["action.joint_pos"] = np.asarray(u0_action["action.joint_pos"])
                joints_done = True
            except Exception as e:  # noqa: BLE001
                print(f"[fallback] U0 joints unavailable ({e}); holding joints", flush=True)
        if not joints_done:
            jp = obs.get("state.joint_pos")
            if jp is not None:
                j = np.asarray(jp, np.float64).reshape(-1)[:N_JOINTS]
                action[0]["action.joint_pos"] = np.tile(j[None], (CHUNK, 1))
        action[1]["arm"] = "wam_fallback"
        action[1]["alpha"] = float(self.wam.alpha)
        return action


def make_handler(policy):
    import json_numpy

    json_numpy.patch()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({"status": "healthy", "model": "U0+WAM-fallback"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/act":
                self.send_response(404)
                self.end_headers()
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n).decode())
                if "encoded" in payload:
                    payload = json.loads(payload["encoded"])
                t0 = time.time()
                action = policy.act(payload["observation"])
                dt_ms = (time.time() - t0) * 1e3
                if dt_ms > 900:
                    print(f"[fallback] slow act: {dt_ms:.0f} ms", flush=True)
                body = json.dumps(action).encode()
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
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--u0-url", default="http://127.0.0.1:8001/act")
    ap.add_argument("--blind-joints", choices=["hold", "u0"], default="u0",
                    help="arm joints during dropout: hold current pose, or keep the "
                         "VLA's joint commands (split-channel fallback, default)")
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_scenes.pt")
    ap.add_argument("--vel-ens", default="/hy-tmp/models/uwam/vel_ens_scenes.pt")
    ap.add_argument("--gate-calib", default="/hy-tmp/models/uwam/gate_calib.json")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore")
    ap.add_argument("--vmax-scale", type=float, default=1.0)
    ap.add_argument("--eval-root", default="/hy-tmp/u0env/dataset/eval")
    ap.add_argument("--hold-anchor", choices=["mean", "median", "mixer"], default="mixer")
    ap.add_argument("--dump-dir", default="",
                    help="forwarded to WAM; empty = no dump (U0 is in charge while sighted)")
    args = ap.parse_args()

    policy = FallbackPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"[fallback] listening on {args.host}:{args.port}, upstream U0: {args.u0_url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
