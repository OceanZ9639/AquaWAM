"""Visual-inertial goal filter: fuse the vision head's per-frame goal predictions over the last few
frames using the vehicle's own pose (odometry, or dead reckoning when the DVL is out), so that the
vehicle's rotation and translation between frames are removed before averaging.

What the head gives (measured offline): a good BEARING (median 5-6 deg beyond 0.5 m) with 10-15 deg
per-frame scatter on the deployed arm's own trajectories, a usable RANGE only inside ~2 m, and a
range that is biased SHORT far away (2-3 m predicted at 6-9 m). The last point rules out averaging
world POINTS over time: past points sit behind the true goal and, once the vehicle has moved on,
behind the vehicle itself (a point-fusion prototype steered backwards on recorded episodes). So the
filter fuses what is actually consistent across frames:
  * the world-frame DIRECTION to the goal (weighted circular mean of the recent rays; recency weighted)
  * the RANGE only from the most recent frames and only when they agree; otherwise the newest frame
A direction jump beyond the gate for K consecutive frames = the expert's node advanced: restart.

The same object gives a visual fix during a DVL outage: the goal estimated while sighted is a
landmark; re-observing it blind at close range yields the vehicle position that explains it.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _rot(yaw: float, d_body: np.ndarray) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([cy * d_body[0] - sy * d_body[1], sy * d_body[0] + cy * d_body[1], d_body[2]])


class GoalFilter:
    def __init__(self, n_keep: int = 6, reset_k: int = 3, gate_deg: float = 35.0, age_decay: float = 0.75):
        self.n_keep = n_keep
        self.reset_k = reset_k
        self.gate = np.radians(gate_deg)
        self.age_decay = age_decay
        self.reset()

    def reset(self) -> None:
        self.rays: deque = deque(maxlen=self.n_keep)   # (p_i, u_i, r_i): pose, world unit direction, range
        self.x: np.ndarray | None = None
        self.pending: list = []
        self.n = 0

    def _solve(self, p_now: np.ndarray) -> np.ndarray:
        m = len(self.rays)
        acc = np.zeros(3)
        for i, (p_i, u_i, r_i) in enumerate(self.rays):
            w = self.age_decay ** (m - 1 - i)
            # re-express ray i from the current pose: direction to the point it observed
            pt = p_i + r_i * u_i
            v = pt - p_now
            nv = float(np.linalg.norm(v))
            # far observations: their range is biased short, so use their DIRECTION as seen from
            # where they were taken (parallel transport); close ones: direction to their point
            acc += w * (u_i if (r_i > 2.0 or nv < 0.3) else v / nv)
        u = acc / max(float(np.linalg.norm(acc)), 1e-9)
        # range: newest frames only, and only if they agree
        recent = [r for _, _, r in list(self.rays)[-3:]]
        r_new = recent[-1]
        if len(recent) >= 2 and (max(recent) - min(recent)) < 0.3 * max(r_new, 0.5):
            r_use = float(np.median(recent))
        else:
            r_use = r_new
        return p_now + r_use * u

    def update(self, pos, yaw: float, d_body) -> np.ndarray:
        """Fuse one body-frame goal prediction made at (pos, yaw); returns the world-frame estimate."""
        p = np.asarray(pos, np.float64).reshape(3)
        d = np.asarray(d_body, np.float64).reshape(3)
        d_w = _rot(yaw, d)
        r = float(np.linalg.norm(d_w))
        if r < 1e-3:
            return (self.x.copy() if self.x is not None else p.copy())
        u = d_w / r
        if self.x is not None and self.n >= 2:
            v = self.x - p
            nv = float(np.linalg.norm(v))
            ang = float(np.arccos(np.clip(u @ (v / nv), -1.0, 1.0))) if nv > 0.3 else 0.0
            if ang > self.gate:
                self.pending.append((p.copy(), u.copy(), r))
                if len(self.pending) >= self.reset_k:
                    pend = self.pending
                    self.reset()
                    for tup in pend:
                        self.rays.append(tup)
                    self.n = len(pend)
                    self.x = self._solve(p)
                return self.x.copy()
            self.pending = []
        self.rays.append((p.copy(), u.copy(), r))
        self.n += 1
        self.x = self._solve(p)
        return self.x.copy()

    def position_fix(self, landmark_w, yaw: float, d_body, gate_m: float = 1.5, pos_guess=None):
        """Vehicle position implied by observing `landmark_w` at body-frame offset d_body, or None if
        the observation is not consistent with the landmark (pos_guess needed for the gate)."""
        d = np.asarray(d_body, np.float64).reshape(3)
        fix = np.asarray(landmark_w, np.float64) - _rot(yaw, d)
        if pos_guess is not None and np.linalg.norm(fix - np.asarray(pos_guess, np.float64)) > gate_m:
            return None
        return fix
