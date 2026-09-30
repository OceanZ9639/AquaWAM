"""
Grasp planner: sampling MPC over the 13-d action (8 thrusters + 5 joint targets) through the
manipulation world model (35-d state: vehicle 19 + joints 10 + object relative pose 6).

Every candidate is rolled out in imagination; the learned forward kinematics (uwam/arm_kin.py)
turns the predicted joint angles into the gripper point, the predicted object channels give the
object; the cost is the judge's own geometry:

    err_world = R_yaw(psi) (ee_body - obj_body)         |dx| < 3.5 cm, |dy| < 1 cm, |d| < 4 cm

weighted so the 1 cm lateral tolerance dominates, plus an approach shaping (stay ~6 cm above the
object until the xy alignment is inside half the tolerance, then descend), predicted spin /
control penalties and a joint-motion penalty (small, calm arm moves). The vehicle candidates come
from the mixer family the locomotion planner uses (structured, then CEM-refined); the joint
candidates are small steps around the current angles plus a damped least-squares seed that
cancels the current xy error through the FK Jacobian.  The gripper is NOT part of the search:
closing is decided separately (rule now, contact model next).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ARM_DYN_SLICES
from .control import AXIS_SIGN, attitude_pwm, sixdof_thrust_to_pwm, vel_track_pwm

JP = slice(*ARM_DYN_SLICES["joint_pos"])
OBJ = slice(*ARM_DYN_SLICES["obj"])
TOL_X, TOL_Y, TOL_D = 0.035, 0.010, 0.040        # judge tolerances (m)
ARMED = np.array([0.0, 0.4965, 0.4965, 0.5056, 0.0], np.float32)
# Joint limits measured with the probe (uwam/arm_kin.py data): every arm joint runs 0 .. 1.2 rad
# (one-sided!), the gripper 0 .. 0.015. The learned FK is valid inside this box.
JOINT_LO = np.array([0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
JOINT_HI = np.array([0.015, 1.19, 1.19, 1.20, 1.19], np.float32)
ARM_BOX = np.array([0.0, 0.35, 0.35, 0.35, 0.60], np.float32)   # global box around ARMED for the fine alignment
# per-act step box around the current angles (MoveIt executes ~1-2 targets/s; keep moves small)
JOINT_SPAN = np.array([0.0, 0.10, 0.10, 0.10, 0.20], np.float32)


@dataclass
class GraspCfg:
    n_samples: int = 128
    cem_iters: int = 2
    cem_elites: int = 16
    horizon_chunks: int = 4          # 4 x 0.5 s = 2 s of imagination: the hull needs ~2 s to respond, a
                                     # 1 s horizon cannot see its own overshoot (pilot v4 oscillated +-5-10 cm)
    joint_step_std: float = 0.02     # rad per candidate
    n_joint_hold: int = 32           # candidates that keep the arm still
    n_ik_seeds: int = 16             # candidates around the FK-Jacobian least-squares step
    hover_above: float = 0.06        # m above the object before alignment
    grasp_above: float = 0.012       # m above the object once aligned (inside |d| < 4 cm)
    align_x: float = 0.020           # alignment gate (half the judge tolerances)
    align_y: float = 0.006
    w_x: float = 1.0 / 0.02 ** 2
    w_y: float = 1.0 / 0.007 ** 2
    w_z: float = 1.0 / 0.02 ** 2
    w_spin: float = 0.5
    w_ctrl: float = 0.02
    w_joint: float = 200.0           # rad^2 penalty on joint motion per step (calm arm; MoveIt executes ~1-2 targets/s)
    kp_vehicle: float = 1.0          # body velocity goal = kp * error - kd * v (PD on the gripper error), capped
    kd_vehicle: float = 0.8          # damping on the measured body velocity (hull time constant ~2 s)
    v_cap: float = 0.08


class GraspPlanner:
    def __init__(self, model, dyn_norm, pwm_norm, arm_kin, cfg: GraspCfg | None = None, device: str = "cuda"):
        self.model, self.dyn_norm, self.pwm_norm, self.fk = model, dyn_norm, pwm_norm, arm_kin
        self.cfg = cfg or GraspCfg()
        self.device = device
        self._prev = None
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self.dm, self.ds, self.pm, self.ps = f(dyn_norm.mean), f(dyn_norm.std), f(pwm_norm.mean), f(pwm_norm.std)
        self.aligned_ticks = 0

    def reset(self):
        self._prev = None
        self.aligned_ticks = 0

    # ------------------------------------------------------------------ geometry helpers
    def ee_body(self, q5: np.ndarray) -> np.ndarray:
        return self.fk.ee_body(q5)[0]

    def _jacobian(self, q5: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """d ee_body / d q (3 x 4) for the arm joints b..e by finite differences on the learned FK."""
        J = np.zeros((3, 4), np.float64)
        base = self.ee_body(q5)
        for j in range(4):
            qq = q5.copy(); qq[1 + j] += eps
            J[:, j] = (self.ee_body(qq) - base) / eps
        return J

    def _ik_step(self, q5: np.ndarray, err_body: np.ndarray, damping: float = 0.05, max_step: float = 0.08) -> np.ndarray:
        """Damped least-squares joint step that moves the gripper by -err_body (clipped)."""
        J = self._jacobian(q5)
        dq = np.linalg.solve(J.T @ J + damping * np.eye(4), J.T @ (-err_body))
        dq = np.clip(dq, -max_step, max_step)
        q = q5.copy(); q[1:5] += dq.astype(np.float32)
        return np.clip(q, np.maximum(JOINT_LO, q5 - JOINT_SPAN), np.minimum(JOINT_HI, q5 + JOINT_SPAN))

    # ------------------------------------------------------------------ candidates
    def _candidates(self, s_t, obj_body, yaw, rpy, omega, ee_now, q_now, target_body, n, k):
        c = self.cfg
        # vehicle part: body velocity goal toward the hull displacement that would put the gripper
        # on the target (the arm can only do the last centimetres)
        err = ee_now - target_body                  # gripper - target, ODOMETRY body frame
        # the mixer / DVL convention flips sway and heave relative to the odometry body frame
        # (control.AXIS_SIGN, same as the server's _world_to_vgoal); the first planner pilot fed the
        # unflipped error and the hull fled sideways at exactly the commanded speed
        v = s_t[0:3]                                # measured DVL velocity (mixer / DVL frame)
        v_odom = AXIS_SIGN * v                      # back to the odometry body frame for the damping term
        v_goal = (AXIS_SIGN * np.clip(-c.kp_vehicle * err - c.kd_vehicle * v_odom, -c.v_cap, c.v_cap)).astype(np.float32)
        bases = [vel_track_pwm(v, v_goal, omega, rpy, kp=kp, kff=kff) for kp in (1.0, 2.0, 3.0) for kff in (1.4, 2.0)]
        # a few mirrored / scaled goals so the search is never hostage to the heuristic seed
        for scale in (0.5, -0.5):
            bases.append(vel_track_pwm(v, v_goal * scale, omega, rpy, kp=2.0, kff=2.0))
        bases.append(np.zeros(8, np.float32))
        pw = np.stack([bases[i % len(bases)] for i in range(n)])[:, None, :].repeat(k, axis=1)
        pw = pw + 0.04 * np.random.randn(*pw.shape).astype(np.float32)
        lvl = attitude_pwm(rpy, omega)
        pw = np.clip(pw + lvl[None, None, :], -1, 1)
        if self._prev is not None:
            pw[:4] = np.clip(np.vstack([self._prev[1:, :8], self._prev[-1:, :8]])[None], -1, 1)
        # joint part
        q_hold = np.repeat(q_now[None], k, axis=0)
        q_ik = self._ik_step(q_now, err)
        jc = np.zeros((n, k, 5), np.float32)
        for i in range(n):
            if i < c.n_joint_hold:
                q = q_hold
            elif i < c.n_joint_hold + c.n_ik_seeds:
                q = np.repeat((q_ik + np.r_[0, np.random.randn(4) * 0.01].astype(np.float32))[None], k, axis=0)
            else:
                step = np.r_[0, np.random.randn(4) * c.joint_step_std].astype(np.float32)
                q = np.repeat((q_now + step)[None], k, axis=0)
            jc[i] = np.clip(q, np.maximum(JOINT_LO, q_now - JOINT_SPAN), np.minimum(JOINT_HI, q_now + JOINT_SPAN))
        jc[:, :, 0] = q_now[0]   # gripper is decided outside the search
        return np.concatenate([pw, jc], axis=-1).astype(np.float32), v_goal

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _score(self, cands, hist_s, hist_a, s_t, yaw, z_above):
        c = self.cfg
        n, k, _ = cands.shape
        dev = self.device
        hs = ((torch.as_tensor(hist_s, device=dev) - self.dm) / self.ds).unsqueeze(0).repeat(n, 1, 1)
        ha = ((torch.as_tensor(hist_a, device=dev) - self.pm) / self.ps).unsqueeze(0).repeat(n, 1, 1)
        st = ((torch.as_tensor(s_t, device=dev) - self.dm) / self.ds).unsqueeze(0).repeat(n, 1)
        af = (torch.as_tensor(cands, device=dev) - self.pm) / self.ps
        d = self.model.disturbance(hs, ha)
        segs, cur = [], st
        for _ in range(c.horizon_chunks):
            s_hat, _, _ = self.model.rollout(cur, af, d)
            segs.append(s_hat)
            cur = s_hat[:, -1, :]
        sh = torch.cat(segs, 1) * self.ds + self.dm            # [n, K*chunks, 35] raw
        q = sh[..., JP]                                          # predicted joints
        pos, _ = self.fk(q.reshape(-1, 5))
        ee = pos.reshape(n, -1, 3)
        obj = sh[..., OBJ][..., :3]
        e = ee - obj                                             # body frame
        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        ex = cy * e[..., 0] - sy * e[..., 1]
        ey = sy * e[..., 0] + cy * e[..., 1]
        ez = e[..., 2] - (-z_above)                              # want the gripper z_above metres above (body z down)
        # weight later steps more (we care where we end up)
        wt = torch.linspace(0.5, 1.5, e.shape[1], device=dev)
        align = (c.w_x * ex ** 2 + c.w_y * ey ** 2 + c.w_z * ez ** 2) * wt
        spin = sh[..., 3:6].norm(dim=-1).mean(1)
        ctrl = (torch.as_tensor(cands[..., :8], device=dev) ** 2).mean((1, 2))
        dq = (torch.as_tensor(cands[..., 8:13], device=dev)[:, 0, 1:] - torch.as_tensor(s_t[JP][1:], device=dev)) ** 2
        j = align.mean(1) + c.w_spin * spin + c.w_ctrl * ctrl + c.w_joint * dq.sum(1)
        # a rollout that leaves the physically possible range is a model failure, not a plan:
        # never let such a candidate win (the first pilot picked 1e8-metre "predictions")
        insane = (obj.abs() > 5.0).any(dim=(1, 2)) | (ee.abs() > 2.0).any(dim=(1, 2)) | ~torch.isfinite(j)
        j = torch.where(insane, torch.full_like(j, 1e9), j)
        self.n_insane = int(insane.sum())
        return j.cpu().numpy(), {"ex": ex[:, -1].cpu().numpy(), "ey": ey[:, -1].cpu().numpy(), "ez": ez[:, -1].cpu().numpy()}

    # ------------------------------------------------------------------ hull pulse plan
    @torch.no_grad()
    def _score_seq(self, seqs, hist_s, hist_a, s_t, yaw, z_above):
        """Score full action sequences [n, K*C, 13] chunk by chunk (different actions per chunk)."""
        c = self.cfg
        n, T, _ = seqs.shape
        K = self.model.K
        dev = self.device
        hs = ((torch.as_tensor(hist_s, device=dev) - self.dm) / self.ds).unsqueeze(0).repeat(n, 1, 1)
        ha = ((torch.as_tensor(hist_a, device=dev) - self.pm) / self.ps).unsqueeze(0).repeat(n, 1, 1)
        st = ((torch.as_tensor(s_t, device=dev) - self.dm) / self.ds).unsqueeze(0).repeat(n, 1)
        af = (torch.as_tensor(seqs, device=dev) - self.pm) / self.ps
        d = self.model.disturbance(hs, ha)
        segs, cur = [], st
        for ci in range(T // K):
            s_hat, _, _ = self.model.rollout(cur, af[:, ci * K:(ci + 1) * K], d)
            segs.append(s_hat)
            cur = s_hat[:, -1, :]
        sh = torch.cat(segs, 1) * self.ds + self.dm
        pos, _ = self.fk(sh[..., JP].reshape(-1, 5))
        ee = pos.reshape(n, -1, 3)
        obj = sh[..., OBJ][..., :3]
        e = ee - obj
        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        ex = cy * e[..., 0] - sy * e[..., 1]
        ey = sy * e[..., 0] + cy * e[..., 1]
        ez = e[..., 2] + z_above
        # what matters is where we END (settled): the final third of the horizon, plus the final speed
        tail = slice(T - T // 3, T)
        align = (c.w_x * ex[:, tail] ** 2 + c.w_y * ey[:, tail] ** 2 + c.w_z * ez[:, tail] ** 2).mean(1)
        v_end = sh[:, -1, 0:3].norm(dim=-1)
        ctrl = (torch.as_tensor(seqs[..., :8], device=dev) ** 2).mean((1, 2))
        j = align + 400.0 * v_end ** 2 + c.w_ctrl * ctrl
        insane = (obj.abs() > 5.0).any(dim=(1, 2)) | ~torch.isfinite(j)
        j = torch.where(insane, torch.full_like(j, 1e9), j)
        return j.cpu().numpy(), {"ex": ex[:, -1].cpu().numpy(), "ey": ey[:, -1].cpu().numpy(), "ez": ez[:, -1].cpu().numpy(),
                                 "v_end": v_end.cpu().numpy(), "ex1": ex[:, K - 1].cpu().numpy(), "ey1": ey[:, K - 1].cpu().numpy(),
                                 "ez1": ez[:, K - 1].cpu().numpy()}

    def plan_pulse(self, hist_s, hist_a, s_t, obj_body, yaw, rpy=None, omega=None, z_above=None, chunks=4):
        """Imagination-optimized pulse-and-settle for the HULL (arm fixed): candidates are short
        translational thrust pulses (direction, magnitude, duration) followed by attitude-only hold;
        the world model predicts where the gripper-object offset ends 2 s later; the candidate whose
        imagined end state sits inside the judge's gate at rest wins. Returns (pwm chunk [16, 8]
        WITHOUT yaw/leveling post-processing, info)."""
        c = self.cfg
        K = self.model.K
        q_now = np.asarray(s_t[JP], np.float32)
        ee_now = self.ee_body(q_now)
        err = ee_now - obj_body                       # odometry body frame
        cy, sy = np.cos(yaw), np.sin(yaw)
        ex, ey = cy * err[0] - sy * err[1], sy * err[0] + cy * err[1]
        if z_above is None:
            z_above = c.grasp_above if (abs(ex) < c.align_x and abs(ey) < c.align_y) else c.hover_above
        target = obj_body + np.array([0, 0, -z_above], np.float32)
        disp = target - ee_now                        # desired hull displacement (odom body frame)
        omega = s_t[3:6] if omega is None else omega
        lvl = attitude_pwm(rpy, omega)
        s_t = s_t.copy(); s_t[OBJ.start:OBJ.start + 3] = obj_body; s_t[OBJ.start + 5] = 1.0
        hist_s = hist_s.copy(); hist_s[:, OBJ.start:OBJ.start + 3] = obj_body; hist_s[:, OBJ.start + 5] = 1.0
        T = K * chunks
        dirs = []
        n_disp = np.linalg.norm(disp)
        if n_disp > 1e-4:
            dirs.append(disp / n_disp)
        for ax in range(3):                            # axis-wise corrections too
            if abs(disp[ax]) > 0.004:
                u = np.zeros(3, np.float32); u[ax] = np.sign(disp[ax]); dirs.append(u)
        seqs = [np.zeros((T, 13), np.float32)]      # candidate 0: pure hold
        meta = [("hold", 0.0, 0, 0)]
        # Magnitudes start where the vehicle actually responds: measured on 45 k hover windows, a sway
        # command below 0.15 changes the velocity by < 0.4 cm/s in a second (thruster dead band + drag),
        # 0.3-0.6 gives ~4 cm/s. The hand-tuned primitive pulses (~0.1) sit in the dead zone, which is
        # why only 24 % of its episodes ever reach the alignment gate. Each pulse can be followed by a
        # reverse "brake" of half magnitude so momentum does not carry the hull through the gate.
        # The server executes exactly the first K steps (0.5 s) before re-planning, so every candidate
        # pulse fits inside K steps and the rest of the horizon is coasting: what is imagined is what
        # is executed, and momentum is accounted for. Brake candidates push against the current
        # velocity (odometry body frame: DVL frame flipped by AXIS_SIGN) so an arriving hull can stop.
        v_odom = AXIS_SIGN * np.asarray(s_t[0:3], np.float32)
        if np.linalg.norm(v_odom) > 0.015:
            dirs.append(-v_odom / np.linalg.norm(v_odom))
        for u in dirs:
            # stay where the model has data: the closed-loop pilot with 0.5-0.7 pulses moved the hull
            # 4-5x farther than imagined (out of distribution: the recordings hold smooth <0.3 commands)
            # imagination-budget ablation (WAM_PULSE_GRID): coarse = 2 magnitudes x 2 durations, full = the
            # deployed 5 x 4 grid, fine = 9 x 4
            grid = __import__("os").environ.get("WAM_PULSE_GRID", "full")
            mags = {"coarse": (0.18, 0.32), "fine": (0.12, 0.15, 0.18, 0.22, 0.25, 0.29, 0.32, 0.36, 0.4)}.get(grid, (0.12, 0.18, 0.25, 0.32, 0.4))
            durs = (2, 4) if grid == "coarse" else range(1, K)
            for mag in mags:
                # the bridge holds the LAST executed step's PWM while it waits for the next chunk, so the
                # K-th step must already be the coast (zero) step: pulses are at most K-1 steps long
                for dur in durs:
                    # mixer thrust axes coincide with the odometry body axes (the DVL-frame flip in
                    # AXIS_SIGN is undone by body_thrust_for_velocity's own flip); no sign change here
                    thr = (u * mag).astype(np.float32)
                    pwm = sixdof_thrust_to_pwm(thr, None)
                    seq = np.zeros((T, 13), np.float32)
                    seq[:dur, :8] = pwm
                    seqs.append(seq); meta.append((tuple(np.round(u, 2)), mag, dur, 0))
        seqs = np.stack(seqs)
        seqs[:, :, :8] = np.clip(seqs[:, :, :8] + lvl[None, None, :], -1, 1)
        seqs[:, :, 8:13] = q_now                     # arm fixed
        j, info = self._score_seq(seqs, hist_s, hist_a, s_t, yaw, z_above)
        i = int(np.argmin(j))
        out = np.zeros((16, 8), np.float32)
        best = seqs[i, :, :8] - lvl[None, :]         # leveling is re-added by the server's post-processing
        out[:min(16, T)] = best[:min(16, T)]
        return out, {"J": float(j[i]), "hold_J": float(j[0]), "choice": meta[i], "ex": float(ex), "ey": float(ey),
                     "ez": float(err[2] + z_above), "dist": float(np.linalg.norm(err)), "z_above": z_above,
                     "pred_ex": float(info["ex"][i]), "pred_ey": float(info["ey"][i]), "pred_ez": float(info["ez"][i]),
                     "pred_v_end": float(info["v_end"][i]), "n_cands": int(len(j)),
                     "pred1": (float(info["ex1"][i]), float(info["ey1"][i]), float(info["ez1"][i]))}

    # ------------------------------------------------------------------ arm-only plan
    def plan_arm(self, hist_s, hist_a, s_t, obj_body, yaw, pwm_seq, z_above=None, n=96):
        """Search the JOINT targets only, with the hull's PWM sequence given (the hull primitive's
        pulse-and-settle law, which reaches 4 cm reliably): the world model imagines hull + arm +
        object under that PWM for each joint candidate and the cost is the judge's geometry.
        Returns (joint_target [5], info)."""
        c = self.cfg
        q_now = np.asarray(s_t[JP], np.float32)
        ee_now = self.ee_body(q_now)
        err_now = ee_now - obj_body
        cy, sy = np.cos(yaw), np.sin(yaw)
        ex, ey = cy * err_now[0] - sy * err_now[1], sy * err_now[0] + cy * err_now[1]
        if z_above is None:
            z_above = c.grasp_above if (abs(ex) < c.align_x and abs(ey) < c.align_y) else c.hover_above
        k = self.model.K
        pwm = np.asarray(pwm_seq, np.float32)[:k]
        if len(pwm) < k:
            pwm = np.vstack([pwm, np.repeat(pwm[-1:], k - len(pwm), axis=0)])
        s_t = s_t.copy(); s_t[OBJ.start:OBJ.start + 3] = obj_body; s_t[OBJ.start + 5] = 1.0
        hist_s = hist_s.copy(); hist_s[:, OBJ.start:OBJ.start + 3] = obj_body; hist_s[:, OBJ.start + 5] = 1.0
        # per-act step box around the current angles AND a global box around the armed pose: the hull
        # standoffs and the wrist-camera view assume the arm stays near that geometry
        lo = np.maximum(np.maximum(JOINT_LO, q_now - JOINT_SPAN), ARMED - ARM_BOX)
        hi = np.minimum(np.minimum(JOINT_HI, q_now + JOINT_SPAN), ARMED + ARM_BOX)
        target = obj_body + np.array([0, 0, -z_above], np.float32)
        q_ik = self._ik_step(q_now, ee_now - target)
        jc = np.zeros((n, k, 5), np.float32)
        for i in range(n):
            if i < n // 3:
                q = q_now
            elif i < 2 * n // 3:
                q = q_ik + np.r_[0, np.random.randn(4) * 0.01].astype(np.float32)
            else:
                q = q_now + np.r_[0, np.random.randn(4) * c.joint_step_std].astype(np.float32)
            jc[i] = np.repeat(np.clip(q, lo, hi)[None], k, axis=0)
        jc[:, :, 0] = q_now[0]
        cands = np.concatenate([np.repeat(pwm[None], n, axis=0), jc], axis=-1).astype(np.float32)
        j, info = self._score(cands, hist_s, hist_a, s_t, yaw, z_above)
        i = int(np.argmin(j))
        if j[i] >= 1e9:
            return q_now.copy(), {"J": float("inf"), "ex": float(ex), "ey": float(ey), "ez": float(err_now[2]),
                                  "dist": float(np.linalg.norm(err_now)), "z_above": z_above, "ee_body": ee_now,
                                  "pred_ex": float("nan"), "pred_ey": float("nan"), "pred_ez": float("nan"), "hold_J": float("nan")}
        return cands[i, 0, 8:13].copy(), {"J": float(j[i]), "hold_J": float(j[0]), "ex": float(ex), "ey": float(ey),
                                          "ez": float(err_now[2]), "dist": float(np.linalg.norm(err_now)), "z_above": z_above,
                                          "ee_body": ee_now, "pred_ex": float(info["ex"][i]), "pred_ey": float(info["ey"][i]),
                                          "pred_ez": float(info["ez"][i])}

    # ------------------------------------------------------------------ plan
    def plan(self, hist_s, hist_a, s_t, obj_body, yaw, rpy=None, omega=None):
        """hist_s [16,35], hist_a [16,13], s_t [35] (raw); obj_body: current object estimate (3, body)
        -> (pwm_seq [K,8], joint_target [5], info)."""
        c = self.cfg
        q_now = np.asarray(s_t[JP], np.float32)
        ee_now = self.ee_body(q_now)
        err_now = ee_now - obj_body
        cy, sy = np.cos(yaw), np.sin(yaw)
        ex, ey = cy * err_now[0] - sy * err_now[1], sy * err_now[0] + cy * err_now[1]
        aligned = abs(ex) < c.align_x and abs(ey) < c.align_y
        self.aligned_ticks = self.aligned_ticks + 1 if aligned else 0
        z_above = c.grasp_above if self.aligned_ticks >= 1 else c.hover_above
        target = obj_body + np.array([0, 0, -z_above], np.float32)
        omega = s_t[3:6] if omega is None else omega
        n, k = c.n_samples, self.model.K
        # object channel in s_t / hist_s from the CURRENT estimate (perception or privileged)
        s_t = s_t.copy(); s_t[OBJ.start:OBJ.start + 3] = obj_body; s_t[OBJ.start + 5] = 1.0
        hist_s = hist_s.copy(); hist_s[:, OBJ.start:OBJ.start + 3] = obj_body; hist_s[:, OBJ.start + 5] = 1.0
        cands, v_goal = self._candidates(s_t, obj_body, yaw, rpy, omega, ee_now, q_now, target, n, k)
        j, info = self._score(cands, hist_s, hist_a, s_t, yaw, z_above)
        for _ in range(max(0, c.cem_iters - 1)):
            order = np.argsort(j)
            elites = cands[order[: c.cem_elites]]
            mu, sd = elites.mean(0), elites.std(0) + 0.01
            res = mu[None] + sd[None] * np.random.randn(max(8, n // 2), k, 13).astype(np.float32)
            res[..., :8] = np.clip(res[..., :8], -1, 1)
            res[..., 8:13] = np.clip(res[..., 8:13], np.maximum(JOINT_LO, q_now - JOINT_SPAN), np.minimum(JOINT_HI, q_now + JOINT_SPAN))
            res[..., 8] = q_now[0]
            cands = np.concatenate([elites, res], 0)
            j, info = self._score(cands, hist_s, hist_a, s_t, yaw, z_above)
        i = int(np.argmin(j))
        if j[i] >= 1e9:
            # every candidate insane -> hold still (zero thrust, joints held) and say so
            print(f"[grasp-planner] all {len(j)} imagined rollouts out of range; holding", flush=True)
            hold = np.zeros((k, 13), np.float32); hold[:, 8:13] = q_now
            return hold[:, :8], q_now.copy(), {"J": float("inf"), "ex": float(ex), "ey": float(ey), "ez": float(err_now[2]),
                                                "dist": float(np.linalg.norm(err_now)), "aligned": False, "z_above": z_above,
                                                "pred_ex": float("nan"), "pred_ey": float("nan"), "pred_ez": float("nan"),
                                                "v_goal": np.zeros(3, np.float32), "ee_body": ee_now, "n_cands": int(len(j))}
        best = cands[i]
        self._prev = best.copy()
        return best[:, :8], best[0, 8:13], {
            "J": float(j[i]), "ex": float(ex), "ey": float(ey), "ez": float(err_now[2]),
            "dist": float(np.linalg.norm(err_now)), "aligned": aligned, "z_above": z_above,
            "pred_ex": float(info["ex"][i]), "pred_ey": float(info["ey"][i]), "pred_ez": float(info["ez"][i]),
            "v_goal": v_goal, "ee_body": ee_now, "n_cands": int(len(j))}
