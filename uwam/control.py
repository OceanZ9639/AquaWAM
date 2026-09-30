"""System-ID baseline, OU exploration policy, and sampling MPC over the WAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .config import ControlCfg


def sixdof_thrust_to_pwm(thrust_xyz, thrust_rpy=None) -> np.ndarray:
    """Official BlueROV2 Heavy mixer from u0env (8 PWM in [-1, 1])."""
    t = np.asarray(thrust_xyz, dtype=np.float32).reshape(3)
    r = np.zeros(3, np.float32) if thrust_rpy is None else np.asarray(thrust_rpy, dtype=np.float32).reshape(3)
    u = np.zeros(8, np.float32)
    x, y, z = float(t[0]), float(t[1]), float(t[2])
    rr, pp, yy = float(r[0]), float(r[1]), float(r[2])
    u[0] += -x + y + yy
    u[1] += -x - y - yy
    u[2] += x + y - yy
    u[3] += x - y + yy
    u[4] += z + rr - pp
    u[5] += z - rr - pp
    u[6] += z + rr + pp
    u[7] += z - rr + pp
    return np.clip(u, -1.0, 1.0).astype(np.float32)


# Sign of d(DVL axis) / d(mixer axis command), measured on the 16200-frame OU set and
# confirmed live (thrust_x = +0.5 -> v_x = +0.25 m/s):
# corr(surge_mix, v_x) = +0.59, corr(sway_mix, v_y) = -0.62, corr(vert_mix, v_z) = -0.59.
AXIS_SIGN = np.array([1.0, -1.0, -1.0], dtype=np.float32)
THRUST_LIMIT = np.array([1.0, 1.0, 0.6], dtype=np.float32)
# DVL +z points up while Stonefish world +z points down (depth), so world-frame z errors
# must be flipped before they are treated as a DVL-frame velocity request.
WORLD_Z_TO_DVL_Z = -1.0


def body_thrust_for_velocity(v, v_goal, kp: float = 2.0, kff: float = 1.2) -> np.ndarray:
    """Body-frame thrust command for a DVL velocity goal, in mixer sign convention."""
    v = np.asarray(v, dtype=np.float32).reshape(3)
    g = np.asarray(v_goal, dtype=np.float32).reshape(3)
    raw = kp * (g - v) + kff * g
    return np.clip(AXIS_SIGN * raw, -THRUST_LIMIT, THRUST_LIMIT).astype(np.float32)


def level_rpy(rpy=None, omega=None, kp_rp: float = 1.2, kd: float = 0.25, yaw_err: float = 0.0,
              kp_yaw: float = 0.6, clip: float = 0.5) -> np.ndarray:
    """Hold the vehicle level (and yaw fixed) so body axes stay aligned with the goal frame."""
    out = np.zeros(3, np.float32)
    if rpy is not None:
        r = np.asarray(rpy, dtype=np.float32).reshape(3)
        out[0] = -kp_rp * float(r[0])
        out[1] = -kp_rp * float(r[1])
    out[2] = kp_yaw * float(yaw_err)
    if omega is not None:
        w = np.asarray(omega, dtype=np.float32).reshape(3)
        out = out - kd * w
    return np.clip(out, -clip, clip).astype(np.float32)


def attitude_pwm(rpy=None, omega=None, yaw_err: float = 0.0, kd_w: float = 0.25,
                 clip: float = 0.5) -> np.ndarray:
    """The attitude inner loop alone (IMU-driven, so it keeps running during DVL outages)."""
    return sixdof_thrust_to_pwm(np.zeros(3, np.float32),
                                level_rpy(rpy, omega, kd=kd_w, yaw_err=yaw_err, clip=clip))


def vel_track_pwm(
    v,
    v_goal,
    omega=None,
    rpy=None,
    yaw_err: float = 0.0,
    kp: float = 2.0,
    kff: float = 2.0,
    kd_w: float = 0.25,
) -> np.ndarray:
    """P + feedforward on DVL error, mapped through the BlueROV2 mixer, kept level.

    kff inverts the measured plant gain (axis_probe: thrust 0.5 -> |v| ~= 0.25, so v ~= 0.5*thrust),
    which removes the steady-state shortfall a pure P law leaves behind.
    """
    thrust = body_thrust_for_velocity(v, v_goal, kp=kp, kff=kff)
    return sixdof_thrust_to_pwm(thrust, level_rpy(rpy, omega, kd=kd_w, yaw_err=yaw_err))


def pose_home_pwm(pos_err_world, yaw: float, v, rpy=None, omega=None,
                  kp_pos: float = 0.6, kd_vel: float = 1.4, yaw_target: float = 0.0) -> np.ndarray:
    """PD station-keeping toward a world-frame waypoint; used to re-home between trials."""
    e = np.asarray(pos_err_world, dtype=np.float32).reshape(3)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    e_body = np.array(
        [c * e[0] + s * e[1], -s * e[0] + c * e[1], WORLD_Z_TO_DVL_Z * e[2]], np.float32
    )
    v = np.asarray(v, dtype=np.float32).reshape(3)
    raw = kp_pos * e_body - kd_vel * v
    thrust = np.clip(AXIS_SIGN * raw, -THRUST_LIMIT, THRUST_LIMIT)
    yaw_err = float(np.arctan2(np.sin(yaw_target - yaw), np.cos(yaw_target - yaw)))
    return sixdof_thrust_to_pwm(thrust, level_rpy(rpy, omega, yaw_err=yaw_err))


class OUProcess:
    """Ornstein–Uhlenbeck thruster exploration: x += θ(μ-x)dt + σ√dt ε."""

    def __init__(self, dim: int = 8, cfg: Optional[ControlCfg] = None):
        self.cfg = cfg or ControlCfg()
        self.dim = dim
        self.x = np.zeros(dim, dtype=np.float32)

    def reset(self, x0: Optional[np.ndarray] = None):
        self.x = np.zeros(self.dim, dtype=np.float32) if x0 is None else x0.astype(np.float32)

    def sample(self) -> np.ndarray:
        c = self.cfg
        eps = np.random.randn(self.dim).astype(np.float32)
        self.x = self.x + c.ou_theta * (c.ou_mu - self.x) * c.ou_dt + c.ou_sigma * np.sqrt(c.ou_dt) * eps
        self.x = np.clip(self.x, -1.0, 1.0)
        return self.x.copy()


class LinearSystemID:
    """v_{t+1} = A v_t + B u_t + c  least-squares baseline."""

    def __init__(self):
        self.A = np.eye(3, dtype=np.float32)
        self.B = np.zeros((3, 8), dtype=np.float32)
        self.c = np.zeros(3, dtype=np.float32)

    def fit(self, v: np.ndarray, u: np.ndarray, v_next: np.ndarray) -> "LinearSystemID":
        # [v, u, 1] @ W = v_next
        x = np.concatenate([v, u, np.ones((v.shape[0], 1), dtype=np.float32)], axis=1)
        w, *_ = np.linalg.lstsq(x, v_next, rcond=None)
        self.A = w[:3].T.astype(np.float32)
        self.B = w[3:11].T.astype(np.float32)
        self.c = w[11].astype(np.float32)
        return self

    def predict(self, v: np.ndarray, u: np.ndarray) -> np.ndarray:
        return (self.A @ v + self.B @ u + self.c).astype(np.float32)

    def closed_loop_pwm(self, v: np.ndarray, v_goal: np.ndarray, kp: float = 2.0, omega=None,
                        rpy=None, yaw_err: float = 0.0) -> np.ndarray:
        """Thrust-allocation P+FF, lightly pulled toward the LS inverse of B.

        The attitude inner loop is added on top: the LS solve knows nothing about roll/pitch,
        and without it the vehicle slowly tips until the body-frame goal stops meaning anything.
        """
        u0 = vel_track_pwm(v, v_goal, omega=omega, rpy=rpy, kp=kp, yaw_err=yaw_err)
        v_des = np.asarray(v, np.float32) + np.clip(np.asarray(v_goal, np.float32) - v, -0.4, 0.4)
        rhs = v_des - self.A @ v - self.c
        bt = self.B
        lam = 0.35
        gram = bt.T @ bt + lam * np.eye(8, dtype=np.float32)
        u = np.linalg.solve(gram, bt.T @ rhs + lam * u0)
        u = u + attitude_pwm(rpy, omega, yaw_err=yaw_err)
        return np.clip(u, -1.0, 1.0).astype(np.float32)


class RecursiveSystemID(LinearSystemID):
    """RLS-adaptive variant: re-fits v_{t+1} = A v_t + B u_t + c online with forgetting.

    The strongest classical baseline for regime changes; it can only adapt while velocity
    feedback exists, which is exactly the contrast the blackout experiments are about.
    """

    def __init__(self, lam: float = 0.995, p0: float = 10.0):
        super().__init__()
        self.lam = float(lam)
        self.P = np.eye(12, dtype=np.float32) * p0
        self.W = np.zeros((12, 3), np.float32)  # rows: v(3), u(8), 1

    def sync_from(self, sid: LinearSystemID) -> "RecursiveSystemID":
        self.W[0:3] = sid.A.T
        self.W[3:11] = sid.B.T
        self.W[11] = sid.c
        self._pull()
        return self

    def _pull(self):
        self.A = self.W[0:3].T.astype(np.float32)
        self.B = self.W[3:11].T.astype(np.float32)
        self.c = self.W[11].astype(np.float32)

    def update(self, v_prev: np.ndarray, u_prev: np.ndarray, v_now: np.ndarray) -> None:
        x = np.concatenate([np.asarray(v_prev, np.float32).reshape(3),
                            np.asarray(u_prev, np.float32).reshape(8),
                            [1.0]]).astype(np.float32)
        Px = self.P @ x
        k = Px / (self.lam + float(x @ Px))
        err = np.asarray(v_now, np.float32).reshape(3) - x @ self.W
        self.W = self.W + np.outer(k, err)
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self._pull()


class SamplingMPC:
    """
    Sample N candidate PWM sequences (from an action library + noise),
    rollout the WAM, pick argmin J.
    """

    def __init__(self, model, dyn_norm, pwm_norm, cfg: Optional[ControlCfg] = None, device: str = "cuda"):
        self.model = model
        self.dyn_norm = dyn_norm
        self.pwm_norm = pwm_norm
        self.cfg = cfg or ControlCfg()
        self.device = device
        self.library: Optional[np.ndarray] = None  # [M, 8] typical PWM vectors
        self._prev_seq: Optional[np.ndarray] = None  # warm start, cleared per trial
        self.dt_s: float = 0.1  # sample period fed to dt-conditioned models
        self._vel_ens = None  # optional NLL ensemble of dead-reckoning heads
        # Deployment canonicalization of scene-level channels for the ESTIMATOR
        # only: absolute pressure / altitude carry no velocity information, but
        # the training mixture had them in inconsistent units (USIM 1e4 Pa vs
        # OU raw Pa), so unseen scene depths push the heads far out of
        # distribution (measured: v_est ~ 0.2 v_true on eval scenes). Pinning the
        # columns to one training-typical value removes the scene dependence.
        self.canon_pressure: Optional[float] = None
        self.canon_alt: Optional[float] = None

    def _canon(self, hist_s: np.ndarray) -> np.ndarray:
        if self.canon_pressure is None and self.canon_alt is None:
            return hist_s
        h = np.array(hist_s, dtype=np.float32, copy=True)
        if self.canon_pressure is not None:
            h[:, 9] = self.canon_pressure
        if self.canon_alt is not None:
            h[:, 10] = self.canon_alt
        return h

    def reset(self) -> None:
        self._prev_seq = None

    def _dt_feat(self, n: int):
        import torch

        if not getattr(self.model, "use_dt", False):
            return None
        val = self.model.dt_feature(torch.full((n,), float(self.dt_s)))
        return val.to(self.device)

    def set_library(self, pwms: np.ndarray) -> None:
        self.library = pwms.astype(np.float32)

    def estimate_velocity(self, hist_s: np.ndarray, hist_a: np.ndarray):
        """Dead-reckoned DVL velocity (raw units) from a DVL-free history, or None if untrained."""
        import torch

        if not hasattr(self.model, "estimate_velocity"):
            return None
        with torch.no_grad():
            hs = torch.from_numpy(self.dyn_norm(self._canon(hist_s))).unsqueeze(0).to(self.device)
            ha = torch.from_numpy(self.pwm_norm(hist_a)).unsqueeze(0).to(self.device)
            v_n = self.model.estimate_velocity(
                self.model.mask_dvl(hs), ha, dt_feat=self._dt_feat(1)
            )[0].cpu().numpy()
        return (v_n * self.dyn_norm.std[0:3] + self.dyn_norm.mean[0:3]).astype(np.float32)

    def load_vel_ensemble(self, path) -> bool:
        """Attach the heteroscedastic dead-reckoning ensemble (trained separately)."""
        import torch

        from .models import MLP

        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            return False
        from .config import Cfg

        default_hidden = Cfg().model.hidden
        heads = []
        for sd in ck["heads"]:
            h = MLP(ck["in_dim"], 6, ck.get("hidden", default_hidden), depth=ck.get("depth", 3),
                    dropout=ck.get("dropout", 0.0)).to(self.device)
            h.load_state_dict(sd)
            h.eval()
            heads.append(h)
        self._vel_ens = heads
        # heads trained on canonicalized inputs must see canonicalized inputs
        self.canon_pressure = ck.get("canon_pressure")
        self.canon_alt = ck.get("canon_alt")
        return True

    def estimate_velocity_ens(self, hist_s: np.ndarray, hist_a: np.ndarray):
        """(v_raw, sigma_alea_norm, sigma_epi_norm): ensemble mean in raw m/s plus the
        aleatoric / epistemic decomposition in normalized units. Aleatoric noise is
        i.i.d. per tick (averages out under smoothing); epistemic disagreement is a
        systematic offset (it does not). None when the ensemble is not loaded."""
        import torch

        if self._vel_ens is None:
            return None
        with torch.no_grad():
            hs = torch.from_numpy(self.dyn_norm(self._canon(hist_s))).unsqueeze(0).to(self.device)
            ha = torch.from_numpy(self.pwm_norm(hist_a)).unsqueeze(0).to(self.device)
            hm = self.model.mask_dvl(hs)
            d = self.model.disturbance(hm, ha, dt_feat=self._dt_feat(1))
            x = torch.cat([torch.cat([hm, ha], dim=-1).flatten(1), d], dim=1)
            # The heads were fitted on ONE core's disturbance token; a core of another width
            # (scale ablation) changes the input size. Fall back to the core's own head instead
            # of crashing the act loop.
            in_dim = getattr(self._vel_ens[0].net[0], "in_features", x.shape[1])
            if x.shape[1] != in_dim:
                print(f"[mpc] vel ensemble input {in_dim} != core features {x.shape[1]}; ensemble disabled "
                      "(estimator = the core's own head)", flush=True)
                self._vel_ens = None
                return None
            outs = torch.stack([h(x) for h in self._vel_ens], 0)  # [K, 1, 6]
            mus, vars_ = outs[:, :, :3], outs[:, :, 3:].clamp(-8, 4).exp()
            mu = mus.mean(0)
            sd_alea = vars_.mean(0).clamp_min(1e-8).sqrt()[0].cpu().numpy()
            sd_epi = mus.var(0, unbiased=False).clamp_min(1e-12).sqrt()[0].cpu().numpy()
            mu_n = mu[0].cpu().numpy()
        v_raw = (mu_n * self.dyn_norm.std[0:3] + self.dyn_norm.mean[0:3]).astype(np.float32)
        return v_raw, sd_alea.astype(np.float32), sd_epi.astype(np.float32)

    def _repeat(self, u: np.ndarray, k: int, n: int, std: float) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, k, 8), np.float32)
        base = np.repeat(np.asarray(u, np.float32).reshape(1, 1, 8), k, axis=1)
        noise = std * np.random.randn(n, k, 8).astype(np.float32)
        return np.clip(base + noise, -1.0, 1.0)

    def _mixer_family(self, v, v_goal, omega, rpy, n: int, k: int, yaw_err: float = 0.0) -> np.ndarray:
        """Structured candidates: the mixer P+FF law over a gain grid, plus per-axis primitives."""
        c = self.cfg
        if n <= 0:
            return np.zeros((0, k, 8), np.float32)
        std = float(getattr(c, "mixer_std", 0.05))
        lvl = level_rpy(rpy, omega, kd=c.mixer_kd_w, yaw_err=yaw_err)
        bases = [vel_track_pwm(v, v_goal, omega, rpy, yaw_err=yaw_err,
                               kp=c.mixer_kp, kff=c.mixer_kff, kd_w=c.mixer_kd_w)]
        for kp in (1.0, 2.0, 3.5):
            for kff in (1.4, 2.0, 2.6):
                bases.append(vel_track_pwm(v, v_goal, omega, rpy, yaw_err=yaw_err,
                                           kp=kp, kff=kff, kd_w=c.mixer_kd_w))
        g = np.asarray(v_goal, np.float32).reshape(3)
        for axis in range(3):
            sgn = 1.0 if abs(float(g[axis])) < 1e-6 else float(np.sign(g[axis]))
            for mag in (0.3, 0.5, 0.8):
                t = np.zeros(3, np.float32)
                t[axis] = float(np.clip(AXIS_SIGN[axis] * mag * sgn, -THRUST_LIMIT[axis], THRUST_LIMIT[axis]))
                bases.append(sixdof_thrust_to_pwm(t, lvl))
        idx = [i % len(bases) for i in range(n)]
        out = np.stack([np.repeat(bases[i].reshape(1, 8), k, axis=0) for i in idx], axis=0)
        out = out + std * np.random.randn(*out.shape).astype(np.float32)
        return np.clip(out, -1.0, 1.0)

    def _candidates(
        self,
        n: int,
        k: int,
        sid_u: Optional[np.ndarray] = None,
        last_u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        v_goal: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        rpy: Optional[np.ndarray] = None,
        yaw_err: float = 0.0,
    ) -> np.ndarray:
        c = self.cfg
        parts = []
        used = 0
        n_warm = 0
        if self._prev_seq is not None:
            shifted = np.vstack([self._prev_seq[1:], self._prev_seq[-1:]])
            n_warm = min(int(getattr(c, "n_warm_seeds", 8)), n)
            noise = float(getattr(c, "mixer_std", 0.05)) * np.random.randn(n_warm, k, 8).astype(np.float32)
            parts.append(np.clip(shifted.reshape(1, k, 8) + noise, -1.0, 1.0))
            used += n_warm
        n_mix = min(int(getattr(c, "n_mixer_seeds", 40)), n - used) if v is not None and v_goal is not None else 0
        if n_mix > 0:
            parts.append(self._mixer_family(v, v_goal, omega, rpy, n_mix, k, yaw_err=yaw_err))
            used += n_mix
        n_sid = min(int(getattr(c, "n_sysid_seeds", 16)), n - used) if sid_u is not None else 0
        if n_sid > 0:
            parts.append(self._repeat(sid_u, k, n_sid, float(getattr(c, "mixer_std", 0.05))))
            used += n_sid
        n_hold = min(int(getattr(c, "n_hold_seeds", 8)), n - used) if last_u is not None else 0
        if n_hold > 0:
            parts.append(self._repeat(last_u, k, n_hold, float(getattr(c, "mixer_std", 0.05))))
            used += n_hold
        n_rest = n - used
        if n_rest > 0:
            lib = self.library
            std = float(c.action_std)
            if lib is not None and len(lib) >= k:
                starts = np.random.randint(0, len(lib) - k + 1, size=n_rest)
                seq = np.stack([lib[s : s + k] for s in starts], axis=0)
                seq = seq + std * np.random.randn(*seq.shape).astype(np.float32)
            else:
                seq = np.random.uniform(-1.0, 1.0, size=(n_rest, k, 8)).astype(np.float32)
            parts.append(np.clip(seq, -1.0, 1.0))
        cands = np.concatenate(parts, axis=0) if parts else np.zeros((n, k, 8), np.float32)
        # Raw OU snippets carry arbitrary roll/pitch moments; without the attitude inner loop on
        # every candidate the vehicle tumbles and the body-frame goal loses meaning.
        lvl = attitude_pwm(rpy, omega, yaw_err=yaw_err)
        return np.clip(cands + lvl.reshape(1, 1, 8), -1.0, 1.0)

    def plan(
        self,
        hist_s: np.ndarray,
        hist_a: np.ndarray,
        s_t: np.ndarray,
        v_goal: np.ndarray,
        task_index: int = 0,
        sid_u: Optional[np.ndarray] = None,
        rpy: Optional[np.ndarray] = None,
        yaw_err: float = 0.0,
    ) -> Tuple[np.ndarray, dict]:
        c = self.cfg
        n, k = c.n_samples, c.horizon
        last_u = hist_a[-1] if hist_a is not None and len(hist_a) else np.zeros(8, np.float32)
        cands = self._candidates(
            n, k, sid_u=sid_u, last_u=last_u, v=s_t[:3], v_goal=v_goal,
            omega=s_t[3:6], rpy=rpy, yaw_err=yaw_err,
        )
        j, v_err, spin = self._score(cands, hist_s, hist_a, s_t, v_goal, task_index)
        # CEM refinement: refit a Gaussian on the elites and resample around them; the structured
        # candidates provide global coverage, the CEM rounds do the local polish
        for _ in range(max(0, int(getattr(c, "cem_iters", 1)) - 1)):
            order = np.argsort(j)
            elites = cands[order[: int(getattr(c, "cem_elites", 16))]]
            mu = elites.mean(axis=0)
            sd = elites.std(axis=0) + 0.02
            n_res = max(8, n // 2)
            resamp = np.clip(
                mu.reshape(1, k, 8) + sd.reshape(1, k, 8) * np.random.randn(n_res, k, 8).astype(np.float32),
                -1.0, 1.0,
            )
            cands = np.concatenate([elites, resamp], axis=0)
            j, v_err, spin = self._score(cands, hist_s, hist_a, s_t, v_goal, task_index)
        i = int(np.argmin(j))
        self._prev_seq = cands[i].copy()
        return cands[i, 0], {"J": j, "best": i, "v_err": v_err[i], "spin": spin[i], "seq": cands[i]}

    def _score(self, cands: np.ndarray, hist_s: np.ndarray, hist_a: np.ndarray,
               s_t: np.ndarray, v_goal: np.ndarray, task_index: int):
        import torch

        c = self.cfg
        n = cands.shape[0]
        device = self.device
        # The model's native horizon is 0.5 s, but the vehicle needs ~2 s to reach 0.25 m/s, so a
        # single chunk leaves every candidate equally far from the goal and the ranking is noise.
        chunks = max(1, int(getattr(c, "horizon_chunks", 1)))
        with torch.no_grad():
            hs = torch.from_numpy(self.dyn_norm(hist_s)).unsqueeze(0).repeat(n, 1, 1).to(device)
            ha = torch.from_numpy(self.pwm_norm(hist_a)).unsqueeze(0).repeat(n, 1, 1).to(device)
            st = torch.from_numpy(self.dyn_norm(s_t)).unsqueeze(0).repeat(n, 1).to(device)
            af = torch.from_numpy(np.stack([self.pwm_norm(cands[i]) for i in range(n)])).to(device)
            dtf = self._dt_feat(n)
            d = self.model.disturbance(hs, ha, dt_feat=dtf) if dtf is not None \
                else self.model.disturbance(hs, ha)
            task = torch.full((n,), task_index, device=device, dtype=torch.long)
            segs = []
            cur = st
            for _ in range(chunks):
                s_hat, _, _ = self.model.rollout(cur, af, d, task, dt_feat=dtf)
                segs.append(s_hat)
                cur = s_hat[:, -1, :]
            s_hat_np = torch.cat(segs, dim=1).cpu().numpy()
        inv = np.stack([self.dyn_norm.invert(s_hat_np[i]) for i in range(n)])
        dvl_hat = inv[:, :, 0:3]
        omega_hat = inv[:, :, 3:6]
        # score every horizon step, not only the last
        v_err = np.linalg.norm(dvl_hat - v_goal[None, None, :], axis=-1).mean(axis=1)
        ctrl = (cands ** 2).mean(axis=(1, 2))
        safety = np.clip(np.linalg.norm(dvl_hat, axis=-1).max(axis=-1) - 1.5, 0, None)
        # penalise candidates the model expects to spin the vehicle up
        spin = np.linalg.norm(omega_hat, axis=-1).mean(axis=1)
        j = c.goal_w * v_err + c.control_w * ctrl + c.safety_w * safety + c.att_w * spin
        return j, v_err, spin
