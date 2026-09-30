"""
WAM-direct: an amortized action head on top of the frozen dynamics core.

The sampling MPC (control.SamplingMPC) answers "which 0.5 s PWM sequence drives the
predicted DVL velocity to v_goal" by scoring hundreds of imagined rollouts every act.
This module answers the same question with one forward pass:

    a_{t:t+K} = pi_theta( d_t, s_t, v_goal, a_{t-1} )

where d_t is the core's history/disturbance token (so the head sees the same
embodiment evidence -- thruster wear, currents -- the planner does).  The core then
rolls a_{t:t+K} out to s_{t+1:t+K}: the pair (s_hat, a_hat) comes out of one pass
through one model, which is the DreamZero "world action model" reading (actions and
the futures they cause, aligned by construction); the sampling MPC becomes an optional
refinement (server flag --action-source direct_cem).

Training (scripts/train_direct.py) = distillation from a *larger* offline teacher
(512 samples x 3 CEM rounds vs. 128 x 2 online) + the planner's own objective
evaluated through the frozen core, so the head is pulled toward the teacher's
choices and toward low imagined cost at the same time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Cfg, enable_arm, enable_object
from .data import RunningNorm
from .models import MLP, DynamicsWAM

V_SCALE = 0.5  # v_goal is fed as v / V_SCALE (nav cruise 0.45 m/s -> ~0.9)


def load_core(ckpt_path, device: str = "cuda"):
    """Rebuild a stage-1 core from its checkpoint (same rules as the policy server)."""
    cfg = Cfg()
    cfg.model.use_language = False
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck.get("cfg", {})
    if "disturbance_dim" in c:
        cfg.model.disturbance_dim = int(c["disturbance_dim"])
    if "hidden" in c:
        cfg.model.hidden = int(c["hidden"])
    cfg.model.use_dt = bool(c.get("use_dt", False))
    cfg.model.no_token = bool(c.get("no_token", False))
    if c.get("use_object"):
        enable_object(cfg)
    elif c.get("use_arm"):
        enable_arm(cfg)
    model = DynamicsWAM(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dyn_norm, pwm_norm = RunningNorm(), RunningNorm()
    dyn_norm.load_state_dict(ck["dyn_norm"])
    pwm_norm.load_state_dict(ck["pwm_norm"])
    return model, dyn_norm, pwm_norm, cfg


class DirectHead(nn.Module):
    """(d_t, s_t normalized, v_goal / V_SCALE, a_{t-1}) -> K x 8 raw PWM in [-1, 1]."""

    def __init__(self, dist_dim: int, state_dim: int = 19, K: int = 5, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.K = K
        self.net = MLP(dist_dim + state_dim + 3 + 8, K * 8, hidden, depth=depth)

    def forward(self, d, s_n, g, last_u):
        x = torch.cat([d, s_n, g, last_u], dim=-1)
        return torch.tanh(self.net(x)).view(-1, self.K, 8)


class ImaginedCost:
    """Differentiable copy of SamplingMPC._score (same weights, same 4-chunk chaining)."""

    def __init__(self, model, dyn_norm, pwm_norm, cfg, device):
        self.model, self.cfg, self.device = model, cfg.control, device
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self.dm, self.ds = f(dyn_norm.mean), f(dyn_norm.std)
        self.pm, self.ps = f(pwm_norm.mean), f(pwm_norm.std)

    def norm_s(self, s):
        return (s - self.dm) / self.ds

    def norm_a(self, a):
        return (a - self.pm) / self.ps

    def __call__(self, hist_s_raw, hist_a_raw, s_t_raw, v_goal, seq_raw, d=None):
        c = self.cfg
        hs, ha, st = self.norm_s(hist_s_raw), self.norm_a(hist_a_raw), self.norm_s(s_t_raw)
        if d is None:
            d = self.model.disturbance(hs, ha)
        af = self.norm_a(seq_raw)
        segs, cur = [], st
        for _ in range(max(1, int(getattr(c, "horizon_chunks", 1)))):
            s_hat, _, _ = self.model.rollout(cur, af, d)
            segs.append(s_hat)
            cur = s_hat[:, -1, :]
        s_hat = torch.cat(segs, dim=1) * self.ds + self.dm
        dvl_hat, omega_hat = s_hat[..., 0:3], s_hat[..., 3:6]
        v_err = (dvl_hat - v_goal[:, None, :]).norm(dim=-1).mean(dim=1)
        ctrl = (seq_raw ** 2).mean(dim=(1, 2))
        safety = torch.relu(dvl_hat.norm(dim=-1).max(dim=-1).values - 1.5)
        spin = omega_hat.norm(dim=-1).mean(dim=1)
        j = c.goal_w * v_err + c.control_w * ctrl + c.safety_w * safety + c.att_w * spin
        return j, {"v_err": v_err, "spin": spin, "s_hat": s_hat}


class DirectPolicy:
    """Deployment wrapper: same call signature the server uses for SamplingMPC.plan."""

    def __init__(self, head: DirectHead, model, dyn_norm, pwm_norm, device: str = "cuda"):
        self.head, self.model, self.device = head.to(device).eval(), model, device
        self.dyn_norm, self.pwm_norm = dyn_norm, pwm_norm
        self.n_params = sum(p.numel() for p in head.parameters())

    @classmethod
    def load(cls, path, model, dyn_norm, pwm_norm, device: str = "cuda") -> "DirectPolicy":
        ck = torch.load(path, map_location=device, weights_only=False)
        head = DirectHead(ck["dist_dim"], ck.get("state_dim", 19), ck.get("K", 5),
                          ck.get("hidden", 256), ck.get("depth", 3))
        head.load_state_dict(ck["head"])
        return cls(head, model, dyn_norm, pwm_norm, device)

    @torch.no_grad()
    def plan(self, hist_s: np.ndarray, hist_a: np.ndarray, s_t: np.ndarray, v_goal: np.ndarray):
        dev = self.device
        hs = torch.from_numpy(self.dyn_norm(np.asarray(hist_s, np.float32))).unsqueeze(0).to(dev)
        ha = torch.from_numpy(self.pwm_norm(np.asarray(hist_a, np.float32))).unsqueeze(0).to(dev)
        st = torch.from_numpy(self.dyn_norm(np.asarray(s_t, np.float32))).unsqueeze(0).to(dev)
        g = torch.as_tensor(np.asarray(v_goal, np.float32).reshape(1, 3) / V_SCALE, device=dev)
        last_u = torch.as_tensor(np.asarray(hist_a[-1], np.float32).reshape(1, 8), device=dev)
        d = self.model.disturbance(hs, ha)
        seq = self.head(d, st, g, last_u)[0].cpu().numpy().astype(np.float32)
        return seq[0].copy(), {"seq": seq, "J": None, "best": 0}
