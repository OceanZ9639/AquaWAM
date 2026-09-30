"""
Learned forward kinematics of the Alpha 5 arm in the VEHICLE BODY frame, fitted on probe
recordings (u0eval/arm_probe_server.py -> percept/pack_grasp_recordings.py):

    q (5: gripper, b, c, d, e in bridge order) -> end-effector position (3) and rotation (3x3)

A small MLP is exact enough (mm) and differentiable, so the grasp planner can score joint
candidates with fk(q) - obj_body inside the CEM / gradient loop. Also fits the first-order
joint-tracking constant used by the structured world model's arm prior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .models import MLP


class ArmKin(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = MLP(4, 3 + 6, hidden, depth=3)   # b, c, d, e -> xyz + first two rotation columns
        self.register_buffer("q_mean", torch.zeros(4))
        self.register_buffer("q_std", torch.ones(4))
        self.register_buffer("p_mean", torch.zeros(3))
        self.register_buffer("p_std", torch.ones(3))

    def forward(self, q5):
        q = (q5[..., 1:5] - self.q_mean) / self.q_std
        out = self.net(q)
        pos = out[..., :3] * self.p_std + self.p_mean
        a, b = out[..., 3:6], out[..., 6:9]
        # Gram-Schmidt -> proper rotation
        c1 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b2 = b - (c1 * b).sum(-1, keepdim=True) * c1
        c2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c3 = torch.cross(c1, c2, dim=-1)
        R = torch.stack([c1, c2, c3], dim=-1)   # columns
        return pos, R

    @torch.no_grad()
    def ee_body(self, q5: np.ndarray) -> np.ndarray:
        dev = self.q_mean.device
        pos, _ = self(torch.as_tensor(np.asarray(q5, np.float32), device=dev).reshape(-1, 5))
        return pos.cpu().numpy().reshape(-1, 3)


def fit(pack: str, out: str, epochs: int = 60, device: str = "cuda") -> dict:
    z = np.load(pack, allow_pickle=True)
    q = z["joints"].astype(np.float32)
    p = z["ee_body"].astype(np.float32)
    R = z["ee_R"].astype(np.float32).reshape(-1, 3, 3)
    ok = np.isfinite(q).all(1) & np.isfinite(p).all(1)
    q, p, R = q[ok], p[ok], R[ok]
    n = len(q)
    perm = np.random.default_rng(0).permutation(n)
    va, tr = perm[: max(200, n // 10)], perm[max(200, n // 10):]
    m = ArmKin().to(device)
    m.q_mean.copy_(torch.as_tensor(q[:, 1:5].mean(0))); m.q_std.copy_(torch.as_tensor(q[:, 1:5].std(0) + 1e-3))
    m.p_mean.copy_(torch.as_tensor(p.mean(0))); m.p_std.copy_(torch.as_tensor(p.std(0) + 1e-3))
    Q, P, RR = (torch.as_tensor(x, device=device) for x in (q, p, R))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 512
    for ep in range(epochs):
        idx = torch.as_tensor(np.random.permutation(tr), device=device)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            pos, Rh = m(Q[b])
            loss = ((pos - P[b]) ** 2).mean() * 1e3 + ((Rh - RR[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        pos, Rh = m(Q[va])
        pos_mae_mm = float((pos - P[va]).abs().mean() * 1e3)
        rot_err = float(torch.rad2deg(torch.arccos((((Rh * RR[va]).sum((1, 2)) - 1) / 2).clamp(-1, 1))).mean())
    meta = {"n": int(n), "val_pos_mae_mm": pos_mae_mm, "val_rot_err_deg": rot_err, "pack": pack}
    torch.save({"model": m.state_dict(), "meta": meta}, out)
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def load(path: str, device: str = "cuda") -> ArmKin:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ArmKin().to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/hy-tmp/data/grasp/armprobe.npz")
    ap.add_argument("--out", default="/hy-tmp/models/uwam/arm_kin.pt")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    print(json.dumps(fit(a.pack, a.out, a.epochs), indent=1))
