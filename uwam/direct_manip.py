"""WAM-direct for manipulation: one forward pass -> a 16-step chunk of the FULL 13-D action
(8 thrusters + 5 joints incl. the jaw), no imagination search, no stage program, no close gate.

Distilled from the system's own rollouts (imagination planner + task program + learned/hand gate) the
way DreamZero's action head is trained: the model sees the last L frames of the 35-D physical state
(vehicle, arm joints, object relative pose) and the last L actions, and regresses the next chunk.
The object relative pose comes from the same source as for the planner (privileged or wrist head).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

STATE_DIM, ACT_DIM = 35, 13


class DirectManip(nn.Module):
    def __init__(self, L: int = 16, K: int = 16, hidden: int = 512, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        self.L, self.K = L, K
        # per-frame encoder -> GRU over the history -> MLP head
        self.enc = nn.Sequential(nn.Linear(STATE_DIM + ACT_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        layers = []
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, K * ACT_DIM)]
        self.head = nn.Sequential(*layers)
        # per-channel normalizers (filled at train time, saved in the checkpoint)
        self.register_buffer("s_mu", torch.zeros(STATE_DIM)); self.register_buffer("s_sd", torch.ones(STATE_DIM))
        self.register_buffer("a_mu", torch.zeros(ACT_DIM)); self.register_buffer("a_sd", torch.ones(ACT_DIM))

    def forward(self, hs, ha):
        """hs [B, L, 35] raw, ha [B, L, 13] raw -> [B, K, 13] raw actions."""
        x = torch.cat([(hs - self.s_mu) / self.s_sd, (ha - self.a_mu) / self.a_sd], -1)
        h, _ = self.gru(self.enc(x))
        out = self.head(h[:, -1]).view(-1, self.K, ACT_DIM)
        return out * self.a_sd + self.a_mu


class DirectManipPolicy:
    """Deployment wrapper: numpy in, numpy chunk out (PWM clipped to [-1, 1], jaw clipped to [0, 0.015])."""

    def __init__(self, ckpt: str, device: str = "cpu"):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model = DirectManip(**ck["arch"]).to(device)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        self.device = device
        self.meta = {k: v for k, v in ck.items() if k not in ("state_dict",)}

    @torch.no_grad()
    def act(self, hist_s: np.ndarray, hist_a: np.ndarray) -> np.ndarray:
        hs = torch.as_tensor(np.asarray(hist_s, np.float32), device=self.device)[None, -self.model.L:]
        ha = torch.as_tensor(np.asarray(hist_a, np.float32), device=self.device)[None, -self.model.L:]
        out = self.model(hs, ha)[0].cpu().numpy()
        out[:, :8] = np.clip(out[:, :8], -1.0, 1.0)
        out[:, 8] = np.clip(out[:, 8], 0.0, 0.015)           # jaw
        return out
