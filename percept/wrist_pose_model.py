"""
Wrist-camera relative-pose head: (hand image, front image, joint angles) -> object position in
the end-effector frame with per-axis uncertainty, plus the object's yaw modulo pi (cylinders and
pipes are symmetric; the grasp approach only needs the axis direction).

Fine-tuned DINOv2 over both views (camera embeddings, attention pooling), the same recipe as the
navigation intent head (percept/train_e2e.py). Deployment wrapper at the bottom.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ENCODERS = {
    "small": "/hy-tmp/models/hf_cache/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "base": "/hy-tmp/models/hf_cache/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
N_JOINTS = 5
OUT_DIM = 3 + 3 + 2   # obj_ee mean, obj_ee log-variance, (sin 2psi, cos 2psi)


class WristPose(nn.Module):
    def __init__(self, encoder: str = "base", n_queries: int = 4, freeze_blocks: int = 4, use_ego: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(ENCODERS[encoder])
        d = self.backbone.config.hidden_size
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[:freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
        self.use_ego = use_ego
        self.cam_emb = nn.Parameter(torch.zeros(2, 1, d))
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(n_queries * d + N_JOINTS, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, OUT_DIM),
        )

    def tokens(self, x):
        return self.backbone(pixel_values=x).last_hidden_state

    def forward(self, wrist, ego, joints):
        B = wrist.shape[0]
        t = self.tokens(wrist) + self.cam_emb[1]
        if self.use_ego and ego is not None:
            t = torch.cat([t, self.tokens(ego) + self.cam_emb[0]], dim=1)
        t = self.norm(t)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attn(q, t, t)
        out = self.head(torch.cat([pooled.flatten(1), joints], dim=-1))
        mu, logvar, yaw2 = out[:, 0:3], out[:, 3:6].clamp(-9, 2), out[:, 6:8]
        return mu, logvar, yaw2


def preprocess_rgb(im_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB (the bridge's camera frames) -> normalized 3x224x224 tensor."""
    im = cv2.resize(im_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    im = (im - MEAN) / STD
    return torch.from_numpy(im.transpose(2, 0, 1).copy())


class WristPoseHead:
    """Deployment wrapper: images as uint8 HxWx3 (RGB from the bridge), joints (5,) -> dict."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model = WristPose(ck.get("encoder", "base"), ck.get("n_queries", 4), ck.get("freeze_blocks", 4),
                               ck.get("use_ego", True)).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.joint_norm = ck.get("joint_norm")
        self.device = device

    @torch.no_grad()
    def predict(self, wrist_rgb: np.ndarray, ego_rgb: np.ndarray | None, joints: np.ndarray) -> dict:
        w = preprocess_rgb(wrist_rgb).unsqueeze(0).to(self.device)
        e = None
        if ego_rgb is not None and self.model.use_ego:
            e = preprocess_rgb(ego_rgb).unsqueeze(0).to(self.device)
        j = np.asarray(joints, np.float32).reshape(1, N_JOINTS)
        if self.joint_norm is not None:
            j = (j - self.joint_norm[0]) / self.joint_norm[1]
        mu, logvar, yaw2 = self.model(w, e, torch.from_numpy(j).to(self.device))
        s = np.exp(0.5 * logvar[0].cpu().numpy())
        y = yaw2[0].cpu().numpy()
        return {"obj_ee": mu[0].cpu().numpy(), "sigma": s,
                "yaw_mod_pi": float(0.5 * np.arctan2(y[0], y[1]))}
