"""Multimodal dead reckoning: camera + sonar substitute for the DVL during outages.

The proprioceptive estimator infers velocity from IMU + commanded PWM, which goes blind to
whatever the commands cannot explain (an unobserved actuator fault). Optical flow in the camera
and range structure in the FLS are direct evidence of ego-motion, so this head fuses BOTH the
current and the previous frame of each modality with the masked proprio window.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .models import ConvEncoder, MLP


class VelMM(nn.Module):
    """v_t from (DVL-masked window, disturbance token, two stacked RGB+FLS frames)."""

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        # 2 frames x (RGB 3 + FLS 1) channels, stacked -> ego-motion is a two-frame signal
        self.img_enc = ConvEncoder(8, m.latent_dim)
        in_dim = m.history_len * (m.dyn_state_dim + m.pwm_dim) + m.disturbance_dim + m.latent_dim
        self.head = MLP(in_dim, 3, m.hidden, depth=3, dropout=m.dropout)

    def forward(self, dyn_model, hist_s_masked, hist_a, img_pair):
        with torch.no_grad():
            d = dyn_model.hist(hist_s_masked, hist_a)
        flat = torch.cat([hist_s_masked, hist_a], dim=-1).flatten(1)
        z_img = self.img_enc(img_pair)
        return self.head(torch.cat([flat, d, z_img], dim=1))


def pack_img_pair(rgb_prev, fls_prev, rgb_now, fls_now, hw=(96, 128)) -> np.ndarray:
    """Two frames of RGB+FLS -> [8, H, W] float32 in [0, 1]; missing channels are zeros."""
    import cv2

    H, W = hw
    out = np.zeros((8, H, W), np.float32)

    def put(dst, img, is_rgb):
        if img is None:
            return
        a = np.asarray(img)
        if a.ndim == 3 and (a.shape[0] != H or a.shape[1] != W):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        elif a.ndim == 2 and (a.shape[0] != H or a.shape[1] != W):
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        a = a.astype(np.float32) / 255.0
        if is_rgb:
            out[dst:dst + 3] = a.transpose(2, 0, 1)
        else:
            out[dst] = a

    put(0, rgb_prev, True)
    put(3, fls_prev, False)
    put(4, rgb_now, True)
    put(7, fls_now, False)
    return out


class VelMMRuntime:
    """Closed-loop wrapper: keeps the previous frame, returns raw-unit velocity estimates."""

    def __init__(self, velmm: VelMM, dyn_model, dyn_norm, pwm_norm, device: str = "cuda"):
        self.velmm = velmm.to(device).eval()
        self.dyn = dyn_model
        self.dyn_norm = dyn_norm
        self.pwm_norm = pwm_norm
        self.device = device
        self._prev: Optional[tuple] = None

    def reset(self):
        self._prev = None

    def estimate(self, hist_s: np.ndarray, hist_a: np.ndarray, rgb, fls):
        if rgb is None and fls is None:
            return None
        prev_rgb, prev_fls = self._prev if self._prev is not None else (rgb, fls)
        pair = pack_img_pair(prev_rgb, prev_fls, rgb, fls)
        self._prev = (rgb, fls)
        with torch.no_grad():
            hs = torch.from_numpy(self.dyn_norm(hist_s)).unsqueeze(0).to(self.device)
            ha = torch.from_numpy(self.pwm_norm(hist_a)).unsqueeze(0).to(self.device)
            img = torch.from_numpy(pair).unsqueeze(0).to(self.device)
            v_n = self.velmm(self.dyn, self.dyn.mask_dvl(hs), ha, img)[0].cpu().numpy()
        return (v_n * self.dyn_norm.std[0:3] + self.dyn_norm.mean[0:3]).astype(np.float32)
