"""Disturbance-aware multimodal action-conditioned underwater world model."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, ins, outs, hidden=256, depth=2, dropout=0.0):
        super().__init__()
        layers = []
        d = ins
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, outs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):
    """AquaJEPA-style stride-2 5x5 stack: 32, 64, 96, 128 -> GAP -> latent."""

    def __init__(self, in_ch: int, latent: int):
        super().__init__()
        chs = [in_ch, 32, 64, 96, 128]
        blocks = []
        for a, b in zip(chs[:-1], chs[1:]):
            blocks += [
                nn.Conv2d(a, b, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(8 if b >= 8 else 1, b),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*blocks)
        self.proj = nn.Linear(128, latent)

    def forward(self, x):
        # x: [B, C, H, W] in [0, 1]
        h = self.conv(x)
        h = h.mean(dim=(-2, -1))
        return self.proj(h)


class ResidualUNet(nn.Module):
    """Predict ΔI so that Î_{t+K} = I_t + ΔI."""

    def __init__(self, in_ch: int = 3, cond_dim: int = 128, base: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.GELU(),
        )
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.GELU())
        self.down3 = nn.Sequential(nn.Conv2d(base * 4, base * 8, 4, 2, 1), nn.GELU())
        self.cond = nn.Linear(cond_dim, base * 8)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.GELU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 2, 4, 2, 1), nn.GELU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base * 4, base, 4, 2, 1), nn.GELU())
        self.head = nn.Conv2d(base * 2, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.delta_scale = 0.4
        # FiLM so PWM/disturbance actually modulate the residual (bottleneck-only add was too weak)
        self.film3 = nn.Linear(cond_dim, base * 8 * 2)
        self.film2 = nn.Linear(cond_dim, base * 4 * 2)
        self.film1 = nn.Linear(cond_dim, base * 2 * 2)

    def _film(self, x, cond, layer):
        gb = layer(cond)
        b, c, _, _ = x.shape
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma.view(b, c, 1, 1)) + beta.view(b, c, 1, 1)

    def forward(self, img, cond):
        x0 = self.stem(img)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = self._film(x3, cond, self.film3)
        y2 = self.up3(x3)
        y2 = torch.cat([self._film(y2, cond, self.film2), x2], dim=1)
        y1 = self.up2(y2)
        y1 = torch.cat([self._film(y1, cond, self.film1), x1], dim=1)
        y0 = self.up1(y1)
        y0 = torch.cat([y0, x0], dim=1)
        delta = self.delta_scale * torch.tanh(self.head(y0))
        return delta


class HistoryDisturbanceEncoder(nn.Module):
    """d_t = E_D(H_t). GRU over (state, past action[, sample period])."""

    def __init__(self, state_dim: int, pwm_dim: int, hidden: int, dist_dim: int, dropout: float,
                 use_dt: bool = False):
        super().__init__()
        self.use_dt = use_dt
        self.inp = nn.Linear(state_dim + pwm_dim + (1 if use_dt else 0), hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, dist_dim),
        )

    def forward(self, hist_s, hist_a, dt_feat=None):
        x = torch.cat([hist_s, hist_a], dim=-1)
        if self.use_dt:
            b, L, _ = x.shape
            f = torch.zeros(b, L, 1, device=x.device, dtype=x.dtype) if dt_feat is None \
                else dt_feat.view(b, 1, 1).expand(b, L, 1)
            x = torch.cat([x, f], dim=-1)
        x = F.gelu(self.inp(x))
        h, _ = self.gru(x)
        return self.out(h[:, -1])


class DynamicsWAM(nn.Module):
    """
    Nominal + history-conditioned residual:

        ŝ = f_nom(s, a) + f_disturbance(s, a, d)
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.K = m.horizon_dyn
        self.use_dt = bool(getattr(m, "use_dt", False))
        # ablation "w/o disturbance token": the history encoder is bypassed and d_t = 0, so the
        # residual net sees only (s_t, a_{t:t+K}); trained and served with the same flag.
        self.no_token = bool(getattr(m, "no_token", False))
        self.hist = HistoryDisturbanceEncoder(
            m.dyn_state_dim, m.pwm_dim, m.hidden, m.disturbance_dim, m.dropout, use_dt=self.use_dt
        )
        in_nom = m.dyn_state_dim + m.pwm_dim * m.horizon_dyn + (1 if self.use_dt else 0)
        self.f_nom = MLP(in_nom, m.dyn_state_dim * m.horizon_dyn, m.hidden, depth=3, dropout=m.dropout)
        self.f_res = MLP(
            in_nom + m.disturbance_dim,
            m.dyn_state_dim * m.horizon_dyn,
            m.hidden,
            depth=3,
            dropout=m.dropout,
        )
        self.task_emb = nn.Embedding(m.n_tasks, m.latent_dim)
        self.task_to_d = nn.Linear(m.latent_dim, m.disturbance_dim)
        self.flow_head = nn.Linear(m.disturbance_dim, 3)
        self.fail_head = nn.Linear(m.disturbance_dim, 3)
        # per-thruster efficiency regression: the fault is unobserved in the commanded-action
        # coordinates, so d has to carry it; this head makes that quantitative (MAE on eta)
        self.eta_head = MLP(m.disturbance_dim, 8, m.hidden, depth=2, dropout=m.dropout)
        # Dead-reckons the current DVL velocity from IMU + PWM history when the DVL is gone, so the
        # MPC can keep closing the loop. Reads the raw masked window (a ridge fit on the window
        # beats a head on d alone) PLUS the disturbance token: with commanded-action semantics the
        # actuator fault is only visible as a command-vs-IMU discrepancy, which d is trained to
        # carry (fault CE, eta regression, d-consistency).
        self.vel_win = m.history_len
        self.vel_head = MLP(
            m.history_len * (m.dyn_state_dim + m.pwm_dim) + m.disturbance_dim,
            3, m.hidden, depth=3, dropout=m.dropout,
        )
        self.use_language = m.use_language
        self.state_dim = m.dyn_state_dim
        self.dist_dim = m.disturbance_dim

    @staticmethod
    def dt_feature(dt, ref: float = 0.1):
        """Normalized sample-period feature: 0 at the native 10 Hz grid."""
        return dt / ref - 1.0

    def disturbance(self, hist_s, hist_a, dt_feat=None):
        if self.no_token:
            return hist_s.new_zeros(hist_s.shape[0], self.dist_dim)
        if self.use_dt:
            return self.hist(hist_s, hist_a, dt_feat=dt_feat)
        return self.hist(hist_s, hist_a)

    @staticmethod
    def mask_dvl(hist_s):
        h = hist_s.clone()
        h[..., 0:3] = 0.0
        return h

    def estimate_velocity(self, hist_s, hist_a, already_masked: bool = True, dt_feat=None):
        """Predicted normalized DVL velocity from a DVL-free history window."""
        h = hist_s if already_masked else self.mask_dvl(hist_s)
        d_masked = self.disturbance(h, hist_a, dt_feat=dt_feat)
        x = torch.cat([torch.cat([h, hist_a], dim=-1).flatten(1), d_masked], dim=1)
        return self.vel_head(x)

    def rollout(self, s_t, a_fut, d, task_index=None, dt_feat=None):
        b = s_t.size(0)
        a_flat = a_fut.reshape(b, -1)
        parts = [s_t, a_flat]
        if self.use_dt:
            f = torch.zeros(b, 1, device=s_t.device, dtype=s_t.dtype) if dt_feat is None \
                else dt_feat.view(b, 1)
            parts.append(f)
        base = torch.cat(parts, dim=-1)
        nom = self.f_nom(base)
        d_use = d
        if self.use_language and task_index is not None:
            d_use = d_use + self.task_to_d(self.task_emb(task_index))
        res = self.f_res(torch.cat([base, d_use], dim=-1))
        delta = (nom + res).view(b, self.K, self.state_dim)
        # integrate residual deltas from current state (shared across horizon as offsets)
        # Use cumulative: s_{t+k} = s_t + sum_{i<=k} delta_i   -- too strong.
        # Document form is one-step residual; we predict K-step offsets from s_t.
        s_hat = s_t.unsqueeze(1) + delta
        return s_hat, nom.view(b, self.K, self.state_dim), res.view(b, self.K, self.state_dim)

    def forward(self, batch, use_history: bool = True, counterfactuals: bool = True) -> Dict[str, torch.Tensor]:
        dt_feat = None
        if self.use_dt and "dt" in batch:
            dt_feat = self.dt_feature(batch["dt"])
        if use_history:
            d = self.disturbance(batch["hist_s"], batch["hist_a"], dt_feat=dt_feat)
        else:
            d = torch.zeros(
                batch["s_t"].size(0),
                self.dist_dim,
                device=batch["s_t"].device,
                dtype=batch["s_t"].dtype,
            )
        s_hat, nom, res = self.rollout(batch["s_t"], batch["a_fut"], d,
                                       None if not self.use_language else batch.get("task_index"),
                                       dt_feat=dt_feat)
        out = {
            "s_hat": s_hat,
            "d": d,
            "nom": nom,
            "res": res,
            "flow_logits": self.flow_head(d),
            "fail_logits": self.fail_head(d),
            "eta_hat": self.eta_head(d),
        }
        if counterfactuals:
            for name, key in (("zero", "a_zero"), ("rev", "a_rev"), ("rand", "a_rand"), ("near", "a_near")):
                if key in batch:
                    s_cf, _, _ = self.rollout(
                        batch["s_t"],
                        batch[key],
                        d.detach(),
                        None if not self.use_language else batch.get("task_index"),
                        dt_feat=dt_feat,
                    )
                    out[f"s_hat_{name}"] = s_cf
        return out


class VisualWAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.enc = ConvEncoder(3, m.latent_dim)
        self.act = MLP(m.pwm_dim * m.horizon_vis, m.latent_dim, m.hidden, depth=2)
        self.unet = ResidualUNet(in_ch=3, cond_dim=m.latent_dim * 2 + m.disturbance_dim)

    def forward(self, img, a_fut, d):
        z_v = self.enc(img)
        z_a = self.act(a_fut.reshape(a_fut.size(0), -1))
        cond = torch.cat([z_v, z_a, d], dim=-1)
        delta = self.unet(img, cond)
        return torch.clamp(img + delta, 0.0, 1.0), delta


class MultimodalWAM(nn.Module):
    """
    RGB/wrist + optional FLS + proprio + disturbance + action (+ language)
    -> future state / DVL / RGB / sonar profile.
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.dyn = DynamicsWAM(cfg)
        self.rgb_enc = ConvEncoder(3, m.latent_dim)
        self.wrist_enc = ConvEncoder(3, m.latent_dim)
        self.sonar_enc = ConvEncoder(1, m.latent_dim)
        self.prop_enc = MLP(m.dyn_state_dim, m.latent_dim, m.hidden, depth=2, dropout=m.dropout)
        self.act_enc = MLP(m.pwm_dim * m.horizon_dyn, m.latent_dim, m.hidden, depth=2)
        fuse_in = m.latent_dim * 5 + m.disturbance_dim
        self.fuse = MLP(fuse_in, m.latent_dim, m.hidden, depth=2, dropout=m.dropout)
        self.state_head = nn.Linear(m.latent_dim, m.dyn_state_dim * m.horizon_dyn)
        self.sonar_state_head = nn.Linear(m.latent_dim, m.dyn_state_dim * m.horizon_dyn)
        self.sonar_profile_head = nn.Linear(m.latent_dim, 128)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)
        nn.init.zeros_(self.sonar_state_head.weight)
        nn.init.zeros_(self.sonar_state_head.bias)
        self.vis = VisualWAM(cfg)
        self.drop = m.dropout
        self.K = m.horizon_dyn
        self.state_dim = m.dyn_state_dim

    def forward(
        self,
        batch,
        rgb: Optional[torch.Tensor] = None,
        wrist: Optional[torch.Tensor] = None,
        sonar: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        dyn_out = self.dyn(batch, counterfactuals=False)
        d = dyn_out["d"]
        z_p = self.prop_enc(batch["s_t"])
        z_a = self.act_enc(batch["a_fut"].reshape(batch["a_fut"].size(0), -1))
        b = batch["s_t"].size(0)
        device = batch["s_t"].device
        z_v = self.rgb_enc(rgb) if rgb is not None else torch.zeros(b, z_p.size(-1), device=device)
        z_w = self.wrist_enc(wrist) if wrist is not None else torch.zeros_like(z_v)
        z_s = self.sonar_enc(sonar) if sonar is not None else torch.zeros_like(z_v)
        if masks is not None:
            z_v = z_v * masks.get("rgb", 1)
            z_w = z_w * masks.get("wrist", 1)
            z_s = z_s * masks.get("sonar", 1)
        z = self.fuse(torch.cat([z_v, z_w, z_s, z_p, z_a, d], dim=-1))
        # residual on frozen dyn so FLS/RGB can help DVL; no_fls zeros z_s and the sonar residual
        res = self.state_head(z).view(b, self.K, self.state_dim)
        res_s = self.sonar_state_head(z_s).view(b, self.K, self.state_dim)
        s_mm = dyn_out["s_hat"] + res + res_s
        dyn_out["s_hat_mm"] = s_mm
        dyn_out["z"] = z
        dyn_out["sonar_profile"] = self.sonar_profile_head(z_s)
        if rgb is not None:
            a_v = batch["a_vis"] if "a_vis" in batch else batch["a_fut"]
            img_hat, delta = self.vis(rgb, a_v, d)
            dyn_out["rgb_hat"] = img_hat
            dyn_out["rgb_delta"] = delta
        return dyn_out
