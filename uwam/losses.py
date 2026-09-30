"""Training losses: state prediction, visual L1/LPIPS, action ranking."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _dvl(s):
    return s[..., 0:3]


def state_prediction_loss(s_hat, s_fut, dvl_weight: float = 2.0):
    l_all = F.smooth_l1_loss(s_hat, s_fut)
    l_dvl = F.smooth_l1_loss(_dvl(s_hat), _dvl(s_fut))
    return l_all + dvl_weight * l_dvl, {"state_l1": l_all.detach(), "dvl_l1": l_dvl.detach()}


def ranking_loss(err_true, err_cf, margin: float):
    """max(0, m + D(true, gt) - D(cf, gt)); want true closer than counterfactual."""
    return torch.relu(margin + err_true - err_cf).mean()


def pairwise_dvl_mae(pred, gt):
    return (pred[..., 0:3] - gt[..., 0:3]).abs().mean(dim=(1, 2))


def action_rank_losses(out: Dict[str, torch.Tensor], s_fut: torch.Tensor, margin: float) -> Dict[str, torch.Tensor]:
    err_t = pairwise_dvl_mae(out["s_hat"], s_fut)
    losses = {}
    total = err_t.new_zeros(())
    for name in ("zero", "rev", "rand", "near"):
        key = f"s_hat_{name}"
        if key not in out:
            continue
        err_cf = pairwise_dvl_mae(out[key], s_fut)
        losses[f"rank_{name}"] = ranking_loss(err_t, err_cf, margin)
        total = total + losses[f"rank_{name}"]
        losses[f"rank_acc_{name}"] = (err_t < err_cf).float().mean().detach()
    losses["rank"] = total
    return losses


def pixel_l1(pred, gt):
    return (pred - gt).abs().mean()


class OptionalLPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = None
        try:
            import lpips  # type: ignore

            self.fn = lpips.LPIPS(net="vgg")
            for p in self.fn.parameters():
                p.requires_grad_(False)
        except Exception:
            self.fn = None

    def forward(self, pred, gt):
        if self.fn is None:
            # downsampled L2 fallback (no extra weights)
            p = torch.nn.functional.avg_pool2d(pred, 4)
            g = torch.nn.functional.avg_pool2d(gt, 4)
            return ((p - g) ** 2).mean()
        # lpips expects [-1, 1]
        return self.fn(pred * 2 - 1, gt * 2 - 1).mean()


def dynamics_batch_loss(out, batch, cfg, lpips_mod: Optional[OptionalLPIPS] = None):
    logs = {}
    l_state, extra = state_prediction_loss(out["s_hat"], batch["s_fut"], cfg.train.lambda_dvl)
    logs.update(extra)
    ranks = action_rank_losses(out, batch["s_fut"], cfg.train.rank_margin)
    logs.update({k: v.detach() if torch.is_tensor(v) else v for k, v in ranks.items()})
    loss = cfg.train.lambda_state * l_state + cfg.train.lambda_rank * ranks["rank"]
    if "s_hat_nohist" in out:
        err_h = pairwise_dvl_mae(out["s_hat"], batch["s_fut"])
        err_n = pairwise_dvl_mae(out["s_hat_nohist"], batch["s_fut"])
        l_hist = ranking_loss(err_h, err_n, cfg.train.rank_margin)
        logs["hist_rank"] = l_hist.detach()
        loss = loss + cfg.train.lambda_hist * l_hist
    if "rgb_hat" in out and "rgb_gt" in batch:
        l_pix = pixel_l1(out["rgb_hat"], batch["rgb_gt"])
        logs["pix_l1"] = l_pix.detach()
        loss = loss + cfg.train.lambda_pix * l_pix
        if lpips_mod is not None:
            l_lp = lpips_mod(out["rgb_hat"], batch["rgb_gt"])
            logs["lpips"] = l_lp.detach()
            loss = loss + cfg.train.lambda_lpips * l_lp
    logs["loss"] = loss.detach()
    return loss, logs
