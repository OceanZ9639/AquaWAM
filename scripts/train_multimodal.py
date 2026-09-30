#!/usr/bin/env python3
"""Multimodal WAM: RGB + optional FLS + DVL with modality dropout=0.2."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from uwam.config import Cfg, ensure_dirs
from uwam.data import DynamicsWindowDataset, VisualWindowDataset, collate, load_ou_split, load_split
from uwam.losses import dynamics_batch_loss, pixel_l1, state_prediction_loss
from uwam.models import DynamicsWAM, MultimodalWAM


class DropoutMM(Dataset):
    def __init__(self, vis: VisualWindowDataset, p_drop: float = 0.2):
        self.vis = vis
        self.p = p_drop

    def __len__(self):
        return len(self.vis)

    def __getitem__(self, i):
        item = self.vis[i]
        rng = np.random.RandomState(i * 13 + 7)
        item["mask_rgb"] = torch.tensor(0.0 if rng.random() < self.p else 1.0)
        native_sonar = float(item["mask_sonar"]) if "mask_sonar" in item else 0.0
        drop_s = native_sonar > 0.5 and rng.random() < self.p
        item["mask_sonar"] = torch.tensor(0.0 if drop_s else native_sonar)
        item["mask_dvl"] = torch.tensor(0.0 if rng.random() < self.p else 1.0)
        if "sonar" not in item:
            item["sonar"] = torch.zeros(1, 96, 128)
        return item


def _masks(batch, device):
    b = batch["s_t"].size(0)
    def m(key):
        if key not in batch:
            return torch.ones(b, 1, device=device)
        return batch[key].view(b, 1).to(device)
    return {"rgb": m("mask_rgb"), "sonar": m("mask_sonar"), "wrist": torch.ones(b, 1, device=device)}


@torch.no_grad()
def eval_dropout(mm, loader, device, dyn_norm) -> dict:
    mm.eval()
    acc = {
        "full": 0.0, "cam_blackout": 0.0, "dvl_loss": 0.0, "no_fls": 0.0,
        "sonar_profile_l1": 0.0, "sonar_profile_l1_no_fls": 0.0, "n_sonar": 0, "n": 0,
    }

    def mae_ms(pred, gt):
        p = dyn_norm.invert_torch(pred)[..., 0:3]
        g = dyn_norm.invert_torch(gt)[..., 0:3]
        return (p - g).abs().mean().item()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        bsz = batch["s_t"].size(0)
        rgb = batch["rgb"]
        sonar = batch.get("sonar")
        out = mm(batch, rgb=rgb, sonar=sonar, masks=None)
        pred = out.get("s_hat_mm", out["s_hat"])
        acc["full"] += mae_ms(pred, batch["s_fut"]) * bsz
        zrgb = torch.zeros_like(rgb)
        out_b = mm(batch, rgb=zrgb, sonar=sonar, masks={"rgb": torch.zeros(bsz, 1, device=device)})
        acc["cam_blackout"] += mae_ms(out_b.get("s_hat_mm", out_b["s_hat"]), batch["s_fut"]) * bsz
        b2 = dict(batch)
        b2["hist_s"] = batch["hist_s"].clone()
        b2["hist_s"][..., 0:3] = 0
        out_d = mm(b2, rgb=rgb, sonar=sonar)
        acc["dvl_loss"] += mae_ms(out_d.get("s_hat_mm", out_d["s_hat"]), batch["s_fut"]) * bsz
        out_f = mm(batch, rgb=rgb, sonar=None)
        acc["no_fls"] += mae_ms(out_f.get("s_hat_mm", out_f["s_hat"]), batch["s_fut"]) * bsz
        if "sonar_gt_profile" in batch and "sonar_profile" in out:
            m = batch["mask_sonar"].view(-1) > 0.5 if "mask_sonar" in batch else torch.ones(bsz, dtype=torch.bool, device=device)
            if m.any():
                ns = int(m.sum().item())
                acc["sonar_profile_l1"] += (out["sonar_profile"][m] - batch["sonar_gt_profile"][m]).abs().mean().item() * ns
                acc["sonar_profile_l1_no_fls"] += (out_f["sonar_profile"][m] - batch["sonar_gt_profile"][m]).abs().mean().item() * ns
                acc["n_sonar"] += ns
        acc["n"] += bsz
    n = max(1, acc["n"])
    ns = max(1, acc["n_sonar"])
    mm.train()
    outm = {k: (acc[k] / n if k not in ("n", "n_sonar", "sonar_profile_l1", "sonar_profile_l1_no_fls") else acc[k]) for k in acc}
    outm["n"] = acc["n"]
    outm["n_sonar"] = acc["n_sonar"]
    outm["sonar_profile_l1"] = acc["sonar_profile_l1"] / ns
    outm["sonar_profile_l1_no_fls"] = acc["sonar_profile_l1_no_fls"] / ns
    return outm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dyn", default="/hy-tmp/models/uwam/best.pt")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-episodes", type=int, default=80)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--mm-ckpt", default="/hy-tmp/models/uwam/multimodal_last.pt")
    args = ap.parse_args()
    cfg = Cfg()
    cfg.paths.usim = Path(args.usim)
    cfg.model.use_language = False
    cfg.train.modality_dropout = 0.2
    ensure_dirs(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt_dyn, map_location=device, weights_only=False)
    ou_eps = load_ou_split(Path("/hy-tmp/data/ou_explore"), load_images=True)
    train_eps = [] if args.eval_only else load_split(cfg.paths.usim, "train", cfg.schema, max_episodes=args.max_episodes)
    dyn_ds = DynamicsWindowDataset(train_eps + ou_eps, cfg, fit_norm=True)
    if "dyn_norm" in ckpt:
        dyn_ds.dyn_norm.load_state_dict(ckpt["dyn_norm"])
        dyn_ds.pwm_norm.load_state_dict(ckpt["pwm_norm"])
    test_v = DropoutMM(VisualWindowDataset(ou_eps, cfg, dyn_ds.dyn_norm, dyn_ds.pwm_norm, stride=8), 0.0)
    print(f"ou_test={len(test_v)} ou_eps={len(ou_eps)}", flush=True)
    if len(test_v) == 0:
        raise FileNotFoundError("OU episodes have no RGB/FLS arrays; cannot eval no_fls vs full")
    mm = MultimodalWAM(cfg).to(device)
    mm.dyn.load_state_dict(ckpt["model"], strict=False)
    for p in mm.dyn.parameters():
        p.requires_grad_(False)
    if args.eval_only:
        mm_ckpt = torch.load(args.mm_ckpt, map_location=device, weights_only=False)
        missing, unexpected = mm.load_state_dict(mm_ckpt["mm"], strict=False)
        print(f"eval {args.mm_ckpt} missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}", flush=True)
        te = DataLoader(test_v, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        drop = eval_dropout(mm, te, device, dyn_ds.dyn_norm)
        extra = {
            "ou_regimes": len(ou_eps),
            "ou_has_fls": any(ep.fls_arr is not None for ep in ou_eps),
            "no_fls_minus_full": drop["no_fls"] - drop["full"],
            "selected": drop,
        }
        hist_path = cfg.paths.logs / "multimodal_history.json"
        prev = json.loads(hist_path.read_text()) if hist_path.exists() else [drop]
        payload = {"history": prev, **extra}
        (cfg.paths.logs / "multimodal_eval.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(drop, indent=2), flush=True)
        return
    usim_v = VisualWindowDataset(train_eps, cfg, dyn_ds.dyn_norm, dyn_ds.pwm_norm, stride=8)
    ou_v = VisualWindowDataset(ou_eps, cfg, dyn_ds.dyn_norm, dyn_ds.pwm_norm, stride=4)
    train_v = ConcatDataset([DropoutMM(usim_v, 0.2), DropoutMM(ou_v, 0.2)])
    print(f"mm windows train={len(train_v)} ou_test={len(test_v)}", flush=True)
    opt = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    loader = DataLoader(train_v, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    te = DataLoader(test_v, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    history = []
    best_gap = -1e9
    for epoch in range(1, args.epochs + 1):
        mm.train()
        mm.dyn.eval()
        running, n_seen = 0.0, 0
        pbar = tqdm(loader, desc=f"mm {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if batch.get("mask_dvl") is not None:
                drop = batch["mask_dvl"].view(-1, 1, 1) < 0.5
                if drop.any():
                    hs = batch["hist_s"].clone()
                    hs[drop.view(-1)][..., 0:3] = 0
                    batch["hist_s"] = hs
            opt.zero_grad(set_to_none=True)
            masks = _masks(batch, device)
            rgb = batch["rgb"] * masks["rgb"].view(-1, 1, 1, 1)
            sonar = batch["sonar"] * masks["sonar"].view(-1, 1, 1, 1)
            out = mm(batch, rgb=rgb, sonar=sonar, masks=masks)
            loss, logs = dynamics_batch_loss(out, batch, cfg)
            if "s_hat_mm" in out:
                l_mm, extra = state_prediction_loss(out["s_hat_mm"], batch["s_fut"], cfg.train.lambda_dvl)
                loss = loss + l_mm
                logs["mm_dvl"] = extra["dvl_l1"]
            if "rgb_hat" in out:
                loss = loss + 0.2 * cfg.train.lambda_pix * pixel_l1(out["rgb_hat"], batch["rgb_gt"])
            if "sonar_profile" in out and "sonar_gt_profile" in batch and "mask_sonar" in batch:
                m = batch["mask_sonar"].view(-1) > 0.5
                if m.any():
                    loss = loss + 5.0 * pixel_l1(out["sonar_profile"][m], batch["sonar_gt_profile"][m])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mm.parameters(), 1.0)
            opt.step()
            running += loss.item() * batch["s_t"].size(0)
            n_seen += batch["s_t"].size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        drop = eval_dropout(mm, te, device, dyn_ds.dyn_norm)
        rec = {"epoch": epoch, "train_loss": running / max(1, n_seen), **drop}
        history.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
        torch.save({"mm": mm.state_dict(), "metrics": rec, "dyn_norm": dyn_ds.dyn_norm.state_dict(),
                    "pwm_norm": dyn_ds.pwm_norm.state_dict()}, cfg.paths.ckpt / "multimodal_last.pt")
        gap = rec["no_fls"] - rec["full"]
        if gap > best_gap:
            best_gap = gap
            torch.save({"mm": mm.state_dict(), "metrics": rec, "dyn_norm": dyn_ds.dyn_norm.state_dict(),
                        "pwm_norm": dyn_ds.pwm_norm.state_dict()}, cfg.paths.ckpt / "multimodal_best.pt")
    (cfg.paths.logs / "multimodal_history.json").write_text(json.dumps(history, indent=2))
    best = max(history, key=lambda r: (r.get("no_fls") or 0) - (r.get("full") or 0)) if history else {}
    extra = {
        "ou_regimes": len(ou_eps),
        "ou_has_fls": any(ep.fls_arr is not None for ep in ou_eps),
        "no_fls_minus_full": (best.get("no_fls") - best.get("full")) if best else None,
        "selected": best,
    }
    (cfg.paths.logs / "multimodal_eval.json").write_text(json.dumps({"history": history, **extra}, indent=2))
    print("saved multimodal_last.pt", flush=True)


if __name__ == "__main__":
    main()
