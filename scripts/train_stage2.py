#!/usr/bin/env python3
"""Stage-2 residual visual WAM: freeze Stage-1 disturbance token, train U-Net + rank."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uwam.config import Cfg, ensure_dirs
from uwam.data import DynamicsWindowDataset, VisualWindowDataset, collate, load_split
from uwam.losses import OptionalLPIPS, pixel_l1, ranking_loss
from uwam.models import DynamicsWAM, VisualWAM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def visual_rank(vis, dyn, batch, device) -> dict:
    vis.eval()
    d = dyn.disturbance(batch["hist_s"], batch["hist_a"])
    pred, _ = vis(batch["rgb"], batch["a_vis"], d)
    err_t = (pred - batch["rgb_gt"]).abs().mean(dim=(1, 2, 3))
    copy_err = (batch["rgb"] - batch["rgb_gt"]).abs().mean(dim=(1, 2, 3))
    out = {
        "pix": float((pred - batch["rgb_gt"]).abs().mean().item()),
        "copy": float(copy_err.mean().item()),
        "beat_copy": float((err_t < copy_err).float().mean().item()),
    }
    for name, key in (("zero", "a_zero"), ("rev", "a_rev"), ("rand", "a_rand")):
        p_cf, _ = vis(batch["rgb"], batch[key], d)
        err_cf = (p_cf - batch["rgb_gt"]).abs().mean(dim=(1, 2, 3))
        out[f"rank_{name}"] = float((err_t < err_cf).float().mean().item())
    vis.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dyn", default="/hy-tmp/models/uwam/best.pt")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume-vis", default="", help="optional visual_*.pt to continue")
    args = ap.parse_args()

    cfg = Cfg()
    cfg.paths.usim = Path(args.usim)
    cfg.model.use_language = False
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.workers
    ensure_dirs(cfg)
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt_dyn, map_location=device, weights_only=False)
    train_eps = load_split(cfg.paths.usim, "train", cfg.schema, max_episodes=args.max_episodes)
    test_eps = load_split(cfg.paths.usim, "test", cfg.schema, max_episodes=max(40, args.max_episodes // 5))
    dyn_ds = DynamicsWindowDataset(train_eps, cfg, fit_norm=True)
    if "dyn_norm" in ckpt:
        dyn_ds.dyn_norm.load_state_dict(ckpt["dyn_norm"])
        dyn_ds.pwm_norm.load_state_dict(ckpt["pwm_norm"])
    train_v = VisualWindowDataset(train_eps, cfg, dyn_ds.dyn_norm, dyn_ds.pwm_norm, stride=8)
    test_v = VisualWindowDataset(test_eps or train_eps[:20], cfg, dyn_ds.dyn_norm, dyn_ds.pwm_norm, stride=12)
    print(f"visual windows train={len(train_v)} test={len(test_v)}", flush=True)
    if len(train_v) == 0:
        raise FileNotFoundError("no ego mp4 videos found under USIM")

    dyn = DynamicsWAM(cfg).to(device)
    dyn.load_state_dict(ckpt["model"])
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)
    vis = VisualWAM(cfg).to(device)
    if args.resume_vis:
        vckpt = torch.load(args.resume_vis, map_location=device, weights_only=False)
        vis.load_state_dict(vckpt["vis"])
        print(f"resume vis {args.resume_vis} metrics={vckpt.get('metrics')}", flush=True)
    opt = torch.optim.AdamW(vis.parameters(), lr=1e-4, weight_decay=1e-4)
    lpips_mod = OptionalLPIPS().to(device)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    nw = int(args.workers)
    train_loader = DataLoader(train_v, batch_size=args.batch_size, shuffle=True, num_workers=nw,
                              collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_v, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    history = []
    best_rank = -1.0
    warmup = 0 if args.resume_vis else max(1, args.epochs // 3)
    for epoch in range(1, args.epochs + 1):
        vis.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        use_rank = epoch > warmup
        pbar = tqdm(train_loader, desc=f"vis {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                with torch.no_grad():
                    d = dyn.disturbance(batch["hist_s"], batch["hist_a"])
                pred, _ = vis(batch["rgb"], batch["a_vis"], d)
                l_pix = pixel_l1(pred, batch["rgb_gt"])
                l_lp = lpips_mod(pred, batch["rgb_gt"])
                loss = cfg.train.lambda_pix * l_pix + cfg.train.lambda_lpips * l_lp
                if use_rank:
                    err_t = (pred - batch["rgb_gt"]).abs().mean(dim=(1, 2, 3))
                    l_rank = err_t.new_zeros(())
                    w = (batch["a_vis"].abs().mean(dim=(1, 2)) > 0.12).float()
                    for key in ("a_rand", "a_zero", "a_rev"):
                        pred_cf, _ = vis(batch["rgb"], batch[key], d)
                        err_cf = (pred_cf - batch["rgb_gt"]).abs().mean(dim=(1, 2, 3))
                        per = torch.relu(0.01 + err_t - err_cf)
                        l_rank = l_rank + (per * w).sum() / w.sum().clamp(min=1.0)
                    loss = loss + 0.8 * l_rank
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(vis.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * batch["rgb"].size(0)
            n_seen += batch["rgb"].size(0)
            pbar.set_postfix(pix=f"{float(l_pix.detach()):.4f}")

        vis.eval()
        acc = {"pix": 0.0, "copy": 0.0, "beat_copy": 0.0, "rank_zero": 0.0, "rank_rev": 0.0, "rank_rand": 0.0, "n": 0}
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                m = visual_rank(vis, dyn, batch, device)
                bsz = batch["rgb"].size(0)
                for k in ("pix", "copy", "beat_copy", "rank_zero", "rank_rev", "rank_rand"):
                    acc[k] += m[k] * bsz
                acc["n"] += bsz
        n = max(1, acc["n"])
        rec = {k: (acc[k] / n if k != "n" else acc[k]) for k in acc}
        rec.update({"epoch": epoch, "train_loss": running / max(1, n_seen), "sec": time.time() - t0, "rank_on": use_rank})
        history.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
        torch.save({"vis": vis.state_dict(), "metrics": rec}, cfg.paths.ckpt / "visual_last.pt")
        # paper metric is true-vs-random; keep pix from exploding vs copy
        ok_pix = rec["pix"] <= 2.0 * max(1e-6, rec["copy"]) or rec["pix"] <= 0.12
        if ok_pix and rec["rank_rand"] >= best_rank:
            best_rank = rec["rank_rand"]
            torch.save({"vis": vis.state_dict(), "metrics": rec}, cfg.paths.ckpt / "visual_best.pt")
    (cfg.paths.logs / "stage2_history.json").write_text(json.dumps(history, indent=2))
    print("saved visual_best.pt", flush=True)


if __name__ == "__main__":
    main()
