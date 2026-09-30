#!/usr/bin/env python3
"""
Train the wrist-camera relative-pose head on packs from percept/pack_grasp_recordings.py.

  loss = Gaussian NLL(obj_ee | mu, sigma)  +  MSE((sin 2psi, cos 2psi))
Episodes (not frames) are held out for validation. Reports xyz MAE split at 0.4 m gripper-object
distance (the acceptance test: < 2 cm inside 0.4 m) and the error/sigma correlation.
"""
from __future__ import annotations

import argparse, os
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrist_pose_model import MEAN, STD, WristPose  # noqa: E402


MAX_RANGE = float(os.environ.get('WRIST_MAX_RANGE', '2.5'))


class GraspFrames(Dataset):
    def __init__(self, files, reps, idx_filter=None, train=True, joint_norm=None):
        arrs = {k: [] for k in ("wrist_jpg", "ego_jpg", "obj_ee", "rel_yaw", "joints", "dist", "episode")}
        for f, rep in zip(files, reps):
            z = np.load(f, allow_pickle=True)
            for _ in range(rep):
                for k in arrs:
                    arrs[k].append(z[k])
            print(f"  {Path(f).name}: {len(z['dist'])} frames x{rep}", flush=True)
        self.d = {k: np.concatenate(v) for k, v in arrs.items()}
        # simulator blow-ups (vehicle teleported to ~3e5 m) leave absurd labels; also drop frames with
        # the object far outside any plausible reach of the wrist camera
        oe = self.d["obj_ee"].astype(np.float64)
        ok = np.isfinite(oe).all(1) & (np.linalg.norm(oe, axis=1) < MAX_RANGE) & np.isfinite(self.d["dist"]) & \
             np.isfinite(self.d["joints"].astype(np.float64)).all(1) & (np.abs(self.d["joints"]).max(1) < 10)
        if idx_filter is not None:
            ok &= idx_filter
        dropped = int((~ok).sum()) if idx_filter is None else int((~ok & idx_filter).sum())
        self.d = {k: v[ok] for k, v in self.d.items()}
        if dropped:
            print(f"  dropped {dropped} frames with blow-up / out-of-range labels", flush=True)
        self.train = train
        j = self.d["joints"].astype(np.float32)
        self.joint_norm = joint_norm or (j.mean(0), j.std(0) + 1e-3)

    def __len__(self):
        return len(self.d["dist"])

    def _img(self, buf):
        if buf is None or len(buf) == 0:
            return torch.zeros(3, 224, 224)
        im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.train:
            a = np.random.uniform(0.8, 1.2); b = np.random.uniform(-20, 20)
            im = np.clip(im.astype(np.float32) * a + b, 0, 255)
        im = (im.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(im.transpose(2, 0, 1).copy())

    def __getitem__(self, i):
        j = (self.d["joints"][i].astype(np.float32) - self.joint_norm[0]) / self.joint_norm[1]
        psi = float(self.d["rel_yaw"][i])
        return (self._img(self.d["wrist_jpg"][i]), self._img(self.d["ego_jpg"][i]), torch.from_numpy(j.astype(np.float32)),
                torch.from_numpy(self.d["obj_ee"][i].astype(np.float32)),
                torch.tensor([math.sin(2 * psi), math.cos(2 * psi)], dtype=torch.float32),
                torch.tensor(float(self.d["dist"][i])))


def evaluate(model, loader, device):
    model.eval()
    err, sig, dist = [], [], []
    with torch.no_grad():
        for w, e, j, y, yaw2, d in loader:
            mu, logvar, _ = model(w.to(device), e.to(device), j.to(device))
            err.append((mu.cpu() - y).abs().numpy()); sig.append(torch.exp(0.5 * logvar).cpu().numpy()); dist.append(d.numpy())
    err, sig, dist = np.concatenate(err), np.concatenate(sig), np.concatenate(dist)
    near = dist < 0.4
    out = {"mae_xyz_near_cm": float(err[near].mean() * 100) if near.any() else None,
           "mae_xy_near_cm": float(err[near][:, :2].mean() * 100) if near.any() else None,
           "mae_xyz_far_cm": float(err[~near].mean() * 100) if (~near).any() else None,
           "err_sigma_corr": float(np.corrcoef(err.mean(1), sig.mean(1))[0, 1]) if len(err) > 2 else None,
           "n_near": int(near.sum()), "n_far": int((~near).sum())}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/hy-tmp/data/grasp/wrist_pose.npz", help="comma-separated packs")
    ap.add_argument("--reps", default="", help="repeats per pack, comma-separated")
    ap.add_argument("--encoder", default="base")
    ap.add_argument("--no-ego", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--freeze-blocks", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="/hy-tmp/models/uwam/wrist_pose.pt")
    ap.add_argument("--report", default="/hy-tmp/logs/uwam/wrist_pose_report.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = [f for f in args.data.split(",") if f]
    reps = [int(r) for r in args.reps.split(",") if r] or [1] * len(files)

    # episode-level split (on the first pack's episode ids, replicated packs share them)
    all_eps = np.concatenate([np.load(f, allow_pickle=True)["episode"] for f in files])
    uniq = np.unique(all_eps)
    rng = np.random.default_rng(0)
    val_eps = set(rng.choice(uniq, size=max(1, int(len(uniq) * args.val_frac)), replace=False).tolist())
    eps_all = np.concatenate([np.repeat(np.load(f, allow_pickle=True)["episode"][None], r, axis=0).reshape(-1)
                              for f, r in zip(files, reps)])
    is_val = np.array([e in val_eps for e in eps_all])
    tr = GraspFrames(files, reps, idx_filter=~is_val, train=True)
    va = GraspFrames(files, reps, idx_filter=is_val, train=False, joint_norm=tr.joint_norm)
    print(f"train {len(tr)} frames / val {len(va)} frames ({len(val_eps)} held-out episodes)", flush=True)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=64, shuffle=False, num_workers=args.workers)

    model = WristPose(args.encoder, freeze_blocks=args.freeze_blocks, use_ego=not args.no_ego).to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone") and p.requires_grad]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": hd, "lr": args.lr_head}], weight_decay=0.05)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr_head], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, hist, t0 = float("inf"), [], time.time()
    for ep in range(args.epochs):
        run = {"nll": 0.0, "yaw": 0.0, "n": 0}
        for w, e, j, y, yaw2, _ in tl:
            w, e, j, y, yaw2 = (t.to(device, non_blocking=True) for t in (w, e, j, y, yaw2))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                mu, logvar, yh = model(w, e, j)
            mu, logvar, yh = mu.float(), logvar.float(), yh.float()
            nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp()).mean()
            lyaw = F.mse_loss(yh, yaw2)
            loss = nll + 0.5 * lyaw
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run["nll"] += nll.item() * len(y); run["yaw"] += lyaw.item() * len(y); run["n"] += len(y)
        ev = evaluate(model, vl, device)
        rec = {"epoch": ep + 1, "train_nll": run["nll"] / run["n"], "train_yaw": run["yaw"] / run["n"], **ev,
               "minutes": (time.time() - t0) / 60}
        hist.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        score = ev["mae_xyz_near_cm"] if ev["mae_xyz_near_cm"] is not None else ev["mae_xyz_far_cm"]
        if score is not None and score < best:
            best = score
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "n_queries": 4,
                        "freeze_blocks": args.freeze_blocks, "use_ego": not args.no_ego,
                        "joint_norm": tuple(np.asarray(x, np.float32) for x in tr.joint_norm), "val": ev, "epoch": ep + 1},
                       args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"args": vars(args), "history": hist, "best_mae_xyz_near_cm": best}, indent=1))
    print(f"saved {args.out}; best near-field xyz MAE {best:.2f} cm", flush=True)


if __name__ == "__main__":
    main()
