#!/usr/bin/env python3
"""Perception-head feasibility probe: frozen DINOv2 features -> target_pos.

Input:  [ego_feat 768, wrist_feat 768, task one-hot 9] per frame
Output: 6-d body-frame goal pose [dx, dy, dz, roll, pitch, yaw]
Loss:   L1 on position + wrapped-angle L1 on orientation

The verdict metric is close-range position error on the GRASP task family
(|label position| < 0.5 m): the expert's own stage tolerance was 3 cm and the
judge's grasp criterion is 4 cm gripper-object distance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GRASP_TASKS = {1, 2, 6, 7}
N_TASKS = 9


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class Head(nn.Module):
    """6-d goal pose + 1 gripper-close logit."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_split(path: Path, state_norm=None):
    z = np.load(path)
    parts = [z["ego"], z["wrist"], np.eye(N_TASKS, dtype=np.float32)[z["task"]]]
    if "state" in z.files:
        st = z["state"].astype(np.float32)
        if state_norm is None:
            state_norm = (st.mean(0), st.std(0) + 1e-6)
        parts.append((st - state_norm[0]) / state_norm[1])
    x = np.concatenate(parts, axis=1)
    return x, z["target"], z["task"], z["valid"], state_norm


def bucket_report(err_pos, yaw_err, target, task, prefix="", bearing_err=None):
    d = np.linalg.norm(target[:, :3], axis=1)
    rows = {}
    for name, mask in (
        ("close(<0.5m)", d < 0.5),
        ("mid(0.5-2m)", (d >= 0.5) & (d < 2.0)),
        ("far(>2m)", d >= 2.0),
        ("all", np.ones_like(d, bool)),
    ):
        for fam, fmask in (("grasp", np.isin(task, list(GRASP_TASKS))),
                           ("nav", ~np.isin(task, list(GRASP_TASKS)))):
            m = mask & fmask
            if m.sum() == 0:
                continue
            rows[f"{prefix}{fam}/{name}"] = {
                "n": int(m.sum()),
                "pos_mae_m": float(err_pos[m].mean()),
                "pos_p90_m": float(np.percentile(err_pos[m], 90)),
                "yaw_mae_rad": float(yaw_err[m].mean()),
            }
            if bearing_err is not None:
                # navigation steers by the *bearing* of the goal; range is refined on approach
                rows[f"{prefix}{fam}/{name}"]["bearing_med_deg"] = float(np.median(bearing_err[m]))
                rows[f"{prefix}{fam}/{name}"]["bearing_p75_deg"] = float(np.percentile(bearing_err[m], 75))
    return rows


def bearing_deg(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    b = np.degrees(np.abs(np.arctan2(pred_xy[:, 1], pred_xy[:, 0]) - np.arctan2(gt_xy[:, 1], gt_xy[:, 0])))
    return np.minimum(b, 360.0 - b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/hy-tmp/data/percept_probe")
    ap.add_argument("--suffix", default="", help="'' for small, '_base' for dinov2-base feats")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dist-weight", action="store_true",
                    help="inverse-distance loss weighting (close-range precision matters most)")
    ap.add_argument("--extra", default="", help="extra set stem to mix (e.g. dagger)")
    ap.add_argument("--extra-frac", type=float, default=0.35,
                    help="target share of the extra set in the training mix")
    ap.add_argument("--out", default="/hy-tmp/models/uwam/percept_head.pt")
    ap.add_argument("--report", default="/hy-tmp/logs/uwam/percept_probe.json")
    ap.add_argument("--tasks", default="", help="comma-separated task indices to train/eval on (default all)")
    ap.add_argument("--bearing-loss", type=float, default=0.0,
                    help="weight of a 1-cos(bearing) term on the xy direction (navigation heads)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr, ttr, vtr, snorm = load_split(Path(args.data) / f"train_feats{args.suffix}.npz")
    xte, yte, tte, vte, _ = load_split(Path(args.data) / f"test_feats{args.suffix}.npz", snorm)
    gtr = gte = None
    gp = Path(args.data) / f"train_grip{args.suffix}.npy"
    if gp.exists():
        gtr = np.load(gp)
        gte = np.load(Path(args.data) / f"test_grip{args.suffix}.npy")
        print(f"gripper labels: train close-rate {gtr.mean():.3f}", flush=True)
    if args.extra:
        xd, yd, td, vd, _ = load_split(Path(args.data) / f"{args.extra}_feats{args.suffix}.npz", snorm)
        gpath = Path(args.data) / f"{args.extra}_grip{args.suffix}.npy"
        gd = np.load(gpath) if gpath.exists() else np.zeros(len(xd), np.float32)  # nav sets have no jaw
        if args.tasks:  # size the mix against the tasks actually trained on
            keep0 = [int(t) for t in args.tasks.split(",") if t.strip()]
            xd, yd, td, vd, gd = (a[np.isin(td, keep0)] for a in (xd, yd, td, vd, gd))
            n_ref = int(np.isin(ttr, keep0).sum())
        else:
            n_ref = len(xtr)
        reps = max(1, int(round(args.extra_frac * n_ref / max(1, len(xd)))))
        print(f"mixing {len(xd)} {args.extra} frames x{reps} "
              f"({reps*len(xd)/(n_ref+reps*len(xd)):.1%} of the trained tasks)", flush=True)
        xtr = np.concatenate([xtr] + [xd] * reps)
        ytr = np.concatenate([ytr] + [yd] * reps)
        ttr = np.concatenate([ttr] + [td] * reps)
        vtr = np.concatenate([vtr, np.ones(reps * len(xd), bool)])
        if gtr is not None:
            gtr = np.concatenate([gtr] + [gd] * reps)
    # zero-label frames = "expert had no goal yet": excluded from train and eval
    if gtr is not None:
        gtr, gte = gtr[vtr], gte[vte]
    xtr, ytr, ttr = xtr[vtr], ytr[vtr], ttr[vtr]
    xte, yte, tte = xte[vte], yte[vte], tte[vte]
    if args.tasks:
        keep = [int(t) for t in args.tasks.split(",") if t.strip()]
        ktr, kte = np.isin(ttr, keep), np.isin(tte, keep)
        xtr, ytr, ttr = xtr[ktr], ytr[ktr], ttr[ktr]
        xte, yte, tte = xte[kte], yte[kte], tte[kte]
        if gtr is not None:
            gtr, gte = gtr[ktr], gte[kte]
        print(f"task filter {keep}: train {len(xtr)} test {len(xte)}", flush=True)
    print(f"train {len(xtr)} frames, test {len(xte)} frames (invalid dropped: "
          f"{int((~vtr).sum())}/{int((~vte).sum())})", flush=True)

    model = Head(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr)
    gtr_t = None if gtr is None else torch.from_numpy(gtr.astype(np.float32))
    bce = nn.BCEWithLogitsLoss()

    n = len(xtr_t)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            pred = model(xb)
            if args.dist_weight:
                w = 1.0 / yb[:, :3].norm(dim=1).clamp(min=0.3)
                w = (w / w.mean()).unsqueeze(1)
            else:
                w = torch.ones(len(yb), 1, device=device)
            loss_pos = (w * (pred[:, :3] - yb[:, :3]).abs()).mean()
            loss_ang = (w * wrap(pred[:, 3:6] - yb[:, 3:6]).abs()).mean()
            loss = loss_pos + 0.5 * loss_ang
            if args.bearing_loss > 0:
                # direction of the xy goal, for targets far enough that a bearing is defined
                gxy = yb[:, :2]
                far = (gxy.norm(dim=1) > 0.5).float()
                cos = nn.functional.cosine_similarity(pred[:, :2], gxy, dim=1, eps=1e-6)
                loss = loss + args.bearing_loss * ((1.0 - cos) * far).sum() / far.sum().clamp(min=1.0)
            if gtr_t is not None:
                loss = loss + 0.3 * bce(pred[:, 6], gtr_t[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        print(f"epoch {ep+1}/{args.epochs} loss {tot/n:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(xte), args.batch):
            preds.append(model(torch.from_numpy(xte[i:i + args.batch]).to(device)).cpu().numpy())
        pred = np.concatenate(preds)
    err_pos = np.linalg.norm(pred[:, :3] - yte[:, :3], axis=1)
    yaw_err = np.abs(np.arctan2(np.sin(pred[:, 5] - yte[:, 5]), np.cos(pred[:, 5] - yte[:, 5])))
    report = bucket_report(err_pos, yaw_err, yte, tte, bearing_err=bearing_deg(pred[:, :2], yte[:, :2]))
    if gte is not None and pred.shape[1] > 6:
        p = 1 / (1 + np.exp(-pred[:, 6]))
        grasp_mask = np.isin(tte, list(GRASP_TASKS))
        report["gripper"] = {
            "acc": float(((p > 0.5) == (gte > 0.5)).mean()),
            "acc_grasp_family": float(((p > 0.5) == (gte > 0.5))[grasp_mask].mean()),
            "close_recall": float(((p > 0.5) & (gte > 0.5)).sum() / max(1, (gte > 0.5).sum())),
            "close_precision": float(((p > 0.5) & (gte > 0.5)).sum() / max(1, (p > 0.5).sum())),
        }
    print(json.dumps(report, indent=1))
    torch.save({"model": model.state_dict(), "in_dim": xte.shape[1],
                "state_norm": None if snorm is None else [snorm[0].tolist(), snorm[1].tolist()],
                "suffix": args.suffix, "dist_weight": args.dist_weight}, args.out)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
