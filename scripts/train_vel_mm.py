#!/usr/bin/env python3
"""Train the multimodal dead-reckoning head (camera + FLS replacing the DVL).

Trains only VelMM on image-bearing OU episodes; the dynamics model stays frozen. Evaluation is a
time split (t_frac >= 0.8 held out) reporting proprio-only vs multimodal MAE on the same windows,
overall and on fault frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg, ensure_dirs
from uwam.data import RunningNorm, load_ou_split
from uwam.models import DynamicsWAM
from uwam.velmm import VelMM, pack_img_pair


class MMWindow(Dataset):
    """Masked proprio window + stacked (prev, now) RGB+FLS frames + target velocity.

    Episodes either carry in-memory arrays (our OU/planner npz) or an ego video on disk
    (USIM LeRobot episodes), which is decoded lazily with a small per-episode cache.
    """

    def __init__(self, episodes, cfg, dyn_norm, pwm_norm, stride: int = 2, video_cache: int = 24):
        self.cfg = cfg
        self.L = cfg.model.history_len
        self.dyn_norm = dyn_norm
        self.pwm_norm = pwm_norm
        self.eps = [
            e for e in episodes
            if e.rgb_arr is not None or e.fls_arr is not None
            or (e.ego_video is not None and e.ego_video.exists())
        ]
        self._cache: dict = {}
        self._cache_max = video_cache
        self.index = []
        for i, ep in enumerate(self.eps):
            n_img = ep.dyn.shape[0]
            for t in range(self.L, min(ep.dyn.shape[0], n_img) - 1, stride):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def _video_rgb(self, ei: int):
        """Decoded ego video as uint8 [T, H, W, 3], LRU-cached per episode."""
        if ei not in self._cache:
            from uwam.data import _read_video_frames

            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            fr = _read_video_frames(self.eps[ei].ego_video, self.cfg.schema.image_hw)
            if fr is not None:  # [T,3,H,W] float01 -> uint8 HWC
                fr = (fr.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            self._cache[ei] = fr
        return self._cache[ei]

    def __getitem__(self, idx):
        ei, t = self.index[idx]
        ep = self.eps[ei]
        dyn = self.dyn_norm(ep.dyn)
        pwm = self.pwm_norm(ep.pwm)
        rgb = ep.rgb_arr
        if rgb is None and ep.ego_video is not None:
            rgb = self._video_rgb(ei)
        fls = ep.fls_arr
        t_img = min(t, (len(rgb) - 1) if rgb is not None else t)
        pair = pack_img_pair(
            None if rgb is None else rgb[max(0, t_img - 1)],
            None if fls is None else fls[t - 1],
            None if rgb is None else rgb[t_img],
            None if fls is None else fls[t],
            hw=self.cfg.schema.image_hw,
        )
        eta_t = ep.eta_arr[t] if ep.eta_arr is not None else np.ones(8, np.float32)
        return {
            "hist_s": torch.from_numpy(dyn[t - self.L : t].astype(np.float32)),
            "hist_a": torch.from_numpy(pwm[t - self.L : t].astype(np.float32)),
            "v_t": torch.from_numpy(dyn[t, 0:3].astype(np.float32)),
            "img": torch.from_numpy(pair),
            "faulty": torch.tensor(float(eta_t.min() < 0.99)),
            "task": torch.tensor(int(ep.task_index), dtype=torch.long),
            "t_frac": torch.tensor(t / max(1, ep.dyn.shape[0] - 1), dtype=torch.float32),
        }


def run_usim_paired(args, cfg, dyn, dyn_norm, pwm_norm, device):
    """Does RGB help dead reckoning in USIM's rich task scenes?

    Two identical-capacity VelMM heads are trained on the same batches; one sees the real ego
    frames, the other sees zeros. The only variable is image content, so the comparison is paired
    by construction. Reported overall and per task (pipeline, ship, charge station, ...).
    """
    from uwam.data import load_split, load_tasks

    all_eps = load_split(Path(args.usim), "train", cfg.schema)
    tasks = load_tasks(Path(args.usim) / "train")
    with_video = [e for e in all_eps if e.ego_video is not None and e.ego_video.exists()]
    # balance across tasks so the per-task breakdown has support everywhere
    by_task = {}
    for e in with_video:
        by_task.setdefault(e.task_index, []).append(e)
    per = max(1, args.usim_episodes // max(1, len(by_task)))
    eps = [e for k in sorted(by_task) for e in by_task[k][:per]]
    print(f"usim episodes: {len(eps)} across {len(by_task)} tasks ({per}/task)", flush=True)

    ds = MMWindow(eps, cfg, dyn_norm, pwm_norm, stride=3, video_cache=len(eps) + 1)
    print(f"pre-decoding {len(ds.eps)} ego videos ...", flush=True)
    for i in range(len(ds.eps)):
        ds._video_rgb(i)
        if (i + 1) % 30 == 0:
            print(f"  decoded {i + 1}/{len(ds.eps)}", flush=True)
    print(f"windows: {len(ds)}", flush=True)
    tr_idx = [i for i, (ei, t) in enumerate(ds.index)
              if t / max(1, ds.eps[ei].dyn.shape[0] - 1) < 0.8]
    te_set = set(range(len(ds))) - set(tr_idx)
    ltr = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=args.batch_size,
                     shuffle=True, num_workers=0)
    lte = DataLoader(torch.utils.data.Subset(ds, sorted(te_set)), batch_size=args.batch_size,
                     shuffle=False, num_workers=0)

    arm_rgb = VelMM(cfg).to(device)
    arm_blank = VelMM(cfg).to(device)
    opt_r = torch.optim.AdamW(arm_rgb.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(arm_blank.parameters(), lr=args.lr, weight_decay=1e-4)
    scale = torch.as_tensor(dyn_norm.std[0:3], dtype=torch.float32, device=device)

    @torch.no_grad()
    def evaluate():
        arm_rgb.eval(); arm_blank.eval()
        tot = {"rgb": 0.0, "blank": 0.0, "hover": 0.0}
        per_task = {}
        n = 0
        for b in lte:
            hs = dyn.mask_dvl(b["hist_s"].to(device))
            ha = b["hist_a"].to(device)
            gt = b["v_t"].to(device)
            img = b["img"].to(device)
            e_r = ((arm_rgb(dyn, hs, ha, img) - gt) * scale).abs().mean(dim=1)
            e_b = ((arm_blank(dyn, hs, ha, torch.zeros_like(img)) - gt) * scale).abs().mean(dim=1)
            tot["rgb"] += e_r.sum().item()
            tot["blank"] += e_b.sum().item()
            tot["hover"] += (gt * scale).abs().mean(dim=1).sum().item()
            n += gt.size(0)
            for k in torch.unique(b["task"]):
                m = (b["task"] == k).to(device)
                d = per_task.setdefault(int(k), {"rgb": 0.0, "blank": 0.0, "n": 0})
                d["rgb"] += e_r[m].sum().item()
                d["blank"] += e_b[m].sum().item()
                d["n"] += int(m.sum())
        arm_rgb.train(); arm_blank.train()
        out = {k: tot[k] / max(1, n) for k in tot}
        out["n"] = n
        out["per_task"] = {
            tasks.get(k, str(k)): {"rgb": round(v["rgb"] / v["n"], 5),
                                   "blank": round(v["blank"] / v["n"], 5), "n": v["n"]}
            for k, v in sorted(per_task.items()) if v["n"] > 0
        }
        return out

    history = []
    for ep_i in range(1, args.epochs + 1):
        run_r = run_b = seen = 0
        for b in ltr:
            hs = dyn.mask_dvl(b["hist_s"].to(device))
            ha = b["hist_a"].to(device)
            gt = b["v_t"].to(device)
            img = b["img"].to(device)
            l_r = torch.nn.functional.smooth_l1_loss(arm_rgb(dyn, hs, ha, img), gt)
            opt_r.zero_grad(set_to_none=True); l_r.backward()
            torch.nn.utils.clip_grad_norm_(arm_rgb.parameters(), 1.0); opt_r.step()
            l_b = torch.nn.functional.smooth_l1_loss(
                arm_blank(dyn, hs, ha, torch.zeros_like(img)), gt)
            opt_b.zero_grad(set_to_none=True); l_b.backward()
            torch.nn.utils.clip_grad_norm_(arm_blank.parameters(), 1.0); opt_b.step()
            run_r += l_r.item() * gt.size(0); run_b += l_b.item() * gt.size(0); seen += gt.size(0)
        rec = {"epoch": ep_i, "loss_rgb": run_r / max(1, seen), "loss_blank": run_b / max(1, seen),
               **evaluate()}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"velmm": arm_rgb.state_dict(), "metrics": rec},
                   "/hy-tmp/models/uwam/vel_mm_usim.pt")
    Path("/hy-tmp/logs/uwam/vel_mm_usim_history.json").write_text(json.dumps(history, indent=2))
    print("saved /hy-tmp/models/uwam/vel_mm_usim.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dyn", default="/hy-tmp/models/uwam/best_ou.pt")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore")
    ap.add_argument("--extra", default="/hy-tmp/data/planner_mix_img",
                    help="extra image-bearing OU-schema dir (planner re-collect)")
    ap.add_argument("--usim-episodes", type=int, default=0,
                    help=">0: paired rich-scene experiment on USIM ego videos instead of OU data")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="/hy-tmp/models/uwam/vel_mm.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Cfg()
    cfg.model.use_language = False
    ensure_dirs(cfg)
    ck = torch.load(args.ckpt_dyn, map_location=device, weights_only=False)
    dyn = DynamicsWAM(cfg).to(device)
    dyn.load_state_dict(ck["model"], strict=False)
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)
    dyn_norm, pwm_norm = RunningNorm(), RunningNorm()
    dyn_norm.load_state_dict(ck["dyn_norm"])
    pwm_norm.load_state_dict(ck["pwm_norm"])

    if args.usim_episodes > 0:
        run_usim_paired(args, cfg, dyn, dyn_norm, pwm_norm, device)
        return

    eps = load_ou_split(Path(args.ou), load_images=True)
    extra_dir = Path(args.extra)
    if extra_dir.exists():
        more = load_ou_split(extra_dir, load_images=True)
        print(f"extra image episodes from {extra_dir}: {len(more)}", flush=True)
        eps += more
    ds = MMWindow(eps, cfg, dyn_norm, pwm_norm, stride=2)
    print(f"mm-vel windows: {len(ds)} from {len(ds.eps)} episodes", flush=True)
    tr_idx = [i for i, (ei, t) in enumerate(ds.index)
              if t / max(1, ds.eps[ei].dyn.shape[0] - 1) < 0.8]
    te_idx = [i for i in range(len(ds)) if i not in set(tr_idx)]
    tr = torch.utils.data.Subset(ds, tr_idx)
    te = torch.utils.data.Subset(ds, te_idx)
    ltr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
    lte = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=4)

    velmm = VelMM(cfg).to(device)
    opt = torch.optim.AdamW(velmm.parameters(), lr=args.lr, weight_decay=1e-4)
    scale = torch.as_tensor(dyn_norm.std[0:3], dtype=torch.float32, device=device)

    @torch.no_grad()
    def evaluate():
        velmm.eval()
        tot_mm, tot_prop, tot_hover, n = 0.0, 0.0, 0.0, 0
        tot_mm_f, tot_prop_f, n_f = 0.0, 0.0, 0
        for b in lte:
            hs = dyn.mask_dvl(b["hist_s"].to(device))
            ha = b["hist_a"].to(device)
            gt = b["v_t"].to(device)
            v_mm = velmm(dyn, hs, ha, b["img"].to(device))
            v_prop = dyn.estimate_velocity(hs, ha)
            e_mm = ((v_mm - gt) * scale).abs().mean(dim=1)
            e_pr = ((v_prop - gt) * scale).abs().mean(dim=1)
            tot_mm += e_mm.sum().item(); tot_prop += e_pr.sum().item()
            tot_hover += (gt * scale).abs().mean(dim=1).sum().item()
            n += gt.size(0)
            f = b["faulty"].to(device) > 0.5
            if f.any():
                tot_mm_f += e_mm[f].sum().item(); tot_prop_f += e_pr[f].sum().item()
                n_f += int(f.sum())
        velmm.train()
        return {
            "vel_mm_mae_ms": tot_mm / n,
            "vel_proprio_mae_ms": tot_prop / n,
            "hover_mae_ms": tot_hover / n,
            "vel_mm_mae_ms_fault": (tot_mm_f / n_f) if n_f else None,
            "vel_proprio_mae_ms_fault": (tot_prop_f / n_f) if n_f else None,
            "n": n, "n_fault": n_f,
        }

    history = []
    for ep_i in range(1, args.epochs + 1):
        run, seen = 0.0, 0
        for b in ltr:
            hs = dyn.mask_dvl(b["hist_s"].to(device))
            ha = b["hist_a"].to(device)
            gt = b["v_t"].to(device)
            v = velmm(dyn, hs, ha, b["img"].to(device))
            per = torch.nn.functional.smooth_l1_loss(v, gt, reduction="none").mean(dim=1)
            w = torch.where(b["faulty"].to(device) > 0.5,
                            torch.full_like(per, 3.0), torch.ones_like(per))
            loss = (w * per).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velmm.parameters(), 1.0)
            opt.step()
            run += loss.item() * gt.size(0); seen += gt.size(0)
        rec = {"epoch": ep_i, "train_loss": run / max(1, seen), **evaluate()}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"velmm": velmm.state_dict(), "metrics": rec}, args.out)
    Path("/hy-tmp/logs/uwam/vel_mm_history.json").write_text(json.dumps(history, indent=2))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
