#!/usr/bin/env python3
"""End-to-end perception for the world-model system: fine-tuned DINOv2 on the stereo/ego+wrist views
with spatial attention pooling, predicting the expert's INTENT.

Why this replaces the frozen-feature head (percept/train_head.py):
  * mean-pooled CLS+patch features discard WHERE the target is in the image; the two scenes that
    collapsed in closed loop (inspect_pipeline_sea, scan_ship_ancient) are exactly the ones that need
    image position (a pipe on the seabed, the edge of a wreck)
  * frozen ImageNet-domain features under-fit underwater rendering (range biased 2-3 m short far out)
  * "the expert's current node" is a discrete, jumping label; the expert's displacement over the
    next 3 s is smooth, always defined and carries the obstacle detours

Inputs match the VLA's: two RGB views, proprioception (joints, pressure, altitude), task id.
Outputs: disp3 (body-frame displacement over 3 s), wp6 (node in body frame, auxiliary; its yaw is
the look-at heading scan/inspect are judged on).

  python3 percept/train_e2e.py --epochs 6 --out /hy-tmp/models/uwam/percept_e2e.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ENCODERS = {
    "small": "/hy-tmp/models/hf_cache/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "base": "/hy-tmp/models/hf_cache/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}
N_TASKS = 9
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NAV_TASK_NAMES = {0: "charge station", 3: "scan ship", 4: "inspect pipeline", 5: "follow boat", 8: "water tower"}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PairSet(Dataset):
    def __init__(self, files, state_norm=None, train=True, reps=None):
        self.ego, self.wrist, self.disp, self.wp, self.state, self.task, self.valid = [], [], [], [], [], [], []
        for f, rep in zip(files, reps or [1] * len(files)):
            z = np.load(f, allow_pickle=True)
            n = len(z["task"])
            valid = z["valid"] if "valid" in z.files else np.ones(n, bool)
            for _ in range(rep):
                self.ego.append(z["ego"]); self.wrist.append(z["wrist"])
                self.disp.append(z["disp3"]); self.wp.append(z["wp6"]); self.state.append(z["state"])
                self.task.append(z["task"]); self.valid.append(valid)
            print(f"  {Path(f).name}: {n} frames x{rep}", flush=True)
        self.ego = np.concatenate(self.ego); self.wrist = np.concatenate(self.wrist)
        self.disp = np.concatenate(self.disp).astype(np.float32); self.wp = np.concatenate(self.wp).astype(np.float32)
        self.state = np.concatenate(self.state).astype(np.float32); self.task = np.concatenate(self.task)
        self.valid = np.concatenate(self.valid)
        if state_norm is None:
            state_norm = (self.state.mean(0), self.state.std(0) + 1e-6)
        self.state_norm = state_norm
        self.train = train

    def __len__(self):
        return len(self.task)

    def _img(self, buf):
        im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.train:
            # mild photometric jitter only (no flips / rotations: the labels are geometric)
            a = np.random.uniform(0.8, 1.2); b = np.random.uniform(-20, 20)
            im = np.clip(im.astype(np.float32) * a + b, 0, 255)
            if np.random.rand() < 0.5:
                s = np.random.uniform(0.88, 1.0); h = w = int(224 * s)
                y0 = np.random.randint(0, 225 - h); x0 = np.random.randint(0, 225 - w)
                im = cv2.resize(im[y0:y0 + h, x0:x0 + w], (224, 224), interpolation=cv2.INTER_LINEAR)
        im = (im.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(im.transpose(2, 0, 1).copy())

    def __getitem__(self, i):
        st = (self.state[i] - self.state_norm[0]) / self.state_norm[1]
        return (self._img(self.ego[i]), self._img(self.wrist[i]), torch.from_numpy(st),
                torch.tensor(int(self.task[i])), torch.from_numpy(self.disp[i]), torch.from_numpy(self.wp[i]),
                torch.tensor(bool(self.valid[i])))


class PerceptE2E(nn.Module):
    """Shared DINOv2 over both views, camera embeddings, attention pooling, MLP -> disp3 + wp6."""

    def __init__(self, encoder="base", n_queries=4, freeze_blocks=4):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(ENCODERS[encoder])
        d = self.backbone.config.hidden_size
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[:freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
        self.cam_emb = nn.Parameter(torch.zeros(2, 1, d))
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.task_emb = nn.Embedding(N_TASKS, 64)
        self.head = nn.Sequential(
            nn.Linear(n_queries * d + 64 + 7, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, 9),
        )

    def tokens(self, x):
        return self.backbone(pixel_values=x).last_hidden_state  # [B, 1+256, d]

    def forward(self, ego, wrist, state, task):
        B = ego.shape[0]
        t = torch.cat([self.tokens(ego) + self.cam_emb[0], self.tokens(wrist) + self.cam_emb[1]], dim=1)
        t = self.norm(t)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attn(q, t, t)
        h = torch.cat([pooled.flatten(1), self.task_emb(task), state], dim=-1)
        return self.head(h)   # [:, :3] disp3, [:, 3:9] wp6


def evaluate(model, loader, device):
    model.eval()
    acc = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ego, wrist, st, task, disp, wp, valid in loader:
            out = model(ego.to(device), wrist.to(device), st.to(device), task.to(device)).float().cpu()
            pd, pw = out[:, :3], out[:, 3:9]
            for k in range(len(task)):
                ti = int(task[k]); a = acc.setdefault(ti, {"n": 0, "disp_l1": 0.0, "bear": [], "mag": [], "wp_bear": [], "wp_yaw": []})
                a["n"] += 1
                a["disp_l1"] += float((pd[k] - disp[k]).abs().mean())
                gm = float(disp[k, :2].norm())
                if gm > 0.3:
                    b = math.degrees(abs(wrap(math.atan2(pd[k, 1], pd[k, 0]) - math.atan2(disp[k, 1], disp[k, 0]))))
                    a["bear"].append(b); a["mag"].append(abs(float(pd[k, :2].norm()) - gm))
                if bool(valid[k]) and float(wp[k, :2].norm()) > 0.5:
                    a["wp_bear"].append(math.degrees(abs(wrap(math.atan2(pw[k, 1], pw[k, 0]) - math.atan2(wp[k, 1], wp[k, 0])))))
                    a["wp_yaw"].append(abs(wrap(float(pw[k, 5] - wp[k, 5]))))
    rep = {}
    for ti, a in sorted(acc.items()):
        rep[NAV_TASK_NAMES.get(ti, str(ti))] = {
            "n": a["n"], "disp_l1": a["disp_l1"] / a["n"],
            "disp_bearing_med_deg": float(np.median(a["bear"])) if a["bear"] else None,
            "disp_bearing_p75_deg": float(np.percentile(a["bear"], 75)) if a["bear"] else None,
            "disp_mag_err_med_m": float(np.median(a["mag"])) if a["mag"] else None,
            "wp_bearing_med_deg": float(np.median(a["wp_bear"])) if a["wp_bear"] else None,
            "wp_yaw_mae_rad": float(np.mean(a["wp_yaw"])) if a["wp_yaw"] else None,
        }
    model.train()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/hy-tmp/data/percept_e2e")
    ap.add_argument("--extra", default="dagger_e2e.npz",
                    help="DAgger set file name(s) in --data, comma-separated ('' = none)")
    ap.add_argument("--extra-reps", default="1", help="repeats per --extra file, comma-separated")
    ap.add_argument("--encoder", default="base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--freeze-blocks", type=int, default=4)
    ap.add_argument("--out", default="/hy-tmp/models/uwam/percept_e2e.pt")
    ap.add_argument("--report", default="/hy-tmp/logs/uwam/percept_e2e_report.json")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    device = "cuda"
    files = [str(Path(args.data) / "train_e2e.npz")]; reps = [1]
    extras = [e for e in args.extra.split(",") if e]
    extra_reps = [int(r) for r in str(args.extra_reps).split(",") if r]
    for i, e in enumerate(extras):
        if not (Path(args.data) / e).exists():
            raise SystemExit(f"missing DAgger set {Path(args.data) / e}")
        files.append(str(Path(args.data) / e)); reps.append(extra_reps[min(i, len(extra_reps) - 1)])
    print("loading train set", flush=True)
    tr = PairSet(files, train=True, reps=reps)
    te = PairSet([str(Path(args.data) / "test_e2e.npz")], state_norm=tr.state_norm, train=False)
    print(f"train {len(tr)} frames, test {len(te)} frames", flush=True)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4)
    tel = DataLoader(te, batch_size=96, shuffle=False, num_workers=8)
    model = PerceptE2E(args.encoder, freeze_blocks=args.freeze_blocks).to(device)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone}, {"params": hd, "lr": args.lr_head}], weight_decay=0.05)
    steps = args.epochs * len(trl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr_head], total_steps=steps, pct_start=0.05)
    print(f"params trainable {sum(p.numel() for p in bb) / 1e6:.1f}M backbone + {sum(p.numel() for p in hd) / 1e6:.1f}M head; {steps} steps", flush=True)
    best = None
    for ep in range(args.epochs):
        t0 = time.time(); tot = 0.0; n = 0
        for i, (ego, wrist, st, task, disp, wp, valid) in enumerate(trl):
            ego, wrist, st, task = ego.to(device, non_blocking=True), wrist.to(device, non_blocking=True), st.to(device), task.to(device)
            disp, wp, valid = disp.to(device), wp.to(device), valid.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ego, wrist, st, task)
            out = out.float()
            pd, pw = out[:, :3], out[:, 3:9]
            l_disp = F.smooth_l1_loss(pd, disp, beta=0.1)
            far = (disp[:, :2].norm(dim=1) > 0.3).float()
            cos = F.cosine_similarity(pd[:, :2], disp[:, :2], dim=1, eps=1e-6)
            l_bear = ((1 - cos) * far).sum() / far.sum().clamp(min=1)
            vm = valid.float()
            l_wp = ((F.smooth_l1_loss(pw[:, :3], wp[:, :3], beta=0.1, reduction="none").mean(1)
                     + 0.5 * torch.atan2(torch.sin(pw[:, 5] - wp[:, 5]), torch.cos(pw[:, 5] - wp[:, 5])).abs()) * vm).sum() / vm.sum().clamp(min=1)
            loss = l_disp + 1.0 * l_bear + 0.3 * l_wp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss); n += 1
            if i % 200 == 0:
                print(f"ep {ep + 1}/{args.epochs} it {i}/{len(trl)} loss {loss.item():.4f} (disp {l_disp.item():.3f} bear {l_bear.item():.3f} wp {l_wp.item():.3f}) {(time.time() - t0) / max(i, 1):.2f}s/it", flush=True)
        rep = evaluate(model, tel, device)
        key = float(np.mean([v["disp_bearing_med_deg"] for v in rep.values() if v["disp_bearing_med_deg"] is not None]))
        print(f"=== epoch {ep + 1} train loss {tot / max(n, 1):.4f}  test mean disp-bearing {key:.1f} deg  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        for k, v in rep.items():
            print(f"    {k:18s} n={v['n']:6d} disp bearing med {v['disp_bearing_med_deg']} p75 {v['disp_bearing_p75_deg']} mag err {v['disp_mag_err_med_m']} | wp bearing {v['wp_bearing_med_deg']} yaw {v['wp_yaw_mae_rad']}", flush=True)
        if best is None or key < best:
            best = key
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "freeze_blocks": args.freeze_blocks,
                        "state_norm": [tr.state_norm[0].tolist(), tr.state_norm[1].tolist()], "epoch": ep + 1,
                        "report": rep}, args.out)
            print(f"    saved {args.out}", flush=True)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=1))
    print("E2E_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
