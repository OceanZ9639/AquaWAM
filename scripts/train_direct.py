#!/usr/bin/env python3
"""
Train the amortized action head (uwam/direct.py) on teacher labels from
scripts/build_direct_dataset.py.

    loss = MSE(a_hat, a_teacher)  +  beta * J_core(a_hat)

J_core is the planner's objective evaluated through the frozen dynamics core
(uwam.direct.ImaginedCost), so the head is pulled toward the big offline teacher's
choices AND toward low imagined cost.  Validation reports the head's imagined cost
against the teacher's on held-out windows -- if the head matches the 512x3 teacher
it also beats the 128x2 planner that actually runs online.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.direct import V_SCALE, DirectHead, ImaginedCost, load_core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/hy-tmp/data/direct_teacher.npz")
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_scenes.pt")
    ap.add_argument("--out", default="/hy-tmp/models/uwam/direct_head.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.1, help="weight of the imagined-cost term")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dyn_norm, pwm_norm, cfg = load_core(args.ckpt, device)
    cost = ImaginedCost(model, dyn_norm, pwm_norm, cfg, device)

    z = np.load(args.data)
    # simulator blow-ups leak into the MPC-in-the-loop dumps (|accel| in the thousands, |omega| ~800):
    # 0.3 % of the windows, all with absurd teacher costs; they must not shape the head
    hs_raw = z["hist_s"]
    ok = (z["J"] <= 5.0) & (np.abs(hs_raw[:, :, 6:9]).max(axis=(1, 2)) <= 30.0) & \
         (np.abs(hs_raw[:, :, 3:6]).max(axis=(1, 2)) <= 10.0)
    z = {k: (z[k][ok] if k in ("hist_s", "hist_a", "s_t", "v_goal", "seq", "J", "src") else z[k]) for k in z.files}
    print(f"dropped {int((~ok).sum())} blow-up / unreachable windows", flush=True)
    N = len(z["J"])
    idx = np.random.default_rng(args.seed).permutation(N)
    n_val = max(256, int(N * args.val_frac))
    va, tr = idx[:n_val], idx[n_val:]
    T = {k: torch.as_tensor(z[k], device=device) for k in ("hist_s", "hist_a", "s_t", "v_goal", "seq")}
    J_teacher = torch.as_tensor(z["J"], device=device)
    print(f"{N} samples ({len(tr)} train / {n_val} val), teacher {int(z['n_samples'])}x{int(z['cem_iters'])}, "
          f"teacher J mean {J_teacher.mean():.3f}", flush=True)

    head = DirectHead(cfg.model.disturbance_dim, cfg.model.dyn_state_dim, cfg.model.horizon_dyn,
                      args.hidden, args.depth).to(device)
    n_par = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(tr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def batch(ids):
        hs, ha, st, g, a = (T[k][ids] for k in ("hist_s", "hist_a", "s_t", "v_goal", "seq"))
        with torch.no_grad():
            d = model.disturbance(cost.norm_s(hs), cost.norm_a(ha))
        a_hat = head(d, cost.norm_s(st), g / V_SCALE, ha[:, -1])
        return hs, ha, st, g, a, a_hat, d

    def evaluate():
        head.eval()
        out = {"mse": 0.0, "J_head": 0.0, "J_teacher_core": 0.0, "n": 0}
        with torch.no_grad():
            for i in range(0, n_val, 1024):
                ids = torch.as_tensor(va[i:i + 1024], device=device)
                hs, ha, st, g, a, a_hat, d = batch(ids)
                jh, _ = cost(hs, ha, st, g, a_hat, d=d)
                jt, _ = cost(hs, ha, st, g, a, d=d)
                out["mse"] += ((a_hat - a) ** 2).mean(dim=(1, 2)).sum().item()
                out["J_head"] += jh.sum().item()
                out["J_teacher_core"] += jt.sum().item()
                out["n"] += len(ids)
        head.train()
        n = out.pop("n")
        return {k: v / n for k, v in out.items()}

    print(f"head params {n_par / 1e3:.0f}k; {steps} steps", flush=True)
    best, hist, t0 = float("inf"), [], time.time()
    for ep in range(args.epochs):
        perm = torch.as_tensor(np.random.permutation(tr), device=device)
        tot = {"loss": 0.0, "mse": 0.0, "J": 0.0, "n": 0}
        for i in range(0, len(perm), args.batch_size):
            ids = perm[i:i + args.batch_size]
            hs, ha, st, g, a, a_hat, d = batch(ids)
            mse = ((a_hat - a) ** 2).mean()
            j, _ = cost(hs, ha, st, g, a_hat, d=d)
            loss = mse + args.beta * j.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot["loss"] += loss.item() * len(ids); tot["mse"] += mse.item() * len(ids)
            tot["J"] += j.mean().item() * len(ids); tot["n"] += len(ids)
        ev = evaluate()
        rec = {"epoch": ep + 1, **{k: tot[k] / tot["n"] for k in ("loss", "mse", "J")}, **{f"val_{k}": v for k, v in ev.items()}}
        hist.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        score = ev["mse"] + args.beta * ev["J_head"]
        if score < best:
            best = score
            torch.save({"head": head.state_dict(), "dist_dim": cfg.model.disturbance_dim,
                        "state_dim": cfg.model.dyn_state_dim, "K": cfg.model.horizon_dyn,
                        "hidden": args.hidden, "depth": args.depth, "core": str(args.ckpt),
                        "teacher": str(args.data), "epoch": ep + 1, "val": ev, "beta": args.beta}, args.out)
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(hist, indent=1))
    print(f"saved {args.out}  best val score {best:.4f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
