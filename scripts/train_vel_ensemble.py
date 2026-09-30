#!/usr/bin/env python3
"""Bootstrap ensemble of dead-reckoning heads: calibrated uncertainty for the trust gate.

The gate needs a scale for "how wrong can the estimate be right now"; hand-tuned m/s thresholds
are exactly the magic numbers we want to remove. K heads, each trained on a bootstrap resample
with its own init, give a per-tick disagreement that tracks the estimation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg, enable_arm, ensure_dirs
from uwam.data import DynamicsWindowDataset, RunningNorm, collate, load_ou_split, load_split
from uwam.models import MLP, DynamicsWAM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dyn", default="/hy-tmp/models/uwam/best_ou.pt")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore")
    ap.add_argument("--extra", default="/hy-tmp/data/planner_mix",
                    help="comma-separated extra OU-schema dirs (e.g. planner_mix,deploy_rec)")
    ap.add_argument("--extra-reps", default="",
                    help="comma-separated repeat factors matching --extra (default 1 each)")
    ap.add_argument("--canon-pressure", type=float, default=None,
                    help="pin the pressure column (raw units) to this value for the heads")
    ap.add_argument("--canon-alt", type=float, default=None,
                    help="pin the altitude column (m) to this value for the heads")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="/hy-tmp/models/uwam/vel_ens.pt")
    ap.add_argument("--init", default="",
                    help="warm-start the heads from an existing ensemble checkpoint (few-shot adaptation: "
                         "fine-tune the deployed estimator instead of refitting it on 30 min of data)")
    ap.add_argument("--usim-episodes", type=int, default=0,
                    help="mix in this many USIM train episodes (armed-vehicle dynamics "
                         "for the manipulator-extended ensemble); 0 = OU/planner only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Cfg()
    cfg.model.use_language = False
    ensure_dirs(cfg)
    ck = torch.load(args.ckpt_dyn, map_location=device, weights_only=False)
    cfg.model.use_dt = bool(ck.get("cfg", {}).get("use_dt", False))
    if ck.get("cfg", {}).get("use_arm"):
        enable_arm(cfg)
        print(f"arm layout from checkpoint: dyn={cfg.model.dyn_state_dim} pwm={cfg.model.pwm_dim}",
              flush=True)
    dyn = DynamicsWAM(cfg).to(device)
    dyn.load_state_dict(ck["model"], strict=False)
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)
    dyn_norm, pwm_norm = RunningNorm(), RunningNorm()
    dyn_norm.load_state_dict(ck["dyn_norm"])
    pwm_norm.load_state_dict(ck["pwm_norm"])

    use_arm = bool(cfg.model.use_arm)
    eps = load_ou_split(Path(args.ou), use_arm=use_arm, n_joints=cfg.model.n_joints)
    extras = [e for e in args.extra.split(",") if e]
    reps = [int(r) for r in args.extra_reps.split(",") if r] or [1] * len(extras)
    for extra, rep in zip(extras, reps):
        extra = Path(extra)
        if extra.exists():
            more = load_ou_split(extra, use_arm=use_arm, n_joints=cfg.model.n_joints)
            print(f"extra {extra}: {len(more)} episodes x{rep}", flush=True)
            eps += more * rep
    if args.usim_episodes > 0:
        eps += load_split(cfg.paths.usim, "train", cfg.schema,
                          max_episodes=args.usim_episodes, use_arm=use_arm)
    ds = DynamicsWindowDataset(eps, cfg, dyn_norm=dyn_norm, pwm_norm=pwm_norm, fit_norm=False)
    n = len(ds)
    print(f"windows: {n} from {len(eps)} episodes", flush=True)

    in_dim = cfg.model.history_len * (cfg.model.dyn_state_dim + cfg.model.pwm_dim) + cfg.model.disturbance_dim
    # heteroscedastic heads: (mu[3], logvar[3]) so sigma includes aleatoric noise
    heads = [MLP(in_dim, 6, cfg.model.hidden, depth=3, dropout=cfg.model.dropout).to(device)
             for _ in range(args.k)]
    if args.init:
        ck0 = torch.load(args.init, map_location=device, weights_only=False)
        if int(ck0["in_dim"]) == in_dim and len(ck0["heads"]) >= args.k:
            for h, sd in zip(heads, ck0["heads"]):
                h.load_state_dict(sd)
            print(f"warm-started {args.k} heads from {args.init}", flush=True)
        else:
            print(f"WARNING: --init {args.init} incompatible (in_dim {ck0['in_dim']} vs {in_dim}); training from scratch",
                  flush=True)
    opts = [torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=1e-4) for h in heads]
    scale = torch.as_tensor(dyn_norm.std[0:3], dtype=torch.float32, device=device)

    # bootstrap resample per head, fixed across epochs; train only on the first 80% of
    # each episode's timeline so the calibration check below is honestly held out
    rng = np.random.default_rng(0)
    tr_pool = [i for i, (ei, t) in enumerate(ds.index)
               if t / max(1, ds.episodes[ei].dyn.shape[0] - 1) < 0.8]
    subsets = [torch.utils.data.Subset(ds, rng.choice(tr_pool, size=len(tr_pool)).tolist())
               for _ in range(args.k)]

    # canonical values in the dataset's normalized units (SamplingMPC applies the
    # same pinning on raw inputs before normalizing, so train == deploy)
    canon = {}
    if args.canon_pressure is not None:
        canon[9] = (args.canon_pressure - float(dyn_norm.mean[9])) / float(dyn_norm.std[9])
    if args.canon_alt is not None:
        canon[10] = (args.canon_alt - float(dyn_norm.mean[10])) / float(dyn_norm.std[10])
    if canon:
        print(f"canonicalized columns (normalized): {canon}", flush=True)

    def head_input(batch):
        hs = batch["hist_s"].to(device)
        if canon:
            hs = hs.clone()
            for col, val in canon.items():
                hs[:, :, col] = val
        hs = dyn.mask_dvl(hs)
        ha = batch["hist_a"].to(device)
        with torch.no_grad():
            d = dyn.disturbance(hs, ha)
        return torch.cat([torch.cat([hs, ha], dim=-1).flatten(1), d], dim=1)

    history = []
    for ep_i in range(1, args.epochs + 1):
        for k in range(args.k):
            torch.manual_seed(1000 * k + ep_i)
            loader = DataLoader(subsets[k], batch_size=args.batch_size, shuffle=True,
                                num_workers=6, collate_fn=collate)
            run, seen = 0.0, 0
            for batch in loader:
                x = head_input(batch)
                gt = batch["s_t"][:, 0:3].to(device)
                o = heads[k](x)
                mu, var = o[:, :3], o[:, 3:].clamp(-8, 4).exp()
                loss = torch.nn.functional.gaussian_nll_loss(mu, gt, var)
                opts[k].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads[k].parameters(), 1.0)
                opts[k].step()
                run += loss.item() * gt.size(0)
                seen += gt.size(0)
            print(f"epoch {ep_i} head {k}: loss {run / max(1, seen):.5f}", flush=True)

        # calibration check: does ensemble std track the actual error?
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=6, collate_fn=collate)
        errs, stds, n_te = 0.0, 0.0, 0
        cal_pairs = []
        with torch.no_grad():
            for batch in loader:
                m = batch["t_frac"] >= 0.8
                if not m.any():
                    continue
                x = head_input({k2: v[m] for k2, v in batch.items()})
                gt = batch["s_t"][m][:, 0:3].to(device)
                outs = torch.stack([h(x) for h in heads], 0)       # [K, B, 6]
                mus, vars_ = outs[:, :, :3], outs[:, :, 3:].clamp(-8, 4).exp()
                mu = mus.mean(0)
                var = (vars_ + mus.pow(2)).mean(0) - mu.pow(2)
                sd = var.clamp_min(1e-8).sqrt()
                e = ((mu - gt) * scale).abs().mean(dim=1)
                s = (sd * scale).mean(dim=1)
                errs += e.sum().item(); stds += s.sum().item(); n_te += int(m.sum())
                cal_pairs.append(torch.stack([e, s], 1).cpu())
        cal = torch.cat(cal_pairs)
        corr = float(np.corrcoef(cal[:, 0].numpy(), cal[:, 1].numpy())[0, 1])
        rec = {"epoch": ep_i, "vel_mae_ms": errs / n_te, "mean_std_ms": stds / n_te,
               "err_std_corr": round(corr, 4), "n_test": n_te}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"heads": [h.state_dict() for h in heads], "k": args.k,
                    "in_dim": in_dim, "hidden": cfg.model.hidden, "depth": 3,
                    "dropout": cfg.model.dropout, "metrics": rec,
                    "canon_pressure": args.canon_pressure, "canon_alt": args.canon_alt},
                   args.out)
    hist_path = Path(args.out).with_suffix("").as_posix() + "_history.json"
    Path("/hy-tmp/logs/uwam").mkdir(parents=True, exist_ok=True)
    Path("/hy-tmp/logs/uwam") .joinpath(Path(hist_path).name).write_text(json.dumps(history, indent=2))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
