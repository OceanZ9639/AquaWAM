#!/usr/bin/env python3
"""USIM-Hard P2/P3: per-task offline metrics of one or more stage-1 cores on the USIM *test* split.

For every checkpoint and every USIM task we report, in m/s:
  dyn_mae   : DVL-velocity prediction error of the action-conditioned rollout (planning quality)
  est_mae   : dead-reckoning estimator error with the DVL columns masked (blind-path quality)
  hover_mae : the trivial zero-velocity guess (what a blinded controller has without a model)
plus the same three numbers pooled over all tasks. Used for the data-efficiency curve
(cores trained on 5/10/25/50/100 % of USIM) and the held-out-task cores (train on some tasks,
read the error on the tasks never seen).

  python3 scripts/eval_per_task.py --ckpts a.pt,b.pt --out /hy-tmp/results/usim_hard/per_task.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/hy-tmp/underwater_wam")
sys.path.insert(0, "/hy-tmp/underwater_wam/u0eval")
from uwam.data import DynamicsWindowDataset, collate, load_split, load_tasks  # noqa: E402
from wam_policy_server import _load_model  # noqa: E402


@torch.no_grad()
def per_task(model, dyn_norm, loader, device) -> dict:
    scale = torch.as_tensor(dyn_norm.std[0:3], dtype=torch.float32, device=device)
    acc: dict[int, dict[str, float]] = {}
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b, use_history=True)
        p = dyn_norm.invert_torch(out["s_hat"])[..., 0:3]
        g = dyn_norm.invert_torch(b["s_fut"])[..., 0:3]
        dyn_err = (p - g).abs().mean(dim=(1, 2))
        v_hat = model.estimate_velocity(model.mask_dvl(b["hist_s"]), b["hist_a"])
        gt = b["s_t"][:, 0:3]
        est_err = ((v_hat - gt) * scale).abs().mean(dim=1)
        hover = (gt * scale).abs().mean(dim=1)
        ti = b["task_index"].view(-1).cpu().numpy()
        for t in np.unique(ti):
            m = torch.as_tensor(ti == t, device=device)
            a = acc.setdefault(int(t), {"dyn": 0.0, "est": 0.0, "hover": 0.0, "n": 0})
            a["dyn"] += float(dyn_err[m].sum())
            a["est"] += float(est_err[m].sum())
            a["hover"] += float(hover[m].sum())
            a["n"] += int(m.sum())
    res = {}
    tot = {"dyn": 0.0, "est": 0.0, "hover": 0.0, "n": 0}
    for t, a in sorted(acc.items()):
        res[t] = {"dyn_mae": a["dyn"] / a["n"], "est_mae": a["est"] / a["n"],
                  "hover_mae": a["hover"] / a["n"], "n": a["n"]}
        for k in tot:
            tot[k] += a[k]
    res["all"] = {"dyn_mae": tot["dyn"] / tot["n"], "est_mae": tot["est"] / tot["n"],
                  "hover_mae": tot["hover"] / tot["n"], "n": tot["n"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--out", default="/hy-tmp/results/usim_hard/per_task.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    tasks = load_tasks(Path(args.usim) / "test")

    test_eps = None
    for ck in [c for c in args.ckpts.split(",") if c.strip()]:
        model, dyn_norm, pwm_norm, cfg = _load_model(Path(ck), device)
        model.eval()
        if test_eps is None or getattr(cfg.model, "use_arm", False):
            test_eps = load_split(Path(args.usim), "test", cfg.schema,
                                  use_arm=bool(getattr(cfg.model, "use_arm", False)))
        ds = DynamicsWindowDataset(test_eps, cfg, dyn_norm=dyn_norm, pwm_norm=pwm_norm, fit_norm=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.workers)
        res = per_task(model, dyn_norm, loader, device)
        name = Path(ck).stem
        report[name] = {(tasks.get(k, str(k)) if isinstance(k, int) else k): v for k, v in res.items()}
        a = res["all"]
        print(f"{name:28s} dyn {a['dyn_mae']:.4f}  est {a['est_mae']:.4f}  hover {a['hover_mae']:.4f}  "
              f"(n={a['n']})", flush=True)
        for k, v in res.items():
            if k != "all":
                print(f"    {tasks.get(k, str(k)):52s} dyn {v['dyn_mae']:.4f}  est {v['est_mae']:.4f}  "
                      f"hover {v['hover_mae']:.4f}", flush=True)
        out_path.write_text(json.dumps(report, indent=2))
        del model, ds, loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
