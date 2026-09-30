#!/usr/bin/env python3
"""Which-thruster-failed linear probe on the disturbance token d.

Replaces the flow probe as the regime-identification claim: the ocean current is a no-op in this
Stonefish build, so "flow" has no physical referent, while thruster-efficiency faults are really
applied. Classes: 0 = healthy, 1 = thruster1@0.4, 2 = thruster2@0.5, 3 = thrusters4&5@0.5
(change_point after t=8 s). Time-split fit (t_frac < 0.8 train, rest test), no window shuffling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg
from uwam.data import DynamicsWindowDataset, RunningNorm, collate, load_ou_split
from uwam.models import DynamicsWAM

FAIL_ID = {
    "ou_nominal": 0,
    "ou_const_current": 0,
    "ou_time_varying": 0,
    "ou_current_and_fail": 1,
    "ou_thruster_degrade": 2,
    # change_point: 0 before the 8 s switch, 3 after
}


def fail_id_label(task: str, t_sec: float) -> int:
    if task == "ou_change_point":
        return 3 if t_sec >= 8.0 else 0
    return FAIL_ID.get(task, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_ou.pt")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore")
    ap.add_argument("--out", default="/hy-tmp/logs/uwam/fail_id_probe.json")
    ap.add_argument("--t-cut", type=float, default=0.8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Cfg()
    cfg.model.use_language = False
    model = DynamicsWAM(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    dyn_norm, pwm_norm = RunningNorm(), RunningNorm()
    dyn_norm.load_state_dict(ck["dyn_norm"])
    pwm_norm.load_state_dict(ck["pwm_norm"])

    eps = load_ou_split(Path(args.ou))
    names = {ep.episode_index: ep.task for ep in eps}
    ds = DynamicsWindowDataset(eps, cfg, dyn_norm=dyn_norm, pwm_norm=pwm_norm, fit_norm=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, collate_fn=collate)

    ztr, ytr, zte, yte = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            d = model.disturbance(batch["hist_s"].to(device), batch["hist_a"].to(device)).cpu()
            for i in range(d.size(0)):
                task = names[int(batch["episode_index"][i])]
                y = fail_id_label(task, float(batch["t_sec"][i]))
                if float(batch["t_frac"][i]) < args.t_cut:
                    ztr.append(d[i]); ytr.append(y)
                else:
                    zte.append(d[i]); yte.append(y)
    ztr = torch.stack(ztr); zte = torch.stack(zte)
    ytr = torch.tensor(ytr); yte = torch.tensor(yte)
    n_cls = 4
    A = torch.cat([ztr, torch.ones(ztr.size(0), 1)], 1)
    B = torch.cat([zte, torch.ones(zte.size(0), 1)], 1)
    W = torch.linalg.solve(A.T @ A + 1e-2 * torch.eye(A.size(1)), A.T @ torch.nn.functional.one_hot(ytr, n_cls).float())
    pred = (B @ W).argmax(-1)
    acc = float((pred == yte).float().mean())
    recalls = {}
    for c in range(n_cls):
        m = yte == c
        recalls[c] = float((pred[m] == c).float().mean()) if m.any() else None
    bal = float(np.mean([v for v in recalls.values() if v is not None]))
    out = {
        "label": "fail_id (0 healthy / 1 thruster1 / 2 thruster2 / 3 thrusters4+5 post-switch)",
        "acc": round(acc, 4),
        "balanced_acc": round(bal, 4),
        "per_class_recall": {str(k): (None if v is None else round(v, 4)) for k, v in recalls.items()},
        "random": 0.25,
        "majority": round(float(max(np.bincount(yte.numpy(), minlength=n_cls)) / len(yte)), 4),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "split": f"time (t_frac < {args.t_cut} train)",
        "ckpt": args.ckpt,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
