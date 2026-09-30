#!/usr/bin/env python3
"""Re-run recovery-paper metrics on a trained stage-1 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from uwam.config import Cfg
from uwam.data import DynamicsWindowDataset, RunningNorm, collate, load_split
from uwam.models import DynamicsWAM
from train_stage1 import _summarize, evaluate, linear_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best.pt")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    cfg = Cfg()
    cfg.paths.usim = Path(args.usim)
    cfg.model.n_tasks = 9
    cfg.model.use_language = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_eps = load_split(cfg.paths.usim, "train", cfg.schema, max_episodes=args.max_episodes)
    test_eps = load_split(cfg.paths.usim, "test", cfg.schema, max_episodes=args.max_episodes)
    train_ds = DynamicsWindowDataset(train_eps, cfg, fit_norm=True)
    if "dyn_norm" in ckpt:
        train_ds.dyn_norm.load_state_dict(ckpt["dyn_norm"])
        train_ds.pwm_norm.load_state_dict(ckpt["pwm_norm"])
    test_ds = DynamicsWindowDataset(test_eps or train_eps, cfg, dyn_norm=train_ds.dyn_norm,
                                    pwm_norm=train_ds.pwm_norm, fit_norm=False)
    slow_ds = DynamicsWindowDataset(test_eps or train_eps, cfg, dyn_norm=train_ds.dyn_norm,
                                    pwm_norm=train_ds.pwm_norm, fit_norm=False, slow_mode=True)
    model = DynamicsWAM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)
    slow_loader = DataLoader(slow_ds, batch_size=128, shuffle=False, collate_fn=collate)
    ev = evaluate(model, loader, device, True, train_ds.dyn_norm)
    ev_no = evaluate(model, loader, device, False, train_ds.dyn_norm)
    ev_slow = evaluate(model, slow_loader, device, True, train_ds.dyn_norm)
    probe = linear_probe(model, loader, device, 3, "current_bin")
    extra = _summarize(ev, ev_no, ev_slow, probe)
    report = {
        "eval": ev,
        "no_history": ev_no,
        "slow_mode": ev_slow,
        **extra,
        "paper_targets": {
            "ctx_gain_ms": 0.35,
            "slow_retained": 0.67,
            "probe_acc": 0.73,
            "probe_random": 0.33,
            "dvl_action_rank_k5_true_vs_random": 0.86,
        },
    }
    print(json.dumps(report, indent=2))
    Path("/hy-tmp/logs/uwam/eval_stage1.json").parent.mkdir(parents=True, exist_ok=True)
    Path("/hy-tmp/logs/uwam/eval_stage1.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
