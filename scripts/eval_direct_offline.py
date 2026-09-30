#!/usr/bin/env python3
"""Offline check of the amortized head against the online planner on held-out teacher windows:
imagined cost (same frozen core, same objective) and per-decision latency."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import SamplingMPC  # noqa: E402
from uwam.data import load_ou_split  # noqa: E402
from uwam.direct import DirectPolicy, ImaginedCost, load_core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_scenes.pt")
    ap.add_argument("--head", default="/hy-tmp/models/uwam/direct_head.pt")
    ap.add_argument("--data", default="/hy-tmp/data/direct_teacher.npz")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="/hy-tmp/results/direct_offline.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, dn, pn, cfg = load_core(args.ckpt, dev)
    cost = ImaginedCost(m, dn, pn, cfg, dev)
    mpc = SamplingMPC(m, dn, pn, cfg.control, device=dev)  # online config (128 x 2)
    mpc.set_library(np.concatenate([e.pwm for e in load_ou_split(Path("/hy-tmp/data/ou_explore"))], 0))
    head = DirectPolicy.load(args.head, m, dn, pn, dev)
    z = np.load(args.data)
    rng = np.random.default_rng(1)
    # the trainer's validation split is the first 5 % of permutation(seed 0); use the same windows
    N = len(z["J"])
    perm = np.random.default_rng(0).permutation(N)
    val = [i for i in perm[: max(256, int(N * 0.05))] if z["J"][i] <= 5][: args.n]
    f = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None], device=dev)
    Jo, Jh, Jt, to, th = [], [], [], [], []
    for i in val:
        hs, ha, st, g = z["hist_s"][i], z["hist_a"][i], z["s_t"][i], z["v_goal"][i]
        mpc.reset()
        t0 = time.perf_counter()
        _, info = mpc.plan(hs, ha, st, g, task_index=0, sid_u=None, rpy=None, yaw_err=0.0)
        to.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _, ih = head.plan(hs, ha, st, g)
        th.append(time.perf_counter() - t0)
        with torch.no_grad():
            jo, _ = cost(f(hs), f(ha), f(st), f(g), f(info["seq"]))
            jh, _ = cost(f(hs), f(ha), f(st), f(g), f(ih["seq"]))
        Jo.append(jo.item()); Jh.append(jh.item()); Jt.append(float(z["J"][i]))
    Jo, Jh, Jt = map(np.array, (Jo, Jh, Jt))
    res = {"n": int(len(Jo)), "J_online_mpc_128x2": float(Jo.mean()), "J_teacher_512x3": float(Jt.mean()),
           "J_direct_head": float(Jh.mean()), "frac_head_le_online": float(np.mean(Jh <= Jo)),
           "ratio_head_over_online_p50": float(np.median(Jh / np.maximum(Jo, 1e-6))),
           "latency_ms_mpc": float(1e3 * np.mean(to)), "latency_ms_head": float(1e3 * np.mean(th)),
           "head_params": int(head.n_params), "note": "latency measured on a GPU shared with training and the simulator"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
