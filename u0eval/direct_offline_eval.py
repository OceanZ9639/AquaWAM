#!/usr/bin/env python3
"""Offline check of the amortized action head against the planners it replaces.

On held-out (window, goal) pairs from the teacher set, compare the imagined cost J (the planner's
own objective through the frozen core) of: the ONLINE sampling MPC (128 x 2, what the deployed
server runs), the OFFLINE teacher (512 x 3), and the WAM-direct head (one forward pass).
Writes u0eval/direct_offline_eval.json (read by ablation_tables.py).
"""
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

dev = "cuda" if torch.cuda.is_available() else "cpu"
core = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/models/uwam/best_scenes.pt"
head_ckpt = sys.argv[2] if len(sys.argv) > 2 else "/hy-tmp/models/uwam/direct_head.pt"
data = sys.argv[3] if len(sys.argv) > 3 else "/hy-tmp/data/direct_teacher.npz"
n_eval = int(sys.argv[4]) if len(sys.argv) > 4 else 300

m, dn, pn, cfg = load_core(core, dev)
cost = ImaginedCost(m, dn, pn, cfg, dev)
mpc = SamplingMPC(m, dn, pn, cfg.control, device=dev)  # online config (128 x 2)
mpc.set_library(np.concatenate([e.pwm for e in load_ou_split(Path("/hy-tmp/data/ou_explore"))], 0))
head = DirectPolicy.load(head_ckpt, m, dn, pn, dev)
z = np.load(data)
rng = np.random.default_rng(1)
# the trainer's validation split = first 5 % of permutation(seed 0); sample from it so nothing was trained on
N = len(z["J"]); val = np.random.default_rng(0).permutation(N)[: max(256, int(N * 0.05))]
idx = [i for i in rng.permutation(val) if z["J"][i] <= 5][:n_eval]
Jo, Jh, Jt, to, th = [], [], [], [], []
f = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None], device=dev)
for i in idx:
    hs, ha, st, g = z["hist_s"][i], z["hist_a"][i], z["s_t"][i], z["v_goal"][i]
    mpc.reset(); t0 = time.perf_counter()
    _, info = mpc.plan(hs, ha, st, g, task_index=0, sid_u=None, rpy=None, yaw_err=0.0); to.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); _, ih = head.plan(hs, ha, st, g); th.append(time.perf_counter() - t0)
    with torch.no_grad():
        jo, _ = cost(f(hs), f(ha), f(st), f(g), f(info["seq"])); jh, _ = cost(f(hs), f(ha), f(st), f(g), f(ih["seq"]))
    Jo.append(jo.item()); Jh.append(jh.item()); Jt.append(float(z["J"][i]))
Jo, Jh, Jt = map(np.array, (Jo, Jh, Jt))
out = {"n": int(len(Jo)), "J_online_mpc_128x2": float(Jo.mean()), "J_teacher_512x3": float(Jt.mean()),
       "J_direct_head": float(Jh.mean()), "frac_head_le_online": float(np.mean(Jh <= Jo)),
       "ratio_head_over_online_p50": float(np.median(Jh / Jo)), "ms_mpc": float(1e3 * np.mean(to)),
       "ms_head": float(1e3 * np.mean(th)), "core": core, "head": head_ckpt}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
