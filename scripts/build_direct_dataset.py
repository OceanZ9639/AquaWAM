#!/usr/bin/env python3
"""
Teacher labels for the amortized action head (uwam/direct.py).

Windows (16-step history, current state) are drawn from every dynamics source the core
was trained on -- USIM expert demos, OU exploration, scene collection, deployment
recordings, MPC-in-the-loop dumps -- and paired with a sampled body-frame velocity goal.
A LARGE sampling MPC (512 candidates x 3 CEM rounds; the online planner runs 128 x 2)
solves each (window, goal) through the frozen core; its best 0.5 s PWM sequence and
imagined cost are the targets.  Attitude leveling is left out (rpy=None): the server
strips roll/pitch/yaw moments from the planned sequence and re-adds deterministic
leveling afterwards, so the head only has to learn the translational decision.

Output: <out>.npz with hist_s [N,16,19], hist_a [N,16,8], s_t [N,19], v_goal [N,3],
        seq [N,5,8] (raw PWM), J [N], src [N] (source id).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import SamplingMPC  # noqa: E402
from uwam.data import load_ou_split, load_split  # noqa: E402
from uwam.direct import load_core  # noqa: E402


def sample_goal(rng: np.random.Generator) -> np.ndarray:
    """Goal distribution matching the server's task goals: cruise-speed horizontal
    vectors dominate, plus hover and pure-heave regimes (grasp / depth legs)."""
    u = rng.random()
    if u < 0.15:
        return np.zeros(3, np.float32)
    if u < 0.30:
        return np.array([0.0, 0.0, rng.choice([-1, 1]) * rng.uniform(0.05, 0.3)], np.float32)
    ang = rng.uniform(-np.pi, np.pi)
    # forward-biased: 75 % of horizontal goals point into the front half-plane
    if rng.random() < 0.75:
        ang = rng.uniform(-np.pi / 2, np.pi / 2)
    mag = rng.uniform(0.05, 0.5)
    g = np.array([np.cos(ang) * mag, np.sin(ang) * mag, rng.normal(0, 0.06)], np.float32)
    return np.clip(g, -0.5, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_scenes.pt")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--usim-episodes", type=int, default=300, help="USIM episodes to draw windows from")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore,/hy-tmp/data/collect_scenes,"
                                    "/hy-tmp/data/deploy_rec,/hy-tmp/data/planner_task")
    ap.add_argument("--n", type=int, default=60000, help="total (window, goal) pairs")
    ap.add_argument("--usim-share", type=float, default=0.4)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--cem-iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/hy-tmp/data/direct_teacher.npz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    model, dyn_norm, pwm_norm, cfg = load_core(args.ckpt, device)
    cfg.control.n_samples = args.n_samples
    cfg.control.cem_iters = args.cem_iters
    mpc = SamplingMPC(model, dyn_norm, pwm_norm, cfg.control, device=device)
    L, K = cfg.model.history_len, cfg.model.horizon_dyn

    print("loading episodes ...", flush=True)
    sources = []  # (name, [episodes])
    if args.usim_episodes > 0:
        eps = load_split(Path(args.usim), "train", cfg.schema, max_episodes=args.usim_episodes)
        sources.append(("usim", eps))
    lib = []
    for d in args.ou.split(","):
        d = d.strip()
        if not d or not Path(d).exists():
            continue
        eps = load_ou_split(Path(d))
        if eps:
            sources.append((Path(d).name, eps))
            if Path(d).name == "ou_explore":
                lib = [e.pwm for e in eps]
    if lib:
        mpc.set_library(np.concatenate(lib, axis=0))
    for name, eps in sources:
        print(f"  {name}: {len(eps)} episodes, {sum(len(e.pwm) for e in eps)} frames", flush=True)

    # allocation: usim_share to USIM, the rest split evenly over the play/deployment sources
    n_usim = int(args.n * args.usim_share) if sources and sources[0][0] == "usim" else 0
    others = [s for s in sources if s[0] != "usim"]
    n_other = (args.n - n_usim) // max(1, len(others))
    plan = [(s, n_usim if s[0] == "usim" else n_other) for s in sources]

    H_S, H_A, S_T, G, SEQ, J, SRC = [], [], [], [], [], [], []
    t0 = time.time()
    done = 0
    for si, ((name, eps), n_take) in enumerate(plan):
        lengths = np.array([len(e.pwm) for e in eps])
        ok = np.where(lengths > L + K + 2)[0]
        w = lengths[ok] - (L + K + 1)
        p = w / w.sum()
        for _ in range(n_take):
            e = eps[ok[rng.choice(len(ok), p=p)]]
            t = int(rng.integers(L, len(e.pwm) - K - 1))
            hist_s = e.dyn[t - L:t].astype(np.float32)
            hist_a = e.pwm[t - L:t].astype(np.float32)
            s_t = e.dyn[t].astype(np.float32)
            if not np.all(np.isfinite(hist_s)) or np.abs(hist_s[:, 0:3]).max() > 3.0:
                continue
            g = sample_goal(rng)
            mpc.reset()
            _, info = mpc.plan(hist_s, hist_a, s_t, g, task_index=0, sid_u=None, rpy=None, yaw_err=0.0)
            H_S.append(hist_s); H_A.append(hist_a); S_T.append(s_t); G.append(g)
            SEQ.append(np.asarray(info["seq"], np.float32)); J.append(float(info["J"][info["best"]])); SRC.append(si)
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done}/{args.n}  {el / done * 1e3:.1f} ms/sample  eta {(args.n - done) * el / done / 60:.1f} min  "
                      f"J mean {np.mean(J[-2000:]):.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, hist_s=np.stack(H_S), hist_a=np.stack(H_A), s_t=np.stack(S_T), v_goal=np.stack(G),
                        seq=np.stack(SEQ), J=np.asarray(J, np.float32), src=np.asarray(SRC, np.int16),
                        sources=np.asarray([s[0] for s in sources]), n_samples=args.n_samples, cem_iters=args.cem_iters,
                        ckpt=str(args.ckpt))
    print(f"saved {len(J)} samples -> {out}  ({time.time() - t0:.0f} s)  J: mean {np.mean(J):.3f} p50 {np.median(J):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
