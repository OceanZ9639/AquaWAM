#!/usr/bin/env python3
"""
Offline checks of the manipulation world model + grasp planner on real grasp recordings
(OU-schema episodes with joint / object channels from scripts/recordings_to_ou.py):

  1. open-loop prediction of the object channels and joints over 0.5 s under the RECORDED actions
     (what the planner will rely on), split by gripper-object distance;
  2. planner sanity: from windows near the object, the imagined gripper-object error after the
     planned 1 s vs. under a hold-still action; per-decision latency.
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
from uwam.arm_kin import load as load_arm_kin  # noqa: E402
from uwam.config import ARM_DYN_SLICES  # noqa: E402
from uwam.data import load_ou_split  # noqa: E402
from uwam.direct import load_core  # noqa: E402
from uwam.grasp_planner import GraspCfg, GraspPlanner  # noqa: E402

JP = slice(*ARM_DYN_SLICES["joint_pos"]); OBJ = slice(*ARM_DYN_SLICES["obj"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/grasp_core.pt")
    ap.add_argument("--arm-kin", default="/hy-tmp/models/uwam/arm_kin.pt")
    ap.add_argument("--data", default="/hy-tmp/data/grasp_ou_box2")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="/hy-tmp/results/grasp_core_offline.json")
    args = ap.parse_args()
    dev = "cuda"
    m, dn, pn, cfg = load_core(args.ckpt, dev)
    assert cfg.model.dyn_state_dim == 35
    fk = load_arm_kin(args.arm_kin, dev)
    eps = [e for e in load_ou_split(Path(args.data), use_arm=True, use_object=True) if e.dyn[:, OBJ.start + 5].mean() > 0.5]
    print(f"{len(eps)} grasp-scene episodes", flush=True)
    rng = np.random.default_rng(0)
    L, K = 16, m.K
    rows = []
    for _ in range(args.n):
        e = eps[rng.integers(len(eps))]
        T = len(e.dyn)
        if T < L + K + 2:
            continue
        t = int(rng.integers(L, T - K - 1))
        hs, ha, st = e.dyn[t - L:t], e.pwm[t - L:t], e.dyn[t]
        af, sf = e.pwm[t:t + K], e.dyn[t + 1:t + K + 1]
        f = lambda a: torch.as_tensor(dn(a) if a.shape[-1] == 35 else pn(a), device=dev, dtype=torch.float32)
        with torch.no_grad():
            d = m.disturbance(f(hs)[None], f(ha)[None])
            s_hat, _, _ = m.rollout(f(st)[None], f(af)[None], d)
        pred = dn.invert(s_hat[0].cpu().numpy())
        q_now = st[JP]; ee = fk.ee_body(q_now)[0]; dist = float(np.linalg.norm(ee - st[OBJ.start:OBJ.start + 3]))
        rows.append({"dist": dist,
                     "obj_err_cm": float(np.abs(pred[:, OBJ.start:OBJ.start + 3] - sf[:, OBJ.start:OBJ.start + 3]).mean() * 100),
                     "obj_err_last_cm": float(np.abs(pred[-1, OBJ.start:OBJ.start + 3] - sf[-1, OBJ.start:OBJ.start + 3]).mean() * 100),
                     "obj_motion_cm": float(np.abs(sf[-1, OBJ.start:OBJ.start + 3] - st[OBJ.start:OBJ.start + 3]).mean() * 100),
                     "joint_err_rad": float(np.abs(pred[:, JP] - sf[:, JP]).mean()),
                     "dvl_err_ms": float(np.abs(pred[:, 0:3] - sf[:, 0:3]).mean())})
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    near = R["dist"] < 0.3
    summary = {"n": len(rows), "n_near": int(near.sum()),
               "obj_pred_err_cm_near": float(R["obj_err_cm"][near].mean()), "obj_pred_err_cm_far": float(R["obj_err_cm"][~near].mean()),
               "obj_err_last_step_cm_near": float(R["obj_err_last_cm"][near].mean()),
               "obj_actual_motion_cm_near": float(R["obj_motion_cm"][near].mean()),
               "joint_pred_err_rad": float(R["joint_err_rad"].mean()), "dvl_pred_err_ms": float(R["dvl_err_ms"].mean())}
    print(json.dumps(summary, indent=1))

    # planner sanity on near windows
    pl = GraspPlanner(m, dn, pn, fk, GraspCfg(), device=dev)
    imp, lat = [], []
    tries = 0
    while len(imp) < 40 and tries < 2000:
        tries += 1
        e = eps[rng.integers(len(eps))]; T = len(e.dyn)
        if T < L + K + 2:
            continue
        t = int(rng.integers(L, T - K - 1)); st = e.dyn[t]
        obj = st[OBJ.start:OBJ.start + 3].copy()
        ee = fk.ee_body(st[JP])[0]
        if np.linalg.norm(ee - obj) > 0.25 or np.linalg.norm(ee - obj) < 0.02:
            continue
        pl.reset()
        t0 = time.perf_counter()
        pwm, q, info = pl.plan(e.dyn[t - L:t], e.pwm[t - L:t], st, obj, yaw=0.0, rpy=np.zeros(3), omega=st[3:6])
        lat.append(time.perf_counter() - t0)
        # imagined error under the plan vs. under hold (zero thrust, joints held)
        hold = np.zeros((K, 13), np.float32); hold[:, 8:13] = st[JP]
        plan_a = np.concatenate([pwm, np.tile(q[None], (K, 1))], -1).astype(np.float32)
        f = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float32)
        with torch.no_grad():
            hsn, han, stn = f(dn(e.dyn[t - L:t]))[None], f(pn(e.pwm[t - L:t]))[None], f(dn(st))[None]
            d = m.disturbance(hsn, han)
            outs = {}
            for name, a in (("plan", plan_a), ("hold", hold)):
                cur, segs = stn, []
                for _ in range(2):
                    sh, _, _ = m.rollout(cur, f(pn(a))[None], d); segs.append(sh); cur = sh[:, -1]
                s = dn.invert(torch.cat(segs, 1)[0].cpu().numpy())
                ee_k = fk.ee_body(s[-1, JP])[0]; o_k = s[-1, OBJ.start:OBJ.start + 3]
                outs[name] = ee_k - o_k - np.array([0, 0, -info["z_above"]])
        imp.append((np.linalg.norm(outs["hold"][:2]), np.linalg.norm(outs["plan"][:2]), abs(outs["hold"][2]), abs(outs["plan"][2])))
    imp = np.array(imp)
    summary.update({"planner_windows": int(len(imp)),
                    "imagined_xy_err_cm_hold": float(imp[:, 0].mean() * 100), "imagined_xy_err_cm_plan": float(imp[:, 1].mean() * 100),
                    "imagined_z_err_cm_hold": float(imp[:, 2].mean() * 100), "imagined_z_err_cm_plan": float(imp[:, 3].mean() * 100),
                    "plan_latency_ms": float(np.mean(lat) * 1e3)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("planner", "imagined", "plan_"))}, indent=1))


if __name__ == "__main__":
    main()
