#!/usr/bin/env python3
"""
USIM-Hard P5: can the world model predict the VLA's failures?

Every U0 manipulation episode under DVL dropout (u0_drop30s_zero, 11 tasks x 40) has a 10 Hz
sensor log written by the extended bridge (episode_i_sensors.npz: true + effective DVL, IMU,
pressure, altitude, commanded PWM, odometry, AHRS).  U0 never saw the WAM; the WAM never
touched these episodes.  We replay the log through the frozen core exactly as the policy
server would build its windows and ask, per episode, how "surprised" the world model is by
what the VLA does to the vehicle:

  imu_res   mean |s_hat - s| on the always-live IMU channels (gyro + accel) over the 0.5 s
            rollout of the commanded PWM  -- prediction error of the core
  epi_sigma mean ensemble disagreement (epistemic sigma) of the dead-reckoning estimator
            during the blind phase        -- the trust gate's own statistic
  dr_err    |v_hat - v_true| during the blind phase (privileged truth; diagnostic only)
  baselines: mean speed, mean |PWM|, mean |gyro|.

Features are computed on the first `--horizon` seconds after the first action (default 60 s;
U0 successes typically finish at 50-70 s, timeouts at 185 s), i.e. BEFORE the outcome is
known, and reported as AUROC for predicting `timeout`, pooled and after per-task z-scoring
(removes the between-task difficulty confound).  Writes a JSON next to the results doc.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import SamplingMPC  # noqa: E402
from uwam.direct import load_core  # noqa: E402

L, K = 16, 5


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` for label==1 (ties count half)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def load_log(path):
    z = np.load(path, allow_pickle=True)
    rows, cols, layout = z["rows"], list(z["cols"]), list(z["layout"])
    off, out = 0, {}
    for c, n in zip(cols, layout):
        out[str(c)] = rows[:, off:off + int(n)]
        off += int(n)
    return out


@torch.no_grad()
def episode_features(log, model, dyn_norm, pwm_norm, mpc, device, t_max: float):
    t = log["t"][:, 0]
    keep = t <= t_max
    n = int(keep.sum())
    if n < L + K + 5:
        return None
    dvl_eff, dvl_true = log["dvl_eff"][:n], log["dvl_true"][:n]
    imu_av, imu_la = log["imu_av"][:n], log["imu_la"][:n]
    pressure = log["pressure"][:n] * 1e4
    alt = log["alt_eff"][:n]
    pwm = log["pwm_cmd"][:n]
    valid = log["dvl_valid"][:n, 0] > 0.5
    pwm_prev = np.vstack([pwm[:1] * 0, pwm[:-1]])
    dyn = np.concatenate([dvl_eff, imu_av, imu_la, pressure, alt, pwm_prev], axis=1).astype(np.float32)
    ts = np.arange(L, n - K)
    hist_s = np.stack([dyn[i - L:i] for i in ts])
    hist_a = np.stack([pwm[i - L:i] for i in ts]).astype(np.float32)
    s_t = dyn[ts]
    a_fut = np.stack([pwm[i:i + K] for i in ts]).astype(np.float32)
    s_fut = np.stack([dyn[i + 1:i + K + 1] for i in ts])
    v_true_fut = np.stack([dvl_true[i + 1:i + K + 1] for i in ts]).astype(np.float32)

    f = lambda a: torch.as_tensor(a, device=device)
    hs, ha = f(dyn_norm(hist_s)), f(pwm_norm(hist_a))
    st, af = f(dyn_norm(s_t)), f(pwm_norm(a_fut))
    d = model.disturbance(hs, ha)
    s_hat, _, _ = model.rollout(st, af, d)
    res = (s_hat - f(dyn_norm(s_fut))).abs()  # normalized units
    imu_res = res[:, :, 3:9].mean(dim=(1, 2)).cpu().numpy()
    # DVL channels of the prediction vs privileged truth (the effective DVL is zero when dropped)
    v_hat = s_hat[:, :, 0:3].cpu().numpy() * dyn_norm.std[0:3] + dyn_norm.mean[0:3]
    dvl_res = np.linalg.norm(v_hat - v_true_fut, axis=-1).mean(axis=1)

    # dead-reckoning ensemble on the DVL-masked, canonicalized window (as the server does)
    heads = mpc._vel_ens
    epi = dr_err = None
    if heads:
        hs_c = f(dyn_norm(np.stack([mpc._canon(w) for w in hist_s])))
        hm = model.mask_dvl(hs_c)
        dm = model.disturbance(hm, ha)
        x = torch.cat([torch.cat([hm, ha], dim=-1).flatten(1), dm], dim=1)
        outs = torch.stack([h(x) for h in heads], 0)
        mus = outs[:, :, :3]
        sd_epi = mus.var(0, unbiased=False).clamp_min(1e-12).sqrt().mean(dim=1).cpu().numpy()
        mu = mus.mean(0).cpu().numpy() * dyn_norm.std[0:3] + dyn_norm.mean[0:3]
        epi = sd_epi
        dr_err = np.linalg.norm(mu - dvl_true[ts], axis=1)

    blind = ~valid[ts]
    blind_any = blind.any()
    feats = {
        "imu_res": float(imu_res.mean()),
        "imu_res_blind": float(imu_res[blind].mean()) if blind_any else float("nan"),
        "dvl_res_sighted": float(dvl_res[~blind].mean()) if (~blind).any() else float("nan"),
        "epi_sigma_blind": float(epi[blind].mean()) if (epi is not None and blind_any) else float("nan"),
        "dr_err_blind": float(dr_err[blind].mean()) if (dr_err is not None and blind_any) else float("nan"),
        "speed": float(np.linalg.norm(dvl_true[ts], axis=1).mean()),
        "pwm_abs": float(np.abs(pwm[ts]).mean()),
        "gyro_abs": float(np.linalg.norm(imu_av[ts], axis=1).mean()),
        "n_windows": int(len(ts)),
        "blind_frac": float(blind.mean()),
    }
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--arm", default="u0_drop30s_zero")
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best_scenes.pt")
    ap.add_argument("--vel-ens", default="/hy-tmp/models/uwam/vel_ens_scenes.pt")
    ap.add_argument("--horizon", type=float, default=60.0, help="seconds of each episode used for the features")
    ap.add_argument("--out", default="/hy-tmp/underwater_wam/u0eval/p5_failure_auroc.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dyn_norm, pwm_norm, cfg = load_core(args.ckpt, device)
    mpc = SamplingMPC(model, dyn_norm, pwm_norm, cfg.control, device=device)
    ens_ok = mpc.load_vel_ensemble(args.vel_ens)
    print(f"core {args.ckpt}; ensemble {'loaded' if ens_ok else 'MISSING'}; horizon {args.horizon}s", flush=True)

    rows = []
    for task_dir in sorted(glob.glob(f"{args.runs}/{args.arm}/*")):
        task = Path(task_dir).name
        res_csv = Path(task_dir) / "results.csv"
        if not res_csv.exists():
            continue
        outcome = {}
        for line in res_csv.read_text().strip().splitlines()[1:]:
            p = line.split(",")
            if len(p) >= 2 and p[0].isdigit():
                outcome[int(p[0])] = p[1].strip()
        for f_ in sorted(glob.glob(f"{task_dir}/logs/episode_*_sensors.npz")):
            ep = int(Path(f_).stem.split("_")[1])
            if ep not in outcome:
                continue
            try:
                feats = episode_features(load_log(f_), model, dyn_norm, pwm_norm, mpc, device, args.horizon)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {task}/{ep}: {e}", flush=True)
                continue
            if feats is None:
                continue
            feats.update({"task": task, "episode": ep, "fail": int(outcome[ep] != "success")})
            rows.append(feats)
        n_t = sum(1 for r in rows if r["task"] == task)
        n_f = sum(r["fail"] for r in rows if r["task"] == task)
        print(f"  {task:<24} {n_t:3d} episodes, {n_f} failures", flush=True)

    names = ["imu_res", "imu_res_blind", "dvl_res_sighted", "epi_sigma_blind", "dr_err_blind", "speed", "pwm_abs", "gyro_abs"]
    fail = np.array([r["fail"] for r in rows])
    tasks = np.array([r["task"] for r in rows])
    table = {}
    for nm in names:
        x = np.array([r[nm] for r in rows], np.float64)
        ok = np.isfinite(x)
        a_pool = auroc(x[ok], fail[ok])
        z = np.full_like(x, np.nan)
        for t in np.unique(tasks):
            m = (tasks == t) & ok
            if m.sum() >= 3 and x[m].std() > 0:
                z[m] = (x[m] - x[m].mean()) / x[m].std()
        okz = np.isfinite(z)
        a_z = auroc(z[okz], fail[okz])
        # per-task AUROCs (only tasks with both outcomes), median
        per = []
        for t in np.unique(tasks):
            m = (tasks == t) & ok
            if m.sum() >= 6 and 0 < fail[m].sum() < m.sum():
                per.append(auroc(x[m], fail[m]))
        table[nm] = {"auroc_pooled": a_pool, "auroc_task_z": a_z,
                     "auroc_per_task_median": float(np.median(per)) if per else float("nan"), "n_tasks": len(per)}
    out = {"arm": args.arm, "horizon_s": args.horizon, "n_episodes": int(len(rows)), "n_fail": int(fail.sum()),
           "core": args.ckpt, "vel_ens": args.vel_ens, "features": table,
           "episodes": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n{len(rows)} episodes, {fail.sum()} failures ({100 * fail.mean():.0f} %); features on the first {args.horizon:.0f} s")
    print(f"{'feature':<18}{'AUROC pooled':>14}{'AUROC task-z':>14}{'per-task median':>17}")
    for nm, v in table.items():
        print(f"{nm:<18}{v['auroc_pooled']:>14.3f}{v['auroc_task_z']:>14.3f}{v['auroc_per_task_median']:>17.3f}  ({v['n_tasks']} tasks)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
