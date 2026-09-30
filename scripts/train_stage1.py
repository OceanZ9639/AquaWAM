#!/usr/bin/env python3
"""Stage-1 body-dynamics WAM training + recovery evaluations."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uwam.config import Cfg, enable_arm, enable_object, ensure_dirs
from uwam.data import DynamicsWindowDataset, collate, load_ou_split, load_split
from uwam.losses import dynamics_batch_loss, pairwise_dvl_mae
from uwam.models import DynamicsWAM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=shuffle,
    )


def _dvl_mae_ms(pred, gt, dyn_norm) -> torch.Tensor:
    p = dyn_norm.invert_torch(pred)[..., 0:3]
    g = dyn_norm.invert_torch(gt)[..., 0:3]
    return (p - g).abs().mean(dim=(1, 2))


@torch.no_grad()
def evaluate(model: DynamicsWAM, loader: DataLoader, device: str, use_history: bool = True, dyn_norm=None) -> dict:
    model.eval()
    keys = ("dvl_mae", "dvl_mae_ms", "state_mae", "rank_rand", "rank_zero", "rank_rev", "rank_near", "n")
    totals = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, use_history=use_history)
        dvl_z = pairwise_dvl_mae(out["s_hat"], batch["s_fut"])
        dvl_ms = _dvl_mae_ms(out["s_hat"], batch["s_fut"], dyn_norm) if dyn_norm is not None else dvl_z
        state = (out["s_hat"] - batch["s_fut"]).abs().mean(dim=(1, 2))
        bsz = int(dvl_z.numel())
        totals["dvl_mae"] += dvl_z.sum().item()
        totals["dvl_mae_ms"] += dvl_ms.sum().item()
        totals["state_mae"] += state.sum().item()
        err_t = dvl_z
        for name in ("rand", "zero", "rev", "near"):
            key = f"s_hat_{name}"
            if key in out:
                err_cf = pairwise_dvl_mae(out[key], batch["s_fut"])
                totals[f"rank_{name}"] += (err_t < err_cf).float().sum().item()
        n += bsz
    model.train()
    if n == 0:
        return {k: 0.0 for k in totals}
    return {
        "dvl_mae": totals["dvl_mae"] / n,
        "dvl_mae_ms": totals["dvl_mae_ms"] / n,
        "state_mae": totals["state_mae"] / n,
        "rank_rand": totals["rank_rand"] / n,
        "rank_zero": totals["rank_zero"] / n,
        "rank_rev": totals["rank_rev"] / n,
        "rank_near": totals["rank_near"] / n,
        "n": n,
    }


@torch.no_grad()
def eval_vel_estimator(model: DynamicsWAM, loader: DataLoader, device: str, dyn_norm) -> dict:
    """Dead-reckoning error in m/s: how well the DVL can be replaced by IMU + PWM history."""
    model.eval()
    tot, tot_hover, n = 0.0, 0.0, 0
    tot_fault, n_fault = 0.0, 0
    scale = torch.as_tensor(dyn_norm.std[0:3], dtype=torch.float32, device=device)
    for batch in loader:
        hs = batch["hist_s"].to(device)
        ha = batch["hist_a"].to(device)
        gt = batch["s_t"][:, 0:3].to(device)
        v_hat = model.estimate_velocity(model.mask_dvl(hs), ha)
        err = ((v_hat - gt) * scale).abs().mean(dim=1)
        tot += err.sum().item()
        # a zero-velocity guess is the trivial fallback a blinded controller has
        tot_hover += (gt * scale).abs().mean(dim=1).sum().item()
        n += gt.size(0)
        if "eta_target" in batch:
            faulty = (batch["eta_target"].min(dim=1).values < 0.99).to(device)
            if faulty.any():
                tot_fault += err[faulty].sum().item()
                n_fault += int(faulty.sum().item())
    model.train()
    if not n:
        return {"vel_mae_ms": None, "hover_mae_ms": None, "n": 0}
    return {
        "vel_mae_ms": tot / n,
        "vel_mae_ms_fault": (tot_fault / n_fault) if n_fault else None,
        "hover_mae_ms": tot_hover / n,
        "n": n,
        "n_fault": n_fault,
    }


@torch.no_grad()
def eval_eta(model: DynamicsWAM, loader: DataLoader, device: str) -> dict:
    """Per-thruster efficiency decoding error, overall and on frames with an active fault."""
    model.eval()
    tot, n = 0.0, 0
    tot_fault, n_fault = 0.0, 0
    worst = 0.0
    for batch in loader:
        if "eta_target" not in batch:
            continue
        d = model.disturbance(batch["hist_s"].to(device), batch["hist_a"].to(device))
        eta_hat = model.eta_head(d)
        gt = batch["eta_target"].to(device)
        err = (eta_hat - gt).abs().mean(dim=1)
        tot += err.sum().item()
        n += gt.size(0)
        faulty = gt.min(dim=1).values < 0.99
        if faulty.any():
            # error on the degraded channels only, the quantity the claim is about
            bad = (gt[faulty] < 0.99)
            ch_err = ((eta_hat[faulty] - gt[faulty]).abs() * bad).sum() / bad.sum()
            tot_fault += float(ch_err) * int(faulty.sum())
            n_fault += int(faulty.sum().item())
            worst = max(worst, float((eta_hat[faulty] - gt[faulty]).abs().max()))
    model.train()
    if not n:
        return {"mae": None, "n": 0}
    return {
        "mae": tot / n,
        "mae_fault_channels": (tot_fault / n_fault) if n_fault else None,
        "worst_abs_err": worst,
        "n": n,
        "n_fault": n_fault,
    }


@torch.no_grad()
def linear_probe(model: DynamicsWAM, loader: DataLoader, device: str, n_classes: int, label_key: str = "current_bin") -> dict:
    model.eval()
    zs, ys = [], []
    for batch in loader:
        batch_d = {k: v.to(device) for k, v in batch.items()}
        d = model.disturbance(batch_d["hist_s"], batch_d["hist_a"])
        zs.append(d.cpu())
        ys.append(batch[label_key])
    if not zs:
        return {"acc": 0.0, "n": 0, "n_classes": n_classes, "label_key": label_key}
    z = torch.cat(zs)
    y = torch.cat(ys).clamp(min=0, max=n_classes - 1)
    n = z.size(0)
    perm = torch.randperm(n)
    n_tr = max(1, int(0.8 * n))
    tr, te = perm[:n_tr], perm[n_tr:] if n_tr < n else perm[:1]
    ztr = torch.cat([z[tr], torch.ones(len(tr), 1)], 1)
    zte = torch.cat([z[te], torch.ones(len(te), 1)], 1)
    ytr = torch.nn.functional.one_hot(y[tr], n_classes).float()
    A = ztr.T @ ztr + 1e-2 * torch.eye(ztr.size(1))
    W = torch.linalg.solve(A, ztr.T @ ytr)
    pred = zte @ W
    acc = (pred.argmax(-1) == y[te]).float().mean().item()
    return {
        "acc": acc,
        "random": 1.0 / n_classes,
        "n": int(n),
        "n_te": int(len(te)),
        "n_classes": n_classes,
        "label_key": label_key,
    }


@torch.no_grad()
def linear_probe_time(model: DynamicsWAM, loader: DataLoader, device: str, n_classes: int, label_key: str, t_cut: float = 0.8) -> dict:
    """Fit on early windows (t_frac < t_cut), test on later windows — no adjacent-window leak."""
    model.eval()
    ztr, ytr, zte, yte = [], [], [], []
    for batch in loader:
        hs = batch["hist_s"].to(device)
        ha = batch["hist_a"].to(device)
        d = model.disturbance(hs, ha).cpu()
        y = batch[label_key].long()
        frac = batch["t_frac"]
        ou = batch["is_ou"] > 0.5 if "is_ou" in batch else torch.ones(y.shape[0], dtype=torch.bool)
        tr = (frac < t_cut) & ou
        te = (frac >= t_cut) & ou
        if tr.any():
            ztr.append(d[tr])
            ytr.append(y[tr])
        if te.any():
            zte.append(d[te])
            yte.append(y[te])
    if not ztr or not zte:
        return {"acc": 0.0, "n": 0, "n_classes": n_classes, "label_key": label_key, "split": "time"}
    ztr = torch.cat(ztr)
    ytr = torch.cat(ytr).clamp(0, n_classes - 1)
    zte = torch.cat(zte)
    yte = torch.cat(yte).clamp(0, n_classes - 1)
    ztrb = torch.cat([ztr, torch.ones(ztr.size(0), 1)], 1)
    zteb = torch.cat([zte, torch.ones(zte.size(0), 1)], 1)
    yoh = torch.nn.functional.one_hot(ytr, n_classes).float()
    A = ztrb.T @ ztrb + 1e-2 * torch.eye(ztrb.size(1))
    W = torch.linalg.solve(A, ztrb.T @ yoh)
    pred = zteb @ W
    acc = (pred.argmax(-1) == yte).float().mean().item()
    return {
        "acc": acc,
        "random": 1.0 / n_classes,
        "n": int(ztr.size(0) + zte.size(0)),
        "n_te": int(zte.size(0)),
        "n_classes": n_classes,
        "label_key": label_key,
        "split": "time",
    }


@torch.no_grad()
def change_point_d_shift(model: DynamicsWAM, loader: DataLoader, device: str, t_change: float = 8.0) -> dict:
    model.eval()
    bef, aft = [], []
    for batch in loader:
        if "regime_index" not in batch:
            continue
        d = model.disturbance(batch["hist_s"].to(device), batch["hist_a"].to(device)).cpu()
        for i in range(d.size(0)):
            if int(batch["regime_index"][i]) != 5:
                continue
            tsec = float(batch["t_sec"][i]) if "t_sec" in batch else float(batch["t_frac"][i]) * 270.0
            (bef if tsec < t_change else aft).append(d[i])
    if not bef or not aft:
        return {"l2": 0.0, "n_before": len(bef), "n_after": len(aft)}
    mb = torch.stack(bef).mean(0)
    ma = torch.stack(aft).mean(0)
    return {"l2": float((mb - ma).norm().item()), "n_before": len(bef), "n_after": len(aft)}


def _summarize(ev, ev_no, ev_slow, probe) -> dict:
    ctx_z = ev_no["dvl_mae"] - ev["dvl_mae"]
    ctx_ms = ev_no["dvl_mae_ms"] - ev["dvl_mae_ms"]
    rel = ctx_ms / max(1e-8, ev_no["dvl_mae_ms"])
    retained = 0.0
    if ev_no["dvl_mae_ms"] > ev["dvl_mae_ms"] + 1e-8:
        retained = (ev_no["dvl_mae_ms"] - ev_slow["dvl_mae_ms"]) / (ev_no["dvl_mae_ms"] - ev["dvl_mae_ms"])
    return {
        "ctx_gain_z": ctx_z,
        "ctx_gain_ms": ctx_ms,
        "ctx_gain_rel": rel,
        "slow_retained": retained,
        "probe": probe,
    }


def subset_usim(eps: list, frac: float = 1.0, seed: int = 0,
                include: list[int] | None = None, exclude: list[int] | None = None) -> list:
    """USIM-Hard data-efficiency / held-out-task splits.

    Task filters first (by task_index), then a seeded *stratified* subsample keeping round(frac * n)
    episodes of every remaining task (>= 1), so a 10 % core still sees every task's dynamics regime.
    frac=1.0 with no filters is the identity, i.e. existing recipes are untouched.
    """
    if include:
        eps = [e for e in eps if e.task_index in include]
    if exclude:
        eps = [e for e in eps if e.task_index not in exclude]
    if frac >= 1.0:
        return eps
    rng = random.Random(seed)
    by_task: dict[int, list] = {}
    for e in eps:
        by_task.setdefault(e.task_index, []).append(e)
    out = []
    for ti in sorted(by_task):
        group = sorted(by_task[ti], key=lambda e: str(e.parquet_path))
        k = max(1, int(round(frac * len(group))))
        out += rng.sample(group, k)
    return out


def train(
    cfg: Cfg,
    max_episodes: int = 0,
    epochs: int | None = None,
    mix_ou: bool = False,
    resume: Path | None = None,
    extra_ou: list[str] | None = None,
    extra_reps: int = 48,
    no_usim: bool = False,
    ckpt_name: str = "best",
    use_arm: bool = False,
    use_object: bool = False,
    usim_frac: float = 1.0,
    usim_seed: int = 0,
    usim_include: list[int] | None = None,
    usim_exclude: list[int] | None = None,
) -> Path:
    ensure_dirs(cfg)
    set_seed(cfg.train.seed)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    cfg.model.use_language = False
    if use_object:
        use_arm = True
        enable_object(cfg)
        print(f"object extension ON: dyn={cfg.model.dyn_state_dim} (arm + object relative pose)", flush=True)
    elif use_arm:
        enable_arm(cfg)
        print(f"manipulator extension ON: dyn={cfg.model.dyn_state_dim} pwm={cfg.model.pwm_dim}",
              flush=True)

    if no_usim:
        print("USIM ablation: training on OU + planner data only", flush=True)
        train_eps, test_eps = [], []
    else:
        print("Loading USIM parquet ...", flush=True)
        train_eps = load_split(cfg.paths.usim, "train", cfg.schema, max_episodes=max_episodes,
                               use_arm=use_arm, use_object=use_object)
        test_eps = load_split(cfg.paths.usim, "test", cfg.schema, max_episodes=max_episodes,
                              use_arm=use_arm, use_object=use_object)
        if usim_frac < 1.0 or usim_include or usim_exclude:
            n0 = len(train_eps)
            train_eps = subset_usim(train_eps, usim_frac, usim_seed, usim_include, usim_exclude)
            hours = sum(len(e.dyn) for e in train_eps) / 10.0 / 3600.0
            print(f"USIM subset: frac={usim_frac} seed={usim_seed} include={usim_include} "
                  f"exclude={usim_exclude} -> {len(train_eps)}/{n0} episodes ({hours:.2f} h @10 Hz); "
                  f"test split kept whole ({len(test_eps)} episodes)", flush=True)
    ou_eps = []
    if mix_ou:
        ou_eps = load_ou_split(cfg.paths.ou_data, use_arm=use_arm, n_joints=cfg.model.n_joints, use_object=use_object)
        extra_eps = []
        for extra in extra_ou or []:
            more = load_ou_split(Path(extra), use_arm=use_arm, n_joints=cfg.model.n_joints, use_object=use_object)
            print(f"mixing {len(more)} planner-distribution episodes from {extra}", flush=True)
            extra_eps += more
        # upsample so expert-PWM OOD is not drowned by ~1.7k USIM episodes; the planner-distribution
        # set is 3x shorter per episode, so it needs its own factor to reach a comparable share
        reps = 12
        print(f"mixing {len(ou_eps)} OU episodes x{reps} + {len(extra_eps)} planner x{extra_reps}",
              flush=True)
        train_eps = train_eps + ou_eps * reps + extra_eps * extra_reps
        ou_eps = ou_eps + extra_eps
    if not train_eps:
        raise FileNotFoundError(f"No training episodes (usim={cfg.paths.usim}, no_usim={no_usim})")

    resume_ckpt = None
    if resume is not None and Path(resume).exists():
        resume_ckpt = torch.load(resume, map_location=device, weights_only=False)
        print(f"resume {resume}", flush=True)

    ck_dyn_dim = len(resume_ckpt["dyn_norm"]["mean"]) if resume_ckpt is not None and "dyn_norm" in resume_ckpt else None
    same_layout = ck_dyn_dim == cfg.model.dyn_state_dim
    train_ds = DynamicsWindowDataset(train_eps, cfg, fit_norm=(resume_ckpt is None) or not same_layout)
    if resume_ckpt is not None and "dyn_norm" in resume_ckpt:
        if same_layout:
            train_ds.dyn_norm.load_state_dict(resume_ckpt["dyn_norm"])
            train_ds.pwm_norm.load_state_dict(resume_ckpt["pwm_norm"])
        else:
            # state layout grew (e.g. 29-d arm core -> 35-d arm+object): keep the checkpoint's
            # normalization on the channels it was trained with, fit only the new ones from data
            import numpy as _np
            for norm, key in ((train_ds.dyn_norm, "dyn_norm"), (train_ds.pwm_norm, "pwm_norm")):
                m0 = _np.asarray(resume_ckpt[key]["mean"], _np.float32); s0 = _np.asarray(resume_ckpt[key]["std"], _np.float32)
                k = min(len(m0), len(norm.mean))
                norm.mean[:k] = m0[:k]; norm.std[:k] = s0[:k]
            print(f"resume: norm layout {ck_dyn_dim} -> {cfg.model.dyn_state_dim}; kept the first {ck_dyn_dim} channels' "
                  "statistics, fitted the new ones", flush=True)
    # Normalization floors for channels that are (nearly) constant in parts of the corpus: the
    # expert never moves the arm, so USIM alone gives joint stds of 1e-5 and any real joint motion
    # at deployment is normalized to 1e4 -- the planner then extrapolates into nonsense. Floors are
    # physical units: 0.05 rad / rad/s for joints, 5 cm for the object position, 0.2 for its
    # yaw cos/sin and validity flag, 0.05 for joint targets.
    if use_arm:
        floors = {}
        nj = cfg.model.n_joints
        for i in range(19, 19 + nj):
            floors[i] = 0.05
        for i in range(19 + nj, 19 + 2 * nj):
            floors[i] = 0.05
        if use_object:
            for i in range(19 + 2 * nj, 19 + 2 * nj + 3):
                floors[i] = 0.05
            for i in range(19 + 2 * nj + 3, 19 + 2 * nj + 6):
                floors[i] = 0.2
        sd = train_ds.dyn_norm.std
        for i, f in floors.items():
            if i < len(sd):
                sd[i] = max(float(sd[i]), f)
        psd = train_ds.pwm_norm.std
        for i in range(8, 8 + nj):
            if i < len(psd):
                psd[i] = max(float(psd[i]), 0.05)
        print(f"normalization floors applied: dyn std[19:] = {np.round(train_ds.dyn_norm.std[19:], 4).tolist()}, "
              f"pwm std[8:] = {np.round(train_ds.pwm_norm.std[8:], 4).tolist()}", flush=True)
    test_ds = DynamicsWindowDataset(
        test_eps or train_eps[-max(1, len(train_eps) // 10) :],
        cfg,
        dyn_norm=train_ds.dyn_norm,
        pwm_norm=train_ds.pwm_norm,
        fit_norm=False,
    )
    slow_ds = DynamicsWindowDataset(
        test_eps or train_eps[-max(1, len(train_eps) // 10) :],
        cfg,
        dyn_norm=train_ds.dyn_norm,
        pwm_norm=train_ds.pwm_norm,
        fit_norm=False,
        slow_mode=True,
    )
    ou_loader = None
    if ou_eps:
        ou_ds = DynamicsWindowDataset(
            ou_eps, cfg, dyn_norm=train_ds.dyn_norm, pwm_norm=train_ds.pwm_norm, fit_norm=False
        )
        ou_loader = make_loader(ou_ds, cfg, False)
    print(f"train windows={len(train_ds)} test windows={len(test_ds)} episodes={len(train_eps)}", flush=True)

    train_loader = make_loader(train_ds, cfg, True)
    test_loader = make_loader(test_ds, cfg, False)
    slow_loader = make_loader(slow_ds, cfg, False)

    n_tasks = 1 + max(ep.task_index for ep in train_eps)
    cfg.model.n_tasks = max(cfg.model.n_tasks, n_tasks)
    model = DynamicsWAM(cfg).to(device)
    if resume_ckpt is not None:
        # heads change shape across experiments; keep the matching weights instead of refusing
        own = model.state_dict()
        src = resume_ckpt["model"]
        reshaped = [k for k, v in src.items() if k in own and own[k].shape != v.shape]
        state = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"resume missing={list(missing)} unexpected={list(unexpected)} reshaped={reshaped}",
              flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    amp = cfg.train.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = epochs if epochs is not None else cfg.train.epochs
    best = 1e9
    ckpt_dir = cfg.paths.ckpt
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    hist_stem = "stage1_ou_history" if mix_ou else "stage1_history"
    if ckpt_name != "best":
        hist_stem = f"stage1_{ckpt_name}_history"
    hist_path = cfg.paths.logs / f"{hist_stem}.json"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(batch, use_history=True, counterfactuals=True)
                with torch.no_grad():
                    out_no = model(batch, use_history=False, counterfactuals=False)
                out["s_hat_nohist"] = out_no["s_hat"]
                loss, logs = dynamics_batch_loss(out, batch, cfg)
                # dead-reckoning head: same window with the DVL wiped, regress the true velocity.
                # Fault windows are upweighted: the P0 traces showed the estimator under-reads
                # surge exactly there, which made the MPC overshoot.
                if hasattr(model, "vel_head"):
                    v_hat = model.estimate_velocity(model.mask_dvl(batch["hist_s"]), batch["hist_a"])
                    per = torch.nn.functional.smooth_l1_loss(
                        v_hat, batch["s_t"][:, 0:3], reduction="none"
                    ).mean(dim=1)
                    w = torch.ones_like(per)
                    if "eta_target" in batch:
                        faulty = batch["eta_target"].min(dim=1).values < 0.99
                        w = torch.where(faulty, torch.full_like(per, cfg.train.vel_fault_weight), w)
                    l_vel = (w * per).sum() / w.sum()
                    loss = loss + cfg.train.lambda_vel * l_vel
                    logs["vel_est"] = l_vel.detach()
                    # the deployed blind loop fills the history's DVL columns with the estimator's
                    # own outputs; keep d stable under that input distribution
                    if cfg.train.lambda_dcons > 0 and "is_ou" in batch:
                        ou_m2 = batch["is_ou"] > 0.5
                        if ou_m2.any():
                            h_est = batch["hist_s"][ou_m2].clone()
                            h_est[..., 0:3] = v_hat[ou_m2].detach().unsqueeze(1)
                            d_est = model.disturbance(h_est, batch["hist_a"][ou_m2])
                            l_dc = torch.nn.functional.mse_loss(d_est, out["d"][ou_m2].detach())
                            loss = loss + cfg.train.lambda_dcons * l_dc
                            logs["d_cons"] = l_dc.detach()
                if "eta_hat" in out and "eta_target" in batch:
                    ou_e = batch["is_ou"] > 0.5 if "is_ou" in batch else torch.ones(
                        batch["eta_target"].size(0), dtype=torch.bool, device=batch["eta_target"].device)
                    if ou_e.any():
                        l_eta = torch.nn.functional.smooth_l1_loss(
                            out["eta_hat"][ou_e], batch["eta_target"][ou_e])
                        loss = loss + cfg.train.lambda_eta * l_eta
                        logs["eta"] = l_eta.detach()
                if "flow_logits" in out and "flow_class" in batch:
                    ou_m = batch["is_ou"] > 0.5 if "is_ou" in batch else torch.ones_like(batch["flow_class"], dtype=torch.bool)
                    if ou_m.any():
                        l_f = torch.nn.functional.cross_entropy(out["flow_logits"][ou_m], batch["flow_class"][ou_m])
                        l_a = torch.nn.functional.cross_entropy(out["fail_logits"][ou_m], batch["fail_class"][ou_m])
                        loss = loss + cfg.train.lambda_aux * (l_f + l_a)
                        logs["aux_flow"] = l_f.detach()
                        logs["aux_fail"] = l_a.detach()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            bsz = batch["s_t"].size(0)
            running += loss.item() * bsz
            n_seen += bsz
            pbar.set_postfix(loss=f"{loss.item():.4f}", dvl=f"{float(logs['dvl_l1']):.4f}")

        ev = evaluate(model, test_loader, device, True, train_ds.dyn_norm)
        ev_no = evaluate(model, test_loader, device, False, train_ds.dyn_norm)
        ev_slow = evaluate(model, slow_loader, device, True, train_ds.dyn_norm)
        probe = linear_probe(model, test_loader, device, n_classes=3, label_key="current_bin")
        extra = _summarize(ev, ev_no, ev_slow, probe)
        rec = {
            "epoch": epoch,
            "train_loss": running / max(1, n_seen),
            "sec": time.time() - t0,
            "eval": ev,
            "eval_no_hist": ev_no,
            "eval_slow": ev_slow,
            **extra,
        }
        if ou_loader is not None:
            rec["probe_regime"] = linear_probe(model, ou_loader, device, n_classes=6, label_key="regime_index")
            rec["probe_flow"] = linear_probe_time(model, ou_loader, device, n_classes=3, label_key="flow_class")
            rec["probe_fail"] = linear_probe_time(model, ou_loader, device, n_classes=3, label_key="fail_class")
            rec["eval_ou"] = evaluate(model, ou_loader, device, True, train_ds.dyn_norm)
            rec["change_point_d"] = change_point_d_shift(model, ou_loader, device)
            rec["vel_estimator_ou"] = eval_vel_estimator(model, ou_loader, device, train_ds.dyn_norm)
            rec["eta_probe_ou"] = eval_eta(model, ou_loader, device)
        history.append(rec)
        hist_path.write_text(json.dumps(history, indent=2))
        print(json.dumps(rec, indent=2), flush=True)
        ckpt = {
            "model": model.state_dict(),
            "cfg": {"use_language": False, "disturbance_dim": cfg.model.disturbance_dim,
                    "hidden": cfg.model.hidden, "use_dt": cfg.model.use_dt,
                    "no_token": bool(getattr(cfg.model, "no_token", False)),
                    "use_arm": cfg.model.use_arm, "use_object": bool(getattr(cfg.model, "use_object", False)),
                    "dyn_state_dim": cfg.model.dyn_state_dim,
                    "pwm_dim": cfg.model.pwm_dim},
            "dyn_norm": train_ds.dyn_norm.state_dict(),
            "pwm_norm": train_ds.pwm_norm.state_dict(),
            "epoch": epoch,
            "metrics": rec,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if mix_ou and ou_loader is not None:
            ou_mae = rec["eval_ou"]["dvl_mae_ms"]
            if ou_mae < best:
                best = ou_mae
                torch.save(ckpt, ckpt_dir / f"{ckpt_name}.pt")
                if ckpt_name == "best":
                    torch.save(ckpt, ckpt_dir / "best_ou.pt")
        elif ev["dvl_mae_ms"] < best:
            best = ev["dvl_mae_ms"]
            torch.save(ckpt, ckpt_dir / f"{ckpt_name}.pt")

    hist_path.write_text(json.dumps(history, indent=2))
    print("saved", ckpt_dir / f"{ckpt_name}.pt", flush=True)
    return ckpt_dir / f"{ckpt_name}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all downloaded episodes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mix-ou", action="store_true")
    ap.add_argument("--resume", default="", help="optional ckpt to fine-tune (keep USIM norms)")
    ap.add_argument("--extra-ou", default="", help="comma-separated extra OU-schema dirs to mix in")
    ap.add_argument("--extra-reps", type=int, default=48, help="upsampling for --extra-ou episodes")
    ap.add_argument("--no-usim", action="store_true", help="ablation: train on OU+planner only")
    ap.add_argument("--ckpt-name", default="best", help="checkpoint stem (ablations must not clobber best)")
    ap.add_argument("--use-dt", action="store_true", help="dt-conditioned variant (rate generalization)")
    ap.add_argument("--use-object", action="store_true",
                    help="manipulation world model: arm + object relative pose (35-d state, 13-d action)")
    ap.add_argument("--use-arm", action="store_true",
                    help="manipulator extension: 29-d state / 13-d action (USIM arm episodes)")
    ap.add_argument("--usim-frac", type=float, default=1.0,
                    help="USIM-Hard data efficiency: seeded stratified fraction of USIM train episodes")
    ap.add_argument("--usim-seed", type=int, default=0)
    ap.add_argument("--usim-include", default="", help="comma-separated task_index list to keep")
    ap.add_argument("--usim-exclude", default="", help="comma-separated task_index list to hold out")
    ap.add_argument("--hidden", type=int, default=0,
                    help="model-scale ablation: MLP width (0 = config default 384; 192 -> 0.68M, 768 -> 8.9M)")
    ap.add_argument("--dist-dim", type=int, default=0,
                    help="model-scale ablation: disturbance token dim (0 = config default 96)")
    ap.add_argument("--no-token", action="store_true",
                    help="ablation: no disturbance token (d_t = 0 in training and at inference)")
    args = ap.parse_args()
    cfg = Cfg()
    cfg.paths.usim = Path(args.usim)
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.workers
    cfg.train.lr = args.lr
    cfg.model.n_tasks = 9
    cfg.model.use_language = False
    cfg.model.use_dt = args.use_dt
    cfg.model.no_token = bool(args.no_token)
    if args.hidden > 0:
        cfg.model.hidden = args.hidden
    if args.dist_dim > 0:
        cfg.model.disturbance_dim = args.dist_dim
    resume = Path(args.resume) if args.resume else None
    extra = [p.strip() for p in args.extra_ou.split(",") if p.strip()]
    parse_ids = lambda s: [int(x) for x in s.split(",") if x.strip()]
    train(cfg, max_episodes=args.max_episodes, epochs=args.epochs, mix_ou=args.mix_ou,
          resume=resume, extra_ou=extra, extra_reps=args.extra_reps,
          no_usim=args.no_usim, ckpt_name=args.ckpt_name, use_arm=args.use_arm, use_object=args.use_object,
          usim_frac=args.usim_frac, usim_seed=args.usim_seed,
          usim_include=parse_ids(args.usim_include), usim_exclude=parse_ids(args.usim_exclude))


if __name__ == "__main__":
    main()
