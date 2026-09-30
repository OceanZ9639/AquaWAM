#!/usr/bin/env python3
"""Sampling-MPC closed loop: 6 regimes x 3 goals = 18 trials, stepped on the DVL 10 Hz clock.

Every trial is re-homed to the spawn pose first, so all 54 trials share one initial condition
(`respawn_robot` aborts the simulator, so homing is done with a pose PD controller).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.config import Cfg
from uwam.control import (
    AXIS_SIGN,
    LinearSystemID,
    RecursiveSystemID,
    SamplingMPC,
    attitude_pwm,
    sixdof_thrust_to_pwm,
    vel_track_pwm,
)
from uwam.data import RunningNorm, load_ou_split, load_split
from uwam.gate import CusumGate
from uwam.models import DynamicsWAM
from uwam.sim import REGIME_SETS, REGIMES, apply_efficiency, regime_at

GOALS = [
    np.array([0.25, 0.0, 0.0], np.float32),
    np.array([0.0, 0.20, 0.0], np.float32),
    np.array([0.0, 0.0, 0.15], np.float32),
]
HARD_REGIMES = ("time_varying", "thruster_degrade", "current_and_fail")
# setpoint steps for the goal-step experiment: surge->sway, sway->surge, surge reversal
GOAL_STEPS = [
    (np.array([0.20, 0.0, 0.0], np.float32), np.array([0.0, 0.20, 0.0], np.float32)),
    (np.array([0.0, 0.20, 0.0], np.float32), np.array([0.20, 0.0, 0.0], np.float32)),
    (np.array([0.20, 0.0, 0.0], np.float32), np.array([-0.20, 0.0, 0.0], np.float32)),
]
TRACK_TICKS = 30  # last 3.0 s @ 10 Hz
HOVER_FRAC = 0.70
GOAL_FRAC = 0.45
TRACK_FLOOR = 0.08
TARGET_DT = 0.1
SETTLE_TICKS = 20


def track_threshold(goal: np.ndarray) -> float:
    g = float(np.linalg.norm(goal))
    return max(TRACK_FLOOR, GOAL_FRAC * g)


def _controller_stats(rows: list) -> dict:
    if not rows:
        return {"n_ok": 0, "n": 0, "mean_track": None, "mean_rel_err": None}
    tracks = [float(r.get("track_err", r.get("final_err", 99))) for r in rows]
    rels = [float(r["rel_err"]) for r in rows if r.get("rel_err") is not None]
    return {
        "n_ok": sum(bool(r.get("ok")) for r in rows),
        "n": len(rows),
        "n_unstable": sum(bool(r.get("unstable")) for r in rows),
        "n_beat_hover": sum(bool(r.get("beat_hover")) for r in rows),
        "mean_track": float(np.mean(tracks)),
        "max_track": float(np.max(tracks)),
        "min_track": float(np.min(tracks)),
        "mean_rel_err": float(np.mean(rels)) if rels else None,
        "track_floor": TRACK_FLOOR,
        "goal_frac": GOAL_FRAC,
    }


def _load_model(ckpt_path: Path, device: str):
    import torch

    cfg = Cfg()
    cfg.model.use_language = False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" in ckpt:
        c = ckpt["cfg"]
        if "disturbance_dim" in c:
            cfg.model.disturbance_dim = int(c["disturbance_dim"])
        if "hidden" in c:
            cfg.model.hidden = int(c["hidden"])
        cfg.model.use_dt = bool(c.get("use_dt", False))
    model = DynamicsWAM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    dyn_norm = RunningNorm()
    pwm_norm = RunningNorm()
    dyn_norm.load_state_dict(ckpt["dyn_norm"])
    pwm_norm.load_state_dict(ckpt["pwm_norm"])
    return model, dyn_norm, pwm_norm, cfg


def _fit_sysid(usim: Path, ou_dir: Path) -> LinearSystemID:
    cfg = Cfg()
    eps = load_split(usim, "train", cfg.schema, max_episodes=80)
    eps += load_ou_split(ou_dir)
    vs, us, v2 = [], [], []
    for ep in eps:
        v = ep.dyn[:, 0:3]
        u = ep.pwm
        if len(v) < 3:
            continue
        vs.append(v[:-1])
        us.append(u[:-1])
        v2.append(v[1:])
    sid = LinearSystemID()
    if vs:
        sid.fit(np.concatenate(vs), np.concatenate(us), np.concatenate(v2))
    return sid


def _recovery_metrics(times: list, errs: list, thr: float, t_change: float, window: float = 4.0) -> dict:
    """Seconds to get back under the gate after a regime switch, plus the post-switch error integral."""
    post = [(t, e) for t, e in zip(times, errs) if t >= t_change]
    if not post:
        return {"recover_s": None, "err_integral": None, "n_post": 0}
    recover = None
    for t, e in post:
        if e < thr:
            recover = round(t - t_change, 2)
            break
    seg = [e for t, e in post if t <= t_change + window]
    return {
        "recover_s": recover,
        "err_integral": round(float(np.mean(seg) * min(window, post[-1][0] - t_change)), 4) if seg else None,
        "n_post": len(post),
    }


def run_trial(bridge, mpc, sid, mode: str, regime, goal, sim_seconds: float,
              drop_dvl_at: float | None = None,
              goal2=None, goal_step_at: float | None = None,
              trace_path: Path | None = None,
              control_every: int = 1,
              native_coarse: bool = False,
              velmm_rt=None,
              calib_dump: Path | None = None,
              hold_anchor: str = "mean",
              hold_anchor_win: int = 10) -> dict:
    """Advance one controller step per DVL tick (Δt ≈ 0.1 s sim time).

    `drop_dvl_at` blinds the controller's DVL after that many seconds of scored time; scoring
    always uses the true DVL. Blind model-free arms hold their last pre-dropout command plus a
    feedforward correction for any commanded goal change (goals are commands, not measurements,
    so the baselines are allowed to react to them) — with a constant goal this reduces to plain
    hold, the strongest simple blind policy. `wam` dead-reckons per tick and keeps replanning.

    `goal2`/`goal_step_at` switch the setpoint mid-trial (absolute trial seconds, like change_at).
    """
    home = bridge.home()
    L = 16
    hist_s, hist_a = [], []
    last_u = np.zeros(8, np.float32)
    u_trans_hold = None
    u_trans_hist: list = []
    goal_at_blind = None
    rls = RecursiveSystemID().sync_from(sid) if mode == "rls" else None
    rls_prev = None  # (dvl, commanded u) of the previous fully-sighted tick
    if hasattr(mpc, "reset"):
        mpc.reset()
    if velmm_rt is not None:
        velmm_rt.reset()
    errs, times = [], []
    unstable = ""
    t0_sim = None
    yaw0 = None
    tick_idx = -1
    u_zoh = None  # last controller output, held between control ticks (ZOH rate adapter)
    t_start = SETTLE_TICKS * TARGET_DT
    trace = {k: [] for k in ("t", "v_true", "v_used", "u", "rpy", "goal", "blind")} if trace_path else None
    goal = np.asarray(goal, np.float32)
    goal2 = None if goal2 is None else np.asarray(goal2, np.float32)
    # uncertainty gate for the blind WAM policy: replan only as much as the evidence warrants
    c_gate = getattr(mpc, "cfg", None)
    gate_lo = float(getattr(c_gate, "blind_gate_lo", 0.05))
    gate_hi = float(getattr(c_gate, "blind_gate_hi", 0.12))
    alpha_decay = float(getattr(c_gate, "blind_alpha_decay", 0.90))
    n_smooth = int(getattr(c_gate, "blind_innov_smooth", 5))
    alpha = 0.0
    alphas = []
    # anchor and signal must share the estimator's bias so it cancels in the innovation:
    # sighted ticks also run the estimator, and the gate compares estimate against estimate
    est_sighted: list = []
    est_hist: list = []   # blind estimates, smoothed before the innovation gate
    v_anchor = None
    goal_latched = False
    # dimensionless CUSUM gate (same code as the gym ports); engages when the sigma
    # ensemble is loaded, otherwise the legacy linear gate below remains in effect.
    # kappa/h come from held-out false-alarm calibration when available.
    cusum = CusumGate(getattr(mpc, "gate_cfg", None))
    _std = getattr(mpc, "dyn_norm", None)
    vel_std = _std.std[0:3].astype(np.float64) if _std is not None else np.ones(3)
    g_prev = None
    # calibration mode: pure hold during blind (alpha forced 0) while the gate inputs
    # (smoothed innovation, sigma) are dumped -- deployment-condition z streams
    calib = {k: [] for k in ("t", "v_sm_n", "anchor_n", "sig_eff", "dvl_true")} \
        if calib_dump is not None else None
    while True:
        st = bridge.tick()
        if st is None:
            break
        if t0_sim is None:
            t0_sim = st["stamp"]
        t = st["stamp"] - t0_sim
        g_now = goal2 if (goal2 is not None and goal_step_at is not None and t >= goal_step_at) else goal
        if g_prev is not None and not np.allclose(g_now, g_prev):
            # stale elites anchor the CEM to the old setpoint for several ticks after a step
            if hasattr(mpc, "reset"):
                mpc.reset()
        g_prev = g_now.copy()
        dvl_true, imu_av, rpy = st["dvl"], st["imu_av"], st["rpy"]
        if yaw0 is None:
            yaw0 = float(rpy[2])
        # hold the initial heading for every arm: asymmetric faults otherwise yaw the vehicle
        # tens of degrees (P0 traces: 27-54 deg) and the body-frame goal slowly loses meaning
        yaw_err = float(np.arctan2(np.sin(yaw0 - rpy[2]), np.cos(yaw0 - rpy[2])))
        blind = drop_dvl_at is not None and t >= t_start + drop_dvl_at
        dvl = np.zeros(3, np.float32) if blind else dvl_true
        cur, eta = regime_at(regime, max(0.0, t))
        bridge.publish_current(cur)
        alt = st["alt"] if st["alt_valid"] else 1.0
        s = np.concatenate([dvl, imu_av, st["imu_la"], [st["pressure"]], [alt], last_u]).astype(np.float32)
        # history stays strictly in the past ([t-L, t-1]), matching the training windows; the
        # current tick is appended only after the controller has acted on it
        if blind and mode in ("wam", "no_disturb", "wam_mm") and len(hist_s) >= L:
            # per-tick dead reckoning: write the estimate into this tick's state so the history
            # keeps a plausible velocity profile, not zeros (estimate_velocity re-masks the DVL
            # columns internally, so the estimator itself never sees what we store here)
            v_est = None
            sig_alea = sig_epi = None
            if mode == "wam_mm" and velmm_rt is not None:
                v_est = velmm_rt.estimate(np.stack(hist_s, 0), np.stack(hist_a, 0),
                                          st.get("rgb"), st.get("fls"))
            if v_est is None:
                # mean from the d-informed head (the strong dead-reckoner); the NLL
                # ensemble contributes ONLY the sigma scale for the gate -- its own mean
                # is a weaker estimator and demonstrably degrades blind replanning
                v_est = mpc.estimate_velocity(np.stack(hist_s, 0), np.stack(hist_a, 0))
                if hasattr(mpc, "estimate_velocity_ens"):
                    ens = mpc.estimate_velocity_ens(np.stack(hist_s, 0), np.stack(hist_a, 0))
                    if ens is not None:
                        sig_alea, sig_epi = ens[1], ens[2]
            if v_est is not None:
                s[0:3] = v_est
                dvl = v_est
                # innovation vs the hold hypothesis, in the estimator's OWN coordinates: the
                # estimator carries a persistent per-operating-point bias (~0.05 at cruise), so a
                # ground-truth anchor would read that bias as a permanent phantom change
                if v_anchor is None and est_sighted:
                    v_anchor = np.mean(np.stack(est_sighted, 0), axis=0)
                    cusum.start_blind(v_anchor / vel_std)
                est_hist.append(v_est.copy())
                est_hist = est_hist[-n_smooth:]
                if goal_at_blind is not None and not np.allclose(g_now, goal_at_blind):
                    goal_latched = True
                v_sm = np.mean(np.stack(est_hist, 0), axis=0)
                if calib is not None and sig_alea is not None:
                    sig_eff = np.sqrt(sig_alea ** 2 / max(1, len(est_hist)) + sig_epi ** 2)
                    if v_anchor is not None:
                        calib["t"].append(t)
                        calib["v_sm_n"].append(v_sm / vel_std)
                        calib["anchor_n"].append(v_anchor / vel_std)
                        calib["sig_eff"].append(sig_eff)
                        calib["dvl_true"].append(dvl_true.copy())
                    alpha = 0.0
                elif sig_alea is not None:
                    # calibrated path: dimensionless CUSUM on the smoothed innovation.
                    # The i.i.d. aleatoric part averages out over the smoothing window;
                    # epistemic disagreement is systematic and does not.
                    sig_eff = np.sqrt(sig_alea ** 2 / max(1, len(est_hist)) + sig_epi ** 2)
                    alpha = cusum.step(v_sm / vel_std, sig_eff, goal_changed=goal_latched)
                elif goal_latched:
                    alpha = 1.0
                elif v_anchor is not None and len(est_hist) >= n_smooth:
                    innov = float(np.linalg.norm(v_sm - v_anchor))
                    a_raw = float(np.clip((innov - gate_lo) / max(1e-6, gate_hi - gate_lo), 0.0, 1.0))
                    alpha = max(a_raw, alpha * alpha_decay)
                else:
                    alpha = max(0.0, alpha * alpha_decay)
        if not blind:
            if mode in ("wam", "no_disturb", "wam_mm") and drop_dvl_at is not None and len(hist_s) >= L:
                v_sight = mpc.estimate_velocity(np.stack(hist_s, 0), np.stack(hist_a, 0))
                if v_sight is not None:
                    est_sighted.append(v_sight)
                    est_sighted = est_sighted[-n_smooth:]
            v_anchor = None
            est_hist = []
            goal_latched = False
            alpha = 0.0
            cusum.reset()
        if mode == "rls" and not blind and rls_prev is not None:
            rls.update(rls_prev[0], rls_prev[1], dvl_true)
        tick_idx += 1
        if control_every > 1 and tick_idx % control_every != 0 and u_zoh is not None and t >= t_start:
            # zero-order-hold rate adapter: sensors stay on the native 10 Hz clock, the
            # controller acts every N-th tick; the model itself is untouched
            u = u_zoh.copy()
        elif t < t_start:
            u = vel_track_pwm(dvl, np.zeros(3, np.float32), imu_av, rpy, yaw_err=yaw_err)
        elif mode in ("mixer", "sysid", "rls") and blind and u_trans_hold is not None:
            # hold the translational command; the attitude inner loop is IMU-driven and keeps
            # running, and a commanded goal change gets an open-loop feedforward retarget
            u = u_trans_hold.copy()
            if goal_at_blind is not None and not np.allclose(g_now, goal_at_blind):
                dg = AXIS_SIGN * 2.0 * (g_now - goal_at_blind)
                u = u + sixdof_thrust_to_pwm(np.clip(dg, -1, 1))
            u = np.clip(u + attitude_pwm(rpy, imu_av, yaw_err=yaw_err), -1.0, 1.0)
        elif mode == "mixer":
            u = vel_track_pwm(dvl, g_now, imu_av, rpy, yaw_err=yaw_err)
        elif mode == "sysid":
            u = sid.closed_loop_pwm(dvl, g_now, omega=imu_av, rpy=rpy, yaw_err=yaw_err)
        elif mode == "rls":
            u = rls.closed_loop_pwm(dvl, g_now, omega=imu_av, rpy=rpy, yaw_err=yaw_err)
        elif len(hist_s) < L:
            u = vel_track_pwm(dvl, g_now, imu_av, rpy, yaw_err=yaw_err)
        else:
            hs = np.stack(hist_s, 0)
            ha = np.stack(hist_a, 0)
            if mode == "no_disturb":
                hs = hs.copy()
                hs[:, 0:9] = 0
            sid_u = sid.closed_loop_pwm(dvl, g_now, omega=imu_av, rpy=rpy, yaw_err=yaw_err)
            u, _info = mpc.plan(hs, ha, s, g_now, task_index=0, sid_u=sid_u, rpy=rpy, yaw_err=yaw_err)
            if blind and u_trans_hold is not None:
                # uncertainty-gated hybrid: hold is optimal when nothing changed, replanning is
                # only trusted in proportion to the evidence (goal change latches alpha to 1,
                # dynamics drift raises it through the innovation gate)
                u_hold_now = u_trans_hold.copy()
                if goal_at_blind is not None and not np.allclose(g_now, goal_at_blind):
                    dg = AXIS_SIGN * 2.0 * (g_now - goal_at_blind)
                    u_hold_now = u_hold_now + sixdof_thrust_to_pwm(np.clip(dg, -1, 1))
                u_hold_now = np.clip(u_hold_now + attitude_pwm(rpy, imu_av, yaw_err=yaw_err), -1.0, 1.0)
                u = np.clip(alpha * u + (1.0 - alpha) * u_hold_now, -1.0, 1.0)
                alphas.append(alpha)
        u_zoh = u.copy()
        # hold the pre-fault TRANSLATIONAL command only: efficiency is the plant (saving post-eta
        # would apply the fault twice), and the attitude correction must stay live during blind.
        # Hold the 1 s MEAN command, not the last tick: CEM outputs twitch around the operating
        # point, and freezing a single tick locks the twitch in (costs ~0.01-0.03 m/s for wam;
        # a no-op for the smooth mixer/sysid controllers).
        if not blind and t >= t_start:
            u_trans_hist.append(u - attitude_pwm(rpy, imu_av, yaw_err=yaw_err))
            u_trans_hist = u_trans_hist[-int(hold_anchor_win):]
            if hold_anchor == "mixer":
                # closed-form mixer hold of the current velocity goal: no CEM sampling
                # noise in the frozen command (the underwater-benchmark weak arm).
                u_trans_hold = vel_track_pwm(dvl, g_now, omega=imu_av, rpy=rpy, yaw_err=yaw_err)
            elif hold_anchor == "median":
                # per-channel median over a longer window: same information pathway as the
                # mean, but robust to CEM sampling outliers that a mean drags along
                u_trans_hold = np.clip(np.median(np.stack(u_trans_hist, 0), axis=0), -1.0, 1.0)
            else:
                u_trans_hold = np.clip(np.mean(np.stack(u_trans_hist, 0), axis=0), -1.0, 1.0)
            goal_at_blind = g_now.copy()
        if mode == "rls":
            rls_prev = None if blind else (dvl_true.copy(), u.copy())
        # native_coarse: a dt-conditioned model running at the coarse rate keeps its history on
        # that same grid; otherwise the window would cover 1.6 s regardless of the control rate
        if not native_coarse or tick_idx % control_every == 0:
            hist_s.append(s)
            hist_a.append(last_u.copy())
            if len(hist_s) > L:
                hist_s = hist_s[-L:]
                hist_a = hist_a[-L:]
        # last_u (history + state pwm columns) keeps the COMMANDED value: a real vehicle only
        # knows what it commanded, and the thruster fault is exactly the unobserved gap the
        # disturbance token has to infer from the state trajectory
        last_u = u.copy()
        bridge.publish_pwm(apply_efficiency(u, eta))
        if trace is not None:
            trace["t"].append(t)
            trace["v_true"].append(dvl_true.copy())
            trace["v_used"].append(dvl.copy())
            trace["u"].append(u.copy())
            trace["rpy"].append(rpy.copy())
            trace["goal"].append(g_now.copy())
            trace["blind"].append(bool(blind))
        if t >= t_start:
            errs.append(float(np.linalg.norm(dvl_true - g_now)))
            times.append(t)
            bad, why = bridge.is_unstable(st)
            if bad:
                unstable = why
                break
        if t >= t_start + sim_seconds:
            break
    bridge.publish_pwm(np.zeros(8, np.float32))
    g_final = goal2 if goal2 is not None else goal
    n_track = min(TRACK_TICKS, len(errs))
    track = float(np.mean(errs[-n_track:])) if n_track else 99.0
    g_n = float(np.linalg.norm(g_final))
    thr = track_threshold(g_final)
    rec = {
        "ok": (not unstable) and track < thr,
        "track_err": track,
        "final_err": track,
        "rel_err": track / max(1e-6, g_n),
        "beat_hover": bool(track < HOVER_FRAC * g_n),
        "track_thr": thr,
        "unstable": unstable,
        "n": len(errs),
        "dt": TARGET_DT,
        "controller": mode,
        "mode": mode,
        "goal": [float(x) for x in g_final.tolist()],
        "regime": regime.name,
        "hard": regime.name in HARD_REGIMES,
        "drop_dvl_at": drop_dvl_at,
        "home": home,
    }
    if goal2 is not None:
        rec["goal_pair"] = [[float(x) for x in goal.tolist()], [float(x) for x in goal2.tolist()]]
        rec["goal_step_at"] = goal_step_at
    t_change = regime.change_at if regime.change_at is not None else goal_step_at
    if t_change is not None:
        rec["recovery"] = _recovery_metrics(times, errs, thr, float(t_change))
    if drop_dvl_at is not None:
        blind_errs = [e for t, e in zip(times, errs) if t >= t_start + drop_dvl_at]
        rec["blind_track_err"] = round(float(np.mean(blind_errs)), 4) if blind_errs else None
        rec["blind_ok"] = bool(blind_errs and float(np.mean(blind_errs)) < thr)
        if alphas:
            rec["blind_alpha_mean"] = round(float(np.mean(alphas)), 3)
            rec["blind_alpha_max"] = round(float(np.max(alphas)), 3)
    if trace is not None and trace["t"]:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            t=np.asarray(trace["t"], np.float64),
            v_true=np.stack(trace["v_true"]),
            v_used=np.stack(trace["v_used"]),
            u=np.stack(trace["u"]),
            rpy=np.stack(trace["rpy"]),
            goal=np.stack(trace["goal"]),
            blind=np.asarray(trace["blind"]),
        )
        rec["trace"] = str(trace_path)
    if calib is not None and calib["t"]:
        calib_dump.mkdir(parents=True, exist_ok=True)
        gtag = "".join(f"{x:+.2f}" for x in np.asarray(goal).tolist())
        f = calib_dump / f"calib_{regime.name}_{gtag}.npz"
        np.savez_compressed(
            f,
            t=np.asarray(calib["t"], np.float64),
            v_sm_n=np.stack(calib["v_sm_n"]),
            anchor_n=np.stack(calib["anchor_n"]),
            sig_eff=np.stack(calib["sig_eff"]),
            dvl_true=np.stack(calib["dvl_true"]),
            change_at=np.float64(regime.change_at if regime.change_at is not None else -1.0),
            t_start=np.float64(t_start),
        )
        rec["calib_dump"] = str(f)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/hy-tmp/models/uwam/best.pt")
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--ou", default="/hy-tmp/data/ou_explore")
    ap.add_argument("--seconds", type=float, default=12.0, help="sim-time seconds (DVL ticks * 0.1)")
    ap.add_argument("--modes", default="mixer,sysid,no_disturb,wam")
    ap.add_argument("--out", default="/hy-tmp/logs/uwam/closed_loop.json")
    ap.add_argument("--drop-dvl-at", type=float, default=None,
                    help="blind the controller's DVL this many scored seconds in")
    ap.add_argument("--regime-set", default="main", choices=sorted(REGIME_SETS),
                    help="main = the 6 collection regimes; fault_mid = horizontal-thruster faults at t=8s")
    ap.add_argument("--goal-step-at", type=float, default=None,
                    help="switch to the second goal of each GOAL_STEPS pair at this trial time (s)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the MPC candidate sampler (multi-seed repeats)")
    ap.add_argument("--regimes", default="",
                    help="comma-separated regime-name filter within the chosen set")
    ap.add_argument("--goals", default="",
                    help="semicolon-separated goal triples overriding GOALS, e.g. '0.25,0,0;0,0.2,0'")
    ap.add_argument("--save-traces", default="",
                    help="directory for per-tick npz traces (diagnosis runs)")
    ap.add_argument("--control-every", type=int, default=1,
                    help="ZOH rate adapter: act every N-th DVL tick (2 = 5 Hz control)")
    ap.add_argument("--vel-mm", default="",
                    help="vel_mm.pt checkpoint enabling the wam_mm arm (camera+FLS dead reckoning)")
    ap.add_argument("--vel-ens", default="/hy-tmp/models/uwam/vel_ens_scenes.pt",
                    help="heteroscedastic dead-reckoning ensemble; '' disables (legacy linear gate)")
    ap.add_argument("--gate-calib", default="/hy-tmp/models/uwam/gate_calib.json",
                    help="kappa/h from held-out false-alarm calibration; '' = CUSUM defaults")
    ap.add_argument("--hold-anchor", choices=["mean", "median", "mixer"], default="mean",
                    help="hold-anchor: mean/median of recent CEM cmds, or mixer closed-form "
                         "steady command for the current velocity goal (no CEM twitch)")
    ap.add_argument("--hold-anchor-win", type=int, default=10,
                    help="ticks of pre-dropout commands in the hold anchor (10 = 1 s)")
    ap.add_argument("--calib-dump", default="",
                    help="dump deployment-condition gate streams (forces pure hold) to this dir")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    import torch

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dyn_norm, pwm_norm, cfg = _load_model(Path(args.ckpt), device)
    sid = _fit_sysid(Path(args.usim), Path(args.ou))
    mpc = SamplingMPC(model, dyn_norm, pwm_norm, cfg.control, device=device)
    if args.vel_ens:
        ok = mpc.load_vel_ensemble(args.vel_ens)
        print(f"vel ensemble: {'loaded ' + args.vel_ens if ok else 'NOT FOUND, legacy gate'}", flush=True)
    mpc.gate_cfg = None
    if args.gate_calib:
        from uwam.gate import GateCfg
        mpc.gate_cfg = GateCfg.from_calib(args.gate_calib)
        print(f"gate calib: {mpc.gate_cfg if mpc.gate_cfg else 'NOT FOUND, CUSUM defaults'}", flush=True)
    lib = [ep.pwm for ep in load_ou_split(Path(args.ou))]
    if not lib:
        lib = [ep.pwm for ep in load_split(Path(args.usim), "train", Cfg().schema, max_episodes=40)]
    if lib:
        mpc.set_library(np.concatenate(lib, axis=0))

    results = []
    if args.offline:
        for reg in REGIMES:
            for g in GOALS:
                u = sid.closed_loop_pwm(np.zeros(3, np.float32), g)
                results.append({"ok": bool(np.all(np.isfinite(u))), "mode": "sysid_offline",
                                "controller": "sysid_offline", "regime": reg.name, "goal": g.tolist()})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"n_ok": sum(r["ok"] for r in results), "n": len(results), "trials": results}, indent=2)
        )
        print("offline", len(results))
        return

    from uwam.rosbridge import SimBridge

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    velmm_rt = None
    if args.vel_mm and "wam_mm" in modes:
        from uwam.velmm import VelMM, VelMMRuntime

        mmck = torch.load(args.vel_mm, map_location=device, weights_only=False)
        velmm = VelMM(cfg)
        velmm.load_state_dict(mmck["velmm"])
        velmm_rt = VelMMRuntime(velmm, model, dyn_norm, pwm_norm, device=device)
        print(f"wam_mm estimator loaded from {args.vel_mm}", flush=True)

    bridge = SimBridge(node="uwam_closed_loop", images="wam_mm" in modes)
    if not bridge.wait_ready():
        raise SystemExit("no DVL/odometry from the simulator")

    native_coarse = args.control_every > 1 and bool(getattr(model, "use_dt", False))
    mpc.dt_s = TARGET_DT * args.control_every if native_coarse else TARGET_DT
    if native_coarse:
        print(f"dt-conditioned model at native {mpc.dt_s:.1f}s grid (control_every={args.control_every})",
              flush=True)
    regimes = REGIME_SETS[args.regime_set]
    if args.regimes:
        wanted = {r.strip() for r in args.regimes.split(",") if r.strip()}
        regimes = [r for r in regimes if r.name in wanted]
    goals = GOALS
    if args.goals:
        goals = [np.array([float(x) for x in trip.split(",")], np.float32)
                 for trip in args.goals.split(";") if trip.strip()]
    trace_dir = Path(args.save_traces) if args.save_traces else None

    def _trace_path(mode, reg, g):
        if trace_dir is None:
            return None
        gs = "_".join(f"{x:+.2f}" for x in np.asarray(g).reshape(-1))
        return trace_dir / f"{mode}_{reg.name}_{gs}_s{args.seed}.npz"

    for mode in modes:
        for reg in regimes:
            if args.goal_step_at is not None:
                for g_a, g_b in GOAL_STEPS:
                    rec = run_trial(bridge, mpc, sid, mode, reg, g_a, args.seconds,
                                    drop_dvl_at=args.drop_dvl_at,
                                    goal2=g_b, goal_step_at=args.goal_step_at,
                                    trace_path=_trace_path(mode, reg, g_b),
                                    control_every=args.control_every,
                                    native_coarse=native_coarse,
                                    velmm_rt=velmm_rt if mode == "wam_mm" else None,
                                    hold_anchor=args.hold_anchor,
                                    hold_anchor_win=args.hold_anchor_win)
                    results.append(rec)
                    print(json.dumps(rec), flush=True)
            else:
                for g in goals:
                    rec = run_trial(bridge, mpc, sid, mode, reg, g, args.seconds,
                                    drop_dvl_at=args.drop_dvl_at,
                                    trace_path=_trace_path(mode, reg, g),
                                    control_every=args.control_every,
                                    native_coarse=native_coarse,
                                    velmm_rt=velmm_rt if mode == "wam_mm" else None,
                                    calib_dump=Path(args.calib_dump) if args.calib_dump else None,
                                    hold_anchor=args.hold_anchor,
                                    hold_anchor_win=args.hold_anchor_win)
                    results.append(rec)
                    print(json.dumps(rec), flush=True)

    def _split(ctrl: str):
        rows = [r for r in results if r.get("controller") == ctrl]
        hard = [r for r in rows if r.get("hard") or r.get("regime") in HARD_REGIMES]
        return _controller_stats(rows), _controller_stats(hard)

    summary = {
        "dt_infer": TARGET_DT,
        "track_ticks": TRACK_TICKS,
        "track_rule": "max(0.08, 0.45*||goal||) over last 3.0s",
        "reset": "pose-PD home to spawn (0,0,4) before every trial (respawn_robot aborts the sim)",
        "goals": [[float(x) for x in g] for g in GOALS],
        "drop_dvl_at": args.drop_dvl_at,
        "goal_step_at": args.goal_step_at,
        "goal_steps": [[a.tolist(), b.tolist()] for a, b in GOAL_STEPS] if args.goal_step_at is not None else None,
        "blind_policy": "mixer/sysid hold last pre-dropout command (pre-fault, eta applied once) "
                        "+ feedforward retarget on commanded goal changes; "
                        "wam dead-reckons per tick and keeps replanning",
        "regime_set": args.regime_set,
        "regimes": [r.name for r in regimes],
        "seed": args.seed,
        "control_every": args.control_every,
        "control_hz": round(10.0 / max(1, args.control_every), 2),
        "modes": modes,
        "arms": {},
        "hard_9": {},
        "trials": results,
    }
    for mode in modes:
        summary["arms"][mode], summary["hard_9"][mode] = _split(mode)
    # keep the keys write_paper_table.py already reads
    for mode, key in (("wam", "wam_18"), ("sysid", "sysid"), ("no_disturb", "no_disturb")):
        if mode in summary["arms"]:
            summary[key] = summary["arms"][mode]
    if args.drop_dvl_at is not None:
        summary["blind"] = {
            m: {
                "n_ok": sum(bool(r.get("blind_ok")) for r in results if r["controller"] == m),
                "n": sum(1 for r in results if r["controller"] == m),
                "mean_blind_track": float(np.mean([r["blind_track_err"] for r in results
                                                   if r["controller"] == m and r.get("blind_track_err") is not None]))
                if any(r["controller"] == m and r.get("blind_track_err") is not None for r in results) else None,
            }
            for m in modes
        }
    recs = [r for r in results if r.get("recovery")]
    if recs:
        summary["change_point_recovery"] = {
            m: {
                "recover_s": [r["recovery"]["recover_s"] for r in recs if r["controller"] == m],
                "mean_err_integral": float(np.mean([r["recovery"]["err_integral"] for r in recs
                                                    if r["controller"] == m and r["recovery"]["err_integral"] is not None]))
                if any(r["controller"] == m and r["recovery"]["err_integral"] is not None for r in recs) else None,
            }
            for m in modes
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
