#!/usr/bin/env python3
"""Offline smoke test of the WAM server's amortized action source (no sim, no HTTP).

Runs the same synthetic goto observation through WamPolicy.act() with action_source
mpc / direct / direct_cem and checks that every path returns a well-formed chunk whose
translational intent (surge toward the goal) agrees in sign, plus reports per-act latency.
"""
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_policy_server import WamPolicy  # noqa: E402

ROOT = Path("/tmp/smoke_direct_root")
(ROOT / "goto_charge_station" / "logs").mkdir(parents=True, exist_ok=True)
# waypoint file the server picks up as the current trajectory: 3 nodes straight ahead (+x)
wps = np.array([[2.0, 0.0, 3.0, 0.0], [6.0, 0.0, 3.0, 0.0], [10.0, 0.0, 3.0, 0.0]], np.float64)
np.save(ROOT / "goto_charge_station" / "logs" / "episode_0_traj.npy", wps)


def make(source):
    args = types.SimpleNamespace(
        ckpt="/hy-tmp/models/uwam/best_scenes.pt", vel_ens="/hy-tmp/models/uwam/vel_ens_scenes.pt",
        gate_calib="/hy-tmp/models/uwam/gate_calib.json", ou="", eval_root=str(ROOT), hold_anchor="mixer",
        dump_dir="", action_source=source, direct_ckpt="/hy-tmp/models/uwam/direct_head.pt",
        n_samples=0, cem_iters=0)
    return WamPolicy(args)


def obs(pos, vel=(0.0, 0.0, 0.0)):
    L = 16
    return {
        "state.hist_dvl": np.tile(np.asarray(vel, np.float32), (L, 1)),
        "state.hist_imu_av": np.zeros((L, 3), np.float32),
        "state.hist_imu_la": np.tile(np.array([0, 0, -9.8], np.float32), (L, 1)),
        "state.hist_pressure": np.full((L, 1), 3.0, np.float32),
        "state.hist_dvl_h": np.full((L, 1), 2.0, np.float32),
        "state.hist_pwm": np.zeros((L, 8), np.float32),
        "state.odom_pos": np.asarray(pos, np.float32).reshape(1, 3),
        "state.odom_rpy": np.zeros((1, 3), np.float32),
        "state.imu_rpy": np.zeros((1, 3), np.float32),
        "meta.dvl_valid": np.array([[1.0]], np.float32),
        "annotation.human.action.task_description": ["Go to the charge station"],
    }


def imagined_velocity(pol, o, chunk):
    """Roll the first 0.5 s of the emitted chunk through the core: predicted mean DVL velocity (m/s)."""
    import torch

    from uwam.direct import ImaginedCost

    frames, hist_a, _ = pol._frames_from_obs(o)
    cost = ImaginedCost(pol.model, pol.dyn_norm, pol.pwm_norm, pol.cfg, pol.device)
    f = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None], device=pol.device)
    with torch.no_grad():
        _, info = cost(f(frames), f(hist_a), f(frames[-1]), f(np.zeros(3, np.float32)), f(chunk[:5]))
    return info["s_hat"][0, :5, 0:3].mean(dim=0).cpu().numpy()


out = {}
for src in ("mpc", "direct", "direct_cem"):
    pol = make(src)
    o = obs([0.0, 0.0, 3.0])
    a = pol.act(o)[0]
    pwm = a["action.pwm"]
    assert pwm.shape == (16, 8) and np.all(np.abs(pwm) <= 1.0 + 1e-6), pwm.shape
    v_hat = imagined_velocity(pol, o, pwm)
    t0 = time.perf_counter()
    for k in range(10):
        pol.act(obs([0.2 * k, 0.0, 3.0], vel=(0.2, 0.0, 0.0)))
    dt = (time.perf_counter() - t0) / 10
    out[src] = (v_hat, dt)
    print(f"{src:>11}: imagined DVL over the first 0.5 s from rest = ({v_hat[0]:+.3f}, {v_hat[1]:+.3f}, {v_hat[2]:+.3f}) m/s "
          f"for v_goal (+0.25, 0, 0)  |u| {np.abs(pwm).mean():.3f}  act latency {1e3 * dt:.1f} ms  "
          f"plan latency mean {np.mean(pol._plan_ms):.1f} ms", flush=True)
for src, (v, _) in out.items():
    assert v[0] > 0.02, f"{src}: imagined surge not positive: {v}"
    assert abs(v[1]) < 0.15 and abs(v[2]) < 0.15, f"{src}: imagined sway/heave too large: {v}"
print("SMOKE_DIRECT_OK")
