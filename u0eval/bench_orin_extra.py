#!/usr/bin/env python3
"""Extra on-device measurements of the cruising imagination planner (SamplingMPC), for the appendix:

  1. budget-latency curve: candidates {32,64,128,256,512} x refinement rounds {1,2} x lookahead {1,2,4} s
  2. sustained load: replan continuously for --sustain-s seconds at the deployed budget (128x2, 2 s),
     logging the latency every 10 s (thermal / clock drift check)
  3. power: while sustaining, sample the INA3221 rails through sysfs (Jetson) and report mean W per rail,
     idle W, and energy per decision (J) = (busy - idle GPU/SOC power) * median latency

    usage: bench_orin_extra.py --models ~/models --out out.json [--device cuda] [--sustain-s 600]
Same model loading as bench_latency.py (kept import-light for the Jetson venv)."""
import argparse
import glob
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def timeit(fn, n, sync):
    for _ in range(5):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": statistics.median(ts), "p95_ms": float(np.percentile(ts, 95)), "n": n}


# ---------------------------------------------------------------- power (Jetson INA3221 via sysfs)
def find_rails():
    """Return {rail_name: (curr_path, volt_path)} for every INA3221 channel exposed in sysfs."""
    rails = {}
    for hw in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*") + glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(os.path.join(hw, "name")).read().strip()
        except OSError:
            continue
        if "ina3221" not in name:
            continue
        for lab in glob.glob(os.path.join(hw, "in*_label")):
            idx = os.path.basename(lab)[2:-6]
            try:
                rail = open(lab).read().strip()
            except OSError:
                continue
            curr, volt = os.path.join(hw, f"curr{idx}_input"), os.path.join(hw, f"in{idx}_input")
            if os.path.exists(curr) and os.path.exists(volt):
                rails[rail] = (curr, volt)
    return rails


def read_power_w(rails):
    out = {}
    for rail, (curr, volt) in rails.items():
        try:
            mA = float(open(curr).read()); mV = float(open(volt).read())
            out[rail] = mA * mV / 1e6
        except (OSError, ValueError):
            pass
    return out


class PowerSampler(threading.Thread):
    def __init__(self, rails, period=0.2):
        super().__init__(daemon=True)
        self.rails, self.period, self.samples, self.stop = rails, period, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.samples.append(read_power_w(self.rails))
            time.sleep(self.period)

    def mean(self):
        keys = set().union(*self.samples) if self.samples else set()
        return {k: float(np.mean([s[k] for s in self.samples if k in s])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="/hy-tmp/models/uwam")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--sustain-s", type=int, default=600)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    sync = dev.startswith("cuda")

    from uwam.config import Cfg, enable_arm, enable_object  # noqa: E402
    from uwam.control import SamplingMPC  # noqa: E402
    from uwam.data import RunningNorm  # noqa: E402
    from uwam.models import DynamicsWAM  # noqa: E402

    def _load_model(ckpt_path, device):
        cfg = Cfg(); cfg.model.use_language = False
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "cfg" in ckpt:
            c = ckpt["cfg"]
            if "disturbance_dim" in c: cfg.model.disturbance_dim = int(c["disturbance_dim"])
            if "hidden" in c: cfg.model.hidden = int(c["hidden"])
            cfg.model.use_dt = bool(c.get("use_dt", False))
            if c.get("use_object"): enable_object(cfg)
            elif c.get("use_arm"): enable_arm(cfg)
        m = DynamicsWAM(cfg).to(device); m.load_state_dict(ckpt["model"], strict=False); m.eval()
        dn, pn = RunningNorm(), RunningNorm(); dn.load_state_dict(ckpt["dyn_norm"]); pn.load_state_dict(ckpt["pwm_norm"])
        return m, dn, pn, cfg

    model, dyn_norm, pwm_norm, cfg = _load_model(Path(args.models) / "best_scenes.pt", dev)
    D = cfg.model.dyn_state_dim
    mpc = SamplingMPC(model, dyn_norm, pwm_norm, cfg.control, device=dev)
    rng = np.random.default_rng(0)
    hist_s = (0.05 * rng.standard_normal((16, D))).astype(np.float32)
    hist_a = (0.2 * rng.standard_normal((16, 8))).astype(np.float32)
    v_goal = np.array([0.3, 0.0, 0.0], np.float32)
    plan = lambda: mpc.plan(hist_s, hist_a, hist_s[-1], v_goal, task_index=0, sid_u=None,
                            rpy=np.zeros(3, np.float32), yaw_err=0.0)
    res = {"info": {"device": dev, "torch": torch.__version__, "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "threads": torch.get_num_threads()}}
    try:
        res["info"]["tegra"] = open("/etc/nv_tegra_release").readline().strip()
    except OSError:
        pass

    # 1. budget-latency curve -------------------------------------------------------------------
    base_chunks = int(getattr(cfg.control, "horizon_chunks", 4))     # 4 x 0.5 s = 2 s lookahead
    curve = []
    for chunks, look_s in ((max(1, base_chunks // 2), 1.0), (base_chunks, 2.0), (base_chunks * 2, 4.0)):
        for it in (1, 2):
            for n_s in (32, 64, 128, 256, 512):
                mpc.cfg.n_samples, mpc.cfg.cem_iters, mpc.cfg.horizon_chunks = n_s, it, chunks
                mpc._prev_seq = None
                r = timeit(plan, args.repeats, sync)
                curve.append({"n_samples": n_s, "cem_iters": it, "lookahead_s": look_s, **r})
                print(f"[curve] {n_s}x{it} @ {look_s:.0f} s: median {r['median_ms']:.1f} ms  p95 {r['p95_ms']:.1f} ms", flush=True)
    res["budget_curve"] = curve
    mpc.cfg.n_samples, mpc.cfg.cem_iters, mpc.cfg.horizon_chunks = 128, 2, base_chunks
    mpc._prev_seq = None

    # 3. idle power ----------------------------------------------------------------------------
    rails = find_rails()
    res["power_rails"] = sorted(rails)
    if rails:
        ps = PowerSampler(rails); ps.start(); time.sleep(15); ps.stop.set(); ps.join()
        res["power_idle_w"] = ps.mean()
        print("[power] idle W:", {k: round(v, 2) for k, v in res["power_idle_w"].items()}, flush=True)
    else:
        print("[power] no INA3221 rails readable in sysfs", flush=True)

    # 2 (+3). sustained load at the deployed budget, power sampled meanwhile ----------------------
    ps = PowerSampler(rails) if rails else None
    if ps:
        ps.start()
    t_end = time.time() + args.sustain_s
    window, series, n_calls = [], [], 0
    t_win = time.time()
    while time.time() < t_end:
        t = time.perf_counter(); plan()
        if sync:
            torch.cuda.synchronize()
        window.append((time.perf_counter() - t) * 1e3); n_calls += 1
        if time.time() - t_win >= 10:
            series.append({"t_s": round(time.time() - (t_end - args.sustain_s), 1),
                           "median_ms": statistics.median(window), "p95_ms": float(np.percentile(window, 95)), "n": len(window)})
            print(f"[sustain] t={series[-1]['t_s']:.0f}s median {series[-1]['median_ms']:.1f} ms p95 {series[-1]['p95_ms']:.1f} ms", flush=True)
            window, t_win = [], time.time()
    if ps:
        ps.stop.set(); ps.join()
        res["power_busy_w"] = ps.mean()
        print("[power] busy W:", {k: round(v, 2) for k, v in res["power_busy_w"].items()}, flush=True)
    res["sustain"] = {"seconds": args.sustain_s, "calls": n_calls, "series": series,
                      "median_ms_first": series[0]["median_ms"] if series else None,
                      "median_ms_last": series[-1]["median_ms"] if series else None}
    if ps and series:
        med = statistics.median(s["median_ms"] for s in series)
        gpu_keys = [k for k in res["power_busy_w"] if "GPU" in k.upper() or "SOC" in k.upper() or "CV" in k.upper()]
        busy = sum(res["power_busy_w"][k] for k in gpu_keys); idle = sum(res["power_idle_w"].get(k, 0.0) for k in gpu_keys)
        tot_busy = sum(res["power_busy_w"].values()); tot_idle = sum(res["power_idle_w"].values())
        res["energy"] = {"rails_used": gpu_keys, "busy_w": busy, "idle_w": idle, "total_busy_w": tot_busy, "total_idle_w": tot_idle,
                         "median_ms": med, "incremental_J_per_decision": (busy - idle) * med / 1e3,
                         "total_J_per_decision": tot_busy * med / 1e3}
        print("[energy]", json.dumps(res["energy"]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print("[bench-extra] done", flush=True)


if __name__ == "__main__":
    main()
