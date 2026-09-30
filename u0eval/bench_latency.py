#!/usr/bin/env python3
"""Latency benchmark of the deployed WaterWAM policy components on a given device (desktop CPU/GPU or a
Jetson). Uses the production classes with synthetic inputs of the production shapes:
  - imagination planner for cruising (SamplingMPC, budgets 32x1 / 128x2 / 512x2)
  - pulse planner for fine positioning (GraspPlanner.plan_pulse, 35-d arm+object core)
  - image-goal head r4 (PerceptGoalE2E.predict_nav), wrist-camera head r2, container head r1
Reports median / p95 ms per call after warm-up and a per-control-step total, and writes JSON.
  python3 u0eval/bench_latency.py --device cuda --models /hy-tmp/models/uwam --out /tmp/bench_cuda.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "u0eval"))
sys.path.insert(0, str(ROOT / "percept"))


def timeit(fn, n: int, warm: int = 5, sync: bool = False):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t0))
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="/hy-tmp/models/uwam")
    ap.add_argument("--encoder", default="", help="DINOv2-base snapshot dir (default: percept.goal_head.ENCODER_DIR)")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    sync = dev.startswith("cuda")
    M = Path(args.models)
    info = {"device": dev, "torch": torch.__version__, "cpu": platform.processor() or platform.machine(),
            "threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node()}
    try:
        info["tegra"] = open("/etc/nv_tegra_release").readline().strip()
    except OSError:
        pass
    print("[bench]", json.dumps(info))
    res = {"info": info}

    from uwam.config import Cfg, enable_arm, enable_object  # noqa: E402
    from uwam.control import SamplingMPC  # noqa: E402
    from uwam.data import RunningNorm  # noqa: E402
    from uwam.models import DynamicsWAM  # noqa: E402

    def _load_model(ckpt_path, device):  # same as wam_policy_server._load_model (kept import-light for Jetson)
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

    # --- 1. cruising imagination planner --------------------------------------------------------
    model, dyn_norm, pwm_norm, cfg = _load_model(M / "best_scenes.pt", dev)
    D = cfg.model.dyn_state_dim
    mpc = SamplingMPC(model, dyn_norm, pwm_norm, cfg.control, device=dev)
    rng = np.random.default_rng(0)
    hist_s = (0.05 * rng.standard_normal((16, D))).astype(np.float32)
    hist_a = (0.2 * rng.standard_normal((16, 8))).astype(np.float32)
    v_goal = np.array([0.3, 0.0, 0.0], np.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench] dynamics core: {n_params/1e6:.2f} M params, state dim {D}, horizon {cfg.control.horizon}")
    res["core_params_M"] = n_params / 1e6
    for n_s, it in ((32, 1), (128, 2), (512, 2)):
        mpc.cfg.n_samples, mpc.cfg.cem_iters = n_s, it
        mpc._prev_seq = None
        r = timeit(lambda: mpc.plan(hist_s, hist_a, hist_s[-1], v_goal, task_index=0, sid_u=None, rpy=np.zeros(3, np.float32), yaw_err=0.0),
                   args.repeats, sync=sync)
        res[f"mpc_{n_s}x{it}"] = r
        print(f"[bench] cruise planner {n_s}x{it}: median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms")
    mpc.cfg.n_samples, mpc.cfg.cem_iters = 128, 2

    # --- 2. pulse planner (fine positioning) ----------------------------------------------------
    try:
        from uwam.arm_kin import load as load_arm_kin  # noqa: E402
        from uwam.grasp_planner import GraspCfg, GraspPlanner  # noqa: E402

        gm, gdn, gpn, gcfg = _load_model(M / "grasp_core.pt", dev)
        arm_kin = load_arm_kin(str(M / "arm_kin.pt"), dev)
        gp = GraspPlanner(gm, gdn, gpn, arm_kin, GraspCfg(), device=dev)
        hs = (0.02 * rng.standard_normal((16, gcfg.model.dyn_state_dim))).astype(np.float32)
        hs[:, 6:11] = np.array([0.0, 0.32, 0.32, 0.33, 0.0], np.float32)     # armed joint pose slot (approx.)
        ha = (0.05 * rng.standard_normal((16, 13))).astype(np.float32)   # 5 joints + 8 thrusters
        ha[:, :5] = hs[0, 6:11]
        obj_body = np.array([0.30, 0.02, -0.10], np.float32)
        r = timeit(lambda: gp.plan_pulse(hs, ha, hs[-1], obj_body, 0.0, rpy=np.zeros(3, np.float32), omega=np.zeros(3, np.float32), z_above=None),
                   args.repeats, sync=sync)
        res["pulse_planner"] = r
        print(f"[bench] pulse planner: median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"[bench] pulse planner skipped: {type(e).__name__}: {e}")

    if not args.skip_vision:
        import percept.goal_head as gh  # noqa: E402
        enc = args.encoder or gh.ENCODER_DIR
        ego = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        wrist = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        jp = np.array([0.0, 0.32, 0.32, 0.33, 0.0], np.float32)
        # --- 3. image-goal head r4 ---------------------------------------------------------------
        try:
            pg = gh.PerceptGoalE2E(e2e_ckpt=str(M / "percept_e2e_r4.pt"), head_ckpt=str(M / "percept_head_base.pt"),
                                   encoder_dir=enc, device=dev)
            r = timeit(lambda: pg.predict_nav(ego, wrist, jp, 0.1, 2.5, "Go to the charging station", yaw=0.0), args.repeats, sync=sync)
            res["image_goal_head_r4"] = r
            print(f"[bench] image-goal head r4: median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"[bench] image-goal head skipped: {type(e).__name__}: {e}")
        # --- 4/5. wrist-camera head r2, container head r1 ----------------------------------------
        import percept.wrist_pose_model as wpm  # noqa: E402
        from percept.wrist_pose_model import WristPoseHead  # noqa: E402
        try:  # some embedded OpenCV builds reject numpy arrays (numpy ABI mismatch): fall back to PIL resize
            wpm.cv2.resize(np.zeros((8, 8, 3), np.uint8), (4, 4))
        except Exception:  # noqa: BLE001
            from PIL import Image  # noqa: PLC0415
            wpm.cv2.resize = lambda im, size, interpolation=None: np.asarray(Image.fromarray(im).resize(size, Image.BILINEAR))
            print("[bench] cv2.resize unusable here -> PIL resize (same 224x224 output)")
        for name, f in (("wrist_head_r2", "wrist_pose_r2.pt"), ("box_head_r1", "box_pose_r1.pt")):
            if not (M / f).exists():
                print(f"[bench] {name}: {f} not present, skipped"); continue
            try:
                wh = WristPoseHead(str(M / f), device=dev)
                r = timeit(lambda: wh.predict(wrist, ego, jp), args.repeats, sync=sync)
                res[name] = r
                print(f"[bench] {name}: median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms")
            except Exception as e:  # noqa: BLE001
                print(f"[bench] {name} skipped: {type(e).__name__}: {e}")

    # --- per-control-step totals (perception + planner), vs the 0.5 s replanning budget ----------
    tot = {}
    if "image_goal_head_r4" in res:
        tot["cruise_step_ms"] = res["image_goal_head_r4"]["median_ms"] + res["mpc_128x2"]["median_ms"]
    if "wrist_head_r2" in res and "pulse_planner" in res:
        tot["grasp_step_ms"] = res["wrist_head_r2"]["median_ms"] + res["pulse_planner"]["median_ms"]
    res["per_step"] = tot
    print("[bench] per control step:", {k: round(v, 1) for k, v in tot.items()}, "(budget 500 ms; VLA chunk 1600 ms)")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print("[bench] saved", args.out)


if __name__ == "__main__":
    main()
