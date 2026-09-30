#!/usr/bin/env python3
"""Like-for-like imagination cost of the two instances of the world model: 128 candidate action
sequences, four chained 0.5 s chunks (2 s), disturbance token from a 16-step history, exactly the
rollout path the planners use (GraspPlanner._score_seq / SamplingMPC). Reports the vehicle-scale
instance (19-d state, 8 commands) and the full-scale instance (35-d, 13 commands).
  python3 u0eval/bench_imagine128.py --device cuda --models /hy-tmp/models/uwam --out results/imagine128.json
"""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def timeit(fn, repeats, sync):
    fn()
    if sync:
        torch.cuda.synchronize()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn()
        if sync:
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean()), "n": int(repeats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="/hy-tmp/models/uwam")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = args.device
    sync = dev.startswith("cuda")
    from uwam.config import Cfg, enable_arm, enable_object  # noqa: E402
    from uwam.data import RunningNorm  # noqa: E402
    from uwam.models import DynamicsWAM  # noqa: E402

    def load(path):
        cfg = Cfg(); cfg.model.use_language = False
        ck = torch.load(path, map_location=dev, weights_only=False)
        c = ck.get("cfg", {})
        if "disturbance_dim" in c: cfg.model.disturbance_dim = int(c["disturbance_dim"])
        if "hidden" in c: cfg.model.hidden = int(c["hidden"])
        cfg.model.use_dt = bool(c.get("use_dt", False))
        if c.get("use_object"): enable_object(cfg)
        elif c.get("use_arm"): enable_arm(cfg)
        m = DynamicsWAM(cfg).to(dev); m.load_state_dict(ck["model"], strict=False); m.eval()
        dn, pn = RunningNorm(), RunningNorm(); dn.load_state_dict(ck["dyn_norm"]); pn.load_state_dict(ck["pwm_norm"])
        return m, cfg

    res = {"info": {"device": dev, "torch": torch.__version__, "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "n_candidates": args.n, "chunks": args.chunks}}
    try:
        res["info"]["tegra"] = open("/etc/nv_tegra_release").readline().strip()
    except OSError:
        pass
    rng = np.random.default_rng(0)
    for tag, fname in (("vehicle_scale", "best_scenes.pt"), ("full_scale", "grasp_core.pt")):
        model, cfg = load(Path(args.models) / fname)
        D, A, K = cfg.model.dyn_state_dim, cfg.model.pwm_dim, model.K
        n = args.n
        hs = torch.as_tensor(0.05 * rng.standard_normal((n, 16, D)), dtype=torch.float32, device=dev)
        ha = torch.as_tensor(0.2 * rng.standard_normal((n, 16, A)), dtype=torch.float32, device=dev)
        st = hs[:, -1, :].clone()
        af = torch.as_tensor(0.3 * rng.standard_normal((n, K * args.chunks, A)), dtype=torch.float32, device=dev)

        @torch.no_grad()
        def imagine():
            d = model.disturbance(hs, ha)
            cur, segs = st, []
            for ci in range(args.chunks):
                s_hat, _, _ = model.rollout(cur, af[:, ci * K:(ci + 1) * K], d)
                segs.append(s_hat); cur = s_hat[:, -1, :]
            return torch.cat(segs, 1)

        r = timeit(imagine, args.repeats, sync)
        r.update(params_M=sum(p.numel() for p in model.parameters()) / 1e6, state_dim=D, action_dim=A, K=K)
        res[tag] = r
        print(f"[imagine128] {tag}: {r['params_M']:.2f} M params, {D}-d state, {A} commands, "
              f"{n} candidates x {K * args.chunks} steps: median {r['median_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        print("[imagine128] wrote", args.out)


if __name__ == "__main__":
    main()
