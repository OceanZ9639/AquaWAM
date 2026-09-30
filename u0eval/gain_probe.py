#!/usr/bin/env python3
"""
Open-loop sanity probe of every stage-1 core: imagined surge speed after 2 s of a constant surge
command from rest (hover history, level, 2.8 m depth), for command magnitudes 0.3 / 0.5 / 0.8.

This is the regime the sampling MPC lives in (it accelerates from hover with mid-range commands),
and it is NOT the regime of the USIM demonstrations (expert always at cruise, 44 % saturated
commands). Ground truth from the OU exploration recordings: ~0.30 m/s per unit surge command
after 2 s (steady-state plant gain from the axis probe: 0.5 m/s per unit).

Cores trained only on the repeated demos predict the wrong sign or 2 m/s; this is why de_f50 /
de_f100 / ho_nav crawl in closed loop (0.07-0.18 m/s) although their offline DVL MAE on the demo
test split is the best of all. Writes u0eval/gain_probe.json (read by ablation_tables.py and
write_results_md.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.control import sixdof_thrust_to_pwm  # noqa: E402
from uwam.data import load_ou_split  # noqa: E402
from uwam.direct import load_core  # noqa: E402

MODELS = Path("/hy-tmp/models/uwam")
CORES = ["de_f05", "de_f10", "de_f25", "de_f50", "de_f100", "ho_nav", "ho_manip", "ho_wt", "ho_scan",
         "ou_only", "play_only", "best_scenes", "scale_small", "scale_large", "adapt_eta0"]
MAGS = (0.3, 0.5, 0.8)


def truth_slope() -> tuple[float, int]:
    eps = load_ou_split(Path("/hy-tmp/data/ou_explore"))
    ref = sixdof_thrust_to_pwm(np.array([1, 0, 0], np.float32))
    s = np.sign(ref[:4])
    xs, ys = [], []
    for e in eps:
        sc = (e.pwm[:, :4] * s).mean(1)
        v = e.dyn[:, 0]
        for t in range(20, len(sc) - 20, 10):
            c = sc[t:t + 20]
            if c.std() < 0.08 and abs(c.mean()) > 0.15:
                xs.append(c.mean()); ys.append(v[t + 19])
    return float(np.polyfit(xs, ys, 1)[0]), len(xs)


@torch.no_grad()
def probe(path: Path, dev: str) -> list[float]:
    m, dn, pn, cfg = load_core(path, dev)
    out = []
    for mag in MAGS:
        u = sixdof_thrust_to_pwm(np.array([mag, 0, 0], np.float32))
        hs = np.zeros((16, 19), np.float32); hs[:, 8] = -9.8; hs[:, 9] = 27916; hs[:, 10] = 2.4
        ha = np.zeros((16, 8), np.float32)
        f = lambda a: torch.as_tensor(a[None], device=dev)
        cur = f(dn(hs[-1].copy()))
        d = m.disturbance(f(dn(hs)), f(pn(ha)))
        af = f(pn(np.tile(u, (5, 1)).astype(np.float32)))
        for _ in range(4):
            s_hat, _, _ = m.rollout(cur, af, d)
            cur = s_hat[:, -1, :]
        out.append(float((cur[0].cpu().numpy() * dn.std + dn.mean)[0]))
    return out


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    slope, n = truth_slope()
    res = {"truth_slope_ms_per_unit_cmd_2s": slope, "truth_n_segments": n, "mags": list(MAGS), "cores": {}}
    print(f"truth: {slope:.3f} m/s per unit surge command after 2 s ({n} OU segments) -> "
          f"{' / '.join(f'{slope * m:.2f}' for m in MAGS)} m/s for cmd {MAGS}")
    for c in CORES:
        p = MODELS / f"{c}.pt"
        if not p.exists():
            continue
        v = probe(p, dev)
        res["cores"][c] = v
        print(f"  {c:12s} {' / '.join(f'{x:+.2f}' for x in v)} m/s")
    Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
