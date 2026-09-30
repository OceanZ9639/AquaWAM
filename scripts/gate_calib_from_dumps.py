#!/usr/bin/env python3
"""Fit (kappa, h) from deployment-condition calibration dumps.

Input: npz files written by closed_loop.py --calib-dump (any domain, shared format):
  t [T], v_sm_n [T,D], anchor_n [T,D], sig_eff [T,D], dvl_true [T,D], change_at, t_start
Null streams come from no-change trials, shift streams from trials whose regime changed
at `change_at`. The z computation is definitionally identical to the deployed gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.gate import calibrate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="calib dump dirs")
    ap.add_argument("--far", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--vel-std", default="", help="comma floats to normalize dvl_true for "
                    "effect sizes; '' = use anchor units already normalized")
    ap.add_argument("--n-smooth", type=int, default=5)
    ap.add_argument("--hold-ref", default="", help="results json whose nominal/calib row gives "
                    "the stationary HOLD cost on the calibration seeds")
    ap.add_argument("--replan-ref", default="", help="results json whose nominal/replan row gives "
                    "the stationary REPLAN cost; together they pick alpha_base")
    ap.add_argument("--out", default="/hy-tmp/models/uwam/gate_calib.json")
    args = ap.parse_args()

    vel_std = (np.asarray([float(x) for x in args.vel_std.split(",")], np.float64)
               if args.vel_std else None)
    z_null, z_shift, sizes = [], [], []
    files = []
    for d in args.dumps:
        files += sorted(Path(d).glob("*.npz"))
    for f in files:
        d = np.load(f)
        t = d["t"]
        if len(t) < 20:
            continue
        z = (np.linalg.norm(d["v_sm_n"] - d["anchor_n"], axis=1)
             / np.maximum(np.linalg.norm(d["sig_eff"] + args.floor, axis=1), 1e-9))
        change_at = float(d["change_at"])
        if change_at < 0:
            z_null.append(z)
            # re-anchored sub-windows multiply the null set: same estimator stream,
            # anchor reset to the local smoothed estimate (bias cancels identically)
            v_sm, sig = d["v_sm_n"], d["sig_eff"]
            for s0 in range(30, len(t) - 40, 30):
                anc = v_sm[s0 - args.n_smooth:s0].mean(0)
                zz = (np.linalg.norm(v_sm[s0:] - anc, axis=1)
                      / np.maximum(np.linalg.norm(sig[s0:] + args.floor, axis=1), 1e-9))
                z_null.append(zz)
            continue
        tc = int(np.searchsorted(t, change_at))
        if tc < 3 or tc > len(t) - 10:
            # change outside (or too near the edge of) the blind window: treat pre-change
            # part as null if long enough
            if tc >= len(t) - 10 and tc > 40:
                z_null.append(z[:tc - 5])
            continue
        z_shift.append((z, tc))
        v = d["dvl_true"].astype(np.float64)
        if vel_std is not None:
            v = v / vel_std
        vs = np.stack([v[max(0, i - args.n_smooth + 1):i + 1].mean(0) for i in range(len(v))], 0)
        pre = vs[:tc].mean(0)
        sizes.append(float(np.linalg.norm(vs[tc:] - pre, axis=1).max()))

    print(f"streams: null={len(z_null)} shift={len(z_shift)}", flush=True)
    if not z_null or not z_shift:
        raise SystemExit("not enough dump streams")
    report = calibrate(z_null, z_shift, dt=args.dt, far_target=args.far, shift_sizes=sizes)
    report["source"] = "deployment-condition dumps"
    report["dump_dirs"] = args.dumps
    # stationary-optimal endpoint from the calibration trials themselves: whichever of
    # hold/replan is cheaper with nothing changing becomes the gate's alpha_base
    if args.hold_ref and args.replan_ref:
        hold_d = json.loads(Path(args.hold_ref).read_text())["results"]
        rep_d = json.loads(Path(args.replan_ref).read_text())["results"]
        hold_err = (hold_d.get("nominal/calib") or {}).get("err_blind")
        rep_err = (rep_d.get("nominal/replan") or {}).get("err_blind")
        if hold_err is not None and rep_err is not None:
            # hold is the conservative default; replanning becomes the base endpoint only
            # when it is CLEARLY stationary-better (20% margin beats seed noise on ties)
            report["alpha_base"] = 1.0 if rep_err < 0.8 * hold_err else 0.0
            report["stationary_costs"] = {"hold": hold_err, "replan": rep_err}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
