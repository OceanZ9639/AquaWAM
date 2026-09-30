#!/usr/bin/env python3
"""Is the ocean current actually applied, and does the DVL see it?

If the DVL is water-relative, a uniform current is invisible to any DVL-based controller and the
"current" regimes are dynamically indistinguishable from nominal. That changes what the flow probe
on the disturbance token can possibly mean, so it has to be measured rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwam.rosbridge import SimBridge


def drift(bridge: SimBridge, current, seconds: float) -> dict:
    """Zero thrust, fixed current: compare DVL integral against actual position change."""
    bridge.home()
    t0 = None
    p0 = None
    rows = []
    while True:
        st = bridge.tick()
        if st is None:
            break
        if t0 is None:
            t0, p0 = st["stamp"], st["pos"].copy()
        t = st["stamp"] - t0
        bridge.publish_current(current)
        bridge.publish_pwm(np.zeros(8, np.float32))
        rows.append((t, st["dvl"].copy(), st["pos"].copy()))
        if t >= seconds:
            break
    if len(rows) < 5:
        return {"error": "no data"}
    dvl = np.stack([r[1] for r in rows])
    dpos = rows[-1][2] - p0
    return {
        "current_cmd": [float(x) for x in np.asarray(current, np.float64)],
        "dvl_mean": [round(float(x), 4) for x in dvl.mean(0)],
        "dvl_integral_m": [round(float(x), 3) for x in dvl.mean(0) * rows[-1][0]],
        "actual_dpos_m": [round(float(x), 3) for x in dpos],
        "seconds": round(rows[-1][0], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="/hy-tmp/logs/uwam/current_probe.json")
    args = ap.parse_args()
    bridge = SimBridge(node="uwam_probe_current")
    if not bridge.wait_ready():
        raise SystemExit("no DVL/odometry from the simulator")
    report = {}
    for name, cur in (("zero", (0.0, 0.0, 0.0)),
                      ("x_0.3", (0.3, 0.0, 0.0)),
                      ("x_0.6", (0.6, 0.0, 0.0)),
                      ("y_0.4", (0.0, 0.4, 0.0))):
        report[name] = drift(bridge, cur, args.seconds)
        print(name, json.dumps(report[name]), flush=True)
    bridge.publish_current((0.0, 0.0, 0.0))
    bridge.publish_pwm(np.zeros(8, np.float32))
    zero = report.get("zero", {}).get("actual_dpos_m") or [0, 0, 0]
    x06 = report.get("x_0.6", {}).get("actual_dpos_m") or [0, 0, 0]
    report["verdict"] = {
        "current_moves_vehicle": bool(abs(x06[0] - zero[0]) > 0.5),
        "dvl_sees_current": bool(abs((report.get("x_0.6", {}).get("dvl_mean") or [0])[0]) > 0.05),
    }
    print(json.dumps(report["verdict"], indent=2), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
