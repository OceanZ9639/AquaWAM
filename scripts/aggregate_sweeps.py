#!/usr/bin/env python3
"""Aggregate the blackout-duration and fault-severity sweeps into one summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ARMS = ("mixer", "sysid", "rls", "wam")


def _arm_means(rows: list, key: str = "blind_track_err") -> dict:
    out = {}
    for m in ARMS:
        vals = [r[key] for r in rows if r["controller"] == m and r.get(key) is not None]
        if vals:
            out[m] = {"mean": round(float(np.mean(vals)), 4), "n": len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/hy-tmp/logs/uwam")
    ap.add_argument("--out", default="/hy-tmp/logs/uwam/sweeps_summary.json")
    args = ap.parse_args()
    logs = Path(args.logs)

    blackout = {}
    for p in sorted(logs.glob("closed_loop_bsweep_d*.json")):
        drop = int(re.search(r"_d(\d+)\.json", p.name).group(1))
        d = json.loads(p.read_text())
        blackout[str(drop)] = {
            "blind_seconds": 12.0 - drop + 0.0,
            "arms": _arm_means(d["trials"]),
            "gate": {m: v["n_ok"] for m, v in (d.get("blind") or {}).items()},
        }

    severity = {}
    p = logs / "closed_loop_fault_sweep.json"
    if p.exists():
        d = json.loads(p.read_text())
        for reg in sorted({r["regime"] for r in d["trials"]}):
            eta = int(re.search(r"_(\d+)$", reg).group(1)) / 100.0
            rows = [r for r in d["trials"] if r["regime"] == reg]
            severity[f"{eta:.1f}"] = _arm_means(rows)

    out = {
        "blackout_duration_sweep": {
            "protocol": "goal_step at 8s; DVL dropped at the listed scored second; "
                        "longer blind window = smaller drop value",
            "by_drop_s": blackout,
        },
        "fault_severity_sweep": {
            "protocol": "thruster-2 efficiency drops to the listed value at t=8s while blind from 6s",
            "by_eta": severity,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
