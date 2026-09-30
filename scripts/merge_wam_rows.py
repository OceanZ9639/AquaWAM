#!/usr/bin/env python3
"""Merge rerun wam-only trials into a full-arm closed-loop json (baselines untouched).

The mean-hold fix changes only the wam arm; the sweeps were rerun wam-only to save
simulator hours. This splices the fresh wam rows into the chain files by
(regime, goal) key and refreshes the per-arm summary blocks that aggregators read.
Originals are kept as *.premerge.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def trial_key(r):
    return (r["regime"], tuple(round(float(x), 3) for x in r["goal"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="full-arm json (updated in place)")
    ap.add_argument("--wam", required=True, help="wam-only rerun json")
    args = ap.parse_args()

    base_p, wam_p = Path(args.base), Path(args.wam)
    base = json.loads(base_p.read_text())
    wam = json.loads(wam_p.read_text())
    new_rows = {trial_key(r): r for r in wam["trials"] if r["controller"] == "wam"}

    kept, replaced = [], 0
    for r in base["trials"]:
        if r["controller"] == "wam" and trial_key(r) in new_rows:
            kept.append(new_rows[trial_key(r)])
            replaced += 1
        else:
            kept.append(r)
    base["trials"] = kept

    # refresh the summary blocks the aggregators consume
    for section, key in (("blind", "blind_track_err"), ("arms", "track_err")):
        if section not in base:
            continue
        for arm in base[section]:
            rows = [r for r in kept if r["controller"] == arm and r.get(key) is not None]
            if not rows:
                continue
            blk = base[section][arm]
            if isinstance(blk, dict):
                if "mean_blind_track" in blk:
                    blk["mean_blind_track"] = float(np.mean([r[key] for r in rows]))
                if "mean_track" in blk:
                    blk["mean_track"] = float(np.mean([r[key] for r in rows]))
                if "n_ok" in blk:
                    ok_key = "blind_ok" if section == "blind" else "ok"
                    blk["n_ok"] = sum(bool(r.get(ok_key)) for r in rows)

    backup = base_p.with_suffix(".premerge.json")
    if not backup.exists():
        backup.write_text(base_p.read_text())
    base_p.write_text(json.dumps(base, indent=2))
    print(f"{base_p.name}: replaced {replaced} wam rows (backup at {backup.name})")


if __name__ == "__main__":
    main()
