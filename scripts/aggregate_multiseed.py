#!/usr/bin/env python3
"""Aggregate the multi-seed repeats of the two discriminating experiments.

Reads closed_loop_fault_mid_s*.json / closed_loop_goalstep_s*.json, reports per-arm
mean +/- std across seeds, per-trial win counts, and the paired per-trial difference
(wam minus the best baseline on the same regime/goal/seed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("mixer", "sysid", "rls", "wam")


def trial_key(r: dict) -> tuple:
    return (r["regime"], tuple(round(float(x), 3) for x in r["goal"]))


def load_experiment(paths: list[Path]) -> dict:
    seeds = {}
    for p in paths:
        d = json.loads(p.read_text())
        seed = d.get("seed")
        rows = [r for r in d["trials"] if r.get("blind_track_err") is not None]
        seeds[str(seed)] = {"rows": rows, "recovery": d.get("change_point_recovery")}
    if not seeds:
        return {}

    per_seed_means = {m: [] for m in ARMS}
    per_seed_gate = {m: [] for m in ARMS}
    wins = {m: 0 for m in ARMS}
    paired = []
    for sd in seeds.values():
        rows = sd["rows"]
        by_arm = {m: {trial_key(r): r for r in rows if r["controller"] == m} for m in ARMS}
        for m in ARMS:
            errs = [r["blind_track_err"] for r in by_arm[m].values()]
            per_seed_means[m].append(float(np.mean(errs)))
            per_seed_gate[m].append(sum(bool(r["blind_ok"]) for r in by_arm[m].values()))
        for key in by_arm["wam"]:
            if key in by_arm["mixer"] and key in by_arm["sysid"]:
                vals = {m: by_arm[m][key]["blind_track_err"] for m in ARMS}
                wins[min(vals, key=lambda k: vals[k])] += 1
                paired.append(vals["wam"] - min(vals["mixer"], vals["sysid"]))
    paired = np.asarray(paired)
    n_trials_per_seed = len(next(iter(seeds.values()))["rows"]) // len(ARMS)
    rec = {}
    for m in ARMS:
        ints = [
            (sd["recovery"] or {}).get(m, {}).get("mean_err_integral")
            for sd in seeds.values()
            if sd.get("recovery")
        ]
        ints = [x for x in ints if x is not None]
        if ints:
            rec[m] = {"mean": round(float(np.mean(ints)), 4), "std": round(float(np.std(ints)), 4)}
    regimes = sorted({r["regime"] for sd in seeds.values() for r in sd["rows"]})
    per_regime = {}
    for reg in regimes:
        per_regime[reg] = {}
        for m in ARMS:
            errs = [r["blind_track_err"] for sd in seeds.values() for r in sd["rows"]
                    if r["controller"] == m and r["regime"] == reg]
            alphas = [r.get("blind_alpha_mean") for sd in seeds.values() for r in sd["rows"]
                      if r["controller"] == m and r["regime"] == reg
                      and r.get("blind_alpha_mean") is not None]
            per_regime[reg][m] = {
                "mean": round(float(np.mean(errs)), 4) if errs else None,
                "n": len(errs),
                "alpha_mean": round(float(np.mean(alphas)), 3) if alphas else None,
            }
    return {
        "n_seeds": len(seeds),
        "seeds": sorted(seeds),
        "trials_per_seed_per_arm": n_trials_per_seed,
        "blind_track": {
            m: {
                "mean": round(float(np.mean(per_seed_means[m])), 4),
                "std": round(float(np.std(per_seed_means[m])), 4),
                "per_seed": [round(x, 4) for x in per_seed_means[m]],
            }
            for m in ARMS
        },
        "gate_ok_per_seed": {m: per_seed_gate[m] for m in ARMS},
        "per_trial_wins": wins,
        "paired_wam_minus_best_baseline": {
            "mean": round(float(paired.mean()), 4),
            "std": round(float(paired.std()), 4),
            "frac_wam_better": round(float((paired < 0).mean()), 4),
            "n": int(paired.size),
        },
        "per_regime": per_regime,
        "post_change_err_integral": rec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/hy-tmp/logs/uwam")
    ap.add_argument("--out", default="/hy-tmp/logs/uwam/multiseed_summary.json")
    args = ap.parse_args()
    logs = Path(args.logs)
    out = {
        "fault_during_blackout": load_experiment(sorted(logs.glob("closed_loop_fault_mid_s*.json"))),
        "goal_step_blackout": load_experiment(sorted(logs.glob("closed_loop_goalstep_s*.json"))),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
