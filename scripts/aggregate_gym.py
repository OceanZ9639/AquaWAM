#!/usr/bin/env python3
"""Aggregate the cross-domain (gym) suite into one summary consumed by paper_table.

Per environment: headline matrix (regime x arm), severity/duration sweep curves,
fitted gate constants, and the boundary verdicts the theory predicts:
  nominal  -> who wins the stationary blackout (hold vs replan; sigma_e regime)
  midfault -> does gated replanning beat hold once the observable change > Delta*
  goalstep -> latched replanning must win everywhere
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RES = Path("/hy-tmp/results")
ENVS = ("pointmass", "pointmass_lo", "pointmass_hi", "reacher")
NEGATIVE = {
    "swimmer": "shooting CEM cannot discover coordinated strokes (documented negative)",
    "cheetah": "gait discovery via CEM on offline models fails; see gym_port_notes.md",
}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _verdicts(res: dict) -> dict:
    def e(k):
        r = res.get(k)
        return r["err_blind"] if r else None

    v = {}
    if e("nominal/open") is not None and e("nominal/replan") is not None:
        v["stationary_winner"] = "hold" if e("nominal/open") <= e("nominal/replan") else "replan"
        v["stationary_hold_minus_replan"] = round(e("nominal/open") - e("nominal/replan"), 4)
        if e("nominal/gated") is not None:
            v["gated_vs_best_stationary"] = round(
                e("nominal/gated") - min(e("nominal/open"), e("nominal/replan")), 4)
    if e("midfault/open") is not None and e("midfault/gated") is not None:
        v["midfault_gated_beats_hold"] = bool(e("midfault/gated") < e("midfault/open"))
        v["midfault_hold_minus_gated"] = round(e("midfault/open") - e("midfault/gated"), 4)
    if e("goalstep/open") is not None and e("goalstep/gated") is not None:
        v["goalstep_gated_beats_hold"] = bool(e("goalstep/gated") < e("goalstep/open"))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/hy-tmp/results/gym_summary.json")
    args = ap.parse_args()

    out = {"environments": {}, "negative_domains": NEGATIVE}
    for env in ENVS:
        head = _load(RES / f"gym_{env}_closed_loop.json")
        if head is None:
            continue
        entry = {
            "headline": head["results"],
            "goal": head.get("goal"),
            "goal2": head.get("goal2"),
            "eta": head.get("eta"),
            "verdicts": _verdicts(head["results"]),
        }
        gate = _load(Path(f"/hy-tmp/models/gym_{env}/gate_calib.json"))
        if gate:
            entry["gate"] = {k: gate.get(k) for k in
                             ("kappa", "h", "decay", "achieved_far", "decision_delay_s",
                              "decision_miss", "n_null", "n_shift")}
        sev = {}
        for f in sorted(RES.glob(f"gym_{env}_sev_e*.json")):
            d = json.loads(f.read_text())
            sev[str(d.get("eta"))] = {k: v["err_blind"] for k, v in d["results"].items()}
        if sev:
            entry["severity_sweep"] = sev
        dur = {}
        for f in sorted(RES.glob(f"gym_{env}_dur_d*.json")):
            d = json.loads(f.read_text())
            drop = f.stem.split("_d")[-1]
            dur[drop] = {k: v["err_blind"] for k, v in d["results"].items()}
        if dur:
            entry["duration_sweep"] = dur
        out["environments"][env] = entry

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({e: v.get("verdicts") for e, v in out["environments"].items()}, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
