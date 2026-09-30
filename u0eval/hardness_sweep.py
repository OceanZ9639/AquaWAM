#!/usr/bin/env python3
"""Harder judges, no re-runs: replay every archived locomotion episode through the OFFICIAL judge
rule (eval_tracking.check_points_reached: at each 2 Hz sample a not-yet-reached waypoint is reached
when pos_err <= pos_tol AND yaw_err <= yaw_tol; success = endpoint reached AND reached >= frac * N)
with tighter tolerances / higher coverage. The 2 Hz log (episode_<i>_data.csv) is written by the very
loop that judges, so the official setting reproduces results.csv exactly (checked and reported).

Because the episode ends the moment the official endpoint ball is entered, the endpoint is always
judged at the OFFICIAL tolerance; the sweep tightens the intermediate points only (where the
vehicle actually flew past) and the required coverage. goto has no intermediate judging in the
official protocol; its hard settings require every intermediate waypoint to be passed within t m
(segment-interpolated closest approach, as strict_rescore.py).

  python3 u0eval/hardness_sweep.py --percept-round r4 --out /hy-tmp/results/hardness
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strict_rescore import closest_approach, read_path, read_results  # noqa: E402

TASKS = {
    "goto": ["goto_charge_station", "goto_water_tower"],
    "scan": ["scan_ship_modern", "scan_ship_ancient"],
    "inspect": ["inspect_pipeline_pool", "inspect_pipeline_sea"],
}
OFFICIAL = {"scan": (5.0, 1.0, 0.50), "inspect": (2.0, 0.5, 0.75), "goto": (1.0, 3.14, None)}
# (label, pos_tol, yaw_tol, frac) -- the first row of each family is the official judge
SETTINGS = {
    "scan": [("official 5 m / 1.0 rad / 50 %", 5.0, 1.0, 0.50),
             ("coverage 75 %", 5.0, 1.0, 0.75),
             ("coverage 90 %", 5.0, 1.0, 0.90),
             ("coverage 100 %", 5.0, 1.0, 1.00),
             ("2.5 m ball", 2.5, 1.0, 0.50),
             ("2.5 m ball, 75 %", 2.5, 1.0, 0.75),
             ("1.0 m ball", 1.0, 1.0, 0.50),
             ("yaw 0.5 rad", 5.0, 0.5, 0.50),
             ("2.5 m, 0.5 rad, 75 %", 2.5, 0.5, 0.75)],
    "inspect": [("official 2 m / 0.5 rad / 75 %", 2.0, 0.5, 0.75),
                ("coverage 90 %", 2.0, 0.5, 0.90),
                ("coverage 100 %", 2.0, 0.5, 1.00),
                ("1.0 m ball", 1.0, 0.5, 0.75),
                ("1.0 m ball, 90 %", 1.0, 0.5, 0.90),
                ("0.5 m ball", 0.5, 0.5, 0.75),
                ("yaw 0.25 rad", 2.0, 0.25, 0.75),
                ("1.0 m, 0.25 rad, 90 %", 1.0, 0.25, 0.90)],
    "goto": [("official endpoint 1 m", None, None, None),
             ("+ every waypoint within 1.0 m", 1.0, None, None),
             ("+ every waypoint within 0.5 m", 0.5, None, None),
             ("+ every waypoint within 0.25 m", 0.25, None, None)],
}
DROP = {"goto": "drop8s_zero", "scan": "drop40s_zero", "inspect": "drop40s_zero"}


def ang(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def replay(P: np.ndarray, Y: np.ndarray, wp: np.ndarray, pos_tol: float, yaw_tol: float) -> np.ndarray:
    """Sticky reached flags exactly as the official loop computes them from the 2 Hz samples."""
    reached = np.zeros(len(wp), bool)
    for p, y in zip(P, Y):
        d = np.linalg.norm(wp[:, :3] - p, axis=1)
        e = ang(wp[:, 3], y)
        reached |= (d <= pos_tol) & (e <= yaw_tol)
        if reached[-1]:
            break  # the judge stops the episode on the endpoint
    return reached


def score_episode(fam: str, P, Y, wp, official_success: bool) -> dict[str, bool]:
    out = {}
    if fam == "goto":
        d, _ = closest_approach(P, Y, wp)
        inter = d[:-1] if len(wp) > 1 else np.zeros(0)
        for label, t, _, _ in SETTINGS[fam]:
            out[label] = bool(official_success and (t is None or (inter <= t).all()))
        return out
    pos0, yaw0, _ = OFFICIAL[fam]
    end_ok = bool(replay(P, Y, wp, pos0, yaw0)[-1])
    n = len(wp)
    for label, pos_tol, yaw_tol, frac in SETTINGS[fam]:
        r = replay(P, Y, wp, pos_tol, yaw_tol)
        cnt = int(r[:-1].sum()) + (1 if end_ok else 0)
        out[label] = bool(end_ok and cnt >= frac * n)
    return out


def score_block(block: Path, fam: str) -> tuple[dict[str, int], int, int]:
    """Returns per-setting success counts, n scored, and #episodes where the official replay
    disagrees with results.csv (sanity)."""
    res = read_results(block / "results.csv")
    counts = {label: 0 for label, *_ in SETTINGS[fam]}
    n = 0
    disagree = 0
    for f in sorted(block.glob("logs/episode_*_data.csv")):
        ep = int(re.search(r"episode_(\d+)_data", f.name).group(1))
        tf = block / "logs" / f"episode_{ep}_traj.npy"
        if ep not in res or not tf.exists():
            continue
        wp = np.asarray(np.load(tf, allow_pickle=True), np.float64)
        if wp.ndim != 2 or wp.shape[1] < 4 or len(wp) < 1:
            continue
        P, Y = read_path(f)
        if len(P) < 2:
            continue
        official = res[ep][0] == "success"
        s = score_episode(fam, P, Y, wp, official)
        first = SETTINGS[fam][0][0]
        if s[first] != official:
            disagree += 1
            s = {k: (v and official) for k, v in s.items()}  # never award a success the judge denied
            s[first] = official
        n += 1
        for k, v in s.items():
            counts[k] += int(v)
    return counts, n, disagree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--percept-round", default="r4")
    ap.add_argument("--out", default="/hy-tmp/results/hardness")
    args = ap.parse_args()
    root = Path(args.eval_runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = f"wam_percept_{args.percept_round}"
    arms = [("u0", "U0"), ("wam", "WAM (privileged)"), (vis, f"WAM (vision, {args.percept_round})"),
            ("fallback", "U0+WAM fallback")]

    table = {}   # fam -> cond -> arm_label -> {setting: (succ, n)}
    md = ["## Harder judges (offline replay of the official rule, no re-runs)", "",
          "Each cell is successes / episodes summed over the family's tasks. The first row of every family "
          "is the official judge and reproduces results.csv (U0 full sensing is our re-run here, the paper "
          "publishes no trajectories; its official-row numbers are in the Table IV above). The endpoint is "
          "always judged at the official tolerance because the episode ends on entering it; the tightened "
          "tolerances apply to the intermediate waypoints, the coverage to all waypoints.", ""]
    sanity = []
    for fam, tasks in TASKS.items():
        for cond_name, cond in (("Full sensing", "full"), ("DVL dropout", DROP[fam])):
            cells = {}
            for arm, label in arms:
                agg = {s[0]: [0, 0] for s in SETTINGS[fam]}
                for t in tasks:
                    block = root / f"{arm}_{cond}" / t
                    if not (block / "results.csv").exists():
                        continue
                    counts, n, dis = score_block(block, fam)
                    if n < 20:
                        continue
                    sanity.append((f"{arm}_{cond}/{t}", n, dis))
                    for k, v in counts.items():
                        agg[k][0] += v
                        agg[k][1] += n
                if any(v[1] for v in agg.values()):
                    cells[label] = agg
            if not cells:
                continue
            table.setdefault(fam, {})[cond_name] = cells
            md += [f"### {fam} — {cond_name}", "",
                   "| judge | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
            for label, *_ in SETTINGS[fam]:
                row = []
                for arm_label, agg in cells.items():
                    s, n = agg[label]
                    row.append(f"{s}/{n} ({100 * s / n:.0f} %)" if n else "—")
                md.append(f"| {label} | " + " | ".join(row) + " |")
            md.append("")
    bad = [(b, n, d) for b, n, d in sanity if d]
    md.append(f"*Sanity: official-setting replay agrees with results.csv on "
              f"{sum(n for _, n, _ in sanity) - sum(d for _, _, d in sanity)}/{sum(n for _, n, _ in sanity)} episodes"
              + (f"; disagreements ({len(bad)} blocks) are counted as the judge decided: "
                 + ", ".join(f"{b} {d}" for b, _, d in bad[:8]) if bad else "") + ".*")
    text = "\n".join(md)
    print(text)
    (out / "hardness_sweep.md").write_text(text + "\n")
    (out / "hardness_sweep.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
