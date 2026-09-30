#!/usr/bin/env python3
"""Failure-mode breakdown of grasp blocks from the judge's 1 Hz logs (episode_i_data.csv:
dx, dy, dz, distance, gripper joint, gripper effort). For every non-success episode: best
distance reached, |dx| |dy| |dz| at that moment, max effort, and how many logged seconds the
POSITION test alone passed (d <= 4 cm, |dx| < 3.5 cm, |dy| < 1 cm) vs. position AND grip."""
import csv
import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
rows_out = []
for task in sorted(os.listdir(root)):
    rc = os.path.join(root, task, "results.csv")
    if not os.path.exists(rc):
        continue
    res = {int(r[0]): r[1] for r in csv.reader(open(rc)) if r and r[0].isdigit()}
    for f in sorted(glob.glob(f"{root}/{task}/logs/episode_*_data.csv")):
        ep = int(f.split("_")[-2])
        rows = [r for r in csv.reader(open(f))][1:]
        a = np.array([[float(x) for x in r[1:7]] for r in rows if len(r) >= 7 and r[1] not in ("", "inf")])
        if len(a) == 0:
            continue
        i = int(np.argmin(a[:, 3]))
        pos_ok = (a[:, 3] <= 0.04) & (np.abs(a[:, 0]) < 0.035) & (np.abs(a[:, 1]) < 0.01)
        eff_ok = a[:, 5] >= 0.08
        rows_out.append((task, ep, res.get(ep, "?"), a[i, 3], abs(a[i, 0]), abs(a[i, 1]), abs(a[i, 2]), a[:, 5].max(),
                         int(pos_ok.sum()), int((pos_ok & eff_ok).sum()), a[i, 4]))
hdr = ("task", "ep", "result", "min_d", "|dx|", "|dy|", "|dz|", "max_eff", "pos_ok_s", "both_s", "grip")
print("%-22s%3s %-8s%7s%6s%6s%6s%8s%9s%7s%6s" % hdr)
for r in rows_out:
    if r[2] != "success":
        print("%-22s%3d %-8s%7.3f%6.3f%6.3f%6.3f%8.3f%9d%7d%6.3f" % r)
fails = [r for r in rows_out if r[2] != "success"]
succ = [r for r in rows_out if r[2] == "success"]
print(f"\nepisodes {len(rows_out)}: success {len(succ)}, fail {len(fails)}")
if fails:
    print(f"  best distance <= 4 cm reached      : {sum(r[3] <= 0.04 for r in fails)}/{len(fails)}")
    print(f"  |dy| > 1 cm at best moment          : {sum(r[5] > 0.01 for r in fails)}/{len(fails)}")
    print(f"  |dx| > 3.5 cm at best moment        : {sum(r[4] > 0.035 for r in fails)}/{len(fails)}")
    print(f"  |dz| > 3 cm at best moment          : {sum(r[6] > 0.03 for r in fails)}/{len(fails)}")
    print(f"  never gripped (max effort < 0.08)   : {sum(r[7] < 0.08 for r in fails)}/{len(fails)}")
    print(f"  position test passed >= 1 s         : {sum(r[8] > 0 for r in fails)}/{len(fails)}")
    print(f"  position AND grip passed >= 1 s     : {sum(r[9] > 0 for r in fails)}/{len(fails)}")
    print(f"  median best distance                : {np.median([r[3] for r in fails]):.3f} m")
