#!/usr/bin/env python3
"""Where do closes succeed? From the judge's 1 Hz logs (dx, dy, dz, distance, gripper, effort) of every
grasp episode under --roots: detect close events (gripper angle rising past 0.005), record the offsets at
that second and whether a grip (effort >= 0.08) appeared within the next 4 s, and whether the episode
succeeded. Prints the empirical success region and how the hand-set gate compares."""
import csv
import glob
import sys

import numpy as np

roots = sys.argv[1:] or ["/hy-tmp/u0env/dataset/eval_runs"]
ev = []
n_ep = n_close_ep = 0
for root in roots:
    for f in glob.glob(f"{root}/**/logs/episode_*_data.csv", recursive=True):
        if "/pick_" not in f and "/transfer_" not in f:
            continue
        rows = [r for r in csv.reader(open(f))][1:]
        try:
            a = np.array([[float(x) for x in r[1:7]] for r in rows if len(r) >= 7 and r[1] not in ("", "inf", "nan")])
        except ValueError:
            continue
        if len(a) < 5:
            continue
        n_ep += 1
        res_csv = f.split("/logs/")[0] + "/results.csv"
        ep = int(f.split("_")[-2])
        succ = None
        for r in csv.reader(open(res_csv)):
            if r and r[0].isdigit() and int(r[0]) == ep:
                succ = r[1].strip() == "success"
        grip = a[:, 4]
        closes = np.where((grip[1:] > 0.005) & (grip[:-1] <= 0.005))[0] + 1
        if len(closes):
            n_close_ep += 1
        for i in closes:
            after = a[i:i + 5, 5]
            ev.append((a[i, 0], a[i, 1], a[i, 2], a[i, 3], float(after.max() >= 0.08), float(bool(succ)), f))
E = np.array([e[:6] for e in ev], np.float64)
print(f"{n_ep} episodes with judge logs, {n_close_ep} with at least one close, {len(E)} close events")
if len(E) == 0:
    sys.exit()
dx, dy, dz, d, grip, succ = E.T
print(f"grip within 4 s after close: {grip.mean() * 100:.0f} %   episode success: {succ.mean() * 100:.0f} %")
print("\n|dy| bins (cm)     n   grip%   |  |dx| bins (cm)  n   grip%  |  dz bins (cm, ee-obj)  n  grip%")
for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10)):
    m = (np.abs(dy) >= lo / 100) & (np.abs(dy) < hi / 100)
    mx = (np.abs(dx) >= lo / 100) & (np.abs(dx) < hi / 100)
    mz = (dz >= -hi / 100) & (dz < -lo / 100)
    g = lambda mm: f"{grip[mm].mean() * 100:5.0f}" if mm.sum() else "    -"
    print(f"  [{lo:3.1f},{hi:4.1f})      {m.sum():3d}  {g(m)}    |  [{lo:3.1f},{hi:4.1f})   {mx.sum():3d}  {g(mx)}   |  -[{lo:3.1f},{hi:4.1f})      {mz.sum():3d} {g(mz)}")
gate = (np.abs(dx) < 0.030) & (np.abs(dy) < 0.007) & (d < 0.035)
print(f"\ncloses inside the hand-set gate (|dx|<3, |dy|<0.7, d<3.5 cm): {gate.sum()} -> grip {grip[gate].mean() * 100 if gate.sum() else 0:.0f} %, "
      f"episode success {succ[gate].mean() * 100 if gate.sum() else 0:.0f} %")
print(f"closes outside the gate: {(~gate).sum()} -> grip {grip[~gate].mean() * 100 if (~gate).sum() else 0:.0f} %")
# best simple region by grid search on |dx|, |dy|, dz range
best = None
for tx in (0.02, 0.03, 0.035, 0.05):
    for ty in (0.005, 0.007, 0.01, 0.015, 0.02):
        for zlo, zhi in ((-0.04, 0.0), (-0.03, 0.0), (-0.03, 0.01), (-0.05, 0.02)):
            m = (np.abs(dx) < tx) & (np.abs(dy) < ty) & (dz > zlo) & (dz < zhi)
            if m.sum() >= 8:
                sc = grip[m].mean()
                if best is None or sc > best[0]:
                    best = (sc, tx, ty, zlo, zhi, int(m.sum()))
if best:
    print(f"best region (>=8 events): |dx|<{best[1]*100:.1f} |dy|<{best[2]*100:.1f} dz in [{best[3]*100:.0f},{best[4]*100:.0f}] cm -> grip {best[0]*100:.0f} % on {best[5]} closes")
grip_yes = E[grip > 0.5]; grip_no = E[grip <= 0.5]
print(f"\nmedian |dx|,|dy|,dz at close: gripped {np.median(np.abs(grip_yes[:,0]))*100:.1f},{np.median(np.abs(grip_yes[:,1]))*100:.1f},{np.median(grip_yes[:,2])*100:+.1f} cm ; "
      f"not gripped {np.median(np.abs(grip_no[:,0]))*100:.1f},{np.median(np.abs(grip_no[:,1]))*100:.1f},{np.median(grip_no[:,2])*100:+.1f} cm")
print(f"episodes with a grip but NO success: {int(((grip > 0.5) & (succ < 0.5)).sum())} of {int((grip > 0.5).sum())} gripped closes")
