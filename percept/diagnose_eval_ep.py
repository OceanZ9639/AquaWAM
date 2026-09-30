#!/usr/bin/env python3
"""Replay a recorded closed-loop eval episode through the perception head.

Compares, tick by tick: predicted goal magnitude / gripper probability vs the
judge's ground-truth gripper-object distance. Separates "head predicts wrong
at eval states" from "controller failed to act on good predictions".
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from percept.goal_head import PerceptGoal  # noqa: E402

EP_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/hy-tmp/u0env/dataset/eval_runs/wam_percept_full/pick_pipe0_shallow/episode0")
LOG_CSV = EP_DIR.parent / "logs" / f"episode_{EP_DIR.name.replace('episode','')}_data.csv"
STRIDE = 20  # 2 s

pg = PerceptGoal()
pg.ema_alpha = 1.0  # raw predictions for diagnosis

pkls = sorted(EP_DIR.glob("*.pkl"), key=lambda p: int(p.stem))
rows = []
for p in pkls[::STRIDE]:
    d = pickle.load(open(p, "rb"))
    img_r = EP_DIR / "images" / "right" / f"{p.stem}.jpg"
    img_h = EP_DIR / "images" / "hand" / f"{p.stem}.jpg"
    if not img_r.exists() or not img_h.exists():
        continue
    ego = cv2.cvtColor(cv2.imread(str(img_r)), cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(cv2.imread(str(img_h)), cv2.COLOR_BGR2RGB)
    st = d["observation"]["state"]
    js = st.get("joint_states") if isinstance(st, dict) else None
    jp = np.asarray(js["position"][:5], np.float32) if js else np.zeros(5, np.float32)
    pr = (st.get("pressure") or {}).get("fluid_pressure", 0.0) / 1e4 if isinstance(st, dict) else 0.0
    al = (st.get("dvl") or {}).get("altitude", 0.0) if isinstance(st, dict) else 0.0
    pg.reset()
    g, gp = pg.predict(ego, wrist, jp, float(pr), float(al), d.get("instruction") or "Pick up the pipe")
    rows.append((int(p.stem), float(np.linalg.norm(g[:3])), float(g[0]), float(g[1]), float(g[2]),
                 float(g[5]), gp))

judge = []
if LOG_CSV.exists():
    with open(LOG_CSV) as f:
        for i, r in enumerate(csv.DictReader(f)):
            judge.append(float(r["distance"]))

print(f"{'tick':>6} {'|g|':>7} {'gx':>7} {'gy':>7} {'gz':>7} {'gyaw':>7} {'grip_p':>7} {'judge_d':>8}")
for tick, mag, gx, gy, gz, gyaw, gp in rows:
    jd = judge[min(tick // 10, len(judge) - 1)] if judge else float("nan")
    print(f"{tick:6d} {mag:7.3f} {gx:7.3f} {gy:7.3f} {gz:7.3f} {gyaw:7.3f} {gp:7.3f} {jd:8.3f}")
print(json.dumps({
    "n": len(rows),
    "mean_|g|": float(np.mean([r[1] for r in rows])),
    "max_grip_p": float(max(r[6] for r in rows)),
    "judge_min_d": min(judge) if judge else None,
    "judge_final_d": judge[-1] if judge else None,
}))
