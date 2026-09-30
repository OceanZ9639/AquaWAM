#!/usr/bin/env python3
"""USIM-Hard P4: what the dataset covers, per task, seen through the world model.

Per USIM task: demonstration volume, motion regime (speed quantiles, thruster saturation), how
predictable its dynamics are for the deployed core (offline dyn / estimator MAE from
eval_per_task.py), and the closed-loop success of U0 and WAM on the matching eval task (ours).
The point: a VLA's per-task score tracks demonstration volume and regime, a world model's
tracks physical predictability -- and the latter is nearly flat across tasks.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# USIM task_index -> eval task codes whose success we report
EVAL_TASKS = {
    0: ["goto_charge_station"], 8: ["goto_water_tower"],
    3: ["scan_ship_ancient", "scan_ship_modern"], 4: ["inspect_pipeline_pool", "inspect_pipeline_sea"],
    5: ["follow_boat"], 1: ["pick_pipe0_shallow", "pick_pipe1_shallow"],
    6: ["pick_red_shallow"], 2: ["pick_blue_shallow"], 7: ["transfer_red_shallow"],
}


def success(root: Path, arm: str, tasks: list[str]) -> str:
    ok = n = 0
    for t in tasks:
        p = root / f"{arm}_full" / t / "results.csv"
        if not p.exists():
            continue
        rows = [r for r in csv.reader(open(p)) if r and r[0].isdigit()]
        ok += sum(1 for r in rows if r[1] == "success")
        n += len(rows)
    return f"{100 * ok / n:.0f} % ({ok}/{n})" if n else "–"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usim", default="/hy-tmp/data/usim")
    ap.add_argument("--per-task", default="/hy-tmp/results/usim_hard/per_task.json")
    ap.add_argument("--core", default="best_scenes")
    ap.add_argument("--eval-runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--out", default="/hy-tmp/results/usim_hard/usim_diagnosis.md")
    args = ap.parse_args()
    root = Path(args.usim)
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(root / "train/meta/tasks.jsonl")}
    stats = {k: {"eps": 0, "frames": 0, "speed": [], "sat": 0, "pwm_n": 0} for k in tasks}
    for f in sorted(glob.glob(str(root / "train/data/**/*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=["task_index", "observation.state", "action"])
        ti = int(t.column("task_index")[0].as_py())
        st = np.array([np.asarray(x, np.float32) for x in t.column("observation.state").to_pylist()])
        ac = np.array([np.asarray(x, np.float32) for x in t.column("action").to_pylist()])
        s = stats[ti]
        s["eps"] += 1
        s["frames"] += len(st)
        # USIM state layout (meta/modality.json): dvl_v = columns 18:21
        s["speed"].append(np.linalg.norm(st[:, 18:21], axis=1))
        pwm = ac[:, -8:]
        s["sat"] += int((np.abs(pwm) > 0.95).sum())
        s["pwm_n"] += pwm.size
    per_task = json.loads(Path(args.per_task).read_text()).get(args.core, {}) if Path(args.per_task).exists() else {}
    runs = Path(args.eval_runs)
    lines = ["# USIM through the world model — per-task diagnosis", "",
             f"core = {args.core}; offline errors on the USIM test split (m/s); closed-loop success = our "
             "full-sensing blocks (official judges).", "",
             "| task | demos | hours | speed p50 / p90 (m/s) | PWM saturated | core dyn MAE | core est MAE | hover MAE | U0 SR | WAM SR (priv. goals) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(tasks):
        s = stats[k]
        sp = np.concatenate(s["speed"]) if s["speed"] else np.zeros(1)
        pt = per_task.get(tasks[k], {})
        lines.append(
            f"| {tasks[k]} | {s['eps']} | {s['frames'] / 36000:.1f} | {np.percentile(sp, 50):.2f} / {np.percentile(sp, 90):.2f} | "
            f"{100 * s['sat'] / max(1, s['pwm_n']):.1f} % | "
            f"{pt.get('dyn_mae', float('nan')):.4f} | {pt.get('est_mae', float('nan')):.4f} | {pt.get('hover_mae', float('nan')):.3f} | "
            f"{success(runs, 'u0', EVAL_TASKS.get(k, []))} | {success(runs, 'wam', EVAL_TASKS.get(k, []))} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
