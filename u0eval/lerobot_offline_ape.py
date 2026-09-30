#!/usr/bin/env python3
"""Offline action-prediction error (APE) of a LeRobot-family checkpoint on USIM episodes, computed the
way the U0 repository's eval_policy does it (gr00t/utils/eval.py: every 16 steps of the first 300, a
16-step chunk is predicted; MSE over the unnormalized [joint_pos(5), pwm(8)] actions), so the number
is comparable to USIM's Table III (fine-tuned pi0.5 0.0861, GR00T N1.5 0.0374, U0 0.0359; zero-shot
pi0.5 0.1496, GR00T 0.1834). The policy sees exactly what our HTTP server feeds it (same Policy class).

    lerobot_offline_ape.py --ckpt <pretrained_model dir> --tag <name> [--per-task 4] [--root <lerobot ds>]

Runs in the lerobot env. Writes /hy-tmp/baselines/offline/ape_<tag>.json.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/hy-tmp/underwater_wam/u0eval")
from lerobot_policy_server import CHUNK, STATE_DIMS, STATE_ORDER, Policy  # noqa: E402

FAMILY = {"goto": "navigation", "scan": "navigation", "inspect": "navigation", "pick": "grasping",
          "transfer": "transporting", "follow": "tracking"}


def family_of(task: str) -> str:
    t = task.lower()
    for k, v in FAMILY.items():
        if k in t:
            return v
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="/hy-tmp/data/usim_v3/usim_train")
    ap.add_argument("--per-task", type=int, default=4, help="episodes per task index (last ones of the split)")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    fps = 10
    ds = LeRobotDataset(os.path.basename(args.root), root=args.root,
                        delta_timestamps={"action": [i / fps for i in range(CHUNK)]})
    meta = ds.meta
    # episode table: v3 keeps per-episode from/to indices and the task list in meta.episodes
    ep = meta.episodes
    n_ep = len(ep)
    cols = ep.column_names if hasattr(ep, "column_names") else list(ep.keys())
    def col(name):
        return [ep[i][name] for i in range(n_ep)] if hasattr(ep, "__getitem__") and not isinstance(ep, dict) else list(ep[name])
    from_idx = col("dataset_from_index"); to_idx = col("dataset_to_index")
    tasks_col = col("tasks") if "tasks" in cols else [None] * n_ep
    by_task = defaultdict(list)
    for e in range(n_ep):
        t = tasks_col[e]
        t = t[0] if isinstance(t, (list, tuple)) and t else (t if isinstance(t, str) else str(ds[from_idx[e]].get("task", "")))
        by_task[t].append(e)
    chosen = []
    for t, eps in sorted(by_task.items()):
        chosen += eps[-args.per_task:]          # the last episodes of each task in the split
    print(f"[ape] {len(chosen)} episodes over {len(by_task)} tasks; ckpt {args.ckpt}", flush=True)

    pol = Policy(args.ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    se = defaultdict(lambda: [0.0, 0]); se_task = defaultdict(lambda: [0.0, 0]); lat = []
    for e in chosen:
        f0, f1 = from_idx[e], to_idx[e]
        length = min(f1 - f0, args.steps)
        for s in range(0, length, CHUNK):
            item = ds[f0 + s]
            task = item.get("task", "") if isinstance(item.get("task", ""), str) else str(item.get("task", ""))
            gt = item["action"].numpy() if torch.is_tensor(item["action"]) else np.asarray(item["action"])
            gt = np.atleast_2d(gt)[:CHUNK]                      # [16, 13] (padded past the episode end)
            pad = item.get("action_is_pad")
            valid = (~pad.numpy()) if pad is not None else np.ones(len(gt), bool)
            valid &= (s + np.arange(len(gt))) < (f1 - f0)
            if not valid.any():
                continue
            obs = {}
            for cam, key in (("video.ego", "observation.images.ego"), ("video.wrist", "observation.images.wrist")):
                im = item[key]
                im = (im.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8) if torch.is_tensor(im) else np.asarray(im)
                obs[cam] = im
            st = item["observation.state"].numpy() if torch.is_tensor(item["observation.state"]) else np.asarray(item["observation.state"])
            o = 0
            for k, d in zip(STATE_ORDER, STATE_DIMS):
                obs[f"state.{k}"] = st[o:o + d].astype(np.float32); o += d
            obs["annotation.human.action.task_description"] = [task]
            t0 = time.time()
            out = pol.act(obs)[0]
            lat.append(time.time() - t0)
            pred = np.concatenate([out["action.joint_pos"], out["action.pwm"]], axis=1)[:len(gt)]   # [16, 13]
            err = ((pred[valid] - gt[valid]) ** 2)
            fam = family_of(task)
            se[fam][0] += float(err.sum()); se[fam][1] += int(err.size)
            se["overall"][0] += float(err.sum()); se["overall"][1] += int(err.size)
            se_task[task][0] += float(err.sum()); se_task[task][1] += int(err.size)
    res = {"ckpt": args.ckpt, "episodes": len(chosen), "per_task_episodes": args.per_task,
           "ape": {k: v[0] / v[1] for k, v in se.items() if v[1]},
           "ape_per_task": {k: v[0] / v[1] for k, v in se_task.items() if v[1]},
           "latency_ms_median": float(np.median(lat) * 1e3) if lat else None}
    Path("/hy-tmp/baselines/offline").mkdir(parents=True, exist_ok=True)
    Path(f"/hy-tmp/baselines/offline/ape_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[ape] {args.tag}: " + "  ".join(f"{k} {v:.4f}" for k, v in res["ape"].items()) +
          f"  | per-call {res['latency_ms_median']:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
