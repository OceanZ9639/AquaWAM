#!/usr/bin/env python3
"""Build gripper-command labels aligned row-for-row with the extracted features.

The features npz appended episodes in sorted-parquet order with per-row episode
ids; replaying the same stride over the same ordering reproduces the alignment
exactly. Label = expert's commanded gripper joint (action[0]): 1 if closing.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CLOSE_THRESH = 0.007  # action[0] is 0 (open) or 0.015 (close)


def build(split: str, suffix: str = "_base"):
    feats = np.load(f"/hy-tmp/data/percept_probe/{split}_feats{suffix}.npz")
    ep_rows = {}
    for e in feats["episode"]:
        ep_rows[int(e)] = ep_rows.get(int(e), 0) + 1
    root = Path(f"/hy-tmp/data/usim/{split}")
    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    out = []
    for p in parquets:
        t = pq.read_table(p)
        ei = int(t.column("episode_index")[0].as_py())
        k = ep_rows.get(ei)
        if not k:
            continue
        act = np.array([np.asarray(x, np.float32) for x in t.column("action").to_pylist()])
        grip = (act[::3, 0] > CLOSE_THRESH).astype(np.float32)[:k]
        assert len(grip) == k, (p, len(grip), k)
        out.append(grip)
    arr = np.concatenate(out)
    assert len(arr) == len(feats["episode"]), (len(arr), len(feats["episode"]))
    np.save(f"/hy-tmp/data/percept_probe/{split}_grip{suffix}.npy", arr)
    print(f"{split}: {len(arr)} rows, close-rate {arr.mean():.3f}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["train", "test"]):
        build(s)
