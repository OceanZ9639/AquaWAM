#!/usr/bin/env python3
"""Merge a top-up block into a protocol block, renumbering the new episodes after the existing ones.

    topup_merge.py <eval_runs> <arm_cond> [task ...]

For every task under <eval_runs>/<arm_cond>_topup/ (or only the tasks given), the rows of its
results.csv and its logs/episode_<i>_* files are appended to <eval_runs>/<arm_cond>/<task>/ with
episode index i -> i + offset, offset = 1 + the highest existing episode index. The official mapper
draws every episode's spawn unseeded (random.seed(None)), so the appended trials are fresh draws of
the same protocol. The consumed top-up dir is moved to <eval_runs>/_merged_topup/ so a second call
cannot merge it twice; results.csv of the target is backed up once as results.csv.pre_topup.
"""
import csv
import shutil
import sys
import time
from pathlib import Path


def rows_of(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.reader(f) if r and r[0] != "episode"]


def merge_task(src: Path, dst: Path) -> tuple[int, int]:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "logs").mkdir(exist_ok=True)
    old = rows_of(dst / "results.csv")
    seen: dict[str, list[str]] = {}
    for r in old:                       # keep the first row of a duplicated episode id
        seen.setdefault(r[0], r)
    old = list(seen.values())
    offset = 1 + max((int(r[0]) for r in old if r[0].isdigit()), default=-1)
    new = rows_of(src / "results.csv")
    new_seen: dict[str, list[str]] = {}
    for r in new:
        new_seen.setdefault(r[0], r)
    new = list(new_seen.values())
    if not new:
        return len(old), 0
    if not (dst / "results.csv.pre_topup").exists() and (dst / "results.csv").exists():
        shutil.copy2(dst / "results.csv", dst / "results.csv.pre_topup")
    appended = []
    for r in new:
        i = int(r[0])
        j = i + offset
        for f in (src / "logs").glob(f"episode_{i}_*"):
            suffix = f.name[len(f"episode_{i}_"):]
            shutil.copy2(f, dst / "logs" / f"episode_{j}_{suffix}")
        f = src / "logs" / f"episode_{i}.log"
        if f.exists():
            shutil.copy2(f, dst / "logs" / f"episode_{j}.log")
        appended.append([str(j)] + r[1:])
    with open(dst / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "result"])
        for r in old + appended:
            w.writerow(r)
    return len(old), len(appended)


def main() -> None:
    eval_runs, arm_cond = Path(sys.argv[1]), sys.argv[2]
    only = set(sys.argv[3:])
    src_root = eval_runs / f"{arm_cond}_topup"
    if not src_root.exists():
        print(f"[topup] nothing to merge: {src_root} does not exist")
        return
    done = eval_runs / "_merged_topup" / f"{arm_cond}_topup__{time.strftime('%m%d_%H%M%S')}"
    for task in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if only and task.name not in only:
            continue
        if not (task / "results.csv").exists():
            print(f"[topup] {task.name}: no results.csv, skipped")
            continue
        n_old, n_new = merge_task(task, eval_runs / arm_cond / task.name)
        print(f"[topup] {arm_cond}/{task.name}: {n_old} + {n_new} = {n_old + n_new} episodes")
        done.mkdir(parents=True, exist_ok=True)
        shutil.move(str(task), str(done / task.name))
    if src_root.exists() and not any(src_root.iterdir()):
        src_root.rmdir()


if __name__ == "__main__":
    main()
