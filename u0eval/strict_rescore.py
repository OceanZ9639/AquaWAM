#!/usr/bin/env python3
"""USIM-Hard P1: re-score finished eval blocks under stricter path-adherence criteria.

The official checker (eval_tracking.py) is lenient: goto ("navigation") only needs the endpoint
within 1.0 m at any yaw; scan needs 5.0 m / 1.0 rad on 50% of the points; inspection 2.0 m / 0.5 rad
on 75% of the points. It also terminates the episode the moment the endpoint ball is entered, so
the *endpoint* error can never be measured below the official tolerance from logs alone.

What the logs do allow (episode_<i>_data.csv, 2 Hz vehicle pose; episode_<i>_traj.npy, the
reference waypoints) is a path-adherence re-score:

  * closest approach to every *intermediate* waypoint (segment-interpolated), so "strict success at
    tol t" = official success AND every intermediate waypoint passed within t metres
  * optional yaw gate at the closest-approach sample
  * median cross-track error (XTE) to the reference polyline
  * path-length ratio (flown / reference polyline) for successful episodes

Nothing is re-run; this is a pure function of the archived logs, so it applies identically to all
arms (U0 / WAM / fallback / expert planner "collect_full").
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

TOLS = (2.0, 1.5, 1.0, 0.75, 0.5)
# official yaw tolerance per family (goto: yaw is not scored at all -> gate disabled)
YAW_TOL = {"goto": None, "scan": 1.0, "inspect": 0.5, "follow": 0.5}
ARM_RE = re.compile(r"^(u0|wam|wam_percept(?:_r\d+)?|fallback|collect)_(full|drop\d+s_(zero|freeze)|hard_[a-z0-9_]+)$")
FAMILY = {"goto": "goto", "scan": "scan", "inspect": "inspect", "follow": "follow"}


def read_results(p: Path) -> dict[int, tuple[str, float]]:
    out = {}
    with open(p) as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                out[int(row[0])] = (row[1], float(row[2]) if len(row) > 2 and row[2] else np.nan)
    return out


def read_path(p: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.reader(open(p)))
    if len(rows) < 3:
        return np.zeros((0, 3)), np.zeros(0)
    hdr = rows[0]
    ix = [hdr.index(c) for c in ("rov_x", "rov_y", "rov_z")]
    iy = hdr.index("rov_yaw")
    P, Y = [], []
    for r in rows[1:]:
        if len(r) <= max(ix + [iy]):
            continue
        try:
            P.append([float(r[i]) for i in ix])
            Y.append(float(r[iy]))
        except ValueError:
            continue
    return np.asarray(P, np.float64), np.asarray(Y, np.float64)


def seg_point_dist(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from point q to each segment a[k]->b[k]."""
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-12)
    t = np.clip(((q - a) * ab).sum(-1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(proj - q, axis=-1)


def closest_approach(P: np.ndarray, Y: np.ndarray, wp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-waypoint closest approach (segment-interpolated) and yaw error at the nearest sample."""
    if len(P) < 2:
        d = np.linalg.norm(P[:1] - wp[:, None, :3], axis=-1).min(1) if len(P) else np.full(len(wp), np.inf)
        return d, np.full(len(wp), np.inf)
    a, b = P[:-1], P[1:]
    d = np.empty(len(wp))
    yerr = np.empty(len(wp))
    for k, w in enumerate(wp):
        ds = seg_point_dist(a, b, w[:3])
        d[k] = ds.min()
        j = int(ds.argmin())
        # nearest sample of the two segment endpoints
        i = j if np.linalg.norm(P[j] - w[:3]) <= np.linalg.norm(P[j + 1] - w[:3]) else j + 1
        e = (Y[i] - w[3] + np.pi) % (2 * np.pi) - np.pi
        yerr[k] = abs(e)
    return d, yerr


def cross_track(P: np.ndarray, wp: np.ndarray) -> np.ndarray:
    """Distance of every logged pose to the reference polyline (start pose -> wp_1 -> ... -> wp_M)."""
    poly = np.vstack([P[:1], wp[:, :3]])
    a, b = poly[:-1], poly[1:]
    return np.array([seg_point_dist(a, b, q).min() for q in P])


def polyline_len(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum())


def score_block(block: Path, results: dict[int, tuple[str, float]]) -> list[dict]:
    rows = []
    for f in sorted(block.glob("logs/episode_*_data.csv")):
        ep = int(re.search(r"episode_(\d+)_data", f.name).group(1))
        if ep not in results:
            continue
        tf = block / "logs" / f"episode_{ep}_traj.npy"
        if not tf.exists():
            continue
        wp = np.asarray(np.load(tf, allow_pickle=True), np.float64)
        if wp.ndim != 2 or wp.shape[1] < 4 or len(wp) < 1:
            continue
        P, Y = read_path(f)
        if len(P) < 2:
            continue
        d, yerr = closest_approach(P, Y, wp)
        inter = slice(0, len(wp) - 1)  # intermediate waypoints; endpoint is protocol-capped
        xte = cross_track(P, wp)
        official, dur = results[ep]
        rows.append({
            "episode": ep,
            "official": official == "success",
            "duration": dur,
            "n_wp": int(len(wp)),
            "inter_max_d": float(d[inter].max()) if len(wp) > 1 else 0.0,
            "inter_max_yaw": float(yerr[inter].max()) if len(wp) > 1 else 0.0,
            "end_d": float(d[-1]),
            "xte_med": float(np.median(xte)),
            "xte_p90": float(np.percentile(xte, 90)),
            "path_ratio": float(polyline_len(P) / max(polyline_len(np.vstack([P[:1], wp[:, :3]])), 1e-6)),
        })
    return rows


def summarize(rows: list[dict], yaw_tol: float | None) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    off = np.array([r["official"] for r in rows])
    dmax = np.array([r["inter_max_d"] for r in rows])
    ymax = np.array([r["inter_max_yaw"] for r in rows])
    yaw_ok = np.ones(n, bool) if yaw_tol is None else (ymax <= yaw_tol)
    out = {"n": n, "official": int(off.sum())}
    for t in TOLS:
        out[f"strict@{t}"] = int((off & (dmax <= t)).sum())
        out[f"strict_yaw@{t}"] = int((off & (dmax <= t) & yaw_ok).sum()) if yaw_tol is not None else -1
    out["end_d_med"] = float(np.median([r["end_d"] for r in rows]))
    out["xte_med"] = float(np.median([r["xte_med"] for r in rows]))
    ok = [r for r in rows if r["official"]]
    out["path_ratio_med"] = float(np.median([r["path_ratio"] for r in ok])) if ok else float("nan")
    out["dur_med"] = float(np.median([r["duration"] for r in ok])) if ok else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--out", default="/hy-tmp/results/usim_hard")
    ap.add_argument("--min-n", type=int, default=20, help="skip blocks with fewer scored episodes")
    ap.add_argument("--include-expert", action="store_true", help="also score collect_full (expert planner)")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table: dict[str, dict[str, dict]] = {}
    per_ep = []
    for block in sorted(root.glob("*/*")):
        arm = block.parent.name
        task = block.name
        if not ARM_RE.match(arm) or arm.startswith("_"):
            continue
        if arm.startswith("collect") and not args.include_expert:
            continue
        fam = next((v for k, v in FAMILY.items() if task.startswith(k)), None)
        if fam is None or not (block / "results.csv").exists():
            continue
        rows = score_block(block, read_results(block / "results.csv"))
        if len(rows) < (1 if arm.startswith("collect") else args.min_n):
            continue
        s = summarize(rows, YAW_TOL[fam])
        table.setdefault(task, {})[arm] = s
        for r in rows:
            per_ep.append({"arm": arm, "task": task, **r})

    with open(out / "strict_rescore_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys()) if per_ep else ["arm"])
        w.writeheader()
        w.writerows(per_ep)
    (out / "strict_rescore.json").write_text(json.dumps(table, indent=2))

    lines = ["# USIM-Hard · strictness re-score (path adherence, no re-runs)", "",
             "strict@t = official success AND every intermediate waypoint passed within t m "
             "(segment-interpolated closest approach). +yaw additionally requires the family's official "
             "yaw tolerance (scan 1.0 rad, inspect/follow 0.5 rad; goto is not yaw-scored) at the closest "
             "approach. end d = median closest approach to the endpoint (protocol-capped: the episode ends "
             "on entering the official ball, 1.0 m goto / 5.0 m scan / 2.0 m inspect). "
             "XTE = median cross-track error to the reference polyline; path ratio = flown / reference length.",
             ""]
    for task in sorted(table):
        lines += [f"## {task}", "",
                  "| arm | n | official | " + " | ".join(f"strict@{t}" for t in TOLS) + " | " +
                  " | ".join(f"+yaw@{t}" for t in (2.0, 1.0)) + " | end d (m) | XTE med (m) | path ratio | dur (s) |",
                  "|---|---|---|" + "---|" * (len(TOLS) + 2) + "---|---|---|---|"]
        for arm in sorted(table[task]):
            s = table[task][arm]
            yaw = " | ".join("n/a" if s[f'strict_yaw@{t}'] < 0 else str(s[f'strict_yaw@{t}']) for t in (2.0, 1.0))
            lines.append(
                f"| {arm} | {s['n']} | {s['official']} | " +
                " | ".join(str(s[f'strict@{t}']) for t in TOLS) + f" | {yaw}" +
                f" | {s['end_d_med']:.2f} | {s['xte_med']:.2f} | {s['path_ratio_med']:.2f} | {s['dur_med']:.0f} |")
        lines.append("")
    (out / "strict_rescore.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
