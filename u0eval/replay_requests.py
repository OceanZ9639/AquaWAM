#!/usr/bin/env python3
"""Replay recorded /act requests (WAM_DUMP_REQ dumps of wam_policy_server.py) against a running policy
server and time every call end to end (HTTP + decode + perception + imagination + encode). Run on the
same machine as the server so the number is the on-device per-step cost with real inputs.

    usage: replay_requests.py <dump dir> <server url> <out.json> [--pace 0.5]

--pace replays at the deployed control period (0.5 s between requests) so caching / thermal behaviour
matches deployment; 0 = as fast as possible. Reports median / p95 / max wall time per call and, when
the dump's server_act_ms.csv is present, the same statistics for the machine the dump was recorded on."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--pace", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    files = sorted(Path(args.dump).glob("req_*.json"))
    if not files:
        raise SystemExit(f"no req_*.json in {args.dump}")
    # warm-up on the first requests (model load / cudnn autotune), then time everything
    for f in files[:args.warmup]:
        urllib.request.urlopen(urllib.request.Request(args.url, data=f.read_bytes(),
                               headers={"Content-Type": "application/json"}), timeout=120).read()
    wall, errors = [], 0
    for f in files:
        body = f.read_bytes()
        t = time.perf_counter()
        try:
            r = urllib.request.urlopen(urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"}), timeout=120)
            r.read()
        except Exception:   # noqa: BLE001
            errors += 1
        wall.append((time.perf_counter() - t) * 1e3)
        if args.pace > 0:
            time.sleep(max(0.0, args.pace - (time.perf_counter() - t)))
    res = {"dump": str(args.dump), "n": len(wall), "errors": errors,
           "wall_median_ms": statistics.median(wall), "wall_p95_ms": float(np.percentile(wall, 95)),
           "wall_max_ms": max(wall), "wall_mean_ms": float(np.mean(wall)),
           "over_500ms": int(sum(w > 500 for w in wall)), "pace_s": args.pace}
    src = Path(args.dump) / "server_act_ms.csv"
    if src.exists():
        ms = [float(l.split(",")[1]) for l in src.read_text().splitlines() if l.strip()]
        res["recorded_machine_act_median_ms"] = statistics.median(ms)
        res["recorded_machine_act_p95_ms"] = float(np.percentile(ms, 95))
    print("[replay]", json.dumps(res))
    Path(args.out).write_text(json.dumps({**res, "wall_ms": wall}, indent=1))


if __name__ == "__main__":
    main()
