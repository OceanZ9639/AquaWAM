#!/usr/bin/env python3
"""Produce the USIM/U0 paper's Table IV, extended with our arms and conditions.

Consumes eval_runs directories produced by run_eval_task.sh (each holds
<task_code>/results.csv + logs written by the official judges) and computes the
category metrics with the paper's own tools/evalmetrics code -- no re-implemented
success criteria. Paper-reported rows (OpenVLA / pi0.5 / GR00T N1.5 / U0) are
cited verbatim for the full-sensing block.

Usage:
  python3 u0eval/write_u0_table.py \
      --run "U0 (repro)"=/hy-tmp/u0env/dataset/eval_runs/u0_full \
      --run "WAM (ours)"=/hy-tmp/u0env/dataset/eval_runs/wam_full \
      --condition "Full sensing" \
      --out /hy-tmp/results/u0_table_full.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

U0ENV = Path("/hy-tmp/u0env")
sys.path.insert(0, str(U0ENV))

from tools.evalmetrics.data_loader import DataLoader  # noqa: E402
from tools.evalmetrics.metrics_calculator import (  # noqa: E402
    GraspingMetrics,
    NavigationMetrics,
    TrackingMetrics,
    TransportingMetrics,
)

# arXiv 2510.07869 v4, Table IV (700 trials; cited, not measured here)
PAPER_ROWS = [
    ("OpenVLA [paper]", "25.6%", "0.70±0.21", "0.0%", "—", "0.0%", "0.0%", "5.0%", "4.02 m", "6.0% (42/700)"),
    ("π0.5 [paper]", "46.9%", "0.87±0.18", "37.1%", "97.3 s", "20.0%", "45.0%", "10.0%", "4.52 m", "37.6% (263/700)"),
    ("GR00T N1.5 [paper]", "76.3%", "0.69±0.21", "24.0%", "92.3 s", "15.0%", "30.0%", "30.0%", "4.16 m", "35.6% (249/700)"),
    ("U0 [paper]", "87.5%", "0.71±0.23", "30.0%", "87.6 s", "25.0%", "45.0%", "40.0%", "3.61 m", "43.1% (302/700)"),
]

HEADERS = ["Model", "Nav SR↑", "Nav SPL↑", "Grasp SR↑", "Grasp ASD↓",
           "Transport SR↑", "Transport SSR↑", "Track SR↑", "Track MTD↓", "Overall SR"]
# The paper's other baselines under DVL dropout. No underwater checkpoints were released, so we
# fine-tune each on USIM ourselves (footnote: our training budget, not the paper's) and evaluate with
# the same harness/judges at 20 episodes per task: eval_runs/<arm>_drop*s_zero/<task>. A baseline that
# has no measured block yet is kept as an empty row so the dropout table keeps Table IV's row set.
BASELINE_ARMS = [("openvla", "OpenVLA"), ("pi05", "π0.5"), ("gr00t", "GR00T N1.5"),
                 ("xvla", "X-VLA (ICLR'26)"), ("smolvla", "SmolVLA"),
                 ("gr00t2", "GR00T N1.5, full action-head FT 120k×16"), ("fastwam", "FastWAM (WAM baseline)")]
# the same checkpoints WITHOUT any USIM fine-tune (released weights, our observations mapped onto their
# input slots, their action vector mapped positionally onto our 13-d interface): full sensing only
ZERO_SHOT_ARMS = [("gr00t_zs", "GR00T N1.5"), ("pi05_zs", "π0.5"), ("xvla_zs", "X-VLA"), ("smolvla_zs", "SmolVLA")]
PLACEHOLDER_ROWS = [(f"{m} (DVL dropout)", *(["—"] * 8), "(to run)") for _, m in BASELINE_ARMS[:3]]


def dedup_results(run_dir: Path) -> int:
    """Keep the last row per episode id in every results.csv under run_dir. The harness occasionally
    writes a second row for the same episode (seen in the baseline runs, never in the WAM/U0 blocks)
    and the official metrics loader reads every row, so duplicates must go before merging."""
    n = 0
    for f in run_dir.glob("*/results.csv"):
        rows = [r for r in csv.reader(open(f))]
        head = rows[0] if rows and not rows[0][0].isdigit() else ["episode", "result"]
        data = [r for r in rows if r and r[0].isdigit()]
        last = {}
        for r in data:
            last[int(r[0])] = r
        if len(last) != len(data):
            n += 1
            target = f.resolve()
            with open(target, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(head)
                for k in sorted(last):
                    w.writerow(last[k])
            print(f"[dedup] {target}: {len(data)} rows -> {len(last)} episodes", file=sys.stderr)
    return n


def _n_rows(task_dir: Path) -> int:
    f = task_dir / "results.csv"
    if not f.exists():
        return 0
    return max(0, sum(1 for _ in open(f)) - 1)


def merge_condition(eval_runs: Path, arm: str, kind: str, dest: Path) -> Path:
    """kind='full' links <arm>_full; kind='drop' unions every <arm>_drop*.

    Several <arm>_drop<T>s dirs can hold the same task (early debug runs at other
    drop times next to the protocol run); the one with the most completed trials
    wins, so a 1-episode debug dir can never shadow a 40-trial protocol block.
    """
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    if kind == "full":
        # <arm>_full plus any protocol-suffixed re-runs (<arm>_full_rec1, ..._rec16): per task, the block
        # with the most completed trials wins, exactly as for the dropout dirs below
        srcs = [eval_runs / f"{arm}_full"] + sorted(p for p in eval_runs.iterdir()
                                                    if p.is_dir() and p.name.startswith(f"{arm}_full_rec"))
    else:
        srcs = sorted(p for p in eval_runs.iterdir()
                      if p.is_dir() and p.name.startswith(f"{arm}_drop"))
    best: dict = {}
    for src in srcs:
        if not src.exists():
            continue
        for task in src.iterdir():
            if not task.is_dir():
                continue
            n = _n_rows(task)
            if task.name not in best or n > best[task.name][0]:
                best[task.name] = (n, task)
    for name, (_, task) in best.items():
        tdest = dest / name
        if tdest.exists() or tdest.is_symlink():
            if tdest.is_symlink() or tdest.is_file():
                tdest.unlink()
            else:
                shutil.rmtree(tdest)
        tdest.symlink_to(task.resolve())
    return dest


def apply_manip_override(eval_runs: Path, dest: Path, kind: str, tag: str = "gp_wam", strict: bool = True,
                         only: set | None = None) -> int:
    """The deployed WAM does manipulation with the world-model pulse planner (dirs wam_gp_wam /
    wam_gp_wam_drop<T>s_zero); the plain wam_* dirs hold the earlier hand-tuned primitive, kept as
    the ablation. Re-point the manipulation tasks of a merged WAM condition dir to the pulse blocks
    (only those that exist), return how many were overridden."""
    if kind == "full":
        srcs = [eval_runs / f"wam_{tag}"]
    else:
        srcs = sorted(p for p in eval_runs.iterdir() if p.is_dir() and p.name.startswith(f"wam_{tag}_drop"))
    n = 0
    linked = set()
    for src in srcs:
        if not src.exists():
            continue
        for task in src.iterdir():
            if not task.is_dir() or classify(task.name) not in ("grasping", "transporting") or _n_rows(task) == 0:
                continue
            if only is not None and classify(task.name) not in only:
                continue
            tdest = dest / task.name
            if tdest.is_symlink() or tdest.is_file():
                tdest.unlink()
            elif tdest.exists():
                import shutil
                shutil.rmtree(tdest)
            tdest.symlink_to(task.resolve())
            n += 1
            linked.add(task.name)
    # strict: the row must not silently mix sources -- manipulation tasks the new source has not
    # produced yet are dropped from the merged dir (the table then reports the partial count)
    if n and strict:
        for tdest in list(dest.iterdir()):
            if classify(tdest.name) in ("grasping", "transporting") and tdest.name not in linked:
                if tdest.is_symlink() or tdest.is_file():
                    tdest.unlink()
                else:
                    import shutil
                    shutil.rmtree(tdest)
    return n


def classify(task_code: str) -> str:
    if task_code.startswith(("goto_", "go_to_", "inspect_", "scan_")):
        return "navigation"
    if task_code.startswith("pick_"):
        return "grasping"
    if task_code.startswith(("transfer_", "transport_")):
        return "transporting"
    if task_code.startswith("follow_"):
        return "tracking"
    return "navigation"


_TS_NO_FRAC = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}),", re.M)


def normalize_timestamps(run_dir: Path) -> int:
    """The official recorder writes datetime.isoformat(), which drops the '.%f' part when the
    microsecond happens to be 0 (about one row per 10^6). pandas then infers '%Y-%m-%dT%H:%M:%S.%f'
    from the first row and rejects the whole file, so the official DataLoader silently skips that
    episode's path metrics. Pad such rows with '.000000' in place (value-preserving)."""
    fixed = 0
    for f in run_dir.glob("*/logs/episode_*_data.csv"):
        txt = f.read_text()
        new, n = _TS_NO_FRAC.subn(r"\1.000000,", txt)
        if n:
            f.write_text(new)
            fixed += 1
            print(f"[fix] padded {n} fraction-less timestamp(s) in {f}", file=sys.stderr)
    return fixed


def measure_run(run_dir: Path) -> dict:
    normalize_timestamps(run_dir)
    dl = DataLoader(str(run_dir))
    tasks = dl.get_available_tasks()
    calc = {
        "navigation": NavigationMetrics(dl),
        "grasping": GraspingMetrics(dl),
        "transporting": TransportingMetrics(dl),
        "tracking": TrackingMetrics(dl),
    }
    per_task = {}
    for t in sorted(tasks):
        fam = classify(t)
        try:
            if fam == "transporting":
                per_task[t] = (fam, calc[fam].calculate_all_metrics(t, 3))
            else:
                per_task[t] = (fam, calc[fam].calculate_all_metrics(t))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] metrics failed for {t}: {e}", file=sys.stderr)
    agg = {f: {"succ": 0, "tot": 0, "extra": []} for f in calc}
    for t, (fam, m) in per_task.items():
        succ = int(m.get("successful_episodes", 0) or 0)
        tot = int(m.get("total_episodes", 0) or 0)
        agg[fam]["succ"] += succ
        agg[fam]["tot"] += tot
        if fam == "navigation" and m.get("spl_values"):
            # per-episode SPL of the successful episodes, pooled over tasks (USIM reports mean±std)
            agg[fam]["extra"] += [float(v) for v in m["spl_values"]]
        if fam == "grasping":
            d = (m.get("avg_success_duration") or {}).get("mean")
            if d is not None:
                agg[fam]["extra"] += [float(d)] * max(1, succ)
        if fam == "transporting":
            ssr = m.get("stage1_grasp_rate")
            if ssr is not None:
                agg[fam]["extra"] += [float(ssr)] * max(1, tot)
        if fam == "tracking":
            vals = (m.get("avg_tracking_distance") or {}).get("values") or []
            # per-episode mean distance of the successful episodes, pooled (USIM reports mean±std)
            agg[fam]["extra"] += [float(v) for v in vals]
    return {"per_task": {t: m for t, (_, m) in per_task.items()}, "agg": agg}


def _pct(s, t):
    return f"{100.0 * s / t:.1f}%" if t else "—"


def _mean(xs, fmt):
    return fmt.format(sum(xs) / len(xs)) if xs else "—"


def _mean_std(xs, fmt, unit=""):
    """'mean±std' over per-episode values (population std, as USIM's evalmetrics uses np.std)."""
    if not xs:
        return "—"
    n = len(xs)
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / n) ** 0.5
    return f"{fmt.format(mu)}±{fmt.format(sd)}{unit}"


def row_from_measure(label: str, meas: dict) -> list:
    a = meas["agg"]
    nav, gr, tr, tk = a["navigation"], a["grasping"], a["transporting"], a["tracking"]
    tot_s = sum(x["succ"] for x in a.values())
    tot_n = sum(x["tot"] for x in a.values())
    return [
        label,
        _pct(nav["succ"], nav["tot"]),
        _mean_std(nav["extra"], "{:.2f}"),
        _pct(gr["succ"], gr["tot"]),
        _mean(gr["extra"], "{:.1f} s"),
        _pct(tr["succ"], tr["tot"]),
        _mean(tr["extra"], "{:.1%}"),
        _pct(tk["succ"], tk["tot"]),
        _mean_std(tk["extra"], "{:.2f}", " m"),
        f"{_pct(tot_s, tot_n)} ({tot_s}/{tot_n})",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    help='"label"=/path/to/eval_runs/<arm_cond>; repeatable')
    ap.add_argument("--condition", default="Full sensing")
    ap.add_argument("--paper-rows", action="store_true", default=True)
    ap.add_argument("--no-paper-rows", dest="paper_rows", action="store_false")
    ap.add_argument("--placeholder-rows", action="store_true", default=True,
                    help="dropout condition: empty OpenVLA / π0.5 / GR00T rows to be filled later (default on)")
    ap.add_argument("--no-placeholder-rows", dest="placeholder_rows", action="store_false")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-runs", default="/hy-tmp/u0env/dataset/eval_runs",
                    help="if set with --auto, discover u0/wam/fallback full+drop dirs")
    ap.add_argument("--repro-row", action="store_true", help="also print our own U0 full-sensing re-run")
    ap.add_argument("--percept-round", default="r1",
                    help="DAgger round of the vision-goal arm to report ('' = round 0 dirs)")
    ap.add_argument("--auto", action="store_true",
                    help="build one table per condition from --eval-runs")
    ap.add_argument("--paper", action="store_true",
                    help="paper Table 1: every row is our own run -- U0 from its released weights in our harness "
                         "(u0rep) under full sensing, our U0 run under dropout, our fine-tunes, WaterWAM; plain labels, "
                         "no calibration / zero-shot rows, no paper-reported numbers anywhere")
    args = ap.parse_args()

    if args.auto:
        runs = Path(args.eval_runs)
        tmp = Path("/tmp/u0_table_merged")
        specs = []
        # the vision-goal arm is reported at its latest DAgger round (dirs wam_percept_<round>_*)
        percept_dir = f"wam_percept_{args.percept_round}" if args.percept_round else "wam_percept"
        # full sensing: the U0 row is the paper's (PAPER_ROWS); our re-run is a footnote row only when
        # --repro-row is given. Dropout has no paper counterpart, so our U0 run is the baseline there.
        arms = [("u0", "U0 (ours, DVL dropout)" if not args.condition.lower().startswith("full") else "U0 (repro, footnote)")]
        if args.paper:
            arms = []
        if args.condition.lower().startswith("full") and not args.repro_row:
            arms = []
        if args.paper:
            pass   # no re-run row in the paper table
        elif args.condition.lower().startswith("full") and any(runs.glob("u0rep_full/*/results.csv")):
            # the official U0 weights re-run through this harness, 20 episodes per task, same protocol
            # as the baselines: shows whether the harness reproduces the paper's Table IV numbers
            arms.append(("u0rep", "U0 (official weights, re-run in our harness)"))
        kind = "full" if args.condition.lower().startswith("full") else "drop"
        if args.paper:
            # U0 is our own run in both conditions: the released weights re-run in our harness under full
            # sensing (u0rep, 700-trial protocol), our dropout run otherwise. Then our fine-tunes under both
            # conditions, plain labels; GR00T N1.5 = the full action-head fine-tune when it exists, else LoRA
            paper_arms = ([("u0", "U0")] if kind == "drop" else [("u0rep", "U0")]) + \
                         [("openvla", "OpenVLA"), ("pi05", "π0.5"), ("gr00t2", "GR00T N1.5"), ("gr00t", "GR00T N1.5"),
                          ("xvla", "X-VLA"), ("smolvla", "SmolVLA"), ("fastwam", "FastWAM")]
            seen = set()
            for arm, name in paper_arms:
                if name in seen or not any(runs.glob(f"{arm}_{'full*' if kind == 'full' else 'drop*'}/*/results.csv")):
                    continue
                dest = tmp / f"{arm}_{kind}"
                merge_condition(runs, arm, kind, dest)
                specs.append(f"{name}={dest}")
                seen.add(name)
            PLACEHOLDER_ROWS[:] = []
        elif kind == "drop":
            # the paper's baselines, fine-tuned by us on USIM and run under the same dropout protocol;
            # placeholders stay only for the ones with no measured block yet
            measured = []
            for arm, name in BASELINE_ARMS:
                if any(runs.glob(f"{arm}_drop*/*/results.csv")):
                    dest = tmp / f"{arm}_{kind}"
                    merge_condition(runs, arm, kind, dest)
                    specs.append(f"{name} (USIM fine-tune by us, DVL dropout)†={dest}")
                    measured.append(name)
            PLACEHOLDER_ROWS[:] = [r for r in PLACEHOLDER_ROWS if not any(r[0].startswith(m) for m in measured)]
        else:
            # calibration rows: our USIM fine-tune of a paper baseline re-run under full sensing, so the
            # reader can relate the dropout rows (which only exist for our fine-tune) to the paper's numbers
            for arm, name in BASELINE_ARMS:
                if any(runs.glob(f"{arm}_full/*/results.csv")):
                    dest = tmp / f"{arm}_{kind}"
                    merge_condition(runs, arm, kind, dest)
                    specs.append(f"{name} (USIM fine-tune by us, calibration)†={dest}")
            for arm, name in ZERO_SHOT_ARMS:
                if any(runs.glob(f"{arm}_full/*/results.csv")):
                    dest = tmp / f"{arm}_{kind}"
                    merge_condition(runs, arm, kind, dest)
                    specs.append(f"{name} (released weights, zero-shot, no fine-tune)‡={dest}")
        has_pulse = any(runs.glob(f"wam_gp_wam{'' if kind == 'full' else '_drop*'}/*/results.csv"))
        wam_rows = [("wam", "WAM (privileged wp)"),
                    (percept_dir, f"WAM (vision goals{', ' + args.percept_round if args.percept_round else ''})"),
                    ("fallback", "U0+WAM fallback")]
        if args.paper:
            wam_rows = [(percept_dir, "WaterWAM (camera-driven)"), ("wam", "WaterWAM (given goal poses)")]
            if kind == "drop":
                wam_rows.append(("fallback", "U0 + WaterWAM takeover"))
        for arm, label in arms + wam_rows:
            if not any(runs.glob(f"{arm}_{'full' if kind == 'full' else 'drop*'}/*/results.csv")):
                continue  # arm not evaluated (yet)
            dest = tmp / f"{arm}_{kind}"
            merge_condition(runs, arm, kind, dest)
            if arm == percept_dir:
                # vision row: locomotion from the DAgger'd image-goal head, manipulation from the
                # wrist-camera-driven grasp blocks -- the latest DAgger round of the wrist head that has
                # all 12 pick blocks (wam_wrist_r2 before wam_wrist)
                wtag = next((t for t in ("wrist_r2", "wrist") if len(list(runs.glob(f"wam_{t}{'' if kind == 'full' else '_drop*'}/pick_*/results.csv"))) >= 12), "wrist")
                n_w = apply_manip_override(runs, dest, kind, tag=wtag)
                # transporting: the fully camera-driven block (object from the wrist head, destination
                # from the forward-camera container head) when it exists -- otherwise the wrist-only
                # block, whose destination is still read from the task file
                n_w += apply_manip_override(runs, dest, kind, tag=f"{wtag}_box", strict=False,
                                            only={"transporting"})
                if n_w and not args.paper:
                    label = f"WAM (vision: image goals {args.percept_round} + wrist camera {wtag.replace('wrist_', '').replace('wrist', 'r1')})"
                    print(f"[vision] {n_w} manipulation task(s) taken from the wrist-camera blocks ({kind})", file=sys.stderr)
            if arm == "wam" and has_pulse:
                # deployed system: manipulation by the world-model pulse planner; the hand-tuned
                # primitive stays as an ablation row right below
                n_over = apply_manip_override(runs, dest, kind)
                specs.append(f"{'WaterWAM (given goal poses)' if args.paper else 'WAM (privileged wp, world-model pulses)'}={dest}")
                if args.paper:
                    pass   # the hand-tuned primitive is an ablation-table row, not a Table 1 row
                elif kind == "full":   # the plain wam_full dirs hold the primitive under full sensing
                    prim = tmp / f"wam_prim_{kind}"
                    merge_condition(runs, arm, kind, prim)
                    specs.append(f"WAM, hand-tuned fine positioning (ablation)={prim}")
                elif any(runs.glob("wam_gp_prim_drop*/*/results.csv")):
                    # dropout counterpart, re-run with the final code (primitive planner, 1.6 s cadence,
                    # estimator v2) into wam_gp_prim_drop<T>s_zero/; locomotion rows come from wam_drop*
                    prim = tmp / f"wam_prim_{kind}"
                    merge_condition(runs, arm, kind, prim)
                    n_prim = apply_manip_override(runs, prim, kind, tag="gp_prim")   # strict: no old-code blocks mixed in
                    specs.append(f"WAM, hand-tuned fine positioning (ablation, {n_prim}/13 tasks)={prim}")
                print(f"[wam] {n_over} manipulation task(s) taken from the pulse-planner blocks ({kind})", file=sys.stderr)
                continue
            specs.append(f"{label}={dest}")
        args.run = specs

    lines = [f"### {'Table 1' if args.paper else 'Online evaluation'} — {args.condition}", ""]
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("|" + "|".join(["---"] * len(HEADERS)) + "|")
    if args.paper:
        pass   # no paper-reported rows or numbers in the paper table
    elif args.paper_rows and args.condition.lower().startswith("full"):
        for r in PAPER_ROWS:
            lines.append("| " + " | ".join(r) + " |")
    if args.placeholder_rows and not args.condition.lower().startswith("full"):
        for r in PLACEHOLDER_ROWS:
            lines.append("| " + " | ".join(r) + " |")
    dump = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        dedup_results(Path(path))
        meas = measure_run(Path(path))
        dump[label] = meas["per_task"]
        lines.append("| " + " | ".join(str(x) for x in row_from_measure(label, meas)) + " |")
    lines.append("")
    if args.paper:
        lines.append("*U0 is run from its released weights in the official harness under the same protocol as every other row. "
                     "Baselines other than U0 are fine-tuned by us on the USIM demonstrations (Appendix). "
                     "WaterWAM (given goal poses) reads the benchmark's reference waypoints and object poses, so the difference between the two WaterWAM rows is the cost of perception; "
                     "WaterWAM (camera-driven) and all baselines are vision-driven.*")
        md = "\n".join(lines)
        print(md)
        if args.out:
            Path(args.out).write_text(md + "\n")
            Path(args.out).with_suffix(".json").write_text(json.dumps(dump, indent=1, default=str))
        return
    lines.append("*Paper rows cited from arXiv 2510.07869 v4 Table IV (their hardware, 700 trials). "
                 "Measured rows use the official u0env judges and tools/evalmetrics on this machine. "
                 "WAM (privileged wp) consumes the mapper's reference waypoints / target poses (the same "
                 "privileged source the paper's expert data collector used); WAM (vision goals) and U0 "
                 "are vision-driven.*")
    if any("†" in spec for spec in args.run):
        lines.append("")
        lines.append("*† No underwater checkpoints of these models were released. We fine-tuned them on the USIM "
                     "expert data ourselves (GR00T N1.5: LoRA r=64, 40k steps × batch 16, the U0 recipe; "
                     "π0.5: LeRobot pi05, 15k steps × batch 32, expert-only, chunk 16; X-VLA: lerobot/xvla-base, full Phase-II "
                     "adaptation, 20k × 16; SmolVLA: lerobot/smolvla_base, 20k × 32), a smaller budget than "
                     "the paper's runs, and evaluated 20 episodes per task with the official judges. "
                     "Full-sensing rows marked † are the same fine-tunes re-run under full sensing as a "
                     "calibration against the paper's numbers (the [paper] rows).*")
    if any("‡" in spec for spec in args.run):
        lines.append("")
        lines.append("*‡ Zero-shot: the released checkpoint with no USIM training at all -- our cameras fed into its "
                     "image slots, our 29-d proprio padded/truncated positionally to its state slot, its action "
                     "vector mapped positionally onto our 13-d joint+thruster interface (GR00T: pretrained backbone "
                     "with the untrained new-embodiment head). These models have never seen a thruster-driven "
                     "vehicle, so this row measures how far pretraining alone gets an underwater robot.*")
    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(dump, indent=1, default=str))
        print(f"\nsaved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
