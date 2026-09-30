#!/usr/bin/env python3
"""One results document, regenerated from the archived eval blocks: /hy-tmp/results/RESULTS.md

Sections (each produced by its own generator, all reading eval_runs/<arm>_<cond>/<task>/results.csv
and the official 2 Hz logs; nothing is re-run):
  1. Table IV — Full sensing      (paper rows for OpenVLA / π0.5 / GR00T / U0; our WAM rows)
  2. Table IV — DVL dropout       (empty OpenVLA / π0.5 / GR00T rows to fill later; U0 = our run)
  3. per-task success             (U0 full = paper Table V, everything else ours)
  4. harder judges                (hardness_sweep.py: official rule replayed with tighter tolerances)
  5. path quality                 (strict_rescore.py: cross-track error, path-length ratio)
  6. USIM-Hard protocols          (write_hard_table.py)
  7. data efficiency / held-out   (closed-loop goto blocks + offline per-task metrics)
  8. status                       (what is still running / to be filled)

Convention (fixed): U0 under FULL sensing is always the paper's number; every DVL-dropout number and
every hard-protocol number is our own run through the official u0env judges.

  python3 u0eval/write_results_md.py --percept-round r4
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
RES = Path("/hy-tmp/results")


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"_generator failed: {' '.join(cmd[-3:])}_\n\n```\n{r.stderr[-1500:]}\n```"
    return r.stdout.strip()


def tally(p: Path):
    if p.is_dir():
        p = p / "results.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.reader(open(p)) if r and r[0].isdigit()]
    return (sum(1 for r in rows if r[1] == "success"), len(rows)) if rows else None


def path_quality(root: Path, vis: str) -> str:
    """Median cross-track error and path-length ratio per family / arm / condition from
    strict_rescore.json (episode medians -> median over the family's tasks' medians)."""
    f = RES / "usim_hard" / "strict_rescore.json"
    if not f.exists():
        return "_strict_rescore.json missing_"
    t = json.load(open(f))
    fams = {"goto": ("goto_charge_station", "goto_water_tower"), "scan": ("scan_ship_modern", "scan_ship_ancient"),
            "inspect": ("inspect_pipeline_pool", "inspect_pipeline_sea"), "follow": ("follow_boat",)}
    drop = {"goto": "drop8s_zero", "scan": "drop40s_zero", "inspect": "drop40s_zero", "follow": "drop20s_zero"}
    arms = [("u0", "U0"), ("wam", "WAM (priv.)"), (vis, "WAM (vision)"), ("fallback", "fallback")]
    lines = ["| family | condition | " + " | ".join(f"{l} XTE / ratio" for _, l in arms) + " |",
             "|---|---|" + "---|" * len(arms)]
    import statistics
    for fam, tasks in fams.items():
        for cname, cond in (("full", "full"), ("dropout", drop[fam])):
            cells = []
            for arm, _ in arms:
                x, r = [], []
                for task in tasks:
                    s = t.get(task, {}).get(f"{arm}_{cond}")
                    if s:
                        x.append(s["xte_med"])
                        if s["path_ratio_med"] == s["path_ratio_med"]:
                            r.append(s["path_ratio_med"])
                cells.append(f"{statistics.median(x):.2f} m / {statistics.median(r):.2f}" if x and r else "—")
            lines.append(f"| {fam} | {cname} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("*XTE = median distance of the flown path to the reference polyline (lower = tighter path "
                 "following); ratio = flown / reference path length over successful episodes (1.0 = the "
                 "reference; < 1 = corners cut). U0 full sensing here is our re-run (trajectories are needed).*")
    return "\n".join(lines)


def data_efficiency(root: Path) -> str:
    per = RES / "usim_hard" / "per_task.json"
    off = json.load(open(per)) if per.exists() else {}
    rows = [("de_f05", "5 % of USIM"), ("de_f10", "10 %"), ("de_f25", "25 %"), ("de_f50", "50 %"),
            ("de_f100", "100 % (pure USIM, no play data)"), ("best_scenes", "USIM + exploration play data (deployed)"),
            ("ho_nav", "held-out: trained on locomotion tasks only"), ("ho_manip", "held-out: trained on manipulation only"),
            ("ho_wt", "held-out: water tower excluded"), ("ho_scan", "held-out: scan excluded")]
    lines = ["| core | training data | DVL-vel MAE (m/s, goto tasks) | dead-reckoning est MAE | goto closed-loop (charge + tower) |",
             "|---|---|---|---|---|"]
    for name, desc in rows:
        m = off.get(name, {})
        g = [m.get(k) for k in ("Go to the charge station", "Go to the water tower") if k in m]
        dyn = f"{sum(v['dyn_mae'] for v in g) / len(g):.4f}" if g else "—"
        est = f"{sum(v['est_mae'] for v in g) / len(g):.4f}" if g else "—"
        if name == "best_scenes":
            a = tally(root / "wam_full" / "goto_charge_station"); b = tally(root / "wam_full" / "goto_water_tower")
        else:
            a = tally(root / f"wam_dataeff_{name}" / "goto_charge_station"); b = tally(root / f"wam_dataeff_{name}" / "goto_water_tower")
        cl = f"{a[0]}/{a[1]} + {b[0]}/{b[1]}" if a and b else "—"
        lines.append(f"| {name} | {desc} | {dyn} | {est} | {cl} |")
    lines.append("")
    lines.append("*Closed-loop blocks are 10-episode pilots (deployed core: the 40-episode blocks of Table IV). "
                 "Preliminary: the ordering is not monotonic in data fraction and the locomotion-only core is "
                 "the worst at locomotion, which is under diagnosis (server/checkpoint provenance per block "
                 "to be verified before this goes in the paper).*")
    return "\n".join(lines)


def dagger_curve(root: Path) -> str:
    """Vision system per DAgger round (round 0 = USIM-only e2e model; r_k adds the on-policy
    recordings of rounds < k). Same 7 locomotion tasks, official judges; blocks that were not run in
    a round are left blank rather than back-filled."""
    tasks = ["goto_charge_station", "goto_water_tower", "scan_ship_modern", "scan_ship_ancient",
             "inspect_pipeline_pool", "inspect_pipeline_sea", "follow_boat"]
    drop = {"goto_": "drop8s_zero", "follow_": "drop20s_zero", "scan_": "drop40s_zero", "inspect_": "drop40s_zero"}
    rounds = [("r0", "wam_percept"), ("r1", "wam_percept_r1"), ("r2", "wam_percept_r2"), ("r3", "wam_percept_r3"),
              ("r4", "wam_percept_r4")]
    lines = ["| round | " + " | ".join(t.replace("_", " ") for t in tasks) + " | locomotion total (full) | total (dropout) |",
             "|---|" + "---|" * (len(tasks) + 2)]
    for tag, arm in rounds:
        cells, sf, nf, sd, nd = [], 0, 0, 0, 0
        for t in tasks:
            a = tally(root / f"{arm}_full" / t)
            d = tally(root / f"{arm}_{next(v for k, v in drop.items() if t.startswith(k))}" / t)
            cells.append((f"{a[0]}/{a[1]}" if a else "·") + " / " + (f"{d[0]}/{d[1]}" if d else "·"))
            if a:
                sf += a[0]; nf += a[1]
            if d:
                sd += d[0]; nd += d[1]
        if nf or nd:
            lines.append(f"| {tag} | " + " | ".join(cells) + f" | {sf}/{nf}" + (f" ({100 * sf / nf:.0f} %)" if nf else "")
                         + f" | {sd}/{nd}" + (f" ({100 * sd / nd:.0f} %)" if nd else "") + " |")
    lines.append("")
    lines.append("*Cells: full sensing / DVL dropout. r0–r1: frozen DINOv2-base features + MLP goal head (r1 adds "
                 "the round-0 on-policy recordings, labelled by the expert tracker = DAgger); r2: partial pilot; "
                 "r3: first end-to-end fine-tuned DINOv2 intent model (USIM expert videos + rounds 0–2 recordings); "
                 "r4: r3 recipe + round-3 recordings — the model reported in Tables 1–3. Totals count only the "
                 "blocks run in that round.*")
    return "\n".join(lines)


def status(root: Path, vis: str) -> str:
    items = []
    fg = [tally(root / "fallback_drop30s_zero" / t) for t in
          ("pick_pipe0_shallow", "pick_pipe1_shallow", "pick_pipe0_factory", "pick_pipe1_factory", "pick_red_shallow",
           "pick_redx_shallow", "pick_red_factory", "pick_redx_factory", "pick_blue_shallow", "pick_bluex_shallow",
           "pick_blue_factory", "pick_bluex_factory", "transfer_red_shallow")]
    done = sum(1 for x in fg if x and x[1] >= 40)
    items.append(f"fallback grasp / transfer under DVL dropout (mode-aware fallback): {done}/13 blocks complete"
                 + (" — running on box 2 (queue FG, both instances)" if done < 13 else ""))
    items.append("OpenVLA / π0.5 / GR00T N1.5 under DVL dropout: not run (rows left empty)")
    wam_grasp = sum(1 for t in root.glob("wam_full/pick_*/results.csv")) + sum(1 for t in root.glob("wam_full/transfer_*/results.csv"))
    items.append(f"WAM manipulation ablation (hand-tuned pulse-and-settle primitive, 40 episodes): {wam_grasp}/13 blocks archived")
    gp_full = sum(1 for _ in root.glob("wam_gp_wam/*/results.csv"))
    gp_drop = sum(1 for _ in root.glob("wam_gp_wam_drop30s_zero/*/results.csv"))
    items.append(f"WAM manipulation, world-model pulse planner (imagination-chosen hull pulses, 0.5 s replanning): "
                 f"full sensing {gp_full}/13 blocks, DVL dropout {gp_drop}/13 blocks" +
                 (" — dropout blocks running on five instances (local, box 2 A/B, box 3 A/B); transfer full-sensing "
                  "block being re-run cleanly (the first block's results.csv had duplicated episodes)" if gp_drop < 13 else ""))
    hard_vis = sorted(p.parent.parent.name + "/" + p.parent.name for p in root.glob("wam_percept_hard_*/*/results.csv"))
    items.append(f"USIM-Hard on the vision WAM ({vis}): {len(hard_vis)} blocks archived" +
                 (" — queue H4 running" if len(hard_vis) < 12 else ""))
    abl = {"planner budget (n32 / n512)": ["wam_abl_n32", "wam_abl_n512"],
           "data (usim_only / play_only)": ["wam_abl_usim_only", "wam_abl_play_only"],
           "scale (small / large)": ["wam_abl_small", "wam_abl_large"],
           "few-shot embodiment (zero-shot / adapted, full + drop8s)": ["wam_hard_eta0_00", "wam_hard_eta0_00_drop8s",
                                                                       "wam_adapt_eta0_00", "wam_adapt_eta0_00_drop8s"],
           "WAM-direct closed loop (box 2)": ["wam_abl_direct", "wam_abl_direct_drop8s", "wam_abl_direct_drop40s",
                                             "wam_abl_direct_cem"]}
    for name, arms in abl.items():
        n = sum(1 for a in arms for _ in root.glob(f"{a}/*/results.csv"))
        items.append(f"ablation blocks archived — {name}: {n}")
    return "\n".join(f"- {s}" for s in items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--percept-round", default="r4")
    ap.add_argument("--out", default=str(RES / "RESULTS.md"))
    args = ap.parse_args()
    root = Path(args.eval_runs)
    vis = f"wam_percept_{args.percept_round}"
    pr = ["--percept-round", args.percept_round]

    full = run([PY, str(HERE / "write_u0_table.py"), "--auto", *pr, "--condition", "Full sensing",
                "--out", str(RES / "u0_table_full.md")])
    drop = run([PY, str(HERE / "write_u0_table.py"), "--auto", *pr, "--condition", "DVL dropout",
                "--out", str(RES / "u0_table_drop.md")])
    # paper-facing Table 1 (U0/OpenVLA full-sensing rows = USIM paper numbers; everything else our runs)
    run([PY, str(HERE / "write_u0_table.py"), "--auto", "--paper", *pr, "--condition", "Full sensing",
         "--out", str(RES / "table1_full.md")])
    run([PY, str(HERE / "write_u0_table.py"), "--auto", "--paper", *pr, "--condition", "DVL dropout",
         "--out", str(RES / "table1_drop.md")])
    per_task = run([PY, str(HERE / "write_per_task_table.py"), *pr, "--out", str(RES / "per_task.md")])
    hardness = run([PY, str(HERE / "hardness_sweep.py"), *pr, "--out", str(RES / "hardness")])
    hard = run([PY, str(HERE / "write_hard_table.py"), *pr, "--out", str(RES / "usim_hard" / "hard_table.md")])
    run([PY, str(HERE / "strict_rescore.py"), "--out", str(RES / "usim_hard")])
    ablations = run([PY, str(HERE / "ablation_tables.py"), "--eval-runs", str(root)])

    doc = [f"# Underwater WAM vs U0 — results ({datetime.now():%Y-%m-%d %H:%M})", "",
           "All numbers below come from the official u0env harness and judges. **Convention: U0 under full "
           "sensing is cited from the USIM/U0 paper (arXiv 2510.07869 v4, Table IV / Table V); every DVL-dropout "
           "number and every hard-protocol number is our own run.** WAM (privileged) consumes the mapper's "
           "reference waypoints / target poses; WAM (vision) and U0 are vision-driven; U0+WAM fallback runs U0 "
           "and hands control to WAM only when the DVL dies on a locomotion task.", "",
           "## 1. Table IV — Full sensing", "", full.split("\n", 1)[1].strip(), "",
           "## 2. Table IV — DVL dropout", "", drop.split("\n", 1)[1].strip(), "",
           "## 3. Per-task success", "", per_task.replace("### Per-task success — ", "### "), "",
           "## 4. Harder judges (same episodes, stricter rule)", "", hardness.split("\n", 1)[1].strip(), "",
           "## 5. Path quality", "", path_quality(root, vis), "",
           "## 6. USIM-Hard protocols", "", hard.split("\n", 1)[1].strip().replace("\n## hard_", "\n### hard_"), "",
           "## 7. Data efficiency and held-out tasks (world-model core)", "", data_efficiency(root), "",
           "## 8. Vision system across DAgger rounds", "", dagger_curve(root), "",
           "## 9. World-action-model ablations (planner budget, amortized head, data, scale, embodiment adaptation, "
           "failure prediction)", "", ablations, "",
           "## 10. Status", "", status(root, vis), ""]
    text = "\n".join(doc)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
