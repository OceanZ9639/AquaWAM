#!/usr/bin/env python3
"""USIM-Hard summary table: official-protocol success vs. each hard protocol, per task and arm.

Reads eval_runs/<arm>_<cond>/<task>/results.csv. The official reference for a task is the
arm's *_full block (40 episodes); hard blocks are 20 episodes, so rates are reported in %.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HARD = [
    ("hard_heading", "start heading rotated 90-180 deg"),
    ("hard_tight05", "goto endpoint ball 0.5 m (official 1.0 m)"),
    ("hard_eta2_05", "thruster 3 at 50 % efficiency (regime seen in OU)"),
    ("hard_eta0_05", "thruster 1 at 50 % efficiency (unseen)"),
    ("hard_jerlov05", "turbid water, Jerlov 0.50 (official 0.15)"),
    ("hard_zeroshot", "zero-shot composed tasks (sequential judge)"),
]
ARMS = ["u0", "wam", "fallback", "wam_percept"]
# U0 under the OFFICIAL protocol is cited from the paper (arXiv 2510.07869, Table V "w/" rows);
# our own U0 runs are used only for the conditions the paper does not have (the hard protocols).
PAPER_U0_OFFICIAL = {
    "goto_charge_station": (39, 40), "goto_water_tower": (30, 40),
    "inspect_pipeline_pool": (16, 20), "inspect_pipeline_sea": (20, 20),
    "scan_ship_ancient": (18, 20), "scan_ship_modern": (17, 20),
    "pick_pipe0_shallow": (9, 40), "pick_pipe1_shallow": (12, 40), "pick_pipe0_factory": (21, 40),
    "pick_pipe1_factory": (12, 40), "pick_red_shallow": (8, 40), "pick_redx_shallow": (10, 40),
    "pick_red_factory": (12, 40), "pick_redx_factory": (8, 40), "pick_blue_shallow": (16, 40),
    "pick_bluex_shallow": (14, 40), "pick_blue_factory": (13, 40), "pick_bluex_factory": (9, 40),
    "transfer_red_shallow": (10, 40), "follow_boat": (8, 20),
}


def tally(p: Path):
    if not p.exists():
        return None
    rows = [r for r in csv.reader(open(p)) if r and r[0].isdigit()]
    if not rows:
        return None
    ok = sum(1 for r in rows if r[1] == "success")
    durs = sorted(float(r[2]) for r in rows if r[1] == "success" and len(r) > 2 and r[2])
    med = durs[len(durs) // 2] if durs else float("nan")
    return ok, len(rows), med


def fmt(t):
    if t is None:
        return "–"
    ok, n, med = t
    return f"{ok}/{n} ({100 * ok / n:.0f} %)" + (f" {med:.0f} s" if med == med else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--out", default="/hy-tmp/results/usim_hard/hard_table.md")
    ap.add_argument("--percept-round", default="r4",
                    help="DAgger round whose *_full blocks are the vision arm's official reference")
    args = ap.parse_args()
    root = Path(args.root)

    def full_dir(a: str) -> str:  # the vision arm's official blocks live under wam_percept_<round>_full
        return f"wam_percept_{args.percept_round}_full" if a == "wam_percept" and args.percept_round else f"{a}_full"
    lines = ["# USIM-Hard · same data, same scenes, harder protocol", "",
             "U0 'official' = the paper's own Table V numbers (their hardware); U0 'hard' and every WAM "
             "number = our runs on the official harness. Hard blocks are 20 episodes.", ""]
    for cond, desc in HARD:
        blocks = sorted(root.glob(f"*_{cond}/*/results.csv"))
        if not blocks:
            continue
        tasks = sorted({b.parent.name for b in blocks})
        arms = [a for a in ARMS if any(b.parent.parent.name == f"{a}_{cond}" for b in blocks)]
        lines += [f"## {cond} — {desc}", "",
                  "| task | " + " | ".join(f"{a} official | {a} hard" for a in arms) + " |",
                  "|---|" + "---|---|" * len(arms)]
        for task in tasks:
            base_task = task
            for suffix in ("_roundtrip", "_depth", "_loop"):
                if task.endswith(suffix):
                    base_task = {"scan_ship_loop": "scan_ship_modern"}.get(task, task[: -len(suffix)])
            cells = []
            for a in arms:
                if a == "u0" and base_task in PAPER_U0_OFFICIAL:
                    ok, n = PAPER_U0_OFFICIAL[base_task]
                    cells.append(f"{ok}/{n} ({100 * ok / n:.0f} %) [paper]")
                else:
                    cells.append(fmt(tally(root / full_dir(a) / base_task / "results.csv")))
                cells.append(fmt(tally(root / f"{a}_{cond}" / task / "results.csv")))
            lines.append(f"| {task} | " + " | ".join(cells) + " |")
        lines.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
