#!/usr/bin/env python3
"""Per-task success table (both sensing conditions), the companion of the Table IV pair.

U0 under FULL sensing is cited from the paper (arXiv 2510.07869, Table V "w/" rows), never from
our re-run; our own U0 run is used only for DVL dropout, which the paper does not evaluate.
All other rows are our measurements with the official u0env judges (results.csv).

  python3 u0eval/write_per_task_table.py --percept-round r4 --out /hy-tmp/results/per_task_r4.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_hard_table import PAPER_U0_OFFICIAL  # noqa: E402

LOCOMOTION = ["goto_charge_station", "goto_water_tower", "scan_ship_modern", "scan_ship_ancient",
              "inspect_pipeline_pool", "inspect_pipeline_sea", "follow_boat"]
MANIPULATION = ["pick_pipe0_shallow", "pick_pipe1_shallow", "pick_pipe0_factory", "pick_pipe1_factory",
                "pick_red_shallow", "pick_redx_shallow", "pick_red_factory", "pick_redx_factory",
                "pick_blue_shallow", "pick_bluex_shallow", "pick_blue_factory", "pick_bluex_factory",
                "transfer_red_shallow"]
# protocol dropout time per task family (seconds after the first action)
DROP = {"goto_": "drop8s_zero", "follow_": "drop20s_zero", "scan_": "drop40s_zero", "inspect_": "drop40s_zero",
        "pick_": "drop30s_zero", "transfer_": "drop30s_zero"}


def drop_cond(task: str) -> str:
    return next(v for k, v in DROP.items() if task.startswith(k))


def count(root: Path, arm_cond: str, task: str):
    p = root / arm_cond / task / "results.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.reader(open(p)) if r and r[0].isdigit()]
    if not rows:
        return None
    # the harness occasionally writes two rows for the same episode (seen in the baseline runs, never in
    # the WAM/U0 blocks): keep the last row per episode id so a retry cannot inflate the count
    last = {}
    for r in rows:
        last[int(r[0])] = r
    rows = [last[k] for k in sorted(last)]
    return sum(1 for r in rows if r[1] == "success"), len(rows)


def cell(t, bold=False) -> str:
    if t is None:
        return "—"
    s = f"{t[0]}/{t[1]}"
    return f"**{s}**" if bold else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--percept-round", default="r4")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.eval_runs)
    vis = f"wam_percept_{args.percept_round}"
    lines = []
    for title, tasks in (("Locomotion", LOCOMOTION), ("Manipulation", MANIPULATION)):
        lines += [f"### Per-task success — {title}", "",
                  "| Task | U0 [paper] | WAM (privileged) | WAM (vision" + (", wrist cam" if title == "Manipulation" else ", " + args.percept_round) + ") | U0+WAM fallback "
                  "| | U0 (ours) | WAM (privileged) | WAM (vision" + (", wrist cam" if title == "Manipulation" else ", " + args.percept_round) + ") | U0+WAM fallback |",
                  "|---|---|---|---|---|---|---|---|---|---|",
                  "| *condition* | *full* | *full* | *full* | *full* | | *DVL dropout* | *DVL dropout* | *DVL dropout* | *DVL dropout* |"]
        for t in tasks:
            d = drop_cond(t)
            paper = PAPER_U0_OFFICIAL.get(t)
            # manipulation: the deployed WAM uses the world-model pulse planner (wam_gp_wam*); the plain
            # wam_* dirs are the hand-tuned primitive, shown in the ablation column below
            wam_full = count(root, "wam_gp_wam", t) if title == "Manipulation" and count(root, "wam_gp_wam", t) else count(root, "wam_full", t)
            wam_drop = count(root, f"wam_gp_wam_{d}", t) if title == "Manipulation" and count(root, f"wam_gp_wam_{d}", t) else count(root, f"wam_{d}", t)
            wt = "wam_wrist_r2" if (root / "wam_wrist_r2" / t / "results.csv").exists() else "wam_wrist"
            if (root / "wam_wrist_r2_box" / t / "results.csv").exists():
                wt = "wam_wrist_r2_box"      # fully camera-driven transfer (container head)
            vis_full = count(root, wt, t) if title == "Manipulation" else count(root, f"{vis}_full", t)
            vis_drop = count(root, f"{wt}_{d}", t) if title == "Manipulation" else count(root, f"{vis}_{d}", t)
            full = [paper, wam_full, vis_full, count(root, "fallback_full", t)]
            drop = [count(root, f"u0_{d}", t), wam_drop, vis_drop, count(root, f"fallback_{d}", t)]
            # bold: our arms that meet or beat the paper's U0 on the same task (full sensing)
            full_cells = [cell(paper)] + [cell(x, bold=(x is not None and paper is not None and x[0] / x[1] >= paper[0] / paper[1]))
                                          for x in full[1:]]
            lines.append(f"| {t} | " + " | ".join(full_cells) + " | | " + " | ".join(cell(x) for x in drop) + " |")
        lines.append("")
        if title == "Manipulation":
            lines += ["Ablation — the hand-tuned pulse-and-settle primitive in place of the world-model pulse planner "
                      "(same stage machine, same close rule):", "",
                      "| Task | primitive, full | primitive, DVL dropout |", "|---|---|---|"]
            for t in tasks:
                lines.append(f"| {t} | {cell(count(root, 'wam_full', t))} | {cell(count(root, f'wam_{drop_cond(t)}', t))} |")
            lines.append("")
    # the paper's other baselines, fine-tuned on USIM by us (no underwater checkpoints released) and run
    # under the same dropout protocol at 20 episodes per task: <arm>_drop*s_zero/<task>
    base_arms = [(a, n) for a, n in (("gr00t", "GR00T N1.5"), ("gr00t2", "GR00T N1.5 full-FT"), ("pi05", "π0.5"), ("xvla", "X-VLA"), ("smolvla", "SmolVLA"), ("fastwam", "FastWAM"), ("openvla", "OpenVLA"))
                 if any(root.glob(f"{a}_drop*/*/results.csv"))]
    if base_arms:
        lines += ["### Per-task success under DVL dropout — the paper's baselines (USIM fine-tune by us) vs. ours", "",
                  "| Task | " + " | ".join(n for _, n in base_arms) + " | U0 (ours) | WAM (vision) |",
                  "|---|" + "---|" * (len(base_arms) + 2)]
        for t in LOCOMOTION + MANIPULATION:
            d = drop_cond(t)
            if t in MANIPULATION:
                wt = "wam_wrist_r2_box" if (root / "wam_wrist_r2_box" / t / "results.csv").exists() else "wam_wrist_r2"
                ours = count(root, f"{wt}_{d}", t)
            else:
                ours = count(root, f"{vis}_{d}", t)
            cells = [cell(count(root, f"{a}_{d}", t)) for a, _ in base_arms]
            lines.append(f"| {t} | " + " | ".join(cells) + f" | {cell(count(root, f'u0_{d}', t))} | {cell(ours)} |")
        lines += ["", "*Baselines fine-tuned by us on the USIM expert data (GR00T N1.5: LoRA r=64, 40k×16, the U0 recipe; "
                  "π0.5: LeRobot pi05, 15k×32; X-VLA: xvla-base Phase-II, 20k×16; SmolVLA: smolvla_base, 20k×32) -- a smaller training budget than the paper's; 20 episodes per task, "
                  "official judges. A cell with fewer than 20 episodes is a block still running.*", ""]
    lines.append("*U0 full-sensing numbers are the paper's Table V (arXiv 2510.07869 v4); every other cell is our run "
                 "through the official u0env judges. Bold = meets or beats the paper's U0 on that task. The fallback "
                 "arm under full sensing is U0 itself (WAM never takes over), so its full column is only reported "
                 "where the block was run. Dropout: DVL zeroed at 8 s (goto), 20 s (follow), 40 s (scan/inspect), "
                 "30 s (pick/transfer) after the first action.*")
    md = "\n".join(lines)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"\nsaved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
