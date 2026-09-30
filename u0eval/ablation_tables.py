#!/usr/bin/env python3
"""
DreamZero-style ablations of the underwater WAM, tallied from archived eval blocks.

  A. planner budget & amortization   goto x2, full sensing: CEM 32x1 / 128x2 / 512x2, WAM-direct
                                     (amortized head), WAM-direct + 32x1 polish; plan latency from
                                     the servers' own timing lines (logs/u0eval/wam_plan_latency.log)
  B. repetitive demos vs play data   cores: USIM only (12.6 h repeated task demos) / play only
                                     (3.3 h non-repetitive exploration + deployment) / both (deployed)
  C. model scale                     0.68 M / 2.4 M / 8.9 M cores, same recipe
  D. few-shot embodiment adaptation  dead thruster 1: U0 zero-shot, WAM zero-shot, WAM after 30 min
                                     of faulted play (full sensing and DVL loss at 8 s)
  E. WAM-direct closed loop          4 locomotion tasks x {full, dropout}: MPC vs direct vs direct+CEM
  F. failure prediction (P5)         AUROC of world-model statistics for U0's dropout failures

Every cell is "successes/episodes (mean time-to-success s)" from eval_runs/<arm>_<cond>/<task>/results.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

GOTO = ["goto_charge_station", "goto_water_tower"]
SHORT = {"goto_charge_station": "goto charge", "goto_water_tower": "goto tower", "inspect_pipeline_pool": "inspect pool",
         "scan_ship_ancient": "scan ancient"}


def tally(root: Path, arm_cond: str, task: str):
    p = root / arm_cond / task / "results.csv"
    if not p.exists():
        return None
    rows = [r for r in csv.reader(open(p)) if r and r[0].isdigit()]
    if not rows:
        return None
    ok = [r for r in rows if r[1] == "success"]
    times = [float(r[2]) for r in ok if len(r) > 2 and re.match(r"^[0-9.]+$", r[2])]
    return len(ok), len(rows), (sum(times) / len(times) if times else None)


def cell(t, n_expect: int | None = None):
    if t is None:
        return "—"
    s, n, mt = t
    txt = f"{s}/{n}"
    if mt is not None:
        txt += f" ({mt:.0f} s)"
    if n_expect and n < n_expect:
        txt += " *"
    return txt


def latency_table(path: Path):
    """Group the servers' 'plan latency (source, n=.., iters=..)' lines -> mean of the reported means."""
    if not path.exists():
        return {}
    pat = re.compile(r"plan latency \((\w+), n=(\d+), iters=(\d+)\): mean ([0-9.]+) ms\s+p50 ([0-9.]+)\s+p95 ([0-9.]+)")
    acc = defaultdict(list)
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            acc[(m.group(1), int(m.group(2)), int(m.group(3)))].append((float(m.group(4)), float(m.group(6))))
    return {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v), len(v)) for k, v in acc.items()}


def sec_planner(root: Path, lat: dict) -> str:
    rows = [("sampling MPC, CEM 32 x 1", "wam_abl_n32", ("mpc", 32, 1)),
            ("sampling MPC, CEM 128 x 2 (deployed)", "wam_full", ("mpc", 128, 2)),
            ("sampling MPC, CEM 512 x 2", "wam_abl_n512", ("mpc", 512, 2)),
            ("WAM-direct (amortized head, 1 pass)", "wam_abl_direct", ("direct", 128, 2)),
            ("WAM-direct + CEM 32 x 1 polish", "wam_abl_direct_cem", ("direct_cem", 128, 2))]
    out = ["| action selection | " + " | ".join(SHORT[t] for t in GOTO) + " | plan latency mean / p95 (ms) |",
           "|---|" + "---|" * (len(GOTO) + 1)]
    for name, arm, key in rows:
        cells = [cell(tally(root, arm, t), 20) for t in GOTO]
        l = lat.get(key)
        if l is None and key[0] == "direct_cem":
            l = lat.get(("direct_cem", 32, 1))
        lt = f"{l[0]:.0f} / {l[1]:.0f}" if l else "—"
        out.append(f"| {name} | " + " | ".join(cells) + f" | {lt} |")
    out.append("")
    out.append("Full sensing, 20 episodes per block (the deployed 128 x 2 row is the main-table block, 40 episodes). "
               "Latency is the server's own per-act planning time on a shared RTX 4090 (mean of the logged 50-act "
               "means). WAM-direct emits the 0.5 s PWM sequence in one forward pass of a 176 k-parameter head on the "
               "frozen core; it was distilled from a 512 x 3 offline teacher plus the imagined cost, so it is the "
               "DreamZero-style amortized reading of the same world model.")
    off = Path("/hy-tmp/results/direct_offline.json")
    if off.exists():
        d = json.loads(off.read_text())
        out.append("")
        out.append(f"Offline, on {d['n']} held-out teacher windows (imagined cost through the same frozen core, lower is "
                   f"better): online MPC 128 x 2 = {d['J_online_mpc_128x2']:.3f}, offline teacher 512 x 3 = "
                   f"{d['J_teacher_512x3']:.3f}, WAM-direct head = {d['J_direct_head']:.3f}; the head is at or below the "
                   f"online planner on {100 * d['frac_head_le_online']:.0f} % of windows, at {d['latency_ms_head']:.0f} ms "
                   f"vs {d['latency_ms_mpc']:.0f} ms per decision on the shared GPU.")
    return "\n".join(out)


def sec_data(root: Path) -> str:
    rows = [("USIM demos only (12.6 h, repeated task demonstrations)", "wam_abl_usim_only"),
            ("play only (3.3 h non-repetitive: OU exploration + scene collection + deployment)", "wam_abl_play_only"),
            ("both (deployed core)", "wam_full")]
    out = ["| training data of the core | " + " | ".join(SHORT[t] for t in GOTO) + " |", "|---|---|---|"]
    for name, arm in rows:
        out.append(f"| {name} | " + " | ".join(cell(tally(root, arm, t), 20) for t in GOTO) + " |")
    out.append("")
    out.append("Same architecture (2.4 M), same 8-epoch recipe, same planner and goals; only the dynamics data "
               "differs. Full sensing, 20 episodes per block. `*` = block still filling.")
    return "\n".join(out)


def sec_scale(root: Path, hist_dir: Path) -> str:
    rows = [("0.68 M (hidden 192, d 48)", "wam_abl_small", "scale_small"),
            ("2.4 M (hidden 384, d 96, deployed)", "wam_full", "best_scenes"),
            ("8.9 M (hidden 768, d 192)", "wam_abl_large", "scale_large")]
    out = ["| core size | offline DVL MAE (m/s, USIM test) | " + " | ".join(SHORT[t] for t in GOTO) + " |",
           "|---|---|---|---|"]
    for name, arm, stem in rows:
        mae = "—"
        hp = hist_dir / f"stage1_{stem}_history.json"
        if hp.exists():
            try:
                h = json.loads(hp.read_text())
                vals = [r["eval"]["dvl_mae_ms"] for r in h if "eval" in r and "dvl_mae_ms" in r["eval"]]
                if vals:
                    mae = f"{min(vals):.4f}"
            except Exception:  # noqa: BLE001
                pass
        out.append(f"| {name} | {mae} | " + " | ".join(cell(tally(root, arm, t), 20) for t in GOTO) + " |")
    out.append("")
    out.append("Same data and recipe; width and disturbance-token size scaled. The VLA baseline (GR00T N1.5) has 3 B "
               "parameters, i.e. 1,250x the deployed core.")
    return "\n".join(out)


def sec_fewshot(root: Path) -> str:
    conds = [("full sensing", ""), ("DVL loss at 8 s", "_drop8s")]
    rows = [("U0 (VLA), zero-shot", "u0", "hard_eta0_00", "u0"),
            ("WAM, zero-shot, estimator v1 (before the closed-loop retraining)", "wam", "hard_eta0_00", "v1"),
            ("WAM, zero-shot (deployed: estimator v2)", "wam", "hard_eta0_00", "wam"),
            ("WAM after 30 min of faulted play (core + estimator fine-tuned)", "wam", "adapt_eta0_00", "wam")]
    out = ["| policy on the damaged vehicle (thruster 1 dead) | " +
           " | ".join(f"{SHORT[t]}, {c}" for c, _ in conds for t in GOTO) + " |", "|---|" + "---|" * 4]
    for name, arm, tag, kind in rows:
        cells = []
        for _, suf in conds:
            for t in GOTO:
                if kind == "u0":
                    cells.append(cell(tally(root, f"{arm}_{tag}{suf}", t), 20) if not suf else "n/a")
                elif kind == "v1":
                    if suf:  # v1 dropout blocks were moved to the estimator archive
                        cells.append(cell(tally(root, f"_archive_est1/{arm}_{tag}{suf}__{t}", ""), 20))
                    else:    # full sensing never touches the estimator: same block as the deployed row
                        cells.append(cell(tally(root, f"{arm}_{tag}", t), 20))
                else:
                    cells.append(cell(tally(root, f"{arm}_{tag}{suf}", t), 20))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Thruster 1 (front-left horizontal) actuates at 0 % of its command; nothing in training saw this "
               "fault. The play data are 11 explore-server episodes (no task, no success signal, 34 min = 20.4 k "
               "frames) recorded on the damaged vehicle, converted with the true efficiency vector; the core is "
               "fine-tuned from the deployed checkpoint (2 epochs, OU mix, faulted data x48) and the dead-reckoning "
               "ensemble warm-started from v2 and fine-tuned on it. 20 episodes per block; times are mean "
               "time-to-success. Context: with thruster 1 at 50 % (hard_eta0_05, USIM-Hard table) U0 scored 4/20 and "
               "0/20 while WAM scored 20/20 and 20/20 zero-shot.")
    off = Path(__file__).resolve().parent / "fewshot_offline.json"
    if off.exists():
        z = json.loads(off.read_text())
        k_f = "dead thruster 1 (held-out eval flights)"; k_h = "healthy vehicle (tight-tolerance goto flights)"
        b, a = z["best_scenes"], z["adapt_eta0"]
        out.append("")
        out.append(f"Offline, on the sensor logs of the dead-thruster eval flights themselves (never trained on; {b[k_f]['n_windows']} "
                   f"windows) vs. healthy flights ({b[k_h]['n_windows']}): 0.5 s DVL-prediction MAE deployed core "
                   f"{b[k_f]['dvl_pred_mae_ms']:.4f} m/s faulted vs {b[k_h]['dvl_pred_mae_ms']:.4f} healthy; adapted core "
                   f"{a[k_f]['dvl_pred_mae_ms']:.4f} faulted vs {a[k_h]['dvl_pred_mae_ms']:.4f} healthy. Thruster-efficiency head "
                   f"(1 = nominal) on faulted flights: deployed {b[k_f]['eta_hat_thruster1']:+.2f}, adapted {a[k_f]['eta_hat_thruster1']:+.2f} "
                   f"(on the play windows it was trained on the adapted head reads 0.00).")
        out.append("")
        out.append("**Reading.** The deployed world model already absorbs most of the fault through its history token "
                   "(faulted-flight prediction error only 40 % above healthy), which is why full-sensing success is high "
                   "zero-shot; what the fault costs is time (91 s vs 22 s on goto charge): the vehicle yaws at ~0.44 rad/s "
                   "because the fixed mixer-based yaw/attitude decoupling downstream of the planner assumes the nominal "
                   "thruster geometry and fights the asymmetric thrust. 34 min of play data do not move that: the adapted "
                   "core is no better on task flights (play-driving and task-driving are different regimes), and closed-loop "
                   "success is unchanged. Keeping the planner's own yaw component instead of stripping it was tested in "
                   "imagination and makes the predicted spin worse for both cores, so the fix is not a flag but moving yaw "
                   "compensation into the planned action space (learned mixer) -- left as future work. The result stands "
                   "as an honest negative for few-shot embodiment adaptation in this architecture.")
    return "\n".join(out)


def sec_direct_closed_loop(root: Path) -> str:
    tasks = ["goto_charge_station", "goto_water_tower", "inspect_pipeline_pool", "scan_ship_ancient"]
    drop = {"goto_charge_station": 8, "goto_water_tower": 8, "inspect_pipeline_pool": 40, "scan_ship_ancient": 40}
    out = ["| task | condition | sampling MPC (deployed) | WAM-direct | WAM-direct + CEM 32 x 1 |", "|---|---|---|---|---|"]
    for t in tasks:
        for cond in ("full", "drop"):
            if cond == "full":
                mpc = tally(root, "wam_full", t); d = tally(root, "wam_abl_direct", t); dc = tally(root, "wam_abl_direct_cem", t)
                cname = "full sensing"
            else:
                mpc = tally(root, f"wam_drop{drop[t]}s_zero", t); d = tally(root, f"wam_abl_direct_drop{drop[t]}s", t); dc = None
                cname = f"DVL loss at {drop[t]} s"
            out.append(f"| {SHORT[t]} | {cname} | {cell(mpc)} | {cell(d, 20)} | {cell(dc, 20) if cond == 'full' and t.startswith('goto') else 'n/a'} |")
    out.append("")
    out.append("Same core, same goals, same downstream controller (yaw / attitude decoupling, trust gate, dead "
               "reckoning); only the 0.5 s action decision differs: hundreds of imagined rollouts per act vs. one "
               "forward pass. MPC column = main-table blocks (40 / 20 episodes); direct columns 20 episodes.")
    return "\n".join(out)


def sec_estimator(root: Path, eval_json: Path) -> str:
    """Dead-reckoning estimator retrained on closed-loop deployment logs (v2) vs. the original (v1):
    offline DR metrics on real blind-flight recordings + the re-run main-table dropout column."""
    loco = [("goto_charge_station", 8), ("goto_water_tower", 8), ("inspect_pipeline_pool", 40), ("inspect_pipeline_sea", 40),
            ("scan_ship_ancient", 40), ("scan_ship_modern", 40), ("follow_boat", 20)]
    out = []
    if eval_json.exists():
        z = json.loads(eval_json.read_text())
        out += ["Offline, on the sensor logs of real blind-flight episodes (the estimator only sees IMU + commanded PWM):", "",
                "| recordings | n | v1 |v| bias (m/s) | v1 DR error @30 s (m) | v2 |v| bias (m/s) | v2 DR error @30 s (m) |",
                "|---|---|---|---|---|---|"]
        for name, s in z["sets"].items():
            v1, v2 = s["v1"], s["v2"]
            out.append(f"| {name} | {s['n']} | {v1['bias']:+.3f} | {v1['dr30']:.2f} | {v2['bias']:+.3f} | {v2['dr30']:.2f} |")
        out.append("")
    out += ["Closed loop, WAM (privileged) and U0+WAM fallback under DVL loss, same code, only the estimator changed:", "",
            "| task | WAM, estimator v1 | WAM, estimator v2 (deployed) | fallback, v1 | fallback, v2 (deployed) |", "|---|---|---|---|---|"]
    for t, d in loco:
        cells = []
        for arm in ("wam", "fallback"):
            v1 = tally(root, f"_archive_est1/{arm}_drop{d}s_zero__{t}", "")
            v2 = tally(root, f"{arm}_drop{d}s_zero", t)
            if v1 is None:  # block not re-run yet: the live block still IS the v1 result
                v1, v2 = v2, None
            cells += [cell(v1), cell(v2)]
        out.append(f"| {t.replace('_', ' ')} (DVL loss at {d} s) | " + " | ".join(cells) + " |")
    out.append("")
    out.append("v1 = ensemble trained on exploration + scene-collection + deployment recordings with the DVL alive; v2 = the "
               "same recipe plus the sensor logs of every archived closed-loop dropout episode (the vehicle's own blind "
               "flights, labelled with the privileged true velocity) -- DAgger applied to the estimator instead of the "
               "policy. The v1 blocks are kept under `_archive_est1/`; the main tables use v2 wherever the re-run has "
               "landed. `—` in a v2 cell = re-run still queued.")
    return "\n".join(out)


def sec_p5(path: Path) -> str:
    if not path.exists():
        return "_p5_failure_auroc.py has not been run_"
    z = json.loads(path.read_text())
    feats = z["features"]
    names = {"imu_res": "core prediction error, IMU channels (0.5 s rollout of the VLA's commands)",
             "epi_sigma_blind": "dead-reckoning ensemble disagreement while blind (trust-gate statistic)",
             "dvl_res_sighted": "core DVL prediction error while the DVL is alive",
             "dr_err_blind": "dead-reckoning error vs. true velocity while blind (privileged)",
             "speed": "baseline: mean speed", "pwm_abs": "baseline: mean |PWM|", "gyro_abs": "baseline: mean |gyro|"}
    out = [f"{z['n_episodes']} U0 manipulation episodes under DVL loss at 30 s (11 tasks), {z['n_fail']} failures "
           f"({100 * z['n_fail'] / z['n_episodes']:.0f} %). Statistics are computed on the first {z['horizon_s']:.0f} s "
           "of each episode, i.e. before the outcome (U0 successes finish at 50-70 s, timeouts at 185 s).", "",
           "| statistic | AUROC pooled | AUROC after per-task z-scoring | per-task median |", "|---|---|---|---|"]
    for k, label in names.items():
        if k in feats:
            v = feats[k]
            out.append(f"| {label} | {v['auroc_pooled']:.2f} | {v['auroc_task_z']:.2f} | {v['auroc_per_task_median']:.2f} |")
    out.append("")
    out.append("**Negative result.** Nothing the dynamics world model can see -- how surprising the VLA's commands "
               "are for the hull, how uncertain the estimator is -- separates U0's successful grasps from its "
               "failures (AUROC ~0.5, chance). U0's dropout failures on manipulation are decided by the arm and the "
               "wrist camera, not by vehicle dynamics; this is also why the fallback layer is mode-aware and leaves "
               "grasping to U0 when the DVL dies. A whole-episode version (185 s) reaches AUROC 0.83 for *low* IMU "
               "activity, which merely says that timed-out episodes hover -- detection after the fact, not prediction.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--latency-log", default="/hy-tmp/logs/u0eval/wam_plan_latency.log")
    ap.add_argument("--hist-dir", default="/hy-tmp/logs/uwam")
    ap.add_argument("--p5", default=str(Path(__file__).resolve().parent / "p5_failure_auroc.json"))
    ap.add_argument("--estimator-eval", default=str(Path(__file__).resolve().parent / "estimator_eval.json"))
    ap.add_argument("--section", default="all",
                    choices=["all", "planner", "data", "scale", "fewshot", "direct", "estimator", "p5"])
    args = ap.parse_args()
    root = Path(args.eval_runs)
    lat = latency_table(Path(args.latency_log))
    secs = {"planner": ("### A. Planner budget and amortization", sec_planner(root, lat)),
            "data": ("### B. Repetitive demonstrations vs. non-repetitive play", sec_data(root)),
            "scale": ("### C. Model scale", sec_scale(root, Path(args.hist_dir))),
            "fewshot": ("### D. Few-shot adaptation to a damaged embodiment", sec_fewshot(root)),
            "direct": ("### E. WAM-direct in closed loop", sec_direct_closed_loop(root)),
            "estimator": ("### F. Dead-reckoning estimator: DAgger on the vehicle's own blind flights (v1 -> v2)",
                          sec_estimator(root, Path(args.estimator_eval))),
            "p5": ("### G. Can the world model predict the VLA's failures? (USIM-Hard P5)", sec_p5(Path(args.p5)))}
    keys = list(secs) if args.section == "all" else [args.section]
    print("\n\n".join(f"{secs[k][0]}\n\n{secs[k][1]}" for k in keys))


if __name__ == "__main__":
    main()
