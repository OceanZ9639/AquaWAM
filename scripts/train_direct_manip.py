#!/usr/bin/env python3
"""Train the manipulation WAM-direct head (uwam/direct_manip.py) by behaviour cloning on the system's own
successful grasp rollouts (OU-schema npz from recordings_to_ou.py: dyn [T,35], pwm [T,13]).

  --data   comma-separated OU dirs (e.g. /hy-tmp/data/manip_direct_local,/hy-tmp/data/manip_direct_box2)
  --runs   eval_runs roots holding the results.csv of those recordings (success filter), comma-separated
Episode-level validation split. Reports per-channel action MAE on the held-out episodes (PWM in raw units,
jaw timing accuracy = fraction of held-out steps where sign(jaw_cmd - 0.0075) is predicted right).
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwam.data import load_ou_split  # noqa: E402
from uwam.direct_manip import ACT_DIM, STATE_DIM, DirectManip  # noqa: E402


def success_lookup(runs):
    ok = {}
    for root in runs:
        for f in Path(root).glob("*/*/results.csv"):
            arm, task = f.parts[-3], f.parts[-2]
            for r in csv.reader(open(f)):
                if r and r[0].isdigit():
                    ok[(arm, task, int(r[0]))] = r[1].strip() == "success"
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="/hy-tmp/u0env/dataset/eval_runs")
    ap.add_argument("--successes-only", type=int, default=1)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="/hy-tmp/models/uwam/direct_manip.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    succ = success_lookup([r for r in args.runs.split(",") if r])
    eps = []
    for d in args.data.split(","):
        for e in load_ou_split(Path(d), use_arm=True, use_object=True):
            m = re.match(r"(.+?)__(.+?)__episode(\d+)$", e.task) if isinstance(e.task, str) else None
            if m is None:
                # task field may not carry the file name; fall back to accepting everything
                eps.append(e); continue
            key = (m.group(1), m.group(2), int(m.group(3)))
            if args.successes_only and not succ.get(key, False):
                continue
            eps.append(e)
    print(f"{len(eps)} episodes after success filter; frames {sum(len(e.dyn) for e in eps)}", flush=True)
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); nv = max(1, int(len(eps) * args.val_frac))
    val_eps, tr_eps = [eps[i] for i in idx[:nv]], [eps[i] for i in idx[nv:]]

    def windows(episodes):
        S, A, Y = [], [], []
        for e in episodes:
            dyn, pwm = e.dyn.astype(np.float32), e.pwm.astype(np.float32)
            T = len(dyn)
            for t in range(args.L, T - args.K):
                S.append(dyn[t - args.L:t]); A.append(pwm[t - args.L:t]); Y.append(pwm[t:t + args.K])
        return np.stack(S), np.stack(A), np.stack(Y)
    S, A, Y = windows(tr_eps); Sv, Av, Yv = windows(val_eps)
    print(f"train windows {len(S)}  val windows {len(Sv)}", flush=True)
    model = DirectManip(L=args.L, K=args.K, hidden=args.hidden).to(dev)
    allS = S.reshape(-1, STATE_DIM); allA = np.concatenate([A.reshape(-1, ACT_DIM), Y.reshape(-1, ACT_DIM)])
    model.s_mu.copy_(torch.as_tensor(allS.mean(0))); model.s_sd.copy_(torch.as_tensor(np.maximum(allS.std(0), 1e-3)))
    model.a_mu.copy_(torch.as_tensor(allA.mean(0))); model.a_sd.copy_(torch.as_tensor(np.maximum(allA.std(0), 1e-3)))
    # the jaw channel is nearly binary with a tiny scale: weight it so timing is learned
    w_ch = torch.ones(ACT_DIM, device=dev); w_ch[8] = 5.0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    f = lambda a: torch.as_tensor(a, device=dev)
    St, At, Yt = f(S), f(A), f(Y); Svt, Avt, Yvt = f(Sv), f(Av), f(Yv)
    best, hist, t0 = float("inf"), [], time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(St), device=dev); tot = 0.0
        for i in range(0, len(St), args.batch):
            b = perm[i:i + args.batch]
            pred = model(St[b], At[b])
            loss = (((pred - Yt[b]) / model.a_sd) ** 2 * w_ch).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += float(loss) * len(b)
        sched.step(); model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Svt[i:i + 2048], Avt[i:i + 2048]) for i in range(0, len(Svt), 2048)])
            mae = (pv - Yvt).abs().mean((0, 1)).cpu().numpy()
            jaw_ok = float(((pv[..., 8] > 0.0075) == (Yvt[..., 8] > 0.0075)).float().mean())
            vloss = float((((pv - Yvt) / model.a_sd) ** 2 * w_ch).mean())
        rec = {"epoch": ep + 1, "train_loss": tot / len(St), "val_loss": vloss, "val_pwm_mae": float(mae[:8].mean()),
               "val_joint_mae_rad": float(mae[9:13].mean()), "val_jaw_timing_acc": jaw_ok, "minutes": (time.time() - t0) / 60}
        hist.append(rec); print(json.dumps(rec), flush=True)
        if vloss < best:
            best = vloss
            torch.save({"arch": {"L": args.L, "K": args.K, "hidden": args.hidden}, "state_dict": model.state_dict(),
                        "history": hist, "n_episodes": len(eps), "val_episodes": nv}, args.out)
    print(f"saved {args.out}; best val loss {best:.4f}")


if __name__ == "__main__":
    main()
