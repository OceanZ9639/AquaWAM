#!/usr/bin/env python3
"""Deployment-side perception goal head: frames + proprio -> body-frame goal pose.

Wraps frozen DINOv2-base + the trained regression head behind one predict()
call, with EMA smoothing and outlier rejection across inference ticks. The
output is the same quantity the head was trained on: the expert planner's
staged target pose in the CURRENT vehicle body frame [dx dy dz, r p yaw].

Run as a script for an offline shadow replay over USIM test episodes: decodes
real videos, runs the exact deployment path, and reports per-stage errors --
no simulator needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ENCODER_DIR = "/hy-tmp/models/hf_cache/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"
HEAD_CKPT = "/hy-tmp/models/uwam/percept_head_base.pt"
NAV_HEAD_CKPT = "/hy-tmp/models/uwam/percept_nav_head.pt"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
N_TASKS = 9
# instruction phrase -> USIM task_index (train/meta/tasks.jsonl)
TASK_PHRASES = (
    ("charge station", 0), ("pick up the pipe", 1), ("blue cylinder", 2),
    ("scan the ship", 3), ("inspect the pipeline", 4), ("follow the boat", 5),
    ("transfer it", 7), ("red cylinder", 6), ("water tower", 8),
)


def task_index_for(instruction: str) -> int:
    low = instruction.lower()
    if "transfer it" in low:
        return 7
    for phrase, idx in TASK_PHRASES:
        if phrase in low:
            return idx
    return 0


class PerceptGoal:
    def __init__(self, head_ckpt: str = HEAD_CKPT, encoder_dir: str = ENCODER_DIR,
                 device: str | None = None, ema: float = 0.45, jump_gate_m: float = 0.6,
                 nav_head_ckpt: str | None = NAV_HEAD_CKPT):
        from transformers import AutoModel

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_head import Head  # noqa: PLC0415

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = AutoModel.from_pretrained(encoder_dir).to(self.device).half().eval()
        ck = torch.load(head_ckpt, map_location=self.device, weights_only=False)
        self.head = Head(ck["in_dim"]).to(self.device).eval()
        self.head.load_state_dict(ck["model"])
        sn = ck.get("state_norm")
        self.state_mean = np.asarray(sn[0], np.float32) if sn else None
        self.state_std = np.asarray(sn[1], np.float32) if sn else None
        # navigation head (goto / scan / inspect / follow): same frozen encoder, own MLP trained
        # on the locomotion tasks with a bearing loss; predicts the expert's current waypoint in
        # the body frame. Optional: absent -> nav stays privileged (waypoint file).
        self.nav_head = None
        self.nav_state_mean = self.nav_state_std = None
        if nav_head_ckpt and Path(nav_head_ckpt).exists():
            nk = torch.load(nav_head_ckpt, map_location=self.device, weights_only=False)
            self.nav_head = Head(nk["in_dim"]).to(self.device).eval()
            self.nav_head.load_state_dict(nk["model"])
            nsn = nk.get("state_norm")
            self.nav_state_mean = np.asarray(nsn[0], np.float32) if nsn else None
            self.nav_state_std = np.asarray(nsn[1], np.float32) if nsn else None
        self.ema_alpha = ema
        self.jump_gate_m = jump_gate_m
        self.reset()

    def reset(self):
        self.smoothed = None
        self.grip_p = 0.0
        self.nav_smoothed = None
        self.nav_last_yaw = None

    @torch.inference_mode()
    def predict_nav(self, ego_u8: np.ndarray, wrist_u8: np.ndarray, joint_pos: np.ndarray,
                    pressure: float, dvl_h: float, instruction: str, yaw: float | None = None):
        """Navigation goal: EMA-smoothed body-frame [dx dy dz r p yaw] of the expert's current
        waypoint, plus the raw (unsmoothed) prediction for confidence checks. `yaw` (vehicle heading)
        lets the smoother rotate its previous estimate into the current body frame first, so a
        turning hull does not smear the goal direction (body-frame EMA alone is wrong in turns)."""
        ti = task_index_for(instruction)
        onehot = torch.zeros(1, N_TASKS, device=self.device)
        onehot[0, ti] = 1.0
        st = np.concatenate([np.asarray(joint_pos, np.float32).reshape(-1)[:5],
                             [np.float32(pressure)], [np.float32(dvl_h)]])
        if self.nav_state_mean is not None:
            st = (st - self.nav_state_mean) / self.nav_state_std
        stt = torch.from_numpy(st.astype(np.float32)).unsqueeze(0).to(self.device)
        x = torch.cat([self._feat(ego_u8), self._feat(wrist_u8), onehot, stt], dim=-1)
        raw = self.nav_head(x)[0].cpu().numpy().astype(np.float64)[:6]
        if self.nav_smoothed is not None and yaw is not None and self.nav_last_yaw is not None:
            dpsi = (yaw - self.nav_last_yaw + np.pi) % (2 * np.pi) - np.pi
            c, s_ = np.cos(-dpsi), np.sin(-dpsi)
            sx, sy = self.nav_smoothed[0], self.nav_smoothed[1]
            self.nav_smoothed[0], self.nav_smoothed[1] = c * sx - s_ * sy, s_ * sx + c * sy
            self.nav_smoothed[5] = (self.nav_smoothed[5] - dpsi + np.pi) % (2 * np.pi) - np.pi
        self.nav_last_yaw = yaw
        if self.nav_smoothed is None:
            self.nav_smoothed = raw.copy()
        else:
            # goal 2-3 m out: smooth (the bearing is what steers); inside 1 m the range shrinks
            # fast, so freshness matters more than smoothness
            a = 0.35 if np.linalg.norm(raw[:3]) > 1.0 else 0.7
            self.nav_smoothed[:3] = (1 - a) * self.nav_smoothed[:3] + a * raw[:3]
            d = np.arctan2(np.sin(raw[3:] - self.nav_smoothed[3:]), np.cos(raw[3:] - self.nav_smoothed[3:]))
            self.nav_smoothed[3:] = self.nav_smoothed[3:] + a * d
        return self.nav_smoothed.copy(), raw

    @torch.inference_mode()
    def _feat(self, img_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(img_u8)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
        h = self.encoder(pixel_values=x.half()).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1).float()  # [1, 1536]

    @torch.inference_mode()
    def predict(self, ego_u8: np.ndarray, wrist_u8: np.ndarray,
                joint_pos: np.ndarray, pressure: float, dvl_h: float,
                instruction: str):
        """EMA-smoothed 6-d goal + gripper-close probability."""
        ti = task_index_for(instruction)
        onehot = torch.zeros(1, N_TASKS, device=self.device)
        onehot[0, ti] = 1.0
        st = np.concatenate([np.asarray(joint_pos, np.float32).reshape(-1)[:5],
                             [np.float32(pressure)], [np.float32(dvl_h)]])
        if self.state_mean is not None:
            st = (st - self.state_mean) / self.state_std
        stt = torch.from_numpy(st.astype(np.float32)).unsqueeze(0).to(self.device)
        x = torch.cat([self._feat(ego_u8), self._feat(wrist_u8), onehot, stt], dim=-1)
        out = self.head(x)[0].cpu().numpy().astype(np.float64)
        raw = out[:6]
        grip_p = float(1.0 / (1.0 + np.exp(-out[6]))) if len(out) > 6 else 0.0
        if self.smoothed is None:
            self.smoothed = raw.copy()
            self.grip_p = grip_p
        else:
            # endgame needs freshness more than smoothness: the true delta
            # shrinks fast on final approach and EMA lag causes overshoot
            a = self.ema_alpha if np.linalg.norm(raw[:3]) > 0.25 else 0.75
            # position jump gate: a prediction that teleports the goal is more
            # likely a perception glitch than a real staged-target switch, so it
            # only moves the estimate a little; consistent jumps win in ~3 ticks
            if np.linalg.norm(raw[:3] - self.smoothed[:3]) > self.jump_gate_m:
                a *= 0.35
            self.smoothed[:3] = (1 - a) * self.smoothed[:3] + a * raw[:3]
            d = np.arctan2(np.sin(raw[3:] - self.smoothed[3:]), np.cos(raw[3:] - self.smoothed[3:]))
            self.smoothed[3:] = self.smoothed[3:] + a * d
            self.grip_p = 0.5 * self.grip_p + 0.5 * grip_p
        return self.smoothed.copy(), self.grip_p


class PerceptGoalE2E(PerceptGoal):
    """Same interface as PerceptGoal, but locomotion goals come from the end-to-end model
    (percept/train_e2e.py): fine-tuned DINOv2 over both views with attention pooling, predicting the
    expert's 3 s displacement (disp3) and the current node (wp6). predict_nav() returns a 6-vector
    whose first three entries are the DISPLACEMENT (the planner treats it as the goal offset) and
    whose yaw entry is the node's look-at heading (what scan / inspect are judged on).
    The grasp head (frozen features) is kept for manipulation goals."""

    kind = "e2e"

    def __init__(self, e2e_ckpt: str = "/hy-tmp/models/uwam/percept_e2e.pt", **kw):
        super().__init__(nav_head_ckpt=None, **kw)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_e2e import PerceptE2E  # noqa: PLC0415
        ck = torch.load(e2e_ckpt, map_location=self.device, weights_only=False)
        self.e2e = PerceptE2E(ck.get("encoder", "base"), freeze_blocks=ck.get("freeze_blocks", 4)).to(self.device).eval()
        self.e2e.load_state_dict(ck["model"])
        sn = ck.get("state_norm")
        self.e2e_state_mean = np.asarray(sn[0], np.float32)
        self.e2e_state_std = np.asarray(sn[1], np.float32)
        self.nav_head = self.e2e   # non-None: the server routes locomotion goals through predict_nav
        print(f"[percept] e2e model {e2e_ckpt} (epoch {ck.get('epoch')})", flush=True)

    @torch.inference_mode()
    def _e2e_img(self, img_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(img_u8)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)

    @torch.inference_mode()
    def predict_nav(self, ego_u8, wrist_u8, joint_pos, pressure, dvl_h, instruction, yaw=None):
        ti = task_index_for(instruction)
        st = np.concatenate([np.asarray(joint_pos, np.float32).reshape(-1)[:5],
                             [np.float32(pressure)], [np.float32(dvl_h)]])
        st = (st - self.e2e_state_mean) / self.e2e_state_std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.e2e(self._e2e_img(ego_u8), self._e2e_img(wrist_u8),
                           torch.from_numpy(st.astype(np.float32)).unsqueeze(0).to(self.device),
                           torch.tensor([ti], device=self.device))
        out = out[0].float().cpu().numpy().astype(np.float64)
        disp, wp = out[:3], out[3:9]
        raw = np.array([disp[0], disp[1], disp[2], 0.0, 0.0, wp[5]])
        if self.nav_smoothed is not None and yaw is not None and self.nav_last_yaw is not None:
            dpsi = (yaw - self.nav_last_yaw + np.pi) % (2 * np.pi) - np.pi
            c, s_ = np.cos(-dpsi), np.sin(-dpsi)
            sx, sy = self.nav_smoothed[0], self.nav_smoothed[1]
            self.nav_smoothed[0], self.nav_smoothed[1] = c * sx - s_ * sy, s_ * sx + c * sy
            self.nav_smoothed[5] = (self.nav_smoothed[5] - dpsi + np.pi) % (2 * np.pi) - np.pi
        self.nav_last_yaw = yaw
        if self.nav_smoothed is None:
            self.nav_smoothed = raw.copy()
        else:
            a = 0.5   # the 3 s intent is already a smooth quantity
            self.nav_smoothed[:3] = (1 - a) * self.nav_smoothed[:3] + a * raw[:3]
            d = np.arctan2(np.sin(raw[3:] - self.nav_smoothed[3:]), np.cos(raw[3:] - self.nav_smoothed[3:]))
            self.nav_smoothed[3:] = self.nav_smoothed[3:] + a * d
        return self.nav_smoothed.copy(), raw


def _decode(path: Path, stride: int) -> np.ndarray:
    import av

    c = av.open(str(path))
    frames = [f.to_ndarray(format="rgb24") for i, f in enumerate(c.decode(video=0)) if i % stride == 0]
    c.close()
    return np.stack(frames)


def shadow_replay(split: str, episodes: list[int], stride: int = 3):
    """Replay real episodes through the deployment path; report goal errors."""
    import pyarrow.parquet as pq

    root = Path(f"/hy-tmp/data/usim/{split}")
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(root / "meta" / "tasks.jsonl")}
    pg = PerceptGoal()
    out = {}
    for ei in episodes:
        chunk = f"chunk-{ei // 1000:03d}"
        p = root / "data" / chunk / f"episode_{ei:06d}.parquet"
        t = pq.read_table(p)
        tp = np.array([np.asarray(x, np.float32) for x in t.column("target_pos").to_pylist()])
        st = np.array([np.asarray(x, np.float32) for x in t.column("observation.state").to_pylist()])
        ti = int(t.column("task_index")[0].as_py())
        instr = tasks[ti]
        ego = _decode(root / "videos" / chunk / "observation.images.ego" / f"episode_{ei:06d}.mp4", stride)
        wrist = _decode(root / "videos" / chunk / "observation.images.wrist" / f"episode_{ei:06d}.mp4", stride)
        k = min(len(ego), len(wrist), len(tp[::stride]))
        lab = tp[::stride][:k]
        sts = st[::stride][:k]
        pg.reset()
        errs, close_errs = [], []
        for i in range(k):
            if np.abs(lab[i]).sum() < 1e-8:
                continue
            goal, _gp = pg.predict(ego[i], wrist[i], sts[i, 0:5], float(sts[i, 27]), float(sts[i, 28]), instr)
            e = float(np.linalg.norm(goal[:3] - lab[i, :3]))
            errs.append(e)
            if np.linalg.norm(lab[i, :3]) < 0.5:
                close_errs.append(e)
        out[ei] = {
            "task": instr, "frames": len(errs),
            "pos_mae": float(np.mean(errs)) if errs else None,
            "close_mae": float(np.mean(close_errs)) if close_errs else None,
            "close_p90": float(np.percentile(close_errs, 90)) if close_errs else None,
        }
        print(ei, json.dumps(out[ei]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--episodes", default="", help="comma ids; empty = auto-pick grasp eps")
    args = ap.parse_args()
    if args.episodes:
        eps = [int(x) for x in args.episodes.split(",")]
    else:
        root = Path(f"/hy-tmp/data/usim/{args.split}/meta/episodes.jsonl")
        eps = []
        for i, line in enumerate(open(root)):
            d = json.loads(line)
            if any(w in d["tasks"][0].lower() for w in ("pick", "transfer")) and len(eps) < 6:
                eps.append(d["episode_index"])
    r = shadow_replay(args.split, eps)
    ok = [v for v in r.values() if v["close_mae"] is not None]
    if ok:
        print(f"\nSHADOW: close MAE mean {np.mean([v['close_mae'] for v in ok]):.4f} m over {len(ok)} eps")
    print("SHADOW_DONE")
