#!/usr/bin/env python3
"""Decode USIM videos and extract frozen DINOv2-small features + target_pos labels.

One pass per split: PyAV decodes the AV1 streams, frames go through the frozen
encoder in fp16 batches, features + labels land in a single npz. The head
trains on these features; the encoder is never fine-tuned.

Label semantics (tools/dataprocess/data_process.py): target_pos = the expert
planner's current /bluerov2/target_pose transformed into the vehicle body
frame, [dx, dy, dz, roll, pitch, yaw]. Frames where the expert had not yet
published a target carry an exact-zero label and are marked invalid.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ENCODERS = {
    "small": "/hy-tmp/models/hf_cache/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "base": "/hy-tmp/models/hf_cache/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# proprio columns that locate the cameras: joint_pos (wrist cam rides the arm),
# pressure + altitude (vertical position cues)
STATE_COLS = list(range(0, 5)) + [27, 28]


def decode_sampled(path: Path, stride: int) -> np.ndarray | None:
    try:
        container = av.open(str(path))
        frames = []
        for i, f in enumerate(container.decode(video=0)):
            if i % stride == 0:
                frames.append(f.to_ndarray(format="rgb24"))
        container.close()
        return np.stack(frames) if frames else None
    except Exception as e:  # noqa: BLE001
        print(f"[decode] {path.name}: {e}", flush=True)
        return None


@torch.inference_mode()
def featurize(model, frames_u8: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    outs = []
    for i in range(0, len(frames_u8), batch):
        x = torch.from_numpy(frames_u8[i:i + batch]).to(device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        out = model(pixel_values=x.half())
        h = out.last_hidden_state  # [B, 1+N, 384]
        feat = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1)  # CLS + patch mean -> 768
        outs.append(feat.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth frame")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--encoder", choices=list(ENCODERS), default="small")
    ap.add_argument("--out", default="/hy-tmp/data/percept_probe")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoModel

    root = Path(f"/hy-tmp/data/usim/{args.split}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(ENCODERS[args.encoder]).to(device).half().eval()

    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if args.max_episodes:
        parquets = parquets[: args.max_episodes]
    print(f"{args.split}: {len(parquets)} episodes, stride={args.stride}", flush=True)

    ego_f, wrist_f, targets, task_idx, ep_idx, valid, states = [], [], [], [], [], [], []
    t0 = time.time()
    n_done = 0
    for p in parquets:
        t = pq.read_table(p)
        tp = np.array([np.asarray(x, np.float32) for x in t.column("target_pos").to_pylist()])
        st = np.array([np.asarray(x, np.float32) for x in t.column("observation.state").to_pylist()])
        st = st[:, STATE_COLS]
        ti = int(t.column("task_index")[0].as_py())
        ei = int(t.column("episode_index")[0].as_py())
        chunk = p.parent.name
        vids = {}
        okv = True
        for key in ("ego", "wrist"):
            vp = root / "videos" / chunk / f"observation.images.{key}" / f"{p.stem}.mp4"
            arr = decode_sampled(vp, args.stride)
            if arr is None:
                okv = False
                break
            vids[key] = arr
        if not okv:
            continue
        k = min(len(vids["ego"]), len(vids["wrist"]), len(tp[:: args.stride]))
        if k == 0:
            continue
        lab = tp[:: args.stride][:k]
        ego_f.append(featurize(model, vids["ego"][:k], device))
        wrist_f.append(featurize(model, vids["wrist"][:k], device))
        targets.append(lab)
        states.append(st[:: args.stride][:k])
        task_idx.append(np.full(k, ti, np.int64))
        ep_idx.append(np.full(k, ei, np.int64))
        valid.append((np.abs(lab).sum(axis=1) > 1e-8))
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(parquets)} eps, {sum(len(x) for x in targets)} frames, "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.encoder == "small" else f"_{args.encoder}"
    path = out / f"{args.split}_feats{suffix}.npz"
    np.savez_compressed(
        path,
        ego=np.concatenate(ego_f), wrist=np.concatenate(wrist_f),
        target=np.concatenate(targets).astype(np.float32),
        state=np.concatenate(states).astype(np.float32),
        task=np.concatenate(task_idx), episode=np.concatenate(ep_idx),
        valid=np.concatenate(valid),
    )
    n = sum(len(x) for x in targets)
    print(f"saved {path}  frames={n} eps={n_done} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
