#!/usr/bin/env python3
"""Per-call latency of a LeRobot-family baseline policy (pi0.5 / X-VLA / SmolVLA / FastWAM fine-tunes)
through the same wrapper the evaluation uses (lerobot_policy_server.Policy.act): synthetic observation of
the production shapes (2 x 240x320 RGB, 29-d proprio, task string) -> one 16-step action chunk.
  python3 bench_lerobot_latency.py <ckpt_dir> [repeats] [out.json]"""
import json, os, platform, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1") if os.environ.get("HF_HUB_OFFLINE") is None else None
from lerobot_policy_server import Policy  # noqa: E402

ckpt = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20; out = sys.argv[3] if len(sys.argv) > 3 else ""
rng = np.random.default_rng(0)
obs = {
    "video.ego": rng.integers(0, 255, (1, 240, 320, 3), dtype=np.uint8),
    "video.wrist": rng.integers(0, 255, (1, 240, 320, 3), dtype=np.uint8),
    "state.joint_pos": np.zeros((1, 5), np.float32), "state.pwm": np.zeros((1, 8), np.float32),
    "state.joint_v": np.zeros((1, 5), np.float32), "state.dvl_v": np.zeros((1, 3), np.float32),
    "state.imu_av": np.zeros((1, 3), np.float32), "state.imu_la": np.zeros((1, 3), np.float32),
    "state.pressure": np.zeros((1, 1), np.float32), "state.dvl_h": np.zeros((1, 1), np.float32),
    "annotation.human.action.task_description": ["Go to the charging station"],
}
t0 = time.time()
pol = Policy(ckpt, device="cuda")
load_s = time.time() - t0
n_params = sum(p.numel() for p in pol.policy.parameters())
mem_gb = torch.cuda.memory_allocated() / 2**30
for _ in range(3):
    pol.act(obs)
torch.cuda.synchronize()
ts = []
for _ in range(n):
    torch.cuda.synchronize(); t = time.perf_counter(); pol.act(obs); torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t))
a = np.asarray(ts)
res = {"ckpt": ckpt, "policy_type": pol.cfg.type, "params_B": n_params / 1e9, "weights_gb": mem_gb, "load_s": load_s,
       "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "n": n,
       "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
       "device": torch.cuda.get_device_name(0), "host": platform.node(), "torch": torch.__version__}
print("[bench-lerobot]", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
if out:
    Path(out).write_text(json.dumps(res, indent=1))
