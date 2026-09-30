"""USIM LeRobot v2.1 loaders and sliding-window datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Cfg, Schema


def _slice(x: np.ndarray, sl: Tuple[int, int]) -> np.ndarray:
    return x[..., sl[0] : sl[1]]


PRESSURE_USIM_TO_PA = 1.0e4


def dyn_state_from_obs(state: np.ndarray, schema: Schema, use_arm: bool = False) -> np.ndarray:
    """Build the dynamics state: DVL, IMU, pressure, altitude, PWM (19-d), plus the
    manipulator's joint positions and velocities appended at the end when use_arm (29-d).

    Arm channels must stay last: mask_dvl and vel_head hard-code the DVL at [0:3].
    """
    # USIM stores pressure as fluid_pressure / 1e4 (tools/dataprocess/data_process.py);
    # the OU / planner / deployment recordings and the policy servers all use raw Pa.
    # Convert here so every training source and the deployed input share one unit.
    # (Checkpoints trained before this fix -- best_ou / best_arm -- saw a mixture:
    # USIM at ~0.3-16 and OU at ~3e4 under one normalization.)
    parts = [
        _slice(state, schema.dvl_v),
        _slice(state, schema.imu_av),
        _slice(state, schema.imu_la),
        _slice(state, schema.pressure) * PRESSURE_USIM_TO_PA,
        _slice(state, schema.dvl_h),
        _slice(state, schema.pwm),
    ]
    if use_arm:
        parts += [_slice(state, schema.joint_pos), _slice(state, schema.joint_v)]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def pwm_from_action(action: np.ndarray, schema: Schema, use_arm: bool = False) -> np.ndarray:
    """Commanded action: 8 thruster PWM, plus 5 joint-angle commands when use_arm."""
    parts = [_slice(action, schema.action_pwm)]
    if use_arm:
        parts.append(_slice(action, schema.action_joint))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def pad_arm_channels(dyn: np.ndarray, pwm: np.ndarray, n_joints: int = 5):
    """Zero-pad locomotion-only episodes (OU / planner collections have no arm) to the
    arm-extended layout so both corpora can be mixed in one training run. Zeros are the
    honest value here: the arm is parked and commanded to hold throughout."""
    T = dyn.shape[0]
    dyn_out = np.concatenate([dyn, np.zeros((T, 2 * n_joints), np.float32)], axis=-1)
    pwm_out = np.concatenate([pwm, np.zeros((T, n_joints), np.float32)], axis=-1)
    return dyn_out.astype(np.float32), pwm_out.astype(np.float32)


def dvl_from_dyn(dyn: np.ndarray) -> np.ndarray:
    return dyn[..., 0:3]


@dataclass
class Episode:
    episode_index: int
    task_index: int
    task: str
    state: np.ndarray          # [T, 29]
    action: np.ndarray         # [T, 13]
    dyn: np.ndarray            # [T, 19], or [T, 29] under use_arm
    pwm: np.ndarray            # [T, 8],  or [T, 13] under use_arm
    target_pos: np.ndarray     # [T, 6]
    timestamp: np.ndarray      # [T]
    split: str
    parquet_path: Path
    ego_video: Optional[Path] = None
    wrist_video: Optional[Path] = None
    regime_index: int = 0
    dt: float = 0.1                            # sample period of this episode's grid
    flow_class: Optional[np.ndarray] = None   # [T] 0 none / 1 const / 2 varying
    fail_class: Optional[np.ndarray] = None   # [T] 0 ok / 1 single / 2 multi
    eta_arr: Optional[np.ndarray] = None      # [T, 8] per-thruster efficiency (regression target)
    rgb_arr: Optional[np.ndarray] = None      # uint8 [T,H,W,3]
    fls_arr: Optional[np.ndarray] = None      # uint8 [T,H,W]


def _video_path(root: Path, split: str, key: str, ep_idx: int, chunk_size: int = 1000) -> Path:
    chunk = ep_idx // chunk_size
    return root / split / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"


def load_tasks(split_dir: Path) -> Dict[int, str]:
    path = split_dir / "meta" / "tasks.jsonl"
    out: Dict[int, str] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["task_index"])] = rec["task"]
    return out


def iter_parquet_files(split_dir: Path) -> List[Path]:
    data = split_dir / "data"
    if not data.exists():
        return []
    return sorted(data.glob("chunk-*/episode_*.parquet"))


def _resize_u8(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize a T-stack of images to hw. Gray [T,H,W] or RGB [T,H,W,3] uint8."""
    import cv2

    H, W = hw
    a = np.asarray(arr)
    if a.ndim == 3:
        out = np.empty((a.shape[0], H, W), np.uint8)
        for i in range(a.shape[0]):
            out[i] = cv2.resize(a[i], (W, H), interpolation=cv2.INTER_AREA)
        return out
    if a.ndim == 4:
        c = a.shape[-1]
        out = np.empty((a.shape[0], H, W, c), np.uint8)
        for i in range(a.shape[0]):
            out[i] = cv2.resize(a[i], (W, H), interpolation=cv2.INTER_AREA)
        return out
    raise ValueError(f"unexpected image stack shape {a.shape}")


def flow_fail_labels(regime_name: str, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame flow (0 none / 1 const / 2 varying) and fail (0 ok / 1 single / 2 multi)."""
    from .sim import REGIMES, regime_at

    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    T = ts.size
    flow = np.zeros(T, np.int64)
    fail = np.zeros(T, np.int64)
    name = regime_name.replace("ou_", "")
    reg = next((r for r in REGIMES if r.name == name), None)
    if reg is None:
        return flow, fail
    for i, t in enumerate(ts):
        cur, eta = regime_at(reg, float(t))
        varying = reg.current_omega > 0 and (reg.change_at is None or t < reg.change_at)
        if varying:
            flow[i] = 2
        elif any(abs(float(c)) > 1e-6 for c in cur):
            flow[i] = 1
        else:
            flow[i] = 0
        n_bad = int(np.sum(np.asarray(eta, dtype=np.float32) < 0.99))
        fail[i] = 0 if n_bad == 0 else (1 if n_bad == 1 else 2)
    return flow, fail


def eta_schedule(regime_name: str, timestamps: np.ndarray) -> np.ndarray:
    """Per-frame thruster efficiency [T, 8] from the named regime's schedule."""
    from .sim import REGIMES, FAULT_MID_REGIMES, regime_at

    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    name = regime_name.replace("ou_", "")
    reg = next((r for r in list(REGIMES) + list(FAULT_MID_REGIMES) if r.name == name), None)
    out = np.ones((ts.size, 8), np.float32)
    if reg is None:
        return out
    for i, t in enumerate(ts):
        _, eta = regime_at(reg, float(t))
        out[i] = np.asarray(eta, np.float32)
    return out


def load_episode(path: Path, split: str, root: Path, tasks: Dict[int, str], schema: Schema,
                 use_arm: bool = False, use_object: bool = False) -> Episode:
    df = pd.read_parquet(path)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    target = np.stack(df["target_pos"].to_numpy()).astype(np.float32)
    ts = df["timestamp"].to_numpy().astype(np.float32)
    ep_idx = int(df["episode_index"].iloc[0])
    task_idx = int(df["task_index"].iloc[0])
    T = int(state.shape[0])
    return Episode(
        episode_index=ep_idx,
        task_index=task_idx,
        task=tasks.get(task_idx, str(task_idx)),
        state=state,
        action=action,
        dyn=(np.concatenate([dyn_state_from_obs(state, schema, use_arm=use_arm),
                             np.zeros((len(state), OBJ_DIM), np.float32)], axis=-1) if use_object
             else dyn_state_from_obs(state, schema, use_arm=use_arm)),
        pwm=pwm_from_action(action, schema, use_arm=use_arm),
        target_pos=target,
        timestamp=ts,
        split=split,
        parquet_path=path,
        ego_video=_video_path(root, split, "observation.images.ego", ep_idx),
        wrist_video=_video_path(root, split, "observation.images.wrist", ep_idx),
        flow_class=np.zeros(T, np.int64),
        fail_class=np.zeros(T, np.int64),
    )


def load_split(root: Path, split: str, schema: Schema, max_episodes: int = 0,
               use_arm: bool = False, use_object: bool = False) -> List[Episode]:
    split_dir = root / split
    tasks = load_tasks(split_dir)
    files = iter_parquet_files(split_dir)
    if max_episodes > 0:
        files = files[:max_episodes]
    eps = [load_episode(p, split, root, tasks, schema, use_arm=use_arm, use_object=use_object) for p in files]
    return eps


class RunningNorm:
    """Simple mean/std normalizer fit on stacked arrays."""

    def __init__(self, eps: float = 1e-6):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "RunningNorm":
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std = np.maximum(self.std, self.eps)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x
        return (x - self.mean) / self.std

    def invert(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x
        return x * self.std + self.mean

    def invert_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        return x * std + mean

    def state_dict(self) -> dict:
        return {"mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    def load_state_dict(self, d: dict) -> None:
        self.mean = None if d["mean"] is None else np.asarray(d["mean"], dtype=np.float32)
        self.std = None if d["std"] is None else np.asarray(d["std"], dtype=np.float32)


class DynamicsWindowDataset(Dataset):
    """
    Windows for the body-dynamics WAM:

        history H_t = {s_{t-L:t}, a_{t-L:t-1}}
        future  a_{t:t+K}, s_{t+K} (and the full K-step trajectory)
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        cfg: Cfg,
        dyn_norm: Optional[RunningNorm] = None,
        pwm_norm: Optional[RunningNorm] = None,
        slow_mode: bool = False,
        fit_norm: bool = True,
    ):
        self.cfg = cfg
        self.L = cfg.model.history_len
        self.K = cfg.model.horizon_dyn
        self.slow_mode = slow_mode
        self.episodes = list(episodes)
        self.index: List[Tuple[int, int]] = []  # (ep_i, t)
        for i, ep in enumerate(self.episodes):
            t_max = ep.dyn.shape[0] - self.K - 1
            t_min = self.L
            for t in range(t_min, t_max):
                self.index.append((i, t))

        all_dyn = np.concatenate([ep.dyn for ep in self.episodes], axis=0) if self.episodes else np.zeros((1, 19), np.float32)
        all_pwm = np.concatenate([ep.pwm for ep in self.episodes], axis=0) if self.episodes else np.zeros((1, 8), np.float32)
        self.dyn_norm = dyn_norm or RunningNorm()
        self.pwm_norm = pwm_norm or RunningNorm()
        if fit_norm:
            self.dyn_norm.fit(all_dyn)
            self.pwm_norm.fit(all_pwm)

    def __len__(self) -> int:
        return len(self.index)

    def _hist_dyn(self, dyn: np.ndarray, t: int) -> np.ndarray:
        h = dyn[t - self.L : t].copy()
        if self.slow_mode:
            # wipe DVL + IMU accel/gyro so the encoder cannot copy short-term kinematics
            h[:, 0:9] = 0.0
        return h

    def __getitem__(self, idx: int) -> dict:
        ei, t = self.index[idx]
        ep = self.episodes[ei]
        L, K = self.L, self.K
        dyn = self.dyn_norm(ep.dyn)
        pwm = self.pwm_norm(ep.pwm)

        hist_s = self._hist_dyn(dyn, t)                     # [L, 19]
        # past actions a_{t-L:t-1}
        hist_a = pwm[t - L : t]                             # [L, 8]  (includes a_{t-1} at last step)
        s_t = dyn[t]                                        # [19]
        a_fut = pwm[t : t + K]                              # [K, 8]
        s_fut = dyn[t + 1 : t + K + 1]                      # [K, 19]

        # counterfactual actions for ranking loss
        a_zero = np.zeros_like(a_fut)
        a_rev = -a_fut
        rng = np.random.RandomState((ei * 1000003 + t) & 0xFFFFFFFF)
        a_rand = rng.normal(0.0, 1.0, size=a_fut.shape).astype(np.float32)
        a_near = np.clip(a_fut + 0.15 * rng.normal(size=a_fut.shape).astype(np.float32), -1.0, 1.0)
        speed = float(np.linalg.norm(ep.dyn[t - L : t, 0:3].mean(axis=0)))
        if speed < 0.08:
            cur_bin = 0
        elif speed < 0.25:
            cur_bin = 1
        else:
            cur_bin = 2
        fc = int(ep.flow_class[t]) if ep.flow_class is not None and t < len(ep.flow_class) else 0
        ff = int(ep.fail_class[t]) if ep.fail_class is not None and t < len(ep.fail_class) else 0
        eta_t = (
            ep.eta_arr[t].astype(np.float32)
            if ep.eta_arr is not None and t < len(ep.eta_arr)
            else np.ones(8, np.float32)
        )
        t_frac = float(t / max(1, ep.dyn.shape[0] - 1))

        return {
            "hist_s": torch.from_numpy(hist_s),
            "hist_a": torch.from_numpy(hist_a.astype(np.float32)),
            "s_t": torch.from_numpy(s_t.astype(np.float32)),
            "a_fut": torch.from_numpy(a_fut.astype(np.float32)),
            "s_fut": torch.from_numpy(s_fut.astype(np.float32)),
            "a_zero": torch.from_numpy(a_zero.astype(np.float32)),
            "a_rev": torch.from_numpy(a_rev.astype(np.float32)),
            "a_rand": torch.from_numpy(a_rand),
            "a_near": torch.from_numpy(a_near.astype(np.float32)),
            "task_index": torch.tensor(ep.task_index, dtype=torch.long),
            "current_bin": torch.tensor(cur_bin, dtype=torch.long),
            "regime_index": torch.tensor(int(ep.regime_index), dtype=torch.long),
            "episode_index": torch.tensor(ep.episode_index, dtype=torch.long),
            "flow_class": torch.tensor(fc, dtype=torch.long),
            "fail_class": torch.tensor(ff, dtype=torch.long),
            "eta_target": torch.from_numpy(eta_t),
            "t_frac": torch.tensor(t_frac, dtype=torch.float32),
            "t_sec": torch.tensor(float(t) * ep.dt if ep.split == "ou" else float(ep.timestamp[min(t, len(ep.timestamp) - 1)]), dtype=torch.float32),
            "dt": torch.tensor(float(ep.dt), dtype=torch.float32),
            "is_ou": torch.tensor(1 if ep.split == "ou" else 0, dtype=torch.float32),
        }


def collate(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


TARGET_DT = 0.1
_DT_TOL = 0.02


def _resample_series(ts: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    x = np.asarray(x)
    if x.ndim == 1:
        return np.interp(grid, ts, x).astype(np.float32)
    cols = [np.interp(grid, ts, x[:, j]) for j in range(x.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def ensure_ou_dt(ts: np.ndarray, fields: dict, target_dt: float = TARGET_DT) -> tuple:
    """Force OU streams onto a uniform 10 Hz grid."""
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if ts.size < 8:
        raise ValueError(f"OU episode too short: {ts.size} samples")
    dts = np.diff(ts)
    med = float(np.median(dts))
    if abs(med - target_dt) <= _DT_TOL and float(np.max(np.abs(dts - target_dt))) <= 2 * _DT_TOL:
        return ts.astype(np.float32), fields, {"resampled": False, "native_dt": med}
    t0, t1 = float(ts[0]), float(ts[-1])
    grid = np.arange(t0, t1 + 1e-9, target_dt, dtype=np.float64)
    if grid.size < 8:
        raise ValueError(f"OU resample produced {grid.size} frames (native_dt={med})")
    out = {k: _resample_series(ts, v, grid) for k, v in fields.items()}
    return grid.astype(np.float32), out, {"resampled": True, "native_dt": med}


OBJ_DIM = 6   # object in the body frame: x, y, z, cos(dpsi), sin(dpsi), valid


def ou_npz_to_episode(path: Path, regime_index: int, episode_index: int, load_images: bool = False,
                      use_arm: bool = False, n_joints: int = 5, use_object: bool = False) -> Episode:
    z = np.load(path)
    if path.name.startswith("ou_placeholder"):
        raise ValueError(f"refusing placeholder OU file {path}")
    pwm = z["pwm"].astype(np.float32)
    T = pwm.shape[0]
    ep_dt = float(z["dt"]) if "dt" in z.files else TARGET_DT
    if "dvl_seq" in z.files and int(len(np.unique(z["dvl_seq"]))) == int(T):
        ts = np.arange(T, dtype=np.float64) * ep_dt
    elif "timestamp" in z.files:
        ts = z["timestamp"].astype(np.float64).reshape(-1)
        if T > 1 and float(np.median(np.abs(np.diff(ts)))) < 0.5 * ep_dt:
            ts = np.arange(T, dtype=np.float64) * ep_dt
    else:
        ts = np.arange(T, dtype=np.float64) * ep_dt
    fields = {
        "pwm": pwm,
        "dvl": z["dvl"].astype(np.float32),
        "imu_av": z["imu_av"].astype(np.float32),
        "imu_la": z["imu_la"].astype(np.float32),
        "pressure": z["pressure"].astype(np.float32).reshape(T, 1),
        "dvl_h": z["dvl_h"].astype(np.float32).reshape(T, 1),
    }
    ts, fields, meta = ensure_ou_dt(ts, fields, target_dt=ep_dt)
    if meta["resampled"]:
        print(f"[ou] resampled {path.name} native_dt={meta['native_dt']:.4f} -> {ep_dt}", flush=True)
    pwm = fields["pwm"]
    dvl = fields["dvl"]
    imu_av = fields["imu_av"]
    imu_la = fields["imu_la"]
    pressure = fields["pressure"].reshape(-1, 1)
    dvl_h = fields["dvl_h"].reshape(-1, 1)
    T = pwm.shape[0]
    ts_rel = np.arange(T, dtype=np.float64) * ep_dt
    # The collectors recorded the PUBLISHED pwm (after apply_efficiency), i.e. the delivered
    # thrust. A real vehicle only knows what it COMMANDED; the thruster fault is exactly the
    # unobserved gap the model must infer. The eta schedule is known analytically, so the
    # commanded value is recovered exactly: u_cmd = u_published / eta.
    if "eta" in z.files:
        eta = z["eta"].astype(np.float32).reshape(T, 8)
        pwm_recorded_is_cmd = bool(z["pwm_is_commanded"]) if "pwm_is_commanded" in z.files else False
    else:
        eta = eta_schedule(path.stem, ts_rel)
        pwm_recorded_is_cmd = False
    pwm_cmd = pwm if pwm_recorded_is_cmd else np.clip(pwm / np.maximum(eta, 1e-3), -1.0, 1.0)
    pwm = pwm_cmd
    explicit_eta = "eta" in z.files
    # The state at tick t describes the vehicle BEFORE acting, so its PWM columns must hold the
    # command that has been driving it (published at t-1). Using pwm[t] would leak a_t into s_t,
    # which deployment can never reproduce.
    pwm_prev = np.vstack([np.zeros((1, 8), np.float32), pwm[:-1]])
    dyn = np.concatenate([dvl, imu_av, imu_la, pressure, dvl_h, pwm_prev], axis=-1)
    state = np.zeros((T, 29), np.float32)
    state[:, 5:13] = pwm
    state[:, 18:21] = dvl
    state[:, 21:24] = imu_av
    state[:, 24:27] = imu_la
    state[:, 27:28] = pressure
    state[:, 28:29] = dvl_h
    action = np.zeros((T, 13), np.float32)
    action[:, 5:13] = pwm
    # the OU / planner corpora are locomotion-only (arm parked); zero-padding to the
    # arm layout lets them be mixed with USIM arm episodes in one training run. Harness
    # recordings converted by scripts/recordings_to_ou.py carry the real joint channels
    # (and joint targets); those are used when present.
    if use_arm:
        if "joint_pos" in z.files and "joint_v" in z.files and len(z["joint_pos"]) == T:
            jp = z["joint_pos"].astype(np.float32)[:, :n_joints]
            jv = z["joint_v"].astype(np.float32)[:, :n_joints]
            jc = z["joint_cmd"].astype(np.float32)[:, :n_joints] if "joint_cmd" in z.files else jp
            dyn = np.concatenate([dyn, jp, jv], axis=-1).astype(np.float32)
            pwm = np.concatenate([pwm, jc], axis=-1).astype(np.float32)
            state[:, 0:5] = jp[:, :5]
            state[:, 13:18] = jv[:, :5]
            action[:, 0:5] = jc[:, :5]
        else:
            dyn, pwm = pad_arm_channels(dyn, pwm, n_joints)
    if use_object:
        # object relative pose (body frame xyz, cos/sin relative yaw, valid) -- the manipulation
        # world model's extra state; zeros (valid = 0) for scenes without an object
        if "obj_body" in z.files and len(z["obj_body"]) == T:
            ob = z["obj_body"].astype(np.float32).reshape(T, -1)[:, :OBJ_DIM]
        else:
            ob = np.zeros((T, OBJ_DIM), np.float32)
        dyn = np.concatenate([dyn, ob], axis=-1).astype(np.float32)
    flow, fail = flow_fail_labels(path.stem, ts_rel)
    if explicit_eta:
        # labels straight from the recorded schedule (randomized-fault episodes)
        n_bad = (eta < 0.99).sum(axis=1)
        fail = np.clip(n_bad, 0, 2).astype(np.int64)
    hw = (96, 128)
    rgb_arr = None
    fls_arr = None
    if load_images:
        if "rgb" in z.files:
            rgb_arr = _resize_u8(z["rgb"], hw)
        if "fls" in z.files:
            fls_arr = _resize_u8(z["fls"], hw)
    return Episode(
        episode_index=episode_index,
        task_index=regime_index,
        task=path.stem,
        state=state,
        action=action,
        dyn=dyn,
        pwm=pwm,
        target_pos=np.zeros((T, 6), np.float32),
        timestamp=ts.astype(np.float32),
        split="ou",
        parquet_path=path,
        regime_index=regime_index,
        dt=ep_dt,
        flow_class=flow,
        fail_class=fail,
        eta_arr=eta,
        rgb_arr=rgb_arr,
        fls_arr=fls_arr,
    )


def load_ou_split(ou_dir: Path, load_images: bool = False, use_arm: bool = False,
                  n_joints: int = 5, use_object: bool = False) -> List[Episode]:
    ou_dir = Path(ou_dir)
    names = [
        "ou_nominal.npz",
        "ou_const_current.npz",
        "ou_time_varying.npz",
        "ou_thruster_degrade.npz",
        "ou_current_and_fail.npz",
        "ou_change_point.npz",
    ]
    eps = []
    for i, name in enumerate(names):
        p = ou_dir / name
        if p.exists():
            eps.append(ou_npz_to_episode(p, i, 10_000 + i, load_images=load_images,
                                         use_arm=use_arm, n_joints=n_joints, use_object=use_object))
    # randomized-fault episodes (explicit eta array) living in the same directory
    for i, p in enumerate(sorted(ou_dir.glob("ou_randfault_*.npz"))):
        eps.append(ou_npz_to_episode(p, 0, 20_000 + i, load_images=load_images,
                                     use_arm=use_arm, n_joints=n_joints, use_object=use_object))
    # any other OU-schema recording in the directory (recordings_to_ou.py / collect_planner.py
    # outputs: collect_full__*.npz, <arm>__<task>__episodeN.npz, task_*.npz). Before this branch
    # such directories silently contributed 0 episodes to --extra-ou / --extra.
    known = set(names)
    for i, p in enumerate(sorted(ou_dir.glob("*.npz"))):
        if p.name in known or p.name.startswith("ou_randfault_"):
            continue
        try:
            eps.append(ou_npz_to_episode(p, 0, 30_000 + i, load_images=load_images,
                                         use_arm=use_arm, n_joints=n_joints, use_object=use_object))
        except (KeyError, ValueError) as e:  # not an OU-schema file; skip loudly
            print(f"load_ou_split: skipping {p.name} ({e})", flush=True)
    return eps


def _read_video_frames(path: Path, hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Return [T, 3, H, W] float32 in [0, 1] or None."""
    try:
        import av  # type: ignore
    except Exception:
        return None
    if not path.exists():
        return None
    H, W = hw
    frames = []
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            if img.shape[0] != H or img.shape[1] != W:
                import cv2

                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(img)
    finally:
        container.close()
    if not frames:
        return None
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return arr.transpose(0, 3, 1, 2)


class VisualWindowDataset(Dataset):
    """RGB residual windows: I_t -> I_{t+Kvis} conditioned on future PWM + disturbance hist."""

    def __init__(self, episodes: Sequence[Episode], cfg: Cfg, dyn_norm: RunningNorm, pwm_norm: RunningNorm, stride: int = 1):
        self.cfg = cfg
        self.L = cfg.model.history_len
        self.K = cfg.model.horizon_dyn
        self.Kv = cfg.model.horizon_vis
        self.hw = cfg.schema.image_hw
        self.dyn_norm = dyn_norm
        self.pwm_norm = pwm_norm
        self.episodes = [
            ep
            for ep in episodes
            if (ep.ego_video is not None and ep.ego_video.exists()) or ep.rgb_arr is not None
        ]
        self.index: List[Tuple[int, int]] = []
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_max = 160
        stride = max(1, int(stride))
        for i, ep in enumerate(self.episodes):
            t_max = min(ep.dyn.shape[0] - self.Kv - 1, ep.dyn.shape[0] - self.K - 1)
            for t in range(self.L, max(self.L + 1, t_max), stride):
                self.index.append((i, t))

    def __len__(self) -> int:
        return len(self.index)

    def _frames(self, i: int) -> Optional[np.ndarray]:
        if i not in self._cache:
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            ep = self.episodes[i]
            self._cache[i] = _read_video_frames(ep.ego_video, self.hw)  # type: ignore[arg-type]
        return self._cache[i]

    def __getitem__(self, idx: int) -> dict:
        ei, t = self.index[idx]
        ep = self.episodes[ei]
        frames = None
        if ep.rgb_arr is None:
            frames = self._frames(ei)
        L, K, Kv = self.L, self.K, self.Kv
        dyn = self.dyn_norm(ep.dyn)
        pwm = self.pwm_norm(ep.pwm)
        hist_s = dyn[t - L : t]
        hist_a = pwm[t - L : t]
        s_t = dyn[t]
        a_dyn = pwm[t : t + K]
        a_vis = pwm[t : t + Kv]
        if a_vis.shape[0] < Kv:
            pad = np.repeat(a_vis[-1:], Kv - a_vis.shape[0], axis=0)
            a_vis = np.concatenate([a_vis, pad], axis=0)
        rgb_t = np.zeros((3, *self.hw), np.float32)
        rgb_gt = rgb_t
        if ep.rgb_arr is not None:
            def _chw(img):
                x = img.astype(np.float32) / 255.0
                if x.ndim == 3 and x.shape[-1] == 3:
                    x = x.transpose(2, 0, 1)
                elif x.ndim == 2:
                    x = np.stack([x, x, x], 0)
                return x
            rgb_t = _chw(ep.rgb_arr[min(t, len(ep.rgb_arr) - 1)])
            rgb_gt = _chw(ep.rgb_arr[min(t + Kv, len(ep.rgb_arr) - 1)])
        elif frames is not None:
            rgb_t = frames[t] if t < len(frames) else rgb_t
            gt_i = min(t + Kv, len(frames) - 1)
            rgb_gt = frames[gt_i]
        H, W = self.hw
        sonar = np.zeros((1, H, W), np.float32)
        mask_sonar = 0.0
        sonar_profile = np.zeros(128, np.float32)
        if ep.fls_arr is not None and t < len(ep.fls_arr):
            s = ep.fls_arr[t].astype(np.float32) / 255.0
            if s.shape != (H, W):
                import cv2
                s = cv2.resize(s, (W, H), interpolation=cv2.INTER_AREA)
            sonar = s[None]
            mask_sonar = 1.0
            row = s.mean(axis=1)
            sonar_profile = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, row.shape[0]), row).astype(np.float32)
        rng = np.random.RandomState((ei * 9176 + t) & 0xFFFFFFFF)
        a_zero = np.zeros_like(a_vis)
        a_rev = -a_vis
        a_rand = rng.uniform(-2.5, 2.5, size=a_vis.shape).astype(np.float32)
        a_near = np.clip(a_vis + 0.15 * rng.normal(size=a_vis.shape).astype(np.float32), -1.0, 1.0)
        s_fut = dyn[t + 1 : t + K + 1]
        if s_fut.shape[0] < K:
            pad = np.repeat(s_fut[-1:], K - s_fut.shape[0], axis=0)
            s_fut = np.concatenate([s_fut, pad], axis=0)
        return {
            "hist_s": torch.from_numpy(hist_s.astype(np.float32)),
            "hist_a": torch.from_numpy(hist_a.astype(np.float32)),
            "s_t": torch.from_numpy(s_t.astype(np.float32)),
            "s_fut": torch.from_numpy(s_fut.astype(np.float32)),
            "a_fut": torch.from_numpy(a_dyn.astype(np.float32)),
            "a_vis": torch.from_numpy(a_vis.astype(np.float32)),
            "rgb": torch.from_numpy(rgb_t.astype(np.float32)),
            "rgb_gt": torch.from_numpy(rgb_gt.astype(np.float32)),
            "sonar": torch.from_numpy(sonar.astype(np.float32)),
            "mask_sonar": torch.tensor(mask_sonar, dtype=torch.float32),
            "sonar_gt_profile": torch.from_numpy(sonar_profile),
            "is_ou": torch.tensor(1.0 if ep.split == "ou" else 0.0, dtype=torch.float32),
            "a_zero": torch.from_numpy(a_zero.astype(np.float32)),
            "a_rev": torch.from_numpy(a_rev.astype(np.float32)),
            "a_rand": torch.from_numpy(a_rand),
            "a_near": torch.from_numpy(a_near.astype(np.float32)),
            "task_index": torch.tensor(ep.task_index, dtype=torch.long),
        }
