"""Default hyperparameters recovered from the idea document."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Paths:
    root: Path = Path("/hy-tmp")
    usim: Path = Path("/hy-tmp/data/usim")
    u0env: Path = Path("/hy-tmp/u0env")
    ckpt: Path = Path("/hy-tmp/models/uwam")
    logs: Path = Path("/hy-tmp/logs/uwam")
    ou_data: Path = Path("/hy-tmp/data/ou_explore")


@dataclass
class Schema:
    """USIM LeRobot v2.1 state / action layout (29-d state, 13-d action)."""

    state_dim: int = 29
    action_dim: int = 13
    fps: int = 10                   # WAM clock; train == collect == closed-loop
    dt: float = 0.1                 # 1 / fps; never change this independently of fps
    image_hw: Tuple[int, int] = (96, 128)  # H, W  (AquaJEPA-style)
    ego_key: str = "observation.images.ego"
    wrist_key: str = "observation.images.wrist"
    state_key: str = "observation.state"
    action_key: str = "action"

    # half-open slices matching the published USIM card
    joint_pos: Tuple[int, int] = (0, 5)
    pwm: Tuple[int, int] = (5, 13)
    joint_v: Tuple[int, int] = (13, 18)
    dvl_v: Tuple[int, int] = (18, 21)
    imu_av: Tuple[int, int] = (21, 24)
    imu_la: Tuple[int, int] = (24, 27)
    pressure: Tuple[int, int] = (27, 28)
    dvl_h: Tuple[int, int] = (28, 29)

    action_joint: Tuple[int, int] = (0, 5)
    action_pwm: Tuple[int, int] = (5, 13)


@dataclass
class ModelCfg:
    history_len: int = 16          # L
    horizon_dyn: int = 5           # K for state / DVL (0.5 s @ 10 Hz)
    horizon_vis: int = 15          # ~1.5 s visual horizon
    latent_dim: int = 128
    disturbance_dim: int = 96
    hidden: int = 384
    proprio_dim: int = 16          # dvl(3)+imu_av(3)+imu_la(3)+p(1)+alt(1)+pwm(5? wait 8)=16 with pwm 8 -> 19
    # actual proprio used by the dynamics WAM:
    # dvl_v(3) + imu_av(3) + imu_la(3) + pressure(1) + dvl_h(1) + pwm(8) = 19
    dyn_state_dim: int = 19
    pwm_dim: int = 8
    dropout: float = 0.1
    use_language: bool = False  # language is task-level; do not add it into d_t
    use_dt: bool = False        # dt-conditioned variant (rate-generalization experiment)
    # Manipulator extension for the USIM task suite (grasp / transfer tasks). Off by
    # default so every 19-d checkpoint and published result stays reproducible.
    # When enabled: state 19 -> 29 (append joint_pos 5, joint_v 5), action 8 -> 13
    # (append 5 joint-angle commands). Arm channels go at the END because mask_dvl and
    # vel_head hard-code the DVL at [0:3].
    use_arm: bool = False
    use_object: bool = False   # + object relative pose (6) -> 35-d state
    n_joints: int = 5
    n_tasks: int = 9
    visual_channels: int = 6       # ego RGB + wrist RGB concatenated


@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1
    max_train_windows: int = 0     # 0 = all
    amp: bool = True
    lambda_pix: float = 1.0
    lambda_lpips: float = 0.1
    lambda_rank: float = 0.5
    lambda_state: float = 1.0
    lambda_dvl: float = 2.0
    lambda_hist: float = 0.5
    rank_margin: float = 0.05
    modality_dropout: float = 0.2
    lambda_aux: float = 0.15       # flow/fail CE on d (OU windows only)
    lambda_vel: float = 1.0        # dead-reckoning head trained on DVL-masked history
    lambda_eta: float = 1.0        # per-thruster efficiency regression on d (OU windows only)
    lambda_dcons: float = 0.1      # d(estimate-filled window) ~= d(true window) consistency
    vel_fault_weight: float = 3.0  # upweight dead-reckoning loss on fault windows (P0 diagnosis)


@dataclass
class ControlCfg:
    horizon: int = 5               # must equal ModelCfg.horizon_dyn (0.5 s @ 10 Hz)
    horizon_chunks: int = 4        # chain the 0.5 s rollout to ~2 s, the velocity time constant
    cem_iters: int = 2             # 1 = plain sampling; extra rounds refit on elites
    cem_elites: int = 16
    n_warm_seeds: int = 8          # noisy copies of last tick's best sequence, shifted one step
    n_samples: int = 128
    action_std: float = 0.08       # library snippet noise (keep near OU, not random)
    mixer_std: float = 0.05        # noise on mixer / SysID seeds
    goal_w: float = 5.0
    control_w: float = 0.02
    safety_w: float = 0.4
    att_w: float = 0.5             # penalty on predicted |angular rate| (anti-tumble)
    n_sysid_seeds: int = 16
    n_mixer_seeds: int = 40
    n_hold_seeds: int = 8
    mixer_kp: float = 2.0
    mixer_kff: float = 2.0      # 1 / measured plant gain (see logs/uwam/axis_probe.json)
    mixer_kd_w: float = 0.25
    # uncertainty-gated blind policy: replan only as much as the evidence of change warrants.
    # Thresholds sit above the smoothed estimator noise at cruise speed (~0.03-0.04 m/s),
    # otherwise x/y trials falsely open the gate and re-import replanning noise.
    blind_gate_lo: float = 0.05
    blind_gate_hi: float = 0.12
    blind_alpha_decay: float = 0.90
    blind_innov_smooth: int = 5   # ticks of estimate averaging before the gate
    ou_theta: float = 0.15
    ou_mu: float = 0.0
    ou_sigma: float = 0.35
    ou_dt: float = 0.1


@dataclass
class Cfg:
    paths: Paths = field(default_factory=Paths)
    schema: Schema = field(default_factory=Schema)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    control: ControlCfg = field(default_factory=ControlCfg)


def enable_arm(cfg: Cfg) -> Cfg:
    """Switch the model to the manipulator-extended dimensions (in place).

    Call this before constructing DynamicsWAM; the dims are read straight from
    ModelCfg, so nothing else in the model code needs to know about the arm.
    """
    m = cfg.model
    if m.use_arm:
        return cfg
    m.use_arm = True
    m.dyn_state_dim = 19 + 2 * m.n_joints   # + joint_pos, + joint_v
    m.pwm_dim = 8 + m.n_joints              # thruster PWM + joint-angle commands
    return cfg


def enable_object(cfg: Cfg) -> Cfg:
    """Manipulation world model: append the object's relative pose (6) to the arm state (29 -> 35).
    Implies enable_arm. Slices: see ARM_DYN_SLICES["obj"]."""
    enable_arm(cfg)
    m = cfg.model
    if getattr(m, "use_object", False):
        return cfg
    m.use_object = True
    m.dyn_state_dim = 19 + 2 * m.n_joints + 6
    return cfg


ARM_DYN_SLICES = {
    # index ranges inside the 29-d arm dynamics state (see ModelCfg.use_arm)
    "dvl_v": (0, 3), "imu_av": (3, 6), "imu_la": (6, 9),
    "pressure": (9, 10), "dvl_h": (10, 11), "pwm": (11, 19),
    "joint_pos": (19, 24), "joint_v": (24, 29),
    "obj": (29, 35),   # only under enable_object: x, y, z (body frame), cos(dpsi), sin(dpsi), valid
}


def ensure_dirs(cfg: Cfg) -> None:
    for p in (cfg.paths.ckpt, cfg.paths.logs, cfg.paths.ou_data):
        p.mkdir(parents=True, exist_ok=True)
