# AquaWAM: A Dynamics-aware World Action Model for Underwater Embodied Agents

**Paper:** https://arxiv.org/abs/2609.33299

**Project page:** https://cunhaozhu.github.io/AquaWAM/

AquaWAM is the first World Action Model designed for underwater embodied agents. Instead of predicting future images, it predicts what the vehicle's own sensors will read: a 35-dimensional physical state from the DVL, IMU, pressure sensor and joint encoders. That state includes the thruster dead band, the inertial glide that outlasts each command, and ambient currents. Every command is chosen by imagining candidate futures and scoring them.

On the 20 tasks of the USIM benchmark, AquaWAM succeeds in 72.6% of trials, against 51.7% for U0 fine-tuned on the same demonstrations under the same protocol. With the DVL lost mid-episode, it succeeds in 61.6% against 39.4% for U0. A decision takes 79.5 ms on an NVIDIA Jetson AGX Orin, 2.7× faster than U0.

## Repository layout

| Directory | Contents |
|---|---|
| `uwam/` | Core model and control. `models.py` holds the dynamics predictor (nominal + residual network conditioned on a disturbance token from a recurrent history encoder). `control.py` holds the sampling planner that imagines and scores candidates. `grasp_planner.py` holds the imagined thrust-pulse planner for fine positioning. `gate.py` is the trust gate that decides when the velocity estimate replaces the DVL. |
| `percept/` | Perception heads on a DINOv2-base backbone: the image-goal navigation head, the wrist-camera object pose head and the container head, with their training and DAgger packing scripts. |
| `scripts/` | Training and data collection: dynamics predictor (`train_stage1.py`), velocity estimator ensemble (`train_vel_ensemble.py`), play-data collection (`collect_ou.py`, `collect_planner.py`, `collect_randfault.py`), simulator setup. |
| `u0eval/` | Evaluation on the USIM simulator: the AquaWAM policy server (`wam_policy_server.py`), the takeover layer that lets a VLA fall back on our estimator (`fallback_policy_server.py`), servers for the baselines, the per-task runner (`run_eval_task.sh`), table generation (`write_u0_table.py`, `write_per_task_table.py`, `write_hard_table.py`) and latency benchmarks (`bench_*.py`). |

## Requirements

- The USIM benchmark and its Stonefish + ROS Noetic simulator. Evaluation runs against the benchmark's own judges.
- Python 3.10 or newer, PyTorch with CUDA. See `requirements.txt` and `pyproject.toml`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Paths are currently hard-coded to the layout of our training machine (`/hy-tmp/...`): weights under `/hy-tmp/models/uwam/`, simulator instances under `/hy-tmp/u0env*`, results under `/hy-tmp/u0env/dataset/eval_runs/`. Either reproduce this layout or edit the defaults in `u0eval/start_server.sh`, `u0eval/instance_env.sh` and the argument defaults of the scripts you run.

## Pretrained weights

Weights are not included in this repository. Place them in `/hy-tmp/models/uwam/`.

| File | Role |
|---|---|
| `best_scenes.pt` | Dynamics predictor, vehicle scale (19-d state) |
| `grasp_core.pt` | Dynamics predictor, full scale with arm and object (35-d state), used by the pulse planner |
| `vel_ens_v2.pt` | Velocity estimator ensemble that replaces a failed DVL |
| `percept_e2e_r4.pt`, `percept_head_base.pt` | Image-goal navigation head |
| `wrist_pose_r2.pt` | Wrist-camera object pose head |
| `box_pose_r1.pt` | Container head for transporting |
| `arm_kin.pt`, `arm_kin.json` | Learned arm forward kinematics |
| `close_outcome.json`, `close_outcome_gb.joblib` | Gripper-closure outcome model |
| `gate_calib.json`, `gate_calib_task.json` | Trust-gate calibration |
| `ou_library_pwm.npy` | Play-data command snippets used as planner candidates |
| `vel_ens_scenes.pt`, `percept_nav_head.pt`, `percept_head.pt`, `direct_head.pt` | Defaults some scripts load when no flag is given |

## Running an evaluation

```bash
source u0eval/instance_env.sh A
bash u0eval/start_server.sh wam_percept
bash u0eval/run_eval_task.sh <task> wam_percept <n_episodes> <port> <drop_dvl_at_s|-1.0> zero
python3 u0eval/write_u0_table.py --auto --paper --condition "full sensing" --percept-round r4
```

`<drop_dvl_at_s>` injects a DVL dropout at that time in the episode; `-1.0` keeps full sensing.
