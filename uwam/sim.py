"""Stonefish / u0env helpers: dynamics randomization, FLS recording, OU collection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np

from .control import OUProcess
from .config import ControlCfg


@dataclass
class DynamicsRegime:
    name: str
    current: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_omega: float = 0.0          # if >0, sinusoidal time-varying current
    current_amp: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    eta: Tuple[float, ...] = (1, 1, 1, 1, 1, 1, 1, 1)  # thruster efficiency
    change_at: Optional[float] = None   # seconds; switch to eta_after / current_after
    eta_after: Optional[Tuple[float, ...]] = None
    current_after: Optional[Tuple[float, float, float]] = None


REGIMES = [
    DynamicsRegime("nominal"),
    DynamicsRegime("const_current", current=(0.15, 0.05, 0.0)),
    DynamicsRegime("time_varying", current=(0.05, 0.0, 0.0), current_omega=0.4, current_amp=(0.12, 0.08, 0.0)),
    DynamicsRegime("thruster_degrade", eta=(1, 1, 0.5, 1, 1, 1, 1, 1)),
    DynamicsRegime("current_and_fail", current=(0.1, 0.0, 0.0), eta=(1, 0.4, 1, 1, 1, 1, 1, 1)),
    DynamicsRegime("change_point", change_at=8.0, eta_after=(1, 1, 1, 1, 0.5, 0.5, 1, 1),
                   current_after=(0.15, 0.05, 0.0)),
]

# Mid-trial faults on the HORIZONTAL thrusters (0-3), which are the ones that produce surge/sway.
# change_point above degrades thrusters 4-5 (vertical), so it barely touches the x/y goals and never
# punishes a controller that simply holds its last command. These do, which is what makes the
# fault-during-blackout protocol a real test of re-identification.
FAULT_MID_REGIMES = [
    DynamicsRegime("fail_mid_t2", change_at=8.0, eta_after=(1, 1, 0.35, 1, 1, 1, 1, 1)),
    DynamicsRegime("fail_mid_t0", change_at=8.0, eta_after=(0.4, 1, 1, 1, 1, 1, 1, 1)),
    DynamicsRegime("fail_mid_t1", change_at=8.0, eta_after=(1, 0.4, 1, 1, 1, 1, 1, 1)),
    DynamicsRegime("fail_mid_t03", change_at=8.0, eta_after=(0.5, 1, 1, 0.5, 1, 1, 1, 1)),
    # all-horizontal degradation (net/trawl entanglement): the over-actuated mixer cannot
    # average this one away -- the velocity shift is ~2.3 sigma_e, ABOVE the acting
    # threshold Delta*, i.e. the regime where the gate must open
    DynamicsRegime("fail_mid_all", change_at=8.0, eta_after=(0.4, 0.4, 0.4, 0.4, 1, 1, 1, 1)),
]

# Severity sweep: single-thruster faults land BELOW the acting threshold on this vehicle
# (the mixer spreads surge over 4 thrusters), all-horizontal faults cross it -- together
# they trace the crossover the theory predicts.
FAULT_SWEEP_REGIMES = [
    DynamicsRegime(f"sweep_t2_{int(e * 100):02d}", change_at=8.0,
                   eta_after=(1, 1, e, 1, 1, 1, 1, 1))
    for e in (0.8, 0.6, 0.4, 0.2)
] + [
    DynamicsRegime(f"sweep_all_{int(e * 100):02d}", change_at=8.0,
                   eta_after=(e, e, e, e, 1, 1, 1, 1))
    for e in (0.8, 0.6, 0.4, 0.2)
]

# Gate-calibration regimes: deployment-condition streams (blind + held command) with
# eta values and change times deliberately OFF the evaluation grids above. Used only
# with --calib-dump to fit (kappa, h); never scored as results.
CALIB_REGIMES = [
    DynamicsRegime("calib_nom"),
    DynamicsRegime("calib_f0", change_at=9.0, eta_after=(0.3, 1, 1, 1, 1, 1, 1, 1)),
    DynamicsRegime("calib_f1", change_at=10.0, eta_after=(1, 0.45, 1, 1, 1, 1, 1, 1)),
    DynamicsRegime("calib_f2", change_at=9.5, eta_after=(1, 1, 0.55, 1, 1, 1, 1, 1)),
    DynamicsRegime("calib_f03", change_at=10.5, eta_after=(0.6, 1, 1, 0.6, 1, 1, 1, 1)),
    # decision-relevant (above-Delta*) examples for the delay half of the calibration
    DynamicsRegime("calib_fall", change_at=9.5, eta_after=(0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1)),
    DynamicsRegime("calib_fall2", change_at=10.0, eta_after=(0.3, 0.3, 0.3, 0.3, 1, 1, 1, 1)),
]

REGIME_SETS = {
    "main": REGIMES,
    "fault_mid": FAULT_MID_REGIMES,
    "fault_sweep": FAULT_SWEEP_REGIMES,
    "calib": CALIB_REGIMES,
}


def patch_scenario_current(scn_path: Path, current_xyz: Tuple[float, float, float], dest: Path,
                           current_topic: str = "/bluerov2/ocean_current") -> Path:
    """Rewrite uniform current and attach a ROS velocity subscriber for live updates."""
    tree = ET.parse(scn_path)
    root = tree.getroot()
    for node in root.iter("current"):
        if node.attrib.get("type") == "uniform":
            vel = node.find("velocity")
            if vel is not None:
                vel.set("xyz", f"{current_xyz[0]} {current_xyz[1]} {current_xyz[2]}")
            sub = node.find("ros_subscriber")
            if sub is None:
                sub = ET.SubElement(node, "ros_subscriber")
            sub.set("velocity", current_topic)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest)
    return dest


TARGET_DT = 0.1  # must match Cfg.schema.dt / USIM fps=10
TARGET_HZ = 10.0


def patch_robot_sensor_rates(
    robot_scn: Path,
    dest: Path,
    dvl_rate: float = 10.0,
    camera_rate: float = 10.0,
    fls_rate: float = 10.0,
) -> Path:
    """Keep user-opened cameras/FLS on. Set DVL (and cameras) to the WAM 10 Hz clock.

    Do not drop DVL/cameras to 2–5 Hz: that changes Δt vs USIM and causes train-good / infer-bad.
    IMU may stay faster than 10 Hz; collectors snapshot the latest sample on the DVL tick.
    """
    tree = ET.parse(robot_scn)
    root = tree.getroot()
    for sensor in root.iter("sensor"):
        st = sensor.attrib.get("type", "")
        if st == "dvl":
            sensor.set("rate", str(dvl_rate))
        elif st in ("camera", "depthcamera"):
            sensor.set("rate", str(camera_rate))
        elif st == "fls":
            sensor.set("rate", str(fls_rate))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest)
    return dest


def prepare_ou_scene(
    u0env: Path,
    out_dir: Path,
    dvl_rate: float = 10.0,
    camera_rate: float = 10.0,
    fls_rate: float = 10.0,
) -> Path:
    """Official bluerov2_test + same meshes; DVL=10 Hz and live current topic."""
    src_test = u0env / "ros_ws/src/stonefish_bluerov2/scenarios/bluerov2_test.scn"
    src_robot = u0env / "ros_ws/src/stonefish_bluerov2/scenarios/robots/bluerov2.scn"
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_dest = out_dir / "bluerov2_runtime.scn"
    test_dest = out_dir / "bluerov2_test_runtime.scn"
    patch_robot_sensor_rates(
        src_robot, robot_dest, dvl_rate=dvl_rate, camera_rate=camera_rate, fls_rate=fls_rate
    )
    patch_scenario_current(src_test, (0.0, 0.0, 0.0), test_dest)
    tree = ET.parse(test_dest)
    for inc in tree.getroot().iter("include"):
        f = inc.attrib.get("file", "")
        if "robots/bluerov2.scn" in f and "alpha" not in f:
            inc.set("file", str(robot_dest))
    tree.write(test_dest)
    return test_dest


def apply_efficiency(pwm: np.ndarray, eta: Tuple[float, ...]) -> np.ndarray:
    e = np.asarray(eta, dtype=np.float32)
    return (pwm * e).astype(np.float32)


def regime_at(reg: DynamicsRegime, t: float) -> Tuple[Tuple[float, float, float], Tuple[float, ...]]:
    cur = reg.current
    eta = reg.eta
    if reg.current_omega > 0:
        s = np.sin(reg.current_omega * t)
        cur = tuple(c + a * float(s) for c, a in zip(reg.current, reg.current_amp))  # type: ignore
    if reg.change_at is not None and t >= reg.change_at:
        if reg.current_after is not None:
            cur = reg.current_after
        if reg.eta_after is not None:
            eta = reg.eta_after
    return cur, eta  # type: ignore


def collect_ou_offline_placeholder(out_dir: Path, n_frames: int = 16000, seed: int = 0) -> Path:
    """
    When the live simulator is not yet up, emit a structurally valid OU dataset
    using a simple 6-DoF surrogate so the rest of the pipeline can be tested.
    Replace with ROS collection once u0env is built.
    """
    rng = np.random.default_rng(seed)
    ou = OUProcess()
    out_dir.mkdir(parents=True, exist_ok=True)
    T = n_frames
    pwm = np.zeros((T, 8), np.float32)
    dvl = np.zeros((T, 3), np.float32)
    imu_av = np.zeros((T, 3), np.float32)
    imu_la = np.zeros((T, 3), np.float32)
    v = np.zeros(3, np.float32)
    w = np.zeros(3, np.float32)
    # crude allocation similar to rov_gym_env.sixdof_thrust_to_pwm inverse
    B = rng.normal(0, 0.15, size=(3, 8)).astype(np.float32)
    Bw = rng.normal(0, 0.1, size=(3, 8)).astype(np.float32)
    current = np.array([0.08, 0.02, 0.0], np.float32)
    eta = np.ones(8, np.float32)
    eta[2] = 0.6
    for t in range(T):
        u = ou.sample()
        pwm[t] = u
        tau = B @ (u * eta) + current
        tau_w = Bw @ (u * eta)
        v = 0.92 * v + 0.08 * tau
        w = 0.9 * w + 0.1 * tau_w
        dvl[t] = v + 0.01 * rng.normal(size=3).astype(np.float32)
        imu_av[t] = w
        imu_la[t] = np.array([0, 0, -9.81], np.float32) + tau * 2.0
    np.savez_compressed(
        out_dir / "ou_placeholder.npz",
        pwm=pwm, dvl=dvl, imu_av=imu_av, imu_la=imu_la,
        pressure=np.full((T, 1), 0.3, np.float32),
        dvl_h=np.full((T, 1), 0.8, np.float32),
        eta=eta, current=current,
    )
    (out_dir / "regime.json").write_text(json.dumps({"note": "placeholder until u0env ROS is up", "frames": T}, indent=2))
    return out_dir / "ou_placeholder.npz"
