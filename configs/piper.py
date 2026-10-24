"""Config for the AgileX PiPER track (dataset: aiengineer56/piper_dataset_1100).

This mirrors the LeRobot v3.0 dataset schema exactly so the sim env is a drop-in
match for training/replay:

  robot_type   : piper (6-DoF arm + gripper)
  observation.state [7] : joint1..joint6 (rad) + gripper (0..0.1 m)
  action        [7]     : same 7, ABSOLUTE joint-position targets
  cameras       : cam_global 1280x720, cam_gripper 640x480  @ 30 fps

Joint limits below come from the MuJoCo Menagerie agilex_piper model; the
dataset's observed ranges sit inside them. Camera poses are initial guesses to
be tuned against a real dataset frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PiperCameraConfig:
    # (height, width) — matches the dataset resolutions.
    cam_global_hw: tuple[int, int] = (720, 1280)
    cam_gripper_hw: tuple[int, int] = (480, 640)
    # Downsample factor for faster CPU (swrast) rendering / VLA input. 1 = full.
    render_downsample: int = 1
    # Initial camera extrinsics (TUNE against a real dataset frame).
    # cam_global lives in the world; cam_gripper is attached to link6 (wrist).
    cam_global_pos: tuple[float, float, float] = (1.10, 0.0, 0.70)
    cam_global_xyaxes: tuple[float, ...] = (0, 1, 0, 0, 0, 1)   # front view, +z up
    cam_global_fovy: float = 45.0
    cam_gripper_pos: tuple[float, float, float] = (0.05, 0.0, 0.0)
    cam_gripper_xyaxes: tuple[float, ...] = (0, -1, 0, 0, 0, 1)
    cam_gripper_fovy: float = 60.0


@dataclass
class PiperSceneConfig:
    # Static table (box). Top surface at z=0 so the arm base sits on it.
    table_size: tuple[float, float, float] = (0.35, 0.50, 0.02)
    table_pos: tuple[float, float, float] = (0.35, 0.0, -0.02)
    table_rgba: tuple[float, float, float, float] = (0.72, 0.60, 0.45, 1.0)
    # Placeholder manipulation object — REPLACE once the real task is known.
    add_object: bool = True
    object_size: tuple[float, float, float] = (0.02, 0.02, 0.02)
    object_pos: tuple[float, float, float] = (0.42, 0.0, 0.02)
    object_rgba: tuple[float, float, float, float] = (0.85, 0.1, 0.1, 1.0)
    object_mass: float = 0.05


@dataclass
class PiperConfig:
    # Where the vendored Menagerie model lives (see setup/fetch_piper.sh).
    menagerie_dir: str = "third_party/mujoco_menagerie/agilex_piper"
    control_freq: int = 30                      # matches dataset fps
    n_arm_joints: int = 6

    # Gripper unit conversion: dataset reports 0..0.1; model actuator is 0..0.035.
    gripper_dataset_max: float = 0.10
    gripper_actuator_max: float = 0.035

    # Arm joint command limits (rad), from the Menagerie model. Actions are
    # clamped to these before being sent to the position actuators.
    joint_low: tuple[float, ...] = (-2.618, 0.0, -2.697, -1.832, -1.22, -3.14)
    joint_high: tuple[float, ...] = (2.618, 3.14, 0.0, 1.832, 1.22, 3.14)

    camera: PiperCameraConfig = field(default_factory=PiperCameraConfig)
    scene: PiperSceneConfig = field(default_factory=PiperSceneConfig)

    # Data
    dataset_repo_id: str = "aiengineer56/piper_dataset_1100"
    instruction: str = ""   # filled from the dataset's task metadata once known


def make_piper_config() -> PiperConfig:
    return PiperConfig()
