"""Central configuration via dataclasses.

Every tunable (sim rates, camera size, controller gains, safety limits, VLA
settings) lives here so experiments are reproducible and nothing is hard-coded
deep in a module. Build an `ExperimentConfig` and thread it through.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Simulation / timing
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    control_freq: int = 20          # robosuite env.step() / OSC rate (Hz)
    # physics timestep is control_freq * sim_substeps -> ~500 Hz by default
    horizon: int = 400              # max env steps per episode
    seed: int = 0


@dataclass
class CameraConfig:
    # Keep modest for 6 GB VRAM + latency. 224 or 256 recommended.
    names: tuple[str, ...] = ("agentview",)         # add "robot0_eye_in_hand" for wrist cam
    height: int = 224
    width: int = 224
    flip_vertical: bool = True      # robosuite returns images upside-down


# --------------------------------------------------------------------------- #
# Classical control
# --------------------------------------------------------------------------- #
@dataclass
class ControllerConfig:
    # robosuite OSC_POSE: action = [dx, dy, dz, drx, dry, drz, gripper]
    controller: str = "OSC_POSE"
    # Impedance gains (passed to robosuite OSC). Higher kp = stiffer tracking.
    kp: float = 150.0
    damping_ratio: float = 1.0
    # Scale on the policy's delta action before OSC consumes it.
    input_max: float = 1.0
    input_min: float = -1.0
    # Per-step EE displacement OSC maps the normalized action to (meters / rad).
    output_max_pos: float = 0.05
    output_max_rot: float = 0.5


@dataclass
class SafetyConfig:
    """Sits between the policy and the controller (Phase 5)."""
    # Workspace box for the EE target (meters, world frame). Tune per table.
    ws_low: tuple[float, float, float] = (-0.30, -0.40, 0.78)
    ws_high: tuple[float, float, float] = (0.30, 0.40, 1.20)
    # Rate limiting on the normalized action vector (per control step).
    max_delta_per_step: float = 0.15
    # Low-pass smoothing factor in [0,1]; higher = snappier, lower = smoother.
    lowpass_alpha: float = 0.5
    # Emergency stop thresholds.
    max_joint_vel: float = 2.5      # rad/s; abort if exceeded
    estop_on_nan: bool = True
    enabled: bool = True


# --------------------------------------------------------------------------- #
# VLA (SmolVLA-450M)
# --------------------------------------------------------------------------- #
@dataclass
class VLAConfig:
    model_id: str = "lerobot/smolvla_base"
    device: str = "cuda"            # "cpu" fallback if VRAM insufficient
    dtype: str = "float16"          # mixed precision for 6 GB
    action_chunk: int = 10          # steps predicted per inference
    chunk_truncate: Optional[int] = None  # use first-N of the chunk (VRAM/latency)
    infer_hz: float = 7.0           # target VLA inference rate (5–10 Hz)
    image_size: int = 224
    # Fine-tuning knobs (Phase 4)
    freeze_vision: bool = True      # freeze vision backbone first (saves VRAM)
    gradient_checkpointing: bool = True
    amp: bool = True
    batch_size: int = 2
    lr: float = 1e-4


# --------------------------------------------------------------------------- #
# Task + domain randomization
# --------------------------------------------------------------------------- #
@dataclass
class RandomizationConfig:
    enabled: bool = False
    object_pos: bool = True
    object_color: bool = True
    object_size: bool = False
    object_mass: bool = False
    table_friction: bool = False
    lighting: bool = False
    camera_pose: bool = False
    initial_qpos: bool = False


@dataclass
class TaskConfig:
    name: str = "lift"                          # our task key
    robosuite_env: str = "Lift"                 # robosuite env class name
    instruction: str = "pick up the red cube"   # language goal for the VLA
    randomize: RandomizationConfig = field(default_factory=RandomizationConfig)


# --------------------------------------------------------------------------- #
# Data recording (LeRobot-compatible)
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    root: str = "data"                  # prefer a WSL-native path for speed
    repo_id: str = "local/vla_classical"
    fps: int = 20                       # should match SimConfig.control_freq
    push_to_hub: bool = False


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    vla: VLAConfig = field(default_factory=VLAConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)


# Convenience task presets ---------------------------------------------------
TASK_PRESETS: dict[str, TaskConfig] = {
    "reach": TaskConfig(name="reach", robosuite_env="Lift",
                        instruction="reach the red cube"),
    "lift": TaskConfig(name="lift", robosuite_env="Lift",
                       instruction="pick up the red cube"),
    "pick_place": TaskConfig(name="pick_place", robosuite_env="PickPlace",
                             instruction="place the object into the bin"),
}


def make_config(task: str = "lift") -> ExperimentConfig:
    cfg = ExperimentConfig()
    if task in TASK_PRESETS:
        cfg.task = TASK_PRESETS[task]
    return cfg
