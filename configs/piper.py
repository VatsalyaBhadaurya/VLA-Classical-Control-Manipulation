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
    # cam_global: fixed world camera defined by an eye->target look-at (intuitive
    # to tune). TUNE eye/target against a real dataset frame.
    # Elevated 3/4 view centered on the workspace (box + cubes in front of the
    # arm), like the real rig's angled top-down frame. TUNE to taste.
    cam_global_eye: tuple[float, float, float] = (0.05, -0.45, 0.60)
    cam_global_target: tuple[float, float, float] = (0.30, 0.0, 0.02)
    cam_global_fovy: float = 55.0
    # cam_gripper: attached to link6 (wrist). The tool/fingers extend along
    # link6 local +z, so the camera looks along +z (xyaxes below give that).
    cam_gripper_pos: tuple[float, float, float] = (0.0, 0.06, -0.02)
    cam_gripper_xyaxes: tuple[float, ...] = (1, 0, 0, 0, -1, 0)  # view dir = link6 +z
    cam_gripper_fovy: float = 70.0


@dataclass
class CubeSpec:
    name: str
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    half: float = 0.0125                     # ~2.5 cm cube
    mass: float = 0.03


@dataclass
class BoxSpec:
    """Open-top cardboard receptacle (static), built from 5 thin boxes."""
    name: str = "bin"
    pos: tuple[float, float, float] = (0.28, 0.20, 0.0)   # floor corner of the box
    inner_half: tuple[float, float] = (0.06, 0.05)        # inner half-extents (x, y)
    height: float = 0.05
    wall: float = 0.006
    rgba: tuple[float, float, float, float] = (0.76, 0.60, 0.40, 1.0)


@dataclass
class PiperSceneConfig:
    # Ground plane = work surface; arm base sits at z=0 (gravcomp'd, no push).
    # Real rig is a white tabletop.
    floor_z: float = 0.0
    floor_rgba: tuple[float, float, float, float] = (0.86, 0.86, 0.88, 1.0)
    # Task objects: yellow + black blocks and a box (see task instructions).
    # NOTE: real object poses aren't in the dataset — these are plausible
    # reachable placements, to be randomized for training.
    add_objects: bool = True
    cubes: list = field(default_factory=lambda: [
        CubeSpec("yellow_block", (0.32, -0.12, 0.0125), (0.95, 0.80, 0.10, 1.0)),
        CubeSpec("black_block", (0.32, 0.02, 0.0125), (0.08, 0.08, 0.08, 1.0)),
    ])
    box: BoxSpec = field(default_factory=BoxSpec)
    # Even overhead lighting to approximate the bright indoor rig.
    headlight_ambient: tuple[float, float, float] = (0.4, 0.4, 0.4)
    headlight_diffuse: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass
class JointSafetyConfig:
    """Classical safety layer for joint-space absolute targets (the 'HOW')."""
    enabled: bool = True
    max_joint_step: float = 0.05       # rad per control step (30Hz -> 1.5 rad/s)
    max_gripper_step: float = 0.02     # m per control step
    lowpass_alpha: float = 0.6         # EMA smoothing on the target (1=no smoothing)
    max_joint_vel: float = 3.0         # rad/s; E-stop if measured vel exceeds
    estop_on_nan: bool = True


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
    safety: JointSafetyConfig = field(default_factory=JointSafetyConfig)

    # SmolVLA inference (after fine-tuning). Point at the checkpoint's
    # pretrained_model dir produced by lerobot-train.
    vla_ckpt: str = "outputs/train/piper_smolvla/checkpoints/last/pretrained_model"
    vla_device: str = "cuda"           # "cpu" fallback if VRAM insufficient
    vla_half: bool = False             # fp16 weights to save VRAM (try if OOM)

    # Real resting configuration (from dataset episode 0, row 0) — use instead
    # of the Menagerie "home" keyframe so the sim starts like the real robot.
    rest_state: tuple[float, ...] = (0.012, 0.004, -0.018, 0.052, 0.499, -0.135, 0.0)

    # Data
    dataset_repo_id: str = "aiengineer56/piper_dataset_1100"
    # Task instruction (from dataset meta/tasks). Variants also include
    # "pick the black block put into the box" / "pick the yellow block".
    instruction: str = "pick the yellow block put into the box"


def make_piper_config() -> PiperConfig:
    return PiperConfig()
