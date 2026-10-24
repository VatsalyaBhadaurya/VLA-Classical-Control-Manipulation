"""robosuite-backed manipulation environment wrapper.

Thin, swappable wrapper around robosuite that exposes exactly what the rest of
the pipeline needs:

    env = ManipulationEnv(cfg)
    obs = env.reset()
    obs, reward, done, info = env.step(action7)   # action = [dx dy dz drx dry drz grip]

`Observation` bundles RGB images + proprioception + task state so the VLA, the
recorder, and the eval metrics all read the same structure. The underlying
robosuite OSC_POSE controller *is* the operational-space controller in the
architecture; our SafetyFilter (control/safety.py) wraps the action before it
reaches here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from configs import ExperimentConfig
from .camera import extract_images


@dataclass
class Observation:
    images: dict = field(default_factory=dict)      # {cam_name: uint8 HxWx3}
    eef_pos: np.ndarray = None                       # (3,) world xyz
    eef_quat: np.ndarray = None                      # (4,) xyzw
    joint_pos: np.ndarray = None                     # (nq,)
    joint_vel: np.ndarray = None                     # (nq,)
    gripper_qpos: np.ndarray = None                  # gripper finger state
    object_pose: Optional[np.ndarray] = None         # (7,) pos+quat if available
    instruction: str = ""
    raw: dict = field(default_factory=dict)          # full robosuite obs (escape hatch)

    def proprio_vector(self) -> np.ndarray:
        """Flat proprioception vector for policies that want one tensor."""
        parts = [self.eef_pos, self.eef_quat, self.joint_pos,
                 self.joint_vel, self.gripper_qpos]
        return np.concatenate([np.asarray(p, dtype=np.float32).ravel()
                               for p in parts if p is not None])


class ManipulationEnv:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self._env = self._build()
        self.action_dim = self._env.action_dim          # 7 for OSC_POSE + gripper
        self._instruction = cfg.task.instruction

    # -- construction -------------------------------------------------------
    def _build(self):
        import robosuite
        from robosuite import controllers as rc

        cc = self._load_controller_config(rc)

        return robosuite.make(
            env_name=self.cfg.task.robosuite_env,
            robots="Panda",
            controller_configs=cc,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=list(self.cfg.camera.names),
            camera_heights=self.cfg.camera.height,
            camera_widths=self.cfg.camera.width,
            control_freq=self.cfg.sim.control_freq,
            horizon=self.cfg.sim.horizon,
            reward_shaping=True,
            ignore_done=False,
            hard_reset=False,
        )

    def _load_controller_config(self, rc) -> dict:
        """Load an OSC_POSE controller config across robosuite versions.

        robosuite >= 1.5 uses a *composite* controller config (a dict with a
        top-level ``type`` and a ``body_parts`` map); the Panda default is
        already OSC_POSE delta-pose on ``body_parts["right"]``. robosuite <= 1.4
        uses a flat config from ``load_controller_config``.
        """
        c = self.cfg.controller
        output_max = [c.output_max_pos] * 3 + [c.output_max_rot] * 3
        output_min = [-c.output_max_pos] * 3 + [-c.output_max_rot] * 3

        if hasattr(rc, "load_composite_controller_config"):          # >= 1.5
            cc = rc.load_composite_controller_config(controller=None, robot="Panda")
            arm = (cc.get("body_parts") or {}).get("right")
            if arm is None or arm.get("type") != c.controller:
                raise RuntimeError(
                    f"Expected {c.controller} on body_parts['right'] but got "
                    f"{None if arm is None else arm.get('type')}. "
                    "Adjust ControllerConfig or the composite config.")
            arm.update(kp=c.kp, damping_ratio=c.damping_ratio,
                       input_max=c.input_max, input_min=c.input_min,
                       output_max=output_max, output_min=output_min,
                       input_ref_frame="world")   # world-frame deltas for scripted policy
            return cc

        cc = rc.load_controller_config(default_controller=c.controller)  # <= 1.4
        cc.update(kp=c.kp, damping_ratio=c.damping_ratio,
                  input_max=c.input_max, input_min=c.input_min,
                  output_max=output_max, output_min=output_min)
        return cc

    # -- API ----------------------------------------------------------------
    def reset(self) -> Observation:
        raw = self._env.reset()
        return self._wrap(raw)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).ravel()
        assert action.shape[0] == self.action_dim, (
            f"action dim {action.shape[0]} != env action_dim {self.action_dim}")
        raw, reward, done, info = self._env.step(action)
        obs = self._wrap(raw)
        info = dict(info)
        info["success"] = bool(self._env._check_success())
        return obs, float(reward), bool(done), info

    def _wrap(self, raw: dict) -> Observation:
        pfx = "robot0_"
        obj_pose = None
        if "cube_pos" in raw and "cube_quat" in raw:
            obj_pose = np.concatenate([raw["cube_pos"], raw["cube_quat"]])
        return Observation(
            images=extract_images(raw, self.cfg.camera.names,
                                  self.cfg.camera.flip_vertical),
            eef_pos=np.asarray(raw.get(pfx + "eef_pos"), dtype=np.float32),
            eef_quat=np.asarray(raw.get(pfx + "eef_quat"), dtype=np.float32),
            joint_pos=np.asarray(raw.get(pfx + "joint_pos"), dtype=np.float32),
            joint_vel=np.asarray(raw.get(pfx + "joint_vel"), dtype=np.float32),
            gripper_qpos=np.asarray(raw.get(pfx + "gripper_qpos"), dtype=np.float32),
            object_pose=obj_pose,
            instruction=self._instruction,
            raw=raw,
        )

    # -- introspection used by the safety monitor ---------------------------
    @property
    def joint_vel(self) -> np.ndarray:
        return np.asarray(self._env.sim.data.qvel[:7], dtype=np.float64)

    def render_frame(self, camera: Optional[str] = None) -> np.ndarray:
        """Single uint8 frame from a named camera (for videos/plots)."""
        cam = camera or self.cfg.camera.names[0]
        raw = self._env._get_observations()
        return extract_images(raw, [cam], self.cfg.camera.flip_vertical)[cam]

    def close(self):
        try:
            self._env.close()
        except Exception:  # noqa: BLE001
            pass
