"""Demonstration recorder -> LeRobot dataset (Phase 3).

Records, per control step, exactly what the spec asks for:
  * RGB from the agent-view camera (+ optional wrist cam).
  * Language instruction.
  * Joint positions/velocities, EE pose, gripper state.
  * The raw policy action AND the final (post-safety) applied control.
  * Per-episode task-success label.

Writes directly into a LeRobotDataset so the frames can fine-tune SmolVLA with
no conversion step. LeRobot is imported lazily so Phases 1–2 don't require it.

Feature keys here are the single source of truth — keep
`policies/smolvla_policy._build_batch` aligned with them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from configs import ExperimentConfig
from ..envs.robosuite_env import Observation


class DemoRecorder:
    def __init__(self, cfg: ExperimentConfig, repo_id: Optional[str] = None):
        self.cfg = cfg
        self.repo_id = repo_id or cfg.data.repo_id
        self._ds = None
        self._primary_cam = cfg.camera.names[0]
        self._wrist_cam = cfg.camera.names[1] if len(cfg.camera.names) > 1 else None
        self._build()

    def _build(self):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        h, w = self.cfg.camera.height, self.cfg.camera.width
        features = {
            "observation.images.top": {
                "dtype": "video", "shape": (h, w, 3),
                "names": ["height", "width", "channel"]},
            "observation.state": {
                "dtype": "float32", "shape": (self._state_dim(),),
                "names": None},
            "action": {
                "dtype": "float32", "shape": (7,), "names": None},
            "action.applied": {        # post-safety control actually sent to sim
                "dtype": "float32", "shape": (7,), "names": None},
        }
        if self._wrist_cam:
            features["observation.images.wrist"] = {
                "dtype": "video", "shape": (h, w, 3),
                "names": ["height", "width", "channel"]}

        root = Path(self.cfg.data.root) / self.repo_id.replace("/", "_")
        self._ds = LeRobotDataset.create(
            repo_id=self.repo_id, fps=self.cfg.data.fps,
            root=str(root), features=features, use_videos=True,
        )

    def _state_dim(self) -> int:
        # eef_pos(3)+eef_quat(4)+joint_pos(7)+joint_vel(7)+gripper(2) = 23 (Lift)
        return 23

    def add_step(self, obs: Observation, action: np.ndarray,
                 applied: np.ndarray):
        frame = {
            "observation.images.top": obs.images[self._primary_cam],
            "observation.state": obs.proprio_vector().astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "action.applied": np.asarray(applied, dtype=np.float32),
            "task": obs.instruction,
        }
        if self._wrist_cam:
            frame["observation.images.wrist"] = obs.images[self._wrist_cam]
        self._ds.add_frame(frame)

    def end_episode(self, success: bool):
        """Save the episode. Only keep successes if you want a clean demo set."""
        self._ds.save_episode()

    def finalize(self):
        if self.cfg.data.push_to_hub:
            self._ds.push_to_hub()
