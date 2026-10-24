"""Scripted expert for the Lift task (Phase 2 / demo generation).

A simple finite-state pick policy that uses the privileged cube pose from the
simulator (fine for generating demonstrations — the VLA will later do this from
pixels + language only). Emits OSC_POSE delta actions, so it flows through the
exact same safety filter + controller path as the VLA.

States:  REACH_ABOVE -> DESCEND -> GRASP -> LIFT -> DONE
"""
from __future__ import annotations

import numpy as np

from configs import ControllerConfig
from .base import Policy
from ..envs.robosuite_env import Observation
from ..control.osc_controller import ee_target_to_delta_action

# robosuite gripper convention: +1 closes, -1 opens.
GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0


class ScriptedLiftPolicy(Policy):
    produces_chunks = False

    def __init__(self, controller_cfg: ControllerConfig,
                 hover: float = 0.10, grasp_z: float = 0.015,
                 lift_z: float = 0.25, pos_tol: float = 0.01):
        self.cc = controller_cfg
        self.hover = hover          # height above cube before descent (m)
        self.grasp_z = grasp_z      # offset above cube center at grasp (m)
        self.lift_z = lift_z        # target lift height above table (m)
        self.pos_tol = pos_tol
        self.reset(None)

    def reset(self, obs):
        self.state = "REACH_ABOVE"
        self._grasp_counter = 0
        self._z0 = None

    def act(self, obs: Observation) -> np.ndarray:
        assert obs.object_pose is not None, \
            "ScriptedLiftPolicy needs object_pose (use_object_obs=True, Lift task)"
        cube_pos = obs.object_pose[:3].astype(np.float64)
        cur_pos = obs.eef_pos.astype(np.float64)
        cur_quat = obs.eef_quat.astype(np.float64)
        # keep current orientation as the target (top-down grasp from init pose)
        tgt_quat = cur_quat
        if self._z0 is None:
            self._z0 = cube_pos[2]

        grip = GRIP_OPEN
        if self.state == "REACH_ABOVE":
            tgt = cube_pos + np.array([0, 0, self.hover])
            if np.linalg.norm(cur_pos[:2] - cube_pos[:2]) < self.pos_tol and \
               abs(cur_pos[2] - tgt[2]) < self.pos_tol * 2:
                self.state = "DESCEND"
        elif self.state == "DESCEND":
            tgt = cube_pos + np.array([0, 0, self.grasp_z])
            if abs(cur_pos[2] - tgt[2]) < self.pos_tol:
                self.state = "GRASP"
        elif self.state == "GRASP":
            tgt = cube_pos + np.array([0, 0, self.grasp_z])
            grip = GRIP_CLOSE
            self._grasp_counter += 1
            if self._grasp_counter > 10:        # hold to seat the grasp
                self.state = "LIFT"
        elif self.state == "LIFT":
            tgt = np.array([cube_pos[0], cube_pos[1], self._z0 + self.lift_z])
            grip = GRIP_CLOSE
            if cur_pos[2] > self._z0 + self.lift_z - self.pos_tol * 3:
                self.state = "DONE"
        else:  # DONE
            tgt = cur_pos
            grip = GRIP_CLOSE

        return ee_target_to_delta_action(cur_pos, cur_quat, tgt, tgt_quat,
                                         grip, self.cc)
