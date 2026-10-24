"""Operational-space / impedance control.

For v1 we delegate OSC to robosuite's built-in ``OSC_POSE`` controller (its
gains come from ``ControllerConfig`` and are applied in
``envs/robosuite_env.py``). This module documents that contract and provides a
helper to convert a high-level EE target into the per-step *delta* action that
OSC_POSE expects — used by scripted policies and by the VLA->EE-target path.

If you later move off robosuite (minimal Gymnasium + MuJoCo backend), implement
a standalone OSC here:

    tau = J^T [ Kp (x* - x) - Kd xdot ] + g(q)      # Cartesian impedance
    with nullspace posture control and joint/torque saturation.
"""
from __future__ import annotations

import numpy as np

from configs import ControllerConfig
from ..utils.transforms import quat_xyzw_to_rotvec, rotvec_to_quat_xyzw, quat_mul


def ee_target_to_delta_action(
    cur_pos: np.ndarray,
    cur_quat: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_quat: np.ndarray,
    gripper: float,
    cfg: ControllerConfig,
) -> np.ndarray:
    """Build a 7-dim OSC_POSE action from an absolute EE target.

    OSC_POSE expects a normalized delta in [-1, 1] that it maps to
    [output_min, output_max] meters/radians per step. We compute the position
    and orientation error, scale into that normalized range, and clip.
    """
    dpos = np.asarray(tgt_pos, dtype=np.float64) - np.asarray(cur_pos, dtype=np.float64)

    # orientation error as a rotation vector (axis*angle)
    q_err = quat_mul(np.asarray(tgt_quat, dtype=np.float64),
                     _quat_conj(np.asarray(cur_quat, dtype=np.float64)))
    drot = quat_xyzw_to_rotvec(q_err)

    norm_pos = np.clip(dpos / max(cfg.output_max_pos, 1e-6), -1.0, 1.0)
    norm_rot = np.clip(drot / max(cfg.output_max_rot, 1e-6), -1.0, 1.0)
    action = np.concatenate([norm_pos, norm_rot, [float(gripper)]])
    return action


def _quat_conj(q_xyzw: np.ndarray) -> np.ndarray:
    q = q_xyzw.copy()
    q[:3] *= -1.0
    return q
