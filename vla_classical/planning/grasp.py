"""Grasp + gripper planning from an estimated object bounding box.

Given the perception `ObjectEstimate` (x/y/z dims, center, yaw), compute a
top-down grasp: TCP target poses (pre-grasp / grasp / lift) and the gripper
open/close commands derived from the object's width. The destination is a
hover + drop pose over the box.

Gripper is planned from the object's SHORTER horizontal side:
    open  = short_side + clearance   (clear the object on descent)
    close = short_side - squeeze     (grip it), clamped to [0, gripper_max]
in dataset gripper units (0 = closed, gripper_dataset_max = open). For a cube
the yaw is ambiguous (square footprint), so grasp yaw defaults to 0 unless the
object is clearly elongated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def top_down_mat(yaw: float = 0.0) -> np.ndarray:
    """TCP target rotation: tool axis (site +z) points straight down (world -z),
    rotated by `yaw` about the vertical."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])   # 180° about x: +z -> -z
    return rz @ flip


@dataclass
class GraspPlan:
    pregrasp_pos: np.ndarray
    grasp_pos: np.ndarray
    lift_pos: np.ndarray
    place_hover_pos: np.ndarray
    place_pos: np.ndarray
    grasp_mat: np.ndarray
    gripper_open: float
    gripper_close: float
    graspable: bool
    note: str = ""


def plan_grasp(est, box_center, cfg, hover: float = 0.12,
               clearance: float = 0.015, squeeze: float = 0.004,
               elongated_ratio: float = 1.3) -> GraspPlan:
    gmax = float(cfg.gripper_dataset_max)
    sx, sy = float(est.dims[0]), float(est.dims[1])
    short = min(sx, sy)
    long_ = max(sx, sy)

    # yaw: only trust the PCA orientation for clearly elongated objects; align
    # the gripper to close across the SHORT side.
    if long_ / max(short, 1e-6) >= elongated_ratio:
        yaw = est.yaw if sy >= sx else est.yaw + np.pi / 2
    else:
        yaw = 0.0

    gripper_open = float(np.clip(short + clearance, 0.0, gmax))
    gripper_close = float(np.clip(short - squeeze, 0.0, gmax))
    graspable = short <= gmax
    note = "" if graspable else f"short side {short:.3f} > gripper max {gmax:.3f}"

    cx, cy, cz = est.center
    grasp_z = max(float(cz), float(getattr(cfg.scene, "floor_z", 0.0)) + 0.01)
    grasp_pos = np.array([cx, cy, grasp_z])
    pregrasp_pos = np.array([cx, cy, grasp_z + hover])
    lift_pos = pregrasp_pos.copy()

    bx, by, bz = box_center
    place_hover_pos = np.array([bx, by, bz + hover])
    place_pos = np.array([bx, by, bz + 0.04])     # release just above box floor

    return GraspPlan(
        pregrasp_pos=pregrasp_pos, grasp_pos=grasp_pos, lift_pos=lift_pos,
        place_hover_pos=place_hover_pos, place_pos=place_pos,
        grasp_mat=top_down_mat(yaw),
        gripper_open=gripper_open, gripper_close=gripper_close,
        graspable=graspable, note=note)
