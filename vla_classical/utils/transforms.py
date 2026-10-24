"""Small pose/rotation helpers.

robosuite uses quaternions in (x, y, z, w) order. We keep that convention here
and lean on scipy for the heavy lifting.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """(x,y,z,w) quaternion -> 3-vector axis*angle (rad)."""
    return R.from_quat(np.asarray(quat, dtype=np.float64)).as_rotvec()


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    return R.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_quat()


def quat_mul(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    r = R.from_quat(q1_xyzw) * R.from_quat(q2_xyzw)
    return r.as_quat()


def clamp_vec(v: np.ndarray, low, high) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.float64), np.asarray(low), np.asarray(high))


def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))
