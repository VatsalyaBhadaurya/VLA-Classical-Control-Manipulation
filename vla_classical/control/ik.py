"""Inverse kinematics backends (optional — Phase 2+ alternative to OSC).

Not required for the robosuite/OSC_POSE path (OSC is Cartesian and needs no
explicit IK). Provided for the "IK + joint-space PD" controller option in the
spec, or if you switch to a minimal MuJoCo backend.

Install a backend first (see requirements.txt):
    pip install mink        # MuJoCo-native differential IK (recommended)
    pip install pin         # pinocchio (full analytical/numeric IK)
"""
from __future__ import annotations

import numpy as np


class DifferentialIK:
    """Damped-least-squares differential IK stub (mink-backed).

    Intentionally a stub: the OSC path is the default. Fill in when/if you need
    an explicit joint-space target from a Cartesian goal.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DifferentialIK is a Phase 2+ option. The default controller is "
            "robosuite OSC_POSE, which needs no explicit IK. Install `mink` and "
            "implement here only if you want IK + joint PD control."
        )

    def solve(self, q: np.ndarray, target_pos: np.ndarray,
              target_quat: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
