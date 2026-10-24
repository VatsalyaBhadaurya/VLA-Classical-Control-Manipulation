"""Safety filter + action smoother + emergency stop.

This is the "decide HOW to execute it safely" layer. It sits between the policy
(VLA or scripted) and the environment's OSC controller. Responsibilities:

  * Low-pass smoothing of the raw action (reduces jerk from a 5–10 Hz VLA that
    is held across higher-rate control steps).
  * Rate limiting: clamp per-step change in the normalized action.
  * Workspace clamping: project the *predicted* EE target into a safe box so a
    bad VLA command can't drive the arm off the table.
  * Gripper command pass-through (last action dim).
  * Emergency stop: detect NaN/inf actions or excessive joint velocity and
    freeze the arm (zero delta, hold gripper).

The OSC controller (robosuite) enforces joint position/velocity/torque limits
internally; this layer adds Cartesian workspace + smoothing + an abort path on
top. Call `filter(raw_action, obs, joint_vel)` every control step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from configs import SafetyConfig
from ..utils.transforms import clamp_vec


@dataclass
class SafetyStatus:
    estop: bool = False
    reason: str = ""
    clamped_workspace: bool = False
    rate_limited: bool = False


class SafetyFilter:
    def __init__(self, cfg: SafetyConfig, action_dim: int = 7):
        self.cfg = cfg
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self._prev = np.zeros(self.action_dim, dtype=np.float64)
        self._ema = np.zeros(self.action_dim, dtype=np.float64)
        self._tripped = False

    def filter(self, action: np.ndarray, eef_pos: Optional[np.ndarray] = None,
               joint_vel: Optional[np.ndarray] = None):
        """Return (safe_action, SafetyStatus)."""
        status = SafetyStatus()
        if not self.cfg.enabled:
            return np.asarray(action, dtype=np.float64), status

        a = np.asarray(action, dtype=np.float64).ravel().copy()

        # --- emergency stop: invalid command ------------------------------
        if self.cfg.estop_on_nan and not np.all(np.isfinite(a)):
            return self._trip("non-finite action")
        if joint_vel is not None and np.max(np.abs(joint_vel)) > self.cfg.max_joint_vel:
            return self._trip(
                f"joint vel {np.max(np.abs(joint_vel)):.2f} > "
                f"{self.cfg.max_joint_vel}")
        if self._tripped:
            return self._hold(status)

        # --- low-pass smoothing (EMA) -------------------------------------
        alpha = float(self.cfg.lowpass_alpha)
        self._ema = alpha * a + (1 - alpha) * self._ema
        a = self._ema.copy()

        # --- rate limiting on normalized action ---------------------------
        delta = a - self._prev
        maxd = self.cfg.max_delta_per_step
        if np.any(np.abs(delta) > maxd):
            delta = np.clip(delta, -maxd, maxd)
            a = self._prev + delta
            status.rate_limited = True

        # --- workspace clamp on the *predicted* EE target -----------------
        # action[:3] is a delta applied to current eef_pos by OSC; project the
        # resulting target into the safe box and back out the allowed delta.
        if eef_pos is not None:
            target = np.asarray(eef_pos, dtype=np.float64) + a[:3]
            clamped = clamp_vec(target, self.cfg.ws_low, self.cfg.ws_high)
            if not np.allclose(clamped, target):
                a[:3] = clamped - np.asarray(eef_pos, dtype=np.float64)
                status.clamped_workspace = True

        self._prev = a.copy()
        return a, status

    # -- helpers ------------------------------------------------------------
    def _trip(self, reason: str):
        self._tripped = True
        return (np.zeros(self.action_dim), SafetyStatus(estop=True, reason=reason))

    def _hold(self, status: SafetyStatus):
        status.estop = True
        status.reason = "estop latched"
        hold = np.zeros(self.action_dim)
        hold[-1] = self._prev[-1]        # keep last gripper command
        return hold, status

    @property
    def tripped(self) -> bool:
        return self._tripped
