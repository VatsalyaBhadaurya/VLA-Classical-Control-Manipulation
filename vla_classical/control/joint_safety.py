"""Joint-space safety filter for the PiPER track.

The PiPER dataset action is an **absolute joint-position target** [j1..j6, grip],
so the classical "HOW" layer here differs from the OSC/Cartesian SafetyFilter:

  * clamp each joint target to the model's joint limits; clamp gripper to [0, gmax]
  * rate-limit the per-step change (protects against VLA outliers / jumps)
  * low-pass (EMA) smoothing for jerk-free motion
  * E-stop on non-finite commands or measured joint over-speed (latching)

Sits between the VLA (joint-target chunk) and PiperMujocoEnv.step().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs.piper import PiperConfig


@dataclass
class JointSafetyStatus:
    estop: bool = False
    reason: str = ""
    clamped: bool = False
    rate_limited: bool = False


class JointSafetyFilter:
    def __init__(self, cfg: PiperConfig):
        self.s = cfg.safety
        self.low = np.asarray(cfg.joint_low, dtype=np.float64)
        self.high = np.asarray(cfg.joint_high, dtype=np.float64)
        self.gmax = float(cfg.gripper_dataset_max)
        self.reset()

    def reset(self, init_target: np.ndarray | None = None):
        """Seed the filter with the robot's current state so the first command
        doesn't jump from zero."""
        if init_target is None:
            self._prev = None
            self._ema = None
        else:
            self._prev = np.asarray(init_target, dtype=np.float64).ravel().copy()
            self._ema = self._prev.copy()
        self._tripped = False

    def filter(self, action: np.ndarray, joint_vel: np.ndarray | None = None):
        st = JointSafetyStatus()
        if not self.s.enabled:
            return np.asarray(action, dtype=np.float64).ravel(), st

        a = np.asarray(action, dtype=np.float64).ravel().copy()

        # --- E-stop conditions -------------------------------------------
        if self.s.estop_on_nan and not np.all(np.isfinite(a)):
            return self._trip("non-finite action")
        if joint_vel is not None and np.max(np.abs(joint_vel)) > self.s.max_joint_vel:
            return self._trip(f"joint vel {np.max(np.abs(joint_vel)):.2f} > "
                              f"{self.s.max_joint_vel}")
        if self._tripped:
            return self._hold()

        # --- clamp to limits ---------------------------------------------
        clamped = a.copy()
        clamped[:6] = np.clip(a[:6], self.low, self.high)
        clamped[6] = np.clip(a[6], 0.0, self.gmax)
        st.clamped = not np.allclose(clamped, a)
        a = clamped

        if self._prev is None:          # first call: accept as-is (seeded elsewhere)
            self._prev = a.copy()
            self._ema = a.copy()
            return a, st

        # --- rate limit ---------------------------------------------------
        d = a - self._prev
        d6 = np.clip(d[:6], -self.s.max_joint_step, self.s.max_joint_step)
        dg = np.clip(d[6], -self.s.max_gripper_step, self.s.max_gripper_step)
        st.rate_limited = (not np.allclose(d6, d[:6])) or (dg != d[6])
        a = self._prev + np.concatenate([d6, [dg]])

        # --- low-pass smoothing ------------------------------------------
        al = self.s.lowpass_alpha
        self._ema = al * a + (1 - al) * self._ema
        a = self._ema.copy()
        a[:6] = np.clip(a[:6], self.low, self.high)
        a[6] = np.clip(a[6], 0.0, self.gmax)

        self._prev = a.copy()
        return a, st

    # ------------------------------------------------------------------
    def _trip(self, reason: str):
        self._tripped = True
        return self._hold(reason)

    def _hold(self, reason: str = "estop latched"):
        hold = (self._prev.copy() if self._prev is not None
                else np.zeros(7, dtype=np.float64))
        return hold, JointSafetyStatus(estop=True, reason=reason)

    @property
    def tripped(self) -> bool:
        return self._tripped
