"""Policy interface.

Everything that produces actions — scripted expert, SmolVLA, a future RL
policy — implements this. Keeping the surface tiny lets the eval script swap
policies with a flag and lets the safety layer wrap any of them identically.

Contract:
  reset(obs)        -> called once per episode with the first Observation
  act(obs)          -> returns a raw action vector in the env action space
                       (7-dim for OSC_POSE: [dx dy dz drx dry drz gripper]),
                       OR an action chunk of shape (T, 7) which the runner
                       consumes one row per control step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..envs.robosuite_env import Observation


class Policy(ABC):
    produces_chunks: bool = False      # True if act() may return (T, action_dim)

    def reset(self, obs: Observation) -> None:  # noqa: B027
        """Optional per-episode reset hook."""

    @abstractmethod
    def act(self, obs: Observation) -> np.ndarray:
        """Return action (action_dim,) or chunk (T, action_dim)."""
        raise NotImplementedError
