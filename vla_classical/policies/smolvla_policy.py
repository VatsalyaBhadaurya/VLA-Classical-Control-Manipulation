"""SmolVLA-450M policy via LeRobot (Phase 4/5).

Wraps a LeRobot SmolVLA policy behind the same `Policy` interface as the
scripted expert, so evaluation and the safety layer treat them identically.

Design for 6 GB VRAM:
  * fp16 weights + autocast inference.
  * Action *chunking*: SmolVLA predicts a short chunk; we cache it and return
    one action per control step, re-querying every `action_chunk` steps (this
    is what gives the 5–10 Hz effective inference rate vs the 20 Hz control
    loop). `chunk_truncate` lets you use only the first N actions of each chunk.
  * Optional CPU fallback if CUDA OOMs.

This is a scaffold: the exact LeRobot API (policy class, input/output feature
names, normalization) is pinned at install time. The TODOs mark the spots to
wire once `lerobot[smolvla]` is installed and you've inspected the model's
expected feature dict. Keep it unused until Phase 4.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from configs import VLAConfig
from .base import Policy
from ..envs.robosuite_env import Observation


class SmolVLAPolicy(Policy):
    produces_chunks = False     # we internally buffer the chunk, emit per-step

    def __init__(self, cfg: VLAConfig, instruction: str):
        self.cfg = cfg
        self.instruction = instruction
        self._buffer: list[np.ndarray] = []
        self._model = None
        self._device = None
        self._load()

    def _load(self):
        import torch
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy as LRSmolVLA  # noqa: E501

        self._device = (self.cfg.device
                        if (self.cfg.device == "cpu" or torch.cuda.is_available())
                        else "cpu")
        dtype = getattr(torch, self.cfg.dtype)
        # TODO(phase4): confirm from_pretrained signature / revision pinning.
        self._model = LRSmolVLA.from_pretrained(self.cfg.model_id)
        self._model.to(self._device)
        if self._device != "cpu":
            self._model.half()
        self._model.eval()
        self._torch = torch
        self._dtype = dtype

    def reset(self, obs: Optional[Observation] = None):
        self._buffer = []
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()

    def act(self, obs: Observation) -> np.ndarray:
        if not self._buffer:
            self._buffer = list(self._infer_chunk(obs))
        return self._buffer.pop(0)

    def _infer_chunk(self, obs: Observation) -> np.ndarray:
        torch = self._torch
        batch = self._build_batch(obs)      # TODO(phase4): match model features
        with torch.no_grad(), torch.autocast(
                device_type="cuda" if self._device != "cpu" else "cpu",
                dtype=self._dtype, enabled=self._device != "cpu"):
            actions = self._model.select_action(batch)   # TODO: API name
        actions = np.asarray(actions.float().cpu().numpy())
        if actions.ndim == 1:
            actions = actions[None]
        if self.cfg.chunk_truncate:
            actions = actions[: self.cfg.chunk_truncate]
        return actions

    def _build_batch(self, obs: Observation) -> dict:
        """Map our Observation into the feature dict LeRobot expects.

        TODO(phase4): names like 'observation.images.top', 'observation.state',
        'task' depend on how the dataset was recorded (see data/recorder.py).
        Keep the recorder and this function in sync.
        """
        torch = self._torch
        cam0 = next(iter(obs.images.values()))
        img = torch.from_numpy(cam0).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(obs.proprio_vector()).float()
        return {
            "observation.images.top": img[None].to(self._device),
            "observation.state": state[None].to(self._device),
            "task": [self.instruction],
        }
