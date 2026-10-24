"""Evaluation metrics (Phase 6).

Tracks everything the spec asks to report, per episode and aggregated:
success rate, episode length, completion time, collisions, grasp success,
joint-torque smoothness, inference latency, and a failure category.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np


class FailureCategory(str, Enum):
    NONE = "none"
    PERCEPTION = "perception"
    GRASPING = "grasping"
    PLANNING = "planning"
    CONTROL = "control"
    TIMEOUT = "timeout"


@dataclass
class EpisodeMetrics:
    success: bool = False
    length: int = 0                       # control steps
    completion_time_s: float = 0.0
    collisions: int = 0
    grasp_success: bool = False
    torque_smoothness: float = 0.0        # mean |Δtorque| (lower = smoother)
    mean_infer_latency_ms: float = 0.0
    peak_vram_gb: float = 0.0
    estops: int = 0
    failure: FailureCategory = FailureCategory.NONE

    # running accumulators (not reported directly)
    _torques: list = field(default_factory=list, repr=False)
    _latencies: list = field(default_factory=list, repr=False)

    def record_torque(self, tau: np.ndarray):
        self._torques.append(np.asarray(tau, dtype=np.float64).copy())

    def record_latency(self, ms: float):
        self._latencies.append(float(ms))

    def finalize(self):
        if len(self._torques) > 1:
            diffs = np.diff(np.stack(self._torques), axis=0)
            self.torque_smoothness = float(np.mean(np.abs(diffs)))
        if self._latencies:
            self.mean_infer_latency_ms = float(np.mean(self._latencies))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_torques", None)
        d.pop("_latencies", None)
        d["failure"] = self.failure.value
        return d


class MetricsAggregator:
    def __init__(self):
        self.episodes: list[EpisodeMetrics] = []

    def add(self, m: EpisodeMetrics):
        m.finalize()
        self.episodes.append(m)

    def summary(self) -> dict:
        if not self.episodes:
            return {}
        arr = lambda k: np.array([getattr(e, k) for e in self.episodes], float)  # noqa: E731
        n = len(self.episodes)
        by_fail = {}
        for e in self.episodes:
            by_fail[e.failure.value] = by_fail.get(e.failure.value, 0) + 1
        return {
            "n_episodes": n,
            "success_rate": float(arr("success").mean()),
            "grasp_success_rate": float(arr("grasp_success").mean()),
            "avg_length": float(arr("length").mean()),
            "avg_completion_time_s": float(arr("completion_time_s").mean()),
            "avg_collisions": float(arr("collisions").mean()),
            "avg_torque_smoothness": float(arr("torque_smoothness").mean()),
            "avg_infer_latency_ms": float(arr("mean_infer_latency_ms").mean()),
            "peak_vram_gb": float(arr("peak_vram_gb").max()),
            "total_estops": int(arr("estops").sum()),
            "failure_breakdown": by_fail,
        }
