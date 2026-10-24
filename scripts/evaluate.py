"""Phase 5/6: evaluate a policy, optionally with the safety layer / DR.

Baselines (run each separately, per the spec):
  classical-only     : --policy scripted
  VLA-only           : --policy smolvla
  VLA + classical    : --policy smolvla --safety
  VLA + classical+DR : --policy smolvla --safety --randomize

    python scripts/evaluate.py --task lift --policy scripted --episodes 20
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import time

import numpy as np
from tqdm import trange

from configs import make_config
from vla_classical.envs import ManipulationEnv
from vla_classical.control import SafetyFilter
from vla_classical.policies import ScriptedLiftPolicy
from vla_classical.eval import EpisodeMetrics, MetricsAggregator
from vla_classical.eval.metrics import FailureCategory


def build_policy(name, cfg):
    if name == "scripted":
        return ScriptedLiftPolicy(cfg.controller)
    if name == "smolvla":
        from vla_classical.policies.smolvla_policy import SmolVLAPolicy
        return SmolVLAPolicy(cfg.vla, cfg.task.instruction)
    raise ValueError(f"unknown policy '{name}'")


def peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def run_episode(env, policy, safety, cfg, use_safety) -> EpisodeMetrics:
    m = EpisodeMetrics()
    obs = env.reset()
    if safety:
        safety.reset()
    policy.reset(obs)
    t0 = time.time()
    success = False

    for step in range(cfg.sim.horizon):
        ti = time.time()
        raw = policy.act(obs)
        m.record_latency((time.time() - ti) * 1000.0)

        if use_safety and safety is not None:
            action, status = safety.filter(raw, eef_pos=obs.eef_pos,
                                            joint_vel=env.joint_vel)
            if status.estop:
                m.estops += 1
        else:
            action = raw

        obs, reward, done, info = env.step(action)
        m.length = step + 1
        # torque smoothness from the actuator forces
        try:
            m.record_torque(env._env.sim.data.actuator_force.copy())
        except Exception:  # noqa: BLE001
            pass
        if info.get("success"):
            success = True
        if done:
            break

    m.completion_time_s = time.time() - t0
    m.success = success
    m.grasp_success = success          # refine with contact check per task
    m.peak_vram_gb = peak_vram_gb()
    if not success:
        m.failure = (FailureCategory.TIMEOUT if m.length >= cfg.sim.horizon
                     else FailureCategory.GRASPING)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lift")
    ap.add_argument("--policy", default="scripted", choices=["scripted", "smolvla"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--safety", action="store_true", help="enable classical safety layer")
    ap.add_argument("--randomize", action="store_true", help="domain randomization")
    ap.add_argument("--out", default="out/eval.json")
    args = ap.parse_args()

    cfg = make_config(args.task)
    cfg.safety.enabled = args.safety
    cfg.task.randomize.enabled = args.randomize
    # NOTE(phase6): wire cfg.task.randomize into the env builder (object pose,
    # color, lighting, camera, friction). Stubbed here.

    env = ManipulationEnv(cfg)
    safety = SafetyFilter(cfg.safety, action_dim=env.action_dim) if args.safety else None
    policy = build_policy(args.policy, cfg)

    agg = MetricsAggregator()
    label = f"{args.policy}{'+safety' if args.safety else ''}{'+DR' if args.randomize else ''}"
    for _ in trange(args.episodes, desc=f"eval[{label}]"):
        agg.add(run_episode(env, policy, safety, cfg, args.safety))

    summary = {"config": label, **agg.summary()}
    print(json.dumps(summary, indent=2))
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary,
                   "episodes": [e.to_dict() for e in agg.episodes]}, f, indent=2)
    print(f"[out] {args.out}")
    env.close()


if __name__ == "__main__":
    main()
