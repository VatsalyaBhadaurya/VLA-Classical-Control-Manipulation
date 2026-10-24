"""Phase 3: collect scripted demonstrations into a LeRobot dataset.

    python scripts/collect_demos.py --task lift --episodes 50 --policy scripted

Each step records images + proprio + raw action + applied (post-safety) action
+ instruction; each episode is labeled by task success. Aim for 50–200 clean
successes before fine-tuning SmolVLA.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np
from tqdm import trange

from configs import make_config
from vla_classical.envs import ManipulationEnv
from vla_classical.control import SafetyFilter
from vla_classical.policies import ScriptedLiftPolicy
from vla_classical.data import DemoRecorder


def build_policy(name, cfg):
    if name == "scripted":
        return ScriptedLiftPolicy(cfg.controller)
    raise ValueError(f"unknown policy '{name}' for data collection")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lift")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--policy", default="scripted")
    ap.add_argument("--only-success", action="store_true",
                    help="discard failed episodes from the dataset")
    args = ap.parse_args()

    cfg = make_config(args.task)
    env = ManipulationEnv(cfg)
    safety = SafetyFilter(cfg.safety, action_dim=env.action_dim)
    recorder = DemoRecorder(cfg)

    kept = 0
    for ep in trange(args.episodes, desc="episodes"):
        obs = env.reset()
        safety.reset()
        policy = build_policy(args.policy, cfg)
        policy.reset(obs)

        steps = []
        success = False
        for _ in range(cfg.sim.horizon):
            raw = policy.act(obs)
            action, _ = safety.filter(raw, eef_pos=obs.eef_pos,
                                      joint_vel=env.joint_vel)
            steps.append((obs, raw, action))
            obs, reward, done, info = env.step(action)
            if info.get("success"):
                success = True
            if done:
                break

        if args.only_success and not success:
            continue
        for (o, raw, applied) in steps:
            recorder.add_step(o, raw, applied)
        recorder.end_episode(success)
        kept += 1

    recorder.finalize()
    print(f"[done] recorded {kept}/{args.episodes} episodes to "
          f"{cfg.data.root} (repo_id={cfg.data.repo_id})")
    env.close()


if __name__ == "__main__":
    main()
