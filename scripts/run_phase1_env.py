"""Phase 1 sanity check: build the env, step physics, render, write a video.

Verifies rendering + physics + the action->safety->controller->sim path
*before* any VLA. Drives a gentle, safe oscillation of the EE through the
SafetyFilter and checks for NaNs / instability.

    python scripts/run_phase1_env.py --task lift --steps 200 --out out/phase1.mp4
    python scripts/run_phase1_env.py --task lift --scripted   # run the expert
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os
import time

import numpy as np

from configs import make_config
from vla_classical.envs import ManipulationEnv
from vla_classical.control import SafetyFilter
from vla_classical.policies import ScriptedLiftPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lift")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="out/phase1.mp4")
    ap.add_argument("--scripted", action="store_true",
                    help="drive the scripted expert instead of an oscillation")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    cfg = make_config(args.task)
    print(f"[cfg] task={cfg.task.name} env={cfg.task.robosuite_env} "
          f"cam={cfg.camera.names} {cfg.camera.height}x{cfg.camera.width} "
          f"control_freq={cfg.sim.control_freq}Hz MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    env = ManipulationEnv(cfg)
    safety = SafetyFilter(cfg.safety, action_dim=env.action_dim)
    policy = ScriptedLiftPolicy(cfg.controller) if args.scripted else None

    obs = env.reset()
    safety.reset()
    if policy:
        policy.reset(obs)

    print(f"[obs] image keys={list(obs.images.keys())} "
          f"img0 shape={next(iter(obs.images.values())).shape} "
          f"eef_pos={np.round(obs.eef_pos, 3)} "
          f"proprio_dim={obs.proprio_vector().shape[0]}")

    frames = []
    t0 = time.time()
    estops = 0
    for i in range(args.steps):
        if policy:
            raw = policy.act(obs)
        else:
            # gentle safe oscillation: small xy circle, hold orientation, open grip
            phase = 2 * np.pi * i / 50.0
            raw = np.array([0.2 * np.sin(phase), 0.2 * np.cos(phase),
                            0.0, 0, 0, 0, -1.0])

        action, status = safety.filter(raw, eef_pos=obs.eef_pos,
                                        joint_vel=env.joint_vel)
        if status.estop:
            estops += 1
            print(f"[estop] step {i}: {status.reason}")

        obs, reward, done, info = env.step(action)

        if not np.all(np.isfinite(obs.eef_pos)):
            raise RuntimeError(f"NaN in eef_pos at step {i} — instability")

        if not args.no_video:
            frames.append(next(iter(obs.images.values())))
        if done:
            print(f"[done] step {i} success={info.get('success')}")
            break

    dt = time.time() - t0
    print(f"[perf] {i + 1} steps in {dt:.2f}s = {(i + 1) / dt:.1f} steps/s "
          f"| estops={estops} | final eef={np.round(obs.eef_pos, 3)}")

    if not args.no_video and frames:
        import imageio
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        imageio.mimsave(args.out, frames, fps=cfg.sim.control_freq)
        print(f"[video] wrote {len(frames)} frames -> {args.out}")

    env.close()
    print("[ok] Phase 1 env verified.")


if __name__ == "__main__":
    main()
