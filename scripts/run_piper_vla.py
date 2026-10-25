"""Hybrid rollout: SmolVLA -> joint safety filter -> PiPER MuJoCo env.

    python scripts/run_piper_vla.py --steps 300 --out out/piper_vla.mp4
    python scripts/run_piper_vla.py --ckpt outputs/train/piper_smolvla/checkpoints/last/pretrained_model

Runs the fine-tuned SmolVLA in the dataset-matched sim through the classical
joint-space safety layer, and records cam_global. This is the PiPER equivalent
of the "VLA + classical controller" baseline.

Requires a fine-tuned checkpoint (see scripts/finetune_piper.sh) — the
pretrained smolvla_base is not meaningful zero-shot on PiPER.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os
import time

import numpy as np
from tqdm import trange


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="override cfg.vla_ckpt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--no-safety", action="store_true", help="VLA-only (no filter)")
    ap.add_argument("--out", default="out/piper_vla.mp4")
    args = ap.parse_args()

    from configs.piper import make_piper_config
    from vla_classical.envs.piper_env import PiperMujocoEnv
    from vla_classical.control import JointSafetyFilter
    from vla_classical.policies.smolvla_piper import SmolVLAJointPolicy

    cfg = make_piper_config()
    cfg.camera.render_downsample = args.downsample
    if args.ckpt:
        cfg.vla_ckpt = args.ckpt
    if args.no_safety:
        cfg.safety.enabled = False

    print(f"[cfg] ckpt={cfg.vla_ckpt} device={cfg.vla_device} "
          f"safety={cfg.safety.enabled} instruction={cfg.instruction!r}")

    env = PiperMujocoEnv(cfg)
    env.reset()
    obs = env.set_state(np.array(cfg.rest_state))        # start like the real robot

    policy = SmolVLAJointPolicy(cfg)
    policy.reset()

    safety = JointSafetyFilter(cfg)
    safety.reset(init_target=env.state_vector())         # seed with current state

    frames, latencies, estops = [], [], 0
    for _ in trange(args.steps, desc="rollout"):
        t0 = time.time()
        raw = policy.act(obs)
        latencies.append((time.time() - t0) * 1000.0)

        action, st = safety.filter(raw, joint_vel=env.joint_vel)
        if st.estop:
            estops += 1
        obs = env.step(action)
        frames.append(obs.images["cam_global"])

    print(f"[perf] mean VLA latency={np.mean(latencies):.0f} ms "
          f"({1000/np.mean(latencies):.1f} Hz) | estops={estops}")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[vram] peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    except Exception:  # noqa: BLE001
        pass

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio
    imageio.mimsave(args.out, frames, fps=cfg.control_freq)
    print(f"[out] wrote {len(frames)} frames -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
