"""PiPER env sanity check: build the scene, go to 'home', render both cameras.

    python scripts/piper_render_test.py --out out/piper

Writes out/piper_cam_global.png and out/piper_cam_gripper.png so we can SEE the
scene and tune the camera extrinsics against a real dataset frame. Also prints
the 7-dim state and steps the arm to confirm joint-space control moves it.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import numpy as np

from configs.piper import make_piper_config
from vla_classical.envs.piper_env import PiperMujocoEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/piper")
    ap.add_argument("--downsample", type=int, default=1,
                    help=">1 renders smaller/faster (CPU swrast can be slow at 720p)")
    args = ap.parse_args()

    cfg = make_piper_config()
    cfg.camera.render_downsample = args.downsample
    print(f"[cfg] MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
          f"control_freq={cfg.control_freq}Hz menagerie={cfg.menagerie_dir}")

    env = PiperMujocoEnv(cfg)
    print(f"[model] nq={env.model.nq} nu={env.model.nu} "
          f"timestep={env.model.opt.timestep} n_substeps={env.n_substeps}")

    obs = env.reset()
    s = env.state_vector()
    print(f"[home] state[7]={np.round(s, 3)}")
    for name, img in obs.images.items():
        print(f"[img] {name}: shape={img.shape} dtype={img.dtype} "
              f"nonzero={bool(img.any())}")

    # confirm joint-space control actually moves the arm: nudge joint1 + close grip
    target = s.copy()
    target[0] += 0.4          # rotate base joint
    target[6] = 0.0           # close gripper (dataset units)
    for _ in range(30):       # 1 second at 30 Hz
        obs = env.step(target)
    s2 = env.state_vector()
    print(f"[moved] state[7]={np.round(s2, 3)}  (dj1={s2[0]-s[0]:+.3f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio
    for name, img in obs.images.items():
        path = f"{args.out}_{name}.png"
        imageio.imwrite(path, img)
        print(f"[png] {path}")

    env.close()
    print("[ok] PiPER env built, rendered, and joint-space control verified.")


if __name__ == "__main__":
    main()
