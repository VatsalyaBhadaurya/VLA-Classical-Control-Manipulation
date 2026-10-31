"""Validate object-dimension estimation from simulated depth + segmentation.

    python scripts/estimate_object.py --camera cam_global

For each task cube it prints the ESTIMATED (x,y,z) dims + center next to the
MuJoCo ground truth, so we can confirm the perception is correct before using it
for grasp/trajectory planning. Saves a segmentation-overlay debug image.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_global",
                    choices=["cam_global", "cam_gripper"])
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    from configs.piper import make_piper_config
    from vla_classical.envs.piper_env import PiperMujocoEnv
    from vla_classical.perception import estimate_object

    cfg = make_piper_config()
    env = PiperMujocoEnv(cfg)
    env.reset()
    env.set_state(np.array(cfg.rest_state))        # objects settle on table

    res = (args.height, args.width)
    print(f"[cam] {args.camera} @ {res}")
    for cube in cfg.scene.cubes:
        true_pos, _ = env.body_pose(cube.name)
        true_dims = np.array([2 * cube.half] * 3)
        try:
            est = estimate_object(env, cube.name, camera=args.camera, res=res)
        except RuntimeError as e:
            print(f"[{cube.name}] estimate failed: {e}")
            continue
        d_err = est.dims - true_dims
        c_err = est.center - true_pos
        print(f"\n[{cube.name}]  ({est.n_points} px)")
        print(f"  dims  est={np.round(est.dims, 4)}  true={np.round(true_dims, 4)}"
              f"  err={np.round(d_err, 4)}")
        print(f"  oriented w,l={np.round(est.oriented_wl, 4)}  yaw={np.degrees(est.yaw):.1f} deg")
        print(f"  center est={np.round(est.center, 4)}  true={np.round(true_pos, 4)}"
              f"  err={np.round(c_err, 4)}")

    # debug: save a segmentation overlay on the rgb for the first cube
    os.makedirs(args.out, exist_ok=True)
    import imageio
    h, w = res
    rgb = np.asarray(env.percep_render(args.camera, h, w, mode="rgb"))[..., :3].copy()
    seg = np.asarray(env.percep_render(args.camera, h, w, mode="seg"))
    seg_id = seg[..., 0] if seg.ndim == 3 else seg
    for cube, color in zip(cfg.scene.cubes, ([255, 0, 255], [0, 255, 255])):
        rgb[seg_id == env.geom_id(cube.name)] = color
    path = f"{args.out}/estimate_{args.camera}.png"
    imageio.imwrite(path, rgb.astype(np.uint8))
    print(f"\n[dbg] segmentation overlay -> {path}")
    env.close()


if __name__ == "__main__":
    main()
