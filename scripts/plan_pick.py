"""Classical perception -> grasp/trajectory planning -> execution in the PiPER sim.

    python scripts/plan_pick.py --object yellow_block --out out/plan_pick.mp4

Pipeline:
  1. estimate the object's x/y/z dims from depth+segmentation,
  2. plan a top-down grasp (gripper width from the short side) + a place-in-box
     trajectory, solving IK at each key-pose,
  3. execute through the JointSafetyFilter and record cam_global,
  4. report whether the object ended up inside the box.

This is the classical (no-VLA) pick-and-place baseline driven by perception.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="yellow_block")
    ap.add_argument("--camera", default="cam_global")
    ap.add_argument("--downsample", type=int, default=3, help="video render scale")
    ap.add_argument("--seed", type=int, default=-1,
                    help=">=0 randomizes the scene (object sizes/positions + box)")
    ap.add_argument("--out", default="out/plan_pick.mp4")
    args = ap.parse_args()

    from configs.piper import make_piper_config, randomize_piper_scene
    from vla_classical.envs.piper_env import PiperMujocoEnv
    from vla_classical.perception import estimate_object
    from vla_classical.control import JointSafetyFilter
    from vla_classical.control.ik import DifferentialIK
    from vla_classical.planning import plan_grasp, build_pick_place_trajectory

    cfg = make_piper_config()
    cfg.camera.render_downsample = args.downsample
    if args.seed >= 0:
        randomize_piper_scene(cfg, args.seed)
        print(f"[scene] seed={args.seed} "
              + " ".join(f"{c.name}@{tuple(round(v,3) for v in c.pos)}"
                         f"(half={c.half:.3f})" for c in cfg.scene.cubes)
              + f" box@{tuple(round(v,3) for v in cfg.scene.box.pos)}")
    env = PiperMujocoEnv(cfg)
    env.reset()
    env.set_state(np.array(cfg.rest_state))

    # 1) perceive
    est = estimate_object(env, args.object, camera=args.camera, res=(360, 640))
    print(f"[perceive] {args.object}: dims={np.round(est.dims, 4)} "
          f"center={np.round(est.center, 4)} yaw={np.degrees(est.yaw):.0f}deg")

    # 2) plan
    box_center, _ = env.body_pose("bin")
    grasp = plan_grasp(est, box_center, cfg)
    print(f"[plan] gripper open={grasp.gripper_open:.3f} close={grasp.gripper_close:.3f} "
          f"graspable={grasp.graspable} {grasp.note}")
    print(f"[plan] grasp_pos={np.round(grasp.grasp_pos,3)} "
          f"place_pos={np.round(grasp.place_pos,3)} box={np.round(box_center,3)}")

    ik = DifferentialIK(env)
    traj, info = build_pick_place_trajectory(ik, grasp, env.state_vector()[:6])
    print(f"[ik] converged={info['ik_ok']} per-pose err={info['ik_errs']} "
          f"steps={info['n_steps']}")
    if not info["ik_ok"]:
        print("[ik] WARNING: some key-pose IK did not fully converge "
              "(target may be out of reach / orientation infeasible).")

    # 3) execute through the safety layer
    safety = JointSafetyFilter(cfg)
    safety.reset(init_target=env.state_vector())
    frames, estops = [], 0
    for tgt in tqdm(traj, desc="execute"):
        safe, st = safety.filter(tgt, joint_vel=env.joint_vel)
        env.step(safe, render=False)
        frames.append(env.render("cam_global"))
        estops += int(st.estop)

    # 4) report
    obj_pos, _ = env.body_pose(args.object)
    ih = cfg.scene.box.inner_half
    inside = (abs(obj_pos[0] - box_center[0]) < ih[0] + 0.01 and
              abs(obj_pos[1] - box_center[1]) < ih[1] + 0.01 and
              obj_pos[2] < cfg.scene.box.height)
    print(f"[result] object final pos={np.round(obj_pos,3)} | "
          f"in_box={inside} | estops={estops}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio
    imageio.mimsave(args.out, frames, fps=cfg.control_freq)
    print(f"[out] wrote {len(frames)} frames -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
