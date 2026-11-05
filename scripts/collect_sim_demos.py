"""Collect SIM demonstrations with the classical pipeline -> LeRobot dataset.

The classical perceive->plan->IK->execute pipeline (plan_pick) is the expert.
For each randomized scene we run it, and if the object lands in the box we
record the whole episode: cam_global + cam_gripper images, the 7-DoF state, and
the executed 7-DoF joint action, plus the task instruction.

Crucially the recorded IMAGES are the sim renders, so a SmolVLA fine-tuned on
this dataset sees the same visual domain at train and inference time (unlike the
real piper_dataset_1100). Schema matches that dataset (cam_global/cam_gripper,
state[7], action[7], 30 fps) so the same lerobot-train recipe + rename_map works.

    python scripts/collect_sim_demos.py --num 250 \
        --repo-id local/piper_sim \
        --root /mnt/nextcloud-hdd/hf/datasets/piper_sim

Then fine-tune (on the box with the vla312 env):
    lerobot-train --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=local/piper_sim \
        --dataset.root=/mnt/nextcloud-hdd/hf/datasets/piper_sim \
        --tolerance_s=0.05 --batch_size=16 --steps=40000 --save_freq=10000 \
        --num_workers=8 --policy.device=cuda --policy.push_to_hub=false \
        --output_dir=/mnt/nextcloud-hdd/hf/outputs/piper_sim_vla --job_name=piper_sim_vla \
        --trackio.enable=false \
        --rename_map='{"observation.images.cam_global": "observation.images.camera1", "observation.images.cam_gripper": "observation.images.camera2"}'
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np
from tqdm import tqdm

INSTRUCTIONS = {
    "yellow_block": "pick the yellow block put into the box",
    "black_block": "pick the black block put into the box",
}


def _add_frame(ds, frame, task):
    """add_frame signature varies across lerobot versions."""
    try:
        ds.add_frame(frame, task=task)
    except TypeError:
        frame = dict(frame, task=task)
        ds.add_frame(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=250, help="target SUCCESSFUL episodes")
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = num*2")
    ap.add_argument("--object", default="yellow_block",
                    choices=["yellow_block", "black_block"])
    ap.add_argument("--repo-id", default="local/piper_sim")
    ap.add_argument("--root", required=True, help="dataset output dir (use the big drive)")
    ap.add_argument("--downsample", type=int, default=2, help="record image shrink factor")
    args = ap.parse_args()
    max_attempts = args.max_attempts or args.num * 2

    from configs.piper import make_piper_config, randomize_piper_scene
    from vla_classical.envs.piper_env import PiperMujocoEnv
    from vla_classical.perception import estimate_object
    from vla_classical.control import JointSafetyFilter
    from vla_classical.control.ik import DifferentialIK
    from vla_classical.planning import plan_grasp, build_pick_place_trajectory
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = make_piper_config()
    cfg.camera.render_downsample = args.downsample
    cfg.instruction = INSTRUCTIONS[args.object]
    d = args.downsample
    gh, gw = cfg.camera.cam_global_hw[0] // d, cfg.camera.cam_global_hw[1] // d
    wh, ww = cfg.camera.cam_gripper_hw[0] // d, cfg.camera.cam_gripper_hw[1] // d

    features = {
        "observation.images.cam_global": {
            "dtype": "video", "shape": (gh, gw, 3),
            "names": ["height", "width", "channel"]},
        "observation.images.cam_gripper": {
            "dtype": "video", "shape": (wh, ww, 3),
            "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=cfg.control_freq, root=args.root,
        features=features, use_videos=True)
    print(f"[collect] target={args.num} object={args.object} "
          f"img cam_global={gh}x{gw} cam_gripper={wh}x{ww} -> {args.root}")

    kept, attempts, graspable_fail, ik_fail, exec_fail = 0, 0, 0, 0, 0
    pbar = tqdm(total=args.num, desc="episodes")
    while kept < args.num and attempts < max_attempts:
        seed = attempts
        attempts += 1
        randomize_piper_scene(cfg, seed)
        env = PiperMujocoEnv(cfg)
        try:
            env.reset()
            obs = env.set_state(np.array(cfg.rest_state))
            est = estimate_object(env, args.object, camera="cam_global", res=(256, 256))
            box_center, _ = env.body_pose("bin")
            grasp = plan_grasp(est, box_center, cfg)
            if not grasp.graspable:
                graspable_fail += 1
                env.close()
                continue
            ik = DifferentialIK(env)
            traj, info = build_pick_place_trajectory(ik, grasp, env.state_vector()[:6])
            if not info["ik_ok"]:
                ik_fail += 1
                env.close()
                continue

            safety = JointSafetyFilter(cfg)
            safety.reset(init_target=env.state_vector())
            buf = []
            for tgt in traj:
                safe, _ = safety.filter(tgt, joint_vel=env.joint_vel)
                buf.append({
                    "observation.images.cam_global": obs.images["cam_global"],
                    "observation.images.cam_gripper": obs.images["cam_gripper"],
                    "observation.state": np.asarray(obs.raw["state"], np.float32),
                    "action": safe.astype(np.float32),
                })
                obs = env.step(safe, render=True)

            objp, _ = env.body_pose(args.object)
            ih = cfg.scene.box.inner_half
            success = (abs(objp[0] - box_center[0]) < ih[0] + 0.01 and
                       abs(objp[1] - box_center[1]) < ih[1] + 0.01 and
                       objp[2] < cfg.scene.box.height)
            if success:
                for f in buf:
                    _add_frame(ds, f, cfg.instruction)
                ds.save_episode()
                kept += 1
                pbar.update(1)
            else:
                exec_fail += 1
        except Exception as e:  # noqa: BLE001
            exec_fail += 1
            tqdm.write(f"[ep {seed}] error: {type(e).__name__}: {e}")
        finally:
            env.close()

    pbar.close()
    print(f"[done] kept {kept}/{attempts} attempts "
          f"(ungraspable={graspable_fail} ik_fail={ik_fail} exec/other_fail={exec_fail})")
    print(f"[done] dataset at {args.root} (repo_id={args.repo_id})")


if __name__ == "__main__":
    main()
