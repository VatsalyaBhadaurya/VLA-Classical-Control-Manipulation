"""Replay piper_dataset_1100 episodes in the MuJoCo PiPER sim.

Reproduces a recorded trajectory two ways:
  --mode kinematic : hard-set joints from observation.state each frame (a
                     faithful visual reproduction, independent of dynamics).
  --mode actuated  : feed the recorded `action` into the joint-position
                     controller and let physics run (tests the sim dynamics /
                     controller tracking against the real robot).

    python scripts/replay_dataset.py --episode 0 --mode kinematic --out out/replay.mp4

Also prints the dataset's task instructions (so we learn the real task) and the
feature schema. Uses LeRobot to stream frames — only the needed episode is
downloaded/cached, not the full 12.9 GB.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import numpy as np
from tqdm import trange


def load_dataset(repo_id: str):
    """Import LeRobotDataset across the 0.5/0.6 module layouts."""
    errors = []
    for path in ("lerobot.datasets.lerobot_dataset",
                 "lerobot.common.datasets.lerobot_dataset"):
        try:
            mod = __import__(path, fromlist=["LeRobotDataset"])
            return mod.LeRobotDataset(repo_id)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path}: {type(e).__name__}: {e}")
    raise ImportError("Could not load LeRobotDataset:\n  " + "\n  ".join(errors))


def episode_row_range(ds, ep: int):
    """Return (start, stop) global frame indices for episode `ep`, across API
    variants (episode_data_index tensors, or a filter fallback)."""
    idx = getattr(ds, "episode_data_index", None)
    if idx is not None and "from" in idx:
        return int(idx["from"][ep]), int(idx["to"][ep])
    # fallback: scan episode_index column
    col = ds.hf_dataset["episode_index"]
    col = np.asarray(col)
    where = np.where(col == ep)[0]
    if len(where) == 0:
        raise IndexError(f"episode {ep} not found")
    return int(where[0]), int(where[-1]) + 1


def to_np(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--mode", choices=["kinematic", "actuated"], default="kinematic")
    ap.add_argument("--camera", default="cam_global",
                    choices=["cam_global", "cam_gripper"])
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", default="out/replay.mp4")
    args = ap.parse_args()

    from configs.piper import make_piper_config
    from vla_classical.envs.piper_env import PiperMujocoEnv

    cfg = make_piper_config()
    cfg.camera.render_downsample = args.downsample

    print(f"[data] loading {cfg.dataset_repo_id} ...")
    ds = load_dataset(cfg.dataset_repo_id)
    meta = getattr(ds, "meta", None)
    tasks = getattr(meta, "tasks", None) if meta else None
    print(f"[data] episodes={getattr(ds, 'num_episodes', '?')} "
          f"fps={getattr(meta, 'fps', '?')}")
    print(f"[data] tasks={tasks}")
    print(f"[data] features={list(getattr(meta, 'features', {}) or [])}")

    start, stop = episode_row_range(ds, args.episode)
    stop = min(stop, start + args.max_frames)
    print(f"[replay] episode {args.episode}: frames [{start}, {stop})  mode={args.mode}")

    env = PiperMujocoEnv(cfg)
    env.reset()

    frames = []
    for i in trange(start, stop, desc="replay"):
        sample = ds[i]
        if args.mode == "kinematic":
            obs = env.set_state(to_np(sample["observation.state"]))
        else:
            obs = env.step(to_np(sample["action"]))
        frames.append(obs.images[args.camera])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    import imageio
    imageio.mimsave(args.out, frames, fps=int(getattr(meta, "fps", 30) or 30))
    print(f"[out] wrote {len(frames)} frames -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
