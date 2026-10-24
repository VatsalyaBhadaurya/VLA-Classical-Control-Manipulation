"""Peek at piper_dataset_1100: print tasks/schema and dump REAL camera frames.

    python scripts/peek_dataset.py --episode 0

Saves the dataset's own recorded frames to out/ so we can match the sim cameras,
lighting, and scene objects to the real rig, and prints the task instruction(s)
plus sample state/action values.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import numpy as np


def load_dataset(repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id)


def to_img(x) -> np.ndarray:
    x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3):     # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = (x * 255.0).clip(0, 255) if x.max() <= 1.0 + 1e-6 else x.clip(0, 255)
        x = x.astype(np.uint8)
    return np.ascontiguousarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    from configs.piper import make_piper_config
    cfg = make_piper_config()

    ds = load_dataset(cfg.dataset_repo_id)
    meta = ds.meta
    print("=" * 60)
    print("TASKS       :", getattr(meta, "tasks", None))
    print("FEATURES    :", list(getattr(meta, "features", {}) or []))
    print("N_EPISODES  :", getattr(ds, "num_episodes", "?"))
    print("FPS         :", getattr(meta, "fps", "?"))
    print("=" * 60)

    # row range for the episode
    idx = getattr(ds, "episode_data_index", None)
    if idx is not None and "from" in idx:
        start, stop = int(idx["from"][args.episode]), int(idx["to"][args.episode])
    else:
        start, stop = 0, len(ds)
    mid = (start + stop) // 2

    os.makedirs(args.out, exist_ok=True)
    import imageio
    img_keys = [k for k in meta.features if k.startswith("observation.images.")]
    for row, tag in ((start, "f0"), (mid, "mid")):
        sample = ds[row]
        print(f"\n[row {row}] task_index={int(np.asarray(sample.get('task_index', -1)))}")
        print(f"  state : {np.round(np.asarray(sample['observation.state'], float), 3)}")
        print(f"  action: {np.round(np.asarray(sample['action'], float), 3)}")
        for k in img_keys:
            cam = k.split('.')[-1]
            path = f"{args.out}/real_{cam}_ep{args.episode}_{tag}.png"
            imageio.imwrite(path, to_img(sample[k]))
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
