"""Build a balanced sim+real mixed LeRobot dataset for co-training.

lerobot 0.6.x can't train on multiple datasets at once (MultiLeRobotDataset is
disabled), so we physically merge into ONE dataset:
  * sim episodes (from collect_sim_demos), REPEATED `--sim-repeat` times to
    upweight the sim visual domain,
  * a subset of `--real-episodes` real episodes, images RESIZED down to the sim
    resolution (a LeRobot feature has a single fixed shape).

Both tracks share the schema (cam_global/cam_gripper, state[7], action[7], same
PiPER joint space), so merging is clean. Train on the output with the usual
lerobot-train + rename_map recipe.

    python scripts/build_mixed_dataset.py \
        --sim-root /mnt/nextcloud-hdd/hf/datasets/piper_sim \
        --sim-repeat 3 --real-episodes 150 \
        --out-root /mnt/nextcloud-hdd/hf/datasets/piper_mixed

Balance (defaults): sim 250 eps x3 (~138k frames) vs real 150 eps (~81k) ->
~60/40 sim-favored, good for SIM deployment. Tune --sim-repeat / --real-episodes.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

import numpy as np


def _to_uint8_hwc(t, size):
    """Any image tensor/array -> uint8 HWC resized to `size` (h, w)."""
    import torch
    import torch.nn.functional as F
    t = t if hasattr(t, "dim") else torch.as_tensor(np.asarray(t))
    x = (t.unsqueeze(0) if (t.dim() == 3 and t.shape[0] in (1, 3))
         else t.permute(2, 0, 1).unsqueeze(0)).float()
    if float(x.max()) > 1.5:          # uint8-range -> [0,1]
        x = x / 255.0
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    x = (x.clamp(0, 1) * 255.0).to(torch.uint8)[0].permute(1, 2, 0).contiguous()
    return np.ascontiguousarray(x.numpy())


def _add(ds, frame, task):
    try:
        ds.add_frame(frame, task=task)
    except TypeError:
        ds.add_frame(dict(frame, task=task))


def _task_of(item, meta):
    if isinstance(item.get("task"), str):
        return item["task"]
    ti = int(np.asarray(item["task_index"]))
    tasks = getattr(meta, "tasks", None)
    try:
        return tasks.iloc[ti].name if hasattr(tasks, "iloc") else tasks[ti]
    except Exception:  # noqa: BLE001
        return str(ti)


def _episode_ranges(src):
    """(start, stop) row ranges per episode, across lerobot versions."""
    idx = getattr(src, "episode_data_index", None)
    if idx is not None and "from" in idx:
        return [(int(idx["from"][e]), int(idx["to"][e]))
                for e in range(src.num_episodes)]
    ep = np.asarray(src.hf_dataset["episode_index"]).astype(np.int64).ravel()
    changes = np.where(np.diff(ep) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    stops = np.concatenate([changes, [len(ep)]])
    return list(zip(starts.tolist(), stops.tolist()))


def _copy(src, out, shapes, meta, repeat, limit_eps, desc):
    from tqdm import tqdm
    (gh, gw), (wh, ww) = shapes
    ranges = _episode_ranges(src)
    if limit_eps is not None:
        ranges = ranges[:limit_eps]
    for r in range(repeat):
        for (start, stop) in tqdm(ranges, desc=f"{desc}{'' if repeat == 1 else f' x{r+1}'}"):
            for i in range(start, stop):
                it = src[i]
                _add(out, {
                    "observation.images.cam_global":
                        _to_uint8_hwc(it["observation.images.cam_global"], (gh, gw)),
                    "observation.images.cam_gripper":
                        _to_uint8_hwc(it["observation.images.cam_gripper"], (wh, ww)),
                    "observation.state": np.asarray(it["observation.state"], np.float32),
                    "action": np.asarray(it["action"], np.float32),
                }, _task_of(it, meta))
            out.save_episode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-repo", default="local/piper_sim")
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--sim-repeat", type=int, default=3)
    ap.add_argument("--real-repo", default="aiengineer56/piper_dataset_1100")
    ap.add_argument("--real-episodes", type=int, default=150)
    ap.add_argument("--tolerance-s", type=float, default=0.05)
    ap.add_argument("--out-repo", default="local/piper_mixed")
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sim = LeRobotDataset(args.sim_repo, root=args.sim_root)
    gh, gw = sim.meta.features["observation.images.cam_global"]["shape"][:2]
    wh, ww = sim.meta.features["observation.images.cam_gripper"]["shape"][:2]
    shapes = ((gh, gw), (wh, ww))
    fps = sim.meta.fps

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
    out = LeRobotDataset.create(repo_id=args.out_repo, fps=fps, root=args.out_root,
                                features=features, use_videos=True)
    print(f"[mix] sim {sim.num_episodes}eps x{args.sim_repeat} + real "
          f"{args.real_episodes}eps -> {args.out_root} "
          f"(img {gh}x{gw}/{wh}x{ww}, fps {fps})")

    _copy(sim, out, shapes, sim.meta, repeat=args.sim_repeat, limit_eps=None, desc="sim")

    real = LeRobotDataset(args.real_repo, episodes=list(range(args.real_episodes)),
                          tolerance_s=args.tolerance_s)
    _copy(real, out, shapes, real.meta, repeat=1, limit_eps=args.real_episodes, desc="real")

    print(f"[done] mixed dataset at {args.out_root} (repo_id={args.out_repo})")


if __name__ == "__main__":
    main()
