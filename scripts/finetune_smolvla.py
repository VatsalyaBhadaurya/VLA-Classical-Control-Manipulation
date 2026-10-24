"""Phase 4: fine-tune SmolVLA-450M on collected demos (6 GB-friendly).

    python scripts/finetune_smolvla.py --dataset local/vla_classical \
        --batch-size 2 --amp --freeze-vision --steps 20000

This is a thin wrapper documenting the 6 GB recipe. LeRobot ships its own
training entry point; prefer it once the dataset exists, e.g.:

    python -m lerobot.scripts.train \
        --policy.type=smolvla \
        --policy.pretrained_path=lerobot/smolvla_base \
        --dataset.repo_id=local/vla_classical \
        --batch_size=2 --steps=20000 \
        --policy.use_amp=true

VRAM recipe (apply in this order if you OOM):
  batch_size 1 -> freeze vision backbone -> image 128 -> gradient checkpointing
  -> truncate action chunk -> CPU (last resort, very slow).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse

from configs import make_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="local/vla_classical")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    cfg = make_config("lift")
    cfg.vla.batch_size = args.batch_size
    cfg.vla.amp = args.amp
    cfg.vla.freeze_vision = args.freeze_vision
    cfg.vla.image_size = args.image_size

    print("[plan] Fine-tune SmolVLA with 6 GB recipe:")
    print(f"       dataset        = {args.dataset}")
    print(f"       batch_size     = {cfg.vla.batch_size}")
    print(f"       amp            = {cfg.vla.amp}")
    print(f"       freeze_vision  = {cfg.vla.freeze_vision}")
    print(f"       grad_ckpt      = {cfg.vla.gradient_checkpointing}")
    print(f"       image_size     = {cfg.vla.image_size}")
    print(f"       steps          = {args.steps}")
    print()
    print("Recommended: run LeRobot's trainer directly (see module docstring).")
    print("This wrapper exists to pin the config; integrate the LeRobot")
    print("training loop here once the dataset from collect_demos.py exists.")
    raise SystemExit(
        "TODO(phase4): wire lerobot.scripts.train or a custom loop. "
        "Collect demos first (scripts/collect_demos.py)."
    )


if __name__ == "__main__":
    main()
