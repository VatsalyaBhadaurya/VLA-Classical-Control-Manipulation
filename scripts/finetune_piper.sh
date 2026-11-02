#!/usr/bin/env bash
# Fine-tune SmolVLA-450M on piper_dataset_1100 (joint-space pick-into-box).
#
# 6 GB RTX 4050: this is the aggressive low-VRAM recipe. SmolVLA already only
# trains the action expert (freeze_vision_encoder + train_expert_only default
# true); we also batch=1 and shrink input images 512->256. It may STILL OOM on
# 6 GB — if so, use Colab (see bottom). Inference always fits 6 GB.
#
# First run downloads the full dataset (~13 GB) into the HF cache.
set -euo pipefail

DATASET="aiengineer56/piper_dataset_1100"
OUT="outputs/train/piper_smolvla"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="${DATASET}" \
  --batch_size=1 \
  --steps=20000 \
  --save_freq=5000 \
  --num_workers=2 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.resize_imgs_with_padding='[256, 256]' \
  --output_dir="${OUT}" \
  --job_name=piper_smolvla \
  --wandb.enable=false

# Checkpoint lands in: ${OUT}/checkpoints/last/pretrained_model
# Then run inference locally:
#   python scripts/run_piper_vla.py --ckpt ${OUT}/checkpoints/last/pretrained_model
#
# ------------------------------------------------------------------ Colab ----
# Free T4 (16 GB) fits this easily. In a Colab GPU notebook:
#   !pip install -q "lerobot[smolvla]"
#   from huggingface_hub import notebook_login; notebook_login()
#   !lerobot-train --policy.path=lerobot/smolvla_base \
#       --dataset.repo_id=aiengineer56/piper_dataset_1100 \
#       --batch_size=8 --steps=20000 --num_workers=2 \
#       --output_dir=outputs/train/piper_smolvla --job_name=piper_smolvla \
#       --policy.device=cuda --policy.push_to_hub=true \
#       --policy.repo_id=<your-hf-user>/piper_smolvla --wandb.enable=false
# Then locally download just the checkpoint for inference:
#   hf download <your-hf-user>/piper_smolvla --local-dir ${OUT}/checkpoints/last/pretrained_model
