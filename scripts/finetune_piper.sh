#!/usr/bin/env bash
# Fine-tune SmolVLA-450M on piper_dataset_1100 (joint-space pick-into-box).
#
# IMPORTANT (6 GB RTX 4050): training a 450M VLA is tight-to-infeasible on 6 GB.
# Recommended split for this project:
#   * TRAIN on a cloud GPU (see "Cloud" below) — SmolVLA paper used ~20k steps,
#     ~4 h on one A100.
#   * INFER locally on the 4050 (SmolVLA-450M inference fits 6 GB) via
#     scripts/run_piper_vla.py.
#
# First run downloads the full dataset (~13 GB) into the HF cache.
set -euo pipefail

DATASET="aiengineer56/piper_dataset_1100"
OUT="outputs/train/piper_smolvla"

# ---- Local attempt (expect OOM risk on 6 GB; batch_size=1) -----------------
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="${DATASET}" \
  --batch_size=1 \
  --steps=20000 \
  --save_freq=5000 \
  --num_workers=2 \
  --output_dir="${OUT}" \
  --job_name=piper_smolvla \
  --policy.device=cuda \
  --wandb.enable=false

# If it OOMs, in order: keep batch_size=1 -> ensure the vision encoder is frozen
# (SmolVLA default) -> reduce image size -> otherwise train on cloud.
# Inspect every tunable with:  lerobot-train --help
#
# ---- Cloud options ---------------------------------------------------------
# Colab (free/pro GPU), notebook linked from HF docs:
#   https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb
# Or a rented GPU / HF Jobs (l4x1 or a10g is plenty for batch_size=8-16):
#   hf jobs run --flavor l4x1 --timeout 6h <lerobot-image> \
#     "lerobot-train --policy.path=lerobot/smolvla_base \
#        --dataset.repo_id=${DATASET} --batch_size=16 --steps=20000 \
#        --output_dir=/data/piper_smolvla --policy.push_to_hub=true \
#        --policy.repo_id=<your-hf-user>/piper_smolvla"
# Then download the checkpoint for local inference:
#   hf download <your-hf-user>/piper_smolvla --local-dir ${OUT}/checkpoints/last/pretrained_model
