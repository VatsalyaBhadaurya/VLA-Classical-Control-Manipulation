#!/usr/bin/env bash
# Vendor the AgileX PiPER model from MuJoCo Menagerie (sparse — just that folder).
# Run from the repo root in WSL2.
set -euo pipefail

DEST="third_party/mujoco_menagerie"
if [ -d "$DEST/agilex_piper" ]; then
    echo "[fetch_piper] already present at $DEST/agilex_piper"
    exit 0
fi

mkdir -p third_party
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
( cd "$DEST" && git sparse-checkout set agilex_piper )

echo "[fetch_piper] done:"
ls -1 "$DEST/agilex_piper"
echo "[fetch_piper] meshes:"
ls -1 "$DEST/agilex_piper/assets" | head
