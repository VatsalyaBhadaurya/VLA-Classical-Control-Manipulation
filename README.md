# vla-classical

Lightweight **Vision-Language-Action (VLA) + classical control** manipulation
in MuJoCo, sized for a **6 GB RTX 4050 laptop**.

```
instruction ─► SmolVLA-450M (LeRobot) ─► EE target / action chunk  (5–10 Hz)
                                              │
                                   SafetyFilter + ActionSmoother   (rate-limit, low-pass, workspace clamp, E-stop)
                                              │
                                   OSC / impedance controller      (every sim step, via robosuite)
                                              │
                                   MuJoCo actuators ─► physics (≈500 Hz) ─► cameras+proprio ─► back to VLA
```

The **VLA decides *what*** to do; the **classical controller decides *how*** to
execute it safely. The VLA is a swappable `Policy`; the first working system
(scripted policy + classical control + all eval baselines) runs **before**
SmolVLA is ever loaded.

## Status by phase

| Phase | What | State |
|---|---|---|
| 1 | MuJoCo env + cameras + physics sanity | **runnable** (`scripts/run_phase1_env.py`) |
| 2 | OSC/impedance + scripted pick-place | scaffolded (`vla_classical/control`, `policies/scripted.py`) |
| 3 | Demo collection → LeRobot format | scaffolded (`vla_classical/data/recorder.py`) |
| 4 | SmolVLA load / fine-tune | scaffolded (`policies/smolvla_policy.py`, `scripts/finetune_smolvla.py`) |
| 5 | Hybrid VLA + classical execution | wired via `Policy` interface |
| 6 | Domain randomization + eval/metrics | scaffolded (`vla_classical/eval`) |

---

## Setup (WSL2 Ubuntu, CUDA via your 4050)

> Your system Python is 3.8; LeRobot/SmolVLA need **≥3.10**. We install 3.11 in
> an isolated venv. See `setup/install_wsl2.md` for the one-time WSL2/CUDA steps.

```bash
# in WSL2, from the repo (lives on the Windows D: drive)
cd /mnt/d/vatty/work/vla-classical

# 1) Python 3.11 venv
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# 2) Core sim + utils
pip install -r requirements.txt

# 3) PyTorch with CUDA 12.1 (fits 6 GB; fp16 inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) LeRobot + SmolVLA (Phase 4 — can defer until the classical pipeline works)
pip install "lerobot[smolvla]"

# 5) Offscreen GPU rendering for MuJoCo in headless WSL2
echo 'export MUJOCO_GL=egl' >> ~/.bashrc && export MUJOCO_GL=egl

# 6) Verify the stack
python setup/verify_stack.py
```

> ⚠️ **Perf tip:** builds/data I/O on `/mnt/d` (Windows drive) are slow under
> WSL2. Keep the repo here if you want it on D:, but write large datasets to a
> WSL-native path (e.g. `~/vla_data`) and point `DataConfig.root` at it.

---

## Run

```bash
# Phase 1 — render + physics sanity check (writes a video, prints obs/FPS)
python scripts/run_phase1_env.py --task lift --steps 200 --out out/phase1.mp4

# Phase 2/3 — collect scripted demonstrations (LeRobot format)
python scripts/collect_demos.py --task lift --episodes 50 --policy scripted

# Phase 4 — fine-tune SmolVLA on collected demos (fp16, small batch)
python scripts/finetune_smolvla.py --dataset lift_demos --batch-size 2 --amp

# Phase 5/6 — evaluate baselines
python scripts/evaluate.py --task lift --policy scripted         # classical-only
python scripts/evaluate.py --task lift --policy smolvla          # VLA-only
python scripts/evaluate.py --task lift --policy smolvla --safety # VLA + classical
python scripts/evaluate.py --task lift --policy smolvla --safety --randomize
```

## PiPER dataset track (`aiengineer56/piper_dataset_1100`)

A second, dataset-matched track targeting the **AgileX PiPER** (6-DoF + gripper,
LeRobot **v3.0** dataset). The sim env reproduces the dataset's I/O exactly:
`action`/`observation.state` = `[joint1..joint6, gripper]` (absolute joint
targets, radians + 0–0.1 m gripper), cameras `cam_global` (1280×720) and
`cam_gripper` (640×480) @ 30 fps. Control is **joint-space position** (no OSC/IK).

```bash
# 1) Vendor the PiPER model from MuJoCo Menagerie (sparse checkout)
bash setup/fetch_piper.sh

# 2) Build the scene, render both cameras, verify joint-space control
#    (720p software render is slow — use --downsample 2 or 3 to iterate fast)
python scripts/piper_render_test.py --out out/piper --downsample 2

# 3) Pull dataset metadata (task text + stats); episodes stream via LeRobot
hf download aiengineer56/piper_dataset_1100 --repo-type dataset \
    --include "meta/*" --local-dir data/piper_dataset_1100
```

Open `out/piper_cam_global.png` / `out/piper_cam_gripper.png` to check the view;
camera extrinsics in `configs/piper.py` are initial guesses to be tuned against a
real dataset frame. The task's scene objects are a placeholder cube until the
task instruction is read from the dataset metadata.

## Hardware / feasibility notes (RTX 4050, 6 GB)

- **SmolVLA-450M** fp16 ≈ ~1 GB weights; inference fits comfortably. Fine-tuning
  is the tight part — use `--amp`, batch size 1–2, gradient checkpointing, and
  freeze the vision backbone first (`VLAConfig.freeze_vision=True`).
- Cameras capped at **224×224** (`CameraConfig`). Raising resolution or adding
  the wrist cam increases VRAM and latency.
- If training OOMs: lower batch to 1 → freeze vision → shrink image to 128 →
  truncate action chunk → fall back to CPU inference (`VLAConfig.device=cpu`,
  slower but unblocks) → 8-bit quantization (`bitsandbytes`).
- Separate rates (configurable): **VLA 5–10 Hz**, env/OSC step **20 Hz**,
  physics **~500 Hz**.

See `setup/install_wsl2.md` for GPU driver / CUDA passthrough details.
