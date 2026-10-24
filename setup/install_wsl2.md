# One-time WSL2 + CUDA setup for the 4050

Goal: run MuJoCo (GPU offscreen rendering) + PyTorch CUDA + robosuite + LeRobot
inside WSL2 Ubuntu, using the Windows NVIDIA driver (617.42) via WSL passthrough.

## 1. Install WSL2 Ubuntu (PowerShell, as admin — run on Windows, not here)

```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

Reboot if prompted, create your Ubuntu user, then open the Ubuntu shell.

## 2. CUDA passthrough

You do **not** install an NVIDIA driver *inside* WSL — the Windows driver is
shared. Just verify it's visible:

```bash
nvidia-smi          # should list "NVIDIA GeForce RTX 4050 Laptop GPU", 6 GB
```

If `nvidia-smi` is missing in WSL, update the Windows NVIDIA driver (you have
617.42, which supports WSL) and ensure WSL is version 2 (`wsl -l -v`).

Install the CUDA toolkit headers only if you compile extensions; the PyTorch
CUDA 12.1 wheels bundle their own runtime, so for this project you normally
**do not** need a system CUDA toolkit.

## 3. System libs for MuJoCo offscreen (EGL) rendering

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libglfw3 libosmesa6 ffmpeg \
                        python3.11 python3.11-venv python3.11-dev build-essential
```

## 4. Rendering backend

Headless WSL has no X display, so use EGL (GPU) for offscreen rendering:

```bash
export MUJOCO_GL=egl        # add to ~/.bashrc
# fallbacks if EGL misbehaves:
#   export MUJOCO_GL=osmesa  # CPU software rendering (slower, very robust)
```

## 5. Continue with README "Setup" steps 1–6.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GLFWError` / `cannot open display` | `export MUJOCO_GL=egl` (or `osmesa`). |
| Black/empty camera frames | Try `MUJOCO_GL=osmesa`; confirm `has_offscreen_renderer=True`. |
| `libEGL ... failed to create dri2 screen ... falling back to kms_swrast` | Non-fatal: EGL fell back to **CPU software** rendering (Mesa swrast) — frames are correct but slow. For true GPU EGL in WSL, install NVIDIA's EGL libs / libglvnd and set `__EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d`. Software rendering is fine for Phases 1–3; revisit only if render FPS bottlenecks training-data collection. |
| `torch.cuda.is_available()` False | Update Windows NVIDIA driver; `wsl --shutdown` then reopen; reinstall cu121 torch. |
| CUDA OOM during fine-tune | batch=1 → `freeze_vision=True` → image 128 → CPU inference. |
| Slow pip / file ops | Repo on `/mnt/d` is slow; move venv + datasets to WSL-native `~/`. |
