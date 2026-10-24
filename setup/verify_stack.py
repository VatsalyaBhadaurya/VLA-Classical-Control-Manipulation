"""Smoke-test the installed stack. Run after README setup steps 1-5.

    python setup/verify_stack.py

Checks Python version, numpy, MuJoCo (+ offscreen render), robosuite, PyTorch
CUDA, and (optionally) LeRobot/SmolVLA. Prints a PASS/WARN/FAIL table and exits
non-zero if anything required for Phase 1 is broken.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

REQUIRED_FOR_PHASE1 = {"numpy", "mujoco", "robosuite"}


def _check(name, fn):
    try:
        msg = fn()
        print(f"[ OK ]  {name:<22} {msg}")
        return True
    except Exception as e:  # noqa: BLE001
        level = "FAIL" if name in REQUIRED_FOR_PHASE1 else "WARN"
        print(f"[{level}]  {name:<22} {type(e).__name__}: {e}")
        if os.environ.get("VERIFY_VERBOSE"):
            traceback.print_exc()
        return level != "FAIL"


def check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} < 3.10 (LeRobot needs >=3.10)")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_numpy():
    np = importlib.import_module("numpy")
    if np.__version__.startswith("2."):
        raise RuntimeError(f"numpy {np.__version__}; pin <2.0 for robosuite 1.4")
    return f"numpy {np.__version__}"


def check_mujoco():
    import numpy as np
    import mujoco

    xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/>" \
          "<camera name='c' pos='0 -1 1' xyaxes='1 0 0 0 1 1'/></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    # offscreen render
    r = mujoco.Renderer(model, height=64, width=64)
    r.update_scene(data, camera="c")
    img = r.render()
    assert img.shape == (64, 64, 3) and np.asarray(img).any(), "blank render"
    return f"mujoco {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL','<unset>')}"


def check_robosuite():
    import robosuite
    from robosuite.controllers import load_controller_config

    load_controller_config(default_controller="OSC_POSE")
    return f"robosuite {robosuite.__version__} (OSC_POSE ok)"


def check_torch():
    import torch

    cuda = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if cuda else "CPU only"
    vram = ""
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram = f", {total:.1f} GB VRAM"
    return f"torch {torch.__version__}, cuda={cuda} ({dev}{vram})"


def check_lerobot():
    import lerobot  # noqa: F401

    return f"lerobot {getattr(lerobot, '__version__', 'installed')}"


def main():
    print("=" * 60)
    print("vla-classical stack verification")
    print("=" * 60)
    ok = True
    ok &= _check("python", check_python)
    ok &= _check("numpy", check_numpy)
    ok &= _check("mujoco", check_mujoco)
    ok &= _check("robosuite", check_robosuite)
    ok &= _check("torch", check_torch)        # WARN if CPU-only
    _check("lerobot", check_lerobot)          # optional until Phase 4
    print("=" * 60)
    if ok:
        print("Phase 1 prerequisites satisfied.")
        sys.exit(0)
    print("Missing a Phase 1 requirement — see FAIL rows above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
