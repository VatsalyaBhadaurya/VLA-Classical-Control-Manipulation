"""Camera extraction from robosuite observations.

robosuite returns camera images as HxWx3 uint8 under keys like
``<name>_image``, and they come **vertically flipped**. This helper normalizes
them into a clean {name: uint8 HxWx3} dict ready for the VLA or the recorder.
"""
from __future__ import annotations

import numpy as np


def extract_images(obs: dict, camera_names, flip_vertical: bool = True) -> dict:
    out = {}
    for name in camera_names:
        key = f"{name}_image"
        if key not in obs:
            raise KeyError(
                f"camera '{name}' not in obs (have: "
                f"{[k for k in obs if k.endswith('_image')]}). "
                "Check CameraConfig.names vs robosuite camera_names."
            )
        img = np.asarray(obs[key])
        if flip_vertical:
            img = img[::-1]
        out[name] = np.ascontiguousarray(img.astype(np.uint8))
    return out
