"""Estimate an object's x/y/z dimensions from simulated depth + segmentation.

Pipeline (genuine perception — not reading MuJoCo ground truth):
  1. render a depth map + segmentation mask from a camera,
  2. keep pixels belonging to the target geom,
  3. back-project them to a world-frame point cloud via pinhole intrinsics +
     camera extrinsics,
  4. fit an axis-aligned box for extents and a PCA on the top-view (xy) for the
     grasp yaw.

A single camera only sees the top + near faces, so object *height* is taken as
top_z - table_z (the object rests on the table — a safe, standard assumption).
The horizontal size uses the PCA-oriented extents (tighter than AABB for a
rotated box). Validate with scripts/estimate_object.py (prints error vs truth).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectEstimate:
    name: str
    center: np.ndarray        # (3,) world xyz of the object centroid
    dims: np.ndarray          # (3,) estimated size along world x, y, z
    oriented_wl: tuple        # (width, length) along the PCA minor/major axes
    yaw: float                # rad, orientation of the long horizontal axis
    top_z: float
    n_points: int


def back_project(depth, mask, f, cx, cy, cam_xpos, cam_xmat):
    """Masked depth pixels -> (N,3) world points. Camera looks along -z,
    +x right, +y up (MuJoCo convention)."""
    ys, xs = np.where(mask)
    d = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(d) & (d > 1e-4) & (d < 10.0)
    ys, xs, d = ys[valid], xs[valid], d[valid]
    x_c = (xs - cx) * d / f
    y_c = -(ys - cy) * d / f        # image row down -> camera +y up
    z_c = -d                        # points in front along -z
    pts_c = np.stack([x_c, y_c, z_c], axis=1)
    return cam_xpos[None, :] + pts_c @ cam_xmat.T


def estimate_object(env, geom_name: str, camera: str = "cam_global",
                    res=(360, 640), rest_on_table: bool = True) -> ObjectEstimate:
    import mujoco

    h, w = res
    depth = np.asarray(env.percep_render(camera, h, w, mode="depth"))
    seg = np.asarray(env.percep_render(camera, h, w, mode="seg"))
    gid = env.geom_id(geom_name)

    seg_id = seg[..., 0] if seg.ndim == 3 else seg
    mask = seg_id == gid
    n = int(mask.sum())
    if n < 10:
        raise RuntimeError(
            f"only {n} pixels for geom '{geom_name}' (id {gid}) in {camera}; "
            "object may be occluded/out of frame.")

    f, cx, cy, xpos, xmat = env.camera_params(camera, h, w)
    pts = back_project(depth, mask, f, cx, cy, xpos, xmat)

    mn, mx = pts.min(axis=0), pts.max(axis=0)
    center = (mn + mx) * 0.5
    dims = mx - mn
    top_z = float(mx[2])

    if rest_on_table:
        table_z = float(getattr(env.cfg.scene, "floor_z", 0.0))
        height = max(top_z - table_z, 1e-4)
        dims[2] = height
        center[2] = table_z + height * 0.5

    # PCA on the xy footprint -> yaw + oriented width/length
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    cov = xy.T @ xy / max(len(xy), 1)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    major = evecs[:, -1]
    yaw = float(np.arctan2(major[1], major[0]))
    proj_major = xy @ evecs[:, -1]
    proj_minor = xy @ evecs[:, 0]
    length = float(proj_major.max() - proj_major.min())
    width = float(proj_minor.max() - proj_minor.min())

    return ObjectEstimate(
        name=geom_name, center=center, dims=dims,
        oriented_wl=(width, length), yaw=yaw, top_z=top_z, n_points=n)
