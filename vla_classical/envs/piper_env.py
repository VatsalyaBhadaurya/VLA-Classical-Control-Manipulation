"""AgileX PiPER MuJoCo environment — matches the piper_dataset_1100 schema.

Backend: raw MuJoCo (not robosuite — robosuite has no PiPER). The action and
observation spaces are byte-for-byte compatible with the LeRobot v3.0 dataset:

    action[7]            = [j1, j2, j3, j4, j5, j6, gripper]  (absolute targets)
    observation.state[7] = [j1, j2, j3, j4, j5, j6, gripper]
    images               = {"cam_global": HxWx3, "cam_gripper": HxWx3}

The scene is assembled by light text-surgery on the Menagerie ``piper.xml``
(inject a wrist camera into link6, add a table + object + global camera +
light into the worldbody) and written next to ``piper.xml`` so the model's own
``meshdir="assets"`` keeps resolving. This avoids the <include> meshdir/compiler
pitfalls entirely.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from configs.piper import PiperConfig
from .robosuite_env import Observation   # reuse the shared obs container (no robosuite import)

GRIPPER_ACTUATOR = "gripper"
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
WRIST_BODY = "link6"


class PiperMujocoEnv:
    def __init__(self, cfg: PiperConfig):
        import mujoco

        self.mj = mujoco
        self.cfg = cfg
        self._menagerie = Path(cfg.menagerie_dir)
        self._piper_xml = self._menagerie / "piper.xml"
        if not self._piper_xml.exists():
            raise FileNotFoundError(
                f"Menagerie PiPER model not found at {self._piper_xml}. "
                "Run setup/fetch_piper.sh first.")

        scene_path = self._write_scene()
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self._index()
        self._renderers = {}          # (h, w) -> Renderer
        self.control_dt = 1.0 / cfg.control_freq
        self.n_substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        self.reset()

    # ------------------------------------------------------------------ scene
    def _write_scene(self) -> Path:
        text = self._piper_xml.read_text()
        cam = self.cfg.camera
        sc = self.cfg.scene

        # 1) wrist camera inside link6
        wrist_cam = (
            f'<camera name="cam_gripper" mode="fixed" '
            f'pos="{_f3(cam.cam_gripper_pos)}" '
            f'xyaxes="{_fN(cam.cam_gripper_xyaxes)}" '
            f'fovy="{cam.cam_gripper_fovy}"/>'
        )
        new_text, n = re.subn(
            rf'(<body\s+name="{WRIST_BODY}"[^>]*>)', rf'\1\n      {wrist_cam}',
            text, count=1)
        if n == 0:
            print(f"[piper] WARNING: body '{WRIST_BODY}' not found; "
                  "cam_gripper not attached. Check piper.xml body names.")
        text = new_text

        # 2) worldbody additions: floor, object, global camera.
        #    (piper.xml already defines a light; don't add another or it washes
        #    out the wrist camera.)
        g_xyaxes = _lookat_xyaxes(cam.cam_global_eye, cam.cam_global_target)
        extras = [
            f'<geom name="floor" type="plane" size="0 0 0.05" '
            f'pos="0 0 {sc.floor_z}" rgba="{_fN(sc.floor_rgba)}" '
            f'friction="1 0.05 0.001"/>',
            f'<camera name="cam_global" mode="fixed" pos="{_f3(cam.cam_global_eye)}" '
            f'xyaxes="{_fN(g_xyaxes)}" fovy="{cam.cam_global_fovy}"/>',
        ]
        if sc.add_object:
            extras.append(
                f'<body name="object" pos="{_f3(sc.object_pos)}">'
                f'<freejoint name="object_free"/>'
                f'<geom name="object" type="box" size="{_f3(sc.object_size)}" '
                f'rgba="{_fN(sc.object_rgba)}" mass="{sc.object_mass}" '
                f'friction="1 0.05 0.001"/></body>')
        block = "\n      ".join(extras)
        text = text.replace("</worldbody>", f"      {block}\n    </worldbody>", 1)

        out = self._menagerie / "piper_scene_autogen.xml"
        out.write_text(text)
        return out

    # ------------------------------------------------------------------ index
    def _index(self):
        mj = self.mj
        m = self.model
        # qpos addresses for arm joints + gripper finger (joint7)
        self._arm_qadr = np.array(
            [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j)]
             for j in ARM_JOINTS])
        g7 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "joint7")
        self._grip_qadr = m.jnt_qposadr[g7]
        g8 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "joint8")
        self._grip8_qadr = m.jnt_qposadr[g8] if g8 >= 0 else None
        # qvel addresses (arm)
        self._arm_vadr = np.array(
            [m.jnt_dofadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j)]
             for j in ARM_JOINTS])
        # actuator ids, in dataset order [j1..j6, gripper]
        self._act_ids = np.array(
            [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, j) for j in ARM_JOINTS]
            + [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)])
        if np.any(self._act_ids < 0):
            raise RuntimeError("Missing an expected actuator (joint1..6/gripper).")
        self._wrist_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, WRIST_BODY)
        self._obj_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "object")

    # ------------------------------------------------------------------- units
    def _grip_ds_to_ctrl(self, g: float) -> float:
        frac = np.clip(g / self.cfg.gripper_dataset_max, 0.0, 1.0)
        return float(frac * self.cfg.gripper_actuator_max)

    def _grip_ctrl_to_ds(self, c: float) -> float:
        frac = np.clip(c / self.cfg.gripper_actuator_max, 0.0, 1.0)
        return float(frac * self.cfg.gripper_dataset_max)

    # --------------------------------------------------------------------- API
    def reset(self) -> Observation:
        self.mj.mj_resetDataKeyframe(self.model, self.data, 0)  # "home"
        self.mj.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action: np.ndarray) -> Observation:
        """action = [j1..j6, gripper] absolute targets (dataset units)."""
        a = np.asarray(action, dtype=np.float64).ravel()
        assert a.shape[0] == 7, f"expected 7-dim action, got {a.shape[0]}"
        arm = np.clip(a[:6], self.cfg.joint_low, self.cfg.joint_high)
        ctrl = np.empty(7)
        ctrl[:6] = arm
        ctrl[6] = self._grip_ds_to_ctrl(a[6])
        self.data.ctrl[self._act_ids] = ctrl
        for _ in range(self.n_substeps):
            self.mj.mj_step(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> Observation:
        qpos = self.data.qpos
        arm_q = qpos[self._arm_qadr].astype(np.float32)
        grip_ds = self._grip_ctrl_to_ds(float(qpos[self._grip_qadr]))
        state = np.concatenate([arm_q, [grip_ds]]).astype(np.float32)  # dataset [7]
        images = {name: self.render(name) for name in ("cam_global", "cam_gripper")}
        obj_pose = None
        if self._obj_bid >= 0:
            obj_pose = np.concatenate([self.data.xpos[self._obj_bid],
                                       self.data.xquat[self._obj_bid]]).astype(np.float32)
        return Observation(
            images=images,
            eef_pos=self.data.xpos[self._wrist_bid].astype(np.float32),
            eef_quat=self.data.xquat[self._wrist_bid].astype(np.float32),  # wxyz (mujoco)
            joint_pos=arm_q,
            joint_vel=self.data.qvel[self._arm_vadr].astype(np.float32),
            gripper_qpos=np.array([grip_ds], dtype=np.float32),
            object_pose=obj_pose,
            instruction=self.cfg.instruction,
            raw={"state": state},
        )

    def state_vector(self) -> np.ndarray:
        """Dataset-order [j1..j6, gripper] (7,)."""
        return self.get_observation().raw["state"]

    def set_state(self, state7: np.ndarray) -> Observation:
        """Hard-set the arm+gripper to a dataset state (kinematic replay).

        Bypasses dynamics/controller — used to perfectly reproduce recorded
        trajectories visually. Also sets the mirrored finger (joint8).
        """
        s = np.asarray(state7, dtype=np.float64).ravel()
        self.data.qpos[self._arm_qadr] = s[:6]
        g = self._grip_ds_to_ctrl(s[6])
        self.data.qpos[self._grip_qadr] = g
        if self._grip8_qadr is not None:
            self.data.qpos[self._grip8_qadr] = -g
        self.mj.mj_forward(self.model, self.data)
        return self.get_observation()

    # ---------------------------------------------------------------- render
    def render(self, camera: str) -> np.ndarray:
        cam = self.cfg.camera
        h, w = (cam.cam_global_hw if camera == "cam_global" else cam.cam_gripper_hw)
        d = max(1, cam.render_downsample)
        h, w = h // d, w // d
        r = self._renderers.get((h, w))
        if r is None:
            r = self.mj.Renderer(self.model, height=h, width=w)
            self._renderers[(h, w)] = r
        r.update_scene(self.data, camera=camera)
        return np.ascontiguousarray(r.render().astype(np.uint8))

    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass


# -- tiny formatting helpers for building MJCF attribute strings ------------
def _f3(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _fN(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _lookat_xyaxes(eye, target, world_up=(0, 0, 1)):
    """MJCF camera xyaxes (right, up) so the camera at `eye` looks at `target`.

    MuJoCo cameras view along their local -z; +x is right, +y is up.
    """
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    up = np.asarray(world_up, float)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, fwd)
    return (*right, *true_up)
