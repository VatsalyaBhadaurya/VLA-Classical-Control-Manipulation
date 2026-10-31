"""Damped-least-squares inverse kinematics for the PiPER TCP site.

Solves 6-DoF pose IK (position + orientation) for the `tcp` site using the
MuJoCo site Jacobian. Side-effect free: saves/restores qpos so it doesn't
disturb the live sim. Used by the grasp trajectory planner to turn Cartesian
waypoints into the joint targets the env consumes.

    ik = DifferentialIK(env)
    q6, ok, err = ik.solve(target_pos, target_mat, q_init=current_arm)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot


class DifferentialIK:
    def __init__(self, env, site: str = "tcp"):
        self.env = env
        self.mj = env.mj
        self.model = env.model
        self.data = env.data
        self.sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, site)
        if self.sid < 0:
            raise RuntimeError(f"site '{site}' not found (is the TCP site in the scene?)")
        self.arm_dofs = np.asarray(env._arm_vadr)
        self.arm_qadr = np.asarray(env._arm_qadr)
        self.low = np.asarray(env.cfg.joint_low)
        self.high = np.asarray(env.cfg.joint_high)

    def solve(self, target_pos, target_mat, q_init=None,
              iters: int = 200, pos_tol: float = 2e-3, rot_tol: float = 2e-2,
              damping: float = 5e-2, step: float = 0.5):
        mj, model, data = self.mj, self.model, self.data
        saved = data.qpos.copy()
        savedv = data.qvel.copy()

        if q_init is not None:
            data.qpos[self.arm_qadr] = np.asarray(q_init, float)
        target_pos = np.asarray(target_pos, float)
        target_mat = np.asarray(target_mat, float).reshape(3, 3)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        ok = False
        err_norm = np.inf
        for _ in range(iters):
            mj.mj_forward(model, data)
            cur_pos = data.site_xpos[self.sid]
            cur_mat = data.site_xmat[self.sid].reshape(3, 3)

            e_pos = target_pos - cur_pos
            # orientation error as a world-frame rotation vector (axis*angle).
            # Via R_err = R_tgt @ R_cur^T -> rotvec; correct even for ~180° flips
            # (the cross-product form is degenerate there and stalls the solver).
            e_rot = _Rot.from_matrix(target_mat @ cur_mat.T).as_rotvec()
            err = np.concatenate([e_pos, e_rot])
            err_norm = float(np.linalg.norm(err))
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                ok = True
                break

            mj.mj_jacSite(model, data, jacp, jacr, self.sid)
            J = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])  # 6x6
            # dq = J^T (J J^T + λ²I)^-1 err
            lam2 = damping ** 2
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), err)
            q = data.qpos[self.arm_qadr] + step * dq
            data.qpos[self.arm_qadr] = np.clip(q, self.low, self.high)

        sol = data.qpos[self.arm_qadr].copy()
        data.qpos[:] = saved          # restore live sim
        data.qvel[:] = savedv
        mj.mj_forward(model, data)
        return sol, ok, err_norm
