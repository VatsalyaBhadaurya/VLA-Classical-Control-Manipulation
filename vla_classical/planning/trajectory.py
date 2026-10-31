"""Turn a GraspPlan into a joint-space trajectory for the PiPER env.

Solves IK at each Cartesian key-pose (pre-grasp -> grasp -> lift -> over-box ->
place) and linearly interpolates joint targets between them, scheduling the
gripper open/close at the right segments. Output is a list of 7-dim dataset-unit
actions `[j1..j6, gripper]` to feed through the JointSafetyFilter into env.step.
"""
from __future__ import annotations

import numpy as np


def build_pick_place_trajectory(ik, grasp, q_start,
                                n_move: int = 40, n_descend: int = 25,
                                n_dwell: int = 25):
    M = grasp.grasp_mat
    q_start = np.asarray(q_start, float)[:6]

    q_pre, ok1, e1 = ik.solve(grasp.pregrasp_pos, M, q_init=q_start)
    q_grasp, ok2, e2 = ik.solve(grasp.grasp_pos, M, q_init=q_pre)
    q_lift = q_pre                                   # lift returns to pre-grasp pose
    q_over, ok3, e3 = ik.solve(grasp.place_hover_pos, M, q_init=q_grasp)
    q_place, ok4, e4 = ik.solve(grasp.place_pos, M, q_init=q_over)

    O, C = grasp.gripper_open, grasp.gripper_close
    # (target joints, gripper cmd, #steps)
    segs = [
        (q_pre, O, n_move),       # move above object, gripper open
        (q_grasp, O, n_descend),  # descend to grasp
        (q_grasp, C, n_dwell),    # close on object
        (q_lift, C, n_descend),   # lift
        (q_over, C, n_move),      # traverse over the box
        (q_place, C, n_descend),  # descend into the box
        (q_place, O, n_dwell),    # release
    ]

    traj = []
    q_prev, g_prev = q_start, O
    for q_t, g_t, n in segs:
        for i in range(1, n + 1):
            a = i / n
            q = (1 - a) * q_prev + a * q_t
            g = (1 - a) * g_prev + a * g_t
            traj.append(np.concatenate([q, [g]]))
        q_prev, g_prev = q_t, g_t

    info = {
        "ik_ok": bool(ok1 and ok2 and ok3 and ok4),
        "ik_errs": [round(float(e), 4) for e in (e1, e2, e3, e4)],
        "n_steps": len(traj),
    }
    return traj, info
