#!/usr/bin/env python3
"""DIAGNOSTIC ONLY: kinematic reachability of CUT knife-edge orientations for Franka 0.

Uses textbook Franka Panda DH parameters and joint limits with the CUT-01
hand-to-edge mount and Franka 0's base position, solved by bounded least
squares with warm starts. It is an approximation of, not a substitute for,
the repo's Newton/Isaac articulation: no collisions (hand, wrist cameras,
table, fixtures), no grasp, no dynamics. WP-10E replaces it with a sweep on
the actual model. Run: python scripts/probe_dh_reachability.py [check]
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

BASE = np.array([-.46, .12, -.01])
LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973])
HI = np.array([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973])
DH = [(0, .333, 0), (0, 0, -np.pi/2), (0, .316, np.pi/2), (.0825, 0, np.pi/2),
      (-.0825, .384, -np.pi/2), (0, 0, np.pi/2), (.088, 0, np.pi/2)]
def T(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa], [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])
F_HAND = np.eye(4); F_HAND[2, 3] = .107
F_HAND[:3, :3] = R.from_euler("z", -np.pi/4).as_matrix()   # panda_link8 -> panda_hand
def mount(xyzw, d=.18):
    M = np.eye(4); M[:3, :3] = R.from_quat(xyzw).as_matrix(); M[2, 3] = d; return M
CUT_MOUNT = mount([.70710678, 0, .70710678, 0])
def fk_hand(q):
    M = np.eye(4); M[:3, 3] = BASE
    for (a, d, al), th in zip(DH, q):
        M = M @ T(a, d, al, th)
    return M @ F_HAND
def fk_edge(q): return fk_hand(q) @ CUT_MOUNT
PARK = np.array([0, -.785, 0, -2.356, 0, 1.571, .785])
def ik(target, seeds):
    best = None
    for s in seeds:
        def res(q):
            E = fk_edge(q)
            return np.r_[E[:3, 3]-target[:3, 3], .1*R.from_matrix(target[:3, :3].T @ E[:3, :3]).as_rotvec()]
        r = least_squares(res, s, bounds=(LO, HI), xtol=1e-12, ftol=1e-12)
        E = fk_edge(r.x)
        pe = np.linalg.norm(E[:3, 3]-target[:3, 3]); re = R.from_matrix(target[:3, :3].T @ E[:3, :3]).magnitude()
        cand = (pe < 1e-3 and re < .01, -pe-re, r.x, pe, re)
        if best is None or cand[:2] > best[:2]:
            best = cand
        if cand[0] and s is seeds[0] and len(seeds) < 6:   # warm start succeeded: keep that branch
            break
    ok, _, q, pe, re = best
    margin = float(np.min(np.minimum(q-LO, HI-q)))
    hand_z = fk_hand(q)[2, 3]
    return ok, q, pe, re, margin, hand_z
BASE_ROT = R.from_quat([0, .70710678, 0, .70710678])         # task CUT edge orientation
rng = np.random.default_rng(0)
SEEDS = [PARK] + [rng.uniform(LO, HI) for _ in range(12)]
def target(point, extra):
    M = np.eye(4); M[:3, :3] = (extra*BASE_ROT).as_matrix(); M[:3, 3] = point; return M
if sys.argv[1:] == ["check"]:
    ok, q, pe, re, m, hz = ik(target([0, 0, .05], R.identity()), SEEDS)
    print("reference CUT start pose: ok", ok, "pos_err", pe, "rot_err", re, "q", np.round(q, 3).tolist(), "joint margin", round(m, 3), "hand z", round(hz, 3)); sys.exit()
rows = []
sweeps = {"oblique_about_edge_line_y": "y", "yaw_about_vertical_z": "z", "edge_roll_about_blade_normal_x": "x"}
angles = {"y": range(-60, 61, 15), "z": range(-180, 180, 30), "x": range(-45, 46, 15)}
for name, ax in sweeps.items():
    for deg in angles[ax]:
        extra = R.from_euler(ax, deg, degrees=True)
        path = []
        for z in np.linspace(.03, -.003, 12):                             # above bar, mid, through
            ok, q, pe, re, m, hz = ik(target([0, 0, z], extra), SEEDS if not path else [path[-1][1]] + SEEDS[:3])
            path.append((ok, q, m, hz))
        jump = max(np.abs(path[i+1][1]-path[i][1]).max() for i in range(2))
        rows.append({"sweep": name, "deg": deg, "reachable_all_3": all(p[0] for p in path),
                     "min_joint_margin_rad": round(min(p[2] for p in path), 3),
                     "min_hand_z_m": round(min(p[3] for p in path), 3), "max_joint_jump_rad": round(float(jump), 3)})
        print(json.dumps(rows[-1]), flush=True)
