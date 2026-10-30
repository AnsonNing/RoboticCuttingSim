#!/usr/bin/env python3
"""WP-11 Phase C2: geometric feasibility preflight for ruler_knife_v2 orientations.

Geometry only: no dynamics, no material, no Isaac Sim. For each candidate
(knife arrangement, edge-line direction, blade rake, workpiece placement) the
whole HOME -> REORIENT -> PRE-CUT -> SKIM/CUT -> WITHDRAW -> RETURN_HOME path is
sampled densely (<= 2 mm edge travel and <= 1 deg per sample). At each sample:

* the runtime's Newton analytic-Jacobian IK (``DualFrankaController``) solves
  Franka 0 for the knife EDGE pose (mount = profile ``hand_from_edge``),
  multi-start at the cut start, continued sample-to-sample in both directions;
* a second Newton model built from the same official Franka USD (convex-hull
  link colliders, pad cubes, hand capsule) plus the finger VISUAL meshes as
  hulls, the v2 ToolSpec handle and blade hulls, both v5 wrist D405
  housing/bracket boxes, the cutting board / riser / table boxes, the
  workpiece base box and bump hull, the global D455 box and parked Franka 1 is
  converted by SolverMuJoCo; signed distances come from ``mujoco.mj_geomDistance``.

Checks per candidate: IK reachability, joint-limit margin, joint continuity,
wrist-camera / gripper-pad+finger / knife-handle clearance, blade-vs-fixture
(blade may touch only the workpiece during an active cut, kerf retract or
blade-face push), Franka 0 self-collision, Franka 0 vs workcell and vs Franka 1.
Thresholds are engineering choices recorded in the manifest, not calibrated.

Raised +100/+150 mm placements are COMPUTE-ONLY variants of
``blademaster_workcell_v1`` (in-process only; no scenario file is written).

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/preflight_wp11_orientations.py \
      --out artifacts/wp11/phaseC2/c2_preflight_001 --workers 16 --render-dir docs/media/wp11 \
      --evidence docs/evidence/wp11/phaseC2_preflight_001.json
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

BASE_SCENARIO = 'blademaster_workcell_v1'
PLACEMENTS = {'p050': .05, 'p100': .10, 'p150': .15}   # world z translation of the workpiece/supports
RAKES_DEG = (0., 5., 10., 12.5, 15., 25., 35.)
ARRANGEMENTS = ('ruler_knife_v2_y', 'ruler_knife_v2_x')
EDGE_LINES = ('+y', '-y')   # world direction from the grip toward the blade tip
ARM_BASES = (((-.46, .12, -.01), 0.), ((.46, .12, -.01), math.pi))  # robot.py / workcell.py placement
FINGER_Q_M = .0112           # per-finger opening holding the 22 mm handle (as the WP-10C preflight)

THRESHOLDS = dict(
    ik_position_m=.001, ik_rotation_rad=.01,              # controller convergence (free space)
    ik_active_position_m=.0002, ik_active_rotation_rad=.002,  # active cut/skim/push samples
    joint_margin_rad=.05, joint_step_rad=.06,
    env_clearance_m=.005,        # robot links / hand / fingers / handle / camera vs workcell, workpiece, Franka 1
    blade_fixture_clearance_m=.001,  # blade vs table/board/riser (the SLICE-PUSH push runs 3 mm above the board)
    blade_free_workpiece_clearance_m=.002,  # blade vs workpiece outside active segments
    self_clearance_m=.002,       # non-adjacent Franka 0 link pairs (and hand-group vs links)
    distmax_m=.10,
    max_sample_translation_m=.002, max_sample_rotation_deg=1.,
)
SEED = 42


# ----------------------------------------------------------------------------- geometry helpers

def rot(axis, deg):
    return Rotation.from_rotvec(np.deg2rad(deg)*np.asarray(axis, float))


def edge_rotation_trim(edge_line, rake_deg):
    """Edge frame for a skim along world +x: x_e advance, y_e edge line, z_e blade normal.

    rake rho rotates about world y so the blade body (-x_e) rises behind the
    edge by rho; the edge line stays horizontal (tau = 0, psi = 0).
    """
    s = 1. if edge_line == '+y' else -1.
    R0 = np.column_stack([[1., 0., 0.], [0., s, 0.], np.cross([1., 0., 0.], [0., s, 0.])])
    return rot([0, 1, 0], rake_deg)*Rotation.from_matrix(R0)


def edge_rotation_cut(edge_line, tilt_deg=0.):
    """Vertical through-cut frame: advance -z, edge line along +/-y; tilt about world +y (oblique)."""
    s = 1. if edge_line == '+y' else -1.
    x, y = np.array([0., 0., -1.]), np.array([0., s, 0.])
    R0 = Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)]))
    return rot([0, 1, 0], tilt_deg)*R0


def edge_rotation_home(edge_line):
    """Hand-down HOME: edge line (= hand +z) points to world -z, blade advance along world +x."""
    angle = -90. if edge_line == '+y' else 90.
    return rot([1, 0, 0], angle)*edge_rotation_trim(edge_line, 0.)


def orientation_metrics(R, cut_dir, n_t=(0., 0., 1.)):
    """tau, psi, rho (deg) per ADR 0013 for an edge-frame rotation matrix."""
    n_t, c = np.asarray(n_t, float), np.asarray(cut_dir, float)
    x, e, nb = R[:, 0], R[:, 1], R[:, 2]
    tau = math.degrees(math.asin(min(1., abs(float(e@n_t)))))
    ref = np.cross(n_t, c)
    ref /= np.linalg.norm(ref)
    psi = math.degrees(math.acos(min(1., abs(float(e@ref)))))
    behind = -x   # blade body direction from the edge
    rho = math.degrees(math.asin(np.clip(float(behind@n_t), -1., 1.)))
    return tau, psi, rho


def placement_task(task_id, placement):
    """The task in blademaster_workcell_v1, re-placed at a compute-only height (no file written)."""
    from robotic_cutting_sim import task_scenarios as ts
    from robotic_cutting_sim.tasks import make_task
    base = ts.scenarios()
    name = BASE_SCENARIO if placement == 'p050' else f'wp11c2_compute_only_{placement}'
    if name not in base:
        config = deepcopy(base[BASE_SCENARIO])
        config['world_translation_m'] = [0., 0., PLACEMENTS[placement]]
        config['reason'] = 'WP-11 C2 compute-only raised placement (in-process; never written to configs)'
        original = ts.scenarios
        ts.scenarios = lambda: {**original(), name: config}
    return ts.apply_scenario(make_task(task_id), name)


def dome_hull_local(task):
    pts = []
    for bump in task.geometry.bumps:
        top = task.geometry.base_height_m
        for r in np.linspace(0, 1, 9):
            for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
                x, y = bump.x_m+r*bump.radius_x_m*np.cos(a), bump.y_m+r*bump.radius_y_m*np.sin(a)
                pts.append([x, y, top+float(bump.height(x, y))])
                pts.append([x, y, top-.001])
    return np.asarray(pts)


def hull_mesh(points):
    import newton
    points = np.asarray(points, float)
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    remap = {v: i for i, v in enumerate(hull.vertices)}
    faces = np.array([[remap[i] for i in simplex] for simplex in hull.simplices], np.int32)
    # orient outward
    center = vertices.mean(0)
    for k, f in enumerate(faces):
        n = np.cross(vertices[f[1]]-vertices[f[0]], vertices[f[2]]-vertices[f[0]])
        if n@(vertices[f[0]]-center) < 0:
            faces[k] = f[::-1]
    return newton.Mesh(vertices.astype(np.float32), faces.ravel()), vertices


def finger_visual_points(stage, finger_path):
    """Composed visual finger mesh points in the finger body frame (instance proxies included)."""
    from pxr import Usd, UsdGeom
    body = stage.GetPrimAtPath(finger_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith('finger_'):
            p = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            T = np.asarray(cache.ComputeRelativeTransform(prim, body)[0], float).T
            points.append(p@T[:3, :3].T+T[:3, 3])
    if not points:
        raise RuntimeError(f'No visual finger meshes under {finger_path}')
    return np.concatenate(points)


# ----------------------------------------------------------------------------- the collision model

class GeometryModel:
    """Newton model (official Franka USD colliders + tool/cameras/workcell) -> MuJoCo distances."""

    def __init__(self, profile_id, task):
        import warp as wp
        import newton
        from newton.solvers import SolverMuJoCo
        from pxr import Usd
        import mujoco
        from robotic_cutting_sim.workcell import ASSETS, PARK_JOINTS, SECONDARY_PARK_JOINTS
        from robotic_cutting_sim.tool_profile import PROFILES, make_profile_tool_spec, require_profile_consistency
        from robotic_cutting_sim.camera_rig import make_rig
        from robotic_cutting_sim.scene_frames import pose_matrix
        self.mujoco, self.newton, self.wp = mujoco, newton, wp
        wp.init()
        self.profile = PROFILES[profile_id]
        self.spec = make_profile_tool_spec(self.profile)
        require_profile_consistency(self.spec, self.profile)
        self.task = task
        asset = ROOT/'artifacts/workcell_asset_cache'/ASSETS['franka']
        self.asset = asset
        with wp.ScopedDevice('cpu'):
            single = newton.ModelBuilder()
            single.rigid_gap = .001
            SolverMuJoCo.register_custom_attributes(single)
            stage = Usd.Stage.Open(str(asset))
            stage.SetEditTarget(stage.GetSessionLayer())
            root = stage.GetDefaultPrim()
            root.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('mujoco')
            root.GetVariantSets().GetVariantSet('Colliders').SetVariantSelection('convex_hulls')
            root.GetAttribute('newton:selfCollisionEnabled').Set(True)
            single.add_usd(stage, floating=False, enable_self_collisions=True, collapse_fixed_joints=False,
                           load_visual_shapes=False, skip_mesh_approximation=True)
            labels = list(single.body_label)
            self.bodies_per_arm = single.body_count
            self.local = {name: next(i for i, l in enumerate(labels) if l.endswith('/'+name))
                          for name in ['panda_link%d' % k for k in range(8)]+['panda_hand', 'panda_leftfinger', 'panda_rightfinger']}
            hand_path = next(str(p.GetPath()) for p in stage.Traverse() if p.GetName() == 'panda_hand')
            finger_pts = {side: finger_visual_points(stage, hand_path+f'/panda_{side}finger') for side in ('left', 'right')}
            single.joint_q[:] = list(PARK_JOINTS)+[.04, .04]
            builder = newton.ModelBuilder()
            SolverMuJoCo.register_custom_attributes(builder)
            for index, (position, yaw) in enumerate(ARM_BASES):
                builder.add_builder(single, xform=wp.transform(wp.vec3(*position),
                                    wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw))), label_prefix=f'arm_{index}')
            builder.joint_q[9:16] = SECONDARY_PARK_JOINTS
            self.franka_shapes = builder.shape_count
            cfg = newton.ModelBuilder.ShapeConfig(density=1000., mu=.35, gap=.001)
            self.extra = []   # (label, group, role)

            def add(body, mesh=None, box=None, xform=None, label='', group='', role=''):
                label = 'c2_'+label
                if mesh is not None:
                    builder.add_shape_convex_hull(body, xform=xform or wp.transform_identity(), mesh=mesh, cfg=cfg, label=label)
                else:
                    builder.add_shape_box(body, xform=xform or wp.transform_identity(), hx=box[0], hy=box[1], hz=box[2],
                                          cfg=cfg, label=label)
                self.extra.append((label, group, role))

            def tf(T):
                q = Rotation.from_matrix(T[:3, :3]).as_quat()
                return wp.transform(wp.vec3(*T[:3, 3]), wp.quat(*q))

            hand0 = self.local['panda_hand']
            T_hand_tool = self.spec.T_hand_tool
            for shape in self.spec.shapes:
                mesh, _ = hull_mesh(shape.vertices_tool_m)
                add(hand0, mesh=mesh, xform=tf(T_hand_tool), label='tool_'+shape.role, group='arm0_hand', role=shape.role)
            for arm in (0, 1):
                off = arm*self.bodies_per_arm
                for side in ('left', 'right'):
                    mesh, _ = hull_mesh(finger_pts[side])
                    add(off+self.local[f'panda_{side}finger'], mesh=mesh, label=f'arm{arm}_{side}_finger_visual_hull',
                        group=f'arm{arm}_hand', role='finger')
                rig = make_rig(task.task_id)
                for box in rig.specs[f'wrist_d405_{arm}']['visual_obbs']:
                    add(off+hand0, box=np.asarray(box['dimensions_m'])/2, xform=tf(np.asarray(box['T_hand_obb'])),
                        label=f'arm{arm}_d405_{box["name"]}', group=f'arm{arm}_hand', role='camera')
            # global D455: physical dims on its world transform (visual-only housing)
            g = rig.specs['global_d455']
            dims = g['physical_dimensions_m']
            add(-1, box=(dims['width']/2, dims['height']/2, dims['depth']/2), xform=tf(np.asarray(g['T_world_camera'])),
                label='static_global_d455', group='static', role='global_camera')
            # WP-12 (ADR 0017): a task whose workcell declares its table box (flat-board workcell) uses it;
            # every other task keeps the historical 1.8 x 1.2 m table with its top at z = -0.01.
            table_box = getattr(task, 'table_box_world', None)
            if table_box is None:
                add(-1, box=(.9, .6, .025), xform=wp.transform(wp.vec3(0, 0, -.035)), label='static_table', group='static', role='table')
            else:
                add(-1, box=tuple(map(float, table_box[1])), xform=wp.transform(wp.vec3(*map(float, table_box[0])), wp.quat_identity()),
                    label='static_table', group='static', role='table')
            T = task.T_world_workpiece
            for fx in task.fixture_boxes_local:
                Tb = T.copy()
                Tb[:3, 3] = task.points_to_world(np.asarray(fx.center_m))
                add(-1, box=fx.half_extents_m, xform=tf(Tb), label='static_'+fx.name, group='static',
                    role=getattr(fx, 'role', 'support'))
            geo = task.geometry
            Tw = T.copy()
            Tw[:3, 3] = task.points_to_world([0., 0., geo.base_height_m/2])
            add(-1, box=(geo.length_m/2, geo.width_m/2, geo.base_height_m/2), xform=tf(Tw), label='static_workpiece_base',
                group='static', role='workpiece')
            if geo.bumps:
                mesh, _ = hull_mesh(task.points_to_world(dome_hull_local(task)))
                add(-1, mesh=mesh, label='static_workpiece_bump', group='static', role='workpiece')
            self.model = builder.finalize(device='cpu')
            self.state = self.model.state()
            self.solver = SolverMuJoCo(self.model, use_mujoco_cpu=True, separate_worlds=False, iterations=1,
                                       nconmax=64, njmax=256)
        self.m, self.d = self.solver.mj_model, self.solver.mj_data
        self.q = np.asarray(builder.joint_q, np.float32)
        self.q[7:9] = FINGER_Q_M
        self.lower, self.upper = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        self._index_geoms()
        self._make_pairs()

    # -- geom bookkeeping
    def _index_geoms(self):
        mj = self.mujoco
        names = [mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_GEOM, g) for g in range(self.m.ngeom)]
        labels = list(self.model.shape_label)
        body_of_shape = self.model.shape_body.numpy()
        info = {}
        for s, label in enumerate(labels):
            gid = next((g for g, n in enumerate(names) if n == f'{label}_{s}'), None)
            if gid is None:
                continue
            short = label.split('/')[-1]
            arm = 0 if label.startswith('arm_0') else 1 if label.startswith('arm_1') else None
            body = int(body_of_shape[s])
            local = body % self.bodies_per_arm if body >= 0 else None
            group, role = None, None
            if label.startswith('c2_'):
                _, group, role = next(e for e in self.extra if e[0] == label)
                arm = 0 if group.startswith('arm0') else 1 if group.startswith('arm1') else None
                short = label[3:]
            else:
                inv = {v: k for k, v in self.local.items()}
                link = inv[local]
                group = f'arm{arm}_hand' if link in ('panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger') else f'arm{arm}_links'
                role = 'pad' if 'finger_pad' in short else 'hand' if link == 'panda_hand' else 'link'
                short = f'arm{arm}_{short}'
            info[gid] = dict(name=short, group=group, role=role, arm=arm, link=local)
        missing = [l for l in labels if l.startswith('c2_') and not any(v['name'] == l[3:] for v in info.values())]
        if missing:
            raise RuntimeError(f'Shapes missing in the MuJoCo model: {missing}')
        self.geoms = info
        self.by_name = {v['name']: g for g, v in info.items()}

    def _make_pairs(self):
        """(g1, g2, subject, target) pairs. subject: Franka 0 part role; target: what it may hit.

        Excluded by construction: link0 (bolted to the table), link1 vs table
        (its hull sits on the base flange at the table top), joint-adjacent link
        pairs, and pairs inside the rigid hand group (link7, hand, fingers at the
        fixed grasp opening, tool, camera), whose distances are constant and are
        reported once in ``constant_hand_group``.
        """
        pairs = []
        G = self.geoms
        a0 = [g for g, v in G.items() if v['arm'] == 0]
        a1 = [g for g, v in G.items() if v['arm'] == 1]
        static = [g for g, v in G.items() if v['group'] == 'static']
        link = lambda g: G[g]['link']
        for g in a0:
            v = G[g]
            if v['group'] == 'arm0_links' and link(g) == 0:
                continue
            for s in static:
                if v['group'] == 'arm0_links' and link(g) == 1 and G[s]['role'] == 'table':
                    continue
                pairs.append((g, s, v['role'], G[s]['role']))
            for h in a1:
                pairs.append((g, h, v['role'], 'franka1'))
        hand_group = [g for g in a0 if G[g]['group'] == 'arm0_hand']
        links = [g for g in a0 if G[g]['group'] == 'arm0_links']
        for g, h in itertools.combinations(links, 2):
            if abs(link(g)-link(h)) >= 2:
                pairs.append((g, h, 'link', 'self'))
        for g in links:
            for h in hand_group:
                if link(g) == 6 and G[h]['link'] == self.local['panda_link7']:
                    continue
                pairs.append((h, g, G[h]['role'], 'self'))
        self.pairs = pairs
        self.constant_pairs = [(g, h) for g, h in itertools.combinations(hand_group, 2)
                               if {G[g]['role'], G[h]['role']} & {'handle', 'blade', 'camera'}]

    # -- evaluation
    def set_q(self, q7):
        q = self.q.copy()
        q[:7] = q7
        self.state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.solver._update_mjc_data(self.d, self.model, self.state)
        self.mujoco.mj_kinematics(self.m, self.d)

    def hand_pose(self):
        body = self.state.body_q.numpy()[self.local['panda_hand']]
        return np.asarray(body, float)

    def distance(self, g, h, distmax=THRESHOLDS['distmax_m']):
        # bounding-sphere lower bound first
        c = np.linalg.norm(self.d.geom_xpos[g]-self.d.geom_xpos[h])-self.m.geom_rbound[g]-self.m.geom_rbound[h]
        if c > distmax:
            return float(c)
        return float(self.mujoco.mj_geomDistance(self.m, self.d, g, h, distmax, None))

    def evaluate(self):
        """{(subject, target): (min distance, g, h)} over the classified pairs at the current q."""
        best = {}
        for g, h, subject, target in self.pairs:
            dist = self.distance(g, h)
            key = (subject, target)
            if key not in best or dist < best[key][0]:
                best[key] = (dist, g, h)
        return best

    def part_vertices_world(self, roles):
        """World vertices of hand-group parts (for plane clearances)."""
        out = {}
        for g, v in self.geoms.items():
            if v['arm'] != 0 or v['role'] not in roles:
                continue
            out[v['name']] = self.geom_vertices(g)
        return out

    def geom_vertices(self, g):
        m, d = self.m, self.d
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        t = m.geom_type[g]
        if t == self.mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        elif t == self.mujoco.mjtGeom.mjGEOM_BOX:
            v = np.asarray(list(itertools.product((-1, 1), repeat=3)), float)*m.geom_size[g]
        else:
            r, hl = m.geom_size[g][0], m.geom_size[g][1]
            v = np.asarray([[sx*r, sy*r, sz*(hl+r)] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return v@R.T+p

    def geom_mesh(self, g):
        """(vertices world, faces) for rendering."""
        m = self.m
        t = m.geom_type[g]
        if t == self.mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            faces = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]]
            return self.geom_vertices(g), np.asarray(faces)
        v = self.geom_vertices(g)
        return v, ConvexHull(v).simplices


# ----------------------------------------------------------------------------- kinematics (runtime IK)

class Kinematics:
    """The runtime DualFrankaController IK with the profile's hand_from_edge as the tool frame."""

    def __init__(self, profile_id):
        from robotic_cutting_sim.robot import RobotConfig, DualFrankaController
        from robotic_cutting_sim.tool_profile import PROFILES
        mount = PROFILES[profile_id].hand_from_edge()
        self.mount = np.asarray(mount, float)
        self.c = DualFrankaController(RobotConfig(tool_shape='box', tool_offset_m=tuple(mount[:3]),
                                                  tool_rotation_xyzw=tuple(mount[3:]), tool_half_extents_m=(.001, .001, .001)))
        self.base_q = self.c.initial_q.copy()
        self.base_q[7:9] = FINGER_Q_M
        self.lower, self.upper = self.c.lower[:7].astype(float), self.c.upper[:7].astype(float)

    def solve(self, pose, seed, strict):
        tol_p = THRESHOLDS['ik_active_position_m'] if strict else THRESHOLDS['ik_position_m']*.5
        tol_r = THRESHOLDS['ik_active_rotation_rad'] if strict else THRESHOLDS['ik_rotation_rad']*.5
        q = self.base_q.copy()
        q[:7] = seed
        info = None
        for attempt in range(5):
            self.c.state.joint_q.assign(q)
            q7, info = self.c.inverse_kinematics(pose, iterations=48 if attempt == 0 else 96)
            q[:7] = q7
            if info['position_error_m'] <= tol_p and info['rotation_error_rad'] <= tol_r:
                break
        return q[:7].astype(float).copy(), info

    def edge_pose(self, q7):
        return np.asarray(self.c.forward_kinematics(np.asarray(q7, np.float32)), float)


# ----------------------------------------------------------------------------- paths

HOME_LOCAL = (0., 0., .188)    # edge centre; tip (0.06 below) is 100 mm above a 28 mm bump top
FREE_LOCAL_Z = .088            # 60 mm above the tallest workpiece point (28 mm)


def pose7(position, rotation):
    return np.r_[np.asarray(position, float), rotation.as_quat()]


def world(task, local):
    return task.points_to_world(np.asarray(local, float))


def segment(name, a, b, *, active=False, allow_blade_workpiece=False, cut_dir=None):
    return dict(name=name, kind='cartesian', a=np.asarray(a, float), b=np.asarray(b, float), active=active,
                allow_blade_workpiece=allow_blade_workpiece, cut_dir=cut_dir)


def common_ends(task, edge_line, R_work, first_local_xy):
    """HOME -> REORIENT (above the workpiece centre) -> move above the first point."""
    home = pose7(world(task, HOME_LOCAL), edge_rotation_home(edge_line))
    reo = pose7(world(task, (0., 0., FREE_LOCAL_Z)), R_work)
    above = pose7(world(task, (first_local_xy[0], first_local_xy[1], FREE_LOCAL_Z)), R_work)
    return home, reo, above


def build_path(task, kind, edge_line, rake_deg=0., tilt_deg=0.):
    """Ordered segments for one task kind; local workpiece coordinates are explicit."""
    top = task.geometry.base_height_m
    if kind == 'TRIM':
        R = edge_rotation_trim(edge_line, rake_deg)
        z = top+task.edge_clearance_m            # target plane + 0.2 mm
        x0 = -task.geometry.length_m/2-task.tool_profile.blade_depth_m-.005   # blade body fully off-piece
        x1 = .045                                 # frozen TRIM-01-R skim end
        home, reo, above = common_ends(task, edge_line, R, (x0, 0.))
        pre = pose7(world(task, (x0, 0., z+.03)), R)
        start = pose7(world(task, (x0, 0., z)), R)
        end = pose7(world(task, (x1, 0., z)), R)
        up = pose7(world(task, (x1, 0., z+.04)), R)
        up2 = pose7(world(task, (x1, 0., FREE_LOCAL_Z)), R)
        cut = [1., 0., 0.]
        segs = [segment('REORIENT', home, reo), segment('PRE_CUT_move', reo, above), segment('PRE_CUT_lower', above, pre),
                segment('PRE_CUT_approach', pre, start),
                segment('SKIM', start, end, active=True, allow_blade_workpiece=True, cut_dir=cut),
                segment('WITHDRAW', end, up, allow_blade_workpiece=True), segment('WITHDRAW_up', up, up2),
                segment('RETURN_move', up2, reo), segment('RETURN_HOME', reo, home)]
        return segs, 4
    if kind in ('CUT', 'OBLIQUE'):
        R = edge_rotation_cut(edge_line, tilt_deg)
        a = R.apply([1., 0., 0.])
        center = np.array([0., 0., top/2])
        start_l = center+a*((top+.008-center[2])/a[2])
        end_l = center+a*((-.004-center[2])/a[2])
        pre_l = start_l+[0., 0., .03]
        home, reo, above = common_ends(task, edge_line, R, pre_l[:2])
        pre, start, end = (pose7(world(task, p), R) for p in (pre_l, start_l, end_l))
        segs = [segment('REORIENT', home, reo), segment('PRE_CUT_move', reo, above), segment('PRE_CUT_lower', above, pre),
                segment('PRE_CUT_approach', pre, start),
                segment('CUT', start, end, active=True, allow_blade_workpiece=True, cut_dir=a.tolist()),
                segment('WITHDRAW_kerf', end, start, allow_blade_workpiece=True), segment('WITHDRAW_up', start, pre),
                segment('RETURN_move', pre, above), segment('RETURN_move2', above, reo), segment('RETURN_HOME', reo, home)]
        return segs, 4
    if kind == 'SLICE_PUSH':
        R = edge_rotation_cut(edge_line, 0.)
        # frozen cutting_tasks_v3 SLICE-PUSH-01 knot positions (local x, z), y = 0
        P = {k: pose7(world(task, (x, 0., z)), R) for k, (x, z) in dict(
            pre1=(-.03, .058), s1=(-.03, .028), e1=(-.03, -.004), s2=(.03, .028), e2=(.03, -.004), p0=(.03, .003),
            p1=(.05, .003), p2=(.048, .003), up=(.048, .058)).items()}
        home, reo, above = common_ends(task, edge_line, R, (-.03, 0.))
        down = [0., 0., -1.]
        segs = [segment('REORIENT', home, reo), segment('PRE_CUT_move', reo, above), segment('PRE_CUT_lower', above, P['pre1']),
                segment('PRE_CUT_approach', P['pre1'], P['s1']),
                segment('CUT_1', P['s1'], P['e1'], active=True, allow_blade_workpiece=True, cut_dir=down),
                segment('WITHDRAW_kerf_1', P['e1'], P['s1'], allow_blade_workpiece=True),
                segment('MOVE_to_cut_2', P['s1'], P['s2']),
                segment('CUT_2', P['s2'], P['e2'], active=True, allow_blade_workpiece=True, cut_dir=down),
                segment('WITHDRAW_kerf_2', P['e2'], P['p0'], allow_blade_workpiece=True),
                segment('PUSH', P['p0'], P['p1'], active=True, allow_blade_workpiece=True, cut_dir=[1., 0., 0.]),
                segment('PUSH_release', P['p1'], P['p2'], allow_blade_workpiece=True),
                segment('WITHDRAW_up', P['p2'], P['up'], allow_blade_workpiece=True),
                segment('RETURN_move', P['up'], pose7(world(task, (.048, 0., FREE_LOCAL_Z)), R)),
                segment('RETURN_move2', pose7(world(task, (.048, 0., FREE_LOCAL_Z)), R), reo),
                segment('RETURN_HOME', reo, home)]
        return segs, 4
    raise ValueError(kind)


def sample_path(segs):
    """Dense samples (<= 2 mm, <= 1 deg); the first sample of each segment is its start pose."""
    samples = []
    for index, s in enumerate(segs):
        a, b = s['a'], s['b']
        Ra, Rb = Rotation.from_quat(a[3:]), Rotation.from_quat(b[3:])
        angle = math.degrees((Ra.inv()*Rb).magnitude())
        n = max(1, math.ceil(np.linalg.norm(b[:3]-a[:3])/THRESHOLDS['max_sample_translation_m']-1e-9),
                math.ceil(angle/THRESHOLDS['max_sample_rotation_deg']-1e-9))
        slerp = Slerp([0., 1.], Rotation.concatenate([Ra, Rb]))
        for i in range(0 if index == 0 else 1, n+1):
            t = i/n
            samples.append(dict(segment=s['name'], seg_index=index, pose=np.r_[a[:3]+t*(b[:3]-a[:3]), slerp(t).as_quat()],
                                active=s['active'], allow=s['allow_blade_workpiece'], cut_dir=s['cut_dir']))
    return samples


# ----------------------------------------------------------------------------- checks

CHECKS = ('ik_reachability', 'joint_limit_margin', 'joint_continuity', 'wrist_camera_clearance', 'gripper_pad_clearance',
          'knife_handle_clearance', 'blade_collision', 'franka_self_collision', 'franka_workcell_collision',
          'franka1_collision', 'trim_substrate_clearance')
SLACK_SCALE = dict(clearance=.005, joint_limit_margin=.05, joint_continuity=.06, substrate=.0005)


def pair_threshold(subject, target, allow):
    if target == 'self':
        return THRESHOLDS['self_clearance_m']
    if subject == 'blade' and target == 'workpiece':
        return None if allow else THRESHOLDS['blade_free_workpiece_clearance_m']
    if subject == 'blade' and target in ('table', 'board', 'riser'):
        return THRESHOLDS['blade_fixture_clearance_m']
    return THRESHOLDS['env_clearance_m']


def checks_for(subject, target):
    out = []
    if subject == 'camera':
        out.append('wrist_camera_clearance')
    if subject in ('pad', 'finger'):
        out.append('gripper_pad_clearance')
    if subject == 'handle':
        out.append('knife_handle_clearance')
    if subject == 'blade':
        out.append('blade_collision')
    if target == 'self':
        out.append('franka_self_collision')
    if target == 'franka1':
        out.append('franka1_collision')
    if subject in ('link', 'hand') and target not in ('self', 'franka1'):
        out.append('franka_workcell_collision')
    return out


def ik_ok(sample, info):
    if info.get('joint_space'):
        return True
    strict = sample['active']
    return (info['position_error_m'] <= (THRESHOLDS['ik_active_position_m'] if strict else THRESHOLDS['ik_position_m'])
            and info['rotation_error_rad'] <= (THRESHOLDS['ik_active_rotation_rad'] if strict else THRESHOLDS['ik_rotation_rad']))


_CACHE = {}
TASK_OF_KIND = {'TRIM': 'TRIM-01-R', 'CUT': 'CUT-01-R', 'OBLIQUE': 'CUT-01-R', 'SLICE_PUSH': 'SLICE-PUSH-01'}


def get_models(profile_id, task_id, placement):
    if ('kin', profile_id) not in _CACHE:
        _CACHE[('kin', profile_id)] = Kinematics(profile_id)
    key = ('geo', profile_id, task_id, placement)
    if key not in _CACHE:
        _CACHE[key] = GeometryModel(profile_id, placement_task(task_id, placement))
    return _CACHE[('kin', profile_id)], _CACHE[key]


def propagate(kin, samples, k, q_k, info_k, stop=None):
    N = len(samples) if stop is None else stop
    qs, infos = [None]*len(samples), [None]*len(samples)
    qs[k], infos[k] = q_k, info_k
    for i in range(k+1, N):
        qs[i], infos[i] = kin.solve(samples[i]['pose'], qs[i-1], samples[i]['active'])
    if stop is None:
        for i in range(k-1, -1, -1):
            qs[i], infos[i] = kin.solve(samples[i]['pose'], qs[i+1], samples[i]['active'])
    return qs, infos


def joint_segment(kin, qa, qb, name):
    """Joint-space interpolation (<= 0.5 deg per joint, <= 2 mm edge travel per step)."""
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    n = max(1, math.ceil(np.abs(qb-qa).max()/math.radians(.5)))
    while True:
        qs = [qa+(qb-qa)*i/n for i in range(1, n+1)]
        poses = [kin.edge_pose(q) for q in qs]
        steps = np.linalg.norm(np.diff(np.array([kin.edge_pose(qa)[:3]]+[p[:3] for p in poses]), axis=0), axis=1)
        if steps.max() <= THRESHOLDS['max_sample_translation_m'] or n > 4000:
            break
        n *= 2
    samples = [dict(segment=name, seg_index=-1, pose=p, active=False, allow=False, cut_dir=None) for p in poses]
    return samples, qs, [dict(joint_space=True, position_error_m=0., rotation_error_rad=0., converged=True)]*len(qs)


def evaluate_branch(kin, geo, samples, qs, infos, *, kind, task, rows_out=False):
    """Per-check worst slack over all samples + task metrics."""
    from robotic_cutting_sim.scene_frames import pose_matrix
    mount = pose_matrix(kin.mount)
    G = geo.geoms
    worst = {c: dict(slack=np.inf) for c in CHECKS}
    rows = []
    target_z = task.points_to_world([0., 0., task.geometry.base_height_m])[2]
    trim = dict(tau=[], psi=[], rho=[], edge_err=[], blade_min=[], parts=[], pos=[])
    lower, upper = kin.lower, kin.upper
    fk_err = 0.
    for i, (s, q, info) in enumerate(zip(samples, qs, infos)):
        geo.set_q(q)
        best = geo.evaluate()
        row = dict(i=i, segment=s['segment'], q=np.round(q, 5).tolist(),
                   ik_pos_mm=round(1e3*info['position_error_m'], 4), ik_rot_mrad=round(1e3*info['rotation_error_rad'], 4))
        margin = float(np.minimum(q-lower, upper-q).min())
        row['joint_margin_rad'] = round(margin, 5)
        n = (margin-THRESHOLDS['joint_margin_rad'])/SLACK_SCALE['joint_limit_margin']
        if n < worst['joint_limit_margin']['slack']:
            joint = int(np.argmin(np.minimum(q-lower, upper-q)))
            worst['joint_limit_margin'] = dict(slack=n, value=margin, threshold=THRESHOLDS['joint_margin_rad'], sample=i,
                                               segment=s['segment'], joint=joint+1)
        ok = ik_ok(s, info)
        if not ok and worst['ik_reachability'].get('value', True):
            worst['ik_reachability'] = dict(slack=-1., value=False, sample=i, segment=s['segment'],
                                            position_error_m=info['position_error_m'], rotation_error_rad=info['rotation_error_rad'])
        per_check = {}
        for (subject, target), (dist, g, h) in best.items():
            thr = pair_threshold(subject, target, s['allow'])
            if thr is None:
                continue
            slack = (dist-thr)/SLACK_SCALE['clearance']
            for c in checks_for(subject, target):
                if c not in per_check or slack < per_check[c][0]:
                    per_check[c] = (slack, dist, thr, G[g]['name'], G[h]['name'])
        for c, (slack, dist, thr, a, b) in per_check.items():
            row[c] = round(dist, 5)
            if slack < worst[c]['slack']:
                worst[c] = dict(slack=slack, value=dist, threshold=thr, sample=i, segment=s['segment'], pair=[a, b])
        T_edge = pose_matrix(geo.hand_pose())@mount
        row['edge_world'] = np.round(T_edge[:3, 3], 6).tolist()
        if i == 0 or s['active']:
            fk_err = max(fk_err, float(np.linalg.norm(kin.edge_pose(q)[:3]-T_edge[:3, 3])))
        if kind == 'TRIM' and s['active']:
            tau, psi, rho = orientation_metrics(T_edge[:3, :3], s['cut_dir'])
            trim['tau'].append(tau); trim['psi'].append(psi); trim['rho'].append(rho)
            trim['pos'].append(T_edge[:3, 3])
            trim['edge_err'].append(T_edge[2, 3]-target_z-task.edge_clearance_m)
            blade = geo.geom_vertices(geo.by_name['tool_blade'])
            trim['blade_min'].append(float(blade[:, 2].min()-target_z))
            e = T_edge[:3, 1]
            for name, g in geo.by_name.items():
                v = G[g]
                if v['group'] == 'arm0_hand' and v['role'] != 'blade':
                    pts = geo.geom_vertices(g)
                    height = float(pts[:, 2].min()-target_z)
                    if height < .05:
                        low = pts[pts[:, 2]-target_z < .005]
                        along = float(np.abs((low-T_edge[:3, 3])@e).min()) if len(low) else None
                        trim['parts'].append((height, name, along))
        if rows_out:
            rows.append(row)
    steps = np.abs(np.diff(np.asarray(qs), axis=0)).max(1) if len(qs) > 1 else np.zeros(1)
    j = int(np.argmax(steps))
    worst['joint_continuity'] = dict(slack=(THRESHOLDS['joint_step_rad']-float(steps[j]))/SLACK_SCALE['joint_continuity'],
                                     value=float(steps[j]), threshold=THRESHOLDS['joint_step_rad'], sample=j+1,
                                     segment=samples[j+1]['segment'] if len(samples) > 1 else None)
    if 'value' not in worst['ik_reachability']:
        pos = [x['position_error_m'] for x in infos]
        worst['ik_reachability'] = dict(slack=1., value=True, max_position_error_m=float(max(pos)),
                                        max_rotation_error_rad=float(max(x['rotation_error_rad'] for x in infos)),
                                        max_active_position_error_m=float(max([x['position_error_m'] for x, s in zip(infos, samples) if s['active']] or [0.])))
    if kind == 'TRIM' and trim['blade_min']:
        j = int(np.argmin(trim['blade_min']))
        low = trim['blade_min'][j]
        worst['trim_substrate_clearance'] = dict(
            slack=low/SLACK_SCALE['substrate'], value=low, threshold=0.,
            note='lowest blade vertex above the TRIM target plane during SKIM (edge at plane + edge_clearance); '
                 'negative = the lower bevel/face runs below the plane into the kept substrate')
    elif kind != 'TRIM':
        worst['trim_substrate_clearance'] = dict(slack=10., value=None, not_applicable=True)
    checks = {}
    for c in CHECKS:
        w = worst[c]
        if not np.isfinite(w['slack']):
            w = dict(slack=10., value=None, note='no pair within distmax or not applicable')
        checks[c] = {**{k: v for k, v in w.items()}, 'status': 'PASS' if w['slack'] >= 0 else 'FAIL'}
    failing = [c for c in CHECKS if checks[c]['status'] == 'FAIL']
    binding = min(CHECKS, key=lambda c: checks[c]['slack'])
    metrics = dict(fk_consistency_max_m=fk_err, samples=len(samples))
    if kind == 'TRIM' and trim['tau']:
        pos = np.asarray(trim['pos'])
        d = np.diff(pos, axis=0)
        mu = np.degrees(np.arcsin(np.clip(d[:, 2]/np.maximum(np.linalg.norm(d, axis=1), 1e-12), -1, 1)))
        lowest = min(trim['parts']) if trim['parts'] else None
        metrics.update(tau_max_deg=max(map(abs, trim['tau'])), psi_max_deg=max(trim['psi']),
                       rho_min_deg=min(trim['rho']), rho_max_deg=max(trim['rho']), mu_max_abs_deg=float(np.abs(mu).max()),
                       edge_height_error_max_m=float(np.abs(trim['edge_err']).max()),
                       blade_lowest_above_target_plane_m=min(trim['blade_min']),
                       lowest_non_blade_part_above_target_plane=None if lowest is None else dict(
                           height_m=lowest[0], part=lowest[1], along_edge_from_edge_centre_m=lowest[2]))
    return dict(status='PASS' if not failing else 'FAIL', failing=failing, binding=binding,
                binding_slack=checks[binding]['slack'], checks=checks, metrics=metrics), rows


def score(result):
    return (len(result['failing']), -result['binding_slack'])


def score_full(result):
    """Prefer passing branches, then those that return to q_home without a large null-space/branch swing."""
    swing = (result.get('return_closure_max_joint_rad') or 0.)+(result.get('home_ref_closure_max_joint_rad') or 0.)
    return (len(result['failing']), int(swing > .1), -result['binding_slack'])


def run_candidate(job):
    """One candidate: multi-start at the cut start, continuation both ways, closure to HOME."""
    t0 = time.perf_counter()
    kind, profile_id, edge_line, placement = job['kind'], job['profile_id'], job['edge_line'], job['placement']
    task_id = TASK_OF_KIND[kind]
    kin, geo = get_models(profile_id, task_id, placement)
    task = geo.task
    segs, k_seg = build_path(task, kind, edge_line, rake_deg=job.get('rake_deg', 0.), tilt_deg=job.get('tilt_deg', 0.))
    samples = sample_path(segs)
    k = next(i for i, s in enumerate(samples) if s['active'])
    active_end = next((i for i in range(k, len(samples)) if not samples[i]['active']), len(samples))
    from robotic_cutting_sim.workcell import PARK_JOINTS
    from robotic_cutting_sim.task_scenarios import initial_joints
    seeds = [np.asarray(PARK_JOINTS, float)]
    legacy = initial_joints(BASE_SCENARIO, 'TRIM-01-R')
    if legacy:
        seeds.append(np.asarray(legacy, float))
    q_ref = None if job.get('q_home_ref') is None else np.asarray(job['q_home_ref'], float)
    if q_ref is not None:
        q, info = kin.solve(samples[0]['pose'], q_ref, False)
        qs, _ = propagate(kin, samples, 0, q, info, stop=k+1)
        seeds.insert(0, qs[k])
    ref_start = None
    rng = np.random.default_rng(SEED)
    seeds += list(rng.uniform(kin.lower+.1, kin.upper-.1, (job.get('n_seeds', 40), 7)))
    starts = []
    for seed in seeds:
        q, info = kin.solve(samples[k]['pose'], seed, True)
        if not ik_ok(samples[k], info):
            continue
        if all(np.abs(q-p).max() > .3 for p, _ in starts):
            starts.append((q, info))
            if q_ref is not None and seed is seeds[0]:
                ref_start = q
    screened = []
    for q, info in starts[:job.get('max_screen', 12)]:
        qs, infos = propagate(kin, samples, k, q, info, stop=active_end)
        sub = list(range(k, active_end))
        res, _ = evaluate_branch(kin, geo, [samples[i] for i in sub], [qs[i] for i in sub], [infos[i] for i in sub],
                                 kind=kind, task=task)
        screened.append((score(res), q, info))
    screened.sort(key=lambda x: x[0])
    chosen = screened[:job.get('max_branches', 4)]
    if ref_start is not None and not any(x[1] is ref_start for x in chosen):
        chosen += [x for x in screened if x[1] is ref_start]   # always follow the branch continued from q_home_ref
    branches = []
    best = None
    for _, q, info in chosen:
        qs, infos = propagate(kin, samples, k, q, info)
        full_s, full_q, full_i = list(samples), list(qs), list(infos)
        q_home = q_ref if q_ref is not None else full_q[0]
        if q_ref is not None and np.abs(q_ref-full_q[0]).max() > 1e-3:
            cs, cq, ci = joint_segment(kin, q_ref, full_q[0], 'HOME_ref_closure')
            full_s, full_q, full_i = cs[:-1]+full_s, cq[:-1]+full_q, ci[:-1]+full_i
            full_s.insert(0, dict(segment='HOME', seg_index=-1, pose=kin.edge_pose(q_ref), active=False, allow=False, cut_dir=None))
            full_q.insert(0, q_ref)
            full_i.insert(0, dict(joint_space=True, position_error_m=0., rotation_error_rad=0., converged=True))
        closure = float(np.abs(full_q[-1]-q_home).max())
        if closure > 1e-3:
            cs, cq, ci = joint_segment(kin, full_q[-1], q_home, 'RETURN_HOME_closure')
            full_s, full_q, full_i = full_s+cs, full_q+cq, full_i+ci
        res, rows = evaluate_branch(kin, geo, full_s, full_q, full_i, kind=kind, task=task, rows_out=True)
        res.update(q_home=np.asarray(q_home).tolist(), q_cut_start=np.asarray(q).tolist(), return_closure_max_joint_rad=closure,
                   home_ref_closure_max_joint_rad=None if q_ref is None else float(np.abs(q_ref-qs[0]).max()))
        branches.append(dict(status=res['status'], failing=res['failing'], binding=res['binding'],
                             binding_slack=res['binding_slack'], q_cut_start=res['q_cut_start'],
                             return_closure_max_joint_rad=closure, home_ref_closure_max_joint_rad=res['home_ref_closure_max_joint_rad'],
                             continued_from_q_home_ref=ref_start is not None and q is ref_start))
        if best is None or score_full(res) < score_full(best[0]):
            best = (res, rows, full_s)
    cid = job['cid']
    summary = dict(cid=cid, kind=kind, profile_id=profile_id, edge_line=edge_line, placement=placement,
                   placement_world_z_m=PLACEMENTS[placement], rake_deg=job.get('rake_deg'), tilt_deg=job.get('tilt_deg'),
                   task_id=task_id, samples=len(samples), ik_starts_tried=len(seeds), distinct_ik_starts=len(starts),
                   branches_evaluated=len(branches), branches=branches, wall_s=None)
    if best is None:
        summary.update(status='FAIL', failing=['ik_reachability'], binding='ik_reachability', binding_slack=-1.,
                       checks={c: dict(status='NOT_RUN') for c in CHECKS} | {'ik_reachability': dict(
                           status='FAIL', value=False, note='no strict IK solution at the cut start from any seed')})
    else:
        res, rows, full_s = best
        summary.update({k_: v for k_, v in res.items()})
        summary['home_edge_pose_world'] = np.asarray(samples[0]['pose']).tolist()
        from robotic_cutting_sim.scene_frames import pose_matrix
        geo.set_q(np.asarray(res['q_home']))
        T_hand = pose_matrix(geo.hand_pose())
        blade = geo.geom_vertices(geo.by_name['tool_blade'])
        pads = np.concatenate([geo.geom_vertices(geo.by_name[f'arm0_{side}_finger_pad']) for side in ('left', 'right')])
        summary['home_check'] = dict(hand_approach_axis_world=T_hand[:3, 2].tolist(),
                                     hand_down=bool(T_hand[2, 2] < -.99),
                                     blade_top_below_pad_bottom_m=float(pads[:, 2].min()-blade[:, 2].max()),
                                     blade_tip_height_above_workpiece_top_m=float(
                                         blade[:, 2].min()-task.points_to_world([0., 0., task.geometry.base_height_m])[2]))
        summary['segments'] = [dict(name=s['name'], active=s['active'], allow_blade_workpiece=s['allow_blade_workpiece'],
                                    start=s['a'].tolist(), end=s['b'].tolist()) for s in segs]
        out = Path(job['out_dir'])/'candidates'
        out.mkdir(parents=True, exist_ok=True)
        summary['wall_s'] = time.perf_counter()-t0
        (out/f'{cid}.json').write_text(json.dumps(dict(summary=summary, rows=rows), default=float)+'\n')
    summary['wall_s'] = time.perf_counter()-t0
    return summary


# ----------------------------------------------------------------------------- rendering

ROLE_COLORS = dict(link=(.86, .86, .88), hand=(.93, .93, .95), pad=(.12, .12, .12), finger=(.30, .30, .32),
                   handle=(.20, .18, .17), blade=(.62, .70, .80), camera=(.15, .15, .18), board=(.72, .53, .34),
                   riser=(.55, .55, .57), workpiece=(.86, .62, .60), global_camera=(.2, .2, .2))


def render(geo, q, path, title, close_center, *, extra_note='', half=.22):
    """Painter-sorted matplotlib projection of the actual Newton/MuJoCo collider meshes at q."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    geo.set_q(q)
    tris, cols = [], []
    light = np.array([.35, -.55, .75])
    light /= np.linalg.norm(light)
    for g, v in geo.geoms.items():
        if v['role'] in ('table', 'global_camera') or v['name'].endswith('hand_capsule'):
            continue
        verts, faces = geo.geom_mesh(g)
        base = np.array(ROLE_COLORS.get(v['role'], (.8, .8, .8)))
        if v['arm'] == 1:
            base = .55*base+.45
        center = verts.mean(0)
        for f in faces:
            t = verts[f]
            n = np.cross(t[1]-t[0], t[2]-t[0])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n /= norm
            if n@(t.mean(0)-center) < 0:
                n = -n
            tris.append(t)
            cols.append(np.clip(base*(.55+.45*abs(float(n@light))), 0, 1))
    n_solid = len(tris)
    step = .05
    for x in np.arange(-.75, .6, step):
        for y in np.arange(-.55, .55, step):
            tris.append(np.array([[x, y, -.01], [x+step, y, -.01], [x+step, y+step, -.01], [x, y+step, -.01]]))
            cols.append(np.array((.93, .93, .90)) if (round(x/step)+round(y/step)) % 2 else np.array((.90, .90, .87)))
    c = np.asarray(close_center, float)
    close = ((c[0]-half, c[0]+half), (c[1]-half, c[1]+half), (-.03, max(c[2]+half, .25)))
    views = [(22, -58, ((-.75, .55), (-.55, .55), (-.01, .85)), 'workcell (Franka 1 faded)', False),
             (0, 0, close, 'front: looking along -x (cut direction +x)', True),
             (0, -90, close, 'side: looking along +y', True)]
    fig = plt.figure(figsize=(13.5, 4.4), dpi=90)
    for k, (elev, azim, box, label, clip) in enumerate(views):
        ax = fig.add_subplot(1, 3, k+1, projection='3d')
        if clip:
            keep = [i for i, t in enumerate(tris[:n_solid]) if all(box[j][0] <= t[:, j].mean() <= box[j][1]
                                                                   and np.ptp(t[:, j]) < .5*(box[j][1]-box[j][0]) for j in range(3))]
            T, C = [tris[i] for i in keep], [cols[i] for i in keep]
            ax.set_proj_type('ortho')
            # table (top at z = -0.01) as grey hatching below its top surface
            u = box[1] if azim == 0 else box[0]
            for z in np.arange(-.01, box[2][0], -.004):
                xs = (box[0][1],)*2 if azim == 0 else u
                ys = u if azim == 0 else (box[1][0],)*2
                ax.plot(xs, ys, [z, z], color=(.75, .75, .72), linewidth=1.2)
        else:
            T, C = tris, cols
        if T:
            ax.add_collection3d(Poly3DCollection(T, facecolors=C, edgecolors='none', linewidths=0), autolim=False)
        ax.set_xlim(*box[0]); ax.set_ylim(*box[1]); ax.set_zlim(*box[2])
        ax.set_box_aspect([b[1]-b[0] for b in box])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('x [m]', fontsize=7); ax.set_ylabel('y [m]', fontsize=7); ax.set_zlabel('z [m]', fontsize=7)
    fig.suptitle(title+('\n'+extra_note if extra_note else ''), fontsize=9)
    fig.tight_layout()
    tmp = Path(path).with_suffix('.tmp.png')
    fig.savefig(tmp)
    plt.close(fig)
    Image.open(tmp).convert('RGB').quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    tmp.unlink()
    return dict(path=str(path), bytes=Path(path).stat().st_size)


# ----------------------------------------------------------------------------- main

def cid_of(kind, profile_id, edge_line, placement, angle):
    arr = profile_id.replace('ruler_knife_', '').replace('_', '')
    tag = 'rho' if kind == 'TRIM' else 'tilt'
    return f'{kind}_{arr}_{"py" if edge_line == "+y" else "my"}_{tag}{angle:04.1f}_{placement}'


def compact(summary):
    c = summary['checks']
    pick = lambda name: c.get(name, {}).get('value')
    return dict(cid=summary['cid'], status=summary['status'], failing=summary['failing'], binding=summary['binding'],
                binding_detail={k: v for k, v in c[summary['binding']].items() if k not in ('slack',)},
                min_clearance_m=dict(wrist_camera=pick('wrist_camera_clearance'), gripper=pick('gripper_pad_clearance'),
                                     handle=pick('knife_handle_clearance'), blade=pick('blade_collision'),
                                     self=pick('franka_self_collision'), workcell=pick('franka_workcell_collision'),
                                     franka1=pick('franka1_collision')),
                min_joint_margin_rad=pick('joint_limit_margin'), max_joint_step_rad=pick('joint_continuity'),
                ik=c.get('ik_reachability'), metrics=summary.get('metrics'), q_home=summary.get('q_home'),
                home_check=summary.get('home_check'),
                return_closure_max_joint_rad=summary.get('return_closure_max_joint_rad'),
                home_ref_closure_max_joint_rad=summary.get('home_ref_closure_max_joint_rad'),
                distinct_ik_starts=summary.get('distinct_ik_starts'), branches=[
                    dict(status=b['status'], binding=b['binding'], failing=b['failing']) for b in summary.get('branches', [])],
                wall_s=summary['wall_s'])


def heel_drop_m(profile, rake_deg):
    """Lowest blade point below the edge line for a horizontal edge at rake rho (lower bevel heel or spine)."""
    section = profile.cross_section_xz()
    r = math.radians(rake_deg)
    z = -section[:, 0]*math.sin(r)+section[:, 1]*math.cos(r)
    return float(max(0., -z.min()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--n-seeds', type=int, default=40)
    p.add_argument('--max-screen', type=int, default=12)
    p.add_argument('--max-branches', type=int, default=4)
    p.add_argument('--arrangements', nargs='+', default=list(ARRANGEMENTS))
    p.add_argument('--edge-lines', nargs='+', default=['py', 'my'], choices=['py', 'my'],
                   help='world direction from grip to tip: py = +y, my = -y')
    p.add_argument('--rakes', nargs='+', type=float, default=list(RAKES_DEG))
    p.add_argument('--placements', nargs='+', default=list(PLACEMENTS))
    p.add_argument('--oblique-tilt-deg', type=float, default=20.)
    p.add_argument('--render-dir', type=Path)
    p.add_argument('--evidence', type=Path)
    a = p.parse_args()
    a.edge_lines = [{'py': '+y', 'my': '-y'}[e] for e in a.edge_lines]
    from robotic_cutting_sim.recording import new_output, write_json, manifest
    from robotic_cutting_sim.tool_profile import PROFILES
    out = new_output(a.out)
    t_start = time.perf_counter()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True).stdout
    sources = {rel: sha(ROOT/rel) for rel in ['scripts/preflight_wp11_orientations.py', 'src/robotic_cutting_sim/tool_profile.py',
               'src/robotic_cutting_sim/robot.py', 'src/robotic_cutting_sim/camera_rig.py', 'src/robotic_cutting_sim/task_scenarios.py',
               'configs/wp10_scenarios_v3.json', 'configs/cutting_tasks_v2.json', 'configs/cutting_tasks_v3.json']}
    config = dict(command=' '.join(sys.argv), seed=SEED, thresholds=THRESHOLDS, finger_q_m=FINGER_Q_M,
                  home_local_edge_centre_m=HOME_LOCAL, free_local_z_m=FREE_LOCAL_Z, arrangements=a.arrangements,
                  edge_lines=a.edge_lines, rakes_deg=a.rakes, placements={k: PLACEMENTS[k] for k in a.placements},
                  n_seeds=a.n_seeds, max_screen=a.max_screen, max_branches=a.max_branches, git_commit=commit,
                  git_dirty_tracked=dirty.splitlines(), source_sha256=sources,
                  profiles={pid: PROFILES[pid].to_dict() for pid in a.arrangements},
                  hand_from_edge={pid: list(PROFILES[pid].hand_from_edge()) for pid in a.arrangements})
    write_json(out/'manifest.json', manifest(config, 'newton_ik_mujoco_geom_distance_preflight'))
    common = dict(out_dir=str(out), n_seeds=a.n_seeds, max_screen=a.max_screen, max_branches=a.max_branches)
    jobs = [dict(kind='TRIM', profile_id=pid, edge_line=e, placement=pl, rake_deg=r, cid=cid_of('TRIM', pid, e, pl, r), **common)
            for pl in a.placements for pid in a.arrangements for e in a.edge_lines for r in a.rakes]
    ctx = mp.get_context('spawn')
    print(f'[C2] {len(jobs)} TRIM candidates on {a.workers} workers', flush=True)
    with ctx.Pool(a.workers) as pool:
        trim = []
        for s in pool.imap_unordered(run_candidate, jobs):
            trim.append(s)
            print(f"[C2] {s['cid']}: {s['status']} binding={s['binding']} failing={s['failing']} ({s['wall_s']:.0f}s)", flush=True)
    t_trim = time.perf_counter()-t_start
    trim.sort(key=lambda s: s['cid'])
    passing = [s for s in trim if s['status'] == 'PASS']
    def robust_slack(*summaries):
        """Smallest normalized slack over the clearance and joint-margin checks (tie-break only)."""
        return min(v['slack'] for x in summaries for c, v in x['checks'].items()
                   if c not in ('trim_substrate_clearance', 'joint_continuity', 'ik_reachability') and 'slack' in v)

    near = sorted([s for s in passing if s['rake_deg'] <= 10.+1e-9],
                  key=lambda s: (s['rake_deg'], PLACEMENTS[s['placement']], -robust_slack(s)))
    # Follow-ups: every (arrangement, edge line, placement) with a passing near-horizontal TRIM, at its smallest passing rake.
    combos = {}
    for s in sorted(passing, key=lambda s: (s['rake_deg'], -s['binding_slack'])):
        combos.setdefault((s['profile_id'], s['edge_line'], s['placement']), s)
    follow_jobs = []
    for (pid, e, pl), s in combos.items():
        for kind, tilt in (('CUT', 0.), ('OBLIQUE', a.oblique_tilt_deg), ('SLICE_PUSH', 0.)):
            follow_jobs.append(dict(kind=kind, profile_id=pid, edge_line=e, placement=pl, tilt_deg=tilt, q_home_ref=s['q_home'],
                                    home_from=s['cid'], cid=cid_of(kind, pid, e, pl, tilt), **common))
    print(f'[C2] {len(follow_jobs)} CUT/OBLIQUE/SLICE_PUSH follow-ups', flush=True)
    follow = []
    if follow_jobs:
        with ctx.Pool(min(a.workers, len(follow_jobs))) as pool:
            for s, job in zip(pool.imap(run_candidate, follow_jobs), follow_jobs):
                s['home_from'] = job['home_from']
                follow.append(s)
                print(f"[C2] {s['cid']}: {s['status']} binding={s['binding']} failing={s['failing']} ({s['wall_s']:.0f}s)", flush=True)
    t_follow = time.perf_counter()-t_start-t_trim
    fol = {(s['profile_id'], s['edge_line'], s['placement'], s['kind']): s for s in follow}

    def supports(s):
        key = (s['profile_id'], s['edge_line'], s['placement'])
        return {k: fol[key+(k,)]['status'] if key+(k,) in fol else 'NOT_RUN' for k in ('CUT', 'OBLIQUE', 'SLICE_PUSH')}

    def rec(best, rule):
        return dict(rule=rule, cid=best['cid'], arrangement=best['profile_id'], edge_line_world=best['edge_line'],
                    rake_deg=best['rake_deg'], placement=best['placement'], placement_world_z_m=PLACEMENTS[best['placement']],
                    q_home=best['q_home'], home_edge_pose_world=best['home_edge_pose_world'], home_check=best.get('home_check'),
                    finger_q_m=FINGER_Q_M, binding=best['binding'], binding_detail=best['checks'][best['binding']],
                    trim_metrics=best.get('metrics'), other_tasks=supports(best),
                    other_task_cids={k: fol[(best['profile_id'], best['edge_line'], best['placement'], k)]['cid']
                                     for k in ('CUT', 'OBLIQUE', 'SLICE_PUSH')
                                     if (best['profile_id'], best['edge_line'], best['placement'], k) in fol})

    recommendation = rec(near[0], 'smallest rho in 0-10 deg passing every check (incl. trim_substrate_clearance); '
                         'ties -> lowest placement, then largest smallest-normalized slack over the other checks') if near else None
    all_task = [s for s in near if all(v == 'PASS' for v in supports(s).values())]
    all_task.sort(key=lambda s: (s['rake_deg'], PLACEMENTS[s['placement']], -robust_slack(s, *[
        fol[(s['profile_id'], s['edge_line'], s['placement'], k)] for k in ('CUT', 'OBLIQUE', 'SLICE_PUSH')])))
    all_task_recommendation = rec(all_task[0], 'as above, restricted to candidates whose CUT, 20 deg OBLIQUE and SLICE-PUSH '
                                  'paths from the same q_home also pass; ties -> largest smallest slack over TRIM + the three paths') if all_task else None
    geometry_only = sorted([s for s in trim if s['rake_deg'] <= 10.+1e-9 and set(s['failing']) <= {'trim_substrate_clearance'}],
                           key=lambda s: (s['rake_deg'], PLACEMENTS[s['placement']], -s['binding_slack']))
    geometry_only_smallest = None if not geometry_only else dict(
        cid=geometry_only[0]['cid'], rake_deg=geometry_only[0]['rake_deg'], failing=geometry_only[0]['failing'],
        blade_lowest_above_target_plane_m=geometry_only[0]['metrics'].get('blade_lowest_above_target_plane_m'),
        edge_raise_to_keep_lowest_blade_point_at_plane_plus_clearance_m=heel_drop_m(PROFILES[geometry_only[0]['profile_id']],
                                                                                    geometry_only[0]['rake_deg']),
        note='robot/tool/workcell geometry passes; only the blade-heel-below-target-plane condition fails at the frozen '
             '0.2 mm edge clearance. Raising the edge by the listed amount (a controller parameter) removes it.')
    blocking = {}
    for r in a.rakes:
        rows = [s for s in trim if abs(s['rake_deg']-r) < 1e-9]
        blocking[f'{r:g}'] = dict(passing=[s['cid'] for s in rows if s['status'] == 'PASS'],
                                  failing={s['cid']: dict(binding=s['binding'], failing=s['failing'],
                                           value=s['checks'][s['binding']].get('value'),
                                           pair=s['checks'][s['binding']].get('pair')) for s in rows if s['status'] != 'PASS'})
    smallest_any = min((s['rake_deg'] for s in passing), default=None)
    any_profile = PROFILES[a.arrangements[0]]
    heel = {f'{r:g}': heel_drop_m(any_profile, r) for r in a.rakes}
    images = []
    if a.render_dir and recommendation:
        a.render_dir.mkdir(parents=True, exist_ok=True)
        # render the single setup that serves every task when one exists, else the TRIM-only choice
        best = all_task[0] if all_task else near[0]
        tag = best['cid']
        rows = json.loads((out/'candidates'/f'{tag}.json').read_text())['rows']
        kin, geo = get_models(best['profile_id'], 'TRIM-01-R', best['placement'])
        task = geo.task
        centre = task.points_to_world([0., 0., .03])
        side = np.array([0., -.08 if best['edge_line'] == '+y' else .08, 0.])
        active = [r for r in rows if r['segment'] == 'SKIM']
        mid = active[len(active)//2]
        arr = 'v2_x' if best['profile_id'].endswith('_x') else 'v2_y'
        images.append(render(geo, np.asarray(rows[0]['q']), a.render_dir/'c2_home.png',
                             f'C2 HOME (q_home): ruler_knife_{arr}, edge line {best["edge_line"]}, placement +{PLACEMENTS[best["placement"]]*1e3:.0f} mm',
                             centre+[0., 0., .2], extra_note='hand down, blade hanging below the fingers; Newton/MuJoCo collider hulls'))
        images.append(render(geo, np.asarray(mid['q']), a.render_dir/'c2_trim_skim.png',
                             f'C2 TRIM skim, rho = {best["rake_deg"]:g} deg (tau = psi = mu = 0): ruler_knife_{arr}, edge line {best["edge_line"]}',
                             centre+side, extra_note=f'edge at target plane + 0.2 mm; binding: {best["binding"]}'))
        key = (best['profile_id'], best['edge_line'], best['placement'])
        for kind, seg in (('CUT', 'CUT'), ('SLICE_PUSH', 'PUSH')):
            if key+(kind,) in fol and (out/'candidates'/f"{fol[key+(kind,)]['cid']}.json").exists():
                s = fol[key+(kind,)]
                rows = json.loads((out/'candidates'/f"{s['cid']}.json").read_text())['rows']
                act = [r for r in rows if r['segment'] == seg]
                _, geo2 = get_models(best['profile_id'], TASK_OF_KIND[kind], best['placement'])
                images.append(render(geo2, np.asarray(act[-1]['q']), a.render_dir/f'c2_{kind.lower()}.png',
                                     f'C2 {kind} ({seg} end), same knife/placement/q_home: {s["status"]}',
                                     centre+side+[0., 0., .03], extra_note=f'binding: {s["binding"]} {s["checks"][s["binding"]].get("pair", "")}'))
                break
    result = dict(status='COMPLETE', scope=('Geometry-only Newton IK + MuJoCo signed distances on Newton-converted official '
                  'Franka colliders, finger visual hulls, v2 ToolSpec hulls, v5 wrist D405 boxes and workcell boxes. No dynamics, '
                  'no material, no grasp holding, no sensor rendering. Raised +100/+150 mm placements are compute-only.'),
                  config=config, recommendation=recommendation, all_task_recommendation=all_task_recommendation,
                  geometry_only_smallest_rake=geometry_only_smallest, smallest_passing_rake_deg_any=smallest_any,
                  rake_blocking=blocking, trim_blade_heel_drop_below_edge_m=heel,
                  trim=[compact(s) for s in trim], follow_ups=[{**compact(s), 'home_from': s.get('home_from')} for s in follow],
                  follow_up_support={f'{k[0]}|{k[1]}|{k[2]}': {kk: fol[k+(kk,)]['status'] for kk in ('CUT', 'OBLIQUE', 'SLICE_PUSH')
                                     if k+(kk,) in fol} for k in combos},
                  images=images, wall_s=dict(trim=t_trim, follow_ups=t_follow, total=time.perf_counter()-t_start))
    write_json(out/'result.json', result)
    if a.evidence:
        a.evidence.parent.mkdir(parents=True, exist_ok=True)
        table = [dict(cid=s['cid'], status=s['status'], binding=s['binding'],
                      binding_value=s['checks'][s['binding']].get('value'), binding_pair=s['checks'][s['binding']].get('pair'),
                      failing=s['failing'], min_joint_margin_rad=s['checks']['joint_limit_margin'].get('value'),
                      max_joint_step_rad=s['checks']['joint_continuity'].get('value')) for s in trim+follow]
        write_json(a.evidence, dict(evidence='WP-11 Phase C2 geometric feasibility preflight', result_dir=str(a.out),
                                    result_json=str(a.out/'result.json'), command=config['command'], git_commit=commit,
                                    git_dirty_tracked=config['git_dirty_tracked'], seed=SEED, thresholds=THRESHOLDS,
                                    source_sha256=sources, scope=result['scope'], recommendation=recommendation,
                                    all_task_recommendation=all_task_recommendation,
                                    geometry_only_smallest_rake=geometry_only_smallest,
                                    smallest_passing_rake_deg_any=smallest_any, trim_blade_heel_drop_below_edge_m=heel,
                                    follow_up_support=result['follow_up_support'], table=table, images=images,
                                    wall_s=result['wall_s']))
    print(json.dumps(dict(recommendation=recommendation, all_task_recommendation=all_task_recommendation,
                          geometry_only_smallest_rake=geometry_only_smallest, smallest_passing_rake_deg_any=smallest_any,
                          follow_up_support=result['follow_up_support'], wall_s=result['wall_s']), default=float, indent=1), flush=True)


if __name__ == '__main__':
    main()
