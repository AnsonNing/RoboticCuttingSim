"""Full Native host of the P2-6 rubber recipe (task P6.4; ADR 0023 §23.2 decision 8; plan
``docs/reports/rl_ready/P2_REMAINING_PLAN.md`` §4.6, protocol ``rubber_rl_tests_v1`` RL-09).

A Kit host that runs the SAME rubber material as the P2-6 env (``RubberStripTaskFamily.build_life``: recipe
``rubber_generic_visco_v1``, slot, profile ``rubber_rl_env_v1``, contact law ``grid_node_penalty_coulomb_v3``) inside
the Isaac Sim 6.2 process, fed each control step by the clamp pose that PhysX MEASURES. The loop keeps the shape of the
TLMPM Native bridge (``isaac_sim_runner.LiveIsaacSimCutting.step``: command, PhysX step, measured pose, material
advance, surface update), without touching ``isaac_sim_runner.py`` or the frozen ``live_material.py``:

1. set the kinematic clamp target (world pose of the commanded grip pose, PhysX tensor ``set_kinematic_targets``);
2. step PhysX one control step (``robot_substeps`` PhysX steps; the Frankas are held at the shared safe park);
3. read the clamp's measured world pose from PhysX (float32 tensor -> float64, ``isaac_sim_runner.pose_from_wxyz``);
4. ``LiveRubberMaterial.advance(motions={0: grip_motion(measured pose)})`` on CUDA in the Kit process;
5. update the strip surface mesh (surface particles under the strip pose) through ``DynamicSurfaceBridge``;
6. capture the overview (and close-up) RGB-D every ``capture_every`` steps;
7. record ``rcs_rubber_record_v1`` plus the measured pose and the staged material inputs in float64.

Feedback is off: the clamp is kinematic and its wrench is reported, never applied to PhysX. The material frame is the
strip frame of the task config (table plane z = 0); ``StripPlacement`` maps it to the world (a translation onto the
workcell table top), and only the measured pose expressed in the strip frame enters the material.

Everything here except ``KitRubberScene`` is Kit-free (numpy, plus Warp through ``LiveRubberMaterial``), so the bridge
loop (``run_bridge_episode``) is tested on CPU with a fake PhysX adapter. The headless replay
(``replay_bridge_inputs``) feeds the recorded measured poses through the same ``advance_measured`` call, so Native and
replay material inputs are comparable bitwise. Kit imports live inside ``KitRubberScene`` methods only.
Material status: ``SIMULATOR_ONLY_UNCALIBRATED``; coupling ``kinematic_grip_ideal_tracking (no wrench feedback)``.
"""
import hashlib
import math
from pathlib import Path
import time

import numpy as np

HOST_VERSION = "native_rubber_host_v1"
RECORD_SCHEMA = "rcs_native_rubber_v1"
EPISODE = {
    "id": "native_rubber_episode_v1",
    "source": "plan docs/reports/rl_ready/P2_REMAINING_PLAN.md section 4.6 (run native_rubber_001)",
    "task_config": "rubber_strip_drag_v1",
    "slot": "L125_t4",
    "material": "RubberStripTaskFamily.build_material(): recipe nominal, friction_table = midpoint of the range",
    "traction_n": [0.0, 0.0, 0.0],
    "steps": 100,
    "hold_until_step": 25,          # steps 0-24: hold (gravity settle on the table)
    "ramp_steps": 25,               # steps 25-49: lift + yaw (smootherstep)
    "lift_m": 0.020,
    "yaw_rad": 0.3,                 # about +z through the grip origin
    "capture_every": 10,            # overview RGB-D at steps 0, 10, ..., 100
    "robot_substeps": 10,           # PhysX steps per 20 ms control step
    "schedule": "END pose of control step i (0-based): i < 25: grip origin, identity; 25 <= i < 50: tau = (i - 24) / "
                "25, s = 6 tau^5 - 15 tau^4 + 10 tau^3, p = o + (0, 0, 0.020 s), q = yaw(0.3 s); i >= 50: s = 1",
}
STAT_NAMES = tuple([f"marker{k}_{a}" for k in range(9) for a in "xyz"]
                   + [f"clamp_wrench_{c}" for c in ("fx", "fy", "fz", "tx", "ty", "tz")]
                   + [f"plane_wrench_{c}" for c in ("fx", "fy", "fz", "tx", "ty", "tz")]
                   + ["psi_eq_j", "kinetic_j", "gravity_energy_j", "contact_work_j", "jbar_min", "jbar_max",
                      "cfl_eff_max"])
assert len(STAT_NAMES) == 46
LABELS = {"calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
          "coupling": "kinematic_grip_ideal_tracking (no wrench feedback): kinematic PhysX clamp, measured pose -> "
                      "material, the material wrench is reported and never applied",
          "robot_feedback_coupling": "NOT_IMPLEMENTED_PHASE2",
          "robots": "both Frankas held at the shared safe park (PD targets + gravity compensation); no robot control",
          "self_contact": "NOT_IMPLEMENTED_V1 (detector surface_grid_v1)",
          "surface": "render bridge only (surface particles of the strip, DynamicSurfaceBridge); never physics",
          "cameras": "presentation-only RGB-D (overview, close-up); never an observation or policy input"}


# ----------------------------------------------------------------------------- pure geometry (float64)

def smootherstep(tau):
    tau = min(max(float(tau), 0.0), 1.0)
    return tau*tau*tau*(tau*(6.0*tau-15.0)+10.0)


def yaw_quaternion(angle):
    return np.array([0.0, 0.0, math.sin(0.5*angle), math.cos(0.5*angle)], np.float64)


def commanded_grip_pose(step, origin, episode=EPISODE):
    """Commanded END pose (p [3], q [4] XYZW, strip frame, float64) of control step ``step`` (0-based)."""
    step = int(step)
    if not 0 <= step < int(episode["steps"]):
        raise IndexError(f"step {step} outside [0, {episode['steps']})")
    h, r = int(episode["hold_until_step"]), int(episode["ramp_steps"])
    s = 0.0 if step < h else smootherstep((step-(h-1))/r)
    p = np.asarray(origin, np.float64)+np.array([0.0, 0.0, float(episode["lift_m"])*s])
    return p, yaw_quaternion(float(episode["yaw_rad"])*s)


def quat_to_matrix(q):
    """Rotation matrix (float64) of a unit quaternion XYZW (normalised first)."""
    x, y, z, w = np.asarray(q, np.float64)/np.linalg.norm(np.asarray(q, np.float64))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], np.float64)


def grip_motion(p, q, origin):
    """Group-0 motion of a grip pose (p, q) in the strip frame: the jaw maps X -> R(q) X + p - R(q) o (task config
    ``scene.grip``). Returns the ``LiveRubberMaterial.advance`` motion dict (float64)."""
    R = quat_to_matrix(q)
    b = np.asarray(p, np.float64)-R@np.asarray(origin, np.float64)
    return {"rotation": R, "translation": b, "active": True}


class StripPlacement:
    """World pose of the strip frame: a pure translation (strip x along world x, table plane z = 0 on the table
    top). Only translations are used so the strip-frame pose of a measured world pose is a float64 subtraction."""

    def __init__(self, translation):
        self.translation = np.asarray(translation, np.float64).reshape(3)
        if not np.isfinite(self.translation).all():
            raise ValueError("placement translation must be finite")

    def to_world(self, points):
        return np.asarray(points, np.float64)+self.translation

    def pose_to_world(self, p, q):
        return np.r_[np.asarray(p, np.float64)+self.translation, np.asarray(q, np.float64)]

    def pose_from_world(self, pose):
        pose = np.asarray(pose, np.float64)
        return pose[:3]-self.translation, pose[3:7].copy()

    def receipt(self):
        return {"kind": "translation_only", "world_from_strip_translation_m": self.translation.tolist(),
                "rotation": "identity (strip x = world x, strip z = world z)"}


def surface_mesh(scene):
    """(surface particle ids [V], triangles [T, 3] int32 into those ids) of the six lattice faces of a rubber strip
    scene (particle id = (i ny + j) nz + k). Winding: outward normals. The vertex set equals ``scene.surface``."""
    nx, ny, nz = (int(v) for v in scene.lattice)
    ids = np.flatnonzero(scene.surface)
    compact = np.full(nx*ny*nz, -1, np.int64)
    compact[ids] = np.arange(len(ids))

    def pid(i, j, k):
        return (np.asarray(i)*ny+np.asarray(j))*nz+np.asarray(k)

    tris = []

    def face(grid, outward):
        # grid [A, B] particle ids with a x b along +normal when outward is True
        a00, a10, a11, a01 = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
        t1 = np.stack((a00, a10, a11), -1).reshape(-1, 3)
        t2 = np.stack((a00, a11, a01), -1).reshape(-1, 3)
        t = np.concatenate((t1, t2))
        tris.append(t if outward else t[:, ::-1])

    J, K = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    face(pid(nx-1, J, K), True)                 # +x: (y, z), y x z = +x
    face(pid(0, J, K), False)
    K2, I2 = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    face(pid(I2, ny-1, K2), True)               # +y: (z, x), z x x = +y
    face(pid(I2, 0, K2), False)
    I3, J3 = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    face(pid(I3, J3, nz-1), True)               # +z: (x, y), x x y = +z
    face(pid(I3, J3, 0), False)
    faces = compact[np.concatenate(tris)]
    if (faces < 0).any():
        raise AssertionError("a face vertex is not a surface particle")
    return ids, faces.astype(np.int32)


# ----------------------------------------------------------------------------- material step (shared by Native/replay)

def positions64(live):
    s = live.solver
    return (s.x64 if getattr(s, "state_precision", None) == "f64_shadow_v1" else s.x).numpy().astype(np.float64)


def statistics_row(live, wrench):
    """The 46 RL-09 statistics after a control step (order ``STAT_NAMES``; strip frame, float64)."""
    d = live.last_diagnostics
    c = live.last_contact or {}
    v = live.last_visco or {}
    plane = np.asarray(c.get("plane_wrench", np.full((1, 6), np.nan)), np.float64).reshape(-1, 6)[0]
    row = np.concatenate((np.asarray(live.markers(), np.float64).reshape(-1),
                          np.asarray(wrench, np.float64).reshape(-1, 6)[0], plane,
                          [v.get("psi_eq", np.nan), d["kinetic_particle"], d["gravity"],
                           c.get("contact_work_material", np.nan), d["Jbar_min"], d["Jbar_max"], d["cfl_eff_max"]]))
    if row.shape != (len(STAT_NAMES),):
        raise AssertionError(f"statistics row has shape {row.shape}")
    return row


def advance_measured(live, p_local, q_local, origin, traction=None):
    """One material control step driven by a MEASURED grip pose in the strip frame (Native and replay use this same
    call). Returns (motion, wrench, statistics row [46], staged (q1 [4], b1 [3]) read back from the solver)."""
    motion = grip_motion(p_local, q_local, origin)
    wrench = live.advance(motions={0: motion}, traction=traction)
    s = live.solver
    staged = (np.asarray(s.group_q1.numpy()[0], np.float64).copy(), np.asarray(s.group_b1.numpy()[0],
                                                                              np.float64).copy())
    return motion, np.asarray(wrench, np.float64), statistics_row(live, wrench), staged


class EpisodeArrays:
    """Per-step float64 arrays of one bridge episode (Native or replay); ``save`` writes one npz."""
    KEYS = ("commanded_p", "commanded_q", "measured_world", "material_p", "material_q", "motion_rotation",
            "motion_translation", "staged_q1", "staged_b1", "statistics", "invalid_reason")

    def __init__(self, steps):
        self.steps = int(steps)
        self.commanded_p = np.full((steps, 3), np.nan)
        self.commanded_q = np.full((steps, 4), np.nan)
        self.measured_world = np.full((steps, 7), np.nan)
        self.material_p = np.full((steps, 3), np.nan)
        self.material_q = np.full((steps, 4), np.nan)
        self.motion_rotation = np.full((steps, 3, 3), np.nan)
        self.motion_translation = np.full((steps, 3), np.nan)
        self.staged_q1 = np.full((steps, 4), np.nan)
        self.staged_b1 = np.full((steps, 3), np.nan)
        self.statistics = np.full((steps, len(STAT_NAMES)), np.nan)
        self.invalid_reason = np.full(steps, -1, np.int64)
        self.completed = 0

    def put(self, i, *, p_local, q_local, motion, stats, staged, invalid, commanded=None, measured_world=None):
        self.material_p[i], self.material_q[i] = p_local, q_local
        self.motion_rotation[i], self.motion_translation[i] = motion["rotation"], motion["translation"]
        self.staged_q1[i], self.staged_b1[i] = staged
        self.statistics[i] = stats
        self.invalid_reason[i] = int(invalid)
        if commanded is not None:
            self.commanded_p[i], self.commanded_q[i] = commanded
        if measured_world is not None:
            self.measured_world[i] = measured_world
        self.completed = i+1

    def save(self, path):
        np.savez_compressed(path, stat_names=np.array(STAT_NAMES), completed=np.int64(self.completed),
                            **{k: getattr(self, k) for k in self.KEYS})

    @staticmethod
    def load(path):
        with np.load(path) as z:
            out = {k: z[k] for k in z.files}
        if tuple(out["stat_names"].tolist()) != STAT_NAMES:
            raise ValueError(f"{path}: statistic names differ from STAT_NAMES")
        return out


# ----------------------------------------------------------------------------- the bridge loop (Kit-agnostic)

def bridge_integrity(readback, expected_world, faces_expected=None, faces_readback=None):
    """Max |mesh vertex - surface particle under the strip pose| (m) and topology equality."""
    rb = np.asarray(readback, np.float64)
    ex = np.asarray(expected_world, np.float64)
    out = {"vertices": int(len(rb)), "expected_vertices": int(len(ex)),
           "max_abs_error_m": float(np.max(np.abs(rb-ex))) if rb.shape == ex.shape and len(ex) else math.inf}
    if faces_expected is not None and faces_readback is not None:
        out["topology_equal"] = bool(np.array_equal(np.asarray(faces_expected).ravel(),
                                                    np.asarray(faces_readback).ravel()))
    return out


def run_bridge_episode(live, physx, surface, placement, origin, *, episode=EPISODE, arrays=None, on_step=None,
                       capture=None):
    """The Native loop of the module docstring with a duck-typed PhysX adapter (``set_clamp_target(pose7 world)``,
    ``step()``, ``measured_clamp_pose() -> pose7 world float64``) and a surface sink (``update(vertices_world [V, 3],
    faces, t) -> dict``, ``readback() -> (points [V, 3], faces)``). ``capture(step_index)`` is called after the surface
    update at steps 0, capture_every, ... (index = number of completed control steps). Returns ``EpisodeArrays`` and
    the per-step bridge rows."""
    steps = int(episode["steps"])
    arrays = EpisodeArrays(steps) if arrays is None else arrays
    ids, faces = surface_mesh(live.scene)
    rows = []

    def update_surface(k, t):
        verts = placement.to_world(positions64(live)[ids])
        t0 = time.perf_counter()
        receipt = surface.update(verts, faces, t)
        rb_points, rb_faces = surface.readback()
        row = {"step": k, "t_s": t, **bridge_integrity(rb_points, verts, faces, rb_faces),
               "surface_update_wall_s": time.perf_counter()-t0,
               **{key: receipt[key] for key in ("surface_sha256", "topology_sha256") if key in receipt}}
        rows.append(row)
        return row

    update_surface(0, 0.0)
    if capture is not None:
        capture(0)
    for i in range(steps):
        t_step = time.perf_counter()
        p_cmd, q_cmd = commanded_grip_pose(i, origin, episode)
        physx.set_clamp_target(placement.pose_to_world(p_cmd, q_cmd))
        t0 = time.perf_counter()
        physx.step()
        physx_wall = time.perf_counter()-t0
        measured = np.asarray(physx.measured_clamp_pose(), np.float64)
        p_loc, q_loc = placement.pose_from_world(measured)
        t0 = time.perf_counter()
        motion, wrench, stats, staged = advance_measured(live, p_loc, q_loc, origin)
        material_wall = time.perf_counter()-t0
        arrays.put(i, p_local=p_loc, q_local=q_loc, motion=motion, stats=stats, staged=staged,
                   invalid=live.invalid_reason, commanded=(p_cmd, q_cmd), measured_world=measured)
        row = update_surface(i+1, (i+1)*live.control_dt)
        row.update(physx_wall_s=physx_wall, material_wall_s=material_wall,
                   command_vs_measured_m=float(np.linalg.norm(placement.pose_to_world(p_cmd, q_cmd)[:3]-measured[:3])))
        if capture is not None and (i+1) % int(episode["capture_every"]) == 0:
            capture(i+1)
        row["step_wall_s"] = time.perf_counter()-t_step
        if on_step is not None:
            on_step(i, row)
    return arrays, rows


def replay_bridge_inputs(live, material_p, material_q, origin, *, arrays=None):
    """Headless replay: feed recorded strip-frame measured poses [K, 3] / [K, 4] through ``advance_measured``."""
    material_p, material_q = np.asarray(material_p, np.float64), np.asarray(material_q, np.float64)
    arrays = EpisodeArrays(len(material_p)) if arrays is None else arrays
    for i in range(len(material_p)):
        motion, wrench, stats, staged = advance_measured(live, material_p[i], material_q[i], origin)
        arrays.put(i, p_local=material_p[i], q_local=material_q[i], motion=motion, stats=stats, staged=staged,
                   invalid=live.invalid_reason)
    return arrays


def strip_receipt(family, live, slot):
    """The receipt compared bitwise by RL-09 (a): family receipt, LiveRubberMaterial receipt and record header."""
    return {"family": family.receipt(), "slot": slot, "material": live.receipt(), "header": live.header(),
            "episode": EPISODE, "stat_names": list(STAT_NAMES), "host_version": HOST_VERSION}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- camera consistency (reported)

def marker_depth_consistency(markers_world, K, T_world_camera, depth, *, max_depth_m=10.0):
    """Project world markers into a depth image (OpenCV camera, ``distance_to_image_plane``) and report rendered
    depth minus marker optical depth at the nearest pixel. Reported, never scored (RL-09 e)."""
    pts = np.atleast_2d(np.asarray(markers_world, np.float64))
    T = np.asarray(T_world_camera, np.float64)
    cam = (np.linalg.inv(T)@np.c_[pts, np.ones(len(pts))].T)[:3].T
    pix = (np.asarray(K, np.float64)@cam.T).T
    uv = pix[:, :2]/pix[:, 2:3]
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = depth.shape
    rows = []
    for k, ((u, v), z) in enumerate(zip(uv, cam[:, 2])):
        ui, vi = int(round(u)), int(round(v))
        inside = bool(z > 0 and 0 <= ui < w and 0 <= vi < h)
        d = float(depth[vi, ui]) if inside else math.nan
        ok = inside and math.isfinite(d) and 0 < d < max_depth_m
        rows.append({"marker": k, "u": float(u), "v": float(v), "marker_depth_m": float(z),
                     "rendered_depth_m": d if ok else None, "residual_m": (d-float(z)) if ok else None})
    res = [r["residual_m"] for r in rows if r["residual_m"] is not None]
    return {"markers": rows, "visible": len(res), "residual_min_m": min(res) if res else None,
            "residual_max_m": max(res) if res else None,
            "residual_median_m": float(np.median(res)) if res else None,
            "note": "markers are cross-section centroids (inside the strip); the rendered depth sees the nearest "
                    "surface, so a negative residual of about the half thickness is expected"}


# ----------------------------------------------------------------------------- Kit scene (imports inside methods)

class KitRubberScene:
    """Static D0 workcell in the running Kit app: Frankas held at the shared safe park, the D455 rigid prims made
    kinematic (as the cutting runner does), the cutting specimen and blade deactivated, a kinematic clamp body
    without collider, the strip surface mesh (``DynamicSurfaceBridge`` at ``/World/RubberStrip``) and the
    presentation overview / close-up cameras with RGB + ``distance_to_image_plane``."""
    CLAMP_PATH = "/World/RubberClamp"
    VISUAL_PATH = "/World/RubberClampVisual"
    CLAMP_VISUALS = ("measured_pose_v1", "rigid_child_v0")
    STRIP_PATH = "/World/RubberStrip"

    def __init__(self, app, output, *, seed=42, control_dt=0.02, robot_substeps=10, strip_xy=(-0.0615, -0.13),
                 render_warmup=5, clamp_visual="measured_pose_v1"):
        if clamp_visual not in self.CLAMP_VISUALS:
            raise ValueError(f"clamp_visual {clamp_visual!r} not in {self.CLAMP_VISUALS}")
        self.clamp_visual = clamp_visual
        self.app, self.output = app, Path(output)
        self.seed, self.control_dt, self.robot_substeps = int(seed), float(control_dt), int(robot_substeps)
        self.strip_xy = tuple(float(v) for v in strip_xy)
        self.render_warmup = int(render_warmup)
        self.receipt = {}

    # -- stage ---------------------------------------------------------------------------------------------------
    def build(self, jaw_box_local, origin):
        from argparse import Namespace
        import omni.timeline
        import omni.usd
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from .isaac_sim_runner import wait_for_workcell_stage
        from .workcell import PARK_JOINTS, SECONDARY_PARK_JOINTS, export
        self.SimulationManager = SimulationManager
        omni.timeline.get_timeline_interface().stop()
        context = omni.usd.get_context()
        scene_dir = self.output/"initial_workcell"
        scene_dir.mkdir()
        export(Namespace(variant="D0", family="dome", seed=self.seed, spacing=0.002, asset_root=None), scene_dir)
        source = scene_dir/"workcell.usda"
        if not context.open_stage(str(source)):
            raise RuntimeError("Isaac could not open the workcell")
        wait_for_workcell_stage(self.app, context, self.output/"stage_load.json")
        stage = self.stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError("the workcell must be meter / Z-up")
        deactivated = []
        for name in ("ContinuousSpecimen", "KinematicBlade", "MPMSamples_debug"):
            prim = stage.GetPrimAtPath("/World/"+name)
            if prim:
                prim.SetActive(False)
                deactivated.append("/World/"+name)
        camera_rigid = []
        for prim in stage.Traverse():
            if any(str(prim.GetPath()).startswith(f"/World/D455_{i}/") for i in (0, 1)) and \
                    prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                camera_rigid.append(str(prim.GetPath()))
        for index in (0, 1):
            root = stage.GetPrimAtPath(f"/World/Franka_{index}")
            if not root:
                raise RuntimeError(f"Missing Franka_{index}")
            root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
            root.GetVariantSets().GetVariantSet("Colliders").SetVariantSelection("convex_hulls")
            PhysxSchema.PhysxArticulationAPI.Apply(root)
        for name in ("Table", "TrayFloor", "TrayLeft", "TrayRight", "TrayFront", "TrayBack"):
            prim = stage.GetPrimAtPath("/World/"+name)
            if prim:
                UsdPhysics.CollisionAPI.Apply(prim)
        table = stage.GetPrimAtPath("/World/Table")
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(table)
        top = float(bbox.ComputeAlignedRange().GetMax()[2])
        self.placement = StripPlacement((self.strip_xy[0], self.strip_xy[1], top))
        # kinematic clamp (no collider: kinematic vs static never generates contact, and nothing may touch it)
        lo, hi = (np.asarray(v, np.float64) for v in jaw_box_local)
        clamp = UsdGeom.Xform.Define(stage, self.CLAMP_PATH)
        p0 = self.placement.pose_to_world(origin, [0., 0., 0., 1.])
        xf = UsdGeom.Xformable(clamp)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, p0[:3])))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(clamp.GetPrim()).CreateKinematicEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(clamp.GetPrim())
        mass.CreateMassAttr(0.2)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
        # The rendered jaw is a separate non-physics prim posed from the MEASURED clamp pose after every PhysX step
        # (clamp_visual "measured_pose_v1"). PhysX does not write kinematic-body poses back to USD, so a jaw authored
        # as a child of the kinematic body ("rigid_child_v0", used by native_rubber_001) renders at its reset pose.
        visual_root = self.CLAMP_PATH if self.clamp_visual == "rigid_child_v0" else self.VISUAL_PATH
        if self.clamp_visual != "rigid_child_v0":
            vis = UsdGeom.Xform.Define(stage, self.VISUAL_PATH)
            vxf = UsdGeom.Xformable(vis)
            vxf.ClearXformOpOrder()
            self._visual_ops = (vxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                                vxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble))
            self._set_visual(p0)
        jaw = UsdGeom.Cube.Define(stage, visual_root+"/Jaw")
        jaw.CreateSizeAttr(1.0)
        centre = 0.5*(lo+hi)-np.asarray(origin, np.float64)
        jxf = UsdGeom.Xformable(jaw)
        jxf.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
        jxf.AddScaleOp().Set(Gf.Vec3f(*map(float, (hi-lo)+np.array([0.004, 0.006, 0.004]))))
        jaw.CreateDisplayColorAttr([Gf.Vec3f(.55, .58, .62)])
        for path, color in ((visual_root+"/Jaw", (.55, .58, .62)), ("/World/Franka_0", (.8, .82, .85)),
                            ("/World/Franka_1", (.8, .82, .85)), ("/World/D455_0", (.2, .25, .3)),
                            ("/World/D455_1", (.2, .25, .3))):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                continue
            material = UsdShade.Material.Define(stage, "/World/LiveMaterials/"+path.strip("/").replace("/", "_"))
            shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        # physics
        SimulationManager.setup_simulation(dt=self.control_dt/self.robot_substeps, device="cpu")
        if not SimulationManager.switch_physics_engine("physx"):
            raise RuntimeError("Native Isaac PhysX engine is unavailable")
        SimulationManager.enable_gpu_dynamics(False)
        SimulationManager.enable_fabric(False)
        self.arms = Articulation(["/World/Franka_0", "/World/Franka_1"])
        self.clamp = RigidPrim(self.CLAMP_PATH)
        SimulationManager.initialize_physics()
        sim_view = getattr(SimulationManager, "_physics_sim_view__warp", None)
        if sim_view is not None and hasattr(sim_view, "initialize_kinematic_bodies"):
            sim_view.initialize_kinematic_bodies()     # kinematic bodies report their transforms (tensor API)
        if not self.arms.is_physics_tensor_entity_valid() or not self.clamp.is_physics_tensor_entity_valid():
            raise RuntimeError("PhysX articulation / clamp tensor views did not initialize")
        expected = [f"panda_joint{i}" for i in range(1, 8)]+["panda_finger_joint1", "panda_finger_joint2"]
        if self.arms.dof_names != expected:
            raise RuntimeError(f"Unexpected PhysX DOF order: {self.arms.dof_names}")
        self.arms.set_dof_gains([1000.]*4+[600.]*3+[350., 350.], [80.]*4+[40.]*3+[10., 10.])
        self.arms.set_dof_max_efforts([87.]*4+[12.]*3+[20., 20.])
        self.arms.set_solver_iteration_counts(position_counts=16, velocity_counts=4)
        self.park_q = np.array([[*PARK_JOINTS, .04, .04], [*SECONDARY_PARK_JOINTS, .04, .04]], np.float32)
        self.arms.set_dof_positions(self.park_q)
        self.arms.set_dof_velocities(np.zeros_like(self.park_q))
        self.arms.set_dof_position_targets(self.park_q)
        self.arms.set_dof_velocity_targets(np.zeros_like(self.park_q))
        self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
        import omni.physx
        self.physx = omni.physx.get_physx_interface()
        self.time_origin = SimulationManager.get_simulation_time()
        self.step_origin = SimulationManager.get_num_physics_steps()
        self.steps_done = 0
        self.receipt.update(
            source_scene=str(source), deactivated_prims=deactivated, stationary_camera_kinematic_rigid_prims=camera_rigid,
            table_top_z_m=top, placement=self.placement.receipt(), clamp_prim=self.CLAMP_PATH,
            clamp_visual={"mode": self.clamp_visual, "prim": visual_root+"/Jaw",
                          "scope": "rendering only; the material input is the PhysX tensor pose"},
            clamp="kinematic rigid body, no collider, mass 0.2 kg (unused); targets via the PhysX tensor "
                  "RigidBodyView.set_kinematic_targets; measured pose via RigidPrim.get_world_poses (tensor)",
            physics={"engine": "PhysX (Isaac Sim 6.2, CPU)", "dt_s": self.control_dt/self.robot_substeps,
                     "robot_substeps": self.robot_substeps, "gpu_dynamics": False, "fabric": False},
            robots={"park_q": self.park_q.tolist(), "hold": "PD position targets + gravity compensation efforts "
                    "every PhysX step (gains of isaac_sim_runner)", "control": "none"})
        return self.placement

    def set_clamp_target(self, pose7):
        import warp as wp
        view = self.clamp._physics_rigid_body_view
        device = view.get_transforms().device
        data = wp.array(np.asarray(pose7, np.float32).reshape(1, 7), dtype=wp.float32, device=device)
        idx = wp.array(np.array([0], np.uint32), dtype=wp.uint32, device=device)
        view.set_kinematic_targets(data, idx)
        self._last_target = np.asarray(pose7, np.float64)

    def step(self):
        manager = self.SimulationManager
        for _ in range(self.robot_substeps):
            self.arms.set_dof_efforts(self.arms.get_dof_gravity_compensation_forces())
            manager.step()
        self.steps_done += 1
        rel = manager.get_simulation_time()-self.time_origin
        if abs(rel-self.steps_done*self.control_dt) > 1e-6:
            raise RuntimeError(f"Native physics clock drift: {rel} vs {self.steps_done*self.control_dt}")
        self.physx.update_transformations(False, True, True)
        if self.clamp_visual != "rigid_child_v0":
            self._set_visual(self.measured_clamp_pose())

    def _set_visual(self, pose7):
        from pxr import Gf
        pose7 = np.asarray(pose7, np.float64)
        self._visual_ops[0].Set(Gf.Vec3d(*map(float, pose7[:3])))
        self._visual_ops[1].Set(Gf.Quatd(float(pose7[6]), Gf.Vec3d(*map(float, pose7[3:6]))))

    def measured_clamp_pose(self):
        from .isaac_sim_runner import numpy_data, pose_from_wxyz
        p, q = self.clamp.get_world_poses()
        return pose_from_wxyz(numpy_data(p)[0], numpy_data(q)[0])

    def arm_state(self):
        from .isaac_sim_runner import numpy_data
        q = numpy_data(self.arms.get_dof_positions())
        return {"q": q.tolist(), "max_abs_dev_from_park_rad": float(np.abs(q[:, :7]-self.park_q[:, :7]).max())}

    # -- surface ---------------------------------------------------------------------------------------------------
    def make_surface(self):
        from .usd_surface_bridge import DynamicSurfaceBridge
        from pxr import Gf, Sdf, UsdShade
        bridge = DynamicSurfaceBridge(self.stage, self.STRIP_PATH)
        bridge.mesh.CreateDisplayColorAttr([Gf.Vec3f(.15, .15, .17)])
        material = UsdShade.Material.Define(self.stage, "/World/LiveMaterials/RubberStrip")
        shader = UsdShade.Shader.Define(self.stage, str(material.GetPath())+"/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12, .12, .14))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(bridge.mesh.GetPrim()).Bind(material)
        return _UsdSurfaceSink(bridge)

    # -- cameras ---------------------------------------------------------------------------------------------------
    def make_cameras(self, shim):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from .presentation_camera import author_overview_camera, closeup_camera_spec, overview_camera_spec
        from .usd_surface_bridge import usd_camera_calibration
        self.cameras, self.camera_specs = {}, {}
        for kind, make_spec, path in (("overview", overview_camera_spec, "/World/PresentationCamera/Overview"),
                                      ("closeup", closeup_camera_spec, "/World/PresentationCamera/Closeup")):
            spec = make_spec(shim)
            spec["prim_path"] = path
            spec["modalities"] = ["rgb", "distance_to_image_plane"]
            author_overview_camera(self.stage, spec)
            sensor = CameraSensor(RtxCamera(path, tick_rate=0.), resolution=(spec["height"], spec["width"]),
                                  annotators=["rgb", "distance_to_image_plane"])
            spec["authored_usd_readback"] = usd_camera_calibration(self.stage, path, spec["width"], spec["height"])
            self.cameras[kind], self.camera_specs[kind] = sensor, spec
        self.receipt["cameras"] = {k: {kk: vv for kk, vv in v.items() if kk != "authored_usd_readback"}
                                   for k, v in self.camera_specs.items()}
        return self.camera_specs

    def capture(self):
        import omni.replicator.core as rep
        manager = self.SimulationManager
        before = (manager.get_simulation_time(), manager.get_num_physics_steps())
        clamp_before = self.measured_clamp_pose()
        latest = {}
        for k in range(self.render_warmup+30):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.)
            latest = {name: {a: sensor.get_data(a)[0] for a in ("rgb", "distance_to_image_plane")}
                      for name, sensor in self.cameras.items()}
            if k >= self.render_warmup-1 and all(v is not None for d in latest.values() for v in d.values()):
                break
        else:
            raise TimeoutError("RTX cameras did not produce arrays")
        after = (manager.get_simulation_time(), manager.get_num_physics_steps())
        if before != after or not np.array_equal(clamp_before, self.measured_clamp_pose()):
            raise RuntimeError(f"rendering advanced physics: {before} -> {after}")
        return {name: {a: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for a, v in d.items()}
                for name, d in latest.items()}


class _UsdSurfaceSink:
    def __init__(self, bridge):
        self.bridge = bridge

    def update(self, vertices, faces, t):
        return self.bridge.update(vertices, faces, t)

    def readback(self):
        mesh = self.bridge.mesh
        pts = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
        return pts, idx


class StripCameraShim:
    """The minimal task view ``presentation_camera.overview_camera_spec`` / ``closeup_camera_spec`` need, built from
    the strip: workpiece frame at the strip's bottom centre, base height = thickness, reference edge = grip pose."""

    def __init__(self, placement, length, width, thickness, jaw, origin):
        from types import SimpleNamespace
        T = np.eye(4)
        T[:3, 3] = placement.to_world([0.5*(length-jaw), 0.5*width, 0.0])
        self.T_world_workpiece = T
        self.world_from_workpiece_xyzw = np.r_[T[:3, 3], 0., 0., 0., 1.]
        self.geometry = SimpleNamespace(length_m=float(length+jaw), width_m=float(width), base_height_m=float(thickness))
        self._edge = placement.pose_to_world(origin, [0., 0., 0., 1.])

    def reference_edge_pose_world(self, t):
        return self._edge.copy()


__all__ = ["EPISODE", "STAT_NAMES", "LABELS", "HOST_VERSION", "commanded_grip_pose", "grip_motion", "quat_to_matrix",
           "StripPlacement", "surface_mesh", "statistics_row", "advance_measured", "EpisodeArrays",
           "run_bridge_episode", "replay_bridge_inputs", "bridge_integrity", "strip_receipt",
           "marker_depth_consistency", "KitRubberScene", "StripCameraShim"]
