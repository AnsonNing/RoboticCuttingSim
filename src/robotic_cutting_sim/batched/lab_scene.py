"""Workload B scene (ADR 0022 §3): N Isaac Lab / Newton worlds at the origin, per-env Frankas and tool.

Layout (ADR 0022 §3.1):

* ``{ENV_REGEX_NS}/Franka_0`` (cutting arm) and ``{ENV_REGEX_NS}/Franka_1`` (parked at the shared normal safe
  park) in world ``e``; the same USD, variants, gravity compensation and actuator gains as the single Lab env
  (``isaac_lab_env.make_lab_cfg``, physical-grasp profile, ported here because that module is frozen).
* B1 ``rigid_mount_v1``: the ToolSpec body is a separate rigid body ``.../panda_hand/CuttingTool`` inside the
  Franka 0 subtree, welded by a ``UsdPhysics.FixedJoint`` (``.../CuttingTool/HandMount``) at
  ``ToolSpec.hand_from_tool``. Finger-tool shape pairs are filtered (``physics:filteredPairs``). Reduced fidelity.
* B2 ``physical_grasp_v1``: a free ToolSpec body at ``{ENV_REGEX_NS}/Tool`` authored by
  :func:`author_batched_tool`, a path-parameterized port of the frozen ``newton_tool.author_newton_tool``.
* The static workcell (table, board, fixtures) is shared (world -1) under ``/World/Workcell``.

``env_spacing`` must be 0 (every world at the origin, identical float32 coordinates, ADR 0022 §3.3).

Two builders use the same USD authoring:

* :func:`make_batched_lab_cfg` -- the Isaac Lab ``DirectRLEnvCfg`` (needs ``isaaclab``; imported lazily);
* :func:`build_newton_rigid_scene` -- a pure Newton/MuJoCo-Warp model of the same rigid scene for the CPU
  C tests (``.venv``, no Torch, no Isaac Lab). Its drive gains are set on the builder the way
  ``robot.DualFrankaController`` sets them; it is a test model, not the Lab runtime.

New UNSUPPORTED codes of ADR 0022 §10 are raised through :class:`UnsupportedLabBatchConfiguration`, a subclass
of ``errors.UnsupportedBatchConfiguration`` that also accepts :data:`LAB_UNSUPPORTED_CODES` until those codes are
merged into ``errors.CODES`` (the file is owned by the material work package).
"""
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np

from .errors import CODES as _MATERIAL_CODES
from .errors import UnsupportedBatchConfiguration

LAB_SCENE_VERSION = "lab_batched_scene_v1"
TOOL_MOUNTS = ("rigid_mount_v1", "physical_grasp_v1")
MATERIALS = (None, "tlmpm_v7_2")
ENV_TEMPLATE = "/World/envs/env_{}"
ENV_REGEX_NS = "{ENV_REGEX_NS}"
WORKCELL_ROOT = "/World/Workcell"
TOOL_NAME = "Tool"                       # B2 free tool, sibling of the Frankas in world e
RIGID_TOOL_NAME = "CuttingTool"          # B1 welded tool body, child prim of panda_hand
RIGID_MOUNT_JOINT = "HandMount"
FRANKA_BASE_POSES = (((-.46, .12, -.01), (0., 0., 0., 1.)), ((.46, .12, -.01), (0., 0., 1., 0.)))  # XYZ, XYZW
REDUCED_FIDELITY = {"rigid_mount_v1": ("rigid_tool_mount_no_grasp", "finger_opening_fixed_nominal"),
                    "physical_grasp_v1": ()}
LISTED_DIFFERENCES = {"rigid_engine": "newton_mjwarp (Native reference: physx_cpu)"}

# Actuator gains of the single Lab env, physical-grasp profile (isaac_lab_env.make_lab_cfg). Finger gains of
# Franka 0 come from ToolGraspConfig; Franka 1 keeps the legacy 350/10/20 finger drive.
ARM_ACTUATORS = (("shoulder", "panda_joint[1-4]", 1000., 80., 87., .2),
                 ("forearm", "panda_joint[5-7]", 600., 40., 12., .07))
PARKED_FINGER_ACTUATOR = (350., 10., 20., .1)
FINGER_ARMATURE = .1

# ADR 0022 §10 codes not (yet) in errors.CODES.
LAB_UNSUPPORTED_CODES = (
    "UNSUPPORTED_ENV_SPACING",
    "UNSUPPORTED_NONUNIFORM_RIGID_SCENE",
    "UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET",
    "UNSUPPORTED_RIGID_CPU",
    "UNSUPPORTED_ACTION_SPEC",
    "UNSUPPORTED_ACTUATOR_OVERRIDE",
    "UNSUPPORTED_TOOL_MOUNT",                    # anything but rigid_mount_v1 / physical_grasp_v1
    "UNSUPPORTED_MATERIAL_COUPLING",             # material coupling not available for the requested configuration
)


class UnsupportedLabBatchConfiguration(UnsupportedBatchConfiguration):
    """``UnsupportedBatchConfiguration`` that also knows the ADR 0022 §10 codes (never a fallback)."""

    def __init__(self, code, detail=""):
        if code not in _MATERIAL_CODES and code not in LAB_UNSUPPORTED_CODES:
            raise ValueError(f"unknown UNSUPPORTED code {code!r}")
        self.code = code
        self.detail = str(detail)
        message = code if not detail else f"{code}: {detail}"
        if not message.startswith("UNSUPPORTED"):
            raise AssertionError("UNSUPPORTED messages start with UNSUPPORTED")
        ValueError.__init__(self, message)


def unsupported(code, detail=""):
    if code in _MATERIAL_CODES:
        return UnsupportedBatchConfiguration(code, detail)
    return UnsupportedLabBatchConfiguration(code, detail)


# ----------------------------------------------------------------------------- options and uniformity

def _grasp_default():
    from ..physical_tool import ToolGraspConfig
    return ToolGraspConfig()


@dataclass(frozen=True)
class BatchedLabOptions:
    """Workload B configuration; everything outside the supported set raises an UNSUPPORTED code."""

    num_envs: int = 1
    tool_mount: str = "rigid_mount_v1"
    device: str = "cuda:0"
    env_spacing: float = 0.
    action_spec: str = "edge_delta_v1"
    use_mujoco_cpu: bool = False
    cpu_test_mode: bool = False          # Warp-CPU MJWarp, tests only (as the material's parity_mode)
    use_cuda_graph: bool | None = None   # None: True on CUDA, False on CPU (no graphs on the Warp CPU device)
    deterministic_mode: str = "not_guaranteed"
    tool_init_mode: str = "pregrasped"
    grip_contact_profile: str = "legacy_point_contact"   # pad_contact.POINT_PROFILE
    grasp: object = field(default_factory=_grasp_default)
    ik_iterations: int = 48
    ik_rotation_tolerance_rad: float = .01
    ik_position_tolerance_m: float = .001
    velocity_feedforward: bool = True
    feedback: bool = True
    material: str | None = None          # None: robot-only stepping; "tlmpm_v7_2": BatchedCuttingMaterial(isaac_lab_newton)
    material_dt_s: float = 1e-4
    max_consecutive_ik_failures: int | None = None   # live_env.LiveCuttingEnv semantics; None never terminates
    initial_joint_positions: object = None            # G4 harness only: uniform Franka 0 arm HOME override (7 rad)
    cameras: object = None
    sensors: object = None
    recorders: object = None
    actuator_overrides: object = None
    strict: bool = True
    seed: int = 42

    def validate(self):
        from ..pad_contact import POINT_PROFILE
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, (int, np.integer)) or self.num_envs < 1:
            raise unsupported("UNSUPPORTED_EMPTY_BATCH", f"num_envs={self.num_envs!r}")
        if self.env_spacing != 0:
            raise unsupported("UNSUPPORTED_ENV_SPACING", f"env_spacing={self.env_spacing!r}; worlds sit at the origin")
        if self.tool_mount not in TOOL_MOUNTS:
            raise unsupported("UNSUPPORTED_TOOL_MOUNT", f"{self.tool_mount!r}; supported {TOOL_MOUNTS}")
        if self.action_spec != "edge_delta_v1":
            raise unsupported("UNSUPPORTED_ACTION_SPEC", f"{self.action_spec!r}; only edge_delta_v1")
        if self.use_mujoco_cpu:
            raise unsupported("UNSUPPORTED_RIGID_CPU", "use_mujoco_cpu: the MuJoCo C runtime is not mask-aware")
        if str(self.device).startswith("cpu") and not self.cpu_test_mode:
            raise unsupported("UNSUPPORTED_RIGID_CPU", "Warp-CPU MJWarp is a test runtime (cpu_test_mode=True)")
        if self.cameras or self.sensors:
            raise unsupported("UNSUPPORTED_SENSOR", "no cameras or sensors in the kit-less batched path")
        if self.recorders:
            raise unsupported("UNSUPPORTED_RECORDER", "no raw 2 ms traces, cut events or surfaces")
        if self.actuator_overrides:
            raise unsupported("UNSUPPORTED_ACTUATOR_OVERRIDE", "gains/drive/mimic must equal the single Lab env")
        if self.tool_init_mode != "pregrasped":
            raise unsupported("UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET", f"tool_init_mode={self.tool_init_mode!r}")
        if self.grip_contact_profile != POINT_PROFILE:
            raise unsupported("UNSUPPORTED_ACTUATOR_OVERRIDE", f"grip contact profile {self.grip_contact_profile!r}")
        if self.material not in MATERIALS:
            raise unsupported("UNSUPPORTED_SOLVER" if isinstance(self.material, str) and self.material.startswith("tlmpm")
                              else "UNSUPPORTED_MATERIAL_COUPLING",
                              f"material={self.material!r}; supported {MATERIALS} (None = robot-only)")
        if self.max_consecutive_ik_failures is not None and int(self.max_consecutive_ik_failures) < 1:
            raise ValueError("max_consecutive_ik_failures must be positive or None")
        if self.material is not None and not self.material_dt_s > 0:
            raise ValueError("material_dt_s must be positive")
        if self.deterministic_mode not in ("not_guaranteed", "run_to_run", "gpu_to_gpu"):
            raise ValueError("deterministic_mode is not_guaranteed, run_to_run or gpu_to_gpu")
        if self.ik_iterations < 1 or not self.ik_rotation_tolerance_rad > 0 or not self.ik_position_tolerance_m > 0:
            raise ValueError("IK iterations and tolerances must be positive")
        return self

    @property
    def graph(self):
        return (not str(self.device).startswith("cpu")) if self.use_cuda_graph is None else bool(self.use_cuda_graph)

    def fidelity(self):
        reasons = list(REDUCED_FIDELITY[self.tool_mount])
        if self.material is None:
            reasons.append("robot_only_no_material_coupling")
        return {"reduced_fidelity": bool(reasons), "reasons": reasons, "listed_differences": dict(LISTED_DIFFERENCES),
                "grasp_restore": "grasp_restored_from_settled_snapshot" if self.tool_mount == "physical_grasp_v1"
                else None}


def rigid_scene_key(task):
    """Everything of a task that shapes the shared rigid scene or the per-world HOME (ADR 0022 §3.1)."""
    home = getattr(task, "home", None) or {}
    boxes = tuple((b.name, tuple(map(float, b.center_m)), tuple(map(float, b.half_extents_m)), b.role)
                  for b in task.fixture_boxes_local)
    return {"table_top_world_z_m": float(task.table_top_world_z_m), "table_box": table_box(task),
            "fixture_boxes_local": boxes,
            "world_from_workpiece_xyzw": tuple(map(float, task.world_from_workpiece_xyzw)),
            "hand_from_edge_xyz_xyzw": tuple(map(float, task.hand_from_edge_xyz_xyzw)),
            "home_joint_positions": tuple(map(float, home.get("joint_positions", ()))),
            "home_finger_position_m": float(home.get("finger_position_m", np.nan)) if home else None}


def table_box(task):
    """(center, half extents) of the static table, as the Native runtime places it (``isaac_sim_runner.py:956-966``):
    the task's ``table_box_world`` when it declares one (WP-12 flat-board workcell), else the legacy 1.8 x 1.2 x
    0.05 m table under ``table_top_world_z_m`` (``isaac_lab_env.make_lab_cfg``)."""
    box = getattr(task, "table_box_world", None)
    if box is not None:
        center, half = (tuple(map(float, v)) for v in box)
        return center, half
    return (0., 0., float(task.table_top_world_z_m)-.025), (.9, .6, .025)


def check_uniform_rigid_scene(tasks):
    """All envs of a batch share one rigid scene; per-env tool profiles and rigid differences are refused."""
    tasks = list(tasks)
    if not tasks:
        raise unsupported("UNSUPPORTED_EMPTY_BATCH", "no tasks")
    profiles = {getattr(t, "tool_profile_id", None) for t in tasks}
    if len(profiles) != 1:
        raise unsupported("UNSUPPORTED_PER_ENV_TOOL_PROFILE", f"tool profiles {sorted(map(str, profiles))}")
    reference = rigid_scene_key(tasks[0])
    for e, task in enumerate(tasks[1:], 1):
        key = rigid_scene_key(task)
        differs = [name for name in reference if not _same(reference[name], key[name])]
        if differs:
            raise unsupported("UNSUPPORTED_NONUNIFORM_RIGID_SCENE", f"env {e} differs from env 0 in {differs}")
    return reference


def _same(a, b):
    try:
        return bool(np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)))
    except Exception:
        return a == b


def home_joint_positions(task, tool_mount, grasp, initial_joint_positions=None):
    """(Franka 0 [9], Franka 1 [9]) initial joint positions: declared task HOME and the shared safe park."""
    from ..workcell import PARK_JOINTS, SECONDARY_PARK_JOINTS
    home = getattr(task, "home", None) or {}
    arm = np.asarray(home.get("joint_positions", PARK_JOINTS) if initial_joint_positions is None
                     else initial_joint_positions, float)
    if arm.shape != (7,) or not np.isfinite(arm).all():
        raise ValueError("Task HOME must declare seven finite joint positions")
    finger = float(home.get("finger_position_m", np.nan)) if home else np.nan
    if not np.isfinite(finger):
        finger = grasp.initial_finger_position_m
    if tool_mount == "physical_grasp_v1":
        finger = grasp.initial_finger_position_m if not home else finger
    q0 = np.r_[arm, finger, finger]
    q1 = np.r_[np.asarray(SECONDARY_PARK_JOINTS, float), .04, .04]
    return q0, q1


def finger_hold_target(task, tool_mount, grasp):
    """B1 holds the nominal grasp opening; B2 holds ``grasp.closed_target_m`` (ADR 0022 §4.3, §5.1)."""
    if tool_mount == "physical_grasp_v1":
        return float(grasp.closed_target_m)
    return float(home_joint_positions(task, tool_mount, grasp)[0][7])


def finger_actuator(arm, tool_mount, grasp):
    """(stiffness, damping, effort limit, armature) of the finger drive of Franka ``arm``."""
    if arm == 0:
        return (float(grasp.stiffness_n_m), float(grasp.damping_n_s_m), float(grasp.max_finger_force_n), FINGER_ARMATURE)
    return PARKED_FINGER_ACTUATOR


# ----------------------------------------------------------------------------- USD authoring

def _gf_pose(pose):
    from pxr import Gf
    pose = np.asarray(pose, float)
    return Gf.Vec3d(*pose[:3]), Gf.Quatf(float(pose[6]), Gf.Vec3f(*pose[3:6]))


def _make_grip_material(stage, path, grasp, patch=None):
    """Port of the frozen ``author_newton_tool.make_material``."""
    from pxr import Sdf, UsdPhysics, UsdShade
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(grasp.friction))
    physics.CreateDynamicFrictionAttr(float(grasp.friction))
    physics.CreateRestitutionAttr(0.)
    if patch:
        for name in ("mjc:torsionalfriction", "newton:torsionalFriction"):
            material.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Double).Set(patch["torsional_friction_length_m"])
    return material


def _author_tool_body(stage, path, spec, local_pose):
    """Rigid body prim with ToolSpec mass properties (port of ``author_newton_tool``)."""
    from pxr import Gf, UsdGeom, UsdPhysics
    body = UsdGeom.Xform.Define(stage, path)
    translation, orientation = _gf_pose(local_pose)
    body.AddTranslateOp().Set(translation)
    body.AddOrientOp().Set(orientation)
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    rigid.CreateVelocityAttr(Gf.Vec3f(0))
    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(float(spec.mass_kg))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*spec.com_tool_m))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec.principal_moments_kg_m2))
    q = spec.principal_axes_xyzw
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    return body


def _author_tool_shapes(stage, path, spec, material):
    """Handle as an exact box, other shapes as convex hulls; the frozen solref/solimp; physics binding."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    for shape in spec.shapes:
        if shape.role == "handle":
            vertices = np.asarray(shape.vertices_tool_m)
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            if len(vertices) != 8 or not np.all(np.isclose(vertices, lo) | np.isclose(vertices, hi)):
                raise ValueError("Native box handle requires the exact shared rectangular geometry")
            shape_prim = UsdGeom.Cube.Define(stage, path+"/"+shape.name)
            shape_prim.CreateSizeAttr(2.)
            shape_prim.AddTranslateOp().Set(Gf.Vec3d(*((lo+hi)/2)))
            shape_prim.AddScaleOp().Set(Gf.Vec3f(*((hi-lo)/2)))
        else:
            shape_prim = UsdGeom.Mesh.Define(stage, path+"/"+shape.name)
            shape_prim.CreatePointsAttr(np.asarray(shape.vertices_tool_m).tolist())
            faces = np.asarray(shape.triangles)
            shape_prim.CreateFaceVertexCountsAttr([3]*len(faces))
            shape_prim.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            shape_prim.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.MeshCollisionAPI.Apply(shape_prim.GetPrim()).CreateApproximationAttr("convexHull")
        shape_prim.CreateDisplayColorAttr([Gf.Vec3f(*((.12, .13, .16) if shape.role == "handle" else (.64, .68, .73)))])
        UsdPhysics.CollisionAPI.Apply(shape_prim.GetPrim())
        shape_prim.GetPrim().AddAppliedSchema("MjcCollisionAPI")
        shape_prim.GetPrim().CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([.01, 1.])
        shape_prim.GetPrim().CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([.99, .995, .01, .5, 2.])
        UsdShade.MaterialBindingAPI.Apply(shape_prim.GetPrim()).Bind(material, materialPurpose="physics")


def _author_edge_xform(stage, path, spec):
    from pxr import UsdGeom
    edge = UsdGeom.Xform.Define(stage, path+"/CuttingEdge")
    translation, orientation = _gf_pose(spec.tool_from_edge_xyz_xyzw)
    edge.AddTranslateOp().Set(translation)
    edge.AddOrientOp().Set(orientation)
    return edge


def finger_body_paths(stage, franka_root):
    """USD paths of the Franka's left and right finger rigid bodies (sorted left, right)."""
    from pxr import Usd, UsdPhysics
    found = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(franka_root)):
        if prim.GetName() in ("panda_leftfinger", "panda_rightfinger") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found[prim.GetName()] = str(prim.GetPath())
    if set(found) != {"panda_leftfinger", "panda_rightfinger"}:
        raise RuntimeError(f"Franka at {franka_root} has no rigid finger bodies: {found}")
    return found["panda_leftfinger"], found["panda_rightfinger"]


def hand_body_path(stage, franka_root):
    from pxr import Usd, UsdPhysics
    prim = stage.GetPrimAtPath(franka_root)
    if not prim:
        raise RuntimeError(f"No Franka prim at {franka_root}")
    hands = [str(p.GetPath()) for p in Usd.PrimRange(prim)
             if p.GetName() == "panda_hand" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(hands) != 1:
        raise RuntimeError(f"Expected one rigid panda_hand under {franka_root}, got {hands}")
    return hands[0]


def author_batched_tool(stage, root, spec, world_pose, grasp, finger_root, *, contact_profile=None):
    """``newton_tool.author_newton_tool`` with the tool path ``root`` and the Franka path ``finger_root``.

    The frozen function hard-codes ``/World/Tools/CuttingTool_0`` and ``/World/Franka_0``; this port only
    parameterizes those paths (C4 checks attribute equality after rebasing). ``finger_root=None`` skips the
    finger material binding exactly as the frozen function does when ``/World/Franka_0`` is absent.
    """
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
    from ..pad_contact import PATCH_PROFILE, POINT_PROFILE, patch_receipt
    contact_profile = POINT_PROFILE if contact_profile is None else contact_profile
    if contact_profile not in (POINT_PROFILE, PATCH_PROFILE):
        raise ValueError("Unknown Lab grip contact profile")
    patch = patch_receipt(spec.handle_half_extents_m, grasp.friction) if contact_profile == PATCH_PROFILE else None
    parent = root.rsplit("/", 1)[0]
    if parent and not stage.GetPrimAtPath(parent):
        UsdGeom.Xform.Define(stage, parent)
    body = _author_tool_body(stage, root, spec, world_pose)
    material = _make_grip_material(stage, root+"/GripMaterial", grasp, patch)
    finger_material = (_make_grip_material(stage, finger_root+"/PhysicalGripMaterial", grasp, patch)
                       if finger_root and stage.GetPrimAtPath(finger_root) else None)
    _author_tool_shapes(stage, root, spec, material)
    if finger_material is not None:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(finger_root+"/") and any(name in path for name in ("panda_leftfinger", "panda_rightfinger")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(finger_material, materialPurpose="physics")
                    if patch and prim.GetName() in ("left_finger_pad", "right_finger_pad"):
                        prim.AddAppliedSchema("MjcCollisionAPI")
                        prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(4)
    body.GetPrim().SetCustomDataByKey("research:tool_attachment_mode", "physical_grasp")
    body.GetPrim().SetCustomDataByKey("research:grip_contact_profile", contact_profile)
    body.GetPrim().SetCustomDataByKey("research:contact_owner", "MuJoCo-Warp rigid contacts; TLMPM soft contact only")
    _author_edge_xform(stage, root, spec)
    return body.GetPrim()


def author_rigid_mount_tool(stage, franka_root, spec, grasp):
    """B1: the ToolSpec body welded to ``panda_hand`` by a FixedJoint at ``hand_from_tool`` (ADR 0022 §5.1).

    The tool prim is a child of the hand prim (nested rigid bodies are separate bodies), so its local
    transform is ``hand_from_tool``. Finger-tool shape pairs are filtered; the handle would otherwise be
    squeezed by fingers held at the nominal opening.
    """
    from pxr import Gf, Sdf, UsdPhysics
    hand = hand_body_path(stage, franka_root)
    left, right = finger_body_paths(stage, franka_root)
    root = f"{hand}/{RIGID_TOOL_NAME}"
    body = _author_tool_body(stage, root, spec, spec.hand_from_tool_xyz_xyzw)
    material = _make_grip_material(stage, root+"/GripMaterial", grasp)
    _author_tool_shapes(stage, root, spec, material)
    filtered = UsdPhysics.FilteredPairsAPI.Apply(body.GetPrim())
    filtered.CreateFilteredPairsRel().SetTargets([Sdf.Path(left), Sdf.Path(right)])
    joint = UsdPhysics.FixedJoint.Define(stage, root+"/"+RIGID_MOUNT_JOINT)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(root)])
    translation, orientation = _gf_pose(spec.hand_from_tool_xyz_xyzw)
    joint.CreateLocalPos0Attr(Gf.Vec3f(translation))
    joint.CreateLocalRot0Attr(orientation)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    body.GetPrim().SetCustomDataByKey("research:tool_attachment_mode", "rigid_mount_v1")
    body.GetPrim().SetCustomDataByKey("research:reduced_fidelity", "rigid_tool_mount_no_grasp")
    _author_edge_xform(stage, root, spec)
    return body.GetPrim()


def author_franka_reference(stage, path, asset, base_pose):
    """In-memory stand-in for Lab's ``spawn_from_usd``: reference, variants, self collision, base pose."""
    from pxr import UsdGeom
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(asset))
    prim.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("mujoco")
    prim.GetVariantSets().GetVariantSet("Colliders").SetVariantSelection("convex_hulls")
    attribute = prim.GetAttribute("newton:selfCollisionEnabled")
    if attribute:
        attribute.Set(True)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    translation, orientation = _gf_pose([*base_pose[0], *base_pose[1]])
    xf.AddTranslateOp().Set(translation)
    xf.AddOrientOp().Set(orientation)
    return prim


# ----------------------------------------------------------------------------- Isaac Lab spawn functions

def spawn_franka_batched(prim_path, cfg, translation=None, orientation=None, *, arm, spec=None, grasp=None,
                         tool_mount="rigid_mount_v1", **kwargs):
    """Lab spawn: the official Franka, plus the B1 welded tool inside Franka 0's subtree."""
    import isaaclab.sim as sim_utils
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if arm == 0 and tool_mount == "rigid_mount_v1":
        author_rigid_mount_tool(prim.GetStage(), prim_path, spec, grasp)
    return prim


def spawn_tool_batched(prim_path, cfg, translation=None, orientation=None, *, spec, grasp, contact_profile=None,
                       **kwargs):
    """Lab spawn of the B2 free tool at ``{ENV_REGEX_NS}/Tool`` (the prototype path of world 0)."""
    import isaaclab.sim as sim_utils
    if not prim_path.endswith("/"+TOOL_NAME):
        raise ValueError(f"The batched B2 tool lives at {{ENV_REGEX_NS}}/{TOOL_NAME}, got {prim_path}")
    orientation = (0., 0., 0., 1.) if orientation is None else orientation
    translation = (0., 0., .4) if translation is None else translation
    pose = np.array([*translation, *orientation], float)
    finger_root = prim_path.rsplit("/", 1)[0]+"/Franka_0"
    return author_batched_tool(sim_utils.SimulationContext.instance().stage, prim_path, spec, pose, grasp,
                               finger_root, contact_profile=contact_profile)


# ----------------------------------------------------------------------------- Isaac Lab configuration

def make_batched_lab_cfg(options, task):
    """``DirectRLEnvCfg`` of workload B (ADR 0022 §3.1): N worlds, env_spacing 0, shared workcell."""
    options = options.validate()
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
    from isaaclab_newton.sim.schemas import MujocoJointCfg, MujocoRigidBodyCfg, NewtonArticulationCfg

    from ..recording import ROOT
    from ..tool_profile import tool_spec_for_task
    from ..workcell import ASSETS
    asset = ROOT/"artifacts/workcell_asset_cache"/ASSETS["franka"]
    if not asset.is_file():
        raise FileNotFoundError(asset)
    spec = tool_spec_for_task(task)
    grasp = options.grasp
    q0, q1 = home_joint_positions(task, options.tool_mount, grasp, options.initial_joint_positions)
    names = [f"panda_joint{i}" for i in range(1, 8)]

    def robot(arm):
        q = q0 if arm == 0 else q1
        joint_pos = dict(zip(names, map(float, q[:7])))
        joint_pos["panda_finger_joint.*"] = float(q[7])
        stiffness, damping, effort, armature = finger_actuator(arm, options.tool_mount, grasp)
        actuators = {name: ImplicitActuatorCfg(joint_names_expr=[expr], stiffness=ke, damping=kd,
                                               joint_effort_limit=limit, armature=arm_a, viscous_friction=0.)
                     for name, expr, ke, kd, limit, arm_a in ARM_ACTUATORS}
        actuators["fingers"] = ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], stiffness=stiffness,
            damping=damping, joint_effort_limit=effort, armature=armature, viscous_friction=0.)
        position, rotation = FRANKA_BASE_POSES[arm]
        return ArticulationCfg(prim_path=f"{ENV_REGEX_NS}/Franka_{arm}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset),
                func=partial(spawn_franka_batched, arm=arm, spec=spec, grasp=grasp, tool_mount=options.tool_mount),
                variants={"Physics": "mujoco", "Colliders": "convex_hulls"},
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
                rigid_props={"/.*": [MujocoRigidBodyCfg(gravcomp=1.)]},
                joint_drive_props={"/.*": [MujocoJointCfg(actuatorgravcomp=True)]}),
            init_state=ArticulationCfg.InitialStateCfg(pos=position, rot=rotation, joint_pos=joint_pos),
            actuators=actuators)

    def fixed_box(name, center, size, rotation=(0., 0., 0., 1.)):
        return AssetBaseCfg(prim_path=f"{WORKCELL_ROOT}/{name}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(map(float, center)), rot=tuple(map(float, rotation))),
            spawn=sim_utils.CuboidCfg(size=tuple(map(float, size)),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)]))

    @configclass
    class BatchedCuttingSceneCfg(InteractiveSceneCfg):
        table = fixed_box("Table", table_box(task)[0], 2*np.asarray(table_box(task)[1]))
        franka_0 = robot(0)
        franka_1 = robot(1)

    scene = BatchedCuttingSceneCfg(num_envs=int(options.num_envs), env_spacing=0., replicate_physics=True)
    for index, box in enumerate(task.fixture_boxes_local):
        setattr(scene, f"task_fixture_{index}", fixed_box(f"TaskFixture_{index}", task.points_to_world(box.center_m),
                2*np.asarray(box.half_extents_m), task.world_from_workpiece_xyzw[3:]))
    if options.tool_mount == "physical_grasp_v1":
        scene.cutting_tool = RigidObjectCfg(prim_path=f"{ENV_REGEX_NS}/{TOOL_NAME}",
            spawn=sim_utils.SpawnerCfg(func=partial(spawn_tool_batched, spec=spec, grasp=grasp,
                                                    contact_profile=options.grip_contact_profile)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., .4)))

    @configclass
    class SeparateWorldsMJWarpSolverCfg(MJWarpSolverCfg):
        """RCS: N=1 and N>1 take one MuJoCo code path (``separate_worlds=True``; ADR 0022 §3.1)."""
        separate_worlds: bool = True

    from ..policy_env import ACTION_STATUSES  # noqa: F401  (import check: the frozen status table exists)
    from ..rl.actions import ACTION_DIM
    from ..rl.observations import POLICY_TERMS
    POLICY_DIM = sum(term[1] for term in POLICY_TERMS)
    cfg = DirectRLEnvCfg(seed=int(options.seed), decimation=10, episode_length_s=float(task.duration_s),
        action_space=ACTION_DIM, observation_space=POLICY_DIM, ui_window_class_type=None, scene=scene,
        sim=sim_utils.SimulationCfg(device=options.device, dt=.002, render_interval=10,
            physics=NewtonCfg(use_cuda_graph=options.graph, load_visual_shapes=False,
                deterministic_mode=options.deterministic_mode, default_shape_cfg=NewtonShapeCfg(gap=.001),
                solver_cfg=SeparateWorldsMJWarpSolverCfg(use_mujoco_cpu=False, iterations=50, nconmax=256,
                    njmax=1024, integrator="implicitfast", cone="elliptic",
                    # Newton's run_to_run guarantee needs MJWarp's internal sensor stage off (no sensors are used)
                    disable_sensors=options.deterministic_mode != "not_guaranteed"))))
    return cfg


# ----------------------------------------------------------------------------- pure Newton rigid scene (C tests)

@dataclass
class NewtonRigidScene:
    """A finalized Newton model of the workload B rigid scene (world e = env e; statics in world -1)."""

    model: object
    num_envs: int
    tool_mount: str
    spec: object
    prototype_builder: object      # Franka 0 source builder (B1: with the welded tool) -- the IK prototype
    initial_q: np.ndarray
    hand_from_edge: np.ndarray
    stage: object
    labels_prefix: str = ENV_TEMPLATE

    def make_solver(self, **overrides):
        from newton.solvers import SolverMuJoCo
        settings = dict(use_mujoco_cpu=False, separate_worlds=True, iterations=50, nconmax=256, njmax=1024,
                        integrator="implicitfast", cone="elliptic")
        settings.update(overrides)
        return SolverMuJoCo(self.model, **settings)


def _set_franka_drives(builder, arm, tool_mount, grasp, q):
    """Drive gains on a single-Franka source builder (9 DOFs, robot.DualFrankaController convention)."""
    import newton
    if builder.joint_dof_count != 9 or builder.joint_coord_count != 9:
        raise RuntimeError(f"Expected 7 arm + 2 finger DOFs, got {builder.joint_coord_count}/{builder.joint_dof_count}")
    ke, kd, limit, armature = finger_actuator(arm, tool_mount, grasp)
    builder.joint_q[:] = list(map(float, q))
    builder.joint_target_q[:] = list(map(float, q))
    builder.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)]*9
    builder.joint_target_ke[:] = [1000.]*4+[600.]*3+[ke, ke]
    builder.joint_target_kd[:] = [80.]*4+[40.]*3+[kd, kd]
    builder.joint_effort_limit[:] = [87.]*4+[12.]*3+[limit, limit]
    builder.joint_armature[:] = [.2]*4+[.07]*3+[armature, armature]
    builder.custom_attributes["mujoco:gravcomp"].values = {i: 1. for i in range(builder.body_count)}
    builder.custom_attributes["mujoco:jnt_actgravcomp"].values = {i: True for i in range(9)}


def build_newton_rigid_scene(task, num_envs, tool_mount="rigid_mount_v1", *, device="cpu", grasp=None,
                             asset=None):
    """Pure Newton model of the rigid scene, for the CPU C tests (ADR 0022 §8 C1/C3).

    The same USD authoring as the Lab spawn functions; one source builder per prototype prim, composed into a
    world prototype and replicated N times with ``ModelBuilder.replicate`` (labels ``/World/envs/env_{e}/...``
    as in Lab); the table and fixtures are static shapes in world -1.
    """
    import newton
    import warp as wp
    from newton.solvers import SolverMuJoCo
    from pxr import Usd, UsdGeom

    from ..recording import ROOT
    from ..scene_frames import compose_pose
    from ..tool_profile import tool_spec_for_task
    from ..workcell import ASSETS
    if tool_mount not in TOOL_MOUNTS:
        raise unsupported("UNSUPPORTED_TOOL_MOUNT", repr(tool_mount))
    grasp = grasp or _grasp_default()
    asset = asset or ROOT/"artifacts/workcell_asset_cache"/ASSETS["franka"]
    spec = tool_spec_for_task(task)
    q0, q1 = home_joint_positions(task, tool_mount, grasp)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    env0 = ENV_TEMPLATE.format(0)
    UsdGeom.Xform.Define(stage, env0)
    sources = []
    for arm in (0, 1):
        path = f"{env0}/Franka_{arm}"
        author_franka_reference(stage, path, asset, FRANKA_BASE_POSES[arm])
        sources.append(path)
    if tool_mount == "rigid_mount_v1":
        author_rigid_mount_tool(stage, sources[0], spec, grasp)
    else:
        home = getattr(task, "home", None) or {}
        hand = np.asarray(home["hand_pose_world_xyzw"], float)
        tool_pose = compose_pose(hand, spec.hand_from_tool_xyz_xyzw)
        author_batched_tool(stage, f"{env0}/{TOOL_NAME}", spec, tool_pose, grasp, sources[0])
        sources.append(f"{env0}/{TOOL_NAME}")
    with wp.ScopedDevice(device):
        source_builders = []
        for index, path in enumerate(sources):
            b = newton.ModelBuilder()
            b.rigid_gap = .001
            SolverMuJoCo.register_custom_attributes(b)
            b.add_usd(stage, root_path=path, floating=None, enable_self_collisions=True,
                      collapse_fixed_joints=False, load_visual_shapes=False, skip_mesh_approximation=False)
            if index < 2:
                _set_franka_drives(b, index, tool_mount, grasp, q0 if index == 0 else q1)
            b.shape_gap[:] = [.001]*len(b.shape_gap)
            source_builders.append(b)
        prototype = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(prototype)
        for b in source_builders:
            prototype.add_builder(b)
        prefix = env0+"/"
        for attribute in ("body_label", "joint_label", "shape_label", "articulation_label"):
            values = getattr(prototype, attribute, None)
            if values is not None:
                values[:] = [v[len(prefix):] if isinstance(v, str) and v.startswith(prefix) else v for v in values]
        builder = newton.ModelBuilder()
        builder.rigid_gap = .001
        SolverMuJoCo.register_custom_attributes(builder)
        center, half = table_box(task)
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(*center)), hx=half[0], hy=half[1], hz=half[2],
                              label=f"{WORKCELL_ROOT}/Table")
        rotation = np.asarray(task.world_from_workpiece_xyzw[3:], float)
        for index, box in enumerate(task.fixture_boxes_local):
            center = np.asarray(task.points_to_world(box.center_m), float)
            half = np.asarray(box.half_extents_m, float)
            builder.add_shape_box(-1, xform=wp.transform(wp.vec3(*center), wp.quat(*rotation)),
                                  hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
                                  label=f"{WORKCELL_ROOT}/TaskFixture_{index}")
        builder.replicate(prototype, int(num_envs), label_prefixes=[ENV_TEMPLATE.format(e) for e in range(num_envs)])
        model = builder.finalize(device=device)
    initial_q = model.joint_q.numpy().copy()
    return NewtonRigidScene(model=model, num_envs=int(num_envs), tool_mount=tool_mount, spec=spec,
                            prototype_builder=source_builders[0], initial_q=initial_q,
                            hand_from_edge=np.asarray(task.hand_from_edge_xyz_xyzw, float), stage=stage)


def scene_receipt(options, task):
    """Static description of the batched scene (no simulator access)."""
    return {"version": LAB_SCENE_VERSION, "num_envs": int(options.num_envs), "env_spacing": 0.,
            "tool_mount": options.tool_mount, "env_template": ENV_TEMPLATE, "workcell_root": WORKCELL_ROOT,
            "franka_paths": [f"{ENV_REGEX_NS}/Franka_0", f"{ENV_REGEX_NS}/Franka_1"],
            "tool_path": (f"{ENV_REGEX_NS}/Franka_0/.../panda_hand/{RIGID_TOOL_NAME}"
                          if options.tool_mount == "rigid_mount_v1" else f"{ENV_REGEX_NS}/{TOOL_NAME}"),
            "separate_worlds": True, "rigid_scene_key": rigid_scene_key(task), **options.fidelity()}


__all__ = ["LAB_SCENE_VERSION", "TOOL_MOUNTS", "LAB_UNSUPPORTED_CODES", "UnsupportedLabBatchConfiguration",
           "BatchedLabOptions", "rigid_scene_key", "check_uniform_rigid_scene", "home_joint_positions",
           "finger_hold_target", "author_batched_tool", "author_rigid_mount_tool", "make_batched_lab_cfg",
           "build_newton_rigid_scene", "NewtonRigidScene", "scene_receipt", "replace"]
