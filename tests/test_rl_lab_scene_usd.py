"""ADR 0022 §8 C4: USD authoring of workload B (torch-free; ``.venv``) and the scene/option refusals (§10).

* ``author_batched_tool`` equals the frozen ``newton_tool.author_newton_tool`` after rebasing
  ``/World/Tools/CuttingTool_0 -> /World/envs/env_0/Tool`` and ``/World/Franka_0 -> /World/envs/env_0/Franka_0``
  (every attribute, relationship target, applied schema, custom data and material binding, exactly);
* B1 FixedJoint structure (hand -> tool at ``hand_from_tool``, filtered finger pairs, mass properties);
* no ``RigidBodyAPI`` under the shared workcell paths;
* every ADR 0022 §10 refusal raises its UNSUPPORTED code.
"""
from functools import lru_cache

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from robotic_cutting_sim.batched import lab_scene
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration


@lru_cache(maxsize=None)
def task():
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(512), goal_config(), seed=42, particle_spacing_m=.0025)


def spec():
    from robotic_cutting_sim.tool_profile import tool_spec_for_task
    return tool_spec_for_task(task())


def franka_asset():
    from robotic_cutting_sim.recording import ROOT
    from robotic_cutting_sim.workcell import ASSETS
    return ROOT/"artifacts/workcell_asset_cache"/ASSETS["franka"]


def _stage(franka_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    lab_scene.author_franka_reference(stage, franka_path, franka_asset(), lab_scene.FRANKA_BASE_POSES[0])
    return stage


def _value(value, rebase):
    if isinstance(value, (Sdf.Path,)):
        return rebase(str(value))
    if hasattr(value, "__len__") and not isinstance(value, str):
        try:
            return [(_value(v, rebase)) for v in value]
        except TypeError:
            pass
    if isinstance(value, float):
        return float(value)
    return value if isinstance(value, (int, str, bool, type(None))) else str(value)


def describe(stage, roots, rebase):
    """Every authored property / schema / custom datum of the prims under ``roots`` (paths rebased)."""
    out = {}
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            path = rebase(str(prim.GetPath()))
            props = {}
            for prop in prim.GetAuthoredProperties():
                if isinstance(prop, Usd.Attribute):
                    props[prop.GetName()] = _value(prop.Get(), rebase)
                else:
                    props[prop.GetName()] = sorted(rebase(str(t)) for t in prop.GetTargets())
            out[path] = {"type": str(prim.GetTypeName()), "schemas": sorted(prim.GetAppliedSchemas()),
                         "custom": {k: str(v) for k, v in sorted(prim.GetCustomData().items())}, "props": props}
    return out


def test_batched_tool_equals_frozen_author_newton_tool_after_rebasing():
    from robotic_cutting_sim.newton_tool import TOOL_PATH, author_newton_tool
    from robotic_cutting_sim.physical_tool import ToolGraspConfig
    grasp = ToolGraspConfig()
    pose = np.array([.01, -.02, .3, 0., 0., .3826834323650898, .9238795325112867])
    frozen = _stage("/World/Franka_0")
    author_newton_tool(frozen, spec(), pose, grasp)
    batched = _stage("/World/envs/env_0/Franka_0")
    lab_scene.author_batched_tool(batched, "/World/envs/env_0/Tool", spec(), pose, grasp, "/World/envs/env_0/Franka_0")

    def rebase(path):
        return (path.replace(TOOL_PATH, "TOOL").replace("/World/Franka_0", "FRANKA")
                .replace("/World/envs/env_0/Tool", "TOOL").replace("/World/envs/env_0/Franka_0", "FRANKA"))

    a = describe(frozen, [TOOL_PATH, "/World/Franka_0"], rebase)
    b = describe(batched, ["/World/envs/env_0/Tool", "/World/envs/env_0/Franka_0"], rebase)
    assert set(a) == set(b)
    for path in a:
        assert a[path] == b[path], f"authoring differs at {path}"
    # the finger physics material is bound inside the Franka subtree (Lab imports subtrees separately)
    assert "FRANKA/PhysicalGripMaterial" in b


def test_rigid_mount_fixed_joint_structure():
    from robotic_cutting_sim.physical_tool import ToolGraspConfig
    root = "/World/envs/env_0/Franka_0"
    stage = _stage(root)
    s = spec()
    lab_scene.author_rigid_mount_tool(stage, root, s, ToolGraspConfig())
    hand = lab_scene.hand_body_path(stage, root)
    tool = stage.GetPrimAtPath(hand+"/CuttingTool")
    assert tool and tool.HasAPI(UsdPhysics.RigidBodyAPI) and tool.HasAPI(UsdPhysics.MassAPI)
    mass = UsdPhysics.MassAPI(tool)
    assert mass.GetMassAttr().Get() == pytest.approx(s.mass_kg, rel=1e-6)
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(hand+"/CuttingTool/HandMount"))
    assert joint and [str(t) for t in joint.GetBody0Rel().GetTargets()] == [hand]
    assert [str(t) for t in joint.GetBody1Rel().GetTargets()] == [hand+"/CuttingTool"]
    np.testing.assert_allclose(np.asarray(joint.GetLocalPos0Attr().Get()), s.hand_from_tool_xyz_xyzw[:3], atol=1e-7)
    q = joint.GetLocalRot0Attr().Get()
    np.testing.assert_allclose([*q.GetImaginary(), q.GetReal()], s.hand_from_tool_xyz_xyzw[3:], atol=1e-7)
    xf = UsdGeom.Xformable(tool).GetLocalTransformation()
    np.testing.assert_allclose(np.asarray(xf).T[:3, 3], s.hand_from_tool_xyz_xyzw[:3], atol=1e-7)
    left, right = lab_scene.finger_body_paths(stage, root)
    filtered = UsdPhysics.FilteredPairsAPI(tool).GetFilteredPairsRel().GetTargets()
    assert sorted(map(str, filtered)) == sorted([left, right])
    assert {c.GetName() for c in tool.GetChildren()} >= {"Handle", "Blade", "GripMaterial", "HandMount", "CuttingEdge"}
    assert tool.GetCustomDataByKey("research:tool_attachment_mode") == "rigid_mount_v1"


@pytest.mark.parametrize("mount", lab_scene.TOOL_MOUNTS)
def test_no_rigid_body_under_shared_paths_and_per_world_bodies(mount):
    sc = lab_scene.build_newton_rigid_scene(task(), 2, mount, device="cpu")
    for prim in sc.stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert path.startswith("/World/envs/env_0/"), f"rigid body outside {{ENV_REGEX_NS}}: {path}"
    world = sc.model.body_world.numpy()
    assert np.all(world >= 0), "no global (world -1) bodies"
    assert np.array_equal(np.bincount(world), [23, 23])
    shapes = np.asarray(sc.model.shape_world.numpy())
    labels = list(sc.model.shape_label)
    statics = [labels[i] for i in np.flatnonzero(shapes < 0)]
    assert statics and all(label.startswith(lab_scene.WORKCELL_ROOT) for label in statics)
    dofs = 18 if mount == "rigid_mount_v1" else 24
    assert sc.model.joint_dof_count == 2*dofs


@pytest.mark.parametrize("kwargs, code", [
    (dict(env_spacing=2.), "UNSUPPORTED_ENV_SPACING"),
    (dict(tool_mount="welded_v0"), "UNSUPPORTED_TOOL_MOUNT"),
    (dict(action_spec="joint_position_rad"), "UNSUPPORTED_ACTION_SPEC"),
    (dict(use_mujoco_cpu=True), "UNSUPPORTED_RIGID_CPU"),
    (dict(device="cpu"), "UNSUPPORTED_RIGID_CPU"),
    (dict(cameras=("wrist_d405_0",)), "UNSUPPORTED_SENSOR"),
    (dict(recorders=("raw2ms",)), "UNSUPPORTED_RECORDER"),
    (dict(actuator_overrides={"fingers": 1.}), "UNSUPPORTED_ACTUATOR_OVERRIDE"),
    (dict(tool_init_mode="free"), "UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET"),
    (dict(material="tlmpm_v7_2"), "UNSUPPORTED_MATERIAL_COUPLING"),
    (dict(num_envs=0), "UNSUPPORTED_EMPTY_BATCH"),
])
def test_options_refuse_unsupported(kwargs, code):
    with pytest.raises(UnsupportedBatchConfiguration) as error:
        lab_scene.BatchedLabOptions(**kwargs).validate()
    assert error.value.code == code and str(error.value).startswith("UNSUPPORTED")
    assert lab_scene.BatchedLabOptions(device="cpu", cpu_test_mode=True).validate()


def test_nonuniform_rigid_scene_and_tool_profile_refused():
    from dataclasses import replace
    t = task()
    assert lab_scene.check_uniform_rigid_scene([t, t])
    moved = replace(t, table_top_world_z_m=t.table_top_world_z_m+.01)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_RIGID_SCENE"):
        lab_scene.check_uniform_rigid_scene([t, moved])
    from types import SimpleNamespace
    fields = ("table_top_world_z_m", "fixture_boxes_local", "world_from_workpiece_xyzw", "hand_from_edge_xyz_xyzw", "home")
    other = SimpleNamespace(**{name: getattr(t, name) for name in fields}, tool_profile_id="ruler_knife_v2_x")
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_PER_ENV_TOOL_PROFILE"):
        lab_scene.check_uniform_rigid_scene([t, other])


def test_fidelity_labels():
    b1 = lab_scene.BatchedLabOptions(device="cpu", cpu_test_mode=True).fidelity()
    assert b1["reduced_fidelity"] and "rigid_tool_mount_no_grasp" in b1["reasons"]
    assert "robot_only_no_material_coupling" in b1["reasons"]
    b2 = lab_scene.BatchedLabOptions(device="cpu", cpu_test_mode=True, tool_mount="physical_grasp_v1").fidelity()
    assert b2["reasons"] == ["robot_only_no_material_coupling"]
    assert b2["listed_differences"]["rigid_engine"].startswith("newton_mjwarp")
