"""Analytic mass/frame/contact bookkeeping controls; not grasp-physics passes."""
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from robotic_cutting_sim.newton_tool import (
    TOOL_PATH, apply_tool_wrench, author_newton_tool, collect_tool_contacts,
    integrate_contact_sample, relative_pose_error, relative_tool_pose,
)
from robotic_cutting_sim.physical_tool import ToolGraspConfig, make_tool_spec
from robotic_cutting_sim.scene_frames import compose_pose


def test_tool_wrench_has_actual_lever_and_no_hand_receiver():
    wp.init()
    rotation = Rotation.from_euler("z", 90, degrees=True).as_quat()
    poses = wp.array([[0, 0, 0, 0, 0, 0, 1], [.3, .2, .4, *rotation]], dtype=wp.transform, device="cpu")
    com = wp.array([[0, 0, 0], [.01, .02, .03]], dtype=wp.vec3, device="cpu")
    wrench = np.array([1., 2., -3., .1, -.2, .3], np.float32)
    payload = wp.array([wrench], dtype=wp.spatial_vector, device="cpu")
    forces = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
    impulse = wp.zeros(6, dtype=float, device="cpu")
    applied_com = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
    applied_pose = wp.zeros(1, dtype=wp.transform, device="cpu")
    hand_wrench = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
    edge = np.array([.02, -.03, .08])
    wp.launch(apply_tool_wrench, dim=1, inputs=[poses, com, payload, 1, wp.vec3(*edge), .002, forces, impulse,
        applied_com, applied_pose, 0, hand_wrench], device="cpu")
    lever = Rotation.from_quat(rotation).apply(edge-np.array([.01, .02, .03]))
    expected = np.r_[wrench[:3], wrench[3:]+np.cross(lever, wrench[:3])]
    assert np.array_equal(forces.numpy()[0], np.zeros(6))
    assert np.allclose(forces.numpy()[1], expected, atol=1e-7)
    assert np.allclose(impulse.numpy(), .002*expected, atol=1e-9)
    assert np.array_equal(hand_wrench.numpy(), np.zeros((1, 6)))


def test_contact_order_preserves_action_reaction_and_world_com_moment():
    wp.init()
    # Pair0: tool/left; pair1: right/tool. Both physical forces on tool are
    # upward, while the sensor force convention is force on shape0.
    kwargs = {"device": "cpu"}
    count = wp.array([2], dtype=int, **kwargs)
    a, b = wp.array([0, 2], dtype=int, **kwargs), wp.array([1, 0], dtype=int, **kwargs)
    force = wp.array([[2, 0, .5, 0, 0, 0], [2, 0, -.5, 0, 0, 0]], dtype=wp.spatial_vector, **kwargs)
    normals = wp.array([[1, 0, 0], [1, 0, 0]], dtype=wp.vec3, **kwargs)
    points = wp.array([[0, -.01, .1], [0, .01, .1]], dtype=wp.vec3, **kwargs)
    dist = wp.array([-.0001, -.0002], dtype=float, **kwargs)
    bodies = wp.array([0, 1, 2], dtype=int, **kwargs)
    handle = wp.array([1, 0, 0], dtype=int, **kwargs)
    poses = wp.array([[0, 0, 0, 0, 0, 0, 1]]*3, dtype=wp.transform, **kwargs)
    com = wp.zeros(3, dtype=wp.vec3, **kwargs)
    sample, integral = wp.zeros(16, dtype=float, **kwargs), wp.zeros(16, dtype=float, **kwargs)
    wp.launch(collect_tool_contacts, dim=2, inputs=[count, a, b, force, normals, points, dist,
        bodies, handle, poses, com, 0, 1, 2, sample], device="cpu")
    wp.launch(integrate_contact_sample, dim=1, inputs=[sample, .002, integral], device="cpu")
    measured = sample.numpy()
    assert np.allclose(measured[:6], [0, 0, 1, 0, 0, .04], atol=1e-7)
    assert np.allclose(measured[6:8], [2, 2])
    assert integral.numpy()[8] == 1 and integral.numpy()[9] == 1
    assert np.isclose(integral.numpy()[10], .0002)
    assert np.allclose(integral.numpy()[:6], .002*measured[:6], atol=1e-9)


def test_generic_lab_usd_keeps_tool_independent_and_full_mass_tensor():
    from pxr import Usd, UsdGeom, UsdPhysics
    spec = make_tool_spec()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    author_newton_tool(stage, spec, np.array([.1, .2, .3, 0, 0, 0, 1]), ToolGraspConfig())
    prim = stage.GetPrimAtPath(TOOL_PATH)
    assert prim.GetParent().GetPath().pathString == "/World/Tools"
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
    assert not any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse())
    mass = UsdPhysics.MassAPI(prim)
    assert np.isclose(mass.GetMassAttr().Get(), spec.mass_kg)
    q = mass.GetPrincipalAxesAttr().Get()
    r = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
    tensor = r @ np.diag(mass.GetDiagonalInertiaAttr().Get()) @ r.T
    assert np.allclose(tensor, spec.inertia_com_tool_kg_m2, atol=1e-10)


def test_relative_slip_metrics_are_invariant_to_world_frame():
    hand = np.array([.1, -.2, .4, 0, 0, 0, 1])
    tool = np.array([.1, -.2, .5, 0, 0, 0, 1])
    transform = np.r_[[.4, -.3, .2], Rotation.from_euler("xyz", [20, 40, 60], degrees=True).as_quat()]
    expected = relative_tool_pose(hand, tool)
    measured = relative_tool_pose(compose_pose(transform, hand), compose_pose(transform, tool))
    assert np.allclose(relative_pose_error(expected, measured), [0, 0], atol=1e-12)
    slipped = expected.copy();slipped[0] += .003
    assert np.isclose(relative_pose_error(expected, slipped)[0], .003)


def test_optional_patch_authors_only_soft_pad_dimensions_and_finite_pair_material():
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from robotic_cutting_sim.pad_contact import PATCH_PROFILE, patch_receipt
    spec = make_tool_spec();stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Franka_0")
    pads = []
    for side in ("left", "right"):
        pad = UsdGeom.Cube.Define(stage, f"/World/Franka_0/panda_{side}finger/{side}_finger_pad")
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim());pads.append(pad.GetPrim())
    author_newton_tool(stage, spec, np.array([0, 0, .3, 0, 0, 0, 1]), ToolGraspConfig(), contact_profile=PATCH_PROFILE)
    expected = patch_receipt(spec.handle_half_extents_m, .6)["torsional_friction_length_m"]
    for pad in pads:
        assert pad.GetAttribute("mjc:condim").Get() == 4
        material, _ = UsdShade.MaterialBindingAPI(pad).ComputeBoundMaterial(materialPurpose="physics")
        assert np.isclose(material.GetPrim().GetAttribute("mjc:torsionalfriction").Get(), expected)
    for shape in spec.shapes:
        prim = stage.GetPrimAtPath(TOOL_PATH+"/"+shape.name)
        assert not prim.GetAttribute("mjc:condim").HasAuthoredValueOpinion()
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        assert np.isclose(material.GetPrim().GetAttribute("mjc:torsionalfriction").Get(), expected)
