"""Adversarial offline controls; generated fixtures are not live physics proof."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robotic_cutting_sim.physical_tool import ToolGraspConfig, make_tool_spec
from robotic_cutting_sim.scene_frames import compose_pose, invert_transform, matrix_pose, pose_matrix

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_physical_tool.py"
loader = importlib.util.spec_from_file_location("physical_tool_audit", SCRIPT)
audit = importlib.util.module_from_spec(loader)
loader.loader.exec_module(audit)
TOOL, HAND = "/World/Tools/CuttingTool_0", "/World/Franka_0/panda_hand"


def json_file(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def contact_row(time=.1, phase="pregrasp_settling"):
    spec = make_tool_spec()
    body = np.array([0., 0., .3, 0., 0., 0., 1.])
    hand = matrix_pose(pose_matrix(body) @ invert_transform(spec.T_hand_tool))
    records = []
    for role, side in [("left", 1.), ("right", -1.)]:
        point = body[:3] + [0., side * .011, 0.]
        records.append({"body0": TOOL, "body1": HAND + "/panda_" + role + "finger",
                        "dt_s": .1, "time_s": time, "position_world_m": point.tolist(),
                        "raw_impulse_world_ns": [0., -side * .1, 0.], "in_handle_region": True})
    return {"phase": phase, "native_time_s": time, "hand_pose_xyzw": hand.tolist(),
        "tool_body_pose_xyzw": body.tolist(), "measured_edge_pose_xyzw": compose_pose(body, spec.tool_from_edge_xyz_xyzw).tolist(),
        "tool_linear_velocity_world_m_s": [0., 0., 0.], "tool_angular_velocity_world_rad_s": [0., 0., 0.],
        "finger_q_m": [[.011, .011], [.04, .04]], "finger_qdot_m_s": [[0., 0.], [0., 0.]],
        "contacts": {"sensor_valid": True, "sensor_time_s": time, "sensor_fresh": True, "new_physics_step": True,
            "bilateral_loaded_handle_contact": True, "contacts": records,
            "tensor_forces": {"status": "VALID_NATIVE_NORMAL_AND_FRICTION_DATA", "records": [
                {"kind": "normal", "separation_m": -.00001}], "capacity_may_be_reached": False}}}


def author_stage(path, spec):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    for target in (HAND, HAND + "/panda_leftfinger", HAND + "/panda_rightfinger", TOOL):
        prim = UsdGeom.Xform.Define(stage, target).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(TOOL))
    mass.CreateMassAttr(spec.mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*spec.com_tool_m))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec.principal_moments_kg_m2))
    q = spec.principal_axes_xyzw
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    for shape in spec.shapes:
        mesh = UsdGeom.Mesh.Define(stage, TOOL + "/" + shape.name)
        mesh.CreatePointsAttr(shape.vertices_tool_m.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(shape.triangles))
        mesh.CreateFaceVertexIndicesAttr(shape.triangles.ravel().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    stage.GetRootLayer().Save()


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "synthetic_not_a_live_episode"
    path.mkdir(); (path / "implementation").mkdir(); (path / "oracle_debug").mkdir()
    archived = path / "implementation/recorded.py"
    archived.write_text("# Synthetic archived source, no runtime evidence\n")
    json_file(path / "manifest.json", {"backend": "LIVE_IsaacSim6.2_cutting", "implementation_sha256": {
        "src/recorded.py": audit.digest(archived)}})
    spec = make_tool_spec()
    json_file(path / "setup.json", {"tool_attachment_mode": "physical_grasp", "physical_tool": spec.to_dict(),
        "hand_prims": [HAND], "native_time_origin_s": 2., "physical_knife_body_index": 0,
        "config": {"control_dt": .1, "robot_substeps": 1}})
    json_file(path / "grasp_reset_00.json", {"status": "DO_NOT_TRUST_SUMMARY", "config": ToolGraspConfig().to_dict()})
    rows = [contact_row(i / 10) for i in range(1, 21)]
    for i in range(21, 23):
        row = contact_row(i / 10, "coupled_episode")
        point = row["measured_edge_pose_xyzw"][:3]
        center = np.asarray(row["tool_body_pose_xyzw"][:3]) + spec.com_tool_m
        # Nonzero force with a known lever arm; torque includes r x F.
        wrench = np.array([.2, 0., -.3, 0., 0., 0.])
        torque = np.cross(np.asarray(point) - center, wrench[:3])
        row.update(applied_edge_material_wrench_world=wrench.tolist(), applied_external_edge_wrench_world=[0.] * 6,
                   force_application_point_world_m=point, tool_com_before_step_world_m=center.tolist(),
                   applied_total_wrench_about_com_world=np.r_[wrench[:3], torque].tolist(),
                   wrench_interval_dt_s=.1, direct_hand_wrench_injection=False)
        rows.append(row)
    (path / "physical_grasp_trace_00.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    sample = rows[0]
    np.savez(path / "robot_state.npz", time_s=[0., .1, .2], q=np.zeros((3, 2, 7)), qdot=np.zeros((3, 2, 7)),
        hand_pose_xyzw=[sample["hand_pose_xyzw"]] * 3, tool_body_pose_xyzw=[sample["tool_body_pose_xyzw"]] * 3,
        tool_pose_xyzw=[sample["measured_edge_pose_xyzw"]] * 3, desired_tool_pose_xyzw=[sample["measured_edge_pose_xyzw"]] * 3,
        tool_linear_velocity_world_m_s=np.zeros((3, 3)), tool_angular_velocity_world_rad_s=np.zeros((3, 3)),
        finger_q_m=[sample["finger_q_m"]] * 3, finger_qdot_m_s=[sample["finger_qdot_m_s"]] * 3,
        body_pose=np.array([[sample["tool_body_pose_xyzw"]]] * 3))
    json_file(path / "oracle_debug/diagnostics.json", [{"wrench_recipient": TOOL, "direct_hand_material_wrench": False}] * 2)
    author_stage(path / "final_live_scene.usda", spec)
    return path


def test_short_synthetic_fixture_only_passes_offline_checks_never_physical_acceptance(episode):
    result = audit.audit_episode(episode)
    assert result["status"] == "PASS_OFFLINE_ENGINEERING_AUDIT"
    assert result["pregrasp"]["status"] == "PASS_PREGRASP_ENGINEERING"
    assert result["trace_state_alignment"]["matched_robot_states"] == 3
    assert result["wrench"]["nonzero_load_records"] == 2
    assert all(value == "NOT_RUN" for value in result["required_controls"].values())
    assert not result["TOOL_RIGID_BODY_VALIDATED"] and not result["PHYSICAL_GRASP_VALIDATED"]
    assert not result["TWO_WAY_COUPLING_VALIDATED"] and not result["full_physics_acceptance"]
    json.dumps(result, allow_nan=False)


def test_saved_source_tampering_fails_without_comparing_mutable_checkout(episode):
    (episode / "implementation/recorded.py").write_text("changed\n")
    result = audit.audit_episode(episode)
    assert result["status"] == "FAIL_OFFLINE_ENGINEERING_AUDIT"
    assert result["sources"]["mismatches"] == ["src/recorded.py"]


@pytest.mark.parametrize("violation", ["fixed_joint", "kinematic", "disabled_collision"])
def test_welded_kinematic_or_noncolliding_tool_fails_real_usd_inspection(episode, violation):
    stage = Usd.Stage.Open(str(episode / "final_live_scene.usda"))
    if violation == "fixed_joint":
        joint = UsdPhysics.FixedJoint.Define(stage, "/HiddenToolJoint")
        joint.CreateBody0Rel().SetTargets([HAND]); joint.CreateBody1Rel().SetTargets([TOOL])
    elif violation == "kinematic":
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(TOOL)).CreateKinematicEnabledAttr(True)
    else:
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(TOOL + "/Handle")).CreateCollisionEnabledAttr(False)
    stage.GetRootLayer().Save()
    assert audit.audit_episode(episode)["status"] == "FAIL_OFFLINE_ENGINEERING_AUDIT"


def test_nominal_hand_edge_cannot_replace_measured_slipping_body(episode):
    path = episode / "robot_state.npz"
    with np.load(path) as loaded:
        state = {k: loaded[k] for k in loaded.files}
    state["tool_body_pose_xyzw"][1:, 0] += .005
    state["body_pose"][1:, 0, 0] += .005
    np.savez(path, **state)
    result = audit.audit_episode(episode)
    assert not result["measured_edge"]["pass"]
    assert result["trace_state_alignment"]["status"] == "FAIL"
    assert result["status"] == "FAIL_OFFLINE_ENGINEERING_AUDIT"


def test_equivalent_native_box_collider_preserves_shared_tool_geometry(episode):
    path = episode / "final_live_scene.usda"
    stage = Usd.Stage.Open(str(path))
    stage.RemovePrim(TOOL + "/Handle")
    cube = UsdGeom.Cube.Define(stage, TOOL + "/Handle")
    cube.CreateSizeAttr(2.)
    cube.AddScaleOp().Set(Gf.Vec3f(.008, .011, .020))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    assert audit.audit_episode(episode)["usd"]["checks"]["compound_colliders_match_owned_geometry"]
    cube.GetSizeAttr().Set(3.)
    stage.GetRootLayer().Save()
    assert not audit.audit_episode(episode)["usd"]["checks"]["compound_colliders_match_owned_geometry"]


@pytest.mark.parametrize("failure", ["zero_impulse", "stale", "blade_point", "wrong_arm"])
def test_contact_summary_cannot_override_raw_impulse_time_point_or_body(failure):
    row = contact_row()
    report = row["contacts"]["contacts"][0]
    if failure == "zero_impulse": report["raw_impulse_world_ns"] = [0., 0., 0.]
    if failure == "stale": report["time_s"] -= .1
    if failure == "blade_point": report["position_world_m"][2] += .07
    if failure == "wrong_arm": report["body1"] = report["body1"].replace("Franka_0", "Franka_1")
    result = audit.native_contacts(row, TOOL, np.array([.008, .011, .020]), [HAND], .1)
    assert row["contacts"]["bilateral_loaded_handle_contact"]  # Deliberately false summary.
    assert not result["bilateral_loaded"]


def test_wrong_com_lever_arm_and_direct_hand_load_fail_independently():
    row = {"applied_edge_material_wrench_world": [0., 2., 0., 0., 0., 0.],
        "applied_external_edge_wrench_world": [0.] * 6, "force_application_point_world_m": [.1, 0., 0.],
        "tool_com_before_step_world_m": [0., 0., 0.], "applied_total_wrench_about_com_world": [0., 2., 0., 0., 0., .2],
        "wrench_interval_dt_s": .002, "direct_hand_wrench_injection": False}
    assert audit.wrench_audit([row])["status"] == "PASS_NUMERIC_ROUTING"
    wrong_sign = {**row, "applied_total_wrench_about_com_world": [0., 2., 0., 0., 0., -.2]}
    assert not audit.wrench_audit([wrong_sign])["checks"]["com_lever_arm"]
    hand_load = {**row, "direct_hand_wrench_injection": True}
    assert audit.wrench_audit([hand_load])["status"] == "FAIL"


@pytest.mark.parametrize("stale_flag", ["sensor_valid", "sensor_fresh", "new_physics_step"])
def test_advancing_simulation_clock_cannot_hide_stale_contact_views(episode, stale_flag):
    path = episode / "physical_grasp_trace_00.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["contacts"][stale_flag] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    json_file(episode / "result.json", {"status": "PASS_SMOKE"})
    result = audit.audit_episode(episode)
    assert result["state"]["pass"]  # All control clocks and poses still look valid.
    assert result["pregrasp"]["status"] == "PASS_PREGRASP_ENGINEERING"
    assert result["formal_contact_freshness"]["invalid_substeps"] == 1
    assert result["status"] == "FAIL_OFFLINE_ENGINEERING_AUDIT"
    assert "rejected_recorded_pass" in result


def test_lab_missing_hand_trace_remains_unverified_even_if_relative_pose_looks_perfect():
    body = contact_row()["tool_body_pose_xyzw"]
    relative = make_tool_spec().hand_from_tool_xyz_xyzw
    rows = [{"settling_time_s": i / 10, "tool_body_pose_xyzw": body,
        "relative_tool_in_hand_xyzw": relative, "max_tool_rigid_penetration_m": .00001,
        "left_handle_normal_n": 2., "right_handle_normal_n": 2., "bilateral_substep_fraction": 1.,
        "contact_pairs": [{"shape0": TOOL + "/Handle", "shape1": HAND + "/panda_" + role + "finger",
                           "force_on_tool_world_n": [0., 2., 0.]} for role in ["left", "right"]]} for i in range(1, 21)]
    result = audit.lab_hold_audit(rows, ToolGraspConfig().to_dict(), HAND)
    assert result["status"] == "NOT_RUN_INCOMPLETE_HAND_EVIDENCE"
    assert not result["raw_hand_poses_available"]


def test_source_lifecycle_catches_camera_rigid_deactivation_despite_fresh_metadata(episode):
    saved = episode / "implementation"
    (saved / "camera_rig.py").write_text("def author_camera_rig():\n    prim.SetActive(False)\n")
    (saved / "isaac_sim_runner.py").write_text("class Runtime:\n    def reset(self):\n        SimulationManager.initialize_physics()\n        self._create_cameras()\n")
    config = {"camera_layout": "wrist_d405_global_d455"}
    assert audit.native_lifecycle_audit(episode, config)["status"] == "FAIL_UNSAFE_POST_PHYSICS_RIG_MUTATION"
    (saved / "isaac_sim_runner.py").write_text("class Runtime:\n    def reset(self):\n        self.pre_authored_camera_specs=author_camera_rig()\n        SimulationManager.initialize_physics()\n        self._create_cameras()\n")
    assert audit.native_lifecycle_audit(episode, config)["status"] == "PASS_KNOWN_LIFECYCLE_CONTRACT"


def test_cxx_view_errors_reject_fresh_python_contact_flags_and_finite_cached_state(episode):
    (episode / "stdout.log").write_text("[Error] [omni.physx.tensors.plugin] Simulation view object is invalidated and cannot be used again to call getVelocities\n")
    json_file(episode / "result.json", {"status": "PASS_SMOKE"})
    result = audit.audit_episode(episode)
    assert result["formal_contact_freshness"]["status"] == "PASS_FORMAL_CONTACT_FRESHNESS"
    assert result["state"]["pass"]
    assert result["runtime_logs"]["status"] == "FAIL_INVALIDATED_PHYSICS_VIEW"
    assert result["status"] == "FAIL_OFFLINE_ENGINEERING_AUDIT"
    assert "rejected_recorded_pass" in result


def test_raw_lab_contacts_and_nonzero_com_wrench_are_checked_without_summary_flags(episode):
    spec = make_tool_spec()
    with np.load(episode / "robot_state.npz") as data:
        state = {k: data[k] for k in data.files}
    n, dt = 1100, .002
    pose = state["tool_body_pose_xyzw"][0]
    edge = compose_pose(pose, spec.tool_from_edge_xyz_xyzw)
    center = pose[:3] + spec.com_tool_m
    payload = np.array([1., 0., 0., 0., 0., 0.])
    expected = np.r_[payload[:3], np.cross(edge[:3] - center, payload[:3])]
    applied = np.zeros((n, 6)); applied[1000:] = expected
    path = episode / "tool_substeps.npz"
    raw = dict(rigid_time_s=np.arange(1, n + 1) * dt,
        phase=np.array(["settling"] * 1000 + ["formal_episode"] * 100),
        tool_body_pose_xyzw=np.tile(pose, (n, 1)), hand_pose_xyzw=np.tile(state["hand_pose_xyzw"][0], (n, 1)),
        tool_velocity_com_world=np.zeros((n, 6)), finger_q=np.full((n, 2), .011), finger_qdot=np.zeros((n, 2)),
        applied_wrench_com_world=applied, applied_at_tool_pose_xyzw=np.tile(pose, (n, 1)),
        hand_external_wrench_world=np.zeros((n, 6)), contact_offsets=np.arange(n + 1) * 2,
        shape0=np.tile([0, 2], n), shape1=np.tile([1, 0], n),
        contact_wrench_on_shape0_world=np.tile([[0., 1., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]], (n, 1)),
        point_world=np.zeros((2 * n, 3)), normal_world=np.tile([[0., 1., 0.], [0., 1., 0.]], (n, 1)),
        friction=np.zeros((2 * n, 5)), distance_m=np.full(2 * n, -.0001))
    metadata = {"shape_body": [0, 1, 2], "tool_body_index": 0, "left_finger_body_index": 1,
        "right_finger_body_index": 2, "handle_shape_indices": [0], "tool_mass_kg": spec.mass_kg,
        "tool_com_local_m": spec.com_tool_m.tolist(), "tool_inertia_com_local_kg_m2": spec.inertia_com_tool_kg_m2.tolist(), "dt_s": dt}
    np.savez(path, **raw); json_file(path.with_suffix(".json"), metadata)
    setup = {"settling_runtime_origin_s": 2., "grasp": ToolGraspConfig().to_dict()}
    diagnostics = [{"native_float32_wrench_payload_world": payload}] * 2
    result = audit.lab_substeps_audit(path, spec.to_dict(), state, setup, diagnostics)
    assert result["status"] == "PASS_RAW_SUBSTEP_ENGINEERING"
    assert result["pregrasp"]["bilateral_loaded_handle_fraction"] == 1.
    assert result["wrench"]["nonzero_load_records"] == 100
    raw["applied_wrench_com_world"][-1, 4] *= -1
    np.savez(path, **raw)
    assert audit.lab_substeps_audit(path, spec.to_dict(), state, setup, diagnostics)["wrench"]["status"] == "FAIL"
    raw["hand_external_wrench_world"][-1, 0] = .1
    np.savez(path, **raw)
    assert not audit.lab_substeps_audit(path, spec.to_dict(), state, setup, diagnostics)["checks"]["no_hand_external_wrench"]


def test_invalid_evidence_writes_failure_receipt_instead_of_success_or_serialization_crash(episode, monkeypatch):
    path = episode / "robot_state.npz"
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    state["time_s"][-1] = np.nan
    np.savez(path, **state)
    output = episode.parent / "invalid_audit"
    monkeypatch.setattr(audit.sys, "argv", [str(SCRIPT), str(episode), str(output)])
    assert audit.main() == 1
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "FAIL_INVALID_OR_UNSUPPORTED_EVIDENCE"
    assert not result["full_physics_acceptance"]


def test_explicit_contact_geometry_must_match_before_pose_and_current_minus_dt():
    previous, row = contact_row(.1), contact_row(.2, "coupled_episode")
    row["tool_body_pose_before_step_xyzw"] = previous["tool_body_pose_xyzw"]
    row["wrench_interval_dt_s"] = .1
    sensor = row["contacts"]
    sensor.update(contact_geometry_frame="PRE_STEP_MEASURED_TOOL_POSE",
                  contact_geometry_tool_pose_xyzw=previous["tool_body_pose_xyzw"], contact_geometry_time_s=.1)
    for item in sensor["contacts"]:
        item["position_tool_m"] = (np.asarray(item["position_world_m"]) - np.asarray(previous["tool_body_pose_xyzw"][:3])).tolist()
    spec = make_tool_spec().to_dict()
    good = audit.contact_geometry_schema_audit([previous, row], spec, [HAND])
    assert good["status"] == "PASS_EXPLICIT_PRESTEP_CONTACT_GEOMETRY"
    assert good["bilateral_loaded_handle_fraction"] == 1.
    sensor["contact_geometry_time_s"] = .2
    assert audit.contact_geometry_schema_audit([previous, row], spec, [HAND])["status"] == "FAIL"
