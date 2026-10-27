from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.benchmark import Bump, HeightFieldSpec
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.scene_frames import compose_pose, pose_errors
from robotic_cutting_sim.tasks import FixtureBox, make_task


def reference_stencil(points, dx):
    """Analytic trilinear reference control, not a solver rollout."""
    origin = np.floor(points.min(0)/dx)*dx-2*dx
    cell = np.floor((points-origin)/dx).astype(int)
    alpha = (points-origin)/dx-cell
    offsets = np.stack(np.meshgrid(*[[0, 1]]*3, indexing="ij"), -1).reshape(-1, 3)
    index = cell[:, None, :]+offsets
    dimensions = index.max((0, 1))+1
    nodes = (index[:, :, 0]*dimensions[1]+index[:, :, 1])*dimensions[2]+index[:, :, 2]
    factors = np.where(offsets[None] == 1, alpha[:, None], 1-alpha[:, None])
    weights = factors.prod(-1)
    gradients = np.stack([(2*offsets[None, :, axis]-1)/dx*
                          np.prod(factors[:, :, [i for i in range(3) if i != axis]], axis=-1)
                          for axis in range(3)], -1)
    return nodes, weights, gradients


def small_recording(kind="CUT-01"):
    # Smaller analytic controls retain the same sampling/goal/frame semantics;
    # they never stand in for the factory's actual160mm live workpiece.
    if kind == "CUT-01":
        geometry = HeightFieldSpec(.016, .008, .008, ())
        task = replace(make_task(kind), geometry=geometry, particle_spacing_m=.002)
    else:
        geometry = HeightFieldSpec(.016, .012, .008, (Bump(0, 0, .003, .003, .004),))
        task = replace(make_task(kind), geometry=geometry, target_origin_local_m=(0, 0, .008))
    points, mass, _ = task.sample_particles()
    nodes, weights, gradients = reference_stencil(points.astype(float), task.grid_dx_m)
    side = points[:, 0] < 0 if kind == "CUT-01" else points[:, 2] < .008
    final_codes = np.where(side, 1, 2).astype(int)
    data = {"time_s": np.array([0., 5.6, 8.]), "positions": np.stack([points]*3),
        "reference_positions": points, "mass_kg": mass, "codes": np.stack([np.zeros(len(points), int), final_codes, final_codes]),
        "reference_stencil_nodes": nodes, "reference_stencil_weights": weights, "reference_stencil_gradients": gradients}
    return task, data


def test_cut_bar_cell_volume_and_dimensions_are_exact_without_presegmentation():
    task = make_task("CUT-01")
    points, mass, lattice = task.sample_particles()
    assert len(points) == 96000
    np.testing.assert_allclose(points.min(0)-.0005, [-.08, -.015, 0], atol=5e-9)
    np.testing.assert_allclose(points.max(0)+.0005, [.08, .015, .02], atol=5e-9)
    assert mass.sum() == pytest.approx(.16*.03*.02*1500, abs=1e-14)
    assert lattice.shape == points.shape
    assert task.geometry.bumps == ()
    assert len(task.sample_particles()) == 3  # No supplied cut/component labels.


def test_trim_is_one_base_union_with_bump_and_fixed_seed_has_no_hidden_randomness():
    first, second = make_task("TRIM-01", seed=42), make_task("TRIM-01", seed=43)
    for a, b in zip(first.sample_particles(), second.sample_particles()):
        assert np.array_equal(a, b)
    assert first.geometry.height(0, 0) == pytest.approx(.028)
    assert first.geometry.height(.04, 0) == pytest.approx(.020)
    assert first.material.friction == .3 and first.material.hardening_ratio == 0


def test_cut_mount_keeps_downward_hand180mm_above_edge_including_full_plunge():
    task = make_task("CUT-01")
    for time in [0, 1.6, 3., 5.6, 8.]:
        hand, edge = task.reference_hand_pose_world(time), task.reference_edge_pose_world(time)
        np.testing.assert_allclose(hand[:3]-edge[:3], [0, 0, .18], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(hand[3:]).as_matrix(), np.diag([1, -1, -1]), atol=1e-12)
        pe, re = pose_errors(compose_pose(hand, task.hand_from_edge_xyz_xyzw), edge)
        assert pe < 1e-12 and re < 1e-12
    assert task.reference_hand_pose_world(5.6)[2] == pytest.approx(.177)
    assert task.blade_length_m == .035


def test_blade_clears_cut_support_slot_and_table_at_full_plunge():
    task = make_task("CUT-01")
    left, right = task.fixture_boxes_local
    assert left.center_m[0]+left.half_extents_m[0] == pytest.approx(-.020)
    assert right.center_m[0]-right.half_extents_m[0] == pytest.approx(.020)
    edge = task.reference_edge_pose_local(5.6)
    # Lower rear z in knife frame projects into workpieceX, inside the40mm slot.
    local_vertices = np.array([[-task.blade_length_m, 0, -.0004], [0, 0, 0], [-task.blade_length_m, 0, task.blade_height_m]])
    world_vertices = Rotation.from_quat(edge[3:]).apply(local_vertices)+edge[:3]
    assert np.max(np.abs(world_vertices[:, 0])) < .02
    assert world_vertices[:, 2].min()-task.table_top_world_z_m == pytest.approx(.007)
    assert world_vertices[:, 2].max() > task.geometry.base_height_m


def test_trim_target_and_schedule_are_explicit_workpiece_local():
    task = make_task("TRIM-01")
    assert task.reference_edge_pose_local(1.6)[2] == pytest.approx(.0202)
    assert task.reference_edge_pose_local(5.6)[0] == pytest.approx(.045)
    assert [task.active_cut_index(t) for t in [0, 1.6, 5.599, 5.6, 8]] == [-1, 0, 0, -1, -1]
    assert task.control_dt == task.control_dt_s == .02
    assert not task.reference_action(6.4)["cut_enabled"]


def test_rotated_translated_task_plane_and_mount_compose_without_world_z_assumption():
    pose = (1., -.2, .3, *Rotation.from_euler("y", 35, degrees=True).as_quat())
    task = make_task("CUT-01", world_from_workpiece_xyzw=pose)
    point = task.points_to_world([0, .01, .015])
    assert task.target_plane_world()[:3]@point+task.target_plane_world()[3] == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(task.points_to_local(point), [0, .01, .015], atol=1e-12)
    error = pose_errors(compose_pose(task.reference_hand_pose_world(4), task.hand_from_edge_xyz_xyzw), task.reference_edge_pose_world(4))
    assert max(error) < 1e-12


@pytest.mark.parametrize("shape", [(7,), (2, 7)])
def test_joint_action_has_explicit_active_or_both_arm_semantics(shape):
    result = make_task().validate_action({"mode": "joint_position_rad", "target": np.zeros(shape), "cut_enabled": False})
    assert result["target"].shape == shape and not result["cut_enabled"]


def test_actions_reject_ambiguous_orientation_force_or_nonboolean_arming():
    task = make_task()
    with pytest.raises(ValueError):
        task.validate_action({"mode": "cartesian_edge_pose_world", "target": [0, 0, 0, 1, 0, 0, 1]})
    with pytest.raises(ValueError):
        task.validate_action({"mode": "joint_position_rad", "target": np.zeros(7), "force": [0, 0, 1]})
    with pytest.raises(ValueError):
        task.validate_action({"mode": "joint_position_rad", "target": np.zeros(7), "cut_enabled": 1})


def test_full_reference_cut_passes_oracle_metrics_without_claiming_live_execution():
    task, data = small_recording()
    result = task.evaluate_oracle(data)
    assert result["status"] == "PASS_TASK_METRICS"
    assert result["metrics"]["initial_components"] == 1
    assert result["metrics"]["final_components"] == 2
    assert result["metrics"]["substantial_parts"] == 2
    assert result["metrics"]["maximum_reference_midpoint_plane_distance_m"] <= .002
    assert not result["full_simulation_acceptance"] and not result["policy_observation"]
    assert json.loads(json.dumps(result))["status"] == "PASS_TASK_METRICS"


def test_short_smoke_is_not_scored_as_a_completed_task():
    task, data = small_recording()
    data["time_s"] = np.array([0, .02, .04])
    data["codes"][:] = 0
    result = task.evaluate_oracle(data)
    assert result["status"] == "NOT_RUN" and not result["checks"]["complete_episode"]
    assert json.loads(json.dumps(result))["status"] == "NOT_RUN"


def test_oracle_target_score_is_invariant_to_a_consistent_world_frame_change():
    task, data = small_recording()
    moved = replace(task, world_from_workpiece_xyzw=(.4, -.2, .3, *Rotation.from_euler("y", 40, degrees=True).as_quat()))
    data["reference_positions"] = moved.points_to_world(data["reference_positions"])
    data["positions"] = moved.points_to_world(data["positions"])
    assert moved.evaluate_oracle(data)["status"] == "PASS_TASK_METRICS"


def test_partial_depth_cut_leaves_actual_zero_code_stencil_bridge():
    task, data = small_recording()
    q = data["reference_positions"]
    partial = np.where(q[:, 2] > .004, np.where(q[:, 0] < 0, 1, 2), 0)
    data["codes"][1:] = partial
    result = task.evaluate_oracle(data)
    assert result["status"] == "FAIL_TASK_METRICS"
    assert result["metrics"]["final_components"] == 1


def test_real_severing_at_wrong_plane_does_not_pass_target_location():
    task, data = small_recording()
    data["codes"][1:] = np.where(data["reference_positions"][:, 0] < .002, 1, 2)
    result = task.evaluate_oracle(data)
    assert result["metrics"]["final_components"] == 2
    assert not result["checks"]["cut_at_workpiece_local_target_plane"]


def test_lost_mass_or_erased_history_cannot_pass_a_separation_picture():
    task, data = small_recording()
    data["mass_kg"] = data["mass_kg"]*.5
    data["codes"][-1, 0] = 0
    result = task.evaluate_oracle(data)
    assert not result["checks"]["reference_mass_budget_and_unchanged_count"]
    assert not result["checks"]["persistent_history_no_digit_rewrites"]


def test_missing_reference_gradients_cannot_be_replaced_by_side_codes():
    task, data = small_recording()
    del data["reference_stencil_gradients"]
    assert task.evaluate_oracle(data)["status"] == "NOT_RUN"


def test_trim_requires_actual_attributed_surface_and_preserves_flatness_failure():
    task, data = small_recording("TRIM-01")
    assert task.evaluate_oracle(data)["status"] == "NOT_RUN"
    _, labels = mechanical_components(data["reference_stencil_nodes"], data["reference_stencil_weights"], data["codes"][-1],
        1, gradients=data["reference_stencil_gradients"])
    substrate_id = labels[np.argmin(data["reference_positions"][:, 2])]
    vertices = np.array([[-.008, -.006, .008], [.008, -.006, .008], [.008, .006, .008], [-.008, .006, .008]])
    data.update(surface_vertices_world=vertices, surface_triangles=np.array([[0, 1, 2], [0, 2, 3]]),
        surface_triangle_component_ids=np.array([substrate_id]*2), surface_component_provenance="actual reference-stencil graph control",
        surface_timestamp_s=8.)
    assert task.evaluate_oracle(data)["status"] == "PASS_TASK_METRICS"
    data["surface_vertices_world"][:, 2] += .0015
    result = task.evaluate_oracle(data)
    assert result["status"] == "FAIL_TASK_METRICS"
    assert not result["checks"]["maximum_substrate_surface_residual_height_m"]


def test_invalid_task_variants_and_fixture_dimensions_are_rejected():
    with pytest.raises(ValueError):
        make_task("D0")
    with pytest.raises(ValueError):
        make_task(variant="random")
    with pytest.raises(ValueError):
        FixtureBox("bad", (0, 0, 0), (0, 1, 1))


def test_explicit_quadrature_preserves_nominal_bar_goal_and_immutable_mass_budget():
    original = make_task("TRIM-01")
    coarse = make_task("TRIM-01", particle_spacing_m=.002)
    assert coarse.geometry == original.geometry and coarse.material == original.material
    assert coarse.target_origin_local_m == original.target_origin_local_m
    assert coarse.fixture_boxes_local == original.fixture_boxes_local
    assert np.array_equal(coarse.reference_edge_pose_world(3.), original.reference_edge_pose_world(3.))
    points, mass, lattice = coarse.sample_particles()
    assert len(points) == 12136 and len(np.unique(lattice, axis=0)) == len(points)
    assert np.all(mass == coarse.material.density_kg_m3*.002**3)
    assert mass[points[:, 2] < .020].sum() == pytest.approx(.144, abs=1e-12)
    assert coarse.to_dict()["numerical_overrides"]["particle_spacing_m"] == {"protocol": .001, "actual": .002}
    assert original.particle_spacing_m == .001 and original.to_dict()["numerical_overrides"] == {}
