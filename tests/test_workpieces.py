"""Initial-volume and unchanged-score controls, with no solver-success claim."""
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.benchmark import Bump, HeightFieldSpec
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.physical_tool import make_tool_spec
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.workpieces import FootprintCuttingTaskSpec, WorkpieceSpec, with_workpiece_footprint


def reference_stencil(points, dx):
    # Pure-CPU trilinear reference control; no solver rollout or cut labels.
    origin = np.floor(points.min(0) / dx) * dx - 2 * dx
    cell = np.floor((points - origin) / dx).astype(int)
    alpha = (points - origin) / dx - cell
    bits = np.array([[(corner >> axis) & 1 for axis in range(3)] for corner in range(8)])
    xyz = cell[:, None] + bits
    dims = xyz.max((0, 1)) + 1
    nodes = (xyz[:, :, 0] * dims[1] + xyz[:, :, 1]) * dims[2] + xyz[:, :, 2]
    factors = np.where(bits[None], alpha[:, None], 1 - alpha[:, None])
    weights = factors.prod(-1)
    gradients = np.stack([(2 * bits[None, :, axis] - 1) / dx
                         * np.prod(factors[:, :, [k for k in range(3) if k != axis]], axis=-1)
                         for axis in range(3)], axis=-1)
    return nodes, weights, gradients


def synthetic_capsule_recording(task_id):
    # Deliberately small analytic graph fixture, never an actual cutting result.
    geometry = HeightFieldSpec(.024, .012, .008,
                    () if task_id == "CUT-01" else (Bump(0, 0, .004, .004, .004),))
    task = replace(make_task(task_id, workpiece_kind="capsule", particle_spacing_m=.002),
                   geometry=geometry, target_origin_local_m=(0, 0, 0 if task_id == "CUT-01" else .008))
    points, masses, _ = task.sample_particles()
    nodes, weights, gradients = reference_stencil(points, task.grid_dx_m)
    side = points[:, 0] < 0 if task_id == "CUT-01" else points[:, 2] <= .008
    final_codes = np.where(side, 1, 2)
    return task, {"time_s": np.array([0., 5.6, 8.]), "positions": np.stack([points] * 3),
                  "reference_positions": points, "mass_kg": masses,
                  "codes": np.stack([np.zeros(len(points), int), final_codes, final_codes]),
                  "reference_stencil_nodes": nodes, "reference_stencil_weights": weights,
                  "reference_stencil_gradients": gradients}


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
@pytest.mark.parametrize("variant", ["fixed", "randomized"])
def test_rectangle_legacy_serialization_and_all_particle_bytes_remain_frozen(task_id, variant):
    # asdict hashes were captured on the pre-workpiece implementation. Unlike a
    # package-version receipt they do not depend on the installed NumPy version.
    golden = {
        ("CUT-01", "fixed"): ("7f18ef96a9e749969d69da77df08caa2f7ea897a8813055bab7acfe11fd0eece",
            "9d98780b0ece29ad03bf309f0d511690684102fe39b6d060825e89e2919e0004", "9a0487e0122aa43c305d17fc6640cfa03aa1cf7c2c606df5a41687fcc3ca4e97", "fcfbdd72e1b25d6c52b367615a8a511a08e448800a876555d52cc56b98eafe78"),
        ("CUT-01", "randomized"): ("fbe36936e32179f35557f74a4519ab3ceaec47a4df731cba2daf9773d6b597df",
            "fcd769c891785def4d7275bf1fa0f8b8e2ec6ac56e7b33da625f7a38cd8eeee9", "95ed75e675241b1dbf46eae5a425d82eced6bbddc49f8cf0e3a74023cb501db6", "e0747982c07474ada39cb8af4137cc824c77af3a8b53aff9498b76d6f9c41793"),
        ("TRIM-01", "fixed"): ("5ce10c3a0f83aab8e7440a14ecdd66d2de30830fd47bc97398d5d380c0958040",
            "feeeba6ef6e00c0ba3920a841ac8b180fc801c7068d9404667a29825fee2215b", "b389ee50253b28a761354614eb4d33319305310d7493ebc4460b5c807be5ea51", "6d3c11295ea4e4b5c0d464b91ca81edffba5ec1de7a8250309bf71c7d2bf9772"),
        ("TRIM-01", "randomized"): ("0c1c6d668534c36009f2666fe8230a6de1cabdc38bcc06d5634ff8b4ad982890",
            "47408e47be8f51d827e1882a2a6550d48744ea887c3ab3c4efb7efa59c6a2b71", "26c6f78a84640ef9bb23e372a0f80161e4015577e79d9970755b8f3e68b4339f", "8aa492f2103528c7fe5d4b7c7f44cd5fc738dbae16f2d045bd728da2bf0515b6"),
    }
    task = make_task(task_id, seed=42, variant=variant)
    signature = hashlib.sha256(json.dumps(asdict(task), sort_keys=True).encode()).hexdigest()
    samples = [hashlib.sha256(array.tobytes()).hexdigest() for array in task.sample_particles()]
    assert (signature, *samples) == golden[(task_id, variant)]
    assert task.workpiece_kind == "rectangle" and "workpiece_kind" not in task.to_dict()
    assert "workpiece_kind" not in asdict(task)


def test_capsule_footprint_is_rounded_not_the_original_rectangular_volume():
    workpiece = WorkpieceSpec(HeightFieldSpec(.16, .03, .02, ()), footprint="capsule")
    inside = workpiece.contains_xy([[0, .015], [.08, 0], [.075, .012], [.065, .014]])
    np.testing.assert_array_equal(inside, [True, True, False, True])
    assert workpiece.height(.075, .012) == 0 and workpiece.height(0, 0) == .02
    expected = ((.16 - .03) * .03 + np.pi * .015**2) * .02
    assert workpiece.analytic_base_volume_m3 == pytest.approx(expected)
    assert workpiece.analytic_base_volume_m3 < .16 * .03 * .02
    points, masses, lattice = workpiece.sample_particles(.002)
    assert len(points) == 11520 and masses.sum() == pytest.approx(.13824)
    assert np.all(masses == 1500 * .002**3)
    assert len(np.unique(lattice, axis=0)) == len(points)
    assert workpiece.contains_xy(points[:, :2]).all()
    assert abs(masses.sum() - expected * 1500) / (expected * 1500) < .01
    receipt = workpiece.to_dict()
    assert not receipt["component_labels_supplied"] and not receipt["runtime_geometry_deletion"]
    assert receipt["analytic_total_volume_m3"] == pytest.approx(expected)
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_initial_capsule_has_one_actual_reference_stencil_component_with_no_presegmentation(task_id):
    task = make_task(task_id, workpiece_kind="capsule", particle_spacing_m=.002)
    points, masses, lattice = task.sample_particles()
    nodes, weights, gradients = reference_stencil(points, task.grid_dx_m)
    count, _ = mechanical_components(nodes, weights, np.zeros(len(points), int), 1, gradients=gradients)
    assert count == 1
    assert len(task.sample_particles()) == 3
    assert len(points) == len(masses) == len(lattice)
    assert isinstance(task, FootprintCuttingTaskSpec)
    assert replace(task, seed=7).workpiece_kind == "capsule"
    assert task.to_dict()["workpiece"]["footprint"] == "capsule"
    original = with_workpiece_footprint(task, "rectangle")
    assert "workpiece_kind" not in asdict(original)
    assert original.sample_particles()[1].sum() > masses.sum()


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_seeded_world_pose_fixtures_targets_and_tool_interface_remain_independent(task_id):
    world_pose = (.3, -.2, .4, *Rotation.from_euler("xyz", [15, -25, 35], degrees=True).as_quat())
    task = make_task(task_id, seed=43, variant="randomized", particle_spacing_m=.002,
                     world_from_workpiece_xyzw=world_pose, workpiece_kind="capsule")
    rect = make_task(task_id, seed=43, variant="randomized", particle_spacing_m=.002,
                     world_from_workpiece_xyzw=world_pose)
    assert task.geometry == rect.geometry and task.material == rect.material
    assert task.fixture_boxes_local == rect.fixture_boxes_local
    np.testing.assert_allclose(task.T_world_workpiece, rect.T_world_workpiece)
    points, mass, _ = task.sample_particles()
    np.testing.assert_allclose(task.workpiece.points_to_local(task.workpiece.points_to_world(points)), points, atol=1e-15)
    np.testing.assert_allclose(task.reference_edge_pose_world(4), rect.reference_edge_pose_world(4))
    np.testing.assert_allclose(task.target_plane_world(), rect.target_plane_world())
    assert mass.sum() < rect.sample_particles()[1].sum()
    tool, original_tool = make_tool_spec(task), make_tool_spec(rect)
    np.testing.assert_allclose(tool.T_tool_edge, original_tool.T_tool_edge)
    for shape, original in zip(tool.shapes, original_tool.shapes):
        np.testing.assert_array_equal(shape.vertices_tool_m, original.vertices_tool_m)
    assert task.to_dict()["randomization"]["draws"] == rect.to_dict()["randomization"]["draws"]


def test_capsule_cut_uses_same_severing_gates_and_rejects_rectangular_reference():
    task, recording = synthetic_capsule_recording("CUT-01")
    result = task.evaluate_oracle(recording)
    assert result["status"] == "PASS_TASK_METRICS"
    assert result["checks"]["reference_matches_single_continuous_task_sampler"]
    assert not result["original_rectangular_benchmark"] and not result["full_simulation_acceptance"]
    assert result["metrics"]["initial_components"] == 1 and result["metrics"]["final_components"] == 2
    rectangular_task = replace(task, workpiece_kind="rectangle")
    assert rectangular_task.evaluate_oracle(recording)["status"] == "FAIL_TASK_METRICS"


def test_capsule_trim_reports_unsupported_surface_profile_with_independent_topology_screen():
    task, recording = synthetic_capsule_recording("TRIM-01")
    # Even an apparently perfect rectangular plane cannot score capsule surface.
    recording.update(surface_vertices_world=np.array([[-1, -1, .008], [1, -1, .008], [0, 1, .008]]),
                     surface_triangles=np.array([[0, 1, 2]]), surface_triangle_component_ids=np.array([0]),
                     surface_component_provenance="Synthetic fixture", surface_timestamp_s=8.)
    result = task.evaluate_oracle(recording)
    assert result["status"] == "NOT_RUN_UNSUPPORTED_SURFACE_FOOTPRINT"
    assert result["particle_topology_status"] == "PASS_PARTICLE_TOPOLOGY_METRICS"
    assert "surface" not in result["metrics"]
    uncut = {**recording, "codes": np.zeros_like(recording["codes"])}
    negative = task.evaluate_oracle(uncut)
    assert negative["status"] == "NOT_RUN_UNSUPPORTED_SURFACE_FOOTPRINT"
    assert negative["particle_topology_status"] == "FAIL_PARTICLE_TOPOLOGY_METRICS"
    assert not negative["checks"]["reference_above_plane_material_removed"]


def test_invalid_geometry_or_unknown_imported_shape_fails_explicitly():
    with pytest.raises(ValueError):
        make_task(workpiece_kind="imported_stl")
    with pytest.raises(ValueError):
        WorkpieceSpec(HeightFieldSpec(.01, .03, .01, ()), "capsule")
    with pytest.raises(ValueError):
        WorkpieceSpec(HeightFieldSpec(.03, .01, .01, ()), "capsule", (0, 0, 0, 1, 0, 0, 1))
    with pytest.raises(ValueError):
        make_task(workpiece_kind="capsule").workpiece.contains_xy([np.inf, 0])
