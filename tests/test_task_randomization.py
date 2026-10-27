"""Geometry, mass, frame covariance and legacy-output controls; no solver claims."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.benchmark import (BenchmarkSpec, Bump, FAMILIES, HeightFieldSpec,
    sample_specimen)
from robotic_cutting_sim.cut_topology import reference_components
from robotic_cutting_sim.scene_frames import (compose_pose, pose_errors, pose_matrix,
    transform_points, transform_poses)
from robotic_cutting_sim.tasks import make_task


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_seed_reproduces_complete_geometry_and_particles_and_changes_with_seed(task_id):
    a = make_task(task_id, 42, "randomized", particle_spacing_m=.002)
    b = make_task(task_id, 42, "randomized", particle_spacing_m=.002)
    assert asdict(a) == asdict(b)
    for x, y in zip(a.sample_particles(), b.sample_particles()):
        assert np.array_equal(x, y)
    assert asdict(a) != asdict(make_task(task_id, 43, "randomized", particle_spacing_m=.002))
    receipt = a.to_dict()
    assert receipt["randomization"]["seed"] == 42
    assert receipt["randomization"]["protocol_sha256"]
    assert not receipt["randomization"]["policy_observation_contains_force_or_oracle"]
    assert receipt["particle_spacing_m"] == .002
    assert json.loads(json.dumps(receipt, allow_nan=False))["variant"] == "randomized"


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_actual_sampler_has_one_continuous_base_unique_cells_and_reference_mass(task_id):
    task = make_task(task_id, 42, "randomized", particle_spacing_m=.002)
    points, mass, lattice = task.sample_particles()
    base = points[:, 2] < task.geometry.base_height_m
    expected_cells = round(task.geometry.length_m/.002)*15*10
    assert base.sum() == expected_cells
    assert mass[base].sum() == pytest.approx(task.geometry.length_m*.03*.02*1500)
    assert len(np.unique(lattice, axis=0)) == len(points)
    assert np.all(mass == 1500*.002**3)
    assert reference_components(points, lattice, np.zeros(len(points), int), 1)[0] == 1
    assert len(task.sample_particles()) == 3  # No initial side/component assignment.


def test_duplicate_overlapping_bumps_are_unioned_without_duplicate_particles_or_mass():
    task = make_task("TRIM-01", 42, "randomized", particle_spacing_m=.002)
    repeated = replace(task, geometry=replace(task.geometry, bumps=task.geometry.bumps*2))
    for first, second in zip(task.sample_particles(), repeated.sample_particles()):
        assert np.array_equal(first, second)


@pytest.mark.parametrize("task_id", ["CUT-01", "TRIM-01"])
def test_provided_frame_is_composed_and_targets_actions_fixtures_transform_together(task_id):
    initial = make_task(task_id, 42, "randomized")
    world_change = (.3, -.2, .4, *Rotation.from_euler("xyz", [20, -15, 30], degrees=True).as_quat())
    moved = make_task(task_id, 42, "randomized", world_from_workpiece_xyzw=world_change)
    np.testing.assert_allclose(moved.T_world_workpiece, pose_matrix(world_change)@initial.T_world_workpiece, atol=1e-14)
    assert moved.geometry == initial.geometry
    for time in [0., 1.6, 4., 5.6, 8.]:
        expected = transform_poses(pose_matrix(world_change), initial.reference_edge_pose_world(time))
        assert max(pose_errors(expected, moved.reference_edge_pose_world(time))) < 1e-12
        assert max(pose_errors(compose_pose(moved.reference_hand_pose_world(time), moved.hand_from_edge_xyz_xyzw),
                               moved.reference_edge_pose_world(time))) < 1e-12
    local_target = np.asarray(moved.target_origin_local_m)
    normal, distance = moved.target_plane_world()[:3], moved.target_plane_world()[3]
    assert normal@moved.points_to_world(local_target)+distance == pytest.approx(0, abs=1e-12)
    for a, b in zip(initial.fixture_boxes_local, moved.fixture_boxes_local):
        np.testing.assert_allclose(moved.points_to_world(b.corners_local()),
            transform_points(pose_matrix(world_change), initial.points_to_world(a.corners_local())), atol=1e-14)


def test_cut_random_positions_preserve_mass_fractions_open_slot_and_full_plunge():
    lengths, cuts = set(), set()
    for seed in range(100):
        task = make_task("CUT-01", seed, "randomized")
        length, cut = task.geometry.length_m, task.target_origin_local_m[0]
        lengths.add(length)
        cuts.add(cut)
        assert min(length/2-cut, length/2+cut)/length >= .4
        assert all(row[1] == cut for row in task.edge_position_knots_local_txyz)
        left, right = task.fixture_boxes_local
        assert left.center_m[0]+left.half_extents_m[0] == pytest.approx(cut-.02)
        assert right.center_m[0]-right.half_extents_m[0] == pytest.approx(cut+.02)
        assert left.center_m[0]-left.half_extents_m[0] == pytest.approx(-length/2-.01)
        assert right.center_m[0]+right.half_extents_m[0] == pytest.approx(length/2+.01)
        assert task.reference_edge_pose_local(5.6)[2] == -.003
    assert lengths == {.14, .16, .18} and cuts == set(np.arange(-8, 9, 2)*.001)


def test_trim_shapes_stay_inside_base_and_reference_sweep_covers_each_support():
    families, counts = set(), set()
    for seed in range(100):
        task = make_task("TRIM-01", seed, "randomized")
        g = task.geometry
        counts.add(len(g.bumps))
        start = task.reference_edge_pose_local(1.6)
        end = task.reference_edge_pose_local(5.6)
        assert start[2] == end[2] == .0202
        for bump in g.bumps:
            families.add(bump.family)
            assert abs(bump.x_m)+bump.radius_x_m < g.length_m/2
            assert abs(bump.y_m)+bump.radius_y_m < g.width_m/2
            assert start[0] <= bump.x_m-bump.radius_x_m
            assert end[0] >= bump.x_m+bump.radius_x_m
        assert start[0] >= -g.length_m/2+.002
        assert end[0] <= g.length_m/2-.002
        assert task.duration_s == 8 and task.control_dt_s == .02
    assert families == {"dome", "gaussian", "ridge", "irregular"} and counts == {1, 2, 3}


def test_irregular_family_is_compact_nonnegative_bounded_and_asymmetric():
    bump = Bump(family="irregular", radius_x_m=1, radius_y_m=1, height_m=.008)
    x, y = np.meshgrid(np.linspace(-1.1, 1.1, 101), np.linspace(-1.1, 1.1, 101))
    height = bump.height(x, y)
    assert height.min() == 0 and height.max() <= .008
    assert np.all(height[x*x+y*y >= 1] == 0)
    assert bump.height(0, 0) == .008
    assert bump.height(.4, .2) != pytest.approx(bump.height(-.4, -.2))
    assert FAMILIES == ("dome", "gaussian", "ellipsoid_cap", "ridge")


@pytest.mark.parametrize("case", [f"{variant}:{seed}" for variant in ["D0", "D1", "D2", "D3"] for seed in [42, 2026]])
def test_legacy_d0_to_d3_match_prechange_geometry_mass_lattice_and_spec_hashes(case):
    frozen = json.loads((Path(__file__).parent/"fixtures/benchmark_legacy_specimens_v1.json").read_text())
    variant, seed = case.split(":")
    spec = BenchmarkSpec.generate(int(seed), variant)
    _, _, _, receipt = sample_specimen(spec, frozen["spacing_m"])
    assert {key: receipt[key] for key in frozen["cases"][case]} == frozen["cases"][case]


@pytest.mark.parametrize("seed", [-1, True, 42.5])
def test_randomized_rejects_ambiguous_or_negative_seed(seed):
    with pytest.raises(ValueError):
        make_task("CUT-01", seed, "randomized")


def test_fixed_defaults_stay_unrandomized_and_random_receipt_frame_roundtrips():
    for name in ["CUT-01", "TRIM-01"]:
        fixed = make_task(name, 42)
        assert fixed.particle_spacing_m == .001 and "randomization" not in fixed.to_dict()
        for x, y in zip(fixed.sample_particles(), make_task(name, 43).sample_particles()):
            assert np.array_equal(x, y)
        moved = make_task(name, 42, "randomized", world_from_workpiece_xyzw=(.2, -.3, .4, *Rotation.from_euler("x", 20, degrees=True).as_quat()))
        rebuilt = make_task(name, 42, "randomized", world_from_workpiece_xyzw=moved.to_dict()["randomization"]["input_world_from_workpiece_xyzw"])
        assert max(pose_errors(moved.world_from_workpiece_xyzw, rebuilt.world_from_workpiece_xyzw)) < 1e-12
