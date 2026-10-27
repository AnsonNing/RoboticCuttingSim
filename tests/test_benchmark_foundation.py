"""Analytic CPU controls; none of these fixtures constitutes a physics experiment."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import label

from robotic_cutting_sim.benchmark import (
    BenchmarkSpec, Bump, HeightFieldSpec, MaterialParameters, MetricConfig,
    analyze_recording, compare_recordings, sample_specimen,
)


@pytest.mark.parametrize("variant", ["D0", "D1", "D2", "D3"])
def test_seeded_recipes_sample_one_connected_unsegmented_material(variant):
    spec = BenchmarkSpec.generate(42, variant)
    assert spec == BenchmarkSpec.generate(42, variant)
    q, mass, indices, receipt = sample_specimen(spec, .003)
    occupancy = np.zeros(indices.max(0)+1, bool)
    occupancy[tuple(indices.T)] = True
    assert label(occupancy)[1] == 1
    assert np.all(q[:, 2] <= spec.heightfield.height(q[:, 0], q[:, 1])+1e-7)
    assert np.isclose(mass.sum(), len(q)*spec.material.density_kg_m3*.003**3)
    assert not receipt["preassigned_components"] and not receipt["benchmark_ready"]
    assert receipt == sample_specimen(spec, .003)[3]
    for i, j, k in indices[q[:, 2] > spec.heightfield.base_height_m]:
        assert occupancy[i, j, :k+1].all()


def test_variants_change_only_the_documented_recipe_families():
    fixed = BenchmarkSpec.generate(1, "D0")
    assert fixed.heightfield == BenchmarkSpec.generate(2, "D0").heightfield
    for variant in ("D1", "D2", "D3"):
        assert BenchmarkSpec.generate(1, variant) != BenchmarkSpec.generate(2, variant)
    assert len(BenchmarkSpec.generate(42, "D2").heightfield.bumps) == 3
    changed = BenchmarkSpec.generate(42, "D3")
    assert changed.material != fixed.material
    assert changed.material.calibration == "provisional_uncalibrated"
    assert changed.material.failure_model == "not_configured"
    assert changed.camera_translation_m != (0, 0, 0)


def test_overlapping_bumps_form_a_height_union_over_the_same_base():
    bump = Bump(radius_x_m=.010, radius_y_m=.010, height_m=.005)
    h = HeightFieldSpec(bumps=(bump, replace(bump, height_m=.008)))
    assert h.height(0, 0) == pytest.approx(.023)
    assert h.height(.03, 0) == pytest.approx(.015)


@pytest.mark.parametrize("factory", [
    lambda: Bump(height_m=-.1), lambda: HeightFieldSpec(bumps=(Bump(x_m=.059),)),
    lambda: MaterialParameters(friction=-1), lambda: MaterialParameters(poisson_ratio=.5),
    lambda: MaterialParameters(fracture_energy_j_m2=float("nan")),
    lambda: MetricConfig(surface_cell_size_m=0),
])
def test_invalid_scenarios_and_metric_units_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def toy_recording(path, **updates):
    base = np.array([[-.5, -.5, .75], [.5, -.5, .75],
                     [-.5, .5, .75], [.5, .5, .75], [-.5, -.5, 1.25]])
    data = {"positions": np.stack([base, base]), "velocities": np.zeros((2, 5, 3)),
            "mass_kg": np.ones(5), "time_s": np.array([0., 1.]),
            "tool_pose_xyzw": np.array([[0, 0, 5, 0, 0, 0, 1]]*2),
            "tool_half_extents": np.array([.1, .2, .3])}
    data.update(updates)
    np.savez(path, **data)
    return path


TOY_CONFIG = MetricConfig(length_m=2, width_m=2, target_height_m=1,
                          density_kg_m3=2, surface_cell_size_m=1,
                          particle_support_radius_m=.25)


def test_surface_and_outside_mass_are_measured_without_inventing_a_chip(tmp_path):
    path = toy_recording(tmp_path/"recording.npz")
    with np.load(path) as state:
        q = state["positions"].copy()
    q[-1, -1, 0] = 3
    path = toy_recording(path, positions=q)
    result = analyze_recording(path, TOY_CONFIG)
    assert result["initial_surface"]["residual_height_proxy_m"] == pytest.approx(.5)
    assert result["final_surface"]["residual_height_proxy_m"] == 0
    assert result["final_surface"]["overcut_volume_proxy_m3"] == 0
    assert result["mass"]["outside_xy_footprint_mass_kg"] == 1
    assert result["mass"]["outside_xy_footprint_reference_volume_m3"] == .5
    assert "not an independent" in result["mass"]["temporal_measurement"]
    assert result["components"]["detached_mass_kg"] is None
    assert result["formal_cutting_success"] is None and not result["benchmark_ready"]


def test_one_empty_surface_bin_has_explicit_gouging_proxy(tmp_path):
    path = toy_recording(tmp_path/"recording.npz")
    with np.load(path) as state:
        q = state["positions"].copy()
    # Empty the upper-right 1 m² bin. This is an oracle particle-bin control.
    q[-1, 3, 0] = 3
    result = analyze_recording(toy_recording(path, positions=q), TOY_CONFIG)
    assert result["final_surface"]["occupied_bin_fraction"] == .75
    assert result["final_surface"]["overcut_volume_proxy_m3"] == 1


def test_component_partition_com_momentum_and_unknown_mass(tmp_path):
    path = toy_recording(tmp_path/"recording.npz", mass_kg=np.arange(1., 6),
                         component_ids=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
                         component_provenance=np.array("analytic test graph, not a physics validation"),
                         substrate_anchor_mask=np.array([True, True, True, True, False]))
    result = analyze_recording(path, TOY_CONFIG)
    components = result["components"]
    assert components["substrate_mass_kg"] == 10 and components["detached_mass_kg"] == 5
    assert components["partition_residual_kg"] == 0 and components["detached_component_count"] == 1
    chip = next(row for row in components["largest_components"] if row["id"] == 1)
    assert chip["center_of_mass_m"] == [-.5, -.5, 1.25]
    assert chip["linear_momentum_kg_m_s"] == [0, 0, 0]
    assert result["final_surface"]["residual_height_proxy_m"] == 0
    assert components["permanent_separation_proven"] is None

    unknown = toy_recording(tmp_path/"unknown.npz", component_ids=np.array([0, 0, 0, 0, -1]),
                            component_provenance=np.array("analytic test graph"))
    result = analyze_recording(unknown, TOY_CONFIG)
    assert result["components"]["unclassified_mass_kg"] == 1
    assert result["components"]["detached_mass_kg"] == 0
    # Unknown material stays in the height estimate instead of being hidden.
    assert result["final_surface"]["residual_height_proxy_m"] == .5


def test_side_codes_or_unattributed_components_do_not_establish_separation(tmp_path):
    for extra in ({"cut_codes": np.array([1, 1, 1, 1, 2])},
                  {"component_ids": np.array([0, 0, 0, 0, 1])}):
        path = toy_recording(tmp_path/f"case{len(extra)}_{next(iter(extra))}.npz", **extra)
        result = analyze_recording(path, TOY_CONFIG)
        assert result["components"]["status"] == "NOT_AVAILABLE"
        assert result["components"]["detached_mass_kg"] is None
        assert result["final_surface"]["residual_height_proxy_m"] == .5


def test_per_frame_mass_loss_is_visible_and_not_called_removed_material(tmp_path):
    mass = np.ones((2, 5))
    mass[-1, -1] = .5
    result = analyze_recording(toy_recording(tmp_path/"mass.npz", mass_kg=mass), TOY_CONFIG)
    assert result["mass"]["max_relative_change"] == pytest.approx(.1)
    assert result["mass"]["temporal_measurement"] == "per-frame masses recorded"
    assert result["components"]["detached_mass_kg"] is None


def test_tool_box_proxy_respects_rotation_and_reports_its_scope(tmp_path):
    q = np.array([[[.15, 0, .5]], [[.15, 0, .5]]])
    tool = np.array([[0, 0, .5, 0, 0, np.sqrt(.5), np.sqrt(.5)]]*2)
    path = toy_recording(tmp_path/"rotated.npz", positions=q, velocities=np.zeros_like(q),
                         tool_pose_xyzw=tool, mass_kg=np.ones(1))
    result = analyze_recording(path, TOY_CONFIG)
    assert result["tool_box_proxy_particle_center_penetration_m"] == pytest.approx(.05)
    assert "bounding-box proxy" in result["penetration_scope"]


@pytest.mark.parametrize("updates", [
    {"time_s": np.array([0., 0.])}, {"mass_kg": np.full(5, -1.)},
    {"velocities": np.full((2, 5, 3), np.nan)},
    {"component_ids": np.array([0, 0, 0, 0, 1.5]), "component_provenance": np.array("invalid floats")},
    {"tool_pose_xyzw": np.zeros((2, 7))},
])
def test_corrupted_recordings_fail_loudly(tmp_path, updates):
    with pytest.raises(ValueError):
        analyze_recording(toy_recording(tmp_path/"bad.npz", **updates), TOY_CONFIG)


def test_comparison_checks_inputs_but_does_not_rank_or_assert_formal_success(tmp_path):
    a, b = toy_recording(tmp_path/"a.npz"), toy_recording(tmp_path/"b.npz")
    result = compare_recordings({"baseline": a, "candidate": b}, TOY_CONFIG,
                                {"baseline": {"wall_s": 2}})
    assert all(result["recorded_inputs_match"].values())
    assert result["runs"]["baseline"]["simulation_s_per_reported_wall_s"] == .5
    assert result["runs"]["candidate"]["simulation_s_per_reported_wall_s"] is None
    assert result["solver_ranking"] is None and not result["benchmark_ready"]
    assert result["formal_paired_protocol"].startswith("NOT_ESTABLISHED")
