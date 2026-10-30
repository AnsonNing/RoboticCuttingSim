"""WP-12 Phase 4c: opt-in geometry providers (heightfield_v2, sdf_csg_experimental_v1) and materials_wp12_v1."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim import geometry_providers as gp
from robotic_cutting_sim import materials_wp12 as mw
from robotic_cutting_sim.benchmark import BenchmarkSpec, Bump, HeightFieldSpec, MaterialParameters, sample_specimen
from robotic_cutting_sim.scene_config import (SceneConfigError, SceneInfeasible, compile_task, scene_material,
                                              validate_scene)

ROOT = Path(__file__).resolve().parents[1]
SEVER = json.loads((ROOT/"configs/wp11/goals/sever_mid.json").read_text())
TRIM = json.loads((ROOT/"configs/wp11/goals/trim_bump_rho10.json").read_text())


def small_scene(**workpiece):
    scene = json.loads((ROOT/"configs/wp12/scenes/trim01_flat_board.json").read_text())
    scene["scene_id"] = "p4c_test"
    scene["workpiece"]["base_box_m"] = {"length_m": .04, "width_m": .016, "height_m": .008}
    scene["workpiece"]["protrusions"] = [{"id": "p0", "family": "dome", "x_m": 0., "y_m": 0., "radius_x_m": .006,
                                          "radius_y_m": .004, "height_m": .004}]
    scene["workpiece"].update(workpiece)
    return scene


# ----------------------------------------------------------------------------- default path unchanged

@pytest.mark.parametrize("name", ["bar_flat_board", "trim01_flat_board", "bar_three_bumps_flat_board"])
def test_default_scenes_carry_no_provider_key(name):
    scene = validate_scene(ROOT/f"configs/wp12/scenes/{name}.json")
    assert "geometry_provider" not in scene["workpiece"]


def test_explicit_heightfield_v1_normalizes_to_the_default():
    plain = validate_scene(small_scene())
    explicit = validate_scene(small_scene(geometry_provider={"id": "heightfield_v1"}))
    assert plain == explicit
    with pytest.raises(SceneConfigError):
        validate_scene(small_scene(geometry_provider={"id": "heightfield_v1", "surface_noise": {}}))


def test_v1_rows_reject_v2_keys_and_families():
    scene = small_scene()
    scene["workpiece"]["protrusions"][0]["yaw_deg"] = 10.
    with pytest.raises(SceneConfigError):
        validate_scene(scene)
    scene = small_scene()
    scene["workpiece"]["protrusions"][0]["family"] = "irregular"
    with pytest.raises(SceneConfigError):
        validate_scene(scene)


def test_heightfield_v2_without_features_samples_the_v1_particles_exactly():
    v1 = compile_task(small_scene(), TRIM)
    v2 = compile_task(small_scene(geometry_provider={"id": "heightfield_v2"}), TRIM)
    assert isinstance(v2.geometry, gp.HeightFieldSpecV2) and not isinstance(v1.geometry, gp.ProviderGeometry)
    for a, b in zip(v1.sample_particles(), v2.sample_particles()):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    reference = sample_specimen(BenchmarkSpec("x", 42, v1.geometry, v1.material), .001)
    assert np.array_equal(reference[0], v2.sample_particles()[0])


# ----------------------------------------------------------------------------- heightfield_v2 features

def test_oriented_bump_identity_yaw_offset_matches_bump():
    x, y = np.meshgrid(np.linspace(-.02, .02, 41), np.linspace(-.01, .01, 21))
    for family in gp.V2_FAMILIES:
        a = Bump(.003, -.001, .008, .005, .004, family).height(x, y)
        b = gp.OrientedBump(.003, -.001, .008, .005, .004, family).height(x, y)
        assert np.array_equal(a, b)


def test_oriented_bump_yaw_and_skew():
    x, y = np.meshgrid(np.linspace(-.02, .02, 401), np.linspace(-.02, .02, 401))
    plain = gp.OrientedBump(0., 0., .012, .004, .005, "dome")
    turned = gp.OrientedBump(0., 0., .012, .004, .005, "dome", yaw_deg=90.)
    assert np.allclose(turned.height(x, y), plain.height(y, -x), atol=1e-12)
    assert np.allclose(turned.half_extents_xy(), (.004, .012))
    skewed = gp.OrientedBump(0., 0., .012, .004, .005, "dome", peak_offset=.5)
    peak = np.unravel_index(np.argmax(skewed.height(x, y)), x.shape)
    assert abs(x[peak]-.006) < 1e-4 and abs(skewed.height(x, y).max()-.005) < 1e-6
    foot = skewed.height(x, y) > 0   # two half-ellipses joined at the peak, inside the declared rectangle
    assert np.abs(x[foot]).max() < .012 and np.abs(y[foot]).max() < .004
    assert x[foot].min() < -.0118 and x[foot].max() > .0118
    rotated = gp.OrientedBump(0., 0., .012, .004, .005, "dome", yaw_deg=30., peak_offset=.5)
    hx, hy = rotated.half_extents_xy()
    foot = rotated.height(x, y) > 0
    assert np.abs(x[foot]).max() <= hx and np.abs(y[foot]).max() <= hy
    with pytest.raises(ValueError):
        gp.OrientedBump(peak_offset=.9)


def test_surface_noise_bounded_and_seeded():
    def spec(seed):
        return gp.HeightFieldSpecV2(.08, .024, .012, (), surface_noise_amplitude_m=.0005, surface_noise_seed=seed)
    x, y = np.meshgrid(np.linspace(-.04, .04, 161), np.linspace(-.012, .012, 49))
    a, b, c = spec(3).noise(x, y), spec(3).noise(x, y), spec(4).noise(x, y)
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert np.abs(a).max() <= .0005+1e-15 and np.abs(a).max() > .0002
    assert np.allclose(spec(3).height(x, y), .012+a)


def test_v2_scene_with_features_compiles_and_validates_lattice():
    scene = small_scene(geometry_provider={"id": "heightfield_v2", "surface_noise": {"amplitude_m": .0004,
                                                                                    "wavelength_m": .01, "seed": 2}})
    scene["workpiece"]["protrusions"][0].update(yaw_deg=35., peak_offset=-.4, family="irregular")
    task = compile_task(scene, TRIM)
    row = task.scene["workpiece"]["protrusions"][0]
    assert row["yaw_deg"] == 35. and row["peak_offset"] == -.4
    points, masses, lattice = task.sample_particles()
    report = gp.lattice_report(lattice)
    assert report["ok"] and report["columns_touching_support_fraction"] == 1.
    assert np.allclose(masses, task.material.density_kg_m3*1e-9)
    assert points.dtype == np.float32 and lattice.dtype == np.int32


def test_rotated_bump_outside_base_is_infeasible_and_planner_uses_aabb():
    scene = small_scene(geometry_provider={"id": "heightfield_v2"})
    scene["workpiece"]["protrusions"][0].update(radius_x_m=.009, radius_y_m=.003, yaw_deg=90.)
    with pytest.raises(SceneInfeasible) as error:
        validate_scene(scene)
    assert error.value.reason == "protrusion_outside_base"
    from robotic_cutting_sim.goal_planner import _support_along
    row = {"radius_x_m": .01, "radius_y_m": .002}
    assert _support_along(row, np.array([1., 0., 0.])) == .01
    assert np.isclose(_support_along({**row, "yaw_deg": 90.}, np.array([1., 0., 0.])), .002)


def test_noise_needs_two_layers_under_its_lowest_point():
    scene = small_scene(geometry_provider={"id": "heightfield_v2", "surface_noise": {"amplitude_m": .007,
                                                                                    "wavelength_m": .01, "seed": 1}})
    with pytest.raises(SceneInfeasible) as error:
        validate_scene(scene)
    assert error.value.reason == "spacing_too_coarse"


# ----------------------------------------------------------------------------- sdf_csg_experimental_v1

def test_sdf_tunnel_is_one_body_with_a_hole():
    scene = small_scene(protrusions=[], geometry_provider={"id": "sdf_csg_experimental_v1", "subtract": [
        {"shape": "cylinder", "axis": "y", "center_m": [0., .004], "radius_m": .002}]})
    task = compile_task(scene, SEVER)
    points, _, lattice = task.sample_particles()
    assert not (((points[:, 0])**2+(points[:, 2]-.004)**2) <= .002**2).any()
    full = 40*16*8
    assert 0 < full-len(points) < full and gp.lattice_report(lattice)["ok"]
    assert gp.capability_receipt(task.geometry)["experimental"]


def test_sdf_refusals():
    split = small_scene(protrusions=[], geometry_provider={"id": "sdf_csg_experimental_v1", "subtract": [
        {"shape": "box", "center_m": [0., 0., .004], "half_extents_m": [.001, .02, .02]}]})
    with pytest.raises(SceneInfeasible) as error:
        validate_scene(split)
    assert error.value.reason == "disconnected_material"
    with pytest.raises(SceneConfigError):   # protrusions are not part of the CSG provider
        validate_scene(small_scene(geometry_provider={"id": "sdf_csg_experimental_v1", "subtract": [
            {"shape": "sphere", "center_m": [0., 0., .008], "radius_m": .002}]}))
    with pytest.raises(SceneConfigError):
        validate_scene(small_scene(protrusions=[], geometry_provider={"id": "sdf_csg_experimental_v1", "subtract": [
            {"shape": "torus", "center_m": [0., 0., 0.]}]}))


def test_lattice_report_flags_floating_particles():
    lattice = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 3]])
    report = gp.lattice_report(lattice)
    assert not report["ok"] and report["floating_particles"] == 1 and "disconnected_material" in report["problems"]


def test_workpiece_yaw_needs_a_corotated_flat_board():
    scene = small_scene()
    scene["workpiece"]["world_pose"]["yaw_deg"] = 30.
    with pytest.raises(SceneInfeasible) as error:
        validate_scene(scene)
    assert error.value.reason == "workpiece_yaw_differs_from_board"
    scene["supports"]["workcell"]["overrides"] = {"cutting_board": {"yaw_deg": 30., "center_xy_world_m": [0., .1],
                                                                    "length_x_m": .3, "width_y_m": .16}}
    scene["workpiece"]["world_pose"]["translation_m"] = [0., .1, .01]
    assert validate_scene(scene)["workpiece"]["world_pose"]["yaw_deg"] == 30.


# ----------------------------------------------------------------------------- materials_wp12_v1

def test_profiles_are_interpretable_provisional_parameter_sets():
    doc = mw.document()
    assert doc["status"] == "PROVISIONAL_UNCALIBRATED"
    reference = doc["profiles"]["wp12_reference_v1"]
    for name, profile in doc["profiles"].items():
        assert profile["interpretation"] and all(isinstance(profile[k], float) for k in mw.RECIPE_KEYS)
        assert not any("colo" in key for key in profile)
        if name != "wp12_reference_v1":
            assert any(profile[k] != reference[k] for k in mw.RECIPE_KEYS), name
        recipe = mw.recipe(name, MaterialParameters())
        assert recipe.calibration == "provisional_uncalibrated" and recipe.hardening_ratio == 0.


def test_reference_profile_equals_the_frozen_cutting_tasks_v1_recipe():
    frozen = scene_material(validate_scene(small_scene()))
    profile = scene_material(validate_scene(small_scene(material_profile="wp12_reference_v1")))
    assert frozen == profile


def test_stability_rule_refuses_stiff_at_100us():
    assert np.isclose(mw.wave_speed(5e4, .3, 1500.), 6.6987, atol=1e-3)
    with pytest.raises(mw.MaterialStabilityError):
        mw.admit("wp12_stiff_v1", .004, 1e-4)
    assert mw.admit("wp12_stiff_v1", .004, 5e-5)["grid_cfl"] < .2
    for name in mw.profiles():
        if name != "wp12_stiff_v1":
            assert mw.admit(name, .004, 1e-4, .001)["admitted"], name


def test_configured_env_refuses_an_unstable_material_dt_before_building():
    from robotic_cutting_sim.scene_config import make_configured_kinematic_env
    task = compile_task(small_scene(material_profile="wp12_stiff_v1"), TRIM)
    with pytest.raises(mw.MaterialStabilityError):
        make_configured_kinematic_env(task, device="cpu", material_dt=1e-4)


def test_unknown_material_profile_is_rejected():
    with pytest.raises(SceneConfigError):
        validate_scene(small_scene(material_profile="wp12_rubber_v9"))
