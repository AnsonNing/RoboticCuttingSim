"""Numerical profile provenance and the Reference default at the public task boundary."""
import dataclasses

import numpy as np
import pytest

from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily, UnsupportedRubberConfiguration
from robotic_cutting_sim.rubber_material import p_wave_speed


@pytest.mark.parametrize(("profile", "n_t"), [("rubber_fast_v1", 3), ("rubber_turbo_v1", 2)])
def test_speed_profile_uses_effective_material_and_contact_labels(profile, n_t):
    ref = RubberStripTaskFamily(["L100_t4", "L150_t4"])
    task = RubberStripTaskFamily(ref.env_slots, speed_profile=profile)
    sample = task.sample(4000, "L100_t4")
    reference = ref.sample(4000, "L100_t4")
    assert sample["draws"] == reference["draws"]
    np.testing.assert_array_equal(sample["goal_m"], reference["goal_m"])
    assert task.spec("L100_t4").n_t == n_t
    assert task.spec("L100_t4").thickness == ref.spec("L100_t4").thickness
    assert task.spec("L100_t4").ppc == 2
    assert task.control_dt == ref.control_dt == .02
    assert task.build_material().beta == task.step_material.beta == sample["material"].beta
    assert sample["material"].friction_table == .3
    receipt = task.receipt()
    assert receipt["profile"]["id"] == profile
    assert receipt["envelope_rows"][str(n_t)]["status"] == "NOT_ESTABLISHED"
    assert receipt["resolution"]["per_n_t"][str(n_t)]["measured_bias"]["status"] == "NOT_MEASURED"
    assert receipt["effective_material"]["bulk_ratio"] == sample["material"].beta
    with pytest.raises(ValueError, match="bulk ratio"):
        inconsistent = dataclasses.replace(sample["material"], kappa=20 * sample["material"].mu0)
        task.build_life("L100_t4", material=inconsistent)
    with pytest.raises(UnsupportedRubberConfiguration, match="fixes CFL and n_t"):
        task.build_life("L100_t4", cfl=0.5)


def test_reference_defaults_and_schedule_remain_the_declared_v2():
    family = RubberStripTaskFamily(["L100_t4", "L150_t4"])
    assert family.profile["id"] == "rubber_rl_env_v2"
    assert family.profile["cfl"] == 0.5
    assert family.profile["state_precision"] == "f64_shadow_v1"
    assert family.profile["dirichlet"] == "consistent_mass_projection_v1"
    assert family.profile["constraint_work"] == "kick_average_v1"
    assert family.spec("L100_t4").resolution_profile == "rubber_ref_v1"
    assert family.spec("L100_t4").grid_dx() == .001
    with np.errstate(all="raise"):
        live = family.build_life("L100_t4", device="cpu")
    assert live.substeps == 5226
    assert live.dt == .02 / 5226
    assert live.dt / live.scene.dx * p_wave_speed(
        family.step_material.mu0, family.step_material.kappa, family.step_material.density) <= 0.5
    assert family.receipt()["envelope_rows"]["4"]["status"] == "ESTABLISHED"
    assert family.receipt()["resolution"]["per_n_t"]["4"]["bias_measured_at_this_n_t"]
    assert family.sample(4000, "L100_t4")["material"].beta == 20


def test_speed_profile_refuses_legacy_resolution_and_off_lattice_geometry():
    with pytest.raises(UnsupportedRubberConfiguration, match="requires the v2 task"):
        RubberStripTaskFamily(["L100_t4"], speed_profile="rubber_fast_v1", resolution="rubber_rl_v1")
    with pytest.raises(UnsupportedRubberConfiguration, match="unknown slots"):
        RubberStripTaskFamily(["L125_t4"], speed_profile="rubber_fast_v1")
