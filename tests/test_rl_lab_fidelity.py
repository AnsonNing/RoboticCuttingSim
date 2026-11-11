"""Review finding 3 (Phase 1 xhigh acceptance review): workload B ``fidelity()`` reflects the material time step and
every other physics-relevant non-default (``rl.lab_vector_env.lab_fidelity``). CPU, no Isaac Lab: the pure function
the live ``LabVectorCuttingEnv.fidelity()`` calls; the live path is exercised in the Isaac Lab venv by
``tests/test_rl_lab_grip_profile.py::test_open_vector_env_passes_mjwarp_grip_v1_to_the_lab_env`` (dt 2e-4)."""
import pytest

from robotic_cutting_sim.batched.lab_scene import BatchedLabOptions
from robotic_cutting_sim.rl.lab_vector_env import LAB_PHYSICS_DEFAULTS, lab_fidelity
from robotic_cutting_sim.rl.vector_env import FIDELITY_REFERENCE_KEYS, NATIVE_TRIM_V7_2_REFERENCE


def reference_actual(**changes):
    ref = NATIVE_TRIM_V7_2_REFERENCE
    actual = dict(solver="tlmpm_v7_2", **{k: ref[k] for k in FIDELITY_REFERENCE_KEYS})
    actual.update(changes)
    return actual


def b2_scene():
    return BatchedLabOptions(tool_mount="physical_grasp_v1", grip_contact_profile="mjwarp_grip_v1", device="cpu",
                             cpu_test_mode=True, material="tlmpm_v7_2").validate().fidelity()


def defaults():
    return dict(LAB_PHYSICS_DEFAULTS)


def test_recorded_b2_configuration_stays_full_fidelity():
    """Every recorded B2 row: dt 1e-4, 200 substeps, defaults -> not reduced (the listed differences remain)."""
    fid = lab_fidelity(b2_scene(), reference_actual(), defaults(), declared_episode_steps=1200)
    assert fid["reduced_fidelity"] is False and fid["reasons"] == []
    assert fid["listed_differences"]["grip_contact_profile"] == "mjwarp_grip_v1"
    assert fid["computed_not_hand_set"] is True and fid["fidelity_version"] == "lab_fidelity_v2"


@pytest.mark.parametrize("dt, substeps", [(2e-4, 100), (5e-4, 40)])
def test_non_default_material_dt_is_reduced_never_silently_full(dt, substeps):
    """The review's scenario: B2 + mjwarp_grip_v1 with material_dt_s 2e-4 / 5e-4 (100 / 40 substeps)."""
    fid = lab_fidelity(b2_scene(), reference_actual(material_dt_s=dt, substeps_per_control_step=substeps),
                       defaults(), declared_episode_steps=1200)
    assert fid["reduced_fidelity"] is True
    assert f"material_dt_s={dt} (reference 0.0001)" in fid["reasons"]
    assert f"substeps_per_control_step={substeps} (reference 200)" in fid["reasons"]
    assert fid["actual"]["material_dt_s"] == dt and fid["reference"]["material_dt_s"] == 1e-4


@pytest.mark.parametrize("change, reason", [
    (dict(particle_spacing_m=.0025), "particle_spacing_m=0.0025 (reference 0.001)"),
    (dict(grid_dx_m=.008), "grid_dx_m=0.008 (reference 0.004)"),
    (dict(control_dt_s=.01), "control_dt_s=0.01 (reference 0.02)"),
    (dict(cutting_law=7), "cutting_law=7 (reference 8)"),
    (dict(episode_steps=600), "episode_steps=600 (reference 1200)"),
])
def test_every_reference_key_difference_is_a_reason(change, reason):
    fid = lab_fidelity(b2_scene(), reference_actual(**change), defaults(), declared_episode_steps=1200)
    assert fid["reduced_fidelity"] is True and reason in fid["reasons"]


@pytest.mark.parametrize("option, value, reason", [
    ("feedback", False, "reaction_not_fed_back"),
    ("ik_iterations", 8, "ik_iterations=8 (default 48)"),
    ("velocity_feedforward", False, "velocity_feedforward=False (default True)"),
])
def test_non_default_robot_or_coupling_options_are_reasons(option, value, reason):
    options = dict(defaults(), **{option: value})
    fid = lab_fidelity(b2_scene(), reference_actual(), options, declared_episode_steps=1200)
    assert fid["reduced_fidelity"] is True and reason in fid["reasons"]
    assert fid["non_default_options"] == {option: value}


def test_b1_rigid_mount_keeps_its_scene_reasons():
    scene = BatchedLabOptions(device="cpu", cpu_test_mode=True, material="tlmpm_v7_2").validate().fidelity()
    fid = lab_fidelity(scene, reference_actual(material_dt_s=2e-4, substeps_per_control_step=100), defaults(),
                       declared_episode_steps=1200)
    assert fid["reasons"][:2] == ["rigid_tool_mount_no_grasp", "finger_opening_fixed_nominal"]
    assert "material_dt_s=0.0002 (reference 0.0001)" in fid["reasons"]


def test_lab_physics_defaults_are_the_options_defaults():
    options = BatchedLabOptions()
    assert {k: getattr(options, k) for k in LAB_PHYSICS_DEFAULTS} == LAB_PHYSICS_DEFAULTS
