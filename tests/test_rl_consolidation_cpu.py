"""Phase 1 consolidation checks that need neither Torch nor Isaac Lab (project ``.venv``): the workload B grip
contact profile refusals of ``open_vector_env(backend="isaac_lab_newton")`` (raised before Isaac Lab is imported) and
the receipt record, and ``tasks=`` entries for configured scenes (``ConfiguredTaskFamily``), including the refused
mix of seeded-TRIM and configured-scene slots whose batched layouts differ."""
import pytest

from robotic_cutting_sim.batched import lab_scene
from robotic_cutting_sim.rl import TaskFamily, open_vector_env
from robotic_cutting_sim.rl import vector_env as VE
from robotic_cutting_sim.rl.configured_family import (FAMILY_ID, NATIVE_CUT_V7_2_REFERENCE, ConfiguredTaskFamily)
from robotic_cutting_sim.rl.lab_vector_env import check_grip_contact_profile, grip_profile_record
from robotic_cutting_sim.rl.vector_env import UnsupportedVectorConfiguration

FAM = TaskFamily(scene_seeds=(512, 15), particle_spacing_m=.0025, max_episode_steps=3)


def _open(**kw):
    return open_vector_env(2, task=FAM, backend="isaac_lab_newton", device="cuda", **kw).__enter__()


def test_grip_contact_profile_refusals_before_isaac_lab():
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_GRIP_CONTACT_PROFILE"):
        _open(tool_mount="physical_grasp_v1", grip_contact_profile="mjwarp_grip_v2")
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_GRIP_CONTACT_PROFILE"):
        _open(tool_mount="physical_grasp_v1", grip_contact_profile=3)
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TOOL_MOUNT"):
        _open(grip_contact_profile=lab_scene.MJWARP_GRIP_V1)                  # B1 rigid mount (default)
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
        open_vector_env(2, task=FAM, device="cpu", parity_mode=True,
                        grip_contact_profile=lab_scene.MJWARP_GRIP_V1).__enter__()   # workload A has no grasp
    for profile in lab_scene.GRIP_PROFILES:                     # every declared profile is accepted with B2
        assert check_grip_contact_profile(profile, "physical_grasp_v1") == (profile, "physical_grasp_v1")
    assert check_grip_contact_profile(None) == ("legacy_point_contact", "rigid_mount_v1")


def test_grip_profile_record_lists_the_mjwarp_solver_options():
    record = grip_profile_record(lab_scene.MJWARP_GRIP_V1, "physical_grasp_v1")
    solver = record["contact_solver_profile"]
    assert record["grip_contact_profile"] == "mjwarp_grip_v1" and record["applies"] is True
    assert solver["mjwarp_disable_bits"] == ("NATIVECCD",) and solver["impratio"] == 10.
    assert solver["physical_calibration"].startswith("NOT_VALIDATED")
    default = grip_profile_record("legacy_point_contact", "rigid_mount_v1")
    assert default["contact_solver_profile"] is None and default["applies"] is False


def test_tasks_configured_scene_entries_resolve_to_a_configured_family():
    resolved = VE.resolve_tasks(3, None, [{"scene": "cut_v7_2_v1", "seed": 3},
                                          {"family_id": FAMILY_ID, "scene": "cut_v7_2_v1"},
                                          {"scene": "cut_v7_2_v1", "seed": 0}])
    assert isinstance(resolved, ConfiguredTaskFamily) and resolved.num_envs == 3
    assert resolved.env_seeds == (3, resolved.sim_seed, 0) and resolved.workload_id == "cut_v7_2_v1"
    assert resolved.native_reference["id"] == NATIVE_CUT_V7_2_REFERENCE["id"]
    base = ConfiguredTaskFamily.from_cut_workload(2, particle_spacing_m=.0025, max_episode_steps=4)
    kept = VE.resolve_tasks(2, base, [{"scene": "cut_v7_2_v1"}, {"scene": "cut_v7_2_v1", "seed": 9}])
    assert kept.particle_spacing_m == .0025 and kept.max_episode_steps == 4 and kept.env_seeds == (base.sim_seed, 9)
    # a configured scene path/dict needs its goal; the workload id brings its own
    custom = VE.resolve_tasks(1, None, [{"scene": base.scenes[0], "goal": base.goals[0],
                                         "planner_options": base.planner_options[0]}])
    assert custom.workload_id is None and custom.native_reference["status"] == "NOT_RUN"
    bad = ([{"scene": "cut_v7_2_v1", "goal": base.goals[0]}],         # workload declares its goal
           [{"scene": base.scenes[0]}],                               # path without goal
           [{"scene": "cut_v7_2_v1", "family_id": "agentic_trim_id"}],
           [{"scene": "cut_v7_2_v1", "scene_seed": 512}],
           [{"scene": "cut_v7_2_v1", "geometry": "x"}])
    for items in bad:
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TASK"):
            VE.resolve_tasks(1, None, items)
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TASK"):
        VE.resolve_tasks(1, FAM, [{"scene": "cut_v7_2_v1"}])          # TRIM TaskFamily base with a CUT entry
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TASK"):
        VE.resolve_tasks(1, base, [512])                              # configured base with a TRIM entry


def test_mixed_trim_and_cut_slots_are_refused_by_the_batched_layout():
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_NONUNIFORM_BATCH") as error:
        VE.resolve_tasks(2, None, [512, {"scene": "cut_v7_2_v1"}])
    message = str(error.value)
    for field in ("max_cuts", "duration_s", "reward_kind"):
        assert field in message
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_NONUNIFORM_BATCH"):
        VE.resolve_tasks(2, FAM, [{"scene": "cut_v7_2_v1", "seed": 1}, 15])
