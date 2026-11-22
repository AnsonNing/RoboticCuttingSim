"""The existing coarse numerics are reachable through both headless task families."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.rl import ConfiguredTaskFamily, TaskFamily
from robotic_cutting_sim.rl.vector_env import compile_family
from robotic_cutting_sim.runtime_profiles import compile_profiled_trim_seed
from robotic_cutting_sim.scene_config import task_from_payload
from robotic_cutting_sim.seeded_trim import compile_seed, digest_array


def test_public_cutting_fidelity_uses_existing_headless_profiles(monkeypatch):
    from robotic_cutting_sim.rl import vector_env

    def stub(num_envs, **options):
        return SimpleNamespace(close=lambda: None, **options)

    monkeypatch.setattr(vector_env, "VectorCuttingEnv", stub)
    task = ConfiguredTaskFamily.from_cut_workload(1)
    with vector_env.open_vector_env(1, task=task, fidelity="reference") as reference:
        assert reference.task is task
    with vector_env.open_vector_env(1, task=task, fidelity="turbo") as turbo:
        assert turbo.task.runtime_profile == "research_fast_4mm_v1"
        assert turbo.task.compile_tasks()[0].grid_dx_m == .004
        assert turbo.task.material_dt_s == .0002
        assert turbo.particle_contact_mode == "alias_free_v2"
    with pytest.raises(NotImplementedError, match="turbo"), vector_env.open_vector_env(
            1, task=task, backend=vector_env.LAB_BACKEND, fidelity="turbo"):
        pytest.fail("Unsupported Isaac Lab fidelity succeeded")


def test_public_native_fidelity_rejects_low_cadence_before_launch():
    from robotic_cutting_sim.live_env import open_cutting_env

    with pytest.raises(ValueError, match="Native Vision fidelity"), open_cutting_env(
            output_dir="unused", record_every=10, fidelity="reference"):
        pytest.fail("Native fidelity accepted a reduced recording rate")


@pytest.mark.parametrize("kind", ["cut", "trim"])
@pytest.mark.parametrize(("profile", "grid_dx"), [
    ("research_fast_v1", 0.008),
    ("research_fast_4mm_v1", 0.004),
])
def test_batched_opt_in_research_fast_profile(kind, profile, grid_dx):
    if kind == "cut":
        full = ConfiguredTaskFamily.from_cut_workload(1)
        fast = ConfiguredTaskFamily.from_cut_workload(1, runtime_profile=profile)
        reference, candidate = full.compile_tasks()[0], fast.compile_tasks()[0]
    else:
        full = TaskFamily(scene_seeds=(512,))
        fast = TaskFamily(scene_seeds=(512,), runtime_profile=profile)
        reference, candidate = compile_family(full)[0], compile_family(fast)[0]
    assert (reference.particle_spacing_m, reference.grid_dx_m, full.material_dt_s) == (0.001, 0.004, 0.0001)
    assert (candidate.particle_spacing_m, candidate.grid_dx_m, fast.material_dt_s) == (0.002, grid_dx, 0.0002)
    assert candidate.goal == reference.goal
    assert candidate.control_dt_s == reference.control_dt_s
    assert candidate.task_id != reference.task_id
    np.testing.assert_array_equal(candidate.reference_edge_pose_world(0), reference.reference_edge_pose_world(0))
    assert fast.to_dict()["runtime_profile"] == profile
    assert "runtime_profile" not in full.to_dict()


def test_batch_runtime_profile_rejects_conflicting_numerics():
    with pytest.raises(ValueError, match="fixes.*particle_spacing_m"):
        ConfiguredTaskFamily.from_cut_workload(1, runtime_profile="research_fast_v1", particle_spacing_m=.003)
    with pytest.raises(ValueError, match="fixes material_dt"):
        TaskFamily((512,), runtime_profile="research_fast_v1", material_dt_s=.0005)


@pytest.mark.parametrize("profile", ["research_fast_v1", "research_fast_4mm_v1"])
def test_native_trim_receipt_hashes_effective_resolution(tmp_path, profile):
    options = dict(policy_seed=0, sim_seed=42, distribution="id")
    reference, original = compile_seed(512, **options)
    out = tmp_path / "scene"
    task, manifest = compile_profiled_trim_seed(
        512, name=profile, output_dir=out, **options)
    replay = task_from_payload(json.loads((out / "compiled_task.json").read_text()))
    points, masses, lattice = replay.sample_particles()
    assert manifest == json.loads((out / "scene_manifest.json").read_text())
    assert task.config_sha256 == replay.config_sha256 == manifest["task_hash"]
    assert task.goal == reference.goal
    assert manifest["geometry_hash"] == original["geometry_hash"]
    assert manifest["material_hash"] == original["material_hash"]
    assert manifest["solver_hash"] != original["solver_hash"]
    assert manifest["acceptance_valid"] is False
    assert manifest["initial_particles_sha256"] == digest_array(points)
    assert manifest["initial_masses_sha256"] == digest_array(masses)
    assert manifest["initial_lattice_sha256"] == digest_array(lattice)
