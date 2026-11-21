"""The existing coarse numerics are reachable through both headless task families."""
import numpy as np
import pytest

from robotic_cutting_sim.rl import ConfiguredTaskFamily, TaskFamily
from robotic_cutting_sim.rl.vector_env import compile_family


@pytest.mark.parametrize("kind", ["cut", "trim"])
def test_batched_opt_in_research_fast_profile(kind):
    if kind == "cut":
        full = ConfiguredTaskFamily.from_cut_workload(1)
        fast = ConfiguredTaskFamily.from_cut_workload(1, runtime_profile="research_fast_v1")
        reference, candidate = full.compile_tasks()[0], fast.compile_tasks()[0]
    else:
        full = TaskFamily(scene_seeds=(512,))
        fast = TaskFamily(scene_seeds=(512,), runtime_profile="research_fast_v1")
        reference, candidate = compile_family(full)[0], compile_family(fast)[0]
    assert (reference.particle_spacing_m, reference.grid_dx_m, full.material_dt_s) == (0.001, 0.004, 0.0001)
    assert (candidate.particle_spacing_m, candidate.grid_dx_m, fast.material_dt_s) == (0.002, 0.008, 0.0002)
    assert candidate.goal == reference.goal
    assert candidate.control_dt_s == reference.control_dt_s
    assert candidate.task_id != reference.task_id
    np.testing.assert_array_equal(candidate.reference_edge_pose_world(0), reference.reference_edge_pose_world(0))
    assert fast.to_dict()["runtime_profile"] == "research_fast_v1"
    assert "runtime_profile" not in full.to_dict()


def test_batch_runtime_profile_rejects_conflicting_numerics():
    with pytest.raises(ValueError, match="fixes.*particle_spacing_m"):
        ConfiguredTaskFamily.from_cut_workload(1, runtime_profile="research_fast_v1", particle_spacing_m=.003)
    with pytest.raises(ValueError, match="fixes material_dt"):
        TaskFamily((512,), runtime_profile="research_fast_v1", material_dt_s=.0005)
