"""P1-5 T5.4: reference_command_policy_v1 (compiled planner schedule -> edge_delta_v1 actions).

NumPy parts run in .venv; the Torch parts (device action path, device policy) need .deps/isaacsim-venv (plan D-09)
and are NOT_RUN where torch is missing. CPU only."""
import numpy as np
import pytest

from robotic_cutting_sim.rl import reference_policy as R
from robotic_cutting_sim.rl.cut_workload import compile_workload_task


@pytest.fixture(scope="module")
def cut():
    task = compile_workload_task()
    actions, reconstructed, schedule, report = R.reference_command_actions(task)
    return task, actions, reconstructed, schedule, report


def test_schedule_is_the_compiled_planner_schedule(cut):
    task, _, _, schedule, _ = cut
    assert schedule["steps"] == 700 and schedule["targets"].shape == (700, 7)
    dt = task.control_dt_s
    for k in (1, 5, 207, 350, 574, 700):
        np.testing.assert_array_equal(schedule["targets"][k - 1], task.reference_edge_pose_world(k * dt))
        assert schedule["resolved_cut"][k - 1] == task.active_cut_index((k - 1) * dt)
    knots = np.asarray(task.reference_cut_indices[:700])
    np.testing.assert_array_equal(schedule["resolved_cut"], knots)            # knot i -> i+1 motion digit
    np.testing.assert_array_equal(schedule["initial_pose"], task.reference_edge_pose_world(0.))
    assert set(np.unique(schedule["resolved_cut"])) == {-1, 0}
    assert schedule["labels"][0] == task.reference_phase_labels[0]


def test_reconstruction_residual_commands_and_digits(cut):
    task, actions, reconstructed, schedule, report = cut
    assert actions.shape == (700, 8) and actions.dtype == np.float32
    assert report["steps"] == 700 and report["policy"] == R.POLICY_REFERENCE
    assert report["max_position_diff_m"] <= 1e-9
    assert report["max_quaternion_component_diff"] <= 1e-8
    assert report["digit_schedule_equal"] and report["armed_steps"] == int((schedule["resolved_cut"] >= 0).sum())
    assert np.all(np.abs(actions[:, :6]) <= 1. + 1e-6) and np.isfinite(actions).all()
    armed = actions[:, 6] > 0
    np.testing.assert_array_equal(armed, schedule["resolved_cut"] >= 0)
    np.testing.assert_allclose(reconstructed[:, :3], schedule["targets"][:, :3], atol=1e-9, rtol=0)


def test_schedule_is_independent_of_particles_and_shorter_runs_are_prefixes(cut):
    task, actions, _, _, _ = cut
    coarse = compile_workload_task(particle_spacing_m=.002)      # numerical override: same plan
    a2, _, _, _ = R.reference_command_actions(coarse)
    np.testing.assert_array_equal(a2, actions)
    a3, _, _, _ = R.reference_command_actions(task, steps=20)
    np.testing.assert_array_equal(a3, actions[:20])
    with pytest.raises(ValueError):
        R.reference_schedule(task, steps=701)


def test_torch_action_path_reproduces_the_reconstruction(cut):
    torch = pytest.importorskip("torch")
    from robotic_cutting_sim.rl.actions import EdgeDeltaActionV1
    task, actions, reconstructed, schedule, _ = cut
    T_world = np.asarray(task.T_world_workpiece, float)[None]
    path = EdgeDeltaActionV1(task.action_limits(), T_world, task.max_cuts, device="cpu")
    pose = torch.as_tensor(schedule["initial_pose"][None], dtype=torch.float64)
    digits = []
    for t in range(len(actions)):
        result = path.apply(torch.as_tensor(actions[t:t + 1]), pose)
        pose = result.commanded_pose
        np.testing.assert_allclose(pose.numpy()[0, :3], reconstructed[t, :3], atol=1e-12, rtol=0)
        digits.append(int(result.cut_digit[0]) if bool(result.armed[0]) else -1)
    np.testing.assert_array_equal(digits, schedule["resolved_cut"])


def test_device_policy_gathers_by_step_count(cut):
    torch = pytest.importorskip("torch")
    task, actions, _, _, _ = cut
    policy = R.ReferenceCommandPolicyV1.from_tasks([task, task, task], device="cpu")
    assert policy.version == R.POLICY_REFERENCE and policy.tables.shape == (1, 700, 8)
    info = {"step_count": torch.tensor([0, 206, 700])}
    out = policy(None, info).numpy()
    np.testing.assert_array_equal(out[0], actions[0])
    np.testing.assert_array_equal(out[1], actions[206])
    np.testing.assert_array_equal(out[2], [0, 0, 0, 0, 0, 0, -1, 0])       # past the table: HOLD
