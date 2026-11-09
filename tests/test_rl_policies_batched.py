"""Workload-A policies (``rl.policies_batched``): recorded_command_policy_v1 reconstruction and the replay driver.

The reconstruction tests read the frozen WP-17 Native recordings (``artifacts/wp17/native_optimized_001``);
where they are absent the tests are skipped (NOT_RUN). Torch parts run in ``.deps/isaacsim-venv`` (plan D-09)."""
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim.rl import policies_batched as P

ROOT = Path(__file__).resolve().parents[1]
WP17 = ROOT / "artifacts" / "wp17" / "native_optimized_001"
F32_TOL = 2.**-23


def _attempt(seed):
    path = WP17 / f"seed_{seed}" / "box" / "env" / "recording" / "attempt_000"
    if not (path / "actions.jsonl").exists():
        pytest.skip(f"WP-17 recording for seed {seed} not present (NOT_RUN)")
    return path


def _task(seed):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(seed), goal_config(), seed=42)


def test_selector_maps_back_to_the_digit():
    torch = pytest.importorskip("torch")
    from robotic_cutting_sim.rl.actions import EdgeDeltaActionV1
    from robotic_cutting_sim.control import ActionLimits
    for K in (1, 2, 3, 15):
        spec = EdgeDeltaActionV1(ActionLimits(), np.eye(4)[None], K)
        sel = torch.tensor([P.selector_for_digit(d, K) for d in range(K)], dtype=torch.float32)
        assert spec.cut_digit_from_selector(sel).tolist() == list(range(K))


@pytest.mark.parametrize("seed", [512, 15])
def test_recorded_commands_are_reconstructed_by_the_frozen_and_the_torch_path(seed):
    rec = P.load_recorded_commands(_attempt(seed))
    task = _task(seed)
    assert rec["config_sha256"] == task.config_sha256 and set(rec["statuses"]) == {"ACCEPTED"}
    limits = task.action_limits()
    actions, reconstructed, report = P.recorded_command_actions(
        rec["targets"], rec["cut_enabled"], rec["cut_index"], rec["reset_edge"], limits, task.T_world_workpiece,
        task.max_cuts)
    assert actions.shape == (1200, 8) and actions.dtype == np.float32
    assert report["clamped_steps"] == 0 and report["max_abs_normalized_component"] <= 1.
    # frozen NumPy path: within float32 tolerance of the recorded targets (no accumulation)
    assert report["max_position_diff_m"] <= F32_TOL * 1. and report["max_rotation_diff_rad"] <= 1e-6
    torch = pytest.importorskip("torch")
    from robotic_cutting_sim.rl.actions import ACTION_STATUSES, EdgeDeltaActionV1
    spec = EdgeDeltaActionV1(limits, np.asarray(task.T_world_workpiece)[None], task.max_cuts)
    commanded = torch.as_tensor(rec["reset_edge"][None], dtype=torch.float64)
    poses, digits = [], []
    for t in range(len(actions)):
        result = spec.apply(torch.as_tensor(actions[t:t + 1]), commanded)
        assert ACTION_STATUSES[int(result.status[0])] == "ACCEPTED"
        commanded = result.commanded_pose
        poses.append(commanded[0].numpy().copy())
        digits.append(int(result.cut_digit[0]))
    torch_report = P.residual_report(np.stack(poses), rec["targets"])
    assert torch_report["max_position_diff_m"] <= F32_TOL and torch_report["max_rotation_diff_rad"] <= 1e-6
    assert np.array_equal(np.asarray(digits), np.where(rec["cut_enabled"], rec["cut_index"], -1))


def test_policy_and_driver_gather_by_step_count():
    torch = pytest.importorskip("torch")
    tables = [np.full((4, 8), 0.1 * (k + 1), np.float32) for k in range(2)]
    policy = P.RecordedCommandPolicyV1(tables, [1, -1, 0], device="cpu")
    out = policy(None, dict(step_count=torch.tensor([0, 2, 9])))
    assert torch.allclose(out[0], torch.full((8,), .2))
    assert out[1].tolist() == [0., 0., 0., 0., 0., 0., -1., 0.]       # no table -> HOLD
    assert out[2].tolist() == [0., 0., 0., 0., 0., 0., -1., 0.]       # past the end -> HOLD

    class Traj:
        def __init__(self, k):
            self.steps = 3
            self.edge_pose_world_xyzw = np.full((3, 7), k, float) + np.arange(3)[:, None]
            self.resolved_cut = np.array([-1, k, k], np.int32)
            self.pad_boxes_edge = np.zeros((3, 3, 2, 3), np.float32) + k
    driver = P.MeasuredTrajectoryDriver([Traj(0), Traj(1)], [1, 0], device="cpu")
    pose, cut, pads = driver.inputs(torch.tensor([2, 0]))
    assert pose[:, 0].tolist() == [3., 0.] and cut.tolist() == [1, -1] and pads[0].max().item() == 1.
