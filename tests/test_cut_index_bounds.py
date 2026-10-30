"""Out-of-range cut digits are rejected (WP-11 review finding 7); valid schedules are unchanged (CPU)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, drive_blade
from robotic_cutting_sim.coupled_robot import CouplingConfig
from robotic_cutting_sim.live_material import LiveCuttingMaterial
from robotic_cutting_sim.scene_config import compile_task, load_json
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.tool_profile import PROFILES

ROOT = Path(__file__).resolve().parents[1]
TWO_PLANES = {"schema": "rcs_goal_v1", "goal_id": "sever_two_bounds", "type": "sever",
              "episode": {"duration_s": 14.0, "control_dt_s": 0.02},
              "planes": [{"origin_local_m": [-.02, 0., 0.], "normal_local": [1., 0., 0.]},
                         {"origin_local_m": [.02, 0., 0.], "normal_local": [1., 0., 0.]}]}


@pytest.mark.parametrize("solver_version,law", [(5, 8), (4, 8), (None, 2)])
@pytest.mark.parametrize("cut", [1, 2, -2, True, 0.])
def test_blade_probe_rejects_out_of_range_phase_digits(solver_version, law, cut):
    profile = PROFILES["ruler_knife_v1"]
    with pytest.raises(ValueError, match="Phase cut index"):
        drive_blade(profile, VERTICAL_CUT, [.001, 0., .02], [Phase((0., 0., -.05), .01, cut)], cuts=1,
                    cutting_law=law, solver_version=solver_version)


def test_blade_probe_valid_digits_still_run():
    profile = PROFILES["ruler_knife_v1"]
    result = drive_blade(profile, VERTICAL_CUT, [.001, 0., .02], [Phase((0., 0., -.05), .002, 0),
                                                                    Phase((0., 0., .05), .002, -1)],
                         cuts=1, cutting_law=8, solver_version=5)
    assert result["code_encoding"] == "packed2" and len(result["labelled_after_phase"]) == 2


def _config(task):
    return CouplingConfig(control_dt=task.control_dt_s, material_dt=1e-4, dx=task.grid_dx_m, device="cpu")


def test_v5_material_rejects_a_task_with_more_digits_than_it_keeps():
    goal = load_json("configs/wp11/goals/remove_all_protrusions.json")
    task = compile_task(load_json("configs/wp11/scenes/bar_three_bumps.json"), goal, particle_spacing_m=.004)
    assert task.max_cuts == 3
    with pytest.raises(ValueError, match="declares max_cuts 3"):
        LiveCuttingMaterial(_config(task), task=task, max_cuts=1, cutting_law=8, solver="tlmpm_v5")
    assert LiveCuttingMaterial(_config(task), task=task, max_cuts=3, cutting_law=8, solver="tlmpm_v5").max_cuts == 3


def test_scheduled_digit_beyond_the_history_raises_before_any_substep():
    # Legacy (tlmpm_v4) construction is unchanged, so a two-digit schedule on a one-digit material constructs;
    # the scheduled second digit is then rejected instead of releasing contact without a label.
    task = compile_task(load_json("configs/wp11/scenes/bar_160x30x20.json"), TWO_PLANES, particle_spacing_m=.004)
    assert task.solver_module == "tlmpm_v4" and task.max_cuts == 2 and 1 in task.reference_cut_indices
    material = LiveCuttingMaterial(_config(task), task=task, max_cuts=1, cutting_law=8)
    edge = np.asarray(task.reference_edge_pose_world(0.), float)
    material.reset(edge, seed=42)
    dt = task.control_dt_s
    times = np.round(np.arange(0., task.duration_s, dt), 9)
    second = next(float(t) for t in times if task.active_cut_index(float(t)) == 1)
    first = next(float(t) for t in times if task.active_cut_index(float(t)) == 0)
    material.time_s, material.last_pose = first, edge
    assert material.advance(edge, round(first+dt, 9))["cut_index"] == 0      # in range: runs as before
    material.time_s = second
    codes = material.bed.code.numpy().copy()
    with pytest.raises(ValueError, match="Scheduled cut index 1 exceeds"):
        material.advance(edge, round(second+dt, 9))
    assert material.time_s == second and np.array_equal(material.bed.code.numpy(), codes)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_v4_demo_schedules_stay_within_their_declared_digits():
    """The pinned final_v4 demos (tests/test_blademaster_demo_runner.py) never schedule a digit >= --max-cuts."""
    runner = _load("demo_runner_bounds_test", ROOT/"scripts"/"run_blademaster_demos.py")
    demo_set = runner.DEMO_SETS["final_v4"]
    for key in demo_set.demos:
        spec = runner.DEMOS[key]
        argv = runner.run_argv(spec, Path("/nonexistent"), python="PY", gpu=0, cutting_law=8, seed=42)
        task_id = argv[argv.index("--task")+1]
        cuts = int(argv[argv.index("--max-cuts")+1]) if "--max-cuts" in argv else 1
        task = make_task(task_id, seed=42)
        times = np.round(np.arange(0., task.duration_s+1e-9, task.control_dt_s), 9)
        assert max(task.active_cut_index(float(t)) for t in times) < cuts, key
