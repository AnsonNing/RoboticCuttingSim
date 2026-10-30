"""Static checks of the Phase R negative-control runner (no material simulation)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("phase_r_runner_test", ROOT/"scripts"/"run_phase_r_negative_controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rows_equal_declared_protocol(runner):
    protocol = json.loads((ROOT/"configs"/"wp11"/"phaseR_negative_controls_v1.json").read_text())
    declared = set(protocol["protocols"]["a"]["rows"]) | set(protocol["protocols"]["b"]["rows"])
    assert set(runner.ROWS) == declared
    assert protocol["protocols"]["b"]["max_cuts"] == 5 and protocol["protocols"]["b"]["armed_cut_index"] == 4
    assert all(row[1] == "TRIM-02-K2-R10" for name, row in runner.ROWS.items() if name.startswith("Ra_"))


def test_flat_press_normal_moves_along_the_blade_normal(runner):
    from robotic_cutting_sim.tasks import make_task
    task = make_task("TRIM-02-K2-R10", seed=42)
    plan = runner.flat_press_normal_plan(task)
    assert [w.label for w in plan] == ["approach", "start", "press", "press_hold", "retract"]
    assert [w.cut for w in plan] == [False, False, True, True, False]
    start, end = np.asarray(plan[1].pose[:3]), np.asarray(plan[2].pose[:3])
    normal = Rotation.from_quat(plan[2].pose[3:]).apply([0., 0., 1.])
    motion = (end-start)/np.linalg.norm(end-start)
    assert np.allclose(np.abs(motion@normal), 1., atol=1e-9)        # no edge-forward or along-edge component
    low = runner.C4.lowest_blade_offset_m(task)
    top = task.geometry.base_height_m
    assert start[2]+low-top == pytest.approx(.003, abs=1e-9) and end[2]+low-top == pytest.approx(-.008, abs=1e-9)


def test_half_depth_cut_retract_is_disarmed_and_digit_wrapper(runner):
    from robotic_cutting_sim.tasks import make_task
    task = make_task("CUT-03-K2", seed=42)
    plan = runner.half_depth_cut_plan(task)
    assert [(w.label, w.cut) for w in plan] == [("approach", False), ("start", False), ("descend", True), ("retract", False)]
    wrapped = runner.armed_digit(runner.half_depth_cut_plan, 4)(task)
    assert [w.cut_index for w in wrapped if w.cut] == [4] and all(w.cut_index == 0 for w in wrapped if not w.cut)
    assert runner.armed_digit(runner.half_depth_cut_plan, 0) is runner.half_depth_cut_plan


def test_single_digit_view_is_exact_reencoding(runner):
    digit = runner.DIGIT_B
    codes = np.array([[0, 0, 0], [1 << 2*digit, 2 << 2*digit, 0]], np.int32)
    steps = [{"stroke": digit+1, "cut_index": digit, "cut_enabled": True},
             {"stroke": 0, "cut_index": -1, "cut_enabled": False}]
    records = {"label_record_old": np.zeros(2, np.int32), "label_record_new": codes[1, :2].copy()}
    steps_v, data_v, records_v, context_v = runner.single_digit_view(steps, {"codes": codes}, records,
                                                                      {"packed": True, "max_cuts": 5})
    assert data_v["codes"].tolist() == [[0, 0, 0], [1, 2, 0]] and data_v["codes"].dtype == codes.dtype
    assert records_v["label_record_new"].tolist() == [1, 2]
    assert [(s["stroke"], s["cut_index"]) for s in steps_v] == [(1, 0), (0, -1)]
    assert steps[0]["stroke"] == digit+1                                 # the recorded rows are not modified
    assert context_v == {"packed": False, "max_cuts": 1}
