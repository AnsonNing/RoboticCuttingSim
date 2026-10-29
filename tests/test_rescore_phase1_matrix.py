"""Versioned Phase 1 rescore (scripts/rescore_phase1_matrix.py, adr0006a1_adr0008) on synthetic rows (CPU, read-only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACING = .001


def _module():
    spec = importlib.util.spec_from_file_location("rescore_phase1_matrix", ROOT/"scripts/rescore_phase1_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain_stencil(x):
    """Chain stencils along x (4 mm nodes): neighbouring cells share a node."""
    cell = np.floor((x+.1)/.004).astype(int)
    nodes = np.zeros((len(x), 8), np.int32)
    nodes[:, 0], nodes[:, 1] = cell, cell+1
    weights = np.zeros((len(x), 8), np.float32)
    weights[:, :2] = .5
    return nodes, weights, np.zeros((*nodes.shape, 3), np.float32)


def _write_row(directory, result, steps, data):
    directory.mkdir()
    (directory/"result.json").write_text(json.dumps(result))
    with (directory/"actions.jsonl").open("w") as stream:
        for step in steps:
            stream.write(json.dumps(step)+"\n")
    np.savez_compressed(directory/"physics_state.npz", **data)
    return directory


def _steps(phases, counts, pose=None):
    return [{"step": i+1, "time_s": .02*(i+1), "phase": p, "labelled_count": int(c), "stroke": 0,
             "edge_pose_world": pose or [0., 0., 0., 0., 0., 0., 1.]} for i, (p, c) in enumerate(zip(phases, counts))]


def _ligament_row(tmp_path, *, core_label=False, withdraw_change=False, row="M4"):
    """CUT-01-R-like 1 mm x-z slab in the workpiece frame, cut to local z = 10 mm.

    Above the stop both sides are labelled; below it only off-core particles
    (|x| > dx, lead pre-assignment) are labelled, so the core band is code 0.
    """
    from robotic_cutting_sim.tasks import make_task
    task = make_task("CUT-01-R", seed=42)
    x, z = np.meshgrid(np.arange(-.0195, .02, SPACING), np.arange(.0005, .02, SPACING), indexing="ij")
    local = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
    world = task.points_to_world(local)
    nodes, weights, gradients = _chain_stencil(local[:, 0])
    side = np.where(local[:, 0] < 0, 1, 2).astype(np.int32)
    above = local[:, 2] > .010
    off_core = ~above & (np.abs(local[:, 0]) > .0045)
    codes = np.where(above | off_core, side, 0).astype(np.int32)
    if core_label:
        codes[np.flatnonzero(~above & (np.abs(local[:, 0]) < .002))[0]] = 1
    frames = 6
    history = np.stack([codes]*frames)
    if withdraw_change:
        history[-1, np.flatnonzero(codes == 0)[0]] = 2
    labelled = int(np.count_nonzero(codes))
    phases = ["descend"]*5+(["hold"]*5 if row == "M6" else [])+["withdraw"]*5
    counts = [labelled]*len(phases)
    if withdraw_change:
        counts[-1] += 1
    stop = task.points_to_world(np.array([[0., 0., .010]]))[0]
    steps = _steps(phases, counts, pose=[*stop, 0., 0., 0., 1.])
    normal = task.T_world_workpiece[:3, 0]
    result = {"row": row, "verdict": "FAIL", "cutting_law": 5, "task_id": "CUT-01-R", "seed": 42,
              "formal_resolution": True, "checks": {"code0_ligament_below_stop": False},
              "metrics": {"swept_planes": {"0": {"status": "FIT", "fit": {
                  "origin": task.points_to_world(np.array([[0., 0., .014]]))[0].tolist(), "normal": normal.tolist()}}}}}
    data = dict(time_s=np.linspace(.02*len(phases)/frames, .02*len(phases), frames), codes=history,
                mass_kg=np.full(len(world), 1e-6), reference_positions=world, reference_stencil_nodes=nodes,
                reference_stencil_weights=weights, reference_stencil_gradients=gradients)
    return _write_row(tmp_path/f"{row.lower()}_law5_row", result, steps, data)


@pytest.mark.parametrize("row", ["M4", "M6"])
def test_ligament_core_band_passes_while_law4_definition_fails(tmp_path, row):
    m = _module()
    entry = m.rescore_row(_ligament_row(tmp_path, row=row), "adr0006a1_adr0008")
    assert entry["runner_verdict"] == "FAIL" and entry["computed_verdict"] == "PASS", entry["checks"]
    metrics = entry["metrics"]
    assert metrics["core_band_below_stop_particles"] == metrics["core_band_below_stop_code0"] > 0
    lead = metrics["off_core_lead_labels_below_stop"]
    assert lead["count"] > 0 and lead["min_abs_s_m"] > m.GRID_DX_M
    assert metrics["law4_definition"]["verdict"] == "FAIL"
    assert metrics["law4_definition"]["labelled_below_stop"] == lead["count"]
    assert metrics["final_mechanical_components_1digit"] == 1


def test_ligament_fails_on_a_core_label_or_a_withdrawal_label(tmp_path):
    m = _module()
    core = m.rescore_row(_ligament_row(tmp_path, core_label=True), "adr0006a1_adr0008")
    assert core["computed_verdict"] == "FAIL" and not core["checks"]["core_band_below_stop_all_code0"]
    (tmp_path/"b").mkdir()
    moved = m.rescore_row(_ligament_row(tmp_path/"b", withdraw_change=True), "adr0006a1_adr0008")
    assert moved["computed_verdict"] == "FAIL" and not moved["checks"]["labels_unchanged_during_withdraw"]


def _three_part_m8b(tmp_path, slice_push_status):
    x = np.arange(-.0295, .03, SPACING)
    reference = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
    nodes, weights, gradients = _chain_stencil(x)
    digit0 = np.where(x < -.010, 1, 2)
    digit1 = np.where(x < .010, 1, 2)
    codes = (digit0+3*digit1).astype(np.int32)
    phases = ["cut_0"]*3+["hold_0"]*2+["lift_0"]*2+["move_1"]*2+["cut_1"]*3
    steps = _steps(phases, [len(x)]*len(phases))
    result = {"row": "M8b", "verdict": "PASS" if slice_push_status == "PASS_SLICE_PUSH" else "FAIL", "cutting_law": 5,
              "formal_resolution": True, "checks": {"slice_push_report_pass": slice_push_status == "PASS_SLICE_PUSH"},
              "metrics": {"slice_push_report": {"status": slice_push_status,
                                                "multi_cut": {"checks": {"declared_cut_planes":
                                                                         slice_push_status == "PASS_SLICE_PUSH"}}}}}
    data = dict(time_s=np.array([.06, .12, .24]), codes=np.stack([codes]*3), mass_kg=np.full(len(x), 1e-6),
                reference_positions=reference, reference_stencil_nodes=nodes, reference_stencil_weights=weights,
                reference_stencil_gradients=gradients)
    return _write_row(tmp_path/"m8b_law5_row", result, steps, data)


@pytest.mark.parametrize("status", ["PASS_SLICE_PUSH", "FAIL_SLICE_PUSH"])
def test_m8b_verdict_is_topology_and_slice_push(tmp_path, status):
    m = _module()
    entry = m.rescore_row(_three_part_m8b(tmp_path, status), "adr0006a1_adr0008")
    topology = entry["metrics"]["adr0006_topology"]
    assert topology["verdict"] == "PASS" and topology["final_mechanical_components_2digit"] == 3
    assert topology["lift_move_window"]["steps"] == 6 and topology["lift_move_window"]["change"] == 0
    slice_push = entry["metrics"]["runner_m8b_slice_push"]
    assert slice_push["verdict"] == ("PASS" if status == "PASS_SLICE_PUSH" else "FAIL")
    # A topology PASS never hides a slice-push FAIL.
    assert entry["computed_verdict"] == slice_push["verdict"]
    assert entry["checks"] == {"adr0006_topology": True, "runner_m8b_slice_push": status == "PASS_SLICE_PUSH"}


def test_other_rows_carry_the_runner_verdict_and_aggregation(tmp_path):
    m = _module()
    entries = []
    for row, verdict in [("M1a", "PASS"), ("M1b", "FAIL"), ("M9a", "PASS"), ("M9b", "PASS"), ("M9c", "REPORT_ONLY")]:
        directory = tmp_path/row
        directory.mkdir()
        (directory/"result.json").write_text(json.dumps({"row": row, "verdict": verdict, "cutting_law": 5}))
        entry = m.rescore_row(directory, "adr0006a1_adr0008")
        assert entry["computed_verdict"] == verdict and entry["basis"] == "carried_runner_verdict"
        entries.append(entry)
    aggregated = m.aggregate(entries)
    assert aggregated["M1"]["verdict"] == "FAIL" and aggregated["M9"]["verdict"] == "PASS"
    assert aggregated["M9"]["report_only"] == {"M9c": "REPORT_ONLY"}
    assert aggregated["M2"]["verdict"] == "NOT_RUN"


def test_frames_identical_anchors_inside_the_window():
    m = _module()
    codes = np.zeros((4, 5), np.int32)
    codes[1:, :2] = 1  # labels added before the window's first frame inside, during the previous phase
    data = {"time_s": np.array([.1, .2, .3, .4]), "codes": codes}
    window = {"steps": 10, "time_s": [.18, .4], "labelled_before": 2}
    assert m.frames_identical(data, window)["identical"] is True
    codes[3, 4] = 2
    assert m.frames_identical(data, window)["identical"] is False


def test_main_refuses_to_overwrite(tmp_path):
    m = _module()
    directory = tmp_path/"m1a_law5"
    directory.mkdir()
    (directory/"result.json").write_text(json.dumps({"row": "M1a", "verdict": "PASS", "cutting_law": 5}))
    out = tmp_path/"evidence.json"
    assert m.main(["--criteria-version", "adr0006a1_adr0008", "--law", "5", "--rows", str(directory),
                   "--out", str(out)]) == 0
    before = out.read_text()
    with pytest.raises(SystemExit):
        m.main(["--criteria-version", "adr0006a1_adr0008", "--law", "5", "--rows", str(directory), "--out", str(out)])
    assert out.read_text() == before
    assert json.loads(before)["rows"][0]["computed_verdict"] == "PASS"
