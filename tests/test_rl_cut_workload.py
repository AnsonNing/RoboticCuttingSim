"""P1-5 T5.1: the pre-declared CUT workload cut_v7_2_v1 (scene, goal and workload pins; equality with the frozen
tlmpm_v5 compile). CPU only (.venv); the stencil check builds both frozen single solvers on the Warp CPU device."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from robotic_cutting_sim.rl import cut_workload as CW

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {"configs/wp12/scenes/bar_flat_board.json": "462b4e155e6359c5afe47372155b10a5c68952fd5541ab4dbbbb669520a83747",
          "configs/wp14/goals/standard_cut.json": "e3c39a595572e2a7551366c074097f963b6218db4bc91324cac91cd27542b04a",
          "configs/wp11/goals/sever_mid.json": "32479130597e2d4c20cb89d9fce67d82536c8602d235dd7dfec9a5e3b1a86918",
          "configs/rl_benchmark/workloads_v1.json": "c90fc82419c5676219bca6f1c3fc3c80ea7c7ab91a9e06be2a2ee0cf88fa01e6"}


def _sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def workload():
    return CW.load_workload()


@pytest.fixture(scope="module")
def tasks(workload):
    return CW.compile_workload_task(workload), CW.compile_source_task(workload)


def test_frozen_inputs_are_unchanged():
    for rel, expected in FROZEN.items():
        assert _sha(rel) == expected, rel


def test_declaration_pins_and_status(workload):
    assert workload["status"] == "PREDECLARED" and workload["id"] == "cut_v7_2_v1"
    assert workload["scene"]["sha256"] == _sha(workload["scene"]["path"])
    assert workload["success_rule"].startswith("goals.evaluate_goal verdict == 'PASS' and nothing else")
    assert workload["driver"]["id"] == "reference_command_policy_v1"
    assert workload["reward"]["status"] == "PROVISIONAL_SHAPING_NOT_SUCCESS"
    assert workload["native_reference"]["status"].startswith("NOT_RUN")
    v2 = json.loads((ROOT / "configs/rl_benchmark/workloads_v2.json").read_text())
    assert v2["extends"]["sha256"] == FROZEN["configs/rl_benchmark/workloads_v1.json"]
    assert [w["id"] for w in v2["workloads"]] == ["C1", "C2", "C3"]
    assert v2["workloads"][1]["workload"] == "configs/rl_benchmark/cut_v7_2_v1.json"


def test_scene_differs_from_the_frozen_scene_only_in_solver_and_description(workload):
    assert CW.scene_differences(workload) == ["description", "solver.module"]
    new = json.loads((ROOT / workload["scene"]["path"]).read_text())
    old = json.loads((ROOT / workload["scene"]["source_scene"]["path"]).read_text())
    assert (old["solver"]["module"], new["solver"]["module"]) == ("tlmpm_v5", "tlmpm_v7_2")
    assert new["scene_id"] == old["scene_id"] == "bar_flat_board"
    # line-for-line copy: only the description and the solver line differ
    a = (ROOT / workload["scene"]["path"]).read_text().splitlines()
    b = (ROOT / workload["scene"]["source_scene"]["path"]).read_text().splitlines()
    assert len(a) == len(b) and [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [3, 12]


def test_tampered_pin_is_refused(tmp_path, workload):
    root = tmp_path / "repo"
    for rel in (workload["scene"]["path"], workload["scene"]["source_scene"]["path"], workload["goal"]["path"]):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    CW.load_workload(CW.WORKLOAD_PATH, root=root)
    text = (root / workload["scene"]["path"]).read_text().replace('"max_cuts": 4', '"max_cuts": 3')
    (root / workload["scene"]["path"]).write_text(text)
    with pytest.raises(CW.WorkloadPinError):
        CW.load_workload(CW.WORKLOAD_PATH, root=root)


def test_compiled_ids_and_counts_match_the_declaration(workload, tasks):
    new, old = tasks
    compiled = workload["compiled"]
    assert (new.task_id, new.config_sha256) == (compiled["task_id"], compiled["config_sha256"])
    assert old.task_id == compiled["source_task_id_tlmpm_v5"]           # = WP-12 Demo 1 task id
    assert (new.solver_module, old.solver_module) == ("tlmpm_v7_2", "tlmpm_v5")
    assert (new.max_cuts, new.cutting_law, new.particle_spacing_m, new.grid_dx_m) == (4, 8, .001, .004)
    points, masses, lattice = new.sample_particles()
    assert len(points) == compiled["particles"] and np.isclose(masses.sum(), compiled["mass_kg"], rtol=1e-12)
    assert int(round(new.duration_s / new.control_dt_s)) == workload["episode"]["control_steps"] == 700


def test_particles_and_reference_schedule_equal_the_frozen_v5_compile(tasks):
    new, old = tasks
    for a, b in zip(new.sample_particles(), old.sample_particles()):
        np.testing.assert_array_equal(a, b)
    assert new.edge_position_knots_local_txyz == old.edge_position_knots_local_txyz
    assert new.reference_orientations_local_xyzw == old.reference_orientations_local_xyzw
    assert new.reference_cut_indices == old.reference_cut_indices
    assert new.reference_phase_labels == old.reference_phase_labels
    assert new.goal == old.goal and new.controller == old.controller
    assert new.action_limits() == old.action_limits()
    for t in (0., 4.14, 7.0, 14.0):
        np.testing.assert_array_equal(new.reference_edge_pose_world(t), old.reference_edge_pose_world(t))


def test_reference_stencils_and_masses_equal_between_the_frozen_solvers(tasks):
    pytest.importorskip("warp")
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    arrays = []
    for task in tasks:
        material = LiveCuttingMaterial(CouplingConfig(.02, 1e-4, .004, device="cpu", feedback=True), task=task,
                                       max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                       solver=task.solver_module, v7_2_optimizations=True)
        material.reset(task.reference_edge_pose_world(0.), 42)
        assert material.slots_per_node == 8 and material.substeps == 200
        assert np.shape(material.obstacle_boxes_edge) == (1, 2, 3)          # handle_only_v1
        arrays.append(dict(nodes=material.bed.nodes.numpy(), weights=material.bed.weights.numpy(),
                           gradients=material.bed.gradients.numpy(), masses=material.masses.copy(),
                           points=np.asarray(material.points).copy(), world=np.asarray(material.world_points).copy()))
    for key in arrays[0]:
        np.testing.assert_array_equal(arrays[0][key], arrays[1][key], err_msg=key)


def test_receipt_and_task_args(workload):
    args = CW.workload_task_args(workload)
    assert args["planner_options"] == {"max_rotation_deg": 1.0, "max_translation_m": .001, "controller_profile": "path_v2"}
    assert args["seed"] == 42 and args["scene"].endswith("bar_flat_board_tlmpm_v7_2_v1.json")
    receipt = CW.receipt(workload)
    assert receipt["task_id"] == workload["compiled"]["task_id"] and len(receipt["declaration_sha256"]) == 64
