"""WP-11 Phase S (ADR 0015): scene/goal configs, compiled tasks, goal evaluators and the goal planner.

Fast CPU tests use synthetic recordings with trilinear reference stencils; the
kinematic smokes (``cpu_sim``) run the planner through ``make_kinematic_env(task=...)``.
"""
from copy import deepcopy
import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import goal_planner
from robotic_cutting_sim.goal_planner import orientation_metrics, plan, plan_with_report, skim_rotation, through_cut_rotation
from robotic_cutting_sim.goals import evaluate_goal, interface_by_plane, validate_goal
from robotic_cutting_sim.scene_config import (ConfiguredTaskSpec, SceneConfigError, SceneInfeasible, compile_task,
                                              load_json, raw_support_boxes, resolve_scene, validate_scene)
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.task_scenarios import apply_scenario, configure_task, fixture_materials
from robotic_cutting_sim.tool_profile import PROFILES

SCENES = "configs/wp11/scenes"
GOALS = "configs/wp11/goals"


def scene(name):
    return load_json(f"{SCENES}/{name}.json")


def goal(name):
    return load_json(f"{GOALS}/{name}.json")


def with_solver(doc, module="tlmpm_v4", max_cuts=2):
    doc = deepcopy(doc)
    doc["solver"].update(module=module, max_cuts=max_cuts)
    return doc


def remove(ids, duration=14.):
    doc = goal("remove_all_protrusions")
    doc.update(goal_id="remove_" + "_".join(ids), select=list(ids))
    doc["episode"]["duration_s"] = duration
    return doc


# ---------------------------------------------------------------- synthetic recordings

def synthetic_dx(task):
    """Synthetic grid spacing: twice the particle spacing, so centres sit mid-cell and the
    declared planes at multiples of the spacing lie on node planes (as in the frozen setup)."""
    return 2*task.particle_spacing_m


def stencils(task, dx=None):
    """Trilinear 8-node reference stencils (all weights > 0) on a grid with nodes at multiples of dx."""
    dx = synthetic_dx(task) if dx is None else dx
    local, _, _ = task.sample_particles()
    # x nodes at multiples of dx (declared x planes on node planes); y/z nodes a quarter or half cell off every centre.
    s = task.particle_spacing_m
    origin = np.r_[-.2, local[:, 1:].min(0)-s/2-20*dx]
    cell = np.floor((local-origin)/dx).astype(np.int64)
    frac = (local-origin)/dx-cell
    dims = np.array([400, 400, 400])
    nodes, weights = [], []
    for corner in np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"), -1).reshape(-1, 3):
        index = cell+corner
        nodes.append((index[:, 0]*dims[1]+index[:, 1])*dims[2]+index[:, 2])
        weights.append(np.prod(np.where(corner, frac, 1-frac), axis=1))
    nodes, weights = np.stack(nodes, 1), np.stack(weights, 1)
    return nodes, weights, np.zeros((*nodes.shape, 3))


def recording(task, final_codes, *, frames=None, moves=None, dx=None):
    """WORLD recording: frame 0 uncut, later frames with ``final_codes``."""
    local, masses, lattice = task.sample_particles()
    world = task.points_to_world(local)
    nodes, weights, gradients = stencils(task, dx)
    times = [0., task.duration_s] if frames is None else frames
    codes = [np.zeros(len(local), np.int64)]+[np.asarray(final_codes, np.int64)]*(len(times)-1)
    positions = [world.copy() for _ in times]
    for frame, mask, offset_local in moves or ():
        positions[frame][mask] += task.T_world_workpiece[:3, :3]@np.asarray(offset_local)
    return {"time_s": np.asarray(times), "positions": np.asarray(positions), "codes": np.asarray(codes),
            "mass_kg": masses, "reference_positions": world, "reference_indices": lattice,
            "reference_stencil_nodes": nodes, "reference_stencil_weights": weights,
            "reference_stencil_gradients": gradients, "particle_spacing_m": task.particle_spacing_m}


def plane_codes(local, planes, digits=None):
    """Ternary codes: digit k = 1 on the negative side of plane k, 2 on the positive side."""
    codes = np.zeros(len(local), np.int64)
    for k, (origin, normal) in enumerate(planes):
        side = np.where((local-np.asarray(origin))@np.asarray(normal) > 0, 2, 1)
        codes += side*3**(k if digits is None else digits[k])
    return codes


def protrusion_codes(task, removed, plane_z=.02):
    """Digit k removes protrusion removed[k]: its above-plane particles 2, a 2-cell band below the plane 1."""
    from robotic_cutting_sim.goals import protrusion_ownership
    local, _, _ = task.sample_particles()
    owner = protrusion_ownership(task, local, plane_z)
    band = (local[:, 2] <= plane_z) & (local[:, 2] >= plane_z-2*synthetic_dx(task))
    codes = np.zeros(len(local), np.int64)
    for digit, index in enumerate(removed):
        codes += np.where(owner == index, 2, np.where(band, 1, 0))*3**digit
    return codes


def to_packed(codes, cuts):
    digits = [(codes//3**k) % 3 for k in range(cuts)]
    return sum(d.astype(np.int64) << (2*k) for k, d in enumerate(digits))


@pytest.fixture(scope="module")
def bumps_task():
    return compile_task(scene("bar_three_bumps"), goal("remove_all_protrusions"), particle_spacing_m=.004)


# ---------------------------------------------------------------- schema validation

@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(schema="rcs_scene_v0"), "schema"),
    (lambda d: d.update(colour="red"), "unknown keys"),
    (lambda d: d["workpiece"]["protrusions"].append(dict(d["workpiece"]["protrusions"][0])), "duplicated"),
    (lambda d: d["workpiece"]["protrusions"][0].update(family="cube"), "family"),
    (lambda d: d["solver"].update(cutting_law=3), "not available"),
    (lambda d: d["solver"].update(module="tlmpm_v4", max_cuts=3), "max_cuts"),
    (lambda d: d["tool"].update(profile_id="cleaver"), "profile_id"),
    (lambda d: d["supports"].update(kind="slotted_board"), "at least one slot"),
    (lambda d: d["workpiece"]["world_pose"].update(roll_deg=5.), "unknown keys"),
    (lambda d: d.update(randomization={"seed": 1, "ranges": [{"path": "solver.max_cuts", "low": 1, "high": 2}]}), "path"),
])
def test_scene_schema_rejects_invalid_configs(mutate, message):
    doc = scene("bar_three_bumps")
    mutate(doc)
    with pytest.raises(SceneConfigError, match=message):
        validate_scene(doc)


def test_infeasible_geometry_names_a_reason():
    doc = scene("bar_three_bumps")
    doc["workpiece"]["protrusions"][2]["x_m"] = .075
    with pytest.raises(SceneInfeasible) as error:
        validate_scene(doc)
    assert error.value.reason == "protrusion_outside_base"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(type="flatten_all"), "goal.type"),
    (lambda d: d.update(planes=[{"origin_local_m": [0, 0, 0], "normal_local": [2, 0, 0]}]), "unit vector"),
    (lambda d: d.update(planner={"rake": 10}), "Unknown planner options"),
    (lambda d: d.update(expected_pieces=1), r"\[2, None\]"),
    (lambda d: d["episode"].update(duration_s=8.005), "integral"),
    (lambda d: d.update(minimum_part_mass_fraction=1.5), r"within \[0, 1\]"),
])
def test_goal_schema_rejects_invalid_configs(mutate, message):
    doc = goal("sever_mid")
    mutate(doc)
    with pytest.raises(SceneConfigError, match=message):
        validate_goal(doc)


def test_sequences_do_not_nest_and_compile_checks_digit_capacity():
    doc = goal("slice_then_push")
    doc["steps"].append({"type": "sequence", "steps": []})
    with pytest.raises(SceneConfigError, match="do not nest"):
        validate_goal(doc)
    with pytest.raises(SceneConfigError, match="needs 3 cut digits"):
        compile_task(with_solver(scene("bar_three_bumps")), goal("remove_all_protrusions"), particle_spacing_m=.004)


def test_shipped_configs_validate_and_compile():
    for name in ("bar_160x30x20", "bar_three_bumps", "bar_slotted_board_4slots"):
        assert validate_scene(f"{SCENES}/{name}.json")["scene_id"] == name
    pairs = [("bar_160x30x20", "sever_mid"), ("bar_three_bumps", "remove_all_protrusions"),
             ("bar_three_bumps", "trim_plane_z20"), ("bar_slotted_board_4slots", "divide_4"),
             ("bar_slotted_board_4slots", "slice_then_push")]
    for scene_name, goal_name in pairs:
        task = compile_task(scene_name, goal_name, particle_spacing_m=.004)
        assert isinstance(task, ConfiguredTaskSpec)
        assert task.task_id.startswith(f"CFG-{scene_name}-{goal_name}-") and len(task.task_id.rsplit("-", 1)[1]) == 8
        receipt = task.to_dict()
        assert receipt["protocol_sha256"] == task.config_sha256 and receipt["configured_task"]["goal"]["goal_id"] == goal_name
        assert task.duration_s == goal(goal_name)["episode"]["duration_s"] and task.control_dt_s == .02
        # HOME is the start pose (reached at t = 0); the schedule ends with RETURN_HOME, then holds.
        home = goal_planner.plan(task.scene, task.goal, task.tool_profile)[0]
        assert home.label == "HOME" and np.allclose(task.reference_edge_pose_world(0.), home.pose, atol=1e-12)
        assert "RETURN_HOME" in task.reference_phase_labels and task.reference_phase_labels[-1] == "DONE"
        assert task.evaluate_oracle({})["status"] == "NOT_APPLICABLE"


# ---------------------------------------------------------------- equivalence with the frozen tasks

def _same_material(a, b):
    pa, ma, la = a.sample_particles()
    pb, mb, lb = b.sample_particles()
    return np.array_equal(pa, pb) and np.array_equal(ma, mb) and np.array_equal(la, lb)


def _same_task_body(frozen, task):
    assert _same_material(frozen, task)
    assert task.fixture_boxes_local == frozen.fixture_boxes_local
    assert fixture_materials(task) == fixture_materials(frozen)
    assert task.hand_from_edge_xyz_xyzw == tuple(frozen.hand_from_edge_xyz_xyzw)
    assert task.tool_profile == frozen.tool_profile
    assert (task.blade_length_m, task.blade_height_m, task.blade_half_width_m) == \
        (frozen.blade_length_m, frozen.blade_height_m, frozen.blade_half_width_m)
    assert task.target_origin_local_m == tuple(frozen.target_origin_local_m)
    assert task.target_normal_local == tuple(frozen.target_normal_local)
    assert task.edge_clearance_m == frozen.edge_clearance_m and task.goal_kind == frozen.goal_kind
    assert np.array_equal(task.world_from_workpiece_xyzw, frozen.world_from_workpiece_xyzw)
    assert task.table_top_world_z_m == frozen.table_top_world_z_m and task.material == frozen.material
    assert (task.particle_spacing_m, task.grid_dx_m) == (frozen.particle_spacing_m, frozen.grid_dx_m)


def test_config_reproduces_cut_01_r():
    task = compile_task("bar_160x30x20", "sever_mid")
    _same_task_body(make_task("CUT-01-R"), task)
    assert np.allclose(task.edge_orientation_local_xyzw, make_task("CUT-01-R").edge_orientation_local_xyzw, atol=1e-12)


def trim_scene(workcell):
    doc = with_solver(scene("bar_three_bumps"), max_cuts=1)
    doc["scene_id"] = "trim01"
    doc["workpiece"]["protrusions"] = [{"id": "bump", "family": "dome", "x_m": 0., "y_m": 0., "radius_x_m": .012,
                                        "radius_y_m": .012, "height_m": .008}]
    if workcell:
        doc["workpiece"]["world_pose"]["translation_m"] = [0., 0., .05]
        doc["supports"]["workcell"] = {"from_scenario": "blademaster_workcell_v1"}
    return doc


@pytest.mark.parametrize("workcell", [False, True])
def test_config_reproduces_trim_01_r(workcell):
    frozen = make_task("TRIM-01-R")
    rake = 25.
    if workcell:
        frozen, rake = configure_task(frozen, scenario="blademaster_workcell_v1"), 35.
    for goal_doc in (goal("trim_plane_z20"), goal("remove_all_protrusions")):
        goal_doc["planner"] = {"rake_deg": rake}
        task = compile_task(trim_scene(workcell), goal_doc)
        _same_task_body(frozen, task)
        # The planner's skim frame (tau = psi = 0, rho = rake) is the frozen edge orientation.
        assert np.allclose(task.edge_orientation_local_xyzw, frozen.edge_orientation_local_xyzw, atol=1e-12)


# ---------------------------------------------------------------- evaluators on synthetic recordings

def test_one_scene_three_goals_three_correct_verdicts(bumps_task):
    data = recording(bumps_task, protrusion_codes(bumps_task, [0, 1]))
    two = evaluate_goal(remove(["p0", "p1"]), data, task=bumps_task)
    every = evaluate_goal(goal("remove_all_protrusions"), data, task=bumps_task)
    sever = evaluate_goal(goal("sever_mid"), data, task=bumps_task)
    assert two["integrity"]["status"] == every["integrity"]["status"] == sever["integrity"]["status"] == "PASS"
    assert two["verdict"] == "PASS"
    assert every["verdict"] == "FAIL" and not every["outcome"]["checks"]["selected_protrusions_removed"]
    items = {row["id"]: row for row in every["outcome"]["metrics"]["items"]}
    assert items["p0"]["removed_mass_fraction"] == items["p1"]["removed_mass_fraction"] == 1.
    assert items["p2"]["removed_mass_fraction"] == 0. and not items["p2"]["pass"]
    assert every["outcome"]["checks"]["substrate_holds_below_plane_mass"]
    assert sever["verdict"] == "FAIL" and not sever["outcome"]["checks"]["expected_substantial_pieces"]
    # Selecting only p0 fails on the over-removed p1 (unselected retention), not on p0.
    only = evaluate_goal(remove(["p0"]), data, task=bumps_task)
    assert only["verdict"] == "FAIL" and only["outcome"]["checks"]["selected_protrusions_removed"]
    assert not only["outcome"]["checks"]["unselected_protrusions_retained"]
    assert evaluate_goal(goal("remove_all_protrusions"), recording(bumps_task, protrusion_codes(bumps_task, [0, 1, 2])),
                         task=bumps_task)["verdict"] == "PASS"


def test_remove_by_region_and_packed_codes(bumps_task):
    codes = protrusion_codes(bumps_task, [0, 1])
    region = remove(["p0"])
    region.update(goal_id="remove_left_half", select={"region_local_m": {"x": [-.08, .01]}})
    ternary = evaluate_goal(region, recording(bumps_task, codes), task=bumps_task)
    packed = evaluate_goal(region, {**recording(bumps_task, to_packed(codes, 2)), "code_encoding": "packed2"}, task=bumps_task)
    assert ternary["verdict"] == packed["verdict"] == "PASS" and packed["code_encoding"] == "packed2"
    assert [row["selected"] for row in ternary["outcome"]["metrics"]["items"]] == [True, True, False]
    assert ternary["outcome"]["metrics"]["items"] == packed["outcome"]["metrics"]["items"]


def test_integrity_layer_is_separate_from_the_outcome(bumps_task):
    data = recording(bumps_task, protrusion_codes(bumps_task, [0, 1]))
    data["codes"] = np.concatenate([data["codes"], data["codes"][-1:]-np.where(data["codes"][-1:] == 2, 1, 0)])
    data["positions"] = np.concatenate([data["positions"], data["positions"][-1:]])
    data["time_s"] = np.r_[data["time_s"], data["time_s"][-1]+.02]
    result = evaluate_goal(remove(["p0", "p1"]), data, task=bumps_task)
    assert result["integrity"]["status"] == "FAIL" and result["verdict"] == "FAIL_INTEGRITY"
    assert not result["integrity"]["checks"]["persistent_history_no_digit_rewrites"]
    missing = evaluate_goal(goal("sever_mid"), {"time_s": [0., 1.]}, task=bumps_task)
    assert missing["verdict"] == "NOT_RUN"


@pytest.fixture(scope="module")
def divide_task():
    return compile_task(with_solver(scene("bar_160x30x20"), "tlmpm_v5", 3), "divide_4", particle_spacing_m=.004)


def test_divide_into_n_pieces(divide_task):
    local, _, _ = divide_task.sample_particles()
    planes = [((x, 0., 0.), (1., 0., 0.)) for x in (-.04, 0., .04)]
    four = evaluate_goal(goal("divide_4"), recording(divide_task, plane_codes(local, planes)), task=divide_task)
    assert four["verdict"] == "PASS" and four["outcome"]["metrics"]["substantial_pieces"] == 4
    assert np.allclose(four["outcome"]["metrics"]["piece_mass_fractions"][:4], .25)
    three = evaluate_goal(goal("divide_4"), recording(divide_task, plane_codes(local, planes[:2])), task=divide_task)
    assert three["verdict"] == "FAIL" and three["outcome"]["metrics"]["substantial_pieces"] == 3


def test_sever_location_rules():
    task = compile_task("bar_160x30x20", "sever_mid", particle_spacing_m=.004)
    local, _, _ = task.sample_particles()
    data = recording(task, plane_codes(local, [((0., 0., 0.), (1., 0., 0.))]))
    result = evaluate_goal(goal("sever_mid"), data, task=task)
    assert result["verdict"] == "PASS"
    # For one plane the per-plane statistic is exactly the frozen/swept-plane statistic.
    from robotic_cutting_sim.cut_topology import mechanical_components
    from robotic_cutting_sim.demo_evaluation import swept_plane_interface
    _, labels = mechanical_components(data["reference_stencil_nodes"], data["reference_stencil_weights"], data["codes"][-1],
                                      1, gradients=data["reference_stencil_gradients"])
    frozen = swept_plane_interface(data["reference_stencil_nodes"], data["reference_stencil_weights"],
                                   data["reference_stencil_gradients"], labels, local, (0., 0., 0.), (1., 0., 0.))
    mine = interface_by_plane(data["reference_stencil_nodes"], data["reference_stencil_weights"],
                              data["reference_stencil_gradients"], labels, local, [(np.zeros(3), np.array([1., 0., 0.]))])[0]
    assert mine["maximum_reference_midpoint_plane_distance_m"] == frozen["maximum_reference_midpoint_plane_distance_m"]
    # A grid-aligned cut 12 mm off the declared plane fails the frozen 2 mm location bound.
    shifted = evaluate_goal(goal("sever_mid"), recording(task, plane_codes(local, [((.012, 0., 0.), (1., 0., 0.))])), task=task)
    assert shifted["verdict"] == "FAIL" and not shifted["outcome"]["checks"]["cut_at_declared_planes"]
    # Oblique plane: ADR 0006 Amendment 1 rule (ideal geometric labels pass, a shifted cut does not).
    normal = (np.cos(np.radians(20)), 0., np.sin(np.radians(20)))
    oblique = goal("sever_mid")
    oblique["planes"] = [{"origin_local_m": [0., 0., .01], "normal_local": list(normal)}]
    exact = evaluate_goal(oblique, recording(task, plane_codes(local, [((0., 0., .01), normal)])), task=task)
    plane_row = exact["quality"]["plane_location"][0]
    assert exact["verdict"] == "PASS" and not plane_row["grid_aligned"] and "Amendment 1" in plane_row["rule"]
    off = evaluate_goal(oblique, recording(task, plane_codes(local, [((.012, 0., .01), normal)])), task=task)
    assert off["verdict"] == "FAIL" and not off["outcome"]["checks"]["no_far_particle_in_another_region_piece"]


@pytest.fixture(scope="module")
def push_task():
    return compile_task("bar_slotted_board_4slots", "slice_then_push", particle_spacing_m=.002)


def _push_recording(task, right_shift, middle_shift=0., label_change=False):
    local, _, _ = task.sample_particles()
    codes = plane_codes(local, [((-.03, 0., 0.), (1., 0., 0.)), ((.03, 0., 0.), (1., 0., 0.))])
    windows = task.phase_intervals_s()
    start = min(v[0] for k, v in windows.items() if k.startswith("PUSH"))
    end = max(v[1] for k, v in windows.items() if k.startswith("PUSH"))
    times = [0., start-.1, end+.1, task.duration_s]
    data = recording(task, codes, frames=times, dx=task.particle_spacing_m,
                     moves=[(f, local[:, 0] > .03, (right_shift, 0., 0.)) for f in (2, 3)]
                     + [(f, (local[:, 0] > -.03) & (local[:, 0] < .03), (middle_shift, 0., 0.)) for f in (2, 3)])
    if label_change:
        data["codes"][2:, local[:, 2] > .018] += 3**2
    return data


def test_move_piece_and_sequence(push_task):
    sequence = goal("slice_then_push")
    good = evaluate_goal(sequence, _push_recording(push_task, .012), task=push_task)
    assert good["verdict"] == "PASS", good["outcome"]
    move = good["outcome"]["steps"][1]["outcome"]
    assert move["metrics"]["window_source"] == "task_reference_schedule"
    assert move["metrics"]["displacement_along_direction_m"] == pytest.approx(.012)
    dragged = evaluate_goal(sequence, _push_recording(push_task, .012, middle_shift=.004), task=push_task)
    step = dragged["outcome"]["steps"][1]["outcome"]
    assert dragged["verdict"] == "FAIL" and not step["checks"]["other_pieces_stationary"]
    assert step["checks"]["selected_piece_displaced"]
    partial = evaluate_goal(sequence, _push_recording(push_task, .005), task=push_task)
    assert partial["verdict"] == "FAIL"
    assert [row["outcome"]["status"] for row in partial["outcome"]["steps"]] == ["PASS", "FAIL"]
    relabel = evaluate_goal(sequence, _push_recording(push_task, .012, label_change=True), task=push_task)
    assert not relabel["outcome"]["steps"][1]["outcome"]["checks"]["no_label_change_while_disarmed"]


# ---------------------------------------------------------------- planner

def test_planner_takes_config_only_and_never_reads_material(monkeypatch):
    assert list(inspect.signature(plan).parameters) == ["scene", "goal", "tool_profile", "options"]
    import robotic_cutting_sim.benchmark as benchmark
    import robotic_cutting_sim.tasks as tasks

    def forbidden(*args, **kwargs):
        raise AssertionError("The planner must not sample or read material state")
    monkeypatch.setattr(benchmark, "sample_specimen", forbidden)
    monkeypatch.setattr(tasks, "sample_specimen", forbidden)
    monkeypatch.setattr(tasks.CuttingTaskSpec, "sample_particles", forbidden)
    waypoints, report = plan_with_report(scene("bar_three_bumps"), goal("remove_all_protrusions"), PROFILES["ruler_knife_v1"])
    assert waypoints and "no particle" in report["inputs"]


def test_through_cut_frames_follow_the_plane_normal():
    cut = make_task("CUT-01-R")
    assert np.allclose(through_cut_rotation([1., 0., 0.]).as_matrix(),
                       Rotation.from_quat(cut.edge_orientation_local_xyzw).as_matrix(), atol=1e-12)
    for normal in ([np.cos(.3), np.sin(.3), 0.], [np.cos(np.radians(20)), 0., np.sin(np.radians(20))]):
        R = through_cut_rotation(normal).as_matrix()
        assert np.allclose(R.T@R, np.eye(3)) and np.linalg.det(R) > 0
        assert np.allclose(R[:, 2], normal) and abs(R[:, 0]@normal) < 1e-12 and R[0:2, 0]@R[0:2, 0] < 1
        assert R[2, 0] < -.9 and np.allclose(R[:, 1], np.cross(R[:, 2], R[:, 0]))
    with pytest.raises(ValueError, match="approach"):
        through_cut_rotation([0., 0., 1.])
    R = through_cut_rotation([0., 0., 1.], approach=[1., 0., 0.]).as_matrix()
    assert np.allclose(R[:, 0], [1., 0., 0.]) and np.allclose(R[:, 2], [0., 0., 1.])


@pytest.mark.parametrize("direction, rake, skew", [((1., 0., 0.), 25., 0.), ((1., 0., 0.), 0., 0.),
                                                   ((np.cos(.5), np.sin(.5), 0.), 10., 0.), ((0., -1., 0.), 35., 15.)])
def test_skim_frames_have_declared_adr0013_metrics(direction, rake, skew):
    rotation = skim_rotation(direction, rake_deg=rake, skew_deg=skew)
    metrics = orientation_metrics(rotation.as_quat(), direction)
    assert metrics["tau_deg"] == pytest.approx(0., abs=1e-9) and metrics["mu_deg"] == pytest.approx(0., abs=1e-9)
    assert metrics["psi_deg"] == pytest.approx(skew, abs=1e-9) and metrics["rho_deg"] == pytest.approx(rake, abs=1e-9)
    if direction == (1., 0., 0.) and not skew:
        assert np.allclose(rotation.as_matrix(), Rotation.from_euler("y", rake, degrees=True).as_matrix(), atol=1e-12)
    # A descending motion has mu < 0; tilting the edge about the cut direction gives tau.
    edge = rotation.apply([0., 1., 0.])
    axis = np.cross(edge, [0., 0., 1.])
    tilted = Rotation.from_rotvec(np.radians(5.)*axis/np.linalg.norm(axis))*rotation
    assert orientation_metrics(tilted.as_quat(), direction)["tau_deg"] == pytest.approx(5., abs=1e-9)
    assert orientation_metrics(rotation.as_quat(), np.asarray(direction)+[0., 0., -.1])["mu_deg"] < 0


def test_surface_trim_phase_order_digits_and_rake():
    waypoints, report = plan_with_report(scene("bar_three_bumps"), goal("remove_all_protrusions"), PROFILES["ruler_knife_v1"],
                                         {"rake_deg": 10.})
    labels = [w.label for w in waypoints]
    expected = ["HOME"]+[f"{phase}:{pid}" for pid in ("p0", "p1", "p2")
                         for phase in ("REORIENT", "PRE_CUT", "SKIM", "WITHDRAW")]+["RETURN_HOME"]
    assert labels == expected and report["digits_used"] == 3
    skims = [w for w in waypoints if w.label.startswith("SKIM")]
    assert [w.cut_index for w in skims] == [0, 1, 2] and all(w.cut for w in skims)
    assert not any(w.cut for w in waypoints if not w.label.startswith("SKIM"))
    for row in report["operations"]:
        metrics = row["orientation_metrics"]
        assert (metrics["tau_deg"], metrics["psi_deg"], metrics["mu_deg"]) == pytest.approx((0., 0., 0.), abs=1e-9)
        assert metrics["rho_deg"] == pytest.approx(10.) and not row["warnings"]
        assert row["start_local_m"][2] == pytest.approx(.0202) and row["end_local_m"][2] == pytest.approx(.0202)
    assert waypoints[-1].pose == waypoints[0].pose
    # Passes are clipped at mid gaps, so no pass runs into a neighbouring protrusion.
    ends = [(row["start_local_m"][0], row["end_local_m"][0]) for row in report["operations"]]
    assert all(a[1] <= b[0]+1e-12 for a, b in zip(ends, ends[1:]))


def test_through_cut_relief_and_push_plan():
    _, report = plan_with_report(scene("bar_160x30x20"), goal("sever_mid"), PROFILES["ruler_knife_v1"])
    cut = report["operations"][0]
    assert not cut["end_on_support"] and cut["end_local_m"][2] == pytest.approx(-.004)
    assert cut["start_local_m"][2] == pytest.approx(.028)
    waypoints, report = plan_with_report(scene("bar_slotted_board_4slots"), goal("slice_then_push"), PROFILES["ruler_knife_v1"])
    cuts = [row for row in report["operations"] if row["kind"] == "through_cut"]
    assert [row["digit"] for row in cuts] == [0, 1] and all(row["end_on_support"] for row in cuts)
    assert all(row["end_local_m"][2] == pytest.approx(0.) for row in cuts)
    push = [w for w in waypoints if w.label.startswith(("LOWER", "PUSH", "RELEASE"))]
    assert [w.label.split(":")[0] for w in push] == ["LOWER", "PUSH", "PUSH_HOLD", "RELEASE"] and not any(w.cut for w in push)
    assert push[1].pose[0]-push[0].pose[0] == pytest.approx(.02) and push[0].pose[2] == pytest.approx(.05+.003)


# ---------------------------------------------------------------- yaw and randomization

def test_yaw_rotated_scene_compiles_to_rotated_particles_fixtures_and_plan():
    base = scene("bar_three_bumps")
    yawed = deepcopy(base)
    yawed["scene_id"] = "bar_three_bumps_yaw30"
    yawed["workpiece"]["world_pose"] = {"translation_m": [.01, -.02, .05], "yaw_deg": 30.}
    a = compile_task(base, goal("remove_all_protrusions"), particle_spacing_m=.004)
    b = compile_task(yawed, goal("remove_all_protrusions"), particle_spacing_m=.004)
    assert _same_material(a, b) and a.fixture_boxes_local == b.fixture_boxes_local
    R = Rotation.from_euler("z", 30., degrees=True)
    local, _, _ = b.sample_particles()
    assert np.allclose(b.points_to_world(local), R.apply(local)+[.01, -.02, .05], atol=1e-12)
    corners = raw_support_boxes(b.scene)[0].corners_local()
    assert np.allclose(b.points_to_world(corners), R.apply(corners)+[.01, -.02, .05])
    assert np.allclose(b.T_world_workpiece[2, :3], [0., 0., 1.])  # table plane normal in the solver frame
    for t in (0., 3., b.duration_s):
        assert np.allclose(b.reference_edge_pose_local(t), a.reference_edge_pose_local(t), atol=1e-9)
        world, expected = b.reference_edge_pose_world(t), R.apply(a.reference_edge_pose_local(t)[:3])+[.01, -.02, .05]
        assert np.allclose(world[:3], expected, atol=1e-9)
    with pytest.raises(ValueError):
        apply_scenario(b, "blademaster_workcell_v1")  # identity-only scenario path stays an explicit error


def test_open_cutting_env_rejects_configured_tasks_on_isaac_lab(tmp_path, bumps_task):
    # WP-11 Phase D: Isaac Sim accepts compiled tasks (tests/test_wp11_phase_d.py); Isaac Lab still raises.
    from robotic_cutting_sim.live_env import open_cutting_env
    with pytest.raises(NotImplementedError, match="registered task ids"):
        with open_cutting_env(runtime="isaac_lab", output_dir=tmp_path/"out", task=bumps_task):
            pass
    assert not (tmp_path/"out").exists()


def test_seeded_randomization_is_recorded_and_reproducible():
    doc = scene("bar_three_bumps")
    doc["randomization"] = {"seed": 7, "ranges": [
        {"path": "workpiece.world_pose.yaw_deg", "low": -20., "high": 20.},
        {"path": "workpiece.protrusions.*.x_m", "low": -.003, "high": .003, "mode": "offset"},
        {"path": "workpiece.base_box_m.length_m", "low": .15, "high": .17}]}
    first, again = resolve_scene(doc), resolve_scene(doc)
    assert first == again and len(first["randomization"]["sampled"]) == 5
    assert [row["path"] for row in first["randomization"]["sampled"]][1:4] == \
        [f"workpiece.protrusions.{i}.x_m" for i in range(3)]
    assert first["workpiece"]["world_pose"]["yaw_deg"] == first["randomization"]["sampled"][0]["value"]
    assert resolve_scene(first) == first  # a resolved scene is not drawn again
    other = deepcopy(doc)
    other["randomization"]["seed"] = 8
    assert resolve_scene(other)["workpiece"] != first["workpiece"]
    a = compile_task(doc, goal("remove_all_protrusions"), particle_spacing_m=.004)
    assert a.scene == first and a.task_id == compile_task(doc, goal("remove_all_protrusions"), particle_spacing_m=.004).task_id
    bad = deepcopy(doc)
    bad["randomization"]["ranges"] = [{"path": "workpiece.protrusions.2.x_m", "low": .07, "high": .075}]
    with pytest.raises(SceneInfeasible):
        resolve_scene(bad)


# ---------------------------------------------------------------- kinematic smokes (Warp CPU)

def _run_planner(env, task, limit_s):
    from robotic_cutting_sim.policies import WaypointPolicy
    options = goal_planner.planner_options(task.goal, None)
    policy = WaypointPolicy(env.task, plan(task.scene, task.goal, task.tool_profile, options),
                            goal_planner.action_limits(options), **goal_planner.policy_options(options))
    observation, _ = env.reset()
    while env.time_s < task.duration_s-1e-9:
        action = policy(observation)
        observation, _, terminated, truncated, info = env.step(action)
        assert info["ik_status"] == "ok" and not terminated
    assert policy.done
    return env.evaluate()


@pytest.mark.cpu_sim
def test_kinematic_smoke_two_protrusions_tlmpm_v4_law8():
    pytest.importorskip("warp")
    from robotic_cutting_sim.scene_config import make_configured_kinematic_env
    task = compile_task(with_solver(scene("bar_three_bumps")), remove(["p0", "p1"], 10.), particle_spacing_m=.004)
    env = make_configured_kinematic_env(task, material_dt=.0005, action_limits=goal_planner.action_limits(
        goal_planner.planner_options(task.goal)), delta_frame="world")
    assert env.task.task_id == task.task_id and env.backend.material is None
    result = _run_planner(env, task, task.duration_s)
    env.close()
    assert result["runtime_source"] == "kinematic_ideal_tracking" and result["episode_complete"]
    assert result["integrity"]["status"] == "PASS", result["integrity"]
    items = {row["id"]: row for row in result["outcome"]["metrics"]["items"]}
    assert items["p0"]["removed_mass_fraction"] >= .9 and items["p1"]["removed_mass_fraction"] >= .9
    assert items["p2"]["removed_mass_fraction"] <= .1
    assert result["verdict"] == "PASS", result["outcome"]


@pytest.mark.cpu_sim
def test_kinematic_smoke_three_protrusions_tlmpm_v5():
    pytest.importorskip("warp")
    from robotic_cutting_sim.kinematic_env import KinematicConfig
    if "solver" not in KinematicConfig.__dataclass_fields__:
        pytest.skip("NOT_RUN: tlmpm_v5 is not available in the kinematic backend (ADR 0014 pending)")
    from robotic_cutting_sim.scene_config import make_configured_kinematic_env
    task = compile_task(scene("bar_three_bumps"), "remove_all_protrusions", particle_spacing_m=.004)
    env = make_configured_kinematic_env(task, material_dt=.0005, action_limits=goal_planner.action_limits(
        goal_planner.planner_options(task.goal)), delta_frame="world")
    result = _run_planner(env, task, task.duration_s)
    env.close()
    assert result["integrity"]["status"] == "PASS", result["integrity"]
    assert result["code_encoding"] == "packed2"
    assert result["verdict"] == "PASS", result["outcome"]
