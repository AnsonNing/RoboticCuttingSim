"""BladeMaster demo runner (scripts/run_blademaster_demos.py): commands, refusals, evaluation, publish (CPU)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.policies import through_cut
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "blademaster_workcell_v1"
SPACING = .005
DT = .02


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load("run_blademaster_demos_test", ROOT/"scripts/run_blademaster_demos.py")


COMMON = ["--runtime", "isaac_sim"]
FLAGS = ["--preset", "policy", "--tool-attachment-mode", "physical_grasp", "--controller", "scripted",
         "--scenario", SCENARIO, "--record-cameras", "--overview-camera", "--record-every", "5",
         "--edge-feedback", "measured"]
EXPECTED_ARGS = {
    "1a": ["--task", "CUT-01-R", *FLAGS, "--cutting-law", "5", "--motion", "push", "--steps", "400"],
    "1b": ["--task", "CUT-01-R", *FLAGS, "--cutting-law", "5", "--motion", "oblique", "--angle", "20", "--steps", "400"],
    "2": ["--task", "TRIM-01-R", *FLAGS, "--cutting-law", "5", "--steps", "400"],
    "3": ["--task", "SLICE-PUSH-01", *FLAGS, "--cutting-law", "5", "--max-cuts", "2", "--steps", "800"],
}


# ------------------------------------------------------------------------------------------------ commands

@pytest.mark.parametrize("key", ["1a", "1b", "2", "3"])
def test_run_command_lines_are_exact(runner, key, tmp_path):
    spec = runner.DEMOS[key]
    demo_dir = tmp_path/"R"/spec.dirname
    argv = runner.run_argv(spec, demo_dir, python="PY", gpu=1, cutting_law=5, seed=42)
    assert argv == ["PY", str(ROOT/"scripts/run_wp10_live.py"), *COMMON, *EXPECTED_ARGS[key],
                    "--seed", "42", "--gpu", "1", "--out", str(demo_dir)]
    assert runner.run_argv(spec, demo_dir, python="PY", gpu=0, cutting_law=4, seed=7)[-8:] == \
        ["--steps", str(spec.steps), "--seed", "7", "--gpu", "0", "--out", str(demo_dir)]


def _parse_live(argv):
    """Parse a demo command with the real run_wp10_live parser, stopping before any runtime work."""
    module = _load("run_wp10_live_runner_test", ROOT/"scripts/run_wp10_live.py")
    captured = {}

    class Stop(Exception):
        pass

    def stop(task, **options):
        captured.update(task=task, options=options)
        raise Stop

    module.open_cutting_env = stop
    module.new_output = lambda path: Path(path)
    module.write_json = lambda *a, **k: None
    module.manifest = lambda config, kind: captured.setdefault("config", config)
    captured["status"] = module.main(argv)
    return captured


@pytest.mark.parametrize("key", ["1a", "1b", "2", "3"])
def test_run_commands_pass_the_real_run_wp10_live_parser_and_match_the_expected_manifest(runner, key, tmp_path):
    spec = runner.DEMOS[key]
    argv = runner.run_argv(spec, tmp_path/spec.dirname, python="PY")[2:]
    out = _parse_live(argv)
    assert out["task"] == spec.task_id and out["status"] == 1  # stopped at runtime launch, after validation
    for name, value in runner.expected_config(spec).items():
        assert out["config"][name] == value, name
    assert out["config"]["cutting_law"] == 5 and out["options"]["overview_camera"] is True
    assert out["options"]["max_cuts"] == (2 if key == "3" else 1)


def test_dry_run_prints_exact_commands_and_writes_nothing(runner, tmp_path, capsys):
    root, media = tmp_path/"demos", tmp_path/"media"
    assert runner.main(["--demo", "all", "--dry-run", "--run-id", "R1", "--root", str(root),
                        "--media-root", str(media), "--publish", "--gpu", "0"]) == 0
    text = capsys.readouterr().out
    assert not root.exists() and not media.exists()
    for key in ("1a", "1b", "2", "3"):
        spec = runner.DEMOS[key]
        line = runner.shell_line(runner.run_argv(spec, root/"R1"/spec.dirname, gpu=0), runner.DEFAULT_TMPDIR)
        assert line in text
        assert f"--title '{spec.title}'" in text
        assert f"R1_{spec.dirname}.gif" in text
    assert text.count("run_wp10_live.py") == 4 and text.count("render_demo_episode.py") == 4


def test_stage_selection_and_demo_selection(runner):
    assert runner.selected_demos(None) == ["1a", "1b", "2", "3"]
    assert runner.selected_demos(["3", "1a", "3"]) == ["1a", "3"]
    assert runner.selected_stages(None, False) == ["run", "evaluate", "render"]
    assert runner.selected_stages(["all"], True) == ["run", "evaluate", "render", "publish"]
    assert runner.selected_stages(["render", "evaluate"], False) == ["evaluate", "render"]
    assert runner.selected_stages(["publish"], False) == ["publish"]


def test_refuses_to_overwrite_an_existing_episode_dir(runner, tmp_path, monkeypatch):
    root = tmp_path/"demos"
    existing = root/"R1"/"demo1a"
    existing.mkdir(parents=True)
    (existing/"marker").write_text("keep")
    launched = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: launched.append(a))
    with pytest.raises(SystemExit):
        runner.main(["--demo", "1a", "--demo", "2", "--stage", "run", "--run-id", "R1", "--root", str(root)])
    assert not launched and (existing/"marker").read_text() == "keep" and not (root/"R1"/"demo2").exists()
    options = runner.build_parser().parse_args(["--run-id", "R1", "--root", str(root)])
    with pytest.raises(runner.StageError):
        runner.stage_run(options, runner.DEMOS["1a"])


def test_next_free_revisions_never_reuse_a_name(runner, tmp_path):
    path = tmp_path/"demo_summary.json"
    assert runner.next_free(path) == path and runner.latest_existing(path) is None
    path.write_text("{}")
    assert runner.next_free(path) == tmp_path/"demo_summary_r002.json" and runner.latest_existing(path) == path
    (tmp_path/"render").mkdir()
    assert runner.next_free(tmp_path/"render") == tmp_path/"render_r002"
    assert runner.revision_suffix(tmp_path/"render_r002", tmp_path/"render") == "_r002"
    assert runner.revision_suffix(tmp_path/"render", tmp_path/"render") == ""
    with pytest.raises(FileExistsError):
        runner.write_new_json(path, {"a": 1})


def test_run_stage_writes_receipts_and_records_failures(runner, tmp_path, monkeypatch):
    root = tmp_path/"demos"
    options = runner.build_parser().parse_args(["--run-id", "R1", "--root", str(root), "--native-python", "PY"])
    calls = []

    class Process:
        pid = 1

        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
            Path(argv[argv.index("--out")+1]).mkdir(parents=True)

        def wait(self, timeout=None):
            return 3

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: {"stub": True})
    monkeypatch.setattr(runner, "git_info", lambda: {"stub": True})
    with pytest.raises(runner.StageError, match="exited with 3"):
        runner.stage_run(options, runner.DEMOS["2"])
    argv, kwargs = calls[0]
    assert argv[0] == "PY" and kwargs["env"]["TMPDIR"] == runner.DEFAULT_TMPDIR and kwargs["cwd"] == runner.ROOT
    receipt = json.loads((root/"R1"/"demo2"/"run_receipt.json").read_text())
    assert receipt["exit_code"] == 3 and receipt["command"] == argv
    assert json.loads((root/"R1"/"logs"/"demo2_run.json").read_text()) == receipt


# ------------------------------------------------------------------------------------------------ fixtures

def _chain(x, dx=.008, origin=-.2):
    cell = np.floor((np.asarray(x)-origin)/dx).astype(int)
    nodes = np.zeros((len(x), 8), np.int32)
    nodes[:, 0], nodes[:, 1] = cell, cell+1
    weights = np.zeros((len(x), 8), np.float32)
    weights[:, :2] = .5
    return nodes, weights, np.zeros((len(x), 8, 3), np.float32)


def _task(task_id):
    return configure_task(make_task(task_id, particle_spacing_m=SPACING), scenario=SCENARIO)


def _rows(steps, edge_at, active_at, grasp_lost_at=None):
    rows = []
    for k in range(steps):
        t = round((k+1)*DT, 10)
        lost = grasp_lost_at is not None and t >= grasp_lost_at
        pose = edge_at(t)
        rows.append(dict(step=k+1, time_s=t, info={"ik_status": "ok"},
                         grasp_status={"status": "LOST" if lost else "HOLDING", "grasp_lost": lost,
                                       "slip_since_settling_m": 1e-6},
                         privileged_diagnostics={"active_cut": bool(active_at(t)), "cut_index": 0 if active_at(t) else -1,
                                                 "measured_tool_pose": list(map(float, pose)), "grasp_lost": lost,
                                                 "material_contact_statistics": [0, 5 if active_at(t) else 0]}))
    return rows


def _cut_edge(task, tilt_deg):
    plan = through_cut(task, tilt_deg=tilt_deg)
    above, start, end, _, retract = (np.asarray(w.pose, float) for w in plan)
    knots = [(0., above), (1.0, above), (1.6, start), (4.6, end), (5.0, end), (6.0, retract), (8.0, retract)]

    def edge(t):
        for (t0, p0), (t1, p1) in zip(knots, knots[1:]):
            if t <= t1+1e-12:
                a = (t-t0)/(t1-t0)
                return np.r_[p0[:3]+a*(p1[:3]-p0[:3]), p0[3:]]
        return knots[-1][1]
    rotation = Rotation.from_quat(start[3:])
    plane_normal = np.cross(rotation.apply([1., 0., 0.]), rotation.apply([0., 1., 0.]))
    return edge, (lambda t: 1.6 < t <= 4.6+1e-12), (start[:3]+end[:3])/2, plane_normal


def _write_episode(runner, demo_dir, spec, task, physics, rows, *, strict=None, research=None, demo=None,
                   push_phase=None, exit_code=0, steps=None, runtime_source="live_isaac_sim"):
    episode = demo_dir/"episode_000"
    (episode/"observations").mkdir(parents=True)
    times = np.asarray(physics["time_s"], float)
    for index, t in enumerate(times):
        np.savez_compressed(episode/"observations"/f"overview_rgb_{index:04d}.npz", rgb=np.zeros((2, 2, 3), np.uint8),
                            physics_timestamp_s=float(t))
    np.savez_compressed(episode/"physics_state.npz", **physics)
    config = {**runner.expected_config(spec), "cutting_law": 5, "gpu": 1, "seed": 42}
    runner.write_new_json(demo_dir/"manifest.json", {"config": config})
    row = dict(runtime="isaac_sim", task_id=spec.task_id, controller="scripted", seed=42, status="PASS_EXECUTION",
               strict=strict or {"status": "NOT_RUN"}, research=research or {"status": "EVALUATED"},
               runtime_source=runtime_source, observation_profile="state", wall_s=100., policy_plan="through_cut",
               policy_final_phase="done", steps=spec.steps if steps is None else steps,
               events={"rejected": 0, "clamped": 1, "cutting_steps": 150},
               tracking_error_m={"p50": 1e-4, "p95": 2e-4, "max": 3e-4})
    if demo is not None:
        row.update(demo=demo, push_phase=push_phase)
    runner.write_new_json(episode/"result.json", row)
    runner.write_new_json(demo_dir/"result.json", {"status": "PASS_EXECUTION", "rows": [row], "wall_s": 120.})
    runner.write_new_json(demo_dir/"run_receipt.json", {"exit_code": exit_code, "wall_s": 130., "timed_out": False})
    runner.write_new_json(episode/"task.json", task.to_dict())
    runner.write_new_json(episode/"frame_manifest.json", [{"frame": i, "time_s": float(t)} for i, t in enumerate(times)])
    with (episode/"actions.jsonl").open("w") as stream:
        for r in rows:
            stream.write(json.dumps(r)+"\n")
    return episode


def _cut_physics(task, labels_after_cut, remerge_at=None):
    points, masses, lattice = task.sample_particles()
    world = task.points_to_world(points)
    nodes, weights, gradients = _chain(points[:, 0])
    time = np.round(np.arange(0., 8.0001, .1), 10)
    codes = np.zeros((len(time), len(points)), np.int32)
    codes[time >= 4.0] = labels_after_cut
    if remerge_at is not None:
        codes[time >= remerge_at] = 0
    return dict(time_s=time, positions=np.repeat(world[None], len(time), 0), codes=codes, mass_kg=masses,
                reference_positions=world, reference_positions_local=points.astype(np.float32), reference_indices=lattice,
                reference_stencil_nodes=nodes, reference_stencil_weights=weights, reference_stencil_gradients=gradients)


def _hashes(episode):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(episode.rglob("*")) if p.is_file()}


def _cut_episode(runner, tmp_path, key, *, label_shift_m=0., remerge_at=None, grasp_lost_at=None, exit_code=0,
                 runtime_source="live_isaac_sim"):
    spec = runner.DEMOS[key]
    task = _task("CUT-01-R")
    edge, active, origin, normal = _cut_edge(task, 20. if key == "1b" else 0.)
    world = task.points_to_world(task.sample_particles()[0])
    labels = np.where((world-origin)@normal < label_shift_m, 1, 2).astype(np.int32)
    physics = _cut_physics(task, labels, remerge_at)
    rows = _rows(spec.steps, edge, active, grasp_lost_at)
    strict = {"status": "PASS_TASK_METRICS" if key == "1a" else "FAIL_TASK_METRICS",
              "checks": {"cut_at_workpiece_local_target_plane": key == "1a"},
              "metrics": {"maximum_reference_midpoint_plane_distance_m": .0015 if key == "1a" else .009}}
    demo_dir = tmp_path/spec.dirname
    episode = _write_episode(runner, demo_dir, spec, task, physics, rows, strict=strict, exit_code=exit_code,
                             runtime_source=runtime_source)
    return spec, demo_dir, episode


# ------------------------------------------------------------------------------------------------ evaluation

def test_demo1a_complete_cut_pass_and_read_only(runner, tmp_path):
    spec, demo_dir, episode = _cut_episode(runner, tmp_path, "1a")
    before = _hashes(episode)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    assert _hashes(episode) == before and summary["episode_unchanged_by_evaluation"]
    assert summary["execution"]["status"] == "PASS_EXECUTION", summary["execution"]["checks"]
    assert summary["outcome"]["status"] == "PASS", summary["outcome"]["checks"]
    assert summary["verdict"] == "PASS" and summary["control_mode"] == runner.CONTROL_MODE
    assert summary["cutting_law"] == 5
    persistence = summary["outcome"]["metrics"]["withdrawal_and_persistence"]
    assert 5.0 < persistence["withdrawal_time_s"] <= 6.0 and min(persistence["components_after_withdrawal"]) == 2
    location = summary["quality"]["location"]
    assert location["check_cut_at_workpiece_local_target_plane"] is True and location["threshold_m"] == .002
    assert summary["quality"]["frozen_strict"]["status"] == "PASS_TASK_METRICS"
    assert summary["measured_capture_rate"]["rate_hz"] == pytest.approx(10.)
    assert summary["artifacts"]["files"]["physics_state.npz"]["sha256"] == before["physics_state.npz"]
    assert summary["label"].startswith("outcome PASS (2 components 50/50%")
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("case,failed", [
    (dict(label_shift_m=-.2), "two_substantial_parts"),                      # nothing severed
    (dict(remerge_at=6.5), "separation_persists_after_withdrawal"),
    (dict(grasp_lost_at=3.0), "no_tool_drop")])
def test_demo1a_complete_cut_failures(runner, tmp_path, case, failed):
    spec, demo_dir, _ = _cut_episode(runner, tmp_path, "1a", **case)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    assert summary["outcome"]["status"] == "FAIL" and summary["outcome"]["checks"][failed] is False
    assert summary["verdict"] == "FAIL"


def test_execution_layer_failures_are_separate_from_the_outcome(runner, tmp_path):
    spec, demo_dir, _ = _cut_episode(runner, tmp_path, "1a", exit_code=1, runtime_source="kinematic_ideal_tracking")
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    checks = summary["execution"]["checks"]
    assert not checks["physics_process_exit_code_zero"] and not checks["live_native_runtime"]
    assert summary["outcome"]["status"] == "PASS" and summary["verdict"] == "FAIL"
    assert "execution FAIL_EXECUTION" in summary["label"]


def test_demo1b_oblique_swept_plane_pass(runner, tmp_path):
    spec, demo_dir, _ = _cut_episode(runner, tmp_path, "1b")
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    outcome = summary["outcome"]
    assert outcome["status"] == "PASS", outcome["checks"]
    swept = outcome["metrics"]["swept_plane"]
    assert swept["status"] == "FIT" and swept["active_cut_steps"] == 150
    assert swept["rescore"]["verdict"] == "PASS"
    assert summary["quality"]["swept_plane_tilt_from_target_plane_deg"] == pytest.approx(20., abs=.1)
    # The frozen vertical-plane location check is reported as recorded, never used for the outcome.
    assert summary["quality"]["location"]["check_cut_at_workpiece_local_target_plane"] is False
    assert summary["verdict"] == "PASS"


def test_demo1b_labels_off_the_measured_swept_plane_fail(runner, tmp_path):
    spec, demo_dir, _ = _cut_episode(runner, tmp_path, "1b", label_shift_m=4*SPACING)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    checks = summary["outcome"]["checks"]
    assert summary["outcome"]["status"] == "FAIL"
    assert not checks["swept_plane_interface_le_ideal_plus_one_spacing"] or not checks["no_far_particle_in_opposite_side_component"]


def _trim_episode(runner, tmp_path, *, cut=True, surface_ok=False):
    spec = runner.DEMOS["2"]
    task = _task("TRIM-01-R")
    points, masses, lattice = task.sample_particles()
    world = task.points_to_world(points)
    nodes, weights, gradients = _chain(points[:, 2], dx=.008, origin=-.02)
    time = np.round(np.arange(0., 8.0001, .1), 10)
    codes = np.zeros((len(time), len(points)), np.int32)
    if cut:
        codes[time >= 5.] = np.where(points[:, 2] > task.target_origin_local_m[2], 2, 1)
    physics = dict(time_s=time, positions=np.repeat(world[None], len(time), 0), codes=codes, mass_kg=masses,
                   reference_positions=world, reference_indices=lattice, reference_stencil_nodes=nodes,
                   reference_stencil_weights=weights, reference_stencil_gradients=gradients)
    edge = np.asarray(task.reference_edge_pose_world(0.), float)
    rows = _rows(spec.steps, lambda t: edge, lambda t: 2. < t < 5.)
    surface = {name: surface_ok for name in ("maximum_substrate_surface_residual_height_m", "maximum_gouge_depth_m",
                                             "overcut_volume_fraction_of_initial_base", "missing_surface_fraction")}
    strict = {"status": "PASS_TASK_METRICS" if surface_ok and cut else "FAIL_TASK_METRICS",
              "checks": {"reference_above_plane_material_removed": cut, "complete_episode": True, **surface},
              "metrics": {"surface": {"maximum_gouge_depth_m": .003}}}
    demo_dir = tmp_path/spec.dirname
    _write_episode(runner, demo_dir, spec, task, physics, rows, strict=strict)
    return spec, demo_dir


def test_demo2_trim_separation_outcome_is_separate_from_precision(runner, tmp_path):
    spec, demo_dir = _trim_episode(runner, tmp_path, cut=True, surface_ok=False)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    assert summary["outcome"]["status"] == "PASS", summary["outcome"]["checks"]
    assert summary["outcome"]["metrics"]["removed_reference_above_plane_mass_fraction"] == 1.
    precision = summary["quality"]["precision_score"]
    assert precision["status"] == "FAIL_PRECISION" and precision["passed"] == 0 and precision["total"] == 4
    assert "reference_above_plane_material_removed" not in precision["checks"]
    assert summary["verdict"] == "PASS" and "precision 0/4" in summary["label"]


def test_demo2_uncut_trim_fails(runner, tmp_path):
    spec, demo_dir = _trim_episode(runner, tmp_path, cut=False, surface_ok=True)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    assert summary["outcome"]["status"] == "FAIL"
    assert not summary["outcome"]["checks"]["reference_above_plane_material_removed"]
    assert summary["quality"]["precision_score"]["status"] == "PASS_PRECISION" and summary["verdict"] == "FAIL"


def _slice_episode(runner, tmp_path, *, push_m=.015, push_completed=True):
    spec = runner.DEMOS["3"]
    task = _task("SLICE-PUSH-01")
    points, masses, lattice = task.sample_particles()
    world = task.points_to_world(points)
    nodes, weights, gradients = _chain(points[:, 0])
    time = np.round(np.arange(0., 16.0001, .1), 10)
    codes = np.zeros((len(time), len(points)), np.int32)
    codes[time >= task.cut_intervals_s[0][1]] += np.where(points[:, 0] < -.03, 1, 2).astype(np.int32)
    codes[time >= task.cut_intervals_s[1][1]] += 3*np.where(points[:, 0] < .03, 1, 2).astype(np.int32)
    start, stop = task.push_interval_s
    ramp = np.clip((time-start)/(stop-start), 0., 1.)
    positions = np.repeat(world[None], len(time), 0)
    positions[:, points[:, 0] > .03, :] += (push_m*ramp)[:, None, None]*task.push_direction_world()
    physics = dict(time_s=time, positions=positions, codes=codes, mass_kg=masses, reference_positions=world,
                   reference_indices=lattice, reference_stencil_nodes=nodes, reference_stencil_weights=weights,
                   reference_stencil_gradients=gradients, push_interval_s=np.asarray([start, stop]),
                   push_completed=np.asarray(push_completed))
    edge = np.asarray(task.reference_edge_pose_world(0.), float)
    rows = _rows(spec.steps, lambda t: edge, lambda t: False)
    demo_dir = tmp_path/spec.dirname
    _write_episode(runner, demo_dir, spec, task, physics, rows, strict={"status": "NOT_APPLICABLE"},
                   demo={"status": "PASS_SLICE_PUSH"}, push_phase={"interval_s": [start, stop], "completed": push_completed})
    return spec, demo_dir


def test_demo3_slice_push_pass(runner, tmp_path):
    spec, demo_dir = _slice_episode(runner, tmp_path)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    outcome = summary["outcome"]
    assert outcome["status"] == "PASS", outcome["checks"]
    assert outcome["slice_push_report"]["push_interval_source"] == "recorded_push_phase"
    assert outcome["push_phase"]["completed"] is True
    assert summary["quality"]["recorded_live_demo_agrees"] is True and summary["verdict"] == "PASS"


@pytest.mark.parametrize("case,failed", [(dict(push_m=0.), "slice_push_report_pass"),
                                         (dict(push_completed=False), "push_phase_completed")])
def test_demo3_slice_push_failures(runner, tmp_path, case, failed):
    spec, demo_dir = _slice_episode(runner, tmp_path, **case)
    summary = runner.evaluate_demo_dir(spec, demo_dir)
    assert summary["outcome"]["status"] == "FAIL" and summary["outcome"]["checks"][failed] is False


def test_missing_episode_is_reported_not_scored(runner, tmp_path):
    demo_dir = tmp_path/"demo3"
    demo_dir.mkdir()
    runner.write_new_json(demo_dir/"failure.json", {"error": "RuntimeError('boom')"})
    summary = runner.evaluate_demo_dir(runner.DEMOS["3"], demo_dir)
    assert summary["execution"]["status"] == "FAIL_EXECUTION" and summary["outcome"]["status"] == "NOT_RUN"
    assert summary["verdict"] == "FAIL"
    assert runner.evaluate_demo_dir(runner.DEMOS["3"], tmp_path/"absent")["verdict"] == "NOT_RUN"


def test_evaluate_stage_never_overwrites_and_index_summarises(runner, tmp_path):
    root = tmp_path/"demos"
    _cut_episode(runner, root/"R1", "1a")
    options = runner.build_parser().parse_args(["--run-id", "R1", "--root", str(root)])
    first = runner.stage_evaluate(options, runner.DEMOS["1a"])
    original = (root/"R1"/"demo1a"/"demo_summary.json").read_text()
    second = runner.stage_evaluate(options, runner.DEMOS["1a"])
    assert second["summary_path"].endswith("demo_summary_r002.json")
    assert (root/"R1"/"demo1a"/"demo_summary.json").read_text() == original and first["verdict"] == "PASS"
    index = json.loads(runner.write_index(options).read_text())
    assert index["demos"]["1a"]["verdict"] == "PASS" and index["demos"]["1a"]["summary"] == second["summary_path"]
    assert set(index["demos"]) == {"1a"} and index["control_mode"] == runner.CONTROL_MODE
    assert runner.write_index(options).name == "index_r002.json"


# ------------------------------------------------------------------------------------------------ render/publish

def test_render_uses_the_evaluation_label_and_a_fresh_output(runner, tmp_path, monkeypatch):
    root = tmp_path/"demos"
    _cut_episode(runner, root/"R1", "1a")
    options = runner.build_parser().parse_args(["--run-id", "R1", "--root", str(root), "--render-python", "PY"])
    with pytest.raises(runner.StageError, match="evaluate stage first"):
        runner.stage_render(options, runner.DEMOS["1a"])
    summary = runner.stage_evaluate(options, runner.DEMOS["1a"])
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv) or
                        type("R", (), {"returncode": 0})())
    (root/"R1"/"demo1a"/"render").mkdir()  # an earlier render is kept
    out = runner.stage_render(options, runner.DEMOS["1a"])
    assert out == root/"R1"/"demo1a"/"render_r002"
    assert calls[0] == ["PY", str(ROOT/"scripts/render_demo_episode.py"), str(root/"R1"/"demo1a"/"episode_000"),
                        "--out", str(out), "--title", runner.DEMOS["1a"].title, "--label", summary["label"]]


def _fake_render(runner, demo_dir, name="render"):
    render = demo_dir/name
    render.mkdir()
    for file, size in (("demo.gif", 100), ("poster.png", 50), ("demo.mp4", 2_000_000)):
        (render/file).write_bytes(b"x"*size)
    runner.write_new_json(render/"render_receipt.json", {"outputs": {}})
    return render


def test_publish_names_include_run_id_and_never_overwrite(runner, tmp_path):
    root, media = tmp_path/"demos", tmp_path/"media"
    _cut_episode(runner, root/"R1", "1a")
    options = runner.build_parser().parse_args(["--run-id", "R1", "--root", str(root), "--media-root", str(media),
                                                "--publish-max-mb", "1"])
    spec = runner.DEMOS["1a"]
    runner.stage_evaluate(options, spec)
    _fake_render(runner, root/"R1"/"demo1a")
    published = runner.stage_publish(options, spec)
    target = media/"demo1a"
    assert sorted(p.name for p in target.iterdir()) == ["R1_demo1a.gif", "R1_demo1a.json", "R1_demo1a_poster.png"]
    assert set(published) == {"demo.gif", "poster.png"}  # the 2 MB mp4 exceeds --publish-max-mb 1
    provenance = json.loads((target/"R1_demo1a.json").read_text())
    assert provenance["verdict"] == "PASS" and "demo.mp4" in provenance["skipped"]
    assert provenance["runtime_source"] == "live_isaac_sim" and provenance["cutting_law"] == 5
    gif = (target/"R1_demo1a.gif").read_bytes()
    with pytest.raises(runner.StageError, match="refusing to overwrite"):
        runner.stage_publish(options, spec)
    assert (target/"R1_demo1a.gif").read_bytes() == gif
    _fake_render(runner, root/"R1"/"demo1a", "render_r002")       # a re-render publishes under a new name
    runner.stage_publish(options, spec)
    assert (target/"R1_r002_demo1a.gif").is_file() and (root/"R1"/"demo1a"/"publish_receipt_r002.json").is_file()
    assert runner.publish_names(runner.DEMOS["3"], "X") == {"demo.gif": "X_demo3.gif", "poster.png": "X_demo3_poster.png",
                                                             "demo.mp4": "X_demo3.mp4", "provenance": "X_demo3.json"}


# ------------------------------------------------------------------------------------------------ controller profile

def _set_manifest(demo_dir, **changes):
    path = demo_dir/"manifest.json"
    manifest = json.loads(path.read_text())
    config = manifest["config"]
    for key, value in changes.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("key", ["1a", "1b", "2", "3"])
def test_controller_profile_passthrough_and_v1_default(runner, key, tmp_path):
    spec = runner.DEMOS[key]
    demo_dir = tmp_path/spec.dirname
    v1 = runner.run_argv(spec, demo_dir, python="PY")
    assert v1 == runner.run_argv(spec, demo_dir, python="PY", controller_profile="v1") and "--controller-profile" not in v1
    argv = runner.run_argv(spec, demo_dir, python="PY", controller_profile="path_v2")
    assert argv == [*v1[:-8], "--controller-profile", "path_v2", *v1[-8:]]
    out = _parse_live(argv[2:])
    assert out["status"] == 1
    for name, value in runner.expected_config(spec, "path_v2").items():
        assert out["config"][name] == value, name
    assert runner.expected_config(spec)["controller_profile"] == "v1"
    assert runner.build_parser().parse_args([]).controller_profile == "v1"
    with pytest.raises(ValueError):
        runner.run_argv(spec, demo_dir, controller_profile="v9")


def test_dry_run_and_run_stage_pass_the_profile(runner, tmp_path, capsys, monkeypatch):
    root = tmp_path/"demos"
    assert runner.main(["--demo", "2", "--dry-run", "--run-id", "R1", "--root", str(root), "--controller-profile",
                        "path_v2"]) == 0
    assert "--controller-profile path_v2 --steps 400" in capsys.readouterr().out and not root.exists()
    calls = []

    class Process:
        pid = 1

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            Path(argv[argv.index("--out")+1]).mkdir(parents=True)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: {"stub": True})
    monkeypatch.setattr(runner, "git_info", lambda: {"stub": True})
    options = runner.build_parser().parse_args(["--run-id", "R1", "--root", str(root), "--native-python", "PY",
                                                "--controller-profile", "path_v2"])
    receipt = runner.stage_run(options, runner.DEMOS["2"])
    assert calls[0][calls[0].index("--controller-profile")+1] == "path_v2" and receipt["controller_profile"] == "path_v2"


def test_evaluation_checks_the_recorded_controller_profile(runner, tmp_path):
    spec, demo_dir, _ = _cut_episode(runner, tmp_path/"legacy", "1a")
    _set_manifest(demo_dir, controller_profile=None)                    # recorded before the flag existed
    legacy = runner.evaluate_demo_dir(spec, demo_dir)
    assert legacy["verdict"] == "PASS" and legacy["controller_profile"]["recorded"] == "v1"
    assert legacy["execution"]["controller_profile"]["expected"] == "v1" and "controller" not in legacy["label"]
    wrong = runner.evaluate_demo_dir(spec, demo_dir, controller_profile="path_v2")
    assert wrong["verdict"] == "FAIL" and not wrong["execution"]["checks"]["command_matches_demo_definition"]
    assert wrong["execution"]["manifest_mismatches"] == {"controller_profile": {"expected": "path_v2", "recorded": "v1"}}
    spec, demo_dir, _ = _cut_episode(runner, tmp_path/"v2", "1a")
    _set_manifest(demo_dir, controller_profile="path_v2")
    summary = runner.evaluate_demo_dir(spec, demo_dir, controller_profile="path_v2")
    assert summary["verdict"] == "PASS" and summary["label"].endswith("| controller path_v2")
    assert summary["controller_profile"]["parameters"] == runner.CONTROLLER_PROFILES["path_v2"]
    assert runner.evaluate_demo_dir(spec, demo_dir)["verdict"] == "FAIL"   # v1 definition, path_v2 recording


# ------------------------------------------------------------------------------------------------ demo set final_v1

DEMO_ROOT = ROOT/"artifacts"/"wp10_claude"/"demos"
FINAL_V1_RUNS = {"1a": "live_20261029_pathv2_d", "1b": "live_20261029_pathv2_d", "2": "live_20261029_pathv2_c",
                 "3": "live_20261029_vision_e"}
FINAL_V1_PROFILES = {"1a": "path_v2", "1b": "path_v2", "2": "path_v2", "3": "path_v2_vision"}


def _normalise(argv):
    """Command without the values of --out and --gpu (the only parts allowed to differ)."""
    argv = list(argv)
    for flag in ("--out", "--gpu"):
        index = argv.index(flag)
        argv[index+1] = "<" + flag[2:] + ">"
    return argv


def _dry_run_commands(text):
    """run_wp10_live argv per demo key parsed from dry-run output (``cd ROOT && TMPDIR=... argv``)."""
    import shlex
    commands = {}
    for line in text.splitlines():
        if "run_wp10_live.py" not in line:
            continue
        argv = shlex.split(line.split(" && ", 1)[1])
        assert argv[0].startswith("TMPDIR=")
        argv = argv[1:]
        commands[{"demo1a": "1a", "demo1b": "1b", "demo2": "2", "demo3": "3"}[Path(argv[argv.index("--out")+1]).name]] = argv
    return commands


def test_final_v1_definition(runner):
    demo_set = runner.DEMO_SETS["final_v1"]
    assert demo_set.demos == ("1a", "1b", "2", "3") and dict(demo_set.profiles) == FINAL_V1_PROFILES
    assert demo_set.cutting_law == 5 and demo_set.seed == 42 and dict(demo_set.reference_runs) == FINAL_V1_RUNS
    assert set(FINAL_V1_PROFILES.values()) <= set(runner.CONTROLLER_PROFILES)


def test_final_v1_dry_run_matches_the_recorded_final_runs(runner, tmp_path, capsys):
    receipts = {key: DEMO_ROOT/run/runner.DEMOS[key].dirname/"run_receipt.json" for key, run in FINAL_V1_RUNS.items()}
    missing = [str(p) for p in receipts.values() if not p.is_file()]
    if missing:
        pytest.skip(f"recorded final runs absent: {missing}")
    root = tmp_path/"demos"
    assert runner.main(["--demo-set", "final_v1", "--dry-run", "--run-id", "R1", "--root", str(root), "--gpu", "0"]) == 0
    out = capsys.readouterr()
    assert not root.exists() and "first-round" not in out.err
    commands = _dry_run_commands(out.out)
    assert set(commands) == set(FINAL_V1_RUNS)
    for key, path in receipts.items():
        recorded = json.loads(path.read_text())
        assert recorded["exit_code"] == 0 and recorded["controller_profile"] == FINAL_V1_PROFILES[key]
        assert _normalise(commands[key]) == _normalise(recorded["command"]), key
        assert commands[key][commands[key].index("--gpu")+1] == "0"
        assert commands[key][commands[key].index("--out")+1] == str(root/"R1"/runner.DEMOS[key].dirname)
        manifest = json.loads((path.parent/"manifest.json").read_text())["config"]
        assert manifest["controller_profile"] == FINAL_V1_PROFILES[key] and manifest["cutting_law"] == 5


@pytest.mark.parametrize("key", ["1a", "1b", "2", "3"])
def test_final_v1_commands_pass_the_real_run_wp10_live_parser(runner, key, tmp_path):
    spec, profile = runner.DEMOS[key], FINAL_V1_PROFILES[key]
    argv = runner.run_argv(spec, tmp_path/spec.dirname, python="PY", controller_profile=profile)
    assert argv == [*runner.run_argv(spec, tmp_path/spec.dirname, python="PY")[:-8], "--controller-profile", profile,
                    *argv[-8:]]
    out = _parse_live(argv[2:])
    assert out["status"] == 1
    for name, value in runner.expected_config(spec, profile, 5).items():
        assert out["config"][name] == value, name


def test_final_v2_is_final_v1_with_cutting_law_6(runner, tmp_path, capsys):
    v1, v2 = runner.DEMO_SETS["final_v1"], runner.DEMO_SETS["final_v2"]
    assert v2.demos == v1.demos and v2.profiles == v1.profiles and v2.seed == v1.seed == 42
    assert v1.cutting_law == 5 and v2.cutting_law == 6 and v2.reference_runs == ()
    commands = {}
    for name in ("final_v1", "final_v2"):
        assert runner.main(["--demo-set", name, "--dry-run", "--run-id", "R1", "--root", str(tmp_path/"demos")]) == 0
        commands[name] = _dry_run_commands(capsys.readouterr().out)
    assert set(commands["final_v2"]) == set(FINAL_V1_PROFILES)
    for key, argv in commands["final_v2"].items():
        old = commands["final_v1"][key]
        law = argv.index("--cutting-law")
        assert argv[law+1] == "6" and old[law+1] == "5"
        assert argv[:law+1] == old[:law+1] and argv[law+2:] == old[law+2:]
        assert argv[1].endswith("run_wp10_live.py")
        out = _parse_live(argv[2:])
        assert out["status"] == 1 and out["config"]["cutting_law"] == 6
    with pytest.raises(SystemExit) as error:
        runner.main(["--demo-set", "final_v2", "--dry-run", "--run-id", "R1", "--root", str(tmp_path),
                     "--cutting-law", "5"])
    assert error.value.code == 2 and "pins --cutting-law 6" in capsys.readouterr().err


def test_final_v3_is_final_v1_with_cutting_law_7(runner, tmp_path, capsys):
    v1, v2, v3 = (runner.DEMO_SETS[name] for name in ("final_v1", "final_v2", "final_v3"))
    assert v3.demos == v1.demos and v3.profiles == v1.profiles and v3.seed == v1.seed == 42
    assert v1.cutting_law == 5 and v2.cutting_law == 6 and v3.cutting_law == 7 and v3.reference_runs == ()
    commands = {}
    for name in ("final_v1", "final_v3"):
        assert runner.main(["--demo-set", name, "--dry-run", "--run-id", "R1", "--root", str(tmp_path/"demos")]) == 0
        commands[name] = _dry_run_commands(capsys.readouterr().out)
    assert set(commands["final_v3"]) == set(FINAL_V1_PROFILES)
    for key, argv in commands["final_v3"].items():
        old = commands["final_v1"][key]
        law = argv.index("--cutting-law")
        assert argv[law+1] == "7" and old[law+1] == "5"
        assert argv[:law+1] == old[:law+1] and argv[law+2:] == old[law+2:]
        assert argv[1].endswith("run_wp10_live.py")
        out = _parse_live(argv[2:])
        assert out["status"] == 1 and out["config"]["cutting_law"] == 7
    with pytest.raises(SystemExit) as error:
        runner.main(["--demo-set", "final_v3", "--dry-run", "--run-id", "R1", "--root", str(tmp_path),
                     "--cutting-law", "5"])
    assert error.value.code == 2 and "pins --cutting-law 7" in capsys.readouterr().err


def test_final_v4_is_final_v1_with_cutting_law_8(runner, tmp_path, capsys):
    v1, v4 = runner.DEMO_SETS["final_v1"], runner.DEMO_SETS["final_v4"]
    assert v4.demos == v1.demos and v4.profiles == v1.profiles and v4.seed == v1.seed == 42
    assert v1.cutting_law == 5 and v4.cutting_law == 8 and v4.reference_runs == ()
    assert [runner.DEMO_SETS[n].cutting_law for n in ("final_v2", "final_v3")] == [6, 7]
    commands = {}
    for name in ("final_v1", "final_v4"):
        assert runner.main(["--demo-set", name, "--dry-run", "--run-id", "R1", "--root", str(tmp_path/"demos")]) == 0
        commands[name] = _dry_run_commands(capsys.readouterr().out)
    assert set(commands["final_v4"]) == set(FINAL_V1_PROFILES)
    for key, argv in commands["final_v4"].items():
        old = commands["final_v1"][key]
        law = argv.index("--cutting-law")
        assert argv[law+1] == "8" and old[law+1] == "5"
        assert argv[:law+1] == old[:law+1] and argv[law+2:] == old[law+2:]
        out = _parse_live(argv[2:])
        assert out["status"] == 1 and out["config"]["cutting_law"] == 8
    with pytest.raises(SystemExit) as error:
        runner.main(["--demo-set", "final_v4", "--dry-run", "--run-id", "R1", "--root", str(tmp_path),
                     "--cutting-law", "5"])
    assert error.value.code == 2 and "pins --cutting-law 8" in capsys.readouterr().err


@pytest.mark.parametrize("extra,message", [
    (["--demo", "all"], "mutually exclusive with --demo"),
    (["--demo", "3"], "mutually exclusive with --demo"),
    (["--controller-profile", "path_v2"], "mutually exclusive with --controller-profile"),
    (["--controller-profile", "v1"], "mutually exclusive with --controller-profile"),
    (["--cutting-law", "4"], "pins --cutting-law 5"),
    (["--seed", "7"], "pins --seed 42")])
def test_demo_set_exclusions(runner, tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as error:
        runner.main(["--demo-set", "final_v1", "--dry-run", "--run-id", "R1", "--root", str(tmp_path), *extra])
    assert error.value.code == 2 and message in capsys.readouterr().err


def test_demo_set_accepts_its_own_pinned_values(runner, tmp_path, capsys):
    assert runner.main(["--demo-set", "final_v1", "--dry-run", "--run-id", "R1", "--root", str(tmp_path),
                        "--cutting-law", "5", "--seed", "42"]) == 0
    assert capsys.readouterr().out.count("run_wp10_live.py") == 4


def test_v1_demo_all_prints_the_first_round_notice(runner, tmp_path, capsys):
    assert runner.main(["--demo", "all", "--dry-run", "--run-id", "R1", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert "this reproduces the first-round v1 configuration; use --demo-set final_v1 for the final set" in out.err
    assert "--controller-profile" not in out.out and out.out.count("run_wp10_live.py") == 4
    assert runner.main(["--dry-run", "--run-id", "R1", "--root", str(tmp_path)]) == 0       # no --demo = all
    assert "first-round" in capsys.readouterr().err
    assert runner.main(["--demo", "2", "--dry-run", "--run-id", "R1", "--root", str(tmp_path)]) == 0
    assert "first-round" not in capsys.readouterr().err                                   # a single demo: no notice


def test_v1_demo_all_with_path_v2_is_unchanged(runner, tmp_path, capsys):
    """--demo all --controller-profile path_v2 was valid before (path_v2 has no survey) and stays valid."""
    assert runner.main(["--demo", "all", "--controller-profile", "path_v2", "--dry-run", "--run-id", "R1",
                        "--root", str(tmp_path)]) == 0
    out = capsys.readouterr()
    assert out.out.count("--controller-profile path_v2 ") == 4 and "first-round" not in out.err
    with pytest.raises(SystemExit):
        runner.main(["--demo", "all", "--controller-profile", "path_v2_vision", "--dry-run", "--run-id", "R1",
                     "--root", str(tmp_path)])


def test_final_v1_run_stage_records_per_demo_profiles(runner, tmp_path, monkeypatch, capsys):
    root = tmp_path/"demos"
    calls = []

    class Process:
        pid = 1

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            Path(argv[argv.index("--out")+1]).mkdir(parents=True)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: {"stub": True})
    monkeypatch.setattr(runner, "git_info", lambda: {"stub": True})
    assert runner.main(["--demo-set", "final_v1", "--stage", "run", "--run-id", "R1", "--root", str(root),
                        "--native-python", "PY"]) == 0
    capsys.readouterr()
    assert len(calls) == 4
    for key, profile in FINAL_V1_PROFILES.items():
        dirname = runner.DEMOS[key].dirname
        receipt = json.loads((root/"R1"/dirname/"run_receipt.json").read_text())
        assert receipt["controller_profile"] == profile and receipt["demo_set"] == "final_v1"
        assert receipt["command"][receipt["command"].index("--controller-profile")+1] == profile
        assert receipt["command"][receipt["command"].index("--cutting-law")+1] == "5"
        assert json.loads((root/"R1"/"logs"/f"{dirname}_run.json").read_text()) == receipt
    index = json.loads((root/"R1"/"index.json").read_text())
    assert index["controller_profile"] == "per_demo" and index["controller_profiles"] == FINAL_V1_PROFILES
    assert index["demo_set"]["name"] == "final_v1"
    assert {k: v["expected_controller_profile"] for k, v in index["demos"].items()} == FINAL_V1_PROFILES


def _final_v1_run(runner, root):
    """A fake recorded final_v1 run R1 with one passing episode per demo and its per-demo profile."""
    run = root/"R1"
    for key in ("1a", "1b"):
        _cut_episode(runner, run, key)
    _trim_episode(runner, run)
    _slice_episode(runner, run)
    for key, profile in FINAL_V1_PROFILES.items():
        _set_manifest(run/runner.DEMOS[key].dirname, controller_profile=profile)
    return run


def test_final_v1_stage_rerun_evaluate_render_publish(runner, tmp_path, monkeypatch, capsys):
    root, media = tmp_path/"demos", tmp_path/"media"
    run = _final_v1_run(runner, root)
    base = ["--demo-set", "final_v1", "--run-id", "R1", "--root", str(root), "--media-root", str(media),
            "--render-python", "PY"]
    assert runner.main([*base, "--stage", "evaluate"]) == 0
    for key, profile in FINAL_V1_PROFILES.items():
        summary = json.loads((run/runner.DEMOS[key].dirname/"demo_summary.json").read_text())
        assert summary["verdict"] == "PASS", (key, summary["execution"]["checks"], summary["outcome"].get("checks"))
        assert summary["controller_profile"]["expected"] == profile == summary["controller_profile"]["recorded"]
        assert summary["execution"]["controller_profile"]["expected"] == profile
        assert summary["demo_set"]["name"] == "final_v1" and summary["label"].endswith(f"| controller {profile}")
        assert summary["execution"]["checks"]["command_matches_demo_definition"]
    index = json.loads((run/"index.json").read_text())
    assert {k: v["controller_profile"] for k, v in index["demos"].items()} == FINAL_V1_PROFILES
    assert {k: v["verdict"] for k, v in index["demos"].items()} == {k: "PASS" for k in FINAL_V1_PROFILES}

    calls = []
    monkeypatch.setattr(runner, "git_info", lambda: {"stub": True})
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv) or
                        type("R", (), {"returncode": 0})())
    assert runner.main([*base, "--stage", "render"]) == 0
    assert len(calls) == 4 and calls[3][calls[3].index("--label")+1].endswith("| controller path_v2_vision")
    for key in FINAL_V1_PROFILES:
        _fake_render(runner, run/runner.DEMOS[key].dirname)
    assert runner.main([*base, "--stage", "publish", "--publish-max-mb", "1"]) == 0
    for key in FINAL_V1_PROFILES:
        dirname = runner.DEMOS[key].dirname
        assert (media/dirname/f"R1_{dirname}.gif").is_file()
        assert json.loads((media/dirname/f"R1_{dirname}.json").read_text())["verdict"] == "PASS"
    index = json.loads(runner.latest_existing(run/"index.json").read_text())
    assert all(v["publish_receipt"] for v in index["demos"].values()) and index["controller_profiles"] == FINAL_V1_PROFILES
    capsys.readouterr()


def test_final_v1_evaluate_checks_each_demo_profile_and_cutting_law(runner, tmp_path, capsys):
    root = tmp_path/"demos"
    run = _final_v1_run(runner, root)
    _set_manifest(run/"demo3", controller_profile="path_v2")          # demo 3 recorded without the vision survey
    _set_manifest(run/"demo2", cutting_law=4)                         # demo 2 recorded with another cutting law
    assert runner.main(["--demo-set", "final_v1", "--run-id", "R1", "--root", str(root), "--stage", "evaluate"]) == 0
    capsys.readouterr()
    verdicts = {}
    for key in FINAL_V1_PROFILES:
        verdicts[key] = json.loads((run/runner.DEMOS[key].dirname/"demo_summary.json").read_text())
    assert verdicts["1a"]["verdict"] == verdicts["1b"]["verdict"] == "PASS"
    assert verdicts["3"]["verdict"] == "FAIL" and verdicts["3"]["execution"]["manifest_mismatches"] == {
        "controller_profile": {"expected": "path_v2_vision", "recorded": "path_v2"}}
    assert verdicts["2"]["verdict"] == "FAIL" and verdicts["2"]["execution"]["manifest_mismatches"] == {
        "cutting_law": {"expected": 5, "recorded": 4}}
    # The same run evaluated with the v1 single-profile definitions fails on the profile, as before.
    assert runner.main(["--demo", "1a", "--run-id", "R1", "--root", str(root), "--stage", "evaluate"]) == 0
    capsys.readouterr()
    v1 = json.loads((run/"demo1a"/"demo_summary_r002.json").read_text())
    assert v1["verdict"] == "FAIL" and v1["demo_set"] is None
    with pytest.raises(ValueError):
        runner.evaluate_demo_dir(runner.DEMOS["3"], run/"demo3", controller_profile="path_v2", demo_set="final_v1")
