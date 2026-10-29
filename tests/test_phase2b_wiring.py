"""Phase 2B (ADR 0007): runtime wiring of blademaster_workcell_v1 and SLICE-PUSH-01 (CPU, no Isaac)."""
import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting
from robotic_cutting_sim.native_fixture import LEGACY_FIXTURE_COLOR, authored_material_receipt, fixture_authoring_plan
from robotic_cutting_sim.presentation_camera import overview_camera_spec
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import PROFILE_TASK_IDS, DemoTaskSpec, ProfileTaskSpec, make_task

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "blademaster_workcell_v1"


def _script():
    spec = importlib.util.spec_from_file_location("run_wp10_live_phase2b", ROOT/"scripts/run_wp10_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native(**options):
    return IsaacSimCuttingConfig(**{"camera_mode": "native_single_view", "camera_layout": "wrist_d405_global_d455", **options})


# ---------------------------------------------------------------- guards
def test_profile_task_ids_add_only_the_demo_task():
    assert PROFILE_TASK_IDS == ProfileTaskSpec.TASK_IDS+DemoTaskSpec.TASK_IDS
    assert ProfileTaskSpec.TASK_IDS == ("CUT-01-R", "TRIM-01-R", "CUT-02")
    assert "SLICE-PUSH-01" in PROFILE_TASK_IDS and "CUT-01" not in PROFILE_TASK_IDS


def test_native_config_accepts_slice_push_with_physical_grasp_and_two_digits():
    config = IsaacSimCuttingConfig(task_id="SLICE-PUSH-01", tool_attachment_mode="physical_grasp", max_cuts=2,
                                   scenario=SCENARIO, cutting_law=4)
    assert config.max_cuts == 2 and config.overview_camera is False
    with pytest.raises(ValueError, match="physical_grasp"):
        IsaacSimCuttingConfig(task_id="SLICE-PUSH-01", max_cuts=2)
    with pytest.raises(ValueError, match="two cut digits"):
        IsaacSimCuttingConfig(task_id="SLICE-PUSH-01", tool_attachment_mode="physical_grasp")
    # Old behaviour: ruler tasks still force physical_grasp; v1 tasks still reject scenarios/two digits.
    with pytest.raises(ValueError, match="physical_grasp"):
        IsaacSimCuttingConfig(task_id="CUT-01-R")
    for options in (dict(scenario=SCENARIO), dict(max_cuts=2)):
        with pytest.raises(ValueError, match="opt-in v2"):
            IsaacSimCuttingConfig(task_id="CUT-01", **options)
    IsaacSimCuttingConfig(task_id="CUT-01-R", tool_attachment_mode="physical_grasp", max_cuts=2, scenario=SCENARIO)


def test_overview_camera_is_opt_in_and_needs_the_live_wrist_layout():
    assert _native(task_id="CUT-01-R", tool_attachment_mode="physical_grasp", overview_camera=True).overview_camera
    for options in (dict(camera=False), dict(camera_layout="legacy_fixed_d455"), dict(task_id=None)):
        with pytest.raises(ValueError, match="overview"):
            _native(**{"task_id": "CUT-01-R", "tool_attachment_mode": "physical_grasp", "overview_camera": True, **options})


def test_slice_push_loads_through_material_tool_and_scenario_consumers():
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.tool_profile import require_task_tool, tool_spec_for_task
    task = configure_task(make_task("SLICE-PUSH-01"), scenario=SCENARIO)
    assert isinstance(task, DemoTaskSpec)
    require_task_tool(tool_spec_for_task(task), task)
    material = LiveCuttingMaterial(task=task, max_cuts=2, cutting_law=4)
    assert material.max_cuts == 2 and material.cutting_law == 4 and material.tool_profile is task.tool_profile


# ---------------------------------------------------------------- fixture materials
def test_fixture_plan_uses_declared_colours_and_materials_only_for_v3():
    task = configure_task(make_task("SLICE-PUSH-01"), scenario=SCENARIO)
    plan = fixture_authoring_plan(task, SCENARIO)
    assert plan["any_material_declared"] and len(plan["boxes"]) == 6
    roles = [row["role"] for row in plan["boxes"]]
    assert roles == ["board", "riser"]*3
    for row in plan["boxes"]:
        assert row["material_declared"] and row["physics_material"]["static_friction"] == pytest.approx(task.material.friction)
        assert row["physics_material"]["dynamic_friction"] == pytest.approx(task.material.friction)
        assert row["physics_material"]["restitution"] == 0.
        expected = [.72, .53, .34] if row["role"] == "board" else [.55, .55, .57]
        assert row["display_color_rgb"] == pytest.approx(expected)
    assert plan["table"]["prim_path"] == "/World/Table"
    assert plan["table"]["physics_material"]["static_friction"] == pytest.approx(task.material.friction)
    for scenario in ("ruler_workcell_v2", None):
        legacy = configure_task(make_task("CUT-01-R"), scenario=scenario)
        old = fixture_authoring_plan(legacy, scenario)
        assert not old["any_material_declared"] and old["table"] is None
        assert all(row["display_color_rgb"] == list(LEGACY_FIXTURE_COLOR) and row["physics_material"] is None
                   for row in old["boxes"])


def _stage():
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.035))
    UsdGeom.Cube.Define(stage, "/World/TrayFloor")
    return stage


def _historical_author(stage, task):
    """Verbatim copy of the pre-Phase-2B _author_task_fixture body."""
    from pxr import Gf, UsdGeom, UsdPhysics
    for prim in list(stage.Traverse()):
        if str(prim.GetPath()).startswith("/World/Tray"):
            prim.SetActive(False)
    root = UsdGeom.Xform.Define(stage, "/World/TaskFixture")
    pose = np.asarray(task.world_from_workpiece_xyzw, float)
    root.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
    root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(pose[6]), Gf.Vec3d(*pose[3:6])))
    for index, box in enumerate(task.fixture_boxes_local):
        cube = UsdGeom.Cube.Define(stage, f"/World/TaskFixture/Box_{index}")
        cube.CreateSizeAttr(2.)
        cube.GetPrim().SetCustomDataByKey("research:fixture_name", box.name)
        cube.AddTranslateOp().Set(Gf.Vec3d(*box.center_m))
        cube.AddScaleOp().Set(Gf.Vec3f(*box.half_extents_m))
        cube.CreateDisplayColorAttr([Gf.Vec3f(.22, .27, .31)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    table = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Table"))
    for op in table.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(0, 0, task.table_top_world_z_m-.025))


def _runner_author(task, scenario):
    stage = _stage()
    env = LiveIsaacSimCutting(None, SimpleNamespace(scenario=scenario), "unused")
    env.stage, env.task = stage, task
    env._author_task_fixture()
    return stage, env


@pytest.mark.parametrize("task_id,scenario", [("CUT-01", None), ("CUT-01-R", None), ("CUT-01-R", "ruler_workcell_v2"),
                                              ("TRIM-01-R", "raised_workpiece_50mm_v1")])
def test_old_scenarios_author_bitwise_identical_usd(task_id, scenario):
    task = configure_task(make_task(task_id), scenario=scenario)
    expected = _stage()
    _historical_author(expected, task)
    actual, _ = _runner_author(task, scenario)
    assert actual.GetRootLayer().ExportToString() == expected.GetRootLayer().ExportToString()


@pytest.mark.parametrize("task_id", ["SLICE-PUSH-01", "CUT-01-R", "TRIM-01-R"])
def test_v3_fixture_binds_physics_materials_and_colours(task_id):
    from pxr import Usd, UsdGeom, UsdPhysics
    task = configure_task(make_task(task_id), scenario=SCENARIO)
    stage, env = _runner_author(task, SCENARIO)
    receipt = authored_material_receipt(stage, env.fixture_plan)
    assert len(receipt["boxes"]) == len(task.fixture_boxes_local)
    for index, (box, row) in enumerate(zip(task.fixture_boxes_local, receipt["boxes"])):
        prim = stage.GetPrimAtPath(f"/World/TaskFixture/Box_{index}")
        assert prim.HasAPI(UsdPhysics.CollisionAPI) and row["collision_api"]
        assert row["physics_material"] == f"/World/TaskFixtureMaterials/Box_{index}"
        assert (row["static_friction"], row["dynamic_friction"], row["restitution"]) == pytest.approx((.3, .3, 0.))
        assert prim.GetCustomDataByKey("research:fixture_role") == box.role
        color = np.asarray(UsdGeom.Cube(prim).GetDisplayColorAttr().Get()[0])
        np.testing.assert_allclose(color, box.display_color_rgb, atol=1e-6)
        # The visual cube is the collider: world corners equal the declared box corners.
        world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        np.testing.assert_allclose(world[:3, 3], task.points_to_world(np.asarray(box.center_m)), atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(world[:3, :3], axis=0), box.half_extents_m, atol=1e-7)
    table = receipt["table"]
    assert table["physics_material"] == "/World/TaskFixtureMaterials/Table"
    assert (table["static_friction"], table["dynamic_friction"], table["restitution"]) == pytest.approx((.3, .3, 0.))


# ---------------------------------------------------------------- overview camera
def test_overview_camera_frames_the_workpiece_franka0_and_knife():
    from robotic_cutting_sim.presentation_camera import PRIM_PATH, author_overview_camera
    from robotic_cutting_sim.usd_surface_bridge import usd_camera_calibration
    from pxr import Usd, UsdGeom
    for task_id in ("SLICE-PUSH-01", "CUT-01-R", "TRIM-01-R"):
        task = configure_task(make_task(task_id), scenario=SCENARIO)
        spec = overview_camera_spec(task)
        assert spec["observation"] is False and spec["camera_contract"] is False and spec["modalities"] == ["rgb"]
        assert .7 <= spec["working_distance_m"] <= .9
        np.testing.assert_allclose(spec["look_at_target_world_m"], [0., 0., .07], atol=1e-12)  # raised workpiece top
        eye = np.asarray(spec["eye_world_m"])
        assert eye[0] > 0 and eye[1] < 0  # front-right
        for name in ("workpiece_corners", "franka_0", "franka_0_shoulder_approx", "reference_edge_t0"):
            assert spec["framing_check"][name]["all_inside"], name
        stage = Usd.Stage.CreateInMemory()
        with pytest.raises(ValueError, match="SI"):
            author_overview_camera(stage, spec)  # default in-memory stage is centimetres
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        author_overview_camera(stage, spec)
        authored = usd_camera_calibration(stage, PRIM_PATH, spec["width"], spec["height"])
        np.testing.assert_allclose(authored["T_world_camera"], spec["T_world_camera"], atol=1e-6)
        np.testing.assert_allclose(authored["K"], spec["K"], rtol=1e-5)


# ---------------------------------------------------------------- run_wp10_live
def _args(*extra):
    return ["--runtime", "isaac_sim", "--out", "unused", *extra]


def test_step_limit_follows_the_task_duration_and_defaults_are_unchanged():
    module = _script()
    assert module.episode_step_limit("CUT-01") == 400 and module.episode_step_limit("TRIM-01-R") == 400
    assert module.episode_step_limit("SLICE-PUSH-01") == 800


def _parse(module, argv):
    """Parse with the real parser, stopping before any runtime work."""
    captured = {}

    class Stop(Exception):
        pass

    def stop(task, **options):
        captured.update(task=task, options=options)
        raise Stop

    module.open_cutting_env = stop
    module.new_output = lambda path: Path(path)
    module.write_json = lambda *a, **k: None
    module.manifest = lambda *a, **k: {}
    captured["status"] = module.main(argv)
    return captured


def test_run_wp10_live_parses_new_options_and_passes_them():
    module = _script()
    out = _parse(module, _args("--task", "SLICE-PUSH-01", "--scenario", SCENARIO, "--preset", "policy",
                               "--tool-attachment-mode", "physical_grasp", "--controller", "scripted", "--max-cuts", "2",
                               "--record-cameras", "--overview-camera", "--edge-feedback", "measured",
                               "--position-tolerance", "0.0003", "--steps", "800", "--record-every", "5"))
    assert out["task"] == "SLICE-PUSH-01" and out["status"] == 1  # stopped at runtime launch
    options = out["options"]
    assert options["overview_camera"] is True and options["scenario"] == SCENARIO and options["max_cuts"] == 2
    defaults = _parse(_script(), _args("--task", "CUT-01-R"))
    assert "overview_camera" not in defaults["options"] and defaults["options"]["scenario"] is None


@pytest.mark.parametrize("argv", [
    _args("--task", "CUT-01-R", "--steps", "401"),
    _args("--task", "SLICE-PUSH-01", "--steps", "801"),
    _args("--task", "CUT-01-R", "--overview-camera"),                     # no recorded cameras
    _args("--task", "CUT-01-R", "--edge-feedback", "measured"),           # not scripted
    _args("--task", "CUT-01-R", "--controller", "scripted", "--position-tolerance", "0"),
    _args("--task", "CUT-01-R", "--scenario", "unknown_v9"),
])
def test_run_wp10_live_rejects_invalid_new_options(argv):
    with pytest.raises(SystemExit):
        _parse(_script(), argv)


def _stub_env(task_id, scenario=SCENARIO):
    task = configure_task(make_task(task_id), scenario=scenario)
    calls = []

    class Env:
        runtime_source = "stub"
        action_limits = None
        time_s = 0.

        def evaluate(self, profile=None, **kwargs):
            calls.append((profile, kwargs))
            return {"status": f"stub_{profile}"}

    env = Env()
    env.task = task
    env.backend = SimpleNamespace(proprioception=lambda: {"timestamp_s": env.time_s,
                                                          "tool_pose": task.reference_edge_pose_world(0.)})
    return env, calls


def test_scripted_policy_selection_and_measured_edge_feedback():
    module = _script()
    namespace = Namespace(motion="push", speed=.01, angle=0., edge_feedback="nominal", position_tolerance=None)
    env, _ = _stub_env("SLICE-PUSH-01")
    policy = module.scripted_policy(env, namespace)
    labels = [waypoint.label for waypoint in policy.waypoints]
    assert labels[:3] == ["approach_0", "start_0", "cut_0"] and "push" in labels
    assert policy.edge_estimator is None and policy.position_tolerance_m == .0005
    assert module.policy_plan_name(env.task) == "slice_push"
    measured = module.scripted_policy(env, Namespace(**{**vars(namespace), "edge_feedback": "measured",
                                                        "position_tolerance": .0003}))
    assert measured.position_tolerance_m == .0003
    np.testing.assert_allclose(measured.edge_estimator({"timestamp_s": 0.}), env.task.reference_edge_pose_world(0.))
    with pytest.raises(ValueError, match="observation time"):
        measured.edge_estimator({"timestamp_s": .02})
    assert module.edge_feedback_receipt(Namespace(edge_feedback="measured"))["source"] == "measured_tool_edge_feedback"
    cut, _ = _stub_env("CUT-01-R")
    assert module.policy_plan_name(cut.task) == "through_cut"
    assert module.scripted_policy(cut, namespace).waypoints[2].cut


def test_evaluate_episode_routes_demo_tasks_and_keeps_old_tasks():
    module = _script()
    env, calls = _stub_env("SLICE-PUSH-01")
    strict, research, demo = module.evaluate_episode(env, {"interval_s": [12.3, 14.1]})
    assert strict["status"] == "NOT_APPLICABLE" and strict["success"] is False
    assert calls[0][0] == "research" and calls[0][1]["goal"] == env.task.multi_cut_goal()
    assert calls[1] == (None, {"push_interval_s": [12.3, 14.1]}) and demo["status"] == "stub_None"
    old, old_calls = _stub_env("CUT-01-R")
    _, _, none = module.evaluate_episode(old)
    assert none is None and old_calls == [(None, {}), ("research", {})]


def test_recorder_push_interval_and_phase_arrays():
    module = _script()
    recorder = object.__new__(module.EpisodeRecorder)
    recorder.env = SimpleNamespace(time_s=1.0)
    recorder.phases = []
    assert recorder.push_interval() is None and recorder.phase_arrays() == {}
    recorder.phases = [(.0, "cut_1"), (.02, "push"), (.04, "push"), (.06, "push_hold"), (.08, "push")]
    push = recorder.push_interval()
    assert push["interval_s"] == [.02, .06] and push["completed"]
    arrays = recorder.phase_arrays()
    assert list(arrays["policy_phases"]) == ["cut_1", "push", "push", "push_hold", "push"]
    np.testing.assert_allclose(arrays["push_interval_s"], [.02, .06])
    recorder.phases = [(.0, "raise_in_kerf"), (.02, "push")]
    assert recorder.push_interval() == {**recorder.push_interval(), "interval_s": [.02, 1.0], "completed": False}


# ---------------------------------------------------------------- Phase 2B completion
def _config_from(captured):
    """Build the Native config exactly as live_env.open_cutting_env does, from the parsed options."""
    from robotic_cutting_sim.live_env import WRAPPER_OPTIONS
    from robotic_cutting_sim.tasks import ACTION_MODES
    options = dict(captured["options"])
    for key in ("runtime", "output_dir", "seed", "observation_profile", *WRAPPER_OPTIONS):
        options.pop(key, None)
    mode = "joint" if options.pop("action_mode", ACTION_MODES[0]) == ACTION_MODES[1] else "cartesian"
    return IsaacSimCuttingConfig(task_id=captured["task"], seed=42, action_mode=mode,
                                 **{"camera_mode": "native_single_view", "camera_layout": "wrist_d405_global_d455", **options})


SMOKE_COMMON = ("--scenario", SCENARIO, "--preset", "policy", "--tool-attachment-mode", "physical_grasp",
                "--controller", "scripted", "--cutting-law", "5", "--record-cameras", "--overview-camera",
                "--record-every", "5")


@pytest.mark.parametrize("task_id,extra", [
    ("SLICE-PUSH-01", ("--max-cuts", "2", "--steps", "800", "--edge-feedback", "measured")),
    ("SLICE-PUSH-01", ("--max-cuts", "2", "--steps", "60", "--edge-feedback", "measured")),
    ("CUT-01-R", ("--steps", "60")),
    ("TRIM-01-R", ("--steps", "60")),
])
def test_native_smoke_commands_build_a_valid_native_config(task_id, extra):
    captured = _parse(_script(), _args("--task", task_id, *SMOKE_COMMON, *extra))
    assert captured["task"] == task_id and captured["status"] == 1
    config = _config_from(captured)
    assert config.overview_camera and config.camera and config.scenario == SCENARIO and config.cutting_law == 5
    assert config.tool_attachment_mode == "physical_grasp" and config.max_cuts == (2 if task_id == "SLICE-PUSH-01" else 1)


def test_oblique_angle_and_offset_pass_through_to_through_cut():
    from robotic_cutting_sim.policies import through_cut
    module = _script()
    env, _ = _stub_env("CUT-01-R")
    base = dict(speed=.01, edge_feedback="nominal", position_tolerance=None)
    oblique = module.scripted_policy(env, Namespace(motion="oblique", angle=20., offset=0., **base))
    assert [w.pose for w in oblique.waypoints] == [w.pose for w in through_cut(env.task, tilt_deg=20.)]
    shifted = module.scripted_policy(env, Namespace(motion="push", angle=0., offset=.004, **base))
    assert [w.pose for w in shifted.waypoints] == [w.pose for w in through_cut(env.task, offset_m=.004)]
    default = module.scripted_policy(env, Namespace(motion="push", angle=0., offset=0., **base))
    assert [w.pose for w in default.waypoints] == [w.pose for w in through_cut(env.task)]
    captured = _parse(_script(), _args("--task", "CUT-01-R", "--tool-attachment-mode", "physical_grasp",
                                       "--controller", "scripted", "--motion", "oblique", "--angle", "20",
                                       "--offset", "0.004"))
    assert captured["status"] == 1


@pytest.mark.parametrize("argv", [
    _args("--task", "CUT-01-R", "--offset", "0.004"),                                       # not scripted
    _args("--task", "TRIM-01-R", "--controller", "scripted", "--offset", "0.004"),          # skim plan
    _args("--task", "SLICE-PUSH-01", "--controller", "scripted", "--max-cuts", "2", "--offset", "0.004"),
    _args("--task", "SLICE-PUSH-01", "--controller", "scripted", "--max-cuts", "2", "--motion", "saw"),
    _args("--task", "CUT-01-R", "--controller", "scripted", "--offset", "0.2"),
    _args("--task", "NOPE-01"),
])
def test_run_wp10_live_rejects_unsupported_motion_variants(argv):
    with pytest.raises(SystemExit):
        _parse(_script(), argv)


class _Array:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


def _live_env(task_id, times, codes):
    """A CuttingTaskEnv around a stub backend whose history holds the given codes (no Isaac)."""
    from robotic_cutting_sim.live_env import CuttingTaskEnv
    task = configure_task(make_task(task_id, particle_spacing_m=.005), scenario=SCENARIO)
    points, masses, lattice = task.sample_particles()
    world = task.points_to_world(points)
    cell = np.floor((points[:, 0]+.1)/.008).astype(int)
    nodes = np.zeros((len(points), 8), np.int32)
    nodes[:, 0], nodes[:, 1] = cell, cell+1
    weights = np.zeros((len(points), 8))
    weights[:, :2] = .5
    material = SimpleNamespace(masses=masses, world_points=world, lattice=lattice, max_cuts=2,
                               bed=SimpleNamespace(nodes=_Array(nodes), weights=_Array(weights),
                                                   gradients=_Array(np.zeros((*nodes.shape, 3)))))
    backend = SimpleNamespace(material=material, task=task, time_s=float(times[-1]), runtime_source="stub_backend",
                              surface_arrays=(np.zeros((0, 3)), np.zeros((0, 3), np.int32)),
                              surface_info={"component_ranges": [], "partition_rule": "stub"})
    env = object.__new__(CuttingTaskEnv)
    env.backend, env.task, env._closed, env._ready = backend, task, False, True
    env._history = [dict(time_s=float(t), positions=world.copy(), velocities=np.zeros_like(world), codes=c)
                    for t, c in zip(times, codes)]
    return env, points


def test_live_env_routes_demo_tasks_to_the_slice_push_report():
    times = np.round(np.arange(0., 16.01, .5), 10)
    env, points = _live_env("SLICE-PUSH-01", times, [np.zeros(0, np.int32)]*len(times))
    task = env.task
    d0, d1 = np.where(points[:, 0] < -.03, 1, 2), np.where(points[:, 0] < .03, 1, 2)
    for row, t in zip(env._history, times):
        row["codes"] = ((t >= task.cut_intervals_s[0][1])*d0+(t >= task.cut_intervals_s[1][1])*3*d1).astype(np.int32)
    report = env.evaluate()
    assert report["protocol"] == "slice_push_report_v1" and report["runtime_source"] == "stub_backend"
    assert report["multi_cut"]["status"] == "PASS_MULTI_CUT_OUTCOME"
    assert report["push_interval_source"] == "task_reference_schedule"
    # No push in the synthetic history: the pushed piece does not move -> an honest FAIL, never a PASS.
    assert report["status"] == "FAIL_SLICE_PUSH"
    recorded = env.evaluate(push_interval_s=(12., 15.))
    assert recorded["push_interval_source"] == "recorded_push_phase"
    research = env.evaluate(profile="research", goal=task.multi_cut_goal())
    assert research["status"] == "PASS_MULTI_CUT_OUTCOME" and "post_cut_manipulation" not in research


def test_short_slice_push_smoke_history_scores_not_pass_without_crashing():
    times = np.round(np.arange(.02, 1.21, .02), 10)  # 60 control steps, no cut yet
    env, points = _live_env("SLICE-PUSH-01", times, [np.zeros(0, np.int32)]*len(times))
    for row in env._history:
        row["codes"] = np.zeros(len(points), np.int32)
    report = env.evaluate()
    assert report["status"] != "PASS_SLICE_PUSH"
    assert report["post_cut_manipulation"]["status"] == "NOT_RUN"
