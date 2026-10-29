"""Presentation-only close-up camera: spec framing, recorder naming and renderer layout (CPU, no Isaac)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim import demo_media as dm
from robotic_cutting_sim import presentation_camera as pc
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "blademaster_workcell_v1"


@pytest.mark.parametrize("task_id", ("CUT-01-R", "SLICE-PUSH-01", "TRIM-01-R"))
def test_closeup_spec_is_presentation_only_and_frames_the_bar_and_edge(task_id):
    task = configure_task(make_task(task_id), scenario=SCENARIO)
    spec = pc.closeup_camera_spec(task)
    overview = pc.overview_camera_spec(task)
    assert spec["observation"] is False and spec["camera_contract"] is False and spec["modalities"] == ["rgb"]
    assert spec["role"] == "presentation_only" and spec["camera_id"] == "closeup_rgb"
    assert spec["prim_path"] != overview["prim_path"] and spec["camera_id"] != overview["camera_id"]
    assert 22 <= spec["hfov_deg"] <= 26 and .45 <= spec["working_distance_m"] <= .6
    assert (spec["width"], spec["height"]) == (960, 540)
    # Raised scenario aware: aimed at the same workpiece top centre as the overview.
    np.testing.assert_allclose(spec["look_at_target_world_m"], overview["look_at_target_world_m"], atol=1e-12)
    np.testing.assert_allclose(spec["look_at_target_world_m"], [0., 0., .07], atol=1e-12)
    # Same front-right direction family as the overview.
    view = np.subtract(spec["eye_world_m"], spec["look_at_target_world_m"])
    over = np.subtract(overview["eye_world_m"], overview["look_at_target_world_m"])
    assert np.dot(view, over)/np.linalg.norm(view)/np.linalg.norm(over) == pytest.approx(1.)
    assert view[0] > 0 and view[1] < 0 and view[2] > 0
    for name in ("workpiece_corners", "reference_edge_t0", "reference_edge_ends_t0"):
        assert spec["framing_check"][name]["all_inside"], name
    assert "franka_0" not in spec["framing_check"]
    # The 160 mm bar fills most of the frame width; ~0.25 mm per pixel at the target.
    corners = np.asarray(spec["framing_check"]["workpiece_corners"]["pixels_uv"])
    assert np.ptp(corners[:, 0]) > .6*spec["width"]
    assert spec["mm_per_pixel_at_target"] < .3
    K = np.asarray(spec["K"])
    assert K[0, 0] == pytest.approx(K[1, 1]) and K[0, 2] == spec["width"]/2
    json.dumps(spec)


def test_closeup_follows_an_unraised_workpiece():
    task = make_task("CUT-01")
    spec = pc.closeup_camera_spec(task)
    np.testing.assert_allclose(spec["look_at_target_world_m"],
                               task.T_world_workpiece[:3, 3]+[0., 0., task.geometry.base_height_m], atol=1e-12)
    assert all(check["all_inside"] for check in spec["framing_check"].values())


def test_closeup_authors_at_its_own_prim_path():
    from pxr import Usd, UsdGeom
    from robotic_cutting_sim.usd_surface_bridge import usd_camera_calibration
    task = configure_task(make_task("CUT-01-R"), scenario=SCENARIO)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    spec = pc.closeup_camera_spec(task)
    pc.author_overview_camera(stage, spec)
    pc.author_overview_camera(stage, pc.overview_camera_spec(task))
    assert stage.GetPrimAtPath(pc.CLOSEUP_PRIM_PATH).IsValid() and stage.GetPrimAtPath(pc.PRIM_PATH).IsValid()
    authored = usd_camera_calibration(stage, pc.CLOSEUP_PRIM_PATH, spec["width"], spec["height"])
    np.testing.assert_allclose(authored["T_world_camera"], spec["T_world_camera"], atol=1e-6)
    np.testing.assert_allclose(authored["K"], spec["K"], rtol=1e-5)


def _recorder(tmp_path, backend):
    spec = importlib.util.spec_from_file_location("run_wp10_live_closeup", ROOT/"scripts/run_wp10_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = object.__new__(module.EpisodeRecorder)
    (tmp_path/"observations").mkdir()
    frame = {"rgb": np.zeros((4, 6, 3), np.uint8), "physics_timestamp_s": 1.0}
    recorder.env = SimpleNamespace(time_s=1.0, backend=backend, _frames={"global_d455": frame})
    recorder.output, recorder.controller = tmp_path, "scripted"
    recorder.rows, recorder.samples, recorder.states = [], [], []
    return recorder


def _backend(**frames):
    state = {key: np.zeros((2, 3)) for key in ("positions", "velocities", "codes")}
    return SimpleNamespace(calibration={}, material=SimpleNamespace(history_state=lambda: state),
                           measured=lambda: {"q": np.zeros(2)}, **frames)


def test_recorder_saves_closeup_frames_separately(tmp_path):
    rgb = np.full((5, 7, 3), 9, np.uint8)
    backend = _backend(overview_frame={"rgb": rgb, "physics_timestamp_s": 1.0},
                       closeup_frame={"rgb": rgb+1, "physics_timestamp_s": 1.0, "camera_id": "closeup_rgb"})
    recorder = _recorder(tmp_path, backend)
    recorder.capture()
    names = sorted(p.name for p in (tmp_path/"observations").iterdir())
    assert names == ["closeup_rgb_0000.npz", "global_d455_0000.npz", "overview_rgb_0000.npz"]
    with np.load(tmp_path/"observations"/"closeup_rgb_0000.npz") as data:
        assert sorted(data.files) == ["physics_timestamp_s", "rgb"]
        assert (data["rgb"] == 10).all() and float(data["physics_timestamp_s"]) == 1.0
    # A stale close-up frame is refused, like the overview.
    backend.closeup_frame = {"rgb": rgb, "physics_timestamp_s": .5}
    with pytest.raises(ValueError, match="close-up"):
        recorder.capture()


def test_recorder_without_closeup_writes_no_closeup_file(tmp_path):
    recorder = _recorder(tmp_path, _backend(overview_frame=None))
    recorder.capture()
    assert sorted(p.name for p in (tmp_path/"observations").iterdir()) == ["global_d455_0000.npz"]


def test_renderer_puts_the_closeup_in_the_secondary_slot_only_when_recorded():
    base = ["global_d455_0000.npz", "wrist_d405_0_0000.npz", "wrist_d405_1_0000.npz", "overview_rgb_0000.npz"]
    without = dm.choose_panels(dm.discover_streams(base))
    assert without == dict(large="overview_rgb", secondary="global_d455", wrist_rgb="wrist_d405_0",
                           wrist_depth="wrist_d405_0", overview="overview_rgb")
    assert dm.panel_layout(without) == dm.LAYOUT_DEFAULT
    with_closeup = dm.choose_panels(dm.discover_streams(base + ["closeup_rgb_0000.npz", "closeup_rgb_0001.npz"]))
    assert with_closeup["secondary"] == "closeup_rgb" and with_closeup["closeup"] == "closeup_rgb"
    assert with_closeup["large"] == "overview_rgb" and with_closeup["wrist_rgb"] == "wrist_d405_0"
    assert dm.panel_layout(with_closeup) == dm.LAYOUT_CLOSEUP
    assert dm.panel_title("closeup_rgb", {}) == "Close-up (presentation camera, not an observation)"
    assert dm.panel_title("wrist_d405_1", {"wrist_d405_1": "Wrist D405_1 RGB"}) == "Wrist D405_1 RGB"
    assert dm.panel_title(None, {}) == "secondary RGB"
    # No overview: close-up still takes the secondary slot; Global D455 stays the large panel.
    plain = dm.choose_panels(dm.discover_streams(["global_d455_0000.npz", "wrist_d405_1_0000.npz",
                                                  "closeup_rgb_0000.npz"]))
    assert plain["large"] == "global_d455" and plain["secondary"] == "closeup_rgb"
