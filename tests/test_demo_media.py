import numpy as np
import pytest

from robotic_cutting_sim import demo_media as dm


def bar(n=40):
    """Reference points along x: left half, right half."""
    return np.c_[np.linspace(-.08, .08, n), np.zeros(n), np.zeros(n)]


def test_component_colours_follow_reference_position_not_size():
    reference = bar(40)
    labels = np.r_[np.ones(15, int), np.zeros(25, int)]  # component 1 is on the left
    colors, slot, sizes = dm.component_colors(labels, reference)
    assert sizes.tolist() == [25, 15]
    assert slot.tolist() == [1, 0]  # the leftmost piece keeps slot 0
    assert np.allclose(colors[0], dm.hex_rgba(dm.COMPONENT_COLORS[0]))
    assert np.allclose(colors[-1], dm.hex_rgba(dm.COMPONENT_COLORS[1]))
    # Relabelling (e.g. union-find root order) does not repaint the pieces.
    colors2, _, _ = dm.component_colors(1 - labels, reference)
    assert np.allclose(colors, colors2)


def test_minor_fragments_are_grey_and_extra_components_fold():
    reference = bar(200)
    labels = np.zeros(200, int)
    labels[100:] = 1
    labels[0] = 2            # 0.5 % fragment -> minor
    labels[150:170] = 3      # four major pieces, three hue slots
    labels[180:199] = 4
    colors, slot, sizes = dm.component_colors(labels, reference)
    assert slot[2] == -1
    assert np.allclose(colors[0], dm.hex_rgba(dm.MINOR_COLOR))
    assert sorted(s for s in slot if s >= 0) == [0, 1, 2]
    assert (slot == -1).sum() == 2


def test_component_cache_and_cut_count_on_synthetic_stencil():
    # Four particles sharing one node; codes 1 and 2 are incompatible for cut 0.
    nodes = np.array([[0, 1], [0, 1], [1, 2], [2, 3]])
    weights = np.ones((4, 2), np.float32)
    codes = np.array([0, 0, 0, 0], np.int32)
    assert dm.cut_count(codes) == 1 and dm.cut_count(np.array([0, 4])) == 2
    cache = dm.ComponentCache(nodes, weights, None, cuts=1)
    assert cache(codes)[0] == 1
    split = np.array([1, 1, 2, 2], np.int32)
    nodes_split = np.array([[0, 1], [0, 1], [2, 3], [2, 3]])
    cache2 = dm.ComponentCache(nodes_split, weights, None, cuts=1)
    assert cache2(split)[0] == 2
    cache(codes.copy()), cache(codes)
    assert cache.calls == 1 and len(cache.cache) == 1


def test_frame_annotation_text_and_status_line():
    status, source = dm.status_line(None, {"task_status": "PASS_TASK_METRICS", "status": "PASS_EXECUTION"})
    assert source == "result.json" and "PASS_TASK_METRICS" in status and "PASS_EXECUTION" in status
    assert dm.status_line("FAIL_TASK_METRICS (research)", {"task_status": "PASS"}) == \
        ("FAIL_TASK_METRICS (research)", "--label")
    assert dm.status_line(None, {})[1] == "missing"
    lines = dm.frame_annotation(runtime=dm.runtime_display_name("live_isaac_sim"), controller="reference",
                                time_s=3.2, active_cut=True, cut_index=1, status=status, task_id="CUT-02")
    assert lines[0] == "CUT-02 | Live Isaac Sim Native | controller: reference"
    assert "3.20 s" in lines[1] and "CUT ON (cut_index 1)" in lines[1] and status in lines[1]
    off = dm.frame_annotation(runtime="x", controller="c", time_s=0., active_cut=False, cut_index=-1, status="s")
    assert "CUT OFF (cut_index -1)" in off[1]
    assert dm.runtime_display_name("something_new") == "something_new"


def test_edge_segment_force_and_rows():
    # Identity rotation: the edge runs along local/world +y.
    seg = dm.edge_segment([0, 0, .03, 0, 0, 0, 1], .06)
    assert np.allclose(seg, [[0, -.06, .03], [0, .06, .03]])
    # CUT-01-R orientation: 90 deg about y keeps the edge along world y.
    seg = dm.edge_segment([0, 0, 0, 0, np.sqrt(.5), 0, np.sqrt(.5)], .06)
    assert np.allclose(seg[:, 1], [-.06, .06]) and np.allclose(seg[:, [0, 2]], 0)
    assert dm.reaction_force_n([.003, .004, 0, 9, 9, 9], .02) == pytest.approx(.25)
    assert dm.edge_half_length({"tool_profile": {"half_width_m": .06}}) == .06
    assert dm.edge_half_length({"blade_half_width_m": .025}) == .025
    assert dm.edge_half_length({}) is None
    assert dm.nearest_rows([.02, .04, .06], [0., .04, .2]).tolist() == [-1, 1, 2]


def test_stream_discovery_and_panels():
    names = ["global_d455_0000.npz", "global_d455_0001.npz", "wrist_d405_0_0000.npz",
             "wrist_d405_1_0000.npz", "notes.txt"]
    streams = dm.discover_streams(names)
    assert streams == {"global_d455": [0, 1], "wrist_d405_0": [0], "wrist_d405_1": [0]}
    panels = dm.choose_panels(streams)
    assert panels["large"] == "global_d455" and panels["secondary"] == "wrist_d405_1"
    assert panels["overview"] is None
    panels = dm.choose_panels(dm.discover_streams(names + ["overview_rgb_0000.npz"]))
    assert panels["large"] == "overview_rgb" and panels["secondary"] == "global_d455"
    assert dm.choose_panels({})["large"] is None


def test_measured_rate_and_receipt_fields():
    rate = dm.measured_rate([0., .2, .4, .6, .8, float("nan")])
    assert rate["rate_hz"] == pytest.approx(5.) and rate["frames"] == 5 and rate["monotonic"]
    assert dm.measured_rate([1.])["rate_hz"] is None
    plans = dm.gif_plans(41, 10., 480)
    assert plans[0] == dict(stride=1, fps=10., width=480)
    assert all(p["width"] % 2 == 0 for p in plans) and plans[-1]["width"] < 480
    receipt = dm.render_receipt(
        episode="ep", out="o", inputs={"result.json": {"sha256": "a"}}, frames=41, rate=rate, video_fps=5,
        gif={"bytes": 1}, panels={"overview_present": False}, components=[1, 2], cuts=1, particle_stride=3,
        status="s", status_source="result.json", git={"sha": "x"}, wall_s=1.5, command=["r"],
        outputs={}, label_source_runtime={}, versions={})
    assert receipt["frame_count"] == 41 and receipt["measured_capture_rate"]["rate_hz"] == pytest.approx(5.)
    assert receipt["particle_view"]["label"] == dm.ORACLE_LABEL
    assert receipt["particle_view"]["component_count_per_frame"] == [1, 2]
    assert receipt["recording_modified"] is False and "control_dt" in receipt["force_definition"]
    for key in ("inputs", "git", "wall_time_s", "panels", "outputs", "status_line"):
        assert key in receipt
