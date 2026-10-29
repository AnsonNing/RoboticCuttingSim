"""cutting_tasks_v2: ruler-knife variants of the v1 tasks and CUT-02 (CPU)."""
from dataclasses import asdict
import hashlib

import numpy as np
import pytest

from robotic_cutting_sim.scene_frames import pose_matrix, transform_points
from robotic_cutting_sim.tasks import (PROTOCOL_PATH, PROTOCOL_V2_PATH, CuttingTaskSpec, ProfileTaskSpec,
                                       _protocol, make_task)
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1, make_profile_tool_spec


@pytest.mark.parametrize("task_id,base", [("CUT-01-R", "CUT-01"), ("TRIM-01-R", "TRIM-01"), ("CUT-02", "CUT-01")])
def test_v2_tasks_use_the_profile_and_inherit_thresholds(task_id, base):
    task = make_task(task_id)
    assert isinstance(task, ProfileTaskSpec) and task.tool_profile is RULER_KNIFE_V1
    assert task.base_task_id == base and task.goal_kind == make_task(base).goal_kind
    assert np.allclose(task.hand_from_edge_xyz_xyzw, RULER_KNIFE_V1.hand_from_edge())
    v1_limits, v2_limits = _protocol()[base], task.protocol_document()[task_id]
    threshold_keys = [k for k in v1_limits if k.startswith(("minimum", "maximum", "required", "surface"))]
    assert threshold_keys and all(v2_limits[k] == v1_limits[k] for k in threshold_keys)
    receipt = task.to_dict()
    assert receipt["protocol_sha256"] == hashlib.sha256(PROTOCOL_V2_PATH.read_bytes()).hexdigest()
    assert receipt["inherited_protocol_sha256"] == hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    assert receipt["tool_profile"]["profile_id"] == "ruler_knife_v1"
    with pytest.raises(NotImplementedError):
        make_task(task_id, variant="randomized")


def test_v1_tasks_keep_their_exact_receipt_fields():
    for task_id in ("CUT-01", "TRIM-01"):
        task = make_task(task_id)
        assert type(task) is CuttingTaskSpec
        assert "tool_profile_id" not in asdict(task) and "tool_profile" not in task.to_dict()
        assert task.to_dict()["protocol_sha256"] == hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def test_cut02_workpiece_and_path():
    task = make_task("CUT-02", particle_spacing_m=.005)
    points, masses, _ = task.sample_particles()
    extent = points.max(0)-points.min(0)
    assert np.allclose(extent+.005, [.16, .06, .025], atol=1e-6)
    knots = np.asarray(task.edge_position_knots_local_txyz)
    start, stop = task.active_cut_interval_s
    assert np.interp(start, knots[:, 0], knots[:, 3]) > task.geometry.base_height_m
    assert np.interp(stop, knots[:, 0], knots[:, 3]) < 0
    assert 2*task.blade_half_width_m > task.geometry.width_m
    pedestals = [box for box in task.fixture_boxes_local]
    assert all(box.half_extents_m[1] > task.geometry.width_m/2 for box in pedestals)


def _tool_points_world(task, time_s):
    """Blade and handle collider vertices at the scripted reference pose."""
    spec = make_profile_tool_spec(task.tool_profile)
    T_world_edge = pose_matrix(task.reference_edge_pose_world(time_s))
    T_world_tool = T_world_edge@np.linalg.inv(spec.T_tool_edge)
    return {shape.role: transform_points(T_world_tool, shape.vertices_tool_m) for shape in spec.shapes}


def test_trim_rake_keeps_the_edge_lowest_and_the_handle_clear():
    task = make_task("TRIM-01-R")
    plane = task.target_plane_world()
    start, stop = task.active_cut_interval_s
    for time_s in np.linspace(start, stop, 9):
        edge = task.reference_edge_pose_world(time_s)[:3]
        points = _tool_points_world(task, time_s)
        assert points["blade"][:, 2].min() >= edge[2]-1e-9         # nothing below the edge
        handle_height = points["handle"]@plane[:3]+plane[3]
        assert handle_height.min() > .006                            # handle clears the trimmed plane
        assert edge@plane[:3]+plane[3] == pytest.approx(task.edge_clearance_m, abs=1e-9)


def test_cut_vertical_ruler_spine_stays_above_the_bar_at_full_depth():
    for task_id in ("CUT-01-R", "CUT-02"):
        task = make_task(task_id)
        points = _tool_points_world(task, task.active_cut_interval_s[1])
        top = task.geometry.base_height_m
        assert points["handle"][:, 2].min() > top                    # only the blade enters the bar


def test_runtime_tool_dispatch_keeps_v1_and_uses_profiles_for_v2():
    from robotic_cutting_sim.physical_tool import make_tool_spec
    from robotic_cutting_sim.tool_profile import require_task_tool, tool_spec_for_task
    for task_id in ("CUT-01", "TRIM-01"):
        task = make_task(task_id)
        assert tool_spec_for_task(task).to_dict() == make_tool_spec(task).to_dict()
        require_task_tool(tool_spec_for_task(task), task)
    for task_id in ("CUT-01-R", "TRIM-01-R", "CUT-02"):
        task = make_task(task_id)
        spec = tool_spec_for_task(task)
        assert spec.tool_id == "ruler_knife_v1" and [s.role for s in spec.shapes] == ["handle", "blade"]
        require_task_tool(spec, task)
        with pytest.raises(NotImplementedError):
            require_task_tool(make_tool_spec(make_task("CUT-01")), task)
