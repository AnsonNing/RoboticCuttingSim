import numpy as np
import pytest
from robotic_cutting_sim.tasks import make_task
from robotic_cutting_sim.task_scenarios import apply_scenario
from robotic_cutting_sim.food_materials import apply_profile, receipt


def test_raised_workcell_preserves_task_material_and_goal_but_anchors_supports():
    original = make_task('TRIM-01-R')
    before = original.to_dict()
    raised = apply_scenario(original, 'raised_workpiece_50mm_v1')
    assert original.to_dict() == before
    assert raised.geometry == original.geometry and raised.material == original.material
    assert raised.goal_kind == original.goal_kind
    assert raised.protocol_document() == original.protocol_document()
    assert np.allclose(raised.reference_edge_pose_world(3.)[:3] - original.reference_edge_pose_world(3.)[:3], [0., 0., .05])
    for old, new in zip(original.fixture_boxes_local, raised.fixture_boxes_local):
        assert np.isclose(old.center_m[2] + old.half_extents_m[2], new.center_m[2] + new.half_extents_m[2])
        assert np.isclose(new.center_m[2] - new.half_extents_m[2] + .05, raised.table_top_world_z_m)


def test_legacy_task_cannot_silently_select_new_scene_or_food():
    task = make_task('CUT-01')
    with pytest.raises(ValueError):
        apply_scenario(task, 'raised_workpiece_50mm_v1')
    with pytest.raises(ValueError):
        apply_profile(task, 'cheddar_provisional_v1')
    assert receipt('cream_cheese_provisional_v1')['grid_cfl'] < .2
    assert receipt('cheddar_provisional_v1')['grid_cfl'] < .2
