"""Versioned workcell placements; never alter a frozen v1 task or its scoring."""
from dataclasses import replace
import json
import numpy as np
from .recording import ROOT
from .tasks import ProfileTaskSpec, FixtureBox

PATH = ROOT / 'configs/wp10_scenarios_v1.json'
PATH_V2 = ROOT / 'configs/wp10_scenarios_v2.json'


def scenarios():
    return {**json.loads(PATH.read_text())['scenarios'], **json.loads(PATH_V2.read_text())['scenarios']}


def configure_task(task, *, scenario=None, material_profile_id=None):
    if scenario is not None:
        task = apply_scenario(task, scenario)
    if material_profile_id is not None:
        from .food_materials import apply_profile
        task = apply_profile(task, material_profile_id)
    return task


def initial_joints(scenario, task_id):
    if scenario is None:
        return None
    return scenarios()[scenario].get('initial_joint_positions', {}).get(task_id)


def apply_scenario(task, name):
    config = scenarios()[name]
    if not isinstance(task, ProfileTaskSpec) or task.task_id not in config['allowed_tasks']:
        raise ValueError('This execution scenario is explicitly restricted to the declared v2 tasks')
    pose = np.array(task.world_from_workpiece_xyzw, float)
    if not np.allclose(pose[3:], [0., 0., 0., 1.], atol=1e-12):
        raise ValueError('Raised scenario currently requires the identity workpiece orientation')
    pose[:3] += config['world_translation_m']
    boxes = []
    for box in task.fixture_boxes_local:
        center, half = np.array(box.center_m, float), np.array(box.half_extents_m, float)
        if config['extend_fixture_bases_to_table']:
            top = center[2] + half[2]
            bottom = task.table_top_world_z_m - pose[2]
            half[2], center[2] = (top - bottom) / 2, (top + bottom) / 2
        boxes.append(FixtureBox(box.name, tuple(center), tuple(half)))
    options = dict(world_from_workpiece_xyzw=tuple(pose), fixture_boxes_local=tuple(boxes))
    if task.task_id == 'TRIM-01-R' and 'trim_rake_deg' in config:
        from scipy.spatial.transform import Rotation
        options['edge_orientation_local_xyzw'] = tuple(Rotation.from_euler('y', config['trim_rake_deg'], degrees=True).as_quat())
    return replace(task, **options)
