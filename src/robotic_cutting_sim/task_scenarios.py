"""Versioned workcell placements; never alter a frozen v1 task or its scoring."""
from dataclasses import replace
import json
import numpy as np
from .recording import ROOT
from .tasks import ProfileTaskSpec, FixtureBox, MaterialFixtureBox

PATH = ROOT / 'configs/wp10_scenarios_v1.json'
PATH_V2 = ROOT / 'configs/wp10_scenarios_v2.json'
PATH_V3 = ROOT / 'configs/wp10_scenarios_v3.json'


def scenarios():
    return {**json.loads(PATH.read_text())['scenarios'], **json.loads(PATH_V2.read_text())['scenarios'],
            **json.loads(PATH_V3.read_text())['scenarios']}


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
    mode = config.get('fixture_mode')
    if mode not in (None, 'cutting_board_v1'):
        raise ValueError(f'Unsupported fixture mode {mode}')
    boxes = _cutting_board_boxes(task, config, pose[2]) if mode else []
    for box in () if mode else task.fixture_boxes_local:
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


def _floats(*values):
    return tuple(float(value) for value in values)


def _resolve_friction(task, value):
    return float(task.material.friction) if value == 'task_material' else float(value)


def board_particle_clearance(task, config):
    """Analytic |y| reach of the material versus the widened board (no contact change)."""
    board = config['cutting_board']
    reach = task.geometry.width_m / 2
    for bump in task.geometry.bumps:
        reach = max(reach, abs(bump.y_m) + bump.radius_y_m)
    limit = board['half_width_y_m'] - board['particle_y_margin_m']
    return dict(material_half_width_y_m=float(reach), board_half_width_y_m=float(board['half_width_y_m']),
                margin_m=float(board['particle_y_margin_m']), clear=bool(reach < limit))


def _cutting_board_boxes(task, config, workpiece_world_z):
    """Each support -> wood board (same top, widened y) + grey riser down to the table."""
    if not config['extend_fixture_bases_to_table']:
        raise ValueError('cutting_board_v1 places risers on the table')
    board, riser = config['cutting_board'], config['riser']
    if riser['footprint'] != 'original_support':
        raise ValueError('Unsupported riser footprint')
    clearance = board_particle_clearance(task, config)
    if not clearance['clear']:
        raise ValueError(f'Widened board would reach the material: {clearance}')
    table = task.table_top_world_z_m - workpiece_world_z
    thickness = board['thickness_m']
    boxes = []
    for box in task.fixture_boxes_local:
        center, half = np.array(box.center_m, float), np.array(box.half_extents_m, float)
        top = center[2] + half[2]
        bottom = top - thickness
        if bottom <= table + 1e-9 or half[1] > board['half_width_y_m']:
            raise ValueError('Board must sit above the table and contain the original support footprint')
        common = dict(support_name=box.name)
        boxes.append(MaterialFixtureBox(f'{box.name}_board', _floats(center[0], center[1], top - thickness / 2),
                                        _floats(half[0], board['half_width_y_m'], thickness / 2), role='board',
                                        display_color_rgb=tuple(board['display_color_rgb']),
                                        static_friction=_resolve_friction(task, board['friction']),
                                        dynamic_friction=_resolve_friction(task, board['friction']),
                                        restitution=float(board['restitution']), **common))
        boxes.append(MaterialFixtureBox(f'{box.name}_riser', _floats(center[0], center[1], (bottom + table) / 2),
                                        _floats(half[0], half[1], (bottom - table) / 2), role='riser',
                                        display_color_rgb=tuple(riser['display_color_rgb']),
                                        static_friction=_resolve_friction(task, riser['friction']),
                                        dynamic_friction=_resolve_friction(task, riser['friction']),
                                        restitution=float(riser['restitution']), **common))
    return boxes


def fixture_materials(task):
    """Per-fixture-box role/colour/contact material, aligned with task.fixture_boxes_local.

    Boxes from scenarios without declared materials (v1/v2) report
    material_declared False; runtimes then keep their current default.
    """
    rows = []
    for box in task.fixture_boxes_local:
        if isinstance(box, MaterialFixtureBox):
            rows.append({**box.material_dict(), 'material_declared': True})
        else:
            rows.append(dict(name=box.name, role='support', support_name=box.name, material_declared=False))
    return tuple(rows)


def table_material(task, scenario):
    """Declared table contact material for a scenario, or None when none is declared."""
    if scenario is None or 'table' not in scenarios()[scenario]:
        return None
    table = scenarios()[scenario]['table']
    friction = _resolve_friction(task, table['friction'])
    return dict(static_friction=friction, dynamic_friction=friction, restitution=float(table['restitution']))
