"""Explicit scoped solver selection; no defaults, physics edits or fallback."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {'v7_cut_research_v1': ROOT/'configs/wp14/v7_cut_research_v1.json'}


def document(name):
    if name not in PROFILES:
        raise ValueError(f'Unknown research profile {name!r}; choices: {tuple(PROFILES)}')
    return json.loads(PROFILES[name].read_text())


def select_profile(scene, name):
    """Return an independent scene with the explicitly selected solver/profile."""
    from .scene_config import resolve_scene
    doc = document(name)
    scene = deepcopy(resolve_scene(scene))
    scene['solver'].update(module=doc['solver'], research_profile=name)
    return scene


def compile_research_task(name, scene, goal, *, seed=42):
    """Compile the named profile's declared controller and validated solver path."""
    from .scene_config import compile_task
    return compile_task(select_profile(scene, name), goal, planner_options=document(name)['planner'], seed=seed)


def validate_task(scene, goal, planner, spacing):
    """Refuse an out-of-envelope request before any physics or robot motion."""
    name = scene['solver'].get('research_profile')
    if name is None:
        return None
    doc = document(name); env = doc['envelope']; solver = scene['solver']; wp = scene['workpiece']
    errors = []
    for filename, expected in doc['source_sha256'].items():
        actual = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
        if actual != expected: errors.append(f'validated solver source differs: {filename}')
    if solver['module'] != doc['solver'] or solver['cutting_law'] != doc['cutting_law']:
        errors.append('solver/law differs from explicit profile')
    if solver['max_cuts'] != doc['max_cuts'] or solver.get('slots_per_node', 8) != doc['slots_per_node']:
        errors.append('retained history/slot capacity differs from profile')
    if spacing != env['particle_spacing_m'] or solver['grid_dx_m'] != env['grid_dx_m']:
        errors.append('discretization outside profile')
    if wp['protrusions'] or wp.get('geometry_provider') is not None:
        errors.append('profile requires a plain continuous rectangular bar')
    for key, limits in ((k, env[k]) for k in ('length_m', 'width_m', 'height_m')):
        if not limits[0]-1.e-12 <= wp['base_box_m'][key] <= limits[1]+1.e-12:
            errors.append(f'{key} outside profile')
    if wp['material_profile'] not in env['material_profiles']:
        errors.append('material outside profile')
    pose = wp['world_pose']
    if not np.allclose(pose['translation_m'], [0., 0., .01], atol=1.e-12, rtol=0) or not env['yaw_deg'][0] <= pose['yaw_deg'] <= env['yaw_deg'][1]:
        errors.append('workpiece placement outside profile')
    support = scene['supports']; workcell = support.get('workcell') or {}
    if support['kind'] != 'flat_board' or workcell.get('profile') != 'blade_master_flat_board_v1':
        errors.append('profile requires the declared solid flat board')
    if scene['tool']['profile_id'] != env['tool']:
        errors.append('tool outside profile')
    if goal['type'] != 'sever' or len(goal.get('planes', [])) != 1:
        errors.append('only one straight sever is supported by this profile')
    else:
        plane = goal['planes'][0]; normal = plane['normal_local']
        if not any(np.allclose(normal, n, atol=1.e-12, rtol=0) for n in ([1.,0.,0.],[-1.,0.,0.])):
            errors.append('oblique normal outside profile')
        if plane.get('approach_local') is not None:
            errors.append('custom approach outside profile')
        if not env['cut_x_m'][0] <= plane['origin_local_m'][0] <= env['cut_x_m'][1]:
            errors.append('cut location outside profile')
        if (goal['expected_pieces'] != 2 or goal['minimum_part_mass_fraction'] < .4 or
                goal['minimum_main_parts_mass_fraction'] < .99 or goal['plane_error_bound_m'] > .002):
            errors.append('goal evaluator requirements weaker than validated sever')
    if planner != doc['planner'] or goal['episode']['control_dt_s'] != .02:
        errors.append('effective controller outside profile')
    if errors:
        raise ValueError(f'{name}: '+ '; '.join(errors))
    return {'id':name, 'solver':doc['solver'], 'front':doc['front'], 'contact':doc['contact'],
            'position_update':doc['position_update'], 'envelope':env, 'evaluator':doc['evaluator'],
            'source_sha256':doc['source_sha256'], 'configuration_sha256':hashlib.sha256(PROFILES[name].read_bytes()).hexdigest()}


def validate_runtime(task, solver, law, dt, cuts, slots, slice_efficiency):
    if not getattr(task, 'scene_json', '') or not task.scene['solver'].get('research_profile'):
        return None
    receipt = validate_task(task.scene, task.goal, task.controller['planner'], task.particle_spacing_m)
    doc = document(receipt['id'])
    if (solver != doc['solver'] or law != doc['cutting_law'] or dt != doc['envelope']['material_dt_s']
            or cuts != doc['max_cuts'] or slots != doc['slots_per_node'] or slice_efficiency != 1.):
        raise ValueError(f"{receipt['id']}: runtime solver/contact/discretization overrides outside profile")
    return receipt
