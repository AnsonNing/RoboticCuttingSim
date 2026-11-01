"""Explicit scoped solver selection; no defaults, physics edits or fallback."""
from copy import deepcopy
import hashlib
import json
import math
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
    """Compile the profile controller, recording enough time to finish its plan.

    A sufficient requested duration is retained. A shorter goal receives the
    same completion margin used by the frozen paired driver (10% plus 1 s,
    rounded up to 0.5 s). The actual duration is explicit in the compiled goal.
    """
    from .scene_config import compile_task
    from .goals import validate_goal
    scene=select_profile(scene,name);goal=validate_goal(goal)
    requested=goal['episode']['duration_s']
    probe_goal=deepcopy(goal);probe_goal['episode']['duration_s']=120.
    options=document(name)['planner']
    probe=compile_task(scene,probe_goal,planner_options=options,seed=seed)
    done=next(float(t) for t,label in zip(probe._times(),probe.reference_phase_labels) if label=='DONE')
    goal['episode']['duration_s']=max(requested,.5*math.ceil((1.1*done+1.)/.5))
    return compile_task(scene,goal,planner_options=options,seed=seed)


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
    for filename,expected in doc['configuration_source_sha256'].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != expected:
            errors.append(f'validated material/workcell source differs: {filename}')
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
    length_units=wp['base_box_m']['length_m']/env['length_increment_m']
    if abs(length_units-round(length_units))>1.e-10:
        errors.append('bar length reference-lattice phase outside profile')
    if wp['material_profile'] not in env['material_profiles']:
        errors.append('material outside profile')
    pose = wp['world_pose']
    if not np.allclose(pose['translation_m'], [0., 0., .01], atol=1.e-12, rtol=0) or not env['yaw_deg'][0] <= pose['yaw_deg'] <= env['yaw_deg'][1]:
        errors.append('workpiece placement outside profile')
    support = scene['supports']; workcell = support.get('workcell') or {}
    if support['kind'] != 'flat_board' or workcell.get('profile') != 'blade_master_flat_board_v1':
        errors.append('profile requires the declared solid flat board')
    overrides=workcell.get('overrides',{})
    board=overrides.get('cutting_board',{})
    if (set(overrides)-{'cutting_board'} or set(board)-{'yaw_deg','center_xy_world_m'}
            or board.get('yaw_deg',0.) != pose['yaw_deg']
            or not any(np.allclose(board.get('center_xy_world_m',[0.,.03]),p,atol=1.e-12,rtol=0)
                       for p in ([0.,.03],[0.,.055]))):
        errors.append('workcell/board overrides outside profile')
    if scene['tool']['profile_id'] != env['tool']:
        errors.append('tool outside profile')
    if goal['type'] != 'sever' or len(goal.get('planes', [])) != 1:
        errors.append('only one straight sever is supported by this profile')
    else:
        plane = goal['planes'][0]; normal = plane['normal_local']
        normals=[{'+x': [1.,0.,0.], '-x': [-1.,0.,0.]}[sign] for sign in env['normal_local']]
        if not any(np.allclose(normal,n,atol=1.e-12,rtol=0) for n in normals):
            errors.append('cut/blade direction outside profile')
        if plane.get('approach_local') is not None:
            errors.append('custom approach outside profile')
        if not env['cut_x_m'][0] <= plane['origin_local_m'][0] <= env['cut_x_m'][1]:
            errors.append('cut location outside profile')
        if not any(abs(plane['origin_local_m'][0]-value)<1.e-12 for value in env['cut_x_values_m']):
            errors.append('cut reference-lattice phase outside profile')
        if (goal['expected_pieces'] != 2 or goal['minimum_part_mass_fraction'] < .4 or
                goal['minimum_main_parts_mass_fraction'] < .99 or goal['plane_error_bound_m'] > .002):
            errors.append('goal evaluator requirements weaker than validated sever')
    if planner != doc['planner'] or goal['episode']['control_dt_s'] != .02:
        errors.append('effective controller outside profile')
    if errors:
        raise ValueError(f'{name}: '+ '; '.join(errors))
    return {'id':name, 'solver':doc['solver'], 'front':doc['front'], 'contact':doc['contact'],
            'position_update':doc['position_update'], 'envelope':env, 'evaluator':doc['evaluator'],
            'scope_decision':doc['scope_decision'],
            'source_sha256':doc['source_sha256'], 'configuration_source_sha256':doc['configuration_source_sha256'],
            'configuration_sha256':hashlib.sha256(PROFILES[name].read_bytes()).hexdigest()}


def validate_runtime(task, solver, law, dt, cuts, slots, slice_efficiency):
    if not getattr(task, 'scene_json', '') or not task.scene['solver'].get('research_profile'):
        return None
    receipt = validate_task(task.scene, task.goal, task.controller['planner'], task.particle_spacing_m)
    doc = document(receipt['id'])
    from .scene_config import scene_material
    if task.material != scene_material(task.scene):
        raise ValueError(f"{receipt['id']}: effective material differs from the validated scene recipe")
    if (solver != doc['solver'] or law != doc['cutting_law'] or dt != doc['envelope']['material_dt_s']
            or cuts != doc['max_cuts'] or slots != doc['slots_per_node'] or slice_efficiency != 1.):
        raise ValueError(f"{receipt['id']}: runtime solver/contact/discretization overrides outside profile")
    return receipt
