"""Versioned scene compilation for the agentic control robot TRIM challenge.

Generator truth and mechanical evaluation belong to the harness, never to a
controller observation. Existing solvers, goals and thresholds are unchanged.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry_providers import lattice_report, protrusion_half_extents
from .scene_config import (SceneInfeasible, canonical_json, compile_task,
                           configured_task_payload, resolve_scene, sha256_json)

VERSION = 'rcs_trim_random_sim_v1'
FAMILIES = ('dome', 'gaussian', 'irregular', 'ridge')
GOAL_TEXT = 'Remove all protrusions from the continuous material while preserving the substrate.'


def digest_array(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def generate_scene(scene_seed, *, distribution='id'):
    """One deterministic PCG64 draw, with no retries or outcome-based seed changes.

    OOD changes yaw and asymmetry only. Material, board and workpiece pose stay
    fixed. Conservative rectangular footprint bounds include skew and rotation.
    """
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, (int, np.integer)) or scene_seed < 0:
        raise ValueError('scene_seed must be a nonnegative integer')
    if distribution not in ('id', 'ood'):
        raise ValueError('Unknown distribution')
    rng = np.random.Generator(np.random.PCG64(int(scene_seed)))
    count = int(rng.integers(1, 3))
    bumps = []
    for i in range(count):
        row = dict(id=f'p{i}', family=FAMILIES[int(rng.integers(4))],
                   x_m=float(rng.uniform(-.045, .045) if count == 1 else
                             rng.uniform(-.048, -.025) if i == 0 else rng.uniform(.025, .048)),
                   y_m=0., radius_x_m=float(rng.uniform(.007, .013)),
                   radius_y_m=float(rng.uniform(.007, .013)), height_m=float(rng.uniform(.005, .010)),
                   yaw_deg=float(rng.uniform(-12, 12) if distribution == 'id' else rng.uniform(-35, 35)),
                   peak_offset=float(rng.uniform(-.35, .35) if distribution == 'id' else rng.uniform(-.65, .65)))
        hx, hy = protrusion_half_extents(row)
        room = .016-hy-.0005
        if room <= 0:
            raise SceneInfeasible('rotated_footprint_outside_base', f'protrusion {i}, half width {hy}')
        row['y_m'] = float(rng.uniform(-min(room, .002), min(room, .002)))
        if abs(row['x_m'])+hx >= .0795:
            raise SceneInfeasible('protrusion_outside_base')
        bumps.append(row)
    if count == 2:
        gap = bumps[1]['x_m']-bumps[0]['x_m']-sum(protrusion_half_extents(b)[0] for b in bumps)
        if gap < .008:
            raise SceneInfeasible('insufficient_inter_bump_clearance', str(gap))
    return resolve_scene(dict(schema='rcs_scene_v1', scene_id=f'agentic_trim_{distribution}_s{int(scene_seed)}',
        description='Seeded continuous heightfield for agentic control robot research; provisional material.',
        workpiece=dict(base_box_m=dict(length_m=.16, width_m=.032, height_m=.02), protrusions=bumps,
            world_pose=dict(translation_m=[0., 0., .01], yaw_deg=0.), material_profile='cutting_tasks_v1',
            geometry_provider=dict(id='heightfield_v2')),
        supports=dict(kind='flat_board', workcell=dict(profile='blade_master_flat_board_v1')),
        tool=dict(profile_id='ruler_knife_v2_y', grasp_preset='default'),
        solver=dict(module='tlmpm_v7_2', cutting_law=8, max_cuts=2,
                    particle_spacing_m=.001, grid_dx_m=.004)))


def goal_config():
    """Fixed duration and two available digits do not disclose bump count."""
    return dict(schema='rcs_goal_v1', goal_id='agentic_trim_all_v1', type='remove_protrusions',
        description=GOAL_TEXT, episode=dict(duration_s=24., control_dt_s=.02),
        planner=dict(controller_profile='path_v2', rake_deg=10., max_translation_m=.001, max_rotation_deg=1.),
        select='all', plane_local_z_m=None, edge_clearance_m=.0002,
        minimum_removed_fraction=.9, minimum_substrate_share=.99, maximum_detached_below_fraction=.01)


def compile_seed(scene_seed, *, policy_seed=0, sim_seed=42, distribution='id', output_dir=None):
    """Compile before Native startup, recording all draws and initial-state hashes."""
    scene = generate_scene(scene_seed, distribution=distribution)
    task = compile_task(scene, goal_config(), seed=int(sim_seed))
    points, masses, lattice = task.sample_particles()
    connectivity = lattice_report(lattice)
    if not connectivity['ok']:
        raise SceneInfeasible('initial_connectivity', canonical_json(connectivity))
    # Every target must contain resolved material above the actual scoring plane.
    above = points[:, 2] > task.geometry.base_height_m
    counts = []
    for bump in task.geometry.bumps:
        target = above & (bump.height(points[:, 0], points[:, 1]) >
                          points[:, 2]-task.geometry.base_height_m)
        counts.append(int(target.sum()))
    if min(counts) < 20:
        raise SceneInfeasible('unresolved_protrusion', str(counts))
    manifest = dict(version=VERSION, status='PASS', scene_seed=int(scene_seed), policy_seed=int(policy_seed),
        sim_seed=int(sim_seed), distribution=distribution, rng='numpy.random.PCG64', numpy_version=np.__version__,
        sampled_parameters=scene['workpiece'], particle_count=len(points), target_particle_counts=counts,
        initial_lattice_connectivity=connectivity, initial_mechanical_connectivity='MEASURE_AT_NATIVE_RESET',
        scene_hash=sha256_json(scene), geometry_hash=sha256_json(scene['workpiece']),
        material_hash=sha256_json(asdict(task.material)), solver_hash=sha256_json(scene['solver']),
        goal_hash=sha256_json(task.goal), task_hash=task.config_sha256,
        initial_particles_sha256=digest_array(points), initial_masses_sha256=digest_array(masses),
        initial_lattice_sha256=digest_array(lattice), sampled_scene=scene,
        material_calibration='provisional_uncalibrated', coordinate_convention='SI; world Z up; quaternion XYZW')
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=False)
        for name, data in [('scene', scene), ('goal', task.goal), ('scene_manifest', manifest),
                           ('compiled_task', json.loads(configured_task_payload(task)))]:
            (out/f'{name}.json').write_text(json.dumps(data, indent=2)+'\n')
    return task, manifest


def evaluate_recording(recording_dir):
    """Read-only original support-interface evaluator over a finalized recording.

    Call in a fresh process with the frozen simulator source; controller return
    values and images never determine success. Gradients participate in the
    original mechanical graph, including unassigned (code-zero) bridges.
    """
    from .scene_config import task_from_payload
    from .recording import load_physics_recording
    out = Path(recording_dir)
    if (out/'attempts.json').exists():
        attempts = json.loads((out/'attempts.json').read_text())
        if not attempts:
            return dict(status='NOT_RUN', success=False, reason='No attempt recorded')
        results = []
        for i, attempt in enumerate(attempts):
            expected = f'attempt_{i:03d}'
            if attempt['path'] != expected or (out/expected).is_symlink():
                raise ValueError('Invalid attempt recording path')
            results.append(evaluate_recording(out/expected))
        return {**results[-1], 'attempt_count': len(results),
                'attempt_outcomes': [dict(status=r['status'], success=bool(r.get('success'))) for r in results]}
    task = task_from_payload(json.loads((out/'compiled_task.json').read_text()))
    data, checkpoint = load_physics_recording(out)
    if len(data['time_s']) < 2 or data['time_s'][-1] <= 0:
        return dict(status='NOT_RUN', success=False, reason='No physical transition recorded',
                    **({'recording_completion': checkpoint} if checkpoint is not None else {}))
    data.update(code_encoding='packed2', max_cuts=2, particle_spacing_m=task.particle_spacing_m)
    if (out/'surface_state.npz').exists():
        with np.load(out/'surface_state.npz', allow_pickle=False) as source:
            data.update({key: source[key] for key in source.files})
    result = task.evaluate_goal(data)
    completion = {}
    if checkpoint is not None:
        # These are recording-I/O checks. The original physical graph, metrics
        # and thresholds above remain unchanged, even for interrupted prefixes.
        integrity = result['integrity']
        checks = {**integrity.get('checks', {}),
                  'stream_finalized': checkpoint['finalized'] is True,
                  'stream_complete': checkpoint['complete'] is True}
        result['integrity'] = {**integrity, 'checks': checks,
            'status': 'PASS' if integrity['status'] == 'PASS' and all(checks.values()) else 'FAIL'}
        if result['integrity']['status'] != 'PASS':
            result.update(verdict='FAIL_INTEGRITY', success=False)
        completion = dict(recording_format=checkpoint['schema'],
            recording_completion={key: checkpoint[key] for key in
                ('sample_count', 'first_time_s', 'last_time_s', 'finalized', 'complete')})
    receipt = out/'episode_receipt.json'
    home_status = (json.loads(receipt.read_text()).get('home_status', 'NOT_RUN')
                   if receipt.exists() or checkpoint is None else 'NOT_RUN')
    return {**result, 'status': result['verdict'], 'success': bool(result.get('success', False)),
            **completion,
            'evaluator_version': VERSION, 'task_id': task.task_id,
            'simulation_duration_s': float(data['time_s'][-1]),
            'surface_quality_status': 'MEASURED' if (out/'surface_state.npz').exists() else 'NOT_RUN_MESH_NOT_EXPORTED',
            'home_status': home_status}
