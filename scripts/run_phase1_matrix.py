#!/usr/bin/env python3
"""Phase 1 cutting-law-4 acceptance matrix (ADR 0006), one row per invocation.

Rows M1a..M9d run kinematic ideal-tracking episodes (or measured-trajectory
material replays for M9) with cutting law 4 and score them with the frozen
oracle plus the read-only BladeMaster evaluators. Every component count uses
the actual eight-node stencil graph (``mechanical_components`` with recorded
weights and gradients). The row verdict is computed exactly from the ADR 0006
criterion; law parameters and thresholds are never changed here.

Kinematic ideal tracking says nothing about robot reachability, grasp, slip or
cameras. M8b is a kinematic preflight of Demo 3, not a demo. Oracle material
state is evaluation-only, never an observation.

  .venv/bin/python scripts/run_phase1_matrix.py --row M4 --device cuda:1 --out <fresh dir containing law4>
  .venv/bin/python scripts/run_phase1_matrix.py --row M1a --ingest <probe_wp10_task_policy output> --out <fresh dir>
  .venv/bin/python scripts/run_phase1_matrix.py --row M7 --sources <M1a dir> <M1b dir> --out <fresh dir>
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.control import ActionLimits  # noqa: E402
from robotic_cutting_sim.cut_topology import mechanical_components  # noqa: E402
from robotic_cutting_sim.demo_evaluation import (fit_swept_plane, label_distance_distribution,  # noqa: E402
                                                 slice_push_report, swept_plane_interface, trim_separation_for_task)
from robotic_cutting_sim.multi_cut_evaluation import MultiCutGoal, evaluate_multi_cut  # noqa: E402
from robotic_cutting_sim.policies import Waypoint, WaypointPolicy, skim, slice_push, through_cut  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402

DEFAULT_LAW = 4  # ADR 0006; --cutting-law selects a later versioned law for the same matrix
SEED = 42
# Force-present levels: the law-4 CPU test levels (tests/test_cutting_law_v4.py, coarse): side > 0.1 N, flat > 1 N.
SIDE_FORCE_N, FLAT_FORCE_N = .1, 1.
NATIVE = {'M9a': ('CUT-01-R', 'artifacts/wp10/c_native_cut01r_reference_001/episode_000', True),
          'M9b': ('CUT-02', 'artifacts/wp10/c_native_cut02_reference_001/episode_000', True),
          'M9c': ('CUT-01-R', 'artifacts/wp10/c_lab_cut01r_reference_001/episode_000', False),
          'M9d': ('CUT-02', 'artifacts/wp10/c_lab_cut02_reference_001/episode_000', False)}

CRITERIA = {
    'M1a': 'CUT-01-R reference vertical cut: frozen strict oracle PASS (two parts, mass, location <= 2 mm, persistence)',
    'M1b': 'CUT-02 reference vertical cut: frozen strict oracle PASS',
    'M2': '20 deg oblique cut (tilt about the edge line) through CUT-01-R: >= 2 substantial parts (>= 40% each, '
          '>= 99% together) and swept-plane interface statistic <= 2 mm; frozen strict reported as is',
    'M3a': 'TRIM-01-R 25 deg reference skim: frozen reference_above_plane_material_removed AND largest component '
           '>= 99% of below-plane reference mass; precision reported separately',
    'M3b': 'M3a criterion on TRIM-01-R scenario ruler_workcell_v2 (35 deg)',
    'M3c': 'M3a criterion on TRIM-01-R scenario blademaster_workcell_v1 (35 deg, cutting board)',
    'M4': 'Armed vertical cut to half depth (local z = 10 mm), then armed withdrawal: one component, labelled '
          'count unchanged during withdrawal, code-0 ligament below the stopping depth',
    'M5a': 'Armed side push (+x 20 mm into the bar end face, edge at z = 10 mm): contact force present (mean |F| '
           '> 0.1 N), zero labels, one component',
    'M5b': 'Armed flat press (blade plane horizontal, n = +z, 5 mm into the bar top): force present (mean |F| > 1 N), '
           'zero labels, one component',
    'M5c': 'Armed half-depth cut then armed blade-normal move +x 10 mm: labels unchanged during the sideways move',
    'M6': 'Armed half-depth cut, armed stop 0.5 s, armed withdrawal: labels unchanged during the stop and withdrawal',
    'M7': 'From M1a/M1b: separated topology persists >= 0.5 s after withdrawal (frozen '
          'postcut_disconnected_topology_persists) and the component count never drops after separation',
    'M8a': 'Two sequential cuts (cut_index 0 at x = -10 mm, 1 at x = +10 mm) with a disarmed lift/move: >= 3 '
           'substantial mechanical parts in the 2-digit graph; lift/move adds no labels',
    'M8b': 'Demo-3 kinematic preflight SLICE-PUSH-01 (blademaster_workcell_v1): slice_push_report PASS',
    'M9a': 'Native CUT-01-R measured-trajectory replay with law 4: frozen strict PASS',
    'M9b': 'Native CUT-02 measured-trajectory replay with law 4: frozen strict PASS',
    'M9c': 'Lab CUT-01-R measured-trajectory replay with law 4: report only (Lab parity is P1)',
    'M9d': 'Lab CUT-02 measured-trajectory replay with law 4: report only (Lab parity is P1)'}


def git_state():
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    import hashlib
    return {'git_sha': sha, 'git_dirty': dirty, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


# ----------------------------------------------------------------------------- plans

def _pose(position, rotation):
    return tuple(np.r_[position, rotation.as_quat()])


def _local(task, xyz):
    T = task.T_world_workpiece
    return T[:3, :3] @ np.asarray(xyz, float) + T[:3, 3]


def half_depth_plan(task, *, after):
    """Armed vertical descent at x = 0 to local z = 10 mm, then a row-specific tail."""
    rotation = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])  # d = -z, e = +y, n = +x
    top = task.geometry.base_height_m
    start, stop = _local(task, [0., 0., top + .008]), _local(task, [0., 0., .010])
    plan = [Waypoint(_pose(start + [0., 0., .03], rotation), label='approach'),
            Waypoint(_pose(start, rotation), label='start'),
            Waypoint(_pose(stop, rotation), cut=True, speed_m_s=.01, label='descend')]
    if after == 'withdraw':
        plan.append(Waypoint(_pose(start, rotation), cut=True, speed_m_s=.01, label='withdraw'))
    elif after == 'hold_withdraw':
        plan += [Waypoint(_pose(stop, rotation), cut=True, hold_s=.5, label='hold'),
                 Waypoint(_pose(start, rotation), cut=True, speed_m_s=.01, label='withdraw')]
    elif after == 'side':
        plan.append(Waypoint(_pose(stop + task.T_world_workpiece[:3, 0] * .010, rotation), cut=True, speed_m_s=.01,
                             label='side_move'))
    return plan


def side_push_plan(task):
    rotation = Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])
    begin, end = _local(task, [-.088, 0., .010]), _local(task, [-.068, 0., .010])
    return [Waypoint(_pose(begin + [0., 0., .048], rotation), label='approach'),
            Waypoint(_pose(begin, rotation), label='lower_outside'),
            Waypoint(_pose(end, rotation), cut=True, speed_m_s=.01, label='side_push')]


def flat_press_plan(task):
    """Blade plane horizontal (n = +z), lower face pressed 5 mm into the bar top.

    The edge points along +y (d = +y, e = -x) instead of +x so that the blunt
    handle box (edge frame x in [-80, -40] mm, |n| <= 11 mm) stays beside the
    bar (|y| <= 15 mm) rather than ramming its top; only the blade face
    contacts the material.
    """
    rotation = Rotation.from_rotvec([0., 0., np.pi / 2])
    half_thickness = task.tool_profile.thickness_m / 2
    top = task.geometry.base_height_m
    start = _local(task, [0., .020, top + .003 + half_thickness])
    end = _local(task, [0., .020, top - .005 + half_thickness])
    return [Waypoint(_pose(start + [0., 0., .03], rotation), label='approach'),
            Waypoint(_pose(start, rotation), label='start'),
            Waypoint(_pose(end, rotation), cut=True, speed_m_s=.01, label='press')]


def labelled(plan, labels):
    return [replace(w, label=l) for w, l in zip(plan, labels)]


def two_cut_plan(task):
    names = ('approach', 'start', 'cut', 'end', 'lift')
    first = labelled(through_cut(task, offset_m=-.010, cut_index=0), [f'{n}_0' for n in names])
    second = labelled(through_cut(task, offset_m=.010, cut_index=1), ['move_1', 'start_1', 'cut_1', 'end_1', 'retract_1'])
    return first + second


ROW_CONFIG = {
    # row: (task, scenario, max_cuts, plan builder, position tolerance or None for the policy default)
    'M1a': ('CUT-01-R', None, 1, lambda t: labelled(through_cut(t), ('approach', 'start', 'cut', 'end', 'retract')), None),
    'M1b': ('CUT-02', None, 1, lambda t: labelled(through_cut(t), ('approach', 'start', 'cut', 'end', 'retract')), None),
    'M2': ('CUT-01-R', None, 1, lambda t: labelled(through_cut(t, tilt_deg=20.), ('approach', 'start', 'cut', 'end', 'retract')), None),
    'M3a': ('TRIM-01-R', None, 1, lambda t: labelled(skim(t), ('approach', 'start', 'skim', 'retract')), None),
    'M3b': ('TRIM-01-R', 'ruler_workcell_v2', 1, lambda t: labelled(skim(t), ('approach', 'start', 'skim', 'retract')), None),
    'M3c': ('TRIM-01-R', 'blademaster_workcell_v1', 1, lambda t: labelled(skim(t), ('approach', 'start', 'skim', 'retract')), None),
    'M4': ('CUT-01-R', None, 1, lambda t: half_depth_plan(t, after='withdraw'), 1e-5),
    'M5a': ('CUT-01-R', None, 1, side_push_plan, 1e-5),
    'M5b': ('CUT-01-R', None, 1, flat_press_plan, 1e-5),
    'M5c': ('CUT-01-R', None, 1, lambda t: half_depth_plan(t, after='side'), 1e-5),
    'M6': ('CUT-01-R', None, 1, lambda t: half_depth_plan(t, after='hold_withdraw'), 1e-5),
    'M8a': ('CUT-01-R', None, 2, two_cut_plan, None),
    'M8b': ('SLICE-PUSH-01', 'blademaster_workcell_v1', 2, slice_push, None)}


# ----------------------------------------------------------------------------- episodes

def run_episode(row, args, out):
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    task_id, scenario, max_cuts, builder, tolerance = ROW_CONFIG[row]
    env = make_kinematic_env(task_id, seed=SEED, device=args.device, particle_spacing_m=args.spacing,
                             material_dt=args.material_dt, scenario=scenario, cutting_law=args.cutting_law, max_cuts=max_cuts,
                             action_limits=ActionLimits(), ik_failure='reject', delta_frame='world',
                             history_stride=args.history_stride)
    observation, _ = env.reset(seed=SEED)
    plan = builder(env.task)
    policy = WaypointPolicy(env.task, plan, env.action_limits,
                            **({} if tolerance is None else {'position_tolerance_m': tolerance}))
    dt = env.task.control_dt_s
    steps = []
    limit = int(round(env.task.duration_s / dt)) if args.steps is None else args.steps
    with (out / 'actions.jsonl').open('w') as stream:
        for step in range(limit):
            action = policy(observation)
            phase = policy.phase
            observation, _, terminated, truncated, info = env.step(action)
            material = env.backend.last_diagnostics['material']
            stroke = env.backend.stroke_history[-1]
            blade = np.asarray(material['current_material_reaction_impulse_world'], float) / dt
            total = np.asarray(material['current_total_material_reaction_impulse_world'], float) / dt
            row_data = dict(step=step + 1, time_s=env.time_s, phase=phase, cut_enabled=bool(action['cut_enabled']),
                            cut_index=int(material['cut_index']), edge_pose_world=env.backend.edge.tolist(),
                            blade_force_world_N=blade[:3].tolist(), total_force_world_N=total[:3].tolist(),
                            contact_statistics=np.asarray(material['material_contact_statistics']).tolist(),
                            stroke=stroke['stroke'], travel_m=stroke['travel_m'], engaged=stroke['engaged'],
                            any_motion=stroke['any_motion'], labelled_count=stroke['labelled_count'],
                            action={**action, 'target': np.asarray(action['target']).tolist()},
                            ik_status=info['ik_status'],
                            workspace_clamped=bool((info.get('action_report') or {}).get('workspace_clamped', False)))
            stream.write(json.dumps(jsonable(row_data)) + '\n')
            steps.append(row_data)
            if terminated or truncated:
                break
    material = env.backend.material
    recording = {'time_s': np.asarray([s['time_s'] for s in env._history], float),
                 **{k: np.asarray([s[k] for s in env._history]) for k in ('positions', 'velocities', 'codes')},
                 'mass_kg': material.masses, 'reference_positions': material.world_points,
                 'reference_indices': material.lattice,
                 'reference_stencil_nodes': material.bed.nodes.numpy(),
                 'reference_stencil_weights': material.bed.weights.numpy(),
                 'reference_stencil_gradients': material.bed.gradients.numpy(), **env.backend.law4_stroke_arrays()}
    evaluations = {}
    if max_cuts == 1:
        evaluations['strict'] = env.evaluate()
        if row in ('M1a', 'M1b'):
            evaluations['research'] = env.evaluate(profile='research')
    np.savez_compressed(out / 'physics_state.npz', **recording)
    context = {'task': env.task, 'max_cuts': max_cuts, 'policy_done': policy.done, 'blade_half_width_m': float(
        material.bed.config.blade_half_width), 'law4_parameters': material.bed.law4_parameters,
        'material_receipt': material.parameter_receipt, 'runtime_source': env.runtime_source,
        'plan': [dict(label=w.label, pose=list(w.pose), cut=w.cut, cut_index=w.cut_index, speed_m_s=w.speed_m_s,
                      hold_s=w.hold_s) for w in plan], 'position_tolerance_m': policy.position_tolerance_m}
    env.close()
    return steps, recording, evaluations, context


def ingest_probe(row, source, law):
    """Reuse a completed scripts/probe_wp10_task_policy.py output (same settings) as M1a/M3a evidence."""
    from robotic_cutting_sim.task_scenarios import configure_task
    from robotic_cutting_sim.tasks import make_task
    task_id, scenario, max_cuts, _, _ = ROW_CONFIG[row]
    source = Path(source)
    result, probe_manifest = json.loads((source / 'result.json').read_text()), json.loads((source / 'manifest.json').read_text())
    config = probe_manifest['config']
    if (config.get('task') != task_id or config.get('cutting_law') != law or config.get('scenario') != scenario
            or abs(float(config.get('spacing', -1)) - .001) > 1e-12 or float(config.get('saw', 0)) != 0.):
        raise ValueError(f'{source} is not a law-{law} 1 mm {task_id} reference probe')
    data = dict(np.load(source / 'physics_state.npz'))
    data['time_s'] = np.asarray(data['time_s'], float)
    task = configure_task(make_task(task_id, seed=SEED), scenario=scenario)
    data['reference_indices'] = task.sample_particles()[2]
    law4 = {k: data[k] for k in data if k.startswith('law4_')}
    steps = []
    for index, line in enumerate((source / 'actions.jsonl').open()):
        record = json.loads(line)
        steps.append(dict(step=record['step'], time_s=record['time_s'], phase=None,
                          cut_enabled=bool(record['action']['cut_enabled']),
                          cut_index=int(law4['law4_cut_index'][index]),
                          edge_pose_world=record['info']['action_report']['resolved_target_world_xyz_xyzw'],
                          stroke=int(law4['law4_stroke'][index]), labelled_count=int(law4['law4_labelled_count'][index])))
        if not np.isclose(law4['law4_time_s'][index], record['time_s'], atol=1e-9):
            raise ValueError('Probe actions and law-4 stroke record are misaligned')
    evaluations = {'strict': result['strict']}
    if 'research' in result:
        evaluations['research'] = result['research']
    context = {'task': task, 'max_cuts': max_cuts, 'policy_done': None, 'blade_half_width_m': task.tool_profile.half_width_m,
               'law4_parameters': None, 'material_receipt': result.get('material_receipt'),
               'runtime_source': result.get('runtime_source'), 'ingested_from': str(source),
               'ingested_manifest': {k: probe_manifest.get(k) for k in ('command', 'git_sha', 'git_dirty', 'timestamp', 'config')},
               'ingested_wall_s': result.get('wall_s'),
               'note': 'Coordinator smoke ingested as row evidence; history_stride 10 (probe default); per-step forces not recorded'}
    return steps, data, evaluations, context


def run_replay(row, args, out):
    task_id, episode, _ = NATIVE[row]
    replay_out = out / 'replay'
    command = [sys.executable, str(ROOT / 'scripts/replay_measured_edge.py'), '--episode', str(ROOT / episode),
               '--task', task_id, '--cutting-law', str(args.cutting_law), '--device', args.device, '--out', str(replay_out)]
    with (out / 'replay.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'replay_measured_edge.py exited {completed.returncode}; see {out / "replay.log"}')
    from robotic_cutting_sim.tasks import make_task
    task = make_task(task_id, seed=SEED)
    result = json.loads((replay_out / 'result.json').read_text())
    data = dict(np.load(replay_out / 'physics_state.npz'))
    data['time_s'] = np.asarray(data['time_s'], float)
    data['reference_indices'] = task.sample_particles()[2]
    steps = []
    for index, line in enumerate((ROOT / episode / 'actions.jsonl').open()):
        material = json.loads(line)['privileged_diagnostics']
        steps.append(dict(step=index + 1, time_s=json.loads(line)['time_s'], phase=None,
                          cut_enabled=bool(material['active_cut']), cut_index=int(material['cut_index']),
                          edge_pose_world=list(material['measured_tool_pose']),
                          stroke=int(data['law4_stroke'][index]), labelled_count=int(data['law4_labelled_count'][index])))
    context = {'task': task, 'max_cuts': 1, 'policy_done': None, 'blade_half_width_m': task.tool_profile.half_width_m,
               'law4_parameters': None, 'material_receipt': None, 'runtime_source': result['runtime_source'],
               'replay_command': command, 'source_episode': episode, 'replay_wall_s': result['wall_s'],
               'replay_steps': result['steps']}
    return steps, data, {'strict': result['strict'], 'research': result['research']}, context


# ----------------------------------------------------------------------------- analysis

def components(data, codes, cuts):
    return mechanical_components(np.asarray(data['reference_stencil_nodes']), np.asarray(data['reference_stencil_weights']),
                                 np.asarray(codes), cuts, gradients=np.asarray(data['reference_stencil_gradients']))


def component_history(data, cuts):
    cache, counts = {}, []
    for codes in np.asarray(data['codes']):
        key = codes.tobytes()
        if key not in cache:
            cache[key] = int(components(data, codes, cuts)[0])
        counts.append(cache[key])
    return counts


def mass_fractions(data, labels, count):
    masses = np.asarray(data['mass_kg'], float)
    return np.sort(np.bincount(labels, weights=masses, minlength=count) / masses.sum())[::-1]


def swept_planes(steps, max_cuts, half_width):
    """Plane per cut digit from edge poses at control steps with law4_stroke == digit + 1."""
    planes = {}
    for digit in range(max_cuts):
        rows = [s for s in steps if s['stroke'] == digit + 1]
        entry = {'digit': digit, 'stroke_active_steps': len(rows)}
        if len(rows) < 2:
            planes[digit] = {**entry, 'status': 'NOT_RUN', 'reason': 'Fewer than two control steps with an active stroke'}
            continue
        poses = np.asarray([s['edge_pose_world'] for s in rows], float)
        rotations = Rotation.from_quat(poses[:, 3:])
        lines, normals = rotations.apply([0., 1., 0.]), rotations.apply([0., 0., 1.])
        try:
            fit = fit_swept_plane(poses[:, :3], lines, edge_half_length_m=half_width, reference_normal=normals.mean(0))
        except ValueError as error:
            planes[digit] = {**entry, 'status': 'NOT_RUN', 'reason': str(error)}
            continue
        planes[digit] = {**entry, 'status': 'FIT', 'fit': fit,
                         'stroke_active_time_s': [float(rows[0]['time_s']), float(rows[-1]['time_s'])],
                         'reference_normal_mean_blade_normal': normals.mean(0).tolist()}
    return planes


def plane_reports(data, planes, max_cuts):
    codes = np.asarray(data['codes'])[-1]
    reference = np.asarray(data['reference_positions'], float)
    out = {}
    for digit, plane in planes.items():
        if plane['status'] != 'FIT':
            out[digit] = {'label_distance': None, 'interface': None, 'reason': plane['reason']}
            continue
        origin, normal = plane['fit']['origin'], plane['fit']['normal']
        distance = label_distance_distribution(reference, codes, origin, normal, digit=digit)
        # Separation by this digit alone (1-digit graph of the digit) for the interface statistic.
        digit_codes = (codes // 3 ** digit) % 3
        count, labels = components(data, digit_codes.astype(codes.dtype), 1)
        interface = swept_plane_interface(data['reference_stencil_nodes'], data['reference_stencil_weights'],
                                          data['reference_stencil_gradients'], labels, reference, origin, normal)
        out[digit] = {'label_distance': distance, 'interface': interface, 'digit_only_components': int(count),
                      'ideal_partition_interface_floor_m': ideal_partition_interface_floor(data, origin, normal)}
    return out


def ideal_partition_interface_floor(data, origin, normal):
    """Report-only: the swept-plane midpoint statistic for a perfect sign-of-plane partition.

    Every reference particle is put on the side given by the sign of its signed
    distance to the plane (no physics). The value is the smallest the frozen
    midpoint statistic can be for that plane on this reference lattice and
    stencil; it is never used in a verdict.
    """
    reference = np.asarray(data['reference_positions'], float)
    side = ((reference - np.asarray(origin, float)) @ np.asarray(normal, float) > 0).astype(np.int32)
    return swept_plane_interface(data['reference_stencil_nodes'], data['reference_stencil_weights'],
                                 data['reference_stencil_gradients'], side, reference, origin,
                                 normal)['maximum_reference_midpoint_plane_distance_m']


def phase_label_changes(steps, phases):
    """Labelled-count change across the control steps of the given phases (must be one contiguous block)."""
    indices = [i for i, s in enumerate(steps) if s['phase'] in phases]
    if not indices:
        return {'phases': list(phases), 'steps': 0, 'status': 'NOT_RUN'}
    first, last = indices[0], indices[-1]
    if last - first + 1 != len(indices):
        raise ValueError(f'Phases {phases} are not one contiguous block of control steps')
    before = steps[first - 1]['labelled_count'] if first > 0 else 0
    counts = [steps[i]['labelled_count'] for i in range(first, last + 1)]
    return {'phases': list(phases), 'steps': last - first + 1, 'time_s': [steps[first]['time_s'] - .02, steps[last]['time_s']],
            'labelled_before': before, 'labelled_after': counts[-1], 'labelled_max': max(counts), 'labelled_min': min(counts),
            'change': int(max(abs(max(counts) - before), abs(min(counts) - before))),
            'stroke_active_steps': int(sum(steps[i]['stroke'] != 0 for i in range(first, last + 1)))}


def force_summary(steps, phase):
    rows = [s for s in steps if s['phase'] == phase]
    if not rows or 'blade_force_world_N' not in rows[0]:
        return {'phase': phase, 'steps': len(rows), 'status': 'NOT_RUN'}
    blade = np.linalg.norm([s['blade_force_world_N'] for s in rows], axis=1)
    total = np.linalg.norm([s['total_force_world_N'] for s in rows], axis=1)
    mean_vector = np.mean([s['total_force_world_N'] for s in rows], 0)
    contacts = np.asarray([s['contact_statistics'][1] for s in rows])
    return {'phase': phase, 'steps': len(rows), 'blade_force_mean_N': float(blade.mean()), 'blade_force_max_N': float(blade.max()),
            'total_force_mean_N': float(total.mean()), 'total_force_max_N': float(total.max()),
            'total_force_mean_vector_world_N': mean_vector.tolist(),
            'steps_with_blade_contacts': int(np.count_nonzero(contacts > 0)),
            'scope': 'Material reaction impulse / control dt (current step, world); provisional uncalibrated material'}


def strict_summary(strict):
    keys = ('final_components', 'substantial_parts', 'two_main_parts_mass_fraction', 'maximum_reference_midpoint_plane_distance_m',
            'postcut_persistence_time_s', 'postcut_graph_count_minimum', 'removed_reference_above_plane_mass_fraction')
    metrics = strict.get('metrics', {})
    return {'status': strict.get('status'), 'checks': strict.get('checks'),
            'metrics': {k: metrics[k] for k in keys if k in metrics}}


def ligament(data, stop_local_z, task):
    codes = np.asarray(data['codes'])[-1]
    local = task.points_to_local(np.asarray(data['reference_positions'], float))
    below = local[:, 2] < stop_local_z
    near = below & (np.abs(local[:, 0]) <= 2 * task.grid_dx_m)
    lab = codes != 0
    return {'stop_local_z_m': float(stop_local_z), 'labelled_below_stop': int(np.count_nonzero(lab & below)),
            'code0_fraction_near_plane_below_stop': float(np.mean(codes[near] == 0)) if near.any() else None,
            'near_plane_below_stop_particles': int(near.sum()),
            'deepest_labelled_reference_local_z_m': float(local[lab, 2].min()) if lab.any() else None,
            'labelled_total': int(np.count_nonzero(lab))}


def evaluate_row(row, steps, data, evaluations, context):
    task, max_cuts = context['task'], context['max_cuts']
    planes = swept_planes(steps, max_cuts, context['blade_half_width_m'])
    reports = plane_reports(data, planes, max_cuts)
    count, labels = components(data, np.asarray(data['codes'])[-1], max_cuts)
    fractions = mass_fractions(data, labels, count)
    metrics = {'final_mechanical_components': int(count), 'final_component_mass_fractions': fractions[:6].tolist(),
               'final_labelled_count': int(np.count_nonzero(np.asarray(data['codes'])[-1])),
               'swept_planes': planes, 'swept_plane_reports': reports,
               'component_count_history': component_history(data, max_cuts),
               'history_time_s': np.asarray(data['time_s']).tolist()}
    if 'strict' in evaluations:
        metrics['frozen_strict'] = strict_summary(evaluations['strict'])
    checks = {}
    if row in ('M1a', 'M1b', 'M9a', 'M9b', 'M9c', 'M9d'):
        checks['frozen_strict_pass'] = evaluations['strict'].get('status') == 'PASS_TASK_METRICS'
    elif row == 'M2':
        interface = (reports.get(0) or {}).get('interface') or {}
        distance = interface.get('maximum_reference_midpoint_plane_distance_m')
        checks = {'two_substantial_parts_each_ge_40pct': len(fractions) >= 2 and fractions[1] >= .40,
                  'two_parts_together_ge_99pct': float(fractions[:2].sum()) >= .99,
                  'swept_plane_interface_le_2mm': distance is not None and distance <= .002}
        metrics['swept_plane_interface_m'] = distance
    elif row.startswith('M3'):
        separation = trim_separation_for_task(task, data)
        metrics['trim_separation'] = separation
        strict_checks = evaluations['strict'].get('checks', {})
        checks = {'frozen_reference_above_plane_material_removed': bool(strict_checks.get('reference_above_plane_material_removed')),
                  'largest_component_holds_ge_99pct_below_plane_mass':
                      bool(separation['checks']['largest_component_holds_below_plane_mass'])}
        metrics['frozen_precision_checks'] = {k: v for k, v in strict_checks.items() if k != 'reference_above_plane_material_removed'}
        metrics['detached_below_plane_mass_fraction'] = separation['metrics']['detached_below_plane_mass_fraction']
        metrics['trim_separation_cross_check_removed'] = separation['checks']['reference_above_plane_material_removed']
    elif row in ('M4', 'M6'):
        descent = [s for s in steps if s['phase'] == 'descend']
        stop_z = task.points_to_local(np.asarray([descent[-1]['edge_pose_world'][:3]]))[0, 2] if descent else np.nan
        metrics['ligament'] = ligament(data, stop_z, task)
        metrics['descent_label_change'] = phase_label_changes(steps, ('descend',))
        metrics['withdraw_label_change'] = phase_label_changes(steps, ('withdraw',))
        checks = {'one_component': count == 1,
                  'labels_unchanged_during_withdrawal': metrics['withdraw_label_change'].get('change') == 0,
                  'code0_ligament_below_stop': metrics['ligament']['labelled_below_stop'] == 0}
        if row == 'M6':
            metrics['hold_label_change'] = phase_label_changes(steps, ('hold',))
            checks = {'labels_unchanged_during_hold': metrics['hold_label_change'].get('change') == 0,
                      'labels_unchanged_during_withdrawal': checks['labels_unchanged_during_withdrawal'],
                      'one_component': checks['one_component'], 'code0_ligament_below_stop': checks['code0_ligament_below_stop']}
            metrics['hold_steps_at_zero_velocity'] = metrics['hold_label_change'].get('steps')
    elif row in ('M5a', 'M5b'):
        phase, level = ('side_push', SIDE_FORCE_N) if row == 'M5a' else ('press', FLAT_FORCE_N)
        metrics['force'] = force_summary(steps, phase)
        metrics['phase_label_change'] = phase_label_changes(steps, (phase,))
        checks = {'contact_force_present': metrics['force'].get('total_force_mean_N', 0.) > level,
                  'zero_labels': metrics['final_labelled_count'] == 0 and max(s['labelled_count'] for s in steps) == 0,
                  'one_component': count == 1}
        metrics['force_present_level_N'] = level
    elif row == 'M5c':
        metrics['side_move_label_change'] = phase_label_changes(steps, ('side_move',))
        metrics['force'] = force_summary(steps, 'side_move')
        checks = {'labels_unchanged_during_side_move': metrics['side_move_label_change'].get('change') == 0}
        metrics['one_component'] = count == 1
    elif row == 'M8a':
        goal = MultiCutGoal(planes=((tuple(_local(task, [-.010, 0., 0.])), tuple(task.T_world_workpiece[:3, 0])),
                                    (tuple(_local(task, [.010, 0., 0.])), tuple(task.T_world_workpiece[:3, 0]))))
        multi = evaluate_multi_cut(data, goal)
        metrics['multi_cut'] = multi
        metrics['lift_move_label_change'] = phase_label_changes(steps, ('end_0', 'lift_0', 'move_1', 'start_1'))
        checks = {'three_mechanical_parts': multi['checks']['three_mechanical_parts'],
                  'three_substantial_parts': multi['checks']['three_substantial_parts'],
                  'lift_move_adds_no_labels': metrics['lift_move_label_change'].get('change') == 0}
        metrics['multi_cut_status_full'] = multi['status']
    elif row == 'M8b':
        push = [s for s in steps if s['phase'] == 'push']
        recording = dict(data)
        if push:
            recording['push_interval_s'] = (float(push[0]['time_s'] - task.control_dt_s), float(push[-1]['time_s']))
        report = slice_push_report(task, recording)
        metrics['slice_push_report'] = report
        metrics['lift_move_label_change'] = phase_label_changes(steps, ('hold_0', 'lift_0', 'move_1'))
        metrics['disabled_post_cut_label_change'] = phase_label_changes(
            steps, ('raise_in_kerf', 'push', 'push_hold', 'release', 'retract', 'done'))
        checks = {'slice_push_report_pass': report['status'] == 'PASS_SLICE_PUSH'}
        metrics['preflight_scope'] = 'Kinematic ideal-tracking preflight of Demo 3; not a demo, not a live episode'
    if row in ('M9c', 'M9d'):
        verdict = 'REPORT_ONLY'
    else:
        verdict = 'PASS' if checks and all(checks.values()) else 'FAIL'
    return verdict, {k: bool(v) for k, v in checks.items()}, metrics


def evaluate_m7(sources):
    rows, checks = [], {}
    for source in sources:
        source = Path(source)
        result = json.loads((source / 'result.json').read_text())
        # An ingested row keeps its physics state in the original probe output.
        state = source / 'physics_state.npz'
        if not state.exists():
            state = Path(result['context']['ingested_from']) / 'physics_state.npz'
        data = np.load(state)
        strict = result['evaluations']['strict']
        counts = result['metrics']['component_count_history']
        time_s = result['metrics']['history_time_s']
        separated = next((i for i, c in enumerate(counts) if c >= 2), None)
        drops = [] if separated is None else [float(time_s[i + 1]) for i in range(separated, len(counts) - 1)
                                              if counts[i + 1] < counts[i]]
        persists = bool(strict['checks'].get('postcut_disconnected_topology_persists'))
        rows.append({'source': str(source), 'row': result['row'], 'frozen_postcut_disconnected_topology_persists': persists,
                     'postcut_persistence_time_s': strict['metrics'].get('postcut_persistence_time_s'),
                     'postcut_graph_count_minimum': strict['metrics'].get('postcut_graph_count_minimum'),
                     'first_separated_frame_time_s': None if separated is None else float(time_s[separated]),
                     'component_count_after_separation': counts[separated:] if separated is not None else [],
                     'count_drop_times_s': drops, 'history_frames': int(len(data['time_s']))})
        checks[f'{result["row"]}_persists'] = persists
        checks[f'{result["row"]}_count_never_drops'] = separated is not None and not drops
    return ('PASS' if checks and all(checks.values()) else 'FAIL'), checks, {'sources': rows}


def bridge_diagnosis(npz, out, task_id, scenario):
    """Run scripts/analyze_cut_bridges.py on a failed row's physics state (oracle-only diagnostic)."""
    command = [sys.executable, str(ROOT / 'scripts/analyze_cut_bridges.py'), str(npz), '--task', task_id]
    if scenario:
        command += ['--scenario', scenario]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out / 'bridges.json').write_text(completed.stdout)
    if completed.returncode:
        return {'command': command, 'returncode': completed.returncode, 'stderr': completed.stderr[-4000:]}
    return {'command': command, 'report': json.loads(completed.stdout)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', required=True, choices=sorted([*ROW_CONFIG, *NATIVE, 'M7']))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--cutting-law', type=int, default=DEFAULT_LAW, choices=(4, 5, 6, 7),
                   help='Opt-in stroke-based cutting law (default 4, ADR 0006; 5 ADR 0008; 6 ADR 0010; 7 ADR 0011); recorded in '
                        'result.json and the output name check')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=.001, help='Particle spacing (formal: 1 mm)')
    p.add_argument('--material-dt', type=float, default=1e-4, help='Material dt (formal: 1e-4 s)')
    p.add_argument('--history-stride', type=int, default=5)
    p.add_argument('--steps', type=int, help='Validation only: cap control steps (rows are not formal then)')
    p.add_argument('--ingest', type=Path, help='M1a/M3a only: reuse a completed law-4 probe_wp10_task_policy output')
    p.add_argument('--sources', type=Path, nargs='*', help='M7 only: completed M1a and M1b row directories')
    a = p.parse_args()
    if f'law{a.cutting_law}' not in a.out.name:
        p.error(f'--out name must contain law{a.cutting_law} so rows of different laws never mix')
    out = new_output(a.out)  # refuses to overwrite (exist_ok=False)
    start = time.monotonic()
    config = {**vars(a), 'out': str(a.out), 'ingest': None if a.ingest is None else str(a.ingest),
              'sources': None if a.sources is None else [str(s) for s in a.sources], 'seed': SEED}
    write_json(out / 'manifest.json', manifest(config, f'phase1_matrix_law{a.cutting_law}'))
    formal = a.spacing == .001 and a.material_dt == 1e-4 and a.steps is None
    base = {'row': a.row, 'criterion': CRITERIA[a.row], 'command': sys.argv, **git_state(), 'seed': SEED,
            'cutting_law': a.cutting_law, 'particle_spacing_m': a.spacing, 'material_dt_s': a.material_dt, 'grid_dx_m': .004,
            'history_stride': a.history_stride, 'device': a.device, 'formal_resolution': formal}
    try:
        if a.row == 'M7':
            if not a.sources:
                raise ValueError('M7 needs --sources <M1a dir> <M1b dir>')
            verdict, checks, metrics = evaluate_m7(a.sources)
            write_json(out / 'result.json', jsonable({**base, 'verdict': verdict, 'checks': checks, 'metrics': metrics,
                                                      'runtime_source': 'derived_from_rows', 'wall_s': time.monotonic() - start}))
            print(a.row, verdict, checks, flush=True)
            return
        if a.ingest is not None:
            if a.row not in ('M1a', 'M3a'):
                raise ValueError('--ingest is only for M1a/M3a')
            steps, data, evaluations, context = ingest_probe(a.row, a.ingest, a.cutting_law)
            base['history_stride'] = 10
            with (out / 'steps.jsonl').open('w') as stream:
                for s in steps:
                    stream.write(json.dumps(jsonable(s)) + '\n')
        elif a.row in NATIVE:
            steps, data, evaluations, context = run_replay(a.row, a, out)
            base['history_stride'] = 10
        else:
            steps, data, evaluations, context = run_episode(a.row, a, out)
        verdict, checks, metrics = evaluate_row(a.row, steps, data, evaluations, context)
        if not formal and verdict == 'PASS':
            verdict = 'PASS_NONFORMAL'
        task = context.pop('task')
        diagnosis = None
        if verdict == 'FAIL':
            npz = (a.ingest / 'physics_state.npz' if a.ingest else
                   out / 'replay' / 'physics_state.npz' if a.row in NATIVE else out / 'physics_state.npz')
            diagnosis = bridge_diagnosis(npz, out, task.task_id, ROW_CONFIG.get(a.row, (None, None))[1])
        write_json(out / 'result.json', jsonable({
            **base, 'task_id': task.task_id, 'scenario': ROW_CONFIG.get(a.row, (None, None))[1], 'verdict': verdict,
            'checks': checks, 'metrics': metrics, 'evaluations': evaluations, 'context': context,
            'bridge_diagnosis': diagnosis, 'steps_run': len(steps), 'wall_s': time.monotonic() - start}))
        print(a.row, verdict, checks, flush=True)
    except Exception as error:
        write_json(out / 'result.json', jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                  'wall_s': time.monotonic() - start}))
        raise


if __name__ == '__main__':
    main()
