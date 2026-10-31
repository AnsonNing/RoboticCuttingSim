#!/usr/bin/env python3
"""WP-12 Phase 2b: read-only analysis of one live Native workcell episode (configs/wp12/workcell_live_v1.json).

Reads ``ROOT/inputs`` (prepare.json, scene.json, goal.json), ``ROOT/attempt.json`` and ``ROOT/run`` (the
``scripts/run_wp10_live.py`` output) and writes ``--out`` (new directory):

* ``analysis.json``: outcome class (WP-11 live precedence and gates, or the declared partial-cut rule), goal
  evaluation, IK / grasp / contact audits (reused from ``run_generalization_live.py`` and
  ``analyze_phase_c5_live.py``), knife lowest point vs the board top, material particles vs the board top, blade
  reaction, inferred board reaction, object displacement (whole workpiece, per piece, per cut), crossing links and
  direct code-0 bridges per through-cut plane, camera start/end checks, performance;
* ``frames/``: start/end RGB and depth PNGs of the global D455 and both wrist D405 (and presentation views).

Oracle material state and privileged tool state are used for evaluation only, never as controller input.

  .venv/bin/python scripts/run_wp12_workcell_live_analysis.py --root artifacts/wp12/workcell_live_001
"""
import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
GRASP_GATES = {'max_slip_m': 0.002, 'max_rotation_rad': math.pi/60,
               'source': 'configs/wp11/generalization_round1_live.json outcome_classes.GRASP_FAILURE.gates'}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GL = _load_script('run_generalization_live')
C5 = _load_script('analyze_phase_c5_live')
TC = _load_script('run_wp12_board_throughcut')


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                else v.item() if isinstance(v, np.generic) else str(v)))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def stats(values):
    return GL._stats(list(values))


def pose_matrix(pose):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_quat(np.asarray(pose[3:], float)).as_matrix(), np.asarray(pose[:3], float)
    return T


def blade_points_tool(task):
    from robotic_cutting_sim.tool_profile import tool_spec_for_task
    spec = tool_spec_for_task(task)
    rows = [np.asarray(s.vertices_tool_m, float) for s in spec.shapes if s.role == 'blade']
    return np.vstack(rows), spec


def knife_vs_board(task, rows, board_top):
    blade, spec = blade_points_tool(task)
    out = []
    for r in rows:
        T = pose_matrix(r['measured']['tool_body_pose_xyzw'])
        world = blade@T[:3, :3].T+T[:3, 3]
        out.append(float(world[:, 2].min())-board_top)
    return np.asarray(out), spec


def finger_force(row):
    grasp = (row.get('privileged_diagnostics') or {}).get('grasp') or {}
    by_role = ((grasp.get('contacts') or {}).get('tensor_forces') or {}).get('force_by_role_world_on_tool_n') or {}
    return np.sum([np.asarray(by_role.get(k, [0., 0., 0.]), float) for k in ('left_finger', 'right_finger', 'palm')], axis=0)


def material_force(row, dt):
    return np.asarray((row.get('privileged_diagnostics') or {}).get('current_material_reaction_impulse_world', [0.]*6), float)[:3]/dt


def applied_material_force(row):
    return np.asarray((row.get('privileged_diagnostics') or {}).get('applied_previous_material_wrench_world', [0.]*6), float)[:3]


def board_contact_roles(row):
    roles = ((((row.get('privileged_diagnostics') or {}).get('grasp') or {}).get('contacts') or {}).get('roles') or {})
    return {k: int((roles.get(k) or {}).get('contact_count', 0)) for k in ('fixture', 'table', 'other')}


def phase_base(row):
    return str(row.get('policy_phase', '')).split(':', 1)[0]


def op_of(row):
    return GL._op_of(row.get('policy_phase'))


def load_physics(path, task):
    with np.load(path) as data:
        d = {k: data[k] for k in data.files}
    d['positions_local'] = np.stack([task.points_to_local(p.astype(float)) for p in d['positions']])
    d['reference_local'] = np.asarray(d['reference_positions_local'], float)
    return d


def components(d, codes, cuts):
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    return mechanical_components_packed(d['reference_stencil_nodes'], d['reference_stencil_weights'], codes, cuts,
                                        gradients=d['reference_stencil_gradients'])


def frame_index_at(times, t, after=True):
    times = np.asarray(times, float)
    if after:
        idx = np.flatnonzero(times >= t-1e-9)
        return int(idx[0]) if len(idx) else len(times)-1
    idx = np.flatnonzero(times <= t+1e-9)
    return int(idx[-1]) if len(idx) else 0


def displacement(d, task, rows, prep):
    times = np.asarray(d['time_s'], float)
    m = np.asarray(d['mass_kg'], float)
    x = d['positions_local']
    c = np.einsum('i,fij->fj', m, x)/m.sum()
    final_count, final_labels = components(d, d['codes'][-1], task.max_cuts)
    out = {'whole_workpiece_centroid_local_m': {'start': c[0].tolist(), 'end': c[-1].tolist(),
                                                 'drift_m': (c[-1]-c[0]).tolist(),
                                                 'horizontal_drift_m': float(np.linalg.norm(c[-1, :2]-c[0, :2])),
                                                 'max_horizontal_drift_m': float(np.linalg.norm(c[:, :2]-c[0, :2], axis=1).max())},
           'final_components': int(final_count)}
    pieces = []
    for label in range(final_count):
        sel = final_labels == label
        if m[sel].sum() < .01*m.sum():
            continue
        cp = np.einsum('i,fij->fj', m[sel], x[:, sel])/m[sel].sum()
        pieces.append({'label': int(label), 'mass_fraction': float(m[sel].sum()/m.sum()),
                       'reference_x_range_m': [float(d['reference_local'][sel, 0].min()), float(d['reference_local'][sel, 0].max())],
                       'drift_end_minus_start_m': (cp[-1]-cp[0]).tolist(),
                       'horizontal_drift_m': float(np.linalg.norm(cp[-1, :2]-cp[0, :2]))})
    out['pieces'] = sorted(pieces, key=lambda p: p['reference_x_range_m'][0])
    # Per operation: frame before the armed segment starts vs the frame after the operation's last row.
    per_op = []
    for op in prep.get('plan_report_operations') or []:
        sub = [r for r in rows if op_of(r) == op['op']]
        armed = [r for r in sub if (r.get('submitted') or {}).get('cut_enabled')]
        if not armed:
            continue
        i0 = frame_index_at(times, float(armed[0]['time_s'])-.02, after=False)
        i1 = frame_index_at(times, float(sub[-1]['time_s']), after=True)
        disp = x[i1]-x[i0]
        whole = (np.einsum('i,ij->j', m, disp)/m.sum())
        rec = {'op': op['op'], 'kind': op['kind'], 'frames': [i0, i1], 'times_s': [float(times[i0]), float(times[i1])],
               'whole_workpiece_mean_displacement_m': whole.tolist(),
               'whole_workpiece_horizontal_m': float(np.linalg.norm(whole[:2]))}
        if op['kind'] == 'through_cut':
            x0 = float(op['plane_origin_local_m'][0])
            ref_x = d['reference_local'][:, 0]
            sides = {}
            for name, sel in (('negative_side', ref_x < x0), ('positive_side', ref_x > x0)):
                if sel.any():
                    v = np.einsum('i,ij->j', m[sel], disp[sel])/m[sel].sum()
                    sides[name] = {'mean_displacement_m': v.tolist(), 'x_m': float(v[0]), 'horizontal_m': float(np.linalg.norm(v[:2]))}
            rec['by_side_of_plane'] = sides
        per_op.append(rec)
    out['per_operation'] = per_op
    out['scope'] = ('Mass-weighted means of the recorded particle centres (privileged material state, evaluation only), '
                    'workpiece-local frame; the per-operation interval runs from the last recorded frame before the armed '
                    'segment to the first frame after the operation (record_every steps).')
    return out


def material_vs_board(d):
    z = d['positions_local'][..., 2]
    per_frame_min = z.min(1)
    below = (z < 0).sum(1)
    i = int(np.argmin(per_frame_min))
    return {'min_particle_centre_minus_board_top_m': float(per_frame_min.min()), 'at_frame': i,
            'at_time_s': float(d['time_s'][i]), 'max_particles_below_board_top_in_a_frame': int(below.max()),
            'frames_with_particles_below_board_top': int((below > 0).sum()), 'frames': int(len(per_frame_min)),
            'final_min_particle_centre_minus_board_top_m': float(per_frame_min[-1]),
            'scope': 'Particle centres (not the contact shell) vs the board top (workpiece-local z = 0), every recorded frame'}


def bridges(d, task, prep):
    from robotic_cutting_sim.cut_topology import decode_digits
    out = []
    codes = d['codes'][-1]
    digits_all = decode_digits(codes, task.max_cuts)
    lattice = np.asarray(d['reference_indices'])
    ref = d['reference_local']
    valid = TC.stencil_valid(d['reference_stencil_weights'], d['reference_stencil_gradients'])
    for op in prep.get('plan_report_operations') or []:
        if op['kind'] != 'through_cut':
            continue
        x0 = float(op['plane_origin_local_m'][0])
        digit = int(op['digit'])
        digits = digits_all[:, digit]
        # lattice i of the last particle column below the plane
        columns = sorted({(int(i), float(x)) for i, x in zip(lattice[:, 0], ref[:, 0])})
        left = [i for i, x in columns if x < x0]
        plane_i = max(left)
        # The live kerf can land off the declared plane (the bar slides between cuts): also locate the lattice
        # column pair with the most conflicting links for this digit (searched over the labelled x range).
        lab = np.flatnonzero(digits != 0)
        best_i, best = plane_i, -1
        if len(lab):
            for i in range(int(lattice[lab, 0].min())-1, int(lattice[lab, 0].max())+1):
                c = sum(r['conflicting'] for r in TC.crossing_links(lattice, digits, i).values())
                if c > best:
                    best_i, best = i, c
        x_of = dict(columns)
        best_x = 0.5*(x_of.get(best_i, np.nan)+x_of.get(best_i+1, np.nan))
        links = TC.crossing_links(lattice, digits, best_i)
        declared = TC.crossing_links(lattice, digits, plane_i)
        direct = TC.direct_bridges(d['reference_stencil_nodes'], valid, digits)
        k = lattice[:, 2]
        out.append({'op': op['op'], 'digit': digit, 'plane_x_local_m': x0, 'plane_i': int(plane_i),
                    'declared_plane_conflicting_links': [int(sum(r['conflicting'] for r in declared.values())),
                                                         int(sum(r['links'] for r in declared.values()))],
                    'kerf_plane_i': int(best_i), 'kerf_reference_x_local_m': float(best_x),
                    'kerf_offset_from_declared_m': float(best_x-x0),
                    'crossing_links': {'links': int(sum(r['links'] for r in links.values())),
                                       'conflicting': int(sum(r['conflicting'] for r in links.values())),
                                       'by_layer_first4': {int(z): links[z] for z in list(links)[:4]},
                                       'layers_with_unconflicted': [int(z) for z, r in links.items() if r['unconflicted']]},
                    'direct_code0_bridges': {'count': int(len(direct)),
                                             'by_layer': {int(z): int(c) for z, c in zip(*np.unique(k[direct], return_counts=True))}},
                    'labelled_particles_this_digit': int(np.count_nonzero(digits)),
                    'labelled_reference_z_min_m': float(ref[digits != 0, 2].min()) if np.any(digits != 0) else None})
    return out


def camera_frames(run, out):
    from PIL import Image
    obs = run/'episode_000'/'observations'
    names = sorted({p.name.rsplit('_', 1)[0] for p in obs.glob('*.npz')})
    frames_dir = out/'frames'
    frames_dir.mkdir()
    report = {}
    for name in names:
        files = sorted(obs.glob(f'{name}_[0-9][0-9][0-9][0-9].npz'))
        if not files:
            continue
        rec = {'frames': len(files)}
        for tag, f in (('start', files[0]), ('end', files[-1])):
            with np.load(f) as z:
                data = {k: z[k] for k in z.files}
            Image.fromarray(np.asarray(data['rgb'], np.uint8)).save(frames_dir/f'{tag}_{name}_rgb.png')
            entry = {'file': str(f.relative_to(ROOT)), 'time_s': float(data.get('physics_timestamp_s', np.nan))}
            if 'depth_m' in data:
                depth = np.asarray(data['depth_m'], float)
                valid = np.asarray(data['valid_mask'], bool)
                entry.update(depth_valid_fraction=float(valid.mean()),
                             depth_valid_min_m=float(depth[valid].min()) if valid.any() else None,
                             depth_valid_median_m=float(np.median(depth[valid])) if valid.any() else None,
                             depth_valid_max_m=float(depth[valid].max()) if valid.any() else None)
                if valid.any():
                    lo, hi = np.percentile(depth[valid], [1, 99])
                    img = np.zeros(depth.shape, np.uint8)
                    img[valid] = (255*(1-np.clip((depth[valid]-lo)/max(hi-lo, 1e-6), 0, 1))).astype(np.uint8)
                    Image.fromarray(img).save(frames_dir/f'{tag}_{name}_depth.png')
            if 'T_world_camera' in data:
                entry['T_world_camera_translation_m'] = np.asarray(data['T_world_camera'])[:3, 3].tolist()
            rec[tag] = entry
        if 'T_world_camera_translation_m' in rec.get('start', {}) and 'T_world_camera_translation_m' in rec.get('end', {}):
            rec['camera_translation_start_to_end_m'] = float(np.linalg.norm(np.subtract(rec['end']['T_world_camera_translation_m'],
                                                                                         rec['start']['T_world_camera_translation_m'])))
        report[name] = rec
    return report


def camera_motion_trace(run, rows):
    """Max wrist / D455 optical translation over the episode from the runner camera pose trace."""
    trace = read_json(run/'episode_000'/'camera_pose_trace.json')
    if not trace:
        return None
    samples = trace['samples']
    out = {}
    for name in samples[0]['cameras']:
        p = np.asarray([np.asarray(s['cameras'][name]['T_world_camera'])[:3, 3] for s in samples])
        out[name] = {'samples': len(p), 'max_displacement_from_first_m': float(np.linalg.norm(p-p[0], axis=1).max()),
                     'max_measured_vs_current_USD_translation_m': float(max(s['cameras'][name]['measured_vs_current_USD']['translation_error_m']
                                                                           for s in samples))}
    return out


def live_classification(run):
    """Phase 2b class (``GL.classify_live`` with ``GRASP_GATES`` and the task-phase IK / workspace-clamp counts) of one
    ``scripts/run_wp10_live.py`` output directory, computed exactly as ``main`` does but without compiling the task
    (the class needs no task geometry). Read-only. Used by ``run_wp12_phase5.native_execution`` (the declared Phase 5
    Native success definition); ``tests/test_wp12_phase5_native_success.py`` checks it reproduces the stored
    ``analysis.json`` class of every Phase 2b record ``workcell_live_001-007``."""
    run = Path(run)
    top, failure = read_json(run/'result.json'), read_json(run/'failure.json')
    episode = read_json(run/'episode_000'/'result.json')
    rows, evidence_errors = [], []
    actions = run/'episode_000'/'actions.jsonl'
    if not actions.is_file():
        actions = actions.with_suffix('.jsonl.gz')
    try:
        opener = gzip.open if actions.suffix == '.gz' else open
        with opener(actions, 'rt') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not all(isinstance(r, dict) for r in rows):
            raise ValueError('action rows must be mappings')
    except (OSError, ValueError, EOFError) as error:
        rows = []
        evidence_errors.append(f'actions unavailable: {type(error).__name__}')
    task_rows = [r for r in rows if not GL.is_lifecycle_home(r)]
    if not task_rows:
        evidence_errors.append('no recorded task-phase actions')
    for row in task_rows:
        grasp = row.get('grasp_status')
        info = row.get('info') or {}
        try:
            complete = (isinstance(grasp, dict) and 'grasp_lost' in grasp
                        and all(np.isfinite(float(grasp[k])) for k in
                                ('slip_since_settling_m', 'rotation_since_settling_rad'))
                        and 'ik_status' in info and info['ik_status'] is not None
                        and 'workspace_clamped' in (info.get('action_report') or {}))
        except (KeyError, TypeError, ValueError):
            complete = False
        if not complete:
            evidence_errors.append('incomplete/non-finite task-phase grasp or control measurements')
            break
    ik = {'task_steps_ik_not_ok': int(sum((r.get('info') or {}).get('ik_status') not in (None, 'ok') for r in task_rows)),
          **{f'task_steps_{k}': int(sum(bool(((r.get('info') or {}).get('action_report') or {}).get(k)) for r in task_rows))
             for k in ('workspace_clamped', 'translation_clamped', 'rotation_clamped')}}
    grasp_task = C5.grasp_audit(task_rows)
    finite, physics = None, run/'episode_000'/'physics_state.npz'
    try:
        with np.load(physics) as data:
            positions, velocities = data['positions'], data['velocities']
            if positions.ndim != 3 or positions.shape[-1] != 3 or not positions.size or velocities.shape != positions.shape:
                raise ValueError('empty or inconsistent recorded state')
            finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
    except (OSError, ValueError, KeyError, EOFError) as error:
        evidence_errors.append(f'physics state unavailable: {type(error).__name__}')
    live = {'outcome_classes': {'GRASP_FAILURE': {'gates': GRASP_GATES}}}
    cls, reason = GL.classify_live(live, top, episode, rows, failure, finite is not False, grasp_task, ik)
    # Preserve the frozen classifier's failure precedence and numerical gates.
    # Missing receipts are unknown evidence, never measured zero slip or finite state.
    if cls == 'TASK_SUCCESS' and evidence_errors:
        cls, reason = 'CUTTING_FAILURE', 'integrity: '+ '; '.join(evidence_errors)
    evaluation = (episode or {}).get('task_outcome') or {}
    return {'class': cls, 'reason': reason, 'ik': ik, 'grasp_gates': GRASP_GATES,
            'grasp_task_phase': {k: grasp_task.get(k) for k in ('grasp_lost', 'max_slip_m', 'max_rotation_rad')},
            'finite_recorded_state': finite, 'physics_state_present': physics.exists(),
            'recorded_evidence_errors': evidence_errors, 'actions_source': str(actions),
            'result_present': top is not None, 'refused_before_launch': bool((top or {}).get('refused_before_launch')),
            'run_status': (top or {}).get('status'), 'episode_result_present': episode is not None,
            'failure_error': ((failure or {}).get('error') or '')[:500] or None,
            'task_end_reason': (episode or {}).get('task_end_reason'),
            'policy_final_phase': (episode or {}).get('policy_final_phase'), 'verdict': evaluation.get('verdict'),
            'task_id': (episode or {}).get('task_id'), 'runtime_source': (episode or {}).get('runtime_source'),
            'home_outcome': ((episode or {}).get('home_outcome') or {}).get('status'), 'control_rows': len(rows)}


def partial_rule(evaluation, d, task, finished, grasp_ok, ik_ok):
    from robotic_cutting_sim.cut_topology import decode_digits
    digits = decode_digits(d['codes'][-1], task.max_cuts)[:, 0]
    ref_z = d['reference_local'][:, 2]
    labelled = digits != 0
    integrity = (evaluation.get('integrity') or {}).get('status')
    n = ((evaluation.get('outcome') or {}).get('metrics') or {}).get('mechanical_components')
    floor = 0.010-0.004
    checks = {'integrity_pass': integrity == 'PASS', 'one_mechanical_component': n == 1,
              'kerf_exists': bool(labelled.any()),
              'no_label_below_declared_depth_minus_one_cell': bool(labelled.any() and ref_z[labelled].min() >= floor-1e-9),
              'plan_finished_no_grasp_or_ik_failure': bool(finished and grasp_ok and ik_ok)}
    return {'class': 'PARTIAL_CUT_ACCEPTED' if all(checks.values()) else 'PARTIAL_CUT_FAIL', 'checks': checks,
            'mechanical_components': n, 'labelled_particles_cut0': int(labelled.sum()),
            'labelled_reference_z_min_m': float(ref_z[labelled].min()) if labelled.any() else None,
            'labelled_reference_z_floor_m': floor, 'sever_verdict': evaluation.get('verdict'),
            'rule': 'configs/wp12/workcell_live_v1.json episodes[PARTIAL].success'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--skip-path-check', action='store_true')
    p.add_argument('--wp11-sample-dir', type=Path,
                   help='Analyse a WP-11 Phase E live sample instead (DIR/inputs, DIR/run_001, DIR/attempt_001.json); '
                        '--root is then only the parent of --out')
    a = p.parse_args(argv)
    from robotic_cutting_sim.scene_config import compile_task
    root = a.root if a.root.is_absolute() else ROOT/a.root
    out = a.out or root/'analysis_001'
    out = out if out.is_absolute() else ROOT/out
    out.mkdir(parents=False, exist_ok=False)
    inputs, attempt_path, run = root/'inputs', root/'attempt.json', root/'run'
    if a.wp11_sample_dir is not None:
        sample = a.wp11_sample_dir if a.wp11_sample_dir.is_absolute() else ROOT/a.wp11_sample_dir
        inputs, attempt_path, run = sample/'inputs', sample/'attempt_001.json', sample/'run_001'
    prep = read_json(inputs/'prepare.json')
    prep.setdefault('episode', prep.get('sample_id'))
    attempt = read_json(attempt_path)
    top, failure = read_json(run/'result.json'), read_json(run/'failure.json')
    episode = read_json(run/'episode_000'/'result.json')
    rows = []
    if (run/'episode_000'/'actions.jsonl').exists():
        rows = [json.loads(line) for line in (run/'episode_000'/'actions.jsonl').open() if line.strip()]
    task = compile_task(str(inputs/'scene.json'), str(inputs/'goal.json'),
                        planner_options=prep['planner_overrides'], seed=42)
    geometry = task.workcell_geometry()
    board_top = geometry['board_top_world_z_m'] if geometry else float(task.T_world_workpiece[2, 3])
    task_rows = [r for r in rows if not GL.is_lifecycle_home(r)]
    ik = {'task_steps_ik_not_ok': int(sum((r.get('info') or {}).get('ik_status') not in (None, 'ok') for r in task_rows)),
          **{f'task_steps_{k}': int(sum(bool(((r.get('info') or {}).get('action_report') or {}).get(k)) for r in task_rows))
             for k in ('workspace_clamped', 'translation_clamped', 'rotation_clamped')}}
    grasp_task, grasp_all = C5.grasp_audit(task_rows), C5.grasp_audit(rows)
    finite, d = True, None
    physics = run/'episode_000'/'physics_state.npz'
    if physics.exists():
        d = load_physics(physics, task)
        finite = bool(np.isfinite(d['positions']).all() and np.isfinite(d['velocities']).all())
    live = {'outcome_classes': {'GRASP_FAILURE': {'gates': GRASP_GATES}}}
    cls, reason = GL.classify_live(live, top, episode, rows, failure, finite, grasp_task, ik)
    evaluation = (episode or {}).get('task_outcome') or {}
    analysis = {'root': str(root.relative_to(ROOT)), 'run': str(run), 'episode': prep['episode'], 'class': cls, 'class_reason': reason,
                'runtime_source': (episode or {}).get('runtime_source'), 'attempt': attempt,
                'task_id_recorded': (episode or {}).get('task_id'), 'task_id_recompiled_cpu': task.task_id,
                'task_id_match': (episode or {}).get('task_id') == task.task_id,
                'board_top_world_z_m': board_top, 'workcell_geometry_present': geometry is not None}
    analysis['task_outcome'] = {'verdict': evaluation.get('verdict'),
                                'evaluator_version': evaluation.get('evaluator_version', 'goal_eval_v1 (non-sequence goal: same evaluator as goal_eval_v2a)'),
                                'integrity_checks': (evaluation.get('integrity') or {}).get('checks'),
                                'outcome_checks': (evaluation.get('outcome') or {}).get('checks'),
                                'failed_checks': GL.G.failed_checks(evaluation) if evaluation else None,
                                'outcome_metrics': (evaluation.get('outcome') or {}).get('metrics'),
                                'quality': evaluation.get('quality')}
    try:
        analysis['metrics'] = GL.G.episode_metrics(json.loads((inputs/'goal.json').read_text()), evaluation) if evaluation else None
    except Exception as error:  # report-only
        analysis['metrics'] = {'error': repr(error)}
    finished = (episode or {}).get('task_end_reason') == 'CONTROLLER_DONE'
    if prep['episode'] == 'PARTIAL' and d is not None and evaluation:
        analysis['partial_cut'] = partial_rule(evaluation, d, task, finished,
                                               not cls.startswith('GRASP'), not cls.startswith('IK'))
        analysis['class_declared_rule'] = analysis['partial_cut']['class']
    analysis['home'] = {'reset_info': (episode or {}).get('reset_info'),
                        'home_outcome': ((episode or {}).get('home_outcome') or {}),
                        'home_measurement': (episode or {}).get('home_measurement'),
                        'episode_state_final': (episode or {}).get('episode_state_final')}
    analysis['plan_preflight'] = {k: (read_json(run/'plan_preflight.json') or {}).get(k) for k in ('status', 'reason', 'required')}
    analysis['ik'] = ik
    analysis['grasp_task_phase'], analysis['grasp_all'] = grasp_task, grasp_all
    analysis['contacts_task_phase'] = C5.contact_audit(task_rows)
    analysis['task_end_reason'] = (episode or {}).get('task_end_reason')
    analysis['policy_final_phase'] = (episode or {}).get('policy_final_phase')
    analysis['phases'] = GL.phase_summary(rows)
    if rows:
        analysis['operations'] = GL.operation_reports(task, prep, rows)
        dt = task.control_dt_s
        h, spec = knife_vs_board(task, rows, board_top)
        task_mask = np.asarray([not GL.is_lifecycle_home(r) for r in rows])
        kb = {'min_over_episode_m': float(h.min()), 'min_over_task_phase_m': float(h[task_mask].min()),
              'max_penetration_into_board_m': float(max(0., -h.min())),
              'at_step': int(np.argmin(h))+1, 'at_phase': rows[int(np.argmin(h))].get('policy_phase'),
              'per_operation': []}
        roles = [board_contact_roles(r) for r in rows]
        kb['steps_with_tool_board_contact_pairs'] = int(sum(r['fixture'] > 0 for r in roles))
        kb['steps_with_tool_table_contact_pairs'] = int(sum(r['table'] > 0 for r in roles))
        ff = np.asarray([finger_force(r) for r in rows])
        fm = np.asarray([material_force(r, dt) for r in rows])
        fa = np.asarray([applied_material_force(r) for r in rows])
        # Free hover: last 20 steps of the first APPROACH / REORIENT phase (no material contact).
        first = next((i for i, r in enumerate(rows) if phase_base(r) in ('APPROACH', 'REORIENT')), None)
        last = first
        while last is not None and last+1 < len(rows) and phase_base(rows[last+1]) == phase_base(rows[first]):
            last += 1
        free_idx = list(range(max(first, last-19), last+1)) if first is not None else []
        free = ff[free_idx].mean(0) if free_idx else np.zeros(3)
        board = -(ff-free)-fa
        weight = float(spec.mass_kg*9.81)
        forces = {'free_hover_finger_force_on_tool_world_n': free.tolist(), 'tool_weight_n': weight,
                  'free_hover_steps': [free_idx[0]+1, free_idx[-1]+1] if free_idx else None,
                  'per_operation': []}
        for op in prep.get('plan_report_operations') or []:
            idx = [i for i, r in enumerate(rows) if op_of(r) == op['op']]
            armed = [i for i in idx if (rows[i].get('submitted') or {}).get('cut_enabled')]
            hold = [i for i in idx if phase_base(rows[i]) == 'HOLD']
            entry = {'op': op['op'], 'kind': op['kind'],
                     'knife_lowest_minus_board_top_armed_m': stats(h[armed]) if armed else None,
                     'knife_lowest_minus_board_top_hold_m': stats(h[hold]) if hold else None,
                     'knife_lowest_minus_board_top_op_min_m': float(h[idx].min()) if idx else None,
                     'steps_with_tool_board_contact_pairs': int(sum(roles[i]['fixture'] > 0 for i in idx))}
            if armed:
                last_armed = armed[-1]
                entry['knife_lowest_minus_board_top_at_end_of_armed_m'] = float(h[last_armed])
                entry['clean_cut_window_h_m'] = [0.0, 0.000375]
                if op['kind'] == 'through_cut':
                    entry['end_of_armed_within_clean_cut_window'] = bool(-1e-6 <= h[last_armed] <= 0.000375)
            kb['per_operation'].append(entry)
            seg = sorted(set(armed+hold))
            forces['per_operation'].append({
                'op': op['op'], 'blade_reaction_armed_norm_n': stats(np.linalg.norm(fm[armed], axis=1)) if armed else None,
                'blade_reaction_armed_z_n': stats(fm[armed, 2]) if armed else None,
                'board_reaction_inferred_armed_hold_z_n': stats(board[seg, 2]) if seg else None,
                'board_reaction_inferred_hold_mean_world_n': board[hold].mean(0).tolist() if hold else None,
                'board_reaction_inferred_hold_z_max_n': float(board[hold, 2].max()) if hold else None})
        forces['board_reaction_inferred_task_max_z_n'] = float(board[task_mask, 2].max())
        forces['board_reaction_inferred_scope'] = ('Quasi-static: F_board = -(F_fingers - F_fingers,free) - F_material,applied '
                                                   '(native tensor finger/palm forces of the last substep of each control step; '
                                                   'inertia neglected, so values during fast motion are not board forces).')
        analysis['knife_vs_board'] = kb
        analysis['forces'] = forces
        wall = [r.get('step_wall_s') for r in rows if r.get('step_wall_s') is not None]
        mat = [(r.get('privileged_diagnostics') or {}).get('material_wall_s') for r in rows]
        analysis['performance'] = {'control_steps': len(rows), 'step_wall_ms_mean': 1e3*float(np.mean(wall)) if wall else None,
                                   'step_wall_ms_p95': 1e3*float(np.percentile(wall, 95)) if wall else None,
                                   'material_wall_ms_mean': 1e3*float(np.mean([m for m in mat if m is not None])) if any(m is not None for m in mat) else None,
                                   'episode_wall_s': (episode or {}).get('wall_s'), 'run_total_wall_s': (top or {}).get('wall_s'),
                                   'process_wall_s': (attempt or {}).get('wall_s'), 'run_dir_bytes': (attempt or {}).get('run_dir_bytes')}
    if d is not None:
        analysis['material_vs_board'] = material_vs_board(d)
        analysis['displacement'] = displacement(d, task, rows, prep)
        analysis['bridges'] = bridges(d, task, prep)
        analysis['finite_recorded_state'] = finite
    analysis['cameras'] = camera_frames(run, out) if (run/'episode_000'/'observations').exists() else None
    analysis['camera_pose_trace'] = camera_motion_trace(run, rows)
    if rows and not a.skip_path_check:
        try:
            analysis['measured_task_path_clearance'] = GL.measured_path_clearance(task, task_rows, out)
        except Exception as error:  # report-only
            analysis['measured_task_path_clearance'] = {'status': 'NOT_RUN', 'reason': repr(error)}
        if len(getattr(task, 'protrusion_ids', ())) >= 2:
            try:
                analysis['measured_task_path_clearance_staged'] = GL.measured_path_clearance_staged(task, task_rows)
            except Exception as error:  # report-only
                analysis['measured_task_path_clearance_staged'] = {'status': 'NOT_RUN', 'reason': repr(error)}
    analysis['failure'] = None if failure is None else {'error': failure.get('error')}
    git = lambda *x: subprocess.run(['git', *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    analysis['analysis_tool'] = {'git_sha': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                                 'command': sys.argv}
    analysis['scope'] = 'Read-only analysis of a live Native episode; oracle material and privileged tool state are evaluation-only'
    (out/'analysis.json').write_text(json.dumps(jsonable(analysis), indent=1, allow_nan=True))
    print(json.dumps(jsonable({'episode': prep['episode'], 'class': cls, 'reason': reason, 'verdict': evaluation.get('verdict'),
                               'declared': analysis.get('class_declared_rule'),
                               'home': ((episode or {}).get('home_outcome') or {}).get('status')})))
    return 0


if __name__ == '__main__':
    sys.exit(main())
