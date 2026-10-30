#!/usr/bin/env python3
"""WP-11 Phase C4: material-level near-horizontal TRIM with ruler_knife_v2 (kinematic, one run per invocation).

Each invocation runs one kinematic ideal-tracking episode of a cutting_tasks_v4
TRIM-02 task (end-held v2 knife at a declared rake rho, tau = psi = 0) with a
versioned cutting law (default 8, ADR 0012) and records the outcome plus
diagnostics. The outcome is ``demo_evaluation.trim_separation_for_task``
(>= 90% of reference above-plane mass removed, substrate component >= 99% of the
below-plane mass, detached below-plane mass <= 1%); the frozen precision checks
(residual, p95, RMS, gouge, overcut) are reported separately. Negative controls
(``--control``) use the same task and score the predeclared CONTROL_CRITERIA.
Law parameters and thresholds are never changed here.

Kinematic ideal tracking says nothing about robot reachability, grasp, slip or
cameras. Oracle material state is evaluation-only, never an observation.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_phase_c4_trim_material.py \
      --task TRIM-02-K2-R05 --device cuda:0 --out artifacts/wp11/phaseC4/k2_r05_skim_law8_001
  ... --task TRIM-02-K2-R05 --control lift_mid --out artifacts/wp11/phaseC4/k2_r05_nc_lift_mid_law8_001
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.control import ActionLimits  # noqa: E402
from robotic_cutting_sim.demo_evaluation import trim_separation_for_task  # noqa: E402
from robotic_cutting_sim.policies import Waypoint, WaypointPolicy, skim  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402
from robotic_cutting_sim.skim_orientation import skim_orientation_report  # noqa: E402
from robotic_cutting_sim.tasks import V4_TASK_IDS  # noqa: E402
from robotic_cutting_sim.tool_profile import handle_box_edge  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_script('run_phase1_matrix')        # read-only reuse: swept planes, label distances, forces
bridges = _load_script('analyze_cut_bridges')     # read-only reuse: code-0 stencil bridges

SEED = 42
DEFAULT_LAW = 8
SIDE_FORCE_N, FLAT_FORCE_N, HANDLE_FORCE_N = .1, 1., .1   # run_phase1_matrix M5 force-present levels
FAR_MARGIN_M = .002          # far region starts two particle spacings beyond the furthest armed edge x
LEAD_BOUND_DX = 2            # law-5/8 lead region b_lead = 2 dx: no label may start further ahead than this
CONTROL_CRITERIA = {
    'skim': 'trim_separation_for_task PASS (>= 90% removed, substrate >= 99% of below-plane mass, detached '
            'below-plane <= 1%, one substrate component); frozen precision reported separately',
    'side_push': 'Armed spine-first push (-x 20 mm into the +x bar end face, edge at local z = 10 mm, task rake): mean '
                 '|F| > 0.1 N during the push, zero labels at every step, one final component',
    'flat_press': 'Armed flat press (rho = 0 task of the same bevel family, lowest blade face from 3 mm above to 5 mm '
                  'into the bar top beside the dome): mean |F| > 1 N during the press, zero labels, one component',
    'no_contact': 'Armed full-length skim with the lowest blade point 3 mm above the dome top: zero labels, zero '
                  'blade contacts during the skim, one component',
    'stop_mid': 'Armed skim to the dome centre (local x = 0), armed stop 0.5 s, armed resume to the end: labels '
                'unchanged during the stop; resumed outcome reported only',
    'lift_mid': 'Armed skim to the dome centre, then armed lift-out +z 30 mm: labels unchanged during the lift; '
                '>= 99% of the reference above-plane mass beyond the furthest armed edge x + 2 mm stays in the '
                'substrate; no labelled particle initially beyond that x + 2 dx',
    'half_length': 'Armed skim to the dome centre, armed stop, armed withdrawal backward along the path to the start, '
                   'then lift: labels unchanged during the stop and withdrawal; far-half attachment and lead bound '
                   'as lift_mid',
    'handle': 'Armed sweep of the blunt grip through the dome top (grip lowest point 3 mm below the dome top, blade '
              'beside the bar): grip obstacle mean |F| > 0.1 N during the sweep, zero labels, one component'}


def jsonable(value):
    return matrix.jsonable(value)


def _pose(position, rotation):
    return tuple(np.r_[position, rotation.as_quat()])


def _local(task, xyz):
    T = task.T_world_workpiece
    return T[:3, :3] @ np.asarray(xyz, float) + T[:3, 3]


def _rotation(task):
    return Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])


def lowest_blade_offset_m(task):
    """Lowest blade-section point relative to the edge, along world z, at the task orientation."""
    R = _rotation(task).as_matrix()
    section = task.tool_profile.cross_section_xz()
    points = np.c_[section[:, 0], np.zeros(len(section)), section[:, 1]] @ R.T
    return float(points[:, 2].min())


def _skim_points(task, **kwargs):
    plan = skim(task, **kwargs)
    return [np.asarray(w.pose[:3], float) for w in plan]


def skim_plan(task):
    return matrix.labelled(skim(task), ('approach', 'start', 'skim', 'retract'))


def _mid(task, start, end, local_x=0.):
    """Point on the start->end skim segment at the given workpiece-local x."""
    a, b = task.points_to_local(np.asarray([start, end]))
    s = (local_x - a[0]) / (b[0] - a[0])
    return start + s * (end - start)


def stop_mid_plan(task):
    above, start, end, _ = _skim_points(task)
    R, mid = _rotation(task), _mid(task, *_skim_points(task)[1:3])
    return [Waypoint(_pose(above, R), label='approach'), Waypoint(_pose(start, R), label='start'),
            Waypoint(_pose(mid, R), cut=True, speed_m_s=.025, label='skim_half'),
            Waypoint(_pose(mid, R), cut=True, hold_s=.5, label='hold'),
            Waypoint(_pose(end, R), cut=True, speed_m_s=.025, label='skim_resume'),
            Waypoint(_pose(end + [0., 0., .03], R), label='retract')]


def lift_mid_plan(task):
    above, start, end, _ = _skim_points(task)
    R, mid = _rotation(task), _mid(task, start, end)
    return [Waypoint(_pose(above, R), label='approach'), Waypoint(_pose(start, R), label='start'),
            Waypoint(_pose(mid, R), cut=True, speed_m_s=.025, label='skim_half'),
            Waypoint(_pose(mid + [0., 0., .03], R), cut=True, speed_m_s=.01, label='withdraw'),
            Waypoint(_pose(mid + [0., 0., .03], R), label='done_hold')]


def half_length_plan(task):
    above, start, end, _ = _skim_points(task)
    R, mid = _rotation(task), _mid(task, start, end)
    return [Waypoint(_pose(above, R), label='approach'), Waypoint(_pose(start, R), label='start'),
            Waypoint(_pose(mid, R), cut=True, speed_m_s=.025, label='skim_half'),
            Waypoint(_pose(mid, R), cut=True, hold_s=.2, label='hold'),
            Waypoint(_pose(start, R), cut=True, speed_m_s=.025, label='withdraw'),
            Waypoint(_pose(start + [0., 0., .03], R), cut=True, speed_m_s=.01, label='lift')]


def no_contact_plan(task):
    dome_top = max(task.target_origin_local_m[2], task.geometry.base_height_m + max(
        (b.height_m for b in task.geometry.bumps), default=0.))
    clearance = dome_top + .003 - task.target_origin_local_m[2] - lowest_blade_offset_m(task)
    return matrix.labelled(skim(task, clearance_m=clearance), ('approach', 'start', 'skim', 'retract'))


def side_push_plan(task):
    R = _rotation(task)
    spine = R.apply([-task.tool_profile.blade_depth_m, 0., 0.])
    end_face = task.geometry.length_m / 2
    begin = _local(task, [end_face - spine[0] + .008, 0., .010])
    end = begin - task.T_world_workpiece[:3, 0] * .020
    return [Waypoint(_pose(begin + [0., 0., .048], R), label='approach'),
            Waypoint(_pose(begin, R), label='lower_outside'),
            Waypoint(_pose(end, R), cut=True, speed_m_s=.01, label='side_push')]


def flat_press_plan(task):
    if task.blade_rake_deg != 0:
        raise ValueError('flat_press runs on the rho = 0 task of the bevel family')
    R = _rotation(task)
    low = lowest_blade_offset_m(task)
    top = task.geometry.base_height_m
    x = .065  # blade body over local x in [25, 65] mm, beside the dome (|x| <= 12 mm)
    start = _local(task, [x, 0., top + .003 - low])
    end = _local(task, [x, 0., top - .005 - low])
    return [Waypoint(_pose(start + [0., 0., .03], R), label='approach'), Waypoint(_pose(start, R), label='start'),
            Waypoint(_pose(end, R), cut=True, speed_m_s=.01, label='press')]


def handle_corners_world_offset(task):
    center, half = handle_box_edge(task.tool_profile)
    signs = np.stack(np.meshgrid(*[[-1., 1.]] * 3, indexing='ij'), -1).reshape(-1, 3)
    return _rotation(task).apply(center + signs * half)


def handle_plan(task):
    R = _rotation(task)
    corners = handle_corners_world_offset(task)          # world offsets of the grip corners from the edge
    dome = task.geometry.bumps[0]
    dome_top = task.geometry.base_height_m + dome.height_m
    y0 = -corners[:, 1].mean()                            # grip centred across the bar
    z0 = dome_top - .003 - corners[:, 2].min()            # grip lowest point 3 mm below the dome top
    x_start = dome.x_m - dome.radius_x_m - .006 - corners[:, 0].max()
    x_end = dome.x_m + dome.radius_x_m + .006 - corners[:, 0].min()
    blade_low_y = y0 - task.tool_profile.half_width_m
    if blade_low_y <= task.geometry.width_m / 2 + .005:
        raise ValueError('The blade would reach the bar during the grip sweep')
    begin, finish = _local(task, [x_start, y0, z0]), _local(task, [x_end, y0, z0])
    return [Waypoint(_pose(begin + [0., 0., .03], R), label='approach'), Waypoint(_pose(begin, R), label='lower'),
            Waypoint(_pose(finish, R), cut=True, speed_m_s=.025, label='handle_sweep'),
            Waypoint(_pose(finish + [0., 0., .03], R), label='retract')]


PLANS = {'skim': (skim_plan, None), 'side_push': (side_push_plan, 1e-5), 'flat_press': (flat_press_plan, 1e-5),
         'no_contact': (no_contact_plan, None), 'stop_mid': (stop_mid_plan, 1e-5), 'lift_mid': (lift_mid_plan, 1e-5),
         'half_length': (half_length_plan, 1e-5), 'handle': (handle_plan, 1e-5)}


# ----------------------------------------------------------------------------- episode

def run_episode(args, out):
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    env = make_kinematic_env(args.task, seed=SEED, device=args.device, particle_spacing_m=args.spacing,
                             material_dt=args.material_dt, cutting_law=args.cutting_law, max_cuts=1,
                             action_limits=ActionLimits(), ik_failure='reject', delta_frame='world',
                             history_stride=args.history_stride)
    observation, _ = env.reset(seed=SEED)
    builder, tolerance = PLANS[args.control]
    if args.position_tolerance is not None:  # opt-in override (supplementary runs); default keeps the plan's
        tolerance = args.position_tolerance
    plan = builder(env.task)
    policy = WaypointPolicy(env.task, plan, env.action_limits,
                            **({} if tolerance is None else {'position_tolerance_m': tolerance}))
    dt = env.task.control_dt_s
    bed = env.backend.material.bed
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
            boxes = bed.box_reactions.numpy().astype(float) / dt if hasattr(bed, 'box_reactions') else np.zeros((0, 6))
            plow = int(bed.segment_plow.numpy().sum()) if hasattr(bed, 'segment_plow') and args.cutting_law == 8 else None
            row = dict(step=step + 1, time_s=env.time_s, phase=phase, cut_enabled=bool(action['cut_enabled']),
                       cut_index=int(material['cut_index']), edge_pose_world=env.backend.edge.tolist(),
                       blade_force_world_N=blade[:3].tolist(), total_force_world_N=total[:3].tolist(),
                       handle_force_local_N=boxes[0, :3].tolist() if len(boxes) else None,
                       contact_statistics=np.asarray(material['material_contact_statistics']).tolist(),
                       stroke=stroke['stroke'], travel_m=stroke['travel_m'], engaged=stroke['engaged'],
                       any_motion=stroke['any_motion'], labelled_count=stroke['labelled_count'],
                       plowing_segments_last_substep=plow,
                       action={**action, 'target': np.asarray(action['target']).tolist()}, ik_status=info['ik_status'],
                       workspace_clamped=bool((info.get('action_report') or {}).get('workspace_clamped', False)))
            stream.write(json.dumps(jsonable(row)) + '\n')
            steps.append(row)
            if terminated or truncated:
                break
    material = env.backend.material
    recording = {'time_s': np.asarray([s['time_s'] for s in env._history], float),
                 **{k: np.asarray([s[k] for s in env._history]) for k in ('positions', 'velocities', 'codes')},
                 'mass_kg': material.masses, 'reference_positions': material.world_points,
                 'reference_indices': material.lattice, 'reference_stencil_nodes': bed.nodes.numpy(),
                 'reference_stencil_weights': bed.weights.numpy(), 'reference_stencil_gradients': bed.gradients.numpy(),
                 **env.backend.law4_stroke_arrays()}
    strict = env.evaluate()
    np.savez_compressed(out / 'physics_state.npz', **recording)
    context = {'task': env.task, 'policy_done': policy.done, 'law4_parameters': bed.law4_parameters,
               'law8_parameters': getattr(bed, 'law8_parameters', None), 'material_receipt': material.parameter_receipt,
               'runtime_source': env.runtime_source, 'blade_half_width_m': float(bed.config.blade_half_width),
               'plan': [dict(label=w.label, pose=list(w.pose), cut=w.cut, speed_m_s=w.speed_m_s, hold_s=w.hold_s)
                        for w in plan], 'position_tolerance_m': policy.position_tolerance_m}
    env.close()
    return steps, recording, strict, context


# ----------------------------------------------------------------------------- analysis

def _edge_at(steps, time_s):
    times = np.asarray([s['time_s'] for s in steps])
    index = int(np.clip(np.searchsorted(times, time_s - 1e-9), 0, len(steps) - 1))
    return np.asarray(steps[index]['edge_pose_world'], float), steps[index]


def _phase_window(steps, phases):
    rows = [s for s in steps if s['phase'] in phases]
    return (rows[0]['time_s'] - .02, rows[-1]['time_s']) if rows else None


def orientation_reports(task, steps):
    n_t = task.T_world_workpiece[:3, :3] @ np.asarray(task.target_normal_local, float)
    c = task.T_world_workpiece[:3, 0]
    out = {}
    phases = dict.fromkeys(s['phase'] for s in steps if s['cut_enabled'])
    for name, rows in [(f'armed_phase_{phase}', [s for s in steps if s['phase'] == phase and s['cut_enabled']])
                       for phase in phases]:
        if len(rows) >= 2:
            out[name] = skim_orientation_report([r['edge_pose_world'] for r in rows], n_t, c,
                                                source='commanded = measured (kinematic ideal tracking)')
    return out


def precision(strict):
    surface = (strict.get('metrics') or {}).get('surface') or {}
    keys = ('maximum_substrate_surface_residual_height_m', 'surface_p95_absolute_plane_error_m', 'surface_rms_plane_error_m',
            'maximum_gouge_depth_m', 'overcut_volume_fraction_of_initial_base', 'overcut_volume_m3', 'missing_surface_fraction')
    checks = {k: v for k, v in (strict.get('checks') or {}).items() if k != 'reference_above_plane_material_removed'}
    return {'surface': {k: surface.get(k) for k in keys}, 'frozen_checks': checks, 'frozen_status': strict.get('status'),
            'scope': 'Frozen TRIM-01 surface precision (reported separately from separation; not part of the outcome)'}


def chip_analysis(task, steps, data, labels, substrate_ids):
    """Where the above-plane material went (oracle-only; reference particle centres)."""
    reference = task.points_to_local(np.asarray(data['reference_positions'], float))
    target_z = task.target_origin_local_m[2]
    above = reference[:, 2] > target_z + 1e-9
    masses = np.asarray(data['mass_kg'], float)
    removed = above & ~np.isin(labels, substrate_ids)
    codes = np.asarray(data['codes'])
    final_codes = codes[-1]
    times = np.asarray(data['time_s'], float)
    depth = task.tool_profile.blade_depth_m * np.cos(np.deg2rad(task.blade_rake_deg or 0.))
    windows = {}
    for name in ('skim', 'skim_half', 'skim_resume', 'side_push', 'press', 'handle_sweep'):
        window = _phase_window(steps, (name,))
        if window:
            windows[name] = window
    armed_end = max((w[1] for w in windows.values()), default=times[-1])
    frame = int(np.flatnonzero(times <= armed_end + 1e-9)[-1])
    edge, _ = _edge_at(steps, times[frame])
    edge_local = task.points_to_local(edge[None, :3])[0]
    at = task.points_to_local(np.asarray(data['positions'][frame], float))
    final = task.points_to_local(np.asarray(data['positions'][-1], float))
    rel = at[:, 0] - edge_local[0]
    over = at[:, 2] > edge_local[2]

    def split(mask):
        total = float(masses[mask].sum())
        if total <= 0:
            return None
        part = lambda m: float(masses[mask & m].sum() / total)  # noqa: E731
        return {'particles': int(mask.sum()), 'mass_fraction_of_above_plane': total / float(masses[above].sum()),
                'at_armed_end': {'ahead_of_edge': part(rel > .0005),
                                 'over_blade': part((rel <= .0005) & (rel >= -depth - .001) & over),
                                 'under_or_beside_blade': part((rel <= .0005) & (rel >= -depth - .001) & ~over),
                                 'behind_blade': part(rel < -depth - .001)},
                'final': {'mean_local_xyz_m': (masses[mask] @ final[mask] / total).tolist(),
                          'lifted_above_target_plus_10mm': part(final[:, 2] > target_z + .010),
                          'off_bar_footprint': part((np.abs(final[:, 0]) > task.geometry.length_m / 2)
                                                    | (np.abs(final[:, 1]) > task.geometry.width_m / 2)),
                          'mean_displacement_m': (masses[mask] @ (final[mask] - reference[mask]) / total).tolist()}}

    timeline = []
    for index in range(0, len(times)):
        e, s = _edge_at(steps, times[index])
        ex = task.points_to_local(e[None, :3])[0, 0]
        x = task.points_to_local(np.asarray(data['positions'][index], float))
        zero = above & (codes[index] == 0)
        timeline.append({'time_s': float(times[index]), 'edge_local_x_m': float(ex), 'stroke': int(s['stroke']),
                         'above_plane_labelled': int(np.count_nonzero(above & (codes[index] != 0))),
                         'above_plane_code0_ahead_of_edge': int(np.count_nonzero(zero & (x[:, 0] > ex))),
                         'above_plane_mean_dx_m': float((x[above, 0] - reference[above, 0]).mean())})
    return {'armed_end_time_s': float(times[frame]), 'edge_local_at_armed_end_m': edge_local.tolist(),
            'removed_above_plane': split(removed), 'attached_above_plane': split(above & ~removed),
            'above_plane_code0_final': split(above & (final_codes == 0)),
            'timeline': timeline,
            'scope': 'Reference above-plane particle centres; ahead/over/behind are relative to the edge at the end '
                     'of the last armed phase; oracle-only'}


def code0_diagnosis(task, steps, data, law4):
    """Why above-plane particles stayed code 0: band geometry vs the stroke state (frame-sampled, approximate)."""
    reference = task.points_to_local(np.asarray(data['reference_positions'], float))
    above = reference[:, 2] > task.target_origin_local_m[2] + 1e-9
    codes = np.asarray(data['codes'])
    zero = np.flatnonzero(above & (codes[-1] == 0))
    if not len(zero):
        return {'above_plane_code0_final': 0}
    times = np.asarray(data['time_s'], float)
    band, rear = law4['band_h_m'], law4['rear_b_back_m']
    half = task.tool_profile.half_width_m
    ever_band_stroke = np.zeros(len(zero), bool)
    ever_band_no_stroke = np.zeros(len(zero), bool)
    ever_behind = np.zeros(len(zero), bool)
    min_f = np.full(len(zero), np.inf)
    s_at_min = np.zeros(len(zero))
    for index in range(len(times)):
        e, row = _edge_at(steps, times[index])
        R = Rotation.from_quat(e[3:]).as_matrix()
        p = np.asarray(data['positions'][index], float)[zero] - e[:3]
        f, l = p @ R[:, 0], p @ R[:, 1]
        s = p @ (task.T_world_workpiece[:3, 2])   # swept surface of a horizontal skim: world-up normal
        in_band = (f <= 0) & (f >= -rear) & (np.abs(l) <= half) & (np.abs(s) <= band)
        if row['cut_enabled']:
            ever_behind |= f <= 0
            if row['stroke']:
                ever_band_stroke |= in_band
            else:
                ever_band_no_stroke |= in_band
            better = f < min_f
            min_f[better], s_at_min[better] = f[better], s[better]
    classes = {'never_behind_edge_while_armed': int(np.count_nonzero(~ever_behind)),
               'in_band_geometry_only_without_stroke': int(np.count_nonzero(~ever_band_stroke & ever_band_no_stroke)),
               'in_band_geometry_with_stroke_frame_sampled': int(np.count_nonzero(ever_band_stroke)),
               'behind_edge_but_outside_band': int(np.count_nonzero(ever_behind & ~ever_band_stroke & ~ever_band_no_stroke))}
    initial = reference[zero]
    return {'above_plane_code0_final': int(len(zero)), 'classes': classes,
            'initial_local_x_mm_histogram': np.histogram(initial[:, 0] * 1e3, bins=np.arange(-14, 15, 2))[0].tolist(),
            'initial_local_x_bins_mm': np.arange(-14, 15, 2).tolist(),
            'initial_local_z_mm_counts': {f'{z:.1f}': int(c) for z, c in zip(*np.unique(np.round(initial[:, 2] * 1e3, 1),
                                                                                       return_counts=True))},
            'min_f_relative_to_edge_mm': {'percentiles': np.percentile(min_f[np.isfinite(min_f)] * 1e3,
                                                                       [0, 10, 50, 90, 100]).tolist()
                                          if np.isfinite(min_f).any() else None},
            's_at_min_f_mm_percentiles': np.percentile(s_at_min * 1e3, [0, 10, 50, 90, 100]).tolist(),
            'band_parameters_m': {'band_h': band, 'rear_b_back': rear, 'edge_half_width': half},
            'scope': 'History frames every history_stride control steps; band membership uses current positions in '
                     'the edge frame with the horizontal swept surface, approximating _assign_band (oracle-only)'}


def control_checks(control, task, steps, data, labels, separation, law4):
    final_labelled = int(np.count_nonzero(np.asarray(data['codes'])[-1]))
    count = int(separation['metrics']['mechanical_components'])
    max_labelled = max(s['labelled_count'] for s in steps)
    metrics, checks = {}, {}
    if control in ('side_push', 'flat_press'):
        phase, level = ('side_push', SIDE_FORCE_N) if control == 'side_push' else ('press', FLAT_FORCE_N)
        metrics['force'] = matrix.force_summary(steps, phase)
        metrics['phase_label_change'] = matrix.phase_label_changes(steps, (phase,))
        checks = {'contact_force_present': metrics['force'].get('total_force_mean_N', 0.) > level,
                  'zero_labels': final_labelled == 0 and max_labelled == 0, 'one_component': count == 1}
        metrics['force_present_level_N'] = level
    elif control == 'no_contact':
        metrics['force'] = matrix.force_summary(steps, 'skim')
        checks = {'zero_labels': final_labelled == 0 and max_labelled == 0,
                  'no_blade_contacts_during_skim': metrics['force'].get('steps_with_blade_contacts') == 0,
                  'one_component': count == 1}
    elif control == 'handle':
        rows = [s for s in steps if s['phase'] == 'handle_sweep']
        grip = np.linalg.norm([s['handle_force_local_N'] for s in rows], axis=1) if rows else np.zeros(1)
        metrics['handle_force'] = {'mean_N': float(grip.mean()), 'max_N': float(grip.max()), 'steps': len(rows),
                                   'steps_with_grip_force': int(np.count_nonzero(grip > 0))}
        metrics['force'] = matrix.force_summary(steps, 'handle_sweep')
        checks = {'grip_contact_force_present': metrics['handle_force']['mean_N'] > HANDLE_FORCE_N,
                  'zero_labels': final_labelled == 0 and max_labelled == 0, 'one_component': count == 1}
    elif control == 'stop_mid':
        metrics['hold_label_change'] = matrix.phase_label_changes(steps, ('hold',))
        checks = {'labels_unchanged_during_stop': metrics['hold_label_change'].get('change') == 0}
        metrics['resumed_outcome_report_only'] = separation['status']
    elif control in ('lift_mid', 'half_length'):
        quiet = ('withdraw',) if control == 'lift_mid' else ('hold', 'withdraw', 'lift')
        metrics.update({f'{p}_label_change': matrix.phase_label_changes(steps, (p,)) for p in quiet})
        armed_x = [task.points_to_local(np.asarray([s['edge_pose_world'][:3]]))[0, 0] for s in steps
                   if s['phase'] == 'skim_half']
        x_stop = float(max(armed_x))
        reference = task.points_to_local(np.asarray(data['reference_positions'], float))
        masses = np.asarray(data['mass_kg'], float)
        above = reference[:, 2] > task.target_origin_local_m[2] + 1e-9
        far = above & (reference[:, 0] > x_stop + FAR_MARGIN_M)
        substrate = np.isin(labels, separation['metrics']['reference_substrate_component_ids'])
        codes = np.asarray(data['codes'])[-1]
        labelled_x = reference[codes != 0, 0]
        lead = LEAD_BOUND_DX * task.grid_dx_m
        metrics.update(furthest_armed_edge_local_x_m=x_stop, far_region_particles=int(far.sum()),
                       far_region_attached_mass_fraction=float(masses[far & substrate].sum() / masses[far].sum())
                       if far.any() else None,
                       far_region_labelled_particles=int(np.count_nonzero(far & (codes != 0))),
                       max_labelled_initial_local_x_m=float(labelled_x.max()) if len(labelled_x) else None,
                       lead_bound_m=lead)
        checks = {f'labels_unchanged_during_{p}': metrics[f'{p}_label_change'].get('change') == 0 for p in quiet}
        checks['far_region_stays_attached_ge_99pct'] = (metrics['far_region_attached_mass_fraction'] or 0.) >= .99
        checks['no_label_beyond_lead_bound'] = not len(labelled_x) or labelled_x.max() <= x_stop + lead
    return checks, metrics


def evaluate(args, task, steps, data, strict, context, npz):
    count, labels = matrix.components(data, np.asarray(data['codes'])[-1], 1)
    separation = trim_separation_for_task(task, data)
    planes = matrix.swept_planes(steps, 1, context['blade_half_width_m'])
    metrics = {'trim_separation': separation, 'precision': precision(strict),
               'final_mechanical_components': int(count),
               'final_component_mass_fractions': matrix.mass_fractions(data, labels, count)[:6].tolist(),
               'final_labelled_count': int(np.count_nonzero(np.asarray(data['codes'])[-1])),
               'swept_planes': planes, 'label_to_swept_surface': matrix.plane_reports(data, planes, 1),
               'orientation_metrics': orientation_reports(task, steps),
               'lowest_blade_point_relative_to_edge_m': lowest_blade_offset_m(task),
               'forces': {phase: matrix.force_summary(steps, phase) for phase in
                          dict.fromkeys(s['phase'] for s in steps if s['cut_enabled'])},
               'stroke': {'armed_steps': sum(s['cut_enabled'] for s in steps),
                          'stroke_active_steps': sum(s['stroke'] != 0 for s in steps),
                          'engaged_steps': sum(bool(s['engaged']) for s in steps),
                          'plowing_steps_last_substep': (sum(bool(s['plowing_segments_last_substep']) for s in steps)
                                                         if steps[0]['plowing_segments_last_substep'] is not None else None)},
               'component_count_history': matrix.component_history(data, 1),
               'history_time_s': np.asarray(data['time_s']).tolist()}
    try:
        report = bridges.analyze(npz, task.task_id)
        metrics['bridges'] = {k: report[k] for k in report if k not in ('source',)}
    except ValueError as error:  # e.g. a non-formal particle spacing (the analyzer samples the protocol spacing)
        metrics['bridges'] = {'status': 'NOT_RUN', 'reason': str(error)}
    metrics['chip'] = chip_analysis(task, steps, data, labels, separation['metrics']['reference_substrate_component_ids'])
    metrics['code0_diagnosis'] = code0_diagnosis(task, steps, data, context['law4_parameters'])
    if args.control == 'skim':
        checks = dict(separation['checks'])
    else:
        checks, extra = control_checks(args.control, task, steps, data, labels, separation, context['law4_parameters'])
        metrics['control'] = extra
    verdict = 'PASS' if checks and all(checks.values()) else 'FAIL'
    return verdict, {k: bool(v) for k, v in checks.items()}, metrics


def git_state():
    state = matrix.git_state()
    state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state['reused_scripts_sha256'] = {name: hashlib.sha256((ROOT / 'scripts' / f'{name}.py').read_bytes()).hexdigest()
                                      for name in ('run_phase1_matrix', 'analyze_cut_bridges')}
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=[t for t in V4_TASK_IDS if t.startswith('TRIM-02')])
    p.add_argument('--control', default='skim', choices=sorted(PLANS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cutting-law', type=int, default=DEFAULT_LAW, choices=(4, 5, 6, 7, 8))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=.001, help='Particle spacing (formal: 1 mm)')
    p.add_argument('--material-dt', type=float, default=1e-4, help='Material dt (formal: 1e-4 s)')
    p.add_argument('--history-stride', type=int, default=5)
    p.add_argument('--steps', type=int, help='Validation only: cap control steps (not formal)')
    p.add_argument('--position-tolerance', type=float,
                   help='Override the waypoint position tolerance (default: the plan default; the skim keeps the '
                        'WaypointPolicy 0.5 mm, so the armed edge starts up to 0.5 mm above target + clearance)')
    a = p.parse_args()
    if f'law{a.cutting_law}' not in a.out.name:
        p.error(f'--out name must contain law{a.cutting_law} so runs of different laws never mix')
    out = new_output(a.out)  # refuses to overwrite
    start = time.monotonic()
    config = {**vars(a), 'out': str(a.out), 'seed': SEED}
    write_json(out / 'manifest.json', manifest(config, f'wp11_phaseC4_law{a.cutting_law}'))
    formal = a.spacing == .001 and a.material_dt == 1e-4 and a.steps is None
    base = {'task_id': a.task, 'control': a.control, 'criterion': CONTROL_CRITERIA[a.control], 'command': sys.argv,
            **git_state(), 'seed': SEED, 'cutting_law': a.cutting_law, 'particle_spacing_m': a.spacing,
            'material_dt_s': a.material_dt, 'grid_dx_m': .004, 'history_stride': a.history_stride, 'device': a.device,
            'formal_resolution': formal, 'runtime_source': 'kinematic_ideal_tracking',
            'position_tolerance_override_m': a.position_tolerance}
    try:
        steps, data, strict, context = run_episode(a, out)
        task = context.pop('task')
        verdict, checks, metrics = evaluate(a, task, steps, data, strict, context, out / 'physics_state.npz')
        if not formal and verdict == 'PASS':
            verdict = 'PASS_NONFORMAL'
        write_json(out / 'result.json', jsonable({
            **base, 'verdict': verdict, 'checks': checks, 'metrics': metrics, 'frozen_strict': strict,
            'task': task.to_dict(), 'context': context, 'steps_run': len(steps),
            'wall_s': time.monotonic() - start}))
        print(a.task, a.control, verdict, checks, f'{time.monotonic() - start:.0f}s', flush=True)
    except Exception as error:
        write_json(out / 'result.json', jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                  'wall_s': time.monotonic() - start}))
        raise


if __name__ == '__main__':
    main()
