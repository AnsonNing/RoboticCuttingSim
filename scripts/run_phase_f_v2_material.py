#!/usr/bin/env python3
"""WP-11 Phase F v2 follow-up: corrected blunt-handle negative control (protocol wp11_phaseF_v2).

Implements the single row ``F2.9_blunt_handle_v2`` of ``configs/wp11/phaseF_protocol_v2.json``.
The v1 runner ``scripts/run_phase_f_material.py`` is loaded read-only as a private module
instance (episode loop, common metrics, git state); its file and every phaseF_v1 row are
unchanged, and the v1 F2.9_blunt_handle FAIL record stands. The v2 grip (handle box of the
same ToolProfile, the obstacle box the v1 runner already uses) presses -z onto the bar top
over the right pedestal with the blade extending upward, so the pedestal reacts the press.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_phase_f_v2_material.py --device cuda:0 \
      --out artifacts/wp11/phaseF/f2_9_blunt_handle_v2_law8_001
"""
import argparse
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
from robotic_cutting_sim.policies import Waypoint  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402
from robotic_cutting_sim.tool_profile import handle_box_edge  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f'{name}_for_phase_f_v2', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _load_script('run_phase_f_material')   # read-only reuse of the frozen phaseF_v1 runner
L = F.L

PROTOCOL_V2_PATH = ROOT / 'configs' / 'wp11' / 'phaseF_protocol_v2.json'
PROTOCOL_V2 = json.loads(PROTOCOL_V2_PATH.read_text())
ROW = 'F2.9_blunt_handle_v2'
KIND = 'blunt_handle_v2'
SPEC = dict(task='CUT-03-K2', kind=KIND)
PARAMS = PROTOCOL_V2['rows'][ROW]['parameters']
PRESS_PHASES = ('handle_press', 'handle_hold')
HANDLE_FORCE_N = .1   # unchanged from v1 (F.HANDLE_FORCE_N)
INTERPRETATION = (
    'CUT-03-K2; edge frame +x -> local +x, edge line +y -> local +z, blade normal +z -> local -y (blade extends '
    'upward from the grip); grip = tool_profile.handle_box_edge far (-y edge) face, 24 x 22 mm, centred at local '
    '(55, 0) mm over the right pedestal; lift and reorient disarmed at a 15 mm grip gap, lower to a 3 mm gap, armed '
    '-z press at 10 mm/s ending 2 mm into the bar top, armed hold 0.3 s, disarmed retract (WaypointPolicy 0.01 mm); '
    'checks: zero labels on every step; one component; mean |grip obstacle force| over handle_press/handle_hold steps '
    'with grip contact > 0.1 N; mass check; control validity: zero blade contact steps over the episode')


# ----------------------------------------------------------------------------- geometry and plan

def press_rotation(task):
    """Edge-frame orientation of the press: columns = local images of the edge x, y, z axes."""
    axes = PARAMS['edge_axes_local']
    local = np.column_stack([axes['advance_x'], axes['edge_line_y'], axes['normal_z']]).astype(float)
    if not np.allclose(local.T @ local, np.eye(3)) or not np.isclose(np.linalg.det(local), 1.):
        raise ValueError('edge_axes_local is not a right-handed orthonormal frame')
    return Rotation.from_matrix(task.T_world_workpiece[:3, :3] @ local)


def _box_corners(center, half):
    signs = np.stack(np.meshgrid(*[[-1., 1.]] * 3, indexing='ij'), -1).reshape(-1, 3)
    return np.asarray(center, float) + signs * np.asarray(half, float)


def grip_corners_local(task, edge_pose):
    """Handle-box corners in workpiece local coordinates at a world edge pose."""
    center, half = handle_box_edge(task.tool_profile)
    pose = np.asarray(edge_pose, float)
    world = Rotation.from_quat(pose[3:]).apply(_box_corners(center, half)) + pose[:3]
    return task.points_to_local(world)


def grip_edge_position(task, R, gap_m):
    """World edge position putting the grip's lowest face ``gap_m`` above the bar top (negative: into it)."""
    corners = R.apply(_box_corners(*handle_box_edge(task.tool_profile)))   # world offsets from the edge
    T = task.T_world_workpiece
    local_offsets = corners @ T[:3, :3]                                   # (rotation only) local offsets
    cx, cy = PARAMS['grip_footprint_center_local_xy_m']
    centre_xy = (local_offsets[:, :2].min(0) + local_offsets[:, :2].max(0)) / 2
    z = task.geometry.base_height_m + gap_m - local_offsets[:, 2].min()
    return F._local(task, [cx - centre_xy[0], cy - centre_xy[1], z])


def blunt_handle_v2_plan(task):
    R0 = F._rotation(task)
    R = press_rotation(task)
    initial = np.asarray(task.reference_edge_pose_world(0.)[:3], float)
    travel = grip_edge_position(task, R, PARAMS['travel_grip_gap_m'])
    start = grip_edge_position(task, R, PARAMS['start_grip_gap_m'])
    end = grip_edge_position(task, R, -PARAMS['press_depth_m'])
    lift = initial.copy()
    lift[2] = travel[2]
    return [Waypoint(F._pose(lift, R0), label='lift'), Waypoint(F._pose(travel, R), label='reorient'),
            Waypoint(F._pose(start, R), label='lower_outside'),
            Waypoint(F._pose(end, R), cut=True, speed_m_s=PARAMS['press_speed_m_s'], label='handle_press'),
            Waypoint(F._pose(end, R), cut=True, hold_s=PARAMS['hold_s'], label='handle_hold'),
            Waypoint(F._pose(travel, R), label='retract')]


F.WAYPOINT_PLANS[KIND] = (blunt_handle_v2_plan, PARAMS['position_tolerance_m'])   # this process's module instance only


def bar_box_local(task):
    g = task.geometry
    return np.array([-g.length_m / 2, -g.width_m / 2, 0.]), np.array([g.length_m / 2, g.width_m / 2, g.base_height_m])


def _box_distance(points, low, high):
    """Signed distance of points to an axis-aligned box (negative inside)."""
    centre, half = (low + high) / 2, (high - low) / 2
    q = np.abs(points - centre) - half
    outside = np.linalg.norm(np.maximum(q, 0.), axis=-1)
    return outside + np.minimum(q.max(-1), 0.)


def blade_samples_edge(profile, step_m=5e-4):
    """Points on the blade prism surface (edge frame): the cross-section boundary densified, times the edge line."""
    section = profile.cross_section_xz()
    ring = []
    for a, b in zip(section, np.roll(section, -1, 0)):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step_m)), 1)
        ring.append(a + (b - a) * np.linspace(0., 1., n, endpoint=False)[:, None])
    ring = np.concatenate(ring)
    ys = np.linspace(-profile.half_width_m, profile.half_width_m, int(np.ceil(2 * profile.half_width_m / step_m)) + 1)
    return np.concatenate([np.c_[ring[:, 0], np.full(len(ring), y), ring[:, 1]] for y in ys])


def grip_samples_edge(profile, step_m=5e-4):
    center, half = handle_box_edge(profile)
    axes = [np.linspace(-h, h, max(int(np.ceil(2 * h / step_m)), 1) + 1) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    on_surface = np.isclose(np.abs(grid), half).any(1)
    return center + grid[on_surface]


def tool_clearance(task, poses, phases):
    """Minimum distance of blade and grip surface samples to the reference bar box, per phase."""
    low, high = bar_box_local(task)
    blade, grip = blade_samples_edge(task.tool_profile), grip_samples_edge(task.tool_profile)
    out = {}
    for pose, phase in zip(np.asarray(poses, float), phases):
        R = Rotation.from_quat(pose[3:])
        d_blade = float(_box_distance(task.points_to_local(R.apply(blade) + pose[:3]), low, high).min())
        d_grip = float(_box_distance(task.points_to_local(R.apply(grip) + pose[:3]), low, high).min())
        entry = out.setdefault(phase, {'steps': 0, 'blade_min_m': np.inf, 'grip_min_m': np.inf})
        entry['steps'] += 1
        entry['blade_min_m'] = min(entry['blade_min_m'], d_blade)
        entry['grip_min_m'] = min(entry['grip_min_m'], d_grip)
    return out


def footprint_report(task, edge_pose):
    """Grip footprint (local xy) at the press pose vs the right pedestal and the bar."""
    corners = grip_corners_local(task, edge_pose)
    lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
    pedestal = next(b for b in task.fixture_boxes_local if b.name == 'right_pedestal')
    pc, ph = np.asarray(pedestal.center_m, float), np.asarray(pedestal.half_extents_m, float)
    blow, bhigh = bar_box_local(task)
    return {'footprint_local_x_m': [float(lo[0]), float(hi[0])], 'footprint_local_y_m': [float(lo[1]), float(hi[1])],
            'footprint_area_m2': float(np.prod(hi - lo)), 'grip_bottom_local_z_m': float(corners[:, 2].min()),
            'right_pedestal_local_x_m': [float(pc[0] - ph[0]), float(pc[0] + ph[0])],
            'right_pedestal_local_y_m': [float(pc[1] - ph[1]), float(pc[1] + ph[1])],
            'inside_pedestal_span': bool(lo[0] >= pc[0] - ph[0] and hi[0] <= pc[0] + ph[0]
                                         and lo[1] >= pc[1] - ph[1] and hi[1] <= pc[1] + ph[1]),
            'inside_bar_top': bool(np.all(lo >= blow[:2]) and np.all(hi <= bhigh[:2]))}


# ----------------------------------------------------------------------------- workpiece motion

def rigid_fit(reference, current, weights):
    """Mass-weighted best rigid transform reference -> current (Kabsch)."""
    w = np.asarray(weights, float) / np.sum(weights)
    a, b = np.asarray(reference, float), np.asarray(current, float)
    ca, cb = w @ a, w @ b
    H = (a - ca).T @ ((b - cb) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1., 1., np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    residual = b - ((a - ca) @ Rm.T + cb)
    rotvec = Rotation.from_matrix(Rm).as_rotvec()
    return {'com_displacement_m': (cb - ca).tolist(), 'com_displacement_norm_m': float(np.linalg.norm(cb - ca)),
            'rotation_deg': float(np.rad2deg(np.linalg.norm(rotvec))), 'yaw_deg': float(np.rad2deg(rotvec[2])),
            'residual_rms_m': float(np.sqrt(w @ np.sum(residual ** 2, 1))),
            'residual_max_m': float(np.linalg.norm(residual, axis=1).max())}


def workpiece_motion(task, positions, baseline, masses, footprint, spacing):
    """Rigid displacement vs compression under the grip, between ``baseline`` and ``positions`` (world)."""
    ref_local = task.points_to_local(np.asarray(baseline, float))
    cur_local = task.points_to_local(np.asarray(positions, float))
    (x0, x1), (y0, y1) = footprint['footprint_local_x_m'], footprint['footprint_local_y_m']
    under = (ref_local[:, 0] >= x0) & (ref_local[:, 0] <= x1) & (ref_local[:, 1] >= y0) & (ref_local[:, 1] <= y1)
    top_z, bottom_z = ref_local[under, 2].max(), ref_local[under, 2].min()
    top = under & (ref_local[:, 2] >= top_z - spacing / 2)
    bottom = under & (ref_local[:, 2] <= bottom_z + spacing / 2)
    dz = cur_local[:, 2] - ref_local[:, 2]
    fit = rigid_fit(ref_local, cur_local, masses)
    far = ref_local[:, 0] < 0.
    compression = float(dz[bottom].mean() - dz[top].mean())
    return {'rigid_fit_all_particles': fit,
            'under_footprint': {'particles': int(under.sum()), 'top_layer': int(top.sum()), 'bottom_layer': int(bottom.sum()),
                                'top_mean_dz_m': float(dz[top].mean()), 'top_min_dz_m': float(dz[top].min()),
                                'bottom_mean_dz_m': float(dz[bottom].mean()),
                                'column_height_m': float(top_z - bottom_z), 'compression_m': compression,
                                'compression_strain': compression / float(top_z - bottom_z)},
            'left_half_max_displacement_m': float(np.linalg.norm(cur_local[far] - ref_local[far], axis=1).max()),
            'compression_over_rigid_com_displacement': (compression / fit['com_displacement_norm_m']
                                                        if fit['com_displacement_norm_m'] > 0 else None)}


def _frame_at_or_before(times, t):
    candidates = np.flatnonzero(np.asarray(times) <= t + 1e-9)
    return int(candidates[-1]) if len(candidates) else 0


# ----------------------------------------------------------------------------- evaluation

def _report_only(fn, *args):
    """A report-only metric; an analysis error is recorded instead of losing the row's criteria."""
    try:
        return fn(*args)
    except Exception as error:   # noqa: BLE001 - recorded in result.json
        return {'error': repr(error)}


def _motion_report(task, steps, press, data, footprint, spacing):
    times = np.asarray(data['time_s'], float)
    positions = np.asarray(data['positions'])
    masses = np.asarray(data['mass_kg'], float)
    first_press = min(s['time_s'] for s in press) - task.control_dt_s
    hold_end = max(s['time_s'] for s in press)
    base_i, hold_i = _frame_at_or_before(times, first_press), _frame_at_or_before(times, hold_end)
    return {'baseline_frame_time_s': float(times[base_i]), 'hold_end_frame_time_s': float(times[hold_i]),
            'final_frame_time_s': float(times[-1]),
            'hold_end_vs_pre_press': workpiece_motion(task, positions[hold_i], positions[base_i], masses, footprint, spacing),
            'hold_end_vs_reference': workpiece_motion(task, positions[hold_i], data['reference_positions'], masses,
                                                      footprint, spacing),
            'final_vs_pre_press': workpiece_motion(task, positions[-1], positions[base_i], masses, footprint, spacing),
            'scope': 'history frames every history_stride control steps; local coordinates; compression = bottom minus '
                     'top layer mean z displacement under the grip footprint (positive = compressed)'}


def evaluate_v2(task, steps, data, records, evaluation, context):
    metrics, labels = F.common_metrics(task, steps, data, records, evaluation, context)
    count = metrics['final_mechanical_components']
    zero = metrics['final_labelled_count'] == 0 and metrics['max_labelled_count'] == 0
    window = L.contact_window_force(steps, PRESS_PHASES, use='grip', contact='grip')
    metrics['contact_window'] = window
    press = [s for s in steps if s['phase'] in PRESS_PHASES]
    armed = [s for s in steps if s['cut_enabled']]
    metrics['blade_contact_steps'] = {
        'episode': int(sum(s['contact_statistics'][1] > 0 for s in steps)),
        'armed': int(sum(s['contact_statistics'][1] > 0 for s in armed)),
        'press_hold': int(sum(s['contact_statistics'][1] > 0 for s in press)),
        'episode_blade_force_max_N': float(np.linalg.norm([s['blade_force_world_N'] for s in steps], axis=1).max())}
    grip = np.asarray([np.linalg.norm(s['handle_force_local_N']) if s['handle_force_local_N'] is not None else 0.
                       for s in steps])
    metrics['grip_force_by_phase'] = {
        phase: {'steps': len(idx), 'contact_steps': int((grip[idx] > 0).sum()), 'mean_N': float(grip[idx].mean()),
                'max_N': float(grip[idx].max()), 'last_N': float(grip[idx[-1]])}
        for phase in dict.fromkeys(s['phase'] for s in steps)
        for idx in [np.asarray([i for i, s in enumerate(steps) if s['phase'] == phase])]}
    metrics['grip_force_press_hold_series_N'] = [[float(s['time_s']), s['phase'], float(np.linalg.norm(s['handle_force_local_N'] or [0., 0., 0.])),
                                                 list(map(float, s['handle_force_local_N'] or [0., 0., 0.]))] for s in press]
    recorded = [s['edge_pose_world'] for s in press]
    # Footprint at the last recorded press/hold pose (the declared plan pose if the run never reached the press).
    press_pose = recorded[-1] if recorded else blunt_handle_v2_plan(task)[4].pose
    footprint = footprint_report(task, press_pose)
    metrics['footprint'] = {**footprint, 'pose_source': 'recorded' if recorded else 'declared_plan'}
    metrics['tool_clearance_recorded_poses'] = _report_only(
        tool_clearance, task, [s['edge_pose_world'] for s in steps], [s['phase'] for s in steps])
    metrics['workpiece_motion'] = _report_only(_motion_report, task, steps, press, data, footprint,
                                               context['particle_spacing_m']) if press else None
    prediction = PROTOCOL_V2['rows'][ROW]['prediction']
    lo, hi = prediction['grip_contact_window_mean_N']
    metrics['prediction_check_report_only'] = {
        'predicted_contact_window_mean_N': [lo, hi], 'measured_contact_window_mean_N': window['mean_N'],
        'within_predicted_range': bool(lo <= window['mean_N'] <= hi),
        'predicted_force_after_yield_onset_N': prediction['grip_force_after_yield_onset_N'],
        'measured_contact_window_max_N': window['max_N']}
    metrics['armed_stroke'] = L.stroke_summary(steps, tuple(dict.fromkeys(s['phase'] for s in armed)))
    metrics['armed_stroke'].pop('stroke_per_step', None)
    checks = {'zero_new_labels': zero, 'one_component': count == 1,
              'grip_contact_window_mean_force_gt_0_1N': window['mean_N'] > HANDLE_FORCE_N,
              'mass_conserved': bool(metrics['mass_check']['pass'])}
    validity = {'zero_blade_contact_steps_episode': metrics['blade_contact_steps']['episode'] == 0,
                'grip_footprint_inside_right_pedestal_span': footprint['inside_pedestal_span'],
                'grip_footprint_inside_bar_top': footprint['inside_bar_top']}
    if not context['stability']['stable']:
        metrics['stability_warning'] = 'NaN/inf or det(F) <= 0 occurred; see stability'
    return {k: bool(v) for k, v in checks.items()}, {k: bool(v) for k, v in validity.items()}, metrics


def verdict_of(checks, validity, formal):
    outcome = 'PASS' if checks and all(checks.values()) else 'FAIL'
    if outcome == 'PASS' and not all(validity.values()):
        outcome = 'INVALID_CONTROL'
    return outcome, (outcome if formal else f'{outcome}_NONFORMAL')


def git_state():
    state = F.git_state()   # runner_sha256 there is the v1 runner (reused read-only)
    state['v1_runner_sha256'] = state.pop('runner_sha256')
    state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state['v1_protocol_sha256'] = state.pop('protocol_sha256')
    state['protocol_sha256'] = L.sha256_file(PROTOCOL_V2_PATH)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, help='Validation only: cap control steps (verdict *_NONFORMAL)')
    a = p.parse_args()
    if 'law8' not in a.out.name or 'v2' not in a.out.name:
        p.error('--out name must contain law8 and v2')
    a.history_stride = PARAMS['history_stride']
    out = new_output(a.out)  # refuses to overwrite
    start = time.monotonic()
    config = {**vars(a), 'out': str(a.out), 'seed': F.SEED, 'row': ROW, 'variant': 'formal'}
    write_json(out / 'manifest.json', manifest(config, 'wp11_phaseF_v2_law8'))
    formal = a.steps is None
    row = PROTOCOL_V2['rows'][ROW]
    spacing, grid, dt = F.FORMAL
    base = {'row': ROW, 'variant': 'formal', 'protocol_id': PROTOCOL_V2['protocol_id'], 'protocol_row': ROW,
            'protocol_setup': row['setup'], 'protocol_pass': row['pass'], 'protocol_parameters': PARAMS,
            'force_estimate': row['force_estimate'], 'prediction': row['prediction'],
            'blade_clearance_check': row['blade_clearance_check'], 'interpretation': INTERPRETATION,
            'command': sys.argv, **git_state(), 'seed': F.SEED, 'cutting_law': F.LAW, 'solver': 'tlmpm_v4',
            'particle_spacing_m': spacing, 'grid_dx_m': grid, 'material_dt_s': dt, 'control_dt_s': .02,
            'history_stride': a.history_stride, 'device': a.device, 'formal_resolution': formal,
            'runtime_source': 'kinematic_ideal_tracking', 'task_id': SPEC['task'],
            'v1_record': 'artifacts/wp11/phaseF/f2_9_blunt_handle_law8_001 (FAIL, unchanged)'}
    try:
        steps, data, records, evaluation, context = F.run_episode(SPEC, 'formal', a, out)
        task = context.pop('task')
        checks, validity, metrics = evaluate_v2(task, steps, data, records, evaluation, context)
        outcome, verdict = verdict_of(checks, validity, formal)
        result = {**base, 'verdict': verdict, 'criterion_outcome': outcome, 'checks': checks,
                  'control_validity': validity, 'metrics': metrics,
                  'diagnosis': F.diagnose(SPEC, steps, data, metrics, context) if outcome != 'PASS' else None,
                  'frozen_strict': evaluation, 'task': task.to_dict(), 'context': context, 'steps_run': len(steps),
                  'wall_s': time.monotonic() - start}
        write_json(out / 'result.json', F.jsonable(result))
        print(ROW, verdict, checks, validity, f"mean {metrics['contact_window']['mean_N']:.4f} N",
              f'{time.monotonic() - start:.0f}s', flush=True)
    except Exception as error:
        write_json(out / 'result.json', F.jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                    'wall_s': time.monotonic() - start}))
        raise


if __name__ == '__main__':
    main()
