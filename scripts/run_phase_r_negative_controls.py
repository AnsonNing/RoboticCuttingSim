#!/usr/bin/env python3
"""WP-11 review negative controls (protocol ``configs/wp11/phaseR_negative_controls_v1.json``), one row per run.

Protocol (a), POST_LIVE: the Phase C4 negative-control set at the chosen rake rho = 10 deg
(TRIM-02-K2-R10, tlmpm_v4, law 8, K = 1). Protocol (b): the tlmpm_v5 negative controls
(packed codes, K = 5, every armed step writes digit 4) at formal resolution, each on the
task and plan of its tlmpm_v4 counterpart row.

The episode loop is ``scripts/run_phase_f_material.py``'s ``run_episode``. That runner, the
phaseF_v2 runner (``run_phase_f_v2_material.py``) and the C4 runner
(``run_phase_c4_trim_material.py``) are loaded read-only as private module instances. Their
files, rows and records are unchanged; this script only registers additional plans in its own
instance and selects the solver (tlmpm_v4 K = 1 or tlmpm_v5 K = 5). Protocol (b) rows are
scored by the counterpart's unchanged v4 analysis code on an exact single-digit view of the
packed history (digit 4 -> digit 0), and the view's exactness is itself checked natively.
Law parameters and thresholds are never changed; oracle state is evaluation-only.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_phase_r_negative_controls.py --row Ra_side_push \
      --device cuda:1 --out artifacts/wp11/phaseR/ra_side_push_law8_001
  ... --row Rb_withdrawal --out artifacts/wp11/phaseR/rb_withdrawal_law8_v5_001
  ... --row Rb_withdrawal --variant coarse --out artifacts/wp11/phaseR/smoke_coarse_rb_withdrawal_law8_v5_001
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.control import ActionLimits  # noqa: E402
from robotic_cutting_sim.cut_topology import decode_digits  # noqa: E402
from robotic_cutting_sim.demo_evaluation import trim_separation_for_task  # noqa: E402
from robotic_cutting_sim.policies import Waypoint  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f'{name}_for_phase_r', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_script('run_phase_f_v2_material')   # loads run_phase_f_material (and C4, matrix) read-only
F = V2.F
C4 = F.c4
M = F.matrix
L = F.L

PROTOCOL_PATH = ROOT / 'configs' / 'wp11' / 'phaseR_negative_controls_v1.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
P = PROTOCOL['protocols']
FLAT = P['a']['rows']['Ra_flat_press_normal']['parameters']
DIGIT_B = int(P['b']['armed_cut_index'])
K_B = int(P['b']['max_cuts'])

# row -> (protocol, task, plan kind, evaluation kind)
ROWS = {
    'Ra_no_contact': ('a', P['a']['task'], 'no_contact', 'no_contact_c4'),
    'Ra_side_push': ('a', P['a']['task'], 'c4_side_push', 'side_push'),
    'Ra_flat_press_normal': ('a', P['a']['task'], 'flat_press_normal', 'flat_press_normal'),
    'Ra_blunt_handle_v2': ('a', P['a']['task'], V2.KIND, 'blunt_handle_v2_floor'),
    'Ra_stop_mid': ('a', P['a']['task'], 'c4_stop_mid', 'c4:stop_mid'),
    'Ra_lift_mid': ('a', P['a']['task'], 'c4_lift_mid', 'c4:lift_mid'),
    'Ra_half_length': ('a', P['a']['task'], 'c4_half_length', 'c4:half_length'),
    'Rb_side_push': ('b', P['b']['rows']['Rb_side_push']['task'], 'side_push', 'side_push'),
    'Rb_flat_press': ('b', P['b']['rows']['Rb_flat_press']['task'], 'flat_press', 'flat_press'),
    'Rb_no_contact': ('b', P['b']['rows']['Rb_no_contact']['task'], 'no_contact', 'no_contact'),
    'Rb_stop_mid': ('b', P['b']['rows']['Rb_stop_mid']['task'], 'c4_stop_mid', 'c4:stop_mid'),
    'Rb_withdrawal': ('b', P['b']['rows']['Rb_withdrawal']['task'], 'withdrawal', 'withdrawal'),
    'Rb_half_depth_cut': ('b', P['b']['rows']['Rb_half_depth_cut']['task'], 'half_depth_cut', 'half_depth_cut'),
    'Rb_blunt_handle_v2': ('b', P['b']['rows']['Rb_blunt_handle_v2']['task'], V2.KIND, 'blunt_handle_v2'),
}
assert set(ROWS) == set(P['a']['rows']) | set(P['b']['rows']), 'runner rows must equal the declared rows'


# ----------------------------------------------------------------------------- plans (this module instance only)

def flat_press_normal_plan(task):
    """Armed press along the downward blade-face normal at the task rake (declared Ra_flat_press_normal)."""
    R = C4._rotation(task)
    n = R.apply([0., 0., 1.])
    down = -n if n[2] > 0 else n
    low = C4.lowest_blade_offset_m(task)
    top = task.geometry.base_height_m
    start = C4._local(task, [FLAT['edge_local_x_m'], 0., top + FLAT['start_gap_m'] - low])
    vertical = FLAT['start_gap_m'] + FLAT['end_depth_below_top_m']
    end = start + down * (vertical / abs(down[2]))
    return [Waypoint(C4._pose(start + [0., 0., .03], R), label='approach'), Waypoint(C4._pose(start, R), label='start'),
            Waypoint(C4._pose(end, R), cut=True, speed_m_s=FLAT['press_speed_m_s'], label='press'),
            Waypoint(C4._pose(end, R), cut=True, hold_s=FLAT['hold_s'], label='press_hold'),
            Waypoint(C4._pose(end + [0., 0., .03], R), label='retract')]


def half_depth_cut_plan(task):
    """matrix half_depth_plan geometry (armed descent to local z = 10 mm), then a DISARMED retract to the start."""
    plan = M.half_depth_plan(task, after=None)
    start = plan[1]
    return plan + [Waypoint(start.pose, speed_m_s=.01, label='retract')]


EXTRA_PLANS = {'c4_side_push': (C4.side_push_plan, 1e-5), 'c4_stop_mid': (C4.stop_mid_plan, 1e-5),
               'c4_lift_mid': (C4.lift_mid_plan, 1e-5), 'c4_half_length': (C4.half_length_plan, 1e-5),
               'flat_press_normal': (flat_press_normal_plan, FLAT['position_tolerance_m']),
               'half_depth_cut': (half_depth_cut_plan, 1e-5)}


def armed_digit(builder, digit):
    """The same plan with every armed waypoint writing ``digit`` (digit 0 keeps the plan unchanged)."""
    if digit == 0:
        return builder
    return lambda task: [replace(w, cut_index=digit) if w.cut else w for w in builder(task)]


def install_plans(digit):
    """Register the plans for this process in the private F instance (armed waypoints write ``digit``)."""
    base = {**F.WAYPOINT_PLANS, **EXTRA_PLANS}
    for kind, (builder, tolerance) in base.items():
        F.WAYPOINT_PLANS[kind] = (armed_digit(builder, digit), tolerance)


def make_env_for(protocol):
    """F.make_env with the protocol's solver and history size (tlmpm_v4 K = 1 or tlmpm_v5 K = 5)."""
    solver = {} if P[protocol]['solver'] == 'tlmpm_v4' else {'solver': 'tlmpm_v5', 'slots_per_node': P['b']['slots_per_node']}
    cuts = int(P[protocol]['max_cuts'])

    def make_env(spec, variant, args):
        from robotic_cutting_sim.kinematic_env import make_kinematic_env
        spacing, grid, dt = F.F3_VARIANTS[variant]
        common = dict(seed=F.SEED, device=args.device, material_dt=dt, delta_frame='world', ik_failure='reject',
                      history_stride=args.history_stride)
        task_id = spec['task']
        if task_id in L.ProfileTaskSpecV4F.TASK_IDS or grid != F.FORMAL[1]:
            task = L.make_phase_f_task(task_id, F.SEED, particle_spacing_m=spacing, grid_dx_m=grid)
            env = make_kinematic_env(task=task, cutting_law=F.LAW, max_cuts=cuts, action_limits=ActionLimits(),
                                     **solver, **common)
        else:
            env = make_kinematic_env(task_id, particle_spacing_m=spacing, cutting_law=F.LAW, max_cuts=cuts,
                                     action_limits=ActionLimits(), **solver, **common)
        return env, {'limits': ActionLimits()}
    return make_env


# ----------------------------------------------------------------------------- protocol (b): native checks and view

def native_packed(steps, data, records, context):
    codes = np.asarray(data['codes'])
    digits = decode_digits(codes.reshape(-1), K_B).reshape(*codes.shape, K_B)
    others = [k for k in range(K_B) if k != DIGIT_B]
    record_codes = np.concatenate([np.asarray(records['label_record_old']).reshape(-1),
                                   np.asarray(records['label_record_new']).reshape(-1)])
    record_digits = decode_digits(record_codes, K_B) if len(record_codes) else np.zeros((0, K_B), int)
    armed = [s for s in steps if s['cut_enabled']]
    strokes = sorted({int(s['stroke']) for s in steps})
    slots = context.get('slots') or {}
    packed_count, _ = F.final_components(data, context)
    checks = {
        'native_packed_encoding': bool(context['packed']) and int(context['max_cuts']) == K_B,
        'native_zero_slot_overflow': bool(slots) and slots.get('overflow_entries') == 0,
        'native_single_active_digit': bool(not digits[..., others].any() and not record_digits[:, others].any()
                                           and all(int(s['cut_index']) == DIGIT_B for s in armed)
                                           and all(int(s['cut_index']) == -1 for s in steps if not s['cut_enabled'])
                                           and set(strokes) <= {0, DIGIT_B + 1}),
        'native_digit_values_valid': int(digits.max(initial=0)) <= 2}
    metrics = {'max_cuts': int(context['max_cuts']), 'armed_digit': DIGIT_B, 'armed_steps': len(armed),
               'stroke_values': strokes, 'slots': slots, 'packed_graph_final_components': int(packed_count),
               'final_digit_labelled_counts': np.count_nonzero(digits[-1], axis=0).tolist(),
               'max_digit_labelled_counts_over_frames': np.count_nonzero(digits, axis=1).max(0).tolist(),
               'label_records': int(len(record_codes) // 2)}
    return checks, metrics, int(packed_count)


def single_digit_view(steps, data, records, context):
    """Exact digit-DIGIT_B -> digit-0 re-encoding (valid only with no other digit ever nonzero; checked natively)."""
    def digit(codes):
        codes = np.asarray(codes)
        return ((codes.astype(np.int64) >> (2 * DIGIT_B)) & 3).astype(codes.dtype) if codes.size else codes
    data_v = {**data, 'codes': digit(data['codes'])}
    records_v = {**records, 'label_record_old': digit(records['label_record_old']),
                 'label_record_new': digit(records['label_record_new'])}
    steps_v = []
    for s in steps:
        row = copy.copy(s)
        row['stroke'] = 1 if s['stroke'] == DIGIT_B + 1 else s['stroke']
        row['cut_index'] = 0 if s['cut_index'] == DIGIT_B else s['cut_index']
        steps_v.append(row)
    context_v = {**context, 'packed': False, 'max_cuts': 1}
    return steps_v, data_v, records_v, context_v


# ----------------------------------------------------------------------------- evaluation

def _c4_checks(control, task, steps, data, records, evaluation, context):
    metrics, _ = F.common_metrics(task, steps, data, records, evaluation, context)
    count, labels = M.components(data, np.asarray(data['codes'])[-1], 1)   # as C4 evaluate()
    separation = trim_separation_for_task(task, data)
    checks, extra = C4.control_checks(control, task, steps, data, labels, separation, context['law4_parameters'])
    metrics.update(control=extra, trim_separation_report_only=separation, c4_final_components=int(count),
                   forces_all_steps_c4={p: M.force_summary(steps, p) for p in
                                        dict.fromkeys(s['phase'] for s in steps if s['cut_enabled'])})
    checks['mass_conserved'] = bool(metrics['mass_check']['pass'])
    return checks, metrics


def _floor_footprint(task, edge_pose):
    corners = V2.grip_corners_local(task, edge_pose)
    lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
    floor = next(b for b in task.fixture_boxes_local if b.name == 'support_floor')
    fc, fh = np.asarray(floor.center_m, float), np.asarray(floor.half_extents_m, float)
    blow, bhigh = V2.bar_box_local(task)
    dome = task.geometry.bumps[0]
    return {'footprint_local_x_m': [float(lo[0]), float(hi[0])], 'footprint_local_y_m': [float(lo[1]), float(hi[1])],
            'footprint_area_m2': float(np.prod(hi - lo)), 'grip_bottom_local_z_m': float(corners[:, 2].min()),
            'support_floor_local_x_m': [float(fc[0] - fh[0]), float(fc[0] + fh[0])],
            'support_floor_local_y_m': [float(fc[1] - fh[1]), float(fc[1] + fh[1])],
            'inside_support_floor_span': bool(lo[0] >= fc[0] - fh[0] and hi[0] <= fc[0] + fh[0]
                                              and lo[1] >= fc[1] - fh[1] and hi[1] <= fc[1] + fh[1]),
            'inside_bar_top': bool(np.all(lo >= blow[:2]) and np.all(hi <= bhigh[:2])),
            'dome_x_m': [float(dome.x_m - dome.radius_x_m), float(dome.x_m + dome.radius_x_m)],
            'outside_dome_footprint': bool(lo[0] > dome.x_m + dome.radius_x_m or hi[0] < dome.x_m - dome.radius_x_m)}


def _handle_floor(task, steps, data, records, evaluation, context):
    """F2.9_blunt_handle_v2 criteria on the TRIM-02 bar (support floor instead of the CUT-03 right pedestal)."""
    metrics, _ = F.common_metrics(task, steps, data, records, evaluation, context)
    count = metrics['final_mechanical_components']
    zero = metrics['final_labelled_count'] == 0 and metrics['max_labelled_count'] == 0
    window = L.contact_window_force(steps, V2.PRESS_PHASES, use='grip', contact='grip')
    press = [s for s in steps if s['phase'] in V2.PRESS_PHASES]
    armed = [s for s in steps if s['cut_enabled']]
    metrics['contact_window'] = window
    metrics['blade_contact_steps'] = {
        'episode': int(sum(s['contact_statistics'][1] > 0 for s in steps)),
        'armed': int(sum(s['contact_statistics'][1] > 0 for s in armed)),
        'press_hold': int(sum(s['contact_statistics'][1] > 0 for s in press))}
    press_pose = press[-1]['edge_pose_world'] if press else V2.blunt_handle_v2_plan(task)[4].pose
    footprint = _floor_footprint(task, press_pose)
    metrics['footprint'] = {**footprint, 'pose_source': 'recorded' if press else 'declared_plan'}
    metrics['tool_clearance_recorded_poses'] = V2._report_only(
        V2.tool_clearance, task, [s['edge_pose_world'] for s in steps], [s['phase'] for s in steps])
    metrics['workpiece_motion'] = V2._report_only(V2._motion_report, task, steps, press, data, footprint,
                                                  context['particle_spacing_m']) if press else None
    metrics['armed_stroke'] = L.stroke_summary(steps, tuple(dict.fromkeys(s['phase'] for s in armed)))
    metrics['armed_stroke'].pop('stroke_per_step', None)
    checks = {'zero_new_labels': zero, 'one_component': count == 1,
              'grip_contact_window_mean_force_gt_0_1N': window['mean_N'] > V2.HANDLE_FORCE_N,
              'mass_conserved': bool(metrics['mass_check']['pass'])}
    validity = {'zero_blade_contact_steps_episode': metrics['blade_contact_steps']['episode'] == 0,
                'grip_footprint_inside_bar_top': footprint['inside_bar_top'],
                'grip_footprint_inside_support_floor_span': footprint['inside_support_floor_span'],
                'grip_footprint_outside_dome': footprint['outside_dome_footprint']}
    return checks, validity, metrics


def evaluate_row(row, task, steps, data, records, evaluation, context):
    protocol, _, _, kind = ROWS[row]
    native = None
    if protocol == 'b':
        native_checks, native_metrics, packed_count = native_packed(steps, data, records, context)
        steps, data, records, context = single_digit_view(steps, data, records, context)
        native = (native_checks, native_metrics, packed_count)
    validity = {}
    if kind in ('side_push', 'flat_press', 'no_contact', 'withdrawal'):
        checks, metrics = F.evaluate({'kind': kind}, 'formal', task, steps, data, records, evaluation, context)
    elif kind == 'no_contact_c4':
        checks, metrics = F.evaluate({'kind': 'no_contact'}, 'formal', task, steps, data, records, evaluation, context)
        metrics['c4_skim_force'] = M.force_summary(steps, 'skim')
        checks['no_blade_contacts_during_skim_c4'] = metrics['c4_skim_force'].get('steps_with_blade_contacts') == 0
        checks['one_component_c4'] = metrics['final_mechanical_components'] == 1
    elif kind == 'flat_press_normal':
        checks, metrics = F.evaluate({'kind': 'flat_press'}, 'formal', task, steps, data, records, evaluation, context)
        metrics['contact_window_press_only_report'] = metrics.pop('contact_window')
        metrics['contact_window'] = L.contact_window_force(steps, ('press', 'press_hold'), use='total', contact='blade')
        metrics['c4_all_steps_force_hold'] = M.force_summary(steps, 'press_hold')
        checks['contact_window_mean_force_gt_1N'] = metrics['contact_window']['mean_N'] > F.FLAT_FORCE_N
        metrics['prediction_report_only'] = P['a']['rows'][row]['prediction_report_only']
    elif kind == 'half_depth_cut':
        checks, metrics = F.evaluate({'kind': 'withdrawal'}, 'formal', task, steps, data, records, evaluation, context)
        checks.pop('labels_unchanged_during_withdrawal')
        metrics.pop('withdraw_label_change', None)
        metrics['retract_label_change'] = M.phase_label_changes(steps, ('retract',))
        metrics['retract_armed_steps'] = sum(s['cut_enabled'] for s in steps if s['phase'] == 'retract')
        checks['labels_unchanged_during_disarmed_retract'] = metrics['retract_label_change'].get('change') == 0
    elif kind.startswith('c4:'):
        checks, metrics = _c4_checks(kind[3:], task, steps, data, records, evaluation, context)
    elif kind == 'blunt_handle_v2':
        checks, validity, metrics = V2.evaluate_v2(task, steps, data, records, evaluation, context)
    elif kind == 'blunt_handle_v2_floor':
        checks, validity, metrics = _handle_floor(task, steps, data, records, evaluation, context)
    else:
        raise ValueError(kind)
    checks = {k: bool(v) for k, v in checks.items()}
    if native is not None:
        native_checks, native_metrics, packed_count = native
        native_checks['native_packed_graph_matches_view'] = packed_count == metrics['final_mechanical_components']
        checks.update({k: bool(v) for k, v in native_checks.items()})
        metrics['native_packed'] = {**native_metrics, 'scope': P['b']['evaluation']}
    return checks, {k: bool(v) for k, v in validity.items()}, metrics


def verdict_of(checks, validity, formal):
    outcome = 'PASS' if checks and all(checks.values()) else 'FAIL'
    if outcome == 'PASS' and validity and not all(validity.values()):
        outcome = 'INVALID_CONTROL'
    return outcome, (outcome if formal else f'{outcome}_NONFORMAL')


def git_state():
    state = F.git_state()
    state['phase_f_runner_sha256'] = state.pop('runner_sha256')
    state['phase_f_protocol_sha256'] = state.pop('protocol_sha256')
    state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state['protocol_sha256'] = L.sha256_file(PROTOCOL_PATH)
    state['reused_scripts_sha256']['run_phase_f_material'] = state['phase_f_runner_sha256']
    state['reused_scripts_sha256']['run_phase_f_v2_material'] = hashlib.sha256(
        (ROOT / 'scripts' / 'run_phase_f_v2_material.py').read_bytes()).hexdigest()
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', required=True, choices=sorted(ROWS))
    p.add_argument('--variant', default='formal', choices=('formal', 'coarse'),
                   help="'coarse' (2 mm, 8 mm, 0.2 ms) is a smoke validation only (verdict *_NONFORMAL)")
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, help='Validation only: cap control steps (verdict *_NONFORMAL)')
    a = p.parse_args()
    protocol, task_id, plan_kind, _ = ROWS[a.row]
    if 'law8' not in a.out.name or (protocol == 'b') != ('v5' in a.out.name):
        p.error('--out name must contain law8, and v5 exactly for protocol (b) rows')
    if a.variant != 'formal' and not a.out.name.startswith('smoke'):
        p.error("a non-formal variant must write to a 'smoke*' directory")
    a.history_stride = int(PROTOCOL['common']['history_stride'])
    install_plans(DIGIT_B if protocol == 'b' else int(P['a']['armed_cut_index']))
    F.make_env = make_env_for(protocol)          # this process's module instance only
    spec = {'task': task_id, 'kind': plan_kind}
    out = new_output(a.out)  # refuses to overwrite
    start = time.monotonic()
    spacing, grid, dt = F.F3_VARIANTS[a.variant]
    config = {**vars(a), 'out': str(a.out), 'seed': F.SEED}
    write_json(out / 'manifest.json', manifest(config, f'wp11_phaseR_{protocol}_law8' + ('_tlmpm_v5' if protocol == 'b' else '')))
    formal = a.variant == 'formal' and a.steps is None
    declared = P[protocol]['rows'][a.row]
    base = {'row': a.row, 'protocol': protocol, 'protocol_id': PROTOCOL['protocol_id'], 'protocol_title': P[protocol]['title'],
            'label': PROTOCOL['label'] if protocol == 'a' else None, 'declared_row': declared,
            'variant': a.variant, 'command': sys.argv, **git_state(), 'seed': F.SEED, 'cutting_law': F.LAW,
            'solver': P[protocol]['solver'], 'max_cuts': int(P[protocol]['max_cuts']),
            'armed_cut_index': int(P[protocol]['armed_cut_index']), 'particle_spacing_m': spacing, 'grid_dx_m': grid,
            'material_dt_s': dt, 'control_dt_s': .02, 'history_stride': a.history_stride, 'device': a.device,
            'formal_resolution': formal, 'runtime_source': 'kinematic_ideal_tracking', 'task_id': task_id,
            'plan_kind': plan_kind}
    try:
        steps, data, records, evaluation, context = F.run_episode(spec, a.variant, a, out)
        task = context.pop('task')
        checks, validity, metrics = evaluate_row(a.row, task, steps, data, records, evaluation, context)
        outcome, verdict = verdict_of(checks, validity, formal)
        result = {**base, 'verdict': verdict, 'criterion_outcome': outcome, 'checks': checks,
                  'control_validity': validity, 'metrics': metrics,
                  'diagnosis': F.diagnose(spec, steps, data, metrics, context) if outcome != 'PASS' else None,
                  'env_evaluate_report_only': {k: evaluation.get(k) for k in ('status', 'reason', 'success')},
                  'task': task.to_dict(), 'context': context, 'steps_run': len(steps),
                  'wall_s': time.monotonic() - start}
        write_json(out / 'result.json', F.jsonable(result))
        print(a.row, a.variant, verdict, checks, validity, f'{time.monotonic() - start:.0f}s', flush=True)
    except Exception as error:
        write_json(out / 'result.json', F.jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                    'wall_s': time.monotonic() - start}))
        raise


if __name__ == '__main__':
    main()
