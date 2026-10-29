#!/usr/bin/env python3
"""Collect Phase 1 matrix row results (scripts/run_phase1_matrix.py) into one evidence JSON.

Read-only over the row directories; the evidence file is new (never overwritten).

  .venv/bin/python scripts/summarize_phase1_matrix.py --law 4 \
      --rows artifacts/wp10_claude/phase1_law4/m1a_law4_ingest_001 ... --out docs/evidence/phase1/matrix_law4_001.json
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['M1a', 'M1b', 'M2', 'M3a', 'M3b', 'M3c', 'M4', 'M5a', 'M5b', 'M5c', 'M6', 'M7', 'M8a', 'M8b',
         'M9a', 'M9b', 'M9c', 'M9d']


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def key_numbers(result):
    """Row-specific headline numbers (all also present in the row's result.json)."""
    m, row = result.get('metrics', {}), result['row']
    if row == 'M7':
        return {s['row']: {k: s[k] for k in ('frozen_postcut_disconnected_topology_persists', 'postcut_persistence_time_s',
                                              'first_separated_frame_time_s', 'count_drop_times_s')} for s in m['sources']}
    plane = _get(m, 'swept_planes', '0') or {}
    report = _get(m, 'swept_plane_reports', '0') or {}
    numbers = {'final_components': m.get('final_mechanical_components'),
               'final_mass_fractions': [round(v, 4) for v in (m.get('final_component_mass_fractions') or [])[:4]],
               'final_labelled': m.get('final_labelled_count'),
               'frozen_strict_status': _get(m, 'frozen_strict', 'status'),
               'frozen_location_m': _get(m, 'frozen_strict', 'metrics', 'maximum_reference_midpoint_plane_distance_m'),
               'swept_plane_fit_rms_m': _get(plane, 'fit', 'rms_residual_m'),
               'swept_plane_fit_max_m': _get(plane, 'fit', 'max_residual_m'),
               'swept_plane_normal': _get(plane, 'fit', 'normal'),
               'swept_interface_m': _get(report, 'interface', 'maximum_reference_midpoint_plane_distance_m'),
               'ideal_partition_interface_floor_m': report.get('ideal_partition_interface_floor_m'),
               'label_distance_p50_p99_max_m': None if not _get(report, 'label_distance', 'percentiles_m') else
                   [report['label_distance']['percentiles_m'][p] for p in ('50', '99', '100')],
               'label_side_agreement': _get(report, 'label_distance', 'side_agreement_fraction')}
    if row.startswith('M3'):
        numbers.update(removed=_get(m, 'trim_separation', 'metrics', 'removed_reference_above_plane_mass_fraction'),
                       substrate_share=_get(m, 'trim_separation', 'metrics', 'largest_component_below_plane_mass_share'),
                       detached_below=m.get('detached_below_plane_mass_fraction'),
                       precision_checks=m.get('frozen_precision_checks'),
                       bridges=_get(result, 'bridge_diagnosis', 'report', 'direct_code_zero_bridges'),
                       above_plane_codes=_get(result, 'bridge_diagnosis', 'report', 'above_plane_code_counts'))
    for name in ('ligament', 'withdraw_label_change', 'hold_label_change', 'side_move_label_change', 'phase_label_change',
                 'lift_move_label_change', 'disabled_phase_label_change'):
        if name in m:
            value = m[name]
            numbers[name] = value if name == 'ligament' else {k: value.get(k) for k in ('steps', 'change', 'stroke_active_steps')}
    if 'force' in m:
        numbers['force'] = {k: m['force'].get(k) for k in ('total_force_mean_N', 'total_force_max_N', 'blade_force_mean_N',
                                                           'blade_force_max_N', 'steps_with_blade_contacts')}
    if 'multi_cut' in m:
        numbers['multi_cut'] = {'status': m['multi_cut']['status'], 'checks': m['multi_cut']['checks'],
                                'parts': [round(v, 4) for v in m['multi_cut']['metrics']['part_mass_fractions'][:4]],
                                'cuts': m['multi_cut']['metrics']['cuts']}
        numbers['digit1_swept_interface_m'] = _get(m, 'swept_plane_reports', '1', 'interface',
                                                   'maximum_reference_midpoint_plane_distance_m')
    if 'slice_push_report' in m:
        report = m['slice_push_report']
        numbers['slice_push'] = {'status': report['status'], 'multi_cut': _get(report, 'multi_cut', 'checks'),
                                 'post_cut_manipulation': _get(report, 'post_cut_manipulation', 'checks'),
                                 'pushed_displacement_m': _get(report, 'post_cut_manipulation', 'metrics',
                                                               'pushed_displacement_along_push_m'),
                                 'other_displacement_m': _get(report, 'post_cut_manipulation', 'metrics',
                                                              'other_piece_displacement_norm_m'),
                                 'parts': _get(report, 'multi_cut', 'metrics', 'part_mass_fractions')}
    return numbers


def supplement(directory, result):
    """Read-only report numbers added after a row ran (older runner versions): the ideal-partition
    interface floor per fitted plane and, for M8b, the contiguous disabled-phase label windows."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_phase1_matrix import ideal_partition_interface_floor, phase_label_changes
    extra, data = {}, None
    for digit, plane in (_get(result, 'metrics', 'swept_planes') or {}).items():
        report = _get(result, 'metrics', 'swept_plane_reports', digit) or {}
        if plane.get('status') != 'FIT' or 'ideal_partition_interface_floor_m' in report:
            continue
        if data is None:
            npz = directory / 'physics_state.npz'
            if not npz.exists():
                npz = Path(result.get('context', {}).get('ingested_from') or directory / 'replay') / 'physics_state.npz'
            data = np.load(npz)
        extra[f'ideal_partition_interface_floor_m_digit{digit}'] = ideal_partition_interface_floor(
            data, plane['fit']['origin'], plane['fit']['normal'])
    if result['row'] == 'M8b' and 'lift_move_label_change' not in result['metrics']:
        steps = [json.loads(line) for line in (directory / 'actions.jsonl').open()]
        extra['lift_move_label_change'] = phase_label_changes(steps, ('hold_0', 'lift_0', 'move_1'))
        extra['disabled_post_cut_label_change'] = phase_label_changes(
            steps, ('raise_in_kerf', 'push', 'push_hold', 'release', 'retract', 'done'))
        extra['note'] = ('Recomputed from actions.jsonl: the row result.json metric disabled_phase_label_change '
                         'spanned cut_1 (runner bug, report-only, fixed) and is superseded by these windows')
    return extra


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--law', type=int, default=4)
    p.add_argument('--rows', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; evidence files are never overwritten')
    rows = []
    for directory in a.rows:
        result = json.loads((directory / 'result.json').read_text())
        if result.get('cutting_law') != a.law:
            raise SystemExit(f'{directory} is law {result.get("cutting_law")}, not {a.law}')
        rows.append({'row': result['row'], 'verdict': result['verdict'], 'criterion': result['criterion'],
                     'checks': result.get('checks'), 'key_numbers': key_numbers(result) if result['verdict'] != 'ERROR' else None,
                     'supplement': supplement(directory, result) if result['verdict'] != 'ERROR' and result['row'] != 'M7' else {},
                     'error': result.get('error'), 'wall_s': result.get('wall_s'),
                     'ingested_from': _get(result, 'context', 'ingested_from'),
                     'ingested_wall_s': _get(result, 'context', 'ingested_wall_s'),
                     'replay_wall_s': _get(result, 'context', 'replay_wall_s'),
                     'artifact': str(directory), 'command': result.get('command'), 'git_sha': result.get('git_sha'),
                     'git_dirty': result.get('git_dirty'), 'formal_resolution': result.get('formal_resolution'),
                     'history_stride': result.get('history_stride'), 'device': result.get('device')})
    rows.sort(key=lambda r: ORDER.index(r['row']) if r['row'] in ORDER else len(ORDER))
    counts = {}
    for row in rows:
        counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
    missing = [row for row in ORDER if row not in {r['row'] for r in rows}]
    evidence = {'protocol': 'phase1_acceptance_matrix', 'adr': 'docs/adr/0006-swept-band-cutting-law-4.md',
                'cutting_law': a.law, 'created': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                'common_settings': {'runtime_source': 'kinematic_ideal_tracking (M9: measured_trajectory_material_replay)',
                                    'seed': 42, 'particle_spacing_m': .001, 'material_dt_s': 1e-4, 'grid_dx_m': .004,
                                    'tool_profile': 'ruler_knife_v1',
                                    'component_graph': 'mechanical_components with recorded stencil weights and gradients'},
                'verdict_counts': counts, 'rows_not_run': missing,
                'scope': ('Kinematic ideal tracking and material-only replays; no robot, grasp, slip or camera evidence. '
                          'Law-4 parameters are provisional and uncalibrated. M8b is a kinematic preflight, not a demo.'),
                'rows': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(evidence, stream, indent=1)
    for row in rows:
        print(f"{row['row']:4s} {row['verdict']:12s} {row['wall_s'] and round(row['wall_s'])!s:>6s}s  {row['artifact']}")
    print(counts, 'not run:', missing)


if __name__ == '__main__':
    main()
