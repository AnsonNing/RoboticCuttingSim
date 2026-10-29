#!/usr/bin/env python3
"""Read-only summary of WP-11 Phase F run directories into one compact evidence JSON.

Reads each ``result.json`` written by ``scripts/run_phase_f_material.py``, copies the
verdicts, checks and the protocol's always_report metrics unchanged (plus the row-specific
numbers the report cites) and records the input digests. Nothing is rescored here; an
existing output file is never overwritten.

Per row it also reads the per-row log ``<logs>/phaseF_<run>.log`` (GNU time output; for an
ERROR row the traceback summary) and the queue log END line, and records whether the
row's protocol digest equals the protocol file and whether the queue verdict text equals
the result verdict. These are integrity cross-checks, not scores.

  .venv/bin/python scripts/summarize_phase_f.py --runs artifacts/wp11/phaseF/*_law8*_001 \
      --notes <notes.json> --out docs/evidence/wp11/phaseF_material_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _get(d, *keys):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        elif isinstance(d, list) and isinstance(key, int) and -len(d) <= key < len(d):
            d = d[key]
        else:
            return None
    return d


def _round(value, digits=6):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


def _log_summary(log_path, root=None):
    """Exit status, wall time, timed command and (on failure) the traceback summary of one row log."""
    log_path = Path(log_path)
    if not log_path.exists():
        return {'path': str(log_path), 'status': 'MISSING'}
    text = log_path.read_text(errors='replace')
    lines = text.splitlines()
    out = {'path': str(log_path.relative_to(root)) if root and log_path.is_relative_to(root) else str(log_path),
           'sha256': _sha(log_path), 'lines': len(lines)}
    for key, pattern in (('exit_status', r'^\s*Exit status: (\d+)'), ('elapsed_wall_clock', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)'),
                         ('command_timed', r'Command being timed: "(.*)"'), ('max_rss_kb', r'Maximum resident set size \(kbytes\): (\d+)')):
        match = re.search(pattern, text, re.M)
        if match:
            out[key] = int(match.group(1)) if key in ('exit_status', 'max_rss_kb') else match.group(1)
    if 'Traceback (most recent call last):' in text:
        start = text.rindex('Traceback (most recent call last):')
        block = [l for l in text[start:].splitlines() if not l.startswith('\t') and 'Command exited with' not in l]
        frames = [l.strip() for l in block if l.strip().startswith('File ')]
        exception = next((l.strip() for l in reversed(block) if re.match(r'^\w+(Error|Exception)\b', l.strip())), None)
        out['traceback_summary'] = {'innermost_frames': frames[-3:], 'exception': exception, 'frames': len(frames)}
    return out


def _queue_ends(queue_log):
    """END lines of the queue log keyed by run directory name (rc and the printed verdict text)."""
    ends = {}
    if queue_log is None or not Path(queue_log).exists():
        return ends
    for line in Path(queue_log).read_text().splitlines():
        match = re.match(r'^END (\S+) (\S+) (\S+) (\S+) rc=(-?\d+) ?(.*)$', line)
        if match:
            time_s, row, variant, run, rc, rest = match.groups()
            ends[run] = {'time': time_s, 'row': row, 'variant': variant, 'rc': int(rc), 'text': rest.strip()}
    return ends


PROTOCOL_ORDER = ['F2.1_gate_14', 'F2.1_gate_15', 'F2.1_gate_16', 'F2.3_inplane_v2', 'F2.4_rocking', 'F2.5_pure_rotation',
                  'F2.6_saw_cut', 'F2.6_pure_draw', 'F2.7_withdrawal_v2', 'F2.8_finite_edge', 'F2.9_flat_press',
                  'F2.9_blunt_handle', 'F2.9_side_push', 'F2.9_no_contact', 'F2.10_recontact', 'F2.11_multicut_history',
                  'F3_cut', 'F3_trim']
VARIANT_ORDER = ['formal', 'coarse', 'dt_half', 'medium', 'medium_dt133_supplementary']


def _order(row):
    name, variant = row.get('row') or '', row.get('variant') or ''
    return (PROTOCOL_ORDER.index(name) if name in PROTOCOL_ORDER else len(PROTOCOL_ORDER),
            VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else len(VARIANT_ORDER), row['run'])


def _protocol_status(r):
    """Which kind of record a row is (copied labels; nothing is rescored)."""
    variant = r.get('variant') or 'formal'
    if r.get('verdict') == 'ERROR':
        return 'ERROR_NOT_RUN'
    if variant.endswith('_supplementary'):
        return 'SUPPLEMENTARY_NON_PROTOCOL'
    if r.get('protocol_row') == 'F3_resolution' and variant != 'formal':
        return 'PROTOCOL_CHARACTERIZATION'
    return 'PROTOCOL_PASS_FAIL'


def _run_length(values):
    out = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def _forces(m):
    out = {}
    for phase, f in (m.get('forces') or {}).items():
        if f.get('status') == 'NOT_RUN':
            continue
        out[phase] = {k: f.get(k) for k in ('steps', 'blade_force_mean_N', 'blade_force_max_N', 'total_force_mean_N',
                                             'total_force_max_N', 'steps_with_blade_contacts')}
    return out


def _labels_to_surface(m):
    out = {}
    for digit, report in (m.get('label_to_swept_surface') or {}).items():
        distance = (report or {}).get('label_distance') or {}
        interface = (report or {}).get('interface') or {}
        out[digit] = {'labelled_particles': distance.get('labelled_particles'),
                      'percentiles_m': distance.get('percentiles_m'), 'side_agreement_fraction': distance.get('side_agreement_fraction'),
                      'interface_max_midpoint_distance_m': interface.get('maximum_reference_midpoint_plane_distance_m'),
                      'ideal_partition_interface_floor_m': (report or {}).get('ideal_partition_interface_floor_m'),
                      'reason': (report or {}).get('reason')}
    planes = {d: {k: _get(p, 'fit', k) for k in ('normal', 'rms_residual_m', 'max_edge_line_out_of_plane_rad')}
              if p.get('status') == 'FIT' else {'status': p.get('status'), 'reason': p.get('reason')}
              for d, p in (m.get('swept_planes') or {}).items()}
    return {'planes': planes, 'labels': out}


ROW_KEYS = {
    'trim_skim': [('trim_separation', 'status'), ('trim_separation', 'metrics', 'removed_reference_above_plane_mass_fraction'),
                  ('trim_separation', 'metrics', 'largest_component_below_plane_mass_share'),
                  ('trim_separation', 'metrics', 'detached_below_plane_mass_fraction'),
                  ('precision', 'surface'), ('precision', 'frozen_status'),
                  ('code0_diagnosis', 'above_plane_code0_final'), ('code0_diagnosis', 'classes'),
                  ('chip', 'removed_above_plane', 'at_armed_end'), ('chip', 'attached_above_plane', 'particles'),
                  ('chip', 'attached_above_plane', 'at_armed_end'), ('lowest_blade_point_relative_to_edge_m',)],
    'through_cut': [('frozen_strict',)],
    'saw_cut': [('frozen_strict',), ('swept_normal', 'segments'), ('stroke_cut_phase', 'stroke_active_steps'),
                ('stroke_cut_phase', 'armed_steps')],
    'rocking': [('rocking_interface', 'checks'), ('rocking_interface', 'metrics', 'actual_interface_m'),
                ('rocking_interface', 'metrics', 'ideal_geometric_label_interface_m'),
                ('rocking_interface', 'metrics', 'far_particles_in_opposite_side_component'),
                ('rocking_interface', 'metrics', 'far_particles_in_minor_components'),
                ('rocking_interface', 'metrics', 'label_geometry_disagreements_max_abs_distance_m'),
                ('roll_deg_range',), ('frozen_strict_report_only',), ('swept_normal', 'segments'),
                ('stroke_cut_phase', 'stroke_active_steps'), ('stroke_cut_phase', 'armed_steps')],
    'pure_rotation': [('swept_region',), ('labels_vs_swept_region',), ('labels_added_during_rotate',),
                      ('stroke_during_rotate',), ('swept_normal_during_rotate', 'segments'), ('forces_during_phase',),
                      ('labels_added_during_withdraw',), ('ligament',)],
    'pure_draw': [('swept_region',), ('labels_vs_swept_region',), ('depth_below_deepest_edge',), ('labels_added_during_draw',),
                  ('stroke_during_draw',), ('swept_normal_during_draw', 'segments'), ('forces_during_phase',),
                  ('labels_added_during_withdraw',), ('ligament',)],
    'withdrawal': [('core_band_ligament',), ('ligament_matrix_definition',), ('withdraw_label_change',),
                   ('descent_label_change',), ('stroke_during_withdraw', 'stroke_active_steps')],
    'finite_edge': [('finite_edge',), ('frozen_strict_report_only', 'metrics')],
    'flat_press': [('contact_window',), ('c4_all_steps_force',), ('armed_stroke',)],
    'side_push': [('contact_window',), ('c4_all_steps_force',), ('armed_stroke',)],
    'blunt_handle': [('contact_window',), ('blade_contacts_during_press',), ('armed_stroke',)],
    'no_contact': [('armed_contacts',), ('one_component_report',), ('armed_stroke',)],
    'recontact': [('recontact',)],
    'multicut': [('multicut',)],
}
KINDS = {'F2.1': 'trim_skim', 'F2.3': 'through_cut', 'F2.4': 'rocking', 'F2.5': 'pure_rotation', 'F2.6_saw': 'saw_cut',
         'F2.6_pure': 'pure_draw', 'F2.7': 'withdrawal', 'F2.8': 'finite_edge', 'F2.9_flat': 'flat_press',
         'F2.9_blunt': 'blunt_handle', 'F2.9_side': 'side_push', 'F2.9_no': 'no_contact', 'F2.10': 'recontact',
         'F2.11': 'multicut', 'F3_cut': 'through_cut', 'F3_trim': 'trim_skim'}


def _kind(row):
    for prefix, kind in sorted(KINDS.items(), key=lambda kv: -len(kv[0])):
        if row.startswith(prefix):
            return kind
    return None


def summarize(path, logs_dir=None, queue=None, protocol_sha256=None, root=None):
    path = Path(path)
    r = json.loads((path / 'result.json').read_text())
    m = r.get('metrics') or {}
    ctx = r.get('context') or {}
    row = {'run': path.name, 'source_dir': str(path), 'row': r.get('row'), 'variant': r.get('variant'),
           'task_id': r.get('task_id'), 'protocol_row': r.get('protocol_row'), 'protocol_pass': r.get('protocol_pass'),
           'interpretation': r.get('interpretation'), 'verdict': r.get('verdict'),
           'criterion_outcome': r.get('criterion_outcome'), 'checks': r.get('checks'),
           'git_sha': r.get('git_sha'), 'git_dirty': r.get('git_dirty'), 'runner_sha256': r.get('runner_sha256'),
           'helper_sha256': r.get('helper_sha256'), 'protocol_sha256': r.get('protocol_sha256'),
           'seed': r.get('seed'), 'cutting_law': r.get('cutting_law'), 'solver': r.get('solver'),
           'particle_spacing_m': r.get('particle_spacing_m'), 'grid_dx_m': r.get('grid_dx_m'),
           'material_dt_s': r.get('material_dt_s'), 'history_stride': r.get('history_stride'), 'device': r.get('device'),
           'formal_resolution': r.get('formal_resolution'), 'runtime_source': r.get('runtime_source'),
           'command': r.get('command'), 'wall_s': r.get('wall_s'), 'episode_wall_s': ctx.get('episode_wall_s'),
           'evaluation_wall_s': ctx.get('evaluation_wall_s'), 'steps_run': r.get('steps_run'),
           'inputs_sha256': {name: _sha(path / name) for name in ('result.json', 'manifest.json', 'actions.jsonl')
                             if (path / name).exists()}}
    row['protocol_status'] = _protocol_status(r)
    if protocol_sha256 is not None:
        row['protocol_sha256_matches_protocol_file'] = r.get('protocol_sha256') == protocol_sha256
    npz = path / 'physics_state.npz'
    if npz.exists():
        row['physics_state_bytes'] = npz.stat().st_size
    if logs_dir is not None:
        row['log'] = _log_summary(Path(logs_dir) / f'phaseF_{path.name}.log', root)
        if 'sha256' in row['log']:
            row['inputs_sha256']['log'] = row['log']['sha256']
    if queue is not None:
        end = queue.get(path.name)
        row['queue_end'] = end
        if end is not None:
            printed = end['text'].split()
            # END text: "<row> <variant> <verdict> <criterion_outcome> {checks} <wall>s"; empty for a raised row
            row['queue_verdict_matches_result'] = (printed[2] == r.get('verdict')) if len(printed) > 2 else (r.get('verdict') == 'ERROR')
    if r.get('verdict') == 'ERROR':
        row['error'] = r.get('error')
        return row
    components = m.get('components') or {}
    row['always_report'] = _round({
        'final_mechanical_components': m.get('final_mechanical_components'),
        'component_mass_fractions': (components.get('component_mass_fraction') or [])[:6],
        'component_mass_kg': (components.get('component_mass_kg') or [])[:6],
        'mass_check': m.get('mass_check'),
        'labelled_count': {'final': m.get('final_labelled_count'), 'max': m.get('max_labelled_count'),
                           'history_every_10_steps': (m.get('labelled_count_history') or [])[9::10]},
        'component_count_history_run_length': _run_length(m.get('component_count_history') or []),
        'label_to_swept_surface': _labels_to_surface(m),
        'material_reaction': {'armed_phases': _forces(m), 'episode': m.get('blade_reaction_episode')},
        'plowing': m.get('plowing'), 'stroke': m.get('stroke'),
        'stability': {k: v for k, v in (m.get('stability') or {}).items() if k != 'scope'},
        'bridges': [{k: v for k, v in b.items() if k != 'bridge_reference_xyz_mm'} for b in m.get('bridges') or []],
        'workspace_clamped_steps': m.get('workspace_clamped_steps'), 'wall_s': r.get('wall_s')})
    kind = _kind(r.get('row') or '')
    row['row_metrics'] = _round({'.'.join(map(str, keys)): _get(m, *keys) for keys in ROW_KEYS.get(kind, [])})
    if r.get('diagnosis'):
        row['diagnosis'] = _round({k: v for k, v in r['diagnosis'].items() if k not in ('bridges', 'stroke', 'plowing')})
    row['context'] = {k: ctx.get(k) for k in ('solver_module', 'solver_particles', 'substeps_per_control_step', 'edge_segments',
                                              'slots', 'schedule_max_translation_m', 'schedule_max_rotation_deg',
                                              'position_tolerance_m', 'configured_task_id', 'policy_done')}
    return row


def f3_table(rows):
    """Copied per-variant numbers of the F3 resolution study (no new scoring)."""
    table = []
    for row in rows:
        if row.get('protocol_row') != 'F3_resolution':
            continue
        entry = {'run': row['run'], 'row': row['row'], 'variant': row['variant'], 'protocol_status': row['protocol_status'],
                 'verdict': row['verdict'], 'criterion_outcome': row.get('criterion_outcome'),
                 'particle_spacing_m': row['particle_spacing_m'], 'grid_dx_m': row['grid_dx_m'],
                 'material_dt_s': row['material_dt_s'], 'wall_s': row.get('wall_s')}
        if row['verdict'] == 'ERROR':
            entry['error'] = row.get('error')
            table.append(entry)
            continue
        a, rm = row['always_report'], row['row_metrics']
        armed = a['material_reaction']['armed_phases']
        phase = next(iter(armed.values()), {})
        entry.update({'checks': row['checks'], 'components': a['final_mechanical_components'],
                      'component_mass_kg': a['component_mass_kg'][:2], 'recorded_mass_kg': (a['mass_check'] or {}).get('recorded_total_mass_kg'),
                      'armed_phase_force_mean_N': phase.get('total_force_mean_N'), 'armed_phase_force_max_N': phase.get('total_force_max_N'),
                      'episode_contact_steps_mean_N': (a['material_reaction']['episode'] or {}).get('mean_N_contact_steps'),
                      'minimum_det_F': a['stability'].get('minimum_det_F'), 'inverted_steps': a['stability'].get('inverted_steps'),
                      'stable': a['stability'].get('stable'), 'labelled_final': a['labelled_count']['final'],
                      'solver_particles': (row.get('context') or {}).get('solver_particles'),
                      'substeps_per_control_step': (row.get('context') or {}).get('substeps_per_control_step')})
        labels = a['label_to_swept_surface']['labels']
        first = next(iter(labels.values()), {})
        entry['interface_max_midpoint_distance_m'] = first.get('interface_max_midpoint_distance_m')
        entry['ideal_partition_interface_floor_m'] = first.get('ideal_partition_interface_floor_m')
        if row['row'] == 'F3_trim':
            entry.update({'trim_separation': rm.get('trim_separation.status'),
                          'removed_above_plane_fraction': rm.get('trim_separation.metrics.removed_reference_above_plane_mass_fraction'),
                          'substrate_share': rm.get('trim_separation.metrics.largest_component_below_plane_mass_share'),
                          'detached_below_plane_fraction': rm.get('trim_separation.metrics.detached_below_plane_mass_fraction'),
                          'surface': rm.get('precision.surface')})
        else:
            strict = rm.get('frozen_strict') or {}
            entry.update({'frozen_strict': strict.get('status'),
                          'cut_location_midpoint_statistic_m': (strict.get('metrics') or {}).get('maximum_reference_midpoint_plane_distance_m'),
                          'cut_at_target_plane_check': (strict.get('checks') or {}).get('cut_at_workpiece_local_target_plane')})
        table.append(entry)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--notes', type=Path, help='JSON with scope/diagnosis/limitation text written after reading the runs')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--logs', type=Path, default=ROOT / 'artifacts/wp11/logs', help='directory of phaseF_<run>.log files')
    p.add_argument('--queue-log', type=Path, default=ROOT / 'artifacts/wp11/phaseF/queue.log')
    a = p.parse_args()
    if a.out.exists():
        p.error(f'{a.out} exists; refusing to overwrite')
    missing = [str(path) for path in a.runs if not (path / 'result.json').exists()]
    if missing:
        p.error(f'no result.json in {missing}')
    protocol_path = ROOT / 'configs/wp11/phaseF_protocol_v1.json'
    protocol_sha = _sha(protocol_path)
    queue = _queue_ends(a.queue_log)
    rows = sorted((summarize(path, a.logs, queue, protocol_sha, ROOT) for path in a.runs), key=_order)
    notes = json.loads(a.notes.read_text()) if a.notes else {}
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    integrity = {'rows': len(rows), 'run_git_shas': sorted({r.get('git_sha') for r in rows if r.get('git_sha')}),
                 'any_dirty_run': any(r.get('git_dirty') for r in rows),
                 'devices': sorted({r.get('device') for r in rows if r.get('device')}),
                 'seeds': sorted({r.get('seed') for r in rows if r.get('seed') is not None}),
                 'all_protocol_sha256_match': all(r.get('protocol_sha256_matches_protocol_file') for r in rows),
                 'queue_log': str(a.queue_log.relative_to(ROOT)) if a.queue_log.is_relative_to(ROOT) else str(a.queue_log),
                 'queue_log_sha256': _sha(a.queue_log) if a.queue_log.exists() else None,
                 'queue_verdict_mismatches': [r['run'] for r in rows if r.get('queue_verdict_matches_result') is False],
                 'rows_without_queue_end': [r['run'] for r in rows if r.get('queue_end') is None],
                 'queue_rc0_but_log_exit_nonzero': [r['run'] for r in rows if (r.get('queue_end') or {}).get('rc') == 0
                                                    and (r.get('log') or {}).get('exit_status', 0) != 0],
                 'verdict_counts': {v: sum(r.get('verdict') == v for r in rows) for v in sorted({r.get('verdict') for r in rows})}}
    evidence = {'protocol': 'wp11_phaseF_v1', 'protocol_file': 'configs/wp11/phaseF_protocol_v1.json',
                'protocol_sha256': protocol_sha, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'summary_git_sha': sha,
                'script_sha256': _sha(Path(__file__)), 'read_only': True, **notes, 'integrity': integrity,
                'f3_resolution_table': [{**_round(entry), 'material_dt_s': entry['material_dt_s'], 'wall_s': _round(entry['wall_s'], 3)}
                                        for entry in f3_table(rows)], 'rows': rows}
    with a.out.open('x') as stream:
        json.dump(evidence, stream, indent=1, allow_nan=False)
        stream.write('\n')
    print(f'{len(rows)} rows -> {a.out}')


if __name__ == '__main__':
    main()
