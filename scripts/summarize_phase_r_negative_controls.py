#!/usr/bin/env python3
"""Read-only summary of the Phase R negative-control rows into docs/evidence/wp11/phaseR_negative_controls_001.json.

Reads ``result.json`` of every formal row (``<row>_law8[_v5]_001``), the smoke rows and the declared v4
counterpart records; writes one evidence file and refuses to overwrite it. Nothing under the run directories
changes.

  .venv/bin/python scripts/summarize_phase_r_negative_controls.py --root artifacts/wp11 \\
      --out docs/evidence/wp11/phaseR_negative_controls_001.json
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs'/'wp11'/'phaseR_negative_controls_v1.json'
COUNTERPARTS = {
    'Ra_no_contact': ['phaseC4/k2_r05_nc_no_contact_law8_001', 'phaseF/f2_9_no_contact_law8_001'],
    'Ra_side_push': ['phaseC4/k2_r05_nc_side_push_law8_001', 'phaseF/f2_9_side_push_law8_001'],
    'Ra_flat_press_normal': ['phaseC4/k2_r00_nc_flat_press_law8_001', 'phaseF/f2_9_flat_press_law8_001'],
    'Ra_blunt_handle_v2': ['phaseC4/k2_r05_nc_handle_law8_001', 'phaseF/f2_9_blunt_handle_v2_law8_001'],
    'Ra_stop_mid': ['phaseC4/k2_r05_nc_stop_mid_law8_001'],
    'Ra_lift_mid': ['phaseC4/k2_r05_nc_lift_mid_law8_001'],
    'Ra_half_length': ['phaseC4/k2_r05_nc_half_length_law8_001'],
    'Rb_side_push': ['phaseF/f2_9_side_push_law8_001'],
    'Rb_flat_press': ['phaseF/f2_9_flat_press_law8_001'],
    'Rb_no_contact': ['phaseF/f2_9_no_contact_law8_001'],
    'Rb_stop_mid': ['phaseC4/k2_r05_nc_stop_mid_law8_001'],
    'Rb_withdrawal': ['phaseF/f2_7_withdrawal_v2_law8_001'],
    'Rb_half_depth_cut': ['phaseF/f2_7_withdrawal_v2_law8_001'],
    'Rb_blunt_handle_v2': ['phaseF/f2_9_blunt_handle_v2_law8_001'],
}


def _get(d, *path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def key_metrics(result):
    m = result.get('metrics') or {}
    control = m.get('control') or {}
    window = m.get('contact_window') or {}
    out = {
        'final_mechanical_components': m.get('final_mechanical_components', m.get('final_mechanical_components')),
        'final_labelled_count': m.get('final_labelled_count'), 'max_labelled_count': m.get('max_labelled_count'),
        'mass_check_pass': _get(m, 'mass_check', 'pass'),
        'contact_window': {k: window.get(k) for k in ('phases', 'contact_steps', 'phase_steps', 'mean_N', 'max_N',
                                                       'all_phase_steps_mean_N')} if window else None,
        'blade_reaction_episode': m.get('blade_reaction_episode'),
        'armed_contacts': m.get('armed_contacts'),
        'core_band_ligament': m.get('core_band_ligament'),
        'withdraw_label_change': m.get('withdraw_label_change'), 'retract_label_change': m.get('retract_label_change'),
        'hold_label_change': control.get('hold_label_change'),
        'lift_or_withdraw': {k: control.get(k) for k in ('withdraw_label_change', 'lift_label_change',
                                                          'furthest_armed_edge_local_x_m', 'far_region_particles',
                                                          'far_region_attached_mass_fraction',
                                                          'far_region_labelled_particles',
                                                          'max_labelled_initial_local_x_m', 'lead_bound_m')
                             if k in control} or None,
        'resumed_outcome_report_only': control.get('resumed_outcome_report_only'),
        'footprint': m.get('footprint'), 'blade_contact_steps': m.get('blade_contact_steps'),
        'workpiece_motion_hold_end_vs_pre_press': _get(m, 'workpiece_motion', 'hold_end_vs_pre_press'),
        'plowing': m.get('plowing'), 'stability_stable': _get(m, 'stability', 'stable'),
        'native_packed': m.get('native_packed'),
        'c4_force_all_steps': m.get('c4_all_steps_force') or m.get('forces_all_steps_c4') or m.get('c4_skim_force'),
        'c4_forces': m.get('forces') if 'control' in m and 'trim_separation' in m else None,
        'c4_control': control or None}
    return {k: v for k, v in out.items() if v is not None}


def row_summary(path):
    result = json.loads((path/'result.json').read_text())
    return {'dir': str(path), 'verdict': result.get('verdict'), 'criterion_outcome': result.get('criterion_outcome'),
            'checks': result.get('checks'), 'control_validity': result.get('control_validity'),
            'task_id': result.get('task_id'), 'solver': result.get('solver'), 'max_cuts': result.get('max_cuts'),
            'armed_cut_index': result.get('armed_cut_index'), 'formal_resolution': result.get('formal_resolution'),
            'device': result.get('device'), 'seed': result.get('seed'), 'git_sha': result.get('git_sha'),
            'git_dirty': result.get('git_dirty'), 'runner_sha256': result.get('runner_sha256'),
            'protocol_sha256': result.get('protocol_sha256'), 'package_file': result.get('package_file'),
            'command': result.get('command'), 'wall_s': result.get('wall_s'), 'steps_run': result.get('steps_run'),
            'error': result.get('error'), 'key_metrics': key_metrics(result)}


def counterpart_summary(path):
    result = json.loads((path/'result.json').read_text())
    return {'dir': str(path), 'verdict': result.get('verdict'), 'checks': result.get('checks'),
            'task_id': result.get('task_id'), 'solver': result.get('solver', 'tlmpm_v4'),
            'git_sha': result.get('git_sha'), 'key_metrics': key_metrics(result)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True, help='artifacts/wp11')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; evidence is never overwritten')
    protocol = json.loads(PROTOCOL.read_text())
    git = lambda *x: subprocess.run(['git', *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    rows, smoke = {}, {}
    for proto in ('a', 'b'):
        for row in protocol['protocols'][proto]['rows']:
            suffix = 'law8_v5' if proto == 'b' else 'law8'
            formal = a.root/'phaseR'/f'{row.lower()}_{suffix}_001'
            rows[row] = ({**row_summary(formal), 'protocol': proto} if (formal/'result.json').exists()
                         else {'protocol': proto, 'verdict': 'NOT_RUN', 'dir': str(formal)})
            rows[row]['counterparts'] = [counterpart_summary(a.root/c) for c in COUNTERPARTS[row]]
            s = a.root/'phaseR'/f'smoke_coarse_{row.lower()}_{suffix}_001'
            if (s/'result.json').exists():
                r = json.loads((s/'result.json').read_text())
                smoke[row] = {'dir': str(s), 'verdict': r.get('verdict'), 'checks': r.get('checks'),
                              'git_sha': r.get('git_sha'), 'wall_s': r.get('wall_s')}
    verdicts = [r['verdict'] for r in rows.values()]
    evidence = {'evidence_id': 'phaseR_negative_controls_001', 'protocol_id': protocol['protocol_id'],
                'protocol_file': str(PROTOCOL.relative_to(ROOT)),
                'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                'label_protocol_a': protocol['label'], 'created': datetime.now().astimezone().isoformat(),
                'summary_git_sha': git('rev-parse', 'HEAD'), 'read_only': True,
                'scope': 'Kinematic ideal-tracking material runs, 1 mm particles, 4 mm grid, 0.1 ms material dt, law 8, '
                         'seed 42; provisional uncalibrated material; no robot, IK, grasp, slip or camera evidence',
                'verdict_counts': {v: verdicts.count(v) for v in sorted(set(verdicts))},
                'rows': rows, 'smoke_nonformal': smoke}
    a.out.write_text(json.dumps(evidence, indent=1, default=str) + '\n')
    print(a.out, evidence['verdict_counts'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
