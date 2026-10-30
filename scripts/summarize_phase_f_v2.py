#!/usr/bin/env python3
"""Read-only evidence summary of the WP-11 Phase F v2 row (protocol wp11_phaseF_v2).

Copies the verdict, checks, control validity, declared prediction and measured numbers of
one ``run_phase_f_v2_material.py`` run into an evidence JSON; never re-scores and never
overwrites an existing file. The v1 evidence (phaseF_material_001.json) is not touched.

  .venv/bin/python scripts/summarize_phase_f_v2.py --run artifacts/wp11/phaseF/f2_9_blunt_handle_v2_law8_001 \
      --log artifacts/wp11/logs/phaseF_f2_9_blunt_handle_v2_law8_001.log --out docs/evidence/wp11/phaseF_material_v2_001.json
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs' / 'wp11' / 'phaseF_protocol_v2.json'
V1_RUN = 'artifacts/wp11/phaseF/f2_9_blunt_handle_law8_001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(run, log=None):
    result = json.loads((run / 'result.json').read_text())
    m = result.get('metrics', {})
    motion = m.get('workpiece_motion') or {}
    window = m.get('contact_window', {})
    v1 = json.loads((ROOT / V1_RUN / 'result.json').read_text()) if (ROOT / V1_RUN / 'result.json').exists() else {}
    timing = {}
    if log is not None and Path(log).exists():
        for line in Path(log).read_text().splitlines():
            if 'Elapsed (wall clock)' in line or 'Maximum resident' in line or 'Exit status' in line:
                key, _, value = line.strip().rpartition(': ')
                timing[key] = value
    return {
        'protocol': 'wp11_phaseF_v2', 'protocol_file': str(PROTOCOL.relative_to(ROOT)), 'protocol_sha256': sha(PROTOCOL),
        'created': datetime.now().astimezone().isoformat(), 'read_only': True,
        'summary_git_sha': subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True,
                                          text=True).stdout.strip(),
        'script_sha256': sha(__file__),
        'v1_record': {'run': V1_RUN, 'protocol_id': v1.get('protocol_id'), 'verdict': v1.get('verdict'),
                      'contact_window_mean_N': (v1.get('metrics') or {}).get('contact_window', {}).get('mean_N'),
                      'statement': 'v1 F2.9_blunt_handle stays FAIL; v2 is a separate, newly declared row'},
        'row': {
            'row': result.get('row'), 'run': str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
            'verdict': result.get('verdict'), 'criterion_outcome': result.get('criterion_outcome'),
            'checks': result.get('checks'), 'control_validity': result.get('control_validity'),
            'protocol_sha256_recorded': result.get('protocol_sha256'),
            'protocol_digest_matches': result.get('protocol_sha256') == sha(PROTOCOL),
            'git_sha': result.get('git_sha'), 'git_dirty': result.get('git_dirty'), 'device': result.get('device'),
            'seed': result.get('seed'), 'command': result.get('command'), 'runner_sha256': result.get('runner_sha256'),
            'v1_runner_sha256': result.get('v1_runner_sha256'), 'formal_resolution': result.get('formal_resolution'),
            'resolution': {k: result.get(k) for k in ('particle_spacing_m', 'grid_dx_m', 'material_dt_s', 'control_dt_s')},
            'runtime_source': result.get('runtime_source'), 'wall_s': result.get('wall_s'), 'log': timing,
            'error': result.get('error')},
        'declared': {'force_estimate': result.get('force_estimate'), 'prediction': result.get('prediction'),
                     'blade_clearance_check': result.get('blade_clearance_check')},
        'measured': {
            'contact_window': window, 'grip_force_by_phase': m.get('grip_force_by_phase'),
            'grip_force_press_hold_series_N': m.get('grip_force_press_hold_series_N'),
            'prediction_check_report_only': m.get('prediction_check_report_only'),
            'final_mechanical_components': m.get('final_mechanical_components'),
            'components': m.get('components'), 'final_labelled_count': m.get('final_labelled_count'),
            'max_labelled_count': m.get('max_labelled_count'), 'mass_check': m.get('mass_check'),
            'blade_contact_steps': m.get('blade_contact_steps'), 'footprint': m.get('footprint'),
            'tool_clearance_recorded_poses': m.get('tool_clearance_recorded_poses'),
            'workpiece_motion': motion, 'stability': m.get('stability'), 'armed_stroke': m.get('armed_stroke'),
            'workspace_clamped_steps': m.get('workspace_clamped_steps'), 'ik_not_ok_steps': m.get('ik_not_ok_steps')},
        'diagnosis': result.get('diagnosis'),
        'scope': 'Kinematic ideal tracking, seed 42, formal resolution, provisional uncalibrated material. A blunt-contact '
                 'negative control only: no robot, IK, grasp, Isaac Sim/Lab or cameras. Full physics acceptance: false.'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--log', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error(f'{a.out} exists; evidence is never overwritten')
    evidence = summarize(a.run.resolve(), a.log)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(evidence, indent=1) + '\n')
    print(a.out, evidence['row']['verdict'])


if __name__ == '__main__':
    main()
