#!/usr/bin/env python3
"""Read-only summary of WP-11 Phase C4 run directories into one evidence JSON.

Reads each ``result.json`` written by ``scripts/run_phase_c4_trim_material.py``,
copies the verdicts and the key metrics unchanged and records the input
digests. Nothing is rescored here.

  .venv/bin/python scripts/summarize_phase_c4.py --runs artifacts/wp11/phaseC4/*_law8_001 \
      --diagnosis <diagnosis.json> --out docs/evidence/wp11/phaseC4_trim_material_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _get(d, *keys):
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def summarize(path):
    path = Path(path)
    r = json.loads((path / 'result.json').read_text())
    m = r.get('metrics', {})
    sep = _get(m, 'trim_separation') or {}
    chip = _get(m, 'chip') or {}
    br = _get(m, 'bridges') or {}
    orient = _get(m, 'orientation_metrics') or {}
    main_phase = next((k for k in orient if k.endswith(('_skim', '_skim_half', '_side_push', '_press', '_handle_sweep'))), None)
    o = orient.get(main_phase) or {}
    forces = _get(m, 'forces') or {}
    row = {'run': path.name, 'source_dir': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
           'task_id': r.get('task_id'), 'control': r.get('control'), 'verdict': r.get('verdict'),
           'checks': r.get('checks'), 'criterion': r.get('criterion'), 'git_sha': r.get('git_sha'),
           'git_dirty': r.get('git_dirty'), 'runner_sha256': r.get('runner_sha256'), 'seed': r.get('seed'),
           'cutting_law': r.get('cutting_law'), 'formal_resolution': r.get('formal_resolution'),
           'particle_spacing_m': r.get('particle_spacing_m'), 'material_dt_s': r.get('material_dt_s'),
           'device': r.get('device'), 'wall_s': r.get('wall_s'), 'steps_run': r.get('steps_run'),
           'command': r.get('command'), 'inputs_sha256': {'result.json': _sha(path / 'result.json')}}
    if r.get('verdict') == 'ERROR':
        row['error'] = r.get('error')
        return row
    row.update({
        'trim_separation_status': sep.get('status'), 'trim_separation_checks': sep.get('checks'),
        'removed_reference_above_plane_mass_fraction': _get(sep, 'metrics', 'removed_reference_above_plane_mass_fraction'),
        'substrate_main_component_below_plane_share': _get(sep, 'metrics', 'largest_component_below_plane_mass_share'),
        'detached_below_plane_mass_fraction': _get(sep, 'metrics', 'detached_below_plane_mass_fraction'),
        'final_mechanical_components': m.get('final_mechanical_components'),
        'final_component_mass_fractions': m.get('final_component_mass_fractions'),
        'final_labelled_count': m.get('final_labelled_count'),
        'code0_direct_bridges': br.get('direct_code_zero_bridges'),
        'bridge_nodes': {k: br.get(k) for k in ('nodes_with_codes_0_1', 'nodes_with_codes_0_2', 'nodes_with_codes_1_2')},
        'above_plane_code_counts': br.get('above_plane_code_counts'),
        'chip': {k: chip.get(k) for k in ('armed_end_time_s', 'removed_above_plane', 'attached_above_plane',
                                          'above_plane_code0_final')},
        'code0_diagnosis': {k: v for k, v in (m.get('code0_diagnosis') or {}).items() if k != 'scope'},
        'forces': {phase: {k: f.get(k) for k in ('blade_force_mean_N', 'blade_force_max_N', 'total_force_mean_N',
                                                  'total_force_max_N', 'total_force_mean_vector_world_N',
                                                  'steps_with_blade_contacts', 'steps')}
                   for phase, f in forces.items()},
        'precision_reported_separately': _get(m, 'precision', 'surface'),
        'frozen_precision_checks': _get(m, 'precision', 'frozen_checks'),
        'frozen_strict_status': _get(m, 'precision', 'frozen_status'),
        'label_to_swept_surface': _get(m, 'label_to_swept_surface', '0', 'label_distance'),
        'swept_plane_fit': _get(m, 'swept_planes', '0', 'fit'),
        'orientation_metrics_phase': main_phase,
        'orientation_metrics': {k: o.get(k) for k in ('tau_edge_tilt_deg', 'psi_edge_skew_deg', 'rho_blade_rake_deg',
                                                      'mu_motion_elevation_deg', 'source')},
        'lowest_blade_point_relative_to_edge_m': m.get('lowest_blade_point_relative_to_edge_m'),
        'stroke': m.get('stroke'), 'control_metrics': m.get('control')})
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--diagnosis', type=Path, help='JSON with the written diagnosis/recommendation (copied verbatim)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; evidence files are never overwritten')
    rows = [summarize(path) for path in a.runs]
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    evidence = {'protocol': 'wp11_phaseC4_trim_material_v1', 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'summary_git_sha': sha, 'script_sha256': _sha(__file__), 'read_only': True,
                'scope': ('Kinematic ideal-tracking material runs (cutting_tasks_v4, ruler_knife_v2_y double bevel and '
                          'ruler_knife_v2_y_chisel), 1 mm particles, 0.1 ms material dt, 4 mm grid; provisional '
                          'uncalibrated material; no robot, IK, grasp, slip or camera evidence'),
                'outcome_definition': 'demo_evaluation.trim_separation_for_task (>= 90% removed, substrate >= 99%, '
                                      'detached below-plane <= 1%, one substrate component); precision separate',
                'rows': rows}
    if a.diagnosis:
        evidence['diagnosis'] = json.loads(a.diagnosis.read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(evidence, indent=1) + '\n')
    print(a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
