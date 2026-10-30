#!/usr/bin/env python3
"""WP-12 Phase 2b: compact evidence over every workcell_live_NNN episode and the WP-11 reference analyses.

Reads ``artifacts/wp12/workcell_live_*/analysis_*/analysis.json`` (every attempt, none excluded) and
``artifacts/wp12/wp11_reference_002/*/analysis.json`` and writes one new JSON (never overwritten).

  .venv/bin/python scripts/summarize_wp12_workcell_live.py --out docs/evidence/wp12/phase2b_native_001.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def compact(a):
    kb, forces, disp = a.get('knife_vs_board') or {}, a.get('forces') or {}, a.get('displacement') or {}
    ops = []
    for i, op in enumerate(kb.get('per_operation') or []):
        f = (forces.get('per_operation') or [{}]*99)[i] if i < len(forces.get('per_operation') or []) else {}
        d = next((x for x in disp.get('per_operation') or [] if x['op'] == op['op']), {})
        b = next((x for x in a.get('bridges') or [] if x['op'] == op['op']), None)
        o = next((x for x in a.get('operations') or [] if x['op'] == op['op']), {})
        ops.append({'op': op['op'], 'kind': op['kind'],
                    'knife_lowest_minus_support_top_end_of_armed_m': op.get('knife_lowest_minus_board_top_at_end_of_armed_m'),
                    'knife_lowest_minus_support_top_hold_m': op.get('knife_lowest_minus_board_top_hold_m'),
                    'knife_lowest_minus_support_top_op_min_m': op.get('knife_lowest_minus_board_top_op_min_m'),
                    'end_of_armed_within_clean_cut_window': op.get('end_of_armed_within_clean_cut_window'),
                    'steps_with_tool_board_contact_pairs': op.get('steps_with_tool_board_contact_pairs'),
                    'blade_reaction_armed_norm_n': {k: g(f, 'blade_reaction_armed_norm_n', k) for k in ('mean', 'max')},
                    'board_reaction_inferred_hold_z_max_n': f.get('board_reaction_inferred_hold_z_max_n'),
                    'board_reaction_inferred_hold_mean_world_n': f.get('board_reaction_inferred_hold_mean_world_n'),
                    'workpiece_horizontal_displacement_m': d.get('whole_workpiece_horizontal_m'),
                    'by_side_of_plane_x_m': {k: v.get('x_m') for k, v in (d.get('by_side_of_plane') or {}).items()} or None,
                    'crossing_links_conflicting': None if b is None else [b['crossing_links']['conflicting'], b['crossing_links']['links']],
                    'direct_code0_bridges': None if b is None else b['direct_code0_bridges']['count'],
                    'kerf_offset_from_declared_m': None if b is None else b.get('kerf_offset_from_declared_m'),
                    'tracking_error_m': o.get('tracking_error_m'),
                    'lowest_blade_point_above_bar_top_m': o.get('lowest_blade_point_above_bar_top_m'),
                    'measured_orientation': {k: g(o, 'measured_orientation', k) for k in ('rho_deg', 'tau_deg', 'psi_deg', 'mu_deg')}
                    if o.get('measured_orientation') else None})
    perf = a.get('performance') or {}
    return {'episode': a.get('episode'), 'root': a.get('root'), 'run': a.get('run'), 'class': a.get('class'),
            'class_reason': a.get('class_reason'), 'declared_rule_class': a.get('class_declared_rule'),
            'partial_cut': a.get('partial_cut'), 'task_id': a.get('task_id_recorded'), 'task_id_match': a.get('task_id_match'),
            'runtime_source': a.get('runtime_source'), 'verdict': g(a, 'task_outcome', 'verdict'),
            'failed_checks': g(a, 'task_outcome', 'failed_checks'), 'outcome_metrics': g(a, 'task_outcome', 'outcome_metrics'),
            'metrics': a.get('metrics'),
            'home_outcome': g(a, 'home', 'home_outcome', 'status'),
            'home_joint_error_max_rad': g(a, 'home', 'home_measurement', 'home_joint_error_max_rad'),
            'reset_pregrasp': (read(Path(a['run'])/'runtime'/'grasp_reset_00.json').get('status')
                               if a.get('run') and (Path(a['run'])/'runtime'/'grasp_reset_00.json').is_file() else None),
            'plan_preflight': g(a, 'plan_preflight', 'status'), 'ik': a.get('ik'),
            'grasp_task_phase': {k: g(a, 'grasp_task_phase', k) for k in ('max_slip_m', 'max_rotation_rad', 'grasp_lost')},
            'tool_contacts': {k: g(a, 'contacts_task_phase', k) for k in ('unintended_contact_count_max', 'tensor_contact_max_penetration_m',
                                                                            'non_finger_roles_in_contact')},
            'knife_vs_support_top': {k: kb.get(k) for k in ('min_over_task_phase_m', 'max_penetration_into_board_m', 'at_phase',
                                                            'steps_with_tool_board_contact_pairs', 'steps_with_tool_table_contact_pairs')},
            'material_vs_support_top': a.get('material_vs_board'),
            'workpiece_displacement': {'whole_horizontal_drift_m': g(disp, 'whole_workpiece_centroid_local_m', 'horizontal_drift_m'),
                                       'whole_drift_m': g(disp, 'whole_workpiece_centroid_local_m', 'drift_m'),
                                       'final_components': disp.get('final_components'),
                                       'pieces': disp.get('pieces')} if disp else None,
            'free_hover_check_n': {'finger_force_z': (forces.get('free_hover_finger_force_on_tool_world_n') or [None]*3)[2],
                                   'tool_weight': forces.get('tool_weight_n')} if forces else None,
            'board_reaction_inferred_task_max_z_n': forces.get('board_reaction_inferred_task_max_z_n'),
            'operations': ops,
            'measured_path_clearance': g(a, 'measured_task_path_clearance', 'status'),
            'measured_path_clearance_staged': g(a, 'measured_task_path_clearance_staged', 'status'),
            'cameras': {n: {'frames': c.get('frames'), 'start_depth_valid_fraction': g(c, 'start', 'depth_valid_fraction'),
                            'end_depth_valid_fraction': g(c, 'end', 'depth_valid_fraction'),
                            'start_to_end_translation_m': c.get('camera_translation_start_to_end_m')}
                        for n, c in (a.get('cameras') or {}).items()},
            'camera_pose_trace': a.get('camera_pose_trace'),
            'performance': {k: perf.get(k) for k in ('control_steps', 'step_wall_ms_mean', 'material_wall_ms_mean', 'episode_wall_s',
                                                     'process_wall_s', 'run_dir_bytes')},
            'attempt': {k: g(a, 'attempt', k) for k in ('command', 'started', 'ended', 'returncode', 'wall_s', 'git_sha', 'git_dirty',
                                                        'result_status', 'gpu_before')},
            'failure': a.get('failure')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--extra', type=Path, help='JSON merged under "extra" (probe check, notes)')
    a = p.parse_args(argv)
    out = a.out if a.out.is_absolute() else ROOT/a.out
    if out.exists():
        raise FileExistsError(out)
    live = []
    for root in sorted((ROOT/'artifacts/wp12').glob('workcell_live_*')):
        analyses = sorted(root.glob('analysis_*/analysis.json'))
        if not analyses:
            live.append({'root': str(root.relative_to(ROOT)), 'class': 'NOT_ANALYSED'})
            continue
        latest = read(analyses[-1])
        source = analyses[-1]
        if 'measured_task_path_clearance' not in latest:   # --skip-path-check rerun: take it from the latest that has it
            for path in reversed(analyses[:-1]):
                earlier = read(path)
                if 'measured_task_path_clearance' in earlier:
                    latest['measured_task_path_clearance'] = {**earlier['measured_task_path_clearance'],
                                                              'from_analysis': str(path.relative_to(ROOT))}
                    break
        live.append({'analysis': str(source.relative_to(ROOT)),
                     'superseded_analyses': [str(p.relative_to(ROOT)) for p in analyses[:-1]], **compact(latest)})
    reference = []
    for path in sorted((ROOT/'artifacts/wp12/wp11_reference_002').glob('*/analysis.json')):
        reference.append({'analysis': str(path.relative_to(ROOT)), **compact(read(path))})
    git = lambda *x: subprocess.run(['git', *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    protocol = ROOT/'configs/wp12/workcell_live_v1.json'
    evidence = {'evidence': 'WP-12 Phase 2b: live Native workcell acceptance of blade_master_flat_board_v1 and the old/new comparison',
                'report': 'docs/reports/WP12_WORKCELL.md (Phase 2b Native acceptance)',
                'protocol': {'path': str(protocol.relative_to(ROOT)), 'sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()},
                'scope': read(protocol)['scope'], 'live_episodes': live,
                'wp11_reference': {'scope': 'Read-only re-analysis of preserved WP-11 live records (actions.jsonl, results); their '
                                            'physics_state.npz files were deleted on 2026-10-30 (artifacts/wp11/phaseE/'
                                            'DELETED_NPZ_20261030.txt), so material-side metrics (displacement, bridges, particles vs '
                                            'support) come only from the WP-11 evidence files or are NOT_AVAILABLE. "support top" for '
                                            'wp11_workcell_v1 is the workpiece-bottom plane (board segments with 40 mm gaps).',
                                   'episodes': reference},
                'summarizer': {'git_sha': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')), 'command': sys.argv}}
    if a.extra:
        evidence['extra'] = read(a.extra)
    out.write_text(json.dumps(evidence, indent=1, allow_nan=False)+'\n')
    print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
