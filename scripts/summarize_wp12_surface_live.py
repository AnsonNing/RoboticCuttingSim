#!/usr/bin/env python3
"""WP-12 Phase 3 Layer 6: compare surface_live_NNN Native runs with their Phase 2b records (read-only).

For every ``--run ID=ROOT`` (ID a configs/wp12/surface_live_v1.json episode):

* physics unchanged (the protocol's rule): task id, outcome class, goal verdict, piece count, piece mass fractions /
  removal metrics (|diff| <= 0.02), final labelled particles, conflicting cross-plane links; from the unchanged
  Phase 2b analysis tool output (``ROOT/analysis_NNN/analysis.json`` of ``scripts/run_wp12_workcell_live_analysis.py``)
  and the recorded final codes;
* surface (B): ``run/episode_000/surface_provenance.json`` summary (audit, Hermite sources, temporal stability,
  mesh size and defects, timings);
* profiling (A and B): ``run/episode_000/runtime_timing_trace.json`` per-step material / robot / surface / capture /
  step wall times, surface ms per stage, mesh size, GPU process memory; plus actions.jsonl step wall.

Writes ``--out`` (a new JSON file; never overwritten).

  .venv/bin/python scripts/summarize_wp12_surface_live.py --run B_CUT=artifacts/wp12/surface_live_001 ... \\
      --out docs/evidence/wp12/phase3_surface_live_001.json
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PROTOCOL = ROOT/'configs/wp12/surface_live_v1.json'


def load(path):
    return json.loads(Path(path).read_text())


def latest_analysis(root):
    found = sorted(Path(root).glob('analysis_[0-9][0-9][0-9]/analysis.json'))
    return (load(found[-1]), str(found[-1].relative_to(ROOT))) if found else (None, None)


def stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return None
    return {'n': int(len(values)), 'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max()), 'sum': float(values.sum())}


def numeric_leaves(tree, prefix=''):
    out = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(numeric_leaves(value, f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(tree, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in tree):
        for i, value in enumerate(tree):
            out[f'{prefix}[{i}]'] = float(value)
    elif isinstance(tree, (int, float)) and not isinstance(tree, bool):
        out[prefix] = float(tree)
    return out


def final_codes(root):
    with np.load(Path(root)/'run/episode_000/physics_state.npz', allow_pickle=False) as data:
        return data['codes'][-1], data['reference_indices']


def conflicting_links(codes, lattice):
    from robotic_cutting_sim.cut_events import lattice_neighbours
    from robotic_cutting_sim.cut_topology import side_masks
    neighbours = lattice_neighbours(lattice)
    a1, a2 = side_masks(codes.astype(np.int64))
    p = np.repeat(np.arange(len(codes)), neighbours.shape[1])
    q = neighbours.reshape(-1)
    keep = (q > p)
    p, q = p[keep], q[keep]
    return int((((a1[p] & a2[q]) | (a2[p] & a1[q])) != 0).sum())


def physics_comparison(b_root, a_root, tolerance):
    b, b_path = latest_analysis(b_root)
    a, a_path = latest_analysis(a_root)
    out = {'analysis_b': b_path, 'analysis_phase2b': a_path}
    if b is None or a is None:
        return {**out, 'status': 'NOT_RUN', 'reason': 'missing analysis.json'}
    ob, oa = b['task_outcome'] or {}, a['task_outcome'] or {}
    mb, ma = numeric_leaves(ob.get('outcome_metrics')), numeric_leaves(oa.get('outcome_metrics'))
    # plus the analysis tool's goal metrics (cut completion, per-protrusion removal, substrate share)
    mb.update({f'analysis.{k}': v for k, v in numeric_leaves(b.get('metrics')).items()})
    ma.update({f'analysis.{k}': v for k, v in numeric_leaves(a.get('metrics')).items()})
    diffs = {k: abs(mb[k]-ma[k]) for k in sorted(set(mb) & set(ma))}
    fraction_keys = [k for k in diffs if any(w in k for w in ('fraction', 'removed', 'share', 'completion'))]
    count_keys = [k for k in diffs if k.endswith('mechanical_components') or k.endswith('substantial_pieces')]
    codes_b, lattice = final_codes(b_root)
    codes_a, _ = final_codes(a_root)
    checks = {'task_id_equal': b['task_id_recorded'] == a['task_id_recorded'],
              'class_equal': b['class'] == a['class'] and b.get('class_declared_rule') == a.get('class_declared_rule'),
              'verdict_equal': ob.get('verdict') == oa.get('verdict'),
              'piece_counts_equal': all(diffs[k] == 0 for k in count_keys) and len(set(mb) & set(ma)) > 0,
              'fractions_within_tolerance': all(diffs[k] <= tolerance for k in fraction_keys)}
    out.update(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
               class_b=b['class'], class_phase2b=a['class'], class_declared_rule_b=b.get('class_declared_rule'),
               class_declared_rule_phase2b=a.get('class_declared_rule'), verdict_b=ob.get('verdict'), verdict_phase2b=oa.get('verdict'),
               task_id=b['task_id_recorded'],
               outcome_metrics_b=mb, outcome_metrics_phase2b=ma,
               max_fraction_abs_diff=max((diffs[k] for k in fraction_keys), default=None),
               fraction_keys_compared=fraction_keys,
               surface_dependent_quality={side: (((x.get('task_outcome') or {}).get('quality') or {}).get('surface') or {})
                                          for side, x in (('b', b), ('phase2b', a))},
               labelled_particles={'b': int(np.count_nonzero(codes_b)), 'phase2b': int(np.count_nonzero(codes_a))},
               conflicting_links={'b': conflicting_links(codes_b, lattice), 'phase2b': conflicting_links(codes_a, lattice)},
               final_codes_identical=bool(np.array_equal(codes_b, codes_a)),
               bridges_b=[{k: row.get(k) for k in ('op', 'digit', 'kerf_offset_from_declared_m', 'crossing_links')}
                          for row in b.get('bridges') or []],
               bridges_phase2b=[{k: row.get(k) for k in ('op', 'digit', 'kerf_offset_from_declared_m', 'crossing_links')}
                                for row in a.get('bridges') or []],
               home_outcome={'b': (b.get('home') or {}).get('home_outcome'), 'phase2b': (a.get('home') or {}).get('home_outcome')},
               knife_vs_board_min_m={'b': (b.get('knife_vs_board') or {}).get('min_over_task_phase_m'),
                                     'phase2b': (a.get('knife_vs_board') or {}).get('min_over_task_phase_m')},
               performance={'b': b.get('performance'), 'phase2b': a.get('performance')})
    return out


def profiling(root):
    episode = Path(root)/'run/episode_000'
    trace_path = episode/'runtime_timing_trace.json'
    if not trace_path.exists():
        return {'status': 'NOT_RUN', 'reason': 'no runtime_timing_trace.json'}
    trace = load(trace_path)
    steps, surface, capture = trace['steps'], trace['surface'], trace['capture']
    keys = sorted({k for row in surface for k in row['timings_ms']})
    previous, unchanged, triangles_changed = None, 0, 0
    for row in surface:
        if previous is not None and row['codes_sha256'] == previous['codes_sha256']:
            unchanged += 1
            triangles_changed += row['triangles'] != previous['triangles'] or row['vertices'] != previous['vertices']
        previous = row
    actions = [json.loads(line) for line in (episode/'actions.jsonl').read_text().splitlines() if line.strip()]
    return {'status': 'MEASURED', 'surface_profile': trace['surface_profile'], 'kernel_or_profile': trace['surface_profile'],
            'gpu_process_memory_mib_end': trace.get('gpu_process_memory_mib'),
            'per_step_ms': {name: stats([1e3*row[key] for row in steps if row.get(key) is not None])
                            for name, key in (('tlmpm_material_incl_recorder', 'material_wall_s'),
                                              ('runner_robot_physics_wall_field', 'robot_physics_wall_s'),
                                              ('surface_update', 'surface_update_wall_s'),
                                              ('runner_step_total', 'step_wall_s'))},
            'robot_and_other_ms_derived': stats([1e3*(row['step_wall_s']-row['material_wall_s']-row['surface_update_wall_s'])
                                                 for row in steps if row.get('material_wall_s') is not None]),
            'note': 'robot_physics_wall_s of the runner spans the PhysX substeps through the end of step() (material and '
                    'surface included); robot_and_other_ms_derived = step - material - surface_update',
            'env_step_wall_ms': stats([1e3*row['step_wall_s'] for row in actions if row.get('step_wall_s') is not None]),
            'camera_capture_ms': stats([1e3*row['capture_wall_s'] for row in capture]),
            'surface_ms': {k: stats([row['timings_ms'].get(k) for row in surface]) for k in keys},
            'mesh': {'vertices': stats([row['vertices'] for row in surface]),
                     'triangles': stats([row['triangles'] for row in surface])},
            'extractions': len(surface), 'control_steps_traced': len(steps), 'captures': len(capture),
            'unchanged_codes_extractions': unchanged,
            'unchanged_codes_extractions_with_mesh_size_change': int(triangles_changed)}


def surface_summary(root):
    path = Path(root)/'run/episode_000/surface_provenance.json'
    if not path.exists():
        return None
    data = load(path)
    summary = data['summary']
    keep = ('extractions', 'audit_error_extractions', 'audit_errors_first', 'max_mesh_shells',
            'max_visually_separated_mechanically_bridged', 'max_shell_component_violations', 'mesh_size',
            'mesh_defects_max', 'degenerate_normal_corners_max', 'degenerate_normal_triangles_max', 'update_modes', 'dirty_cells', 'temporal_unchanged_codes', 'final')
    sources = {}
    for row in data['rows']:
        for key, value in row['hermite']['crossings_by_source'].items():
            sources[key] = max(sources.get(key, 0), value)
    return {**{k: summary.get(k) for k in keep}, 'max_crossings_by_source': sources,
            'selection': {k: data['selection'].get(k) for k in ('surface_profile', 'cut_events', 'options',
                                                                'bridge_method', 'runtime_version', 'source_sha256')},
            'cut_events': {k: (data.get('cut_events') or {}).get(k) for k in
                           ('events_logged', 'event_overflow', 'substeps', 'sign_inconsistent_links', 'lead_links',
                            'unexpected_code_changes', 'log_sha256', 'device_bytes')},
            'frames_codes_match': all(f['codes_match_extraction'] for f in data['frames']),
            'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, help='ID=ROOT')
    p.add_argument('--failed', action='append', default=[], help='ID=ROOT of a kept failed attempt (recorded, not compared)')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    out = a.out if a.out.is_absolute() else ROOT/a.out
    if out.exists():
        raise FileExistsError(out)
    protocol = load(PROTOCOL)
    specs = {spec['id']: spec for spec in protocol['episodes']}
    tolerance = float(protocol['physics_unchanged_rule']['removal_fraction_tolerance_abs'])
    rows = {}
    for item in a.run:
        episode, root = item.split('=', 1)
        root = Path(root) if Path(root).is_absolute() else ROOT/root
        spec = specs[episode]
        attempt = load(root/'attempt.json') if (root/'attempt.json').exists() else None
        rows[episode] = {'root': str(root.relative_to(ROOT)), 'spec': spec,
                         'attempt': None if attempt is None else {k: attempt.get(k) for k in
                             ('returncode', 'wall_s', 'result_status', 'live_task_id', 'phase2b_task_id', 'git_sha',
                              'git_dirty', 'started', 'ended', 'run_dir_bytes', 'command')},
                         'physics_vs_phase2b': physics_comparison(root, ROOT/spec['phase2b_record'], tolerance),
                         'surface': surface_summary(root), 'profiling': profiling(root)}
    failed = {}
    for item in a.failed:
        episode, root = item.split('=', 1)
        root = Path(root) if Path(root).is_absolute() else ROOT/root
        attempt = load(root/'attempt.json') if (root/'attempt.json').exists() else {}
        result = load(root/'run/result.json') if (root/'run/result.json').exists() else {}
        actions = root/'run/episode_000/actions.jsonl'
        last = None
        if actions.exists():
            lines = [line for line in actions.read_text().splitlines() if line.strip()]
            if lines:
                row = json.loads(lines[-1])
                last = {'steps': len(lines), 'time_s': row.get('time_s'), 'policy_phase': row.get('policy_phase'),
                        'surface': (row.get('privileged_diagnostics') or {}).get('surface')}
        failed[f'{episode}@{root.name}'] = {'root': str(root.relative_to(ROOT)), 'status': result.get('status'),
                                            'error': result.get('error'), 'wall_s': attempt.get('wall_s'),
                                            'git_sha': attempt.get('git_sha'), 'last_recorded_step': last}
    git = lambda *x: subprocess.run(['git', *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    payload = {'schema': 'rcs_wp12_surface_live_evidence_v1', 'protocol': str(PROTOCOL.relative_to(ROOT)),
               'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
               'created': datetime.now().astimezone().isoformat(), 'git_sha': git('rev-parse', 'HEAD'),
               'git_dirty': bool(git('status', '--porcelain')), 'command': sys.argv, 'episodes': rows,
               'failed_attempts_kept': failed,
               'scope': 'Native Isaac Sim 6.2 live runs, seed 42, one run per episode; material parameters provisional '
                        'and uncalibrated; the surface is render geometry only'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))
    print(json.dumps({k: {'physics': v['physics_vs_phase2b'].get('status'),
                          'audit_errors': (v['surface'] or {}).get('audit_error_extractions')} for k, v in rows.items()}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
