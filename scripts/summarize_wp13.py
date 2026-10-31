#!/usr/bin/env python3
"""Summarize completed WP-13 evidence without changing any outcome or threshold."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from run_wp13 import ROOT, G, build


def supplemental(state_path, case, solver):
    from robotic_cutting_sim import root_cause_diagnostics as R
    from robotic_cutting_sim.goals import goal_planes
    z = np.load(state_path)
    X, x, F, codes, lattice = (z[k] for k in ('reference', 'positions', 'F', 'codes', 'lattice'))
    task, _ = build(case, solver)
    s = task.particle_spacing_m
    planes = (goal_planes(task.goal, task.scene) if task.goal['type'] in ('sever', 'divide') else None) or [
        ([0., 0., task.scene['workpiece']['base_box_m']['height_m']], [0., 0., 1.])]
    pp, qq = [], []
    for p, q, _ in R.lattice_links(lattice):
        keep = ~R.conflict_packed(codes[p], codes[q])
        pp.append(p[keep]); qq.append(q[keep])
    p, q = np.concatenate(pp), np.concatenate(qq)
    delta = (x[q]-x[p]).astype(float)
    residual = np.linalg.norm(delta-np.einsum('nij,nj->ni', .5*(F[p]+F[q]), X[q]-X[p]), axis=1)/s
    midpoint = .5*(X[p]+X[q])
    distance = np.min(np.stack([np.abs((midpoint-origin)@np.asarray(normal))
                               for origin, normal in planes]), axis=0)
    first = residual[distance <= s]
    stretch = np.linalg.norm(delta, axis=1)/np.linalg.norm(X[q]-X[p], axis=1)
    return {'near_cut': {'links': int(len(first)),
                         **{k: float(np.percentile(first, p)) if len(first) else None
                            for k, p in [('p50', 50), ('p95', 95), ('p99', 99)]}},
            'compatible_link_stretch_max': float(stretch.max()),
            'compatible_links_stretched_over_2x': int((stretch > 2.).sum()),
            'state_sha256': G.sha(state_path),
            'scope': 'Final compatible reference links; no renderer and no fracture inferred from stretch.'}


def roughness(summary):
    rows = summary['metrics']['roughness']
    vals = [(r['patch_roughness'].get('quadratic_mean'), r['patch_roughness'].get('count', 0)) for r in rows]
    vals = [(v, n) for v, n in vals if v is not None and n > 0]
    return float(np.sqrt(sum(v*v*n for v, n in vals)/sum(n for _, n in vals))) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate-status', default='NOT_ASSESSED', choices=('NOT_ASSESSED', 'PARTIAL', 'REJECTED', 'ACCEPTED'))
    a = p.parse_args()
    protocol = json.loads((ROOT/'configs/wp13/focused_v1.json').read_text())
    rows, pairs, pending = [], [], []
    for case_path in protocol['rows']:
        case = json.loads((ROOT/case_path).read_text()); name = case['id']
        arms = {}
        for solver in protocol['arms']:
            out = a.root/solver/name
            path = out/'summary.json'
            if not path.exists():
                pending.append({'case': name, 'solver': solver, 'status': 'NOT_RUN_OR_INCOMPLETE'})
                continue
            summary = json.loads(path.read_text())
            result = json.loads((out/'result.json').read_text())
            actions = [json.loads(line) for line in (out/'actions.jsonl').read_text().splitlines()]
            force = np.linalg.norm(np.asarray([r['blade_force_world_N'] for r in actions]), axis=1)
            contact_force = force[force > 1.e-12]
            extra = supplemental(out/'final_state.npz', case, solver)
            is_partial = case['episode'] == 'PARTIAL'
            passed = bool((summary.get('partial_cut_rule') or {}).get('accepted')) if is_partial else summary['class'] == 'task_success'
            runtime = summary['runtime']
            manifest_path = Path(summary['source_manifest'])
            manifest = json.loads(manifest_path.read_text())
            task, _ = build(case, solver)
            expected_options = task.controller['planner']
            actual_options = manifest['config']['planner_options']
            mismatch = {k: {'compiled': v, 'executed': actual_options.get(k)}
                        for k, v in expected_options.items() if actual_options.get(k) != v}
            row = {'case': name, 'family': case['episode'], 'solver': solver, 'outcome_pass': passed,
                   'original_class': summary['class'], 'partial_cut_rule': summary.get('partial_cut_rule'),
                   'metrics': summary['metrics'], 'supplemental': extra,
                   'roughness_rms_m': roughness(summary), 'runtime': runtime,
                   'episode_ms_per_control_step': 1000*runtime['episode_wall_s']/(runtime['simulated_s']/.02),
                   'summary_path': str(path), 'source_manifest': summary['source_manifest'],
                   'class_reason': result['class_reason'],
                   'sampled_stability': result['stability'],
                   'recorded_goal_evaluation': {k: result['evaluation'][k] for k in
                        ('verdict', 'integrity', 'outcome')},
                   'blade_reaction_N': {'control_average_peak_norm': float(force.max()),
                        'contact_controls_mean_norm': float(contact_force.mean()) if len(contact_force) else 0.,
                        'contact_controls': int(len(contact_force)),
                        'scope': 'Norm of recorded reaction averaged over each control interval; '
                                 'not an instantaneous substep peak. Fixture v5 reaction not recorded.'},
                   'summary_sha256': G.sha(path), 'manifest_sha256': G.sha(manifest_path),
                   'provenance': {k: manifest[k] for k in ('timestamp', 'command', 'python', 'backend',
                        'git_sha', 'git_dirty', 'packages', 'source_checkouts')},
                   'source_sha256': {k: v for k, v in manifest['local_code_sha256'].items()
                        if k.endswith(('tlmpm_v5.py', 'tlmpm_v6.py', 'tlmpm_v7.py', 'run_wp13.py'))},
                   'effective_planner_options': actual_options,
                   'execution_integrity': {'status': 'FAIL' if mismatch else 'PASS',
                        'compiled_vs_executed': mismatch,
                        'scope': 'Legacy driver dropped compiled overrides. Observed outcomes remain real; '
                                 'they do not validate the requested planner configuration. '
                                 'P01 duplicates P00 depth; no completed episode rerun.'}}
            if solver == 'tlmpm_v7':
                row['contact_partition_status'] = ('MEASURED_FLOAT64' if
                    (summary.get('solver_receipt') or {}).get('diagnostic_accumulation') == 'float64' else
                    'INVALID_FLOAT32_ACCUMULATION: cumulative sums can stagnate; use per-control reactions/penetration')
            arms[solver] = row; rows.append(row)
        if len(arms) == 2:
            b, c = arms['tlmpm_v5'], arms['tlmpm_v7']
            pair = {'case': name, 'family': case['episode'], 'v5_pass': b['outcome_pass'], 'v7_pass': c['outcome_pass']}
            pair['same_effective_planner_options'] = b['effective_planner_options'] == c['effective_planner_options']
            pair['both_match_compiled_planner'] = all(r['execution_integrity']['status'] == 'PASS' for r in (b, c))
            for metric in ('p50', 'p95', 'p99'):
                vb, vc = b['supplemental']['near_cut'][metric], c['supplemental']['near_cut'][metric]
                pair[f'consistency_{metric}_reduction_factor'] = vb/vc if vb is not None and vc else None
            rb, rc = b['roughness_rms_m'], c['roughness_rms_m']
            pair['roughness_reduction_fraction'] = 1.-rc/rb if rb and rc is not None else None
            pair['runtime_ratio_v7_over_v5'] = c['runtime']['episode_wall_s']/b['runtime']['episode_wall_s']
            pairs.append(pair)
    counts = {s: {'completed': sum(r['solver'] == s for r in rows),
                  'outcome_pass': sum(r['solver'] == s and r['outcome_pass'] for r in rows),
                  'compiled_planner_integrity_pass': sum(r['solver'] == s and r['execution_integrity']['status'] == 'PASS' for r in rows)} for s in protocol['arms']}
    output = {'protocol': 'configs/wp13/focused_v1.json', 'planned_episodes': 32, 'counts': counts,
              'candidate_status': a.candidate_status, 'rows': rows, 'pairs': pairs, 'pending': pending,
              'scope': 'Deterministic frozen scenes and provisional uncalibrated materials. '
                       'Outcome success, kinematic consistency, cut-face roughness and contact validity remain separate. '
                       'Runtime includes observation/evaluation instrumentation, not pure GPU kernel time. '
                       'The frozen batch used legacy goal-derived planner options, not all compiled overrides; '
                       'matched-arm measurements are descriptive under those actual paths. P01 is duplicate-depth '
                       'evidence and does not establish partial-depth diversity. Candidate remains REJECTED.'}
    G.write_new(a.out, output)
    print(json.dumps({'counts': counts, 'pairs': len(pairs), 'pending': len(pending)}))


if __name__ == '__main__':
    main()
