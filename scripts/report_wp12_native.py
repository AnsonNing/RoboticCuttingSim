#!/usr/bin/env python3
"""Read-only analysis of the Phase 5 Native sub-sample (phase5_native_001): per-stratum / population tables over the
27 SELECTED samples (Wilson 95 %), failure classes, grasp / IK stats, wall time, Native-vs-kinematic agreement.
Classes come from run_wp12_phase5.native_execution (the declared Native success definition) and the kinematic
result.json class. Writes one new evidence JSON (never overwritten).

  .venv/bin/python scripts/report_wp12_native.py --out docs/evidence/wp12/phase5_native_001.json
"""
import argparse, json, sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
import run_wp12_phase5 as R  # noqa: E402
from robotic_cutting_sim import generalization_wp12 as W  # noqa: E402

NATIVE = ROOT/'artifacts/wp12/phase5_native_001'
KIN = ROOT/'artifacts/wp12/phase5_001/kinematic_benchmark'
FEAS = ROOT/'artifacts/wp12/phase5_001/feasibility_benchmark'
SEL = ROOT/'docs/evidence/wp12/phase5_native_selection_001.json'


def table(rows):
    n = len(rows)
    feasible = [r for r in rows if r['feasibility_class'] == 'valid_feasible']
    launched = [r for r in feasible if r['native_class'] != 'not_run']
    k = sum(r['native_class'] == 'task_success' for r in feasible)
    return {'selected': n, 'feasible': len(feasible), 'launched': len(launched),
            'not_run': len(feasible)-len(launched), 'task_success': k,
            'success_among_feasible': W.wilson(k, len(feasible)),
            'success_among_launched': W.wilson(k, len(launched)),
            'classes': dict(Counter(f"{r['native_class']}" + (f":{r['native_failure_kind']}" if r['native_failure_kind'] else '')
                                    for r in rows)),
            'kinematic_success_same_samples': W.wilson(sum(r['kinematic_class'] == 'task_success' for r in feasible),
                                                       len(feasible)),
            'wall_s_sum': sum(r['wall_s'] or 0 for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    config = W.load_benchmark(ROOT/'configs/wp12/benchmark_v1.json')
    sel = json.loads(SEL.read_text())
    order = [sid for st in config['native']['selection']['strata'] for sid in sel['strata'][st]['selected']]
    rows = []
    for sid in order:
        feas = json.loads((FEAS/f'{sid}.json').read_text())
        sample = W.sample_by_id(config, sid)
        ex = R.native_execution(NATIVE, sid)
        nc = (ex or {}).get('native_classification') or {}
        live = (ex or {}).get('live') or {}
        attempt_p = NATIVE/sid/'attempt.json'
        attempt = json.loads(attempt_p.read_text()) if attempt_p.exists() else {}
        prep_p = NATIVE/sid/'inputs'/'prepare.json'
        prep = json.loads(prep_p.read_text()) if prep_p.exists() else {}
        kin_p = KIN/sid/'result.json'
        kin = json.loads(kin_p.read_text()) if kin_p.exists() else None
        if ex is None:
            ncls, nkind, nreason = 'not_run', None, 'never launched (no attempt directory)'
        else:
            ncls, nkind, nreason = nc.get('class'), nc.get('failure_kind'), nc.get('reason')
        episode = None
        ep_p = NATIVE/sid/'run'/'episode_000'/'result.json'
        if ep_p.exists():
            episode = json.loads(ep_p.read_text())
        outcome = (episode or {}).get('task_outcome') or {}
        rows.append({
            'sample_id': sid, 'stratum': f"{sample['family']}_{sample['range_set']}", 'family': sample['family'],
            'range_set': sample['range_set'], 'variant': sample['variant'], 'seed': sample['seed'],
            'id_population': W.id_population(config, sample['family'], sample['variant']) if sample['range_set'] == 'ID' else None,
            'ood_dimensions': feas.get('ood_dimensions'), 'feasibility_class': feas.get('feasibility_class'),
            'native_class': ncls, 'native_failure_kind': nkind, 'native_reason': nreason,
            'live_class': live.get('class'), 'live_reason': live.get('reason'),
            'verdict': live.get('verdict'), 'task_end_reason': live.get('task_end_reason'),
            'policy_final_phase': live.get('policy_final_phase'), 'home_outcome': live.get('home_outcome'),
            'grasp_task_phase': live.get('grasp_task_phase'), 'ik': live.get('ik'),
            'finite_recorded_state': live.get('finite_recorded_state'), 'control_rows': live.get('control_rows'),
            'failed_checks': R.G.failed_checks(outcome) if outcome else None,
            'recorded_evidence_errors': live.get('recorded_evidence_errors'),
            'physics_state_present': live.get('physics_state_present'),
            'native_task_id': prep.get('native_task_id'), 'steps': prep.get('steps'), 'duration_s': prep.get('duration_s'),
            'returncode': attempt.get('returncode'), 'wall_s': attempt.get('wall_s'), 'started': attempt.get('started'),
            'ended': attempt.get('ended'), 'run_dir_bytes': attempt.get('run_dir_bytes'),
            'kinematic_class': (kin or {}).get('class'), 'kinematic_failure_kind': (kin or {}).get('failure_kind'),
            'kinematic_reason': (kin or {}).get('class_reason'), 'kinematic_device': (kin or {}).get('device'),
            'kinematic_episode_wall_s': ((kin or {}).get('runtime') or {}).get('episode_wall_s')})
        r = rows[-1]
        r['agreement'] = None if ncls == 'not_run' or r['kinematic_class'] is None else \
            ('both_success' if ncls == r['kinematic_class'] == 'task_success' else
             'both_fail' if 'task_success' not in (ncls, r['kinematic_class']) else
             'native_only_success' if ncls == 'task_success' else 'kinematic_only_success')
    strata = {st: table([r for r in rows if r['stratum'] == st]) for st in config['native']['selection']['strata']}
    pops = {'ID_all': table([r for r in rows if r['range_set'] == 'ID']),
            'ID_validated': table([r for r in rows if r['id_population'] == 'ID_validated']),
            'ID_unconfirmed': table([r for r in rows if r['id_population'] == 'ID_unconfirmed']),
            'OOD (A_OOD)': table([r for r in rows if r['range_set'] == 'OOD']), 'all_selected': table(rows)}
    target = config['targets']['native_ID']
    v = pops['ID_validated']['success_among_feasible']
    agree = Counter(r['agreement'] for r in rows)
    compared = sum(agree[k] for k in ('both_success', 'both_fail', 'native_only_success', 'kinematic_only_success'))
    slips = [r['grasp_task_phase']['max_slip_m'] for r in rows if r['grasp_task_phase'] and r['grasp_task_phase'].get('max_slip_m') is not None]
    rots = [r['grasp_task_phase']['max_rotation_rad'] for r in rows if r['grasp_task_phase'] and r['grasp_task_phase'].get('max_rotation_rad') is not None]
    walls = [r['wall_s'] for r in rows if r['wall_s']]
    out = {'schema': 'rcs_wp12_phase5_native_subsample_v1', 'created': datetime.now().astimezone().isoformat(),
           'git': R.git_state(), 'benchmark_id': config['benchmark_id'],
           'command': sys.argv, 'report_tool_sha256': R._sha(__file__),
           'config_sha256': R._sha(ROOT/'configs/wp12/benchmark_v1.json'),
           'envelope_sha256': R._sha(ROOT/'configs/wp12/supported_envelope_v2.json'),
           'selection_file': str(SEL.relative_to(ROOT)), 'selection_sha256': R._sha(SEL),
           'selection_rule': config['native']['selection']['text'],
           'success_definition': config['native']['success_definition'],
           'native_root': str(NATIVE), 'kinematic_root': str(KIN),
           'scope': 'Live Native Isaac Sim 6.2 episodes (runtime_source live_isaac_sim) of the declared stratified '
                    'sub-sample (27 of 170 benchmark samples). Denominators: the selected valid_feasible samples of each '
                    'stratum (not the whole stratum). Kinematic classes are KINEMATIC ideal-tracking results of the same '
                    'sample ids (artifacts/wp12/phase5_001/kinematic_benchmark), not live.',
           'target': {**target, 'estimate': v, 'met_point': None if v['p'] is None else v['p'] >= target['value'],
                      'met_lower_bound': None if v['lo'] is None else v['lo'] >= target['value']},
           'strata': strata, 'populations': pops,
           'agreement': {'counts': dict(agree), 'compared': compared,
                         'agreement_rate': W.wilson(agree['both_success']+agree['both_fail'], compared)},
           'grasp': {'max_slip_m': max(slips) if slips else None, 'max_rotation_rad': max(rots) if rots else None,
                     'episodes_with_grasp_record': len(slips)},
           'ik': {'episodes_with_ik_not_ok': sum(bool((r['ik'] or {}).get('task_steps_ik_not_ok')) for r in rows),
                  'episodes_with_workspace_clamp': sum(bool((r['ik'] or {}).get('task_steps_workspace_clamped')) for r in rows)},
           'wall': {'episodes': len(walls), 'sum_h': sum(walls)/3600, 'min_s': min(walls) if walls else None,
                    'max_s': max(walls) if walls else None, 'mean_s': sum(walls)/len(walls) if walls else None},
           'home_outcomes': dict(Counter(r['home_outcome'] for r in rows)),
           'samples': rows}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    R.write_new(out_path, out)
    print(json.dumps({'target': out['target']['estimate'], 'agreement': out['agreement'], 'wall': out['wall']}, default=str))
    for st, t in {**strata, **pops}.items():
        w = t['success_among_feasible']
        print(f"{st:14s} {t['task_success']}/{t['feasible']} lo={w['lo']} hi={w['hi']} classes={t['classes']}")


if __name__ == '__main__':
    main()
