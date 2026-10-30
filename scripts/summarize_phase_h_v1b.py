#!/usr/bin/env python3
"""WP-11 Phase H Native reliability v1b evidence summarizer (read-only).

Reads the declared v1b run (configs/wp11/phaseG_reliability_v1b.json, plan
configs/wp11/phaseH/native_reliability_trim_v1b_8ep.json) and, when present, the
declared recycling mitigation run (configs/wp11/phaseH/native_recycle_mitigation_v1.json),
and writes one evidence file. The gating memory slopes are the ones in the run's own
reliability reports (dataset.reliability_report); they are recomputed here only to
check them. The diagnostics (Theil-Sen, OLS over episodes 1..3, last-minus-first,
deltas) are the ones the v1b declaration lists as non-gating. Never modifies an
artifact and refuses to overwrite an existing evidence file.

  .venv/bin/python scripts/summarize_phase_h_v1b.py --out docs/evidence/wp11/phaseH_native_reliability_v1b_001.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from summarize_phase_h import dataset_summary, load, rel  # noqa: E402

H = ROOT/'artifacts/wp11/phaseH'
RUN = H/'native_reliability_v1b_001'
MITIGATION = H/'native_recycle_mitigation_001'
VERSIONS = ('phaseG_reliability_v1', 'phaseG_reliability_v1a', 'phaseG_reliability_v1b')


def ols(values):
    ys = np.asarray([v for v in values if v is not None], float)
    return None if len(ys) < 2 else float(np.polyfit(np.arange(len(ys)), ys, 1)[0])


def theil_sen(values):
    from scipy.stats import theilslopes
    ys = np.asarray([v for v in values if v is not None], float)
    return None if len(ys) < 2 else float(theilslopes(ys, np.arange(len(ys)))[0])


def memory_block(series, fit_from):
    fitted = series[fit_from:]
    return {'after_episode_mib': series, 'fit_episodes': list(range(fit_from, len(series))),
            'declared_ols_slope_mib_per_episode': ols(fitted),
            'diagnostic_theil_sen_slope_mib_per_episode': theil_sen(fitted),
            'diagnostic_ols_slope_episodes_1_3_mib_per_episode': ols(series[1:4]),
            'diagnostic_last_minus_first_fitted_mib': (None if len(fitted) < 2 or None in (fitted[0], fitted[-1])
                                                        else float(fitted[-1]-fitted[0])),
            'diagnostic_deltas_mib': [None if None in (b, a) else float(b-a) for a, b in zip(series, series[1:])]}


def reports(directory):
    out = {}
    for version in VERSIONS:
        path = directory/f'reliability_{version}.json'
        if not path.exists():
            out[version] = {'status': 'MISSING'}
            continue
        report = load(path)
        out[version] = {'POLICY_READY': report['POLICY_READY'], 'failing': report['failing'],
                        'cross_episode': report['cross_episode'],
                        'per_episode_failing': {e['episode_id']: e['failing'] for e in report['episodes']},
                        'memory': report['memory'], 'determinism': report['determinism'],
                        'execution_failures': [{k: f.get(k) for k in ('episode_index', 'label', 'error')}
                                               for f in report['execution_failures']],
                        'validations': {v['episode_id']: v['status'] for v in report['validations']},
                        'primary': report.get('primary'), 'wall_s': report.get('wall_s')}
    return out


def same_scoring(a, b):
    strip = lambda r: {k: v for k, v in r.items() if k not in ('version', 'config_path', 'primary', 'wall_s')}
    return json.dumps(strip(a), sort_keys=True, default=str) == json.dumps(strip(b), sort_keys=True, default=str)


def run_summary(directory, fit_from):
    summary = dataset_summary(directory)
    if summary.get('status') == 'MISSING':
        return summary
    episodes = [e for e in summary['episodes'] if 'host_rss_mib_after' in e]
    for e in episodes:   # per-GPU split of the gated process total, the pid, and the recycle receipt if any
        meta = load(directory/e['episode_id']/'episode.json')
        e['gpu_used_mib_after_per_gpu'] = meta['memory_after'].get('gpu_used_mib')
        e['pid'] = meta['memory_after'].get('pid')
        e['home_failed_checks'] = (meta.get('home_outcome') or {}).get('failed_checks')
        e['final_codes_sha256'] = meta.get('final_codes_sha256')
        e['git'] = {k: (meta.get('git') or {}).get(k) for k in ('git_sha', 'git_dirty')}
        if meta.get('recycle'):
            e['recycle'] = meta['recycle']
    summary['memory'] = {'host_rss': memory_block([e['host_rss_mib_after'] for e in episodes], fit_from),
                         'gpu_process': memory_block([e['gpu_used_mib_after'] for e in episodes], fit_from)}
    summary['reliability'] = reports(directory)
    paths = {v: directory/f'reliability_{v}.json' for v in ('phaseG_reliability_v1a', 'phaseG_reliability_v1b')}
    if all(p.exists() for p in paths.values()):
        summary['v1a_v1b_scoring_identical'] = same_scoring(*(load(p) for p in paths.values()))
    for name in ('failure.json',):
        if (directory/name).exists():
            summary[name] = load(directory/name)
    summary['execution_failure_files'] = sorted(p.name for p in directory.glob('episode_*_failure.json'))
    reader = directory.parent/f'{directory.name}_read_dataset.json'
    if reader.exists():
        doc = load(reader)
        summary['read_dataset'] = {'report': rel(reader), 'status': doc['status'],
                                   'episodes': {e['validation']['episode_id']: e['validation']['status']
                                                for e in doc['episodes']}}
    sha = directory.parent/f'{directory.name}_grasp_trace_sha256_before_gzip.txt'
    if sha.exists():
        summary['grasp_trace_gzip_sha256'] = rel(sha)
    if (directory/'recycle').is_dir():
        summary['process_recycling'] = {'blocks': [load(p) | {'index_rows': len(load(p)['index_rows'])}
                                                   for p in sorted((directory/'recycle').glob('block_*_result.json'))]}
    return summary


def on_disk_validation(directory):
    """Diagnostic only: dataset.validate_episode on every episode directory that has an episode.json, whatever its
    index status; the index-sha check is replaced by export sha256 == episode.json export.sha256."""
    sys.path.insert(0, str(ROOT/'src'))
    from robotic_cutting_sim import dataset as ds
    index = load(directory/'index.json')
    rows = []
    for row in index['episodes']:
        meta_path = directory/row['dir']/'episode.json'
        if not meta_path.exists():
            rows.append({'episode_id': row['episode_id'], 'status': 'NO_EPISODE_JSON'})
            continue
        meta = load(meta_path)
        export = ds.sha256_file(directory/row['dir']/ds.EXPORT_NAME)
        patched = dict(index, episodes=[dict(r, export_sha256=export) if r is row else r for r in index['episodes']])
        original = ds.load_index
        ds.load_index = lambda root, _p=patched: _p   # read-only: the index file on disk is not changed
        try:
            result = ds.validate_episode(directory, row['episode_id'])
        finally:
            ds.load_index = original
        rows.append({'episode_id': row['episode_id'], 'index_status': row['status'],
                     'export_sha256_matches_episode_json': export == meta['export']['sha256'],
                     'validation_with_export_sha_from_episode_json': result['status'], 'failing': result['failing'],
                     'verdict': meta.get('verdict'), 'home': (meta.get('home_outcome') or {}).get('status'),
                     'recycle': meta.get('recycle')})
    return {'scope': 'diagnostic only; the declared criteria use the index as written', 'episodes': rows}


def mitigation_criteria(directory, summary):
    """The four declared criteria of configs/wp11/phaseH/native_recycle_mitigation_v1.json."""
    declaration = load(ROOT/'configs/wp11/phaseH/native_recycle_mitigation_v1.json')
    index = load(directory/'index.json')
    blocks = (index.get('process_recycling') or {}).get('blocks', [])
    plan = load(ROOT/'configs/wp11/phaseG/native_reliability_trim_v1.json')['episodes']
    rows = index['episodes']
    episodes = {e['episode_id']: e for e in summary['episodes'] if 'host_rss_mib_after' in e}
    pids = [b.get('pid') for b in blocks]
    c1 = len(blocks) == 2 and all(b['returncode'] == 0 for b in blocks) and None not in pids and len(set(pids)) == 2
    c2 = ([r['episode_id'] for r in rows] == [f'episode_{i:03d}' for i in range(len(plan))]
          and all(r['status'] == 'STORED' for r in rows)
          and [r['seed'] for r in rows] == [e['seed'] for e in plan]
          and (summary.get('read_dataset') or {}).get('status') == 'PASS')
    per_episode = ((summary.get('reliability') or {}).get('phaseG_reliability_v1b') or {}).get('per_episode_failing')
    c3 = per_episode is not None and len(per_episode) == len(plan) and not any(per_episode.values())
    first = [next((e for e in episodes.values() if (e.get('recycle') or {}).get('block') == b
                   and e['recycle']['block_episode'] == 0), None) for b in (0, 1)]
    delta = None if None in first else first[1]['host_rss_mib_after'] - first[0]['host_rss_mib_after']
    c4 = delta is not None and abs(delta) <= 64
    checks = {'1_both_children_exit_0_distinct_pids': c1, '2_continuous_ids_seeds_index_reader_pass': c2,
              '3_every_per_episode_v1b_check_passes': c3, '4_block1_first_rss_within_64_mib_of_block0_first': c4}
    return {'declaration': declaration['version'], 'checks': checks, 'block_pids': pids,
            'block_returncodes': [b['returncode'] for b in blocks], 'first_episode_rss_delta_mib': delta,
            'verdict': 'MITIGATION_DEMONSTRATED' if all(checks.values()) else 'MITIGATION_NOT_DEMONSTRATED',
            'failing': sorted(k for k, v in checks.items() if not v),
            'not_claimed': declaration['criteria']['not_claimed']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; never overwrite')
    declaration = load(ROOT/'configs/wp11/phaseG_reliability_v1b.json')
    fit_from = int(declaration['memory']['fit_from_episode'])
    evidence = {'schema': 'rcs_evidence_v1', 'id': 'phaseH_native_reliability_v1b_001',
                'declaration': 'configs/wp11/phaseG_reliability_v1b.json',
                'plan': 'configs/wp11/phaseH/native_reliability_trim_v1b_8ep.json',
                'memory_estimator': declaration['memory_estimator'], 'run': run_summary(RUN, fit_from)}
    run = evidence['run']
    primary = (run.get('reliability') or {}).get('phaseG_reliability_v1b') or {}
    evidence['native_policy_ready'] = {'verdict_source': 'phaseG_reliability_v1b POLICY_READY of native_reliability_v1b_001',
                                       'POLICY_READY': primary.get('POLICY_READY'), 'failing': primary.get('failing')}
    mitigation = {'declaration': 'configs/wp11/phaseH/native_recycle_mitigation_v1.json'}
    if (MITIGATION/'index.json').exists():
        mitigation['run'] = run_summary(MITIGATION, fit_from)
        mitigation['criteria'] = mitigation_criteria(MITIGATION, mitigation['run'])
        mitigation['diagnostic_on_disk_validation'] = on_disk_validation(MITIGATION)
    else:
        mitigation['status'] = 'NOT_RUN'
    evidence['recycle_mitigation'] = mitigation
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(evidence, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')
    print(json.dumps({'POLICY_READY': primary.get('POLICY_READY'), 'failing': primary.get('failing'),
                      'host': run.get('memory', {}).get('host_rss', {}).get('declared_ols_slope_mib_per_episode'),
                      'gpu': run.get('memory', {}).get('gpu_process', {}).get('declared_ols_slope_mib_per_episode')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
