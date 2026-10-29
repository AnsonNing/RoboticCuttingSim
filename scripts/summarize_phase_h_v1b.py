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
