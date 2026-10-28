#!/usr/bin/env python3
"""Compact source-linked receipt; analysis completion is separate from success."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from robotic_cutting_sim.recording import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,action='append',required=True)
    p.add_argument('--flat-controls',type=Path,required=True)
    p.add_argument('--cpu-log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the previous summary; choose a new output')
    cases={}
    for directory in args.case:
        result=json.loads((directory/'result.json').read_text())
        profiles={name:json.loads((directory/(name+'.json')).read_text()) for name in ('diagnostic','research','precision')}
        diagnostic=profiles['diagnostic']
        cases[directory.name]={'episode':result['episode'],'task_id':result['task_id'],'seed':result['seed'],'variant':result['variant'],
            'analysis_status':result['status'],'legacy_score_matches_recorded':result['legacy_score_matches_recorded_status_checks_metrics'],
            'input_files_unchanged':result['input_files_unchanged'],'analysis_sources_unchanged':result['analysis_sources_unchanged'],
            'physical_integrity':diagnostic['physical_integrity'],'cutting_outcome':diagnostic['cutting_outcome'],
            'surface_quality':diagnostic['surface_quality'],
            'legacy_status':diagnostic['legacy_precision_benchmark']['status'],
            'profiles':{name:{'success':profile['success'],'assessment':profile['profile_assessment']} for name,profile in profiles.items()},
            'receipt':{'path':str(directory/'result.json'),'sha256':digest(directory/'result.json')},
            'profile_sha256':{name:digest(directory/(name+'.json')) for name in profiles},
            'raw_geometry_samples_sha256':digest(directory/'raw_geometry_samples.npz'),
            'source_sha256':result['analysis_source_sha256']}
    flat=json.loads(args.flat_controls.read_text())
    cpu=args.cpu_log.read_text()
    tests=re.search(r'(\d+) passed',cpu)
    passed=bool(tests) and flat['status']=='PASS_CONTROLS' and all(
        c['analysis_status']=='ANALYSIS_COMPLETE' and c['legacy_score_matches_recorded'] is True
        and c['input_files_unchanged'] and c['analysis_sources_unchanged'] for c in cases.values())
    write_json(args.output,{'schema':'evaluation_v2_validation_receipt_20261029','status':'PASS_EVALUATION_DELIVERY_CHECKS' if passed else 'FAIL_DELIVERY_CHECKS',
        'scope':'Evaluation/identity regression, distinct from per-task and full scientific physics acceptance',
        'cases':cases,'flat_controls':flat,'flat_control_receipt':{'path':str(args.flat_controls),'sha256':digest(args.flat_controls)},
        'cpu_tests':{'passed':None if not tests else int(tests[1]),'log':str(args.cpu_log),'sha256':digest(args.cpu_log)},
        'summary_source_sha256':digest(Path(__file__)),'full_wp08_physics_acceptance':False,
        'new_universal_research_thresholds':None,'legacy_benchmark_thresholds_modified':False})
    print('PASS_EVALUATION_DELIVERY_CHECKS' if passed else 'FAIL_DELIVERY_CHECKS')
    return int(not passed)

if __name__=='__main__':
    raise SystemExit(main())
