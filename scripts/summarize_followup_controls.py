#!/usr/bin/env python3
"""Read-only candidate regression against the frozen compiled-planner v7 CUT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from run_wp13 import ROOT,build,G
from summarize_wp13 import supplemental
from summarize_wp14 import replay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    baseline=ROOT/'artifacts/wp14/core_001/tlmpm_v7/C00_cut';candidate=a.root/'C00_cut'
    records=[];tasks=[]
    for folder in (baseline,candidate):
        summary=json.loads((folder/'summary.json').read_text());task,_=build(summary['case'],summary['solver'])
        tasks.append(task);actions=[json.loads(s) for s in (folder/'actions.jsonl').read_text().splitlines()]
        rec={'root':str(folder),'summary_sha256':G.sha(folder/'summary.json'),'solver':summary['solver'],
            'task_class':summary['class'],'mechanical_components':summary['metrics']['mechanical_components'],
            'supplemental':supplemental(folder/'final_state.npz',summary['case'],summary['solver']),
            'roughness_p95_m':[r['patch_roughness']['p95'] for r in summary['metrics']['roughness']],
            'actual_command_replay':replay(task,actions),'finite_final':summary['metrics']['finite_final'],
            'slots':summary['metrics']['slot_diagnostics'],'solver_receipt':summary['solver_receipt']}
        records.append(rec)
    normalized=[]
    for task in tasks:
        scene=task.scene;scene['solver'].pop('module');scene['solver'].pop('research_profile',None)
        normalized.append(json.dumps([scene,task.goal,task.controller,task.particle_spacing_m],sort_keys=True))
    b,c=records;ratio=c['supplemental']['near_cut']['p95']/b['supplemental']['near_cut']['p95']
    rough_ratio=np.array(c['roughness_p95_m'])/b['roughness_p95_m']
    checks={'same_frozen_inputs_except_solver':normalized[0]==normalized[1],
            'candidate_true_sever':c['task_class']=='task_success' and c['mechanical_components']==2,
            'x_F_P95_retained':ratio<=1.2,'roughness_P95_retained':bool(np.all(rough_ratio<=1.2)),
            'finite':c['finite_final'],'no_slot_overflow':c['slots']['overflow_entries']==0}
    controls=[]
    for name in ('T00_trim','P01_shallow'):
        root=a.root/name
        if (root/'summary.json').exists():
            s=json.loads((root/'summary.json').read_text());controls.append({'case':name,'class':s['class'],
                'components':s['metrics']['mechanical_components'],'partial_cut_rule':s['partial_cut_rule'],
                'metrics':s['metrics'],'solver_receipt':s['solver_receipt'],'source_summary_sha256':G.sha(root/'summary.json')})
        else:controls.append({'case':name,'status':'NOT_RUN'})
    G.write_new(a.out,{'schema':'wp14_followup_candidate_controls_v1','CUT_regression_status':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'near_cut_x_F_P95_ratio_candidate_over_original':ratio,
        'local_roughness_P95_ratios_candidate_over_original':rough_ratio.tolist(),
        'records':records,'controls':controls,'scope':'Same GPU1, seed, compiled scene/goal/controller and numerics; unchanged frozen evaluator. '
            'Candidate regression is separate from the immutable original v7 CUT profile. Failed TRIM controls remain explicit.'})


if __name__=='__main__':main()
