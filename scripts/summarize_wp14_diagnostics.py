#!/usr/bin/env python3
"""Compact immutable summaries of actual substeps and the finite candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_wp13 import G,ROOT,build
from robotic_cutting_sim.cut_diagnostics import direct_bridges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();traces=[]
    for name in ('trim_trace_001','trim_support_trace_001'):
        root=ROOT/'artifacts/wp14'/name;path=root/'front_trace.json'
        d=json.loads(path.read_text());counts=np.asarray(d['counts']);samples=np.asarray(d['samples'])
        valid=np.asarray(d['sample_valid']);rows={}
        for i,label in enumerate(d['sample_names']):
            selected=samples[valid[:,i],i]
            rows[label]={'particles':len(selected)}
            if len(selected):
                for column in ('new_forward_m','swept_side_m','board_gap_after_m','substep'):
                    values=selected[:,d['columns'].index(column)]
                    rows[label][column]=[float(values.min()),float(values.max())]
        traces.append({'source':str(path.relative_to(ROOT)),'sha256':G.sha(path),
            'manifest':json.loads((root/'manifest.json').read_text()),'original_selection_run':d['source_run'],
            'selected_particles':len(d['particle_ids']),'link_selection':d.get('link_selection','nearest_vertical'),
            'counters':{label:{'particles':int((counts[:,i]>0).sum()),'total_substeps_or_events':int(counts[:,i].sum())}
                        for i,label in enumerate(d['counters'])},'samples':rows,
            'bevel_residence_s_range':[float(counts[:,3].min()*d['material_dt_s']),float(counts[:,3].max()*d['material_dt_s'])],
            'final_measured_class':json.loads((root/'summary.json').read_text())['class'],
            'scope':d['scope']})
    candidates=[]
    for case in ('T00_trim','C00_cut'):
        root=ROOT/'artifacts/wp14/candidate_001/tlmpm_v7_1'/case
        summary=json.loads((root/'summary.json').read_text());task,_=build(summary['case'],'tlmpm_v7_1')
        state=np.load(root/'final_state.npz')
        from robotic_cutting_sim.tlmpm_v5 import TLMPM
        from robotic_cutting_sim.material_config import TLMPMConfig
        reference=state['reference'];spacing=task.particle_spacing_m
        bed=TLMPM(reference,np.full(len(reference),task.material.density_kg_m3*spacing**3),spacing,
                   TLMPMConfig(dx=task.grid_dx_m,cuts=task.max_cuts),'cpu')
        ids=direct_bridges(bed.nodes.numpy(),bed.weights.numpy(),bed.gradients.numpy(),state['codes']&3)
        candidates.append({'case':case,'solver':summary['solver'],'class':summary['class'],
            'source_summary':str((root/'summary.json').relative_to(ROOT)),'summary_sha256':G.sha(root/'summary.json'),
            'direct_code0_bridges':len(ids),'solver_receipt':summary['solver_receipt'],
            'metrics':summary['metrics'],'runtime':summary['runtime'],
            'scope':'Exploratory candidate on GPU 0; original formal pairs use GPU 1. Not an isolated cross-GPU quality attribution.'})
    G.write_new(a.out,{'traces':traces,'candidate':candidates,
        'decision':'TRIM remains unsupported by the CUT profile. One finite candidate retained as experimental; '
        'no permissive correction or replacement of original v7 follows its failed primary TRIM.',
        'scope':'All diagnostics observe fresh actual substeps. No historical bridge is retrospectively assigned a crossing. '
        'Selected bridge particles are above the board; this does not explain or clear all other material-board contacts.'})


if __name__=='__main__':
    main()
