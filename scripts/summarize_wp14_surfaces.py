#!/usr/bin/env python3
"""Compact all completed saved-state reconstructions, retaining every defect."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_wp13 import G


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted(a.root.glob('*/*/wp14_summary.json')):
        d=json.loads(path.read_text());profiles=[]
        for r in d['profiles']:
            metrics=r['metrics']
            profiles.append({k:r[k] for k in ('profile','vertices','triangles','extraction_wall_s',
                                               'basic_mesh_validity','self_intersections','mesh_sha256','hermite_source')})
            profiles[-1]['metrics']={k:metrics[k] for k in ('locus_source','conflicting_link_rows','volume','region',
                'distance_to_side_locus_plane_core','signed_bias','coverage','plane_coverage','normal_error_deg_core',
                'normal_error_vs_F_mapped_normal_deg_core','normal_outward_fraction_core','roughness_mesh_core',
                'roughness_locus_core','rim','status')}
            profiles[-1]['topology']={k:v for k,v in metrics['topology'].items() if k!='shell_rows'}
        state=Path(d['physics_run'])/d['state'];z=np.load(state)
        rows.append({'case':d['case']['id'],'solver':d['solver'],'state':d['state'],
            'capture_time_s':float(z['time_s']) if 'time_s' in z else None,
            'source_state':str(state),'source_state_sha256':d['state_sha256'],
            'physical_faces':d['physical_faces'],'profiles':profiles,
            'source_summary':str(path),'source_summary_sha256':G.sha(path)})
    if len(rows)!=12:
        raise ValueError(f'Expected all 12 declared state characterizations; found {len(rows)}')
    G.write_new(a.out,{'measured_states':len(rows),'profile_measurements':sum(len(r['profiles']) for r in rows),'rows':rows,
        'scope':'Unchanged v1 profiles on saved actual x/F/codes, no simulation step or renderer change. '
        'Partial and curled TRIM have no whole-face planar penalty. During states retain their actual capture times '
        'and are not asserted to be matched-time comparisons. Bounded intersection zeros do not prove a valid full mesh.'})


if __name__=='__main__':
    main()
