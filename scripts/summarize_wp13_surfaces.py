#!/usr/bin/env python3
"""Compact actual-state surface characterization using unchanged profiles."""
import argparse
import json
from pathlib import Path

import numpy as np
from run_wp13 import G


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); rows = []
    for case in ('C00_cut', 'P00_partial', 'T00_trim', 'M00_four_slices'):
        for solver in ('tlmpm_v5', 'tlmpm_v7'):
            for frame in ('during', 'final'):
                path = a.root/solver/(case+'_'+frame)/'summary.json'
                if not path.exists():
                    rows.append({'case':case,'solver':solver,'frame':frame,'status':'NOT_RUN',
                                 'reason':'No saved deformation state; completed physics not repeated'})
                    continue
                d = json.loads(path.read_text()); profiles = []
                for r in d['profiles']:
                    m = r['metrics']; topology = m['topology']
                    profiles.append({k:r[k] for k in ('profile','vertices','triangles','extraction_wall_s',
                                                       'basic_mesh_validity','self_intersections','hermite_source')})
                    profiles[-1]['metrics'] = {k:m[k] for k in ('locus_source','conflicting_link_rows',
                        'volume','region','distance_to_side_locus_plane_core','signed_bias','coverage',
                        'plane_coverage','normal_error_deg_core','normal_error_vs_F_mapped_normal_deg_core',
                        'normal_outward_fraction_core','roughness_mesh_core','roughness_locus_core','rim','status')}
                    profiles[-1]['topology'] = {k:v for k,v in topology.items() if k != 'shell_rows'}
                state = Path(d['physics_run'])/d['state']; z = np.load(state)
                physical = d.get('physical_faces')
                if physical is None:
                    supplemental = Path('artifacts/wp13/face_planarity_001')/solver/(case+'_'+frame)/'summary.json'
                    if supplemental.exists(): physical = json.loads(supplemental.read_text())['physical_faces']
                rows.append({'case':case,'solver':solver,'frame':frame,'status':'MEASURED',
                    'time_s':float(z['time_s']) if 'time_s' in z else None,
                    'state':str(state),'state_sha256':d['state_sha256'],'profiles':profiles,
                    'physical_faces':physical,'summary_path':str(path),'summary_sha256':G.sha(path)})
    output = {'scope':'Unchanged v1 profiles over saved physics, midpoint fallback explicitly labelled. '
              'No physical integration, mesh smoothing or visual tear splitting. During states are the first '
              'saved nonzero-front sample, not necessarily matched times. Self intersections NOT_RUN; '
              'watertight does not imply absence of degenerates or self intersections. '
              'TRIM and partial global planarity not scored as a severed planar face.',
              'rows':rows,'measured_states':sum(r['status']=='MEASURED' for r in rows),
              'not_run_states':sum(r['status']=='NOT_RUN' for r in rows)}
    G.write_new(a.out,output)
    print(json.dumps({k:output[k] for k in ('measured_states','not_run_states')}))


if __name__ == '__main__':
    main()
