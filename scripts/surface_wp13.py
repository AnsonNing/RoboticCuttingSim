#!/usr/bin/env python3
"""Offline, unchanged v1 surface profiles on saved WP-13 physics; no simulation step."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from run_wp13 import G, ROOT, build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--state', default='final_state.npz')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((a.run/'summary.json').read_text())
    task, _ = build(summary['case'], summary['solver'])
    state_path = a.run/a.state
    z = np.load(state_path)
    x, F, codes, X, lattice = (z[k] for k in ('positions', 'F', 'codes', 'reference', 'lattice'))
    s = task.particle_spacing_m
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.cutting_surface import CuttingSurface, CurrentMechanicalPartition, substrate_anchor_mask
    from robotic_cutting_sim.cut_aware_surface import CutAwareSurface
    from robotic_cutting_sim.surface_runtime import LIVE_CUT_AWARE_OPTIONS
    from robotic_cutting_sim import surface_metrics_v1 as S
    masses = np.full(len(X), task.material.density_kg_m3*s**3)
    # Rebuild only the fixed reference stencil, from the exact saved original
    # reference points. This constructor does not evolve or edit the saved state.
    bed = TLMPM(X, masses, s, TLMPMConfig(dx=task.grid_dx_m, cuts=task.max_cuts,
                floor=False, fixture_walls=False, tool=False), 'cpu',
                slots_per_node=task.scene['solver'].get('slots_per_node', 8))
    partition = CurrentMechanicalPartition(bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy(),
                                          task.max_cuts, encoding='packed2')
    count, labels, _ = partition.partition(codes)
    profiles = {
        'particle_surface_v1': CuttingSurface(partition, s, s/2, a.device, profile='current_volume_v1',
                                 unmeshable='skip_presentation_v1', substrate_anchors=substrate_anchor_mask(X)),
        'cut_aware_surface_v1': CutAwareSurface(partition, s, lattice, bed.reference.numpy(), masses_kg=masses,
                                  density_kg_m3=task.material.density_kg_m3, **LIVE_CUT_AWARE_OPTIONS),
    }
    rows = []
    for name, surface in profiles.items():
        started = time.monotonic()
        vertices, faces, info = surface.extract(x, codes, F)
        elapsed = time.monotonic()-started
        # Common midpoint locus in both arms: no invented Hermite data.
        report = S.surface_report(X, lattice, x, codes, vertices, faces, spacing=s, labels=labels,
            masses=masses, density=task.material.density_kg_m3, encoding='packed2', cuts=task.max_cuts,
            F=F, reference_space=False, validity=False)
        validity = S.mesh_validity(vertices, faces)
        np.savez_compressed(a.out/f'{name}.npz', vertices=vertices, faces=faces)
        row = {'profile': name, 'surface_info': info, 'metrics': report, 'basic_mesh_validity': validity,
               'extraction_wall_s': elapsed, 'vertices': len(vertices), 'triangles': len(faces),
               'self_intersections': {'status': 'NOT_RUN', 'reason': 'focused offline geometry characterization'},
               'hermite_source': 'midpoint_fallback_R4; no cut-event log recorded in this particle-profile episode'}
        rows.append(row)
        G.write_new(a.out/f'{name}.json', row)
    G.write_new(a.out/'summary.json', {'physics_run': str(a.run), 'state': a.state,
        'state_sha256': G.sha(state_path), 'case': summary['case'], 'solver': summary['solver'],
        'mechanical_components': count, 'profiles': rows,
        'scope': 'Unchanged one-way v1 profiles on saved x/F/codes. No physics updates, renderer optimization, '
                 'plane flattening, distance-based tearing or invented Hermite record.'})


if __name__ == '__main__':
    main()
