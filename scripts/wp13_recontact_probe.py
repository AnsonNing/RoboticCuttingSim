#!/usr/bin/env python3
"""One explicit cut/withdraw/side-push control, with unchanged solver parameters."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def run(spacing=.002, solver=7, device='cpu'):
    from robotic_cutting_sim.blade_probe import block, drive_blade, Phase, VERTICAL_CUT
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    points, masses = block(size_m=(.024, .012, .012), spacing=spacing)
    phases = [Phase((0., 0., -.05), .32, 0), Phase((0., 0., .05), .4, -1),
              Phase((.05, 0., 0.), .4, -1), Phase((0., 0., -.05), .36, -1),
              Phase((-.02, 0., 0.), .5, -1)]
    result = drive_blade(RULER_KNIFE_V2_Y, VERTICAL_CUT, [0., 0., .016], phases,
                         points=points, masses=masses, spacing=spacing, solver_version=solver,
                         cutting_law=8, cutting_energy=.05, record_history=True,
                         cut_events='cut_events_v1', device=device, gpu_graph=device.startswith('cuda'))
    count, labels = mechanical_components_packed(result['reference_stencil_nodes'], result['reference_stencil_weights'],
        result['state']['codes'], 1, gradients=result['reference_stencil_gradients'])
    gap = None
    if count == 2:
        x = result['state']['positions']
        gap = float(cKDTree(x[labels == 0]).query(x[labels == 1])[0].min())
    events = result['cut_events']['log']
    checks = {'separated_first': count == 2, 'disarmed_labels_unchanged': len(set(result['labelled_after_phase'])) == 1,
              'disarmed_bonds_unchanged': len(set(result['severed_bonds_after_phase'])) == 1,
              'push_contact_measured': max(result['forces'][-1]['peak_abs_N']) > 1e-5,
              'faces_recontacted': gap is not None and gap <= .9*spacing+.0002,
              'finite_Hermite_events': bool(len(events) and np.isfinite(events['hermite_reference']).all()),
              'no_sign_inconsistent_events': not bool((events['flags'] & 1).any())}
    receipt = {'solver_version': solver, 'spacing_m': spacing, 'device': device, 'components': count,
               'cross_component_min_particle_distance_m': gap, 'checks': checks,
               'status': 'PASS' if all(checks.values()) else 'FAIL', 'forces': result['forces'],
               'labels_after_phase': result['labelled_after_phase'], 'events': len(events),
               'scope': 'Kinematic control only, not Native or calibrated material. Failure of the first cut '
                        'blocks a non-welding claim; topology alone is not evidence of actual face contact.'}
    return result, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=.001)
    p.add_argument('--solver', type=int, choices=(5, 7), default=7)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    result, receipt = run(a.spacing, a.solver, a.device)
    receipt['command'] = sys.argv
    (a.out/'result.json').open('x').write(json.dumps(receipt, indent=2)+'\n')
    np.savez_compressed(a.out/'state.npz', **result['state'], points=result['points'],
                        events=result['cut_events']['log'])
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
