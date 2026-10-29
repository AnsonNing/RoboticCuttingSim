#!/usr/bin/env python3
"""Declared 1 mm / 0.1 ms GPU knife-law and material experiments; no robot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.blade_probe import VERTICAL_CUT, FLAT_SKIM, Phase, block, drive_blade
from robotic_cutting_sim.food_materials import profiles, receipt
from robotic_cutting_sim.multi_cut_evaluation import MultiCutGoal, evaluate_multi_cut, reference_bonds
from robotic_cutting_sim.recording import new_output, manifest, write_json
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1


def cases():
    top = .012
    result = dict(push=(VERTICAL_CUT, [.001, 0., top+.004], [Phase((0., 0., -.05), .4)], {}),
        slow_push=(VERTICAL_CUT, [.001, 0., top+.001], [Phase((0., 0., -.01), .4)], {}),
        slicing=(VERTICAL_CUT, [.001, -.03, top+.001], [Phase((0., .2, -.01), .4)], {}),
        slicing_disabled=(VERTICAL_CUT, [.001, -.03, top+.001], [Phase((0., .2, -.01), .4)], {'slice_efficiency': 0.}),
        side_push=(VERTICAL_CUT, [-.034, 0., .002], [Phase((.05, 0., 0.), .3)], {}),
        flat_press=(FLAT_SKIM, [.010, 0., top+.002], [Phase((0., 0., -.02), .3)], {}),
        half_then_withdraw=(VERTICAL_CUT, [.001, 0., top+.004], [Phase((0., 0., -.05), .2), Phase((0., 0., .05), .2, -1)], {}),
        handle_only=(VERTICAL_CUT, [-.05, 0., top+.01], [Phase((0., 0., -.05), .2)],
                     {'obstacle_boxes_edge': [[[0., 0., .05], [.006, .006, .006]]]}),
        two_digits=(VERTICAL_CUT, [-.012, 0., top+.004], [Phase((0., 0., -.1), .2, 0),
            Phase((0., 0., .1), .2, -1), Phase((.24, 0., 0.), .1, -1), Phase((0., 0., -.1), .2, 1)], {'cuts': 2}))
    for tilt in (-40, -20, 20, 40):
        rotation = Rotation.from_euler('y', tilt, degrees=True) * VERTICAL_CUT
        advance = rotation.apply([1., 0., 0.])
        anchor = np.array([.001, 0., top/2])
        start = anchor - advance * ((top+.005-anchor[2])/-advance[2])
        duration = (top+.010)/(-advance[2]*.05)
        result[f'oblique_{tilt:+d}'] = (rotation, start, [Phase(tuple(advance*.05), duration)], {})
    for yaw in (30, 60):
        rotation = Rotation.from_euler('z', yaw, degrees=True) * VERTICAL_CUT
        result[f'yaw_{yaw}'] = (rotation, [.001, 0., top+.004], [Phase((0., 0., -.05), .4)], {})
    for roll in (-15, 15):
        rotation = VERTICAL_CUT * Rotation.from_euler('x', roll, degrees=True)
        result[f'edge_roll_{roll:+d}'] = (rotation, [.001, 0., top+.006], [Phase((0., 0., -.05), .5)], {})
    result['saw_through'] = (VERTICAL_CUT, [.001, -.015, top+.004],
        [Phase((0., speed, -.05), .1) for speed in (.3, -.3, .3, -.3)], {})
    result['slice_through'] = (VERTICAL_CUT, [.001, -.02, top+.004],
        [Phase((0., speed, -.025), .2) for speed in (.2, -.2, .2, -.2)], {})
    result['skim'] = (Rotation.from_euler('y', 25, degrees=True), [-.04, 0., top-.004],
                      [Phase((.1, 0., 0.), .8)], {})
    return result


def run_case(name, rotation, start, phases, options, out, device, spacing, config=None, energy=5.):
    output = out / name
    output.mkdir()
    began = time.monotonic()
    row = dict(name=name, command_device=device, spacing_m=spacing, material_dt_s=(config or {}).get('dt', .0001),
               rotation_xyzw=rotation.as_quat(), start_m=start, phases=[asdict(phase) for phase in phases],
               config_overrides=config or {}, cutting_energy_j_m2=energy)
    try:
        points, masses = block(spacing=spacing, density=(config or {}).get('density', 1500.))
        result = drive_blade(RULER_KNIFE_V1, rotation, start, phases, device=device, spacing=spacing,
            points=points, masses=masses, gpu_graph=True, record_history=True,
            config_overrides=config, cutting_energy=energy, **options)
        forces = np.concatenate(result['force_history_N'])
        state = result['state']
        np.savez_compressed(output / 'physical_state.npz', **state, reference_positions=result['points'], mass_kg=masses,
            force_history_N=forces, time_s=np.arange(1, len(forces)+1)*row['material_dt_s'],
            reference_stencil_nodes=result['reference_stencil_nodes'],
            reference_stencil_weights=result['reference_stencil_weights'],
            reference_stencil_gradients=result['reference_stencil_gradients'])
        determinant = np.linalg.det(state['F'])
        finite = all(np.isfinite(state[key]).all() for key in ('positions', 'velocities', 'F', 'Fpi'))
        row.update(status='PASS_EXECUTION' if finite else 'FAIL_NONFINITE',
                   components=result['components'], severed_bonds=result['severed_bonds'],
                   labelled_after_phase=result['labelled_after_phase'],
                   severed_bonds_after_phase=result['severed_bonds_after_phase'],
                   mass_fractions=result['mass_fractions'], forces=result['forces'],
                   minimum_detF=float(determinant.min()),
                   maximum_speed_m_s=float(np.linalg.norm(state['velocities'], axis=1).max()))
        lattice = np.round((result['points']-result['points'].min(0))/spacing).astype(int)
        bonds = reference_bonds(lattice)
        a, b = bonds.T
        from robotic_cutting_sim.cut_topology import conflicting
        severed = conflicting(state['codes'][a], state['codes'][b], options.get('cuts', 1))
        midpoints = (result['points'][a[severed]]+result['points'][b[severed]])/2
        error = np.abs((midpoints-start) @ rotation.apply([0., 0., 1.]))
        row['cut_plane_reference_midpoint_error_m'] = dict(max=float(error.max()), p95=float(np.percentile(error, 95))) if len(error) else None
        if name == 'two_digits':
            goal = MultiCutGoal((((-.012, 0., 0.), (1., 0., 0.)), ((.012, 0., 0.), (1., 0., 0.))))
            row['multi_cut_evaluation'] = evaluate_multi_cut(dict(codes=state['codes'], mass_kg=masses,
                reference_positions=result['points'], reference_indices=lattice,
                **{key: result[key] for key in ('reference_stencil_nodes', 'reference_stencil_weights', 'reference_stencil_gradients')}), goal)
    except Exception as error:
        row.update(status='FAIL_EXECUTION', error=repr(error), traceback=traceback.format_exc())
    row['wall_s'] = time.monotonic()-began
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    row = finite_json(row)
    write_json(output / 'result.json', row)
    print(name, row['status'], row.get('components'), row.get('severed_bonds'), flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--spacing', type=float, default=.001)
    p.add_argument('--mode', choices=('law', 'materials', 'graph_check'), default='law')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = new_output(a.out)
    write_json(out / 'manifest.json', manifest(dict(device=a.device, spacing_m=a.spacing, mode=a.mode,
        seed=42, protocol='wp10_gpu_law_matrix_v1'), 'kinematic_gpu_blade_probe'))
    rows = []
    if a.mode == 'law':
        write_json(out / 'declared_cases.json', {name: dict(rotation=rotation.as_quat(), start=start,
            phases=[asdict(phase) for phase in phases], options=options) for name,(rotation,start,phases,options) in cases().items()})
        for name, (rotation, start, phases, options) in cases().items():
            rows.append(run_case(name, rotation, start, phases, options, out, a.device, a.spacing))
            write_json(out / 'rows.json', rows)
        indexed = {row['name']: row for row in rows}
        def good(name):
            return indexed[name]['status'] == 'PASS_EXECUTION'
        def force(name):
            return indexed[name]['forces'][0]['mean_N'][2] if good(name) else float('inf')
        checks = dict(aligned_push=good('push') and indexed['push']['components'] >= 2,
            side_no_bonds=good('side_push') and indexed['side_push']['severed_bonds'] == 0,
            flat_no_bonds=good('flat_press') and indexed['flat_press']['severed_bonds'] == 0,
            slicing_less_than_quarter=good('slicing') and good('slow_push') and force('slicing') < .25*force('slow_push'),
            withdraw_adds_no_bonds=good('half_then_withdraw') and len(set(indexed['half_then_withdraw']['severed_bonds_after_phase'])) == 1,
            blunt_handle_no_bonds=good('handle_only') and indexed['handle_only']['severed_bonds'] == 0,
            two_digits_three_actual_parts=indexed['two_digits'].get('multi_cut_evaluation', {}).get('status') == 'PASS_MULTI_CUT_OUTCOME')
        write_json(out / 'result.json', dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
            slicing_to_push_resistance_ratio=force('slicing')/force('slow_push') if good('slicing') and good('slow_push') and force('slow_push') else None,
            rows=rows, scope='GPU kinematic probes; actual articulated motion is a separate matrix'))
    elif a.mode == 'materials':
        write_json(out / 'declared_materials.json', dict(profiles=profiles(), dt_candidates_s=[.0001, .00005, .00002, .00001],
            finite_speed_screen_m_s=5., positive_detF=True, seed=42))
        for name, profile in profiles().items():
            for dt in (.0001, .00005, .00002, .00001):
                config = dict(dt=dt, density=profile['density_kg_m3'], young=profile['young_modulus_pa'],
                    poisson=profile['poisson_ratio'], yield_stress=profile['yield_stress_pa'], contact_friction=profile['friction'])
                row = run_case(name+f'_dt{dt:g}', VERTICAL_CUT, [.001, 0., .016], [Phase((0., 0., -.05), .4)], {},
                               out, a.device, a.spacing, config, profile['fracture_energy_j_m2'])
                row['material'] = receipt(name, dt=dt)
                row['stability_status'] = 'PASS' if row['status'] == 'PASS_EXECUTION' and row['minimum_detF'] > 0 and row['maximum_speed_m_s'] < 5. else 'FAIL'
                rows.append(row)
                write_json(out / 'rows.json', rows)
        write_json(out / 'result.json', dict(status='PASS' if all(any(row['stability_status']=='PASS' and row['material']['grid_cfl']<=.2
            for row in rows if row['material']['profile_id']==name) for name in profiles()) else 'FAIL', rows=rows,
            calibration='NOT_VALIDATED', scope='Force and stability orders only, no measured food calibration'))
    else:
        args = (RULER_KNIFE_V1, VERTICAL_CUT, [.001, 0., .013], [Phase((0., 0., -.05), .15)])
        direct = drive_blade(*args, device=a.device, spacing=a.spacing, record_history=True)
        graph = drive_blade(*args, device=a.device, spacing=a.spacing, record_history=True, gpu_graph=True)
        error = float(np.max(np.abs(direct['state']['positions'] - graph['state']['positions'])))
        checks = dict(actual_cut_observed=int(np.count_nonzero(direct['state']['codes'])) > 0,
                      equal_cut_history=np.array_equal(direct['state']['codes'], graph['state']['codes']),
                      position_max_error_below_1um=error < 1e-6)
        for name, result in [('direct', direct), ('graph', graph)]:
            np.savez_compressed(out / (name+'.npz'), **result['state'], force_history_N=np.concatenate(result['force_history_N']))
        write_json(out / 'result.json', dict(status='PASS' if all(checks.values()) else 'FAIL',
            checks={key:bool(value) for key,value in checks.items()}, position_max_error_m=error,
            labelled_particles=int(np.count_nonzero(direct['state']['codes'])),
            scope='Actual CUDA partial cut with and without graph execution; no robot'))


if __name__ == '__main__':
    main()
