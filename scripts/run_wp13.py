#!/usr/bin/env python3
"""Focused WP-13 paired episodes using the existing planner and evaluator.

Large histories stay in the new artifact directory; summary.json contains only
completed measurements. A zero process exit never certifies physical success.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_wp12_gridpos as G
import run_wp12_workcell_live as WL
import run_wp12_phase4_e2 as E2
import run_wp12_phase5 as P5


def build(row, solver):
    protocol = WL.load_protocol()
    spec = WL.episode_spec(protocol, row['episode'])
    scene = json.loads((ROOT/spec['scene']).read_text())
    goal = json.loads((ROOT/spec['goal']).read_text())
    # Optional source pairs include the explicit slotted-board case.
    for name in ('scene', 'goal'):
        if name in row:
            if name == 'scene':
                scene = json.loads((ROOT/row[name]).read_text())
            else:
                goal = json.loads((ROOT/row[name]).read_text())
    scene['solver']['module'] = solver
    scene['solver'].update(row.get('solver_options', {}))
    scene['workpiece'].update(deepcopy(row.get('workpiece', {})))
    if 'supports' in row:
        scene['supports'] = deepcopy(row['supports'])
    goal.update(deepcopy(row.get('goal_options', {})))
    options = {**protocol['controller']['planner_overrides'], **row.get('planner_options', {})}
    from robotic_cutting_sim.scene_config import compile_task
    goal['episode']['duration_s'] = 120.
    probe = compile_task(scene, goal, planner_options=options, seed=row.get('seed', 42))
    done = next(float(t) for t, label in zip(probe._times(), probe.reference_phase_labels) if label == 'DONE')
    goal['episode']['duration_s'] = float(np.ceil((1.1*done+1.)/.5)*.5)
    task = compile_task(scene, goal, planner_options=options, seed=row.get('seed', 42))
    return task, {'solver_module': task.solver_module, 'max_cuts': task.max_cuts,
                  'slots_per_node': scene['solver'].get('slots_per_node'), 'scene': scene, 'goal': goal}


def measures(material, task):
    from robotic_cutting_sim import root_cause_diagnostics as R, surface_metrics_v1 as S
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    bed = material.bed
    X, x, F, codes = bed.reference.numpy(), bed.x.numpy(), bed.F.numpy(), bed.code.numpy()
    planes = task.goal.get('planes')
    if planes:
        origins = [p['origin_local_m'] for p in planes]
        normals = [p['normal_local'] for p in planes]
    else:
        origins = [[0., 0., task.scene['workpiece']['base_box_m']['height_m']]]
        normals = [[0., 0., 1.]]
    distance = lambda q: np.min(np.stack([np.abs((q-np.asarray(o))@np.asarray(n))
                                        for o, n in zip(origins, normals)]), axis=0)
    consistency = R.link_consistency(X, material.lattice, x, F, codes, material.spacing, distance)
    count, labels = mechanical_components_packed(bed.nodes.numpy(), bed.weights.numpy(), codes,
                                                bed.max_cuts, gradients=bed.gradients.numpy())
    locus = S.material_locus(X, material.lattice, x, codes, encoding='packed2', cuts=bed.max_cuts, F=F,
                            spacing=material.spacing)
    rough = []
    for cut in np.unique(locus['cut']):
        mask = locus['cut'] == cut
        for side in (0, 1):
            points = locus['side_points'][mask, side]
            groups = labels[locus['pairs'][mask, side]]
            rough.append({'cut': int(cut), 'side': side+1,
                          'patch_roughness': S.patch_roughness(points, .002, groups=groups)})
    mass = np.bincount(labels, weights=material.masses)
    return {'consistency': consistency, 'locus_source': locus['source'], 'roughness': rough,
            'mechanical_components': int(count), 'component_masses_kg': sorted(mass.tolist(), reverse=True),
            'mass_error_fraction': float(abs(mass.sum()-material.masses.sum())/material.masses.sum()),
            'minimum_final_det_F': float(np.linalg.det(F).min()), 'minimum_local_z_m': float(x[:, 2].min()),
            'finite_final': bool(np.isfinite(x).all() and np.isfinite(F).all()),
            'slot_diagnostics': bed.slot_diagnostics()}


def run(a):
    row = json.loads(a.case.read_text()) if a.case else {'id': a.episode, 'episode': a.episode, 'seed': 42}
    task, info = build(row, a.solver)
    config = E2.pseudo_config(E2.load_protocol())
    config['common']['material_dt_s'] = a.dt or row.get('dt_s', .0001)
    P5.W.compile_sample = lambda config, sample: (task, info)
    sample = {'sample_id': row.get('id', row['episode']), 'family': row['episode'], 'range_set': 'wp13',
              'split': 'wp13', 'seed': row.get('seed', 42), 'variant': row['episode'], 'draws': [],
              'derived': {}, 'envelope_parameters': {}, 'ood_dimensions': []}
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    original_reset, original_advance = LiveCuttingMaterial.reset, LiveCuttingMaterial.advance
    materials, contact = [], []
    armed_steps = 0
    during_saved = False

    def reset(material, *args, **kwargs):
        result = original_reset(material, *args, **kwargs)
        materials.append(material)
        return result

    def advance(material, *args, **kwargs):
        nonlocal armed_steps, during_saved
        result = original_advance(material, *args, **kwargs)
        stats = material.bed.stats.numpy()
        contact.append([material.time_s, *stats.tolist()])
        if kwargs.get('cutting', True):
            armed_steps += 1
        if armed_steps >= 50 and not during_saved:
            np.savez_compressed(a.out/'during_state.npz', **material.bed.snapshot(),
                                reference=material.points, lattice=material.lattice,
                                time_s=material.time_s)
            during_saved = True
        return result

    LiveCuttingMaterial.reset, LiveCuttingMaterial.advance = reset, advance
    ns = SimpleNamespace(out=a.out, device=a.device, steps=None, expect_task_id=None, feasibility_class=None,
                         surface_profile=a.surface, geometry_level=None, no_geometry=True)
    started = time.monotonic()
    P5.kinematic_episode(config, sample, ns, extra={'label': a.label, 'wp13_case': row,
        'candidate_source_sha256': hashlib.sha256((ROOT/'src/robotic_cutting_sim/tlmpm_v7.py').read_bytes()).hexdigest(),
        'argv': sys.argv})
    result = json.loads((a.out/'result.json').read_text())
    if result.get('status') != 'COMPLETE':
        return 1
    material = materials[-1]
    metrics = measures(material, task)
    c = np.asarray(contact)
    metrics['contact'] = {'fixture_overlap_max_m': float(c[:, 1].max()),
                          'blade_overlap_max_m': float(c[:, 4].max()),
                          'tool_work_j': float(c[:, 3].sum()),
                          'contact_steps': int((c[:, 2] > 0).sum())}
    if hasattr(material.bed, 'contact_receipt'):
        metrics['candidate_contact_partitions'] = material.bed.contact_receipt()
    # One final F state; do not duplicate full per-particle histories.
    state = material.bed.snapshot()
    np.savez_compressed(a.out/'final_state.npz', **state, reference=material.points, lattice=material.lattice)
    if row['episode'] == 'PARTIAL':
        partial = G.partial_cut_rule(result, a.out, material.points)
    else:
        partial = None
    summary = {'case': row, 'solver': a.solver, 'status': result['status'], 'class': result['class'],
               'partial_cut_rule': partial, 'metrics': metrics, 'runtime': result['runtime'],
               'process_wall_s': time.monotonic()-started, 'source_manifest': str(a.out/'manifest.json'),
               'solver_receipt': material.bed.solver_receipt() if hasattr(material.bed, 'solver_receipt') else None}
    G.write_new(a.out/'summary.json', summary)
    print(json.dumps(G.jsonable(summary)), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case', type=Path)
    p.add_argument('--episode', default='CUT', choices=('CUT', 'TRIM', 'PARTIAL', 'TESTB', 'TESTC'))
    p.add_argument('--solver', default='tlmpm_v7', choices=('tlmpm_v5', 'tlmpm_v7'))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--dt', type=float)
    p.add_argument('--surface', default='particle_surface_v1')
    p.add_argument('--label', default='DEVELOPMENT_NOT_ACCEPTANCE')
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
