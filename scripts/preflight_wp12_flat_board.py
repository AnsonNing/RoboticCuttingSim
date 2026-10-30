"""WP-12 Phase 2: geometric layout/preflight study of the blade_master_flat_board_v1 workcell.

Compiles the CUT-01-R-like sever, TRIM rho = 10 deg, Test B four slices (solid
board and the explicit slotted variant) and Test C three bumps on the flat
board and runs the WP-11 motion preflight (runtime Newton IK + C2 MuJoCo
distances; motion_preflight.preflight_plan) from the workcell HOME. For every
task it also records the lowest world point of each Franka 0 part relative to
the board top and its world y range (why the table's front edge is where it is),
and repeats the study with compute-only table/board variants (in-process
overrides, never written to configs):

* ``table_full``: the historical 1.8 x 1.2 m table centred at the origin (front
  edge y = -0.6) -> the hand of a near-horizontal skim hits the table;
* ``board_thick_30mm``: board thickness 30 mm (the workpiece z follows) ->
  contact plane, IK targets and preflight move together.

Geometry only (no dynamics, no material, no grasp holding).

  .venv/bin/python scripts/preflight_wp12_flat_board.py --out artifacts/wp12/workcell_NNN
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from robotic_cutting_sim import motion_preflight as mp  # noqa: E402
from robotic_cutting_sim import scene_config as sc  # noqa: E402
from robotic_cutting_sim.flat_board_workcell import place_on_board  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402

SCENES = ROOT/'configs/wp12/scenes'
GOALS = ROOT/'configs/wp12/goals'
TASKS = {
    'CUT': ('bar_flat_board', ROOT/'configs/wp11/goals/sever_mid.json'),
    'TRIM_rho10': ('trim01_flat_board', ROOT/'configs/wp11/goals/trim_bump_rho10.json'),
    'TESTB_4slices_solid': ('bar_flat_board', GOALS/'divide_5_slices.json'),
    'TESTB_4slices_slotted': ('bar_flat_board_slotted', GOALS/'divide_5_slices.json'),
    'TESTC_3bumps_rho10': ('bar_three_bumps_flat_board', GOALS/'remove_all_protrusions_rho10.json'),
}
VARIANTS = {
    'default': {},
    'table_full': {'table': {'center_xy_world_m': [0.0, 0.0], 'half_extents_xy_m': [0.9, 0.6]}},
    'board_thick_30mm': {'cutting_board': {'thickness_m': 0.03}},
}


def scene_for(name, overrides):
    scene = sc.load_json(SCENES/f'{name}.json')
    if overrides:
        workcell = scene['supports']['workcell']
        merged = deepcopy(workcell.get('overrides', {}))
        for key, value in overrides.items():
            merged.setdefault(key, {}).update(value)
        workcell['overrides'] = merged
        scene = place_on_board(scene)
        scene['scene_id'] = scene['scene_id']+'_'+'_'.join(sorted(overrides))
    return scene


def low_points(checker, task, qs, labels):
    top = sc.flat_board_record(task.scene)
    board_top = top['table']['top_world_z_m']+top['cutting_board']['thickness_m']
    out = {}
    for q, label in zip(qs, labels):
        checker.geo.set_q(q)
        for g, v in checker.geo.geoms.items():
            if v['arm'] != 0 or (v['group'] == 'arm0_links' and v['link'] in (0, 1)):
                continue
            V = checker.geo.geom_vertices(g)
            i = int(np.argmin(V[:, 2]))
            z = float(V[i, 2]-board_top)
            if v['role'] not in out or z < out[v['role']]['min_z_above_board_top_m']:
                out[v['role']] = {'min_z_above_board_top_m': z, 'geom': v['name'], 'label': label,
                                  'xy_world_m': V[i, :2].tolist(), 'y_range_world_m': [float(V[:, 1].min()), float(V[:, 1].max())]}
    return out


def run(task_key, variant):
    scene_name, goal = TASKS[task_key]
    start = time.perf_counter()
    try:
        scene = scene_for(scene_name, VARIANTS[variant])
        task = sc.compile_task(scene, goal)
    except Exception as error:   # report, never hide
        return {'task': task_key, 'variant': variant, 'status': 'COMPILE_ERROR', 'error': f'{type(error).__name__}: {error}'}
    checker = mp.MotionChecker(task)
    if len(task.protrusion_ids) >= 2:
        # The C2 model's single hull over several protrusions bridges the gaps between them (WP-11 Phase E
        # round 1, rule 2); multi-protrusion scenes use the staged per-protrusion-hull preflight (round 2).
        from robotic_cutting_sim.generalization import staged_preflight
        receipt, _ = staged_preflight(task, checker, geometry='staged_v2')
    else:
        receipt = mp.preflight_plan(task, task.home['joint_positions'], checker=checker)
    poses, labels, armed = mp.plan_samples(task)
    qs, q = [], np.asarray(task.home['joint_positions'], float)
    for pose, flag in zip(poses, armed):
        q, _ = checker.solve_ik(pose, q, flag)
        qs.append(q)
    geometry = task.workcell_geometry()
    path = receipt.get('path', {})
    worst = {k: {kk: vv for kk, vv in v.items() if kk in ('slack', 'value', 'threshold', 'label', 'pair')}
             for k, v in path.get('worst', {}).items()}
    edge_z = [float(task.reference_edge_pose_local(float(t))[2]) for t in np.asarray(task.edge_position_knots_local_txyz)[:, 0]]
    return {'task': task_key, 'variant': variant, 'scene_id': task.scene['scene_id'], 'task_id': task.task_id,
            'status': receipt['status'], 'reason': receipt.get('reason'), 'binding': path.get('binding'),
            'failing': path.get('failing'), 'worst': worst, 'ik_worst': receipt.get('ik_worst'),
            'ik_failure': receipt.get('ik_failure'), 'samples': receipt.get('samples'),
            'samples_by_phase': receipt.get('samples_by_phase'), 'preflight_geometry': receipt.get('geometry', 'declared'),
            'min_reference_edge_local_z_m': min(edge_z),
            'board_top_world_z_m': geometry['board_top_world_z_m'], 'table_box_world': geometry['table_box_world'],
            'board_boxes_local': geometry['board_boxes_local'], 'workpiece_world_z_m': task.world_from_workpiece_xyzw[2],
            'home_edge_world_z_m': float(task.reference_edge_pose_world(0.)[2]),
            'low_points': low_points(checker, task, qs, labels), 'wall_s': time.perf_counter()-start}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--variants', nargs='+', default=list(VARIANTS))
    a = p.parse_args()
    out = new_output(a.out)
    config = {'tasks': a.tasks, 'variants': a.variants, 'seed': 'deterministic (no RNG)'}
    write_json(out/'manifest.json', manifest({**config, 'command': sys.argv}, 'wp12_flat_board_preflight'))
    rows = []
    for variant in a.variants:
        for key in a.tasks:
            row = run(key, variant)
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ('task', 'variant', 'status', 'binding', 'failing', 'wall_s')}), flush=True)
            write_json(out/'result.json', {'rows': rows})
    write_json(out/'result.json', {'rows': rows, 'complete': True})


if __name__ == '__main__':
    main()
