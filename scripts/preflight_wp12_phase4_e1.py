#!/usr/bin/env python3
"""WP-12 Phase 4 E1.3: kinematic reachability of every E1 row (configs/wp12/phase4_e1_protocol_v1.json).

For each declared row, the scene is compiled exactly as the physics runner compiles it (blade_master_flat_board_v1
solid board or its slotted variant, max_cuts 1). The plan is then checked with the WP-11/WP-12 motion preflight
(``motion_preflight.MotionChecker``: runtime Newton IK with ruler_knife_v2_y's hand_from_edge, C2 MuJoCo distances
on the Franka colliders, finger hulls, handle/blade hulls, wrist D405s, D455, board and table boxes, the declared
bar and the parked Franka 1):

  HOME (workcell q_home) -> APPROACH (30 mm before the armed start along -d) -> PRE_CUT (armed start)
  -> CUT (armed, to the end pose) -> HOLD -> WITHDRAW (to the start) -> LIFT (to APPROACH)

Samples are spaced at most 2 mm and 1 deg; IK is chained from q_home. Blade-workpiece and blade-board thresholds
follow ``motion_preflight.phase_mode`` and the workcell record: free 2 mm / 1 mm; PRE_CUT >= 0; CUT, HOLD and
WITHDRAW contact allowed for the workpiece and >= 0 for the board. Geometry only: no dynamics, no material, no
grasp holding. A PASS means **kinematically reachable**, never validated cutting.

  .venv/bin/python scripts/preflight_wp12_phase4_e1.py --out artifacts/wp12/phase4_e1_001/reachability
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

from run_wp12_phase4_e1 import (PROTOCOL, compile_row_task, edge_path, edge_rotation, git_state,  # noqa: E402
                                jsonable, protocol_rows)

APPROACH_BACK_M = .030


def world_pose(task, p_local, rotation):
    T = np.asarray(task.T_world_workpiece, float)
    return np.r_[T[:3, :3]@p_local+T[:3, 3], (Rotation.from_matrix(T[:3, :3])*rotation).as_quat()]


def plan(task, row):
    """Knot poses (world) and the label/armed flag of the motion that ends at each knot."""
    path = edge_path(row, float(task.scene['workpiece']['base_box_m']['height_m']))
    rotation = edge_rotation(row['tau_deg'])
    start, end = path['start'], path['end']
    approach = start-APPROACH_BACK_M*path['d']
    home = np.asarray(task.reference_edge_pose_world(0.), float)
    knots = [(home, 'HOME', False), (world_pose(task, approach, rotation), 'APPROACH', False),
             (world_pose(task, start, rotation), 'PRE_CUT', False), (world_pose(task, end, rotation), 'CUT', True),
             (world_pose(task, end, rotation), 'HOLD', False), (world_pose(task, start, rotation), 'WITHDRAW', False),
             (world_pose(task, approach, rotation), 'LIFT', False)]
    return knots, path


def dense(knots, max_translation=.002, max_rotation_deg=1.):
    from robotic_cutting_sim.episode_lifecycle import slerp_poses
    poses, labels, armed = [knots[0][0]], [knots[0][1]], [knots[0][2]]
    for pose, label, flag in knots[1:]:
        a = poses[-1]
        distance = np.linalg.norm(pose[:3]-a[:3])
        angle = np.degrees((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude())
        n = max(1, int(np.ceil(distance/max_translation-1e-9)), int(np.ceil(angle/max_rotation_deg-1e-9)))
        if distance < 1e-12 and angle < 1e-9:
            n = 1   # a HOLD knot: one sample at the same pose
        for p in slerp_poses(a, pose, n):
            poses.append(p)
            labels.append(label)
            armed.append(flag)
    return np.asarray(poses), labels, armed


def check_row(row, checkers):
    from robotic_cutting_sim import motion_preflight as mp
    start = time.perf_counter()
    task, scene = compile_row_task(row)
    if row['scene'] not in checkers:
        checkers[row['scene']] = mp.MotionChecker(task)
    checker = checkers[row['scene']]
    knots, path = plan(task, row)
    poses, labels, armed = dense(knots)
    th = checker.thresholds
    q_home = np.asarray(task.home['joint_positions'], float)
    qs, seed, failure = [], q_home, None
    ik_worst = {'position_error_m': 0., 'rotation_error_rad': 0.}
    for i, (pose, flag) in enumerate(zip(poses, armed)):
        q, info = checker.solve_ik(pose, seed, flag)
        tol_p = th['ik_active_position_m'] if flag else th['ik_position_m']
        tol_r = th['ik_active_rotation_rad'] if flag else th['ik_rotation_rad']
        ik_worst = {k: max(ik_worst[k], float(info[k])) for k in ik_worst}
        if info['position_error_m'] > tol_p or info['rotation_error_rad'] > tol_r:
            failure = {'sample': i, 'label': labels[i], 'ik_info': info}
            break
        qs.append(q)
        seed = q
    out = {'row': row['id'], 'tau_deg': row['tau_deg'], 'scene': row['scene'], 'task_id': task.task_id,
           'samples': int(len(poses)), 'ik_worst': ik_worst, 'q_home': q_home.tolist(),
           'knots_world_xyzw': [k[0].tolist() for k in knots], 'knot_labels': [k[1] for k in knots],
           'armed_start_local_m': path['start'].tolist(), 'armed_end_local_m': path['end'].tolist()}
    if failure is not None:
        out.update(status='FAIL', reason='IK_FAILURE', ik_failure=failure, wall_s=time.perf_counter()-start)
        return out
    modes = [mp.phase_mode(label, flag) for label, flag in zip(labels, armed)]
    result = checker.check_joint_path(qs, modes, labels)
    worst = result['worst']
    out.update(status=result['status'], reason=None if result['status'] == 'PASS' else 'SCENE_INFEASIBLE',
               failing=result['failing'], binding=result['binding'],
               binding_row=worst.get(result['binding']) if result['binding'] else None,
               slack_by_check={k: v['slack'] for k, v in worst.items()},
               worst=worst, modes=result['blade_workpiece_modes'],
               joint_limit_margin_rad=worst.get('joint_limit_margin', {}).get('value'),
               wall_s=time.perf_counter()-start)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows', nargs='*', help='subset of row ids (default: all declared)')
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    p.add_argument('--dev', action='store_true')
    a = p.parse_args()
    parent = a.out.resolve().parent.name
    if a.dev != parent.startswith('phase4_e1_dev_') or not parent.startswith('phase4_e1_'):
        p.error('formal output under artifacts/wp12/phase4_e1_NNN/, --dev output under phase4_e1_dev_NNN/')
    rows = protocol_rows(a.protocol)
    names = a.rows or list(rows)
    unknown = [n for n in names if n not in rows]
    if unknown:
        p.error(f'undeclared rows {unknown}')
    a.out.mkdir(parents=True, exist_ok=False)
    base = {'tool': 'scripts/preflight_wp12_phase4_e1.py', 'label': 'DEV' if a.dev else 'E1.3_FORMAL', 'command': sys.argv,
            **git_state(), 'protocol_sha256': hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest(),
            'scope': 'Geometry only (IK + C2 collision preflight); PASS = kinematically reachable, not validated cutting'}
    checkers, results, start = {}, {}, time.perf_counter()
    for name in names:
        try:
            results[name] = check_row(rows[name], checkers)
        except Exception as error:  # recorded, never hidden
            results[name] = {'row': name, 'status': 'ERROR', 'error': repr(error)}
        r = results[name]
        print(name, r['status'], r.get('binding'), None if not r.get('binding_row') else round(r['binding_row']['slack']*1e3, 3),
              'mm/rad', flush=True)
        (a.out/f'{name}.json').write_text(json.dumps(jsonable(r), indent=1))
    summary = {**base, 'wall_s': time.perf_counter()-start,
               'rows': {n: {k: r.get(k) for k in ('status', 'reason', 'failing', 'binding', 'joint_limit_margin_rad', 'samples',
                                                   'ik_worst')} | {'binding_slack': (r.get('binding_row') or {}).get('slack')}
                        for n, r in results.items()}}
    (a.out/'summary.json').write_text(json.dumps(jsonable(summary), indent=1))


if __name__ == '__main__':
    main()
