#!/usr/bin/env python3
"""C2 geometric preflight of an open-loop ``rcs_action_program_v1`` on a configured task (CPU ``.venv``).

The commanded edge path is reproduced exactly as the env bounds it (the
task's ActionLimits via ``control.bound_target`` / ``resolve_delta`` from the
reset edge pose), then IK-chained from the scene HOME joints with the
runtime's Newton IK and checked with the WP-11 C2 MuJoCo collision model
(``motion_preflight.MotionChecker``; blade-workpiece modes from the program
labels: SKIM/HOLD/WITHDRAW contact allowed, PRE_CUT contact free, the rest
free). Geometry only: no dynamics, grasp or material.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/preflight_action_program.py \
      --scene trim01_wp11_workcell --goal trim_bump_rho10 --planner-options '{...}' \
      --program configs/wp11/action_programs/trim_rho10_handwritten_v1.json --out <new>.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import dataset as ds  # noqa: E402
from robotic_cutting_sim.policy_env import jsonable  # noqa: E402


def commanded_path(task, rows):
    from robotic_cutting_sim.control import bound_target, resolve_delta
    limits = task.action_limits()
    edge = np.asarray(task.reference_edge_pose_world(0.), float)
    poses, labels, armed, clamped = [edge.copy()], [rows[0][0] if rows else 'START'], [False], 0
    for label, action in rows:
        if action['type'] == 'edge_pose':
            edge, report = bound_target(edge, action['target'], limits, task.T_world_workpiece)
        else:
            edge, report = resolve_delta(edge, action['target'], limits, frame=action.get('frame', 'world'),
                                         T_world_workpiece=task.T_world_workpiece)
        clamped += any(report.get(k, False) for k in ('translation_clamped', 'rotation_clamped', 'workspace_clamped'))
        poses.append(edge.copy())
        labels.append(label)
        armed.append(bool(action.get('cut_enabled', False)))
    return np.asarray(poses), labels, armed, clamped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', required=True)
    p.add_argument('--goal', required=True)
    p.add_argument('--planner-options')
    p.add_argument('--program', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; never overwrite')
    from robotic_cutting_sim.motion_preflight import MotionChecker, phase_mode
    from robotic_cutting_sim.scene_config import compile_task
    start = time.perf_counter()
    task = compile_task(a.scene, a.goal, planner_options=json.loads(a.planner_options) if a.planner_options else None)
    if task.home is None:
        raise SystemExit('The scene declares no HOME joints to chain IK from')
    program = ds.load_action_program(a.program)
    rows = ds.expand_action_program(program, task.control_dt_s, int(round(task.duration_s/task.control_dt_s)))
    poses, labels, armed, clamped = commanded_path(task, rows)
    keep = [0]+[i for i in range(1, len(poses)) if np.abs(poses[i]-poses[keep[-1] if False else i-1]).max() > 1e-12]
    checker = MotionChecker(task)
    th = checker.thresholds
    q_home = np.asarray(task.home['joint_positions'], float)
    qs, seed, failure, worst = [], q_home, None, dict(position_error_m=0., rotation_error_rad=0.)
    for i in keep:
        q, info = checker.solve_ik(poses[i], seed, armed[i])
        tol_p = th['ik_active_position_m'] if armed[i] else th['ik_position_m']
        tol_r = th['ik_active_rotation_rad'] if armed[i] else th['ik_rotation_rad']
        worst = {k: max(worst[k], float(info[k])) for k in worst}
        if info['position_error_m'] > tol_p or info['rotation_error_rad'] > tol_r:
            failure = dict(sample=int(i), label=labels[i], ik_info=info, position_tolerance_m=tol_p, rotation_tolerance_rad=tol_r)
            break
        qs.append(q)
        seed = q
    receipt = dict(version='phaseG_action_program_preflight_v1', task_id=task.task_id, config_sha256=task.config_sha256,
                   program=str(a.program), program_sha256=ds.sha256_file(a.program), program_id=program.get('program_id'),
                   steps=len(rows), path_samples=len(keep), clamped_steps=int(clamped), ik_worst=worst,
                   q_home=q_home.tolist(), checker_build_wall_s=checker.build_wall_s)
    if failure is not None:
        receipt.update(status='FAIL', reason='IK_FAILURE', ik_failure=failure)
    else:
        modes = [phase_mode(labels[i], armed[i]) for i in keep]
        path = checker.check_joint_path(qs, modes, [labels[i] for i in keep])
        receipt.update(status=path['status'], reason=None if path['status'] == 'PASS' else 'SCENE_INFEASIBLE', path=path,
                       q_end=qs[-1].tolist())
    receipt['wall_s'] = time.perf_counter() - start
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(jsonable(receipt), indent=2) + '\n')
    print(json.dumps({k: receipt.get(k) for k in ('status', 'reason', 'task_id', 'steps', 'path_samples', 'wall_s')}))
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
