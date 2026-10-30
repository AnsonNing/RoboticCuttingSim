#!/usr/bin/env python3
"""WP-11 Phase D motion preflight in the CPU .venv (geometry only; C2 checker).

Modes (request JSON ``{"mode": ..., "payload": <configured_task_payload>, ...}``):

* ``plan``: IK chain from the scene HOME over the compiled reference plan and the
  joint-path clearance check. ``until_phase`` (optional) also checks the prefix
  that will actually execute; the receipt reports both.
  Scenes with two or more protrusions use the round-1 staged geometry
  (``generalization.staged_preflight``; receipt key ``preflight_geometry``); scenes
  with zero or one protrusion keep exactly the declared-geometry check.
* ``return-home``: plan the physical return-to-home from ``q_start`` (7 rad):
  Cartesian lift, hand-down rotation, joint-space move to q_home, preflighted.
* ``check-path``: clearance check of measured joints ``q`` (N x 7) with ``modes``.

Direct use (writes a fresh receipt; never overwrites):

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/wp11_motion_preflight.py \
      --scene trim01_wp11_workcell --goal trim_bump_rho10 --until-phase REORIENT \
      --out artifacts/wp11/phaseD/preflight_001.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                else v.item() if isinstance(v, np.generic) else str(v)))


def run(request):
    from robotic_cutting_sim import motion_preflight as mp
    from robotic_cutting_sim.scene_config import task_from_payload
    start = time.perf_counter()
    task = task_from_payload(request['payload'])
    checker = mp.MotionChecker(task)
    mode = request['mode']
    if mode == 'plan':
        home = task.home
        if home is None:
            return dict(status='FAIL', reason='SCENE_INFEASIBLE', detail='The scene declares no HOME to start the plan from')
        until = request.get('until_phase')
        if len(getattr(task, 'protrusion_ids', ())) >= 2:
            # Two or more protrusions: the round-1 staged geometry (generalization.staged_preflight; each plan chunk
            # without the protrusions removed by earlier remove operations, same C2 thresholds). One convex hull over
            # all protrusions bridges their gaps and keeps removed ones (Phase E live_001 Test C SCENE_INFEASIBLE).
            from robotic_cutting_sim.generalization import staged_preflight
            full, _ = staged_preflight(task, checker)
            prefix = None if until is None else staged_preflight(task, checker, until_phase=until)[0]
            geometry = dict(rule='staged_multi_protrusion_v1', helper='generalization.staged_preflight',
                            protrusions=len(task.protrusion_ids))
        else:
            # Zero or one protrusion: exactly the Phase D/C5 check, unchanged.
            full = mp.preflight_plan(task, home['joint_positions'], checker=checker)
            prefix = None if until is None else mp.preflight_plan(task, home['joint_positions'], until_phase=until, checker=checker)
            geometry = None
        required = prefix if prefix is not None else full
        out = dict(status=required['status'], reason=required.get('reason'), required=('prefix' if prefix else 'full'),
                   full_plan=full, executed_prefix=prefix)
        if geometry is not None:
            out['preflight_geometry'] = geometry
    elif mode == 'return-home':
        plan = mp.plan_return_home_for_task(task, np.asarray(request['q_start'], float), checker=checker)
        out = dict(plan, status=plan['status'], q=np.asarray(plan['q']).tolist())
    elif mode == 'check-path':
        out = checker.check_joint_path(np.asarray(request['q'], float), request['modes'], request.get('labels'))
    else:
        raise ValueError(f'Unknown mode {mode!r}')
    return dict(out, mode=mode, task_id=task.task_id, config_sha256=task.config_sha256,
                checker_build_wall_s=checker.build_wall_s, wall_s=time.perf_counter()-start)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--request', type=Path, help='Request JSON (the runtime uses this)')
    p.add_argument('--scene', help='rcs_scene_v1 config (direct use)')
    p.add_argument('--goal', help='rcs_goal_v1 config (direct use)')
    p.add_argument('--until-phase')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        p.error(f'{a.out} exists; preflight receipts are never overwritten')
    if a.request is not None:
        request = json.loads(a.request.read_text())
    else:
        if not (a.scene and a.goal):
            p.error('Give --request, or --scene and --goal')
        from robotic_cutting_sim.scene_config import compile_task, configured_task_payload
        request = dict(mode='plan', payload=configured_task_payload(compile_task(a.scene, a.goal)), until_phase=a.until_phase)
    out = run(request)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(jsonable(out), indent=1, allow_nan=False))
    print(json.dumps({key: out.get(key) for key in ('mode', 'status', 'reason', 'task_id', 'wall_s')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
