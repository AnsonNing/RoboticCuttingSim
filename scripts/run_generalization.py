#!/usr/bin/env python3
"""WP-11 Phase E generalization round (predeclared in configs/wp11/generalization_round1.json).

Subcommands (every output directory/file is new; nothing is ever overwritten):

* ``preflight --out DIR``: stage 1 for every declared sample (or ``--only FAMILY[:SEED]``):
  build scene + goal by the declared rules, compile (solver + duration rules), C2
  motion preflight (Newton IK + MuJoCo collision, held ruler_knife_v2_y) over the full
  compiled schedule HOME -> ... -> RETURN_HOME, classify SCENE_INFEASIBLE / IK_FAILURE.
  Writes ``DIR/<sample_id>.json`` per sample and ``DIR/stage1_summary.json``.
* ``material --family F --seed S --out DIR``: stage 2 for one sample: kinematic
  ideal-tracking episode (formal 1 mm / 0.1 ms / 4 mm grid, law 8, tlmpm_v4 or v5
  by the solver rule) with the goal planner's WaypointPolicy, scored by the goal
  evaluator. Writes manifest.json, actions.jsonl, physics_state.npz, result.json.
  Results are KINEMATIC (runtime_source = kinematic_ideal_tracking), never live.
* ``queue --stage1 DIR --out-root DIR``: runs ``material`` for every stage-1 feasible
  sample in declared order, at most 2 at a time, and only while at most 2 other
  active jobs use GPU 0 (see the round file, stages.stage2).
* ``summarize --stage1 DIR --material-root DIR --out FILE``: per-sample classes and
  per-family tables (no exclusion), metrics, runtimes.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_generalization.py preflight \
      --out artifacts/wp11/phaseE/round1_stage1_001
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import generalization as G  # noqa: E402


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                else v.item() if isinstance(v, np.generic) else str(v)))


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.write_text(json.dumps(jsonable(payload), indent=1, allow_nan=True))


def git_state():
    run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'git_sha': run('rev-parse', 'HEAD'), 'git_dirty': bool(run('status', '--porcelain')), 'root': str(ROOT)}


def selected(config, only):
    ids = G.sample_ids(config)
    if not only:
        return ids
    out = []
    for token in only:
        family, _, seed = token.partition(':')
        out += [(f, s) for f, s in ids if f == family and (not seed or s == int(seed))]
    return out


# ----------------------------------------------------------------------------- stage 1

def last_task_phase(task):
    labels = [label for label in task.reference_phase_labels if label not in ('RETURN_HOME', 'DONE')]
    return labels[-1].split(':', 1)[0]


def stage1(config, family, seed, index=None):
    from robotic_cutting_sim import motion_preflight as mp
    from robotic_cutting_sim.scene_config import SceneConfigError, SceneInfeasible
    start = time.perf_counter()
    sample = G.build_sample(config, family, seed, index=index)
    row = {'family': family, 'seed': seed, 'sample_id': sample['sample_id'], 'sample': sample,
           'started': datetime.now().astimezone().isoformat()}
    try:
        task, info = G.compile_sample(config, sample)
    except SceneInfeasible as error:
        return {**row, 'class': 'SCENE_INFEASIBLE', 'reason': f'scene validation: {error}', 'stage': 'scene',
                'wall_s': time.perf_counter()-start}
    except (SceneConfigError, ValueError, NotImplementedError, RuntimeError) as error:
        return {**row, 'class': 'SCENE_INFEASIBLE', 'reason': f'planner/compile: {type(error).__name__}: {error}',
                'stage': 'compile', 'wall_s': time.perf_counter()-start}
    particles = int(task.sample_particles()[0].shape[0])
    row['compile'] = {**{k: v for k, v in info.items() if k not in ('scene', 'goal')}, 'task_id': task.task_id,
                      'config_sha256': task.config_sha256, 'particles': particles,
                      'control_steps': int(round(task.duration_s/task.control_dt_s))}
    row['scene_compiled'], row['goal_compiled'] = info['scene'], info['goal']
    checker = mp.MotionChecker(task)
    receipt, checkers = G.staged_preflight(task, checker)
    cls, reason = G.classify_preflight(receipt)
    row['preflight'] = receipt
    if cls is not None:
        return {**row, 'class': cls, 'reason': reason, 'stage': 'preflight', 'wall_s': time.perf_counter()-start,
                'checker_build_wall_s': checker.build_wall_s}
    # Informational (never changes the class): the Phase D physical return-to-home plan from the
    # joint state at the end of the last task phase (geometry after the last remove operation).
    try:
        phase = last_task_phase(task)
        checker = G.final_geometry_checker(task, checkers)
        prefix = mp.preflight_plan(checker.task, task.home['joint_positions'], until_phase=phase, checker=checker)
        if prefix['status'] == 'PASS':
            plan = mp.plan_return_home_for_task(checker.task, np.asarray(prefix['q_until_end'], float), checker=checker)
            row['return_home_lifecycle'] = {'from_phase': phase, 'status': plan.get('status'),
                                            'reason': plan.get('reason'), 'route': plan.get('route'),
                                            'samples': len(plan.get('q', []))}
        else:
            row['return_home_lifecycle'] = {'from_phase': phase, 'status': 'NOT_RUN', 'reason': prefix.get('reason')}
    except Exception as error:  # informational only
        row['return_home_lifecycle'] = {'status': 'ERROR', 'reason': repr(error)}
    return {**row, 'class': None, 'feasible': True, 'reason': 'preflight PASS', 'stage': 'preflight',
            'wall_s': time.perf_counter()-start, 'checker_build_wall_s': checker.build_wall_s}


def cmd_preflight(args):
    config = G.load_round(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    meta = {'command': sys.argv, 'round_id': config['round_id'], 'config': str(args.config), **git_state(),
            'started': datetime.now().astimezone().isoformat(), 'tmpdir': os.environ.get('TMPDIR')}
    write_new(out/'stage1_meta.json', meta)
    rows = []
    for family, seed in selected(config, args.only):
        try:
            row = stage1(config, family, seed)
        except Exception as error:
            row = {'family': family, 'seed': seed, 'sample_id': f'E1_{family}_s{seed}', 'class': 'NOT_RUN',
                   'reason': f'stage-1 error: {error!r}', 'traceback': traceback.format_exc()}
        write_new(out/f"{row['sample_id']}.json", row)
        rows.append(row)
        print(f"[E1] {row['sample_id']}: {row.get('class') or 'FEASIBLE'} | {row['reason']} | "
              f"{row.get('wall_s', 0):.1f} s", flush=True)
    summary = {**meta, 'finished': datetime.now().astimezone().isoformat(),
               'samples': [{k: r.get(k) for k in ('family', 'seed', 'sample_id', 'class', 'reason', 'stage', 'wall_s')}
                           | {'feasible': bool(r.get('feasible')), 'task_id': r.get('compile', {}).get('task_id'),
                              'particles': r.get('compile', {}).get('particles'),
                              'duration_s': r.get('compile', {}).get('duration_s'),
                              'binding': (r.get('preflight') or {}).get('path', {}).get('binding'),
                              'return_home_lifecycle': (r.get('return_home_lifecycle') or {}).get('status')}
                           for r in rows]}
    write_new(out/'stage1_summary.json', summary)
    return 0


# ----------------------------------------------------------------------------- stage 2

def cmd_material(args):
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.recording import manifest, new_output
    from robotic_cutting_sim.scene_config import make_configured_kinematic_env
    config = G.load_round(args.config)
    common = config['common']
    sample = G.build_sample(config, args.family, args.seed, index=args.dev_index)
    task, info = G.compile_sample(config, sample)
    if args.expect_task_id and task.task_id != args.expect_task_id:
        raise SystemExit(f'Compiled {task.task_id}, stage 1 recorded {args.expect_task_id}')
    out = new_output(args.out)
    steps_total = int(round(task.duration_s/task.control_dt_s))
    stride = G.history_stride(config, steps_total)
    options = goal_planner.planner_options(task.goal, None)
    limits = goal_planner.action_limits(options, task.scene)
    run_config = {'round_id': config['round_id'], 'family': args.family, 'seed': args.seed, 'sample_id': sample['sample_id'],
                  'task_id': task.task_id, 'config_sha256': task.config_sha256, 'device': args.device,
                  'material_dt_s': common['material_dt_s'], 'material_seed': common['material_seed'],
                  'history_stride': stride, 'control_steps': steps_total, 'solver_module': info['solver_module'],
                  'max_cuts': info['max_cuts'], 'cutting_law': common['solver']['cutting_law'],
                  'particle_spacing_m': task.particle_spacing_m, 'grid_dx_m': task.grid_dx_m,
                  'runtime_source': 'kinematic_ideal_tracking', 'scene': info['scene'], 'goal': info['goal'],
                  'planner_options': options, 'draws': sample['draws'], 'derived': sample['derived']}
    (out/'manifest.json').write_text(json.dumps(jsonable(manifest(run_config, 'kinematic_ideal_tracking')), indent=1))
    run_config['declared_sample'] = sample['declared']
    result = {'status': 'RUNNING', 'declared_sample': sample['declared'], **{k: run_config[k] for k in ('round_id', 'family', 'seed', 'sample_id', 'task_id',
                                                                    'config_sha256', 'solver_module', 'max_cuts',
                                                                    'history_stride', 'control_steps', 'runtime_source')}}
    start = time.monotonic()
    env = make_configured_kinematic_env(task, seed=common['material_seed'], device=args.device,
                                        material_dt=common['material_dt_s'], delta_frame='world', ik_failure='reject',
                                        history_stride=stride, action_limits=limits)
    observation, _ = env.reset(seed=common['material_seed'])
    setup_wall = time.monotonic()-start
    plan = goal_planner.plan(task.scene, task.goal, task.tool_profile, options)
    policy = WaypointPolicy(env.task, plan, limits, **goal_planner.policy_options(options))
    material = env.backend.material
    bed = material.bed
    dt = task.control_dt_s
    finite, min_det, max_speed = True, float('inf'), 0.
    limit = steps_total if args.steps is None else min(args.steps, steps_total)
    loop_start = time.monotonic()
    with (out/'actions.jsonl').open('w') as stream:
        for step in range(limit):
            action = policy(observation)
            observation, _, terminated, truncated, info_step = env.step(action)
            diag = env.backend.last_diagnostics['material']
            stroke = env.backend.stroke_history[-1] if env.backend.stroke_history else {}
            if (step+1) % 10 == 0 or step == limit-1 or terminated or truncated:
                x, v = bed.x.numpy(), bed.v.numpy()
                det = np.linalg.det(bed.F.numpy().astype(float))
                ok = bool(np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(det).all())
                finite &= ok
                if np.isfinite(det).any():
                    min_det = min(min_det, float(np.nanmin(det)))
                max_speed = max(max_speed, float(np.nanmax(np.linalg.norm(v, axis=1))))
            blade = np.asarray(diag['current_material_reaction_impulse_world'], float)/dt
            stream.write(json.dumps(jsonable({
                'step': step+1, 'time_s': env.time_s, 'phase': policy.phase, 'cut_enabled': bool(action['cut_enabled']),
                'cut_index': int(diag['cut_index']), 'edge_pose_world': env.backend.edge,
                'blade_force_world_N': blade[:3], 'labelled_count': stroke.get('labelled_count'),
                'engaged': stroke.get('engaged'), 'workspace_clamped': bool((info_step.get('action_report') or {})
                                                                           .get('workspace_clamped', False))}))+'\n')
            if terminated or truncated:
                break
            if step % 100 == 0:
                print(f'[E2] {sample["sample_id"]} step {step+1}/{limit} t={env.time_s:.2f}s phase={policy.phase} '
                      f'wall={time.monotonic()-loop_start:.0f}s', flush=True)
    episode_wall = time.monotonic()-loop_start
    slots = None
    solver_invalid = None if finite else 'non-finite particle state or deformation gradient'
    if hasattr(bed, 'check_slots'):
        try:
            slots = bed.check_slots()
        except Exception as error:  # SlotOverflowError: the affected substeps are invalid
            solver_invalid, slots = f'slot overflow: {error}', {'overflow': str(error)}
    eval_start = time.monotonic()
    try:
        evaluation = env.evaluate()
    except Exception as error:  # preserved as a result, never lost
        evaluation = {'verdict': None, 'status': 'NOT_RUN', 'error': repr(error), 'traceback': traceback.format_exc()}
    evaluation_wall = time.monotonic()-eval_start
    history = env._history
    np.savez_compressed(out/'physics_state.npz',
                        time_s=np.asarray([s['time_s'] for s in history], float),
                        positions=np.asarray([s['positions'] for s in history], np.float32),
                        velocities=np.asarray([s['velocities'] for s in history], np.float32),
                        codes=np.asarray([s['codes'] for s in history]),
                        mass_kg=material.masses, reference_positions=material.world_points,
                        reference_indices=material.lattice, reference_stencil_nodes=bed.nodes.numpy(),
                        reference_stencil_weights=bed.weights.numpy(), reference_stencil_gradients=bed.gradients.numpy(),
                        code_encoding=getattr(material, 'code_encoding', 'ternary'),
                        max_cuts=int(getattr(material, 'max_cuts', 1)), **env.backend.law4_stroke_arrays())
    try:
        metrics = G.episode_metrics(task.goal, evaluation)
    except Exception as error:  # metrics are report-only
        metrics = {'error': repr(error), 'traceback': traceback.format_exc()}
    result.update(status='COMPLETE', policy_done=bool(policy.done), steps_run=limit, partial=limit < steps_total,
                  evaluation=evaluation, metrics=metrics, solver_invalid=solver_invalid, slots=slots,
                  stability={'finite': finite, 'min_det_F_sampled': min_det, 'max_speed_m_s_sampled': max_speed,
                             'sampling': 'every 10 control steps'},
                  wall_s={'setup': setup_wall, 'episode': episode_wall, 'evaluation': evaluation_wall,
                          'per_simulated_second': episode_wall/max(env.time_s, 1e-9)},
                  simulated_s=env.time_s, particles=int(bed.n), substeps_per_control_step=int(material.substeps),
                  history_frames=len(history), physics_state_bytes=(out/'physics_state.npz').stat().st_size,
                  finished=datetime.now().astimezone().isoformat())
    cls, reason = G.classify_material(result)
    result.update({'class': cls, 'class_reason': reason,
                   'label': 'KINEMATIC ideal-tracking result; not a live result (no robot, grasp, slip, dynamics or cameras)'})
    env.close()
    (out/'result.json').write_text(json.dumps(jsonable(result), indent=1))
    print(f'[E2] {sample["sample_id"]}: {cls} | {reason} | episode {episode_wall:.0f} s', flush=True)
    return 0


# ----------------------------------------------------------------------------- queue

def gpu0_other_jobs(mine):
    """PIDs of active jobs on GPU 0 not started by this queue (see the round file, stages.stage2)."""
    query = lambda *a: subprocess.run(['nvidia-smi', *a], capture_output=True, text=True).stdout
    uuid0 = next((line.split(',')[1].strip() for line in query('--query-gpu=index,uuid', '--format=csv,noheader').splitlines()
                  if line.split(',')[0].strip() == '0'), None)
    pids = [int(line.split(',')[0]) for line in query('--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader').splitlines()
            if line.strip() and line.split(',')[1].strip() == uuid0]
    active = set()
    for line in query('pmon', '-c', '3', '-s', 'u').splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[0] == '0' and parts[1].isdigit() and parts[3].isdigit() and int(parts[3]) > 0:
            active.add(int(parts[1]))
    others = []
    for pid in pids:
        if pid in mine:
            continue
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            cmdline = ''
        if 'RoboticCuttingSim' in cmdline or pid in active:
            others.append(pid)
    return others


def cmd_queue(args):
    config = G.load_round(args.config)
    stage1 = json.loads((Path(args.stage1)/'stage1_summary.json').read_text())
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    pending = [row for row in stage1['samples'] if row['feasible']]
    if args.only:
        keep = set(selected(config, args.only))
        pending = [row for row in pending if (row['family'], row['seed']) in keep]
    running = {}
    log = open(args.queue_log, 'a')

    def note(text):
        log.write(f'{datetime.now().astimezone().isoformat()} {text}\n')
        log.flush()
    note(f'QUEUE start {len(pending)} feasible samples; max {args.max_concurrent} concurrent; '
         f'launch when <= {args.max_other} other active GPU-0 jobs; {git_state()}')
    last_wait = None
    while pending or running:
        for pid, (proc, row, began) in list(running.items()):
            if proc.poll() is not None:
                note(f"END {row['sample_id']} rc={proc.returncode} wall={time.monotonic()-began:.0f}s")
                del running[pid]
        if pending and len(running) < args.max_concurrent:
            others = gpu0_other_jobs(set(running))
            if len(others) <= args.max_other:
                row = pending.pop(0)
                out = root/row['sample_id']
                if out.exists():
                    note(f"SKIP {row['sample_id']} (output exists; preserved, not rerun)")
                    continue
                log_path = logs/f"phaseE_{row['sample_id']}_001.log"
                command = [sys.executable, str(Path(__file__).resolve()), 'material', '--config', str(args.config),
                           '--family', row['family'], '--seed', str(row['seed']), '--device', args.device,
                           '--expect-task-id', row['task_id'], '--out', str(out)]
                proc = subprocess.Popen(command, cwd=str(ROOT), stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
                                        env={**os.environ, 'TMPDIR': os.environ.get('TMPDIR', '/mnt/data/ningan/tmp')})
                running[proc.pid] = (proc, row, time.monotonic())
                note(f"START {row['sample_id']} pid={proc.pid} others={others} log={log_path}")
                last_wait = None
                time.sleep(20)
                continue
            if last_wait is None or time.monotonic()-last_wait > 600:
                note(f'WAIT {len(others)} other active GPU-0 jobs {others}; {len(running)} mine running; '
                     f'{len(pending)} pending')
                last_wait = time.monotonic()
        time.sleep(args.poll_s)
    note('QUEUE done')
    return 0


# ----------------------------------------------------------------------------- summary

def cmd_summarize(args):
    config = G.load_round(args.config)
    stage1_dir, material_root = Path(args.stage1), Path(args.material_root)
    rows = []
    for family, seed in G.sample_ids(config):
        sample_id = f'E1_{family}_s{seed}'
        s1 = json.loads((stage1_dir/f'{sample_id}.json').read_text())
        row = {'family': family, 'seed': seed, 'sample_id': sample_id,
               'goal_id': s1['sample']['goal']['goal_id'], 'goal_type': s1['sample']['goal']['type'],
               'goal_description': s1['sample']['goal']['description'], 'draws': {d['name']: d['value'] for d in s1['sample']['draws']},
               'derived': s1['sample']['derived'], 'task_id': s1.get('compile', {}).get('task_id'),
               'particles': s1.get('compile', {}).get('particles'), 'duration_s': s1.get('compile', {}).get('duration_s'),
               'solver_module': s1.get('compile', {}).get('solver_module'), 'digits': s1.get('compile', {}).get('digits_used'),
               'stage1': {'class': s1.get('class'), 'reason': s1['reason'], 'wall_s': s1.get('wall_s'),
                          'binding': (s1.get('preflight') or {}).get('path', {}).get('binding'),
                          'worst_slack': {k: v['slack'] for k, v in ((s1.get('preflight') or {}).get('path', {}).get('worst') or {}).items()},
                          'ik_worst': (s1.get('preflight') or {}).get('ik_worst'),
                          'return_home_lifecycle': s1.get('return_home_lifecycle')}}
        if s1.get('class'):
            row['class'], row['reason'] = s1['class'], s1['reason']
        else:
            path = material_root/sample_id/'result.json'
            result = json.loads(path.read_text()) if path.exists() else None
            if result is not None and result.get('task_id') != row['task_id']:
                result = {'status': 'NOT_RUN', 'reason': f"stage-2 task id {result.get('task_id')} != stage-1 {row['task_id']}"}
            if result is None:
                result = {'status': 'NOT_RUN', 'reason': ('stage-2 output exists without result.json (crashed or running)'
                                                          if (material_root/sample_id).exists() else 'stage 2 not run')}
            row['class'], row['reason'] = G.classify_material(result)
            if result.get('status') == 'COMPLETE':
                ev = result['evaluation']
                row['stage2'] = {'verdict': ev.get('verdict'), 'failed_checks': G.failed_checks(ev),
                                 'integrity': ev.get('integrity', {}).get('status'),
                                 'metrics': result['metrics'], 'wall_s': result['wall_s'],
                                 'simulated_s': result['simulated_s'], 'particles': result['particles'],
                                 'history_frames': result['history_frames'],
                                 'physics_state_mb': result['physics_state_bytes']/1e6, 'slots': result.get('slots'),
                                 'stability': result['stability'], 'artifact': str(material_root/sample_id),
                                 'git': {k: v for k, v in json.loads((material_root/sample_id/'manifest.json').read_text()).items()
                                         if k in ('git_sha', 'git_dirty')}}
        rows.append(row)
    families = list(config['families'])
    out = {'schema': 'rcs_generalization_round_evidence_v1', 'round_id': config['round_id'], 'config': str(args.config),
           'generated': datetime.now().astimezone().isoformat(), 'command': sys.argv,
           'stage1_dir': str(stage1_dir), 'material_root': str(material_root),
           'label': 'Stage-2 classes are KINEMATIC ideal-tracking results, never live results; GRASP_FAILURE is not '
                    'exercised in the kinematic stage.',
           'tables': G.summarize(rows, families), 'samples': rows}
    write_new(args.out, out)
    for name, table in out['tables'].items():
        print(name, json.dumps({k: table[k] for k in ('samples', 'counts', 'feasible_fraction',
                                                       'kinematic_task_success_among_feasible', 'overall_kinematic_success')}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=G.ROUND1)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('preflight')
    a.add_argument('--out', required=True)
    a.add_argument('--only', nargs='*')
    b = sub.add_parser('material')
    b.add_argument('--family', required=True)
    b.add_argument('--seed', type=int, required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--device', default='cuda:0')
    b.add_argument('--expect-task-id')
    b.add_argument('--steps', type=int, help='Smoke only: stop after this many control steps (result marked partial)')
    b.add_argument('--dev-index', type=int, help='Development seeds outside the round only (declared seeds keep their index)')
    c = sub.add_parser('queue')
    c.add_argument('--stage1', required=True)
    c.add_argument('--out-root', required=True)
    c.add_argument('--log-dir', required=True)
    c.add_argument('--queue-log', required=True)
    c.add_argument('--device', default='cuda:0')
    c.add_argument('--max-concurrent', type=int, default=2)
    c.add_argument('--max-other', type=int, default=2)
    c.add_argument('--poll-s', type=float, default=60.)
    c.add_argument('--only', nargs='*')
    d = sub.add_parser('summarize')
    d.add_argument('--stage1', required=True)
    d.add_argument('--material-root', required=True)
    d.add_argument('--out', required=True)
    args = p.parse_args(argv)
    for key in ('config',):
        setattr(args, key, Path(getattr(args, key)).resolve())
    return {'preflight': cmd_preflight, 'material': cmd_material, 'queue': cmd_queue, 'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
