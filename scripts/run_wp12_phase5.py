#!/usr/bin/env python3
"""WP-12 Phase 5 generalization benchmark runner (config: configs/wp12/benchmark_v1_DRAFT.json until frozen).

Every output path is new; nothing is ever overwritten (a resumed stage skips and preserves existing outputs).
Subcommands:

* ``samples --split S --out FILE``: the declared samples of a split (scene, goal, draws, envelope parameters). CPU.
* ``feasibility --split S --out DIR [--only ID ...] [--resume] [--no-preflight]``: pre-execution class of every
  sample (generalization_wp12.classify_feasibility: scene geometry -> envelope -> C2 motion preflight).
  ``DIR/<sample_id>.json`` per sample and ``DIR/feasibility_summary[_NNN].json``. CPU (.venv).
* ``kinematic --sample-id ID --out DIR [--device cuda:1] [--steps N] [--surface-profile P]``: one kinematic
  ideal-tracking episode (goal planner WaypointPolicy, goal_eval_v2a, physical/geometry quality, runtime). Results
  are KINEMATIC (runtime_source = kinematic_ideal_tracking), never live. ``--steps`` marks a partial smoke.
* ``queue --feasibility DIR --out-root DIR --log-dir DIR --queue-log FILE``: ``kinematic`` once for every
  valid_feasible sample in declared order (resumable; <= N concurrent per device; waits while other jobs use a GPU).
* ``native-select --feasibility DIR --out FILE``: the declared Native stratified sub-sample (>= 3 per stratum).
* ``native-prepare --sample-id ID --out DIR`` / ``native-launch --sample-id ID --out DIR``: one Native Isaac Sim
  episode through scripts/run_wp10_live.py --runtime isaac_sim --scene/--goal (WP-12 Phase 2b pattern).
* ``summarize --feasibility DIR [--kinematic-root DIR] [--native-root DIR] --out FILE``: feasibility rate, success
  among feasible, overall success, class distribution, Wilson 95 % intervals and denominators per stratum / family /
  range set; every declared sample counted once.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase5.py feasibility --split dev \\
      --out artifacts/wp12/phase5_dev_001/feasibility
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from robotic_cutting_sim import generalization as G  # noqa: E402
from robotic_cutting_sim import generalization_wp12 as W  # noqa: E402
import run_generalization as RG  # noqa: E402  (WP-11 Phase E helpers: jsonable, write_new, git_state, gpu_other_jobs)

jsonable, write_new, git_state = RG.jsonable, RG.write_new, RG.git_state


def next_free(path):
    path = Path(path)
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f'{path.stem}_{i:03d}{path.suffix}')
        if not candidate.exists():
            return candidate
    raise FileExistsError(path)


def declared(config, split):
    return [(W.make_sample_id(config, f, rs, split, s), {'family': f, 'range_set': rs, 'split': split, 'stratum': f'{f}_{rs}',
                                                          'seed': s, 'variant': None, 'ood_dimensions': None})
            for f, rs, s, _ in W.sample_ids(config, split)]


def selected_ids(config, split, only):
    ids = [sid for sid, _ in declared(config, split)]
    if not only:
        return ids
    keep = [sid for sid in ids if any(sid == tok or sid.startswith(tok) or f'_{tok}_' in sid for tok in only)]
    return keep


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meta(args, config):
    return {'command': sys.argv, 'config': str(args.config), 'config_sha256': _sha(args.config),
            'envelope_sha256': _sha(args.envelope), 'module_sha256': _sha(W.__file__), 'runner_sha256': _sha(__file__),
            'benchmark_id': config['benchmark_id'],
            'benchmark_status': config['status'], 'envelope': str(args.envelope), 'started': datetime.now().astimezone().isoformat(),
            'tmpdir': os.environ.get('TMPDIR'), 'python': sys.executable, **git_state(),
            'label': 'DEV' if args.cmd in ('feasibility', 'kinematic') and getattr(args, 'split', 'dev') == 'dev' else None}


# ----------------------------------------------------------------------------- samples / feasibility

def cmd_samples(args):
    config = W.load_benchmark(args.config)
    rows = [W.sample_by_id(config, sid) for sid in selected_ids(config, args.split, args.only)]
    write_new(args.out, {'meta': meta(args, config), 'split': args.split, 'samples': rows})
    print(f'{len(rows)} samples -> {args.out}')
    return 0


def feasibility_row(config, envelope, sample_id, no_preflight=False):
    start = time.perf_counter()
    try:
        sample = W.sample_by_id(config, sample_id)
    except W.NotExpressible as error:
        return {'sample_id': sample_id, 'feasibility_class': None, 'classification_error': True,
                'reason': f'not expressible: {error}', 'wall_s': time.perf_counter()-start}
    row = W.classify_feasibility(
        sample, envelope, compile_fn=lambda s: W.compile_sample(config, s),
        preflight_fn=None if no_preflight else (lambda task: W.preflight(task, config['common']['preflight_geometry'])),
        precedence=tuple(config['feasibility']['precedence']),
        run_preflight_when_unsupported=bool(config['feasibility']['run_preflight_when_unsupported']), common=config['common'])
    row['sample'] = {k: sample[k] for k in ('scene', 'goal', 'draws', 'derived', 'envelope_parameters', 'index', 'rng')}
    row['wall_s'] = time.perf_counter()-start
    return row


def cmd_feasibility(args):
    config, envelope = W.load_benchmark(args.config), W.load_envelope(args.envelope)
    out = Path(args.out)
    if out.exists() and not args.resume:
        raise FileExistsError(f'{out} exists; pass --resume to continue it (existing sample files are kept)')
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    write_new(next_free(out/'feasibility_meta.json'), meta(args, config))
    rows = []
    for sid in selected_ids(config, args.split, args.only):
        path = out/f'{sid}.json'
        if path.exists():
            rows.append(json.loads(path.read_text()))
            print(f'[P5] {sid}: kept ({rows[-1].get("feasibility_class")})', flush=True)
            continue
        try:
            row = feasibility_row(config, envelope, sid, args.no_preflight)
        except Exception as error:   # never lost: a classification error row
            row = {'sample_id': sid, 'feasibility_class': None, 'classification_error': True,
                   'reason': f'feasibility error: {error!r}', 'traceback': traceback.format_exc()}
        write_new(path, row)
        rows.append(row)
        print(f"[P5] {sid}: {row.get('feasibility_class')} | {str(row.get('reason'))[:160]} | {row.get('wall_s', 0):.1f} s",
              flush=True)
    summary = {'meta': meta(args, config), 'split': args.split, 'finished': datetime.now().astimezone().isoformat(),
               'samples': [{k: r.get(k) for k in ('sample_id', 'family', 'range_set', 'stratum', 'seed', 'variant',
                                                  'ood_dimensions', 'feasibility_class', 'failing_classes', 'reason',
                                                  'envelope_pending', 'wall_s')} | {
                   'task_id': (r.get('compile') or {}).get('task_id'), 'digits': (r.get('compile') or {}).get('digits_used'),
                   'duration_s': (r.get('compile') or {}).get('duration_s'),
                   'preflight_binding': ((r.get('checks') or {}).get('preflight', {}).get('receipt') or {}).get('path', {}).get('binding')}
                   for r in rows]}
    write_new(next_free(out/'feasibility_summary.json'), summary)
    return 0


# ----------------------------------------------------------------------------- kinematic episode

def _gpu_memory():
    try:
        from robotic_cutting_sim.surface_runtime import gpu_process_memory_mib
        return gpu_process_memory_mib(os.getpid())
    except Exception as error:
        return {'error': repr(error)}


def cmd_kinematic(args):
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.recording import manifest, new_output
    from robotic_cutting_sim.scene_config import make_configured_kinematic_env
    config = W.load_benchmark(args.config)
    common = config['common']
    sample = W.sample_by_id(config, args.sample_id)
    task, info = W.compile_sample(config, sample)
    task_id = task.task_id
    if args.expect_task_id and task_id != args.expect_task_id:
        raise SystemExit(f'Compiled {task_id}, feasibility recorded {args.expect_task_id}')
    surface_profile = args.surface_profile or common['surface_profile']
    out = new_output(args.out)
    steps_total = int(round(task.duration_s/task.control_dt_s))
    stride = W.history_stride(config, steps_total)
    options = goal_planner.planner_options(task.goal, None)
    limits = goal_planner.action_limits(options, task.scene)
    run_config = {'benchmark_id': config['benchmark_id'], 'benchmark_status': config['status'], 'sample_id': sample['sample_id'],
                  'family': sample['family'], 'range_set': sample['range_set'], 'split': sample['split'], 'seed': sample['seed'],
                  'variant': sample['variant'], 'task_id': task_id, 'config_sha256': task.config_sha256, 'device': args.device,
                  'feasibility_class': args.feasibility_class, 'material_dt_s': common['material_dt_s'],
                  'material_seed': common['material_seed'], 'history_stride': stride, 'control_steps': steps_total,
                  'solver_module': info['solver_module'], 'max_cuts': info['max_cuts'], 'slots_per_node': info.get('slots_per_node'),
                  'surface_profile': surface_profile, 'cutting_law': common['solver']['cutting_law'],
                  'particle_spacing_m': task.particle_spacing_m, 'grid_dx_m': task.grid_dx_m,
                  'runtime_source': 'kinematic_ideal_tracking', 'scene': info['scene'], 'goal': info['goal'],
                  'planner_options': options, 'draws': sample['draws'], 'derived': sample['derived'],
                  'envelope_parameters': sample['envelope_parameters'], 'ood_dimensions': sample['ood_dimensions'],
                  'label': 'DEV' if sample['split'] == 'dev' or args.steps is not None else None}
    (out/'manifest.json').write_text(json.dumps(jsonable(manifest(run_config, 'kinematic_ideal_tracking')), indent=1))
    result = {'status': 'RUNNING', **{k: run_config[k] for k in ('benchmark_id', 'sample_id', 'family', 'range_set', 'split',
                                                                 'seed', 'variant', 'task_id', 'config_sha256', 'device',
                                                                 'feasibility_class', 'solver_module', 'max_cuts',
                                                                 'surface_profile', 'history_stride', 'control_steps',
                                                                 'runtime_source', 'label')}}
    start = time.monotonic()
    reset_ok, reset_error = True, None
    try:
        env = make_configured_kinematic_env(task, seed=common['material_seed'], device=args.device,
                                            material_dt=common['material_dt_s'], delta_frame='world', ik_failure='reject',
                                            history_stride=stride, action_limits=limits,
                                            surface_profile=None if surface_profile == 'particle_surface_v1' else surface_profile)
        observation, _ = env.reset(seed=common['material_seed'])
    except Exception as error:   # reset failure is a reliability result, preserved
        reset_ok, reset_error = False, f'{type(error).__name__}: {error}'
        result.update(status='ABORTED', reason=f'reset failed: {reset_error}', traceback=traceback.format_exc(),
                      reliability={'reset_success': False, 'reset_error': reset_error})
        (out/'result.json').write_text(json.dumps(jsonable(result), indent=1))
        print(f'[P5K] {sample["sample_id"]}: reset failed {reset_error}', flush=True)
        return 1
    setup_wall = time.monotonic()-start
    plan = goal_planner.plan(task.scene, task.goal, task.tool_profile, options)
    policy = WaypointPolicy(env.task, plan, limits, **goal_planner.policy_options(options))
    material = env.backend.material
    bed = material.bed
    dt = task.control_dt_s
    finite, min_det, max_speed = True, float('inf'), 0.
    limit = steps_total if args.steps is None else min(args.steps, steps_total)
    actions = []
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
            row = {'step': step+1, 'time_s': env.time_s, 'phase': policy.phase, 'cut_enabled': bool(action['cut_enabled']),
                   'cut_index': int(diag['cut_index']), 'edge_pose_world': env.backend.edge, 'blade_force_world_N': blade[:3],
                   'labelled_count': stroke.get('labelled_count'), 'engaged': stroke.get('engaged'),
                   'workspace_clamped': bool((info_step.get('action_report') or {}).get('workspace_clamped', False))}
            actions.append({'workspace_clamped': row['workspace_clamped']})
            stream.write(json.dumps(jsonable(row))+'\n')
            if terminated or truncated:
                break
            if step % 100 == 0:
                print(f'[P5K] {sample["sample_id"]} step {step+1}/{limit} t={env.time_s:.2f}s phase={policy.phase} '
                      f'wall={time.monotonic()-loop_start:.0f}s', flush=True)
    episode_wall = time.monotonic()-loop_start
    slots, solver_invalid = None, None if finite else 'non-finite particle state or deformation gradient'
    if hasattr(bed, 'check_slots'):
        try:
            slots = bed.check_slots()
        except Exception as error:   # SlotOverflowError: the affected substeps are invalid
            solver_invalid, slots = f'slot overflow: {error}', {'overflow': str(error)}
    eval_start = time.monotonic()
    try:
        evaluation = env.evaluate()
    except Exception as error:
        evaluation = {'verdict': None, 'status': 'NOT_RUN', 'error': repr(error), 'traceback': traceback.format_exc()}
    evaluation_wall = time.monotonic()-eval_start
    try:
        metrics = G.episode_metrics(task.goal, evaluation)
    except Exception as error:
        metrics = {'error': repr(error)}
    stability = {'finite': finite, 'min_det_F_sampled': min_det, 'max_speed_m_s_sampled': max_speed,
                 'sampling': 'every 10 control steps'}
    physical = W.physical_quality(task, material, evaluation, metrics, stability, actions)
    geometry = W.geometry_quality(env, task, args.geometry_level or common['geometry_quality_level']) \
        if not args.no_geometry else {'status': 'NOT_RUN', 'reason': '--no-geometry'}
    history = env._history
    np.savez_compressed(out/'physics_state.npz', time_s=np.asarray([s['time_s'] for s in history], float),
                        positions=np.asarray([s['positions'] for s in history], np.float32),
                        codes=np.asarray([s['codes'] for s in history]), mass_kg=material.masses,
                        reference_positions=material.world_points, reference_indices=material.lattice,
                        code_encoding=getattr(material, 'code_encoding', 'ternary'),
                        max_cuts=int(getattr(material, 'max_cuts', 1)))
    gpu = _gpu_memory()
    result.update(status='COMPLETE', policy_done=bool(policy.done), steps_run=len(actions), partial=limit < steps_total,
                  evaluation=evaluation, metrics=metrics, solver_invalid=solver_invalid, slots=slots, stability=stability,
                  physical_quality=physical, geometry_quality=geometry,
                  runtime={'reset_overhead_s': setup_wall, 'episode_wall_s': episode_wall, 'evaluation_wall_s': evaluation_wall,
                           'simulated_s': env.time_s, 'simulation_throughput_sim_s_per_wall_s': env.time_s/max(episode_wall, 1e-9),
                           'surface_reconstruction_s': geometry.get('extract_wall_s'), 'gpu_process_memory_mib': gpu},
                  particles=int(bed.n), substeps_per_control_step=int(material.substeps), history_frames=len(history),
                  physics_state_bytes=(out/'physics_state.npz').stat().st_size,
                  finished=datetime.now().astimezone().isoformat())
    cls, kind, reason = W.classify_execution(result)
    result.update({'class': cls, 'failure_kind': kind, 'class_reason': reason,
                   'reliability': {'reset_success': reset_ok, 'outputs': None},
                   'label_text': 'KINEMATIC ideal-tracking result; not a live result (no robot, grasp, slip, dynamics or cameras)'})
    env.close()
    (out/'result.json').write_text(json.dumps(jsonable(result), indent=1))
    valid = {}
    for name in ('manifest.json', 'result.json'):
        try:
            json.loads((out/name).read_text())
            valid[name] = True
        except Exception:
            valid[name] = False
    valid['actions.jsonl'] = all(json.loads(line) is not None for line in (out/'actions.jsonl').read_text().splitlines())
    write_new(out/'reliability.json', {'reset_success': reset_ok, 'recorded_output_validity': valid})
    print(f'[P5K] {sample["sample_id"]}: {cls} | {reason} | episode {episode_wall:.0f} s', flush=True)
    return 0


# ----------------------------------------------------------------------------- queue

def cmd_queue(args):
    config = W.load_benchmark(args.config)
    feas = Path(args.feasibility)
    rows = {p.stem: json.loads(p.read_text()) for p in feas.glob(f"{config['sample_prefix']}_*.json")}
    order = [sid for split in W.SPLITS for sid, _ in declared(config, split)]
    pending = [rows[sid] for sid in order if sid in rows and rows[sid].get('feasibility_class') == 'valid_feasible']
    if args.only:
        pending = [r for r in pending if r['sample_id'] in set(selected_ids(config, r['split'], args.only))]
    root, logs = Path(args.out_root), Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    running, log = {}, open(args.queue_log, 'a')

    def note(text):
        log.write(f'{datetime.now().astimezone().isoformat()} {text}\n')
        log.flush()
    note(f'QUEUE start {len(pending)} valid_feasible samples; devices {args.devices}; max {args.max_concurrent} per device; '
         f'launch when <= {args.max_other} other active jobs on that GPU; {git_state()}')
    fast_failures, last_wait = 0, None
    while pending or running:
        for pid, (proc, row, began, device) in list(running.items()):
            if proc.poll() is not None:
                wall = time.monotonic()-began
                note(f"END {row['sample_id']} device={device} rc={proc.returncode} wall={wall:.0f}s")
                del running[pid]
                fast = proc.returncode != 0 and wall < 60 and not (root/row['sample_id']/'result.json').exists()
                fast_failures = fast_failures+1 if fast else 0
                if fast_failures >= 2:
                    note(f'ABORT two consecutive fast launch failures; {len(pending)} pending left unlaunched')
                    pending = []
        launched, waits = False, {}
        for device in args.devices:
            if not pending:
                break
            if args.min_free_gb is not None and shutil.disk_usage(root).free/1e9 < args.min_free_gb:
                waits['disk'] = f'< {args.min_free_gb} GB free'
                break
            if len([1 for e in running.values() if e[3] == device]) >= args.max_concurrent:
                continue
            others = RG.gpu_other_jobs(int(device.split(':')[1]), set(running))
            if len(others) > args.max_other:
                waits[device] = others
                continue
            row = pending.pop(0)
            out = root/row['sample_id']
            if out.exists():
                note(f"SKIP {row['sample_id']} (output exists; preserved, not rerun)")
                continue
            log_path = next_free(logs/f"phase5_{row['sample_id']}_001.log")
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config), 'kinematic',
                       '--sample-id', row['sample_id'], '--device', device, '--out', str(out),
                       '--feasibility-class', 'valid_feasible']
            task_id = (row.get('compile') or {}).get('task_id')
            if task_id:
                command += ['--expect-task-id', task_id]
            if args.surface_profile:
                command += ['--surface-profile', args.surface_profile]
            proc = subprocess.Popen(command, cwd=str(ROOT), stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
                                    env={**os.environ, 'TMPDIR': os.environ.get('TMPDIR', '/mnt/data/ningan/tmp')})
            running[proc.pid] = (proc, row, time.monotonic(), device)
            note(f"START {row['sample_id']} device={device} pid={proc.pid} others={others} log={log_path}")
            launched = True
        if launched:
            time.sleep(20)
            continue
        if waits and (last_wait is None or time.monotonic()-last_wait > 600):
            note(f'WAIT {waits}; {len(running)} running; {len(pending)} pending')
            last_wait = time.monotonic()
        time.sleep(args.poll_s)
    note('QUEUE done')
    return 0


# ----------------------------------------------------------------------------- Native

def cmd_native_select(args):
    config = W.load_benchmark(args.config)
    feas = Path(args.feasibility)
    rows = [json.loads(p.read_text()) for p in feas.glob(f"{config['sample_prefix']}_*_{args.split}_*.json")]
    selection = W.native_selection(config, rows, split=args.split)
    write_new(args.out, {'meta': meta(args, config), 'rule': config['native']['selection'], 'strata': selection})
    for stratum, row in selection.items():
        print(stratum, len(row['selected']), row['shortfall_reason'] or '')
    return 0


def native_compile(config, sample):
    from robotic_cutting_sim.scene_config import compile_task
    native = config['native']
    options = native['planner_overrides']
    rule = native['duration_rule']
    _, info = W.compile_sample(config, sample)   # solver + digits as the kinematic stage
    scene, goal = info['scene'], deepcopy(info['goal'])
    goal['episode']['duration_s'] = float(rule['probe_duration_s'])
    probe = compile_task(deepcopy(scene), deepcopy(goal), planner_options=options, seed=native['seed'])
    done = next((float(t) for t, label in zip(probe._times(), probe.reference_phase_labels) if label == 'DONE'), None)
    if done is None:
        raise RuntimeError('The Native probe compile has no DONE knot')
    step = float(rule['round_up_to_s'])
    goal['episode']['duration_s'] = math.ceil((float(rule['plan_end_factor'])*done+float(rule['settle_after_plan_s']))/step-1e-9)*step
    task = compile_task(deepcopy(scene), deepcopy(goal), planner_options=options, seed=native['seed'])
    steps = int(round(task.duration_s/task.control_dt_s))
    return task, scene, goal, {'plan_end_s': done, 'duration_s': goal['episode']['duration_s'], 'steps': steps,
                               'history_stride': max(5, math.ceil(steps/40)), 'planner_overrides': options}


def native_prepare(config, sample_id, out):
    sample = W.sample_by_id(config, sample_id)
    task, scene, goal, info = native_compile(config, sample)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'scene.json', scene)
    write_new(out/'goal.json', goal)
    receipt = {'sample_id': sample_id, 'family': sample['family'], 'range_set': sample['range_set'], 'split': sample['split'],
               'native_task_id': task.task_id, 'config_sha256': task.config_sha256, 'solver_module': task.solver_module,
               'max_cuts': task.max_cuts, 'particles': int(task.sample_particles()[0].shape[0]), **info,
               'record_every': config['native']['record_every'], 'workcell_geometry': task.workcell_geometry(),
               'compiled_with': sys.executable, **git_state(), 'prepared': datetime.now().astimezone().isoformat()}
    write_new(out/'prepare.json', receipt)
    return receipt


def cmd_native_prepare(args):
    receipt = native_prepare(W.load_benchmark(args.config), args.sample_id, args.out)
    print(json.dumps({k: receipt[k] for k in ('sample_id', 'native_task_id', 'plan_end_s', 'duration_s', 'steps')}))
    return 0


def cmd_native_launch(args):
    import run_wp12_workcell_live as WL   # gpu_snapshot, du_bytes (WP-12 Phase 2b launcher)
    config = W.load_benchmark(args.config)
    native = config['native']
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT/out
    out.mkdir(parents=True, exist_ok=False)
    prep = native_prepare(config, args.sample_id, out/'inputs')
    free = shutil.disk_usage(out).free/1e9
    if free < args.disk_floor_gb:
        write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'disk {free:.1f} GB < floor {args.disk_floor_gb} GB'})
        return 3
    run_dir = out/'run'
    command = [native['python'], 'scripts/run_wp10_live.py', '--runtime', 'isaac_sim',
               '--scene', str((out/'inputs'/'scene.json').relative_to(ROOT)), '--goal', str((out/'inputs'/'goal.json').relative_to(ROOT)),
               '--planner-options', json.dumps(prep['planner_overrides']), *native['flags'],
               '--record-every', str(prep['record_every']), '--oracle-stride', str(prep['history_stride']),
               '--steps', str(prep['steps']), '--gpu', str(args.gpu if args.gpu is not None else native['gpu']),
               '--seed', str(native['seed']), '--out', str(run_dir.relative_to(ROOT))]
    if args.surface_profile:
        command += ['--surface-profile', args.surface_profile]
    env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONUNBUFFERED': '1', 'TMPDIR': '/mnt/data/ningan/tmp'}
    log = next_free(ROOT/'artifacts/wp12/logs'/f'{out.name}_{args.sample_id}.log')
    attempt = {'sample_id': args.sample_id, 'command': command, 'cwd': str(ROOT), 'log': str(log),
               'started': datetime.now().astimezone().isoformat(), 'gpu_before': WL.gpu_snapshot(), 'disk_free_gb_before': free,
               **git_state()}
    if args.dry_run:
        write_new(out/'attempt.json', {**attempt, 'status': 'NOT_RUN', 'reason': '--dry-run (command recorded only)'})
        print(json.dumps({'dry_run': True, 'command': command}))
        return 0
    start = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        attempt['pid'] = proc.pid
        returncode = proc.wait()
    result = json.loads((run_dir/'result.json').read_text()) if (run_dir/'result.json').exists() else None
    attempt.update(returncode=returncode, wall_s=time.monotonic()-start, ended=datetime.now().astimezone().isoformat(),
                   run_dir_bytes=WL.du_bytes(run_dir), result_status=(result or {}).get('status'),
                   note='A zero return code is not task success; the summary classifies the result.')
    write_new(out/'attempt.json', attempt)
    return 0


def native_execution(root, sample_id):
    """Execution result (classify_execution shape) of a Native attempt directory, or None if absent.

    PROVISIONAL mapping, to be confirmed against run_wp12_workcell_live_analysis before the freeze: run_wp10_live
    status FAIL_EXECUTION -> crash; refused_before_launch -> not_run (Native preflight refusal); a row's task_outcome is
    the goal evaluation; a task_end_reason containing GRASP -> grasp failure; policy_final_phase DONE = plan finished."""
    attempt = Path(root)/sample_id
    if not attempt.exists():
        return None
    path = attempt/'run'/'result.json'
    if not path.exists():
        return {'status': 'ABORTED', 'reason': 'Native run ended without result.json'}
    data = json.loads(path.read_text())
    if data.get('refused_before_launch'):
        return {'status': 'NOT_RUN', 'reason': f"Native plan preflight refused: {data.get('status')}"}
    if data.get('status') != 'PASS_EXECUTION' or not data.get('rows'):
        return {'status': 'ABORTED', 'reason': f"Native status {data.get('status')}"}
    row = data['rows'][0]
    if 'GRASP' in str(row.get('task_end_reason') or ''):
        return {'status': 'COMPLETE', 'grasp_failure': row.get('task_end_reason')}
    return {'status': 'COMPLETE', 'policy_done': row.get('policy_final_phase') in ('DONE', 'RETURN_HOME'),
            'evaluation': row.get('task_outcome') or {}, 'solver_invalid': None}


# ----------------------------------------------------------------------------- summary

def cmd_summarize(args):
    config = W.load_benchmark(args.config)
    feas = Path(args.feasibility)
    out = {'schema': 'rcs_wp12_benchmark_evidence_v1', 'meta': meta(args, config), 'split': args.split,
           'label': ('DEV' if args.split == 'dev' else None),
           'scope': 'Kinematic classes are KINEMATIC ideal-tracking results, never live results. Native classes come from '
                    'live Native Isaac Sim episodes of the declared stratified sub-sample.'}
    decl = declared(config, args.split)
    rows = {p.stem: json.loads(p.read_text()) for p in feas.glob(f"{config['sample_prefix']}_*_{args.split}_*.json")}
    for runtime, root in (('kinematic', args.kinematic_root), ('native', args.native_root)):
        if root is None and runtime == 'native':
            continue
        final = []
        for sid, metadata in decl:
            if sid not in rows:
                continue
            feas_row = rows[sid]
            execution = None
            if feas_row.get('feasibility_class') == 'valid_feasible':
                if runtime == 'kinematic' and root is not None:
                    path = Path(root)/sid/'result.json'
                    execution = json.loads(path.read_text()) if path.exists() else (
                        {'status': 'ABORTED', 'reason': 'output directory without result.json'} if (Path(root)/sid).exists() else None)
                elif runtime == 'native':
                    execution = native_execution(root, sid)
            row = W.final_row({**metadata, **feas_row}, execution)
            final.append(row)
        targets = {k: v for k, v in config['targets'].items() if isinstance(v, dict) and v.get('runtime') == runtime}
        out[runtime] = {'tables': W.summarize(final, decl, targets), 'samples': final}
    write_new(args.out, out)
    t = out['kinematic']['tables']['overall']
    print(json.dumps({k: t[k] for k in ('samples', 'counts', 'feasibility_rate', 'success_among_feasible', 'overall_success')}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=W.DRAFT)
    p.add_argument('--envelope', type=Path, default=W.ENVELOPE)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('samples')
    s.add_argument('--split', choices=W.SPLITS, default='dev')
    s.add_argument('--only', nargs='*')
    s.add_argument('--out', required=True)
    s = sub.add_parser('feasibility')
    s.add_argument('--split', choices=W.SPLITS, default='dev')
    s.add_argument('--only', nargs='*', help='sample ids or prefixes (e.g. P5_A_ID_dev)')
    s.add_argument('--out', required=True)
    s.add_argument('--resume', action='store_true')
    s.add_argument('--no-preflight', action='store_true', help='DEV only: skip the motion preflight (samples stay unclassified)')
    s = sub.add_parser('kinematic')
    s.add_argument('--sample-id', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--device', default='cuda:1')
    s.add_argument('--steps', type=int, help='Smoke only: stop after this many control steps (result partial, never a task result)')
    s.add_argument('--expect-task-id')
    s.add_argument('--feasibility-class')
    s.add_argument('--surface-profile', choices=('particle_surface_v1', 'cut_aware_surface_v1'))
    s.add_argument('--geometry-level', choices=('basic', 'full'))
    s.add_argument('--no-geometry', action='store_true')
    s = sub.add_parser('queue')
    s.add_argument('--feasibility', required=True)
    s.add_argument('--out-root', required=True)
    s.add_argument('--log-dir', default=str(ROOT/'artifacts/wp12/logs'))
    s.add_argument('--queue-log', required=True)
    s.add_argument('--devices', nargs='+', default=['cuda:1'])
    s.add_argument('--max-concurrent', type=int, default=2)
    s.add_argument('--max-other', type=int, default=2)
    s.add_argument('--min-free-gb', type=float)
    s.add_argument('--poll-s', type=float, default=60.)
    s.add_argument('--only', nargs='*')
    s.add_argument('--surface-profile', choices=('particle_surface_v1', 'cut_aware_surface_v1'))
    s = sub.add_parser('native-select')
    s.add_argument('--split', choices=W.SPLITS, default='benchmark', help='dev only to exercise the rule (DEV)')
    s.add_argument('--feasibility', required=True)
    s.add_argument('--out', required=True)
    s = sub.add_parser('native-prepare')
    s.add_argument('--sample-id', required=True)
    s.add_argument('--out', required=True)
    s = sub.add_parser('native-launch')
    s.add_argument('--sample-id', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--gpu', type=int)
    s.add_argument('--surface-profile', choices=('particle_surface_v1', 'cut_aware_surface_v1'))
    s.add_argument('--disk-floor-gb', type=float, default=15.)
    s.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('summarize')
    s.add_argument('--split', choices=W.SPLITS, default='benchmark')
    s.add_argument('--feasibility', required=True)
    s.add_argument('--kinematic-root')
    s.add_argument('--native-root')
    s.add_argument('--out', required=True)
    args = p.parse_args(argv)
    args.config, args.envelope = args.config.resolve(), args.envelope.resolve()
    return {'samples': cmd_samples, 'feasibility': cmd_feasibility, 'kinematic': cmd_kinematic, 'queue': cmd_queue,
            'native-select': cmd_native_select, 'native-prepare': cmd_native_prepare, 'native-launch': cmd_native_launch,
            'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
