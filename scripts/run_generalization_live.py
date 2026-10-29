#!/usr/bin/env python3
"""WP-11 Phase E round 1, live stage (declared in configs/wp11/generalization_round1_live.json).

Native (Isaac Sim) runs of the seeds the kinematic stage nominated. Every task is built
by the round-1 builders (``generalization.build_sample`` / ``compile_sample``, unchanged)
and recompiled with the declared live controller options and live duration rule; the
goal evaluator is the same ``goals.evaluate_goal`` the kinematic stage used. The live
episode runs through ``scripts/run_wp10_live.py --scene/--goal`` (physical grasp,
goal planner, measured edge feedback, hold to the time limit, physical return home).

Subcommands (every output is new; nothing is ever overwritten):

* ``prepare --sample ID --out DIR``: build + compile one sample, write ``DIR/scene.json``,
  ``DIR/goal.json`` and ``DIR/prepare.json`` (kinematic and live task ids, durations,
  steps, history stride, the exact runner command template). No physics.
* ``queue --root DIR``: in the declared priority order, prepare each sample into
  ``DIR/<sample>/inputs`` and launch its Native run into ``DIR/<sample>/run_001`` when the
  device policy allows (at most ``max_mine`` of these runs at once; a GPU only with
  <= ``max_other`` other active jobs and >= ``min_free_gib`` free; disk floor). An
  infrastructure crash before physics starts (no recorded control step) is relaunched once
  into ``run_002``; any other failure is kept and not retried.
* ``analyze --sample-dir DIR --run run_001 --out DIR/analysis_001``: read-only analysis and
  the declared live outcome class.
* ``summarize --root DIR --out FILE``: evidence over every declared sample (no exclusion),
  with the kinematic result of the same seed alongside, never merged.

Oracle material state is used for evaluation only, never as a controller input.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import generalization as G  # noqa: E402

LIVE_CONFIG = ROOT/'configs/wp11/generalization_round1_live.json'


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                else v.item() if isinstance(v, np.generic) else str(v)))


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.write_text(json.dumps(jsonable(payload), indent=1, allow_nan=True))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def git_state():
    run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'git_sha': run('rev-parse', 'HEAD'), 'git_dirty': bool(run('status', '--porcelain')), 'root': str(ROOT)}


def load_live(path=LIVE_CONFIG):
    live = json.loads(Path(path).read_text())
    if 'inherits' in live:
        # A supplementary stage file: the parent stage's rules, with only the keys it declares replaced.
        parent, _ = load_live(ROOT/live['inherits']['file'])
        digest = hashlib.sha256((ROOT/live['inherits']['file']).read_bytes()).hexdigest()
        if digest != live['inherits']['sha256']:
            raise ValueError(f'{live["inherits"]["file"]} sha256 {digest} != declared {live["inherits"]["sha256"]}')
        live = {**parent, **{k: v for k, v in live.items() if k != 'inherits'}, 'inherits': live['inherits']}
    round_path = ROOT/live['round_file']
    digest = hashlib.sha256(round_path.read_bytes()).hexdigest()
    if digest != live['round_file_sha256']:
        raise ValueError(f'{round_path} sha256 {digest} != declared {live["round_file_sha256"]}')
    return live, G.load_round(round_path)


def sample_spec(live, sample_id):
    for spec in live['samples']:
        if spec['sample_id'] == sample_id:
            return spec
    raise KeyError(f'{sample_id} is not a declared live sample')


def _flat(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flat(item, f'{prefix}.{key}' if prefix else key))
        return out
    return {prefix: value}


def compile_live(live, config, family, seed):
    """Round-1 sample + kinematic compile (unchanged) and the declared live recompile."""
    from robotic_cutting_sim.scene_config import compile_task
    sample = G.build_sample(config, family, seed)
    kinematic, info = G.compile_sample(config, sample)
    rule, options = live['task']['duration_rule'], live['controller']['planner_overrides']
    material_seed = config['common']['material_seed']
    scene, goal = deepcopy(info['scene']), deepcopy(info['goal'])
    goal['episode']['duration_s'] = float(rule['probe_duration_s'])
    probe = compile_task(scene, goal, planner_options=options, seed=material_seed)
    done = next((float(t) for t, label in zip(probe._times(), probe.reference_phase_labels) if label == 'DONE'), None)
    if done is None:
        raise RuntimeError('The live probe compile has no DONE knot')
    step = float(rule['round_up_to_s'])
    duration = math.ceil((float(rule['plan_end_factor'])*done+float(rule['settle_after_plan_s']))/step-1e-9)*step
    goal['episode']['duration_s'] = float(duration)
    task = compile_task(scene, goal, planner_options=options, seed=material_seed)
    steps = int(round(task.duration_s/task.control_dt_s))
    a, b = _flat(info['goal']), _flat(goal)
    goal_diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {'sample': sample, 'kinematic_task': kinematic, 'info': info, 'task': task, 'scene': scene, 'goal': goal,
            'plan_end_s': done, 'duration_s': float(duration), 'steps': steps,
            'history_stride': G.history_stride(config, steps), 'goal_diff_vs_kinematic': goal_diff}


def runner_command(live, spec, prep, inputs, run_dir, gpu):
    runtime = live['runtime']
    return [runtime['python'], 'scripts/run_wp10_live.py', '--runtime', 'isaac_sim',
            '--scene', str(Path(inputs)/'scene.json'), '--goal', str(Path(inputs)/'goal.json'),
            '--planner-options', json.dumps(live['controller']['planner_overrides']),
            '--controller', 'scripted', '--edge-feedback', live['controller']['edge_feedback'],
            '--hold-until-time-limit', '--return-home', '--record-cameras', '--overview-camera',
            '--record-every', str(spec['record_every']), '--oracle-stride', str(prep['history_stride']),
            '--steps', str(prep['steps']), '--gpu', str(gpu), '--seed', str(runtime['seed']), '--out', str(run_dir)]


def prepare(live, config, sample_id, out):
    spec = sample_spec(live, sample_id)
    built = compile_live(live, config, spec['family'], spec['seed'])
    stage1 = read_json(ROOT/live['kinematic']['stage1_dir']/f'{sample_id}.json') or {}
    recorded = (stage1.get('compile') or {}).get('task_id')
    if built['kinematic_task'].task_id != recorded:
        raise SystemExit(f'Kinematic recompile {built["kinematic_task"].task_id} != stage-1 record {recorded}')
    if built['goal_diff_vs_kinematic'] != ['episode.duration_s']:
        raise SystemExit(f'Live goal differs from the kinematic goal beyond the duration: {built["goal_diff_vs_kinematic"]}')
    expected = spec.get('declared_live_task') or {}
    for key in ('duration_s', 'steps', 'history_stride'):
        if key in expected and expected[key] != built[key]:
            raise SystemExit(f'{sample_id}: live {key} {built[key]} != declared {expected[key]}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'scene.json', built['scene'])
    write_new(out/'goal.json', built['goal'])
    task = built['task']
    receipt = {'sample_id': sample_id, 'family': spec['family'], 'seed': spec['seed'],
               'kinematic_task_id': recorded, 'live_task_id': task.task_id, 'live_config_sha256': task.config_sha256,
               'solver_module': task.solver_module, 'max_cuts': task.max_cuts, 'particles': int(task.sample_particles()[0].shape[0]),
               'kinematic_duration_s': built['info']['duration_s'], 'kinematic_plan_end_s': built['info']['plan_end_s'],
               'live_plan_end_s': built['plan_end_s'], 'duration_s': built['duration_s'], 'steps': built['steps'],
               'history_stride': built['history_stride'], 'record_every': spec['record_every'],
               'goal_diff_vs_kinematic': built['goal_diff_vs_kinematic'],
               'planner_overrides': live['controller']['planner_overrides'],
               'plan_report_operations': built['info']['plan_report_operations'], 'phases_kinematic': built['info']['phases'],
               'phases_live': list(dict.fromkeys(task.reference_phase_labels)),
               'compiled_with': sys.executable, **git_state(), 'prepared': datetime.now().astimezone().isoformat()}
    write_new(out/'prepare.json', receipt)
    return receipt


def cmd_prepare(args):
    live, config = load_live(args.live_config)
    receipt = prepare(live, config, args.sample, args.out)
    print(json.dumps({k: receipt[k] for k in ('sample_id', 'kinematic_task_id', 'live_task_id', 'live_plan_end_s',
                                               'duration_s', 'steps', 'history_stride')}))
    return 0


def cmd_plan(args):
    """Print the live task numbers for every declared sample (no files written)."""
    live, config = load_live(args.live_config)
    for spec in live['samples']:
        built = compile_live(live, config, spec['family'], spec['seed'])
        print(json.dumps({'sample_id': spec['sample_id'], 'kinematic_task_id': built['kinematic_task'].task_id,
                          'live_task_id': built['task'].task_id, 'kinematic_plan_end_s': built['info']['plan_end_s'],
                          'live_plan_end_s': round(built['plan_end_s'], 4), 'duration_s': built['duration_s'],
                          'steps': built['steps'], 'history_stride': built['history_stride'],
                          'solver': [built['task'].solver_module, built['task'].max_cuts],
                          'particles': int(built['task'].sample_particles()[0].shape[0]),
                          'goal_diff': built['goal_diff_vs_kinematic']}), flush=True)
    return 0


# ----------------------------------------------------------------------------- queue

def gpu_free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (line.split(',') for line in out.strip().splitlines())}


def disk_free_gb(path):
    return shutil.disk_usage(path).free/1e9


def du_bytes(path):
    out = subprocess.run(['du', '-sb', str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def staged_gate(live, inputs, attempt):
    """Round-1 stage-1 geometry (generalization.staged_preflight) over the full live plan; receipt in inputs/."""
    from robotic_cutting_sim import motion_preflight as mp
    from robotic_cutting_sim.scene_config import compile_task
    path = Path(inputs)/'staged_preflight.json'
    if attempt > 1 and path.exists():
        return read_json(path)
    start = time.perf_counter()
    task = compile_task(str(Path(inputs)/'scene.json'), str(Path(inputs)/'goal.json'),
                        planner_options=live['controller']['planner_overrides'], seed=live['runtime']['seed'])
    receipt, _ = G.staged_preflight(task, mp.MotionChecker(task))
    cls, reason = G.classify_preflight(receipt)
    gate = {'status': 'PASS' if cls is None else 'FAIL', 'class': cls, 'reason': reason or 'staged preflight PASS',
            'task_id': task.task_id, 'rule': live['preflight']['text'], 'receipt': receipt,
            'wall_s': time.perf_counter()-start, **git_state()}
    write_new(path, gate)
    return gate


def recorded_steps(run_dir):
    path = Path(run_dir)/'episode_000'/'actions.jsonl'
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(1 for line in stream if line.strip())


def cmd_queue(args):
    live, config = load_live(args.live_config)
    policy = live['runtime']['device_policy']
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    gen = _load_script('run_generalization')
    order = [s['sample_id'] for s in sorted(live['samples'], key=lambda s: s['priority'])]
    if args.only:
        order = [s for s in order if s in set(args.only)]
    pending = [(sample_id, 1) for sample_id in order]
    running, last_launch = {}, {}
    queue_log = open(args.queue_log, 'a')

    def note(text):
        line = f'{datetime.now().astimezone().isoformat()} {text}'
        queue_log.write(line+'\n')
        queue_log.flush()
        print(line, flush=True)

    note(f'QUEUE start {order}; devices {args.devices}; max_mine {policy["max_mine"]}; max_other {policy["max_other_active"]}; '
         f'min_free_gib {policy["min_free_gib"]}; disk floor {policy["disk_floor_gb"]} GB; {git_state()}')
    last_wait = None
    while pending or running:
        for pid, entry in list(running.items()):
            proc = entry['proc']
            if proc.poll() is None:
                continue
            del running[pid]
            run_dir = entry['run_dir']
            wall = time.monotonic()-entry['began']
            steps = recorded_steps(run_dir)
            receipt = {**entry['receipt'], 'returncode': proc.returncode, 'wall_s': wall,
                       'ended': datetime.now().astimezone().isoformat(), 'recorded_control_steps': steps,
                       'run_dir_bytes': du_bytes(run_dir), 'disk_free_gb_after': disk_free_gb(root),
                       'result_status': (read_json(Path(run_dir)/'result.json') or {}).get('status')}
            write_new(Path(run_dir).parent/f'attempt_{entry["attempt"]:03d}.json', receipt)
            note(f'END {entry["sample_id"]} attempt={entry["attempt"]} gpu={entry["gpu"]} rc={proc.returncode} '
                 f'wall={wall:.0f}s steps={steps} status={receipt["result_status"]} bytes={receipt["run_dir_bytes"]}')
            top = read_json(Path(run_dir)/'result.json') or {}
            infrastructure = steps == 0 and not top.get('refused_before_launch') and top.get('status') != 'PASS_EXECUTION'
            if infrastructure and entry['attempt'] == 1:
                note(f'RELAUNCH {entry["sample_id"]}: crashed before any control step (infrastructure); one relaunch')
                pending.insert(0, (entry['sample_id'], 2))
        free_disk = disk_free_gb(root)
        if free_disk < policy['disk_floor_gb']:
            note(f'DISK {free_disk:.1f} GB free < floor {policy["disk_floor_gb"]} GB: no further launches; '
                 f'{len(running)} running left to finish; pending {pending}')
            pending = []
            if not running:
                break
        if pending and len(running) < policy['max_mine']:
            sample_id, attempt = pending[0]
            spec = sample_spec(live, sample_id)
            in_flight = sum(max(0., sample_spec(live, e['sample_id'])['estimated_disk_gb']-(du_bytes(e['run_dir']) or 0)/1e9)
                            for e in running.values())
            if free_disk-in_flight-spec['estimated_disk_gb'] < policy['disk_floor_gb']:
                note(f'DISK {free_disk:.1f} GB free - {in_flight:.1f} GB still to be written by running jobs - estimate '
                     f'{spec["estimated_disk_gb"]} GB for {sample_id} would cross the {policy["disk_floor_gb"]} GB floor: '
                     f'no further launches; pending {pending}')
                pending = []
                continue
            free = gpu_free_mib()
            mine = {pid for pid in running}
            candidates, waits = [], {}
            for gpu in args.devices:
                others = gen.gpu_other_jobs(gpu, mine)
                settled = time.monotonic()-last_launch.get(gpu, -1e9) >= policy['settle_after_launch_s']
                ok = (len(others) <= policy['max_other_active'] and free.get(gpu, 0) >= policy['min_free_gib']*1024
                      and settled)
                if ok:
                    candidates.append((gpu, free.get(gpu), others, settled))
                else:
                    waits[gpu] = {'free_mib': free.get(gpu), 'other_active': others, 'settled': settled}
            if candidates:
                gpu, gpu_free, others, _ = max(candidates, key=lambda c: c[1])
                pending.pop(0)
                sample_dir = root/sample_id
                inputs = sample_dir/'inputs'
                if attempt == 1:
                    if sample_dir.exists():
                        note(f'SKIP {sample_id}: {sample_dir} exists (preserved, not rerun)')
                        continue
                    prep = prepare(live, config, sample_id, inputs)
                else:
                    prep = read_json(inputs/'prepare.json')
                run_dir = sample_dir/f'run_{attempt:03d}'
                command = runner_command(live, spec, prep, inputs, run_dir, gpu)
                if live.get('preflight', {}).get('mode') == 'round1_staged':
                    gate = staged_gate(live, inputs, attempt)
                    if gate['status'] != 'PASS':
                        run_dir.mkdir(parents=True, exist_ok=False)
                        write_new(run_dir/'result.json', {'status': gate['class'] or 'SCENE_INFEASIBLE',
                                                          'refused_before_launch': True, 'preflight': 'round1_staged',
                                                          'reason': gate['reason'], 'task_id': prep['live_task_id']})
                        note(f'REFUSED {sample_id}: staged preflight {gate["status"]} ({gate["reason"]})')
                        continue
                    command.append('--skip-plan-preflight')
                log_path = logs/f'{live.get("log_prefix", "phaseE_live")}_{sample_id}_{attempt:03d}.log'
                if log_path.exists():
                    log_path = logs/f'{live.get("log_prefix", "phaseE_live")}_{sample_id}_{attempt:03d}_{time.strftime("%Y%m%d_%H%M%S")}.log'
                env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONUNBUFFERED': '1',
                       'TMPDIR': live['runtime']['tmpdir']}
                proc = subprocess.Popen(command, cwd=str(ROOT), stdout=open(log_path, 'w'), stderr=subprocess.STDOUT, env=env)
                receipt = {'sample_id': sample_id, 'attempt': attempt, 'gpu': gpu, 'device': f'cuda:{gpu}', 'pid': proc.pid,
                           'command': command, 'cwd': str(ROOT), 'env': {k: env[k] for k in ('OMNI_KIT_ACCEPT_EULA',
                                                                                              'PYTHONUNBUFFERED', 'TMPDIR')},
                           'log': str(log_path), 'started': datetime.now().astimezone().isoformat(),
                           'gpu_free_mib_at_launch': gpu_free, 'other_active_jobs_at_launch': others,
                           'disk_free_gb_before': free_disk, **git_state()}
                running[proc.pid] = {'proc': proc, 'sample_id': sample_id, 'attempt': attempt, 'gpu': gpu,
                                     'run_dir': run_dir, 'began': time.monotonic(), 'receipt': receipt}
                last_launch[gpu] = time.monotonic()
                note(f'START {sample_id} attempt={attempt} gpu={gpu} pid={proc.pid} free_mib={gpu_free} others={others} '
                     f'disk={free_disk:.1f}GB log={log_path}')
                continue
            if last_wait is None or time.monotonic()-last_wait > 600:
                note(f'WAIT {sample_id}: no GPU meets the policy {waits}; {len(running)} mine running')
                last_wait = time.monotonic()
        time.sleep(args.poll_s)
    note('QUEUE done')
    return 0


# ----------------------------------------------------------------------------- analysis

def _stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return None
    return {'min': float(values.min()), 'max': float(values.max()), 'mean': float(values.mean()),
            'p95': float(np.percentile(values, 95)), 'max_abs': float(np.abs(values).max()), 'samples': int(len(values))}


def is_lifecycle_home(row):
    """Rows of the Phase D physical return-to-home (labelled RETURN_HOME:<stage> by the runner).

    The planner's own RETURN_HOME waypoint phase (label exactly 'RETURN_HOME') and the hold that
    follows it belong to the task phase (the episode before TASK_COMPLETE)."""
    return str(row.get('policy_phase', '')).startswith('RETURN_HOME:')


def measured_path_clearance(task, task_rows, out):
    """C2 geometric check of the measured task-phase joint path (report-only; as analyze_phase_c5_live,
    over every task-phase row including the planner RETURN_HOME and the hold)."""
    from robotic_cutting_sim.motion_preflight import phase_mode, run_cli
    from robotic_cutting_sim.scene_config import configured_task_payload
    qs = [np.asarray(r['measured']['q'], float)[0] for r in task_rows]
    modes = [phase_mode(str(r.get('policy_phase')), bool((r.get('submitted') or {}).get('cut_enabled'))) for r in task_rows]
    receipt = run_cli('check-path', configured_task_payload(task), q=np.asarray(qs).tolist(), modes=modes,
                      log_dir=Path(out)/'measured_path')
    return {'status': receipt.get('status'), 'failing': receipt.get('failing'), 'binding': receipt.get('binding'),
            'worst': receipt.get('worst'), 'samples': receipt.get('samples'),
            'blade_workpiece_modes': receipt.get('blade_workpiece_modes'), 'scope': receipt.get('scope'),
            'note': 'Task phase only (episode before TASK_COMPLETE); the physical return-home path has its own check in '
                    'home_outcome. Declared uncut workpiece, nominal grasp: material removed earlier in the episode is '
                    'still treated as present.'}


def _op_of(phase):
    phase = str(phase)
    return phase.split(':', 1)[1] if ':' in phase and not phase.startswith('RETURN_HOME') else None


def operation_reports(task, prep, rows):
    from robotic_cutting_sim.skim_orientation import skim_orientation_report
    c5 = _load_script('analyze_phase_c5_live')
    R, t = task.T_world_workpiece[:3, :3], task.T_world_workpiece[:3, 3]
    bar_top = float(t[2]+task.geometry.base_height_m)
    dt = task.control_dt_s
    out = []
    for op in prep['plan_report_operations']:
        sub = [r for r in rows if _op_of(r.get('policy_phase')) == op['op']]
        armed = [r for r in sub if (r.get('submitted') or {}).get('cut_enabled')]
        report = {'op': op['op'], 'kind': op['kind'], 'digit': op.get('digit'), 'steps': len(sub), 'armed_steps': len(armed),
                  'time_s': [sub[0]['time_s'], sub[-1]['time_s']] if sub else None,
                  'armed_time_s': [armed[0]['time_s'], armed[-1]['time_s']] if armed else None,
                  'tracking_error_m': _stats([r.get('tracking_error_m') for r in sub]),
                  'material_reaction_armed': c5.reaction(armed, dt) if armed else None}
        poses = np.asarray([r['measured']['tool_pose'] for r in armed], float) if armed else np.zeros((0, 7))
        if op['kind'] == 'surface_trim' and len(armed) >= 2:
            normal = R@np.array([0., 0., 1.])
            direction = R@np.asarray(op['direction_local'], float)
            target_z = float(t[2]+op['plane_local_z_m']+op['edge_clearance_m'])
            report.update(
                declared={k: op[k] for k in ('declared_rake_deg', 'declared_skew_deg', 'direction_local', 'edge_line_local')},
                measured_orientation=skim_orientation_report(poses, normal, direction, source='measured tool edge (physically held knife)'),
                commanded_orientation=skim_orientation_report([r['accepted_target_world_xyz_xyzw'] for r in armed], normal,
                                                              direction, source='accepted commanded edge target'),
                mu_chord_deg=float(np.degrees(np.arcsin(np.clip((poses[-1, :3]-poses[0, :3])@normal
                                                                 / max(np.linalg.norm(poses[-1, :3]-poses[0, :3]), 1e-12), -1, 1)))),
                measured_edge_z_minus_target_plus_clearance_m=_stats(poses[:, 2]-target_z),
                lowest_blade_point_above_bar_top_m=_stats([c5.lowest_blade_z(task, p)-bar_top for p in poses]))
        elif op['kind'] == 'through_cut' and len(armed):
            origin = R@np.asarray(op['plane_origin_local_m'], float)+t
            normal = R@np.asarray(op['plane_normal_local'], float)
            report.update(measured_edge_signed_distance_from_plane_m=_stats((poses[:, :3]-origin)@normal),
                          measured_edge_start_local_m=task.points_to_local(poses[:1, :3])[0].tolist(),
                          measured_edge_end_local_m=task.points_to_local(poses[-1:, :3])[0].tolist(),
                          planned_start_local_m=op['start_local_m'], planned_end_local_m=op['end_local_m'])
        elif op['kind'] == 'blade_face_push' and sub:
            push = [r for r in sub if str(r.get('policy_phase', '')).startswith('PUSH')]
            report.update(push_time_s=[push[0]['time_s'], push[-1]['time_s']] if push else None,
                          cut_enabled_steps=int(sum(bool((r.get('submitted') or {}).get('cut_enabled')) for r in sub)))
        out.append(report)
    return out


def phase_summary(rows):
    out = {}
    for phase in dict.fromkeys(str(r.get('policy_phase')) for r in rows):
        sub = [r for r in rows if str(r.get('policy_phase')) == phase]
        out[phase] = {'steps': len(sub), 't_s': [sub[0]['time_s'], sub[-1]['time_s']],
                      'tracking_error_max_m': (_stats([r.get('tracking_error_m') for r in sub]) or {}).get('max'),
                      'rotation_error_max_rad': (_stats([r.get('rotation_error_rad') for r in sub]) or {}).get('max')}
    return out


def classify_live(live, top, episode, rows, failure, finite, grasp, ik):
    """(class, reason) under the declared precedence."""
    if top is None:
        return 'NOT_RUN', 'no result.json'
    if top.get('refused_before_launch'):
        status = str(top.get('status'))
        return ('IK_FAILURE' if 'IK' in status.upper() else 'SCENE_INFEASIBLE'), f'live plan preflight refused: {status}'
    if episode is None:
        error = (failure or {}).get('error', '')
        if 'SlotOverflowError' in error:
            return 'CUTTING_FAILURE', f'solver state invalid: slot overflow ({error[:200]})'
        return 'NOT_RUN', f'execution ended without an episode result: {error[:300] or top.get("status")}'
    if ik['task_steps_ik_not_ok'] or ik['task_steps_workspace_clamped']:
        return 'IK_FAILURE', (f'live IK not ok on {ik["task_steps_ik_not_ok"]} task steps, workspace clamp on '
                              f'{ik["task_steps_workspace_clamped"]} task steps')
    gates = live['outcome_classes']['GRASP_FAILURE']['gates']
    if grasp['grasp_lost'] or (grasp['max_slip_m'] or 0) > gates['max_slip_m'] or \
            (grasp['max_rotation_rad'] or 0) > gates['max_rotation_rad']:
        return 'GRASP_FAILURE', (f'grasp lost={grasp["grasp_lost"]}, max slip {grasp["max_slip_m"]} m, max rotation '
                                 f'{grasp["max_rotation_rad"]} rad (gates {gates["max_slip_m"]} m / {gates["max_rotation_rad"]} rad)')
    if not finite:
        return 'CUTTING_FAILURE', 'solver state invalid: non-finite recorded particle state'
    if episode.get('task_end_reason') != 'CONTROLLER_DONE':
        return 'CUTTING_FAILURE', f'the scripted plan did not finish within the episode (task end {episode.get("task_end_reason")}, ' \
                                  f'final phase {episode.get("policy_final_phase")})'
    evaluation = episode.get('task_outcome') or {}
    verdict = evaluation.get('verdict')
    if verdict == 'PASS':
        return 'TASK_SUCCESS', 'goal evaluator PASS (live Native), plan finished, grasp within gates, no IK failure'
    if verdict == 'FAIL_INTEGRITY':
        failed = [k for k, ok in evaluation['integrity']['checks'].items() if not ok]
        return 'CUTTING_FAILURE', 'integrity: '+', '.join(failed)
    if verdict == 'FAIL':
        return 'CUTTING_FAILURE', 'outcome: '+', '.join(G.failed_checks(evaluation))
    return 'NOT_RUN', f'goal evaluator verdict {verdict!r}'


def cmd_analyze(args):
    from robotic_cutting_sim.recording import new_output
    from robotic_cutting_sim.scene_config import compile_task
    live, config = load_live(args.live_config)
    c5 = _load_script('analyze_phase_c5_live')
    sample_dir = Path(args.sample_dir)
    run = sample_dir/args.run
    prep = read_json(sample_dir/'inputs'/'prepare.json')
    attempt = read_json(sample_dir/f'attempt_{int(args.run.split("_")[-1]):03d}.json')
    out = new_output(args.out)
    top = read_json(run/'result.json')
    failure = read_json(run/'failure.json')
    episode = read_json(run/'episode_000'/'result.json')
    rows = []
    if (run/'episode_000'/'actions.jsonl').exists():
        rows = [json.loads(line) for line in (run/'episode_000'/'actions.jsonl').open() if line.strip()]
    task = compile_task(str(sample_dir/'inputs'/'scene.json'), str(sample_dir/'inputs'/'goal.json'),
                        planner_options=live['controller']['planner_overrides'], seed=live['runtime']['seed'])
    task_rows = [r for r in rows if not is_lifecycle_home(r)]
    ik = {'task_steps_ik_not_ok': int(sum((r.get('info') or {}).get('ik_status') not in (None, 'ok') for r in task_rows)),
          'task_steps_workspace_clamped': int(sum(bool(((r.get('info') or {}).get('action_report') or {}).get('workspace_clamped'))
                                                  for r in task_rows)),
          'task_steps_translation_clamped': int(sum(bool(((r.get('info') or {}).get('action_report') or {}).get('translation_clamped'))
                                                    for r in task_rows)),
          'task_steps_rotation_clamped': int(sum(bool(((r.get('info') or {}).get('action_report') or {}).get('rotation_clamped'))
                                                 for r in task_rows)),
          'ik_status_counts': {}}
    for r in task_rows:
        key = str((r.get('info') or {}).get('ik_status'))
        ik['ik_status_counts'][key] = ik['ik_status_counts'].get(key, 0)+1
    grasp_task, grasp_all = c5.grasp_audit(task_rows), c5.grasp_audit(rows)
    finite = True
    physics = run/'episode_000'/'physics_state.npz'
    if physics.exists():
        with np.load(physics) as data:
            finite = bool(np.isfinite(data['positions']).all() and np.isfinite(data['velocities']).all())
    cls, reason = classify_live(live, top, episode, rows, failure, finite, grasp_task, ik)
    evaluation = (episode or {}).get('task_outcome') or {}
    goal = json.loads((sample_dir/'inputs'/'goal.json').read_text())
    try:
        metrics = G.episode_metrics(goal, evaluation) if evaluation else None
    except Exception as error:  # report-only
        metrics = {'error': repr(error)}
    controller_rows = [r for r in task_rows if r.get('policy_phase') != 'HOLD_AFTER_STOP']
    analysis_note = ('analysis_001 of the first sample (E1_TRIM_s1203, run_generalization_live.py at 1ecfe5e) treated '
                     'the planner RETURN_HOME phase as return-home rows; fixed: only the physical RETURN_HOME:<stage> '
                     'rows are excluded from the task phase')
    done_time = (float(controller_rows[-1]['time_s']) if episode and controller_rows
                 and episode.get('task_end_reason') == 'CONTROLLER_DONE' else None)
    analysis = {
        'sample_id': prep['sample_id'], 'family': prep['family'], 'seed': prep['seed'], 'run': str(run),
        'class': cls, 'class_reason': reason, 'runtime_source': (episode or {}).get('runtime_source'),
        'device': (attempt or {}).get('device'), 'attempt': attempt,
        'task_id_recorded': (episode or {}).get('task_id'), 'task_id_recompiled_cpu': task.task_id,
        'task_id_match': (episode or {}).get('task_id') == task.task_id, 'kinematic_task_id': prep['kinematic_task_id'],
        'task_outcome': {'verdict': evaluation.get('verdict'),
                         'integrity_checks': (evaluation.get('integrity') or {}).get('checks'),
                         'outcome_checks': (evaluation.get('outcome') or {}).get('checks'),
                         'failed_checks': G.failed_checks(evaluation) if evaluation else None,
                         'outcome_metrics': (evaluation.get('outcome') or {}).get('metrics'),
                         'quality': evaluation.get('quality')},
        'metrics': metrics,
        'home_outcome': {'status': ((episode or {}).get('home_outcome') or {}).get('status'),
                         'failed_checks': ((episode or {}).get('home_outcome') or {}).get('failed_checks'),
                         'reason': ((episode or {}).get('home_outcome') or {}).get('reason'),
                         'measurement': (episode or {}).get('home_measurement'),
                         'variant': ((episode or {}).get('home_plan') or {}).get('variant')},
        'episode_state_final': (episode or {}).get('episode_state_final'),
        'state_path': [(t['from'], t['to'], t['time_s'], t['reason']) for t in (episode or {}).get('episode_transitions', [])],
        'task_end_reason': (episode or {}).get('task_end_reason'), 'policy_final_phase': (episode or {}).get('policy_final_phase'),
        'plan_timing': {'live_reference_plan_end_s': prep['live_plan_end_s'], 'live_controller_done_s': done_time,
                        'lag_s': None if done_time is None else done_time-prep['live_plan_end_s'],
                        'duration_s': prep['duration_s']},
        'ik': ik, 'grasp_task_phase': grasp_task, 'grasp_all': grasp_all,
        'contacts_task_phase': c5.contact_audit(task_rows), 'contacts_all': c5.contact_audit(rows),
        'operations': operation_reports(task, prep, rows), 'phases': phase_summary(rows),
        'finite_recorded_state': finite, 'recorded_control_steps': len(rows),
        'wall_s': {'run_total': (top or {}).get('wall_s'), 'episode': (episode or {}).get('wall_s'),
                   'process': (attempt or {}).get('wall_s')},
        'run_dir_bytes': (attempt or {}).get('run_dir_bytes'),
        'failure': None if failure is None else {'error': failure.get('error')},
        'scope': 'Read-only analysis of a live Native episode; oracle material state is evaluation-only',
        'analysis_tool': {**git_state(), 'note': analysis_note}}
    if rows and not args.skip_path_check:
        try:
            analysis['measured_task_path_clearance'] = measured_path_clearance(task, task_rows, out)
        except Exception as error:  # report-only
            analysis['measured_task_path_clearance'] = {'status': 'NOT_RUN', 'reason': repr(error)}
    write_new(out/'analysis.json', analysis)
    print(json.dumps(jsonable({k: analysis[k] for k in ('sample_id', 'class', 'class_reason')}
                              | {'verdict': analysis['task_outcome']['verdict'], 'home': analysis['home_outcome']['status']})))
    return 0


# ----------------------------------------------------------------------------- summary

def stage_rows(live, root, kinematic):
    rows = []
    for spec in sorted(live['samples'], key=lambda s: s['priority']):
        sample_id = spec['sample_id']
        sample_dir = root/sample_id
        analyses = sorted(sample_dir.glob('analysis_*/analysis.json')) if sample_dir.exists() else []
        attempts = [read_json(p) for p in sorted(sample_dir.glob('attempt_*.json'))] if sample_dir.exists() else []
        k = kinematic.get(sample_id, {})
        row = {'sample_id': sample_id, 'family': spec['family'], 'seed': spec['seed'], 'priority': spec['priority'],
               'purpose': spec['purpose'],
               'kinematic': {'class': k.get('class'), 'reason': k.get('reason'), 'task_id': k.get('task_id'),
                             'metrics': (k.get('stage2') or {}).get('metrics'), 'device': (k.get('stage2') or {}).get('device'),
                             'scope': 'KINEMATIC ideal tracking (round 1 stage 2), default planner limits'},
               'attempts': attempts}
        if analyses:
            row['live'] = read_json(analyses[-1])
            row['class'], row['reason'] = row['live']['class'], row['live']['class_reason']
        else:
            row['class'], row['reason'] = 'NOT_RUN', 'no live analysis (not launched or not finished)'
        row['stage_id'] = live['stage_id']
        rows.append(row)
    return rows


def cmd_summarize(args):
    live, config = load_live(args.live_config)
    root = Path(args.root)
    kinematic = {s['sample_id']: s for s in read_json(ROOT/live['kinematic']['evidence'])['samples']}
    rows = stage_rows(live, root, kinematic)
    classes = list(live['outcome_classes'])
    table = {name: {c: sum(r['class'] == c for r in rows if r['family'] == name) for c in classes}
             for name in dict.fromkeys(r['family'] for r in rows)}
    supplementary = []
    for token in args.supplementary or []:
        config_path, _, sub_root = token.partition(':')
        sub_live, _ = load_live(ROOT/config_path)
        supplementary.append({'stage_id': sub_live['stage_id'], 'live_config': config_path, 'root': sub_root,
                              'declared': sub_live['declared'], 'reason': sub_live.get('reason'),
                              'preflight': sub_live.get('preflight'),
                              'samples': stage_rows(sub_live, Path(sub_root), kinematic)})
    out = {'schema': 'rcs_generalization_live_evidence_v1', 'round_id': live['round_id'], 'stage_id': live['stage_id'],
           'live_config': str(args.live_config), 'generated': datetime.now().astimezone().isoformat(), 'command': sys.argv,
           'root': str(root), 'label': live['scope'], 'counts': table,
           'totals': {c: sum(r['class'] == c for r in rows) for c in classes}, 'held': live['held'], 'samples': rows,
           'supplementary_stages': supplementary,
           'supplementary_rule': 'Supplementary stages are separately declared, labelled results; they never change the '
                                 'class of a sample in this stage.', **git_state()}
    write_new(args.out, out)
    for r in rows+[x for stage in supplementary for x in stage['samples']]:
        print(r['stage_id'], r['sample_id'], r['kinematic']['class'], '->', r['class'], '|', r['reason'][:140])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--live-config', type=Path, default=LIVE_CONFIG)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--sample', required=True)
    a.add_argument('--out', required=True)
    sub.add_parser('plan')
    b = sub.add_parser('queue')
    b.add_argument('--root', required=True)
    b.add_argument('--log-dir', default=str(ROOT/'artifacts/wp11/logs'))
    b.add_argument('--queue-log', required=True)
    b.add_argument('--devices', nargs='+', type=int, default=[0, 1])
    b.add_argument('--only', nargs='*')
    b.add_argument('--poll-s', type=float, default=30.)
    c = sub.add_parser('analyze')
    c.add_argument('--sample-dir', required=True)
    c.add_argument('--run', default='run_001')
    c.add_argument('--out', required=True)
    c.add_argument('--skip-path-check', action='store_true')
    d = sub.add_parser('summarize')
    d.add_argument('--root', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--supplementary', nargs='*', help='STAGE_CONFIG:ROOT of separately declared supplementary stages')
    args = p.parse_args(argv)
    args.live_config = Path(args.live_config).resolve()
    return {'prepare': cmd_prepare, 'plan': cmd_plan, 'queue': cmd_queue, 'analyze': cmd_analyze,
            'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
