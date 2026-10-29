#!/usr/bin/env python3
"""Policy dataset collector (WP-11 Phase G, ``rcs_dataset_v1``) over the ``cutting_research_v1`` env.

One process, one runtime, N consecutive episodes of one compiled scene + goal:
Reset -> episode (scripted goal planner | open-loop action program | external
policy callable) -> end task -> Evaluate -> optional physical Return Home ->
store -> Reset. Every episode is stored, successful or failed, with its
actions (and action timestamps), robot state, camera RGB-D + calibration,
privileged physics history (separate file), task result, termination reason,
physical diagnostics, frequencies and the source commit / physics version; an
``index.json`` lists them and ``reliability.json`` holds the multi-episode
checks (``configs/wp11/phaseG_reliability_v1.json``) and the POLICY_READY flag.

  # kinematic (CPU/GPU, no robot):
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/collect_dataset.py --runtime kinematic \
      --scene trim01_wp11_workcell --goal trim_bump_rho10 --episode-plan PLAN.json --spacing 0.004 \
      --material-dt 0.0005 --device cpu --out artifacts/wp11/phaseG/<name>
  # Native Isaac Sim:
  OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/collect_dataset.py \
      --runtime isaac_sim --gpu 1 ... --out artifacts/wp11/phaseG/<name>
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import dataset as ds  # noqa: E402
from robotic_cutting_sim.policy_env import (PRESET_ID, PRIVILEGED_CONTROLLER_INPUTS, find_privileged_keys,  # noqa: E402
                                            jsonable, process_memory)
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402

CONTROLLERS = ('scripted', 'action_file', 'policy')
PLAN_SCHEMA = 'rcs_episode_plan_v1'
RELIABILITY_CONFIG = ROOT/'configs/wp11/phaseG_reliability_v1.json'
HOLD = {'type': 'edge_delta', 'target': [0.]*6, 'frame': 'world', 'cut_enabled': False}


def episode_entries(a):
    """Episode plan rows (controller, seed, options); a --episode-plan file or N copies of the CLI options."""
    if a.episode_plan:
        plan = json.loads(Path(a.episode_plan).read_text())
        if plan.get('schema') != PLAN_SCHEMA:
            raise SystemExit(f'--episode-plan must be an {PLAN_SCHEMA} document')
        entries = [dict(e) for e in plan['episodes']]
    else:
        entries = [dict(controller=a.controller, action_file=a.action_file, policy=a.policy, return_home=a.return_home,
                        stop_after_phase=a.stop_after_phase, hold_until_time_limit=a.hold_until_time_limit,
                        edge_feedback=a.edge_feedback, label=f'{a.controller}_{i:03d}') for i in range(a.episodes)]
    rng = np.random.default_rng(a.seed_base)
    fixed = [int(s) for s in a.seeds.split(',')] if a.seeds else [a.seed_base]
    for i, entry in enumerate(entries):
        if entry.get('controller') not in CONTROLLERS:
            raise SystemExit(f'episode {i}: controller must be one of {CONTROLLERS}')
        seed = entry.get('seed', 'random' if a.random_seeds else fixed[i % len(fixed)])
        if seed == 'random':
            entry['seed'], entry['seed_source'] = int(rng.integers(0, 2**31-1)), f'random(np.random.default_rng({a.seed_base}))'
        else:
            entry['seed'], entry['seed_source'] = int(seed), 'fixed'
        entry.setdefault('label', f"{entry['controller']}_{i:03d}")
        entry.setdefault('edge_feedback', 'nominal')
    return entries


def measured_edge_estimator(env):
    """Privileged measured tool-edge feedback (simulator truth of the held knife), checked against the obs time."""
    backend = env.backend

    def estimate(observation):
        measured = backend.measured() if hasattr(backend, 'measured') else backend.proprioception()
        if abs(float(measured.get('timestamp_s', env.time_s)) - float(observation['timestamp_s'])) > 1e-9:
            raise ValueError('Measured tool-edge feedback must match the observation time')
        return np.asarray(measured['tool_pose'], float).copy()
    return estimate


def make_controller(env, entry):
    """(callable obs -> (action, phase, tracking), done(), receipt, privileged inputs)."""
    kind = entry['controller']
    if kind == 'scripted':
        from robotic_cutting_sim import goal_planner
        from robotic_cutting_sim.policies import WaypointPolicy
        task = env.task
        options = task.controller['planner']
        waypoints = goal_planner.plan(task.scene, task.goal, task.tool_profile, options)
        policy_options = dict(goal_planner.policy_options(options))
        privileged = []
        if entry.get('edge_feedback') == 'measured':
            policy_options['edge_estimator'] = measured_edge_estimator(env)
            privileged.append('measured tool-edge pose (simulator tool truth) as waypoint feedback')
        policy = WaypointPolicy(task, waypoints, task.action_limits(), **policy_options)
        receipt = dict(kind=kind, planner=goal_planner.PLANNER_VERSION, planner_options=options,
                       controller_profile=goal_planner.CONTROLLER_PROFILES[options['controller_profile']],
                       phases=[w.label for w in waypoints], edge_feedback=entry.get('edge_feedback'),
                       stop_after_phase=entry.get('stop_after_phase'))

        def act(observation):
            action = policy(observation)
            return action, policy.phase, policy.last_tracking
        return act, (lambda: policy.done), receipt, privileged
    if kind == 'action_file':
        path = Path(entry['action_file'])
        program = ds.load_action_program(path)
        steps = int(round(env.task.duration_s/env.task.control_dt_s))
        rows = ds.expand_action_program(program, env.task.control_dt_s, steps)
        state = {'i': 0}

        def act(observation):
            label, action = rows[state['i']] if state['i'] < len(rows) else ('PROGRAM_END_HOLD', HOLD)
            state['i'] += 1
            return action, label, None
        receipt = dict(kind=kind, action_file=str(path), program_id=program.get('program_id'),
                       program_sha256=ds.sha256_file(path), expanded_steps=len(rows),
                       description=program.get('description'), uses_planner=False)
        return act, (lambda: False), receipt, []
    module, _, name = entry['policy'].partition(':')
    factory = getattr(importlib.import_module(module), name)
    policy = factory(env.task_spec(), entry['seed'])

    def act(observation):
        return policy(observation), 'POLICY', None
    return act, (lambda: False), dict(kind=kind, policy=entry['policy']), list(entry.get('privileged_inputs', []))


def run_episode(env, entry, index, out, a, clocks):
    start, memory_before = time.monotonic(), process_memory()
    observation, reset_info = env.reset(seed=entry['seed'], controller=entry['label'])
    act, done, receipt, privileged_inputs = make_controller(env, entry)
    receipt = dict(receipt, hold_until_time_limit=bool(entry.get('hold_until_time_limit')),
                   return_home=bool(entry.get('return_home')))
    task = env.task
    dt, duration = task.control_dt_s, task.duration_s
    stop = entry.get('stop_after_phase')
    seen_stop, stop_reason, statuses = False, None, {}
    leaks, clock_errors, camera_age_max = 0, 0, 0.
    steps = int(round(duration/dt))
    if entry.get('max_steps') is not None:   # smoke/debug budget: the episode then ends with STEP_BUDGET
        steps = min(steps, int(entry['max_steps']))

    def note(observation, info):
        nonlocal leaks, clock_errors, camera_age_max
        leaks += bool(find_privileged_keys(observation))
        executed = (info.get('action_executed') or {}).get('kind') != 'NOT_EXECUTED'
        clock_errors += bool(abs(observation['timestamp_s'] - env.time_s) > 1e-9
                             or (executed and abs(info['timestamp_s'] - info['action_timestamp_s'] - dt) > 1e-9))
        if observation.get('camera_age_s') is not None:
            camera_age_max = max(camera_age_max, float(observation['camera_age_s']))
        statuses[info['action_status']] = statuses.get(info['action_status'], 0) + 1
    leaks += bool(find_privileged_keys(observation))
    for _ in range(steps):
        if env.time_s >= duration - 1e-9:
            break
        action, phase, tracking = act(observation)
        base = None if phase is None else str(phase).split(':', 1)[0]
        if stop is not None:
            seen_stop |= base == stop
            if seen_stop and base != stop:
                stop_reason = f'STOP_AFTER_PHASE:{stop}'
                break
        if done():
            stop_reason = 'CONTROLLER_DONE'
            break
        observation, _, terminated, truncated, info = env.step(action, annotation=dict(policy_phase=phase,
                                                                                       policy_tracking=tracking))
        note(observation, info)
        if terminated or truncated:
            break
    if stop_reason and entry.get('hold_until_time_limit') and env.episode_state == 'RUNNING':
        while env.time_s < duration - 1e-9:
            observation, _, terminated, truncated, info = env.step(HOLD, annotation=dict(policy_phase='HOLD_AFTER_STOP'))
            note(observation, info)
            if terminated:
                break
    if env.episode_state == 'RUNNING':
        env.end_task(stop_reason or 'STEP_BUDGET')
    task_end = env.lifecycle.snapshot()
    evaluation_start = time.monotonic()
    result = env.evaluate()
    evaluation_wall = time.monotonic() - evaluation_start
    verdict = result.get('verdict', result.get('status'))
    home = None
    if entry.get('return_home'):
        home = env.return_home()
    material = env.backend.material
    codes = np.asarray(material.history_state()['codes'])
    recorder = env.recorder
    directory = Path(recorder.output)
    with (directory/'action_log.jsonl').open('w') as stream:
        for row in env.action_log:
            stream.write(json.dumps(jsonable(row)) + '\n')
    task_steps = sum(1 for row in env.action_log if row['executed_kind'] != 'NOT_EXECUTED')
    row = dict(episode_index=index, label=entry['label'], controller=entry['controller'], seed=entry['seed'],
               verdict=verdict, task_result=result, task_phase_end=task_end,
               home_outcome=None if home is None else home['home_outcome'])
    metrics = env.finish_recording(row)
    recorder_closed = bool(recorder.stream.closed and (directory/'result.json').exists())
    exported, export = ds.export_episode(directory)
    memory_after = process_memory()
    every = int(env.recording.get('every', 10)) if env.recording else int(a.record_every or 10)
    frequencies = dict(control_rate_hz=1./dt, control_dt_s=dt, recorded_camera_every_steps=every if env.cameras_available else None,
                       recorded_camera_rate_hz=(1./(every*dt)) if env.cameras_available else None,
                       policy_camera_every_steps=env.preset.camera_every if env.cameras_available else None,
                       policy_camera_rate_hz=(1./(env.preset.camera_every*dt)) if env.cameras_available else None,
                       physics_state_record_every_steps=every, measured_camera=export['cameras'],
                       note='Camera content is stored at recorded_camera_rate_hz; it is never 50 Hz unless every == 1')
    from robotic_cutting_sim.scene_config import configured_task_payload
    meta = dict(episode_id=f'episode_{index:03d}', dataset_version=ds.DATASET_VERSION, label=entry['label'],
                controller=entry['controller'], controller_receipt=receipt, expect=entry.get('expect'),
                seed=entry['seed'], seed_source=entry['seed_source'], task_id=task.task_id,
                config_sha256=task.config_sha256, configured_task_payload=configured_task_payload(task),
                scene=task.scene, goal=task.goal, workpiece=task.scene['workpiece'], supports=task.scene['supports'],
                tool_profile=task.tool_profile.to_dict(), runtime_source=env.runtime_source,
                runtime_options=clocks['runtime_options'], preset=env.preset.to_dict(),
                observation_profile=env.observation_profile, frequencies=frequencies,
                control_dt_s=dt, duration_s=duration, steps_recorded=int(metrics['steps']), task_steps=task_steps,
                task_result=jsonable(result), verdict=verdict, evaluation_wall_s=evaluation_wall,
                home_outcome=None if home is None else jsonable(home['home_outcome']),
                home_measurement=None if home is None else jsonable(home.get('measurement')),
                termination_reason=task_end.get('termination_reason'), task_end_reason=stop_reason,
                episode_state_final=env.lifecycle.snapshot(), episode_transitions=jsonable(env.lifecycle.history),
                action_status_counts=statuses, recorder_metrics=jsonable(metrics),
                physical_diagnostics=dict(tracking_error_m=metrics.get('tracking_error_m'),
                                          rotation_error_rad=metrics.get('rotation_error_rad'),
                                          events=metrics.get('events'),
                                          grasp_status_final=jsonable((env.backend.last_diagnostics or {}).get('grasp_status'))
                                          if isinstance(env.backend.last_diagnostics, dict) else None),
                reset_info=jsonable(reset_info), reset_checks=reset_info['reset_checks'],
                observation_privileged_leaks=leaks, clock_errors=clock_errors, camera_age_max_s=camera_age_max,
                recorder_closed=recorder_closed, final_codes_sha256=hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest(),
                privileged_controller_inputs=privileged_inputs, memory_before=memory_before, memory_after=memory_after,
                runtime_profile=getattr(env, 'runtime_profile', None),
                acceptance_valid=(None if getattr(env, 'runtime_profile', None) is None
                                  else bool(env.runtime_profile['acceptance_valid'])),
                wall_s=time.monotonic() - start, git=clocks['git'],
                physics_version=dict(solver=task.solver_module, cutting_law=task.cutting_law, max_cuts=task.max_cuts,
                                     material_receipt_sha256=hashlib.sha256(json.dumps(jsonable(material.parameter_receipt),
                                                                                         sort_keys=True).encode()).hexdigest(),
                                     particle_spacing_m=task.particle_spacing_m, grid_dx_m=task.grid_dx_m),
                export=dict(path=str(exported.relative_to(out)), sha256=export['export_sha256'], keys=export['keys']))
    write_json(directory/'episode.json', jsonable(meta))
    return meta


def live_preflight(a, entries, out, planner_options):
    """C2 geometric preflight before any Isaac launch: the scripted plan (per stop phase) and every action program."""
    import subprocess
    from robotic_cutting_sim.motion_preflight import run_cli
    from robotic_cutting_sim.policy_env import compile_policy_task
    from robotic_cutting_sim.scene_config import configured_task_payload
    task = compile_policy_task(a.scene, a.goal, tool_profile=a.tool_profile, planner_options=planner_options,
                               particle_spacing_m=a.spacing, seed=entries[0]['seed'])
    receipts, refused = [], False
    for until in sorted({e.get('stop_after_phase') or '' for e in entries if e['controller'] == 'scripted'}):
        receipt = run_cli('plan', configured_task_payload(task), until_phase=until or None, log_dir=out/'preflight')
        receipts.append(dict(kind='scripted_plan', until_phase=until or None, status=receipt.get('status'),
                             reason=receipt.get('reason')))
        refused |= receipt.get('status') != 'PASS'
    for path in sorted({e['action_file'] for e in entries if e['controller'] == 'action_file'}):
        target = out/'preflight'/f'action_program_{Path(path).stem}.json'
        command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/preflight_action_program.py'), '--scene', a.scene,
                   '--goal', a.goal, '--program', path, '--out', str(target)]
        if a.planner_options:
            command += ['--planner-options', a.planner_options]
        clean = {k: v for k, v in __import__('os').environ.items()
                 if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH')
                 and not k.startswith(('CARB_', 'OMNI_', 'EXP_PATH', 'ISAAC_'))}
        done = subprocess.run(command, cwd=ROOT, env=clean, capture_output=True, text=True, timeout=1800)
        status = json.loads(target.read_text()).get('status') if target.exists() else 'ERROR'
        receipts.append(dict(kind='action_program', program=path, status=status, receipt=str(target),
                             returncode=done.returncode))
        refused |= status != 'PASS'
    write_json(out/'preflight.json', dict(task_id=task.task_id, receipts=receipts, refused=refused))
    print(f'[collect] live preflight: {receipts}', flush=True)
    return refused


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runtime', choices=('kinematic', 'isaac_sim'), required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--goal', required=True)
    p.add_argument('--planner-options', help='JSON goal-planner option overrides (recorded in the task id)')
    p.add_argument('--tool-profile', help='Override the scene tool profile id (recorded in the task id)')
    p.add_argument('--spacing', type=float, help='Particle spacing override (coarse runs; recorded)')
    p.add_argument('--material-dt', type=float, help='Material dt (default 0.0001; a --runtime-profile fixes it)')
    p.add_argument('--device', default='cpu', help='Kinematic material device')
    p.add_argument('--gpu', type=int, default=1, help='Isaac Sim GPU')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--episode-plan', help=f'{PLAN_SCHEMA} JSON: per-episode controller/seed/options (overrides --episodes)')
    p.add_argument('--controller', choices=CONTROLLERS, default='scripted')
    p.add_argument('--action-file', help='rcs_action_program_v1 JSON for --controller action_file')
    p.add_argument('--policy', help='module:factory for --controller policy (factory(task_spec, seed) -> policy)')
    p.add_argument('--seeds', help='Comma-separated fixed seeds (cycled)')
    p.add_argument('--random-seeds', action='store_true', help='Draw seeds from np.random.default_rng(--seed-base)')
    p.add_argument('--seed-base', type=int, default=42)
    p.add_argument('--return-home', action='store_true')
    p.add_argument('--stop-after-phase')
    p.add_argument('--hold-until-time-limit', action='store_true')
    p.add_argument('--edge-feedback', choices=('nominal', 'measured'), default='nominal')
    p.add_argument('--record-every', type=int, help='Recorded frame interval in control steps (default 10, or the '
                                                     '--runtime-profile recording rate)')
    p.add_argument('--camera-every', type=int, help='Live: control steps between policy camera refreshes (default 1, or '
                                                     'the --runtime-profile value)')
    p.add_argument('--observation-profile', choices=('deployable', 'privileged_state'), default='deployable')
    p.add_argument('--history-stride', type=int, help='Oracle history stride (default 1)')
    p.add_argument('--runtime-profile', help='WP-11 Phase H4 runtime profile id (configs/wp11/runtime_profiles_v1.json: '
                                             'full_physics_v1 | research_fast_v1); default: none (unchanged behaviour)')
    p.add_argument('--stage-reset', choices=('reopen', 'reload_in_place'),
                   help='Isaac Sim: IsaacSimCuttingConfig.stage_reset (WP-11 Phase H1; default: the runtime default '
                        'or the --runtime-profile value)')
    p.add_argument('--profile', choices=('stages', 'kernels'),
                   help='Opt-in per-stage timing (robotic_cutting_sim.profiling); writes episode_###/profile.json. '
                        "'kernels' times every Warp kernel eagerly (perturbing; use for the split only)")
    p.add_argument('--skip-preflight', action='store_true', help='Isaac Sim: skip the C2 preflight (recorded; default: '
                                                                    'run it and refuse to launch unless every check passes)')
    p.add_argument('--reliability-config', action='append',
                   help='Reliability config(s); the first is primary (reliability.json), every one is also written as '
                        'reliability_<version>.json (default: configs/wp11/phaseG_reliability_v1.json)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    entries = episode_entries(a)
    out = new_output(a.out)
    git = manifest({}, 'probe')
    git = {key: git.get(key) for key in ('git_sha', 'git_dirty', 'packages', 'python')}
    planner_options = json.loads(a.planner_options) if a.planner_options else None
    if a.runtime_profile is None:   # unchanged defaults without a profile
        a.material_dt = .0001 if a.material_dt is None else a.material_dt
        a.history_stride = 1 if a.history_stride is None else a.history_stride
        a.record_every = 10 if a.record_every is None else a.record_every
        a.camera_every = 1 if a.camera_every is None else a.camera_every
    runtime_options = {key: value for key, value in dict(material_dt=a.material_dt, history_stride=a.history_stride).items()
                       if value is not None}
    if a.runtime_profile is not None:
        runtime_options['runtime_profile'] = a.runtime_profile
    if a.stage_reset is not None:
        if a.runtime != 'isaac_sim':
            raise SystemExit('--stage-reset applies to --runtime isaac_sim only')
        runtime_options['stage_reset'] = a.stage_reset
    if a.runtime == 'kinematic':
        runtime_options['device'] = a.device
    else:
        runtime_options.update(gpu=a.gpu, device=f'cuda:{a.gpu}')
    write_json(out/'dataset_manifest.json', manifest(jsonable(dict(
        dataset_version=ds.DATASET_VERSION, preset=PRESET_ID, args=vars(a), episodes=entries,
        runtime_options=runtime_options, privileged_controller_inputs_catalog=PRIVILEGED_CONTROLLER_INPUTS,
        reliability_config=json.loads(RELIABILITY_CONFIG.read_text()))), 'rcs_policy_dataset_collector'))
    if a.runtime == 'isaac_sim' and not a.skip_preflight:
        refused = live_preflight(a, entries, out, planner_options)
        if refused:
            return 2
    from robotic_cutting_sim.live_env import open_cutting_env
    index = dict(version=ds.DATASET_VERSION, preset=PRESET_ID, runtime=a.runtime, scene=a.scene, goal=a.goal,
                 created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), episodes=[])
    metas, failures, start = [], [], time.monotonic()
    clocks = dict(runtime_options=runtime_options, git=git)

    def write_index():
        write_json(out/'index.json', jsonable(index))

    try:
        with open_cutting_env(runtime=a.runtime, output_dir=out/'runtime', seed=entries[0]['seed'],
                              preset=PRESET_ID, scene=a.scene, goal=a.goal, planner_options=planner_options,
                              tool_profile=a.tool_profile, particle_spacing_m=a.spacing,
                              observation_profile=a.observation_profile,
                              recording=dict(root=out, **({} if a.record_every is None else dict(every=a.record_every))),
                              preset_overrides={} if a.camera_every is None else dict(camera_every=a.camera_every),
                              **runtime_options) as env:
            consecutive_errors = 0
            for i, entry in enumerate(entries):
                print(f'[collect] episode {i} {entry["label"]} controller={entry["controller"]} seed={entry["seed"]}',
                      flush=True)
                try:
                    profiler = None
                    if a.profile:
                        from robotic_cutting_sim.profiling import StageProfiler
                        device = runtime_options.get('device')
                        profiler = StageProfiler(sync_device=device if str(device).startswith('cuda') else None,
                                                 kernels=a.profile == 'kernels').install()
                    try:
                        meta = run_episode(env, entry, i, out, a, clocks)
                    finally:
                        if profiler is not None:
                            profiler.uninstall()
                    if profiler is not None:
                        summary = dict(profiler.summary(), episode_id=meta['episode_id'], mode=a.profile,
                                       wall_s=meta['wall_s'], runtime=a.runtime, device=runtime_options.get('device'))
                        write_json(out/meta['episode_id']/'profile.json', jsonable(summary))
                        meta['profile'] = dict(mode=a.profile, path=f"{meta['episode_id']}/profile.json")
                        write_json(out/meta['episode_id']/'episode.json', jsonable(meta))
                    consecutive_errors = 0
                except Exception as error:
                    consecutive_errors += 1
                    failure = dict(episode_index=i, label=entry['label'], error=repr(error),
                                   traceback=traceback.format_exc())
                    failures.append(failure)
                    write_json(out/f'episode_{i:03d}_failure.json', failure)
                    traceback.print_exc()
                    index['episodes'].append(dict(episode_id=f'episode_{i:03d}', dir=f'episode_{i:03d}',
                                                  label=entry['label'], controller=entry['controller'],
                                                  seed=entry['seed'], status='FAIL_EXECUTION', error=repr(error)))
                    write_index()
                    if consecutive_errors >= 2:
                        break
                    continue
                metas.append(meta)
                index['episodes'].append(dict(
                    episode_id=meta['episode_id'], dir=meta['episode_id'], label=meta['label'],
                    controller=meta['controller'], seed=meta['seed'], status='STORED', verdict=meta['verdict'],
                    expect=meta['expect'], termination_reason=meta['termination_reason'],
                    home_outcome=(meta['home_outcome'] or {}).get('status'), steps=meta['steps_recorded'],
                    export_sha256=meta['export']['sha256'], wall_s=meta['wall_s'],
                    **({} if meta.get('runtime_profile') is None else dict(
                        runtime_profile=meta['runtime_profile']['profile_id'],
                        acceptance_valid=meta['acceptance_valid']))))
                write_index()
                print(f'[collect] episode {i} verdict={meta["verdict"]} statuses={meta["action_status_counts"]} '
                      f'wall={meta["wall_s"]:.1f}s rss={meta["memory_after"]["host_rss_mib"]}', flush=True)
            validations = [ds.validate_episode(out, row['episode_id']) for row in index['episodes']
                           if row.get('status') == 'STORED']
            configs = [Path(c) for c in (a.reliability_config or [RELIABILITY_CONFIG])]
            for k, path in enumerate(configs):
                config = json.loads(path.read_text())
                scored = ds.reliability_report(metas, config, a.runtime, validations=validations,
                                               execution_failures=failures, planned_episodes=len(entries))
                scored.update(wall_s=time.monotonic() - start, config_path=str(path), primary=k == 0)
                write_json(out/f"reliability_{config['version']}.json", jsonable(scored))
                if k == 0:
                    report = scored
                    write_json(out/'reliability.json', jsonable(report))
                else:
                    print(f"[collect] also scored {config['version']}: POLICY_READY={scored['POLICY_READY']} "
                          f"failing={scored['failing']}", flush=True)
            write_index()
            print(f'[collect] POLICY_READY={report["POLICY_READY"]} failing={report["failing"]}', flush=True)
            status = 0 if report['POLICY_READY'] else 3
    except BaseException as error:
        write_json(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
