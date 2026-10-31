#!/usr/bin/env python3
"""WP-12 Phase 6: mixed-task multi-episode readiness run in ONE process through the public policy env API.

  # kinematic (GPU or CPU)
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase6.py episodes \
      --plan configs/wp12/phase6/kinematic_mixed_v1.json --device cuda:1 --out artifacts/wp12/phase6_NNN
  # Native Isaac Sim (C2 plan preflight of every scripted task first; refuses to launch on FAIL)
  OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/run_wp12_phase6.py \
      episodes --plan configs/wp12/phase6/native_cut_trim_cut_v1.json --gpu 0 --out artifacts/wp12/phase6_NNN
  # re-validate a finished run (read-only)
  .venv/bin/python scripts/run_wp12_phase6.py validate --root artifacts/wp12/phase6_NNN

One ``open_cutting_env(..., preset="cutting_research_v1", scene=, goal=, runtime_profile=, seed=)`` process runs
the declared episodes; tasks change between episodes through the opt-in ``PolicyReadyEnv.switch_task``
(``task_switch_v1``). Each episode is the collector's own ``collect_dataset.run_episode`` (reset -> controller ->
end task -> evaluate -> store -> export), so the storage layout is ``rcs_dataset_v1``. Controllers:
``scripted`` (goal planner + WaypointPolicy), ``independent_sequence`` (open-loop action program from the public
task_spec(), no planner) and ``probe_then_scripted`` (a declared list of invalid actions, then the scripted
controller: failed action followed by valid action). The harness (never the policy) records post-reset state
(labels, material, history, contact wrench, cut-event recorder, surface topology), the knife/camera mounts, the
failure classification and memory per episode, then writes ``phase6_report.json``. Nothing here changes
physics, a cutting law, an evaluator or a threshold.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import agentic_readiness as ar  # noqa: E402
from robotic_cutting_sim import dataset as ds  # noqa: E402
from robotic_cutting_sim.policy_env import PRESET_ID, jsonable, process_memory  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402

PLAN_SCHEMA = 'rcs_wp12_phase6_plan_v1'
CONTROLLERS = ('scripted', 'independent_sequence', 'probe_then_scripted')
# Invalid actions of the probe controller and the rejection each must receive (runtime-specific where noted).
PROBES = (
    ('non_unit_quaternion', {'type': 'edge_pose', 'target': [0., 0., .3, 0., 0., 0., 2.]}, 'REJECTED_INVALID_TARGET'),
    ('unknown_action_type', {'type': 'teleport', 'target': [0.]*7}, 'REJECTED_INVALID_SCHEMA'),
    ('unknown_key', {'type': 'edge_delta', 'target': [0.]*6, 'force_n': 5.}, 'REJECTED_INVALID_SCHEMA'),
    ('cut_index_out_of_range', {'type': 'edge_delta', 'target': [0.]*6, 'cut_index': 99}, 'REJECTED_CUT_INDEX_OUT_OF_RANGE'),
    ('non_finite_delta', {'type': 'edge_delta', 'target': [float('nan')]+[0.]*5}, 'REJECTED_INVALID_TARGET'),
    ('gripper_release', {'gripper': 'release'}, {'kinematic': 'REJECTED_GRIPPER_UNSUPPORTED',
                                                 'isaac_sim': 'REJECTED_GRIPPER_LOCKED'}),
)


def load_collector():
    spec = importlib.util.spec_from_file_location('rcs_collect_dataset_phase6', ROOT/'scripts/collect_dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != PLAN_SCHEMA:
        raise SystemExit(f'--plan must be an {PLAN_SCHEMA} document')
    for i, e in enumerate(plan['episodes']):
        if e.get('controller') not in CONTROLLERS or e.get('task') not in plan['tasks']:
            raise SystemExit(f'episode {i}: controller must be one of {CONTROLLERS} and task one of {sorted(plan["tasks"])}')
        e.setdefault('label', f"{e['task']}_{e['controller']}_{i:03d}")
        e.setdefault('seed', int(plan.get('seed', 42)))
        e['seed_source'] = 'fixed (plan)'
        e.setdefault('edge_feedback', 'nominal')
    return plan


def task_args(plan, name):
    row = plan['tasks'][name]
    return dict(scene=row['scene'], goal=row['goal'], planner_options=row.get('planner_options'))


def make_controller_factory(collector, runtime, state):
    """collect_dataset.make_controller extended with the Phase 6 controllers (patched in for this process only)."""
    original = collector.make_controller

    def make(env, entry):
        state['post_reset'] = ar.post_reset_state(env, state.get('previous_reset'))
        state['attachment_reset'] = ar.attachment_snapshot(env)
        state['probe_rows'] = []
        kind = entry['controller']
        if kind == 'scripted':
            return original(env, entry)
        if kind == 'probe_then_scripted':
            act, done, receipt, privileged = original(env, dict(entry, controller='scripted'))
            probes = [p for p in PROBES if p[0] in entry.get('probes', [p[0] for p in PROBES])]
            queue = list(probes)

            def probe_act(observation):
                if queue:
                    name, action, expected = queue.pop(0)
                    expected = expected[runtime] if isinstance(expected, dict) else expected
                    state['probe_rows'].append(dict(name=name, expected_status=expected, control_step=len(state['probe_rows'])))
                    return copy.deepcopy(action), f'PROBE:{name}', None
                return act(observation)
            return probe_act, done, dict(receipt, kind=kind, probes=[p[0] for p in probes]), privileged
        if kind == 'independent_sequence':
            spec = env.task_spec()
            start = np.asarray(env.observe()['edge_command_world_xyzw'], float)
            steps = int(round(env.task.duration_s/env.task.control_dt_s))
            program = ar.independent_through_cut_program(spec, start, **entry.get('program_options', {}))
            rows = ds.expand_action_program(program, env.task.control_dt_s, steps)
            cursor = {'i': 0}

            def act(observation):
                label, action = rows[cursor['i']] if cursor['i'] < len(rows) else ('PROGRAM_END_HOLD', collector.HOLD)
                cursor['i'] += 1
                return action, label, None
            receipt = dict(kind=kind, program_id=program['program_id'], expanded_steps=len(rows), uses_planner=False,
                           inputs='env.task_spec() (scene/goal configs, world_from_workpiece) and the deployable '
                                  'edge command of the reset observation', program=program)
            return act, (lambda: False), receipt, []
        raise ValueError(f'Unknown Phase 6 controller {kind!r}')
    return make


def watch_wp12_leaks(env):
    """Walk every reset/step observation and info the policy receives with agentic_readiness.leak_paths
    (PRIVILEGED_INFO_KEYS | LEAK_KEYS_WP12) by wrapping this env instance's reset/step; returns the counters."""
    counters = dict(observations=0, infos=0, paths=[])
    # resolve the class methods at call time, so an installed StageProfiler (class-level patches) still sees them
    reset = lambda *args, **kwargs: type(env).reset(env, *args, **kwargs)  # noqa: E731
    step = lambda *args, **kwargs: type(env).step(env, *args, **kwargs)  # noqa: E731

    def note(observation, info, where):
        for tree, key in ((observation, 'observations'), (info, 'infos')):
            paths = ar.leak_paths(tree)
            counters[key] += bool(paths)
            counters['paths'] += [f'{where}:{p}' for p in paths if f'{where}:{p}' not in counters['paths']][:20]

    def wrapped_reset(*args, **kwargs):
        observation, info = reset(*args, **kwargs)
        note(observation, info, 'reset')
        return observation, info

    def wrapped_step(*args, **kwargs):
        result = step(*args, **kwargs)
        note(result[0], result[4], 'step')
        return result
    env.reset, env.step = wrapped_reset, wrapped_step
    return counters


def live_preflight(plan, out):
    """C2 plan preflight of every scripted task (Native only); returns (refused, receipts)."""
    from robotic_cutting_sim.motion_preflight import run_cli
    from robotic_cutting_sim.policy_env import compile_policy_task
    from robotic_cutting_sim.runtime_profiles import profile_scene, runtime_profile
    from robotic_cutting_sim.scene_config import configured_task_payload
    receipts, refused = [], False
    for name in sorted({e['task'] for e in plan['episodes']}):
        args = task_args(plan, name)
        scene = args['scene']
        if plan.get('runtime_profile'):
            scene = profile_scene(scene, runtime_profile(plan['runtime_profile']))
        task = compile_policy_task(scene, args['goal'], planner_options=args['planner_options'],
                                   seed=int(plan.get('seed', 42)))
        receipt = run_cli('plan', configured_task_payload(task), log_dir=out/'preflight')
        receipts.append(dict(task=name, task_id=task.task_id, status=receipt.get('status'), reason=receipt.get('reason')))
        refused |= receipt.get('status') != 'PASS'
    write_json(out/'preflight.json', dict(receipts=receipts, refused=refused))
    return refused, receipts


def probe_report(meta, rows, runtime):
    """Expected vs actual status of every probe step and the first valid step after them (from action_log)."""
    log = ds.read_jsonl(Path(meta['_dir'])/'action_log.jsonl')
    out = []
    for i, row in enumerate(rows):
        actual = log[i] if i < len(log) else {}
        out.append(dict(row, actual_status=actual.get('status'), executed_kind=actual.get('executed_kind'),
                        ok=actual.get('status') == row['expected_status']
                        and actual.get('executed_kind') == 'HOLD_AFTER_REJECTION'))
    after = log[len(rows)] if len(log) > len(rows) else {}
    return dict(probes=out, first_valid_after=dict(status=after.get('status'), executed_kind=after.get('executed_kind')),
                all_rejected_as_declared=all(r['ok'] for r in out),
                valid_action_executes_after_rejections=after.get('status') in ('ACCEPTED', 'CLAMPED')
                and after.get('executed_kind') == 'REQUESTED')


def run(a):
    plan = load_plan(a.plan)
    runtime = plan['runtime']
    if runtime not in ('kinematic', 'isaac_sim'):
        raise SystemExit('plan runtime must be kinematic or isaac_sim')
    out = new_output(a.out)
    collector = load_collector()
    git = manifest({}, 'probe')
    git = {key: git.get(key) for key in ('git_sha', 'git_dirty', 'packages', 'python')}
    runtime_options = {}
    if plan.get('runtime_profile'):
        runtime_options['runtime_profile'] = plan['runtime_profile']
    else:
        runtime_options['material_dt'] = float(plan['material_dt'])
    if plan.get('surface_profile'):
        runtime_options['surface_profile'] = plan['surface_profile']
    for key in ('stage_reset', 'surface_update'):
        if plan.get(key):
            runtime_options[key] = plan[key]
    if runtime == 'kinematic':
        runtime_options['device'] = a.device or plan.get('device', 'cpu')
    else:
        runtime_options.update(gpu=a.gpu, device=f'cuda:{a.gpu}')
    episodes = plan['episodes'] if a.episodes is None else plan['episodes'][:a.episodes]
    if a.max_steps is not None:
        episodes = [dict(e, max_steps=a.max_steps) for e in episodes]
    write_json(out/'phase6_manifest.json', manifest(jsonable(dict(
        version=ar.READINESS_VERSION, plan=plan, plan_path=str(a.plan), args=vars(a), runtime_options=runtime_options,
        episodes=episodes, dataset_version=ds.DATASET_VERSION, preset=PRESET_ID)), 'wp12_phase6_readiness'))
    if runtime == 'isaac_sim' and not a.skip_preflight:
        refused, receipts = live_preflight(plan, out)
        print(f'[phase6] preflight {receipts}', flush=True)
        if refused:
            return 2
    index = dict(version=ds.DATASET_VERSION, preset=PRESET_ID, runtime=runtime, scene='mixed (per episode)',
                 goal='mixed (per episode)', created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), episodes=[])

    def write_index():
        write_json(out/'index.json', jsonable(index))
    state, rows, failures = {}, [], []
    collector.make_controller = make_controller_factory(collector, runtime, state)
    clocks = dict(runtime_options=runtime_options, git=git)
    ns = argparse.Namespace(record_every=int(plan.get('record_every', 10)))
    first = task_args(plan, episodes[0]['task'])
    from robotic_cutting_sim.live_env import open_cutting_env
    start = time.monotonic()
    current = episodes[0]['task']
    status = 0
    recording = dict(root=out, every=int(plan.get('record_every', 10)))
    preset_overrides = dict(plan.get('preset_overrides') or {})
    try:
        with open_cutting_env(runtime=runtime, output_dir=out/'runtime', seed=int(episodes[0]['seed']),
                              preset=PRESET_ID, recording=recording, preset_overrides=preset_overrides,
                              **first, **runtime_options) as env:
            leaks = watch_wp12_leaks(env)
            for i, entry in enumerate(episodes):
                switch = None
                if entry['task'] != current:
                    switch = env.switch_task(**task_args(plan, entry['task']), seed=int(entry['seed']))
                    current = entry['task']
                leaks.update(observations=0, infos=0, paths=[])
                print(f"[phase6] episode {i} {entry['label']} task={entry['task']} controller={entry['controller']} "
                      f"seed={entry['seed']}", flush=True)
                wall = time.monotonic()
                profiler = None
                try:
                    if a.profile:
                        from robotic_cutting_sim.profiling import StageProfiler
                        device = runtime_options.get('device')
                        profiler = StageProfiler(sync_device=device if str(device).startswith('cuda') else None,
                                                 kernels=a.profile == 'kernels').install()
                    try:
                        meta = collector.run_episode(env, entry, i, out, ns, clocks)
                    finally:
                        if profiler is not None:
                            profiler.uninstall()
                    if profiler is not None:
                        write_json(out/meta['episode_id']/'profile.json', jsonable(dict(
                            profiler.summary(), episode_id=meta['episode_id'], mode=a.profile, wall_s=meta['wall_s'],
                            runtime=runtime, device=runtime_options.get('device'))))
                except Exception as error:
                    failure = dict(episode_index=i, label=entry['label'], error=repr(error), traceback=traceback.format_exc())
                    failures.append(failure)
                    write_json(out/f'episode_{i:03d}_failure.json', failure)
                    traceback.print_exc()
                    index['episodes'].append(collector.failure_row(i, entry, repr(error)))
                    write_index()
                    if len(failures) >= 2:
                        break
                    continue
                meta['_dir'] = str(out/meta['episode_id'])
                cls, reason = ar.classify_episode(meta)
                sidecar = dict(version=ar.READINESS_VERSION, episode_id=meta['episode_id'], task=entry['task'],
                               task_id=meta['task_id'], controller=entry['controller'], label=entry['label'],
                               task_switch=switch, post_reset=state.get('post_reset'),
                               attachment_reset=state.get('attachment_reset'),
                               attachment_end=ar.attachment_snapshot(env),
                               failure_classification=dict(**{'class': cls}, reason=reason),
                               probe=probe_report(meta, state['probe_rows'], runtime) if state.get('probe_rows') else None,
                               wp12_leak_walk=dict(leaks, keys='PRIVILEGED_INFO_KEYS | LEAK_KEYS_WP12'),
                               memory_after=meta['memory_after'], episode_wall_s=time.monotonic()-wall,
                               surface_summary=(meta.get('surface') or {}).get('summary'),
                               scope='harness-side (privileged) receipts; never a policy input')
                write_json(out/meta['episode_id']/'phase6.json', jsonable(sidecar))
                state['previous_reset'] = state.get('post_reset')
                index['episodes'].append(collector.index_row(meta))
                write_index()
                rows.append(dict(meta=meta, phase6=sidecar))
                print(f"[phase6] episode {i} verdict={meta['verdict']} class={cls} wall={meta['wall_s']:.1f}s "
                      f"rss={meta['memory_after']['host_rss_mib']} gpu={meta['memory_after']['gpu_used_mib_total']}",
                      flush=True)
            report = score(out, plan, rows, failures, runtime, time.monotonic()-start, planned=len(episodes))
            write_index()
            status = 0 if report['status'] == 'PASS' else 3
    except BaseException as error:
        write_json(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
        status = 1
    return status


def _slope(values):
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return None
    return float(np.polyfit(np.arange(len(values), dtype=float), np.asarray(values), 1)[0])


def score(out, plan, rows, failures, runtime, wall_s, limits=None, planned=None, report_name='phase6_report.json'):
    """phase6_report.json from the stored episodes (read-only over what run() wrote)."""
    limits = limits or plan.get('limits') or {}
    episodes, validations = [], []
    for row in rows:
        meta, side = row['meta'], row['phase6']
        validation = ds.validate_episode(out, meta['episode_id'])
        record = ar.validate_episode_dir(out/meta['episode_id'])
        validations.append(dict(episode_id=meta['episode_id'], dataset=validation['status'],
                                dataset_failing=validation['failing'], spec_6_4=record['status'],
                                spec_6_4_failing=record['failing'], spec_6_4_not_applicable=record['not_applicable']))
        if not (out/meta['episode_id']/'spec_6_4_validation.json').exists():
            write_json(out/meta['episode_id']/'spec_6_4_validation.json', jsonable(record))
        episodes.append(dict(episode_id=meta['episode_id'], label=meta['label'], task=side['task'], task_id=meta['task_id'],
                             controller=side['controller'], seed=meta['seed'], verdict=meta['verdict'],
                             stop_after_phase=(meta.get('controller_receipt') or {}).get('stop_after_phase'),
                             expect=meta.get('expect'), failure_class=side['failure_classification']['class'],
                             reason=side['failure_classification']['reason'],
                             post_reset_failing=(side['post_reset'] or {}).get('failing'),
                             leaks=dict(observation=meta['observation_privileged_leaks'],
                                        info=meta['info_privileged_leaks']),
                             clock_errors=meta['clock_errors'], action_status_counts=meta['action_status_counts'],
                             probe=side['probe'], task_switch=bool(side['task_switch']),
                             final_codes_sha256=meta['final_codes_sha256'],
                             labelled_final=None, wall_s=meta['wall_s'],
                             host_rss_mib=meta['memory_after']['host_rss_mib'],
                             gpu_mib=meta['memory_after']['gpu_used_mib_total']))
    groups = {}
    for e in episodes:
        groups.setdefault((e['task_id'], e['controller'], e['seed'], json.dumps(e.get('probe') is not None),
                           e.get('stop_after_phase')), []).append(e)
    cpu = runtime == 'kinematic' and all(r['meta']['runtime_options'].get('device') == 'cpu' for r in rows)
    repeat = [dict(episodes=[e['episode_id'] for e in g], task_id=k[0], controller=k[1], seed=k[2],
                   final_codes_identical=len({e['final_codes_sha256'] for e in g}) == 1,
                   verdict_identical=len({e['verdict'] for e in g}) == 1,
                   rule='CPU deterministic: exact final codes' if cpu else 'verdict-level (GPU float32 atomics)')
              for k, g in groups.items() if len(g) > 1]
    rss = [e['host_rss_mib'] for e in episodes]
    gpu = [e['gpu_mib'] for e in episodes]
    fit_from = int(limits.get('memory_fit_from_episode', 1))
    drift = ar.attachment_drift([s for r in rows for s in (r['phase6']['attachment_reset'], r['phase6']['attachment_end'])])
    checks = {
        'all_planned_episodes_stored': not failures and len(rows) == (len(plan['episodes']) if planned is None
                                                                      else planned),
        'every_episode_validates_dataset': all(v['dataset'] == 'PASS' for v in validations),
        'every_episode_validates_spec_6_4': all(v['spec_6_4'] == 'PASS' for v in validations),
        'no_privileged_keys_in_observations_or_info': all(e['leaks']['observation'] == 0 and e['leaks']['info'] == 0
                                                         for e in episodes),
        'no_wp12_surface_material_geometry_keys_in_observations_or_info': all(
            (r['phase6'].get('wp12_leak_walk') or {}).get('observations', 0) == 0
            and (r['phase6'].get('wp12_leak_walk') or {}).get('infos', 0) == 0 for r in rows),
        'no_clock_errors': all(e['clock_errors'] == 0 for e in episodes),
        'every_reset_clean': all(not e['post_reset_failing'] for e in episodes),
        'every_episode_evaluated': all(e['verdict'] is not None for e in episodes),
        'expected_outcomes': all(e['expect'] is None or e['verdict'] == e['expect'] for e in episodes),
        'probes_rejected_and_valid_action_follows': all(
            e['probe']['all_rejected_as_declared'] and e['probe']['valid_action_executes_after_rejections']
            for e in episodes if e['probe']),
        'repeat_groups': all((g['final_codes_identical'] if cpu else g['verdict_identical']) for g in repeat),
    }
    slope_rss, slope_gpu = _slope(rss[fit_from:]), _slope(gpu[fit_from:])
    if 'host_rss_slope_max_mib_per_episode' in limits:
        checks['host_memory_bounded'] = slope_rss is not None and slope_rss <= limits['host_rss_slope_max_mib_per_episode']
    if 'gpu_slope_max_mib_per_episode' in limits:
        checks['gpu_memory_bounded'] = (slope_gpu is None and all(v is None for v in gpu)) or (
            slope_gpu is not None and slope_gpu <= limits['gpu_slope_max_mib_per_episode'])
    if runtime == 'isaac_sim':
        checks['no_tool_detachment'] = all((r['phase6']['attachment_end'].get('grasp') or {}).get('status') not in
                                           ('GRASP_LOST', 'LOST') for r in rows) and all(
            e['failure_class'] != 'TERMINATED_GRASP' for e in episodes)
        checks['tool_in_hand_drift_bounded'] = drift.get('tool_in_hand_translation_drift_max_m', 0.) <= float(
            limits.get('tool_in_hand_drift_max_m', .002))
        checks['no_camera_attachment_drift'] = drift.get('camera_mount_max_abs_change', 0.) <= 1e-9 and drift.get(
            'camera_measured_vs_usd_translation_max_m', 0.) <= 1e-5
    failing = sorted(k for k, ok in checks.items() if not ok)
    report = dict(version=ar.READINESS_VERSION, runtime=runtime, status='PASS' if not failing else 'FAIL',
                  checks=checks, failing=failing, episodes=episodes, validations=validations, repeatability=repeat,
                  memory=dict(host_rss_mib_after_episode=rss, gpu_used_mib_after_episode=gpu, fit_from_episode=fit_from,
                              host_rss_slope_mib_per_episode=slope_rss, gpu_slope_mib_per_episode=slope_gpu,
                              estimator='unweighted OLS over episodes fit_from..end (numpy.polyfit)'),
                  attachment_drift=drift, execution_failures=failures, wall_s=wall_s,
                  determinism_scope=('CPU kinematic: bitwise; GPU kinematic and Native: verdict-level only (float32 '
                                     'atomics); numerical variation is reported in repeatability, never gated'),
                  scope='Readiness of the API/lifecycle/storage, not a task-success or policy-quality claim')
    write_json(out/report_name, jsonable(report))
    print(f"[phase6] status={report['status']} failing={failing}", flush=True)
    return report


def rescore(a):
    """Re-score a finished run from its stored files (read-only over the episodes); never overwrites a report."""
    root = Path(a.root)
    manifest = json.loads((root/'phase6_manifest.json').read_text())['config']
    rows = []
    for row in json.loads((root/'index.json').read_text())['episodes']:
        if row.get('status') == 'STORED':
            meta = json.loads((root/row['dir']/'episode.json').read_text())
            rows.append(dict(meta=meta, phase6=json.loads((root/row['dir']/'phase6.json').read_text())))
    failures = [json.loads(p.read_text()) for p in sorted(root.glob('episode_*_failure.json'))]
    target = root/f'phase6_report_rescore_{a.tag}.json'
    if target.exists():
        raise SystemExit(f'{target} exists; never overwritten')
    report = score(root, manifest['plan'], rows, failures, manifest['plan']['runtime'], None,
                   planned=len(manifest['episodes']), report_name=target.name)
    return 0 if report['status'] == 'PASS' else 3


def validate(a):
    root = Path(a.root)
    rows = []
    for d in sorted(root.glob('episode_[0-9][0-9][0-9]')):
        if d.is_dir():
            record = ar.validate_episode_dir(d)
            rows.append(dict(episode=d.name, status=record['status'], failing=record['failing'],
                             not_applicable=record['not_applicable']))
    out = dict(version=ar.EPISODE_VALIDATOR_VERSION, root=str(root), episodes=rows,
               status='PASS' if rows and all(r['status'] == 'PASS' for r in rows) else 'FAIL')
    print(json.dumps(out, indent=1))
    if a.out:
        write_json(Path(a.out), out)
    return 0 if out['status'] == 'PASS' else 3


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('episodes')
    e.add_argument('--plan', required=True)
    e.add_argument('--out', type=Path, required=True)
    e.add_argument('--device', help='kinematic material device (default: plan device or cpu)')
    e.add_argument('--gpu', type=int, default=0, help='Isaac Sim GPU')
    e.add_argument('--episodes', type=int, help='run only the first N plan episodes (smoke; recorded)')
    e.add_argument('--max-steps', type=int, help='step budget per episode (smoke; recorded; the episode ends STEP_BUDGET)')
    e.add_argument('--skip-preflight', action='store_true')
    e.add_argument('--profile', choices=('stages', 'kernels'), help='opt-in StageProfiler per episode (profile.json)')
    r = sub.add_parser('rescore')
    r.add_argument('--root', required=True)
    r.add_argument('--tag', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--root', required=True)
    v.add_argument('--out')
    a = p.parse_args(argv)
    return {'episodes': run, 'rescore': rescore, 'validate': validate}[a.command](a)


if __name__ == '__main__':
    sys.exit(main())
