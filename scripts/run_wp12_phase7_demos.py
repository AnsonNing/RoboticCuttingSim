#!/usr/bin/env python3
"""WP-12 Phase 7: Native Demos 1-6 exactly as PREDECLARED in configs/wp12/phase7_demos_v1.json.

Each demo runs twice on identical inputs (same scene, goal, planner overrides, duration, history stride, seed):
first ``particle_surface_v1`` (the regression record), then ``cut_aware_surface_v1``. Each run is a live Native
Isaac Sim 6.2 episode through the unchanged Phase 2b path ``scripts/run_wp10_live.py --runtime isaac_sim
--scene/--goal`` with the declared flags plus ``--surface-profile P --surface-timing-log`` (as surface_live_v1).
Never a kinematic replay; never cut_aware_surface_v2. Nothing here changes physics, evaluators or thresholds.

Subcommands (outputs are new directories; nothing is ever overwritten; failed runs are kept):

* ``check``: verify the demo file, the source hashes and compile every demo's live inputs (CPU only, no write).
* ``launch --demo DEMO2 --profile particle_surface_v1 --root artifacts/wp12/demos_001``: prepare into
  ``ROOT/<DEMO>_<A|B>/inputs`` and run one Native episode into ``.../run``.
* ``queue --root R --demos DEMO2 DEMO6 ...``: for each demo, A then B, one Native process at a time, waiting for GPU
  memory headroom before each launch.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase7_demos.py queue \\
      --root artifacts/wp12/demos_001 --demos DEMO2 DEMO6 DEMO1 DEMO3 DEMO4 DEMO5
"""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT/'configs/wp12/phase7_demos_v1.json'
PROFILES = {'particle_surface_v1': 'A', 'cut_aware_surface_v1': 'B'}
# The Phase 2b record of the same scene/goal (rerun demos): the live task id must equal it.
PHASE2B_RECORD = {'DEMO1': 'artifacts/wp12/workcell_live_001', 'DEMO3': 'artifacts/wp12/workcell_live_003',
                  'DEMO4': 'artifacts/wp12/workcell_live_004', 'DEMO5': 'artifacts/wp12/workcell_live_005'}


def workcell_module():
    spec = importlib.util.spec_from_file_location('rcs_run_wp12_workcell_live', ROOT/'scripts/run_wp12_workcell_live.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WC = workcell_module()


def load_demos():
    demos = json.loads(DEMOS.read_text())
    if demos.get('status') != 'PREDECLARED':
        raise SystemExit(f'{DEMOS.name} status {demos.get("status")!r} is not PREDECLARED')
    return demos


def demo_spec(demos, demo_id):
    for spec in demos['demos']:
        if spec['id'] == demo_id:
            if 'scene' not in spec:
                raise SystemExit(f'{demo_id} has no fixed inputs (Demo 7 comes from the Phase 5 held-out selection)')
            return spec
    raise KeyError(demo_id)


def pseudo_protocol(demos):
    """The phase7 common block in the shape WC.compile_live reads (planner overrides, duration rule, seed)."""
    common = demos['common']
    return {'controller': {'planner_overrides': common['planner_overrides'], 'duration_rule': common['duration_rule']},
            'runtime': {**common['runtime'], 'seed': common['physics']['seed']}}


def verify_sources(spec):
    problems = []
    for kind in ('scene', 'goal'):
        got = WC.sha256(ROOT/spec[kind])
        if got != spec[f'{kind}_sha256']:
            problems.append(f'{kind} {spec[kind]} sha256 {got} != declared {spec[f"{kind}_sha256"]}')
    return problems


def check_physics(demos, task, scene):
    phys = demos['common']['physics']
    solver = scene.get('solver') or {}
    problems = []
    if task.solver_module != phys['solver']:
        problems.append(f'solver {task.solver_module} != {phys["solver"]}')
    for key, want in (('cutting_law', phys['cutting_law']), ('particle_spacing_m', phys['particle_spacing_m']),
                      ('grid_dx_m', phys['grid_dx_m'])):
        if key in solver and solver[key] != want:
            problems.append(f'scene solver.{key} {solver[key]} != {want}')
    if scene['workpiece'].get('material_profile') != phys['material_profile']:
        problems.append(f'material_profile {scene["workpiece"].get("material_profile")} != {phys["material_profile"]}')
    return problems


def prepare(demos, demo_id, out):
    from robotic_cutting_sim import goal_planner
    spec = demo_spec(demos, demo_id)
    problems = verify_sources(spec)
    if problems:
        raise SystemExit('; '.join(problems))
    protocol = pseudo_protocol(demos)
    built = WC.compile_live(protocol, {'id': demo_id, 'scene': spec['scene'], 'goal': spec['goal']})
    if built['goal_diff_vs_source'] not in ([], ['episode.duration_s']):
        raise SystemExit(f'Live goal differs from the source goal beyond the duration: {built["goal_diff_vs_source"]}')
    task = built['task']
    problems = check_physics(demos, task, built['scene'])
    if problems:
        raise SystemExit('; '.join(problems))
    _, report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, task.controller['planner'])
    record = PHASE2B_RECORD.get(demo_id)
    recorded_id = None
    if record is not None:
        recorded_id = json.loads((ROOT/record/'inputs'/'prepare.json').read_text())['live_task_id']
        if recorded_id != task.task_id:
            raise SystemExit(f'{demo_id}: live task id {task.task_id} != Phase 2b record {recorded_id} ({record})')
    receipt = {'episode': demo_id, 'demo': demo_id, 'spec': spec, 'demos_file': str(DEMOS.relative_to(ROOT)),
               'demos_file_sha256': WC.sha256(DEMOS), 'source_scene_sha256': WC.sha256(ROOT/spec['scene']),
               'source_goal_sha256': WC.sha256(ROOT/spec['goal']), 'phase2b_record': record,
               'phase2b_task_id': recorded_id, 'live_task_id': task.task_id, 'live_config_sha256': task.config_sha256,
               'solver_module': task.solver_module, 'max_cuts': task.max_cuts,
               'particles': int(task.sample_particles()[0].shape[0]), 'live_plan_end_s': built['plan_end_s'],
               'duration_s': built['duration_s'], 'steps': built['steps'], 'history_stride': built['history_stride'],
               'record_every': demos['common']['runtime']['record_every'],
               'goal_diff_vs_source': built['goal_diff_vs_source'], 'planner_overrides': built['options'],
               'planner_options_resolved': task.controller['planner'], 'plan_report_operations': report.get('operations'),
               'phases_live': list(dict.fromkeys(task.reference_phase_labels)), 'workcell_geometry': task.workcell_geometry(),
               'compiled_with': sys.executable, **WC.git_state(), 'prepared': datetime.now().astimezone().isoformat()}
    if out is None:
        return receipt, built
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    WC.write_new(out/'scene.json', built['scene'])
    WC.write_new(out/'goal.json', built['goal'])
    WC.write_new(out/'prepare.json', receipt)
    return receipt, built


def cmd_check(args):
    demos = load_demos()
    ok = True
    for spec in demos['demos']:
        if 'scene' not in spec:
            print(json.dumps({'demo': spec['id'], 'status': 'NOT_FIXED_INPUTS'}))
            continue
        try:
            receipt, _ = prepare(demos, spec['id'], None)
            print(json.dumps({k: receipt[k] for k in ('demo', 'live_task_id', 'phase2b_task_id', 'live_plan_end_s',
                                                      'duration_s', 'steps', 'history_stride', 'max_cuts', 'particles')}))
        except SystemExit as error:
            ok = False
            print(json.dumps({'demo': spec['id'], 'problem': str(error)}))
    return 0 if ok else 1


def gpu_free_mib(gpu):
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        index, free = (x.strip() for x in line.split(','))
        if int(index) == int(gpu):
            return int(free)
    return None


def launch(demos, demo_id, profile, root, disk_floor_gb=15., gpu_floor_mib=12000, wait_s=60):
    if profile not in PROFILES:
        raise SystemExit(f'profile {profile} is not one of {sorted(PROFILES)} (never cut_aware_surface_v2)')
    root = Path(root)
    root = root if root.is_absolute() else ROOT/root
    out = root/f'{demo_id}_{PROFILES[profile]}'
    out.mkdir(parents=True, exist_ok=False)
    prep, _ = prepare(demos, demo_id, out/'inputs')
    pair_note = None
    if PROFILES[profile] == 'B':
        a_inputs = root/f'{demo_id}_A'/'inputs'
        if not a_inputs.exists():
            WC.write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'{a_inputs} missing: particle_surface_v1 runs first'})
            return 3
        diffs = [n for n in ('scene.json', 'goal.json') if WC.sha256(a_inputs/n) != WC.sha256(out/'inputs'/n)]
        a_prep = json.loads((a_inputs/'prepare.json').read_text())
        for key in ('live_task_id', 'steps', 'history_stride', 'planner_overrides', 'duration_s'):
            if a_prep[key] != prep[key]:
                diffs.append(key)
        if diffs:
            WC.write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'inputs differ from the A run: {diffs}'})
            return 3
        pair_note = f'inputs identical to {a_inputs.relative_to(ROOT)} (scene.json, goal.json sha256; task id, steps, stride)'
    free = shutil.disk_usage(out).free/1e9
    if free < disk_floor_gb:
        WC.write_new(out/'attempt.json', {'status': 'NOT_RUN', 'reason': f'disk {free:.1f} GB < floor {disk_floor_gb} GB'})
        return 3
    gpu = demos['common']['runtime']['gpu']
    waited = 0.
    while (gpu_free_mib(gpu) or 0) < gpu_floor_mib:
        print(json.dumps({'waiting_for_gpu_memory': gpu_free_mib(gpu), 'floor_mib': gpu_floor_mib}), flush=True)
        time.sleep(wait_s)
        waited += wait_s
    run_dir = out/'run'
    command = WC.runner_command(pseudo_protocol(demos), prep, (out/'inputs').relative_to(ROOT), run_dir.relative_to(ROOT))
    command += ['--surface-profile', profile, '--surface-timing-log']
    tmpdir = demos['common']['runtime']['tmpdir']
    env = {**os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES', 'PYTHONUNBUFFERED': '1', 'TMPDIR': tmpdir}
    log = ROOT/'artifacts/wp12/logs'/f'{root.name}_{out.name}.log'
    if log.exists():
        raise FileExistsError(log)
    attempt = {'demo': demo_id, 'profile': profile, 'pair': pair_note, 'demos_file': str(DEMOS.relative_to(ROOT)),
               'demos_file_sha256': WC.sha256(DEMOS), 'live_task_id': prep['live_task_id'], 'command': command,
               'cwd': str(ROOT), 'env': {k: env[k] for k in ('OMNI_KIT_ACCEPT_EULA', 'PYTHONUNBUFFERED', 'TMPDIR')},
               'log': str(log), 'gpu_wait_s': waited, 'started': datetime.now().astimezone().isoformat(),
               'gpu_before': WC.gpu_snapshot(), 'disk_free_gb_before': free, **WC.git_state()}
    print(json.dumps({k: attempt[k] for k in ('demo', 'profile', 'live_task_id', 'log', 'started')}), flush=True)
    WC.write_new(out/'launch.json', {**attempt, 'status': 'LAUNCHING',
                                     'note': 'written before the Native process starts: from here on a missing result is a crash'})
    start = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        attempt['pid'] = proc.pid
        (out/'attempt_running.json').write_text(json.dumps(WC.jsonable(attempt), indent=1))
        returncode = proc.wait()
    result = json.loads((run_dir/'result.json').read_text()) if (run_dir/'result.json').exists() else None
    attempt.update(returncode=returncode, wall_s=time.monotonic()-start, ended=datetime.now().astimezone().isoformat(),
                   gpu_after=WC.gpu_snapshot(), run_dir_bytes=WC.du_bytes(run_dir),
                   disk_free_gb_after=shutil.disk_usage(out).free/1e9, result_status=(result or {}).get('status'),
                   note='A zero return code is not task success; see the analysis.')
    WC.write_new(out/'attempt.json', attempt)
    (out/'attempt_running.json').unlink()
    print(json.dumps({k: attempt[k] for k in ('demo', 'profile', 'returncode', 'wall_s', 'result_status')}), flush=True)
    return 0


def cmd_launch(args):
    return launch(load_demos(), args.demo, args.profile, args.root, args.disk_floor_gb, args.gpu_floor_mib)


def cmd_queue(args):
    demos = load_demos()
    order = demos['common']['surface_profiles']['order']
    for demo_id in args.demos:
        for profile in order:
            root = Path(args.root)
            root = root if root.is_absolute() else ROOT/root
            if (root/f'{demo_id}_{PROFILES[profile]}').exists():
                print(json.dumps({'skip_existing': f'{demo_id}_{PROFILES[profile]}'}), flush=True)
                continue
            try:
                launch(demos, demo_id, profile, root, args.disk_floor_gb, args.gpu_floor_mib)
            except Exception as error:   # keep going: every run is kept and reported
                print(json.dumps({'demo': demo_id, 'profile': profile, 'launcher_error': repr(error)}), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('check')
    s.set_defaults(func=cmd_check)
    for name, func in (('launch', cmd_launch), ('queue', cmd_queue)):
        s = sub.add_parser(name)
        if name == 'launch':
            s.add_argument('--demo', required=True)
            s.add_argument('--profile', required=True, choices=sorted(PROFILES))
        else:
            s.add_argument('--demos', nargs='+', required=True)
        s.add_argument('--root', required=True)
        s.add_argument('--disk-floor-gb', type=float, default=15.)
        s.add_argument('--gpu-floor-mib', type=int, default=12000)
        s.set_defaults(func=func)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
