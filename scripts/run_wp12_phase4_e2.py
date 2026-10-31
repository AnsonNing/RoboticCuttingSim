#!/usr/bin/env python3
"""WP-12 Phase 4 E2: envelope confirmation through the REAL planner pipeline (configs/wp12/phase4_e2_protocol_v1.json).

Each declared row is a scene + goal built by the Phase 5 generators (``generalization_wp12`` builders, the
protocol's own frozen ``common`` block, nominal values overridden by the row's ``values``), compiled by
``scene_config``, planned by ``goal_planner`` with ``controller_profile path_v2``, tracked by ``WaypointPolicy`` on the
kinematic ideal-tracking backend (law 8, tlmpm_v5, 1 mm / 4 mm / 0.1 ms, seed 42) and scored by ``goal_eval_v2a``.
The episode is ``scripts/run_wp12_phase5.py kinematic_episode`` unchanged; ``queue`` launches each row through the
read-only ``cut_diagnostics_v1`` wrapper ``scripts/run_wp12_phase4a_diagnose.py`` (no law parameter overridden).
Nothing here changes physics, a law, a threshold or the planner. Outputs are never overwritten.

Subcommands:

* ``list``: row ids and what they test.
* ``feasibility --out DIR``: CPU (.venv). Per row: generator geometry, envelope v1, compile, staged C2 motion
  preflight (``generalization_wp12.classify_feasibility``); records every preflight slack (the knife-handle slack of
  the slotted rows) and the plan (armed segments, planned edge end height). ``DIR/<row>.json`` + ``summary.json``.
* ``run --row ID --device cuda:N --out DIR [--dev] [--steps N]``: one episode. Refuses ids not declared in the
  protocol unless ``--dev``; ``--dev`` or ``--steps`` label the output DEV (never evidence).
* ``queue --out-root DIR --feasibility DIR``: the declared rows in declared order, at most 2 concurrent jobs, one
  per lane, GNU time per job in ``artifacts/wp12/logs/<out name>_<row>.log``; a row whose directory exists is skipped
  and kept; a lane with less than ``--min-free-mib`` free waits; no row starts once the GPU-hour cap would be
  exceeded (those rows are NOT_RUN, budget). Other users' processes are never touched.
* ``summarize --out-root DIR --feasibility DIR --evidence FILE``: read-only; applies the protocol's verdict rules.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4_e2.py feasibility \\
      --out artifacts/wp12/phase4_e2_001/feasibility
  TMPDIR=/mnt/data/ningan/tmp nohup .venv/bin/python scripts/run_wp12_phase4_e2.py queue \\
      --out-root artifacts/wp12/phase4_e2_001 --feasibility artifacts/wp12/phase4_e2_001/feasibility \\
      >> artifacts/wp12/logs/phase4_e2_001_queue.log 2>&1 &
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from robotic_cutting_sim import generalization_wp12 as W  # noqa: E402

PROTOCOL = ROOT/'configs/wp12/phase4_e2_protocol_v1.json'
SOURCES = ('src/robotic_cutting_sim/tlmpm_v5.py', 'src/robotic_cutting_sim/goal_planner.py',
           'src/robotic_cutting_sim/policies.py', 'src/robotic_cutting_sim/goals.py',
           'src/robotic_cutting_sim/generalization_wp12.py', 'src/robotic_cutting_sim/cut_diagnostics.py',
           'scripts/run_wp12_phase5.py', 'scripts/run_wp12_phase4a_diagnose.py', 'scripts/run_wp12_phase4_e2.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {'git_sha': run('rev-parse', 'HEAD'), 'git_status_porcelain': run('status', '--porcelain').splitlines()}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=1))


def load_protocol(path=PROTOCOL):
    doc = json.loads(Path(path).read_text())
    ids = [r['id'] for r in doc['rows']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate E2 row ids')
    return doc


def pseudo_config(protocol):
    """The minimal benchmark-like config the Phase 5 builders and kinematic_episode read (common block only)."""
    return {'benchmark_id': protocol['protocol_id'], 'status': 'E2_PROTOCOL', 'common': deepcopy(protocol['common']),
            'families': {}, 'sample_prefix': 'E2'}


# ----------------------------------------------------------------------------- row -> sample

def _cut_fraction_for_plane_x0(values, tilt_deg):
    """cut_fraction that puts a tilted plane through local x = plane_x_at_board_top_m at z = 0 (the E1 convention;
    the Phase 5 builder puts the plane through (cut_fraction * L, 0, H/2))."""
    common_round = 0.001   # heights are whole millimetres (dimension_rule)
    H = round(values['height_m']/common_round)*common_round
    L = values['length_m']
    return (float(values['plane_x_at_board_top_m'])+H/2*math.tan(math.radians(tilt_deg)))/L


def _cut_push_trim(config, values, sample_id):
    """Demo 6 candidate (decision D5): sever at node-snapped x1, kerf_face push of the +x piece by d along +x, then
    remove the protrusion p0 that sits on the STATIONARY -x piece. One episode, flat board. The trim needs no planner
    offset, so the static motion preflight sees the true geometry of every step."""
    common = config['common']
    L, Wd, H = W._dims(common, values)
    d = float(values['push_distance_m'])
    x1 = W._node_x(common, float(values['cut_fraction'])*L)
    protrusions = W._protrusions(values, [float(values['protrusion_x_fraction'])*L])
    p0 = protrusions[0]
    if p0['x_m']+p0['radius_x_m'] >= x1-2*common['solver']['grid_dx_m']:
        raise ValueError('the protrusion must lie on the stationary (-x) piece, clear of the cut')
    plane = W._plane(x1, H)
    scene, dims, pose = W._common_scene(config, values, sample_id, protrusions, [x1], 'WP-12 E2 cut -> push -> trim')
    fields, fractions = W._sever_fields(common, [plane], L)
    push = {'type': 'move_piece', 'goal_id': 'push', 'direction_local': [1., 0., 0.], 'distance_m': d,
            'selector': {'side_of_plane': {'origin_local_m': [x1, 0., 0.], 'normal_local': [1., 0., 0.], 'side': 'positive'}}}
    goal = W._goal(common, f'{sample_id}_cut_push_trim', 'sequence',
                   f'Sever at x = {x1*1e3:.1f} mm, push the +x piece {d*1e3:.1f} mm along +x (cut disabled), then remove '
                   f'p0 at x = {p0["x_m"]*1e3:.1f} mm on the stationary piece (one surface_trim pass).',
                   {'steps': [{'type': 'sever', 'goal_id': 'sever', **fields}, push, W._remove_step(['p0'], 'trim_p0')]},
                   {**W._cut_planner(values), **W._skim_planner(values)})
    params = {**W._base_params(values, dims, protrusions), **W._sever_params([plane], values), **W._skim_params(values),
              'cut_digits': 2, 'operation_pattern': 'sequence:sever>move_piece>remove_protrusions'}
    derived = {'cut_x_local_m': x1, 'push_distance_m': d, 'protrusion_x_local_m': p0['x_m'], 'nominal_piece_fractions': fractions}
    return scene, goal, derived, params, {'planes': [plane], 'protrusions': protrusions}


CUSTOM_BUILDERS = {'cut_push_trim': _cut_push_trim}


def build_row_sample(protocol, row):
    config = pseudo_config(protocol)
    values = deepcopy(config['common']['nominal'])
    values.update(deepcopy(row.get('values', {})))
    tilt = values.get('cut_tilt_solid_deg') if row['variant'] == 'tilt_solid' else values.get('cut_tilt_deg')
    if 'plane_x_at_board_top_m' in values:
        values['cut_fraction'] = _cut_fraction_for_plane_x0(values, float(tilt or 0.))
    sample_id = f"E2_{row['id']}"
    if row['builder'] in CUSTOM_BUILDERS:
        scene, goal, derived, params, geometry = CUSTOM_BUILDERS[row['builder']](config, values, sample_id)
    else:
        builder, variants = W.BUILDERS[row['builder']]
        if row['variant'] not in variants:
            raise ValueError(f"{row['id']}: variant {row['variant']} not in {variants}")
        scene, goal, derived, params, geometry = builder(config, row['phase5_family'], values, row['variant'], sample_id)
    draws = [{'name': k, 'group': 'e2_row', 'range_set': 'E2', 'spec': {'fixed': v}, 'value': v, 'applied': True}
             for k, v in sorted(row.get('values', {}).items())]
    if 'plane_x_at_board_top_m' in values:
        draws.append({'name': 'cut_fraction', 'group': 'e2_derived', 'range_set': 'E2', 'spec': {'fixed': values['cut_fraction']},
                      'value': values['cut_fraction'], 'applied': True})
    return {'sample_id': sample_id, 'family': row['phase5_family'], 'range_set': 'E2', 'split': 'e2', 'stratum': 'E2',
            'seed': int(protocol['common']['material_seed']), 'index': 0, 'variant': row['variant'], 'declared': True,
            'scene': scene, 'goal': goal, 'draws': draws, 'ood_dimensions': [], 'derived': derived,
            'envelope_parameters': params, 'geometry': geometry, 'row': row}


# ----------------------------------------------------------------------------- plan facts (CPU)

def plan_facts(task):
    """Armed segments of the compiled plan (planned end of each armed waypoint, lowest edge point vs the board top)."""
    from robotic_cutting_sim import goal_planner as gp
    from robotic_cutting_sim.scene_frames import pose_matrix
    from scipy.spatial.transform import Rotation
    options = gp.planner_options(task.goal, None)
    waypoints, report = gp.plan_with_report(task.scene, task.goal, task.tool_profile, options)
    T = np.asarray(pose_matrix(task.world_from_workpiece_xyzw), float)
    out = []
    half = float(task.tool_profile.half_width_m)
    for w in waypoints:
        if not getattr(w, 'cut', False):
            continue
        pose = np.asarray(w.pose, float)
        R_world = Rotation.from_quat(pose[3:]).as_matrix()
        p_local = np.linalg.solve(T, np.r_[pose[:3], 1.])[:3]
        R_local = T[:3, :3].T@R_world
        e = R_local[:, 1]
        out.append({'label': w.label, 'cut_index': getattr(w, 'cut_index', None), 'speed_m_s': getattr(w, 'speed_m_s', None),
                    'planned_edge_centre_local_m': p_local.tolist(),
                    'planned_lowest_edge_point_local_z_m': float(p_local[2]-half*abs(e[2]))})
    return {'armed_waypoints': out, 'controller_profile': options.get('controller_profile'),
            'position_tolerance_rule': 'WaypointPolicy completes a waypoint within the controller position tolerance '
                                       '(path_v2 0.3 mm): the armed end lies 0-0.3 mm short of the planned end',
            'operations': [{k: v for k, v in r.items() if k in ('op', 'kind', 'digit', 'end_local_m', 'end_on_support',
                                                                 'cut_extent', 'planner_offset_local_m')}
                           for r in report.get('operations', [])] if isinstance(report, dict) else None}


def cmd_list(args):
    protocol = load_protocol(args.protocol)
    for row in protocol['rows']:
        print(f"{row['id']:16s} {row['phase5_family']} {row['builder']}/{row['variant']:22s} {row['tests']}")
    return 0


def cmd_feasibility(args):
    protocol = load_protocol(args.protocol)
    config = pseudo_config(protocol)
    envelope = W.load_envelope(ROOT/protocol['envelope'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [r for r in protocol['rows'] if not args.rows or r['id'] in args.rows]
    summary = []
    for row in rows:
        start = time.perf_counter()
        try:
            sample = build_row_sample(protocol, row)
            res = W.classify_feasibility(sample, envelope, compile_fn=lambda s: W.compile_sample(config, s),
                                         preflight_fn=lambda task: W.preflight(task, config['common']['preflight_geometry']),
                                         run_preflight_when_unsupported=True, common=config['common'])
            try:
                task, _ = W.compile_sample(config, sample)
                res['plan_facts'] = plan_facts(task)
            except Exception as error:
                res['plan_facts'] = {'error': repr(error)}
            res['sample'] = {k: sample[k] for k in ('scene', 'goal', 'draws', 'derived', 'envelope_parameters', 'row')}
        except Exception as error:
            res = {'sample_id': f"E2_{row['id']}", 'feasibility_class': None, 'classification_error': True,
                   'reason': f'feasibility error: {error!r}', 'traceback': traceback.format_exc()}
        res['wall_s'] = time.perf_counter()-start
        write_new(out/f"{row['id']}.json", res)
        receipt = ((res.get('checks') or {}).get('preflight') or {}).get('receipt') or {}
        worst = (receipt.get('path') or {}).get('worst') or {}
        summary.append({'row': row['id'], 'feasibility_class': res.get('feasibility_class'), 'reason': res.get('reason'),
                        'envelope': ((res.get('checks') or {}).get('envelope') or {}).get('status'),
                        'task_id': (res.get('compile') or {}).get('task_id'),
                        'duration_s': (res.get('compile') or {}).get('duration_s'),
                        'preflight_slack_mm': {k: round(v['slack']*1e3, 3) for k, v in worst.items() if 'slack' in v},
                        'planned_lowest_edge_z_mm': [round(a['planned_lowest_edge_point_local_z_m']*1e3, 3)
                                                     for a in (res.get('plan_facts') or {}).get('armed_waypoints', [])],
                        'wall_s': res['wall_s']})
        print(f"[E2F] {row['id']}: {res.get('feasibility_class')} | {str(res.get('reason'))[:140]} | "
              f"{res['wall_s']:.1f} s", flush=True)
    write_new(out/'summary.json', {'protocol': str(Path(args.protocol).relative_to(ROOT)), 'protocol_sha256': sha(args.protocol),
                                   'runner_sha256': sha(__file__), **git_state(), 'command': sys.argv,
                                   'finished': datetime.now().astimezone().isoformat(), 'rows': summary})
    return 0


# ----------------------------------------------------------------------------- run

def cmd_run(args):
    import run_wp12_phase5 as P5
    protocol = load_protocol(args.protocol)
    rows = {r['id']: r for r in protocol['rows']}
    if args.row not in rows:
        if not args.dev:
            raise SystemExit(f'{args.row} is not a declared E2 row (use --dev for development rows)')
        row = json.loads(args.dev_row_json) if args.dev_row_json else None
        if row is None:
            raise SystemExit('--dev with an undeclared id needs --dev-row-json')
    else:
        row = rows[args.row]
    config = pseudo_config(protocol)
    sample = build_row_sample(protocol, row)
    label = 'DEV' if (args.dev or args.steps is not None) else 'E2_FORMAL'
    extra = {'label': label, 'e2_row': row, 'e2_protocol': str(Path(args.protocol).relative_to(ROOT)),
             'e2_protocol_sha256': sha(args.protocol), 'e2_runner_sha256': sha(__file__),
             'e2_source_sha256': {s: sha(ROOT/s) for s in SOURCES if (ROOT/s).exists()}, **git_state(),
             'e2_command': sys.argv, 'e2_started': datetime.now().astimezone().isoformat()}
    ns = SimpleNamespace(out=args.out, device=args.device, steps=args.steps, expect_task_id=args.expect_task_id,
                         feasibility_class=args.feasibility_class, surface_profile=None, geometry_level=None,
                         no_geometry=False)
    return P5.kinematic_episode(config, sample, ns, extra=extra)


# ----------------------------------------------------------------------------- queue

BOARD_BODIES = ('flat_board', 'board_segment', 'cutting_board')


def board_collision(feas):
    """Failing preflight checks whose worst sample overlaps a board body (negative slack): the kinematic blade does not
    see the rigid board, so an episode would cut with the blade inside the board (protocol verdict_rules)."""
    receipt = (((feas or {}).get('checks') or {}).get('preflight') or {}).get('receipt') or {}
    path = receipt.get('path') or {}
    out = []
    for name in path.get('failing', []):
        worst = (path.get('worst') or {}).get(name) or {}
        if worst.get('slack', 0.) < 0 and any(b in str(p) for p in worst.get('pair') or () for b in BOARD_BODIES):
            out.append({'check': name, 'slack_mm': worst['slack']*1e3, 'label': worst.get('label'), 'pair': worst.get('pair')})
    return out


def free_mib(device):
    index = device.split(':')[1]
    out = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else 0


def cmd_queue(args):
    protocol = load_protocol(args.protocol)
    budget = protocol['budget']
    cap_h = float(budget['gpu_hours_cap'])
    if args.max_jobs > int(budget['max_concurrent_jobs']):
        raise SystemExit(f"the protocol allows at most {budget['max_concurrent_jobs']} concurrent E2 jobs")
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    feas = Path(args.feasibility)
    logs = ROOT/'artifacts/wp12/logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'TMPDIR': os.environ.get('TMPDIR', '/mnt/data/ningan/tmp')}
    order = [r for r in protocol['rows'] if not args.rows or r['id'] in args.rows]
    qlog = open(root/'queue.jsonl', 'a')

    def record(**kw):
        qlog.write(json.dumps(jsonable({'time': datetime.now().astimezone().isoformat(), **kw}))+'\n')
        qlog.flush()

    pending = []
    for row in order:
        f = feas/f"{row['id']}.json"
        fr = json.loads(f.read_text()) if f.exists() else None
        cls = (fr or {}).get('feasibility_class')
        if (root/row['id']).exists():
            record(row=row['id'], event='skipped_existing')
        elif cls in (None, 'scene_geometrically_infeasible'):   # protocol verdict_rules.infeasible_rows
            record(row=row['id'], event='not_run_infeasible', feasibility_class=cls)
        elif board_collision(fr):   # protocol verdict_rules.NOT_EXECUTABLE_BOARD_COLLISION
            record(row=row['id'], event='not_run_board_collision', collisions=board_collision(fr))
        else:
            pending.append((row, cls))
    lanes = {d: None for d in args.devices}
    used_s = 0.
    while pending or any(lanes.values()):
        for dev, job in list(lanes.items()):
            if job is not None and job['proc'].poll() is not None:
                wall = time.monotonic()-job['start']
                used_s += wall
                record(row=job['row'], event='finished', device=dev, exit=job['proc'].returncode, wall_s=wall,
                       gpu_hours_used=used_s/3600)
                lanes[dev] = None
        running_s = sum(time.monotonic()-j['start'] for j in lanes.values() if j)
        for dev in lanes:
            if lanes[dev] is not None or not pending:
                continue
            if sum(1 for j in lanes.values() if j) >= args.max_jobs:
                break
            row, cls = pending[0]
            estimate_s = float(row.get('estimate_gpu_s', budget['default_row_estimate_gpu_s']))
            if (used_s+running_s+estimate_s)/3600 > cap_h:
                for r, _ in pending:
                    record(row=r['id'], event='not_run_budget', gpu_hours_used=used_s/3600,
                           gpu_hours_running=running_s/3600, estimate_gpu_s=r.get('estimate_gpu_s'))
                pending = []
                break
            if free_mib(dev) < args.min_free_mib:
                continue
            pending.pop(0)
            out = root/row['id']
            log = logs/f"{root.name}_{row['id']}.log"
            cmd = ['/usr/bin/time', '-v', sys.executable, str(ROOT/'scripts/run_wp12_phase4a_diagnose.py'), '--',
                   'scripts/run_wp12_phase4_e2.py', '--protocol', str(args.protocol), 'run', '--row', row['id'],
                   '--device', dev, '--out', str(out), '--feasibility-class', str(cls)]
            task_id = None
            f = feas/f"{row['id']}.json"
            if f.exists():
                task_id = (json.loads(f.read_text()).get('compile') or {}).get('task_id')
            if task_id:
                cmd += ['--expect-task-id', task_id]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(log, 'a'), stderr=subprocess.STDOUT)
            lanes[dev] = {'row': row['id'], 'proc': proc, 'start': time.monotonic()}
            record(row=row['id'], event='started', device=dev, pid=proc.pid, command=cmd, log=str(log))
            running_s = sum(time.monotonic()-j['start'] for j in lanes.values() if j)
        time.sleep(args.poll_s)
    record(event='queue_done', gpu_hours_used=used_s/3600)
    return 0


# ----------------------------------------------------------------------------- summary (read-only)

def _armed_ends(diag):
    """Per armed segment (consecutive control steps with cut index >= 0): the lowest edge point height above the
    workpiece-local board top (z = 0) at the segment's last armed step, from the solver-local end poses."""
    from scipy.spatial.transform import Rotation
    cut = np.asarray(diag['step_cut'])
    poses = np.asarray(diag['step_pose_end_local'], float)
    half = float(diag['blade_half_width_m'])
    out, i = [], 0
    while i < len(cut):
        if cut[i] < 0:
            i += 1
            continue
        j = i
        while j+1 < len(cut) and cut[j+1] == cut[i]:
            j += 1
        p = poses[j]
        e = Rotation.from_quat(p[3:]).as_matrix()[:, 1]
        out.append({'cut_index': int(cut[i]), 'first_step': int(i+1), 'last_step': int(j+1), 'armed_steps': int(j-i+1),
                    'edge_centre_local_z_mm': float(p[2]*1e3),
                    'lowest_edge_point_local_z_mm': float((p[2]-half*abs(e[2]))*1e3)})
        i = j+1
    return out


def analyze_row(out_dir):
    """Row metrics from result.json (kinematic_episode) and wp12_phase4a_diag.npz (cut_diagnostics wrapper)."""
    from robotic_cutting_sim import cut_diagnostics as cd
    from robotic_cutting_sim.cut_topology import mechanical_components, mechanical_components_packed
    out_dir = Path(out_dir)
    res_path = out_dir/'result.json'
    if not res_path.exists():
        return {'status': 'NO_RESULT' if out_dir.exists() else 'NOT_RUN'}
    r = json.loads(res_path.read_text())
    ev = r.get('evaluation') or {}
    row = {'status': r.get('status'), 'label': r.get('label'), 'partial': r.get('partial'), 'policy_done': r.get('policy_done'),
           'solver_invalid': r.get('solver_invalid'), 'class': r.get('class'), 'class_reason': r.get('class_reason'),
           'verdict': ev.get('verdict'), 'integrity': (ev.get('integrity') or {}).get('status'),
           'step_verdicts': [s.get('verdict') for s in ((ev.get('outcome') or {}).get('steps') or [])],
           'metrics': r.get('metrics'), 'physical_quality': r.get('physical_quality'), 'runtime': r.get('runtime'),
           'stability': r.get('stability'), 'task_id': r.get('task_id'), 'device': r.get('device'),
           'e2_protocol_sha256': r.get('e2_protocol_sha256'), 'git_sha': r.get('git_sha'),
           'steps_run': r.get('steps_run'), 'control_steps': r.get('control_steps')}
    diag_npz = out_dir/'wp12_phase4a_diag.npz'
    meta_json = out_dir/'wp12_phase4a_diag.json'
    if diag_npz.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text())
        m = meta.get('meta') or {}
        with np.load(diag_npz) as z:
            codes = z['final_codes']
            nodes, weights, grads = z['stencil_nodes'], z['stencil_weights'], z['stencil_gradients']
            encoding, K = m.get('code_encoding', 'ternary'), int(m.get('max_cuts', 1))
            if encoding == 'packed2':
                count, _ = mechanical_components_packed(nodes, weights, codes, K, gradients=grads)
            else:
                count, _ = mechanical_components(nodes, weights, codes, gradients=grads)
            bridges = {}
            for k in range(K):
                digits = cd.digits_of(codes, k, encoding)
                if np.any(digits > 0):
                    bridges[str(k)] = int(len(cd.direct_bridges(nodes, weights, grads, digits)))
            diag = {'step_cut': z['step_cut'], 'step_pose_end_local': z['step_pose_end_local'],
                    'blade_half_width_m': m.get('blade_half_width_m', .06)}
            row.update(mechanical_components=int(count), direct_code0_bridges_by_digit=bridges,
                       direct_code0_bridges_total=int(sum(bridges.values())), armed_ends=_armed_ends(diag),
                       min_particle_local_z_mm=float(z['final_positions'][:, 2].min()*1e3)
                       if 'final_positions' in z.files else None,
                       diag_wrapped_exit=meta.get('wrapped_exit'))
    else:
        row['diagnostics'] = 'MISSING (cut_diagnostics wrapper output not found)'
    return row


def verdict(row_decl, feas, run):
    """Protocol verdict of one row (see the protocol's verdict_rules)."""
    pre = ((feas or {}).get('checks') or {}).get('preflight') or {}
    reachable = pre.get('status') == 'PASS'
    if feas is not None and board_collision(feas):
        return 'NOT_EXECUTABLE_BOARD_COLLISION', {'collisions': board_collision(feas)}
    if run.get('status') in ('NOT_RUN', None):
        return 'NOT_RUN', 'not executed'
    if run.get('status') == 'NO_RESULT':
        return 'ERROR', 'output directory without result.json (kept)'
    if run.get('label') != 'E2_FORMAL' or run.get('partial'):
        return 'NOT_RUN', 'DEV or partial output is never evidence'
    if run.get('status') != 'COMPLETE':
        return 'ERROR', f"episode status {run.get('status')}"
    kind = row_decl['kind']
    checks = {'policy_done': bool(run.get('policy_done')), 'solver_valid': run.get('solver_invalid') is None,
              'goal_eval_v2a_PASS': run.get('verdict') == 'PASS'}
    if kind in ('sever', 'sequence_with_sever'):
        checks['components_ge_expected'] = (run.get('mechanical_components') or 0) >= int(row_decl['expected_pieces'])
        by_digit = run.get('direct_code0_bridges_by_digit') or {}
        checks['zero_direct_bridges'] = run.get('direct_code0_bridges_by_digit') is not None and \
            all(by_digit.get(str(k), 0) == 0 for k in row_decl.get('bridge_digits', [0]))
    outcome_ok = all(checks.values())
    if outcome_ok and reachable:
        return 'CONFIRMED', checks
    if outcome_ok:
        return 'PASS_KINEMATIC_PREFLIGHT_FAIL', checks
    return 'FAIL', checks


def cmd_summarize(args):
    protocol = load_protocol(args.protocol)
    root, feas = Path(args.out_root), Path(args.feasibility)
    rows = []
    for decl in protocol['rows']:
        f = feas/f"{decl['id']}.json"
        fr = json.loads(f.read_text()) if f.exists() else None
        run = analyze_row(root/decl['id'])
        v, why = verdict(decl, fr, run)
        pred = decl.get('prediction', {})
        expected = pred.get('verdict')
        rows.append({'id': decl['id'], 'tests': decl['tests'], 'verdict': v, 'checks': why, 'prediction': pred,
                     'prediction_met': None if expected in (None, 'NO_DIRECTIONAL_PREDICTION') or v in ('NOT_RUN', 'ERROR')
                     else (v == expected),
                     'feasibility_class': (fr or {}).get('feasibility_class'),
                     'preflight_status': (((fr or {}).get('checks') or {}).get('preflight') or {}).get('status'),
                     'preflight_worst_slack_mm': {k: round(x['slack']*1e3, 3) for k, x in
                                                  ((((((fr or {}).get('checks') or {}).get('preflight') or {}).get('receipt')
                                                     or {}).get('path') or {}).get('worst') or {}).items() if 'slack' in x},
                     'plan_facts': (fr or {}).get('plan_facts'), 'run': run})
    counts = {}
    for r in rows:
        counts[r['verdict']] = counts.get(r['verdict'], 0)+1
    evidence = {'schema': 'wp12_phase4_e2_evidence_v1', 'protocol': str(Path(args.protocol).relative_to(ROOT)),
                'protocol_sha256': sha(args.protocol), 'summarizer_sha256': sha(__file__), **git_state(),
                'command': sys.argv, 'out_root': str(root), 'feasibility': str(feas), 'verdict_counts': counts,
                'scope': protocol['scope'], 'rows': rows}
    q = root/'queue.jsonl'
    if q.exists():
        evidence['queue'] = [json.loads(line) for line in q.read_text().splitlines() if line.strip()]
    write_new(args.evidence, evidence)
    print(json.dumps(counts))
    for r in rows:
        run = r['run']
        print(f"{r['id']:16s} {r['verdict']:30s} pred {str(r['prediction'].get('verdict')):26s} met {r['prediction_met']} "
              f"| parts {run.get('mechanical_components')} bridges {run.get('direct_code0_bridges_total')} "
              f"verdict {run.get('verdict')} armed_end_mm "
              f"{[round(a['lowest_edge_point_local_z_mm'], 3) for a in run.get('armed_ends', [])]}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    s = sub.add_parser('feasibility')
    s.add_argument('--out', required=True)
    s.add_argument('--rows', nargs='*')
    s = sub.add_parser('run')
    s.add_argument('--row', required=True)
    s.add_argument('--device', default='cuda:1')
    s.add_argument('--out', required=True)
    s.add_argument('--steps', type=int, help='DEV smoke: stop after N control steps (partial, never evidence)')
    s.add_argument('--dev', action='store_true', help='DEV label (never evidence); allows --dev-row-json')
    s.add_argument('--dev-row-json')
    s.add_argument('--expect-task-id')
    s.add_argument('--feasibility-class')
    s = sub.add_parser('queue')
    s.add_argument('--out-root', required=True)
    s.add_argument('--feasibility', required=True)
    s.add_argument('--devices', nargs='+', default=['cuda:1', 'cuda:0'])
    s.add_argument('--rows', nargs='*')
    s.add_argument('--max-jobs', type=int, default=2)
    s.add_argument('--min-free-mib', type=int, default=4000)
    s.add_argument('--poll-s', type=float, default=30.)
    s = sub.add_parser('summarize')
    s.add_argument('--out-root', required=True)
    s.add_argument('--feasibility', required=True)
    s.add_argument('--evidence', required=True)
    a = p.parse_args(argv)
    a.protocol = Path(a.protocol).resolve()
    return {'list': cmd_list, 'feasibility': cmd_feasibility, 'run': cmd_run, 'queue': cmd_queue,
            'summarize': cmd_summarize}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
