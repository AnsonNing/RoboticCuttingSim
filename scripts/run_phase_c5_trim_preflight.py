#!/usr/bin/env python3
"""WP-11 Phase C5 step 1: material preflight of a configured TRIM scene + goal (kinematic, no robot).

Runs the SAME compiled scene/goal task the live Native C5 episode runs
(``compile_task(scene, goal, planner_options)``) through the kinematic
ideal-tracking backend at the scene's declared resolution (formal: 1 mm
particles, 4 mm grid, law 8, K = 1), material dt 0.1 ms, seed 42, with the goal
planner's full waypoint plan (HOME -> REORIENT -> PRE_CUT -> SKIM -> WITHDRAW
-> RETURN_HOME; cutting armed only on the SKIM waypoint) and the task's own
action limits, then holds still until the goal's episode time limit. The
outcome is the goal evaluator (``env.evaluate()``: integrity / outcome /
quality layers). Diagnostics: per-phase edge positions, speeds, the lowest
blade point vs the bar top, the ADR 0013 tau/psi/rho/mu during SKIM, material
reactions and code-0 stencil bridges (``analyze_cut_bridges``).

Kinematic ideal tracking says nothing about reachability, grasp, slip or
cameras. Oracle material state is evaluation-only. Laws, thresholds and frozen
configs are never changed here.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_phase_c5_trim_preflight.py \
      --device cuda:1 --out artifacts/wp11/phaseC5/preflight_kin_law8_001 \
      [--planner-options '{"max_rotation_deg": 1.0}']
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402
from robotic_cutting_sim.skim_orientation import skim_orientation_report  # noqa: E402

SEED = 42


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_script('run_phase1_matrix')        # read-only reuse: jsonable, components, force summaries
bridges = _load_script('analyze_cut_bridges')     # read-only reuse: code-0 stencil bridges


def jsonable(value):
    return matrix.jsonable(value)


def lowest_blade_point_z(task, pose):
    """World z of the lowest blade cross-section point at an edge pose (blade section from the tool profile)."""
    R = Rotation.from_quat(np.asarray(pose[3:], float)).as_matrix()
    section = task.tool_profile.cross_section_xz()
    points = np.c_[section[:, 0], np.zeros(len(section)), section[:, 1]] @ R.T
    return float(pose[2] + points[:, 2].min())


def phase_summary(task, steps, bar_top_world_z, target_world_z):
    out = {}
    for phase in dict.fromkeys(s['phase'] for s in steps):
        rows = [s for s in steps if s['phase'] == phase]
        poses = np.asarray([r['edge_pose_world'] for r in rows], float)
        speeds = np.linalg.norm(np.diff(np.r_[poses[:1, :3], poses[:, :3]], axis=0), axis=1) / task.control_dt_s
        low = np.asarray([lowest_blade_point_z(task, p) for p in poses])
        out[phase] = dict(steps=len(rows), t_start_s=rows[0]['time_s'], t_end_s=rows[-1]['time_s'],
                          edge_start_world_m=poses[0, :3].tolist(), edge_end_world_m=poses[-1, :3].tolist(),
                          edge_speed_max_m_s=float(speeds.max()), edge_speed_mean_m_s=float(speeds.mean()),
                          cut_enabled_steps=int(sum(r['cut_enabled'] for r in rows)),
                          lowest_blade_point_above_bar_top_min_m=float(low.min() - bar_top_world_z),
                          edge_height_vs_target_plus_clearance_m=dict(
                              min=float(poses[:, 2].min() - target_world_z), max=float(poses[:, 2].max() - target_world_z)),
                          blade_force_max_N=float(max(np.linalg.norm(r['blade_force_world_N']) for r in rows)),
                          labelled_count_end=int(rows[-1]['labelled_count']))
    return out


def run(args, out):
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.scene_config import compile_task, make_configured_kinematic_env
    overrides = json.loads(args.planner_options) if args.planner_options else None
    task = compile_task(args.scene, args.goal, planner_options=overrides, seed=SEED,
                        **({} if args.spacing is None else {'particle_spacing_m': args.spacing}))
    options = task.controller['planner']
    limits = task.action_limits()
    env = make_configured_kinematic_env(task, seed=SEED, device=args.device, material_dt=args.material_dt,
                                        action_limits=limits, delta_frame='world', ik_failure='reject',
                                        history_stride=args.history_stride)
    observation, _ = env.reset(seed=SEED)
    plan, plan_report = goal_planner.plan_with_report(task.scene, task.goal, task.tool_profile, options)
    policy = WaypointPolicy(task, plan, limits, **goal_planner.policy_options(options))
    dt = task.control_dt_s
    bed = env.backend.material.bed
    steps = []
    limit = int(round(task.duration_s / dt)) if args.steps is None else args.steps
    hold = dict(mode='cartesian_edge_delta', target=np.zeros(6), frame='world', cut_enabled=False)
    with (out / 'actions.jsonl').open('w') as stream:
        for step in range(limit):
            if policy.done:
                action, phase = hold, 'HOLD_TO_TIME_LIMIT'
            else:
                action = policy(observation)
                phase = policy.phase
            observation, _, terminated, truncated, info = env.step(action)
            material = env.backend.last_diagnostics['material']
            stroke = env.backend.stroke_history[-1]
            blade = np.asarray(material['current_material_reaction_impulse_world'], float) / dt
            total = np.asarray(material['current_total_material_reaction_impulse_world'], float) / dt
            row = dict(step=step + 1, time_s=env.time_s, phase=phase, cut_enabled=bool(action['cut_enabled']),
                       cut_index=int(material['cut_index']), edge_pose_world=env.backend.edge.tolist(),
                       blade_force_world_N=blade[:3].tolist(), total_force_world_N=total[:3].tolist(),
                       contact_statistics=np.asarray(material['material_contact_statistics']).tolist(),
                       stroke=stroke['stroke'], travel_m=stroke['travel_m'], engaged=stroke['engaged'],
                       labelled_count=stroke['labelled_count'], ik_status=info['ik_status'],
                       workspace_clamped=bool((info.get('action_report') or {}).get('workspace_clamped', False)),
                       tracking=policy.last_tracking)
            stream.write(json.dumps(jsonable(row)) + '\n')
            steps.append(row)
            if terminated or truncated:
                break
    material = env.backend.material
    recording = {'time_s': np.asarray([s['time_s'] for s in env._history], float),
                 **{k: np.asarray([s[k] for s in env._history]) for k in ('positions', 'velocities', 'codes')},
                 'mass_kg': material.masses, 'reference_positions': material.world_points,
                 'reference_indices': material.lattice, 'reference_stencil_nodes': bed.nodes.numpy(),
                 'reference_stencil_weights': bed.weights.numpy(), 'reference_stencil_gradients': bed.gradients.numpy()}
    for key in ('code_encoding', 'max_cuts'):
        if hasattr(material, key):
            recording[key] = np.asarray(getattr(material, key))
    result = env.evaluate()
    np.savez_compressed(out / 'physics_state.npz', **recording)
    context = dict(plan=[dict(label=w.label, pose=list(w.pose), cut=w.cut, cut_index=w.cut_index, speed_m_s=w.speed_m_s,
                              hold_s=w.hold_s) for w in plan], plan_report=plan_report, planner_options=options,
                   policy_options=goal_planner.policy_options(options), action_limits=limits.to_dict(),
                   policy_done=policy.done, runtime_source=env.runtime_source,
                   law8_parameters=getattr(bed, 'law8_parameters', None), material_receipt=material.parameter_receipt)
    env.close()
    return task, steps, result, context


def analyze(task, steps, out):
    target_world_z = float(task.T_world_workpiece[2, 3] + task.target_origin_local_m[2] + task.edge_clearance_m)
    bar_top_world_z = float(task.T_world_workpiece[2, 3] + task.geometry.base_height_m)
    n_t = task.T_world_workpiece[:3, :3] @ np.asarray(task.target_normal_local, float)
    c = task.T_world_workpiece[:3, 0]
    skim = [s for s in steps if s['cut_enabled']]
    orientation = (skim_orientation_report([s['edge_pose_world'] for s in skim], n_t, c,
                                           source='commanded = measured (kinematic ideal tracking)')
                   if len(skim) >= 2 else None)
    metrics = dict(phases=phase_summary(task, steps, bar_top_world_z, target_world_z),
                   skim_orientation=orientation, target_edge_world_z_m=target_world_z, bar_top_world_z_m=bar_top_world_z,
                   armed_steps=len(skim), ik_rejections=sum(s['ik_status'] != 'ok' for s in steps),
                   workspace_clamped_steps=sum(s['workspace_clamped'] for s in steps),
                   skim_force=matrix.force_summary(steps, next((s['phase'] for s in skim), 'SKIM:dome')))
    original_make, original_configure = bridges.make_task, bridges.configure_task
    try:  # read-only reuse of the analyzer with the compiled task instead of a registered id
        bridges.make_task = lambda task_id: task
        bridges.configure_task = lambda t, scenario=None: t
        report = bridges.analyze(out / 'physics_state.npz', task.task_id)
        metrics['bridges'] = {k: report[k] for k in report if k != 'source'}
    except Exception as error:  # noqa: BLE001 - diagnostics only
        metrics['bridges'] = {'status': 'NOT_RUN', 'reason': repr(error)}
    finally:
        bridges.make_task, bridges.configure_task = original_make, original_configure
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', default='trim01_wp11_workcell')
    p.add_argument('--goal', default='trim_bump_rho10')
    p.add_argument('--planner-options', help='JSON goal-planner option overrides (recorded in the task id)')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spacing', type=float, help='Numerical particle-spacing override (non-formal)')
    p.add_argument('--material-dt', type=float, default=1e-4, help='Material dt (formal: 1e-4 s)')
    p.add_argument('--history-stride', type=int, default=5)
    p.add_argument('--steps', type=int, help='Validation only: cap control steps (non-formal)')
    a = p.parse_args()
    out = new_output(a.out)  # refuses to overwrite
    start = time.monotonic()
    write_json(out / 'manifest.json', manifest({**vars(a), 'out': str(a.out), 'seed': SEED}, 'wp11_phaseC5_preflight'))
    formal = a.spacing is None and a.material_dt == 1e-4 and a.steps is None
    git = {**matrix.git_state(), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    base = dict(command=sys.argv, **git,
                seed=SEED, device=a.device, material_dt_s=a.material_dt, formal_resolution=formal,
                runtime_source='kinematic_ideal_tracking', scene=a.scene, goal=a.goal,
                planner_option_overrides=json.loads(a.planner_options) if a.planner_options else None)
    try:
        task, steps, result, context = run(a, out)
        metrics = analyze(task, steps, out)
        verdict = result.get('verdict')
        if not formal and verdict == 'PASS':
            verdict = 'PASS_NONFORMAL'
        write_json(out / 'result.json', jsonable({
            **base, 'task_id': task.task_id, 'config_sha256': task.config_sha256, 'cutting_law': task.cutting_law,
            'solver': task.solver_module, 'particle_spacing_m': task.particle_spacing_m, 'grid_dx_m': task.grid_dx_m,
            'verdict': verdict, 'goal_result': result, 'metrics': metrics, 'context': context, 'steps_run': len(steps),
            'wall_s': time.monotonic() - start}))
        items = (result.get('outcome') or {}).get('metrics', {}).get('items')
        print(task.task_id, verdict, json.dumps(jsonable(items))[:400], f'{time.monotonic() - start:.0f}s', flush=True)
    except Exception as error:
        write_json(out / 'result.json', jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                  'wall_s': time.monotonic() - start}))
        raise


if __name__ == '__main__':
    main()
