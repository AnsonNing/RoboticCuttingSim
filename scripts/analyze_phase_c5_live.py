#!/usr/bin/env python3
"""WP-11 Phase C5: read-only analysis of a live Native configured TRIM episode.

Reads ``<run>/episode_000`` written by ``scripts/run_wp10_live.py --scene/--goal``
and writes ``<out>/analysis.json`` (refuses to overwrite; the recording is never
modified). Reports, separately:

* the goal verdict and per-protrusion removal (copied from the live goal
  evaluator) and ``home_outcome``;
* per-phase MEASURED knife-edge positions/speeds, the lowest blade point vs the
  bar top, and the measured edge height vs target + edge clearance;
* ADR 0013 tau/psi/rho/mu of the measured (and commanded) edge on armed steps;
* the material reaction on the blade during SKIM;
* grasp slip/rotation vs the GraspMonitor warning/loss gates; native tool
  contacts (unintended contact count, non-finger roles, tensor-contact
  penetration);
* the C2 geometric clearance of the measured task-phase joint path (fingers,
  handle, wrist cameras, table, board, Franka 1; uncut workpiece; blade-workpiece
  mode per phase) in the CPU ``.venv``;
* the chip after withdrawal: mechanical components and label changes on the
  recorded frames from the end of the armed skim to the end of the episode,
  chip centroid motion, and code-0 stencil bridges (``analyze_cut_bridges``).

All of this is privileged/oracle evaluation data, never an observation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.recording import new_output, write_json  # noqa: E402
from robotic_cutting_sim.skim_orientation import skim_orientation_report  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_script('run_phase1_matrix')
bridges = _load_script('analyze_cut_bridges')


def stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return None
    return dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                p95=float(np.percentile(values, 95)), samples=int(len(values)))


def lowest_blade_z(task, pose):
    R = Rotation.from_quat(np.asarray(pose[3:], float)).as_matrix()
    section = task.tool_profile.cross_section_xz()
    points = np.c_[section[:, 0], np.zeros(len(section)), section[:, 1]] @ R.T
    return float(pose[2] + points[:, 2].min())


def contact_audit(rows):
    unintended, roles, separations, statuses = [], {}, [], {}
    for row in rows:
        grasp = (row.get('privileged_diagnostics') or {}).get('grasp') or {}
        contacts = grasp.get('contacts') or {}
        unintended.append(int(contacts.get('unintended_contact_count', 0)))
        for role, value in (contacts.get('roles') or {}).items():
            if role in ('left_finger', 'right_finger'):
                continue
            count = int(value.get('contact_count', 0))
            if count:
                roles.setdefault(role, dict(steps=0, max_contacts=0, phases=set()))
                roles[role]['steps'] += 1
                roles[role]['max_contacts'] = max(roles[role]['max_contacts'], count)
                roles[role]['phases'].add(row.get('policy_phase'))
        for record in (contacts.get('tensor_forces') or {}).get('records', []):
            if record.get('separation_m') is not None:
                separations.append((float(record['separation_m']), record.get('role'), row.get('policy_phase')))
        status = (row.get('grasp_status') or {}).get('status') if isinstance(row.get('grasp_status'), dict) else row.get('grasp_status')
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    worst = min(separations, default=None, key=lambda item: item[0])
    return dict(unintended_contact_count_max=max(unintended, default=0),
                steps_with_unintended_contacts=int(sum(u > 0 for u in unintended)),
                non_finger_roles_in_contact={k: {**v, 'phases': sorted(map(str, v['phases']))} for k, v in roles.items()},
                tensor_contact_min_separation_m=None if worst is None else worst[0],
                tensor_contact_max_penetration_m=None if worst is None else max(0., -worst[0]),
                tensor_contact_worst_role=None if worst is None else worst[1],
                tensor_contact_worst_phase=None if worst is None else worst[2],
                grasp_status_steps=statuses,
                scope='Native ContactSensor on the tool body (fingers, palm, table, fixture, other) and the tensor '
                      'contact view (finger/palm pairs); robot links vs workcell are not natively sensed here')


def grasp_audit(rows):
    from robotic_cutting_sim.grasp_monitor import GraspLossConfig
    gates = GraspLossConfig()
    slip, rotation, lost = [], [], False
    for row in rows:
        status = row.get('grasp_status')
        if isinstance(status, dict):
            slip.append(float(status.get('slip_since_settling_m', 0.)))
            rotation.append(float(status.get('rotation_since_settling_rad', 0.)))
            lost |= bool(status.get('grasp_lost', False))
    return dict(max_slip_m=max(slip, default=None), max_rotation_rad=max(rotation, default=None),
                final_slip_m=slip[-1] if slip else None, final_rotation_rad=rotation[-1] if rotation else None,
                grasp_lost=lost, gates=dict(translation_warning_m=gates.translation_warning_m,
                                            rotation_warning_rad=gates.rotation_warning_rad,
                                            translation_loss_m=gates.translation_loss_m,
                                            rotation_loss_rad=gates.rotation_loss_rad),
                within_warning_gates=bool(slip and max(slip) <= gates.translation_warning_m
                                          and max(rotation) <= gates.rotation_warning_rad and not lost))


def phase_table(task, rows, target_z, bar_top_z):
    out = {}
    for phase in dict.fromkeys(r.get('policy_phase') for r in rows):
        sub = [r for r in rows if r.get('policy_phase') == phase]
        poses = np.asarray([r['measured']['tool_pose'] for r in sub], float)
        times = np.asarray([r['time_s'] for r in sub], float)
        speed = (np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / np.maximum(np.diff(times), 1e-9)
                 if len(sub) > 1 else np.zeros(1))
        low = [lowest_blade_z(task, p) - bar_top_z for p in poses]
        out[str(phase)] = dict(steps=len(sub), t_start_s=float(times[0]), t_end_s=float(times[-1]),
                               measured_edge_start_world_m=poses[0, :3].tolist(),
                               measured_edge_end_world_m=poses[-1, :3].tolist(),
                               measured_edge_speed_max_m_s=float(speed.max()),
                               cut_enabled_steps=int(sum(bool((r.get('submitted') or {}).get('cut_enabled')) for r in sub)),
                               lowest_blade_point_above_bar_top_m=stats(low),
                               measured_edge_z_minus_target_m=stats(poses[:, 2] - target_z),
                               tracking_error_m=stats([r.get('tracking_error_m') for r in sub]),
                               rotation_error_rad=stats([r.get('rotation_error_rad') for r in sub]))
    return out


def reaction(rows, dt):
    force = [np.asarray((r.get('privileged_diagnostics') or {}).get('current_material_reaction_impulse_world',
                                                                    [0.] * 6), float)[:3] / dt for r in rows]
    total = [np.asarray((r.get('privileged_diagnostics') or {}).get('current_total_material_reaction_impulse_world',
                                                                    [0.] * 6), float)[:3] / dt for r in rows]
    if not force:
        return None
    force, total = np.asarray(force), np.asarray(total)
    return dict(steps=len(force), blade_force_norm_N=stats(np.linalg.norm(force, axis=1)),
                blade_force_world_mean_N=force.mean(0).tolist(), blade_force_z_N=stats(force[:, 2]),
                total_force_norm_N=stats(np.linalg.norm(total, axis=1)),
                steps_with_blade_force=int(np.count_nonzero(np.linalg.norm(force, axis=1) > 0)))


def chip_audit(task, npz_path, rows, out):
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    data = dict(np.load(npz_path))
    times = np.asarray(data['time_s'], float)
    phase_times = np.asarray(data.get('policy_phase_time_s', []), float)
    armed = [r['time_s'] for r in rows if (r.get('submitted') or {}).get('cut_enabled')]
    armed_end = max(armed) if armed else None
    nodes, weights, gradients = (np.asarray(data[k]) for k in
                                 ('reference_stencil_nodes', 'reference_stencil_weights', 'reference_stencil_gradients'))
    masses = np.asarray(data['mass_kg'], float)
    reference_local = task.points_to_local(np.asarray(data['reference_positions'], float))
    above = reference_local[:, 2] > task.target_origin_local_m[2] + 1e-9
    codes = np.asarray(data['codes'])
    cache, frames = {}, []
    for index in range(len(times)):
        key = codes[index].tobytes()
        if key not in cache:
            cache[key] = mechanical_components_packed(nodes, weights, codes[index], task.max_cuts, gradients=gradients)
        count, labels = cache[key]
        substrate = int(np.bincount(labels[~above], weights=masses[~above]).argmax()) if (~above).any() else 0
        removed = above & (labels != substrate)
        local = task.points_to_local(np.asarray(data['positions'][index], float))
        chip_mass = float(masses[removed].sum())
        frames.append(dict(time_s=float(times[index]), components=int(count),
                           removed_above_plane_mass_fraction=float(chip_mass / masses[above].sum()),
                           chip_centroid_local_m=(masses[removed] @ local[removed] / chip_mass).tolist() if chip_mass else None,
                           labelled_particles=int(np.count_nonzero(codes[index]))))
    after = [f for f in frames if armed_end is not None and f['time_s'] >= armed_end - 1e-9]
    label_changes_after = (int(np.count_nonzero(codes[-1] != codes[[i for i, t in enumerate(times) if armed_end is not None
                                                                         and t >= armed_end - 1e-9][0]]))
                           if after else None)
    derived = out / 'physics_state_with_encoding.npz'
    np.savez_compressed(derived, **data, code_encoding=np.asarray('packed2'), max_cuts=np.asarray(task.max_cuts))
    original_make, original_configure = bridges.make_task, bridges.configure_task
    try:
        bridges.make_task = lambda task_id: task
        bridges.configure_task = lambda t, scenario=None: t
        report = bridges.analyze(derived, task.task_id)
        bridge = {k: report[k] for k in report if k != 'source'}
    except Exception as error:  # noqa: BLE001 - diagnostics only
        bridge = {'status': 'NOT_RUN', 'reason': repr(error)}
    finally:
        bridges.make_task, bridges.configure_task = original_make, original_configure
    derived.unlink()   # derived copy only for the analyzer (adds the packed-code keys); the recording is untouched
    first, last = (after[0], after[-1]) if after else (None, None)
    return dict(recorded_frames=len(frames), armed_end_time_s=armed_end,
                frames_after_armed_end=len(after),
                components_after_armed_end=sorted({f['components'] for f in after}),
                removed_fraction_after_armed_end=stats([f['removed_above_plane_mass_fraction'] for f in after]),
                label_changes_after_armed_end=label_changes_after,
                chip_centroid_at_armed_end_local_m=None if first is None else first['chip_centroid_local_m'],
                chip_centroid_final_local_m=None if last is None else last['chip_centroid_local_m'],
                chip_stays_independent=bool(after and all(f['components'] >= 2 and f['removed_above_plane_mass_fraction']
                                                          >= .9 for f in after) and label_changes_after == 0),
                bridges_final=bridge, timeline=frames[::max(1, len(frames) // 60)],
                scope='Recorded frames (every --record-every steps, incl. hold and return home); actual packed-code '
                      'stencil components; oracle-only')


def measured_path_clearance(task, rows, out):
    from robotic_cutting_sim.motion_preflight import phase_mode, run_cli
    from robotic_cutting_sim.scene_config import configured_task_payload
    task_rows = [r for r in rows if not str(r.get('policy_phase', '')).startswith('RETURN_HOME')]
    qs = [np.asarray(r['measured']['q'], float)[0] for r in task_rows]
    labels = [str(r.get('policy_phase')) for r in task_rows]
    modes = [phase_mode(label, bool((r.get('submitted') or {}).get('cut_enabled'))) for label, r in zip(labels, task_rows)]
    receipt = run_cli('check-path', configured_task_payload(task), q=np.asarray(qs).tolist(), modes=modes,
                      log_dir=out / 'measured_path')
    return dict(status=receipt.get('status'), failing=receipt.get('failing'), binding=receipt.get('binding'),
                worst=receipt.get('worst'), samples=receipt.get('samples'),
                blade_workpiece_modes=receipt.get('blade_workpiece_modes'), scope=receipt.get('scope'),
                note='Task phase only (the return-home path has its own check in home_outcome)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-path-check', action='store_true')
    a = p.parse_args()
    out = new_output(a.out)
    episode = a.run / 'episode_000'
    result = json.loads((episode / 'result.json').read_text())
    rows = [json.loads(line) for line in (episode / 'actions.jsonl').open()]
    from robotic_cutting_sim.scene_config import compile_task
    configured = result['configured_task']
    task = compile_task(configured['scene'], configured['goal'], planner_options=configured['planner_options'],
                        seed=result['seed'])
    if task.task_id != result['task_id']:
        raise ValueError(f'Recompiled task {task.task_id} differs from the recorded {result["task_id"]}')
    dt = task.control_dt_s
    target_z = float(task.T_world_workpiece[2, 3] + task.target_origin_local_m[2] + task.edge_clearance_m)
    bar_top_z = float(task.T_world_workpiece[2, 3] + task.geometry.base_height_m)
    n_t = task.T_world_workpiece[:3, :3] @ np.asarray(task.target_normal_local, float)
    c = task.T_world_workpiece[:3, 0]
    armed = [r for r in rows if (r.get('submitted') or {}).get('cut_enabled')]
    task_rows = [r for r in rows if not str(r.get('policy_phase', '')).startswith('RETURN_HOME')]
    skim = dict(
        armed_steps=len(armed),
        armed_time_s=[armed[0]['time_s'], armed[-1]['time_s']] if armed else None,
        measured_orientation=skim_orientation_report([r['measured']['tool_pose'] for r in armed], n_t, c,
                                                     source='measured tool edge (physically held knife)')
        if len(armed) >= 2 else None,
        commanded_orientation=skim_orientation_report([r['accepted_target_world_xyz_xyzw'] for r in armed], n_t, c,
                                                      source='accepted commanded edge target')
        if len(armed) >= 2 else None,
        measured_edge_z_minus_target_plus_clearance_m=stats([r['measured']['tool_pose'][2] - target_z for r in armed]),
        measured_edge_lateral_y_m=stats([r['measured']['tool_pose'][1] for r in armed]),
        lowest_blade_point_above_bar_top_m=stats([lowest_blade_z(task, r['measured']['tool_pose']) - bar_top_z
                                                  for r in armed]),
        measured_edge_start_world_m=armed[0]['measured']['tool_pose'][:3] if armed else None,
        measured_edge_end_world_m=armed[-1]['measured']['tool_pose'][:3] if armed else None,
        target_edge_world_z_m=target_z, bar_top_world_z_m=bar_top_z,
        material_reaction=reaction(armed, dt))
    analysis = dict(
        run=str(a.run), task_id=task.task_id, goal_verdict=result.get('task_status'),
        goal_items=(((result.get('task_outcome') or {}).get('outcome') or {}).get('metrics') or {}).get('items'),
        goal_checks=dict(integrity=((result.get('task_outcome') or {}).get('integrity') or {}).get('checks'),
                         outcome=((result.get('task_outcome') or {}).get('outcome') or {}).get('checks')),
        home_outcome=(result.get('home_outcome') or {}).get('status'),
        home_failed_checks=(result.get('home_outcome') or {}).get('failed_checks'),
        episode_state_final=result.get('episode_state_final'),
        state_path=[(t['from'], t['to'], t['time_s'], t['reason']) for t in result.get('episode_transitions', [])],
        phases=phase_table(task, rows, target_z, bar_top_z), skim=skim,
        material_reaction_all_task_steps=reaction(task_rows, dt),
        grasp_task_phase=grasp_audit(task_rows), grasp_all=grasp_audit(rows),
        contacts_task_phase=contact_audit(task_rows), contacts_all=contact_audit(rows),
        ik_rejections=int(sum((r.get('info') or {}).get('ik_status') == 'rejected' for r in rows)),
        chip=chip_audit(task, episode / 'physics_state.npz', rows, out))
    if not a.skip_path_check:
        analysis['measured_task_path_clearance'] = measured_path_clearance(task, rows, out)
    write_json(out / 'analysis.json', matrix.jsonable(analysis))
    print(json.dumps(matrix.jsonable({k: analysis[k] for k in ('goal_verdict', 'home_outcome', 'goal_items')}))[:800])


if __name__ == '__main__':
    main()
