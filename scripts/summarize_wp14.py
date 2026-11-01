#!/usr/bin/env python3
"""Read-only paired WP-14 integrity, outcome and quality receipts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from run_wp13 import ROOT, G, build
from summarize_wp13 import supplemental, roughness


def digest(value):
    return hashlib.sha256(json.dumps(G.jsonable(value), sort_keys=True).encode()).hexdigest()


def replay(task, actions):
    """Recompute each requested control from deployable kinematic proprioception.

    No material state is read. Compare every resolved pose and phase, rather
    than trusting a compiled task id or a manifest option dictionary alone.
    """
    from robotic_cutting_sim import goal_planner as P
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.control import nominal_edge_pose, resolve_delta
    from robotic_cutting_sim.scene_frames import matrix_pose, pose_matrix, invert_transform
    options = task.controller['planner']
    limits = P.action_limits(options, task.scene)
    plan = P.plan(task.scene, task.goal, task.tool_profile, options)
    policy = WaypointPolicy(task, plan, limits, **P.policy_options(options))
    edge = task.reference_edge_pose_world(0.)
    def observation(edge, time_s):
        hand = matrix_pose(pose_matrix(edge)@invert_transform(pose_matrix(task.hand_from_edge_xyz_xyzw)))
        return {'timestamp_s': time_s, 'proprio': {'tool_pose': edge, 'hand_pose': hand}}
    obs = observation(edge, 0.)
    commanded = nominal_edge_pose(obs['proprio'], task.hand_from_edge_xyz_xyzw)
    position, angle, mismatches, phase_rows = [], [], [], {}
    previous_time = 0.
    for r in actions:
        action = policy(obs)
        expected, _ = resolve_delta(commanded, action['target'], limits,
                                   frame=action.get('frame', 'world'), T_world_workpiece=task.T_world_workpiece)
        actual = np.asarray(r['edge_pose_world'], float)
        position.append(float(np.linalg.norm(actual[:3]-expected[:3])))
        angle.append(float((Rotation.from_quat(actual[3:])*Rotation.from_quat(expected[3:]).inv()).magnitude()))
        expected_cut = int(action.get('cut_index', 0)) if action['cut_enabled'] else -1
        if (r['phase'] != policy.phase or r['cut_enabled'] != action['cut_enabled'] or r['cut_index'] != expected_cut):
            mismatches.append({'step': r['step'], 'actual': [r['phase'], r['cut_enabled'], r['cut_index']],
                               'expected': [policy.phase, action['cut_enabled'], expected_cut]})
        group = phase_rows.setdefault(r['phase'], {'steps': 0, 'min_world_xyz_m': actual[:3].copy(),
            'max_world_xyz_m': actual[:3].copy(), 'speed_max_m_s': 0., 'world_xyzw': actual[3:].tolist(),
            'cut_ids': set(), 'cut_enabled': set()})
        group['steps'] += 1
        group['min_world_xyz_m'] = np.minimum(group['min_world_xyz_m'], actual[:3])
        group['max_world_xyz_m'] = np.maximum(group['max_world_xyz_m'], actual[:3])
        group['speed_max_m_s'] = max(group['speed_max_m_s'], float(np.linalg.norm(actual[:3]-edge[:3])/(r['time_s']-previous_time)))
        group['cut_ids'].add(r['cut_index']); group['cut_enabled'].add(r['cut_enabled'])
        edge, commanded = actual, actual
        previous_time = r['time_s']
        obs = observation(actual, previous_time)
    for r in phase_rows.values():
        r['cut_ids'] = sorted(r['cut_ids']); r['cut_enabled'] = sorted(r['cut_enabled'])
    return {'status': 'PASS' if position and max(position) < 1.e-10 and max(angle) < 1.e-10 and not mismatches else 'FAIL',
            'controls': len(actions), 'max_position_error_m': max(position, default=None),
            'max_orientation_error_rad': max(angle, default=None), 'phase_cut_mismatches': mismatches[:20],
            'required_waypoints': [asdict(w) for w in plan], 'executed_phases': phase_rows,
            'policy_done': policy.done, 'scope': 'All recorded kinematic commands replayed from compiled planner and prior recorded edge; no physics replay.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    rows, pairs, pending = [], [], []
    for case_path in protocol['rows']:
        case = json.loads(Path(case_path).read_text()); arms = {}
        for solver in protocol['arms']:
            out = a.root/solver/case['id']
            if not (out/'summary.json').exists():
                pending.append({'case': case['id'], 'solver': solver, 'status': 'NOT_RUN_OR_INCOMPLETE'})
                continue
            summary = json.loads((out/'summary.json').read_text())
            result = json.loads((out/'result.json').read_text())
            manifest = json.loads((out/'manifest.json').read_text())
            actions = [json.loads(line) for line in (out/'actions.jsonl').read_text().splitlines()]
            task, _ = build(case, solver)
            audit = replay(task, actions)
            audit['compiled_options_match'] = manifest['config']['planner_options'] == task.controller['planner']
            audit['compiled_source_selected'] = manifest['config']['planner_source'] == 'compiled'
            if not audit['compiled_options_match'] or not audit['compiled_source_selected']:
                audit['status'] = 'FAIL'
            fields = ('time_s', 'phase', 'cut_enabled', 'cut_index', 'edge_pose_world')
            force = np.linalg.norm(np.asarray([r['blade_force_world_N'] for r in actions]), axis=1)
            physical_pass = summary['class'] == 'task_success'
            if case['episode'] == 'PARTIAL':
                physical_pass = bool((summary.get('partial_cut_rule') or {}).get('accepted'))
            row = dict(case=case['id'], family=case['episode'], solver=solver, outcome_pass=physical_pass,
                original_class=summary['class'], original_reason=result['class_reason'],
                partial_control=summary.get('partial_cut_rule'), execution_integrity=audit,
                config=manifest['config'], metrics=summary['metrics'], sampled_stability=result['stability'],
                solver_receipt=summary.get('solver_receipt'), supplemental=supplemental(out/'final_state.npz', case, solver),
                roughness_rms_m=roughness(summary), control_path_sha256=digest([{k:r[k] for k in fields} for r in actions]),
                reaction_control_average_peak_N=float(force.max()), runtime=summary['runtime'],
                process_wall_s=summary['process_wall_s'], provenance={k:manifest[k] for k in
                    ('timestamp','command','python','backend','git_sha','git_dirty','packages','source_checkouts','local_code_sha256')},
                evidence={str(out/n):G.sha(out/n) for n in ('summary.json','manifest.json','result.json','actions.jsonl','final_state.npz')})
            arms[solver] = row; rows.append(row)
        if len(arms) == 2:
            b, c = [arms[s] for s in protocol['arms']]
            same_input = []
            for r in (b,c):
                config = r['config']
                scene = json.loads(json.dumps(config['scene'])); scene['solver'].pop('module')
                same_input.append(digest([scene, config['goal'], config['planner_options'], config['material_seed'],
                                          config['material_dt_s'], config['surface_profile']]))
            pair = dict(case=case['id'], family=case['episode'], baseline_solver=b['solver'], candidate_solver=c['solver'],
                baseline_pass=b['outcome_pass'], candidate_pass=c['outcome_pass'],
                matched_inputs=same_input[0] == same_input[1], identical_actual_commands=b['control_path_sha256'] == c['control_path_sha256'],
                both_compiled_integrity_pass=all(r['execution_integrity']['status']=='PASS' for r in (b,c)))
            for k in ('p95','p99'):
                bv, cv = b['supplemental']['near_cut'][k], c['supplemental']['near_cut'][k]
                pair[f'consistency_{k}_reduction_factor'] = bv/cv if bv is not None and cv else None
            rb, rc = b['roughness_rms_m'], c['roughness_rms_m']
            pair['roughness_reduction_fraction'] = 1.-rc/rb if rb and rc is not None else None
            pairs.append(pair)
    G.write_new(a.out, dict(protocol=str(a.protocol), protocol_sha256=G.sha(a.protocol), planned_episodes=protocol['episodes'],
        completed=len(rows), rows=rows, pairs=pairs, pending=pending,
        scope='Frozen compiled paths. Integrity, task outcome, surface quality and contact validity assessed independently. Uncalibrated materials; no default promotion.'))
    print(json.dumps(G.jsonable({'completed':len(rows),'pending':pending,'pairs':pairs}), indent=2))


if __name__ == '__main__':
    main()
