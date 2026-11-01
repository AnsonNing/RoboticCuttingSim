#!/usr/bin/env python3
"""Compact completed Native receipts; task, quality and HOME remain separate."""
import argparse
import json
from pathlib import Path

from run_wp13 import G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    protocol=json.loads(args.protocol.read_text())
    for case_path in protocol['native_rows']:
        case=Path(case_path).stem
        for solver in ('tlmpm_v5', 'tlmpm_v7'):
            root = args.root/solver/case
            path = root/'analysis/analysis.json'
            d = json.loads(path.read_text())
            episode = root/'run/episode_000'
            material = json.loads((episode/'material.json').read_text())
            manifest = json.loads((episode/'manifest.json').read_text())
            result = json.loads((episode/'result.json').read_text())
            if (d['runtime_source'] != 'live_isaac_sim' or
                    material['solver_module'] != 'robotic_cutting_sim.'+solver):
                raise ValueError(f'Native or solver provenance mismatch: {root}')
            row = {'case': case, 'solver': solver, 'scene_seed': json.loads((root/'launch.json').read_text())['scene_seed'],
                   'material_seed': 42, 'class': d['class'], 'reason': d['class_reason'],
                   'runtime_source': d['runtime_source'], 'task_outcome': d['task_outcome'],
                   'legacy_v1_metrics': d['metrics'], 'home_status': d['home']['home_outcome']['status'],
                   'home_failed_checks': d['home']['home_outcome']['failed_checks'],
                   'grasp': d['grasp_task_phase'], 'ik': d['ik'], 'contacts': d['contacts_task_phase'],
                   'tracking_error_m': result['tracking_error_m'], 'rotation_error_rad': result['rotation_error_rad'],
                   'operations': d['operations'], 'knife_vs_board': d['knife_vs_board'],
                   'material_vs_board': d['material_vs_board'], 'forces': d['forces'],
                   'performance': d['performance'], 'finite_recorded_state': d['finite_recorded_state'],
                   'camera_frames': {k:v['frames'] for k,v in d['cameras'].items()},
                   'camera_pose_trace': d['camera_pose_trace'],
                   'source_sha256': {n:G.sha(episode/n) for n in ('result.json','material.json','manifest.json','physics_state.npz')},
                   'local_code_sha256': manifest['local_code_sha256'],
                   'analysis_path': str(path), 'analysis_sha256': G.sha(path),
                   'x_F_consistency': {'status':'NOT_RUN','reason':'Native recorder has no F'},
                   'research_profile':material.get('research_profile'),
                   'resolved_material':material.get('actual_backend_material'),
                   'plan_preflight':json.loads((root/'run/plan_preflight.json').read_text())}
            geometry = root/'physical_faces_001/geometry.json'
            if not geometry.exists(): geometry = root/'geometry.json'
            if geometry.exists(): row['position_only_geometry'] = json.loads(geometry.read_text())
            rows.append(row)
    output = {'planned':protocol['native_episodes'], 'completed':len(rows),
              'task_success':sum(r['class']=='TASK_SUCCESS' for r in rows),
              'home_pass':sum(r['home_status']=='PASS' for r in rows), 'rows':rows,
              'setup_failures':{'count':sum(not (p.parent/'run/episode_000/result.json').exists() for p in args.root.glob('*/*/attempt.json')),'scope':'All preserved attempts included; a prephysics failure is not a completed task.'},
              'scope':'Fresh predeclared paired Native demonstrations, not random generalization. '
                      'Independent task evaluation, legacy v1 precision and quality are separate; '
                      'a v1 cut-position metric may exceed its frozen benchmark while the task succeeds. '
                      'Complete operation metadata recorded before launch; analysis is read-only. '
                      'No calibration or force convergence established.'}
    for row in rows:
        if row['solver']=='tlmpm_v7' and (row['research_profile'] or {}).get('id')!=protocol['candidate_profile']:
            raise ValueError('Native did not execute the explicit declared profile')
    G.write_new(args.out,output)
    print(json.dumps({k:output[k] for k in ('planned','completed','task_success','home_pass')}))


if __name__ == '__main__':
    main()
