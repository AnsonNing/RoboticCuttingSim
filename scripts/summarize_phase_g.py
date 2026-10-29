#!/usr/bin/env python3
"""Summarize WP-11 Phase G dataset runs (reliability, Test E comparison) into one JSON (read-only over the runs).

  .venv/bin/python scripts/summarize_phase_g.py --run NAME=artifacts/wp11/phaseG/<dir> [...] --out <new>.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim.policy_env import jsonable  # noqa: E402


def outcome_metrics(result):
    """Goal-evaluator headline numbers (remove_protrusions / sever), whichever are present."""
    if not isinstance(result, dict):
        return None
    outcome = result.get('outcome') or {}
    metrics = outcome.get('metrics') or {}
    out = dict(verdict=result.get('verdict', result.get('status')), integrity=(result.get('integrity') or {}).get('status'),
               outcome=outcome.get('status'))
    items = metrics.get('items')
    if items:
        out['items'] = [{k: row.get(k) for k in ('id', 'removed_mass_fraction', 'pass')} for row in items]
    for key in ('mechanical_components', 'largest_component_below_plane_mass_share', 'detached_below_plane_mass_fraction',
                'piece_mass_fractions', 'main_parts_mass_fraction', 'plane_error_m', 'pieces'):
        if key in metrics:
            out[key] = metrics[key]
    return out


def summarize(name, path):
    path = Path(path)
    row = dict(name=name, dir=str(path))
    if not (path/'index.json').exists():
        row['status'] = 'NO_INDEX'
        for f in ('result.json', 'failure.json', 'preflight.json'):
            if (path/f).exists():
                row[f] = json.loads((path/f).read_text())
        return row
    index = json.loads((path/'index.json').read_text())
    manifest = json.loads((path/'dataset_manifest.json').read_text())
    row.update(runtime=index['runtime'], scene=index['scene'], goal=index['goal'], git_sha=manifest.get('git_sha'),
               git_dirty=manifest.get('git_dirty'), command=manifest.get('command'))
    reliability = json.loads((path/'reliability.json').read_text()) if (path/'reliability.json').exists() else None
    if reliability:
        row.update(POLICY_READY=reliability['POLICY_READY'], failing=reliability['failing'],
                   cross_episode=reliability['cross_episode'], memory=reliability['memory'],
                   determinism=reliability['determinism'], reliability_wall_s=reliability.get('wall_s'))
    episodes = []
    for entry in index['episodes']:
        meta_path = path/entry['dir']/'episode.json'
        if not meta_path.exists():
            episodes.append(entry)
            continue
        meta = json.loads(meta_path.read_text())
        rel = next((e for e in (reliability or {}).get('episodes', []) if e['episode_id'] == entry['episode_id']), {})
        episodes.append(dict(
            episode_id=entry['episode_id'], label=meta['label'], controller=meta['controller'], seed=meta['seed'],
            seed_source=meta['seed_source'], task_id=meta['task_id'], expect=meta.get('expect'),
            outcome=outcome_metrics(meta.get('task_result')), home_outcome=(meta.get('home_outcome') or {}).get('status'),
            home_reason=(meta.get('home_outcome') or {}).get('reason'), termination_reason=meta.get('termination_reason'),
            task_end_reason=meta.get('task_end_reason'), action_status_counts=meta.get('action_status_counts'),
            steps_recorded=meta.get('steps_recorded'), task_steps=meta.get('task_steps'), wall_s=meta.get('wall_s'),
            frequencies={k: meta['frequencies'].get(k) for k in ('control_rate_hz', 'recorded_camera_rate_hz',
                                                                 'policy_camera_rate_hz', 'measured_camera')},
            tracking_error_m=(meta.get('physical_diagnostics') or {}).get('tracking_error_m'),
            privileged_controller_inputs=meta.get('privileged_controller_inputs'),
            grasp_reset=meta['reset_checks'].get('grasp_reset_status'),
            memory_after=meta.get('memory_after'), reliability_failing=rel.get('failing'),
            tool_pose_repeat=rel.get('tool_pose_repeat')))
    row['episodes'] = episodes
    return row


def rescore(path, config_path):
    """Re-apply a reliability config to a stored dataset (episode.json files + fresh reader validation)."""
    from robotic_cutting_sim import dataset as ds
    path = Path(path)
    index = json.loads((path/'index.json').read_text())
    stored = [row for row in index['episodes'] if row.get('status') == 'STORED']
    metas = [json.loads((path/row['dir']/'episode.json').read_text()) for row in stored]
    validations = [ds.validate_episode(path, row['episode_id']) for row in stored]
    failures = [row for row in index['episodes'] if row.get('status') != 'STORED']
    config = json.loads(Path(config_path).read_text())
    plan = json.loads((path/'dataset_manifest.json').read_text())['config']['episodes']
    report = ds.reliability_report(metas, config, index['runtime'], validations=validations,
                                   execution_failures=failures, planned_episodes=len(plan))
    return {'config': str(config_path), 'version': config['version'], 'POLICY_READY': report['POLICY_READY'],
            'failing': report['failing'], 'cross_episode': report['cross_episode'],
            'scope': 'post-hoc rescoring of the stored episodes (read-only)'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, help='NAME=DIR')
    p.add_argument('--rescore', action='append', help='Also re-apply this reliability config to every run (read-only)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; never overwrite')
    rows = [summarize(*item.split('=', 1)) for item in a.run]
    for row in rows:
        if a.rescore and (Path(row['dir'])/'index.json').exists():
            row['rescored'] = [rescore(row['dir'], config) for config in a.rescore]
    a.out.write_text(json.dumps(jsonable(dict(runs=rows)), indent=2) + '\n')
    for row in rows:
        print(row['name'], row.get('POLICY_READY'), row.get('failing'))
        for e in row.get('episodes', []):
            print('  ', e.get('episode_id'), e.get('label'), (e.get('outcome') or {}).get('verdict'), e.get('home_outcome'),
                  e.get('action_status_counts'))


if __name__ == '__main__':
    sys.exit(main())
