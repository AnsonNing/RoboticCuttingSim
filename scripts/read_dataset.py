#!/usr/bin/env python3
"""Load, validate, analyze and (kinematic episodes) replay an ``rcs_dataset_v1`` policy dataset.

  .venv/bin/python scripts/read_dataset.py artifacts/wp11/phaseG/<dataset> [--episode episode_000] \
      [--replay-kinematic] [--device cpu] [--out report.json]

Validation checks every stored episode (required files, export sha256 vs the
index, 50 Hz task clock, action timestamps, frame/physics timestamps, no
privileged keys in the deployable export, evaluation and termination present,
final codes hash). Replay re-executes a kinematic episode's executed absolute
edge targets open loop on a fresh kinematic env and compares the final codes;
Native episodes are analyzed only (replay NOT_APPLICABLE without Isaac Sim).
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import dataset as ds  # noqa: E402
from robotic_cutting_sim.policy_env import jsonable  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    p.add_argument('--episode', action='append', help='Episode id (repeatable); default: every stored episode')
    p.add_argument('--replay-kinematic', action='store_true')
    p.add_argument('--device', help='Replay device override (default: the recorded one)')
    p.add_argument('--out', type=Path, help='Write the JSON report here (must not exist)')
    a = p.parse_args(argv)
    index = ds.load_index(a.root)
    ids = a.episode or [row['episode_id'] for row in index['episodes'] if row.get('status') == 'STORED']
    report = dict(root=str(a.root), version=index['version'], runtime=index.get('runtime'), episodes=[])
    for episode in ids:
        row = dict(validation=ds.validate_episode(a.root, episode), analysis=ds.analyze_episode(a.root, episode))
        if a.replay_kinematic:
            row['replay'] = ds.replay_kinematic(a.root, episode, device=a.device)
        report['episodes'].append(row)
        print(json.dumps(dict(episode=episode, validation=row['validation']['status'],
                              failing=row['validation']['failing'], verdict=row['analysis']['verdict'],
                              steps=row['analysis']['steps'], statuses=row['analysis']['status_counts'],
                              replay=(row.get('replay') or {}).get('status'))), flush=True)
    report['status'] = 'PASS' if all(r['validation']['status'] == 'PASS' and
                                     (r.get('replay') or {}).get('status', 'PASS') in ('PASS', 'NOT_APPLICABLE')
                                     for r in report['episodes']) else 'FAIL'
    if a.out:
        if a.out.exists():
            raise SystemExit(f'{a.out} exists; never overwrite')
        a.out.write_text(json.dumps(jsonable(report), indent=2) + '\n')
    print('READ_DATASET', report['status'])
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
