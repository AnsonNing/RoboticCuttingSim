#!/usr/bin/env python3
"""WP-12 Phase 7: write the Demo 2 and Demo 6 scene / goal files from the E2 protocol rows (decision E-4, freeze
decision section 3). CPU only, no physics; new files only (an existing file is refused, never overwritten).

Each pair is exactly the E2 episode input: ``run_wp12_phase4_e2.build_row_sample`` (the Phase 5 generators with the
E2 protocol's frozen ``common`` block) compiled by ``generalization_wp12.compile_sample`` (max_cuts = planner digits,
kinematic duration rule), i.e. the ``scene`` / ``goal`` recorded in ``artifacts/wp12/phase4_e2_001/<row>/manifest.json``.
The Native run keeps the scene and replaces only ``goal.episode.duration_s`` by the Native duration rule
(``configs/wp12/phase7_demos_v1.json`` common.duration_rule, as workcell_live_v1).

  .venv/bin/python scripts/make_wp12_phase7_demo_inputs.py            # writes configs/wp12/phase7/*.json
  .venv/bin/python scripts/make_wp12_phase7_demo_inputs.py --check    # compares the committed files (no write)
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
OUT = ROOT/'configs/wp12/phase7'
DEMO_ROWS = {'demo2_slot_p20': 'SLOT_p20', 'demo6_cut_push_trim': 'CUT_PUSH_TRIM'}


def build_inputs():
    """{name: {'row', 'scene', 'goal', 'task_id'}} rebuilt from the committed E2 protocol."""
    import run_wp12_phase4_e2 as E2
    from robotic_cutting_sim import generalization_wp12 as W
    protocol = E2.load_protocol()
    config = E2.pseudo_config(protocol)
    out = {}
    for name, row_id in DEMO_ROWS.items():
        row = next(r for r in protocol['rows'] if r['id'] == row_id)
        sample = E2.build_row_sample(protocol, row)
        task, info = W.compile_sample(config, sample)
        out[name] = {'row': row_id, 'scene': info['scene'], 'goal': info['goal'], 'task_id': task.task_id}
    return out


def text(payload):
    return json.dumps(payload, indent=1)+'\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true')
    a = p.parse_args(argv)
    built = build_inputs()
    report, bad = {}, []
    for name, item in built.items():
        for kind in ('scene', 'goal'):
            path = OUT/f'{name}_{kind}.json'
            body = text(item[kind])
            if a.check:
                if not path.exists() or path.read_text() != body:
                    bad.append(str(path.relative_to(ROOT)))
            else:
                if path.exists():
                    raise SystemExit(f'{path} exists; never overwritten')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
            report[str(path.relative_to(ROOT))] = hashlib.sha256(body.encode()).hexdigest()
        report[f'{name}.e2_task_id'] = item['task_id']
    print(json.dumps({'check': a.check, 'mismatch': bad, 'files': report}, indent=1))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
