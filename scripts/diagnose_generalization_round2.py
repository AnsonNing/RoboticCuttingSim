#!/usr/bin/env python3
"""Read-only failure diagnostics for WP-11 Phase E round 2 (never changes a class or verdict).

For each preserved kinematic run it recompiles the task from the run's manifest (the id must match) and
reads ``physics_state.npz``:

* ``skim``: per (cut digit, protrusion) of a remove/trim step: the protrusion's above-plane particles by
  digit value (0 unassigned / 1 / 2), the own-digit components and removal, the unassigned owned particles,
  and a hypothetical relabel test (those particles set to side 2) that shows whether they are the code-zero
  bridge. The relabel is a diagnostic only.
* ``oblique``: per sever plane: digit values by signed distance to the plane (1 mm bins, +-8 mm) and where the
  unassigned particles within 8 mm lie.
* ``labels``: nonzero codes, armed / engaged control steps and the blade reaction force on armed steps.

    .venv/bin/python scripts/diagnose_generalization_round2.py --run-root artifacts/wp11/phaseE/round2_kinematic_001 \
        --extra artifacts/wp11/phaseE/round2_diag_001/E2_TEST_D2_s2501_S8 ... --out FILE
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import goals  # noqa: E402
from robotic_cutting_sim.scene_config import compile_task  # noqa: E402


def load(run):
    run = Path(run)
    config = json.loads((run/'manifest.json').read_text())['config']
    result = json.loads((run/'result.json').read_text())
    task = compile_task(config['scene'], config['goal'], seed=config['material_seed'])
    if task.task_id != result['task_id']:
        raise SystemExit(f'{run}: compiled {task.task_id}, recorded {result["task_id"]}')
    d = np.load(run/'physics_state.npz')
    data = {k: d[k] for k in ('time_s', 'codes', 'mass_kg', 'reference_positions', 'reference_stencil_nodes',
                              'reference_stencil_weights', 'reference_stencil_gradients')}
    data.update(positions=d['positions'].astype(float), code_encoding=str(d['code_encoding']), max_cuts=int(d['max_cuts']))
    return task, result, goals._Recording(data, task), [json.loads(line) for line in (run/'actions.jsonl').read_text().splitlines()]


def skim_rows(task, rec):
    goal, scene = task.goal, task.scene
    steps = goal['steps'] if goal['type'] == 'sequence' else [goal]
    alloc = goals.step_digit_allocation({'steps': steps}, scene)
    ref, final = rec.local(rec.reference), rec.codes[-1]
    H = scene['workpiece']['base_box_m']['height_m']
    owner = goals.protrusion_ownership(task, ref, H)
    anchors = ref[:, 2] <= ref[:, 2].min()+1e-7
    ids = list(task.protrusion_ids)
    rows = []
    for step, digits in zip(steps, alloc):
        if step['type'] == 'remove_protrusions':
            chosen = sorted(goals.selected_protrusions(step, task), key=lambda i: scene['workpiece']['protrusions'][i]['x_m'])
        elif step['type'] == 'trim_plane':
            region = step['region_local_m'] or {}
            chosen = [i for i, p in enumerate(scene['workpiece']['protrusions'])
                      if all(region[a][0] <= p[f'{a}_m'] <= region[a][1] for a in region)]
        else:
            continue
        for k, i in zip(digits, chosen):  # skims run in +x order (protrusion_order along_skim)
            dk = goals.decode_digit(final, k, rec.encoding)
            member = owner == i
            n, labels = rec.components_of(goals.mask_digits(final, [k], rec.encoding))
            substrate = np.isin(labels, np.unique(labels[anchors]))
            zero = np.flatnonzero(member & (dk == 0))
            fixed = final.copy()
            fixed[zero] += (2 << (2*k)) if rec.encoding == 'packed2' else 2*3**k
            n2, labels2 = rec.components_of(goals.mask_digits(fixed, [k], rec.encoding))
            substrate2 = np.isin(labels2, np.unique(labels2[anchors]))
            bump = scene['workpiece']['protrusions'][i]
            rows.append({'step': step['goal_id'], 'digit': int(k), 'protrusion': ids[i], 'family': bump['family'],
                         'x_m': bump['x_m'], 'radius_x_m': bump['radius_x_m'], 'height_m': bump['height_m'],
                         'owned_above_plane': int(member.sum()), 'digit_values_0_1_2': np.bincount(dk[member], minlength=3).tolist(),
                         'own_digit_components': int(n), 'own_digit_removed': int((member & ~substrate).sum()),
                         'unassigned_owned_local_mm': np.round(ref[zero]*1e3, 2).tolist(),
                         'hypothetical_relabel_removed': int((member & ~substrate2).sum()),
                         'hypothetical_relabel_rule': 'DIAGNOSTIC ONLY: the unassigned owned particles set to side 2'})
    return rows


def oblique_rows(task, rec):
    goal = task.goal
    steps = goal['steps'] if goal['type'] == 'sequence' else [goal]
    ref, final = rec.local(rec.reference), rec.codes[-1]
    alloc = goals.step_digit_allocation({'steps': steps}, task.scene)
    rows = []
    for step, digits in zip(steps, alloc):
        if step['type'] != 'sever':
            continue
        for plane, k in zip(step['planes'], digits):
            o, n = np.asarray(plane['origin_local_m']), np.asarray(plane['normal_local'])
            s = (ref-o)@n
            dk = goals.decode_digit(final, k, rec.encoding)
            bins = [{'signed_distance_mm': [float(lo), float(lo+1)],
                     'digit_values_0_1_2': np.bincount(dk[(s*1e3 >= lo) & (s*1e3 < lo+1)], minlength=3).tolist()}
                    for lo in np.arange(-8, 8, 1.)]
            zero = (dk == 0) & (np.abs(s) < 8e-3)
            rows.append({'step': step['goal_id'], 'digit': int(k), 'normal_local': n.tolist(), 'bins': bins,
                         'unassigned_within_8mm': int(zero.sum()),
                         'unassigned_z_mm': sorted(set(np.round(ref[zero, 2]*1e3, 1).tolist())),
                         'unassigned_x_range_mm': (np.round([ref[zero, 0].min()*1e3, ref[zero, 0].max()*1e3], 1).tolist()
                                                   if zero.any() else None)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-root', required=True)
    p.add_argument('--extra', nargs='*', default=[])
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} exists; outputs are never overwritten')
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report = {'label': 'READ_ONLY_DIAGNOSTIC (no class or verdict changes)', 'command': sys.argv,
              'git_sha': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
              'started': datetime.now().astimezone().isoformat(), 'runs': []}
    runs = sorted(p for p in Path(args.run_root).iterdir() if (p/'result.json').exists()) + [Path(e) for e in args.extra]
    for run in runs:
        task, result, rec, actions = load(run)
        armed = [r for r in actions if r['cut_enabled']]
        row = {'run': str(run), 'sample_id': result['sample_id'], 'task_id': result['task_id'], 'class': result['class'],
               'class_reason': result['class_reason'], 'slots': result.get('slots'),
               'labels': {'nonzero_codes': int(np.count_nonzero(rec.codes[-1])), 'particles': int(rec.codes.shape[1]),
                          'armed_steps': len(armed), 'engaged_steps': sum(bool(r['engaged']) for r in armed),
                          'max_blade_force_norm_N_armed': max((float(np.linalg.norm(r['blade_force_world_N'])) for r in armed), default=None)},
               'skim': skim_rows(task, rec), 'oblique': [r for r in oblique_rows(task, rec)
                                                          if abs(r['normal_local'][2]) > 1e-9]}
        report['runs'].append(row)
        print(row['sample_id'], run.name, row['class'], row['labels'], flush=True)
    report['finished'] = datetime.now().astimezone().isoformat()
    out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
