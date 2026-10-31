#!/usr/bin/env python3
"""WP-12 Phase 5 benchmark evidence (read-only): tables, per-episode metrics, spec 5.8 checks, repeats, BF boundary.

Reads the feasibility and kinematic output directories written by ``scripts/run_wp12_phase5.py`` and writes ONE new
evidence JSON (never overwritten) plus optional small figures. Nothing is executed; classes come from
``generalization_wp12.final_row`` / ``summarize`` exactly as ``run_wp12_phase5.py summarize`` computes them (every
declared sample counted once, missing outputs are not_run, the spec 5.7 targets over the ID_validated population).

  .venv/bin/python scripts/report_wp12_phase5.py --root artifacts/wp12/phase5_001 \\
      --evidence docs/evidence/wp12/phase5_benchmark_001.json --figures docs/media/wp12/phase5
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import generalization_wp12 as W  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(f'wp12_p5report_{name}', ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P5 = _load('run_wp12_phase5')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ----------------------------------------------------------------------------- per-episode

def _steps(evaluation):
    return (evaluation.get('outcome') or {}).get('steps') or []


def episode_row(final, feas, result, config):
    """One report row: identity, feasibility, class, task / physical / geometry metrics, runtime (report-only)."""
    sample = feas.get('sample') or {}
    row = {k: final.get(k) for k in ('sample_id', 'family', 'range_set', 'split', 'stratum', 'seed', 'variant',
                                      'ood_dimensions', 'feasibility_class', 'class', 'failure_kind', 'reason')}
    row['population'] = W.id_population(config, row['family'], row['variant']) if row['range_set'] == 'ID' else 'OOD'
    row['failing_classes'] = feas.get('failing_classes')
    row['derived'] = sample.get('derived')
    row['envelope_parameters'] = sample.get('envelope_parameters')
    row['task_id'] = (feas.get('compile') or {}).get('task_id')
    pre = (feas.get('checks') or {}).get('preflight') or {}
    row['preflight'] = {'status': pre.get('status'), 'classes': pre.get('classes'),
                        'binding': ((pre.get('receipt') or {}).get('path') or {}).get('binding')}
    if result is None:
        row['executed'] = False
        return row
    evaluation = result.get('evaluation') or {}
    row['executed'] = True
    row['device'] = result.get('device')
    row['label'] = result.get('label')
    row['provenance_frozen_hashes_verified'] = (result.get('provenance') or {}).get('frozen_hashes_verified')
    row['task'] = {'verdict': evaluation.get('verdict'), 'policy_done': result.get('policy_done'),
                   'failed_checks': W.G.failed_checks(evaluation) if evaluation.get('verdict') else None,
                   'metrics': result.get('metrics'),
                   'outcome_metrics': (evaluation.get('outcome') or {}).get('metrics'),
                   'step_status': [{'index': s.get('index'), 'goal_id': s.get('goal_id'), 'type': s.get('type'),
                                    'status': (s.get('outcome') or {}).get('status'),
                                    'metrics': (s.get('outcome') or {}).get('metrics')} for s in _steps(evaluation)]}
    geometry = result.get('geometry_quality') or {}
    validity = geometry.get('mesh_validity') or {}
    row['physical'] = {**(result.get('physical_quality') or {}), 'solver_invalid': result.get('solver_invalid'),
                       'slots': result.get('slots')}
    row['geometry'] = {'status': geometry.get('status', 'RUN' if geometry else None), 'level': geometry.get('level'),
                       'surface_profile': geometry.get('surface_profile'), 'watertight': validity.get('watertight'),
                       'shells': validity.get('shells'), 'non_manifold_edges': validity.get('non_manifold_edges'),
                       'boundary_edges': validity.get('boundary_edges'),
                       'mesh_shells_vs_components': [(geometry.get('topology') or {}).get('mesh_shells'),
                                                     (geometry.get('topology') or {}).get('mechanical_components')],
                       'volume': geometry.get('volume_report') or geometry.get('volume'),
                       'surface_error': geometry.get('surface_report')}
    row['runtime'] = result.get('runtime')
    row['reconnection_check'] = {k: v for k, v in (result.get('reconnection_check') or {}).items()
                                 if k != 'times_s'}
    row['evaluation_scope'] = {'evaluator_version': evaluation.get('evaluator_version'),
                               'digit_allocation': (evaluation.get('outcome') or {}).get('digit_allocation'),
                               'integrity_checks': (evaluation.get('integrity') or {}).get('checks'),
                               'outcome_checks': (evaluation.get('outcome') or {}).get('checks')}
    return row


# ----------------------------------------------------------------------------- spec 5.8 mixed operations

def mixed_checks(row, goal_steps):
    """Spec 5.8 per-sample checks (report-only; goal_eval_v2a decides task success)."""
    scope = row.get('evaluation_scope') or {}
    integrity, outcome = scope.get('integrity_checks') or {}, scope.get('outcome_checks') or {}
    allocation = scope.get('digit_allocation') or {}
    steps = (row.get('task') or {}).get('step_status') or []
    physical = row.get('physical') or {}
    penetration = physical.get('penetration') or {}
    reconnection = row.get('reconnection_check') or {}
    return {
        'cut_history_persists': integrity.get('persistent_history_no_digit_rewrites'),
        'cut_indices_correct': (allocation.get('plan_cross_check') == 'MATCH' and outcome.get('no_unowned_digit_labels') is not False)
        if allocation else None,
        'earlier_cuts_remain_valid': outcome.get('final_all_digit_substrate_integrity'),
        'no_accidental_reconnection': None if reconnection.get('status') not in ('PASS', 'FAIL') else reconnection['status'] == 'PASS',
        'evaluator_scopes_each_operation': (len(steps) == goal_steps and all(s.get('status') in ('PASS', 'FAIL') for s in steps))
        if goal_steps else None,
        'robot_and_tool_valid': (row.get('preflight', {}).get('status') == 'PASS' and physical.get('workspace_clamped_steps') == 0
                                 and penetration.get('particles_below_board_top', 0) == 0),
        'robot_and_tool_rule': 'motion preflight PASS (IK, collision, clearance), 0 workspace-clamped steps, 0 particle '
                               'centres below the rigid board top; kinematic ideal tracking has no grasp (Native only)',
        'verdict': (row.get('task') or {}).get('verdict'), 'class': row.get('class'),
    }


def mixed_summary(rows, config):
    out = {}
    for name, where in config['mixed_operations'].items():
        if name == 'checks':
            continue
        family, variant = where.split()[0], where.split()[1]
        group = [r for r in rows if r['family'] == family and r['variant'] == variant]
        executed = [r for r in group if r.get('executed')]
        per = {}
        for r in executed:
            goal = ((r.get('task') or {}).get('metrics') or {}).get('steps')
            per[r['sample_id']] = mixed_checks(r, len(goal) if goal is not None else None)
        keys = ('cut_history_persists', 'cut_indices_correct', 'earlier_cuts_remain_valid', 'no_accidental_reconnection',
                'evaluator_scopes_each_operation', 'robot_and_tool_valid')
        out[name] = {'family': family, 'variant': variant, 'declared_samples': len(group),
                     'by_range_set': {rs: sum(r['range_set'] == rs for r in group) for rs in ('ID', 'OOD')},
                     'executed': len(executed), 'task_success': sum(r['class'] == 'task_success' for r in group),
                     'classes': {c: sum(r['class'] == c for r in group) for c in W.REPORT_CLASSES if any(r['class'] == c for r in group)},
                     'checks_true_of_executed': {k: f"{sum(v[k] is True for v in per.values())}/{len(per)}"
                                                 + (f" ({sum(v[k] is None for v in per.values())} not evaluable)"
                                                    if any(v[k] is None for v in per.values()) else '') for k in keys},
                     'per_sample': per}
    return out


# ----------------------------------------------------------------------------- repeats / BF / budget

def repeats(root, kroot, rows_by_id):
    sel = sorted(Path(root).glob('repeat_selection*.json'))
    if not sel:
        return {'status': 'NOT_RUN', 'reason': 'no repeat_selection.json'}
    selection = read(sel[0])
    out = []
    for job in selection['jobs']:
        sid = job['sample_id']
        first, second = read(Path(kroot)/sid/'result.json'), read(Path(root)/sid/'result.json')
        entry = {'sample_id': sid, 'device_first': (first or {}).get('device'), 'device_repeat': (second or {}).get('device')}
        for tag, r in (('first', first), ('repeat', second)):
            entry[f'class_{tag}'] = None if r is None else r.get('class')
            entry[f'verdict_{tag}'] = None if r is None else (r.get('evaluation') or {}).get('verdict')
            entry[f'components_{tag}'] = None if r is None else (r.get('physical_quality') or {}).get('mechanical_components')
        entry['same_verdict'] = None if second is None else entry['verdict_first'] == entry['verdict_repeat']
        entry['same_components'] = None if second is None else entry['components_first'] == entry['components_repeat']
        out.append(entry)
    done = [e for e in out if e['same_verdict'] is not None]
    return {'rule': selection['rule'], 'skipped': selection['skipped'], 'pairs': out,
            'verdict_agreement': f"{sum(e['same_verdict'] for e in done)}/{len(done)}",
            'component_agreement': f"{sum(e['same_components'] for e in done)}/{len(done)}"}


def bf_table(rows):
    out = []
    for r in rows:
        if r['family'] != 'BF':
            continue
        d = r.get('derived') or {}
        steps = (r.get('task') or {}).get('step_status') or []
        om = (r.get('task') or {}).get('outcome_metrics') or {}
        out.append({'sample_id': r['sample_id'], 'theta_deg': d.get('applied_tilt_deg'),
                    'abs_theta_deg': None if d.get('applied_tilt_deg') is None else abs(d['applied_tilt_deg']),
                    'class': r['class'], 'failure_kind': r['failure_kind'], 'verdict': (r.get('task') or {}).get('verdict'),
                    'mechanical_components': om.get('mechanical_components', (r.get('physical') or {}).get('mechanical_components')),
                    'failed_checks': (r.get('task') or {}).get('failed_checks'),
                    'cut_position_error_m': ((r.get('task') or {}).get('metrics') or {}).get('cut_position_error_m'),
                    'armed_end_mm': 'NOT_RUN (the Phase 5 episode records no cut_diagnostics_v1 armed-end trace)',
                    'bridges': 'NOT_RUN (no bridge analysis in the Phase 5 episode; parts = mechanical components)',
                    'steps': steps or None})
    return sorted(out, key=lambda e: (e['abs_theta_deg'] is None, e['abs_theta_deg']))


def budget(queue_logs):
    walls, starts, ends, pauses = [], 0, 0, []
    for log in queue_logs:
        for line in Path(log).read_text().splitlines():
            m = re.search(r' END (\S+) device=(\S+) rc=(-?\d+) wall=(\d+)s', line)
            if m:
                ends += 1
                walls.append({'sample_id': m.group(1), 'device': m.group(2), 'rc': int(m.group(3)), 'wall_s': int(m.group(4))})
            starts += ' START ' in line
            if ' PAUSE ' in line or ' ABORT ' in line:
                pauses.append(line)
    return {'queue_logs': [rel(p) for p in queue_logs], 'jobs_started': starts, 'jobs_ended': ends,
            'job_hours': sum(w['wall_s'] for w in walls)/3600, 'nonzero_rc': [w for w in walls if w['rc'] != 0],
            'pauses_or_aborts': pauses, 'job_wall_s': walls}


# ----------------------------------------------------------------------------- figures

def figures(out_dir, tables, rows, bf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    # dataviz reference categorical palette, slots 1-7 in fixed order over W.REPORT_CLASSES (validated adjacent-pair
    # order, light mode); identity is never colour-alone: legend + the evidence JSON tables.
    slots = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6')
    colours = dict(zip(W.REPORT_CLASSES, slots))
    strata = list(tables['by_family_range_set'])
    # 1. class distribution per stratum (every declared sample, counts)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bottom = [0]*len(strata)
    for cls in W.REPORT_CLASSES:
        values = [tables['by_family_range_set'][s]['counts'][cls] for s in strata]
        if not any(values):
            continue
        ax.bar(strata, values, bottom=bottom, color=colours[cls], label=cls.replace('_', ' '), width=0.7, edgecolor='#fcfcfb',
               linewidth=1.5)
        bottom = [b+v for b, v in zip(bottom, values)]
    ax.set_ylabel('samples (of 10 declared)', color=ink)
    ax.set_title('Phase 5 benchmark split (kinematic): class of every declared sample per stratum', color=ink, fontsize=11)
    ax.tick_params(axis='x', rotation=45, colors=ink)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    path = out_dir/'phase5_class_by_stratum.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    # 2. success among feasible with Wilson 95 % intervals
    fig, ax = plt.subplots(figsize=(10, 4.0))
    xs, ps, lo, hi, labels = [], [], [], [], []
    for i, s in enumerate(strata):
        stat = tables['by_family_range_set'][s]['success_among_feasible']
        if stat['n']:
            xs.append(i)
            ps.append(stat['p'])
            lo.append(stat['p']-stat['lo'])
            hi.append(stat['hi']-stat['p'])
        labels.append(f"{s}\n{stat['k']}/{stat['n']}")
    ax.errorbar(xs, ps, yerr=[lo, hi], fmt='o', color=ink, ecolor=muted, capsize=3, markersize=5)
    ax.axhline(0.9, color=muted, linewidth=0.8, linestyle='--')
    ax.text(len(strata)-0.5, 0.905, 'kinematic target 0.90 (ID_validated only)', color=muted, fontsize=8, ha='right', va='bottom')
    ax.set_xticks(range(len(strata)), labels, fontsize=8, color=ink)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel('success among feasible', color=ink)
    ax.set_title('Success among valid_feasible samples per stratum, Wilson 95 % interval (k/n under each label)', color=ink,
                 fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = out_dir/'phase5_success_among_feasible.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    # 3. BF boundary: |theta| vs outcome
    pts = [b for b in bf if b['abs_theta_deg'] is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        for b in pts:
            ok = b['class'] == 'task_success'
            ax.scatter(b['abs_theta_deg'], 1 if ok else 0, s=48, color=slots[0] if ok else slots[1], edgecolors='#fcfcfb',
                       linewidths=1.5,
                       marker='o' if (b['theta_deg'] or 0) >= 0 else 's', zorder=3)
        ax.set_yticks([0, 1], ['not success', 'task success'], color=ink)
        ax.set_xlabel('|solid-board tilt| (deg); circle = positive, square = negative', color=ink)
        ax.set_title('BF (OOD only): solid-board tilt boundary study, kinematic', color=ink, fontsize=11)
        ax.set_ylim(-0.5, 1.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.grid(True, color=grid, linewidth=0.6)
        fig.tight_layout()
        path = out_dir/'phase5_bf_boundary.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return [rel(p) for p in written]


# ----------------------------------------------------------------------------- main

def split_block(config, split, feas_dir, kroot):
    decl = P5.declared(config, split)
    feas = {p.stem: json.loads(p.read_text()) for p in Path(feas_dir).glob(f"{config['sample_prefix']}_*_{split}_*.json")}
    finals, rows = [], []
    for sid, metadata in decl:
        if sid not in feas:
            continue
        frow = feas[sid]
        execution = None
        if frow.get('feasibility_class') == 'valid_feasible' and kroot is not None:
            path = Path(kroot)/sid/'result.json'
            execution = read(path) if path.exists() else (
                {'status': 'ABORTED', 'reason': 'output directory without result.json'} if (Path(kroot)/sid).exists() else None)
        final = W.final_row({**metadata, **frow}, execution)
        finals.append(final)
        result = execution if execution is not None and execution.get('status') == 'COMPLETE' else None
        rows.append(episode_row(final, frow, result, config))
    targets = {k: v for k, v in config['targets'].items() if isinstance(v, dict) and v.get('runtime') == 'kinematic'}
    tables = W.summarize(finals, decl, targets, lambda r: W.id_population(config, r['family'], r.get('variant')))
    return tables, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--evidence', required=True, type=Path)
    p.add_argument('--figures', type=Path)
    p.add_argument('--interim', action='store_true', help='label the evidence INTERIM (held-out not yet run)')
    a = p.parse_args(argv)
    if a.evidence.exists():
        raise SystemExit(f'{a.evidence} exists; never overwritten')
    config = W.load_benchmark(W.BENCHMARK_V1)
    root = a.root if a.root.is_absolute() else ROOT/a.root
    out = {'schema': 'rcs_wp12_phase5_benchmark_evidence_v1', 'label': 'INTERIM' if a.interim else None,
           'generated': datetime.now().astimezone().isoformat(), 'root': rel(root),
           'config': {'benchmark': 'configs/wp12/benchmark_v1.json', 'benchmark_sha256': sha(W.BENCHMARK_V1),
                      'envelope': 'configs/wp12/supported_envelope_v2.json', 'envelope_sha256': sha(W.ENVELOPE_V2),
                      'pinned_match': sha(W.BENCHMARK_V1) == W.FROZEN_SHA256['configs/wp12/benchmark_v1.json']
                      and sha(W.ENVELOPE_V2) == W.FROZEN_SHA256['configs/wp12/supported_envelope_v2.json']},
           'scope': ('KINEMATIC ideal-tracking results (runtime_source kinematic_ideal_tracking): goal planner + WaypointPolicy '
                     'on the kinematic backend, tlmpm_v5 law 8, 1 mm / 4 mm / 0.1 ms, material seed 42, goal_eval_v2a. '
                     'No robot dynamics, grasp, slip or cameras; never a live result. Material parameters are provisional '
                     'and uncalibrated. Native sub-sample NOT_RUN here.'),
           'report_tool': {'script': 'scripts/report_wp12_phase5.py', 'sha256': sha(__file__),
                           'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                           'command': sys.argv}}
    splits = {}
    for split in ('benchmark', 'held_out'):
        feas_dir, kroot = root/f'feasibility_{split}', root/f'kinematic_{split}'
        if not feas_dir.exists():
            splits[split] = {'status': 'NOT_RUN', 'reason': f'{rel(feas_dir)} absent'}
            continue
        tables, rows = split_block(config, split, feas_dir, kroot if kroot.exists() else None)
        meta = [read(m) for m in sorted(feas_dir.glob('feasibility_meta*.json'))]
        splits[split] = {'feasibility_dir': rel(feas_dir), 'kinematic_root': rel(kroot) if kroot.exists() else None,
                         'feasibility_meta': meta, 'tables': tables, 'mixed_operations_5_8': mixed_summary(rows, config),
                         'bf_boundary': bf_table(rows), 'episodes': rows,
                         'budget': budget(sorted(root.glob(f'queue_{split}*.log')))}
    out['splits'] = splits
    if (root/'repeats').exists():
        out['reproducibility_repeats'] = repeats(root/'repeats', root/'kinematic_benchmark', None)
        out['reproducibility_repeats']['budget'] = budget(sorted(root.glob('queue_repeats*.log')))
    else:
        out['reproducibility_repeats'] = {'status': 'NOT_RUN'}
    demo7 = sorted((ROOT/'artifacts/wp12').glob('phase7_demo7_selection_*.json'))
    out['demo7_selection'] = None if not demo7 else {'file': rel(demo7[0]), 'demo_7_sample_id': read(demo7[0])['demo_7_sample_id'],
                                                     'pool_size': read(demo7[0])['pool_size']}
    if a.figures and 'tables' in splits.get('benchmark', {}):
        out['figures'] = figures(a.figures, splits['benchmark']['tables'], splits['benchmark']['episodes'],
                                 splits['benchmark']['bf_boundary'])
    a.evidence.parent.mkdir(parents=True, exist_ok=True)
    a.evidence.write_text(json.dumps(P5.jsonable(out), indent=1, allow_nan=False, default=str))
    bench = splits.get('benchmark', {}).get('tables', {}).get('overall')
    if bench:
        print(json.dumps({k: bench[k] for k in ('samples', 'counts', 'feasibility_rate', 'success_among_feasible', 'overall_success')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
