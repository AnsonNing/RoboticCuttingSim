#!/usr/bin/env python3
"""WP-12 Phase 4c summary: geometry capability matrix, material table, figures (read-only over run results).

  .venv/bin/python scripts/summarize_wp12_phase4c.py --preflight artifacts/wp12/phase4c_003/preflight.json \\
      --runs artifacts/wp12/phase4c_002 --out docs/evidence/wp12/phase4c_geometry_material_001.json \\
      --media docs/media/wp12/phase4c
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

COLUMNS = ('scene_compile', 'particle_sampling', 'settle_flat_board', 'kinematic_CUT', 'kinematic_TRIM',
           'cut_aware_extraction', 'goal_evaluation')
RULES = {
    'scene_compile': 'VALIDATED: compile_task PASS (flat-board placement included) and the run recompiled the same task id',
    'particle_sampling': 'VALIDATED: geometry_providers.lattice_report ok (one 6-connected body, 0 floating particles, '
                         'bottom layer on the board)',
    'settle_flat_board': 'VALIDATED: every run of the geometry met the settle rule at its first armed step '
                         '(speed < 10 mm/s, max |x - X| < 1 mm, min particle z > board top, centroid xy drift < 0.5 mm)',
    'kinematic_CUT': 'VALIDATED: sever goal PASS, 0 direct code-0 bridges, finite state; PARTIAL: >= 2 components but '
                     'goal FAIL or bridges > 0; FAIL: one component',
    'kinematic_TRIM': 'VALIDATED: remove_protrusions goal PASS; PARTIAL: goal FAIL but every protrusion removed >= 0.5; '
                      'FAIL otherwise; N/A without protrusions; UNSUPPORTED on the SDF provider (no top height field)',
    'cut_aware_extraction': 'VALIDATED: cut_aware_surface_v1 audit ok, mesh shells == lattice pieces of every '
                            'component and 0 visually-separated-but-bridged pairs in every run of the geometry',
    'goal_evaluation': 'VALIDATED: env.evaluate() returned a verdict with integrity PASS in every run of the geometry',
}


def load(path):
    return json.loads(Path(path).read_text())


def runs(root):
    out = {}
    for path in sorted(Path(root).glob('*/result.json')):
        out[path.parent.name] = load(path)
    return out


def cut_class(r):
    p, e = r.get('physics'), r.get('evaluation') or {}
    if r.get('status') != 'DONE' or p is None:
        return 'ERROR'
    if e.get('verdict') == 'PASS' and p['bridges_total'] == 0 and p['finite']:
        return 'VALIDATED'
    return 'PARTIAL' if p['mechanical_components'] >= 2 else 'FAIL'


def removed_fractions(r):
    items = (((r.get('evaluation') or {}).get('outcome') or {}).get('metrics') or {}).get('items') or []
    return {row['id']: row['removed_mass_fraction'] for row in items}


def trim_class(r):
    if r.get('status') != 'DONE':
        return 'ERROR'
    if (r.get('evaluation') or {}).get('verdict') == 'PASS':
        return 'VALIDATED'
    fractions = [v for v in removed_fractions(r).values() if v is not None]
    return 'PARTIAL' if fractions and min(fractions) >= .5 else 'FAIL'


def surface_ok(r):
    s = r.get('cut_aware_surface') or {}
    if 'audit_ok' not in s:
        return False
    return bool(s['audit_ok'] and s['visually_separated_mechanically_bridged'] == 0
                and all(c['shells'] == c['lattice_pieces'] for c in s['components']))


def evaluation_ok(r):
    e = r.get('evaluation') or {}
    return e.get('verdict') is not None and (e.get('integrity') or {}).get('status') == 'PASS'


def matrix(preflight, results):
    from run_wp12_phase4c import GEOMETRIES
    rows = {}
    for geometry, (kwargs, ops) in GEOMETRIES.items():
        names = [f'{geometry}_{op}' for op in ops]
        pre = [preflight['cases'][n] for n in names]
        got = [results.get(n) for n in names]
        row = {'ops': list(ops), 'provider': pre[0].get('provider'), 'particles': pre[0].get('particles'),
               'scene_kwargs': kwargs, 'runs': {n: (f'artifacts/wp12/phase4c_002/{n}' if results.get(n) else None)
                                                for n in names}}
        row['scene_compile'] = 'VALIDATED' if all(p['compile'] == 'PASS' for p in pre) and all(
            g is not None and g.get('task_id') == p.get('task_id') for g, p in zip(got, pre)) else (
            'FAIL' if any(p['compile'] != 'PASS' for p in pre) else 'PARTIAL (task id mismatch or run missing)')
        row['particle_sampling'] = 'VALIDATED' if all(p.get('lattice', {}).get('ok') for p in pre) else 'FAIL'
        if any(g is None for g in got):
            for key in COLUMNS[2:]:
                row[key] = 'NOT_RUN'
            rows[geometry] = row
            continue
        done = [g for g in got if g.get('status') == 'DONE']
        row['notes'] = [f"{n}: {g['status']} {g.get('error')}" for n, g in zip(names, got) if g.get('status') != 'DONE']
        settles = [g.get('settle') or {} for g in done]
        row['settle_flat_board'] = ('VALIDATED' if settles and all(s.get('pass') for s in settles) else 'FAIL') if done else 'NOT_RUN'
        by_op = dict(zip(ops, got))
        row['kinematic_CUT'] = cut_class(by_op['CUT']) if 'CUT' in by_op else 'N/A (TRIM-only follow-up row)'
        if 'TRIM' in by_op:
            row['kinematic_TRIM'] = trim_class(by_op['TRIM'])
        elif row['provider'] == 'sdf_csg_experimental_v1':
            row['kinematic_TRIM'] = 'UNSUPPORTED'
        else:
            row['kinematic_TRIM'] = 'N/A'
        partial = ' (DONE runs only)' if len(done) < len(got) else ''
        if not done:
            row['cut_aware_extraction'] = 'NOT_RUN'
        elif all(surface_ok(g) for g in done):
            row['cut_aware_extraction'] = 'VALIDATED'+partial
        else:
            audits = [g.get('cut_aware_surface') or {} for g in done]
            bridged = sum(a.get('visually_separated_mechanically_bridged', 0) for a in audits)
            row['cut_aware_extraction'] = (f'FAIL (audit ok; shells != components: {bridged} mechanically bridged '
                                           'multi-shell component(s))' if all(a.get('audit_ok') for a in audits)
                                           else 'FAIL (audit errors)')+partial
        row['goal_evaluation'] = ('VALIDATED' if all(evaluation_ok(g) for g in done) else 'FAIL')+partial if done else 'NOT_RUN'
        row['measurements'] = {op: summary(g) for op, g in by_op.items()}
        rows[geometry] = row
    return rows


def summary(r):
    p = r.get('physics') or {}
    s = r.get('cut_aware_surface') or {}
    d = r.get('deformation') or {}
    return {'status': r.get('status'), 'task_id': r.get('task_id'), 'verdict': (r.get('evaluation') or {}).get('verdict'),
            'components': p.get('mechanical_components'), 'mass_fractions': [round(x, 4) for x in p.get('component_mass_fractions', [])[:6]],
            'bridges': p.get('bridges_total'), 'removed_fractions': removed_fractions(r) or None,
            'min_detF': p.get('min_detF_episode'), 'max_speed_m_s': p.get('max_particle_speed_episode_m_s'),
            'blade_force_armed_mean_N': p.get('blade_force_armed_mean_N'), 'blade_force_armed_max_N': p.get('blade_force_armed_max_N'),
            'near_abs_detF_minus_1_p95': (d.get('near') or {}).get('abs_detF_minus_1_p95'),
            'near_displacement_p95_m': (d.get('near') or {}).get('displacement_p95_m'),
            'far_displacement_p95_m': (d.get('far') or {}).get('displacement_p95_m'),
            'top_near_sag_max_m': d.get('top_surface_near_sag_max_m'),
            'settle_pass': (r.get('settle') or {}).get('pass'), 'settle_max_displacement_m': (r.get('settle') or {}).get('max_displacement_m'),
            'shells': s.get('mesh_shells'), 'surface_components': s.get('mechanical_components'), 'surface_audit_ok': s.get('audit_ok'),
            'min_particle_z_final_m': p.get('min_particle_z_final_m'), 'labelled_particles': p.get('labelled_particles'),
            'wall_s': r.get('wall_s'), 'ms_per_control_step': r.get('ms_per_control_step'),
            'material_dt_s': r.get('material_dt_s'), 'device': r.get('device'), 'seed': r.get('seed'),
            'workspace_clamped_steps': r.get('workspace_clamped_steps'), 'plan_done': r.get('plan_done'),
            'error': r.get('error')}


def material_table(results):
    from run_wp12_phase4c import MATERIALS
    from robotic_cutting_sim import materials_wp12 as mw
    rows = {}
    for name, (profile, dt) in MATERIALS.items():
        p = mw.profiles()[profile]
        rows[name] = {'profile_id': profile, 'interpretation': p['interpretation'],
                      'recipe': {k: p[k] for k in mw.RECIPE_KEYS}, 'material_dt_s': dt,
                      'stability': mw.stability(profile, .004, dt, .001),
                      'CUT': summary(results[f'{name}_CUT']) if f'{name}_CUT' in results else None,
                      'TRIM': summary(results[f'{name}_TRIM']) if f'{name}_TRIM' in results else None}
    return rows


# ----------------------------------------------------------------------------- figures

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis='x', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def geometry_gallery(media):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_wp12_phase4c import GEOMETRIES, build, cases
    from robotic_cutting_sim.scene_config import compile_task
    names = list(GEOMETRIES)
    fig, axes = plt.subplots(len(names)//2+len(names) % 2, 4, figsize=(11, 1.25*len(names)/2+.6))
    specs = cases()
    for i, name in enumerate(names):
        op = GEOMETRIES[name][1][0]
        sc, goal = build(f'{name}_{op}', specs[f'{name}_{op}'])
        points = compile_task(sc, goal).sample_particles()[0].astype(float)*1e3
        row, col = divmod(i, 2)
        top, side = axes[row, 2*col], axes[row, 2*col+1]
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))   # last row of each (x, y) column = top
        columns = points[order]
        last = np.r_[np.any(np.diff(columns[:, :2], axis=0) != 0, axis=1), True]
        tops = columns[last]
        z0 = float(np.median(tops[:, 2]))
        image = top.scatter(tops[:, 0], tops[:, 1], c=tops[:, 2]-z0, s=.4, cmap='Blues', vmin=-1.5,
                            vmax=max(float(tops[:, 2].max()-z0), 1.), linewidths=0, rasterized=True)
        fig.colorbar(image, ax=top, fraction=.04, pad=.01).ax.tick_params(labelsize=5, colors=MUTED)
        mid = np.abs(points[:, 1]) < .6
        if name.startswith('G16'):
            mid = np.abs(points[:, 1]-0.) < .6
        side.scatter(points[mid, 0], points[mid, 2], s=.4, c=BLUE, linewidths=0, rasterized=True)
        for ax in (top, side):
            ax.set_aspect('equal')
            ax.tick_params(labelsize=5, colors=MUTED)
            for spine in ax.spines.values():
                spine.set_color(GRID)
        top.set_title(f'{name} (top surface, colour = z - median top z, mm)', fontsize=6.5, color=INK, loc='left')
        side.set_title('section y = 0 (x-z, mm)', fontsize=6.5, color=MUTED, loc='left')
    for ax in axes.flat[len(names)*2:]:
        ax.axis('off')
    fig.tight_layout()
    path = media/'geometry_gallery.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def final_states(media, root, names, filename, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = 4
    rows = (len(names)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 1.6*rows+.4))
    for ax, name in zip(axes.flat, names):
        path = Path(root)/name/'final_state.npz'
        if not path.exists():
            ax.axis('off')
            continue
        data = np.load(path)
        x, labels = data['positions_local'].astype(float)*1e3, data['labels']
        masses = np.bincount(labels)
        rank = np.argsort(np.argsort(-masses))   # colour follows the piece's mass rank, capped at 8 slots
        colours = np.array(CATEGORICAL)[np.minimum(rank[labels], 7)]
        mid = np.abs(data['reference_local'][:, 1]) < 6e-4
        ax.scatter(x[mid, 0], x[mid, 2], s=.5, c=colours[mid], linewidths=0, rasterized=True)
        ax.axhline(0, color=MUTED, linewidth=.6)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=5, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_title(f'{name}  ({len(masses)} comp.)', fontsize=6.5, color=INK, loc='left')
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=8, color=INK, x=.01, ha='left')
    fig.tight_layout()
    path = media/filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def material_bars(media, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(table)
    metrics = [('CUT', 'blade_force_armed_mean_N', 'CUT blade reaction, armed mean (N)', 1.),
               ('CUT', 'near_abs_detF_minus_1_p95', 'CUT |det F - 1| p95 within 3 mm of the plane', 1.),
               ('CUT', 'top_near_sag_max_m', 'CUT top-surface sag next to the kerf (mm)', 1e3),
               ('CUT', 'max_speed_m_s', 'CUT max particle speed (m/s)', 1.),
               ('TRIM', 'blade_force_armed_mean_N', 'TRIM blade reaction, armed mean (N)', 1.),
               ('TRIM', 'near_abs_detF_minus_1_p95', 'TRIM substrate |det F - 1| p95 (top 3 mm)', 1.),
               ('TRIM', 'removed', 'TRIM dome removed mass fraction', 1.),
               ('CUT', 'wall_s', 'CUT wall time (s)', 1.)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), sharey=True)
    y = np.arange(len(names))
    for ax, (op, key, label, scale) in zip(axes.flat, metrics):
        values = []
        for n in names:
            row = table[n][op] or {}
            v = (list((row.get('removed_fractions') or {}).values()) or [np.nan])[0] if key == 'removed' else row.get(key)
            values.append(np.nan if v is None else v*scale)
        ax.barh(y, values, color=BLUE, height=.62)
        ref = values[0]
        if np.isfinite(ref):
            ax.axvline(ref, color=MUTED, linewidth=.7, linestyle=(0, (3, 2)))
        for yi, v in zip(y, values):
            if np.isfinite(v):
                ax.text(v, yi, f' {v:.3g}', va='center', fontsize=5.5, color=MUTED)
        ax.set_title(label, fontsize=7, color=INK, loc='left')
        style(ax)
    axes[0, 0].set_yticks(y, [n.split('_', 1)[1] for n in names], fontsize=7)
    axes[0, 0].invert_yaxis()
    fig.suptitle('materials_wp12_v1 on the flat board (kinematic, tlmpm_v5 law 8, 1 mm / 4 mm, seed 42; dashed = '
                 'reference profile)', fontsize=8, color=INK, x=.01, ha='left')
    fig.tight_layout()
    path = media/'material_comparison.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def matrix_case(name):
    return 'G15_overlapping_TRIM' if 'TRIMPLANE' in name else name.replace('_pathv2_DIAG', '')


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def markdown_matrix(rows):
    head = '| Geometry | Provider | Particles | ' + ' | '.join(COLUMNS) + ' |\n|' + ' --- |'*(3+len(COLUMNS))
    lines = [head]
    for name, row in rows.items():
        lines.append(f"| {name} | {row['provider']} | {row['particles']} | " + ' | '.join(row[c] for c in COLUMNS) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--media', required=True)
    args = parser.parse_args()
    out, media = Path(args.out), Path(args.media)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    media.mkdir(parents=True, exist_ok=True)
    preflight, results = load(args.preflight), runs(args.runs)
    rows = matrix(preflight, results)
    table = material_table(results)
    from run_wp12_phase4c import GEOMETRIES
    figures = {'geometry_gallery': rel(geometry_gallery(media)),
               'cut_final_states': rel(final_states(media, args.runs, [f'{g}_CUT' for g, (_, ops) in GEOMETRIES.items()
                                                                       if 'CUT' in ops], 'geometry_cut_final.png',
                                                    'Final CUT states, section |y_ref| < 0.6 mm, colour = mechanical '
                                                    'component (mass rank), mm')),
               'trim_final_states': rel(final_states(media, args.runs, [f'{g}_TRIM' for g, (_, ops) in GEOMETRIES.items()
                                                                        if 'TRIM' in ops], 'geometry_trim_final.png',
                                                     'Final TRIM states, section |y_ref| < 0.6 mm, colour = mechanical '
                                                     'component (mass rank), mm')),
               'material_cut_final_states': rel(final_states(media, args.runs, [f'{m}_CUT' for m in table],
                                                             'material_cut_final.png', 'Final CUT states per material '
                                                             'profile, section |y_ref| < 0.6 mm, mm')),
               'material_comparison': rel(material_bars(media, table))}
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    evidence = {'evidence_id': 'wp12_phase4c_geometry_material_001', 'spec': 'docs/WP12_SPECIFICATION.md 4.5-4.6',
                'command': [sys.executable, *sys.argv], 'git_sha': sha,
                'scope': 'Kinematic ideal tracking (runtime_source = kinematic_ideal_tracking) on blade_master_flat_board_v1, '
                         'tlmpm_v5 cutting law 8 unchanged, 1 mm particles, 4 mm grid, material dt 0.1 ms (50 us for '
                         'wp12_stiff_v1), seed 42, one run per cell. Materials are PROVISIONAL_UNCALIBRATED.',
                'preflight': args.preflight, 'runs_root': args.runs,
                'preflight_sha256': hashlib.sha256(Path(args.preflight).read_bytes()).hexdigest(),
                'classification_rules': RULES, 'capability_matrix': rows,
                'capability_matrix_markdown': markdown_matrix(rows), 'refusals': preflight['refusals'],
                'material_stability': preflight['material_stability'], 'material_table': table, 'figures': figures,
                'diagnostics': {n: {'label': results[n].get('label'), 'goal_planner': results[n].get('goal_planner'),
                                    'goal': results[n].get('goal'), 'matrix_case': matrix_case(n), **summary(results[n]),
                                    'trim_plane_outcome': ((results[n].get('evaluation') or {}).get('outcome')
                                                           if results[n].get('op') == 'TRIM_PLANE' else None),
                                    'matrix_case_summary': summary(results[matrix_case(n)]) if matrix_case(n) in results else None}
                                for n in results if n.endswith('_DIAG')},
                'runs': {n: summary(r) for n, r in results.items()},
                'runs_missing': sorted(set(preflight['cases'])-set(results))}
    from run_wp12_phase4c import jsonable
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(evidence), indent=1))
    print(evidence['capability_matrix_markdown'])
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
