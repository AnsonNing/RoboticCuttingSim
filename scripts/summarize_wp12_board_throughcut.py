#!/usr/bin/env python3
"""Summarize scripts/run_wp12_board_throughcut.py rows: compact table, evidence JSON and one PNG.

  .venv/bin/python scripts/summarize_wp12_board_throughcut.py artifacts/wp12/workcell_throughcut_00{1..9} \\
      --evidence docs/evidence/wp12/phase2_throughcut_001.json \\
      --figure docs/media/wp12/phase2_throughcut_bottom_layers_001.png [--notes notes.json]

Read-only over the row outputs; refuses to overwrite the evidence JSON or the figure. ``--notes`` is an
optional JSON object merged under ``analysis_notes`` (code answer, recommendation) so the evidence file
keeps the written conclusions next to the measured numbers.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_LABEL = 'DIAGNOSTIC_PENETRATION_NOT_ADOPTABLE'
ORDER = ['S_P100', 'S_P050', 'S_P0375', 'S_P025', 'S_000', 'S_M025_DIAG', 'D8_M400', 'D8_000', 'D40_M400', 'D40_000']


def row_metrics(path):
    r = json.loads((path/'result.json').read_text())
    if r.get('verdict') == 'ERROR':
        return {'row': r['row'], 'dir': str(path), 'status': 'ERROR', 'error': r.get('error')}
    hold, final = r['analysis']['HOLD'], r['analysis']['FINAL']
    contact = r['contact']
    clean = (final['mechanical_components'] >= 2 and final['two_substantial_parts_each_ge_40pct']
             and final['two_main_parts_together_ge_99pct'] and r['persistence']['persists_ge_0_5_s']
             and final['direct_code0_bridges']['count'] == 0)
    m = {'row': r['row'], 'label': r['label'], 'option': r['option'], 'support': r['support'],
         'declared_h_m': r['h_m'], 'edge_stop_local_z_m': r['blade_board']['edge_lowest_point_local_z_m'],
         'blade_overlaps_board_box_m': r['blade_board']['blade_overlaps_board_box_m'],
         'lowest_point_is_edge_vertex': r['blade_board']['lowest_point_is_edge_vertex'],
         'dir': str(path), 'command': r['command'], 'seed': r['seed'], 'device': r['device'],
         'wall_s': r['wall_s'], 'loop_wall_s': r['context']['loop_wall_s'], 'formal': r['formal'],
         'git_sha': r['git_sha'], 'git_dirty': r['git_dirty'],
         'components_end_hold': hold['mechanical_components'], 'component_mass_fractions_end_hold': hold['component_mass_fractions'][:4],
         'components_end_settle': final['mechanical_components'],
         'component_masses_end_settle_kg': final['component_masses_kg'][:4],
         'component_mass_fractions_end_settle': final['component_mass_fractions'][:4],
         'two_part_v1_style': {k: final[k] for k in ('two_substantial_parts_each_ge_40pct', 'two_main_parts_together_ge_99pct')},
         'persistence': r['persistence'],
         'configured_sever_goal': {'verdict': r['evaluation_configured_sever_goal'].get('verdict'),
                                   'outcome_status': (r['evaluation_configured_sever_goal'].get('outcome') or {}).get('status')},
         'bottom_layer_links_end_hold': hold['crossing_links_by_layer'],
         'bottom_layer_links_end_settle': final['crossing_links_by_layer'],
         'layers_with_unconflicted_links_end_settle': final['crossing_links_all_layers']['layers_with_unconflicted'],
         'column_unassigned_by_layer_end_settle': final['column_unassigned_by_layer'],
         'column_particles_per_layer': final['column_particles_per_layer'],
         'direct_code0_bridges_end_hold': hold['direct_code0_bridges']['count'],
         'direct_code0_bridges_end_settle': final['direct_code0_bridges']['count'],
         'bridge_layers_end_settle': final['direct_code0_bridges']['by_layer'],
         'separation_end_settle': final.get('two_largest_pieces'),
         'board_contact': {'weight_N': contact['weight_N'],
                           **{phase: None if contact[phase] is None else {
                               'fixture_Fz_mean_N': contact[phase]['fixture_force_z_momentum_balance_N']['mean'],
                               'fixture_Fz_max_N': contact[phase]['fixture_force_z_momentum_balance_N']['max'],
                               'fixture_Fz_sample_mean_N': contact[phase]['fixture_force_z_sample_N']['mean'],
                               'fixture_impulse_z_Ns': contact[phase]['fixture_impulse_z_momentum_balance_Ns'],
                               'blade_Fz_on_tool_mean_N': contact[phase]['blade_force_on_tool_z_N']['mean'],
                               'min_particle_z_m': contact[phase]['min_particle_z_m'],
                               'max_particles_below_board_top': contact[phase]['max_particles_below_board_top'],
                               'max_particle_centres_inside_board': contact[phase]['max_particle_centres_inside_board'],
                               'max_fixture_penetration_stat_m': contact[phase]['max_fixture_penetration_stat_m'],
                               'min_det_F': contact[phase]['min_det_F']}
                               for phase in ('pre_contact_rest', 'cut', 'hold', 'withdraw', 'settle')}},
         'board_end_settle': final['board'], 'det_F_end_hold': hold['det_F'], 'det_F_end_settle': final['det_F'],
         'clean_through_cut': bool(clean)}
    m['bottom_layer_lift'] = bottom_layer_lift(path)
    m['status'] = 'PASS' if clean else 'FAIL'
    m['status_meaning'] = 'PASS = clean through-cut (>= 2 components, v1-style mass split, persists >= 0.5 s, zero bridges)'
    return m


def bottom_layer_lift(path):
    """Current z of the reference bottom layer (k = 0) near the cut plane (|x_ref| < 4.5 mm) per snapshot.

    Rest height on the board is ~0.43 mm (particle centre, contact radius 0.45 mm); larger values mean the
    bottom of the bar near the cut has lifted off the board."""
    data = np.load(Path(path)/'snapshots.npz')
    lattice, ref = data['lattice'].astype(int), data['reference_local']
    layer = lattice[:, 2] == 0
    near = layer & (np.abs(ref[:, 0]) < .0045)
    far = layer & (np.abs(ref[:, 0]) > .03)
    out = {}
    for name in ('CUT', 'HOLD', 'FINAL'):
        z = data[f'{name}_positions'][:, 2].astype(float)
        out[name] = {'near_plane_z_max_m': float(z[near].max()), 'near_plane_z_median_m': float(np.median(z[near])),
                     'far_z_mean_m': float(z[far].mean()), 'layer0_particles_above_0_6mm': int((z[layer] > .0006).sum())}
    return out


def table(rows):
    head = ('| row | h (mm) | comp hold/settle | mass fractions (settle) | L0 conflict/links | L1 | L2 | bridges | '
            'min z (mm) | below top | Fz hold mean/max (N) | min detF | wall (s) | result |')
    lines = [head, '|' + '---|'*14]
    for m in rows:
        if m.get('status') == 'ERROR':
            lines.append(f"| {m['row']} | ERROR {m['error']} |")
            continue
        links = m['bottom_layer_links_end_settle']
        cell = lambda k: f"{links[str(k)]['conflicting']}/{links[str(k)]['links']}" if str(k) in links else '-'  # noqa: E731
        hold = m['board_contact']['hold'] or {}
        fr = ', '.join(f'{f:.3f}' for f in m['component_mass_fractions_end_settle'][:3])
        tag = ' DIAG' if m['label'] == DIAGNOSTIC_LABEL else ''
        lines.append(
            f"| {m['row']}{tag} | {m['declared_h_m']*1e3:+.3f} | {m['components_end_hold']}/{m['components_end_settle']} | {fr} | "
            f"{cell(0)} | {cell(1)} | {cell(2)} | {m['direct_code0_bridges_end_settle']} | "
            f"{m['board_end_settle']['min_particle_z_m']*1e3:.3f} | {m['board_end_settle']['particles_below_board_top']} | "
            f"{hold.get('fixture_Fz_mean_N', float('nan')):.2f}/{hold.get('fixture_Fz_max_N', float('nan')):.2f} | "
            f"{min(m['det_F_end_hold']['min'], m['det_F_end_settle']['min']):.3f} | {m['wall_s']:.0f} | "
            f"{'DIAGNOSTIC' if tag else m['status']} |")
    return '\n'.join(lines)


def figure(rows, dirs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    sys.path.insert(0, str(ROOT/'src'))
    from robotic_cutting_sim.cut_topology import decode_digits
    sys.path.insert(0, str(ROOT/'scripts'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('study', ROOT/'scripts/run_wp12_board_throughcut.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    # Unassigned = neutral gray; negative side = slot 1 blue; positive side = slot 2 orange (light surface).
    cmap = ListedColormap(['#c3c2b7', '#2a78d6', '#eb6834'])
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(6.4, 1.05*n+.6), dpi=110, sharex=True)
    fig.patch.set_facecolor('#fcfcfb')
    for ax, m, d in zip(np.atleast_1d(axes), rows, dirs):
        data = np.load(Path(d)/'snapshots.npz')
        lattice = data['lattice'].astype(int)
        digits = decode_digits(data['FINAL_codes'], 1)[:, 0]
        nodes = data['stencil_nodes']
        weights = data['stencil_weights']
        bridges = study.direct_bridges(nodes, weights > 0, digits)  # weight>0 entries (gradient rule adds none here)
        j = 15
        sel = lattice[:, 1] == j
        grid = np.full((6, 16), np.nan)
        for (i, _, k), dgt in zip(lattice[sel], digits[sel]):
            if 72 <= i < 88 and k < 6:
                grid[k, i-72] = dgt
        ax.set_facecolor('#fcfcfb')
        ax.imshow(grid, origin='lower', cmap=cmap, vmin=-.5, vmax=2.5, extent=(-8, 8, 0, 6), aspect='auto',
                  interpolation='nearest')
        bsel = bridges[(lattice[bridges, 1] == j) & (np.abs(lattice[bridges, 0]-79.5) < 8) & (lattice[bridges, 2] < 6)]
        if len(bsel):
            ax.scatter(lattice[bsel, 0]-79.5, lattice[bsel, 2]+.5, s=10, c='#0b0b0b', marker='o', linewidths=0)
        for x in (-4, 0, 4):
            ax.axvline(x, color='#52514e', lw=.6, ls=':' if x else '-')
        tag = ' DIAGNOSTIC' if m['label'] == DIAGNOSTIC_LABEL else ''
        ax.set_title(f"{m['row']}{tag}: h = {m['declared_h_m']*1e3:+.3f} mm, components {m['components_end_settle']}, "
                     f"bridges {m['direct_code0_bridges_end_settle']}, L0 |x|<4 mm unassigned "
                     f"{m['column_unassigned_by_layer_end_settle']['0']}/{m['column_particles_per_layer']}", fontsize=7, color='#0b0b0b', loc='left', pad=2)
        ax.set_yticks([0, 2, 4, 6])
        ax.tick_params(labelsize=6.5, colors='#52514e')
        ax.set_ylabel('z (mm)', fontsize=6.5, color='#52514e')
        for s in ax.spines.values():
            s.set_color('#c3c2b7')
    np.atleast_1d(axes)[-1].set_xlabel('reference x (mm); cut plane x = 0, 4 mm grid nodes at x = -4, 0, 4', fontsize=7,
                                       color='#52514e')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color='#c3c2b7', label='unassigned (digit 0)'), Patch(color='#2a78d6', label='negative side (1)'),
               Patch(color='#eb6834', label='positive side (2)'),
               Line2D([], [], marker='o', ls='', color='#0b0b0b', markersize=3, label='code-0 bridge')]
    top = 1-.42/fig.get_figheight()
    fig.suptitle('Bottom 6 particle layers at y index 15, end of settle (reference lattice; bridges counted over all y)', fontsize=8, y=.997,
                 color='#0b0b0b', x=.5, va='top')
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=6.5, frameon=False,
               bbox_to_anchor=(.5, 1-.25/fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=.4)
    fig.savefig(path, dpi=110, facecolor='#fcfcfb')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', type=Path, nargs='+')
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--figure', type=Path)
    p.add_argument('--notes', type=Path)
    a = p.parse_args()
    for target in (a.evidence, a.figure):
        if target is not None and target.exists():
            p.error(f'Refusing to overwrite {target}')
    rows = sorted((row_metrics(d) for d in a.dirs), key=lambda m: ORDER.index(m['row']) if m['row'] in ORDER else 99)
    dirs = [m['dir'] for m in rows]
    print(table(rows))
    solid = [m for m in rows if m.get('support') == 'solid' and m.get('label') != DIAGNOSTIC_LABEL and m.get('status') != 'ERROR']
    clean = sorted(m['declared_h_m'] for m in solid if m['clean_through_cut'])
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    evidence = {'evidence_id': 'wp12_phase2_throughcut_001', 'spec': 'docs/WP12_SPECIFICATION.md section 2.4 options A-D',
                'scope': 'Kinematic ideal tracking, tlmpm_v5, cutting law 8, 1 mm particles, 4 mm grid, material dt 0.1 ms, '
                         'seed 42, CUT-01-R bar 160 x 30 x 20 mm (cutting_tasks_v1 material, provisional uncalibrated), '
                         'ruler_knife_v2_y; no PhysX, robot or cameras',
                'summary_command': sys.argv, 'summary_git_sha': sha, 'summary_git_dirty': dirty,
                'rows': rows, 'table_markdown': table(rows),
                'smallest_nonnegative_h_with_clean_through_cut_m': (min(h for h in clean if h >= 0)
                                                                    if any(h >= 0 for h in clean) else None),
                'diagnostic_rows': [m['row'] for m in rows if m.get('label') == DIAGNOSTIC_LABEL],
                'diagnostic_rule': f'{DIAGNOSTIC_LABEL}: the edge enters the rigid board; never a candidate'}
    if a.notes:
        evidence['analysis_notes'] = json.loads(a.notes.read_text())
    a.evidence.parent.mkdir(parents=True, exist_ok=True)
    a.evidence.write_text(json.dumps(evidence, indent=1, allow_nan=False)+'\n')
    if a.figure is not None:
        a.figure.parent.mkdir(parents=True, exist_ok=True)
        figure(rows, dirs, a.figure)
        print('figure', a.figure, a.figure.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
