#!/usr/bin/env python3
"""WP-12 Phase 4a figures (read-only): docs/media/wp12/phase4/*.png from the phase4a diagnostics runs and sweeps.

  .venv/bin/python scripts/plot_wp12_phase4a.py --runs artifacts/wp12/phase4a_002 --sweep artifacts/wp12/phase4a_003/sweep \
      --out docs/media/wp12/phase4
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import cut_diagnostics as cd  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': .5})


def oblique_map(run, out, column_y_mm=-16.5):
    data = np.load(run/'wp12_phase4a_diag.npz')
    meta = json.loads((run/'wp12_phase4a_diag.json').read_text())['meta']
    X, Xf, codes = data['reference_local'].astype(float), data['final_positions'].astype(float), data['final_codes']
    digits = cd.digits_of(codes, 0, meta['code_encoding'])
    last = np.flatnonzero(data['step_cut'] == 0)[-1]
    pose = data['step_pose_end_local'][last].astype(float)
    R = Rotation.from_quat(pose[3:]).as_matrix()
    d, n, o = R[:, 0], R[:, 2], pose[:3]
    bridges = cd.direct_bridges(data['stencil_nodes'], data['stencil_weights'], data['stencil_gradients'], digits)
    column = np.abs(X[:, 1]*1e3-column_y_mm) < .25
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    for ax, P, title in ((axes[0], X, 'reference positions'), (axes[1], Xf, 'final positions (after sag)')):
        f, s = (P-o)@d*1e3, (P-o)@n*1e3
        window = column & (np.abs(s) < 14) & (f > -16) & (f < 8)
        for value, colour, label, marker in ((1, BLUE, 'digit 1 (s < 0 side)', 'o'), (2, ORANGE, 'digit 2', 's'),
                                             (0, GRAY, 'unassigned (code 0)', '.')):
            m = window & (digits == value)
            ax.scatter(f[m], s[m], s=7 if value else 10, c=colour, marker=marker, linewidths=0, label=label)
        m = np.isin(np.arange(len(X)), bridges) & window
        ax.scatter(f[m], s[m], s=40, facecolors='none', edgecolors=INK, linewidths=1., label='direct code-0 bridge')
        ax.axvline(0, color=INK, lw=1)
        ax.text(-.3, -14.6, 'final edge f = 0', color=INK, ha='right')
        for y in (-4, 4):
            ax.axhline(y, color=MUTED, lw=.6, ls='--')
        ax.text(-15.5, -3.6, 'core |s| ≤ Δx', color=MUTED, fontsize=6.5)
        ax.set_title(title, color=INK, fontsize=8)
        ax.set_xlabel('f along the advance d (mm), >0 = ahead of the edge')
    axes[0].set_ylabel('s along the swept normal (mm)')
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, .93), fontsize=6.5, frameon=True, framealpha=.9, edgecolor='none')
    fig.suptitle(f'E2_OBLIQUE_W_s2201 (+35°), law 8: outer lattice column y = {column_y_mm} mm, bottom of the bar',
                 fontsize=8, color=INK)
    fig.savefig(out/'phase4a_oblique_s2201_sf_map.png', dpi=150)
    plt.close(fig)


def sweep_figures(sweep, out):
    summary = json.loads((sweep/'summary.json').read_text())['rows']
    obl = sorted([(v['spec']['angle'], v) for k, v in summary.items() if k.startswith('obl_') and k[4:].isdigit()])
    if obl:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
        a = [x for x, _ in obl]
        axes[0].plot(a, [v['unassigned_within_12mm'] for _, v in obl], color=BLUE, lw=2, marker='o', ms=4,
                     label='unassigned within 12 mm')
        axes[0].plot(a, [v['direct_code0_bridges'] for _, v in obl], color=ORANGE, lw=2, marker='s', ms=4,
                     label='direct code-0 bridges')
        axes[0].set_xlabel('oblique angle θ (deg), overtravel 4 mm along d')
        axes[0].set_ylabel('particles')
        axes[0].legend(frameon=False, fontsize=6.5)
        axes[1].step(a, [v['components'] for _, v in obl], where='mid', color=AQUA, lw=2)
        axes[1].scatter(a, [v['components'] for _, v in obl], color=AQUA, s=14, zorder=3)
        axes[1].set_yticks([1, 2])
        axes[1].set_xlabel('oblique angle θ (deg)')
        axes[1].set_ylabel('mechanical components')
        fig.suptitle('Controlled oblique sweep (80 x 24 x 16 mm bar, 30 mm relief slot, law 8)', fontsize=8)
        fig.savefig(out/'phase4a_oblique_sweep.png', dpi=150)
        plt.close(fig)
    slab = sorted([(v['spec']['rake'], v) for k, v in summary.items()
                   if k.startswith('rake_') and not v['spec'].get('counterfactual')])
    dome = sorted([(v['spec']['rake'], v) for k, v in summary.items()
                   if k.startswith('dome_r') and not v['spec'].get('counterfactual')])
    if slab or dome:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
        for rows, colour, marker, name in ((slab, BLUE, 'o', 'slab'), (dome, ORANGE, 's', 'dome')):
            if rows:
                axes[0].plot([x for x, _ in rows], [v['segment_summary']['seg_ok_fraction'] for _, v in rows], color=colour,
                             lw=2, marker=marker, ms=4, label=f'motion gate open, {name} rows')
        if dome:
            axes[0].plot([x for x, _ in dome], [v['segment_summary']['seg_plow_fraction'] for _, v in dome], color=AQUA,
                         lw=1.5, ls='--', marker='^', ms=4, label='plow mask (dome rows)')
        axes[0].axvline(45, color=MUTED, ls=':', lw=.8)
        axes[0].text(45.4, .5, 'θ_max = 45°', color=MUTED, fontsize=6.5)
        axes[0].set_xlabel('blade rake ρ (deg), horizontal skim')
        axes[0].set_ylabel('fraction of armed segment-substeps')
        axes[0].legend(frameon=False, fontsize=6)
        if dome:
            axes[1].plot([x for x, _ in dome], [v['removal'] for _, v in dome], color=ORANGE, lw=2, marker='s', ms=4,
                         label='dome chip removed (law 8)')
            cf = sorted([(v['spec']['rake'], v) for k, v in summary.items() if k.startswith('dome_') and v['spec'].get('counterfactual')])
            if cf:
                axes[1].scatter([x for x, _ in cf], [v['removal'] for _, v in cf], marker='x', color=INK, s=30, zorder=3,
                                label='COUNTERFACTUAL θ_max = 60° (not law 8)')
            axes[1].axhline(.9, color=MUTED, ls=':', lw=.8)
            axes[1].text(10, .92, 'TRIM criterion 0.9', color=MUTED, fontsize=6.5)
            axes[1].set_xlabel('blade rake ρ (deg)')
            axes[1].set_ylabel('removed fraction of above-plane mass')
            axes[1].set_ylim(-.05, 1.05)
            axes[1].legend(frameon=False, fontsize=6, loc='center left')
        fig.suptitle('Controlled rake sweep, law 8 (slab 80 x 24 x 12 mm; dome r 12 h 8 mm on a 100 x 30 x 20 mm bar)',
                     fontsize=8)
        fig.savefig(out/'phase4a_rake_sweep.png', dpi=150)
        plt.close(fig)
    phase = sorted([(v['spec'].get('edge_z', .0202), v) for k, v in summary.items()
                    if k.startswith('dome_phase') or k == 'dome_r10'])
    if phase:
        fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
        z = [(e-.020)*1e3 for e, _ in phase]
        ax.plot(z, [v['lowest_above_layer']['attached']/max(v['lowest_above_layer']['count'], 1) for _, v in phase],
                color=BLUE, lw=2, marker='o', ms=4, label='lowest chip layer left on the substrate')
        ax.plot(z, [1-v['removal'] for _, v in phase], color=ORANGE, lw=2, marker='s', ms=4, label='1 - removal')
        ax.set_xlabel('edge above base top (mm); chip layer at +0.5')
        ax.set_ylabel('fraction')
        ax.legend(frameon=False, fontsize=6)
        ax.set_title('Thin-chip residue vs lattice phase (dome, ρ = 10°)', fontsize=8)
        fig.savefig(out/'phase4a_trim_phase_sweep.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--sweep', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.runs and (args.runs/'s2201'/'wp12_phase4a_diag.npz').exists():
        oblique_map(args.runs/'s2201', args.out)
    if args.sweep and (args.sweep/'summary.json').exists():
        sweep_figures(args.sweep, args.out)


if __name__ == '__main__':
    main()
