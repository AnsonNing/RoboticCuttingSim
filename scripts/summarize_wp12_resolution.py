#!/usr/bin/env python3
"""Read-only summary of the WP-12 resolution study (configs/wp12/resolution_protocol_v1.json).

Reads ``<run>/<row_id>/result.json``, ``<run>/queue.jsonl`` and the GNU-time logs; applies the protocol's
convergence rules verbatim (paths, noise band, tolerances, discrete / continuous / zero-limit rules); writes the
evidence JSON and the figures. Never writes into the run directory.

  .venv/bin/python scripts/summarize_wp12_resolution.py --run artifacts/wp12/resolution_002 \\
      --evidence docs/evidence/wp12/phase4_resolution_001.json --media docs/media/wp12/resolution
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/wp12/resolution_protocol_v1.json'
CASES = ('CUT', 'OBL20', 'TRIM', 'MULTI2')
SPACINGS = (2e-3, 1.5e-3, 1e-3, .75e-3, .5e-3)


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def last_cut_state(analysis):
    names = [n for n in analysis if n.startswith('END_CUT')]
    return analysis[sorted(names)[-1]] if names else None


def flat(r):
    """Flat metric dict of one result.json."""
    a = r.get('analysis') or {}
    fin = a.get('FINAL') or {}
    cut = last_cut_state(a) or {}
    case = r['row']['case']
    out = {'status': r.get('status'), 'particles': r.get('particles'), 'grid_nodes': r.get('grid_nodes'),
           'components': fin.get('mechanical_components'), 'wall_s': r.get('wall_s'),
           'ms_per_substep': g(r, 'timing', 'ms_per_substep_median'),
           'mempool_high_bytes': g(r, 'memory', 'mempool_high_bytes'),
           'mempool_high_after_surface_bytes': g(r, 'memory', 'mempool_high_after_surface_bytes'),
           'solver_device_bytes': g(r, 'memory', 'solver_device_bytes'),
           'stable': g(r, 'stability', 'stable'), 'min_det_F': g(r, 'stability', 'min_det_F_all_steps'),
           'max_speed_m_s': g(r, 'stability', 'max_speed_m_s'),
           'contact_force_mean_abs_N': g(r, 'contact_force', 'mean_abs_N'),
           'contact_force_peak_abs_N': g(r, 'contact_force', 'peak_abs_N'),
           'contact_force_steady_mean_abs_N': g(r, 'contact_force', 'steady_mean_abs_N'),
           'contact_force_resistance_N': g(r, 'contact_force', 'mean_resistance_along_advance_N'),
           'max_fixture_overlap_m': g(r, 'penetration', 'max_fixture_contact_overlap_m'),
           'max_blade_overlap_m': g(r, 'penetration', 'max_blade_contact_overlap_m'),
           'centres_below_board_max': g(r, 'penetration', 'max_particle_centres_below_board_top'),
           'min_particle_z_m': g(r, 'penetration', 'min_particle_z_m'),
           'direct_code0_bridges': fin.get('direct_code0_bridges'),
           'direct_code0_bridges_end_cut': cut.get('direct_code0_bridges')}
    out['one_minus_min_det_F'] = None if out['min_det_F'] is None else 1-out['min_det_F']
    out['bridges_zero'] = None if out['direct_code0_bridges'] is None else out['direct_code0_bridges'] == 0
    out['centres_below_board_zero'] = None if out['centres_below_board_max'] is None else out['centres_below_board_max'] == 0
    if case == 'TRIM':
        out.update(removal_fraction=fin.get('removal_fraction'), substrate_retained=fin.get('substrate_retained'),
                   criterion=fin.get('trim_criterion_wp11'),
                   lowest_layer_attached=g(fin, 'lowest_dome_layer', 'attached_to_substrate'),
                   lowest_layer_count=g(fin, 'lowest_dome_layer', 'count'),
                   lowest_layer_attached_fraction=(g(fin, 'lowest_dome_layer', 'attached_to_substrate')
                                                   / max(g(fin, 'lowest_dome_layer', 'count') or 1, 1)
                                                   if fin else None),
                   link_coverage_all=g(fin, 'trim_plane_links', 'coverage'),
                   severed_line_fraction=g(fin, 'severed_lines', 'fraction'),
                   trim_p95_depression_m=g(fin, 'column', 'p95_depression_m'),
                   trim_median_depression_m=g(fin, 'column', 'median_depression_m'),
                   column_p95_minus_piece_translation_m=g(fin, 'column', 'p95_abs_displacement_minus_piece_translation_m'),
                   column_p95_abs_m=g(fin, 'column', 'p95_abs_displacement_m'),
                   edge_gap_m=g(r, 'geometry', 'edge_gap_to_lowest_dome_layer_m'),
                   dome_volume_ratio=(g(r, 'geometry', 'dome_volume_lattice_m3') or np.nan)
                   / (g(r, 'geometry', 'dome_volume_analytic_m3') or np.nan))
    else:
        per = fin.get('per_plane') or []
        out.update(criterion=fin.get('separation_criterion'),
                   link_coverage_all=min((p['links_all']['coverage'] for p in per if p['links_all']['coverage'] is not None),
                                         default=None),
                   link_coverage_layer0=min((p['links_layer0']['coverage'] for p in per
                                             if p['links_layer0']['coverage'] is not None), default=None),
                   severed_line_fraction=min((g(p, 'severed_lines', 'fraction') for p in per
                                              if g(p, 'severed_lines', 'fraction') is not None), default=None),
                   severed_line_fraction_layer0=min((g(p, 'severed_lines_layer0', 'fraction') for p in per
                                                     if g(p, 'severed_lines_layer0', 'fraction') is not None), default=None),
                   unassigned_within_3mm=sum(p['unassigned_within_3mm'] for p in per) if per else None,
                   column_p95_minus_piece_translation_m=g(cut, 'column', 'p95_abs_displacement_minus_piece_translation_m'),
                   column_p95_minus_piece_translation_final_m=g(fin, 'column', 'p95_abs_displacement_minus_piece_translation_m'),
                   column_p95_abs_m=g(cut, 'column', 'p95_abs_displacement_m'),
                   column_p95_abs_final_m=g(fin, 'column', 'p95_abs_displacement_m'))
    sf = r.get('surface_final') or {}
    for key, name in (('A', 'particle_surface_v1'), ('B', 'cut_aware_surface_v1')):
        m = sf.get(name) or {}
        out[f'surface_{key}_status'] = m.get('status') or sf.get('status')
        out[f'surface_{key}_current_p95_m'] = m.get('current_p95_m')
        out[f'surface_{key}_reference_p95_m'] = m.get('reference_p95_m')
        out[f'surface_{key}_current_p95_s'] = g(m, 'scalars', 'current_p95_s')
        out[f'surface_{key}_reference_p95_s'] = g(m, 'scalars', 'reference_p95_s')
    return out


# ----------------------------------------------------------------------------- convergence rules (protocol)

def ratio_fn(h, p):
    h1, h2, h3 = h
    return (h1**p-h2**p)/(h2**p-h3**p)


def observed_order(h, q):
    """p solving (Q2-Q1)/(Q3-Q2) = (h1^p-h2^p)/(h2^p-h3^p); None if the differences change sign or vanish."""
    d1, d2 = q[1]-q[0], q[2]-q[1]
    if d1 == 0 or d2 == 0 or np.sign(d1) != np.sign(d2):
        return None, 'oscillatory_or_flat'
    target = d1/d2
    lo, hi = .05, 8.
    if target < ratio_fn(h, lo):
        return lo, 'below_range'
    if target > ratio_fn(h, hi):
        return hi, 'above_range'
    for _ in range(200):
        mid = (lo+hi)/2
        if ratio_fn(h, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2, 'solved'


def continuous_verdict(h, q, tol, nu, relative=True):
    """Protocol continuous rule on the three finest points (h descending)."""
    h, q = list(h), list(q)
    row = {'h': h, 'Q': q}
    if len(q) < 3:
        return {**row, 'verdict': 'INCONCLUSIVE', 'reason': 'fewer than 3 valid points'}
    h, q = h[-3:], q[-3:]
    row.update(h3=h, Q3=q)
    scale = abs(q[2]) if relative else 1.
    band = max(nu, tol*scale)
    spread = max(abs(q[i]-q[j]) for i in range(3) for j in range(i+1, 3))
    row.update(band=band, max_pairwise_difference=spread)
    if spread <= band:
        return {**row, 'verdict': 'CONVERGED', 'reason': 'insensitive within tolerance (order not identifiable)'}
    p, how = observed_order(h, q)
    row.update(p=p, p_status=how)
    if p is not None and how == 'solved':
        r = h[1]/h[2]
        gci = 1.25*abs(q[2]-q[1])/(r**p-1)
        row['richardson'] = q[2]-(q[1]-q[2])*h[2]**p/(h[1]**p-h[2]**p)
        row['gci_fine'] = gci/scale if relative and scale > 0 else gci
        if .5 <= p <= 3 and row['gci_fine'] <= tol:
            return {**row, 'verdict': 'CONVERGED', 'reason': f'order {p:.2f}, GCI {row["gci_fine"]:.3g} <= {tol}'}
    if abs(q[2]-q[1]) <= nu:
        return {**row, 'verdict': 'INCONCLUSIVE', 'reason': 'finest difference within the noise band, coarser not'}
    if p is not None and how in ('solved', 'above_range') and p > 3:
        return {**row, 'verdict': 'INCONCLUSIVE', 'reason': f'observed order {p:.2f} > 3'}
    if p is None:
        return {**row, 'verdict': 'NOT_CONVERGED', 'reason': 'differences change sign above the noise band'}
    if p < .5:
        return {**row, 'verdict': 'NOT_CONVERGED', 'reason': f'observed order {p:.2f} < 0.5'}
    return {**row, 'verdict': 'NOT_CONVERGED', 'reason': f'order {p:.2f} in range but GCI {row.get("gci_fine"):.3g} > {tol}'}


def zero_limit_verdict(h, q, floor, nu):
    h, q = list(h), list(q)
    row = {'h': h, 'Q': q}
    if len(q) < 3:
        return {**row, 'verdict': 'INCONCLUSIVE', 'reason': 'fewer than 3 valid points'}
    h, q = np.asarray(h[-3:]), np.asarray(q[-3:], float)
    if np.all(np.abs(q) <= floor):
        return {**row, 'verdict': 'CONVERGED', 'reason': f'all three <= floor {floor:g} m'}
    if np.all(q > 0):
        p = float(np.polyfit(np.log(h), np.log(q), 1)[0])
        row['loglog_slope'] = p
        if q[0] > q[1] > q[2] and .5 <= p <= 3:
            return {**row, 'verdict': 'CONVERGED', 'reason': f'monotone decreasing, log-log order {p:.2f}'}
    if np.max(q)-np.min(q) <= nu:
        return {**row, 'verdict': 'INCONCLUSIVE', 'reason': 'differences within the noise band (no decrease shown)'}
    return {**row, 'verdict': 'NOT_CONVERGED', 'reason': 'not monotone decreasing at order >= 0.5'}


def discrete_verdict(h, q):
    h, q = list(h), list(q)
    row = {'h': h, 'Q': q}
    if len(q) < 3:
        return {**row, 'verdict': 'INCONCLUSIVE', 'reason': 'fewer than 3 valid points'}
    if q[-1] == q[-2] == q[-3]:
        return {**row, 'verdict': 'CONVERGED', 'reason': f'identical at the three finest ({q[-1]})'}
    if q[-1] != q[-2]:
        return {**row, 'verdict': 'NOT_CONVERGED', 'reason': 'the two finest differ'}
    return {**row, 'verdict': 'INCONCLUSIVE', 'reason': 'two finest agree, third differs'}


DISCRETE = ('components', 'criterion', 'bridges_zero', 'centres_below_board_zero')
CONTINUOUS = ('contact_force_mean_abs_N', 'contact_force_steady_mean_abs_N', 'contact_force_peak_abs_N',
              'removal_fraction', 'substrate_retained', 'severed_line_fraction', 'severed_line_fraction_layer0', 'column_p95_minus_piece_translation_m',
              'trim_p95_depression_m', 'one_minus_min_det_F')
ZERO = ('max_fixture_overlap_m', 'surface_A_current_p95_m', 'surface_A_reference_p95_m', 'surface_B_current_p95_m',
        'surface_B_reference_p95_m')
TOLKEY = {'surface_A_current_p95_m': 'surface_current_p95_m', 'surface_B_current_p95_m': 'surface_current_p95_m',
          'surface_A_reference_p95_m': 'surface_reference_p95_m', 'surface_B_reference_p95_m': 'surface_reference_p95_m'}


def path_rows(rows, case, kind, s=None):
    """[(h, row_id)] coarse to fine for a convergence path."""
    def rid(case, s, grid, dt):
        return f'{case}_s{int(round(s*1e5)):03d}_{grid}_dt{int(round(dt*1e6)):04d}'
    if kind == 'P_s_primary':
        return [(sp, rid(case, sp, 'g4s', 5e-5)) for sp in SPACINGS]
    if kind == 'P_s_formal_dt':
        return [(sp, rid(case, sp, 'g4s', 1e-4)) for sp in SPACINGS]
    if kind == 'P_dt':
        dts = (2e-4, 1e-4, 5e-5, 2.5e-5) if abs(s-1e-3) < 1e-9 else (2e-4, 1e-4, 5e-5)
        return [(dt, rid(case, s, 'g4s', dt)) for dt in dts]
    if kind == 'P_ratio':
        return [(1e-3, rid(case, 1e-3, 'g4s', 1e-4)), (.75e-3, rid(case, .75e-3, 'g4mm', 1e-4)),
                (.5e-3, rid(case, .5e-3, 'g4mm', 1e-4))]
    raise ValueError(kind)


def noise(flats, case, metric, nu0):
    diffs = []
    for dt in ('0100', '0050'):
        a, b = flats.get(f'{case}_s100_g4s_dt{dt}'), flats.get(f'{case}_s100_g4s_dt{dt}_rep1')
        if a and b and a.get(metric) is not None and b.get(metric) is not None:
            diffs.append(abs(float(a[metric])-float(b[metric])))
    return max([nu0]+[2*d for d in diffs]), diffs


def verdicts(flats, protocol):
    tols = protocol['convergence']['tolerances']
    out = {}
    for case in CASES:
        out[case] = {}
        paths = {'P_s_primary': path_rows(flats, case, 'P_s_primary'), 'P_s_formal_dt': path_rows(flats, case, 'P_s_formal_dt'),
                 'P_ratio': path_rows(flats, case, 'P_ratio')}
        for sp in SPACINGS:
            paths[f'P_dt_s{int(round(sp*1e5)):03d}'] = path_rows(flats, case, 'P_dt', sp)
        for metric in DISCRETE+CONTINUOUS+ZERO:
            res = {}
            tk = tols.get(TOLKEY.get(metric, metric), {})
            nu, diffs = noise(flats, case, metric, tk.get('nu0', 0.))
            for pname, pr in paths.items():
                pts = [(h, flats[r][metric]) for h, r in pr
                       if r in flats and flats[r].get('status') == 'COMPLETED' and flats[r].get('stable')
                       and flats[r].get(metric) is not None]
                if not pts:
                    continue
                h, q = [p[0] for p in pts], [p[1] for p in pts]
                if metric in DISCRETE:
                    v = discrete_verdict(h, q)
                elif metric in ZERO:
                    v = zero_limit_verdict(h, q, tk.get('floor_m', 1e-6), nu)
                else:
                    rel = 'tol' in tk
                    v = continuous_verdict(h, [float(x) for x in q], tk.get('tol', tk.get('tol_abs', .1)), nu, rel)
                v['noise_band'] = nu
                v['repeat_differences'] = diffs
                res[pname] = v
            if not res:
                continue
            a, b = (res.get('P_s_primary') or {}).get('verdict'), (res.get('P_dt_s100') or {}).get('verdict')
            head = ('CONVERGED' if a == b == 'CONVERGED' else 'NOT_CONVERGED' if 'NOT_CONVERGED' in (a, b)
                    else 'INCONCLUSIVE')
            res['headline'] = {'verdict': head, 'P_s_primary': a, 'P_dt_s100': b}
            out[case][metric] = res
    return out


def scaling(flats, protocol):
    """Power-law fit of exclusive-timing ms/substep and solver memory against particle count (F8 rows)."""
    out = {}
    for case in CASES:
        pts = [(flats[r]['particles'], flats[r]['ms_per_substep'], flats[r]['solver_device_bytes'])
               for r in protocol['rows'] if r.startswith(case+'_') and r.endswith('_timing') and r in flats
               and flats[r].get('ms_per_substep')]
        if len(pts) >= 3:
            n, t, mb = (np.asarray(v, float) for v in zip(*pts))
            out[case] = {'points': pts, 'ms_per_substep_exponent': float(np.polyfit(np.log(n), np.log(t), 1)[0]),
                         'memory_exponent': float(np.polyfit(np.log(n), np.log(mb), 1)[0])}
    return out


# ----------------------------------------------------------------------------- figures

COLORS = {'CUT': '#2a78d6', 'OBL20': '#eb6834', 'TRIM': '#1baf7a', 'MULTI2': '#eda100'}
MARKERS = {'CUT': 'o', 'OBL20': 's', 'TRIM': '^', 'MULTI2': 'D'}


def figures(flats, media):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    media.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e0',
                         'grid.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False})
    written = []

    def series(case, metric, kind, s=None):
        pts = [(h, flats[r].get(metric)) for h, r in path_rows(flats, case, kind, s)
               if r in flats and flats[r].get('status') == 'COMPLETED' and flats[r].get(metric) is not None]
        return [p[0] for p in pts], [p[1] for p in pts]

    panels = [('contact_force_mean_abs_N', 'contact-window mean |F| (N)', True),
              ('column_p95_minus_piece_translation_m', 'P95 |u - piece translation| near cut (mm)', False),
              ('one_minus_min_det_F', '1 - min det F', False),
              ('surface_B_current_p95_m', 'cut_aware_surface_v1 current P95 (mm)', False)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (metric, label, logy) in zip(axes, panels):
        for case in CASES:
            for kind, ls in (('P_s_primary', '-'), ('P_s_formal_dt', ':')):
                h, q = series(case, metric, kind)
                if not h:
                    continue
                scale = 1e3 if metric.endswith('_m') else 1.
                ax.plot(np.asarray(h)*1e3, np.asarray(q, float)*scale, ls, color=COLORS[case], marker=MARKERS[case],
                        ms=5, lw=1.6, label=f'{case} dt {"0.05" if kind == "P_s_primary" else "0.1"} ms')
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xticks([.5, .75, 1, 1.5, 2])
        ax.set_xticklabels(['0.5', '0.75', '1', '1.5', '2'])
        ax.invert_xaxis()
        ax.set_xlabel('particle spacing s (mm), dx = 4 s')
        ax.set_title(label, fontsize=9)
    axes[0].legend(fontsize=6.5, ncol=2, frameon=False)
    fig.suptitle('Resolution study: spacing paths (solid dt 0.05 ms, dotted dt 0.1 ms)', fontsize=10)
    fig.tight_layout()
    path = media/'resolution_spacing_paths.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=False)
    for ax, case in zip(axes, CASES):
        shades = ['#9ec5f4', '#5b9be6', '#2a78d6', '#1a56a3', '#0d3466']
        for sp, col in zip(SPACINGS, shades):
            h, q = series(case, 'contact_force_mean_abs_N', 'P_dt', sp)
            if h:
                ax.plot(np.asarray(h)*1e3, q, '-o', color=col, ms=5, lw=1.6, label=f's {sp*1e3:g} mm')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('material dt (ms)')
        ax.set_title(f'{case}: contact-window mean |F| (N)', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = media/'resolution_force_vs_dt.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for variant, ls, lab in (('', '-', 'edge at 18.2 mm (metres fixed)'), ('_gap_s', '--', 'edge at 18 + 0.2 s (gap 0.3 s)')):
        xs, ys = [], []
        for sp in SPACINGS:
            r = f'TRIM_s{int(round(sp*1e5)):03d}_g4s_dt0050{variant}'
            if variant and abs(sp-1e-3) < 1e-9:
                r = 'TRIM_s100_g4s_dt0050'
            if r in flats and flats[r].get('removal_fraction') is not None:
                xs.append(sp*1e3)
                ys.append(flats[r]['removal_fraction'])
        ax.plot(xs, ys, ls, marker='^', color='#1baf7a', ms=6, lw=1.6, label=lab)
    ax.axhline(.9, color='#52514e', lw=.8, ls=':')
    ax.text(1.9, .905, 'WP-11 criterion 0.9', fontsize=7, color='#52514e')
    ax.set_xlabel('particle spacing s (mm)')
    ax.set_ylabel('TRIM removal fraction (dt 0.05 ms)')
    ax.invert_xaxis()
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    for variant, ls, lab in (('', '-', 'h = 0 (board top)'), ('_h0375m', '--', 'h = +0.375 mm'), ('_h0375s', ':', 'h = +0.375 s')):
        xs, ys = [], []
        for sp in SPACINGS:
            r = f'CUT_s{int(round(sp*1e5)):03d}_g4s_dt0050{variant}'
            if variant == '_h0375s' and abs(sp-1e-3) < 1e-9:
                r = 'CUT_s100_g4s_dt0050_h0375m'
            if r in flats and flats[r].get('direct_code0_bridges') is not None:
                xs.append(sp*1e3)
                ys.append(flats[r]['direct_code0_bridges'])
        ax.plot(xs, ys, ls, marker='o', color='#2a78d6', ms=6, lw=1.6, label=lab)
    ax.set_xlabel('particle spacing s (mm)')
    ax.set_ylabel('CUT direct code-0 bridges at the end')
    ax.invert_xaxis()
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = media/'resolution_confounds.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for case in CASES:
        pts = sorted((flats[r]['particles'], flats[r]['ms_per_substep']) for r in flats
                     if r.startswith(case+'_') and r.endswith('_timing') and flats[r].get('ms_per_substep'))
        if pts:
            ax.plot(*zip(*pts), '-', marker=MARKERS[case], color=COLORS[case], ms=5, lw=1.6, label=case)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('particles')
    ax.set_ylabel('ms per substep (exclusive, dt 0.1 ms)')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = media/'resolution_runtime_scaling.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(path)
    return [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in written]


def gnu(log):
    text = Path(log).read_text(errors='replace') if Path(log).exists() else ''
    st = re.findall(r'Exit status: (\d+)', text)
    rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', text)
    wall = re.findall(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', text)
    return {'exit_status': int(st[-1]) if st else None, 'max_rss_kb': int(rss[-1]) if rss else None,
            'wall': wall[-1] if wall else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--media', type=Path)
    a = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    psha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    results, flats, rows = {}, {}, {}
    for rid in list(protocol['rows'])+[p.name for p in sorted(a.run.glob('*_slots16'))]:
        f = a.run/rid/'result.json'
        log = ROOT/'artifacts/wp12/logs'/f'{a.run.name}_{rid}.log'
        if not f.exists():
            rows[rid] = {'status': 'NOT_RUN' if not (a.run/rid).exists() else 'INCOMPLETE'}
            continue
        r = json.loads(f.read_text())
        results[rid] = r
        fl = flat(r) if r.get('status') in ('COMPLETED', 'UNSTABLE') else {'status': r.get('status')}
        flats[rid] = fl
        rows[rid] = {**fl, 'row': r.get('row'), 'label': r.get('label'), 'git_sha': r.get('git_sha'),
                     'git_dirty': r.get('git_dirty'), 'runner_sha256': r.get('runner_sha256'),
                     'protocol_sha256': r.get('protocol_sha256'), 'protocol_sha256_matches': r.get('protocol_sha256') == psha,
                     'command': r.get('command'), 'device': r.get('device'), 'seed': r.get('seed'),
                     'error': r.get('error'), 'gnu_time': gnu(log), 'log': str(log.relative_to(ROOT)),
                     'cfl': r.get('cfl'), 'timing': r.get('timing'), 'memory': r.get('memory'),
                     'contact_force': r.get('contact_force'), 'penetration': r.get('penetration'),
                     'stability': r.get('stability'), 'cut_events': r.get('cut_events'),
                     'slot_diagnostics': r.get('slot_diagnostics'),
                     'analysis_final': (r.get('analysis') or {}).get('FINAL'),
                     'analysis_end_cut': last_cut_state(r.get('analysis') or {})}
    queue = []
    q = a.run/'queue.jsonl'
    if q.exists():
        queue = [json.loads(line) for line in q.read_text().splitlines() if line.strip()]
    conv = verdicts(flats, protocol)
    head = {case: {m: v['headline'] for m, v in d.items()} for case, d in conv.items()}
    status_counts = {}
    for r in rows.values():
        status_counts[r['status']] = status_counts.get(r['status'], 0)+1
    evidence = {
        'schema': 'wp12_phase4_resolution_evidence_v1', 'protocol': str(PROTOCOL.relative_to(ROOT)), 'protocol_sha256': psha,
        'protocol_commit': subprocess.run(['git', 'log', '-1', '--format=%H', '--', str(PROTOCOL)], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip() or None,
        'run_dir': str(a.run), 'status_counts': status_counts,
        'integrity': {'protocol_sha_mismatch_rows': [k for k, r in rows.items() if r.get('protocol_sha256_matches') is False],
                      'git_shas': sorted({r.get('git_sha') for r in rows.values() if r.get('git_sha')}),
                      'runner_sha256': sorted({r.get('runner_sha256') for r in rows.values() if r.get('runner_sha256')}),
                      'nonzero_exit_rows': [k for k, r in rows.items() if (r.get('gnu_time') or {}).get('exit_status') not in (0, None)],
                      'unstable_rows': [k for k, r in rows.items() if r.get('status') == 'UNSTABLE' or r.get('stable') is False]},
        'convergence_headline': head, 'convergence': conv, 'runtime_memory_scaling': scaling(flats, protocol),
        'rows': rows, 'queue': queue,
        'scope': protocol['scope']}
    if a.media:
        evidence['figures'] = figures(flats, a.media)
    a.evidence.parent.mkdir(parents=True, exist_ok=True)
    a.evidence.write_text(json.dumps(evidence, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))+'\n')
    print(json.dumps(status_counts))
    for case, d in head.items():
        print(case, {m: v['verdict'] for m, v in d.items()})


if __name__ == '__main__':
    main()
