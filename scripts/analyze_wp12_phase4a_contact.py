#!/usr/bin/env python3
"""WP-12 Phase 4a: read-only contact-mechanism analyses (thin-chip residue / depression, chip-carry merge).

  .venv/bin/python scripts/analyze_wp12_phase4a_contact.py trim artifacts/wp12/phase4a_004/trim_k2_r10_law8_001
  .venv/bin/python scripts/analyze_wp12_phase4a_contact.py bumps3 artifacts/wp12/phase4a_004/bumps3_law8_v5_001

``trim`` (TRIM-02-K2-R10: base 20 mm, dome r 12 mm at the origin): the lowest above-plane layer (z = 20.5 mm), how
many particles took the substrate digit, where they were relative to their reference at the assignment instant, and
the final depression of the substrate top layers under the dome versus elsewhere.
``bumps3`` (Phase M Test C): the mechanical body holding chips, its non-bump particles, their codes, carry distance and
the assignment instant (s, f, lift) for every digit. Writes ``<run>/phase4a_contact_NNN.json``.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import cut_diagnostics as cd  # noqa: E402
from robotic_cutting_sim.cut_topology import mechanical_components_packed  # noqa: E402


def trim(data):
    X, Xf = data['reference_local'].astype(float), data['final_positions'].astype(float)
    digits = cd.digits_of(data['final_codes'], 0, 'ternary')
    dz = (Xf-X)[:, 2]*1e3
    dome = np.hypot(X[:, 0], X[:, 1]) <= .012
    out = {}
    for name, mask in (('substrate_top_layer_19.5mm_under_dome', np.isclose(X[:, 2], .0195) & dome),
                       ('substrate_top_layer_19.5mm_elsewhere_abs_x_lt_40mm', np.isclose(X[:, 2], .0195) & ~dome & (np.abs(X[:, 0]) < .04)),
                       ('substrate_layer_18.5mm_under_dome', np.isclose(X[:, 2], .0185) & dome)):
        out[name] = {'count': int(mask.sum()), 'final_dz_median_mm': float(np.median(dz[mask])),
                     'final_dz_p05_mm': float(np.percentile(dz[mask], 5)), 'final_dz_min_mm': float(dz[mask].min())}
    layer = np.isclose(X[:, 2], .0205)
    sub = layer & (digits == 1)
    chip = layer & (digits == 2)
    s, ax = data['diag_particle_assign_s_m'][:, 0], data['diag_particle_assign_x_m'][:, 0]
    out['lowest_chip_layer_20.5mm'] = {
        'count': int(layer.sum()), 'substrate_digit': int(sub.sum()), 'chip_digit': int(chip.sum()),
        'unassigned': int(np.count_nonzero(layer & (digits == 0))),
        'assign_s_median_mm': {'substrate_digit': float(np.median(s[sub])*1e3), 'chip_digit': float(np.median(s[chip])*1e3)},
        'assign_dz_median_mm': {'substrate_digit': float(np.median(ax[sub, 2]-X[sub, 2])*1e3),
                                'chip_digit': float(np.median(ax[chip, 2]-X[chip, 2])*1e3)},
        'substrate_digit_x_range_mm': [float(X[sub, 0].min()*1e3), float(X[sub, 0].max()*1e3)],
        'substrate_digit_final_z_median_mm': float(np.median(Xf[sub, 2])*1e3)}
    out['max_tool_penetration_mm'] = float(data['step_stats'][:, 3].max()*1e3)
    return out


def bumps3(data):
    X, Xf, codes = data['reference_local'].astype(float), data['final_positions'].astype(float), data['final_codes']
    masses = data['mass_kg']
    count, labels = mechanical_components_packed(data['stencil_nodes'], data['stencil_weights'], codes, 3,
                                                 gradients=data['stencil_gradients'])
    fractions = np.bincount(labels, weights=masses)/masses.sum()
    above = X[:, 2] > .012
    out = {'components': int(count), 'mass_fractions': np.sort(fractions)[::-1][:6].round(4).tolist(), 'bodies': []}
    for comp in np.argsort(fractions)[::-1][:6]:
        member = labels == comp
        foreign = member & ~above
        body = {'particles': int(member.sum()), 'bump_particles': int(np.count_nonzero(member & above)),
                'non_bump_particles': int(foreign.sum())}
        if 0 < foreign.sum() < 500:
            body['non_bump_reference_z_mm'] = sorted({round(float(z)*1e3, 1) for z in X[foreign, 2]})
            body['non_bump_codes'] = {int(a): int(b) for a, b in zip(*np.unique(codes[foreign], return_counts=True))}
            body['non_bump_carry_median_mm'] = float(np.median(np.linalg.norm(Xf[foreign]-X[foreign], axis=1))*1e3)
            body['assignment'] = {}
            for k in range(3):
                assigned = foreign & (data['diag_particle_assign_substep'][:, k] >= 0)
                if assigned.any():
                    body['assignment'][str(k)] = {
                        'count': int(assigned.sum()),
                        'digits': {int(a): int(b) for a, b in zip(*np.unique(cd.digits_of(codes[assigned], k, 'packed2'),
                                                                             return_counts=True))},
                        's_median_mm': float(np.median(data['diag_particle_assign_s_m'][assigned, k])*1e3),
                        'f_median_mm': float(np.median(data['diag_particle_assign_f_m'][assigned, k])*1e3),
                        'lift_at_assignment_median_mm': float(np.median(data['diag_particle_assign_x_m'][assigned, k, 2]
                                                                        - X[assigned, 2])*1e3),
                        'lead_rule': int(np.count_nonzero(data['diag_particle_assign_flags'][assigned, k] & 1))}
        out['bodies'].append(body)
    bump_codes = {}
    for i, bx in enumerate((-.032, 0., .032)):
        mask = above & (np.abs(X[:, 0]-bx) < .012)
        values, counts = np.unique(codes[mask], return_counts=True)
        bump_codes[f'bump{i}'] = {int(v): int(c) for v, c in zip(values, counts) if c >= 10}
    out['bump_codes_ge_10'] = bump_codes
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('kind', choices=('trim', 'bumps3'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    data = np.load(args.run/'wp12_phase4a_diag.npz')
    report = trim(data) if args.kind == 'trim' else bumps3(data)
    n = 1
    while (args.run/f'phase4a_contact_{n:03d}.json').exists():
        n += 1
    (args.run/f'phase4a_contact_{n:03d}.json').write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main()
