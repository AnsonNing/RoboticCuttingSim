#!/usr/bin/env python3
"""WP-12 Phase 4a: read-only root-cause analysis of one ``run_wp12_phase4a_diagnose.py`` run (CPU, NumPy).

For every cut digit used in the run it reports, from ``wp12_phase4a_diag.npz``:

* stroke timeline (armed / stroke-on / engaged substeps, plow fraction, motion-gate fraction, the off-plane angle
  of the segment motion, cutting work done while the gate was closed, contact particles);
* the final mechanical components (actual 8-node stencil) and the direct code-0 bridges of the digit;
* the unassigned particles within 12 mm of the digit's final swept plane (s along the swept normal of the last
  armed pose, f along its advance), and for both groups the decisive ``cut_diagnostics`` class
  (``UNASSIGNED_CLASSES``), their reference layers, s / f ranges and whether they were ahead of the final edge
  in their reference position already or only after moving (sag / carry);
* swept-normal conditioning flags of the new digits (task item (e)).

Writes ``<run>/phase4a_analysis_NNN.json`` (never overwrites) and prints a short summary.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from robotic_cutting_sim import cut_diagnostics as cd  # noqa: E402
from robotic_cutting_sim.cut_topology import mechanical_components, mechanical_components_packed  # noqa: E402


def _new_path(run, stem):
    k = 1
    while (run/f'{stem}_{k:03d}.json').exists():
        k += 1
    return run/f'{stem}_{k:03d}.json'


def _hist(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return {f'{a:g}..{b:g}': int(c) for a, b, c in zip(edges[:-1], edges[1:], counts) if c}


def analyze(run):
    data = np.load(run/'wp12_phase4a_diag.npz')
    meta = json.loads((run/'wp12_phase4a_diag.json').read_text())
    result = json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {}
    m = meta['meta']
    encoding, K = m['code_encoding'], int(m['max_cuts'])
    X = data['reference_local'].astype(np.float64)
    Xf = data['final_positions'].astype(np.float64)
    codes = data['final_codes']
    masses = data['mass_kg']
    nodes, weights, gradients = data['stencil_nodes'], data['stencil_weights'], data['stencil_gradients']
    if encoding == 'packed2':
        count, labels = mechanical_components_packed(nodes, weights, codes, K, gradients=gradients)
    else:
        count, labels = mechanical_components(nodes, weights, codes, cuts=K, gradients=gradients)
    fractions = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    table = {k[len('diag_particle_'):]: data[k] for k in data.files if k.startswith('diag_particle_')}
    rows = data['diag_substep_rows']
    segs = data['diag_segment_rows']
    sc, gc = cd.SUBSTEP_COLUMNS.index, cd.SEGMENT_COLUMNS.index
    step_cut = data['step_cut']
    out = {'run': str(run), 'task_id': m.get('task_id'), 'solver_module': m['solver_module'],
           'cutting_law': m['cutting_law'], 'code_encoding': encoding, 'max_cuts': K, 'particles': int(len(X)),
           'counterfactual': meta.get('counterfactual'), 'counterfactual_label': meta.get('counterfactual_label'),
           'wrapped_exit': meta['wrapped_exit'], 'git_sha': meta.get('git_sha'), 'wall_s': meta.get('wall_s'),
           'class': result.get('class'), 'class_reason': result.get('class_reason'),
           'evaluation_verdict': (result.get('evaluation') or {}).get('verdict'),
           'mechanical_components': int(count), 'component_mass_fractions': fractions[:6].round(4).tolist(),
           'labelled_any_digit': int(np.count_nonzero(codes)), 'digits': {}}
    half = float(m['blade_half_width_m'])
    for k in range(K):
        armed_steps = np.flatnonzero(step_cut == k)
        if not len(armed_steps):
            continue
        digits = cd.digits_of(codes, k, encoding)
        krows = rows[rows[:, sc('cut')] == k]
        ksegs = segs[armed_steps] if len(segs) >= armed_steps.max()+1 else segs[:0]
        armed_segment = ksegs[:, :, gc('armed_substeps')].sum()
        motion = np.stack([ksegs[:, :, gc(c)].sum() for c in ('motion_d_sum_m_s', 'motion_e_sum_m_s',
                                                               'motion_n_sum_m_s')])
        info = {'armed_control_steps': int(len(armed_steps)), 'armed_substeps': int(len(krows)),
                'stroke_on_substeps': int(np.count_nonzero(krows[:, sc('stroke')] == k+1)),
                'engaged_substeps': int(np.count_nonzero(krows[:, sc('engaged')] > 0)),
                'seg_ok_fraction': float(ksegs[:, :, gc('seg_ok_substeps')].sum()/max(armed_segment, 1)),
                'seg_plow_fraction': float(ksegs[:, :, gc('seg_plow_substeps')].sum()/max(armed_segment, 1)),
                'mean_segment_motion_offplane_deg': float(np.degrees(np.arctan2(abs(motion[2]), np.hypot(*motion[:2])))),
                'contact_particle_substeps': float(ksegs[:, :, gc('contact_particle_substeps')].sum()),
                'cutting_work_J': float(ksegs[:, :, gc('cutting_work_sum_J')].sum()),
                'work_with_gate_closed_particle_substeps': float(ksegs[:, :, gc('work_with_gate_closed_particle_substeps')].sum()),
                'band_new': int(krows[:, sc('band_new')].sum()), 'lead_new': int(krows[:, sc('lead_new')].sum()),
                'digit_counts': {int(a): int(b) for a, b in zip(*np.unique(digits, return_counts=True))}}
        # Final armed pose of this digit (solver-local), its swept plane and advance.
        last = armed_steps[-1]
        pose = data['step_pose_end_local'][last].astype(np.float64)
        start = data['step_pose_start_local'][last].astype(np.float64)
        R = Rotation.from_quat(pose[3:]).as_matrix()
        d, e, n = R[:, 0], R[:, 1], R[:, 2]
        motion_vec = (pose[:3]-start[:3])
        swept = np.cross(motion_vec, e)
        if np.linalg.norm(swept) > 1e-9*max(np.linalg.norm(motion_vec), 1e-12):
            swept /= np.linalg.norm(swept)
            swept *= np.sign(swept@n) or 1.
        else:
            swept = n
        o = pose[:3]
        s_ref, s_fin = (X-o)@swept, (Xf-o)@swept
        f_ref, f_fin = (X-o)@d, (Xf-o)@d
        l_fin = (Xf-o)@e
        info['final_edge_local_mm'] = (o*1e3).round(3).tolist()
        info['final_frame'] = {'advance_d': d.round(5).tolist(), 'edge_e': e.round(5).tolist(),
                               'blade_normal_n': n.round(5).tolist(), 'swept_normal': swept.round(5).tolist()}
        bridges = cd.direct_bridges(nodes, weights, gradients, digits)
        info['direct_code0_bridges'] = int(len(bridges))
        near = (digits == 0) & (np.abs(s_fin) <= .012) & (np.abs(l_fin) <= half)
        for name, mask in (('unassigned_within_12mm', near), ('bridges', np.isin(np.arange(len(X)), bridges))):
            if not mask.any():
                info[name] = {'count': 0}
                continue
            idx, classes = cd.classify_unassigned(table, k, mask)
            group = {'count': int(mask.sum()),
                     'by_class': {c: int(np.count_nonzero(classes == j)) for j, c in enumerate(cd.UNASSIGNED_CLASSES)
                                  if np.count_nonzero(classes == j)},
                     'reference_z_layers_mm': {str(round(float(z)*1e3, 1)): int(c)
                                               for z, c in zip(*np.unique(X[mask, 2], return_counts=True))},
                     's_final_mm': _hist(s_fin[mask]*1e3, np.arange(-24, 25, 2.)),
                     'f_final_mm': {'min': float(f_fin[mask].min()*1e3), 'max': float(f_fin[mask].max()*1e3)},
                     'ahead_of_final_edge': {'reference_f_gt_0': int(np.count_nonzero(f_ref[mask] > 0)),
                                             'final_f_gt_0': int(np.count_nonzero(f_fin[mask] > 0))},
                     'displacement_mm': {'median_d': float(np.median((Xf-X)[mask]@d)*1e3),
                                         'median_n': float(np.median((Xf-X)[mask]@swept)*1e3),
                                         'median_z': float(np.median((Xf-X)[mask, 2])*1e3),
                                         'max_norm': float(np.linalg.norm(Xf-X, axis=1)[mask].max()*1e3)},
                     'min_f_stroke_mm': {'min': float(table['min_f_stroke_m'][mask, k].min()*1e3),
                                         'median': float(np.median(table['min_f_stroke_m'][mask, k])*1e3)},
                     'lead_rule_blocked_by_plow_gate_particles': int(np.count_nonzero(table['lead_plow_blocked'][mask, k])),
                     'first_fail_totals': {r: int(table['first_fail'][mask, k, j].sum()) for j, r in enumerate(cd.REASONS)},
                     'only_fail_totals': {r: int(table['only_fail'][mask, k, j].sum()) for j, r in enumerate(cd.REASONS)},
                     'sample_reference_mm': np.round(X[mask][:8]*1e3, 2).tolist()}
            info[name] = group
        # Relabel tests (diagnostic only, never a result): give a set of digit-0 particles the side of the final
        # swept plane (current positions) and recount the mechanical components.
        def relabel(mask):
            new = codes.astype(np.int64).copy()
            side = np.where(s_fin[mask] >= 0, 2, 1)
            if encoding == 'packed2':
                new[mask] |= side << (2*k)
                c, _ = mechanical_components_packed(nodes, weights, new.astype(codes.dtype), K, gradients=gradients)
            else:
                new[mask] += side*3**k
                c, _ = mechanical_components(nodes, weights, new.astype(codes.dtype), cuts=K, gradients=gradients)
            return int(c)
        tests = {}
        if len(bridges):
            tests['direct_bridges_only'] = {'relabelled': int(len(bridges)),
                                            'components': relabel(np.isin(np.arange(len(X)), bridges))}
        if near.any():
            tests['all_unassigned_within_12mm'] = {'relabelled': int(near.sum()), 'components': relabel(near)}
            lead_ok = near & (table['lead_plow_blocked'][:, k] > 0)
            tests['lead_rule_blocked_by_plow_gate'] = {'relabelled': int(lead_ok.sum()), 'components': relabel(lead_ok)}
        info['relabel_tests_diagnostic_only'] = tests
        # Mislabels: labelled against the sign of the final swept plane by more than 2 particle spacings.
        lab = digits != 0
        expected = np.where(s_ref >= 0, 2, 1)
        info['labels_against_final_plane_sign_beyond_2_spacings'] = int(np.count_nonzero(
            lab & (digits != expected) & (np.abs(s_ref) > 2*m['spacing_m'])))
        # Swept-normal conditioning of this digit's new labels.
        assigned = table['assign_substep'][:, k] >= 0
        flags = table['assign_flags'][assigned, k]
        info['new_digit_flags'] = {name: int(np.count_nonzero(flags & bit)) for name, bit in cd.ASSIGN_FLAGS.items()}
        info['ill_conditioned_segment_substeps'] = {
            'float': float(ksegs[:, :, gc('swept_ill_float_substeps')].sum()),
            'geom': float(ksegs[:, :, gc('swept_ill_geom_substeps')].sum())}
        a_s = table['assign_s_m'][assigned, k]
        info['assign_s_abs_mm'] = {'p50': float(np.median(np.abs(a_s))*1e3), 'max': float(np.abs(a_s).max()*1e3)} if assigned.any() else None
        out['digits'][str(k)] = info
    stats = data['step_stats']
    out['max_tool_penetration_mm'] = float(stats[:, 3].max()*1e3)
    out['max_fixture_penetration_mm'] = float(stats[:, 0].max()*1e3)
    out['flipped_by_float_fallback_total'] = int(rows[:, sc('new_digits_flipped_by_float_fallback')].sum())
    out['flipped_by_geom_fallback_total'] = int(rows[:, sc('new_digits_flipped_by_geom_fallback')].sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for run in args.runs:
        report = analyze(run)
        path = _new_path(run, 'phase4a_analysis')
        path.write_text(json.dumps(report, indent=1))
        print(f'{run}: components {report["mechanical_components"]} {report["class"]} -> {path.name}')
        for k, info in report['digits'].items():
            print(f'  digit {k}: stroke {info["stroke_on_substeps"]}/{info["armed_substeps"]} ok {info["seg_ok_fraction"]:.3f} '
                  f'plow {info["seg_plow_fraction"]:.3f} bridges {info["direct_code0_bridges"]} '
                  f'unassigned12 {info["unassigned_within_12mm"]["count"]} '
                  f'classes {info["unassigned_within_12mm"].get("by_class")} flags {info["new_digit_flags"]}')


if __name__ == '__main__':
    main()
