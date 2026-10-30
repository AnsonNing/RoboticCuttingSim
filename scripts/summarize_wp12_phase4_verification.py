#!/usr/bin/env python3
"""Read-only summary and verdicts of configs/wp12/phase4_verification_protocol_v1.json (rows V1.1-V4.6).

Reads the completed rows under ``--runs`` (artifacts/wp12/phase4_verify_NNN), computes the READ_ONLY rows (V1.1,
V2.1, V3.1, V4.1) and the STATIC relabel (V3.3) from the saved states of ``resolution_002``, ``phase4_e1_001`` and
``phase3_surface_001`` (never modified), applies the declared case and hypothesis rules and writes the evidence JSON.
No run is started, nothing in any run directory is written except ``--cache`` (a new file under --runs).
COUNTERFACTUAL rows stay labelled COUNTERFACTUAL_NOT_A_GATE in every output.

  .venv/bin/python scripts/summarize_wp12_phase4_verification.py --runs artifacts/wp12/phase4_verify_001 \\
      --out docs/evidence/wp12/phase4_verification_001.json --markdown artifacts/wp12/phase4_verify_001/tables.md
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PROTOCOL = ROOT/'configs/wp12/phase4_verification_protocol_v1.json'
RES = ROOT/'artifacts/wp12/resolution_002'
E1 = ROOT/'artifacts/wp12/phase4_e1_001'
P3 = ROOT/'artifacts/wp12/phase3_surface_001'
FIRST, FAR = '0-1s', '>6s'


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def combine(cases):
    """Hypothesis verdict from case verdicts (protocol verdict_rules)."""
    vals = [c['verdict'] for c in cases if c['verdict'] not in ('NOT_EVALUABLE', 'NOT_RUN', 'REPORTED')]
    counts = {v: sum(1 for c in cases if c['verdict'] == v) for v in
              ('CONFIRMED', 'REFUTED', 'INCONCLUSIVE', 'NOT_EVALUABLE', 'NOT_RUN')}
    if not vals:
        verdict = 'INCONCLUSIVE'
    elif all(v == 'CONFIRMED' for v in vals):
        verdict = 'CONFIRMED'
    elif all(v == 'REFUTED' for v in vals):
        verdict = 'REFUTED'
    else:
        verdict = 'INCONCLUSIVE'
    if counts['NOT_RUN'] and verdict == 'CONFIRMED':
        verdict = 'INCONCLUSIVE'
    return verdict, counts


def case(name, verdict, **kw):
    return {'case': name, 'verdict': verdict, **kw}


# ----------------------------------------------------------------------------- inputs

def load_runs(runs):
    out = {}
    for path in sorted(Path(runs).glob('*/result.json')):
        d = json.loads(path.read_text())
        out[d['row']] = d
    return out


def ok(runs, rid):
    d = runs.get(rid)
    return d if d is not None and d.get('status') == 'COMPLETED' else None


def first_band(d, state='links_FINAL'):
    return (d['analysis'].get(state) or {}).get(FIRST) if d else None


def force(d):
    return (d or {}).get('driver_result', {}).get('contact_force', {}).get('steady_mean_abs_N')


def res_locus(case_name, geometry):
    from robotic_cutting_sim import root_cause_diagnostics as rd
    if case_name == 'TRIM':
        z = float(geometry['plane_z_m'])
        return lambda P: np.abs(np.asarray(P)[:, 2]-z)
    planes = [(p['origin'], p['normal']) for p in geometry['planes']]
    return lambda P: rd.plane_distance(P, planes)


def readonly_resolution(cache):
    """V1.1 bands, V2.1 TRIM chip links, V3.1/V3.3 OBL20 split and STATIC relabel, V4.1 forces on resolution_002."""
    from robotic_cutting_sim import root_cause_diagnostics as rd
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for d in sorted(RES.iterdir()):
        if not (d/'result.json').exists():
            continue
        r = json.loads((d/'result.json').read_text())
        if r.get('status') != 'COMPLETED':
            out[d.name] = {'status': r.get('status')}
            continue
        row, geo = r['row'], r['geometry']
        z = np.load(d/'state.npz')
        X = z['reference'].astype(np.float64)
        lat = z['lattice']
        s = row['spacing_m']
        locus = res_locus(row['case'], geo)
        entry = {'case': row['case'], 'spacing_m': s, 'grid_dx_m': row['grid_dx_m'], 'dt_s': row['dt_s'],
                 'variant': row.get('variant'), 'steady_mean_abs_N': r['contact_force']['steady_mean_abs_N'],
                 'components': r['analysis']['FINAL']['mechanical_components'],
                 'separation_criterion': r['analysis']['FINAL'].get('separation_criterion'),
                 'removal_fraction': r['analysis']['FINAL'].get('removal_fraction'), 'states': {}}
        for name in sorted({k[:-len('_positions')] for k in z.files if k.endswith('_positions')}):
            b = rd.link_consistency(X, lat, z[f'{name}_positions'], z[f'{name}_F'], z[f'{name}_codes'], s, locus)
            entry['states'][name] = {k: v for k, v in b.items()}
        if row['case'] == 'TRIM':
            entry['chip_links'] = rd.chip_links(X, lat, z['FINAL_positions'], z['FINAL_F'], z['FINAL_codes'], s,
                                                X[:, 2] > geo['plane_z_m'])
        if row['case'] == 'OBL20':
            plane = geo['planes'][0]
            nodes, weights = rd.reference_stencil(z['reference'], row['grid_dx_m'])
            n = np.asarray(plane['normal'])
            s_ref = (X-np.asarray(plane['origin']))@n
            codes = z['FINAL_codes'].astype(np.int64)
            mis = rd.mis_sided_bridges(codes, nodes, weights, s_ref)
            mis.pop('bridge_indices')
            mis['recorded_bridges'] = r['analysis']['FINAL']['direct_code0_bridges']
            entry['p3_split'] = mis
            new, lab = rd.visibility_relabel(X, codes, nodes, weights, plane['origin'], n, geo['edge_stop_z_m'],
                                             near_m=2*row['grid_dx_m'])
            k1, l1 = mechanical_components_packed(nodes, weights, new)
            digits = new & 3
            from robotic_cutting_sim import cut_diagnostics as cd
            br = cd.direct_bridges(nodes, weights, np.zeros(nodes.shape+(3,)), digits)
            lab.update(predicted_components=int(k1), predicted_bridges=int(len(br)),
                       law8_components=entry['components'])
            entry['visibility_static'] = lab
        out[d.name] = entry
        print('readonly', d.name, flush=True)
    cache.write_text(json.dumps(jsonable(out)))
    return out


def readonly_e1(cache):
    """V3.1 split and V3.3 STATIC relabel on every phase4_e1_001 row."""
    from robotic_cutting_sim import cut_diagnostics as cd
    from robotic_cutting_sim import root_cause_diagnostics as rd
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for d in sorted(E1.iterdir()):
        if not (d/'snapshots.npz').exists():
            continue
        r = json.loads((d/'result.json').read_text())
        spec = r['row_spec']
        z = np.load(d/'snapshots.npz')
        X = z['reference_local'].astype(np.float64)
        codes = z['final_codes'].astype(np.int64)
        nodes, weights, grads = z['stencil_nodes'], z['stencil_weights'], z['stencil_gradients']
        tau = np.radians(spec['tau_deg'])
        n = np.array([np.cos(tau), 0., -np.sin(tau)])
        o = np.array([spec['plane_x0_mm']*1e-3, 0., 0.])
        s_ref = (X-o)@n
        mis = rd.mis_sided_bridges(codes, nodes, weights, s_ref)
        mis.pop('bridge_indices')
        h = spec['edge_end_z_mm']*1e-3
        new, lab = rd.visibility_relabel(X, codes, nodes, weights, o, n, h)
        k1, l1 = mechanical_components_packed(nodes, weights, new, 1, gradients=grads)
        mass = z['mass_kg']
        fr = np.sort(np.bincount(l1, weights=mass))[::-1]/mass.sum()
        br = cd.direct_bridges(nodes, weights, grads, new & 3)
        pieces = int(np.count_nonzero(fr >= .01))
        lab.update(predicted_components=int(k1), predicted_bridges=int(len(br)), predicted_mass_fractions=fr[:3].tolist(),
                   predicted_outcome='SEPARATED' if k1 >= 2 and pieces >= 2 and len(br) == 0 else
                   'CONNECTED' if k1 == 1 else 'OTHER')
        law8 = r['analysis']['law8_final']
        out[d.name] = {'spec': spec, 'row_status': r.get('row_status'), 'law8': {k: law8[k] for k in
                       ('components', 'direct_bridges', 'outcome', 'labelled')}, 'p3_split': mis, 'visibility_static': lab}
        print('readonly E1', d.name, flush=True)
    cache.write_text(json.dumps(jsonable(out)))
    return out


def readonly_s7():
    """V2.1 on the Phase 3 S7 final states (r0, r1)."""
    from robotic_cutting_sim import root_cause_diagnostics as rd
    out = {}
    for rep in ('r0', 'r1'):
        f = P3/f'S7_{rep}_states.npz'
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        X = z['reference'].astype(np.float64)
        out[f'phase3_surface_001/S7_{rep}'] = rd.chip_links(X, z['lattice'], z['positions'][-1], z['F'][-1], z['codes'][-1],
                                                           float(z['spacing_m']), X[:, 2] > .010)
    return out


# ----------------------------------------------------------------------------- hypotheses

def v11(P, ro):
    h = P['hypotheses']['V1.1']
    cases, repro = [], []
    for rid, e in sorted(ro.items()):
        for state, b in (e.get('states') or {}).items():
            ratio = b.get('first_to_far_p99_ratio')
            v = 'NOT_EVALUABLE' if ratio is None else 'CONFIRMED' if ratio >= 5 else 'REFUTED' if ratio < 2 else 'INCONCLUSIVE'
            cases.append(case(f'{rid}/{state}', v, ratio=ratio, first_p50=b[FIRST]['p50'], first_p99=b[FIRST]['p99'],
                              far_p99=b[FAR]['p99']))
    for rid, (p50, p99) in h['declared_values_first_band'].items():
        b = ((ro.get(rid) or {}).get('states') or {}).get('FINAL')
        if b is None:
            repro.append({'row': rid, 'reproduced': None})
            continue
        good = (p50 is None or abs(b[FIRST]['p50']-p50) <= .005) and abs(b[FIRST]['p99']-p99) <= .02
        repro.append({'row': rid, 'declared': [p50, p99], 'measured': [b[FIRST]['p50'], b[FIRST]['p99']], 'reproduced': good})
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'reproduction_of_section_2_1': repro,
            'ratio_min': min((c['ratio'] for c in cases if c['ratio'] is not None), default=None),
            'cases_not_confirmed': [c for c in cases if c['verdict'] != 'CONFIRMED']}


def v12(P, runs):
    cases = []
    for rid in P['hypotheses']['V1.2']['rows']:
        d = ok(runs, rid)
        if d is None:
            cases.append(case(rid, 'NOT_RUN' if rid not in runs else 'NOT_EVALUABLE', status=(runs.get(rid) or {}).get('status')))
            continue
        L = d['analysis']['links_FINAL']
        e1, e4 = L[FIRST]['identity']['explained_fraction'], L[FAR]['identity']['explained_fraction']
        v = 'CONFIRMED' if e1 >= .95 and e4 >= .999 else 'REFUTED' if e1 < .8 else 'INCONCLUSIVE'
        cases.append(case(rid, v, e1=e1, e2=L['1-3s']['identity']['explained_fraction'],
                          e3=L['3-6s']['identity']['explained_fraction'], e4=e4,
                          residual_p99_first=L[FIRST]['identity']['residual_p99'], first_p50=L[FIRST]['p50'],
                          first_p99=L[FIRST]['p99'], identity_position_residual_m=d['analysis']['drift_identity_position_residual_m'],
                          identity_integrity_le_1e5_m=d['analysis']['drift_identity_position_residual_m'] <= 1e-5,
                          far_residual_p99=L[FAR]['identity']['residual_p99'], far_p99=L[FAR]['p99']))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases}


def v13(P, runs, ro):
    cases = []
    base = {'CUT': 'V12_CUT_dt0100', 'TRIM': 'V12_TRIM_dt0100', 'OBL20': 'V12_OBL20_dt0100', 'S7': 'B_S7_dt0100'}
    sweep = {}
    for c, b in base.items():
        db = ok(runs, b)
        row = {}
        for a in ('025', '050', '100'):
            d = ok(runs, f'V13_{c}_pic{a}')
            row[a] = first_band(d)['p99'] if d else None
        row['005'] = first_band(db)['p99'] if db else None
        sweep[c] = row
        d1 = ok(runs, f'V13_{c}_pic100')
        if db is None or d1 is None:
            cases.append(case(f'(a) {c} alpha 1', 'NOT_RUN' if d1 is None and f'V13_{c}_pic100' not in runs else 'NOT_EVALUABLE'))
            continue
        p0, p1 = first_band(db)['p99'], first_band(d1)['p99']
        r = p1/p0
        v = 'CONFIRMED' if .2 <= r <= .5 and p1 > .05 else 'REFUTED' if r >= .8 else 'INCONCLUSIVE'
        cases.append(case(f'(a) {c} alpha 1', v, label='COUNTERFACTUAL_NOT_A_GATE', p99_alpha005=p0, p99_alpha1=p1, ratio=r,
                          p50_alpha005=first_band(db)['p50'], p50_alpha1=first_band(d1)['p50']))
    for c in ('CUT', 'TRIM', 'OBL20'):
        d01 = ok(runs, f'V12_{c}_dt0100')
        for dt, tag in ((.05, '0050'), (.025, '0025')):
            if dt == .05:
                df = ok(runs, f'V12_{c}_dt0050')
                formal = first_band(df)['p99'] if df else None
                src = f'V12_{c}_dt0050'
            else:
                e = ro.get(f'{c}_s100_g4s_dt0025') or {}
                formal = e.get('states', {}).get('FINAL', {}).get(FIRST, {}).get('p99')
                src = f'resolution_002/{c}_s100_g4s_dt0025'
            dx = ok(runs, f'V13_{c}_dt{tag}_pictime')
            if d01 is None or dx is None or formal is None:
                cases.append(case(f'(b) {c} {dt} ms', 'NOT_RUN' if f'V13_{c}_dt{tag}_pictime' not in runs else 'NOT_EVALUABLE'))
                continue
            p01, pfx = first_band(d01)['p99'], first_band(dx)['p99']
            tf, tx = p01-formal, p01-pfx
            v = 'CONFIRMED' if abs(tx) <= .5*abs(tf) else 'REFUTED' if abs(tx) > abs(tf) else 'INCONCLUSIVE'
            cases.append(case(f'(b) {c} {dt} ms', v, label='COUNTERFACTUAL_NOT_A_GATE', p99_0p1ms=p01, p99_formal=formal,
                              formal_source=src, p99_fixed_alpha_per_time=pfx, trend_formal=tf, trend_fixed=tx,
                              alpha=dx['effective_pic']))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases, 'alpha_sweep_first_band_p99': sweep}


def v14(P, runs, ro):
    cases = []
    pairs = [(f'V14_{c}_s100_grid', f'V12_{c}_dt0100', None) for c in ('CUT', 'TRIM', 'OBL20', 'MULTI2')]
    pairs += [('V14_S7_grid', 'B_S7_dt0100', None), ('V14_G3b_grid', 'B_G3b_dt0100', None)]
    pairs += [(f'V14_{c}_s050_grid', None, f'{c}_s050_g4s_dt0050') for c in ('CUT', 'TRIM', 'OBL20', 'MULTI2')]
    for cf, base, rbase in pairs:
        dc = ok(runs, cf)
        if base:
            db = ok(runs, base)
            fb = first_band(db) if db else None
            topo_b = db['analysis']['topology_FINAL'] if db else None
        else:
            e = ro.get(rbase) or {}
            fb = e.get('states', {}).get('FINAL', {}).get(FIRST)
            topo_b = {'mechanical_components': e.get('components'), 'separation_criterion': e.get('separation_criterion'),
                      'removal_fraction': e.get('removal_fraction')} if e else None
        if dc is None or fb is None:
            cases.append(case(cf, 'NOT_RUN' if cf not in runs else 'NOT_EVALUABLE', status=(runs.get(cf) or {}).get('status')))
            continue
        fc = first_band(dc)
        topo_c = dc['analysis']['topology_FINAL']
        q50, q99 = fb['p50']/fc['p50'], fb['p99']/fc['p99']
        lost = (topo_c.get('mechanical_components', 0) < (topo_b.get('mechanical_components') or 0)
                or (topo_b.get('separation_criterion') is True and topo_c.get('separation_criterion') is not True))
        v = 'CONFIRMED' if q50 >= 2 and q99 >= 2 and not lost else 'REFUTED' if q50 < 2 or q99 < 2 else 'INCONCLUSIVE'
        L = dc['analysis']['links_FINAL']
        hist = [L[k].get('same_assignment_history', {}).get('p99') for k in ('1-3s', '3-6s', '>6s')]
        cases.append(case(cf, v, label='COUNTERFACTUAL_NOT_A_GATE', baseline=base or f'resolution_002/{rbase}',
                          p50_base=fb['p50'], p99_base=fb['p99'], p50_grid=fc['p50'], p99_grid=fc['p99'], q50=q50, q99=q99,
                          q_ge_3=bool(q50 >= 3 and q99 >= 3), lost_separation=bool(lost),
                          first_band_same_history_p99=L[FIRST].get('same_assignment_history', {}).get('p99'),
                          first_band_different_history_p99=L[FIRST].get('different_assignment_history', {}).get('p99'),
                          beyond_1s_same_history_p99=hist, topology_base=topo_b, topology_grid=topo_c,
                          force_base=force(ok(runs, base)) if base else (ro.get(rbase) or {}).get('steady_mean_abs_N'),
                          force_grid=force(dc)))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases}


def v21(P, ro, s7):
    cases = []
    items = [(f'resolution_002/{k}', e.get('chip_links')) for k, e in sorted(ro.items()) if e.get('case') == 'TRIM']
    items += sorted(s7.items())
    for name, c in items:
        if not c or not c.get('long_links'):
            cases.append(case(name, 'NOT_EVALUABLE', long_links=(c or {}).get('long_links'), note='no long links'))
            continue
        f = c['long_with_F_length_below_rule']
        cases.append(case(name, 'CONFIRMED' if f >= .95 else 'REFUTED' if f < .5 else 'INCONCLUSIVE', f=f,
                          long_links=c['long_links'], links=c['links'],
                          stretch=c['endpoint_max_principal_stretch'], det_F_p5=c['det_F_p5'],
                          length_s=c['long_link_length_s']))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases}


def v22(P, runs):
    cases = []
    for rid in P['hypotheses']['V2.2']['rows']:
        d = ok(runs, rid)
        if d is None or 'chip_energy' not in d['analysis']:
            cases.append(case(rid, 'NOT_RUN' if rid not in runs else 'NOT_EVALUABLE'))
            continue
        e = d['analysis']['chip_energy']
        g = e['griffith']
        cases.append(case(rid, 'CONFIRMED' if g['thin_particles_with_psi_t_ge_gc'] == 0 else 'REFUTED', griffith=g,
                          chip_psi_dev=e['chip']['psi_dev'], chip_psi_tensile_history_max=e['chip'].get('psi_tensile_history_max'),
                          chip_psi_total_history_max=e['chip'].get('psi_total_history_max'),
                          chip_plastic_strain=e['chip']['plastic_strain'], chip_von_mises=e['chip']['von_mises_kirchhoff_pa'],
                          chip_pressure=e['chip']['pressure_pa']))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases}


def v23(P, runs, ro):
    cases = []
    for cf, base in P['hypotheses']['V2.3']['pairs']:
        dc = ok(runs, cf)
        if base.startswith('resolution_002/'):
            cb = (ro.get(base.split('/', 1)[1]) or {}).get('chip_links')
        else:
            db = ok(runs, base)
            cb = db['analysis'].get('chip_links') if db else None
        cc = dc['analysis'].get('chip_links') if dc else None
        if cc is None or cb is None:
            cases.append(case(cf, 'NOT_RUN' if cf not in runs else 'NOT_EVALUABLE'))
            continue
        fb, fc = cb['long_fraction'], cc['long_fraction']
        if cc['long_links'] == 0:
            ratio = None
            v = 'CONFIRMED' if cb['long_links'] >= 5 else 'NOT_EVALUABLE'
        else:
            ratio = fb/fc
            v = 'CONFIRMED' if ratio >= 5 else 'REFUTED' if ratio < 2 else 'INCONCLUSIVE'
        cases.append(case(cf, v, label='COUNTERFACTUAL_NOT_A_GATE', baseline=base, long_links_base=cb['long_links'],
                          long_links_grid=cc['long_links'], long_fraction_base=fb, long_fraction_grid=fc, ratio=ratio))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases}


def v31(P, ro, e1):
    cases = []
    declared = P['hypotheses']['V3.1']['declared_counts']
    for rid, (nb, nm) in declared.items():
        if rid == 'OBL20_*_g4s_*':
            for k, e in sorted(ro.items()):
                if e.get('case') == 'OBL20' and '_g4s_' in k and 'p3_split' in e:
                    m = e['p3_split']['need_mis_sided_label']
                    cases.append(case(f'resolution_002/{k}', 'CONFIRMED' if m == 0 else 'REFUTED', bridges=e['p3_split']['bridges'],
                                      recorded_bridges=e['p3_split']['recorded_bridges'], need_mis_sided=m))
            continue
        e = (ro.get(rid) or {}).get('p3_split') if rid.startswith('OBL20') else (e1.get(rid) or {}).get('p3_split')
        if e is None:
            cases.append(case(rid, 'NOT_EVALUABLE'))
            continue
        good = (nb is None or e['bridges'] == nb) and e['need_mis_sided_label'] == nm
        cases.append(case(rid, 'CONFIRMED' if good else 'REFUTED', declared=[nb, nm],
                          measured=[e['bridges'], e['need_mis_sided_label']]))
    verdict, counts = combine(cases)
    table = {k: {'law8': v['law8']['outcome'], 'bridges': v['p3_split']['bridges'], 'p3b': v['p3_split']['need_mis_sided_label'],
                 'p3a': v['p3_split']['in_core_wedge'], 'mis_sided_labels': v['p3_split']['mis_sided_labelled_particles']}
             for k, v in sorted(e1.items())}
    return {'verdict': verdict, 'counts': counts, 'cases': cases, 'e1_table': table}


def e1_outcome(d):
    return d['driver_result']['analysis']['law8_final']


def v32(P, runs, e1):
    base = {t: e1.get(f'A_p{t}') for t in (10, 20, 30)}
    rows, changed, shifted, missing = [], False, False, []
    for x0 in ('05', '10', '20'):
        for t in (10, 20, 30):
            rid = f'E1b_x{x0}_p{t}'
            d = ok(runs, rid)
            b = base[t]
            if d is None or b is None:
                missing.append(rid)
                continue
            law8 = e1_outcome(d)
            sp = d['analysis']['p3_split']
            share, bshare = sp['share_p3b'], b['p3_split']['share_p3b']
            out_change = law8['outcome'] != b['law8']['outcome']
            share_change = share is not None and bshare is not None and abs(share-bshare) >= .25
            changed |= out_change
            shifted |= share_change
            rows.append({'row': rid, 'plane_x0_mm': float(x0)/10, 'tau_deg': t, 'outcome': law8['outcome'],
                         'components': law8['components'], 'bridges': law8['direct_bridges'], 'p3b': sp['need_mis_sided_label'],
                         'share_p3b': share, 'x0_0_outcome': b['law8']['outcome'], 'x0_0_share_p3b': bshare,
                         'outcome_changed': out_change, 'share_changed': share_change})
    if missing and not (changed or shifted):
        verdict = 'INCONCLUSIVE'
    else:
        verdict = 'CONFIRMED' if (changed or shifted) else 'REFUTED'
    return {'verdict': verdict, 'rows': rows, 'missing': missing, 'any_outcome_changed': changed, 'any_share_changed': shifted}


def v33(P, runs, ro, e1):
    cases, refute = [], []
    for k, e in sorted(e1.items()):
        vs, spec = e['visibility_static'], e['spec']
        h, t = spec['edge_end_z_mm'], abs(spec['tau_deg'])
        if k in ('A_p10', 'A_m10', 'A_p30'):
            expect = 'CONNECTED'
        elif k.startswith('A_') and t in (15, 20, 25, 35):
            expect = 'SEPARATED'
        elif k.startswith('R_') and h >= .75:
            expect = 'CONNECTED'
        else:
            expect = None
        pred = vs['predicted_outcome']
        v = 'REPORTED' if expect is None else 'CONFIRMED' if pred == expect else 'REFUTED'
        if k.startswith('R_') and h >= .75 and pred == 'SEPARATED':
            refute.append(f'{k}: an h >= 0.75 mm ligament row separates under the relabel')
        if vs['labels'] > 1000:
            refute.append(f'{k}: {vs["labels"]} labels (> 1000)')
        cases.append(case(f'phase4_e1_001/{k}', v, expected=expect, predicted=pred, labels=vs['labels'],
                          labels_below_edge_stop=vs['labels_below_edge_stop'], law8=e['law8']['outcome'],
                          labels_le_300=vs['labels'] <= 300))
    for rid, d in sorted(runs.items()):
        if rid.startswith('E1b_') and d.get('status') == 'COMPLETED':
            vs = d['analysis']['visibility_static']
            if vs['labels'] > 1000:
                refute.append(f'{rid}: {vs["labels"]} labels (> 1000)')
            cases.append(case(rid, 'REPORTED', predicted_components=vs['predicted_components'], labels=vs['labels'],
                              law8=e1_outcome(d)['outcome']))
    for k, e in sorted(ro.items()):
        if 'visibility_static' in e:
            vs = e['visibility_static']
            if vs['labels'] > 1000:
                refute.append(f'resolution_002/{k}: {vs["labels"]} labels (> 1000)')
            cases.append(case(f'resolution_002/{k}', 'REPORTED', predicted_components=vs['predicted_components'],
                              predicted_bridges=vs['predicted_bridges'], labels=vs['labels'], law8_components=vs['law8_components']))
    scored = [c for c in cases if c['verdict'] in ('CONFIRMED', 'REFUTED')]
    static_consistent = all(c['verdict'] == 'CONFIRMED' for c in scored)
    verdict = 'REFUTED' if refute else 'INCONCLUSIVE'
    return {'verdict': verdict, 'static_part_consistent_with_prediction': static_consistent,
            'refute_triggers': refute, 'not_run': P['hypotheses']['V3.3']['not_run'],
            'counts': {v: sum(1 for c in cases if c['verdict'] == v) for v in ('CONFIRMED', 'REFUTED', 'REPORTED')},
            'cases': cases, 'label': 'STATIC shadow relabel (not a law, not a simulation)'}


def v34(P, runs, e1):
    rows = []
    for t in (10, 30):
        b = e1.get(f'A_p{t}')
        nb = b['p3_split']['need_mis_sided_label'] if b else None
        for var in ('dt0050', 'k040', 'grid'):
            rid = f'V34_A_p{t}_{var}'
            d = ok(runs, rid)
            if d is None:
                rows.append({'row': rid, 'status': 'NOT_RUN' if rid not in runs else 'NOT_EVALUABLE'})
                continue
            sp = d['analysis']['p3_split']
            n = sp['need_mis_sided_label']
            cls = 'VANISHES' if n == 0 else 'REDUCED' if nb and n < .5*nb else 'PERSISTS'
            rows.append({'row': rid, 'label': 'COUNTERFACTUAL_NOT_A_GATE' if d['counterfactual'] else 'UNCHANGED_SOLVER',
                         'baseline_p3b': nb, 'baseline_bridges': b['p3_split']['bridges'] if b else None,
                         'p3b': n, 'bridges': sp['bridges'], 'class': cls, 'outcome': e1_outcome(d)['outcome'],
                         'components': e1_outcome(d)['components'], 'mis_sided_labels': sp['mis_sided_labels_detail']})
    by = {r['row']: r.get('class') for r in rows}
    law_input = {t: by.get(f'V34_A_p{t}_k040') == 'PERSISTS' and by.get(f'V34_A_p{t}_grid') == 'PERSISTS' for t in (10, 30)}
    return {'verdict': 'INCONCLUSIVE', 'note': 'NO_PREDICTION declared (review: direction not predicted); measured classes',
            'rows': rows, 'p3b_is_law_input_problem': law_input}


def v41(P, ro):
    from robotic_cutting_sim import root_cause_diagnostics as rd
    cases, fits = [], {}
    for c in ('CUT', 'OBL20', 'MULTI2'):
        F = {dt: (ro.get(f'{c}_s100_g4s_dt{dt}') or {}).get('steady_mean_abs_N') for dt in ('0200', '0100', '0050', '0025')}
        if F['0050'] is None or F['0025'] is None:
            continue
        f_inf, dt0 = rd.series_fit(.05, F['0050'], .025, F['0025'])
        fits[c] = {'forces_N': F, 'f_inf_N': f_inf, 'dt0_ms': dt0,
                   'check_0p1': rd.series_predict(f_inf, dt0, .1) if dt0 else None,
                   'check_0p2': rd.series_predict(f_inf, dt0, .2) if dt0 else None}
        decl = P['hypotheses']['V4.1']['declared_fit'].get(c)
        if decl:
            good = dt0 is not None and abs(dt0/decl['dt0_ms']-1) <= .05 and abs(f_inf/decl['f_inf_N']-1) <= .05
            cases.append(case(c, 'CONFIRMED' if good else 'REFUTED', declared=decl, measured=fits[c]))
    verdict, counts = combine(cases)
    return {'verdict': verdict, 'counts': counts, 'cases': cases, 'fits': fits}


def v42_43_44(P, runs):
    out = {}
    pred = P['hypotheses']['V4.2']['predicted_N']
    cases, trim = [], []
    for c in ('CUT', 'OBL20', 'TRIM'):
        for k in ('020', '040'):
            rid = f'V42_{c}_k{k}'
            d = ok(runs, rid)
            if c == 'TRIM':
                trim.append({'row': rid, 'steady_mean_abs_N': force(d)})
                continue
            if d is None:
                cases.append(case(rid, 'NOT_RUN' if rid not in runs else 'NOT_EVALUABLE'))
                continue
            p, f = pred[c][f'k{k}'], force(d)
            dev = f/p-1
            cases.append(case(rid, 'CONFIRMED' if abs(dev) <= .10 else 'REFUTED' if abs(dev) > .25 else 'INCONCLUSIVE',
                              label='COUNTERFACTUAL_NOT_A_GATE', predicted_N=p, measured_N=f, deviation=dev,
                              twin_V12_dt0050_N=force(ok(runs, f'V12_{c}_dt0050')) if k == '040' else None,
                              out_of_zone_max_penetration_m=((d['analysis']['force_split'].get('steady') or {}).get('out') or {}).get('max_penetration_m'),
                              momentum_final=d['analysis']['momentum_energy_final']['momentum_kg_m_s']))
    v, cnt = combine(cases)
    out['V4.2'] = {'verdict': v, 'counts': cnt, 'cases': cases, 'TRIM_reported': trim}
    pred3 = P['hypotheses']['V4.3']['predicted_N']
    cases, trim = [], []
    for c in ('CUT', 'OBL20', 'TRIM'):
        for tag in ('0050', '0025'):
            rid = f'V43_{c}_dt{tag}_kfix'
            d = ok(runs, rid)
            if c == 'TRIM':
                trim.append({'row': rid, 'steady_mean_abs_N': force(d), 'V12_TRIM_dt0100_N': force(ok(runs, 'V12_TRIM_dt0100'))})
                continue
            if d is None:
                cases.append(case(rid, 'NOT_RUN' if rid not in runs else 'NOT_EVALUABLE'))
                continue
            p, f = pred3[c], force(d)
            dev = f/p-1
            cases.append(case(rid, 'CONFIRMED' if abs(dev) <= .10 else 'REFUTED' if abs(dev) > .25 else 'INCONCLUSIVE',
                              label='COUNTERFACTUAL_NOT_A_GATE', predicted_N=p, measured_N=f, deviation=dev,
                              twin_V12_dt0100_N=force(ok(runs, f'V12_{c}_dt0100'))))
    v, cnt = combine(cases)
    out['V4.3'] = {'verdict': v, 'counts': cnt, 'cases': cases, 'TRIM_reported': trim}
    acc = P['hypotheses']['V4.4']['accepted_N']
    cases = []
    for c in ('CUT', 'OBL20'):
        rid = f'V44_{c}_dt00125'
        d = ok(runs, rid)
        if d is None:
            cases.append(case(rid, 'NOT_RUN' if rid not in runs else 'NOT_EVALUABLE'))
            continue
        f = force(d)
        lo, hi = acc[c]
        cases.append(case(rid, 'CONFIRMED' if lo <= f <= hi else 'REFUTED', accepted_N=[lo, hi], measured_N=f))
    v, cnt = combine(cases)
    out['V4.4'] = {'verdict': v, 'counts': cnt, 'cases': cases}
    return out


def split_summary(d, window='steady'):
    fs = (d or {}).get('analysis', {}).get('force_split', {}).get(window)
    if not fs:
        return None
    return {'in_mean_abs_N': fs['in']['mean_abs_N'], 'out_mean_abs_N': fs['out']['mean_abs_N'],
            'in_along_d_N': fs['in']['mean_reaction_along_d_N'], 'out_along_d_N': fs['out']['mean_reaction_along_d_N'],
            'in_along_e_N': fs['in']['mean_reaction_along_e_N'], 'out_along_e_N': fs['out']['mean_reaction_along_e_N'],
            'in_along_n_N': fs['in']['mean_reaction_along_n_N'], 'out_along_n_N': fs['out']['mean_reaction_along_n_N'],
            'in_contacts_per_substep': fs['in']['contacts_per_substep'], 'out_contacts_per_substep': fs['out']['contacts_per_substep'],
            'total_blade_mean_abs_N': fs['total_blade_mean_abs_N'], 'released_along_d_N': fs['released_mean_along_d_N'],
            'in_zone_raw_along_d_N': fs['in_zone_raw_reaction_along_d_N'], 'out_share': fs['out_share_of_total_abs']}


def v45(P, runs):
    ref = ok(runs, 'V12_CUT_dt0050')
    ref_total = (split_summary(ref) or {}).get('total_blade_mean_abs_N')
    cases, table = [], []
    for s in ('075', '100', '150', '200'):
        base = ok(runs, 'V12_CUT_dt0050' if s == '100' else f'V45_CUT_s{s}_dt0050')
        cf = ok(runs, f'V45_CUT_s{s}_dt0050_zone1mm')
        table.append({'spacing_mm': int(s)/100, 'unchanged': split_summary(base), 'unchanged_steady_N': force(base),
                      'zone1mm_COUNTERFACTUAL': split_summary(cf), 'zone1mm_steady_N': force(cf)})
        if s in ('150', '200'):
            sb = split_summary(base)
            if sb is None or ref_total is None:
                cases.append(case(f'CUT s {int(s)/100} mm', 'NOT_RUN'))
                continue
            limit = .1*ref_total
            cases.append(case(f'CUT s {int(s)/100} mm', 'CONFIRMED' if sb['out_mean_abs_N'] <= limit else 'REFUTED',
                              out_of_zone_steady_mean_abs_N=sb['out_mean_abs_N'], limit_N=limit, reference_total_N=ref_total))
    v, cnt = combine(cases)
    return {'verdict': v, 'counts': cnt, 'cases': cases, 'table': table}


def v46(P, runs, ro):
    from robotic_cutting_sim import root_cause_diagnostics as rd
    F = {dt: (ro.get(f'TRIM_s100_g4s_dt{dt}') or {}).get('steady_mean_abs_N') for dt in ('0200', '0100', '0050', '0025')}
    cases = []
    if all(F.values()):
        ratios = [F['0100']/F['0200'], F['0050']/F['0100'], F['0025']/F['0050']]
        good = all(abs(a-b) <= .02 for a, b in zip(ratios, P['hypotheses']['V4.6']['declared_ratios']))
        f_inf, dt0 = rd.series_fit(.05, F['0050'], .025, F['0025'])
        p01 = rd.series_predict(f_inf, dt0, .1) if dt0 else None
        off = abs(p01/F['0100']-1) if p01 else None
        v = 'CONFIRMED' if good and (off is None or off > .25) else 'REFUTED' if off is not None and off <= .10 else 'INCONCLUSIVE'
        cases.append(case('TRIM dt series (resolution_002)', v, ratios=ratios, fit={'f_inf_N': f_inf, 'dt0_ms': dt0},
                          fit_check_0p1_N=p01, observed_0p1_N=F['0100'], fit_off=off))
    table = []
    for rid in P['hypotheses']['V4.6']['rows']:
        d = ok(runs, rid)
        table.append({'row': rid, 'label': 'COUNTERFACTUAL_NOT_A_GATE' if (d or {}).get('counterfactual') else 'UNCHANGED_SOLVER',
                      'dt_s': (d or {}).get('material_dt_s'), 'pic': (d or {}).get('effective_pic'),
                      'steady_N': force(d), 'split_steady': split_summary(d), 'split_window': split_summary(d, 'window')})
    v, cnt = combine(cases)
    return {'verdict': v, 'counts': cnt, 'cases': cases, 'attribution_table': table}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--markdown', type=Path)
    a = p.parse_args()
    P = json.loads(PROTOCOL.read_text())
    runs = load_runs(a.runs)
    t0 = time.monotonic()
    ro = readonly_resolution(a.runs/'readonly_resolution_002.json')
    e1 = readonly_e1(a.runs/'readonly_phase4_e1_001.json')
    s7 = readonly_s7()
    H = {'V1.1': v11(P, ro), 'V1.2': v12(P, runs), 'V1.3': v13(P, runs, ro), 'V1.4': v14(P, runs, ro),
         'V2.1': v21(P, ro, s7), 'V2.2': v22(P, runs), 'V2.3': v23(P, runs, ro), 'V3.1': v31(P, ro, e1),
         'V3.2': v32(P, runs, e1), 'V3.3': v33(P, runs, ro, e1), 'V3.4': v34(P, runs, e1), 'V4.1': v41(P, ro)}
    H.update(v42_43_44(P, runs))
    H['V4.5'] = v45(P, runs)
    H['V4.6'] = v46(P, runs, ro)
    declared = [r['id'] for r in P['rows']]
    status = {rid: (runs[rid]['status'] if rid in runs else 'NOT_RUN') for rid in declared}
    walls = {rid: runs[rid].get('wall_s') for rid in declared if rid in runs}
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    evidence = {'schema': 'wp12_phase4_verification_evidence_v1', 'protocol': str(PROTOCOL.relative_to(ROOT)),
                'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), 'git_sha': sha,
                'runs': str(a.runs), 'generated_unix': time.time(), 'summary_wall_s': time.monotonic()-t0,
                'labels': P['labels'], 'row_status': status,
                'row_walls_s': walls, 'gpu_h_summed_process_walls': sum(w for w in walls.values() if w)/3600,
                'verdicts': {k: v['verdict'] for k, v in H.items()}, 'hypotheses': H,
                'scope': 'Kinematic ideal tracking; provisional uncalibrated material; tlmpm_v5 and law 8 unchanged. '
                         'COUNTERFACTUAL rows test mechanisms; they are not acceptance results, laws or profiles.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(jsonable(evidence), indent=1)+'\n')
    print(json.dumps(evidence['verdicts'], indent=1))
    print('rows', {s: list(status.values()).count(s) for s in set(status.values())}, 'GPU-h',
          round(evidence['gpu_h_summed_process_walls'], 2))


if __name__ == '__main__':
    main()
