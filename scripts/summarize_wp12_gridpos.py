#!/usr/bin/env python3
"""Read-only summary of the tlmpm_v6 candidate acceptance (configs/wp12/phase4_gridpos_protocol_v1.json).

Reads ``artifacts/wp12/gridpos_NNN/<arm>/<row>/gridpos_result.json`` (and the frozen runner subdirectories) and applies
the protocol's gates: per row the V6 and V5 verdicts, integrity, component equality on separating rows, TRIM removal /
substrate, "not worse than V5"; per stage PASS / FAIL / INCOMPLETE; the S1 kill rule; the reported (never gated)
differences (blade force, tool work, overlap, KE, elastic energy, labels, piece velocities); the repeat bands; and the
Phase 3 reopen measure (first-band link consistency P50 / P99, V5 / V6) on the declared effect rows, only computed as
ELIGIBLE when v6 is ACCEPTED. Nothing is re-run or re-scored; a missing row is NOT_RUN.

  .venv/bin/python scripts/summarize_wp12_gridpos.py --root artifacts/wp12/gridpos_001 \\
      --evidence docs/evidence/wp12/phase4_gridpos_001.json
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PROTOCOL = ROOT/'configs/wp12/phase4_gridpos_protocol_v1.json'
RANK = {'PASS': 3, 'SEPARATED': 3, 'task_success': 3, 'TASK_SUCCESS': 3, 'PASS_NONFORMAL': 2, 'CHARACTERIZATION': 2,
        'REPORT_ONLY': 2, 'CONNECTED': 1, 'OTHER': 1, 'FAIL': 1, 'CUTTING_FAILURE': 1, 'FAIL_NONFORMAL': 1,
        'FAIL_INTEGRITY': 0, 'ERROR': 0, 'INVALID': 0, None: 0}


def protocol_rows():
    return {r['id']: r for r in json.loads(PROTOCOL.read_text())['rows']}


def load(root, arm, rid):
    p = Path(root)/arm/rid/'gridpos_result.json'
    return json.loads(p.read_text()) if p.exists() else None


def rank(verdict):
    if isinstance(verdict, str) and verdict.startswith('components_'):
        return None
    return RANK.get(verdict, 1)


def components(r):
    return ((r or {}).get('final_topology') or {}).get('mechanical_components')


def row_gate(row, v6, v5, stage=None):
    """(fail: bool, reasons, details) for one row id from the two arms (missing V5 -> comparison NOT_RUN)."""
    reasons, details = [], {}
    if v6 is None:
        return {'fail': False, 'pending': True, 'reasons': ['V6 NOT_RUN'], 'details': {}}
    gate = row['gate']
    if v6.get('status') != 'COMPLETED':
        reasons.append(f"V6 status {v6.get('status')}: {v6.get('error')}")
    if not (v6.get('integrity') or {}).get('pass', False):
        reasons.append(f"V6 integrity {(v6.get('integrity') or {}).get('checks')}")
    verdict6 = v6.get('verdict')
    verdict5 = (v5 or {}).get('verdict')
    details.update(verdict_v6=verdict6, verdict_v5=verdict5)
    v5_also = False
    if gate in ('kill', 'acceptance') and row['kind'] != 'verify' and verdict6 != 'PASS':
        v5_also = v5 is not None and verdict5 != 'PASS'
        reasons.append(f'V6 frozen criterion {verdict6}' + (f' (V5 also {verdict5})' if v5_also else ''))
    if v5 is not None and gate in ('kill', 'acceptance', 'known_failure', 'effect'):
        r6, r5 = rank(verdict6), rank(verdict5)
        if r6 is not None and r5 is not None and r6 < r5:
            reasons.append(f'V6 verdict {verdict6} worse than V5 {verdict5}')
    if row.get('separating') and v5 is not None and gate != 'characterization':
        c6, c5 = components(v6), components(v5)
        details.update(components_v6=c6, components_v5=c5)
        if c6 != c5:
            reasons.append(f'components V6 {c6} != V5 {c5}')
    if row.get('trim') and gate != 'characterization':
        t = v6.get('trim_numbers') or {}
        details['trim_v6'] = {k: t.get(k) for k in ('removal', 'substrate')}
        if t.get('removal') is not None and t['removal'] < .9:
            reasons.append(f"TRIM removal {t['removal']:.4f} < 0.9")
        if t.get('substrate') is not None and t['substrate'] < .99:
            reasons.append(f"TRIM substrate {t['substrate']:.4f} < 0.99")
    if gate in ('report', 'characterization', 'diagnostic'):
        hard = [r for r in reasons if r.startswith(('V6 status', 'V6 integrity'))]
        details['not_gated'] = [r for r in reasons if r not in hard]
        reasons = hard
    # S2/S3 acceptance rows whose frozen criterion also fails on the V5 arm (same commit) are not attributable to v6:
    # reported, never hidden; the S1 kill rule counts every failure.
    attributable = [r for r in reasons if not r.endswith(')') or '(V5 also' not in r]
    if gate == 'kill':
        attributable = list(reasons)
    return {'fail': bool(attributable), 'pending': False, 'reasons': reasons, 'attributable': attributable,
            'v5_also_fails': v5_also, 'details': details}


def s1_gate(root, rid):
    rows = protocol_rows()
    base = rid.rsplit('_r', 1)[0] if rid.rsplit('_r', 1)[-1].isdigit() and rid not in rows else rid
    row = rows[base]
    v6, v5 = load(root, 'V6', rid), load(root, 'V5', rid)
    return row_gate(row, v6, v5, 'S1')


def _get(r, *path):
    for k in path:
        if not isinstance(r, dict):
            return None
        r = r.get(k)
    return r


REPORTED = {'blade_force_mean_abs_contact_N': ('monitor', 'blade_force_mean_abs_contact_N'),
            'blade_force_peak_abs_N': ('monitor', 'blade_force_peak_abs_N'), 'tool_work_j': ('monitor', 'tool_work_j'),
            'max_blade_overlap_m': ('monitor', 'max_blade_overlap_m'), 'max_fixture_overlap_m': ('monitor', 'max_fixture_overlap_m'),
            'final_particle_ke_j': ('monitor', 'final_particle_ke_j'), 'final_grid_ke_j': ('monitor', 'final_grid_ke_j'),
            'final_null_space_ke_j': ('monitor', 'final_null_space_ke_j'), 'max_null_space_ke_j': ('monitor', 'max_null_space_ke_j'),
            'final_elastic_j': ('monitor', 'final_elastic_j'), 'min_det_F': ('monitor', 'min_det_F'),
            'final_labelled': ('final_topology', 'labelled'), 'mechanical_components': ('final_topology', 'mechanical_components'),
            'grid_position_residual_m': ('grid_position_residual_m',), 'drift_max_m': ('drift_max_m',),
            'min_local_z_m': ('monitor', 'min_local_z_m'), 'wall_s': ('wall_s',)}


def reported(r):
    return {k: _get(r, *p) for k, p in REPORTED.items()} if r else None


def verify_links(r):
    """First-band (0-1 s) link consistency P50 / P99 of a verify row (the phase4_verify_001 analysers)."""
    if r is None:
        return None
    d = Path(r['driver']['frozen_result_dir'])
    rec = json.loads((d/'result.json').read_text()) if (d/'result.json').exists() else {}
    band = _get(rec, 'analysis', 'links_FINAL', '0-1s') or {}
    return {'p50': band.get('p50'), 'p99': band.get('p99'), 'n': band.get('n') or band.get('count')}


def standin_links(row):
    """V5 stand-in (resolution_002) first-band link metrics computed with the same diagnostics as the verify analyser."""
    path = ROOT/row['v5_standin']
    res = path/'result.json'
    if not res.exists():
        return {'status': 'NOT_FOUND', 'path': str(path)}
    rec = json.loads(res.read_text())
    runner_ok = rec.get('runner_sha256') == hashlib.sha256((ROOT/'scripts/run_wp12_resolution.py').read_bytes()).hexdigest()
    v5_unchanged = subprocess.run(['git', 'log', '--format=%H', f"{rec['git_sha']}..HEAD", '--',
                                   'src/robotic_cutting_sim/tlmpm_v5.py'], cwd=ROOT, capture_output=True, text=True).stdout.strip() == ''
    spec = row['args']['row']
    same = all(abs(float(rec['row'][k])-float(spec[k])) < 1e-12 for k in ('spacing_m', 'grid_dx_m', 'dt_s')) and rec['row']['case'] == spec['case']
    out = {'path': str(path), 'runner_sha256_identical': runner_ok, 'tlmpm_v5_unchanged_since_record': v5_unchanged,
           'row_config_identical': same, 'status': 'OK' if (runner_ok and v5_unchanged and same) else 'MISMATCH'}
    try:
        from robotic_cutting_sim import root_cause_diagnostics as rd
        z = np.load(path/'state.npz')
        X = z['reference'].astype(np.float64)
        geometry = rec['geometry']
        if spec['case'] == 'TRIM':
            zc = float(geometry['plane_z_m'])
            dist = lambda P: np.abs(np.asarray(P)[:, 2]-zc)   # noqa: E731
        else:
            planes = [(p['origin'], p['normal']) for p in geometry['planes']]
            dist = lambda P: rd.plane_distance(P, planes)   # noqa: E731
        links = rd.link_consistency(X, z['lattice'], z['FINAL_positions'], z['FINAL_F'], z['FINAL_codes'], spec['spacing_m'], dist)
        band = links.get('0-1s') or {}
        out.update(p50=band.get('p50'), p99=band.get('p99'),
                   verdict='PASS' if (rec.get('analysis', {}).get('FINAL', {}).get('separation_criterion') or
                                      rec.get('analysis', {}).get('FINAL', {}).get('trim_criterion_wp11')) else 'FAIL',
                   mechanical_components=rec.get('analysis', {}).get('FINAL', {}).get('mechanical_components'))
    except Exception as error:   # reported, never silently dropped
        out['links_error'] = repr(error)
    return out


def summarize(root, accepted_override=None):
    rows = protocol_rows()
    stages = {}
    table = []
    for rid, row in rows.items():
        if row['stage'] == 'S4':
            table.append({'row': rid, 'stage': 'S4', 'status': 'NOT_RUN', 'reason': 'not scheduled (Native use only)'})
            continue
        ids = [rid]+[f'{rid}_r{k}' for k in range(2, int(row.get('repeat', 1))+1)]
        for i in ids:
            v6, v5 = load(root, 'V6', i), load(root, 'V5', i)
            entry = {'row': i, 'stage': row['stage'], 'kind': row['kind'], 'gate_class': row['gate']}
            standin = None
            if row.get('v5_standin'):
                standin = standin_links(row)
                entry['v5_standin'] = standin
            if v6 is None:
                entry.update(status='NOT_RUN')
            else:
                g = row_gate(row, v6, v5 if v5 is not None else ({'verdict': standin.get('verdict'), 'status': 'COMPLETED',
                                                                   'final_topology': {'mechanical_components': standin.get('mechanical_components')}}
                                                                  if standin and standin.get('status') == 'OK' else None))
                entry.update(status='FAIL' if g['fail'] else ('PASS_V5_ALSO_FAILS' if g.get('v5_also_fails') else 'PASS'),
                             reasons=g['reasons'], attributable=g.get('attributable'), details=g['details'],
                             v6=reported(v6), v5=reported(v5), wall_s={'V6': (v6 or {}).get('wall_s'), 'V5': (v5 or {}).get('wall_s')},
                             grid_position_residual_v6_m=v6.get('grid_position_residual_m'),
                             solver_verified={'V6': _get(v6, 'solver_selection', 'verified'), 'V5': _get(v5, 'solver_selection', 'verified')})
                if row['kind'] == 'verify':
                    entry['links_first_band'] = {'V6': verify_links(v6), 'V5': verify_links(v5) if v5 else
                                                 ({'p50': standin.get('p50'), 'p99': standin.get('p99'), 'source': 'v5_standin'} if standin else None)}
                if v5 is None and not standin:
                    entry['comparison'] = 'V5 NOT_RUN'
            table.append(entry)
            st = stages.setdefault(row['stage'], {'pass': 0, 'fail': 0, 'not_run': 0, 'failed_rows': []})
            if entry['status'] == 'NOT_RUN':
                st['not_run'] += 1
            elif entry['status'] == 'FAIL':
                st['fail'] += 1
                st['failed_rows'].append(i)
            else:
                st['pass'] += 1
                if entry['status'] == 'PASS_V5_ALSO_FAILS':
                    st.setdefault('v5_also_fails_rows', []).append(i)
    for name, st in stages.items():
        st['verdict'] = 'FAIL' if st['fail'] else ('INCOMPLETE' if st['not_run'] else
                                                   'PASS_WITH_V5_FAILURES' if st.get('v5_also_fails_rows') else 'PASS')
    s1 = stages.get('S1', {}).get('verdict')
    decision = ('REJECTED' if any(st['verdict'] == 'FAIL' for st in stages.values()) else
                'ACCEPTED_BY_GATES' if all(stages.get(s, {}).get('verdict') == 'PASS' for s in ('S1', 'S2', 'S3')) else
                'ACCEPTED_BY_GATES_V5_FAILURES_TO_REVIEW' if all(stages.get(s, {}).get('verdict') in ('PASS', 'PASS_WITH_V5_FAILURES')
                                                                 for s in ('S1', 'S2', 'S3')) else 'INCOMPLETE')
    # repeat bands
    bands = {}
    for rid, row in rows.items():
        if int(row.get('repeat', 1)) < 2:
            continue
        for arm in ('V6', 'V5'):
            a, b = load(root, arm, rid), load(root, arm, f'{rid}_r2')
            if a and b:
                ra, rb = reported(a), reported(b)
                bands.setdefault(rid, {})[arm] = {k: (abs(ra[k]-rb[k]) if isinstance(ra.get(k), (int, float)) and isinstance(rb.get(k), (int, float)) else None)
                                                 for k in ra}
    # reopen measure
    reopen = {'rule': 'q50 = P50_V5/P50_V6 >= 2 and q99 >= 2 on each of EFF_CUT, EFF_TRIM, EFF_MULTI2, EFF_S7, EFF_G3b; only if ACCEPTED',
              'rows': {}}
    for rid in ('EFF_CUT', 'EFF_TRIM', 'EFF_MULTI2', 'EFF_S7', 'EFF_G3b', 'EFF_OBL20', 'RES_CUT_s050_dt0050',
                'RES_TRIM_s050_dt0050', 'RES_MULTI2_s050_dt0050'):
        e = next((t for t in table if t['row'] == rid), None)
        lk = (e or {}).get('links_first_band') or {}
        a, b = lk.get('V5') or {}, lk.get('V6') or {}
        q50 = a['p50']/b['p50'] if a.get('p50') and b.get('p50') else None
        q99 = a['p99']/b['p99'] if a.get('p99') and b.get('p99') else None
        reopen['rows'][rid] = {'V5': a, 'V6': b, 'q50': q50, 'q99': q99}
    measured = [reopen['rows'][r] for r in ('EFF_CUT', 'EFF_TRIM', 'EFF_MULTI2', 'EFF_S7', 'EFF_G3b')]
    holds = all(m['q50'] is not None and m['q99'] is not None and m['q50'] >= 2 and m['q99'] >= 2 for m in measured)
    reopen['measure_holds'] = holds if all(m['q50'] is not None for m in measured) else None
    accepted = decision.startswith('ACCEPTED_BY_GATES')
    reopen['status'] = ('ELIGIBLE_IF_ACCEPTED_BY_REVIEW' if accepted and holds else
                        'NOT_ELIGIBLE' if accepted or decision == 'REJECTED' else 'NOT_EVALUATED (gates incomplete)')
    spent = 0.
    ledger = Path(root)/'queue_ledger.jsonl'
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            rec = json.loads(line)
            if rec.get('event') == 'finished':
                spent += rec['wall_s']/3600
    return {'stages': stages, 'decision_by_gates': decision, 'S1': s1, 'rows': table, 'repeat_bands': bands,
            'reopen_measure': reopen, 'gpu_h_spent_queue': spent}


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return jsonable(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--evidence')
    p.add_argument('--extra', help='JSON file embedded verbatim under "extra" (read-only diagnoses, notes)')
    a = p.parse_args()
    summary = summarize(a.root)
    if a.extra:
        summary['extra'] = json.loads(Path(a.extra).read_text())
    head = {'evidence': 'WP-12 Phase 4: tlmpm_v6 CANDIDATE staged acceptance (grid-consistent position update)',
            'protocol': str(PROTOCOL.relative_to(ROOT)), 'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            'root': a.root, 'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'created': datetime.now().astimezone().isoformat(),
            'scope': 'KINEMATIC ideal-tracking rows (and measured-trajectory replays); provisional uncalibrated material; a '
                     'candidate acceptance, not adoption; forces are resolution-specific diagnostics (NOT_CONVERGED)'}
    out = {**head, **summary}
    text = json.dumps(jsonable(out), indent=1)
    if a.evidence:
        path = Path(a.evidence)
        if path.exists():
            raise SystemExit(f'{path} exists; evidence is never overwritten')
        path.write_text(text+'\n')
    print(json.dumps(jsonable({k: out[k] for k in ('stages', 'decision_by_gates', 'gpu_h_spent_queue')}), indent=1))


if __name__ == '__main__':
    main()
