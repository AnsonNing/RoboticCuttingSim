#!/usr/bin/env python3
"""Read-only summary of E1 (configs/wp12/phase4_e1_protocol_v1.json) -> evidence JSON and Markdown tables.

  .venv/bin/python scripts/summarize_wp12_phase4_e1.py --runs artifacts/wp12/phase4_e1_001 \\
      --reach artifacts/wp12/phase4_e1_001/reachability --out docs/evidence/wp12/phase4_e1_001.json --markdown <file>

Rows that have no result.json are reported NOT_RUN. Nothing is rescored: every number comes from result.json and
the reachability receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_wp12_phase4_e1 import PROTOCOL, protocol_rows  # noqa: E402

TIE_UM = 1.0


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def reach_status(r):
    if r is None:
        return 'NOT_RUN', None, None
    status, binding = r.get('status'), r.get('binding')
    slack = (r.get('binding_row') or {}).get('slack')
    if status == 'FAIL' and r.get('failing') == ['blade_collision'] and slack is not None and abs(slack) < TIE_UM*1e-6 \
            and (r.get('binding_row') or {}).get('pair', [None, ''])[1].startswith('static_flat_board'):
        return 'FAIL (TIE_AT_CONTACT)', binding, slack
    return status, binding, slack


def compact(row, result, reach):
    out = {'id': row['id'], 'family': row['family'], 'tau_deg': row['tau_deg'], 'edge_end_z_mm': row['edge_end_z_mm'],
           'overtravel_mm': row.get('overtravel_mm'), 'overtravel_rule': row.get('overtravel_rule')}
    status, binding, slack = reach_status(reach)
    out['reachability'] = {'status': status, 'binding': binding, 'binding_slack': slack,
                           'knife_handle_clearance_slack_m': ((reach or {}).get('slack_by_check') or {}).get('knife_handle_clearance'),
                           'joint_limit_margin_rad': (reach or {}).get('joint_limit_margin_rad')}
    if result is None:
        out['row_status'] = 'NOT_RUN'
        return out
    out['row_status'] = result.get('row_status')
    if result.get('row_status') == 'ERROR':
        out['error'] = result.get('error')
        return out
    a = result['analysis']
    law8 = a['law8_final']
    u = a['unassigned_within_12mm']
    ef = a['exit_face_assignment']
    cand = a['candidates_offline']
    out.update({
        'device': result.get('device'), 'wall_s': result.get('wall_s'), 'git_sha': result.get('git_sha'),
        'law8': {k: law8.get(k) for k in ('outcome', 'components', 'pieces_ge_1pct', 'mass_fractions', 'direct_bridges',
                                          'bridge_s_ref_mm', 'bridge_layers', 'labelled', 'bridges_by_class')},
        'unassigned_within_12mm': {k: u.get(k) for k in ('count', 'by_class', 'only_fail_particles', 'in_core_abs_s_le_4mm',
                                                         'lead_rule_blocked_by_plow_gate', 'ahead_of_final_edge_final_f_gt_0',
                                                         's_ref_mm', 'layers_z_ref_mm')},
        'exit_face_layer0': {side: ef.get(f'layer0_{side}') for side in ('s_lt_0', 's_ge_0')},
        'exit_face_digit0_by_layer': {k: v['digit0'] for k, v in ef.items()},
        'edge_to_board': a['edge_to_board'], 'board_penetration': a['board_penetration'],
        'dynamics_context': a['dynamics_context'],
        'candidates': {key: {'outcome': cand[key]['predicted']['outcome'], 'components': cand[key]['predicted']['components'],
                             'mass_fractions': cand[key]['predicted']['mass_fractions'][:3],
                             'direct_bridges': cand[key]['predicted']['direct_bridges'],
                             'shadow_labels': cand[key]['shadow_labels'], 'side1': cand[key]['side1'], 'side2': cand[key]['side2'],
                             'later_law8_band_gave_other_side': cand[key]['later_law8_band_gave_other_side'],
                             'still_law8_unassigned': cand[key]['still_law8_unassigned'],
                             's_at_assign_mm': cand[key]['s_at_assign_mm'], 'f_at_assign_mm': cand[key]['f_at_assign_mm']}
                       for key in ('proposed', 'exit_ligament')},
        'candidate_segments': cand['segments'],
        'candidate_blocking': {'proposed_near_unassigned_blocked_by_B': cand['proposed'].get('near_unassigned_blocked_by_B'),
                               'proposed_flags': cand['proposed'].get('flags'),
                               'el_near_unassigned_blocked_ligament_present': cand['exit_ligament'].get('near_unassigned_blocked_ligament_present'),
                               'el_near_unassigned_segment_never_seen': cand['exit_ligament'].get('near_unassigned_segment_never_seen')},
        'integrity': result.get('integrity')})
    return out


def onset(rows, sign):
    series = sorted((r for r in rows if r['family'] == 'E1.1_option_A' and 'repeat_of' not in r['spec']
                     and (r['tau_deg'] * sign > 0 or r['tau_deg'] == 0)), key=lambda r: abs(r['tau_deg']))
    outcomes = [(r['tau_deg'], (r.get('law8') or {}).get('outcome', r['row_status'])) for r in series]
    first = next((t for t, o in outcomes if o == 'CONNECTED'), None)
    robust = None
    for i, (t, o) in enumerate(outcomes):
        if all(oo == 'CONNECTED' for _, oo in outcomes[i:]) and o == 'CONNECTED':
            robust = t
            break
    return {'outcomes_by_tau': outcomes, 'first_connected_tau_deg': first, 'connected_from_tau_deg_onwards': robust}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--reach', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--markdown', type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; never overwritten')
    declared = protocol_rows(PROTOCOL)
    rows = []
    for rid, spec in declared.items():
        result = load(a.runs/rid/'result.json')
        reach_id = spec.get('repeat_of', rid)
        reach = load(a.reach/f'{reach_id}.json')
        r = compact(spec, result, reach)
        r['spec'] = spec
        rows.append(r)
    by = {r['id']: r for r in rows}
    raised = {}
    for r in rows:
        if r['family'] == 'E1.1_option_A_raised':
            raised.setdefault(f"{r['tau_deg']:g}", {})[f"{r['edge_end_z_mm']:g}"] = {
                'outcome': (r.get('law8') or {}).get('outcome', r['row_status']),
                'bridges': (r.get('law8') or {}).get('direct_bridges'),
                'unassigned_within_12mm': (r.get('unassigned_within_12mm') or {}).get('count'),
                'L9-EL': (r.get('candidates') or {}).get('exit_ligament', {}).get('outcome')}
    for tau in ('5', '20', '35'):
        base = by.get(f'A_p{int(tau):02d}')
        raised.setdefault(tau, {})['0'] = {'outcome': (base.get('law8') or {}).get('outcome', base['row_status']),
                                           'bridges': (base.get('law8') or {}).get('direct_bridges'),
                                           'unassigned_within_12mm': (base.get('unassigned_within_12mm') or {}).get('count'),
                                           'L9-EL': (base.get('candidates') or {}).get('exit_ligament', {}).get('outcome')}
    slotted = {}
    for r in rows:
        if r['family'] == 'E1.2_option_D_slotted':
            slotted.setdefault(f"{r['tau_deg']:g}", {})[r['overtravel_rule']] = {
                'overtravel_mm': r['overtravel_mm'], 'outcome': (r.get('law8') or {}).get('outcome', r['row_status']),
                'bridges': (r.get('law8') or {}).get('direct_bridges'), 'reachability': r['reachability']['status'],
                'L9-EL': (r.get('candidates') or {}).get('exit_ligament', {}).get('outcome')}
    repeats = {}
    for r in rows:
        if 'repeat_of' in r['spec'] and r.get('law8') and by[r['spec']['repeat_of']].get('law8'):
            b = by[r['spec']['repeat_of']]
            repeats[r['id']] = {k: [b['law8'].get(k), r['law8'].get(k)] for k in ('components', 'direct_bridges', 'labelled')}
            repeats[r['id']]['unassigned_within_12mm'] = [b['unassigned_within_12mm']['count'], r['unassigned_within_12mm']['count']]
            repeats[r['id']]['L9-EL_shadow_labels'] = [b['candidates']['exit_ligament']['shadow_labels'],
                                                       r['candidates']['exit_ligament']['shadow_labels']]
    def outcome(rid, key=None):
        r = by.get(rid) or {}
        if key is None:
            return (r.get('law8') or {}).get('outcome', r.get('row_status'))
        return ((r.get('candidates') or {}).get(key) or {}).get('outcome')
    a_rows = [r for r in rows if r['family'] == 'E1.1_option_A' and 'repeat_of' not in r['spec']]
    frac = {r['id']: (r.get('candidate_segments') or {}).get('proposed_B_zero_fraction_of_material_at_end') for r in a_rows}
    mirror = {}
    for r in a_rows:
        if r['tau_deg'] > 0:
            m = by.get(f"A_m{int(r['tau_deg']):02d}")
            if m and m.get('law8') and r.get('law8'):
                mirror[r['id']] = {'same_outcome': r['law8']['outcome'] == m['law8']['outcome'],
                                   'bridges': [r['law8']['direct_bridges'], m['law8']['direct_bridges']],
                                   'unassigned_within_12mm': [r['unassigned_within_12mm']['count'], m['unassigned_within_12mm']['count']],
                                   'labelled': [r['law8']['labelled'], m['law8']['labelled']]}
    predictions = {
        'P1_E1.1_onset': {'measured': {r['id']: outcome(r['id']) for r in a_rows},
                          'note': 'prediction: fails from 5-10 deg (margin) and robustly from 15-20 deg'},
        'P2_E1.1_proposed_B_k': {'B_zero_fraction_of_material_segments_at_end': frac,
                                 'ge_0.40_for_all_abs_tau_ge_20': all((frac[k] or 0) >= .40 for k in frac
                                                                      if abs(by[k]['tau_deg']) >= 20),
                                 'ge_0.75_for_all_abs_tau_30_35': all((frac[k] or 0) >= .75 for k in frac
                                                                      if abs(by[k]['tau_deg']) >= 30)},
        'P3_E1.2_slotted_groove_limit': {rid: outcome(rid) for rid in ('D_p20', 'D_p25', 'D_p30', 'D_p35')},
        'P3_reach_limited_not_predicted': {rid: outcome(rid) for rid in ('D_p20_reach', 'D_p25_reach', 'D_p30_reach',
                                                                         'D_p35_reach')},
        'P4_mirror': mirror,
        'P5_candidates_on_option_A_h0': {r['id']: {'law8': outcome(r['id']), 'proposed': outcome(r['id'], 'proposed'),
                                                   'exit_ligament': outcome(r['id'], 'exit_ligament')} for r in a_rows},
        'P6_tau0': {'law8': outcome('A_p00'), 'proposed': outcome('A_p00', 'proposed'),
                    'exit_ligament': outcome('A_p00', 'exit_ligament')}}
    reach_summary = load(a.reach/'summary.json')
    evidence = {
        'evidence_id': 'wp12_phase4_e1_001', 'protocol': str(PROTOCOL.relative_to(ROOT)),
        'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        'runs': str(a.runs), 'reachability': str(a.reach),
        'label': 'E1 measurement (law 8 unchanged; candidate rules offline/shadow). No adoption decision: D1 is decided at xhigh.',
        'commits': sorted({r.get('git_sha') for r in rows if r.get('git_sha')}),
        'row_status_counts': {s: sum(1 for r in rows if r['row_status'] == s) for s in sorted({r['row_status'] for r in rows})},
        'onset_flat_board_h0': {'positive_tau': onset(rows, 1), 'negative_tau': onset(rows, -1)},
        'raised_edge': raised, 'slotted': slotted, 'repeat_band': repeats, 'declared_predictions_measured': predictions,
        'reachability_wall_s': (reach_summary or {}).get('wall_s'),
        'gpu_wall_s_total': sum(r.get('wall_s') or 0 for r in rows),
        'rows': {r['id']: {k: v for k, v in r.items() if k != 'spec'} for r in rows}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(evidence, indent=1))
    if a.markdown:
        lines = ['| Row | τ (°) | Edge end z (mm) | Law 8 | Bridges | Unassigned ≤12 mm (in core) | Layer-0 digit-0 s<0 / s≥0 '
                 '| Proposed (B=0 frac at end) | L9-EL (labels) | Reach (binding, slack) | Wall (s) |',
                 '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in rows:
            if 'law8' not in r:
                lines.append(f"| {r['id']} | {r['tau_deg']:g} | {r['edge_end_z_mm']:g} | {r['row_status']} | | | | | | "
                             f"{r['reachability']['status']} | |")
                continue
            u = r['unassigned_within_12mm']
            l0 = r['exit_face_layer0']
            seg = r['candidate_segments']
            frac = seg.get('proposed_B_zero_fraction_of_material_at_end')
            slack = r['reachability']['binding_slack']
            lines.append(
                f"| {r['id']} | {r['tau_deg']:g} | {r['edge_end_z_mm']:g} | {r['law8']['outcome']} ({r['law8']['components']}) "
                f"| {r['law8']['direct_bridges']} | {u['count']} ({u['in_core_abs_s_le_4mm']}) "
                f"| {l0['s_lt_0']['digit0']} / {l0['s_ge_0']['digit0']} "
                f"| {r['candidates']['proposed']['outcome']} ({'' if frac is None else f'{frac:.2f}'}) "
                f"| {r['candidates']['exit_ligament']['outcome']} ({r['candidates']['exit_ligament']['shadow_labels']}) "
                f"| {r['reachability']['status']} ({r['reachability']['binding']}, "
                f"{'' if slack is None else f'{slack*1e3:+.4f} mm'}) | {r.get('wall_s') or 0:.0f} |")
        a.markdown.write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: evidence[k] for k in ('row_status_counts', 'onset_flat_board_h0', 'slotted', 'repeat_band')}, indent=1))


if __name__ == '__main__':
    main()
