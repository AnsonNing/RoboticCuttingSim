#!/usr/bin/env python3
"""WP-12 Phase 4a: compact evidence for the law-8 root-cause study (read-only; never overwrites).

  .venv/bin/python scripts/summarize_wp12_phase4a.py --out docs/evidence/wp12/phase4a_law8_rootcause_001.json

Collects, from the phase4a run directories: each run's wrapped command, exit status (GNU time), wall time, git
sha, the latest ``phase4a_analysis_*.json`` / ``phase4a_chips_*.json`` digests, and the controlled-sweep summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'oblique_s2201': ('artifacts/wp12/phase4a_002/s2201', 'artifacts/wp12/logs/phase4a_002_s2201.log'),
    'oblique_s2202': ('artifacts/wp12/phase4a_002/s2202', 'artifacts/wp12/logs/phase4a_002_s2202.log'),
    'oblique_s2203': ('artifacts/wp12/phase4a_002/s2203', 'artifacts/wp12/logs/phase4a_002_s2203.log'),
    'rake45_s2307': ('artifacts/wp12/phase4a_002/s2307', 'artifacts/wp12/logs/phase4a_002_s2307.log'),
    'rake45_s2307_COUNTERFACTUAL_theta_max_60': ('artifacts/wp12/phase4a_002/s2307_cf_thetamax60',
                                                 'artifacts/wp12/logs/phase4a_002_s2307_cf_thetamax60.log'),
    'test_d2_s2501': ('artifacts/wp12/phase4a_002/s2501', 'artifacts/wp12/logs/phase4a_002_s2501.log'),
    'trim_k2_r10': ('artifacts/wp12/phase4a_004/trim_k2_r10_law8_001', 'artifacts/wp12/logs/phase4a_004_trim_k2_r10_law8.log'),
    'phaseM_bumps3': ('artifacts/wp12/phase4a_004/bumps3_law8_v5_001', 'artifacts/wp12/logs/phase4a_004_bumps3_law8.log'),
    'f26_pure_draw': ('artifacts/wp12/phase4a_005/f2_6_pure_draw_law8_001', 'artifacts/wp12/logs/phase4a_005_f2_6_pure_draw.log'),
    'f26_saw_cut': ('artifacts/wp12/phase4a_005/f2_6_saw_cut_law8_001', 'artifacts/wp12/logs/phase4a_005_f2_6_saw_cut.log'),
    'f24_rocking': ('artifacts/wp12/phase4a_005/f2_4_rocking_law8_001', 'artifacts/wp12/logs/phase4a_005_f2_4_rocking.log'),
    'f25_pure_rotation': ('artifacts/wp12/phase4a_005/f2_5_pure_rotation_law8_001',
                          'artifacts/wp12/logs/phase4a_005_f2_5_pure_rotation.log'),
}
PRESERVED_FAILED = {
    'artifacts/wp12/phase4a_002/trim_k2_r10': 'not created: run_phase_c4_trim_material.py rejected --out without "law8" '
                                              '(exit 2, 0.4 s); rerun as phase4a_004/trim_k2_r10_law8_001',
    'artifacts/wp12/phase4a_002/f26_pure_draw': 'not created: run_phase_f_material.py rejected --out without "law8" (exit 2); '
                                                'rerun in phase4a_005',
    'artifacts/wp12/phase4a_002/f26_saw_cut': 'same as f26_pure_draw',
    'queue_rc_note': 'phase4a_002/queue.log prints rc of $(date) (always 0); the exit status is the GNU time '
                     '"Exit status" line of each log',
}


def latest(run, stem):
    files = sorted(run.glob(f'{stem}_*.json'))
    return json.loads(files[-1].read_text()) if files else None


def log_info(path):
    if not path.exists():
        return None
    text = path.read_text(errors='replace')
    exit_ = re.findall(r'Exit status: (\d+)', text)
    wall = re.findall(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)', text)
    return {'log': str(path.relative_to(ROOT)), 'exit_status': int(exit_[-1]) if exit_ else None,
            'wall_clock': wall[-1] if wall else None}


def compact_analysis(a):
    if a is None:
        return None
    keep = {k: a[k] for k in ('class', 'class_reason', 'evaluation_verdict', 'mechanical_components',
                              'component_mass_fractions', 'labelled_any_digit', 'max_tool_penetration_mm',
                              'max_fixture_penetration_mm', 'flipped_by_float_fallback_total',
                              'flipped_by_geom_fallback_total', 'counterfactual') if k in a}
    keep['digits'] = {}
    for k, info in a['digits'].items():
        d = {x: info[x] for x in ('armed_substeps', 'stroke_on_substeps', 'engaged_substeps', 'seg_ok_fraction',
                                  'seg_plow_fraction', 'mean_segment_motion_offplane_deg', 'cutting_work_J',
                                  'work_with_gate_closed_particle_substeps', 'band_new', 'lead_new',
                                  'direct_code0_bridges', 'relabel_tests_diagnostic_only',
                                  'labels_against_final_plane_sign_beyond_2_spacings', 'new_digit_flags',
                                  'ill_conditioned_segment_substeps') if x in info}
        for group in ('unassigned_within_12mm', 'bridges'):
            g = info.get(group) or {}
            d[group] = {x: g[x] for x in ('count', 'by_class', 'reference_z_layers_mm', 's_final_mm', 'f_final_mm',
                                          'ahead_of_final_edge', 'displacement_mm', 'min_f_stroke_mm',
                                          'lead_rule_blocked_by_plow_gate_particles', 'only_fail_totals',
                                          'sample_reference_mm') if x in g}
        keep['digits'][k] = d
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sweep', type=Path, default=ROOT/'artifacts/wp12/phase4a_003/sweep')
    parser.add_argument('--notes', type=Path, help='JSON with classification/remedy text copied into the evidence')
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; never overwritten')
    evidence = {'schema': 'wp12_phase4a_law8_rootcause_v1',
                'scope': ('Diagnosis only: kinematic ideal tracking, formal resolution (1 mm particles, 4 mm grid, '
                          '0.1 ms), provisional uncalibrated material, cutting law 8 unchanged (tlmpm_v4 / tlmpm_v5). '
                          'No new law, no threshold change. COUNTERFACTUAL rows override one law parameter on the '
                          'solver instance and are never law-8 or acceptance results.'),
                'tools': {}, 'runs': {}, 'preserved_failed_attempts': PRESERVED_FAILED}
    for tool in ('src/robotic_cutting_sim/cut_diagnostics.py', 'scripts/run_wp12_phase4a_diagnose.py',
                 'scripts/run_wp12_phase4a_sweep.py', 'scripts/analyze_wp12_phase4a.py',
                 'scripts/analyze_wp12_phase4a_chips.py', 'scripts/plot_wp12_phase4a.py', 'tests/test_cut_diagnostics.py'):
        path = ROOT/tool
        evidence['tools'][tool] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    for name, (run, log) in RUNS.items():
        run, log = ROOT/run, ROOT/log
        entry = {'run_dir': str(run.relative_to(ROOT)), 'present': run.exists(), **(log_info(log) or {})}
        meta_path = run/'wp12_phase4a_diag.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            entry.update(wrapped_command=meta['wrapped_command'], wrapped_exit=meta['wrapped_exit'],
                         git_sha=meta.get('git_sha'), git_dirty_paths=len(meta.get('git_dirty_paths') or []),
                         wall_s=meta.get('wall_s'), counterfactual=meta.get('counterfactual'),
                         solver_module=(meta.get('meta') or {}).get('solver_module'),
                         diag_substeps=(meta.get('cut_diagnostics') or {}).get('substeps'),
                         diag_overflow=(meta.get('cut_diagnostics') or {}).get('summary_overflow'))
        entry['analysis'] = compact_analysis(latest(run, 'phase4a_analysis')) if run.exists() else None
        chips = latest(run, 'phase4a_chips') if run.exists() else None
        if chips:
            entry['chips'] = chips
        contact = latest(run, 'phase4a_contact') if run.exists() else None
        if contact:
            entry['contact'] = contact
        evidence['runs'][name] = entry
    summary = args.sweep/'summary.json'
    if summary.exists():
        rows = json.loads(summary.read_text())
        evidence['sweep'] = {'summary': str(summary.relative_to(ROOT)), 'device': rows.get('device'),
                             'rows': rows['rows']}
    if args.notes:
        evidence['classification'] = json.loads(args.notes.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=1, default=float))
    print(f'wrote {args.out} ({args.out.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
