#!/usr/bin/env python3
"""WP-12 Phase 0: assemble docs/evidence/wp12/phase0_baseline_001.json from the baseline extracts (read-only).

  .venv/bin/python scripts/wp12_phase0_summarize.py --scenes artifacts/wp12/baseline/scenes_002.json \
      --pytest-log artifacts/wp12/logs/phase0_pytest_full_001.log --freeze artifacts/wp12/logs/phase0_freeze_001.json \
      --out docs/evidence/wp12/phase0_baseline_001.json

``--scenes`` lists {scene, run_dir, extract, frozen_dir, protocol, command_note}. Refuses to overwrite ``--out``.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
AUDIT_FILES = [
    'README.md', 'docs/PROGRESS.md', 'docs/WP11_PLAN.md', 'docs/SURFACE_GEOMETRY_VALIDATION.md', 'docs/POLICY_READY_API.md',
    'docs/reports/WP11_PHASE_F_LAW8.md', 'docs/reports/WP11_PHASE_E_ROUND1.md', 'docs/reports/WP11_PHASE_E_ROUND2.md',
    'docs/reports/WP11_PHASE_H_PROFILING.md', 'docs/adr/0012-cutting-law-8-plow-gated-lead.md',
    'docs/adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md', 'docs/adr/0014-scalable-cut-history-sparse-logical-fields.md',
    'docs/adr/0015-scene-goal-controller-evaluation-separation.md', 'src/robotic_cutting_sim/cutting_surface.py',
    'src/robotic_cutting_sim/usd_surface_bridge.py', 'src/robotic_cutting_sim/cut_topology.py',
    'src/robotic_cutting_sim/tlmpm_v4.py', 'src/robotic_cutting_sim/tlmpm_v5.py', 'src/robotic_cutting_sim/scene_config.py',
    'src/robotic_cutting_sim/native_fixture.py', 'src/robotic_cutting_sim/fixture_contacts.py',
    'src/robotic_cutting_sim/isaac_sim_runner.py', 'src/robotic_cutting_sim/live_env.py',
    'src/robotic_cutting_sim/live_material.py', 'src/robotic_cutting_sim/task_scenarios.py', 'src/robotic_cutting_sim/workcell.py',
    'configs/wp11_scenarios_v1.json']


def pytest_summary(path):
    text = Path(path).read_text(errors='replace')
    tail = [line for line in text.splitlines() if re.search(r'\d+ (passed|failed)', line)]
    failed = [line for line in text.splitlines() if line.startswith(('FAILED ', 'ERROR ', 'SKIPPED '))]
    wall = re.search(r'Elapsed \(wall clock\) time.*: (\S+)', text)
    exit_ = re.search(r'EXIT=(\d+)', text)
    return {'log': str(path), 'summary_line': tail[-1].strip('= ') if tail else None, 'failed_or_error_lines': failed,
            'wall_clock': wall.group(1) if wall else None, 'exit_code': int(exit_.group(1)) if exit_ else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', required=True)
    p.add_argument('--pytest-log', required=True, nargs='+')
    p.add_argument('--freeze', required=True)
    p.add_argument('--failures', help='JSON list of existing failure cases')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = ROOT/a.out
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    scenes = []
    for row in json.loads((ROOT/a.scenes).read_text()):
        entry = dict(row)
        extract = ROOT/row['run_dir']/row['extract'] if row.get('extract') else None
        if extract is not None and (extract/'baseline_summary.json').exists():
            summary = json.loads((extract/'baseline_summary.json').read_text())
            metrics = summary['surface_metrics_v1']
            keep = ('cut_links', 'cut_face_vertices', 'core_cut_face_vertices', 'distance_plane_coverage', 'status',
                    'distance_to_side_locus_plane_m', 'distance_to_side_locus_plane_core_m', 'distance_to_side_locus_sample_m',
                    'distance_to_link_midpoint_sample_m', 'distance_to_link_midpoint_plane_m', 'normal_deviation_deg',
                    'normal_deviation_core_deg', 'roughness_mesh_rms_m', 'roughness_side_locus_rms_m', 'patch_radius_m',
                    'normal_radius_m', 'core_exclusion_layers', 'deformation_map')
            entry.update(verdict=summary['verdict'], frozen=summary['frozen_comparison'], wall_s=summary['wall_s'],
                         git=summary['git'], config=summary['config'], particles=summary['particles'],
                         final_components=summary['final_components'],
                         final_component_mass_fractions=summary['final_component_mass_fractions'],
                         diagnostics=summary['diagnostics'], cut_locus_per_digit=summary['cut_locus_per_digit'],
                         surface={k: summary['surface'][k] for k in ('profile', 'policy', 'raise_error', 'extract_wall_s',
                                                                    'mesh_components_by_ranges', 'validity_final',
                                                                    'validity_initial')},
                         surface_metrics_v1={k: metrics[k] for k in keep if k in metrics},
                         files={'extract': str(row['run_dir'])+'/'+row['extract'], **summary['files']})
        scenes.append(entry)
    files = {path: (sum(1 for _ in (ROOT/path).open(errors='replace')) if (ROOT/path).exists() else 'MISSING')
             for path in AUDIT_FILES}
    evidence = {'schema': 'wp12_phase0_baseline_v1', 'label': 'kinematic ideal-tracking baselines; provisional uncalibrated material',
                'freeze': json.loads((ROOT/a.freeze).read_text()), 'cpu_suite': [pytest_summary(ROOT/path) for path in a.pytest_log],
                'audit_files_line_counts': files, 'scenes': scenes,
                'existing_failures': json.loads((ROOT/a.failures).read_text()) if a.failures else None,
                'metric_module': 'src/robotic_cutting_sim/surface_metrics_v1.py (tests/test_surface_metrics_v1.py)',
                'tools': ['scripts/wp12_baseline_capture.py', 'scripts/wp12_baseline_extract.py', 'scripts/wp12_phase0_summarize.py']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=1))
    print(out, out.stat().st_size)


if __name__ == '__main__':
    main()
