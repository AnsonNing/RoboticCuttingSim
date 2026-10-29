#!/usr/bin/env python3
"""Read-only, versioned rescore of finished Phase 1 matrix rows (scripts/run_phase1_matrix.py).

Verdicts under a named criteria version are computed from the recorded row
outputs (``result.json``, ``actions.jsonl``, ``physics_state.npz``) instead of
being adjudicated by hand. Nothing in a row directory is modified; the output
is ONE new JSON file (exclusive create, never overwritten, never inside a row
directory). CPU only; no simulation is run.

Criteria version ``adr0006a1_adr0008``:

* **M2**: ADR 0006 Amendment 1 (``rescore_phase1_oblique.rescore_m2``): two
  substantial parts, interface <= ideal-geometric-label statistic + one
  particle spacing, no far particle in the opposite-side component. The
  original 2 mm verdict is reported next to it.
* **M4 / M6**: ADR 0008 ligament. One component in the final 1-digit stencil
  graph; labels unchanged during the withdrawal (M4) or hold and withdrawal
  (M6), both as labelled count and as identical code arrays in every recorded
  history frame inside the window (anchored on the first frame inside); the
  CORE band |s| <= dx below the stopping depth is 100 % code 0, with s the signed distance of the reference positions to the
  fitted swept plane of digit 0 (``metrics.swept_planes['0'].fit``) and dx the
  4 mm grid spacing. Off-core labels below the stop (lead pre-assignment) are
  reported separately with their count and extent. The law-4 definition (no
  label anywhere below the stop) is reported as ``law4_definition`` and is not
  part of this verdict.
* **M8a / M8b**: two separate fields. ``adr0006_topology``: >= 3 mechanical
  parts in the final 2-digit stencil graph AND the labelled count unchanged
  through the disarmed lift/move (code-array identity over the history frames
  is reported alongside). ``runner_m8b_slice_push``: the runner's
  M8b criterion, ``slice_push_report`` status PASS_SLICE_PUSH (recorded in the
  row result.json). The M8b verdict under this version is topology AND
  slice_push_report, so a slice-push FAIL is never hidden. M8a (Amendment 1
  diagnostic) is scored on topology alone, as in the runner.
* **All other rows**: the runner verdict is carried unchanged.

ADR-level aggregation: M1 = M1a and M1b; M3 = M3a and M3b and M3c (as in
``matrix_law5_adjudication_001``); M5 = M5a and M5b and M5c; M8 = M8b
(Amendment 1 acceptance row; M8a is a reported diagnostic); M9 = M9a and M9b
(M9c/M9d are report only).

  .venv/bin/python scripts/rescore_phase1_matrix.py --criteria-version adr0006a1_adr0008 --law 5 \\
      --rows artifacts/wp10_claude/phase1_law5/m1a_law5_ingest_001 ... --out docs/evidence/phase1/matrix_law5_002.json
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from robotic_cutting_sim.cut_topology import mechanical_components  # noqa: E402
from rescore_phase1_oblique import rescore_m2, swept_fit  # noqa: E402

ORDER = ['M1a', 'M1b', 'M2', 'M3a', 'M3b', 'M3c', 'M4', 'M5a', 'M5b', 'M5c', 'M6', 'M7', 'M8a', 'M8b',
         'M9a', 'M9b', 'M9c', 'M9d']
GRID_DX_M = .004
CRITERIA_VERSIONS = {
    'adr0006a1_adr0008': {
        'M2': 'ADR 0006 Amendment 1: two substantial parts (>= 40% each, >= 99% together) AND actual swept-plane '
              'interface <= ideal-geometric-label interface + one particle spacing AND no particle with |s| > 2 '
              'particle spacings in the opposite-side substantial component (scripts/rescore_phase1_oblique.py)',
        'M4': 'ADR 0008 ligament: one component (final 1-digit stencil graph) AND labels unchanged during the armed '
              'withdrawal AND the core band |s| <= dx (s: reference signed distance to the fitted digit-0 swept '
              'plane, dx = 4 mm) below the stopping depth is 100% code 0; off-core lead labels and the law-4 '
              'all-material definition are reported, not scored',
        'M6': 'M4 criterion, with labels unchanged during the armed hold AND the armed withdrawal',
        'M8a': 'Amendment 1 diagnostic: ADR 0006 topology (>= 3 mechanical parts in the final 2-digit stencil graph '
               'AND labelled count unchanged through the disarmed lift/move)',
        'M8b': 'Amendment 1 acceptance row: ADR 0006 topology (>= 3 mechanical parts in the final 2-digit stencil '
               'graph AND labelled count unchanged through the disarmed lift/move) AND the runner M8b criterion '
               '(slice_push_report status PASS_SLICE_PUSH); both reported as separate fields',
        'other': 'Runner verdict carried unchanged'}}
LIFT_MOVE_PHASES = {'M8a': ('end_0', 'lift_0', 'move_1', 'start_1'), 'M8b': ('hold_0', 'lift_0', 'move_1')}
AGGREGATE = {'M1': ('M1a', 'M1b'), 'M2': ('M2',), 'M3': ('M3a', 'M3b', 'M3c'), 'M4': ('M4',),
             'M5': ('M5a', 'M5b', 'M5c'), 'M6': ('M6',), 'M7': ('M7',), 'M8': ('M8b',), 'M9': ('M9a', 'M9b')}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def load_state(path, keys):
    with np.load(path) as npz:
        missing = [k for k in keys if k not in npz.files]
        if missing:
            raise ValueError(f'{path} lacks {missing}')
        return {k: npz[k] for k in keys}


def load_steps(row_dir):
    with (row_dir / 'actions.jsonl').open() as stream:
        return [json.loads(line) for line in stream]


def phase_window(steps, phases):
    """Labelled-count change across the control steps of ``phases`` (one contiguous block).

    Same definition as ``run_phase1_matrix.phase_label_changes``; re-implemented
    here so that the rescore does not import the episode runner (and with it the
    scripted policies).
    """
    indices = [i for i, s in enumerate(steps) if s.get('phase') in phases]
    if not indices:
        return {'phases': list(phases), 'steps': 0, 'status': 'NOT_RUN'}
    first, last = indices[0], indices[-1]
    if last - first + 1 != len(indices):
        raise ValueError(f'Phases {phases} are not one contiguous block of control steps')
    before = steps[first - 1]['labelled_count'] if first > 0 else 0
    counts = [steps[i]['labelled_count'] for i in range(first, last + 1)]
    dt = steps[first]['time_s'] - (steps[first - 1]['time_s'] if first > 0 else 0.)
    return {'phases': list(phases), 'steps': last - first + 1,
            'time_s': [float(steps[first]['time_s'] - dt), float(steps[last]['time_s'])],
            'labelled_before': int(before), 'labelled_after': int(counts[-1]), 'labelled_max': int(max(counts)),
            'labelled_min': int(min(counts)),
            'change': int(max(abs(max(counts) - before), abs(min(counts) - before))),
            'stroke_active_steps': int(sum(steps[i]['stroke'] != 0 for i in range(first, last + 1)))}


def frames_identical(data, window):
    """Code-array identity over the recorded history frames inside a label window.

    History frames are strided, so the last frame before the window can predate
    the end of the previous (labelling) phase. The anchor is therefore the first
    frame inside the window; its labelled count must equal the step-record count
    at the window start, and every later frame inside must equal it exactly.
    """
    if window.get('steps', 0) == 0:
        return {'status': 'NOT_RUN'}
    time_s, codes = np.asarray(data['time_s'], float), np.asarray(data['codes'])
    start, end = window['time_s']
    inside = np.flatnonzero((time_s > start + 1e-9) & (time_s <= end + 1e-9))
    if not len(inside):
        return {'status': 'NOT_RUN', 'frames_inside': 0}
    anchor = codes[inside[0]]
    anchor_count = int(np.count_nonzero(anchor))
    changed = [int(np.count_nonzero(codes[i] != anchor)) for i in inside[1:]] or [0]
    return {'status': 'FIT', 'anchor_frame_time_s': float(time_s[inside[0]]), 'frames_inside': int(len(inside)),
            'anchor_labelled_count': anchor_count, 'step_record_labelled_before': window['labelled_before'],
            'max_particles_changed_vs_anchor': int(max(changed)),
            'identical': anchor_count == window['labelled_before'] and max(changed) == 0}


def stencil(data):
    return (np.asarray(data['reference_stencil_nodes']), np.asarray(data['reference_stencil_weights']),
            np.asarray(data['reference_stencil_gradients']))


STATE_KEYS = ('time_s', 'codes', 'mass_kg', 'reference_positions', 'reference_stencil_nodes',
              'reference_stencil_weights', 'reference_stencil_gradients')


def score_ligament(row, row_dir, result, inputs):
    """ADR 0008 ligament score for M4/M6."""
    from robotic_cutting_sim.tasks import make_task
    plane = swept_fit(result)
    if plane is None:
        return 'NOT_RUN', {}, {'reason': "No fitted swept plane in metrics.swept_planes['0']"}
    npz = row_dir / 'physics_state.npz'
    inputs['physics_state.npz'] = sha256(npz)
    inputs['actions.jsonl'] = sha256(row_dir / 'actions.jsonl')
    data, steps = load_state(npz, STATE_KEYS), load_steps(row_dir)
    task = make_task(result['task_id'], seed=int(result.get('seed', 42)))
    descent = [s for s in steps if s.get('phase') == 'descend']
    if not descent:
        return 'NOT_RUN', {}, {'reason': 'No descend phase in actions.jsonl'}
    stop_z = float(task.points_to_local(np.asarray([descent[-1]['edge_pose_world'][:3]], float))[0, 2])
    codes = np.asarray(data['codes'])[-1]
    reference = np.asarray(data['reference_positions'], float)
    masses = np.asarray(data['mass_kg'], float)
    origin, normal = plane
    signed = (reference - origin) @ normal
    local = task.points_to_local(reference)
    below = local[:, 2] < stop_z
    core = below & (np.abs(signed) <= GRID_DX_M)
    labelled = codes != 0
    off_core = below & ~core & labelled
    nodes, weights, gradients = stencil(data)
    count, _ = mechanical_components(nodes, weights, codes, 1, gradients=gradients)
    windows = {'withdraw': phase_window(steps, ('withdraw',))}
    if row == 'M6':
        windows = {'hold': phase_window(steps, ('hold',)), **windows}
    identical = {name: frames_identical(data, window) for name, window in windows.items()}
    checks = {'one_component': int(count) == 1}
    for name, window in windows.items():
        checks[f'labels_unchanged_during_{name}'] = (window.get('change') == 0
                                                       and identical[name].get('identical', True) is True)
    checks['core_band_below_stop_all_code0'] = bool(core.any() and not np.any(labelled[core]))
    metrics = {
        'final_mechanical_components_1digit': int(count),
        'stop_local_z_m': stop_z, 'swept_plane': {'origin': origin.tolist(), 'normal': normal.tolist(),
                                                  'source': "result.json metrics.swept_planes['0'].fit"},
        'core_half_width_m': GRID_DX_M,
        'core_band_below_stop_particles': int(core.sum()),
        'core_band_below_stop_code0': int(np.count_nonzero(core & ~labelled)),
        'below_stop_particles': int(below.sum()),
        'off_core_lead_labels_below_stop': {
            'count': int(off_core.sum()),
            'particle_fraction_of_below_stop': float(off_core.sum() / max(int(below.sum()), 1)),
            'mass_fraction_of_below_stop': float(masses[off_core].sum() / masses[below].sum()) if below.any() else None,
            'min_abs_s_m': float(np.abs(signed[off_core]).min()) if off_core.any() else None,
            'max_abs_s_m': float(np.abs(signed[off_core]).max()) if off_core.any() else None,
            'deepest_reference_local_z_m': float(local[off_core, 2].min()) if off_core.any() else None,
            'side_counts_code1_code2': [int(np.count_nonzero(off_core & (codes == 1))),
                                        int(np.count_nonzero(off_core & (codes == 2)))],
            'scope': 'Reported, not scored: ADR 0008 lead pre-assignment labels only |s| > dx by design'},
        'label_windows': windows, 'label_windows_code_frames': identical,
        'law4_definition': {
            'criterion': 'Law-4 runner ligament: no labelled particle anywhere below the stopping depth',
            'labelled_below_stop': int(np.count_nonzero(labelled & below)),
            'verdict': 'PASS' if not np.any(labelled & below) else 'FAIL',
            'scope': 'Reported, not part of the adr0006a1_adr0008 verdict'},
        'final_labelled_count': int(labelled.sum())}
    return ('PASS' if all(checks.values()) else 'FAIL'), checks, metrics


def score_m2(row_dir, result, inputs):
    plane = swept_fit(result)
    if plane is None:
        return 'NOT_RUN', {}, {'reason': "No fitted swept plane in metrics.swept_planes['0']"}
    npz = row_dir / 'physics_state.npz'
    inputs['physics_state.npz'] = sha256(npz)
    data = load_state(npz, STATE_KEYS)
    spacing = float(result.get('particle_spacing_m') or .001)
    scored = rescore_m2(data, *plane, spacing, cuts=int((result.get('context') or {}).get('max_cuts') or 1))
    metrics = {**scored['metrics'], 'original_2mm': scored['original_2mm'],
               'swept_plane': {'origin': plane[0].tolist(), 'normal': plane[1].tolist(),
                               'source': "result.json metrics.swept_planes['0'].fit"}}
    for key in ('actual_interface', 'ideal_interface'):  # full interface reports stay in the row/rescore files
        metrics[key] = {k: v for k, v in metrics[key].items() if not isinstance(v, (list, dict))}
    return scored['verdict'], scored['checks'], metrics


def score_m8(row, row_dir, result, inputs):
    npz = row_dir / 'physics_state.npz'
    inputs['physics_state.npz'] = sha256(npz)
    inputs['actions.jsonl'] = sha256(row_dir / 'actions.jsonl')
    data, steps = load_state(npz, STATE_KEYS), load_steps(row_dir)
    codes = np.asarray(data['codes'])[-1]
    masses = np.asarray(data['mass_kg'], float)
    nodes, weights, gradients = stencil(data)
    count, labels = mechanical_components(nodes, weights, codes, 2, gradients=gradients)
    fractions = np.sort(np.bincount(labels, weights=masses, minlength=count) / masses.sum())[::-1]
    window = phase_window(steps, LIFT_MOVE_PHASES[row])
    topology_checks = {'three_mechanical_parts_2digit_graph': int(count) >= 3,
                       'lift_move_labelled_count_unchanged': window.get('change') == 0}
    topology = {'criterion': CRITERIA_VERSIONS['adr0006a1_adr0008']['M8a'].split(': ', 1)[1],
                'verdict': 'PASS' if all(topology_checks.values()) else 'FAIL', 'checks': topology_checks,
                'final_mechanical_components_2digit': int(count), 'final_component_mass_fractions': fractions[:6].tolist(),
                'lift_move_window': window, 'lift_move_code_frames': frames_identical(data, window)}
    metrics = {'adr0006_topology': topology}
    if row == 'M8a':
        multi = (result.get('metrics') or {}).get('multi_cut') or {}
        metrics['runner_multi_cut_checks_reported'] = multi.get('checks')
        return topology['verdict'], {f'adr0006_topology_{k}': v for k, v in topology_checks.items()}, metrics
    report = (result.get('metrics') or {}).get('slice_push_report') or {}
    multi = report.get('multi_cut') or {}
    failed = {k: v for k, v in (multi.get('checks') or {}).items() if not v}
    post = report.get('post_cut_manipulation') or {}
    failed.update({f'post_cut_manipulation.{k}': v for k, v in (post.get('checks') or {}).items() if not v})
    slice_push = {'criterion': 'Runner M8b criterion: slice_push_report status PASS_SLICE_PUSH (from row result.json)',
                  'status': report.get('status'), 'verdict': 'PASS' if report.get('status') == 'PASS_SLICE_PUSH' else 'FAIL',
                  'failed_checks': failed, 'multi_cut_cuts': (multi.get('metrics') or {}).get('cuts'),
                  'multi_cut_part_mass_fractions': (multi.get('metrics') or {}).get('part_mass_fractions')}
    metrics['runner_m8b_slice_push'] = slice_push
    metrics['verdict_rule'] = ('Under adr0006a1_adr0008 the M8b verdict is adr0006_topology AND runner_m8b_slice_push; '
                               'a slice-push FAIL is a row FAIL and is not hidden by the topology PASS')
    checks = {'adr0006_topology': topology['verdict'] == 'PASS', 'runner_m8b_slice_push': slice_push['verdict'] == 'PASS'}
    return ('PASS' if all(checks.values()) else 'FAIL'), checks, metrics


def rescore_row(row_dir, criteria_version):
    row_dir = Path(row_dir)
    result_path = row_dir / 'result.json'
    result = json.loads(result_path.read_text())
    row, runner = result['row'], result['verdict']
    inputs = {'result.json': sha256(result_path)}
    entry = {'row': row, 'source_dir': str(row_dir), 'runner_verdict': runner, 'runner_checks': result.get('checks'),
             'runner_criterion': result.get('criterion'), 'cutting_law': result.get('cutting_law'),
             'formal_resolution': result.get('formal_resolution'), 'row_git_sha': result.get('git_sha')}
    if runner == 'ERROR':
        return {**entry, 'computed_verdict': 'ERROR', 'basis': 'carried_runner_error', 'inputs_sha256': inputs}
    if row == 'M2':
        verdict, checks, metrics = score_m2(row_dir, result, inputs)
    elif row in ('M4', 'M6'):
        verdict, checks, metrics = score_ligament(row, row_dir, result, inputs)
    elif row in ('M8a', 'M8b'):
        verdict, checks, metrics = score_m8(row, row_dir, result, inputs)
    else:
        return {**entry, 'computed_verdict': runner, 'basis': 'carried_runner_verdict',
                'criterion': CRITERIA_VERSIONS[criteria_version]['other'], 'inputs_sha256': inputs}
    if verdict == 'PASS' and result.get('formal_resolution') is False:
        verdict = 'PASS_NONFORMAL'
    return {**entry, 'computed_verdict': verdict, 'basis': 'computed', 'criterion': CRITERIA_VERSIONS[criteria_version][row],
            'checks': {k: bool(v) for k, v in checks.items()}, 'metrics': metrics, 'inputs_sha256': inputs}


def aggregate(rows):
    by_row = {r['row']: r for r in rows}
    out = {}
    for name, members in AGGREGATE.items():
        present = [by_row[m] for m in members if m in by_row]
        if len(present) != len(members):
            out[name] = {'verdict': 'NOT_RUN', 'members': list(members),
                         'missing': [m for m in members if m not in by_row]}
            continue
        verdicts = {r['row']: r['computed_verdict'] for r in present}
        out[name] = {'verdict': 'PASS' if all(v == 'PASS' for v in verdicts.values()) else 'FAIL',
                     'members': verdicts}
    if 'M8a' in by_row:
        out['M8']['diagnostic_M8a'] = by_row['M8a']['computed_verdict']
    out['M9']['report_only'] = {m: by_row[m]['computed_verdict'] for m in ('M9c', 'M9d') if m in by_row}
    return out


def compare_adjudication(path, aggregated):
    adjudication = json.loads(Path(path).read_text())
    diffs = {}
    for name, value in aggregated.items():
        adjudicated = (adjudication.get('rows') or {}).get(name, {}).get('verdict')
        diffs[name] = {'adjudicated': adjudicated, 'computed': value['verdict'],
                       'differs': adjudicated != value['verdict']}
    return {'source': str(path), 'inputs_sha256': sha256(path), 'rows': diffs,
            'differing_rows': [k for k, v in diffs.items() if v['differs']]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--criteria-version', required=True, choices=sorted(CRITERIA_VERSIONS))
    p.add_argument('--law', type=int, required=True, help='Cutting law every row must record')
    p.add_argument('--rows', type=Path, nargs='+', required=True, help='Finished run_phase1_matrix.py row directories')
    p.add_argument('--out', type=Path, required=True, help='New JSON file (never overwritten)')
    p.add_argument('--adjudication', type=Path, help='Optional hand adjudication to compare the computed verdicts with')
    a = p.parse_args(argv)
    out = a.out.resolve()
    if out.exists():
        p.error(f'{out} exists; refusing to overwrite')
    for directory in a.rows:
        if out.is_relative_to(directory.resolve()):
            p.error('--out must lie outside the read-only row directories')
    rows = []
    for directory in a.rows:
        entry = rescore_row(directory, a.criteria_version)
        if entry['cutting_law'] != a.law:
            raise SystemExit(f"{directory} is law {entry['cutting_law']}, not {a.law}")
        rows.append(entry)
        print(f"{entry['row']:4s} runner {entry['runner_verdict']:12s} computed {entry['computed_verdict']:12s} "
              f"{entry['basis']}  {directory}", flush=True)
    rows.sort(key=lambda r: ORDER.index(r['row']) if r['row'] in ORDER else len(ORDER))
    aggregated = aggregate(rows)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    counts = {}
    for row in rows:
        counts[row['computed_verdict']] = counts.get(row['computed_verdict'], 0) + 1
    evidence = {
        'protocol': 'phase1_acceptance_matrix_rescore', 'criteria_version': a.criteria_version,
        'criteria': CRITERIA_VERSIONS[a.criteria_version],
        'aggregation': {k: list(v) for k, v in AGGREGATE.items()},
        'aggregation_note': ('M3 = M3a and M3b and M3c (as in matrix_law5_adjudication_001); M8 = M8b, the Amendment 1 '
                             'acceptance row, with M8a a reported diagnostic; M9c/M9d are report only'),
        'adrs': ['docs/adr/0006-swept-band-cutting-law-4.md (incl. Amendment 1)',
                 'docs/adr/0008-cutting-law-5-lead-preassignment.md',
                 *(['docs/adr/0010-cutting-law-6-deferred-lead-commit.md'] if a.law == 6 else []),
                 *(['docs/adr/0011-cutting-law-7-eq8-coupled-flip.md'] if a.law == 7 else [])],
        'cutting_law': a.law, 'created': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'command': [sys.argv[0], *(sys.argv[1:] if argv is None else argv)], 'git_sha': sha,
        'git_dirty_paths': status, 'script_sha256': sha256(__file__),
        'read_only': 'Row directories are only read; no row result.json is rewritten; no simulation is run',
        'scope': ('Kinematic ideal tracking and material-only replays; no robot, grasp, slip or camera evidence. '
                  'Law parameters are provisional and uncalibrated.'),
        'computed_verdict_counts': counts, 'adr_rows': aggregated,
        'all_acceptance_rows_pass': all(v['verdict'] == 'PASS' for v in aggregated.values()),
        'rows': rows}
    if a.law == 4:
        evidence['law4_note'] = ('Amendment 1 and ADR 0008 were declared after the law-4 rows ran. These law-4 verdicts '
                                 'are a post-hoc rescore under the later criteria for comparison; the law-4 record of '
                                 'matrix_law4_001.json (runner verdicts) stays the law-4 evidence.')
    if a.adjudication is not None:
        evidence['comparison_to_adjudication'] = compare_adjudication(a.adjudication, aggregated)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:  # exclusive create: never overwrite
        json.dump(jsonable(evidence), stream, indent=1, allow_nan=False)
        stream.write('\n')
    for name, value in aggregated.items():
        print(f'{name}: {value["verdict"]}', flush=True)
    if 'comparison_to_adjudication' in evidence:
        print('differs from adjudication:', evidence['comparison_to_adjudication']['differing_rows'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
