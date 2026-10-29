#!/usr/bin/env python3
"""Read-only rescore of a finished Phase 1 M2 (oblique cut) row: ADR 0006 Amendment 1.

Reads ``result.json`` and ``physics_state.npz`` of one finished
``scripts/run_phase1_matrix.py --row M2`` output directory and writes ONE new
JSON file (``--out``; an existing file is never overwritten, and the file may
not lie inside the row directory). Nothing in the row directory is modified.

Amended M2 criterion (declared before any law-5 row ran):
  * two substantial parts: >= 40 % mass each and >= 99 % together (unchanged);
  * actual interface statistic <= ideal-geometric-label statistic + one
    particle spacing, where both use the frozen midpoint statistic
    (``demo_evaluation.swept_plane_interface``) measured to the fitted swept
    plane of digit 0 (``metrics.swept_planes['0'].fit`` in result.json). The
    actual statistic uses the final stencil-graph mechanical components; the
    ideal one uses labels = sign of the reference positions' signed distance
    to that plane;
  * no particle farther than two particle spacings from the swept plane sits
    in the substantial component that belongs to the opposite side.
The original <= 2 mm verdict is reported unchanged next to the amended one.

  .venv/bin/python scripts/rescore_phase1_oblique.py --row-dir <finished M2 dir> --out <new file.json>
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.cut_topology import mechanical_components  # noqa: E402
from robotic_cutting_sim.demo_evaluation import swept_plane_interface  # noqa: E402

ORIGINAL_LIMIT_M = .002
CRITERION = ('ADR 0006 Amendment 1 M2: two substantial parts (>= 40% each, >= 99% together) AND actual swept-plane '
             'interface statistic <= ideal-geometric-label statistic + one particle spacing AND no particle with '
             '|signed distance| > 2 particle spacings in the substantial component of the opposite side')


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def swept_fit(result, digit=0):
    """The fitted swept plane (origin, unit normal) of ``digit`` from a matrix result, or None."""
    planes = (result.get('metrics') or {}).get('swept_planes') or {}
    plane = planes.get(str(digit), planes.get(digit))
    if not plane or plane.get('status') != 'FIT' or 'fit' not in plane:
        return None
    fit = plane['fit']
    origin, normal = np.asarray(fit['origin'], float), np.asarray(fit['normal'], float)
    if origin.shape != (3,) or normal.shape != (3,) or not np.isfinite([*origin, *normal]).all() or not np.linalg.norm(normal):
        raise ValueError('The recorded swept-plane fit is malformed')
    return origin, normal/np.linalg.norm(normal)


def rescore_m2(data, origin, normal, spacing_m, cuts=1):
    """Amended M2 verdict from one recording (dict of arrays) and the swept plane."""
    nodes, weights, gradients = (np.asarray(data[k]) for k in
                                 ('reference_stencil_nodes', 'reference_stencil_weights', 'reference_stencil_gradients'))
    reference = np.asarray(data['reference_positions'], float)
    masses = np.asarray(data['mass_kg'], float)
    codes = np.asarray(data['codes'])
    codes = codes[-1] if codes.ndim == 2 else codes
    if reference.shape != (len(masses), 3) or codes.shape != (len(masses),) or nodes.shape[0] != len(masses):
        raise ValueError('Recording arrays are not aligned')
    count, labels = mechanical_components(nodes, weights, codes, cuts, gradients=gradients)
    component_mass = np.bincount(labels, weights=masses, minlength=count)
    order = np.argsort(component_mass)[::-1]
    fractions = component_mass[order]/masses.sum()
    signed = (reference-origin)@normal
    side = (signed > 0).astype(np.int32)
    actual = swept_plane_interface(nodes, weights, gradients, labels, reference, origin, normal)
    ideal = swept_plane_interface(nodes, weights, gradients, side, reference, origin, normal)
    actual_m = actual['maximum_reference_midpoint_plane_distance_m']
    ideal_m = ideal['maximum_reference_midpoint_plane_distance_m']
    substantial = len(fractions) >= 2 and fractions[1] >= .40
    together = len(fractions) >= 2 and float(fractions[:2].sum()) >= .99
    # Side of each of the two largest components: sign of its mass-weighted mean signed distance.
    far = np.abs(signed) > 2*spacing_m
    wrong, wrong_max, sides = None, None, None
    if len(order) >= 2:
        main = [int(c) for c in order[:2]]
        sides = {c: int(np.average(signed[labels == c], weights=masses[labels == c]) > 0) for c in main}
        if len(set(sides.values())) == 2:
            opposite = {1: next(c for c in main if sides[c] == 0), 0: next(c for c in main if sides[c] == 1)}
            wrong_mask = far & (((side == 1) & (labels == opposite[1])) | ((side == 0) & (labels == opposite[0])))
            wrong = int(np.count_nonzero(wrong_mask))
            wrong_max = float(np.abs(signed[wrong_mask]).max()) if wrong else None
    in_minor = far & ~np.isin(labels, order[:2]) if len(order) >= 2 else far
    checks = {'two_substantial_parts_each_ge_40pct': bool(substantial),
              'two_parts_together_ge_99pct': bool(together),
              'interface_le_ideal_plus_one_spacing': actual_m is not None and ideal_m is not None
                  and actual_m <= ideal_m+spacing_m+1e-12,
              'no_far_particle_in_opposite_side_component': wrong == 0}
    verdict = 'PASS' if all(checks.values()) else 'FAIL'
    original = {'two_substantial_parts_each_ge_40pct': bool(substantial),
                'two_parts_together_ge_99pct': bool(together),
                'swept_plane_interface_le_2mm': actual_m is not None and actual_m <= ORIGINAL_LIMIT_M}
    metrics = {'final_mechanical_components': int(count), 'final_component_mass_fractions': fractions[:6].tolist(),
               'actual_interface_m': actual_m, 'ideal_geometric_label_interface_m': ideal_m,
               'allowed_interface_m': None if ideal_m is None else float(ideal_m+spacing_m),
               'particle_spacing_m': float(spacing_m), 'far_threshold_m': float(2*spacing_m),
               'main_component_sides_positive': None if sides is None else {str(k): v for k, v in sides.items()},
               'far_particles_in_opposite_side_component': wrong,
               'far_particles_in_opposite_side_component_max_abs_distance_m': wrong_max,
               'far_particles_in_minor_components': int(np.count_nonzero(in_minor)),
               'label_geometry_disagreements_max_abs_distance_m': _disagreement(labels, order, side, signed),
               'actual_interface': actual, 'ideal_interface': ideal}
    return {'verdict': verdict, 'checks': checks, 'metrics': metrics,
            'original_2mm': {'verdict': 'PASS' if all(original.values()) else 'FAIL', 'checks': original,
                             'limit_m': ORIGINAL_LIMIT_M}}


def _disagreement(labels, order, side, signed):
    """Largest |s| of a particle in a main component whose majority side differs from its own sign."""
    if len(order) < 2:
        return None
    worst = 0.
    for component in order[:2]:
        member = labels == component
        majority = int(np.mean(side[member]) > .5)
        mismatch = member & (side != majority)
        if mismatch.any():
            worst = max(worst, float(np.abs(signed[mismatch]).max()))
    return worst


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row-dir', type=Path, required=True, help='Finished run_phase1_matrix.py --row M2 output directory')
    p.add_argument('--out', type=Path, required=True, help='New JSON file (never overwritten)')
    p.add_argument('--spacing', type=float, help='Particle spacing in m (default: result.json particle_spacing_m)')
    a = p.parse_args(argv)
    row_dir = a.row_dir.resolve()
    out = a.out.resolve()
    if out.exists():
        p.error(f'{out} exists; refusing to overwrite')
    if out.is_relative_to(row_dir):
        p.error('--out must lie outside the read-only row directory')
    result_path, npz_path = row_dir / 'result.json', row_dir / 'physics_state.npz'
    result = json.loads(result_path.read_text())
    if result.get('row') != 'M2':
        p.error(f"{result_path} is row {result.get('row')!r}, not M2")
    spacing = float(a.spacing if a.spacing is not None else result.get('particle_spacing_m') or .001)
    base = {'protocol': 'phase1_m2_rescore_adr0006_amendment1', 'criterion': CRITERION,
            'row_dir': str(row_dir), 'inputs_sha256': {'result.json': sha256(result_path)},
            'command': [sys.argv[0], *(sys.argv[1:] if argv is None else argv)],
            'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'script_sha256': sha256(__file__), 'source_row_verdict': result.get('verdict'),
            'source_row_checks': result.get('checks'), 'cutting_law': result.get('cutting_law'),
            'formal_resolution': result.get('formal_resolution'), 'read_only': True}
    plane = swept_fit(result)
    if plane is None:
        report = {**base, 'verdict': 'NOT_RUN', 'reason': "No fitted swept plane in metrics.swept_planes['0']"}
    elif not npz_path.is_file():
        report = {**base, 'verdict': 'NOT_RUN', 'reason': f'Missing {npz_path}'}
    else:
        base['inputs_sha256']['physics_state.npz'] = sha256(npz_path)
        with np.load(npz_path) as npz:
            data = {key: npz[key] for key in npz.files}
        cuts = int((result.get('context') or {}).get('max_cuts') or 1)
        scored = rescore_m2(data, *plane, spacing, cuts=cuts)
        if scored['verdict'] == 'PASS' and result.get('formal_resolution') is False:
            scored['verdict'] = 'PASS_NONFORMAL'
        report = {**base, **scored, 'swept_plane': {'origin': plane[0].tolist(), 'normal': plane[1].tolist(),
                                                    'source': "result.json metrics.swept_planes['0'].fit"}}
    with out.open('x') as stream:  # exclusive create: never overwrite
        json.dump(jsonable(report), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(report['verdict'], report.get('checks'), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
