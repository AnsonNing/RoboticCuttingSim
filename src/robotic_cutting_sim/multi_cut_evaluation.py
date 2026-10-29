"""Read-only multi-cut outcome evaluation, independent of the frozen v1 oracle."""
from dataclasses import dataclass
import numpy as np
from .cut_topology import (PACKED_MAX_CUTS, decode_digits, mechanical_components, mechanical_components_packed,
                           validate_packed)


@dataclass(frozen=True)
class MultiCutGoal:
    planes: tuple
    minimum_part_mass_fraction: float = .05
    minimum_main_parts_mass_fraction: float = .95
    plane_error_bound_m: float = .0015

    def __post_init__(self):
        if len(self.planes) != 2 or not 0 < self.minimum_part_mass_fraction < 1 / 3:
            raise ValueError('Declare two cut planes and a meaningful substantial-part mass bound')
        if not 0 < self.minimum_main_parts_mass_fraction <= 1 or self.plane_error_bound_m <= 0:
            raise ValueError('Multi-cut bounds must be positive')
        for origin, normal in self.planes:
            if (np.asarray(origin).shape != (3,) or np.asarray(normal).shape != (3,)
                    or not np.isfinite([origin, normal]).all() or not np.isclose(np.linalg.norm(normal), 1.)):
                raise ValueError('Cut planes need finite origins and unit normals')


def reference_bonds(lattice):
    lattice = np.asarray(lattice)
    lookup = {tuple(point): index for index, point in enumerate(lattice)}
    if len(lookup) != len(lattice):
        raise ValueError('Reference lattice indices must be unique')
    pairs = [(i, lookup[tuple(point + axis)]) for i, point in enumerate(lattice)
             for axis in np.eye(3, dtype=int) if tuple(point + axis) in lookup]
    return np.asarray(pairs, int).reshape(-1, 2)


def evaluate_multi_cut(data, goal):
    if str(np.asarray(data.get('code_encoding', 'ternary'))) == 'packed2':
        # tlmpm_v5 packed history (ADR 0014): same two-plane goal, packed evaluator.
        return evaluate_multi_cut_packed(data, MultiCutGoalK(
            planes=goal.planes, expected_parts=3, minimum_part_mass_fraction=goal.minimum_part_mass_fraction,
            minimum_main_parts_mass_fraction=goal.minimum_main_parts_mass_fraction,
            plane_error_bound_m=goal.plane_error_bound_m))
    codes, masses = np.asarray(data['codes']), np.asarray(data['mass_kg'])
    if codes.ndim == 2:
        codes = codes[-1]
    points = np.asarray(data['reference_positions'])
    if (codes.shape != (len(masses),) or not np.issubdtype(codes.dtype, np.integer)
            or np.any(codes < 0) or np.any(codes >= 9) or np.any(masses <= 0)
            or points.shape != (len(masses), 3) or not np.isfinite(masses).all()
            or not np.isfinite(points).all()):
        raise ValueError('Invalid actual two-digit material state')
    nodes, weights = np.asarray(data['reference_stencil_nodes']), np.asarray(data['reference_stencil_weights'])
    gradients = np.asarray(data['reference_stencil_gradients'])
    if (nodes.shape != weights.shape or nodes.shape[0] != len(masses)
            or gradients.shape != (*nodes.shape, 3) or not np.isfinite(weights).all()
            or not np.isfinite(gradients).all()):
        raise ValueError('Actual complete reference transfer stencils are required')
    count, ids = mechanical_components(nodes, weights, codes, 2, gradients=gradients)
    fractions = np.sort(np.bincount(ids, weights=masses) / masses.sum())[::-1]
    bonds = reference_bonds(data['reference_indices'])
    a, b = bonds.T if len(bonds) else (np.array([], int), np.array([], int))
    cuts = []
    for digit, (origin, normal) in enumerate(goal.planes):
        left, right = (codes[a] // 3 ** digit) % 3, (codes[b] // 3 ** digit) % 3
        severed = ((left == 1) & (right == 2)) | ((left == 2) & (right == 1))
        midpoints = (points[a[severed]] + points[b[severed]]) / 2
        error = np.abs((midpoints - origin) @ normal)
        cuts.append(dict(digit=digit, severed_bonds=int(severed.sum()),
                         maximum_reference_plane_error_m=float(error.max()) if len(error) else None,
                         p95_reference_plane_error_m=float(np.percentile(error, 95)) if len(error) else None))
    checks = dict(three_mechanical_parts=count >= 3,
                  three_substantial_parts=int(np.sum(fractions >= goal.minimum_part_mass_fraction)) >= 3,
                  main_parts_retain_mass=float(fractions[:3].sum()) >= goal.minimum_main_parts_mass_fraction,
                  both_digits_sever_bonds=all(cut['severed_bonds'] > 0 for cut in cuts),
                  declared_cut_planes=all(cut['maximum_reference_plane_error_m'] is not None
                      and cut['maximum_reference_plane_error_m'] <= goal.plane_error_bound_m for cut in cuts))
    return dict(protocol='multi_cut_outcome_v1', status='PASS_MULTI_CUT_OUTCOME' if all(checks.values()) else 'FAIL_MULTI_CUT_OUTCOME',
                checks={key: bool(value) for key, value in checks.items()},
                metrics=dict(mechanical_components=int(count), part_mass_fractions=fractions.tolist(), cuts=cuts),
                scope='Actual positive-weight/stress-gradient transfer graph with two ternary cut digits; not the v1 task oracle')


# ----------------------------------------------------------------------------- packed codes (ADR 0014, tlmpm_v5)

@dataclass(frozen=True)
class MultiCutGoalK:
    """K declared cut planes (cut index k = plane k) for packed tlmpm_v5 histories.

    ``expected_parts`` defaults to K + 1 (K parallel through-slices); a goal
    with other topology (e.g. crossing cuts) declares it explicitly.
    """
    planes: tuple
    expected_parts: int | None = None
    minimum_part_mass_fraction: float = .05
    minimum_main_parts_mass_fraction: float = .95
    plane_error_bound_m: float = .0015

    def __post_init__(self):
        if not 1 <= len(self.planes) <= PACKED_MAX_CUTS:
            raise ValueError(f'Declare 1..{PACKED_MAX_CUTS} cut planes')
        parts = self.parts
        if not isinstance(parts, (int, np.integer)) or parts < 2 or not 0 < self.minimum_part_mass_fraction < 1 / parts:
            raise ValueError('Expected parts must be >= 2 with a meaningful substantial-part mass bound')
        if not 0 < self.minimum_main_parts_mass_fraction <= 1 or self.plane_error_bound_m <= 0:
            raise ValueError('Multi-cut bounds must be positive')
        for origin, normal in self.planes:
            if (np.asarray(origin).shape != (3,) or np.asarray(normal).shape != (3,)
                    or not np.isfinite([origin, normal]).all() or not np.isclose(np.linalg.norm(normal), 1.)):
                raise ValueError('Cut planes need finite origins and unit normals')

    @property
    def parts(self):
        return len(self.planes) + 1 if self.expected_parts is None else self.expected_parts


def evaluate_multi_cut_packed(data, goal):
    """evaluate_multi_cut for packed codes and K planes (read-only; not the v1 oracle).

    ``data`` holds the final (or history of) packed codes, masses, reference
    positions/indices and the complete reference stencil, as for
    evaluate_multi_cut. ``data['max_cuts']`` (default K = len(goal.planes))
    bounds the valid digits.
    """
    cuts = int(np.asarray(data.get('max_cuts', len(goal.planes))))
    codes, masses = np.asarray(data['codes']), np.asarray(data['mass_kg'])
    if codes.ndim == 2:
        codes = codes[-1]
    points = np.asarray(data['reference_positions'])
    if (cuts < len(goal.planes) or codes.shape != (len(masses),) or not np.issubdtype(codes.dtype, np.integer)
            or not validate_packed(codes, cuts) or np.any(masses <= 0)
            or points.shape != (len(masses), 3) or not np.isfinite(masses).all() or not np.isfinite(points).all()):
        raise ValueError('Invalid actual packed multi-digit material state')
    nodes, weights = np.asarray(data['reference_stencil_nodes']), np.asarray(data['reference_stencil_weights'])
    gradients = np.asarray(data['reference_stencil_gradients'])
    if (nodes.shape != weights.shape or nodes.shape[0] != len(masses)
            or gradients.shape != (*nodes.shape, 3) or not np.isfinite(weights).all()
            or not np.isfinite(gradients).all()):
        raise ValueError('Actual complete reference transfer stencils are required')
    count, ids = mechanical_components_packed(nodes, weights, codes, cuts, gradients=gradients)
    fractions = np.sort(np.bincount(ids, weights=masses) / masses.sum())[::-1]
    bonds = reference_bonds(data['reference_indices'])
    a, b = bonds.T if len(bonds) else (np.array([], int), np.array([], int))
    digits = decode_digits(codes, cuts)
    rows = []
    for digit, (origin, normal) in enumerate(goal.planes):
        left, right = digits[a, digit], digits[b, digit]
        severed = ((left == 1) & (right == 2)) | ((left == 2) & (right == 1))
        midpoints = (points[a[severed]] + points[b[severed]]) / 2
        error = np.abs((midpoints - np.asarray(origin)) @ np.asarray(normal))
        rows.append(dict(digit=digit, labelled_particles=int(np.count_nonzero(digits[:, digit])),
                         severed_bonds=int(severed.sum()),
                         maximum_reference_plane_error_m=float(error.max()) if len(error) else None,
                         p95_reference_plane_error_m=float(np.percentile(error, 95)) if len(error) else None))
    parts = goal.parts
    checks = dict(expected_mechanical_parts=count >= parts,
                  expected_substantial_parts=int(np.sum(fractions >= goal.minimum_part_mass_fraction)) >= parts,
                  main_parts_retain_mass=float(fractions[:parts].sum()) >= goal.minimum_main_parts_mass_fraction,
                  every_digit_severs_bonds=all(row['severed_bonds'] > 0 for row in rows),
                  declared_cut_planes=all(row['maximum_reference_plane_error_m'] is not None
                      and row['maximum_reference_plane_error_m'] <= goal.plane_error_bound_m for row in rows))
    return dict(protocol='multi_cut_outcome_packed_v1',
                status='PASS_MULTI_CUT_OUTCOME' if all(checks.values()) else 'FAIL_MULTI_CUT_OUTCOME',
                checks={key: bool(value) for key, value in checks.items()},
                metrics=dict(mechanical_components=int(count), expected_parts=int(parts),
                             part_mass_fractions=fractions.tolist(), cuts=rows, code_encoding='packed2', max_cuts=cuts),
                scope='Actual positive-weight/stress-gradient transfer graph with packed cut digits (tlmpm_v5); '
                      'not the v1 task oracle')
