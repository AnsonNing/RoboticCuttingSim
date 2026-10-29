#!/usr/bin/env python3
"""ADR 0014 multi-cut physics on tlmpm_v5 at 1 mm (kinematic blade_probe; no robot, no renderer).

Scenarios (law 8, ruler_knife_v1 edge, 1 mm particles, 4 mm grid, dt 1e-4 s):
- ``slices3`` / ``slices5``: a 100 x 24 x 12 mm block lying on the legacy tray
  floor (fully supported: no pedestal gap, no overhang) is cut by K vertical
  through-slices, one cut digit each (x = -25/0/+25 mm, or -40/-20/0/+20/+40 mm),
  each followed by a disarmed lift and move. Test B: K + 1 mechanically
  independent pieces, persistent per-digit history, no labels while disarmed.
- ``slices3_walls`` / ``slices5_walls``: the same with a 120 mm block whose end
  faces touch the tray walls (x = +-60 mm), so the block cannot slide in x
  (slices at -30/0/+30 mm, or -40/-20/0/+20/+40 mm). Added after the free-block
  smoke showed the unclamped block sliding about 1 mm in +x per cut (wedge
  displacement), which moves later cuts off their declared reference planes.
- ``bumps3``: a 100 x 24 x 12 mm bar with three 6 mm domes (radius 9 mm) at
  x = -32/0/+32 mm is skimmed three times at a 25 degree rake, one digit per
  bump, with disarmed lift/move/lower between skims. Test C: each dome removed,
  substrate one body. Skims 0 and 1 run from 20 mm before to 20 mm after their
  dome; the last skim exits the material (to 15 mm past the bar end at +50 mm).
  The first smoke stopped it 2 mm past the end: substrate below a raked edge
  reads as ahead of it until the edge has travelled about 2.1x its depth, so the
  bar-end particles kept digit 0 and bridged dome 2 to the substrate
  (artifacts/wp11/phaseM/smoke_nonformal_bumps3_law8_v5_001, preserved).
Scored by the read-only packed evaluators (cut_topology.mechanical_components_packed,
multi_cut_evaluation.evaluate_multi_cut_packed). Kinematic ideal tracking says
nothing about reachability, grasp or cameras; the material is uncalibrated.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/probe_v5_multicut.py --scenario slices3 --device cuda:0 \
      --out artifacts/wp11/phaseM/slices3_law8_v5_001
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block, drive_blade  # noqa: E402
from robotic_cutting_sim.cut_topology import decode_digits, mechanical_components_packed  # noqa: E402
from robotic_cutting_sim.multi_cut_evaluation import MultiCutGoalK, evaluate_multi_cut_packed  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1  # noqa: E402

TOP, SPACING = .012, .001
CUT_SPEED, MOVE_SPEED = .02, .05
SLICES = {'slices3': ((-.025, 0., .025), .1), 'slices5': ((-.04, -.02, 0., .02, .04), .1),
          'slices3_walls': ((-.03, 0., .03), .12), 'slices5_walls': ((-.04, -.02, 0., .02, .04), .12)}
BUMPS = (-.032, 0., .032)
BAR_END = .05


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


def slice_plan(xs):
    depth = TOP + .004 + .004  # from 4 mm above the top to 4 mm below the block bottom
    phases, kinds = [], []
    for k, x in enumerate(xs):
        phases += [Phase((0, 0, -CUT_SPEED), depth / CUT_SPEED, k), Phase((0, 0, MOVE_SPEED), depth / MOVE_SPEED, -1)]
        kinds += [f'cut_{k}', f'lift_{k}']
        if k + 1 < len(xs):
            phases.append(Phase(((xs[k + 1] - x) / .5, 0, 0), .5, -1))
            kinds.append(f'move_{k}')
    return phases, kinds


def bump_block(spacing=SPACING, density=1500.):
    grid = np.mgrid[-.05:.05 - 1e-9:spacing, -.012:.012 - 1e-9:spacing, 0:.018 - 1e-9:spacing]
    points = grid.reshape(3, -1).T + spacing / 2
    owner = np.full(len(points), -1)
    for k, cx in enumerate(BUMPS):
        dome = (points[:, 2] > TOP) & ((points[:, 2] - TOP) <= .006 * (1 - ((points[:, 0] - cx) ** 2 + points[:, 1] ** 2) / .009 ** 2))
        owner[dome] = k
    keep = (points[:, 2] < TOP) | (owner >= 0)
    return points[keep].astype(np.float32), np.full(int(keep.sum()), density * spacing ** 3, np.float32), owner[keep]


def bump_plan():
    phases, kinds, lift = [], [], .02
    for k, cx in enumerate(BUMPS):
        end = cx + .020 if k + 1 < len(BUMPS) else BAR_END + .015  # the last skim exits the material
        phases.append(Phase((CUT_SPEED, 0, 0), (end - (cx - .020)) / CUT_SPEED, k))
        kinds.append(f'skim_{k}')
        if k + 1 < len(BUMPS):
            travel = (BUMPS[k + 1] - .020) - (cx + .020)
            phases += [Phase((0, 0, lift / .4), .4, -1), Phase((travel / .4, 0, 0), .4, -1), Phase((0, 0, -lift / .4), .4, -1)]
            kinds += [f'lift_{k}', f'move_{k}', f'lower_{k}']
    phases.append(Phase((0, 0, lift / .4), .4, -1))
    kinds.append('lift_end')
    return phases, kinds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenario', choices=(*SLICES, 'bumps3'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--slots-per-node', type=int, default=8)
    a = p.parse_args()
    if 'v5' not in a.out.name or 'law8' not in a.out.name:
        p.error('--out name must contain law8 and v5')
    out = new_output(a.out)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'scripts'], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    write_json(out / 'manifest.json', manifest({**vars(a), 'out': str(a.out)}, 'tlmpm_v5_multicut_probe'))
    start = time.monotonic()
    if a.scenario in SLICES:
        xs, length = SLICES[a.scenario]
        points, masses = block(size_m=(length, .024, .012), spacing=SPACING)
        owner = None
        phases, kinds = slice_plan(xs)
        rotation, origin, cuts = VERTICAL_CUT, [xs[0], 0, TOP + .004], len(xs)
        planes = [((x, 0., 0.), (1., 0., 0.)) for x in xs]
    else:
        points, masses, owner = bump_block()
        phases, kinds = bump_plan()
        rotation, origin, cuts = Rotation.from_euler('y', 25., degrees=True), [BUMPS[0] - .020, 0, TOP + .0002], 3
        planes = None
    result = drive_blade(RULER_KNIFE_V1, rotation, origin, phases, points=points, masses=masses, spacing=SPACING,
                         cuts=cuts, cutting_law=8, solver_version=5, slots_per_node=a.slots_per_node, device=a.device,
                         gpu_graph=a.device.startswith('cuda'), record_history=True)
    state = result['state']
    codes = state['codes']
    nodes, weights, gradients = (result[k] for k in ('reference_stencil_nodes', 'reference_stencil_weights',
                                                     'reference_stencil_gradients'))
    count, labels = mechanical_components_packed(nodes, weights, codes, cuts, gradients=gradients)
    fractions = np.sort(np.bincount(labels, weights=masses) / masses.sum())[::-1]
    digits = decode_digits(codes, cuts)
    labelled = result['labelled_after_phase']
    digit_counts = np.asarray(result['digit_labelled_after_phase'])
    disarmed = [i for i, kind in enumerate(kinds) if not kind.startswith(('cut_', 'skim_'))]
    armed = [i for i, kind in enumerate(kinds) if kind.startswith(('cut_', 'skim_'))]
    history = {f'digit_{k}': {'first_labelled_phase': kinds[armed[k]],
                              'labelled_after_own_phase': int(digit_counts[armed[k], k]),
                              'labelled_final': int(digit_counts[-1, k]),
                              'zero_before_own_phase': bool(np.all(digit_counts[:armed[k], k] == 0)),
                              'persistent_after_own_phase': bool(np.all(digit_counts[armed[k]:, k] == digit_counts[armed[k], k]))}
               for k in range(cuts)}
    disarmed_changes = {kinds[i]: int(labelled[i] - labelled[i - 1]) for i in disarmed}
    report = {'scenario': a.scenario, 'solver_module': result['solver_module'], 'code_encoding': result['code_encoding'],
              'max_cuts': cuts, 'slot_diagnostics': result['slot_diagnostics'], 'particles': int(len(points)),
              'mechanical_components': int(count), 'component_mass_fractions': fractions[:10].tolist(),
              'labelled_after_phase': dict(zip(kinds, labelled)), 'digit_history': history,
              'disarmed_label_changes': disarmed_changes, 'forces_per_phase': dict(zip(kinds, result['forces'])),
              'severed_bonds': result['severed_bonds']}
    lattice = np.round((points - points.min(0)) / SPACING).astype(int)
    if planes is not None:
        data = {'codes': codes, 'mass_kg': masses, 'reference_positions': points, 'reference_indices': lattice,
                'reference_stencil_nodes': nodes, 'reference_stencil_weights': weights,
                'reference_stencil_gradients': gradients, 'max_cuts': cuts}
        evaluation = evaluate_multi_cut_packed(data, MultiCutGoalK(tuple(planes)))
        expected = np.searchsorted(np.asarray(xs), points[:, 0])
        pairs = set(zip(labels.tolist(), expected.tolist()))
        piece_mass = [float(masses[expected == k].sum() / masses.sum()) for k in range(cuts + 1)]
        report.update(multi_cut_evaluation=evaluation, components_equal_declared_slabs=len(pairs) == count == cuts + 1,
                      declared_slab_mass_fractions=piece_mass)
        checks = {'k_plus_one_pieces': count == cuts + 1 and evaluation['checks']['expected_substantial_parts'],
                  'packed_multi_cut_outcome_pass': evaluation['status'] == 'PASS_MULTI_CUT_OUTCOME',
                  'components_equal_declared_slabs': report['components_equal_declared_slabs'],
                  'every_cut_history_persistent': all(h['persistent_after_own_phase'] and h['zero_before_own_phase']
                                                      for h in history.values()),
                  'disarmed_phases_add_no_labels': not any(disarmed_changes.values()),
                  'no_slot_overflow': result['slot_diagnostics']['overflow_entries'] == 0}
    else:
        substrate = points[:, 2] < TOP
        base = int(np.bincount(labels[substrate]).argmax())
        per_bump = []
        for k in range(3):
            bump = owner == k
            per_bump.append({'bump': k, 'center_x_m': BUMPS[k], 'mass_kg': float(masses[bump].sum()),
                             'removed_mass_fraction': float(masses[bump & (labels != base)].sum() / masses[bump].sum()),
                             'own_digit_positive_fraction': float(np.mean(digits[bump, k] == 2)),
                             'own_digit_unassigned': int(np.count_nonzero(digits[bump, k] == 0))})
        substrate_in_base = float(masses[substrate & (labels == base)].sum() / masses[substrate].sum())
        report.update(per_bump=per_bump, substrate_mass_fraction_in_largest_body=substrate_in_base)
        checks = {'every_bump_removed_ge_90pct': all(b['removed_mass_fraction'] >= .9 for b in per_bump),
                  'substrate_one_body_ge_99pct': substrate_in_base >= .99,
                  'every_cut_history_persistent': all(h['persistent_after_own_phase'] and h['zero_before_own_phase']
                                                      for h in history.values()),
                  'disarmed_phases_add_no_labels': not any(disarmed_changes.values()),
                  'no_slot_overflow': result['slot_diagnostics']['overflow_entries'] == 0}
    verdict = 'PASS' if all(checks.values()) else 'FAIL'
    np.savez_compressed(out / 'physics_state.npz', codes=codes[None], positions=state['positions'][None],
                        mass_kg=masses, reference_positions=points, reference_indices=lattice,
                        reference_stencil_nodes=nodes, reference_stencil_weights=weights,
                        reference_stencil_gradients=gradients, code_encoding='packed2', max_cuts=cuts,
                        **({'bump_owner': owner} if owner is not None else {}))
    write_json(out / 'result.json', jsonable({'verdict': verdict, 'checks': checks, 'report': report, 'command': sys.argv,
                                              'git_sha': sha, 'git_dirty_src_scripts': dirty, 'device': a.device,
                                              'cutting_law': 8, 'particle_spacing_m': SPACING, 'grid_dx_m': .004,
                                              'material_dt_s': 1e-4, 'cut_speed_m_s': CUT_SPEED,
                                              'wall_s': time.monotonic() - start,
                                              'scope': 'Kinematic blade_probe on a supported block; uncalibrated material'}))
    print(a.scenario, verdict, json.dumps(jsonable(checks)), flush=True)


if __name__ == '__main__':
    main()
