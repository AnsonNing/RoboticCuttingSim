#!/usr/bin/env python3
"""WP-12 Phase 4 root-cause verification (methods review §5, rows V1-V4): one declared row per process.

Rows are declared in ``configs/wp12/phase4_verification_protocol_v1.json``. Each row drives one existing, unchanged
scenario driver with two physics-neutral recorders attached to every solver step (``flip_drift_v1``,
``tool_contact_split_v1``; ``root_cause_recorders``):

* ``kind: res``   the resolution-study driver ``scripts/run_wp12_resolution.py`` (CUT / OBL20 / TRIM / MULTI2 on the
                  blade_master_flat_board_v1 geometry, kinematic ideal tracking), surface extraction off;
* ``kind: e1``    the E1 driver ``scripts/run_wp12_phase4_e1.py`` (160 x 30 x 20 mm bar on the board, oblique
                  through-cuts, ``LiveCuttingMaterial``) for the E1b lattice-phase rows and V3.4;
* ``kind: probe`` ``blade_probe.drive_blade`` with the Phase 3 S7 (raked thin-chip skim) and Phase 1 G3b (20 deg yawed
                  vertical cut) geometry, phases and material, copied from ``run_wp12_phase3_surface.py``.

The drivers are imported read-only (private module instances); nothing in them changes. The solver class they
construct is ``tlmpm_v5.TLMPM`` (UNCHANGED_SOLVER rows) or, for rows with a ``counterfactual`` block, the
``tlmpm_v5_counterfactual.CounterfactualTLMPM`` variant, labelled ``COUNTERFACTUAL_NOT_A_GATE``: such a row is a
counterfactual measurement, never acceptance, never a law, never a profile. ``tlmpm_v5.py`` and law 8 are unchanged.

Formal rows need the protocol file committed and unmodified (checked with git) and write under
``artifacts/wp12/phase4_verify_NNN/<row>``; ``--dev`` rows (label DEV, never evidence) write under
``artifacts/wp12/phase4_verify_dev_NNN/<row>`` and may cap the control steps. Output directories are created and
never overwritten.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4_verification.py --row V12_CUT_dt0100 \\
      --device cuda:1 --out artifacts/wp12/phase4_verify_001/V12_CUT_dt0100
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
PROTOCOL = ROOT/'configs/wp12/phase4_verification_protocol_v1.json'
SEED = 42
CONTROL_DT = .02
SOURCES = ('src/robotic_cutting_sim/tlmpm_v5.py', 'src/robotic_cutting_sim/tlmpm_v5_counterfactual.py',
           'src/robotic_cutting_sim/root_cause_recorders.py', 'src/robotic_cutting_sim/root_cause_diagnostics.py',
           'scripts/run_wp12_resolution.py', 'scripts/run_wp12_phase4_e1.py', 'scripts/run_wp12_phase3_surface.py',
           'src/robotic_cutting_sim/blade_probe.py', 'configs/wp12/phase4_verification_protocol_v1.json',
           'scripts/run_wp12_phase4_verification.py')
# Phase 3 / Phase 1 probe constants (run_wp12_phase1_geometry.py: S, DX, DT, CUT_SPEED, MOVE_SPEED, ENERGY)
PROBE_S, PROBE_CUT_SPEED, PROBE_MOVE_SPEED, PROBE_ENERGY = .001, .02, .05, 5.
MATERIAL_RES = dict(young=5e4, poisson=.3, yield_stress=1e3)


def load_script(name):
    """Private module instance of scripts/<name>.py (read-only reuse; the file is not modified)."""
    spec = importlib.util.spec_from_file_location(f'wp12_verify_private_{name}', ROOT/f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol_rows(path=PROTOCOL):
    return {row['id']: row for row in json.loads(Path(path).read_text())['rows']}


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


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def git_state():
    return {'git_sha': git('rev-parse', 'HEAD').stdout.strip(),
            'git_status_porcelain': git('status', '--porcelain').stdout.splitlines(),
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES if (ROOT/p).exists()}}


def protocol_committed(path=PROTOCOL):
    rel = str(Path(path).resolve().relative_to(ROOT))
    tracked = git('ls-files', '--error-unmatch', rel).returncode == 0
    clean = git('diff', '--quiet', 'HEAD', '--', rel).returncode == 0
    return tracked and clean, git('log', '-1', '--format=%H', '--', rel).stdout.strip()


# ----------------------------------------------------------------------------- solver installation

def install(row, split_capacity, holder):
    """Make ``tlmpm_v5.TLMPM`` (as the drivers import it) the unchanged solver or the COUNTERFACTUAL variant, with
    both read-only recorders bracketing every step. Returns the installed class."""
    import warp as wp
    from robotic_cutting_sim import root_cause_recorders as rr
    from robotic_cutting_sim import tlmpm_v5
    original = tlmpm_v5.TLMPM
    options = row.get('counterfactual')
    if options:
        from robotic_cutting_sim.tlmpm_v5_counterfactual import CounterfactualOptions, CounterfactualTLMPM
        options = CounterfactualOptions(**options)
        base = CounterfactualTLMPM
    else:
        base = original

    class Installed(base):
        def __init__(self, *a, **kw):
            if options:
                kw['counterfactual'] = options
            super().__init__(*a, **kw)
            self._rc_drift = rr.FlipDriftRecorder(self)
            self._rc_split = rr.ToolContactSplitRecorder(self, split_capacity)
            self._rc_energy = rr.TensileEnergyRecorder(self)
            if wp.get_device(self.device).is_cuda:
                for module in ('robotic_cutting_sim.tlmpm_v5', 'robotic_cutting_sim.root_cause_recorders',
                               'robotic_cutting_sim.fixture_contacts') + (
                               ('robotic_cutting_sim.tlmpm_v5_counterfactual',) if options else ()):
                    if module in sys.modules:   # fixture_contacts is imported only with custom fixtures
                        wp.load_module(module=module, device=self.device)
            holder['bed'] = self
            holder['constructed'] = holder.get('constructed', 0)+1

        def step(self, advance=True):
            self._rc_split.before_step()
            super().step(advance)
            self._rc_drift.after_step()
            self._rc_energy.after_step()

    Installed.__module__ = base.__module__
    Installed.__qualname__ = Installed.__name__ = f'Recorded{base.__name__}'
    tlmpm_v5.TLMPM = Installed
    holder['original'] = original
    return Installed


def patch_material_dt(dt):
    """E1 rows at a non-default material dt: the E1 driver builds CouplingConfig(material_dt=1e-4); replace it."""
    from robotic_cutting_sim import coupled_robot
    original = coupled_robot.CouplingConfig

    def make(**kw):
        return original(**{**kw, 'material_dt': dt})
    coupled_robot.CouplingConfig = make
    return original


# ----------------------------------------------------------------------------- drivers

def run_res(row, args, out, holder):
    R = load_script('run_wp12_resolution')
    rrow = {k: row[k] for k in ('case', 'spacing_m', 'grid_dx_m', 'dt_s')}
    rrow['variant'] = row.get('variant', 'formal')
    substeps = int(round(CONTROL_DT/row['dt_s']))
    poses, _ = R.plan(rrow['case'], rrow['spacing_m'], rrow['variant'])
    steps = len(poses) if not args.dev_max_steps else min(len(poses), args.dev_max_steps)
    install(row, steps*substeps+16, holder)
    ns = SimpleNamespace(device=args.device, max_steps=args.dev_max_steps, no_surface=True)
    result = R.run(ns, rrow, out)
    return result, {'R': R, 'substeps': substeps}


def run_e1(row, args, out, holder):
    E = load_script('run_wp12_phase4_e1')
    spec = dict(row['e1_spec'])
    dt = row.get('material_dt_s', 1e-4)
    substeps = int(round(CONTROL_DT/dt))
    schedule = E.control_schedule(spec)
    steps = len(schedule)-1 if args.dev_max_steps is None else min(len(schedule)-1, args.dev_max_steps)
    install(row, steps*substeps+16, holder)
    if dt != 1e-4:
        patch_material_dt(dt)
    ns = SimpleNamespace(device=args.device, dev_bar=None, dev_max_steps=args.dev_max_steps)
    result = E.run(spec, ns, out)
    return result, {'E': E, 'substeps': substeps, 'spec': spec}


def probe_case(case, s=PROBE_S):
    """(points, masses, rotation, start, phases, locus planes / heights, chip region rule) of Phase 3 S7 / Phase 1 G3b,
    copied from run_wp12_phase3_surface.case_s7 / case_s1b (1 mm, ruler_knife_v1, legacy tray, G_c 5 J/m^2)."""
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    if case == 'S7':
        top, dome_h, dome_r, edge_above = .010, .004, .009, .002
        grid = np.mgrid[-.04:.04-1e-9:s, -.012:.012-1e-9:s, 0:top+dome_h-1e-9:s]
        pts = grid.reshape(3, -1).T+s/2
        dome = (pts[:, 2] > top) & ((pts[:, 2]-top) <= dome_h*(1-(pts[:, 0]**2+pts[:, 1]**2)/dome_r**2))
        points = pts[(pts[:, 2] < top) | dome].astype(np.float32)
        masses = np.full(len(points), 1500*s**3, np.float32)
        rot = Rotation.from_euler('y', 25., degrees=True)
        x0, x1 = -.018, .026
        phases = [Phase((PROBE_CUT_SPEED, 0, 0), (x1-x0)/PROBE_CUT_SPEED, 0), Phase((0, 0, PROBE_MOVE_SPEED), .4, -1)]
        return points, masses, rot, [x0, 0, top+edge_above], phases, {
            'locus': {'type': 'z', 'z_m': top+edge_above, 'note': 'edge path height (top + 2 mm)'},
            'chip_region': {'type': 'z_above', 'z_m': top}, 'substrate_top_m': top, 'rake_deg': 25.}
    if case == 'G3b':
        top = .012
        points, masses = block(size_m=(.08, .03, top), spacing=s)
        rot = Rotation.from_euler('z', 20., degrees=True)*VERTICAL_CUT
        depth = top+.008
        phases = [Phase((0, 0, -PROBE_CUT_SPEED), depth/PROBE_CUT_SPEED, 0),
                  Phase((0, 0, PROBE_MOVE_SPEED), depth/PROBE_MOVE_SPEED, -1)]
        normal = rot.apply([0., 0., 1.])
        return points, masses, rot, [.0003, .0002, top+.004], phases, {
            'locus': {'type': 'planes', 'planes': [[[.0003, .0002, 0.], normal.tolist()]]}, 'yaw_deg': 20.}
    raise ValueError(case)


def run_probe(row, args, out, holder):
    from robotic_cutting_sim.blade_probe import Phase, drive_blade
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
    points, masses, rot, start, phases, geo = probe_case(row['case'])
    if args.dev_max_steps:
        budget = args.dev_max_steps*CONTROL_DT
        capped = []
        for ph in phases:
            if budget <= 1e-12:
                break
            capped.append(Phase(ph.velocity_m_s, min(ph.duration_s, budget), ph.cut))
            budget -= capped[-1].duration_s
        phases = capped
    total = sum(int(round(p.duration_s/1e-4)) for p in phases)
    install(row, total+16, holder)
    t0 = time.monotonic()
    r = drive_blade(RULER_KNIFE_V1, rot, start, phases, points=points, masses=masses, spacing=PROBE_S,
                    cutting_energy=PROBE_ENERGY, cuts=1, cutting_law=8, solver_version=5, device=args.device,
                    gpu_graph=args.device.startswith('cuda'), record_history=True, cut_events='cut_events_v1')
    wall = time.monotonic()-t0
    st = r['state']
    lattice = np.round((points-points.min(0))/PROBE_S).astype(int)
    result = {'geometry': geo, 'particles': int(len(points)), 'components': r['components'],
              'mass_fractions': r['mass_fractions'][:4], 'labelled_after_phase': r['labelled_after_phase'],
              'forces': r['forces'], 'slot_diagnostics': r['slot_diagnostics'], 'drive_wall_s': wall,
              'cut_events': {k: v for k, v in r['cut_events']['metadata'].items() if not k.endswith('sha256')},
              'phases': [[list(p.velocity_m_s), p.duration_s, p.cut] for p in phases]}
    np.savez_compressed(out/'state.npz', reference=points, lattice=lattice.astype(np.int32), masses=masses,
                        FINAL_positions=st['positions'], FINAL_codes=st['codes'], FINAL_F=st['F'],
                        cut_events_log=r['cut_events']['log'],
                        force_history_N=np.concatenate(r['force_history_N']) if r['force_history_N'] else np.zeros((0, 3)))
    return result, {'points': points, 'lattice': lattice, 'masses': masses, 'geo': geo,
                    'nodes': r['reference_stencil_nodes'], 'weights': r['reference_stencil_weights'],
                    'gradients': r['reference_stencil_gradients']}


# ----------------------------------------------------------------------------- analysis (host, read-only)

def locus_function(case, geometry):
    """Per-point reference distance (m) to the declared cut locus of a resolution case."""
    from robotic_cutting_sim import root_cause_diagnostics as rd
    if case == 'TRIM':
        z = float(geometry['plane_z_m'])
        return lambda P: np.abs(np.asarray(P)[:, 2]-z), {'type': 'z', 'z_m': z}
    planes = [(p['origin'], p['normal']) for p in geometry['planes']]
    return lambda P: rd.plane_distance(P, planes), {'type': 'planes', 'planes': planes}


def momentum_energy(bed):
    v = bed.v.numpy().astype(np.float64)
    m = bed.mass.numpy().astype(np.float64)
    return {'momentum_kg_m_s': (m[:, None]*v).sum(0).tolist(), 'kinetic_energy_j': float(.5*(m*(v**2).sum(1)).sum()),
            'max_speed_m_s': float(np.sqrt((v**2).sum(1)).max())}


def force_split(table, substeps, window, steady):
    from robotic_cutting_sim import root_cause_recorders as rr
    per = rr.split_per_control_step(table, substeps)
    n = min(len(per), len(window))
    per, window, steady = per[:n], np.asarray(window[:n], bool), np.asarray(steady[:n], bool)
    col = rr.SPLIT_COLUMNS.index
    out = {}
    for name, mask in (('window', window), ('steady', steady)):
        if not mask.any():
            out[name] = None
            continue
        sel = per[mask]
        block = {}
        for zone in ('in', 'out'):
            f = sel[:, [col(f'{zone}_d'), col(f'{zone}_e'), col(f'{zone}_n')]]/CONTROL_DT
            block[zone] = {'mean_reaction_along_d_N': float(f[:, 0].mean()),
                           'mean_reaction_along_e_N': float(f[:, 1].mean()),
                           'mean_reaction_along_n_N': float(f[:, 2].mean()),
                           'mean_abs_N': float(np.linalg.norm(f, axis=1).mean()),
                           'contacts_per_substep': float(sel[:, col(f'{zone}_count')].mean()/substeps),
                           'max_penetration_m': float(sel[:, col(f'{zone}_max_penetration_m')].max())}
        tot = (sel[:, [col('in_d'), col('in_e'), col('in_n')]]+sel[:, [col('out_d'), col('out_e'), col('out_n')]])/CONTROL_DT
        block['total_blade_mean_abs_N'] = float(np.linalg.norm(tot, axis=1).mean())
        block['released_mean_along_d_N'] = float(sel[:, col('released_d')].mean()/CONTROL_DT)
        block['in_zone_raw_reaction_along_d_N'] = float(sel[:, col('in_raw_d')].mean()/CONTROL_DT)
        block['out_share_of_total_abs'] = (float(np.linalg.norm(sel[:, [col('out_d'), col('out_e'), col('out_n')]], axis=1).mean()
                                                 / CONTROL_DT/block['total_blade_mean_abs_N'])
                                           if block['total_blade_mean_abs_N'] > 0 else None)
        out[name] = block
    return out, per


def analyse_res(row, result, ctx, out, bed):
    from robotic_cutting_sim import root_cause_diagnostics as rd
    case, s = row['case'], row['spacing_m']
    z = np.load(out/'state.npz')
    steps = np.load(out/'steps.npz')
    X = z['reference'].astype(np.float64)
    lattice = z['lattice']
    geometry = result['geometry']
    dist, locus = locus_function(case, geometry)
    drift = bed._rc_drift.arrays()
    analysis = {'locus': locus}
    states = sorted({k[:-len('_positions')] for k in z.files if k.endswith('_positions')})
    for name in states:
        final = name == 'FINAL'
        analysis[f'links_{name}'] = rd.link_consistency(
            X, lattice, z[f'{name}_positions'], z[f'{name}_F'], z[f'{name}_codes'], s, dist,
            drift=drift['drift_m'] if final else None, assign_substep=drift['assign_substep'] if final else None)
    x = z['FINAL_positions'].astype(np.float64)
    analysis['drift_identity_position_residual_m'] = float(np.abs(x-X-drift['ugrid_m']-drift['drift_m']).max())
    analysis['grid_position_residual_m'] = float(np.abs(x-X-drift['ugrid_m']).max())
    snap = bed.snapshot()
    if case == 'TRIM':
        region = X[:, 2] > geometry['plane_z_m']
        analysis['chip_links'] = rd.chip_links(X, lattice, x, z['FINAL_F'], z['FINAL_codes'], s, region)
        analysis['chip_energy'] = rd.chip_energy(X, snap['F'], snap['Fpi'], snap['plastic_strain'], region, s,
                                                 MATERIAL_RES, history=bed._rc_energy.arrays(), codes=snap['codes'],
                                                 lattice=lattice, locus_distance=dist)
    split, per = force_split(bed._rc_split.table(), ctx['substeps'], steps['window'], steps['steady'])
    analysis['force_split'] = split
    analysis['tool_work_j'] = float(steps['stats'][:, 2].sum())
    analysis['momentum_energy_final'] = momentum_energy(bed)
    fin = result['analysis']['FINAL']
    keys = ('mechanical_components', 'mass_fractions', 'labelled_particles', 'direct_code0_bridges', 'separation_criterion',
            'removal_fraction', 'substrate_retained', 'trim_criterion_wp11', 'det_F', 'per_plane')
    analysis['topology_FINAL'] = {k: fin[k] for k in keys if k in fin}
    return analysis, per, snap


def analyse_e1(row, result, ctx, out, bed):
    from robotic_cutting_sim import root_cause_diagnostics as rd
    spec, E = ctx['spec'], ctx['E']
    z = np.load(out/'snapshots.npz')
    X = z['reference_local'].astype(np.float64)
    codes = z['final_codes'].astype(np.int64)
    nodes, weights = z['stencil_nodes'], z['stencil_weights']
    path = E.edge_path(spec)
    s_ref = (X-path['o'])@path['n']
    mis = rd.mis_sided_bridges(codes, nodes, weights, s_ref)
    mis.pop('bridge_indices')
    drift = bed._rc_drift.arrays()
    digit = codes & 3
    wrong = ((digit == 2) & (s_ref < 0)) | ((digit == 1) & (s_ref >= 0))
    disp = drift['assign_position_m'][:, 0].astype(np.float64)-X
    mis['mis_sided_labels_detail'] = {
        'count': int(wrong.sum()),
        's_ref_mm': [float(s_ref[wrong].min()*1e3), float(s_ref[wrong].max()*1e3)] if wrong.any() else None,
        'displacement_at_assignment_mm': rd._stats(np.linalg.norm(disp[wrong], axis=1)*1e3) if wrong.any() else None,
        'displacement_along_plane_normal_at_assignment_mm': rd._stats((disp[wrong]@path['n'])*1e3) if wrong.any() else None,
        'reference_z_mm': rd._stats(X[wrong, 2]*1e3) if wrong.any() else None}
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    new, lab = rd.visibility_relabel(X, codes, nodes, weights, path['o'], path['n'], spec['edge_end_z_mm']*1e-3)
    k0 = mechanical_components_packed(nodes, weights, codes)[0]
    k1, l1 = mechanical_components_packed(nodes, weights, new)
    mass = z['mass_kg']
    fr = np.sort(np.bincount(l1, weights=mass))[::-1]/mass.sum()
    lab.update(law8_components=int(k0), predicted_components=int(k1), predicted_mass_fractions=fr[:3].tolist(),
               label='STATIC shadow (final-state relabel; not a law, not a simulation)')
    return {'p3_split': mis, 'visibility_static': lab, 'momentum_energy_final': momentum_energy(bed)}, None, bed.snapshot()


def analyse_probe(row, result, ctx, out, bed):
    from robotic_cutting_sim import root_cause_diagnostics as rd
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    X = ctx['points'].astype(np.float64)
    lattice, geo = ctx['lattice'], ctx['geo']
    snap = bed.snapshot()
    x, F, codes = snap['positions'].astype(np.float64), snap['F'], snap['codes']
    loc = geo['locus']
    if loc['type'] == 'z':
        dist = lambda P: np.abs(np.asarray(P)[:, 2]-loc['z_m'])  # noqa: E731
    else:
        dist = lambda P: rd.plane_distance(P, loc['planes'])  # noqa: E731
    drift = bed._rc_drift.arrays()
    analysis = {'locus': loc, 'links_FINAL': rd.link_consistency(X, lattice, x, F, codes, PROBE_S, dist,
                                                                  drift=drift['drift_m'],
                                                                  assign_substep=drift['assign_substep'])}
    analysis['drift_identity_position_residual_m'] = float(np.abs(x-X-drift['ugrid_m']-drift['drift_m']).max())
    analysis['grid_position_residual_m'] = float(np.abs(x-X-drift['ugrid_m']).max())
    count, labels = mechanical_components_packed(ctx['nodes'], ctx['weights'], codes, 1, gradients=ctx['gradients'])
    m = ctx['masses'].astype(np.float64)
    topo = {'mechanical_components': int(count),
            'mass_fractions': (np.sort(np.bincount(labels, weights=m))[::-1]/m.sum())[:4].tolist(),
            'labelled_particles': int(np.count_nonzero(codes))}
    if 'chip_region' in geo:
        top = geo['chip_region']['z_m']
        region = X[:, 2] > top
        base = np.bincount(labels[~region], weights=m[~region]).argmax()
        topo['removal_fraction'] = float(m[region & (labels != base)].sum()/m[region].sum())
        topo['substrate_retained'] = float(m[~region & (labels == base)].sum()/m[~region].sum())
        analysis['chip_links'] = rd.chip_links(X, lattice, x, F, codes, PROBE_S, region)
        analysis['chip_energy'] = rd.chip_energy(X, snap['F'], snap['Fpi'], snap['plastic_strain'], region, PROBE_S,
                                                 MATERIAL_RES, history=bed._rc_energy.arrays(), codes=codes,
                                                 lattice=lattice, locus_distance=dist)
    analysis['topology_FINAL'] = topo
    analysis['momentum_energy_final'] = momentum_energy(bed)
    analysis['force_split_raw_summary'] = {'substeps': int(len(bed._rc_split.table()))}
    return analysis, None, snap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    p.add_argument('--dev', action='store_true', help='DEV smoke (label DEV, never evidence)')
    p.add_argument('--dev-max-steps', type=int, help='DEV only: cap the control steps')
    a = p.parse_args()
    rows = protocol_rows(a.protocol)
    if a.row not in rows:
        p.error(f'row {a.row!r} is not declared in {a.protocol}')
    if a.dev_max_steps is not None and not a.dev:
        p.error('--dev-max-steps needs --dev')
    parent = a.out.resolve().parent.name
    if a.dev and not re.fullmatch(r'phase4_verify_dev_\d{3}', parent):
        p.error('DEV output must be under artifacts/wp12/phase4_verify_dev_NNN/')
    committed, protocol_commit = protocol_committed(a.protocol)
    if not a.dev:
        if not re.fullmatch(r'phase4_verify_\d{3}', parent):
            p.error('formal output must be under artifacts/wp12/phase4_verify_NNN/')
        if not committed:
            p.error('formal rows need the verification protocol committed and unmodified (git)')
    a.out.mkdir(parents=True, exist_ok=False)   # never overwrite
    row = rows[a.row]
    counterfactual = bool(row.get('counterfactual'))
    label = ('DEV' if a.dev else 'FORMAL') + (' COUNTERFACTUAL_NOT_A_GATE' if counterfactual else ' UNCHANGED_SOLVER')
    base = {'row': a.row, 'row_spec': row, 'label': label, 'counterfactual': counterfactual,
            'command': sys.argv, **git_state(), 'protocol_sha256': hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest(),
            'protocol_committed': committed, 'protocol_last_commit': protocol_commit,
            'seed': SEED, 'device': a.device, 'host': os.uname().nodename, 'started_unix': time.time(),
            'dev_max_steps': a.dev_max_steps, 'recorders': ['flip_drift_v1', 'tool_contact_split_v1', 'tensile_energy_max_v1'],
            'scope': 'Kinematic ideal tracking; provisional uncalibrated material; tlmpm_v5 and law 8 unchanged; '
                     'COUNTERFACTUAL rows are measurements of a mechanism, never acceptance, a law or a profile'}
    (a.out/'manifest.json').write_text(json.dumps(jsonable(base), indent=1))
    holder = {}
    start = time.monotonic()
    try:
        kind = row['kind']
        driver = {'res': run_res, 'e1': run_e1, 'probe': run_probe}[kind]
        analyser = {'res': analyse_res, 'e1': analyse_e1, 'probe': analyse_probe}[kind]
        result, ctx = driver(row, a, a.out, holder)
        drive_wall = time.monotonic()-start
        bed = holder['bed']
        t = time.monotonic()
        analysis, per_step, snap = analyser(row, result, ctx, a.out, bed)
        rec = bed._rc_drift.arrays()
        np.savez_compressed(a.out/'verify.npz', drift_m=rec['drift_m'], ugrid_m=rec['ugrid_m'],
                            assign_substep=rec['assign_substep'], assign_position_m=rec['assign_position_m'],
                            split_per_control_step=per_step if per_step is not None else bed._rc_split.table(),
                            final_F=snap['F'], final_Fpi=snap['Fpi'], final_plastic=snap['plastic_strain'],
                            final_stress=bed.stress.numpy(), final_velocities=snap['velocities'],
                            final_positions=snap['positions'], final_codes=snap['codes'],
                            **bed._rc_energy.arrays())
        split_meta = bed._rc_split.metadata()
        integrity = {'solver_instances': holder.get('constructed'), 'split_overflow': split_meta['overflow'],
                     'drift_substeps': bed._rc_drift.metadata()['substeps'], 'split_substeps': split_meta['substeps'],
                     'recorder_substeps_agree': bed._rc_drift.metadata()['substeps'] == split_meta['substeps']}
        if kind == 'res':
            finite = bool(result['stability']['finite'])
        elif kind == 'e1':
            finite = result.get('row_status') == 'PASS'
        else:
            finite = bool(np.isfinite(snap['positions']).all())
        ok = finite and integrity['split_overflow'] == 0 and integrity['solver_instances'] == 1 and \
            integrity['recorder_substeps_agree']
        receipt = bed.counterfactual_receipt() if counterfactual else {'label': 'UNCHANGED_SOLVER', 'module': 'tlmpm_v5'}
        record = {**base, 'status': 'COMPLETED' if ok else 'FAIL_INTEGRITY', 'integrity': integrity,
                  'solver_receipt': receipt, 'effective_pic': float(bed.config.pic), 'material_dt_s': float(bed.config.dt),
                  'recorder_metadata': {'flip_drift_v1': bed._rc_drift.metadata(), 'tool_contact_split_v1': split_meta,
                                        'tensile_energy_max_v1': bed._rc_energy.metadata()},
                  'driver_result': result, 'analysis': analysis, 'drive_wall_s': drive_wall,
                  'analysis_wall_s': time.monotonic()-t, 'wall_s': time.monotonic()-start}
        (a.out/'result.json').write_text(json.dumps(jsonable(record), indent=1))
        first = (analysis.get('links_FINAL') or {}).get('0-1s', {})
        print(a.row, label, record['status'], 'first band P50/P99', first.get('p50'), first.get('p99'),
              'explained', (first.get('identity') or {}).get('explained_fraction'),
              'wall', round(record['wall_s'], 1), flush=True)
    except Exception as error:
        (a.out/'result.json').write_text(json.dumps(jsonable({**base, 'status': 'ERROR', 'error': repr(error),
                                                               'wall_s': time.monotonic()-start}), indent=1))
        raise


if __name__ == '__main__':
    main()
