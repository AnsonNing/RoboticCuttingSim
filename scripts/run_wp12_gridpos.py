#!/usr/bin/env python3
"""WP-12 Phase 4: staged acceptance of the CANDIDATE solver tlmpm_v6 (protocol configs/wp12/phase4_gridpos_protocol_v1.json).

One declared row in one arm per process: ``V6`` (tlmpm_v6, position_update grid_musl_v1, law 8) or ``V5`` (tlmpm_v5, law
8, the same commit). Every row drives an existing, unchanged driver; this script never edits a frozen file. The frozen
runners are executed read-only as private module instances (their ``main`` with their own argument checks, writing
their own ``result.json`` into a ``run_*`` subdirectory of the row directory):

* ``matrix``          scripts/run_phase1_matrix.py (episode rows with ``--solver v5`` = packed path; M9 replays without,
                      with ``replay_measured_edge.py`` executed in this process instead of a subprocess);
* ``matrix_m7``       run_phase1_matrix.py M7 from this arm's M1a and M1b row directories (CPU);
* ``phase_f``         scripts/run_phase_f_material.py (the CPU-pinned phaseF_v1 runner, unchanged);
* ``phase_f_v2``      scripts/run_phase_f_v2_material.py (F2.9_blunt_handle_v2);
* ``phase_r``         scripts/run_phase_r_negative_controls.py (protocol (b) rows, K = 5, digit 4);
* ``workcell``        Phase 2b episodes (configs/wp12/workcell_live_v1.json: scene, goal, path_v2 overrides, duration rule
                      via run_wp12_workcell_live.compile_live) on the kinematic backend through
                      run_wp12_phase5.kinematic_episode (the E2 pipeline: goal_planner, WaypointPolicy, goal evaluator);
* ``postcut``         new ADR-0014 post-cut controls: run_phase1_matrix.run_episode with plans registered in this
                      process's private instance (through-cut digit 0, then digit 1 armed for a press/push/stop+withdraw);
* ``verify``          scripts/run_wp12_phase4_verification.py drivers and analysers (res: run_wp12_resolution.py,
                      e1: run_wp12_phase4_e1.py, probe: blade_probe S7 / G3b), recorders flip_drift_v1,
                      tool_contact_split_v1, tensile_energy_max_v1;
* ``phase4c``         scripts/run_wp12_phase4c.py run --case;
* ``generalization``  scripts/run_generalization.py material (WP-11 round 2 samples).

Solver selection (the only intervention, recorded in ``gridpos_result.json``): ``LiveCuttingMaterial.__init__`` is
wrapped in this process so every material is built with ``solver`` = the arm solver (the driver asked for None or
"tlmpm_v5"); drivers that construct ``tlmpm_v5.TLMPM`` directly (verify/res, verify/probe) get the arm class under
that name. Frozen runners keep writing their hard-coded solver label into their own result.json; the effective solver
is the material receipt / bed module recorded here. The read-only ``gridpos_monitor_v1`` copies state to the host after
every control step (integrity and reported quantities). Nothing changes physics, a law, a threshold or a record.

Formal rows need the protocol committed and unmodified (git) and write ``artifacts/wp12/gridpos_NNN/<arm>/<row>``;
``--dev`` rows (label DEV, never evidence) write ``artifacts/wp12/gridpos_dev_NNN/<arm>/<row>`` and may cap the
control steps. Directories are created once and never overwritten.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_gridpos.py run --row M4 --arm V6 --device cuda:1 \\
      --out artifacts/wp12/gridpos_001/V6/M4
  TMPDIR=/mnt/data/ningan/tmp nohup .venv/bin/python scripts/run_wp12_gridpos.py queue --stage S1 \\
      --out-root artifacts/wp12/gridpos_001 >> artifacts/wp12/logs/gridpos_001_queue.log 2>&1 &
"""
import argparse
import contextlib
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
PROTOCOL = ROOT/'configs/wp12/phase4_gridpos_protocol_v1.json'
ARMS = {'V5': 'tlmpm_v5', 'V6': 'tlmpm_v6'}
SEED = 42
SOURCES = ('src/robotic_cutting_sim/tlmpm_v4.py', 'src/robotic_cutting_sim/tlmpm_v5.py', 'src/robotic_cutting_sim/tlmpm_v6.py',
           'src/robotic_cutting_sim/tlmpm_v5_counterfactual.py', 'src/robotic_cutting_sim/packed_solvers.py',
           'src/robotic_cutting_sim/live_material.py', 'src/robotic_cutting_sim/kinematic_env.py',
           'src/robotic_cutting_sim/blade_probe.py', 'src/robotic_cutting_sim/root_cause_recorders.py',
           'scripts/run_phase1_matrix.py', 'scripts/run_phase_f_material.py', 'scripts/run_phase_f_v2_material.py',
           'scripts/run_phase_r_negative_controls.py', 'scripts/run_wp12_workcell_live.py', 'scripts/run_wp12_phase5.py',
           'scripts/run_wp12_phase4_e2.py', 'scripts/run_wp12_phase4_verification.py', 'scripts/run_wp12_resolution.py',
           'scripts/run_wp12_phase4_e1.py', 'scripts/run_wp12_phase4c.py', 'scripts/run_generalization.py',
           'scripts/replay_measured_edge.py', 'configs/wp12/phase4_gridpos_protocol_v1.json', 'scripts/run_wp12_gridpos.py')
VERIFY_KINDS = ('res', 'e1', 'probe')


# ----------------------------------------------------------------------------- utilities

def load_script(name, tag='gridpos'):
    """Private module instance of scripts/<name>.py (read-only reuse; the file is not modified)."""
    spec = importlib.util.spec_from_file_location(f'wp12_{tag}_private_{name}', ROOT/f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def git_state():
    return {'git_sha': git('rev-parse', 'HEAD').stdout.strip(),
            'git_status_porcelain': git('status', '--porcelain').stdout.splitlines(),
            'source_sha256': {p: sha(ROOT/p) for p in SOURCES if (ROOT/p).exists()}}


def protocol_committed(path=PROTOCOL):
    rel = str(Path(path).resolve().relative_to(ROOT))
    tracked = git('ls-files', '--error-unmatch', rel).returncode == 0
    clean = git('diff', '--quiet', 'HEAD', '--', rel).returncode == 0
    return tracked and clean, git('log', '-1', '--format=%H', '--', rel).stdout.strip()


def load_protocol(path=PROTOCOL):
    doc = json.loads(Path(path).read_text())
    ids = [r['id'] for r in doc['rows']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate gridpos row ids')
    return doc


def rows_by_id(protocol):
    return {r['id']: r for r in protocol['rows']}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.write_text(json.dumps(jsonable(payload), indent=1))


@contextlib.contextmanager
def argv(args):
    saved = sys.argv
    sys.argv = [str(a) for a in args]
    try:
        yield
    finally:
        sys.argv = saved


def deep_find(value, keys, found=None):
    """Every numeric value under any of ``keys`` (recursive), as (path, value)."""
    found = [] if found is None else found

    def walk(v, path):
        if isinstance(v, dict):
            if v.get('selected') is False:     # an unselected protrusion item is retained by design
                return
            for k, item in v.items():
                if k in keys and isinstance(item, (int, float)) and not isinstance(item, bool):
                    found.append((f'{path}.{k}', float(item)))
                walk(item, f'{path}.{k}')
        elif isinstance(v, list):
            for i, item in enumerate(v):
                walk(item, f'{path}[{i}]')
    walk(value, '')
    return found


REMOVAL_KEYS = ('removed_reference_above_plane_mass_fraction', 'removed_fraction', 'removal_fraction',
                'protrusion_removed_fraction', 'removed_mass_fraction')
SUBSTRATE_KEYS = ('largest_component_below_plane_mass_share', 'substrate_share', 'substrate_retained',
                  'substrate_mass_share', 'largest_substrate_share')


# ----------------------------------------------------------------------------- solver selection and monitor

class Selection:
    """Arm solver selection + gridpos_monitor_v1 for one process (read-only apart from the declared redirect)."""

    def __init__(self, arm, kind, verify_kind=None, split_capacity=None, monitor_flat_board=None):
        self.arm, self.kind, self.verify_kind = arm, kind, verify_kind
        self.solver = ARMS[arm]
        self.redirects, self.materials, self.beds = [], [], []
        self.split_capacity = split_capacity
        self.steps = []
        self.codes_before_digit = {}
        self.elastic = []
        self.grid_ke = []
        self.monitor_flat_board = monitor_flat_board
        self.prev_codes = None
        self.substitution = None
        self._flat = False

    # -- installation
    def install(self):
        import warp as wp
        from robotic_cutting_sim import live_material, root_cause_recorders as rr, tlmpm_v5, tlmpm_v6
        sel = self
        bases = {'V5': tlmpm_v5.TLMPM, 'V6': tlmpm_v6.TLMPM}
        base = bases[self.arm]
        full = self.verify_kind is not None

        class Installed(base):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._rc_drift = rr.FlipDriftRecorder(self)
                self._rc_split = rr.ToolContactSplitRecorder(self, sel.split_capacity) if full and sel.split_capacity else None
                self._rc_energy = rr.TensileEnergyRecorder(self) if full else None
                if wp.get_device(self.device).is_cuda:
                    for module in ('robotic_cutting_sim.tlmpm_v5', 'robotic_cutting_sim.tlmpm_v6',
                                   'robotic_cutting_sim.root_cause_recorders', 'robotic_cutting_sim.fixture_contacts'):
                        if module in sys.modules:
                            wp.load_module(module=module, device=self.device)
                sel.beds.append(self)

            def step(self, advance=True):
                # E1's read-only recorders (run_wp12_phase4_e1.attach_recorders) bracket tlmpm_v5.TLMPM.step, which
                # tlmpm_v6 overrides: bracket them here for the V6 arm (same calls, same order as E1's bracket).
                e1 = getattr(self, '_e1_diag', None) if sel.arm == 'V6' else None
                if e1 is not None:
                    e1.before_step()
                if self._rc_split is not None:
                    self._rc_split.before_step()
                super().step(advance)
                self._rc_drift.after_step()
                if self._rc_energy is not None:
                    self._rc_energy.after_step()
                if e1 is not None:
                    e1.after_step(advanced=advance)
                    self._e1_cand.after_step(advanced=advance)

        Installed.__module__ = base.__module__
        Installed.__qualname__ = Installed.__name__ = f'GridposRecorded{base.__name__}'
        if self.arm == 'V6':
            tlmpm_v6.TLMPM = Installed
        else:
            tlmpm_v5.TLMPM = Installed
        if self.verify_kind in ('res', 'probe') and self.arm == 'V6':
            tlmpm_v5.TLMPM = Installed     # direct `from tlmpm_v5 import TLMPM` in the frozen driver
            self.substitution = 'tlmpm_v5.TLMPM -> tlmpm_v6 (direct_import_substitution, this process only)'

        LCM = live_material.LiveCuttingMaterial
        original_init, original_reset, original_advance = LCM.__init__, LCM.reset, LCM.advance

        def init(material, *a, solver=None, **kw):
            requested = solver
            if requested not in (None, 'tlmpm_v5'):
                raise RuntimeError(f'gridpos: the driver requested solver {requested!r}; only None or tlmpm_v5 are redirected')
            if requested is None and int(kw.get('max_cuts', 1)) > 1:
                raise RuntimeError('gridpos: a ternary (solver None) material with K > 1 cannot be redirected to packed codes')
            original_init(material, *a, solver=sel.solver, **kw)
            sel.redirects.append({'requested': requested, 'effective': sel.solver, 'cutting_law': material.cutting_law,
                                  'max_cuts': material.max_cuts, 'slots_per_node': material.slots_per_node})

        def reset(material, *a, **kw):
            out = original_reset(material, *a, **kw)
            sel.materials.append(material)
            sel.prev_codes = material.bed.code.numpy().copy()
            sel._flat = sel._flat_board(material)
            return out

        def advance(material, *a, **kw):
            out = original_advance(material, *a, **kw)
            sel.after_advance(material)
            return out
        LCM.__init__, LCM.reset, LCM.advance = init, reset, advance
        return self

    @staticmethod
    def _flat_board(material):
        """Solid flat board (no slots): the workpiece-local board top is z = 0 (place_on_board)."""
        task = getattr(material, 'task', None)
        scene = getattr(task, 'scene', None) if task is not None else None
        if not isinstance(scene, dict):
            return False
        supports = scene.get('supports') or {}
        if supports.get('kind') != 'flat_board':
            return False
        text = json.dumps(supports)
        return 'slot' not in text

    # -- per control step (host copies; read-only)
    def after_advance(self, material):
        bed = material.bed
        x, v = bed.x.numpy(), bed.v.numpy()
        F = bed.F.numpy().astype(np.float64)
        det = np.linalg.det(F)
        codes = bed.code.numpy()
        m = bed.mass.numpy().astype(np.float64)
        stats = bed.stats.numpy().astype(np.float64)
        diag = material.last_diagnostics or {}
        dt = material.config.control_dt
        blade = np.asarray(diag.get('current_material_reaction_impulse_world', np.zeros(6)), float)/dt
        total = np.asarray(diag.get('current_total_material_reaction_impulse_world', np.zeros(6)), float)/dt
        K = int(getattr(material, 'max_cuts', 1))
        digits = [int(np.count_nonzero((codes >> (2*k)) & 3)) for k in range(min(K, 6))]
        cut_index = diag.get('cut_index')
        armed = cut_index is not None and int(cut_index) >= 0
        if armed and int(cut_index) not in self.codes_before_digit and self.prev_codes is not None:
            self.codes_before_digit[int(cut_index)] = {'step': len(self.steps)+1, 'codes': self.prev_codes.copy()}
        finite = bool(np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(F).all())
        row = {'t': float(material.time_s), 'finite': finite, 'min_det_F': float(np.nanmin(det)) if np.isfinite(det).any() else float('nan'),
               'max_speed': float(np.sqrt((v.astype(np.float64)**2).sum(1)).max()), 'cut_index': -1 if cut_index is None else int(cut_index),
               'labelled': int(np.count_nonzero(codes)), 'digits': digits + [0]*(6-len(digits)),
               'blade_force': blade[:3].tolist(), 'total_force': total[:3].tolist(), 'work_j': float(stats[2]),
               'blade_overlap_m': float(stats[3]), 'fixture_overlap_m': float(stats[0]), 'blade_contacts': float(stats[1]),
               'particle_ke_j': float(.5*(m*(v.astype(np.float64)**2).sum(1)).sum()),
               'min_local_z_m': float(x[:, 2].min()) if self._flat else float('nan')}
        self.steps.append(row)
        n = len(self.steps)
        if n % 5 == 0:
            self.grid_ke.append((n, *self._grid_ke(bed)))
        if n % 10 == 0:
            self.elastic.append((n, self._elastic(bed)))
        self.prev_codes = codes

    @staticmethod
    def _grid_ke(bed):
        """(grid KE of the MUSL field, null-space KE 1/2 sum m |v_p - sum_c w v^MUSL|^2) after the last substep."""
        gm, nv = bed.gm.numpy().astype(np.float64), bed.new_v.numpy().astype(np.float64)
        grid = .5*float((gm*(nv**2).sum(1)).sum())
        slot, nodes, w = bed.slot.numpy(), bed.nodes.numpy(), bed.weights.numpy().astype(np.float64)
        index = nodes*bed.slots_per_node+np.maximum(slot, 0)
        vg = (w[..., None]*np.where((slot >= 0)[..., None], nv[index], 0.)).sum(1)
        v = bed.v.numpy().astype(np.float64)
        m = bed.mass.numpy().astype(np.float64)
        return grid, .5*float((m*((v-vg)**2).sum(1)).sum())

    @staticmethod
    def _elastic(bed):
        """Hencky elastic energy sum V (mu |e|^2 + lambda/2 tr(e)^2), e = log singular values of F Fpi."""
        c = bed.config
        lam = c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu = c.young/(2*(1+c.poisson))
        fe = bed.F.numpy().astype(np.float64)@bed.Fpi.numpy().astype(np.float64)
        s = np.linalg.svd(fe, compute_uv=False)
        e = np.log(np.maximum(s, .05))
        return float((bed.volume*(mu*(e**2).sum(1)+.5*lam*e.sum(1)**2)).sum())

    # -- end of row
    def final(self):
        from robotic_cutting_sim.cut_topology import mechanical_components_packed
        out = {'solver_selection': {'arm': self.arm, 'arm_solver': self.solver, 'redirects': self.redirects,
                                    'direct_import_substitution': self.substitution,
                                    'beds': [{'module': type(b).__module__, 'class': type(b).__name__,
                                              'mro_solver': [k.__module__ for k in type(b).__mro__ if 'tlmpm' in k.__module__],
                                              'position_update': getattr(b, 'position_update', None),
                                              'receipt': b.solver_receipt() if hasattr(b, 'solver_receipt') else None}
                                             for b in self.beds]}}
        ok = bool(self.beds) and all(self.solver in [k.__module__.rsplit('.', 1)[-1] for k in type(b).__mro__][:2]
                                     for b in self.beds)
        if self.arm == 'V6':
            ok = ok and all(getattr(b, 'position_update', None) == 'grid_musl_v1' for b in self.beds)
        else:
            ok = ok and all(not hasattr(b, 'position_update') for b in self.beds)
        ok = ok and all(r['effective'] == self.solver for r in self.redirects)
        out['solver_selection']['verified'] = bool(ok)
        if not self.beds:
            return out
        bed = self.beds[-1]
        drift = bed._rc_drift.arrays()
        X = bed.reference.numpy().astype(np.float64)
        x = bed.x.numpy().astype(np.float64)
        out['grid_position_residual_m'] = float(np.abs(x-X-drift['ugrid_m']).max())
        out['drift_max_m'] = float(np.abs(drift['drift_m']).max())
        material = self.materials[-1] if self.materials else None
        K = int(getattr(material, 'max_cuts', bed.max_cuts))
        nodes, weights, grads = bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy()
        codes = bed.code.numpy()
        count, labels = mechanical_components_packed(nodes, weights, codes, K, gradients=grads)
        m = bed.mass.numpy().astype(np.float64)
        v = bed.v.numpy().astype(np.float64)
        masses = np.bincount(labels, weights=m, minlength=count)
        order = np.argsort(masses)[::-1]
        pieces = [{'mass_fraction': float(masses[i]/m.sum()),
                   'com_velocity_m_s': (np.bincount(labels, weights=m*v[:, 0], minlength=count)[i]/masses[i],
                                        np.bincount(labels, weights=m*v[:, 1], minlength=count)[i]/masses[i],
                                        np.bincount(labels, weights=m*v[:, 2], minlength=count)[i]/masses[i])}
                  for i in order[:6] if masses[i] > 0]
        out['final_topology'] = {'mechanical_components': int(count), 'mass_fractions': [p['mass_fraction'] for p in pieces],
                                 'pieces': pieces, 'labelled': int(np.count_nonzero(codes)), 'max_cuts': K}
        if self.codes_before_digit:
            out['codes_before_digit'] = {}
            for digit, rec in sorted(self.codes_before_digit.items()):
                before = rec['codes']
                c0, _ = mechanical_components_packed(nodes, weights, before, K, gradients=grads)
                out['codes_before_digit'][digit] = {'step': rec['step'], 'components': int(c0),
                                                    'labelled': int(np.count_nonzero(before))}
        if self.steps:
            s = self.steps
            det = np.array([r['min_det_F'] for r in s])
            zmin = np.array([r['min_local_z_m'] for r in s])
            force = np.array([r['blade_force'] for r in s])
            contact = np.array([r['blade_contacts'] for r in s]) > 0
            out['monitor'] = {
                'version': 'gridpos_monitor_v1', 'control_steps': len(s), 'finite_all': bool(all(r['finite'] for r in s)),
                'min_det_F': float(np.nanmin(det)), 'max_speed_m_s': float(max(r['max_speed'] for r in s)),
                'tool_work_j': float(sum(r['work_j'] for r in s)), 'max_blade_overlap_m': float(max(r['blade_overlap_m'] for r in s)),
                'max_fixture_overlap_m': float(max(r['fixture_overlap_m'] for r in s)),
                'blade_force_mean_abs_contact_N': float(np.linalg.norm(force[contact], axis=1).mean()) if contact.any() else 0.,
                'blade_force_peak_abs_N': float(np.linalg.norm(force, axis=1).max()),
                'contact_steps': int(contact.sum()),
                'final_particle_ke_j': s[-1]['particle_ke_j'], 'max_particle_ke_j': float(max(r['particle_ke_j'] for r in s)),
                'final_labelled': s[-1]['labelled'], 'final_digit_labels': s[-1]['digits'],
                'flat_board_checked': bool(self._flat), 'min_local_z_m': float(np.nanmin(zmin)) if self._flat else None,
                'grid_ke_samples': len(self.grid_ke), 'elastic_samples': len(self.elastic)}
            g = self._grid_ke(bed)
            out['monitor'].update(final_grid_ke_j=g[0], final_null_space_ke_j=g[1], final_elastic_j=self._elastic(bed),
                                  max_null_space_ke_j=max([g[1]]+[k[2] for k in self.grid_ke]),
                                  max_elastic_j=max([e for _, e in self.elastic] or [0.]))
        return out

    def save(self, out_dir):
        if not self.steps:
            return
        s = self.steps
        np.savez_compressed(Path(out_dir)/'monitor.npz', **{
            't_s': np.array([r['t'] for r in s]), 'finite': np.array([r['finite'] for r in s]),
            'min_det_F': np.array([r['min_det_F'] for r in s]), 'max_speed': np.array([r['max_speed'] for r in s]),
            'cut_index': np.array([r['cut_index'] for r in s]), 'labelled': np.array([r['labelled'] for r in s]),
            'digit_labels': np.array([r['digits'] for r in s]), 'blade_force_N': np.array([r['blade_force'] for r in s]),
            'total_force_N': np.array([r['total_force'] for r in s]), 'work_j': np.array([r['work_j'] for r in s]),
            'blade_overlap_m': np.array([r['blade_overlap_m'] for r in s]),
            'fixture_overlap_m': np.array([r['fixture_overlap_m'] for r in s]),
            'blade_contacts': np.array([r['blade_contacts'] for r in s]), 'particle_ke_j': np.array([r['particle_ke_j'] for r in s]),
            'min_local_z_m': np.array([r['min_local_z_m'] for r in s]),
            'grid_ke': np.array(self.grid_ke).reshape(-1, 3), 'elastic': np.array(self.elastic).reshape(-1, 2)})


# ----------------------------------------------------------------------------- drivers

def _subdir(out, name):
    return Path(out)/name


def _read(path):
    p = Path(path)/'result.json'
    return json.loads(p.read_text()) if p.exists() else None


def run_matrix(row, args, out, sel):
    M = load_script('run_phase1_matrix')
    mrow = row['args']['row']
    replay = mrow.startswith('M9')
    if replay:
        _inprocess_replay(M, out)
    sub = _subdir(out, f'run_{mrow.lower()}_law8' + ('' if replay else '_v5path'))
    cmd = ['run_phase1_matrix.py', '--row', mrow, '--cutting-law', '8', '--device', args.device, '--out', sub]
    if not replay:
        cmd += ['--solver', 'v5']
    if args.dev_max_steps:
        cmd += ['--steps', args.dev_max_steps]
    with argv(cmd):
        M.main()
    r = _read(sub)
    amended = None
    if mrow == 'M2':   # the law-8 matrix adjudicated M2 by ADR 0006 Amendment 1 (read-only rescore of the row directory)
        O = load_script('rescore_phase1_oblique')
        target = Path(out)/'m2_rescore_amendment1.json'
        O.main(['--row-dir', str(sub), '--out', str(target)])
        amended = json.loads(target.read_text())
    return {'frozen_result_dir': str(sub),
            'frozen_verdict': amended['verdict'] if amended is not None else r.get('verdict'),
            'runner_verdict': r.get('verdict'), 'm2_amendment1': amended, 'frozen_checks': r.get('checks'),
            'frozen_metrics_head': {k: r.get('metrics', {}).get(k) for k in ('final_mechanical_components', 'final_component_mass_fractions',
                                                                             'final_labelled_count', 'detached_below_plane_mass_fraction')},
            'trim': (r.get('metrics') or {}).get('trim_separation'), '_result': r}


def _inprocess_replay(M, out):
    """M.run_replay launches replay_measured_edge.py as a subprocess; run it in this process instead so the solver
    selection applies (same arguments, same script, read-only)."""
    real = subprocess

    class Shim:
        STDOUT = real.STDOUT
        PIPE = real.PIPE

        @staticmethod
        def run(command, *a, **kw):
            if len(command) > 1 and str(command[1]).endswith('replay_measured_edge.py'):
                stream = kw.get('stdout')
                buffer = io.StringIO()
                with argv(command[1:]), contextlib.redirect_stdout(buffer):
                    runpy.run_path(str(command[1]), run_name='__main__')
                if stream is not None and hasattr(stream, 'write'):
                    stream.write(buffer.getvalue())
                return SimpleNamespace(returncode=0, stdout=buffer.getvalue(), stderr='')
            return real.run(command, *a, **kw)
    M.subprocess = Shim


def run_matrix_m7(row, args, out, sel):
    M = load_script('run_phase1_matrix')
    arm_root = Path(out).parent
    sources = []
    for src in row['args']['sources']:
        res = json.loads((arm_root/src/'gridpos_result.json').read_text())
        sources.append(res['driver']['frozen_result_dir'])
    sub = _subdir(out, 'run_m7_law8')
    with argv(['run_phase1_matrix.py', '--row', 'M7', '--cutting-law', '8', '--out', sub, '--sources', *sources]):
        M.main()
    r = _read(sub)
    return {'frozen_result_dir': str(sub), 'frozen_verdict': r.get('verdict'), 'frozen_checks': r.get('checks'), 'sources': sources,
            '_result': r}


def run_phase_f(row, args, out, sel):
    F = load_script('run_phase_f_material')
    frow, variant = row['args']['row'], row['args'].get('variant', 'formal')
    v5 = frow == 'F2.11_multicut_history'
    sub = _subdir(out, f'run_{frow.lower().replace(".", "_")}_{variant}_law8' + ('_v5' if v5 else ''))
    cmd = ['run_phase_f_material.py', '--row', frow, '--variant', variant, '--device', args.device, '--out', sub]
    if args.dev_max_steps:
        cmd += ['--steps', args.dev_max_steps]
    with argv(cmd):
        F.main()
    r = _read(sub)
    return {'frozen_result_dir': str(sub), 'frozen_verdict': r.get('verdict'), 'frozen_checks': r.get('checks'),
            'criterion_outcome': r.get('criterion_outcome'), 'trim': (r.get('metrics') or {}).get('trim_separation'), '_result': r}


def run_phase_f_v2(row, args, out, sel):
    V2 = load_script('run_phase_f_v2_material')
    sub = _subdir(out, 'run_f2_9_blunt_handle_v2_law8')
    cmd = ['run_phase_f_v2_material.py', '--device', args.device, '--out', sub]
    if args.dev_max_steps:
        cmd += ['--steps', args.dev_max_steps]
    with argv(cmd):
        V2.main()
    r = _read(sub)
    return {'frozen_result_dir': str(sub), 'frozen_verdict': r.get('verdict'), 'frozen_checks': r.get('checks'),
            'control_validity': r.get('control_validity'), '_result': r}


def run_phase_r(row, args, out, sel):
    R = load_script('run_phase_r_negative_controls')
    rrow = row['args']['row']
    sub = _subdir(out, f'run_{rrow.lower()}_law8_v5')
    cmd = ['run_phase_r_negative_controls.py', '--row', rrow, '--device', args.device, '--out', sub]
    if args.dev_max_steps:
        cmd += ['--steps', args.dev_max_steps]
    with argv(cmd):
        R.main()
    r = _read(sub)
    return {'frozen_result_dir': str(sub), 'frozen_verdict': r.get('verdict'), 'frozen_checks': r.get('checks'),
            'control_validity': r.get('control_validity'), '_result': r}


def run_workcell(row, args, out, sel):
    """Phase 2b episode compiled exactly as the Native stage (scene, goal, path_v2 overrides, duration rule), run on
    the kinematic backend through run_wp12_phase5.kinematic_episode (E2 pipeline) with this compiled task."""
    WL = load_script('run_wp12_workcell_live')
    E2 = load_script('run_wp12_phase4_e2')
    P5 = load_script('run_wp12_phase5')
    protocol = WL.load_protocol()
    spec = WL.episode_spec(protocol, row['args']['episode'])
    built = WL.compile_live(protocol, spec)
    task = built['task']
    info = {'solver_module': task.solver_module, 'max_cuts': task.max_cuts,
            'slots_per_node': task.scene['solver'].get('slots_per_node'), 'scene': built['scene'], 'goal': built['goal']}
    P5.W.compile_sample = lambda config, sample: (task, info)     # this process's private P5 module only
    config = E2.pseudo_config(E2.load_protocol())
    sample = {'sample_id': f"GRIDPOS_FB_{row['args']['episode']}", 'family': 'phase2b', 'range_set': 'phase2b', 'split': 'gridpos',
              'seed': SEED, 'variant': row['args']['episode'], 'draws': [], 'derived': {}, 'envelope_parameters': {},
              'ood_dimensions': []}
    sub = _subdir(out, f"run_{row['args']['episode'].lower()}")
    ns = SimpleNamespace(out=sub, device=args.device, steps=args.dev_max_steps, expect_task_id=None,
                         feasibility_class=None, surface_profile=None, geometry_level=None, no_geometry=True)
    extra = {'label': 'DEV' if args.dev else 'GRIDPOS_FORMAL', 'gridpos_row': row['id'], 'workcell_episode': spec,
             'workcell_protocol_sha256': sha(WL.PROTOCOL), 'live_task_id': task.task_id, 'duration_s': built['duration_s'],
             'goal_diff_vs_source': built['goal_diff_vs_source']}
    P5.kinematic_episode(config, sample, ns, extra=extra)
    r = _read(sub)
    verdict = 'PASS' if r.get('class') == 'task_success' else 'FAIL'
    partial = None
    if row['args']['episode'] == 'PARTIAL':
        partial = partial_cut_rule(r, sub, sel.materials[-1].points)
        verdict = 'PASS' if partial['accepted'] else 'FAIL'
    if r.get('partial'):
        verdict = verdict+'_NONFORMAL'
    return {'frozen_result_dir': str(sub), 'frozen_verdict': verdict, 'class': r.get('class'), 'class_reason': r.get('class_reason'),
            'evaluation_verdict': (r.get('evaluation') or {}).get('verdict'), 'policy_done': r.get('policy_done'),
            'live_task_id': task.task_id, 'partial_cut_rule': partial, '_result': r}


def partial_cut_rule(result, sub, reference_local):
    """workcell_live_v1 PARTIAL_CUT_ACCEPTED on the kinematic result: the same five checks and expressions as
    run_wp12_workcell_live_analysis.partial_rule (grasp / IK failures cannot occur on the kinematic backend)."""
    from robotic_cutting_sim.cut_topology import decode_digits
    evaluation = result.get('evaluation') or {}
    z = np.load(Path(sub)/'physics_state.npz')
    digits = decode_digits(z['codes'][-1], int(z['max_cuts']))[:, 0]
    ref_z = np.asarray(reference_local, float)[:, 2]
    labelled = digits != 0
    integrity = (evaluation.get('integrity') or {}).get('status')
    n = ((evaluation.get('outcome') or {}).get('metrics') or {}).get('mechanical_components')
    floor = 0.010-0.004
    checks = {'integrity_pass': integrity == 'PASS', 'one_mechanical_component': n == 1,
              'kerf_exists': bool(labelled.any()),
              'no_label_below_declared_depth_minus_one_cell': bool(labelled.any() and ref_z[labelled].min() >= floor-1e-9),
              'plan_finished_no_grasp_or_ik_failure': bool(result.get('policy_done'))}
    return {'class': 'PARTIAL_CUT_ACCEPTED' if all(checks.values()) else 'PARTIAL_CUT_FAIL', 'checks': checks,
            'accepted': all(checks.values()), 'mechanical_components': n, 'labelled_particles_cut0': int(labelled.sum()),
            'labelled_reference_z_min_m': float(ref_z[labelled].min()) if labelled.any() else None,
            'labelled_reference_z_floor_m': floor, 'sever_verdict': evaluation.get('verdict'),
            'rule': 'configs/wp12/workcell_live_v1.json episodes[PARTIAL].success (run_wp12_workcell_live_analysis.partial_rule)'}


POSTCUT_DURATION_S = 16.


def run_postcut(row, args, out, sel):
    M = load_script('run_phase1_matrix')
    from dataclasses import replace
    from robotic_cutting_sim.policies import through_cut
    control = row['args']['control']
    name = f'POSTCUT_{control}'

    def plan(task):
        cut = M.labelled(through_cut(task, cut_index=0), ('approach', 'start', 'cut', 'end', 'retract'))
        if control == 'stop_withdraw':
            body = cut[:4]              # approach, start, cut, end: the edge stays at the bottom of the kerf
            end, start = body[3], body[1]
            tail = [replace(end, cut=True, cut_index=1, hold_s=.5, speed_m_s=None, label='post_hold'),
                    replace(start, cut=True, cut_index=1, speed_m_s=.01, hold_s=0., label='post_withdraw')]
            return body + tail
        builder = M.flat_press_plan if control == 'press' else M.side_push_plan
        extra = [replace(w, cut_index=1, label=f'post_{w.label}') if w.cut else replace(w, label=f'post_{w.label}')
                 for w in builder(task)]
        return cut + extra
    M.ROW_CONFIG[name] = ('CUT-01-R', None, 2, plan, 1e-5)
    # The CUT-01-R episode (8 s) is too short for a through-cut plus a second action: the same task with
    # duration_s = POSTCUT_DURATION_S (this process only; the task, plan geometry and material are otherwise unchanged).
    from robotic_cutting_sim import kinematic_env
    from robotic_cutting_sim.tasks import make_task
    original = kinematic_env.make_kinematic_env

    def make_env(task_id='CUT-01-R', **kw):
        base = make_task(task_id, seed=kw.get('seed', SEED), particle_spacing_m=kw.pop('particle_spacing_m', None))
        knots = [list(k) for k in base.edge_position_knots_local_txyz]
        knots.append([POSTCUT_DURATION_S, *knots[-1][1:]])     # the scripted reference holds its last point
        task = replace(base, duration_s=POSTCUT_DURATION_S, edge_position_knots_local_txyz=tuple(map(tuple, knots)))
        kw.pop('scenario', None)
        return original(task=task, **kw)
    kinematic_env.make_kinematic_env = make_env
    sub = _subdir(out, f'run_{name.lower()}_law8_v5path')
    sub.mkdir(parents=True, exist_ok=False)
    ns = SimpleNamespace(device=args.device, spacing=.001, material_dt=1e-4, cutting_law=8, history_stride=5,
                         steps=args.dev_max_steps, solver='v5')
    steps, data, evaluations, context = M.run_episode(name, ns, sub)
    phases = [s['phase'] for s in steps]
    armed1 = [s for s in steps if s['cut_enabled'] and s['cut_index'] == 1]
    result = {'row': name, 'steps_run': len(steps), 'policy_done': context.get('policy_done'), 'phases': sorted(set(map(str, phases))),
              'armed_digit1_steps': len(armed1), 'plan': context.get('plan'), 'duration_s': POSTCUT_DURATION_S,
              'material_receipt_solver': (context.get('material_receipt') or {}).get('solver_module'),
              'first_digit1_step': armed1[0]['step'] if armed1 else None,
              'labelled_count_at_first_digit1_step': armed1[0]['labelled_count'] if armed1 else None,
              'final_labelled_count': steps[-1]['labelled_count'] if steps else None}
    (sub/'result.json').write_text(json.dumps(jsonable({k: v for k, v in result.items()}), indent=1))
    return {'frozen_result_dir': str(sub), 'postcut': result, 'frozen_verdict': None, '_result': result}


def postcut_verdict(drv, mon):
    before = (mon.get('codes_before_digit') or {}).get(1)
    topo = mon.get('final_topology') or {}
    digits = (mon.get('monitor') or {}).get('final_digit_labels') or [0, 0]
    if before is None or drv['postcut']['armed_digit1_steps'] == 0:
        return 'INVALID', {'reason': 'digit 1 was never armed'}
    checks = {'zero_digit1_labels': digits[1] == 0,
              'digit0_unchanged': None,   # filled by the caller from the saved arrays
              'components_unchanged': topo.get('mechanical_components') == before['components'],
              'precondition_separated': before['components'] >= 2,
              'plan_finished': bool(drv['postcut']['policy_done'])}
    return checks


def run_verify(row, args, out, sel):
    V = load_script('run_wp12_phase4_verification')
    vrow = dict(row['args']['row'])
    vrow['id'] = row['id']
    holder = {}

    def install(_row, capacity, _holder):    # the protocol's selection is installed before the driver runs
        sel.split_capacity = capacity
        return None
    V.install = install
    kind = vrow['kind']
    ns = SimpleNamespace(device=args.device, dev_max_steps=args.dev_max_steps)
    driver = {'res': V.run_res, 'e1': V.run_e1, 'probe': V.run_probe}[kind]
    analyser = {'res': V.analyse_res, 'e1': V.analyse_e1, 'probe': V.analyse_probe}[kind]
    sub = _subdir(out, f'run_{kind}')
    sub.mkdir(parents=True, exist_ok=False)
    result, ctx = driver(vrow, ns, sub, holder)
    bed = sel.beds[-1]
    analysis, per_step, snap = analyser(vrow, result, ctx, sub, bed)
    rec = bed._rc_drift.arrays()
    np.savez_compressed(sub/'verify.npz', drift_m=rec['drift_m'], ugrid_m=rec['ugrid_m'], assign_substep=rec['assign_substep'],
                        assign_position_m=rec['assign_position_m'],
                        split_per_control_step=per_step if per_step is not None else (bed._rc_split.table() if bed._rc_split else np.zeros(0)),
                        final_F=snap['F'], final_Fpi=snap['Fpi'], final_positions=snap['positions'], final_codes=snap['codes'],
                        final_velocities=snap['velocities'], **(bed._rc_energy.arrays() if bed._rc_energy else {}))
    record = {'driver_result': result, 'analysis': analysis,
              'recorder_metadata': {'flip_drift_v1': bed._rc_drift.metadata(),
                                    'tool_contact_split_v1': bed._rc_split.metadata() if bed._rc_split else None,
                                    'tensile_energy_max_v1': bed._rc_energy.metadata() if bed._rc_energy else None}}
    (sub/'result.json').write_text(json.dumps(jsonable(record), indent=1))
    verdict, detail = verify_outcome(kind, result, analysis)
    return {'frozen_result_dir': str(sub), 'frozen_verdict': verdict, 'outcome_detail': detail, 'verify_kind': kind,
            '_result': record}


def verify_outcome(kind, result, analysis):
    """Outcome label of a verify row (compared between arms): res -> separation / trim criterion of the resolution driver;
    e1 -> the E1 outcome label; probe -> components (and removal / substrate for S7)."""
    if kind == 'res':
        fin = (result.get('analysis') or {}).get('FINAL') or {}
        sep = fin.get('separation_criterion')
        trim = fin.get('trim_criterion_wp11')
        crit = trim if trim is not None else sep
        if isinstance(crit, dict):
            ok = crit.get('pass', crit.get('status') in ('PASS', True))
        else:
            ok = bool(crit)
        return ('PASS' if ok else 'FAIL'), {'separation_criterion': sep, 'trim_criterion_wp11': trim,
                                             'mechanical_components': fin.get('mechanical_components'),
                                             'removal_fraction': fin.get('removal_fraction'),
                                             'substrate_retained': fin.get('substrate_retained'),
                                             'stability': result.get('stability')}
    if kind == 'e1':
        law8 = (result.get('analysis') or {}).get('law8_final') or {}
        label = law8.get('outcome') if result.get('row_status') == 'PASS' else result.get('row_status')
        return label, {'row_status': result.get('row_status'), 'law8_final': law8,
                       'law8_end_of_cut': (result.get('analysis') or {}).get('law8_end_of_cut'),
                       'board_penetration': (result.get('analysis') or {}).get('board_penetration')}
    topo = analysis.get('topology_FINAL') or {}
    return f"components_{topo.get('mechanical_components')}", topo


def run_phase4c(row, args, out, sel):
    C = load_script('run_wp12_phase4c')
    sub = _subdir(out, f"run_{row['args']['case']}")
    with argv(['run_wp12_phase4c.py', 'run', '--case', row['args']['case'], '--device', args.device, '--out', sub]):
        C.main()
    r = _read(sub)
    ev = r.get('evaluation') or {}
    return {'frozen_result_dir': str(sub), 'frozen_verdict': ev.get('verdict'), 'status': r.get('status'), '_result': r}


def run_generalization(row, args, out, sel):
    G = load_script('run_generalization')
    a = row['args']
    sub = _subdir(out, f"run_{a['family'].lower()}_s{a['seed']}")
    cmd = ['--config', a['config'], 'material', '--family', a['family'], '--seed', a['seed'], '--device', args.device,
           '--expect-task-id', a['expect_task_id'], '--out', sub]
    if args.dev_max_steps:
        cmd += ['--steps', args.dev_max_steps]
    G.main([str(c) for c in cmd])
    r = _read(sub)
    return {'frozen_result_dir': str(sub), 'frozen_verdict': r.get('class'), 'class_reason': r.get('class_reason'),
            'evaluation_verdict': (r.get('evaluation') or {}).get('verdict'), '_result': r}


DRIVERS = {'matrix': run_matrix, 'matrix_m7': run_matrix_m7, 'phase_f': run_phase_f, 'phase_f_v2': run_phase_f_v2,
           'phase_r': run_phase_r, 'workcell': run_workcell, 'postcut': run_postcut, 'verify': run_verify,
           'phase4c': run_phase4c, 'generalization': run_generalization}


# ----------------------------------------------------------------------------- integrity and row verdict

def trim_numbers(drv):
    final = lambda found: [(k, v) for k, v in found if '.END_' not in k]   # noqa: E731  (FINAL state, not mid-episode)
    found_r = final(deep_find(drv.get('_result') or {}, REMOVAL_KEYS))
    found_s = final(deep_find(drv.get('_result') or {}, SUBSTRATE_KEYS))
    return ({'removal': min(v for _, v in found_r), 'removal_source': found_r[:6]} if found_r else {'removal': None}) | \
           ({'substrate': min(v for _, v in found_s), 'substrate_source': found_s[:6]} if found_s else {'substrate': None})


def integrity(row, drv, mon):
    checks = {'solver_selection_verified': bool((mon.get('solver_selection') or {}).get('verified'))}
    m = mon.get('monitor')
    if row['kind'] == 'matrix_m7':
        return {'not_applicable': 'derived from this arm\'s M1a/M1b rows'}, True
    if m is not None:
        checks.update(finite=m['finite_all'], det_F_positive=m['min_det_F'] > 0, min_det_F_gt_0_5=m['min_det_F'] > .5)
        if m['flat_board_checked']:
            checks['no_centre_below_board_top'] = m['min_local_z_m'] >= 0.
    else:   # verify/res, verify/probe: no LiveCuttingMaterial; final state + driver stability
        res = drv.get('_result') or {}
        stability = (res.get('driver_result') or {}).get('stability') or {}
        checks['finite'] = bool(stability.get('finite', True))
        topo = (res.get('analysis') or {}).get('topology_FINAL') or {}
        det = topo.get('det_F') or {}
        if isinstance(det, dict) and det.get('min') is not None:
            checks['min_det_F_gt_0_5'] = float(det['min']) > .5
    slots = []
    for bed in drv.get('_beds', []):
        slots.append(bed.slot_diagnostics()['overflow_entries'])
    checks['slot_overflow_zero'] = all(s == 0 for s in slots) if slots else None
    ok = all(v is not False for v in checks.values())
    return checks, ok


# ----------------------------------------------------------------------------- run one row

def cmd_run(args):
    protocol = load_protocol(args.protocol)
    rows = rows_by_id(protocol)
    base_id = args.row if args.row in rows else re.sub(r'_r\d+$', '', args.row)   # FB_TRIM_r10 is a row id
    if base_id not in rows:
        raise SystemExit(f'row {args.row!r} is not declared in {args.protocol}')
    row = rows[base_id]
    if args.row != base_id and int(row.get('repeat', 1)) < int(args.row.rsplit('_r', 1)[1]):
        raise SystemExit(f'{args.row}: repeat index beyond the declared repeat count')
    if args.arm not in row.get('arms', list(ARMS)):
        raise SystemExit(f'{args.row} does not declare arm {args.arm}')
    if row['kind'] == 'native':
        raise SystemExit('S4 Native rows are not scheduled by this protocol execution')
    if args.dev_max_steps is not None and not args.dev:
        raise SystemExit('--dev-max-steps needs --dev')
    out = Path(args.out)
    parent = out.resolve().parent
    pattern = r'gridpos_dev_\d{3}' if args.dev else r'gridpos_\d{3}'
    if parent.name != args.arm or not re.fullmatch(pattern, parent.parent.name):
        raise SystemExit(f'output must be artifacts/wp12/{"gridpos_dev_NNN" if args.dev else "gridpos_NNN"}/{args.arm}/<row>')
    committed, protocol_commit = protocol_committed(args.protocol)
    if not args.dev and not committed:
        raise SystemExit('formal rows need the gridpos protocol committed and unmodified (git)')
    out.mkdir(parents=True, exist_ok=False)
    label = 'DEV' if args.dev else 'GRIDPOS_FORMAL'
    base = {'row': args.row, 'row_spec': row, 'arm': args.arm, 'arm_solver': ARMS[args.arm], 'label': label,
            'command': sys.argv, **git_state(), 'protocol_sha256': sha(args.protocol), 'protocol_committed': committed,
            'protocol_last_commit': protocol_commit, 'seed': SEED, 'device': args.device, 'host': os.uname().nodename,
            'started': datetime.now().astimezone().isoformat(), 'dev_max_steps': args.dev_max_steps,
            'scope': 'tlmpm_v6 CANDIDATE acceptance (not adoption); kinematic ideal tracking; provisional uncalibrated material; '
                     'law 8 and tlmpm_v5 unchanged'}
    write_new(out/'manifest.json', base)
    verify_kind = row['args']['row']['kind'] if row['kind'] == 'verify' else None
    sel = Selection(args.arm, row['kind'], verify_kind)
    if row['kind'] != 'matrix_m7':
        sel.install()
    start = time.monotonic()
    try:
        drv = DRIVERS[row['kind']](row, args, out, sel)
        status = 'COMPLETED'
        error = None
    except BaseException as exc:   # SystemExit from a frozen main included: kept as ERROR
        drv, status, error = {}, 'ERROR', f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    drive_wall = time.monotonic()-start
    mon = sel.final() if row['kind'] != 'matrix_m7' else {'solver_selection': {'verified': True, 'note': 'CPU, derived'}}
    sel.save(out)
    drv['_beds'] = sel.beds
    checks, ok = integrity(row, drv, mon) if status == 'COMPLETED' else ({}, False)
    verdict = drv.get('frozen_verdict')
    extra = {}
    if row['kind'] == 'postcut' and status == 'COMPLETED':
        pc = postcut_verdict(drv, mon)
        if isinstance(pc, tuple):
            verdict, extra = pc
        else:
            bed = sel.beds[-1]
            K = int(sel.materials[-1].max_cuts)
            from robotic_cutting_sim.cut_topology import decode_digits
            before = sel.codes_before_digit[1]['codes']
            pc['digit0_unchanged'] = bool(np.array_equal(decode_digits(before, K)[:, 0], decode_digits(bed.code.numpy(), K)[:, 0]))
            extra = {'postcut_checks': pc}
            verdict = 'PASS' if all(pc.values()) else 'FAIL'
            if args.dev_max_steps:
                verdict += '_NONFORMAL'
    trim = trim_numbers(drv) if row.get('trim') else None
    result = {**base, 'status': status, 'error': error, 'verdict': verdict, 'integrity': {'checks': checks, 'pass': ok},
              'driver': {k: v for k, v in drv.items() if not k.startswith('_')}, **mon, 'trim_numbers': trim, **extra,
              'drive_wall_s': drive_wall, 'wall_s': time.monotonic()-start, 'finished': datetime.now().astimezone().isoformat()}
    write_new(out/'gridpos_result.json', result)
    print(f"[gridpos] {args.row} {args.arm} {status} verdict={verdict} integrity={ok} "
          f"components={(mon.get('final_topology') or {}).get('mechanical_components')} wall={result['wall_s']:.0f}s", flush=True)
    return 0 if status == 'COMPLETED' else 1


# ----------------------------------------------------------------------------- queue

def free_mib(device):
    index = device.split(':')[1]
    q = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', index],
                       capture_output=True, text=True)
    try:
        return int(q.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def queue_jobs(protocol, stage):
    jobs = []
    rows = [r for r in protocol['rows'] if r['stage'] == stage and r.get('scheduled', True)]
    for row in rows:
        for arm in row.get('arms', ['V6', 'V5']):
            if arm == 'V5' and row.get('v5_standin'):
                continue
            jobs.append((row, arm, row['id']))
    for row in rows:                                   # repeats last
        for r in range(2, int(row.get('repeat', 1))+1):
            for arm in row.get('arms', ['V6', 'V5']):
                jobs.append((row, arm, f"{row['id']}_r{r}"))
    return jobs


def cmd_queue(args):
    protocol = load_protocol(args.protocol)
    committed, _ = protocol_committed(args.protocol)
    if not committed:
        raise SystemExit('the queue runs formal rows only: commit the protocol first')
    root = Path(args.out_root)
    if not re.fullmatch(r'gridpos_\d{3}', root.name):
        raise SystemExit('--out-root must be artifacts/wp12/gridpos_NNN')
    logs = ROOT/'artifacts/wp12/logs'
    logs.mkdir(parents=True, exist_ok=True)
    ledger = root/'queue_ledger.jsonl'
    root.mkdir(parents=True, exist_ok=True)
    spent = 0.
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            rec = json.loads(line)
            if rec.get('event') == 'finished':
                spent += rec['wall_s']/3600
    cap = float(protocol['budget']['cap_gpu_h'])
    jobs = queue_jobs(protocol, args.stage)
    lanes = {d: None for d in args.devices}

    def record(**kw):
        with ledger.open('a') as f:
            f.write(json.dumps(jsonable({'time': datetime.now().astimezone().isoformat(), **kw}))+'\n')
    record(event='queue_start', stage=args.stage, jobs=len(jobs), spent_gpu_h=spent, cap_gpu_h=cap, pid=os.getpid())
    pending = list(jobs)
    stop_reason = None
    while pending or any(lanes.values()):
        for device, job in list(lanes.items()):
            if job is None:
                continue
            proc, row, arm, rid, t0, log = job
            if proc.poll() is None:
                continue
            wall = time.monotonic()-t0
            spent += wall/3600
            record(event='finished', row=rid, arm=arm, device=device, returncode=proc.returncode, wall_s=wall, log=str(log))
            lanes[device] = None
            if args.stage == 'S1':
                gate = s1_row_gate(root, rid)
                record(event='s1_gate', row=rid, arm=arm, **gate)
                if gate['fail']:
                    stop_reason = f'S1 kill: {rid} FAIL ({gate["reasons"]})'
        if stop_reason and pending:
            record(event='stop', reason=stop_reason, not_launched=[f'{a}/{i}' for _, a, i in pending])
            pending = []
        for device in lanes:
            if lanes[device] is not None or not pending:
                continue
            row, arm, rid = pending[0]
            out = root/arm/rid
            if out.exists():
                record(event='skip_exists', row=rid, arm=arm)
                pending.pop(0)
                continue
            if row.get('depends_on'):
                missing = [d for d in row['depends_on'] if not (root/arm/d/'gridpos_result.json').exists()]
                if missing:
                    running = any(j is not None and j[3] in missing and j[2] == arm for j in lanes.values())
                    if running or any(r2 == d and a2 == arm for _, a2, r2 in pending[1:] for d in missing):
                        pending.append(pending.pop(0))
                        break
                    record(event='not_run', row=rid, arm=arm, reason=f'missing {missing}')
                    pending.pop(0)
                    continue
            estimate = row['estimated_wall_s_per_arm']/3600
            if spent+estimate > cap:
                record(event='not_run', row=rid, arm=arm, reason=f'budget: spent {spent:.2f} + {estimate:.2f} > cap {cap}')
                pending.pop(0)
                continue
            if row.get('gpu', True) and free_mib(device) < args.min_free_mib:
                continue
            pending.pop(0)
            log = logs/f'{root.name}_{arm}_{rid}.log'
            command = [sys.executable, str(Path(__file__).resolve()), 'run', '--row', rid, '--arm', arm,
                       '--device', device, '--out', str(out)]
            with log.open('a') as stream:
                proc = subprocess.Popen(['/usr/bin/time', '-v', *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'TMPDIR': '/mnt/data/ningan/tmp'})
            lanes[device] = (proc, row, arm, rid, time.monotonic(), log)
            record(event='launched', row=rid, arm=arm, device=device, pid=proc.pid, log=str(log), spent_gpu_h=spent)
        time.sleep(args.poll_s)
    record(event='queue_end', stage=args.stage, spent_gpu_h=spent, stop_reason=stop_reason)
    print(f'[gridpos queue] {args.stage} done; spent {spent:.2f} GPU-h; stop: {stop_reason}', flush=True)
    return 0


def s1_row_gate(root, rid):
    """S1 kill rule for one row id (base or repeat), from both arms where present (summarizer logic, minimal)."""
    sys.path.insert(0, str(ROOT/'scripts'))
    import summarize_wp12_gridpos as S
    return S.s1_gate(root, rid)


def cmd_list(args):
    protocol = load_protocol(args.protocol)
    for r in protocol['rows']:
        print(f"{r['stage']} {r['id']:28s} {r['kind']:14s} {r['gate']:16s} est {r['estimated_wall_s_per_arm']:5d} s/arm "
              f"repeat {r.get('repeat', 1)}{' standin' if r.get('v5_standin') else ''}")
    return 0


def main(argv_=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('list')
    a.set_defaults(fn=cmd_list)
    b = sub.add_parser('run')
    b.add_argument('--row', required=True)
    b.add_argument('--arm', required=True, choices=sorted(ARMS))
    b.add_argument('--device', default='cuda:1')
    b.add_argument('--out', required=True)
    b.add_argument('--dev', action='store_true', help='DEV smoke (label DEV, never evidence)')
    b.add_argument('--dev-max-steps', type=int, help='DEV only: cap the control steps')
    b.set_defaults(fn=cmd_run)
    c = sub.add_parser('queue')
    c.add_argument('--stage', required=True, choices=('S1', 'S2', 'S3'))
    c.add_argument('--out-root', required=True)
    c.add_argument('--devices', nargs='+', default=['cuda:1', 'cuda:0'])
    c.add_argument('--min-free-mib', type=int, default=6000)
    c.add_argument('--poll-s', type=float, default=30.)
    c.set_defaults(fn=cmd_queue)
    args = p.parse_args(argv_)
    if hasattr(args, 'protocol'):
        args.protocol = Path(args.protocol)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
