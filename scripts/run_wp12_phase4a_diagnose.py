#!/usr/bin/env python3
"""WP-12 Phase 4a: run an existing WP-11 kinematic runner unchanged with the read-only ``cut_diagnostics_v1`` recorder.

Usage (the wrapped script and its arguments follow ``--``; its ``--out`` is also this tool's output directory)::

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4a_diagnose.py -- \\
      scripts/run_generalization.py --config configs/wp11/generalization_round2.json material \\
      --family OBLIQUE_W --seed 2201 --device cuda:1 --out artifacts/wp12/phase4a_001/s2201

The wrapped runner executes exactly as on the command line. This tool adds read-only hooks only:

* ``tlmpm_v4.TLMPM`` / ``tlmpm_v5.TLMPM`` construction attaches a ``cut_diagnostics.CutDiagnosticsRecorder``
  and ``TLMPM.step`` is bracketed with its ``before_step`` / ``after_step`` (inside the runtime's CUDA graph);
* ``LiveCuttingMaterial.advance``: after each control step, the recorder's per-segment sums are moved to the host,
  plus the solver-local start/end edge poses, contact statistics, reaction impulse and cut digit;
* ``KinematicCuttingBackend.close``: the final solver snapshot, reference lattice and 8-node stencil and all
  diagnostics are written to ``wp12_phase4a_diag.npz`` / ``wp12_phase4a_diag.json``.

``--counterfactual KEY=VALUE`` (repeatable, before ``--``) overrides one law parameter on the solver instance
(``theta_max_deg``, ``theta_plow_deg``, ``band_h_m``, ``rear_b_back_m``, ``persist_b_persist_m``,
``lead_b_lead_m``, ``core_s_core_m``) before the runtime captures its graph. Such a run is a labelled
COUNTERFACTUAL diagnostic of what one condition decides; it is not cutting law 8 and never an acceptance
result. Without it no physics, law parameter or threshold is touched.
"""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

LAW_KEYS = {'theta_max_deg': 'law4_parameters', 'band_h_m': 'law4_parameters', 'rear_b_back_m': 'law4_parameters',
            'persist_b_persist_m': 'law4_parameters', 'min_speed_m_s': 'law4_parameters',
            'lead_b_lead_m': ('law5_parameters', 'law8_parameters'), 'core_s_core_m': ('law5_parameters', 'law8_parameters'),
            'theta_plow_deg': 'law8_parameters'}


def _out_dir(argv):
    for i, arg in enumerate(argv):
        if arg == '--out' and i+1 < len(argv):
            return Path(argv[i+1])
        if arg.startswith('--out='):
            return Path(arg.split('=', 1)[1])
    raise SystemExit('The wrapped command needs --out')


def _git():
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'git_sha': run('rev-parse', 'HEAD'), 'git_dirty_paths': run('status', '--porcelain').splitlines()}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    if '--' not in sys.argv:
        raise SystemExit(__doc__)
    split = sys.argv.index('--')
    own, target, args = sys.argv[1:split], sys.argv[split+1], sys.argv[split+2:]
    counterfactual = {}
    i = 0
    while i < len(own):
        if own[i] == '--counterfactual':
            key, value = own[i+1].split('=', 1)
            if key not in LAW_KEYS:
                raise SystemExit(f'Unknown law parameter {key!r}; one of {sorted(LAW_KEYS)}')
            counterfactual[key] = float(value)
            i += 2
        else:
            raise SystemExit(f'Unknown option {own[i]!r}')
    out = _out_dir(args)
    import warp as wp
    from robotic_cutting_sim import cut_diagnostics, kinematic_env, live_material, tlmpm_v4, tlmpm_v5
    state = {'recorders': [], 'steps': [], 'extra': {}, 'overrides_applied': []}

    def patch_solver(module):
        original_init, original_step = module.TLMPM.__init__, module.TLMPM.step

        def init(self, *a, **kw):
            original_init(self, *a, **kw)
            for key, value in counterfactual.items():
                targets = LAW_KEYS[key] if isinstance(LAW_KEYS[key], tuple) else (LAW_KEYS[key],)
                for name in targets:
                    if hasattr(self, name) and key in getattr(self, name):
                        before = getattr(self, name)[key]
                        getattr(self, name)[key] = value
                        state['overrides_applied'].append({'module': module.__name__, 'dict': name, 'key': key,
                                                           'frozen_value': before, 'counterfactual_value': value})
            if int(getattr(self, 'cutting_law', -1)) in (4, 5, 8):
                self._wp12_diag = cut_diagnostics.CutDiagnosticsRecorder(self)
                state['recorders'].append(self._wp12_diag)
                if wp.get_device(self.device).is_cuda:
                    wp.load_module(module='robotic_cutting_sim.cut_diagnostics', device=self.device)

        def step(self, advance=True):
            recorder = getattr(self, '_wp12_diag', None)
            if recorder is not None:
                recorder.before_step()
            original_step(self, advance)
            if recorder is not None:
                recorder.after_step(advanced=advance)

        module.TLMPM.__init__, module.TLMPM.step = init, step

    patch_solver(tlmpm_v4)
    patch_solver(tlmpm_v5)
    original_advance = live_material.LiveCuttingMaterial.advance
    original_close = kinematic_env.KinematicCuttingBackend.close

    def advance(self, measured_edge_pose, time_s, **kwargs):
        if 'T_world_local' not in state['extra']:
            state['extra']['T_world_local'] = np.asarray(self.T_world_local, float)
        start_local = np.asarray(self._pose_local(self.last_pose), np.float32)
        result = original_advance(self, measured_edge_pose, time_s, **kwargs)
        bed = self.bed
        recorder = getattr(bed, '_wp12_diag', None)
        if recorder is not None:
            recorder.end_control_step()
        state['steps'].append({'time_s': float(self.time_s), 'cut': int(result['cut_index']),
                               'pose_start_local': start_local,
                               'pose_end_local': np.asarray(self._pose_local(self.last_pose), np.float32),
                               'stats': bed.stats.numpy().astype(np.float32).copy(),
                               'reaction_impulse_local': bed.reaction.numpy()[0].astype(np.float32).copy(),
                               'labelled': int(np.count_nonzero(bed.code.numpy()))})
        return result

    def close(self):
        material = self.material
        if material is not None and getattr(material, 'bed', None) is not None and 'final' not in state['extra']:
            bed = material.bed
            extra = state['extra']
            extra['final'] = {k: np.asarray(v) for k, v in bed.snapshot().items()}
            extra['final_time_s'] = float(self.time_s)
            extra['reference_local'] = bed.reference.numpy().astype(np.float32)
            extra['lattice'] = np.asarray(material.lattice, np.int32)
            extra['mass_kg'] = np.asarray(material.masses, np.float64)
            extra['stencil_nodes'] = bed.nodes.numpy()
            extra['stencil_weights'] = bed.weights.numpy()
            extra['stencil_gradients'] = bed.gradients.numpy()
            extra['meta'] = {'substeps': int(material.substeps), 'control_dt_s': float(material.config.control_dt),
                             'spacing_m': float(material.spacing), 'grid_dx_m': float(bed.config.dx),
                             'material_dt_s': float(bed.config.dt),
                             'code_encoding': getattr(bed, 'code_encoding', 'ternary'),
                             'max_cuts': int(getattr(material, 'max_cuts', 1)), 'solver_module': type(bed).__module__,
                             'cutting_law': int(bed.cutting_law), 'edge_segments': int(bed.edge_segments),
                             'blade_half_width_m': float(bed.config.blade_half_width),
                             'blade_profile_xz': np.asarray(bed.blade_profile_xz, float).tolist(),
                             'law_parameters': {k: dict(getattr(bed, k)) for k in
                                                ('law4_parameters', 'law5_parameters', 'law8_parameters')
                                                if hasattr(bed, k)},
                             'fixture_boxes_local': [[list(map(float, b.center_m)), list(map(float, b.half_extents_m)), b.name]
                                                     for b in getattr(material.task, 'fixture_boxes_local', ())]
                             if material.task else [],
                             'task_id': getattr(material.task, 'task_id', None)}
            recorder = getattr(bed, '_wp12_diag', None)
            if recorder is not None:
                extra['diag_arrays'] = recorder.arrays()
                extra['diag_meta'] = recorder.metadata()
        return original_close(self)

    live_material.LiveCuttingMaterial.advance = advance
    kinematic_env.KinematicCuttingBackend.close = close
    sys.argv = [target, *args]
    status, error, start = 0, None, time.monotonic()
    try:
        runpy.run_path(str(ROOT/target if not os.path.isabs(target) else target), run_name='__main__')
    except SystemExit as exit_:
        status = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    except Exception:
        status, error = 1, traceback.format_exc()
        print(error, file=sys.stderr)
    wall = time.monotonic()-start
    if not out.is_dir():
        return status
    extra, steps = state['extra'], state['steps']
    if 'final' not in extra and state['recorders']:
        # Runner without KinematicCuttingBackend (e.g. a direct TLMPM probe): dump the last solver at exit.
        recorder = state['recorders'][-1]
        bed = recorder.bed
        extra['final'] = {k: np.asarray(v) for k, v in bed.snapshot().items()}
        extra['reference_local'] = bed.reference.numpy().astype(np.float32)
        extra['mass_kg'] = bed.mass.numpy().astype(np.float64)
        spacing = float(bed.spacing)
        extra['lattice'] = np.round((extra['reference_local']-extra['reference_local'].min(0))/spacing).astype(np.int32)
        extra['stencil_nodes'], extra['stencil_weights'] = bed.nodes.numpy(), bed.weights.numpy()
        extra['stencil_gradients'] = bed.gradients.numpy()
        extra['meta'] = {'substeps': None, 'control_dt_s': None, 'spacing_m': spacing, 'grid_dx_m': float(bed.config.dx),
                         'material_dt_s': float(bed.config.dt), 'code_encoding': getattr(bed, 'code_encoding', 'ternary'),
                         'max_cuts': int(getattr(bed, 'max_cuts', bed.config.cuts)), 'solver_module': type(bed).__module__,
                         'cutting_law': int(bed.cutting_law), 'edge_segments': int(bed.edge_segments),
                         'blade_half_width_m': float(bed.config.blade_half_width), 'task_id': None,
                         'dumped_at': 'process exit (no KinematicCuttingBackend.close)'}
        extra['diag_arrays'] = recorder.arrays()
        extra['diag_meta'] = recorder.metadata()
    arrays = {}
    if steps:
        arrays.update(step_time_s=np.array([s['time_s'] for s in steps], np.float64),
                      step_cut=np.array([s['cut'] for s in steps], np.int16),
                      step_pose_start_local=np.stack([s['pose_start_local'] for s in steps]),
                      step_pose_end_local=np.stack([s['pose_end_local'] for s in steps]),
                      step_stats=np.stack([s['stats'] for s in steps]),
                      step_reaction_impulse_local=np.stack([s['reaction_impulse_local'] for s in steps]),
                      step_labelled=np.array([s['labelled'] for s in steps], np.int32))
    for key in ('T_world_local', 'reference_local', 'lattice', 'mass_kg', 'stencil_nodes', 'stencil_weights',
                'stencil_gradients'):
        if key in extra:
            arrays[key] = extra[key]
    for key, value in (extra.get('final') or {}).items():
        arrays[f'final_{key}'] = value
    for key, value in (extra.get('diag_arrays') or {}).items():
        arrays[f'diag_{key}'] = value
    target_file = out/'wp12_phase4a_diag.npz'
    if target_file.exists():
        raise SystemExit(f'{target_file} exists; never overwritten')
    np.savez_compressed(target_file, **arrays)
    meta = {'tool': 'scripts/run_wp12_phase4a_diagnose.py', 'wrapped_command': [target, *args],
            'wrapped_exit': status, 'wrapped_error': error, 'wall_s': wall, **_git(),
            'counterfactual': counterfactual or None,
            'counterfactual_label': ('COUNTERFACTUAL diagnostic: law parameter(s) overridden on the solver instance; '
                                     'not cutting law 8, never an acceptance result') if counterfactual else None,
            'overrides_applied': state['overrides_applied'], 'solvers_constructed': len(state['recorders']),
            'final_time_s': extra.get('final_time_s'), 'meta': extra.get('meta'),
            'cut_diagnostics': extra.get('diag_meta'),
            'stats_layout': ['max_fixture_or_table_penetration_m', 'tool_contact_particles', 'tool_power_sum',
                             'max_tool_penetration_m', 'particle_contact_pairs', 'particle_contact_impulse_sum',
                             'max_particle_overlap_m'],
            'scope': 'read-only diagnostics of an unchanged WP-11 kinematic runner (oracle data)'}
    (out/'wp12_phase4a_diag.json').write_text(json.dumps(plain(meta), indent=1, default=str))
    return status


if __name__ == '__main__':
    sys.exit(main())
