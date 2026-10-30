#!/usr/bin/env python3
"""WP-12 Phase 0.3: run an existing WP-11 kinematic runner unchanged and record extra read-only state.

Usage (the wrapped script and its arguments follow ``--``; its ``--out`` is also this tool's output dir):

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/wp12_baseline_capture.py -- \
      scripts/run_generalization.py --config configs/wp11/generalization_round1.json material \
      --family TEST_B --seed 1502 --device cuda:1 --out artifacts/wp12/baseline/s3_testb_s1502_002

The wrapped runner (``run_generalization.py material``, ``run_phase1_matrix.py``,
``run_phase_c4_trim_material.py``) executes exactly as on the command line. This tool only adds
two read-only hooks and writes ``wp12_capture.npz`` + ``wp12_capture.json`` next to the runner's output:

* ``LiveCuttingMaterial.advance``: after each control step, host copies of the solver's per-step
  contact statistics (``bed.stats``: [0] max fixture/table penetration, [1] tool-contact particle count,
  [2] tool power sum, [3] max tool penetration, [4] particle-contact pair count, [5] particle-contact
  impulse sum, [6] max particle overlap), the edge-segment masks of the LAST substep (law-4 ``segment_ok``
  and law-8 ``segment_plow`` counts), stroke state, labelled-particle count, the commanded cut digit and
  the solver-local start/end edge poses of the step (the per-substep poses are exactly
  ``coupled_robot.interpolate_measured_poses(start, end, substeps, control_dt)``). At the first step it also
  extracts the initial ParticleSurface mesh.
* ``KinematicCuttingBackend.close``: before the material is released, ``bed.snapshot()`` (positions,
  velocities, codes, F, Fpi, plastic strain, cut progress, contact work; solver-local frame) and the final
  ParticleSurface mesh, built exactly as the kinematic/live runtimes do (``cutting_surface.ComponentSurface``,
  profile ``current_volume_v1``, voxel = spacing/2; ``unmeshable='raise'`` and, only if that raises, a
  recorded retry with ``skip_presentation_v1``).

Reading device arrays after the control step's synchronize does not change any solver state, label,
force or evaluation. No physics, law parameter or threshold is touched.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def _out_dir(argv):
    for i, arg in enumerate(argv):
        if arg == '--out' and i+1 < len(argv):
            return Path(argv[i+1])
        if arg.startswith('--out='):
            return Path(arg.split('=', 1)[1])
    raise SystemExit('The wrapped command needs --out')


def _surface(bed, spacing, device, substrate):
    from robotic_cutting_sim.cutting_surface import ComponentSurface
    record = {'profile': 'current_volume_v1', 'voxel_size_m': spacing/2, 'policy': 'raise'}
    start = time.perf_counter()
    try:
        vertices, faces, info = ComponentSurface(spacing, device, voxel_size=spacing/2,
                                                 profile='current_volume_v1').extract(bed)
    except Exception as error:  # recorded; presentation-only retry, as live_003 does
        record.update(raise_error=repr(error), policy='skip_presentation_v1')
        vertices, faces, info = ComponentSurface(spacing, device, voxel_size=spacing/2, profile='current_volume_v1',
                                                 unmeshable='skip_presentation_v1',
                                                 substrate_anchors=substrate).extract(bed)
    record['wall_s'] = time.perf_counter()-start
    record['info'] = {k: v for k, v in info.items() if k != 'component_ranges'}
    record['component_ranges'] = info.get('component_ranges')
    return np.asarray(vertices, np.float32), np.asarray(faces, np.int32), record


def main():
    if '--' not in sys.argv:
        raise SystemExit(__doc__)
    split = sys.argv.index('--')
    target, args = sys.argv[split+1], sys.argv[split+2:]
    out = _out_dir(args)
    from robotic_cutting_sim import kinematic_env, live_material
    from robotic_cutting_sim.cutting_surface import substrate_anchor_mask
    steps, extra = [], {}
    original_advance = live_material.LiveCuttingMaterial.advance
    original_close = kinematic_env.KinematicCuttingBackend.close

    def advance(self, measured_edge_pose, time_s, **kwargs):
        if 'initial_surface' not in extra:
            extra['T_world_local'] = np.asarray(self.T_world_local, float)
            extra['reference_local'] = self.bed.reference.numpy().astype(np.float32)
            extra['substrate_anchor'] = substrate_anchor_mask(self.points)
            try:
                extra['initial_surface'] = _surface(self.bed, self.spacing, self.config.device, extra['substrate_anchor'])
            except Exception as error:
                extra['initial_surface'] = None
                extra['initial_surface_error'] = traceback.format_exc()
        start_local = np.asarray(self._pose_local(self.last_pose), np.float32)
        result = original_advance(self, measured_edge_pose, time_s, **kwargs)
        bed = self.bed
        state = bed.cutting_state() if hasattr(bed, 'cutting_state') else {}
        steps.append({
            'time_s': float(self.time_s), 'cut': int(result['cut_index']),
            'pose_start_local': start_local, 'pose_end_local': np.asarray(self._pose_local(self.last_pose), np.float32),
            'stats': bed.stats.numpy().astype(np.float32).copy(),
            'segment_ok': int(bed.segment_ok.numpy().sum()) if hasattr(bed, 'segment_ok') else -1,
            'segment_plow': int(bed.segment_plow.numpy().sum()) if hasattr(bed, 'segment_plow') else -1,
            'stroke': int(state.get('stroke', -1)), 'engaged': int(state.get('engaged', -1)),
            'labelled': int(np.count_nonzero(bed.code.numpy())),
            'reaction_impulse_local': bed.reaction.numpy()[0].astype(np.float32).copy()})
        return result

    def close(self):
        material = self.material
        if material is not None and getattr(material, 'bed', None) is not None and 'final' not in extra:
            bed = material.bed
            extra['final'] = {k: np.asarray(v) for k, v in bed.snapshot().items()}
            extra['final_time_s'] = float(self.time_s)
            extra['edge_segments'] = int(getattr(bed, 'edge_segments', 0))
            extra['substeps'] = int(material.substeps)
            extra['control_dt_s'] = float(material.config.control_dt)
            extra['spacing_m'] = float(material.spacing)
            extra['grid_dx_m'] = float(bed.config.dx)
            extra['code_encoding'] = getattr(bed, 'code_encoding', 'ternary')
            extra['max_cuts'] = int(getattr(material, 'max_cuts', 1))
            extra['lattice'] = np.asarray(material.lattice, np.int32)
            extra['blade_profile_xz'] = np.asarray(getattr(bed, 'blade_profile_xz', np.zeros((0, 2))), np.float32)
            extra['blade_half_width_m'] = float(bed.config.blade_half_width)
            extra['law_parameters'] = {k: getattr(bed, k) for k in ('law4_parameters', 'law5_parameters', 'law8_parameters')
                                       if hasattr(bed, k)}
            extra['fixture_boxes_local'] = [[list(map(float, b.center_m)), list(map(float, b.half_extents_m)), b.name]
                                            for b in getattr(material.task, 'fixture_boxes_local', ())] if material.task else []
            try:
                extra['final_surface'] = _surface(bed, material.spacing, material.config.device, extra.get('substrate_anchor'))
            except Exception:
                extra['final_surface'] = None
                extra['final_surface_error'] = traceback.format_exc()
        return original_close(self)

    live_material.LiveCuttingMaterial.advance = advance
    kinematic_env.KinematicCuttingBackend.close = close
    sys.argv = [target, *args]
    status, error = 0, None
    try:
        runpy.run_path(str(ROOT/target if not os.path.isabs(target) else target), run_name='__main__')
    except SystemExit as exit_:
        status = exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    except Exception:
        status, error = 1, traceback.format_exc()
        print(error, file=sys.stderr)
    if not out.is_dir():
        return status
    arrays, meta = {}, {'wrapped_command': [target, *args], 'wrapped_exit': status, 'wrapped_error': error,
                        'scope': __doc__.split('\n\n')[2].strip()}
    if steps:
        arrays.update(step_time_s=np.array([s['time_s'] for s in steps], np.float64),
                      step_cut=np.array([s['cut'] for s in steps], np.int16),
                      step_pose_start_local=np.stack([s['pose_start_local'] for s in steps]),
                      step_pose_end_local=np.stack([s['pose_end_local'] for s in steps]),
                      step_stats=np.stack([s['stats'] for s in steps]),
                      step_segment_ok=np.array([s['segment_ok'] for s in steps], np.int32),
                      step_segment_plow=np.array([s['segment_plow'] for s in steps], np.int32),
                      step_stroke=np.array([s['stroke'] for s in steps], np.int16),
                      step_engaged=np.array([s['engaged'] for s in steps], np.int16),
                      step_labelled=np.array([s['labelled'] for s in steps], np.int32),
                      step_reaction_impulse_local=np.stack([s['reaction_impulse_local'] for s in steps]))
    for key in ('T_world_local', 'reference_local', 'substrate_anchor', 'lattice', 'blade_profile_xz'):
        if key in extra:
            arrays[key] = extra[key]
    for key, value in (extra.get('final') or {}).items():
        arrays[f'final_{key}'] = value.astype(np.float32) if value.dtype.kind == 'f' else value
    for name in ('initial_surface', 'final_surface'):
        surface = extra.get(name)
        if surface is not None:
            arrays[f'{name}_vertices_local'], arrays[f'{name}_faces'] = surface[0], surface[1]
            meta[name] = surface[2]
        elif f'{name}_error' in extra:
            meta[name] = {'error': extra[f'{name}_error']}
    for key in ('final_time_s', 'edge_segments', 'substeps', 'control_dt_s', 'spacing_m', 'grid_dx_m', 'code_encoding',
                'max_cuts', 'blade_half_width_m', 'law_parameters', 'fixture_boxes_local'):
        if key in extra:
            meta[key] = extra[key]
    meta['stats_layout'] = ['max_fixture_or_table_penetration_m', 'tool_contact_particles', 'tool_power_sum',
                            'max_tool_penetration_m', 'particle_contact_pairs', 'particle_contact_impulse_sum',
                            'max_particle_overlap_m']
    meta['segment_masks'] = 'counts of edge segments with law-4 M_k (segment_ok) / law-8 plow mask at the LAST substep'
    np.savez_compressed(out/'wp12_capture.npz', **arrays)

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
    (out/'wp12_capture.json').write_text(json.dumps(plain(meta), indent=1, default=str))
    return status


if __name__ == '__main__':
    sys.exit(main())
