#!/usr/bin/env python3
"""Read-only substep trace of the v6 flat-board bottom-corner failure."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from robotic_cutting_sim import tlmpm_v5 as v5, tlmpm_v6 as v6
import run_wp12_gridpos as G


@wp.kernel
def trace(x: wp.array(dtype=wp.vec3), ids: wp.array(dtype=int),
          poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
          cuts: wp.array(dtype=int), index: wp.array(dtype=int), stroke: wp.array(dtype=int),
          segment_ok: wp.array(dtype=int), width: float, segments: int,
          values: wp.array2d(dtype=float)):
    j = wp.tid()
    i = index[0]
    if cuts[i] >= 0:
        local = wp.transform_point(wp.transform_inverse(poses[i]), x[ids[j]])
        if local[0] < values[j, 0]:
            values[j, 0] = local[0]
            values[j, 1] = local[1]
            values[j, 2] = local[2]
            values[j, 3] = v5._swept_side(x[ids[j]], poses[i], twists[i], local)
            values[j, 4] = float(stroke[0])
            values[j, 5] = float(segment_ok[v5._edge_segment(local[1], width, segments)])
            values[j, 6] = wp.transform_get_translation(poses[i])[2]
        if stroke[0] != 0:
            values[j, 7] = wp.min(values[j, 7], local[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    original = v6.TLMPM

    class Traced(original):
        def __init__(self, points, *args, **kwargs):
            super().__init__(points, *args, **kwargs)
            mask = (points[:, 2] < .001) & (np.abs(points[:, 0]) < .004) & (np.abs(points[:, 1]) > .014)
            self.trace_ids_host = np.flatnonzero(mask).astype(np.int32)
            self.trace_ids = wp.array(self.trace_ids_host, dtype=int, device=self.device)
            self.trace_values = wp.full((len(self.trace_ids_host), 8), 1.e6, dtype=float, device=self.device)
            wp.load_module(module=__name__, device=self.device)

        def step(self, advance=True):
            super().step(advance=False)
            wp.launch(trace, len(self.trace_ids_host), [self.x, self.trace_ids, self.poses, self.twists,
                self.cuts, self.index, self.stroke, self.segment_ok, self.config.blade_half_width,
                self.edge_segments, self.trace_values], device=self.device)
            if advance:
                wp.launch(v5._advance, 1, [self.index], device=self.device)

    v6.TLMPM = Traced
    sel = G.Selection('V6', 'workcell').install()
    row = G.rows_by_id(G.load_protocol())['FB_CUT']
    args = SimpleNamespace(device=a.device, dev=True, dev_max_steps=None)
    result = G.run_workcell(row, args, a.out, sel)
    bed = sel.beds[-1]
    ids = bed.trace_ids_host
    payload = {'label': 'DIAGNOSTIC_NOT_ACCEPTANCE', 'driver': result,
               'summary': sel.final(), 'columns': ['min_local_forward_m', 'y_at_min_m', 'z_at_min_m',
                 'swept_side_at_min_m', 'stroke_at_min', 'segment_ok_at_min', 'edge_height_at_min_m',
                 'min_forward_during_engaged_stroke_m'],
               'particle_ids': ids, 'reference_m': bed.reference.numpy()[ids],
               'final_positions_m': bed.x.numpy()[ids], 'final_codes': bed.code.numpy()[ids],
               'trace': bed.trace_values.numpy()}
    G.write_new(a.out/'front_trace.json', payload)
    sel.save_arrays(a.out) if hasattr(sel, 'save_arrays') else None
    print(json.dumps({'out': str(a.out), 'verdict': result['frozen_verdict']}))


if __name__ == '__main__':
    main()
