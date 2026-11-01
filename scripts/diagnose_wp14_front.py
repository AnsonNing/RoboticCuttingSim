#!/usr/bin/env python3
"""Fresh, read-only substep observations of corrected-path TRIM bridges.

Selected ids come from a completed failed episode. The traced episode is a new
GPU run; its final bridges can differ. No historical crossing is reconstructed.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import warp as wp

from run_wp13 import ROOT, G, build, run
from robotic_cutting_sim import tlmpm_v5 as V5, tlmpm_v7 as V7

COLUMNS = ['substep', 'cut', 'old_forward_m', 'new_forward_m', 'old_y_m', 'new_y_m',
           'old_z_m', 'new_z_m', 'swept_side_m', 'stroke', 'segment_ok', 'engaged',
           'reached_digit', 'side_digit', 'progress', 'consumed_work_increment_J',
           'cumulative_particle_work_J', 'blade_gap_before_m', 'board_gap_after_m',
           'world_local_dx_m', 'world_local_dy_m', 'world_local_dz_m',
           'impulse_x_Ns', 'impulse_y_Ns', 'impulse_z_Ns', 'edge_x_m', 'edge_y_m', 'edge_z_m',
           'link_old_forward_m', 'link_new_forward_m', 'partner_swept_side_m']
SAMPLES = ['first_crossing', 'first_particle_work', 'closest_front', 'last_bevel',
           'first_reached', 'first_assigned', 'deepest_board', 'closest_engaged_front',
           'first_finite_link_crossing', 'first_engaged_finite_link_crossing']
COUNTERS = ['armed_substeps', 'leading_crossings', 'finite_crossings', 'bevel_residence_substeps',
            'engaged_bevel_substeps', 'particle_work_substeps', 'reached_but_no_stroke',
            'reached_but_outside_bevel', 'link_plane_intersection_substeps', 'finite_link_front_crossings',
            'engaged_finite_link_front_crossings']


@wp.kernel
def before(ids: wp.array(dtype=int), work: wp.array(dtype=float),
           x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
           poses: wp.array(dtype=wp.transform), previous: wp.array(dtype=wp.transform),
           saved_pose: wp.array(dtype=wp.transform), pre: wp.array2d(dtype=float)):
    j = wp.tid()
    p = ids[j]
    pre[j, 0] = work[p]
    pre[j, 1] = v[p][0]; pre[j, 2] = v[p][1]; pre[j, 3] = v[p][2]
    if j == 0:
        saved_pose[0] = previous[0]


@wp.kernel
def observe(ids: wp.array(dtype=int), partner: wp.array(dtype=int),
            x0: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
            code: wp.array(dtype=int), work: wp.array(dtype=float), progress: wp.array(dtype=float),
            impulses: wp.array(dtype=wp.vec3), poses: wp.array(dtype=wp.transform),
            previous: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
            cuts: wp.array(dtype=int), index: wp.array(dtype=int), stroke: wp.array(dtype=int),
            segment: wp.array(dtype=int), engaged: wp.array(dtype=int), reached: wp.array(dtype=int),
            profile: wp.array(dtype=wp.vec2), profile_count: int, width: float, band: float,
            bevel: float, radius: float, segments: int,
            centers: wp.array(dtype=wp.vec3), halves: wp.array(dtype=wp.vec3), plane: wp.vec4,
            step: wp.array(dtype=int), pre: wp.array2d(dtype=float),
            counts: wp.array2d(dtype=int), keys: wp.array2d(dtype=float),
            saved: wp.array3d(dtype=float)):
    j = wp.tid(); p = ids[j]; q = partner[j]
    i = index[0]; cut = cuts[i]
    if cut < 0:
        return
    a = wp.transform_point(wp.transform_inverse(previous[0]), x0[p])
    b = wp.transform_point(wp.transform_inverse(poses[i]), x[p])
    side = V5._swept_side(x[p], poses[i], twists[i], b)
    seg = segment[V5._edge_segment(b[1], width, segments)]
    d = V5._digit(code[p], cut); rd = V5._digit(reached[p], cut)
    crossing = a[0] >= 0.0 and b[0] <= 0.0 and a[0] > b[0]
    finite = False
    if crossing:
        t = a[0]/(a[0]-b[0])
        y = a[1]+t*(b[1]-a[1])
        finite = wp.abs(y) <= width and wp.abs(side) <= band and segment[V5._edge_segment(y, width, segments)] != 0
    bevel_here = b[0] <= 0.0 and b[0] >= -bevel and wp.abs(b[1]) <= width
    dw = work[p]-pre[j, 0]
    board_gap = wp.dot(wp.vec3(plane[0], plane[1], plane[2]), x[p])+plane[3]-radius
    for k in range(centers.shape[0]):
        hit = V5._box_sdf(x[p]-centers[k], halves[k])
        board_gap = wp.min(board_gap, hit[3]-radius)
    old_local = wp.transform_point(wp.transform_inverse(poses[i]), x0[p])
    blade = V5._profile_sdf(old_local, profile, profile_count, width)
    counts[j, 0] += 1
    if crossing: counts[j, 1] += 1
    if finite: counts[j, 2] += 1
    if bevel_here: counts[j, 3] += 1
    if bevel_here and stroke[0] == cut+1 and seg != 0: counts[j, 4] += 1
    if dw > 0.0: counts[j, 5] += 1
    if rd != 0 and d == 0 and stroke[0] != cut+1: counts[j, 6] += 1
    if rd != 0 and d == 0 and not bevel_here: counts[j, 7] += 1
    # Diagnostic intersection of the moving reference-neighbour segment with
    # the swept-side plane. Evaluate old/new intersection points separately;
    # only a finite leading-plane traversal counts. This never assigns codes.
    aq = wp.transform_point(wp.transform_inverse(previous[0]), x0[q])
    bq = wp.transform_point(wp.transform_inverse(poses[i]), x[q])
    sa = V5._swept_side(x0[p], previous[0], twists[i], a)
    sqa = V5._swept_side(x0[q], previous[0], twists[i], aq)
    sqb = V5._swept_side(x[q], poses[i], twists[i], bq)
    link_crossing = False
    link_a = float(1.e20); link_b = float(1.e20)
    if sa*sqa < 0.0 and side*sqb < 0.0:
        counts[j, 8] += 1
        la = a+(aq-a)*(sa/(sa-sqa))
        lb = b+(bq-b)*(side/(side-sqb))
        link_a = la[0]; link_b = lb[0]
        if la[0] >= 0.0 and lb[0] <= 0.0 and la[0] > lb[0]:
            t = la[0]/(la[0]-lb[0])
            y = la[1]+t*(lb[1]-la[1])
            if wp.abs(y) <= width:
                counts[j, 9] += 1
                link_crossing = True
                if stroke[0] == cut+1:
                    counts[j, 10] += 1
    for slot in range(10):
        update = False
        key = float(step[0])
        if slot == 0: update = crossing and keys[j, slot] == 1.e20
        elif slot == 1: update = dw > 0.0 and keys[j, slot] == 1.e20
        elif slot == 2:
            key = wp.abs(b[0]); update = key < keys[j, slot]
        elif slot == 3: update = bevel_here
        elif slot == 4: update = rd != 0 and keys[j, slot] == 1.e20
        elif slot == 5: update = d != 0 and keys[j, slot] == 1.e20
        elif slot == 6:
            key = board_gap; update = key < keys[j, slot]
        elif slot == 7:
            key = wp.abs(b[0]); update = stroke[0] == cut+1 and key < keys[j, slot]
        elif slot == 8: update = link_crossing and keys[j, slot] == 1.e20
        elif slot == 9: update = link_crossing and stroke[0] == cut+1 and keys[j, slot] == 1.e20
        if update:
            keys[j, slot] = key
            saved[j, slot, 0] = float(step[0]); saved[j, slot, 1] = float(cut)
            saved[j, slot, 2] = a[0]; saved[j, slot, 3] = b[0]
            saved[j, slot, 4] = a[1]; saved[j, slot, 5] = b[1]
            saved[j, slot, 6] = a[2]; saved[j, slot, 7] = b[2]
            saved[j, slot, 8] = side; saved[j, slot, 9] = float(stroke[0])
            saved[j, slot, 10] = float(seg); saved[j, slot, 11] = float(engaged[0])
            saved[j, slot, 12] = float(rd); saved[j, slot, 13] = float(d)
            saved[j, slot, 14] = progress[p]; saved[j, slot, 15] = dw
            saved[j, slot, 16] = work[p]; saved[j, slot, 17] = blade[3]-radius
            saved[j, slot, 18] = board_gap
            delta = x[p]-x0[p]
            saved[j, slot, 19] = delta[0]; saved[j, slot, 20] = delta[1]; saved[j, slot, 21] = delta[2]
            saved[j, slot, 22] = impulses[p][0]; saved[j, slot, 23] = impulses[p][1]; saved[j, slot, 24] = impulses[p][2]
            origin = wp.transform_get_translation(poses[i])
            saved[j, slot, 25] = origin[0]; saved[j, slot, 26] = origin[1]; saved[j, slot, 27] = origin[2]
            saved[j, slot, 28] = link_a; saved[j, slot, 29] = link_b; saved[j, slot, 30] = sqb


@wp.kernel
def increment(step: wp.array(dtype=int)):
    step[0] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--links', choices=('nearest_vertical', 'actual_support'), default='nearest_vertical')
    a = p.parse_args()
    original_summary = json.loads((a.source_run/'summary.json').read_text())
    if original_summary['class'] == 'task_success':
        raise ValueError('Trace requires a diagnosed residual failure')
    row = original_summary['case']; task, _ = build(row, 'tlmpm_v7')
    state = np.load(a.source_run/'final_state.npz')
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.cut_diagnostics import direct_bridges
    from robotic_cutting_sim.root_cause_diagnostics import lattice_links
    ref = state['reference']; codes = state['codes']
    probe = V5.TLMPM(ref, np.full(len(ref), task.material.density_kg_m3*task.particle_spacing_m**3),
                    task.particle_spacing_m, TLMPMConfig(dx=task.grid_dx_m, cuts=task.max_cuts), 'cpu')
    ids = direct_bridges(probe.nodes.numpy(), probe.weights.numpy(), probe.gradients.numpy(), codes & 3).astype(np.int32)
    if not len(ids):
        raise ValueError('No direct bridges; diagnose a different residual before tracing')
    neighbours = {int(i): [] for i in ids}
    for left, right, _ in lattice_links(state['lattice']):
        for u, v in zip(left, right):
            if int(u) in neighbours: neighbours[int(u)].append(int(v))
            if int(v) in neighbours: neighbours[int(v)].append(int(u))
    partner = np.array([max(neighbours[int(i)], key=lambda j: abs(ref[j, 2]-ref[i, 2])) for i in ids], np.int32)
    if a.links == 'actual_support':
        nodes, weights, gradients = probe.nodes.numpy(), probe.weights.numpy(), probe.gradients.numpy()
        valid = (weights > 0) | (np.linalg.norm(gradients, axis=-1) > 0)
        positive = np.flatnonzero((codes & 3) == 2)
        node_particles = {}
        for q in positive:
            for node in nodes[q][valid[q]]:
                node_particles.setdefault(int(node), set()).add(int(q))
        partners = []
        for i in ids:
            candidates = set().union(*(node_particles.get(int(node), set()) for node in nodes[i][valid[i]]))
            if not candidates:
                raise ValueError(f'No positive-side support neighbour for direct bridge {i}')
            partners.append(min(candidates, key=lambda q: (float(np.linalg.norm(ref[q]-ref[i])), q)))
        partner = np.array(partners, np.int32)
    original = V7.TLMPM; beds = []

    class Traced(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with wp.ScopedDevice(self.device):
                self.tids = wp.array(ids, dtype=int); self.partners = wp.array(partner, dtype=int)
                self.pre = wp.zeros((len(ids), 4), dtype=float)
                self.saved_pose = wp.zeros(1, dtype=wp.transform)
                self.tstep = wp.zeros(1, dtype=int)
                self.counts = wp.zeros((len(ids), len(COUNTERS)), dtype=int)
                self.keys = wp.full((len(ids), len(SAMPLES)), 1.e20, dtype=float)
                self.samples = wp.zeros((len(ids), len(SAMPLES), len(COLUMNS)), dtype=float)
            wp.load_module(module=__name__, device=self.device)
            beds.append(self)

        def step(self, advance=True):
            with wp.ScopedDevice(self.device):
                wp.launch(before, len(ids), [self.tids, self.contact_work, self.x, self.v, self.poses,
                                            self.previous_pose, self.saved_pose, self.pre])
                super().step(advance=False)
                _, centers, halves, plane = self.fixture_data
                wp.launch(observe, len(ids), [self.tids, self.partners, self.x_start, self.x, self.code,
                    self.contact_work, self.progress, self.impulse, self.poses, self.saved_pose, self.twists,
                    self.cuts, self.index, self.stroke, self.segment_ok, self.stroke_engaged, self.front_reached,
                    self.profile, len(self.blade_profile_xz), self.config.blade_half_width,
                    self.law4_parameters['band_h_m'], self.bevel_depth_m, self.radius, self.edge_segments,
                    centers, halves, plane, self.tstep, self.pre, self.counts, self.keys, self.samples])
                wp.launch(increment, 1, [self.tstep])
                if advance: wp.launch(V5._advance, 1, [self.index])

    args = argparse.Namespace(case=ROOT/'configs/wp14/cases'/f"{row['id']}.json", episode=row['episode'],
        solver='tlmpm_v7', dt=None, out=a.out, device=a.device, surface='particle_surface_v1',
        label='WP14_RESIDUAL_DIAGNOSIS_NOT_ACCEPTANCE', planner_source='compiled')
    with patch.object(V7, 'TLMPM', Traced):
        code = run(args)
    if code: return code
    bed = beds[-1]
    counts = bed.counts.numpy(); keys = bed.keys.numpy(); samples = bed.samples.numpy()
    G.write_new(a.out/'front_trace.json', dict(source_run=str(a.source_run), source_sha256=G.sha(a.source_run/'summary.json'),
        particle_ids=ids, partner_ids=partner, link_selection=a.links,
        diagnostic_source_sha256=G.sha(Path(__file__)), reference_m=ref[ids], reference_partner_m=ref[partner],
        final_positions_m=bed.x.numpy()[ids], final_codes=bed.code.numpy()[ids],
        final_reached=bed.front_reached.numpy()[ids], counters=COUNTERS, counts=counts,
        sample_names=SAMPLES, columns=COLUMNS, sample_valid=keys != np.float32(1.e20), samples=samples,
        material_dt_s=bed.config.dt, scope='Actual fresh substeps. Read-only selected particle/nearest vertical link observations; '
        'counter times include the full episode. Link test uses old/new swept-plane intersection, not a fracture criterion. '
        'Impulse is all contacts combined; consumed work is particle cutting work. No historical event is inferred.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
