#!/usr/bin/env python3
"""WP-12 Phase 4a controlled sweeps that isolate the law-8 failure mechanisms (diagnostics, not acceptance runs).

Each row drives the edge of ``ruler_knife_v2_y`` kinematically (ideal tracking, 20 ms control steps interpolated
into 200 substeps exactly like ``LiveCuttingMaterial.advance``) through a lattice block on the round-2
``wp11_workcell_v1`` support semantics (10 mm board on risers, table 100 mm lower), with the formal resolution
(1 mm particles, 4 mm grid, 0.1 ms) and the provisional WP-11 material (E 50 kPa, nu 0.3, yield 1 kPa, rho 1500,
mu 0.3, G_c 0.05 J/m^2), gravity -z, ``tlmpm_v5`` cutting law 8 unchanged, with the read-only
``cut_diagnostics_v1`` recorder. The handle box acts as the blunt obstacle, as in every profile task.

Families (``--family``; rows are declared in ``ROWS`` before running):

* ``oblique``: 80 x 24 x 16 mm bar; through-cut along the plane through (x0, 0, H/2) with normal +x turned about +y
  by theta (round-2 OBLIQUE_W frame: d = (-sin, 0, -cos), n = (cos, 0, -sin)); 30 mm relief slot centred where the
  plane meets z = -6 mm (round-2 rule); armed from 8 mm before the section top to ``overtravel`` past the section
  bottom along d at 10 mm/s; then a 0.4 s disarmed hold. Variants: overtravel, gravity off, 12 mm slot, plane phase.
* ``rake``: 80 x 24 x 12 mm slab on a solid board; horizontal +x skim at 25 mm/s with the edge at z = 9.2 mm (target
  9 mm + 0.2 mm clearance), blade raked by rho (d = (cos, 0, -sin), n = (sin, 0, cos)). Counterfactual rows override
  theta_max on the solver instance (labelled COUNTERFACTUAL: not law 8).
* ``phase``: the rho = 10 deg skim with the edge at z = 9.0 + delta mm (the lowest above-plane layer is at 9.5 mm),
  to measure thin-chip residue against the lattice phase.

Outputs per row: ``<out>/<row>/row.json`` and ``row.npz``; ``<out>/summary.json``. Never overwrites.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

SLOT_RULE_Z = -.006
BOARD = .010
TABLE = -.110


def oblique_rows():
    rows = {}
    for angle in range(0, 50, 5):
        rows[f'obl_{angle:02d}'] = dict(family='oblique', angle=float(angle))
    rows['obl_m35'] = dict(family='oblique', angle=-35.)
    for ot in (6, 8, 10):
        rows[f'obl_35_ot{ot:02d}'] = dict(family='oblique', angle=35., overtravel=ot*1e-3)
    rows['obl_35_nograv'] = dict(family='oblique', angle=35., gravity=False)
    rows['obl_35_slot12'] = dict(family='oblique', angle=35., slot=.012)
    for shift in (.25, .5, .75):
        rows[f'obl_35_phase{int(shift*100):03d}'] = dict(family='oblique', angle=35., x_shift=shift*1e-3)
    return rows


def rake_rows():
    rows = {f'rake_{r:02d}': dict(family='rake', rake=float(r)) for r in (10, 20, 30, 35, 40, 44, 45, 46, 50)}
    rows['rake_45_cf60'] = dict(family='rake', rake=45., counterfactual={'theta_max_deg': 60.})
    rows['rake_50_cf60'] = dict(family='rake', rake=50., counterfactual={'theta_max_deg': 60.})
    return rows


def phase_rows():
    return {f'phase_d{int(round(d*100)):03d}': dict(family='phase', rake=10., edge_z=.009+d*1e-3)
            for d in (0., .1, .2, .3, .4, .5)}


def dome_rows():
    """Added after the slab rows (whose 80 mm continuous chip piles up and, at rho >= 30 deg, pushes the unrestrained
    slab off the board): a TRIM-02-K2-like dome (r 12 mm, h 8 mm) on a 100 x 30 x 20 mm bar, skim x = -30..+30 mm."""
    rows = {f'dome_r{r:02d}': dict(family='dome', rake=float(r)) for r in (10, 25, 35, 40, 44, 45, 46, 50)}
    rows['dome_r45_cf60'] = dict(family='dome', rake=45., counterfactual={'theta_max_deg': 60.})
    rows['dome_r50_cf60'] = dict(family='dome', rake=50., counterfactual={'theta_max_deg': 60.})
    for d in (0., .1, .3, .4, .5):
        rows[f'dome_phase_d{int(round(d*100)):03d}'] = dict(family='dome', rake=10., edge_z=.020+d*1e-3)
    return rows


ROWS = {**oblique_rows(), **rake_rows(), **phase_rows(), **dome_rows()}


def material_config(spacing, gravity):
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y
    p = RULER_KNIFE_V2_Y
    return TLMPMConfig(dx=4*spacing, dt=1e-4, density=1500., young=5e4, poisson=.3, yield_stress=1e3,
                       contact_friction=.3, cutting_energy=.05, blade_length=p.blade_depth_m,
                       blade_height=p.thickness_m, blade_half_width=p.half_width_m, cuts=1, floor=False,
                       fixture_walls=False, gravity=(0., 0., -9.81) if gravity else (0., 0., 0.))


def board_boxes(length, width, slot):
    """Board top z = 0 (10 mm thick) over [-(L/2+10), L/2+10] x [-(W/2+6), W/2+6], risers to the table."""
    half_y = width/2+.006
    x0, x1 = -(length/2+.010), length/2+.010
    parts = [(x0, x1)] if slot is None else [(x0, slot[0]), (slot[1], x1)]
    boxes = []
    for a, b in parts:
        if b-a <= 1e-6:
            continue
        c, h = (a+b)/2, (b-a)/2
        boxes.append(((c, 0., -BOARD/2), (h, half_y, BOARD/2)))
        boxes.append(((c, 0., (TABLE-BOARD)/2), (h, max(width/2-.002, .004), (-BOARD-TABLE)/2)))
    return boxes


def edge_rotation(d, e, n):
    return Rotation.from_matrix(np.column_stack([d, e, n]))


def plan(row, spacing):
    """(points, masses, fixture boxes, list of (pose xyzw world, armed) per control step, geometry dict)."""
    from robotic_cutting_sim.blade_probe import block
    fam = row['family']
    if fam == 'oblique':
        L, W, H = .080, .024, .016
        theta = np.deg2rad(row['angle'])
        n = np.array([np.cos(theta), 0., -np.sin(theta)])
        d = np.array([-np.sin(theta), 0., -np.cos(theta)])
        e = np.array([0., 1., 0.])
        origin = np.array([row.get('x_shift', 0.), 0., H/2])
        # Section points on the centre line: z = H (top) and z = 0 (bottom), t along d from origin.
        t_top, t_bottom = -(H-origin[2])/np.cos(theta), origin[2]/np.cos(theta)
        start_t, end_t = t_top-.008, t_bottom+row.get('overtravel', .004)
        x_slot = origin[0]-(origin[2]-SLOT_RULE_Z)*np.tan(theta)    # plane meets z = -6 mm
        slot_w = row.get('slot', .030)
        slot = (x_slot-slot_w/2, x_slot+slot_w/2)
        speed, armed_path = .010, (origin+d*start_t, origin+d*end_t)
        geometry = dict(L=L, W=W, H=H, plane_origin=origin.tolist(), plane_normal=n.tolist(), advance=d.tolist(),
                        slot=list(slot), start=armed_path[0].tolist(), end=armed_path[1].tolist())
    elif fam == 'dome':
        L, W, H, r, h = .100, .030, .020, .012, .008
        rho = np.deg2rad(row['rake'])
        d = np.array([np.cos(rho), 0., -np.sin(rho)])
        n = np.array([np.sin(rho), 0., np.cos(rho)])
        e = np.array([0., 1., 0.])
        z = row.get('edge_z', H+.0002)
        slot, speed = None, .025
        armed_path = (np.array([-.030, 0., z]), np.array([.030, 0., z]))
        geometry = dict(L=L, W=W, H=H, plane_z=H, edge_z=z, rake_deg=row['rake'], dome_radius=r, dome_height=h,
                        advance=d.tolist(), normal=n.tolist(), start=armed_path[0].tolist(), end=armed_path[1].tolist())
    else:
        L, W, H = .080, .024, .012
        rho = np.deg2rad(row['rake'])
        d = np.array([np.cos(rho), 0., -np.sin(rho)])
        n = np.array([np.sin(rho), 0., np.cos(rho)])
        e = np.array([0., 1., 0.])
        z = row.get('edge_z', .0092)
        slot, speed = None, .025
        armed_path = (np.array([-L/2-.008, 0., z]), np.array([L/2+.005, 0., z]))
        geometry = dict(L=L, W=W, H=H, plane_z=.009, edge_z=z, rake_deg=row['rake'], advance=d.tolist(),
                        normal=n.tolist(), start=armed_path[0].tolist(), end=armed_path[1].tolist())
    points, masses = block((L, W, H), spacing=spacing)
    if fam == 'dome':
        g = np.mgrid[-r:r+1e-9:spacing, -r:r+1e-9:spacing, 0:h+1e-9:spacing].reshape(3, -1).T
        g = np.round((g-np.array([-L/2, -W/2, 0.]))/spacing-.5)*spacing+np.array([-L/2, -W/2, 0.])+spacing/2
        g[:, 2] += H
        cap = g[((g[:, 0]**2+g[:, 1]**2)/r**2+((g[:, 2]-H)/h)**2 <= 1.) & (g[:, 2] > H)]
        cap = np.unique(np.round(cap/spacing*2).astype(np.int64), axis=0).astype(np.float64)*spacing/2
        points = np.concatenate([points, cap.astype(np.float32)])
        masses = np.concatenate([masses, np.full(len(cap), masses[0], np.float32)])
    quat = edge_rotation(d, e, n).as_quat()
    dt = .02
    poses = []
    lift = armed_path[0]-d*.004 if fam == 'oblique' else armed_path[0]+np.array([0, 0, .004])
    for _ in range(25):                                   # 0.5 s settle, edge parked clear of the bar
        poses.append((np.r_[lift, quat], False))
    length = np.linalg.norm(armed_path[1]-armed_path[0])
    direction = (armed_path[1]-armed_path[0])/length
    # Approach the armed start (disarmed), then the armed path at constant speed.
    for k in range(1, 11):
        poses.append((np.r_[lift+(armed_path[0]-lift)*k/10, quat], False))
    steps = int(np.ceil(length/(speed*dt)))
    for k in range(1, steps+1):
        poses.append((np.r_[armed_path[0]+direction*min(k*speed*dt, length), quat], True))
    for _ in range(20):                                   # 0.4 s disarmed hold at the end of the stroke
        poses.append((np.r_[armed_path[1], quat], False))
    geometry.update(armed_steps=steps, speed_m_s=speed, control_steps=len(poses))
    return points, masses, board_boxes(L, W, slot), poses, geometry


def run_row(name, row, out, device, spacing):
    import warp as wp
    from robotic_cutting_sim import cut_diagnostics as cd
    from robotic_cutting_sim.coupled_robot import interpolate_measured_poses
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y, handle_box_edge
    start_wall = time.monotonic()
    points, masses, boxes, poses, geometry = plan(row, spacing)
    config = material_config(spacing, row.get('gravity', True))
    polygon, _ = RULER_KNIFE_V2_Y.tlmpm_profile()
    bed = TLMPM(points, masses, spacing, config, device, fixture_boxes=boxes, table_plane=[0., 0., 1., -TABLE],
                blade_profile_xz=polygon, cutting_law=8, obstacle_boxes_edge=[handle_box_edge(RULER_KNIFE_V2_Y)])
    overrides = []
    for key, value in (row.get('counterfactual') or {}).items():
        overrides.append({'key': key, 'frozen_value': bed.law4_parameters[key], 'counterfactual_value': value})
        bed.law4_parameters[key] = value
    recorder = cd.CutDiagnosticsRecorder(bed)
    substeps = 200
    bed.set_trajectory(np.tile(poses[0][0], (substeps+1, 1)).astype(np.float32), np.zeros((substeps+1, 6), np.float32),
                       np.full(substeps+1, -1, np.int32))
    cuda = wp.get_device(device).is_cuda
    graph = None
    if cuda:
        wp.load_module(module=TLMPM.__module__, device=device)
        wp.load_module(module=cd.__name__, device=device)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(substeps):
                recorder.before_step()
                bed.step()
                recorder.after_step()
        graph = capture.graph
    previous = poses[0][0]
    step_stats, step_reaction, step_labels = [], [], []
    for pose, armed in poses[1:]:
        p, t = interpolate_measured_poses(previous, pose, substeps, .02)
        bed.poses.assign(p)
        bed.twists.assign(t)
        bed.cuts.assign(np.full(substeps+1, 0 if armed else -1, np.int32))
        bed.index.zero_()
        bed.reaction.zero_()
        bed.box_reactions.zero_()
        bed.stats.zero_()
        if graph is not None:
            wp.capture_launch(graph)
        else:
            for _ in range(substeps):
                recorder.before_step()
                bed.step()
                recorder.after_step()
        wp.synchronize_device(device)
        bed.check_slots()
        recorder.end_control_step()
        step_stats.append(bed.stats.numpy().copy())
        step_reaction.append(bed.reaction.numpy()[0].copy())
        step_labels.append(int(np.count_nonzero(bed.code.numpy())))
        previous = pose
    recorder.check()
    snap = bed.snapshot()
    codes = snap['codes']
    nodes, weights, gradients = bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy()
    count, labels = mechanical_components_packed(nodes, weights, codes, 1, gradients=gradients)
    fractions = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    digits = cd.digits_of(codes, 0, 'packed2')
    bridges = cd.direct_bridges(nodes, weights, gradients, digits)
    table = recorder.particle_table()
    X = points.astype(np.float64)
    result = dict(row=name, spec={k: v for k, v in row.items()}, geometry=geometry, particles=int(len(points)),
                  counterfactual=overrides or None,
                  label=('COUNTERFACTUAL diagnostic (law parameter overridden on the solver instance; not law 8)'
                         if overrides else 'law 8 unchanged (tlmpm_v5), diagnostic sweep row, not an acceptance run'),
                  components=int(count), mass_fractions=fractions[:4].tolist(), labelled=int(np.count_nonzero(digits)),
                  digit_counts={int(k): int(v) for k, v in zip(*np.unique(digits, return_counts=True))},
                  direct_code0_bridges=int(len(bridges)),
                  max_tool_penetration_mm=float(np.max([s[3] for s in step_stats])*1e3),
                  max_fixture_penetration_mm=float(np.max([s[0] for s in step_stats])*1e3),
                  peak_reaction_N=float(np.max(np.linalg.norm(np.asarray(step_reaction)[:, :3], axis=1))/.02))
    segs = recorder.segment_table()
    col = cd.SEGMENT_COLUMNS.index
    armed = segs[:, :, col('armed_substeps')].sum()
    result['segment_summary'] = {
        'armed_segment_substeps': float(armed),
        'seg_ok_fraction': float(segs[:, :, col('seg_ok_substeps')].sum()/max(armed, 1)),
        'seg_plow_fraction': float(segs[:, :, col('seg_plow_substeps')].sum()/max(armed, 1)),
        'cutting_work_J': float(segs[:, :, col('cutting_work_sum_J')].sum()),
        'work_with_gate_closed_particle_substeps': float(segs[:, :, col('work_with_gate_closed_particle_substeps')].sum()),
        'contact_particle_substeps': float(segs[:, :, col('contact_particle_substeps')].sum()),
        'band_new': float(segs[:, :, [col('band_new_digit1'), col('band_new_digit2')]].sum()),
        'lead_new': float(segs[:, :, [col('lead_new_digit1'), col('lead_new_digit2')]].sum())}
    rows = recorder.substep_table()
    sc = cd.SUBSTEP_COLUMNS.index
    result['stroke_on_substeps'] = int(np.count_nonzero(rows[:, sc('stroke')] > 0))
    result['engaged_substeps'] = int(np.count_nonzero(rows[:, sc('engaged')] > 0))
    if row['family'] == 'oblique':
        n = np.asarray(geometry['plane_normal'])
        s = (X-np.asarray(geometry['plane_origin']))@n
        near = (np.abs(s) <= .012) & (digits == 0)
        idx, classes = cd.classify_unassigned(table, 0, near)
        f_final = (snap['positions'].astype(np.float64)-np.asarray(geometry['end']))@np.asarray(geometry['advance'])
        f_ref = (X-np.asarray(geometry['end']))@np.asarray(geometry['advance'])
        result['unassigned_within_12mm'] = int(near.sum())
        result['unassigned_by_class'] = {c: int(np.count_nonzero(classes == i)) for i, c in enumerate(cd.UNASSIGNED_CLASSES)}
        result['unassigned_s_mm'] = {'min': float(s[near].min()*1e3) if near.any() else None,
                                     'max': float(s[near].max()*1e3) if near.any() else None}
        result['unassigned_layers_z_mm'] = {str(round(float(z)*1e3, 1)): int(c) for z, c in
                                            zip(*np.unique(X[near, 2], return_counts=True))}
        result['unassigned_ahead_of_final_edge_ref_vs_final'] = {
            'reference_f_gt_0': int(np.count_nonzero(f_ref[near] > 0)), 'final_f_gt_0': int(np.count_nonzero(f_final[near] > 0))}
        result['bridges_s_mm'] = {'min': float(s[bridges].min()*1e3) if len(bridges) else None,
                                  'max': float(s[bridges].max()*1e3) if len(bridges) else None}
        result['bridges_by_class'] = {}
        if len(bridges):
            bidx, bcls = cd.classify_unassigned(table, 0, np.isin(np.arange(len(points)), bridges))
            result['bridges_by_class'] = {c: int(np.count_nonzero(bcls == i)) for i, c in enumerate(cd.UNASSIGNED_CLASSES)}
        side = np.where(s >= 0, 2, 1)
        lab = digits != 0
        result['label_vs_plane_sign_disagreements'] = int(np.count_nonzero(lab & (digits != side) & (np.abs(s) > .002)))
        # Overhang sag: mean z displacement of the side-1 (s < 0) and side-2 (s >= 0) halves, bottom layer.
        disp = snap['positions'][:, 2]-points[:, 2]
        bottom = points[:, 2] < spacing
        result['bottom_layer_mean_dz_mm'] = {'s_lt_0': float(disp[bottom & (s < 0)].mean()*1e3),
                                             's_ge_0': float(disp[bottom & (s >= 0)].mean()*1e3)}
    else:
        plane = geometry['plane_z']
        above = X[:, 2] > plane
        base = np.bincount(labels[~above], weights=masses[~above]).argmax()
        removed = float(masses[above & (labels != base)].sum()/masses[above].sum())
        substrate = float(masses[~above & (labels == base)].sum()/masses[~above].sum())
        result['removal'] = removed
        result['substrate_retained'] = substrate
        lowest = above & (X[:, 2] < plane+spacing)
        result['lowest_above_layer'] = {'z_mm': float(X[lowest, 2].min()*1e3) if lowest.any() else None,
                                        'count': int(lowest.sum()), 'attached': int(np.count_nonzero(lowest & (labels == base))),
                                        'digit_counts': {int(k): int(v) for k, v in
                                                         zip(*np.unique(digits[lowest], return_counts=True))}}
        assigned = table['assign_substep'][:, 0] >= 0
        a_s = table['assign_s_m'][:, 0]
        result['lowest_layer_assign_s_mm'] = {
            'median': float(np.median(a_s[lowest & assigned])*1e3) if (lowest & assigned).any() else None,
            'digit1_count': int(np.count_nonzero(lowest & (digits == 1)))}
        disp = snap['positions'][:, 2]-points[:, 2]
        result['assign_dz_lowest_layer_mm'] = (float(np.median(table['assign_x_m'][lowest & assigned, 0, 2]
                                                                 - X[lowest & assigned, 2])*1e3)
                                               if (lowest & assigned).any() else None)
        top_sub = (~above) & (X[:, 2] > plane-spacing)
        result['substrate_top_layer_final_dz_mm'] = {'median': float(np.median(disp[top_sub])*1e3),
                                                     'min': float(disp[top_sub].min()*1e3)}
        result['unassigned_above_plane'] = int(np.count_nonzero(above & (digits == 0)))
        if result['unassigned_above_plane']:
            idx, classes = cd.classify_unassigned(table, 0, above & (digits == 0))
            result['unassigned_above_by_class'] = {c: int(np.count_nonzero(classes == i))
                                                   for i, c in enumerate(cd.UNASSIGNED_CLASSES)}
        result['first_fail_totals_above'] = {r: int(table['first_fail'][above, 0, i].sum()) for i, r in enumerate(cd.REASONS)}
    result['wall_s'] = time.monotonic()-start_wall
    target = out/name
    target.mkdir(parents=False, exist_ok=False)
    np.savez_compressed(target/'row.npz', reference=points, final_positions=snap['positions'], codes=codes,
                        component=labels, masses=masses, step_stats=np.asarray(step_stats),
                        step_reaction=np.asarray(step_reaction), step_labelled=np.asarray(step_labels),
                        **{f'diag_{k}': v for k, v in recorder.arrays().items()})
    (target/'row.json').write_text(json.dumps(result, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--rows', nargs='+', required=True, help='row names or family names (oblique, rake, phase, dome)')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--spacing', type=float, default=.001)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    names = []
    for item in args.rows:
        names += [k for k, v in ROWS.items() if v['family'] == item] if item in ('oblique', 'rake', 'phase', 'dome') else [item]
    unknown = [n for n in names if n not in ROWS]
    if unknown:
        raise SystemExit(f'Unknown rows {unknown}')
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'rows': {}}
    summary.update(tool='scripts/run_wp12_phase4a_sweep.py', device=args.device, spacing_m=args.spacing,
                   material='E 50 kPa, nu 0.3, yield 1 kPa, rho 1500, mu 0.3, G_c 0.05 (provisional, uncalibrated)')
    for name in names:
        if name in summary['rows'] or (args.out/name).exists():
            print(f'{name}: exists, skipped (never overwritten)', flush=True)
            continue
        print(f'{name}: start', flush=True)
        result = run_row(name, ROWS[name], args.out, args.device, args.spacing)
        summary['rows'][name] = {k: v for k, v in result.items() if k not in ('geometry',)}
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"{name}: components {result['components']} labelled {result['labelled']} bridges "
              f"{result['direct_code0_bridges']} wall {result['wall_s']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
