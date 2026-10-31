#!/usr/bin/env python3
"""WP-12 Phase 4 experiment E1 (configs/wp12/phase4_e1_protocol_v1.json): oblique through-cuts on the workcell board.

One declared row per process. Law 8 on tlmpm_v5 is UNCHANGED. The knife edge is driven by kinematic ideal tracking
through ``LiveCuttingMaterial`` (built as ``KinematicCuttingBackend.reset`` builds it), with the standard CUT-01-R bar
(160 x 30 x 20 mm, cutting_tasks_v1, 1 mm / 4 mm / 0.1 ms, seed 42) on ``blade_master_flat_board_v1`` (solid board,
Option A) or its explicit slotted variant (Option D). Two read-only recorders bracket every substep inside the
CUDA graph:

* ``cut_diagnostics_v1``: the solver's own cutting decisions;
* ``exit_candidates_v1``: an OFFLINE shadow evaluation of the rejected breakthrough mask and the exit-ligament rule
  L9-EL. It writes no label into the physics run.

Usage::

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4_e1.py --row A_p20 --device cuda:1 \\
      --out artifacts/wp12/phase4_e1_001/A_p20
  # DEV smoke (label DEV, never evidence): reduced bar and/or capped control steps
  ... --row A_p20 --dev --dev-bar 40 12 12 --out artifacts/wp12/phase4_e1_dev_001/A_p20_bar40

Outputs in ``--out`` (created, never overwritten): ``result.json``, ``steps.jsonl``, ``snapshots.npz``, ``diag.npz``.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

PROTOCOL = ROOT/'configs/wp12/phase4_e1_protocol_v1.json'
SEED = 42
BAR_H = .020
CLEARANCE, CUT_SPEED, KERF_SPEED, HOLD_S, SETTLE_S, SETTLE_END_S = .008, .010, .020, .3, .5, .5
CONTROL_DT = .02
SLOT_MIN_CLEARANCE = .001
SOURCES = ('src/robotic_cutting_sim/tlmpm_v5.py', 'src/robotic_cutting_sim/cut_diagnostics.py',
           'src/robotic_cutting_sim/exit_candidates.py', 'src/robotic_cutting_sim/live_material.py',
           'src/robotic_cutting_sim/fixture_contacts.py', 'src/robotic_cutting_sim/scene_config.py',
           'src/robotic_cutting_sim/coupled_robot.py', 'configs/wp12/scenes/bar_flat_board.json',
           'configs/wp12/scenes/bar_flat_board_slotted.json', 'configs/wp12_workcells_v1.json',
           'configs/wp12/phase4_e1_protocol_v1.json', 'scripts/run_wp12_phase4_e1.py')


# ----------------------------------------------------------------------------- pure helpers (CPU tested)

def protocol_rows(path=PROTOCOL):
    return {row['id']: row for row in json.loads(Path(path).read_text())['rows']}


def frame(tau_deg):
    """(d, e, n) of the protocol: d = (-sin, 0, -cos), e = +y, n = d x e = (cos, 0, -sin)."""
    t = np.deg2rad(float(tau_deg))
    d = np.array([-np.sin(t), 0., -np.cos(t)])
    e = np.array([0., 1., 0.])
    return d, e, np.cross(d, e)


def edge_rotation(tau_deg):
    d, e, n = frame(tau_deg)
    return Rotation.from_matrix(np.column_stack([d, e, n]))


def edge_path(row, height=BAR_H):
    """Plane origin o, armed start and end edge positions (local) and their parameters t along d."""
    d, _, n = frame(row['tau_deg'])
    cos = float(-d[2])
    o = np.array([row['plane_x0_mm']*1e-3, 0., 0.])
    t_start = -height/cos-CLEARANCE
    t_end = -row['edge_end_z_mm']*1e-3/cos
    return {'o': o, 'd': d, 'n': n, 't_start': t_start, 't_end': t_end, 'start': o+t_start*d, 'end': o+t_end*d,
            'armed_length_m': t_end-t_start}


def control_schedule(row, height=BAR_H, dt=CONTROL_DT):
    """[(local edge position, armed, phase)] per control step end (the material starts at the first entry)."""
    path = edge_path(row, height)
    start, end = path['start'], path['end']
    out = [(start.copy(), False, 'RESET')]
    out += [(start.copy(), False, 'SETTLE') for _ in range(int(round(SETTLE_S/dt)))]

    def move(a, b, speed, armed, phase):
        length = float(np.linalg.norm(b-a))
        steps = int(np.ceil(length/(speed*dt)-1e-9))
        for k in range(1, steps+1):
            out.append((a+(b-a)*min(k*speed*dt/length, 1.), armed, phase))
    move(start, end, CUT_SPEED, True, 'CUT')
    out += [(end.copy(), False, 'HOLD') for _ in range(int(round(HOLD_S/dt)))]
    move(end, start, KERF_SPEED, False, 'WITHDRAW')
    out += [(start.copy(), False, 'SETTLE_END') for _ in range(int(round(SETTLE_END_S/dt)))]
    return out


def lowest_blade_vertex(profile_xz, rotation):
    R = rotation.as_matrix()
    pts = np.c_[profile_xz[:, 0], np.zeros(len(profile_xz)), profile_xz[:, 1]]
    z = pts @ R[2]
    return float(z.min()), int(np.argmin(z))


def swept_blade_points(profile_xz, rotation, a, b, samples=400):
    """Dense boundary of the region swept by the blade polygon translating from edge position a to b (x-z plane)."""
    R = rotation.as_matrix()
    pts = []
    for o in (np.asarray(a, float), np.asarray(b, float)):
        pts.append(o+profile_xz[:, :1]*R[:, 0]+profile_xz[:, 1:2]*R[:, 2])
    xz = np.concatenate(pts)[:, [0, 2]]
    hull = xz[ConvexHull(xz).vertices]
    dense = np.concatenate([hull[i]+np.linspace(0, 1, samples)[:, None]*(np.roll(hull, -1, 0)[i]-hull[i])
                            for i in range(len(hull))])
    return np.c_[dense[:, 0], np.zeros(len(dense)), dense[:, 1]]


def blade_board_clearance(profile_xz, rotation, a, b, boxes):
    """Minimum signed distance (m) from the swept blade region to the board boxes (negative = overlap)."""
    from run_wp12_board_throughcut import box_sdf
    pts = swept_blade_points(profile_xz, rotation, a, b)
    return float(min(box_sdf(pts, c, h)[1].min() for c, h in boxes))


def exit_face_table(reference, digits, s, rule_lead, layers=4, reach=.012, spacing=.001):
    """Bottom layers within |s| <= reach: per layer and side, digit counts and lead-rule labels."""
    z = reference[:, 2]
    out = {}
    for k in range(layers):
        layer = (z >= k*spacing) & (z < (k+1)*spacing) & (np.abs(s) <= reach)
        for name, side in (('s_lt_0', s < 0), ('s_ge_0', s >= 0)):
            sel = layer & side
            plane_side = np.where(s >= 0, 2, 1)
            out[f'layer{k}_{name}'] = {
                'particles': int(sel.sum()), 'digit0': int(np.count_nonzero(sel & (digits == 0))),
                'digit1': int(np.count_nonzero(sel & (digits == 1))), 'digit2': int(np.count_nonzero(sel & (digits == 2))),
                'lead_rule_labels': int(np.count_nonzero(sel & rule_lead)),
                'disagree_with_plane_side_abs_s_gt_2mm': int(np.count_nonzero(sel & (digits != 0) & (digits != plane_side)
                                                                              & (np.abs(s) > .002)))}
    return out


def git_state():
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout
    return {'git_sha': run('rev-parse', 'HEAD').strip(), 'git_status_porcelain': run('status', '--porcelain').splitlines(),
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES if (ROOT/p).exists()}}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ----------------------------------------------------------------------------- run

def compile_row_task(row, dev_bar=None):
    from robotic_cutting_sim import scene_config as sc
    from robotic_cutting_sim.task_scenarios import configure_task
    scene = deepcopy(sc.load_json(ROOT/f"configs/wp12/scenes/{row['scene']}.json"))
    scene['solver']['max_cuts'] = 1
    scene['scene_id'] = scene['scene_id']+'_e1_k1'
    if dev_bar is not None:
        scene['workpiece']['base_box_m'] = {'length_m': dev_bar[0]*1e-3, 'width_m': dev_bar[1]*1e-3,
                                            'height_m': dev_bar[2]*1e-3}
        scene['scene_id'] += '_dev_bar'
    goal = json.loads((ROOT/'configs/wp11/goals/sever_mid.json').read_text())
    task = sc.compile_task(scene, goal, seed=SEED)
    return configure_task(replace(task, seed=SEED)), scene


def attach_recorders(state):
    """Bracket every tlmpm_v5 step of this process with the two read-only recorders (inside the CUDA graph)."""
    import warp as wp
    from robotic_cutting_sim import cut_diagnostics as cd
    from robotic_cutting_sim import exit_candidates as ec
    from robotic_cutting_sim import tlmpm_v5
    original_init, original_step = tlmpm_v5.TLMPM.__init__, tlmpm_v5.TLMPM.step

    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        self._e1_diag = cd.CutDiagnosticsRecorder(self)
        self._e1_cand = ec.ExitCandidateEvaluator(self)
        state['bed'] = self
        if wp.get_device(self.device).is_cuda:
            wp.load_module(module=cd.__name__, device=self.device)
            wp.load_module(module=ec.__name__, device=self.device)

    def step(self, advance=True):
        diag = getattr(self, '_e1_diag', None)
        if diag is not None:
            diag.before_step()
        original_step(self, advance)
        if diag is not None:
            diag.after_step(advanced=advance)
            self._e1_cand.after_step(advanced=advance)

    tlmpm_v5.TLMPM.__init__, tlmpm_v5.TLMPM.step = init, step


def run(row, args, out):
    import warp as wp
    from robotic_cutting_sim import cut_diagnostics as cd
    from robotic_cutting_sim import exit_candidates as ec
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from run_wp12_board_throughcut import box_sdf
    wp.config.quiet = True
    task, scene = compile_row_task(row, args.dev_bar)
    height = float(scene['workpiece']['base_box_m']['height_m'])
    state = {}
    attach_recorders(state)
    coupling = CouplingConfig(control_dt=task.control_dt_s, material_dt=1e-4, dx=task.grid_dx_m, device=args.device)
    material = LiveCuttingMaterial(coupling, task=task, cutting_law=8, max_cuts=1, slice_efficiency=1.,
                                   solver='tlmpm_v5', slots_per_node=None)
    rotation = edge_rotation(row['tau_deg'])
    quat = rotation.as_quat()
    T = np.asarray(task.T_world_workpiece, float)
    world = lambda p: np.r_[T[:3, :3]@p+T[:3, 3], (Rotation.from_matrix(T[:3, :3])*rotation).as_quat()]  # noqa: E731
    schedule = control_schedule(row, height)
    if args.dev_max_steps is not None:
        schedule = schedule[:args.dev_max_steps+1]
    path = edge_path(row, height)
    reset_start = time.monotonic()
    receipt = material.reset(world(schedule[0][0]), seed=SEED)
    reset_wall = time.monotonic()-reset_start
    bed = state['bed']
    diag, cand = bed._e1_diag, bed._e1_cand
    boxes = [(np.asarray(b.center_m, float), np.asarray(b.half_extents_m, float)) for b in task.fixture_boxes_local]
    board_top = max(c[2]+h[2] for c, h in boxes)
    profile = np.asarray(bed.blade_profile_xz, float)
    lowest, vertex = lowest_blade_vertex(profile, rotation)
    clearance = blade_board_clearance(profile, rotation, path['start'], path['end'], boxes)
    mass = np.asarray(material.masses, float)
    reference = np.asarray(material.points, float)
    snaps, steps = {}, []
    loop_start = time.monotonic()
    with (out/'steps.jsonl').open('w') as stream:
        for index, (position, armed, phase) in enumerate(schedule[1:], start=1):
            result = material.advance(world(position), round(index*CONTROL_DT, 9), cutting=armed, cut_index=0)
            diag.end_control_step()
            cand.end_control_step()
            x = bed.x.numpy().astype(np.float64)
            v = bed.v.numpy().astype(np.float64)
            inside = sum(int((box_sdf(x, c, h)[1] < 0).sum()) for c, h in boxes)
            stats = bed.stats.numpy().astype(float)
            record = {'step': index, 'time_s': index*CONTROL_DT, 'phase': phase, 'armed': armed,
                      'edge_local_m': position.tolist(), 'cut_index': int(result['cut_index']),
                      'stroke': bed.cutting_state(count_labels=True), 'stats': stats.tolist(),
                      'reaction_N': (bed.reaction.numpy()[0].astype(float)[:3]/CONTROL_DT).tolist(),
                      'min_particle_z_m': float(x[:, 2].min()), 'particles_below_board_top': int((x[:, 2] < board_top).sum()),
                      'particle_centres_inside_board': inside, 'max_speed_m_s': float(np.linalg.norm(v, axis=1).max()),
                      'min_det_F': float(np.linalg.det(bed.F.numpy().astype(np.float64)).min())}
            stream.write(json.dumps(jsonable(record))+'\n')
            steps.append(record)
            snaps[phase] = {'positions': x.astype(np.float32), 'codes': bed.code.numpy().astype(np.int32).copy(),
                            'time_s': index*CONTROL_DT}
    loop_wall = time.monotonic()-loop_start
    diag_check, cand_check = diag.check(), cand.check()
    bed.check_slots()
    final = bed.snapshot()
    codes = final['codes'].astype(np.int64)
    x = final['positions'].astype(np.float64)
    nodes, weights, gradients = bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy()
    digits = cd.digits_of(codes, 0, 'packed2')
    s_ref = (reference-path['o'])@path['n']
    f_final = (x-path['end'])@path['d']
    f_ref = (reference-path['end'])@path['d']

    def topology(c):
        count, labels = mechanical_components_packed(nodes, weights, np.asarray(c, np.int32), 1, gradients=gradients)
        fractions = np.sort(np.bincount(labels, weights=mass, minlength=count))[::-1]/mass.sum()
        d = cd.digits_of(c, 0, 'packed2')
        bridges = cd.direct_bridges(nodes, weights, gradients, d)
        pieces = int(np.count_nonzero(fractions >= .01))
        outcome = ('SEPARATED' if count >= 2 and pieces >= 2 and len(bridges) == 0 else
                   'CONNECTED' if count == 1 else 'OTHER')
        return {'components': int(count), 'pieces_ge_1pct': pieces, 'mass_fractions': fractions[:4].tolist(),
                'direct_bridges': int(len(bridges)), 'outcome': outcome,
                'bridge_s_ref_mm': [float(s_ref[bridges].min()*1e3), float(s_ref[bridges].max()*1e3)] if len(bridges) else None,
                'bridge_f_final_mm': [float(f_final[bridges].min()*1e3), float(f_final[bridges].max()*1e3)] if len(bridges) else None,
                'bridge_layers': {str(int(k)): int(n) for k, n in zip(*np.unique(np.floor(reference[bridges, 2]*1e3).astype(int),
                                                                                 return_counts=True))},
                'labelled': int(np.count_nonzero(d))}, bridges

    law8, bridges = topology(codes)
    analysis = {'law8_final': law8}
    if 'CUT' in snaps:
        analysis['law8_end_of_cut'] = topology(snaps['CUT']['codes'])[0]
    table = diag.particle_table()
    near = (np.abs(s_ref) <= .012) & (digits == 0)
    idx, classes = cd.classify_unassigned(table, 0, near)
    analysis['unassigned_within_12mm'] = {
        'count': int(near.sum()),
        'by_class': {c: int(np.count_nonzero(classes == i)) for i, c in enumerate(cd.UNASSIGNED_CLASSES)},
        'first_fail_totals': {r: int(table['first_fail'][near, 0, i].sum()) for i, r in enumerate(cd.REASONS)},
        'only_fail_particles': {r: int(np.count_nonzero(table['only_fail'][near, 0, i] > 0)) for i, r in enumerate(cd.REASONS)},
        'lead_rule_blocked_by_plow_gate': int(np.count_nonzero(table['lead_plow_blocked'][near, 0] > 0)),
        'in_core_abs_s_le_4mm': int(np.count_nonzero(near & (np.abs(s_ref) <= .004))),
        'ahead_of_final_edge_reference_f_gt_0': int(np.count_nonzero(near & (f_ref > 0))),
        'ahead_of_final_edge_final_f_gt_0': int(np.count_nonzero(near & (f_final > 0))),
        's_ref_mm': [float(s_ref[near].min()*1e3), float(s_ref[near].max()*1e3)] if near.any() else None,
        'layers_z_ref_mm': {str(round(float(z)*1e3, 1)): int(c) for z, c in zip(*np.unique(reference[near, 2], return_counts=True))},
        'min_f_stroke_mm': [float(table['min_f_stroke_m'][near, 0].min()*1e3), float(table['min_f_stroke_m'][near, 0].max()*1e3)]
        if near.any() else None}
    if len(bridges):
        _, bclasses = cd.classify_unassigned(table, 0, np.isin(np.arange(len(reference)), bridges))
        analysis['law8_final']['bridges_by_class'] = {c: int(np.count_nonzero(bclasses == i))
                                                      for i, c in enumerate(cd.UNASSIGNED_CLASSES)}
    lead = (table['assign_substep'][:, 0] >= 0) & ((table['assign_flags'][:, 0] & cd.ASSIGN_FLAGS['lead_rule']) != 0)
    analysis['exit_face_assignment'] = exit_face_table(reference, digits, s_ref, lead)
    # Candidates (offline shadow).
    shadow = cand.shadow_codes()
    ctable = cand.particle_table()
    candidates = {}
    for key in ('proposed', 'exit_ligament'):
        sh = shadow[key].astype(np.int64)
        hit = sh != 0
        predicted = ec.predicted_codes(codes, sh)
        topo, _ = topology(predicted)
        prefix = 'proposed' if key == 'proposed' else 'exit_ligament'
        s_at = ctable[f'{prefix}_s_m'][:, 0]
        f_at = ctable[f'{prefix}_f_m'][:, 0]
        later = hit & (digits != 0) & (digits != (sh & 3))
        candidates[key] = {'shadow_labels': int(hit.sum()), 'side1': int(np.count_nonzero(hit & ((sh & 3) == 1))),
                           'side2': int(np.count_nonzero(hit & ((sh & 3) == 2))),
                           'later_law8_band_gave_other_side': int(later.sum()),
                           'later_law8_same_side': int(np.count_nonzero(hit & (digits == (sh & 3)))),
                           'still_law8_unassigned': int(np.count_nonzero(hit & (digits == 0))),
                           's_at_assign_mm': [float(s_at[hit].min()*1e3), float(s_at[hit].max()*1e3)] if hit.any() else None,
                           'f_at_assign_mm': [float(f_at[hit].min()*1e3), float(f_at[hit].max()*1e3)] if hit.any() else None,
                           'first_substep': int(ctable[f'{prefix}_substep'][hit, 0].min()) if hit.any() else None,
                           'predicted': topo}
    candidates['proposed']['flags'] = {'B_only': int(np.count_nonzero((ctable['proposed_flags'][:, 0] & 3) == 1)),
                                       'plow': int(np.count_nonzero(ctable['proposed_flags'][:, 0] & 2))}
    candidates['proposed']['near_unassigned_blocked_by_B'] = int(np.count_nonzero(near & (ctable['proposed_blocked_substeps'][:, 0] > 0)))
    candidates['exit_ligament']['near_unassigned_blocked_ligament_present'] = int(np.count_nonzero(
        near & (ctable['exit_ligament_blocked_substeps'][:, 0] > 0)))
    candidates['exit_ligament']['near_unassigned_segment_never_seen'] = int(np.count_nonzero(
        near & (ctable['exit_ligament_unseen_substeps'][:, 0] > 0)))
    rows = cand.substep_table()
    col = ec.SUBSTEP_COLUMNS.index
    on = (rows[:, col('stroke')] > 0) & (rows[:, col('stroke_gate_segments')] > 0)
    states = cand.state_table()
    scol = ec.STATE_COLUMNS.index
    moving = [i for i in range(len(states)) if states[i, :, scol('stroke_gate')].any()]
    end_state = states[moving[-1]] if moving else None
    candidates['segments'] = {
        'stroke_on_substeps': int(on.sum()),
        'last_moving_control_step': int(moving[-1]+1) if moving else None,
        'material_segments_at_end': int(np.count_nonzero(end_state[:, scol('n_any')] > 0)) if end_state is not None else None,
        'proposed_B_on_material_at_end': int(np.count_nonzero((end_state[:, scol('n_any')] > 0) & (end_state[:, scol('proposed_B')] > 0)))
        if end_state is not None else None,
        'exit_X_at_end': int(end_state[:, scol('exit_X')].sum()) if end_state is not None else None,
        'seen_at_end': int(end_state[:, scol('seen')].sum()) if end_state is not None else None,
        'ligament_at_end': int(end_state[:, scol('ligament')].sum()) if end_state is not None else None,
        'first_exit_X_substep': int(rows[rows[:, col('exit_X_segments')] > 0, 0].min()) if (rows[:, col('exit_X_segments')] > 0).any() else None,
        'first_proposed_B_on_material_substep': int(rows[on & (rows[:, col('proposed_B_on_material')] > 0), 0].min())
        if (on & (rows[:, col('proposed_B_on_material')] > 0)).any() else None,
        'relative_fallback_particle_evals': float(rows[:, col('relative_fallback_particle_evals')].sum())}
    if end_state is not None and candidates['segments']['material_segments_at_end']:
        m = candidates['segments']['material_segments_at_end']
        candidates['segments']['proposed_B_zero_fraction_of_material_at_end'] = 1.-candidates['segments']['proposed_B_on_material_at_end']/m
    analysis['candidates_offline'] = candidates
    # Clearance, penetration, dynamics.
    cut_rows = [s for s in steps if s['phase'] == 'CUT']
    end_z = cut_rows[-1]['edge_local_m'][2] if cut_rows else None
    analysis['edge_to_board'] = {'edge_end_local_z_mm': None if end_z is None else end_z*1e3,
                                 'board_top_local_z_mm': board_top*1e3,
                                 'edge_minus_board_top_mm': None if end_z is None else (end_z-board_top)*1e3,
                                 'lowest_blade_vertex_offset_mm': lowest*1e3, 'lowest_blade_vertex_index': vertex,
                                 'lowest_point_is_edge_vertex': bool(vertex == 0 and abs(lowest) < 1e-12),
                                 'swept_blade_min_signed_distance_to_board_mm': clearance*1e3,
                                 'blade_board_overlap_mm': max(0., -clearance*1e3)}
    final_inside = sum(int((box_sdf(x, c, h)[1] < 0).sum()) for c, h in boxes)
    analysis['board_penetration'] = {
        'particle_centres_inside_board_max_over_steps': int(max(s['particle_centres_inside_board'] for s in steps)),
        'particle_centres_inside_board_final': final_inside,
        'min_particle_signed_distance_to_board_mm': float(min(box_sdf(x, c, h)[1].min() for c, h in boxes)*1e3),
        'particles_below_board_top_max_over_steps': int(max(s['particles_below_board_top'] for s in steps)),
        'min_particle_z_mm': float(min(s['min_particle_z_m'] for s in steps)*1e3),
        'max_fixture_penetration_stat_mm': float(max(s['stats'][0] for s in steps)*1e3)}
    disp = x[:, 2]-reference[:, 2]
    bottom = reference[:, 2] < .001
    analysis['dynamics_context'] = {
        'peak_reaction_N': float(max(np.linalg.norm(s['reaction_N']) for s in steps)),
        'max_tool_penetration_mm': float(max(s['stats'][3] for s in steps)*1e3),
        'min_det_F': float(min(s['min_det_F'] for s in steps)),
        'bottom_layer_mean_dz_mm': {'s_lt_0': float(disp[bottom & (s_ref < 0)].mean()*1e3),
                                    's_ge_0': float(disp[bottom & (s_ref >= 0)].mean()*1e3)},
        'max_speed_final_m_s': float(steps[-1]['max_speed_m_s'])}
    integrity = {'recorder_overflow': diag_check['summary_overflow'], 'evaluator_overflow': cand_check['summary_overflow'],
                 'slot_overflow': int(bed.slot_diagnostics()['overflow_entries']),
                 'blade_board_overlap': bool(clearance < -1e-9),
                 'slotted_clearance_ok': (row['scene'] != 'bar_flat_board_slotted') or clearance >= SLOT_MIN_CLEARANCE-1e-6,
                 'lowest_point_is_edge_vertex': analysis['edge_to_board']['lowest_point_is_edge_vertex'],
                 'complete': args.dev_max_steps is None and len(steps) == len(schedule)-1}
    ok = (not integrity['recorder_overflow'] and not integrity['evaluator_overflow'] and not integrity['slot_overflow']
          and not integrity['blade_board_overlap'] and integrity['slotted_clearance_ok']
          and integrity['lowest_point_is_edge_vertex'] and integrity['complete'])
    np.savez_compressed(out/'snapshots.npz', reference_local=reference.astype(np.float32), mass_kg=mass,
                        lattice=np.asarray(material.lattice, np.int16), stencil_nodes=nodes.astype(np.int32),
                        stencil_weights=weights.astype(np.float32), stencil_gradients=gradients.astype(np.float32),
                        final_positions=x.astype(np.float32), final_codes=codes.astype(np.int32),
                        shadow_proposed=shadow['proposed'], shadow_exit_ligament=shadow['exit_ligament'],
                        **{f'{name}_{key}': value for name, snap in snaps.items() for key, value in snap.items()})
    np.savez_compressed(out/'diag.npz', **{f'cd_{k}': v for k, v in diag.arrays().items()},
                        **{f'ec_{k}': v for k, v in cand.arrays().items()})
    context = {'task_id': task.task_id, 'scene': scene, 'fixture_boxes_local': [[c.tolist(), h.tolist()] for c, h in boxes],
               'path': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in path.items()},
               'quaternion_local_xyzw': quat.tolist(), 'control_steps': len(schedule)-1,
               'phase_steps': {p: sum(1 for s in schedule[1:] if s[2] == p) for p in ('SETTLE', 'CUT', 'HOLD', 'WITHDRAW', 'SETTLE_END')},
               'particles': int(len(mass)), 'material_receipt': receipt.get('parameter_receipt'),
               'law4_parameters': bed.law4_parameters, 'law8_parameters': bed.law8_parameters,
               'cut_diagnostics': diag.metadata(), 'exit_candidates': cand.metadata(),
               'slots': bed.slot_diagnostics(), 'reset_wall_s': reset_wall, 'loop_wall_s': loop_wall}
    return {'row_status': 'PASS' if ok else 'FAIL_INTEGRITY', 'integrity': integrity, 'analysis': analysis,
            'context': context, 'steps_run': len(steps)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=PROTOCOL)
    p.add_argument('--dev', action='store_true', help='DEV smoke run (label DEV, never evidence)')
    p.add_argument('--dev-bar', type=float, nargs=3, metavar=('L_MM', 'W_MM', 'H_MM'), help='DEV: reduced bar')
    p.add_argument('--dev-max-steps', type=int, help='DEV: cap the control steps')
    a = p.parse_args()
    rows = protocol_rows(a.protocol)
    if a.row not in rows:
        p.error(f'row {a.row!r} is not declared in {a.protocol}')
    if (a.dev_bar is not None or a.dev_max_steps is not None) and not a.dev:
        p.error('--dev-bar / --dev-max-steps need --dev')
    parent = a.out.resolve().parent.name
    if a.dev and not parent.startswith('phase4_e1_dev_'):
        p.error('DEV output must be under artifacts/wp12/phase4_e1_dev_NNN/')
    if not a.dev and (not parent.startswith('phase4_e1_') or parent.startswith('phase4_e1_dev_')):
        p.error('formal output must be under artifacts/wp12/phase4_e1_NNN/')
    a.out.mkdir(parents=True, exist_ok=False)
    from robotic_cutting_sim.recording import manifest, write_json
    row = rows[a.row]
    label = 'DEV' if a.dev else 'E1_FORMAL'
    write_json(a.out/'manifest.json', manifest({**vars(a), 'out': str(a.out), 'protocol': str(a.protocol), 'seed': SEED,
                                                'row_spec': row}, 'wp12_phase4_e1_tlmpm_v5_law8_kinematic'))
    base = {'row': a.row, 'row_spec': row, 'label': label, 'command': sys.argv, **git_state(),
            'protocol_sha256': hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seed': SEED, 'device': a.device, 'solver': 'tlmpm_v5', 'cutting_law': 8, 'particle_spacing_m': .001,
            'grid_dx_m': .004, 'material_dt_s': 1e-4, 'tool_profile': 'ruler_knife_v2_y',
            'dev_bar_mm': a.dev_bar, 'dev_max_steps': a.dev_max_steps,
            'scope': 'Kinematic ideal tracking; provisional uncalibrated material; law 8 unchanged; candidate rules '
                     'evaluated offline (shadow), never written into the physics run'}
    start = time.monotonic()
    try:
        result = run(row, a, a.out)
        result['wall_s'] = time.monotonic()-start
        write_json(a.out/'result.json', jsonable({**base, **result}))
        law8 = result['analysis']['law8_final']
        cands = result['analysis']['candidates_offline']
        print(a.row, label, result['row_status'], 'law8', law8['outcome'], law8['components'], 'bridges',
              law8['direct_bridges'], '| proposed', cands['proposed']['predicted']['components'],
              '| L9-EL', cands['exit_ligament']['predicted']['components'], 'wall_s', round(result['wall_s'], 1), flush=True)
    except Exception as error:
        write_json(a.out/'result.json', jsonable({**base, 'row_status': 'ERROR', 'error': repr(error),
                                                   'wall_s': time.monotonic()-start}))
        raise


if __name__ == '__main__':
    main()
