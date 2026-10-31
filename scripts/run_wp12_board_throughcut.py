#!/usr/bin/env python3
"""WP-12 Phase 2 (spec 2.4) kinematic study: vertical through-cut of the standard bar on a solid board.

One row per invocation; each row writes a fresh output directory (never overwritten).
Kinematic ideal tracking on tlmpm_v5 with cutting law 8, 1 mm particles, 4 mm grid,
material dt 0.1 ms, seed 42, the CUT-01-R bar (160 x 30 x 20 mm, material cutting_tasks_v1)
and ruler_knife_v2_y (TLMPM blade cross-section identical to ruler_knife_v1). The scene is an
inline rcs_scene_v1 dict (``supports.workcell`` null) compiled with ``scene_config.compile_task``
and the read-only goal ``configs/wp11/goals/sever_mid.json``; the knife is driven by an explicit
waypoint plan with the goal planner's numbers (clearance 8 mm, cut 10 mm/s, disarmed hold 0.3 s,
disarmed withdrawal 20 mm/s, 30 mm lift) and a declared stop depth ``h`` of the edge's lowest point
relative to the board top (workpiece local z = 0), followed by a settle period to the end of the
8 s episode.

Rows (spec 2.4 options A/B and D):

* ``S_P100``, ``S_P050``, ``S_P025``, ``S_000``: one continuous solid board, h = +1.0/+0.5/+0.25/0 mm.
* ``S_P0375``: solid board, h = +0.375 mm (bracketing row between S_P050 and S_P025; added after those ran).
* ``S_M025_DIAG``: solid board, h = -0.25 mm. DIAGNOSTIC_PENETRATION_NOT_ADOPTABLE: the knife edge
  enters the rigid board, which TLMPM does not detect or resist; it only shows what penetrating
  would buy and is never a candidate.
* ``D8_M400``, ``D8_000``: explicit slotted board (option D), one 8 mm relief slot at x = 0
  (Test B slot width), h = -4 mm (the goal planner / policies.through_cut overtravel below the
  workpiece bottom) and h = 0.
* ``D40_M400``, ``D40_000``: the CUT-01-R 40 mm gap (bar_160x30x20 pedestals), h = -4 mm and 0.

No physics, law parameter or threshold is changed: tlmpm_v5, law 8, the fixture contact kernel
and every evaluator are used unchanged. All extra measurements are read-only host copies after a
control step. Oracle material state is evaluation-only.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_board_throughcut.py \\
      --row S_000 --device cuda:0 --out artifacts/wp12/workcell_throughcut_001
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

SEED = 42
DIAGNOSTIC_LABEL = 'DIAGNOSTIC_PENETRATION_NOT_ADOPTABLE'
BAR = {'length_m': .16, 'width_m': .03, 'height_m': .02}
BOARD_X_RANGE = [-.09, .09]
BOARD_THICKNESS = .01
TOOL = 'ruler_knife_v2_y'
# Goal planner numbers (goal_planner.DEFAULT_OPTIONS): cut clearance, cut speed, hold, kerf speed, safe height.
CLEARANCE, CUT_SPEED, HOLD_S, KERF_SPEED, SAFE_HEIGHT = .008, .01, .3, .02, .03

ROWS = {
    'S_P100': dict(support='solid', h_m=.001, option='A/B'),
    'S_P050': dict(support='solid', h_m=.0005, option='A/B'),
    'S_P0375': dict(support='solid', h_m=.000375, option='A/B (bracketing row, added after S_P025/S_P050)'),
    'S_P025': dict(support='solid', h_m=.00025, option='A/B'),
    'S_000': dict(support='solid', h_m=0., option='A/B'),
    'S_M025_DIAG': dict(support='solid', h_m=-.00025, option='diagnostic', label=DIAGNOSTIC_LABEL),
    'D8_M400': dict(support='slot8', h_m=-.004, option='D'),
    'D8_000': dict(support='slot8', h_m=0., option='D'),
    'D40_M400': dict(support='slot40', h_m=-.004, option='D (CUT-01-R reference gap)'),
    'D40_000': dict(support='slot40', h_m=0., option='D (CUT-01-R reference gap)'),
}


# ----------------------------------------------------------------------------- pure helpers (CPU tested)

def supports_block(kind):
    """rcs_scene_v1 supports for a row (workcell null: only the TLMPM-local boxes exist)."""
    if kind == 'solid':
        return {'kind': 'solid_board', 'x_range_m': list(BOARD_X_RANGE), 'half_width_y_m': .025,
                'thickness_m': BOARD_THICKNESS, 'workcell': None}
    if kind == 'slot8':
        return {'kind': 'slotted_board', 'x_range_m': list(BOARD_X_RANGE), 'half_width_y_m': .025,
                'thickness_m': BOARD_THICKNESS, 'slots': [{'x_m': 0., 'width_m': .008}], 'workcell': None}
    if kind == 'slot40':  # exactly configs/wp11/scenes/bar_160x30x20.json (CUT-01-R pedestals)
        return {'kind': 'slotted_board', 'x_range_m': list(BOARD_X_RANGE), 'half_width_y_m': .021,
                'thickness_m': BOARD_THICKNESS, 'slots': [{'x_m': 0., 'width_m': .04}],
                'names': ['left_pedestal', 'right_pedestal'], 'workcell': None}
    raise ValueError(kind)


def scene_for(row):
    spec = ROWS[row]
    return {'schema': 'rcs_scene_v1', 'scene_id': f'wp12_throughcut_{row.lower()}',
            'description': f'WP-12 spec 2.4 through-cut study row {row}: CUT-01-R bar on {spec["support"]} support, '
                           f'edge stop h = {spec["h_m"]*1e3:+.2f} mm relative to the board top',
            'workpiece': {'base_box_m': dict(BAR), 'protrusions': [],
                          'world_pose': {'translation_m': [0., 0., 0.], 'yaw_deg': 0.},
                          'material_profile': 'cutting_tasks_v1'},
            'supports': supports_block(spec['support']),
            'tool': {'profile_id': TOOL, 'grasp_preset': 'default'},
            'solver': {'module': 'tlmpm_v5', 'cutting_law': 8, 'max_cuts': 1, 'particle_spacing_m': .001,
                       'grid_dx_m': .004}}


def lowest_blade_point(profile_xz, rotation):
    """(lowest world z offset from the edge origin, polygon vertex index) of the blade cross-section.

    The TLMPM blade is the convex polygon (edge x, edge z) extruded along edge y. For any rotation the
    lowest world point over the extrusion is attained at a polygon vertex (z is linear along y; for
    an edge line with zero world-z component, every y gives the same value)."""
    R = np.asarray(rotation.as_matrix() if hasattr(rotation, 'as_matrix') else rotation, float)
    pts = np.c_[profile_xz[:, 0], np.zeros(len(profile_xz)), profile_xz[:, 1]]
    z = pts @ R[2]
    return float(z.min()), int(np.argmin(z)), float(abs(R[2, 1]))


def box_sdf(points, center, half):
    """NumPy replica of tlmpm._box_sdf: (unit outward normal (N,3), signed distance (N,))."""
    p = np.asarray(points, float)-np.asarray(center, float)
    q = np.abs(p)-np.asarray(half, float)
    outside = np.maximum(q, 0.)
    distance = np.linalg.norm(outside, axis=1)
    sdf = distance+np.minimum(q.max(1), 0.)
    normal = np.zeros_like(p)
    out = distance > 1e-10
    normal[out] = np.sign(p[out])*outside[out]/distance[out, None]
    inside = ~out
    axis = np.argmax(q, axis=1)  # ties: x before y before z, as the kernel's >= chain
    rows = np.flatnonzero(inside)
    normal[rows, axis[rows]] = np.sign(p[rows, axis[rows]])
    return normal, sdf


def contact_impulse(gap, normal, velocity, mass, dt, stiffness, friction):
    """NumPy replica of tlmpm._contact_impulse (penalty normal impulse + Coulomb-capped friction)."""
    vn = np.einsum('ij,ij->i', velocity, normal)
    predicted = gap+dt*np.minimum(vn, 0.)
    jn = stiffness*mass/dt*np.maximum(-predicted, 0.)
    tangent = velocity-vn[:, None]*normal
    speed = np.linalg.norm(tangent, axis=1)
    jt = np.zeros_like(tangent)
    moving = speed > 1e-10
    jt[moving] = -(np.minimum(mass[moving]*speed[moving], friction*jn[moving])/speed[moving])[:, None]*tangent[moving]
    return jn[:, None]*normal+jt


def fixture_force_sample(x, v, m, boxes, radius, dt, stiffness, friction):
    """Per-box force the fixture kernel would apply at this state (one substep impulse / dt)."""
    out = []
    for center, half in boxes:
        normal, sdf = box_sdf(x, center, half)
        gap = sdf-radius
        active = gap+dt*np.minimum(np.einsum('ij,ij->i', v, normal), 0.) < 0
        impulse = np.zeros(3)
        if active.any():
            impulse = contact_impulse(gap[active], normal[active], v[active], m[active], dt, stiffness, friction).sum(0)
        out.append({'force_N': (impulse/dt).tolist(), 'active_particles': int(active.sum()),
                    'min_sdf_m': float(sdf.min()), 'inside_particles': int((sdf < 0).sum())})
    return out


def crossing_links(lattice, digits, plane_i):
    """Per z layer, classify the reference x-links (i = plane_i, j, k) - (plane_i + 1, j, k) crossing the plane.

    ``digits`` are the cut-0 digits (0 unassigned, 1 negative side, 2 positive side). Returns
    {k: {conflicting, unconflicted, both_unassigned, one_unassigned, same_side, links}}; only
    ``conflicting`` links (1-2 or 2-1) are severed in the reference six-neighbour sense."""
    lattice = np.asarray(lattice)
    digits = np.asarray(digits)
    key = {tuple(v): i for i, v in enumerate(lattice.tolist())}
    rows = {}
    left = np.flatnonzero(lattice[:, 0] == plane_i)
    for a in left:
        i, j, k = lattice[a]
        b = key.get((i+1, j, k))
        if b is None:
            continue
        da, db = int(digits[a]), int(digits[b])
        r = rows.setdefault(int(k), dict(links=0, conflicting=0, unconflicted=0, both_unassigned=0,
                                         one_unassigned=0, same_side=0))
        r['links'] += 1
        if {da, db} == {1, 2}:
            r['conflicting'] += 1
        else:
            r['unconflicted'] += 1
            if da == 0 and db == 0:
                r['both_unassigned'] += 1
            elif da == 0 or db == 0:
                r['one_unassigned'] += 1
            else:
                r['same_side'] += 1
    return dict(sorted(rows.items()))


def stencil_valid(weights, gradients):
    """Same validity rule as cut_topology.mechanical_components: weight > 0 or |gradient| > 0."""
    return (np.asarray(weights) > 0) | (np.linalg.norm(np.asarray(gradients), axis=-1) > 0)


def direct_bridges(nodes, valid, digits):
    """Indices of digit-0 particles whose valid stencil nodes host both a digit-1 and a digit-2 particle.

    Same rule as scripts/analyze_cut_bridges.py (``direct_code_zero_bridges``)."""
    nodes = np.asarray(nodes, np.int64)
    digits = np.asarray(digits)
    has = {}
    for d in (1, 2):
        mask = np.zeros(int(nodes.max())+1, bool)
        sel = digits == d
        mask[nodes[sel][valid[sel]]] = True
        has[d] = mask
    zero = np.flatnonzero(digits == 0)
    touches = lambda d: (has[d][nodes[zero]] & valid[zero]).any(1)  # noqa: E731
    return zero[touches(1) & touches(2)]


def two_part_check(fractions, minimum_each=.4, minimum_total=.99):
    """CUT-01 v1-style mass criterion (cutting_tasks_v1: >= 40% each, >= 99% together); not the frozen oracle."""
    f = np.sort(np.asarray(fractions, float))[::-1]
    return {'two_substantial_parts_each_ge_40pct': bool(len(f) >= 2 and f[1] >= minimum_each),
            'two_main_parts_together_ge_99pct': bool(len(f) >= 2 and f[:2].sum() >= minimum_total)}


# ----------------------------------------------------------------------------- run

def git_state():
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
    return {'git_sha': sha, 'git_dirty': bool(status.strip()), 'git_status_porcelain': status.splitlines(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


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


def plan(task, h_m):
    from robotic_cutting_sim.goal_planner import through_cut_rotation
    from robotic_cutting_sim.policies import Waypoint
    rotation = through_cut_rotation([1., 0., 0.])  # advance -z, edge +y, blade normal +x
    T = task.T_world_workpiece
    world = lambda p: T[:3, :3] @ np.asarray(p, float)+T[:3, 3]  # noqa: E731
    pose = lambda p: tuple(np.r_[world(p), rotation.as_quat()])  # noqa: E731
    top = BAR['height_m']
    start, stop, up = [0., 0., top+CLEARANCE], [0., 0., h_m], [0., 0., top+CLEARANCE+SAFE_HEIGHT]
    return rotation, [Waypoint(pose(up), label='APPROACH'), Waypoint(pose(start), label='PRE_CUT'),
                      Waypoint(pose(stop), cut=True, speed_m_s=CUT_SPEED, label='CUT'),
                      Waypoint(pose(stop), hold_s=HOLD_S, label='HOLD'),
                      Waypoint(pose(start), speed_m_s=KERF_SPEED, label='WITHDRAW'),
                      Waypoint(pose(up), label='LIFT')]


def snapshot(bed, time_s, phase, det=None):
    if det is None:
        det = np.linalg.det(bed.F.numpy().astype(np.float64))
    return {'time_s': float(time_s), 'phase': phase, 'positions': bed.x.numpy().astype(np.float32).copy(),
            'velocities': bed.v.numpy().astype(np.float32).copy(), 'codes': bed.code.numpy().astype(np.int32).copy(),
            'det_F': np.asarray(det, np.float32)}


def analyze_state(snap, ctx):
    """Topology, bottom-layer links, bridges, separation and board penetration of one snapshot."""
    from robotic_cutting_sim.cut_topology import decode_digits, mechanical_components_packed
    from scipy.spatial import cKDTree
    codes = snap['codes']
    digits = decode_digits(codes, 1)[:, 0]
    count, labels = mechanical_components_packed(ctx['nodes'], ctx['weights'], codes, 1, gradients=ctx['gradients'])
    masses = np.bincount(labels, weights=ctx['mass'], minlength=count)
    order = np.argsort(masses)[::-1]
    fractions = masses[order]/ctx['mass'].sum()
    x = snap['positions'].astype(float)
    ref = ctx['reference']
    out = {'time_s': snap['time_s'], 'phase': snap['phase'], 'mechanical_components': int(count),
           'component_masses_kg': masses[order][:8].tolist(), 'component_mass_fractions': fractions[:8].tolist(),
           **two_part_check(fractions), 'labelled_particles': int(np.count_nonzero(codes)),
           'digit_counts': {int(k): int(v) for k, v in zip(*np.unique(digits, return_counts=True))}}
    if count >= 2:
        a, b = order[0], order[1]
        ca, cb = [np.average(x[labels == c], axis=0, weights=ctx['mass'][labels == c]) for c in (a, b)]
        near = np.abs(x[:, 0]) < .006
        pa, pb = x[near & (labels == a)], x[near & (labels == b)]
        gap = float(cKDTree(pa).query(pb)[0].min()) if len(pa) and len(pb) else None
        out['two_largest_pieces'] = {'centroid_local_m': [ca.tolist(), cb.tolist()],
                                     'centroid_x_gap_m': float(abs(ca[0]-cb[0])),
                                     'min_particle_center_distance_near_plane_m': gap,
                                     'note': 'Current particle centres; separation is the reference stencil topology, '
                                             'not this geometric distance'}
    # Bottom layers across x = 0: reference x-links between lattice i = 79 and 80 (x = -0.5 / +0.5 mm).
    links = crossing_links(ctx['lattice'], digits, ctx['plane_i'])
    out['crossing_links_by_layer'] = {k: links[k] for k in list(links)[:4]}
    out['crossing_links_all_layers'] = {'links': int(sum(r['links'] for r in links.values())),
                                        'conflicting': int(sum(r['conflicting'] for r in links.values())),
                                        'layers_fully_conflicting': [k for k, r in links.items()
                                                                     if r['conflicting'] == r['links']],
                                        'layers_with_unconflicted': [k for k, r in links.items() if r['unconflicted']]}
    # Unassigned particles in the 8 mm stencil column of the x = 0 node plane (|x_ref| < dx), by layer.
    column = np.abs(ref[:, 0]) < ctx['dx']
    k = ctx['lattice'][:, 2]
    out['column_unassigned_by_layer'] = {int(z): int(np.count_nonzero(column & (k == z) & (digits == 0)))
                                         for z in range(4)}
    out['column_particles_per_layer'] = int(np.count_nonzero(column & (k == 0)))
    bridges = direct_bridges(ctx['nodes'], ctx['valid'], digits)
    out['direct_code0_bridges'] = {'count': int(len(bridges)),
                                   'by_layer': {int(z): int(c) for z, c in zip(*np.unique(k[bridges], return_counts=True))},
                                   'reference_x_range_m': ([float(ref[bridges, 0].min()), float(ref[bridges, 0].max())]
                                                           if len(bridges) else None),
                                   'reference_z_max_m': float(ref[bridges, 2].max()) if len(bridges) else None,
                                   'rule': 'scripts/analyze_cut_bridges.py direct_code_zero_bridges (digit 0 touching '
                                           'stencil nodes that host digit 1 and digit 2)'}
    # Board penetration of material particle centres.
    inside, min_sdf = 0, np.inf
    for center, half in ctx['boxes']:
        _, sdf = box_sdf(x, center, half)
        inside += int((sdf < 0).sum())
        min_sdf = min(min_sdf, float(sdf.min()))
    out['board'] = {'min_particle_z_m': float(x[:, 2].min()),
                    'particles_below_board_top': int((x[:, 2] < 0).sum()),
                    'particle_centres_inside_board_boxes': inside, 'min_board_sdf_m': min_sdf,
                    'min_particle_z_near_plane_m': float(x[np.abs(x[:, 0]) < .004, 2].min())}
    det = snap['det_F']
    i = int(np.argmin(det))
    out['det_F'] = {'min': float(det[i]), 'reference_local_m': ref[i].tolist(), 'current_local_m': x[i].tolist(),
                    'below_0_5': int((det < .5).sum()), 'nonpositive': int((det <= 0).sum())}
    return out


def run(row, args, out):
    from robotic_cutting_sim.control import ActionLimits
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.scene_config import compile_task, make_configured_kinematic_env
    from robotic_cutting_sim.tool_profile import PROFILES
    spec = ROWS[row]
    scene = scene_for(row)
    goal = json.loads((ROOT/'configs/wp11/goals/sever_mid.json').read_text())
    task = compile_task(scene, goal, seed=SEED)
    env = make_configured_kinematic_env(task, seed=SEED, device=args.device, material_dt=1e-4,
                                        action_limits=ActionLimits(), ik_failure='reject', delta_frame='world',
                                        history_stride=args.history_stride)
    reset_start = time.monotonic()
    observation, _ = env.reset(seed=SEED)
    reset_wall = time.monotonic()-reset_start
    material = env.backend.material
    bed = material.bed
    c = bed.config
    rotation, waypoints = plan(env.task, spec['h_m'])
    profile = PROFILES[TOOL].tlmpm_profile()[0]
    lowest, vertex, edge_z_slope = lowest_blade_point(profile, rotation)
    boxes = [(np.asarray(b.center_m, float), np.asarray(b.half_extents_m, float)) for b in env.task.fixture_boxes_local]
    board_top = max(cz+hz for (_, _, cz), (_, _, hz) in boxes)
    policy = WaypointPolicy(env.task, waypoints, env.action_limits, position_tolerance_m=1e-6)
    mass = np.asarray(material.masses, float)
    M = mass.sum()
    gravity = np.asarray(c.gravity, float)
    control_dt = env.task.control_dt_s
    nodes, weights, gradients = bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy()
    ctx = {'nodes': nodes, 'weights': weights, 'gradients': gradients, 'valid': stencil_valid(weights, gradients),
           'mass': mass, 'reference': np.asarray(material.points, float), 'lattice': np.asarray(material.lattice),
           'plane_i': int(np.asarray(material.lattice)[np.asarray(material.points)[:, 0] < 0, 0].max()),
           'dx': float(c.dx), 'boxes': boxes}
    snaps = {'RESET': snapshot(bed, 0., 'RESET')}
    steps, momentum = [], np.zeros(3)
    limit = int(round(env.task.duration_s/control_dt)) if args.steps is None else args.steps
    loop_start = time.monotonic()
    with (out/'steps.jsonl').open('w') as stream:
        for step in range(limit):
            action = policy(observation)
            phase = policy.phase if not policy.done else 'SETTLE'
            observation, _, terminated, truncated, info = env.step(action)
            diag = env.backend.last_diagnostics['material']
            x, v = bed.x.numpy().astype(float), bed.v.numpy().astype(float)
            new_momentum = mass @ v
            reaction = bed.reaction.numpy()[0].astype(float)[:3]  # sum of -impulse of blade + handle on material
            fixture_impulse = (new_momentum-momentum)-M*gravity*control_dt+reaction
            momentum = new_momentum
            sample = fixture_force_sample(x, v, mass, boxes, bed.radius, c.dt, c.contact_stiffness, c.contact_friction)
            stats = bed.stats.numpy().astype(float)
            det = np.linalg.det(bed.F.numpy().astype(np.float64))
            edge = np.asarray(env.backend.edge, float)
            stroke = env.backend.stroke_history[-1]
            record = {'step': step+1, 'time_s': env.time_s, 'phase': phase, 'cut_enabled': bool(action['cut_enabled']),
                      'edge_local_z_m': float(env.task.points_to_local(edge[None, :3])[0, 2]),
                      'blade_force_on_tool_N': (np.asarray(diag['current_material_reaction_impulse_world'][:3])
                                                / control_dt).tolist(),
                      'fixture_force_momentum_balance_N': (fixture_impulse/control_dt).tolist(),
                      'fixture_force_sample_N': [s['force_N'] for s in sample],
                      'fixture_active_particles': [s['active_particles'] for s in sample],
                      'stats': stats.tolist(), 'stroke': stroke['stroke'], 'engaged': stroke['engaged'],
                      'labelled_count': stroke['labelled_count'],
                      'min_particle_z_m': float(x[:, 2].min()), 'particles_below_board_top': int((x[:, 2] < 0).sum()),
                      'particle_centres_inside_board': int(sum(s['inside_particles'] for s in sample)),
                      'min_det_F': float(det.min()), 'ik_status': info.get('ik_status')}
            stream.write(json.dumps(jsonable(record))+'\n')
            steps.append(record)
            snaps[phase] = snapshot(bed, env.time_s, phase, det)
            if terminated or truncated:
                break
    loop_wall = time.monotonic()-loop_start
    snaps['FINAL'] = snapshot(bed, env.time_s, 'FINAL')
    analysis = {name: analyze_state(s, ctx) for name, s in snaps.items() if name in ('CUT', 'HOLD', 'WITHDRAW', 'FINAL')}
    # Component-count history (env oracle history, stride history_stride control steps).
    cache, history = {}, []
    for frame in env._history:
        key = frame['codes'].tobytes()
        if key not in cache:
            cache[key] = int(mechanical_components_packed(nodes, weights, frame['codes'], 1, gradients=gradients)[0])
        history.append([float(frame['time_s']), cache[key]])
    separated = next((t for t, n in history if n >= 2), None)
    drops = [t for (t0, n0), (t, n) in zip(history, history[1:]) if separated is not None and t0 >= separated and n < n0]
    persistence = {'first_separated_history_time_s': separated,
                   'separated_duration_to_end_s': None if separated is None else float(history[-1][0]-separated),
                   'count_drops_after_separation_s': drops,
                   'persists_ge_0_5_s': bool(separated is not None and history[-1][0]-separated >= .5 and not drops),
                   'history_stride_control_steps': args.history_stride}
    evaluation_start = time.monotonic()
    try:
        evaluation = env.evaluate()
        evaluation = {'verdict': evaluation.get('verdict'), 'outcome': evaluation.get('outcome'),
                      'integrity_status': (evaluation.get('integrity') or {}).get('status')}
    except Exception as error:  # recorded, never hides the measurements
        evaluation = {'verdict': 'ERROR', 'error': repr(error)}
    evaluation_wall = time.monotonic()-evaluation_start
    # Contact summaries by phase.
    def phase_summary(names):
        rows = [s for s in steps if s['phase'] in names]
        if not rows:
            return None
        fz = np.asarray([s['fixture_force_momentum_balance_N'] for s in rows])
        sample = np.asarray([np.sum(s['fixture_force_sample_N'], axis=0) for s in rows])
        blade = np.asarray([s['blade_force_on_tool_N'] for s in rows])
        return {'steps': len(rows), 'time_s': [rows[0]['time_s']-control_dt, rows[-1]['time_s']],
                'fixture_force_z_momentum_balance_N': {'mean': float(fz[:, 2].mean()), 'max': float(fz[:, 2].max()),
                                                       'min': float(fz[:, 2].min())},
                'fixture_force_xy_momentum_balance_max_N': float(np.linalg.norm(fz[:, :2], axis=1).max()),
                'fixture_force_z_sample_N': {'mean': float(sample[:, 2].mean()), 'max': float(sample[:, 2].max())},
                'fixture_impulse_z_momentum_balance_Ns': float(fz[:, 2].sum()*control_dt),
                'blade_force_on_tool_z_N': {'mean': float(blade[:, 2].mean()), 'max_abs': float(np.abs(blade[:, 2]).max())},
                'max_fixture_penetration_stat_m': float(max(s['stats'][0] for s in rows)),
                'max_tool_penetration_stat_m': float(max(s['stats'][3] for s in rows)),
                'min_particle_z_m': float(min(s['min_particle_z_m'] for s in rows)),
                'max_particles_below_board_top': int(max(s['particles_below_board_top'] for s in rows)),
                'max_particle_centres_inside_board': int(max(s['particle_centres_inside_board'] for s in rows)),
                'min_det_F': float(min(s['min_det_F'] for s in rows))}
    contact = {name: phase_summary(names) for name, names in
               (('pre_contact_rest', ('APPROACH', 'PRE_CUT')), ('cut', ('CUT',)), ('hold', ('HOLD',)),
                ('withdraw', ('WITHDRAW', 'LIFT')), ('settle', ('SETTLE',)))}
    contact['weight_N'] = float(M*9.81)
    cut_rows = [s for s in steps if s['phase'] == 'CUT']
    stop_z = cut_rows[-1]['edge_local_z_m'] if cut_rows else None
    blade_board = {'edge_lowest_point_local_z_m': stop_z, 'board_top_local_z_m': board_top,
                   'declared_h_m': spec['h_m'],
                   'lowest_blade_point_offset_from_edge_m': lowest, 'lowest_vertex_index': vertex,
                   'edge_line_world_z_slope': edge_z_slope,
                   'lowest_point_is_edge_vertex': bool(vertex == 0 and abs(lowest) < 1e-9 and edge_z_slope < 1e-9),
                   'blade_contact_radius_m': float(bed.radius)}
    blade_board['blade_overlaps_board_box_m'] = _blade_board_overlap(stop_z, profile, rotation, boxes, PROFILES[TOOL])
    # Snapshots: start/end frames plus phase ends (no full history on disk).
    arrays = {'reference_local': ctx['reference'].astype(np.float32), 'lattice': ctx['lattice'].astype(np.int16),
              'mass_kg': mass, 'stencil_nodes': nodes.astype(np.int32), 'stencil_weights': weights.astype(np.float32),
              'code_encoding': 'packed2', 'max_cuts': 1, 'snapshot_names': np.asarray(list(snaps))}
    for name, s in snaps.items():
        for key in ('positions', 'codes', 'det_F'):
            arrays[f'{name}_{key}'] = s[key]
        arrays[f'{name}_time_s'] = s['time_s']
    np.savez_compressed(out/'snapshots.npz', **arrays)
    slots = bed.check_slots()
    context = {'task_id': env.task.task_id, 'config_sha256': getattr(env.task, 'config_sha256', None),
               'scene': scene, 'goal_id': goal['goal_id'], 'goal_source': 'configs/wp11/goals/sever_mid.json',
               'fixture_boxes_local': [[list(map(float, cc)), list(map(float, hh))] for cc, hh in boxes],
               'plan': [dict(label=w.label, pose=list(map(float, w.pose)), cut=w.cut, speed_m_s=w.speed_m_s, hold_s=w.hold_s)
                        for w in waypoints], 'position_tolerance_m': 1e-6,
               'law8_parameters': bed.law8_parameters, 'law4_parameters': bed.law4_parameters,
               'material_receipt': material.parameter_receipt, 'tlmpm_v5_slots': slots,
               'particles': int(len(mass)), 'grid_dims': np.asarray(bed.grid_dims).tolist(),
               'grid_origin_m': np.asarray(bed.grid_origin).tolist(), 'node_count': int(bed.node_count),
               'reset_wall_s': reset_wall, 'loop_wall_s': loop_wall, 'evaluation_wall_s': evaluation_wall,
               'runtime_source': env.runtime_source}
    env.close()
    return {'analysis': analysis, 'persistence': persistence, 'component_count_history': history,
            'evaluation_configured_sever_goal': evaluation, 'contact': contact, 'blade_board': blade_board,
            'context': context, 'steps_run': len(steps)}


def _blade_board_overlap(stop_z, profile, rotation, boxes, tool):
    """Geometric overlap of the blade (polygon extruded along the edge) with each board box at the stop pose.

    Returns the largest depth (m) by which blade points lie inside a box (0 = no overlap). TLMPM does not
    use this: the kinematic blade has no contact with fixtures (it is reported, never enforced)."""
    if stop_z is None:
        return None
    R = rotation.as_matrix()
    ys = np.linspace(-tool.half_width_m, tool.half_width_m, 241)
    # Densify the polygon boundary.
    poly = np.asarray(profile, float)
    dense = np.concatenate([poly[i]+np.linspace(0, 1, 40)[:, None]*(np.roll(poly, -1, 0)[i]-poly[i]) for i in range(len(poly))])
    pts = np.array([[px, y, pz] for px, pz in dense for y in ys]) @ R.T+np.array([0., 0., stop_z])
    depth = 0.
    for center, half in boxes:
        _, sdf = box_sdf(pts, center, half)
        depth = max(depth, float(max(-sdf.min(), 0.)))
    return depth


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', required=True, choices=sorted(ROWS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--history-stride', type=int, default=5)
    p.add_argument('--steps', type=int, help='Smoke only: cap control steps (row is then not formal)')
    a = p.parse_args()
    if 'workcell_throughcut' not in a.out.name:
        p.error('--out name must contain workcell_throughcut')
    out = a.out
    out.mkdir(parents=True, exist_ok=False)  # never overwrite (recording.new_output semantics)
    from robotic_cutting_sim.recording import manifest, write_json
    spec = ROWS[a.row]
    config = {**vars(a), 'out': str(a.out), 'seed': SEED, 'row_spec': spec}
    write_json(out/'manifest.json', manifest(config, 'wp12_board_throughcut_tlmpm_v5_law8_kinematic'))
    base = {'row': a.row, **spec, 'label': spec.get('label', 'CANDIDATE_STUDY_ROW'), 'command': sys.argv, **git_state(),
            'seed': SEED, 'device': a.device, 'solver': 'tlmpm_v5', 'cutting_law': 8, 'particle_spacing_m': .001,
            'grid_dx_m': .004, 'material_dt_s': 1e-4, 'tool_profile': TOOL, 'formal': a.steps is None,
            'scope': 'Kinematic ideal tracking; provisional uncalibrated material; no robot, PhysX or cameras'}
    start = time.monotonic()
    try:
        result = run(a.row, a, out)
        result['wall_s'] = time.monotonic()-start
        write_json(out/'result.json', jsonable({**base, **result}))
        final = result['analysis'].get('FINAL', {})
        print(a.row, base['label'], 'components', final.get('mechanical_components'),
              'bridges', (final.get('direct_code0_bridges') or {}).get('count'), 'wall_s', round(result['wall_s'], 1),
              flush=True)
    except Exception as error:
        write_json(out/'result.json', jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                 'wall_s': time.monotonic()-start}))
        raise


if __name__ == '__main__':
    main()
