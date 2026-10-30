#!/usr/bin/env python3
"""WP-12 Phase 4 (spec 4.7): systematic resolution study, one row per invocation (never overwrites).

Kinematic ideal tracking of ``ruler_knife_v2_y`` (double bevel 25 deg, 1.5 mm thick, 120 mm edge) through a
lattice workpiece lying on the ``blade_master_flat_board_v1`` board (one solid 20 mm board on the table, board top =
workpiece-local z = 0, table plane 20 mm lower; no risers, no slot). Solver ``tlmpm_v5`` with cutting law 8,
unchanged; the provisional, uncalibrated WP-11 material (E 50 kPa, nu 0.3, yield 1 kPa, rho 1500 kg/m^3,
mu 0.3, G_c 0.05 J/m^2, gravity -z); the read-only ``cut_events_v1`` recorder (Hermite log for
``cut_aware_surface_v1``). 20 ms control steps interpolated into 0.02/dt substeps exactly like
``LiveCuttingMaterial.advance`` (``coupled_robot.interpolate_measured_poses``).

Cases (geometry and blade path fixed in metres; workpiece dimensions are multiples of 6 mm so every declared
spacing 2 / 1.5 / 1 / 0.75 / 0.5 mm tiles them exactly and every vertical cut plane lies between two particle
columns):

* ``CUT``    72 x 24 x 18 mm bar, vertical through-cut at x = 0, armed 10 mm/s from 8 mm above the top to the edge
             at z = h (default 0, the board top: Option A), disarmed 0.3 s hold, disarmed 20 mm/s withdrawal, settle.
* ``OBL20``  same bar, oblique through-cut, blade plane through (0, 0, H/2) turned 20 deg about +y
             (d = (-sin, 0, -cos), n = (cos, 0, -sin)), armed along d from 8 mm before the section top until the edge
             reaches the board top (no overtravel exists on a solid board), hold, withdrawal along -d, settle.
* ``TRIM``   72 x 30 x 18 mm base with a centred ellipsoidal dome (r 12 mm, h 8 mm) on its top (plane z = 18 mm);
             rho = 10 deg rake skim along +x at 25 mm/s from x = -30 to +30 mm with the edge at z = 18 + 0.2 mm
             (variant ``gap_s``: 18 + 0.2 s, i.e. the gap to the lowest dome layer fixed at 0.3 s), 0.4 s hold.
* ``MULTI2`` the CUT bar, two sequential vertical through-cuts at x = -12 mm (digit 0) and x = +12 mm (digit 1),
             each as CUT; reduced form of Test B (2 slices -> 3 pieces).

Rows are declared in ``configs/wp12/resolution_protocol_v1.json``; ``--row`` must name one of them unless
``--dev`` (a labelled development / feasibility run, never a formal row). Physics, law parameters and thresholds
are not changed; every extra quantity is a read-only host copy after a control step. Oracle material state is
evaluation-only.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_resolution.py --row CUT_s100_g4s_dt0100 \\
      --device cuda:1 --out artifacts/wp12/resolution_002/CUT_s100_g4s_dt0100
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PROTOCOL = ROOT/'configs/wp12/resolution_protocol_v1.json'

SEED = 42
CONTROL_DT = .02
TOOL = 'ruler_knife_v2_y'
# blade_master_flat_board_v1: board 360 x 190 x 20 mm, workpiece centred; table top 20 mm below the board top.
BOARD_BOX = ((0., 0., -.010), (.180, .095, .010))
TABLE_PLANE = [0., 0., 1., .020]
CLEARANCE, CUT_SPEED, HOLD_S, WITHDRAW_SPEED, MOVE_SPEED, SETTLE_S = .008, .010, .3, .020, .020, .5
TRIM_SPEED, TRIM_HOLD_S = .025, .4
BAR = (.072, .024, .018)
TRIM_BASE = (.072, .030, .018)
DOME_R, DOME_H = .012, .008
MATERIAL = dict(density=1500., young=5e4, poisson=.3, yield_stress=1e3, contact_friction=.3, cutting_energy=.05)


# ----------------------------------------------------------------------------- geometry (CPU tested)

def lattice_box(size, s):
    """Particles at (i + 1/2) s - L/2 in x, y and (k + 1/2) s in z; exact for L/s integer (raises otherwise)."""
    counts = np.round(np.asarray(size)/s).astype(int)
    if np.any(np.abs(counts*s-np.asarray(size)) > 1e-9):
        raise ValueError(f'{size} is not a multiple of the spacing {s}')
    i, j, k = np.meshgrid(*[np.arange(c) for c in counts], indexing='ij')
    lat = np.c_[i.ravel(), j.ravel(), k.ravel()]
    pts = (lat+.5)*s-np.array([size[0]/2, size[1]/2, 0.])
    return pts, lat


def dome_cap(base, s, r=DOME_R, h=DOME_H):
    """Lattice cells above the base top (z > H) whose centres lie inside the ellipsoid cap; same lattice as the base."""
    L, W, H = base
    ni, nj, nk = int(round(L/s)), int(round(W/s)), int(round(h/s))+1
    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk)+int(round(H/s)), indexing='ij')
    lat = np.c_[i.ravel(), j.ravel(), k.ravel()]
    pts = (lat+.5)*s-np.array([L/2, W/2, 0.])
    keep = ((pts[:, 0]**2+pts[:, 1]**2)/r**2+((pts[:, 2]-H)/h)**2 <= 1.) & (pts[:, 2] > H)
    return pts[keep], lat[keep]


def workpiece(case, s):
    """(points float64, lattice int, masses, geometry dict) of a case at spacing s."""
    if case == 'TRIM':
        base, lat = lattice_box(TRIM_BASE, s)
        cap, clat = dome_cap(TRIM_BASE, s)
        pts, lat = np.concatenate([base, cap]), np.concatenate([lat, clat])
        geo = dict(base_m=list(TRIM_BASE), plane_z_m=TRIM_BASE[2], dome_radius_m=DOME_R, dome_height_m=DOME_H,
                   dome_particles=int(len(cap)), dome_volume_lattice_m3=float(len(cap)*s**3),
                   dome_volume_analytic_m3=float(2/3*np.pi*DOME_R**2*DOME_H))
    else:
        pts, lat = lattice_box(BAR, s)
        geo = dict(bar_m=list(BAR))
    masses = np.full(len(pts), MATERIAL['density']*s**3)
    return pts, lat, masses, geo


def edge_rotation(d, e, n):
    """Edge frame columns: local x = advance d, local y = edge tangent e, local z = blade normal n."""
    return Rotation.from_matrix(np.column_stack([d, e, n]))


def segment(poses, a, b, speed, quat, armed, cut):
    """Append constant-speed control-step poses from a to b (excluding a, ending exactly at b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b-a))
    steps = max(1, int(np.ceil(length/(speed*CONTROL_DT)-1e-9)))
    for k in range(1, steps+1):
        poses.append(dict(pos=a+(b-a)*min(k*speed*CONTROL_DT/length, 1.) if length > 0 else b, quat=quat,
                          armed=armed, cut=cut))
    return b


def hold(poses, p, seconds, quat, label_cut=-1):
    for _ in range(int(round(seconds/CONTROL_DT))):
        poses.append(dict(pos=np.asarray(p, float), quat=quat, armed=False, cut=label_cut))


def plan(case, s, variant):
    """Control-step plan: list of dicts (pos edge-origin local m, quat xyzw, armed, cut digit, phase) + geometry."""
    poses, geo = [], {}
    H = BAR[2]
    if case in ('CUT', 'MULTI2'):
        d, e, n = np.array([0., 0., -1.]), np.array([0., 1., 0.]), np.array([1., 0., 0.])
        q = edge_rotation(d, e, n).as_quat()
        h = variant_h(variant, s)
        xs = [0.] if case == 'CUT' else [-.012, .012]
        park = np.array([xs[0], 0., H+CLEARANCE+.004])
        hold(poses, park, SETTLE_S, q)
        for ph in poses:
            ph['phase'] = 'SETTLE0'
        here = park
        planes = []
        for cut, x in enumerate(xs):
            start, stop = np.array([x, 0., H+CLEARANCE]), np.array([x, 0., h])
            n0 = len(poses)
            here = segment(poses, here, start, MOVE_SPEED, q, False, -1)
            mark(poses, n0, 'MOVE' if cut else 'APPROACH')
            n0 = len(poses)
            here = segment(poses, here, stop, CUT_SPEED, q, True, cut)
            mark(poses, n0, f'CUT{cut}')
            n0 = len(poses)
            hold(poses, stop, HOLD_S, q)
            mark(poses, n0, f'HOLD{cut}')
            n0 = len(poses)
            here = segment(poses, here, start, WITHDRAW_SPEED, q, False, -1)
            mark(poses, n0, f'WITHDRAW{cut}')
            planes.append(dict(cut=cut, origin=[x, 0., 0.], normal=[1., 0., 0.]))
        geo.update(planes=planes, edge_stop_z_m=h, advance=d.tolist(), blade_normal=n.tolist())
    elif case == 'OBL20':
        th = np.deg2rad(20.)
        n = np.array([np.cos(th), 0., -np.sin(th)])
        d = np.array([-np.sin(th), 0., -np.cos(th)])
        e = np.array([0., 1., 0.])
        q = edge_rotation(d, e, n).as_quat()
        h = variant_h(variant, s)
        origin = np.array([0., 0., H/2])
        t_top = -(H-origin[2])/np.cos(th)
        t_end = (origin[2]-h)/np.cos(th)            # the edge reaches z = h (board top for h = 0)
        start, stop = origin+d*(t_top-CLEARANCE), origin+d*t_end
        park = start-d*.004
        hold(poses, park, SETTLE_S, q)
        mark(poses, 0, 'SETTLE0')
        n0 = len(poses)
        here = segment(poses, park, start, MOVE_SPEED, q, False, -1)
        mark(poses, n0, 'APPROACH')
        n0 = len(poses)
        here = segment(poses, here, stop, CUT_SPEED, q, True, 0)
        mark(poses, n0, 'CUT0')
        n0 = len(poses)
        hold(poses, stop, HOLD_S, q)
        mark(poses, n0, 'HOLD0')
        n0 = len(poses)
        segment(poses, here, start, WITHDRAW_SPEED, q, False, -1)
        mark(poses, n0, 'WITHDRAW0')
        geo.update(planes=[dict(cut=0, origin=origin.tolist(), normal=n.tolist())], edge_stop_z_m=h, angle_deg=20.,
                   advance=d.tolist(), blade_normal=n.tolist(), armed_start=start.tolist(), armed_end=stop.tolist())
    elif case == 'TRIM':
        rho = np.deg2rad(10.)
        d = np.array([np.cos(rho), 0., -np.sin(rho)])
        n = np.array([np.sin(rho), 0., np.cos(rho)])
        e = np.array([0., 1., 0.])
        q = edge_rotation(d, e, n).as_quat()
        plane = TRIM_BASE[2]
        z = plane+(.2*s if variant == 'gap_s' else .0002)
        start, stop = np.array([-.030, 0., z]), np.array([.030, 0., z])
        park = start+np.array([0., 0., .004])
        hold(poses, park, SETTLE_S, q)
        mark(poses, 0, 'SETTLE0')
        n0 = len(poses)
        here = segment(poses, park, start, MOVE_SPEED, q, False, -1)
        mark(poses, n0, 'APPROACH')
        n0 = len(poses)
        here = segment(poses, here, stop, TRIM_SPEED, q, True, 0)
        mark(poses, n0, 'CUT0')
        n0 = len(poses)
        hold(poses, stop, TRIM_HOLD_S, q)
        mark(poses, n0, 'HOLD0')
        geo.update(plane_z_m=plane, edge_z_m=z, rake_deg=10., advance=d.tolist(), blade_normal=n.tolist(),
                   edge_gap_to_lowest_dome_layer_m=plane+s/2-z, armed_start=start.tolist(), armed_end=stop.tolist())
    else:
        raise ValueError(case)
    if case != 'TRIM':
        n0 = len(poses)
        hold(poses, poses[-1]['pos'], SETTLE_S, q)
        mark(poses, n0, 'SETTLE')
    geo['control_steps'] = len(poses)
    geo['armed_control_steps'] = int(sum(p['armed'] for p in poses))
    return poses, geo


def mark(poses, first, name):
    for p in poses[first:]:
        p.setdefault('phase', name)


def variant_h(variant, s):
    """Edge stop height above the board top (CUT / OBL20 / MULTI2)."""
    if variant in (None, 'formal', 'gap_s', 'rep1'):
        return 0.
    if variant == 'h0375m':
        return .000375
    if variant == 'h0375s':
        return .375*s
    raise ValueError(variant)


def contact_window(case, poses, geo):
    """Predeclared contact-window mask (per control step, from the PLANNED edge pose in metres, never from
    measured contact): CUT/OBL20/MULTI2: armed steps with the edge z in [h, H]; steady sub-window z in [H/4, 3H/4].
    TRIM: armed steps with the edge x in [-r, r]; steady sub-window x in [-r/2, r/2]."""
    z = np.array([p['pos'][2] for p in poses])
    x = np.array([p['pos'][0] for p in poses])
    armed = np.array([p['armed'] for p in poses])
    if case == 'TRIM':
        win = armed & (x >= -DOME_R-1e-9) & (x <= DOME_R+1e-9)
        steady = armed & (np.abs(x) <= DOME_R/2+1e-9)
    else:
        H = BAR[2]
        win = armed & (z <= H+1e-9) & (z >= geo['edge_stop_z_m']-1e-9)
        steady = armed & (z >= H/4-1e-9) & (z <= 3*H/4+1e-9)
    return win, steady


def crossing_links(reference, lattice, plane_origin, plane_normal, region=None):
    """Reference six-neighbour links (i, j) whose endpoints lie strictly on opposite sides of a plane."""
    from robotic_cutting_sim.cut_events import lattice_neighbours
    nb = lattice_neighbours(lattice)
    sd = (np.asarray(reference)-np.asarray(plane_origin))@np.asarray(plane_normal)
    rows = []
    for col in (1, 3, 5):   # +x, +y, +z neighbours: each link once
        a = np.flatnonzero(nb[:, col] >= 0)
        b = nb[a, col]
        cross = (sd[a] < 0) != (sd[b] < 0)
        rows.append(np.c_[a[cross], b[cross]])
    links = np.concatenate(rows)
    if region is not None and len(links):
        links = links[region[links[:, 0]] & region[links[:, 1]]]
    return links


def link_coverage(links, digits, lattice=None, layer=None):
    """Fraction of links with conflicting digits (1 vs 2); optional restriction to both endpoints in z layer."""
    if layer is not None:
        keep = (lattice[links[:, 0], 2] == layer) & (lattice[links[:, 1], 2] == layer)
        links = links[keep]
    if not len(links):
        return {'links': 0, 'conflicting': 0, 'coverage': None}
    da, db = digits[links[:, 0]], digits[links[:, 1]]
    conflict = ((da == 1) & (db == 2)) | ((da == 2) & (db == 1))
    return {'links': int(len(links)), 'conflicting': int(conflict.sum()), 'coverage': float(conflict.mean()),
            'both_unassigned': int(((da == 0) & (db == 0)).sum()), 'one_unassigned': int(((da == 0) ^ (db == 0)).sum())}


def severed_lines(reference, lattice, digits, plane_origin, plane_normal, band=.004, layer=None):
    """Fraction of reference lattice lines crossing a declared plane that are severed near it.

    Lines run along the lattice axis closest to the plane normal. A line is counted if one of its six-neighbour
    links straddles the plane; it is severed if any of its links whose reference midpoint lies within ``band``
    (4 mm) of the plane carries conflicting digits (1 vs 2). Robust to a cut realised a few tenths of a millimetre
    to millimetres off the declared reference plane (e.g. the second MULTI2 cut on material displaced by the first).
    ``layer``: only lines in that z layer (for lines not along z)."""
    from robotic_cutting_sim.cut_events import lattice_neighbours
    X, lat = np.asarray(reference), np.asarray(lattice)
    normal = np.asarray(plane_normal, float)
    axis = int(np.argmax(np.abs(normal)))
    nb = lattice_neighbours(lat)
    a = np.flatnonzero(nb[:, 2*axis+1] >= 0)
    b = nb[a, 2*axis+1]
    sd = (X-np.asarray(plane_origin))@normal
    mid = (sd[a]+sd[b])/2
    straddle = (sd[a] < 0) != (sd[b] < 0)
    da, db = digits[a], digits[b]
    conflict = ((da == 1) & (db == 2)) | ((da == 2) & (db == 1))
    other = [i for i in range(3) if i != axis]
    keys = lat[a][:, other[0]].astype(np.int64)*100000+lat[a][:, other[1]]
    keep = np.ones(len(a), bool) if layer is None else lat[a, 2] == layer
    lines = np.unique(keys[straddle & keep])
    severed = np.unique(keys[(np.abs(mid) <= band) & conflict & keep])
    severed = np.intersect1d(severed, lines)
    return {'lines': int(len(lines)), 'severed': int(len(severed)),
            'fraction': float(len(severed)/len(lines)) if len(lines) else None, 'axis': axis, 'band_m': band}


def cfl(dx, dt):
    lam = MATERIAL['young']*MATERIAL['poisson']/((1+MATERIAL['poisson'])*(1-2*MATERIAL['poisson']))
    mu = MATERIAL['young']/(2*(1+MATERIAL['poisson']))
    c = float(np.sqrt((lam+2*mu)/MATERIAL['density']))
    return {'p_wave_speed_m_s': c, 'grid_cfl': float(dt*c/dx), 'particle_cfl': None}


def pct(values, q=95):
    values = np.asarray(values, float)
    return float(np.percentile(values, q)) if len(values) else None


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
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def gpu_process_memory_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            pid, mem = [v.strip() for v in line.split(',')]
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:  # reported as unavailable
        return None
    return None


def analyze_state(name, x, codes, F, ctx):
    """Topology, bridges, link coverage, deformation near the cut, board penetration and det F of one state."""
    from robotic_cutting_sim import cut_diagnostics as cd
    from robotic_cutting_sim.cut_topology import mechanical_components_packed
    X, lat, m, case, geo = ctx['reference'], ctx['lattice'], ctx['mass'], ctx['case'], ctx['geometry']
    count, labels = mechanical_components_packed(ctx['nodes'], ctx['weights'], codes, ctx['cuts'],
                                                 gradients=ctx['gradients'])
    mass = np.bincount(labels, weights=m, minlength=count)
    order = np.argsort(-mass, kind='stable')
    fractions = mass[order]/m.sum()
    out = {'state': name, 'mechanical_components': int(count), 'mass_fractions': fractions[:8].tolist(),
           'labelled_particles': int(np.count_nonzero(codes))}
    u = x-X
    # Displacement relative to the mass-weighted mean displacement of the particle's own mechanical component
    # (removes the rigid translation of separated pieces; rotation is not removed).
    comp_u = np.stack([np.bincount(labels, weights=m*u[:, i], minlength=count) for i in range(3)], 1)/mass[:, None]
    w = u-comp_u[labels]
    det = np.linalg.det(F.astype(np.float64))
    out['det_F'] = {'min': float(det.min()), 'p01': pct(det, 1), 'nonpositive': int((det <= 0).sum()),
                    'below_0_5': int((det < .5).sum()), 'min_reference_m': X[int(np.argmin(det))].tolist()}
    out['board'] = {'min_particle_z_m': float(x[:, 2].min()), 'particle_centres_below_board_top': int((x[:, 2] < 0).sum())}
    if case == 'TRIM':
        plane = geo['plane_z_m']
        above = X[:, 2] > plane
        base = np.bincount(labels[~above], weights=m[~above]).argmax()
        removal = float(m[above & (labels != base)].sum()/m[above].sum())
        substrate = float(m[~above & (labels == base)].sum()/m[~above].sum())
        lowest = above & (X[:, 2] < plane+ctx['spacing'])
        digits = cd.digits_of(codes, 0, 'packed2')
        bridges = cd.direct_bridges(ctx['nodes'], ctx['weights'], ctx['gradients'], digits)
        foot = (X[:, 0]**2+X[:, 1]**2 <= DOME_R**2)
        column = (~above) & (X[:, 2] >= plane-.003) & foot
        links = crossing_links(X, lat, [0., 0., plane], [0., 0., 1.])
        out.update(removal_fraction=removal, substrate_retained=substrate,
                   trim_criterion_wp11=bool(removal >= .9 and substrate >= .99),
                   lowest_dome_layer={'count': int(lowest.sum()), 'attached_to_substrate': int((lowest & (labels == base)).sum()),
                                      'z_ref_m': float(X[lowest, 2].min()) if lowest.any() else None},
                   unassigned_dome_particles=int((above & (digits == 0)).sum()),
                   direct_code0_bridges=int(len(bridges)),
                   trim_plane_links=link_coverage(links, digits),
                   severed_lines=severed_lines(X, lat, digits, [0., 0., plane], [0., 0., 1.]),
                   column={'definition': 'substrate particles with z_ref in [plane - 3 mm, plane) under the dome footprint',
                           'particles': int(column.sum()), 'p95_abs_displacement_m': pct(np.linalg.norm(u[column], axis=1)),
                       'p95_abs_displacement_minus_piece_translation_m': pct(np.linalg.norm(w[column], axis=1)),
                           'p95_depression_m': pct(-u[column, 2]), 'median_depression_m': pct(-u[column, 2], 50),
                           'max_depression_m': float((-u[column, 2]).max()) if column.any() else None})
        return out, labels
    planes = geo['planes']
    per = []
    near_any = np.zeros(len(X), bool)
    bridges_total = 0
    for p in planes:
        digits = cd.digits_of(codes, p['cut'], 'packed2')
        bridges = cd.direct_bridges(ctx['nodes'], ctx['weights'], ctx['gradients'], digits)
        bridges_total += len(bridges)
        sd = (X-np.asarray(p['origin']))@np.asarray(p['normal'])
        near = np.abs(sd) <= .003
        near_any |= near
        links = crossing_links(X, lat, p['origin'], p['normal'])
        per.append({'cut': p['cut'], 'direct_code0_bridges': int(len(bridges)),
                    'bridges_reference_z_max_m': float(X[bridges, 2].max()) if len(bridges) else None,
                    'links_all': link_coverage(links, digits), 'links_layer0': link_coverage(links, digits, lat, 0),
                    'links_layer1': link_coverage(links, digits, lat, 1),
                    'severed_lines': severed_lines(X, lat, digits, p['origin'], p['normal']),
                    'severed_lines_layer0': severed_lines(X, lat, digits, p['origin'], p['normal'], layer=0),
                    'unassigned_within_3mm': int((near & (digits == 0)).sum()),
                    'labels_wrong_side_beyond_2mm': int(((digits != 0) & (digits != np.where(sd >= 0, 2, 1))
                                                         & (np.abs(sd) > .002)).sum())})
    expected = len(planes)+1
    f = fractions
    each = .4 if expected == 2 else .25
    out.update(per_plane=per, direct_code0_bridges=int(bridges_total), expected_pieces=expected,
               separation_criterion=bool(count >= expected and len(f) >= expected and f[expected-1] >= each
                                         and f[:expected].sum() >= .99),
               separation_rule=f'{expected} mechanical parts, each >= {each:.2f} and together >= 0.99 of the mass',
               column={'definition': 'particles with |reference distance to a declared cut plane| <= 3 mm',
                       'particles': int(near_any.sum()), 'p95_abs_displacement_m': pct(np.linalg.norm(u[near_any], axis=1)),
                       'p95_abs_displacement_minus_piece_translation_m': pct(np.linalg.norm(w[near_any], axis=1)),
                       'p95_abs_dz_m': pct(np.abs(u[near_any, 2])), 'p95_abs_dx_m': pct(np.abs(u[near_any, 0]))})
    return out, labels


def surface_fidelity(state, ctx, device):
    """surface_metrics_v1 current- and reference-space cut-face P95 of particle_surface_v1 (A, CuttingSurface
    current_volume_v1) and cut_aware_surface_v1 (B, recorder Hermite log, total-F embedding) on one state.

    Reuses the Phase 3 driver functions read-only from private module instances whose spacing globals are set to
    this row's spacing (the Phase 1/3 drivers are written for 1 mm; nothing in those files changes)."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    s = ctx['spacing']

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    P3 = load('wp12_phase3_surface_resolution_private', ROOT/'scripts/run_wp12_phase3_surface.py')
    P3.S = P3.P1.S = s
    P3.DX = P3.P1.DX = ctx['dx']
    run = {'reference': ctx['reference'], 'lattice': ctx['lattice'], 'masses': ctx['mass'], 'density': MATERIAL['density'],
           'positions': state['positions'].astype(float), 'codes': state['codes'], 'F': state['F'].astype(float),
           'nodes': ctx['nodes'], 'weights': ctx['weights'], 'gradients': ctx['gradients'], 'cuts': ctx['cuts'],
           'log': state['log']}
    count, labels, lat_count, fractions = P3.P1.components(run)
    rank = run['_rank']
    locus = SM.material_locus(run['reference'], run['lattice'], run['positions'], run['codes'], cuts=run['cuts'],
                              F=run['F'], events=run['log'])
    out = {'mechanical_components': int(count), 'lattice_components': int(lat_count),
           'conflicting_link_rows': int(len(locus['cut'])), 'events_in_log': int(len(run['log']))}
    meshes = {}
    t = time.perf_counter()
    meshes['particle_surface_v1'] = P3.P1.extract_surfaces(run, device, profiles=('current_volume_v1',))['current_volume_v1']
    meshes['particle_surface_v1']['wall_s'] = time.perf_counter()-t
    b = P3.new_b(run, run['log'])
    m = P3.b_extract(b, run, run['log'])
    m['face_component'] = P3.P1.face_components(m['info'], rank, len(m['faces']))
    meshes['cut_aware_surface_v1'] = m
    ext = SM.exterior_samples(run['lattice'], run['positions'], s, run['F'])
    for name, mesh in meshes.items():
        v, f = np.asarray(mesh['vertices'], float), np.asarray(mesh['faces'], np.int64)
        row = {'vertices': int(len(v)), 'triangles': int(len(f)), 'extract_wall_s': mesh['wall_s']}
        try:
            rep = SM.surface_report(run['reference'], run['lattice'], run['positions'], run['codes'], v, f, spacing=s,
                                    labels=labels, masses=run['masses'], density=run['density'], cuts=run['cuts'],
                                    F=run['F'], locus=locus, face_component=mesh.get('face_component'),
                                    reference_space=False)
            cls = SM.classify_mesh_vertices(v, run['positions'], labels, locus, ext, spacing=s)
            ref = P3.reference_space(run, labels, locus, cls, v, f,
                                     mesh['last']['vertex_reference'] if name == 'cut_aware_surface_v1' else None)
            sc = P3.scalars(rep, ref)
            row.update(status='MEASURED', scalars=sc,
                       current_p95_m=None if sc['current_p95_s'] is None else sc['current_p95_s']*s,
                       reference_p95_m=None if sc['reference_p95_s'] is None else sc['reference_p95_s']*s,
                       reference_source=ref.get('source'), reference_status=ref.get('status'))
        except Exception as error:   # recorded, never hides the physics row
            row.update(status='ERROR', error=repr(error))
        out[name] = row
    out['labels'] = {'particle_surface_v1': 'A = CuttingSurface current_volume_v1 (voxel s/2), reference space = '
                                            'approximate pull-back (diagnostic, as Phase 3 declared)',
                     'cut_aware_surface_v1': 'B = CutAwareSurface, cut_events_v1 Hermite log, total-F embedding, '
                                             'reference space = exact pre-embedding vertices'}
    return out


def run(args, row, out):
    import warp as wp
    from robotic_cutting_sim.coupled_robot import interpolate_measured_poses
    from robotic_cutting_sim.cut_events import CutEventRecorder
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    from robotic_cutting_sim.tool_profile import PROFILES, handle_box_edge
    case, s, dx, dt, variant = row['case'], row['spacing_m'], row['grid_dx_m'], row['dt_s'], row.get('variant', 'formal')
    substeps = int(round(CONTROL_DT/dt))
    if abs(substeps*dt-CONTROL_DT) > 1e-12:
        raise ValueError(f'dt {dt} does not divide the 20 ms control step')
    profile = PROFILES[TOOL]
    points, lattice, masses, geo = workpiece(case, s)
    poses, pgeo = plan(case, s, variant)
    geo.update(pgeo)
    max_steps = args.max_steps or row.get('max_steps')
    if max_steps:
        poses = poses[:max_steps]
    cuts = 2 if case == 'MULTI2' else 1
    config = TLMPMConfig(dx=dx, dt=dt, density=MATERIAL['density'], young=MATERIAL['young'], poisson=MATERIAL['poisson'],
                         yield_stress=MATERIAL['yield_stress'], contact_friction=MATERIAL['contact_friction'],
                         cutting_energy=MATERIAL['cutting_energy'], blade_length=profile.blade_depth_m,
                         blade_height=profile.thickness_m, blade_half_width=profile.half_width_m, cuts=cuts,
                         floor=False, fixture_walls=False, gravity=(0., 0., -9.81))
    polygon, _ = profile.tlmpm_profile()
    t0 = time.monotonic()
    bed = TLMPM(points.astype(np.float32), masses.astype(np.float32), s, config, args.device, fixture_boxes=[BOARD_BOX],
                table_plane=TABLE_PLANE, blade_profile_xz=polygon, cutting_law=8,
                obstacle_boxes_edge=[handle_box_edge(profile)], slots_per_node=row.get('slots_per_node', 8))
    recorder = CutEventRecorder(bed, lattice, summary_capacity=max(1 << 18, len(poses)*substeps+16))
    init_wall = time.monotonic()-t0
    first = np.r_[poses[0]['pos'], poses[0]['quat']].astype(np.float32)
    bed.set_trajectory(np.tile(first, (substeps+1, 1)), np.zeros((substeps+1, 6), np.float32),
                       np.full(substeps+1, -1, np.int32))
    cuda = wp.get_device(args.device).is_cuda
    graph = None
    t0 = time.monotonic()
    if cuda:
        wp.load_module(module=TLMPM.__module__, device=args.device)
        wp.load_module(module='robotic_cutting_sim.cut_events', device=args.device)
        with wp.ScopedCapture(device=args.device) as capture:
            for _ in range(substeps):
                recorder.before_step()
                bed.step()
                recorder.after_step()
        graph = capture.graph
    capture_wall = time.monotonic()-t0
    X = points.astype(np.float64)
    ctx = {'reference': X, 'lattice': lattice, 'mass': masses, 'case': case, 'geometry': geo, 'spacing': s, 'dx': dx,
           'cuts': cuts, 'nodes': bed.nodes.numpy(), 'weights': bed.weights.numpy(), 'gradients': bed.gradients.numpy()}
    previous = first
    n_steps = len(poses)
    force = np.zeros((n_steps, 3))
    stats = np.zeros((n_steps, 7))
    det_min = np.zeros(n_steps)
    vmax = np.zeros(n_steps)
    zmin = np.zeros(n_steps)
    below = np.zeros(n_steps, int)
    labelled = np.zeros(n_steps, int)
    launch_ms = np.zeros(n_steps)
    finite = True
    snaps = {}
    loop_start = time.monotonic()
    for k, p in enumerate(poses):
        pose = np.r_[p['pos'], p['quat']].astype(np.float32)
        P, T = interpolate_measured_poses(previous, pose, substeps, CONTROL_DT)
        bed.poses.assign(P)
        bed.twists.assign(T)
        bed.cuts.assign(np.full(substeps+1, p['cut'] if p['armed'] else -1, np.int32))
        bed.index.zero_()
        bed.reaction.zero_()
        bed.box_reactions.zero_()
        bed.stats.zero_()
        wp.synchronize_device(args.device)
        t = time.perf_counter()
        if graph is not None:
            wp.capture_launch(graph)
        else:
            for _ in range(substeps):
                recorder.before_step()
                bed.step()
                recorder.after_step()
        wp.synchronize_device(args.device)
        launch_ms[k] = (time.perf_counter()-t)*1e3
        force[k] = bed.reaction.numpy()[0][:3].astype(float)/CONTROL_DT
        stats[k] = bed.stats.numpy().astype(float)
        x = bed.x.numpy()
        v = bed.v.numpy()
        F = bed.F.numpy()
        det = np.linalg.det(F.astype(np.float64))
        det_min[k] = det.min()
        vmax[k] = float(np.sqrt((v.astype(np.float64)**2).sum(1).max()))
        zmin[k] = float(x[:, 2].min())
        below[k] = int((x[:, 2] < 0).sum())
        code = bed.code.numpy()
        labelled[k] = int(np.count_nonzero(code))
        if not (np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(det).all()):
            finite = False
            break
        nxt = poses[k+1]['phase'] if k+1 < n_steps else None
        if p['phase'].startswith('CUT') and nxt != p['phase']:   # deepest point of each armed cut
            snaps[f'END_{p["phase"]}'] = {'positions': x.copy(), 'codes': code.copy(), 'F': F.copy(), 'step': k+1}
        previous = pose
    loop_wall = time.monotonic()-loop_start
    bed.check_slots()
    counters = recorder.check()
    log = recorder.log()
    st = bed.snapshot()
    snaps['FINAL'] = {'positions': st['positions'], 'codes': st['codes'], 'F': st['F'], 'step': k+1}
    analysis = {}
    for name, sn in snaps.items():
        a, _ = analyze_state(name, sn['positions'].astype(np.float64), sn['codes'], sn['F'], ctx)
        a['control_step'] = int(sn['step'])
        analysis[name] = a
    ran = k+1
    win, steady = contact_window(case, poses[:ran], geo)
    fmag = np.linalg.norm(force[:ran], axis=1)
    adv = np.asarray(geo['advance'])
    f_adv = -force[:ran]@adv    # resistance against the advance (> 0 when the material pushes back)
    contact = {'window_rule': contact_window.__doc__.split('\n')[0].strip(),
               'window_steps': int(win.sum()), 'steady_steps': int(steady.sum()),
               'mean_abs_N': float(fmag[win].mean()) if win.any() else None,
               'peak_abs_N': float(fmag[win].max()) if win.any() else None,
               'mean_resistance_along_advance_N': float(f_adv[win].mean()) if win.any() else None,
               'steady_mean_abs_N': float(fmag[steady].mean()) if steady.any() else None,
               'steady_mean_resistance_along_advance_N': float(f_adv[steady].mean()) if steady.any() else None,
               'peak_is': 'max over control steps of the 20 ms mean |F| (not a substep peak)',
               'contact_steps_any_blade_contact': int((stats[:ran, 1] > 0).sum())}
    armed = np.array([p['armed'] for p in poses[:ran]])
    timing = {'init_wall_s': init_wall, 'graph_capture_wall_s': capture_wall, 'loop_wall_s': loop_wall,
              'ms_per_control_step_median': float(np.median(launch_ms[:ran])),
              'ms_per_substep_median': float(np.median(launch_ms[:ran]))/substeps,
              'ms_per_substep_mean': float(launch_ms[:ran].mean())/substeps,
              'ms_per_substep_armed_median': float(np.median(launch_ms[:ran][armed]))/substeps if armed.any() else None,
              'scope': 'CUDA-graph launch of the substeps + device wait, cut_events_v1 recorder on, host analysis excluded'}
    memory = {'solver_device_bytes': bed.memory_bytes(), 'recorder_device_bytes': recorder.metadata()['device_bytes'],
              'mempool_high_bytes': (int(wp.get_mempool_used_mem_high(args.device))
                                     if cuda and wp.is_mempool_enabled(args.device) else None),
              'nvidia_smi_process_mib': gpu_process_memory_mib()}
    stability = {'finite': bool(finite), 'min_det_F_all_steps': float(det_min[:ran].min()),
                 'min_det_F_step': int(np.argmin(det_min[:ran]))+1, 'max_speed_m_s': float(vmax[:ran].max()),
                 'stable': bool(finite and det_min[:ran].min() > 0 and vmax[:ran].max() < 5.),
                 'rule': 'finite x, v, det F at every control step; min det F > 0; max particle speed < 5 m/s '
                         '(WP-10 screen); sampled at control-step ends (20 ms), not substeps'}
    penetration = {'max_fixture_contact_overlap_m': float(stats[:ran, 0].max()),
                   'max_blade_contact_overlap_m': float(stats[:ran, 3].max()),
                   'max_particle_pair_overlap_m': float(stats[:ran, 6].max()),
                   'min_particle_z_m': float(zmin[:ran].min()),
                   'max_particle_centres_below_board_top': int(below[:ran].max()),
                   'planned_min_edge_z_m': float(min(p['pos'][2] for p in poses[:ran])),
                   'contact_radius_m': float(bed.radius),
                   'definition': 'fixture overlap = max(radius - board/table SDF) of material contact spheres (tlmpm '
                                 'stats[0]); centres below board top = particle centres with z < 0 (inside the board '
                                 'footprint); blade overlap = tlmpm stats[3]'}
    surface = None
    if not (args.no_surface or row.get('no_surface')) and finite:
        t = time.monotonic()
        try:
            surface = surface_fidelity({'positions': snaps['FINAL']['positions'], 'codes': snaps['FINAL']['codes'],
                                        'F': snaps['FINAL']['F'], 'log': log}, ctx, args.device)
        except Exception as error:
            surface = {'status': 'ERROR', 'error': repr(error)}
        surface['wall_s'] = time.monotonic()-t
        memory['mempool_high_after_surface_bytes'] = (int(wp.get_mempool_used_mem_high(args.device))
                                                      if cuda and wp.is_mempool_enabled(args.device) else None)
    np.savez_compressed(out/'state.npz', reference=X.astype(np.float32), lattice=lattice.astype(np.int32),
                        masses=masses.astype(np.float32), cut_events_log=log,
                        **{f'{n}_{k2}': sn[k2] for n, sn in snaps.items() for k2 in ('positions', 'codes', 'F')})
    np.savez_compressed(out/'steps.npz', force_N=force[:ran], stats=stats[:ran], det_min=det_min[:ran],
                        vmax=vmax[:ran], zmin=zmin[:ran], below=below[:ran], labelled=labelled[:ran],
                        launch_ms=launch_ms[:ran], window=win, steady=steady, armed=armed,
                        edge_pos=np.array([p['pos'] for p in poses[:ran]]),
                        phase=np.array([p['phase'] for p in poses[:ran]]))
    return {'geometry': geo, 'particles': int(len(points)), 'grid_nodes': int(bed.node_count),
            'grid_dims': np.asarray(bed.grid_dims).tolist(), 'substeps_per_control_step': substeps,
            'control_steps_run': ran, 'control_steps_planned': n_steps, 'cfl': cfl(dx, dt),
            'particles_per_cell_edge': dx/s, 'contact_radius_m': float(bed.radius),
            'law8_parameters': bed.law8_parameters, 'law4_parameters': bed.law4_parameters,
            'slot_diagnostics': bed.slot_diagnostics(), 'cut_events': counters, 'analysis': analysis,
            'contact_force': contact, 'timing': timing, 'memory': memory, 'stability': stability,
            'penetration': penetration, 'surface_final': surface}


def declared_rows():
    if not PROTOCOL.exists():
        return {}, None
    protocol = json.loads(PROTOCOL.read_text())
    return protocol.get('rows', {}), hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', help='row id declared in configs/wp12/resolution_protocol_v1.json')
    p.add_argument('--dev', nargs=5, metavar=('CASE', 'SPACING_MM', 'GRID_MM', 'DT_MS', 'VARIANT'),
                   help='development/feasibility row (labelled DEV, never formal)')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-steps', type=int, help='DEV only: cap control steps')
    p.add_argument('--no-surface', action='store_true')
    a = p.parse_args()
    rows, protocol_sha = declared_rows()
    if a.dev:
        case, sm, gm, dtm, variant = a.dev
        row_id = f'DEV_{case}_s{sm}_g{gm}_dt{dtm}_{variant}'
        row = dict(case=case, spacing_m=float(sm)*1e-3, grid_dx_m=float(gm)*1e-3, dt_s=float(dtm)*1e-3, variant=variant)
        label = 'DEV (development / feasibility run; never a formal protocol row)'
    else:
        if a.row and a.row.endswith('_slots16') and a.row[:-len('_slots16')] in rows:
            # Declared failure rule: a SlotOverflowError row is rerun once with slots_per_node 16 under this id.
            rows[a.row] = {**rows[a.row[:-len('_slots16')]], 'slots_per_node': 16, 'slot_overflow_rerun': True}
        if a.row not in rows:
            p.error(f'--row {a.row!r} is not declared in {PROTOCOL.name}')
        if a.max_steps:
            p.error('--max-steps is DEV only')
        row_id, row, label = a.row, rows[a.row], 'FORMAL resolution_protocol_v1 row'
    out = a.out
    out.mkdir(parents=True, exist_ok=False)  # never overwrite
    base = {'row_id': row_id, 'row': row, 'label': label, 'command': sys.argv, **git_state(), 'seed': SEED,
            'device': a.device, 'protocol_sha256': protocol_sha, 'solver': 'tlmpm_v5', 'cutting_law': 8,
            'tool_profile': TOOL, 'material': {**MATERIAL, 'calibration': 'provisional_uncalibrated'},
            'workcell': 'blade_master_flat_board_v1 geometry (board box + table plane), kinematic ideal tracking',
            'host': os.uname().nodename, 'started_unix': time.time(),
            'scope': 'Kinematic ideal tracking; provisional uncalibrated material; no robot, PhysX or cameras'}
    (out/'manifest.json').write_text(json.dumps(jsonable(base), indent=1))
    start = time.monotonic()
    try:
        result = run(a, row, out)
        result['wall_s'] = time.monotonic()-start
        result['status'] = 'COMPLETED' if result['stability']['finite'] else 'UNSTABLE'
        (out/'result.json').write_text(json.dumps(jsonable({**base, **result}), indent=1))
        final = result['analysis']['FINAL']
        print(row_id, result['status'], 'components', final['mechanical_components'],
              'removal' if row['case'] == 'TRIM' else 'bridges',
              final.get('removal_fraction', final.get('direct_code0_bridges')),
              'force', result['contact_force']['mean_abs_N'], 'wall', round(result['wall_s'], 1), flush=True)
    except Exception as error:
        (out/'result.json').write_text(json.dumps(jsonable({**base, 'status': 'ERROR', 'error': repr(error),
                                                            'wall_s': time.monotonic()-start}), indent=1))
        raise


if __name__ == '__main__':
    main()
