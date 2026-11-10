"""Single-env reference rubber solver ``tlmpm_rubber_v1`` (ADR 0023 D1-D6, §5, §7, §9; task R1.2).

Status ``EXPERIMENTAL_NOT_ACCEPTED`` (frozen only at the P2-2 acceptance). Opt-in: nothing in the cutting simulator
imports it. Material ``neo_hookean_jbar_v1`` (``rubber_constitutive``), pure FLIP (pic = 0, anything else refused),
MUSL, grid-consistent positions, no artificial damping in dynamics (the mass-proportional damping impulse exists only
for the labelled static initializer, ``damping_rate`` > 0 refused unless ``initializer=True`` is passed).

One substep (17 launches, ADR §13):

1.  x_start <- x
2.  J-bar pass ``rubber_constitutive.cell_jbar`` (cell CSR gather, float64, deterministic)
3.  stress pass ``_stress``: P = PK1(F, Jbar_cell); J / Jbar outside the domain counts a violation (sticky flag)
4.  ``_loads``: impulse = traction + table / fixture contacts (frozen ``_contact_impulse``) [+ initializer damping]
5.  zero grid; frozen ``tlmpm_v5._p2g``; frozen ``tlmpm_v5._grid_update`` (S = 1, slot tables constant 0)
6.  ``_bc`` (reaction pass): nodal Dirichlet groups set the masked velocity components to the position-exact
    prescribed velocity (x(i+1) - x(i)) / dt of the group's interpolated motion; the impulse m (v - v~) and its torque
    about the group origin are accumulated per group in float64 (the measured clamp / grip wrench)
7.  frozen ``_g2p`` (pic = 0); zero gp/gj; frozen ``_musl``; frozen ``_grid_update``; ``_bc`` (no reaction)
8.  frozen ``_deformation`` (F += dt sum v_I (x) grad N_I); frozen ``tlmpm_v6._grid_positions``; frozen ``_advance``

Frozen kernels are launched unchanged (``tlmpm_v5.py``, ``tlmpm_v6.py`` byte-identical). Group motions are rigid
(slerp of the rotation and linear translation, x = R(s) X + b(s)) or affine (x = A(s) X + b(s), linear in s; patch /
homogeneous tests), s = substep index / substeps per control step, reference node positions X in float64.
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from . import tlmpm_v5 as v5
from . import rubber_constitutive as rc
from .rubber_constitutive import det_d, in_domain, nh_jbar_pk1, iso_energy_density, vol_energy_density
from .tlmpm_v6 import _grid_positions

wp.set_module_options({"enable_backward": False})

SOLVER_NAME = "tlmpm_rubber_v1"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
ADR = "docs/adr/0023-rubber-material-v1.md"
GROUP_RIGID, GROUP_AFFINE = 0, 1
DIAG_COLUMNS = ("nonfinite", "J_min", "J_max", "volume_ratio", "Jbar_min", "Jbar_max", "frac_cells_jbar_dev_gt_0_05",
                "elastic_iso", "elastic_vol", "elastic", "kinetic_particle", "kinetic_grid", "gravity",
                "max_speed_m_s", "cells", "particles", "max_grid_speed_m_s")
NDIAG = len(DIAG_COLUMNS)
_HERE = Path(__file__).resolve().parent
SOURCES = ("tlmpm_rubber_v1.py", "rubber_constitutive.py")
FROZEN_KERNELS = ("tlmpm_v5._p2g", "tlmpm_v5._grid_update", "tlmpm_v5._g2p", "tlmpm_v5._musl",
                  "tlmpm_v5._deformation", "tlmpm_v5._advance", "tlmpm_v5._contact_impulse", "tlmpm_v5._box_sdf",
                  "tlmpm_v6._grid_positions")


def source_sha256():
    out = {name: hashlib.sha256((_HERE/name).read_bytes()).hexdigest() for name in SOURCES}
    out.update({name: hashlib.sha256((_HERE/name).read_bytes()).hexdigest() for name in ("tlmpm_v5.py", "tlmpm_v6.py")})
    return out


# ----------------------------------------------------------------------------- shared functions (single and batch)

@wp.func
def group_target(kind: int, q0: wp.quatd, q1: wp.quatd, b0: wp.vec3d, b1: wp.vec3d, A0: wp.mat33d, A1: wp.mat33d,
                 s: wp.float64, X: wp.vec3d):
    """Target position of reference point X at interpolation parameter s (rigid: slerp; affine: linear)."""
    b = b0+s*(b1-b0)
    M = A0+s*(A1-A0)
    if kind == GROUP_RIGID:
        M = wp.quat_to_matrix(wp.quat_slerp(q0, q1, s))
    return M*X+b


@wp.func
def apply_mask(v: wp.vec3, vp: wp.vec3d, mask: int):
    out = v
    if (mask & 1) != 0:
        out[0] = float(vp[0])
    if (mask & 2) != 0:
        out[1] = float(vp[1])
    if (mask & 4) != 0:
        out[2] = float(vp[2])
    return out


@wp.func
def bc_reaction(m: float, v_old: wp.vec3, v_new: wp.vec3, x1: wp.vec3d, o1: wp.vec3d):
    """Impulse j = m (v_new - v_old) applied to the material by the group and its moment about the group origin."""
    j = wp.float64(m)*(wp.vec3d(v_new)-wp.vec3d(v_old))
    tau = wp.cross(x1-o1, j)
    return wp.spatial_vectord(j[0], j[1], j[2], tau[0], tau[1], tau[2])


@wp.func
def bc_work(m: float, v_old: wp.vec3, v_new: wp.vec3):
    """Work of the constraint on the material: impulse j = m (v_new - v_old) times the prescribed velocity v_new
    (zero on unmasked components, where j = 0; zero for a fixed group). Numerical constraint dissipation is
    therefore NOT counted as external work and shows up in the energy balance."""
    return wp.float64(m)*wp.dot(wp.vec3d(v_new)-wp.vec3d(v_old), wp.vec3d(v_new))


@wp.func
def contact_load(xp: wp.vec3, vp: wp.vec3, mp: float, dt: float, radius: float, stiffness: float, friction: float,
                 use_plane: int, plane: wp.vec4, box_start: int, box_stop: int, box_c: wp.array(dtype=wp.vec3),
                 box_h: wp.array(dtype=wp.vec3)):
    """Table plane and fixture boxes with the frozen projection/Coulomb helper (as ``fixture_contacts``)."""
    imp = wp.vec3(0.0)
    for b in range(box_start, box_stop):
        hit = v5._box_sdf(xp-box_c[b], box_h[b])
        imp = imp+v5._contact_impulse(hit[3]-radius, wp.vec3(hit[0], hit[1], hit[2]), vp, mp, dt, stiffness, friction)
    if use_plane != 0:
        n = wp.vec3(plane[0], plane[1], plane[2])
        imp = imp+v5._contact_impulse(wp.dot(n, xp)+plane[3]-radius, n, vp, mp, dt, stiffness, friction)
    return imp


# ----------------------------------------------------------------------------- single-env kernels

@wp.kernel
def _stress(F: wp.array(dtype=wp.mat33), nodes: wp.array2d(dtype=int), jbar: wp.array(dtype=wp.float64),
            stress: wp.array(dtype=wp.mat33), mu: float, kappa: float, lo: wp.float64, hi: wp.float64,
            violations: wp.array(dtype=int)):
    p = wp.tid()
    f = F[p]
    jb = jbar[nodes[p, 0]]
    if not (in_domain(det_d(f), lo, hi) and in_domain(jb, lo, hi)):
        wp.atomic_add(violations, 0, 1)
    stress[p] = nh_jbar_pk1(f, jb, mu, kappa)


@wp.kernel
def _loads(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
           impulse: wp.array(dtype=wp.vec3), particle_traction: wp.array(dtype=int),
           traction_force: wp.array(dtype=wp.vec3), traction_mass: wp.array(dtype=float), dt: float, damping: float,
           radius: float, stiffness: float, friction: float, use_plane: int, plane: wp.vec4, boxes: int,
           box_c: wp.array(dtype=wp.vec3), box_h: wp.array(dtype=wp.vec3), work: wp.array(dtype=wp.float64)):
    p = wp.tid()
    xp, vp, mp = x[p], v[p], mass[p]
    imp = contact_load(xp, vp, mp, dt, radius, stiffness, friction, use_plane, plane, 0, boxes, box_c, box_h)
    t = particle_traction[p]
    if t >= 0:
        jt = traction_force[t]*(dt*mp/traction_mass[t])
        imp = imp+jt
        wp.atomic_add(work, 0, wp.dot(wp.vec3d(jt), wp.vec3d(vp)))
    if damping > 0.0:
        imp = imp-(damping*dt*mp)*vp
    impulse[p] = imp


@wp.kernel
def _bc(node_group: wp.array(dtype=int), node_mask: wp.array(dtype=int), node_X: wp.array(dtype=wp.vec3d),
        gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), index: wp.array(dtype=int), substeps: int,
        dt: wp.float64, kind: wp.array(dtype=int), active: wp.array(dtype=int), q0: wp.array(dtype=wp.quatd),
        q1: wp.array(dtype=wp.quatd), b0: wp.array(dtype=wp.vec3d), b1: wp.array(dtype=wp.vec3d),
        A0: wp.array(dtype=wp.mat33d), A1: wp.array(dtype=wp.mat33d), origin: wp.array(dtype=wp.vec3d),
        reaction: wp.array(dtype=wp.spatial_vectord), work: wp.array(dtype=wp.float64), record: int):
    i = wp.tid()
    g = node_group[i]
    if g >= 0:
        if active[g] != 0:
            k = index[0]
            s0 = wp.float64(k)/wp.float64(substeps)
            s1 = wp.float64(k+1)/wp.float64(substeps)
            X = node_X[i]
            x0 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s0, X)
            x1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, X)
            v = new_v[i]
            vn = apply_mask(v, (x1-x0)/dt, node_mask[i])
            new_v[i] = vn
            if record != 0:
                o1 = group_target(kind[g], q0[g], q1[g], b0[g], b1[g], A0[g], A1[g], s1, origin[g])
                wp.atomic_add(reaction, g, bc_reaction(gm[i], v, vn, x1, o1))
                wp.atomic_add(work, 0, bc_work(gm[i], v, vn))


# ----------------------------------------------------------------------------- segmented diagnostics (shared)

@wp.kernel
def diag_particles(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33),
                   mass: wp.array(dtype=float), reference: wp.array(dtype=wp.vec3), gravity: wp.vec3, volume: float,
                   dparams: wp.array2d(dtype=float), particle_offset: wp.array(dtype=int), groups: int,
                   partials: wp.array2d(dtype=wp.float64)):
    """Thread (e, g) strides over env e's particles: nonfinite, J min/max/sum, isochoric energy, particle KE,
    gravity potential relative to the reference, max speed (float64, fixed order: deterministic on any device)."""
    e, grp = wp.tid()
    nonfinite = wp.float64(0.0)
    jmin = wp.float64(1.0e30)
    jmax = wp.float64(-1.0e30)
    jsum = wp.float64(0.0)
    eiso = wp.float64(0.0)
    ke = wp.float64(0.0)
    egrav = wp.float64(0.0)
    vmax = wp.float64(0.0)
    mu = dparams[e, 0]
    vol = wp.float64(volume)
    g = wp.vec3d(gravity)
    for p in range(particle_offset[e]+grp, particle_offset[e+1], groups):
        xp, vp, fp = x[p], v[p], F[p]
        ok = True
        for a in range(3):
            if not wp.isfinite(xp[a]) or not wp.isfinite(vp[a]):
                ok = False
            for b in range(3):
                if not wp.isfinite(fp[a, b]):
                    ok = False
        if ok:
            j = det_d(fp)
            jmin = wp.min(jmin, j)
            jmax = wp.max(jmax, j)
            jsum += j
            eiso += vol*iso_energy_density(fp, mu)
            vd = wp.vec3d(vp)
            m = wp.float64(mass[p])
            ke += wp.float64(0.5)*m*wp.dot(vd, vd)
            egrav -= m*wp.dot(g, wp.vec3d(xp)-wp.vec3d(reference[p]))
            vmax = wp.max(vmax, wp.sqrt(wp.dot(vd, vd)))
        else:
            nonfinite = wp.float64(1.0)
    row = e*groups+grp
    partials[row, 0] = nonfinite
    partials[row, 1] = jmin
    partials[row, 2] = jmax
    partials[row, 3] = jsum
    partials[row, 4] = eiso
    partials[row, 5] = ke
    partials[row, 6] = egrav
    partials[row, 7] = vmax


@wp.kernel
def diag_cells(jbar: wp.array(dtype=wp.float64), cell_node: wp.array(dtype=int), cell_offsets: wp.array(dtype=int),
               cell_env_offset: wp.array(dtype=int), groups: int, dparams: wp.array2d(dtype=float), volume: float,
               partials: wp.array2d(dtype=wp.float64)):
    """Thread (e, g) strides over env e's cells: Jbar min/max, volumetric energy, count |Jbar - 1| > 0.05."""
    e, grp = wp.tid()
    jmin = wp.float64(1.0e30)
    jmax = wp.float64(-1.0e30)
    evol = wp.float64(0.0)
    dev = wp.float64(0.0)
    kappa = dparams[e, 1]
    for c in range(cell_env_offset[e]+grp, cell_env_offset[e+1], groups):
        jb = jbar[cell_node[c]]
        jmin = wp.min(jmin, jb)
        jmax = wp.max(jmax, jb)
        evol += wp.float64(cell_offsets[c+1]-cell_offsets[c])*wp.float64(volume)*vol_energy_density(jb, kappa)
        if wp.abs(jb-wp.float64(1.0)) > wp.float64(0.05):
            dev += wp.float64(1.0)
    row = e*groups+grp
    partials[row, 0] = jmin
    partials[row, 1] = jmax
    partials[row, 2] = evol
    partials[row, 3] = dev


@wp.kernel
def diag_nodes(gm: wp.array(dtype=float), new_v: wp.array(dtype=wp.vec3), node_offset: wp.array(dtype=int),
               groups: int, partials: wp.array2d(dtype=wp.float64)):
    e, grp = wp.tid()
    ke = wp.float64(0.0)
    vmax = wp.float64(0.0)
    for i in range(node_offset[e]+grp, node_offset[e+1], groups):
        vd = wp.vec3d(new_v[i])
        ke += wp.float64(0.5)*wp.float64(gm[i])*wp.dot(vd, vd)
        if gm[i] > 0.0:
            vmax = wp.max(vmax, wp.sqrt(wp.dot(vd, vd)))
    partials[e*groups+grp, 0] = ke
    partials[e*groups+grp, 1] = vmax


@wp.kernel
def diag_combine(pp: wp.array2d(dtype=wp.float64), pc: wp.array2d(dtype=wp.float64),
                 pn: wp.array2d(dtype=wp.float64), gp_: int, gc: int, gn: int, particle_offset: wp.array(dtype=int),
                 cell_env_offset: wp.array(dtype=int), summary: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    nonfinite = wp.float64(0.0)
    jmin = wp.float64(1.0e30)
    jmax = wp.float64(-1.0e30)
    jsum = wp.float64(0.0)
    eiso = wp.float64(0.0)
    ke = wp.float64(0.0)
    egrav = wp.float64(0.0)
    vmax = wp.float64(0.0)
    for k in range(gp_):
        r = e*gp_+k
        nonfinite = wp.max(nonfinite, pp[r, 0])
        jmin = wp.min(jmin, pp[r, 1])
        jmax = wp.max(jmax, pp[r, 2])
        jsum += pp[r, 3]
        eiso += pp[r, 4]
        ke += pp[r, 5]
        egrav += pp[r, 6]
        vmax = wp.max(vmax, pp[r, 7])
    jbmin = wp.float64(1.0e30)
    jbmax = wp.float64(-1.0e30)
    evol = wp.float64(0.0)
    dev = wp.float64(0.0)
    for k in range(gc):
        r = e*gc+k
        jbmin = wp.min(jbmin, pc[r, 0])
        jbmax = wp.max(jbmax, pc[r, 1])
        evol += pc[r, 2]
        dev += pc[r, 3]
    keg = wp.float64(0.0)
    vgrid = wp.float64(0.0)
    for k in range(gn):
        keg += pn[e*gn+k, 0]
        vgrid = wp.max(vgrid, pn[e*gn+k, 1])
    n = wp.float64(particle_offset[e+1]-particle_offset[e])
    cells = wp.float64(cell_env_offset[e+1]-cell_env_offset[e])
    summary[e, 0] = nonfinite
    summary[e, 1] = jmin
    summary[e, 2] = jmax
    summary[e, 3] = jsum/n
    summary[e, 4] = jbmin
    summary[e, 5] = jbmax
    summary[e, 6] = dev/cells
    summary[e, 7] = eiso
    summary[e, 8] = evol
    summary[e, 9] = eiso+evol
    summary[e, 10] = ke
    summary[e, 11] = keg
    summary[e, 12] = egrav
    summary[e, 13] = vmax
    summary[e, 14] = cells
    summary[e, 15] = n
    summary[e, 16] = vgrid


@wp.kernel
def markers_kernel(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), marker_offsets: wp.array(dtype=int),
                   marker_particles: wp.array(dtype=int), out_x: wp.array(dtype=wp.vec3d),
                   out_v: wp.array(dtype=wp.vec3d)):
    """Cross-section centroids (position, mean particle velocity) in float64, one thread per marker."""
    m = wp.tid()
    sx = wp.vec3d()
    sv = wp.vec3d()
    start = marker_offsets[m]
    stop = marker_offsets[m+1]
    for k in range(start, stop):
        p = marker_particles[k]
        sx += wp.vec3d(x[p])
        sv += wp.vec3d(v[p])
    inv = wp.float64(1.0)/wp.float64(stop-start)
    out_x[m] = sx*inv
    out_v[m] = sv*inv


class SegmentedRubberDiagnostics:
    """Per-env diagnostics over flat arrays (single solver: one env). ``launch()`` allocates nothing."""

    def __init__(self, solver, particle_counts, cell_counts, node_counts):
        self.solver = solver
        d = solver.device
        self.gp = int((int(np.max(particle_counts))+63)//64)
        self.gc = int((int(np.max(cell_counts))+63)//64)
        self.gn = int((int(np.max(node_counts))+63)//64)
        N = len(particle_counts)
        with wp.ScopedDevice(d):
            self.pp = wp.zeros((N*self.gp, 8), dtype=wp.float64)
            self.pc = wp.zeros((N*self.gc, 4), dtype=wp.float64)
            self.pn = wp.zeros((N*self.gn, 2), dtype=wp.float64)
            self.summary = wp.zeros((N, NDIAG), dtype=wp.float64)
            self.marker_x = wp.zeros(len(solver.marker_offsets_host)-1, dtype=wp.vec3d)
            self.marker_v = wp.zeros(len(solver.marker_offsets_host)-1, dtype=wp.vec3d)

    def launch(self):
        s = self.solver
        N = s.num_envs
        with wp.ScopedDevice(s.device):
            wp.launch(rc.cell_jbar, s.cells, [s.cell_node, s.cell_offsets, s.cell_particles, s.F, s.jbar])
            wp.launch(diag_particles, (N, self.gp), [s.x, s.v, s.F, s.mass, s.reference, s.gravity, s.volume,
                                                     s.dparams, s.particle_offset, self.gp, self.pp])
            wp.launch(diag_cells, (N, self.gc), [s.jbar, s.cell_node, s.cell_offsets, s.cell_env_offset, self.gc,
                                                 s.dparams, s.volume, self.pc])
            wp.launch(diag_nodes, (N, self.gn), [s.gm, s.new_v, s.node_offset, self.gn, self.pn])
            wp.launch(diag_combine, N, [self.pp, self.pc, self.pn, self.gp, self.gc, self.gn, s.particle_offset,
                                        s.cell_env_offset, self.summary])
            wp.launch(markers_kernel, len(self.marker_x), [s.x, s.v, s.marker_offsets, s.marker_particles,
                                                           self.marker_x, self.marker_v])

    def read(self):
        self.launch()
        rows = self.summary.numpy()
        return [dict(zip(DIAG_COLUMNS, map(float, r))) for r in rows]


# ----------------------------------------------------------------------------- solver

class TLMPMRubber:
    """Single-env reference solver (see the module docstring).

    ``scene``: :class:`rubber_scene.RubberStripScene`; ``material``: :class:`rubber_material.MaterialValues`;
    ``dt`` material time step; ``substeps``: substeps per control step (the group interpolation parameter is
    index / substeps). ``table_plane`` (unit normal, offset; inward normal) and ``fixture_boxes`` ([B, 2, 3]
    centres, half extents) enable contacts with friction ``material.friction_table`` (P2-4 validates them)."""
    solver_name = SOLVER_NAME
    status = STATUS
    num_envs = 1

    def __init__(self, scene, material, dt, *, substeps=1, gravity=(0.0, 0.0, -9.81), device="cpu", pic=0.0,
                 j_domain=(0.2, 5.0), contact_stiffness=0.1, contact_radius_scale=0.9, table_plane=None,
                 fixture_boxes=None):
        from .rubber_material import mass_per_particle
        if pic != 0.0:
            raise ValueError("tlmpm_rubber_v1 refuses pic != 0: rubber runs pure FLIP (ADR 0023 §2 E3, D4)")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not (isinstance(substeps, (int, np.integer)) and substeps >= 1):
            raise ValueError("substeps must be a positive integer")
        lo, hi = map(float, j_domain)
        if not 0 < lo < 1 < hi:
            raise ValueError("j_domain must satisfy 0 < lo < 1 < hi")
        wp.init()
        self.scene, self.material, self.device = scene, material, str(wp.get_device(device))
        self.dt, self.substeps, self.pic = dt, int(substeps), 0.0
        self.j_domain = (lo, hi)
        self.gravity = wp.vec3(*map(float, gravity))
        self.gravity_tuple = tuple(map(float, gravity))
        self.spacing, self.dx = float(scene.spacing), float(scene.dx)
        self.volume = float(scene.volume)
        self.radius = float(contact_radius_scale)*self.spacing*0.5
        self.contact_stiffness = float(contact_stiffness)
        self.damping_rate = 0.0
        self.n = int(scene.n)
        self.node_count = int(scene.node_count)
        self.cells = len(scene.cell_node)
        self.G = len(scene.groups)
        self.Gt = len(scene.traction_groups)
        self.mass_value = mass_per_particle(material.density, self.spacing)
        plane = np.zeros(4, np.float32) if table_plane is None else np.asarray(table_plane, np.float32)
        if table_plane is not None and (plane.shape != (4,) or not np.isclose(np.linalg.norm(plane[:3]), 1.0)):
            raise ValueError("table_plane needs a unit inward normal and an offset")
        self.use_plane = int(table_plane is not None)
        self.plane = wp.vec4(*plane.tolist())
        boxes = np.zeros((0, 2, 3), np.float32) if fixture_boxes is None else np.asarray(fixture_boxes, np.float32
                                                                                        ).reshape(-1, 2, 3)
        if np.any(boxes[:, 1] <= 0) or not np.isfinite(boxes).all():
            raise ValueError("fixture boxes need finite centres and positive half extents")
        self.boxes = len(boxes)
        self.marker_offsets_host = scene.marker_offsets
        kind = np.array([GROUP_AFFINE if g["kind"] == "affine" else GROUP_RIGID for g in scene.groups] or [0], np.int32)
        G1 = max(self.G, 1)
        tmass = np.array([float(material.density)*0 for _ in range(max(self.Gt, 1))], np.float64)
        for t in range(self.Gt):
            tmass[t] = float(np.count_nonzero(scene.particle_traction == t))*float(self.mass_value)
        with wp.ScopedDevice(self.device):
            self.reference = wp.array(scene.X, dtype=wp.vec3)
            self.x = wp.array(scene.X, dtype=wp.vec3)
            self.x_start = wp.zeros(self.n, dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=wp.vec3)
            self.mass = wp.array(np.full(self.n, self.mass_value, np.float32), dtype=float)
            self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)), dtype=wp.mat33)
            self.stress = wp.zeros(self.n, dtype=wp.mat33)
            self.impulse = wp.zeros(self.n, dtype=wp.vec3)
            self.nodes = wp.array(scene.nodes, dtype=int)
            self.weights = wp.array(scene.weights, dtype=float)
            self.gradients = wp.array(scene.gradients, dtype=wp.vec3)
            self.slot = wp.zeros((self.n, 8), dtype=int)               # S = 1: constant slot 0
            self.slot_code = wp.zeros(self.node_count, dtype=int)      # constant code 0 (no cutting)
            self.gm = wp.zeros(self.node_count, dtype=float)
            self.gp = wp.zeros(self.node_count, dtype=wp.vec3)
            self.gj = wp.zeros(self.node_count, dtype=wp.vec3)
            self.old = wp.zeros(self.node_count, dtype=wp.vec3)
            self.new_v = wp.zeros(self.node_count, dtype=wp.vec3)
            self.jbar = wp.ones(self.node_count, dtype=wp.float64)
            self.cell_node = wp.array(scene.cell_node, dtype=int)
            self.cell_offsets = wp.array(scene.cell_offsets, dtype=int)
            self.cell_particles = wp.array(scene.cell_particles, dtype=int)
            self.node_group = wp.array(scene.node_group, dtype=int)
            self.node_mask = wp.array(scene.node_mask, dtype=int)
            self.node_X = wp.array(scene.node_X, dtype=wp.vec3d)
            self.group_kind = wp.array(kind, dtype=int)
            self.group_active = wp.array(np.ones(G1, np.int32), dtype=int)
            self.group_origin = wp.array(np.array([g["origin"] for g in scene.groups] or [[0, 0, 0]], np.float64),
                                         dtype=wp.vec3d)
            ident_q = np.tile(np.array([0, 0, 0, 1], np.float64), (G1, 1))
            ident_A = np.tile(np.eye(3), (G1, 1, 1))
            self.group_q0 = wp.array(ident_q, dtype=wp.quatd)
            self.group_q1 = wp.array(ident_q, dtype=wp.quatd)
            self.group_b0 = wp.zeros(G1, dtype=wp.vec3d)
            self.group_b1 = wp.zeros(G1, dtype=wp.vec3d)
            self.group_A0 = wp.array(ident_A, dtype=wp.mat33d)
            self.group_A1 = wp.array(ident_A, dtype=wp.mat33d)
            self.reaction = wp.zeros(G1, dtype=wp.spatial_vectord)
            self.work = wp.zeros(1, dtype=wp.float64)
            self.particle_traction = wp.array(scene.particle_traction, dtype=int)
            self.traction_force = wp.zeros(max(self.Gt, 1), dtype=wp.vec3)
            self.traction_mass = wp.array(tmass.astype(np.float32), dtype=float)
            self.box_c = wp.array(boxes[:, 0].copy() if len(boxes) else np.zeros((1, 3), np.float32), dtype=wp.vec3)
            self.box_h = wp.array(boxes[:, 1].copy() if len(boxes) else np.ones((1, 3), np.float32), dtype=wp.vec3)
            self.index = wp.zeros(1, dtype=int)
            self.violations = wp.zeros(1, dtype=int)
            self.dparams = wp.array(np.array([[material.mu0, material.kappa]], np.float64).astype(np.float32),
                                    dtype=float)
            self.particle_offset = wp.array(np.array([0, self.n], np.int32), dtype=int)
            self.cell_env_offset = wp.array(np.array([0, self.cells], np.int32), dtype=int)
            self.node_offset = wp.array(np.array([0, self.node_count], np.int32), dtype=int)
            self.marker_offsets = wp.array(scene.marker_offsets, dtype=int)
            self.marker_particles = wp.array(scene.marker_particles, dtype=int)
        self.diagnostics = SegmentedRubberDiagnostics(self, [self.n], [self.cells], [self.node_count])
        for module in (v5.__name__, "robotic_cutting_sim.tlmpm_v6", rc.__name__, __name__):
            wp.load_module(module=module, device=self.device)

    # ------------------------------------------------------------------ inputs

    def set_initializer_damping(self, rate, *, initializer=False):
        """Mass-proportional damping impulse -rate m v dt: ONLY for the labelled static initializer (ADR §7)."""
        rate = float(rate)
        if rate != 0.0 and not initializer:
            raise ValueError("artificial damping is refused in dynamics; only the static initializer may set it")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("damping rate must be finite and >= 0")
        self.damping_rate = rate

    def set_group_motion(self, g, *, rotation=None, translation=None, affine=None, active=None):
        """Set the END of group ``g``'s motion in the current control step (call after ``begin_control_step``,
        which makes every group hold: start = end = previous end). The map is x = R X + b (rigid) or
        x = A X + b (affine), relative to the reference configuration."""
        if not 0 <= g < self.G:
            raise IndexError(f"group {g} not in [0, {self.G})")
        kind = self.scene.groups[g]["kind"]
        if rotation is not None:
            if kind != "rigid":
                raise ValueError("rotation is for rigid groups")
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(np.asarray(rotation, float)).as_quat()
            prev = self.group_q1.numpy()[g]
            if np.dot(prev, q) < 0:
                q = -q
            self._set_row(self.group_q1, g, q)
        if affine is not None:
            if kind != "affine":
                raise ValueError("affine maps are for affine groups")
            self._set_row(self.group_A1, g, np.asarray(affine, float))
        if translation is not None:
            self._set_row(self.group_b1, g, np.asarray(translation, float))
        if active is not None:
            self._set_row(self.group_active, g, int(bool(active)))

    def _set_row(self, array, row, value):
        host = array.numpy()
        host[row] = value
        array.assign(host)

    def set_traction(self, t, force):
        if not 0 <= t < self.Gt:
            raise IndexError(f"traction group {t} not in [0, {self.Gt})")
        self._set_row(self.traction_force, t, np.asarray(force, np.float32))

    # ------------------------------------------------------------------ substep

    def step(self):
        n, nn, dt = self.n, self.node_count, self.dt
        with wp.ScopedDevice(self.device):
            wp.copy(self.x_start, self.x)
            wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                                 self.jbar])
            wp.launch(_stress, n, [self.F, self.nodes, self.jbar, self.stress, self.material.mu0,
                                   self.material.kappa, wp.float64(self.j_domain[0]), wp.float64(self.j_domain[1]),
                                   self.violations])
            wp.launch(_loads, n, [self.x, self.v, self.mass, self.impulse, self.particle_traction, self.traction_force,
                                  self.traction_mass, dt, self.damping_rate, self.radius, self.contact_stiffness,
                                  self.material.friction_table, self.use_plane, self.plane, self.boxes, self.box_c,
                                  self.box_h, self.work])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse, self.nodes,
                                        self.weights, self.gradients, self.gm, self.gp, self.gj, 1, self.volume, dt,
                                        self.gravity])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            self._launch_bc(1)
            wp.launch(v5._g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights, self.old, self.new_v, 1, dt,
                                   self.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes, self.weights, self.gp, 1])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            self._launch_bc(0)
            wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients, self.new_v, 1, dt])
            wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights, self.new_v, 1,
                                           dt])
            wp.launch(v5._advance, 1, [self.index])

    def _launch_bc(self, record):
        if self.G == 0:
            return
        wp.launch(_bc, self.node_count, [self.node_group, self.node_mask, self.node_X, self.gm, self.new_v,
                                         self.index, self.substeps, wp.float64(self.dt), self.group_kind,
                                         self.group_active, self.group_q0, self.group_q1, self.group_b0,
                                         self.group_b1, self.group_A0, self.group_A1, self.group_origin,
                                         self.reaction, self.work, record], device=self.device)

    # ------------------------------------------------------------------ control-step bookkeeping

    def begin_control_step(self):
        """Zero the substep index and the per-control-step reaction accumulators and make every group hold
        (start = end); new motions are then written with ``set_group_motion`` (no allocation)."""
        self.index.zero_()
        self.reaction.zero_()
        for a0, a1 in ((self.group_q0, self.group_q1), (self.group_b0, self.group_b1), (self.group_A0, self.group_A1)):
            wp.copy(a0, a1)

    def end_control_step(self, control_dt):
        """Group wrenches [G, 6]: force (N) and torque (N m) the material exerts on each group, averaged over the
        control step (= -sum of BC impulses / control_dt)."""
        return -self.reaction.numpy()[:self.G]/float(control_dt)

    def reset(self):
        """Fresh reference state (x = X, v = 0, F = I, zero grid/stress, identity group motions, counters 0)."""
        with wp.ScopedDevice(self.device):
            wp.copy(self.x, self.reference)
            for a in (self.v, self.x_start, self.stress, self.impulse, self.gm, self.gp, self.gj, self.old,
                      self.new_v, self.reaction, self.work, self.index, self.violations, self.traction_force,
                      self.group_b0, self.group_b1):
                a.zero_()
            self.F.assign(np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)))
            self.jbar.fill_(1.0)
            G1 = max(self.G, 1)
            ident_q = np.tile(np.array([0, 0, 0, 1], np.float64), (G1, 1))
            self.group_q0.assign(ident_q)
            self.group_q1.assign(ident_q)
            self.group_A0.assign(np.tile(np.eye(3), (G1, 1, 1)))
            self.group_A1.assign(np.tile(np.eye(3), (G1, 1, 1)))
            self.group_active.fill_(1)
        self.damping_rate = 0.0

    # ------------------------------------------------------------------ queries

    def nodal_internal_force(self):
        """Diagnostic (does not advance): J-bar + stress + frozen P2G with zero gravity, impulse and velocity
        contributions removed; returns f_I = gj / dt [nodes, 3] (float64) = -sum_p V P_p grad N_I."""
        with wp.ScopedDevice(self.device):
            wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                                 self.jbar])
            dummy = wp.zeros(1, dtype=int)
            wp.launch(_stress, self.n, [self.F, self.nodes, self.jbar, self.stress, self.material.mu0,
                                        self.material.kappa, wp.float64(self.j_domain[0]),
                                        wp.float64(self.j_domain[1]), dummy])
            gm = wp.zeros(self.node_count, dtype=float)
            gp = wp.zeros(self.node_count, dtype=wp.vec3)
            gj = wp.zeros(self.node_count, dtype=wp.vec3)
            zero_v = wp.zeros(self.n, dtype=wp.vec3)
            wp.launch(v5._p2g, (self.n, 8), [zero_v, self.mass, self.slot, self.stress, zero_v, self.nodes,
                                             self.weights, self.gradients, gm, gp, gj, 1, self.volume, self.dt,
                                             wp.vec3(0.0)])
            return gj.numpy().astype(np.float64)/self.dt

    def jbar_cells(self):
        """Current J-bar per occupied cell (float64), recomputed from F."""
        wp.launch(rc.cell_jbar, self.cells, [self.cell_node, self.cell_offsets, self.cell_particles, self.F,
                                             self.jbar], device=self.device)
        return self.jbar.numpy()[self.scene.cell_node]

    def memory_bytes(self):
        return int(sum(v.capacity for v in vars(self).values() if isinstance(v, wp.array))
                   + sum(v.capacity for v in vars(self.diagnostics).values() if isinstance(v, wp.array)))

    def device_arrays(self):
        return {k: v for k, v in vars(self).items() if isinstance(v, wp.array)}

    def solver_receipt(self):
        m = self.material
        return {"solver": SOLVER_NAME, "status": STATUS, "adr": ADR, "model": rc.MODEL_ID,
                "model_version": rc.MODEL_VERSION, "pic": 0.0, "transfer": "pure_flip_musl_grid_positions",
                "dt_s": self.dt, "substeps_per_control_step": self.substeps, "dx_m": self.dx,
                "particle_spacing_m": self.spacing, "particles": self.n, "nodes": self.node_count,
                "cells": self.cells, "groups": self.G, "traction_groups": self.Gt, "gravity": list(self.gravity_tuple),
                "j_domain": list(self.j_domain), "material": m.to_dict(),
                "cfl_actual": self.dt*m.c_p()/self.dx, "contact": {"table_plane": bool(self.use_plane),
                "fixture_boxes": self.boxes, "stiffness_fraction": self.contact_stiffness, "radius_m": self.radius,
                "status": "implemented, validated in P2-4 (RUBBER-CONT)"},
                "damping_rate": self.damping_rate, "frozen_kernels": list(FROZEN_KERNELS),
                "source_sha256": source_sha256(), "device": self.device, "warp_version": wp.config.version}
