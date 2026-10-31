"""Physics-neutral recorders for the WP-12 Phase 4 root-cause verification (methods review §5, rows V1-V4).

Both recorders only read solver arrays; they never write x, v, F, Fpi, code, progress, impulse, reaction or any
other solver array, and they do not change ``tlmpm_v5.py``. They are opt-in (a run that does not construct them
launches nothing new) and CUDA-graph capturable (every counter lives on the device). Neutrality is tested bitwise
on CPU (``tests/test_root_cause_recorders.py``). Their output is oracle / diagnostic data, never an observation.

``flip_drift_v1`` (``FlipDriftRecorder``; call ``after_step()`` after ``step()``)
    After ``step()`` the arrays ``v`` (the FLIP-blended particle velocity that moved x in ``_g2p``), ``new_v``
    (the MUSL grid velocity that updated F in ``_deformation``) and ``slot`` are still those of the step (the side
    assignment kernels that follow touch neither). Per particle it accumulates, in float64,

        drift_p  += dt (v_p - sum_I w_pI v_I^MUSL)      (the FLIP null-space part that F never sees)
        ugrid_p  += dt sum_I w_pI v_I^MUSL              (x~_p = X_p + ugrid_p, the F-consistent position)

    so x_p = X_p + ugrid_p + drift_p up to float32 rounding of x (methods review §1.1). It also records, per cut
    digit k, the global substep at which the digit first became nonzero and the particle position at that instant
    (the position the side assignment saw).

``tool_contact_split_v1`` (``ToolContactSplitRecorder``; call ``before_step()`` before ``step()``)
    Recomputes the blade branch of ``tlmpm_v5._rigid_contacts`` from the state the solver will read (same
    ``_profile_sdf`` / ``_contact_impulse`` functions, same operations; the progress reset of ``_start_cut`` is
    emulated) and accumulates the blade reaction (-impulse, N s) per substep, split into contacts inside the armed
    cutting zone (|local z| <= zone, -zone_rear <= local x <= 2r, |local y| <= half width, cut armed) and all other
    blade contacts, each projected on the edge-frame advance d, edge tangent e and blade normal n. Also: contact
    counts, the maximum penetration of each class, and the in-zone impulse released by the cutting rule
    ((1 - fraction)(push d + drag slice_dir)). The handle obstacle, the board/table and particle-particle contact
    are not included (the solver's ``reaction`` also contains the handle; ``box_reactions`` holds it).

``tensile_energy_max_v1`` (``TensileEnergyRecorder``; ``after_step()``)
    Per-particle history maximum of the Hencky energy density of the trial elastic state (F Fpi after the J2 return
    tlmpm_v5._stress applies), tensile part (Amor split mu |dev e|^2 + K/2 <tr e>_+^2) and total, for the V2.2
    Griffith check. A diagnostic of the constitutive state; not a fracture model.
"""
import numpy as np
import warp as wp

from . import tlmpm_v5 as v5

FLIP_DRIFT_VERSION = "flip_drift_v1"
TENSILE_ENERGY_VERSION = "tensile_energy_max_v1"
TOOL_CONTACT_SPLIT_VERSION = "tool_contact_split_v1"
SPLIT_COLUMNS = ("in_d", "in_e", "in_n", "out_d", "out_e", "out_n", "in_count", "out_count",
                 "in_max_penetration_m", "out_max_penetration_m", "released_d", "released_e", "released_n",
                 "in_raw_d", "armed")
_NSPLIT = len(SPLIT_COLUMNS)


class RecorderOverflowError(RuntimeError):
    """More substeps than the recorder's fixed history capacity."""


# ----------------------------------------------------------------------------- flip_drift_v1

@wp.func
def _rc_digit(code: int, cut: int):
    return (code >> (2*cut)) & 3


@wp.kernel
def _record_flip_drift(v: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3), code: wp.array(dtype=int),
                       slot: wp.array2d(dtype=int), nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
                       new_v: wp.array(dtype=wp.vec3), slots: int, dt: float, max_cuts: int,
                       substep: wp.array(dtype=int), drift: wp.array(dtype=wp.vec3d), ugrid: wp.array(dtype=wp.vec3d),
                       assign_substep: wp.array2d(dtype=int), assign_x: wp.array2d(dtype=wp.vec3)):
    p = wp.tid()
    vg = wp.vec3(0.0)
    for corner in range(8):
        s = slot[p, corner]
        if s >= 0:
            vg += weights[p, corner]*new_v[nodes[p, corner]*slots+s]
    d = dt*(v[p]-vg)
    g = dt*vg
    drift[p] = drift[p]+wp.vec3d(wp.float64(d[0]), wp.float64(d[1]), wp.float64(d[2]))
    ugrid[p] = ugrid[p]+wp.vec3d(wp.float64(g[0]), wp.float64(g[1]), wp.float64(g[2]))
    c = code[p]
    for k in range(max_cuts):
        if _rc_digit(c, k) != 0 and assign_substep[p, k] < 0:
            assign_substep[p, k] = substep[0]
            assign_x[p, k] = x[p]


@wp.kernel
def _rc_tick(substep: wp.array(dtype=int)):
    substep[0] = substep[0]+1


class FlipDriftRecorder:
    """flip_drift_v1 (see the module docstring). ``after_step()`` after every ``bed.step()``."""

    def __init__(self, bed):
        self.bed, self.device = bed, bed.device
        with wp.ScopedDevice(self.device):
            self.drift = wp.zeros(bed.n, dtype=wp.vec3d)
            self.ugrid = wp.zeros(bed.n, dtype=wp.vec3d)
            self.assign_substep = wp.full((bed.n, bed.max_cuts), -1, dtype=int)
            self.assign_x = wp.zeros((bed.n, bed.max_cuts), dtype=wp.vec3)
            self.substep = wp.zeros(1, dtype=int)

    def after_step(self):
        bed = self.bed
        with wp.ScopedDevice(self.device):
            wp.launch(_record_flip_drift, bed.n, [bed.v, bed.x, bed.code, bed.slot, bed.nodes, bed.weights,
                bed.new_v, bed.slots_per_node, float(bed.config.dt), bed.max_cuts, self.substep, self.drift,
                self.ugrid, self.assign_substep, self.assign_x])
            wp.launch(_rc_tick, 1, [self.substep])

    def arrays(self):
        return {"drift_m": self.drift.numpy(), "ugrid_m": self.ugrid.numpy(),
                "assign_substep": self.assign_substep.numpy(), "assign_position_m": self.assign_x.numpy()}

    def metadata(self):
        return {"version": FLIP_DRIFT_VERSION, "substeps": int(self.substep.numpy()[0]), "particles": int(self.bed.n),
                "solver_module": type(self.bed).__module__,
                "definition": "drift_p = sum dt (v_p - sum_I w_pI v_I^MUSL); ugrid_p = sum dt sum_I w_pI v_I^MUSL "
                              "(float64 accumulation of float32 terms); assign_substep[p, k] = recorder substep "
                              "(0-based) after which digit k of p was first nonzero; assign_position = x then",
                "scope": "read-only diagnostic (oracle), physics-neutral"}


# ----------------------------------------------------------------------------- tensile_energy_max_v1

@wp.kernel
def _record_tensile_energy(F: wp.array(dtype=wp.mat33), Fpi: wp.array(dtype=wp.mat33), lam: float, mu: float,
                           yield_stress: float, psi_max: wp.array(dtype=float), psi_total_max: wp.array(dtype=float)):
    """Trial elastic state the next tlmpm_v5._stress sees (same SVD, clamp and J2 return; nothing is written back):
    psi+ = mu |dev e|^2 + K/2 <tr e>_+^2 (Amor split: deviatoric + dilatational part), psi = mu |dev e|^2 + K/2 tr^2."""
    p = wp.tid()
    fe = F[p]*Fpi[p]
    u, sigma, v = wp.svd3(fe)
    logs = wp.vec3(wp.log(wp.max(sigma[0], 0.05)), wp.log(wp.max(sigma[1], 0.05)), wp.log(wp.max(sigma[2], 0.05)))
    tr = logs[0]+logs[1]+logs[2]
    dev = logs-wp.vec3(tr/3.0)
    norm = wp.length(dev)
    limit = wp.sqrt(2.0/3.0)*yield_stress/(2.0*mu)
    if norm > limit:
        dev = dev*(limit/norm)
    bulk = lam+2.0*mu/3.0
    psi_dev = mu*wp.dot(dev, dev)
    pos = wp.max(tr, 0.0)
    psi_max[p] = wp.max(psi_max[p], psi_dev+0.5*bulk*pos*pos)
    psi_total_max[p] = wp.max(psi_total_max[p], psi_dev+0.5*bulk*tr*tr)


class TensileEnergyRecorder:
    """tensile_energy_max_v1: per-particle history maximum of the tensile (Amor-split) and total Hencky energy density
    (J/m^3) of the trial elastic state after every step. Read-only; ``after_step()`` after ``bed.step()``."""
    version = "tensile_energy_max_v1"

    def __init__(self, bed):
        self.bed, self.device = bed, bed.device
        c = bed.config
        self.lam = float(c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson)))
        self.mu = float(c.young/(2*(1+c.poisson)))
        with wp.ScopedDevice(self.device):
            self.psi_max = wp.zeros(bed.n, dtype=float)
            self.psi_total_max = wp.zeros(bed.n, dtype=float)

    def after_step(self):
        bed = self.bed
        with wp.ScopedDevice(self.device):
            wp.launch(_record_tensile_energy, bed.n, [bed.F, bed.Fpi, self.lam, self.mu, float(bed.config.yield_stress),
                                                       self.psi_max, self.psi_total_max])

    def arrays(self):
        return {"psi_tensile_max_j_m3": self.psi_max.numpy(), "psi_total_max_j_m3": self.psi_total_max.numpy()}

    def metadata(self):
        return {"version": self.version, "definition": "max over substeps of psi+ = mu |dev e|^2 + K/2 <tr e>_+^2 and of "
                "psi = mu |dev e|^2 + K/2 tr^2, e = Hencky strain of F Fpi after the J2 return of tlmpm_v5._stress",
                "scope": "read-only diagnostic (oracle), physics-neutral"}


# ----------------------------------------------------------------------------- tool_contact_split_v1

@wp.kernel
def _record_tool_split(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                       progress: wp.array(dtype=float), previous: wp.array(dtype=int),
                       poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                       cuts: wp.array(dtype=int), index: wp.array(dtype=int), radius: float, volume: float,
                       dt: float, stiffness: float, friction: float, gc: float, sharpness: float,
                       profile: wp.array(dtype=wp.vec2), profile_count: int, blade_width: float, tool: int,
                       slice_efficiency: float, zone: float, zone_rear: float, acc: wp.array(dtype=float)):
    p = wp.tid()
    if tool == 0:
        return
    xp, vp, mp = x[p], v[p], mass[p]
    i = index[0]
    pose = poses[i]
    local = wp.transform_point(wp.transform_inverse(pose), xp)
    hit = v5._profile_sdf(local, profile, profile_count, blade_width)
    normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
    gap = hit[3]-radius
    twist = twists[i]
    linear = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    lever = xp-wp.transform_get_translation(pose)-normal*radius
    relative = vp-linear-wp.cross(angular, lever)
    ji = v5._contact_impulse(gap, normal, relative, mp, dt, stiffness, friction)
    if wp.length_sq(ji) <= 0.0:
        return
    advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    blade_normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    uc = wp.dot(-relative, advance)
    active = cuts[i]
    prog = progress[p]
    if active >= 0 and active != previous[0]:   # tlmpm_v5._start_cut runs before _rigid_contacts
        prog = 0.0
    in_zone = (local[0] >= -zone_rear and local[0] <= 2.0*radius and wp.abs(local[2]) <= zone
               and wp.abs(local[1]) <= blade_width)
    released = wp.vec3(0.0)
    raw = ji
    if active >= 0 and in_zone:
        us = wp.dot(-relative, along)
        slice_dir = along*wp.sign(us)
        push = wp.max(wp.dot(ji, advance), 0.0)
        drag = wp.max(wp.dot(ji, slice_dir), 0.0)
        un = wp.dot(-relative, blade_normal)
        in_plane = wp.max(wp.length_sq(relative)-un*un, 0.0)
        alignment = in_plane/(in_plane+un*un+1.0e-20)
        dw = (push*wp.max(uc, 0.0)+slice_efficiency*drag*wp.abs(us))*alignment
        if dw > 1.0e-20:
            threshold = gc*wp.pow(volume, 2.0/3.0)*(1.0-sharpness)/sharpness
            remaining = wp.max((1.0-prog)*threshold, 0.0)
            fraction = wp.min(remaining/dw, 1.0)
            released = (1.0-fraction)*(push*advance+drag*slice_dir)
            ji = ji-released
    if wp.length_sq(ji) <= 0.0:
        return
    r = -ji
    pen = wp.max(-gap, 0.0)
    if active >= 0 and in_zone:
        wp.atomic_add(acc, 0, wp.dot(r, advance))
        wp.atomic_add(acc, 1, wp.dot(r, along))
        wp.atomic_add(acc, 2, wp.dot(r, blade_normal))
        wp.atomic_add(acc, 6, 1.0)
        wp.atomic_max(acc, 8, pen)
        wp.atomic_add(acc, 10, wp.dot(released, advance))
        wp.atomic_add(acc, 11, wp.dot(released, along))
        wp.atomic_add(acc, 12, wp.dot(released, blade_normal))
        wp.atomic_add(acc, 13, -wp.dot(raw, advance))
    else:
        wp.atomic_add(acc, 3, wp.dot(r, advance))
        wp.atomic_add(acc, 4, wp.dot(r, along))
        wp.atomic_add(acc, 5, wp.dot(r, blade_normal))
        wp.atomic_add(acc, 7, 1.0)
        wp.atomic_max(acc, 9, pen)


@wp.kernel
def _commit_tool_split(acc: wp.array(dtype=float), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                       history: wp.array2d(dtype=float), substep: wp.array(dtype=int), capacity: int,
                       overflow: wp.array(dtype=int), columns: int):
    k = substep[0]
    if k < capacity:
        for j in range(columns-1):
            history[k, j] = acc[j]
        armed = float(0.0)
        if cuts[index[0]] >= 0:
            armed = 1.0
        history[k, columns-1] = armed
    else:
        overflow[0] = overflow[0]+1
    for j in range(columns):
        acc[j] = 0.0
    substep[0] = k+1


def blade_contact_parameters(bed):
    """(k, zone, zone_rear) the solver passes to _rigid_contacts (counterfactual-aware)."""
    if hasattr(bed, "blade_contact_parameters"):
        return bed.blade_contact_parameters()
    return float(bed.config.contact_stiffness), float(bed.spacing), float(2*bed.config.dx)


class ToolContactSplitRecorder:
    """tool_contact_split_v1 (see the module docstring). ``before_step()`` before every ``bed.step()``."""

    def __init__(self, bed, capacity):
        if bed.cutting_law not in (4, 5, 8):
            raise ValueError("tool_contact_split_v1 reproduces the law-4/5/8 contact path only")
        self.bed, self.device, self.capacity = bed, bed.device, int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        with wp.ScopedDevice(self.device):
            self.acc = wp.zeros(_NSPLIT, dtype=float)
            self.history = wp.zeros((self.capacity, _NSPLIT), dtype=float)
            self.substep = wp.zeros(1, dtype=int)
            self.overflow = wp.zeros(1, dtype=int)

    def before_step(self):
        bed, c = self.bed, self.bed.config
        k, zone, zone_rear = blade_contact_parameters(bed)
        with wp.ScopedDevice(self.device):
            wp.launch(_record_tool_split, bed.n, [bed.x, bed.v, bed.mass, bed.progress, bed.previous_cut, bed.poses,
                bed.twists, bed.cuts, bed.index, bed.radius, bed.volume, c.dt, k, c.contact_friction,
                c.cutting_energy, c.sharpness, bed.profile, len(bed.blade_profile_xz), c.blade_half_width,
                int(c.tool), bed.slice_efficiency, zone, zone_rear, self.acc])
            wp.launch(_commit_tool_split, 1, [self.acc, bed.cuts, bed.index, self.history, self.substep,
                self.capacity, self.overflow, _NSPLIT])

    def check(self):
        overflow = int(self.overflow.numpy()[0])
        if overflow:
            raise RecorderOverflowError(f"tool_contact_split_v1: {overflow} substeps beyond capacity {self.capacity}")
        return {"substeps": int(self.substep.numpy()[0]), "overflow": overflow}

    def table(self):
        rows = min(int(self.substep.numpy()[0]), self.capacity)
        return self.history.numpy()[:rows]

    def metadata(self):
        k, zone, rear = blade_contact_parameters(self.bed)
        return {"version": TOOL_CONTACT_SPLIT_VERSION, "columns": list(SPLIT_COLUMNS), "capacity": self.capacity,
                **self.check(), "blade_stiffness": k, "zone_m": zone, "zone_rear_m": rear,
                "units": "reaction impulse on the blade per substep (N s), projected on the edge frame d, e, n; "
                         "counts; penetration (m); released = impulse removed by the cutting rule; in_raw_d = "
                         "advance reaction before the release; armed = cut digit >= 0 at the substep",
                "scope": "read-only recomputation of the tlmpm_v5 blade contact; excludes handle, board, particles"}


def split_per_control_step(table, substeps):
    """(steps, columns) per control step: impulses summed, counts summed, penetrations max, armed max."""
    table = np.asarray(table, float)
    steps = len(table)//substeps
    t = table[:steps*substeps].reshape(steps, substeps, _NSPLIT)
    out = t.sum(1)
    for name in ("in_max_penetration_m", "out_max_penetration_m", "armed"):
        j = SPLIT_COLUMNS.index(name)
        out[:, j] = t[:, :, j].max(1)
    return out
