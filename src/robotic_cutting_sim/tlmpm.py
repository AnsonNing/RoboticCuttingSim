"""Experimental total-Lagrangian MPM with online persistent cutting history.

Independent implementation of logical-grid separation (BladeMaster, 2026), not
the authors' implementation. Linear reference-grid interpolation, MUSL, Hencky
J2 plasticity, PIC/FLIP, explicit sphere contact and a finite triangular blade.
Two cut digits are retained. There is no deletion, spawn, prescribed crack, or
fragment-specific body motion. Parameters are provisional, not calibrated.
"""
import numpy as np
import warp as wp

from .cut_topology import compatibility
from .material_config import TLMPMConfig


@wp.func
def _conflict(a: int, b: int):
    answer = False
    for k in range(2):
        da = (a // int(wp.pow(3.0, float(k)))) % 3
        db = (b // int(wp.pow(3.0, float(k)))) % 3
        if (da == 1 and db == 2) or (da == 2 and db == 1):
            answer = True
    return answer


@wp.func
def _outer(a: wp.vec3, b: wp.vec3):
    return wp.mat33(a[0]*b[0], a[0]*b[1], a[0]*b[2],
                    a[1]*b[0], a[1]*b[1], a[1]*b[2],
                    a[2]*b[0], a[2]*b[1], a[2]*b[2])


@wp.kernel
def _stress(F: wp.array(dtype=wp.mat33), Fpi: wp.array(dtype=wp.mat33),
            stress: wp.array(dtype=wp.mat33), plastic: wp.array(dtype=float),
            lam: float, mu: float, yield_stress: float):
    p = wp.tid()
    fe = F[p]*Fpi[p]
    u, sigma, v = wp.svd3(fe)
    # Clamp only for finite evaluation; the host separately rejects inverted or
    # extreme total deformation, rather than silently accepting this safeguard.
    logs = wp.vec3(wp.log(wp.max(sigma[0], 0.05)), wp.log(wp.max(sigma[1], 0.05)),
                   wp.log(wp.max(sigma[2], 0.05)))
    tr = logs[0]+logs[1]+logs[2]
    dev = logs-wp.vec3(tr/3.0)
    norm = wp.length(dev)
    limit = wp.sqrt(2.0/3.0)*yield_stress/(2.0*mu)
    if norm > limit:
        delta = norm-limit
        logs = logs-delta*dev/norm
        corrected = u*wp.diag(wp.vec3(wp.exp(logs[0]), wp.exp(logs[1]), wp.exp(logs[2])))*wp.transpose(v)
        Fpi[p] = wp.inverse(F[p])*corrected
        plastic[p] = plastic[p]+delta
    tau = u*wp.diag(2.0*mu*logs+wp.vec3(lam*tr))*wp.transpose(u)
    stress[p] = tau*wp.transpose(wp.inverse(F[p]))


@wp.func
def _segment(p: wp.vec2, a: wp.vec2, b: wp.vec2):
    e = b-a
    return a+wp.clamp(wp.dot(p-a, e)/wp.dot(e, e), 0.0, 1.0)*e


@wp.func
def _wedge_sdf(p: wp.vec3, length: float, height: float, half_width: float):
    """Exact distance/normal of an extruded triangular x/z cross-section."""
    a = wp.vec2(0.0, 0.0)
    b = wp.vec2(-length, height)
    c = wp.vec2(-length, -0.0004)
    x = wp.vec2(p[0], p[2])
    best = float(1.0e10)
    closest = wp.vec2(0.0)
    outward = wp.vec2(0.0)
    inside = True
    for edge in range(3):
        v0 = a
        v1 = b
        if edge == 1:
            v0 = b
            v1 = c
        elif edge == 2:
            v0 = c
            v1 = a
        e = v1-v0
        if e[0]*(x[1]-v0[1])-e[1]*(x[0]-v0[0]) < 0.0:
            inside = False
        q = _segment(x, v0, v1)
        distance = wp.length_sq(x-q)
        if distance < best:
            best = distance
            closest = q
            outward = wp.normalize(wp.vec2(e[1], -e[0]))
    distance = wp.sqrt(best)
    n2 = outward
    sd2 = -distance
    if not inside:
        sd2 = distance
        if distance > 1.0e-10:
            n2 = (x-closest)/distance
    sy = wp.abs(p[1])-half_width
    ny = wp.sign(p[1])
    sdf = wp.sqrt(wp.max(sd2, 0.0)**2.0+wp.max(sy, 0.0)**2.0)+wp.min(wp.max(sd2, sy), 0.0)
    n = wp.vec3(n2[0], 0.0, n2[1])
    if sy > 0.0 and sd2 > 0.0:
        n = wp.normalize(wp.vec3(n2[0]*sd2, ny*sy, n2[1]*sd2))
    elif sy > sd2:
        n = wp.vec3(0.0, ny, 0.0)
    return wp.vec4(n[0], n[1], n[2], sdf)


@wp.func
def _contact_impulse(gap: float, normal: wp.vec3, relative_v: wp.vec3,
                     mass: float, dt: float, stiffness: float, friction: float):
    vn = wp.dot(relative_v, normal)
    predicted_gap = gap+dt*wp.min(vn, 0.0)
    jn = stiffness*mass/dt*wp.max(-predicted_gap, 0.0)
    tangent = relative_v-vn*normal
    speed = wp.length(tangent)
    jt = wp.vec3(0.0)
    if speed > 1.0e-10:
        jt = -wp.min(mass*speed, friction*jn)*tangent/speed
    return jn*normal+jt


@wp.func
def _box_sdf(p: wp.vec3, half: wp.vec3):
    q = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2]))-half
    outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    distance = wp.length(outside)
    sdf = distance+wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    normal = wp.vec3(0.0)
    if distance > 1.0e-10:
        normal = wp.vec3(wp.sign(p[0])*outside[0], wp.sign(p[1])*outside[1], wp.sign(p[2])*outside[2])/distance
    elif q[0] >= q[1] and q[0] >= q[2]:
        normal = wp.vec3(wp.sign(p[0]), 0.0, 0.0)
    elif q[1] >= q[2]:
        normal = wp.vec3(0.0, wp.sign(p[1]), 0.0)
    else:
        normal = wp.vec3(0.0, 0.0, wp.sign(p[2]))
    return wp.vec4(normal[0], normal[1], normal[2], sdf)


@wp.kernel
def _rigid_contacts(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                    mass: wp.array(dtype=float), impulse: wp.array(dtype=wp.vec3),
                    progress: wp.array(dtype=float), contact_work: wp.array(dtype=float),
                    poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                    cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                    reaction: wp.array(dtype=wp.spatial_vector), stats: wp.array(dtype=float),
                    radius: float, volume: float, dt: float, stiffness: float,
                    friction: float, gc: float, sharpness: float, blade_length: float,
                    blade_height: float, blade_width: float, floor: int, walls: int, tool: int):
    p = wp.tid()
    impulse[p] = wp.vec3(0.0)
    xp, vp, mp = x[p], v[p], mass[p]
    if floor != 0:
        # Finite tray boxes, so a detached chip can clear a wall and land on the
        # surrounding table. Half-space wall tests would spuriously accelerate
        # an already-outside chip when it falls below the wall's top edge.
        for side in range(6):
            center = wp.vec3(0.0, 0.0, -0.005)
            half = wp.vec3(0.068, 0.048, 0.005)
            if side == 1:
                center = wp.vec3(-0.064, 0.0, 0.0075)
                half = wp.vec3(0.004, 0.044, 0.0075)
            elif side == 2:
                center = wp.vec3(0.064, 0.0, 0.0075)
                half = wp.vec3(0.004, 0.044, 0.0075)
            elif side == 3:
                center = wp.vec3(0.0, -0.044, 0.0075)
                half = wp.vec3(0.060, 0.004, 0.0075)
            elif side == 4:
                center = wp.vec3(0.0, 0.044, 0.0075)
                half = wp.vec3(0.060, 0.004, 0.0075)
            hit = _box_sdf(xp-center, half)
            normal = wp.vec3(hit[0], hit[1], hit[2])
            gap = hit[3]-radius
            if side == 5:
                normal = wp.vec3(0.0, 0.0, 1.0)
                gap = xp[2]+0.010-radius  # support table's top z=-10mm
            if side == 0 or side == 5 or walls != 0:
                impulse[p] = impulse[p]+_contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
                wp.atomic_max(stats, 0, wp.max(-gap, 0.0))
    if tool != 0:
        i = index[0]
        pose = poses[i]
        local = wp.transform_point(wp.transform_inverse(pose), xp)
        hit = _wedge_sdf(local, blade_length, blade_height, blade_width)
        normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
        gap = hit[3]-radius
        twist = twists[i]
        # This module's twist convention is [linear, angular], unlike wp.spatial.
        linear = wp.vec3(twist[0], twist[1], twist[2])
        angular = wp.vec3(twist[3], twist[4], twist[5])
        lever = xp-wp.transform_get_translation(pose)-normal*radius
        relative = vp-linear-wp.cross(angular, lever)
        ji = _contact_impulse(gap, normal, relative, mp, dt, stiffness, friction)
        advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
        uc = wp.dot(-relative, advance)
        if cuts[i] >= 0 and local[0] >= 0.0 and local[0] <= 2.0*radius and uc > 1.0e-5:
            dw = wp.max(wp.dot(ji, advance), 0.0)*uc
            threshold = gc*wp.pow(volume, 2.0/3.0)*(1.0-sharpness)/sharpness
            remaining = wp.max((1.0-progress[p])*threshold, 0.0)
            fraction = float(1.0)
            if dw > 1.0e-20:
                fraction = wp.min(remaining/dw, 1.0)
            contact_work[p] = contact_work[p]+fraction*dw
            progress[p] = wp.min(progress[p]+fraction*dw/wp.max(threshold, 1.0e-12), 1.0)
            ji = ji-(1.0-fraction)*wp.max(wp.dot(ji, advance), 0.0)*advance
        impulse[p] = impulse[p]+ji
        torque = wp.cross(lever, -ji)
        wp.atomic_add(reaction, 0, wp.spatial_vector(-ji[0], -ji[1], -ji[2], torque[0], torque[1], torque[2]))
        if wp.length_sq(ji) > 0.0:
            wp.atomic_add(stats, 1, 1.0)
            wp.atomic_add(stats, 2, wp.dot(ji, linear+wp.cross(angular, lever)))
            wp.atomic_max(stats, 3, wp.max(-gap, 0.0))


@wp.kernel
def _particle_contacts(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
                       reference: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                       mass: wp.array(dtype=float), code: wp.array(dtype=int),
                       impulse: wp.array(dtype=wp.vec3), stats: wp.array(dtype=float),
                       radius: float, reference_exclusion: float, dt: float,
                       stiffness: float, friction: float):
    p = wp.tid()
    query = wp.hash_grid_query(grid, x[p], 2.4*radius)
    for q in query:
        if q > p:
            if _conflict(code[p], code[q]) or wp.length_sq(reference[p]-reference[q]) > reference_exclusion**2.0:
                delta = x[p]-x[q]
                distance = wp.length(delta)
                if distance > 1.0e-10:
                    normal = delta/distance
                    effective = mass[p]*mass[q]/(mass[p]+mass[q])
                    ji = _contact_impulse(distance-2.0*radius, normal, v[p]-v[q], effective,
                                          dt, stiffness, friction)
                    wp.atomic_add(impulse, p, ji)
                    wp.atomic_sub(impulse, q, ji)
                    if wp.length_sq(ji) > 0.0:
                        wp.atomic_add(stats, 4, 1.0)
                        wp.atomic_add(stats, 5, wp.length(ji))
                        wp.atomic_max(stats, 6, wp.max(2.0*radius-distance, 0.0))


@wp.kernel
def _p2g(v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), code: wp.array(dtype=int),
         stress: wp.array(dtype=wp.mat33), impulse: wp.array(dtype=wp.vec3),
         nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
         gm: wp.array(dtype=float), gp: wp.array(dtype=wp.vec3), gj: wp.array(dtype=wp.vec3),
         node_count: int, volume: float, dt: float, gravity: wp.vec3):
    p, corner = wp.tid()
    i = code[p]*node_count+nodes[p, corner]
    weight = weights[p, corner]
    wp.atomic_add(gm, i, weight*mass[p])
    wp.atomic_add(gp, i, weight*mass[p]*v[p])
    wp.atomic_add(gj, i, weight*(impulse[p]+mass[p]*gravity*dt)-dt*volume*(stress[p]*gradients[p, corner]))


@wp.kernel
def _grid_update(gm: wp.array(dtype=float), gp: wp.array(dtype=wp.vec3), gj: wp.array(dtype=wp.vec3),
                 old: wp.array(dtype=wp.vec3), new: wp.array(dtype=wp.vec3),
                 compatible: wp.array2d(dtype=int), node_count: int, code_count: int):
    c, node = wp.tid()
    i = c*node_count+node
    m = gm[i]
    old[i] = wp.vec3(0.0)
    new[i] = wp.vec3(0.0)
    if m > 1.0e-15:
        total_mass = float(0.0)
        for other in range(code_count):
            total_mass += gm[other*node_count+node]
        raw = gp[i]+gj[i]
        coupled = raw
        for other in range(code_count):
            if other != c and compatible[c, other] != 0:
                j = other*node_count+node
                coupled += (m*(gp[j]+gj[j])-gm[j]*raw)/total_mass
        old[i] = gp[i]/m
        new[i] = coupled/m


@wp.kernel
def _g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), code: wp.array(dtype=int),
         nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
         old: wp.array(dtype=wp.vec3), new: wp.array(dtype=wp.vec3),
         node_count: int, dt: float, pic: float):
    p = wp.tid()
    vpic = wp.vec3(0.0)
    delta = wp.vec3(0.0)
    for corner in range(8):
        i = code[p]*node_count+nodes[p, corner]
        w = weights[p, corner]
        vpic += w*new[i]
        delta += w*(new[i]-old[i])
    v[p] = pic*vpic+(1.0-pic)*(v[p]+delta)
    # MUSL remaps this same updated particle velocity for Fdot. Advecting with
    # vpic while using FLIP velocity for Fdot makes x and F inconsistent.
    x[p] = x[p]+dt*v[p]


@wp.kernel
def _musl(v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), code: wp.array(dtype=int),
          nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
          gp: wp.array(dtype=wp.vec3), node_count: int):
    p, corner = wp.tid()
    i = code[p]*node_count+nodes[p, corner]
    wp.atomic_add(gp, i, weights[p, corner]*mass[p]*v[p])


@wp.kernel
def _deformation(F: wp.array(dtype=wp.mat33), code: wp.array(dtype=int),
                 nodes: wp.array2d(dtype=int), gradients: wp.array2d(dtype=wp.vec3),
                 velocity: wp.array(dtype=wp.vec3), node_count: int, dt: float):
    p = wp.tid()
    gradient = wp.mat33(0.0)
    for corner in range(8):
        i = code[p]*node_count+nodes[p, corner]
        gradient += _outer(velocity[i], gradients[p, corner])
    F[p] = F[p]+dt*gradient


@wp.kernel
def _assign(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), code: wp.array(dtype=int),
            poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
            cuts: wp.array(dtype=int), index: wp.array(dtype=int),
            band: float, edge_width: float, rear: float, dt: float):
    p = wp.tid()
    i = index[0]
    cut = cuts[i]
    if cut >= 0:
        factor = 1
        if cut == 1:
            factor = 3
        if (code[p]//factor) % 3 == 0:
            pose = poses[i]
            local = wp.transform_point(wp.transform_inverse(pose), x[p])
            twist = twists[i]
            tool_v = wp.vec3(twist[0], twist[1], twist[2])
            forward = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
            displacement = wp.max(wp.dot(tool_v-v[p], forward)*dt, 0.0)
            if wp.abs(local[2]) <= band and wp.abs(local[1]) <= edge_width:
                if local[0] <= 0.0 and local[0] >= -rear-displacement:
                    digit = 2
                    if local[2] < 0.0:
                        digit = 1
                    code[p] = code[p]+factor*digit


@wp.kernel
def _advance(index: wp.array(dtype=int)):
    index[0] += 1


@wp.kernel
def _start_cut(progress: wp.array(dtype=float), cuts: wp.array(dtype=int),
               index: wp.array(dtype=int), previous: wp.array(dtype=int)):
    p = wp.tid()
    active = cuts[index[0]]
    if active >= 0 and active != previous[0]:
        progress[p] = 0.0


@wp.kernel
def _remember_cut(cuts: wp.array(dtype=int), index: wp.array(dtype=int), previous: wp.array(dtype=int)):
    active = cuts[index[0]]
    if active >= 0:
        previous[0] = active


class TLMPM:
    """GPU physics state; all particles are initially one uncut continuum."""
    def __init__(self, points, masses, spacing, config=TLMPMConfig(), device="cuda:1", *,
                 fixture_boxes=None, table_plane=None):
        if config.cuts not in (1, 2):
            raise ValueError("One or two retained cuts supported by this prototype")
        if not 0 <= config.pic <= 1 or not 0 < config.sharpness < 1:
            raise ValueError("Invalid transfer or sharpness parameter")
        wp.init()
        self.config, self.device, self.spacing = config, device, spacing
        self.n = len(points)
        self.volume = spacing**3
        self.radius = config.contact_radius_scale*spacing*.5
        self.code_count = 3**config.cuts
        self.fixture_data = None
        if fixture_boxes is not None or table_plane is not None:
            if config.floor:
                raise ValueError("Custom fixtures require floor=False; do not also apply the legacy tray")
            boxes = np.asarray(fixture_boxes, dtype=np.float32).reshape(-1, 2, 3)
            plane = np.asarray(table_plane, dtype=np.float32)
            if not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
                raise ValueError("Fixture boxes need finite centers and positive half extents")
            if plane.shape != (4,) or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]), 1.):
                raise ValueError("Table plane needs a unit inward normal and signed offset")
            from .fixture_contacts import fixture_contacts
            with wp.ScopedDevice(device):
                self.fixture_data = (fixture_contacts,
                    wp.array(boxes[:, 0].copy(), dtype=wp.vec3),
                    wp.array(boxes[:, 1].copy(), dtype=wp.vec3), wp.vec4(*plane))
        origin = np.floor(points.min(0)/config.dx)*config.dx-2*config.dx
        dims = np.ceil((points.max(0)-origin)/config.dx).astype(int)+3
        cells = np.floor((points-origin)/config.dx).astype(int)
        fraction = (points-origin)/config.dx-cells
        nodes = np.empty((self.n, 8), np.int32)
        weights = np.empty((self.n, 8), np.float32)
        gradients = np.empty((self.n, 8, 3), np.float32)
        for corner in range(8):
            bit = np.array([(corner >> j) & 1 for j in range(3)])
            xyz = cells+bit
            nodes[:, corner] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
            f = np.where(bit, fraction, 1-fraction)
            weights[:, corner] = f.prod(1)
            for axis in range(3):
                gradients[:, corner, axis] = (2*bit[axis]-1)*f[:, (axis+1)%3]*f[:, (axis+2)%3]/config.dx
        self.node_count = int(dims.prod())
        self.grid_origin, self.grid_dims = origin, dims
        total = self.node_count*self.code_count
        with wp.ScopedDevice(device):
            self.reference = wp.array(points, dtype=wp.vec3)
            self.x = wp.array(points, dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=wp.vec3)
            self.mass = wp.array(masses, dtype=float)
            self.code = wp.zeros(self.n, dtype=int)
            self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)), dtype=wp.mat33)
            self.Fpi = wp.clone(self.F)
            self.stress = wp.zeros(self.n, dtype=wp.mat33)
            self.plastic = wp.zeros(self.n, dtype=float)
            self.progress = wp.zeros(self.n, dtype=float)
            self.contact_work = wp.zeros(self.n, dtype=float)
            self.impulse = wp.zeros(self.n, dtype=wp.vec3)
            self.nodes = wp.array(nodes, dtype=int)
            self.weights = wp.array(weights, dtype=float)
            self.gradients = wp.array(gradients, dtype=wp.vec3)
            self.compatible = wp.array(compatibility(config.cuts).astype(np.int32), dtype=int)
            self.gm = wp.zeros(total, dtype=float)
            self.gp = wp.zeros(total, dtype=wp.vec3)
            self.gj = wp.zeros(total, dtype=wp.vec3)
            self.old = wp.zeros(total, dtype=wp.vec3)
            self.new = wp.zeros(total, dtype=wp.vec3)
            self.index = wp.zeros(1, dtype=int)
            self.previous_cut = wp.full(1, -1, dtype=int)
            self.reaction = wp.zeros(1, dtype=wp.spatial_vector)
            self.stats = wp.zeros(7, dtype=float)
            self.hash = wp.HashGrid(64, 64, 32)
            self.hash.build(self.x, 2.4*self.radius)
        # Default stationary tool outside the specimen. A probe supplies a timed
        # sequence; interactive use can supply one pose before each substep.
        self.set_trajectory(np.array([[.2, 0, .1, 0, 0, 0, 1]], np.float32),
                            np.zeros((1, 6), np.float32), np.array([-1], np.int32))

    def set_trajectory(self, poses, twists, cuts):
        if len(poses) != len(twists) or len(poses) != len(cuts):
            raise ValueError("Tool timestamps must align")
        with wp.ScopedDevice(self.device):
            self.poses = wp.array(poses, dtype=wp.transform)
            self.twists = wp.array(twists, dtype=wp.spatial_vector)
            self.cuts = wp.array(cuts, dtype=int)
            self.index.zero_()
        self.trajectory_length = len(poses)

    def step(self, advance=True):
        c, n = self.config, self.n
        lam = c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu = c.young/(2*(1+c.poisson))
        with wp.ScopedDevice(self.device):
            wp.launch(_start_cut, n, [self.progress, self.cuts, self.index, self.previous_cut])
            wp.launch(_remember_cut, 1, [self.cuts, self.index, self.previous_cut])
            wp.launch(_stress, n, [self.F, self.Fpi, self.stress, self.plastic, lam, mu, c.yield_stress])
            wp.launch(_rigid_contacts, n, [self.x, self.v, self.mass, self.impulse, self.progress,
                self.contact_work, self.poses, self.twists, self.cuts, self.index,
                self.reaction, self.stats, self.radius, self.volume, c.dt, c.contact_stiffness,
                c.contact_friction, c.cutting_energy, c.sharpness, c.blade_length,
                c.blade_height, c.blade_half_width, int(c.floor), int(c.fixture_walls), int(c.tool)])
            if self.fixture_data is not None:
                kernel, centers, halves, plane = self.fixture_data
                wp.launch(kernel, n, [self.x, self.v, self.mass, self.impulse, centers, halves,
                    plane, self.radius, c.dt, c.contact_stiffness, c.contact_friction, self.stats])
            if c.material_contact:
                self.hash.build(self.x, 2.4*self.radius)
                wp.launch(_particle_contacts, n, [self.hash.id, self.x, self.reference,
                    self.v, self.mass, self.code, self.impulse, self.stats, self.radius,
                    3*c.dx, c.dt, c.particle_contact_stiffness, c.contact_friction])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(_p2g, (n, 8), [self.v, self.mass, self.code, self.stress, self.impulse,
                self.nodes, self.weights, self.gradients, self.gm, self.gp, self.gj,
                self.node_count, self.volume, c.dt, wp.vec3(*c.gravity)])
            wp.launch(_grid_update, (self.code_count, self.node_count), [self.gm, self.gp, self.gj,
                self.old, self.new, self.compatible, self.node_count, self.code_count])
            wp.launch(_g2p, n, [self.x, self.v, self.code, self.nodes, self.weights,
                self.old, self.new, self.node_count, c.dt, c.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(_musl, (n, 8), [self.v, self.mass, self.code, self.nodes,
                self.weights, self.gp, self.node_count])
            wp.launch(_grid_update, (self.code_count, self.node_count), [self.gm, self.gp, self.gj,
                self.old, self.new, self.compatible, self.node_count, self.code_count])
            wp.launch(_deformation, n, [self.F, self.code, self.nodes, self.gradients,
                self.new, self.node_count, c.dt])
            wp.launch(_assign, n, [self.x, self.v, self.code, self.poses, self.twists,
                self.cuts, self.index, 6*c.dx, c.blade_half_width, 2*c.dx, c.dt])
            if advance:
                wp.launch(_advance, 1, [self.index])

    def snapshot(self):
        return {"positions": self.x.numpy(), "velocities": self.v.numpy(),
                "codes": self.code.numpy(), "F": self.F.numpy(), "Fpi": self.Fpi.numpy(),
                "plastic_strain": self.plastic.numpy(), "cut_progress": self.progress.numpy(),
                "contact_work_j": self.contact_work.numpy()}
