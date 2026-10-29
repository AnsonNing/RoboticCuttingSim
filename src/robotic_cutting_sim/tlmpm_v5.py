"""Scalable persistent cut history (ADR 0014): packed 2-bit digits and sparse per-node logical fields.

Derived from tlmpm_v4 without changing tlmpm/tlmpm_v2/tlmpm_v3/tlmpm_v4. The
cutting rules are law 4 (ADR 0006), law 5 (ADR 0008) and law 8 (ADR 0012)
exactly as in tlmpm_v4: the contact, progress, engagement, segment-motion,
plowing-mask, stroke, band and lead kernels are copied verbatim, and the
per-substep launch order is the same. Laws 1-3, 6 and 7 are not provided here
(they stay in tlmpm_v4). Only the representation of the cut history changes:

- Packed digits. A particle code is one int32 with 2 bits per cut,
  d_k = (code >> 2k) & 3: 0 unassigned, 1 negative side, 2 positive side,
  3 invalid, so K <= 15 cuts. Assignment writes code |= digit << 2k (only from
  digit 0). Two codes conflict iff (A1 & B2) | (A2 & B1) != 0, with A1/A2 the
  masks of the cuts where code a has digit 1/2 (BladeMaster Eq. (7), same
  semantics as v4's ternary digits; no 3^K table).
- Sparse per-node logical fields. Each reference node owns S slots (default 8).
  Every substep _claim_slots maps each (particle, stencil corner) to the slot
  holding its code at that node with wp.atomic_cas (slot_code starts at -1;
  slots fill as a contiguous prefix). P2G, G2P, MUSL and the deformation update
  address node*S + slot. The grid update visits only the occupied slots of a
  node and applies v4's pairwise antisymmetric exchange between compatible
  fields, visiting the other fields in ascending code order, which is v4's
  field order (ternary and packed codes order digits identically). On CPU the
  result therefore equals tlmpm_v4 bit for bit for K <= 2; on GPU the slot
  index of a code is race dependent but no floating-point sum depends on it,
  so v5 has the same atomic-accumulation nondeterminism as v4 and no more.
- Overflow (more than S distinct codes at one node) never merges fields: the
  (particle, corner) entry is dropped from that substep, slot_overflow counts
  it, and the host raises (check_slots(); LiveCuttingMaterial and blade_probe
  call it after every control step/chunk). slot_max_occupied records the
  largest number of occupied slots at any node since construction.

Memory is O(S N) for any K <= 15 and the grid update is O(S^2) per node.
Labels change only P2G/MUSL compatibility and contact selection from the next
substep on; no particle is deleted, moved or duplicated. The law parameters are
v4's, provisional and uncalibrated; this is not a fracture-energy model.
"""
import numpy as np
import warp as wp

from .cut_topology import PACKED_MAX_CUTS, decode_digits, packed_to_ternary  # noqa: F401  (re-exported helpers)
from .material_config import TLMPMConfig

CODE_ENCODING = "packed2"
DEFAULT_SLOTS = 8
EMPTY_SLOT = wp.constant(-1)
LOW_BITS = wp.constant(0x55555555)  # bit 2k of every digit k


@wp.func
def _digit(code: int, cut: int):
    return (code >> (2*cut)) & 3


@wp.func
def _conflict_packed(a: int, b: int):
    """Opposite sides of some cut: (A1 & B2) | (A2 & B1) != 0 (digit 3 never occurs)."""
    la = a & LOW_BITS
    ha = (a >> 1) & LOW_BITS
    lb = b & LOW_BITS
    hb = (b >> 1) & LOW_BITS
    a1 = la ^ (la & ha)
    a2 = ha ^ (la & ha)
    b1 = lb ^ (lb & hb)
    b2 = hb ^ (lb & hb)
    return ((a1 & b2) | (a2 & b1)) != 0


# ----------------------------------------------------------------------------- copied verbatim from tlmpm_v4

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
def _profile_sdf(p: wp.vec3, profile: wp.array(dtype=wp.vec2), count: int, half_width: float):
    """Exact distance/normal of an extruded convex CCW x/z cross-section.

    Same per-edge operations and order as _wedge_sdf, so the legacy triangle
    [(0,0), (-L,H), (-L,-0.0004)] reproduces it exactly.
    """
    x = wp.vec2(p[0], p[2])
    best = float(1.0e10)
    # Dynamic-loop variables must not be Warp constants; the first edge
    # always overwrites both vectors because every distance is < 1e10.
    closest = x*0.0
    outward = x*0.0
    inside = int(1)
    for edge in range(count):
        v0 = profile[edge]
        v1 = profile[(edge+1) % count]
        e = v1-v0
        if e[0]*(x[1]-v0[1])-e[1]*(x[0]-v0[0]) < 0.0:
            inside = 0
        q = _segment(x, v0, v1)
        distance = wp.length_sq(x-q)
        if distance < best:
            best = distance
            closest = q
            outward = wp.normalize(wp.vec2(e[1], -e[0]))
    distance = wp.sqrt(best)
    n2 = outward
    sd2 = -distance
    if inside == 0:
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
def _edge_segment(y: float, half_width: float, segments: int):
    k = int((y+half_width)/(2.0*half_width)*float(segments))
    return wp.clamp(k, 0, segments-1)

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
                    friction: float, gc: float, sharpness: float, profile: wp.array(dtype=wp.vec2),
                    profile_count: int, blade_width: float, floor: int, walls: int, tool: int,
                    law: int, slice_efficiency: float, zone: float, zone_rear: float, flags: wp.array(dtype=int),
                    segments: int, seg_ok: wp.array(dtype=int), engaged: wp.array(dtype=int)):
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
        hit = _profile_sdf(local, profile, profile_count, blade_width)
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
        if law == 1:
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
        else:
            in_zone = (local[0] >= -zone_rear and local[0] <= 2.0*radius and wp.abs(local[2]) <= zone
                       and wp.abs(local[1]) <= blade_width)
            if cuts[i] >= 0 and in_zone:
                along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
                us = wp.dot(-relative, along)
                slice_dir = along*wp.sign(us)
                push = wp.max(wp.dot(ji, advance), 0.0)
                drag = wp.max(wp.dot(ji, slice_dir), 0.0)
                # Only the in-plane part of the edge's relative motion opens the
                # crack: weight by cos^2 of the angle between motion and blade plane.
                blade_normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
                un = wp.dot(-relative, blade_normal)
                in_plane = wp.max(wp.length_sq(relative)-un*un, 0.0)
                alignment = in_plane/(in_plane+un*un+1.0e-20)
                dw = (push*wp.max(uc, 0.0)+slice_efficiency*drag*wp.abs(us))*alignment
                if dw > 1.0e-20:
                    threshold = gc*wp.pow(volume, 2.0/3.0)*(1.0-sharpness)/sharpness
                    remaining = wp.max((1.0-progress[p])*threshold, 0.0)
                    fraction = wp.min(remaining/dw, 1.0)
                    contact_work[p] = contact_work[p]+fraction*dw
                    progress[p] = wp.min(progress[p]+fraction*dw/wp.max(threshold, 1.0e-12), 1.0)
                    # Material already cut no longer resists in-plane edge motion.
                    ji = ji-(1.0-fraction)*(push*advance+drag*slice_dir)
                if law == 4:
                    # Law 4 engagement W: directional cutting work at a segment
                    # with actual cutting motion M_k (stroke gate, not per segment).
                    if dw > 1.0e-20 and seg_ok[_edge_segment(local[1], blade_width, segments)] != 0:
                        wp.atomic_max(engaged, 0, 1)
                else:
                    segment_motion = linear + wp.cross(angular, along * local[1])
                    advancing = wp.dot(segment_motion, advance)
                    drawing = wp.dot(segment_motion, along)
                    actual_edge_motion = advancing > 1.0e-5 or (wp.abs(drawing) > 1.0e-5 and advancing >= -1.0e-5)
                    if progress[p] >= 1.0 and (law != 3 or (actual_edge_motion and dw > 1.0e-20)):
                        wp.atomic_max(flags, _edge_segment(local[1], blade_width, segments), 1)
        impulse[p] = impulse[p]+ji
        torque = wp.cross(lever, -ji)
        wp.atomic_add(reaction, 0, wp.spatial_vector(-ji[0], -ji[1], -ji[2], torque[0], torque[1], torque[2]))
        if wp.length_sq(ji) > 0.0:
            wp.atomic_add(stats, 1, 1.0)
            wp.atomic_add(stats, 2, wp.dot(ji, linear+wp.cross(angular, lever)))
            wp.atomic_max(stats, 3, wp.max(-gap, 0.0))

@wp.kernel
def _obstacle_contacts(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                       impulse: wp.array(dtype=wp.vec3), poses: wp.array(dtype=wp.transform),
                       twists: wp.array(dtype=wp.spatial_vector), index: wp.array(dtype=int),
                       reaction: wp.array(dtype=wp.spatial_vector), radius: float, dt: float,
                       stiffness: float, friction: float, box_centers: wp.array(dtype=wp.vec3),
                       box_halves: wp.array(dtype=wp.vec3), box: int,
                       box_reactions: wp.array(dtype=wp.spatial_vector)):
    """Blunt tool parts (handle, finger pads): push material, never cut.

    One launch per edge-frame box keeps _rigid_contacts identical to v1 law 1.
    """
    p = wp.tid()
    xp, vp, mp = x[p], v[p], mass[p]
    i = index[0]
    pose = poses[i]
    local = wp.transform_point(wp.transform_inverse(pose), xp)
    hit = _box_sdf(local-box_centers[box], box_halves[box])
    normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
    twist = twists[i]
    linear = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    lever = xp-wp.transform_get_translation(pose)-normal*radius
    relative = vp-linear-wp.cross(angular, lever)
    jb = _contact_impulse(hit[3]-radius, normal, relative, mp, dt, stiffness, friction)
    if wp.length_sq(jb) > 0.0:
        impulse[p] = impulse[p]+jb
        torque = wp.cross(lever, -jb)
        wp.atomic_add(reaction, 0, wp.spatial_vector(-jb[0], -jb[1], -jb[2], torque[0], torque[1], torque[2]))
        wp.atomic_add(box_reactions, box, wp.spatial_vector(-jb[0], -jb[1], -jb[2], torque[0], torque[1], torque[2]))

@wp.func
def _swept_side(xp: wp.vec3, pose: wp.transform, twist: wp.spatial_vector, local: wp.vec3):
    """Signed distance to the surface swept by the moving edge (law 3)."""
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    tool_v = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    segment_motion = tool_v + wp.cross(angular, along * local[1])
    swept_normal = wp.cross(segment_motion, along)
    side = local[2]
    if wp.length_sq(swept_normal) > 1.0e-20:
        swept_normal = wp.normalize(swept_normal)
        if wp.dot(swept_normal, normal) < 0.0:
            swept_normal = -swept_normal
        side = wp.dot(xp - wp.transform_get_translation(pose), swept_normal)
    return side

@wp.kernel
def _segment_motion(poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                    cuts: wp.array(dtype=int), index: wp.array(dtype=int), edge_width: float, segments: int,
                    min_speed: float, tan2_max: float, seg_ok: wp.array(dtype=int), any_motion: wp.array(dtype=int)):
    """Law 4 M_k: an armed edge segment moves forward or draws, within theta_max
    of the blade plane. Pure blade-normal motion, stopping and withdrawal fail."""
    k = wp.tid()
    i = index[0]
    pose = poses[i]
    twist = twists[i]
    y = (float(k)+0.5)/float(segments)*2.0*edge_width-edge_width
    advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    motion = wp.vec3(twist[0], twist[1], twist[2]) + wp.cross(wp.vec3(twist[3], twist[4], twist[5]), along*y)
    advancing = wp.dot(motion, advance)
    drawing = wp.dot(motion, along)
    off_plane = wp.dot(motion, normal)
    moving = advancing > min_speed or (wp.abs(drawing) > min_speed and advancing >= -min_speed)
    in_plane = off_plane*off_plane <= tan2_max*(advancing*advancing+drawing*drawing)
    ok = int(0)
    if cuts[i] >= 0 and moving and in_plane:
        ok = 1
        wp.atomic_max(any_motion, 0, 1)
    seg_ok[k] = ok

@wp.kernel
def _segment_plow(poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                  index: wp.array(dtype=int), edge_width: float, segments: int, tan2_plow: float,
                  seg_ok: wp.array(dtype=int), seg_plow: wp.array(dtype=int)):
    """Law 8 (ADR 0012) plowing mask: segment k has M_k (seg_ok, from _segment_motion
    this substep) and its motion leaves the blade plane by at least theta_plow,
    (v.n)^2 >= tan^2(theta_plow) [(v.d)^2 + (v.e)^2]. Same segment motion as _segment_motion."""
    k = wp.tid()
    i = index[0]
    pose = poses[i]
    twist = twists[i]
    y = (float(k)+0.5)/float(segments)*2.0*edge_width-edge_width
    advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    motion = wp.vec3(twist[0], twist[1], twist[2]) + wp.cross(wp.vec3(twist[3], twist[4], twist[5]), along*y)
    advancing = wp.dot(motion, advance)
    drawing = wp.dot(motion, along)
    off_plane = wp.dot(motion, normal)
    plowing = off_plane*off_plane >= tan2_plow*(advancing*advancing+drawing*drawing)
    ok = int(0)
    if seg_ok[k] != 0 and plowing:
        ok = 1
    seg_plow[k] = ok

@wp.kernel
def _update_stroke(twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int),
                   index: wp.array(dtype=int), engaged: wp.array(dtype=int), any_motion: wp.array(dtype=int),
                   stroke: wp.array(dtype=int), travel: wp.array(dtype=float), dt: float, persist: float):
    """Law 4 stroke state S (one per blade) and edge-centre travel since the
    last engaged substep; S = cut+1 enables side assignment this substep."""
    i = index[0]
    cut = cuts[i]
    if cut >= 0 and any_motion[0] != 0:
        if engaged[0] != 0:
            stroke[0] = cut+1
            travel[0] = 0.0
        elif stroke[0] == cut+1:
            twist = twists[i]
            travel[0] = travel[0]+wp.length(wp.vec3(twist[0], twist[1], twist[2]))*dt
            if travel[0] > persist:
                stroke[0] = 0
        else:
            stroke[0] = 0
    else:
        stroke[0] = 0
        travel[0] = 0.0

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



# ----------------------------------------------------------------------------- packed / sparse kernels

@wp.kernel
def _claim_slots(code: wp.array(dtype=int), nodes: wp.array2d(dtype=int), slot_code: wp.array(dtype=int),
                 slot: wp.array2d(dtype=int), slots: int, overflow: wp.array(dtype=int),
                 max_occupied: wp.array(dtype=int)):
    """Map (particle, corner) to the slot of its code at that node (ADR 0014).

    Slots are tried in order; an empty slot (-1) is claimed with atomic_cas, so
    slots fill as a contiguous prefix and each occupied slot holds exactly one
    distinct code. More than ``slots`` distinct codes: the entry gets -1 and is
    counted in ``overflow`` (never merged into another field)."""
    p, corner = wp.tid()
    c = code[p]
    base = nodes[p, corner]*slots
    found = int(-1)
    for s in range(slots):
        current = slot_code[base+s]
        if current == EMPTY_SLOT:
            current = wp.atomic_cas(slot_code, base+s, EMPTY_SLOT, c)
            if current == EMPTY_SLOT:
                current = c
        if current == c:
            found = s
            break
    slot[p, corner] = found
    if found < 0:
        wp.atomic_add(overflow, 0, 1)
    else:
        wp.atomic_max(max_occupied, 0, found+1)


@wp.kernel
def _particle_contacts(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
                       reference: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                       mass: wp.array(dtype=float), code: wp.array(dtype=int),
                       impulse: wp.array(dtype=wp.vec3), stats: wp.array(dtype=float),
                       radius: float, reference_exclusion: float, dt: float,
                       stiffness: float, friction: float):
    """tlmpm_v4 _particle_contacts with the packed conflict test."""
    p = wp.tid()
    query = wp.hash_grid_query(grid, x[p], 2.4*radius)
    for q in query:
        if q > p:
            if _conflict_packed(code[p], code[q]) or wp.length_sq(reference[p]-reference[q]) > reference_exclusion**2.0:
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
def _p2g(v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), slot: wp.array2d(dtype=int),
         stress: wp.array(dtype=wp.mat33), impulse: wp.array(dtype=wp.vec3),
         nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
         gm: wp.array(dtype=float), gp: wp.array(dtype=wp.vec3), gj: wp.array(dtype=wp.vec3),
         slots: int, volume: float, dt: float, gravity: wp.vec3):
    p, corner = wp.tid()
    s = slot[p, corner]
    if s >= 0:
        i = nodes[p, corner]*slots+s
        weight = weights[p, corner]
        wp.atomic_add(gm, i, weight*mass[p])
        wp.atomic_add(gp, i, weight*mass[p]*v[p])
        wp.atomic_add(gj, i, weight*(impulse[p]+mass[p]*gravity*dt)-dt*volume*(stress[p]*gradients[p, corner]))


@wp.func
def _next_slot(slot_code: wp.array(dtype=int), base: int, slots: int, previous: int):
    """Slot holding the smallest code > previous at this node (-1 if none)."""
    best = int(2147483647)
    chosen = int(-1)
    for u in range(slots):
        cu = slot_code[base+u]
        if cu == EMPTY_SLOT:
            break
        if cu > previous and cu < best:
            best = cu
            chosen = u
    return chosen


@wp.kernel
def _grid_update(gm: wp.array(dtype=float), gp: wp.array(dtype=wp.vec3), gj: wp.array(dtype=wp.vec3),
                 old: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
                 slot_code: wp.array(dtype=int), slots: int):
    """tlmpm_v4 _grid_update over the occupied slots of a node.

    The same pairwise antisymmetric exchange with every compatible field
    (bitmask test); total mass and the exchange sum visit fields in ascending
    code order, which is v4's loop order, so no sum depends on slot order."""
    s, node = wp.tid()
    base = node*slots
    i = base+s
    old[i] = wp.vec3(0.0)
    new_v[i] = wp.vec3(0.0)
    c = slot_code[i]
    if c != EMPTY_SLOT:
        m = gm[i]
        if m > 1.0e-15:
            total_mass = float(0.0)
            previous = int(-1)
            for t in range(slots):
                j = _next_slot(slot_code, base, slots, previous)
                if j < 0:
                    break
                previous = slot_code[base+j]
                total_mass += gm[base+j]
            raw = gp[i]+gj[i]
            coupled = raw
            previous = int(-1)
            for t in range(slots):
                j = _next_slot(slot_code, base, slots, previous)
                if j < 0:
                    break
                other = slot_code[base+j]
                previous = other
                if j != s and not _conflict_packed(c, other):
                    k = base+j
                    coupled += (m*(gp[k]+gj[k])-gm[k]*raw)/total_mass
            old[i] = gp[i]/m
            new_v[i] = coupled/m


@wp.kernel
def _g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), slot: wp.array2d(dtype=int),
         nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
         old: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3),
         slots: int, dt: float, pic: float):
    p = wp.tid()
    vpic = wp.vec3(0.0)
    delta = wp.vec3(0.0)
    for corner in range(8):
        s = slot[p, corner]
        if s >= 0:
            i = nodes[p, corner]*slots+s
            w = weights[p, corner]
            vpic += w*new_v[i]
            delta += w*(new_v[i]-old[i])
    v[p] = pic*vpic+(1.0-pic)*(v[p]+delta)
    # MUSL remaps this same updated particle velocity for Fdot (as tlmpm_v4).
    x[p] = x[p]+dt*v[p]


@wp.kernel
def _musl(v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), slot: wp.array2d(dtype=int),
          nodes: wp.array2d(dtype=int), weights: wp.array2d(dtype=float),
          gp: wp.array(dtype=wp.vec3), slots: int):
    p, corner = wp.tid()
    s = slot[p, corner]
    if s >= 0:
        wp.atomic_add(gp, nodes[p, corner]*slots+s, weights[p, corner]*mass[p]*v[p])


@wp.kernel
def _deformation(F: wp.array(dtype=wp.mat33), slot: wp.array2d(dtype=int),
                 nodes: wp.array2d(dtype=int), gradients: wp.array2d(dtype=wp.vec3),
                 velocity: wp.array(dtype=wp.vec3), slots: int, dt: float):
    p = wp.tid()
    gradient = wp.mat33(0.0)
    for corner in range(8):
        s = slot[p, corner]
        if s >= 0:
            gradient += _outer(velocity[nodes[p, corner]*slots+s], gradients[p, corner])
    F[p] = F[p]+dt*gradient


@wp.kernel
def _assign_band(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), code: wp.array(dtype=int),
                 poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                 cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                 band: float, edge_width: float, rear: float, dt: float,
                 stroke: wp.array(dtype=int), seg_ok: wp.array(dtype=int), segments: int, max_cuts: int):
    """tlmpm_v4 law-4 band assignment writing packed digit ``cut``."""
    p = wp.tid()
    i = index[0]
    cut = cuts[i]
    if cut >= 0 and cut < max_cuts and stroke[0] == cut+1:
        if _digit(code[p], cut) == 0:
            pose = poses[i]
            local = wp.transform_point(wp.transform_inverse(pose), x[p])
            if wp.abs(local[1]) <= edge_width and local[0] <= 0.0:
                twist = twists[i]
                tool_v = wp.vec3(twist[0], twist[1], twist[2])
                forward = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
                displacement = wp.max(wp.dot(tool_v-v[p], forward)*dt, 0.0)
                if local[0] >= -rear-displacement and seg_ok[_edge_segment(local[1], edge_width, segments)] != 0:
                    side = _swept_side(x[p], pose, twist, local)
                    if wp.abs(side) <= band:
                        digit = 2
                        if side < 0.0:
                            digit = 1
                        code[p] = code[p] | (digit << (2*cut))


@wp.kernel
def _assign_lead(x: wp.array(dtype=wp.vec3), code: wp.array(dtype=int),
                 poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                 cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                 band: float, edge_width: float, lead: float, core: float,
                 stroke: wp.array(dtype=int), seg_ok: wp.array(dtype=int), segments: int, max_cuts: int):
    """tlmpm_v4 law-5 lead pre-assignment (law 8: with the plowing mask) writing packed digit ``cut``."""
    p = wp.tid()
    i = index[0]
    cut = cuts[i]
    if cut >= 0 and cut < max_cuts and stroke[0] == cut+1:
        if _digit(code[p], cut) == 0:
            pose = poses[i]
            local = wp.transform_point(wp.transform_inverse(pose), x[p])
            if (wp.abs(local[1]) <= edge_width and local[0] > 0.0 and local[0] <= lead
                    and seg_ok[_edge_segment(local[1], edge_width, segments)] != 0):
                side = _swept_side(x[p], pose, twists[i], local)
                if wp.abs(side) > core and wp.abs(side) <= band:
                    digit = 2
                    if side < 0.0:
                        digit = 1
                    code[p] = code[p] | (digit << (2*cut))


class SlotOverflowError(RuntimeError):
    """More distinct cut codes met at one reference node than it has slots."""


class TLMPM:
    """GPU physics state with packed cut codes and S sparse logical-field slots per node.

    Same constructor as tlmpm_v4.TLMPM plus ``slots_per_node``; ``config.cuts``
    is the number of packed digits K (1..15) and ``cutting_law`` is 4, 5 or 8.
    """
    code_encoding = CODE_ENCODING

    def __init__(self, points, masses, spacing, config=TLMPMConfig(), device="cuda:0", *,
                 fixture_boxes=None, table_plane=None, blade_profile_xz=None,
                 cutting_law=8, slice_efficiency=1., obstacle_boxes_edge=None, edge_segments=None,
                 slots_per_node=DEFAULT_SLOTS):
        if not isinstance(config.cuts, (int, np.integer)) or not 1 <= int(config.cuts) <= PACKED_MAX_CUTS:
            raise ValueError(f"tlmpm_v5 keeps 1..{PACKED_MAX_CUTS} packed cut digits")
        if not isinstance(slots_per_node, (int, np.integer)) or not 1 <= int(slots_per_node) <= 64:
            raise ValueError("slots_per_node must be an integer in 1..64")
        if not 0 <= config.pic <= 1 or not 0 < config.sharpness < 1:
            raise ValueError("Invalid transfer or sharpness parameter")
        if blade_profile_xz is None:
            blade_profile_xz = [[0., 0.], [-config.blade_length, config.blade_height], [-config.blade_length, -.0004]]
        profile = np.asarray(blade_profile_xz, np.float32)
        edges = np.roll(profile, -1, 0)-profile
        if (profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 3 or not np.isfinite(profile).all()
                or np.any(edges[:, 0]*np.roll(edges, -1, 0)[:, 1]-edges[:, 1]*np.roll(edges, -1, 0)[:, 0] <= 0)):
            raise ValueError("Blade profile must be a finite counter-clockwise strictly convex polygon")
        if cutting_law not in (4, 5, 8) or not np.isfinite(slice_efficiency) or slice_efficiency < 0:
            raise ValueError('tlmpm_v5 cutting_law must be 4, 5 or 8 with finite nonnegative slice efficiency')
        obstacles = np.asarray([] if obstacle_boxes_edge is None else obstacle_boxes_edge, np.float32).reshape(-1, 2, 3)
        if edge_segments is None:
            edge_segments = int(np.clip(np.ceil(2*config.blade_half_width/spacing), 1, 4096))
        if not np.isfinite(obstacles).all() or np.any(obstacles[:, 1] <= 0) or not 1 <= int(edge_segments) <= 4096:
            raise ValueError("Obstacle boxes need finite centers and positive half extents in the edge frame")
        wp.init()
        self.config, self.device, self.spacing = config, device, spacing
        self.blade_profile_xz = profile
        self.cutting_law, self.slice_efficiency = int(cutting_law), float(slice_efficiency)
        self.edge_segments, self.obstacle_boxes_edge = int(edge_segments), obstacles
        self.max_cuts, self.slots_per_node = int(config.cuts), int(slots_per_node)
        # Same provisional, uncalibrated parameters as tlmpm_v4 (ADR 0006/0008/0012).
        self.law4_parameters = {"band_h_m": 6*config.dx, "rear_b_back_m": 2*config.dx,
                                "persist_b_persist_m": 2*config.dx, "theta_max_deg": 45.,
                                "min_speed_m_s": 1e-5}
        self.law5_parameters = {"lead_b_lead_m": 2*config.dx, "core_s_core_m": config.dx}
        self.law8_parameters = {**self.law5_parameters, "theta_plow_deg": 15.}
        self.n = len(points)
        self.volume = spacing**3
        self.radius = config.contact_radius_scale*spacing*.5
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
        total = self.node_count*self.slots_per_node
        if total >= 2**31:
            raise ValueError("node_count * slots_per_node exceeds int32 indexing")
        with wp.ScopedDevice(device):
            self.profile = wp.array(profile, dtype=wp.vec2)
            self.edge_flags = wp.zeros(self.edge_segments, dtype=int)
            # Law-4 device state; reset only here (the live material captures
            # step() in a CUDA graph and the stroke persists across control steps).
            self.segment_ok = wp.zeros(self.edge_segments, dtype=int)
            self.segment_plow = wp.zeros(self.edge_segments, dtype=int)  # law 8 only (ADR 0012)
            self.stroke_engaged = wp.zeros(1, dtype=int)
            self.stroke_any_motion = wp.zeros(1, dtype=int)
            self.stroke = wp.zeros(1, dtype=int)
            self.stroke_travel = wp.zeros(1, dtype=float)
            padded = obstacles if len(obstacles) else np.ones((1, 2, 3), np.float32)
            self.box_centers = wp.array(padded[:, 0].copy(), dtype=wp.vec3)
            self.box_halves = wp.array(padded[:, 1].copy(), dtype=wp.vec3)
            self.box_reactions = wp.zeros(len(padded), dtype=wp.spatial_vector)
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
            # Sparse logical fields (ADR 0014): node*S + slot.
            self.slot_code = wp.full(total, -1, dtype=int)
            self.slot = wp.zeros((self.n, 8), dtype=int)
            self.slot_overflow = wp.zeros(1, dtype=int)      # cumulative dropped (particle, corner) entries
            self.slot_max_occupied = wp.zeros(1, dtype=int)  # cumulative max occupied slots at any node
            self.gm = wp.zeros(total, dtype=float)
            self.gp = wp.zeros(total, dtype=wp.vec3)
            self.gj = wp.zeros(total, dtype=wp.vec3)
            self.old = wp.zeros(total, dtype=wp.vec3)
            self.new_v = wp.zeros(total, dtype=wp.vec3)
            self.index = wp.zeros(1, dtype=int)
            self.previous_cut = wp.full(1, -1, dtype=int)
            self.reaction = wp.zeros(1, dtype=wp.spatial_vector)
            self.stats = wp.zeros(7, dtype=float)
            self.hash = wp.HashGrid(64, 64, 32)
            self.hash.build(self.x, 2.4*self.radius)
        self.set_trajectory(np.array([[.2, 0, .1, 0, 0, 0, 1]], np.float32),
                            np.zeros((1, 6), np.float32), np.array([-1], np.int32))

    def set_trajectory(self, poses, twists, cuts):
        if len(poses) != len(twists) or len(poses) != len(cuts):
            raise ValueError("Tool timestamps must align")
        cuts = np.asarray(cuts, np.int32)
        if np.any(cuts >= self.max_cuts) or np.any(cuts < -1):
            raise ValueError(f"Cut digits must be -1 (disarmed) or 0..{self.max_cuts-1}")
        with wp.ScopedDevice(self.device):
            self.poses = wp.array(poses, dtype=wp.transform)
            self.twists = wp.array(twists, dtype=wp.spatial_vector)
            self.cuts = wp.array(cuts, dtype=int)
            self.index.zero_()
        self.trajectory_length = len(poses)

    def step(self, advance=True):
        c, n, S = self.config, self.n, self.slots_per_node
        lam = c.young*c.poisson/((1+c.poisson)*(1-2*c.poisson))
        mu = c.young/(2*(1+c.poisson))
        with wp.ScopedDevice(self.device):
            wp.launch(_start_cut, n, [self.progress, self.cuts, self.index, self.previous_cut])
            wp.launch(_remember_cut, 1, [self.cuts, self.index, self.previous_cut])
            wp.launch(_stress, n, [self.F, self.Fpi, self.stress, self.plastic, lam, mu, c.yield_stress])
            self.edge_flags.zero_()
            self.stroke_engaged.zero_(); self.stroke_any_motion.zero_()
            wp.launch(_segment_motion, self.edge_segments, [self.poses, self.twists, self.cuts, self.index,
                c.blade_half_width, self.edge_segments, self.law4_parameters["min_speed_m_s"],
                float(np.tan(np.deg2rad(self.law4_parameters["theta_max_deg"]))**2),
                self.segment_ok, self.stroke_any_motion])
            if self.cutting_law == 8:
                wp.launch(_segment_plow, self.edge_segments, [self.poses, self.twists, self.index,
                    c.blade_half_width, self.edge_segments,
                    float(np.tan(np.deg2rad(self.law8_parameters["theta_plow_deg"]))**2),
                    self.segment_ok, self.segment_plow])
            wp.launch(_rigid_contacts, n, [self.x, self.v, self.mass, self.impulse, self.progress,
                self.contact_work, self.poses, self.twists, self.cuts, self.index,
                self.reaction, self.stats, self.radius, self.volume, c.dt, c.contact_stiffness,
                c.contact_friction, c.cutting_energy, c.sharpness, self.profile,
                len(self.blade_profile_xz), c.blade_half_width, int(c.floor), int(c.fixture_walls), int(c.tool),
                4, self.slice_efficiency, float(self.spacing),  # laws 5 and 8 use the law-4 contact path
                2*c.dx, self.edge_flags,
                self.edge_segments, self.segment_ok, self.stroke_engaged])
            if c.tool:
                for box in range(len(self.obstacle_boxes_edge)):
                    wp.launch(_obstacle_contacts, n, [self.x, self.v, self.mass, self.impulse, self.poses,
                        self.twists, self.index, self.reaction, self.radius, c.dt, c.contact_stiffness,
                        c.contact_friction, self.box_centers, self.box_halves, box, self.box_reactions])
            if self.fixture_data is not None:
                kernel, centers, halves, plane = self.fixture_data
                wp.launch(kernel, n, [self.x, self.v, self.mass, self.impulse, centers, halves,
                    plane, self.radius, c.dt, c.contact_stiffness, c.contact_friction, self.stats])
            if c.material_contact:
                self.hash.build(self.x, 2.4*self.radius)
                wp.launch(_particle_contacts, n, [self.hash.id, self.x, self.reference,
                    self.v, self.mass, self.code, self.impulse, self.stats, self.radius,
                    3*c.dx, c.dt, c.particle_contact_stiffness, c.contact_friction])
            # Sparse slots for this substep's codes (codes change only after the deformation update).
            self.slot_code.fill_(-1)
            wp.launch(_claim_slots, (n, 8), [self.code, self.nodes, self.slot_code, self.slot, S,
                self.slot_overflow, self.slot_max_occupied])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(_p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse,
                self.nodes, self.weights, self.gradients, self.gm, self.gp, self.gj,
                S, self.volume, c.dt, wp.vec3(*c.gravity)])
            wp.launch(_grid_update, (S, self.node_count), [self.gm, self.gp, self.gj,
                self.old, self.new_v, self.slot_code, S])
            wp.launch(_g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights,
                self.old, self.new_v, S, c.dt, c.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(_musl, (n, 8), [self.v, self.mass, self.slot, self.nodes,
                self.weights, self.gp, S])
            wp.launch(_grid_update, (S, self.node_count), [self.gm, self.gp, self.gj,
                self.old, self.new_v, self.slot_code, S])
            wp.launch(_deformation, n, [self.F, self.slot, self.nodes, self.gradients,
                self.new_v, S, c.dt])
            law4 = self.law4_parameters
            wp.launch(_update_stroke, 1, [self.twists, self.cuts, self.index, self.stroke_engaged,
                self.stroke_any_motion, self.stroke, self.stroke_travel, c.dt, law4["persist_b_persist_m"]])
            wp.launch(_assign_band, n, [self.x, self.v, self.code, self.poses, self.twists, self.cuts,
                self.index, law4["band_h_m"], c.blade_half_width, law4["rear_b_back_m"], c.dt,
                self.stroke, self.segment_ok, self.edge_segments, self.max_cuts])
            if self.cutting_law in (5, 8):
                # Law 5: lead rule on every cutting segment; law 8: only on cutting AND plowing segments.
                law5 = self.law5_parameters
                mask = self.segment_plow if self.cutting_law == 8 else self.segment_ok
                wp.launch(_assign_lead, n, [self.x, self.code, self.poses, self.twists, self.cuts, self.index,
                    law4["band_h_m"], c.blade_half_width, law5["lead_b_lead_m"], law5["core_s_core_m"],
                    self.stroke, mask, self.edge_segments, self.max_cuts])
            if advance:
                wp.launch(_advance, 1, [self.index])

    def slot_diagnostics(self):
        """Cumulative sparse-field diagnostics since construction (host copy; oracle only)."""
        return {"code_encoding": CODE_ENCODING, "max_cuts": self.max_cuts, "slots_per_node": self.slots_per_node,
                "max_occupied_slots": int(self.slot_max_occupied.numpy()[0]),
                "overflow_entries": int(self.slot_overflow.numpy()[0])}

    def check_slots(self):
        """Raise SlotOverflowError if any substep had more distinct codes at a node than slots."""
        diagnostics = self.slot_diagnostics()
        if diagnostics["overflow_entries"]:
            raise SlotOverflowError(
                f"{diagnostics['overflow_entries']} (particle, corner) entries found all {self.slots_per_node} slots "
                "of their node held other codes; the affected substeps are invalid (fields are never merged). "
                "Rerun with a larger slots_per_node.")
        return diagnostics

    def memory_bytes(self):
        """Bytes of every device array owned by this solver (excluding the hash grid)."""
        return int(sum(value.capacity for value in vars(self).values() if isinstance(value, wp.array)))

    def cutting_state(self, count_labels=False):
        """Law-4/5/8 stroke state after the last substep (same keys as tlmpm_v4; oracle only)."""
        state = {"cutting_law": self.cutting_law, "stroke": int(self.stroke.numpy()[0]),
                 "travel_m": float(self.stroke_travel.numpy()[0]), "engaged": int(self.stroke_engaged.numpy()[0]),
                 "any_motion": int(self.stroke_any_motion.numpy()[0])}
        if count_labels:
            state["labelled_count"] = int(np.count_nonzero(self.code.numpy()))
        return state

    def snapshot(self):
        """Same keys as tlmpm_v4; ``codes`` are packed (cut_topology.decode_digits / packed_to_ternary)."""
        return {"positions": self.x.numpy(), "velocities": self.v.numpy(),
                "codes": self.code.numpy(), "F": self.F.numpy(), "Fpi": self.Fpi.numpy(),
                "plastic_strain": self.plastic.numpy(), "cut_progress": self.progress.numpy(),
                "contact_work_j": self.contact_work.numpy()}

    def decoded_digits(self):
        """(N, K) digits of the current codes (0 unassigned, 1 negative, 2 positive)."""
        return decode_digits(self.code.numpy(), self.max_cuts)
