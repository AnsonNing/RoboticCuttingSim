"""Batched execution variant ``tlmpm_v7_2_batch`` of the frozen ``tlmpm_v7_2`` solver (ADR 0021 §3.3).

One launch per operation covers all N envs. Kernels labelled UNCHANGED are the
frozen kernels launched on the flat batch arrays (global node ids make their
node ranges disjoint). Kernels named ``*_b`` here are ENV kernels: their bodies
are the frozen bodies with env-indexed loads and stores (tool pose/twist/cut of
row ``e*T + index``, per-env scalars from ``params[e]``, per-env reductions into
``[N*k]`` rows), and they call the same frozen ``@wp.func`` helpers. Dead
branches of the frozen kernels that the production v7_2 step never takes are
removed and refused instead (``floor=True`` tray; laws other than the law-4
contact path that ``tlmpm_v5.step`` always passes for laws 5/8). Nothing here
changes a frozen module, and v7_2 physics is unchanged: batching changes how
v7_2 executes, not what it computes.

Float atomics: every float atomic target belongs to one env and that env's
particles are contiguous and in single-env order, so on the Warp CPU device
(sequential kernels) every target receives the same contributions in the same
order as in the single solver; that is why CPU parity is bitwise (ADR §5 L0).
``stats`` and ``contact_totals`` are split into sum and max arrays (§5 L4).
"""
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from .. import tlmpm_v5 as v5
from .. import tlmpm_v7 as V
from .. import tlmpm_v7_2_frontier as Fr
from ..tlmpm_v6 import _grid_positions
from ..tlmpm_v7_1 import _interface, _endpoint_work
from . import constitutive
from .constitutive import P_FRICTION, P_GC, P_PARTICLE_STIFFNESS, P_SHARPNESS, P_SLICE, P_STIFFNESS
from .errors import UnsupportedBatchConfiguration
from .layout import (ENV_ARRAYS, LINK_ARRAYS, NODE_SLOT_ARRAYS, PARTICLE_ARRAYS, STATS_MAX, STATS_SUM,
                     STATS_WIDTH, TOTALS_MAX, TOTALS_SUM, TOTALS_WIDTH, frozen_source_sha256)

SOLVER_NAME = "tlmpm_v7_2_batch"
BASE_SOLVER = "tlmpm_v7_2"
STATUS = "EXPERIMENTAL_NOT_ACCEPTED"
ST = wp.constant(STATS_WIDTH)
TT = wp.constant(TOTALS_WIDTH)
_HERE = Path(__file__).resolve()


# ----------------------------------------------------------------------------- ENV kernels (steps 4-5)

@wp.kernel
def _start_cut_b(progress: wp.array(dtype=float), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                 previous: wp.array(dtype=int), env_of_particle: wp.array(dtype=int), T: int):
    """tlmpm_v5._start_cut with cuts[e*T + index] and previous[e]."""
    p = wp.tid()
    e = env_of_particle[p]
    active = cuts[e*T+index[0]]
    if active >= 0 and active != previous[e]:
        progress[p] = 0.0


@wp.kernel
def _remember_cut_b(cuts: wp.array(dtype=int), index: wp.array(dtype=int), previous: wp.array(dtype=int),
                    T: int):
    """tlmpm_v5._remember_cut, one thread per env."""
    e = wp.tid()
    active = cuts[e*T+index[0]]
    if active >= 0:
        previous[e] = active


# ----------------------------------------------------------------------------- ENV kernels (steps 8-9)

@wp.kernel
def _segment_motion_b(poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                      cuts: wp.array(dtype=int), index: wp.array(dtype=int), edge_width: float, segments: int,
                      min_speed: float, tan2_max: float, seg_ok: wp.array(dtype=int),
                      any_motion: wp.array(dtype=int), T: int):
    """tlmpm_v5._segment_motion launched (N, segments)."""
    e, k = wp.tid()
    i = e*T+index[0]
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
        wp.atomic_max(any_motion, e, 1)
    seg_ok[e*segments+k] = ok


@wp.kernel
def _segment_plow_b(poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                    index: wp.array(dtype=int), edge_width: float, segments: int, tan2_plow: float,
                    seg_ok: wp.array(dtype=int), seg_plow: wp.array(dtype=int), T: int):
    """tlmpm_v5._segment_plow launched (N, segments)."""
    e, k = wp.tid()
    i = e*T+index[0]
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
    if seg_ok[e*segments+k] != 0 and plowing:
        ok = 1
    seg_plow[e*segments+k] = ok


# ----------------------------------------------------------------------------- ENV kernel (step 10)

@wp.kernel
def _rigid_contacts_b(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                      mass: wp.array(dtype=float), impulse: wp.array(dtype=wp.vec3),
                      progress: wp.array(dtype=float), contact_work: wp.array(dtype=float),
                      poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                      cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                      reaction: wp.array(dtype=wp.spatial_vector), stats_sum: wp.array(dtype=float),
                      stats_max: wp.array(dtype=float), radius: float, volume: float, dt: float,
                      params: wp.array2d(dtype=float), profile: wp.array(dtype=wp.vec2),
                      profile_count: int, blade_width: float, tool: int,
                      zone: float, zone_rear: float,
                      segments: int, seg_ok: wp.array(dtype=int), engaged: wp.array(dtype=int),
                      totals_sum: wp.array(dtype=wp.float64), totals_max: wp.array(dtype=wp.float64),
                      env_of_particle: wp.array(dtype=int), T: int):
    """tlmpm_v7_2_contacts._rigid_contacts (law-4 contact path, floor=False), env-indexed.

    The frozen step always passes law=4 (laws 5 and 8 use the law-4 contact path) and the batch refuses
    floor=True, so the law-1 branch, the law!=4 flag branch and the tray loop are not reachable and are
    omitted; every reachable statement is the frozen one in the frozen order."""
    p = wp.tid()
    e = env_of_particle[p]
    stiffness = params[e, P_STIFFNESS]
    friction = params[e, P_FRICTION]
    gc = params[e, P_GC]
    sharpness = params[e, P_SHARPNESS]
    slice_efficiency = params[e, P_SLICE]
    impulse[p] = wp.vec3(0.0)
    xp, vp, mp = x[p], v[p], mass[p]
    if tool != 0:
        i = e*T+index[0]
        pose = poses[i]
        local = wp.transform_point(wp.transform_inverse(pose), xp)
        hit = v5._profile_sdf(local, profile, profile_count, blade_width)
        normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
        gap = hit[3]-radius
        twist = twists[i]
        # This module's twist convention is [linear, angular], unlike wp.spatial.
        linear = wp.vec3(twist[0], twist[1], twist[2])
        angular = wp.vec3(twist[3], twist[4], twist[5])
        lever = xp-wp.transform_get_translation(pose)-normal*radius
        relative = vp-linear-wp.cross(angular, lever)
        ji = V._contact_impulse(gap, normal, relative, mp, dt, stiffness, friction)
        advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
        uc = wp.dot(-relative, advance)
        in_zone = (local[0] >= -zone_rear and local[0] <= 2.0*radius and wp.abs(local[2]) <= zone+radius
                   and wp.abs(local[1]) <= blade_width)
        if cuts[i] >= 0 and in_zone:
            along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
            us = wp.dot(-relative, along)
            slice_dir = along*wp.sign(us)
            push = wp.max(wp.dot(ji, advance), 0.0)
            drag = wp.max(wp.dot(ji, slice_dir), 0.0)
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
                wp.atomic_add(totals_sum, e*TT+5, wp.float64(fraction*dw))
                progress[p] = wp.min(progress[p]+fraction*dw/wp.max(threshold, 1.0e-12), 1.0)
                # Material already cut no longer resists in-plane edge motion.
                if local[0] >= -0.0005 and wp.abs(local[2]) <= 0.00025+radius:
                    ji = ji-(1.0-fraction)*(push*advance+drag*slice_dir)
            # Law 4 engagement W (the frozen step passes law=4).
            if dw > 1.0e-20 and seg_ok[e*segments+v5._edge_segment(local[1], blade_width, segments)] != 0:
                wp.atomic_max(engaged, e, 1)
        if wp.length_sq(ji) > 0.0:
            jn = wp.dot(ji, normal)
            wp.atomic_add(totals_sum, e*TT+0, wp.float64(jn))
            wp.atomic_add(totals_sum, e*TT+1, wp.float64(wp.length(ji-jn*normal)))
            partition = int(3)
            if local[0] >= -zone_rear and local[0] <= 2.0*radius:
                partition = 2
            wp.atomic_add(totals_sum, e*TT+partition, wp.float64(wp.length(ji)))
            wp.atomic_add(totals_sum, e*TT+4, wp.float64(wp.dot(ji, linear+wp.cross(angular, lever))))
            wp.atomic_add(totals_sum, e*TT+9, wp.float64(wp.dot(ji, vp)+0.5*wp.length_sq(ji)/mp))
            wp.atomic_max(totals_max, e*TT+11, wp.float64(wp.max(-gap, 0.0)))
        if ji[0] != 0.0 or ji[1] != 0.0 or ji[2] != 0.0:
            impulse[p] = impulse[p]+ji
            torque = wp.cross(lever, -ji)
            wp.atomic_add(reaction, e, wp.spatial_vector(-ji[0], -ji[1], -ji[2], torque[0], torque[1], torque[2]))
        if wp.length_sq(ji) > 0.0:
            wp.atomic_add(stats_sum, e*ST+1, 1.0)
            wp.atomic_add(stats_sum, e*ST+2, wp.dot(ji, linear+wp.cross(angular, lever)))
            wp.atomic_max(stats_max, e*ST+3, wp.max(-gap, 0.0))


# ----------------------------------------------------------------------------- ENV kernels (steps 11-12)

@wp.kernel
def _obstacle_contacts_b(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                         impulse: wp.array(dtype=wp.vec3), poses: wp.array(dtype=wp.transform),
                         twists: wp.array(dtype=wp.spatial_vector), index: wp.array(dtype=int),
                         reaction: wp.array(dtype=wp.spatial_vector), radius: float, dt: float,
                         params: wp.array2d(dtype=float), box_centers: wp.array(dtype=wp.vec3),
                         box_halves: wp.array(dtype=wp.vec3), box: int, boxes: int,
                         box_reactions: wp.array(dtype=wp.spatial_vector),
                         env_of_particle: wp.array(dtype=int), T: int):
    """tlmpm_v5._obstacle_contacts for box ``box`` of every env (one launch per box, as frozen)."""
    p = wp.tid()
    e = env_of_particle[p]
    stiffness = params[e, P_STIFFNESS]
    friction = params[e, P_FRICTION]
    xp, vp, mp = x[p], v[p], mass[p]
    i = e*T+index[0]
    pose = poses[i]
    local = wp.transform_point(wp.transform_inverse(pose), xp)
    b = e*boxes+box
    hit = v5._box_sdf(local-box_centers[b], box_halves[b])
    normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
    twist = twists[i]
    linear = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    lever = xp-wp.transform_get_translation(pose)-normal*radius
    relative = vp-linear-wp.cross(angular, lever)
    jb = v5._contact_impulse(hit[3]-radius, normal, relative, mp, dt, stiffness, friction)
    if wp.length_sq(jb) > 0.0:
        impulse[p] = impulse[p]+jb
        torque = wp.cross(lever, -jb)
        wp.atomic_add(reaction, e, wp.spatial_vector(-jb[0], -jb[1], -jb[2], torque[0], torque[1], torque[2]))
        wp.atomic_add(box_reactions, b, wp.spatial_vector(-jb[0], -jb[1], -jb[2], torque[0], torque[1], torque[2]))


@wp.kernel
def _fixture_contacts_b(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                        mass: wp.array(dtype=float), impulse: wp.array(dtype=wp.vec3),
                        centers: wp.array(dtype=wp.vec3), halves: wp.array(dtype=wp.vec3),
                        fixture_offset: wp.array(dtype=int), planes: wp.array(dtype=wp.vec4),
                        radius: float, dt: float, params: wp.array2d(dtype=float),
                        stats_max: wp.array(dtype=float), totals_sum: wp.array(dtype=wp.float64),
                        totals_max: wp.array(dtype=wp.float64), env_of_particle: wp.array(dtype=int)):
    """tlmpm_v7.fixture_contacts with the env's box range and table plane."""
    p = wp.tid()
    e = env_of_particle[p]
    stiffness = params[e, P_STIFFNESS]
    friction = params[e, P_FRICTION]
    xp, vp, mp = x[p], v[p], mass[p]
    for box in range(fixture_offset[e], fixture_offset[e+1]):
        hit = v5._box_sdf(xp-centers[box], halves[box])
        normal = wp.vec3(hit[0], hit[1], hit[2])
        gap = hit[3]-radius
        ji = V._contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
        impulse[p] = impulse[p]+ji
        if wp.length_sq(ji) > 0.0:
            jn = wp.dot(ji, normal)
            wp.atomic_add(totals_sum, e*TT+6, wp.float64(jn))
            wp.atomic_add(totals_sum, e*TT+7, wp.float64(wp.length(ji-jn*normal)))
            wp.atomic_add(totals_sum, e*TT+8, wp.float64(wp.dot(ji, vp)+0.5*wp.length_sq(ji)/mp))
            wp.atomic_max(totals_max, e*TT+10, wp.float64(wp.max(-gap, 0.0)))
        wp.atomic_max(stats_max, e*ST+0, wp.max(-gap, 0.0))
    plane = planes[e]
    normal = wp.vec3(plane[0], plane[1], plane[2])
    gap = wp.dot(normal, xp)+plane[3]-radius
    ji = V._contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
    impulse[p] = impulse[p]+ji
    if wp.length_sq(ji) > 0.0:
        jn = wp.dot(ji, normal)
        wp.atomic_add(totals_sum, e*TT+6, wp.float64(jn))
        wp.atomic_add(totals_sum, e*TT+7, wp.float64(wp.length(ji-jn*normal)))
        wp.atomic_add(totals_sum, e*TT+8, wp.float64(wp.dot(ji, vp)+0.5*wp.length_sq(ji)/mp))
        wp.atomic_max(totals_max, e*TT+10, wp.float64(wp.max(-gap, 0.0)))
    wp.atomic_max(stats_max, e*ST+0, wp.max(-gap, 0.0))


# ----------------------------------------------------------------------------- ENV kernel (step 14)

@wp.kernel
def _particle_contacts_b(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
                         reference: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                         mass: wp.array(dtype=float), code: wp.array(dtype=int),
                         impulse: wp.array(dtype=wp.vec3), stats_sum: wp.array(dtype=float),
                         stats_max: wp.array(dtype=float), radius: float, reference_exclusion: float,
                         dt: float, params: wp.array2d(dtype=float), env_of_particle: wp.array(dtype=int)):
    """tlmpm_v5._particle_contacts with a grouped query restricted to the particle's env."""
    p = wp.tid()
    e = env_of_particle[p]
    stiffness = params[e, P_PARTICLE_STIFFNESS]
    friction = params[e, P_FRICTION]
    query = wp.hash_grid_query(grid, x[p], 2.4*radius, e)
    for q in query:
        if q > p:
            if v5._conflict_packed(code[p], code[q]) or wp.length_sq(reference[p]-reference[q]) > reference_exclusion**2.0:
                delta = x[p]-x[q]
                distance = wp.length(delta)
                if distance > 1.0e-10:
                    normal = delta/distance
                    effective = mass[p]*mass[q]/(mass[p]+mass[q])
                    ji = v5._contact_impulse(distance-2.0*radius, normal, v[p]-v[q], effective,
                                             dt, stiffness, friction)
                    wp.atomic_add(impulse, p, ji)
                    wp.atomic_sub(impulse, q, ji)
                    if wp.length_sq(ji) > 0.0:
                        wp.atomic_add(stats_sum, e*ST+4, 1.0)
                        wp.atomic_add(stats_sum, e*ST+5, wp.length(ji))
                        wp.atomic_max(stats_max, e*ST+6, wp.max(2.0*radius-distance, 0.0))


# ----------------------------------------------------------------------------- ENV kernel (step 16)

@wp.kernel
def _claim_slots_b(code: wp.array(dtype=int), nodes: wp.array2d(dtype=int), slot_code: wp.array(dtype=int),
                   slot: wp.array2d(dtype=int), slots: int, overflow: wp.array(dtype=int),
                   max_occupied: wp.array(dtype=int), env_of_particle: wp.array(dtype=int)):
    """tlmpm_v5._claim_slots; only the integer overflow/max counters are per env."""
    p, corner = wp.tid()
    c = code[p]
    base = nodes[p, corner]*slots
    found = int(-1)
    for s in range(slots):
        current = slot_code[base+s]
        if current == v5.EMPTY_SLOT:
            current = wp.atomic_cas(slot_code, base+s, v5.EMPTY_SLOT, c)
            if current == v5.EMPTY_SLOT:
                current = c
        if current == c:
            found = s
            break
    slot[p, corner] = found
    e = env_of_particle[p]
    if found < 0:
        wp.atomic_add(overflow, e, 1)
    else:
        wp.atomic_max(max_occupied, e, found+1)


# ----------------------------------------------------------------------------- ENV kernels (steps 26-28)

@wp.kernel
def _update_stroke_b(twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int),
                     index: wp.array(dtype=int), engaged: wp.array(dtype=int), any_motion: wp.array(dtype=int),
                     stroke: wp.array(dtype=int), travel: wp.array(dtype=float), dt: float, persist: float, T: int):
    """tlmpm_v5._update_stroke, one thread per env."""
    e = wp.tid()
    i = e*T+index[0]
    cut = cuts[i]
    if cut >= 0 and any_motion[e] != 0:
        if engaged[e] != 0:
            stroke[e] = cut+1
            travel[e] = 0.0
        elif stroke[e] == cut+1:
            twist = twists[i]
            travel[e] = travel[e]+wp.length(wp.vec3(twist[0], twist[1], twist[2]))*dt
            if travel[e] > persist:
                stroke[e] = 0
        else:
            stroke[e] = 0
    else:
        stroke[e] = 0
        travel[e] = 0.0


@wp.kernel
def _assign_front_b(x0: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
                    codes: wp.array(dtype=int), previous: wp.array(dtype=wp.transform),
                    poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                    cuts: wp.array(dtype=int), index: wp.array(dtype=int), stroke: wp.array(dtype=int),
                    seg_ok: wp.array(dtype=int), width: float, band: float, segments: int,
                    max_cuts: int, crossed: wp.array(dtype=int), reached: wp.array(dtype=int),
                    bevel_depth: float, env_of_particle: wp.array(dtype=int), T: int):
    """tlmpm_v7._assign_front, env-indexed."""
    p = wp.tid()
    e = env_of_particle[p]
    i = e*T+index[0]
    cut = cuts[i]
    so = e*segments
    if cut >= 0 and cut < max_cuts and v5._digit(codes[p], cut) == 0:
        a = wp.transform_point(wp.transform_inverse(previous[e]), x0[p])
        b = wp.transform_point(wp.transform_inverse(poses[i]), x[p])
        # Actual relative sweep across the leading plane; no rear/lead closure.
        if a[0] >= 0.0 and b[0] <= 0.0 and a[0] > b[0] and v5._digit(reached[p], cut) == 0:
            t = a[0] / (a[0]-b[0])
            y = a[1] + t*(b[1]-a[1])
            side = v5._swept_side(x[p], poses[i], twists[i], b)
            if wp.abs(y) <= width and wp.abs(side) <= band:
                if seg_ok[so+v5._edge_segment(y, width, segments)] != 0:
                    digit = int(2)
                    if side < 0.0:
                        digit = 1
                    reached[p] = reached[p] | (digit << (2*cut))
                    crossed[p] = crossed[p]+1
        d = v5._digit(reached[p], cut)
        if d != 0 and stroke[e] == cut+1 and b[0] <= 0.0 and b[0] >= -bevel_depth:
            if wp.abs(b[1]) <= width and seg_ok[so+v5._edge_segment(b[1], width, segments)] != 0:
                codes[p] = codes[p] | (d << (2*cut))


@wp.kernel
def _remember_pose_b(previous: wp.array(dtype=wp.transform), poses: wp.array(dtype=wp.transform),
                     index: wp.array(dtype=int), T: int):
    """tlmpm_v7._remember_pose, one thread per env."""
    e = wp.tid()
    previous[e] = poses[e*T+index[0]]


# ----------------------------------------------------------------------------- ENV kernel (step 30)

@wp.kernel
def _complete_links_b(start: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
                      velocity: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                      old_codes: wp.array(dtype=int), codes: wp.array(dtype=int),
                      reached: wp.array(dtype=int), nodes: wp.array2d(dtype=int),
                      weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
                      offsets: wp.array(dtype=int), neighbours: wp.array(dtype=int),
                      poses: wp.array(dtype=wp.transform), previous: wp.array(dtype=wp.transform),
                      twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int),
                      index: wp.array(dtype=int),
                      stroke: wp.array(dtype=int), segment_ok: wp.array(dtype=int), segments: int,
                      width: float, band: float, bevel: float, half_thickness: float,
                      profile: wp.array(dtype=wp.vec2), profile_count: int, radius: float,
                      dt: float, params: wp.array2d(dtype=float),
                      pending_q: wp.array(dtype=int), pending_cut: wp.array(dtype=int),
                      diagnostics: wp.array(dtype=int), env_of_particle: wp.array(dtype=int), T: int):
    """tlmpm_v7_2_frontier._complete_links (hoisted interface), env-indexed.

    Node ids, CSR offsets, neighbours and pending ids are global; each env's CSR only references that env's
    particles (layout.range_audit), so the traversal order equals the single-env order."""
    p = wp.tid()
    e = env_of_particle[p]
    friction = params[e, P_FRICTION]
    slice_efficiency = params[e, P_SLICE]
    so = e*segments
    i = e*T+index[0]
    cut = cuts[i]
    if cut < 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    if v5._digit(old_codes[p], cut) != 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    d = v5._digit(reached[p], cut)
    if d == 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    twist = twists[i]
    if wp.length_sq(wp.vec3(twist[3], twist[4], twist[5]))*dt*dt > 1.e-8:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    pose = poses[i]
    inverse = wp.transform_inverse(pose)
    local = wp.transform_point(inverse, x[p])
    side = v5._swept_side(x[p], pose, twist, local)
    if wp.abs(side) > band or (d == 1 and side >= 0.0) or (d == 2 and side < 0.0):
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    q = int(-1)
    if pending_cut[p] != cut:
        pending_q[p] = -1
        pending_cut[p] = -1
    if pending_cut[p] == cut:
        q = pending_q[p]
        if q >= 0:
            here = _interface(x[p], x[q], poses[i], twist)
            if here[3] == 0.0 or here[0] < -bevel or here[0] > 0.0 or wp.abs(here[1]) > width:
                q = -1
                pending_q[p] = -1
                pending_cut[p] = -1
    if q < 0:
        previous_pose = previous[e]
        previous_inverse = wp.transform_inverse(previous_pose)
        previous_local = wp.transform_point(previous_inverse, start[p])
        # Both interfaces intentionally use the current substep's twist.
        previous_side = v5._swept_side(start[p], previous_pose, twist, previous_local)
        for node_index in range(8):
            if weights[p, node_index] > 0.0 or wp.length_sq(gradients[p, node_index]) > 0.0:
                node = nodes[p, node_index]
                for j in range(offsets[node], offsets[node+1]):
                    other = neighbours[j]
                    if other != p and not v5._conflict_packed(old_codes[p], old_codes[other]):
                        new = Fr._interface_from_p(local, side, x[other], pose, inverse, twist)
                        if (new[3] != 0.0 and new[0] <= 0.0 and new[0] >= -bevel
                                and wp.abs(new[1]) <= width):
                            old = Fr._interface_from_p(previous_local, previous_side, start[other],
                                                       previous_pose, previous_inverse, twist)
                            if old[3] != 0.0 and old[0] >= 0.0 and old[0] > new[0]:
                                t = old[0]/(old[0]-new[0])
                                y = old[1]+t*(new[1]-old[1])
                                if wp.abs(y) <= width and segment_ok[so+v5._edge_segment(y, width, segments)] != 0:
                                    q = other
                                    pending_q[p] = q
                                    pending_cut[p] = cut
                                    wp.atomic_add(diagnostics, e*2+0, 1)
                                    break
            if q >= 0:
                break
    if q >= 0 and stroke[e] == cut+1:
        here = _interface(x[p], x[q], poses[i], twist)
        if here[3] != 0.0 and here[0] <= 0.0 and here[0] >= -bevel and wp.abs(here[1]) <= width:
            if segment_ok[so+v5._edge_segment(here[1], width, segments)] != 0:
                dw = _endpoint_work(q, start, velocity, mass, poses[i], twist, profile, profile_count,
                                    width, radius, dt, friction, bevel, half_thickness, slice_efficiency)
                if dw > 1.e-20:
                    codes[p] = codes[p] | (d << (2*cut))
                    pending_q[p] = -1
                    pending_cut[p] = -1
                    wp.atomic_add(diagnostics, e*2+1, 1)


# ----------------------------------------------------------------------------- solver

def source_sha256():
    return {"tlmpm_v7_2_batch.py": hashlib.sha256(_HERE.read_bytes()).hexdigest(),
            "constitutive.py": hashlib.sha256(_HERE.with_name("constitutive.py").read_bytes()).hexdigest(),
            "layout.py": hashlib.sha256(_HERE.with_name("layout.py").read_bytes()).hexdigest()}


class BatchedTLMPMv7_2:
    """N independent tlmpm_v7_2 envs in flat batch arrays; ``step()`` is one substep of every env.

    Built from a ``layout.PackedBatch``. Arrays keep the single solver's names; per-env arrays are
    ``[N*k]`` rows; ``stats``/``contact_totals`` are ``stats_sum``/``stats_max`` and
    ``contact_totals_sum``/``contact_totals_max``. ``snapshot(e)`` maps env e back to single-env form."""
    solver_name = SOLVER_NAME

    # dtype of every array by name (host arrays are numpy; these are the Warp dtypes).
    _DTYPES = {
        **dict.fromkeys(("x", "v", "reference", "impulse", "x_start", "link_velocity_start", "gp", "gj", "old",
                         "new_v", "box_centers", "box_halves", "fixture_centers", "fixture_halves"), wp.vec3),
        **dict.fromkeys(("F", "Fpi", "stress"), wp.mat33),
        **dict.fromkeys(("mass", "plastic", "progress", "contact_work", "weights", "gm", "stroke_travel",
                         "stats_sum", "stats_max"), float),
        **dict.fromkeys(("code", "nodes", "slot", "front_crossings", "front_reached", "link_codes", "pending_q",
                         "pending_cut", "slot_code", "edge_flags", "segment_ok", "segment_plow", "stroke_engaged",
                         "stroke_any_motion", "stroke", "slot_overflow", "slot_max_occupied", "previous_cut",
                         "cuts", "link_diagnostics", "link_offsets", "link_neighbours", "index",
                         "env_of_particle", "particle_offset", "node_offset", "fixture_offset"), int),
        **dict.fromkeys(("contact_totals_sum", "contact_totals_max"), wp.float64),
        **dict.fromkeys(("reaction", "box_reactions", "twists"), wp.spatial_vector),
        **dict.fromkeys(("poses", "previous_pose", "link_previous_pose"), wp.transform),
        "gradients": wp.vec3, "profile": wp.vec2, "fixture_plane": wp.vec4,
    }

    def __init__(self, packed, device):
        self.packed = packed
        self.device = str(wp.get_device(device))
        u = packed.uniform
        self.num_envs = int(packed.num_envs)
        self.total_particles = int(packed.total_particles)
        self.total_nodes = int(packed.total_nodes)
        self.slots_per_node = int(u["slots"])
        self.trajectory_length = int(u["trajectory"])
        self.substeps = int(u["substeps"])
        self.edge_segments = int(u["segments"])
        self.boxes = int(u["boxes"])
        self.obstacle_count = int(u["obstacle_count"])
        self.max_cuts = int(u["max_cuts"])
        self.dt, self.dx = float(u["dt"]), float(u["dx"])
        self.radius, self.volume = float(u["radius"]), float(u["volume"])
        self.pic = float(u["pic"])
        self.gravity = wp.vec3(*u["gravity"])
        self.tool, self.material_contact = bool(u["tool"]), bool(u["material_contact"])
        self.fixtures = bool(u["fixtures"])
        self.blade_half_width = float(u["blade_half_width"])
        self.bevel_depth_m = float(u["bevel_depth_m"])
        self.blade_half_thickness_m = float(u["blade_half_thickness_m"])
        self.law4_parameters = dict(u["law4_parameters"])
        self.law8_parameters = dict(u["law8_parameters"])
        self.profile_count = len(packed.host["profile"])
        self.tan2_max = float(np.tan(np.deg2rad(self.law4_parameters["theta_max_deg"]))**2)
        self.tan2_plow = float(np.tan(np.deg2rad(self.law8_parameters["theta_plow_deg"]))**2)
        self.model = constitutive.get_model(packed.constitutive_model)
        if u["cutting_law"] != 8:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_SOLVER", f"cutting law {u['cutting_law']}")
        if u["floor"]:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_LEGACY_FLOOR", "floor=True tray")
        if not (u["optimize_contacts"] and u["optimize_frontier"]):
            raise UnsupportedBatchConfiguration("UNSUPPORTED_V7_2_OPTIMIZATIONS_DISABLED", "batch")
        h = packed.host
        names = (PARTICLE_ARRAYS + NODE_SLOT_ARRAYS + tuple(ENV_ARRAYS) + LINK_ARRAYS
                 + ("stats_sum", "stats_max", "contact_totals_sum", "contact_totals_max", "profile", "index",
                    "fixture_centers", "fixture_halves", "fixture_plane", "env_of_particle", "particle_offset",
                    "node_offset"))
        with wp.ScopedDevice(self.device):
            for name in names:
                setattr(self, name, wp.array(h[name], dtype=self._DTYPES[name]))
            self.fixture_offset = wp.array(packed.fixture_offset.astype(np.int32), dtype=int)
            self.params = wp.array(h["params"], dtype=float)
            self.hash = wp.HashGrid(64, 64, 32)
            # Reserve grouped-build buffers once (before any graph capture), then build like the frozen
            # constructor does (tlmpm_v5.py:798-799), here with groups.
            self.hash.reserve(self.total_particles, with_groups=True)
            self.hash.build(self.x, 2.4*self.radius, groups=self.env_of_particle)
        for module in (v5.__name__, "robotic_cutting_sim.tlmpm_v6", __name__, constitutive.__name__):
            wp.load_module(module=module, device=self.device)

    # ------------------------------------------------------------------ substep (frozen launch order)

    def step(self, advance=True):
        """One substep of every env: ``tlmpm_v7_2.TLMPM.step`` with optimized contacts and frontier.

        Order: copy link_previous_pose, copy link_velocity_start; tlmpm_v7_2_contacts.step (GRID_X copy,
        start_cut, remember_cut, stress, zero flags, segment motion/plow, rigid contacts, obstacle boxes,
        fixtures, grouped hash + particle contacts, slots, P2G, grid update, G2P, MUSL, grid update,
        deformation, grid positions, stroke, front, remember pose); copy link_codes; frontier; advance."""
        n, S, T = self.total_particles, self.slots_per_node, self.trajectory_length
        N, segs = self.num_envs, self.edge_segments
        dt, r = self.dt, self.radius
        with wp.ScopedDevice(self.device):
            wp.copy(self.link_previous_pose, self.previous_pose)
            wp.copy(self.link_velocity_start, self.v)
            # ---- tlmpm_v7_2_contacts.step(advance=False)
            wp.copy(self.x_start, self.x)
            wp.launch(_start_cut_b, n, [self.progress, self.cuts, self.index, self.previous_cut,
                                         self.env_of_particle, T])
            wp.launch(_remember_cut_b, N, [self.cuts, self.index, self.previous_cut, T])
            self.model.launch_stress(self)
            self.edge_flags.zero_()
            self.stroke_engaged.zero_(); self.stroke_any_motion.zero_()
            wp.launch(_segment_motion_b, (N, segs), [self.poses, self.twists, self.cuts, self.index,
                self.blade_half_width, segs, self.law4_parameters["min_speed_m_s"], self.tan2_max,
                self.segment_ok, self.stroke_any_motion, T])
            wp.launch(_segment_plow_b, (N, segs), [self.poses, self.twists, self.index, self.blade_half_width,
                segs, self.tan2_plow, self.segment_ok, self.segment_plow, T])
            wp.launch(_rigid_contacts_b, n, [self.x, self.v, self.mass, self.impulse, self.progress,
                self.contact_work, self.poses, self.twists, self.cuts, self.index, self.reaction,
                self.stats_sum, self.stats_max, r, self.volume, dt, self.params, self.profile,
                self.profile_count, self.blade_half_width, int(self.tool), self.blade_half_thickness_m,
                self.bevel_depth_m, segs, self.segment_ok, self.stroke_engaged,
                self.contact_totals_sum, self.contact_totals_max, self.env_of_particle, T])
            if self.tool:
                for box in range(self.obstacle_count):
                    wp.launch(_obstacle_contacts_b, n, [self.x, self.v, self.mass, self.impulse, self.poses,
                        self.twists, self.index, self.reaction, r, dt, self.params, self.box_centers,
                        self.box_halves, box, self.boxes, self.box_reactions, self.env_of_particle, T])
            if self.fixtures:
                wp.launch(_fixture_contacts_b, n, [self.x, self.v, self.mass, self.impulse, self.fixture_centers,
                    self.fixture_halves, self.fixture_offset, self.fixture_plane, r, dt, self.params,
                    self.stats_max, self.contact_totals_sum, self.contact_totals_max, self.env_of_particle])
            if self.material_contact:
                self.hash.build(self.x, 2.4*r, groups=self.env_of_particle)
                wp.launch(_particle_contacts_b, n, [self.hash.id, self.x, self.reference, self.v, self.mass,
                    self.code, self.impulse, self.stats_sum, self.stats_max, r, 3*self.dx, dt, self.params,
                    self.env_of_particle])
            self.slot_code.fill_(-1)
            wp.launch(_claim_slots_b, (n, 8), [self.code, self.nodes, self.slot_code, self.slot, S,
                self.slot_overflow, self.slot_max_occupied, self.env_of_particle])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse,
                self.nodes, self.weights, self.gradients, self.gm, self.gp, self.gj,
                S, self.volume, dt, self.gravity])
            wp.launch(v5._grid_update, (S, self.total_nodes), [self.gm, self.gp, self.gj,
                self.old, self.new_v, self.slot_code, S])
            wp.launch(v5._g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights,
                self.old, self.new_v, S, dt, self.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes, self.weights, self.gp, S])
            wp.launch(v5._grid_update, (S, self.total_nodes), [self.gm, self.gp, self.gj,
                self.old, self.new_v, self.slot_code, S])
            wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients, self.new_v, S, dt])
            wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights,
                self.new_v, S, dt])
            law4 = self.law4_parameters
            wp.launch(_update_stroke_b, N, [self.twists, self.cuts, self.index, self.stroke_engaged,
                self.stroke_any_motion, self.stroke, self.stroke_travel, dt, law4["persist_b_persist_m"], T])
            wp.launch(_assign_front_b, n, [self.x_start, self.x, self.code, self.previous_pose,
                self.poses, self.twists, self.cuts, self.index, self.stroke, self.segment_ok,
                self.blade_half_width, law4["band_h_m"], segs, self.max_cuts, self.front_crossings,
                self.front_reached, self.bevel_depth_m, self.env_of_particle, T])
            wp.launch(_remember_pose_b, N, [self.previous_pose, self.poses, self.index, T])
            # ---- end of contacts step
            wp.copy(self.link_codes, self.code)
            wp.launch(_complete_links_b, n, [self.x_start, self.x, self.link_velocity_start, self.mass,
                self.link_codes, self.code, self.front_reached, self.nodes, self.weights, self.gradients,
                self.link_offsets, self.link_neighbours, self.poses, self.link_previous_pose, self.twists,
                self.cuts, self.index, self.stroke, self.segment_ok, segs, self.blade_half_width,
                law4["band_h_m"], self.bevel_depth_m, self.blade_half_thickness_m, self.profile,
                self.profile_count, r, dt, self.params, self.pending_q, self.pending_cut,
                self.link_diagnostics, self.env_of_particle, T])
            if advance:
                wp.launch(v5._advance, 1, [self.index])

    # ------------------------------------------------------------------ bookkeeping

    def zero_control_step_accumulators(self):
        """What ``LiveCuttingMaterial.advance`` zeroes before each control step, for all envs."""
        self.index.zero_()
        self.reaction.zero_()
        self.box_reactions.zero_()
        self.stats_sum.zero_()
        self.stats_max.zero_()

    def memory_bytes(self):
        """Bytes of every device array owned by this solver (excluding the hash grid), as tlmpm_v5."""
        return int(sum(value.capacity for value in vars(self).values() if isinstance(value, wp.array)))

    def device_arrays(self):
        return {name: value for name, value in vars(self).items() if isinstance(value, wp.array)}

    def snapshot(self, e):
        """Every array of env ``e`` in the single solver's names, shapes and id spaces (host copies).

        ``nodes`` become local node ids, ``link_offsets``/``link_neighbours``/``pending_q`` local ids,
        ``stats``/``contact_totals`` are recomposed from the sum and max arrays, the node-slot arrays are the
        env's node range, and the fixture data is returned as (centers, halves, plane)."""
        pk = self.packed
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e+1])
        n0, n1 = int(pk.node_offset[e]), int(pk.node_offset[e+1])
        l0, l1 = int(pk.link_offset[e]), int(pk.link_offset[e+1])
        f0, f1 = int(pk.fixture_offset[e]), int(pk.fixture_offset[e+1])
        S = self.slots_per_node
        out = {}
        for name in PARTICLE_ARRAYS:
            value = getattr(self, name).numpy()[p0:p1].copy()
            if name == "nodes":
                value = value-np.int32(n0)
            elif name == "pending_q":
                value = np.where(value >= 0, value-np.int32(p0), value).astype(np.int32)
            out[name] = value
        for name in NODE_SLOT_ARRAYS:
            out[name] = getattr(self, name).numpy()[n0*S:n1*S].copy()
        widths = {"segments": self.edge_segments, "boxes": self.boxes, "trajectory": self.trajectory_length}
        for name, width in ENV_ARRAYS.items():
            k = widths.get(width, width)
            out[name] = getattr(self, name).numpy()[e*k:(e+1)*k].copy()
        for name, width, sums, maxima in (("stats", STATS_WIDTH, STATS_SUM, STATS_MAX),
                                          ("contact_totals", TOTALS_WIDTH, TOTALS_SUM, TOTALS_MAX)):
            total = getattr(self, f"{name}_sum").numpy()[e*width:(e+1)*width]
            maximum = getattr(self, f"{name}_max").numpy()[e*width:(e+1)*width]
            value = np.zeros_like(total)
            value[list(sums)] = total[list(sums)]
            value[list(maxima)] = maximum[list(maxima)]
            out[name] = value
        out["link_offsets"] = self.link_offsets.numpy()[n0:n1+1]-np.int32(l0)
        out["link_neighbours"] = self.link_neighbours.numpy()[l0:l1]-np.int32(p0)
        out["index"] = self.index.numpy().copy()
        out["profile"] = self.profile.numpy().copy()
        out["fixture_data"] = (self.fixture_centers.numpy()[f0:f1].copy(), self.fixture_halves.numpy()[f0:f1].copy(),
                               self.fixture_plane.numpy()[e].copy())
        return out

    def solver_receipt(self):
        return {"solver": SOLVER_NAME, "base_solver": BASE_SOLVER, "status": STATUS,
                "execution": "batched_flat_soa_env_kernels_v1", "contact_execution": "skip_zero_reaction_atomics_v1",
                "frontier_execution": "hoisted_interface_v1", "particle_contact_hash": "warp_grouped_hashgrid",
                "constitutive_model": self.model.receipt(), "source_sha256": source_sha256(),
                "frozen_source_sha256": frozen_source_sha256(), "device": self.device,
                "warp_version": wp.config.version}
