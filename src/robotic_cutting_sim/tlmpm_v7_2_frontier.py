"""Reuse particle transforms and reject current interfaces before old ones.

Traversal, interface arithmetic and pending evidence follow the original v7_2
kernel. No support entries are reordered or removed; the original kernel stays
available through ``optimize_frontier=False``.
"""
import warp as wp

from . import tlmpm_v5 as B
from .tlmpm_v7_1 import _interface, _endpoint_work


@wp.func
def _interface_from_p(a: wp.vec3, sa: float, q: wp.vec3, pose: wp.transform,
                      inverse: wp.transform, twist: wp.spatial_vector):
    b = wp.transform_point(inverse, q)
    sb = B._swept_side(q, pose, twist, b)
    result = wp.vec4(0.0)
    # Keep the multiplication: opposite signs alone would admit underflowed
    # products that the original interface rejects.
    if sa*sb < 0.0:
        point = a+(b-a)*(sa/(sa-sb))
        result = wp.vec4(point[0], point[1], point[2], 1.0)
    return result


@wp.kernel
def _complete_links(start: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
                    velocity: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
                    old_codes: wp.array(dtype=int), codes: wp.array(dtype=int),
                    reached: wp.array(dtype=int), nodes: wp.array2d(dtype=int),
                    weights: wp.array2d(dtype=float), gradients: wp.array2d(dtype=wp.vec3),
                    offsets: wp.array(dtype=int), neighbours: wp.array(dtype=int),
                    poses: wp.array(dtype=wp.transform), previous: wp.array(dtype=wp.transform),
                    twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                    stroke: wp.array(dtype=int), segment_ok: wp.array(dtype=int), segments: int,
                    width: float, band: float, bevel: float, half_thickness: float,
                    profile: wp.array(dtype=wp.vec2), profile_count: int, radius: float,
                    dt: float, friction: float, slice_efficiency: float,
                    pending_q: wp.array(dtype=int), pending_cut: wp.array(dtype=int),
                    diagnostics: wp.array(dtype=int)):
    p = wp.tid(); i = index[0]; cut = cuts[i]
    if cut < 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    if B._digit(old_codes[p],cut) != 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    d = B._digit(reached[p],cut)
    if d == 0:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    twist = twists[i]
    if wp.length_sq(wp.vec3(twist[3],twist[4],twist[5]))*dt*dt > 1.e-8:
        pending_q[p] = -1
        pending_cut[p] = -1
        return
    pose = poses[i]
    inverse = wp.transform_inverse(pose)
    local = wp.transform_point(inverse,x[p])
    side = B._swept_side(x[p],pose,twist,local)
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
            here = _interface(x[p],x[q],poses[i],twist)
            if here[3] == 0.0 or here[0] < -bevel or here[0] > 0.0 or wp.abs(here[1]) > width:
                q = -1
                pending_q[p] = -1
                pending_cut[p] = -1
    if q < 0:
        previous_pose = previous[0]
        previous_inverse = wp.transform_inverse(previous_pose)
        previous_local = wp.transform_point(previous_inverse,start[p])
        # Both interfaces intentionally use the current substep's twist.
        previous_side = B._swept_side(start[p],previous_pose,twist,previous_local)
        for node_index in range(8):
            if weights[p,node_index] > 0.0 or wp.length_sq(gradients[p,node_index]) > 0.0:
                node = nodes[p,node_index]
                for j in range(offsets[node],offsets[node+1]):
                    other = neighbours[j]
                    if other != p and not B._conflict_packed(old_codes[p],old_codes[other]):
                        new = _interface_from_p(local,side,x[other],pose,inverse,twist)
                        # These are the original positive acceptance predicates;
                        # an invalid current interface cannot select this pair.
                        if (new[3] != 0.0 and new[0] <= 0.0 and new[0] >= -bevel
                                and wp.abs(new[1]) <= width):
                            old = _interface_from_p(previous_local,previous_side,start[other],
                                                    previous_pose,previous_inverse,twist)
                            if old[3] != 0.0 and old[0] >= 0.0 and old[0] > new[0]:
                                t = old[0]/(old[0]-new[0])
                                y = old[1]+t*(new[1]-old[1])
                                if wp.abs(y) <= width and segment_ok[B._edge_segment(y,width,segments)] != 0:
                                    q = other
                                    pending_q[p] = q; pending_cut[p] = cut
                                    wp.atomic_add(diagnostics,0,1)
                                    break
            if q >= 0:
                break
    if q >= 0 and stroke[0] == cut+1:
        here = _interface(x[p],x[q],poses[i],twist)
        if here[3] != 0.0 and here[0] <= 0.0 and here[0] >= -bevel and wp.abs(here[1]) <= width:
            if segment_ok[B._edge_segment(here[1],width,segments)] != 0:
                dw = _endpoint_work(q,start,velocity,mass,poses[i],twist,profile,profile_count,
                                    width,radius,dt,friction,bevel,half_thickness,slice_efficiency)
                if dw > 1.e-20:
                    codes[p] = codes[p] | (d << (2*cut))
                    pending_q[p] = -1
                    pending_cut[p] = -1
                    wp.atomic_add(diagnostics,1,1)
