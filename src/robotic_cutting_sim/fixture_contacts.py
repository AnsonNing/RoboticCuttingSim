"""Finite workpiece-local fixture boxes and a world-table plane for live tasks.

Uses the retained solver's contact law. These contacts add impulses to physical
particles; they neither constrain positions nor assign cutting history.
"""
import warp as wp

from .tlmpm import _box_sdf, _contact_impulse


@wp.kernel
def fixture_contacts(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                     mass: wp.array(dtype=float), impulse: wp.array(dtype=wp.vec3),
                     centers: wp.array(dtype=wp.vec3), halves: wp.array(dtype=wp.vec3),
                     plane: wp.vec4, radius: float, dt: float,
                     stiffness: float, friction: float, stats: wp.array(dtype=float)):
    p = wp.tid()
    xp, vp, mp = x[p], v[p], mass[p]
    for box in range(centers.shape[0]):
        hit = _box_sdf(xp-centers[box], halves[box])
        normal = wp.vec3(hit[0], hit[1], hit[2])
        gap = hit[3]-radius
        impulse[p] = impulse[p]+_contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
        wp.atomic_max(stats, 0, wp.max(-gap, 0.0))
    normal = wp.vec3(plane[0], plane[1], plane[2])
    gap = wp.dot(normal, xp)+plane[3]-radius
    impulse[p] = impulse[p]+_contact_impulse(gap, normal, vp, mp, dt, stiffness, friction)
    wp.atomic_max(stats, 0, wp.max(-gap, 0.0))
