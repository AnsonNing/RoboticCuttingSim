"""Kinematically driven blade probes for tlmpm_v2 (no robot, no renderer).

Engineering diagnostics for the cutting law: a blade pose follows piecewise
constant velocities through a particle block, and the result reports labels,
mechanical components and the tool reaction. Runs on "cpu" or a CUDA device.
These probes do not replace robot-coupled or calibrated validation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .cut_topology import conflicting, reference_components
from .material_config import TLMPMConfig


def block(size_m=(.06, .024, .012), spacing=.004, density=1500.):
    """Lattice block with its bottom on z=0 and centered in x/y."""
    sx, sy, sz = size_m
    grid = np.mgrid[-sx/2:sx/2-1e-9:spacing, -sy/2:sy/2-1e-9:spacing, 0:sz-1e-9:spacing]
    points = grid.reshape(3, -1).T.astype(np.float32)+np.float32(spacing/2)
    return points, np.full(len(points), density*spacing**3, np.float32)


def severed_bonds(lattice, codes, cuts):
    """Reference 6-neighbor bonds whose two particles carry conflicting cut sides.

    Labels on one side only sever nothing; this counts actual separation.
    """
    lookup = {tuple(v): i for i, v in enumerate(lattice)}
    pairs = [(i, lookup[tuple(v+step)]) for i, v in enumerate(lattice)
             for step in np.eye(3, dtype=int) if tuple(v+step) in lookup]
    if not pairs:
        return 0
    a, b = np.asarray(pairs).T
    return int(conflicting(codes[a], codes[b], cuts).sum())


@dataclass(frozen=True)
class Phase:
    """Constant edge velocity (world, m/s) for a duration; cut digit or -1."""
    velocity_m_s: tuple
    duration_s: float
    cut: int = 0


def drive_blade(profile, rotation, start_m, phases, *, points=None, masses=None, spacing=.004,
                cutting_law=2, cutting_energy=5., slice_efficiency=1., cuts=1, obstacle_boxes_edge=None,
                device="cpu", chunk=200, config_overrides=None):
    """Drive the edge frame (rotation: edge->world Rotation) through phases.

    Returns per-phase label counts, final components, mean and peak tool
    reaction force (world, N) per phase, and the solver snapshot.
    """
    import warp as wp
    from .tlmpm_v2 import TLMPM

    if points is None:
        points, masses = block(spacing=spacing)
    polygon, half = profile.tlmpm_profile()
    config = TLMPMConfig(cuts=cuts, blade_half_width=float(half), cutting_energy=float(cutting_energy),
                         **(config_overrides or {}))
    bed = TLMPM(points, masses, spacing, config, device, blade_profile_xz=polygon, cutting_law=cutting_law,
                slice_efficiency=slice_efficiency, obstacle_boxes_edge=obstacle_boxes_edge)
    quaternion = rotation.as_quat()
    position = np.asarray(start_m, float)
    labels, forces = [], []
    for phase in phases:
        velocity = np.asarray(phase.velocity_m_s, float)
        steps = int(round(phase.duration_s/config.dt))
        reaction = []
        for first in range(0, steps, chunk):
            count = min(chunk, steps-first)
            poses = np.array([np.r_[position+velocity*config.dt*i, quaternion] for i in range(count+1)], np.float32)
            twists = np.tile(np.r_[velocity, 0., 0., 0.], (count+1, 1)).astype(np.float32)
            bed.set_trajectory(poses, twists, np.full(count+1, phase.cut, np.int32))
            for _ in range(count):
                bed.reaction.zero_()
                bed.step()
                reaction.append(bed.reaction.numpy()[0][:3].copy()/config.dt)
            position = position+velocity*config.dt*count
        wp.synchronize_device(device)
        labels.append(int(np.count_nonzero(bed.code.numpy())))
        reaction = np.asarray(reaction)
        forces.append({"mean_N": reaction.mean(0).tolist(), "peak_abs_N": np.abs(reaction).max(0).tolist()})
    state = bed.snapshot()
    lattice = np.round((points-points.min(0))/spacing).astype(int)
    count, component = reference_components(points, lattice, state["codes"], cuts=cuts)
    fractions = np.sort(np.bincount(component, weights=masses))[::-1]/masses.sum()
    return {"labelled_after_phase": labels, "components": int(count),
            "severed_bonds": severed_bonds(lattice, state["codes"], cuts),
            "mass_fractions": fractions.tolist(), "forces": forces,
            "edge_position_m": position.tolist(), "state": state, "points": points,
            "component": component}


VERTICAL_CUT = Rotation.from_euler("y", 90, degrees=True)  # advance -z, edge along y, normal +x
FLAT_SKIM = Rotation.identity()                             # advance +x, edge along y, normal +z
