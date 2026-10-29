"""Kinematically driven blade probes for tlmpm_v2 (no robot, no renderer).

Engineering diagnostics for the cutting law: a blade pose follows piecewise
constant velocities through a particle block, and the result reports labels,
mechanical components and the tool reaction. Runs on "cpu" or a CUDA device.
These probes do not replace robot-coupled or calibrated validation.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .cut_topology import (conflicting, conflicting_packed, decode_digits, reference_components,
                           reference_components_packed)
from .material_config import TLMPMConfig


def block(size_m=(.06, .024, .012), spacing=.004, density=1500.):
    """Lattice block with its bottom on z=0 and centered in x/y."""
    sx, sy, sz = size_m
    grid = np.mgrid[-sx/2:sx/2-1e-9:spacing, -sy/2:sy/2-1e-9:spacing, 0:sz-1e-9:spacing]
    points = grid.reshape(3, -1).T.astype(np.float32)+np.float32(spacing/2)
    return points, np.full(len(points), density*spacing**3, np.float32)


def severed_bonds(lattice, codes, cuts, *, packed=False):
    """Reference 6-neighbor bonds whose two particles carry conflicting cut sides.

    Labels on one side only sever nothing; this counts actual separation.
    ``packed`` selects the tlmpm_v5 packed-code conflict test (ADR 0014).
    """
    lookup = {tuple(v): i for i, v in enumerate(lattice)}
    pairs = [(i, lookup[tuple(v+step)]) for i, v in enumerate(lattice)
             for step in np.eye(3, dtype=int) if tuple(v+step) in lookup]
    if not pairs:
        return 0
    a, b = np.asarray(pairs).T
    if packed:
        return int(conflicting_packed(codes[a], codes[b]).sum())
    return int(conflicting(codes[a], codes[b], cuts).sum())


@dataclass(frozen=True)
class Phase:
    """Constant edge velocity (world, m/s) for a duration; cut digit or -1."""
    velocity_m_s: tuple
    duration_s: float
    cut: int = 0


def drive_blade(profile, rotation, start_m, phases, *, points=None, masses=None, spacing=.004,
                cutting_law=2, cutting_energy=5., slice_efficiency=1., cuts=1, obstacle_boxes_edge=None,
                device="cpu", chunk=200, config_overrides=None, gpu_graph=False, record_history=False,
                solver_version=None, slots_per_node=None):
    """Drive the edge frame (rotation: edge->world Rotation) through phases.

    ``solver_version`` None uses tlmpm_v4 for laws 4-8, tlmpm_v3 for law 3 and
    tlmpm_v2 otherwise; 3 or 4 forces that module (for equivalence checks of
    the older laws against tlmpm_v2/tlmpm_v3). ``solver_version`` 5 uses tlmpm_v5
    (ADR 0014: laws 4, 5, 8; packed codes, ``cuts`` up to 15, ``slots_per_node``
    default 8); its codes stay packed in the result, which then also carries
    ``code_encoding``, per-phase per-digit label counts and slot diagnostics,
    and slot overflow raises after the chunk where it occurred.

    Returns per-phase label counts, final components, mean and peak tool
    reaction force (world, N) per phase, and the solver snapshot.
    """
    import warp as wp
    if (solver_version not in (None, 2, 3, 4, 5) or (solver_version == 2 and cutting_law in (3, 4, 5, 6, 7, 8))
            or (solver_version == 3 and cutting_law in (4, 5, 6, 7, 8))
            or (solver_version == 5 and cutting_law not in (4, 5, 8))):
        raise ValueError('solver_version is 2, 3, 4 or 5; law 3 needs tlmpm_v3/v4, laws 4-8 tlmpm_v4, '
                         'and tlmpm_v5 runs laws 4, 5 and 8')
    if slots_per_node is not None and solver_version != 5:
        raise ValueError('slots_per_node applies to tlmpm_v5 only')
    packed = solver_version == 5
    if packed:
        from .tlmpm_v5 import TLMPM
    elif cutting_law in (4, 5, 6, 7, 8) or solver_version == 4:
        from .tlmpm_v4 import TLMPM
    elif cutting_law == 3 or solver_version == 3:
        from .tlmpm_v3 import TLMPM
    else:
        from .tlmpm_v2 import TLMPM

    if points is None:
        points, masses = block(spacing=spacing)
    polygon, half = profile.tlmpm_profile()
    config = TLMPMConfig(cuts=cuts, blade_half_width=float(half), cutting_energy=float(cutting_energy),
                         **(config_overrides or {}))
    bed = TLMPM(points, masses, spacing, config, device, blade_profile_xz=polygon, cutting_law=cutting_law,
                slice_efficiency=slice_efficiency, obstacle_boxes_edge=obstacle_boxes_edge,
                **({'slots_per_node': slots_per_node} if packed and slots_per_node is not None else {}))
    graphs = {}
    if gpu_graph:
        if not wp.get_device(device).is_cuda:
            raise ValueError('GPU graph probes require an actual CUDA device')
        from .probe_kernels import record_reaction
        wp.load_module(module=TLMPM.__module__, device=device)
        wp.load_module(module='robotic_cutting_sim.probe_kernels', device=device)
    quaternion = rotation.as_quat()
    position = np.asarray(start_m, float)
    labels, forces, histories, bonds_after_phase, strokes, digit_labels = [], [], [], [], [], []
    for phase in phases:
        velocity = np.asarray(phase.velocity_m_s, float)
        steps = int(round(phase.duration_s/config.dt))
        reaction = []
        for first in range(0, steps, chunk):
            count = min(chunk, steps-first)
            poses = np.array([np.r_[position+velocity*config.dt*i, quaternion] for i in range(count+1)], np.float32)
            twists = np.tile(np.r_[velocity, 0., 0., 0.], (count+1, 1)).astype(np.float32)
            digits = np.full(count+1, phase.cut, np.int32)
            if gpu_graph:
                if count not in graphs:
                    bed.set_trajectory(poses, twists, digits)
                    history = wp.empty((count, 3), dtype=float, device=device)
                    with wp.ScopedCapture(device=device) as capture:
                        for index in range(count):
                            bed.reaction.zero_()
                            bed.step()
                            wp.launch(record_reaction, dim=1, inputs=[bed.reaction, history, index, config.dt], device=device)
                    graphs[count] = (capture.graph, bed.poses, bed.twists, bed.cuts, history)
                graph, pose_buffer, twist_buffer, digit_buffer, history = graphs[count]
                pose_buffer.assign(poses)
                twist_buffer.assign(twists)
                digit_buffer.assign(digits)
                bed.index.zero_()
                wp.capture_launch(graph)
                reaction.extend(history.numpy())
            else:
                bed.set_trajectory(poses, twists, digits)
                for _ in range(count):
                    bed.reaction.zero_()
                    bed.step()
                    reaction.append(bed.reaction.numpy()[0][:3].copy()/config.dt)
            position = position+velocity*config.dt*count
            if packed:
                bed.check_slots()
        wp.synchronize_device(device)
        labels.append(int(np.count_nonzero(bed.code.numpy())))
        if packed:
            digit_labels.append(np.count_nonzero(decode_digits(bed.code.numpy(), cuts), axis=0).tolist())
        if cutting_law in (4, 5, 6, 7, 8):
            strokes.append(bed.cutting_state())
        reaction = np.asarray(reaction)
        if record_history:
            histories.append(reaction.copy())
            lattice = np.round((points-points.min(0))/spacing).astype(int)
            bonds_after_phase.append(severed_bonds(lattice, bed.code.numpy(), cuts, packed=packed))
        forces.append({"mean_N": reaction.mean(0).tolist(), "peak_abs_N": np.abs(reaction).max(0).tolist()})
    state = bed.snapshot()
    lattice = np.round((points-points.min(0))/spacing).astype(int)
    count, component = (reference_components_packed(points, lattice, state["codes"]) if packed
                        else reference_components(points, lattice, state["codes"], cuts=cuts))
    fractions = np.sort(np.bincount(component, weights=masses))[::-1]/masses.sum()
    return {"labelled_after_phase": labels, "components": int(count),
            "severed_bonds": severed_bonds(lattice, state["codes"], cuts, packed=packed),
            "mass_fractions": fractions.tolist(), "forces": forces,
            "edge_position_m": position.tolist(), "state": state, "points": points,
            "component": component,
            **({"cutting_state_after_phase": strokes} if cutting_law in (4, 5, 6, 7, 8) else {}),
            **({"code_encoding": bed.code_encoding, "digit_labelled_after_phase": digit_labels,
                "slot_diagnostics": bed.check_slots(), "solver_module": TLMPM.__module__} if packed else {}),
            **({'force_history_N': histories, 'severed_bonds_after_phase': bonds_after_phase,
                'reference_stencil_nodes': bed.nodes.numpy(), 'reference_stencil_weights': bed.weights.numpy(),
                'reference_stencil_gradients': bed.gradients.numpy()} if record_history else {})}


VERTICAL_CUT = Rotation.from_euler("y", 90, degrees=True)  # advance -z, edge along y, normal +x
FLAT_SKIM = Rotation.identity()                             # advance +x, edge along y, normal +z
