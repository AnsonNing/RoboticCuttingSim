"""Kinematic blade probe with a per-phase edge orientation (WP-12 Phase 1 geometry tests; no robot, no renderer).

``drive_blade_sequence`` is ``blade_probe.drive_blade`` with one change: every phase carries its own edge
rotation, applied as a pose jump at the phase boundary (use it only while the blade is clear of the material,
e.g. in a disarmed reorientation phase). Within a phase the motion is the same constant-velocity, zero-angular
twist trajectory in 200-substep chunks, the same tlmpm_v5 construction (``TLMPMConfig(cuts, blade_half_width,
cutting_energy)``), the same optional cut_events_v1 bracketing and CUDA-graph replay. With one constant rotation it
reproduces ``drive_blade`` (tests/test_sequence_blade_probe.py, CPU, bit for bit). Physics is not changed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .blade_probe import block, severed_bonds
from .cut_topology import decode_digits, reference_components_packed
from .material_config import TLMPMConfig


@dataclass(frozen=True)
class OrientedPhase:
    """Constant edge velocity (world, m/s) for a duration with edge->world rotation (xyzw); cut digit or -1."""
    velocity_m_s: tuple
    duration_s: float
    cut: int = 0
    rotation_xyzw: tuple = (0., 0., 0., 1.)


def drive_blade_sequence(profile, start_m, phases, *, points=None, masses=None, spacing=.004, cutting_energy=5.,
                         cuts=1, device="cpu", chunk=200, gpu_graph=False, slots_per_node=None, cut_events=None,
                         cut_events_capacity=None, config_overrides=None):
    """tlmpm_v5 law 8 only. Returns the drive_blade keys used by the WP-12 Phase 1 driver (state, points,
    component, labelled/digit counts per phase, stencil, cut_events)."""
    import warp as wp
    from .tlmpm_v5 import TLMPM
    for phase in phases:
        if not isinstance(phase.cut, (int, np.integer)) or isinstance(phase.cut, bool) or not -1 <= phase.cut < cuts:
            raise ValueError(f"Phase cut index {phase.cut!r} must be -1 or an integer in [0, {cuts})")
    if points is None:
        points, masses = block(spacing=spacing)
    polygon, half = profile.tlmpm_profile()
    config = TLMPMConfig(cuts=cuts, blade_half_width=float(half), cutting_energy=float(cutting_energy),
                         **(config_overrides or {}))
    bed = TLMPM(points, masses, spacing, config, device, blade_profile_xz=polygon, cutting_law=8,
                **({"slots_per_node": slots_per_node} if slots_per_node is not None else {}))
    lattice = np.round((points-points.min(0))/spacing).astype(int)
    recorder = None
    if cut_events is not None:
        if cut_events != "cut_events_v1":
            raise ValueError("cut_events is None or 'cut_events_v1'")
        from . import cut_events as ce
        recorder = ce.CutEventRecorder(bed, lattice, capacity=cut_events_capacity)
    graphs = {}
    if gpu_graph:
        wp.load_module(module=TLMPM.__module__, device=device)
        if recorder is not None:
            wp.load_module(module="robotic_cutting_sim.cut_events", device=device)
    position = np.asarray(start_m, float)
    labels, digit_labels = [], []
    for phase in phases:
        quaternion = Rotation.from_quat(phase.rotation_xyzw).as_quat()
        velocity = np.asarray(phase.velocity_m_s, float)
        steps = int(round(phase.duration_s/config.dt))
        for first in range(0, steps, chunk):
            count = min(chunk, steps-first)
            poses = np.array([np.r_[position+velocity*config.dt*i, quaternion] for i in range(count+1)], np.float32)
            twists = np.tile(np.r_[velocity, 0., 0., 0.], (count+1, 1)).astype(np.float32)
            digits = np.full(count+1, phase.cut, np.int32)
            if gpu_graph:
                if count not in graphs:
                    bed.set_trajectory(poses, twists, digits)
                    with wp.ScopedCapture(device=device) as capture:
                        for _ in range(count):
                            bed.reaction.zero_()
                            if recorder is not None:
                                recorder.before_step()
                            bed.step()
                            if recorder is not None:
                                recorder.after_step()
                    graphs[count] = (capture.graph, bed.poses, bed.twists, bed.cuts)
                graph, pose_buffer, twist_buffer, digit_buffer = graphs[count]
                pose_buffer.assign(poses)
                twist_buffer.assign(twists)
                digit_buffer.assign(digits)
                bed.index.zero_()
                wp.capture_launch(graph)
            else:
                bed.set_trajectory(poses, twists, digits)
                for _ in range(count):
                    bed.reaction.zero_()
                    if recorder is not None:
                        recorder.before_step()
                    bed.step()
                    if recorder is not None:
                        recorder.after_step()
            position = position+velocity*config.dt*count
            bed.check_slots()
            if recorder is not None:
                recorder.check()
        wp.synchronize_device(device)
        labels.append(int(np.count_nonzero(bed.code.numpy())))
        digit_labels.append(np.count_nonzero(decode_digits(bed.code.numpy(), cuts), axis=0).tolist())
    state = bed.snapshot()
    count, component = reference_components_packed(points, lattice, state["codes"])
    return {"labelled_after_phase": labels, "digit_labelled_after_phase": digit_labels, "components": int(count),
            "severed_bonds": severed_bonds(lattice, state["codes"], cuts, packed=True), "state": state,
            "points": points, "component": component, "edge_position_m": position.tolist(),
            "code_encoding": bed.code_encoding, "slot_diagnostics": bed.check_slots(), "solver_module": TLMPM.__module__,
            "reference_stencil_nodes": bed.nodes.numpy(), "reference_stencil_weights": bed.weights.numpy(),
            "reference_stencil_gradients": bed.gradients.numpy(),
            **({"cut_events": {"metadata": recorder.metadata(), "log": recorder.log(),
                               "activation": recorder.activation(), "neighbours": recorder.neighbours_host}}
               if recorder is not None else {})}
