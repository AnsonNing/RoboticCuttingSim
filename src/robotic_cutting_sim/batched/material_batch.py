"""Batched cutting material for ``rl_gpu_headless`` (ADR 0021 §6-7): graph, coupling, wrench, validity, reset.

``BatchedCuttingMaterial`` owns one ``BatchedTLMPMv7_2`` and does, for all N envs at once, what
``LiveCuttingMaterial.advance`` does for one (``live_material.py:318-376``):

1. per-substep LOCAL tool poses/twists ``[N, T]`` and cut digits are written in place into the solver's
   arrays -- by the device ``pose_interp`` kernel (production) or, in ``parity_mode``, from the frozen
   SciPy ``interpolate_measured_poses`` (the L0 input path), or by the caller directly;
2. ``index``, ``reaction``, ``box_reactions`` and ``stats`` are zeroed;
3. one CUDA graph (captured once per batch shape) replays the 200 substeps; on CPU the substeps run eagerly;
4. a kernel converts ``reaction[N]`` and ``box_reactions[N*B]`` to world wrenches: the 20 ms delayed handoff
   keeps ``pending_wrench[N,6]`` (applied in the next interval) and ``applied_previous[N,6]``, pad wrenches
   ``pad_wrench[N,B,6]`` (rows of non-recipient boxes stay zero);
5. a segmented per-env diagnostics reduction and a validity kernel set ``invalid_reason[N]`` (bit flags,
   sticky until reset) and ``invalid_count[1]``; strict mode reads that one int32 and raises with env ids.

``reset(env_ids)`` is a masked device reset (no allocation, no recapture) to the exact freshly packed
state. A geometry change on partial reset is refused; ``rebuild(specs)`` repacks and recaptures.
"""
import time

import numpy as np
import warp as wp

from . import constitutive, layout
from .diagnostics import SegmentedDiagnostics
from .errors import UnsupportedBatchConfiguration
from .tlmpm_v7_2_batch import BatchedTLMPMv7_2

RUNTIME = "rl_gpu_headless"
MATERIAL_BATCH_VERSION = "batched_cutting_material_v1"
BACKEND = "kinematic_ideal_tracking_batched"
INVALID_NONFINITE_STATE = 1
INVALID_SLOT_OVERFLOW = 2
INVALID_POSE = 4
INVALID_CUT_DIGIT = 8
INVALID_NONFINITE_WRENCH = 16
INVALID_REASONS = {INVALID_NONFINITE_STATE: "NONFINITE_STATE", INVALID_SLOT_OVERFLOW: "SLOT_OVERFLOW",
                   INVALID_POSE: "INVALID_MEASURED_POSE", INVALID_CUT_DIGIT: "INVALID_CUT_DIGIT",
                   INVALID_NONFINITE_WRENCH: "NONFINITE_WRENCH"}

# ---- reset coverage (ADR §6; tested by tests/test_batched_partial_reset.py over vars())
# Solver arrays restored by reset() to their freshly packed values.
SOLVER_RESET_ARRAYS = (
    "x", "v", "x_start", "impulse", "stress", "F", "Fpi", "plastic", "progress", "contact_work", "code",
    "link_codes", "link_velocity_start", "front_crossings", "front_reached", "pending_q", "pending_cut", "slot",
    "slot_code", "gm", "gp", "gj", "old", "new_v",
    "edge_flags", "segment_ok", "segment_plow", "stroke_engaged", "stroke_any_motion", "stroke", "stroke_travel",
    "box_centers", "box_halves", "box_reactions", "slot_overflow", "slot_max_occupied", "previous_cut", "reaction",
    "poses", "twists", "cuts", "previous_pose", "link_previous_pose", "link_diagnostics",
    "stats_sum", "stats_max", "contact_totals_sum", "contact_totals_max")
# Solver arrays that never change after packing (geometry, stencil, CSR, parameters, tables).
SOLVER_CONSTANT_ARRAYS = (
    "reference", "mass", "nodes", "weights", "gradients", "link_offsets", "link_neighbours", "profile",
    "fixture_centers", "fixture_halves", "fixture_plane", "fixture_offset", "params", "env_of_particle",
    "particle_offset", "node_offset", "env_of_node")
# Shared lockstep substep counter: zeroed before every control step (and by a full reset).
SOLVER_SHARED_ARRAYS = ("index",)
# Material arrays restored by reset().
MATERIAL_RESET_ARRAYS = ("pending_wrench", "applied_previous", "pad_wrench", "step_count", "seed", "last_pose",
                         "measured_pose", "cut_input", "pose_status", "invalid_reason")
# Material constants and reset tables.
MATERIAL_CONSTANT_ARRAYS = ("R_world_local", "T_local_world", "q_local_world", "recipients",
                            "initial_local_pose", "initial_previous_pose", "initial_link_previous_pose",
                            "initial_box_centers", "initial_box_halves", "initial_measured_pose", "initial_seed")
# Scratch recomputed from scratch in every control step before being read.
RECOMPUTED_EVERY_CONTROL_STEP = ("invalid_count", "actual_pose")
# Inputs of reset_masked(): written by reset() (or by a caller through device views) right before it, read
# only by it, never by a control step. They keep the last reset's values (not a fresh-pack value).
RESET_INPUTS = ("reset_mask", "reset_seed")
DIAGNOSTICS_SCRATCH = ("partials", "summary")


# ----------------------------------------------------------------------------- kernels

vec6d = wp.types.vector(length=6, dtype=wp.float64)

@wp.func
def _rotate(R: wp.mat33d, a: wp.float64, b: wp.float64, c: wp.float64, row: int):
    return a*R[row, 0] + b*R[row, 1] + c*R[row, 2]


@wp.kernel
def _wrench_b(reaction: wp.array(dtype=wp.spatial_vector), box_reactions: wp.array(dtype=wp.spatial_vector),
              rotation: wp.array(dtype=wp.mat33d), recipients: wp.array(dtype=int), boxes: int,
              control_dt: wp.float64, feedback: int, pending: wp.array2d(dtype=wp.float64),
              applied_previous: wp.array2d(dtype=wp.float64), pad: wp.array3d(dtype=wp.float64)):
    """live_material.advance wrench handoff (live_material.py:353-374) for env e, in float64."""
    e = wp.tid()
    R = rotation[e]
    r = reaction[e]
    w = vec6d()
    for k in range(3):
        w[k] = _rotate(R, wp.float64(r[0]), wp.float64(r[1]), wp.float64(r[2]), k)
        w[3+k] = _rotate(R, wp.float64(r[3]), wp.float64(r[4]), wp.float64(r[5]), k)
    for b in range(boxes):
        if recipients[b] != 0:
            q = box_reactions[e*boxes+b]
            for k in range(3):
                value = _rotate(R, wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), k)
                torque = _rotate(R, wp.float64(q[3]), wp.float64(q[4]), wp.float64(q[5]), k)
                w[k] = w[k]-value
                w[3+k] = w[3+k]-torque
                pad[e, b, k] = value/control_dt
                pad[e, b, 3+k] = torque/control_dt
        else:
            for k in range(6):
                pad[e, b, k] = wp.float64(0.0)
    for k in range(6):
        previous = pending[e, k]
        if feedback == 0:
            previous = wp.float64(0.0)
        applied_previous[e, k] = previous
        pending[e, k] = w[k]/control_dt


@wp.kernel
def _validity_b(summary: wp.array2d(dtype=float), slot_overflow: wp.array(dtype=int),
                pose_status: wp.array(dtype=wp.int32), cut_input: wp.array(dtype=wp.int32), max_cuts: int,
                pending: wp.array2d(dtype=wp.float64), invalid_reason: wp.array(dtype=int),
                invalid_count: wp.array(dtype=int)):
    e = wp.tid()
    reason = int(0)
    if summary[e, 0] != 0.0:
        reason = reason | 1
    if slot_overflow[e] > 0:
        reason = reason | 2
    if pose_status[e] != 0:
        reason = reason | 4
    if cut_input[e] < -1 or cut_input[e] >= max_cuts:
        reason = reason | 8
    for k in range(6):
        if not wp.isfinite(pending[e, k]):
            reason = reason | 16
    reason = reason | invalid_reason[e]
    invalid_reason[e] = reason
    if reason != 0:
        wp.atomic_add(invalid_count, 0, 1)


@wp.kernel
def _finish_control_step(step_count: wp.array(dtype=wp.int64), last_pose: wp.array2d(dtype=wp.float64),
                         actual_pose: wp.array2d(dtype=wp.float64)):
    """``LiveCuttingMaterial.advance``: ``self.last_pose = actual`` (= ``validate_pose(measured)``)."""
    e = wp.tid()
    step_count[e] = step_count[e]+wp.int64(1)
    for k in range(7):
        last_pose[e, k] = actual_pose[e, k]


@wp.kernel
def _reset_particles(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                     reference: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                     x_start: wp.array(dtype=wp.vec3), impulse: wp.array(dtype=wp.vec3),
                     link_velocity_start: wp.array(dtype=wp.vec3), stress: wp.array(dtype=wp.mat33),
                     F: wp.array(dtype=wp.mat33), progress: wp.array(dtype=float),
                     contact_work: wp.array(dtype=float), code: wp.array(dtype=int), link_codes: wp.array(dtype=int),
                     front_crossings: wp.array(dtype=int), front_reached: wp.array(dtype=int),
                     pending_q: wp.array(dtype=int), pending_cut: wp.array(dtype=int)):
    """Freshly constructed single-solver values (tlmpm_v5.py:769-786, tlmpm_v7.py:250-254, tlmpm_v7_1.py:161-167)."""
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        x[p] = reference[p]
        v[p] = wp.vec3(0.0)
        x_start[p] = wp.vec3(0.0)
        impulse[p] = wp.vec3(0.0)
        link_velocity_start[p] = wp.vec3(0.0)
        stress[p] = wp.mat33(0.0)
        F[p] = wp.identity(n=3, dtype=float)
        progress[p] = 0.0
        contact_work[p] = 0.0
        code[p] = 0
        link_codes[p] = 0
        front_crossings[p] = 0
        front_reached[p] = 0
        pending_q[p] = -1
        pending_cut[p] = -1


@wp.kernel
def _reset_slots(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int), slot: wp.array2d(dtype=int)):
    p, corner = wp.tid()
    if mask[env_of_particle[p]] != 0:
        slot[p, corner] = 0


@wp.kernel
def _reset_state_mat33_identity(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                                values: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        values[p] = wp.identity(n=3, dtype=float)


@wp.kernel
def _reset_state_float_zero(mask: wp.array(dtype=int), env_of_particle: wp.array(dtype=int),
                            values: wp.array(dtype=float)):
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        values[p] = 0.0


@wp.kernel
def _reset_node_slots(mask: wp.array(dtype=int), env_of_node: wp.array(dtype=int), slots: int,
                      slot_code: wp.array(dtype=int), gm: wp.array(dtype=float), gp: wp.array(dtype=wp.vec3),
                      gj: wp.array(dtype=wp.vec3), old: wp.array(dtype=wp.vec3), new_v: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if mask[env_of_node[i/slots]] != 0:
        slot_code[i] = -1
        gm[i] = 0.0
        gp[i] = wp.vec3(0.0)
        gj[i] = wp.vec3(0.0)
        old[i] = wp.vec3(0.0)
        new_v[i] = wp.vec3(0.0)


@wp.kernel
def _reset_envs(mask: wp.array(dtype=int), segments: int, boxes: int, T: int,
                edge_flags: wp.array(dtype=int), segment_ok: wp.array(dtype=int), segment_plow: wp.array(dtype=int),
                stroke_engaged: wp.array(dtype=int), stroke_any_motion: wp.array(dtype=int),
                stroke: wp.array(dtype=int), stroke_travel: wp.array(dtype=float),
                box_centers: wp.array(dtype=wp.vec3), box_halves: wp.array(dtype=wp.vec3),
                box_reactions: wp.array(dtype=wp.spatial_vector), initial_box_centers: wp.array(dtype=wp.vec3),
                initial_box_halves: wp.array(dtype=wp.vec3), slot_overflow: wp.array(dtype=int),
                slot_max_occupied: wp.array(dtype=int), previous_cut: wp.array(dtype=int),
                reaction: wp.array(dtype=wp.spatial_vector), poses: wp.array(dtype=wp.transform),
                twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int),
                initial_local_pose: wp.array(dtype=wp.transform), previous_pose: wp.array(dtype=wp.transform),
                initial_previous_pose: wp.array(dtype=wp.transform),
                link_previous_pose: wp.array(dtype=wp.transform),
                initial_link_previous_pose: wp.array(dtype=wp.transform), link_diagnostics: wp.array(dtype=int),
                stats_sum: wp.array(dtype=float), stats_max: wp.array(dtype=float), stats_width: int,
                totals_sum: wp.array(dtype=wp.float64), totals_max: wp.array(dtype=wp.float64), totals_width: int,
                pending_wrench: wp.array2d(dtype=wp.float64), applied_previous: wp.array2d(dtype=wp.float64),
                pad_wrench: wp.array3d(dtype=wp.float64), step_count: wp.array(dtype=wp.int64),
                seed: wp.array(dtype=wp.int64), reset_seed: wp.array(dtype=wp.int64),
                last_pose: wp.array2d(dtype=wp.float64), measured_pose: wp.array2d(dtype=wp.float64),
                initial_measured_pose: wp.array2d(dtype=wp.float64), cut_input: wp.array(dtype=wp.int32),
                pose_status: wp.array(dtype=wp.int32), invalid_reason: wp.array(dtype=int)):
    """Per-env state of a freshly packed env (live_material.py:276-277 trajectory tile, tlmpm_v5.py:756-797)."""
    e = wp.tid()
    if mask[e] == 0:
        return
    for k in range(segments):
        edge_flags[e*segments+k] = 0
        segment_ok[e*segments+k] = 0
        segment_plow[e*segments+k] = 0
    stroke_engaged[e] = 0
    stroke_any_motion[e] = 0
    stroke[e] = 0
    stroke_travel[e] = 0.0
    for b in range(boxes):
        box_centers[e*boxes+b] = initial_box_centers[e*boxes+b]
        box_halves[e*boxes+b] = initial_box_halves[e*boxes+b]
        box_reactions[e*boxes+b] = wp.spatial_vector()
    slot_overflow[e] = 0
    slot_max_occupied[e] = 0
    previous_cut[e] = -1
    reaction[e] = wp.spatial_vector()
    for k in range(T):
        poses[e*T+k] = initial_local_pose[e]
        twists[e*T+k] = wp.spatial_vector()
        cuts[e*T+k] = -1
    previous_pose[e] = initial_previous_pose[e]
    link_previous_pose[e] = initial_link_previous_pose[e]
    link_diagnostics[e*2+0] = 0
    link_diagnostics[e*2+1] = 0
    for k in range(stats_width):
        stats_sum[e*stats_width+k] = 0.0
        stats_max[e*stats_width+k] = 0.0
    for k in range(totals_width):
        totals_sum[e*totals_width+k] = wp.float64(0.0)
        totals_max[e*totals_width+k] = wp.float64(0.0)
    for k in range(6):
        pending_wrench[e, k] = wp.float64(0.0)
        applied_previous[e, k] = wp.float64(0.0)
        for b in range(boxes):
            pad_wrench[e, b, k] = wp.float64(0.0)
    step_count[e] = wp.int64(0)
    seed[e] = reset_seed[e]
    for k in range(7):
        last_pose[e, k] = initial_measured_pose[e, k]
        measured_pose[e, k] = initial_measured_pose[e, k]
    cut_input[e] = -1
    pose_status[e] = 0
    invalid_reason[e] = 0


# ----------------------------------------------------------------------------- material

class InvalidBatchEnvs(FloatingPointError):
    """Strict mode: some envs became invalid in the last control step (ids and reasons attached)."""

    def __init__(self, env_ids, reasons):
        self.env_ids, self.reasons = list(env_ids), dict(reasons)
        super().__init__(f"Invalid batched envs {self.env_ids}: {self.reasons}")


def decode_reason(value):
    return [name for bit, name in INVALID_REASONS.items() if int(value) & bit]


class BatchedCuttingMaterial:
    """N independent v7_2 cutting materials in one process (ADR 0021 §6-7). See the module docstring.

    ``specs``: ``layout.EnvSpec`` list (un-reset ``LiveCuttingMaterial`` on device 'cpu', initial measured edge
    pose in WORLD, seed). ``device``: the batch device; ``"cpu"`` only with ``parity_mode=True`` (tests).
    ``strict``: raise ``InvalidBatchEnvs`` after a control step that left any env invalid (one int32 read);
    ``strict=False`` (training) only records ``invalid_reason`` for the caller to terminate those envs."""
    runtime = RUNTIME

    def __init__(self, specs, *, device="cuda:0", parity_mode=False, strict=True, capture_graph=None,
                 backend=BACKEND, cameras=None, recorders=None, sensors=None,
                 constitutive_model=constitutive.DEFAULT_MODEL_ID):
        if backend != BACKEND:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_BACKEND", f"{backend!r}; only {BACKEND} (P1-1b later)")
        if cameras or sensors:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_SENSOR", f"cameras={cameras!r} sensors={sensors!r}")
        if recorders:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_RECORDER", f"recorders={recorders!r}")
        device = str(wp.get_device(device))
        if device == "cpu" and not parity_mode:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_CPU_WITHOUT_PARITY_MODE",
                                                "device='cpu' is for parity tests (parity_mode=True)")
        self.device, self.parity_mode, self.strict = device, bool(parity_mode), bool(strict)
        self.capture_graph = wp.get_device(device).is_cuda if capture_graph is None else bool(capture_graph)
        self.constitutive_model = constitutive_model
        self._build(specs)

    # ------------------------------------------------------------------ construction

    def _build(self, specs):
        from .pose_interp import POSE_INTERPOLATION_VERSION, local_frames
        start = time.perf_counter()
        specs = list(specs)
        self.specs_feedback = [bool(s.material.config.feedback) for s in specs]
        if any(f != self.specs_feedback[0] for f in self.specs_feedback):
            raise UnsupportedBatchConfiguration("UNSUPPORTED_NONUNIFORM_BATCH", "coupling feedback differs")
        self.packed = layout.pack(specs, constitutive_model=self.constitutive_model)
        pk = self.packed
        if not all(s.material.cut_event_recorder is None for s in specs):
            raise UnsupportedBatchConfiguration("UNSUPPORTED_RECORDER", "cut_events recorder attached")
        self.control_dt = float(pk.uniform["control_dt"])
        self.feedback = self.specs_feedback[0]
        self.pose_interpolation = POSE_INTERPOLATION_VERSION
        self.num_envs = pk.num_envs
        self.solver = BatchedTLMPMv7_2(pk, self.device)
        s = self.solver
        N, B = self.num_envs, s.boxes
        recipients = pk.uniform["recipients"]
        recipient_mask = np.zeros(B, np.int32)
        if recipients is not None:
            recipient_mask[:len(recipients)] = [r is not None for r in recipients]
        T_world_local = pk.env["T_world_local"]
        from ..scene_frames import invert_transform
        T_local_world = np.stack([invert_transform(m) for m in T_world_local])
        with wp.ScopedDevice(self.device):
            s.env_of_node = wp.array(np.repeat(np.arange(N, dtype=np.int32), pk.node_counts), dtype=int)
            self.R_world_local = wp.array(np.ascontiguousarray(T_world_local[:, :3, :3]), dtype=wp.mat33d)
            self.T_local_world, self.q_local_world = local_frames(T_local_world, device=self.device)
            self.recipients = wp.array(recipient_mask, dtype=int)
            self.pending_wrench = wp.zeros((N, 6), dtype=wp.float64)
            self.applied_previous = wp.zeros((N, 6), dtype=wp.float64)
            self.pad_wrench = wp.zeros((N, B, 6), dtype=wp.float64)
            self.step_count = wp.zeros(N, dtype=wp.int64)
            self.initial_seed = wp.array(pk.env["seed"], dtype=wp.int64)
            self.seed = wp.array(pk.env["seed"], dtype=wp.int64)
            self.initial_measured_pose = wp.array(pk.env["measured_edge_pose"], dtype=wp.float64)
            self.last_pose = wp.array(pk.env["measured_edge_pose"], dtype=wp.float64)
            self.measured_pose = wp.array(pk.env["measured_edge_pose"], dtype=wp.float64)
            self.actual_pose = wp.array(pk.env["measured_edge_pose"], dtype=wp.float64)
            self.cut_input = wp.full(N, -1, dtype=wp.int32)
            self.pose_status = wp.zeros(N, dtype=wp.int32)
            self.invalid_reason = wp.zeros(N, dtype=int)
            self.invalid_count = wp.zeros(1, dtype=int)
            self.reset_mask = wp.zeros(N, dtype=int)
            self.reset_seed = wp.zeros(N, dtype=wp.int64)
            self.initial_local_pose = wp.array(pk.env["initial_local_pose"], dtype=wp.transform)
            self.initial_previous_pose = wp.array(pk.env["initial_previous_pose"], dtype=wp.transform)
            self.initial_link_previous_pose = wp.array(pk.env["initial_link_previous_pose"], dtype=wp.transform)
            self.initial_box_centers = wp.array(pk.env["initial_box_centers"].reshape(-1, 3), dtype=wp.vec3)
            self.initial_box_halves = wp.array(pk.env["initial_box_halves"].reshape(-1, 3), dtype=wp.vec3)
        self.diagnostics = SegmentedDiagnostics(s)
        for module in (__name__, "robotic_cutting_sim.batched.diagnostics", "robotic_cutting_sim.batched.pose_interp"):
            wp.load_module(module=module, device=self.device)
        self.graph = None
        self.capture_receipt = None
        if self.capture_graph:
            self._capture()
        self.build_wall_s = time.perf_counter()-start

    def _capture(self):
        if not wp.get_device(self.device).is_cuda:
            raise ValueError("CUDA graph capture needs a CUDA device")
        mempool = wp.is_mempool_enabled(self.device)
        before = wp.get_mempool_used_mem_current(self.device) if mempool else None
        start = time.perf_counter()
        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(self.solver.substeps):
                self.solver.step()
        self.graph = capture.graph
        after = wp.get_mempool_used_mem_current(self.device) if mempool else None
        self.capture_receipt = {"substeps_per_graph": self.solver.substeps, "capture_wall_s": time.perf_counter()-start,
                                "mempool_enabled": bool(mempool), "mempool_used_before": before,
                                "mempool_used_after": after,
                                "allocated_during_capture_bytes": None if mempool is False else int(after-before)}

    def rebuild(self, specs):
        """Explicit repack + recapture for a new geometry or batch shape; resets every env."""
        self.graph = None
        self._build(specs)

    # ------------------------------------------------------------------ inputs

    def set_measured_poses(self, measured_pose, cut):
        """Write WORLD measured edge poses [N,7] (float64) and cut digits [N] (int32) into the input buffers.

        Host arrays are validated (shapes, digit range) before either buffer is written, then copied; device
        arrays (Warp, same device) are copied without a host sync and checked on the device (``advance``)."""
        pose = digits = None
        if not isinstance(measured_pose, wp.array):
            pose = np.asarray(measured_pose, np.float64)
            if pose.shape != (self.num_envs, 7):
                raise ValueError(f"measured_pose must be [{self.num_envs}, 7]")
        if not isinstance(cut, wp.array):
            digits = np.asarray(cut)
            if digits.shape != (self.num_envs,) or digits.dtype.kind not in "iu":
                raise ValueError(f"cut must be [{self.num_envs}] integers")
            if np.any(digits < -1) or np.any(digits >= self.solver.max_cuts):
                raise ValueError(f"Cut digits must be -1 (disarmed) or 0..{self.solver.max_cuts-1}")
            digits = digits.astype(np.int32)
        if pose is None:
            wp.copy(self.measured_pose, measured_pose)
        else:
            self.measured_pose.assign(pose)
        if digits is None:
            wp.copy(self.cut_input, cut)
        else:
            self.cut_input.assign(digits)

    def _interpolate_device(self):
        from .pose_interp import launch_fill_cuts, launch_pose_interp
        s, N, T = self.solver, self.num_envs, self.solver.trajectory_length
        launch_pose_interp(self.last_pose, self.measured_pose, self.T_local_world, self.q_local_world,
                           s.poses.reshape((N, T)), s.twists.reshape((N, T)), self.pose_status, s.substeps,
                           self.control_dt, device=self.device, actual=self.actual_pose)
        # Out-of-range digits are disarmed and flagged before the substeps run (review finding 4).
        launch_fill_cuts(self.cut_input, s.cuts.reshape((N, T)), device=self.device, max_cuts=s.max_cuts,
                         invalid_reason=self.invalid_reason, flag=INVALID_CUT_DIGIT)

    def _interpolate_host_reference(self, measured_pose, cut):
        """Parity path: the frozen SciPy interpolation per env (input preparation only, never production)."""
        from ..coupled_robot import interpolate_measured_poses
        from ..robot import validate_pose
        from ..scene_frames import transform_poses
        if not self.parity_mode:
            raise ValueError("Host reference interpolation is the L0 parity input path (parity_mode=True)")
        if measured_pose is None:
            raise ValueError("host_reference interpolation needs measured_pose and cut")
        from ..scene_frames import invert_transform
        s, T = self.solver, self.solver.trajectory_length
        # As LiveCuttingMaterial.advance: actual = validate_pose(measured); last_pose holds the previous actual.
        raw = np.asarray(measured_pose, np.float64)
        if raw.shape != (self.num_envs, 7):
            raise ValueError(f"measured_pose must be [{self.num_envs}, 7]")
        actual = np.stack([validate_pose(p) for p in raw])
        self.set_measured_poses(raw, cut)
        self.actual_pose.assign(actual)
        last = self.last_pose.numpy()
        T_world_local = self.packed.env["T_world_local"]
        poses, twists = [], []
        for e in range(self.num_envs):
            T_lw = invert_transform(T_world_local[e])
            p, w = interpolate_measured_poses(transform_poses(T_lw, last[e]), transform_poses(T_lw, actual[e]),
                                              s.substeps, self.control_dt)
            poses.append(p)
            twists.append(w)
        s.poses.assign(np.concatenate(poses))
        s.twists.assign(np.concatenate(twists))
        s.cuts.assign(np.repeat(np.asarray(cut, np.int32), T))
        self.pose_status.zero_()

    # ------------------------------------------------------------------ control step

    def advance(self, measured_pose=None, cut=None, *, interpolation="device"):
        """One 20 ms control step of every env.

        ``interpolation``: ``"device"`` (pose_interp kernel from ``last_pose`` to ``measured_pose``),
        ``"host_reference"`` (parity mode: frozen SciPy path) or ``"preset"`` (the caller already wrote
        ``solver.poses/twists/cuts`` in place; ``measured_pose``/``cut`` still update the buffers if given).
        Cut digits: host digits outside ``-1..K-1`` raise ``ValueError`` before any state changes (the frozen
        ``LiveCuttingMaterial.advance`` refusal). Device digits (a Warp ``cut`` array, or caller-written preset
        rows) cannot raise: they are written as ``-1`` (disarmed) and the env gets ``INVALID_CUT_DIGIT`` before
        the substeps run; that env still advances one control step with a disarmed blade, then strict mode
        raises / training mode terminates it.
        Returns nothing; read ``pending_wrench``, ``applied_previous``, ``invalid_reason`` on the device."""
        if (measured_pose is None) != (cut is None):
            raise ValueError("measured_pose and cut are given together")
        if interpolation not in ("device", "host_reference", "preset"):
            raise ValueError("interpolation is 'device', 'host_reference' or 'preset'")
        if interpolation == "host_reference":
            self._interpolate_host_reference(measured_pose, cut)
        elif measured_pose is not None:
            self.set_measured_poses(measured_pose, cut)
        if interpolation == "device":
            self._interpolate_device()
        elif interpolation == "preset":
            from .pose_interp import launch_disarm_invalid_cuts, launch_validate_measured
            s = self.solver
            # last_pose bookkeeping and pose_status follow the measured buffer as in the device path.
            launch_validate_measured(self.last_pose, self.measured_pose, self.actual_pose, self.pose_status,
                                     device=self.device)
            launch_disarm_invalid_cuts(s.cuts.reshape((self.num_envs, s.trajectory_length)), s.max_cuts,
                                       self.invalid_reason, INVALID_CUT_DIGIT, device=self.device)
        s = self.solver
        with wp.ScopedDevice(self.device):
            s.zero_control_step_accumulators()
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                for _ in range(s.substeps):
                    s.step()
            wp.launch(_wrench_b, self.num_envs, [s.reaction, s.box_reactions, self.R_world_local, self.recipients,
                s.boxes, wp.float64(self.control_dt), int(self.feedback), self.pending_wrench,
                self.applied_previous, self.pad_wrench])
            self.diagnostics.launch()
            self.invalid_count.zero_()
            wp.launch(_validity_b, self.num_envs, [self.diagnostics.summary, s.slot_overflow, self.pose_status,
                self.cut_input, s.max_cuts, self.pending_wrench, self.invalid_reason, self.invalid_count])
            wp.launch(_finish_control_step, self.num_envs, [self.step_count, self.last_pose, self.actual_pose])
        if self.strict:
            self.check_valid()

    def check_valid(self):
        """Strict mode check: one int32 read; raises InvalidBatchEnvs with env ids and reasons."""
        count = int(self.invalid_count.numpy()[0])
        if count:
            reasons = self.invalid_reason.numpy()
            ids = np.flatnonzero(reasons)
            raise InvalidBatchEnvs(ids.tolist(), {int(e): decode_reason(reasons[e]) for e in ids})
        return 0

    # ------------------------------------------------------------------ reset

    def reset(self, env_ids=None, seeds=None, scene_keys=None):
        """Masked partial reset to the freshly packed state (ADR §6); no allocation, no recapture.

        ``env_ids`` None resets all (in env order 0..N-1). ``seeds`` and ``scene_keys`` are matched to
        ``env_ids`` by position, in the caller's order (``reset([2, 0], seeds=[s2, s0])``); duplicate ids
        are refused. ``seeds`` replace the env seeds; None keeps them. ``scene_keys`` must equal the packed
        geometry of each env, else UNSUPPORTED (use ``rebuild``). Every argument is validated before any
        state changes."""
        N = self.num_envs
        if env_ids is None:
            ids = np.arange(N, dtype=np.int64)
        else:
            raw = np.asarray(env_ids).reshape(-1)
            if raw.size == 0:
                if (seeds is not None and np.asarray(seeds).size) or (scene_keys is not None and len(scene_keys)):
                    raise ValueError("seeds/scene_keys given for an empty env_ids")
                return
            if raw.dtype.kind not in "iu":
                raise ValueError(f"env_ids must be integers, got dtype {raw.dtype}")
            ids = raw.astype(np.int64)
            if ids.min() < 0 or ids.max() >= N:
                raise ValueError(f"env_ids must be in [0, {N})")
            if np.unique(ids).size != ids.size:
                raise ValueError(f"duplicate env_ids {ids.tolist()}: each env is reset at most once per call")
        if scene_keys is not None:
            keys = list(scene_keys)
            if len(keys) != len(ids):
                raise ValueError("one scene key per reset env")
            for e, key in zip(ids, keys):
                if key != self.packed.scene_keys[e]:
                    raise UnsupportedBatchConfiguration("UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET",
                        f"env {int(e)} holds scene {self.packed.scene_keys[e]!r}; use rebuild()")
        mask = np.zeros(N, np.int32)
        mask[ids] = 1
        new_seed = self.seed.numpy().astype(np.int64)
        if seeds is not None:
            values = np.asarray(seeds).reshape(-1)
            if len(values) != len(ids):
                raise ValueError("one seed per reset env")
            if values.dtype.kind not in "iu":
                raise ValueError(f"seeds must be integers, got dtype {values.dtype}")
            new_seed[ids] = values.astype(np.int64)   # ids has no duplicates: caller order is kept
        self.reset_mask.assign(mask)
        self.reset_seed.assign(new_seed)
        self.reset_masked(full=len(ids) == N)

    def reset_masked(self, full=False):
        """Device part of ``reset``: uses ``reset_mask[N]`` and ``reset_seed[N]`` already on the device."""
        s = self.solver
        with wp.ScopedDevice(self.device):
            wp.launch(_reset_particles, s.total_particles, [self.reset_mask, s.env_of_particle, s.reference, s.x,
                s.v, s.x_start, s.impulse, s.link_velocity_start, s.stress, s.F, s.progress, s.contact_work,
                s.code, s.link_codes, s.front_crossings, s.front_reached, s.pending_q, s.pending_cut])
            wp.launch(_reset_slots, (s.total_particles, 8), [self.reset_mask, s.env_of_particle, s.slot])
            for variable in s.model.per_particle_state:   # constitutive registry (ADR §11)
                kernel = {"identity": _reset_state_mat33_identity, "zero": _reset_state_float_zero}[variable.initial]
                wp.launch(kernel, s.total_particles, [self.reset_mask, s.env_of_particle, getattr(s, variable.name)])
            wp.launch(_reset_node_slots, s.total_nodes*s.slots_per_node, [self.reset_mask, s.env_of_node,
                s.slots_per_node, s.slot_code, s.gm, s.gp, s.gj, s.old, s.new_v])
            wp.launch(_reset_envs, self.num_envs, [self.reset_mask, s.edge_segments, s.boxes, s.trajectory_length,
                s.edge_flags, s.segment_ok, s.segment_plow, s.stroke_engaged, s.stroke_any_motion, s.stroke,
                s.stroke_travel, s.box_centers, s.box_halves, s.box_reactions, self.initial_box_centers,
                self.initial_box_halves, s.slot_overflow, s.slot_max_occupied, s.previous_cut, s.reaction,
                s.poses, s.twists, s.cuts, self.initial_local_pose, s.previous_pose, self.initial_previous_pose,
                s.link_previous_pose, self.initial_link_previous_pose, s.link_diagnostics,
                s.stats_sum, s.stats_max, layout.STATS_WIDTH, s.contact_totals_sum, s.contact_totals_max,
                layout.TOTALS_WIDTH, self.pending_wrench, self.applied_previous, self.pad_wrench, self.step_count,
                self.seed, self.reset_seed, self.last_pose, self.measured_pose, self.initial_measured_pose,
                self.cut_input, self.pose_status, self.invalid_reason])
            if full:
                s.index.zero_()

    # ------------------------------------------------------------------ queries

    def set_workpiece_pose(self, *args, **kwargs):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_WORKPIECE_MOTION",
                                            "the single solver keeps a static workpiece-local frame")

    def diagnostics_read(self):
        return self.diagnostics.read()

    def snapshot(self, e):
        return self.solver.snapshot(e)

    def world_state(self, e):
        """WORLD oracle state of env e (as ``LiveCuttingMaterial.world_state``); never an observation."""
        from ..scene_frames import transform_points
        snap = self.snapshot(e)
        T = self.packed.env["T_world_local"][e]
        return {"positions": transform_points(T, snap["x"]), "velocities": snap["v"]@T[:3, :3].T,
                "codes": snap["code"], "F": snap["F"], "Fpi": snap["Fpi"], "plastic_strain": snap["plastic"],
                "cut_progress": snap["progress"], "contact_work_j": snap["contact_work"]}

    def device_arrays(self):
        out = {f"solver.{k}": v for k, v in self.solver.device_arrays().items()}
        out.update({k: v for k, v in vars(self).items() if isinstance(v, wp.array)})
        out.update({f"diagnostics.{k}": v for k, v in vars(self.diagnostics).items() if isinstance(v, wp.array)})
        return out

    def memory_bytes(self):
        """Every device array of the batch (solver, coupling, diagnostics), excluding the hash grid."""
        return int(sum(v.capacity for v in self.device_arrays().values()))

    def receipt(self):
        pk = self.packed
        B = self.solver.boxes
        pads = pk.uniform["recipients"] is not None and any(r is not None for r in pk.uniform["recipients"])
        reasons = [] if pads else ["no_finger_pad_obstacles"]
        return {"runtime": RUNTIME, "version": MATERIAL_BATCH_VERSION, "backend": BACKEND, "device": self.device,
                "parity_mode": self.parity_mode, "strict": self.strict, "graph": self.capture_receipt,
                "pose_interpolation": self.pose_interpolation, "control_dt_s": self.control_dt,
                "feedback": self.feedback, "obstacle_boxes": B, "layout": pk.layout_receipt(),
                "solver": self.solver.solver_receipt(), "envs": pk.receipts,
                "material_fidelity_reasons": reasons,
                "calibration": "SIMULATOR_ONLY_UNCALIBRATED", "memory_bytes": self.memory_bytes(),
                "build_wall_s": self.build_wall_s}
