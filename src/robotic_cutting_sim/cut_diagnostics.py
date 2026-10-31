"""Read-only per-segment and per-particle cutting diagnostics ``cut_diagnostics_v1`` (WP-12 Phase 4a, spec §4.2).

The recorder observes a tlmpm_v4 (ternary codes, K <= 2) or tlmpm_v5 (packed codes) solver running cutting
law 4, 5 or 8. It never writes x, v, F, code, progress, contact_work or any other solver array and does not
change either solver module. It is opt-in: nothing creates it unless a caller constructs it, and a solver
without it launches nothing new. Per substep the caller brackets the unchanged solver step::

    diagnostics.before_step()   # copies of code, contact_work, x, v (four device copies)
    bed.step()                  # unchanged
    diagnostics.after_step()    # three read-only kernels

All counters live on the device, so the bracket is CUDA-graph capturable. ``end_control_step()`` (host) moves
the per-segment accumulators of the finished control step into a host table and zeroes them.

What is recorded (spec §4.2)
----------------------------
Per edge segment k (the solver's 1-spacing segments) and control step, summed over its substeps
(``SEGMENT_COLUMNS``): armed substeps; law-4 motion gate ``seg_ok`` (moving forward or drawing, within
theta_max of the blade plane); law-8 plow mask ``seg_plow``; the segment motion components along the edge
frame (d = advance, e = edge tangent, n = blade normal); blade-contact particles and their normal contact
impulse before the cutting release (``_contact_impulse`` recomputed at the pre-step x and v, exactly as
``_rigid_contacts`` does); directional cutting work (the solver's own ``contact_work`` increment, so it is the
law-4 dW); particles that received cutting work while the motion gate was closed; contact particles whose
progress was complete; new side digits by rule (band behind the edge / law-5 lead) and digit 1/2; substeps
with an ill-conditioned swept normal.

Per substep (``SUBSTEP_COLUMNS``): trajectory index, cut digit, stroke state S, engagement W, any_motion,
travel since the last engaged substep, seg_ok / seg_plow counts, new band / lead digits, contact particles,
normal impulse, cutting work, the edge-centre motion (d, e, n), swept-normal conditioning counts and the number
of new digits that a corrected swept-normal fallback would flip.

Per particle and cut digit (``particle_table``): for every armed substep in which the digit is still 0 after
the step, the first failing law-4 band condition in the solver's own order (``REASONS``: stroke off, outside
the edge width, ahead of the edge f > 0, beyond the rear limit, segment motion gate closed, outside the band
|s| > h) and the substeps in which that condition was the only one failing; the substeps in which the law-5
lead rule held except for the law-8 plow mask; the smallest edge-frame f seen while within the width and band
(with the stroke on, and while armed at all). For each new digit: substep, rule, s and f at assignment, the
position, and flags (``ASSIGN_FLAGS``) comparing the solver's swept normal with a corrected fallback.

Swept-normal check (task item (e)). The solver uses n_s = m x e (m = segment motion, e = edge tangent) unless
|m x e|^2 <= 1e-20 (absolute), then the blade normal. ``ill_float`` marks |m x e| <= 1e-5 |m| (the
component normal to the edge line is at the float32 rounding level of m); ``ill_geom`` marks |m x e| <= 1e-2 |m|
(motion within 0.57 deg of the edge line). For each new digit the side is recomputed with the blade-normal
fallback applied whenever the flag holds, and digits that would differ are counted. This is a diagnostic of
the frozen rule, not a new law.

Oracle data: these values are solver internals and must stay out of policy observations.
"""
import hashlib

import numpy as np
import warp as wp

CUT_DIAGNOSTICS_VERSION = "cut_diagnostics_v1"

SEGMENT_COLUMNS = ("armed_substeps", "seg_ok_substeps", "seg_plow_substeps", "motion_d_sum_m_s", "motion_e_sum_m_s",
                   "motion_n_sum_m_s", "contact_particle_substeps", "normal_impulse_sum_N_s", "cutting_work_sum_J",
                   "work_with_gate_closed_particle_substeps", "progress_complete_contact_particle_substeps",
                   "band_new_digit1", "band_new_digit2", "lead_new_digit1", "lead_new_digit2",
                   "swept_ill_float_substeps", "swept_ill_geom_substeps")
SUBSTEP_COLUMNS = ("substep", "trajectory_index", "cut", "stroke", "engaged", "any_motion", "travel_m",
                   "seg_ok_count", "seg_plow_count", "band_new", "lead_new", "contact_particles",
                   "normal_impulse_N_s", "cutting_work_J", "centre_motion_d_m_s", "centre_motion_e_m_s",
                   "centre_motion_n_m_s", "swept_ill_float_segments", "swept_ill_geom_segments",
                   "swept_solver_fallback_segments", "new_digits_flipped_by_float_fallback",
                   "new_digits_flipped_by_geom_fallback")
REASONS = ("stroke_off", "outside_edge_width", "ahead_of_edge", "beyond_rear_limit", "segment_gate_closed",
           "outside_band")
ASSIGN_FLAGS = {"lead_rule": 1, "solver_blade_normal_fallback": 2, "ill_float": 4, "ill_geom": 8,
                "flipped_by_float_fallback": 16, "flipped_by_geom_fallback": 32, "swept_normal_over_10deg": 64}
ILL_FLOAT_REL = 1e-5
ILL_GEOM_REL = 1e-2
COS_10_DEG = float(np.cos(np.deg2rad(10.)))
_SCRATCH = 12   # float scratch per substep (indices below)
_S_BAND = wp.constant(0)
_S_LEAD = wp.constant(1)
_S_CONTACT = wp.constant(2)
_S_IMPULSE = wp.constant(3)
_S_WORK = wp.constant(4)
_S_ILL_F = wp.constant(5)
_S_ILL_G = wp.constant(6)
_S_FALLBACK = wp.constant(7)
_S_FLIP_F = wp.constant(8)
_S_FLIP_G = wp.constant(9)


class CutDiagnosticsOverflowError(RuntimeError):
    """More substeps than the recorder's fixed per-substep capacity."""


@wp.func
def _cd_digit(code: int, cut: int, ternary: int):
    if ternary != 0:
        factor = int(1)
        for j in range(cut):
            factor = factor*3
        return (code//factor) % 3
    return (code >> (2*cut)) & 3


@wp.func
def _cd_edge_segment(y: float, half_width: float, segments: int):
    # tlmpm_v4/_v5 _edge_segment (same operations).
    k = int((y+half_width)/(2.0*half_width)*float(segments))
    return wp.clamp(k, 0, segments-1)


@wp.func
def _cd_segment(p: wp.vec2, a: wp.vec2, b: wp.vec2):
    e = b-a
    return a+wp.clamp(wp.dot(p-a, e)/wp.dot(e, e), 0.0, 1.0)*e


@wp.func
def _cd_profile_sdf(p: wp.vec3, profile: wp.array(dtype=wp.vec2), count: int, half_width: float):
    # tlmpm_v4/_v5 _profile_sdf (same operations and order).
    x = wp.vec2(p[0], p[2])
    best = float(1.0e10)
    closest = x*0.0
    outward = x*0.0
    inside = int(1)
    for edge in range(count):
        v0 = profile[edge]
        v1 = profile[(edge+1) % count]
        e = v1-v0
        if e[0]*(x[1]-v0[1])-e[1]*(x[0]-v0[0]) < 0.0:
            inside = 0
        q = _cd_segment(x, v0, v1)
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


@wp.kernel
def _cd_segments(poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                 cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int, max_cuts: int,
                 edge_width: float, segments: int, seg_ok: wp.array(dtype=int), seg_plow: wp.array(dtype=int),
                 acc: wp.array2d(dtype=float), scratch: wp.array(dtype=float)):
    k = wp.tid()
    i = index[0]-shift
    if i < 0:
        return
    cut = cuts[i]
    if cut < 0 or cut >= max_cuts:
        return
    pose = poses[i]
    twist = twists[i]
    y = (float(k)+0.5)/float(segments)*2.0*edge_width-edge_width
    advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    motion = wp.vec3(twist[0], twist[1], twist[2]) + wp.cross(wp.vec3(twist[3], twist[4], twist[5]), along*y)
    acc[k, 0] = acc[k, 0]+1.0
    acc[k, 1] = acc[k, 1]+float(seg_ok[k])
    acc[k, 2] = acc[k, 2]+float(seg_plow[k])
    acc[k, 3] = acc[k, 3]+wp.dot(motion, advance)
    acc[k, 4] = acc[k, 4]+wp.dot(motion, along)
    acc[k, 5] = acc[k, 5]+wp.dot(motion, normal)
    swept = wp.cross(motion, along)
    swept2 = wp.length_sq(swept)
    motion2 = wp.length_sq(motion)
    if motion2 > 0.0:
        if swept2 <= 1.0e-10*motion2:
            acc[k, 15] = acc[k, 15]+1.0
            wp.atomic_add(scratch, _S_ILL_F, 1.0)
        if swept2 <= 1.0e-4*motion2:
            acc[k, 16] = acc[k, 16]+1.0
            wp.atomic_add(scratch, _S_ILL_G, 1.0)
    if swept2 <= 1.0e-20:
        wp.atomic_add(scratch, _S_FALLBACK, 1.0)


@wp.kernel
def _cd_particles(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), x_prev: wp.array(dtype=wp.vec3),
                  v_prev: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), code: wp.array(dtype=int),
                  code_prev: wp.array(dtype=int), work: wp.array(dtype=float), work_prev: wp.array(dtype=float),
                  progress: wp.array(dtype=float), poses: wp.array(dtype=wp.transform),
                  twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
                  shift: int, max_cuts: int, ternary: int, stroke: wp.array(dtype=int),
                  seg_ok: wp.array(dtype=int), seg_plow: wp.array(dtype=int), segments: int, edge_width: float,
                  band: float, rear: float, lead: float, core: float, lead_mode: int, dt: float,
                  profile: wp.array(dtype=wp.vec2), profile_count: int, radius: float, stiffness: float, tool: int,
                  substep: wp.array(dtype=int), acc: wp.array2d(dtype=float), scratch: wp.array(dtype=float),
                  first_fail: wp.array2d(dtype=int), only_fail: wp.array2d(dtype=int),
                  lead_plow_blocked: wp.array(dtype=int), min_f_stroke: wp.array(dtype=float),
                  min_f_armed: wp.array(dtype=float), assign_substep: wp.array(dtype=int),
                  assign_s: wp.array(dtype=float), assign_f: wp.array(dtype=float),
                  assign_x: wp.array(dtype=wp.vec3), assign_flags: wp.array(dtype=int)):
    p = wp.tid()
    i = index[0]-shift
    if i < 0:
        return
    pose = poses[i]
    twist = twists[i]
    cut = cuts[i]
    origin = wp.transform_get_translation(pose)
    inverse = wp.transform_inverse(pose)
    linear = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    advance = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    blade_normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    # ---- blade contact at the pre-step state (as _rigid_contacts, before the cutting release)
    if tool != 0:
        local0 = wp.transform_point(inverse, x_prev[p])
        hit = _cd_profile_sdf(local0, profile, profile_count, edge_width)
        normal = wp.transform_vector(pose, wp.vec3(hit[0], hit[1], hit[2]))
        gap = hit[3]-radius
        lever = x_prev[p]-origin-normal*radius
        relative = v_prev[p]-linear-wp.cross(angular, lever)
        vn = wp.dot(relative, normal)
        predicted_gap = gap+dt*wp.min(vn, 0.0)
        jn = stiffness*mass[p]/dt*wp.max(-predicted_gap, 0.0)
        k0 = _cd_edge_segment(local0[1], edge_width, segments)
        if jn > 0.0:
            wp.atomic_add(acc, k0, 6, 1.0)
            wp.atomic_add(acc, k0, 7, jn)
            wp.atomic_add(scratch, _S_CONTACT, 1.0)
            wp.atomic_add(scratch, _S_IMPULSE, jn)
            if progress[p] >= 1.0:
                wp.atomic_add(acc, k0, 10, 1.0)
        dwork = work[p]-work_prev[p]
        if dwork > 0.0:
            wp.atomic_add(acc, k0, 8, dwork)
            wp.atomic_add(scratch, _S_WORK, dwork)
            if seg_ok[k0] == 0:
                wp.atomic_add(acc, k0, 9, 1.0)
    if cut < 0 or cut >= max_cuts:
        return
    d_now = _cd_digit(code[p], cut, ternary)
    d_prev = _cd_digit(code_prev[p], cut, ternary)
    if d_prev != 0:
        return
    xp = x[p]
    local = wp.transform_point(inverse, xp)
    # Exactly tlmpm_v4/_v5 _swept_side (same operations and order), plus the conditioning of m x e.
    segment_motion = linear + wp.cross(angular, along * local[1])
    swept_normal = wp.cross(segment_motion, along)
    swept2 = wp.length_sq(swept_normal)
    motion2 = wp.length_sq(segment_motion)
    side = local[2]
    fallback = int(1)
    cos_normal = float(1.0)
    if swept2 > 1.0e-20:
        swept_normal = wp.normalize(swept_normal)
        if wp.dot(swept_normal, blade_normal) < 0.0:
            swept_normal = -swept_normal
        side = wp.dot(xp - origin, swept_normal)
        fallback = 0
        cos_normal = wp.dot(swept_normal, blade_normal)
    k = _cd_edge_segment(local[1], edge_width, segments)
    slot = p*max_cuts+cut
    if d_now != 0:
        flags = int(0)
        is_lead = local[0] > 0.0
        if is_lead:
            flags = flags | 1
        if fallback != 0:
            flags = flags | 2
        ill_f = motion2 > 0.0 and swept2 <= 1.0e-10*motion2
        ill_g = motion2 > 0.0 and swept2 <= 1.0e-4*motion2
        if ill_f:
            flags = flags | 4
        if ill_g:
            flags = flags | 8
        robust = 2
        if local[2] < 0.0:
            robust = 1
        if ill_f and robust != d_now:
            flags = flags | 16
            wp.atomic_add(scratch, _S_FLIP_F, 1.0)
        if ill_g and robust != d_now:
            flags = flags | 32
            wp.atomic_add(scratch, _S_FLIP_G, 1.0)
        if fallback == 0 and cos_normal < 0.98480775:
            flags = flags | 64
        assign_substep[slot] = substep[0]
        assign_s[slot] = side
        assign_f[slot] = local[0]
        assign_x[slot] = xp
        assign_flags[slot] = flags
        column = 11
        if is_lead:
            column = 13
            wp.atomic_add(scratch, _S_LEAD, 1.0)
        else:
            wp.atomic_add(scratch, _S_BAND, 1.0)
        if d_now == 2:
            column = column+1
        wp.atomic_add(acc, k, column, 1.0)
        return
    # ---- still unassigned: the law-4 band conditions in the solver's order
    stroke_on = stroke[0] == cut+1
    width_ok = wp.abs(local[1]) <= edge_width
    behind = local[0] <= 0.0
    forward = advance
    displacement = wp.max(wp.dot(linear-v[p], forward)*dt, 0.0)
    rear_ok = local[0] >= -rear-displacement
    gate = seg_ok[k] != 0
    band_ok = wp.abs(side) <= band
    fails = int(0)
    first = int(-1)
    only = int(-1)
    if not stroke_on:
        fails += 1
        first = 0
        only = 0
    if not width_ok:
        fails += 1
        if first < 0:
            first = 1
        only = 1
    if not behind:
        fails += 1
        if first < 0:
            first = 2
        only = 2
    if not rear_ok:
        fails += 1
        if first < 0:
            first = 3
        only = 3
    if not gate:
        fails += 1
        if first < 0:
            first = 4
        only = 4
    if not band_ok:
        fails += 1
        if first < 0:
            first = 5
        only = 5
    if first >= 0:
        first_fail[slot, first] = first_fail[slot, first]+1
    if fails == 1:
        only_fail[slot, only] = only_fail[slot, only]+1
    if width_ok and band_ok:
        min_f_armed[slot] = wp.min(min_f_armed[slot], local[0])
        if stroke_on:
            min_f_stroke[slot] = wp.min(min_f_stroke[slot], local[0])
    if lead_mode == 8 and stroke_on and width_ok and gate and seg_plow[k] == 0:
        if local[0] > 0.0 and local[0] <= lead and wp.abs(side) > core and wp.abs(side) <= band:
            lead_plow_blocked[slot] = lead_plow_blocked[slot]+1


@wp.kernel
def _cd_summary(poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int, stroke: wp.array(dtype=int),
                engaged: wp.array(dtype=int), any_motion: wp.array(dtype=int), travel: wp.array(dtype=float),
                seg_ok: wp.array(dtype=int), seg_plow: wp.array(dtype=int), segments: int,
                substep: wp.array(dtype=int), scratch: wp.array(dtype=float), capacity: int,
                rows: wp.array2d(dtype=float), overflow: wp.array(dtype=int)):
    i = index[0]-shift
    row = substep[0]
    if i >= 0 and row < capacity:
        pose = poses[i]
        twist = twists[i]
        linear = wp.vec3(twist[0], twist[1], twist[2])
        ok = int(0)
        plow = int(0)
        for k in range(segments):
            ok += seg_ok[k]
            plow += seg_plow[k]
        rows[row, 0] = float(row)
        rows[row, 1] = float(i)
        rows[row, 2] = float(cuts[i])
        rows[row, 3] = float(stroke[0])
        rows[row, 4] = float(engaged[0])
        rows[row, 5] = float(any_motion[0])
        rows[row, 6] = travel[0]
        rows[row, 7] = float(ok)
        rows[row, 8] = float(plow)
        rows[row, 9] = scratch[_S_BAND]
        rows[row, 10] = scratch[_S_LEAD]
        rows[row, 11] = scratch[_S_CONTACT]
        rows[row, 12] = scratch[_S_IMPULSE]
        rows[row, 13] = scratch[_S_WORK]
        rows[row, 14] = wp.dot(linear, wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0)))
        rows[row, 15] = wp.dot(linear, wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0)))
        rows[row, 16] = wp.dot(linear, wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0)))
        rows[row, 17] = scratch[_S_ILL_F]
        rows[row, 18] = scratch[_S_ILL_G]
        rows[row, 19] = scratch[_S_FALLBACK]
        rows[row, 20] = scratch[_S_FLIP_F]
        rows[row, 21] = scratch[_S_FLIP_G]
    elif row >= capacity:
        wp.atomic_add(overflow, 0, 1)
    substep[0] = row+1
    for j in range(12):
        scratch[j] = 0.0


def _sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class CutDiagnosticsRecorder:
    """``cut_diagnostics_v1`` bound to one tlmpm_v4 / tlmpm_v5 ``TLMPM`` (laws 4, 5, 8; read-only, oracle-only)."""

    version = CUT_DIAGNOSTICS_VERSION

    def __init__(self, bed, *, summary_capacity=1 << 18):
        law = int(getattr(bed, "cutting_law", -1))
        if law not in (4, 5, 8) or not hasattr(bed, "segment_ok") or not hasattr(bed, "stroke"):
            raise ValueError("cut_diagnostics_v1 records tlmpm_v4/tlmpm_v5 solvers running cutting law 4, 5 or 8")
        self.bed, self.device = bed, bed.device
        self.ternary = 0 if getattr(bed, "code_encoding", None) == "packed2" else 1
        self.max_cuts = int(bed.max_cuts) if hasattr(bed, "max_cuts") else int(bed.config.cuts)
        if self.ternary and self.max_cuts > 2:
            raise ValueError("Ternary (tlmpm_v4) codes keep at most 2 cut digits")
        self.law = law
        self.segments = int(bed.edge_segments)
        self.summary_capacity = int(summary_capacity)
        n, k = int(bed.n), self.max_cuts
        self.n = n
        self.segment_rows = []           # host: one (segments, len(SEGMENT_COLUMNS)) array per end_control_step
        with wp.ScopedDevice(self.device):
            self.code_prev = wp.zeros(n, dtype=int)
            self.work_prev = wp.zeros(n, dtype=float)
            self.x_prev = wp.zeros(n, dtype=wp.vec3)
            self.v_prev = wp.zeros(n, dtype=wp.vec3)
            self.substep = wp.zeros(1, dtype=int)
            self.scratch = wp.zeros(_SCRATCH, dtype=float)
            self.overflow = wp.zeros(1, dtype=int)
            self.acc = wp.zeros((self.segments, len(SEGMENT_COLUMNS)), dtype=float)
            self.rows = wp.zeros((self.summary_capacity, len(SUBSTEP_COLUMNS)), dtype=float)
            self.first_fail = wp.zeros((n*k, len(REASONS)), dtype=int)
            self.only_fail = wp.zeros((n*k, len(REASONS)), dtype=int)
            self.lead_plow_blocked = wp.zeros(n*k, dtype=int)
            self.min_f_stroke = wp.full(n*k, 1.0e9, dtype=float)
            self.min_f_armed = wp.full(n*k, 1.0e9, dtype=float)
            self.assign_substep = wp.full(n*k, -1, dtype=int)
            self.assign_s = wp.zeros(n*k, dtype=float)
            self.assign_f = wp.zeros(n*k, dtype=float)
            self.assign_x = wp.zeros(n*k, dtype=wp.vec3)
            self.assign_flags = wp.zeros(n*k, dtype=int)

    def parameters(self):
        bed = self.bed
        law4 = dict(bed.law4_parameters)
        law5 = dict(getattr(bed, "law5_parameters", {}))
        return {"band_h_m": float(law4["band_h_m"]), "rear_b_back_m": float(law4["rear_b_back_m"]),
                "lead_b_lead_m": float(law5.get("lead_b_lead_m", 0.)), "core_s_core_m": float(law5.get("core_s_core_m", 0.)),
                "edge_half_width_m": float(bed.config.blade_half_width), "theta_max_deg": float(law4["theta_max_deg"]),
                "theta_plow_deg": float(getattr(bed, "law8_parameters", {}).get("theta_plow_deg", float("nan")))}

    # ------------------------------------------------------------------ device side (graph capturable)
    def before_step(self):
        bed = self.bed
        wp.copy(self.code_prev, bed.code)
        wp.copy(self.work_prev, bed.contact_work)
        wp.copy(self.x_prev, bed.x)
        wp.copy(self.v_prev, bed.v)

    def after_step(self, advanced=True):
        bed, shift, c = self.bed, 1 if advanced else 0, self.bed.config
        law4 = bed.law4_parameters
        law5 = getattr(bed, "law5_parameters", {"lead_b_lead_m": 0., "core_s_core_m": 0.})
        with wp.ScopedDevice(self.device):
            wp.launch(_cd_segments, self.segments, [bed.poses, bed.twists, bed.cuts, bed.index, shift, self.max_cuts,
                c.blade_half_width, self.segments, bed.segment_ok, bed.segment_plow, self.acc, self.scratch])
            wp.launch(_cd_particles, bed.n, [bed.x, bed.v, self.x_prev, self.v_prev, bed.mass, bed.code, self.code_prev,
                bed.contact_work, self.work_prev, bed.progress, bed.poses, bed.twists, bed.cuts, bed.index, shift,
                self.max_cuts, self.ternary, bed.stroke, bed.segment_ok, bed.segment_plow, self.segments,
                c.blade_half_width, float(law4["band_h_m"]), float(law4["rear_b_back_m"]),
                float(law5["lead_b_lead_m"]), float(law5["core_s_core_m"]), self.law, c.dt, bed.profile,
                len(bed.blade_profile_xz), float(bed.radius), c.contact_stiffness, int(c.tool), self.substep,
                self.acc, self.scratch, self.first_fail, self.only_fail, self.lead_plow_blocked, self.min_f_stroke,
                self.min_f_armed, self.assign_substep, self.assign_s, self.assign_f, self.assign_x,
                self.assign_flags])
            wp.launch(_cd_summary, 1, [bed.poses, bed.twists, bed.cuts, bed.index, shift, bed.stroke,
                bed.stroke_engaged, bed.stroke_any_motion, bed.stroke_travel, bed.segment_ok, bed.segment_plow,
                self.segments, self.substep, self.scratch, self.summary_capacity, self.rows, self.overflow])

    # ------------------------------------------------------------------ host side
    def end_control_step(self):
        """Move the per-segment sums of the finished control step to the host table and zero them."""
        self.segment_rows.append(self.acc.numpy().copy())
        self.acc.zero_()

    def check(self):
        substeps, overflow = int(self.substep.numpy()[0]), int(self.overflow.numpy()[0])
        if overflow:
            raise CutDiagnosticsOverflowError(
                f"cut_diagnostics_v1: {overflow} substeps beyond the summary capacity {self.summary_capacity}")
        return {"substeps": substeps, "summary_overflow": overflow}

    def substep_table(self):
        rows = min(int(self.substep.numpy()[0]), self.summary_capacity)
        return self.rows.numpy()[:rows].copy()

    def segment_table(self):
        """(control steps, segments, columns); a final partial step is included if it has data."""
        tables = list(self.segment_rows)
        pending = self.acc.numpy()
        if np.any(pending):
            tables.append(pending.copy())
        if not tables:
            return np.zeros((0, self.segments, len(SEGMENT_COLUMNS)), np.float32)
        return np.stack(tables)

    def particle_table(self):
        n, k = self.n, self.max_cuts
        return {"first_fail": self.first_fail.numpy().reshape(n, k, len(REASONS)),
                "only_fail": self.only_fail.numpy().reshape(n, k, len(REASONS)),
                "lead_plow_blocked": self.lead_plow_blocked.numpy().reshape(n, k),
                "min_f_stroke_m": self.min_f_stroke.numpy().reshape(n, k),
                "min_f_armed_m": self.min_f_armed.numpy().reshape(n, k),
                "assign_substep": self.assign_substep.numpy().reshape(n, k),
                "assign_s_m": self.assign_s.numpy().reshape(n, k),
                "assign_f_m": self.assign_f.numpy().reshape(n, k),
                "assign_x_m": self.assign_x.numpy().reshape(n, k, 3),
                "assign_flags": self.assign_flags.numpy().reshape(n, k)}

    def arrays(self):
        """Everything as NumPy arrays (for np.savez)."""
        out = {f"particle_{key}": value for key, value in self.particle_table().items()}
        out["substep_rows"] = self.substep_table()
        out["segment_rows"] = self.segment_table()
        return out

    def metadata(self):
        table = self.particle_table()
        flags = table["assign_flags"][table["assign_substep"] >= 0]
        rows = self.substep_table()
        return {"version": CUT_DIAGNOSTICS_VERSION, "enabled": True, "solver_module": type(self.bed).__module__,
                "cutting_law": self.law, "code_encoding": "ternary" if self.ternary else "packed2",
                "max_cuts": self.max_cuts, "particles": self.n, "segments": self.segments,
                "summary_capacity": self.summary_capacity, **self.check(), "parameters": self.parameters(),
                "segment_columns": list(SEGMENT_COLUMNS), "substep_columns": list(SUBSTEP_COLUMNS),
                "reasons": list(REASONS), "assign_flags": dict(ASSIGN_FLAGS),
                "ill_float_relative": ILL_FLOAT_REL, "ill_geom_relative": ILL_GEOM_REL,
                "new_digits": int(len(flags)),
                "new_digits_by_flag": {name: int(np.count_nonzero(flags & bit)) for name, bit in ASSIGN_FLAGS.items()},
                "substep_rows_sha256": _sha(rows),
                "index_semantics": "pose/twist/cut at index-1 after step(advance=True)",
                "scope": "Read-only record of the solver's own cutting decisions (oracle data, not an observation)"}


# ---------------------------------------------------------------------- host-side analysis helpers (NumPy)

def digits_of(codes, cut, encoding):
    codes = np.asarray(codes, np.int64)
    if encoding == "packed2":
        return (codes >> (2*cut)) & 3
    return (codes//3**cut) % 3


UNASSIGNED_CLASSES = ("never_within_width_and_band_while_armed", "within_band_only_while_stroke_off",
                      "edge_never_passed_while_stroke_on", "passed_but_beyond_rear_limit",
                      "passed_but_segment_gate_closed", "passed_other")


def classify_unassigned(table, cut, mask):
    """Decisive reason for each particle in ``mask`` whose digit ``cut`` stayed 0 (UNASSIGNED_CLASSES index).

    - never_within_width_and_band_while_armed: |l| > L or |s| > h whenever the digit was armed;
    - within_band_only_while_stroke_off: inside width and band at some armed substep, never with the stroke on;
    - edge_never_passed_while_stroke_on: with the stroke on and inside width and band, f > 0 at every substep
      (the edge never got behind it: insufficient travel, or it moved ahead with the edge);
    - passed_but_beyond_rear_limit / passed_but_segment_gate_closed: it was behind the edge with the stroke on but
      the rear limit or the motion gate was the only failing condition at some substep;
    - passed_other: behind the edge with the stroke on, failing a combination of conditions.
    """
    mask = np.asarray(mask, bool)
    idx = np.flatnonzero(mask)
    armed = table["min_f_armed_m"][idx, cut]
    stroke = table["min_f_stroke_m"][idx, cut]
    only = table["only_fail"][idx, cut]
    out = np.full(len(idx), 5, np.int8)
    out[armed > 1e8] = 0
    out[(armed <= 1e8) & (stroke > 1e8)] = 1
    passed = stroke <= 0.
    out[(stroke <= 1e8) & ~passed] = 2
    out[passed & (only[:, 3] > 0)] = 3
    out[passed & (only[:, 3] == 0) & (only[:, 4] > 0)] = 4
    return idx, out


def direct_bridges(nodes, weights, gradients, digits):
    """Indices of digit-0 particles whose stencil touches nodes holding digit 1 and digit 2 particles
    (law8_validation.bridge_report rule, returning the particles)."""
    nodes = np.asarray(nodes, np.int64)
    valid = (np.asarray(weights) > 0) | (np.linalg.norm(np.asarray(gradients), axis=-1) > 0)
    has = {}
    for k in (1, 2):
        has[k] = np.zeros(nodes.max()+1, bool)
        has[k][nodes[digits == k][valid[digits == k]]] = True
    zero = np.flatnonzero(digits == 0)
    touches = lambda k: (has[k][nodes[zero]] & valid[zero]).any(1)  # noqa: E731
    return zero[touches(1) & touches(2)]
