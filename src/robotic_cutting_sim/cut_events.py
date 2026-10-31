"""Read-only Hermite cut-event recorder ``cut_events_v1`` for tlmpm_v5 (WP-12 Phase 3, review §4 R1).

The recorder observes the solver; it never writes x, v, F, Fpi, code or any other
solver array, and it does not change ``tlmpm_v5.py``. It is opt-in: a material
wrapper that is not given ``cut_events="cut_events_v1"`` creates no recorder and
launches nothing new.

Per substep the caller brackets the solver step::

    recorder.before_step()      # code_prev <- code (one device copy)
    bed.step()                  # unchanged tlmpm_v5 step (advance=True)
    recorder.after_step()       # two read-only kernels

Both calls are CUDA-graph capturable (all counters live on the device).

What ``after_step`` computes (review §4 R1, steps 1-5)
-----------------------------------------------------
Side assignment (``_assign_band`` / ``_assign_lead``) is the last code-writing
launch of ``step()`` and runs after G2P, so after ``step()`` the positions ``x``
and the codes are exactly the state the assignment saw. ``step(advance=True)``
increments ``index`` after the assignment, so the pose, twist and cut digit the
assignment used are at ``index - 1`` (``index`` for ``advance=False``; pass
``advanced=False`` then).

1. A particle p is *newly assigned* in cut k = cuts[i] when its digit k went
   0 -> d in {1, 2} during this step. Its swept normal n is recomputed exactly as
   ``tlmpm_v5._swept_side`` does (same operations; if the swept normal is
   degenerate, |n|^2 <= 1e-20, the side is the blade-frame z coordinate, i.e. the
   blade normal, flag ``FLAG_BLADE_Z_FALLBACK``). s_p = (x_p - o) . n with o the
   edge origin, which reproduces the solver's own side value bit for bit.
   Observed (not changed, tlmpm_v5 is frozen): for a pure draw along the edge at a
   generic orientation, m x e is float32 rounding noise above the 1e-20 threshold,
   so the solver's (and this record's) swept normal is not the blade normal
   (43 degrees off in one probed pose); the fallback triggers only for exactly
   parallel float32 vectors or zero motion.
2. For each lattice 6-neighbour q of p whose final digit k is 3 - d, s_q is
   evaluated at the same instant with the same normal and origin. If q was also
   newly assigned in this substep the link is written only from min(p, q)
   (``FLAG_BOTH_NEW``). If q got its digit in an earlier substep, s_q is still
   evaluated at p's assignment instant (the link becomes conflicting now).
   If the signs agree with the digits (digit 2 <=> s >= 0, the solver's rule),
   t = s_p / (s_p - s_q) in [0, 1); otherwise t = NaN and
   ``FLAG_SIGN_INCONSISTENT`` is set (the Hermite point then uses t = 1/2).
3. Reference Hermite point P = X_p + t (X_q - X_p) and reference normal
   N ∝ F̄ᵀ n with F̄ = (F_p + F_q)/2 (Nanson: n da = J F⁻ᵀ N dA).
4. The event (p, q, k, substep, flags, t, s_p, s_q, P, N, n, o) is appended with
   an atomic counter; entries beyond the capacity are counted, and ``check()``
   raises ``CutEventOverflowError`` like slot overflow.
5. One activation row per substep: global substep, trajectory index, cut, stroke,
   stroke_engaged, stroke_any_motion, number of segment_ok and segment_plow
   segments, newly assigned particles, links logged, and code changes that are
   not a 0 -> d write of the active digit (expected 0).

Every link of cut k that conflicts in the final codes is logged exactly once:
when its second endpoint is assigned. Codes are monotone (digits are written only
from 0), so every logged link stays conflicting. This is an audit record of what
the solver decided; it is oracle data and must stay out of policy observations.
"""
import hashlib

import numpy as np
import warp as wp

CUT_EVENTS_VERSION = "cut_events_v1"

FLAG_SIGN_INCONSISTENT = 1   # s_p, s_q signs disagree with the digits; t = NaN, P uses t = 1/2
FLAG_BOTH_NEW = 2            # q was also newly assigned in this substep (link written from min(p, q))
FLAG_LEAD = 4                # p lies ahead of the edge (edge-local x > 0): assigned by the lead rule
FLAG_BLADE_Z_FALLBACK = 8    # degenerate swept normal; side = blade-frame z (as _swept_side)
FLAG_DEGENERATE_REFERENCE_NORMAL = 16   # |F̄ᵀ n| <= 1e-12; N is zero

NEIGHBOUR_OFFSETS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], np.int64)

SUMMARY_COLUMNS = ("substep", "trajectory_index", "cut", "stroke", "stroke_engaged", "stroke_any_motion",
                   "segment_ok_count", "segment_plow_count", "newly_assigned", "links_logged",
                   "unexpected_code_changes")

LOG_DTYPE = np.dtype([("p", np.int32), ("q", np.int32), ("cut", np.int32), ("substep", np.int32),
                      ("flags", np.int32), ("t", np.float32), ("s_p", np.float32), ("s_q", np.float32),
                      ("hermite_reference", np.float32, (3,)), ("normal_reference", np.float32, (3,)),
                      ("normal_current", np.float32, (3,)), ("edge_origin", np.float32, (3,))])
SUMMARY_DTYPE = np.dtype([(name, np.int32) for name in SUMMARY_COLUMNS])

_INT_COLUMNS = ("p", "q", "cut", "substep", "flags")
_FLOAT_COLUMNS = ("t", "s_p", "s_q")
_VECTOR_COLUMNS = ("hermite_reference", "normal_reference", "normal_current", "edge_origin")


class CutEventOverflowError(RuntimeError):
    """More cut events (or substeps) than the recorder's fixed device capacity."""


@wp.func
def _ce_digit(code: int, cut: int):
    return (code >> (2*cut)) & 3


@wp.kernel
def _record_cut_events(x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33),
                       reference: wp.array(dtype=wp.vec3), code: wp.array(dtype=int),
                       code_prev: wp.array(dtype=int), neighbours: wp.array2d(dtype=int),
                       poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
                       cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int, max_cuts: int,
                       substep: wp.array(dtype=int), count: wp.array(dtype=int), overflow: wp.array(dtype=int),
                       scratch: wp.array(dtype=int), capacity: int,
                       out_int: wp.array2d(dtype=int), out_float: wp.array2d(dtype=float),
                       out_point: wp.array(dtype=wp.vec3), out_normal_ref: wp.array(dtype=wp.vec3),
                       out_normal_cur: wp.array(dtype=wp.vec3), out_origin: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    i = index[0]-shift
    cut = int(-1)
    if i >= 0:
        cut = cuts[i]
    cp = code[p]
    pp = code_prev[p]
    armed = cut >= 0 and cut < max_cuts
    if cp != pp:
        expected = int(0)
        if armed:
            if _ce_digit(pp, cut) == 0 and (cp ^ pp) == (_ce_digit(cp, cut) << (2*cut)):
                expected = 1
        if expected == 0:
            wp.atomic_add(scratch, 2, 1)
    if not armed:
        return
    d = _ce_digit(cp, cut)
    if d == 0 or _ce_digit(pp, cut) != 0:
        return
    wp.atomic_add(scratch, 0, 1)
    pose = poses[i]
    twist = twists[i]
    xp = x[p]
    local = wp.transform_point(wp.transform_inverse(pose), xp)
    origin = wp.transform_get_translation(pose)
    # Exactly tlmpm_v5._swept_side (same operations and order).
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    blade_normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    tool_v = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    segment_motion = tool_v + wp.cross(angular, along * local[1])
    swept_normal = wp.cross(segment_motion, along)
    side_p = local[2]
    n = blade_normal
    fallback = int(1)
    if wp.length_sq(swept_normal) > 1.0e-20:
        swept_normal = wp.normalize(swept_normal)
        if wp.dot(swept_normal, blade_normal) < 0.0:
            swept_normal = -swept_normal
        side_p = wp.dot(xp - origin, swept_normal)
        n = swept_normal
        fallback = 0
    base_flags = int(0)
    if fallback != 0:
        base_flags = base_flags | 8
    if local[0] > 0.0:
        base_flags = base_flags | 4
    other = 3-d
    for j in range(6):
        q = neighbours[p, j]
        if q >= 0:
            if _ce_digit(code[q], cut) == other:
                both_new = _ce_digit(code_prev[q], cut) == 0
                if not (both_new and q < p):
                    flags = base_flags
                    if both_new:
                        flags = flags | 2
                    xq = x[q]
                    side_q = float(0.0)
                    if fallback != 0:
                        side_q = wp.transform_point(wp.transform_inverse(pose), xq)[2]
                    else:
                        side_q = wp.dot(xq - origin, n)
                    consistent = (d == 2 and side_p >= 0.0 and side_q < 0.0) or (
                        d == 1 and side_p < 0.0 and side_q >= 0.0)
                    t = float(wp.nan)
                    tt = float(0.5)
                    if consistent:
                        t = side_p/(side_p-side_q)
                        tt = t
                    else:
                        flags = flags | 1
                    point = reference[p] + tt*(reference[q]-reference[p])
                    fbar = 0.5*(F[p]+F[q])
                    nref = wp.transpose(fbar)*n
                    length = wp.length(nref)
                    if length > 1.0e-12:
                        nref = nref/length
                    else:
                        nref = wp.vec3(0.0)
                        flags = flags | 16
                    wp.atomic_add(scratch, 1, 1)
                    slot = wp.atomic_add(count, 0, 1)
                    if slot < capacity:
                        out_int[slot, 0] = p
                        out_int[slot, 1] = q
                        out_int[slot, 2] = cut
                        out_int[slot, 3] = substep[0]
                        out_int[slot, 4] = flags
                        out_float[slot, 0] = t
                        out_float[slot, 1] = side_p
                        out_float[slot, 2] = side_q
                        out_point[slot] = point
                        out_normal_ref[slot] = nref
                        out_normal_cur[slot] = n
                        out_origin[slot] = origin
                    else:
                        wp.atomic_add(overflow, 0, 1)


@wp.kernel
def _record_cut_summary(cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int,
                        stroke: wp.array(dtype=int), engaged: wp.array(dtype=int), any_motion: wp.array(dtype=int),
                        segment_ok: wp.array(dtype=int), segment_plow: wp.array(dtype=int), segments: int,
                        substep: wp.array(dtype=int), scratch: wp.array(dtype=int), capacity: int,
                        summary: wp.array2d(dtype=int), summary_overflow: wp.array(dtype=int)):
    i = index[0]-shift
    cut = int(-1)
    if i >= 0:
        cut = cuts[i]
    ok = int(0)
    plow = int(0)
    for k in range(segments):
        if segment_ok[k] != 0:
            ok += 1
        if segment_plow[k] != 0:
            plow += 1
    row = substep[0]
    if row < capacity:
        summary[row, 0] = row
        summary[row, 1] = i
        summary[row, 2] = cut
        summary[row, 3] = stroke[0]
        summary[row, 4] = engaged[0]
        summary[row, 5] = any_motion[0]
        summary[row, 6] = ok
        summary[row, 7] = plow
        summary[row, 8] = scratch[0]
        summary[row, 9] = scratch[1]
        summary[row, 10] = scratch[2]
    else:
        wp.atomic_add(summary_overflow, 0, 1)
    wp.atomic_add(summary_overflow, 1, scratch[2])
    substep[0] = row+1
    scratch[0] = 0
    scratch[1] = 0
    scratch[2] = 0


def lattice_neighbours(lattice):
    """(N, 6) int32 particle index of each 6-neighbour (-x, +x, -y, +y, -z, +z) on the integer lattice; -1 absent."""
    lattice = np.asarray(lattice)
    if lattice.ndim != 2 or lattice.shape[1] != 3 or not np.issubdtype(lattice.dtype, np.integer):
        raise ValueError("The reference lattice must be an (N, 3) integer array")
    shifted = lattice.astype(np.int64)-lattice.min(0)+1
    dims = shifted.max(0)+2
    key = lambda v: (v[:, 0]*dims[1]+v[:, 1])*dims[2]+v[:, 2]  # noqa: E731
    keys = key(shifted)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    if len(ordered) > 1 and np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("Reference lattice indices must be unique")
    table = np.full((len(lattice), 6), -1, np.int32)
    for j, offset in enumerate(NEIGHBOUR_OFFSETS):
        wanted = key(shifted+offset)
        position = np.minimum(np.searchsorted(ordered, wanted), len(ordered)-1)
        found = ordered[position] == wanted
        table[found, j] = order[position[found]]
    return table


def _sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sort_log(log):
    """Canonical order (substep, cut, p, q): GPU atomics append in a race-dependent order."""
    return log[np.lexsort((log["q"], log["p"], log["cut"], log["substep"]))]


class CutEventRecorder:
    """``cut_events_v1`` bound to one tlmpm_v5 ``TLMPM`` and its reference lattice (oracle-only, read-only)."""

    version = CUT_EVENTS_VERSION

    def __init__(self, bed, lattice, *, capacity=None, summary_capacity=1 << 18, check_lattice=True):
        if getattr(bed, "code_encoding", None) != "packed2" or not hasattr(bed, "slot_code"):
            raise ValueError("cut_events_v1 records tlmpm_v5 (packed2) solvers only")
        lattice = np.asarray(lattice)
        if len(lattice) != bed.n:
            raise ValueError("The lattice must have one row per solver particle")
        self.bed = bed
        self.device = bed.device
        self.lattice = lattice.astype(np.int32)
        self.neighbours_host = lattice_neighbours(lattice)
        self.reference_host = bed.reference.numpy()
        if check_lattice:
            residual = self.reference_host.astype(np.float64)-self.lattice.astype(np.float64)*float(bed.spacing)
            if np.any(np.ptp(residual, axis=0) > 1e-3*float(bed.spacing)):
                raise ValueError("Reference points are not origin + lattice * spacing")
        self.capacity = int(max(65536, bed.n) if capacity is None else capacity)
        self.summary_capacity = int(summary_capacity)
        if self.capacity < 1 or self.summary_capacity < 1:
            raise ValueError("Recorder capacities must be positive")
        with wp.ScopedDevice(self.device):
            self.neighbours = wp.array(self.neighbours_host, dtype=int)
            self.code_prev = wp.zeros(bed.n, dtype=int)
            self.substep = wp.zeros(1, dtype=int)
            self.count = wp.zeros(1, dtype=int)
            self.overflow = wp.zeros(1, dtype=int)
            self.scratch = wp.zeros(3, dtype=int)
            self.summary_overflow = wp.zeros(2, dtype=int)  # [rows beyond capacity, unexpected code changes]
            self.out_int = wp.zeros((self.capacity, len(_INT_COLUMNS)), dtype=int)
            self.out_float = wp.zeros((self.capacity, len(_FLOAT_COLUMNS)), dtype=float)
            self.out_point = wp.zeros(self.capacity, dtype=wp.vec3)
            self.out_normal_ref = wp.zeros(self.capacity, dtype=wp.vec3)
            self.out_normal_cur = wp.zeros(self.capacity, dtype=wp.vec3)
            self.out_origin = wp.zeros(self.capacity, dtype=wp.vec3)
            self.summary = wp.zeros((self.summary_capacity, len(SUMMARY_COLUMNS)), dtype=int)

    # ------------------------------------------------------------------ device side (graph capturable)
    def before_step(self):
        wp.copy(self.code_prev, self.bed.code)

    def after_step(self, advanced=True):
        bed, shift = self.bed, 1 if advanced else 0
        with wp.ScopedDevice(self.device):
            wp.launch(_record_cut_events, bed.n, [bed.x, bed.F, bed.reference, bed.code, self.code_prev,
                self.neighbours, bed.poses, bed.twists, bed.cuts, bed.index, shift, bed.max_cuts,
                self.substep, self.count, self.overflow, self.scratch, self.capacity,
                self.out_int, self.out_float, self.out_point, self.out_normal_ref, self.out_normal_cur,
                self.out_origin])
            wp.launch(_record_cut_summary, 1, [bed.cuts, bed.index, shift, bed.stroke, bed.stroke_engaged,
                bed.stroke_any_motion, bed.segment_ok, bed.segment_plow, bed.edge_segments,
                self.substep, self.scratch, self.summary_capacity, self.summary, self.summary_overflow])

    def reset(self):
        """Clear the log, the activation rows and the global substep counter (arrays keep their addresses)."""
        for array in (self.substep, self.count, self.overflow, self.scratch, self.summary_overflow):
            array.zero_()

    # ------------------------------------------------------------------ host side
    def counters(self):
        count, overflow = int(self.count.numpy()[0]), int(self.overflow.numpy()[0])
        rows, unexpected = (int(v) for v in self.summary_overflow.numpy())
        return {"events_attempted": count, "events_logged": min(count, self.capacity), "event_overflow": overflow,
                "substeps": int(self.substep.numpy()[0]), "summary_overflow": rows,
                "unexpected_code_changes": unexpected}

    def check(self):
        """Raise CutEventOverflowError if events or substeps exceeded the fixed capacity (never silently truncated)."""
        counters = self.counters()
        if counters["event_overflow"] or counters["summary_overflow"]:
            raise CutEventOverflowError(
                f"cut_events_v1 overflow: {counters['event_overflow']} events beyond capacity {self.capacity}, "
                f"{counters['summary_overflow']} substeps beyond summary capacity {self.summary_capacity}; "
                "the log is incomplete. Rerun with a larger capacity.")
        return counters

    def log(self):
        """Logged events as a NumPy structured array (LOG_DTYPE), sorted by (substep, cut, p, q)."""
        count = min(int(self.count.numpy()[0]), self.capacity)
        out = np.zeros(count, LOG_DTYPE)
        ints, floats = self.out_int.numpy()[:count], self.out_float.numpy()[:count]
        for j, name in enumerate(_INT_COLUMNS):
            out[name] = ints[:, j]
        for j, name in enumerate(_FLOAT_COLUMNS):
            out[name] = floats[:, j]
        for name, array in zip(_VECTOR_COLUMNS, (self.out_point, self.out_normal_ref, self.out_normal_cur,
                                                 self.out_origin)):
            out[name] = array.numpy()[:count]
        return sort_log(out)

    def activation(self):
        """Per-substep activation rows (SUMMARY_DTYPE) in substep order."""
        rows = min(int(self.substep.numpy()[0]), self.summary_capacity)
        data = self.summary.numpy()[:rows]
        out = np.zeros(rows, SUMMARY_DTYPE)
        for j, name in enumerate(SUMMARY_COLUMNS):
            out[name] = data[:, j]
        return out

    def sha256(self):
        """SHA256 of the canonically sorted log (shape, dtype and bytes)."""
        return _sha(self.log())

    def metadata(self):
        log, activation = self.log(), self.activation()
        flags = log["flags"]
        return {"version": CUT_EVENTS_VERSION, "enabled": True, "solver_module": type(self.bed).__module__,
                "cutting_law": int(self.bed.cutting_law), "max_cuts": int(self.bed.max_cuts),
                "particles": int(self.bed.n), "capacity": self.capacity, "summary_capacity": self.summary_capacity,
                **self.counters(),
                "events_per_cut": {str(int(k)): int(np.count_nonzero(log["cut"] == k)) for k in np.unique(log["cut"])},
                "sign_inconsistent_links": int(np.count_nonzero(flags & FLAG_SIGN_INCONSISTENT)),
                "both_new_links": int(np.count_nonzero(flags & FLAG_BOTH_NEW)),
                "lead_links": int(np.count_nonzero(flags & FLAG_LEAD)),
                "blade_z_fallback_links": int(np.count_nonzero(flags & FLAG_BLADE_Z_FALLBACK)),
                "degenerate_reference_normal_links": int(np.count_nonzero(flags & FLAG_DEGENERATE_REFERENCE_NORMAL)),
                "log_sha256": _sha(log), "activation_sha256": _sha(activation),
                "lattice_sha256": _sha(self.lattice), "neighbours_sha256": _sha(self.neighbours_host),
                "reference_sha256": _sha(self.reference_host),
                "log_bytes": int(log.nbytes), "activation_bytes": int(activation.nbytes),
                "device_bytes": int(sum(a.capacity for a in vars(self).values() if isinstance(a, wp.array))),
                "index_semantics": "pose/twist/cut at index-1 after step(advance=True)",
                "scope": "Read-only record of the solver's own side assignment (oracle data, not an observation)"}


# ---------------------------------------------------------------------- pure NumPy reference (tests)

def _rotate(q, v):
    """Rotate v by the unit quaternion q (xyzw)."""
    q = np.asarray(q, np.float64)
    u, w = q[:3], q[3]
    v = np.asarray(v, np.float64)
    return v+2*w*np.cross(u, v)+2*np.cross(u, np.cross(u, v))


def reference_substep_events(code_prev, code, x, F, reference, neighbours, pose, twist, cut, substep, max_cuts):
    """NumPy (float64) reference of one ``after_step`` for the pose/twist/cut the assignment used.

    Returns (events LOG_DTYPE sorted, newly_assigned count). Same rules as ``_record_cut_events``.
    """
    code_prev, code = np.asarray(code_prev, np.int64), np.asarray(code, np.int64)
    if not 0 <= cut < max_cuts:
        return np.zeros(0, LOG_DTYPE), 0
    digit = lambda c: (c >> (2*cut)) & 3  # noqa: E731
    d_now, d_prev = digit(code), digit(code_prev)
    new = np.flatnonzero((d_now != 0) & (d_prev == 0))
    pose = np.asarray(pose, np.float64)
    origin, quat = pose[:3], pose[3:]
    along, blade_normal = _rotate(quat, [0, 1, 0]), _rotate(quat, [0, 0, 1])
    twist = np.asarray(twist, np.float64)
    x = np.asarray(x, np.float64)
    rows = []
    for p in new:
        d = int(d_now[p])
        rel = x[p]-origin
        local_y = rel@along
        motion = twist[:3]+np.cross(twist[3:], along*local_y)
        swept = np.cross(motion, along)
        flags = 0
        if swept@swept > 1e-20:
            n = swept/np.linalg.norm(swept)
            if n@blade_normal < 0:
                n = -n
        else:
            n = blade_normal
            flags |= FLAG_BLADE_Z_FALLBACK
        if rel@_rotate(quat, [1, 0, 0]) > 0:
            flags |= FLAG_LEAD
        s_p = rel@n
        for q in neighbours[p]:
            if q < 0 or d_now[q] != 3-d:
                continue
            both_new = d_prev[q] == 0
            if both_new and q < p:
                continue
            f = flags | (FLAG_BOTH_NEW if both_new else 0)
            s_q = (x[q]-origin)@n
            consistent = (d == 2 and s_p >= 0 and s_q < 0) or (d == 1 and s_p < 0 and s_q >= 0)
            if consistent:
                t = s_p/(s_p-s_q)
                tt = t
            else:
                t, tt = np.nan, .5
                f |= FLAG_SIGN_INCONSISTENT
            point = reference[p]+tt*(reference[q]-reference[p])
            fbar = .5*(np.asarray(F[p], np.float64)+np.asarray(F[q], np.float64))
            nref = fbar.T@n
            length = np.linalg.norm(nref)
            if length > 1e-12:
                nref = nref/length
            else:
                nref = np.zeros(3)
                f |= FLAG_DEGENERATE_REFERENCE_NORMAL
            rows.append((p, q, cut, substep, f, t, s_p, s_q, point, nref, n, origin))
    out = np.zeros(len(rows), LOG_DTYPE)
    for j, row in enumerate(rows):
        out[j] = row
    return sort_log(out), len(new)


def links_conflicting_in_cut(codes, neighbours, cut):
    """Set of (min, max) lattice links whose cut-``cut`` digits are {1, 2} in ``codes``."""
    digits = (np.asarray(codes, np.int64) >> (2*cut)) & 3
    p = np.repeat(np.arange(len(codes)), 6)
    q = neighbours.reshape(-1)
    keep = (q >= 0) & (p < q)
    p, q = p[keep], q[keep]
    hit = ((digits[p] == 1) & (digits[q] == 2)) | ((digits[p] == 2) & (digits[q] == 1))
    return set(zip(p[hit].tolist(), q[hit].tolist()))


def log_links(log, cut=None):
    rows = log if cut is None else log[log["cut"] == cut]
    return set(zip(np.minimum(rows["p"], rows["q"]).tolist(), np.maximum(rows["p"], rows["q"]).tolist()))
