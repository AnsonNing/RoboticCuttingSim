"""Read-only offline evaluator of two candidate exit rules, ``exit_candidates_v1`` (WP-12 Phase 4, E1).

This is a DIAGNOSTIC. It implements no cutting law, writes no label into the physics run and changes no solver
module. It observes an unchanged ``tlmpm_v5`` solver running cutting law 8. In every armed substep, after the
solver's own step, it evaluates on the solver's current state what two candidate rules *would* label. The
results go into shadow arrays owned by this object:

* ``proposed``: the breakthrough-gated lead rule of the Phase 4a draft (WP12_CUTTING_PHYSICS.md §8), REJECTED by
  the xhigh review (WP12_PHASE4_XHIGH_REVIEW.md §3.1-3.2). B_k = 1 iff S = c+1, M_k (seg_ok) and no particle whose
  predicted digit c is 0 lies in segment k's core strip ahead {0 < f <= b_lead, |s| <= s_core}. The law-5 lead
  rule (0 < f <= b_lead, s_core < |s| <= h) then runs on segments with plow_k OR B_k.
* ``exit_ligament``: L9-EL, the two-sided exit-ligament rule of the review §3.3 (evaluated, not approved).
  - N-[k] / N+[k] count particles of ALL codes in segment k's core strip ahead, by the sign of s.
  - lig = N- > 0 and N+ > 0. While S = c+1: seen[k] |= lig and X[k] = seg_ok[k] and seen[k] and not lig. When
    S != c+1, or the armed digit changes, seen = X = 0.
  - Then every particle whose predicted digit c is 0, with |l| <= L, 0 < f <= b_lead, X[seg(l)] and |s| <= h,
    would take the side of s.

Both use the solver's parameters unchanged: b_lead = 2 dx, s_core = dx, h = 6 dx, L = the blade half width and
the solver's 1-spacing edge segments. s is ``_swept_side`` with the relative swept-normal fallback of the review
§4 (blade normal when |m x e|^2 <= max(1e-20, (1e-5 |m|)^2)). Substeps in which that fallback differs from the
solver's absolute one are counted (``relative_fallback_particle_evals``).

Why the shadow is exact in the labelling logic (up to dynamics)
-------------------------------------------------------------
- A real rule would run after ``_update_stroke`` and before ``_assign_band``. This evaluator runs after the full
  step, at the same x (assignment moves no particle).
- The L9-EL counts include all codes, so this substep's band/lead labels cannot change them.
- The proposed B_k counts digit-0 particles with f > 0 and |s| <= s_core. This substep's band labels (f <= 0) and
  lead labels (|s| > s_core) are disjoint from that strip.
- A particle is eligible only while the law-8 digit and the candidate's shadow digit are both 0. So the predicted
  digit is the shadow digit if the candidate labelled the particle first, else the law-8 digit.
- What the shadow cannot show is the dynamics that real labels would cause (field splits, FLIP transfer, ADR 0010).
  It predicts labels and topology only.

Usage (after ``cut_diagnostics.CutDiagnosticsRecorder.after_step`` if both are attached; graph capturable)::

    bed.step()
    candidates.after_step()        # five read-only launches
    ...
    candidates.end_control_step()  # host: per-segment sums and end-of-step state of the finished control step

Oracle data: solver internals, never a policy observation.
"""
import hashlib

import numpy as np
import warp as wp

EXIT_CANDIDATES_VERSION = "exit_candidates_v1"

# Per edge segment, summed over the substeps of a control step.
SEGMENT_COLUMNS = ("armed_substeps", "stroke_gate_substeps", "material_substeps", "proposed_B_substeps",
                   "proposed_B_on_material_substeps", "ligament_substeps", "seen_substeps", "exit_X_substeps",
                   "n_neg_sum", "n_pos_sum", "core_digit0_sum", "proposed_new", "exit_ligament_new")
# Per edge segment, instantaneous value at the last substep of a control step.
STATE_COLUMNS = ("armed", "stroke_gate", "n_any", "n_neg", "n_pos", "core_digit0", "proposed_B", "ligament",
                 "seen", "exit_X")
# Per substep.
SUBSTEP_COLUMNS = ("substep", "trajectory_index", "cut", "stroke", "stroke_gate_segments", "material_segments",
                   "proposed_B_segments", "proposed_B_on_material", "ligament_segments", "seen_segments",
                   "exit_X_segments", "proposed_new", "exit_ligament_new", "relative_fallback_particle_evals")
_S_PROP = wp.constant(0)
_S_EL = wp.constant(1)
_S_FALLBACK = wp.constant(2)
_SCRATCH = 4


class ExitCandidatesOverflowError(RuntimeError):
    """More substeps than the evaluator's per-substep capacity."""


@wp.func
def _ec_segment(y: float, half_width: float, segments: int):
    # tlmpm_v5 _edge_segment (same operations).
    k = int((y+half_width)/(2.0*half_width)*float(segments))
    return wp.clamp(k, 0, segments-1)


@wp.func
def _ec_side(xp: wp.vec3, pose: wp.transform, twist: wp.spatial_vector, local: wp.vec3):
    """tlmpm_v5 _swept_side with the relative fallback (review §4); returns (s, fallback differs from solver's)."""
    along = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    normal = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))
    tool_v = wp.vec3(twist[0], twist[1], twist[2])
    angular = wp.vec3(twist[3], twist[4], twist[5])
    segment_motion = tool_v + wp.cross(angular, along * local[1])
    swept_normal = wp.cross(segment_motion, along)
    swept2 = wp.length_sq(swept_normal)
    threshold = wp.max(1.0e-20, 1.0e-10*wp.length_sq(segment_motion))
    side = local[2]
    differs = float(0.0)
    if swept2 > threshold:
        swept_normal = wp.normalize(swept_normal)
        if wp.dot(swept_normal, normal) < 0.0:
            swept_normal = -swept_normal
        side = wp.dot(xp - wp.transform_get_translation(pose), swept_normal)
    elif swept2 > 1.0e-20:
        differs = 1.0
    return wp.vec2(side, differs)


@wp.kernel
def _ec_count(x: wp.array(dtype=wp.vec3), code: wp.array(dtype=int), shadow_prop: wp.array(dtype=int),
              poses: wp.array(dtype=wp.transform), twists: wp.array(dtype=wp.spatial_vector),
              cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int, max_cuts: int,
              edge_width: float, segments: int, lead: float, core: float,
              n_any: wp.array(dtype=int), n_neg: wp.array(dtype=int), n_pos: wp.array(dtype=int),
              core_digit0: wp.array(dtype=int)):
    p = wp.tid()
    i = index[0]-shift
    if i < 0:
        return
    cut = cuts[i]
    if cut < 0 or cut >= max_cuts:
        return
    pose = poses[i]
    local = wp.transform_point(wp.transform_inverse(pose), x[p])
    if wp.abs(local[1]) > edge_width:
        return
    k = _ec_segment(local[1], edge_width, segments)
    wp.atomic_add(n_any, k, 1)
    if local[0] > 0.0 and local[0] <= lead:
        side = _ec_side(x[p], pose, twists[i], local)[0]
        if wp.abs(side) <= core:
            if side < 0.0:
                wp.atomic_add(n_neg, k, 1)
            else:
                wp.atomic_add(n_pos, k, 1)
            if ((code[p] >> (2*cut)) & 3) == 0 and ((shadow_prop[p] >> (2*cut)) & 3) == 0:
                wp.atomic_add(core_digit0, k, 1)


@wp.kernel
def _ec_segments(cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int, max_cuts: int,
                 stroke: wp.array(dtype=int), seg_ok: wp.array(dtype=int), last_cut: wp.array(dtype=int),
                 n_any: wp.array(dtype=int), n_neg: wp.array(dtype=int), n_pos: wp.array(dtype=int),
                 core_digit0: wp.array(dtype=int), prop_B: wp.array(dtype=int), lig: wp.array(dtype=int),
                 seen: wp.array(dtype=int), exit_X: wp.array(dtype=int), acc: wp.array2d(dtype=float),
                 state: wp.array2d(dtype=int)):
    k = wp.tid()
    i = index[0]-shift
    armed = int(0)
    cut = int(-1)
    if i >= 0:
        cut = cuts[i]
        if cut >= 0 and cut < max_cuts:
            armed = 1
    on = int(0)
    if armed != 0 and stroke[0] == cut+1:
        on = 1
    gate = int(0)
    if on != 0 and seg_ok[k] != 0:
        gate = 1
    if on == 0 or cut != last_cut[0]:
        seen[k] = 0
        exit_X[k] = 0
    b = int(0)
    l = int(0)
    if on != 0:
        if n_neg[k] > 0 and n_pos[k] > 0:
            l = 1
            seen[k] = 1
        x_k = int(0)
        if gate != 0 and seen[k] != 0 and l == 0:
            x_k = 1
        exit_X[k] = x_k
        if gate != 0 and core_digit0[k] == 0:
            b = 1
    prop_B[k] = b
    lig[k] = l
    if armed != 0:
        acc[k, 0] = acc[k, 0]+1.0
        acc[k, 1] = acc[k, 1]+float(gate)
        if n_any[k] > 0:
            acc[k, 2] = acc[k, 2]+1.0
            acc[k, 4] = acc[k, 4]+float(b)
        acc[k, 3] = acc[k, 3]+float(b)
        acc[k, 5] = acc[k, 5]+float(l)
        acc[k, 6] = acc[k, 6]+float(seen[k])
        acc[k, 7] = acc[k, 7]+float(exit_X[k])
        acc[k, 8] = acc[k, 8]+float(n_neg[k])
        acc[k, 9] = acc[k, 9]+float(n_pos[k])
        acc[k, 10] = acc[k, 10]+float(core_digit0[k])
    state[k, 0] = armed
    state[k, 1] = gate
    state[k, 2] = n_any[k]
    state[k, 3] = n_neg[k]
    state[k, 4] = n_pos[k]
    state[k, 5] = core_digit0[k]
    state[k, 6] = b
    state[k, 7] = l
    state[k, 8] = seen[k]
    state[k, 9] = exit_X[k]


@wp.kernel
def _ec_assign(x: wp.array(dtype=wp.vec3), code: wp.array(dtype=int), poses: wp.array(dtype=wp.transform),
               twists: wp.array(dtype=wp.spatial_vector), cuts: wp.array(dtype=int), index: wp.array(dtype=int),
               shift: int, max_cuts: int, stroke: wp.array(dtype=int), seg_ok: wp.array(dtype=int),
               seg_plow: wp.array(dtype=int), prop_B: wp.array(dtype=int), exit_X: wp.array(dtype=int),
               seen: wp.array(dtype=int), edge_width: float, segments: int, band: float, lead: float, core: float,
               substep: wp.array(dtype=int), acc: wp.array2d(dtype=float), scratch: wp.array(dtype=float),
               shadow_prop: wp.array(dtype=int), shadow_el: wp.array(dtype=int),
               prop_substep: wp.array(dtype=int), prop_s: wp.array(dtype=float), prop_f: wp.array(dtype=float),
               prop_flags: wp.array(dtype=int), el_substep: wp.array(dtype=int), el_s: wp.array(dtype=float),
               el_f: wp.array(dtype=float), prop_blocked: wp.array(dtype=int), el_blocked: wp.array(dtype=int),
               el_unseen: wp.array(dtype=int)):
    p = wp.tid()
    i = index[0]-shift
    if i < 0:
        return
    cut = cuts[i]
    if cut < 0 or cut >= max_cuts or stroke[0] != cut+1:
        return
    if ((code[p] >> (2*cut)) & 3) != 0:
        return
    pose = poses[i]
    local = wp.transform_point(wp.transform_inverse(pose), x[p])
    if wp.abs(local[1]) > edge_width or local[0] <= 0.0 or local[0] > lead:
        return
    k = _ec_segment(local[1], edge_width, segments)
    if seg_ok[k] == 0:
        return
    sv = _ec_side(x[p], pose, twists[i], local)
    side = sv[0]
    if sv[1] > 0.0:
        wp.atomic_add(scratch, _S_FALLBACK, 1.0)
    if wp.abs(side) > band:
        return
    digit = 2
    if side < 0.0:
        digit = 1
    slot = p*max_cuts+cut
    # Proposed (rejected) breakthrough-gated lead: law-5 lead region on plow_k OR B_k.
    if ((shadow_prop[p] >> (2*cut)) & 3) == 0 and wp.abs(side) > core:
        if seg_plow[k] != 0 or prop_B[k] != 0:
            shadow_prop[p] = shadow_prop[p] | (digit << (2*cut))
            prop_substep[slot] = substep[0]
            prop_s[slot] = side
            prop_f[slot] = local[0]
            flags = int(0)
            if prop_B[k] != 0:
                flags = flags | 1
            if seg_plow[k] != 0:
                flags = flags | 2
            prop_flags[slot] = flags
            wp.atomic_add(acc, k, 11, 1.0)
            wp.atomic_add(scratch, _S_PROP, 1.0)
        else:
            prop_blocked[slot] = prop_blocked[slot]+1
    # L9-EL exit-ligament rule: lead region including the core, on exit segments X_k.
    if ((shadow_el[p] >> (2*cut)) & 3) == 0:
        if exit_X[k] != 0:
            shadow_el[p] = shadow_el[p] | (digit << (2*cut))
            el_substep[slot] = substep[0]
            el_s[slot] = side
            el_f[slot] = local[0]
            wp.atomic_add(acc, k, 12, 1.0)
            wp.atomic_add(scratch, _S_EL, 1.0)
        elif seen[k] != 0:
            el_blocked[slot] = el_blocked[slot]+1      # the segment's ligament is still there (or the gate shut)
        else:
            el_unseen[slot] = el_unseen[slot]+1        # the segment never had a two-sided ligament in this stroke


@wp.kernel
def _ec_summary(cuts: wp.array(dtype=int), index: wp.array(dtype=int), shift: int, max_cuts: int,
                stroke: wp.array(dtype=int), segments: int, state: wp.array2d(dtype=int),
                last_cut: wp.array(dtype=int), substep: wp.array(dtype=int), scratch: wp.array(dtype=float),
                capacity: int, rows: wp.array2d(dtype=float), overflow: wp.array(dtype=int)):
    i = index[0]-shift
    row = substep[0]
    cut = int(-1)
    if i >= 0:
        cut = cuts[i]
    if row < capacity:
        gate = int(0)
        material = int(0)
        b = int(0)
        b_mat = int(0)
        l = int(0)
        s = int(0)
        x = int(0)
        for k in range(segments):
            gate += state[k, 1]
            if state[k, 2] > 0:
                material += 1
                b_mat += state[k, 6]
            b += state[k, 6]
            l += state[k, 7]
            s += state[k, 8]
            x += state[k, 9]
        rows[row, 0] = float(row)
        rows[row, 1] = float(i)
        rows[row, 2] = float(cut)
        rows[row, 3] = float(stroke[0])
        rows[row, 4] = float(gate)
        rows[row, 5] = float(material)
        rows[row, 6] = float(b)
        rows[row, 7] = float(b_mat)
        rows[row, 8] = float(l)
        rows[row, 9] = float(s)
        rows[row, 10] = float(x)
        rows[row, 11] = scratch[_S_PROP]
        rows[row, 12] = scratch[_S_EL]
        rows[row, 13] = scratch[_S_FALLBACK]
    else:
        wp.atomic_add(overflow, 0, 1)
    if cut >= 0 and cut < max_cuts:
        last_cut[0] = cut
    else:
        last_cut[0] = -1
    substep[0] = row+1
    for j in range(4):
        scratch[j] = 0.0


def _sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def predicted_codes(codes, shadow, max_cuts=1):
    """Law-8 packed codes with each digit replaced by the shadow digit wherever the candidate labelled first."""
    out = np.asarray(codes, np.int64).copy()
    shadow = np.asarray(shadow, np.int64)
    for c in range(int(max_cuts)):
        mask = 3 << (2*c)
        digit = (shadow >> (2*c)) & 3
        hit = digit != 0
        out[hit] = (out[hit] & ~mask) | (digit[hit] << (2*c))
    return out.astype(np.int32)


class ExitCandidateEvaluator:
    """``exit_candidates_v1`` bound to one tlmpm_v5 ``TLMPM`` running law 8 (read-only, offline, oracle-only)."""

    version = EXIT_CANDIDATES_VERSION

    def __init__(self, bed, *, summary_capacity=1 << 18):
        if getattr(bed, "code_encoding", None) != "packed2" or int(getattr(bed, "cutting_law", -1)) != 8:
            raise ValueError("exit_candidates_v1 evaluates a tlmpm_v5 (packed codes) solver running cutting law 8")
        self.bed, self.device = bed, bed.device
        self.max_cuts = int(bed.max_cuts)
        self.segments = int(bed.edge_segments)
        self.summary_capacity = int(summary_capacity)
        self.segment_rows, self.state_rows = [], []
        n, k, s = int(bed.n), self.max_cuts, self.segments
        self.n = n
        with wp.ScopedDevice(self.device):
            self.n_any = wp.zeros(s, dtype=int)
            self.n_neg = wp.zeros(s, dtype=int)
            self.n_pos = wp.zeros(s, dtype=int)
            self.core_digit0 = wp.zeros(s, dtype=int)
            self.prop_B = wp.zeros(s, dtype=int)
            self.lig = wp.zeros(s, dtype=int)
            self.seen = wp.zeros(s, dtype=int)
            self.exit_X = wp.zeros(s, dtype=int)
            self.last_cut = wp.full(1, -1, dtype=int)
            self.acc = wp.zeros((s, len(SEGMENT_COLUMNS)), dtype=float)
            self.state = wp.zeros((s, len(STATE_COLUMNS)), dtype=int)
            self.substep = wp.zeros(1, dtype=int)
            self.scratch = wp.zeros(_SCRATCH, dtype=float)
            self.overflow = wp.zeros(1, dtype=int)
            self.rows = wp.zeros((self.summary_capacity, len(SUBSTEP_COLUMNS)), dtype=float)
            self.shadow_prop = wp.zeros(n, dtype=int)
            self.shadow_el = wp.zeros(n, dtype=int)
            self.prop_substep = wp.full(n*k, -1, dtype=int)
            self.prop_s = wp.zeros(n*k, dtype=float)
            self.prop_f = wp.zeros(n*k, dtype=float)
            self.prop_flags = wp.zeros(n*k, dtype=int)
            self.el_substep = wp.full(n*k, -1, dtype=int)
            self.el_s = wp.zeros(n*k, dtype=float)
            self.el_f = wp.zeros(n*k, dtype=float)
            self.prop_blocked = wp.zeros(n*k, dtype=int)
            self.el_blocked = wp.zeros(n*k, dtype=int)
            self.el_unseen = wp.zeros(n*k, dtype=int)

    def parameters(self):
        bed = self.bed
        law4, law5 = bed.law4_parameters, bed.law5_parameters
        return {"band_h_m": float(law4["band_h_m"]), "lead_b_lead_m": float(law5["lead_b_lead_m"]),
                "core_s_core_m": float(law5["core_s_core_m"]), "edge_half_width_m": float(bed.config.blade_half_width),
                "edge_segments": self.segments, "theta_plow_deg": float(bed.law8_parameters["theta_plow_deg"]),
                "swept_normal_fallback": "relative: blade normal when |m x e|^2 <= max(1e-20, (1e-5 |m|)^2)"}

    # ------------------------------------------------------------------ device side (graph capturable)
    def after_step(self, advanced=True):
        bed, shift = self.bed, 1 if advanced else 0
        c = bed.config
        law4, law5 = bed.law4_parameters, bed.law5_parameters
        band, lead, core = float(law4["band_h_m"]), float(law5["lead_b_lead_m"]), float(law5["core_s_core_m"])
        with wp.ScopedDevice(self.device):
            self.n_any.zero_()
            self.n_neg.zero_()
            self.n_pos.zero_()
            self.core_digit0.zero_()
            wp.launch(_ec_count, bed.n, [bed.x, bed.code, self.shadow_prop, bed.poses, bed.twists, bed.cuts, bed.index,
                shift, self.max_cuts, c.blade_half_width, self.segments, lead, core, self.n_any, self.n_neg, self.n_pos,
                self.core_digit0])
            wp.launch(_ec_segments, self.segments, [bed.cuts, bed.index, shift, self.max_cuts, bed.stroke,
                bed.segment_ok, self.last_cut, self.n_any, self.n_neg, self.n_pos, self.core_digit0, self.prop_B,
                self.lig, self.seen, self.exit_X, self.acc, self.state])
            wp.launch(_ec_assign, bed.n, [bed.x, bed.code, bed.poses, bed.twists, bed.cuts, bed.index, shift,
                self.max_cuts, bed.stroke, bed.segment_ok, bed.segment_plow, self.prop_B, self.exit_X, self.seen,
                c.blade_half_width, self.segments, band, lead, core, self.substep, self.acc, self.scratch,
                self.shadow_prop, self.shadow_el, self.prop_substep, self.prop_s, self.prop_f, self.prop_flags,
                self.el_substep, self.el_s, self.el_f, self.prop_blocked, self.el_blocked, self.el_unseen])
            wp.launch(_ec_summary, 1, [bed.cuts, bed.index, shift, self.max_cuts, bed.stroke, self.segments,
                self.state, self.last_cut, self.substep, self.scratch, self.summary_capacity, self.rows,
                self.overflow])

    # ------------------------------------------------------------------ host side
    def end_control_step(self):
        self.segment_rows.append(self.acc.numpy().copy())
        self.state_rows.append(self.state.numpy().copy())
        self.acc.zero_()

    def check(self):
        substeps, overflow = int(self.substep.numpy()[0]), int(self.overflow.numpy()[0])
        if overflow:
            raise ExitCandidatesOverflowError(
                f"exit_candidates_v1: {overflow} substeps beyond the summary capacity {self.summary_capacity}")
        return {"substeps": substeps, "summary_overflow": overflow}

    def substep_table(self):
        rows = min(int(self.substep.numpy()[0]), self.summary_capacity)
        return self.rows.numpy()[:rows].copy()

    def segment_table(self):
        tables = list(self.segment_rows)
        pending = self.acc.numpy()
        if np.any(pending):
            tables.append(pending.copy())
        if not tables:
            return np.zeros((0, self.segments, len(SEGMENT_COLUMNS)), np.float32)
        return np.stack(tables)

    def state_table(self):
        if not self.state_rows:
            return np.zeros((0, self.segments, len(STATE_COLUMNS)), np.int32)
        return np.stack(self.state_rows)

    def shadow_codes(self):
        return {"proposed": self.shadow_prop.numpy().copy(), "exit_ligament": self.shadow_el.numpy().copy()}

    def particle_table(self):
        n, k = self.n, self.max_cuts
        return {"proposed_substep": self.prop_substep.numpy().reshape(n, k),
                "proposed_s_m": self.prop_s.numpy().reshape(n, k), "proposed_f_m": self.prop_f.numpy().reshape(n, k),
                "proposed_flags": self.prop_flags.numpy().reshape(n, k),
                "proposed_blocked_substeps": self.prop_blocked.numpy().reshape(n, k),
                "exit_ligament_substep": self.el_substep.numpy().reshape(n, k),
                "exit_ligament_s_m": self.el_s.numpy().reshape(n, k),
                "exit_ligament_f_m": self.el_f.numpy().reshape(n, k),
                "exit_ligament_blocked_substeps": self.el_blocked.numpy().reshape(n, k),
                "exit_ligament_unseen_substeps": self.el_unseen.numpy().reshape(n, k)}

    def arrays(self):
        out = {f"particle_{key}": value for key, value in self.particle_table().items()}
        out.update({f"shadow_{key}": value for key, value in self.shadow_codes().items()})
        out["substep_rows"] = self.substep_table()
        out["segment_rows"] = self.segment_table()
        out["state_rows"] = self.state_table()
        return out

    def metadata(self):
        shadow = self.shadow_codes()
        return {"version": EXIT_CANDIDATES_VERSION, "label": "OFFLINE_CANDIDATE_EVALUATION (read-only; no label "
                "written into the physics run; not a cutting law)", "solver_module": type(self.bed).__module__,
                "cutting_law": int(self.bed.cutting_law), "max_cuts": self.max_cuts, "particles": self.n,
                "segments": self.segments, "summary_capacity": self.summary_capacity, **self.check(),
                "parameters": self.parameters(), "segment_columns": list(SEGMENT_COLUMNS),
                "state_columns": list(STATE_COLUMNS), "substep_columns": list(SUBSTEP_COLUMNS),
                "proposed_flags": {"B_k": 1, "plow_k": 2},
                "shadow_labels": {key: int(np.count_nonzero(value)) for key, value in shadow.items()},
                "substep_rows_sha256": _sha(self.substep_table()),
                "candidates": {"proposed": "Phase 4a draft breakthrough-gated lead (REJECTED by the xhigh review)",
                               "exit_ligament": "L9-EL two-sided exit-ligament rule (review §3.3; not approved)"},
                "index_semantics": "pose/twist/cut at index-1 after step(advance=True)",
                "scope": "Shadow labels predict what each candidate would assign on the law-8 trajectory; they do not "
                         "include the dynamics real labels would cause."}
