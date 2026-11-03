"""Read-only live material safety statistics without a full state download."""

import warp as wp


# Match the float32 NumPy norm's separate multiply/add operations. This module
# reads solver arrays only; its compiler options do not affect solver kernels.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.kernel
def _state_partials(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                    deformation: wp.array(dtype=wp.mat33), plastic: wp.array(dtype=float),
                    count: int, groups: int, partials: wp.array2d(dtype=float)):
    group = wp.tid()
    invalid = float(0.0)
    minimum_det = float(wp.inf)
    maximum_speed = float(0.0)
    maximum_plastic = -float(wp.inf)
    # Strided groups preserve coalesced particle reads and write disjoint rows.
    # Each thread reduces at most 64 particles; there are no global atomics.
    for p in range(group, count, groups):
        xp, vp, fp = x[p], v[p], deformation[p]
        valid = wp.isfinite(plastic[p])
        for axis in range(3):
            if not wp.isfinite(xp[axis]) or not wp.isfinite(vp[axis]):
                valid = False
            for column in range(3):
                if not wp.isfinite(fp[axis, column]):
                    valid = False
        if valid:
            # NumPy det computes in float64, then returns float32 for these
            # float32 inputs. Keep that precision here; no solver state changes.
            determinant = float(wp.determinant(wp.mat33d(fp)))
            squared = vp[0]*vp[0] + vp[1]*vp[1] + vp[2]*vp[2]
            minimum_det = wp.min(minimum_det, determinant)
            maximum_speed = wp.max(maximum_speed, wp.sqrt(squared))
            maximum_plastic = wp.max(maximum_plastic, plastic[p])
        else:
            invalid = 1.0
    partials[group, 0] = invalid
    partials[group, 1] = minimum_det
    partials[group, 2] = maximum_speed
    partials[group, 3] = maximum_plastic


@wp.kernel
def _combine_partials(partials: wp.array2d(dtype=float), groups: int,
                      summary: wp.array(dtype=float)):
    invalid = float(0.0)
    minimum_det = float(wp.inf)
    maximum_speed = float(0.0)
    maximum_plastic = -float(wp.inf)
    for group in range(groups):
        invalid = wp.max(invalid, partials[group, 0])
        minimum_det = wp.min(minimum_det, partials[group, 1])
        maximum_speed = wp.max(maximum_speed, partials[group, 2])
        maximum_plastic = wp.max(maximum_plastic, partials[group, 3])
    summary[0] = invalid
    summary[1] = minimum_det
    summary[2] = maximum_speed
    summary[3] = maximum_plastic


class MaterialDiagnosticsReduction:
    """Reusable scratch buffers for a single retained solver's float32 state."""

    def __init__(self, bed):
        self.bed = bed
        self.count = len(bed.x)
        if self.count == 0:
            raise ValueError("Material diagnostics require at least one particle")
        self.groups = (self.count + 63)//64
        self.partials = wp.empty((self.groups, 4), dtype=float, device=bed.x.device)
        self.summary = wp.empty(4, dtype=float, device=bed.x.device)

    def read(self):
        bed = self.bed
        wp.launch(_state_partials, self.groups,
                  [bed.x, bed.v, bed.F, bed.plastic, self.count, self.groups, self.partials],
                  device=bed.x.device)
        wp.launch(_combine_partials, 1, [self.partials, self.groups, self.summary], device=bed.x.device)
        values = self.summary.numpy()
        if values[0] != 0.:
            raise FloatingPointError("Nonfinite live TLMPM state")
        # As in the snapshot path, finite inputs are checked before reductions.
        # Float32 norm/determinant overflow may produce inf diagnostics and does
        # not introduce a new rejection criterion here.
        return {"minimum_detF": float(values[1]),
                "maximum_material_speed_m_s": float(values[2]),
                "maximum_plastic_strain": float(values[3])}
