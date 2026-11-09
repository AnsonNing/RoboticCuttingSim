"""Segmented per-env replacement of ``material_diagnostics._state_partials`` (ADR 0021 §3.3 note).

The frozen reduction strides its groups over all particles (``material_diagnostics.py:22``), so on a batch
its partials would mix envs. Here thread (e, g) strides over env e's particle range only; the per-particle
arithmetic (validity test, float64 determinant cast to float32, separate multiply/add speed norm) and the
module option ``fuse_fp=False`` are the frozen ones, and min/max combination is order independent, so each
env's summary equals the frozen reduction of that env alone.
"""
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

# summary columns
INVALID, MINIMUM_DET, MAXIMUM_SPEED, MAXIMUM_PLASTIC = 0, 1, 2, 3


@wp.kernel
def _env_state_partials(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                        deformation: wp.array(dtype=wp.mat33), plastic: wp.array(dtype=float),
                        particle_offset: wp.array(dtype=int), groups: int, partials: wp.array2d(dtype=float)):
    e, group = wp.tid()
    invalid = float(0.0)
    minimum_det = float(wp.inf)
    maximum_speed = float(0.0)
    maximum_plastic = -float(wp.inf)
    for p in range(particle_offset[e]+group, particle_offset[e+1], groups):
        xp, vp, fp = x[p], v[p], deformation[p]
        valid = wp.isfinite(plastic[p])
        for axis in range(3):
            if not wp.isfinite(xp[axis]) or not wp.isfinite(vp[axis]):
                valid = False
            for column in range(3):
                if not wp.isfinite(fp[axis, column]):
                    valid = False
        if valid:
            determinant = float(wp.determinant(wp.mat33d(fp)))
            squared = vp[0]*vp[0] + vp[1]*vp[1] + vp[2]*vp[2]
            minimum_det = wp.min(minimum_det, determinant)
            maximum_speed = wp.max(maximum_speed, wp.sqrt(squared))
            maximum_plastic = wp.max(maximum_plastic, plastic[p])
        else:
            invalid = 1.0
    row = e*groups+group
    partials[row, 0] = invalid
    partials[row, 1] = minimum_det
    partials[row, 2] = maximum_speed
    partials[row, 3] = maximum_plastic


@wp.kernel
def _env_combine(partials: wp.array2d(dtype=float), groups: int, summary: wp.array2d(dtype=float)):
    e = wp.tid()
    invalid = float(0.0)
    minimum_det = float(wp.inf)
    maximum_speed = float(0.0)
    maximum_plastic = -float(wp.inf)
    for group in range(groups):
        row = e*groups+group
        invalid = wp.max(invalid, partials[row, 0])
        minimum_det = wp.min(minimum_det, partials[row, 1])
        maximum_speed = wp.max(maximum_speed, partials[row, 2])
        maximum_plastic = wp.max(maximum_plastic, partials[row, 3])
    summary[e, 0] = invalid
    summary[e, 1] = minimum_det
    summary[e, 2] = maximum_speed
    summary[e, 3] = maximum_plastic


class SegmentedDiagnostics:
    """Preallocated per-env reduction buffers; ``launch()`` allocates nothing and is graph-safe."""

    def __init__(self, solver):
        self.solver = solver
        counts = solver.packed.particle_counts
        if int(counts.min()) < 1:
            raise ValueError("Material diagnostics require at least one particle per env")
        self.groups = int((int(counts.max())+63)//64)
        with wp.ScopedDevice(solver.device):
            self.partials = wp.zeros((solver.num_envs*self.groups, 4), dtype=float)
            self.summary = wp.zeros((solver.num_envs, 4), dtype=float)

    def launch(self):
        s = self.solver
        wp.launch(_env_state_partials, (s.num_envs, self.groups),
                  [s.x, s.v, s.F, s.plastic, s.particle_offset, self.groups, self.partials], device=s.device)
        wp.launch(_env_combine, s.num_envs, [self.partials, self.groups, self.summary], device=s.device)

    def read(self):
        """Host dicts in the frozen ``MaterialDiagnosticsReduction.read`` keys, one per env (no raise)."""
        self.launch()
        values = self.summary.numpy()
        return [{"nonfinite": bool(row[INVALID] != 0.), "minimum_detF": float(row[MINIMUM_DET]),
                 "maximum_material_speed_m_s": float(row[MAXIMUM_SPEED]),
                 "maximum_plastic_strain": float(row[MAXIMUM_PLASTIC])} for row in values]
