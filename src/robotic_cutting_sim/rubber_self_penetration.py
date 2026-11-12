"""S1 self-penetration detector ``surface_grid_v1`` (ADR 0023 §22.3, §23.2 decision 2; plan P2_REMAINING_PLAN §2.2).

A check, not contact physics: it adds no force and changes no solver state (self-contact stays ``NOT_IMPLEMENTED_V1``).
It flags two surface particles of the same env that are closer than the particle spacing s in the current
configuration (|x_p - x_q| < d, d = s: the particle volumes overlap) although they are more than 3 dx apart in the
reference configuration (|X_p - X_q| > 3 dx, the frozen reference exclusion). Through-thickness pairs and legal bends
cannot satisfy both (plan §2.2).

Grid: the env-grouped sort grid of the Phase 1 ``FallbackGrid`` (``batched/particle_contacts.py``), re-implemented here
for surface particles (that Phase 1 file is not touched): key ``env * C + hash(cell)`` with Warp's hashgrid cell hash
(origin offset, clamp, per-axis modulo ``dims``), cell width d, a stable radix sort and per-cell ranges, then a 27-cell
query. Kernels, memsets and the radix sort only: graph-capturable, no ``wp.HashGrid`` (so neither the Warp 1.17
one-descriptor hazard nor the Phase 1 device-HashGrid guard applies). Envs cannot mix: the env index is part of the key.

Cadence: after substep k (1-based within the control step) when k % check_substeps == 0, and after the last substep of
every control step; inside the CUDA graph chunks (the owner aligns the chunk to a multiple of check_substeps), eagerly on
CPU. The same check positions on every device and in the single-env and batched owners.

Outputs per env (device arrays, int32): ``hits`` (pairs found, summed over checks since the last reset), ``first_check``
(ordinal of the first check with a hit, ``NO_HIT`` if none), ``checks`` (checks run since the reset), ``blind_steps``
(control steps with 2 v_max dt_check >= t - d, information only: a crossing faster than that could pass between two
checks). After the control step's diagnostics, ``post`` ORs the sticky bit ``RUBBER_SELF_CONTACT_UNRESOLVED`` (8) into
the owner's invalid-reason array of every env with hits > 0.

The single-env owner (``LiveRubberMaterial``) and the batched owner (``BatchedRubberMaterial``) build this class with
N = 1 or N envs and launch the same kernels. Default ``off_v1``: the owners build no detector (no array, no launch).
Material status: ``SIMULATOR_ONLY_UNCALIBRATED``.
"""
from typing import Any

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

CHECK_OFF = "off_v1"
CHECK_SURFACE_GRID = "surface_grid_v1"
SELF_PENETRATION_CHECKS = (CHECK_OFF, CHECK_SURFACE_GRID)
DEFAULT_SELF_PENETRATION_CHECK = CHECK_OFF
DEFAULT_CHECK_SUBSTEPS = 256           # ~1 ms at dt 3.8 us (plan §2.2)
INVALID_RUBBER_SELF_CONTACT_UNRESOLVED = 8
REASON_NAME = "RUBBER_SELF_CONTACT_UNRESOLVED"
REFERENCE_EXCLUSION_DX = 3.0
NO_HIT = 2**31-1
DETECTOR_VERSION = "rubber_self_penetration_v1"
# vars()-complete classification of the detector's device arrays (tests)
RESET_ARRAYS = ("hits", "first_check", "checks", "blind_steps")
CONSTANT_ARRAYS = ("surf", "surf_env", "clearance")
SCRATCH_ARRAYS = ("keys", "ids", "starts", "ends")


@wp.func
def _cell_index(x: int, y: int, z: int, dims: wp.vec3i):
    """Warp hashgrid.h ``hash_grid_index``: origin offset, clamp, per-axis modulo, z-major (as Phase 1)."""
    origin = 1 << 20
    cx = wp.max(x+origin, 0) % dims[0]
    cy = wp.max(y+origin, 0) % dims[1]
    cz = wp.max(z+origin, 0) % dims[2]
    return cz*(dims[0]*dims[1])+cy*dims[0]+cx


@wp.kernel
def _sp_keys(x: wp.array(dtype=Any), surf: wp.array(dtype=int), surf_env: wp.array(dtype=int), inv: wp.float64,
             dims: wp.vec3i, cells: int, keys: wp.array(dtype=int), values: wp.array(dtype=int)):
    i = wp.tid()
    xp = x[surf[i]]
    cx = int(wp.floor(wp.float64(xp[0])*inv))
    cy = int(wp.floor(wp.float64(xp[1])*inv))
    cz = int(wp.floor(wp.float64(xp[2])*inv))
    keys[i] = surf_env[i]*cells+_cell_index(cx, cy, cz, dims)
    values[i] = i


@wp.kernel
def _sp_ranges(keys: wp.array(dtype=int), n: int, starts: wp.array(dtype=int), ends: wp.array(dtype=int)):
    i = wp.tid()
    k = keys[i]
    if i == 0:
        starts[k] = i
    elif keys[i-1] != k:
        starts[k] = i
    if i == n-1:
        ends[k] = i+1
    elif keys[i+1] != k:
        ends[k] = i+1


@wp.kernel
def _sp_query(x: wp.array(dtype=Any), reference: wp.array(dtype=wp.vec3), surf: wp.array(dtype=int),
              surf_env: wp.array(dtype=int), inv: wp.float64, dims: wp.vec3i, cells: int,
              starts: wp.array(dtype=int), ends: wp.array(dtype=int), ids: wp.array(dtype=int), d2: wp.float64,
              exclusion2: float, checks: wp.array(dtype=int), hits: wp.array(dtype=int),
              first_check: wp.array(dtype=int)):
    """Surface particle i against the surface particles of its env in the 27 cells around it; each pair once (j > i).
    dims >= 3 per axis, so the 27 hashed cells are distinct; a hash collision only adds candidates that fail the
    distance test (they are at least one cell width d apart)."""
    i = wp.tid()
    p = surf[i]
    e = surf_env[i]
    xp = x[p]
    px = wp.float64(xp[0])
    py = wp.float64(xp[1])
    pz = wp.float64(xp[2])
    cx = int(wp.floor(px*inv))
    cy = int(wp.floor(py*inv))
    cz = int(wp.floor(pz*inv))
    Xp = reference[p]
    found = int(0)
    for oz in range(-1, 2):
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                k = e*cells+_cell_index(cx+ox, cy+oy, cz+oz, dims)
                for jj in range(starts[k], ends[k]):
                    j = ids[jj]
                    if j > i:
                        q = surf[j]
                        if wp.length_sq(Xp-reference[q]) > exclusion2:
                            xq = x[q]
                            ax = px-wp.float64(xq[0])
                            ay = py-wp.float64(xq[1])
                            az = pz-wp.float64(xq[2])
                            if ax*ax+ay*ay+az*az < d2:
                                found += 1
    if found > 0:
        wp.atomic_add(hits, e, found)
        wp.atomic_min(first_check, e, checks[e])


@wp.kernel
def _sp_tick(checks: wp.array(dtype=int)):
    e = wp.tid()
    checks[e] = checks[e]+1


@wp.kernel
def _sp_post(summary: wp.array2d(dtype=wp.float64), speed_column: int, span: wp.float64,
             clearance: wp.array(dtype=wp.float64), hits: wp.array(dtype=int), blind_steps: wp.array(dtype=int),
             invalid_reason: wp.array(dtype=int), bit: int):
    """Once per control step, after the diagnostics: blind-interval count (2 v_max dt_check >= t - d) and the sticky
    bit for every env with hits."""
    e = wp.tid()
    if summary[e, speed_column]*span >= clearance[e]:
        blind_steps[e] = blind_steps[e]+1
    if hits[e] > 0:
        invalid_reason[e] = invalid_reason[e] | bit


@wp.kernel
def _sp_reset(mask: wp.array(dtype=int), hits: wp.array(dtype=int), first_check: wp.array(dtype=int),
              checks: wp.array(dtype=int), blind_steps: wp.array(dtype=int), no_hit: int):
    e = wp.tid()
    if mask[e] != 0:
        hits[e] = 0
        first_check[e] = no_hit
        checks[e] = 0
        blind_steps[e] = 0


for _vec in (wp.vec3, wp.vec3d):
    wp.overload(_sp_keys, {"x": wp.array(dtype=_vec)})
    wp.overload(_sp_query, {"x": wp.array(dtype=_vec)})


# ----------------------------------------------------------------------------- host helpers

def check_option(check, check_substeps):
    if check not in SELF_PENETRATION_CHECKS:
        raise ValueError(f"self_penetration_check={check!r} not in {SELF_PENETRATION_CHECKS}")
    if int(check_substeps) != check_substeps or int(check_substeps) < 1:
        raise ValueError(f"self_penetration_check_substeps={check_substeps!r} must be a positive integer")
    return check, int(check_substeps)


def check_positions(substeps, check_substeps):
    """1-based substeps of a control step after which the check runs: multiples of check_substeps and the last."""
    pos = list(range(check_substeps, substeps+1, check_substeps))
    if not pos or pos[-1] != substeps:
        pos.append(substeps)
    return pos


def aligned_chunk(chunk, substeps, check_substeps):
    """A graph chunk whose replays all put the checks at the same local positions: a multiple of check_substeps (or
    the whole control step when check_substeps exceeds it)."""
    chunk = max(1, min(int(chunk), int(substeps)))
    if chunk == substeps or check_substeps > substeps:
        return int(substeps)
    return max(check_substeps, chunk-chunk % check_substeps)


class SelfPenetrationDetector:
    """Detector state and launches for N envs (see the module docstring).

    ``scenes``: one ``RubberStripScene`` per env (surface flags, reference positions, thickness); ``positions``: the
    owner's current positions (``x`` vec3, or the float64 shadow ``x64`` vec3d when F10 is on); ``reference``: the
    owner's reference positions (vec3); ``particle_offsets``: first particle of each env in ``positions``."""

    def __init__(self, scenes, positions, reference, particle_offsets, *, device, dt, substeps,
                 check_substeps=DEFAULT_CHECK_SUBSTEPS):
        scenes = list(scenes)
        self.num_envs = N = len(scenes)
        spacing = {float(sc.spacing) for sc in scenes}
        dx = {float(sc.dx) for sc in scenes}
        if len(spacing) != 1 or len(dx) != 1:
            raise ValueError("self-penetration detector: spacing and dx must be uniform over the envs")
        self.spacing, self.dx = spacing.pop(), dx.pop()
        self.distance = self.spacing                         # d = s
        self.reference_exclusion = REFERENCE_EXCLUSION_DX*self.dx
        self.cell_width = self.distance                      # 27-cell query covers |x_p - x_q| < d
        self.check_substeps = int(check_substeps)
        self.substeps = int(substeps)
        self.positions_per_step = check_positions(self.substeps, self.check_substeps)
        self.dt = float(dt)
        self.dt_check = self.check_substeps*self.dt
        offsets = np.asarray(particle_offsets, np.int64)
        surf = [offsets[e]+np.flatnonzero(sc.surface) for e, sc in enumerate(scenes)]
        surf_env = [np.full(len(s), e, np.int32) for e, s in enumerate(surf)]
        self.surface_counts = [int(len(s)) for s in surf]
        self.surface_total = S = int(sum(self.surface_counts))
        if S == 0:
            raise ValueError("self-penetration detector: no surface particles")
        extent = np.max([np.ptp(sc.X.astype(np.float64), axis=0) for sc in scenes], axis=0)
        self.dims = tuple(int(max(3, np.ceil(v/self.cell_width)+3)) for v in extent)
        self.cells = int(np.prod(self.dims))
        if N*self.cells >= 2**31:
            raise ValueError(f"self-penetration detector: N*C = {N*self.cells} overflows int32 keys")
        self.end_bit = max(int(N*self.cells-1).bit_length(), 1)
        # Warp's own cell transform floor(x * (1 / w)) with the inverse taken in float64
        self.inv = 1.0/self.cell_width
        thickness = np.array([float(sc.spec.thickness) for sc in scenes], np.float64)
        self.thickness = thickness
        self.positions, self.reference = positions, reference
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            self.surf = wp.array(np.concatenate(surf).astype(np.int32), dtype=int)
            self.surf_env = wp.array(np.concatenate(surf_env), dtype=int)
            self.clearance = wp.array(thickness-self.distance, dtype=wp.float64)
            self.keys = wp.zeros(2*S, dtype=int)
            self.ids = wp.zeros(2*S, dtype=int)
            self.starts = wp.zeros(N*self.cells, dtype=int)
            self.ends = wp.zeros(N*self.cells, dtype=int)
            self.hits = wp.zeros(N, dtype=int)
            self.first_check = wp.full(N, NO_HIT, dtype=int)
            self.checks = wp.zeros(N, dtype=int)
            self.blind_steps = wp.zeros(N, dtype=int)
        wp.load_module(module=__name__, device=self.device)

    # ------------------------------------------------------------------ launches

    def check(self):
        """One check of every env (kernels + memsets + stable radix sort; graph-capturable)."""
        S, N = self.surface_total, self.num_envs
        inv = wp.float64(self.inv)
        dims = wp.vec3i(*self.dims)
        with wp.ScopedDevice(self.device):
            wp.launch(_sp_keys, S, [self.positions, self.surf, self.surf_env, inv, dims, self.cells, self.keys,
                                    self.ids], device=self.device)
            wp.utils.radix_sort_pairs(self.keys, self.ids, S, 0, self.end_bit)
            self.starts.zero_()
            self.ends.zero_()
            wp.launch(_sp_ranges, S, [self.keys, S, self.starts, self.ends], device=self.device)
            wp.launch(_sp_query, S, [self.positions, self.reference, self.surf, self.surf_env, inv, dims, self.cells,
                                     self.starts, self.ends, self.ids, wp.float64(self.distance**2),
                                     float(self.reference_exclusion**2), self.checks, self.hits, self.first_check],
                      device=self.device)
            wp.launch(_sp_tick, N, [self.checks], device=self.device)

    def run_substeps(self, step, count, start=0):
        """``count`` solver substeps starting after substep ``start`` of the control step, with the checks at their
        positions (eager, or inside a graph capture)."""
        for j in range(1, count+1):
            step()
            k = start+j
            if k % self.check_substeps == 0 or k == self.substeps:
                self.check()

    def post(self, summary, invalid_reason, speed_column):
        """Per control step, after the owner's diagnostics (``summary[N, NDIAG]``)."""
        with wp.ScopedDevice(self.device):
            wp.launch(_sp_post, self.num_envs, [summary, int(speed_column), wp.float64(2.0*self.dt_check),
                                                self.clearance, self.hits, self.blind_steps, invalid_reason,
                                                INVALID_RUBBER_SELF_CONTACT_UNRESOLVED], device=self.device)

    def reset_masked(self, mask):
        with wp.ScopedDevice(self.device):
            wp.launch(_sp_reset, self.num_envs, [mask, self.hits, self.first_check, self.checks, self.blind_steps,
                                                 NO_HIT], device=self.device)

    def reset_all(self):
        self.hits.zero_()
        self.first_check.fill_(NO_HIT)
        self.checks.zero_()
        self.blind_steps.zero_()

    # ------------------------------------------------------------------ queries

    def arrays(self):
        return {k: getattr(self, k) for k in RESET_ARRAYS+CONSTANT_ARRAYS+SCRATCH_ARRAYS}

    def nbytes(self):
        return int(sum(a.capacity for a in self.arrays().values()))

    def locate(self, ordinal):
        """(control step, substep) of a check ordinal (control steps counted from 1 since the reset)."""
        per = len(self.positions_per_step)
        return int(ordinal)//per+1, int(self.positions_per_step[int(ordinal) % per])

    def read(self):
        """Host summary per env: hits, first hit (check ordinal, control step, substep), checks, blind steps."""
        hits, first, checks, blind = (a.numpy() for a in (self.hits, self.first_check, self.checks,
                                                          self.blind_steps))
        out = []
        for e in range(self.num_envs):
            hit = int(first[e]) != NO_HIT
            step, sub = self.locate(first[e]) if hit else (None, None)
            out.append({"hits": int(hits[e]), "first_hit_check": int(first[e]) if hit else None,
                        "first_hit_control_step": step, "first_hit_substep": sub, "checks": int(checks[e]),
                        "blind_steps": int(blind[e])})
        return out

    def receipt(self):
        return {"self_penetration_check": CHECK_SURFACE_GRID, "version": DETECTOR_VERSION,
                "check_substeps": self.check_substeps, "checks_per_control_step": len(self.positions_per_step),
                "check_interval_s": self.dt_check, "distance_threshold_m": self.distance,
                "reference_exclusion_m": self.reference_exclusion, "cell_width_m": self.cell_width,
                "grid_dims": list(self.dims), "surface_particles": self.surface_counts,
                "grid": "env-grouped sort grid (keys env*C + hash(cell), stable radix sort, ranges, 27-cell query); "
                        "no wp.HashGrid",
                "positions": "x64 (f64 shadow)" if self.positions.dtype == wp.vec3d else "x",
                "invalid_bit": {INVALID_RUBBER_SELF_CONTACT_UNRESOLVED: REASON_NAME},
                "physics": "detector only (no force); self-contact NOT_IMPLEMENTED_V1", "memory_bytes": self.nbytes()}
