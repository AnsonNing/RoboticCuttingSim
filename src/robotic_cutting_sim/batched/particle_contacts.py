"""Opt-in particle-contact execution modes of the batched v7_2 solver (P1-3a, analysis C1 + C2 + guard).

``frozen_v1`` (default, the rollback): ``tlmpm_v7_2_batch`` builds the grouped ``HashGrid(64, 64, 32)`` and
launches ``_particle_contacts_b`` exactly as before; nothing in this module runs.

``alias_free_v1`` changes how the frozen contact step executes, not what it computes. Independent sub-flags
(``docs/reports/rl_ready/P1_3_BOTTLENECK_ANALYSIS.md`` §2, §4):

* ``skip_zero_impulse_atomics`` (C1): the two impulse atomics of a pair are skipped when the pair impulse is
  exactly zero component-wise (``!= 0``, NaN still propagates). ``impulse`` starts at +0 every substep and only
  receives sums, so a +-0 add is the identity and changes no bit (analysis fact (i)); unconditional.
* ``alias_free_hash_dims`` (C2): the grouped hash grid gets dims derived from the packed reference bounding box
  of every env plus a motion margin (never smaller than the frozen dims), so no cell aliasing occurs within an
  env while the particles stay inside the margin. The query still visits the same virtual cells in the same
  order with ascending ids per cell; only aliased candidates disappear. Those candidates are at least
  ``(D_min - 2) * h`` apart (D_min of the frozen reference dims), so they contribute exactly zero while every
  particle speed of the env is below the certified bound (analysis certificate (ii)).
* ``speed_guard`` (needs C2): a per-env device check before the contact kernel, every substep, of
  ``|v_p|^2 <= bound^2`` and finite ``x_p`` for all particles of the env. ``"fallback"``: an env that fails
  is skipped by the alias-free kernel and processed in the same substep by the frozen kernel body on a grouped
  grid with the frozen reference dims, inside a conditional graph node (``wp.capture_if``; eager on CPU), so
  its result is the frozen result; the substep is counted in ``contact_guard_violations[e]``. ``"flag"``: no
  fallback, the substep is only counted (the parity claim does not cover that env). ``"off"``: unchecked
  (A/B timing only; the receipt says ``certificate: unchecked``). A non-zero count makes the material set the
  sticky ``INVALID_CONTACT_SPEED_GUARD`` bit.

Nothing here edits a frozen module; the kernels below are the frozen ``_particle_contacts_b`` body with the
listed changes only.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import warp as wp

from .. import tlmpm_v5 as v5
from .constitutive import P_FRICTION, P_PARTICLE_STIFFNESS
from .errors import UnsupportedBatchConfiguration
from .layout import STATS_WIDTH

DEFAULT_MODE = "frozen_v1"
ALIAS_FREE_MODE = "alias_free_v1"
MODES = (DEFAULT_MODE, ALIAS_FREE_MODE)
SPEED_GUARDS = ("fallback", "flag", "off")
# The frozen grid dims (tlmpm_v5.py:798, tlmpm_v7_2_batch.py:613): the reference of every parity claim.
FROZEN_HASH_DIMS = (64, 64, 32)
DEFAULT_MARGIN_M = 0.02
# Relative safety on the alias distance of certificate (ii) (covers float32 floor/rounding by a wide margin).
GUARD_DISTANCE_SAFETY = 0.99
ST = wp.constant(STATS_WIDTH)


@dataclass(frozen=True)
class ParticleContactConfig:
    """Resolved particle-contact execution (see the module docstring)."""
    mode: str = DEFAULT_MODE
    skip_zero_impulse_atomics: bool = False
    alias_free_hash_dims: bool = False
    speed_guard: str = "off"
    margin_m: float = DEFAULT_MARGIN_M
    reference_hash_dims: tuple = FROZEN_HASH_DIMS
    speed_bound_m_s: float = None

    @property
    def is_default(self):
        return self.mode == DEFAULT_MODE


def resolve(mode=DEFAULT_MODE, options=None):
    """``mode`` plus optional sub-flag overrides -> ``ParticleContactConfig``.

    ``alias_free_v1`` defaults: C1 on, C2 on, guard ``"fallback"``, margin 20 mm. ``frozen_v1`` takes no
    options. ``reference_hash_dims`` (default the frozen 64x64x32) is the grid the parity claim and the fallback
    refer to; ``speed_bound_m_s`` may lower (never raise) the certified guard bound."""
    options = dict(options or {})
    if mode not in MODES:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE", f"{mode!r}; supported {MODES}")
    if mode == DEFAULT_MODE:
        if options:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE",
                                                f"{DEFAULT_MODE} takes no options, got {sorted(options)}")
        return ParticleContactConfig()
    known = {"skip_zero_impulse_atomics", "alias_free_hash_dims", "speed_guard", "margin_m", "reference_hash_dims",
             "speed_bound_m_s"}
    unknown = set(options)-known
    if unknown:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE", f"unknown options {sorted(unknown)}")
    c1 = bool(options.get("skip_zero_impulse_atomics", True))
    c2 = bool(options.get("alias_free_hash_dims", True))
    guard = options.get("speed_guard", "fallback" if c2 else "off")
    if guard not in SPEED_GUARDS:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE", f"speed_guard={guard!r}")
    if guard != "off" and not c2:
        raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE",
                                            "speed_guard certifies alias_free_hash_dims; C1 alone needs none")
    if not (c1 or c2):
        raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE",
                                            f"{ALIAS_FREE_MODE} with every sub-flag off is {DEFAULT_MODE}")
    margin = float(options.get("margin_m", DEFAULT_MARGIN_M))
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError("margin_m must be finite and >= 0")
    dims = tuple(int(d) for d in options.get("reference_hash_dims", FROZEN_HASH_DIMS))
    if len(dims) != 3 or min(dims) < 3:
        raise ValueError("reference_hash_dims must be three ints >= 3")
    bound = options.get("speed_bound_m_s")
    if bound is not None:
        bound = float(bound)
        if not (math.isfinite(bound) and bound >= 0.0):
            raise ValueError("speed_bound_m_s must be finite and >= 0")
    return ParticleContactConfig(mode=mode, skip_zero_impulse_atomics=c1, alias_free_hash_dims=c2,
                                 speed_guard=guard, margin_m=margin, reference_hash_dims=dims,
                                 speed_bound_m_s=bound)


# ----------------------------------------------------------------------------- dims and certificate

def alias_free_dims(packed, cell_width, margin_m, reference_dims=FROZEN_HASH_DIMS):
    """Uniform grid dims with no cell aliasing inside any env's reference bounding box +- ``margin_m``.

    The query visits virtual cells floor((x_p -+ R)/h) per axis: normally c_p-1..c_p+1, at most c_p-2..c_p+2
    when float32 rounding of x*inv and R*inv ~ 1 meet. A candidate q aliases iff its virtual cell differs by a
    non-zero multiple of D from a visited cell. With every cell of an env inside [c_lo, c_hi] per axis,
    |c_q - c| <= c_hi-c_lo+2, so D >= c_hi-c_lo+3 = span+2 excludes it; one more cell of slack is added
    (D >= span+3). Never smaller than ``reference_dims``. Returns (dims, per-env cell spans)."""
    reference = np.asarray(packed.host["reference"], np.float64)
    offsets = packed.particle_offset
    dims = np.array(reference_dims, np.int64)
    spans = []
    for e in range(packed.num_envs):
        part = reference[int(offsets[e]):int(offsets[e+1])]
        lo = np.floor((part.min(0)-margin_m)/cell_width).astype(np.int64)
        hi = np.floor((part.max(0)+margin_m)/cell_width).astype(np.int64)
        span = hi-lo+1
        spans.append(span.tolist())
        dims = np.maximum(dims, span+3)
    return tuple(int(d) for d in dims), spans


def certified_speed_bound(dims, reference_dims, cell_width, radius, dt):
    """Certificate (ii): an aliased candidate (in either grid) has a virtual cell >= D-2 away from p's cell
    (the visited cells are within c_p +- 2, see ``alias_free_dims``), so it is >= (D_min-3)*h away (the analysis
    used D_min-2 for the nominal c_p +- 1 box; this is the rounding-safe form). A pair impulse is non-zero only
    if d < 2r + dt*|v_rel| <= 2r + 2*dt*V. Returns the largest V (m/s) that keeps every aliased candidate at
    exactly zero, with ``GUARD_DISTANCE_SAFETY`` on the distance."""
    d_min = min(min(dims), min(reference_dims))
    alias_distance = (d_min-3)*cell_width
    return max(0.5*(alias_distance*GUARD_DISTANCE_SAFETY-2.0*radius)/dt, 0.0), alias_distance


# ----------------------------------------------------------------------------- kernels

@wp.kernel
def _speed_guard_b(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), env_of_particle: wp.array(dtype=int),
                   bound_sq: float, guard_flag: wp.array(dtype=int), guard_any: wp.array(dtype=int)):
    """Flag env e when a particle's speed exceeds the certified bound or its state is not finite (plain
    stores of 1: no atomics, no contention; the flags are zero between substeps)."""
    p = wp.tid()
    xp = x[p]
    speed_sq = wp.length_sq(v[p])
    if not (speed_sq <= bound_sq) or not (wp.isfinite(xp[0]) and wp.isfinite(xp[1]) and wp.isfinite(xp[2])):
        guard_flag[env_of_particle[p]] = 1
        guard_any[0] = 1


@wp.kernel
def _guard_consume_b(guard_flag: wp.array(dtype=int), guard_any: wp.array(dtype=int),
                     violations: wp.array(dtype=int)):
    """Count the flagged substep per env and clear the substep flags."""
    e = wp.tid()
    if guard_flag[e] != 0:
        violations[e] = violations[e]+1
        guard_flag[e] = 0
    if e == 0:
        guard_any[0] = 0


@wp.kernel
def _particle_contacts_skip_zero_b(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
                                   reference: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                                   mass: wp.array(dtype=float), code: wp.array(dtype=int),
                                   impulse: wp.array(dtype=wp.vec3), stats_sum: wp.array(dtype=float),
                                   stats_max: wp.array(dtype=float), radius: float, reference_exclusion: float,
                                   dt: float, params: wp.array2d(dtype=float), env_of_particle: wp.array(dtype=int),
                                   guard_flag: wp.array(dtype=int), skip_flagged: int):
    """``_particle_contacts_b`` with C1 (exactly-zero pair impulses skip their two atomics); with
    ``skip_flagged`` the envs flagged by the speed guard are left to the fallback kernel."""
    p = wp.tid()
    e = env_of_particle[p]
    if skip_flagged != 0:
        if guard_flag[e] != 0:
            return
    stiffness = params[e, P_PARTICLE_STIFFNESS]
    friction = params[e, P_FRICTION]
    query = wp.hash_grid_query(grid, x[p], 2.4*radius, e)
    for q in query:
        if q > p:
            if v5._conflict_packed(code[p], code[q]) or wp.length_sq(reference[p]-reference[q]) > reference_exclusion**2.0:
                delta = x[p]-x[q]
                distance = wp.length(delta)
                if distance > 1.0e-10:
                    normal = delta/distance
                    effective = mass[p]*mass[q]/(mass[p]+mass[q])
                    ji = v5._contact_impulse(distance-2.0*radius, normal, v[p]-v[q], effective,
                                             dt, stiffness, friction)
                    if ji[0] != 0.0 or ji[1] != 0.0 or ji[2] != 0.0:
                        wp.atomic_add(impulse, p, ji)
                        wp.atomic_sub(impulse, q, ji)
                    if wp.length_sq(ji) > 0.0:
                        wp.atomic_add(stats_sum, e*ST+4, 1.0)
                        wp.atomic_add(stats_sum, e*ST+5, wp.length(ji))
                        wp.atomic_max(stats_max, e*ST+6, wp.max(2.0*radius-distance, 0.0))


@wp.kernel
def _particle_contacts_env_filter_b(grid: wp.uint64, x: wp.array(dtype=wp.vec3),
                                    reference: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                                    mass: wp.array(dtype=float), code: wp.array(dtype=int),
                                    impulse: wp.array(dtype=wp.vec3), stats_sum: wp.array(dtype=float),
                                    stats_max: wp.array(dtype=float), radius: float, reference_exclusion: float,
                                    dt: float, params: wp.array2d(dtype=float),
                                    env_of_particle: wp.array(dtype=int), guard_flag: wp.array(dtype=int),
                                    flagged: int):
    """The frozen ``_particle_contacts_b`` body (no C1) for the envs whose guard flag equals ``flagged != 0``:
    ``flagged=1`` is the fallback (flagged envs on the reference-dims grid), ``flagged=0`` the alias-free
    launch without C1 (unflagged envs)."""
    p = wp.tid()
    e = env_of_particle[p]
    if (guard_flag[e] != 0) != (flagged != 0):
        return
    stiffness = params[e, P_PARTICLE_STIFFNESS]
    friction = params[e, P_FRICTION]
    query = wp.hash_grid_query(grid, x[p], 2.4*radius, e)
    for q in query:
        if q > p:
            if v5._conflict_packed(code[p], code[q]) or wp.length_sq(reference[p]-reference[q]) > reference_exclusion**2.0:
                delta = x[p]-x[q]
                distance = wp.length(delta)
                if distance > 1.0e-10:
                    normal = delta/distance
                    effective = mass[p]*mass[q]/(mass[p]+mass[q])
                    ji = v5._contact_impulse(distance-2.0*radius, normal, v[p]-v[q], effective,
                                             dt, stiffness, friction)
                    wp.atomic_add(impulse, p, ji)
                    wp.atomic_sub(impulse, q, ji)
                    if wp.length_sq(ji) > 0.0:
                        wp.atomic_add(stats_sum, e*ST+4, 1.0)
                        wp.atomic_add(stats_sum, e*ST+5, wp.length(ji))
                        wp.atomic_max(stats_max, e*ST+6, wp.max(2.0*radius-distance, 0.0))


@wp.kernel
def _guard_invalid_b(violations: wp.array(dtype=int), invalid_reason: wp.array(dtype=int), flag: int):
    """Material, once per control step before ``_validity_b``: a counted guard substep sets the sticky bit."""
    e = wp.tid()
    if violations[e] > 0:
        invalid_reason[e] = invalid_reason[e] | flag


@wp.kernel
def _reset_guard_b(mask: wp.array(dtype=int), guard_flag: wp.array(dtype=int), violations: wp.array(dtype=int)):
    """Masked partial reset of the per-env guard state (``contact_guard_any`` is zero between substeps)."""
    e = wp.tid()
    if mask[e] != 0:
        guard_flag[e] = 0
        violations[e] = 0


# ----------------------------------------------------------------------------- execution

class AliasFreeContacts:
    """The ``alias_free_v1`` contact stage of one ``BatchedTLMPMv7_2`` (constructed by the solver).

    Owns the grids; the guard arrays are attached to the solver (``contact_guard_flag[N]``,
    ``contact_guard_any[1]``, ``contact_guard_violations[N]``) so memory accounting and reset coverage see them."""

    def __init__(self, solver, config):
        from .tlmpm_v7_2_batch import _particle_contacts_b
        self.config = config
        self._frozen_kernel = _particle_contacts_b
        s = solver
        self.cell_width = 2.4*s.radius
        if config.alias_free_hash_dims:
            self.dims, self.env_cell_spans = alias_free_dims(s.packed, self.cell_width, config.margin_m,
                                                            config.reference_hash_dims)
        else:
            self.dims, self.env_cell_spans = tuple(config.reference_hash_dims), None
        certified, self.alias_distance_m = certified_speed_bound(self.dims, config.reference_hash_dims,
                                                                 self.cell_width, s.radius, s.dt)
        self.certified_speed_bound_m_s = certified
        if config.speed_bound_m_s is not None and config.speed_bound_m_s > certified:
            raise UnsupportedBatchConfiguration("UNSUPPORTED_PARTICLE_CONTACT_MODE",
                                                f"speed_bound_m_s {config.speed_bound_m_s} > certified {certified}")
        self.speed_bound_m_s = certified if config.speed_bound_m_s is None else config.speed_bound_m_s
        self.conditional_graph_supported = bool(wp.is_conditional_graph_supported())
        N, n = s.num_envs, s.total_particles
        with wp.ScopedDevice(s.device):
            if self.dims == FROZEN_HASH_DIMS:
                self.grid = s.hash     # C1 only: the solver's frozen-dims grid (already reserved and built)
            else:
                self.grid = wp.HashGrid(*self.dims)
                self.grid.reserve(n, with_groups=True)
                self.grid.build(s.x, self.cell_width, groups=s.env_of_particle)
            self.fallback_grid = None
            if config.speed_guard != "off":
                s.contact_guard_flag = wp.zeros(N, dtype=int)
                s.contact_guard_any = wp.zeros(1, dtype=int)
                s.contact_guard_violations = wp.zeros(N, dtype=int)
            if config.speed_guard == "fallback":
                self.fallback_grid = wp.HashGrid(*config.reference_hash_dims)
                self.fallback_grid.reserve(n, with_groups=True)
                self.fallback_grid.build(s.x, self.cell_width, groups=s.env_of_particle)
        wp.load_module(module=__name__, device=s.device)
        # Bound arrays used only for argument lists when the guard is off (never read then).
        self._no_flags = wp.zeros(N, dtype=int, device=s.device) if config.speed_guard == "off" else None

    # ------------------------------------------------------------------ one substep (inside the solver step)

    def launch(self, s, r, dt):
        c = self.config
        n = s.total_particles
        h = self.cell_width
        guarded = c.speed_guard != "off"
        flags = s.contact_guard_flag if guarded else self._no_flags
        if guarded:
            wp.launch(_speed_guard_b, n, [s.x, s.v, s.env_of_particle, self.speed_bound_m_s**2,
                                          s.contact_guard_flag, s.contact_guard_any])
        self.grid.build(s.x, h, groups=s.env_of_particle)
        skip_flagged = int(c.speed_guard == "fallback")
        common = [self.grid.id, s.x, s.reference, s.v, s.mass, s.code, s.impulse, s.stats_sum, s.stats_max, r,
                  3*s.dx, dt, s.params, s.env_of_particle]
        if c.skip_zero_impulse_atomics:
            wp.launch(_particle_contacts_skip_zero_b, n, common+[flags, skip_flagged])
        elif skip_flagged:
            wp.launch(_particle_contacts_env_filter_b, n, common+[flags, 0])
        else:
            wp.launch(self._frozen_kernel, n, common)
        if c.speed_guard == "fallback":
            wp.capture_if(s.contact_guard_any, on_true=self._fallback, s=s, r=r, dt=dt)
        elif c.speed_guard == "flag":
            wp.launch(_guard_consume_b, s.num_envs, [s.contact_guard_flag, s.contact_guard_any,
                                                     s.contact_guard_violations])

    def _fallback(self, s, r, dt):
        """Flagged envs: the frozen body on the reference-dims grid (the frozen candidate sequence)."""
        self.fallback_grid.build(s.x, self.cell_width, groups=s.env_of_particle)
        wp.launch(_particle_contacts_env_filter_b, s.total_particles, [self.fallback_grid.id, s.x, s.reference, s.v,
            s.mass, s.code, s.impulse, s.stats_sum, s.stats_max, r, 3*s.dx, dt, s.params, s.env_of_particle,
            s.contact_guard_flag, 1])
        wp.launch(_guard_consume_b, s.num_envs, [s.contact_guard_flag, s.contact_guard_any,
                                                 s.contact_guard_violations])

    # ------------------------------------------------------------------ receipts

    def receipt(self, solver=None):
        c = self.config
        out = {"particle_contact_mode": c.mode, **{k: v for k, v in asdict(c).items() if k != "mode"},
               "reference_hash_dims": list(c.reference_hash_dims), "hash_dims": list(self.dims),
               "hash_cells": int(np.prod(self.dims)), "cell_width_m": self.cell_width,
               "env_cell_spans": self.env_cell_spans,
               "alias_free_while_inside_margin": bool(c.alias_free_hash_dims),
               "alias_distance_m": self.alias_distance_m,
               "certified_speed_bound_m_s": self.certified_speed_bound_m_s,
               "guard_speed_bound_m_s": self.speed_bound_m_s if c.speed_guard != "off" else None,
               "guard_distance_safety": GUARD_DISTANCE_SAFETY,
               "certificate": {"fallback": "checked_every_substep_frozen_fallback",
                               "flag": "checked_every_substep_flag_only",
                               "off": "unchecked" if c.alias_free_hash_dims else "not_needed_c1_only"}[c.speed_guard],
               "conditional_graph_supported": self.conditional_graph_supported,
               "parity_level": "A_bitwise_given_certificate" if c.alias_free_hash_dims else "A_bitwise_unconditional",
               "fallback_hash_dims": list(c.reference_hash_dims) if c.speed_guard == "fallback" else None}
        if solver is not None and c.speed_guard != "off":
            out["guard_violation_substeps"] = solver.contact_guard_violations.numpy().tolist()
        return out
