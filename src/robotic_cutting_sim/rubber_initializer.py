"""Static-equilibrium initializer for rubber scenes (ADR 0023 §7, task R1.5).

Dynamic relaxation, labelled ``initializer``: it is never part of an episode and its result is the starting state,
not a dynamic observable. Two methods:

- ``mass_proportional`` (id ``relax_mass_proportional_v1``, used by RUBBER-PHYS-02/03/05): damping impulse
  -c m_p v_p dt at the solver CFL, default c = 2 omega_1 (critical for the first linear bending mode; every mode then
  decays at least like exp(-omega_1 t));
- ``kinetic`` (kinetic damping: particle velocities zeroed when the particle kinetic energy falls below its previous
  check value): admitted only at CFL <= 0.25. ADR 0023 E4 preserved failure: at CFL 0.5 it went non-finite.

Stop rule ``static_stop_v2`` (protocol ``rubber_phys_tests_v2``), checked every ``check_interval`` seconds:

  (a) marker stationarity: max_k |m_k(t) - m_k(t_prev)| <= stationarity_tol * max_k |m_k(t) - m_k(reference)| for
      ``consecutive`` checks in a row (m_k: the 9 cross-section centroids), and
  (b) when the scene has Dirichlet groups and gravity: |R_z + M g| / (M g) < reaction_tol (R the group reaction
      averaged over the interval).

The ADR 0023 §7 / v1 velocity stop ("max |v| < 1e-5 m/s") is not used: the grid velocity settles at a float32 noise
floor of about 1e-5 to 1e-4 m/s in small stiff strips, and the pure-FLIP particle velocity v_p = W^T a exceeds the
grid velocity M_l^-1 M_c a it maps to for grid-scale modes (P2-1 protocol diagnostics). Both speeds are reported.

Stop rule ``static_stop_v3`` (protocol ``rubber_phys_tests_v3``, ADR 0023 §7 amendment). P2-1 follow-up finding: the
v2 marker test is defeated by float32 position rounding. Positions are advanced as x = x_start + dt v_grid in float32;
once dt |v| falls below half an ulp of x (about 2e-5 m/s for z ~ 3 mm at dt 5.8 us) the increment rounds to zero and
the markers stop moving exactly (every formal v2 initializer stopped with a marker change of exactly 0.0, at
1.25-1.45 T1) while F, and the float64 displacement, still creep (0.2 % of the deflection and 0.36 % of the elastic
energy in the re-run of RUBBER-PHYS-02 Lt10_t3mm_Wt1_nt3). v3 therefore
  (i)   integrates the MUSL grid displacement u_I = sum dt v_I in float64 after every substep (initializer only) and
        evaluates the markers from x64_p = x_p(start) + sum_c w_pc u_I (rounding-free; ``markers_f64_m`` in the
        receipt is the static state to measure);
  (ii)  requires a minimum relaxation time of ``min_time`` (default 3 T1: with c = 2 omega_1 every mode amplitude has
        decayed by exp(-6 pi) ~ 7e-9, the critically damped first mode by (1 + 6 pi) exp(-6 pi) ~ 1.4e-7);
  (iii) applies the stationarity test (a) to the float64 markers and keeps the reaction test (b).
At convergence the particle velocities are set to zero (the static state) and the damping is removed.
"""
import math
import time

import numpy as np
import warp as wp

METHOD_IDS = {"mass_proportional": "relax_mass_proportional_v1", "kinetic": "relax_kinetic_v1"}
STOP_RULE = "static_stop_v2"
STOP_RULES = ("static_stop_v2", "static_stop_v3")
KINETIC_MAX_CFL = 0.25


@wp.kernel
def _accumulate_displacement(u: wp.array(dtype=wp.vec3d), new_v: wp.array(dtype=wp.vec3), dt: wp.float64):
    """static_stop_v3: float64 grid displacement u_I += dt v_I (MUSL velocity after the BC pass of this substep)."""
    i = wp.tid()
    u[i] = u[i]+dt*wp.vec3d(new_v[i])


class _Float64Markers:
    """Rounding-free marker centroids x64 = x(start) + sum_c w_pc u_I (initializer measurement only)."""

    def __init__(self, live):
        s, sc = live.solver, live.scene
        self.solver = s
        self.ids = sc.marker_particles
        self.offsets = sc.marker_offsets
        self.x0 = s.x.numpy()[self.ids].astype(np.float64)
        self.w = sc.weights[self.ids].astype(np.float64)
        self.nodes = sc.nodes[self.ids]
        self.u = wp.zeros(s.node_count, dtype=wp.vec3d, device=s.device)

    def accumulate(self):
        s = self.solver
        wp.launch(_accumulate_displacement, s.node_count, [self.u, s.new_v, wp.float64(s.dt)], device=s.device)

    def markers(self):
        u = self.u.numpy()
        x = self.x0+np.einsum("pc,pcj->pj", self.w, u[self.nodes])
        return np.array([x[a:b].mean(0) for a, b in zip(self.offsets[:-1], self.offsets[1:])])


def relax_to_static(live, *, method="mass_proportional", omega1=None, rate=None, check_interval=None,
                    max_time=None, stationarity_tol=1e-4, consecutive=3, reaction_tol=1e-3,
                    kinetic_check_substeps=10, stop_rule=STOP_RULE, min_time=None, keep_history=False):
    """Relax ``live`` (a ``LiveRubberMaterial`` just reset) to static equilibrium; returns a receipt dict.
    ``stop_rule``: ``static_stop_v2`` (protocol v2, default; float32 markers) or ``static_stop_v3`` (float64 markers,
    minimum relaxation time ``min_time``, default 3 T1). ``keep_history=True`` returns every check row (with the tip
    marker position, float64 under static_stop_v3) as ``history`` (P2-2 F10 diagnostics; the stop is unchanged)."""
    if method not in METHOD_IDS:
        raise ValueError(f"method {method!r} not in {sorted(METHOD_IDS)}")
    if stop_rule not in STOP_RULES:
        raise ValueError(f"stop_rule {stop_rule!r} not in {STOP_RULES}")
    s = live.solver
    cfl = live.step_rule["cfl_actual"]
    if method == "kinetic" and cfl > KINETIC_MAX_CFL+1e-12:
        raise ValueError(f"kinetic damping is refused at CFL {cfl:.3f} > {KINETIC_MAX_CFL} "
                         "(ADR 0023 E4 preserved failure); use CFL <= 0.25 or mass_proportional")
    if omega1 is None and (rate is None or check_interval is None or max_time is None):
        raise ValueError("omega1 (first-mode angular frequency) or explicit rate, check_interval and max_time")
    period = None if omega1 is None else 2*math.pi/omega1
    rate = (2*omega1 if rate is None else float(rate)) if method == "mass_proportional" else 0.0
    check_interval = period/20 if check_interval is None else float(check_interval)
    max_time = 15*period if max_time is None else float(max_time)
    v3 = stop_rule == "static_stop_v3"
    if v3:
        if min_time is None and period is None:
            raise ValueError("static_stop_v3 needs omega1 or an explicit min_time")
        min_time = 3*period if min_time is None else float(min_time)
    else:
        min_time = 0.0
    f64 = _Float64Markers(live) if v3 else None
    chunk = max(1, int(round(check_interval/s.dt)))
    g = float(np.linalg.norm(s.gravity_tuple))
    weight = live.total_mass*g
    use_reaction = s.G > 0 and g > 0
    reference = live.scene.marker_reference
    s.set_initializer_damping(rate, initializer=True)
    start = time.perf_counter()
    substeps, history, converged, streak = 0, [], False, 0
    previous = None
    ke_prev = math.inf
    while substeps*s.dt < max_time-1e-12:
        s.reaction.zero_()
        s.index.zero_()       # groups hold (start = end after reset), so the interpolation parameter is irrelevant
        for k in range(chunk):
            s.step()
            if f64 is not None:
                f64.accumulate()
            if method == "kinetic" and (k+1) % kinetic_check_substeps == 0:
                ke = s.diagnostics.read()[0]["kinetic_particle"]
                if ke < ke_prev:
                    s.v.zero_()
                    ke = 0.0
                ke_prev = ke
        substeps += chunk
        reaction = -s.reaction.numpy()[:s.G].sum(0)/(chunk*s.dt) if s.G else np.zeros(6)
        diag = s.diagnostics.read()[0]
        markers = f64.markers() if f64 is not None else s.diagnostics.marker_x.numpy()
        displacement = float(np.abs(markers-reference).max())
        change = math.inf if previous is None else float(np.abs(markers-previous).max())
        previous = markers
        stationary = change <= stationarity_tol*displacement
        residual = abs(reaction[2]+weight)/weight if use_reaction else 0.0
        history.append({"t_s": substeps*s.dt, "marker_change_m": change, "max_marker_displacement_m": displacement,
                        "reaction_residual": float(residual), "max_grid_speed_m_s": diag["max_grid_speed_m_s"],
                        "max_particle_speed_m_s": diag["max_speed_m_s"], "tip_m": markers[-1].tolist(),
                        "elastic_j": diag["elastic"]})
        if diag["nonfinite"] or not np.isfinite(displacement):
            break
        streak = streak+1 if stationary else 0
        if (streak >= consecutive and (not use_reaction or residual < reaction_tol)
                and substeps*s.dt >= min_time-1e-12):
            converged = True
            break
    s.v.zero_()
    s.set_initializer_damping(0.0)
    s.reaction.zero_()
    s.work.zero_()
    s.index.zero_()
    final = s.diagnostics.read()[0]
    last = history[-1] if history else {}
    out_f64 = {}
    if f64 is not None:
        m64 = f64.markers()
        out_f64 = {"markers_f64_m": m64.tolist(),
                   "markers_f32_minus_f64_max_m": float(np.abs(s.diagnostics.marker_x.numpy()-m64).max()),
                   "min_time_s": min_time}
    return {"initializer": METHOD_IDS[method], "stop_rule": stop_rule, **out_f64,
            "label": "initializer (not a dynamic observable)", "converged": bool(converged),
            "time_s": substeps*s.dt, "substeps": substeps, "rate_1_s": rate, "check_interval_s": chunk*s.dt,
            "max_time_s": max_time, "stationarity_tol": stationarity_tol, "consecutive": consecutive,
            "reaction_tol": reaction_tol, "weight_n": weight, "cfl": cfl, "wall_s": time.perf_counter()-start,
            "nonfinite": bool(final["nonfinite"]), "final": last, "history_tail": history[-5:],
            **({"history": history} if keep_history else {})}
