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

The ADR 0023 §7 / v1 velocity stop ("max |v| < 1e-5 m/s") is not used: under pure FLIP the particle velocity keeps a
grid null-space component that never reaches the grid and no grid-applied damping removes, and the grid velocity
settles at a float32 noise floor of about 1e-5 to 1e-4 m/s in small stiff strips (P2-1 protocol diagnostics). Both
speeds are reported. At convergence the particle velocities are set to zero (the static state) and the damping is
removed.
"""
import math
import time

import numpy as np

METHOD_IDS = {"mass_proportional": "relax_mass_proportional_v1", "kinetic": "relax_kinetic_v1"}
STOP_RULE = "static_stop_v2"
KINETIC_MAX_CFL = 0.25


def relax_to_static(live, *, method="mass_proportional", omega1=None, rate=None, check_interval=None,
                    max_time=None, stationarity_tol=1e-4, consecutive=3, reaction_tol=1e-3,
                    kinetic_check_substeps=10):
    """Relax ``live`` (a ``LiveRubberMaterial`` just reset) to static equilibrium; returns a receipt dict."""
    if method not in METHOD_IDS:
        raise ValueError(f"method {method!r} not in {sorted(METHOD_IDS)}")
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
            if method == "kinetic" and (k+1) % kinetic_check_substeps == 0:
                ke = s.diagnostics.read()[0]["kinetic_particle"]
                if ke < ke_prev:
                    s.v.zero_()
                    ke = 0.0
                ke_prev = ke
        substeps += chunk
        reaction = -s.reaction.numpy()[:s.G].sum(0)/(chunk*s.dt) if s.G else np.zeros(6)
        diag = s.diagnostics.read()[0]
        markers = s.diagnostics.marker_x.numpy()
        displacement = float(np.abs(markers-reference).max())
        change = math.inf if previous is None else float(np.abs(markers-previous).max())
        previous = markers
        stationary = change <= stationarity_tol*displacement
        residual = abs(reaction[2]+weight)/weight if use_reaction else 0.0
        history.append({"t_s": substeps*s.dt, "marker_change_m": change, "max_marker_displacement_m": displacement,
                        "reaction_residual": float(residual), "max_grid_speed_m_s": diag["max_grid_speed_m_s"],
                        "max_particle_speed_m_s": diag["max_speed_m_s"]})
        if diag["nonfinite"] or not np.isfinite(displacement):
            break
        streak = streak+1 if stationary else 0
        if streak >= consecutive and (not use_reaction or residual < reaction_tol):
            converged = True
            break
    s.v.zero_()
    s.set_initializer_damping(0.0)
    s.reaction.zero_()
    s.work.zero_()
    s.index.zero_()
    final = s.diagnostics.read()[0]
    last = history[-1] if history else {}
    return {"initializer": METHOD_IDS[method], "stop_rule": STOP_RULE,
            "label": "initializer (not a dynamic observable)", "converged": bool(converged),
            "time_s": substeps*s.dt, "substeps": substeps, "rate_1_s": rate, "check_interval_s": chunk*s.dt,
            "max_time_s": max_time, "stationarity_tol": stationarity_tol, "consecutive": consecutive,
            "reaction_tol": reaction_tol, "weight_n": weight, "cfl": cfl, "wall_s": time.perf_counter()-start,
            "nonfinite": bool(final["nonfinite"]), "final": last, "history_tail": history[-5:]}
