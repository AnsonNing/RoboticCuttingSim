"""Throwaway P2-0 probe: large-deflection gravity sag of a TLMPM strip vs the inextensible Euler elastica.

Static equilibrium is reached by PROBE-ONLY kinetic damping (dynamic relaxation: particle velocities are zeroed
whenever the kinetic energy passes a peak). This is an equilibrium initializer, not dynamics.
"""
import json
import sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_tlmpm_dynamic import Strip


def elastica(alpha):
    """theta'' = alpha (1 - s) cos(theta) on s in [0,1] (lengths in L, theta < 0 downwards), theta(0)=0, theta'(1)=0.
    Returns (tip drop / L, tip horizontal position / L)."""
    def f(s, y):
        return np.vstack([y[1], alpha*(1-s)*np.cos(y[0])])
    def bc(a, b):
        return np.array([a[0], b[1]])
    s = np.linspace(0, 1, 401)
    y0 = np.vstack([-alpha*(s-s**2/2+s**3/6)*0+-.1*s, -.1*np.ones_like(s)])
    sol = solve_bvp(f, bc, s, y0, tol=1e-8, max_nodes=100000)
    th = sol.sol(s)[0]
    z = np.concatenate([[0], np.cumsum((np.sin(th[1:])+np.sin(th[:-1]))/2*np.diff(s))])
    x = np.concatenate([[0], np.cumsum((np.cos(th[1:])+np.cos(th[:-1]))/2*np.diff(s))])
    return float(-z[-1]), float(x[-1]), bool(sol.success)


def relax(n_t, T, cfl=.25, mu=2e5, beta=20., rho=1150., L=.1, t=.005):
    s = Strip(L=L, W=t, t=t, n_t=n_t, mu=mu, beta=beta, rho=rho, cfl=cfl, pic=0.0, device="cpu")
    steps = int(round(T/s.dt))
    ke_prev, resets, t0 = 0., 0, time.perf_counter()
    m = s.m.astype(np.float64)
    for k in range(steps):
        s.step()
        if k % 10 == 0:
            v = s.v.numpy().astype(np.float64)
            ke = .5*np.sum(m*np.sum(v*v, 1))
            if ke < ke_prev:
                s.v.zero_(); resets += 1; ke = 0.
            ke_prev = ke
    x = s.x.numpy().astype(np.float64)
    tip = s.tip
    v = s.v.numpy(); e = s.energy()
    E = 9*s.kappa*s.mu/(3*s.kappa+s.mu)
    q = rho*9.81*t*t; EI = E*t**4/12
    alpha = q*L**3/EI
    drop, xtip, ok = elastica(alpha)
    eb = q*L**4/(8*EI)
    return dict(n_t=n_t, cfl=cfl, particles=s.n, dt=s.dt, steps=steps, resets=resets, wall_s=round(time.perf_counter()-t0, 1),
                alpha=alpha, tip_drop_m=float(s.X[tip, 2].mean()-x[tip, 2].mean()),
                tip_x_m=float(x[tip, 0].mean()), elastica_drop_m=drop*L, elastica_tip_x_m=xtip*L, elastica_ok=ok,
                linear_eb_drop_m=eb, ratio_elastica=float((s.X[tip, 2].mean()-x[tip, 2].mean())/(drop*L)),
                max_speed_final=float(np.abs(v).max()), Jbar=[e["Jbar_min"], e["Jbar_max"]], J=[e["J_min"], e["J_max"]])


if __name__ == "__main__":
    out = []
    for n_t in [int(a) for a in sys.argv[1].split(",")]:
        r = relax(n_t, float(sys.argv[2]))
        out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f"large_sag_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
