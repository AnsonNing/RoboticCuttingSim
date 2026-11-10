"""Throwaway P2-0 probe (Newton 1.6.1 SolverVBD, Warp CPU): clamped cantilever under gravity.

Measures the mean tip deflection over the run and the oscillation amplitude decay with k_damp = 0 (no physical
damping), for several dt / iteration counts. Masses are re-lumped (rho V_cell / 8 per cell corner) because
add_soft_grid gives every node a full cell mass (builder.py:10755-10777).
"""
import json
import sys
import time

import numpy as np
import warp as wp
import newton

wp.config.quiet = True


def run(dt, iterations, T, mu=2e6, beta=20., rho=1150., L=.1, t=.01, cell=.002, device="cpu"):
    kappa = beta*mu
    lam = kappa-2*mu/3
    nx, ny, nz = int(round(L/cell)), int(round(t/cell)), int(round(t/cell))
    b = newton.ModelBuilder()
    b.add_soft_grid(pos=wp.vec3(0., 0., 0.), rot=wp.quat_identity(), vel=wp.vec3(0., 0., 0.), dim_x=nx, dim_y=ny,
                    dim_z=nz, cell_x=cell, cell_y=cell, cell_z=cell, density=rho, k_mu=mu, k_lambda=lam, k_damp=0.,
                    fix_left=True, add_surface_mesh_edges=False)
    b.color()
    model = b.finalize(device=device)
    q0 = model.particle_q.numpy()
    # lumped mass: each cell gives rho*V/8 to each corner; fixed nodes stay kinematic (mass 0)
    ijk = np.rint(q0/cell).astype(int)
    m = np.zeros(len(q0))
    ex = lambda a, n: np.minimum(a, 1)+np.minimum(n-a, 1)   # number of adjacent cells along an axis (1 or 2)
    m = rho*cell**3/8*ex(ijk[:, 0], nx)*ex(ijk[:, 1], ny)*ex(ijk[:, 2], nz)
    fixed = ijk[:, 0] == 0
    m[fixed] = 0.
    inv = np.where(m > 0, 1/np.where(m > 0, m, 1), 0.)
    model.particle_mass.assign(m.astype(np.float32))
    model.particle_inv_mass.assign(inv.astype(np.float32))
    solver = newton.solvers.SolverVBD(model, iterations=iterations)
    s0, s1 = model.state(), model.state()
    control = model.control()
    tip = ijk[:, 0] == nx
    steps = int(round(T/dt))
    zs = np.empty(steps)
    t0 = time.perf_counter()
    for k in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        zs[k] = s0.particle_q.numpy()[tip, 2].mean()-q0[tip, 2].mean()
    wall = time.perf_counter()-t0
    E = 9*kappa*mu/(3*kappa+mu)
    A, I = t*t, t**4/12
    eb = -rho*9.81*A*L**4/(8*E*I)
    f1 = .5596*np.sqrt(E*I/(rho*A*L**4))
    period = 1/f1
    # amplitude decay: peak-to-peak in the first and last full period about the running mean
    n_per = int(round(period/dt))
    first = zs[:n_per]; last = zs[-n_per:]
    a_first = (first.max()-first.min())/2; a_last = (last.max()-last.min())/2
    cycles = (steps-n_per)/n_per
    return dict(dt=dt, iterations=iterations, T=T, nodes=len(q0), tets=int(model.tet_count), mass_kg=float(m.sum()),
                mean_tip_m=float(zs[n_per:].mean()), last_period_mean_tip_m=float(last.mean()), euler_bernoulli_m=eb,
                ratio_eb=float(last.mean()/eb), f1_hz=float(f1), amp_first_m=float(a_first), amp_last_m=float(a_last),
                amp_ratio=float(a_last/a_first) if a_first > 0 else None, cycles=float(cycles), wall_s=round(wall, 1))


if __name__ == "__main__":
    dt, it, T, device = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    r = run(dt, it, T=T, device=device)
    print(json.dumps(r), flush=True)
    with open("vbd_cantilever.jsonl", "a") as fh:
        fh.write(json.dumps(r)+"\n")
