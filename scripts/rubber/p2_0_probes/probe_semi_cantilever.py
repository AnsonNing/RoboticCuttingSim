"""Throwaway: same Newton model as probe_vbd_cantilever (add_soft_grid, re-lumped masses, fix_left) stepped with
SolverSemiImplicit (explicit FEM, no iterations) to check the setup independently of VBD convergence."""
import json, sys, time
import numpy as np
import warp as wp
import newton
wp.config.quiet = True
mu, beta, rho, L, t, cell = 2e6, 20., 1150., .1, .01, .002
kappa = beta*mu; lam = kappa-2*mu/3
nx, ny, nz = 50, 5, 5
b = newton.ModelBuilder()
b.add_soft_grid(pos=wp.vec3(0., 0., 0.), rot=wp.quat_identity(), vel=wp.vec3(0., 0., 0.), dim_x=nx, dim_y=ny, dim_z=nz,
                cell_x=cell, cell_y=cell, cell_z=cell, density=rho, k_mu=mu, k_lambda=lam, k_damp=0., fix_left=True,
                add_surface_mesh_edges=False)
model = b.finalize(device="cpu")
q0 = model.particle_q.numpy(); ijk = np.rint(q0/cell).astype(int)
ex = lambda a, n: np.minimum(a, 1)+np.minimum(n-a, 1)
m = rho*cell**3/8*ex(ijk[:, 0], nx)*ex(ijk[:, 1], ny)*ex(ijk[:, 2], nz)
m[ijk[:, 0] == 0] = 0.
model.particle_mass.assign(m.astype(np.float32))
model.particle_inv_mass.assign(np.where(m > 0, 1/np.where(m > 0, m, 1), 0.).astype(np.float32))
model.particle_grid = None
solver = newton.solvers.SolverSemiImplicit(model)
s0, s1 = model.state(), model.state(); control = model.control()
dt = float(sys.argv[1]); T = float(sys.argv[2]); steps = int(round(T/dt))
tip = ijk[:, 0] == nx
zs = []
t0 = time.perf_counter()
for k in range(steps):
    s0.clear_forces()
    solver.step(s0, s1, control, None, dt)
    s0, s1 = s1, s0
    if k % 20 == 0:
        zs.append(float(s0.particle_q.numpy()[tip, 2].mean()-q0[tip, 2].mean()))
zs = np.array(zs)
E = 9*kappa*mu/(3*kappa+mu); eb = -rho*9.81*t*t*L**4/(8*E*t**4/12)
r = dict(solver="SemiImplicit", dt=dt, T=T, steps=steps, tip_min=float(zs.min()), tip_max=float(zs.max()),
         midpoint=float((zs.min()+zs.max())/2), euler_bernoulli=eb, ratio_eb=float((zs.min()+zs.max())/2/eb),
         finite=bool(np.isfinite(zs).all()), wall_s=round(time.perf_counter()-t0, 1))
print(json.dumps(r)); open("semi_cantilever.jsonl", "a").write(json.dumps(r)+"\n")
