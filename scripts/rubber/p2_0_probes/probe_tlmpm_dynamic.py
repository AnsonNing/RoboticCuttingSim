"""Throwaway P2-0 probe (Warp CPU or CUDA): dynamic TLMPM cantilever with the frozen grid kernels.

Unchanged frozen kernels: tlmpm_v5._p2g/_grid_update/_g2p/_musl/_deformation and tlmpm_v6._grid_positions (S = 1,
all codes 0, constant slot table). New probe-only kernels: J-bar cell sum, neo_hookean_jbar stress, nodal Dirichlet
clamp. No contact, no damping. Reports the stability limit, the mean tip deflection (static sag) and the energy /
amplitude history with pic = 0 and pic = 0.05.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from robotic_cutting_sim import tlmpm_v5 as v5            # noqa: E402
from robotic_cutting_sim.tlmpm_v6 import _grid_positions   # noqa: E402

wp.config.quiet = True


@wp.kernel
def _jsum(F: wp.array(dtype=wp.mat33), cell: wp.array(dtype=int), cell_sum: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(cell_sum, cell[p], wp.determinant(F[p]))


@wp.kernel
def _stress_nh(F: wp.array(dtype=wp.mat33), cell: wp.array(dtype=int), cell_sum: wp.array(dtype=float),
               cell_count: wp.array(dtype=float), stress: wp.array(dtype=wp.mat33), mu: float, kappa: float,
               jbar: int):
    p = wp.tid()
    f = F[p]
    J = wp.determinant(f)
    Jb = J
    if jbar != 0:
        c = cell[p]
        Jb = cell_sum[c]/cell_count[c]
    # cof(F) = J F^-T without dividing by J
    c0 = wp.cross(wp.vec3(f[0, 1], f[1, 1], f[2, 1]), wp.vec3(f[0, 2], f[1, 2], f[2, 2]))
    c1 = wp.cross(wp.vec3(f[0, 2], f[1, 2], f[2, 2]), wp.vec3(f[0, 0], f[1, 0], f[2, 0]))
    c2 = wp.cross(wp.vec3(f[0, 0], f[1, 0], f[2, 0]), wp.vec3(f[0, 1], f[1, 1], f[2, 1]))
    cof = wp.mat33(c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2])
    I1 = wp.ddot(f, f)
    Jc = wp.max(J, 0.05)
    a = mu*wp.pow(Jc, -2.0/3.0)
    p_bar = 0.5*kappa*(Jb-1.0/Jb)
    stress[p] = a*f+(p_bar-a*I1/(3.0*Jc))*cof


@wp.kernel
def _clamp(new_v: wp.array(dtype=wp.vec3), fixed: wp.array(dtype=int)):
    i = wp.tid()
    if fixed[i] != 0:
        new_v[i] = wp.vec3(0.0)


class Strip:
    def __init__(self, L, W, t, n_t, mu, beta, rho, ppc=2, device="cpu", jbar=True, pic=0.0, cfl=.5, dt=None):
        dx = t/n_t; s = dx/ppc
        nx, ny, nz = int(round((L+dx)/s)), int(round(W/s)), int(round(t/s))
        X = np.stack(np.meshgrid(-dx+(np.arange(nx)+.5)*s, (np.arange(ny)+.5)*s, (np.arange(nz)+.5)*s,
                                 indexing="ij"), -1).reshape(-1, 3)
        origin = np.floor(X.min(0)/dx)*dx-2*dx
        dims = np.ceil((X.max(0)-origin)/dx).astype(int)+3
        cells = np.floor((X-origin)/dx).astype(int)
        fr = (X-origin)/dx-cells
        n = len(X)
        nodes = np.empty((n, 8), np.int32); w = np.empty((n, 8), np.float32); g = np.empty((n, 8, 3), np.float32)
        for corner in range(8):
            bit = np.array([(corner >> j) & 1 for j in range(3)])
            xyz = cells+bit
            nodes[:, corner] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
            f = np.where(bit, fr, 1-fr)
            w[:, corner] = f.prod(1)
            for axis in range(3):
                g[:, corner, axis] = (2*bit[axis]-1)*f[:, (axis+1) % 3]*f[:, (axis+2) % 3]/dx
        cell_id = nodes[:, 0].copy()   # min-corner node id = cell id
        count = np.bincount(cell_id, minlength=int(dims.prod())).astype(np.float32)
        node_x = origin[0]+np.arange(int(dims.prod()))//(dims[1]*dims[2])*dx
        fixed = (node_x <= 1e-9*dx).astype(np.int32)
        self.kappa, self.mu, self.rho = beta*mu, mu, rho
        cp = np.sqrt((self.kappa+4*mu/3)/rho)
        self.dt = float(dt if dt is not None else cfl*dx/cp)
        self.cfl = self.dt*cp/dx
        self.dx, self.s, self.n, self.nn, self.X = dx, s, n, int(dims.prod()), X
        self.V = s**3
        self.m = np.full(n, rho*self.V, np.float32)
        self.jbar, self.pic, self.device = int(jbar), float(pic), device
        self.tip = X[:, 0] > X[:, 0].max()-1e-9
        with wp.ScopedDevice(device):
            self.x = wp.array(X.astype(np.float32), dtype=wp.vec3)
            self.x_start = wp.zeros(n, dtype=wp.vec3)
            self.v = wp.zeros(n, dtype=wp.vec3)
            self.mass = wp.array(self.m, dtype=float)
            self.F = wp.array(np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), dtype=wp.mat33)
            self.stress = wp.zeros(n, dtype=wp.mat33)
            self.impulse = wp.zeros(n, dtype=wp.vec3)
            self.nodes = wp.array(nodes, dtype=int); self.weights = wp.array(w, dtype=float)
            self.gradients = wp.array(g, dtype=wp.vec3)
            self.slot = wp.zeros((n, 8), dtype=int); self.slot_code = wp.zeros(self.nn, dtype=int)
            self.gm = wp.zeros(self.nn, dtype=float); self.gp = wp.zeros(self.nn, dtype=wp.vec3)
            self.gj = wp.zeros(self.nn, dtype=wp.vec3); self.old = wp.zeros(self.nn, dtype=wp.vec3)
            self.new_v = wp.zeros(self.nn, dtype=wp.vec3)
            self.cell = wp.array(cell_id, dtype=int); self.cell_sum = wp.zeros(self.nn, dtype=float)
            self.cell_count = wp.array(np.maximum(count, 1), dtype=float)
            self.fixed = wp.array(fixed, dtype=int)
        self.count_host = count
        self.gravity = wp.vec3(0., 0., -9.81)

    def step(self):
        n, nn, dt = self.n, self.nn, self.dt
        with wp.ScopedDevice(self.device):
            wp.copy(self.x_start, self.x)
            if self.jbar:
                self.cell_sum.zero_()
                wp.launch(_jsum, n, [self.F, self.cell, self.cell_sum])
            wp.launch(_stress_nh, n, [self.F, self.cell, self.cell_sum, self.cell_count, self.stress, self.mu,
                                      self.kappa, self.jbar])
            self.gm.zero_(); self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._p2g, (n, 8), [self.v, self.mass, self.slot, self.stress, self.impulse, self.nodes,
                                        self.weights, self.gradients, self.gm, self.gp, self.gj, 1, self.V, dt,
                                        self.gravity])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            wp.launch(_clamp, nn, [self.new_v, self.fixed])
            wp.launch(v5._g2p, n, [self.x, self.v, self.slot, self.nodes, self.weights, self.old, self.new_v, 1, dt,
                                   self.pic])
            self.gp.zero_(); self.gj.zero_()
            wp.launch(v5._musl, (n, 8), [self.v, self.mass, self.slot, self.nodes, self.weights, self.gp, 1])
            wp.launch(v5._grid_update, (1, nn), [self.gm, self.gp, self.gj, self.old, self.new_v, self.slot_code, 1])
            wp.launch(_clamp, nn, [self.new_v, self.fixed])
            wp.launch(v5._deformation, n, [self.F, self.slot, self.nodes, self.gradients, self.new_v, 1, dt])
            wp.launch(_grid_positions, n, [self.x, self.x_start, self.slot, self.nodes, self.weights, self.new_v, 1,
                                           dt])

    def energy(self):
        F = self.F.numpy().astype(np.float64); x = self.x.numpy().astype(np.float64)
        J = np.linalg.det(F); I1 = np.einsum("pij,pij->p", F, F)
        w_iso = self.mu/2*(np.abs(J)**(-2/3)*I1-3)
        if self.jbar:
            cid = self.cell.numpy()
            sums = np.bincount(cid, weights=J, minlength=self.nn); cnt = np.bincount(cid, minlength=self.nn)
            used = cnt > 0
            Jb = sums[used]/cnt[used]
            e_vol = np.sum(cnt[used]*self.V*self.kappa/4*(Jb**2-1-2*np.log(Jb)))
            jb_min, jb_max = Jb.min(), Jb.max()
        else:
            e_vol = np.sum(self.V*self.kappa/4*(J**2-1-2*np.log(J)))
            jb_min, jb_max = J.min(), J.max()
        e_el = np.sum(self.V*w_iso)+e_vol
        gm = self.gm.numpy().astype(np.float64); gv = self.new_v.numpy().astype(np.float64)
        ke_grid = .5*np.sum(gm*np.sum(gv*gv, 1))
        v = self.v.numpy().astype(np.float64)
        ke_p = .5*np.sum(self.m*np.sum(v*v, 1))
        pe = np.sum(self.m*9.81*x[:, 2])
        tip = x[self.tip, 2].mean()-self.X[self.tip, 2].mean()
        return dict(e_el=e_el, ke_grid=ke_grid, ke_particle=ke_p, pe=pe, tip=tip, J_min=J.min(), J_max=J.max(),
                    Jbar_min=jb_min, Jbar_max=jb_max, finite=bool(np.isfinite(x).all()))


def run(label, T, every, **kw):
    s = Strip(**kw)
    steps = int(round(T/s.dt))
    hist = []
    t0 = time.perf_counter()
    e0 = s.energy()
    for k in range(1, steps+1):
        s.step()
        if k % every == 0 or k == steps:
            e = s.energy(); e["t"] = k*s.dt
            hist.append(e)
            if not e["finite"] or abs(e["tip"]) > 1.0:
                break
    wall = time.perf_counter()-t0
    tips = np.array([h["tip"] for h in hist]); ts = np.array([h["t"] for h in hist])
    tot = np.array([h["e_el"]+h["ke_particle"]+h["pe"]-e0["pe"] for h in hist])
    totg = np.array([h["e_el"]+h["ke_grid"]+h["pe"]-e0["pe"] for h in hist])
    return dict(label=label, particles=s.n, nodes=s.nn, dt=s.dt, cfl=s.cfl, steps=k, wall_s=round(wall, 1),
                us_per_step=1e6*wall/k, finite=bool(hist[-1]["finite"]), tip_final=float(tips[-1]),
                tip_mean_second_half=float(tips[len(tips)//2:].mean()),
                tip_min=float(tips.min()), tip_max=float(tips.max()), t=ts.tolist(), tip=tips.tolist(),
                total_energy_particle_ke=tot.tolist(), total_energy_grid_ke=totg.tolist(),
                Jbar_min=float(min(h["Jbar_min"] for h in hist)), Jbar_max=float(max(h["Jbar_max"] for h in hist)),
                J_min=float(min(h["J_min"] for h in hist)), J_max=float(max(h["J_max"] for h in hist)), kw={
                    k: v for k, v in kw.items()})


if __name__ == "__main__":
    mode = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    base = dict(L=.1, W=.01, t=.01, mu=2e6, beta=20., rho=1150., device=device)
    rows = []
    if mode == "stability":
        for cfl in [float(c) for c in sys.argv[3].split(",")]:
            r = run(f"cfl{cfl}", T=float(sys.argv[4]), every=50, n_t=3, cfl=cfl, **base)
            rows.append(r)
            print(json.dumps({k: r[k] for k in ("label", "dt", "cfl", "steps", "finite", "tip_final", "us_per_step",
                                                "Jbar_min", "Jbar_max")}), flush=True)
    elif mode == "dynamic":
        n_t = int(sys.argv[3]); T = float(sys.argv[4])
        for pic, cfl in ((0.0, .5), (0.05, .5), (0.05, .25), (0.0, .25)):
            r = run(f"nt{n_t}_pic{pic}_cfl{cfl}", T=T, every=20, n_t=n_t, pic=pic, cfl=cfl, **base)
            rows.append(r)
            e = np.array(r["total_energy_particle_ke"]); eg = np.array(r["total_energy_grid_ke"])
            tips = np.array(r["tip"])
            print(json.dumps({k: r[k] for k in ("label", "particles", "dt", "cfl", "steps", "finite",
                                                "tip_mean_second_half", "tip_min", "tip_max", "us_per_step",
                                                "Jbar_min", "Jbar_max", "J_min", "J_max")}
                             | {"E_particle_first_last": [e[len(e)//10], e[-1]],
                                "E_grid_first_last": [eg[len(eg)//10], eg[-1]],
                                "amp_first_quarter": float((tips[:len(tips)//4].max()-tips[:len(tips)//4].min())/2),
                                "amp_last_quarter": float((tips[-len(tips)//4:].max()-tips[-len(tips)//4:].min())/2)}),
                  flush=True)
    elif mode == "jbar_off":
        n_t = int(sys.argv[3]); T = float(sys.argv[4])
        r = run(f"nt{n_t}_nojbar", T=T, every=20, n_t=n_t, pic=0.0, cfl=.5, jbar=False, **base)
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("label", "particles", "dt", "steps", "tip_mean_second_half", "tip_min",
                                            "tip_max", "finite")}), flush=True)
    json.dump(rows, open(f"tlmpm_{mode}_{device.replace(':', '')}_{'_'.join(sys.argv[3:])}.json", "w"))


def timing(device, L, W, t, n_t, reps=5, substeps=100):
    """Per-substep GPU cost of the lean step (no contacts) via one CUDA graph of ``substeps`` substeps."""
    s = Strip(L=L, W=W, t=t, n_t=n_t, mu=5e5, beta=20., rho=1150., device=device, cfl=.5)
    wp.load_module(module=v5.__name__, device=device)
    wp.load_module(module="robotic_cutting_sim.tlmpm_v6", device=device)
    wp.load_module(module=__name__, device=device)
    for _ in range(3):
        s.step()
    wp.synchronize_device(device)
    with wp.ScopedCapture(device=device) as cap:
        for _ in range(substeps):
            s.step()
    times = []
    for _ in range(reps):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize_device(device)
        times.append((time.perf_counter()-t0)/substeps)
    e = s.energy()
    return dict(device=device, particles=s.n, nodes=s.nn, dt=s.dt, ms_per_substep=[1e3*x for x in times],
                median_ms=1e3*float(np.median(times)), finite=e["finite"], Jbar=[e["Jbar_min"], e["Jbar_max"]])


if __name__ == "__main__" and sys.argv[1] == "timing":
    dev = sys.argv[2]
    out = []
    for (L, W, t, n_t) in ((.2, .02, .004, 4), (.2, .02, .004, 2), (.4, .04, .008, 4)):
        r = timing(dev, L, W, t, n_t)
        out.append(r)
        print(json.dumps(r), flush=True)
    json.dump(out, open(f"tlmpm_timing_{dev.replace(':', '')}.json", "w"))
