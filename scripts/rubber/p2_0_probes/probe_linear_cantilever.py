"""Throwaway P2-0 probe: small-deflection cantilever with the linearised TLMPM operator.

The frozen tlmpm_v5 stencil (trilinear weights/gradients on the reference grid, particles as quadrature points,
constructor formula tlmpm_v5.py:735-750) is a Q1 finite-element discretisation with particle quadrature. For small
displacements the explicit TLMPM converges to the static solution of K u = f with
K = sum_p V_p B_p^T D B_p (standard) or the J-bar / B-bar variant (deviatoric part per particle, volumetric part with
the reference-cell-averaged dilatation). This probe assembles K from the same stencil, clamps the nodes at X <= 0,
applies gravity and compares the mean tip deflection with Euler-Bernoulli and Timoshenko beam theory.
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def strip_particles(L, W, t, dx, ppc, clamp_cells=1):
    s = dx/ppc
    nx = int(round((L+clamp_cells*dx)/s)); ny = int(round(W/s)); nz = int(round(t/s))
    xs = -clamp_cells*dx+(np.arange(nx)+.5)*s
    ys = (np.arange(ny)+.5)*s
    zs = (np.arange(nz)+.5)*s
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return X, s


def stencil(points, dx):
    """tlmpm_v5.TLMPM.__init__ stencil formula (float64)."""
    origin = np.floor(points.min(0)/dx)*dx-2*dx
    dims = np.ceil((points.max(0)-origin)/dx).astype(int)+3
    cells = np.floor((points-origin)/dx).astype(int)
    fraction = (points-origin)/dx-cells
    n = len(points)
    nodes = np.empty((n, 8), np.int64); weights = np.empty((n, 8)); gradients = np.empty((n, 8, 3))
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells+bit
        nodes[:, corner] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
        f = np.where(bit, fraction, 1-fraction)
        weights[:, corner] = f.prod(1)
        for axis in range(3):
            gradients[:, corner, axis] = (2*bit[axis]-1)*f[:, (axis+1) % 3]*f[:, (axis+2) % 3]/dx
    cell_id = (cells[:, 0]*dims[1]+cells[:, 1])*dims[2]+cells[:, 2]
    return origin, dims, nodes, weights, gradients, cell_id


def bmatrix(G):
    """G [n,8,3] -> B [n,6,24] (Voigt xx,yy,zz,yz,xz,xy; engineering shear)."""
    n = len(G)
    B = np.zeros((n, 6, 24))
    for c in range(8):
        gx, gy, gz = G[:, c, 0], G[:, c, 1], G[:, c, 2]
        B[:, 0, 3*c] = gx; B[:, 1, 3*c+1] = gy; B[:, 2, 3*c+2] = gz
        B[:, 3, 3*c+1] = gz; B[:, 3, 3*c+2] = gy
        B[:, 4, 3*c] = gz; B[:, 4, 3*c+2] = gx
        B[:, 5, 3*c] = gy; B[:, 5, 3*c+1] = gx
    return B


def solve(L, W, t, n_t, mu, kappa, rho, ppc=2, jbar=False, g=9.81):
    dx = t/n_t
    X, s = strip_particles(L, W, t, dx, ppc)
    V = s**3
    origin, dims, nodes, weights, gradients, cell_id = stencil(X, dx)
    n = len(X)
    lam = kappa-2*mu/3
    B = bmatrix(gradients)
    if jbar:
        Ddev = np.zeros((6, 6))
        Ddev[:3, :3] = 2*mu*(np.eye(3)-np.ones((3, 3))/3)
        Ddev[3:, 3:] = mu*np.eye(3)
        Ke = V*np.einsum("pik,ij,pjl->pkl", B, Ddev, B)
        # volumetric: cell-averaged divergence row b_c = sum_p (V_p/V_c) div_p (all particles of a cell share nodes)
        div = B[:, 0]+B[:, 1]+B[:, 2]            # [n,24]
        uc, inv = np.unique(cell_id, return_inverse=True)
        count = np.bincount(inv).astype(float)
        bsum = np.zeros((len(uc), 24)); np.add.at(bsum, inv, div)
        bbar = bsum/count[:, None]
        Vc = count*V
        Kc = kappa*Vc[:, None, None]*np.einsum("ck,cl->ckl", bbar, bbar)
        # scatter cell blocks with the node list of any particle of the cell
        first = np.zeros(len(uc), int); first[inv[::-1]] = np.arange(n)[::-1]
        cell_nodes = nodes[first]
    else:
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[:3, :3] += 2*mu*np.eye(3)
        D[3:, 3:] = mu*np.eye(3)
        Ke = V*np.einsum("pik,ij,pjl->pkl", B, D, B)
    dof = (3*nodes[:, :, None]+np.arange(3)).reshape(n, 24)
    rows = np.repeat(dof, 24, 1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(3*dims.prod(),)*2).tocsr()
    if jbar:
        cdof = (3*cell_nodes[:, :, None]+np.arange(3)).reshape(-1, 24)
        K = K+sp.coo_matrix((Kc.ravel(), (np.repeat(cdof, 24, 1).ravel(), np.tile(cdof, (1, 24)).ravel())),
                            shape=K.shape).tocsr()
    f = np.zeros(3*dims.prod())
    np.add.at(f, 3*nodes.ravel()+2, (weights*(-rho*V*g)).ravel())
    used = np.unique(nodes)
    node_x = origin[0]+(used//(dims[1]*dims[2]))*dx
    free_nodes = used[node_x > 1e-12*dx]
    free = (3*free_nodes[:, None]+np.arange(3)).ravel()
    Kf = K[free][:, free]
    u = np.zeros_like(f)
    u[free] = spla.spsolve(Kf.tocsc(), f[free])
    up = np.einsum("pc,pcj->pj", weights, u.reshape(-1, 3)[nodes])
    tip = X[:, 0] > X[:, 0].max()-1e-9
    return float(up[tip, 2].mean()), n, len(used)


def beam(L, W, t, mu, kappa, rho, g=9.81):
    E = 9*kappa*mu/(3*kappa+mu)
    nu = (3*kappa-2*mu)/(2*(3*kappa+mu))
    A, I = W*t, W*t**3/12
    q = rho*g*A
    eb = q*L**4/(8*E*I)
    ks = 10*(1+nu)/(12+11*nu)
    timo = eb+q*L**2/(2*ks*mu*A)
    return -eb, -timo, E, nu


if __name__ == "__main__":
    rho, mu = 1150., 5e5
    out = []
    slender = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10]
    for L_over_t in slender:
        t = .004; L = L_over_t*t; W = t
        for beta in (2.1667, 20., 1000.):
            kappa = beta*mu
            eb, timo, E, nu = beam(L, W, t, mu, kappa, rho)
            for jbar in (False, True):
                for n_t in (1, 2, 3, 4, 6, 8):
                    if L_over_t*n_t*2*n_t*2*n_t*2 > 260000:
                        continue
                    t0 = time.perf_counter()
                    tip, n, nodes = solve(L, W, t, n_t, mu, kappa, rho, jbar=jbar)
                    row = dict(L_over_t=L_over_t, beta=beta, nu=round(nu, 4), jbar=jbar, n_t=n_t, particles=n,
                               nodes=nodes, tip_m=tip, euler_bernoulli_m=eb, timoshenko_m=timo,
                               ratio_eb=tip/eb, ratio_timo=tip/timo, wall_s=round(time.perf_counter()-t0, 2))
                    out.append(row)
                    print(json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "linear_cantilever.json", "w"), indent=1)
