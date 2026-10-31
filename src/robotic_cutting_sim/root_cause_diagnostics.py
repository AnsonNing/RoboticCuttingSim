"""Read-only NumPy diagnostics for the WP-12 Phase 4 root-cause verification (methods review §5, rows V1-V4).

Folded from the reviewer's scratch scripts (``p1_links``, ``p1_identity_eval``, ``p2_chip``, ``p3_mis_sided``,
``p3_obl20``, ``p3_visibility_all``, ``p3_obl20_vis``; review §8) into tested repository code. Every function reads
saved or host-copied states and returns numbers; none writes a solver array, a code or a record. Nothing here is a
cutting law, a fracture criterion or an acceptance rule: the STATIC visibility relabel is a shadow prediction, and
a link-length threshold is only a locator of drift, never fracture evidence (user rule, 2026-10-31).

Definitions
-----------
* Lattice links: reference six-neighbour pairs (p, q) with q = p + e_axis on the particle lattice.
* Link inconsistency: |dx - F̄ dX| / s with dx = x_q - x_p, dX = X_q - X_p, F̄ = (F_p + F_q)/2, on links whose
  packed codes do not conflict (methods review §2.1).
* Drift identity: with d the ``flip_drift_v1`` record, r = dx - F̄ dX is explained by d_q - d_p on same-code links;
  explained fraction = 1 - sum |r - (d_q - d_p)|^2 / sum |r|^2 per band (review §2.2).
* Bands: by the reference distance of the link midpoint to the declared cut locus, in units of s:
  (0, 1], (1, 3], (3, 6], (6, inf) (the first band includes 0).
"""
import numpy as np

LOW_BITS = 0x55555555
BANDS = ((0., 1.), (1., 3.), (3., 6.), (6., np.inf))


def band_name(lo, hi):
    return f"{lo:g}-{hi:g}s" if np.isfinite(hi) else f">{lo:g}s"


def conflict_packed(a, b):
    """NumPy version of tlmpm_v5._conflict_packed (elementwise)."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    la, ha = a & LOW_BITS, (a >> 1) & LOW_BITS
    lb, hb = b & LOW_BITS, (b >> 1) & LOW_BITS
    a1, a2 = la ^ (la & ha), ha ^ (la & ha)
    b1, b2 = lb ^ (lb & hb), hb ^ (lb & hb)
    return ((a1 & b2) | (a2 & b1)) != 0


def lattice_links(lattice):
    """[(p, q, axis)] reference links with lattice[q] = lattice[p] + e_axis (each link once)."""
    lat = np.asarray(lattice, np.int64)
    lo = lat.min(0)
    dims = lat.max(0)-lo+2

    def key(L):
        return ((L[:, 0]-lo[0])*dims[1]+(L[:, 1]-lo[1]))*dims[2]+(L[:, 2]-lo[2])
    k = key(lat)
    order = np.argsort(k, kind="stable")
    ks = k[order]
    out = []
    for axis in range(3):
        e = np.zeros(3, np.int64)
        e[axis] = 1
        kq = key(lat+e)
        pos = np.searchsorted(ks, kq)
        pos[pos >= len(ks)] = 0
        ok = ks[pos] == kq
        out.append((np.flatnonzero(ok), order[pos[ok]], axis))
    return out


def plane_distance(points, planes):
    """min over planes of |(X - o) . n| (planes: [(origin, normal)])."""
    X = np.asarray(points, float)
    d = [np.abs((X-np.asarray(o, float))@(np.asarray(n, float)/np.linalg.norm(n))) for o, n in planes]
    return np.min(d, axis=0)


def _stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0, "p50": None, "p99": None, "max": None}
    return {"n": int(len(values)), "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def link_consistency(reference, lattice, positions, F, codes, spacing, locus_distance, *, drift=None,
                     assign_substep=None, bands=BANDS):
    """Banded |dx - F̄dX|/s on non-conflicting links (+ the drift identity and assignment-history split).

    ``locus_distance``: a callable mapping (n, 3) reference points to their distance (m) from the declared cut
    locus, evaluated at the link midpoints; or a per-particle distance array (the link then gets the mean of its
    endpoint values)."""
    X = np.asarray(reference, float)
    x = np.asarray(positions, float)
    F = np.asarray(F, float)
    codes = np.asarray(codes, np.int64)
    s = float(spacing)
    P, Q = [], []
    for p, q, _ in lattice_links(lattice):
        keep = ~conflict_packed(codes[p], codes[q])
        P.append(p[keep])
        Q.append(q[keep])
    p, q = np.concatenate(P), np.concatenate(Q)
    dX = X[q]-X[p]
    r = (x[q]-x[p])-np.einsum("nij,nj->ni", .5*(F[p]+F[q]), dX)
    inc = np.linalg.norm(r, axis=1)/s
    mid = .5*(X[p]+X[q])
    dist = (locus_distance(mid) if callable(locus_distance) else
            .5*(np.asarray(locus_distance)[p]+np.asarray(locus_distance)[q]))/s
    same = codes[p] == codes[q]
    res = None
    if drift is not None:
        d = np.asarray(drift, float)
        res = np.linalg.norm(r-(d[q]-d[p]), axis=1)/s
    hist_same = None
    if assign_substep is not None:
        a = np.asarray(assign_substep).reshape(len(X), -1)
        hist_same = (a[p] == a[q]).all(1)
    out = {}
    for lo, hi in bands:
        m = (dist <= hi) if lo == 0 else (dist > lo) & (dist <= hi)
        row = _stats(inc[m])
        if res is not None:
            ms = m & same
            denom = float((inc[ms]**2).sum())
            row["identity"] = {"same_code_links": int(ms.sum()),
                               "explained_fraction": (1.-float((res[ms]**2).sum())/denom) if denom > 0 else None,
                               "residual_p99": _stats(res[ms])["p99"], "residual_max": _stats(res[ms])["max"]}
        if hist_same is not None:
            row["same_assignment_history"] = _stats(inc[m & hist_same])
            row["different_assignment_history"] = _stats(inc[m & ~hist_same])
        out[band_name(lo, hi)] = row
    first, far = out[band_name(*bands[0])], out[band_name(*bands[-1])]
    out["first_to_far_p99_ratio"] = (first["p99"]/far["p99"]) if first["p99"] is not None and far["p99"] else None
    return out


def principal_stretch_max(F):
    return np.linalg.svd(np.asarray(F, float), compute_uv=False)[:, 0]


def chip_links(reference, lattice, positions, F, codes, spacing, region, *, long_s=2.0, consistent_s=1.25):
    """V2.1: links with both endpoints in ``region`` and non-conflicting codes; long links (|dx| > long_s s) and how
    many of them are F-consistent (|F̄dX| < consistent_s s means F does not stretch the link)."""
    X, x, F = np.asarray(reference, float), np.asarray(positions, float), np.asarray(F, float)
    codes, region, s = np.asarray(codes, np.int64), np.asarray(region, bool), float(spacing)
    sv = principal_stretch_max(F)
    J = np.linalg.det(F)
    L, FL, St, Jm = [], [], [], []
    for p, q, _ in lattice_links(lattice):
        k = (~conflict_packed(codes[p], codes[q])) & region[p] & region[q]
        p, q = p[k], q[k]
        dX = X[q]-X[p]
        L.append(np.linalg.norm(x[q]-x[p], axis=1)/s)
        FL.append(np.linalg.norm(np.einsum("nij,nj->ni", .5*(F[p]+F[q]), dX), axis=1)/s)
        St.append(np.maximum(sv[p], sv[q]))
        Jm.append(np.minimum(J[p], J[q]))
    L, FL, St, Jm = (np.concatenate(v) for v in (L, FL, St, Jm))
    long = L > long_s
    out = {"links": int(len(L)), "long_links": int(long.sum()),
           "long_fraction": float(long.mean()) if len(L) else None,
           "long_rule_s": long_s, "f_consistent_rule_s": consistent_s}
    if long.any():
        out.update(long_link_length_s={"p50": float(np.median(L[long])), "max": float(L[long].max())},
                   long_with_F_length_below_rule=float((FL[long] < consistent_s).mean()),
                   long_ratio_dx_over_FdX={"p50": float(np.median(L[long]/FL[long])), "min": float((L[long]/FL[long]).min())},
                   endpoint_max_principal_stretch={"p50": float(np.median(St[long])), "max": float(St[long].max())},
                   det_F_p5=float(np.percentile(Jm[long], 5)))
    out["all_links_F_max_stretch_p99"] = float(np.percentile(St, 99)) if len(St) else None
    return out


def elastic_state(F, Fpi, young, poisson, yield_stress):
    """Per particle: trial elastic state of tlmpm_v5._stress (Fe = F Fpi, Hencky log strain, J2 return mapping).

    Returns dict of arrays (J/m^3 for energies): psi_dev = mu |dev e|^2, psi_vol = K/2 tr^2, psi = psi_dev + psi_vol
    (the Hencky energy mu |e|^2 + lam/2 tr^2 after the return), psi_tensile = psi_dev + K/2 <tr>_+^2 (Amor split),
    von_mises_kirchhoff_pa, pressure_pa (Cauchy, -K tr / J), J."""
    F, Fpi = np.asarray(F, float), np.asarray(Fpi, float)
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    fe = F@Fpi
    sigma = np.linalg.svd(fe, compute_uv=False)
    logs = np.log(np.maximum(sigma, .05))
    tr = logs.sum(1)
    dev = logs-tr[:, None]/3
    norm = np.linalg.norm(dev, axis=1)
    limit = np.sqrt(2/3)*yield_stress/(2*mu)
    scale = np.where(norm > limit, limit/np.maximum(norm, 1e-30), 1.)
    dev = dev*scale[:, None]
    psi_dev = mu*(dev**2).sum(1)
    bulk = lam+2*mu/3
    psi_vol = .5*bulk*tr**2
    J = np.linalg.det(F)
    return {"psi_dev": psi_dev, "psi_vol": psi_vol, "psi": psi_dev+psi_vol,
            "psi_tensile": psi_dev+.5*bulk*np.maximum(tr, 0.)**2,
            "von_mises_kirchhoff_pa": np.sqrt(1.5)*2*mu*np.linalg.norm(dev, axis=1),
            "pressure_pa": -(lam+2*mu/3)*tr/np.where(np.abs(J) > 1e-12, J, 1.), "J": J,
            "psi_dev_bound": yield_stress**2/(6*mu)}


def column_thickness(reference, region, spacing):
    """Reference thickness (m) of ``region`` in the vertical lattice column of each particle (0 outside)."""
    X, region, s = np.asarray(reference, float), np.asarray(region, bool), float(spacing)
    ij = np.floor((X[:, :2]-X[:, :2].min(0))/s+.5).astype(np.int64)
    key = ij[:, 0]*1000003+ij[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv, weights=region.astype(float))
    return np.where(region, counts[inv]*s, 0.)


def cut_face_particles(lattice, codes):
    """Particles with at least one lattice six-neighbour whose packed code conflicts (the solver's cut faces)."""
    codes = np.asarray(codes, np.int64)
    face = np.zeros(len(codes), bool)
    for p, q, _ in lattice_links(lattice):
        c = conflict_packed(codes[p], codes[q])
        face[p[c]] = True
        face[q[c]] = True
    return face


def chip_energy(reference, F, Fpi, plastic, region, spacing, material, *, gc=.05, griffith_t=.006, history=None,
                codes=None, lattice=None, locus_distance=None):
    """V2.2: stress state in the chip ``region`` vs the rest; Griffith check psi+ * t vs G_c (provisional).

    ``history``: optional dict with per-particle history maxima ``psi_tensile_max_j_m3`` / ``psi_total_max_j_m3``
    (tensile_energy_max_v1); the gating Griffith number uses the history maximum of the tensile energy when given,
    the final state otherwise. With ``codes`` and ``lattice`` the gating set excludes cut-face particles (a lattice
    neighbour with a conflicting code: there law 8's codes already separate the material); they are reported."""
    st = elastic_state(F, Fpi, material["young"], material["poisson"], material["yield_stress"])
    if history is not None:
        st["psi_tensile_history_max"] = np.asarray(history["psi_tensile_max_j_m3"], float)
        st["psi_total_history_max"] = np.asarray(history["psi_total_max_j_m3"], float)
    region = np.asarray(region, bool)
    t = column_thickness(reference, region, spacing)
    eps_p = np.asarray(plastic, float)

    def summary(mask):
        if not mask.any():
            return None
        keys = ("psi_dev", "psi_vol", "psi", "psi_tensile", "von_mises_kirchhoff_pa", "pressure_pa",
                "psi_tensile_history_max", "psi_total_history_max")
        return {k: _stats(st[k][mask]) for k in keys if k in st} | {
            "plastic_strain": {**_stats(eps_p[mask]), "p90": float(np.percentile(eps_p[mask], 90)),
                               "fraction_yielded": float((eps_p[mask] > 0).mean())}}
    face = (cut_face_particles(lattice, codes) if codes is not None and lattice is not None
            else np.zeros(len(region), bool))
    source = "psi_tensile_history_max" if "psi_tensile_history_max" in st else "psi_tensile"
    gt = st[source]*t
    total = (st["psi_total_history_max"] if "psi_total_history_max" in st else st["psi"])*t
    thin_all = region & (t < griffith_t)

    def where(mask):
        over = mask & (gt >= gc)
        w = {"particles": int(mask.sum()), "with_psi_t_ge_gc": int(over.sum()),
             "max_psi_times_t_j_m2": float(gt[mask].max()) if mask.any() else None,
             "total_energy_max_psi_times_t_j_m2": float(total[mask].max()) if mask.any() else None,
             "total_energy_with_psi_t_ge_gc": int((mask & (total >= gc)).sum())}
        if over.any():
            w["exceeding_psi_j_m3"] = _stats(st[source][over])
            w["exceeding_column_thickness_m"] = _stats(t[over])
            if locus_distance is not None:
                dist = np.asarray(locus_distance(np.asarray(reference, float)[over]) if callable(locus_distance)
                                  else np.asarray(locus_distance)[over], float)/float(spacing)
                w["exceeding_locus_distance_s"] = _stats(dist)
        return w
    gate = where(thin_all & ~face)
    return {"chip": summary(region), "substrate": summary(~region),
            "psi_dev_bound_j_m3": float(st["psi_dev_bound"]), "griffith": {
                "energy": source, "gc_j_m2": gc, "thin_rule_t_m": griffith_t,
                "gating_set": ("thin chip particles (column thickness < thin_rule_t_m) not on a cut face (no conflicting "
                               "lattice neighbour)" if face.any() or codes is not None
                               else "thin chip particles (column thickness < thin_rule_t_m)"),
                "thin_chip_particles": gate["particles"], "thin_particles_with_psi_t_ge_gc": gate["with_psi_t_ge_gc"],
                "max_psi_times_t_thin_j_m2": gate["max_psi_times_t_j_m2"], "gating": gate,
                "cut_face_thin_reported": where(thin_all & face) if codes is not None else None,
                "all_chip_max_psi_times_t_j_m2": float(gt[region].max()) if region.any() else None,
                "chip_thickness_max_m": float(t.max())},
            "note": "plastic_strain is the accumulated deviatoric return (no failure strain is calibrated)"}


def reference_stencil(reference, dx):
    """(nodes, weights) of the tlmpm_v5 reference stencil rebuilt from the reference points, in their own dtype
    (pass the float32 points the solver received to reproduce it exactly; float64 input can differ at points lying
    exactly on a node plane, where one of the two weights is ~0)."""
    X = np.asarray(reference)
    origin = np.floor(X.min(0)/dx)*dx-2*dx
    dims = np.ceil((X.max(0)-origin)/dx).astype(int)+3
    cells = np.floor((X-origin)/dx).astype(int)
    frac = (X-origin)/dx-cells
    nodes = np.empty((len(X), 8), np.int64)
    weights = np.empty((len(X), 8))
    for k in range(8):
        b = np.array([(k >> j) & 1 for j in range(3)])
        xyz = cells+b
        nodes[:, k] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
        weights[:, k] = np.where(b, frac, 1-frac).prod(1)
    return nodes, weights


def mis_sided_bridges(codes, nodes, weights, s_ref, cut=0):
    """V3.1: code-0 bridges of digit ``cut`` and how many need a mis-sided label (P3b).

    A bridge is a digit-0 particle whose stencil touches nodes holding digit-1 and digit-2 particles. It needs a
    mis-sided label if it does not touch both a node holding a digit-1 particle with s_ref < 0 and a node holding
    a digit-2 particle with s_ref >= 0 (review §2.4, P3b); otherwise it is an in-core wedge (P3a)."""
    c = (np.asarray(codes, np.int64) >> (2*cut)) & 3
    nodes = np.asarray(nodes, np.int64)
    valid = np.asarray(weights) > 0
    s_ref = np.asarray(s_ref, float)
    N = int(nodes.max())+1

    def has(mask):
        a = np.zeros(N, bool)
        a[nodes[mask][valid[mask]]] = True
        return a
    h1, h2 = has(c == 1), has(c == 2)
    h1ok, h2ok = has((c == 1) & (s_ref < 0)), has((c == 2) & (s_ref >= 0))
    zero = np.flatnonzero(c == 0)
    nz, vz = nodes[zero], valid[zero]
    touch = lambda h: (h[nz] & vz).any(1)  # noqa: E731
    bridge = touch(h1) & touch(h2)
    proper = touch(h1ok) & touch(h2ok)
    wrong = int(((c == 2) & (s_ref < 0)).sum()+((c == 1) & (s_ref >= 0)).sum())
    b = zero[bridge]
    return {"bridges": int(bridge.sum()), "need_mis_sided_label": int((bridge & ~proper).sum()),
            "in_core_wedge": int((bridge & proper).sum()), "mis_sided_labelled_particles": wrong,
            "share_p3b": float((bridge & ~proper).sum()/bridge.sum()) if bridge.any() else None,
            "bridge_indices": b}


def visibility_relabel(reference, codes, nodes, weights, plane_origin, plane_normal, edge_stop_z, *, near_m=.008,
                       cut=0):
    """V3.3 STATIC CRAMP-style all-pairs visibility shadow relabel (never a law; review §2.4).

    A digit-0 particle p with |s| <= near_m takes its reference side iff it shares a stencil node with at least one
    opposite-side particle within near_m, and every such reference segment p-q crosses the plane at z >= h (the
    swept patch above the final edge height). Returns the predicted codes and the label statistics."""
    from scipy.sparse import coo_matrix
    X = np.asarray(reference, float)
    codes = np.asarray(codes, np.int64)
    nodes = np.asarray(nodes, np.int64)
    valid = np.asarray(weights) > 0
    n = np.asarray(plane_normal, float)/np.linalg.norm(plane_normal)
    s = (X-np.asarray(plane_origin, float))@n
    N = len(X)
    pid = np.repeat(np.arange(N), nodes.shape[1]).reshape(nodes.shape)
    inc = coo_matrix((np.ones(valid.sum()), (pid[valid], nodes[valid])), shape=(N, int(nodes.max())+1)).tocsr()
    digit = (codes >> (2*cut)) & 3
    near = np.abs(s) <= near_m
    cand = np.flatnonzero((digit == 0) & near)
    sub = np.flatnonzero(near)
    new = codes.copy()
    if not len(cand):
        return new, {"labels": 0, "labels_below_edge_stop": 0, "max_abs_s_m": 0.}
    share = (inc[cand]@inc[sub].T).tocoo()
    p, q = cand[share.row], sub[share.col]
    opp = np.sign(s[p])*np.sign(s[q]) < 0
    p, q = p[opp], q[opp]
    t = s[p]/(s[p]-s[q])
    zc = X[p, 2]+t*(X[q, 2]-X[p, 2])
    hit = zc >= edge_stop_z
    anyp = np.zeros(N, bool)
    anyp[p] = True
    miss = np.zeros(N, bool)
    miss[p[~hit]] = True
    lab = anyp & ~miss
    new[lab] |= np.where(s[lab] < 0, 1, 2) << (2*cut)
    return new, {"labels": int(lab.sum()), "labels_below_edge_stop": int((lab & (X[:, 2] < edge_stop_z)).sum()),
                 "max_abs_s_m": float(np.abs(s[lab]).max()) if lab.any() else 0.}


def series_fit(dt1, f1, dt2, f2):
    """Two-point fit of F(dt) = F_inf / (1 + (dt/dt0)^2) (review §2.5). Returns (F_inf, dt0) or (None, None)."""
    num = f2*dt2**2-f1*dt1**2
    den = f1-f2
    if den == 0 or num/den <= 0:
        return None, None
    dt0 = float(np.sqrt(num/den))
    return float(f1*(1+(dt1/dt0)**2)), dt0


def series_predict(f_inf, dt0, dt):
    return float(f_inf/(1+(dt/dt0)**2))
