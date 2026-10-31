"""Layered attribution of cut-face unevenness (WP-12 Phase 1, ``surface_attribution_v1``; read-only, NumPy/SciPy).

For one cut digit of one run, the same set of conflicting lattice links is followed through the pipeline:

  L1 blade path        commanded swept surface of the edge (recorded substep poses) vs the declared cut plane
                       (if the scene declares one) and its own patch roughness;
  L2 assignment        current-space Hermite points at their assignment instant h_a (cut_event_snapshots_v1)
                       vs the commanded swept surface (signed: + toward digit 2);
  L3 pre-assignment    reference Hermite points P vs the commanded swept surface, in the solver-local frame where
     displacement      reference = initial positions: how far the material that was cut had moved (normal to the
                       sweep) before the solver assigned it (compression / drag ahead of and at the edge);
  L4 lattice           reference link midpoints vs P along N (what the side digits alone resolve);
  L5 post-cut          final side locus q = x_p + F_p (P - X_p) vs h_a after a rigid (Kabsch) fit per (piece, side):
     deformation       the non-rigid motion of the face after it was created (relaxation, wedge opening, push);
  L6 reconstruction    mesh vs final side locus (``surface_metrics_v1.surface_report``), per surface profile;
  L7 topology          code-0 particles at the face, stencil bridges, coverage, shells per component.

Unevenness budget: the mean square of the 4 mm-patch (radius 2 mm) plane-fit RMS at successive stages on the same
links (core links, >= 3 lattice layers from the original exterior): sweep (L1), h_a (L1+L2), final Hermite side
locus with a translation-only map x_p + (P - X_p) (+L5; the F-mapped locus and the raw face particles are reported
alongside: the first-order F map adds its own noise under large local strain), final midpoint side locus,
translation-only (+L4 as seen in current space), mesh (+L6). Successive differences are the
layer contributions (a negative reconstruction term means the extractor smooths the material roughness). These
are measurement numbers for the Phase 1 diagnosis, never thresholds, and nothing here is fed back into physics.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import surface_metrics_v1 as SM

VERSION = "surface_attribution_v1"


def edge_sweep_mesh(chains, digit, half_length_m, *, step_m=1e-3, segment_m=1e-3, footprint=None, margin_m=.004):
    """Triangulated surface swept by the edge line over consecutive poses of ``digit`` within each chain.

    ``chains``: list of ((T, 7) solver-local poses xyz+xyzw, (T,) digits). Poses are thinned to >= ``step_m`` of
    edge-centre travel (plus every rotation step > 0.5 deg), the edge (local y in [-L/2, L/2]) is split into
    ``segment_m`` pieces. ``footprint`` (points) limits the edge to the part within ``margin_m`` of the points'
    bounding box (keeps the mesh small). Returns vertices, faces, per-face unit normals oriented with the
    pose's blade normal (+ = digit-2 side, the solver's rule) and the number of strips."""
    verts, faces, normals = [], [], []
    count = 0
    lo = hi = None
    if footprint is not None and len(footprint):
        lo, hi = footprint.min(0)-margin_m, footprint.max(0)+margin_m
    for poses, cuts in chains:
        poses, cuts = np.asarray(poses, float), np.asarray(cuts)
        runs = np.flatnonzero(np.diff(np.r_[0, (cuts == digit).astype(int), 0]))
        for start, stop in zip(runs[::2], runs[1::2]):
            seg = poses[start:stop]
            if len(seg) < 2:
                continue
            keep = [0]
            for i in range(1, len(seg)):
                a, b = seg[keep[-1]], seg[i]
                angle = (Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude()
                if np.linalg.norm(b[:3]-a[:3]) >= step_m or angle > np.radians(.5) or i == len(seg)-1:
                    keep.append(i)
            seg = seg[keep]
            R = Rotation.from_quat(seg[:, 3:])
            along = R.apply([0, 1, 0])
            blade_n = R.apply([0, 0, 1])
            ys = np.linspace(-half_length_m, half_length_m, int(round(2*half_length_m/segment_m))+1)
            grid = seg[:, None, :3]+ys[None, :, None]*along[:, None, :]           # (T, M, 3)
            if lo is not None:
                inside = np.all((grid >= lo) & (grid <= hi), axis=2).any(0)
                cols = np.flatnonzero(inside)
                if not len(cols):
                    continue
                cols = np.arange(max(cols.min()-1, 0), min(cols.max()+2, len(ys)))
                grid = grid[:, cols]
            T, M = grid.shape[:2]
            base = sum(len(v) for v in verts)
            verts.append(grid.reshape(-1, 3))
            i, j = np.meshgrid(np.arange(T-1), np.arange(M-1), indexing="ij")
            a = (i*M+j).ravel()
            b, c, d = a+M, a+M+1, a+1
            f = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
            tri = verts[-1][f]
            n = SM._unit(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]))
            ref = np.r_[blade_n[i.ravel()], blade_n[i.ravel()]]
            n *= np.where(np.einsum("ij,ij->i", n, ref) < 0, -1., 1.)[:, None]
            faces.append(f+base)
            normals.append(n)
            count += 1
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3)), 0
    return np.vstack(verts), np.vstack(faces), np.vstack(normals), count


def signed_distance_to_mesh(points, vertices, faces, face_normals, *, k=24):
    """Exact unsigned point-triangle distance over the k nearest-centroid triangles, signed by the closest
    triangle's oriented normal."""
    points = np.asarray(points, float)
    if not len(points) or not len(faces):
        return np.full(len(points), np.nan)
    tri = np.asarray(vertices, float)[faces]
    tree = cKDTree(tri.mean(1))
    k = min(k, len(faces))
    _, idx = tree.query(points, k=k)
    idx = idx.reshape(len(points), k)
    best = np.full(len(points), np.inf)
    sign = np.ones(len(points))
    for j in range(k):
        t = tri[idx[:, j]]
        d, closest = SM.point_triangle_distance(points, t[:, 0], t[:, 1], t[:, 2])
        better = d < best
        best[better] = d[better]
        s = np.einsum("ij,ij->i", points-closest, face_normals[idx[:, j]])
        sign[better] = np.where(s < 0, -1., 1.)[better]
    return best*sign


def kabsch(A, B):
    """Rigid R, t minimising |R A + t - B| (no reflection)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A-ca).T@(B-cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1., 1., np.sign(np.linalg.det(Vt.T@U.T))])
    R = Vt.T@D@U.T
    return R, cb-R@ca


def _q(values, s):
    return SM._quantiles(np.abs(np.asarray(values, float)), s)


def _signed_summary(values, s):
    v = SM._finite(values)
    out = SM._quantiles(np.abs(v), s)
    out["mean_signed_m"] = float(v.mean()) if len(v) else None
    return out


def _rough(points, groups, radius, s):
    r = SM.patch_roughness(points, radius, groups=groups, max_centres=6000)
    r["quadratic_mean_spacings"] = None if r["quadratic_mean"] is None else r["quadratic_mean"]/s
    return r


def core_link_mask(lattice, pairs, layers=3):
    interior = ~SM.exterior_mask(lattice, layers)
    return interior[pairs[:, 0]] & interior[pairs[:, 1]]


def match_events(locus, log):
    """Locus row -> log row (or -1) for (cut, min(p,q), max(p,q))."""
    n = np.int64(max(locus["pairs"].max(), log["p"].max(), log["q"].max())+1) if len(log) and len(locus["cut"]) else np.int64(1)
    a, b = locus["pairs"][:, 0], locus["pairs"][:, 1]
    rkey = (locus["cut"]*n+np.minimum(a, b))*n+np.maximum(a, b)
    lo = np.minimum(log["p"], log["q"]).astype(np.int64)
    hi = np.maximum(log["p"], log["q"]).astype(np.int64)
    ekey = (log["cut"].astype(np.int64)*n+lo)*n+hi
    order = np.argsort(ekey)
    pos = np.minimum(np.searchsorted(ekey[order], rkey), max(len(ekey)-1, 0))
    out = np.full(len(rkey), -1, np.int64)
    if len(ekey):
        found = ekey[order][pos] == rkey
        out[found] = order[pos[found]]
    return out


def attribute_cut(digit, *, locus, lattice, reference, positions, labels, spacing, chains=None, half_length_m=None,
                  log=None, hermite_assign=None, assign_endpoints=None, declared_plane=None, mesh_reports=None,
                  patch_radius_m=.002, midpoint_locus=None, core_layers=3):
    """Attribution of one cut digit (module docstring). ``locus``: ``surface_metrics_v1.material_locus`` of the
    final state (Hermite source when ``log`` is given). ``hermite_assign``: (E, 3) assignment-instant points aligned
    with ``log``. ``mesh_reports``: {profile: surface_report} of the final meshes. ``midpoint_locus``: the same
    locus built without events (fallback), for L4 in current space."""
    s = float(spacing)
    rows = np.flatnonzero(locus["cut"] == digit)
    out = {"version": VERSION, "cut": int(digit), "links": int(len(rows)), "locus_source": locus["source"],
           "patch_radius_m": patch_radius_m, "core_layers": core_layers}
    if not len(rows):
        out["status"] = "NO_CONFLICTING_LINKS"
        return out
    pairs = locus["pairs"][rows]
    core = core_link_mask(lattice, pairs, core_layers)
    out["core_links"] = int(core.sum())
    out["core_relaxed"] = bool(core.sum() < 20)
    if out["core_relaxed"]:     # thin features (chips, shallow skims): every link is within 3 layers of the exterior
        core = np.ones(len(rows), bool)
    P, N, mid = locus["P"][rows], locus["N"][rows], locus["midpoint_reference"][rows]
    side = locus["side_points"][rows]
    comp = np.asarray(labels)[pairs]
    budget = {}
    # ---------------- L1 / L2 / L3 against the commanded sweep
    if chains is not None and half_length_m is not None:
        foot = np.r_[P, positions[pairs.ravel()]] if hermite_assign is None else np.r_[P, hermite_assign]
        sv, sf, sn, strips = edge_sweep_mesh(chains, digit, half_length_m, footprint=foot)
        out["sweep"] = {"strips": int(strips), "triangles": int(len(sf))}
        if len(sf):
            centres = sv[sf].mean(1)
            # sweep patch near the face (centroids within 2 mm of a core reference Hermite point)
            dP, _ = cKDTree(P[core] if core.any() else P).query(centres)
            local = centres[dP <= .002]
            out["L1_path_sweep_roughness"] = _rough(local, None, patch_radius_m, s)
            budget["sweep"] = out["L1_path_sweep_roughness"]["quadratic_mean"]
            if declared_plane is not None:
                o, n = np.asarray(declared_plane[0], float), SM._unit(np.asarray(declared_plane[1], float))
                out["L1_path_vs_declared_plane"] = _signed_summary((local-o)@n, s)
            dpre = signed_distance_to_mesh(P, sv, sf, sn)
            out["L3_pre_assignment_displacement"] = {
                "all": _signed_summary(dpre, s), "core": _signed_summary(dpre[core], s),
                "meaning": "reference Hermite point vs commanded sweep (solver-local frame, reference = initial "
                           "positions): normal displacement of the cut material before its assignment"}
            if hermite_assign is not None and log is not None:
                ev = match_events(locus, log)[rows]
                have = ev >= 0
                ha = np.full((len(rows), 3), np.nan)
                ha[have] = hermite_assign[ev[have]]
                dass = signed_distance_to_mesh(ha[have], sv, sf, sn)
                full = np.full(len(rows), np.nan)
                full[have] = dass
                out["L2_assignment_vs_sweep"] = {"all": _signed_summary(full, s), "core": _signed_summary(full[core], s),
                                                 "links_with_assignment_point": int(have.sum()),
                                                 "lead_rule_links": int(np.count_nonzero(log["flags"][ev[have]] & 4)),
                                                 "blade_z_fallback_links": int(np.count_nonzero(log["flags"][ev[have]] & 8)),
                                                 "sign_inconsistent_links": int(np.count_nonzero(log["flags"][ev[have]] & 1))}
                m = have & core
                out["L2_assignment_roughness"] = _rough(ha[m], None, patch_radius_m, s)
                budget["assignment"] = out["L2_assignment_roughness"]["quadratic_mean"]
                # ---------------- L5 post-cut deformation (rigid fit per piece and side)
                resid_n, resid, rigid = np.full((len(rows), 2), np.nan), np.full((len(rows), 2), np.nan), []
                for sd in (0, 1):
                    for c in np.unique(comp[:, sd]):
                        g = np.flatnonzero(have & (comp[:, sd] == c))
                        if len(g) < 4:
                            continue
                        R, t = kabsch(ha[g], side[g, sd])
                        moved = ha[g]@R.T+t
                        r = side[g, sd]-moved
                        nrm = locus["side_normals"][rows][g, sd]
                        resid[g, sd] = np.linalg.norm(r, axis=1)
                        resid_n[g, sd] = np.einsum("ij,ij->i", r, nrm)
                        rigid.append({"side": sd, "component": int(c), "links": int(len(g)),
                                      "rotation_deg": float(np.degrees(Rotation.from_matrix(R).magnitude())),
                                      "translation_m": float(np.linalg.norm(t+R@ha[g].mean(0)-ha[g].mean(0)))})
                if assign_endpoints is not None:
                    # particle-level post-cut residual (no F map): endpoint positions at the assignment instant vs final
                    xp, xq = assign_endpoints
                    lp = log["p"][ev[have]].astype(np.int64)
                    a_is_p = lp == pairs[have, 0]
                    xa_t = np.where(a_is_p[:, None], xp[ev[have]], xq[ev[have]])
                    xb_t = np.where(a_is_p[:, None], xq[ev[have]], xp[ev[have]])
                    pres = np.full((len(rows), 2), np.nan)
                    hv = np.flatnonzero(have)
                    for sd, xt in ((0, xa_t), (1, xb_t)):
                        for c in np.unique(comp[hv, sd]):
                            g = np.flatnonzero(comp[hv, sd] == c)
                            if len(g) < 4:
                                continue
                            idx = pairs[hv[g], sd]
                            _, first = np.unique(idx, return_index=True)
                            g = g[first]
                            idx = idx[first]
                            R, t = kabsch(xt[g], positions[idx])
                            r = positions[idx]-(xt[g]@R.T+t)
                            pres[hv[g], sd] = np.einsum("ij,ij->i", r, locus["side_normals"][rows][hv[g], sd])
                    out["L5_post_cut_particle_residual"] = {
                        "normal_residual_core": _signed_summary(pres[core].ravel(), s),
                        "meaning": "face particles: final position vs position at the assignment instant after a rigid fit "
                                   "per (piece, side), along the outward face normal (no F map)"}
                out["L5_post_cut_deformation"] = {
                    "normal_residual_all": _signed_summary(resid_n.ravel(), s),
                    "normal_residual_core": _signed_summary(resid_n[core].ravel(), s),
                    "full_residual_core": _q(resid[core].ravel(), s), "rigid_fits": rigid[:12],
                    "meaning": "final side locus vs assignment-instant Hermite point after a rigid fit per (piece, side); "
                               "physical material behaviour, never reconstruction error"}
    # ---------------- L4 lattice (reference space)
    stair = np.einsum("ij,ij->i", mid-P, N)
    out["L4_lattice_staircase_reference"] = {"all": _signed_summary(stair, s), "core": _signed_summary(stair[core], s),
                                             "expected": "uniform within +-s/2 |n_axis| (RMS ~0.29 s |n_axis|) for a planar cut"}
    out["reference_roughness_hermite"] = _rough(P[core], None, patch_radius_m, s)
    out["reference_roughness_midpoint"] = _rough(mid[core], None, patch_radius_m, s)
    # ---------------- final loci (current space)
    groups = np.r_[comp[core, 0]*2, comp[core, 1]*2+1]
    out["final_hermite_side_locus_roughness"] = _rough(np.r_[side[core, 0], side[core, 1]], groups, patch_radius_m, s)
    budget["final_hermite_locus"] = out["final_hermite_side_locus_roughness"]["quadratic_mean"]
    # the same faces without the F map: translation-only side points, and the face particles themselves
    tr = np.r_[positions[pairs[core, 0]]+(P[core]-reference[pairs[core, 0]]),
               positions[pairs[core, 1]]+(P[core]-reference[pairs[core, 1]])]
    out["final_translation_only_side_locus_roughness"] = _rough(tr, groups, patch_radius_m, s)
    trm = np.r_[positions[pairs[core, 0]]+(mid[core]-reference[pairs[core, 0]]),
                positions[pairs[core, 1]]+(mid[core]-reference[pairs[core, 1]])]
    out["final_translation_only_midpoint_locus_roughness"] = _rough(trm, groups, patch_radius_m, s)
    budget["final_physics"] = out["final_translation_only_side_locus_roughness"]["quadratic_mean"]
    budget["final_lattice"] = out["final_translation_only_midpoint_locus_roughness"]["quadratic_mean"]
    ids, first = np.unique(np.r_[pairs[core, 0], pairs[core, 1]], return_index=True)
    out["final_face_particles_roughness"] = _rough(positions[ids], groups[first], patch_radius_m, s)
    budget["final_face_particles"] = out["final_face_particles_roughness"]["quadratic_mean"]
    if midpoint_locus is not None:
        mrow = match_rows(locus, midpoint_locus)[rows]
        ok = mrow >= 0
        ms = midpoint_locus["side_points"][mrow[ok]]
        mc = core[ok]
        g2 = np.r_[comp[ok][mc, 0]*2, comp[ok][mc, 1]*2+1]
        out["final_midpoint_side_locus_roughness"] = _rough(np.r_[ms[mc, 0], ms[mc, 1]], g2, patch_radius_m, s)
        budget["final_midpoint_locus"] = out["final_midpoint_side_locus_roughness"]["quadratic_mean"]
        dm = np.einsum("ij,ij->i", ms[:, 0]-side[ok, 0], locus["side_normals"][rows][ok, 0])
        out["L4_lattice_staircase_current"] = _signed_summary(dm[mc], s)
    # ---------------- L6 reconstruction
    if mesh_reports:
        out["L6_reconstruction"] = {}
        for profile, rep in mesh_reports.items():
            if rep.get("status") != "MEASURED":
                out["L6_reconstruction"][profile] = {"status": rep.get("status")}
                continue
            per = (rep.get("per_cut") or {}).get(str(int(digit)))
            src = per if per is not None else rep
            out["L6_reconstruction"][profile] = {k: src.get(k) for k in (
                "distance_to_side_locus_plane", "distance_to_side_locus_plane_core", "signed_bias",
                "locus_to_mesh_distance", "coverage", "normal_error_deg_core", "roughness_mesh_core")}
            out["L6_reconstruction"][profile]["scope"] = "this cut digit" if per is not None else "all cuts pooled"
            out["L6_reconstruction"][profile]["rim_all_cuts"] = rep.get("rim")
            budget[f"mesh_{profile}"] = (src.get("roughness_mesh_core") or {}).get("quadratic_mean")
    out["budget_quadratic_mean_patch_rms_m"] = budget
    out["budget_increments_m"] = increments(budget)
    out["status"] = "MEASURED"
    return out


def match_rows(locus_a, locus_b):
    """Row of locus_b for each row of locus_a (same (cut, side-1, side-2) link) or -1."""
    n = np.int64(max(locus_a["pairs"].max(initial=0), locus_b["pairs"].max(initial=0))+1)
    ka = (locus_a["cut"]*n+locus_a["pairs"][:, 0])*n+locus_a["pairs"][:, 1]
    kb = (locus_b["cut"]*n+locus_b["pairs"][:, 0])*n+locus_b["pairs"][:, 1]
    order = np.argsort(kb)
    out = np.full(len(ka), -1, np.int64)
    if len(kb):
        pos = np.minimum(np.searchsorted(kb[order], ka), len(kb)-1)
        found = kb[order][pos] == ka
        out[found] = order[pos[found]]
    return out


def increments(budget):
    """Signed RMS increments sign(d) sqrt(|d|), d = b_k^2 - b_{k-1}^2, along the stage order: sweep (L1),
    assignment-instant Hermite points (L2), final Hermite side locus with a translation-only map (L5, physical
    post-cut motion of the face particles), final link-midpoint side locus, translation-only (L4, what the labels
    alone resolve), mesh (L6, relative to the L4 stage, i.e. what the extractor adds or removes)."""
    order = ["sweep", "assignment", "final_physics", "final_lattice"]
    names = {"sweep": "L1_blade_path", "assignment": "L2_assignment", "final_physics": "L5_post_cut_physics",
             "final_lattice": "L4_lattice_in_current_space"}
    out, prev = {}, 0.
    for key in order:
        v = budget.get(key)
        if v is None:
            continue
        d = v**2-prev**2
        out[names[key]] = float(np.sign(d)*np.sqrt(abs(d)))
        prev = v
    base = budget.get("final_lattice", budget.get("final_physics"))
    for key, v in budget.items():
        if key.startswith("mesh_") and v is not None and base is not None:
            d = v**2-base**2
            out[f"L6_reconstruction_{key[5:]}"] = float(np.sign(d)*np.sqrt(abs(d)))
    return out


def topology_layer(nodes, weights, gradients, codes, cuts, *, lattice, labels, masses, reference=None):
    """L7: code-0 particles adjacent to conflicting links per digit, stencil bridge report, detached mass."""
    from .law8_validation import bridge_report
    from .cut_topology import decode_digits
    digits = decode_digits(np.asarray(codes), cuts)
    pairs, _ = SM.lattice_links(lattice)
    out = {"per_cut": {}}
    labels = np.asarray(labels)
    mass = np.bincount(labels, weights=masses)
    out["mechanical_components"] = int(len(mass))
    out["mass_fractions"] = (np.sort(mass)[::-1]/mass.sum())[:10].tolist()
    out["detached_mass_fraction"] = float(1-mass.max()/mass.sum())
    for k in range(int(cuts)):
        d = digits[:, k]
        if not np.any(d):
            continue
        conflict = ((d[pairs[:, 0]] == 1) & (d[pairs[:, 1]] == 2)) | ((d[pairs[:, 0]] == 2) & (d[pairs[:, 1]] == 1))
        touch = np.zeros(len(d), bool)
        touch[pairs[conflict].ravel()] = True
        # code-0 particles that are lattice neighbours of a labelled particle of each side
        zero = d == 0
        n1 = np.zeros(len(d), bool)
        n2 = np.zeros(len(d), bool)
        for a, b in ((0, 1), (1, 0)):
            pa, pb = pairs[:, a], pairs[:, b]
            n1[pa[(d[pb] == 1)]] = True
            n2[pa[(d[pb] == 2)]] = True
        out["per_cut"][str(k)] = {"conflicting_links": int(conflict.sum()), "labelled": int(np.count_nonzero(d)),
                                  "code0_touching_both_sides_on_lattice": int((zero & n1 & n2).sum()),
                                  "bridge_report": bridge_report(nodes, weights, gradients, codes, digit_index=k, cuts=cuts,
                                                                 packed=True, reference_positions=reference, limit=10)}
    return out


def stencil(points, dx):
    """The tlmpm_v5 reference stencil (nodes, weights, gradients) in NumPy (same formula as TLMPM.__init__)."""
    points = np.asarray(points, np.float32)
    origin = np.floor(points.min(0)/dx)*dx-2*dx
    dims = np.ceil((points.max(0)-origin)/dx).astype(int)+3
    cells = np.floor((points-origin)/dx).astype(int)
    fraction = (points-origin)/dx-cells
    n = len(points)
    nodes = np.empty((n, 8), np.int32)
    weights = np.empty((n, 8), np.float32)
    gradients = np.empty((n, 8, 3), np.float32)
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells+bit
        nodes[:, corner] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
        f = np.where(bit, fraction, 1-fraction)
        weights[:, corner] = f.prod(1)
        for axis in range(3):
            gradients[:, corner, axis] = (2*bit[axis]-1)*f[:, (axis+1) % 3]*f[:, (axis+2) % 3]/dx
    return nodes, weights, gradients
