"""Cut-surface metrics changed by ``surface_protocol_v2`` (WP-12 Phase 3, after the 2026-10-31 decision review).

Version ``surface_metrics_v2``. ``surface_metrics_v1`` is not modified: it stays the implementation of the frozen v1
protocol and of every v2 row that ``configs/wp12/surface_protocol_v2.json`` keeps unchanged (coverage, rim, validity,
topology, the reference-volume rows, S0, G2, S9 ...). This module adds only the changed definitions P1-P7 of
``docs/reports/WP12_PHASE3_DECISION_REVIEW.md`` section 3; each is applied identically to every method (A =
``particle_surface_v1``, B v1, B v2). Pure NumPy/SciPy, read-only: nothing here changes particles, labels or meshes.

* P1 ``classify_mesh_vertices_v2`` (``region_rule_v2``): a vertex's mechanical component comes from the extractor's
  own provenance (``vertex_component_from_ranges``: CuttingSurface / CutAwareSurface ``component_ranges``), not from
  its nearest particle. Cut row and side from the nearest same-component sample, as in v1. Every vertex also gets a
  classification ``margin``: distance to the nearest sample of the next sample set (the exterior or another cut digit)
  minus the distance to the chosen set. Max rows use vertices with margin >= 0.5 s; P95 rows use all vertices.
* P2 ``matched_scale_normal`` / ``normal_floor_split_halves``: the angle between the mesh's own 3 mm patch plane and the
  3 mm locus plane (same component, cut and side), against a per-state floor = P95 angle between 3 mm planes of two
  disjoint random halves of the locus samples (declared seed) + 1 deg.
* P3 ``current_distance_floor`` / ``current_p95_rows``: current-distance P95 limit per side group = the locus
  leave-one-out residual P95 against its own 2 mm plane + 0.10 s.
* P4 ``reference_consistency`` / ``volume_current_row``: the current-volume row reads INCONCLUSIVE_REFERENCE (never
  PASS) when the cells' V / det F contradicts the reference (a component's mean departs from 1 by > 1 %, or > 5 % of its
  cells are < 0.9).
* P5 ``slow_frame_pairs`` / ``temporal_stability_v2``: slow-motion residual only on frame pairs with material
  displacement P95 <= 0.10 s; transport by inverse-distance interpolation of same-component particle displacement.
* P6 ``intersecting_pairs`` / ``attribute_intersections`` / ``faces_nearer_other_component_v2``: physical overlap
  (different pieces or opposite sides of one cut, with particle gap < 1 s or crossed sides) is reported apart from
  unexplained intersections; every other intra-component pair is unexplained.
* P7 ``chip_volume_v2``: the chip current-volume row on the lattice-faced part only; a particle-cell fallback part is
  reported separately as "by construction".

Lengths are metres unless a key ends in ``_s`` (spacings) or ``_deg``.
"""
import numpy as np

from . import surface_metrics_v1 as SM1

VERSION = "surface_metrics_v2"
PROTOCOL = "surface_protocol_v2"
REGION_RULE = "region_rule_v2"
NORMAL_FLOOR_SEED = 20261031
INCONCLUSIVE = "INCONCLUSIVE_REFERENCE"
CORNER_OFFSETS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], np.int64)
CELL_TETS = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))
CELL_EDGES = tuple((a, b) for a in range(8) for b in range(a+1, 8) if bin(a ^ b).count("1") == 1)


def _q(values, scale):
    return SM1._quantiles(values, scale)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0, 1)))


# ------------------------------------------------------------------------------------------------ P1 region_rule_v2

def vertex_component_from_ranges(n_vertices, component_ranges, rank=None):
    """Per-vertex mechanical component from an extractor's ``component_ranges`` (rows with ``mechanical_component``,
    ``first_vertex``, ``vertices``), mapped through ``rank`` (raw -> ranked label) when given; -1 where no row covers a
    vertex. CuttingSurface and CutAwareSurface both write these rows."""
    out = np.full(int(n_vertices), -1, np.int64)
    for row in component_ranges:
        c = int(row["mechanical_component"])
        out[row["first_vertex"]:row["first_vertex"]+row["vertices"]] = c if rank is None else int(np.asarray(rank)[c])
    return out


def classify_mesh_vertices_v2(vertices, vertex_component, positions, labels, locus, ext, *, spacing, region_max_m=None,
                              core_margin_m=None, margin_m=None):
    """``region_rule_v2``. As ``surface_metrics_v1.classify_mesh_vertices`` except that the vertex component is the
    extractor provenance ``vertex_component`` (same labelling as ``labels``; a vertex with -1 falls back to its nearest
    particle and is counted), and every vertex gets ``margin`` = (distance to the nearest sample of the next sample
    set) - (distance to the chosen set), where the sets are {exterior, cut 0, cut 1, ...} of its component; ``unambiguous``
    = margin >= ``margin_m`` (default 0.5 s). Returns the v1 keys plus ``margin``, ``unambiguous``,
    ``nearest_particle_component`` and counts."""
    from scipy.spatial import cKDTree
    s = float(spacing)
    region_max_m = 1.5*s if region_max_m is None else region_max_m
    core_margin_m = 3*s if core_margin_m is None else core_margin_m
    margin_m = .5*s if margin_m is None else margin_m
    vertices = np.asarray(vertices, float)
    labels = np.asarray(labels)
    _, nearest = cKDTree(positions).query(vertices) if len(vertices) else (None, np.zeros(0, int))
    near_comp = labels[nearest]
    component = np.asarray(vertex_component, np.int64).copy()
    fallback = component < 0
    component[fallback] = near_comp[fallback]
    side_pts = locus["side_points"].reshape(-1, 3)
    side_particle = locus["pairs"].reshape(-1)
    side_cut = np.repeat(np.asarray(locus["cut"], np.int64), 2)
    cuts = np.unique(locus["cut"]) if len(locus["cut"]) else np.zeros(0, np.int64)
    ext_particle, ext_pts, _ = ext
    n = len(vertices)
    kind = np.zeros(n, np.int8)
    row = np.full(n, -1, np.int64)
    side = np.full(n, -1, np.int8)
    dist = np.full(n, np.inf)
    dext = np.full(n, np.inf)
    margin = np.full(n, np.inf)
    for c in np.unique(component):
        vm = np.flatnonzero(component == c)
        q = vertices[vm]
        D = np.full((len(vm), 1+len(cuts)), np.inf)
        J = np.zeros((len(vm), 1+len(cuts)), np.int64)
        es = np.flatnonzero(labels[ext_particle] == c)
        if len(es):
            D[:, 0], _ = cKDTree(ext_pts[es]).query(q)
        for j, k in enumerate(cuts):
            cs = np.flatnonzero((labels[side_particle] == c) & (side_cut == k))
            if len(cs):
                D[:, 1+j], idx = cKDTree(side_pts[cs]).query(q)
                J[:, 1+j] = cs[idx]
        dc_col = 1+np.argmin(D[:, 1:], 1) if len(cuts) else np.zeros(len(vm), np.int64)
        dc = D[np.arange(len(vm)), dc_col] if len(cuts) else np.full(len(vm), np.inf)
        de = D[:, 0]
        is_cut = (dc <= de) & (dc <= region_max_m)
        is_ext = ~is_cut & (de <= region_max_m)
        ic = J[np.arange(len(vm)), dc_col] if len(cuts) else np.zeros(len(vm), np.int64)
        kind[vm[is_cut]] = 2
        kind[vm[is_ext]] = 1
        row[vm[is_cut]] = ic[is_cut]//2
        side[vm[is_cut]] = ic[is_cut] % 2
        dist[vm] = np.minimum(dc, de)
        dext[vm] = de
        Ds = np.sort(D, 1)
        margin[vm] = (Ds[:, 1]-Ds[:, 0]) if D.shape[1] > 1 else np.inf
    core = (kind == 2) & (dext >= core_margin_m)
    unambiguous = margin >= margin_m
    return {"rule": REGION_RULE, "component": component, "kind": kind, "row": row, "side": side,
            "distance_to_sample": dist, "distance_to_exterior": dext, "core": core, "margin": margin,
            "unambiguous": unambiguous, "nearest_particle_component": near_comp,
            "region_max_m": float(region_max_m), "core_margin_m": float(core_margin_m), "margin_m": float(margin_m),
            "counts": {"cut_face": int((kind == 2).sum()), "exterior": int((kind == 1).sum()),
                       "unmatched": int((kind == 0).sum()), "core": int(core.sum()), "vertices": int(n),
                       "cut_face_ambiguous": int(((kind == 2) & ~unambiguous).sum()),
                       "component_fallback_nearest_particle": int(fallback.sum()),
                       "component_differs_from_nearest_particle": int((component != near_comp).sum()),
                       "cut_face_component_differs_from_nearest_particle": int(((kind == 2) & (component != near_comp)).sum())}}


def side_keys(cls, locus, cuts, select):
    """(component, cut, side) key of the selected vertices, as ``surface_metrics_v1.side_group_keys``."""
    return (cls["component"][select]*max(int(cuts), 1)+locus["cut"][cls["row"][select]])*2+cls["side"][select]


def cut_face_signed_distances(vertices, cls, locus, labels, cuts, patch_radius_m=.002):
    """Signed distance (outward > 0) of every region_rule_v2 cut-face vertex to the local plane of the same
    (component, cut, side) side samples within ``patch_radius_m`` (the v1 distance definition). Returns
    (region vertex indices, signed distances, keys)."""
    region = np.flatnonzero(cls["kind"] == 2)
    side_pts, side_nrm = locus["side_points"].reshape(-1, 3), locus["side_normals"].reshape(-1, 3)
    keys = SM1.side_group_keys(locus, labels, cuts)
    vkey = side_keys(cls, locus, cuts, region)
    signed = np.full(len(region), np.nan)
    for key in np.unique(vkey):
        vm, sm = vkey == key, keys == key
        signed[vm], _ = SM1.local_plane_signed(np.asarray(vertices, float)[region[vm]], side_pts[sm], side_nrm[sm], patch_radius_m)
    return region, signed, vkey


def distance_rows(signed, region, cls, locus, *, spacing, select=None):
    """P1 reading of the distance / bias rows: P95 (and P50) over all cut-face vertices, max over unambiguous ones
    (margin >= 0.5 s); bias toward side 2 and thickness bias over all vertices (as v1)."""
    s = float(spacing)
    sel = np.ones(len(region), bool) if select is None else np.asarray(select, bool)
    unamb = cls["unambiguous"][region] & sel
    a = np.abs(signed)
    along = np.where(cls["side"][region] == 0, signed, -signed)
    s1, s2 = along[sel & (cls["side"][region] == 0)], along[sel & (cls["side"][region] == 1)]
    m1 = float(np.nanmean(s1)) if np.isfinite(s1).any() else None
    m2 = float(np.nanmean(s2)) if np.isfinite(s2).any() else None
    allq = _q(a[sel], s)
    unq = _q(a[unamb], s)
    return {"all": allq, "unambiguous": unq, "ambiguous_vertices": int((sel & ~cls["unambiguous"][region]).sum()),
            "p50_s": allq.get("p50_spacings"), "p95_s": allq.get("p95_spacings"), "max_unambiguous_s": unq.get("max_spacings"),
            "max_all_s": allq.get("max_spacings"),
            "bias_toward_side2_s": None if m1 is None or m2 is None else (m1+m2)/2/s,
            "thickness_bias_s": None if m1 is None or m2 is None else (m1-m2)/s}


def reference_distance_rows(vertex_reference, region, cls, locus, *, spacing, patch_radius_m=.002, select=None):
    """Reference-space rows under P1: |distance| of the vertices' reference positions (B: X_r exact; A: a pull-back
    through its P1 component) to the local plane of the Hermite P of the same cut within 2 mm; P95 over all, max over
    unambiguous vertices."""
    s = float(spacing)
    Xv = np.asarray(vertex_reference, float)
    rows = cls["row"][region]
    d = np.full(len(region), np.nan)
    for k in np.unique(locus["cut"]):
        vm = locus["cut"][rows] == k
        pm = locus["cut"] == k
        if vm.any():
            d[vm], _ = SM1.local_plane_signed(Xv[vm], locus["P"][pm], locus["N"][pm], patch_radius_m)
    sel = np.ones(len(region), bool) if select is None else np.asarray(select, bool)
    allq, unq = _q(np.abs(d[sel]), s), _q(np.abs(d[sel & cls["unambiguous"][region]]), s)
    return {"all": allq, "unambiguous": unq, "p95_s": allq.get("p95_spacings"), "max_unambiguous_s": unq.get("max_spacings"),
            "max_all_s": allq.get("max_spacings"), "distances_m": d}


# ------------------------------------------------------------------------------------------------ P3 current floor

def current_distance_floor(locus, labels, cuts, *, patch_radius_m=.002, min_points=4):
    """Leave-one-out residual of every side-locus sample: |distance| of q_i to the least-squares plane of the other
    samples of its (component, cut, side) group within ``patch_radius_m`` (NaN with < ``min_points`` neighbours or a
    nearly collinear patch, as ``local_plane_signed``). Returns (residual per side sample (2R,), group key per sample)."""
    from scipy.spatial import cKDTree
    pts = locus["side_points"].reshape(-1, 3)
    keys = SM1.side_group_keys(locus, labels, cuts)
    res = np.full(len(pts), np.nan)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        own = pts[idx]
        tree = cKDTree(own)
        for i, g in enumerate(tree.query_ball_point(own, patch_radius_m)):
            g = [j for j in g if j != i]
            if len(g) < min_points:
                continue
            local = own[g]
            centre = local.mean(0)
            _, sv, vt = np.linalg.svd(local-centre)
            if sv[1] < .2*sv[0]:
                continue
            res[idx[i]] = abs((own[i]-centre)@vt[-1])
    return res, keys


def current_p95_rows(signed, vkey, floor_residual, floor_keys, *, spacing, offset_s=.10, min_group_vertices=20,
                     select=None):
    """P3 current P95 row: per side group g with >= ``min_group_vertices`` cut-face vertices, mesh P95_g <= floor P95_g
    + ``offset_s`` (floor = locus leave-one-out residual P95 of g); groups with fewer vertices are pooled into one
    remainder whose floor is the P95 of their samples' residuals. The row passes when every evaluated group passes.
    ``pooled`` (all vertices vs all residuals) is reported for information."""
    s = float(spacing)
    sel = np.ones(len(vkey), bool) if select is None else np.asarray(select, bool)
    a = np.abs(np.asarray(signed, float))/s
    r = np.asarray(floor_residual, float)/s
    groups, small = [], []
    for key in np.unique(vkey[sel]):
        vm = sel & (vkey == key)
        (groups if vm.sum() >= min_group_vertices else small).append(key)
    rows = []

    def row(name, vm, sm):
        va, fa = a[vm], r[sm]
        va, fa = va[np.isfinite(va)], fa[np.isfinite(fa)]
        if not len(va) or not len(fa):
            return {"group": name, "vertices": int(len(va)), "floor_samples": int(len(fa)), "mesh_p95_s": None,
                    "floor_p95_s": None, "limit_s": None, "pass": None}
        mp, fp = float(np.percentile(va, 95)), float(np.percentile(fa, 95))
        return {"group": name, "vertices": int(len(va)), "floor_samples": int(len(fa)), "mesh_p95_s": mp,
                "floor_p95_s": fp, "limit_s": fp+offset_s, "pass": bool(mp <= fp+offset_s)}
    for key in groups:
        rows.append(row(int(key), sel & (vkey == key), floor_keys == key))
    if small:
        rows.append(row("pooled_small_groups", sel & np.isin(vkey, small), np.isin(floor_keys, small)))
    evaluated = [x for x in rows if x["pass"] is not None]
    used = np.isin(floor_keys, np.unique(vkey[sel]))
    return {"groups": rows, "groups_evaluated": len(evaluated), "offset_s": offset_s,
            "min_group_vertices": min_group_vertices,
            "all_groups_pass": None if not evaluated else all(x["pass"] for x in evaluated),
            "worst_excess_s": None if not evaluated else max(x["mesh_p95_s"]-x["limit_s"] for x in evaluated),
            "pooled": row("pooled", sel, used)}


# ------------------------------------------------------------------------------------------------ P2 matched normal

def matched_scale_normal(vertices, faces, cls, locus, labels, cuts, *, normal_radius_m=.003):
    """P2: per core cut-face triangle (all three vertices core, key of its first vertex, as v1), the unsigned angle
    between the least-squares plane of the mesh's own cut-face vertices of that (component, cut, side) within
    ``normal_radius_m`` of the centroid and the plane of the same-key locus side samples within the same radius.
    Returns (angles deg (T,), cut digit per triangle, triangle indices)."""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    side_pts, side_nrm = locus["side_points"].reshape(-1, 3), locus["side_normals"].reshape(-1, 3)
    keys = SM1.side_group_keys(locus, labels, cuts)
    tri = np.flatnonzero(cls["core"][faces].all(1)) if len(faces) else np.zeros(0, np.int64)
    angles = np.full(len(tri), np.nan)
    if not len(tri):
        return angles, np.zeros(0, np.int64), tri
    centroid = vertices[faces[tri]].mean(1)
    tkey = side_keys(cls, locus, cuts, faces[tri, 0])
    region = np.flatnonzero(cls["kind"] == 2)
    vkey = side_keys(cls, locus, cuts, region)
    for key in np.unique(tkey):
        tm = np.flatnonzero(tkey == key)
        sm = keys == key
        mv = vertices[region[vkey == key]]
        if not sm.any() or len(mv) < 4:
            continue
        _, pl = SM1.local_plane_signed(centroid[tm], side_pts[sm], side_nrm[sm], normal_radius_m)
        _, pm = SM1.local_plane_signed(centroid[tm], mv, np.tile(side_nrm[sm].mean(0), (len(mv), 1)), normal_radius_m)
        ok = np.isfinite(pl[:, 0]) & np.isfinite(pm[:, 0])
        angles[tm[ok]] = _angle(pl[ok], pm[ok])
    return angles, locus["cut"][cls["row"][faces[tri, 0]]], tri


def normal_floor_split_halves(locus, labels, cuts, ext, *, normal_radius_m=.003, core_margin_m, seed=NORMAL_FLOOR_SEED):
    """P2 per-state floor: the side samples of every (component, cut, side) group are split into two disjoint random
    halves (``numpy.random.default_rng(seed)``, groups in ascending key order); at every core sample (>= ``core_margin_m``
    from the exterior samples) the unsigned angle between the planes of the two halves within ``normal_radius_m``.
    Returns (angles deg, cut digit per angle)."""
    from scipy.spatial import cKDTree
    pts, nrm = locus["side_points"].reshape(-1, 3), locus["side_normals"].reshape(-1, 3)
    keys = SM1.side_group_keys(locus, labels, cuts)
    cut = np.repeat(np.asarray(locus["cut"], np.int64), 2)
    dext = cKDTree(ext[1]).query(pts)[0] if len(ext[1]) else np.full(len(pts), np.inf)
    rng = np.random.default_rng(seed)
    out, outcut = [], []
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        perm = rng.permutation(idx)
        h1, h2 = perm[:len(perm)//2], perm[len(perm)//2:]
        q = idx[dext[idx] >= core_margin_m]
        if len(h1) < 4 or len(h2) < 4 or not len(q):
            continue
        _, p1 = SM1.local_plane_signed(pts[q], pts[h1], nrm[h1], normal_radius_m)
        _, p2 = SM1.local_plane_signed(pts[q], pts[h2], nrm[h2], normal_radius_m)
        ok = np.isfinite(p1[:, 0]) & np.isfinite(p2[:, 0])
        out.append(_angle(p1[ok], p2[ok]))
        outcut.append(cut[q[ok]])
    if not out:
        return np.zeros(0), np.zeros(0, np.int64)
    return np.concatenate(out), np.concatenate(outcut)


def normal_row(angles, floor_angles, *, band_deg=1.):
    """P2 decision row: mesh matched-scale P95 <= floor P95 + ``band_deg``."""
    a, f = np.asarray(angles, float), np.asarray(floor_angles, float)
    a, f = a[np.isfinite(a)], f[np.isfinite(f)]
    if not len(a) or not len(f):
        return {"mesh_p50_deg": None, "mesh_p95_deg": None, "floor_p95_deg": None, "limit_deg": None, "pass": None,
                "triangles": int(len(a)), "floor_samples": int(len(f))}
    mp, fp = float(np.percentile(a, 95)), float(np.percentile(f, 95))
    return {"mesh_p50_deg": float(np.percentile(a, 50)), "mesh_p95_deg": mp, "floor_p50_deg": float(np.percentile(f, 50)),
            "floor_p95_deg": fp, "limit_deg": fp+band_deg, "pass": bool(mp <= fp+band_deg),
            "triangles": int(len(a)), "floor_samples": int(len(f))}


# ------------------------------------------------------------------------------------------------ P4 volume reference

def lattice_cells(lattice):
    """(M, 8) particle indices of every complete reference-lattice cell (corner order ``CORNER_OFFSETS``)."""
    lattice = np.asarray(lattice, np.int64)
    low = lattice.min(0)
    dims = lattice.max(0)-low+2
    index = np.full(int(np.prod(dims)), -1, np.int64)
    index[np.ravel_multi_index((lattice-low).T, dims)] = np.arange(len(lattice))
    corners = np.stack([index[np.ravel_multi_index((lattice-low+o).T, dims)] for o in CORNER_OFFSETS], 1)
    return corners[(corners >= 0).all(1)]


def reference_consistency(lattice, positions, F, codes, labels, particle_volume, *, spacing, cuts=1, encoding="packed2",
                          min_cells=8, mean_tolerance=.01, low_ratio=.9, low_fraction=.05, unverified_volume_fraction=.01):
    """P4 precondition. Fully linked cells = complete lattice cells none of whose 12 edges is a conflicting link (any
    cut). Per cell r = V(6 tetrahedra on the current particle positions) / (s^3 * mean corner det F); a cell belongs to
    its corners' component (cells spanning components are skipped and counted). Per component: ``consistent`` when
    |mean r - 1| <= ``mean_tolerance`` and the fraction of cells with r < ``low_ratio`` <= ``low_fraction``;
    ``unverified`` when it has < ``min_cells`` cells. ``status``: INCONSISTENT if any verified component is inconsistent,
    UNVERIFIED if the unverified components hold > ``unverified_volume_fraction`` of the reference volume, otherwise
    CONSISTENT."""
    s = float(spacing)
    positions = np.asarray(positions, float)
    labels = np.asarray(labels)
    cells = lattice_cells(lattice)
    digits = SM1._digits(codes, encoding, int(cuts))
    ok = np.ones(len(cells), bool)
    for a, b in CELL_EDGES:
        da, db = digits[cells[:, a]], digits[cells[:, b]]
        ok &= ~(((da == 1) & (db == 2)) | ((da == 2) & (db == 1))).any(1)
    same = (labels[cells] == labels[cells[:, :1]]).all(1)
    spanning = int((ok & ~same).sum())
    c = cells[ok & same]
    V = np.zeros(len(c))
    for a, b, d, e in CELL_TETS:
        V += np.einsum("ij,ij->i", positions[c[:, b]]-positions[c[:, a]],
                       np.cross(positions[c[:, d]]-positions[c[:, a]], positions[c[:, e]]-positions[c[:, a]]))/6
    J = np.linalg.det(np.asarray(F, float))
    r = np.abs(V)/s**3/J[c].mean(1) if len(c) else np.zeros(0)
    cl = labels[c[:, 0]] if len(c) else np.zeros(0, labels.dtype)
    vol = np.bincount(labels, weights=np.broadcast_to(np.asarray(particle_volume, float), labels.shape))
    comps, unverified_volume = {}, 0.
    for k in np.unique(labels):
        m = cl == k
        if m.sum() < min_cells:
            comps[str(int(k))] = {"cells": int(m.sum()), "status": "UNVERIFIED", "reference_volume_m3": float(vol[k])}
            unverified_volume += float(vol[k])
            continue
        mean, low = float(r[m].mean()), float((r[m] < low_ratio).mean())
        comps[str(int(k))] = {"cells": int(m.sum()), "mean_V_over_J": mean, "fraction_below_low_ratio": low,
                              "p5_V_over_J": float(np.percentile(r[m], 5)),
                              "status": "CONSISTENT" if abs(mean-1) <= mean_tolerance and low <= low_fraction else "INCONSISTENT",
                              "reference_volume_m3": float(vol[k])}
    inconsistent = [k for k, v in comps.items() if v["status"] == "INCONSISTENT"]
    unverified = unverified_volume/max(float(vol.sum()), 1e-300)
    status = "INCONSISTENT" if inconsistent else ("UNVERIFIED" if unverified > unverified_volume_fraction else "CONSISTENT")
    return {"status": status, "components": comps, "inconsistent_components": inconsistent,
            "unverified_volume_fraction": unverified, "cells": int(len(c)), "cells_spanning_components": spanning,
            "mean_V_over_J_all": float(r.mean()) if len(r) else None,
            "rule": {"mean_tolerance": mean_tolerance, "low_ratio": low_ratio, "low_fraction": low_fraction,
                     "min_cells": min_cells, "unverified_volume_fraction": unverified_volume_fraction}}


def component_status(consistency, components):
    """Precondition status restricted to ``components``: INCONSISTENT / UNVERIFIED (any of them) / CONSISTENT."""
    rows = [consistency["components"].get(str(int(k)), {"status": "UNVERIFIED"}) for k in components]
    if any(r["status"] == "INCONSISTENT" for r in rows):
        return "INCONSISTENT"
    if any(r["status"] == "UNVERIFIED" for r in rows):
        return "UNVERIFIED"
    return "CONSISTENT"


def volume_current_row(rel_error, status, limit):
    """P4 row: |relative current-volume error| <= ``limit`` decides only when the reference precondition ``status`` is
    CONSISTENT; otherwise the verdict is INCONCLUSIVE_REFERENCE (never PASS)."""
    value = None if rel_error is None else abs(float(rel_error))
    if status != "CONSISTENT":
        return {"value": value, "op": "<=", "threshold": limit, "verdict": INCONCLUSIVE, "reference_status": status}
    if value is None or not np.isfinite(value):
        return {"value": None, "op": "<=", "threshold": limit, "verdict": "NOT_EVALUABLE", "reference_status": status}
    return {"value": value, "op": "<=", "threshold": limit, "verdict": "PASS" if value <= limit else "FAIL",
            "reference_status": status}


# ------------------------------------------------------------------------------------------------ P5 slow motion

def slow_frame_pairs(frames, *, spacing, max_displacement_p95_s=.10):
    """Consecutive frame pairs (i, i+1) without a code change whose material displacement P95 <= the limit; ``frames``
    are dicts with ``positions`` and ``codes``. Returns (eligible pairs, all pair rows)."""
    rows, ok = [], []
    for i in range(len(frames)-1):
        a, b = frames[i], frames[i+1]
        same = bool(np.array_equal(a["codes"], b["codes"]))
        disp = float(np.percentile(np.linalg.norm(np.asarray(b["positions"])-np.asarray(a["positions"]), axis=1), 95)/spacing)
        eligible = same and disp <= max_displacement_p95_s
        rows.append({"pair": (i, i+1), "codes_equal": same, "material_displacement_p95_s": disp, "eligible": eligible})
        if eligible:
            ok.append((i, i+1))
    return ok, rows


def transport_idw(vertices, vertex_component, positions_a, positions_b, labels, *, k=8, power=2., eps_m=1e-6):
    """Vertex displacement by inverse-distance weighting (weights 1/max(d, eps)^power) of the displacement of its
    ``k`` nearest particles of its own (provenance) component at frame a."""
    from scipy.spatial import cKDTree
    vertices = np.asarray(vertices, float)
    pa, pb = np.asarray(positions_a, float), np.asarray(positions_b, float)
    labels = np.asarray(labels)
    out = vertices.copy()
    comp = np.asarray(vertex_component)
    for c in np.unique(comp):
        vm = np.flatnonzero(comp == c)
        own = np.flatnonzero(labels == c)
        if not len(own):
            continue
        kk = min(k, len(own))
        d, j = cKDTree(pa[own]).query(vertices[vm], k=kk)
        d, j = d.reshape(len(vm), kk), j.reshape(len(vm), kk)
        w = 1/np.maximum(d, eps_m)**power
        disp = pb[own[j]]-pa[own[j]]
        out[vm] = vertices[vm]+(w[:, :, None]*disp).sum(1)/w.sum(1, keepdims=True)
    return out


def temporal_stability_v2(vertices_a, faces_a, vertices_b, faces_b, positions_a, positions_b, vertex_component_a, labels,
                          *, spacing, k=8, power=2.):
    """P5: symmetric exact point-triangle residual between mesh A transported by ``transport_idw`` and mesh B (the v1
    residual with the declared interpolating transport)."""
    va, vb = np.asarray(vertices_a, float), np.asarray(vertices_b, float)
    fa, fb = np.asarray(faces_a, np.int64), np.asarray(faces_b, np.int64)
    moved = transport_idw(va, vertex_component_a, positions_a, positions_b, labels, k=k, power=power)
    ab = SM1.point_mesh_distance(vb, moved, fa)
    ba = SM1.point_mesh_distance(moved, vb, fb)
    return {"same_topology": bool(fa.shape == fb.shape and np.array_equal(fa, fb)),
            "residual_after_transport": _q(np.r_[ab, ba], spacing),
            "material_displacement": _q(np.linalg.norm(np.asarray(positions_b)-np.asarray(positions_a), axis=1), spacing),
            "transport": f"IDW k={k} power={power} same-component particles"}


# ------------------------------------------------------------------------------------------------ P6 intersections

def intersecting_pairs(vertices, faces, *, max_triangles=20000, seed=0):
    """(P, 2) intersecting triangle pairs found by ``surface_metrics_v1.self_intersections``' test (same sample of up
    to ``max_triangles`` triangles, same seed; segment-triangle crossing among vertex-disjoint neighbours), so
    ``len(pairs)`` equals its ``intersecting_pairs_sampled``."""
    from scipy.spatial import cKDTree
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    if len(faces) < 2:
        return np.zeros((0, 2), np.int64)
    tri = vertices[faces]
    centroid = tri.mean(1)
    radius = np.linalg.norm(tri-centroid[:, None], axis=2).max(1)
    rng = np.random.default_rng(seed)
    sample = np.arange(len(faces)) if len(faces) <= max_triangles else np.sort(rng.choice(len(faces), max_triangles, replace=False))
    cand = cKDTree(centroid).query_ball_point(centroid[sample], 2*radius.max())
    ia = np.repeat(sample, [len(g) for g in cand])
    ib = np.concatenate([np.asarray(g, np.int64) for g in cand]) if len(cand) else np.zeros(0, np.int64)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    share = (faces[ia][:, :, None] == faces[ib][:, None, :]).any((1, 2))
    ia, ib = ia[~share], ib[~share]
    close = np.linalg.norm(centroid[ia]-centroid[ib], axis=1) <= radius[ia]+radius[ib]
    ia, ib = ia[close], ib[close]

    def crosses(seg0, seg1, A, B, C, eps=1e-12):
        e1, e2, d = B-A, C-A, seg1-seg0
        h = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > eps*np.linalg.norm(e1, axis=1)*np.linalg.norm(e2, axis=1)*np.linalg.norm(d, axis=1)+1e-300
        f = np.where(ok, 1/np.where(ok, det, 1), 0)
        s = seg0-A
        u = f*np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = f*np.einsum("ij,ij->i", d, q)
        t = f*np.einsum("ij,ij->i", e2, q)
        tol = 1e-9
        return ok & (u > tol) & (v > tol) & (u+v < 1-tol) & (t > tol) & (t < 1-tol)

    hit = np.zeros(len(ia), bool)
    for (x, y) in ((ia, ib), (ib, ia)):
        T, Sg = tri[y], tri[x]
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            hit |= crosses(Sg[:, e0], Sg[:, e1], T[:, 0], T[:, 1], T[:, 2])
    return np.unique(np.sort(np.c_[ia[hit], ib[hit]], 1), axis=0) if hit.any() else np.zeros((0, 2), np.int64)


def _face_class(faces, cls, locus):
    """Per triangle: cut-face (>= 2 vertices cut-face), its (cut, side) from its first cut-face vertex."""
    kind = cls["kind"][faces]
    is_cut = (kind == 2).sum(1) >= 2
    first = np.argmax(kind == 2, 1)
    v = faces[np.arange(len(faces)), first]
    lcut = np.asarray(locus["cut"], np.int64)
    row = cls["row"][v]
    cut = np.full(len(faces), -1, np.int64)
    ok = is_cut & (row >= 0) & (row < len(lcut))
    cut[ok] = lcut[row[ok]]
    side = np.where(is_cut, cls["side"][v], -1)
    return is_cut, cut, side


def attribute_intersections(vertices, faces, pairs, face_component, cls, positions, labels, codes, locus, *, spacing,
                            cuts=1, encoding="packed2", gap_m=None, crossing_radius_m=None):
    """P6: split intersecting pairs into *physical* and *unexplained*. Candidate physical pairs: triangles of different
    mechanical components (provenance), or cut-face triangles of opposite sides of one cut digit in one component. A
    candidate is physical when, at the pair midpoint m, the particle gap between the two sides' particle sets is < 1 s
    (``gap_m``; gap = min over the two directions of the distance from the set's particle nearest m to the other set) or
    a locus row joining the two sides within ``crossing_radius_m`` (2 s) of m has crossed sides ((q2 - q1) . N < 0).
    Every other pair, including every other pair within one component (folds), is unexplained."""
    from scipy.spatial import cKDTree
    s = float(spacing)
    gap_m = s if gap_m is None else gap_m
    crossing_radius_m = 2*s if crossing_radius_m is None else crossing_radius_m
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    positions, labels = np.asarray(positions, float), np.asarray(labels)
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    fc = np.asarray(face_component, np.int64)
    is_cut, fcut, fside = _face_class(faces, cls, locus)
    digits = SM1._digits(codes, encoding, int(cuts))
    centroid = vertices[faces].mean(1)
    lp = locus["pairs"]
    lcomp = labels[lp] if len(lp) else np.zeros((0, 2), labels.dtype)
    sp = locus["side_points"]
    opening = np.einsum("ij,ij->i", sp[:, 1]-sp[:, 0], locus["N"]) if len(lp) else np.zeros(0)
    lmid = sp.mean(1) if len(lp) else np.zeros((0, 3))
    trees = {}

    def tree(key, mask):
        if key not in trees:
            idx = np.flatnonzero(mask)
            trees[key] = (cKDTree(positions[idx]) if len(idx) else None, idx)
        return trees[key]

    counts = {"physical_different_pieces": 0, "physical_opposite_sides": 0, "unexplained_different_pieces_apart": 0,
              "unexplained_opposite_sides_apart": 0, "unexplained_same_component": 0}
    gaps = []
    for a, b in pairs:
        ca, cb = int(fc[a]), int(fc[b])
        if ca != cb:
            kind = "different_pieces"
            ta, tb = tree(("c", ca), labels == ca), tree(("c", cb), labels == cb)
            rows = ((lcomp[:, 0] == ca) & (lcomp[:, 1] == cb)) | ((lcomp[:, 0] == cb) & (lcomp[:, 1] == ca))
        elif is_cut[a] and is_cut[b] and fcut[a] == fcut[b] and fcut[a] >= 0 and fside[a] != fside[b]:
            kind, k = "opposite_sides", int(fcut[a])
            ta = tree(("s", ca, k, 1), (labels == ca) & (digits[:, k] == 1))
            tb = tree(("s", ca, k, 2), (labels == ca) & (digits[:, k] == 2))
            rows = (locus["cut"] == k) & (lcomp[:, 0] == ca) & (lcomp[:, 1] == ca)
        else:
            counts["unexplained_same_component"] += 1
            continue
        m = (centroid[a]+centroid[b])/2
        gap = np.inf
        if ta[0] is not None and tb[0] is not None:
            _, ia = ta[0].query(m)
            _, ib = tb[0].query(m)
            gap = min(tb[0].query(positions[ta[1][ia]])[0], ta[0].query(positions[tb[1][ib]])[0])
        crossed = bool(rows.any() and np.any((opening[rows] < 0) & (np.linalg.norm(lmid[rows]-m, axis=1) <= crossing_radius_m)))
        gaps.append(gap)
        if gap < gap_m or crossed:
            counts[f"physical_{kind}"] += 1
        else:
            counts[f"unexplained_{kind}_apart"] += 1
    unexplained = sum(v for k, v in counts.items() if k.startswith("unexplained"))
    return {"pairs": int(len(pairs)), **counts, "physical": int(len(pairs)-unexplained), "unexplained": int(unexplained),
            "candidate_gap_s": _q(np.asarray(gaps)[np.isfinite(gaps)] if gaps else np.zeros(0), s),
            "gap_m": gap_m, "crossing_radius_m": crossing_radius_m}


def faces_nearer_other_component_v2(vertices, faces, positions, labels, face_component, *, spacing, gap_m=None):
    """P6 reading of ``faces_nearer_other_component``: triangles whose first vertex is nearer to another component's
    particle than to its own (the v1 count), split into *physical* (the particle gap between the own component's particle
    nearest the vertex and that other component is < 1 s) and *unexplained*."""
    from scipy.spatial import cKDTree
    s = float(spacing)
    gap_m = s if gap_m is None else gap_m
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    positions, labels = np.asarray(positions, float), np.asarray(labels)
    fc = np.asarray(face_component, np.int64)
    if not len(faces):
        return {"faces_nearer_other_component": 0, "physical": 0, "unexplained": 0}
    v0 = vertices[faces[:, 0]]
    _, near = cKDTree(positions).query(v0)
    other = labels[near] != fc
    trees = {}

    def tree(c):
        if c not in trees:
            idx = np.flatnonzero(labels == c)
            trees[c] = (cKDTree(positions[idx]) if len(idx) else None, idx)
        return trees[c]
    physical = 0
    for i in np.flatnonzero(other):
        own, oth = tree(int(fc[i])), tree(int(labels[near[i]]))
        if own[0] is None or oth[0] is None:
            continue
        _, j = own[0].query(v0[i])
        physical += oth[0].query(positions[own[1][j]])[0] < gap_m
    return {"faces_nearer_other_component": int(other.sum()), "physical": int(physical),
            "unexplained": int(other.sum()-physical), "gap_m": gap_m}


# ------------------------------------------------------------------------------------------------ P7 chip accounting

def chip_volume_v2(vertices, faces, face_component, labels, particle_volume, F, chip_components, *, limit, status,
                   fallback_faces=None, fallback_particles=None):
    """P7: chip current-volume row on the lattice-faced part only: |V(chip faces not in ``fallback_faces``) -
    sum (m/rho) det F (chip particles not in ``fallback_particles``)| / that sum <= ``limit``, with the P4 precondition
    ``status`` of the chip components. The fallback (particle-cell) part is reported separately as by construction.
    Methods without a fallback pass ``None`` (the whole chip is lattice-faced)."""
    tri = np.asarray(vertices, float)[np.asarray(faces, np.int64)]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))/6 if len(tri) else np.zeros(0)
    fc = np.asarray(face_component, np.int64)
    labels = np.asarray(labels)
    chip_f = np.isin(fc, chip_components)
    chip_p = np.isin(labels, chip_components)
    ff = np.zeros(len(fc), bool) if fallback_faces is None else np.asarray(fallback_faces, bool)
    fp = np.zeros(len(labels), bool) if fallback_particles is None else np.asarray(fallback_particles, bool)
    vcur = np.broadcast_to(np.asarray(particle_volume, float), labels.shape)*np.linalg.det(np.asarray(F, float))
    mesh_l, mat_l = float(signed[chip_f & ~ff].sum()), float(vcur[chip_p & ~fp].sum())
    rel = (mesh_l-mat_l)/mat_l if mat_l > 0 else None
    row = volume_current_row(rel, status, limit)
    row["lattice_faced"] = {"mesh_m3": mesh_l, "material_current_m3": mat_l, "faces": int((chip_f & ~ff).sum()),
                            "particles": int((chip_p & ~fp).sum())}
    row["particle_cell_fallback_by_construction"] = {"mesh_m3": float(signed[chip_f & ff].sum()),
                                                     "material_current_m3": float(vcur[chip_p & fp].sum()),
                                                     "faces": int((chip_f & ff).sum()), "particles": int((chip_p & fp).sum()),
                                                     "note": "volume matches sum (m/rho) det F by construction; never reconstruction quality"}
    return row


# ------------------------------------------------------------------------------------------------ comparison metrics

GATE_METRICS_V2 = {   # name: (direction, band kind); v1 GATE_METRICS with the P1-P7 replacements (protocol v2)
    "current_p50_s": ("lower", "distance"), "current_p95_s": ("lower", "distance"),
    "current_max_unambiguous_s": ("lower", "distance"), "current_core_p95_s": ("lower", "distance"),
    "reference_p95_s": ("lower", "distance"), "reference_max_unambiguous_s": ("lower", "distance"),
    "locus_to_mesh_p95_s": ("lower", "distance"), "coverage": ("higher", "spread"),
    "abs_bias_toward_side2_s": ("lower", "distance"), "abs_thickness_bias_s": ("lower", "distance"),
    "matched_normal_p50_deg": ("lower", "normal"), "matched_normal_p95_deg": ("lower", "normal"),
    "rim_corner_p95_s": ("lower", "distance"), "rim_corner_max_s": ("lower", "distance"),
    "open_edges_near_junction": ("lower", "spread"), "boundary_edges": ("lower", "spread"),
    "non_manifold_edges": ("lower", "spread"), "inconsistently_oriented_edges": ("lower", "spread"),
    "degenerate_triangles": ("lower", "spread"), "shells_negative_volume": ("lower", "spread"),
    "unexplained_intersections_sampled": ("lower", "spread"), "closed": ("true", "spread"),
    "shells_equal_components": ("true", "spread"), "shells_spanning_components": ("lower", "spread"),
    "components_without_shell": ("lower", "spread"), "faces_nearer_other_component_unexplained": ("lower", "spread"),
    "volume_current_abs_rel_if_consistent": ("lower", "volume"), "volume_reference_abs_rel": ("lower", "volume"),
    "fixed_state_topology_stable": ("true", "spread"), "fixed_state_residual_s": ("lower", "distance"),
}
SLOW_MOTION_METRIC_V2 = {"slow_motion_residual_p95_s_eligible_pairs": ("lower", "distance")}
REPORT_ONLY_V2 = ("triangle_normal_p50_deg", "triangle_normal_p95_deg", "self_intersections_sampled",
                  "faces_nearer_other_component", "volume_current_abs_rel", "roughness_mesh_core_qm_s",
                  "roughness_locus_core_qm_s", "current_max_all_s", "reference_max_all_s")
