"""Cut-surface geometry metrics measured against the material state (WP-12 Phase 0 draft + Phase 1 v1, read-only).

Phase 1 (final ``surface_metrics_v1``) adds ``material_locus`` (Hermite points from ``cut_events_v1`` when
available, labelled link-midpoint fallback otherwise), the boundary-sample region rule ``region_rule_v1``
(``classify_mesh_vertices``), signed/normal/roughness/coverage/rim/volume/topology/temporal metrics and the
umbrella ``surface_report`` in current space and (approximate pull-back) material-reference space. The Phase 0
draft ``cut_face_metrics`` (nearest-particle region rule) is kept unchanged so the Phase 0 numbers reproduce.

Version ``surface_metrics_v1``. Pure NumPy/SciPy: no Newton, Warp or renderer import. Nothing here
changes particles, labels, meshes or any evaluation; it only measures a reconstructed mesh against the
cut history the solver actually wrote.

Material cut locus (the reference that the mesh is compared with)
---------------------------------------------------------------
The particles sit on an integer reference lattice (``lattice``, one (i, j, k) per particle, as returned by
``task.sample_particles()``). A 6-neighbour lattice link (a, b) is *cut* when the two codes are on opposite
sides of some cut digit (``cut_topology.conflicting`` for ternary codes, ``conflicting_packed`` for packed
codes). For every cut link this module forms

* the shared locus point ``m = (x_a + x_b)/2`` (midpoint of the link in current positions), and
* two side locus points ``q_a = x_a + F_a (X_b - X_a)/2`` and ``q_b = x_b + F_b (X_a - X_b)/2``: the reference
  link midpoint mapped into current space through each endpoint's deformation (F = identity when no
  deformation gradient is given, i.e. a translation-only map). When the two pieces have separated, ``m``
  lies in the gap while ``q_a``/``q_b`` stay on their own piece; both are reported.

The signed link direction ``s = (X_b - X_a)/|X_b - X_a|`` is oriented from the lower to the higher digit of
the first conflicting cut, so a patch mean of ``s`` over a planar cut estimates its unit normal (the number
of axis links crossing a plane per unit area is proportional to |n_axis|).

Metrics (all lengths in metres, angles in degrees)
-------------------------------------------------
* ``cut_face_distance``: P50/P95/max distance from cut-face mesh vertices to the side locus of the same
  mechanical component and to the shared midpoints, each both to the nearest sample (quantised by the lattice)
  and to the local least-squares plane of the samples within ``patch_radius_m`` (primary). A mesh vertex belongs to the
  cut face when its nearest particle (current positions) is an endpoint of a cut link and that particle is not
  on the original exterior surface (a particle with a missing 6-neighbour in the reference lattice).
* ``normal_deviation``: angle between each cut-face triangle normal and the local link direction (mean of the
  signed link directions of the side locus points within ``normal_radius_m``, default 3 mm, of the triangle
  centroid), unsigned
  (|cos|), since ParticleSurface winding is not part of this check.
* ``*_core``: the same distance/normal numbers restricted to cut-face vertices/triangles whose nearest particles are
  more than ``core_exclusion_layers`` (default 3) lattice steps from the original exterior surface, i.e. away from
  the face edges where the kernel isosurface rounds the corner.
* ``roughness``: RMS residual of a least-squares plane over each ``patch_radius_m`` neighbourhood
  (default 1.5 mm, i.e. 3 mm patches), for the cut-face mesh vertices and, separately, for the side locus
  points themselves, each patch kept inside one mechanical component. The locus value is the material-state roughness; the mesh minus locus difference is what
  reconstruction adds.
* ``mesh_validity``: degenerate triangles (repeated index or area below ``area_eps``), non-manifold edges
  (more than two triangles), boundary edges (one triangle), inconsistently oriented interior edges, connected
  components, unreferenced vertices.

Limits (stated so results are not over-read): the locus is quantised to the lattice (a planar cut is a
staircase of link midpoints, so even an ideal reconstruction scores up to ~spacing/2 against it on oblique
cuts); the side map uses one particle's F (a first-order map, exact for affine motion); the cut-face region
is chosen by the nearest particle, so a vertex of an exterior face near a cut edge can be included; a cut
that is still in contact puts both sides' locus points at the same place. These are measurement tools for
the Phase 1 diagnosis, not acceptance thresholds. Measured floor of the method on an ideal mesh lying exactly on
a 30 deg oblique plane through a 1 mm lattice (tests/test_surface_metrics_v1.py): local-plane distance P95 about
0.2 mm, normal deviation P50 about 2 deg (P95 about 7 deg), side-locus roughness P95 about 0.18 mm; axis-aligned
planes score 0.
"""
import numpy as np

VERSION = "surface_metrics_v1"
_AXES = np.eye(3, dtype=np.int64)


def _digits(codes, encoding, cuts):
    codes = np.asarray(codes, np.int64)
    if encoding == "packed2":
        return ((codes[:, None] >> (2*np.arange(cuts))) & 3).astype(np.int8)
    if encoding == "ternary":
        return ((codes[:, None] // 3**np.arange(cuts)) % 3).astype(np.int8)
    raise ValueError("encoding is 'ternary' or 'packed2'")


def lattice_links(lattice):
    """(L, 2) particle index pairs of all +x/+y/+z 6-neighbour links and (L,) axis ids."""
    lattice = np.asarray(lattice, np.int64)
    if lattice.ndim != 2 or lattice.shape[1] != 3:
        raise ValueError("lattice must be (N, 3) integer reference indices")
    low = lattice.min(0)
    dims = lattice.max(0)-low+2
    keys = np.ravel_multi_index((lattice-low).T, dims)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise ValueError("Duplicate lattice indices")
    pairs, axes = [], []
    for axis in range(3):
        neighbour = np.ravel_multi_index((lattice-low+_AXES[axis]).T, dims)
        position = np.searchsorted(sorted_keys, neighbour)
        position = np.minimum(position, len(sorted_keys)-1)
        found = sorted_keys[position] == neighbour
        a = np.flatnonzero(found)
        pairs.append(np.c_[a, order[position[found]]])
        axes.append(np.full(len(a), axis))
    return np.vstack(pairs), np.concatenate(axes)


def exterior_mask(lattice, layers=1):
    """Particles within ``layers`` lattice steps of the original surface (layer 1: fewer than six 6-neighbours)."""
    pairs, _ = lattice_links(lattice)
    count = np.bincount(pairs.ravel(), minlength=len(lattice))
    mask = count < 6
    for _ in range(int(layers)-1):
        grown = mask.copy()
        grown[pairs[mask[pairs[:, 0]], 1]] = True
        grown[pairs[mask[pairs[:, 1]], 0]] = True
        mask = grown
    return mask


def cut_locus(reference, lattice, positions, codes, *, encoding="ternary", cuts=1, F=None):
    """Cut links and their current-space locus (see module docstring).

    Returns a dict with ``pairs`` (C, 2), ``cut_digit`` (C,), ``midpoint`` (C, 3), ``side_points`` (C, 2, 3),
    ``direction`` (C, 3) signed reference link directions, ``direction_current`` (the same mapped by the mean F
    of the two endpoints, or the reference direction without F) and ``links_total``.
    """
    reference = np.asarray(reference, float)
    positions = np.asarray(positions, float)
    if reference.shape != positions.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and positions must both be (N, 3)")
    pairs, _ = lattice_links(lattice)
    digits = _digits(codes, encoding, int(cuts))
    da, db = digits[pairs[:, 0]], digits[pairs[:, 1]]
    opposite = ((da == 1) & (db == 2)) | ((da == 2) & (db == 1))
    cut = opposite.any(1)
    pairs, da, db, opposite = pairs[cut], da[cut], db[cut], opposite[cut]
    first = np.argmax(opposite, 1) if len(pairs) else np.zeros(0, int)
    a, b = pairs[:, 0], pairs[:, 1]
    sign = np.where(da[np.arange(len(a)), first] < db[np.arange(len(a)), first], 1., -1.)
    delta = reference[b]-reference[a]
    direction = sign[:, None]*delta/np.linalg.norm(delta, axis=1, keepdims=True)
    if F is None:
        qa, qb = positions[a]+delta/2, positions[b]-delta/2
        current = direction
    else:
        F = np.asarray(F, float)
        qa = positions[a]+np.einsum("nij,nj->ni", F[a], delta/2)
        qb = positions[b]-np.einsum("nij,nj->ni", F[b], delta/2)
        current = np.einsum("nij,nj->ni", (F[a]+F[b])/2, direction)
        current /= np.linalg.norm(current, axis=1, keepdims=True)
    return {"pairs": pairs, "cut_digit": first, "midpoint": (positions[a]+positions[b])/2,
            "side_points": np.stack([qa, qb], 1), "direction": direction, "direction_current": current,
            "links_total": int(cut.size),
            "deformation_map": "identity" if F is None else "F"}


def _finite(values):
    values = np.asarray(values, float)
    return values[np.isfinite(values)]


def _percentiles(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"count": 0, "p50": None, "p95": None, "max": None, "mean": None}
    return {"count": int(len(values)), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max()), "mean": float(values.mean())}


def plane_fit_rms(points):
    """RMS distance of points to their least-squares plane (needs >= 3 points)."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return np.nan
    centred = points-points.mean(0)
    singular = np.linalg.svd(centred, compute_uv=False)
    return float(singular[-1]/np.sqrt(len(points)))


def local_plane_distance(queries, points, radius_m, *, min_points=4):
    """Distance from each query to the least-squares plane of the ``points`` within ``radius_m`` of it.

    The locus is a sampled staircase, so nearest-sample distances are quantised (up to ~spacing/sqrt(2) for a
    query between samples on an ideal face); the local plane removes that sampling term. NaN where fewer
    than ``min_points`` samples are in range."""
    from scipy.spatial import cKDTree
    queries = np.asarray(queries, float)
    points = np.asarray(points, float)
    out = np.full(len(queries), np.nan)
    if len(points) < min_points or not len(queries):
        return out
    tree = cKDTree(points)
    for i, group in enumerate(tree.query_ball_point(queries, radius_m)):
        if len(group) >= min_points:
            local = points[group]
            centre = local.mean(0)
            normal = np.linalg.svd(local-centre)[2][-1]
            out[i] = abs((queries[i]-centre)@normal)
    return out


def patch_roughness(points, radius_m, *, groups=None, min_points=6, max_centres=4000, seed=0):
    """Per-centre RMS plane residual over ``radius_m`` neighbourhoods (P50/P95/max summary).

    ``groups`` (optional, one label per point, e.g. the mechanical component of each side point) keeps every
    patch inside one group, so two faces of a narrow gap are never fitted as one plane; values are pooled."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    labels = np.zeros(len(points), int) if groups is None else np.asarray(groups)
    values, sampled = [], 0
    rng = np.random.default_rng(seed)
    budget = max_centres/max(len(points), 1)
    for label in np.unique(labels):
        own = points[labels == label]
        if len(own) < min_points:
            continue
        centres = own
        if budget < 1:
            centres = own[rng.random(len(own)) < budget]
        sampled += len(centres)
        tree = cKDTree(own)
        for group in tree.query_ball_point(centres, radius_m):
            if len(group) >= min_points:
                values.append(plane_fit_rms(own[group]))
    values = np.asarray(values, float)
    return {**_percentiles(values), "quadratic_mean": float(np.sqrt(np.mean(values**2))) if len(values) else None,
            "radius_m": float(radius_m), "min_points": int(min_points),
            "centres_sampled": int(sampled), "grouped": groups is not None}


def mesh_validity(vertices, faces, area_eps=1e-14):
    """Topological validity counts of a triangle soup (no welding is performed)."""
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (M, 3)")
    repeated = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    tri = vertices[faces]
    area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    degenerate = repeated | (area <= area_eps)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, 1)
    keys, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    # Orientation: an interior manifold edge is consistent when its two triangles traverse it oppositely.
    forward = (directed[:, 0] < directed[:, 1]).astype(np.int64)
    forward_count = np.bincount(inverse, weights=forward, minlength=len(keys))
    two = counts == 2
    inconsistent = two & (forward_count != 1)
    # Connected components over shared vertices (only referenced vertices count).
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    used = np.unique(faces)
    edges = np.r_[faces[:, [0, 1]], faces[:, [1, 2]]]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),)*2)
    _, labels = connected_components(graph, directed=False)
    roots = np.unique(labels[used])
    return {"vertices": int(len(vertices)), "triangles": int(len(faces)),
            "degenerate_triangles": int(degenerate.sum()), "repeated_index_triangles": int(repeated.sum()),
            "min_triangle_area_m2": float(area.min()) if len(area) else None,
            "edges": int(len(keys)), "boundary_edges": int((counts == 1).sum()),
            "non_manifold_edges": int((counts > 2).sum()), "inconsistently_oriented_edges": int(inconsistent.sum()),
            "connected_components": int(len(roots)), "unreferenced_vertices": int(len(vertices)-len(used)),
            "watertight": bool((counts == 2).all()) and not (counts > 2).any()}


def cut_face_metrics(reference, lattice, positions, codes, vertices, faces, *, encoding="ternary", cuts=1, F=None,
                     component_labels=None, patch_radius_m=.0015, normal_radius_m=.003, region_particle_max_distance_m=None,
                     core_exclusion_layers=3,
                     max_roughness_centres=4000):
    """Distance / normal / roughness of the cut-face part of a mesh against the material cut locus.

    ``vertices`` must be in the same frame as ``positions``. ``component_labels`` (optional, (N,)) are the
    current mechanical components; with them, side distances use only locus points of the vertex's component.
    """
    from scipy.spatial import cKDTree
    positions = np.asarray(positions, float)
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    locus = cut_locus(reference, lattice, positions, codes, encoding=encoding, cuts=cuts, F=F)
    result = {"version": VERSION, "cut_links": int(len(locus["pairs"])), "links_total": locus["links_total"],
              "deformation_map": locus["deformation_map"], "patch_radius_m": float(patch_radius_m),
              "normal_radius_m": float(normal_radius_m)}
    if not len(locus["pairs"]):
        result["status"] = "NO_CUT_LINKS"
        return result
    exterior = exterior_mask(lattice)
    endpoint = np.zeros(len(positions), bool)
    endpoint[locus["pairs"].ravel()] = True
    interior_endpoint = endpoint & ~exterior
    particle_tree = cKDTree(positions)
    distance_to_particle, nearest = particle_tree.query(vertices)
    in_region = interior_endpoint[nearest]
    if region_particle_max_distance_m is not None:
        in_region &= distance_to_particle <= region_particle_max_distance_m
    region_vertices = np.flatnonzero(in_region)
    result["cut_face_vertices"] = int(len(region_vertices))
    result["cut_face_vertex_fraction"] = float(len(region_vertices)/max(len(vertices), 1))
    # Distances: to the nearest locus sample (quantised) and to the local locus plane (primary), for the
    # shared midpoints and for the side points of the vertex's own component.
    side_points = locus["side_points"].reshape(-1, 3)
    side_particle = locus["pairs"].reshape(-1)
    query = vertices[region_vertices]
    midpoint_tree = cKDTree(locus["midpoint"])
    d_mid, _ = midpoint_tree.query(query)
    result["distance_to_link_midpoint_sample_m"] = _percentiles(d_mid)
    result["distance_to_link_midpoint_plane_m"] = _percentiles(
        _finite(local_plane_distance(query, locus["midpoint"], patch_radius_m)))
    d_side, p_side = np.full(len(region_vertices), np.nan), np.full(len(region_vertices), np.nan)
    if component_labels is not None:
        labels = np.asarray(component_labels)
        vertex_label = labels[nearest[region_vertices]]
        side_label = labels[side_particle]
        for label in np.unique(vertex_label):
            mask = vertex_label == label
            own = side_label == label
            if own.any():
                d_side[mask], _ = cKDTree(side_points[own]).query(query[mask])
                p_side[mask] = local_plane_distance(query[mask], side_points[own], patch_radius_m)
    else:
        d_side, _ = cKDTree(side_points).query(query)
        p_side = local_plane_distance(query, side_points, patch_radius_m)
    result["distance_to_side_locus_sample_m"] = _percentiles(_finite(d_side))
    result["distance_to_side_locus_plane_m"] = _percentiles(_finite(p_side))
    result["distance_plane_coverage"] = float(np.isfinite(p_side).mean()) if len(p_side) else None
    # Normals: cut-face triangles (all three vertices in the region).
    face_in = in_region[faces].all(1)
    tri = vertices[faces[face_in]]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    keep = length > 0
    normal = normal[keep]/length[keep, None]
    centroid = tri[keep].mean(1)
    # Local link direction from the side locus points (each cut link contributes its direction at both of its
    # side points), so separated pieces keep their own nearby samples.
    side_direction = np.repeat(locus["direction_current"], 2, axis=0)
    groups = cKDTree(side_points).query_ball_point(centroid, normal_radius_m)
    angles, angle_rows = [], []
    for row, (n, group) in enumerate(zip(normal, groups)):
        if not group:
            continue
        mean = side_direction[group].sum(0)
        norm = np.linalg.norm(mean)
        if norm > 1e-12:
            angles.append(np.degrees(np.arccos(np.clip(abs(n@mean)/norm, 0., 1.))))
            angle_rows.append(row)
    angle_by_row = np.asarray(angles)
    result["cut_face_triangles"] = int(face_in.sum())
    result["normal_deviation_deg"] = _percentiles(angles)
    # Core: cut-face triangles whose vertices' nearest particles are more than ``core_exclusion_layers`` lattice
    # steps from the original exterior surface, i.e. away from the kernel-rounded face edges.
    core_particle = ~exterior_mask(lattice, core_exclusion_layers)
    face_core = face_in & core_particle[nearest[faces]].all(1)
    core_rows = np.flatnonzero(face_core[face_in][keep])
    result["core_exclusion_layers"] = int(core_exclusion_layers)
    result["core_cut_face_triangles"] = int(face_core.sum())
    result["normal_deviation_core_deg"] = _percentiles(np.asarray(angle_by_row)[np.isin(angle_rows, core_rows)])
    core_vertices = region_vertices[core_particle[nearest[region_vertices]]]
    result["core_cut_face_vertices"] = int(len(core_vertices))
    result["distance_to_side_locus_plane_core_m"] = _percentiles(_finite(p_side[core_particle[nearest[region_vertices]]]))
    vertex_groups = None if component_labels is None else np.asarray(component_labels)[nearest[region_vertices]]
    side_groups = None if component_labels is None else np.asarray(component_labels)[side_particle]
    result["roughness_mesh_rms_m"] = patch_roughness(vertices[region_vertices], patch_radius_m, groups=vertex_groups,
                                                     max_centres=max_roughness_centres)
    result["roughness_side_locus_rms_m"] = patch_roughness(side_points, patch_radius_m, groups=side_groups,
                                                           max_centres=max_roughness_centres)
    result["roughness_midpoint_locus_rms_m"] = patch_roughness(locus["midpoint"], patch_radius_m,
                                                               max_centres=max_roughness_centres)
    result["status"] = "MEASURED"
    return result


def plane_distance(points, origin, normal):
    """Signed distances to a declared plane (for comparison only; never the sole reference)."""
    normal = np.asarray(normal, float)
    return (np.asarray(points, float)-np.asarray(origin, float))@(normal/np.linalg.norm(normal))


# =====================================================================================================
# Phase 1 (WP-12): final surface_metrics_v1 metric set
# =====================================================================================================
#
# Spaces. "current" = solver-local current positions x; "reference" = material reference positions X (the
# solver-local positions at t = 0). Every reference-space number of a mesh uses an approximate pull-back
# X_v = X_p + F_p^-1 (v - x_p) through the nearest same-component particle p (declared approximate; exact
# for affine motion of that particle's neighbourhood).
#
# Material cut locus (``material_locus``). One row per (conflicting lattice link, cut digit). Endpoint a carries
# digit 1, endpoint b digit 2 of that cut. Reference locus point P and reference normal N (unit, from side 1
# toward side 2):
#   * source "hermite_cut_events_v1": P = recorded Hermite point, N = recorded reference normal (Nanson pull-back
#     of the solver's swept normal at the assignment instant); sign-inconsistent records use t = 1/2 (flagged);
#   * source "link_midpoint_fallback": P = (X_a + X_b)/2, N = normalised mean of the signed link directions of
#     the same cut within ``normal_radius_m`` (a staircase estimate: number of axis links crossing a plane per unit
#     area is proportional to |n_axis|).
# Side points in current space: q_a = x_a + F_a (P - X_a), q_b = x_b + F_b (P - X_b); outward side normals
# n_a = normalise(F_a^-T N), n_b = -normalise(F_b^-T N) (outward from that side's material).
#
# Region rule ``region_rule_v1`` (``classify_mesh_vertices``). Boundary samples of every mechanical component:
# its cut side points (type "cut") and its exterior cell-face samples x_p + F_p (s/2 e) for every missing
# 6-neighbour direction e of an exterior particle (type "exterior"). A mesh vertex takes the component of its
# nearest particle; it is a cut-face vertex when its nearest boundary sample *of that component* is a cut sample
# within ``region_max_m`` (default 1.5 s), an exterior vertex when that sample is an exterior sample within
# ``region_max_m``, and "unmatched" otherwise (mesh far from every material boundary: e.g. an isosurface bridging
# a gap). Core cut-face vertices are additionally >= ``core_margin_m`` (default 3 s) from every exterior sample
# of their component. A triangle is cut-face (core) when all three vertices are.

REGION_RULE = "region_rule_v1"
LOCUS_HERMITE = "hermite_cut_events_v1"
LOCUS_MIDPOINT = "link_midpoint_fallback"


def _unit(v, eps=1e-30):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v/np.maximum(n, eps)


def _quantiles(values, scale=None):
    out = _percentiles(_finite(values))
    if scale and out["count"]:
        out.update({f"{k}_spacings": (out[k]/scale if out[k] is not None else None) for k in ("p50", "p95", "max", "mean")})
    return out


def conflicting_links_per_cut(lattice, codes, *, encoding="packed2", cuts=1):
    """(R, 2) particle pairs (side 1, side 2) and (R,) cut ids of every (6-neighbour link, cut) whose digits are {1, 2}."""
    pairs, _ = lattice_links(lattice)
    digits = _digits(codes, encoding, int(cuts))
    rows, cut_ids = [], []
    for k in range(int(cuts)):
        da, db = digits[pairs[:, 0], k], digits[pairs[:, 1], k]
        hit = ((da == 1) & (db == 2)) | ((da == 2) & (db == 1))
        a = np.where(da[hit] == 1, pairs[hit, 0], pairs[hit, 1])
        b = np.where(da[hit] == 1, pairs[hit, 1], pairs[hit, 0])
        rows.append(np.c_[a, b])
        cut_ids.append(np.full(len(a), k))
    return (np.vstack(rows) if rows else np.zeros((0, 2), int)).astype(np.int64), np.concatenate(cut_ids).astype(np.int64)


def material_locus(reference, lattice, positions, codes, *, encoding="packed2", cuts=1, F=None, events=None,
                   normal_radius_m=.003, spacing=None):
    """Material cut locus (see the Phase 1 header). ``events``: cut_events_v1 LOG_DTYPE array or None.

    Returns a dict of arrays (one row per conflicting (link, cut)): ``pairs`` (side-1, side-2 particle),
    ``cut``, ``P`` reference locus point, ``N`` reference unit normal (side 1 -> 2), ``midpoint_reference``,
    ``hermite`` (bool: row has a recorded Hermite point), ``sign_inconsistent``, ``side_points`` (R, 2, 3)
    current, ``side_normals`` (R, 2, 3) outward current normals, plus counts and the ``source`` label."""
    from scipy.spatial import cKDTree
    reference = np.asarray(reference, float)
    positions = np.asarray(positions, float)
    pairs, cut = conflicting_links_per_cut(lattice, codes, encoding=encoding, cuts=cuts)
    a, b = pairs[:, 0], pairs[:, 1]
    Xa, Xb = reference[a], reference[b]
    mid = (Xa+Xb)/2
    link = _unit(Xb-Xa)
    P, N = mid.copy(), np.zeros_like(mid)
    hermite = np.zeros(len(pairs), bool)
    inconsistent = np.zeros(len(pairs), bool)
    duplicates = 0
    if events is not None and len(events) and len(pairs):
        lo, hi = np.minimum(events["p"], events["q"]).astype(np.int64), np.maximum(events["p"], events["q"]).astype(np.int64)
        n = np.int64(len(reference))
        ekey = (events["cut"].astype(np.int64)*n+lo)*n+hi
        rkey = (cut*n+np.minimum(a, b))*n+np.maximum(a, b)
        order = np.argsort(ekey, kind="stable")
        duplicates = int(len(ekey)-len(np.unique(ekey)))
        pos = np.minimum(np.searchsorted(ekey[order], rkey), len(ekey)-1)
        found = ekey[order][pos] == rkey
        rows = order[pos[found]]
        hermite[found] = True
        P[found] = events["hermite_reference"][rows].astype(float)
        nref = events["normal_reference"][rows].astype(float)
        N[found] = nref
        inconsistent[found] = (events["flags"][rows] & 1) != 0
    # Midpoint-fallback normals (and any degenerate recorded normal): staircase estimate per cut.
    need = ~hermite | (np.linalg.norm(N, axis=1) < .5)
    for k in np.unique(cut[need]) if need.any() else []:
        own = cut == k
        tree = cKDTree(mid[own])
        idx = np.flatnonzero(need & own)
        groups = tree.query_ball_point(mid[idx], normal_radius_m)
        own_link = link[own]
        for i, g in zip(idx, groups):
            N[i] = own_link[g].sum(0)
    N = _unit(N)
    flips = int(np.count_nonzero(np.einsum("ij,ij->i", N, link) < 0))
    if F is None:
        Fa = Fb = None
        qa, qb = positions[a]+(P-Xa), positions[b]+(P-Xb)
        na, nb = N.copy(), -N
    else:
        F = np.asarray(F, float)
        Fa, Fb = F[a], F[b]
        qa = positions[a]+np.einsum("nij,nj->ni", Fa, P-Xa)
        qb = positions[b]+np.einsum("nij,nj->ni", Fb, P-Xb)
        na = _unit(np.einsum("nji,nj->ni", np.linalg.inv(Fa), N))
        nb = -_unit(np.einsum("nji,nj->ni", np.linalg.inv(Fb), N))
    sources = {LOCUS_HERMITE: int(hermite.sum()), LOCUS_MIDPOINT: int((~hermite).sum())}
    source = LOCUS_HERMITE if hermite.all() and len(pairs) else (LOCUS_MIDPOINT if not hermite.any() else "mixed")
    return {"pairs": pairs, "cut": cut, "P": P, "N": N, "midpoint_reference": mid, "link_direction": link,
            "hermite": hermite, "sign_inconsistent": inconsistent, "side_points": np.stack([qa, qb], 1),
            "side_normals": np.stack([na, nb], 1), "source": source, "rows_by_source": sources,
            "event_duplicates": duplicates, "normal_against_link": flips,
            "deformation_map": "identity" if F is None else "F",
            "events_unmatched": None if events is None else int(len(events)-int(hermite.sum()))}


def exterior_samples(lattice, positions, spacing, F=None):
    """Exterior cell-face samples x_p + F_p (s/2 e) and outward normals for every missing 6-neighbour (p, e)."""
    lattice = np.asarray(lattice, np.int64)
    positions = np.asarray(positions, float)
    low = lattice.min(0)
    dims = lattice.max(0)-low+3
    keys = np.ravel_multi_index((lattice-low+1).T, dims)
    present = np.zeros(int(np.prod(dims)), bool)
    present[keys] = True
    particles, directions = [], []
    for e in np.r_[_AXES, -_AXES]:
        nb = np.ravel_multi_index((lattice-low+1+e).T, dims)
        miss = np.flatnonzero(~present[nb])
        particles.append(miss)
        directions.append(np.tile(e, (len(miss), 1)))
    p = np.concatenate(particles)
    e = np.vstack(directions).astype(float)
    if F is None:
        return p, positions[p]+.5*spacing*e, e
    F = np.asarray(F, float)
    return (p, positions[p]+.5*spacing*np.einsum("nij,nj->ni", F[p], e),
            _unit(np.einsum("nji,nj->ni", np.linalg.inv(F[p]), e)))


def classify_mesh_vertices(vertices, positions, labels, locus, ext, *, spacing, region_max_m=None, core_margin_m=None):
    """``region_rule_v1`` (module header). Returns per-vertex arrays: ``component``, ``kind`` (0 unmatched,
    1 exterior, 2 cut), ``row`` (locus row of the nearest cut sample, -1 otherwise), ``side`` (0/1),
    ``distance_to_sample``, ``core`` (cut-face and >= core margin from the exterior samples)."""
    from scipy.spatial import cKDTree
    region_max_m = 1.5*spacing if region_max_m is None else region_max_m
    core_margin_m = 3*spacing if core_margin_m is None else core_margin_m
    vertices = np.asarray(vertices, float)
    labels = np.asarray(labels)
    _, nearest = cKDTree(positions).query(vertices)
    component = labels[nearest]
    side_pts = locus["side_points"].reshape(-1, 3)
    side_particle = locus["pairs"].reshape(-1)
    ext_particle, ext_pts, _ = ext
    kind = np.zeros(len(vertices), np.int8)
    row = np.full(len(vertices), -1, np.int64)
    side = np.full(len(vertices), -1, np.int8)
    dist = np.full(len(vertices), np.inf)
    dext = np.full(len(vertices), np.inf)
    for c in np.unique(component):
        vm = np.flatnonzero(component == c)
        cs = np.flatnonzero(labels[side_particle] == c)
        es = np.flatnonzero(labels[ext_particle] == c)
        dc = np.full(len(vm), np.inf)
        ic = np.zeros(len(vm), np.int64)
        if len(cs):
            dc, j = cKDTree(side_pts[cs]).query(vertices[vm])
            ic = cs[j]
        de = np.full(len(vm), np.inf)
        if len(es):
            de, _ = cKDTree(ext_pts[es]).query(vertices[vm])
        is_cut = (dc <= de) & (dc <= region_max_m)
        is_ext = ~is_cut & (de <= region_max_m)
        kind[vm[is_cut]] = 2
        kind[vm[is_ext]] = 1
        row[vm[is_cut]] = ic[is_cut]//2
        side[vm[is_cut]] = ic[is_cut] % 2
        dist[vm] = np.minimum(dc, de)
        dext[vm] = de
    core = (kind == 2) & (dext >= core_margin_m)
    return {"rule": REGION_RULE, "component": component, "kind": kind, "row": row, "side": side,
            "distance_to_sample": dist, "distance_to_exterior": dext, "core": core,
            "region_max_m": float(region_max_m), "core_margin_m": float(core_margin_m),
            "counts": {"cut_face": int((kind == 2).sum()), "exterior": int((kind == 1).sum()),
                       "unmatched": int((kind == 0).sum()), "core": int(core.sum()), "vertices": int(len(vertices))}}


def local_plane_signed(queries, points, normals, radius_m, *, min_points=4):
    """Signed distance of each query to the local least-squares plane of ``points`` within ``radius_m``, sign
    along the mean of the corresponding ``normals``; also returns that patch normal (NaN where fewer than
    ``min_points`` samples or a nearly collinear patch, second singular value < 0.2 x the first)."""
    from scipy.spatial import cKDTree
    queries = np.asarray(queries, float)
    out = np.full(len(queries), np.nan)
    patch_normal = np.full((len(queries), 3), np.nan)
    if len(points) < min_points or not len(queries):
        return out, patch_normal
    tree = cKDTree(points)
    for i, g in enumerate(tree.query_ball_point(queries, radius_m)):
        if len(g) >= min_points:
            local = points[g]
            centre = local.mean(0)
            _, sv, vt = np.linalg.svd(local-centre)
            if sv[1] < .2*sv[0]:      # (nearly) collinear patch (face edge): the plane is undetermined
                continue
            n = vt[-1]
            ref = normals[g].sum(0)
            if n@ref < 0:
                n = -n
            out[i] = (queries[i]-centre)@n
            patch_normal[i] = n
    return out, patch_normal


def point_triangle_distance(points, a, b, c):
    """Exact Euclidean distance from points (Q, 3) to triangles (Q, 3) (Ericson, vectorised); returns (d, closest)."""
    p = np.asarray(points, float)
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p-b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p-c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va+vb+vc
        v = np.where(np.abs(denom) > 0, vb/denom, 0.)
        w = np.where(np.abs(denom) > 0, vc/denom, 0.)
        closest = a+v[:, None]*ab+w[:, None]*ac                                    # interior
        # edges
        vab = d1/(d1-d3)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        closest[m] = a[m]+vab[m, None]*ab[m]
        wac = d2/(d2-d6)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        closest[m] = a[m]+wac[m, None]*ac[m]
        wbc = (d4-d3)/((d4-d3)+(d5-d6))
        m = (va <= 0) & ((d4-d3) >= 0) & ((d5-d6) >= 0)
        closest[m] = b[m]+wbc[m, None]*(c[m]-b[m])
    # vertices
    m = (d1 <= 0) & (d2 <= 0)
    closest[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3)
    closest[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6)
    closest[m] = c[m]
    return np.linalg.norm(p-closest, axis=1), closest


def point_mesh_distance(points, vertices, faces, *, k=24, tree=None):
    """Distance from points to a triangle mesh: exact point-triangle distance over the k triangles with the
    nearest centroids (exact whenever the closest triangle is among them; k = 24 at our triangle sizes)."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    if not len(points) or not len(faces):
        return np.full(len(points), np.inf)
    tri = vertices[faces]
    tree = tree or cKDTree(tri.mean(1))
    k = min(k, len(faces))
    _, idx = tree.query(points, k=k)
    idx = idx.reshape(len(points), k)
    best = np.full(len(points), np.inf)
    for j in range(k):
        t = tri[idx[:, j]]
        d, _ = point_triangle_distance(points, t[:, 0], t[:, 1], t[:, 2])
        best = np.minimum(best, d)
    return best


def triangle_normals(vertices, faces):
    tri = np.asarray(vertices, float)[np.asarray(faces, np.int64)]
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area = .5*np.linalg.norm(n, axis=1)
    return _unit(n), area, tri.mean(1)


def shells(vertices, faces):
    """Connected components of a triangle mesh over shared vertices: (count, per-face shell id)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    faces = np.asarray(faces, np.int64)
    edges = np.r_[faces[:, [0, 1]], faces[:, [1, 2]]]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),)*2)
    _, labels = connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    _, face_shell = np.unique(face_label, return_inverse=True)
    return int(face_shell.max()+1) if len(faces) else 0, face_shell.ravel()


def shell_properties(vertices, faces):
    """Per shell: signed volume (divergence theorem), area, Euler characteristic V - E + F, triangles."""
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    count, shell = shells(vertices, faces)
    tri = vertices[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))/6
    area = .5*np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    rows = []
    for s in range(count):
        f = faces[shell == s]
        e = np.unique(np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1), axis=0)
        rows.append({"shell": s, "triangles": int(len(f)), "signed_volume_m3": float(signed[shell == s].sum()),
                     "area_m2": float(area[shell == s].sum()),
                     "euler_characteristic": int(len(np.unique(f))-len(e)+len(f))})
    return rows, shell


def self_intersections(vertices, faces, *, max_triangles=20000, seed=0, shell=None):
    """Bounded self-intersection proxy: for up to ``max_triangles`` sampled triangles, count intersecting pairs with
    nearby triangles that share no vertex (segment-triangle Moller-Trumbore on the 3 + 3 edges; coplanar contact
    is not counted). Returns sampled counts, the estimate scaled to all triangles, and how many of the hits are
    between different shells (e.g. two pieces drawn interpenetrating)."""
    from scipy.spatial import cKDTree
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    if len(faces) < 2:
        return {"sampled_triangles": int(len(faces)), "intersecting_pairs_sampled": 0, "estimated_pairs": 0,
                "inter_shell_pairs_sampled": 0}
    tri = vertices[faces]
    centroid = tri.mean(1)
    radius = np.linalg.norm(tri-centroid[:, None], axis=2).max(1)
    rng = np.random.default_rng(seed)
    sample = np.arange(len(faces)) if len(faces) <= max_triangles else np.sort(rng.choice(len(faces), max_triangles, replace=False))
    tree = cKDTree(centroid)
    reach = 2*radius.max()
    cand = tree.query_ball_point(centroid[sample], reach)
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
        T, S = tri[y], tri[x]
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            hit |= crosses(S[:, e0], S[:, e1], T[:, 0], T[:, 1], T[:, 2])
    pairs = np.unique(np.sort(np.c_[ia[hit], ib[hit]], 1), axis=0) if hit.any() else np.zeros((0, 2), np.int64)
    inter = int((shell[pairs[:, 0]] != shell[pairs[:, 1]]).sum()) if shell is not None and len(pairs) else 0
    scale = len(faces)/len(sample)
    return {"sampled_triangles": int(len(sample)), "intersecting_pairs_sampled": int(len(pairs)),
            "estimated_pairs": float(len(pairs)*scale), "inter_shell_pairs_sampled": inter,
            "method": "segment-triangle crossing among vertex-disjoint neighbouring triangles (bounded sample)"}


def mesh_validity_v1(vertices, faces, *, max_intersection_triangles=20000):
    """mesh_validity + per-shell orientation (signed volume > 0 for outward winding), Euler characteristic and
    the bounded self-intersection proxy."""
    base = mesh_validity(vertices, faces)
    if not len(faces):
        return {**base, "shells": 0}
    rows, shell = shell_properties(vertices, faces)
    inter = self_intersections(vertices, faces, max_triangles=max_intersection_triangles, shell=shell)
    return {**base, "shells": len(rows), "shell_rows": rows,
            "shells_negative_volume": int(sum(r["signed_volume_m3"] < 0 for r in rows)),
            "shells_euler_not_2": int(sum(r["euler_characteristic"] != 2 for r in rows)),
            "self_intersection": inter}


def topology_consistency(vertices, faces, positions, labels, masses=None, face_component=None):
    """Mesh shells vs mechanical components. A shell's component is taken from the extractor's provenance
    (``face_component``: the component whose particles produced each triangle, e.g. from CuttingSurface
    ``component_ranges``) when given, otherwise from the majority of its triangles' nearest particles. Reported:
    shells built from > 1 component (provenance) or with nearest-particle purity < 0.99 (no provenance),
    components without a shell / with > 1 shell, and ``faces_nearer_other_component`` (triangles whose first
    vertex is nearer to another component's particle: visual interpenetration after recontact)."""
    from scipy.spatial import cKDTree
    labels = np.asarray(labels)
    count, shell = shells(vertices, faces)
    _, nearest = cKDTree(positions).query(np.asarray(vertices)[np.asarray(faces)[:, 0]])
    near_comp = labels[nearest]
    face_comp = near_comp if face_component is None else np.asarray(face_component)
    rows, mixed = [], 0
    for s in range(count):
        comp, n = np.unique(face_comp[shell == s], return_counts=True)
        purity = n.max()/n.sum()
        rows.append({"shell": s, "component": int(comp[n.argmax()]), "purity": float(purity), "triangles": int(n.sum()),
                     "nearest_particle_purity": float(np.mean(near_comp[shell == s] == comp[n.argmax()]))})
        mixed += purity < (1. if face_component is not None else .99)
    comps = np.unique(labels)
    per = {int(c): sum(r["component"] == c for r in rows) for c in comps}
    mass = None if masses is None else np.bincount(labels, weights=masses)
    return {"mesh_shells": count, "mechanical_components": int(len(comps)), "shell_rows": rows[:50],
            "shells_spanning_components": int(mixed),
            "shell_component_source": "extractor provenance" if face_component is not None else "nearest particle majority",
            "faces_nearer_other_component": int(np.count_nonzero(near_comp != face_comp)),
            "components_without_shell": [int(c) for c, k in per.items() if k == 0],
            "components_with_multiple_shells": {str(c): int(k) for c, k in per.items() if k > 1},
            "components_without_shell_mass_kg": None if mass is None else float(sum(mass[c] for c, k in per.items() if k == 0)),
            "shells_equal_components": bool(count == len(comps) and not mixed and all(k == 1 for k in per.values()))}


def volume_report(vertices, faces, positions, labels, particle_volume_ref, F=None):
    """Mesh volume per mechanical component vs material volume: reference Σ m/ρ (particle_volume_ref per
    particle) and current Σ (m/ρ) det F."""
    from scipy.spatial import cKDTree
    labels = np.asarray(labels)
    vref = np.broadcast_to(np.asarray(particle_volume_ref, float), labels.shape)
    J = np.ones(len(labels)) if F is None else np.linalg.det(np.asarray(F, float))
    tri = np.asarray(vertices, float)[np.asarray(faces, np.int64)]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))/6
    _, nearest = cKDTree(positions).query(tri.mean(1))
    face_comp = labels[nearest]
    rows = []
    for c in np.unique(labels):
        mesh = float(signed[face_comp == c].sum())
        ref = float(vref[labels == c].sum())
        cur = float((vref*J)[labels == c].sum())
        rows.append({"component": int(c), "particles": int((labels == c).sum()), "mesh_volume_m3": mesh,
                     "material_reference_volume_m3": ref, "material_current_volume_m3": cur,
                     "relative_error_reference": (mesh-ref)/ref, "relative_error_current": (mesh-cur)/cur})
    total_mesh = float(signed.sum())
    total_ref, total_cur = float(vref.sum()), float((vref*J).sum())
    rows.sort(key=lambda r: -r["material_reference_volume_m3"])
    return {"total_mesh_volume_m3": total_mesh, "total_reference_volume_m3": total_ref,
            "total_current_volume_m3": total_cur, "total_relative_error_reference": (total_mesh-total_ref)/total_ref,
            "total_relative_error_current": (total_mesh-total_cur)/total_cur,
            "max_abs_component_relative_error_current": float(max(abs(r["relative_error_current"]) for r in rows)),
            "components": rows[:20], "face_assignment": "triangle centroid -> nearest particle's component"}


def pull_back(points, positions, reference, F, labels=None, point_labels=None):
    """Approximate reference-space image X = X_p + F_p^-1 (v - x_p) via the nearest (same-component) particle."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    out = np.empty_like(points)
    Finv = None if F is None else np.linalg.inv(np.asarray(F, float))
    groups = [(None, np.arange(len(points)))] if labels is None or point_labels is None else \
        [(c, np.flatnonzero(point_labels == c)) for c in np.unique(point_labels)]
    for c, idx in groups:
        cand = np.arange(len(positions)) if c is None else np.flatnonzero(labels == c)
        _, j = cKDTree(positions[cand]).query(points[idx])
        p = cand[j]
        delta = points[idx]-positions[p]
        out[idx] = reference[p]+(delta if Finv is None else np.einsum("nij,nj->ni", Finv[p], delta))
    return out


def side_group_keys(locus, labels, cuts):
    """Patch-grouping key per side sample: (component, cut, side), so a patch never spans two faces."""
    comp = np.asarray(labels)[locus["pairs"].reshape(-1)]
    cut = np.repeat(locus["cut"], 2)
    side = np.tile([0, 1], len(locus["cut"]))
    return (comp*max(int(cuts), 1)+cut)*2+side


def surface_report(reference, lattice, positions, codes, vertices, faces, *, spacing, labels, masses=None,
                   density=None, encoding="packed2", cuts=1, F=None, events=None, patch_radius_m=.002,
                   normal_radius_m=.003, coverage_radius_m=None, region_max_m=None, core_margin_m=None,
                   max_roughness_centres=4000, reference_space=True, locus=None, validity=True, face_component=None):
    """Phase 1 ``surface_metrics_v1`` report of one mesh against the material state (all lengths m, also in
    spacings). Current space is primary; ``reference_space`` adds the approximate pull-back numbers.

    Distances (cut-face vertices of ``region_rule_v1``): to the nearest same-component side sample and to the
    local plane of the same (component, cut, side) side samples within ``patch_radius_m``; signed by the side's
    outward normal (> 0: the drawn face lies outside the material, into the kerf); ``bias_toward_side2`` is the
    mean offset along the cut normal N (side 1 -> 2) of both faces (a common shift toward one blade side) and
    ``thickness_bias`` = mean(side-1 offset) - mean(side-2 offset) (> 0: faces drawn outward on both sides).
    Reverse distance: side samples -> mesh (exact point-triangle), whose fraction within ``coverage_radius_m``
    (default s) is ``coverage``. Normals: core cut-face triangles vs the normal of the local least-squares plane of
    the same-side locus samples within ``normal_radius_m`` (unsigned angle, plus the fraction pointing outward);
    the patch mean of the F-mapped (Nanson) side normals is reported as a secondary reference. ``per_cut``
    repeats distance/bias/coverage/normal/roughness per cut digit.
    Roughness: patch RMS (radius ``patch_radius_m``) on the core region of the mesh and on the side locus (core
    samples), grouped by (component, cut, side). Rim (boundary continuity): distance from the cut-face/exterior
    corner samples to the mesh. Topology, volume (needs ``masses`` and ``density``), validity."""
    from scipy.spatial import cKDTree
    s = float(spacing)
    reference, positions = np.asarray(reference, float), np.asarray(positions, float)
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    labels = np.asarray(labels)
    coverage_radius_m = s if coverage_radius_m is None else coverage_radius_m
    locus = locus if locus is not None else material_locus(reference, lattice, positions, codes, encoding=encoding,
                                                           cuts=cuts, F=F, events=events, normal_radius_m=normal_radius_m)
    ext = exterior_samples(lattice, positions, s, F)
    out = {"version": VERSION, "region_rule": REGION_RULE, "spacing_m": s, "patch_radius_m": patch_radius_m,
           "normal_radius_m": normal_radius_m, "coverage_radius_m": coverage_radius_m,
           "locus_source": locus["source"], "locus_rows_by_source": locus["rows_by_source"],
           "conflicting_link_rows": int(len(locus["cut"])), "vertices": int(len(vertices)), "triangles": int(len(faces))}
    if validity:
        out["mesh_validity"] = mesh_validity_v1(vertices, faces)
    out["topology"] = topology_consistency(vertices, faces, positions, labels, masses, face_component)
    if masses is not None and density is not None:
        out["volume"] = volume_report(vertices, faces, positions, labels, np.asarray(masses, float)/float(density), F)
    if not len(locus["cut"]):
        out["status"] = "NO_CUT_LINKS"
        cls = classify_mesh_vertices(vertices, positions, labels, locus, ext, spacing=s, region_max_m=region_max_m,
                                     core_margin_m=core_margin_m)
        out["region"] = cls["counts"]
        return out
    cls = classify_mesh_vertices(vertices, positions, labels, locus, ext, spacing=s, region_max_m=region_max_m,
                                 core_margin_m=core_margin_m)
    out["region"] = {**cls["counts"], "region_max_m": cls["region_max_m"], "core_margin_m": cls["core_margin_m"]}
    side_pts = locus["side_points"].reshape(-1, 3)
    side_nrm = locus["side_normals"].reshape(-1, 3)
    keys = side_group_keys(locus, labels, cuts)
    region = np.flatnonzero(cls["kind"] == 2)
    vkey = (cls["component"][region]*max(int(cuts), 1)+locus["cut"][cls["row"][region]])*2+cls["side"][region]
    signed = np.full(len(region), np.nan)
    for key in np.unique(vkey):
        vm = vkey == key
        sm = keys == key
        signed[vm], _ = local_plane_signed(vertices[region[vm]], side_pts[sm], side_nrm[sm], patch_radius_m)
    out["distance_to_side_locus_sample"] = _quantiles(cls["distance_to_sample"][region], s)
    out["distance_to_side_locus_plane"] = _quantiles(np.abs(signed), s)
    core_v = cls["core"][region]
    out["distance_to_side_locus_plane_core"] = _quantiles(np.abs(signed[core_v]), s)
    out["plane_coverage"] = float(np.isfinite(signed).mean()) if len(signed) else None
    out["signed_outward"] = _quantiles(signed, s)
    # bias along N (side 1 -> 2): side-1 outward = +N, side-2 outward = -N
    along = np.where(cls["side"][region] == 0, signed, -signed)
    s1, s2 = along[cls["side"][region] == 0], along[cls["side"][region] == 1]
    m1 = float(np.nanmean(s1)) if np.isfinite(s1).any() else None
    m2 = float(np.nanmean(s2)) if np.isfinite(s2).any() else None
    out["signed_bias"] = {"mean_offset_along_N_side1_m": m1, "mean_offset_along_N_side2_m": m2,
                          "bias_toward_side2_m": None if m1 is None or m2 is None else (m1+m2)/2,
                          "thickness_bias_m": None if m1 is None or m2 is None else m1-m2,
                          "side1_vertices": int(np.isfinite(s1).sum()), "side2_vertices": int(np.isfinite(s2).sum())}
    # reverse: locus -> mesh, coverage (per component mesh would be stricter; the mesh is per component already)
    rev = point_mesh_distance(side_pts, vertices, faces)
    out["locus_to_mesh_distance"] = _quantiles(rev, s)
    out["coverage"] = float((rev <= coverage_radius_m).mean())
    # normals on core triangles
    normal, area, centroid = triangle_normals(vertices, faces)
    tri_core = cls["core"][faces].all(1)
    tri_in = (cls["kind"][faces] == 2).all(1)
    idx = np.flatnonzero(tri_core)
    angles, angles_F, outward = (np.full(len(idx), np.nan) for _ in range(3))
    if len(idx):
        vrow = cls["row"][faces[idx, 0]]
        vside = cls["side"][faces[idx, 0]]
        vcomp = cls["component"][faces[idx, 0]]
        own_key = (vcomp*max(int(cuts), 1)+locus["cut"][vrow])*2+vside
        for key in np.unique(own_key):
            tm = np.flatnonzero(own_key == key)
            sm = keys == key
            if not sm.any():
                continue
            # primary: normal of the local least-squares plane of the same-side locus samples (radius normal_radius_m)
            _, pn = local_plane_signed(centroid[idx[tm]], side_pts[sm], side_nrm[sm], normal_radius_m)
            ok = np.isfinite(pn[:, 0])
            c = np.einsum("ij,ij->i", normal[idx[tm[ok]]], pn[ok])
            angles[tm[ok]] = np.degrees(np.arccos(np.clip(np.abs(c), 0, 1)))
            outward[tm[ok]] = c > 0
            # secondary: patch mean of the F-mapped (Nanson) outward side normals
            groups = cKDTree(side_pts[sm]).query_ball_point(centroid[idx[tm]], normal_radius_m)
            nrm = side_nrm[sm]
            for i, g in zip(tm, groups):
                if g:
                    m = nrm[g].sum(0)
                    nm = np.linalg.norm(m)
                    if nm > 1e-12:
                        angles_F[i] = np.degrees(np.arccos(np.clip(abs(normal[idx[i]]@m/nm), 0, 1)))
    out["normal_error_deg_core"] = _quantiles(angles)
    out["normal_error_vs_F_mapped_normal_deg_core"] = _quantiles(angles_F)
    out["normal_outward_fraction_core"] = float(np.nanmean(outward)) if np.isfinite(outward).any() else None
    out["cut_face_triangles"] = int(tri_in.sum())
    out["cut_face_triangles_core"] = int(tri_core.sum())
    out["cut_face_area_m2"] = float(area[tri_in].sum())
    # roughness: core mesh vertices vs core side samples (side samples >= core margin from the exterior)
    rv = region[core_v]
    rkey = vkey[core_v]
    out["roughness_mesh_core"] = patch_roughness(vertices[rv], patch_radius_m, groups=rkey, max_centres=max_roughness_centres)
    _, ext_d = (cKDTree(ext[1]).query(side_pts) if len(ext[1]) else (None, np.full(len(side_pts), np.inf)))
    core_s = ext_d >= cls["core_margin_m"]
    out["roughness_locus_core"] = patch_roughness(side_pts[core_s], patch_radius_m, groups=keys[core_s],
                                                  max_centres=max_roughness_centres)
    # per cut digit (multi-cut scenes): the same statistics restricted to that digit's samples / vertices
    vcut = locus["cut"][cls["row"][region]]
    scut = np.repeat(locus["cut"], 2)
    tri_cut = locus["cut"][cls["row"][faces[idx, 0]]] if len(idx) else np.zeros(0, int)
    per_cut = {}
    for k in np.unique(locus["cut"]):
        vm, sm = vcut == k, scut == k
        a1 = along[vm & (cls["side"][region] == 0)]
        a2 = along[vm & (cls["side"][region] == 1)]
        b1 = float(np.nanmean(a1)) if np.isfinite(a1).any() else None
        b2 = float(np.nanmean(a2)) if np.isfinite(a2).any() else None
        per_cut[str(int(k))] = {
            "cut_face_vertices": int(vm.sum()),
            "distance_to_side_locus_plane": _quantiles(np.abs(signed[vm]), s),
            "distance_to_side_locus_plane_core": _quantiles(np.abs(signed[vm & core_v]), s),
            "signed_bias": {"mean_offset_along_N_side1_m": b1, "mean_offset_along_N_side2_m": b2,
                            "bias_toward_side2_m": None if b1 is None or b2 is None else (b1+b2)/2,
                            "thickness_bias_m": None if b1 is None or b2 is None else b1-b2},
            "locus_to_mesh_distance": _quantiles(rev[sm], s), "coverage": float((rev[sm] <= coverage_radius_m).mean()),
            "normal_error_deg_core": _quantiles(angles[tri_cut == k]),
            "roughness_mesh_core": patch_roughness(vertices[region[vm & core_v]], patch_radius_m, groups=vkey[vm & core_v],
                                                   max_centres=max_roughness_centres),
            "roughness_locus_core": patch_roughness(side_pts[sm & core_s], patch_radius_m, groups=keys[sm & core_s],
                                                    max_centres=max_roughness_centres)}
    out["per_cut"] = per_cut
    # rim / boundary continuity: corner samples at cut-face/exterior junctions
    out["rim"] = rim_continuity(locus, lattice, reference, positions, F, s, vertices, faces)
    if reference_space:
        Xv = pull_back(vertices[region], positions, reference, F, labels, cls["component"][region])
        P = locus["P"]
        dref = np.full(len(region), np.nan)
        rows = cls["row"][region]
        for k in np.unique(locus["cut"]):
            vm = locus["cut"][rows] == k
            pm = locus["cut"] == k
            dref[vm], _ = local_plane_signed(Xv[vm], P[pm], locus["N"][pm], patch_radius_m)
        out["reference_space"] = {
            "distance_to_locus_plane": _quantiles(np.abs(dref), s),
            "distance_to_locus_plane_core": _quantiles(np.abs(dref[core_v]), s),
            "pull_back": "approximate: X = X_p + F_p^-1 (v - x_p), nearest same-component particle",
            "locus_roughness": patch_roughness(P, patch_radius_m, groups=locus["cut"], max_centres=max_roughness_centres),
            "midpoint_roughness": patch_roughness(locus["midpoint_reference"], patch_radius_m, groups=locus["cut"],
                                                  max_centres=max_roughness_centres)}
    out["status"] = "MEASURED"
    return out


def rim_continuity(locus, lattice, reference, positions, F, spacing, vertices, faces):
    """Boundary continuity: corner samples where a cut face meets the original exterior,
    c = x_e + F_e (P - X_e + s/2 d) for each cut-link endpoint e with a missing 6-neighbour direction d orthogonal
    to the link, and their exact distance to the mesh (P95/max). A kernel isosurface rounds this corner (distance
    grows with the kernel support); a gap or missing junction shows as a large distance. Open (boundary) mesh
    edges within 2 s of a corner sample are counted."""
    from scipy.spatial import cKDTree
    lattice = np.asarray(lattice, np.int64)
    reference, positions = np.asarray(reference, float), np.asarray(positions, float)
    low = lattice.min(0)
    dims = lattice.max(0)-low+3
    present = np.zeros(int(np.prod(dims)), bool)
    present[np.ravel_multi_index((lattice-low+1).T, dims)] = True
    link = np.abs(locus["link_direction"])
    ps, offs = [], []
    for side in (0, 1):
        e_idx = locus["pairs"][:, side]
        for d in np.r_[_AXES, -_AXES]:
            m = (link@np.abs(d) < .5) & ~present[np.ravel_multi_index((lattice[e_idx]-low+1+d).T, dims)]
            if m.any():
                ps.append(e_idx[m])
                offs.append(locus["P"][m]-reference[e_idx[m]]+.5*spacing*d)
    if not ps:
        return {"corner_samples": 0}
    p, off = np.concatenate(ps), np.vstack(offs)
    corner = positions[p]+(off if F is None else np.einsum("nij,nj->ni", np.asarray(F, float)[p], off))
    d = point_mesh_distance(corner, vertices, faces)
    faces = np.asarray(faces, np.int64)
    edges = np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 1)
    keys, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = keys[counts == 1]
    near_open = 0
    if len(boundary):
        mid = np.asarray(vertices, float)[boundary].mean(1)
        dist, _ = cKDTree(corner).query(mid)
        near_open = int((dist <= 2*spacing).sum())
    return {"corner_samples": int(len(p)), "corner_to_mesh_distance": _quantiles(d, spacing),
            "open_edges_near_junction": near_open}


def temporal_stability(vertices_a, faces_a, vertices_b, faces_b, positions_a, positions_b, *, spacing):
    """Frame-to-frame surface jump not explained by material motion.

    Mesh A is transported by the displacement of each vertex's nearest particle (x_b - x_a); the residual is the
    symmetric exact point-triangle distance between transported A and B. At a fixed material state this is the
    pure extraction jump (0 for a deterministic extractor); under slow motion it isolates re-meshing jumps from the
    material's own motion (whose P95 is reported alongside)."""
    from scipy.spatial import cKDTree
    va, vb = np.asarray(vertices_a, float), np.asarray(vertices_b, float)
    fa, fb = np.asarray(faces_a, np.int64), np.asarray(faces_b, np.int64)
    pa, pb = np.asarray(positions_a, float), np.asarray(positions_b, float)
    same_topology = bool(fa.shape == fb.shape and np.array_equal(fa, fb) and va.shape == vb.shape)
    _, near = cKDTree(pa).query(va)
    moved = va+(pb[near]-pa[near])
    ab = point_mesh_distance(vb, moved, fa)
    ba = point_mesh_distance(moved, vb, fb)
    raw = point_mesh_distance(vb, va, fa)
    out = {"same_topology": same_topology, "vertices": [int(len(va)), int(len(vb))], "triangles": [int(len(fa)), int(len(fb))],
           "residual_after_transport": _quantiles(np.r_[ab, ba], spacing), "raw_jump": _quantiles(raw, spacing),
           "material_displacement": _quantiles(np.linalg.norm(pb-pa, axis=1), spacing)}
    if same_topology:
        out["vertexwise_max_m"] = float(np.abs(vb-va).max()) if len(va) else 0.
    return out
