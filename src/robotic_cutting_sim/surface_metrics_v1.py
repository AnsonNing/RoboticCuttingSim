"""Cut-surface geometry metrics measured against the material state (WP-12 Phase 0 draft, read-only).

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
    return {**_percentiles(values), "radius_m": float(radius_m), "min_points": int(min_points),
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
