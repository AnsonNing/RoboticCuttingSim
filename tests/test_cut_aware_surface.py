"""cut_aware_surface_v1 (WP-12 Phase 3 Layers 3-6, tech review §6): geometric unit tests on synthetic data.

These are algorithmic checks with their own analytic expectations. Codes and exact Hermite crossings are set
from a geometric rule (a plane) outside any physics run, as review §7.1 allows for G1/G2-style oracle unit
tests; they are not Phase 3 acceptance measurements and use no surface_protocol_v1 threshold.
"""
import numpy as np
import pytest

from robotic_cutting_sim import cut_aware_surface as cas
from robotic_cutting_sim.cut_topology import pack_digits
from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition

S = .001        # particle spacing
DX = .004       # TLMPM reference grid (material_config default)
EXACT = 1e-9    # m
LOG_RTOL = 1e-7  # volumes from a log: cut_events_v1 stores t as float32 (|dt| ~ 6e-8, i.e. 6e-11 m at 1 mm)

LOG_DTYPE = np.dtype([("p", np.int32), ("q", np.int32), ("cut", np.int32), ("substep", np.int32),
                      ("flags", np.int32), ("t", np.float32), ("s_p", np.float32), ("s_q", np.float32),
                      ("hermite_reference", np.float32, (3,)), ("normal_reference", np.float32, (3,)),
                      ("normal_current", np.float32, (3,)), ("edge_origin", np.float32, (3,))])


def box(dims=(10, 6, 5)):
    ijk = np.stack(np.meshgrid(*(np.arange(n) for n in dims), indexing="ij"), -1).reshape(-1, 3)
    half = np.array(dims)*S/2
    X = (ijk+.5)*S-[half[0], half[1], 0.]          # sampler convention: centred in x, y, base at z = 0
    return ijk, X, np.array([-half[0], -half[1], 0.]), np.array([half[0], half[1], dims[2]*S])


def surface(ijk, X, cuts=1, **options):
    partition = CurrentMechanicalPartition(*cas.reference_stencil(X, DX), cuts, encoding="packed2")
    return cas.CutAwareSurface(partition, S, ijk, X, **options)


def identity_F(n):
    return np.tile(np.eye(3), (n, 1, 1))


def plane_digits(X, origin, normal):
    return np.where((X-origin)@normal >= 0, 2, 1)


def exact_log(ijk, X, digits, planes):
    """Exact Hermite log (cut_events_v1 layout) for codes set by planes: t = s_p/(s_p-s_q), normal = plane normal."""
    lattice = cas.ReferenceLattice(ijk, X, S)
    rows = []
    for k, (origin, normal) in enumerate(planes):
        normal = np.asarray(normal, float)/np.linalg.norm(normal)
        d = digits[:, k]
        a, b = lattice.link_lo, lattice.link_hi
        both = (a >= 0) & (b >= 0)
        a, b = a[both], b[both]
        hit = ((d[a] == 1) & (d[b] == 2)) | ((d[a] == 2) & (d[b] == 1))
        for p, q in zip(a[hit], b[hit]):
            sp, sq = (X[p]-origin)@normal, (X[q]-origin)@normal
            rows.append((p, q, k, 0, 0, sp/(sp-sq), sp, sq, X[p]+sp/(sp-sq)*(X[q]-X[p]), normal, normal, origin))
    log = np.zeros(len(rows), LOG_DTYPE)
    for j, row in enumerate(rows):
        log[j] = row
    return log


def cut_vertices(surf, faces):
    return np.unique(faces[surf.last["face_kind"] == cas.LINK_CUT])


def assert_closed_manifold(info, shells=None):
    v = info["mesh_validity_reference"]
    assert v["boundary_edges"] == 0 and v["non_manifold_edges"] == 0 and v["inconsistently_oriented_edges"] == 0
    assert v["non_manifold_vertices"] == 0 and v["degenerate_triangles"] == 0 and v["unused_vertices"] == 0
    if shells is not None:
        assert info["audit"]["mesh_shells"] == shells and v["euler_characteristic"] == 2*shells


# ----------------------------------------------------------------------------- uncut box

@pytest.mark.parametrize("profile", ["cell_face_v1", "analytic_reference_v1"])
def test_uncut_box_is_closed_exact_and_sharp(profile):
    ijk, X, low, high = box()
    surf = surface(ijk, X, boundary_profile=profile,
                   analytic_shape=cas.AnalyticBox(low, high) if profile != "cell_face_v1" else None)
    vertices, faces, info = surf.extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))
    assert_closed_manifold(info, shells=1)
    assert info["mechanical_components"] == 1 and info["faces"]["cut_triangles"] == 0
    assert np.isclose(cas.signed_volume(vertices, faces), len(X)*S**3, rtol=1e-12, atol=0)
    # every vertex on the box surface, all eight sharp corners present exactly
    inside = np.maximum(low-vertices, vertices-high).max(1)
    assert np.abs(inside).max() < EXACT
    corners = np.array([[(low, high)[(c >> a) & 1][a] for a in range(3)] for c in range(8)])
    assert all(np.linalg.norm(vertices-c, axis=1).min() < EXACT for c in corners)
    # outward axis-aligned normals
    n = cas.triangle_normals(vertices, faces)
    n /= np.linalg.norm(n, axis=1)[:, None]
    assert np.allclose(np.abs(n).max(1), 1, atol=1e-12)
    centroid = vertices[faces].mean(1)
    assert np.all(((centroid-(low+high)/2)*n).sum(1) > 0)
    assert info["boundary_hermite_profile"] == profile


def test_analytic_box_profile_equals_cell_face_for_lattice_aligned_box():
    ijk, X, low, high = box((5, 4, 3))
    codes = np.zeros(len(X), np.int64)
    a = surface(ijk, X).extract(X, codes, identity_F(len(X)))
    b = surface(ijk, X, boundary_profile="analytic_reference_v1",
                analytic_shape=cas.AnalyticBox(low, high)).extract(X, codes, identity_F(len(X)))
    assert np.allclose(a[0], b[0], atol=1e-15) and np.array_equal(a[1], b[1])
    assert b[2]["hermite"]["boundary_links_by_source"] == {"boundary_analytic": b[2]["hermite"]["boundary_links"]}


def test_analytic_heightfield_profile_follows_the_initial_dome():
    from robotic_cutting_sim.benchmark import Bump, HeightFieldSpec
    from robotic_cutting_sim.workpieces import WorkpieceSpec
    spec = WorkpieceSpec(HeightFieldSpec(.030, .020, .006, (Bump(0., 0., .008, .008, .004, "dome"),)))
    X, _, ijk = spec.sample_particles(S)
    X = X.astype(np.float64)
    shape = cas.analytic_shape_from_workpiece(spec)
    codes = np.zeros(len(X), np.int64)
    terr = surface(ijk, X).extract(X, codes, identity_F(len(X)))
    smooth = surface(ijk, X, boundary_profile="analytic_reference_v1", analytic_shape=shape)
    vertices, faces, info = smooth.extract(X, codes, identity_F(len(X)))
    assert info["hermite"]["boundary_links_by_source"] == {"boundary_analytic": info["hermite"]["boundary_links"]}
    assert_closed_manifold(info, shells=1)
    # analytic volume of the initial shape by fine quadrature
    xs = np.linspace(-.015, .015, 1201)[:-1]+.0000125
    ys = np.linspace(-.010, .010, 801)[:-1]+.0000125
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    true = spec.geometry.height(gx, gy).sum()*.000025**2
    v_smooth, v_terr = cas.signed_volume(vertices, faces), cas.signed_volume(terr[0], terr[1])
    assert np.isclose(v_terr, len(X)*S**3, rtol=1e-12, atol=0)
    assert abs(v_smooth-true) < abs(v_terr-true)
    # top vertices of the smooth profile lie close to the height field (sub-spacing, curvature-limited)
    top = vertices[:, 2] > .0061
    gap = vertices[top, 2]-spec.geometry.height(vertices[top, 0], vertices[top, 1])
    assert np.percentile(np.abs(gap), 95) < .1*S


# ----------------------------------------------------------------------------- planar cuts with exact Hermite

def test_axis_plane_cut_with_exact_hermite_is_on_the_plane_and_splits_volume():
    ijk, X, low, high = box()
    origin, normal = np.array([.0013, 0, 0]), np.array([1., 0, 0])      # 0.3 s off the column midplane
    digits = plane_digits(X, origin, normal)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)
    assert info["mechanical_components"] == 2 and info["audit"]["mesh_shells"] == 2
    assert_closed_manifold(info, shells=2)
    on = cut_vertices(surf, faces)
    assert np.abs((vertices[on]-origin)@normal).max() < EXACT
    # two shells never share a vertex
    side = surf.last["vertex_component"]
    assert set(faces[surf.last["face_component"] == 0].ravel()).isdisjoint(faces[surf.last["face_component"] == 1].ravel())
    assert np.array_equal(side[faces[:, 0]], side[faces[:, 1]])
    volumes = sorted(row["represented_volume_reference_m3"] for row in info["audit"]["components"])
    area = (high[1]-low[1])*(high[2]-low[2])
    assert np.allclose(volumes, sorted([(origin[0]-low[0])*area, (high[0]-origin[0])*area]), rtol=LOG_RTOL, atol=0)
    assert info["hermite"]["crossings_by_source"]["recorder"] == info["hermite"]["cut_links"] == len(log)
    assert info["hermite"]["crossings_by_source"]["midpoint_fallback"] == 0
    assert info["faces"]["cut_triangles"] == 4*len(log)        # two triangles per side per conflicting link


@pytest.mark.parametrize("normal", [(1., 2., 3.), (np.cos(np.radians(30)), 0., np.sin(np.radians(30))),
                                    (1., 1., 0.)])
def test_oblique_plane_cut_with_exact_hermite(normal):
    ijk, X, _, _ = box((12, 8, 7))
    normal = np.asarray(normal)/np.linalg.norm(normal)
    origin = np.array([.0003, -.0002, .0031])
    digits = plane_digits(X, origin, normal)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)
    assert info["audit"]["mesh_shells"] == 2 and info["mechanical_components"] == 2
    assert info["mesh_validity_reference"]["boundary_edges"] == 0
    on = cut_vertices(surf, faces)
    distance = np.abs((vertices[on]-origin)@normal)
    unclamped = ~surf.last["vertex_clamped"][on]
    assert distance[unclamped].max() < EXACT
    assert info["qef"]["clamped"] == int((~unclamped).sum())
    comp = surf.last["vertex_component"]
    assert len(np.unique(comp[faces[surf.last["face_component"] == 0]])) == 1
    assert info["audit"]["ok"]


def test_midpoint_fallback_is_labelled_and_axis_cut_lands_on_the_midplane():
    ijk, X, _, _ = box()
    origin, normal = np.array([.0013, 0, 0]), np.array([1., 0, 0])
    digits = plane_digits(X, origin, normal)[:, None]
    codes = pack_digits(digits)
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, codes, identity_F(len(X)))
    h = info["hermite"]
    assert h["source"] == "midpoint_fallback_R4" and h["log"]["given"] is False
    assert h["crossings_by_source"]["midpoint_fallback"] == h["cut_links"] and h["crossings_by_source"]["recorder"] == 0
    midplane = .001                                     # midway between the lattice columns x = 0.5 and 1.5 mm
    assert np.abs(vertices[cut_vertices(surf, faces), 0]-midplane).max() < EXACT
    # a partial log: the rest is labelled midpoint fallback, counted separately
    log = exact_log(ijk, X, digits, [(origin, normal)])
    partial = surface(ijk, X).extract(X, codes, identity_F(len(X)), cut_events_log=log[::2])[2]["hermite"]
    assert partial["source"] == "cut_events_v1"
    assert partial["crossings_by_source"]["recorder"] == len(log[::2])
    assert partial["crossings_by_source"]["midpoint_fallback"] == len(log)-len(log[::2])


def test_flags_match_cut_events_and_sign_inconsistent_entries_use_the_midpoint():
    ce = pytest.importorskip("robotic_cutting_sim.cut_events")
    assert cas.FLAG_SIGN_INCONSISTENT == ce.FLAG_SIGN_INCONSISTENT
    assert cas.FLAG_DEGENERATE_REFERENCE_NORMAL == ce.FLAG_DEGENERATE_REFERENCE_NORMAL
    assert LOG_DTYPE == ce.LOG_DTYPE
    ijk, X, _, _ = box((6, 4, 3))
    origin, normal = np.array([.0002, 0, 0]), np.array([1., 0, 0])
    digits = plane_digits(X, origin, normal)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    log["flags"][:3] = ce.FLAG_SIGN_INCONSISTENT
    log["t"][:3] = np.nan
    log["flags"][3:5] = ce.FLAG_DEGENERATE_REFERENCE_NORMAL
    info = surface(ijk, X).extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)[2]
    c = info["hermite"]["crossings_by_source"]
    assert c["recorder_sign_inconsistent_midpoint"] == 3 and c["recorder_point_fallback_normal"] == 2
    assert c["recorder"] == len(log)-5


# ----------------------------------------------------------------------------- partial cut / crack front

def test_partial_cut_has_a_closed_slit_tip_and_no_face_beyond_the_front():
    ijk, X, _, _ = box((12, 8, 5))
    origin, normal = np.array([0, .0002, 0]), np.array([0., 1, 0])
    front = 6                                              # columns i < 6 are assigned, the rest stays digit 0
    digits = np.where(ijk[:, 0] < front, plane_digits(X, origin, normal), 0)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)
    assert info["mechanical_components"] == 1 and info["audit"]["mesh_shells"] == 1
    assert info["audit"]["lattice_pieces"] == 1
    v = info["mesh_validity_reference"]
    assert v["boundary_edges"] == 0 and v["inconsistently_oriented_edges"] == 0 and v["degenerate_triangles"] == 0
    cut = surf.last["face_kind"] == cas.LINK_CUT
    assert cut.sum() == 4*len(log)
    # faces only on conflicting links: every cut triangle's link has its lower site in an assigned column
    lat = surf.lattice
    lower_i = lat.link_lower[surf.last["face_link"][cut], 0]-1+lat.low[0]
    assert lower_i.max() < front
    front_x = X[ijk[:, 0] == front, 0].min()
    assert vertices[np.unique(faces[cut]), 0].max() <= front_x+EXACT
    # tip: representatives shared by both sides' cut faces, in mixed-side regions at the front
    side_digit = digits[surf.last["face_side_particle"], 0]
    shared = np.intersect1d(faces[cut & (side_digit == 1)], faces[cut & (side_digit == 2)])
    assert len(shared) > 0
    assert np.all(vertices[shared, 0] >= X[ijk[:, 0] == front-1, 0].min()-EXACT)
    assert info["regions"]["mixed_side_regions"] > 0
    assert np.abs((vertices[shared]-origin)@normal).max() < EXACT


# ----------------------------------------------------------------------------- intersecting and parallel cuts

def test_two_intersecting_cuts_give_four_regions_and_a_sharp_edge():
    ijk, X, _, _ = box((10, 8, 4))
    planes = [(np.array([.0003, 0, 0]), np.array([1., 0, 0])), (np.array([0, -.0004, 0]), np.array([0., 1, 0]))]
    digits = np.stack([plane_digits(X, o, n) for o, n in planes], 1)
    log = exact_log(ijk, X, digits, planes)
    surf = surface(ijk, X, cuts=2)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)
    assert info["mechanical_components"] == 4 and info["audit"]["mesh_shells"] == 4
    assert_closed_manifold(info, shells=4)
    assert info["regions"]["multi_region_cells"] > 0
    on_both = (np.abs(vertices[:, 0]-.0003) < EXACT) & (np.abs(vertices[:, 1]+.0004) < EXACT)
    # one vertex per region (4) in every dual cell along the intersection line (z layers + boundary cells)
    assert on_both.sum() == 4*(4+1)
    assert len(np.unique(surf.last["vertex_component"][on_both])) == 4
    assert surf.last["vertex_rank"][on_both].min() >= 2


def test_four_parallel_cuts_give_five_closed_shells():
    ijk, X, low, high = box((15, 5, 4))
    cuts = [-.0042, -.0011, .0018, .0047]
    planes = [(np.array([c, 0, 0]), np.array([1., 0, 0])) for c in cuts]
    digits = np.stack([plane_digits(X, o, n) for o, n in planes], 1)
    log = exact_log(ijk, X, digits, planes)
    surf = surface(ijk, X, cuts=4)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)
    assert info["mechanical_components"] == 5 and info["audit"]["mesh_shells"] == 5
    assert_closed_manifold(info, shells=5)
    area = (high[1]-low[1])*(high[2]-low[2])
    bounds = [low[0], *cuts, high[0]]
    expected = sorted(area*(b-a) for a, b in zip(bounds[:-1], bounds[1:]))
    got = sorted(row["represented_volume_reference_m3"] for row in info["audit"]["components"])
    assert np.allclose(got, expected, rtol=LOG_RTOL, atol=0)


def test_multi_crossing_links_are_counted_not_repaired():
    ijk, X, _, _ = box((8, 4, 3))
    planes = [(np.array([.0001, 0, 0]), np.array([1., 0, 0])), (np.array([.0003, 0, 0]), np.array([1., 0, 0]))]
    digits = np.stack([plane_digits(X, o, n) for o, n in planes], 1)
    log = exact_log(ijk, X, digits, planes)
    surf = surface(ijk, X, cuts=2)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)), cut_events_log=log)
    links = 4*3
    assert info["hermite"]["multi_crossing_links"] == links and info["hermite"]["near_coincident_crossings"] == links
    assert info["audit"]["mesh_shells"] == 2
    lo_side = digits[surf.last["face_side_particle"], 0] == 1
    cut = surf.last["face_kind"] == cas.LINK_CUT
    # each side uses the crossing nearest its own endpoint: a zero-volume gap of 0.2 s stays visible
    assert np.allclose(vertices[np.unique(faces[cut & lo_side]), 0], .0001, atol=EXACT)
    assert np.allclose(vertices[np.unique(faces[cut & ~lo_side]), 0], .0003, atol=EXACT)


# ----------------------------------------------------------------------------- thin chips

def test_thin_one_particle_chip_is_a_closed_shell_with_its_mass():
    ijk, X, low, high = box((10, 6, 5))
    top = ijk[:, 2] == 4
    digits = np.where(top, 2, 1)[:, None]
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)))
    assert info["audit"]["mesh_shells"] == 2 and info["mechanical_components"] == 2
    assert_closed_manifold(info, shells=2)
    volumes = {row["particles"]: row["represented_volume_reference_m3"] for row in info["audit"]["components"]}
    assert np.isclose(volumes[top.sum()], top.sum()*S**3, rtol=1e-12, atol=0)
    assert np.isclose(volumes[(~top).sum()], (~top).sum()*S**3, rtol=1e-12, atol=0)


def test_single_particle_fragment_is_a_closed_cube():
    ijk, X, _, _ = box((5, 4, 3))
    lone = np.flatnonzero((ijk == [2, 1, 2]).all(1))[0]
    digits = np.ones(len(X), int)
    digits[lone] = 2
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, pack_digits(digits[:, None]), identity_F(len(X)))
    assert info["mechanical_components"] == 2 and info["audit"]["mesh_shells"] == 2
    assert_closed_manifold(info, shells=2)
    row = next(r for r in info["audit"]["components"] if r["particles"] == 1)
    assert np.isclose(row["represented_volume_reference_m3"], S**3, rtol=1e-12, atol=0)
    cube = surf.last["vertex_component"] == row["mechanical_component"]
    assert cube.sum() == 8
    assert np.allclose(np.abs(vertices[cube]-X[lone]), S/2, atol=EXACT)
    # R4: the signed directions of the 5 cut links cancel (non-planar pattern); the coherence guard uses each
    # link's own axis normal instead of collapsing the cube to rank-1 vertices (volume (s/3)^3)
    assert info["hermite"]["source"] == "midpoint_fallback_R4"
    assert info["hermite"]["fallback_normal_incoherent_links"] == 5 and info["hermite"]["cut_links"] == 5


# ----------------------------------------------------------------------------- embedding

def affine(name):
    if name == "compression":
        return np.diag([1.05, 1.05, .8])
    if name == "shear":
        return np.array([[1, .25, 0], [0, 1, .1], [0, 0, 1.]])
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec([.3, -.5, .7]).as_matrix()


@pytest.mark.parametrize("name", ["compression", "shear", "rotation"])
@pytest.mark.parametrize("with_F", [True, False])
def test_affine_motion_is_embedded_exactly(name, with_F):
    ijk, X, _, _ = box((10, 7, 6))
    normal = np.array([1., .4, .2])/np.linalg.norm([1., .4, .2])
    origin = np.array([.0004, 0, .003])
    digits = plane_digits(X, origin, normal)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    A, c = affine(name), np.array([.1, -.02, .3])
    x = X@A.T+c
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(x, pack_digits(digits), np.tile(A, (len(X), 1, 1)) if with_F else None,
                                         cut_events_log=log)
    expected = surf.last["vertex_reference"]@A.T+c
    assert np.abs(vertices-expected).max() < EXACT
    e = info["embedding"]
    assert e["estimator"] == ("total_F" if with_F else "lsq_affine")
    assert e["translation_only_representatives"] == 0
    if not with_F:
        assert e["lsq_translation_only_particles"] == 0


@pytest.mark.parametrize("with_F", [True, False])
def test_frame_rotation_maps_reference_frame_motion_to_world(with_F):
    from scipy.spatial.transform import Rotation
    ijk, X, _, _ = box((8, 6, 5))
    A, R, t = affine("shear"), Rotation.from_rotvec([0, 0, 1.1]).as_matrix(), np.array([.4, .1, .8])
    world = (X@A.T)@R.T+t                                    # positions in WORLD, F in the reference (local) frame
    surf = surface(ijk, X)
    vertices, _, info = surf.extract(world, np.zeros(len(X), np.int64), np.tile(A, (len(X), 1, 1)) if with_F else None,
                                     frame_rotation=R)
    assert np.abs(vertices-((surf.last["vertex_reference"]@A.T)@R.T+t)).max() < EXACT


def test_nonpositive_det_F_falls_back_to_translation_and_is_counted():
    ijk, X, _, _ = box((6, 5, 4))
    F = identity_F(len(X))
    F[:5] = -np.eye(3)
    shift = np.array([.01, 0, 0])
    surf = surface(ijk, X)
    vertices, _, info = surf.extract(X+shift, np.zeros(len(X), np.int64), F)
    assert info["embedding"]["det_F_nonpositive_particles"] == 5
    assert info["embedding"]["translation_only_representatives"] > 0
    assert np.abs(vertices-(surf.last["vertex_reference"]+shift)).max() < EXACT


def test_lsq_fallback_uses_translation_only_on_a_coplanar_layer():
    ijk, X, _, _ = box((6, 5, 1))                       # one layer: every neighbourhood is coplanar
    x = X+[0, 0, .001]
    surf = surface(ijk, X)
    vertices, _, info = surf.extract(x, np.zeros(len(X), np.int64), None)
    assert info["embedding"]["lsq_translation_only_particles"] == info["embedding"]["lsq_particles"]
    assert np.abs(vertices-(surf.last["vertex_reference"]+[0, 0, .001])).max() < EXACT


def test_recontact_never_welds_the_two_sides():
    ijk, X, _, _ = box((10, 6, 4))
    origin, normal = np.array([.0002, 0, 0]), np.array([1., 0, 0])
    digits = plane_digits(X, origin, normal)
    log = exact_log(ijk, X, digits[:, None], [(origin, normal)])
    codes = pack_digits(digits[:, None])
    surf = surface(ijk, X)
    apart = surf.extract(X, codes, identity_F(len(X)), cut_events_log=log)
    x = X.copy()
    x[digits == 2] -= [1.5*S, 0, 0]                     # push side 2 into side 1 (overlap)
    vertices, faces, info = surf.extract(x, codes, identity_F(len(X)), cut_events_log=log)
    assert np.array_equal(faces, apart[1]) and len(vertices) == len(apart[0])
    side = digits[surf.last["face_side_particle"]]
    assert set(faces[side == 1].ravel()).isdisjoint(faces[side == 2].ravel())
    for vertex in range(len(vertices)):
        assert len(np.unique(digits[surf.region_particles(vertex)])) == 1
    # the overlap is real: side-2 cut-face vertices lie inside side 1's extent
    cut2 = np.unique(faces[(surf.last["face_kind"] == cas.LINK_CUT) & (side == 2)])
    assert vertices[cut2, 0].max() < origin[0]-S
    assert info["audit"]["mesh_shells"] == 2


# ----------------------------------------------------------------------------- negatives and audit

def test_no_conflicting_codes_means_no_cut_faces_and_the_uncut_mesh():
    ijk, X, _, _ = box((10, 6, 4))
    surf = surface(ijk, X)
    uncut = surf.extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))
    # digits 1 | 0 | 2 with a one-column digit-0 band: no link conflicts (0-1, 0-2 are uncut)
    digits = np.select([ijk[:, 0] < 4, ijk[:, 0] == 4], [1, 0], 2)[:, None]
    vertices, faces, info = surf.extract(X, pack_digits(digits), identity_F(len(X)))
    assert info["faces"]["cut_triangles"] == 0 and info["hermite"]["cut_links"] == 0
    assert np.array_equal(vertices, uncut[0]) and np.array_equal(faces, uncut[1])
    assert info["mechanical_components"] == 1 and info["audit"]["mesh_shells"] == 1


def test_digit_zero_bridges_are_reported_never_hidden():
    ijk, X, _, _ = box((12, 6, 6))
    origin, normal = np.array([0, 0, 0]), np.array([1., 0, 0])
    digits = plane_digits(X, origin, normal)
    digits[ijk[:, 0] == 7] = 0                           # second layer on side 2: code-0 stencil bridge
    surf = surface(ijk, X)
    vertices, faces, info = surf.extract(X, pack_digits(digits[:, None]), identity_F(len(X)))
    audit = info["audit"]
    assert audit["mechanical_components"] == 1 and audit["lattice_pieces"] == 2 and audit["mesh_shells"] == 2
    assert audit["ok"] and audit["shell_component_violations"] == 0
    bridged = audit["visually_separated_mechanically_bridged"]
    assert len(bridged) == 1 and bridged[0]["lattice_pieces"] == 2
    assert audit["bridge_report"][0]["direct_code_zero_bridges"] > 0
    assert info["component_ranges"][0]["triangles"] == len(faces)


def test_shell_spanning_components_is_an_error():
    ijk, X, _, _ = box((6, 4, 3))
    nodes, weights, gradients = cas.reference_stencil(X, DX)
    nodes = np.arange(len(X))[:, None]*8+np.arange(8)          # no shared nodes: every particle is a component
    partition = CurrentMechanicalPartition(nodes, weights, gradients, 1, encoding="packed2")
    surf = cas.CutAwareSurface(partition, S, ijk, X)
    with pytest.raises(cas.CutAwareTopologyError):
        surf.extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))
    lenient = cas.CutAwareSurface(partition, S, ijk, X, strict_audit=False)
    info = lenient.extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))[2]
    assert not info["audit"]["ok"] and info["audit"]["errors"]


# ----------------------------------------------------------------------------- incremental, metadata, purity

def progressive_frames(ijk, X, origin, normal, columns):
    full = plane_digits(X, origin, normal)
    for front in columns:
        digits = np.where(ijk[:, 0] < front, full, 0)[:, None]
        yield pack_digits(digits), exact_log(ijk, X, digits, [(origin, normal)])


@pytest.mark.parametrize("use_log", [True, False])
def test_incremental_update_equals_full_rebuild(use_log):
    ijk, X, _, _ = box((12, 8, 5))
    origin, normal = np.array([0, .0003, 0]), np.array([.2, 1., .1])/np.linalg.norm([.2, 1., .1])
    x = X+.0002*np.sin(X*900)                           # non-affine current positions
    F = identity_F(len(X))
    incremental = surface(ijk, X)
    dirty = []
    for codes, log in progressive_frames(ijk, X, origin, normal, [0, 2, 3, 6, 6, 9, 12]):
        log = log if use_log else None
        a = incremental.extract(x, codes, F, cut_events_log=log)
        b = surface(ijk, X).extract(x, codes, F, cut_events_log=log)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert a[2]["qef"] == b[2]["qef"] and a[2]["audit"]["mesh_shells"] == b[2]["audit"]["mesh_shells"]
        dirty.append(a[2]["update"])
    assert dirty[0]["mode"] == "full" and all(d["mode"] == "incremental" for d in dirty[1:])
    assert dirty[4]["dirty_cells"] == 0                  # unchanged codes: nothing recomputed
    assert all(0 < d["dirty_cells"] < d["cells"] for d in dirty[1:4])


def test_metadata_keys_present():
    ijk, X, _, _ = box((6, 4, 3))
    info = surface(ijk, X).extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))[2]
    for key in ("surface_profile", "codes_sha256", "lattice_sha256", "reference_sha256", "stencil_sha256",
                "boundary_hermite_profile", "hermite", "qef", "embedding", "audit", "timings_s", "update",
                "component_ranges", "mesh_validity_reference", "mechanical_components"):
        assert key in info
    assert info["surface_profile"] == "cut_aware_surface_v1"
    import json
    json.dumps(info)                                       # episode-metadata ready (no NumPy scalars)
    assert info["qef"]["representatives_without_hermite"] == 0
    assert info["renderer_partition_changes_physics"] is False and info["writes_physics_state"] is False
    assert {"log_sha256", "entries"} <= set(info["hermite"]["log"])
    assert {"clamped", "truncation_relative", "rank_counts"} <= set(info["qef"])
    assert {"total_s", "topology_and_qef_s", "embedding_s"} <= set(info["timings_s"])
    record = cas.surface_selection_record("cut_aware_surface_v1", boundary_profile="cell_face_v1")
    assert record["surface_profile"] == "cut_aware_surface_v1" and record["baseline_profile"] == "particle_surface_v1"
    with pytest.raises(ValueError):
        cas.surface_selection_record("hybrid_surface_v1")


def test_inputs_are_never_modified():
    ijk, X, _, _ = box((8, 6, 4))
    origin, normal = np.array([.0002, 0, 0]), np.array([1., .3, 0])/np.linalg.norm([1., .3, 0])
    digits = plane_digits(X, origin, normal)[:, None]
    log = exact_log(ijk, X, digits, [(origin, normal)])
    arrays = {"ijk": ijk, "X": X, "codes": pack_digits(digits), "x": X*1.01, "F": identity_F(len(X))*1.01, "log": log}
    copies = {k: v.copy() for k, v in arrays.items()}
    for v in arrays.values():
        v.flags.writeable = False                         # any in-place write would raise
    surf = surface(arrays["ijk"], arrays["X"])
    surf.extract(arrays["x"], arrays["codes"], arrays["F"], cut_events_log=arrays["log"])
    surf.extract(arrays["x"], arrays["codes"], None, cut_events_log=arrays["log"])
    for k, v in arrays.items():
        assert np.array_equal(v, copies[k]) and v.tobytes() == copies[k].tobytes()


# ----------------------------------------------------------------------------- Layer 6

def test_face_varying_normals_keep_creases_and_regions():
    ijk, X, low, high = box((5, 4, 3))
    surf = surface(ijk, X)
    vertices, faces, _ = surf.extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))
    normals = surf.face_varying_normals(vertices, faces)
    face_n = cas.triangle_normals(vertices, faces)
    face_n /= np.linalg.norm(face_n, axis=1)[:, None]
    assert normals.shape == (3*len(faces), 3)
    assert np.allclose(normals, np.repeat(face_n, 3, axis=0), atol=1e-12)       # 90 degree creases stay sharp
    smooth = cas.face_varying_normals(vertices, faces, crease_angle_deg=100.)
    corner = np.flatnonzero(np.linalg.norm(vertices[faces.reshape(-1)]-low, axis=1) < EXACT)
    assert np.allclose(smooth[corner], -np.ones(3)/np.sqrt(3), atol=1e-12)   # 3 faces, equal area, averaged


def test_bridge_face_varying_method_stages_normals():
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge
    ijk, X, _, _ = box((4, 3, 2))
    surf = surface(ijk, X)
    vertices, faces, _ = surf.extract(X, np.zeros(len(X), np.int64), identity_F(len(X)))
    normals = surf.face_varying_normals(vertices, faces)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    bridge = DynamicSurfaceBridge(stage)
    receipt = bridge.update_face_varying_v1(vertices, faces, normals, .1)
    assert receipt["bridge_method"] == "update_face_varying_v1" and receipt["normals_interpolation"] == "faceVarying"
    assert bridge.mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    staged = np.asarray(bridge.mesh.GetNormalsAttr().Get())
    assert np.allclose(staged, normals, atol=1e-6)
    with pytest.raises(ValueError):
        bridge.update_face_varying_v1(vertices, faces, normals[:-1], .2)
    plain = bridge.update(vertices, faces, .3)                # the frozen method still clears/blocks normals
    assert "bridge_method" not in plain and bridge.mesh.GetNormalsAttr().Get() is None


# ----------------------------------------------------------------------------- helpers vs the solver

def test_reference_stencil_matches_tlmpm_v5_on_cpu():
    pytest.importorskip("warp")
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
    _, X, _, _ = box((6, 5, 4))
    X = (X+[.01, .02, .003]).astype(np.float32)
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    config = TLMPMConfig(cuts=1, blade_half_width=float(half), cutting_energy=5.)
    bed = TLMPM(X, np.full(len(X), 1e-3), S, config, "cpu", blade_profile_xz=polygon, cutting_law=8)
    nodes, weights, gradients = cas.reference_stencil(X, config.dx)
    assert np.array_equal(nodes, bed.nodes.numpy()) and np.array_equal(weights, bed.weights.numpy())
    assert np.array_equal(gradients, bed.gradients.numpy())


def test_lattice_neighbours_match_cut_events_table():
    ce = pytest.importorskip("robotic_cutting_sim.cut_events")
    ijk, X, _, _ = box((5, 4, 3))
    keep = np.arange(len(X)) % 7 != 3
    lattice = cas.ReferenceLattice(ijk[keep], X[keep], S)
    assert np.array_equal(lattice.neighbours(), ce.lattice_neighbours(ijk[keep]))


def test_make_component_surface_selects_by_profile():
    adapter = cas.make_component_surface("cut_aware_surface_v1", S, lattice=np.zeros((1, 3), int))
    assert isinstance(adapter, cas.CutAwareComponentSurface)
    with pytest.raises(ValueError):
        cas.make_component_surface("cut_aware_surface_v1", S)
    with pytest.raises(ValueError):
        cas.make_component_surface("unknown", S)


@pytest.mark.cpu_sim
def test_live_adapter_on_a_real_cpu_law8_cut_with_the_recorder():
    """End to end on Warp CPU: tlmpm_v5 law 8 vertical cut (test_cut_events ALIGNED), recorder log -> surface."""
    pytest.importorskip("warp")
    import test_cut_events as tce
    bed, recorder, _, _, _, points = tce.manual_run(**tce.ALIGNED)
    before = {name: getattr(bed, name).numpy().copy() for name in ("x", "v", "F", "code", "reference")}
    adapter = cas.make_component_surface("cut_aware_surface_v1", tce.SPACING, lattice=tce.lattice_of(points))
    vertices, faces, info = adapter.extract(bed, recorder)
    for name, value in before.items():
        assert np.array_equal(getattr(bed, name).numpy(), value), name     # reads only
    log = recorder.log()
    h = info["hermite"]
    assert h["source"] == "cut_events_v1" and h["log"]["log_sha256"] == recorder.sha256()
    assert h["crossings_by_source"]["recorder"] == h["cut_links"] == len(log) > 0
    assert h["log"]["entries_not_links"] == h["log"]["entries_not_conflicting"] == 0
    assert info["faces"]["cut_triangles"] == 4*h["cut_links"]
    assert info["audit"]["ok"] and info["surface_selection"]["surface_profile"] == "cut_aware_surface_v1"
    assert info["mesh_validity_reference"]["boundary_edges"] == 0
    cut = adapter.surface.last["face_kind"] == cas.LINK_CUT
    offset = np.abs(adapter.surface.last["vertex_reference"][np.unique(faces[cut]), 0]-tce.X0)/tce.SPACING
    print(f"live CPU cut: {len(vertices)} vertices, {len(faces)} triangles, shells {info['audit']['mesh_shells']}, "
          f"components {info['mechanical_components']}, reference cut-face |x - X0| P95 {np.percentile(offset, 95):.4f} s "
          f"(descriptive), clamped {info['qef']['clamped']}")
