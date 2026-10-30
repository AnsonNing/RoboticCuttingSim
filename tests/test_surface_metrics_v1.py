"""CPU tests of the WP-12 Phase 0 surface_metrics_v1 draft (no Newton, Warp or Isaac)."""
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import surface_metrics_v1 as M
from robotic_cutting_sim.cut_topology import ternary_to_packed

H = .001  # 1 mm lattice


def box(nx=10, ny=6, nz=6):
    lattice = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1).reshape(-1, 3)
    return lattice, lattice*H


def x_cut(lattice, at=4):
    """One cut digit between lattice x = at and at+1 (ternary digit 1 / 2)."""
    return np.where(lattice[:, 0] <= at, 1, 2)


def plane_mesh(x, ys, zs):
    """Triangulated rectangle x = const over the y/z grid (consistent winding)."""
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    vertices = np.c_[np.full(yy.size, x), yy.ravel(), zz.ravel()]
    ny, nz = yy.shape
    faces = []
    for i in range(ny-1):
        for j in range(nz-1):
            a, b, c, d = i*nz+j, (i+1)*nz+j, (i+1)*nz+j+1, i*nz+j+1
            faces += [(a, b, c), (a, c, d)]
    return vertices, np.asarray(faces)


def tetra():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def test_module_imports_without_newton_or_warp():
    code = ("import sys; import robotic_cutting_sim.surface_metrics_v1; "
            "print(any(m in sys.modules for m in ('newton', 'warp', 'pxr')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_lattice_links_and_exterior_counts():
    lattice, _ = box(10, 6, 6)
    pairs, axes = M.lattice_links(lattice)
    assert len(pairs) == 9*6*6+10*5*6+10*6*5
    assert np.bincount(axes).tolist() == [9*6*6, 10*5*6, 10*6*5]
    assert np.all(np.abs(lattice[pairs[:, 1]]-lattice[pairs[:, 0]]).sum(1) == 1)
    assert int((~M.exterior_mask(lattice)).sum()) == 8*4*4


def test_lattice_links_reject_duplicates():
    lattice, _ = box(3, 3, 3)
    with pytest.raises(ValueError):
        M.lattice_links(np.r_[lattice, lattice[:1]])


def test_cut_locus_side_points_on_separated_pieces():
    lattice, reference = box()
    codes = x_cut(lattice)
    gap = .002
    positions = reference+np.where(lattice[:, :1] >= 5, [[gap, 0, 0]], 0.)
    locus = M.cut_locus(reference, lattice, positions, codes, cuts=1)
    assert len(locus["pairs"]) == 6*6
    assert np.allclose(locus["midpoint"][:, 0], 4.5*H+gap/2)
    assert np.allclose(locus["side_points"][:, 0, 0], 4.5*H)        # piece x <= 4 keeps its face
    assert np.allclose(locus["side_points"][:, 1, 0], 4.5*H+gap)    # moved piece carries its face
    assert np.allclose(locus["direction"], [1, 0, 0])               # digit 1 -> 2 orientation


def test_packed_and_ternary_codes_give_the_same_locus():
    lattice, reference = box()
    codes = x_cut(lattice)+3*np.where(lattice[:, 1] <= 2, 1, 2)    # two cuts
    a = M.cut_locus(reference, lattice, reference, codes, encoding="ternary", cuts=2)
    b = M.cut_locus(reference, lattice, reference, ternary_to_packed(codes, 2), encoding="packed2", cuts=2)
    assert np.array_equal(a["pairs"], b["pairs"]) and np.array_equal(a["cut_digit"], b["cut_digit"])
    assert len(a["pairs"]) == 6*6+10*6


def test_ideal_cut_face_mesh_scores_zero_and_rotation_is_mapped_by_F():
    lattice, reference = box()
    codes = x_cut(lattice)
    vertices, faces = plane_mesh(4.5*H, np.linspace(1*H, 4*H, 7), np.linspace(1*H, 4*H, 7))
    r = Rotation.from_euler("xyz", [20, -35, 50], degrees=True)
    shift = np.array([.01, -.02, .03])
    positions = r.apply(reference)+shift
    F = np.tile(r.as_matrix(), (len(reference), 1, 1))
    labels = (lattice[:, 0] >= 5).astype(int)
    result = M.cut_face_metrics(reference, lattice, positions, codes, r.apply(vertices)+shift, faces, F=F,
                                component_labels=labels)
    assert result["status"] == "MEASURED" and result["cut_face_vertices"] == len(vertices)
    assert result["distance_to_side_locus_plane_m"]["max"] < 1e-9
    assert result["distance_to_link_midpoint_plane_m"]["max"] < 1e-9
    assert result["distance_to_side_locus_sample_m"]["max"] == pytest.approx(np.sqrt(.5)*H, rel=1e-6)
    assert result["normal_deviation_deg"]["max"] < 1e-4
    assert result["roughness_mesh_rms_m"]["max"] < 1e-9
    assert result["roughness_side_locus_rms_m"]["max"] < 1e-9


def test_offset_and_rough_mesh_are_detected():
    lattice, reference = box()
    codes = x_cut(lattice)
    ys = zs = np.linspace(1*H, 4*H, 13)
    vertices, faces = plane_mesh(4.5*H+.0003, ys, zs)              # 0.3 mm off the locus
    result = M.cut_face_metrics(reference, lattice, reference, codes, vertices, faces)
    assert result["distance_to_side_locus_plane_m"]["p50"] == pytest.approx(.0003, abs=1e-9)
    rng = np.random.default_rng(0)
    noisy = vertices.copy()
    noisy[:, 0] += rng.normal(0, .0001, len(noisy))
    rough = M.cut_face_metrics(reference, lattice, reference, codes, noisy, faces)
    assert 3e-5 < rough["roughness_mesh_rms_m"]["p50"] < 1.5e-4
    assert rough["normal_deviation_deg"]["p95"] > 1.


def test_no_cut_links_reported():
    lattice, reference = box(4, 4, 4)
    vertices, faces = tetra()
    result = M.cut_face_metrics(reference, lattice, reference, np.zeros(len(lattice), int), vertices*H, faces)
    assert result["status"] == "NO_CUT_LINKS" and result["cut_links"] == 0


def test_mesh_validity_counts():
    vertices, faces = tetra()
    ok = M.mesh_validity(vertices, faces)
    assert ok["watertight"] and ok["connected_components"] == 1
    assert ok["boundary_edges"] == ok["non_manifold_edges"] == ok["inconsistently_oriented_edges"] == 0
    flipped = faces.copy()
    flipped[0] = flipped[0, ::-1]
    assert M.mesh_validity(vertices, flipped)["inconsistently_oriented_edges"] == 3
    extra = np.r_[vertices, [[5, 5, 5]]]
    bad = np.r_[faces, [[0, 1, 2], [0, 0, 1]]]                      # extra fin + repeated index
    report = M.mesh_validity(extra, bad)
    assert report["non_manifold_edges"] >= 1 and report["repeated_index_triangles"] == 1
    assert report["degenerate_triangles"] >= 1 and report["unreferenced_vertices"] == 1 and not report["watertight"]
    two = M.mesh_validity(np.r_[vertices, vertices+3], np.r_[faces, faces+4])
    assert two["connected_components"] == 2 and two["watertight"]
    open_ = M.mesh_validity(vertices, faces[:3])
    assert open_["boundary_edges"] == 3 and not open_["watertight"]


def test_plane_fit_and_patch_roughness():
    rng = np.random.default_rng(1)
    points = np.c_[rng.uniform(0, .01, 400), rng.uniform(0, .01, 400), np.zeros(400)]
    assert M.plane_fit_rms(points) < 1e-12
    points[:, 2] = rng.normal(0, 1e-4, 400)
    assert M.plane_fit_rms(points) == pytest.approx(1e-4, rel=.15)
    summary = M.patch_roughness(points, .0015)
    assert summary["count"] > 0 and summary["p50"] == pytest.approx(1e-4, rel=.4)
    assert np.isnan(M.plane_fit_rms(points[:2]))


def test_plane_distance_sign():
    d = M.plane_distance([[0, 0, 1], [0, 0, -2]], [0, 0, 0], [0, 0, 2])
    assert d.tolist() == [1., -2.]


def test_oblique_ideal_mesh_measures_the_staircase_floor():
    """Documents the method floor: a mesh exactly on a 30 deg plane still scores the lattice staircase."""
    lattice, reference = box(20, 12, 12)
    n = np.array([np.cos(np.radians(30)), 0, np.sin(np.radians(30))])
    origin = np.array([9.7, 5.5, 5.5])*H
    codes = np.where((reference-origin)@n < 0, 1, 2)
    u = np.array([0., 1., 0.])
    w = np.cross(n, u)
    g = np.linspace(-3*H, 3*H, 25)
    uu, ww = np.meshgrid(g, g, indexing="ij")
    vertices = origin+uu.ravel()[:, None]*u+ww.ravel()[:, None]*w
    _, faces = plane_mesh(0., g, g)
    result = M.cut_face_metrics(reference, lattice, reference, codes, vertices, faces)
    assert result["cut_face_vertices"] == len(vertices)
    assert result["distance_to_side_locus_plane_m"]["p95"] < .3*H
    assert result["normal_deviation_deg"]["p50"] < 10.
    assert result["roughness_mesh_rms_m"]["max"] < 1e-12
    assert .05*H < result["roughness_side_locus_rms_m"]["p95"] < .3*H


def test_normals_use_side_locus_when_pieces_are_far_apart():
    lattice, reference = box()
    codes = x_cut(lattice)
    positions = reference+np.where(lattice[:, :1] >= 5, [[.03, 0, 0]], 0.)   # 30 mm apart (a pushed chip)
    vertices, faces = plane_mesh(4.5*H, np.linspace(1*H, 4*H, 7), np.linspace(1*H, 4*H, 7))
    result = M.cut_face_metrics(reference, lattice, positions, codes, vertices, faces,
                                component_labels=(lattice[:, 0] >= 5).astype(int))
    assert result["normal_deviation_deg"]["count"] == len(faces) and result["normal_deviation_deg"]["max"] < 1e-6
    assert result["distance_to_side_locus_plane_m"]["max"] < 1e-9
    assert result["distance_to_link_midpoint_sample_m"]["p50"] > .01


def test_grouped_roughness_does_not_fit_two_faces_as_one_plane():
    rng = np.random.default_rng(2)
    a = np.c_[np.zeros(300), rng.uniform(0, .01, 300), rng.uniform(0, .01, 300)]
    b = a+[.0005, 0, 0]
    points = np.r_[a, b]
    mixed = M.patch_roughness(points, .0015)
    grouped = M.patch_roughness(points, .0015, groups=np.r_[np.zeros(300), np.ones(300)])
    assert mixed["p50"] == pytest.approx(.00025, rel=.05) and grouped["max"] < 1e-12


def test_exterior_layers_grow_inward():
    lattice, _ = box(10, 8, 8)
    assert int((~M.exterior_mask(lattice, 1)).sum()) == 8*6*6
    assert int((~M.exterior_mask(lattice, 2)).sum()) == 6*4*4
    assert int((~M.exterior_mask(lattice, 3)).sum()) == 4*2*2


# ------------------------------------------------------------------ Phase 1 (final surface_metrics_v1)

def grid_box_mesh(lo, hi, h=.0005):
    """Closed, outward-wound, subdivided box surface (each face an (n x m) grid; shared corner/edge vertices)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.round((hi-lo)/h).astype(int), 1)
    xs, ys, zs = (np.linspace(lo[i], hi[i], n[i]+1) for i in range(3))
    verts, index, faces = [], {}, []

    def vid(p):
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        gu, gv = (xs, ys, zs)[u], (xs, ys, zs)[v]
        for side, value in ((0, lo[axis]), (1, hi[axis])):
            for i in range(len(gu)-1):
                for j in range(len(gv)-1):
                    q = []
                    for a, b in ((i, j), (i+1, j), (i+1, j+1), (i, j+1)):
                        p = np.zeros(3)
                        p[axis], p[u], p[v] = value, gu[a], gv[b]
                        q.append(vid(p))
                    t1, t2 = (q[0], q[1], q[2]), (q[0], q[2], q[3])
                    tri = np.array([verts[k] for k in t1])
                    nrm = np.cross(tri[1]-tri[0], tri[2]-tri[0])
                    outward = nrm[axis]*(1 if side else -1) > 0
                    faces += [t1, t2] if outward else [t1[::-1], t2[::-1]]
    return np.asarray(verts), np.asarray(faces)


def two_pieces(gap=.002, nx=12):
    lattice, reference = box(nx, 6, 6)
    codes = ternary_to_packed(x_cut(lattice, at=nx//2-1), 1)
    positions = reference+np.where(lattice[:, :1] >= nx//2, [[gap, 0, 0]], 0.)
    labels = (lattice[:, 0] >= nx//2).astype(int)
    lo, hi = reference.min(0)-H/2, reference.max(0)+H/2
    cut_x = (nx//2-.5)*H
    v1, f1 = grid_box_mesh(lo, [cut_x, hi[1], hi[2]])
    v2, f2 = grid_box_mesh([cut_x+gap, lo[1], lo[2]], hi+[gap, 0, 0])
    return lattice, reference, codes, positions, labels, np.r_[v1, v2], np.r_[f1, f2+len(v1)], cut_x


def test_material_locus_hermite_and_midpoint_sources():
    lattice, reference = box(10, 6, 6)
    n = np.array([np.cos(np.radians(30)), 0, np.sin(np.radians(30))])
    origin = reference.mean(0)+.0002
    d = (reference-origin)@n
    codes = np.where(d >= 0, 2, 1)
    pairs, cut = M.conflicting_links_per_cut(lattice, codes, cuts=1)
    a, b = pairs.T
    assert np.all(codes[a] == 1) and np.all(codes[b] == 2)
    from robotic_cutting_sim.cut_events import LOG_DTYPE
    t = d[a]/(d[a]-d[b])
    log = np.zeros(len(a), LOG_DTYPE)
    log["p"], log["q"], log["t"] = b, a, 1-t            # recorded from the other endpoint: matching is symmetric
    log["hermite_reference"] = reference[a]+t[:, None]*(reference[b]-reference[a])
    log["normal_reference"] = n
    hl = M.material_locus(reference, lattice, reference, codes, cuts=1, events=log)
    ml = M.material_locus(reference, lattice, reference, codes, cuts=1)
    assert hl["source"] == M.LOCUS_HERMITE and ml["source"] == M.LOCUS_MIDPOINT
    assert np.abs((hl["P"]-origin)@n).max() < 1e-9      # float32 log storage
    assert np.abs((ml["P"]-origin)@n).max() <= .5*H+1e-12
    assert np.allclose(hl["N"], n) and hl["normal_against_link"] == 0
    assert np.all(ml["N"]@n > .9)                                       # staircase normal estimate
    assert np.allclose(hl["side_points"][:, 0], hl["P"]) and np.allclose(hl["side_normals"][:, 0], n)


def test_material_locus_maps_points_and_normals_by_F():
    lattice, reference, codes, _, _, _, _, _ = two_pieces()
    r = Rotation.from_euler("xyz", [10, 20, -30], degrees=True)
    shear = np.array([[1, .2, 0], [0, 1, 0], [0, 0, 1.]])
    A = r.as_matrix()@shear
    positions = reference@A.T
    F = np.tile(A, (len(reference), 1, 1))
    loc = M.material_locus(reference, lattice, positions, codes, cuts=1, F=F)
    assert np.allclose(loc["side_points"][:, 0], loc["P"]@A.T) and np.allclose(loc["side_points"][:, 1], loc["P"]@A.T)
    expected = np.linalg.inv(A).T@np.array([1., 0, 0])
    assert np.allclose(loc["side_normals"][:, 0], expected/np.linalg.norm(expected))
    assert np.allclose(loc["side_normals"][:, 1], -loc["side_normals"][:, 0])


def test_region_rule_and_exact_two_piece_report():
    lattice, reference, codes, positions, labels, v, f, cut_x = two_pieces()
    loc = M.material_locus(reference, lattice, positions, codes, cuts=1)
    ext = M.exterior_samples(lattice, positions, H)
    cls = M.classify_mesh_vertices(v, positions, labels, loc, ext, spacing=H, core_margin_m=1.5*H)
    on_face = np.isclose(v[:, 0], cut_x) | np.isclose(v[:, 0], cut_x+.002)
    interior = (v[:, 1] > reference[:, 1].min()) & (v[:, 1] < reference[:, 1].max()) & \
               (v[:, 2] > reference[:, 2].min()) & (v[:, 2] < reference[:, 2].max())
    assert np.all(cls["kind"][on_face & interior] == 2)
    assert np.all(cls["kind"][~on_face] == 1) and cls["counts"]["unmatched"] == 0
    rep = M.surface_report(reference, lattice, positions, codes, v, f, spacing=H, labels=labels,
                           masses=np.full(len(reference), 1500*H**3), density=1500., cuts=1, F=None, locus=loc,
                           core_margin_m=1.5*H)
    assert rep["status"] == "MEASURED"
    assert rep["distance_to_side_locus_plane"]["max"] < 1e-12
    assert abs(rep["signed_bias"]["bias_toward_side2_m"]) < 1e-12 and abs(rep["signed_bias"]["thickness_bias_m"]) < 1e-12
    assert rep["coverage"] == 1. and rep["locus_to_mesh_distance"]["max"] < 1e-12
    assert rep["normal_error_deg_core"]["max"] < 1e-6 and rep["normal_outward_fraction_core"] == 1.
    assert rep["topology"]["shells_equal_components"] and rep["mesh_validity"]["watertight"]
    assert rep["mesh_validity"]["shells_negative_volume"] == 0 and rep["mesh_validity"]["shells_euler_not_2"] == 0
    assert rep["mesh_validity"]["self_intersection"]["intersecting_pairs_sampled"] == 0
    assert abs(rep["volume"]["total_relative_error_reference"]) < 1e-9
    assert rep["rim"]["corner_to_mesh_distance"]["max"] < 1e-12 and rep["rim"]["open_edges_near_junction"] == 0
    assert rep["per_cut"]["0"]["distance_to_side_locus_plane"]["max"] < 1e-12


@pytest.mark.parametrize("shift,thick", [(.0002, 0.), (0., .0003)])
def test_signed_bias_separates_shift_and_thickening(shift, thick):
    lattice, reference, codes, positions, labels, v, f, cut_x = two_pieces()
    moved = v.copy()
    face1 = np.isclose(v[:, 0], cut_x)
    face2 = np.isclose(v[:, 0], cut_x+.002)
    moved[face1, 0] += shift+thick/2       # side 1 face (outward +x)
    moved[face2, 0] += shift-thick/2       # side 2 face (outward -x)
    loc = M.material_locus(reference, lattice, positions, codes, cuts=1)
    rep = M.surface_report(reference, lattice, positions, codes, moved, f, spacing=H, labels=labels, cuts=1, locus=loc,
                           validity=False, core_margin_m=1.5*H)
    bias = rep["signed_bias"]
    assert bias["bias_toward_side2_m"] == pytest.approx(shift, abs=2e-5)
    assert bias["thickness_bias_m"] == pytest.approx(thick, abs=3e-5)


def test_point_triangle_distance_matches_dense_sampling():
    rng = np.random.default_rng(3)
    a, b, c = rng.normal(size=(3, 3))
    uv = rng.random((40000, 2))
    uv = np.where(uv.sum(1, keepdims=True) > 1, 1-uv, uv)
    samples = a+uv[:, :1]*(b-a)+uv[:, 1:]*(c-a)
    q = rng.normal(size=(30, 3))*2
    d, closest = M.point_triangle_distance(q, *(np.tile(x, (30, 1)) for x in (a, b, c)))
    brute = np.linalg.norm(q[:, None]-samples[None], axis=2).min(1)
    assert np.all(d <= brute+1e-12) and np.allclose(d, brute, atol=2e-2)
    assert np.allclose(np.linalg.norm(q-closest, axis=1), d)


def test_self_intersections_and_shells():
    v, f = grid_box_mesh([0, 0, 0], [.002, .002, .002], h=.001)
    clean = M.mesh_validity_v1(v, f)
    assert clean["self_intersection"]["intersecting_pairs_sampled"] == 0 and clean["shells"] == 1
    assert clean["shell_rows"][0]["euler_characteristic"] == 2 and clean["shell_rows"][0]["signed_volume_m3"] > 0
    v2 = v+[.0007, .0006, .0005]                           # an interpenetrating second shell (generic offset)
    both = M.mesh_validity_v1(np.r_[v, v2], np.r_[f, f+len(v)])
    assert both["shells"] == 2 and both["self_intersection"]["inter_shell_pairs_sampled"] > 0
    flipped = M.mesh_validity_v1(v, f[:, ::-1])
    assert flipped["shells_negative_volume"] == 1
    # one crossing pair of single triangles
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.2, .2, -.5], [.2, .2, .5], [.9, .9, 0.]])
    assert M.self_intersections(tri, np.array([[0, 1, 2], [3, 4, 5]]))["intersecting_pairs_sampled"] == 1


def test_topology_consistency_uses_provenance_over_proximity():
    lattice, reference, codes, positions, labels, v, f, _ = two_pieces()
    topo = M.topology_consistency(v, f, positions, labels)
    assert topo["shells_equal_components"] and topo["faces_nearer_other_component"] == 0
    count, shell = M.shells(v, f)
    face_component = shell.copy()
    # an extractor that built a shell from two components is caught by provenance
    bad = face_component.copy()
    bad[:5] = 1-bad[:5]
    assert M.topology_consistency(v, f, positions, labels, face_component=bad)["shells_spanning_components"] >= 1
    assert M.topology_consistency(v, f, positions, labels, face_component=face_component)["shells_spanning_components"] == 0


def test_temporal_stability_separates_motion_from_remeshing():
    lattice, reference, codes, positions, labels, v, f, _ = two_pieces()
    same = M.temporal_stability(v, f, v.copy(), f.copy(), positions, positions, spacing=H)
    assert same["same_topology"] and same["residual_after_transport"]["max"] == 0. and same["vertexwise_max_m"] == 0.
    step = np.array([.0007, 0, 0])
    moved = M.temporal_stability(v, f, v+step, f, positions, positions+step, spacing=H)
    assert moved["residual_after_transport"]["max"] < 1e-12 and moved["raw_jump"]["max"] > .0006
    assert moved["material_displacement"]["p50"] == pytest.approx(.0007)


def test_volume_report_reference_and_current():
    lattice, reference, codes, positions, labels, v, f, _ = two_pieces()
    F = np.tile(np.eye(3)*1.1, (len(reference), 1, 1))
    rep = M.volume_report(v, f, positions, labels, H**3, F)
    assert abs(rep["total_relative_error_reference"]) < 1e-9
    assert rep["total_relative_error_current"] == pytest.approx(1/1.331-1, rel=1e-9)


def test_local_plane_ignores_collinear_patches():
    line = np.c_[np.linspace(0, .003, 7), np.zeros(7), np.zeros(7)]
    d, n = M.local_plane_signed(np.array([[.0015, .0005, .0005]]), line, np.tile([0, 0, 1.], (7, 1)), .003)
    assert np.isnan(d[0]) and np.isnan(n[0]).all()


def test_patch_roughness_reports_quadratic_mean():
    rng = np.random.default_rng(1)
    points = np.c_[rng.uniform(0, .01, 400), rng.uniform(0, .01, 400), rng.normal(0, 1e-4, 400)]
    r = M.patch_roughness(points, .002)
    assert r["quadratic_mean"] == pytest.approx(1e-4, rel=.4)
