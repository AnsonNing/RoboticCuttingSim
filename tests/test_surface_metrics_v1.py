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
