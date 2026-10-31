"""cut_debug_view (WP-12 Phase 1 debug viewer): layers, views, section cuts, PNG and PLY output (CPU, no Newton)."""
import numpy as np
import pytest

from robotic_cutting_sim import cut_debug_view as V
from robotic_cutting_sim import surface_metrics_v1 as M
from robotic_cutting_sim.cut_topology import ternary_to_packed

H = .001


def package(tmp_path):
    lattice = np.stack(np.meshgrid(np.arange(10), np.arange(6), np.arange(6), indexing="ij"), -1).reshape(-1, 3)
    reference = lattice*H
    codes = ternary_to_packed(np.where(lattice[:, 0] <= 4, 1, 2), 1)
    positions = reference+np.where(lattice[:, :1] >= 5, [[.002, 0, 0]], 0.)
    labels = (lattice[:, 0] >= 5).astype(int)
    loc = M.material_locus(reference, lattice, positions, codes, cuts=1)
    v = np.array([[0, 0, 0], [.009, 0, 0], [0, .005, 0], [0, 0, .005]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    q = np.tile(np.array([0, .7071068, 0, .7071068]), (5, 1))
    poses = np.c_[np.full(5, .0045), np.zeros(5), np.linspace(.01, -.002, 5), q]
    path = tmp_path/"pkg.npz"
    np.savez_compressed(path, schema="wp12_cut_debug_package_v1", spacing_m=H, cuts=1, reference=reference, positions=positions,
                        codes=codes, labels=labels, lattice=lattice, contact_work=np.zeros(len(codes)),
                        cut_progress=(lattice[:, 0] == 4).astype(float), hermite_P=loc["P"], hermite_N=loc["N"],
                        hermite_cut=loc["cut"], hermite_pairs=loc["pairs"], hermite_is_recorded=loc["hermite"],
                        side_points=loc["side_points"], poses=poses, pose_cuts=np.zeros(5, np.int8),
                        blade_polygon_xz=np.array([[0, 0], [-.01, .0007], [-.01, -.0007]]), blade_half_length_m=.03,
                        mesh_current_volume_v1_vertices=v, mesh_current_volume_v1_faces=f,
                        mesh_legacy_vertices=v, mesh_legacy_faces=f)
    return path


def test_every_layer_builds(tmp_path):
    pkg = V.load(package(tmp_path))
    kinds = {name: V.layer(pkg, name)["kind"] for name in V.LAYERS}
    assert kinds["particle_surface"] == kinds["rendered"] == "mesh" and kinds["sweep"] == "lines"
    assert len(V.layer(pkg, "contact")["points"]) == 36
    lab = V.layer(pkg, "labels")
    assert np.allclose(lab["colors"][0], V._hex(V.DIGIT[1]))
    assert V.layer(pkg, "cut_aware")["points"].shape == (2*36, 3)
    with pytest.raises(ValueError):
        V.layer(pkg, "nope")


def test_section_segments_of_a_tetrahedron():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    seg = V.section_segments(v, f, np.array([0, 0, .5]), np.array([0, 0, 1.]))
    assert len(seg) == 3 and np.allclose(seg[:, :, 2], .5)


def test_views_and_render_and_ply(tmp_path):
    pkg = V.load(package(tmp_path))
    for view in V.VIEWS:
        o, u, v, w = V.view_basis(view, pkg)
        assert np.allclose([u@v, u@w, v@w], 0, atol=1e-9)
    out = V.render(pkg, tmp_path/"p.png", layers=V.LAYERS, views=V.VIEWS, dpi=40)
    assert out.stat().st_size > 1000
    written = V.export_ply(pkg, tmp_path/"ply", layers=V.LAYERS)
    header = V.read_ply_header(written["components"])
    assert "element vertex 360" in header
    assert any(line.startswith("element face 4") for line in V.read_ply_header(written["particle_surface"]))


def test_cut_aware_hook_and_phase3_comparison_figures(tmp_path):
    path = package(tmp_path)
    z = dict(np.load(path))
    v, f = z["mesh_current_volume_v1_vertices"], z["mesh_current_volume_v1_faces"]
    np.savez_compressed(tmp_path/"pkg3.npz", **z, mesh_cut_aware_surface_v1_vertices=v+.0001,
                        mesh_cut_aware_surface_v1_faces=f)
    pkg = V.load(tmp_path/"pkg3.npz")
    assert set(pkg["meshes"]) == {"legacy", "current_volume_v1", "cut_aware_surface_v1"}
    g = V.layer(pkg, "cut_aware", provider="cut_aware_surface_v1")
    assert g["kind"] == "mesh" and np.allclose(g["vertices"], v+.0001) and len(g["faces"]) == 4
    missing = V.layer(V.load(path), "cut_aware", provider="cut_aware_surface_v1")
    assert missing["kind"] == "mesh" and not len(missing["faces"])
    out = V.render_comparison(pkg, tmp_path/"cmp.png", dpi=40)
    assert out.stat().st_size > 1000
    out = V.render_reconstruction_vs_particles(pkg, tmp_path/"rvp.png", dpi=40)
    assert out.stat().st_size > 1000
