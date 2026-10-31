"""surface_metrics_v2 (surface_protocol_v2 P1-P7) on synthetic zero-physics / affine states (CPU, no Warp)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim import surface_metrics_v1 as SM1
from robotic_cutting_sim import surface_metrics_v2 as SM2

ROOT = Path(__file__).resolve().parents[1]
S = 1e-3


def block(nx=12, ny=10, nz=10):
    """Lattice block, particles at (i + 1/2) s with x centred on 0: a cut plane x = 0 lies on a cell face."""
    g = np.mgrid[0:nx, 0:ny, 0:nz].reshape(3, -1).T
    ref = (g+.5)*S
    ref[:, 0] -= nx*S/2
    return g.astype(np.int64), ref


def two_pieces(shift_s):
    """Codes 1 (x < 0) / 2 (x > 0); piece 2 rigidly moved by ``shift_s`` along x; labels = pieces."""
    lat, ref = block()
    codes = np.where(ref[:, 0] > 0, 2, 1).astype(np.int64)
    pos = ref.copy()
    pos[codes == 2, 0] += shift_s*S
    labels = (codes == 2).astype(np.int64)
    F = np.tile(np.eye(3), (len(ref), 1, 1))
    locus = SM1.material_locus(ref, lat, pos, codes, cuts=1, F=F)
    ext = SM1.exterior_samples(lat, pos, S, F)
    return lat, ref, pos, codes, labels, F, locus, ext


def test_vertex_component_from_ranges_maps_rank_and_marks_uncovered():
    ranges = [{"mechanical_component": 1, "first_vertex": 0, "vertices": 3},
              {"mechanical_component": 0, "first_vertex": 3, "vertices": 2}]
    out = SM2.vertex_component_from_ranges(6, ranges, rank=np.array([1, 0]))
    assert out.tolist() == [0, 0, 0, 1, 1, -1]


def test_region_rule_v2_takes_component_from_provenance():
    # pieces 0.6 s apart: a face drawn at the protocol's s/2 boundary is nearer the other piece's particles
    lat, ref, pos, codes, labels, F, locus, ext = two_pieces(-.4)
    y, z = np.meshgrid(np.arange(3, 7)*S+.5*S, np.arange(3, 7)*S+.5*S)
    v = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]        # on piece 0's side locus q_a = x_a + s/2 = 0
    c1 = SM1.classify_mesh_vertices(v, pos, labels, locus, ext, spacing=S)
    c2 = SM2.classify_mesh_vertices_v2(v, np.zeros(len(v), np.int64), pos, labels, locus, ext, spacing=S)
    assert (c1["component"] == 1).all()                       # v1: nearest particle is piece 1 (0.1 s away)
    assert (c2["component"] == 0).all() and (c2["kind"] == 2).all() and (c2["side"] == 0).all()
    assert c2["counts"]["cut_face_component_differs_from_nearest_particle"] == len(v)
    region, signed, vkey = SM2.cut_face_signed_distances(v, c2, locus, labels, 1)
    assert np.nanmax(np.abs(signed)) < 1e-9
    _, signed1, _ = SM2.cut_face_signed_distances(v, {**c2, "component": c1["component"], "side": c1["side"],
                                                      "row": c1["row"]}, locus, labels, 1)
    assert np.nanmin(np.abs(signed1)) > .3*S                  # the v1 assignment measures the opposite face


def test_margin_excludes_junction_vertices_from_max_only():
    lat, ref, pos, codes, labels, F, locus, ext = two_pieces(3.)
    centre = np.array([0., 5*S, 5*S])
    rim = np.array([0., .2*S, 5*S])                           # near the y = 0 exterior face: two sets within s/2
    v = np.vstack([centre+[0, dy*S, dz*S] for dy in (-1, 0, 1) for dz in (-1, 0, 1)]+[rim+[.0004, 0, 0]])
    cls = SM2.classify_mesh_vertices_v2(v, np.zeros(len(v), np.int64), pos, labels, locus, ext, spacing=S)
    assert (cls["kind"] == 2).all()
    assert cls["unambiguous"][:-1].all() and not cls["unambiguous"][-1]
    region, signed, _ = SM2.cut_face_signed_distances(v, cls, locus, labels, 1)
    rows = SM2.distance_rows(signed, region, cls, locus, spacing=S)
    assert rows["ambiguous_vertices"] == 1
    assert rows["max_all_s"] == pytest.approx(.4, abs=1e-6)  # the offset rim vertex
    assert rows["max_unambiguous_s"] < 1e-6                    # max over unambiguous vertices only
    assert rows["all"]["count"] == len(v)                      # P95 rows keep every vertex


def test_current_floor_leave_one_out_and_group_rows():
    lat, ref, pos, codes, labels, F, locus, ext = two_pieces(3.)
    res, keys = SM2.current_distance_floor(locus, labels, 1)
    assert np.nanmax(res) < 1e-12                               # planar locus: zero floor
    rng = np.random.default_rng(0)
    rough = dict(locus, side_points=locus["side_points"].copy())
    rough["side_points"][..., 0] += rng.normal(0, .2*S, rough["side_points"].shape[:2])
    res_r, _ = SM2.current_distance_floor(rough, labels, 1)
    assert .1 < np.nanpercentile(res_r, 95)/S < .8
    # rows: two groups of 30 vertices, one inside floor + 0.10 s, one outside; a small group is pooled
    vkey = np.r_[np.zeros(30, int), np.ones(30, int), np.full(5, 2)]
    signed = np.r_[np.full(30, .15), np.full(30, .5), np.full(5, .05)]*S
    floor = np.r_[np.full(40, .1), np.full(40, .1), np.full(10, .1)]*S
    fkeys = np.r_[np.zeros(40, int), np.ones(40, int), np.full(10, 2)]
    out = SM2.current_p95_rows(signed, vkey, floor, fkeys, spacing=S)
    by = {g["group"]: g for g in out["groups"]}
    assert by[0]["pass"] and not by[1]["pass"] and by["pooled_small_groups"]["pass"]
    assert by[0]["limit_s"] == pytest.approx(.2) and out["all_groups_pass"] is False
    assert out["worst_excess_s"] == pytest.approx(.3)


def face_grid_mesh(points_yz_grid):
    """Triangulated n x n grid of points (row-major)."""
    n = int(round(np.sqrt(len(points_yz_grid))))
    f = []
    for i in range(n-1):
        for j in range(n-1):
            a, b, c, d = i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1
            f += [(a, b, d), (a, d, c)]
    return np.asarray(f, np.int64)


def test_matched_scale_normal_and_split_half_floor():
    lat, ref, pos, codes, labels, F, locus, ext = two_pieces(3.)
    y, z = np.meshgrid(np.arange(1, 9)*S+.5*S, np.arange(1, 9)*S+.5*S, indexing="ij")
    v = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]
    f = face_grid_mesh(v)
    cls = SM2.classify_mesh_vertices_v2(v, np.zeros(len(v), np.int64), pos, labels, locus, ext, spacing=S)
    ang, cut, tri = SM2.matched_scale_normal(v, f, cls, locus, labels, 1)
    assert len(tri) > 0 and np.nanmax(ang) < 1e-6
    floor, fcut = SM2.normal_floor_split_halves(locus, labels, 1, ext, core_margin_m=3*S)
    assert len(floor) and np.max(floor) < 1e-6
    # facet roughness: zig-zag the mesh by +-0.2 s; triangle normals tilt, the 3 mm patch plane hardly does
    vz = v.copy()
    vz[:, 0] += .2*S*np.where((np.arange(len(v)) % 2) == 0, 1, -1)
    cz = SM2.classify_mesh_vertices_v2(vz, np.zeros(len(v), np.int64), pos, labels, locus, ext, spacing=S)
    angz, _, triz = SM2.matched_scale_normal(vz, f, cz, locus, labels, 1)
    nt, _, _ = SM1.triangle_normals(vz, f)
    tri_ang = np.degrees(np.arccos(np.clip(np.abs(nt[triz][:, 0]), 0, 1)))
    assert np.nanpercentile(tri_ang, 95) > 15 and np.nanpercentile(angz, 95) < 5
    row = SM2.normal_row(angz, floor)
    assert row["limit_deg"] == pytest.approx(row["floor_p95_deg"]+1) and row["pass"] is (row["mesh_p95_deg"] <= row["limit_deg"])
    again, _ = SM2.normal_floor_split_halves(locus, labels, 1, ext, core_margin_m=3*S)
    assert np.array_equal(floor, again)                        # declared seed: deterministic


def test_reference_consistency_affine_consistent_and_collapse_inconclusive():
    lat, ref = block(8, 8, 8)
    codes = np.zeros(len(ref), np.int64)
    labels = np.zeros(len(ref), np.int64)
    A = np.array([[.9, .1, 0], [0, 1.05, 0], [0, 0, .95]])
    pos = ref@A.T
    F = np.tile(A, (len(ref), 1, 1))
    vol = np.full(len(ref), S**3)
    ok = SM2.reference_consistency(lat, pos, F, codes, labels, vol, spacing=S)
    assert ok["status"] == "CONSISTENT" and abs(ok["mean_V_over_J_all"]-1) < 1e-9
    assert SM2.volume_current_row(.01, ok["status"], .02)["verdict"] == "PASS"
    assert SM2.volume_current_row(.03, ok["status"], .02)["verdict"] == "FAIL"
    # positions collapse in a layer while F stays identity (the review F3 situation)
    pos2 = ref.copy()
    layer = lat[:, 0] >= 4
    pos2[layer, 0] -= .5*S
    bad = SM2.reference_consistency(lat, pos2, np.tile(np.eye(3), (len(ref), 1, 1)), codes, labels, vol, spacing=S)
    assert bad["status"] == "INCONSISTENT"
    row = SM2.volume_current_row(.001, bad["status"], .02)
    assert row["verdict"] == SM2.INCONCLUSIVE and row["verdict"] != "PASS"
    # a component too small to verify holding > 1 % of the volume: UNVERIFIED -> inconclusive
    lab3 = labels.copy()
    lab3[(lat[:, 0] == 0) & (lat[:, 1] == 0)] = 1
    small = SM2.reference_consistency(lat, pos, F, codes, lab3, vol, spacing=S)
    assert small["components"]["1"]["status"] == "UNVERIFIED" and small["status"] == "UNVERIFIED"
    assert SM2.volume_current_row(.0, small["status"], .02)["verdict"] == SM2.INCONCLUSIVE
    assert SM2.component_status(small, [0]) == "CONSISTENT"


def test_cells_with_a_conflicting_edge_are_excluded():
    lat, ref = block(6, 4, 4)
    codes = np.where(ref[:, 0] > 0, 2, 1).astype(np.int64)
    out = SM2.reference_consistency(lat, ref, np.tile(np.eye(3), (len(ref), 1, 1)), codes, np.zeros(len(ref), int),
                                    np.full(len(ref), S**3), spacing=S, min_cells=1)
    assert out["cells"] == len(SM2.lattice_cells(lat))-3*3      # the 3 x 3 cells straddling x = 0 are cut


def test_slow_frames_and_idw_transport():
    lat, ref = block(6, 6, 6)
    codes = np.zeros(len(ref), np.int64)
    labels = np.zeros(len(ref), np.int64)
    frames = [{"positions": ref, "codes": codes}, {"positions": ref+[.05*S, 0, 0], "codes": codes},
              {"positions": ref+[.3*S, 0, 0], "codes": codes}, {"positions": ref+[.31*S, 0, 0], "codes": codes+1}]
    ok, rows = SM2.slow_frame_pairs(frames, spacing=S)
    assert ok == [(0, 1)] and not rows[1]["eligible"] and not rows[2]["codes_equal"]
    v = ref[[0, 1, 6, 7, 36, 37]]+.3*S                          # every vertex referenced by a face
    f = np.array([[0, 1, 2], [3, 4, 5]])
    moved = SM2.transport_idw(v, np.zeros(len(v), int), ref, ref+[.05*S, .01*S, 0], labels)
    assert np.allclose(moved-v, [.05*S, .01*S, 0])
    ts = SM2.temporal_stability_v2(v, f, v+[.05*S, 0, 0], f, ref, ref+[.05*S, 0, 0], np.zeros(len(v), int), labels, spacing=S)
    assert ts["residual_after_transport"]["max"] < 1e-12


def crossing_mesh():
    """Two vertex-disjoint triangles that cross (plus a third far away)."""
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [.5, .5, -1], [.5, .5, 1], [1.5, -1, 0],
                  [9, 9, 9], [10, 9, 9], [9, 10, 9]], float)*S
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    return v, f


def test_intersecting_pairs_matches_v1_count():
    v, f = crossing_mesh()
    pairs = SM2.intersecting_pairs(v, f)
    assert pairs.tolist() == [[0, 1]]
    assert SM1.self_intersections(v, f)["intersecting_pairs_sampled"] == len(pairs)
    rng = np.random.default_rng(1)
    vr = rng.random((300, 3))*.01
    fr = rng.integers(0, 300, (400, 3))
    fr = fr[(fr[:, 0] != fr[:, 1]) & (fr[:, 1] != fr[:, 2]) & (fr[:, 0] != fr[:, 2])]
    assert len(SM2.intersecting_pairs(vr, fr)) == SM1.self_intersections(vr, fr)["intersecting_pairs_sampled"]


def test_attribute_intersections_physical_vs_unexplained():
    v, f = crossing_mesh()
    pairs = SM2.intersecting_pairs(v, f)
    positions = np.array([[.5, .5, .1], [.5, .5, -.1], [20, 20, 20], [21, 20, 20]])*S
    codes = np.zeros(4, np.int64)
    cls = {"kind": np.ones(len(v), np.int8), "row": np.full(len(v), -1), "side": np.full(len(v), -1)}
    empty = {"pairs": np.zeros((0, 2), int), "cut": np.zeros(0, int), "side_points": np.zeros((0, 2, 3)),
             "N": np.zeros((0, 3))}
    # different pieces with particles 0.2 s apart at the pair: physical
    out = SM2.attribute_intersections(v, f, pairs, np.array([0, 1, 1]), cls, positions, np.array([0, 1, 0, 1]), codes,
                                      empty, spacing=S)
    assert out["physical_different_pieces"] == 1 and out["unexplained"] == 0
    # the same pieces far apart (gap 1 s at the pair: not < 1 s): unexplained
    far = positions.copy()
    far[1] = far[0]+[0, 0, -1.5*S]
    out = SM2.attribute_intersections(v, f, pairs, np.array([0, 1, 1]), cls, far, np.array([0, 1, 0, 1]), codes, empty,
                                      spacing=S)
    assert out["unexplained_different_pieces_apart"] == 1
    # one component, not opposite sides of a cut (a fold): always unexplained, whatever the gap
    out = SM2.attribute_intersections(v, f, pairs, np.array([0, 0, 0]), cls, positions, np.zeros(4, int), codes, empty,
                                      spacing=S)
    assert out["unexplained_same_component"] == 1 and out["physical"] == 0
    # one component, opposite sides of cut 0 whose locus row has crossed sides: physical
    cls_cut = {"kind": np.full(len(v), 2, np.int8), "row": np.zeros(len(v), int),
               "side": np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])}
    pos2 = np.array([[.5, .5, 3], [.5, .5, -3], [20, 20, 20], [21, 20, 20]])*S
    codes2 = np.array([1, 2, 1, 2], np.int64)
    loc = {"pairs": np.array([[0, 1]]), "cut": np.array([0]), "N": np.array([[0, 0, -1.]]),
           "side_points": np.array([[[.5, .5, -.2], [.5, .5, .2]]])*S}
    out = SM2.attribute_intersections(v, f, pairs, np.array([0, 0, 0]), cls_cut, pos2, np.zeros(4, int), codes2, loc,
                                      spacing=S)
    assert out["physical_opposite_sides"] == 1


def test_faces_nearer_other_component_split_by_gap():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)*S
    f = np.array([[0, 1, 2]])
    near = np.array([[-.6, 0, 0], [.2, 0, 0]])*S                 # own particle 0.6 s away, other 0.2 s: gap 0.8 s
    out = SM2.faces_nearer_other_component_v2(v, f, near, np.array([0, 1]), np.array([0]), spacing=S)
    assert out == {"faces_nearer_other_component": 1, "physical": 1, "unexplained": 0, "gap_m": S}
    out = SM2.faces_nearer_other_component_v2(v, f, np.array([[-1.2, 0, 0], [.2, 0, 0]])*S, np.array([0, 1]),
                                              np.array([0]), spacing=S)
    assert out["unexplained"] == 1


def test_chip_volume_v2_lattice_part_and_fallback_by_construction():
    # chip = two unit tetrahedra-like closed meshes; face 0-3 lattice-faced, 4-7 fallback
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)*S
    tf = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    v = np.vstack([tet, tet+5*S])
    f = np.vstack([tf, tf+4])
    vt = S**3/6
    labels = np.array([0, 1, 1, 1])
    pv = np.array([1., vt, vt/2, vt/2])
    F = np.tile(np.eye(3), (4, 1, 1))
    ff = np.r_[np.zeros(4, bool), np.ones(4, bool)]
    fp = np.array([False, False, True, True])
    row = SM2.chip_volume_v2(v, f, np.ones(8, int), labels, pv, F, [1], limit=.05, status="CONSISTENT",
                             fallback_faces=ff, fallback_particles=fp)
    assert row["verdict"] == "PASS" and row["value"] == pytest.approx(0, abs=1e-9)
    assert row["particle_cell_fallback_by_construction"]["faces"] == 4
    assert SM2.chip_volume_v2(v, f, np.ones(8, int), labels, pv, F, [1], limit=.05, status="UNVERIFIED")["verdict"] == SM2.INCONCLUSIVE


def test_surface_metrics_v1_unchanged():
    digest = hashlib.sha256((ROOT/"src/robotic_cutting_sim/surface_metrics_v1.py").read_bytes()).hexdigest()
    assert digest == "639a6ee92d225622e2c8769aa28fe396531913e335053d369151d47b7a756f64"


# ------------------------------------------------------------------------------------------------ protocol v2 file

def protocols():
    v1 = json.loads((ROOT/"configs/wp12/surface_protocol_v1.json").read_text())
    v2 = json.loads((ROOT/"configs/wp12/surface_protocol_v2.json").read_text())
    return v1, v2


def test_protocol_v1_file_still_frozen():
    digest = hashlib.sha256((ROOT/"configs/wp12/surface_protocol_v1.json").read_bytes()).hexdigest()
    assert digest == "afc30c752375c7e4158c66cbf646239943a3b6e6c72f24c5972bbe76defdeeee"


def test_protocol_v2_changes_are_justified_and_implemented():
    _, v2 = protocols()
    assert v2["schema"] == "wp12_surface_protocol_v2"
    keys = sorted(v2["changes"])
    assert [k[:2] for k in keys] == ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    for k in keys:
        c = v2["changes"][k]
        assert len(c["definition"]) > 100 and len(c["justification"]) > 100
        for name in c["implementation"].replace("surface_metrics_v2.", "").split(", "):
            assert callable(getattr(SM2, name)), name
    assert v2["changes"]["P3_current_distance_floor"]["parameters"]["offset_s"] == .10
    assert v2["changes"]["P5_slow_motion_made_slow"]["parameters"]["threshold_p95_s"] == .05
    assert v2["changes"]["P1_region_rule_v2"]["parameters"]["margin_s"] == .5


def test_protocol_v2_unchanged_rows_equal_v1():
    v1, v2 = protocols()
    t1, t2 = v1["thresholds_for_candidate_B"], v2["thresholds_for_candidate_B"]
    for case in ("S0_uncut_box", "G2_static_unit_test", "S6_four_slices", "S9_negative_controls"):
        assert t2[case] == t1[case], case
    for row in ("reference_p95_s", "reference_max_s", "current_max_s", "abs_bias_toward_side2_s", "abs_thickness_bias_s",
                "coverage_min", "rim_corner_p95_s", "closed", "shells_equal_components"):
        assert t2["S1_flat_cut"][row] == t1["S1_flat_cut"][row], row
    for row in ("cut_faces_on_non_conflicting_links", "coverage_of_conflicting_links_min", "mechanical_components",
                "slit_extent_equals_conflicting_extent", "tip_representatives_shared"):
        assert t2["S4_partial_cut"][row] == t1["S4_partial_cut"][row]
    assert t2["S2_deformed_cut"]["current_max_s"] == t1["S2_deformed_cut"]["current_max_s"]
    assert t2["S5_intersecting_cuts"]["slots_per_node"] == t1["S5_intersecting_cuts"]["slots_per_node"]
    assert t2["S7_thin_chip"]["skipped_fragments"] == 0 and t2["S7_thin_chip"]["chip_current_volume_rel_error"] == .05
    assert t2["all"]["volume_current_rel"] == t1["all"]["volume_current_rel"]
    assert t2["all"]["temporal_slow_motion_residual_p95_s"] == t1["all"]["temporal_slow_motion_residual_p95_s"]
    assert v2["measurement_uncertainty"]["gpu_nondeterminism_band_m"] == v1["measurement_uncertainty"]["gpu_nondeterminism_band_m"]
    assert v2["phase3_comparison_gate_v2"]["bands"] == {"distance_s": .06, "normal_deg": 1., "volume_rel": .005,
                                                        "counts_booleans_coverage": v2["phase3_comparison_gate_v2"]["bands"]["counts_booleans_coverage"]}
    assert v2["phase3_comparison_gate_v2"]["smoothness_is_never_a_success_criterion"] is True


def test_protocol_v2_matrix_and_held_out_states():
    _, v2 = protocols()
    m = v2["required_matrix"]
    assert m["methods"] == ["A", "B_v1", "B_v2"] and len(m["protocols"]) == 2 and len(m["state_sets"]) == 2
    ho = v2["held_out_states"]
    seeds = []
    for case in ("S1_G3a", "S1_G3b", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"):
        for h in ("H1", "H2"):
            seeds.append(ho[case][h]["seed"])
            if "offset_s" in ho[case][h]:
                u = .15+.7*np.random.default_rng(ho[case][h]["seed"]).random(2)
                assert np.allclose(np.round(u, 3), ho[case][h]["offset_s"]), (case, h)
    assert len(set(seeds)) == len(seeds) and 42 not in seeds
    assert "35 deg" in ho["S1_G3b"]["H2"]["geometry"] and "DENSE" in ho["S8"]["H1"]["geometry"]
    assert "rake 35 deg" in ho["S7"]["H2"]["geometry"]


def test_gate_metrics_v2_replace_the_v1_rows():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wp12_phase3_surface", ROOT/"scripts/run_wp12_phase3_surface.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    replaced = {"current_max_s", "reference_max_s", "normal_p50_deg", "normal_p95_deg", "self_intersections_sampled",
                "faces_nearer_other_component", "volume_current_abs_rel"}
    assert set(D.GATE_METRICS)-replaced <= set(SM2.GATE_METRICS_V2)
    assert not replaced & set(SM2.GATE_METRICS_V2)
    assert "triangle_normal_p95_deg" in SM2.REPORT_ONLY_V2
