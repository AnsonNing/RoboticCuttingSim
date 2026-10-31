#!/usr/bin/env python3
"""WP-12 Phase 3: read-only attribution of the cut_aware_surface_v1 (B) metric outliers on saved Phase 3 states.

Reads ``<run>/<case>_states.npz`` written by ``scripts/run_wp12_phase3_surface.py`` (final frame), re-extracts B
with the recorder log and total F (deterministic, identical to the run), and reports per case:

* ``vertex_categories``: region_rule_v1 cut-face vertices split by B provenance of their dual cell (cut-only, crease
  = cell with both cut and boundary links, boundary-only = an exterior representative the region rule assigns to the
  cut face, mixed = crack-tip representative whose region holds both side digits), with reference/current distances
  and mean outward offsets per category;
* ``self_intersections``: all intersecting triangle pairs (no sampling) of B's current mesh and, per mechanical
  component, of its pre-embedding (reference) mesh; face kinds; pairs touching crease / clamped vertices;
* ``volume_consistency``: current volume of every fully linked lattice cell (6 tetrahedra on the particle positions)
  vs the mean det F of its corners, i.e. whether sum (m/rho) det F agrees with the particle positions;
* ``uncut_link_stretch``: current length of uncut lattice links per mechanical component and |dx - F dX|;
* ``normals``: core cut triangles vs the local locus plane, the local plane of B's own vertices, and the angle
  between the two planes (3 mm); quad non-planarity.

``--review-checks`` (default on) adds the read-only checks of the Phase 3 decision review
(``docs/reports/WP12_PHASE3_DECISION_REVIEW.md``, commit b67e80c), folded in from the reviewer's ad-hoc scripts so
they are reproducible from the repository (``review_checks`` per case, ``s8_frame_motion``, ``data_reads``):

* F1 ``component_assignment`` / ``side_assignment``: region_rule_v1 cut-face vertices whose nearest-particle component
  differs from B's extractor provenance (``vertex_component``), and vertices whose (cut, side) differs from B's own cut
  face (side particle digit); current P95 with and without them;
* F2 ``tip_weights`` (S4): per crack-tip region the side-1 / side-2 / code-0 corners, the restricted trilinear weight of
  each group at X_r, clamping, the opening along N and the vertex offset from side 1; the current length of the
  code-0 links along N inside the tip regions;
* F3 ``volume_by_distance``: V / det F of fully linked cells binned by the reference distance of the cell centre to the
  nearest Hermite point; share of the deficit per bin;
* F4 ``locus_floor``: residual of the side-locus samples to their own 2 mm plane (core, sample inside the fit) and the
  triangle-normal error, against the 3 mm locus plane, of a Delaunay mesh built exactly through the locus samples;
* F6 ``s5_region_rule`` (S5): vertices with reference error > 0.5 s, split into exterior representatives (no cut face)
  and wrong cut digit at a junction, with the error against B's own face digit;
* F7 ``chip_codes`` (S7): codes of the chip particles;
* F10 ``s8_frame_motion``: per S8 snapshot pair, the material displacement P95 and the stored slow-motion residuals;
* ``data_reads``: G2 clamp numbers (F5), the S6 A/B current P95 (F1) and the B_mid S8 rows re-derived with the current
  driver (F11, NOT_RUN).

``REVIEW_EXPECTED`` holds the numbers as printed in the review; ``review_reproduction`` compares each with the measured
value at the review's printed precision and lists every number that does not reproduce.

  .venv/bin/python scripts/diagnose_wp12_phase3_surface.py --run artifacts/wp12/phase3_surface_001 \\
      --cases S1_G3a_r0,S1_G3b_r0,S2_r0,S3_r0,S4_r0,S5_r0,S6_r0,S7_r0,S8_r0 --jobs 9
"""
import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
_spec = importlib.util.spec_from_file_location("wp12_phase3_surface", ROOT/"scripts/run_wp12_phase3_surface.py")
P3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P3)
S = P3.S


def _pairs_function():
    """surface_metrics_v1.self_intersections returning the intersecting pairs (same test, no sampling limit)."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    src = inspect.getsource(SM.self_intersections).replace("def self_intersections", "def _si_pairs")
    marker = '    return {"sampled_triangles": int(len(sample)), "intersecting_pairs_sampled"'
    src = src.replace(marker, "    return pairs\n"+marker)
    ns = {"np": np}
    exec(src, ns)   # noqa: S102 (verbatim copy of a repository function with an early return)
    return ns["_si_pairs"]


def pct(a, qs=(50, 95, 100), scale=S):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return None if not len(a) else [float(x) for x in np.percentile(a, qs)/scale]


def diagnose(path):
    from scipy.spatial import cKDTree
    from robotic_cutting_sim import cut_aware_surface as CAS
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.cut_topology import side_masks
    si_pairs = _pairs_function()
    run, frames = P3.load_states(path)
    fr = frames[-1]
    st = dict(run, positions=fr["positions"], codes=fr["codes"], F=fr["F"], log=fr["log"])
    count, labels, _, _ = P3.P1.components(st)
    locus = SM.material_locus(st["reference"], st["lattice"], st["positions"], st["codes"], cuts=st["cuts"], F=st["F"],
                              events=st["log"])
    surf = P3.new_b(run, fr["log"])
    m = P3.b_extract(surf, st, fr["log"])
    v, f, last = m["vertices"], m["faces"], m["last"]
    X = last["vertex_reference"]
    lat = surf.lattice
    out = {"state": str(path.name), "mechanical_components": int(count), "conflicting_link_rows": int(len(locus["cut"]))}
    codes = np.asarray(st["codes"], np.int64)
    # ---- vertex categories
    el = lat.cell_edge_link[last["vertex_cell"]]
    es = np.where(el >= 0, last["link_state"][np.maximum(el, 0)], -1)
    has_b, has_c = (es == CAS.LINK_BOUNDARY).any(1), (es == CAS.LINK_CUT).any(1)
    corners = lat.corner_particle[last["vertex_cell"]]
    cc = np.where(last["vertex_corner_mask"], codes[np.maximum(corners, 0)], 0)
    s1, s2 = side_masks(cc)
    mixed = (np.bitwise_or.reduce(s1, axis=1) & np.bitwise_or.reduce(s2, axis=1)) != 0
    cat = np.where(mixed, "mixed_tip", np.where(has_b & has_c, "crease", np.where(has_c, "cut_only",
                   np.where(has_b, "boundary_only", "other"))))
    if len(locus["cut"]):
        ext = SM.exterior_samples(st["lattice"], st["positions"], S, st["F"])
        cls = SM.classify_mesh_vertices(v, st["positions"], labels, locus, ext, spacing=S)
        region = np.flatnonzero(cls["kind"] == 2)
        rows = cls["row"][region]
        dref = np.full(len(region), np.nan)
        for k in np.unique(locus["cut"]):
            vm, pm = locus["cut"][rows] == k, locus["cut"] == k
            if vm.any():
                dref[vm], _ = SM.local_plane_signed(X[region][vm], locus["P"][pm], locus["N"][pm], .002)
        keys = SM.side_group_keys(locus, labels, st["cuts"])
        sp, sn = locus["side_points"].reshape(-1, 3), locus["side_normals"].reshape(-1, 3)
        vkey = (cls["component"][region]*st["cuts"]+locus["cut"][rows])*2+cls["side"][region]
        dcur = np.full(len(region), np.nan)
        for key in np.unique(vkey):
            vm, sm = vkey == key, keys == key
            dcur[vm], _ = SM.local_plane_signed(v[region[vm]], sp[sm], sn[sm], .002)
        side = cls["side"][region]
        rows_out = {}
        for c in np.unique(cat[region]):
            mm = cat[region] == c
            rows_out[str(c)] = {"vertices": int(mm.sum()), "clamped": int(last["vertex_clamped"][region][mm].sum()),
                                "reference_abs_p50_p95_max_s": pct(np.abs(dref[mm])),
                                "current_abs_p50_p95_max_s": pct(np.abs(dcur[mm])),
                                "mean_outward_side1_side2_s": [float(np.nanmean(dcur[mm & (side == j)])/S)
                                                               if np.isfinite(dcur[mm & (side == j)]).any() else None
                                                               for j in (0, 1)]}
        excl = ~np.isin(cat[region], ["boundary_only", "mixed_tip"])
        out["vertex_categories"] = {"region_rule_cut_face_vertices": int(len(region)), "per_category": rows_out,
                                    "without_boundary_only_and_tip": {"reference_abs_p50_p95_max_s": pct(np.abs(dref[excl])),
                                                                      "current_abs_p50_p95_max_s": pct(np.abs(dcur[excl]))},
                                    "B_provenance_cut_face_vertices": int(len(np.unique(f[last["face_kind"] == CAS.LINK_CUT])))}
        gap = np.einsum("ij,ij->i", locus["side_points"][:, 1]-locus["side_points"][:, 0], locus["N"])
        out["side_opening_along_N_mm"] = {"p5_p50_p95_max": [float(x) for x in np.percentile(gap, [5, 50, 95, 100])*1e3],
                                          "links_with_crossed_sides": int((gap < 0).sum())}
        # ---- normals
        nt, _, cen = SM.triangle_normals(v, f)
        core_cut = np.flatnonzero(cls["core"][f].all(1) & (last["face_kind"] == CAS.LINK_CUT))
        if len(core_cut):
            vrow = cls["row"][f[core_cut, 0]]
            own = (cls["component"][f[core_cut, 0]]*st["cuts"]+locus["cut"][vrow])*2+cls["side"][f[core_cut, 0]]
            cutv = region
            a_lt, a_lb, a_tb = [], [], []
            for key in np.unique(own):
                tm = core_cut[own == key]
                _, pl = SM.local_plane_signed(cen[tm], sp[keys == key], sn[keys == key], .003)
                bv = v[cutv[vkey == key]]
                _, pb = SM.local_plane_signed(cen[tm], bv, np.tile(nt[tm].mean(0), (len(bv), 1)), .003)
                ok = np.isfinite(pl[:, 0]) & np.isfinite(pb[:, 0])
                ang = lambda a, b: np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", a, b)), 0, 1)))  # noqa: E731
                a_lt.append(ang(nt[tm][ok], pl[ok]))
                a_lb.append(ang(pl[ok], pb[ok]))
                a_tb.append(ang(nt[tm][ok], pb[ok]))
            fl, fs = last["face_link"][core_cut], last["face_side_particle"][core_cut]
            _, inv = np.unique(fl*(lat.n+1)+fs, return_inverse=True)
            dev = []
            for g in range(int(inv.max())+1):
                vv = np.unique(f[core_cut][inv == g])
                if len(vv) == 4:
                    P = v[vv]
                    c0 = P.mean(0)
                    dev.append(np.abs((P-c0)@np.linalg.svd(P-c0)[2][-1]).max())
            out["normals_core_cut_triangles_deg"] = {
                "triangle_vs_locus_plane_p50_p95": [float(x) for x in np.percentile(np.concatenate(a_lt), [50, 95])],
                "locus_plane_vs_B_vertex_plane_p50_p95": [float(x) for x in np.percentile(np.concatenate(a_lb), [50, 95])],
                "triangle_vs_B_vertex_plane_p50_p95": [float(x) for x in np.percentile(np.concatenate(a_tb), [50, 95])],
                "quad_non_planarity_mm_p50_p95": [float(x) for x in np.percentile(dev, [50, 95])*1e3] if dev else None,
                "face_particle_roughness_qm_s": [SM.patch_roughness(st["positions"][np.unique(locus["pairs"][:, j])], .002)["quadratic_mean"]/S
                                                 for j in (0, 1)]}
    # ---- self-intersections
    pairs = si_pairs(v, f, max_triangles=10**7)
    kind = last["face_kind"]
    fcomp = last["vertex_component"][f[:, 0]]
    crease = has_b & has_c
    row = {"current_pairs": int(len(pairs))}
    if len(pairs):
        kk = np.unique([f"{int(kind[a])}{int(kind[b])}" for a, b in pairs], return_counts=True)
        row.update(face_kind_pairs={str(a): int(b) for a, b in zip(*kk)},
                   pairs_touching_crease_vertex=int(np.sum(crease[f[pairs[:, 0]]].any(1) | crease[f[pairs[:, 1]]].any(1))),
                   pairs_touching_tip_vertex=int(np.sum(mixed[f[pairs[:, 0]]].any(1) | mixed[f[pairs[:, 1]]].any(1))),
                   pairs_touching_clamped_vertex=int(np.sum(last["vertex_clamped"][f[pairs[:, 0]]].any(1)
                                                            | last["vertex_clamped"][f[pairs[:, 1]]].any(1))),
                   inter_component_pairs=int((fcomp[pairs[:, 0]] != fcomp[pairs[:, 1]]).sum()),
                   z_mm_p0_p50_p100=[float(x) for x in np.percentile(v[f[pairs[:, 0]]].mean(1)[:, 2], [0, 50, 100])*1e3])
    row["reference_pairs_within_component"] = {str(int(c)): int(len(si_pairs(X, f[fcomp == c], max_triangles=10**7)))
                                               for c in np.unique(fcomp)}
    out["self_intersections"] = row
    # ---- volume consistency of positions vs det F
    J = np.linalg.det(st["F"])
    cp = lat.corner_particle
    full = np.all(cp >= 0, 1)
    uncut = np.all(last["link_state"][np.maximum(lat.cell_edge_link, 0)] == CAS.LINK_UNCUT, 1) & full
    c = cp[uncut]
    tets = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    V = np.zeros(len(c))
    x = st["positions"]
    for a, b, cc3, d in tets:
        V += np.einsum("ij,ij->i", x[c[:, b]]-x[c[:, a]], np.cross(x[c[:, cc3]]-x[c[:, a]], x[c[:, d]]-x[c[:, a]]))/6
    V = np.abs(V)/S**3
    Jc = J[c].mean(1)
    out["volume_consistency"] = {"fully_linked_cells": int(len(c)), "position_volume_ratio_mean": float(V.mean()),
                                 "position_volume_ratio_p5_p95": [float(y) for y in np.percentile(V, [5, 95])],
                                 "corner_mean_det_F_mean": float(Jc.mean()), "mean_position_volume_over_det_F": float(np.mean(V/Jc)),
                                 "sum_det_F_over_N": float(J.mean())}
    # ---- uncut link stretch per component
    ok = last["link_state"] == CAS.LINK_UNCUT
    a, b = lat.link_lo[ok], lat.link_hi[ok]
    L = np.linalg.norm(x[a]-x[b], axis=1)
    err = np.linalg.norm((x[b]-x[a])-np.einsum("nij,nj->ni", st["F"][a], st["reference"][b]-st["reference"][a]), axis=1)
    out["uncut_link_stretch"] = {str(int(cl)): {"links": int((labels[a] == cl).sum()),
                                                "length_mm_p50_p95_max": [float(y) for y in np.percentile(L[labels[a] == cl], [50, 95, 100])*1e3],
                                                "links_longer_than_2mm": int(((labels[a] == cl) & (L > .002)).sum()),
                                                "dx_minus_F_dX_mm_p50_p95_max": [float(y) for y in np.percentile(err[labels[a] == cl], [50, 95, 100])*1e3]}
                                 for cl in np.unique(labels[a])[:8]}
    # embedding reach: B vertices far from every particle of their own component
    dmin = np.full(len(v), np.nan)
    vcomp = st["_rank"][last["vertex_component"]]    # B uses raw partition labels; ``labels`` are mass-ranked
    for cl in np.unique(vcomp):
        sel = vcomp == cl
        dmin[sel], _ = cKDTree(x[labels == cl]).query(v[sel])
    out["vertex_to_own_component_particle_mm_p50_p95_max"] = [float(y) for y in np.nanpercentile(dmin, [50, 95, 100])*1e3]
    out["embedding"] = m["info"]["embedding"]
    return out


# =============================================================================================== decision review checks
# Folded in from the Phase 3 decision review's read-only scripts (spot.py, side.py, comp.py, s5.py); same computations.

# diagnostics_003.json was written with this entry keyed "..._3_2_1" and the measured ranks read in the order 3, 2, 1, so
# it flags F5 rank counts as not reproduced; the stored values ({"1": 6274, "2": 610, "3": 20}) match the review.
REVIEW_EXPECTED = {   # review number -> (value as printed in WP12_PHASE3_DECISION_REVIEW.md, decimals compared)
    "F1.S6.cut_face_vertices_other_component": (492, 0), "F1.S6.cut_face_vertices": (2194, 0),
    "F1.S8.cut_face_vertices_other_component": (18, 0), "F1.S8.cut_face_vertices": (577, 0),
    "F1.S5.cut_face_vertices_other_component": (2, 0), "F1.S5.cut_face_vertices": (3386, 0),
    "F1.S6.B_current_p95_without_misassigned_s": (.345, 3), "F1.S6.B_current_p95_s": (.530, 3),
    "F1.S6.A_current_p95_s": (.442, 3),
    "F1.S6.side_opening_p50_mm": (.56, 2), "F1.S6.links_with_crossed_sides": (197, 0),
    "F2.S4.tip_regions": (25, 0), "F2.S4.clamped": (14, 0), "F2.S4.code0_zero_weight_regions": (20, 0),
    "F2.S4.side1_corners_min_max": ([1, 2], 0), "F2.S4.side2_corners_min_max": ([1, 2], 0),
    "F2.S4.code0_corners_min_max": ([2, 4], 0), "F2.S4.code0_link_length_along_N_mm": (.99, 2),
    "F2.S4.vertex_offset_from_side1_mm": (2.40, 2), "F2.S4.opening_centre_to_centre_mm": (4.6, 1),
    "F3.reference_cell_volume_mean": (1.000000, 6),
    "F3.S5.near_cut_mean_V_over_J": (.77, 2), "F3.S5.near_cut_p5_V_over_J": (.15, 2),
    "F3.S5.deficit_share_within_3mm": (.87, 2), "F3.S5.far_field_mean_V_over_J": (1.006, 3),
    "F3.S2.near_cut_mean_V_over_J": (.52, 2), "F3.S2.deficit_share_within_3mm": (.65, 2),
    "F4.S1_G3b.locus_self_residual_p95_s": (.512, 3), "F4.S2.locus_self_residual_p95_s": (.402, 3),
    "F4.S6.locus_self_residual_p95_s": (.437, 3), "F4.S8.locus_self_residual_p95_s": (.315, 3),
    "F4.locus_delaunay_normal_p50_deg_min_max": ([17, 28], 0), "F4.locus_delaunay_normal_p95_deg_min_max": ([32, 61], 0),
    "F4.B_plane_vs_locus_plane_p50_deg_min_max": ([2.5, 5.4], 1),
    "F4.S1_G3a.B_plane_vs_locus_plane_p95_deg": (30.2, 1), "F4.S4.B_plane_vs_locus_plane_p95_deg": (27.8, 1),
    "F5.G2.n123_clamped_cut_face_vertices": (55, 0), "F5.G2.n123_clamped_p50_s": (.147, 3),
    "F5.G2.n123_clamped_max_s": (.401, 3), "F5.G2.n123_unclamped_cut_face_vertices": (2834, 0),
    "F5.G2.n123_unclamped_max_s": (7.5e-7, 8), "F5.G2.n123_qef_rank_counts_1_2_3": ([6274, 610, 20], 0),
    "F6.S5.reference_error_over_half_s": (14, 0), "F6.S5.exterior_representatives": (7, 0),
    "F6.S5.wrong_cut_digit": (7, 0), "F6.S5.max_error_vs_own_digit_s": (.018, 3),
    "F6.S1_G3b.exterior_vertices_in_cut_region": (24, 0), "F6.S1_G3b.exterior_error_p50_max_s": ([.90, 1.70], 2),
    "F6.S1_G3b.true_cut_face_reference_max_s": (.17, 2),
    "F7.S7.chip_particles": (86, 0), "F7.S7.chip_particles_code2": (86, 0),
    "F7.S7.chip_uncut_link_length_p50_mm": (2.4, 1), "F7.S7.chip_uncut_link_length_max_mm": (19.8, 1),
    "F7.S7.chip_dx_minus_F_dX_max_mm": (19, 0),
    "F8.S1_G3b.current_pairs": (85, 0), "F8.S3.current_pairs": (27, 0), "F8.S1_G3b.pairs_touching_crease": (59, 0),
    "F8.translation_only_representatives": (0, 0),
    "F9.S6.current_pairs": (1198, 0), "F9.S8.current_pairs": (46, 0), "F9.S4.current_pairs": (198, 0),
    "F9.S4.reference_pairs": (95, 0), "F9.S1_G3b.reference_pairs": (0, 0), "F9.S3.reference_pairs": (0, 0),
    "F10.S8.material_displacement_p95_s_max_over_pairs_r1_r0": ([.46, .48], 2),
    "F10.S8.B_slow_motion_residual_p95_s_r0_r1": ([.072, .052], 3), "F10.S8.B_repeat_spread_s": (.02, 2),
    "F11.S8.B_mid_verdict_rederived": ("NOT_RUN", None),
}


def _load(path):
    run, frames = P3.load_states(path)
    fr = frames[-1]
    st = dict(run, positions=fr["positions"], codes=fr["codes"], F=fr["F"], log=fr["log"])
    count, labels, _, _ = P3.P1.components(st)
    return run, fr, st, count, labels


def _side_digit_of_vertices(last, f, codes, cuts):
    """B's own (cut digit, side 0/1) per vertex from its incident cut faces (side particle digit); -1 if none."""
    from robotic_cutting_sim import cut_aware_surface as CAS
    own_side, own_cut = np.full(len(last["vertex_reference"]), -1), np.full(len(last["vertex_reference"]), -1)
    cut_bits = np.zeros(len(last["vertex_reference"]), np.int64)
    for fi in np.flatnonzero(last["face_kind"] == CAS.LINK_CUT):
        bits, p = int(last["face_cutbits"][fi]), int(last["face_side_particle"][fi])
        for k in range(cuts):
            if (bits >> (2*k)) & 1:
                d = (int(codes[p]) >> (2*k)) & 3
                own_side[f[fi]] = d-1
                own_cut[f[fi]] = k
                cut_bits[f[fi]] |= 1 << k
    return own_cut, own_side, cut_bits


def tip_weights(st, surf, m):
    """F2: crack-tip regions (corners holding both side digits) of the S4 state."""
    from robotic_cutting_sim import cut_aware_surface as CAS
    from robotic_cutting_sim.cut_topology import side_masks
    last, v = m["last"], m["vertices"]
    lat = surf.lattice
    codes = np.asarray(st["codes"], np.int64)
    corners = lat.corner_particle[last["vertex_cell"]]
    cc = np.where(last["vertex_corner_mask"], codes[np.maximum(corners, 0)], -1)
    s1, s2 = side_masks(np.where(cc >= 0, cc, 0))
    mixed = np.flatnonzero((np.bitwise_or.reduce(s1, 1) & np.bitwise_or.reduce(s2, 1)) != 0)
    X = last["vertex_reference"]
    u = (X-lat.cell_origin[last["vertex_cell"]])/S
    N = np.asarray(st["log"]["normal_reference"], float).mean(0)
    N /= np.linalg.norm(N)
    x = st["positions"]
    rows, link_len, link_len_N = [], [], []
    for r in mixed:
        mask = last["vertex_corner_mask"][r]
        cp = corners[r][mask]
        cd = codes[cp]
        a1, a2 = side_masks(cd)
        g1, g2, g0 = cp[a1 != 0], cp[a2 != 0], cp[(a1 == 0) & (a2 == 0)]
        w = np.prod(np.where(CAS.CORNER_OFFSETS.astype(bool), u[r][None], 1-u[r][None]), 1)*mask
        wc = (w/w.sum())[mask]
        gap = (x[g2].mean(0)-x[g1].mean(0))@N if len(g1) and len(g2) else np.nan
        pos = (v[r]-x[g1].mean(0))@N if len(g1) else np.nan
        rows.append([len(g1), len(g2), len(g0), wc[a1 != 0].sum(), wc[a2 != 0].sum(), wc[(a1 == 0) & (a2 == 0)].sum(),
                     gap*1e3, pos*1e3])
        # lattice links (reference neighbours, both code 0 = uncut) between the region's code-0 corners whose reference
        # direction is along N: the joined hinge under the slit tip
        for i in range(len(g0)):
            for j in range(i+1, len(g0)):
                a, b = int(g0[i]), int(g0[j])
                dX = st["reference"][b]-st["reference"][a]
                if abs(np.linalg.norm(dX)-S) > 1e-6 or abs(dX@N)/np.linalg.norm(dX) < .9:
                    continue
                link_len.append(np.linalg.norm(x[b]-x[a])*1e3)
                link_len_N.append(abs((x[b]-x[a])@N)*1e3)
    rows = np.asarray(rows, float)
    return {"tip_regions": int(len(mixed)), "clamped": int(last["vertex_clamped"][mixed].sum()),
            "mean_cut_normal_reference": N.tolist(),
            "columns": ["n_side1", "n_side2", "n_code0", "w_side1", "w_side2", "w_code0", "opening_side2_minus_side1_mm",
                        "vertex_offset_from_side1_mean_mm"],
            "median": np.median(rows, 0).tolist(), "min": rows.min(0).tolist(), "max": rows.max(0).tolist(),
            "regions_with_code0_corner": int((rows[:, 2] > 0).sum()), "code0_zero_weight_regions": int((rows[:, 5] < 1e-9).sum()),
            "code0_links_along_N": int(len(link_len)),
            "code0_link_length_mm_p50_min_max": [float(np.median(link_len)), float(np.min(link_len)), float(np.max(link_len))] if link_len else None,
            "code0_link_length_along_N_mm_p50": float(np.median(link_len_N)) if link_len_N else None}


def volume_by_distance(st, surf, m, locus):
    """F3: V(cell positions) / mean corner det F of fully linked uncut cells vs the distance of the cell centre
    (reference) to the nearest Hermite point."""
    from scipy.spatial import cKDTree
    from robotic_cutting_sim import cut_aware_surface as CAS
    lat, last = surf.lattice, m["last"]
    cp = lat.corner_particle
    full = np.all(cp >= 0, 1)
    uncut = np.all(last["link_state"][np.maximum(lat.cell_edge_link, 0)] == CAS.LINK_UNCUT, 1) & full
    c = cp[uncut]
    tets = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]

    def vol(x):
        V = np.zeros(len(c))
        for a, b, d, e in tets:
            V += np.einsum("ij,ij->i", x[c[:, b]]-x[c[:, a]], np.cross(x[c[:, d]]-x[c[:, a]], x[c[:, e]]-x[c[:, a]]))/6
        return np.abs(V)/S**3
    Vref, V = vol(st["reference"]), vol(st["positions"])
    ratio = V/np.linalg.det(st["F"])[c].mean(1)
    dP, _ = cKDTree(locus["P"]).query(st["reference"][c].mean(1))
    deficit = (1-ratio).sum()
    bins = {}
    for lo, hi in ((0, 1.5e-3), (1.5e-3, 3e-3), (3e-3, 6e-3), (6e-3, np.inf)):
        sel = (dP >= lo) & (dP < hi)
        if sel.any():
            bins[f"{lo*1e3:.1f}-{hi*1e3:.1f}mm"] = {"cells": int(sel.sum()), "mean_V_over_J": float(ratio[sel].mean()),
                                                     "p5_V_over_J": float(np.percentile(ratio[sel], 5)),
                                                     "share_of_deficit": float((1-ratio[sel]).sum()/deficit)}
    worst = ratio < .8
    out = {"cells": int(len(c)), "reference_cell_volume_mean": float(Vref.mean()), "mean_V_over_J": float(ratio.mean()),
           "bins_by_reference_distance_to_hermite_point": bins,
           "deficit_share_within_3mm": float((1-ratio[dP < 3e-3]).sum()/deficit),
           "workpiece_current_z_mm_min_max": [float(st["positions"][:, 2].min()*1e3), float(st["positions"][:, 2].max()*1e3)]}
    if worst.any():
        out["cells_V_over_J_below_0p8"] = {"cells": int(worst.sum()),
                                           "current_z_mm_p5_p50_p95": (np.percentile(st["positions"][c[worst]].mean(1)[:, 2], [5, 50, 95])*1e3).tolist(),
                                           "distance_to_cut_mm_p50": float(np.median(dP[worst])*1e3)}
    return out


def locus_floor(st, labels, locus):
    """F4: locus self-residual vs its own 2 mm plane (core samples, sample inside the fit) and the triangle-normal error
    of a Delaunay mesh through the locus samples (edges < 1.6 mm, core) vs the 3 mm locus plane, per side group >= 30."""
    from scipy.spatial import Delaunay, cKDTree
    from robotic_cutting_sim import surface_metrics_v1 as SM
    keys = SM.side_group_keys(locus, labels, st["cuts"])
    sp, sn = locus["side_points"].reshape(-1, 3), locus["side_normals"].reshape(-1, 3)
    ext = SM.exterior_samples(st["lattice"], st["positions"], S, st["F"])
    dext, _ = cKDTree(ext[1]).query(sp)
    res, ang = [], []
    for key in np.unique(keys):
        sm = np.flatnonzero(keys == key)
        if len(sm) < 30:
            continue
        d, _ = SM.local_plane_signed(sp[sm], sp[sm], sn[sm], .002)
        core = dext[sm] >= .003
        res.append(np.abs(d[core]))
        pts = sp[sm]
        c0 = pts.mean(0)
        _, _, vt = np.linalg.svd(pts-c0)
        uv = (pts-c0)@vt[:2].T
        tri = Delaunay(uv).simplices
        e = np.max([np.linalg.norm(uv[tri[:, i]]-uv[tri[:, (i+1) % 3]], axis=1) for i in range(3)], 0)
        tri = tri[(e < 1.6e-3) & core[tri].all(1)]
        if not len(tri):
            continue
        nt = np.cross(pts[tri[:, 1]]-pts[tri[:, 0]], pts[tri[:, 2]]-pts[tri[:, 0]])
        nt /= np.linalg.norm(nt, axis=1, keepdims=True)
        _, pn = SM.local_plane_signed(pts[tri].mean(1), pts, sn[sm], .003)
        ok = np.isfinite(pn[:, 0])
        ang.append(np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", nt[ok], pn[ok])), 0, 1))))
    res = np.concatenate(res)/S if res else np.zeros(0)
    ang = np.concatenate(ang) if ang else np.zeros(0)
    r = _finite_pct(res, (50, 95, 100))
    a = _finite_pct(ang, (50, 95))
    return {"locus_self_residual_core_s_p50_p95_max": r, "locus_delaunay_triangle_normal_deg_p50_p95": a,
            "delaunay_triangles": int(len(ang)), "rule": "side groups >= 30 samples; core = >= 3 mm from exterior samples"}


def _finite_pct(a, qs):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, qs)] if len(a) else None


def assignment_checks(st, labels, locus, m):
    """F1/F6: region_rule_v1 component (nearest particle) vs B provenance; (cut, side) vs B's own cut faces;
    current P95 with and without the misassigned vertices; reference error vs B's own cut digit."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    v, f, last = m["vertices"], m["faces"], m["last"]
    X = last["vertex_reference"]
    codes = np.asarray(st["codes"], np.int64)
    ext = SM.exterior_samples(st["lattice"], st["positions"], S, st["F"])
    cls = SM.classify_mesh_vertices(v, st["positions"], labels, locus, ext, spacing=S)
    region = np.flatnonzero(cls["kind"] == 2)
    bcomp = st["_rank"][last["vertex_component"]]
    other = cls["component"][region] != bcomp[region]
    out = {"component_assignment": {"cut_face_vertices": int(len(region)), "cut_face_vertices_other_component": int(other.sum()),
                                    "all_vertices": int(len(v)), "all_vertices_other_component": int((cls["component"] != bcomp).sum())}}
    own_cut, own_side, own_bits = _side_digit_of_vertices(last, f, codes, st["cuts"])
    rows = cls["row"][region]
    rc, rs = locus["cut"][rows], cls["side"][region]
    has = own_side[region] >= 0
    pp = locus["pairs"][rows][np.arange(len(rows)), rs]
    digit = (codes[pp] >> (2*rc)) & 3
    mis_cut = has & (own_cut[region] != rc)
    mis_side = has & (own_cut[region] == rc) & ((digit-1) != own_side[region])
    sp, sn = locus["side_points"].reshape(-1, 3), locus["side_normals"].reshape(-1, 3)
    keys = SM.side_group_keys(locus, labels, st["cuts"])
    vkey = (cls["component"][region]*st["cuts"]+rc)*2+rs
    dcur = np.full(len(region), np.nan)
    for key in np.unique(vkey):
        vm = vkey == key
        dcur[vm], _ = SM.local_plane_signed(v[region[vm]], sp[keys == key], sn[keys == key], .002)
    a = np.abs(dcur)/S
    bad = mis_cut | mis_side
    out["side_assignment"] = {
        "with_B_cut_provenance": int(has.sum()), "wrong_cut_digit": int(mis_cut.sum()), "same_cut_wrong_side": int(mis_side.sum()),
        "current_p95_all_s": float(np.nanpercentile(a, 95)),
        "current_p95_without_misassigned_s": float(np.nanpercentile(a[~bad], 95)),
        "current_p95_without_other_component_s": float(np.nanpercentile(a[~other], 95)),
        "misassigned_current_p50_s": float(np.nanpercentile(a[bad], 50)) if bad.any() and np.isfinite(a[bad]).any() else None,
        "misassigned_and_other_component_overlap": int((bad & other).sum())}
    # F6: reference error > 0.5 s and its provenance
    dref = np.full(len(region), np.nan)
    for k in np.unique(locus["cut"]):
        vm, pm = rc == k, locus["cut"] == k
        if vm.any():
            dref[vm], _ = SM.local_plane_signed(X[region][vm], locus["P"][pm], locus["N"][pm], .002)
    big = np.abs(dref) > .5*S
    own = own_bits[region]
    match = ((own >> rc) & 1) != 0
    d2 = []
    for i in np.flatnonzero(big & (own > 0)):
        best = np.inf
        for k in range(st["cuts"]):
            if (own[i] >> k) & 1:
                pm = locus["cut"] == k
                d, _ = SM.local_plane_signed(X[region[i]][None], locus["P"][pm], locus["N"][pm], .002)
                if np.isfinite(d[0]):
                    best = min(best, abs(d[0]))
        d2.append(best/S)
    out["reference_error_over_half_s"] = {
        "vertices": int(big.sum()), "exterior_representatives_no_cut_face": int((big & (own == 0)).sum()),
        "region_digit_among_own_face_digits": int((big & match).sum()),
        "wrong_cut_digit": int((big & (own > 0) & ~match).sum()),
        "error_vs_own_face_digit_s_sorted": sorted(float(x) for x in d2),
        "error_min_max_s": [float(np.abs(dref[big]).min()/S), float(np.abs(dref[big]).max()/S)] if big.any() else None,
        "max_without_them_s": float(np.nanmax(np.abs(dref[~big]))/S) if (~big).any() else None,
        "max_without_exterior_representatives_s": float(np.nanmax(np.abs(dref[own > 0]))/S) if (own > 0).any() else None}
    return out


def chip_codes(st, labels):
    codes = np.asarray(st["codes"])
    chip = labels != 0
    u, n = np.unique(codes[chip], return_counts=True)
    return {"chip_particles": int(chip.sum()), "codes": {str(int(a)): int(b) for a, b in zip(u, n)}}


def review_checks(path):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    case = path.name.replace("_states.npz", "")
    run, fr, st, count, labels = _load(path)
    locus = SM.material_locus(st["reference"], st["lattice"], st["positions"], st["codes"], cuts=st["cuts"], F=st["F"],
                              events=st["log"])
    surf = P3.new_b(run, fr["log"])
    m = P3.b_extract(surf, st, fr["log"])
    out = {"state": path.name}
    if not len(locus["cut"]):
        return out
    out.update(assignment_checks(st, labels, locus, m))
    out["volume_by_distance"] = volume_by_distance(st, surf, m, locus)
    out["locus_floor"] = locus_floor(st, labels, locus)
    if case.startswith("S4"):
        out["tip_weights"] = tip_weights(st, surf, m)
    if case.startswith("S7"):
        out["chip_codes"] = chip_codes(st, labels)
    return out


def s8_frame_motion(run_dir):
    """F10: per consecutive S8 snapshot pair with equal codes, material displacement P95 (s) from the saved states and
    the stored slow-motion residual P95 of A and B (run JSON); at the max-residual frame of each repeat."""
    out = {}
    for r in (0, 1):
        js, npz = run_dir/f"S8_r{r}.json", run_dir/f"S8_r{r}_states.npz"
        if not (js.exists() and npz.exists()):
            continue
        rec = json.loads(js.read_text())
        run, frames = P3.load_states(npz)
        frames = frames[:-1]
        stored = {f["frame"]: f for f in rec["recontact_frames"]}
        rows = []
        for a, b in zip(frames[:-1], frames[1:]):
            if not np.array_equal(a["codes"], b["codes"]):
                continue
            disp = np.percentile(np.linalg.norm(b["positions"]-a["positions"], axis=1), 95)/S
            t = stored.get(b["name"], {}).get("temporal") or {}
            rows.append({"frame": b["name"], "dt_s": (b["substep"]-a["substep"])*P3.DT, "material_displacement_p95_s": float(disp),
                         "stored_material_displacement_p95_s": (t.get(P3.B) or {}).get("material_displacement", {}).get("p95_spacings"),
                         "B_residual_p95_s": (t.get(P3.B) or {}).get("residual_after_transport", {}).get("p95_spacings"),
                         "A_residual_p95_s": (t.get(P3.A) or {}).get("residual_after_transport", {}).get("p95_spacings")})
        best = max((x for x in rows if x["B_residual_p95_s"] is not None), key=lambda x: x["B_residual_p95_s"], default=None)
        disp = [x["material_displacement_p95_s"] for x in rows]
        out[f"r{r}"] = {"pairs": rows, "max_B_residual_frame": best,
                        "material_displacement_p95_s_min_p50_max": [float(np.min(disp)), float(np.median(disp)), float(np.max(disp))] if disp else None,
                        "pairs_with_displacement_p95_le_0p10_s": int(sum(d <= .10 for d in disp))}
    return out


def data_reads(run_dir):
    """Numbers the review read from the stored run JSON (not re-run), plus F11 re-derived with the current driver."""
    out = {}
    g2 = run_dir/"G2_r0.json"
    if g2.exists():
        n = json.loads(g2.read_text())["planes"]["n123"][P3.B]
        out["G2_n123_B"] = {"clamped_cut_face_vertices": n["clamped_cut_face_vertices"], "clamped_vs_analytic": n["clamped_vs_analytic"],
                            "excluding_clamped": n["cut_face_max_excluding_clamped"], "cut_face_vs_analytic": n["cut_face_vs_analytic"],
                            "qef_rank_counts": n["qef"]["rank_counts"]}
    s6 = run_dir/"S6_r0.json"
    if s6.exists():
        rep = json.loads(s6.read_text())["frames"][-1]["reports"]
        out["S6_current_p95_s"] = {m: rep[m]["scalars"]["current_p95_s"] for m in (P3.A, P3.B)}
    T = P3.protocol_thresholds()
    for r in (0, 1):
        s8 = run_dir/f"S8_r{r}.json"
        if s8.exists():
            res = json.loads(s8.read_text())
            stored = res["threshold_checks"][P3.B_MID]["verdict"]
            again = P3.case_checks("S8", res, T)[P3.B_MID]
            out[f"S8_r{r}_B_mid"] = {"stored_verdict": stored, "rederived_verdict": again["verdict"],
                                     "rederived_rows": {k: c["verdict"] for k, c in again["checks"].items()},
                                     "B_slow_motion_residual_p95_s": res["threshold_checks"][P3.B]["checks"]["all_temporal_slow_motion_residual_p95_s"]["value"]}
    # every stored physical-case verdict re-derived with the current driver: only B_mid S8 may differ (F11)
    diffs, compared = [], 0
    for js in sorted(run_dir.glob("S[1-8]*_r[0-9].json")):
        res = json.loads(js.read_text())
        again = P3.case_checks(res["case"], res, T)
        for method, row in res["threshold_checks"].items():
            compared += 1
            if again.get(method, {}).get("verdict") != row["verdict"]:
                diffs.append({"record": js.name, "method": method, "stored": row["verdict"],
                              "rederived": again.get(method, {}).get("verdict")})
    out["stored_verdicts_rederived"] = {"method_case_verdicts_compared": compared, "differences": diffs}
    return out


def _matches(expected, measured, decimals):
    if decimals is None:
        return expected == measured
    if measured is None:
        return False
    e, mm = np.atleast_1d(np.asarray(expected, float)), np.atleast_1d(np.asarray(measured, float))
    if e.shape != mm.shape:
        return False
    return bool(np.all(np.abs(np.round(mm, decimals)-e) <= .5*10.**-decimals+1e-12))


def review_reproduction(out):
    """Measured value for every REVIEW_EXPECTED entry and whether it reproduces at the printed precision."""
    c = out["cases"]
    rc = lambda case: (c.get(f"{case}_r0") or {}).get("review_checks") or {}    # noqa: E731
    base = lambda case: c.get(f"{case}_r0") or {}                              # noqa: E731
    dr = out.get("data_reads", {})
    s8 = out.get("s8_frame_motion", {})
    got = {}
    for case in ("S6", "S8", "S5"):
        ca = rc(case).get("component_assignment", {})
        got[f"F1.{case}.cut_face_vertices_other_component"] = ca.get("cut_face_vertices_other_component")
        got[f"F1.{case}.cut_face_vertices"] = ca.get("cut_face_vertices")
    got["F1.S6.B_current_p95_without_misassigned_s"] = rc("S6").get("side_assignment", {}).get("current_p95_without_misassigned_s")
    got["F1.S6.B_current_p95_s"] = rc("S6").get("side_assignment", {}).get("current_p95_all_s")
    got["F1.S6.A_current_p95_s"] = (dr.get("S6_current_p95_s") or {}).get(P3.A)
    so = base("S6").get("side_opening_along_N_mm", {})
    got["F1.S6.side_opening_p50_mm"] = (so.get("p5_p50_p95_max") or [None, None])[1]
    got["F1.S6.links_with_crossed_sides"] = so.get("links_with_crossed_sides")
    tw = rc("S4").get("tip_weights", {})
    if tw:
        cols = tw["columns"]
        got.update({"F2.S4.tip_regions": tw["tip_regions"], "F2.S4.clamped": tw["clamped"],
                    "F2.S4.code0_zero_weight_regions": tw["code0_zero_weight_regions"],
                    "F2.S4.side1_corners_min_max": [tw["min"][0], tw["max"][0]],
                    "F2.S4.side2_corners_min_max": [tw["min"][1], tw["max"][1]],
                    "F2.S4.code0_corners_min_max": [tw["min"][2], tw["max"][2]],
                    "F2.S4.code0_link_length_along_N_mm": tw["code0_link_length_along_N_mm_p50"],
                    "F2.S4.vertex_offset_from_side1_mm": tw["median"][cols.index("vertex_offset_from_side1_mean_mm")],
                    "F2.S4.opening_centre_to_centre_mm": tw["median"][cols.index("opening_side2_minus_side1_mm")]})
    vb5, vb2 = rc("S5").get("volume_by_distance", {}), rc("S2").get("volume_by_distance", {})
    got["F3.reference_cell_volume_mean"] = vb5.get("reference_cell_volume_mean")
    b5, b2 = vb5.get("bins_by_reference_distance_to_hermite_point", {}), vb2.get("bins_by_reference_distance_to_hermite_point", {})
    got["F3.S5.near_cut_mean_V_over_J"] = (b5.get("0.0-1.5mm") or {}).get("mean_V_over_J")
    got["F3.S5.near_cut_p5_V_over_J"] = (b5.get("0.0-1.5mm") or {}).get("p5_V_over_J")
    got["F3.S5.deficit_share_within_3mm"] = vb5.get("deficit_share_within_3mm")
    got["F3.S5.far_field_mean_V_over_J"] = (b5.get("6.0-inf mm") or b5.get("6.0-infmm") or {}).get("mean_V_over_J")
    got["F3.S2.near_cut_mean_V_over_J"] = (b2.get("0.0-1.5mm") or {}).get("mean_V_over_J")
    got["F3.S2.deficit_share_within_3mm"] = vb2.get("deficit_share_within_3mm")
    p50s, p95s = [], []
    for case in ("S1_G3b", "S2", "S6", "S8"):
        lf = rc(case).get("locus_floor", {})
        got[f"F4.{case}.locus_self_residual_p95_s"] = (lf.get("locus_self_residual_core_s_p50_p95_max") or [None, None])[1]
        if lf.get("locus_delaunay_triangle_normal_deg_p50_p95"):
            p50s.append(lf["locus_delaunay_triangle_normal_deg_p50_p95"][0])
            p95s.append(lf["locus_delaunay_triangle_normal_deg_p50_p95"][1])
    got["F4.locus_delaunay_normal_p50_deg_min_max"] = [min(p50s), max(p50s)] if p50s else None
    got["F4.locus_delaunay_normal_p95_deg_min_max"] = [min(p95s), max(p95s)] if p95s else None
    bp = {case: (base(case).get("normals_core_cut_triangles_deg") or {}).get("locus_plane_vs_B_vertex_plane_p50_p95")
          for case in ("S1_G3a", "S1_G3b", "S2", "S3", "S5", "S6", "S8")}
    vals = [x[0] for x in bp.values() if x]
    got["F4.B_plane_vs_locus_plane_p50_deg_min_max"] = [min(vals), max(vals)] if vals else None
    got["F4.S1_G3a.B_plane_vs_locus_plane_p95_deg"] = (bp.get("S1_G3a") or [None, None])[1]
    s4n = (base("S4").get("normals_core_cut_triangles_deg") or {}).get("locus_plane_vs_B_vertex_plane_p50_p95")
    got["F4.S4.B_plane_vs_locus_plane_p95_deg"] = s4n[1] if s4n else None
    g2 = dr.get("G2_n123_B")
    if g2:
        got.update({"F5.G2.n123_clamped_cut_face_vertices": g2["clamped_cut_face_vertices"],
                    "F5.G2.n123_clamped_p50_s": g2["clamped_vs_analytic"]["p50_spacings"],
                    "F5.G2.n123_clamped_max_s": g2["clamped_vs_analytic"]["max_spacings"],
                    "F5.G2.n123_unclamped_cut_face_vertices": g2["excluding_clamped"]["count"],
                    "F5.G2.n123_unclamped_max_s": g2["excluding_clamped"]["max_spacings"],
                    "F5.G2.n123_qef_rank_counts_1_2_3": [g2["qef_rank_counts"][k] for k in ("1", "2", "3")]})
    r5 = rc("S5").get("reference_error_over_half_s", {})
    got.update({"F6.S5.reference_error_over_half_s": r5.get("vertices"),
                "F6.S5.exterior_representatives": r5.get("exterior_representatives_no_cut_face"),
                "F6.S5.wrong_cut_digit": r5.get("wrong_cut_digit"),
                "F6.S5.max_error_vs_own_digit_s": max(r5["error_vs_own_face_digit_s_sorted"]) if r5.get("error_vs_own_face_digit_s_sorted") else None})
    vc = (base("S1_G3b").get("vertex_categories") or {}).get("per_category", {})
    bo = vc.get("boundary_only", {})
    got["F6.S1_G3b.exterior_vertices_in_cut_region"] = bo.get("vertices")
    ref = bo.get("reference_abs_p50_p95_max_s")
    got["F6.S1_G3b.exterior_error_p50_max_s"] = [ref[0], ref[2]] if ref else None
    wo = (base("S1_G3b").get("vertex_categories") or {}).get("without_boundary_only_and_tip", {})
    got["F6.S1_G3b.true_cut_face_reference_max_s"] = (wo.get("reference_abs_p50_p95_max_s") or [None]*3)[2]
    cc = rc("S7").get("chip_codes", {})
    got["F7.S7.chip_particles"] = cc.get("chip_particles")
    got["F7.S7.chip_particles_code2"] = (cc.get("codes") or {}).get("2")
    us = (base("S7").get("uncut_link_stretch") or {}).get("1", {})
    got["F7.S7.chip_uncut_link_length_p50_mm"] = (us.get("length_mm_p50_p95_max") or [None]*3)[0]
    got["F7.S7.chip_uncut_link_length_max_mm"] = (us.get("length_mm_p50_p95_max") or [None]*3)[2]
    got["F7.S7.chip_dx_minus_F_dX_max_mm"] = (us.get("dx_minus_F_dX_mm_p50_p95_max") or [None]*3)[2]
    si = lambda case: base(case).get("self_intersections") or {}    # noqa: E731
    got.update({"F8.S1_G3b.current_pairs": si("S1_G3b").get("current_pairs"), "F8.S3.current_pairs": si("S3").get("current_pairs"),
                "F8.S1_G3b.pairs_touching_crease": si("S1_G3b").get("pairs_touching_crease_vertex"),
                "F8.translation_only_representatives": sum(int((base(k[:-3]).get("embedding") or {}).get("translation_only_representatives", 0) or 0)
                                                           for k in c)})
    got.update({"F9.S6.current_pairs": si("S6").get("current_pairs"), "F9.S8.current_pairs": si("S8").get("current_pairs"),
                "F9.S4.current_pairs": si("S4").get("current_pairs"),
                "F9.S4.reference_pairs": sum((si("S4").get("reference_pairs_within_component") or {}).values()) if si("S4") else None,
                "F9.S1_G3b.reference_pairs": sum((si("S1_G3b").get("reference_pairs_within_component") or {}).values()) if si("S1_G3b") else None,
                "F9.S3.reference_pairs": sum((si("S3").get("reference_pairs_within_component") or {}).values()) if si("S3") else None})
    mx = sorted(s8[r]["material_displacement_p95_s_min_p50_max"][2] for r in ("r0", "r1") if s8.get(r))
    got["F10.S8.material_displacement_p95_s_max_over_pairs_r1_r0"] = mx if len(mx) == 2 else None
    res = [dr.get(f"S8_r{r}_B_mid", {}).get("B_slow_motion_residual_p95_s") for r in (0, 1)]
    got["F10.S8.B_slow_motion_residual_p95_s_r0_r1"] = res if None not in res else None
    got["F10.S8.B_repeat_spread_s"] = abs(res[0]-res[1]) if None not in res else None
    got["F11.S8.B_mid_verdict_rederived"] = dr.get("S8_r0_B_mid", {}).get("rederived_verdict")
    table, not_reproduced = {}, []
    for k, (e, dec) in REVIEW_EXPECTED.items():
        ok = _matches(e, got.get(k), dec)
        table[k] = {"review": e, "measured": got.get(k), "decimals": dec, "reproduced": ok}
        if not ok:
            not_reproduced.append(k)
    return {"table": table, "not_reproduced": not_reproduced, "entries": len(table)}


def _one(args):
    path, base, review = args
    row = diagnose(path) if base else {"state": path.name}
    if review:
        row["review_checks"] = review_checks(path)
    return path.name.replace("_states.npz", ""), row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--cases", default="S1_G3a_r0,S1_G3b_r0,S2_r0,S3_r0,S4_r0,S5_r0,S6_r0,S7_r0,S8_r0")
    p.add_argument("--no-review-checks", dest="review", action="store_false",
                   help="omit the decision-review checks (diagnostics_001/002 content only)")
    p.add_argument("--no-base", dest="base", action="store_false", help="only the decision-review checks")
    p.add_argument("--jobs", type=int, default=1, help="cases in parallel (separate processes, CPU only)")
    args = p.parse_args()
    run = args.run if args.run.is_absolute() else ROOT/args.run
    k = 1
    while (run/f"diagnostics_{k:03d}.json").exists():
        k += 1
    target = run/f"diagnostics_{k:03d}.json"
    t0 = time.perf_counter()
    out = {"command": " ".join([sys.executable, *sys.argv]), "git_head": P3.git("rev-parse", "HEAD"),
           "git_dirty_paths": len(P3.git("status", "--porcelain").splitlines()),
           "scope": "read-only attribution of B metric outliers; not a threshold or a verdict", "cases": {}}
    if args.review:
        out["review"] = "docs/reports/WP12_PHASE3_DECISION_REVIEW.md (commit b67e80c); checks folded in from its ad-hoc scripts"
    paths = [run/f"{case}_states.npz" for case in args.cases.split(",")]
    jobs = [(path, args.base, args.review) for path in paths if path.exists()]
    out["missing_states"] = [str(path.name) for path in paths if not path.exists()]
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(_one, jobs)
    else:
        results = [_one(j) for j in jobs]
    for case, row in results:
        out["cases"][case] = row
        print(f"[{case}] done", flush=True)
    if args.review:
        out["s8_frame_motion"] = s8_frame_motion(run)
        out["data_reads"] = data_reads(run)
        out["review_reproduction"] = review_reproduction(out)
        print("not reproduced:", out["review_reproduction"]["not_reproduced"], flush=True)
    out["wall_s"] = time.perf_counter()-t0
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(P3.jsonable(out), indent=1))
    print(target)


if __name__ == "__main__":
    main()
