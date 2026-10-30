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

Nothing is written into the run directory except ``diagnostics_NNN.json`` (new file, never overwritten).

  .venv/bin/python scripts/diagnose_wp12_phase3_surface.py --run artifacts/wp12/phase3_surface_001 \\
      --cases S1_G3a_r0,S1_G3b_r0,S2_r0,S3_r0,S4_r0,S5_r0,S6_r0,S7_r0,S8_r0
"""
import argparse
import importlib.util
import inspect
import json
import sys
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--cases", default="S1_G3a_r0,S1_G3b_r0,S2_r0,S3_r0,S4_r0,S5_r0,S6_r0,S7_r0,S8_r0")
    args = p.parse_args()
    run = args.run if args.run.is_absolute() else ROOT/args.run
    k = 1
    while (run/f"diagnostics_{k:03d}.json").exists():
        k += 1
    out = {"command": " ".join([sys.executable, *sys.argv]), "git_head": P3.git("rev-parse", "HEAD"),
           "scope": "read-only attribution of B metric outliers; not a threshold or a verdict", "cases": {}}
    for case in args.cases.split(","):
        path = run/f"{case}_states.npz"
        if path.exists():
            out["cases"][case] = diagnose(path)
            print(f"[{case}] done", flush=True)
        (run/f"diagnostics_{k:03d}.json").write_text(json.dumps(P3.jsonable(out), indent=1))
    print(run/f"diagnostics_{k:03d}.json")


if __name__ == "__main__":
    main()
