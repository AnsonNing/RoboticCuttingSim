#!/usr/bin/env python3
"""WP-12 Phase 3 closure: read-only diagnostics behind the xhigh decision (``docs/reports/WP12_PHASE3_XHIGH_DECISION.md``,
commit a94bd9f, section 6 step 1), folded in from the reviewer's ad-hoc scripts so they are reproducible from the
repository. Reads only; writes one new diagnostics JSON and one compact evidence JSON, never overwrites.

Inputs (read-only): the protocol v2 state sets ``saved`` (``artifacts/wp12/phase3_surface_001/<case>_r<r>_states.npz``),
``H1`` and ``H2`` (``artifacts/wp12/phase3_surface_002/heldout_<set>/<case>_r<r>_states.npz``), the r0 packages
``phase3_surface_002/eval_<set>/<case>_r0_package.npz`` and the stored evaluations ``eval_<set>/<case>_r<r>.json``.
Every check uses the final frame. CPU only.

Per state (9 physical cases x 3 sets x 2 repeats):

``link_layer`` (decision section 1, M4 location, codes and fragmentation; section 2 F3 bands)
    Uncut lattice links (both sites present, codes not conflicting) with current length L and
    |dx - F_mean dX| (F_mean = mean of the endpoints' total F), both in spacings s. A link is *flagged* exactly as
    ``CutAwareSurfaceV2._separation`` (M4) flags it: L > ``SEPARATION_LENGTH_S`` and inconsistency >
    ``SEPARATION_INCONSISTENCY_S`` (imported, not copied; no rotation, as for kinematic states). Reported: reference
    distance of each link midpoint to the nearest recorder Hermite point (s), the 0-1 / 1-2 / 2-3 / 3-6 / > 6 s bands
    (inconsistency P50 / P99 of all uncut links, flagged count), link direction relative to the nearest Hermite normal
    (in plane |n.e| < 0.3, along N |n.e| > 0.7), exterior endpoints (< 6 present neighbours), endpoint codes (same code;
    0-vs-digit mix for some cut; code values), det F at flagged endpoints, the reference z / x of flagged links, and the
    number of lattice pieces (and particles in them) that removing the flagged links splits off their uncut-graph
    component. The B v2 run's own recorded M4 count (``eval_<set>`` JSON) is read beside it as a replication check.
``fold_layer`` (decision section 2)
    Self-intersecting triangle pairs of the ``cut_aware_surface_v1`` mesh by ``surface_metrics_v2.intersecting_pairs``
    (same sampled test as the gate metric). Per triangle the maximum incident uncut-link inconsistency over the 8
    particles nearest its centroid (current positions); fractions > 0.5 s and > 1 s for triangles in pairs and for all
    triangles. r0 uses the stored package mesh; r1 has no package, so B v1 is re-extracted from the states with the v2
    runner's own ``make_b`` / ``b_extract`` (the same path that wrote the package). For r0 the re-extraction is compared
    with the package (``reextraction_vs_package``) to show that the r1 meshes are what the run would have stored.

Data reads from the stored evaluations (no recomputation of any metric):
    the counterfactual protocol-v2 comparison of the labelled no-M4 ablation (``gate_v2`` of the v2 runner applied to
    ``cut_aware_surface_v2_ablation_no_M4``), labelled ``COUNTERFACTUAL_NOT_A_GATE`` -- never a gate verdict -- with the
    B v1 recomputation beside it; S4 current P95 / matched-normal rows and M2 hinge counts; the no-M4 G4 hold verdicts;
    the P3 leave-one-out floor against A and B v1 current P95 (D1); P4 reference consistency per state (F3); S7 P7
    (D6); S6 current P95 (F1); matched-normal P95 (F4); B v2 M4 fallback cells.

``DECISION_EXPECTED`` holds the decision's printed numbers. ``decision_reproduction`` compares each with the measured
value at the printed precision. Link-layer and fold numbers are r0 (as the decision's scripts); for numbers read from
stored evaluations the decision names no repeat, so r0, r1 and the repeat mean are all recorded and the entry records
which of them reproduces (``basis``). Status per entry: REPRODUCED (one basis matches every element),
REPRODUCED_MIXED_REPEATS (each element matches, but only from different repeats) or NOT_REPRODUCED; the last two are
listed with a note.

  .venv/bin/python scripts/diagnose_wp12_phase3_v2.py --jobs 12
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
_spec = importlib.util.spec_from_file_location("wp12_phase3_surface_v2", ROOT/"scripts/run_wp12_phase3_surface_v2.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
P3 = D.P3

OUT_RUN = ROOT/"artifacts/wp12/phase3_surface_002"
EVIDENCE = ROOT/"docs/evidence/wp12/phase3_closure_001.json"
SETS = D.SETS
CASES = D.PHYSICAL
A, B1, B2, NOM4 = D.A, D.B1, D.B2, D.B2NOM4
BANDS = ((0, 1), (1, 2), (2, 3), (3, 6), (6, np.inf))
LABEL_CF = "COUNTERFACTUAL_NOT_A_GATE"
# the cases the decision's ad-hoc scripts ran (m4_where.py, m4_codes.py, fold_where.py), r0
M4_WHERE_CASES = [("saved", c) for c in ("S1_G3a", "S1_G3b", "S3", "S4", "S5", "S6", "S7")]+[("H2", c) for c in ("S4", "S5", "S6", "S7")]
M4_CODES_CASES = [("saved", c) for c in ("S1_G3a", "S1_G3b", "S4", "S5", "S7")]+[("H2", c) for c in ("S5", "S6")]
FOLD_CASES = ("S1_G3b", "S3", "S5", "S4")


def states_path(tag, case, r):
    return D.SAVED_RUN/f"{case}_r{r}_states.npz" if tag == "saved" else OUT_RUN/f"heldout_{tag}/{case}_r{r}_states.npz"


def eval_path(tag, case, r):
    return OUT_RUN/f"eval_{tag}/{case}_r{r}.json"


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def q(v, qs=(0, 50, 95, 100), d=3):
    v = np.asarray(v, float)
    return [float(np.round(t, d)) for t in np.percentile(v, qs)] if len(v) else None


# ------------------------------------------------------------------------------------------------ per-state checks

def uncut_links(lat, x, F, codes, s):
    from robotic_cutting_sim.cut_aware_surface import LINK_UNCUT, link_states
    state, _ = link_states(lat, codes)
    u = np.flatnonzero(state == LINK_UNCUT)
    a, b = lat.link_lo[u], lat.link_hi[u]
    dx = x[b]-x[a]
    dX = lat.reference[b]-lat.reference[a]
    Fm = .5*(F[a]+F[b])
    L = np.linalg.norm(dx, axis=1)/s
    inc = np.linalg.norm(dx-np.einsum("rij,rj->ri", Fm, dX), axis=1)/s
    return state, u, a, b, L, inc


def link_layer(z):
    """M4 location / direction / codes / fragmentation and the near-face inconsistency bands (final frame)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from robotic_cutting_sim import cut_aware_surface_v2 as CV2
    from robotic_cutting_sim.cut_aware_surface import LINK_CUT, ReferenceLattice
    from robotic_cutting_sim.cut_topology import side_masks
    s = float(z["spacing_m"])
    lat = ReferenceLattice(z["lattice"], z["reference"], s)
    x, F = z["positions"][-1].astype(np.float64), z["F"][-1].astype(np.float64)
    codes = z["codes"][-1]
    c = codes.astype(np.int64)
    K = int(z["cuts"])
    state, u, a, b, L, inc = uncut_links(lat, x, F, codes, s)
    flag = (L > CV2.SEPARATION_LENGTH_S) & (inc > CV2.SEPARATION_INCONSISTENCY_S)
    cl = np.flatnonzero(state == LINK_CUT)
    face = np.zeros(lat.n, bool)
    face[lat.link_lo[cl]] = True
    face[lat.link_hi[cl]] = True
    log = z["log"]
    P = np.asarray(log["hermite_reference"], np.float64)
    N = np.asarray(log["normal_reference"], np.float64)
    mid = .5*(lat.reference[a]+lat.reference[b])
    dist, j = cKDTree(P).query(mid)
    dist = dist/s
    ndot = np.abs(np.einsum("ij,ij->i", np.eye(3)[lat.link_axis[u]], N[j]))
    nb = np.zeros(lat.n, int)
    pres = (lat.link_lo >= 0) & (lat.link_hi >= 0)
    np.add.at(nb, lat.link_lo[pres], 1)
    np.add.at(nb, lat.link_hi[pres], 1)
    s1, s2 = side_masks(codes)
    labelled = (s1 | s2) != 0

    def comps(mask):
        g = coo_matrix((np.ones(int(mask.sum())), (a[mask], b[mask])), shape=(lat.n, lat.n))
        return connected_components(g, directed=False)
    n0, lab0 = comps(np.ones(len(u), bool))
    n1, lab1 = comps(~flag)
    sizes1 = np.bincount(lab1)
    new_pieces = new_particles = 0
    for c0 in np.unique(lab0):
        sub = np.unique(lab1[lab0 == c0])
        if len(sub) > 1:
            ss = sorted(sizes1[sub])
            new_pieces += len(sub)-1
            new_particles += int(sum(ss[:-1]))
    same = c[a] == c[b]
    mix = np.zeros(len(u), bool)
    for k in range(K):
        da, db = (c[a] >> (2*k)) & 3, (c[b] >> (2*k)) & 3
        mix |= ((da == 0) & (db != 0)) | ((da != 0) & (db == 0))
    f = flag
    ends = np.concatenate([a[f], b[f]])
    detF = np.linalg.det(F)
    ref = lat.reference
    vals, cnt = np.unique(c[ends], return_counts=True)
    rec = {"particles": int(lat.n), "cuts": K, "uncut_links": int(len(u)), "flagged": int(f.sum()),
           "flag_rule": {"length_gt_s": CV2.SEPARATION_LENGTH_S, "inconsistency_gt_s": CV2.SEPARATION_INCONSISTENCY_S,
                         "source": "cut_aware_surface_v2.CutAwareSurfaceV2._separation (imported constants)"},
           "flagged_ref_dist_to_hermite_s_p0_50_95_100": q(dist[f]),
           "flagged_within_1s": int((dist[f] <= 1).sum()), "flagged_within_2s": int((dist[f] <= 2).sum()),
           "flagged_within_3s": int((dist[f] <= 3).sum()),
           "flagged_3_to_6s": int(((dist[f] > 3) & (dist[f] <= 6)).sum()),
           "flagged_with_face_endpoint": int((face[a[f]] | face[b[f]]).sum()),
           "flagged_along_N": int((ndot[f] > .7).sum()), "flagged_in_plane": int((ndot[f] < .3).sum()),
           "flagged_exterior_endpoint": int(((nb[a[f]] < 6) | (nb[b[f]] < 6)).sum()),
           "flagged_both_labelled": int((labelled[a[f]] & labelled[b[f]]).sum()),
           "flagged_length_s": q(L[f]), "flagged_inconsistency_s": q(inc[f]),
           "flagged_endpoints": int(len(np.unique(ends))),
           "flagged_same_code": int((f & same).sum()), "flagged_zero_vs_digit_mix": int((f & mix).sum()),
           "flagged_endpoint_code_values": {str(int(v)): int(n) for v, n in zip(vals, cnt)},
           "flagged_link_code_pairs": {f"{int(p)}-{int(r)}": int(n) for (p, r), n in
                                       zip(*np.unique(np.sort(np.stack([c[a[f]], c[b[f]]], 1), 1), axis=0, return_counts=True))}
           if f.any() else {},
           "flagged_endpoint_detF_p5_p50_p95": q(detF[ends], (5, 50, 95), 4) if f.any() else None,
           "flagged_ref_z_mm_p0_50_100": [float(np.round(v*1e3, 2)) for v in np.percentile(.5*(ref[a[f], 2]+ref[b[f], 2]), [0, 50, 100])] if f.any() else None,
           "ref_z_range_mm": [float(ref[:, 2].min()*1e3), float(ref[:, 2].max()*1e3)],
           "all_uncut_mix_links": int(mix.sum()),
           "uncut_graph_components_before_after": [int(n0), int(n1)],
           "new_lattice_pieces_from_flagged_removal": int(new_pieces), "particles_in_new_pieces": int(new_particles),
           "uncut_by_ref_distance_s": {}}
    for lo, hi in BANDS:
        m = (dist > lo) & (dist <= hi)
        if m.any():
            rec["uncut_by_ref_distance_s"][f"{lo}-{hi}"] = {
                "n": int(m.sum()), "inc_p50": float(np.round(np.median(inc[m]), 4)),
                "inc_p99": float(np.round(np.percentile(inc[m], 99), 4)),
                "len_p99": float(np.round(np.percentile(L[m], 99), 4)), "flagged": int(f[m].sum()),
                "len_gt2": int((L[m] > 2).sum())}
    return rec


def b1_mesh(tag, case, r, path):
    """B v1 mesh of the final frame: the stored package (r0) and a re-extraction through the v2 runner's own path."""
    run, frames = P3.load_states(path)
    fr = frames[-1]
    st = dict(run, positions=np.asarray(fr["positions"], float), codes=fr["codes"], F=np.asarray(fr["F"], float),
              log=fr["log"])
    m = D.extract(B1, {}, run, st)
    V, Fc = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
    info = {"source": "reextracted (run_wp12_phase3_surface_v2.extract, B v1)", "reextraction_wall_s": m["wall_s"]}
    pkg = OUT_RUN/f"eval_{tag}/{case}_r{r}_package.npz"
    if r == 0 and pkg.exists():
        z = np.load(pkg)
        Vp, Fp = z["mesh_cut_aware_surface_v1_vertices"].astype(float), z["mesh_cut_aware_surface_v1_faces"].astype(np.int64)
        same = Fp.shape == Fc.shape and bool(np.array_equal(Fp, Fc))
        info["reextraction_vs_package"] = {
            "faces_equal": same, "vertex_max_abs_diff_m": float(np.abs(Vp-V).max()) if same and len(V) else None,
            "package_state_equal": bool(np.array_equal(z["positions"], fr["positions"].astype(z["positions"].dtype))
                                        and np.array_equal(z["F"], fr["F"].astype(z["F"].dtype))
                                        and np.array_equal(z["codes"], fr["codes"]))}
        info["source"] = f"package {rel(pkg)}"
        info["package_sha256"] = sha256(pkg)
        V, Fc = Vp, Fp
    return V, Fc, info


def fold_layer(z, V, Fc):
    from scipy.spatial import cKDTree
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    from robotic_cutting_sim.cut_aware_surface import ReferenceLattice
    s = float(z["spacing_m"])
    lat = ReferenceLattice(z["lattice"], z["reference"], s)
    x, F = z["positions"][-1].astype(float), z["F"][-1].astype(float)
    _, u, a, b, _, inc = uncut_links(lat, x, F, z["codes"][-1], s)
    pinc = np.zeros(lat.n)
    np.maximum.at(pinc, a, inc)
    np.maximum.at(pinc, b, inc)
    pairs = np.asarray(SM2.intersecting_pairs(V, Fc), np.int64).reshape(-1, 2)
    _, nn = cKDTree(x).query(V[Fc].mean(1), k=8)
    tri = pinc[nn].max(1)
    hit = np.unique(pairs.ravel()) if len(pairs) else np.zeros(0, np.int64)
    rec = {"triangles": int(len(Fc)), "pairs": int(len(pairs)), "triangles_in_pairs": int(len(hit)),
           "tri_inc_p50_all_s": float(np.median(tri)), "tri_inc_p50_in_pairs_s": float(np.median(tri[hit])) if len(hit) else None,
           "pair_triangles_inc_gt_0.5s": int((tri[hit] > .5).sum()), "pair_triangles_inc_gt_1s": int((tri[hit] > 1).sum()),
           "frac_all_tri_inc_gt_0.5s": float((tri > .5).mean()), "frac_all_tri_inc_gt_1s": float((tri > 1).mean()),
           "frac_pair_tri_inc_gt_0.5s": float((tri[hit] > .5).mean()) if len(hit) else None,
           "frac_pair_tri_inc_gt_1s": float((tri[hit] > 1).mean()) if len(hit) else None,
           "rule": "per triangle: max incident uncut-link |dx - F_mean dX| over the 8 particles nearest the centroid"}
    return rec


def one_state(job):
    tag, case, r = job
    t0 = time.perf_counter()
    path = states_path(tag, case, r)
    z = np.load(path, allow_pickle=False)
    row = {"states_file": rel(path), "states_sha256": sha256(path), "link_layer": link_layer(z)}
    V, Fc, info = b1_mesh(tag, case, r, path)
    row["fold_layer"] = dict(fold_layer(z, V, Fc), mesh=info)
    ev = eval_path(tag, case, r)
    if ev.exists():
        e = json.loads(ev.read_text())
        m4 = (((e["frames"][-1]["reports"].get(B2) or {}).get("cut_aware_info") or {}).get("v2") or {}).get("M4_unlabelled_separation") or {}
        row["b_v2_recorded_M4"] = {"flagged_links": m4.get("flagged_links"), "fallback_particles": m4.get("fallback_particles"),
                                   "matches_this_replication": m4.get("flagged_links") == row["link_layer"]["flagged"]}
    row["wall_s"] = time.perf_counter()-t0
    return f"{tag}/{case}_r{r}", row


# ------------------------------------------------------------------------------------------------ data reads

def _scal(e, kind, key):
    return ((e["frames"][-1]["reports_v2"].get(kind) or {}).get("scalars") or {}).get(key)


def data_reads(repeats):
    evals = {(t, c, r): json.loads(eval_path(t, c, r).read_text()) for t in SETS for c in CASES for r in range(repeats)
             if eval_path(t, c, r).exists()}
    out = {"cases": {}, "missing_evaluations": [f"{t}/{c}_r{r}" for t in SETS for c in CASES for r in range(repeats)
                                                if (t, c, r) not in evals]}
    for (t, c, r), e in evals.items():
        fr = e["frames"][-1]
        vs = fr.get("v2_state") or {}
        row = {"P4_status": (vs.get("P4_reference_consistency") or {}).get("status"),
               "P3_floor_p95_s": (vs.get("P3_floor_residual") or {}).get("p95_spacings"),
               "current_p95_s": {k: _scal(e, k, "current_p95_s") for k in (A, B1, B2, NOM4)},
               "matched_normal_p95_deg": {k: _scal(e, k, "matched_normal_p95_deg") for k in (A, B1, B2, NOM4)},
               "B_v2_M4": {k: v for k, v in ((((fr["reports"].get(B2) or {}).get("cut_aware_info") or {}).get("v2") or {})
                                             .get("M4_unlabelled_separation") or {}).items()
                           if k in ("flagged_links", "fallback_particles")}}
        fl, ca, cb = row["P3_floor_p95_s"], row["current_p95_s"][A], row["current_p95_s"][B1]
        row["A_and_B1_below_floor"] = None if None in (fl, ca, cb) else bool(ca < fl and cb < fl)
        if c == "S4":
            row["M2_hinge_no_M4"] = (((fr["reports"].get(NOM4) or {}).get("cut_aware_info") or {}).get("v2") or {}).get("M2_hinge")
            row["G4_hold_verdicts"] = {k: v.get("verdict") for k, v in (e.get("g4_hold_phase") or {}).items()}
            row["G4_hold_no_M4_check_verdicts"] = sorted({ch.get("verdict") for ch in
                                                          ((e.get("g4_hold_phase") or {}).get(NOM4) or {}).get("checks", {}).values()})
        if c == "S7":
            row["S7_P7"] = {k: {"value": v.get("value"), "verdict": v.get("verdict"), "reference_status": v.get("reference_status")}
                            for k, v in (e.get("s7_p7") or {}).items() if isinstance(v, dict)}
        out["cases"][f"{t}/{c}_r{r}"] = row
    args = type("Args", (), {"out": OUT_RUN, "repeats": repeats})()
    cf, b1 = {}, {}
    for t in SETS:
        res = D.load_results(args, t)
        if not res:
            continue
        for kind, sink in ((NOM4, cf), (B1, b1)):
            g = D.gate_v2(res, kind)
            sink[t] = {"label": LABEL_CF if kind == NOM4 else "recomputation of the recorded protocol-v2 gate (B v1)",
                       "parts": g["parts"], "verdict_if_it_were_a_gate" if kind == NOM4 else "verdict": g["verdict"],
                       "B_worse": {k: v for k, v in g["B_worse"].items() if v},
                       "current_p95_decisions": g["current_p95_decisions"], "coverage_decisions": g["coverage_decisions"],
                       "topology_decisions": g["topology_decisions"]}
    out["counterfactual_no_M4_ablation_protocol_v2"] = {
        "label": LABEL_CF, "kind": NOM4,
        "note": "labelled ablation (M1+M2+M3) compared with A by the v2 runner's gate_v2 on the stored evaluations; "
                "never a gate verdict, never a method result, not a shipped profile", "per_set": cf}
    out["b_v1_protocol_v2_recomputed"] = b1
    return out


# ------------------------------------------------------------------------------------------------ reproduction

# key: (decision value, decimals or None for exact, basis "r0" | "any_repeat", decision section, note)
DECISION_EXPECTED = {
    # section 1, F3 row (cases of m4_where.py, r0)
    "s1.F3.first_layer_inc_p50_min_max_s": ([0.07, 0.71], 2, "r0", "§1 F3", "uncut links <= 1 s from a Hermite point, m4_where cases"),
    "s1.F3.first_layer_inc_p99_min_max_s_excl_S7": ([1.3, 2.5], 1, "r0", "§1 F3", "m4_where cases except S7"),
    "s1.F3.first_layer_inc_p99_S7_H2_saved_s": ([7.5, 11.4], 1, "r0", "§1 F3", "S7 on H2 and saved"),
    "s1.F3.beyond_6s_inc_p50_min_max_s": ([0.001, 0.026], 3, "r0", "§1 F3", "m4_where cases"),
    "s1.F3.flagged_endpoint_detF_p5min_p95max": ([0.95, 1.07], 2, "r0", "§1 F3", "m4_codes cases"),
    "s1.F3.P4_consistent_cases_saved": (["S1_G3a", "S4"], None, "r0", "§1 F3", "P4 INCONSISTENT on every state except G3a and S4"),
    "s1.F3.P4_consistent_cases_H1": (["S1_G3a", "S4"], None, "r0", "§1 F3", ""),
    "s1.F3.P4_consistent_cases_H2": (["S1_G3a", "S4"], None, "r0", "§1 F3", ""),
    # section 1, M4 table
    "s1.M4.saved_G3a_G3b_S6.flagged": ([140, 27, 2], 0, "r0", "§1 M4", ""),
    "s1.M4.saved_G3a_G3b_S6.within_1s": ([140, 27, 2], 0, "r0", "§1 M4", "'all'"),
    "s1.M4.saved_G3a_G3b_S6.ref_dist_min_max_s": ([0.50, 0.71], 2, "r0", "§1 M4", "'0.50-0.71 s: the face layer'"),
    "s1.M4.saved_G3a_G3b_S6.in_plane": ([140, 26, 2], 0, "r0", "§1 M4", ""),
    "s1.M4.saved_G3a_G3b_S6.same_code": ([140, 27, 2], 0, "r0", "§1 M4", "'identical code'"),
    "s1.M4.saved_G3a_G3b_S6.new_pieces": ([0, 0, 0], 0, "r0", "§1 M4", ""),
    "s1.M4.saved_S4.flagged_within1s_inplane_mix": ([48, 48, 48, 48], 0, "r0", "§1 M4", "all <= 1 s, in plane, code 0 vs digit"),
    "s1.M4.saved_S4.ref_z_mm_min_max": ([6.0, 6.0], 1, "r0", "§1 M4", "all at z = 6 mm"),
    "s1.M4.saved_S4.new_pieces": (0, 0, "r0", "§1 M4", ""),
    "s1.M4.saved_S3_S5.flagged": ([29, 97], 0, "r0", "§1 M4", ""),
    "s1.M4.saved_S3_S5.within_1s": ([22, 67], 0, "r0", "§1 M4", ""),
    "s1.M4.saved_S5.same_mixed": ([61, 36], 0, "r0", "§1 M4", ""),
    "s1.M4.saved_S3_S5.new_pieces_and_particles": ([2, 2, 2, 2], 0, "r0", "§1 M4", "2 single particles each"),
    "s1.M4.H2_S5.flagged_within1s_3to6s_alongN": ([621, 232, 238, 271], 0, "r0", "§1 M4", ""),
    "s1.M4.H2_S5.same_mixed_newpieces": ([393, 228, 0], 0, "r0", "§1 M4", ""),
    "s1.M4.H2_S6.flagged_within1s_3to6s_alongN_exterior": ([595, 8, 581, 593, 301], 0, "r0", "§1 M4", ""),
    "s1.M4.H2_S6.same_mixed_newpieces": ([444, 151, 0], 0, "r0", "§1 M4", ""),
    "s1.M4.S7_saved_H2.flagged": ([134, 167], 0, "r0", "§1 M4", ""),
    "s1.M4.S7_saved_H2.within_1s": ([106, 92], 0, "r0", "§1 M4", ""),
    "s1.M4.S7_saved_H2.flagged_links_not_code_2_2": ([0, 0], 0, "r0", "§1 M4", "'Endpoint codes: all code 2'"),
    "s1.M4.S7_saved_H2.new_pieces": ([48, 35], 0, "r0", "§1 M4", ""),
    "s1.M4.S7_saved_H2.particles_in_new_pieces": ([80, 88], 0, "r0", "§1 M4", ""),
    "s1.M4.S2_S8_flagged_max_every_set": (0, 0, "r0", "§1 M4", "S2 and S8 flag 0 on every set"),
    "s1.M4.through_cut_particles_split_off_max_all_states": (0, 0, "r0", "§1 M4", "'splits off nothing in any through-cut' "
                                                             "(every set and repeat, all cases but S7)"),
    "s1.M4.S7_flagged_length_p50_H2_saved_s": ([2.9, 4.8], 1, "r0", "§1 M4", "'low mode (P50 2.9-4.8 s)'"),
    "s1.M4.B_v2_fallback_cells_H2_S5_S6": ([1107, 1140], 0, "any_repeat", "§1 M4", "'1,107 and 1,140 cells'"),
    # section 2, folds (saved r0, B v1 package meshes)
    "s2.fold.S1_G3b.pair_tris_gt_half_s_of_pair_tris": ([83, 83], 0, "r0", "§2", ""),
    "s2.fold.S3.pair_tris_gt_half_s_of_pair_tris": ([22, 22], 0, "r0", "§2", ""),
    "s2.fold.S5.pair_tris_gt_half_s_of_pair_tris": ([19, 19], 0, "r0", "§2", ""),
    "s2.fold.S4.pair_tris_gt_half_s_of_pair_tris": ([145, 163], 0, "r0", "§2", ""),
    "s2.fold.all_tris_gt_half_s_pct_G3b_S3_S5_S4": ([16, 29, 43, 5], 0, "r0", "§2", ""),
    "s2.fold.median_inc_at_folds_min_max_s": ([2.0, 3.1], 1, "r0", "§2", "G3b, S3, S5, S4"),
    "s2.fold.median_inc_overall_min_s": (0.006, 3, "r0", "§2", ""),
    "s2.fold.median_inc_overall_max_s": (0.35, 2, "r0", "§2", ""),
    # section 1 F1 / F2 / F4 (stored evaluations)
    "s1.F1.saved_S6_current_p95_B1_A": ([0.381, 0.424], 3, "any_repeat", "§1 F1", ""),
    "s1.F1.H1_S6_current_p95_B1_A": ([0.38, 0.45], 2, "any_repeat", "§1 F1", ""),
    "s1.F1.H2_S6_current_p95_B1_A": ([0.356, 0.55], [3, 2], "any_repeat", "§1 F1", "B1 at 3, A at 2 decimals"),
    "s1.F2.saved_M2_mixed_regions_moved": ([25, 25], 0, "any_repeat", "§1 F2", "no-M4 ablation, final frame"),
    "s1.F2.S4_current_p95_B1_saved_H1": ([0.96, 0.88], 2, "any_repeat", "§1 F2/§3.1", ""),
    "s1.F2.S4_current_p95_noM4_saved_H1": ([0.44, 0.45], 2, "any_repeat", "§1 F2/§3.1", ""),
    "s1.F2.S4_current_p95_A_saved_H1": ([0.77, 0.69], 2, "any_repeat", "§3.1", ""),
    "s1.F2.H2_S4_matched_normal_p95_B1_noM4_deg": ([9.9, 15.7], 1, "any_repeat", "§1 F2", ""),
    "s1.F2.noM4_G4_hold_check_verdicts_all_sets": (["NOT_RUN"], None, "r0", "§1 F2/D5", "every set, both repeats"),
    "s1.F4.saved_matched_normal_p95_B1_A.S1_G3a": ([30, 49], 0, "any_repeat", "§1 F4", ""),
    "s1.F4.saved_matched_normal_p95_B1_A.S1_G3b": ([8.4, 14.7], 1, "any_repeat", "§1 F4", ""),
    "s1.F4.saved_matched_normal_p95_B1_A.S2": ([5.9, 15.7], 1, "any_repeat", "§1 F4", ""),
    "s1.F4.saved_matched_normal_p95_B1_A.S3": ([6.0, 12.3], 1, "any_repeat", "§1 F4", ""),
    "s1.F4.saved_matched_normal_p95_B1_A.S5": ([6.9, 10.1], 1, "any_repeat", "§1 F4", ""),
    "s1.F4.saved_matched_normal_p95_B1_A.S6": ([5.8, 8.2], 1, "any_repeat", "§1 F4", ""),
    "s1.F4.saved_matched_normal_p95_B1_A.S8": ([5.9, 6.3], 1, "any_repeat", "§1 F4", ""),
    # section 3.1 counterfactual (labelled, never a gate)
    "s3.CF.saved.not_worse_better_p95": ([False, False], None, "r0", "§3.1", LABEL_CF),
    "s3.CF.H1.not_worse_better_p95": ([False, False], None, "r0", "§3.1", LABEL_CF),
    "s3.CF.H2.not_worse_better_p95_topology": ([False, False, False], None, "r0", "§3.1", LABEL_CF),
    "s3.CF.verdicts_saved_H1_H2": (["FAIL", "FAIL", "FAIL"], None, "r0", "§3.1", LABEL_CF),
    "s3.S7_P7_saved_B1_B2": ([4.71, 0.24], 2, "any_repeat", "§3.1", "M4 on S7: P7 4.71 -> 0.24"),
    # section 5 defects
    "s5.D1.saved.cases_not_both_below_floor": (["S1_G3a", "S4"], None, "any_repeat", "§5 D1", "'7 of 9 cases on every set (all but G3a and S4)'"),
    "s5.D1.H1.cases_not_both_below_floor": (["S1_G3a", "S4"], None, "any_repeat", "§5 D1", ""),
    "s5.D1.H2.cases_not_both_below_floor": (["S1_G3a", "S4"], None, "any_repeat", "§5 D1", ""),
    "s5.D1.saved_G3b_floor": (0.654, 3, "any_repeat", "§5 D1", ""),
    "s5.D1.saved_G3b_A_B1": ([0.441, 0.487], 3, "any_repeat", "§5 D1", ""),
    "s5.D1.saved_S2_floor": (0.878, 3, "any_repeat", "§5 D1", ""),
    "s5.D1.saved_S2_A_B1": ([0.58, 0.54], 2, "any_repeat", "§5 D1", ""),
    "s5.D1.saved_S8_floor": (0.578, 3, "any_repeat", "§5 D1", ""),
    "s5.D1.saved_S8_A_B1": ([0.36, 0.39], 2, "any_repeat", "§5 D1", ""),
    "s5.D6.S7_P7_B1_verdicts_all_sets": (["INCONCLUSIVE_REFERENCE"], None, "r0", "§5 D6", "every set, both repeats"),
}


def _matches(expected, measured, decimals):
    if measured is None:
        return False
    if decimals is None:
        return expected == measured
    e, m = np.atleast_1d(np.asarray(expected, float)), np.atleast_1d(np.asarray(measured, float))
    if e.shape != m.shape:
        return False
    dec = np.broadcast_to(np.asarray(decimals, int), e.shape)
    return bool(all(abs(round(float(mi), int(di))-ei) <= .5*10.**-int(di)+1e-12 for ei, mi, di in zip(e, m, dec)))


def extension_all_states(states):
    """The decision's link-layer and fold readings over every state (all sets, both repeats), beyond its r0 case list."""
    rows = {k: (v["link_layer"], v["fold_layer"]) for k, v in states.items()}
    band = lambda l, b, key: ((l["uncut_by_ref_distance_s"].get(b) or {}).get(key))    # noqa: E731
    thru = {k: v for k, v in rows.items() if "/S7_" not in k}
    chip = {k: v for k, v in rows.items() if "/S7_" in k}

    def rng(group, b, key):
        vals = [(band(l, b, key), k) for k, (l, _) in group.items() if band(l, b, key) is not None]
        return {"min": min(vals), "max": max(vals)} if vals else None
    folds = [(k, f) for k, (_, f) in rows.items() if f["triangles_in_pairs"]]
    return {"states": len(rows),
            "first_layer_inc_p50_s_excl_S7": rng(thru, "0-1", "inc_p50"), "first_layer_inc_p99_s_excl_S7": rng(thru, "0-1", "inc_p99"),
            "first_layer_inc_p99_s_S7": rng(chip, "0-1", "inc_p99"), "beyond_6s_inc_p50_s": rng(rows, f"6-{np.inf}", "inc_p50"),
            "flagged_per_state": {k: l["flagged"] for k, (l, _) in rows.items()},
            "split_off_pieces_particles": {k: [l["new_lattice_pieces_from_flagged_removal"], l["particles_in_new_pieces"]]
                                           for k, (l, _) in rows.items() if l["new_lattice_pieces_from_flagged_removal"]},
            "fold_states_with_pairs": len(folds),
            "fold_states_pair_fraction_gt_overall_fraction": sum(f["frac_pair_tri_inc_gt_0.5s"] > f["frac_all_tri_inc_gt_0.5s"] for _, f in folds),
            "fold_pair_fraction_gt_half_s_min": min(((f["frac_pair_tri_inc_gt_0.5s"], k) for k, f in folds), default=None),
            "fold_median_inc_in_pairs_s_min_max": [min(((f["tri_inc_p50_in_pairs_s"], k) for k, f in folds), default=None),
                                                   max(((f["tri_inc_p50_in_pairs_s"], k) for k, f in folds), default=None)]}


def decision_reproduction(states, reads, repeats):
    ll = lambda t, c, r=0: (states.get(f"{t}/{c}_r{r}") or {}).get("link_layer") or {}     # noqa: E731
    fl = lambda c: (states.get(f"saved/{c}_r0") or {}).get("fold_layer") or {}          # noqa: E731
    rd = lambda t, c, r: reads["cases"].get(f"{t}/{c}_r{r}") or {}                     # noqa: E731
    band = lambda rec, k, key: ((rec.get("uncut_by_ref_distance_s") or {}).get(k) or {}).get(key)   # noqa: E731
    INF = f"6-{np.inf}"
    got = {}

    # F3 bands over the m4_where cases
    first = [(t, c, band(ll(t, c), "0-1", "inc_p50"), band(ll(t, c), "0-1", "inc_p99")) for t, c in M4_WHERE_CASES]
    p50 = [x[2] for x in first if x[2] is not None]
    got["s1.F3.first_layer_inc_p50_min_max_s"] = [min(p50), max(p50)] if p50 else None
    p99 = [x[3] for x in first if x[3] is not None and x[1] != "S7"]
    got["s1.F3.first_layer_inc_p99_min_max_s_excl_S7"] = [min(p99), max(p99)] if p99 else None
    got["s1.F3.first_layer_inc_p99_S7_H2_saved_s"] = [band(ll("H2", "S7"), "0-1", "inc_p99"), band(ll("saved", "S7"), "0-1", "inc_p99")]
    far = [band(ll(t, c), INF, "inc_p50") for t, c in M4_WHERE_CASES]
    far = [v for v in far if v is not None]
    got["s1.F3.beyond_6s_inc_p50_min_max_s"] = [min(far), max(far)] if far else None
    dets = [ll(t, c).get("flagged_endpoint_detF_p5_p50_p95") for t, c in M4_CODES_CASES]
    dets = [d for d in dets if d]
    got["s1.F3.flagged_endpoint_detF_p5min_p95max"] = [min(d[0] for d in dets), max(d[2] for d in dets)] if dets else None
    for t in SETS:
        cons = sorted({c for c in CASES for r in range(repeats) if rd(t, c, r).get("P4_status") == "CONSISTENT"})
        got[f"s1.F3.P4_consistent_cases_{t}"] = cons
    # M4 table
    trio = [ll("saved", c) for c in ("S1_G3a", "S1_G3b", "S6")]
    for key, field in (("flagged", "flagged"), ("within_1s", "flagged_within_1s"), ("in_plane", "flagged_in_plane"),
                       ("same_code", "flagged_same_code"), ("new_pieces", "new_lattice_pieces_from_flagged_removal")):
        got[f"s1.M4.saved_G3a_G3b_S6.{key}"] = [x.get(field) for x in trio]
    dd = [x.get("flagged_ref_dist_to_hermite_s_p0_50_95_100") for x in trio]
    got["s1.M4.saved_G3a_G3b_S6.ref_dist_min_max_s"] = [min(d[0] for d in dd), max(d[3] for d in dd)] if all(dd) else None
    s4 = ll("saved", "S4")
    got["s1.M4.saved_S4.flagged_within1s_inplane_mix"] = [s4.get(k) for k in ("flagged", "flagged_within_1s", "flagged_in_plane", "flagged_zero_vs_digit_mix")]
    z4 = s4.get("flagged_ref_z_mm_p0_50_100")
    got["s1.M4.saved_S4.ref_z_mm_min_max"] = [z4[0], z4[2]] if z4 else None
    got["s1.M4.saved_S4.new_pieces"] = s4.get("new_lattice_pieces_from_flagged_removal")
    s3, s5 = ll("saved", "S3"), ll("saved", "S5")
    got["s1.M4.saved_S3_S5.flagged"] = [s3.get("flagged"), s5.get("flagged")]
    got["s1.M4.saved_S3_S5.within_1s"] = [s3.get("flagged_within_1s"), s5.get("flagged_within_1s")]
    got["s1.M4.saved_S5.same_mixed"] = [s5.get("flagged_same_code"), s5.get("flagged_zero_vs_digit_mix")]
    got["s1.M4.saved_S3_S5.new_pieces_and_particles"] = [s3.get("new_lattice_pieces_from_flagged_removal"), s5.get("new_lattice_pieces_from_flagged_removal"),
                                                         s3.get("particles_in_new_pieces"), s5.get("particles_in_new_pieces")]
    h5, h6 = ll("H2", "S5"), ll("H2", "S6")
    got["s1.M4.H2_S5.flagged_within1s_3to6s_alongN"] = [h5.get(k) for k in ("flagged", "flagged_within_1s", "flagged_3_to_6s", "flagged_along_N")]
    got["s1.M4.H2_S5.same_mixed_newpieces"] = [h5.get(k) for k in ("flagged_same_code", "flagged_zero_vs_digit_mix", "new_lattice_pieces_from_flagged_removal")]
    got["s1.M4.H2_S6.flagged_within1s_3to6s_alongN_exterior"] = [h6.get(k) for k in ("flagged", "flagged_within_1s", "flagged_3_to_6s", "flagged_along_N", "flagged_exterior_endpoint")]
    got["s1.M4.H2_S6.same_mixed_newpieces"] = [h6.get(k) for k in ("flagged_same_code", "flagged_zero_vs_digit_mix", "new_lattice_pieces_from_flagged_removal")]
    s7 = [ll("saved", "S7"), ll("H2", "S7")]
    got["s1.M4.S7_saved_H2.flagged"] = [x.get("flagged") for x in s7]
    got["s1.M4.S7_saved_H2.within_1s"] = [x.get("flagged_within_1s") for x in s7]
    got["s1.M4.S7_saved_H2.flagged_links_not_code_2_2"] = [sum(n for v, n in (x.get("flagged_link_code_pairs") or {}).items() if v != "2-2") for x in s7]
    got["s1.M4.S7_saved_H2.new_pieces"] = [x.get("new_lattice_pieces_from_flagged_removal") for x in s7]
    got["s1.M4.S7_saved_H2.particles_in_new_pieces"] = [x.get("particles_in_new_pieces") for x in s7]
    s28 = [ll(t, c, r).get("flagged") for t in SETS for c in ("S2", "S8") for r in range(repeats)]
    got["s1.M4.S2_S8_flagged_max_every_set"] = max(s28) if s28 and None not in s28 else None
    tc = [row["link_layer"]["particles_in_new_pieces"] for k, row in states.items() if "/S7_" not in k]
    got["s1.M4.through_cut_particles_split_off_max_all_states"] = max(tc) if tc else None
    got["s1.M4.S7_flagged_length_p50_H2_saved_s"] = [(ll("H2", "S7").get("flagged_length_s") or [None]*4)[1],
                                                       (ll("saved", "S7").get("flagged_length_s") or [None]*4)[1]]
    # folds
    pct = []
    meds = []
    over = []
    for c in FOLD_CASES:
        f = fl(c)
        got[f"s2.fold.{c}.pair_tris_gt_half_s_of_pair_tris"] = [f.get("pair_triangles_inc_gt_0.5s"), f.get("triangles_in_pairs")]
        pct.append(None if f.get("frac_all_tri_inc_gt_0.5s") is None else 100*f["frac_all_tri_inc_gt_0.5s"])
        meds.append(f.get("tri_inc_p50_in_pairs_s"))
        over.append(f.get("tri_inc_p50_all_s"))
    got["s2.fold.all_tris_gt_half_s_pct_G3b_S3_S5_S4"] = pct if None not in pct else None
    got["s2.fold.median_inc_at_folds_min_max_s"] = [min(meds), max(meds)] if None not in meds else None
    got["s2.fold.median_inc_overall_min_s"] = min(over) if None not in over else None
    got["s2.fold.median_inc_overall_max_s"] = max(over) if None not in over else None

    # stored-evaluation numbers: per repeat and the repeat mean
    def per_rep(fn):
        vals = {f"r{r}": fn(r) for r in range(repeats)}
        ok = [v for v in vals.values() if v is not None]
        if len(ok) == repeats and ok:
            try:
                vals["mean"] = np.mean(np.asarray(ok, float), 0).tolist()
            except (TypeError, ValueError):
                pass
        return vals
    cp = lambda t, c, k: (lambda r: rd(t, c, r).get("current_p95_s", {}).get(k))    # noqa: E731
    mn = lambda t, c, k: (lambda r: rd(t, c, r).get("matched_normal_p95_deg", {}).get(k))   # noqa: E731
    two = lambda f1, f2: (lambda r: None if None in (f1(r), f2(r)) else [f1(r), f2(r)])   # noqa: E731
    for t in SETS:
        got[f"s1.F1.{t}_S6_current_p95_B1_A"] = per_rep(two(cp(t, "S6", B1), cp(t, "S6", A)))
    hinge = lambda r: (lambda h: None if not h else [h.get("mixed_regions"), h.get("moved")])(rd("saved", "S4", r).get("M2_hinge_no_M4"))   # noqa: E731
    got["s1.F2.saved_M2_mixed_regions_moved"] = per_rep(hinge)
    got["s1.F2.S4_current_p95_B1_saved_H1"] = per_rep(two(cp("saved", "S4", B1), cp("H1", "S4", B1)))
    got["s1.F2.S4_current_p95_noM4_saved_H1"] = per_rep(two(cp("saved", "S4", NOM4), cp("H1", "S4", NOM4)))
    got["s1.F2.S4_current_p95_A_saved_H1"] = per_rep(two(cp("saved", "S4", A), cp("H1", "S4", A)))
    got["s1.F2.H2_S4_matched_normal_p95_B1_noM4_deg"] = per_rep(two(mn("H2", "S4", B1), mn("H2", "S4", NOM4)))
    g4 = sorted({v for t in SETS for r in range(repeats) for v in rd(t, "S4", r).get("G4_hold_no_M4_check_verdicts", [])})
    got["s1.F2.noM4_G4_hold_check_verdicts_all_sets"] = g4 or None
    for c in ("S1_G3a", "S1_G3b", "S2", "S3", "S5", "S6", "S8"):
        got[f"s1.F4.saved_matched_normal_p95_B1_A.{c}"] = per_rep(two(mn("saved", c, B1), mn("saved", c, A)))
    cf = reads["counterfactual_no_M4_ablation_protocol_v2"]["per_set"]
    pt = lambda t, ks: [cf[t]["parts"][k] for k in ks] if t in cf else None    # noqa: E731
    got["s3.CF.saved.not_worse_better_p95"] = pt("saved", ("not_worse_on_any_metric", "better_current_p95_every_case"))
    got["s3.CF.H1.not_worse_better_p95"] = pt("H1", ("not_worse_on_any_metric", "better_current_p95_every_case"))
    got["s3.CF.H2.not_worse_better_p95_topology"] = pt("H2", ("not_worse_on_any_metric", "better_current_p95_every_case", "better_topology"))
    got["s3.CF.verdicts_saved_H1_H2"] = [cf[t]["verdict_if_it_were_a_gate"] for t in SETS if t in cf]
    p7 = lambda t, k: (lambda r: ((rd(t, "S7", r).get("S7_P7") or {}).get(k) or {}).get("value"))   # noqa: E731
    got["s3.S7_P7_saved_B1_B2"] = per_rep(two(p7("saved", B1), p7("saved", B2)))
    for t in SETS:
        got[f"s5.D1.{t}.cases_not_both_below_floor"] = {f"r{r}": [c for c in CASES if rd(t, c, r).get("A_and_B1_below_floor") is False]
                                                        for r in range(repeats)}
    for c, short in (("S1_G3b", "G3b"), ("S2", "S2"), ("S8", "S8")):
        got[f"s5.D1.saved_{short}_floor"] = per_rep(lambda r, c=c: rd("saved", c, r).get("P3_floor_p95_s"))
        got[f"s5.D1.saved_{short}_A_B1"] = per_rep(two(cp("saved", c, A), cp("saved", c, B1)))
    got["s5.D6.S7_P7_B1_verdicts_all_sets"] = sorted({((rd(t, "S7", r).get("S7_P7") or {}).get(B1) or {}).get("verdict")
                                                     for t in SETS for r in range(repeats)} - {None}) or None
    fb = lambda r: [((rd("H2", c, r).get("B_v2_M4") or {}).get("fallback_particles")) for c in ("S5", "S6")]   # noqa: E731
    got["s1.M4.B_v2_fallback_cells_H2_S5_S6"] = per_rep(lambda r: None if None in fb(r) else fb(r))

    table, not_reproduced, mixed = {}, [], []
    for key, (exp, dec, basis, sec, note) in DECISION_EXPECTED.items():
        m = got.get(key)
        status, used = "NOT_REPRODUCED", basis
        if basis == "any_repeat" and isinstance(m, dict):
            hits = [b for b, v in m.items() if _matches(exp, v, dec)]
            if hits:
                status, used = "REPRODUCED", hits[0]
            elif isinstance(exp, list) and dec is not None:
                per = [[b for b, v in m.items() if v is not None and len(np.atleast_1d(v)) == len(exp)
                        and _matches(exp[i], np.atleast_1d(v)[i], dec[i] if isinstance(dec, list) else dec)] for i in range(len(exp))]
                if all(per):
                    status, used = "REPRODUCED_MIXED_REPEATS", [p[0] for p in per]
        elif _matches(exp, m, dec):
            status, used = "REPRODUCED", "r0"
        table[key] = {"section": sec, "decision": exp, "measured": m, "decimals": dec, "basis": used,
                      "status": status, "reproduced": status == "REPRODUCED", "note": note}
        if status == "REPRODUCED_MIXED_REPEATS":
            mixed.append(key)
        elif status != "REPRODUCED":
            not_reproduced.append(key)
    return {"rule": "measured rounded to the decision's printed precision; link-layer and fold numbers r0; numbers read "
                    "from stored evaluations accept r0, r1 or the repeat mean (basis records which)",
            "entries": len(table), "reproduced": len(table)-len(not_reproduced)-len(mixed),
            "reproduced_only_with_mixed_repeats": mixed, "not_reproduced": not_reproduced, "table": table}


# Explanations attached to an entry only when it does not reproduce (or only with mixed repeats); the measured values
# they quote are those of the stored, deterministic inputs.
NON_REPRODUCTION_NOTES = {
    "s1.F3.first_layer_inc_p99_min_max_s_excl_S7":
        "H2 S4 (0 flagged links) has first-layer P99 0.92 s; the 1.3 s lower bound holds only for the m4_where states "
        "with flagged links (min saved S6 1.30 s)",
    "s1.M4.S7_saved_H2.flagged_links_not_code_2_2":
        "every flagged S7 link joins two equal codes, but they are code 1-1 (main body) as well as 2-2 (chip): "
        "'identical code', not 'all code 2'",
    "s2.fold.median_inc_at_folds_min_max_s":
        "S4 fold-triangle median is 1.67 s; the 2.0-3.1 s range holds for G3b, S3 and S5 (2.02-3.09 s)",
    "s1.M4.B_v2_fallback_cells_H2_S5_S6":
        "B v2 M4 fallback particles: H2 S5 1,107 (r0) / 1,109 (r1); H2 S6 1,142 (r0) / 1,140 (r1): the decision pairs "
        "S5 r0 with S6 r1",
    "s1.F1.saved_S6_current_p95_B1_A":
        "B v1 0.381 is r0 (r1 0.383); A 0.424 is the repeat mean (r0 0.423, r1 0.425)",
    "s1.F1.H2_S6_current_p95_B1_A":
        "B v1 0.356 is r0 (r1 and mean 0.357); A 0.55 is r1 or the mean (r0 0.559)",
    "s5.D1.H2.cases_not_both_below_floor":
        "H2 S7: A current P95 0.884 / 0.885 s lies above the P3 floor 0.660 / 0.661 s in both repeats, so H2 has 6 (r0) "
        "or 7 (r1) of 9 with S7 an exception; G3a is borderline on every set (A 0.692-0.699 s against floor 0.694-0.695 s, "
        "below it only in saved r1 and H2 r1)",
    "s1.M4.through_cut_particles_split_off_max_all_states":
        "removing the flagged links splits off 1-4 particles in 12 of 48 through-cut states (saved S3 and S5: 2 single "
        "particles, the decision's own table; H1 S3 2 pieces / 3 particles; H1 and H2 G3b 1; H2 S3 4 single particles), "
        "against 35-48 pieces / 80-88 particles in S7: 'nothing' should read 'at most 4 particles'",
    "s5.D1.saved.cases_not_both_below_floor":
        "saved r1: G3a A 0.693 s is below its floor 0.694 s (8 of 9); r0 reproduces",
}


def compact_evidence(out, target):
    rep = out["decision_reproduction"]
    states = out["states"]
    per_state = {}
    for k, row in states.items():
        l, f = row["link_layer"], row["fold_layer"]
        b = l["uncut_by_ref_distance_s"]
        per_state[k] = {"flagged": l["flagged"], "within_1s": l["flagged_within_1s"], "3_to_6s": l["flagged_3_to_6s"],
                        "in_plane": l["flagged_in_plane"], "along_N": l["flagged_along_N"],
                        "same_code": l["flagged_same_code"], "zero_vs_digit": l["flagged_zero_vs_digit_mix"],
                        "new_pieces": l["new_lattice_pieces_from_flagged_removal"], "particles_in_new_pieces": l["particles_in_new_pieces"],
                        "first_layer_inc_p50_p99_s": [(b.get("0-1") or {}).get("inc_p50"), (b.get("0-1") or {}).get("inc_p99")],
                        "beyond_6s_inc_p50_p99_s": [(b.get(f"6-{np.inf}") or {}).get("inc_p50"), (b.get(f"6-{np.inf}") or {}).get("inc_p99")],
                        "b_v1_pairs": f["pairs"], "pair_tris_gt_half_s": [f["pair_triangles_inc_gt_0.5s"], f["triangles_in_pairs"]],
                        "all_tris_gt_half_s_frac": round(f["frac_all_tri_inc_gt_0.5s"], 4),
                        "b_v2_recorded_M4_matches": (row.get("b_v2_recorded_M4") or {}).get("matches_this_replication")}
    cf = out["data_reads"]["counterfactual_no_M4_ablation_protocol_v2"]
    return {"schema": "wp12_phase3_closure_evidence_v1", "decision": "docs/reports/WP12_PHASE3_XHIGH_DECISION.md (commit a94bd9f)",
            "diagnostics": rel(target), "command": out["command"], "git_head": out["git_head"],
            "scope": out["scope"], "wall_s": out["wall_s"],
            "replication_checks": out["replication_checks"],
            "extension_all_states": {k: v for k, v in out["extension_all_states"].items() if k != "flagged_per_state"},
            "counterfactual_no_M4_ablation_protocol_v2": {"label": cf["label"], "note": cf["note"],
                                                           "per_set": {t: {"parts": v["parts"], "verdict_if_it_were_a_gate": v["verdict_if_it_were_a_gate"]}
                                                                       for t, v in cf["per_set"].items()}},
            "b_v1_protocol_v2_recomputed_verdicts": {t: v["verdict"] for t, v in out["data_reads"]["b_v1_protocol_v2_recomputed"].items()},
            "decision_reproduction": {"rule": rep["rule"], "entries": rep["entries"], "reproduced": rep["reproduced"],
                                      "not_reproduced": [{"key": k, "decision": rep["table"][k]["decision"], "measured": rep["table"][k]["measured"],
                                                          "note": NON_REPRODUCTION_NOTES.get(k, "")} for k in rep["not_reproduced"]],
                                      "reproduced_only_with_mixed_repeats": [
                                          {"key": k, "decision": rep["table"][k]["decision"], "measured": rep["table"][k]["measured"],
                                           "basis": rep["table"][k]["basis"], "note": NON_REPRODUCTION_NOTES.get(k, "")}
                                          for k in rep["reproduced_only_with_mixed_repeats"]],
                                      "table": {k: {"decision": v["decision"], "measured": v["measured"], "basis": v["basis"],
                                                    "status": v["status"]} for k, v in rep["table"].items()}},
            "per_state_r0_r1": per_state}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="diagnostics JSON (default: next free "
                   "artifacts/wp12/phase3_surface_002/diagnostics_NNN.json); never overwritten")
    p.add_argument("--evidence", type=Path, default=EVIDENCE, help="compact evidence JSON; never overwritten")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--jobs", type=int, default=1, help="states in parallel (separate processes, CPU only)")
    args = p.parse_args()
    target = args.out or D.next_free(OUT_RUN, "diagnostics", ".json")
    target = target if target.is_absolute() else ROOT/target
    evidence = args.evidence if args.evidence.is_absolute() else ROOT/args.evidence
    for path in (target, evidence):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
    t0 = time.perf_counter()
    out = {"schema": "wp12_phase3_v2_diagnostics_v1", "command": " ".join([sys.executable, *sys.argv]),
           "git_head": P3.git("rev-parse", "HEAD"), "git_dirty_paths": len(P3.git("status", "--porcelain").splitlines()),
           "decision": "docs/reports/WP12_PHASE3_XHIGH_DECISION.md (commit a94bd9f) section 6 step 1",
           "scope": "read-only diagnostics of saved, H1 and H2 final frames (r0, r1); no method, protocol, threshold or "
                    "run directory changed; not a gate and not a verdict",
           "states": {}}
    jobs = [(t, c, r) for t in SETS for c in CASES for r in range(args.repeats)]
    out["missing_states"] = [f"{t}/{c}_r{r}" for t, c, r in jobs if not states_path(t, c, r).exists()]
    jobs = [j for j in jobs if states_path(*j).exists()]
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(one_state, jobs)
    else:
        results = [one_state(j) for j in jobs]
    for key, row in results:
        out["states"][key] = row
    print(f"{len(results)} states done", flush=True)
    out["data_reads"] = data_reads(args.repeats)
    rx = [row["fold_layer"]["mesh"].get("reextraction_vs_package") for row in out["states"].values()]
    rx = [x for x in rx if x]
    m4 = [row.get("b_v2_recorded_M4") for row in out["states"].values() if row.get("b_v2_recorded_M4")]
    out["replication_checks"] = {
        "b_v1_reextraction_vs_r0_packages": {"compared": len(rx), "faces_equal": sum(x["faces_equal"] for x in rx),
                                             "package_state_equal": sum(x["package_state_equal"] for x in rx),
                                             "vertex_max_abs_diff_m": max((x["vertex_max_abs_diff_m"] or 0.) for x in rx) if rx else None},
        "m4_flag_count_vs_b_v2_recorded": {"compared": len(m4), "equal": sum(bool(x["matches_this_replication"]) for x in m4),
                                           "unequal": [k for k, row in out["states"].items()
                                                       if row.get("b_v2_recorded_M4") and not row["b_v2_recorded_M4"]["matches_this_replication"]]}}
    out["decision_reproduction"] = decision_reproduction(out["states"], out["data_reads"], args.repeats)
    out["extension_all_states"] = extension_all_states(out["states"])
    for k in out["decision_reproduction"]["not_reproduced"]+out["decision_reproduction"]["reproduced_only_with_mixed_repeats"]:
        out["decision_reproduction"]["table"][k]["non_reproduction_note"] = NON_REPRODUCTION_NOTES.get(k, "")
    out["wall_s"] = time.perf_counter()-t0
    print("replication:", json.dumps(out["replication_checks"]), flush=True)
    print("not reproduced:", out["decision_reproduction"]["not_reproduced"], flush=True)
    print("reproduced only with mixed repeats:", out["decision_reproduction"]["reproduced_only_with_mixed_repeats"], flush=True)
    for path in (target, evidence):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(P3.jsonable(out), indent=1))
    ev = compact_evidence(out, target)
    ev["diagnostics_sha256"] = sha256(target)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(P3.jsonable(ev), indent=1))
    print(target)
    print(evidence)


if __name__ == "__main__":
    main()
