#!/usr/bin/env python3
"""WP-12 Phase 3 v2 matrix: {A, B v1, B v2} x {surface_protocol_v1, surface_protocol_v2} on saved and held-out states.

* A = ``particle_surface_v1`` (CuttingSurface current_volume_v1), B v1 = ``cut_aware_surface_v1`` (unchanged module),
  B v2 = ``cut_aware_surface_v2`` (``cut_aware_surface_v2.py``, frozen in ``<out>/METHOD_FROZEN.txt`` before any
  measurement). ``cut_aware_surface_v2_position_consistent`` (M5) is a separately labelled variant, reported only,
  final frames only (its recontact / hold rows read NOT_RUN). The midpoint-fallback variants are evaluated in G2 only
  (G2 midpoint rows); every other row reads NOT_RUN for them.
* Protocol v1 readings are the frozen ones of ``scripts/run_wp12_phase3_surface.py`` (imported; its functions are
  called unchanged with the B variant in the B slot). Protocol v2 readings use ``surface_metrics_v2`` (P1-P7) and the
  unchanged v1 rows, exactly as ``configs/wp12/surface_protocol_v2.json`` defines them.
* State sets: ``saved`` = ``artifacts/wp12/phase3_surface_001/<case>_r0/_r1_states.npz`` (read-only; S9 has no saved
  states, so its five v1 controls are re-run fresh at the v1 offsets, labelled); ``H1`` / ``H2`` = the held-out
  states declared in protocol v2 ``held_out_states`` (new physics runs, recorder on, 2 GPU repeats, same seed).
  S0 and G2 are analytic/static (G2 adds the held-out planes n213 and n60_xz).

Stages (each writes new files only; nothing is ever overwritten):
  record     <out>/protocol_record.json: both protocol SHA256s + freeze commits, METHOD_FROZEN.txt SHA256, the case
             list per state set, the declared held-out offsets (recomputed from the seeds), every reading. Must run
             first; <out> may contain only METHOD_FROZEN.txt.
  heldout    physics: <out>/heldout_<H>/<case>_r<k>_states.npz (+ _run.json)
  evaluate   <out>/eval_<set>/<case>_r<k>.json (+ r0 debug packages)
  static     <out>/eval_static/S0.json, G2.json
  gate       <out>/summary_NNN.json: threshold verdicts per (method, protocol, set, case, repeat), comparison gates
  figures    docs/media/wp12/phase3/phase3v2_<name>_A_B1_B2.png

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface_v2.py record
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface_v2.py heldout --variant H1 --device cuda:0
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface_v2.py evaluate --set saved --device cuda:0
  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase3_surface_v2.py gate
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
_spec = importlib.util.spec_from_file_location("wp12_phase3_surface", ROOT/"scripts/run_wp12_phase3_surface.py")
P3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P3)
P1 = P3.P1

S, DX, DT, CUT_SPEED, MOVE_SPEED = P3.S, P3.DX, P3.DT, P3.CUT_SPEED, P3.MOVE_SPEED
A, LEGACY, B1, B1MID = P3.A, P3.LEGACY, P3.B, P3.B_MID
B2 = "cut_aware_surface_v2"
B2PC = "cut_aware_surface_v2_position_consistent"
B2MID = "cut_aware_surface_v2_midpoint_fallback"
B2NOM4 = "cut_aware_surface_v2_ablation_no_M4"
B_GATED = (B1, B2)
REPORT_ONLY_KINDS = (B2PC, B2NOM4)
B_ALL = (B1, B2, B2PC, B2NOM4)
B_S9 = (B1, B2, B2PC)        # the S9 controls (held-out S9 runs inside the heldout stage) use this fixed tuple
PROTO1, PROTO2 = P3.PROTOCOL_FILE, ROOT/"configs/wp12/surface_protocol_v2.json"
FREEZE1, FREEZE2 = "b31553d", "9157725"
SAVED_RUN = ROOT/"artifacts/wp12/phase3_surface_001"
DEFAULT_OUT = ROOT/"artifacts/wp12/phase3_surface_002"
PHYSICAL = ("S1_G3a", "S1_G3b", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
SETS = ("saved", "H1", "H2")
SEEDS = {"S1_G3a": (43, 44), "S1_G3b": (3101, 3201), "S2": (3102, 3202), "S3": (3103, 3203), "S4": (3104, 3204),
         "S5": (3105, 3205), "S6": (3106, 3206), "S7": (3107, 3207), "S8": (3108, 3208), "S9": (3109, 3209)}
INCONCLUSIVE = "INCONCLUSIVE_REFERENCE"
S8_DENSE_EVERY = 40          # substeps (4 ms), protocol v2 held_out_states S8
S8_V1_EVERY = 400            # substeps (40 ms): the v1 S8 frame spacing, used for the protocol v1 reading of held-out S8

DECLARATION = {
    "schema": "wp12_phase3_surface_v2_run_declaration_v1",
    "protocols": {"v1": "configs/wp12/surface_protocol_v1.json (frozen b31553d) with the readings of "
                        "scripts/run_wp12_phase3_surface.py DECLARATION (functions called unchanged)",
                  "v2": "configs/wp12/surface_protocol_v2.json (frozen 9157725) via surface_metrics_v2 P1-P7 + unchanged rows"},
    "methods": {A: "particle_surface_v1 = CuttingSurface current_volume_v1 (as phase3_surface_001)",
                B1: "cut_aware_surface_v1 (cut_aware_surface.py, byte-identical to 89b289d), recorder log, total F, cell_face_v1",
                B2: "cut_aware_surface_v2 (cut_aware_surface_v2.py, M1-M4, frozen before measurement: METHOD_FROZEN.txt), "
                    "recorder log, total F, cell_face_v1",
                B2PC: "cut_aware_surface_v2 with M5 position_consistent_v1 embedding: separately labelled, reported only, "
                      "final frames only (recontact / hold rows NOT_RUN), never in a gate",
                B1MID: "B v1 with the R4 midpoint fallback: G2 midpoint rows only; every other row NOT_RUN",
                B2MID: "B v2 with the R4 midpoint fallback: G2 midpoint rows only; every other row NOT_RUN"},
    "v1_readings_for_B_v2": "every v1 function of run_wp12_phase3_surface.py receives B v2's mesh in the B slot (the "
                            "B-specific branches: reference mesh, provenance, audit components); A's S6 per-component "
                            "bound keeps B v1's reference cut area as in phase3_surface_001",
    "v2_readings": {
        "P1": "vertex component from component_ranges (surface_metrics_v2.vertex_component_from_ranges, ranked labels); "
              "classify_mesh_vertices_v2; P95/P50/bias over all cut-face vertices, max over margin >= 0.5 s",
        "P2": "matched_scale_normal vs normal_floor_split_halves (seed 20261031, core margin 3 s) + 1 deg; S5 per cut",
        "P3": "current_p95_rows (per side group, floor = locus leave-one-out residual P95 + 0.10 s, min 20 vertices, "
              "pooled remainder); the row passes when every evaluated group passes",
        "P4": "reference_consistency per state; the 'all' current-volume row (whole mesh vs sum (m/rho) det F) reads "
              "INCONCLUSIVE_REFERENCE unless CONSISTENT; gate metric volume_current_abs_rel_if_consistent",
        "P5": "S8: slow_frame_pairs (codes equal, material displacement P95 <= 0.10 s) over the state's consecutive "
              "frames (held-out: dense 4 ms frames; saved: the stored 40 ms frames), temporal_stability_v2 (IDW k 8), "
              "max of the per-pair P95; NOT_RUN without an eligible pair",
        "P6": "intersecting_pairs + attribute_intersections (face component from provenance) and "
              "faces_nearer_other_component_v2; only the unexplained counts are gate metrics",
        "P7": "S7: chip_volume_v2 on the lattice-faced part (B v2 particle cells = face_kind 4 and the fallback particles "
              "excluded) with the chip components' P4 status",
        "reference_space_A": "A has no reference mesh: pull-back X_p + F_p^-1 (v - x_p) through its P1 component (diagnostic, as v1)",
        "unchanged_rows": "S0, G2, S4 provenance, S6, S7 skipped fragments, S8 shared vertices / shell set, S9, volume "
                          "reference, fixed-state temporal: the v1 row functions of run_wp12_phase3_surface.py",
        "S8_frames": "held-out S8: light rows (shells, shared vertices) on every dense frame for v2; the protocol v1 "
                     "reading uses the frames at the v1 40 ms spacing (cut end, push start, every 400 substeps), its "
                     "slow-motion row as v1 (nearest-particle transport); self-intersections on the 40 ms frames only",
        "verdict_rank": "FAIL > NOT_EVALUABLE > INCONCLUSIVE_REFERENCE > NOT_RUN > PASS (REPORT_ONLY / NOT_APPLICABLE ignored)"},
    "gate_v2": "surface_protocol_v2 phase3_comparison_gate_v2: B v2 vs A under v2 decides (B v1 vs A under v2 reported "
               "beside; both under v1 printed beside). Per state set and case: P3.compare with GATE_METRICS_V2 (+ S5 per "
               "cut, S8 P5). PASS only if the rule holds on saved AND H1 AND H2",
    "never": "no target plane as a reference (G2 only); no physics, metric or threshold edits; nothing overwritten",
}


# ----------------------------------------------------------------------------------------------- utilities

jsonable, check, not_run, not_applicable = P3.jsonable, P3.check, P3.not_run, P3.not_applicable


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verdict_v2(checks):
    vs = [c["verdict"] for c in checks.values()]
    for v in ("FAIL", "NOT_EVALUABLE", INCONCLUSIVE, "NOT_RUN"):
        if v in vs:
            return v
    return "PASS" if vs else "NOT_EVALUABLE"


def check_bool(flag, *, value=None, threshold=None, note=None):
    if flag is None:
        return {"value": value, "threshold": threshold, "verdict": "NOT_EVALUABLE", **({"note": note} if note else {})}
    return {"value": value, "threshold": threshold, "verdict": "PASS" if flag else "FAIL", **({"note": note} if note else {})}


def offsets(seed):
    return 0.15+0.7*np.random.default_rng(seed).random(2)


def write_new(path, obj):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=1))


def next_free(directory, stem, suffix):
    k = 1
    while (directory/f"{stem}_{k:03d}{suffix}").exists():
        k += 1
    return directory/f"{stem}_{k:03d}{suffix}"


# ----------------------------------------------------------------------------------------------- surfaces

def make_b(kind, run, log):
    from robotic_cutting_sim import cut_aware_surface as CAS
    from robotic_cutting_sim import cut_aware_surface_v2 as CV2
    from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition
    part = CurrentMechanicalPartition(run["nodes"], run["weights"], run["gradients"], run["cuts"], encoding="packed2")
    kw = dict(cut_events_log=log, masses_kg=run["masses"], density_kg_m3=run["density"], strict_audit=False)
    if kind in (B1, B1MID):
        return CAS.CutAwareSurface(part, S, run["lattice"], run["reference"], **kw)
    if kind in (B2, B2MID):
        return CV2.CutAwareSurfaceV2(part, S, run["lattice"], run["reference"], **kw)
    if kind == B2PC:
        return CV2.CutAwareSurfaceV2(part, S, run["lattice"], run["reference"], embedding="position_consistent_v1", **kw)
    if kind == B2NOM4:
        return CV2.CutAwareSurfaceV2(part, S, run["lattice"], run["reference"], fixes=("M1", "M2", "M3"), **kw)
    raise ValueError(kind)


def uses_log(kind):
    return kind in (B1, B2, B2PC, B2NOM4)


def extract(kind, surfaces, run, st):
    log = st["log"] if uses_log(kind) else None
    if kind not in surfaces:
        surfaces[kind] = make_b(kind, run, log)
    return P3.b_extract(surfaces[kind], st, log)


def fixed_state_temporal_b(kind, mesh, st, run):
    """P3.fixed_state_temporal for any B variant: cached re-extraction and a fresh extractor of the same kind."""
    log = st["log"] if uses_log(kind) else None
    v0, f0 = mesh["vertices"], mesh["faces"]
    h0 = P3.topo_hash(f0, mesh["last"]["vertex_reference"])
    rows = {}
    for name, m in (("cached_again", P3.b_extract(mesh["surface"], st, log)),
                    ("fresh_extractor", P3.b_extract(make_b(kind, run, log), st, log))):
        same = m["faces"].shape == f0.shape and P3.topo_hash(m["faces"], m["last"]["vertex_reference"]) == h0
        res = float(np.abs(m["vertices"]-v0).max()) if same and len(v0) else None
        rows[name] = {"topology_hash_equal": bool(same), "vertex_residual_max_m": res, "update": m["info"]["update"],
                      "wall_s": m["wall_s"]}
    stable = all(r["topology_hash_equal"] for r in rows.values())
    residual = max((r["vertex_residual_max_m"] or 0.) for r in rows.values()) if stable else None
    return {"rows": rows, "topology_stable": stable, "residual_max_s": None if residual is None else residual/S,
            "topology_hash": h0}


def compact_v2_info(info):
    out = P3.compact_b_info(info)
    if "v2" in info:
        v2 = {k: v for k, v in info["v2"].items() if k != "frozen_parameters"}
        out["v2"] = v2
        out["faces_v2"] = info["faces"]
    return out


# ----------------------------------------------------------------------------------------------- v2 readings

def v2_state_terms(st, labels, locus, ext):
    """Per-state v2 references (method independent): P3 floor, P2 floor, P4 consistency."""
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    t = time.perf_counter()
    cuts = st["cuts"]
    floor = SM2.current_distance_floor(locus, labels, cuts) if len(locus["cut"]) else (np.zeros(0), np.zeros(0, int))
    nfloor = (SM2.normal_floor_split_halves(locus, labels, cuts, ext, core_margin_m=3*S) if len(locus["cut"])
              else (np.zeros(0), np.zeros(0, int)))
    cons = SM2.reference_consistency(st["lattice"], st["positions"], st["F"], st["codes"], labels,
                                     st["masses"]/st["density"], spacing=S, cuts=cuts)
    fr = np.asarray(floor[0], float)
    return {"floor": floor, "nfloor": nfloor, "consistency": cons, "wall_s": time.perf_counter()-t,
            "floor_summary": P3.q(fr[np.isfinite(fr)]),
            "normal_floor_p95_deg": float(np.percentile(nfloor[0], 95)) if len(nfloor[0]) else None}


def v2_block(signed, region, vkey, cls, locus, terms, Xref, angles, tcut, *, select=None, cut=None):
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    sel = np.ones(len(region), bool) if select is None else np.asarray(select, bool)
    d = SM2.distance_rows(signed, region, cls, locus, spacing=S, select=sel)
    dc = SM2.distance_rows(signed, region, cls, locus, spacing=S, select=sel & cls["core"][region])
    p3 = SM2.current_p95_rows(signed, vkey, terms["floor"][0], terms["floor"][1], spacing=S, select=sel)
    rr = SM2.reference_distance_rows(Xref, region, cls, locus, spacing=S, select=sel)
    rr.pop("distances_m", None)
    fa, fc = terms["nfloor"]
    if cut is None:
        nr = SM2.normal_row(angles, fa)
    else:
        nr = SM2.normal_row(angles[tcut == cut], fa[fc == cut])
    return {"distance": d, "distance_core": dc, "current_p95_P3": p3, "reference": rr, "normal_P2": nr}


def v2_report(kind, m, st, labels, rank, locus, ext, terms, rep1):
    """surface_protocol_v2 readings of one mesh (P1-P6; unchanged rows taken from the v1 report ``rep1``)."""
    from robotic_cutting_sim import surface_metrics_v1 as SM1
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    t0 = time.perf_counter()
    v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
    fc = m["face_component"]
    cuts = st["cuts"]
    vc = SM2.vertex_component_from_ranges(len(v), m["info"]["component_ranges"], rank)
    cls = SM2.classify_mesh_vertices_v2(v, vc, st["positions"], labels, locus, ext, spacing=S)
    out = {"region_counts": cls["counts"]}
    cell_v = (m.get("last") or {}).get("vertex_particle_cell")
    if cell_v is not None and np.any(cell_v):
        # review section 2 M4: particle cells are never cut faces and never in cut-face metrics (addendum 001)
        cls["kind"] = np.where(cell_v, 0, cls["kind"]).astype(cls["kind"].dtype)
        cls["core"] = cls["core"] & ~cell_v
        out["particle_cell_vertices_excluded_from_cut_face_rows"] = int(np.sum(cell_v))
    region = np.flatnonzero(cls["kind"] == 2)
    sc1 = rep1["scalars"]
    blocks = None
    if len(locus["cut"]) and len(region):
        region, signed, vkey = SM2.cut_face_signed_distances(v, cls, locus, labels, cuts)
        if kind == A:
            Xref = SM1.pull_back(v[region], st["positions"], st["reference"], st["F"], labels, cls["component"][region])
        else:
            Xref = np.asarray(m["last"]["vertex_reference"], float)[region]
        angles, tcut, _ = SM2.matched_scale_normal(v, f, cls, locus, labels, cuts)
        blocks = {"all": v2_block(signed, region, vkey, cls, locus, terms, Xref, angles, tcut)}
        if cuts > 1:
            rcut = locus["cut"][cls["row"][region]]
            for k in np.unique(locus["cut"]):
                blocks[f"cut{int(k)}"] = v2_block(signed, region, vkey, cls, locus, terms, Xref, angles, tcut,
                                                  select=rcut == k, cut=int(k))
        out["blocks"] = blocks
    else:
        out["status"] = "NO_CUT_FACE"
    pairs = SM2.intersecting_pairs(v, f)
    att = SM2.attribute_intersections(v, f, pairs, fc, cls, st["positions"], labels, st["codes"], locus, spacing=S, cuts=cuts)
    fn = SM2.faces_nearer_other_component_v2(v, f, st["positions"], labels, fc, spacing=S)
    out["intersections_P6"], out["faces_nearer_other_component_P6"] = att, fn
    if kind in (B2, B2PC, B2NOM4) and "vertices_before_fold_guard" in m["last"]:
        vb = np.asarray(m["last"]["vertices_before_fold_guard"], float)
        lat = ~np.asarray(m["last"]["face_particle_cell"], bool)
        out["intersections_before_fold_guard"] = int(len(SM2.intersecting_pairs(vb, f)))
        out["intersections_lattice_faces_only"] = int(len(SM2.intersecting_pairs(v, f[lat])))
        out["intersections_lattice_faces_only_before_fold_guard"] = int(len(SM2.intersecting_pairs(vb, f[lat])))
    cons = terms["consistency"]
    rel = rep1["volume_checks"]["current_rel_error"]
    out["volume_current_P4"] = SM2.volume_current_row(rel, cons["status"], .02)
    a = (blocks or {}).get("all")
    g = lambda d, k: None if d is None else d.get(k)  # noqa: E731
    dist = None if a is None else a["distance"]
    sc = {"current_p50_s": g(dist, "p50_s"), "current_p95_s": g(dist, "p95_s"),
          "current_max_unambiguous_s": g(dist, "max_unambiguous_s"),
          "current_core_p95_s": None if a is None else a["distance_core"]["p95_s"],
          "reference_p95_s": None if a is None else a["reference"]["p95_s"],
          "reference_max_unambiguous_s": None if a is None else a["reference"]["max_unambiguous_s"],
          "locus_to_mesh_p95_s": sc1.get("locus_to_mesh_p95_s"), "coverage": sc1.get("coverage"),
          "abs_bias_toward_side2_s": None if g(dist, "bias_toward_side2_s") is None else abs(dist["bias_toward_side2_s"]),
          "abs_thickness_bias_s": None if g(dist, "thickness_bias_s") is None else abs(dist["thickness_bias_s"]),
          "matched_normal_p50_deg": None if a is None else a["normal_P2"]["mesh_p50_deg"],
          "matched_normal_p95_deg": None if a is None else a["normal_P2"]["mesh_p95_deg"],
          "unexplained_intersections_sampled": att["unexplained"],
          "faces_nearer_other_component_unexplained": fn["unexplained"],
          "volume_current_abs_rel_if_consistent": abs(rel) if cons["status"] == "CONSISTENT" else None,
          # report only (surface_metrics_v2.REPORT_ONLY_V2)
          "triangle_normal_p50_deg": sc1.get("normal_p50_deg"), "triangle_normal_p95_deg": sc1.get("normal_p95_deg"),
          "self_intersections_sampled": sc1.get("self_intersections_sampled"),
          "faces_nearer_other_component": sc1.get("faces_nearer_other_component"),
          "volume_current_abs_rel": sc1.get("volume_current_abs_rel"),
          "roughness_mesh_core_qm_s": sc1.get("roughness_mesh_core_qm_s"),
          "roughness_locus_core_qm_s": sc1.get("roughness_locus_core_qm_s"),
          "current_max_all_s": g(dist, "max_all_s"), "reference_max_all_s": None if a is None else a["reference"]["max_all_s"],
          "physical_intersections_sampled": att["physical"]}
    for key in ("rim_corner_p95_s", "rim_corner_max_s", "open_edges_near_junction", "boundary_edges", "non_manifold_edges",
                "inconsistently_oriented_edges", "degenerate_triangles", "shells_negative_volume", "closed",
                "shells_equal_components", "shells_spanning_components", "components_without_shell",
                "volume_reference_abs_rel", "fixed_state_topology_stable", "fixed_state_residual_s"):
        sc[key] = sc1.get(key)
    out["scalars"] = sc
    out["wall_s"] = time.perf_counter()-t0
    return out


def v2_per_cut_scalars(rep2, rep1, k):
    b = (rep2.get("blocks") or {}).get(f"cut{k}")
    sc1 = P3.per_cut_scalars(rep1, k)
    if b is None:
        return {"current_p95_s": None}
    d = b["distance"]
    return {"current_p50_s": d["p50_s"], "current_p95_s": d["p95_s"], "current_max_unambiguous_s": d["max_unambiguous_s"],
            "reference_p95_s": b["reference"]["p95_s"], "reference_max_unambiguous_s": b["reference"]["max_unambiguous_s"],
            "abs_bias_toward_side2_s": None if d["bias_toward_side2_s"] is None else abs(d["bias_toward_side2_s"]),
            "abs_thickness_bias_s": None if d["thickness_bias_s"] is None else abs(d["thickness_bias_s"]),
            "matched_normal_p95_deg": b["normal_P2"]["mesh_p95_deg"], "coverage": sc1["coverage"],
            "rim_corner_p95_s": sc1["rim_corner_p95_s"], "closed": sc1["closed"],
            "shells_equal_components": sc1["shells_equal_components"]}


# ----------------------------------------------------------------------------------------------- frame evaluation

def evaluate_full(run, fr, args, surfaces, kinds, *, legacy=False):
    """Final-frame evaluation: v1 report (as P3.evaluate_frame) and v2 report of every mesh."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    t_frame = time.perf_counter()
    st = dict(run, positions=np.asarray(fr["positions"], float), codes=fr["codes"], F=np.asarray(fr["F"], float),
              log=fr["log"])
    st.pop("_rank", None)
    count, labels, lat_count, fractions = P1.components(st)
    rank = st["_rank"]
    locus = SM.material_locus(st["reference"], st["lattice"], st["positions"], st["codes"], cuts=st["cuts"], F=st["F"],
                              events=st["log"])
    out = {"frame": fr["name"], "substep": int(fr["substep"]), "mechanical_components": int(count),
           "lattice_components": int(lat_count), "mass_fractions": fractions[:10].tolist(),
           "conflicting_link_rows": int(len(locus["cut"])), "locus_rows_by_source": locus["rows_by_source"],
           "events_in_log": int(len(st["log"])),
           "codes_sha256": hashlib.sha256(np.asarray(st["codes"]).tobytes()).hexdigest()}
    meshes = dict(P1.extract_surfaces(st, args.device, profiles=(A, LEGACY) if legacy else (A,)))
    for kind in kinds:
        m = extract(kind, surfaces, run, st)
        m["face_component"] = P1.face_components(m["info"], rank, len(m["faces"]))
        meshes[kind] = m
    ext = SM.exterior_samples(st["lattice"], st["positions"], S, st["F"])
    terms = v2_state_terms(st, labels, locus, ext)
    out["v2_state"] = {"P3_floor_residual": terms["floor_summary"], "P2_floor_p95_deg": terms["normal_floor_p95_deg"],
                       "P4_reference_consistency": terms["consistency"], "wall_s": terms["wall_s"]}
    reports1, reports2 = {}, {}
    for name, m in meshes.items():
        v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
        fc = m.get("face_component")
        rep = SM.surface_report(st["reference"], st["lattice"], st["positions"], st["codes"], v, f, spacing=S, labels=labels,
                                masses=st["masses"], density=st["density"], cuts=st["cuts"], F=st["F"], locus=locus,
                                face_component=fc, reference_space=False)
        is_b = name in B_ALL
        cls = SM.classify_mesh_vertices(v, st["positions"], labels, locus, ext, spacing=S)
        ref = P3.reference_space(st, labels, locus, cls, v, f, m["last"]["vertex_reference"] if is_b else None)
        rep["reference_space"] = ref
        if st["cuts"] > 1 and len(locus["cut"]):
            rep["rim_per_cut"] = P3.per_cut_rim(locus, st, v, f)
        vols = P3.provenance_volumes(v, f, fc, count) if fc is not None else None
        J = np.linalg.det(st["F"])
        rep["component_volumes_provenance"] = {
            "mesh_current_m3": None if vols is None else vols.tolist()[:64],
            "material_current_m3": np.bincount(labels, weights=st["masses"]/st["density"]*J, minlength=count).tolist()[:64],
            "material_reference_m3": np.bincount(labels, weights=st["masses"]/st["density"], minlength=count).tolist()[:64]}
        if is_b:
            from robotic_cutting_sim.cut_aware_surface import LINK_CUT
            X = m["last"]["vertex_reference"]
            vr = P3.provenance_volumes(X, f, fc, count)
            tr = X[f]
            area = .5*np.linalg.norm(np.cross(tr[:, 1]-tr[:, 0], tr[:, 2]-tr[:, 0]), axis=1)
            cut = m["last"]["face_kind"] == LINK_CUT
            a_cut = np.bincount(fc[cut], weights=area[cut], minlength=count) if cut.any() else np.zeros(count)
            rep["component_volumes_provenance"].update(mesh_reference_m3=vr.tolist()[:64], cut_area_reference_m2=a_cut.tolist()[:64])
            rep["cut_aware_info"] = compact_v2_info(m["info"])
        else:
            rep["unmeshable_skipped"] = m.get("skipped")
        rep["extract_wall_s"] = m["wall_s"]
        rep["volume_checks"] = P3.volume_checks(B1 if is_b else name, rep, st, labels, m, ref)
        if name == LEGACY:
            tmp = None
        elif is_b:
            tmp = fixed_state_temporal_b(name, m, st, run)
        else:
            tmp = P3.fixed_state_temporal(name, m, st, args, run, None)
        rep["temporal_fixed_state"] = tmp
        rep["scalars"] = P3.scalars(rep, ref, tmp)
        rep["scalars"]["volume_reference_abs_rel"] = abs(rep["volume_checks"]["reference_rel_error"])
        reports1[name] = rep
        if name != LEGACY:
            reports2[name] = v2_report(name, m, st, labels, rank, locus, ext, terms, rep)
    out["reports"] = reports1
    out["reports_v2"] = reports2
    out["frame_wall_s"] = time.perf_counter()-t_frame
    return out, meshes, labels, locus, st


def evaluate_light(run, fr, args, surfaces, kinds, *, intersections=True):
    """Light frame (P3.evaluate_frame light=True) for A and every B kind."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    st = dict(run, positions=np.asarray(fr["positions"], float), codes=fr["codes"], F=np.asarray(fr["F"], float),
              log=fr["log"])
    st.pop("_rank", None)
    count, labels, lat_count, _ = P1.components(st)
    rank = st["_rank"]
    meshes = dict(P1.extract_surfaces(st, args.device, profiles=(A,)))
    for kind in kinds:
        m = extract(kind, surfaces, run, st)
        m["face_component"] = P1.face_components(m["info"], rank, len(m["faces"]))
        meshes[kind] = m
    reports = {}
    for name, m in meshes.items():
        v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
        fc = m.get("face_component")
        topo = SM.topology_consistency(v, f, st["positions"], labels, face_component=fc)
        row = {"vertices": int(len(v)), "triangles": int(len(f)), "mesh_shells": topo["mesh_shells"],
               "shell_components": sorted(int(r["component"]) for r in topo["shell_rows"]),
               "shells_spanning_components": topo["shells_spanning_components"],
               "faces_nearer_other_component": topo["faces_nearer_other_component"], "extract_wall_s": m["wall_s"]}
        if intersections:
            sh = SM.shells(v, f)[1] if len(f) else None
            inter = SM.self_intersections(v, f, shell=sh)
            row["inter_shell_intersections_sampled"] = inter["inter_shell_pairs_sampled"]
            row["self_intersections_sampled"] = inter["intersecting_pairs_sampled"]
        if name in B_ALL:
            row["topology_hash"] = P3.topo_hash(f, m["last"]["vertex_reference"])
            row["mixed_side_regions"] = m["info"]["regions"]["mixed_side_regions"]
            row["cut_triangles"] = m["info"]["faces"]["cut_triangles"]
            row["audit_ok"] = m["info"]["audit"]["ok"]
            if "v2" in m["info"]:
                row["v2_counts"] = {"M4_fallback_particles": m["info"]["v2"]["M4_unlabelled_separation"]["fallback_particles"],
                                    "M3_flipped_before": m["info"]["v2"]["M3_fold_guard"].get("flipped_triangles_before"),
                                    "M2_moved": m["info"]["v2"]["M2_hinge"]["moved"]}
        reports[name] = row
    ev = {"frame": fr["name"], "substep": int(fr["substep"]), "mechanical_components": int(count),
          "lattice_components": int(lat_count), "reports": reports}
    vc = {name: __import__("robotic_cutting_sim.surface_metrics_v2", fromlist=["x"]).vertex_component_from_ranges(
        len(m["vertices"]), m["info"]["component_ranges"], rank) for name, m in meshes.items()}
    return ev, meshes, labels, st, vc


def recontact(run, frames, args, surfaces, kinds, v1_names):
    """S8 frames: light rows on every frame; protocol v1 slow motion (nearest transport) on the v1 frame subset
    ``v1_names`` (consecutive within the subset, codes equal); protocol v2 P5 on consecutive eligible pairs."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    from scipy.spatial import cKDTree
    out1, out2, prev1, prev2 = [], [], None, None
    eligible, pair_rows = SM2.slow_frame_pairs(frames, spacing=S)
    eligible = set(eligible)
    p5 = {name: [] for name in (A,)+tuple(kinds)}
    for i, fr in enumerate(frames):
        in_v1 = fr["name"] in v1_names
        ev, meshes, labels, st, vc = evaluate_light(run, fr, args, surfaces, kinds, intersections=in_v1)
        row = {"frame": fr["name"], "substep": int(fr["substep"]), "time_s": fr["substep"]*DT,
               "components": ev["mechanical_components"], "labelled": int(np.count_nonzero(fr["codes"])),
               "meshes": ev["reports"], "contact_stat_penetration": "NOT_AVAILABLE (not in the snapshots)"}
        if ev["mechanical_components"] == 2:
            d, _ = cKDTree(st["positions"][labels == 0]).query(st["positions"][labels == 1])
            row["min_particle_distance_between_pieces_m"] = float(d.min())
        cur = {n: (np.asarray(meshes[n]["vertices"], float), np.asarray(meshes[n]["faces"], np.int64)) for n in p5}
        if prev2 is not None and (i-1, i) in eligible:
            row["temporal_v2"] = {}
            for name in p5:
                pv, pf = prev2["meshes"][name]
                ts = SM2.temporal_stability_v2(pv, pf, cur[name][0], cur[name][1], prev2["positions"], st["positions"],
                                               prev2["vc"][name], labels, spacing=S)
                row["temporal_v2"][name] = ts
                p5[name].append(ts["residual_after_transport"].get("p95_spacings"))
        prev2 = {"positions": st["positions"], "meshes": cur, "vc": vc}
        out2.append(row)
        if in_v1:
            r1 = dict(row)
            r1.pop("temporal_v2", None)
            if prev1 is not None and np.array_equal(prev1["codes"], fr["codes"]):
                r1["temporal"] = {}
                for name in p5:
                    pv, pf = prev1["meshes"][name]
                    r1["temporal"][name] = SM.temporal_stability(pv, pf, cur[name][0], cur[name][1], prev1["positions"],
                                                                 st["positions"], spacing=S)
            prev1 = {"codes": fr["codes"], "positions": st["positions"], "meshes": cur}
            out1.append(r1)
        print(f"  [S8 frame {i+1}/{len(frames)} {fr['name']}] v1-frame {in_v1} eligible-pair {(i-1, i) in eligible}", flush=True)
    p5max = {n: (max(v) if v else None) for n, v in p5.items()}
    return out1, out2, {"pairs": pair_rows, "eligible_pairs": len(eligible), "max_p95_s": p5max}


# ----------------------------------------------------------------------------------------------- case evaluation

def v1_view(result, kind):
    """The result as P3 expects it: reports {A, B (= kind)}; recontact / hold frames likewise."""
    def rep(d):
        if d is None or kind not in d:
            return {k: v for k, v in (d or {}).items() if k == A}
        return {A: d[A], B1: d[kind]}
    view = {k: v for k, v in result.items() if k not in ("frames", "recontact_frames_v1", "hold_frames", "s4_provenance")}
    view["frames"] = [dict(fr, reports=rep(fr["reports"])) for fr in result["frames"]]
    if "recontact_frames_v1" in result:
        view["recontact_frames"] = []
        for f in result["recontact_frames_v1"]:
            g = dict(f, meshes=rep(f["meshes"]))
            if "temporal" in f:
                g["temporal"] = rep(f["temporal"])
            view["recontact_frames"].append(g)
    if "hold_frames" in result:
        view["hold_frames"] = [dict(f, reports=rep(f["reports"])) for f in result["hold_frames"]]
    if "s4_provenance" in result and kind in result["s4_provenance"]:
        view["s4_provenance"] = {B1: result["s4_provenance"][kind]}
    return view


def g4_hold_generic(frames, log, hold_frames, kind):
    """S9 G4 hold rows of P3.g4_hold from the stored frames (events in the hold window from the log, new digits from
    the codes of the two hold frames), with any B kind."""
    a, b = frames[0], frames[1]
    hold_events = int(np.count_nonzero((log["substep"] >= a["substep"]) & (log["substep"] < b["substep"])))
    new = int(np.count_nonzero(np.asarray(a["codes"]) != np.asarray(b["codes"])))
    b0, b1 = hold_frames[0]["reports"].get(kind), hold_frames[1]["reports"].get(kind)
    if b0 is None or b1 is None:
        return {"checks": {k: not_run("hold frames not evaluated for this variant") for k in
                           ("events", "new_digits", "cut_faces_unchanged", "topology_hash_unchanged")}, "verdict": "NOT_RUN"}
    rows = {"events": check(hold_events, "==", 0), "new_digits": check(new, "==", 0),
            "cut_faces_unchanged": check(b0["cut_triangles"] == b1["cut_triangles"], "==", True),
            "topology_hash_unchanged": check(b0["topology_hash"] == b1["topology_hash"], "==", True)}
    return {"checks": rows, "verdict": P3.verdict_of(rows), "cut_triangles": [b0["cut_triangles"], b1["cut_triangles"]]}


def v2_case_checks(case, result, T2):
    """surface_protocol_v2 threshold rows per method (A, B v1, B v2, B v2 pc)."""
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    fin = result["frames"][-1]
    out = {}
    t1 = T2["S1_flat_cut"]
    cur_max = T2["S2_deformed_cut"]["current_max_s"] if case == "S2" else t1["current_max_s"]

    def s1_rows(b, sc, max_limit):
        if b is None:
            return {"cut_face": {"value": None, "verdict": "NOT_EVALUABLE", "note": "no cut-face vertex"}}
        d, p3, nr, rr = b["distance"], b["current_p95_P3"], b["normal_P2"], b["reference"]
        return {"reference_p95_s": check(rr["p95_s"], "<=", t1["reference_p95_s"]),
                "reference_max_unambiguous_s": check(rr["max_unambiguous_s"], "<=", t1["reference_max_s"]),
                "current_p95_P3": check_bool(p3["all_groups_pass"], value=p3["worst_excess_s"],
                                             note=f"{p3['groups_evaluated']} side groups; value = worst mesh P95 - (floor P95 + 0.10 s)"),
                "current_max_unambiguous_s": check(d["max_unambiguous_s"], "<=", max_limit),
                "abs_bias_toward_side2_s": check(None if d["bias_toward_side2_s"] is None else abs(d["bias_toward_side2_s"]),
                                                 "<=", t1["abs_bias_toward_side2_s"]),
                "abs_thickness_bias_s": check(None if d["thickness_bias_s"] is None else abs(d["thickness_bias_s"]),
                                              "<=", t1["abs_thickness_bias_s"]),
                "normal_P2_matched": check_bool(nr["pass"], value=nr["mesh_p95_deg"], threshold=nr["limit_deg"]),
                "coverage_min": check(sc["coverage"], ">=", t1["coverage_min"]),
                "rim_corner_p95_s": check(sc["rim_corner_p95_s"], "<=", t1["rim_corner_p95_s"]),
                "closed": check(sc["closed"], "==", True), "shells_equal_components": check(sc["shells_equal_components"], "==", True)}

    for kind, r2 in fin.get("reports_v2", {}).items():
        r1 = fin["reports"][kind]
        sc = r2["scalars"]
        blocks = r2.get("blocks") or {}
        rows, strict = {}, None
        if case in ("S1_G3a", "S1_G3b", "S2", "S3"):
            full = s1_rows(blocks.get("all"), sc, cur_max)
            if case == "S2":
                rows = {k: v for k, v in full.items() if not k.startswith("reference")}
                strict = full
            elif case == "S3":
                rows = {k: v for k, v in full.items() if k.startswith("reference")}
                strict = full
            else:
                rows = full
        view = v1_view(result, kind) if kind in B_ALL else v1_view(result, A)
        mk = B1 if kind in B_ALL else A
        if case == "S4":
            rows = P3.s4_rows(mk, view, view["frames"][-1]["reports"][mk], T2)
            b = blocks.get("all")
            rows["report_current_p95_P3"] = {"value": None if b is None else b["current_p95_P3"]["worst_excess_s"],
                                             "verdict": "REPORT_ONLY"}
            rows["report_normal_P2"] = {"value": None if b is None else b["normal_P2"]["mesh_p95_deg"], "verdict": "REPORT_ONLY"}
            rows["report_P6_slit_pairs"] = {"value": r2["intersections_P6"], "verdict": "REPORT_ONLY"}
        elif case == "S5":
            for k in sorted(k for k in blocks if k.startswith("cut")):
                pc = P3.per_cut_scalars(r1, k[3:])
                for name, c in s1_rows(blocks[k], pc, t1["current_max_s"]).items():
                    rows[f"{k}_{name}"] = c
        elif case == "S6":
            if kind == A:
                view["_b_cut_area_per_component"] = result.get("_b_cut_area_per_component")
            rows = P3.s6_rows(mk, view, view["frames"][-1]["reports"][mk], T2)
        elif case == "S7":
            base = P3.s7_rows(mk, view, view["frames"][-1]["reports"][mk], T2)
            rows = {"skipped_fragments": base["skipped_fragments"], "non_manifold_edges_report": base["non_manifold_edges_report"]}
            rows["chip_current_volume_rel_error_P7"] = result.get("s7_p7", {}).get(kind) or not_run("chip row not evaluated")
        elif case == "S8":
            if kind in REPORT_ONLY_KINDS:
                rows = {k: not_run("recontact frames not evaluated for the labelled report-only variants") for k in
                        ("vertices_shared_across_conflicting_codes", "shell_set_unchanged_since_separation",
                         "slow_motion_P5")}
            else:
                base = P3.s8_rows(mk, {"recontact_frames": result_v2_recontact(result, kind)}, T2)
                rows = {k: base[k] for k in ("vertices_shared_across_conflicting_codes", "shell_set_unchanged_since_separation")}
                p5 = result.get("p5", {})
                val = (p5.get("max_p95_s") or {}).get(kind)
                rows["slow_motion_P5"] = (not_run("no eligible frame pair (premise unmet)") if not p5.get("eligible_pairs")
                                          else check(val, "<=", T2["all"]["temporal_slow_motion_residual_p95_s"],
                                                     note=f"{p5.get('eligible_pairs')} eligible pairs"))
        all_rows = P3.all_checks(r1, T2)
        all_rows["volume_current_rel"] = r2["volume_current_P4"]
        rows.update({f"all_{k}": v for k, v in all_rows.items()})
        out[kind] = {"checks": rows, "verdict": verdict_v2(rows)}
        if strict is not None:
            st = {**strict, **{f"all_{k}": v for k, v in all_rows.items()}}
            out[kind]["strict_all_S1_rows"] = {"checks": st, "verdict": verdict_v2(st)}
    return out


def result_v2_recontact(result, kind):
    """Dense recontact frames (v2) in the P3 recontact layout for P3.s8_rows' unchanged rows."""
    return [{"meshes": {B1 if kind in B_ALL else A: f["meshes"][kind]}} for f in result.get("recontact_frames_v2", [])
            if kind in f["meshes"]]


def v1_case_checks(case, result, T1):
    """Protocol v1 rows per method via P3.case_checks on per-variant views."""
    out = {}
    for kind in B_ALL:
        if kind not in result["frames"][-1]["reports"]:
            continue
        view = v1_view(result, kind)
        if case == "S6":
            view["_b_cut_area_per_component"] = result.get("_b_cut_area_per_component")
        if case == "S8" and kind in REPORT_ONLY_KINDS:
            view["recontact_frames"] = []
        c = P3.case_checks(case, view, T1)
        if A not in out:
            out[A] = c[A]
        out[kind] = c[B1]
        if case == "S8" and kind in REPORT_ONLY_KINDS:
            reason = "recontact frames not evaluated for the labelled report-only variants"
            for k in ("vertices_shared_across_conflicting_codes", "shell_set_unchanged_since_separation",
                      "all_temporal_slow_motion_residual_p95_s"):
                out[kind]["checks"][k] = not_run(reason)
            out[kind]["verdict"] = P3.verdict_of(out[kind]["checks"])
    return out


def s7_p7(result, meshes, labels, st):
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    fin = result["frames"][-1]
    count = fin["mechanical_components"]
    chip = list(range(1, count))
    cons = fin["v2_state"]["P4_reference_consistency"]
    status = SM2.component_status(cons, chip) if chip else "UNVERIFIED"
    out = {}
    for kind, m in meshes.items():
        if kind == LEGACY:
            continue
        if not chip:
            out[kind] = check(None, "<=", .05, note="no chip detached")
            continue
        last = m.get("last") or {}
        out[kind] = SM2.chip_volume_v2(m["vertices"], m["faces"], m["face_component"], labels, st["masses"]/st["density"],
                                       st["F"], chip, limit=.05, status=status,
                                       fallback_faces=last.get("face_particle_cell"),
                                       fallback_particles=last.get("particle_cell_fallback"))
    return out


def evaluate_case(case, r, run, frames, args, T1, T2, outdir, tag):
    t0 = time.perf_counter()
    result = {"case": case, "repeat": r, "set": tag, "particles": int(len(frames[-1]["codes"])),
              "frames_available": [f["name"] for f in frames], "runner": run.get("runner")}
    surfaces = {}
    kinds_light = B_GATED
    if case == "S8":
        v1_names = run.get("v1_frame_names") or [f["name"] for f in frames[:-1]]
        result["recontact_frames_v1"], result["recontact_frames_v2"], result["p5"] = recontact(
            run, frames[:-1], args, surfaces, kinds_light, set(v1_names))
    if case == "S6":
        result["slice_frames"] = [evaluate_light(run, fr, args, surfaces, kinds_light)[0] for fr in frames[:-1]]
    if case == "S4":
        result["hold_frames"] = [evaluate_light(run, fr, args, surfaces, kinds_light)[0] for fr in frames[:-1]]
    ev, meshes, labels, locus, st = evaluate_full(run, frames[-1], args, surfaces, B_ALL, legacy=False)
    result["frames"] = [ev]
    if case == "S4":
        result["s4_provenance"] = {k: P3.s4_provenance(meshes[k], locus, st["codes"], st["cuts"]) for k in B_ALL}
        result["g4_hold_phase"] = {k: g4_hold_generic(frames[:2], run["log"], result["hold_frames"], k) for k in B_ALL}
    if case == "S6":
        result["_b_cut_area_per_component"] = ev["reports"][B1]["component_volumes_provenance"]["cut_area_reference_m2"]
    if case == "S7":
        result["s7_p7"] = s7_p7(result, meshes, labels, st)
    if r == 0:
        save_package(outdir/f"{case}_r{r}_package.npz", run, labels, meshes, locus, st)
    result["threshold_checks_v1"] = v1_case_checks(case, result, T1)
    result["threshold_checks_v2"] = v2_case_checks(case, result, T2)
    result["analysis_wall_s"] = time.perf_counter()-t0
    return result


def save_package(path, run, labels, meshes, locus, st):
    if Path(path).exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    n = len(st["codes"])
    pkg = dict(run, positions=st["positions"], codes=st["codes"], F=st["F"],
               contact_work=run.get("contact_work", np.zeros(n)), cut_progress=run.get("cut_progress", np.zeros(n)),
               blade_polygon=run.get("blade_polygon", np.zeros((0, 2))), half_length=run.get("half_length", 0.),
               trail=run.get("trail"), assign=run.get("assign"))
    P1.save_package(path, pkg, labels, {k: {"vertices": m["vertices"], "faces": m["faces"]} for k, m in meshes.items()}, locus)


# ----------------------------------------------------------------------------------------------- S0, G2, S9

def stage_static(args):
    from robotic_cutting_sim import surface_attribution_v1 as SA
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import block
    out_dir = args.out/"eval_static"
    out_dir.mkdir(exist_ok=True)
    T = P3.protocol_thresholds()
    # S0
    points, masses = block(size_m=(.04, .024, .012), spacing=S)
    lo, hi = points.min(0)-S/2, points.max(0)+S/2
    nodes, weights, gradients = SA.stencil(points, DX)
    n = len(points)
    run = {"reference": points.astype(float), "lattice": np.round((points-points.min(0))/S).astype(int),
           "masses": masses.astype(float), "density": 1500., "positions": points.astype(float),
           "codes": np.zeros(n, np.int64), "F": np.tile(np.eye(3), (n, 1, 1)), "nodes": nodes, "weights": weights,
           "gradients": gradients, "cuts": 1, "log": None}
    meshes = dict(P1.extract_surfaces(run, args.device, profiles=(A,)))
    surfaces = {}
    for kind in B_ALL:
        meshes[kind] = extract(kind, surfaces, run, run)
    s0 = {"case": "S0", "methods": {}, "threshold_checks": {}}
    t = T["S0_uncut_box"]
    for name, m in meshes.items():
        v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
        val, box = SM.mesh_validity_v1(v, f), P1.box_metrics(v, f, lo, hi, S)
        row = {"validity": val, "box": box, "extract_wall_s": m["wall_s"]}
        if name in B_ALL:
            row["cut_aware_info"] = compact_v2_info(m["info"])
            row["temporal_fixed_state"] = fixed_state_temporal_b(name, m, run, run)
            row["equals_B_v1_bitwise"] = bool(np.array_equal(m["faces"], meshes[B1]["faces"])
                                              and np.array_equal(m["vertices"], meshes[B1]["vertices"]))
        checks = {"whole_surface_p95_s": check(box["area_weighted_p95_spacings"], "<=", t["whole_surface_p95_s"]),
                  "whole_surface_max_s": check(box["max_m"]/S, "<=", t["whole_surface_max_s"]),
                  "volume_rel_error": check(abs(box["volume_relative_error"]), "<=", t["volume_rel_error"]),
                  "closed_oriented_manifold": check(val["boundary_edges"] == 0 and val["non_manifold_edges"] == 0
                                                    and val["inconsistently_oriented_edges"] == 0
                                                    and val["shells_negative_volume"] == 0, "==", t["closed_oriented_manifold"]),
                  "shells": check(val["shells"], "==", t["shells"]),
                  "euler": check(int(val["shell_rows"][0]["euler_characteristic"]) if val["shell_rows"] else None, "==", t["euler"])}
        s0["methods"][name] = row
        s0["threshold_checks"][name] = {"checks": checks, "verdict": P3.verdict_of(checks)}
    s0["s0_mesh"] = {k: {"faces_sha256": hashlib.sha256(np.ascontiguousarray(meshes[k]["faces"], np.int64).tobytes()).hexdigest()}
                     for k in B_ALL}
    write_new(out_dir/"S0.json", s0)
    np.savez_compressed(out_dir/"S0_B_meshes.npz", **{f"{k}_faces": meshes[k]["faces"] for k in B_ALL},
                        **{f"{k}_vertex_reference": meshes[k]["last"]["vertex_reference"] for k in B_ALL})
    print("[S0]", {k: v["verdict"] for k, v in s0["threshold_checks"].items()}, flush=True)
    # G2: v1 planes + held-out planes (analytic; the only place an analytic plane is a reference)
    points, masses = block(size_m=(.04, .024, .016), spacing=S)
    reference = points.astype(float)
    lattice = np.round((points-points.min(0))/S).astype(int)
    nodes, weights, gradients = SA.stencil(points, DX)
    centre = reference.mean(0)+np.array([.0003, .0002, .0001])
    normals = {"n30_xz": [np.cos(np.radians(30)), 0, np.sin(np.radians(30))], "n45_xy": [np.cos(np.radians(45)), np.sin(np.radians(45)), 0],
               "n123": list(np.array([1., 2., 3.])/np.sqrt(14)), "axis_x": [1., 0, 0],
               "n213_heldout": list(np.array([2., -1., 3.])/np.sqrt(14)),
               "n60_xz_heldout": [np.cos(np.radians(60)), 0, np.sin(np.radians(60))]}
    gap = .003
    tg = T["G2_static_unit_test"]
    tol = tg["numeric_tolerance_m"]
    g2 = {"case": "G2", "gap_m": gap, "plane_centre_m": centre.tolist(), "planes": {}, "threshold_checks": {}}
    for name, nrm in normals.items():
        nrm = np.asarray(nrm, float)
        d = (reference-centre)@nrm
        codes = np.where(d >= 0, 2, 1).astype(np.int64)
        log = P1.synthetic_events(reference, lattice, codes, centre, nrm)
        positions = reference+np.where(d[:, None] >= 0, gap*nrm, 0.)
        F = np.tile(np.eye(3), (len(points), 1, 1))
        run = {"reference": reference, "lattice": lattice, "positions": positions, "codes": codes, "F": F,
               "masses": masses.astype(float), "density": 1500., "nodes": nodes, "weights": weights,
               "gradients": gradients, "cuts": 1, "log": log}
        _, labels, _, _ = P1.components(run)
        locus = SM.material_locus(reference, lattice, positions, codes, cuts=1, F=F, events=log)
        ext = SM.exterior_samples(lattice, positions, S, F)
        meshes = {A: P1.extract_surfaces(run, args.device, profiles=(A,))[A]}
        surfaces = {}
        for kind in (B1, B2, B2PC, B1MID, B2MID):
            meshes[kind] = extract(kind, surfaces, run, run)
        rows = {}
        for method, m in meshes.items():
            v, f = np.asarray(m["vertices"], float), np.asarray(m["faces"], np.int64)
            cls = SM.classify_mesh_vertices(v, positions, labels, locus, ext, spacing=S)
            cut = cls["kind"] == 2
            sd = (v-centre)@nrm-np.where(cls["side"] == 1, gap, 0.)
            nt, _, _ = SM.triangle_normals(v, f)
            core_tri = cls["core"][f].all(1)
            ang = np.degrees(np.arccos(np.clip(np.abs(nt[core_tri]@nrm), 0, 1)))
            val = SM.mesh_validity_v1(v, f)
            row = {"cut_face_vs_analytic": P3.q(np.abs(sd[cut])), "core_vs_analytic": P3.q(np.abs(sd[cls["core"]])),
                   "non_crease_normal_deg": P3.q(ang), "non_manifold_edges": val["non_manifold_edges"],
                   "boundary_edges": val["boundary_edges"], "shells": val["shells"], "region": cls["counts"]}
            if method != A:
                X = m["last"]["vertex_reference"]
                row["reference_cut_face_vs_analytic"] = P3.q(np.abs((X[cut]-centre)@nrm))
                row["qef"] = m["info"]["qef"]
                clamped = m["last"]["vertex_clamped"]
                row["clamped_cut_face_vertices"] = int((clamped & cut).sum())
                row["cut_face_max_excluding_clamped"] = P3.q(np.abs(sd[cut & ~clamped]))
                row["clamped_vs_analytic"] = P3.q(np.abs(sd[cut & clamped]))
                if "v2" in m["info"]:
                    row["v2"] = {k: m["info"]["v2"][k] for k in ("M1_constrained_qef", "M2_hinge", "M3_fold_guard")}
            rows[method] = row
        g2["planes"][name] = rows
        for method, key in ((B1, "hermite"), (B2, "hermite"), (B2PC, "hermite"), (B1MID, "midpoint_fallback"),
                            (B2MID, "midpoint_fallback"), (A, "hermite")):
            tt = tg[key]
            r_ = rows[method]
            c = {"cut_face_p95_s": check(r_["cut_face_vs_analytic"]["p95_spacings"], "<=", tt["cut_face_p95_s"]+tol/S),
                 "cut_face_max_s": check(r_["cut_face_vs_analytic"]["max_spacings"], "<=", tt["cut_face_max_s"]+tol/S)}
            if key == "hermite":
                c["non_crease_normal_p95_deg"] = check(r_["non_crease_normal_deg"]["p95"], "<=", tt["non_crease_normal_p95_deg"])
                c["non_manifold_edges"] = check(r_["non_manifold_edges"], "==", tt["non_manifold_edges"])
            g2["threshold_checks"].setdefault(method, {})[name] = {"checks": c, "verdict": P3.verdict_of(c), "thresholds_set": key}
    for method, per in g2["threshold_checks"].items():
        v1planes = {k: v for k, v in per.items() if isinstance(v, dict) and not k.endswith("_heldout")}
        allp = {k: v for k, v in per.items() if isinstance(v, dict)}
        per["verdict_v1_planes"] = P3.verdict_of({k: {"verdict": v["verdict"]} for k, v in v1planes.items()})
        per["verdict_with_heldout_planes"] = P3.verdict_of({k: {"verdict": v["verdict"]} for k, v in allp.items()})
    write_new(out_dir/"G2.json", g2)
    print("[G2]", {m: (v["verdict_v1_planes"], v["verdict_with_heldout_planes"]) for m, v in g2["threshold_checks"].items()}, flush=True)


S9_BASE = {"approach_armed": (0., 0.), "flat_press_armed": (0., 0.), "side_push_armed": (.025, 0.),
           "slide_disarmed": (-.030, 0.), "plunge_disarmed": (0., 0.)}


def s9_control(args, name, u):
    """P3.case_s9_control with the declared offsets u (None = the exact v1 positions)."""
    from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block
    if u is None:
        return P3.case_s9_control(args, name)
    top = .012
    points, masses = block(size_m=(.04, .024, top), spacing=S)
    bx, by = S9_BASE[name]
    x, y = bx+u[0]*S, by+u[1]*S
    controls = {
        "approach_armed": (VERTICAL_CUT, [x, y, top+.010], [Phase((0, 0, -CUT_SPEED), .45, 0), Phase((0, 0, 0), .2, 0)]),
        "flat_press_armed": (FLAT_SKIM, [x, y, top+.002], [Phase((0, 0, -.01), .4, 0), Phase((0, 0, 0), .2, 0),
                                                           Phase((0, 0, .01), .2, 0)]),
        "side_push_armed": (VERTICAL_CUT, [x, y, .006], [Phase((-.01, 0, 0), .7, 0), Phase((0, 0, 0), .1, 0)]),
        "slide_disarmed": (FLAT_SKIM, [x, y, top-.0005], [Phase((CUT_SPEED, 0, 0), 1.25, -1)]),
        "plunge_disarmed": (VERTICAL_CUT, [x, y, top+.004], [Phase((0, 0, -CUT_SPEED), .5, -1), Phase((0, 0, MOVE_SPEED), .4, -1)]),
    }
    rot, start, phases = controls[name]
    return P3.probe(points, masses, phases, rotation=rot, start=start, device=args.device)


def run_s9(args, tag, r, s0_faces):
    """S9 controls: 0 events, 0 B cut faces, B mesh = uncut extraction = S0 mesh, for A / B v1 / B v2 / B v2 pc."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    u = None if tag == "saved" else offsets(SEEDS["S9"][0 if tag == "H1" else 1])
    out = {"case": "S9", "set": tag, "repeat": r, "offsets_s": None if u is None else u.tolist(), "controls": {}}
    T = P3.protocol_thresholds()["S9_negative_controls"]
    for name in P3.S9_CONTROLS:
        run = s9_control(args, name, u)
        kinds = B_S9
        count, labels, lat, _ = P1.components(run)
        a = P1.extract_surfaces(dict(run), args.device, profiles=(A,))[A]
        av = SM.mesh_validity_v1(a["vertices"], a["faces"])
        row = {"events": int(len(run["log"])), "labelled": int(np.count_nonzero(run["codes"])),
               "unexpected_code_changes": int(run["metadata"]["unexpected_code_changes"]), "mechanical_components": int(count),
               "max_displacement_m": float(np.linalg.norm(run["positions"]-run["reference"], axis=1).max()),
               "physics_wall_s": run["wall_s"], A: {"shells": av["shells"], "watertight": av["watertight"]}}
        for kind in kinds:
            b = P3.b_extract(make_b(kind, run, run["log"]), run, run["log"])
            uncut = P3.b_extract(make_b(kind, run, None), dict(run, codes=np.zeros_like(run["codes"])), None)
            same_ref = (np.array_equal(b["faces"], uncut["faces"])
                        and np.array_equal(b["last"]["vertex_reference"], uncut["last"]["vertex_reference"]))
            same_cur = same_ref and np.array_equal(b["vertices"], uncut["vertices"])
            same_s0 = bool(s0_faces is not None and kind in s0_faces and np.array_equal(b["faces"], s0_faces[kind][0])
                           and np.array_equal(b["last"]["vertex_reference"], s0_faces[kind][1]))
            bv = SM.mesh_validity_v1(b["vertices"], b["faces"])
            row[kind] = {"cut_triangles": b["info"]["faces"]["cut_triangles"], "shells": bv["shells"],
                         "watertight": bv["watertight"], "equals_uncut_extraction": bool(same_cur),
                         "faces_and_reference_equal_S0_box_mesh": same_s0, "triangles": int(len(b["faces"])),
                         "wall_s": b["wall_s"]}
            if "v2" in b["info"]:
                row[kind]["v2"] = {k: b["info"]["v2"][k] for k in ("M1_constrained_qef", "M2_hinge", "M3_fold_guard",
                                                                    "M4_unlabelled_separation")}
        checks = {}
        for kind in kinds:
            c = {"events": check(row["events"], "==", T["events"]),
                 "cut_faces": check(row[kind]["cut_triangles"], "==", T["cut_faces"]),
                 "mesh_equals_S0_mesh_embedded": check(row[kind]["equals_uncut_extraction"], "==", True),
                 "hermite_points_outside_swept_region": check(0 if row["events"] == 0 else None, "==",
                                                              T["hermite_points_outside_swept_region"])}
            checks[kind] = {"checks": c, "verdict": P3.verdict_of(c)}
        ca = {"events": check(row["events"], "==", T["events"]),
              "one_closed_shell": check(row[A]["shells"] == 1 and row[A]["watertight"], "==", True)}
        checks[A] = {"checks": ca, "verdict": P3.verdict_of(ca)}
        row["threshold_checks"] = checks
        out["controls"][name] = row
        print(f"[S9 {tag} r{r} {name}] events {row['events']} | " +
              " | ".join(f"{k} {checks[k]['verdict']}" for k in (A,)+kinds), flush=True)
    return out


# ----------------------------------------------------------------------------------------------- held-out physics

def probe_chunked(points, masses, phases, *, rotation, start, device, snapshot, chunk):
    """P1.probe_run (drive_blade, law 8, tlmpm_v5, recorder on) with a smaller ``chunk``: the recorder hook (and so a
    snapshot) can fire only at chunk ends, so the dense held-out S8 frames (every 40 substeps) need chunk = 40. The
    chunk is the graph-launch granularity of the same per-substep trajectory, not a physics parameter."""
    from robotic_cutting_sim import cut_event_snapshots as cs
    from robotic_cutting_sim.blade_probe import drive_blade
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
    trail = cs.PoseTrail(stride=10, want_snapshot=snapshot, snapshot_F=True)
    t0 = time.perf_counter()
    with cs.installed(hook=trail) as made:
        r = drive_blade(RULER_KNIFE_V1, rotation, start, phases, points=points, masses=masses, spacing=S,
                        cutting_energy=P1.ENERGY, cuts=1, cutting_law=8, solver_version=5, device=device, chunk=int(chunk),
                        gpu_graph=device.startswith("cuda"), record_history=True, cut_events="cut_events_v1")
    wall = time.perf_counter()-t0
    rec = made[0]
    trail.poses, trail.cuts = P1.host_pose_trail(start, phases, rotation, stride=10)
    log = r["cut_events"]["log"]
    st = r["state"]
    return {"reference": np.asarray(points, float), "lattice": np.round((points-points.min(0))/S).astype(int),
            "masses": np.asarray(masses, float), "density": 1500., "positions": st["positions"].astype(float),
            "codes": st["codes"], "F": st["F"].astype(float), "contact_work": st["contact_work_j"],
            "cut_progress": st["cut_progress"], "nodes": r["reference_stencil_nodes"], "weights": r["reference_stencil_weights"],
            "gradients": r["reference_stencil_gradients"], "cuts": 1, "log": log, "activation": r["cut_events"]["activation"],
            "metadata": r["cut_events"]["metadata"], "assign": rec.assignment_positions(log), "trail": trail,
            "half_length": float(RULER_KNIFE_V1.half_width_m), "blade_polygon": RULER_KNIFE_V1.tlmpm_profile()[0],
            "phase_substeps": [int(round(p.duration_s/DT)) for p in phases], "labelled_after_phase": r["labelled_after_phase"],
            "digit_after_phase": r["digit_labelled_after_phase"], "wall_s": wall,
            "runner": f"blade_probe.drive_blade (chunk {int(chunk)})", "slot_diagnostics": r["slot_diagnostics"]}


def heldout_case(case, variant, args):
    """Protocol v2 held_out_states runner for one case and variant (H1 / H2). Returns (run, info)."""
    from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block
    from robotic_cutting_sim.sequence_blade_probe import OrientedPhase as OP
    seed = SEEDS[case][0 if variant == "H1" else 1]
    if case == "S1_G3a":
        run = P3.cut01r_run(args.device, seed=seed)
        return run, {"seed": seed, "geometry": f"registered CUT-01-R, seed {seed}", "declared_plane": ((0, 0, 0), (1, 0, 0))}
    u = offsets(seed)
    ux, uy = u*S
    info = {"seed": seed, "offset_s": u.tolist()}
    top = .012
    depth = top+.008
    if case == "S1_G3b":
        yaw = 20. if variant == "H1" else 35.
        points, masses = block(size_m=(.08, .03, top), spacing=S)
        rot = Rotation.from_euler("z", yaw, degrees=True)*VERTICAL_CUT
        phases = [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1)]
        run = P3.probe(points, masses, phases, rotation=rot, start=[ux, uy, top+.004], device=args.device)
        run["phase_names"] = ["cut", "lift"]
        return run, {**info, "yaw_deg": yaw}
    if case == "S2":
        press = .002 if variant == "H1" else .0015
        x_cut = (-.010 if variant == "H1" else -.005)+ux
        points, masses = block(size_m=(.06, .024, top), spacing=S)
        flat, vert = tuple(FLAT_SKIM.as_quat()), tuple(VERTICAL_CUT.as_quat())
        x_edge, half_thickness = .012, .00075
        z0, z_press, z_clear = top+.002, top-press+half_thickness, top+.004
        phases = [OP((0, 0, -.01), (z0-z_press)/.01, -1, flat), OP((0, 0, 0), .2, -1, flat),
                  OP((0, 0, .025), (z_clear-z_press)/.025, -1, flat), OP(((x_cut-x_edge)/.5, 0, 0), .5, -1, flat),
                  OP((0, 0, 0), .05, -1, vert), OP((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0, vert),
                  OP((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1, vert)]
        run = P3.probe(points, masses, phases, start=[x_edge, uy, z0], device=args.device, sequence=True)
        run["phase_names"] = ["press", "hold", "lift_press", "move", "reorient", "cut", "lift"]
        return run, {**info, "press_depth_m": press, "cut_x_m": x_cut, "declared_plane": ((x_cut, 0, 0), (1, 0, 0))}
    if case == "S3":
        rate_deg = 25. if variant == "H1" else -30.
        points, masses = block(size_m=(.06, .024, top), spacing=S)
        rate = np.radians(rate_deg)
        r0 = Rotation.from_euler("z", -rate_deg*depth/CUT_SPEED/2, degrees=True)*VERTICAL_CUT
        cut = OP((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0, tuple(r0.as_quat()), (0., 0., float(rate)))
        phases = [cut, OP((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1, cut.end_rotation_xyzw(DT))]
        run = P3.probe(points, masses, phases, start=[ux, uy, top+.004], device=args.device, sequence=True)
        run["phase_names"] = ["cut_yawing", "lift"]
        return run, {**info, "yaw_rate_deg_s": rate_deg, "yaw_inside_block_deg": float(np.degrees(abs(rate)*top/CUT_SPEED))}
    if case == "S4":
        cut_depth = .006 if variant == "H1" else .004
        yaw = 0. if variant == "H1" else 10.
        points, masses = block(size_m=(.06, .024, top), spacing=S)
        rot = Rotation.from_euler("z", yaw, degrees=True)*VERTICAL_CUT
        travel = .004+cut_depth
        phases = [Phase((0, 0, -CUT_SPEED), travel/CUT_SPEED, 0), Phase((0, 0, 0), .2, 0), Phase((0, 0, MOVE_SPEED), .4, -1)]
        steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
        want = lambda rec: int(rec.substep.numpy()[0]) in (int(steps[0]), int(steps[1]))  # noqa: E731
        run = P3.probe(points, masses, phases, rotation=rot, start=[ux, uy, top+.004], device=args.device, snapshot=want)
        run["phase_names"] = ["cut_partial", "hold_armed", "withdraw"]
        run["frame_names"] = ["cut_end", "hold_end"]
        return run, {**info, "edge_depth_below_top_m": cut_depth, "yaw_deg": yaw}
    if case == "S5":
        points, masses = block(size_m=(.06, .04, top), spacing=S)
        r0 = tuple(VERTICAL_CUT.as_quat())
        if variant == "H1":
            x0, pitch, cross_yaw = -.012+ux, .024, 90.
        else:
            x0, pitch, cross_yaw = -.010+ux, .020, 75.
        r1 = tuple((Rotation.from_euler("z", cross_yaw, degrees=True)*VERTICAL_CUT).as_quat())
        down, up = depth/CUT_SPEED, depth/MOVE_SPEED
        y2 = .004+uy
        phases = [OP((0, 0, -CUT_SPEED), down, 0, r0), OP((0, 0, MOVE_SPEED), up, -1, r0), OP((pitch/.5, 0, 0), .5, -1, r0),
                  OP((0, 0, -CUT_SPEED), down, 1, r0), OP((0, 0, MOVE_SPEED), up, -1, r0),
                  OP((-pitch/2/.5, y2/.5, 0), .5, -1, r0), OP((0, 0, 0), .05, -1, r1),
                  OP((0, 0, -CUT_SPEED), down, 2, r1), OP((0, 0, MOVE_SPEED), up, -1, r1)]
        run = P3.probe(points, masses, phases, start=[x0, 0, top+.004], cuts=3, device=args.device, sequence=True,
                       slots_per_node=16)
        run["phase_names"] = ["cut0", "lift0", "move0", "cut1", "lift1", "move1", "reorient", "cut2", "lift2"]
        return run, {**info, "slots_per_node": 16, "parallel_x_m": [x0, x0+pitch], "cross_yaw_deg": cross_yaw,
                     "cross_point_m": [x0+pitch/2, y2]}
    if case == "S6":
        base = (-.036, -.012, .012, .036) if variant == "H1" else (-.030, -.010, .010, .030)
        xs = tuple(b+ux for b in base)
        points, masses = block(size_m=(.12, .024, top), spacing=S)
        phases, names = [], []
        for k, x in enumerate(xs):
            phases += [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, k), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1)]
            names += [f"cut_{k}", f"lift_{k}"]
            if k+1 < len(xs):
                phases.append(Phase(((xs[k+1]-x)/.5, 0, 0), .5, -1))
                names.append(f"move_{k}")
        steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
        lift_end = {int(steps[i]) for i, nm in enumerate(names) if nm.startswith("lift_")}
        want = lambda rec: int(rec.substep.numpy()[0]) in lift_end  # noqa: E731
        run = P3.probe(points, masses, phases, rotation=VERTICAL_CUT, start=[xs[0], uy, top+.004], cuts=4,
                       device=args.device, snapshot=want)
        run["phase_names"] = names
        run["frame_names"] = [f"after_cut_{k}" for k in range(len(xs))]
        return run, {**info, "slice_x_m": list(xs)}
    if case == "S7":
        if variant == "H1":
            top7, dome_h, dome_r, edge_above, rake = .010, .004, .009, .002, 25.
        else:
            top7, dome_h, dome_r, edge_above, rake = .010, .005, .009, .0025, 35.
        grid = np.mgrid[-.04:.04-1e-9:S, -.012:.012-1e-9:S, 0:top7+dome_h-1e-9:S]
        pts = grid.reshape(3, -1).T+S/2
        dome = (pts[:, 2] > top7) & ((pts[:, 2]-top7) <= dome_h*(1-(pts[:, 0]**2+pts[:, 1]**2)/dome_r**2))
        points = pts[(pts[:, 2] < top7) | dome].astype(np.float32)
        masses = np.full(len(points), 1500*S**3, np.float32)
        rot = Rotation.from_euler("y", rake, degrees=True)
        x0, x1 = -.018+ux, .026+ux
        phases = [Phase((CUT_SPEED, 0, 0), (x1-x0)/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), .4, -1)]
        run = P3.probe(points, masses, phases, rotation=rot, start=[x0, uy, top7+edge_above], device=args.device)
        run["phase_names"] = ["skim", "lift"]
        return run, {**info, "edge_above_substrate_m": edge_above, "dome_height_m": dome_h, "rake_deg": rake}
    if case == "S8":
        push_s = .6 if variant == "H1" else .5
        points, masses = block(size_m=(.06, .024, top), spacing=S)
        phases = [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1),
                  Phase((.033/.5, 0, 0), .5, -1), Phase((0, 0, -CUT_SPEED), .5, -1),
                  Phase((-.010, 0, 0), push_s, -1), Phase((0, 0, 0), .2, -1), Phase((CUT_SPEED, 0, 0), .4, -1),
                  Phase((0, 0, MOVE_SPEED), .4, -1)]
        steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
        push_start, hold_end = int(steps[3]), int(steps[5])

        def want(rec):
            sub = int(rec.substep.numpy()[0])
            return sub == steps[0] or sub == push_start or (push_start < sub <= hold_end and (sub-push_start) % S8_DENSE_EVERY == 0)
        run = probe_chunked(points, masses, phases, rotation=VERTICAL_CUT, start=[ux, uy, top+.004], device=args.device,
                            snapshot=want, chunk=S8_DENSE_EVERY)
        run["phase_names"] = ["cut", "lift", "move", "lower", "push", "hold", "withdraw", "lift_end"]
        subs = [int(s["substep"]) for s in run["trail"].snapshots if "F" in s]
        run["frame_names"] = [f"substep_{s}" for s in subs]
        v1_names = [f"substep_{s}" for s in subs if s in (int(steps[0]), push_start) or (s > push_start and (s-push_start) % S8_V1_EVERY == 0)]
        return run, {**info, "push_speed_m_s": .010, "push_travel_m": .010*push_s, "push_start_substep": push_start,
                     "hold_end_substep": hold_end, "dense_every_substeps": S8_DENSE_EVERY, "snapshots": len(subs),
                     "v1_frame_names": v1_names, "declared_plane": ((ux, 0, 0), (1, 0, 0))}
    raise ValueError(case)


def stage_heldout(args):
    rec = args.out/"protocol_record.json"
    if not rec.exists():
        raise SystemExit("protocol_record.json must be written first (stage record)")
    d = args.out/f"heldout_{args.variant}"
    d.mkdir(exist_ok=True)
    cases = [c for c in args.cases.split(",") if c]
    for case in cases:
        for r in range(args.repeats):
            path = d/f"{case}_r{r}_states.npz"
            if path.exists() or (d/f"{case}_r{r}_run.json").exists():
                print(f"[{args.variant} {case} r{r}] exists, skipped (never overwritten)", flush=True)
                continue
            t = time.perf_counter()
            try:
                if case == "S9":
                    s0 = load_s0_meshes(args.out)
                    res = run_s9(args, args.variant, r, s0)
                    write_new(d/f"S9_r{r}.json", res)
                    continue
                run, info = heldout_case(case, args.variant, args)
                frames = P3.frames_of(run, run.get("frame_names"))
                P3.save_states(path, run, frames)
                meta = {"case": case, "variant": args.variant, "repeat": r, "info": info, "runner": run["runner"],
                        "physics_wall_s": run["wall_s"], "recorder_audit": P1.recorder_audit(run),
                        "slot_diagnostics": run["slot_diagnostics"], "frames": [f["name"] for f in frames],
                        "mechanical_components_final": P1.components(dict(run))[0],
                        "recorder_metadata": {k: run["metadata"][k] for k in ("events_logged", "event_overflow", "substeps",
                                                                            "unexpected_code_changes", "log_sha256",
                                                                            "lattice_sha256")},
                        "device": args.device, "wall_s": time.perf_counter()-t}
                if run.get("phase_names"):
                    meta["phases"] = P1.phase_activity(run, run["phase_names"])
                write_new(d/f"{case}_r{r}_run.json", meta)
                print(f"[{args.variant} {case} r{r}] components {meta['mechanical_components_final']} events "
                      f"{meta['recorder_metadata']['events_logged']} frames {len(frames)} physics {run['wall_s']:.1f} s", flush=True)
                del run, frames
            except Exception as error:   # recorded, never hidden
                import traceback
                traceback.print_exc()
                write_new(d/f"{case}_r{r}_ERROR.json", {"case": case, "repeat": r, "error": repr(error),
                                                        "traceback": traceback.format_exc()})


def load_s0_meshes(out):
    p = out/"eval_static/S0_B_meshes.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: (z[f"{k}_faces"], z[f"{k}_vertex_reference"]) for k in B_ALL if f"{k}_faces" in z.files}


# ----------------------------------------------------------------------------------------------- evaluate / gate

def states_path(args, tag, case, r):
    if tag == "saved":
        return SAVED_RUN/f"{case}_r{r}_states.npz"
    return args.out/f"heldout_{tag}/{case}_r{r}_states.npz"


def stage_evaluate(args):
    if not (args.out/"protocol_record.json").exists():
        raise SystemExit("protocol_record.json must be written first (stage record)")
    T1, T2 = P3.protocol_thresholds(), json.loads(PROTO2.read_text())["thresholds_for_candidate_B"]
    out_dir = args.out/f"eval_{args.set}"
    out_dir.mkdir(exist_ok=True)
    for case in [c for c in args.cases.split(",") if c]:
        for r in range(args.repeats):
            target = out_dir/f"{case}_r{r}.json"
            if target.exists():
                print(f"[{args.set} {case} r{r}] exists, skipped (never overwritten)", flush=True)
                continue
            t = time.perf_counter()
            try:
                if case == "S9":
                    if args.set != "saved":
                        print("S9 held-out is produced by the heldout stage", flush=True)
                        continue
                    res = run_s9(args, "saved", r, load_s0_meshes(args.out))
                    write_new(target, res)
                    continue
                path = states_path(args, args.set, case, r)
                run, frames = P3.load_states(path)
                run["runner"] = f"states {path.relative_to(ROOT)}"
                if args.set != "saved":
                    meta = json.loads((path.parent/f"{case}_r{r}_run.json").read_text())
                    run["v1_frame_names"] = meta["info"].get("v1_frame_names")
                res = evaluate_case(case, r, run, frames, args, T1, T2, out_dir, args.set)
                res["states_file"] = str(path.relative_to(ROOT))
                res["states_sha256"] = sha256_file(path)
                write_new(target, res)
                v1 = {k: v["verdict"] for k, v in res["threshold_checks_v1"].items()}
                v2 = {k: v["verdict"] for k, v in res["threshold_checks_v2"].items()}
                print(f"[{args.set} {case} r{r}] v1 {v1} | v2 {v2} | {time.perf_counter()-t:.0f} s", flush=True)
            except Exception as error:
                import traceback
                traceback.print_exc()
                write_new(out_dir/f"{case}_r{r}_ERROR.json", {"case": case, "repeat": r, "error": repr(error),
                                                              "traceback": traceback.format_exc()})


def load_results(args, tag):
    d = args.out/f"eval_{tag}"
    res = {}
    for case in PHYSICAL+("S9",):
        reps = []
        for r in range(args.repeats):
            p = d/f"{case}_r{r}.json" if not (tag != "saved" and case == "S9") else args.out/f"heldout_{tag}/S9_r{r}.json"
            if p.exists():
                reps.append(json.loads(p.read_text()))
        if reps:
            res[case] = reps
    return res


def gate_v2(results, kind):
    """phase3_comparison_gate_v2 for one B kind on one state set (P3.compare with GATE_METRICS_V2)."""
    from robotic_cutting_sim import surface_metrics_v2 as SM2
    per_case = {}
    for case, reps in results.items():
        if case in ("S0", "G2", "S9"):
            continue
        ra = [r["frames"][-1]["reports_v2"][A]["scalars"] for r in reps]
        rb = [r["frames"][-1]["reports_v2"][kind]["scalars"] for r in reps]
        per_case[case] = P3.compare(case, ra, rb, metrics=SM2.GATE_METRICS_V2)
        if case == "S5":
            cuts = sorted(k[3:] for k in (reps[0]["frames"][-1]["reports_v2"][kind].get("blocks") or {}) if k.startswith("cut"))
            for k in cuts:
                pa = [v2_per_cut_scalars(r["frames"][-1]["reports_v2"][A], r["frames"][-1]["reports"][A], k) for r in reps]
                pb = [v2_per_cut_scalars(r["frames"][-1]["reports_v2"][kind], r["frames"][-1]["reports"][kind], k) for r in reps]
                metrics = {m: SM2.GATE_METRICS_V2[m] for m in pa[0] if m in SM2.GATE_METRICS_V2}
                per_case[f"S5_cut{k}"] = P3.compare(f"S5_cut{k}", pa, pb, metrics=metrics)
        if case == "S8":
            sa = [{"slow_motion_residual_p95_s_eligible_pairs": (r.get("p5") or {}).get("max_p95_s", {}).get(A)} for r in reps]
            sb = [{"slow_motion_residual_p95_s_eligible_pairs": (r.get("p5") or {}).get("max_p95_s", {}).get(kind)} for r in reps]
            per_case["S8_temporal_P5"] = P3.compare("S8", sa, sb, metrics=SM2.SLOW_MOTION_METRIC_V2)
    worse = {c: [m for m, d in rows.items() if d["decision"] == "B_WORSE"] for c, rows in per_case.items()}
    p95 = {c: rows["current_p95_s"]["decision"] for c, rows in per_case.items() if "current_p95_s" in rows}
    coverage = {c: per_case[c]["coverage"]["decision"] for c in ("S4", "S7") if c in per_case}
    topo_keys = ("shells_equal_components", "components_without_shell", "shells_spanning_components")
    topo = {c: {k: rows[k]["decision"] for k in topo_keys if k in rows} for c, rows in per_case.items()}
    topo_better = any(v == "B_BETTER" for d in topo.values() for v in d.values())
    topo_worse = any(v == "B_WORSE" for d in topo.values() for v in d.values())
    measured = {c: d for c, d in p95.items() if d != "NOT_COMPARABLE"}
    parts = {"not_worse_on_any_metric": not any(worse.values()),
             "better_current_p95_every_case": bool(measured) and all(d == "B_BETTER" for d in measured.values()),
             "better_coverage_partial_and_chip": bool(coverage) and all(d == "B_BETTER" for d in coverage.values()),
             "better_topology": topo_better and not topo_worse}
    return {"per_case": per_case, "B_worse": worse, "current_p95_decisions": p95, "coverage_decisions": coverage,
            "topology_decisions": topo, "parts": parts, "verdict": "PASS" if all(parts.values()) else "FAIL"}


def gate_v1(results, kind):
    views = {}
    for case, reps in results.items():
        if case in ("S0", "G2", "S9"):
            continue
        vs = []
        for r in reps:
            v = v1_view(r, kind)
            v["threshold_checks"] = r["threshold_checks_v1"]
            v["threshold_checks"] = {A: r["threshold_checks_v1"][A], B1: r["threshold_checks_v1"][kind]}
            vs.append(v)
        views[case] = vs
    return P3.gate(views)


def stage_gate(args):
    summary = {"schema": "wp12_phase3_surface_v2_summary_v1", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "protocol_record_sha256": sha256_file(args.out/"protocol_record.json"),
               "method_frozen_sha256": sha256_file(args.out/"METHOD_FROZEN.txt"), "sets": {}}
    for f in ("S0", "G2"):
        p = args.out/f"eval_static/{f}.json"
        if p.exists():
            d = json.loads(p.read_text())
            summary[f] = ({m: v["verdict"] for m, v in d["threshold_checks"].items()} if f == "S0" else
                          {m: {"v1_planes": v["verdict_v1_planes"], "with_heldout_planes": v["verdict_with_heldout_planes"],
                               **{pl: v[pl]["verdict"] for pl in v if isinstance(v[pl], dict)}} for m, v in d["threshold_checks"].items()})
    overall = {}
    for tag in SETS:
        results = load_results(args, tag)
        if not results:
            continue
        verdicts = {}
        for case, reps in results.items():
            if case == "S9":
                verdicts[case] = [{c: {m: v["threshold_checks"][m]["verdict"] for m in v["threshold_checks"]}
                                   for c, v in r["controls"].items()} for r in reps]
                continue
            verdicts[case] = [{"v1": {m: v["verdict"] for m, v in r["threshold_checks_v1"].items()},
                               "v2": {m: v["verdict"] for m, v in r["threshold_checks_v2"].items()}} for r in reps]
            if case == "S4":
                for r, row in zip(reps, verdicts[case]):
                    row["S9_G4_hold"] = {m: v["verdict"] for m, v in r["g4_hold_phase"].items()}
        complete = all(c in results and len(results[c]) == args.repeats for c in PHYSICAL)
        entry = {"cases_present": sorted(results), "complete": complete, "verdicts": verdicts, "gates": {}}
        for kind in B_GATED:
            try:
                entry["gates"][f"{kind}__protocol_v1"] = gate_v1({c: v for c, v in results.items() if c in PHYSICAL}, kind)
            except Exception as error:
                entry["gates"][f"{kind}__protocol_v1"] = {"ERROR": repr(error)}
            try:
                entry["gates"][f"{kind}__protocol_v2"] = gate_v2({c: v for c, v in results.items() if c in PHYSICAL}, kind)
            except Exception as error:
                import traceback
                traceback.print_exc()
                entry["gates"][f"{kind}__protocol_v2"] = {"ERROR": repr(error)}
        summary["sets"][tag] = entry
        overall[tag] = {k: (v.get("verdict", "ERROR"), complete) for k, v in entry["gates"].items()}
    decision = {}
    for kind in B_GATED:
        for proto in ("protocol_v1", "protocol_v2"):
            key = f"{kind}__{proto}"
            per = {tag: overall.get(tag, {}).get(key) for tag in SETS}
            ok = all(p is not None and p[1] and p[0] == "PASS" for p in per.values())
            missing = [t for t, p in per.items() if p is None or not p[1]]
            decision[key] = {"per_set": per, "verdict": "PASS" if ok else ("INCOMPLETE" if missing else "FAIL"),
                             "missing_sets": missing}
    summary["phase3_gate"] = decision
    summary["phase3_gate_rule"] = "B v2 vs A under protocol v2 decides: PASS only if the rule holds on saved AND H1 AND H2"
    path = next_free(args.out, "summary", ".json")
    path.write_text(json.dumps(jsonable(summary), indent=1))
    print(json.dumps(jsonable({"phase3_gate": decision, "S0": summary.get("S0"), "G2": summary.get("G2")}), indent=1))
    print(f"wrote {path}")


# ----------------------------------------------------------------------------------------------- record / figures

def stage_record(args):
    out = args.out
    if out.exists() and sorted(p.name for p in out.iterdir()) != ["METHOD_FROZEN.txt"]:
        raise SystemExit(f"{out} must contain only METHOD_FROZEN.txt before the record stage")
    if not (out/"METHOD_FROZEN.txt").exists():
        raise SystemExit("METHOD_FROZEN.txt is required before any measurement")
    import warp as wp
    import newton
    rel = lambda p: str(Path(p).relative_to(ROOT))  # noqa: E731
    protos = {}
    for tag, path, freeze in (("v1", PROTO1, FREEZE1), ("v2", PROTO2, FREEZE2)):
        last = P3.git("log", "-1", "--format=%H", "--", rel(path))
        dirty = P3.git("status", "--porcelain", "--", rel(path))
        protos[tag] = {"file": rel(path), "sha256": sha256_file(path), "last_commit": last, "freeze_commit": freeze,
                       "unchanged_since_freeze": last.startswith(freeze) and not dirty}
    heldout = {}
    for case, (s1, s2) in SEEDS.items():
        heldout[case] = {"H1": {"seed": s1, "offset_s": None if case == "S1_G3a" else offsets(s1).round(3).tolist()},
                         "H2": {"seed": s2, "offset_s": None if case == "S1_G3a" else offsets(s2).round(3).tolist()}}
    declared = json.loads(PROTO2.read_text())["held_out_states"]
    mismatch = [c for c in heldout if c != "S1_G3a" and any(
        list(np.round(heldout[c][h]["offset_s"], 3)) != list(np.round(declared[c][h]["offset_s"], 3)) for h in ("H1", "H2"))]
    record = {**DECLARATION, "protocols": {**DECLARATION["protocols"], "files": protos},
              "method_frozen": {"file": "METHOD_FROZEN.txt", "sha256": sha256_file(out/"METHOD_FROZEN.txt")},
              "modules": {p: sha256_file(ROOT/p) for p in ("src/robotic_cutting_sim/cut_aware_surface.py",
                                                           "src/robotic_cutting_sim/cut_aware_surface_v2.py",
                                                           "src/robotic_cutting_sim/surface_metrics_v1.py",
                                                           "src/robotic_cutting_sim/surface_metrics_v2.py",
                                                           "scripts/run_wp12_phase3_surface.py",
                                                           "scripts/run_wp12_phase3_surface_v2.py")},
              "cases": {"static": ["S0", "G2 (v1 planes + n213_heldout, n60_xz_heldout)"],
                        "saved": list(PHYSICAL)+["S9 (v1 controls re-run fresh: no saved states)"],
                        "H1": list(PHYSICAL)+["S9 (+ G4 hold of S4 H1)"], "H2": list(PHYSICAL)+["S9 (+ G4 hold of S4 H2)"]},
              "repeats": args.repeats, "saved_states_dir": rel(SAVED_RUN),
              "saved_states_sha256": {f"{c}_r{r}": sha256_file(SAVED_RUN/f"{c}_r{r}_states.npz")
                                      for c in PHYSICAL for r in range(args.repeats)},
              "heldout_offsets_recomputed": heldout, "heldout_offsets_match_protocol": not mismatch,
              "heldout_offset_mismatch": mismatch,
              "command": " ".join([sys.executable, *sys.argv]), "git_head": P3.git("rev-parse", "HEAD"),
              "git_dirty_paths": len(P3.git("status", "--porcelain").splitlines()),
              "warp": wp.__version__, "newton": newton.__version__, "numpy": np.__version__, "python": platform.python_version(),
              "runner_parameters": P1.PROTOCOL, "written_before_any_measurement_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out/"protocol_record.json").write_text(json.dumps(jsonable(record), indent=1))
    print(json.dumps(jsonable({"protocols": protos, "offsets_match": not mismatch}), indent=1))


FIGURES = {   # name: (set, package, digit, plane_point, normal, window)
    "saved_S4_partial": ("saved", "S4_r0_package.npz", 0, None, None, .006),
    "saved_S7_chip": ("saved", "S7_r0_package.npz", 0, "component:1", None, .008),
    "saved_S1_G3b_oblique": ("saved", "S1_G3b_r0_package.npz", 0, None, None, .006),
    "saved_S6_slices": ("saved", "S6_r0_package.npz", 1, None, None, .014),
    "H1_S4_partial": ("H1", "S4_r0_package.npz", 0, None, None, .006),
    "H2_S4_partial_yaw10": ("H2", "S4_r0_package.npz", 0, None, None, .006),
    "H2_S7_chip": ("H2", "S7_r0_package.npz", 0, "component:1", None, .008),
    "H2_S1_G3b_yaw35": ("H2", "S1_G3b_r0_package.npz", 0, None, None, .006),
    "H2_S5_cross75": ("H2", "S5_r0_package.npz", 2, None, None, .010),
}


def stage_figures(args):
    from robotic_cutting_sim import cut_debug_view as V
    media = args.media if args.media.is_absolute() else ROOT/args.media
    media.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, (tag, pkg_name, digit, point, normal, window) in FIGURES.items():
        path = args.out/f"eval_{tag}"/pkg_name
        if not path.exists():
            continue
        pkg = V.load(path)
        if isinstance(point, str) and point.startswith("component:"):
            sel = np.asarray(pkg["labels"]) == int(point.split(":")[1])
            point = np.median(np.asarray(pkg["positions"], float)[sel], 0) if sel.any() else None
        png = media/f"phase3v2_{name}_A_B1_B2.png"
        if png.exists():
            raise SystemExit(f"Refusing to overwrite {png}")
        V.render_comparison(pkg, png, profiles=(A, B1, B2, B2NOM4), digit=digit, window_m=window, plane_point=point,
                            plane_normal=normal, slab_m=1.2*S,
                            title=f"{name} (r0): A / B v1 / B v2 (frozen) / B v2 without M4 (labelled diagnostic ablation)")
        written[name] = str(png.relative_to(ROOT))
    idx = media/"index_phase3v2.json"
    if idx.exists():
        raise SystemExit(f"Refusing to overwrite {idx}")
    idx.write_text(json.dumps({"source_run": str(args.out.relative_to(ROOT)), "figures": written}, indent=1))
    print(json.dumps(written, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("record", "static", "heldout", "evaluate", "gate", "figures"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--variant", choices=("H1", "H2"), default="H1")
    p.add_argument("--set", choices=SETS, default="saved")
    p.add_argument("--cases", default=",".join(PHYSICAL+("S9",)))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--media", type=Path, default=Path("docs/media/wp12/phase3"))
    args = p.parse_args()
    args.out = args.out if args.out.is_absolute() else ROOT/args.out
    {"record": stage_record, "static": stage_static, "heldout": stage_heldout, "evaluate": stage_evaluate,
     "gate": stage_gate, "figures": stage_figures}[args.stage](args)


if __name__ == "__main__":
    main()
