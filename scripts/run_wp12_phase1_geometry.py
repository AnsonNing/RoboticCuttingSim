#!/usr/bin/env python3
"""WP-12 Phase 1: geometry tests G1-G8 and layered cut-surface attribution (kinematic, tlmpm_v5 law 8).

Protocol (declared in ``PROTOCOL`` and written to ``<out>/protocol.json`` before anything runs; a copy is kept
in ``configs/wp12/phase1_geometry_protocol_v1.json``): 1 mm particles, 4 mm grid, dt 0.1 ms, tlmpm_v5, cutting
law 8, the read-only cut_events_v1 recorder with its diagnostic assignment-instant extension
(cut_event_snapshots_v1) on every physical case. Runners: ``kinematic_env`` (G3a, the registered CUT-01-R
scripted through-cut, seed 42), ``blade_probe.drive_blade`` (G1 rest, G3b, G4, G5, G7, G8) and
``sequence_blade_probe.drive_blade_sequence`` (G6, which needs a second blade orientation; it reproduces
drive_blade bit for bit for a fixed orientation). G2 is a geometric unit test with synthetic codes and Hermite
data on analytic planes; it is never injected into physics. Surfaces: Newton ParticleSurface through
``cutting_surface.CuttingSurface`` with both profiles (``legacy``: 1 mm voxel, Newton defaults; ``current_volume_v1``:
the profile every configured/registered task uses). ``--baseline`` adds the eight Phase 0 baseline packages
(midpoint-fallback locus, no recorder).

Outputs (never overwritten): ``<out>/protocol.json``, ``<out>/<case>.json`` (metrics, attribution, verdict),
``<out>/<case>_package.npz`` (compact float32 viewer package), ``<out>/summary.json``.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase1_geometry.py --device cuda:1 \
      --out artifacts/wp12/phase1_001 --baseline
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

S, DX, DT = .001, .004, 1e-4
CUT_SPEED, MOVE_SPEED = .02, .05
ENERGY = 5.   # blade_probe default (WP-11 Phase M probes)

PROTOCOL = {
    "schema": "wp12_phase1_geometry_protocol_v1",
    "solver": "tlmpm_v5", "cutting_law": 8, "particle_spacing_m": S, "grid_dx_m": DX, "material_dt_s": DT,
    "recorder": "cut_events_v1 + cut_event_snapshots_v1 (read-only)", "tool_profile": "ruler_knife_v1",
    "probe_material": "TLMPMConfig defaults (density 1500, E 50 kPa, nu 0.3, yield 1 kPa), cutting_energy 5 J/m^2, "
                      "legacy tray floor/walls (blade_probe); CUT-01-R uses its task recipe. Uncalibrated.",
    "cut_speed_m_s": CUT_SPEED, "move_speed_m_s": MOVE_SPEED,
    "seeds": {"G3a_CUT-01-R": [42, 42], "probes": "deterministic sampling (no RNG); GPU repeats measure "
              "run-to-run nondeterminism", "G3b_repeats": 2, "metric_sampling_seed": 0},
    "surface_profiles": {"legacy": "ParticleSurface voxel 1 mm (min(s, 1.5 mm)), radius s/2, Newton defaults "
                                   "(threshold 0.25), smooth 0", "current_volume_v1": "voxel s/2, kernel 3 s, threshold 0.5, "
                                   "radius (s/2) cbrt(det F), smooth 0 (used by every configured/registered task)"},
    "unmeshable_policy": "skip_presentation_v1 (skips are recorded; outputs identical to 'raise' when none)",
    "metrics": "surface_metrics_v1 (region_rule_v1, patch radius 2 mm, normal radius 3 mm, coverage radius 1 s, "
               "core margin 3 s) and surface_attribution_v1",
    "criteria": {
        "G1": "PASS iff current_volume_v1 on the static box is a closed oriented manifold (0 boundary/non-manifold/"
              "inconsistent edges), 1 shell, Euler 2, |dV|/V <= 5 %, area-weighted whole-surface P95 to the analytic box "
              "<= 0.5 s (SURFACE_GEOMETRY_VALIDATION promotion criteria), and the 0.2 s rest run writes 0 digits and "
              "0 events and stays one closed shell. legacy: MEASURED.",
        "G2": "PASS iff (metric self-check) the synthetic Hermite locus lies on the analytic plane (max <= 1e-9 m), an "
              "exact planar mesh scores distance max <= 1e-9 m and normal error max <= 1e-3 deg, and the link-midpoint "
              "staircase stays within s/2 of the plane. ParticleSurface numbers: MEASURED (zero-physics floor).",
        "G3": "PASS iff mechanical components == 2, every conflicting link is logged exactly once, 0 sign-inconsistent "
              "events, assignment substeps match the log, and the current_volume_v1 mesh has shells == components, "
              "closed. All geometry: MEASURED.",
        "G4": "PASS iff mechanical components == 1 and lattice components == 1 (no false full separation), logged == "
              "conflicting, and every assignment-instant Hermite point lies within 1 s of the commanded sweep and no "
              "lower than the lowest edge position - 1 s. Kerf/crack-front geometry: MEASURED.",
        "G5": "PASS iff logged == conflicting, 0 mesh shells built from two mechanical components (provenance), and every skipped "
              "fragment is recorded. Chip detachment, chip volume and fragments: MEASURED.",
        "G6": "PASS iff mechanical components == 6 and logged == conflicting for each of the 3 digits. Run with "
              "slots_per_node 8 first; a SlotOverflowError is recorded (solver limit, never merged) and the case reruns "
              "with 16. Intersections: MEASURED.",
        "G7": "PASS iff the disarmed push/hold/withdraw phases write 0 digits and 0 events, the mechanical component "
              "count after the cut equals the final count (2), and no mesh shell is built from two components "
              "(extractor provenance) in any snapshot. Interpenetration, gaps and temporal residuals: MEASURED.",
        "G8": "PASS iff each control writes 0 digits, 0 events, 0 unexpected code changes, keeps 1 mechanical "
              "component and a single closed current_volume_v1 shell.",
    },
    "attribution": "surface_attribution_v1 layers L1-L7 on the same core links (>= 3 lattice layers from the exterior)",
}


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        return jsonable(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def git():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    return {"head": head, "dirty_paths": len(dirty.splitlines())}


# ----------------------------------------------------------------------------------------- physics runners

def probe_run(points, masses, phases, *, rotation=None, start, cuts=1, device, sequence=False, snapshot=None,
              trail_stride=10, slots_per_node=None):
    """drive_blade / drive_blade_sequence with the diagnostic recorder; returns the common run dict."""
    from robotic_cutting_sim import cut_event_snapshots as cs
    from robotic_cutting_sim.blade_probe import drive_blade
    from robotic_cutting_sim.sequence_blade_probe import drive_blade_sequence
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
    trail = cs.PoseTrail(stride=trail_stride, want_snapshot=snapshot, snapshot_F=snapshot is not None)
    start_t = time.perf_counter()
    with cs.installed(hook=trail) as made:
        if sequence:
            r = drive_blade_sequence(RULER_KNIFE_V1, start, phases, points=points, masses=masses, spacing=S,
                                     cutting_energy=ENERGY, cuts=cuts, device=device, slots_per_node=slots_per_node,
                                     gpu_graph=device.startswith("cuda"), cut_events="cut_events_v1")
        else:
            r = drive_blade(RULER_KNIFE_V1, rotation, start, phases, points=points, masses=masses, spacing=S,
                            cutting_energy=ENERGY, cuts=cuts, cutting_law=8, solver_version=5, device=device,
                            gpu_graph=device.startswith("cuda"), record_history=True, cut_events="cut_events_v1")
    wall = time.perf_counter()-start_t
    rec = made[0]
    trail.poses, trail.cuts = host_pose_trail(start, phases, rotation, stride=trail_stride)
    log = r["cut_events"]["log"]
    assign = rec.assignment_positions(log)
    st = r["state"]
    lattice = np.round((points-points.min(0))/S).astype(int)
    steps = [int(round(p.duration_s/DT)) for p in phases]
    return {"reference": np.asarray(points, float), "lattice": lattice, "masses": np.asarray(masses, float),
            "density": 1500., "positions": st["positions"].astype(float), "codes": st["codes"], "F": st["F"].astype(float),
            "contact_work": st["contact_work_j"], "cut_progress": st["cut_progress"],
            "nodes": r["reference_stencil_nodes"], "weights": r["reference_stencil_weights"],
            "gradients": r["reference_stencil_gradients"], "cuts": cuts, "log": log,
            "activation": r["cut_events"]["activation"], "metadata": r["cut_events"]["metadata"],
            "assign": assign, "trail": trail, "half_length": float(RULER_KNIFE_V1.half_width_m),
            "blade_polygon": RULER_KNIFE_V1.tlmpm_profile()[0], "phase_substeps": steps,
            "labelled_after_phase": r["labelled_after_phase"], "digit_after_phase": r["digit_labelled_after_phase"],
            "wall_s": wall, "runner": "drive_blade_sequence" if sequence else "blade_probe.drive_blade",
            "slot_diagnostics": r["slot_diagnostics"]}


def host_pose_trail(start, phases, rotation=None, stride=10, chunk=200):
    """Substep edge poses exactly as drive_blade / drive_blade_sequence generate them (position += v dt per substep,
    200-substep chunks), every ``stride``-th kept, one chain per phase. (The device pose buffer cannot be read back
    reliably after the probe rebinds it for a different chunk length.)"""
    position = np.asarray(start, float)
    poses, cuts = [], []
    for phase in phases:
        quat = (Rotation.from_quat(phase.rotation_xyzw) if hasattr(phase, "rotation_xyzw") else rotation).as_quat()
        v = np.asarray(phase.velocity_m_s, float)
        steps = int(round(phase.duration_s/DT))
        rows = []
        for first in range(0, steps, chunk):
            count = min(chunk, steps-first)
            rows.append(np.array([np.r_[position+v*DT*i, quat] for i in range(0, count+1, 1)]))
            position = position+v*DT*count
        chain = np.vstack(rows)
        keep = np.unique(np.r_[np.arange(0, len(chain), stride), len(chain)-1])
        poses.append(chain[keep].astype(np.float32))
        cuts.append(np.full(len(keep), phase.cut, np.int8))
    return poses, cuts


def cut01r_run(device, seed=42):
    """Registered CUT-01-R scripted reference edge (kinematic ideal tracking), law 8, tlmpm_v5, recorder on."""
    import warp as wp
    from robotic_cutting_sim import cut_event_snapshots as cs
    from robotic_cutting_sim.kinematic_env import KinematicConfig, KinematicCuttingBackend
    trail = cs.PoseTrail(stride=10)
    start_t = time.perf_counter()
    with cs.installed(hook=trail) as made:
        backend = KinematicCuttingBackend(KinematicConfig("CUT-01-R", seed=seed, particle_spacing_m=S, material_dt=DT,
                                                          device=device, cutting_law=8, solver="tlmpm_v5",
                                                          cut_events="cut_events_v1"))
        backend.reset(seed=seed)
        task, material = backend.task, backend.material
        steps, armed = 0, []
        while backend.time_s+task.control_dt_s <= task.duration_s+1e-9:
            _, info = backend.step()
            armed.append(int(material.last_diagnostics.get("cut_index", -1)))
            steps += 1
        wp.synchronize_device(device)
    wall = time.perf_counter()-start_t
    rec, bed = made[0], material.bed
    st = bed.snapshot()
    log = rec.log()
    out = {"reference": material.points.astype(float), "lattice": np.asarray(material.lattice), "masses": material.masses.astype(float),
           "density": float(task.material.density_kg_m3), "positions": st["positions"].astype(float), "codes": st["codes"],
           "F": st["F"].astype(float), "contact_work": st["contact_work_j"], "cut_progress": st["cut_progress"],
           "nodes": bed.nodes.numpy(), "weights": bed.weights.numpy(), "gradients": bed.gradients.numpy(), "cuts": 1,
           "log": log, "activation": rec.activation(), "metadata": rec.metadata(), "assign": rec.assignment_positions(log),
           "trail": trail, "half_length": float(bed.config.blade_half_width), "blade_polygon": np.asarray(bed.blade_profile_xz),
           "phase_substeps": [steps*material.substeps], "labelled_after_phase": [int(np.count_nonzero(st["codes"]))],
           "digit_after_phase": [[int(np.count_nonzero(st["codes"]))]], "wall_s": wall,
           "runner": "kinematic_env.KinematicCuttingBackend (CUT-01-R scripted edge)", "seed": seed,
           "control_steps": steps, "armed_control_steps": int(np.sum(np.asarray(armed) >= 0)),
           "slot_diagnostics": bed.check_slots()}
    backend.close()
    return out


# ----------------------------------------------------------------------------------------- analysis

def components(run):
    from robotic_cutting_sim.cut_topology import mechanical_components_packed, reference_components_packed
    count, labels = mechanical_components_packed(run["nodes"], run["weights"], run["codes"], run["cuts"],
                                                 gradients=run["gradients"])
    lat_count, _ = reference_components_packed(run["reference"], run["lattice"], run["codes"])
    mass = np.bincount(labels, weights=run["masses"])
    order = np.argsort(-mass, kind="stable")
    rank = np.empty(count, int)
    rank[order] = np.arange(count)
    run["_rank"] = rank   # CuttingSurface numbers components like mechanical_components_packed (raw)
    return int(count), rank[labels], int(lat_count), (mass[order]/mass.sum())


def face_components(info, rank, triangles):
    """Per-triangle mechanical component (ranked labels) from CuttingSurface component_ranges (provenance)."""
    out = np.full(triangles, -1, np.int64)
    for row in info["component_ranges"]:
        out[row["first_triangle"]:row["first_triangle"]+row["triangles"]] = rank[row["mechanical_component"]]
    return out


def extract_surfaces(run, device, positions=None, codes=None, F=None, profiles=("legacy", "current_volume_v1")):
    from robotic_cutting_sim.cutting_surface import CuttingSurface, CurrentMechanicalPartition
    positions = run["positions"] if positions is None else positions
    codes = run["codes"] if codes is None else codes
    F = run["F"] if F is None else F
    partition = CurrentMechanicalPartition(run["nodes"], run["weights"], run["gradients"], run["cuts"], encoding="packed2")
    out = {}
    for profile in profiles:
        voxel = S/2 if profile == "current_volume_v1" else min(S, .0015)
        surface = CuttingSurface(partition, S, voxel, device, profile=profile, unmeshable="skip_presentation_v1")
        t = time.perf_counter()
        v, f, info = surface.extract(positions, codes, F if profile == "current_volume_v1" else None)
        out[profile] = {"vertices": v.astype(np.float64), "faces": f.astype(np.int64), "wall_s": time.perf_counter()-t,
                        "skipped": info.get("unmeshable_skipped"), "surface": surface, "info": info}
        if run.get("_rank") is not None and positions is run["positions"]:
            out[profile]["face_component"] = face_components(info, run["_rank"], len(f))
    return out


def recorder_audit(run):
    from robotic_cutting_sim import cut_events as ce
    neighbours = ce.lattice_neighbours(run["lattice"])
    out = {}
    for k in range(run["cuts"]):
        conflicting = ce.links_conflicting_in_cut(run["codes"], neighbours, k)
        logged = ce.log_links(run["log"], k)
        out[str(k)] = {"conflicting": len(conflicting), "logged": len(logged), "equal": conflicting == logged}
    flags = run["log"]["flags"]
    return {"per_cut": out, "all_equal": all(v["equal"] for v in out.values()),
            "events": int(len(run["log"])), "sign_inconsistent": int(np.count_nonzero(flags & 1)),
            "lead": int(np.count_nonzero(flags & 4)), "blade_z_fallback": int(np.count_nonzero(flags & 8)),
            "substep_matches_log": bool(run["assign"]["substep_matches_log"]),
            "unexpected_code_changes": int(run["metadata"]["unexpected_code_changes"]),
            "event_overflow": int(run["metadata"]["event_overflow"])}


def phase_activity(run, names):
    """Per phase: substeps, newly assigned particles and links logged (from the activation rows)."""
    act = run["activation"]
    rows, start = {}, 0
    for name, n in zip(names, run["phase_substeps"]):
        a = act[start:start+n]
        rows[name] = {"substeps": int(n), "newly_assigned": int(a["newly_assigned"].sum()),
                      "links_logged": int(a["links_logged"].sum()),
                      "unexpected_code_changes": int(a["unexpected_code_changes"].sum()),
                      "stroke_engaged_substeps": int((a["stroke_engaged"] > 0).sum()),
                      "segment_ok_substeps": int((a["segment_ok_count"] > 0).sum()),
                      "segment_plow_substeps": int((a["segment_plow_count"] > 0).sum())}
        start += n
    return rows


def box_metrics(vertices, faces, lo, hi, spacing):
    """Uncut box: distance of mesh samples to the analytic box (area-weighted over centroids + vertices),
    volume, corner distance and face normals away from edges."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre, half = (lo+hi)/2, (hi-lo)/2

    def sdf(p):
        q = np.abs(p-centre)-half
        return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(q.max(1), 0)
    n, area, centroid = SM.triangle_normals(vertices, faces)
    d_c, d_v = np.abs(sdf(centroid)), np.abs(sdf(vertices))
    w = np.r_[area, np.full(len(d_v), area.sum()/max(len(d_v), 1))]
    allv = np.r_[d_c, d_v]
    order = np.argsort(allv)
    cdf = np.cumsum(w[order])/w.sum()
    p95 = float(allv[order][np.searchsorted(cdf, .95)])
    tri = np.asarray(vertices)[faces]
    volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()/6)
    exact = float(np.prod(hi-lo))
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    dcorner = SM.point_mesh_distance(corners, vertices, faces)
    # face-interior triangles (>= 3 s from every box edge): normal vs the face normal
    q = np.abs(centroid-centre)-half
    axis = np.argmax(q, 1)
    other = np.sort(-q, 1)  # distances inside to the faces
    interior = np.sort(np.abs(q), 1)[:, 1] >= 3*spacing
    face_n = np.zeros_like(centroid)
    face_n[np.arange(len(axis)), axis] = np.sign((centroid-centre)[np.arange(len(axis)), axis])
    ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n, face_n), -1, 1)))
    return {"area_weighted_p95_m": p95, "area_weighted_p95_spacings": p95/spacing, "max_m": float(allv.max()),
            "volume_m3": volume, "analytic_volume_m3": exact, "volume_relative_error": (volume-exact)/exact,
            "corner_distance_m": SM._quantiles(dcorner, spacing),
            "face_interior_normal_error_deg": SM._quantiles(ang[interior]),
            "face_interior_outward_fraction": float((ang[interior] < 90).mean()) if interior.any() else None}


def analyse(case_id, run, args, *, declared_planes=None, phase_names=None, extra=None, surfaces=True):
    from robotic_cutting_sim import surface_attribution_v1 as SA
    from robotic_cutting_sim import surface_metrics_v1 as SM
    t0 = time.perf_counter()
    count, labels, lat_count, fractions = components(run)
    result = {"case": case_id, "runner": run["runner"], "physics_wall_s": run["wall_s"], "particles": int(len(run["codes"])),
              "mechanical_components": count, "lattice_components": lat_count, "mass_fractions": fractions[:10].tolist(),
              "labelled": int(np.count_nonzero(run["codes"])), "recorder_audit": recorder_audit(run),
              "slot_diagnostics": run["slot_diagnostics"], "recorder_metadata": {k: run["metadata"][k] for k in (
                  "events_logged", "event_overflow", "substeps", "unexpected_code_changes", "log_sha256", "lattice_sha256")}}
    if phase_names:
        result["phases"] = phase_activity(run, phase_names)
    locus = SM.material_locus(run["reference"], run["lattice"], run["positions"], run["codes"], encoding="packed2",
                              cuts=run["cuts"], F=run["F"], events=run["log"])
    mid_locus = SM.material_locus(run["reference"], run["lattice"], run["positions"], run["codes"], encoding="packed2",
                                  cuts=run["cuts"], F=run["F"], events=None)
    result["locus"] = {"source": locus["source"], "rows_by_source": locus["rows_by_source"],
                       "events_unmatched": locus["events_unmatched"], "normal_against_link": locus["normal_against_link"]}
    meshes, reports = {}, {}
    if surfaces:
        meshes = extract_surfaces(run, args.device)
        for profile, m in meshes.items():
            rep = SM.surface_report(run["reference"], run["lattice"], run["positions"], run["codes"], m["vertices"], m["faces"],
                                    spacing=S, labels=labels, masses=run["masses"], density=run["density"], cuts=run["cuts"],
                                    F=run["F"], locus=locus, face_component=m.get("face_component"))
            rep["extract_wall_s"] = m["wall_s"]
            rep["unmeshable_skipped"] = m["skipped"]
            reports[profile] = rep
        rep = SM.surface_report(run["reference"], run["lattice"], run["positions"], run["codes"],
                                meshes["current_volume_v1"]["vertices"], meshes["current_volume_v1"]["faces"], spacing=S,
                                labels=labels, cuts=run["cuts"], F=run["F"], locus=mid_locus, validity=False)
        result["current_volume_v1_vs_midpoint_locus"] = {k: rep.get(k) for k in (
            "status", "distance_to_side_locus_plane", "distance_to_side_locus_plane_core", "signed_bias",
            "locus_to_mesh_distance", "coverage", "normal_error_deg_core", "roughness_mesh_core", "roughness_locus_core")}
        # temporal stability at a fixed material state: cached surface again, and a fresh extractor
        again = meshes["current_volume_v1"]["surface"].extract(run["positions"], run["codes"], run["F"])
        fresh = extract_surfaces(run, args.device, profiles=("current_volume_v1",))["current_volume_v1"]
        cv = meshes["current_volume_v1"]
        result["temporal_fixed_state"] = {
            "cached_again": SM.temporal_stability(cv["vertices"], cv["faces"], again[0], again[1], run["positions"],
                                                  run["positions"], spacing=S),
            "fresh_extractor": SM.temporal_stability(cv["vertices"], cv["faces"], fresh["vertices"], fresh["faces"],
                                                     run["positions"], run["positions"], spacing=S)}
    result["surface_reports"] = reports
    result["topology_layer"] = SA.topology_layer(run["nodes"], run["weights"], run["gradients"], run["codes"], run["cuts"],
                                                 lattice=run["lattice"], labels=labels, masses=run["masses"],
                                                 reference=run["reference"])
    chains = run["trail"].chains() if run.get("trail") is not None else None
    result["attribution"] = {}
    for k in range(run["cuts"]):
        plane = None if not declared_planes else declared_planes.get(k)
        result["attribution"][str(k)] = SA.attribute_cut(
            k, locus=locus, lattice=run["lattice"], reference=run["reference"], positions=run["positions"], labels=labels,
            spacing=S, chains=chains, half_length_m=run["half_length"], log=run["log"],
            hermite_assign=run["assign"]["hermite_current_at_assignment"],
            assign_endpoints=(run["assign"]["x_p"], run["assign"]["x_q"]), declared_plane=plane,
            mesh_reports={p: r for p, r in reports.items()}, midpoint_locus=mid_locus)
    if extra:
        result.update(extra(run, labels, meshes, locus))
    result["analysis_wall_s"] = time.perf_counter()-t0
    save_package(args.out/f"{case_id}_package.npz", run, labels, meshes, locus)
    return result, labels, meshes, locus


def save_package(path, run, labels, meshes, locus):
    poses = cuts = np.zeros((0, 7), np.float32)
    if run.get("trail") is not None and run["trail"].poses:
        poses = np.vstack(run["trail"].poses).astype(np.float32)
        cuts = np.concatenate(run["trail"].cuts).astype(np.int8)
    a = {"schema": "wp12_cut_debug_package_v1", "spacing_m": S, "cuts": run["cuts"],
         "reference": run["reference"].astype(np.float32), "positions": run["positions"].astype(np.float32),
         "codes": run["codes"].astype(np.int32), "labels": labels.astype(np.int32), "lattice": run["lattice"].astype(np.int32),
         "masses": run["masses"].astype(np.float32), "F": run["F"].astype(np.float32),
         "contact_work": np.asarray(run["contact_work"], np.float32), "cut_progress": np.asarray(run["cut_progress"], np.float32),
         "hermite_P": locus["P"].astype(np.float32), "hermite_N": locus["N"].astype(np.float32),
         "hermite_cut": locus["cut"].astype(np.int8), "hermite_pairs": locus["pairs"].astype(np.int32),
         "hermite_is_recorded": locus["hermite"], "side_points": locus["side_points"].astype(np.float32),
         "poses": poses, "pose_cuts": cuts, "blade_polygon_xz": np.asarray(run["blade_polygon"], np.float32),
         "blade_half_length_m": run["half_length"]}
    if run.get("assign") is not None:
        a["hermite_assign"] = run["assign"]["hermite_current_at_assignment"].astype(np.float32)
        a["log_p"], a["log_q"], a["log_cut"] = run["log"]["p"], run["log"]["q"], run["log"]["cut"].astype(np.int8)
    for profile, m in meshes.items():
        a[f"mesh_{profile}_vertices"] = m["vertices"].astype(np.float32)
        a[f"mesh_{profile}_faces"] = m["faces"].astype(np.int32)
    np.savez_compressed(path, **a)


def write(args, case_id, result):
    path = args.out/f"{case_id}.json"
    path.write_text(json.dumps(jsonable(result), indent=1))
    print(f"[{case_id}] verdict {result.get('verdict')} | components {result.get('mechanical_components')} | "
          f"physics {result.get('physics_wall_s', 0):.1f} s, analysis {result.get('analysis_wall_s', 0):.1f} s", flush=True)


# ----------------------------------------------------------------------------------------- cases

def g1(args):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    points, masses = block(size_m=(.04, .024, .012), spacing=S)
    lo, hi = points.min(0)-S/2, points.max(0)+S/2
    run = probe_run(points, masses, [Phase((0, 0, 0), .2, -1)], rotation=VERTICAL_CUT, start=[0, 0, .06], device=args.device)
    static = dict(run, positions=run["reference"].copy(), codes=np.zeros_like(run["codes"]),
                  F=np.tile(np.eye(3), (len(points), 1, 1)))
    out = {"static": {}, "rest": {}}
    for name, state in (("static", static), ("rest", run)):
        meshes = extract_surfaces(state, args.device)
        count, labels, _, _ = components(state)
        for profile, m in meshes.items():
            row = {"validity": SM.mesh_validity_v1(m["vertices"], m["faces"]), "extract_wall_s": m["wall_s"],
                   "volume": SM.volume_report(m["vertices"], m["faces"], state["positions"], labels,
                                              state["masses"]/state["density"], state["F"])}
            if name == "static":
                row["analytic_box"] = box_metrics(m["vertices"], m["faces"], lo, hi, S)
            out[name][profile] = row
        if name == "static":
            save_package(args.out/"G1_static_package.npz", state, labels, meshes,
                         SM.material_locus(state["reference"], state["lattice"], state["positions"], state["codes"],
                                           cuts=1, F=state["F"]))
    out["rest"]["recorder_audit"] = recorder_audit(run)
    out["rest"]["labelled"] = int(np.count_nonzero(run["codes"]))
    out["rest"]["max_displacement_m"] = float(np.linalg.norm(run["positions"]-run["reference"], axis=1).max())
    cv, cr = out["static"]["current_volume_v1"], out["rest"]["current_volume_v1"]
    v = cv["validity"]
    checks = {"closed_oriented_manifold": v["boundary_edges"] == 0 and v["non_manifold_edges"] == 0
              and v["inconsistently_oriented_edges"] == 0 and v["shells_negative_volume"] == 0,
              "one_shell": v["shells"] == 1, "euler_2": v["shells_euler_not_2"] == 0,
              "volume_within_5pct": abs(cv["analytic_box"]["volume_relative_error"]) <= .05,
              "p95_within_half_spacing": cv["analytic_box"]["area_weighted_p95_spacings"] <= .5,
              "rest_no_digits_no_events": out["rest"]["labelled"] == 0 and out["rest"]["recorder_audit"]["events"] == 0,
              "rest_one_closed_shell": cr["validity"]["shells"] == 1 and cr["validity"]["watertight"]}
    result = {"case": "G1", "box_m": (hi-lo).tolist(), "particles": len(points), "physics_wall_s": run["wall_s"],
              **out, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL",
              "legacy": "MEASURED (no criterion)"}
    write(args, "G1", result)
    return result


def synthetic_events(reference, lattice, codes, origin, normal):
    from robotic_cutting_sim import cut_events as ce
    from robotic_cutting_sim import surface_metrics_v1 as SM
    pairs, cut = SM.conflicting_links_per_cut(lattice, codes, cuts=1)
    a, b = pairs[:, 0], pairs[:, 1]
    sa, sb = (reference[a]-origin)@normal, (reference[b]-origin)@normal
    t = sa/(sa-sb)
    log = np.zeros(len(a), ce.LOG_DTYPE)
    log["p"], log["q"], log["cut"], log["t"] = a, b, 0, t
    log["s_p"], log["s_q"] = sa, sb
    log["hermite_reference"] = reference[a]+t[:, None]*(reference[b]-reference[a])
    log["normal_reference"] = normal
    log["normal_current"] = normal
    return log


def g2(args):
    """Synthetic codes/Hermite on analytic planes (geometric unit test, never physics)."""
    from robotic_cutting_sim import surface_attribution_v1 as SA
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import block
    points, masses = block(size_m=(.04, .024, .016), spacing=S)
    reference = points.astype(float)
    lattice = np.round((points-points.min(0))/S).astype(int)
    nodes, weights, gradients = SA.stencil(points, DX)
    centre = reference.mean(0)+np.array([.0003, .0002, .0001])
    normals = {"n30_xz": [np.cos(np.radians(30)), 0, np.sin(np.radians(30))],
               "n45_xy": [np.cos(np.radians(45)), np.sin(np.radians(45)), 0],
               "n123": list(np.array([1., 2., 3.])/np.sqrt(14)), "axis_x": [1., 0, 0]}
    gap = .003
    out, checks = {}, {}
    for name, n in normals.items():
        n = np.asarray(n, float)
        d = (reference-centre)@n
        codes = np.where(d >= 0, 2, 1).astype(np.int64)    # packed: digit k=0 in bits 0-1
        log = synthetic_events(reference, lattice, codes, centre, n)
        positions = reference+np.where(d[:, None] >= 0, gap*n, 0.)   # rigid separation (geometry only)
        F = np.tile(np.eye(3), (len(points), 1, 1))
        run = {"reference": reference, "lattice": lattice, "positions": positions, "codes": codes, "F": F, "masses": masses.astype(float),
               "density": 1500., "nodes": nodes, "weights": weights, "gradients": gradients, "cuts": 1, "log": log}
        count, labels, _, _ = components(run)
        locus = SM.material_locus(reference, lattice, positions, codes, cuts=1, F=F, events=log)
        mid = SM.material_locus(reference, lattice, positions, codes, cuts=1, F=F, events=None)
        on_plane = np.abs((locus["P"]-centre)@n)
        stair = (mid["P"]-centre)@n
        # exact planar meshes of both faces inside the box (fine grid, clipped)
        u = SM._unit(np.cross(n, [0, 0, 1.] if abs(n[2]) < .9 else [1., 0, 0]))
        w = np.cross(n, u)
        g = np.arange(-.03, .03+1e-9, .0005)
        uu, ww = np.meshgrid(g, g, indexing="ij")
        grid = centre+uu.ravel()[:, None]*u+ww.ravel()[:, None]*w
        lo, hi = reference.min(0)+2.5*S, reference.max(0)-2.5*S
        m = len(g)
        i, j = np.meshgrid(np.arange(m-1), np.arange(m-1), indexing="ij")
        a0 = (i*m+j).ravel()
        tri = np.r_[np.c_[a0, a0+m, a0+m+1], np.c_[a0, a0+m+1, a0+1]]
        inside = np.all((grid >= lo) & (grid <= hi), 1)
        tri = tri[inside[tri].all(1)]
        v_exact = np.r_[grid, grid+gap*n]
        f_exact = np.r_[tri[:, ::-1], tri+len(grid)]   # side 1 face outward +n, side 2 face outward -n
        exact = SM.surface_report(reference, lattice, positions, codes, v_exact, f_exact, spacing=S, labels=labels, cuts=1,
                                  F=F, locus=locus, validity=False, core_margin_m=0., region_max_m=1.5*S)
        meshes = extract_surfaces(run, args.device)
        row = {"hermite_to_plane_max_m": float(on_plane.max()), "midpoint_staircase": SM._quantiles(np.abs(stair), S),
               "midpoint_staircase_rms_m": float(np.sqrt(np.mean(stair**2))),
               "expected_rms_0.29s_sum_abs_n_m": float(.2887*S*np.abs(n).sum()/np.sqrt((np.abs(n)**2).sum()*3)) ,
               "exact_mesh": {k: exact.get(k) for k in ("distance_to_side_locus_plane", "normal_error_deg_core", "region")},
               "mechanical_components": count, "links": int(len(log))}
        for profile, mm in meshes.items():
            for src, loc in (("hermite", locus), ("midpoint", mid)):
                rep = SM.surface_report(reference, lattice, positions, codes, mm["vertices"], mm["faces"], spacing=S, labels=labels,
                                        masses=masses, density=1500., cuts=1, F=F, locus=loc, validity=(src == "hermite"))
                row[f"{profile}_vs_{src}"] = {k: rep.get(k) for k in (
                    "distance_to_side_locus_plane", "distance_to_side_locus_plane_core", "signed_bias", "locus_to_mesh_distance",
                    "coverage", "normal_error_deg_core", "roughness_mesh_core", "roughness_locus_core", "rim", "volume",
                    "mesh_validity", "topology", "region")}
            # against the analytic plane itself (allowed here: G2 is the analytic unit test)
            cls = SM.classify_mesh_vertices(mm["vertices"], positions, labels, locus,
                                            SM.exterior_samples(lattice, positions, S), spacing=S)
            vv = mm["vertices"][cls["core"]]
            sd = (vv-centre)@n-np.where(cls["side"][cls["core"]] == 1, gap, 0.)
            row[f"{profile}_vs_analytic_plane_core"] = {"abs": SM._quantiles(np.abs(sd), S),
                                                        "mean_signed_along_n_side1_m": float(sd[cls["side"][cls["core"]] == 0].mean()),
                                                        "mean_signed_along_n_side2_m": float(sd[cls["side"][cls["core"]] == 1].mean())}
            row[f"{profile}_extract_wall_s"] = mm["wall_s"]
        if name == "n123":
            save_package(args.out/"G2_n123_package.npz", dict(run, contact_work=np.zeros(len(points)),
                         cut_progress=np.zeros(len(points)), blade_polygon=np.zeros((0, 2)), half_length=0.), labels, meshes, locus)
        out[name] = row
        checks[f"{name}_hermite_on_plane"] = row["hermite_to_plane_max_m"] <= 1e-9
        checks[f"{name}_exact_mesh_zero"] = (exact["distance_to_side_locus_plane"]["max"] or 0) <= 1e-9 and \
            (exact["normal_error_deg_core"]["max"] or 0) <= 1e-3
        checks[f"{name}_staircase_bounded"] = row["midpoint_staircase"]["max"] <= .5*S+1e-9
    result = {"case": "G2", "gap_m": gap, "plane_centre_m": centre.tolist(), "scope": "geometric unit test; synthetic codes "
              "and Hermite data; never injected into physics", **{"planes": out}, "checks": checks,
              "verdict": "PASS" if all(checks.values()) else "FAIL"}
    write(args, "G2", result)
    return result


def g3a(args, repeat):
    run = cut01r_run(args.device, seed=42)
    result, labels, meshes, locus = analyse(f"G3a_r{repeat}", run, args, declared_planes={0: ((0, 0, 0), (1, 0, 0))})
    result["control_steps"], result["armed_control_steps"] = run["control_steps"], run["armed_control_steps"]
    result.update(through_cut_checks(result))
    write(args, f"G3a_r{repeat}", result)
    return result, run


def through_cut_checks(result, expected=2):
    cv = result["surface_reports"]["current_volume_v1"]
    checks = {"components_expected": result["mechanical_components"] == expected,
              "logged_equals_conflicting": result["recorder_audit"]["all_equal"],
              "no_sign_inconsistent": result["recorder_audit"]["sign_inconsistent"] == 0,
              "substeps_match": result["recorder_audit"]["substep_matches_log"],
              "cv1_shells_equal_components": cv["topology"]["shells_equal_components"],
              "cv1_closed": cv["mesh_validity"]["watertight"]}
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def g3b(args, repeat):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.08, .03, top), spacing=S)
    yaw = 20.
    rot = Rotation.from_euler("z", yaw, degrees=True)*VERTICAL_CUT
    depth = top+.008
    phases = [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1)]
    run = probe_run(points, masses, phases, rotation=rot, start=[.0003, .0002, top+.004], device=args.device)
    normal = rot.apply([0, 0, 1.])
    result, *_ = analyse(f"G3b_r{repeat}", run, args, declared_planes={0: ((.0003, .0002, 0), normal)},
                         phase_names=["cut", "lift"])
    result["yaw_deg"] = yaw
    result.update(through_cut_checks(result))
    write(args, f"G3b_r{repeat}", result)
    return result


def g4(args):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.06, .024, top), spacing=S)
    phases = [Phase((0, 0, -CUT_SPEED), .010/CUT_SPEED, 0), Phase((0, 0, 0), .2, 0), Phase((0, 0, MOVE_SPEED), .4, -1)]
    start = [.0003, 0, top+.004]
    run = probe_run(points, masses, phases, rotation=VERTICAL_CUT, start=start, device=args.device)
    lowest_edge_z = start[2]-.010

    def extra(run, labels, meshes, locus):
        from robotic_cutting_sim import surface_attribution_v1 as SA
        ha = run["assign"]["hermite_current_at_assignment"]
        chains = run["trail"].chains()
        sv, sf, sn, _ = SA.edge_sweep_mesh(chains, 0, run["half_length"], footprint=ha)
        d = SA.signed_distance_to_mesh(ha, sv, sf, sn)
        P = locus["P"]
        dig = (run["codes"] & 3)
        return {"partial": {"lowest_edge_z_m": lowest_edge_z, "hermite_assign_min_z_m": float(ha[:, 2].min()) if len(ha) else None,
                            "hermite_reference_min_z_m": float(P[:, 2].min()) if len(P) else None,
                            "hermite_assign_to_sweep_abs_max_m": float(np.abs(d).max()) if len(d) else None,
                            "code0_particles_below_kerf_within_band": int(np.count_nonzero((dig == 0) & (np.abs(run["reference"][:, 0]-start[0]) < .024)
                                                                                          & (run["reference"][:, 2] < lowest_edge_z))),
                            "kerf_depth_from_top_m": float(top-P[:, 2].min()) if len(P) else None}}
    result, *_ = analyse("G4", run, args, declared_planes={0: ((start[0], 0, 0), (1, 0, 0))},
                         phase_names=["cut_half_depth", "hold_armed", "withdraw"], extra=extra)
    p = result["partial"]
    checks = {"one_mechanical_component": result["mechanical_components"] == 1,
              "one_lattice_component": result["lattice_components"] == 1,
              "logged_equals_conflicting": result["recorder_audit"]["all_equal"],
              "hermite_within_1s_of_sweep": p["hermite_assign_to_sweep_abs_max_m"] is not None and p["hermite_assign_to_sweep_abs_max_m"] <= S,
              "no_assignment_below_edge": p["hermite_assign_min_z_m"] is not None and p["hermite_assign_min_z_m"] >= lowest_edge_z-S}
    result.update(checks=checks, verdict="PASS" if all(checks.values()) else "FAIL")
    write(args, "G4", result)
    return result


def g5(args):
    from robotic_cutting_sim.blade_probe import Phase
    top, dome_h, dome_r, edge_above = .010, .004, .009, .002
    grid = np.mgrid[-.04:.04-1e-9:S, -.012:.012-1e-9:S, 0:top+dome_h-1e-9:S]
    pts = grid.reshape(3, -1).T+S/2
    dome = (pts[:, 2] > top) & ((pts[:, 2]-top) <= dome_h*(1-(pts[:, 0]**2+pts[:, 1]**2)/dome_r**2))
    keep = (pts[:, 2] < top) | dome
    points = pts[keep].astype(np.float32)
    masses = np.full(len(points), 1500*S**3, np.float32)
    rot = Rotation.from_euler("y", 25., degrees=True)
    x0, x1 = -.018, .026
    phases = [Phase((CUT_SPEED, 0, 0), (x1-x0)/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), .4, -1)]
    run = probe_run(points, masses, phases, rotation=rot, start=[x0, 0, top+edge_above], device=args.device)

    def extra(run, labels, meshes, locus):
        from robotic_cutting_sim import surface_metrics_v1 as SM
        above = run["reference"][:, 2] > top+edge_above
        chip_labels = np.unique(labels[above])
        rows = []
        for c in np.unique(labels):
            sel = labels == c
            rows.append({"component": int(c), "particles": int(sel.sum()), "mass_fraction": float(run["masses"][sel].sum()/run["masses"].sum()),
                         "reference_above_edge_fraction": float(above[sel].mean()),
                         "max_reference_thickness_m": float(np.ptp(run["reference"][sel, 2])+S)})
        cv = meshes["current_volume_v1"]
        skipped = cv["skipped"]
        return {"thin_chip": {"edge_height_above_substrate_m": edge_above, "dome_height_m": dome_h,
                              "max_chip_thickness_m": dome_h-edge_above, "reference_above_edge_particles": int(above.sum()),
                              "above_edge_in_largest_body": int(np.count_nonzero(above & (labels == 0))),
                              "components": rows[:12], "cv1_skipped": skipped,
                              "legacy_skipped": meshes["legacy"]["skipped"],
                              "every_component_meshed_or_recorded": {
                                  p: len(m["info"]["component_ranges"])+((m["skipped"] or {}).get("fragments", 0))
                                  == len(np.unique(labels)) for p, m in meshes.items()}}}
    result, *_ = analyse("G5", run, args, phase_names=["skim", "lift"], extra=extra)
    checks = {"logged_equals_conflicting": result["recorder_audit"]["all_equal"],
              "no_shell_spans_components": all(r["topology"]["shells_spanning_components"] == 0 for r in result["surface_reports"].values()),
              "skips_recorded": all(result["thin_chip"]["every_component_meshed_or_recorded"].values())}
    result.update(checks=checks, verdict="PASS" if all(checks.values()) else "FAIL")
    write(args, "G5", result)
    return result


def g6(args):
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, block
    from robotic_cutting_sim.sequence_blade_probe import OrientedPhase as P
    top = .012
    points, masses = block(size_m=(.06, .04, top), spacing=S)
    r0 = tuple(VERTICAL_CUT.as_quat())
    r1 = tuple((Rotation.from_euler("z", 90, degrees=True)*VERTICAL_CUT).as_quat())
    depth = top+.008
    down, up = depth/CUT_SPEED, depth/MOVE_SPEED
    phases = [P((0, 0, -CUT_SPEED), down, 0, r0), P((0, 0, MOVE_SPEED), up, -1, r0), P((.024/.5, 0, 0), .5, -1, r0),
              P((0, 0, -CUT_SPEED), down, 1, r0), P((0, 0, MOVE_SPEED), up, -1, r0), P((-.012/.5, .0043/.5, 0), .5, -1, r0),
              P((0, 0, 0), .05, -1, r1), P((0, 0, -CUT_SPEED), down, 2, r1), P((0, 0, MOVE_SPEED), up, -1, r1)]
    names = ["cut0", "lift0", "move0", "cut1", "lift1", "move1", "reorient", "cut2", "lift2"]
    attempts = {}
    for slots in (8, 16):   # 8 = tlmpm_v5 default; overflow is an error, never merged (ADR 0014): record and retry
        try:
            run = probe_run(points, masses, phases, start=[-.0117, 0, top+.004], cuts=3, device=args.device, sequence=True,
                            slots_per_node=slots)
            attempts[str(slots)] = "OK"
            break
        except Exception as error:
            attempts[str(slots)] = repr(error)
            print(f"[G6] slots_per_node {slots}: {error!r}", flush=True)
    planes = {0: ((-.0117, 0, 0), (1, 0, 0)), 1: ((.0123, 0, 0), (1, 0, 0)), 2: ((0, .0043, 0), Rotation.from_quat(r1).apply([0, 0, 1.]))}

    def extra(run, labels, meshes, locus):
        from robotic_cutting_sim.cut_topology import decode_digits
        d = decode_digits(run["codes"], 3)
        multi = {f"{a}{b}": int(np.count_nonzero((d[:, a] > 0) & (d[:, b] > 0))) for a, b in ((0, 2), (1, 2), (0, 1))}
        rows = {}
        pairs = locus["pairs"]
        for a, b in ((0, 2), (1, 2)):
            # links of cut b whose endpoints also carry digit a on both sides of cut a's plane: the junction
            ra = locus["cut"] == b
            both = (d[pairs[ra, 0], a] > 0) & (d[pairs[ra, 1], a] > 0)
            rows[f"cut{b}_links_with_cut{a}_digit"] = int(both.sum())
        return {"multi_cut": {"particles_with_two_digits": multi, **rows, "slots_per_node_attempts": attempts}}
    result, *_ = analyse("G6", run, args, declared_planes=planes, phase_names=names, extra=extra)
    checks = {"six_mechanical_components": result["mechanical_components"] == 6,
              "logged_equals_conflicting_each_cut": result["recorder_audit"]["all_equal"]}
    result.update(checks=checks, verdict="PASS" if all(checks.values()) else "FAIL")
    write(args, "G6", result)
    return result


def g7(args):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.06, .024, top), spacing=S)
    depth = top+.008
    phases = [Phase((0, 0, -CUT_SPEED), depth/CUT_SPEED, 0), Phase((0, 0, MOVE_SPEED), depth/MOVE_SPEED, -1),
              Phase((.033/.5, 0, 0), .5, -1), Phase((0, 0, -CUT_SPEED), .5, -1),
              Phase((-.010, 0, 0), .6, -1), Phase((0, 0, 0), .2, -1), Phase((CUT_SPEED, 0, 0), .4, -1),
              Phase((0, 0, MOVE_SPEED), .4, -1)]
    names = ["cut", "lift", "move", "lower", "push", "hold", "withdraw", "lift_end"]
    steps = np.cumsum([int(round(p.duration_s/DT)) for p in phases])
    push_start, hold_end = steps[3], steps[5]

    def want(rec):
        sub = int(rec.substep.numpy()[0])
        return sub == steps[0] or sub == steps[3] or (push_start < sub <= hold_end and (sub-push_start) % 400 == 0)
    run = probe_run(points, masses, phases, rotation=VERTICAL_CUT, start=[.0003, 0, top+.004], device=args.device,
                    snapshot=want)

    def extra(run, labels, meshes, locus):
        from robotic_cutting_sim.cutting_surface import CuttingSurface, CurrentMechanicalPartition
        from robotic_cutting_sim.cut_topology import mechanical_components_packed
        partition = CurrentMechanicalPartition(run["nodes"], run["weights"], run["gradients"], 1, encoding="packed2")
        surface = CuttingSurface(partition, S, S/2, args.device, profile="current_volume_v1", unmeshable="skip_presentation_v1")
        frames = []
        prev = None
        for snap in run["trail"].snapshots:
            count, lab = mechanical_components_packed(run["nodes"], run["weights"], snap["codes"], 1, gradients=run["gradients"])
            v, f, info = surface.extract(snap["positions"], snap["codes"], snap["F"])
            v, f = v.astype(float), f.astype(np.int64)
            topo = SM.topology_consistency(v, f, snap["positions"], lab,
                                           face_component=face_components(info, np.arange(count), len(f)))
            inter = SM.self_intersections(v, f, shell=SM.shells(v, f)[1])
            # closest particle gap between the two pieces
            gap = None
            if count == 2:
                from scipy.spatial import cKDTree
                d, _ = cKDTree(snap["positions"][lab == lab[0]]).query(snap["positions"][lab != lab[0]])
                gap = float(d.min())
            row = {"substep": snap["substep"], "time_s": snap["substep"]*DT, "components": int(count),
                   "labelled": int(np.count_nonzero(snap["codes"])), "mesh_shells": topo["mesh_shells"],
                   "shells_spanning_components": topo["shells_spanning_components"],
                   "faces_nearer_other_component": topo["faces_nearer_other_component"],
                   "inter_shell_intersections_sampled": inter["inter_shell_pairs_sampled"],
                   "self_intersections_sampled": inter["intersecting_pairs_sampled"],
                   "min_particle_distance_between_pieces_m": gap}
            if prev is not None and np.array_equal(prev[3], snap["codes"]):
                row["temporal_vs_previous"] = SM.temporal_stability(prev[0], prev[1], v, f, prev[2], snap["positions"], spacing=S)
            frames.append(row)
            prev = (v, f, snap["positions"], snap["codes"])
        return {"recontact": {"frames": frames, "push_speed_m_s": .010, "push_travel_m": .006}}
    result, *_ = analyse("G7", run, args, declared_planes={0: ((.0003, 0, 0), (1, 0, 0))}, phase_names=names, extra=extra)
    ph = result["phases"]
    frames = result["recontact"]["frames"]
    after_cut = frames[0]["components"] if frames else None
    checks = {"disarmed_phases_no_digits_no_events": all(ph[n]["newly_assigned"] == 0 and ph[n]["links_logged"] == 0
                                                         for n in names[1:]),
              "components_unchanged_after_cut": after_cut == result["mechanical_components"] == 2,
              "no_shell_spans_components_any_frame": all(f["shells_spanning_components"] == 0 for f in frames)}
    result.update(checks=checks, verdict="PASS" if all(checks.values()) else "FAIL")
    write(args, "G7", result)
    return result


def g8(args):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block
    top = .012
    points, masses = block(size_m=(.04, .024, top), spacing=S)
    controls = {
        "approach_armed": (VERTICAL_CUT, [0, 0, top+.010], [Phase((0, 0, -CUT_SPEED), .45, 0), Phase((0, 0, 0), .2, 0)]),
        "flat_press_armed": (FLAT_SKIM, [0, 0, top+.002], [Phase((0, 0, -.01), .4, 0), Phase((0, 0, 0), .2, 0),
                                                           Phase((0, 0, .01), .2, 0)]),
        "side_push_armed": (VERTICAL_CUT, [.025, 0, .006], [Phase((-.01, 0, 0), .7, 0), Phase((0, 0, 0), .1, 0)]),
        "slide_disarmed": (FLAT_SKIM, [-.03, 0, top-.0005], [Phase((CUT_SPEED, 0, 0), 1.25, -1)]),
        "plunge_disarmed": (VERTICAL_CUT, [.0003, 0, top+.004], [Phase((0, 0, -CUT_SPEED), .5, -1), Phase((0, 0, MOVE_SPEED), .4, -1)]),
    }
    rows, checks = {}, {}
    for name, (rot, start, phases) in controls.items():
        run = probe_run(points, masses, phases, rotation=rot, start=start, device=args.device)
        count, labels, lat, _ = components(run)
        meshes = extract_surfaces(run, args.device)
        cv = meshes["current_volume_v1"]
        val = SM.mesh_validity_v1(cv["vertices"], cv["faces"])
        vol = SM.volume_report(cv["vertices"], cv["faces"], run["positions"], labels, run["masses"]/run["density"], run["F"])
        rows[name] = {"labelled": int(np.count_nonzero(run["codes"])), "events": int(len(run["log"])),
                      "unexpected_code_changes": int(run["metadata"]["unexpected_code_changes"]),
                      "mechanical_components": count, "lattice_components": lat, "cv1_shells": val["shells"],
                      "cv1_watertight": val["watertight"], "cv1_volume_error_current": vol["total_relative_error_current"],
                      "max_contact_work_j": float(np.max(run["contact_work"])), "particles_with_contact_work": int(np.count_nonzero(run["contact_work"] > 0)),
                      "max_cut_progress": float(np.max(run["cut_progress"])),
                      "max_displacement_m": float(np.linalg.norm(run["positions"]-run["reference"], axis=1).max()),
                      "activation_segment_ok_substeps": int((run["activation"]["segment_ok_count"] > 0).sum()),
                      "activation_engaged_substeps": int((run["activation"]["stroke_engaged"] > 0).sum()),
                      "physics_wall_s": run["wall_s"]}
        checks[name] = (rows[name]["labelled"] == 0 and rows[name]["events"] == 0 and rows[name]["unexpected_code_changes"] == 0
                        and count == 1 and val["shells"] == 1 and val["watertight"])
        if name == "side_push_armed":
            save_package(args.out/"G8_side_push_package.npz", run, labels, meshes,
                         SM.material_locus(run["reference"], run["lattice"], run["positions"], run["codes"], cuts=1, F=run["F"]))
        print(f"[G8:{name}] {rows[name]}", flush=True)
    result = {"case": "G8", "controls": rows, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
    write(args, "G8", result)
    return result


# ----------------------------------------------------------------------------------------- Phase 0 baselines

def baselines(args):
    """Midpoint-fallback attribution of the eight Phase 0 baseline packages (no recorder in those runs)."""
    from robotic_cutting_sim import surface_attribution_v1 as SA
    from robotic_cutting_sim import surface_metrics_v1 as SM
    rows = json.loads((ROOT/"artifacts/wp12/baseline/scenes_002.json").read_text())
    out = {}
    for entry in rows:
        folder = ROOT/entry["run_dir"]/entry["extract"]
        pkg = np.load(folder/"baseline_compact.npz")
        sid = str(pkg["scene_id"])
        t0 = time.perf_counter()
        ref, pos = pkg["reference_positions_local"].astype(float), pkg["final_positions_local"].astype(float)
        lattice, codes, F = pkg["lattice"], pkg["final_codes"], pkg["final_F"].astype(float)
        cuts = int(pkg["max_cuts"])
        enc = str(pkg["code_encoding"])
        if enc != "packed2":
            from robotic_cutting_sim.cut_topology import ternary_to_packed
            codes = ternary_to_packed(codes, cuts)
        nodes, weights, gradients = SA.stencil(ref.astype(np.float32), DX)
        run = {"reference": ref, "lattice": lattice, "positions": pos, "codes": codes, "F": F,
               "masses": pkg["mass_kg"].astype(float), "density": float(pkg["mass_kg"][0])/S**3, "nodes": nodes, "weights": weights,
               "gradients": gradients, "cuts": cuts}
        count, labels, lat, fr = components(run)
        recorded = int(pkg["final_component_labels"].max()+1)
        locus = SM.material_locus(ref, lattice, pos, codes, cuts=cuts, F=F)
        meshes = {"current_volume_v1": {"vertices": pkg["final_surface_vertices_local"].astype(float),
                                        "faces": pkg["final_surface_faces"].astype(np.int64), "wall_s": None, "skipped": None}}
        try:
            meshes["legacy"] = extract_surfaces(run, args.device, profiles=("legacy",))["legacy"]
        except Exception as error:  # recorded, not hidden
            meshes["legacy"] = {"error": repr(error)}
        reports = {}
        for profile, m in meshes.items():
            if "error" in m:
                reports[profile] = {"status": "ERROR", "error": m["error"]}
                continue
            reports[profile] = SM.surface_report(ref, lattice, pos, codes, m["vertices"], m["faces"], spacing=S, labels=labels,
                                                 masses=run["masses"], density=run["density"], cuts=cuts, F=F, locus=locus)
        attribution = {}
        for k in range(cuts):
            a = SA.attribute_cut(k, locus=locus, lattice=lattice, reference=ref, positions=pos, labels=labels, spacing=S,
                                 mesh_reports=reports, midpoint_locus=locus)
            # total deformation reference -> final for the midpoint locus (pre + post, inseparable without a recorder)
            rows_k = np.flatnonzero(locus["cut"] == k)
            if len(rows_k):
                pairs = locus["pairs"][rows_k]
                core = SA.core_link_mask(lattice, pairs)
                res = []
                for sd in (0, 1):
                    for c in np.unique(labels[pairs[:, sd]]):
                        g = np.flatnonzero(core & (labels[pairs[:, sd]] == c))
                        if len(g) < 4:
                            continue
                        Rm, t = SA.kabsch(locus["P"][rows_k][g], locus["side_points"][rows_k][g, sd])
                        r = locus["side_points"][rows_k][g, sd]-(locus["P"][rows_k][g]@Rm.T+t)
                        res.append(np.einsum("ij,ij->i", r, locus["side_normals"][rows_k][g, sd]))
                a["total_deformation_normal_residual_core"] = SA._signed_summary(np.concatenate(res) if res else [], S)
            attribution[str(k)] = a
        topo = SA.topology_layer(nodes, weights, gradients, codes, cuts, lattice=lattice, labels=labels, masses=run["masses"],
                                 reference=ref)
        out[sid] = {"package": str(folder.relative_to(ROOT)), "components": count, "recorded_components": recorded,
                    "lattice_components": lat, "mass_fractions": fr[:8].tolist(), "surface_reports": reports,
                    "attribution": attribution, "topology_layer": topo, "wall_s": time.perf_counter()-t0,
                    "locus": "link_midpoint_fallback (Phase 0 runs had no recorder)"}
        print(f"[baseline {sid}] components {count} (recorded {recorded}) {time.perf_counter()-t0:.1f} s", flush=True)
    (args.out/"baselines.json").write_text(json.dumps(jsonable(out), indent=1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tests", default="G1,G2,G3a,G3b,G4,G5,G6,G7,G8")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--protocol-copy", type=Path, default=None)
    args = p.parse_args()
    args.out = args.out if args.out.is_absolute() else ROOT/args.out
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    import warp as wp
    import newton
    protocol = {**PROTOCOL, "command": " ".join([sys.executable, *sys.argv]), "git": git(), "device": args.device,
                "device_name": wp.get_device(args.device).name, "warp": wp.__version__, "newton": newton.__version__,
                "python": platform.python_version(), "tests": args.tests.split(","), "repeats": args.repeats,
                "baseline": args.baseline, "written_before_run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (args.out/"protocol.json").write_text(json.dumps(jsonable(protocol), indent=1))
    if args.protocol_copy is not None and not args.protocol_copy.exists():
        args.protocol_copy.write_text(json.dumps(jsonable(PROTOCOL), indent=1))
    summary, start = {}, time.perf_counter()
    for test in args.tests.split(","):
        t = time.perf_counter()
        try:
            if test == "G1":
                r = g1(args)
            elif test == "G2":
                r = g2(args)
            elif test == "G3a":
                r = [g3a(args, k)[0] for k in range(args.repeats)]
            elif test == "G3b":
                r = [g3b(args, k) for k in range(args.repeats)]
            elif test == "G4":
                r = g4(args)
            elif test == "G5":
                r = g5(args)
            elif test == "G6":
                r = g6(args)
            elif test == "G7":
                r = g7(args)
            elif test == "G8":
                r = g8(args)
            else:
                raise SystemExit(f"unknown test {test}")
            rs = r if isinstance(r, list) else [r]
            summary[test] = {"verdicts": [x.get("verdict") for x in rs], "checks": [x.get("checks") for x in rs],
                             "wall_s": time.perf_counter()-t}
        except Exception as error:   # keep going; the failure is recorded, not hidden
            import traceback
            traceback.print_exc()
            summary[test] = {"verdicts": ["ERROR"], "error": repr(error), "wall_s": time.perf_counter()-t}
        (args.out/"summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    if args.baseline:
        t = time.perf_counter()
        try:
            baselines(args)
            summary["baselines"] = {"status": "MEASURED", "wall_s": time.perf_counter()-t}
        except Exception as error:
            import traceback
            traceback.print_exc()
            summary["baselines"] = {"status": "ERROR", "error": repr(error)}
    summary["total_wall_s"] = time.perf_counter()-start
    (args.out/"summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    print(json.dumps(jsonable({k: v.get("verdicts", v.get("status")) for k, v in summary.items() if isinstance(v, dict)}), indent=1))


if __name__ == "__main__":
    main()
