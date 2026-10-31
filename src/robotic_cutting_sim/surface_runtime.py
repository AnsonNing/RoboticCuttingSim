"""Live surface-profile selection and per-frame surface provenance (WP-12 Phase 3, Layer 6).

``surface_profile`` (opt-in, versioned) chooses what the live runtimes draw for the specimen:

* ``particle_surface_v1`` (default): the existing ``cutting_surface.ComponentSurface`` (Newton ParticleSurface per
  current mechanical component; kernel ``current_volume_v1`` for configured/registered tasks, ``legacy`` for D0).
  Selecting it, or not selecting anything, leaves every output and receipt exactly as before.
* ``cut_aware_surface_v1``: ``cut_aware_surface.CutAwareSurface`` (reference-lattice linked volume, local dual
  contouring, cut-side-aware embedding). It needs ``tlmpm_v5`` and turns the read-only ``cut_events_v1`` recorder on
  (recorder-quality Hermite data); the selection, the recorder state and per-frame provenance are recorded.

The surface is one-way render geometry. It reads the solver state and writes nothing back, it never feeds back into
physics, and it is not the contact surface (contact uses particles and the PhysX colliders). Provenance rows are
oracle/debug data (codes hashes, audit, mesh): keep them out of policy observations and step info.

``LiveCutAwareSurface`` is created once per material reset (never shared across episodes, so no topology cache can
outlive its episode). The solver works in the workpiece-local frame; vertices are mapped to WORLD with the task's
rigid workpiece pose and the face-varying normals are computed from the WORLD vertices (identical to rotating
them; ``tests/test_surface_runtime.py`` also checks the equivalent ``frame_rotation`` route of the extractor).
"""
import hashlib
from pathlib import Path
import time

import numpy as np

PARTICLE_SURFACE = "particle_surface_v1"
CUT_AWARE = "cut_aware_surface_v1"
SURFACE_PROFILES = (PARTICLE_SURFACE, CUT_AWARE)
DEFAULT_SURFACE_PROFILE = PARTICLE_SURFACE
PROVENANCE_VERSION = "surface_provenance_v1"
RUNTIME_VERSION = "surface_runtime_v1"
# The live options of cut_aware_surface_v1. They equal the CutAwareSurface defaults that the Phase 3 S-tests measure,
# except strict_audit: a live episode records audit errors per frame instead of aborting the physics run.
LIVE_CUT_AWARE_OPTIONS = {"boundary_profile": "cell_face_v1", "fallback_normal_radius_spacings": 3.,
                          "qef_truncation": .1, "incremental": True, "strict_audit": False, "crease_angle_deg": 30.}
_SOURCES = ("cut_aware_surface.py", "cut_events.py", "surface_runtime.py", "usd_surface_bridge.py",
            "cutting_surface.py", "cut_topology.py")


RECORDER_MARGIN_STEPS = 1000   # control steps beyond the task duration (return home, holds) the recorder can log


def recorder_summary_capacity(task, material_dt_s):
    """cut_events_v1 activation-summary rows for one live episode: (task steps + RECORDER_MARGIN_STEPS) x substeps,
    at least the recorder default 2^18. The recorder raises on overflow (never truncates), so a live episode sizes
    it from its own duration (Native surface_live_004: Test B needed 370k substeps > 2^18)."""
    steps = int(round(task.duration_s/task.control_dt_s))+RECORDER_MARGIN_STEPS
    substeps = int(round(task.control_dt_s/material_dt_s))
    return max(1 << 18, steps*substeps)


def check_surface_profile(profile):
    if profile not in SURFACE_PROFILES:
        raise ValueError(f"surface_profile must be one of {SURFACE_PROFILES}")
    return profile


def source_sha256():
    """SHA256 of the surface implementation sources (recorded with every cut-aware selection)."""
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in _SOURCES}


def _sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def gpu_process_memory_mib(pid=None):
    """Used GPU memory (MiB) of this process per GPU UUID from nvidia-smi, or None when unavailable."""
    import os
    import subprocess
    pid = os.getpid() if pid is None else int(pid)
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = [line.split(",") for line in out.strip().splitlines() if line.strip()]
    return {uuid.strip(): float(memory) for p, uuid, memory in rows if p.strip() == str(pid)} or None


def _stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return None
    return {"n": int(len(values)), "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()), "sum": float(values.sum())}


def timing_summary(rows, key="timings_ms"):
    keys = sorted({k for row in rows for k in (row.get(key) or {})})
    return {k: _stats([(row.get(key) or {}).get(k) for row in rows]) for k in keys}


class LiveCutAwareSurface:
    """``cut_aware_surface_v1`` bound to one reset ``LiveCuttingMaterial`` (one episode).

    ``extract(time_s, step_index)`` returns WORLD vertices (float32), triangles (int32), face-varying WORLD normals
    (float32, 3 per triangle) and the extractor info (CuttingSurface-compatible ``component_ranges``,
    ``partition_rule``, ``codes_sha256``, ``stencil_sha256``). It appends one provenance row per extraction to
    ``self.rows``; ``note_upload`` adds the USD upload time and the bridge hashes to the latest row."""

    profile = CUT_AWARE

    def __init__(self, material, *, to_world=None, density_kg_m3=None, **options):
        from .cut_aware_surface import CutAwareSurface
        from .cutting_surface import CurrentMechanicalPartition
        bed = material.bed
        if bed is None:
            raise RuntimeError("Reset the material before building its surface")
        if getattr(material, "code_encoding", "ternary") != "packed2":
            raise ValueError("cut_aware_surface_v1 runs on tlmpm_v5 (packed cut codes) in the live runtimes")
        unknown = set(options)-set(LIVE_CUT_AWARE_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown cut_aware_surface_v1 options {sorted(unknown)}")
        self.options = {**LIVE_CUT_AWARE_OPTIONS, **options}
        self.material, self.bed, self.to_world = material, bed, to_world
        self.recorder = getattr(material, "cut_event_recorder", None)
        start = time.perf_counter()
        partition = CurrentMechanicalPartition(bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy(),
                                               bed.config.cuts, encoding=material.code_encoding)
        crease = self.options["crease_angle_deg"]
        surface_options = {k: v for k, v in self.options.items() if k != "crease_angle_deg"}
        self.surface = CutAwareSurface(partition, material.spacing, material.lattice, bed.reference.numpy(),
                                       crease_angle_deg=crease, masses_kg=material.masses,
                                       density_kg_m3=density_kg_m3, **surface_options)
        self.build_wall_s = time.perf_counter()-start
        self.rows = []
        self._previous = None
        self.last_info = None

    def selection(self):
        from .cut_aware_surface import surface_selection_record
        record = surface_selection_record(CUT_AWARE, **{k: v for k, v in self.options.items()})
        recorder = self.recorder
        return {**record, "runtime_version": RUNTIME_VERSION, "provenance_version": PROVENANCE_VERSION,
                "cut_events": "cut_events_v1" if recorder is not None else None,
                "hermite_source_expected": "cut_events_v1" if recorder is not None else "midpoint_fallback_R4",
                "bridge_method": "DynamicSurfaceBridge.update_face_varying_v1 (faceVarying normals, doubleSided)",
                "frame": "solver workpiece-local extraction, rigid workpiece pose to WORLD, normals from WORLD vertices",
                "writes_physics_state": False, "is_contact_surface": False,
                "reference_grid": {"spacing_m": float(self.material.spacing), "sites": int(self.surface.lattice.n),
                                   "links": int(self.surface.lattice.n_links),
                                   "dual_cells": int(self.surface.lattice.n_cells),
                                   **self.surface.lattice.sha256()},
                "static_build_wall_s": float(self.build_wall_s), "source_sha256": source_sha256()}

    def extract(self, *, time_s, step_index):
        from .cut_aware_surface import face_varying_normals
        bed, timings = self.bed, {}
        start = time.perf_counter()
        log = None
        if self.recorder is not None:
            log = self.recorder.log()
        timings["cut_events_log_fetch"] = 1e3*(time.perf_counter()-start)
        t0 = time.perf_counter()
        positions, codes, F = bed.x.numpy(), bed.code.numpy(), bed.F.numpy()
        timings["state_readback"] = 1e3*(time.perf_counter()-t0)
        t0 = time.perf_counter()
        vertices, faces, info = self.surface.extract(positions, codes, F, cut_events_log=log)
        timings["extract_total"] = 1e3*(time.perf_counter()-t0)
        parts = 0.
        for key, value in info["timings_s"].items():
            if key != "total_s":
                timings[key[:-2]] = 1e3*value
                parts += 1e3*value
        # the remainder of extract(): mesh_validity on the reference mesh, component grouping and metadata
        timings["validity_and_metadata"] = max(0., timings["extract_total"]-parts)
        t0 = time.perf_counter()
        world = vertices if self.to_world is None else np.asarray(self.to_world(vertices), np.float64)
        timings["world_transform"] = 1e3*(time.perf_counter()-t0)
        t0 = time.perf_counter()
        normals = face_varying_normals(world, faces, self.options["crease_angle_deg"]) if len(faces) else np.zeros((0, 3))
        normals, degenerate = self._repair_degenerate_normals(normals, faces)
        timings["normals"] = 1e3*(time.perf_counter()-t0)
        temporal = self._temporal(info["codes_sha256"], faces, world, positions)
        timings["surface_total"] = 1e3*(time.perf_counter()-start)
        h, audit, validity = info["hermite"], info["audit"], info["mesh_validity_reference"]
        row = {"version": PROVENANCE_VERSION, "extraction": len(self.rows), "time_s": float(time_s),
               "step_index": int(step_index), "codes_sha256": info["codes_sha256"],
               "cut_events": None if log is None else {
                   "entries": int(len(log)), "log_sha256": h["log"]["log_sha256"],
                   "max_substep": int(log["substep"].max()) if len(log) else -1},
               "hermite": {"source": h["source"], "crossings_by_source": dict(h["crossings_by_source"]),
                           "cut_links": h["cut_links"], "multi_crossing_links": h["multi_crossing_links"],
                           "near_coincident_crossings": h["near_coincident_crossings"],
                           "fallback_normal_incoherent_links": h["fallback_normal_incoherent_links"],
                           "log_entries_not_links": h["log"]["entries_not_links"],
                           "log_entries_not_conflicting": h["log"]["entries_not_conflicting"],
                           "log_duplicate_entries": h["log"]["duplicate_entries"]},
               "qef": {k: info["qef"][k] for k in ("representatives", "clamped", "rank_counts",
                                                   "representatives_without_hermite")},
               "embedding": dict(info["embedding"]), "regions": dict(info["regions"]),
               "audit": {"ok": bool(audit["ok"]), "errors": list(audit["errors"]),
                         "mechanical_components": audit["mechanical_components"],
                         "lattice_pieces": audit["lattice_pieces"], "mesh_shells": audit["mesh_shells"],
                         "piece_component_violations": audit["piece_component_violations"],
                         "shell_component_violations": audit["shell_component_violations"],
                         "visually_separated_mechanically_bridged": len(audit["visually_separated_mechanically_bridged"])},
               "mesh": {"vertices": int(len(world)), "triangles": int(len(faces)),
                        "cut_triangles": info["faces"]["cut_triangles"],
                        "boundary_triangles": info["faces"]["boundary_triangles"],
                        **{k: validity[k] for k in ("boundary_edges", "non_manifold_edges",
                                                    "inconsistently_oriented_edges", "non_manifold_vertices",
                                                    "degenerate_triangles", "euler_characteristic")}},
               "update": dict(info["update"]), "temporal": temporal, "degenerate_normal_corners": degenerate,
               "timings_ms": timings}
        self.rows.append(row)
        info = {**info, "surface_provenance_row": len(self.rows)-1}
        self.last_info = info
        return (np.ascontiguousarray(world, np.float32), faces.astype(np.int32),
                np.ascontiguousarray(normals, np.float32), info)

    def _world_rotation(self):
        if self.to_world is None:
            return np.eye(3)
        basis = np.asarray(self.to_world(np.vstack([np.zeros(3), np.eye(3)])), np.float64)
        return (basis[1:]-basis[0]).T

    def _repair_degenerate_normals(self, normals, faces):
        """Zero-area triangles in CURRENT space (e.g. a collapsed slit-tip quad: both sides use the same tip
        representative, or a fully compressed cell) get no normal from ``face_varying_normals`` (a zero vector).
        They draw nothing, but USD needs unit normals: use the triangle's REFERENCE-space normal rotated to WORLD,
        else world +z. Counted per extraction (``degenerate_normal_corners``); the vertices are unchanged."""
        length = np.linalg.norm(normals, axis=1)
        bad = ~(np.abs(length-1) <= 1e-6)
        out = {"corners": int(bad.sum()), "triangles": int(bad.reshape(-1, 3).any(1).sum()) if len(bad) else 0,
               "reference_normal_used": 0, "world_z_used": 0}
        if not bad.any():
            return normals, out
        normals = normals.copy()
        tri = np.flatnonzero(bad.reshape(-1, 3).any(1))
        reference = self.surface.last["vertex_reference"][faces[tri]]
        n = np.cross(reference[:, 1]-reference[:, 0], reference[:, 2]-reference[:, 0])@self._world_rotation().T
        nl = np.linalg.norm(n, axis=1)
        ok = nl > 1e-30
        fallback = np.where(ok[:, None], n/np.where(ok, nl, 1.)[:, None], np.array([0., 0., 1.]))
        corner_tri = np.repeat(np.arange(len(faces)), 3)
        lookup = np.full(len(faces), -1)
        lookup[tri] = np.arange(len(tri))
        rows = np.flatnonzero(bad)
        normals[rows] = fallback[lookup[corner_tri[rows]]]
        out.update(reference_normal_used=int(ok[lookup[corner_tri[rows]]].sum()),
                   world_z_used=int((~ok[lookup[corner_tri[rows]]]).sum()))
        return normals, out

    def _temporal(self, codes_sha, faces, world, positions):
        """Frame-to-frame vertex motion at unchanged codes (the flicker statistic) vs the particle motion."""
        previous = self._previous
        self._previous = {"codes_sha256": codes_sha, "faces_sha256": _sha(faces), "vertices": world.copy(),
                          "positions": np.asarray(positions, np.float64).copy()}
        if previous is None:
            return {"previous": False}
        codes_same = previous["codes_sha256"] == codes_sha
        topology_same = previous["faces_sha256"] == self._previous["faces_sha256"]
        out = {"previous": True, "codes_unchanged": bool(codes_same), "topology_unchanged": bool(topology_same)}
        particle = np.linalg.norm(self._previous["positions"]-previous["positions"], axis=1)
        out["particle_step_max_m"] = float(particle.max()) if len(particle) else 0.
        if topology_same and len(world) == len(previous["vertices"]):
            step = np.linalg.norm(world-previous["vertices"], axis=1)
            out.update(vertex_step_max_m=float(step.max()) if len(step) else 0.,
                       vertex_step_p95_m=float(np.percentile(step, 95)) if len(step) else 0.)
        return out

    def note_upload(self, receipt, upload_ms):
        if self.rows:
            row = self.rows[-1]
            row["timings_ms"]["usd_upload"] = float(upload_ms)
            row["timings_ms"]["surface_and_upload_total"] = row["timings_ms"]["surface_total"]+float(upload_ms)
            row["bridge"] = {k: receipt[k] for k in ("bridge_method", "normals_interpolation", "topology_sha256",
                                                     "surface_sha256", "normals_sha256") if k in receipt}

    def summary(self):
        rows = self.rows
        unchanged = [r["temporal"] for r in rows if r["temporal"].get("codes_unchanged")]
        steady = [t for t in unchanged if "vertex_step_max_m" in t]
        last = rows[-1] if rows else None
        return {"version": PROVENANCE_VERSION, "surface_profile": CUT_AWARE, "extractions": len(rows),
                "audit_error_extractions": int(sum(not r["audit"]["ok"] for r in rows)),
                "audit_errors_first": next((r["audit"]["errors"] for r in rows if not r["audit"]["ok"]), None),
                "max_mesh_shells": max((r["audit"]["mesh_shells"] for r in rows), default=None),
                "max_visually_separated_mechanically_bridged": max(
                    (r["audit"]["visually_separated_mechanically_bridged"] for r in rows), default=None),
                "max_shell_component_violations": max((r["audit"]["shell_component_violations"] for r in rows),
                                                      default=None),
                "mesh_size": {"vertices": _stats([r["mesh"]["vertices"] for r in rows]),
                              "triangles": _stats([r["mesh"]["triangles"] for r in rows]),
                              "cut_triangles": _stats([r["mesh"]["cut_triangles"] for r in rows])},
                "degenerate_normal_corners_max": max((r["degenerate_normal_corners"]["corners"] for r in rows),
                                                     default=None),
                "degenerate_normal_triangles_max": max((r["degenerate_normal_corners"]["triangles"] for r in rows),
                                                       default=None),
                "mesh_defects_max": {k: max((r["mesh"][k] for r in rows), default=None) for k in
                                     ("boundary_edges", "non_manifold_edges", "inconsistently_oriented_edges",
                                      "degenerate_triangles")},
                "update_modes": {m: int(sum(r["update"]["mode"] == m for r in rows)) for m in ("full", "incremental")},
                "dirty_cells": _stats([r["update"]["dirty_cells"] for r in rows if r["update"]["mode"] == "incremental"]),
                "temporal_unchanged_codes": {
                    "frames": len(unchanged), "topology_unchanged": int(sum(t["topology_unchanged"] for t in unchanged)),
                    "vertex_step_max_m": _stats([t["vertex_step_max_m"] for t in steady]),
                    "particle_step_max_m": _stats([t["particle_step_max_m"] for t in steady]),
                    "vertex_minus_particle_step_max_m": max(
                        (t["vertex_step_max_m"]-t["particle_step_max_m"] for t in steady), default=None),
                    "rule": "codes unchanged since the previous extraction; flicker would be vertex motion beyond the "
                            "material's own motion"},
                "timings_ms": timing_summary(rows),
                "final": None if last is None else {k: last[k] for k in ("time_s", "codes_sha256", "cut_events",
                                                                          "hermite", "audit", "mesh", "embedding")}}
