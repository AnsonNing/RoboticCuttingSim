"""WP-12 Phase 3: cut_aware_surface_v1 smoke run on recorded CPU-readable states (no thresholds, no verdicts).

Two inputs, both 96 000 particles at 1 mm (CUT-01-R, tlmpm_v5, law 8):
  A. Phase 0 baseline S1 M1a (``baseline_compact.npz``): lattice, reference points, final codes, final
     positions and total F; no Hermite log -> labelled R4 midpoint fallback, F embedding.
  B. ``cut_events_001`` run1 (recorder on): final positions, codes and the cut_events_v1 log; the lattice is
     regenerated from the task sampler and checked against the recorder's lattice/reference SHA256; F was not
     saved -> LSQ embedding.
Reports mesh size, shells vs mechanical components, Hermite/QEF/embedding counts and timings (full rebuild, an
unchanged-codes incremental frame). Never overwrites its output directory. CPU only.

    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_cut_aware_smoke.py \
        --out artifacts/wp12/cut_aware_smoke_001
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

BASELINE = ROOT/"artifacts/wp12/baseline/s1_cut01r_m1a_law8_v5_002/wp12_extract_003/baseline_compact.npz"
EVENTS = ROOT/"artifacts/wp12/cut_events_001"
DX = .004


def summarize(info, vertices, faces, wall):
    audit = info["audit"]
    return {"wall_s": wall, "vertices": int(len(vertices)), "triangles": int(len(faces)),
            "cut_triangles": info["faces"]["cut_triangles"], "boundary_triangles": info["faces"]["boundary_triangles"],
            "mechanical_components": audit["mechanical_components"], "lattice_pieces": audit["lattice_pieces"],
            "mesh_shells": audit["mesh_shells"], "audit_ok": audit["ok"], "audit_errors": audit["errors"],
            "visually_separated_mechanically_bridged": len(audit["visually_separated_mechanically_bridged"]),
            "components": audit["components"], "hermite": info["hermite"], "qef": info["qef"],
            "regions": info["regions"], "embedding": info["embedding"],
            "mesh_validity_reference": info["mesh_validity_reference"], "update": info["update"],
            "timings_s": info["timings_s"]}


def run(surface, positions, codes, F, log):
    out = {}
    start = time.perf_counter()
    v, f, info = surface.extract(positions, codes, F, cut_events_log=log)
    out["full"] = summarize(info, v, f, time.perf_counter()-start)
    start = time.perf_counter()
    v2, f2, info2 = surface.extract(positions, codes, F, cut_events_log=log)
    out["unchanged_frame"] = {"wall_s": time.perf_counter()-start, "update": info2["update"],
                              "timings_s": info2["timings_s"], "identical": bool(np.array_equal(v, v2)
                                                                                 and np.array_equal(f, f2))}
    normals_start = time.perf_counter()
    surface.face_varying_normals(v, f)
    out["face_varying_normals_s"] = time.perf_counter()-normals_start
    return out, v, f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/wp12/cut_aware_smoke_001")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    from robotic_cutting_sim import cut_aware_surface as cas
    from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition
    report = {"experiment": "wp12_phase3_cut_aware_smoke", "profile": cas.PROFILE,
              "command": " ".join([sys.executable, *sys.argv]), "cwd": str(ROOT),
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
              "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
              "python": platform.python_version(), "numpy": np.__version__, "device": "cpu",
              "scope": "smoke only: sizes, counts and timings; no Phase 3 threshold is applied"}

    # A: baseline compact (F, no log)
    data = np.load(BASELINE)
    reference, lattice = data["reference_positions_local"].astype(np.float64), data["lattice"]
    partition = CurrentMechanicalPartition(*cas.reference_stencil(data["reference_positions_local"], DX),
                                           int(data["max_cuts"]), encoding=str(data["code_encoding"]))
    start = time.perf_counter()
    surface = cas.CutAwareSurface(partition, .001, lattice, reference)
    setup = time.perf_counter()-start
    result, v, f = run(surface, data["final_positions_local"], data["final_codes"], data["final_F"], None)
    stored = data["final_component_labels"]
    result.update(input=str(BASELINE.relative_to(ROOT)), particles=int(len(reference)), setup_s=setup,
                  stored_final_components=int(stored.max()+1),
                  recomputed_components_equal_stored=bool(np.array_equal(
                      surface.partition.partition(data["final_codes"])[1], stored)))
    np.savez_compressed(args.out/"A_baseline_mesh.npz", vertices=v.astype(np.float32), faces=f,
                        face_kind=surface.last["face_kind"], vertex_component=surface.last["vertex_component"])
    report["A_baseline_s1_m1a_midpoint_F"] = result
    print("A:", json.dumps({k: result["full"][k] for k in ("wall_s", "vertices", "triangles", "mesh_shells",
                                                           "mechanical_components")}), flush=True)

    # B: cut_events_001 run1 (log, no F)
    from robotic_cutting_sim.tasks import make_task
    meta = json.loads((EVENTS/"run1_on_cut_events_metadata.json").read_text())
    task = make_task("CUT-01-R", seed=42, particle_spacing_m=.001)
    points, masses, lattice = task.sample_particles()
    lattice_ok = cas._sha(lattice.astype(np.int32)) == meta["lattice_sha256"]
    reference_ok = cas._sha(np.asarray(points, np.float32)) == meta["reference_sha256"]
    state = np.load(EVENTS/"run1_on_final_state.npz")
    log = np.load(EVENTS/"run1_on_cut_events_log.npy")
    partition = CurrentMechanicalPartition(*cas.reference_stencil(points, DX), meta["max_cuts"], encoding="packed2")
    start = time.perf_counter()
    surface = cas.CutAwareSurface(partition, .001, lattice, np.asarray(points, np.float64))
    setup = time.perf_counter()-start
    result, v, f = run(surface, state["positions"], state["codes"], None, log)
    result.update(input=str(EVENTS.relative_to(ROOT)), particles=int(len(points)), setup_s=setup,
                  lattice_sha256_matches_recorder=lattice_ok, reference_sha256_matches_recorder=reference_ok,
                  log_sha256_matches_recorder=result["full"]["hermite"]["log"]["log_sha256"] == meta["log_sha256"])
    # same state with the midpoint fallback (labelled), to see what the log changes
    fallback = cas.CutAwareSurface(partition, .001, lattice, np.asarray(points, np.float64))
    vm, fm, infom = fallback.extract(state["positions"], state["codes"], None)
    cut_r = np.unique(f[surface.last["face_kind"] == cas.LINK_CUT])
    cut_m = np.unique(fm[fallback.last["face_kind"] == cas.LINK_CUT])
    ref_r, ref_m = surface.last["vertex_reference"][cut_r], fallback.last["vertex_reference"][cut_m]
    result["reference_cut_face_x_spread_m"] = {
        "recorder": {"p5": float(np.percentile(ref_r[:, 0], 5)), "p50": float(np.median(ref_r[:, 0])),
                     "p95": float(np.percentile(ref_r[:, 0], 95))},
        "midpoint_fallback": {"p5": float(np.percentile(ref_m[:, 0], 5)), "p50": float(np.median(ref_m[:, 0])),
                              "p95": float(np.percentile(ref_m[:, 0], 95))},
        "note": "descriptive only (scripted edge at workpiece-local x = 0); not a metric or gate"}
    np.savez_compressed(args.out/"B_cut_events_mesh.npz", vertices=v.astype(np.float32), faces=f,
                        face_kind=surface.last["face_kind"], vertex_component=surface.last["vertex_component"])
    report["B_cut_events_001_run1_recorder_lsq"] = result
    print("B:", json.dumps({k: result["full"][k] for k in ("wall_s", "vertices", "triangles", "mesh_shells",
                                                           "mechanical_components")}), flush=True)
    (args.out/"summary.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {args.out/'summary.json'}")


if __name__ == "__main__":
    main()
