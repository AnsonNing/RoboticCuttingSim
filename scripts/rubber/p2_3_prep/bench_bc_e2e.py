"""P2-3 prerequisites (ADR 0023 §20.2): interleaved end-to-end A/B of the BC launch on workload E (GPU0).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
        PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_3_prep/bench_bc_e2e.py \
        --out artifacts/rl_ready/p2_3_prep/<id>

Two batches in one process, identical except ``bc_launch`` (``all_nodes_v1`` vs ``motion_table_v1``): workload E,
N = 4, v1 options (or ``--options all``: F8 + F9 + F10), guard 1.0, CUDA graph chunks, materials resampled with the
same seeds. Control steps alternate A, B, B, A, ... for ``--rounds`` rounds (each control step = 4,996 substeps of
4.0e-6 s at 50 Hz); median wall per control step, paired ratio per round. The clamp holds still (gravity sag): the
two batches' tip markers are compared at the end (not bitwise on a GPU: float atomics). No cameras, no recording.
SHARED_GPU whenever another process is present.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
ALL = {"dirichlet": "consistent_mass_projection_v1", "traction_work": "grid_displacement_v1",
       "state_precision": "f64_shadow_v1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--options", default="none", choices=("none", "all"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        sys.exit("GPU0 only (CUDA_VISIBLE_DEVICES=0)")
    import warp as wp
    wp.init()
    out.mkdir(parents=True)
    from gpu_smoke_rubber import gpu_sample
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    wl = json.loads((ROOT/"configs"/"rubber"/"workload_e_v1.json").read_text())["workloads"][0]["material"]
    g, rnd = wl["geometry_m"], wl["randomisation"]
    step_mat = MaterialValues(mu0=rnd["shear_modulus_eq_pa"][1], kappa=float(wl["bulk_ratio_beta"])
                              * rnd["shear_modulus_eq_pa"][1], density=rnd["density_kg_m3"][0])
    ranges = {"mu0": tuple(rnd["shear_modulus_eq_pa"]), "density": tuple(rnd["density_kg_m3"])}
    spec = rs.RubberStripSpec(length=g["length"], width=g["width"], thickness=g["thickness"], n_t=wl["n_t"], case="A",
                              resolution_profile=wl["resolution_profile"])
    options = ALL if args.options == "all" else {}
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": wp.get_device("cuda:0").name, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(), "warp": wp.config.version,
            "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    batches = {}
    for launch in ("all_nodes_v1", "motion_table_v1"):
        lives = [LiveRubberMaterial(spec, control_dt=wl["control_dt_s"], device="cpu", step_material=step_mat,
                                    stable_step_guard=1.0, **options) for _ in range(args.N)]
        b = BatchedRubberMaterial(lives, device="cuda:0", ranges=ranges, strict=False, bc_launch=launch)
        b.reset(seeds=np.arange(args.N)+100, rng_ranges=True)
        b.advance()                                       # warm (first replay)
        batches[launch] = b
    walls = {k: [] for k in batches}
    samples = []
    for r in range(args.rounds):
        order = ("all_nodes_v1", "motion_table_v1") if r % 2 == 0 else ("motion_table_v1", "all_nodes_v1")
        for k in order:
            wp.synchronize_device("cuda:0")
            t = time.perf_counter()
            batches[k].advance()
            wp.synchronize_device("cuda:0")
            walls[k].append(time.perf_counter()-t)
        if r % 3 == 0:
            samples.append(gpu_sample())
    a, b = np.array(walls["all_nodes_v1"]), np.array(walls["motion_table_v1"])
    tip = {k: v.markers()[:, -1, 2].tolist() for k, v in batches.items()}
    invalid = {k: v.invalid_reason.numpy().tolist() for k, v in batches.items()}
    sub = batches["all_nodes_v1"].solver.substeps
    result = {"N": args.N, "particles_per_env": int(batches["all_nodes_v1"].packed.particle_counts[0]),
              "options": options or "v1 defaults", "control_hz": 50, "substeps": sub,
              "material_dt_s": batches["all_nodes_v1"].solver.dt, "stable_step_guard": 1.0, "graph": "chunks of 500",
              "cameras": "none", "recording": "none", "warm": True, "rounds": args.rounds,
              "wall_s": {k: v for k, v in walls.items()},
              "median_ms_per_substep": {"all_nodes_v1": 1e3*float(np.median(a))/sub,
                                        "motion_table_v1": 1e3*float(np.median(b))/sub},
              "env_steps_per_s_median": {"all_nodes_v1": args.N/float(np.median(a)),
                                         "motion_table_v1": args.N/float(np.median(b))},
              "paired_ratio_table_over_all_nodes": {"median": float(np.median(b/a)), "min": float(np.min(b/a)),
                                                     "max": float(np.max(b/a))},
              "tip_marker_z": tip, "tip_max_abs_diff_m": float(np.max(np.abs(np.subtract(*tip.values())))),
              "invalid_reason": invalid}
    meta["co_tenant_samples"] = samples
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [x for s in [meta["co_tenant_before"], meta["co_tenant_after"]]+samples for x in s["apps"]
              if x.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    (out/"result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"sharing": meta["gpu_sharing"], **{k: result[k] for k in (
        "median_ms_per_substep", "env_steps_per_s_median", "paired_ratio_table_over_all_nodes", "tip_max_abs_diff_m",
        "invalid_reason")}}, indent=1, default=float))


if __name__ == "__main__":
    main()
