"""P2-3 prerequisites (ADR 0023 §20): GPU smoke of the batched F8 / F9 / F10 options and the constrained-node BC launch.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
        PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_3_prep/run_gpu_smoke_options.py \
        --out artifacts/rl_ready/p2_3_prep/<id>

Workload E (``configs/rubber/workload_e_v1.json``: profile rubber_rl_v1, n_t 3, case A clamp, 50 Hz, per-reset material
randomisation inside the declared ranges, dt from the range maxima), stable-step guard 1.0, N = 4, CUDA graph chunks.
Rows (back to back, same process; ``--rows``): ``default_all_nodes`` (v1 options, the P2-2 all-node BC launch), then
the BC variant under test with v1 options and with F8 + F9 + F10: ``list`` = ``constrained_list_v1`` (gpu_smoke_001),
``motion_table`` = ``motion_table_v1`` (default).
Per row: build, 1 cold + ``--warm`` control steps, masked resets with resampled material, L2 invariants (finite, mass
constant between resets, invalid bits 0, guard violations 0, env-ownership audits 0); for ``options_list`` also the
option invariants (F = float32(F64), x = float32(x64) bitwise; F8 MUSL momentum on the constrained (node, component)
rows minus m v_presc relative to max |p| below 1e-5). Then the kernel split: ``--profile-substeps`` eager substeps per
row under ``wp.timing_begin`` (GPU execution time per kernel). No cameras, no recording. Refuses to start with < 4 GB
free on the device. Co-tenant samples before / after; SHARED_GPU whenever another process is present.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
ALL = {"dirichlet": "consistent_mass_projection_v1", "traction_work": "grid_displacement_v1",
       "state_precision": "f64_shadow_v1"}
ROWS = {"list": (("default_all_nodes", {}, "all_nodes_v1"), ("default_list", {}, "constrained_list_v1"),
                 ("options_list", ALL, "constrained_list_v1")),                       # gpu_smoke_001
        "motion_table": (("default_all_nodes", {}, "all_nodes_v1"), ("default_motion_table", {}, "motion_table_v1"),
                         ("options_motion_table", ALL, "motion_table_v1"))}


def free_mb(index):
    q = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.total,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    total, used = (float(v) for v in q.split(","))
    return total-used


def option_invariants(batch):
    s = batch.solver
    out = {}
    if s.state_precision == "f64_shadow_v1":
        out["F_equals_f32_F64"] = bool(np.array_equal(s.F.numpy(), s.F64.numpy().astype(np.float32)))
        out["x_equals_f32_x64"] = bool(np.array_equal(s.x.numpy(), s.x64.numpy().astype(np.float32)))
    if s.projection:
        gp = s.gp.numpy().astype(np.float64)
        gm = s.gm.numpy().astype(np.float64)
        node, comp = s.proj_row_node.numpy(), s.proj_row_comp.numpy()
        res = gp[node, comp]-gm[node]*s.proj_vpresc.numpy()
        out["projection_residual_rel"] = float(np.abs(res).max()/np.abs(gp).max())
        out["projection_ok"] = out["projection_residual_rel"] < 1e-5
        out["proj_rows"] = int(s.proj_rows)
    if s.traction_work == "grid_displacement_v1":
        out["work_finite"] = bool(np.isfinite(s.work.numpy()).all())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--resets", type=int, default=4)
    ap.add_argument("--profile-substeps", type=int, default=100)
    ap.add_argument("--rows", default="motion_table", choices=sorted(ROWS))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    import warp as wp
    wp.init()
    ordinal = wp.get_device(args.device).ordinal
    physical = int(os.environ.get("CUDA_VISIBLE_DEVICES", str(ordinal)).split(",")[ordinal])
    if physical != 0:
        sys.exit("this smoke runs on GPU0 only")
    if free_mb(physical) < 4096:
        sys.exit(f"less than 4 GB free on GPU{physical}: {free_mb(physical):.0f} MB")
    out.mkdir(parents=True)
    from gpu_smoke_rubber import Peak, gpu_sample
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    cfg_path = ROOT/"configs"/"rubber"/"workload_e_v1.json"
    wl = json.loads(cfg_path.read_text())["workloads"][0]["material"]
    g, rnd = wl["geometry_m"], wl["randomisation"]
    beta = float(wl["bulk_ratio_beta"])
    step_mat = MaterialValues(mu0=rnd["shear_modulus_eq_pa"][1], kappa=beta*rnd["shear_modulus_eq_pa"][1],
                              density=rnd["density_kg_m3"][0])
    ranges = {"mu0": tuple(rnd["shear_modulus_eq_pa"]), "density": tuple(rnd["density_kg_m3"])}
    spec = rs.RubberStripSpec(length=g["length"], width=g["width"], thickness=g["thickness"], n_t=wl["n_t"], case="A",
                              resolution_profile=wl["resolution_profile"])
    guard = 1.0
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device, "physical_gpu": physical, "device_name": wp.get_device(args.device).name,
            "pid": os.getpid(), "lock": "/mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock (flock, blocking)",
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
            "dirty": subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip(),
            "workload_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "warp": wp.config.version,
            "free_mb_before": free_mb(physical), "co_tenant_before": gpu_sample(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    peak = Peak()
    rows, profiles, co_samples = [], {}, []

    def build(options, bc_launch):
        t = time.perf_counter()
        lives = [LiveRubberMaterial(spec, control_dt=wl["control_dt_s"], device="cpu", step_material=step_mat,
                                    stable_step_guard=guard, **options) for _ in range(args.N)]
        lives_s = time.perf_counter()-t
        batch = BatchedRubberMaterial(lives, device=args.device, ranges=ranges, strict=False, bc_launch=bc_launch)
        del lives
        t = time.perf_counter()
        batch.reset(seeds=np.arange(args.N)+100, rng_ranges=True)
        wp.synchronize_device(args.device)
        return batch, lives_s, time.perf_counter()-t

    for label, options, bc_launch in ROWS[args.rows]:
        wp.synchronize_device(args.device)
        t0 = time.perf_counter()
        batch, lives_s, first_reset_s = build(options, bc_launch)
        build_s = time.perf_counter()-t0
        peak.update()
        env_p = batch.packed.host["env_of_particle"]
        mass0 = np.bincount(env_p, weights=batch.solver.mass.numpy().astype(np.float64))
        walls = []
        for _ in range(1+args.warm):
            wp.synchronize_device(args.device)
            t = time.perf_counter()
            batch.advance()
            wp.synchronize_device(args.device)
            walls.append(time.perf_counter()-t)
            peak.update()
        co_samples.append(gpu_sample())
        diag = batch.diagnostics_read()
        mass1 = np.bincount(env_p, weights=batch.solver.mass.numpy().astype(np.float64))
        invalid = batch.invalid_reason.numpy().tolist()
        guard_v = batch.solver.cfl_violations.numpy().tolist()
        opt = option_invariants(batch)
        wrench = batch.wrench.numpy()
        resets = []
        for r in range(args.resets):
            wp.synchronize_device(args.device)
            t = time.perf_counter()
            batch.reset([r % args.N], seeds=[1000+r], rng_ranges=True)
            wp.synchronize_device(args.device)
            resets.append(time.perf_counter()-t)
        # one more control step after the resets: the reset envs (resampled mass, rebuilt F8 blocks) stay valid
        batch.advance()
        diag2 = batch.diagnostics_read()
        opt2 = option_invariants(batch)
        audits = batch.audits()
        invalid2 = batch.invalid_reason.numpy().tolist()
        ok = (all(not d["nonfinite"] for d in diag+diag2) and np.array_equal(mass0, mass1) and not any(invalid)
              and not any(invalid2) and not any(guard_v) and all(v == 0 for v in audits.values())
              and bool(np.isfinite(wrench).all()) and all(v for k, v in {**opt, **opt2}.items()
                                                           if isinstance(v, bool)))
        warm = float(np.median(walls[1:]))
        n_p = int(batch.packed.particle_counts[0])
        rows.append({"row": label, "options": batch.solver.options_receipt(), "N": args.N, "particles_per_env": n_p,
                     "control_hz": 50, "material_dt_s": batch.solver.dt, "substeps": batch.solver.substeps,
                     "n_t": wl["n_t"], "stable_step_guard": guard, "cameras": "none", "recording": "none",
                     "graph": batch.capture_receipt, "build_s": build_s, "lives_build_s": lives_s,
                     "first_reset_s": first_reset_s, "control_step_wall_s": walls, "cold_wall_s": walls[0],
                     "warm_wall_s_median": warm, "env_steps_per_s": args.N/warm,
                     "ms_per_substep": 1e3*warm/batch.solver.substeps,
                     "particle_substeps_per_s": args.N*n_p*batch.solver.substeps/warm,
                     "reset_s": resets, "reset_median_ms": 1e3*float(np.median(resets)),
                     "vram_peak_mb_process_cumulative": peak.value, "memory_bytes_arrays": batch.memory_bytes(),
                     "mempool_used_mb": wp.get_mempool_used_mem_current(args.device)/2**20,
                     "invalid_reason": invalid, "invalid_reason_after_resets": invalid2, "guard_violations": guard_v,
                     "audits": audits, "cfl_eff_max_end": max(d["cfl_eff_max"] for d in diag),
                     "option_invariants": opt, "option_invariants_after_resets": opt2,
                     "wrench_clamp_fz_N": wrench[:, 0, 2].tolist(), "tip_marker_z": batch.markers()[:, -1, 2].tolist(),
                     "physics_verdict": "L2_INVARIANTS_PASS" if ok else "L2_INVARIANTS_FAIL"})
        (out/"rows.json").write_text(json.dumps(rows, indent=1, default=float))
        print(json.dumps({k: rows[-1][k] for k in ("row", "env_steps_per_s", "ms_per_substep", "reset_median_ms",
                                                   "vram_peak_mb_process_cumulative", "physics_verdict")},
                         default=float), flush=True)
        # kernel split (eager substeps, GPU execution time per kernel)
        s = batch.solver
        wp.synchronize_device(args.device)
        wp.timing_begin()
        for _ in range(args.profile_substeps):
            s.step()
        wp.synchronize_device(args.device)
        per = defaultdict(float)
        for r in wp.timing_end():
            per[r.name] += r.elapsed
        total = sum(per.values())
        bc = sum(v for k, v in per.items() if "_bc_" in k)
        profiles[label] = {"N": args.N, "substeps": args.profile_substeps, "gpu_ms_total": total,
                           "gpu_ms_per_substep": total/args.profile_substeps, "bc_share": bc/total,
                           "bc_ms_per_substep": bc/args.profile_substeps,
                           "kernels_ms": dict(sorted(per.items(), key=lambda kv: -kv[1]))}
        (out/"profile.json").write_text(json.dumps(profiles, indent=1, default=float))
        del batch, s
        gc.collect()
        wp.synchronize_device(args.device)
    meta["co_tenant_samples"] = co_samples
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for x in [meta["co_tenant_before"], meta["co_tenant_after"]]+co_samples for a in x["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"sharing": meta["gpu_sharing"],
                      "bc_share": {k: v["bc_share"] for k, v in profiles.items()},
                      "gpu_ms_per_substep": {k: v["gpu_ms_per_substep"] for k, v in profiles.items()}}, default=float))


if __name__ == "__main__":
    main()
