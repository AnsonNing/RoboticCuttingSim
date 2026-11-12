"""R2.4 / F5: workload E throughput, VRAM and guard profiling of the batched rubber material (protocol v4), GPU.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> PYTHONPATH=src \
        .deps/isaacsim-venv/bin/python scripts/rubber/p2_2/run_throughput_v4.py --out artifacts/rl_ready/p2_2/<id>

Setup from ``configs/rubber/rubber_phys_tests_v4.json`` (``throughput_v4``): workload E (``configs/rubber/workload_e_v1.json``,
profile rubber_rl_v1, n_t 3, case A clamp, 50 Hz, per-reset material randomisation inside the declared ranges, dt from
the range maxima), stable-step guard on, N = 1/4/8/16. Per N: build (capture included), one cold control step,
``warm_steps`` timed control steps (median = warm), 20 masked single-env resets (p95), L2 invariants (finite, mass
constant between resets, invalid bits 0, env-ownership audits 0, guard violations 0), VRAM (process, nvidia-smi;
Warp mempool). Then the kernel split (F5): ``profile_substeps`` eager substeps of the N = ``profile_N`` batch under
``wp.timing_begin`` (CUDA activity timing, GPU execution time per kernel; eager launches, so the split, not the
throughput). No cameras, no recording. Co-tenant samples before / after; SHARED_GPU whenever another process is present.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    import warp as wp
    from gpu_smoke_rubber import gpu_sample, Peak
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    wp.init()
    cfg_path = ROOT/"configs"/"rubber"/"rubber_phys_tests_v4.json"
    st = json.loads(cfg_path.read_text())["throughput_v4"]
    wl = json.loads((ROOT/"configs"/"rubber"/"workload_e_v1.json").read_text())["workloads"][0]["material"]
    g, rnd = wl["geometry_m"], wl["randomisation"]
    beta = float(wl["bulk_ratio_beta"])
    step_mat = MaterialValues(mu0=rnd["shear_modulus_eq_pa"][1], kappa=beta*rnd["shear_modulus_eq_pa"][1],
                              density=rnd["density_kg_m3"][0])
    ranges = {"mu0": tuple(rnd["shear_modulus_eq_pa"]), "density": tuple(rnd["density_kg_m3"])}
    spec = rs.RubberStripSpec(length=g["length"], width=g["width"], thickness=g["thickness"], n_t=wl["n_t"], case="A",
                              resolution_profile=wl["resolution_profile"])
    guard = float(st["stable_step_guard"])
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device, "device_name": wp.get_device(args.device).name, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(), "warp": wp.config.version,
            "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    peak = Peak()
    rows, co_samples = [], []

    def build(N, guard_limit):
        lives = [LiveRubberMaterial(spec, control_dt=wl["control_dt_s"], device="cpu", step_material=step_mat,
                                    stable_step_guard=guard_limit) for _ in range(N)]
        batch = BatchedRubberMaterial(lives, device=args.device, ranges=ranges, strict=False)
        del lives
        batch.reset(seeds=np.arange(N)+100, rng_ranges=True)
        return batch

    for N in st["N"]:
        wp.synchronize_device(args.device)
        t0 = time.perf_counter()
        batch = build(N, guard)
        wp.synchronize_device(args.device)
        build_s = time.perf_counter()-t0
        peak.update()
        env_p = batch.packed.host["env_of_particle"]
        mass0 = np.bincount(env_p, weights=batch.solver.mass.numpy().astype(np.float64))
        walls = []
        for _ in range(1+int(st["warm_steps"])):
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
        resets = []
        for r in range(20):
            wp.synchronize_device(args.device)
            t = time.perf_counter()
            batch.reset([r % N], seeds=[1000+r], rng_ranges=True)
            wp.synchronize_device(args.device)
            resets.append(time.perf_counter()-t)
        audits = batch.audits()
        ok = (all(not d["nonfinite"] for d in diag) and np.array_equal(mass0, mass1) and not any(invalid)
              and not any(guard_v) and all(v == 0 for v in audits.values()))
        warm = float(np.median(walls[1:]))
        rows.append({"N": N, "particles_per_env": int(batch.packed.particle_counts[0]), "control_hz": 50,
                     "material_dt_s": batch.solver.dt, "substeps": batch.solver.substeps, "n_t": wl["n_t"], "beta": beta,
                     "cfl_at_range_maxima": 0.5, "stable_step_guard": guard, "cameras": "none", "recording": "none",
                     "graph": batch.capture_receipt, "build_s": build_s, "control_step_wall_s": walls,
                     "cold_wall_s": walls[0], "warm_wall_s_median": warm, "env_steps_per_s": N/warm,
                     "aggregate_sim_time_per_wall": N*wl["control_dt_s"]/warm,
                     "ms_per_substep": 1e3*warm/batch.solver.substeps,
                     "particle_substeps_per_s": N*int(batch.packed.particle_counts[0])*batch.solver.substeps/warm,
                     "reset_p95_ms": 1e3*float(np.percentile(resets, 95)), "reset_median_ms": 1e3*float(np.median(resets)),
                     "vram_peak_mb_process_cumulative": peak.value, "memory_bytes_arrays": batch.memory_bytes(),
                     "mempool_used_mb": wp.get_mempool_used_mem_current(args.device)/2**20,
                     "invalid_reason": invalid, "guard_violations": guard_v, "audits": audits,
                     "cfl_eff_max_end": max(d["cfl_eff_max"] for d in diag),
                     "physics_verdict": "L2_INVARIANTS_PASS" if ok else "L2_INVARIANTS_FAIL"})
        (out/"rows.json").write_text(json.dumps(rows, indent=1, default=float))
        print(json.dumps({k: rows[-1][k] for k in ("N", "env_steps_per_s", "ms_per_substep", "reset_p95_ms",
                                                   "vram_peak_mb_process_cumulative", "physics_verdict")},
                         default=float), flush=True)
        del batch
        gc.collect()
        wp.synchronize_device(args.device)

    # F5: kernel split with and without the guard (eager substeps under CUDA activity timing)
    profiles = {}
    for label, limit in (("guard_on", guard), ("guard_off", None)):
        batch = build(int(st["profile_N"]), limit)
        batch.advance()                            # warm state (graph replay)
        s = batch.solver
        wp.synchronize_device(args.device)
        wp.timing_begin()
        for _ in range(int(st["profile_substeps"])):
            s.step()
        wp.synchronize_device(args.device)
        results = wp.timing_end()
        per = defaultdict(float)
        for r in results:
            per[r.name] += r.elapsed
        total = sum(per.values())
        profiles[label] = {"N": int(st["profile_N"]), "substeps": int(st["profile_substeps"]),
                           "gpu_ms_total": total, "gpu_ms_per_substep": total/int(st["profile_substeps"]),
                           "kernels_ms": dict(sorted(per.items(), key=lambda kv: -kv[1])),
                           "share": {k: v/total for k, v in per.items()}}
        del batch
        gc.collect()
    guard_ms = sum(v for k, v in profiles["guard_on"]["kernels_ms"].items() if "cfl_guard" in k)
    profiles["guard_share_of_gpu_time"] = guard_ms/profiles["guard_on"]["gpu_ms_total"]
    meta["co_tenant_samples"] = co_samples
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for x in [meta["co_tenant_before"], meta["co_tenant_after"]]+co_samples for a in x["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    (out/"profile.json").write_text(json.dumps(profiles, indent=1, default=float))
    print(json.dumps({"sharing": meta["gpu_sharing"], "guard_share": profiles["guard_share_of_gpu_time"]},
                     default=float))


if __name__ == "__main__":
    main()
