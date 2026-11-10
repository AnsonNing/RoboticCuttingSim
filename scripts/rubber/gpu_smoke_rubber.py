"""GPU smoke of the rubber material (ADR 0023 R1.4 / R1.7, RUBBER-PHYS-10 L2 / L3).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> \
        PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/gpu_smoke_rubber.py --out artifacts/rl_ready/p2_1/<id>

Parts: (A) R1.4 single-env LiveRubberMaterial on CUDA, workload E smoke variant (about 100k particles, CUDA graph
chunks); (B) R1.7 L2: BatchedRubberMaterial N = 1, 4, 8 of the same strip with per-env materials resampled inside
the declared ranges: invariants (finite, mass constant between resets, J domain, env-ownership audits), VRAM, timing,
masked-reset latency; (C) L3: batched N = 4 (RUBBER-PHYS-04 heterogeneous envs) per-env observables inside the band
of 3 single-env GPU repeats. Every row states N, particles/env, control Hz, material dt, substeps, cameras (none),
recording (none), warm/cold, the GPU and the co-tenant samples (SHARED_GPU when another process is present).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def gpu_sample():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return {"time": time.strftime("%H:%M:%S"), "gpus": q, "apps": apps}


def own_vram_mb():
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    for line in apps:
        pid, mem = [v.strip() for v in line.split(",")]
        if int(pid) == os.getpid():
            return float(mem)
    return None


class Peak:
    def __init__(self):
        self.value = 0.0

    def update(self):
        v = own_vram_mb()
        if v is not None:
            self.value = max(self.value, v)
        return v


def smoke_spec(rs, cfg):
    g = cfg["smoke_variant"]["geometry_m"]
    return rs.RubberStripSpec(length=g["length"], width=g["width"], thickness=g["thickness"],
                              n_t=cfg["smoke_variant"]["n_t"], case="A", resolution_profile="rubber_ref_v1")


def part_a(device, cfg, peak):
    import warp as wp
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    t0 = time.perf_counter()
    live = LiveRubberMaterial(smoke_spec(rs, cfg), control_dt=0.02, device=device, gravity=(0, 0, -9.81))
    build = time.perf_counter()-t0
    peak.update()
    walls = []
    for k in range(cfg["smoke_variant"]["control_steps"]):
        wp.synchronize_device(device)
        t = time.perf_counter()
        live.advance()
        wp.synchronize_device(device)
        walls.append(time.perf_counter()-t)
        peak.update()
    d = live.last_diagnostics
    return {"part": "A_single_env_R1.4", "N": 1, "particles_per_env": live.scene.n, "control_hz": 50,
            "material_dt_s": live.dt, "substeps": live.substeps, "graph": live.capture_receipt, "build_s": build,
            "control_step_wall_s": walls, "cold_wall_s": walls[0], "warm_wall_s": walls[-1] if len(walls) > 1 else None,
            "sim_time_per_wall_warm": 0.02/walls[-1], "ms_per_substep_warm": 1e3*walls[-1]/live.substeps,
            "finite": not d["nonfinite"], "J_range": [d["J_min"], d["J_max"]], "Jbar_range": [d["Jbar_min"], d["Jbar_max"]],
            "invalid": live.invalid_reason, "vram_peak_mb_process": peak.value,
            "mempool_used_mb": wp.get_mempool_used_mem_current(device)/2**20, "cameras": "none", "recording": "none",
            "pass": (not d["nonfinite"]) and live.invalid_reason == 0 and peak.value <= 1024 and sum(walls) <= 120}


def part_b(device, cfg, Ns, peak, guard_modes=(None,), control_steps=None):
    """``guard_modes``: stable_step_guard values per N (None = off, the v1 launch sequence; a float = the limit),
    run back to back for the same N so that the with/without-guard timing shares the GPU conditions (F2)."""
    rows = []
    for N in Ns:
        for guard in guard_modes:
            rows += _part_b_one(device, cfg, N, peak, guard, control_steps)
    return rows


def _part_b_one(device, cfg, N, peak, guard, control_steps):
    import warp as wp
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    rnd = cfg["material"]["randomisation"]
    step_mat = MaterialValues(mu0=rnd["shear_modulus_eq_pa"][1], kappa=20*rnd["shear_modulus_eq_pa"][1],
                              density=rnd["density_kg_m3"][0])
    ranges = {"mu0": tuple(rnd["shear_modulus_eq_pa"]), "density": tuple(rnd["density_kg_m3"])}
    rows = []
    steps = cfg["smoke_variant"]["control_steps"] if control_steps is None else int(control_steps)
    for N in (N,):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        lives = [LiveRubberMaterial(smoke_spec(rs, cfg), control_dt=0.02, device="cpu", step_material=step_mat,
                                    stable_step_guard=guard) for _ in range(N)]
        batch = BatchedRubberMaterial(lives, device=device, ranges=ranges, strict=False)
        del lives
        batch.reset(seeds=np.arange(N)+100, rng_ranges=True)
        wp.synchronize_device(device)
        build = time.perf_counter()-t0
        peak.update()
        mass0 = batch.solver.mass.numpy().reshape(-1)
        env_p = batch.packed.host["env_of_particle"]
        env_mass0 = np.bincount(env_p, weights=mass0.astype(np.float64))
        walls = []
        for k in range(steps):
            wp.synchronize_device(device)
            t = time.perf_counter()
            batch.advance()
            wp.synchronize_device(device)
            walls.append(time.perf_counter()-t)
            peak.update()
        diag = batch.diagnostics_read()
        env_mass1 = np.bincount(env_p, weights=batch.solver.mass.numpy().astype(np.float64))
        resets = []
        for r in range(20):
            wp.synchronize_device(device)
            t = time.perf_counter()
            batch.reset([r % N], seeds=[1000+r], rng_ranges=True)
            wp.synchronize_device(device)
            resets.append(time.perf_counter()-t)
        audits = batch.audits()
        invalid = batch.invalid_reason.numpy().tolist()
        cfl_violations = batch.solver.cfl_violations.numpy().tolist()
        cfl_eff_max = [d["cfl_eff_max"] for d in diag]
        ok = (all(not d["nonfinite"] for d in diag) and np.array_equal(env_mass0, env_mass1)
              and not any(invalid) and all(v == 0 for v in audits.values()))
        rows.append({"part": "B_batched_L2_R1.7", "N": N, "particles_per_env": int(batch.packed.particle_counts[0]),
                     "control_hz": 50, "material_dt_s": batch.solver.dt, "substeps": batch.solver.substeps,
                     "cameras": "none", "recording": "none", "graph": batch.capture_receipt, "build_s": build,
                     "control_step_wall_s": walls, "cold_wall_s": walls[0], "warm_wall_s": walls[-1],
                     "env_steps_per_s_warm": N/walls[-1], "aggregate_sim_time_per_wall_warm": N*0.02/walls[-1],
                     "ms_per_substep_warm": 1e3*walls[-1]/batch.solver.substeps,
                     "reset_p95_ms": 1e3*float(np.percentile(resets, 95)), "reset_median_ms": 1e3*float(np.median(resets)),
                     "vram_peak_mb_process": peak.value, "memory_bytes_arrays": batch.memory_bytes(),
                     "mempool_used_mb": wp.get_mempool_used_mem_current(device)/2**20,
                     "finite": all(not d["nonfinite"] for d in diag), "mass_constant": bool(np.array_equal(env_mass0, env_mass1)),
                     "invalid_reason": invalid, "audits": audits, "stable_step_guard": guard,
                     "guard_violations_after_run_per_env": cfl_violations, "cfl_eff_max_end_per_env": cfl_eff_max,
                     "ms_per_substep_warm_median": (1e3*float(np.median(walls[1:]))/batch.solver.substeps
                                                    if len(walls) > 1 else None),
                     "Jbar_range": [min(d["Jbar_min"] for d in diag), max(d["Jbar_max"] for d in diag)],
                     "params_mu0": batch.solver.params.numpy()[:, 0].tolist(),
                     "physics_verdict": "L2_INVARIANTS_PASS" if ok else "L2_INVARIANTS_FAIL",
                     "pass": ok and peak.value <= cfg["smoke_variant"]["vram_limit_gb"]*1024})
        del batch
        import gc
        gc.collect()
        wp.synchronize_device(device)
    return rows


def part_c(device, steps=30):
    """L3: RUBBER-PHYS-04 heterogeneous envs, 1 ms control steps; batched N = 4 vs 3 single repeats (GPU)."""
    import warp as wp
    from robotic_cutting_sim import rubber_phys as P
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    envs = batched_04_envs()
    step_mat = P.material(2e6)

    def make(kw, dev):
        spec = rs.RubberStripSpec(length=kw["L"], width=kw["W"], thickness=0.006, n_t=3, case="A")
        return LiveRubberMaterial(spec, material=P.material(kw["mu"], kw["rho"]), control_dt=1e-3,
                                  gravity=(0, 0, -9.81), device=dev, step_material=step_mat)

    def obs_single(live):
        d = live.last_diagnostics
        return [float(live.markers()[-1][2]), d["Jbar_min"], d["Jbar_max"], d["elastic"]]
    singles = {e: [] for e in range(4)}
    for rep in range(3):
        for e, kw in enumerate(envs):
            live = make(kw, device)
            series = []
            for _ in range(steps):
                live.advance()
                series.append(obs_single(live))
            singles[e].append(series)
            del live
    batch = BatchedRubberMaterial([make(kw, "cpu") for kw in envs], device=device)
    bseries = {e: [] for e in range(4)}
    for _ in range(steps):
        batch.advance()
        diag = batch.diagnostics_read()
        mk = batch.markers()
        for e in range(4):
            bseries[e].append([float(mk[e][-1][2]), diag[e]["Jbar_min"], diag[e]["Jbar_max"], diag[e]["elastic"]])
    outside, worst = 0, 0.0
    spread = 0.0
    for e in range(4):
        reps = np.array(singles[e])          # [3, steps, 4]
        lo, hi = reps.min(0), reps.max(0)
        spread = max(spread, float(np.max((hi-lo)/np.maximum(np.abs(hi), 1e-30))))
        b = np.array(bseries[e])
        floor = np.maximum(1e-5*np.abs(b), 1e-9)
        bad = (b < lo-floor) | (b > hi+floor)
        outside += int(bad.sum())
        worst = max(worst, float(np.max(np.maximum(lo-floor-b, b-hi-floor)/np.maximum(np.abs(b), 1e-30))))
    return {"part": "C_L3_band", "N": 4, "envs": envs, "control_steps": steps, "control_dt_s": 1e-3,
            "observables": ["tip marker z", "Jbar_min", "Jbar_max", "elastic energy"],
            "band": "min/max of 3 single-env GPU repeats widened by max(1e-5 |value|, 1e-9)",
            "values_outside_band": outside, "worst_excess_relative": worst,
            "single_repeat_relative_spread_max": spread, "pass": outside == 0}


def batched_04_envs():
    return [dict(L=0.06, W=0.006, mu=2e6, rho=1150.0), dict(L=0.048, W=0.006, mu=2e6, rho=1150.0),
            dict(L=0.06, W=0.012, mu=2e6, rho=1150.0), dict(L=0.06, W=0.006, mu=1e6, rho=1000.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parts", default="A,B,C")
    ap.add_argument("--N", default="1,4,8")
    ap.add_argument("--guard-modes", default="off",
                    help="part B: comma list of stable_step_guard settings per N, 'off' or a limit (e.g. off,1.0)")
    ap.add_argument("--control-steps", type=int, default=None, help="part B: control steps (default: workload E)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    import warp as wp
    wp.init()
    cfg = json.loads((ROOT/"configs"/"rubber"/"workload_e_v1.json").read_text())["workloads"][0]
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device, "device_name": wp.get_device(args.device).name, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "warp": wp.config.version, "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    peak = Peak()
    results = []
    parts = args.parts.split(",")
    try:
        if "A" in parts:
            results.append(part_a(args.device, cfg, peak))
            (out/"results.json").write_text(json.dumps(results, indent=1, default=float))
        if "B" in parts:
            modes = [None if m.strip() == "off" else float(m) for m in args.guard_modes.split(",")]
            results += part_b(args.device, cfg, [int(n) for n in args.N.split(",")], peak, modes, args.control_steps)
            (out/"results.json").write_text(json.dumps(results, indent=1, default=float))
        if "C" in parts:
            results.append(part_c(args.device))
    except Exception:
        import traceback
        results.append({"error": traceback.format_exc()})
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for a in meta["co_tenant_before"]["apps"]+meta["co_tenant_after"]["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    (out/"results.json").write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps({"sharing": meta["gpu_sharing"], "results": [{k: r.get(k) for k in ("part", "N", "pass", "error")}
                                                                   for r in results]}, indent=1))


if __name__ == "__main__":
    main()
