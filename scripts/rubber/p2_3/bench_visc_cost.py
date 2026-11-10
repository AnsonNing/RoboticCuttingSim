"""RUBBER-VISC-10 (reported, not scored): GPU cost of ``neo_hookean_jbar_prony_v1`` vs the elastic model in the batched
rubber material (ADR 0023 §21.2.9, D11 columns; task R3.8).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_3/bench_visc_cost.py \
        --out artifacts/rl_ready/p2_3/<dir> [--warm 3]

Rows: workload E geometry (configs/rubber/workload_e_v1.json: case A clamp 150 x 20 x 4 mm, 50 Hz, gravity -z) at
n_t 4 (rubber_ref_v1, the formal resolution) and the labelled n_t 3 row (rubber_rl_v1); N = 1 / 4 / 8; variants
``elastic`` (neo_hookean_jbar_v1), ``P2_energy_off`` (M = 2, visco_energy off_v1), ``P2_energy_on`` (branch_work_v1).
Every row: the protocol's common options (F8 projection + kick-average work + F9 + F10), stable-step guard 1.0, BC
launch motion_table_v1 (default), the P2 set (mu0 0.4 MPa, beta 20, rho 1150; the elastic rows the same mu0, kappa,
rho), dt from the nominal material (identical dt and substeps for every variant of one resolution), CUDA graph chunks
of 500 substeps, no cameras, no recording. Per row: build (incl. capture), one cold control step, ``--warm`` timed
control steps (median = warm), VRAM (nvidia-smi process, cumulative peak; Warp mempool), L2 invariants (finite,
invalid bits 0, guard violations 0), registry-state bytes per particle. Stage split: ``profile_substeps`` eager
substeps of the N = 4, n_t 4 batch under ``wp.timing_begin`` per variant (GPU kernel time; eager launches, so the split,
not the throughput). Co-tenant samples before / during / after; SHARED_GPU whenever another process is present.
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
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_visc_tests_v1.json"
WORKLOAD = ROOT/"configs"/"rubber"/"workload_e_v1.json"
VARIANTS = (("elastic", (), "off_v1"), ("P2_energy_off", "P2", "off_v1"), ("P2_energy_on", "P2", "branch_work_v1"))


def driver_version():
    return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip().splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--N", type=int, nargs="*", default=[1, 4, 8])
    ap.add_argument("--n_t", type=int, nargs="*", default=[4, 3])
    ap.add_argument("--profile-substeps", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    import warp as wp
    from gpu_smoke_rubber import Peak, gpu_sample
    from robotic_cutting_sim import rubber_visco_protocol as vp
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    wp.init()
    wl = json.loads(WORKLOAD.read_text())["workloads"][0]["material"]
    g = wl["geometry_m"]
    mu0, P2 = vp.prony_set("P2")
    control_dt = float(wl["control_dt_s"])
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    meta = {"test": "RUBBER-VISC-10 (reported, not scored)", "command": " ".join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": args.device,
            "device_name": wp.get_device(args.device).name, "driver": driver_version(), "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
            .strip(), "dirty_tree": bool(dirty),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "workload_sha256": hashlib.sha256(WORKLOAD.read_bytes()).hexdigest(), "warp": wp.config.version,
            "python": sys.executable, "options": vp.OPTIONS, "stable_step_guard": 1.0, "bc_launch": "motion_table_v1",
            "cameras": "none", "recording": "none", "co_tenant_before": gpu_sample(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED (illustrative Prony set P2)"}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    peak = Peak()
    rows, co_samples = [], []

    def spec_of(n_t):
        return vp.strip_spec(L=g["length"], W=g["width"], t=g["thickness"], n_t=n_t, case="A")

    def build(n_t, N, terms, energy):
        mat = vp.material(mu0, terms)
        lives = [vp.make_live(spec_of(n_t), mat, control_dt=control_dt, gravity=(0, 0, -9.81), energy=energy,
                              step_material=vp.material(mu0, ())) for _ in range(N)]
        batch = BatchedRubberMaterial(lives, device=args.device, strict=False)
        del lives
        return batch

    for n_t in args.n_t:
        for N in args.N:
            for name, set_name, energy in VARIANTS:
                terms = P2 if set_name else ()
                wp.synchronize_device(args.device)
                t0 = time.perf_counter()
                batch = build(n_t, N, terms, energy)
                wp.synchronize_device(args.device)
                build_s = time.perf_counter()-t0
                peak.update()
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
                invalid = batch.invalid_reason.numpy().tolist()
                guard_v = batch.solver.cfl_violations.numpy().tolist()
                ok = all(not d["nonfinite"] for d in diag) and not any(invalid) and not any(guard_v)
                warm = float(np.median(walls[1:]))
                s = batch.solver
                P = int(batch.packed.particle_counts[0])
                rows.append({"variant": name, "model": s.model.model_id, "M": s.M, "visco_energy": energy, "N": N,
                             "n_t": n_t, "resolution_profile": "rubber_ref_v1" if n_t == 4 else
                             "rubber_rl_v1 (labelled n_t 3 row)", "particles_per_env": P, "control_hz": 1/control_dt,
                             "material_dt_s": s.dt, "substeps": s.substeps, "beta": 20.0, "cameras": "none",
                             "recording": "none", "graph": batch.capture_receipt, "build_s": build_s,
                             "control_step_wall_s": walls, "cold_wall_s": walls[0], "warm_wall_s_median": warm,
                             "warm_cold": "warm = median of steps 2..", "ms_per_substep": 1e3*warm/s.substeps,
                             "env_steps_per_s": N/warm, "aggregate_sim_time_per_wall": N*control_dt/warm,
                             "particle_substeps_per_s": N*P*s.substeps/warm,
                             "registry_state_bytes_per_particle": 48*(s.M+1) if s.M else 0,
                             "energy_option_bytes_per_particle": 72 if s.visco_energy_on else 0,
                             "vram_peak_mb_process_cumulative": peak.value, "memory_bytes_arrays": batch.memory_bytes(),
                             "mempool_used_mb": wp.get_mempool_used_mem_current(args.device)/2**20,
                             "invalid_reason": invalid, "guard_violations": guard_v,
                             "cfl_eff_max_end": max(d["cfl_eff_max"] for d in diag),
                             "physics_verdict": "L2_INVARIANTS_PASS" if ok else "L2_INVARIANTS_FAIL"})
                (out/"rows.json").write_text(json.dumps(rows, indent=1, default=float))
                print(json.dumps({k: rows[-1][k] for k in ("variant", "N", "n_t", "particles_per_env", "substeps",
                                                           "ms_per_substep", "env_steps_per_s",
                                                           "vram_peak_mb_process_cumulative", "physics_verdict")},
                                 default=float), flush=True)
                del batch
                gc.collect()
                wp.synchronize_device(args.device)
    # stage split (eager substeps under CUDA activity timing) at N = 4, n_t 4
    profiles = {}
    pN, pnt = (4 if 4 in args.N else args.N[0]), args.n_t[0]
    for name, set_name, energy in VARIANTS:
        batch = build(pnt, pN, P2 if set_name else (), energy)
        batch.advance()
        s = batch.solver
        wp.synchronize_device(args.device)
        wp.timing_begin()
        for _ in range(args.profile_substeps):
            s.step()
        wp.synchronize_device(args.device)
        results = wp.timing_end()
        per = defaultdict(float)
        for r in results:
            per[r.name] += r.elapsed
        total = sum(per.values())
        stress = sum(v for k, v in per.items() if "stress" in k or "cell_jbar" in k)
        profiles[name] = {"N": pN, "n_t": pnt, "substeps": args.profile_substeps, "gpu_ms_total": total,
                          "gpu_ms_per_substep": total/args.profile_substeps,
                          "stress_pass_ms_per_substep": stress/args.profile_substeps,
                          "kernels_ms": dict(sorted(per.items(), key=lambda kv: -kv[1])),
                          "share": {k: v/total for k, v in per.items()}}
        del batch
        gc.collect()
    meta["co_tenant_samples"] = co_samples
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for x in [meta["co_tenant_before"], meta["co_tenant_after"]]+co_samples for a in x["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    meta["co_tenant_pids_seen"] = sorted({a.split(",")[0].strip() for a in others})
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    (out/"profile.json").write_text(json.dumps(profiles, indent=1, default=float))
    print(json.dumps({"sharing": meta["gpu_sharing"], "stress_ms": {k: v["stress_pass_ms_per_substep"]
                                                                    for k, v in profiles.items()}}, default=float))


if __name__ == "__main__":
    main()
