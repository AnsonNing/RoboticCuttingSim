"""Protocol v5 ``workload_e_v2_gpu``: batched GPU check of the full-fidelity rubber workload E v2 (n_t 4), N = 1/4/8.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python \
        scripts/rubber/p2_2_v5/run_workload_e_v2.py --out artifacts/rl_ready/p2_2_v5/<id>

Setup from ``configs/rubber/rubber_phys_tests_v5.json`` (``workload_e_v2_gpu``) and ``configs/rubber/workload_e_v2.json``
row E_full: profile rubber_ref_v1, n_t 4, case A clamp, 50 Hz, per-reset material randomisation inside the declared
ranges, dt from the range maxima, stable-step guard 1.0, CUDA graph chunks, v1 solver options, the code's default BC
launch (recorded). Per N: build (capture included), one cold control step, ``warm_steps`` timed control steps
(median = warm), 20 masked single-env resets with resampled material (p95), L2 invariants (finite, mass constant
between resets, invalid bits 0, guard violations 0, env-ownership audits 0), VRAM (process, nvidia-smi, sampled every
second by a thread; Warp mempool). No cameras, no recording, no robot. GPU co-tenants are sampled before, every 5 s
during and after the run, filtered to this GPU's UUID; SHARED_GPU whenever another process is on it. Rows are written
after every N.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/"configs"/"rubber"/"rubber_phys_tests_v5.json"
WORKLOAD = ROOT/"configs"/"rubber"/"workload_e_v2.json"


def _sh(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _gpus():
    rows = _sh("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader,nounits").splitlines()
    return [dict(zip(("index", "name", "uuid", "driver", "mem_used_mb", "mem_total_mb", "util_pct"),
                     [v.strip() for v in r.split(",")])) for r in rows]


def _apps():
    rows = _sh("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits")
    return [dict(zip(("pid", "gpu_uuid", "used_mb"), [v.strip() for v in r.split(",")])) for r in rows.splitlines()
            if r.strip()]


class Sampler(threading.Thread):
    """Own VRAM peak (1 s) and co-tenant samples on this GPU (5 s)."""

    def __init__(self, uuid):
        super().__init__(daemon=True)
        self.uuid, self.peak_mb, self.samples, self.stop_flag = uuid, 0.0, [], threading.Event()

    def sample(self):
        apps = [a for a in _apps() if a["gpu_uuid"] == self.uuid]
        gpu = next((g for g in _gpus() if g["uuid"] == self.uuid), {})
        own = [float(a["used_mb"]) for a in apps if a["pid"] == str(os.getpid())]
        if own:
            self.peak_mb = max(self.peak_mb, own[0])
        return {"time": time.strftime("%H:%M:%S"), "util_pct": gpu.get("util_pct"), "mem_used_mb": gpu.get("mem_used_mb"),
                "others": [a for a in apps if a["pid"] != str(os.getpid())]}

    def run(self):
        k = 0
        while not self.stop_flag.wait(1.0):
            s = self.sample()
            if k % 5 == 0:
                self.samples.append(s)
            k += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT/"src"))
    import warp as wp
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    wp.init()
    st = json.loads(CONFIG.read_text())["workload_e_v2_gpu"]
    row = json.loads(WORKLOAD.read_text())["workloads"][0]
    assert row["variant"] == "E_full" and row["reduced_fidelity"] is False
    wl = row["material"]
    g, rnd = wl["geometry_m"], wl["randomisation"]
    beta = float(wl["bulk_ratio_beta"])
    step_mat = MaterialValues(mu0=rnd["shear_modulus_eq_pa"][1], kappa=beta*rnd["shear_modulus_eq_pa"][1],
                              density=rnd["density_kg_m3"][0])
    ranges = {"mu0": tuple(rnd["shear_modulus_eq_pa"]), "density": tuple(rnd["density_kg_m3"])}
    spec = rs.RubberStripSpec(length=g["length"], width=g["width"], thickness=g["thickness"], n_t=wl["n_t"], case="A",
                              resolution_profile=wl["resolution_profile"])
    guard = float(st["stable_step_guard"])
    dev = wp.get_device(args.device)
    uuid = str(getattr(dev, "uuid", "") or "")
    gpus = _gpus()
    phys = next((x for x in gpus if uuid and x["uuid"].endswith(uuid.replace("GPU-", ""))), None)
    if phys is None:                              # Warp without a uuid: CUDA_VISIBLE_DEVICES with PCI_BUS_ID order
        phys = gpus[int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])]
    status = _sh("git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests")
    meta = {"command": " ".join(sys.argv), "python": sys.executable, "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"), "device": args.device,
            "device_name": dev.name, "warp_device_uuid": uuid or None, "gpu": phys,
            "gpu_driver_version": phys.get("driver"),
            "nvidia_smi_header": next((ln.strip(" |") for ln in _sh("nvidia-smi").splitlines()
                                       if "Driver Version" in ln), None),
            "commit": _sh("git", "rev-parse", "HEAD"), "dirty_tree": bool(status), "dirty_paths": status.splitlines(),
            "protocol_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (CONFIG, WORKLOAD)},
            "warp": wp.config.version, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "workload": "E_full (workload_e_v2.json)"}
    sampler = Sampler(phys["uuid"])
    meta["co_tenant_before"] = sampler.sample()
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    sampler.start()
    rows = []

    def build(N):
        lives = [LiveRubberMaterial(spec, control_dt=wl["control_dt_s"], device="cpu", step_material=step_mat,
                                    stable_step_guard=guard) for _ in range(N)]
        batch = BatchedRubberMaterial(lives, device=args.device, ranges=ranges, strict=False)
        del lives
        batch.reset(seeds=np.arange(N)+100, rng_ranges=True)
        return batch

    for N in st["N"]:
        n_samples0 = len(sampler.samples)
        wp.synchronize_device(args.device)
        t0 = time.perf_counter()
        batch = build(N)
        wp.synchronize_device(args.device)
        build_s = time.perf_counter()-t0
        env_p = batch.packed.host["env_of_particle"]
        mass0 = np.bincount(env_p, weights=batch.solver.mass.numpy().astype(np.float64))
        walls = []
        for _ in range(1+int(st["warm_steps"])):
            wp.synchronize_device(args.device)
            t = time.perf_counter()
            batch.advance()
            wp.synchronize_device(args.device)
            walls.append(time.perf_counter()-t)
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
        batch.advance()                                    # one step after the resets: invariants still hold
        wp.synchronize_device(args.device)
        diag2 = batch.diagnostics_read()
        invalid2 = batch.invalid_reason.numpy().tolist()
        audits = batch.audits()
        sampler.sample()
        ok = (all(not d["nonfinite"] for d in diag+diag2) and np.array_equal(mass0, mass1) and not any(invalid)
              and not any(invalid2) and not any(guard_v) and not any(batch.solver.cfl_violations.numpy().tolist())
              and all(v == 0 for v in audits.values()))
        warm = float(np.median(walls[1:]))
        others = [o for s in [meta["co_tenant_before"]]+sampler.samples[n_samples0:] for o in s["others"]]
        rows.append({"N": N, "particles_per_env": int(batch.packed.particle_counts[0]), "control_hz": wl["control_hz"],
                     "material_dt_s": batch.solver.dt, "substeps": batch.solver.substeps, "n_t": wl["n_t"],
                     "beta": beta, "resolution_profile": wl["resolution_profile"], "cfl_at_range_maxima": wl["cfl"],
                     "stable_step_guard": guard, "bc_launch": getattr(batch.solver, "bc_launch", None),
                     "cameras": "none", "recording": "none", "graph": batch.capture_receipt, "build_s": build_s,
                     "control_step_wall_s": walls, "cold_wall_s": walls[0], "warm_wall_s_median": warm,
                     "env_steps_per_s": N/warm, "aggregate_sim_time_per_wall": N*wl["control_dt_s"]/warm,
                     "ms_per_substep": 1e3*warm/batch.solver.substeps,
                     "particle_substeps_per_s": N*int(batch.packed.particle_counts[0])*batch.solver.substeps/warm,
                     "reset_p95_ms": 1e3*float(np.percentile(resets, 95)), "reset_median_ms": 1e3*float(np.median(resets)),
                     "vram_peak_mb_process_cumulative": sampler.peak_mb, "memory_bytes_arrays": batch.memory_bytes(),
                     "mempool_used_mb": wp.get_mempool_used_mem_current(args.device)/2**20,
                     "invalid_reason": invalid, "invalid_reason_after_resets": invalid2, "guard_violations": guard_v,
                     "audits": audits, "cfl_eff_max_end": max(d["cfl_eff_max"] for d in diag),
                     "Jbar_range": [min(d["Jbar_min"] for d in diag), max(d["Jbar_max"] for d in diag)],
                     "gpu": f"{phys['name']} (index {phys['index']}, uuid {phys['uuid']})",
                     "driver": phys.get("driver"),
                     "sharing": "SHARED_GPU" if others else "EXCLUSIVE",
                     "co_tenants_seen": sorted({f"pid {o['pid']} {o['used_mb']} MB" for o in others}),
                     "physics_verdict": "L2_INVARIANTS_PASS" if ok else "L2_INVARIANTS_FAIL",
                     "vram_criterion_4gb": "PASS" if sampler.peak_mb <= 4096 else "FAIL"})
        (out/"rows.json").write_text(json.dumps(rows, indent=1, default=float))
        print(json.dumps({k: rows[-1][k] for k in ("N", "env_steps_per_s", "ms_per_substep", "reset_p95_ms",
                                                   "vram_peak_mb_process_cumulative", "physics_verdict", "sharing")},
                         default=float), flush=True)
        del batch
        gc.collect()
        wp.synchronize_device(args.device)
    sampler.stop_flag.set()
    sampler.join()
    meta["co_tenant_samples"] = sampler.samples
    meta["co_tenant_after"] = sampler.sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [o for s in [meta["co_tenant_before"], meta["co_tenant_after"]]+sampler.samples for o in s["others"]]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    meta["verdict"] = ("PASS" if all(r["physics_verdict"] == "L2_INVARIANTS_PASS" and r["vram_criterion_4gb"] == "PASS"
                                     for r in rows) and [r["N"] for r in rows] == st["N"] else "FAIL")
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"sharing": meta["gpu_sharing"], "verdict": meta["verdict"]}))


if __name__ == "__main__":
    main()
