"""S1 GPU smoke ``rubber_selfpen_check_v1`` (configs/rubber/rubber_selfpen_check_v1.json, SHA-pinned before this run):
COST (detector on vs off, ms per substep, reported), L2 (invariants of the detector-on cost arms), TRIP (deliberate
crossing, single and batched N = 3) and NO_FALSE_TRIP (v5 RUBBER-PHYS-09 scenes and visco runner jobs with the detector
forced on). Refuses to run with < 4 GB free on the GPU; never overwrites an output directory.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \\
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_s1/gpu_selfpen_check.py \\
        --out artifacts/rl_ready/p2_s1/<dir>
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_selfpen_check_v1.json"
PROTOCOL_SHA = "cc5cf3339e0c69cff13517686833e2293c101c4242b3702ef9980bb5d08b8bc0"
MIN_FREE_MB = 4096


def sh(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def free_mb(index):
    return float(sh("nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
                 .splitlines()[0])


def gpu_sample(index):
    uuid = sh("nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader")
    apps = [a for a in sh("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                          "--format=csv,noheader").splitlines() if uuid in a]
    q = sh("nvidia-smi", f"--id={index}", "--query-gpu=memory.used,memory.total,utilization.gpu",
           "--format=csv,noheader")
    return {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": q, "apps": apps}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warm", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA:
        sys.exit("rubber_selfpen_check_v1.json does not match its pinned SHA-256")
    cfg = json.loads(raw)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = int(vis.split(",")[0]) if vis.strip() else 0
    fm = free_mb(phys)
    if fm < MIN_FREE_MB:
        sys.exit(f"BLOCKED: GPU{phys} has {fm:.0f} MB free < {MIN_FREE_MB} MB; not running")
    out.mkdir(parents=True)
    import warp as wp
    wp.config.log_level = 30
    wp.init()
    status = sh("git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests")
    samples = [gpu_sample(phys)]
    meta = {"check": cfg["id"], "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": PROTOCOL_SHA,
            "commit": sh("git", "rev-parse", "HEAD"), "dirty_tree": bool(status), "dirty_paths": status.splitlines(),
            "command": " ".join([sys.executable]+sys.argv), "pid": os.getpid(), "device": args.device,
            "cuda_visible_devices": vis, "physical_gpu": phys,
            "gpu": sh("nvidia-smi", f"--id={phys}", "--query-gpu=name,driver_version", "--format=csv,noheader"),
            "warp": wp.__version__, "python": platform.python_version(), "free_mb_at_start": fm,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    log = open(out/"progress.log", "w")

    def note(obj):
        line = json.dumps(obj, default=float)
        print(line, flush=True)
        log.write(line+"\n")
        log.flush()

    note({"meta": meta})
    from robotic_cutting_sim import live_rubber_material as lrm
    from robotic_cutting_sim import rubber_contact as rct
    from robotic_cutting_sim import rubber_material as rm
    from robotic_cutting_sim import rubber_phys as rp
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim import rubber_self_penetration as rsp
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    T = load_module("s1_tests", ROOT/"tests"/"test_rubber_self_penetration.py")
    checks_256 = cfg["detector"]["check_substeps"]
    result = {"meta": meta, "rows": {}}

    # ------------------------------------------------------------------ COST + L2
    recipe = rm.load_recipe(ROOT/"configs"/"materials"/"rubber_generic_visco_v1.json")
    law = json.loads((ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json").read_text())["common"]["law_parameters"]
    contact = rct.ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), law="grid_node_penalty_coulomb_v3",
                              normal_stiffness=law["normal_stiffness_kappa_n"],
                              tangential_stiffness=law["tangential_stiffness_kappa_t"],
                              normal_damping_ratio=law["normal_damping_ratio_zeta_n"],
                              damping_ratio=law["tangential_damping_ratio_zeta_t"],
                              slip_rate_damping_ratio=law["slip_rate_damping_ratio_zeta_d"])
    step_mat = rm.MaterialValues(mu0=8e5, kappa=20*8e5, density=1000.0)

    def p26_live(sp):
        spec = rs.RubberStripSpec(length=0.125, width=0.016, thickness=0.004, n_t=4, dx=0.001, case="A",
                                  resolution_profile="rubber_ref_v1")
        return lrm.LiveRubberMaterial(spec, recipe=recipe, material=recipe.nominal(), control_dt=0.02,
                                      step_material=step_mat, gravity=(0.0, 0.0, -9.81), device="cpu", strict=False,
                                      stable_step_guard=1.0, contact=contact, self_penetration_check=sp,
                                      self_penetration_check_substeps=checks_256, **T.P26)

    cost = []
    for N in (1, 8):
        arms = {}
        for sp in ("surface_grid_v1", "off_v1"):
            t0 = time.perf_counter()
            batch = rmb.BatchedRubberMaterial([p26_live(sp) for _ in range(N)], device=args.device, strict=False)
            wp.synchronize_device(args.device)
            arms[sp] = {"batch": batch, "build_s": time.perf_counter()-t0, "walls": [],
                        "mass0": batch.solver.mass.numpy().copy()}
        order = ["surface_grid_v1", "off_v1"]
        for k in range(1+args.warm):
            for sp in order:
                b = arms[sp]["batch"]
                wp.synchronize_device(args.device)
                t1 = time.perf_counter()
                b.advance()
                wp.synchronize_device(args.device)
                arms[sp]["walls"].append(time.perf_counter()-t1)
            order.reverse()                       # interleaved: on, off, off, on, on, off, ...
        samples.append(gpu_sample(phys))
        row = {"N": N}
        for sp, a in arms.items():
            b = a["batch"]
            warm = float(np.median(a["walls"][1:]))
            row[sp] = {"particles_per_env": int(b.packed.particle_offset[1]), "substeps": int(b.solver.substeps),
                       "dt_s": float(b.solver.dt), "graph": b.capture_receipt, "build_capture_s": a["build_s"],
                       "cold_s": a["walls"][0], "warm_walls_s": a["walls"][1:], "warm_median_s": warm,
                       "ms_per_substep": 1e3*warm/b.solver.substeps, "env_steps_per_s": N/warm,
                       "memory_bytes": b.memory_bytes()}
            if sp == "surface_grid_v1":
                diag = b.diagnostics_read()
                reads = b.self_penetration_read()
                steps = 1+args.warm
                per = len(b.self_penetration.positions_per_step)
                l2 = {"finite": all(d["nonfinite"] == 0.0 for d in diag),
                      "mass_unchanged": bool(np.array_equal(b.solver.mass.numpy(), a["mass0"])),
                      "audits": b.audits(), "invalid_reason": b.invalid_reason.numpy().tolist(),
                      "hits": [r["hits"] for r in reads], "checks": [r["checks"] for r in reads],
                      "checks_expected": steps*per, "blind_steps": [r["blind_steps"] for r in reads],
                      "detector_receipt": b.receipt()["self_penetration"]}
                l2["audits_zero"] = all(int(v) == 0 for v in l2["audits"].values())
                l2["pass"] = bool(l2["finite"] and l2["mass_unchanged"] and l2["audits_zero"]
                                  and not any(l2["invalid_reason"]) and not any(l2["hits"])
                                  and all(c == steps*per for c in l2["checks"]))
                row["L2"] = l2
        on, off = row["surface_grid_v1"]["ms_per_substep"], row["off_v1"]["ms_per_substep"]
        row["overhead_rel"] = on/off-1.0
        row["overhead_ms_per_substep"] = on-off
        note({"COST": {k: v for k, v in row.items() if k != "L2"}, "L2_pass": row["L2"]["pass"]})
        cost.append(row)
        del arms
    result["rows"]["COST"] = cost
    result["rows"]["L2"] = {"pass": all(r["L2"]["pass"] for r in cost), "per_N": {r["N"]: r["L2"] for r in cost}}

    # ------------------------------------------------------------------ TRIP
    def fold_single(device):
        live = T._live("surface_grid_v1", device=device)
        bits, first8, other_before = [], None, False
        for k in range(1, T.FOLD_STEPS+1):
            live.advance({1: {"translation": T.fold_target(k*T.CDT)}})
            bits.append(int(live.invalid_reason))
            if live.invalid_reason & 8 and first8 is None:
                first8 = k
            if first8 is None and live.invalid_reason & ~8:
                other_before = True
        return {"device": live.device, "substeps": live.substeps, "bits_final": bits[-1], "first_bit8_step": first8,
                "other_bit_before_8": other_before, "detector": live.last_self_penetration}
    trip = {"single_gpu": fold_single(args.device), "single_cpu": fold_single("cpu")}
    kws = [dict(L=0.018), dict(), dict(L=0.021, W=0.0045, mu=1.5e5, rho=1200.0)]
    batch = rmb.BatchedRubberMaterial([T._live("surface_grid_v1", **kw) for kw in kws], device=args.device,
                                      strict=False)
    first8 = [None]*3
    other_before = [False]*3
    for k in range(1, T.FOLD_STEPS+1):
        batch.advance(translation=T._batch_inputs(k, 1, 3))
        bits = batch.invalid_reason.numpy()
        for e in range(3):
            if bits[e] & 8 and first8[e] is None:
                first8[e] = k
            if first8[e] is None and bits[e] & ~8:
                other_before[e] = True
    trip["batched_n3_gpu"] = {"bits_final": batch.invalid_reason.numpy().tolist(), "first_bit8_step": first8,
                              "other_bit_before_8": other_before, "detector": batch.self_penetration_read(),
                              "fold_env": 1}
    del batch
    sg, bt = trip["single_gpu"], trip["batched_n3_gpu"]
    trip["pass"] = bool(sg["bits_final"] & 8 and not sg["other_bit_before_8"]
                        and bt["bits_final"][1] & 8 and not bt["bits_final"][0] and not bt["bits_final"][2]
                        and not bt["other_bit_before_8"][1])
    result["rows"]["TRIP"] = trip
    samples.append(gpu_sample(phys))
    note({"TRIP": trip})

    # ------------------------------------------------------------------ NO_FALSE_TRIP
    built = []

    class Forced(lrm.LiveRubberMaterial):
        def __init__(self, *a, **kw):
            kw["self_penetration_check"] = "surface_grid_v1"
            kw["self_penetration_check_substeps"] = checks_256
            kw["device"] = args.device
            super().__init__(*a, **kw)
            built.append(self)
    rp.LiveRubberMaterial = Forced
    lrm.LiveRubberMaterial = Forced
    nft = []
    p5 = rp.load_config(rp.CONFIG.parent/"rubber_phys_tests_v5.json")
    for c in rp.configs_09(p5):
        t0 = time.perf_counter()
        r = rp.case_records(c["scenario"], c["t"], c["Wt"], p5, device=args.device, stable_step_guard=1.0)
        live = built[-1]
        sp = live.last_self_penetration
        row = {"scene": f"phys09_{c['scenario']}_t{c['t']*1e3:g}mm_Wt{c['Wt']:g}", "scored":
               c["scenario"] != "stretch_release", "particles": r["particles"], "substeps": r["substeps"],
               "steps": r["steps"], "invalid_flags": r["invalid_flags"], "detector": sp,
               "checks_ok": sp["checks"] > 0, "wall_s": time.perf_counter()-t0}
        row["pass"] = (not row["scored"]) or (row["checks_ok"] and not r["invalid_flags"] and sp["hits"] == 0)
        note({"NO_FALSE_TRIP": row})
        nft.append(row)
    visc = load_module("run_visc_v1_s1", ROOT/"scripts"/"rubber"/"p2_3"/"run_visc_v1.py")
    vcfg, vsha = visc.load_protocol()
    jobs = {name: (kind, p) for name, kind, p in visc.job_list(vcfg, smoke=False)}
    for name in ("v03_strip_16hz", "v02_creep", "v04b_cantilever_x1"):
        kind, p = jobs[name]
        t0 = time.perf_counter()
        info = getattr(visc, f"job_{kind}")(p, False)["info"]
        live = built[-1]
        sp = live.last_self_penetration
        row = {"scene": f"visc_{name}", "scored": True, "particles": info.get("particles"),
               "substeps": info.get("substeps"), "steps": info.get("steps"), "device": live.device,
               "invalid_flags": lrm.decode_reason(live.invalid_reason), "detector": sp,
               "checks_ok": sp["checks"] > 0, "wall_s": time.perf_counter()-t0, "visc_protocol_sha256": vsha}
        row["pass"] = bool(row["checks_ok"] and live.invalid_reason == 0 and sp["hits"] == 0)
        note({"NO_FALSE_TRIP": row})
        nft.append(row)
    result["rows"]["NO_FALSE_TRIP"] = {"pass": all(r["pass"] for r in nft), "runs": nft}
    samples.append(gpu_sample(phys))
    others = sorted({a.split(",")[0].strip() for s in samples for a in s["apps"]} - {str(os.getpid())})
    meta.update({"co_tenant_samples": samples, "co_tenant_pids": others,
                 "gpu_label": "SHARED_GPU" if others else "EXCLUSIVE_AT_SAMPLES",
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    rows = result["rows"]
    result["verdict"] = {"L2": rows["L2"]["pass"], "TRIP": rows["TRIP"]["pass"],
                         "NO_FALSE_TRIP": rows["NO_FALSE_TRIP"]["pass"], "COST": "reported",
                         "check": "PASS" if rows["L2"]["pass"] and rows["TRIP"]["pass"]
                         and rows["NO_FALSE_TRIP"]["pass"] else "FAIL"}
    (out/"result.json").write_text(json.dumps(result, indent=1, default=float))
    note({"verdict": result["verdict"], "gpu_label": meta["gpu_label"], "co_tenant_pids": others})


if __name__ == "__main__":
    main()
