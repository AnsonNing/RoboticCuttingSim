"""P2-4 small GPU checks (RUBBER-CONT-11 cost, reported; single-env GPU invariants of the contact solver). Not
RUBBER-CONT-09 (that needs the batched contact path: NOT_RUN).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/gpu_contact_check.py \
        --out artifacts/rl_ready/p2_4/<dir>

Rows (single env, RTX GPU0, CUDA graph chunks of 500 substeps, workload E geometry: case A clamp 150 x 20 x 4 mm, n_t 3
(rubber_rl_v1), 50 Hz, gravity -z, elastic mu0 0.4 MPa, beta 20, rho 1150, the protocol's options F8 + kick-average +
F9 + F10, guard 1.0):

- ``no_contact``: LiveRubberMaterial without a ContactSpec (TLMPMRubberCW, frozen launches);
- ``contact``: ContactSpec with the table plane z = -0.02 m and two pad frames (1 box each) pinching the tip region
  (squeeze 20 um, held), law grid_node_penalty_coulomb_v1 defaults;
- ``contact_out_of_reach``: the same spec translated out of reach (plane z = -1 m, pads 1 m away): every contact row
  (plane / pad wrenches, contact work, penetration, events) must be exactly 0.0 on the GPU.

Per row: build + capture, 1 cold + ``--warm`` warm control steps (median), ms per substep, VRAM (nvidia-smi process
peak, cumulative), L2 invariants (finite, invalid bits 0, guard violations 0). Stage split of the ``contact`` row:
``--profile-substeps`` eager substeps under ``wp.timing_begin`` (kernel time). A GPU vs CPU comparison of a small pinch
scene (10 control steps) is reported (frozen P2G/MUSL float atomics are order-nondeterministic on the GPU).
Co-tenant samples before / after; SHARED_GPU whenever another process is present. Refuses to run with < 4 GB free.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v1.json"
MIN_FREE_MB = 4096


def free_mb(index=0):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--profile-substeps", type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = int(vis.split(",")[0]) if vis.strip() else 0
    fm = free_mb(phys)
    if fm < MIN_FREE_MB:
        sys.exit(f"GPU{phys}: {fm:.0f} MB free < {MIN_FREE_MB} MB; not running")
    out.mkdir(parents=True)
    import warp as wp
    from gpu_smoke_rubber import Peak, gpu_sample
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_contact import ContactFrame, ContactSpec
    from robotic_cutting_sim.rubber_material import MaterialValues
    wp.init()
    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa
    meta = {"label": "RUBBER-CONT-11 (reported, not scored) + single-env GPU invariants; RUBBER-CONT-09 NOT_RUN",
            "command": " ".join(sys.argv), "cuda_visible_devices": vis, "device": args.device,
            "device_name": wp.get_device(args.device).name,
            "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip().splitlines(),
            "pid": os.getpid(), "commit": git("rev-parse", "HEAD"),
            "dirty_tree": bool(git("status", "--porcelain", "--untracked-files=no")),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "warp": wp.config.version,
            "free_mb_at_start": fm, "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED"}
    peak = Peak()
    opts = dict(dirichlet="consistent_mass_projection_v1", constraint_work="kick_average_v1",
                traction_work="grid_displacement_v1", state_precision="f64_shadow_v1", stable_step_guard=1.0)
    m = MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=0.8, friction_pad=1.0)
    L, W, t = 0.15, 0.02, 0.004
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=3, case="A", resolution_profile="rubber_rl_v1")

    def pads(shift):
        top = ContactFrame(kind="pad", name="top", origin=(L-0.005+shift, W/2, t+0.001),
                           boxes=(((L-0.005+shift, W/2, t+0.001), (0.006, W/2+0.002, 0.001)),))
        bot = ContactFrame(kind="pad", name="bottom", origin=(L-0.005+shift, W/2, -0.001),
                           boxes=(((L-0.005+shift, W/2, -0.001), (0.006, W/2+0.002, 0.001)),))
        return (top, bot)

    variants = {"no_contact": None,
                "contact": ContactSpec(planes=((0.0, 0.0, 1.0, 0.02),), frames=pads(0.0)),
                "contact_out_of_reach": ContactSpec(planes=((0.0, 0.0, 1.0, 1.0),), frames=pads(1.0))}
    rows = []
    for name, spec_c in variants.items():
        t0 = time.perf_counter()
        live = LiveRubberMaterial(spec, material=m, control_dt=0.02, device=args.device, gravity=(0, 0, -9.81),
                                  strict=False, contact=spec_c, **opts)
        build = time.perf_counter()-t0
        peak.update()
        frames = {0: {"translation": np.array([0.0, 0.0, -2e-5])}, 1: {"translation": np.array([0.0, 0.0, 2e-5])}} \
            if spec_c is not None else None
        walls = []
        zero_rows_ok = None
        contact_rows = []
        for k in range(1+args.warm):
            wp.synchronize_device(args.device)
            t1 = time.perf_counter()
            live.advance(frames=frames) if frames else live.advance()
            wp.synchronize_device(args.device)
            walls.append(time.perf_counter()-t1)
            peak.update()
            if spec_c is not None:
                c = live.last_contact
                vals = np.concatenate([np.asarray(c["plane_wrench"]).ravel(), np.asarray(c["box_wrench"]).ravel(),
                                       [c[k2] for k2 in ("normal_work_material", "tangential_work_material",
                                                         "max_penetration_m", "impulse_events")]])
                contact_rows.append({"max_abs_contact_row": float(np.abs(vals).max()),
                                     "pad_force_z": [float(c["box_wrench"][0, 0, 2]), float(c["box_wrench"][1, 0, 2])],
                                     "events": int(c["impulse_events"])})
                if name == "contact_out_of_reach":
                    ok = bool(np.all(vals == 0.0)) and bool(np.all(live.solver.ct_jn.numpy() == 0.0))
                    zero_rows_ok = ok if zero_rows_ok is None else (zero_rows_ok and ok)
        d = live.last_diagnostics
        warm = float(np.median(walls[1:])) if len(walls) > 1 else walls[0]
        row = {"variant": name, "particles": int(live.scene.n), "nodes": int(live.solver.node_count),
               "substeps": live.substeps, "dt_s": live.dt, "build_capture_s": build, "cold_s": walls[0],
               "warm_s": warm, "ms_per_substep": 1e3*warm/live.substeps, "env_steps_per_s": 1.0/warm,
               "vram_peak_mb_cumulative": peak.value, "graph": live.capture_receipt,
               "l2": {"finite": d["nonfinite"] == 0.0, "invalid_reason": int(live.invalid_reason),
                      "guard_violations": int(live.solver.cfl_violations.numpy()[0]), "cfl_eff_max": d["cfl_eff_max"]},
               "contact_rows": contact_rows, "zero_rows_exact": zero_rows_ok}
        row["l2"]["finite"] = bool(row["l2"]["finite"])
        row["l2"]["pass"] = row["l2"]["finite"] and row["l2"]["invalid_reason"] == 0 and row["l2"]["guard_violations"] == 0
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("contact_rows", "graph")}, default=float), flush=True)
        if name == "contact":
            # stage split: eager substeps under Warp kernel timing (the split, not the throughput)
            s = live.solver
            s.begin_control_step()
            wp.synchronize_device(args.device)
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
            for _ in range(args.profile_substeps):
                s.step()
            wp.synchronize_device(args.device)
            res = wp.timing_end()
            agg = {}
            for r in res:
                key = r.name.split("(")[0] if hasattr(r, "name") else str(r)
                agg[key] = agg.get(key, 0.0)+float(r.elapsed)
            row["stage_split_ms_per_substep"] = {k: v/args.profile_substeps for k, v in
                                                 sorted(agg.items(), key=lambda kv: -kv[1])}
        del live
    # GPU vs CPU on a small pinch scene (reported)
    small = rs.RubberStripSpec(length=0.012, width=0.004, thickness=0.002, n_t=2, case="A")
    sp = ContactSpec(frames=(ContactFrame(kind="pad", origin=(0.009, 0.002, 0.0025),
                                          boxes=(((0.009, 0.002, 0.0025), (0.003, 0.003, 0.0005)),)),
                             ContactFrame(kind="pad", origin=(0.009, 0.002, -0.0005),
                                          boxes=(((0.009, 0.002, -0.0005), (0.003, 0.003, 0.0005)),))))
    comp = {}
    for dev in ("cpu", args.device):
        live = LiveRubberMaterial(small, material=m, control_dt=5e-4, device=dev, gravity=(0, 0, -9.81), strict=False,
                                  contact=sp, capture_graph=False, **opts)
        for k in range(10):
            r_ = min(1.0, (k+1)/5)
            live.advance(frames={0: {"translation": np.array([1e-4*k, 0.0, -2e-5*r_])},
                                 1: {"translation": np.array([1e-4*k, 0.0, 2e-5*r_])}})
        comp[dev] = {"box_wrench": np.asarray(live.last_contact["box_wrench"]).tolist(),
                     "markers": live.markers().tolist(), "W_t": live.last_contact["tangential_work_material"]}
    bc, bg = np.asarray(comp["cpu"]["box_wrench"]), np.asarray(comp[args.device]["box_wrench"])
    mc, mg = np.asarray(comp["cpu"]["markers"]), np.asarray(comp[args.device]["markers"])
    gpu_cpu = {"pad_wrench_max_rel_diff": float(np.abs(bg-bc).max()/max(np.abs(bc).max(), 1e-300)),
               "marker_max_abs_diff_m": float(np.abs(mg-mc).max()), "values": comp,
               "note": "reported, not scored: frozen P2G/MUSL and the contact reactions use float atomics, "
                       "order-nondeterministic on the GPU"}
    meta["co_tenant_after"] = gpu_sample()
    others = [a for a in meta["co_tenant_before"]["apps"]+meta["co_tenant_after"]["apps"]
              if a and int(a.split(",")[0]) != os.getpid()]
    meta["shared_gpu"] = bool(others)
    meta["label_gpu"] = "SHARED_GPU" if others else "EXCLUSIVE_AT_SAMPLES"
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out/"result.json").write_text(json.dumps({"meta": meta, "rows": rows, "gpu_vs_cpu": gpu_cpu}, indent=1,
                                              default=float))
    print(json.dumps({"gpu_vs_cpu": {k: v for k, v in gpu_cpu.items() if k != "values"}, "shared": meta["shared_gpu"]}))


if __name__ == "__main__":
    main()
