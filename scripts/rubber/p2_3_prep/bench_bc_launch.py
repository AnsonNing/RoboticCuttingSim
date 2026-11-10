"""P2-3 prerequisites (ADR 0023 §20.2): interleaved GPU micro-benchmark of the rubber BC pass, all-node vs list launch.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
        PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_3_prep/bench_bc_launch.py \
        --out artifacts/rl_ready/p2_3_prep/<id>

Workload E batch (N = 4 by default, v1 options, guard 1.0, eager; no graph). Per round, in an order rotated every round:
the BC pair of one substep (record 1 then record 0) with ``_bc_b`` over every node, the same pair with ``_bc_list_b``
over the constrained-node list, and a one-thread kernel (``tlmpm_v5._advance`` on a scratch counter) as the
launch / scheduling floor. GPU execution time per launch from CUDA activity timing (``wp.timing_begin``); medians and
percentiles over ``--rounds`` rounds. Also ``--chain`` back-to-back pairs of each launch timed by wall clock after a
synchronize (throughput view). The group motions are a nonzero rigid motion (slerp path exercised). No cameras, no
recording. SHARED_GPU whenever another process is present.
"""
import argparse
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
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--chain", type=int, default=500)
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
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim import tlmpm_v5 as v5
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_material import MaterialValues
    wl = json.loads((ROOT/"configs"/"rubber"/"workload_e_v1.json").read_text())["workloads"][0]["material"]
    g, rnd = wl["geometry_m"], wl["randomisation"]
    step_mat = MaterialValues(mu0=rnd["shear_modulus_eq_pa"][1], kappa=float(wl["bulk_ratio_beta"])
                              * rnd["shear_modulus_eq_pa"][1], density=rnd["density_kg_m3"][0])
    spec = rs.RubberStripSpec(length=g["length"], width=g["width"], thickness=g["thickness"], n_t=wl["n_t"], case="A",
                              resolution_profile=wl["resolution_profile"])
    meta = {"command": " ".join(sys.argv), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": wp.get_device(args.device).name, "pid": os.getpid(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(), "warp": wp.config.version,
            "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    lives = [LiveRubberMaterial(spec, control_dt=wl["control_dt_s"], device="cpu", step_material=step_mat,
                                stable_step_guard=1.0) for _ in range(args.N)]
    batch = BatchedRubberMaterial(lives, device=args.device, strict=False, capture_graph=False)
    s = batch.solver
    batch.begin_control_step()
    R = np.tile(Rotation.from_rotvec([0.0, 0.05, 0.0]).as_matrix(), (args.N, 1, 1, 1))
    batch.set_group_motions(rotation=R, translation=np.full((args.N, 1, 3), 1e-4))
    s.step()                                            # nonzero state
    counter = wp.zeros(1, dtype=int, device=args.device)

    def pair(mode):
        s.bc_launch = mode
        s._launch_bc(1)
        s._launch_bc(0)

    def floor():
        wp.launch(v5._advance, 1, [counter], device=args.device)

    labels = {"all_nodes_v1": lambda: pair("all_nodes_v1"), "constrained_list_v1": lambda: pair("constrained_list_v1"),
              "one_thread_x2": lambda: (floor(), floor())}
    for fn in labels.values():                         # warm-up (module load, first launch)
        fn()
    wp.synchronize_device(args.device)
    per = defaultdict(list)
    order = list(labels)
    for r in range(args.rounds):
        rot = order[r % 3:]+order[:r % 3]
        for name in rot:
            wp.synchronize_device(args.device)
            wp.timing_begin()
            labels[name]()
            wp.synchronize_device(args.device)
            res = wp.timing_end()
            per[name].append(sum(x.elapsed for x in res))
    chain = {}
    for name in order*2:
        wp.synchronize_device(args.device)
        t = time.perf_counter()
        for _ in range(args.chain):
            labels[name]()
        wp.synchronize_device(args.device)
        chain.setdefault(name, []).append(1e3*(time.perf_counter()-t)/args.chain)
    stats = {name: {"gpu_ms_per_pair_median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                    "p90": float(np.percentile(v, 90)), "min": float(np.min(v)), "rounds": len(v)}
             for name, v in per.items()}
    result = {"N": args.N, "particles_per_env": int(batch.packed.particle_counts[0]),
              "nodes_total": s.total_nodes, "bc_nodes_total": s.bc_node_count, "substeps_per_control_step": s.substeps,
              "material_dt_s": s.dt, "control_hz": 50, "cameras": "none", "recording": "none", "graph": "none (eager)",
              "activity_timing": stats, "wall_ms_per_pair_chain": chain}
    meta["co_tenant_after"] = gpu_sample()
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    others = [a for x in (meta["co_tenant_before"], meta["co_tenant_after"]) for a in x["apps"]
              if a.split(",")[0].strip() != str(os.getpid())]
    meta["gpu_sharing"] = "SHARED_GPU" if others else "EXCLUSIVE"
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    (out/"result.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"sharing": meta["gpu_sharing"], **result}, indent=1, default=float))


if __name__ == "__main__":
    main()
