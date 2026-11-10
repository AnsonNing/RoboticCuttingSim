"""P1-1b B2 follow-up: material robustness against out-of-domain edge / pad inputs (the post-grasp-loss SIGSEGV).

Material only (no robot): ``BatchedCuttingMaterial(backend="isaac_lab_newton")`` over seeded TRIM scenes, driven by
the measured edge / cut / pad-box sequence recorded by an S3 Lab run (``material_inputs.npz``), with one env's
input replaced from ``--bad-from`` on by a synthetic out-of-domain scenario:

* ``none``     the recorded inputs unchanged (control);
* ``far``      the edge teleports by +1.43 m in x per control step (the deterministic-mode B2 tool ejection);
* ``huge``     the edge jumps to (1e7, 1e7, 1e7) m;
* ``spin``     the edge spins 2.5 rad per control step about z while sinking 10 mm into the workpiece per step;
* ``sweep``    the edge sweeps +-0.25 m in x through the workpiece every control step (lost-grasp swing);
* ``nanpose``  a non-finite edge pose;
* ``hugebox``  the bad env's pad boxes are moved 1e7 m away.

Every env's per-step invalid reasons, particle extent and finiteness are recorded; with ``--isolation`` a second
batch with the clean inputs everywhere runs alongside and the other envs' state must stay bitwise equal (CPU) or
is reported (GPU). One build per process; a crash (SIGSEGV) leaves a faulthandler stack in the log.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rl_ready/probe_material_ood.py --device cpu \
        --scenario far --steps 6 --out artifacts/rl_ready/p1_1b_b2/ood_cpu_far_001
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

INPUTS = "artifacts/rl_ready/p1_1b/s3_b2_n2_001/material_inputs.npz"


def _commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return {"sha": sha, "tracked_src_or_scripts_modified": bool(dirty)}
    except Exception:
        return None


def build(seeds, spacing, device, strict=False, start=0):
    from robotic_cutting_sim.batched import layout
    from robotic_cutting_sim.batched.material_batch import BatchedCuttingMaterial
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    rec = np.load(INPUTS)
    specs = []
    for e, seed in enumerate(seeds):
        task = compile_task(generate_scene(seed), goal_config(), seed=42,
                            **({} if spacing is None else {"particle_spacing_m": spacing}))
        config = CouplingConfig(control_dt=float(task.control_dt_s), material_dt=1e-4, dx=float(task.grid_dx_m),
                                device="cpu", feedback=True)
        m = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                solver=task.solver_module, v7_2_optimizations=True)
        # the batch starts at the recorded state of step start-1 (else the first step would jump to step `start`)
        if start > 0:
            B = rec["box_centers"].shape[1]//2
            boxes = np.stack([rec["box_centers"][start-1].reshape(2, B, 3)[e % 2],
                              rec["box_halves"][start-1].reshape(2, B, 3)[e % 2]], 1)
            edge = rec["edge"][start-1, e % 2]
        else:
            boxes, edge = rec["initial_boxes"][e % 2], rec["initial_edge"][e % 2]
        m.configure_pad_obstacles(np.asarray(boxes, np.float32), [None, "left", "right"])
        specs.append(layout.EnvSpec(m, np.asarray(edge, float), 42, task.config_sha256))
    return BatchedCuttingMaterial(specs, device=device, parity_mode=device == "cpu", strict=strict,
                                  backend="isaac_lab_newton")


def scenario_inputs(name, step, bad_from, edge, boxes):
    """Modify one env's (edge [7], boxes [B,2,3]) at ``step`` (copies)."""
    edge, boxes = edge.copy(), boxes.copy()
    k = step-bad_from+1
    if k <= 0 or name == "none":
        return edge, boxes
    if name == "far":
        edge[0] += 1.43*k
    elif name == "huge":
        edge[:3] = 1e7
    elif name == "spin":
        from scipy.spatial.transform import Rotation
        edge[3:] = (Rotation.from_rotvec([0., 0., 2.5*k])*Rotation.from_quat(edge[3:])).as_quat()
        edge[2] -= .01*k
    elif name == "sweep":
        edge[0] += .25*(-1)**k
        edge[2] -= .01
    elif name == "nanpose":
        edge[:] = np.nan
    elif name == "hugebox":
        boxes[:, 0] += 1e7
    else:
        raise ValueError(name)
    return edge, boxes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--scenario", default="none")
    p.add_argument("--seeds", type=int, nargs="*", default=[512, 15])
    p.add_argument("--spacing", type=float, default=.0025, help="particle spacing (None: the task default 1 mm)")
    p.add_argument("--default-spacing", action="store_true")
    p.add_argument("--start", type=int, default=150, help="first recorded input step used")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--bad-env", type=int, default=1)
    p.add_argument("--bad-from", type=int, default=2)
    p.add_argument("--isolation", action="store_true", help="also run a clean batch and compare the other envs")
    p.add_argument("--debug", action="store_true", help="Warp debug mode (bounds checks; CPU)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cpu = args.device == "cpu"
    if cpu and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("CPU run: set CUDA_VISIBLE_DEVICES=''")
    import warp as wp
    if args.debug:
        wp.config.mode = "debug"
    device = "cpu" if cpu else "cuda:0"
    spacing = None if args.default_spacing else args.spacing
    row = {"probe": "material_ood", "initial_state": "recorded edge/boxes of step start-1 (since 2026-11-10 10:30; "
           "earlier rows started from the recorded initial edge, so their first step jumped to step `start`)", "device": device, "scenario": args.scenario, "seeds": args.seeds,
           "spacing": spacing, "bad_env": args.bad_env, "bad_from": args.bad_from, "start": args.start,
           "commit": _commit(), "command": " ".join(sys.argv), "inputs": INPUTS, "steps": []}
    rec = np.load(INPUTS)
    N = len(args.seeds)
    t0 = time.perf_counter()
    batches = {"run": build(args.seeds, spacing, device, start=args.start)}
    if args.isolation:
        batches["clean"] = build(args.seeds, spacing, device, start=args.start) if cpu else None
    row["build_s"] = time.perf_counter()-t0
    row["particles"] = [int(x) for x in batches["run"].packed.particle_counts]
    row["graph"] = batches["run"].graph is not None
    (out/"row.json").write_text(json.dumps(row, indent=1, default=str))
    B = batches["run"].solver.boxes
    for i in range(args.steps):
        step = args.start+i
        edges = np.stack([rec["edge"][step, e % 2] for e in range(N)]).astype(np.float64)
        cuts = np.stack([rec["cut"][step, e % 2] for e in range(N)]).astype(np.int32)
        boxes = np.stack([np.stack([rec["box_centers"][step].reshape(2, B, 3)[e % 2],
                                    rec["box_halves"][step].reshape(2, B, 3)[e % 2]], 1) for e in range(N)])
        clean_edges, clean_boxes = edges.copy(), boxes.copy()
        edges[args.bad_env], boxes[args.bad_env] = scenario_inputs(args.scenario, i, args.bad_from,
                                                                   edges[args.bad_env], boxes[args.bad_env])
        entry = {"i": i, "step": step, "bad_edge": edges[args.bad_env].tolist()}
        for name, m in batches.items():
            if m is None:
                continue
            e_in, b_in = (edges, boxes) if name == "run" else (clean_edges, clean_boxes)
            m.solver.box_centers.assign(np.ascontiguousarray(b_in[:, :, 0]).reshape(-1, 3).astype(np.float32))
            m.solver.box_halves.assign(np.ascontiguousarray(b_in[:, :, 1]).reshape(-1, 3).astype(np.float32))
            m.set_measured_poses(wp.array(e_in, dtype=wp.float64, device=device),
                                 wp.array(cuts, dtype=wp.int32, device=device))
            t = time.perf_counter()
            m.advance()
            wp.synchronize_device(device)
            x = m.solver.x.numpy()
            off = m.packed.particle_offset
            entry[name] = {"wall_s": time.perf_counter()-t, "invalid_reason": m.invalid_reason.numpy().tolist(),
                           "pose_status": m.pose_status.numpy().tolist(),
                           "max_abs_x": [float(np.nanmax(np.abs(x[off[e]:off[e+1]]))) for e in range(N)],
                           "finite_x": [bool(np.isfinite(x[off[e]:off[e+1]]).all()) for e in range(N)],
                           "pending_wrench": m.pending_wrench.numpy().tolist()}
        if args.isolation and batches.get("clean") is not None:
            a, c = batches["run"], batches["clean"]
            entry["others_bitwise"] = {}
            for e in range(N):
                if e == args.bad_env:
                    continue
                sa, sc = a.snapshot(e), c.snapshot(e)
                diff = [k for k in sa if isinstance(sa[k], np.ndarray) and not np.array_equal(sa[k], sc[k])]
                for name in ("pending_wrench", "pad_wrench", "invalid_reason", "step_count", "last_pose"):
                    if not np.array_equal(getattr(a, name).numpy()[e], getattr(c, name).numpy()[e]):
                        diff.append("material."+name)
                entry["others_bitwise"][str(e)] = diff
        row["steps"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "bad_edge"}, default=str)[:900], flush=True)
        (out/"row.json").write_text(json.dumps(row, indent=1, default=str))
    row["status"] = "COMPLETED"
    (out/"row.json").write_text(json.dumps(row, indent=1, default=str))


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
    main()
