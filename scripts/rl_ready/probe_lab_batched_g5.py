"""G5 (ADR 0022 §8): material parity on identical edge and pad sequences.

The batched Lab run (``probe_lab_batched_gpu.py --material tlmpm_v7_2 --record-inputs --evaluate``) records per
env and step what its material received: the measured edge pose, the cut digit and the three edge-frame boxes
(handle + two measured finger pads), plus the initial edge/boxes and its final codes and local positions. This
script replays that sequence through the frozen single ``LiveCuttingMaterial`` (the class Native runs) with the
Native call order per step (``update_pad_obstacles`` then ``advance(edge, t, cutting, cut_index)``) and compares:

* ``--mode l0``  (CPU, ``--steps`` steps, default 20): single replay on the Warp CPU device against
  ``BatchedCuttingMaterial(backend="isaac_lab_newton", parity_mode=True)`` on CPU fed the same sequence through
  the Lab coupling surface (``set_measured_poses`` / in-place boxes / ``advance``): bitwise codes and positions.
* ``--mode l3``  (GPU, full sequence, ``--replays`` single replays per env): final codes and labelled counts of
  each single replay against the batched GPU run; the band of the single replays is reported alongside.

Usage::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src flock <gpu0.lock> .deps/isaacsim-venv/bin/python \
        scripts/rl_ready/probe_lab_batched_g5.py --mode l3 --run artifacts/rl_ready/p1_1b/<run> --envs 0 1 \
        --replays 3 --out artifacts/rl_ready/p1_1b/g5_l3_<run>_001
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def _task(seed):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(seed), goal_config(), seed=42)


def _single(task, device, initial_boxes, recipients):
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    config = CouplingConfig(control_dt=float(task.control_dt_s), material_dt=1e-4, dx=float(task.grid_dx_m),
                            device=device, feedback=True)
    m = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                            solver=task.solver_module, v7_2_optimizations=True)
    m.configure_pad_obstacles(np.asarray(initial_boxes, np.float32), recipients)
    return m


RECIPIENTS = [None, "left_finger_pad_body", "right_finger_pad_body"]


def replay_single(task, inputs, e, device, steps):
    m = _single(task, device, inputs["initial_boxes"][e], RECIPIENTS)
    m.reset(np.asarray(inputs["initial_edge"][e], float), int(inputs["seed"]))
    boxes = np.stack([inputs["box_centers"], inputs["box_halves"]], 2)   # [T, N*B, 2, 3]
    B = boxes.shape[1]//len(inputs["initial_edge"])
    dt = float(task.control_dt_s)
    for k in range(steps):
        m.update_pad_obstacles(boxes[k, e*B:(e+1)*B].astype(np.float32))
        cut = int(inputs["cut"][k, e])
        m.advance(np.asarray(inputs["edge"][k, e], float), (k+1)*dt, cutting=cut >= 0, cut_index=max(cut, 0))
    return m.bed.code.numpy().copy(), m.bed.x.numpy().copy()


def digit_counts(codes, max_cuts):
    codes = np.asarray(codes, np.int64)
    return {str(d): int(np.count_nonzero((codes >> (2*d)) & 3)) for d in range(max_cuts)}


def mode_l0(args, inputs, tasks):
    """CPU: batched material (Lab backend, parity mode) vs single, same sequence, bitwise."""
    import warp as wp
    from robotic_cutting_sim.batched import layout
    from robotic_cutting_sim.batched.material_batch import BatchedCuttingMaterial
    N = len(inputs["initial_edge"])
    specs = []
    for e in range(N):
        m = _single(tasks[e], "cpu", inputs["initial_boxes"][e], RECIPIENTS)
        specs.append(layout.EnvSpec(m, np.asarray(inputs["initial_edge"][e], float), int(inputs["seed"]),
                                    tasks[e].config_sha256))
    batch = BatchedCuttingMaterial(specs, device="cpu", parity_mode=True, backend="isaac_lab_newton")
    B = batch.solver.boxes
    for k in range(args.steps):
        batch.box_centers.assign(inputs["box_centers"][k])
        batch.box_halves.assign(inputs["box_halves"][k])
        batch.set_measured_poses(wp.array(np.asarray(inputs["edge"][k], np.float64), dtype=wp.float64, device="cpu"),
                                 wp.array(np.asarray(inputs["cut"][k], np.int32), dtype=wp.int32, device="cpu"))
        batch.advance()
    pk = batch.packed
    codes, x = batch.solver.code.numpy(), batch.solver.x.numpy()
    out = []
    for e in args.envs:
        sc, sx = replay_single(tasks[e], inputs, e, "cpu", args.steps)
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e+1])
        out.append(dict(env=e, steps=args.steps, codes_bitwise=bool(np.array_equal(sc, codes[p0:p1])),
                        positions_bitwise=bool(np.array_equal(sx, x[p0:p1])),
                        max_abs_position_diff_m=float(np.abs(sx-x[p0:p1]).max()),
                        labelled_single=int(np.count_nonzero(sc)), labelled_batch=int(np.count_nonzero(codes[p0:p1]))))
    return {"mode": "l0", "device": "cpu", "envs": out,
            "status": "PASS" if all(r["codes_bitwise"] and r["positions_bitwise"] for r in out) else "FAIL"}


def mode_l3(args, inputs, tasks, run):
    final = np.load(run/"final_codes.npz")
    T = len(inputs["edge"])
    out = []
    for e in args.envs:
        task = tasks[e]
        batch_codes, batch_x = final[f"env{e}"], final[f"x_local_env{e}"]
        replays = []
        for r in range(args.replays):
            start = time.perf_counter()
            codes, x = replay_single(task, inputs, e, args.device, T)
            replays.append(dict(replay=r, wall_s=time.perf_counter()-start, labelled=int(np.count_nonzero(codes)),
                                by_digit=digit_counts(codes, task.max_cuts),
                                codes_equal_batch=bool(np.array_equal(codes, batch_codes)),
                                different_code_particles=int(np.count_nonzero(codes != batch_codes)),
                                max_abs_position_diff_m=float(np.abs(x-batch_x).max()),
                                mean_abs_position_diff_m=float(np.linalg.norm(x-batch_x, axis=1).mean()),
                                codes=codes))
        labelled = [r["labelled"] for r in replays]
        pair = [int(np.count_nonzero(replays[i]["codes"] != replays[j]["codes"]))
                for i in range(len(replays)) for j in range(i+1, len(replays))]
        for r in replays:
            r.pop("codes")
        out.append(dict(env=e, scene_seed=int(inputs["scene_seeds"][e]), steps=T,
                        batch_labelled=int(np.count_nonzero(batch_codes)),
                        batch_by_digit=digit_counts(batch_codes, task.max_cuts), replays=replays,
                        single_band_labelled=[min(labelled), max(labelled)],
                        single_pairwise_different_code_particles=pair,
                        batch_inside_single_band=bool(min(labelled) <= int(np.count_nonzero(batch_codes)) <= max(labelled))))
    return {"mode": "l3", "device": args.device, "envs": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("l0", "l3"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    run = Path(args.run)
    inputs = dict(np.load(run/"material_inputs.npz"))
    seeds = [int(s) for s in inputs["scene_seeds"]]
    compiled = {s: _task(s) for s in dict.fromkeys(seeds)}
    tasks = [compiled[s] for s in seeds]
    start = time.perf_counter()
    result = mode_l0(args, inputs, tasks) if args.mode == "l0" else mode_l3(args, inputs, tasks, run)
    result.update(run=str(run), wall_s=time.perf_counter()-start, solver="tlmpm_v7_2 (frozen LiveCuttingMaterial)",
                  call_order="update_pad_obstacles(boxes[k]) -> advance(edge[k], (k+1)*dt, cutting=cut>=0, "
                             "cut_index=max(cut,0))")
    (out/"g5.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
