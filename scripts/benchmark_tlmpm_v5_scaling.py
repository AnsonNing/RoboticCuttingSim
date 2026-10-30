#!/usr/bin/env python3
"""ADR 0014 scaling table: memory and substep wall time of tlmpm_v5 vs tlmpm_v4 on the 1 mm CUT-01-R bar.

For each (solver, K) the CUT-01-R kinematic material (law 8, 1 mm particles,
4 mm grid, dt 1e-4 s) is reset and given a preset K-cut history: K planes
x = x_k evenly spread over the bar, and every particle within the law-4 band
h = 6 dx of plane k gets digit k (1 below, 2 above x_k), the labels a swept
band would leave. The edge then stays parked outside the material
(cutting disabled) while timed control steps run the captured CUDA graph
(200 substeps each). Reported: solver device-array bytes, mempool growth, mean
substep wall time, and for v5 the occupied-slot maximum and overflow. This
measures representation cost only; it is not a cutting result.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/benchmark_tlmpm_v5_scaling.py --device cuda:0 \
      --out artifacts/wp11/phaseM/scaling_v5_001
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.cut_topology import pack_digits  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402


def preset_digits(local_points, cuts, band):
    x = local_points[:, 0]
    planes = [0.] if cuts == 1 else list(np.linspace(-.06, .06, cuts))
    digits = np.zeros((len(x), cuts), np.int64)
    for k, plane in enumerate(planes):
        near = np.abs(x - plane) <= band
        digits[near, k] = np.where(x[near] < plane, 1, 2)
    return digits, planes


def device_bytes(bed):
    import warp as wp
    return int(sum(v.capacity for v in vars(bed).values() if isinstance(v, wp.array)))


def measure(solver, cuts, args):
    import warp as wp
    from robotic_cutting_sim.kinematic_env import KinematicConfig, KinematicCuttingBackend
    wp.synchronize_device(args.device)
    before = wp.get_mempool_used_mem_current(args.device)
    backend = KinematicCuttingBackend(KinematicConfig('CUT-01-R', seed=42, particle_spacing_m=.001, material_dt=1e-4,
                                                      device=args.device, cutting_law=8, max_cuts=cuts,
                                                      **({'solver': 'tlmpm_v5'} if solver == 'v5' else {})))
    receipt = backend.reset(seed=42)
    material, bed = backend.material, backend.material.bed
    wp.synchronize_device(args.device)
    after = wp.get_mempool_used_mem_current(args.device)
    digits, planes = preset_digits(material.points, cuts, bed.law4_parameters['band_h_m'])
    codes = pack_digits(digits) if solver == 'v5' else (digits * 3 ** np.arange(cuts)).sum(1)
    bed.code.assign(codes.astype(np.int32))
    edge = backend.edge.copy()
    walls = []
    for step in range(args.warmup + args.steps):
        diagnostics = material.advance(edge, round(material.time_s + material.config.control_dt, 9), cutting=False)
        if step >= args.warmup:
            walls.append(diagnostics['material_wall_s'])
    substeps = material.substeps
    row = {'solver': solver, 'solver_module': type(bed).__module__, 'max_cuts': cuts, 'particles': int(bed.n),
           'grid_nodes': int(bed.node_count), 'fields_per_node': int(bed.slots_per_node if solver == 'v5' else 3 ** cuts),
           'solver_device_array_bytes': device_bytes(bed), 'mempool_growth_bytes': int(after - before),
           'timed_control_steps': args.steps, 'substeps_per_control_step': int(substeps),
           'substep_wall_mean_ms': float(np.sum(walls) / (len(walls) * substeps) * 1e3),
           'control_step_wall_median_ms': float(np.median(walls) * 1e3),
           'preset_planes_local_x_m': [float(p) for p in planes], 'preset_labelled_particles': int(np.count_nonzero(codes)),
           'reset_wall_s': float(receipt['reset_wall_s'])}
    if solver == 'v5':
        row.update(bed.check_slots())
    backend.close()
    del bed, material, backend
    import gc
    gc.collect()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--v5-cuts', type=int, nargs='*', default=[1, 2, 3, 5, 8])
    p.add_argument('--v4-cuts', type=int, nargs='*', default=[1, 2])
    a = p.parse_args()
    out = new_output(a.out)
    write_json(out / 'manifest.json', manifest({**vars(a), 'out': str(a.out)}, 'tlmpm_v5_scaling'))
    import warp as wp
    wp.init()
    start, rows = time.monotonic(), []
    for solver, cuts in [('v4', k) for k in a.v4_cuts] + [('v5', k) for k in a.v5_cuts]:
        try:
            row = measure(solver, cuts, a)
        except Exception as error:  # preserve the failure in the table
            row = {'solver': solver, 'max_cuts': cuts, 'status': 'ERROR', 'error': repr(error)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_json(out / 'result.json', {'rows': rows, 'command': sys.argv, 'wall_s': time.monotonic() - start,
                                     'scope': 'Representation cost on the 1 mm CUT-01-R bar with a preset K-cut history; '
                                              'edge parked, cutting disabled; GPU timings include graph launch + sync'})


if __name__ == '__main__':
    main()
