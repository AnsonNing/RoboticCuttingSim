#!/usr/bin/env python3
"""P1-3a probe: does a CUDA graph that rebuilds HashGrid A replay with the descriptor of the last HashGrid built
on the host (Warp 1.17.0 ``hash_grid_update_device_impl`` copies the grid descriptor from a function-static host
variable, so the captured memcpy node reads whatever that variable holds at replay time)?

Grid A: points in a 1 cm cube around the origin. Grid B: the same number of points 1 m away. A graph captures
``A.build`` + a kernel that counts, for every A point, the candidates of a query at that point on ``A.id``.
Eager counts give the reference. Then B is built eagerly (host), and the graph is replayed: if the replay uses
A's own descriptor the counts are unchanged; if it uses B's (stale static descriptor) the counts change (B's
points are far away -> 0 candidates). Finally A is built eagerly again and the graph replayed once more.

Run under the GPU lock: ``CUDA_VISIBLE_DEVICES=<gpu> TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python
scripts/rl_ready/probe_hashgrid_descriptor.py <out.json>``.
"""
import json
import sys

import numpy as np
import warp as wp


@wp.kernel
def count(grid: wp.uint64, x: wp.array(dtype=wp.vec3), radius: float, out: wp.array(dtype=int)):
    p = wp.tid()
    n = int(0)
    for q in wp.hash_grid_query(grid, x[p], radius):
        n += 1
    out[p] = n


def main(path):
    wp.init()
    dev = "cuda:0"
    rng = np.random.default_rng(0)
    n, h = 20000, 0.001
    xa = wp.array(rng.uniform(-.005, .005, (n, 3)).astype(np.float32), dtype=wp.vec3, device=dev)
    xb = wp.array((rng.uniform(-.005, .005, (n, 3)) + 1.0).astype(np.float32), dtype=wp.vec3, device=dev)
    a = wp.HashGrid(64, 64, 32, device=dev)
    b = wp.HashGrid(189, 69, 66, device=dev)
    a.reserve(n)
    b.reserve(n)
    out = wp.zeros(n, dtype=int, device=dev)
    a.build(xa, h)
    wp.launch(count, n, [a.id, xa, h, out], device=dev)
    eager = out.numpy().copy()
    with wp.ScopedCapture(device=dev) as capture:
        a.build(xa, h)
        wp.launch(count, n, [a.id, xa, h, out], device=dev)
    rows = {}
    out.zero_()
    wp.capture_launch(capture.graph)
    rows["replay_after_capture"] = out.numpy().copy()
    b.build(xb, h)                      # another grid built on the host after the capture
    out.zero_()
    wp.capture_launch(capture.graph)
    rows["replay_after_host_build_of_other_grid"] = out.numpy().copy()
    a.build(xa, h)                      # this grid built on the host again
    out.zero_()
    wp.capture_launch(capture.graph)
    rows["replay_after_host_build_of_same_grid"] = out.numpy().copy()
    result = dict(warp=wp.config.version, device=wp.get_device(dev).name, points=n,
                  eager_candidates_total=int(eager.sum()),
                  **{k: dict(candidates_total=int(v.sum()), equal_to_eager=bool(np.array_equal(v, eager)))
                     for k, v in rows.items()})
    result["stale_descriptor_hazard"] = (result["replay_after_capture"]["equal_to_eager"]
                                         and not result["replay_after_host_build_of_other_grid"]["equal_to_eager"])
    with open(path, "x") as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
