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

Review fix F1 modes (``PYTHONPATH=src``, one process each, ``--out`` a new directory under artifacts/):

* ``--mode batch-foreign --variant clean|foreign``: a frozen_v1 batch (``open_vector_env``, ``strict=False``, N =
  ``--copies`` copies of seed ``--seed``) is loaded with the recorded Native state of sample ``--sample`` and runs
  one control step through its CUDA graph (``material.advance(interpolation="preset")``). Recorded: per-env
  candidate counts of a graph that mirrors the batch's grouped build + query on the batch's own grid (replayed after
  ``refresh_device_hashgrid()``), ``stats_sum[:, 4]`` (contact count), ``stats_max[:, 6]``, the impulse array and the
  codes after the step. ``foreign`` builds a ``HashGrid(64, 64, 32)`` (a frozen single material's dims) on far-away
  points AFTER the batch, then repeats the step; then, from the same state, once more with the refresh disabled
  (``unprotected``, the pre-fix default path) to show the hazard. Compare with ``--mode compare``.
* ``--mode refresh-cost --num-envs N --refresh on|off``: the P1-3a solo protocol (``run_p1_3a.mode_ab``, frozen_v1,
  steps 3-14 of the WP-17 replay, env k seed 512 / 15 alternating) with the descriptor refresh on (default code) or
  replaced by a no-op (``off``, pre-fix behaviour); ``on`` also times every refresh call (host wall without sync and
  CUDA-event GPU time). One variant per process, alternate on/off.
* ``--mode compare --clean DIR --foreign DIR --out FILE``: contact equality of the two processes.
* ``--mode replay-race --out FILE``: does a replay read the static descriptor at launch or when its memcpy node
  executes (a host build of another grid while the replay still runs)?
"""
import argparse
import json
from pathlib import Path
import sys
import time

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


# ----------------------------------------------------------------------------- review fix F1 modes

@wp.kernel
def spin(iterations: int, out: wp.array(dtype=float)):
    acc = float(0.0)
    for i in range(iterations):
        acc = acc * 0.9999999 + 1.0e-7
    out[0] = acc


def race(path, iterations):
    """When does a replayed graph read the static descriptor: at launch (host) or when the memcpy node executes?
    Graph = [spin kernel (~0.1-1 s), A.build, count on A]. Launch it, build B on the host while the spin kernel still
    runs, then synchronise. Counts equal to eager: read at launch; 0: read at execution (a host build during an
    asynchronous replay corrupts that replay)."""
    import numpy as np
    dev = "cuda:0"
    rng = np.random.default_rng(0)
    n, h = 20000, 0.001
    xa = wp.array(rng.uniform(-.005, .005, (n, 3)).astype(np.float32), dtype=wp.vec3, device=dev)
    xb = wp.array((rng.uniform(-.005, .005, (n, 3)) + 1.0).astype(np.float32), dtype=wp.vec3, device=dev)
    a, b = wp.HashGrid(64, 64, 32, device=dev), wp.HashGrid(189, 69, 66, device=dev)
    a.reserve(n)
    b.reserve(n)
    out, sink = wp.zeros(n, dtype=int, device=dev), wp.zeros(1, dtype=float, device=dev)
    a.build(xa, h)
    wp.launch(count, n, [a.id, xa, h, out], device=dev)
    eager = out.numpy().copy()
    with wp.ScopedCapture(device=dev) as capture:
        wp.launch(spin, 1, [iterations, sink], device=dev)
        a.build(xa, h)
        wp.launch(count, n, [a.id, xa, h, out], device=dev)
    rows = {}
    for label, other in (("idle_host", False), ("host_build_of_other_grid_during_replay", True),
                         ("idle_host_again", False)):
        a.build(xa, h)                                   # A's descriptor in the static before the launch
        wp.synchronize_device(dev)
        out.zero_()
        wp.synchronize_device(dev)
        t0 = time.perf_counter()
        wp.capture_launch(capture.graph)
        t1 = time.perf_counter()
        if other:
            b.build(xb, h)                               # static now holds B while the spin kernel runs
        t2 = time.perf_counter()
        wp.synchronize_device(dev)
        t3 = time.perf_counter()
        v = out.numpy()
        rows[label] = dict(candidates_total=int(v.sum()), equal_to_eager=bool(np.array_equal(v, eager)),
                           launch_host_ms=(t1 - t0) * 1e3, host_build_ms=(t2 - t1) * 1e3, wait_ms=(t3 - t2) * 1e3)
    during = rows["host_build_of_other_grid_during_replay"]
    result = dict(warp=wp.config.version, device=wp.get_device(dev).name, points=n, spin_iterations=iterations,
                  eager_candidates_total=int(eager.sum()), rows=rows,
                  graph_still_running_at_host_build=during["wait_ms"] > 5.0,
                  descriptor_read=("at_execution (host build during an asynchronous replay corrupts it)"
                                   if not during["equal_to_eager"] else "at_launch (snapshot)"))
    with open(path, "x") as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))

ROOT = Path(__file__).resolve().parents[2]


@wp.kernel
def count_grouped(grid: wp.uint64, x: wp.array(dtype=wp.vec3), group: wp.array(dtype=int), radius: float,
                  out: wp.array(dtype=int)):
    """Candidates of the grouped query the frozen ``_particle_contacts_b`` runs (radius 2.4 r, group = env)."""
    p = wp.tid()
    n = int(0)
    for q in wp.hash_grid_query(grid, x[p], radius, group[p]):
        n += 1
    wp.atomic_add(out, group[p], n)


def _helpers():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_p1_3a as P3
    import run_vector_milestones as RV
    return P3, RV


def _new_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)   # never overwrite
    return path


def _manifest(args, out):
    import hashlib
    P3, RV = _helpers()
    sources = sorted((ROOT / "src" / "robotic_cutting_sim" / "batched").glob("*.py")) + [Path(__file__).resolve()]
    data = dict(schema="rcs_rl_review_fix1_probe_v1", argv=sys.argv, args=vars(args),
                git_head=RV.git("rev-parse", "HEAD"), git_dirty=bool(RV.git("status", "--porcelain",
                                                                            "--untracked-files=no")),
                started_unix_s=time.time(), warp=wp.config.version,
                cuda_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    RV.dump(out / "manifest.json", data)


def _step_record(env, label, own_grid_graph, counts):
    """One control step of the batch graph from the loaded state + the mirrored candidate-count graph."""
    import numpy as np
    m, s = env.material, env.material.solver
    N = env.num_envs
    with env._stream():
        m.advance(interpolation="preset")
        stats_sum = s.stats_sum.numpy().reshape(N, -1).copy()
        stats_max = s.stats_max.numpy().reshape(N, -1).copy()
        impulse = s.impulse.numpy().copy()
        code = s.code.numpy().copy()
        x = s.x.numpy().copy()
        counts.zero_()
        if label != "unprotected":
            m.refresh_device_hashgrid()
        wp.capture_launch(own_grid_graph)
        candidates = counts.numpy().copy()
        reasons = m.invalid_reason.numpy().copy()
    return dict(stats4=stats_sum[:, 4], stats5=stats_sum[:, 5], stats6=stats_max[:, 6], impulse=impulse, code=code,
                x=x, candidates=candidates, invalid_reason=reasons)


def mode_batch_foreign(args):
    import numpy as np
    P3, RV = _helpers()
    from robotic_cutting_sim.batched import material_batch as MB
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.replay import native_samples
    out = _new_dir(args.out)
    _manifest(args, out)
    telemetry = RV.telemetry_start(args.gpu_index)
    seeds = (args.seed,) * args.copies
    inputs = RV.load_inputs([args.seed])
    data, _ = native_samples(RV.attempt_dir(args.seed))
    records, result = {}, dict(variant=args.variant, seed=args.seed, sample=args.sample, copies=args.copies)
    with open_vector_env(len(seeds), task=RV.family_for(seeds, inputs), device="cuda", strict=False) as env:
        env.reset(seeds=[42] * args.copies)
        m, s = env.material, env.material.solver
        counts = wp.zeros(args.copies, dtype=int, device=s.device)
        with env._stream():
            with wp.ScopedCapture(device=s.device) as capture:   # mirrors the batch's own build + grouped query
                s.hash.build(s.x, 2.4 * s.radius, groups=s.env_of_particle)
                wp.launch(count_grouped, s.total_particles, [s.hash.id, s.x, s.env_of_particle, 2.4 * s.radius,
                                                             counts], device=s.device)
        P3.sync()
        telemetry.set_phase("run")

        def load():   # env_ids form: the full-reset gc re-check of vector_env would refuse the foreign grid
            env.reset(env_ids=list(range(args.copies)), seeds=[42] * args.copies)
            P3.load_state(env, data, args.sample)
            P3.sync()
        load()
        records["protected"] = _step_record(env, "protected", capture.graph, counts)
        blocking = []   # does the refresh block the host while the batch graph still runs? (review note N3)
        for _ in range(5):
            with env._stream():
                P3.sync()
                m.refresh_device_hashgrid()
                wp.capture_launch(m.graph)
                t0 = time.perf_counter()
                m.refresh_device_hashgrid()
                t1 = time.perf_counter()
                P3.sync()
                t2 = time.perf_counter()
            blocking.append(dict(refresh_host_ms=(t1 - t0) * 1e3, graph_remaining_after_refresh_ms=(t2 - t1) * 1e3))
        result["refresh_while_graph_runs"] = blocking
        foreign = None
        if args.variant == "foreign":
            rng = np.random.default_rng(1)
            far = wp.array((rng.uniform(-.05, .05, (20000, 3)) + 5.0).astype(np.float32), dtype=wp.vec3,
                           device=s.device)
            foreign = wp.HashGrid(64, 64, 32, device=s.device)   # a frozen single material's grid dims
            foreign.reserve(20000)
            foreign.build(far, 2.4 * s.radius)
            result["foreign_grid"] = dict(dims=[64, 64, 32], points=20000, built_after_batch=True)
            load()
            foreign.build(far, 2.4 * s.radius)            # last host build = foreign right before the step
            records["foreign_protected"] = _step_record(env, "foreign_protected", capture.graph, counts)
            result["receipt_after_foreign_step"] = m.device_hashgrid_receipt()
            original = MB.BatchedCuttingMaterial.refresh_device_hashgrid
            MB.BatchedCuttingMaterial.refresh_device_hashgrid = lambda self: None   # pre-fix default path
            try:
                load()
                foreign.build(far, 2.4 * s.radius)
                records["unprotected"] = _step_record(env, "unprotected", capture.graph, counts)
            finally:
                MB.BatchedCuttingMaterial.refresh_device_hashgrid = original
        result["particles_per_env"] = m.packed.particle_counts.tolist()
        result["records"] = {k: dict(stats4=v["stats4"].tolist(), stats6=v["stats6"].tolist(),
                                     candidates=v["candidates"].tolist(),
                                     impulse_nonzero_particles=int(np.any(v["impulse"] != 0, axis=1).sum()),
                                     invalid_reason=v["invalid_reason"].tolist(),
                                     decoded=[MB.decode_reason(r) for r in v["invalid_reason"]])
                             for k, v in records.items()}
        np.savez_compressed(out / "records.npz", **{f"{k}_{f}": v[f] for k, v in records.items()
                                                    for f in ("impulse", "code", "x", "stats4", "stats6",
                                                              "candidates")})
        del foreign
    summary = telemetry.stop()
    result.update(status="COMPLETE", telemetry=summary)
    RV.dump(out / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "telemetry"}, indent=1, default=str)[:4000])


def mode_compare(args):
    import numpy as np
    clean = np.load(Path(args.clean) / "records.npz")
    foreign = np.load(Path(args.foreign) / "records.npz")

    def compare(a, b):
        """``a`` from the clean process, ``b`` from the foreign process."""
        def get(name, field, source):
            return source[f"{name}_{field}"]
        ga = lambda f: get(a, f, clean)   # noqa: E731
        gb = lambda f: get(b, f, foreign)   # noqa: E731
        ia, ib = ga("impulse").astype(np.float64), gb("impulse").astype(np.float64)
        nz_a, nz_b = np.any(ia != 0, axis=1), np.any(ib != 0, axis=1)
        scale = float(np.abs(ia).max())
        d = float(np.abs(ia - ib).max())
        return dict(candidates=[ga("candidates").tolist(), gb("candidates").tolist()],
                    candidates_equal=bool(np.array_equal(ga("candidates"), gb("candidates"))),
                    stats4=[ga("stats4").tolist(), gb("stats4").tolist()],
                    stats4_equal=bool(np.array_equal(ga("stats4"), gb("stats4"))),
                    stats6_equal=bool(np.array_equal(ga("stats6"), gb("stats6"))),
                    impulse_nonzero_sets_equal=bool(np.array_equal(nz_a, nz_b)),
                    impulse_nonzero_particles=[int(nz_a.sum()), int(nz_b.sum())],
                    impulse_bitwise_equal=bool(np.array_equal(ia, ib)), impulse_max_abs_diff=d,
                    impulse_max_rel_diff=d / scale if scale else 0.,
                    codes_equal=bool(np.array_equal(ga("code"), gb("code"))),
                    x_max_abs_diff_m=float(np.abs(ga("x").astype(float) - gb("x").astype(float)).max()))
    result = dict(clean=str(args.clean), foreign=str(args.foreign),
                  clean_vs_foreign_protected=compare("protected", "foreign_protected"),
                  clean_vs_foreign_process_first_step=compare("protected", "protected"),
                  clean_vs_unprotected=compare("protected", "unprotected"))
    c = result["clean_vs_foreign_protected"]
    u = result["clean_vs_unprotected"]
    result["verdict"] = dict(
        protected_matches_clean=bool(c["candidates_equal"] and c["stats4_equal"] and c["impulse_nonzero_sets_equal"]
                                     and c["codes_equal"] and c["impulse_max_rel_diff"] < 1e-6),
        hazard_reproduced_without_refresh=bool(not u["candidates_equal"] or not u["stats4_equal"]))
    with open(args.out, "x") as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


def mode_refresh_cost(args):
    import types
    import numpy as np
    P3, RV = _helpers()
    from robotic_cutting_sim.batched import material_batch as MB
    out = _new_dir(args.out)
    _manifest(args, out)
    original = MB.BatchedCuttingMaterial.refresh_device_hashgrid
    calls = dict(host_wall_s=[], events=[], count=0)
    if args.refresh == "off":
        def refresh(self):   # pre-fix frozen_v1 behaviour: no host rebuild before the launch
            self.descriptor_refreshes += 1
            calls["count"] += 1
    else:
        def refresh(self):
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start)
            t0 = time.perf_counter()
            original(self)
            calls["host_wall_s"].append(time.perf_counter() - t0)
            wp.record_event(end)
            calls["events"].append((start, end))
            calls["count"] += 1
    MB.BatchedCuttingMaterial.refresh_device_hashgrid = refresh
    telemetry = RV.telemetry_start(args.gpu_index)
    ns = types.SimpleNamespace(num_envs=args.num_envs, steps=args.steps, warmup=args.warmup, kernel_every=0,
                               variants="frozen_v1")
    status, error, result = "FAIL", None, {}
    try:
        result = P3.mode_ab(ns, out, telemetry)
        status = "COMPLETE"
    except Exception:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        summary = telemetry.stop()
        with (out / "telemetry_samples.jsonl").open("x") as stream:
            for sample in telemetry.samples:
                stream.write(json.dumps(sample, default=str) + "\n")
    gpu_ms = [wp.get_event_elapsed_time(a, b) for a, b in calls["events"]]
    result.update(status=status, error=error, refresh=args.refresh, telemetry=summary,
                  telemetry_run_phase=telemetry.summary(phases=("run",)), refresh_calls=calls["count"],
                  refresh_host_wall_ms=None if not calls["host_wall_s"] else dict(
                      median=float(np.median(calls["host_wall_s"]) * 1e3), max=float(np.max(calls["host_wall_s"]) * 1e3),
                      n=len(calls["host_wall_s"])),
                  refresh_gpu_event_ms=None if not gpu_ms else dict(median=float(np.median(gpu_ms)),
                                                                    max=float(np.max(gpu_ms)), n=len(gpu_ms)),
                  scope="frozen_v1 batch, run_p1_3a.mode_ab protocol (steps after --warmup, step_measured_edge wall "
                        "with device sync before/after, strict int32 readback); refresh host wall = the Python call "
                        "without a sync (enqueue only unless the call blocks); GPU time = CUDA events around it")
    RV.dump(out / "result.json", result)
    print(json.dumps(dict(out=str(out), status=status, step_wall_s=result.get("step_wall_s"),
                          refresh_host_wall_ms=result["refresh_host_wall_ms"],
                          refresh_gpu_event_ms=result["refresh_gpu_event_ms"])))
    if error:
        raise SystemExit(1)


def cli():
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        return main(sys.argv[1])   # original P1-3a probe
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("batch-foreign", "refresh-cost", "compare", "replay-race"), required=True)
    parser.add_argument("--spin-iterations", type=int, default=20000000)
    parser.add_argument("--variant", choices=("clean", "foreign"), default="clean")
    parser.add_argument("--seed", type=int, default=15, choices=(512, 15))
    parser.add_argument("--sample", type=int, default=43)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--refresh", choices=("on", "off"), default="on")
    parser.add_argument("--clean")
    parser.add_argument("--foreign")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "replay-race":
        wp.init()
        return race(args.out, args.spin_iterations)
    dict(**{"batch-foreign": mode_batch_foreign, "refresh-cost": mode_refresh_cost, "compare": mode_compare})[
        args.mode](args)


if __name__ == "__main__":
    cli()
