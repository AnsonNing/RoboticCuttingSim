#!/usr/bin/env python3
"""P1-3a GPU checks of the opt-in ``alias_free_v1`` particle contacts (batched material, workload A).

Workload A as in ``run_vector_milestones.py``: seeded TRIM seeds 512/15 (WP-17 configuration), 1 mm particles,
4 mm grid, 0.1 ms material dt, 200 substeps per 50 Hz control step, tlmpm_v7_2 law 8, K=2, S=8, handle + two
measured finger pads; no cameras, no recording; ``reduced_fidelity`` (ideal edge tracking). Variants
(``VARIANTS``): ``frozen_v1`` (default, rollback), ``alias_free_v1`` (C1 + C2 + fallback guard) and the
sub-flag isolations ``c1_only``, ``c2_fallback_no_c1``, ``c1_c2_unguarded`` (timing only).

Modes (one process each; GPU lock, ``CUDA_VISIBLE_DEVICES=<gpu>``, ``TMPDIR=/mnt/data/ningan/tmp``,
``PYTHONPATH=src``, ``.deps/isaacsim-venv/bin/python``, no Kit):

* ``samestate --seed S --sample K``: N=``--copies`` copies of seed S with the recorded Native state of sample K
  (positions/velocities/codes, world -> local) written into frozen_v1 and alias_free_v1 batches; the contact
  stage alone (impulse/stats zeroed, hash build + contact kernel(s)) is run eagerly and compared (set of
  particles with non-zero impulse, stats[4] count, impulse/stat values; frozen run twice for the GPU
  float-atomic band), then each stage is captured into its own CUDA graph (alias_free_v1 with its conditional
  fallback node) and replayed interleaved with CUDA-event timing.
* ``ab --num-envs N --steps T --variants a,b``: one env per variant (env k replays seed 512 if k even else 15),
  all stepped through ``step_measured_edge`` in the same process, interleaved per control step (rotating order);
  per-step walls, per-step per-env contact counts (stats[4]), Warp activity timing of graph replays on every
  ``--kernel-every``-th step, final state comparison between variants.
* ``episode --seeds 512 [15 ...] --variants alias_free_v1[,frozen_v1]``: full 1200-step replays, variants
  interleaved per step; L2 monitor, 121 Native sample comparisons, final codes vs Native, frozen evaluator on
  2 and 121 frames, guard receipts.
* ``profile --num-envs N --variants V --steps K``: a short replay inside cudaProfilerStart/Stop for an external
  ``nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi``.

Raw output: a new ``artifacts/rl_ready/p1_3/<name>_NNN/`` per run (never overwritten).
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import resource
import sys
import time

T_SCRIPT = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_vector_milestones as RV  # noqa: E402  (helpers only; its OUT_ROOT is not used)

OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_3"
VARIANTS = {"frozen_v1": ("frozen_v1", None), "alias_free_v1": ("alias_free_v1", None),
            "c1_only": ("alias_free_v1", {"alias_free_hash_dims": False}),
            "c2_fallback_no_c1": ("alias_free_v1", {"skip_zero_impulse_atomics": False}),
            "c1_c2_unguarded": ("alias_free_v1", {"speed_guard": "off"}),
            # guard bound 0: every moving env takes the conditional FallbackGrid path every substep (check only)
            "alias_free_forced_fallback": ("alias_free_v1", {"speed_bound_m_s": 0.0})}
P1_0_POSITION_BAND_M = {512: (1.4e-5, 3.9e-5), 15: (2.0e-4, 4.6e-4)}


def new_run_dir(name):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = OUT_ROOT / f"{name}_{index:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("no free run directory")


def manifest(args, run):
    import hashlib
    import torch
    import warp
    data = dict(schema="rcs_rl_p1_3a_run_v1", argv=sys.argv, args=vars(args), git_head=RV.git("rev-parse", "HEAD"),
                git_branch=RV.git("rev-parse", "--abbrev-ref", "HEAD"),
                git_dirty=bool(RV.git("status", "--porcelain", "--untracked-files=no")), started_unix_s=time.time(),
                source_sha256_at_start={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted((ROOT / "src" / "robotic_cutting_sim" / "batched").glob("*.py"))
                                        + sorted((ROOT / "src" / "robotic_cutting_sim" / "rl").glob("*.py"))
                                        + [Path(__file__).resolve()]},
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), python=sys.version.split()[0],
                interpreter=os.path.relpath(sys.executable, ROOT), kit_started=False,
                versions=dict(warp=warp.__version__, numpy=np.__version__, torch=torch.__version__),
                conditional_graph_supported=bool(warp.is_conditional_graph_supported()),
                workload=dict(id="A", material="seeded_trim tlmpm_v7_2 law 8, 1 mm / 4 mm / 0.1 ms / 200 substeps, "
                              "K=2, S=8", control_hz=50, cameras="none", recording="none", robot_dof=0,
                              obstacles="handle + two measured finger pads (handle_and_measured_pads_v1)",
                              reduced_fidelity=True))
    RV.dump(run / "manifest.json", data)
    return data


def open_env(stack, variant, seeds, inputs):
    from robotic_cutting_sim.rl import open_vector_env
    mode, options = VARIANTS[variant]
    return stack.enter_context(open_vector_env(len(seeds), task=RV.family_for(seeds, inputs), device="cuda",
                                               strict=True, particle_contact_mode=mode,
                                               particle_contact_options=options))


def sync():
    import torch
    torch.cuda.synchronize()


def stats_counts(env):
    s = env.material.solver
    return s.stats_sum.numpy().reshape(env.num_envs, -1)[:, 4].astype(np.int64)


def guard_violations(env):
    s = env.material.solver
    return s.contact_guard_violations.numpy().tolist() if hasattr(s, "contact_guard_violations") else None


class L2MonitorActiveGrid(RV.L2Monitor):
    """``RV.L2Monitor`` whose grouped-query audit uses the grid the contact stage actually queries."""

    def audit(self, step):
        from robotic_cutting_sim.batched import layout
        wp, s, pk = self.wp, self.s, self.env.material.packed
        pc = s.particle_contact
        grid = s.hash if pc is None else pc.grid
        with self.env._stream():
            grid.build(s.x, 2.4 * s.radius, groups=s.env_of_particle)   # the next substep rebuilds it anyway
            foreign = wp.zeros(1, dtype=int, device=s.device)
            visited = wp.zeros(1, dtype=int, device=s.device)
            wp.launch(RV._grouped_query_audit, s.total_particles, [grid.id, s.x, s.env_of_particle,
                                                                    2.4 * s.radius, foreign, visited], device=s.device)
            ranges = layout.range_audit(pk, s.nodes.numpy(), s.link_offsets.numpy(), s.link_neighbours.numpy(),
                                        s.pending_q.numpy(), s.env_of_particle.numpy())
            row = dict(step=step, range_audit=ranges, grouped_query_visited=int(visited.numpy()[0]),
                       grouped_query_foreign=int(foreign.numpy()[0]), grid_dims=list(grid_dims(s)),
                       mass_bitwise_constant=bool(np.array_equal(s.mass.numpy(), self.mass0)))
        self.audits.append(row)
        return row


def grid_dims(s):
    return (64, 64, 32) if s.particle_contact is None else s.particle_contact.dims


# ----------------------------------------------------------------------------- samestate

def contact_stage(env):
    """Zero impulse/stats, then the variant's contact stage exactly as inside ``BatchedTLMPMv7_2.step``."""
    import warp as wp
    from robotic_cutting_sim.batched import tlmpm_v7_2_batch as TB
    s = env.material.solver
    with wp.ScopedDevice(s.device):
        s.impulse.zero_()
        s.stats_sum.zero_()
        s.stats_max.zero_()
        if s.particle_contact is None:
            s.hash.build(s.x, 2.4 * s.radius, groups=s.env_of_particle)
            wp.launch(TB._particle_contacts_b, s.total_particles, [s.hash.id, s.x, s.reference, s.v, s.mass, s.code,
                      s.impulse, s.stats_sum, s.stats_max, s.radius, 3 * s.dx, s.dt, s.params, s.env_of_particle])
        else:
            s.particle_contact.launch(s, s.radius, s.dt)


def load_state(env, data, sample):
    """Write Native sample ``sample`` (world) into every env of ``env`` in solver-local float32."""
    s = env.material.solver
    pk = env.material.packed
    x, v, code = s.x.numpy(), s.v.numpy(), s.code.numpy()
    pw = np.asarray(data["positions"][sample], np.float64)
    vw = np.asarray(data["velocities"][sample], np.float64)
    cw = np.asarray(data["codes"][sample], np.int32)
    for e in range(env.num_envs):
        T = pk.env["T_world_local"][e]
        R, t = T[:3, :3], T[:3, 3]
        rows = slice(int(pk.particle_offset[e]), int(pk.particle_offset[e + 1]))
        x[rows] = ((pw - t) @ R).astype(np.float32)
        v[rows] = (vw @ R).astype(np.float32)
        code[rows] = cw
    s.x.assign(x)
    s.v.assign(v)
    s.code.assign(code)


def read_stage(env):
    s = env.material.solver
    N = env.num_envs
    return dict(impulse=s.impulse.numpy().copy(), stats_sum=s.stats_sum.numpy().reshape(N, -1).copy(),
                stats_max=s.stats_max.numpy().reshape(N, -1).copy())


def compare_stage(a, b):
    nz_a = np.any(a["impulse"] != 0, axis=1)
    nz_b = np.any(b["impulse"] != 0, axis=1)
    d = np.abs(a["impulse"].astype(np.float64) - b["impulse"].astype(np.float64))
    scale = np.abs(a["impulse"].astype(np.float64)).max()
    return dict(nonzero_particles=[int(nz_a.sum()), int(nz_b.sum())],
                nonzero_sets_identical=bool(np.array_equal(nz_a, nz_b)),
                stats4_count=[a["stats_sum"][:, 4].tolist(), b["stats_sum"][:, 4].tolist()],
                stats4_identical=bool(np.array_equal(a["stats_sum"][:, 4], b["stats_sum"][:, 4])),
                stats5_max_abs_diff=float(np.abs(a["stats_sum"][:, 5] - b["stats_sum"][:, 5]).max()),
                stats5=[a["stats_sum"][:, 5].tolist(), b["stats_sum"][:, 5].tolist()],
                stats6_identical=bool(np.array_equal(a["stats_max"][:, 6], b["stats_max"][:, 6])),
                impulse_bitwise_equal=bool(np.array_equal(a["impulse"], b["impulse"])),
                impulse_max_abs_diff=float(d.max()), impulse_max_abs=float(scale),
                impulse_max_rel_diff=float(d.max() / scale) if scale else 0.)


def mode_samestate(args, run, telemetry):
    import warp as wp
    seed, sample = args.seed, args.sample
    inputs = RV.load_inputs([seed])
    from robotic_cutting_sim.rl.replay import native_samples
    data, _ = native_samples(RV.attempt_dir(seed))
    seeds = (seed,) * args.copies
    out = dict(seed=seed, sample=sample, time_s=float(data["time_s"][sample]), copies=args.copies)
    with contextlib.ExitStack() as stack:
        envs = {name: open_env(stack, name, seeds, inputs)
                for name in ("frozen_v1", "alias_free_v1", "alias_free_forced_fallback")}
        for env in envs.values():
            env.reset(seeds=[42] * args.copies)
            load_state(env, data, sample)
        sync()
        telemetry.set_phase("run")
        rows = {}
        for repeat in range(2):
            for name, env in envs.items():
                with env._stream():
                    contact_stage(env)
                sync()
                rows[f"{name}_{repeat}"] = read_stage(env)
        out["frozen_vs_alias_free"] = compare_stage(rows["frozen_v1_0"], rows["alias_free_v1_0"])
        out["frozen_vs_frozen_rerun"] = compare_stage(rows["frozen_v1_0"], rows["frozen_v1_1"])
        out["alias_free_vs_alias_free_rerun"] = compare_stage(rows["alias_free_v1_0"], rows["alias_free_v1_1"])
        out["frozen_vs_forced_fallback"] = compare_stage(rows["frozen_v1_0"], rows["alias_free_forced_fallback_0"])
        out["forced_fallback_guard_violations"] = guard_violations(envs["alias_free_forced_fallback"])
        RV.dump(run / "partial_eager.json", out)
        first = rows["alias_free_v1_0"]
        P = int(envs["alias_free_v1"].material.packed.particle_counts[0])
        out["copies_identical_alias_free"] = [bool(np.array_equal(first["impulse"][:P], first["impulse"][e * P:(e + 1) * P]))
                                              for e in range(args.copies)]
        out["speed_m_s_max"] = float(np.linalg.norm(np.asarray(data["velocities"][sample]), axis=1).max())
        out["alias_free_receipt"] = envs["alias_free_v1"].material.solver.particle_contact_receipt()
        # Stage graphs (alias_free_v1 contains the conditional fallback node), interleaved replay timing.
        out["checks"] = dict(
            nonzero_sets_identical=out["frozen_vs_alias_free"]["nonzero_sets_identical"],
            stats4_identical=out["frozen_vs_alias_free"]["stats4_identical"],
            forced_fallback_nonzero_sets_identical=out["frozen_vs_forced_fallback"]["nonzero_sets_identical"],
            forced_fallback_stats4_identical=out["frozen_vs_forced_fallback"]["stats4_identical"],
            forced_fallback_ran=all(v > 0 for v in out["forced_fallback_guard_violations"]),
            alias_free_no_violation=all(v == 0 for v in guard_violations(envs["alias_free_v1"])))
        graphs = {}
        for name, env in envs.items():
            with wp.ScopedCapture(device=env.material.device) as capture:
                contact_stage(env)
            graphs[name] = capture.graph
        sync()
        times = {name: [] for name in envs}
        for repeat in range(args.repeats):
            order = list(envs) if repeat % 2 == 0 else list(envs)[::-1]
            for name in order:
                env = envs[name]
                start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                wp.record_event(start)
                wp.capture_launch(graphs[name])
                wp.record_event(end)
                wp.synchronize_event(end)
                times[name].append(wp.get_event_elapsed_time(start, end))
        out["stage_graph_ms"] = {name: dict(median=float(np.median(t)), min=float(np.min(t)),
                                            p10=float(np.percentile(t, 10)),
                                            p90=float(np.percentile(t, 90)), n=len(t)) for name, t in times.items()}
        out["stage_graph_scope"] = ("CUDA-event time of one replay of a graph holding impulse/stats memsets + hash "
                                    "build (incl. CUB sort) + contact kernel(s) [+ guard kernel and conditional "
                                    "fallback node]; N copies of the same state")
        replayed = {name: read_stage(env) for name, env in envs.items()}
        out["after_graph_replays_frozen_vs_alias_free"] = compare_stage(replayed["frozen_v1"],
                                                                        replayed["alias_free_v1"])
        out["after_graph_replays_frozen_vs_forced_fallback"] = compare_stage(replayed["frozen_v1"],
                                                                             replayed["alias_free_forced_fallback"])
        # Warp activity timing (kernels, memsets, memcpys) of the eager stage, interleaved.
        activity = {name: {} for name in envs}
        for repeat in range(args.repeats // 4 or 1):
            for name, env in envs.items():
                wp.timing_begin(cuda_filter=wp.TIMING_ALL)
                with env._stream():
                    contact_stage(env)
                for r in wp.timing_end():
                    activity[name].setdefault(r.name, []).append(r.elapsed)
        out["stage_eager_activity_ms"] = {name: {k: dict(median=float(np.median(v)), calls=len(v))
                                                 for k, v in rows_.items()} for name, rows_ in activity.items()}
        out["guard_violations"] = guard_violations(envs["alias_free_v1"])
    return out


# ----------------------------------------------------------------------------- ab timing

def mode_ab(args, run, telemetry):
    import torch
    import warp as wp
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    N = args.num_envs
    seeds = tuple(512 if k % 2 == 0 else 15 for k in range(N))
    order = sorted(set(seeds))
    inputs = RV.load_inputs(seeds)
    variants = args.variants.split(",")
    out = dict(num_envs=N, seeds=list(seeds), variants=variants, steps=args.steps)
    with contextlib.ExitStack() as stack:
        envs = {}
        for name in variants:
            start = time.perf_counter()
            envs[name] = open_env(stack, name, seeds, inputs)
            out.setdefault("build_wall_s", {})[name] = time.perf_counter() - start
        drivers = {name: MeasuredTrajectoryDriver([inputs[s]["trajectory"] for s in order],
                                                  [order.index(s) for s in seeds], device="cuda") for name in envs}
        for env in envs.values():
            env.reset(seeds=[42] * N)
        sync()
        telemetry.set_phase("run")
        walls = {name: [] for name in envs}
        contacts = {name: [] for name in envs}
        activity = {name: {} for name in envs}
        activity_steps = {name: 0 for name in envs}
        for t in range(args.warmup + args.steps):
            rotation = variants[t % len(variants):] + variants[:t % len(variants)]
            for name in rotation:
                env = envs[name]
                pose, cut, pads = drivers[name].inputs(env._step_count)
                profile = t >= args.warmup and args.kernel_every and (t - args.warmup) % args.kernel_every == 0
                if profile:
                    wp.timing_begin(cuda_filter=wp.TIMING_ALL)
                _, wall = RV.timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads), sync)
                if profile:
                    for r in wp.timing_end():
                        activity[name].setdefault(r.name, []).append(r.elapsed)
                    activity_steps[name] += 1
                if t >= args.warmup:
                    walls[name].append(wall)
                contacts[name].append(stats_counts(env).tolist())
        out["step_wall_s"] = {name: dict(median=float(np.median(w)), mean=float(np.mean(w)), min=float(np.min(w)),
                                         p10=float(np.percentile(w, 10)), p90=float(np.percentile(w, 90)),
                                         n=len(w)) for name, w in walls.items()}
        out["env_steps_per_s"] = {name: float(N * len(w) / np.sum(w)) for name, w in walls.items()}
        base = variants[0]
        out["speedup_vs_first_median"] = {name: float(np.median(walls[base]) / np.median(w))
                                          for name, w in walls.items()}
        out["activity_ms_per_profiled_step"] = {
            name: {k: dict(total_per_step=float(np.sum(v) / max(activity_steps[name], 1)), calls=len(v))
                   for k, v in sorted(rows.items(), key=lambda kv: -np.sum(kv[1]))}
            for name, rows in activity.items()}
        out["activity_profiled_steps"] = activity_steps
        out["contact_counts_identical_to_first"] = {name: contacts[name] == contacts[base] for name in envs}
        out["contact_counts_first_difference_step"] = {
            name: next((i for i, (a, b) in enumerate(zip(contacts[base], contacts[name])) if a != b), None)
            for name in envs}
        out["contact_counts_total"] = {name: int(np.sum(c)) for name, c in contacts.items()}
        finals = {name: [env.material.snapshot(e) for e in range(N)] for name, env in envs.items()}
        out["final_vs_first"] = {}
        for name in envs:
            rows = []
            for e in range(N):
                a, b = finals[base][e], finals[name][e]
                rows.append(dict(env=e, codes_equal=bool(np.array_equal(a["code"], b["code"])),
                                 different_code_particles=int(np.count_nonzero(a["code"] != b["code"])),
                                 x_bitwise=bool(np.array_equal(a["x"], b["x"])),
                                 x_max_abs_diff_m=float(np.abs(a["x"].astype(float) - b["x"].astype(float)).max())))
            out["final_vs_first"][name] = rows
        np.savez_compressed(run / "steps.npz", **{f"wall_{k}": np.asarray(v, np.float32) for k, v in walls.items()},
                            **{f"contacts_{k}": np.asarray(v, np.int64) for k, v in contacts.items()})
        out["guard_violations"] = {name: guard_violations(env) for name, env in envs.items()}
        out["receipts"] = {name: env.material.solver.particle_contact_receipt() for name, env in envs.items()}
        out["particles_per_env"] = envs[base].material.packed.particle_counts.tolist()
        out["memory_bytes"] = {name: env.material.memory_bytes() for name, env in envs.items()}
        out["mempool_high_bytes_all_envs"] = RV.mempool_high()
        out["timing_scope"] = ("env.step_measured_edge wall (one 50 Hz control step = 200-substep graph replay + "
                               "coupling + strict int32 readback), device synchronize before/after; variants "
                               "interleaved per control step in rotating order; activity timing steps included")
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    return out


# ----------------------------------------------------------------------------- full episodes

def mode_episode(args, run, telemetry):
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    from robotic_cutting_sim.rl.replay import compare_sample
    seeds = tuple(int(s) for s in args.seeds.split(","))
    N = len(seeds)
    order = sorted(set(seeds))
    inputs = RV.load_inputs(seeds)
    natives = {s: RV.native_final_codes(s) for s in order}
    variants = args.variants.split(",")
    out = dict(num_envs=N, seeds=list(seeds), variants=variants)
    RV._grouped_query_audit = RV._define_kernels()
    with contextlib.ExitStack() as stack:
        envs = {name: open_env(stack, name, seeds, inputs) for name in variants}
        out["payload_check"] = RV.check_payloads(envs[variants[0]], inputs)
        drivers = {name: MeasuredTrajectoryDriver([inputs[s]["trajectory"] for s in order],
                                                  [order.index(s) for s in seeds], device="cuda") for name in envs}
        for env in envs.values():
            env.reset(seeds=[42] * N)
        sync()
        telemetry.set_phase("run")
        monitors = {name: L2MonitorActiveGrid(env) for name, env in envs.items()}
        for m in monitors.values():
            m.audit(0)
        sample_steps = {s: np.rint(np.asarray(natives[s][1]["time_s"]) / .02).astype(int) for s in order}
        frames = {name: {e: [] for e in range(N)} for name in envs}
        comparisons = {name: {e: [] for e in range(N)} for name in envs}
        walls = {name: np.zeros(RV.STEPS) for name in envs}
        contacts = {name: np.zeros((RV.STEPS, N), np.int64) for name in envs}
        steps = min(RV.STEPS, args.max_global_steps or RV.STEPS)
        for t in range(steps):
            rotation = variants[t % len(variants):] + variants[:t % len(variants)]
            for name in rotation:
                env = envs[name]
                pose, cut, pads = drivers[name].inputs(env._step_count)
                _, walls[name][t] = RV.timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads), sync)
                monitors[name].update()
                contacts[name][t] = stats_counts(env)
                step = t + 1
                for e, seed in enumerate(seeds):
                    hits = np.flatnonzero(sample_steps[seed] == step)
                    if hits.size:
                        index = int(hits[0])
                        snap = env.snapshot_world(e)
                        native = natives[seed][1]
                        row = compare_sample({k: native[k][index] for k in ("positions", "velocities", "codes")},
                                             snap, 2)
                        row.update(step=step, sample=index)
                        comparisons[name][e].append(row)
                        frames[name][e].append({k: snap[k] for k in ("positions", "velocities", "codes", "step_count",
                                                                     "time_s")})
                if step % 300 == 0:
                    monitors[name].audit(step)
        out["complete"] = steps == RV.STEPS
        results = {}
        for name, env in envs.items():
            r = dict(l2=monitors[name].result(), guard_violations=guard_violations(env),
                     invalid_reason=env.material.invalid_reason.numpy().tolist(),
                     receipt=env.material.solver.particle_contact_receipt(), envs=[])
            ev2 = env.evaluate(list(range(N)))
            ev121 = env.evaluate(list(range(N)), frames={e: frames[name][e][:-1] for e in range(N)})
            for e, seed in enumerate(seeds):
                final = {k: v for k, v in RV.final_comparison(env, e, natives[seed][0]).items() if k != "positions"}
                comps = comparisons[name][e]
                lo, hi = P1_0_POSITION_BAND_M[seed]
                fmax = comps[-1]["max_position_diff_m"] if comps else None
                r["envs"].append(dict(
                    env=e, seed=seed, final=final, evaluation_2_frames=RV.eval_summary([ev2[e]])[0],
                    evaluation_121_frames=RV.eval_summary([ev121[e]])[0],
                    parity_vs_native=dict(samples=len(comps),
                                          max_position_diff_m_over_time=max(c["max_position_diff_m"] for c in comps),
                                          final_max_position_diff_m=fmax,
                                          final_mean_position_diff_m=comps[-1]["mean_position_diff_m"],
                                          samples_with_code_difference=[dict(step=c["step"],
                                                                             particles=c["different_code_particles"])
                                                                        for c in comps if c["different_code_particles"]],
                                          p1_0_final_max_position_band_m=[lo, hi],
                                          final_max_position_within_or_below_band=bool(fmax is not None and fmax <= hi))))
            results[name] = r
        out["results"] = results
        if len(variants) > 1:
            base = variants[0]
            out["cross_variant"] = {}
            for name in variants[1:]:
                rows = []
                for e in range(N):
                    a, b = envs[base].material.snapshot(e), envs[name].material.snapshot(e)
                    rows.append(dict(env=e, codes_equal=bool(np.array_equal(a["code"], b["code"])),
                                     different_code_particles=int(np.count_nonzero(a["code"] != b["code"])),
                                     x_max_abs_diff_m=float(np.abs(a["x"].astype(float) - b["x"].astype(float)).max()),
                                     contact_count_steps_equal=int(np.sum(np.all(contacts[base] == contacts[name], 1))),
                                     contact_count_total=[int(contacts[base][:, e].sum()),
                                                          int(contacts[name][:, e].sum())]))
                out["cross_variant"][name] = rows
        out["timing"] = {name: dict(steps=steps, warm_step_median_s=float(np.median(w[1:steps])),
                                    warm_step_min_s=float(np.min(w[1:steps])),
                                    warm_step_mean_s=float(np.mean(w[1:steps])),
                                    warm_step_p95_s=float(np.percentile(w[1:steps], 95)), first_step_s=float(w[0]),
                                    warm_env_steps_per_s=float(N * (steps - 1) / np.sum(w[1:steps])))
                         for name, w in walls.items()}
        out["timing_scope"] = ("env.step_measured_edge wall incl. strict int32 readback, device synchronize "
                               "before/after; variants interleaved per control step; L2 monitor, samples, "
                               "contact-count readback outside")
        np.savez_compressed(run / "steps.npz", **{f"wall_{k}": v.astype(np.float32) for k, v in walls.items()},
                            **{f"contacts_{k}": v for k, v in contacts.items()})
    out["mempool_high_bytes"] = RV.mempool_high()
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    return out


# ----------------------------------------------------------------------------- nsys window

def mode_profile(args, run, telemetry):
    import torch
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    N = args.num_envs
    seeds = tuple(512 if k % 2 == 0 else 15 for k in range(N))
    order = sorted(set(seeds))
    inputs = RV.load_inputs(seeds)
    (variant,) = args.variants.split(",")
    with contextlib.ExitStack() as stack:
        env = open_env(stack, variant, seeds, inputs)
        driver = MeasuredTrajectoryDriver([inputs[s]["trajectory"] for s in order], [order.index(s) for s in seeds],
                                          device="cuda")
        env.reset(seeds=[42] * N)
        for _ in range(args.warmup):
            pose, cut, pads = driver.inputs(env._step_count)
            env.step_measured_edge(pose, cut, pad_boxes=pads)
        sync()
        telemetry.set_phase("run")
        torch.cuda.cudart().cudaProfilerStart()
        walls = []
        for _ in range(args.steps):
            pose, cut, pads = driver.inputs(env._step_count)
            _, wall = RV.timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads), sync)
            walls.append(wall)
        torch.cuda.cudart().cudaProfilerStop()
        return dict(num_envs=N, variant=variant, steps=args.steps, warmup=args.warmup, step_walls_s=walls,
                    total_particles=env.material.solver.total_particles,
                    receipt=env.material.solver.particle_contact_receipt())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("samestate", "ab", "episode", "profile"), required=True)
    parser.add_argument("--seed", type=int, default=512, choices=(512, 15))
    parser.add_argument("--sample", type=int, default=33)
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--kernel-every", type=int, default=10)
    parser.add_argument("--seeds", default="512")
    parser.add_argument("--variants", default="frozen_v1,alias_free_v1")
    parser.add_argument("--max-global-steps", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, required=True, help="physical nvidia-smi index (telemetry)")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    for name in args.variants.split(","):
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}")
    name = args.run_name or {"samestate": f"samestate_s{args.seed}_k{args.sample}", "ab": f"ab_n{args.num_envs}",
                             "episode": "episode_" + args.seeds.replace(",", "_"),
                             "profile": f"profile_n{args.num_envs}_{args.variants}"}[args.mode]
    run = new_run_dir(name)
    import torch  # noqa: F401
    import robotic_cutting_sim  # noqa: F401
    manifest(args, run)
    telemetry = RV.telemetry_start(args.gpu_index)
    status, error, result = "FAIL", None, {}
    try:
        result = dict(samestate=mode_samestate, ab=mode_ab, episode=mode_episode, profile=mode_profile)[args.mode](
            args, run, telemetry)
        status = "COMPLETE"
    except Exception:   # preserved as a failed run, never hidden
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        summary = telemetry.stop()
        with (run / "telemetry_samples.jsonl").open("x") as stream:
            for sample in telemetry.samples:
                stream.write(json.dumps(sample, default=str) + "\n")
    out = dict(status=status, error=error, mode=args.mode, telemetry=summary,
               telemetry_run_phase=telemetry.summary(phases=("run",)), wall_s=time.perf_counter() - T_SCRIPT, **result)
    RV.dump(run / "result.json", out)
    print(json.dumps(dict(run=str(run.relative_to(ROOT / "artifacts")), status=status,
                          gpu_sharing=summary.get("gpu_sharing"), wall_s=out["wall_s"])))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
