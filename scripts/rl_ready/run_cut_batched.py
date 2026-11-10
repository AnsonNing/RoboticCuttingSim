#!/usr/bin/env python3
"""P1-5 T5.7: batched CUT (workload C2, ``cut_v7_2_v1``) in the ``rl_gpu_headless`` vector env.

One process, one ``VectorCuttingEnv`` over a ``ConfiguredTaskFamily.from_cut_workload(N)`` (one batched material,
one device HashGrid; the census is recorded), ideal tracking, ``reference_command_policy_v1`` actions (and, for the
envs named in ``--random-envs``, ``RandomPolicyV1``) through the public ``env.step``. L2 invariants every step
(``run_vector_milestones.L2Monitor``: per-env code monotonicity, finite state, min det F, slot overflow, invalid
bits; grouped-hash foreign candidates and range audits at fixed steps). The P1-2b evaluation stream (``every`` 20
control steps: 36 frames per 700-step episode) feeds the frozen sever evaluator (``env.evaluate``) of every env at
the end of every episode, outside timing. ``--forced-resets e:step`` resets env ``e`` with ``reset(env_ids=[e])``
at that global step (its unfinished episode is evaluated first and labelled TRUNCATED_BY_FORCED_RESET);
``--episodes E`` runs E back-to-back 700-step episodes (full reset in between). Benchmark-row timing: the
synchronized wall of every ``env.step`` (policy action computation excluded), resets separately.

    CUDA_VISIBLE_DEVICES=<g> TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock <gpu<g>.lock> \
        .deps/isaacsim-venv/bin/python scripts/rl_ready/run_cut_batched.py --num-envs 4 --gpu-index <g> \
        --random-envs 2 3 --forced-resets 2:150 3:300 --run-name m1_cut_n4

Raw output: a new ``artifacts/rl_ready/p1_5/<run-name>_NNN/`` (never overwritten).
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

T_SCRIPT = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_5"
_spec = importlib.util.spec_from_file_location("run_vector_milestones", ROOT / "scripts/rl_ready/run_vector_milestones.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
M.OUT_ROOT = OUT_ROOT


def dump(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=1, sort_keys=True, allow_nan=True,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def digit_counts(codes, max_cuts=4):
    from robotic_cutting_sim.rl.replay import digit_counts as dc
    return dc(codes, max_cuts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--random-envs", type=int, nargs="*", default=[])
    p.add_argument("--random-arm-probability", type=float, default=.2)
    p.add_argument("--forced-resets", nargs="*", default=[], help="env:global_step")
    p.add_argument("--evaluation-every", type=int, default=20)
    p.add_argument("--particle-contact-mode", default="frozen_v1")
    p.add_argument("--audit-steps", type=int, nargs="*", default=[1, 350, 700])
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--max-steps", type=int, help="truncate every episode (smoke only; never a row)")
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    run = M.new_run_dir(args.run_name)
    manifest = M.manifest(args, run)
    telemetry = M.telemetry_start(args.gpu_index)
    status, error, result = "FAIL", None, {}
    try:
        result = run_episodes(args, run, telemetry)
        status = "COMPLETE" if args.max_steps is None else "TRUNCATED"
    except Exception:
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        summary = telemetry.stop()
        with (run / "telemetry_samples.jsonl").open("x") as stream:
            for sample in telemetry.samples:
                stream.write(json.dumps(sample, default=str) + "\n")
    out = dict(status=status, error=error, telemetry=summary, telemetry_run_phase=telemetry.summary(phases=("baseline", "run")),
               wall_s=time.perf_counter() - T_SCRIPT, git_head=manifest["git_head"], **result)
    dump(run / "result.json", out)
    print(json.dumps(dict(run=str(run.relative_to(ROOT / "artifacts")), status=status,
                          verdicts=[e.get("verdict") for e in result.get("episodes", [])],
                          gpu_sharing=summary.get("gpu_sharing"), wall_s=out["wall_s"])))
    if error:
        raise SystemExit(1)


def run_episodes(args, run, telemetry):
    import torch
    import warp as wp
    from robotic_cutting_sim.batched.material_batch import live_device_hashgrids
    from robotic_cutting_sim.rl import ConfiguredTaskFamily, open_vector_env
    from robotic_cutting_sim.rl.policies_batched import RandomPolicyV1
    from robotic_cutting_sim.rl.reference_policy import ReferenceCommandPolicyV1
    M._grouped_query_audit = M._define_kernels()
    N = args.num_envs
    forced = {}
    for item in args.forced_resets:
        e, step = (int(x) for x in item.split(":"))
        forced.setdefault(step, []).append(e)
    random_mask_host = np.zeros(N, bool)
    random_mask_host[list(args.random_envs)] = True
    family = ConfiguredTaskFamily.from_cut_workload(N, env_seeds=tuple(1000 + e for e in range(N)))
    census = dict(before_open=len(live_device_hashgrids()))
    t0 = time.perf_counter()
    with open_vector_env(N, task=family, device="cuda", strict=False, particle_contact_mode=args.particle_contact_mode,
                         evaluation_stream={"every": args.evaluation_every}) as env:
        open_s = time.perf_counter() - t0
        census["after_open"] = len(live_device_hashgrids())
        dev = env.torch_device
        sync = lambda: torch.cuda.synchronize(dev)  # noqa: E731
        ref = ReferenceCommandPolicyV1.from_tasks(env.tasks, device=dev)
        rnd = RandomPolicyV1(arm_probability=args.random_arm_probability)
        random_mask = torch.as_tensor(random_mask_host, device=dev)
        steps = env.max_episode_steps if args.max_steps is None else min(args.max_steps, env.max_episode_steps)
        monitor = M.L2Monitor(env)
        step_walls, reset_walls, episodes, audits = [], [], [], []
        terminated_events = np.zeros(N, np.int64)
        global_step = 0
        rewards = np.zeros(N)
        for episode in range(args.episodes):
            (obs, info), wall = M.timed(lambda: env.reset(seeds=[1000 + e + 100 * episode for e in range(N)]), sync)
            reset_walls.append(dict(kind="full", wall_s=wall, episode=episode))
            census[f"after_reset_{episode}"] = len(live_device_hashgrids())
            monitor.on_reset(range(N))
            if episode == 0:
                telemetry.set_phase("run")
            episode_rewards = np.zeros(N)
            for k in range(steps):
                actions = torch.where(random_mask[:, None], rnd(obs, info), ref(obs, info))
                (obs, reward, terminated, truncated, info), wall = M.timed(lambda: env.step(actions), sync)
                step_walls.append(wall)
                global_step += 1
                monitor.update()
                episode_rewards += reward.cpu().numpy()
                terminated_events += terminated.cpu().numpy()
                if k + 1 in args.audit_steps and episode == 0:
                    audits.append(monitor.audit(k + 1))
                for e in forced.get(global_step, []):
                    ev = env.evaluate([e])[0]
                    episodes.append(episode_row(env, ev, e, episode, "TRUNCATED_BY_FORCED_RESET", episode_rewards[e],
                                                random_mask_host[e], run))
                    (obs, info), wall = M.timed(lambda: env.reset(env_ids=[e], seeds=[5000 + e + global_step]), sync)
                    reset_walls.append(dict(kind="partial", env=e, wall_s=wall, global_step=global_step))
                    monitor.on_reset([e])
                    episode_rewards[e] = 0.
            telemetry.set_phase("evaluate")
            for ev in env.evaluate():
                e = ev["env"]
                end = "TRUNCATED_TIME_LIMIT" if ev["step_count"] >= env.max_episode_steps else "TRUNCATED_RUN_END"
                episodes.append(episode_row(env, ev, e, episode, end, episode_rewards[e], random_mask_host[e], run))
            telemetry.set_phase("run")
        census["end"] = len(live_device_hashgrids())
        particles = np.diff(np.asarray(env.material.packed.particle_offset)).tolist()
        receipt = env.receipt()
        l2 = monitor.result()
        l2["audits"] = audits
    walls = np.asarray(step_walls)
    warm = walls[1:] if len(walls) > 1 else walls
    partial = [r["wall_s"] for r in reset_walls if r["kind"] == "partial"]
    full = [r["wall_s"] for r in reset_walls if r["kind"] == "full"]
    timing = dict(open_s=open_s, steps=int(len(walls)), first_step_s=float(walls[0]),
                  warm_step_mean_s=float(warm.mean()), warm_step_median_s=float(np.median(warm)),
                  warm_step_p95_s=float(np.percentile(warm, 95)), warm_batch_steps_per_s=float(len(warm) / warm.sum()),
                  warm_env_steps_per_s=float(N * len(warm) / warm.sum()),
                  warm_aggregate_sim_time_per_wall=float(N * len(warm) * env.control_dt / warm.sum()),
                  full_resets_s=full, partial_resets_s=partial,
                  partial_reset_p95_s=float(np.percentile(partial, 95)) if partial else None,
                  full_reset_p95_s=float(np.percentile(full, 95)) if full else None,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
                  mempool_high_bytes=int(wp.get_mempool_used_mem_high("cuda:0")))
    np.savez_compressed(run / "step_walls.npz", step_wall_s=walls.astype(np.float32))
    return dict(num_envs=N, episodes_requested=args.episodes, steps_per_episode=steps, random_envs=list(args.random_envs),
                forced_resets=args.forced_resets, evaluation_every=args.evaluation_every, timing=timing, l2=l2,
                hashgrid_census=census, terminated_events=terminated_events.tolist(), episodes=episodes,
                receipt=dict(fidelity=receipt["fidelity"], reward=receipt["reward"], material=receipt["material"],
                             evaluation_stream=receipt["evaluation_stream"], evaluation_mode=receipt["evaluation_mode"],
                             tasks=receipt["tasks"], task_family=receipt["task_family"], versions=receipt["versions"],
                             input_guard=receipt["input_guard"], device_hashgrid_guard=receipt["device_hashgrid_guard"]),
                particles_per_env=particles)


def episode_row(env, ev, e, episode, end, ret, is_random, run):
    snap = env.snapshot_world(e)
    name = f"final_env{e:02d}_ep{episode:02d}_{end.lower()}.npz"
    if not (run / name).exists():
        np.savez_compressed(run / name, codes=snap["codes"].astype(np.int32), positions=snap["positions"].astype(np.float32),
                            step=np.int64(snap["step_count"]))
    s = ev["summary"]
    return dict(env=e, episode=episode, end=end, drive="random_policy_v1" if is_random else "reference_command_policy_v1",
                step_count=ev["step_count"], frames=ev["frames"], evaluation_mode=ev["evaluation_mode"],
                stream_samples=ev.get("stream_samples"), stream_dropped_samples=ev.get("stream_dropped_samples"),
                verdict=s["verdict"], integrity=s["integrity"], outcome=s["outcome"], outcome_checks=s["outcome_checks"],
                mechanical_components=s["mechanical_components"], piece_mass_fractions=s.get("piece_mass_fractions"),
                labelled=ev["labelled"], labelled_by_digit=digit_counts(snap["codes"]),
                final_codes_sha256=ev["final_codes_sha256"], undiscounted_shaping_return=float(ret),
                evaluation_wall_s=ev["wall_s"], final_state_file=name)


if __name__ == "__main__":
    main()
