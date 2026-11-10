#!/usr/bin/env python3
"""P1-5 T5.12: RL pipeline smoke (plan section 5; no training, no gradient step).

One process, one ``VectorCuttingEnv`` (one material, one device HashGrid), ``autoreset="same_step"``,
``strict=False`` (invalid envs terminate, are reset and counted), ``terminal_capture=True`` (finished episodes are
evaluated with ``evaluate(episode="last_completed")``, host, outside timing, never feeding the reward). Scripted envs
follow ``reference_command_policy_v1`` (the compiled planner schedule; for TRIM optionally the WP-17
``recorded_command_policy_v1``), the others ``RandomPolicyV1``; random envs are additionally force-reset on a
SplitMix64 schedule (every 50-300 steps). Transitions are written as ``rcs_rl_rollout_v1`` (``rl.rollout``) and
checked by ``check_rl_rollout.py``. Runs are declared in ``configs/rl_benchmark/rl_smoke_v1.json``.

    CUDA_VISIBLE_DEVICES=<g> TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock <gpu<g>.lock> \
        .deps/isaacsim-venv/bin/python scripts/rl_ready/run_rl_smoke.py --run S-CUT --gpu-index <g>

``--cpu-test`` (used by ``tests/test_rl_smoke.py``) runs the S-CPU variant on the Warp/Torch CPU device in a given
directory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

T_SCRIPT = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

DECLARATION = ROOT / "configs" / "rl_benchmark" / "rl_smoke_v1.json"
OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_5"


def splitmix64_int(x):
    m = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return z ^ (z >> 31)


def forced_schedule(seed, envs, horizon, low=50, high=300):
    """{global step: [env ids]} forced resets of ``envs``: gaps uniform in [low, high] from SplitMix64."""
    out, counter = {}, 0
    for e in envs:
        t = 0
        while True:
            counter += 1
            t += low + splitmix64_int(seed * 1000003 + e * 7919 + counter) % (high - low + 1)
            if t >= horizon:
                break
            out.setdefault(int(t), []).append(int(e))
    return out


def build_family(spec, cpu=False):
    from robotic_cutting_sim.rl import ConfiguredTaskFamily, TaskFamily
    N = spec["num_envs"]
    extra = {}
    if spec.get("particle_spacing_m") is not None:
        extra["particle_spacing_m"] = spec["particle_spacing_m"]
    if spec.get("max_episode_steps") is not None:
        extra["max_episode_steps"] = spec["max_episode_steps"]
    if spec["task"] == "cut_v7_2_v1":
        return ConfiguredTaskFamily.from_cut_workload(N, env_seeds=tuple(spec["env_seeds"]), **extra)
    if spec["task"] == "seeded_trim":
        poses = None
        if "recorded_command" in spec["drives"]:
            # recorded commands are edge deltas encoded from the WP-17 measured reset edge: every env starts there
            # (as P1-2 M1); the planner HOME is ~1.06 mm away and made the first S-TRIM run fail (preserved)
            from robotic_cutting_sim.rl.policies_batched import load_recorded_commands
            reset = {s: load_recorded_commands(ROOT / "artifacts/wp17/native_optimized_001" / f"seed_{s}"
                                               / "box/env/recording/attempt_000")["reset_edge"]
                     for s in set(spec["scene_seeds"])}
            poses = tuple(tuple(map(float, reset[s])) for s in spec["scene_seeds"])
        return TaskFamily(scene_seeds=tuple(spec["scene_seeds"]), env_seeds=tuple(spec["env_seeds"]),
                          initial_edge_poses=poses, **extra)
    raise ValueError(spec["task"])


def scripted_tables(env, spec):
    """Per-env action table index and the tables (reference schedule per distinct task, or WP-17 commands)."""
    from robotic_cutting_sim.rl.reference_policy import reference_command_actions
    tables, index, sources = [], [], []
    keys = {}
    for e, drive in enumerate(spec["drives"]):
        if drive == "random":
            index.append(-1)
            continue
        task = env.tasks[e]
        if drive == "recorded_command" and spec["task"] == "seeded_trim":
            from robotic_cutting_sim.rl.policies_batched import load_recorded_commands, recorded_command_actions
            seed = env.task_family.scene_seeds[e]
            key = ("recorded", seed)
            if key not in keys:
                attempt = ROOT / "artifacts/wp17/native_optimized_001" / f"seed_{seed}" / "box/env/recording/attempt_000"
                rec = load_recorded_commands(attempt)
                actions, _, _ = recorded_command_actions(rec["targets"], rec["cut_enabled"], rec["cut_index"],
                                                         rec["reset_edge"], task.action_limits(),
                                                         task.T_world_workpiece, task.max_cuts)
                keys[key] = len(tables)
                tables.append(actions)
                sources.append(dict(kind="recorded_command_policy_v1", seed=seed, actions_sha256=rec["actions_sha256"]))
        else:
            key = ("reference", task.config_sha256)
            if key not in keys:
                actions, _, _, report = reference_command_actions(task, env.max_episode_steps)
                keys[key] = len(tables)
                tables.append(actions)
                sources.append(dict(kind="reference_command_policy_v1", task_id=task.task_id,
                                    residual_m=report["max_position_diff_m"]))
        index.append(keys[key])
    return tables, index, sources


def run(spec, out_dir, *, device, gpu_index=None, cpu=False):
    import torch
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.policies_batched import RandomPolicyV1
    from robotic_cutting_sim.rl.reference_policy import ReferenceCommandPolicyV1
    from robotic_cutting_sim.rl.rollout import (END_FORCED, END_INVALID, END_TIME_LIMIT, RESET_AUTO, RESET_NONE,
                                                RolloutWriter)
    from robotic_cutting_sim.rl.termination import decode_termination
    telemetry = None
    if gpu_index is not None:
        from robotic_cutting_sim.rl.telemetry import GpuTelemetry
        telemetry = GpuTelemetry(gpu_index, interval_s=1.)
        telemetry.set_phase("baseline")
        telemetry.start()
        time.sleep(3.)
        telemetry.set_phase("setup")
    N, T = spec["num_envs"], spec["batch_steps"]
    family = build_family(spec, cpu=cpu)
    options = dict(device=device, strict=False, autoreset="same_step", terminal_capture=True,
                   observation_profile=spec.get("observation_profile", "state"))
    if cpu:
        options["parity_mode"] = True
    walls, reset_walls, evaluations = [], [], []
    with open_vector_env(N, task=family, **options) as env:
        dev = env.torch_device
        sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
        tables, index, sources = scripted_tables(env, spec)
        scripted = ReferenceCommandPolicyV1(tables, index, device=dev) if tables else None
        random_policy = RandomPolicyV1(arm_probability=spec.get("random_arm_probability", .2))
        random_mask = torch.as_tensor([d == "random" for d in spec["drives"]], device=dev)
        forced = forced_schedule(spec["forced_reset_seed"], [e for e, d in enumerate(spec["drives"]) if d == "random"],
                                 T, *spec.get("forced_reset_gap", (50, 300)))
        critic = env.observation.critic_dim if env.observation_profile == "privileged_state" else None
        receipt = env.receipt()
        writer = RolloutWriter(out_dir, N, obs_dim=env.observation.policy_dim, chunk_size=spec.get("chunk_size", 256),
                               device=dev, critic_dim=critic,
                               manifest=dict(run=spec["id"], declaration=str(DECLARATION.relative_to(ROOT)),
                                             spec=spec, env_receipt=receipt,
                                             single_observation_space=dict(shape=[env.observation.policy_dim],
                                                                           dtype="float32", low="-inf", high="inf"),
                                             single_action_space=dict(shape=[8], dtype="float32", low=-1., high=1.),
                                             policies=dict(drives=spec["drives"], scripted_tables=sources,
                                                           random=dict(id=random_policy.version,
                                                                       arm_probability=random_policy.arm_probability)),
                                             forced_reset_schedule={str(k): v for k, v in sorted(forced.items())},
                                             autoreset="same_step", device=str(dev)))
        obs, info = env.reset(seeds=list(spec["env_seeds"]))
        ep_return = np.zeros(N)
        ep_start_episode = info["episode"].cpu().numpy().copy()
        if telemetry is not None:
            telemetry.set_phase("run")
        counts = dict(TRUNCATED_TIME_LIMIT=0, TERMINATED_INVALID=0, TRUNCATED_BY_FORCED_RESET=0)
        for t in range(T):
            # policies read the env's LIVE counters (after a same-step autoreset info["step_count"] is the finished
            # episode's length, the live counter is 0)
            live = {"step_count": env._step_count.clone(), "seed": env._seed.clone(), "episode": env.episode.clone()}
            actions = torch.where(random_mask[:, None], random_policy(obs, live),
                                  scripted(obs, live) if scripted is not None else random_policy(obs, live))
            sync()
            start = time.perf_counter()
            next_obs, reward, terminated, truncated, info = env.step(actions)
            sync()
            walls.append(time.perf_counter() - start)
            done = terminated | truncated
            stored_next = {k: torch.where(done[:, None], info["final_obs"][k], next_obs[k]) for k in next_obs}
            writer.add(obs=obs, actions=actions.clamp(-1., 1.), reward=reward, terminated=terminated,
                       truncated=truncated, next_obs=stored_next, info=info,
                       reset_kind=torch.where(done, RESET_AUTO, RESET_NONE))
            ep_return += reward.cpu().numpy()
            done_h = done.cpu().numpy()
            if done_h.any():
                reason = info["termination_reason"].cpu().numpy()
                steps = info["step_count"].cpu().numpy()
                ended = [int(e) for e in np.flatnonzero(done_h)]
                results = {r["env"]: r for r in env.evaluate(ended, episode="last_completed")}
                for e in ended:
                    end = END_INVALID if terminated[e] else END_TIME_LIMIT
                    counts[end] += 1
                    r = results[e]
                    evaluations.append(dict(env=e, verdict=r["summary"]["verdict"], frames=r["frames"]))
                    writer.episode_record(env=e, episode=int(ep_start_episode[e]), seed=int(info["seed"][e]),
                                          length=int(steps[e]), undiscounted_return=float(ep_return[e]), end=end,
                                          termination_reason=int(reason[e]),
                                          termination_bits=decode_termination(int(reason[e])),
                                          drive=spec["drives"][e], evaluator=dict(
                                              verdict=r["summary"]["verdict"], integrity=r["summary"]["integrity"],
                                              outcome=r["summary"]["outcome"], frames=r["frames"],
                                              mode=r["evaluation_mode"], scored="last_completed",
                                              mechanical_components=r["summary"]["mechanical_components"]))
                    ep_return[e] = 0.
                    ep_start_episode[e] += 1
            obs = next_obs
            if t + 1 in forced and t + 1 < T:
                ids = forced[t + 1]
                marked = [e for e in ids if not done_h[e]]     # an env that just autoreset has no open transition
                if marked:
                    writer.mark_forced_reset(marked, obs)
                steps = info["step_count"].cpu().numpy()
                for e in ids:
                    if done_h[e]:       # just autoreset: the forced reset only restarts a fresh episode
                        continue
                    counts[END_FORCED] += 1
                    writer.episode_record(env=e, episode=int(ep_start_episode[e]), seed=int(info["seed"][e]),
                                          length=int(steps[e]), undiscounted_return=float(ep_return[e]), end=END_FORCED,
                                          termination_reason=0, termination_bits=[], drive=spec["drives"][e],
                                          evaluator=None)
                    ep_return[e] = 0.
                    ep_start_episode[e] += 1
                seeds = [int(splitmix64_int(spec["forced_reset_seed"] + 31 * e + t) % (2**62)) for e in ids]
                sync()
                start = time.perf_counter()
                obs, info = env.reset(env_ids=ids, seeds=seeds)
                sync()
                reset_walls.append(time.perf_counter() - start)
                # a forced reset after an autoreset in the same step restarts that env again: its episode counter moved
                ep_start_episode[ids] = info["episode"].cpu().numpy()[ids]
        receipt_end = env.receipt()
    walls = np.asarray(walls)
    summary = None if telemetry is None else telemetry.stop()
    manifest = writer.close(
        end_counts=counts, evaluations=evaluations,
        timing=dict(scope="SMOKE, not a benchmark row: env.step wall (synchronized), forced resets separately",
                    steps=int(len(walls)), step_mean_s=float(walls.mean()), step_p95_s=float(np.percentile(walls, 95)),
                    env_steps_per_s=float(N * len(walls) / walls.sum()),
                    forced_reset_p50_s=float(np.percentile(reset_walls, 50)) if reset_walls else None,
                    forced_reset_p95_s=float(np.percentile(reset_walls, 95)) if reset_walls else None),
        telemetry=summary, autoreset_count=receipt_end.get("terminal_capture"),
        git_head=os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        git_dirty=bool(os.popen(f"git -C {ROOT} status --porcelain --untracked-files=no").read().strip()))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run id in rl_smoke_v1.json (S-TRIM, S-CUT, S-CPU)")
    p.add_argument("--gpu-index", type=int)
    p.add_argument("--cpu-test", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    declaration = json.loads(DECLARATION.read_text())
    spec = next(r for r in declaration["runs"] if r["id"] == a.run)
    if a.out is None:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        for i in range(1, 1000):
            out = OUT_ROOT / f"smoke_{spec['id'].lower().replace('-', '_')}_n{spec['num_envs']}_{i:03d}"
            if not out.exists():
                break
    else:
        out = a.out
    manifest = run(spec, out, device="cpu" if a.cpu_test else "cuda", gpu_index=a.gpu_index, cpu=a.cpu_test)
    print(json.dumps(dict(out=str(out), steps=manifest["steps"], end_counts=manifest["end_counts"],
                          evaluations=manifest["evaluations"], timing=manifest["timing"],
                          wall_s=time.perf_counter() - T_SCRIPT)))


if __name__ == "__main__":
    main()
