#!/usr/bin/env python3
"""P1-2 milestones M0/M1/M2 of the ``rl_gpu_headless`` vector env on workload A (ADR 0021 §13).

Full fidelity material: seeded TRIM seeds 512/15 (WP-17 configuration), 1 mm particles, 4 mm grid, 0.1 ms
material dt, 200 substeps per 50 Hz control step, tlmpm_v7_2 law 8, K=2, S=8; no cameras, no recording. The
batch keeps the Native obstacle set (handle + two measured finger pads, ``handle_and_measured_pads_v1``).
Workload A is always ``reduced_fidelity`` (ideal edge tracking, no robot dynamics, reaction not fed back).

Modes (one process each; run under the GPU lock, ``CUDA_VISIBLE_DEVICES=<gpu>``, ``TMPDIR=/mnt/data/ningan/tmp``,
``PYTHONPATH=src``, ``.deps/isaacsim-venv/bin/python``, no Kit):

* ``m0 --seed S``: N=1, reset + a policy step check, then the full WP-17 replay of seed S through
  ``step_measured_edge`` (recorded measured edge poses, resolved digits, measured pad boxes), 121 host samples
  (outside timing) compared with the Native samples, frozen evaluator on 2 frames and on the 121-frame series.
* ``m1``: N=4 (s512 replay, s15 replay, s512 and s15 driven by ``recorded_command_policy_v1`` through ``step``
  with real edge_delta actions in the SAME batched calls), staggered partial resets, L2 invariants every step,
  per-env frozen evaluation of every completed 1200-step episode.
* ``m2 --num-envs N --episodes E --stagger K``: continuous replay rollout (env k replays seed 512 if k even
  else 15), env k is partially reset at step k*K, then runs E full episodes back to back; VRAM/RSS samples per
  100 steps (leak slope), throughput, L3 per completed episode. ``--num-envs 1`` is the N=1 throughput baseline.
* ``l4``: Warp ``DeterministicMode.RUN_TO_RUN`` probe (set before any kernel module is created): two runs of
  N=2 copies of the seed-512 replay over ``--steps`` steps, bitwise comparison copy-vs-copy and run-vs-run.
* ``profile --num-envs N --steps K``: a short replay for an external ``nsys profile --cuda-graph-trace=node``.

Raw output: a new ``artifacts/rl_ready/p1_2/<name>_NNN/`` per run (never overwritten).
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

T_SCRIPT = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

WP17 = ROOT / "artifacts" / "wp17" / "native_optimized_001"
OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_2"
P1_0_BAND = {512: dict(final_labelled_digit0=37958, final_labelled_digit1=0, final_codes_equal_native="4/4"),
             15: dict(final_labelled_digit0=85786, final_labelled_digit1=0, final_codes_equal_native="4/4")}
STEPS = 1200


def process_age_s():
    ticks = os.sysconf("SC_CLK_TCK")
    start = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    return float(Path("/proc/uptime").read_text().split()[0]) - start / ticks


def rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.
    return None


def git(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


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


def dump(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=1, sort_keys=True, allow_nan=True,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def attempt_dir(seed):
    return WP17 / f"seed_{seed}" / "box" / "env" / "recording" / "attempt_000"


# ----------------------------------------------------------------------------- inputs

def load_inputs(seeds):
    from robotic_cutting_sim.rl.replay import extract_trajectory
    from robotic_cutting_sim.rl.policies_batched import load_recorded_commands
    out = {}
    for seed in sorted(set(seeds)):
        traj = extract_trajectory(attempt_dir(seed), locator_root=ROOT / "artifacts")
        out[seed] = dict(trajectory=traj, commands=load_recorded_commands(attempt_dir(seed)))
    return out


def family_for(seeds, inputs):
    from robotic_cutting_sim.rl import TaskFamily
    recipients = {tuple(inputs[s]["trajectory"].pad_recipients) for s in seeds}
    if len(recipients) != 1:
        raise ValueError("pad recipients differ between seeds")
    return TaskFamily(scene_seeds=tuple(seeds),
                      initial_edge_poses=tuple(tuple(inputs[s]["trajectory"].reset_edge_pose_world_xyzw) for s in seeds),
                      pad_boxes_edge=np.stack([inputs[s]["trajectory"].reset_pad_boxes_edge for s in seeds]).tolist(),
                      pad_recipients=recipients.pop())


def check_payloads(env, inputs):
    out = {}
    for e, task in enumerate(env.tasks):
        seed = env.task_family.scene_seeds[e]
        payload_sha = inputs[seed]["trajectory"].task_payload.get("config_sha256")
        out[e] = dict(seed=seed, env_config_sha256=task.config_sha256, recorded_config_sha256=payload_sha,
                      equal=task.config_sha256 == payload_sha)
        if not out[e]["equal"]:
            raise ValueError(f"env {e}: compiled task differs from the WP-17 payload")
    return out


# ----------------------------------------------------------------------------- L2 monitor (device)

class L2Monitor:
    """ADR §5 L2 per env on the device: code monotonicity per digit, finite state, min detF, slot overflow
    (every step), plus host range audits and the instrumented grouped-hash query at ``audit()``."""

    def __init__(self, env):
        import torch
        import warp as wp
        self.torch, self.wp, self.env = torch, wp, env
        s = env.material.solver
        self.s = s
        self.env_of = wp.to_torch(s.env_of_particle).to(torch.int64)
        self.prev = env._codes.clone()
        N, dev = env.num_envs, env.torch_device
        self.violations = torch.zeros(N, dtype=torch.int64, device=dev)
        self.min_det = torch.full((N,), float("inf"), dtype=torch.float32, device=dev)
        self.max_speed = torch.zeros(N, dtype=torch.float32, device=dev)
        self.nonfinite_steps = torch.zeros(N, dtype=torch.int64, device=dev)
        self.slot_overflow_max = torch.zeros(N, dtype=torch.int64, device=dev)
        self.invalid_any = torch.zeros(N, dtype=torch.int64, device=dev)
        self.summary = wp.to_torch(env.material.diagnostics.summary)
        self.slot_overflow = wp.to_torch(s.slot_overflow)
        self.mass0 = s.mass.numpy().copy()
        self.steps = 0
        self.audits = []

    def update(self):
        torch = self.torch
        cur = self.env._codes
        bad = torch.zeros_like(cur, dtype=torch.bool)
        for k in range(self.s.max_cuts):
            a, b = (self.prev >> (2 * k)) & 3, (cur >> (2 * k)) & 3
            bad |= (a != 0) & (b != a)
        self.violations.index_add_(0, self.env_of, bad.to(torch.int64))
        self.prev.copy_(cur)
        self.min_det = torch.minimum(self.min_det, self.summary[:, 1])
        self.max_speed = torch.maximum(self.max_speed, self.summary[:, 2])
        self.nonfinite_steps += (self.summary[:, 0] != 0).to(torch.int64)
        self.slot_overflow_max = torch.maximum(self.slot_overflow_max, self.slot_overflow.to(torch.int64))
        self.invalid_any = torch.maximum(self.invalid_any, (self.env._invalid != 0).to(torch.int64))
        self.steps += 1

    def on_reset(self, ids):
        """Reset envs restart from code 0: their previous codes are the reset codes."""
        mask = self.torch.zeros(self.env.num_envs, dtype=self.torch.bool, device=self.env.torch_device)
        mask[list(ids)] = True
        self.prev.copy_(self.torch.where(mask[self.env_of], self.env._codes, self.prev))

    def audit(self, step):
        from robotic_cutting_sim.batched import layout
        wp, s, pk = self.wp, self.s, self.env.material.packed
        with self.env._stream():
            foreign = wp.zeros(1, dtype=int, device=s.device)
            visited = wp.zeros(1, dtype=int, device=s.device)
            wp.launch(_grouped_query_audit, s.total_particles, [s.hash.id, s.x, s.env_of_particle, 2.4 * s.radius,
                                                                 foreign, visited], device=s.device)
            ranges = layout.range_audit(pk, s.nodes.numpy(), s.link_offsets.numpy(), s.link_neighbours.numpy(),
                                        s.pending_q.numpy(), s.env_of_particle.numpy())
            row = dict(step=step, range_audit=ranges, grouped_query_visited=int(visited.numpy()[0]),
                       grouped_query_foreign=int(foreign.numpy()[0]),
                       mass_bitwise_constant=bool(np.array_equal(s.mass.numpy(), self.mass0)))
        self.audits.append(row)
        return row

    def result(self):
        out = dict(steps_checked=self.steps, code_monotone_violations=self.violations.cpu().tolist(),
                   minimum_detF=self.min_det.cpu().tolist(), maximum_material_speed_m_s=self.max_speed.cpu().tolist(),
                   nonfinite_steps=self.nonfinite_steps.cpu().tolist(),
                   slot_overflow_max=self.slot_overflow_max.cpu().tolist(), invalid_any=self.invalid_any.cpu().tolist(),
                   audits=self.audits)
        ok = (all(v == 0 for v in out["code_monotone_violations"]) and all(d > 0 for d in out["minimum_detF"])
              and all(v == 0 for v in out["nonfinite_steps"]) and all(v == 0 for v in out["slot_overflow_max"])
              and all(v == 0 for v in out["invalid_any"])
              and all(set(a["range_audit"].values()) == {0} and a["grouped_query_foreign"] == 0
                      and a["grouped_query_visited"] > 0 and a["mass_bitwise_constant"] for a in self.audits))
        out["status"] = "PASS" if ok else "FAIL"
        return out


def _define_kernels():
    import warp as wp

    @wp.kernel
    def grouped_query_audit(grid: wp.uint64, x: wp.array(dtype=wp.vec3), env_of_particle: wp.array(dtype=int),
                            radius: float, foreign: wp.array(dtype=int), visited: wp.array(dtype=int)):
        p = wp.tid()
        e = env_of_particle[p]
        query = wp.hash_grid_query(grid, x[p], radius, e)
        for q in query:
            wp.atomic_add(visited, 0, 1)
            if env_of_particle[q] != e:
                wp.atomic_add(foreign, 0, 1)
    return grouped_query_audit


_grouped_query_audit = None


# ----------------------------------------------------------------------------- comparisons

def final_comparison(env, e, native_final, max_cuts=2):
    from robotic_cutting_sim.rl.replay import digit_counts
    snap = env.snapshot_world(e)
    codes = snap["codes"]
    return dict(env=e, step_count=snap["step_count"], codes_equal_native=bool(np.array_equal(codes, native_final)),
                different_code_particles=int(np.count_nonzero(codes != native_final)),
                labelled=digit_counts(codes, max_cuts), labelled_native=digit_counts(native_final, max_cuts),
                positions=snap["positions"])


def native_final_codes(seed):
    from robotic_cutting_sim.rl.replay import native_samples
    data, _ = native_samples(attempt_dir(seed))
    return np.asarray(data["codes"][len(data["time_s"]) - 1]).copy(), data


def eval_summary(results):
    return [dict(env=r["env"], scene_seed=r["scene_seed"], step_count=r["step_count"], frames=r["frames"],
                 labelled=r["labelled"], final_codes_sha256=r["final_codes_sha256"], wall_s=r["wall_s"],
                 **r["summary"]) for r in results]


def telemetry_start(gpu_index):
    from robotic_cutting_sim.rl.telemetry import GpuTelemetry
    telemetry = GpuTelemetry(gpu_index, interval_s=1.)
    telemetry.set_phase("baseline")
    telemetry.start()
    time.sleep(3.)
    telemetry.set_phase("setup")
    return telemetry


def manifest(args, run):
    import torch
    import warp
    data = dict(schema="rcs_rl_p1_2_run_v1", argv=sys.argv, args=vars(args), git_head=git("rev-parse", "HEAD"),
                git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")), started_unix_s=time.time(),
                git_diff_stat=git("diff", "--stat", "HEAD"),
                source_sha256_at_start={str(p.relative_to(ROOT)): __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                                        for p in sorted((ROOT / "src" / "robotic_cutting_sim" / "batched").glob("*.py"))
                                        + sorted((ROOT / "src" / "robotic_cutting_sim" / "rl").glob("*.py"))},
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), python=sys.version.split()[0],
                interpreter=os.path.relpath(sys.executable, ROOT), kit_started=False,
                package_origin=os.path.relpath(sys.modules["robotic_cutting_sim"].__file__, ROOT),
                versions=dict(warp=warp.__version__, numpy=np.__version__, torch=torch.__version__),
                workload=dict(id="A", material="seeded_trim tlmpm_v7_2 law 8, 1 mm / 4 mm / 0.1 ms / 200 substeps, "
                              "K=2, S=8", control_hz=50, cameras="none", recording="none", robot_dof=0,
                              obstacles="handle + two measured finger pads (handle_and_measured_pads_v1)"))
    dump(run / "manifest.json", data)
    return data


def timed(fn, sync):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


# ----------------------------------------------------------------------------- modes

def mode_m0(args, run, telemetry):
    import torch
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    from robotic_cutting_sim.rl.replay import compare_sample
    seed = args.seed
    inputs = load_inputs([seed])
    traj = inputs[seed]["trajectory"]
    native_final, native = native_final_codes(seed)
    sync = torch.cuda.synchronize
    out = dict(seed=seed)
    with open_vector_env(1, task=family_for([seed], inputs), device="cuda", strict=True,
                         output_dir=run / "env") as env:
        out["build_wall_s"] = env.build_wall_s
        out["payload_check"] = check_payloads(env, inputs)
        global _grouped_query_audit
        _grouped_query_audit = _define_kernels()
        # API: reset + one real edge_delta step, then a full reset (complete reset/step check)
        (obs, info), first_reset_s = timed(lambda: env.reset(seeds=[42]), sync)
        hold = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
        hold[:, 6] = -1.
        (o2, r2, te2, tr2, i2), api_step_s = timed(lambda: env.step(hold), sync)
        (obs, info), reset_s = timed(lambda: env.reset(seeds=[42]), sync)
        out["api_check"] = dict(policy_obs_shape=list(obs["policy"].shape), step_status=int(i2["action_status"][0]),
                                step_count_after_step=int(i2["step_count"][0]),
                                step_count_after_reset=int(info["step_count"][0]), first_reset_s=first_reset_s,
                                api_step_s=api_step_s, reset_s=reset_s, reward=float(r2[0]))
        telemetry.set_phase("run")
        driver = MeasuredTrajectoryDriver([traj], [0], device="cuda")
        monitor = L2Monitor(env)
        monitor.audit(0)
        native_steps = np.rint(np.asarray(native["time_s"]) / .02).astype(int)
        frames, comparisons = [], []
        steps = min(STEPS, args.max_global_steps or STEPS)
        walls = np.zeros(steps)
        first_step_age = None
        for t in range(steps):
            pose, cut, pads = driver.inputs(env._step_count)
            _, walls[t] = timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads), sync)
            if t == 0:
                first_step_age = process_age_s()
            monitor.update()
            step = t + 1
            if step in set(native_steps.tolist()):
                snap = env.snapshot_world(0)
                index = int(np.flatnonzero(native_steps == step)[0])
                row = compare_sample({k: native[k][index] for k in ("positions", "velocities", "codes")}, snap, 2)
                row.update(step=step, sample=index)
                comparisons.append(row)
                frames.append({k: snap[k] for k in ("positions", "velocities", "codes", "step_count", "time_s")})
            if step % 300 == 0:
                monitor.audit(step)
        out["l2"] = monitor.result()
        out["final"] = {k: v for k, v in final_comparison(env, 0, native_final).items() if k != "positions"}
        ev2 = env.evaluate([0])
        ev121 = env.evaluate([0], frames={0: frames[:-1]})
        out["evaluation_2_frames"] = eval_summary(ev2)
        out["evaluation_121_frames"] = eval_summary(ev121)
        out["evaluation_2_vs_121_equal"] = dict(
            verdict=ev2[0]["result"]["verdict"] == ev121[0]["result"]["verdict"],
            outcome=ev2[0]["result"]["outcome"] == ev121[0]["result"]["outcome"])
        out["parity_vs_native"] = dict(
            samples=len(comparisons), max_position_diff_m_over_time=max(c["max_position_diff_m"] for c in comparisons),
            final_max_position_diff_m=comparisons[-1]["max_position_diff_m"],
            final_mean_position_diff_m=comparisons[-1]["mean_position_diff_m"],
            samples_with_code_difference=[dict(step=c["step"], particles=c["different_code_particles"])
                                          for c in comparisons if c["different_code_particles"]],
            p1_0_band=P1_0_BAND[seed])
        out["complete"] = steps == STEPS
        out["timing"] = dict(steps=steps, warm_step_mean_s=float(walls[1:].mean()),
                             warm_step_median_s=float(np.median(walls[1:])), warm_step_p95_s=float(np.percentile(
                                 walls[1:], 95)), first_step_s=float(walls[0]),
                             warm_env_steps_per_s=float((steps - 1) / walls[1:].sum()),
                             cold_process_start_to_first_step_done_s=first_step_age,
                             timing_scope="env.step_measured_edge wall incl. strict int32 readback, device "
                                          "synchronize before/after; L2 monitor and 121 samples outside")
        np.savez_compressed(run / "steps.npz", step_wall_s=walls.astype(np.float32))
        out["receipt_fidelity"] = env.receipt()["fidelity"]
    out["mempool_high_bytes"] = mempool_high()
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    return out


def mempool_high():
    import warp as wp
    return int(wp.get_mempool_used_mem_high("cuda:0")) if wp.is_mempool_enabled("cuda:0") else None


def mode_m1(args, run, telemetry):
    import torch
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.policies_batched import (MeasuredTrajectoryDriver, RecordedCommandPolicyV1,
                                                         recorded_command_actions)
    seeds = (512, 15, 512, 15)
    replay = torch.tensor([True, True, False, False], device="cuda")
    inputs = load_inputs(seeds)
    natives = {s: native_final_codes(s)[0] for s in (512, 15)}
    sync = torch.cuda.synchronize
    out = dict(seeds=list(seeds), drive=["replay", "replay", "recorded_command_policy_v1", "recorded_command_policy_v1"])
    reset_at = {0: args.m1_reset_steps[0], 1: args.m1_reset_steps[1], 2: args.m1_reset_steps[2],
                3: args.m1_reset_steps[3]}
    with open_vector_env(4, task=family_for(seeds, inputs), device="cuda", strict=True,
                         output_dir=run / "env") as env:
        out["build_wall_s"] = env.build_wall_s
        out["payload_check"] = check_payloads(env, inputs)
        global _grouped_query_audit
        _grouped_query_audit = _define_kernels()
        tables, reconstruction = [], {}
        for e in (2, 3):
            s, rec = seeds[e], inputs[seeds[e]]["commands"]
            task = env.tasks[e]
            actions, _, report = recorded_command_actions(rec["targets"], rec["cut_enabled"], rec["cut_index"],
                                                          rec["reset_edge"], env.action_limits, task.T_world_workpiece,
                                                          task.max_cuts)
            tables.append(actions)
            reconstruction[f"env{e}"] = dict(host_frozen_path=report, initial_pose_equals_reset_edge=bool(np.array_equal(
                env.material.initial_measured_pose.numpy()[e], rec["reset_edge"])))
        policy = RecordedCommandPolicyV1(tables, [-1, -1, 0, 1], device="cuda")
        targets = torch.as_tensor(np.stack([inputs[s]["commands"]["targets"] for s in (512, 15)]), device="cuda")
        driver = MeasuredTrajectoryDriver([inputs[512]["trajectory"], inputs[15]["trajectory"]], [0, 1, -1, -1],
                                          device="cuda")
        (obs, info), first_reset_s = timed(lambda: env.reset(seeds=[42] * 4), sync)
        telemetry.set_phase("run")
        monitor = L2Monitor(env)
        monitor.audit(0)
        target_err = torch.zeros(4, dtype=torch.float64, device="cuda")
        quat_err = torch.zeros(4, dtype=torch.float64, device="cuda")
        host_steps = np.zeros(4, int)
        episodes = {e: [] for e in range(4)}
        partial_resets, reset_walls, step_walls, eval_walls = [], [], [], []
        status_counts = torch.zeros((4, 8), dtype=torch.int64, device="cuda")
        done_full = np.zeros(4, int)
        global_step = 0
        while done_full.min() < 1:
            policy_info = dict(step_count=env._step_count, seed=env._seed, episode=env.episode)
            actions = policy(None, policy_info)
            pose, cut, pads = driver.inputs(env._step_count)
            (_, _, _, _, info), wall = timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads,
                                                                             env_mask=replay, actions=actions), sync)
            step_walls.append(wall)
            global_step += 1
            host_steps += 1
            monitor.update()
            # recorded_command_policy_v1 envs: commanded pose vs recorded target of this step
            sc = env._step_count.clamp(1, STEPS) - 1
            tgt = targets[torch.tensor([0, 1, 0, 1], device="cuda"), sc]
            pos = (env.commanded_pose[:, :3] - tgt[:, :3]).abs().amax(1)
            q, qt = env.commanded_pose[:, 3:], tgt[:, 3:]
            dq = torch.minimum((q - qt).abs().amax(1), (q + qt).abs().amax(1))
            valid = (~replay) & (env._step_count <= STEPS) & (env._step_count >= 1)
            target_err = torch.maximum(target_err, torch.where(valid, pos, 0.))
            quat_err = torch.maximum(quat_err, torch.where(valid, dq, 0.))
            st = info["action_status"].clamp(min=-1) + 1
            status_counts[torch.arange(4, device="cuda"), st.long()] += 1
            if global_step % 300 == 0:
                monitor.audit(global_step)
            finished = [e for e in range(4) if host_steps[e] == STEPS and (len(episodes[e]) > 0 or
                                                                           global_step > reset_at[e])]
            staggered = [e for e in range(4) if global_step == reset_at[e]]
            if finished:
                start = time.perf_counter()
                results = env.evaluate(finished)
                eval_walls.append(time.perf_counter() - start)
                for e, r in zip(finished, results):
                    comp = final_comparison(env, e, natives[seeds[e]])
                    comp.pop("positions")
                    episodes[e].append(dict(global_step=global_step, evaluation=eval_summary([r])[0], vs_native=comp))
                    done_full[e] += 1
            ids = sorted(set(staggered) | set(e for e in finished if done_full[e] < 1))
            if ids:
                _, wall = timed(lambda: env.reset(env_ids=ids, seeds=[42] * len(ids)), sync)
                reset_walls.append(wall)
                monitor.on_reset(ids)
                for e in ids:
                    partial_resets.append(dict(env=e, global_step=global_step, steps_before=int(host_steps[e]),
                                               kind="staggered" if e in staggered else "episode_end"))
                    host_steps[e] = 0
            if args.max_global_steps and global_step >= args.max_global_steps:
                out["stopped_at_max_global_steps"] = global_step
                break
        monitor.audit(global_step)
        out["l2"] = monitor.result()
        out["episodes"] = {f"env{e}": v for e, v in episodes.items()}
        out["partial_resets"] = partial_resets
        out["recorded_command_policy"] = dict(
            reconstruction, device_max_position_diff_m=target_err.cpu().tolist(),
            device_max_quaternion_component_diff=quat_err.cpu().tolist(),
            status_counts_columns=["REPLAY(-1)"] + [str(i) for i in range(7)],
            status_counts=status_counts.cpu().tolist())
        walls = np.asarray(step_walls)
        out["timing"] = dict(global_steps=global_step, step_mean_s=float(walls[1:].mean()),
                             step_p95_s=float(np.percentile(walls[1:], 95)),
                             env_steps_per_s=float(4 * (len(walls) - 1) / walls[1:].sum()),
                             partial_reset_mean_s=float(np.mean(reset_walls)),
                             partial_reset_p95_s=float(np.percentile(reset_walls, 95)), first_reset_s=first_reset_s,
                             evaluation_walls_s=eval_walls)
        out["receipt_fidelity"] = env.receipt()["fidelity"]
    out["mempool_high_bytes"] = mempool_high()
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    return out


def mode_m2(args, run, telemetry):
    import torch
    import warp as wp
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    N = args.num_envs
    seeds = tuple(args.seed if N == 1 else (512 if k % 2 == 0 else 15) for k in range(N))
    inputs = load_inputs(seeds)
    natives = {s: native_final_codes(s)[0] for s in set(seeds)}
    sync = torch.cuda.synchronize
    order = sorted(set(seeds))
    out = dict(num_envs=N, seeds=list(seeds), episodes_per_env=args.episodes, stagger=args.stagger)
    with open_vector_env(N, task=family_for(seeds, inputs), device="cuda", strict=True,
                         output_dir=run / "env") as env:
        out["build_wall_s"] = env.build_wall_s
        out["payload_check"] = check_payloads(env, inputs)
        global _grouped_query_audit
        _grouped_query_audit = _define_kernels()
        driver = MeasuredTrajectoryDriver([inputs[s]["trajectory"] for s in order],
                                          [order.index(s) for s in seeds], device="cuda")
        (obs, info), first_reset_s = timed(lambda: env.reset(seeds=[42] * N), sync)
        telemetry.set_phase("run")
        monitor = L2Monitor(env)
        monitor.audit(0)
        reset_at = {e: e * args.stagger for e in range(1, N)}
        host_steps = np.zeros(N, int)
        full_done = np.zeros(N, int)
        in_full = np.array([e not in reset_at for e in range(N)])   # env 0 starts its first full episode at 0
        episodes, resets, step_walls, reset_walls, eval_walls, memory = [], [], [], [], [], []
        global_step = 0
        first_step_age = None
        while full_done.min() < args.episodes:
            pose, cut, pads = driver.inputs(env._step_count)
            _, wall = timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads), sync)
            if first_step_age is None:
                first_step_age = process_age_s()
            step_walls.append(wall)
            global_step += 1
            host_steps += 1
            if args.l2_every and global_step % args.l2_every == 0:
                monitor.update()
            if global_step % 100 == 0:
                memory.append(dict(step=global_step, rss_mib=rss_mib(), t=time.time(),
                                   mempool_current=int(wp.get_mempool_used_mem_current("cuda:0")),
                                   torch_allocated=int(torch.cuda.memory_allocated()),
                                   torch_reserved=int(torch.cuda.memory_reserved())))
            if global_step % 600 == 0:
                monitor.audit(global_step)
            finished = [e for e in range(N) if in_full[e] and host_steps[e] == STEPS]
            staggered = [e for e in range(N) if reset_at.get(e) == global_step]
            if finished:
                start = time.perf_counter()
                results = env.evaluate(finished)
                eval_walls.append(time.perf_counter() - start)
                for e, r in zip(finished, results):
                    comp = final_comparison(env, e, natives[seeds[e]])
                    comp.pop("positions")
                    episodes.append(dict(env=e, seed=seeds[e], episode=int(full_done[e]) + 1, global_step=global_step,
                                         evaluation=eval_summary([r])[0], vs_native=comp))
                    full_done[e] += 1
            ids = sorted(set(staggered) | set(e for e in finished if full_done[e] < args.episodes))
            if ids:
                _, wall = timed(lambda: env.reset(env_ids=ids, seeds=[42] * len(ids)), sync)
                reset_walls.append(wall)
                if args.l2_every:
                    monitor.on_reset(ids)
                for e in ids:
                    resets.append(dict(env=e, global_step=global_step, steps_before=int(host_steps[e]),
                                       kind="staggered" if e in staggered else "episode_end"))
                    host_steps[e] = 0
                    in_full[e] = True
            if args.max_global_steps and global_step >= args.max_global_steps:
                out["stopped_at_max_global_steps"] = global_step
                break
        monitor.audit(global_step)
        out["l2"] = monitor.result()
        out["l2"]["update_every_steps"] = args.l2_every
        out["episodes"] = episodes
        out["resets"] = resets
        walls = np.asarray(step_walls)
        np.savez_compressed(run / "steps.npz", step_wall_s=walls.astype(np.float32))
        mem_steps = np.array([m["step"] for m in memory], float)

        def slope(key, scale=1.):
            if len(memory) < 3:
                return None
            values = np.array([m[key] for m in memory], float) * scale
            keep = mem_steps >= STEPS     # after the first episode (allocations of the first pass settled)
            if keep.sum() < 3:
                keep = np.ones_like(keep)
            return float(np.polyfit(mem_steps[keep], values[keep], 1)[0] * 1000.)   # per 1000 steps
        out["memory"] = dict(samples=memory, rss_mib_slope_per_1000_steps=slope("rss_mib"),
                             mempool_bytes_slope_per_1000_steps=slope("mempool_current"),
                             torch_allocated_bytes_slope_per_1000_steps=slope("torch_allocated"),
                             mempool_high_bytes=mempool_high())
        out["timing"] = dict(global_steps=global_step, first_step_s=float(walls[0]),
                             step_mean_s=float(walls[1:].mean()), step_median_s=float(np.median(walls[1:])),
                             step_p95_s=float(np.percentile(walls[1:], 95)),
                             env_steps_per_s=float(N * (len(walls) - 1) / walls[1:].sum()),
                             env_steps_per_s_including_resets=float(N * (len(walls) - 1) / (walls[1:].sum()
                                                                                             + sum(reset_walls))),
                             partial_reset_mean_s=float(np.mean(reset_walls)) if reset_walls else None,
                             partial_reset_p95_s=float(np.percentile(reset_walls, 95)) if reset_walls else None,
                             first_reset_s=first_reset_s, evaluation_wall_total_s=float(sum(eval_walls)),
                             cold_process_start_to_first_step_done_s=first_step_age,
                             timing_scope="env.step_measured_edge wall incl. strict int32 readback, device "
                                          "synchronize before/after; L2 monitor, resets and evaluation outside")
        receipt = env.receipt()
        out["receipt_fidelity"] = receipt["fidelity"]
        out["particles_per_env"] = receipt["material"]["layout"]["particle_counts"]
        out["memory_bytes_per_env"] = receipt["material"]["memory_bytes"] / N
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    return out


def mode_profile(args, run, telemetry):
    import torch
    from robotic_cutting_sim.rl import open_vector_env
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    N = args.num_envs
    seeds = tuple([512] * N)
    inputs = load_inputs(seeds)
    with open_vector_env(N, task=family_for(seeds, inputs), device="cuda", strict=True) as env:
        driver = MeasuredTrajectoryDriver([inputs[512]["trajectory"]], [0] * N, device="cuda")
        env.reset(seeds=[42] * N)
        torch.cuda.synchronize()
        telemetry.set_phase("run")
        torch.cuda.cudart().cudaProfilerStart()
        walls = []
        for _ in range(args.steps):
            pose, cut, pads = driver.inputs(env._step_count)
            _, wall = timed(lambda: env.step_measured_edge(pose, cut, pad_boxes=pads), torch.cuda.synchronize)
            walls.append(wall)
        torch.cuda.cudart().cudaProfilerStop()
        return dict(num_envs=N, steps=args.steps, step_walls_s=walls,
                    total_particles=env.material.solver.total_particles)


def mode_l4(args, run, telemetry):
    import warp as wp
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN      # before any kernel module is created
    import torch
    out = dict(mode="DeterministicMode.RUN_TO_RUN", steps=args.steps, num_envs=2)
    try:
        from robotic_cutting_sim.rl import open_vector_env
        from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
        seeds = (512, 512)
        inputs = load_inputs(seeds)
        finals = []
        for repeat in range(2):
            with open_vector_env(2, task=family_for(seeds, inputs), device="cuda", strict=True) as env:
                driver = MeasuredTrajectoryDriver([inputs[512]["trajectory"]], [0, 0], device="cuda")
                env.reset(seeds=[42, 42])
                telemetry.set_phase("run")
                start = time.perf_counter()
                for _ in range(args.steps):
                    pose, cut, pads = driver.inputs(env._step_count)
                    env.step_measured_edge(pose, cut, pad_boxes=pads)
                torch.cuda.synchronize()
                wall = time.perf_counter() - start
                state = [env.material.snapshot(e) for e in range(2)]
                finals.append(dict(wall_s=wall, state=state, labelled=[int(np.count_nonzero(s["code"]))
                                                                       for s in state],
                                   wrench=env.material.pending_wrench.numpy().copy()))
        names = ("x", "v", "F", "code", "impulse", "progress", "stress")

        def diff(a, b):
            return {n: (bool(np.array_equal(a[n], b[n])),
                        float(np.abs(a[n].astype(float) - b[n].astype(float)).max())) for n in names}
        copy = [diff(f["state"][0], f["state"][1]) for f in finals]
        r2r = [diff(finals[0]["state"][e], finals[1]["state"][e]) for e in range(2)]
        bitwise = all(all(v[0] for v in d.values()) for d in copy + r2r)
        out.update(status="PASS" if bitwise else "FAIL", copy_vs_copy=copy, run_vs_run=r2r,
                   labelled=[f["labelled"] for f in finals], walls_s=[f["wall_s"] for f in finals],
                   wrench_run_vs_run_equal=bool(np.array_equal(finals[0]["wrench"], finals[1]["wrench"])),
                   scope="deterministic verification build only; never used for production timing")
    except Exception as error:   # report as is: BLOCKED when the mode cannot compile or launch these kernels
        import traceback
        out.update(status="BLOCKED", error=repr(error), traceback=traceback.format_exc())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("m0", "m1", "m2", "l4", "profile"), required=True)
    parser.add_argument("--seed", type=int, default=512, choices=(512, 15))
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--stagger", type=int, default=50)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--l2-every", type=int, default=1, help="L2 device monitor period in steps (0: audits only)")
    parser.add_argument("--max-global-steps", type=int, default=0)
    parser.add_argument("--m1-reset-steps", type=int, nargs=4, default=[300, 400, 500, 600])
    parser.add_argument("--gpu-index", type=int, required=True, help="physical nvidia-smi index (telemetry)")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    name = args.run_name or {"m0": f"m0_s{args.seed}", "m1": "m1_n4", "m2": f"m2_n{args.num_envs}",
                             "l4": "l4_deterministic", "profile": f"profile_n{args.num_envs}"}[args.mode]
    run = new_run_dir(name)
    import torch  # noqa: F401
    import robotic_cutting_sim  # noqa: F401
    manifest(args, run)
    telemetry = telemetry_start(args.gpu_index)
    status, error, result = "FAIL", None, {}
    try:
        result = dict(m0=mode_m0, m1=mode_m1, m2=mode_m2, l4=mode_l4, profile=mode_profile)[args.mode](
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
    out = dict(status=status, error=error, mode=args.mode, telemetry=summary, wall_s=time.perf_counter() - T_SCRIPT,
               **result)
    dump(run / "result.json", out)
    print(json.dumps(dict(run=str(run.relative_to(ROOT / "artifacts")), status=status,
                          gpu_sharing=summary.get("gpu_sharing"), wall_s=out["wall_s"])))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
