#!/usr/bin/env python3
"""P1-5 T5.6 / T5.8: single_env_headless CUT rows (one frozen single material per process, no Kit).

Boundary ``single_env_headless`` (RL_READY_PLAN section 2): the existing single-env ``LiveCuttingMaterial`` (one
CUDA graph of 200 substeps per 50 Hz control step), without Kit, robot, cameras or recording. Modes:

* ``replay`` (T5.6 (i); T5.8 replay): a recorded Native CUT episode (WP-12 ``run_wp10_live`` format, default the
  frozen tlmpm_v5 WP-12 Demo 1) replayed from its measured trajectory (``rl.replay_wp12``), compared with every
  Native physical sample (positions, velocities, codes, labelled counts), the frozen goal evaluator applied to the
  replay samples and, with the same call, to the Native samples; per-step reactions vs Native.
* ``reference`` (T5.6 (ii)): the pre-declared ``cut_v7_2_v1`` workload: frozen single tlmpm_v7_2, ideal tracking
  of the ``reference_command_policy_v1`` commands (reconstructed commanded edge poses), handle box only, a host
  snapshot every 10 control steps (71 frames) and the frozen goal evaluator.

Both modes: cold start (process start -> first control step done), warm stepping (per-step wall of the material
calls: ``update_pad_obstacles`` (replay) + ``advance`` (graph launch + device sync) + read-only diagnostics),
snapshots and evaluation outside the timed calls, invariants, the ``sever_bond_progress_packed_v1`` shaping terms
on every snapshot (NumPy reference; diagnostic, never success), ``--warm-resets`` further resets (the previous
solver is released first, so one device HashGrid lives at a time; the census is recorded), GPU telemetry.

    CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock <gpu1.lock> \
        .deps/isaacsim-venv/bin/python scripts/rl_ready/run_cut_single.py --mode replay --gpu-index 1

``--max-steps`` shortens a run; such a run is labelled ``TRUNCATED`` and never used as a row. Output:
``artifacts/rl_ready/p1_5/<run-name>_NNN/`` (never overwritten).
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

T_SCRIPT = time.perf_counter()
WALL_SCRIPT = time.time()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.benchmark import percentile  # noqa: E402
from robotic_cutting_sim.rl.replay import MaterialReplay, build_material, compare_sample, digit_counts  # noqa: E402
from robotic_cutting_sim.rl.telemetry import GpuTelemetry  # noqa: E402

DEMO1 = ROOT / "artifacts" / "wp12" / "demos_001" / "DEMO1_A" / "run" / "episode_000"
OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_5"
SAMPLE_EVERY = 10
SOURCES = ("src/robotic_cutting_sim/live_material.py", "src/robotic_cutting_sim/tlmpm_v5.py",
           "src/robotic_cutting_sim/tlmpm_v7_2.py", "src/robotic_cutting_sim/tlmpm_v7_2_contacts.py",
           "src/robotic_cutting_sim/tlmpm_v7_2_frontier.py", "src/robotic_cutting_sim/material_diagnostics.py",
           "src/robotic_cutting_sim/finger_geometry.py", "src/robotic_cutting_sim/goals.py",
           "src/robotic_cutting_sim/scene_config.py", "src/robotic_cutting_sim/rl/replay.py",
           "src/robotic_cutting_sim/rl/replay_wp12.py", "src/robotic_cutting_sim/rl/reference_policy.py",
           "src/robotic_cutting_sim/rl/reward_sever.py", "src/robotic_cutting_sim/rl/cut_workload.py",
           "configs/rl_benchmark/cut_v7_2_v1.json", "configs/rl_benchmark/workloads_v2.json",
           "configs/rl_benchmark/scenes/bar_flat_board_tlmpm_v7_2_v1.json", "scripts/rl_ready/run_cut_single.py")


def process_age_s():
    ticks = os.sysconf("SC_CLK_TCK")
    start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    return float(Path("/proc/uptime").read_text().split()[0]) - start_ticks / ticks


AGE_AT_SCRIPT = process_age_s()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    raise RuntimeError("No free run directory")


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=1, sort_keys=True, allow_nan=False,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class WarpModuleLog(io.TextIOBase):
    """Tee stdout; keep Warp module load lines."""

    def __init__(self, stream):
        self.stream, self.lines, self._buffer = stream, [], ""

    def write(self, text):
        self.stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.startswith("Module ") and " took " in line:
                self.lines.append(line.strip())
        return len(text)

    def flush(self):
        self.stream.flush()

    def summary(self):
        total = 0.
        for line in self.lines:
            try:
                total += float(line.split(" took ")[1].split(" ms")[0])
            except (IndexError, ValueError):
                pass
        return dict(module_loads=len(self.lines), compiled=sum("(compiled)" in l for l in self.lines),
                    cached=sum("(cached)" in l for l in self.lines), reported_load_ms_total=total)


def hashgrid_census():
    from robotic_cutting_sim.batched.material_batch import live_device_hashgrids
    return len(live_device_hashgrids())


def release_solver(material):
    """Drop the graph and solver (and its device HashGrid) before a warm reset rebuilds them."""
    material.graph = None
    material.bed = None
    material._numerical_reduction = None
    gc.collect()


def code_rewrites(previous, current, max_cuts):
    previous, current = np.asarray(previous, np.int64), np.asarray(current, np.int64)
    out = 0
    for k in range(max_cuts):
        a, b = (previous >> (2 * k)) & 3, (current >> (2 * k)) & 3
        out += int(np.count_nonzero((a != 0) & (b != a)))
    return out


def summarize_evaluation(result):
    outcome = result.get("outcome", {})
    metrics = outcome.get("metrics", {})
    quality = result.get("quality", {})
    location = quality.get("plane_location") or []
    return dict(verdict=result.get("verdict"), integrity=result.get("integrity", {}).get("status"),
                integrity_checks=result.get("integrity", {}).get("checks"), outcome=outcome.get("status"),
                outcome_checks=outcome.get("checks"), mechanical_components=metrics.get("mechanical_components"),
                piece_mass_fractions=metrics.get("piece_mass_fractions"),
                substantial_pieces=metrics.get("substantial_pieces"),
                main_pieces_mass_fraction=metrics.get("main_pieces_mass_fraction"),
                minor_pieces_mass_fraction=metrics.get("minor_pieces_mass_fraction"),
                frozen_2mm_statistic_m=[row.get("frozen_2mm_statistic_m") for row in location if "plane" in row],
                frames=result.get("integrity", {}).get("metrics", {}).get("frames"))


def evaluate(task, reference, times, positions, velocities, codes):
    data = dict(reference, time_s=np.asarray(times, float), positions=np.stack(positions),
                velocities=np.stack(velocities), codes=np.stack(codes).astype(np.int32), code_encoding="packed2",
                max_cuts=int(task.max_cuts), particle_spacing_m=task.particle_spacing_m)
    start = time.perf_counter()
    result = task.evaluate_goal(data)
    return dict(summary=summarize_evaluation(result), result=result, wall_s=time.perf_counter() - start)


def evaluation_reference(material):
    return dict(mass_kg=material.masses.copy(), reference_positions=np.asarray(material.world_points).copy(),
                reference_positions_local=np.asarray(material.points).copy(),
                reference_indices=np.asarray(material.lattice).copy(),
                reference_stencil_nodes=material.bed.nodes.numpy(), reference_stencil_weights=material.bed.weights.numpy(),
                reference_stencil_gradients=material.bed.gradients.numpy())


def setup_warp(args):
    import warp as wp
    tee = WarpModuleLog(sys.stdout)
    sys.stdout = tee
    t0 = time.perf_counter()
    wp.init()
    init_s = time.perf_counter() - t0
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise SystemExit("single_env_headless rows run the CUDA graph path; a CUDA device is required")
    return wp, device, tee, init_s


class Episode:
    """Common stepping, snapshot, invariant and timing bookkeeping of both modes."""

    def __init__(self, material, task, steps, dt, max_cuts, bonds_packed):
        self.material, self.task, self.steps, self.dt, self.max_cuts = material, task, steps, dt, max_cuts
        self.packed = bonds_packed
        self.step_wall = np.zeros(steps)
        self.diag = np.zeros((steps, 6))
        self.reaction = np.zeros((steps, 6))
        self.fingers = None
        self.kept = dict(steps=[], times=[], positions=[], velocities=[], codes=[])
        self.shaping = []
        self.rewrites, self.finite = 0, True
        self.previous_codes = None
        self.capture_s = 0.
        self.masses = []

    def capture(self, step, keep=True):
        from robotic_cutting_sim.rl.reward_sever import severed_counts_numpy
        start = time.perf_counter()
        state = self.material.history_state()
        codes = np.asarray(state["codes"]).astype(np.int32)
        if self.previous_codes is not None:
            self.rewrites += code_rewrites(self.previous_codes, codes, self.max_cuts)
        self.previous_codes = codes.copy()
        finite = bool(np.isfinite(state["positions"]).all() and np.isfinite(state["velocities"]).all())
        self.finite = self.finite and finite
        target, off = severed_counts_numpy(codes, self.packed)
        n_target = int(self.packed.target_count[0])
        self.shaping.append(dict(step=step, severed_target_bonds=int(target[0]), severed_off_target_bonds=int(off[0]),
                                 target_bond_progress_fraction=float(target[0]) / n_target,
                                 off_target_bond_fraction=float(off[0]) / n_target))
        self.masses.append(float(self.material.masses.sum()))
        if keep:
            self.kept["steps"].append(step)
            self.kept["times"].append(step * self.dt)
            self.kept["positions"].append(np.asarray(state["positions"], float).copy())
            self.kept["velocities"].append(np.asarray(state["velocities"], float).copy())
            self.kept["codes"].append(codes)
        self.capture_s += time.perf_counter() - start
        return state, codes

    def record(self, t, d, wall):
        self.step_wall[t] = wall
        self.diag[t] = (d["minimum_detF"], d["maximum_material_speed_m_s"], d["maximum_plastic_strain"],
                        d["cut_index"], float(np.linalg.norm(d["current_total_material_reaction_impulse_world"][:3])),
                        d["material_wall_s"])
        self.reaction[t] = d["current_total_material_reaction_impulse_world"]
        self.finite = self.finite and bool(np.isfinite(self.diag[t]).all())

    def invariants(self):
        out = dict(finite=self.finite, minimum_detF=float(self.diag[:, 0].min()),
                   minimum_detF_positive=bool(self.diag[:, 0].min() > 0),
                   maximum_material_speed_m_s=float(self.diag[:, 1].max()),
                   maximum_plastic_strain=float(self.diag[:, 2].max()),
                   mass_constant=len(set(self.masses)) == 1, nonzero_digit_rewrites=self.rewrites,
                   shaping_progress_monotone=all(b["severed_target_bonds"] >= a["severed_target_bonds"]
                                                 for a, b in zip(self.shaping, self.shaping[1:])))
        out["status"] = "PASS" if (out["finite"] and out["minimum_detF_positive"] and out["mass_constant"]
                                   and self.rewrites == 0) else "FAIL"
        return out

    def timing(self, first_step_age, loop_s):
        warm = self.step_wall[1:] if self.steps > 1 else self.step_wall
        return dict(first_step_s=float(self.step_wall[0]), cold_process_start_to_first_step_done_s=first_step_age,
                    warm_steps=int(len(warm)), warm_step_mean_s=float(warm.mean()),
                    warm_step_median_s=float(np.median(warm)), warm_step_p95_s=float(np.percentile(warm, 95)),
                    warm_step_max_s=float(warm.max()), warm_env_steps_per_s=float(len(warm) / warm.sum()),
                    warm_sim_time_per_wall=float(len(warm) * self.dt / warm.sum()),
                    all_steps_wall_s=float(self.step_wall.sum()), loop_wall_including_samples_s=loop_s,
                    sample_capture_s=self.capture_s,
                    material_graph_wall_mean_s=float(self.diag[1:, 5].mean()) if self.steps > 1 else None)

    def save(self, run):
        np.savez_compressed(run / "steps.npz", step_wall_s=self.step_wall.astype(np.float32),
                            diagnostics=self.diag.astype(np.float32), reaction_impulse=self.reaction.astype(np.float32),
                            diagnostics_columns=np.asarray(["minimum_detF", "maximum_material_speed_m_s",
                                                            "maximum_plastic_strain", "cut_index",
                                                            "reaction_impulse_norm_n_s", "material_wall_s"]))
        np.savez_compressed(run / "codes_series.npz", steps=np.asarray(self.kept["steps"], np.int32),
                            codes=np.stack(self.kept["codes"]).astype(np.int32))
        final = self.kept["positions"][-1] if self.kept["positions"] else None
        np.savez_compressed(run / "final_state.npz", codes=self.kept["codes"][-1].astype(np.int32),
                            positions=np.asarray(final, np.float32), step=np.int64(self.kept["steps"][-1]))


def warm_resets(args, material, reset_fn):
    out, census = [], []
    for _ in range(args.warm_resets):
        release_solver(material)
        census.append(hashgrid_census())
        seconds = reset_fn()
        census.append(hashgrid_census())
        out.append(seconds)
    return out, census


def bonds_for(task, material):
    from robotic_cutting_sim.rl.reward_packed import pack_bonds
    from robotic_cutting_sim.rl.reward_sever import bonds_for_task
    version, bonds = bonds_for_task(task, particles=(np.asarray(material.points), np.asarray(material.masses),
                                                     np.asarray(material.lattice)))
    return version, bonds.receipt, pack_bonds([bonds])


def material_receipt(material, info):
    receipt = info.get("parameter_receipt", {})
    return dict(solver=material.solver, cutting_law=material.cutting_law, max_cuts=material.max_cuts,
                slots_per_node=material.slots_per_node, v7_2_optimizations=material.v7_2_optimizations,
                substeps=material.substeps, spacing_m=material.spacing, seed=material.seed,
                obstacle_boxes=int(np.shape(material.obstacle_boxes_edge)[0]),
                solver_receipt=receipt.get(material.solver), solver_module=receipt.get("solver_module"))


# ----------------------------------------------------------------------------- replay mode

def run_replay(args, run, telemetry):
    from robotic_cutting_sim.rl.replay_wp12 import (extract_wp12_trajectory, native_reactions, native_samples,
                                                    read_action_rows, source_equivalence_audit)
    t0 = time.perf_counter()
    trajectory = extract_wp12_trajectory(args.recording, locator_root=ROOT / "artifacts")
    extract_s = time.perf_counter() - t0
    trajectory.save(run / "trajectory")
    audit = None
    if args.audit_commit:
        audit = source_equivalence_audit(args.audit_commit, ROOT)
    wp, device, tee, init_s = setup_warp(args)
    census = dict(start=hashgrid_census())
    t0 = time.perf_counter()
    material, task = build_material(trajectory, str(device))
    build_s = time.perf_counter() - t0
    replay = MaterialReplay(trajectory, material)
    reset_s, reset_info = replay.reset()
    census["after_first_reset"] = hashgrid_census()
    first_reset_modules = tee.summary()
    native, checkpoint = native_samples(args.recording)
    dt, max_cuts = trajectory.options["control_dt_s"], int(trajectory.options["max_cuts"])
    sample_steps = np.rint(np.asarray(native["time_s"]) / dt).astype(int)
    if not np.allclose(sample_steps * dt, native["time_s"], atol=1e-9):
        raise ValueError("Native sample times are not control boundaries")
    by_step = {int(s): i for i, s in enumerate(sample_steps)}
    ref = evaluation_reference(material)
    reference = dict(particles=int(len(material.masses)), replay_total_mass_kg=float(material.masses.sum()),
                     native_total_mass_kg=float(np.asarray(native["mass_kg"]).sum()),
                     evaluation_reference_arrays_equal_native={k: bool(np.array_equal(v, native[k]))
                                                               for k, v in ref.items() if k in native})
    version, bonds_receipt, packed = bonds_for(task, material)
    steps = trajectory.steps if args.max_steps is None else min(args.max_steps, trajectory.steps)
    ep = Episode(material, task, steps, dt, max_cuts, packed)
    comparisons = []

    def capture(step):
        state, codes = ep.capture(step)
        i = by_step[step]
        row = compare_sample({k: native[k][i] for k in ("positions", "velocities", "codes")}, state, max_cuts)
        row.update(sample=i, step=step, time_s=float(native["time_s"][i]))
        comparisons.append(row)

    capture(0)
    first_age = None
    loop_start = time.perf_counter()
    for t in range(steps):
        start = time.perf_counter()
        d = replay.step()
        ep.record(t, d, time.perf_counter() - start)
        if t == 0:
            first_age = process_age_s()
            telemetry.set_phase("run")
        if t + 1 in by_step:
            capture(t + 1)
    loop_s = time.perf_counter() - loop_start
    telemetry.set_phase("post")
    complete = steps == trajectory.steps
    ep.save(run)
    _, rows = read_action_rows(args.recording)
    recipients = [r for r in trajectory.pad_recipients if r is not None]
    native_total, _ = native_reactions(rows[:steps], recipients)
    n_contact, r_contact = np.any(native_total != 0, 1), np.any(ep.reaction != 0, 1)
    diff = np.abs(ep.reaction - native_total)
    scale = float(np.abs(native_total).max()) or 1.
    reactions = dict(native_contact_steps=int(n_contact.sum()), replay_contact_steps=int(r_contact.sum()),
                     contact_on_off_mismatch_steps=int(np.count_nonzero(n_contact != r_contact)),
                     total_impulse_max_abs_diff=float(diff.max()), total_impulse_max_abs_native=scale,
                     total_impulse_max_rel_diff_of_peak=float(diff.max() / scale),
                     scope="Per-step reaction from the same measured poses; Native fed it back into PhysX, replay does not")
    evaluation = None
    if complete:
        k = ep.kept
        replay_eval = evaluate(task, ref, k["times"], k["positions"], k["velocities"], k["codes"])
        native_eval = evaluate(task, {key: native[key] for key in ref}, native["time_s"], list(native["positions"]),
                               list(native["velocities"]), list(native["codes"]))
        evaluation = dict(replay=replay_eval, native_same_evaluator=native_eval,
                          evaluator="robotic_cutting_sim.goals.evaluate_goal via ConfiguredTaskSpec.evaluate_goal "
                                    "(frozen sever_mid goal), in-process after the replay, outside timing")
    final_native = np.asarray(native["codes"][len(native["time_s"]) - 1]) if complete else None
    final_codes = ep.kept["codes"][-1]
    parity = dict(
        samples_compared=len(comparisons),
        max_position_diff_m_over_time=max(c["max_position_diff_m"] for c in comparisons),
        max_mean_position_diff_m_over_time=max(c["mean_position_diff_m"] for c in comparisons),
        samples_with_code_difference=[dict(sample=c["sample"], step=c["step"], particles=c["different_code_particles"])
                                      for c in comparisons if c["different_code_particles"]],
        final=None if not complete else dict(
            codes_equal=bool(np.array_equal(final_codes, final_native)),
            different_code_particles=int(np.count_nonzero(final_codes != final_native)),
            labelled_native=digit_counts(final_native, max_cuts), labelled_replay=digit_counts(final_codes, max_cuts),
            labelled_count_diff=comparisons[-1]["labelled_count_diff"],
            max_position_diff_m=comparisons[-1]["max_position_diff_m"],
            mean_position_diff_m=comparisons[-1]["mean_position_diff_m"],
            native_final_codes_sha256=hashlib.sha256(np.ascontiguousarray(final_native)).hexdigest(),
            replay_final_codes_sha256=hashlib.sha256(np.ascontiguousarray(final_codes)).hexdigest()),
        reference=reference, native_checkpoint=checkpoint)
    native.close()
    k = ep.kept
    k.clear()
    gc.collect()
    mempool_after = int(wp.get_mempool_used_mem_high(device)) if wp.is_mempool_enabled(device) else None
    census["end_of_episode"] = hashgrid_census()
    resets, reset_census = warm_resets(args, material, lambda: replay.reset()[0])
    census["warm_resets"] = reset_census
    timing = dict(ep.timing(first_age, loop_s), process_age_at_script_entry_s=AGE_AT_SCRIPT, warp_init_s=init_s,
                  trajectory_extract_s=extract_s, task_compile_and_material_construct_s=build_s, first_reset_s=reset_s,
                  first_reset_wall_reported_by_material_s=reset_info["reset_wall_s"], warm_resets_s=resets,
                  warm_reset_mean_s=float(np.mean(resets)) if resets else None, warm_reset_p95_s=percentile(resets, 95),
                  mempool_used_high_bytes_after_episode=mempool_after,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
                  warp_modules_first_reset=first_reset_modules, warp_modules_end=tee.summary())
    return dict(steps_run=steps, complete=complete, timing=timing, invariants=ep.invariants(), parity=parity,
                reactions=reactions, evaluation=evaluation, shaping=dict(version=version, bonds=bonds_receipt,
                                                                        samples=ep.shaping),
                comparisons=comparisons, particles=reference["particles"], device=str(device),
                hashgrid_census=census, material_receipt=material_receipt(material, reset_info),
                source_equivalence_audit=audit, task_id=task.task_id,
                trajectory=dict(steps=trajectory.steps, checks=trajectory.checks, options=trajectory.options,
                                provenance=trajectory.provenance, source=trajectory.source))


# ----------------------------------------------------------------------------- reference mode

def run_reference(args, run, telemetry):
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.rl.cut_workload import compile_workload_task, load_workload, receipt
    from robotic_cutting_sim.rl.reference_policy import reference_command_actions
    workload = load_workload()
    t0 = time.perf_counter()
    task = compile_workload_task(workload)
    if task.task_id != workload["compiled"]["task_id"]:
        raise ValueError("compiled task id differs from the declaration")
    actions, poses, schedule, policy_report = reference_command_actions(task)
    compile_s = time.perf_counter() - t0
    np.savez_compressed(run / "reference_commands.npz", actions=actions, commanded_pose=poses,
                        resolved_cut=schedule["resolved_cut"])
    m = workload["material"]
    wp, device, tee, init_s = setup_warp(args)
    census = dict(start=hashgrid_census())
    dt = float(task.control_dt_s)
    t0 = time.perf_counter()
    config = CouplingConfig(control_dt=dt, material_dt=float(m["material_dt_s"]), dx=float(task.grid_dx_m),
                            device=str(device), feedback=True)
    material = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                   solver=task.solver_module, v7_2_optimizations=bool(m["v7_2_optimizations"]))
    build_s = time.perf_counter() - t0
    for key, expected in (("solver", m["solver"]), ("max_cuts", m["max_cuts"]), ("slots_per_node", m["slots_per_node"]),
                          ("cutting_law", m["cutting_law"])):
        if getattr(material, key) != expected:
            raise ValueError(f"material {key} {getattr(material, key)} differs from the declaration {expected}")
    initial = schedule["initial_pose"]

    def reset():
        start = time.perf_counter()
        info = material.reset(initial, int(workload["episode"]["sim_seed"]))
        return time.perf_counter() - start, info
    reset_s, reset_info = reset()
    census["after_first_reset"] = hashgrid_census()
    first_reset_modules = tee.summary()
    if material.substeps != m["substeps_per_control_step"] or np.shape(material.obstacle_boxes_edge)[0] != 1:
        raise ValueError("material substeps / obstacles differ from the declaration (handle_only_v1)")
    ref = evaluation_reference(material)
    version, bonds_receipt, packed = bonds_for(task, material)
    steps = schedule["steps"] if args.max_steps is None else min(args.max_steps, schedule["steps"])
    ep = Episode(material, task, steps, dt, int(task.max_cuts), packed)
    ep.capture(0)
    first_age = None
    resolved = schedule["resolved_cut"]
    loop_start = time.perf_counter()
    for t in range(steps):
        start = time.perf_counter()
        d = material.advance(poses[t], (t + 1) * dt, cutting=bool(resolved[t] >= 0), cut_index=int(max(resolved[t], 0)))
        d.update(material.numerical_diagnostics(args.material_diagnostics))
        wall = time.perf_counter() - start
        if d["cut_index"] != int(resolved[t]):
            raise AssertionError("material resolved a different cut digit than the reference schedule")
        ep.record(t, d, wall)
        if t == 0:
            first_age = process_age_s()
            telemetry.set_phase("run")
        if (t + 1) % SAMPLE_EVERY == 0 or t + 1 == steps:
            ep.capture(t + 1)
    loop_s = time.perf_counter() - loop_start
    telemetry.set_phase("post")
    complete = steps == schedule["steps"]
    ep.save(run)
    evaluation = None
    if complete:
        k = ep.kept
        evaluation = dict(reference_run=evaluate(task, ref, k["times"], k["positions"], k["velocities"], k["codes"]),
                          evaluator="robotic_cutting_sim.goals.evaluate_goal via ConfiguredTaskSpec.evaluate_goal "
                                    "(frozen sever_mid goal), in-process after the episode, outside timing")
    final_codes = ep.kept["codes"][-1]
    final = dict(step=int(ep.kept["steps"][-1]), labelled=digit_counts(final_codes, int(task.max_cuts)),
                 codes_sha256=hashlib.sha256(np.ascontiguousarray(final_codes)).hexdigest(),
                 labelled_particles=int(np.count_nonzero(final_codes)))
    ep.kept.clear()
    gc.collect()
    mempool_after = int(wp.get_mempool_used_mem_high(device)) if wp.is_mempool_enabled(device) else None
    census["end_of_episode"] = hashgrid_census()
    resets, reset_census = warm_resets(args, material, lambda: reset()[0])
    census["warm_resets"] = reset_census
    reaction_steps = int(np.count_nonzero(np.any(ep.reaction != 0, 1)))
    timing = dict(ep.timing(first_age, loop_s), process_age_at_script_entry_s=AGE_AT_SCRIPT, warp_init_s=init_s,
                  task_compile_and_commands_s=compile_s, material_construct_s=build_s, first_reset_s=reset_s,
                  first_reset_wall_reported_by_material_s=reset_info["reset_wall_s"], warm_resets_s=resets,
                  warm_reset_mean_s=float(np.mean(resets)) if resets else None, warm_reset_p95_s=percentile(resets, 95),
                  mempool_used_high_bytes_after_episode=mempool_after,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
                  warp_modules_first_reset=first_reset_modules, warp_modules_end=tee.summary())
    return dict(steps_run=steps, complete=complete, timing=timing, invariants=ep.invariants(), final=final,
                evaluation=evaluation, shaping=dict(version=version, bonds=bonds_receipt, samples=ep.shaping),
                reaction_steps=reaction_steps, particles=int(len(material.masses)), device=str(device),
                hashgrid_census=census, material_receipt=material_receipt(material, reset_info),
                workload=receipt(workload), policy=policy_report, task_id=task.task_id,
                material_diagnostics=args.material_diagnostics,
                drive=dict(kind="ideal_tracking_reference_commands", obstacles=m["obstacles"],
                           initial_edge_pose=initial.tolist(), sample_every_steps=SAMPLE_EVERY))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("replay", "reference"), required=True)
    parser.add_argument("--recording", type=Path, default=DEMO1, help="WP-12-format episode_000 directory (replay)")
    parser.add_argument("--audit-commit", default="a960dc6c1aea1060234a94ea7d4a310370484773",
                        help="Recording commit for the source-equivalence audit (replay; empty to skip)")
    parser.add_argument("--device", default="cuda:0", help="CUDA ordinal inside the process (after masking)")
    parser.add_argument("--gpu-index", type=int, required=True, help="Physical nvidia-smi index for telemetry")
    parser.add_argument("--run-name")
    parser.add_argument("--warm-resets", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--material-diagnostics", default="device_reduction_v1",
                        choices=("snapshot", "device_reduction_v1"), help="reference mode (replay uses the recording's)")
    parser.add_argument("--baseline-s", type=float, default=3.)
    args = parser.parse_args()
    run = new_run_dir(args.run_name or ("cut_v5_replay" if args.mode == "replay" else "cut_v7_2_reference"))
    manifest = dict(schema="rcs_rl_p1_5_cut_single_run_v1", argv=sys.argv,
                    args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    boundary="single_env_headless", started_unix_s=WALL_SCRIPT, git_head=git("rev-parse", "HEAD"),
                    git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), python=sys.version.split()[0],
                    interpreter=os.path.relpath(sys.executable, ROOT), kit_started=False,
                    package_origin=os.path.relpath(sys.modules["robotic_cutting_sim"].__file__, ROOT),
                    source_sha256={path: sha256(ROOT / path) for path in SOURCES}, pid=os.getpid())
    import warp
    import numpy
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    manifest["versions"] = dict(warp=warp.__version__, numpy=numpy.__version__, torch=torch_version)
    write_json(run / "manifest.json", manifest)
    telemetry = GpuTelemetry(args.gpu_index, interval_s=1.)
    telemetry.set_phase("baseline")
    telemetry.start()
    time.sleep(args.baseline_s)
    telemetry.set_phase("setup")
    status, error, result = "FAIL", None, {}
    try:
        result = (run_replay if args.mode == "replay" else run_reference)(args, run, telemetry)
        status = "COMPLETE" if result.get("complete") else "TRUNCATED"
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
               telemetry_run_phase=telemetry.summary(phases=("baseline", "run")),
               wall_s=time.perf_counter() - T_SCRIPT, **result)
    write_json(run / "result.json", out)
    verdict = None
    if result.get("evaluation"):
        key = "replay" if args.mode == "replay" else "reference_run"
        verdict = result["evaluation"][key]["summary"]["verdict"]
    print(json.dumps(dict(run=str(run.relative_to(ROOT / "artifacts")), status=status, verdict=verdict,
                          gpu_sharing=summary.get("gpu_sharing"), wall_s=out["wall_s"])), file=sys.__stdout__)
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
