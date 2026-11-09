#!/usr/bin/env python3
"""P1-0 single_env_headless baseline: replay a complete WP-17 Native episode through the material only.

Boundary ``single_env_headless`` (RL_READY_PLAN section 2): the existing single-env ``LiveCuttingMaterial``
(one CUDA graph of 200 substeps per 50 Hz control step) driven by the recorded measured trajectory
(``robotic_cutting_sim.rl.replay``), without Kit, robot, cameras or recording. Run it with the Isaac Sim
venv interpreter and ``PYTHONPATH=src`` WITHOUT starting Kit (decision D-08), one job per GPU under the
RCS flock, e.g.::

    CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock <lock> \
        .deps/isaacsim-venv/bin/python scripts/rl_ready/run_baseline.py --mode replay --seed 512 --gpu-index 1

Modes:

* ``replay``: cold start (process start -> first control step done; Warp module load/compile and graph
  capture inside the first reset), warm stepping of all steps (per-step wall of ``update_pad_obstacles`` +
  ``advance`` (graph launch + device sync) + ``numerical_diagnostics``), comparison with the 121 Native
  samples (outside the timed step calls), material invariants, then ``--warm-resets`` further resets in
  the same process.
* ``kernels``: separate, perturbing run. Graph replay up to each representative step, then K eager steps
  under ``profiling.StageProfiler(kernels=True)`` (CUDA activity timing of every kernel/memset), graph
  restored afterwards. Reports GPU time per kernel and per kernel group per substep.

``--kernel-cache fresh`` uses a new empty Warp kernel cache (true compile cold start); the default uses
the user's existing cache. ``--max-steps`` shortens a run; such a run is labelled ``truncated`` and never
used as a full-episode row.
"""
import argparse
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

T_SCRIPT = time.perf_counter()
WALL_SCRIPT = time.time()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim.rl.benchmark import percentile  # noqa: E402
from robotic_cutting_sim.rl.replay import (MaterialReplay, build_material, compare_sample, digit_counts,  # noqa: E402
                                           extract_trajectory, native_samples)
from robotic_cutting_sim.rl.telemetry import GpuTelemetry  # noqa: E402

WP17 = ROOT / "artifacts" / "wp17" / "native_optimized_001"
OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_0"
SOURCES = ("src/robotic_cutting_sim/live_material.py", "src/robotic_cutting_sim/tlmpm_v7_2.py",
           "src/robotic_cutting_sim/tlmpm_v7_2_contacts.py", "src/robotic_cutting_sim/tlmpm_v7_2_frontier.py",
           "src/robotic_cutting_sim/material_diagnostics.py", "src/robotic_cutting_sim/finger_geometry.py",
           "src/robotic_cutting_sim/profiling.py", "src/robotic_cutting_sim/rl/replay.py",
           "src/robotic_cutting_sim/goals.py", "src/robotic_cutting_sim/seeded_trim.py",
           "scripts/rl_ready/run_baseline.py")


def process_age_s():
    """Seconds since this process was created (Linux /proc), i.e. including interpreter start-up."""
    ticks = os.sysconf("SC_CLK_TCK")
    start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    return uptime - start_ticks / ticks


AGE_AT_SCRIPT = process_age_s()


def sha256(path):
    import hashlib
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
        json.dump(data, stream, indent=1, sort_keys=True, allow_nan=False, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else str(o))


class WarpModuleLog(io.TextIOBase):
    """Tee stdout; keep Warp module load lines ('Module ... load on device ... took ... ms (compiled|cached)')."""

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
        compiled = [l for l in self.lines if "(compiled)" in l]
        cached = [l for l in self.lines if "(cached)" in l]
        total_ms = 0.
        for line in self.lines:
            try:
                total_ms += float(line.split(" took ")[1].split(" ms")[0])
            except (IndexError, ValueError):
                pass
        return dict(module_loads=len(self.lines), compiled=len(compiled), cached=len(cached),
                    reported_load_ms_total=total_ms, lines=self.lines)


def native_reaction_steps(attempt):
    """Steps (1-based) at which the Native material returned a nonzero reaction impulse (contact)."""
    steps = []
    with (Path(attempt) / "actions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            impulse = np.asarray(row["privileged_diagnostics"]["current_total_material_reaction_impulse_world"])
            if np.any(impulse != 0):
                steps.append(int(row["step"]))
    return steps


def native_reactions(attempt, recipients):
    """Recorded Native per-step total material reaction impulse [T,6] and finger wrenches [T,2,6]."""
    total, fingers = [], []
    with (Path(attempt) / "actions.jsonl").open() as stream:
        for line in stream:
            diag = json.loads(line)["privileged_diagnostics"]
            total.append(diag["current_total_material_reaction_impulse_world"])
            fingers.append([diag["next_finger_wrenches_about_edge_world"].get(r, [0.] * 6) for r in recipients])
    return np.asarray(total, float), np.asarray(fingers, float)


def reaction_comparison(native_total, native_fingers, replay_total, replay_fingers):
    n_contact, r_contact = np.any(native_total != 0, 1), np.any(replay_total != 0, 1)
    n_finger, r_finger = np.any(native_fingers != 0, (1, 2)), np.any(replay_fingers != 0, (1, 2))
    diff = np.abs(replay_total - native_total)
    scale = np.abs(native_total).max() if np.abs(native_total).max() > 0 else 1.
    return dict(steps=len(native_total), native_contact_steps=int(n_contact.sum()),
                replay_contact_steps=int(r_contact.sum()),
                contact_on_off_mismatch_steps=int(np.count_nonzero(n_contact != r_contact)),
                total_impulse_max_abs_diff=float(diff.max()), total_impulse_max_abs_native=float(scale),
                total_impulse_max_rel_diff_of_peak=float(diff.max() / scale),
                native_finger_wrench_steps=int(n_finger.sum()), replay_finger_wrench_steps=int(r_finger.sum()),
                finger_wrench_max_abs_diff=float(np.abs(replay_fingers - native_fingers).max()),
                scope="Per-step reaction from the same measured poses; Native fed it back into PhysX, replay does not")


def evaluate_states(task, reference, times, positions, velocities, codes):
    """Frozen seeded_trim goal evaluation (task.evaluate_goal, as seeded_trim.evaluate_recording) of states."""
    data = dict(reference, time_s=np.asarray(times, float), positions=np.stack(positions),
                velocities=np.stack(velocities), codes=np.stack(codes))
    data.update(code_encoding="packed2", max_cuts=2, particle_spacing_m=task.particle_spacing_m)
    return summarize_evaluation(task.evaluate_goal(data))


def summarize_evaluation(result):
    outcome = result.get("outcome", {})
    metrics = outcome.get("metrics", {})
    items = metrics.get("items", [])
    return dict(verdict=result.get("verdict"), integrity=result.get("integrity", {}).get("status"),
                integrity_checks=result.get("integrity", {}).get("checks"), outcome=outcome.get("status"),
                outcome_checks=outcome.get("checks"), mechanical_components=metrics.get("mechanical_components"),
                largest_component_below_plane_mass_share=metrics.get("largest_component_below_plane_mass_share"),
                detached_below_plane_mass_fraction=metrics.get("detached_below_plane_mass_fraction"),
                removed_mass_fraction={i.get("id"): i.get("removed_mass_fraction") for i in items},
                residual_substrate_height_m={i.get("id"): i.get("residual_substrate_height_m") for i in items})


def code_rewrites(previous, current, max_cuts):
    """Particles whose already-assigned packed digit changed (must be 0: labels are persistent)."""
    previous, current = np.asarray(previous, np.int64), np.asarray(current, np.int64)
    out = 0
    for k in range(max_cuts):
        a, b = (previous >> (2 * k)) & 3, (current >> (2 * k)) & 3
        out += int(np.count_nonzero((a != 0) & (b != a)))
    return out


def setup(args, run):
    import warp as wp
    if args.kernel_cache == "fresh":
        cache = Path(tempfile.mkdtemp(prefix="rcs_p1_0_warp_cache_", dir=os.environ.get("TMPDIR")))
        wp.config.kernel_cache_dir = str(cache)
    tee = WarpModuleLog(sys.stdout)
    sys.stdout = tee
    t0 = time.perf_counter()
    wp.init()
    init_s = time.perf_counter() - t0
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise SystemExit("The baseline runs the CUDA graph path; a CUDA device is required")
    return wp, device, tee, init_s


def run_replay(args, run, telemetry, trajectory, attempt):
    wp, device, tee, init_s = setup(args, run)
    t0 = time.perf_counter()
    material, task = build_material(trajectory, str(device))
    build_s = time.perf_counter() - t0
    replay = MaterialReplay(trajectory, material)
    reset_s, reset_info = replay.reset()
    first_reset_module_log = tee.summary()
    native, checkpoint = native_samples(attempt)
    max_cuts = int(trajectory.options["max_cuts"])
    sample_steps = np.rint(np.asarray(native["time_s"]) / trajectory.options["control_dt_s"]).astype(int)
    if not np.allclose(sample_steps * trajectory.options["control_dt_s"], native["time_s"], atol=1e-9):
        raise ValueError("Native sample times are not control boundaries")
    reference = dict(masses_equal=bool(np.array_equal(material.masses, native["mass_kg"])),
                     reference_positions_equal=bool(np.array_equal(material.world_points,
                                                                   native["reference_positions"])),
                     replay_total_mass_kg=float(material.masses.sum()),
                     native_total_mass_kg=float(np.asarray(native["mass_kg"]).sum()),
                     particles=int(len(material.masses)))
    by_step = {int(step): index for index, step in enumerate(sample_steps)}
    eval_reference = dict(mass_kg=material.masses.copy(), reference_positions=material.world_points,
                          reference_positions_local=material.points.copy(), reference_indices=material.lattice.copy(),
                          reference_stencil_nodes=material.bed.nodes.numpy(),
                          reference_stencil_weights=material.bed.weights.numpy(),
                          reference_stencil_gradients=material.bed.gradients.numpy())
    reference["evaluation_reference_arrays_equal_native"] = {
        key: bool(np.array_equal(value, native[key])) for key, value in eval_reference.items() if key in native}
    kept = dict(times=[], positions=[], velocities=[], codes=[])
    comparisons, timings = [], dict(sample_capture_s=0.)
    previous_codes = None
    rewrites, finite = 0, True

    def capture(step):
        nonlocal previous_codes, rewrites
        start = time.perf_counter()
        state = material.history_state()
        index = by_step[step]
        row = compare_sample({key: native[key][index] for key in ("positions", "velocities", "codes")}, state, max_cuts)
        row.update(sample=index, step=step, time_s=float(native["time_s"][index]),
                   replay_total_mass_kg=float(material.masses.sum()),
                   replay_finite=bool(np.isfinite(state["positions"]).all() and np.isfinite(state["velocities"]).all()))
        if previous_codes is not None:
            rewrites += code_rewrites(previous_codes, state["codes"], max_cuts)
        previous_codes = np.asarray(state["codes"]).copy()
        comparisons.append(row)
        if args.evaluate:
            kept["times"].append(row["time_s"])
            for key in ("positions", "velocities", "codes"):
                kept[key].append(np.asarray(state[key]).copy())
        timings["sample_capture_s"] += time.perf_counter() - start
        return state

    capture(0)
    steps = trajectory.steps if args.max_steps is None else min(args.max_steps, trajectory.steps)
    step_wall = np.zeros(steps)
    diag_rows = np.zeros((steps, 6))
    replay_total, replay_fingers = np.zeros((steps, 6)), np.zeros((steps, 2, 6))
    recipients = [r for r in trajectory.pad_recipients if r is not None]
    first_step_age = None
    loop_start = time.perf_counter()
    state = None
    for t in range(steps):
        start = time.perf_counter()
        d = replay.step()
        step_wall[t] = time.perf_counter() - start
        if t == 0:
            first_step_age = process_age_s()
            telemetry.set_phase("run")
        diag_rows[t] = (d["minimum_detF"], d["maximum_material_speed_m_s"], d["maximum_plastic_strain"],
                        d["cut_index"], float(np.linalg.norm(d["current_total_material_reaction_impulse_world"][:3])),
                        d["material_wall_s"])
        replay_total[t] = d["current_total_material_reaction_impulse_world"]
        replay_fingers[t] = [d["next_finger_wrenches_about_edge_world"].get(r, np.zeros(6)) for r in recipients]
        finite = finite and bool(np.isfinite(diag_rows[t]).all())
        if t + 1 in by_step:
            state = capture(t + 1)
    loop_s = time.perf_counter() - loop_start
    final_codes = np.asarray(material.history_state()["codes"]) if state is None else np.asarray(state["codes"])
    np.savez_compressed(run / "final_state.npz", codes=final_codes.astype(np.int32),
                        positions=np.asarray(material.history_state()["positions"], np.float32))
    np.savez_compressed(run / "steps.npz", step_wall_s=step_wall.astype(np.float32),
                        diagnostics=diag_rows.astype(np.float32),
                        diagnostics_columns=np.asarray(["minimum_detF", "maximum_material_speed_m_s",
                                                        "maximum_plastic_strain", "cut_index",
                                                        "reaction_impulse_norm_n_s", "material_wall_s"]))
    native_total, native_fingers = native_reactions(attempt, recipients)
    reactions = reaction_comparison(native_total[:steps], native_fingers[:steps], replay_total, replay_fingers)
    np.savez_compressed(run / "reactions.npz", replay_total_impulse=replay_total.astype(np.float32),
                        replay_finger_wrenches=replay_fingers.astype(np.float32))
    evaluation = None
    if args.evaluate and steps == trajectory.steps:
        start = time.perf_counter()
        replay_eval = evaluate_states(task, eval_reference, kept["times"], kept["positions"], kept["velocities"],
                                      kept["codes"])
        native_eval = evaluate_states(task, {key: native[key] for key in eval_reference}, native["time_s"],
                                      list(native["positions"]), list(native["velocities"]), list(native["codes"]))
        kept.clear()
        evaluation = dict(replay=replay_eval, native_same_evaluator=native_eval, wall_s=time.perf_counter() - start,
                          evaluator="robotic_cutting_sim.goals.evaluate_goal via ConfiguredTaskSpec.evaluate_goal "
                                    "(same inputs as seeded_trim.evaluate_recording, without surface_state, which "
                                    "feeds quality only); in-process after the replay")
    mempool_after_episode = int(wp.get_mempool_used_mem_high(device)) if wp.is_mempool_enabled(device) else None
    # warm resets in the same process (new solver + graph capture each time; old state released)
    warm_resets = []
    for _ in range(args.warm_resets):
        seconds, _ = replay.reset()
        warm_resets.append(seconds)
    warm_reset_module_log = tee.summary()
    complete = steps == trajectory.steps
    final_native = native["codes"][len(native["time_s"]) - 1] if complete else None
    warm = step_wall[1:] if steps > 1 else step_wall
    dt = trajectory.options["control_dt_s"]
    invariants = dict(
        finite_diagnostics=finite, all_samples_finite=all(c["replay_finite"] for c in comparisons),
        minimum_detF=float(diag_rows[:, 0].min()), minimum_detF_positive=bool(diag_rows[:, 0].min() > 0),
        maximum_material_speed_m_s=float(diag_rows[:, 1].max()),
        maximum_plastic_strain=float(diag_rows[:, 2].max()),
        mass_constant=all(c["replay_total_mass_kg"] == comparisons[0]["replay_total_mass_kg"] for c in comparisons),
        nonzero_digit_rewrites=rewrites, resolved_cut_matches_recording=True)
    invariants["status"] = "PASS" if (invariants["finite_diagnostics"] and invariants["all_samples_finite"]
                                      and invariants["minimum_detF_positive"] and invariants["mass_constant"]
                                      and rewrites == 0) else "FAIL"
    parity = dict(
        samples_compared=len(comparisons),
        max_position_diff_m_over_time=max(c["max_position_diff_m"] for c in comparisons),
        max_mean_position_diff_m_over_time=max(c["mean_position_diff_m"] for c in comparisons),
        first_sample_with_code_difference=next((dict(sample=c["sample"], step=c["step"], time_s=c["time_s"],
                                                     particles=c["different_code_particles"])
                                                for c in comparisons if c["different_code_particles"]), None),
        first_sample_with_position_difference=next((dict(sample=c["sample"], step=c["step"], time_s=c["time_s"],
                                                         max_m=c["max_position_diff_m"])
                                                    for c in comparisons if c["max_position_diff_m"] > 0), None),
        final=None if not complete else dict(
            codes_equal=bool(np.array_equal(final_codes, final_native)),
            different_code_particles=int(np.count_nonzero(final_codes != final_native)),
            labelled_native=digit_counts(final_native, max_cuts), labelled_replay=digit_counts(final_codes, max_cuts),
            labelled_count_diff=comparisons[-1]["labelled_count_diff"],
            max_position_diff_m=comparisons[-1]["max_position_diff_m"],
            mean_position_diff_m=comparisons[-1]["mean_position_diff_m"],
            native_final_codes_sha256=__import__("hashlib").sha256(np.ascontiguousarray(final_native)).hexdigest(),
            replay_final_codes_sha256=__import__("hashlib").sha256(np.ascontiguousarray(final_codes)).hexdigest()),
        reference=reference, native_checkpoint=dict(sample_count=checkpoint["sample_count"],
                                                    complete=checkpoint["complete"]))
    timing = dict(
        process_age_at_script_entry_s=AGE_AT_SCRIPT, script_entry_to_warp_init_s=None,
        warp_init_s=init_s, task_compile_and_material_construct_s=build_s,
        first_reset_s=reset_s, first_reset_wall_reported_by_material_s=reset_info["reset_wall_s"],
        first_step_s=float(step_wall[0]), cold_process_start_to_first_step_done_s=first_step_age,
        warm_steps=int(len(warm)), warm_step_mean_s=float(warm.mean()), warm_step_median_s=float(np.median(warm)),
        warm_step_p95_s=float(np.percentile(warm, 95)), warm_step_max_s=float(warm.max()),
        warm_env_steps_per_s=float(len(warm) / warm.sum()),
        warm_sim_time_per_wall=float(len(warm) * dt / warm.sum()),
        all_steps_wall_s=float(step_wall.sum()), loop_wall_including_samples_s=loop_s,
        sample_capture_and_compare_s=timings["sample_capture_s"],
        material_graph_wall_mean_s=float(diag_rows[1:, 5].mean()) if steps > 1 else None,
        warm_resets_s=warm_resets, warm_reset_mean_s=float(np.mean(warm_resets)) if warm_resets else None,
        warm_reset_p95_s=percentile(warm_resets, 95), mempool_used_high_bytes_after_episode=mempool_after_episode,
        mempool_used_high_bytes_end=int(wp.get_mempool_used_mem_high(device)) if wp.is_mempool_enabled(device) else None,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
        warp_modules_first_reset=first_reset_module_log, warp_modules_through_warm_resets=dict(
            {k: v for k, v in warm_reset_module_log.items() if k != "lines"}),
        kernel_cache=args.kernel_cache)
    return dict(steps_run=steps, complete=complete, timing=timing, invariants=invariants, parity=parity,
                reactions=reactions, evaluation=evaluation, replay_obstacles=trajectory.options.get("replay_obstacles"),
                comparisons=comparisons, particles=reference["particles"], device=str(device),
                material_receipt=dict(solver=material.solver, cutting_law=material.cutting_law,
                                      max_cuts=material.max_cuts, slots_per_node=material.slots_per_node,
                                      v7_2_optimizations=material.v7_2_optimizations, substeps=material.substeps,
                                      spacing_m=material.spacing, seed=material.seed,
                                      solver_receipt=reset_info["parameter_receipt"].get("tlmpm_v7_2")))


def representative_steps(trajectory, attempt, explicit):
    if explicit:
        return {f"step_{s}": int(s) for s in explicit}
    contact = native_reaction_steps(attempt)
    first_cut, last_cut = trajectory.checks["first_cutting_step"], trajectory.checks["last_cutting_step"]
    first_contact = contact[0]
    return dict(no_contact=20, approach_contact_no_cut=int((first_contact + first_cut) // 2),
                active_cutting=int((first_cut + last_cut) // 2),
                post_cut=int(last_cut + (trajectory.steps - last_cut) // 2)), dict(
        first_native_contact_step=first_contact, first_cut_step=first_cut, last_cut_step=last_cut,
        native_contact_steps=len(contact))


def run_kernels(args, run, telemetry, trajectory, attempt):
    from robotic_cutting_sim.profiling import StageProfiler
    wp, device, tee, init_s = setup(args, run)
    material, _ = build_material(trajectory, str(device))
    replay = MaterialReplay(trajectory, material)
    reset_s, _ = replay.reset()
    telemetry.set_phase("run")
    chosen = representative_steps(trajectory, attempt, args.kernel_steps)
    targets, phase_info = (chosen, {}) if isinstance(chosen, dict) else chosen
    windows = sorted((step, name) for name, step in targets.items())
    substeps = material.substeps
    results, graph_walls = {}, []
    for step, name in windows:
        while replay.step_index < step - 1:   # graph replay up to the window
            start = time.perf_counter()
            replay.step()
            graph_walls.append((replay.step_index, time.perf_counter() - start))
        graph = material.graph
        profiler = StageProfiler(sync_device=str(device), kernels=True).install()
        eager = []
        try:
            material.graph = None   # identical kernel sequence, launched eagerly (profiling.py kernels mode)
            for _ in range(args.kernel_repeat):
                start = time.perf_counter()
                d = replay.step()
                eager.append(dict(step=replay.step_index, wall_s=time.perf_counter() - start,
                                  cut_index=d["cut_index"],
                                  reaction_impulse_norm=float(np.linalg.norm(
                                      d["current_total_material_reaction_impulse_world"][:3])),
                                  contact_statistics=np.asarray(d["material_contact_statistics"]).tolist()))
        finally:
            profiler.uninstall()
            material.graph = graph
        summary = profiler.summary()
        n_sub = args.kernel_repeat * substeps
        kernels = {key: dict(total_s=v["total_s"], calls=v["calls"], per_substep_us=v["total_s"] / n_sub * 1e6,
                             share=v["share_of_kernel_time"]) for key, v in summary["kernels"].items()}
        groups = {key: dict(total_s=v["total_s"], per_substep_us=v["total_s"] / n_sub * 1e6,
                            share=v["share_of_kernel_time"]) for key, v in summary["kernel_groups"].items()}
        results[name] = dict(first_step=step, steps=[e["step"] for e in eager], eager_steps=eager,
                             substeps_timed=n_sub, kernel_total_s=summary["kernel_total_s"],
                             kernel_gpu_time_per_substep_us=summary["kernel_total_s"] / n_sub * 1e6,
                             kernel_gpu_time_per_control_step_ms=summary["kernel_total_s"] / args.kernel_repeat * 1e3,
                             eager_step_wall_mean_s=float(np.mean([e["wall_s"] for e in eager])),
                             kernels=kernels, kernel_groups=groups)
        # graph wall of the neighbouring graph steps for the same phase (after the eager window)
        neighbours = []
        for _ in range(args.kernel_repeat):
            start = time.perf_counter()
            replay.step()
            neighbours.append(time.perf_counter() - start)
        results[name]["graph_step_wall_after_window_mean_s"] = float(np.mean(neighbours))
    return dict(device=str(device), reset_s=reset_s, warp_init_s=init_s, phase_info=phase_info,
                windows=results, kernel_repeat=args.kernel_repeat, substeps_per_step=substeps,
                scope="Perturbing eager run (profiling.StageProfiler kernels=True): CUDA activity GPU time of each "
                      "kernel/memset; graph replay elsewhere. Kernel times are GPU execution, not host wall.",
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("replay", "kernels"), required=True)
    parser.add_argument("--seed", type=int, choices=(512, 15), required=True)
    parser.add_argument("--attempt", type=Path, help="Override the WP-17 attempt directory")
    parser.add_argument("--device", default="cuda:0", help="CUDA ordinal inside the process (after masking)")
    parser.add_argument("--gpu-index", type=int, required=True, help="Physical nvidia-smi index for telemetry")
    parser.add_argument("--run-name")
    parser.add_argument("--warm-resets", type=int, default=5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--kernel-steps", type=int, nargs="*")
    parser.add_argument("--kernel-repeat", type=int, default=3)
    parser.add_argument("--kernel-cache", choices=("user", "fresh"), default="user")
    parser.add_argument("--baseline-s", type=float, default=3.)
    parser.add_argument("--evaluate", action="store_true",
                        help="Keep the 121 replay states in RAM and apply the frozen seeded_trim goal evaluator")
    args = parser.parse_args()
    attempt = args.attempt or WP17 / f"seed_{args.seed}" / "box" / "env" / "recording" / "attempt_000"
    run = new_run_dir(args.run_name or f"{args.mode}_s{args.seed}")
    manifest = dict(schema="rcs_rl_p1_0_run_v1", argv=sys.argv, args={k: (str(v) if isinstance(v, Path) else v)
                                                                      for k, v in vars(args).items()},
                    boundary="single_env_headless", started_unix_s=WALL_SCRIPT, git_head=git("rev-parse", "HEAD"),
                    git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), python=sys.version.split()[0],
                    interpreter=os.path.relpath(sys.executable, ROOT), kit_started=False,
                    package_origin=os.path.relpath(sys.modules["robotic_cutting_sim"].__file__, ROOT),
                    source_sha256={path: sha256(ROOT / path) for path in SOURCES})
    import warp, numpy
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
    time.sleep(args.baseline_s)   # co-tenant baseline before our first GPU call
    telemetry.set_phase("setup")
    status, error, result = "FAIL", None, {}
    try:
        t0 = time.perf_counter()
        trajectory = extract_trajectory(attempt, locator_root=ROOT / "artifacts")
        extract_s = time.perf_counter() - t0
        trajectory.save(run / "trajectory")
        result = (run_replay if args.mode == "replay" else run_kernels)(args, run, telemetry, trajectory, attempt)
        result["trajectory"] = dict(extract_s=extract_s, steps=trajectory.steps, checks=trajectory.checks,
                                    options=trajectory.options, provenance=trajectory.provenance,
                                    source=trajectory.source)
        status = "COMPLETE" if result.get("complete", True) else "TRUNCATED"
    except Exception as exc:   # preserved as a failed run, never hidden
        import traceback
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        summary = telemetry.stop()
        with (run / "telemetry_samples.jsonl").open("x") as stream:
            for sample in telemetry.samples:
                stream.write(json.dumps(sample, default=str) + "\n")
    out = dict(status=status, error=error, mode=args.mode, seed=args.seed, telemetry=summary,
               wall_s=time.perf_counter() - T_SCRIPT, **result)
    write_json(run / "result.json", out)
    print(json.dumps(dict(run=str(run.relative_to(ROOT / "artifacts")), status=status,
                          gpu_sharing=summary.get("gpu_sharing"), wall_s=out["wall_s"])))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
