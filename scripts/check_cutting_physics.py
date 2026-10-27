#!/usr/bin/env python3
"""Frozen, process-isolated TLMPM engineering validation; no policy or stage promotion."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = ["src/robotic_cutting_sim/tlmpm.py", "src/robotic_cutting_sim/cut_topology.py",
                "src/robotic_cutting_sim/geometry.py", "src/robotic_cutting_sim/recording.py",
                "scripts/run_cutting_probe.py"]
TOLERANCES = {"repeat_max_position_m": 1e-4, "repeat_final_component_com_m": 5e-5,
              "repeat_max_velocity_m_s": .01, "micro_position_m": 1e-6,
              "micro_relative_momentum": 1e-5, "micro_stress_pa": .1,
              "micro_affine_F": 1e-6, "micro_impulse_budget_Ns": 1e-7,
              "supported_settle_peak_speed_m_s": 2., "supported_min_z_m": -.011,
              "plastic_strain_nonzero": 1e-6}


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+"\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_hashes():
    return {name: file_hash(ROOT/name) for name in FROZEN_FILES}


def case_matrix():
    cases = [{"name": name, "case": name, "dx": .004, "dt": .0001, "duration": 2.5}
             for name in ["settle", "no_contact", "indent", "uncut_scrape", "cut"]]
    cases += [{"name": "cut_repeat", "case": "cut", "dx": .004, "dt": .0001, "duration": 2.5},
              {"name": "cut_recontact", "case": "cut_recontact", "dx": .004, "dt": .0001, "duration": 2.5},
              {"name": "two_cuts", "case": "two_cuts", "dx": .004, "dt": .0001, "duration": 4.5},
              {"name": "cut_dx3mm", "case": "cut", "dx": .003, "dt": .0001, "duration": 2.5},
              {"name": "cut_dx2mm", "case": "cut", "dx": .002, "dt": .00005, "duration": 2.5},
              {"name": "cut_dt_half", "case": "cut", "dx": .004, "dt": .00005, "duration": 2.5}]
    return cases


def freeze(output, source_ref, cases=None):
    output.mkdir(parents=True, exist_ok=False)
    expected = {name: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{source_ref}:{name}"], cwd=ROOT)).hexdigest() for name in FROZEN_FILES}
    if current_hashes() != expected:
        raise RuntimeError("Current physics files differ from the requested frozen source revision")
    plan = {"source_ref": subprocess.check_output(["git", "rev-parse", source_ref], cwd=ROOT, text=True).strip(),
            "source_hashes": expected, "tolerances_frozen_before_runs": TOLERANCES,
            "seed": 42, "variant": "D0", "family": "dome", "frame_dt_s": .02,
            "cases": cases or case_matrix(), "created_unix_s": time.time(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version, "command": sys.argv,
            "initial_git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "runner_sha256": file_hash(__file__), "scope": "engineering_validation_only",
            "physics_proven": False, "benchmark_ready": False}
    write_json(output/"frozen_plan.json", plan)
    shutil.copy2(__file__, output/"executed_runner.py")
    return plan


def execute_matrix(output, plan, device):
    results = {}
    for case in plan["cases"]:
        if current_hashes() != plan["source_hashes"]:
            raise RuntimeError("Frozen physics source changed; matrix stopped")
        path = output/case["name"]
        command = [sys.executable, ROOT/"scripts/run_cutting_probe.py", "--output-dir", path,
                   "--case", case["case"], "--dx", case["dx"], "--dt", case["dt"],
                   "--duration", case["duration"], "--frame-dt", .02, "--seed", 42,
                   "--variant", "D0", "--family", "dome", "--device", device]
        command = list(map(str, command))
        monitor_command = ["nvidia-smi", "-i", os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0],
                           "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                           "--loop-ms=100"]
        tick = time.perf_counter()
        print(f"START {case['name']}", flush=True)
        with (output/f"{case['name']}_solver.log").open("w") as log, (output/f"{case['name']}_gpu.csv").open("w") as mem:
            monitor = subprocess.Popen(monitor_command, stdout=mem, stderr=subprocess.STDOUT)
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=1800)
            finally:
                monitor.terminate(); monitor.wait()
        record = {"command": command, "returncode": completed.returncode,
                  "wall_s_including_setup_export": time.perf_counter()-tick,
                  "gpu_sample_command": monitor_command,
                  "gpu_scope": "entire selected physical device including baseline; sampled100ms"}
        if (path/"result.json").is_file():
            record["probe_result"] = json.loads((path/"result.json").read_text())
        write_json(output/f"{case['name']}_execution.json", record)
        results[case["name"]] = record
        write_json(output/"executions.json", results)
        print(f"DONE {case['name']} exit={completed.returncode} wall={record['wall_s_including_setup_export']:.2f}s", flush=True)
    return results


def load_state(path):
    with np.load(path/"physics_state.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def reference_stencils(points, dx):
    """Reconstruct the frozen backend's reference support for its graph diagnostic."""
    origin = np.floor(points.min(0)/dx)*dx-2*dx
    dims = np.ceil((points.max(0)-origin)/dx).astype(int)+3
    cells = np.floor((points-origin)/dx).astype(int)
    fraction = (points-origin)/dx-cells
    nodes = np.empty((len(points), 8), np.int32)
    weights = np.empty((len(points), 8), np.float32)
    gradients = np.empty((len(points), 8, 3), np.float32)
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells+bit
        nodes[:, corner] = (xyz[:, 0]*dims[1]+xyz[:, 1])*dims[2]+xyz[:, 2]
        f = np.where(bit, fraction, 1-fraction)
        weights[:, corner] = f.prod(1)
        for axis in range(3):
            gradients[:, corner, axis] = (2*bit[axis]-1)*f[:, (axis+1)%3]*f[:, (axis+2)%3]/dx
    return nodes, weights, gradients


def analyze_case(path, case):
    from robotic_cutting_sim.cut_topology import mechanical_components
    state = load_state(path)
    q, v, codes, mass, times = (state[name] for name in ["positions", "velocities", "codes", "mass_kg", "time_s"])
    result = json.loads((path/"result.json").read_text())
    with (path/"trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    trace = {key: np.array([float(row[key]) for row in rows]) for key in rows[0] if key != "finite"}
    cut_count = 2 if case["case"] == "two_cuts" else 1
    changed = []
    for k in range(cut_count):
        digits = codes//3**k % 3
        changed.append(int(((digits[:-1] != 0) & (digits[1:] != digits[:-1])).sum()))
    withdraw = 3.55 if cut_count == 2 else 1.45
    post = times >= withdraw
    first_post = int(np.flatnonzero(post)[0]) if post.any() else None
    immutable_post = bool(first_post is not None and np.array_equal(
        codes[first_post:], np.broadcast_to(codes[first_post], codes[first_post:].shape)))
    component = state["component_ids"]
    anchor = q[0, :, 2] <= q[0, :, 2].min()+1e-7
    substrate_ids = np.unique(component[anchor])
    chip = ~np.isin(component, substrate_ids)
    chip_mass = float(mass[chip].sum())
    chip_com = np.average(q[:, chip], axis=1, weights=mass[chip]) if chip.any() else None
    dx = case["dx"]
    nodes, weights, gradients = reference_stencils(state["reference_positions"], dx)
    initial_count, _ = mechanical_components(nodes, weights, codes[0], cut_count, gradients=gradients)
    final_count, independently_recomputed = mechanical_components(nodes, weights, codes[-1], cut_count, gradients=gradients)
    positive_weight_count, _ = mechanical_components(nodes, weights, codes[-1], cut_count)
    graph_matches = bool(np.array_equal(component, independently_recomputed))
    at_withdraw_count = None
    if first_post is not None:
        at_withdraw_count, _ = mechanical_components(nodes, weights, codes[first_post], cut_count, gradients=gradients)
    material_momentum = (mass[None, :, None]*v).sum(1)
    tool_force = np.column_stack([trace[name] for name in ["blade_fx_N", "blade_fy_N", "blade_fz_N"]])
    tool_impulse = tool_force.sum(0)*.02
    gravity_impulse = np.array([0., 0., -9.81])*mass.sum()*(times[-1]-times[0])
    unresolved_fixture = material_momentum[-1]-material_momentum[0]+tool_impulse-gravity_impulse
    second_digits = codes//3 % 3 if cut_count == 2 else None
    first_after_first = np.flatnonzero(times >= 1.45)
    first_preserved = None
    if cut_count == 2 and len(first_after_first):
        first_digit = codes % 3
        start = first_after_first[0]
        first_preserved = bool(np.array_equal(first_digit[start:], np.broadcast_to(first_digit[start], first_digit[start:].shape)))
    return {"case": case, "probe_status": result["status"], "frames": len(q),
            "completed_requested_duration": bool(abs(times[-1]-case["duration"]) < 1e-6),
            "particle_count": q.shape[1], "initial_mass_kg": float(mass.sum()),
            "mass_scope": "immutable source particle IDs/masses; NPZ has static masses, trace repeats their sum",
            "finite_positions_velocities_F": bool(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(state["final_F"]).all()),
            "sampled_max_speed_m_s": float(np.linalg.norm(v, axis=2).max()),
            "min_z_m": float(q[:, :, 2].min()), "min_detF": result["minimum_detF"],
            "initial_mechanical_components": initial_count, "final_mechanical_components": final_count,
            "positive_weight_only_components": positive_weight_count,
            "nonzero_gradient_only_support_entries": int(((weights == 0) & (np.linalg.norm(gradients, axis=2) > 0)).sum()),
            "stored_component_graph_matches_recomputed": graph_matches,
            "component_count_at_withdrawal": at_withdraw_count,
            "component_graph_scope": "conservative actual reference-stencil compatibility incl code0 and nonzero-gradient supports",
            "initial_codes_all_zero": bool(not codes[0].any()),
            "final_codes_histogram": {str(int(c)): int(n) for c, n in zip(*np.unique(codes[-1], return_counts=True))},
            "nonzero_digit_changes": changed, "post_withdraw_codes_bitwise_unchanged": immutable_post,
            "second_digit_assigned_particles": None if second_digits is None else int((second_digits[-1] != 0).sum()),
            "first_digit_preserved_through_second_cut": first_preserved,
            "chip_particle_count": int(chip.sum()), "chip_mass_kg": chip_mass,
            "substrate_mass_kg": float(mass[~chip].sum()),
            "partition_residual_kg": float(mass.sum()-mass[chip].sum()-mass[~chip].sum()),
            "chip_final_com_m": None if chip_com is None else chip_com[-1].tolist(),
            "chip_postwithdraw_com_displacement_m": None if chip_com is None or first_post is None else float(np.linalg.norm(chip_com[-1]-chip_com[first_post])),
            "chip_postwithdraw_min_com_z_m": None if chip_com is None or first_post is None else float(chip_com[first_post:, 2].min()),
            "final_plastic_strain_max": float(state["final_plastic_strain"].max()),
            "final_plastic_strain_mass_mean": float(np.average(state["final_plastic_strain"], weights=mass)),
            "final_accumulated_contact_work_j": float(state["final_contact_work_j"].sum()),
            "mean_final_vertical_displacement_m": float(np.average(q[-1, :, 2]-q[0, :, 2], weights=mass)),
            "frame_sampled_tool_force_finite": bool(np.isfinite(tool_force).all()),
            "frame_sampled_peak_tool_force_n": float(np.linalg.norm(tool_force, axis=1).max()),
            "tool_reaction_impulse_Ns": tool_impulse.tolist(),
            "material_momentum_change_kg_m_s": (material_momentum[-1]-material_momentum[0]).tolist(),
            "gravity_impulse_Ns": gravity_impulse.tolist(),
            "unresolved_fixture_and_numerical_impulse_Ns": unresolved_fixture.tolist(),
            "full_impulse_balance_status": "NOT_EVALUATED: fixture reaction impulse is not recorded",
            "postwithdraw_material_contact_pairs": int(trace["material_contacts"][trace["time_s"] >= withdraw].sum()),
            "postwithdraw_material_impulse_sum_magnitudes_Ns": float(trace["material_contact_impulse_Ns"][trace["time_s"] >= withdraw].sum()),
            "tool_contact_substep_particles": int(trace["tool_contacts"].sum()),
            "max_tool_penetration_m": result["max_tool_penetration_m"],
            "max_material_overlap_m": result["maximum_material_overlap_m"],
            "physics_wall_s": result["physics_wall_s"],
            "state_sha256": file_hash(path/"physics_state.npz"), "trace_sha256": file_hash(path/"trace.csv")}


def repeat_comparison(a_path, b_path, tolerances):
    a, b = load_state(a_path), load_state(b_path)
    if a["positions"].shape != b["positions"].shape:
        return {"status": "FAIL", "reason": "Recorded frame/particle shapes differ"}
    position = float(np.abs(a["positions"]-b["positions"]).max())
    velocity = float(np.abs(a["velocities"]-b["velocities"]).max())
    com_error = 0.
    for label in np.unique(a["component_ids"]):
        mask = a["component_ids"] == label
        ca = np.average(a["positions"][-1, mask], axis=0, weights=a["mass_kg"][mask])
        cb = np.average(b["positions"][-1, mask], axis=0, weights=b["mass_kg"][mask])
        com_error = max(com_error, float(np.linalg.norm(ca-cb)))
    labels_equal = bool(np.array_equal(a["codes"], b["codes"]))
    components_equal = bool(np.array_equal(a["component_ids"], b["component_ids"]))
    passed = (position <= tolerances["repeat_max_position_m"] and velocity <= tolerances["repeat_max_velocity_m_s"]
              and com_error <= tolerances["repeat_final_component_com_m"])
    return {"status": "PASS" if passed else "FAIL", "tolerances_frozen_before_repeat": tolerances,
            "all_frame_max_position_abs_difference_m": position,
            "all_frame_max_velocity_abs_difference_m_s": velocity,
            "max_final_reference_component_com_distance_m": com_error,
            "positions_bitwise_equal": bool(np.array_equal(a["positions"], b["positions"])),
            "velocities_bitwise_equal": bool(np.array_equal(a["velocities"], b["velocities"])),
            "codes_bitwise_equal": labels_equal, "component_ids_bitwise_equal": components_equal,
            "final_codes_bitwise_equal": bool(np.array_equal(a["codes"][-1], b["codes"][-1])),
            "different_code_entries": int((a["codes"] != b["codes"]).sum()),
            "different_code_frame_indices": np.flatnonzero((a["codes"] != b["codes"]).any(axis=1)).tolist(),
            "exact_reproducibility_status": "PASS" if np.array_equal(a["positions"], b["positions"]) and np.array_equal(a["velocities"], b["velocities"]) and labels_equal else "FAIL"}


def gpu_microcontrols(device, tolerances):
    import warp as wp
    from robotic_cutting_sim import tlmpm as solver
    checks = {}
    wp.init()
    axis = np.array([-.003, -.001, .001, .003], np.float32)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    mass = np.full(len(points), 1500*.002**3)
    config = solver.TLMPMConfig(cuts=1, floor=False, tool=False, material_contact=False,
                               gravity=(0., 0., 0.), yield_stress=1e9)
    bed = solver.TLMPM(points, mass, .002, config, device)
    v0 = np.array([.02, -.015, .01], np.float32)
    initial_momentum = bed.mass.numpy().astype(float).sum()*v0
    bed.v.assign(np.broadcast_to(v0, points.shape).copy())
    for _ in range(10):
        bed.step(advance=False)
    state = bed.snapshot()
    wp.synchronize_device(device)
    expected = points+10*config.dt*v0
    position_error = float(np.abs(state["positions"]-expected).max())
    momentum = (bed.mass.numpy()[:, None].astype(float)*state["velocities"]).sum(0)
    momentum_error = float(np.linalg.norm(momentum-initial_momentum)/np.linalg.norm(initial_momentum))
    checks["force_free_uniform_translation"] = {"status": "PASS" if position_error <= tolerances["micro_position_m"] else "FAIL",
        "steps": 10, "max_position_error_m": position_error, "tolerance_m": tolerances["micro_position_m"]}
    checks["force_free_linear_momentum"] = {"status": "PASS" if momentum_error <= tolerances["micro_relative_momentum"] else "FAIL",
        "relative_error": momentum_error, "tolerance_relative": tolerances["micro_relative_momentum"]}
    with wp.ScopedDevice(device):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec([.2, -.15, .3]).as_matrix().astype(np.float32)
        f = wp.array(np.broadcast_to(R, (len(points), 3, 3)).copy(), dtype=wp.mat33)
        identity = wp.array(np.tile(np.eye(3, dtype=np.float32), (len(points), 1, 1)), dtype=wp.mat33)
        stress = wp.zeros(len(points), dtype=wp.mat33)
        plastic = wp.zeros(len(points), dtype=float)
        lam = config.young*config.poisson/((1+config.poisson)*(1-2*config.poisson))
        mu = config.young/(2*(1+config.poisson))
        wp.launch(solver._stress, len(points), [f, identity, stress, plastic, lam, mu, 1e9])
        stress_error = float(np.abs(stress.numpy()).max())
        checks["rigid_rotation_zero_stress"] = {"status": "PASS" if stress_error <= tolerances["micro_stress_pa"] else "FAIL",
            "max_first_piola_stress_pa": stress_error, "tolerance_pa": tolerances["micro_stress_pa"]}
        A = np.array([[0, 1.2, -.3], [-.7, .2, 0], [.4, 0, -.1]], np.float32)
        flat = np.arange(bed.node_count)
        dims = bed.grid_dims
        ijk = np.column_stack([flat//(dims[1]*dims[2]), flat//dims[2] % dims[1], flat % dims[2]])
        node_positions = bed.grid_origin+config.dx*ijk
        field = (node_positions@A.T+v0).astype(np.float32)
        bed.new.assign(np.tile(field, (bed.code_count, 1)))
        bed.F.assign(np.tile(np.eye(3, dtype=np.float32), (len(points), 1, 1)))
        wp.launch(solver._deformation, len(points), [bed.F, bed.code, bed.nodes,
                    bed.gradients, bed.new, bed.node_count, config.dt])
        affine_error = float(np.abs(bed.F.numpy()-(np.eye(3)+config.dt*A)).max())
        checks["affine_reference_velocity_gradient"] = {"status": "PASS" if affine_error <= tolerances["micro_affine_F"] else "FAIL",
            "max_deformation_gradient_error": affine_error, "tolerance": tolerances["micro_affine_F"]}

    # Call actual contact kernels with no rigid integration, and inspect equal/opposite impulses.
    contact_points = np.array([[-.001, 0, 0], [-.001, 0, .001]], np.float32)
    contact = solver.TLMPM(contact_points, np.full(2, 1500*.002**3), .002,
        solver.TLMPMConfig(cuts=2, floor=False, gravity=(0., 0., 0.), material_contact=False), device)
    contact.set_trajectory(np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32), np.zeros((1, 6), np.float32), np.array([-1], np.int32))
    c = contact.config
    with wp.ScopedDevice(device):
        wp.launch(solver._rigid_contacts, 2, [contact.x, contact.v, contact.mass, contact.impulse,
            contact.progress, contact.contact_work, contact.poses, contact.twists, contact.cuts, contact.index,
            contact.reaction, contact.stats, contact.radius, contact.volume, c.dt, c.contact_stiffness,
            c.contact_friction, c.cutting_energy, c.sharpness, c.blade_length, c.blade_height,
            c.blade_half_width, 0, 1])
        material_impulse = contact.impulse.numpy().astype(float).sum(0)
        reaction = contact.reaction.numpy()[0, :3].astype(float)
        error = float(np.linalg.norm(material_impulse+reaction))
        checks["tool_material_impulse_budget"] = {"status": "PASS" if error <= tolerances["micro_impulse_budget_Ns"] and np.linalg.norm(reaction) > 0 else "FAIL",
            "material_impulse_Ns": material_impulse, "tool_reaction_impulse_Ns": reaction,
            "budget_residual_norm_Ns": error, "tolerance_Ns": tolerances["micro_impulse_budget_Ns"]}
        contact.x.assign(np.array([[-.00075, 0, 0], [.00075, 0, 0]], np.float32))
        contact.code.assign(np.array([1, 2], np.int32)); contact.impulse.zero_(); contact.stats.zero_()
        contact.hash.build(contact.x, 2.4*contact.radius)
        wp.launch(solver._particle_contacts, 2, [contact.hash.id, contact.x, contact.reference,
            contact.v, contact.mass, contact.code, contact.impulse, contact.stats, contact.radius,
            3*c.dx, c.dt, c.particle_contact_stiffness, c.contact_friction])
        pair = contact.impulse.numpy().astype(float)
        error = float(np.linalg.norm(pair.sum(0)))
        checks["separated_material_pair_impulse_budget"] = {"status": "PASS" if error <= tolerances["micro_impulse_budget_Ns"] and np.linalg.norm(pair[0]) > 0 else "FAIL",
            "pair_impulses_Ns": pair, "budget_residual_norm_Ns": error,
            "tolerance_Ns": tolerances["micro_impulse_budget_Ns"]}
        contact.x.assign(np.array([[-.001, 0, -.001], [-.001, 0, .001]], np.float32))
        contact.code.zero_()
        before = contact.snapshot(); before_mass = contact.mass.numpy()
        contact.cuts.assign(np.array([0], np.int32))
        wp.launch(solver._assign, 2, [contact.x, contact.v, contact.code, contact.poses, contact.twists,
            contact.cuts, contact.index, 6*c.dx, c.blade_half_width, 2*c.dx, c.dt])
        first = contact.snapshot()
        contact.cuts.assign(np.array([1], np.int32))
        wp.launch(solver._assign, 2, [contact.x, contact.v, contact.code, contact.poses, contact.twists,
            contact.cuts, contact.index, 6*c.dx, c.blade_half_width, 2*c.dx, c.dt])
        second = contact.snapshot()
        unchanged = all(np.array_equal(before[key], second[key]) for key in ["positions", "velocities", "F", "Fpi"])
        unchanged &= np.array_equal(before_mass, contact.mass.numpy())
        checks["online_digits_do_not_modify_state_instantly"] = {"status": "PASS" if unchanged and first["codes"].tolist() == [1, 2] and second["codes"].tolist() == [4, 8] else "FAIL",
            "first_codes": first["codes"], "second_codes": second["codes"],
            "positions_velocities_F_Fpi_mass_bitwise_unchanged": unchanged}
    return {"status": "PASS" if all(c["status"] == "PASS" for c in checks.values()) else "FAIL",
            "scope": "actual_frozen_GPU_kernel_analytic_controls", "checks": checks}


def recontact_saved_frame_kernel_probe(path, device):
    """Reevaluate the actual material-contact kernel on saved post-withdraw states.

    Internal pair impulses cancel in each component. A nonzero component net
    impulse therefore measures cross-component contact, without adding a force
    or changing the recorded state. This is retrospective sampled-frame evidence.
    """
    import warp as wp
    from robotic_cutting_sim import tlmpm as solver
    state = load_state(path)
    config_data = json.loads((path/"manifest.json").read_text())["config"]["solver"]
    config = solver.TLMPMConfig(**config_data)
    bed = solver.TLMPM(state["reference_positions"], state["mass_kg"], config.dx/2, config, device)
    labels = state["component_ids"]
    samples = []
    for frame in np.flatnonzero(state["time_s"] >= 1.45):
        bed.x.assign(state["positions"][frame]); bed.v.assign(state["velocities"][frame]); bed.code.assign(state["codes"][frame])
        bed.impulse.zero_(); bed.stats.zero_()
        with wp.ScopedDevice(device):
            bed.hash.build(bed.x, 2.4*bed.radius)
            wp.launch(solver._particle_contacts, bed.n, [bed.hash.id, bed.x, bed.reference, bed.v,
                bed.mass, bed.code, bed.impulse, bed.stats, bed.radius, 3*config.dx, config.dt,
                config.particle_contact_stiffness, config.contact_friction])
        impulse, stats = bed.impulse.numpy().astype(float), bed.stats.numpy()
        component_impulse = {str(int(c)): impulse[labels == c].sum(0).tolist() for c in np.unique(labels)}
        samples.append({"time_s": float(state["time_s"][frame]), "material_contact_pairs": int(stats[4]),
                        "component_net_impulses_Ns": component_impulse,
                        "closed_pair_budget_residual_Ns": float(np.linalg.norm(impulse.sum(0)))})
    peak = max((np.linalg.norm(v) for sample in samples for v in sample["component_net_impulses_Ns"].values()), default=0.)
    return {"status": "OBSERVED" if peak > 1e-9 else "NOT_OBSERVED",
            "scope": "actual contact kernel reevaluated on recorded frames; not an original-substep pair log",
            "roundoff_reporting_floor_Ns": 1e-9, "max_component_net_contact_impulse_Ns": float(peak),
            "samples": samples}


def plot_overview(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name in ["settle", "uncut_scrape", "cut", "cut_recontact", "two_cuts"]:
        if name not in paths:
            continue
        with (paths[name]/"trace.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        times = [float(r["time_s"]) for r in rows]
        axes[0, 0].plot(times, [float(r["max_speed_m_s"]) for r in rows], label=name)
        if name in ("cut", "uncut_scrape", "cut_recontact"):
            axes[0, 1].plot(times, [float(r["blade_fz_N"]) for r in rows], label=name)
    axes[0, 0].set(xlabel="time [s]", ylabel="sampled peak speed [m/s]")
    axes[0, 1].set(xlabel="time [s]", ylabel="tool reaction Z [N], diagnostic")
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    for ax, name in zip(axes[1], ["cut", "cut_recontact"]):
        state = load_state(paths[name])
        q = state["positions"][-1]
        selected = np.abs(q[:, 1]) < .004
        ax.scatter(q[selected, 0]*1000, q[selected, 2]*1000, s=3,
                   c=state["component_ids"][selected], cmap="tab10")
        ax.axhline(15.5, linestyle="--", linewidth=.7, color="gray")
        ax.set(title=f"{name}, final recorded particle slice", xlabel="world X [mm]", ylabel="world Z [mm]")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Frozen TLMPM diagnostics — uncalibrated material; colors are mechanical components")
    fig.tight_layout(); fig.savefig(output/"validation_overview.png", dpi=140); plt.close(fig)


def analyze_matrix(output, plan, device, supplemental_two_cuts=None):
    index = 1
    while (output/f"analysis_{index:03d}").exists():
        index += 1
    report_dir = output/f"analysis_{index:03d}"
    report_dir.mkdir()
    analysis_plan = {"source_hashes": current_hashes(), "runner_sha256": file_hash(__file__),
                     "created_unix_s": time.time(), "repeat_tolerances": plan["tolerances_frozen_before_runs"],
                     "contact_kernel_roundoff_reporting_floor_Ns": 1e-9,
                     "scope": "engineering analysis of retained runs; no calibration/convergence promotion"}
    write_json(report_dir/"analysis_plan.json", analysis_plan)
    executions = json.loads((output/"executions.json").read_text())
    paths, cases, setup_failures = {}, {}, {}
    for case in plan["cases"]:
        path = output/case["name"]
        if not (path/"physics_state.npz").is_file():
            setup_failures[case["name"]] = executions[case["name"]]
            continue
        cases[case["name"]] = analyze_case(path, case)
        paths[case["name"]] = path
        write_json(report_dir/f"{case['name']}_analysis.json", cases[case["name"]])
    supplemental = None
    if supplemental_two_cuts:
        delta = Path(supplemental_two_cuts).resolve()
        delta_plan = json.loads((delta/"frozen_plan.json").read_text())
        case = next(c for c in delta_plan["cases"] if c["name"] == "two_cuts")
        cases["two_cuts"] = analyze_case(delta/"two_cuts", case)
        paths["two_cuts"] = delta/"two_cuts"
        write_json(report_dir/"two_cuts_delta_analysis.json", cases["two_cuts"])
        supplemental = {"path": str(delta), "source_ref": delta_plan["source_ref"],
                        "source_hashes": delta_plan["source_hashes"],
                        "changed_files_from_initial_freeze": [k for k in plan["source_hashes"]
                                                              if plan["source_hashes"][k] != delta_plan["source_hashes"][k]]}
        expected = delta_plan["source_hashes"]
    else:
        expected = plan["source_hashes"]
    if current_hashes() == expected:
        micro = gpu_microcontrols(device, plan["tolerances_frozen_before_runs"])
        contact = recontact_saved_frame_kernel_probe(paths["cut_recontact"], device) if "cut_recontact" in paths else {"status": "NOT_RUN"}
    else:
        micro = {"status": "BLOCKED", "reason": "Core/probe changed since the specified frozen analysis basis"}
        contact = {"status": "BLOCKED", "reason": "Frozen source mismatch"}
    write_json(report_dir/"gpu_microcontrols.json", micro)
    write_json(report_dir/"saved_frame_cross_component_contact.json", contact)
    repeat = repeat_comparison(paths["cut"], paths["cut_repeat"], plan["tolerances_frozen_before_runs"]) if {"cut", "cut_repeat"} <= paths.keys() else {"status": "NOT_RUN"}
    write_json(report_dir/"repeat_comparison.json", repeat)
    required = {"settle", "no_contact", "indent", "uncut_scrape", "cut", "cut_recontact"}
    tests = {}
    if required <= cases.keys():
        settle, none, indent, uncut, cut, recontact = (cases[n] for n in ["settle", "no_contact", "indent", "uncut_scrape", "cut", "cut_recontact"])
        support = settle["probe_status"] == "PASS" and settle["completed_requested_duration"]
        negative = none["final_codes_histogram"] == {"0": none["particle_count"]} and none["tool_contact_substep_particles"] == 0
        indentation = indent["tool_contact_substep_particles"] > 0 and indent["final_plastic_strain_max"] > plan["tolerances_frozen_before_runs"]["plastic_strain_nonzero"]
        separated = (cut["initial_mechanical_components"] == 1 and cut["final_mechanical_components"] >= 2
                     and cut["chip_mass_kg"] > 0 and cut["stored_component_graph_matches_recomputed"])
        persistent = (cut["component_count_at_withdrawal"] >= 2 and cut["post_withdraw_codes_bitwise_unchanged"]
                      and not any(cut["nonzero_digit_changes"]))
        recontacts = (recontact["final_mechanical_components"] >= 2 and recontact["postwithdraw_material_contact_pairs"] > 0
                      and recontact["post_withdraw_codes_bitwise_unchanged"] and contact["status"] == "OBSERVED")
        tests["T1"] = {"status": "PASS" if support and negative else "FAIL", "scope": "supported settling and active-band noncontact negative control"}
        tests["T2"] = {"status": "PASS" if indentation else "FAIL", "scope": "tool indentation/unload with positive accumulated plastic strain; uncalibrated and no elastic-law control in this matrix"}
        tests["T3"] = {"status": "PASS" if separated else "FAIL", "scope": "online single cut creates disconnected conservative reference transfer/stress graph from one initial component",
                       "uncut_scrape_component_count": uncut["final_mechanical_components"]}
        tests["T4"] = {"status": "PASS" if persistent else "FAIL", "scope": "persistent digits and disconnected conservative graph after withdrawal"}
        tests["T5"] = {"status": "PASS" if recontacts else "FAIL", "scope": "original postwithdraw contact activity plus actual contact-kernel cross-component impulses on saved states; no healing"}
        tests["T6"] = {"status": "NOT_RUN", "scope": "second orthogonal cut"}
        if "two_cuts" in cases:
            second = cases["two_cuts"]
            success = (second["completed_requested_duration"] and second["probe_status"] == "PASS"
                       and second["second_digit_assigned_particles"] > 0 and second["first_digit_preserved_through_second_cut"]
                       and not any(second["nonzero_digit_changes"]) and second["final_mechanical_components"] >= 3)
            tests["T6"] = {"status": "PASS" if success else "FAIL", "scope": "orthogonal second cut retains first digit, assigns second digit and creates at least one further mechanical component"}
        tests["T7"] = {"status": "NOT_RUN", "scope": "Formal convergence not established; exploratory grids/time step separately reported"}
        tests["T8"] = {"status": repeat["status"], "scope": "fresh-process same-seed position/velocity/component-COM tolerances frozen before either run",
                       "exact_reproducibility_status": repeat.get("exact_reproducibility_status")}
        tests["T9"] = {"status": "PASS" if micro["status"] == "PASS" and all(abs(c["partition_residual_kg"]) < 1e-12 for c in cases.values()) else "FAIL",
                       "scope": "immutable mass/ID ledger, component partition, local analytic impulse budgets; full run impulse/energy balance NOT_EVALUATED"}
        tests["T10"] = {"status": "NOT_RUN", "scope": "Dynamic separated mesh/RGB-D validation is external to this matrix"}
    refinement = {name: {key: cases[name][key] for key in ["particle_count", "initial_mass_kg", "chip_mass_kg",
                   "final_mechanical_components", "sampled_max_speed_m_s", "min_detF", "max_tool_penetration_m"]}
                  for name in ["cut", "cut_dx3mm", "cut_dx2mm", "cut_dt_half"] if name in cases}
    refinement["scope"] = "exploratory: grid changes also alter initial particle quadrature, mass, support/contact radii and blade band; not a controlled convergence proof"
    for name, case in cases.items():
        csv_path = output/f"{name}_gpu.csv" if name != "two_cuts" or not supplemental_two_cuts else Path(supplemental_two_cuts)/"two_cuts_gpu.csv"
        memory = []
        if csv_path.is_file():
            for line in csv_path.read_text().splitlines():
                try:
                    memory.append(float(line.split(",")[1]))
                except (ValueError, IndexError):
                    pass
        case["sampled_entire_device_peak_mib"] = max(memory) if memory else None
    if "cut" in paths and "cut_recontact" in paths:
        plot_overview(paths, report_dir)
    summary = {"scope": "frozen_TLMPM_engineering_validation", "source_plan": plan,
               "analysis_plan": analysis_plan, "supplemental_two_cuts": supplemental,
               "original_setup_failures_preserved": setup_failures,
               "tests": tests, "cases": cases, "gpu_microcontrols": micro,
               "repeat": repeat, "refinement": refinement,
               "cross_component_contact": {k: v for k, v in contact.items() if k != "samples"},
               "physics_proven": False, "benchmark_ready": False,
               "formal_limitations": ["Uncalibrated material/contact/fracture parameters",
                                      "Grid/time convergence and full impulse/energy balances unfinished",
                                      "Robot/renderer integration not established by this standalone matrix"]}
    engineering = [t["status"] for name, t in tests.items() if name not in ("T7", "T10")]
    summary["engineering_status"] = "PASS" if engineering and all(status == "PASS" for status in engineering) else "FAIL"
    write_json(report_dir/"summary.json", summary)
    print(json.dumps({"engineering_status": summary["engineering_status"], "summary": str(report_dir/"summary.json"),
                      "tests": tests, "benchmark_ready": False}))
    return 0 if summary["engineering_status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-ref", default="855d1db")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--only-case", action="append", choices=[c["name"] for c in case_matrix()])
    parser.add_argument("--supplemental-two-cuts", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Set an explicit CUDA_VISIBLE_DEVICES for GPU isolation")
    if args.phase != "analyze":
        cases = [c for c in case_matrix() if not args.only_case or c["name"] in args.only_case]
        plan = freeze(output, args.source_ref, cases)
        execute_matrix(output, plan, args.device)
    else:
        plan = json.loads((output/"frozen_plan.json").read_text())
    if args.phase != "run":
        return analyze_matrix(output, plan, args.device, args.supplemental_two_cuts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
