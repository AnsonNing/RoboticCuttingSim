"""Shared helpers for the batched-material CPU tests (ADR 0021 §5 L0/L1).

Small TRIM scenes (seeded_trim seeds, coarse particle spacing 2.5 mm, unchanged 4 mm grid, 0.1 ms material
dt and 200 substeps per 20 ms control step) keep Warp-CPU runtimes to seconds per control step. Every env
of a batch is compared with its own frozen single ``LiveCuttingMaterial`` driven by the same measured edge
poses (the frozen SciPy ``interpolate_measured_poses`` output) and the same cut digits.
"""
from dataclasses import replace

import numpy as np
import warp as wp

from robotic_cutting_sim.coupled_robot import CouplingConfig, interpolate_measured_poses
from robotic_cutting_sim.live_material import LiveCuttingMaterial
from robotic_cutting_sim.scene_config import compile_task
from robotic_cutting_sim.seeded_trim import generate_scene, goal_config

SPACING = .0025
# Edge x (world) about 2 mm before each seed's first protrusion at the reference cutting height (t = 4 s).
BUMP_FRONT_X = {512: .0365, 15: -.0372, 7: -.0390}
# (dx per control step [m], cut digit): arm digit 0, switch to digit 1 (progress restart), then disarm.
SEQUENCE = ((.001, 0), (.001, 0), (.001, 0), (.001, 1), (.0005, -1))

_TASKS = {}


def trim_task(seed, spacing=SPACING, material=None):
    key = (seed, spacing)
    if key not in _TASKS:
        _TASKS[key] = compile_task(generate_scene(seed), goal_config(), seed=42, particle_spacing_m=spacing)
    task = _TASKS[key]
    if material:
        task = replace(task, material=replace(task.material, **material))
    return task


def trim_material(seed, spacing=SPACING, material=None):
    """An un-reset frozen single v7_2 material on the Warp CPU device (the Native configured-task path)."""
    task = trim_task(seed, spacing, material)
    config = CouplingConfig(control_dt=task.control_dt_s, material_dt=1e-4, dx=task.grid_dx_m, device="cpu")
    return LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                               solver="tlmpm_v7_2")


def start_pose(seed, spacing=SPACING):
    pose = np.asarray(trim_task(seed, spacing).reference_edge_pose_world(4.0), float)
    pose[0] = BUMP_FRONT_X[seed]
    return pose


def advance_single(material, pose, step, cut):
    """LiveCuttingMaterial.advance with an explicit digit (-1 disarms)."""
    return material.advance(pose, (step+1)*material.config.control_dt, cutting=cut >= 0, cut_index=max(cut, 0))


def host_inputs(materials, last, current, substeps):
    """Frozen per-env interpolation of the single path (its _pose_local + interpolate_measured_poses)."""
    poses, twists = [], []
    for m, a, b in zip(materials, last, current):
        p, t = interpolate_measured_poses(m._pose_local(a), m._pose_local(b), substeps, m.config.control_dt)
        poses.append(p)
        twists.append(t)
    return np.concatenate(poses), np.concatenate(twists)


def drive_solver(solver, poses, twists, cuts):
    """One control step of a bare BatchedTLMPMv7_2 as LiveCuttingMaterial.advance does it (CPU, no graph)."""
    T = solver.trajectory_length
    solver.poses.assign(poses)
    solver.twists.assign(twists)
    solver.cuts.assign(np.repeat(np.asarray(cuts, np.int32), T))
    solver.zero_control_step_accumulators()
    for _ in range(solver.substeps):
        solver.step()


def single_arrays(bed):
    """Every Warp array of a frozen single solver (vars()-complete), plus its fixture data."""
    out = {name: value.numpy() for name, value in vars(bed).items() if isinstance(value, wp.array)}
    if bed.fixture_data is not None:
        _, centers, halves, plane = bed.fixture_data
        out["fixture_data"] = (centers.numpy(), halves.numpy(), np.array(plane, np.float32))
    return out


def compare_env(solver, e, bed):
    """Names of arrays where env e of the batch differs (np.array_equal) from the single solver."""
    snap = solver.snapshot(e)
    reference = single_arrays(bed)
    missing = sorted(set(reference)-set(snap))
    if missing:
        raise AssertionError(f"snapshot lacks single-solver arrays {missing}")
    bad = []
    for name, value in reference.items():
        if name == "fixture_data":
            if not all(np.array_equal(a, b) for a, b in zip(snap[name], value)):
                bad.append(name)
        elif snap[name].shape != value.shape or not np.array_equal(snap[name], value):
            bad.append(name)
    return bad


def compare_snapshots(a, b):
    bad = []
    for name in a:
        if name == "fixture_data":
            if not all(np.array_equal(x, y) for x, y in zip(a[name], b[name])):
                bad.append(name)
        elif not np.array_equal(a[name], b[name]):
            bad.append(name)
    return bad
