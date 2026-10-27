"""Small seeded task variations, separated from fixed recipes and solver code.

Generation supplies continuous geometry, local targets and reference actions.
It does not supply cut labels, force observations or a runtime success claim.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .benchmark import Bump, HeightFieldSpec
from .scene_frames import compose_pose, invert_transform, matrix_pose, pose_matrix

RANDOMIZATION_PROTOCOL_PATH = Path(__file__).resolve().parents[2]/"configs/task_randomization_v1.json"


def _protocol():
    config = json.loads(RANDOMIZATION_PROTOCOL_PATH.read_text())
    base = RANDOMIZATION_PROTOCOL_PATH.with_name("cutting_tasks_v1.json")
    if hashlib.sha256(base.read_bytes()).hexdigest() != config["base_task_protocol_sha256"]:
        raise ValueError("Randomized tasks require the frozen base task protocol; create a new randomization version for changes")
    return config


def _draws(task_id, seed):
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("Randomized task seed must be a nonnegative integer")
    config = _protocol()
    if task_id not in ("CUT-01", "TRIM-01"):
        raise ValueError("Randomization supports CUT-01 and TRIM-01")
    common, limits = config["common"], config[task_id]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    length = float(rng.choice(common["length_choices_m"]))
    xy = rng.uniform(-common["frame_xy_offset_limit_m"], common["frame_xy_offset_limit_m"], 2)
    yaw = float(rng.uniform(-common["frame_yaw_offset_limit_deg"], common["frame_yaw_offset_limit_deg"]))
    offset = [float(xy[0]), float(xy[1]), common["frame_offset_local_z_m"],
              *Rotation.from_euler("z", yaw, degrees=True).as_quat().tolist()]
    draws = {"length_m": length, "frame_offset_local_xyz_xyzw": offset, "frame_yaw_offset_deg": yaw}
    if task_id == "CUT-01":
        draws["cut_x_m"] = float(rng.choice(limits["cut_x_choices_m"]))
    else:
        count = int(rng.integers(limits["bump_count_inclusive"][0], limits["bump_count_inclusive"][1]+1))
        anchors = sorted(float(v) for v in rng.choice(limits["bump_x_anchors_m"], count, replace=False))
        bumps = []
        for anchor in anchors:
            bumps.append({"x_m": anchor+float(rng.uniform(-limits["bump_x_jitter_limit_m"], limits["bump_x_jitter_limit_m"])),
                "y_m": float(rng.uniform(-limits["bump_y_limit_m"], limits["bump_y_limit_m"])),
                "radius_x_m": float(rng.uniform(*limits["bump_radius_x_range_m"])),
                "radius_y_m": float(rng.uniform(*limits["bump_radius_y_range_m"])),
                "height_m": float(rng.uniform(*limits["bump_height_range_m"])),
                "family": str(rng.choice(limits["families"]))})
        draws.update(bump_anchors_m=anchors, bumps=bumps)
    return draws


def randomize_task(fixed_task):
    """Return one seeded variant of an explicit fixed task, preserving numerics."""
    from .tasks import FixtureBox

    if fixed_task.variant != "fixed":
        raise ValueError("Apply randomization once to a fixed task")
    config = _protocol()
    common, limits = config["common"], config[fixed_task.task_id]
    draws = _draws(fixed_task.task_id, fixed_task.seed)
    length = draws["length_m"]
    geometry = HeightFieldSpec(length, common["width_m"], common["base_height_m"],
        tuple(Bump(**row) for row in draws.get("bumps", [])))
    knots = np.asarray(fixed_task.edge_position_knots_local_txyz, float).copy()
    half_height = limits["support_half_height_m"]
    if fixed_task.task_id == "CUT-01":
        cut = draws["cut_x_m"]
        knots[:, 1] = cut
        bounds = [(-length/2-limits["support_outer_margin_m"], cut-limits["slot_half_width_m"]),
                  (cut+limits["slot_half_width_m"], length/2+limits["support_outer_margin_m"])]
        fixtures = tuple(FixtureBox(name, ((low+high)/2, 0., -half_height),
            ((high-low)/2, limits["support_half_width_m"], half_height))
            for name, (low, high) in zip(("left_pedestal", "right_pedestal"), bounds))
        target = (cut, 0., 0.)
    else:
        minimum = min(b.x_m-b.radius_x_m for b in geometry.bumps)
        maximum = max(b.x_m+b.radius_x_m for b in geometry.bumps)
        inset, margin = limits["reference_inset_from_bar_end_m"], limits["reference_margin_from_bump_bounds_m"]
        start, end = max(-length/2+inset, minimum-margin), min(length/2-inset, maximum+margin)
        withdraw = min(length/2+limits["withdrawal_maximum_beyond_bar_end_m"], end+limits["withdrawal_translation_m"])
        knots[:3, 1], knots[3, 1], knots[4:, 1] = start, end, withdraw
        fixtures = (FixtureBox("support_floor", (0., 0., -half_height),
            (length/2+limits["support_outer_margin_m"], limits["support_half_width_m"], half_height)),)
        target = fixed_task.target_origin_local_m
    actual_pose = compose_pose(fixed_task.world_from_workpiece_xyzw, draws["frame_offset_local_xyz_xyzw"])
    return replace(fixed_task, variant="randomized", geometry=geometry, fixture_boxes_local=fixtures,
        world_from_workpiece_xyzw=tuple(float(v) for v in actual_pose), target_origin_local_m=target,
        edge_position_knots_local_txyz=tuple(tuple(float(v) for v in row) for row in knots))


def randomization_receipt(task):
    """Record seed/draw protocol alongside the caller's complete actual TaskSpec."""
    if task.variant != "randomized":
        raise ValueError("Randomization receipts describe randomized tasks only")
    config = _protocol()
    draws = _draws(task.task_id, task.seed)
    input_matrix = pose_matrix(task.world_from_workpiece_xyzw)@invert_transform(pose_matrix(draws["frame_offset_local_xyz_xyzw"]))
    return {"protocol_id": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(RANDOMIZATION_PROTOCOL_PATH.read_bytes()).hexdigest(),
        "base_task_protocol_sha256": config["base_task_protocol_sha256"], "seed": task.seed,
        "rng": config["rng"], "numpy_version": np.__version__, "draws": draws,
        "input_world_from_workpiece_xyzw": matrix_pose(input_matrix).tolist(),
        "frame_composition": config["common"]["frame_composition"],
        "scope": "Seeded continuous initial geometry, local target, fixtures and scripted actions only; actual cutting and sensors require episode receipts",
        "policy_observation_contains_force_or_oracle": False}
