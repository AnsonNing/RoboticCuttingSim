"""Measured-trajectory extraction (robotic_cutting_sim.rl.replay) on a synthetic attempt and, when present,
on the real WP-17 Native recordings (that test is SKIPPED, i.e. NOT_RUN, when the data are absent)."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.finger_geometry import pad_boxes_edge
from robotic_cutting_sim.rl.replay import (MeasuredTrajectory, compare_sample, digit_counts, extract_trajectory)
from robotic_cutting_sim.scene_frames import compose_pose

ROOT = Path(__file__).resolve().parents[1]
WP17 = ROOT / "artifacts" / "wp17" / "native_optimized_001"
PATHS = ["/W/link0", "/W/hand/leftfinger", "/W/hand/rightfinger", "/W/Tools/Knife"]
HALF = np.array([0.008, 0.007, 0.006])
CORNERS = (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * HALF
           + np.array([0.0, 0.01, 0.04])).tolist()
HANDLE = [[-0.028, -0.09, 0.0], [0.012, 0.03, 0.011]]


def _pose(position, rotvec=(0, 0, 0)):
    return np.r_[position, Rotation.from_rotvec(rotvec).as_quat()]


def make_attempt(tmp_path, *, steps=4, tool_from_edge=(0.028, 0., 0.09, 0., 0., 0., 1.), rotate=False,
                 cut=(False, True, True, False), corrupt_cut=False):
    attempt = tmp_path / "env" / "recording" / "attempt_000"
    native = tmp_path / "env" / "native"
    attempt.mkdir(parents=True)
    native.mkdir(parents=True)
    tool_from_edge = np.asarray(tool_from_edge, float)
    body = np.zeros((steps + 1, len(PATHS), 7))
    for t in range(steps + 1):
        rot = (0.05 * t, -0.02 * t, 0.1) if rotate else (0, 0, 0)
        body[t, 0] = _pose([-0.46, 0.12, -0.01])
        body[t, 3] = _pose([0.01 * t, 0.002, 0.28 - 0.001 * t], rot)
        body[t, 1] = _pose([0.01 * t - 0.01, 0.011, 0.33 - 0.001 * t], rot)
        body[t, 2] = _pose([0.01 * t + 0.01, -0.011, 0.33 - 0.001 * t], rot)
    edges = compose_pose(body[:, 3], tool_from_edge)
    pads = [dict(collider_path=PATHS[i] + "/pad", body_path=PATHS[i], corners_body_m=CORNERS) for i in (1, 2)]
    reset_boxes = np.asarray([HANDLE, *pad_boxes_edge(pads, {PATHS[1]: body[0, 1], PATHS[2]: body[0, 2]}, edges[0])],
                             np.float32)
    payload = {"protocol_id": "rcs_configured_task_v1", "task_id": "CFG-synthetic", "seed": 42,
               "scene": {"solver": {"cutting_law": 8, "grid_dx_m": 0.004, "max_cuts": 2, "module": "tlmpm_v7_2",
                                    "particle_spacing_m": 0.001}}}
    config = dict(seed=42, control_dt=0.02, material_dt=0.0001, dx=0.004, feedback=True, v7_2_optimizations=True,
                  material_diagnostics="device_reduction_v1", max_cuts=2, surface_profile="current_volume_v1",
                  configured_task=json.dumps(payload))
    recipients = [None, PATHS[1], PATHS[2]]
    (native / "setup.json").write_text(json.dumps(dict(
        config=config, physical_tool=dict(tool_from_edge_xyz_xyzw=tool_from_edge.tolist()),
        physical_knife_body_index=3,
        home_reset=dict(measured_after_pregrasp=dict(tool_edge_pose_xyzw=edges[0].tolist())))))
    (native / "finger_geometry_00.json").write_text(json.dumps(dict(pads=pads, boxes_edge=reset_boxes.tolist())))
    (attempt / "compiled_task.json").write_text(json.dumps(payload))
    (attempt / "material.json").write_text(json.dumps(dict(
        solver="tlmpm_v7_2", cutting_law=8, max_cuts=2, slots_per_node=8, code_encoding="packed2",
        obstacle_boxes_edge=reset_boxes.tolist(), obstacle_reaction_recipients=recipients)))
    with (attempt / "actions.jsonl").open("w") as stream:
        for t in range(steps):
            active = bool(cut[t % len(cut)])
            resolved = 0 if active else -1
            if corrupt_cut and t == 1:
                resolved = 1
            stream.write(json.dumps(dict(
                step=t + 1, submitted=dict(cut_enabled=active, cut_index=0),
                privileged_diagnostics=dict(timestamp_s=0.02 * (t + 1), measured_tool_pose=edges[t + 1].tolist(),
                                            active_cut=active, cut_index=resolved))) + "\n")
    np.savez_compressed(attempt / "robot_state.npz", time_s=0.02 * np.arange(steps + 1), body_pose=body,
                        body_prim_paths=np.asarray(PATHS))
    return attempt, body, edges, pads


def test_extracts_recorded_edges_cut_flags_and_options(tmp_path):
    attempt, body, edges, _ = make_attempt(tmp_path)
    traj = extract_trajectory(attempt)
    assert traj.steps == 4
    assert traj.edge_pose_world_xyzw.shape == (4, 7) and traj.pad_boxes_edge.shape == (4, 3, 2, 3)
    assert traj.pad_boxes_edge.dtype == np.float32
    np.testing.assert_array_equal(traj.edge_pose_world_xyzw, edges[1:])
    np.testing.assert_array_equal(traj.reset_edge_pose_world_xyzw, edges[0])
    np.testing.assert_allclose(traj.time_s, [0.02, 0.04, 0.06, 0.08])
    assert traj.cutting.tolist() == [False, True, True, False]
    assert traj.resolved_cut.tolist() == [-1, 0, 0, -1]
    assert traj.pad_recipients == (None, PATHS[1], PATHS[2])
    o = traj.options
    assert (o["solver"], o["cutting_law"], o["max_cuts"], o["slots_per_node"]) == ("tlmpm_v7_2", 8, 2, 8)
    assert (o["dx_m"], o["material_dt_s"], o["substeps_per_control_step"], o["seed"]) == (0.004, 0.0001, 200, 42)
    assert o["v7_2_optimizations"] is True and o["material_diagnostics"] == "device_reduction_v1"
    assert traj.checks["reset_pad_boxes_rebuilt_max_abs_diff_m"] == 0.0
    assert traj.checks["knife_body_to_edge_max_position_diff_m"] == 0.0
    assert traj.checks["cutting_steps"] == 2 and traj.checks["first_cutting_step"] == 2
    assert "RECONSTRUCTED" in traj.provenance["pad_boxes"] and "RECORDED" in traj.provenance["edge_pose"]


def test_pad_boxes_match_independent_axis_aligned_geometry(tmp_path):
    attempt, body, edges, _ = make_attempt(tmp_path)   # identity rotations everywhere
    traj = extract_trajectory(attempt)
    center = np.asarray(CORNERS).mean(0)
    for t in range(traj.steps):
        np.testing.assert_allclose(traj.pad_boxes_edge[t, 0], np.asarray(HANDLE, np.float32))
        for slot, body_index in ((1, 1), (2, 2)):
            expected = body[t + 1, body_index, :3] + center - edges[t + 1, :3]
            np.testing.assert_allclose(traj.pad_boxes_edge[t, slot, 0], expected, atol=1e-7)
            np.testing.assert_allclose(traj.pad_boxes_edge[t, slot, 1], HALF, atol=1e-7)


def test_rotated_pads_use_the_native_code_path(tmp_path):
    attempt, body, edges, pads = make_attempt(tmp_path, rotate=True,
                                              tool_from_edge=(0.028, 0., 0.09, 0.7071067811865475, 0., 0.,
                                                              0.7071067811865477))
    traj = extract_trajectory(attempt)
    for t in range(traj.steps):
        expected = pad_boxes_edge(pads, {PATHS[1]: body[t + 1, 1], PATHS[2]: body[t + 1, 2]}, edges[t + 1])
        np.testing.assert_array_equal(traj.pad_boxes_edge[t, 1:], expected)


def test_save_load_round_trip(tmp_path):
    attempt, *_ = make_attempt(tmp_path)
    traj = extract_trajectory(attempt)
    stem = traj.save(tmp_path / "trajectory")
    loaded = MeasuredTrajectory.load(stem)
    for name in MeasuredTrajectory.ARRAYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(traj, name))
    assert loaded.options == traj.options and loaded.pad_recipients == traj.pad_recipients


def test_inconsistent_cut_record_is_rejected(tmp_path):
    attempt, *_ = make_attempt(tmp_path, corrupt_cut=True)
    with pytest.raises(ValueError, match="cut"):
        extract_trajectory(attempt)


def test_digit_counts_and_sample_comparison():
    codes = np.array([0, 1, 2, 1 | (2 << 2), 2 << 2], np.int32)
    counts = digit_counts(codes, 2)
    assert counts["0"] == dict(labelled=3, negative=2, positive=1)
    assert counts["1"] == dict(labelled=2, negative=0, positive=2)
    positions = np.zeros((5, 3))
    moved = positions.copy()
    moved[2, 0] = 3e-3
    out = compare_sample(dict(positions=positions, velocities=positions, codes=codes),
                         dict(positions=moved, velocities=positions, codes=np.where(codes == 2, 0, codes)), 2)
    assert out["max_position_diff_m"] == pytest.approx(3e-3)
    assert out["different_code_particles"] == 1
    assert out["labelled_count_diff"] == {"0": -1, "1": 0}


@pytest.mark.parametrize("seed", [512, 15])
@pytest.mark.skipif(not WP17.exists(), reason="NOT_RUN: WP-17 Native recordings are absent (artifacts/wp17)")
def test_real_wp17_attempt_shapes(seed):
    attempt = WP17 / f"seed_{seed}" / "box" / "env" / "recording" / "attempt_000"
    if not (attempt / "actions.jsonl").exists():
        pytest.skip(f"NOT_RUN: WP-17 seed {seed} attempt directory is absent")
    traj = extract_trajectory(attempt)
    assert traj.steps == 1200
    assert traj.edge_pose_world_xyzw.shape == (1200, 7) and traj.pad_boxes_edge.shape == (1200, 3, 2, 3)
    assert traj.options["solver"] == "tlmpm_v7_2" and traj.options["cutting_law"] == 8
    assert traj.checks["reset_pad_boxes_rebuilt_max_abs_diff_m"] == 0.0
    assert traj.checks["knife_body_to_edge_max_position_diff_m"] < 1e-12
    assert traj.checks["first_cutting_step"] == 255
